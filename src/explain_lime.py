import argparse
import json
import numpy as np
import pandas as pd
import torch

from transformers import AutoTokenizer

try:
    from lime.lime_text import LimeTextExplainer
except ImportError as error:
    raise ImportError(
        "LIME is not installed. Install it with: python3 -m pip install lime"
    ) from error

from config import (
    TEST_PATH,
    VAL_PATH,
    MODEL_NAME,
    DEVICE,
    MODELS_DIR,
)

from modeling import StudentModel


# ============================================================
# Final model settings
# ============================================================

FINAL_DIR = MODELS_DIR / "final"

FINAL_MODEL_PATH = FINAL_DIR / "final_student_chunked_judgment.pt"
FALLBACK_MODEL_PATH = MODELS_DIR / "best_student_chunked_judgment.pt"

FINAL_THRESHOLD_PATH = FINAL_DIR / "final_chunked_threshold.json"
FALLBACK_THRESHOLD_PATH = MODELS_DIR / "chunked_judgment_threshold.json"

EXPLANATION_DIR = MODELS_DIR / "lime_explanations"

CHUNK_SIZE = 256
CHUNK_STRIDE = 224
MAX_CHUNKS_PER_DOC = 3

LABELS = {
    0: "Rejected",
    1: "Accepted",
}

CLASS_NAMES = ["Rejected", "Accepted"]


# ============================================================
# Loading helpers
# ============================================================

def get_model_path():
    if FINAL_MODEL_PATH.exists():
        return FINAL_MODEL_PATH

    if FALLBACK_MODEL_PATH.exists():
        print("WARNING: final model copy not found. Using fallback model.")
        return FALLBACK_MODEL_PATH

    raise FileNotFoundError(
        f"Model not found. Checked:\n{FINAL_MODEL_PATH}\n{FALLBACK_MODEL_PATH}"
    )


def get_threshold_path():
    if FINAL_THRESHOLD_PATH.exists():
        return FINAL_THRESHOLD_PATH

    if FALLBACK_THRESHOLD_PATH.exists():
        print("WARNING: final threshold copy not found. Using fallback threshold.")
        return FALLBACK_THRESHOLD_PATH

    raise FileNotFoundError(
        f"Threshold file not found. Checked:\n{FINAL_THRESHOLD_PATH}\n{FALLBACK_THRESHOLD_PATH}"
    )


def load_threshold():
    threshold_path = get_threshold_path()

    with open(threshold_path, "r") as f:
        data = json.load(f)

    threshold = float(data["threshold"])

    return threshold, threshold_path


def load_model():
    model_path = get_model_path()

    model = StudentModel().to(DEVICE)
    model.load_state_dict(
        torch.load(model_path, map_location=DEVICE)
    )
    model.eval()

    return model, model_path


# ============================================================
# Chunking helpers
# ============================================================

def select_evenly(chunks, max_chunks):
    """
    Select chunks from different parts of the document.
    This matches the final model setup.
    """
    if len(chunks) <= max_chunks:
        return chunks

    indices = np.linspace(0, len(chunks) - 1, max_chunks, dtype=int)
    indices = sorted(set(indices.tolist()))

    return [chunks[i] for i in indices]


def make_chunk_bodies(text, tokenizer):
    """
    Convert long judgment text into selected chunk token bodies.
    Each body does not include [CLS] and [SEP].
    """
    text = str(text)

    token_ids = tokenizer.encode(
        text,
        add_special_tokens=False,
        truncation=False
    )

    body_size = CHUNK_SIZE - 2
    chunks = []

    for start in range(0, len(token_ids), CHUNK_STRIDE):
        chunk_body = token_ids[start:start + body_size]

        if len(chunk_body) > 0:
            chunks.append(chunk_body)

    if not chunks:
        chunks = [[]]

    chunks = select_evenly(chunks, MAX_CHUNKS_PER_DOC)

    return chunks


def encode_chunk_body(chunk_body, tokenizer):
    """
    Add [CLS], [SEP], padding and attention mask.
    """
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    pad_id = tokenizer.pad_token_id

    input_ids = [cls_id] + chunk_body[:CHUNK_SIZE - 2] + [sep_id]
    attention_mask = [1] * len(input_ids)

    padding_len = CHUNK_SIZE - len(input_ids)

    input_ids = input_ids + [pad_id] * padding_len
    attention_mask = attention_mask + [0] * padding_len

    return input_ids, attention_mask


def decode_chunk_body(chunk_body, tokenizer):
    return tokenizer.decode(
        chunk_body,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )


# ============================================================
# Prediction helpers
# ============================================================

def predict_chunks(model, tokenizer, chunk_bodies):
    input_ids_list = []
    attention_mask_list = []

    for chunk_body in chunk_bodies:
        input_ids, attention_mask = encode_chunk_body(chunk_body, tokenizer)
        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)

    input_ids_tensor = torch.tensor(input_ids_list, dtype=torch.long).to(DEVICE)
    attention_mask_tensor = torch.tensor(attention_mask_list, dtype=torch.long).to(DEVICE)

    with torch.no_grad():
        chunk_logits = model(input_ids_tensor, attention_mask_tensor).view(-1)
        chunk_probs = torch.sigmoid(chunk_logits).detach().cpu().numpy()

    return chunk_probs


def predict_full_judgment(model, tokenizer, text, threshold):
    chunk_bodies = make_chunk_bodies(text, tokenizer)
    chunk_texts = [decode_chunk_body(chunk, tokenizer) for chunk in chunk_bodies]

    chunk_probs = predict_chunks(model, tokenizer, chunk_bodies)

    best_chunk_index = int(np.argmax(chunk_probs))
    final_probability = float(np.max(chunk_probs))

    predicted_label = 1 if final_probability >= threshold else 0
    prediction_name = LABELS[predicted_label]

    if predicted_label == 1:
        confidence = final_probability
    else:
        confidence = 1.0 - final_probability

    return {
        "prediction": prediction_name,
        "predicted_label": predicted_label,
        "probability_accepted": final_probability,
        "confidence": confidence,
        "threshold": threshold,
        "chunks_analyzed": len(chunk_bodies),
        "chunk_probabilities": [float(p) for p in chunk_probs],
        "most_influential_chunk_index": best_chunk_index,
        "most_influential_chunk_probability": float(chunk_probs[best_chunk_index]),
        "most_influential_chunk_text": chunk_texts[best_chunk_index],
    }


def build_lime_classifier_fn(model, tokenizer, batch_size=16):
    """
    LIME calls this function many times with perturbed versions of the selected chunk.
    It must return probabilities for both classes:
    [P(Rejected), P(Accepted)]
    """
    def classifier_fn(texts):
        all_outputs = []

        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start:start + batch_size]

            input_ids_list = []
            attention_mask_list = []

            for text in batch_texts:
                encoded = tokenizer(
                    str(text),
                    max_length=CHUNK_SIZE,
                    padding="max_length",
                    truncation=True,
                    return_tensors=None
                )

                input_ids_list.append(encoded["input_ids"])
                attention_mask_list.append(encoded["attention_mask"])

            input_ids_tensor = torch.tensor(input_ids_list, dtype=torch.long).to(DEVICE)
            attention_mask_tensor = torch.tensor(attention_mask_list, dtype=torch.long).to(DEVICE)

            with torch.no_grad():
                logits = model(input_ids_tensor, attention_mask_tensor).view(-1)
                probs_accepted = torch.sigmoid(logits).detach().cpu().numpy()

            probs_rejected = 1.0 - probs_accepted

            batch_outputs = np.vstack([
                probs_rejected,
                probs_accepted
            ]).T

            all_outputs.append(batch_outputs)

        return np.vstack(all_outputs)

    return classifier_fn


# ============================================================
# LIME explanation
# ============================================================

def explain_with_lime(
    model,
    tokenizer,
    chunk_text,
    predicted_label,
    num_features=12,
    num_samples=500
):
    explainer = LimeTextExplainer(
        class_names=CLASS_NAMES
    )

    classifier_fn = build_lime_classifier_fn(
        model=model,
        tokenizer=tokenizer,
        batch_size=16
    )

    explanation = explainer.explain_instance(
        text_instance=chunk_text,
        classifier_fn=classifier_fn,
        labels=[predicted_label],
        num_features=num_features,
        num_samples=num_samples
    )

    lime_weights = explanation.as_list(label=predicted_label)

    supportive_words = []
    opposing_words = []

    for word, weight in lime_weights:
        item = {
            "word": word,
            "weight": float(weight)
        }

        if weight >= 0:
            supportive_words.append(item)
        else:
            opposing_words.append(item)

    return {
        "explained_label": predicted_label,
        "explained_class": LABELS[predicted_label],
        "num_features": num_features,
        "num_samples": num_samples,
        "lime_weights": [
            {"word": word, "weight": float(weight)}
            for word, weight in lime_weights
        ],
        "supportive_words": supportive_words,
        "opposing_words": opposing_words,
    }


# ============================================================
# Input helpers
# ============================================================

def load_text_from_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_text_from_dataset(split, index):
    if split == "test":
        path = TEST_PATH
    elif split == "val":
        path = VAL_PATH
    else:
        raise ValueError("split must be 'test' or 'val'")

    df = pd.read_csv(path)

    if index < 0 or index >= len(df):
        raise IndexError(
            f"Index {index} is out of range for {split} set with {len(df)} rows."
        )

    row = df.iloc[index]

    text = str(row["judgement_masked"])
    label = int(row["label_encoded"])

    return text, label


def save_explanation(result):
    EXPLANATION_DIR.mkdir(parents=True, exist_ok=True)

    output_path = EXPLANATION_DIR / "latest_lime_explanation.json"

    with open(output_path, "w") as f:
        json.dump(result, f, indent=4)

    return output_path


def print_explanation(prediction_result, lime_result, true_label=None):
    print("\n================ LIME EXPLANATION ================")
    print(f"Prediction: {prediction_result['prediction']}")
    print(f"Predicted label: {prediction_result['predicted_label']}")
    print(f"Probability Accepted: {prediction_result['probability_accepted']:.4f}")
    print(f"Confidence: {prediction_result['confidence']:.4f}")
    print(f"Threshold: {prediction_result['threshold']:.2f}")
    print(f"Chunks analyzed: {prediction_result['chunks_analyzed']}")

    if true_label is not None:
        print("\nGround truth:")
        print(f"True label: {true_label}")
        print(f"True outcome: {LABELS[true_label]}")

    print("\nChunk probabilities:")
    for i, prob in enumerate(prediction_result["chunk_probabilities"]):
        print(f"Chunk {i}: P(Accepted) = {prob:.4f}")

    print("\nMost influential chunk:")
    print(f"Chunk index: {prediction_result['most_influential_chunk_index']}")
    print(f"Chunk probability: {prediction_result['most_influential_chunk_probability']:.4f}")
    print("Text preview:")
    print(prediction_result["most_influential_chunk_text"][:1200])

    print("\nLIME explanation:")
    print(f"Explained class: {lime_result['explained_class']}")
    print(f"Num samples: {lime_result['num_samples']}")

    print("\nWords supporting the predicted class:")
    if lime_result["supportive_words"]:
        for item in lime_result["supportive_words"]:
            print(f"- {item['word']}: {item['weight']:.4f}")
    else:
        print("- None")

    print("\nWords against the predicted class:")
    if lime_result["opposing_words"]:
        for item in lime_result["opposing_words"]:
            print(f"- {item['word']}: {item['weight']:.4f}")
    else:
        print("- None")

    print("==================================================")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="LIME explanation for final chunked legal judgment model."
    )

    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Raw judgment text."
    )

    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Path to a .txt file containing judgment text."
    )

    parser.add_argument(
        "--sample_index",
        type=int,
        default=None,
        help="Index of sample from test/val split."
    )

    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "val"],
        help="Dataset split for sample_index."
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=500,
        help="Number of LIME perturbation samples. Use 300 for speed, 1000 for stronger explanation."
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=12,
        help="Number of important words/features to show."
    )

    args = parser.parse_args()

    threshold, threshold_path = load_threshold()

    print("Loading final model...")
    model, model_path = load_model()

    print(f"Device: {DEVICE}")
    print(f"Model path: {model_path}")
    print(f"Threshold path: {threshold_path}")
    print(f"Threshold: {threshold:.2f}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Suppress long-text tokenizer warning.
    tokenizer.model_max_length = int(1e9)

    true_label = None

    if args.text is not None:
        text = args.text

    elif args.file is not None:
        text = load_text_from_file(args.file)

    elif args.sample_index is not None:
        text, true_label = load_text_from_dataset(args.split, args.sample_index)

    else:
        raise ValueError(
            "Provide one of: --text, --file, or --sample_index."
        )

    prediction_result = predict_full_judgment(
        model=model,
        tokenizer=tokenizer,
        text=text,
        threshold=threshold
    )

    selected_chunk_text = prediction_result["most_influential_chunk_text"]
    predicted_label = prediction_result["predicted_label"]

    print("\nRunning LIME. This may take a few minutes on CPU...")

    lime_result = explain_with_lime(
        model=model,
        tokenizer=tokenizer,
        chunk_text=selected_chunk_text,
        predicted_label=predicted_label,
        num_features=args.top_k,
        num_samples=args.num_samples
    )

    final_result = {
        "prediction_result": prediction_result,
        "lime_result": lime_result,
        "true_label": true_label,
        "model_path": str(model_path),
        "threshold_path": str(threshold_path),
    }

    output_path = save_explanation(final_result)

    print_explanation(
        prediction_result=prediction_result,
        lime_result=lime_result,
        true_label=true_label
    )

    print("\nLIME explanation saved:")
    print(output_path)


if __name__ == "__main__":
    main()