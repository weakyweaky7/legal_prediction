import argparse
import json
import numpy as np
import pandas as pd
import torch

from transformers import AutoTokenizer

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

CHUNK_SIZE = 256
CHUNK_STRIDE = 224
MAX_CHUNKS_PER_DOC = 3

# Safety thresholds for legal decision-support.
# If confidence is below this value, the system returns "Uncertain".
ABSTAIN_CONFIDENCE_THRESHOLD = 0.70

# If the probability gap between Accepted and Rejected is too small,
# the system also returns "Uncertain".
ABSTAIN_MARGIN_THRESHOLD = 0.30

LABELS = {
    0: "Rejected",
    1: "Accepted",
}


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
# Uncertainty / Abstain Mode
# ============================================================

def apply_abstain_mode(
    probability_accepted,
    decision_threshold,
    confidence_threshold=ABSTAIN_CONFIDENCE_THRESHOLD,
    margin_threshold=ABSTAIN_MARGIN_THRESHOLD,
):
    """
    Safety layer for legal AI predictions.

    The model still calculates probabilities for Accepted and Rejected.
    However, the system does not always force a hard legal prediction.

    If confidence is below the required threshold or the probability margin
    between classes is too small, the system returns "Uncertain" and recommends
    human legal review.

    Parameters:
        probability_accepted: float
            Model probability for the Accepted class.

        decision_threshold: float
            The model's selected classification threshold.

        confidence_threshold: float
            Minimum confidence required for a hard Accepted/Rejected prediction.

        margin_threshold: float
            Minimum absolute difference between P(Accepted) and P(Rejected).

    Returns:
        dict with final prediction, raw model tendency, confidence, risk level,
        reason, and recommendation.
    """

    probability_accepted = float(probability_accepted)
    probability_rejected = 1.0 - probability_accepted

    raw_predicted_label = 1 if probability_accepted >= decision_threshold else 0
    raw_prediction = LABELS[raw_predicted_label]

    if raw_predicted_label == 1:
        confidence = probability_accepted
    else:
        confidence = probability_rejected

    probability_margin = abs(probability_accepted - probability_rejected)

    should_abstain = (
        confidence < confidence_threshold
        or probability_margin < margin_threshold
    )

    if should_abstain:
        final_prediction = "Uncertain"
        final_predicted_label = None
        risk_level = "High"
        reason = (
            "The model confidence is below the legal decision threshold "
            "or the probability margin between classes is too small."
        )
        recommendation = (
            "Human legal review is required before making any conclusion."
        )
    else:
        final_prediction = raw_prediction
        final_predicted_label = raw_predicted_label
        risk_level = "Low" if confidence >= 0.85 else "Medium"
        reason = (
            "The model confidence and probability margin are above the required thresholds."
        )
        recommendation = (
            "This result can be used as decision-support, but the final legal decision "
            "must remain human-driven."
        )

    return {
        # Main output for UI/backend
        "prediction": final_prediction,
        "final_prediction": final_prediction,

        # Raw model tendency before abstain logic
        "raw_prediction": raw_prediction,
        "raw_predicted_label": raw_predicted_label,

        # Final label: None when prediction is Uncertain
        "final_predicted_label": final_predicted_label,

        # Backward-compatible field.
        # This stays as raw model label so older code will not crash.
        "predicted_label": raw_predicted_label,

        # Probabilities
        "probability_accepted": round(probability_accepted, 4),
        "probability_rejected": round(probability_rejected, 4),

        # Alternative aliases for easier frontend use
        "prob_accepted": round(probability_accepted, 4),
        "prob_rejected": round(probability_rejected, 4),

        # Confidence and safety information
        "confidence": round(confidence, 4),
        "probability_margin": round(probability_margin, 4),
        "margin": round(probability_margin, 4),

        # Thresholds
        "threshold": round(float(decision_threshold), 4),
        "decision_threshold": round(float(decision_threshold), 4),
        "abstain_confidence_threshold": round(float(confidence_threshold), 4),
        "abstain_margin_threshold": round(float(margin_threshold), 4),

        # Safety decision
        "is_abstained": bool(should_abstain),
        "risk_level": risk_level,
        "reason": reason,
        "recommendation": recommendation,
    }


# ============================================================
# Chunking
# ============================================================

def select_evenly(chunks, max_chunks):
    """
    Select chunks from different parts of the document.
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


def predict_judgment(
    model,
    tokenizer,
    text,
    threshold,
    confidence_threshold=ABSTAIN_CONFIDENCE_THRESHOLD,
    margin_threshold=ABSTAIN_MARGIN_THRESHOLD,
):
    """
    Full judgment -> chunks -> Student -> max pooling -> prediction.

    Important:
    The model first produces raw probabilities.
    Then the abstain mode decides whether the final output should be
    Accepted, Rejected, or Uncertain.
    """
    chunk_bodies = make_chunk_bodies(text, tokenizer)

    input_ids_list = []
    attention_mask_list = []
    chunk_texts = []

    for chunk_body in chunk_bodies:
        input_ids, attention_mask = encode_chunk_body(chunk_body, tokenizer)

        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)

        chunk_text = tokenizer.decode(
            chunk_body,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        chunk_texts.append(chunk_text)

    input_ids_tensor = torch.tensor(input_ids_list, dtype=torch.long).to(DEVICE)
    attention_mask_tensor = torch.tensor(attention_mask_list, dtype=torch.long).to(DEVICE)

    with torch.no_grad():
        chunk_logits = model(input_ids_tensor, attention_mask_tensor).view(-1)
        chunk_probs = torch.sigmoid(chunk_logits).detach().cpu().numpy()

    best_chunk_index = int(np.argmax(chunk_probs))
    final_probability = float(np.max(chunk_probs))

    decision = apply_abstain_mode(
        probability_accepted=final_probability,
        decision_threshold=threshold,
        confidence_threshold=confidence_threshold,
        margin_threshold=margin_threshold,
    )

    result = {
        **decision,
        "chunks_analyzed": len(chunk_bodies),
        "chunk_probabilities": [float(p) for p in chunk_probs],
        "most_influential_chunk_index": best_chunk_index,
        "most_influential_chunk_probability": float(chunk_probs[best_chunk_index]),
        "most_influential_chunk_text": chunk_texts[best_chunk_index],
    }

    return result


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
        raise IndexError(f"Index {index} is out of range for {split} set with {len(df)} rows.")

    row = df.iloc[index]

    text = str(row["judgement_masked"])
    label = int(row["label_encoded"])

    return text, label


def print_prediction(result, true_label=None):
    print("\n================ FINAL PREDICTION ================")
    print(f"Final prediction: {result['prediction']}")

    if result["is_abstained"]:
        print(f"Raw model tendency: {result['raw_prediction']}")
        print(f"Raw predicted label: {result['raw_predicted_label']}")
    else:
        print(f"Predicted label: {result['final_predicted_label']}")

    print(f"Probability Accepted: {result['probability_accepted']:.4f}")
    print(f"Probability Rejected: {result['probability_rejected']:.4f}")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"Probability margin: {result['probability_margin']:.4f}")

    print(f"Model decision threshold: {result['decision_threshold']:.4f}")
    print(f"Abstain confidence threshold: {result['abstain_confidence_threshold']:.4f}")
    print(f"Abstain margin threshold: {result['abstain_margin_threshold']:.4f}")

    print(f"Risk level: {result['risk_level']}")
    print(f"Reason: {result['reason']}")
    print(f"Recommendation: {result['recommendation']}")

    print(f"Chunks analyzed: {result['chunks_analyzed']}")

    if true_label is not None:
        print("\nGround truth:")
        print(f"True label: {true_label}")
        print(f"True outcome: {LABELS[true_label]}")

        if result["is_abstained"]:
            print("Final correctness: not evaluated because the system abstained.")
            print(
                f"Raw model correctness: "
                f"{'correct' if result['raw_predicted_label'] == true_label else 'incorrect'}"
            )
        else:
            print(
                f"Final correctness: "
                f"{'correct' if result['final_predicted_label'] == true_label else 'incorrect'}"
            )

    print("\nChunk probabilities:")
    for i, prob in enumerate(result["chunk_probabilities"]):
        print(f"Chunk {i}: P(Accepted) = {prob:.4f}")

    print("\nMost influential chunk:")
    print(f"Chunk index: {result['most_influential_chunk_index']}")
    print(f"Chunk probability: {result['most_influential_chunk_probability']:.4f}")
    print("Text preview:")
    print(result["most_influential_chunk_text"][:1200])

    print("==================================================")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Final prediction for chunked legal judgment model."
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
        "--confidence_threshold",
        type=float,
        default=ABSTAIN_CONFIDENCE_THRESHOLD,
        help="Minimum confidence required for hard prediction."
    )

    parser.add_argument(
        "--margin_threshold",
        type=float,
        default=ABSTAIN_MARGIN_THRESHOLD,
        help="Minimum probability margin required for hard prediction."
    )

    args = parser.parse_args()

    threshold, threshold_path = load_threshold()

    print("Loading final model...")
    model, model_path = load_model()

    print(f"Device: {DEVICE}")
    print(f"Model path: {model_path}")
    print(f"Threshold path: {threshold_path}")
    print(f"Model decision threshold: {threshold:.4f}")
    print(f"Abstain confidence threshold: {args.confidence_threshold:.4f}")
    print(f"Abstain margin threshold: {args.margin_threshold:.4f}")

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

    result = predict_judgment(
        model=model,
        tokenizer=tokenizer,
        text=text,
        threshold=threshold,
        confidence_threshold=args.confidence_threshold,
        margin_threshold=args.margin_threshold,
    )

    print_prediction(result, true_label=true_label)


if __name__ == "__main__":
    main()