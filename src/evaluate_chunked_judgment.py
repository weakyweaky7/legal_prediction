import json
import numpy as np
import pandas as pd
import torch

from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

from config import (
    VAL_PATH,
    MODEL_NAME,
    DEVICE,
    MODELS_DIR,
    RANDOM_STATE,
)

from modeling import StudentModel


# Fast settings for Mac CPU
VAL_SAMPLE_SIZE = 300

# We use smaller chunks for speed.
# BERT maximum is usually 512, but 256 is faster.
CHUNK_SIZE = 256

# Non-overlapping / light-overlap stride.
# Smaller stride = more chunks = slower but potentially better.
CHUNK_STRIDE = 224

# To keep it fast, we do not evaluate all chunks.
# We select first / middle / later / last chunks.
MAX_CHUNKS_PER_DOC = 4

# Aggregation strategy:
# "max" is usually best for legal documents because one important fragment may decide the case.
AGGREGATION = "max"


def prepare_dataframe(path, sample_size=None):
    df = pd.read_csv(path)

    required_columns = [
        "judgement_masked",
        "label_encoded",
    ]

    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    if sample_size is not None and len(df) > sample_size:
        class_counts = df["label_encoded"].value_counts(normalize=True)

        sampled_parts = []

        for label_value, proportion in class_counts.items():
            class_df = df[df["label_encoded"] == label_value]

            n_samples = int(sample_size * proportion)
            n_samples = min(n_samples, len(class_df))

            sampled_class_df = class_df.sample(
                n=n_samples,
                random_state=RANDOM_STATE
            )

            sampled_parts.append(sampled_class_df)

        df = pd.concat(sampled_parts)

        if len(df) < sample_size:
            remaining = sample_size - len(df)
            full_df = pd.read_csv(path)
            remaining_df = full_df.drop(df.index, errors="ignore")

            extra_df = remaining_df.sample(
                n=min(remaining, len(remaining_df)),
                random_state=RANDOM_STATE
            )

            df = pd.concat([df, extra_df])

        df = df.sample(frac=1, random_state=RANDOM_STATE)

    return df.reset_index(drop=True)


def select_evenly(chunks, max_chunks):
    """
    Select chunks from different parts of the document:
    beginning, middle, later part, end.
    This is faster than using all chunks.
    """
    if len(chunks) <= max_chunks:
        return chunks

    indices = np.linspace(0, len(chunks) - 1, max_chunks, dtype=int)
    indices = sorted(set(indices.tolist()))

    return [chunks[i] for i in indices]


def make_chunks(text, tokenizer):
    """
    Convert long judgment text into several BERT-compatible chunks.
    """
    text = str(text)

    token_ids = tokenizer.encode(
        text,
        add_special_tokens=False,
        truncation=False
    )

    # Reserve space for [CLS] and [SEP]
    body_size = CHUNK_SIZE - 2

    chunks = []

    for start in range(0, len(token_ids), CHUNK_STRIDE):
        chunk_body = token_ids[start:start + body_size]

        if len(chunk_body) == 0:
            continue

        chunks.append(chunk_body)

    if not chunks:
        chunks = [[]]

    chunks = select_evenly(chunks, MAX_CHUNKS_PER_DOC)

    return chunks


def encode_chunk(chunk_body, tokenizer):
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


def aggregate_probs(chunk_probs):
    if AGGREGATION == "max":
        return float(np.max(chunk_probs))

    if AGGREGATION == "mean":
        return float(np.mean(chunk_probs))

    if AGGREGATION == "top2_mean":
        sorted_probs = sorted(chunk_probs, reverse=True)
        top_probs = sorted_probs[:2]
        return float(np.mean(top_probs))

    raise ValueError(f"Unknown aggregation method: {AGGREGATION}")


def predict_document(model, tokenizer, text):
    chunks = make_chunks(text, tokenizer)

    input_ids_list = []
    attention_mask_list = []

    for chunk_body in chunks:
        input_ids, attention_mask = encode_chunk(chunk_body, tokenizer)
        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)

    input_ids_tensor = torch.tensor(input_ids_list, dtype=torch.long).to(DEVICE)
    attention_mask_tensor = torch.tensor(attention_mask_list, dtype=torch.long).to(DEVICE)

    with torch.no_grad():
        logits = model(input_ids_tensor, attention_mask_tensor).view(-1)
        probs = torch.sigmoid(logits).detach().cpu().numpy()

    final_prob = aggregate_probs(probs)

    return final_prob, probs.tolist(), len(chunks)


def compute_metrics(y_true, y_probs, threshold):
    y_probs = np.array(y_probs)
    y_pred = (y_probs >= threshold).astype(int)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_probs)
    gini = 2 * auc - 1

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(auc),
        "gini": float(gini),
        "preds": y_pred.tolist(),
    }


def find_best_threshold(y_true, y_probs):
    best_result = None

    for threshold in np.arange(0.10, 0.91, 0.01):
        result = compute_metrics(y_true, y_probs, threshold)

        if best_result is None or result["f1"] > best_result["f1"]:
            best_result = result

    return best_result


def main():
    print("Chunked evaluation on judgement_masked")
    print(f"Device: {DEVICE}")
    print(f"Chunk size: {CHUNK_SIZE}")
    print(f"Chunk stride: {CHUNK_STRIDE}")
    print(f"Max chunks per document: {MAX_CHUNKS_PER_DOC}")
    print(f"Aggregation: {AGGREGATION}")

    # Prefer the good proof-trained model if it exists.
    proof_model_path = MODELS_DIR / "best_student_proof_alpha08.pt"

    if proof_model_path.exists():
        model_path = proof_model_path
        print(f"\nUsing proof-trained model: {model_path}")
    else:
        model_path = MODELS_DIR / "best_student.pt"
        print("\nWARNING: best_student_proof_alpha08.pt not found.")
        print("Using models/best_student.pt instead.")
        print("Make sure this is not the bad judgement model with F1=0.")
        print(f"Model path: {model_path}")

    df = prepare_dataframe(VAL_PATH, sample_size=VAL_SAMPLE_SIZE)

    print(f"\nValidation rows: {len(df):,}")

    print("\nValidation label distribution:")
    print(df["label_encoded"].value_counts())
    print(df["label_encoded"].value_counts(normalize=True))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = StudentModel().to(DEVICE)
    model.load_state_dict(
        torch.load(model_path, map_location=DEVICE)
    )
    model.eval()

    y_true = []
    y_probs = []
    chunks_used = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Chunked inference"):
        text = row["judgement_masked"]
        label = int(row["label_encoded"])

        final_prob, chunk_probs, n_chunks = predict_document(
            model,
            tokenizer,
            text
        )

        y_true.append(label)
        y_probs.append(final_prob)
        chunks_used.append(n_chunks)

    print("\nProbability stats:")
    print("min:", float(np.min(y_probs)))
    print("max:", float(np.max(y_probs)))
    print("mean:", float(np.mean(y_probs)))
    print("median:", float(np.median(y_probs)))

    print("\nChunk stats:")
    print("min chunks:", int(np.min(chunks_used)))
    print("max chunks:", int(np.max(chunks_used)))
    print("mean chunks:", float(np.mean(chunks_used)))

    result_05 = compute_metrics(y_true, y_probs, threshold=0.5)
    best_result = find_best_threshold(y_true, y_probs)

    print("\nMetrics at threshold 0.50:")
    print(f"Accuracy:  {result_05['accuracy']:.4f}")
    print(f"Precision: {result_05['precision']:.4f}")
    print(f"Recall:    {result_05['recall']:.4f}")
    print(f"F1-score:  {result_05['f1']:.4f}")
    print(f"ROC-AUC:   {result_05['roc_auc']:.4f}")
    print(f"Gini:      {result_05['gini']:.4f}")

    print("\nBest threshold by F1:")
    print(f"Threshold: {best_result['threshold']:.2f}")
    print(f"Accuracy:  {best_result['accuracy']:.4f}")
    print(f"Precision: {best_result['precision']:.4f}")
    print(f"Recall:    {best_result['recall']:.4f}")
    print(f"F1-score:  {best_result['f1']:.4f}")
    print(f"ROC-AUC:   {best_result['roc_auc']:.4f}")
    print(f"Gini:      {best_result['gini']:.4f}")

    print("\nConfusion Matrix at best threshold:")
    print(confusion_matrix(y_true, best_result["preds"]))

    print("\nClassification Report at best threshold:")
    print(classification_report(
        y_true,
        best_result["preds"],
        zero_division=0
    ))

    metrics_to_save = {
        "model": "Chunked judgment inference with Student BERT",
        "base_model_path": str(model_path),
        "input_column": "judgement_masked",
        "training_note": (
            "The model was trained on proof_sentence_masked, "
            "then applied chunk-wise to full judgment text."
        ),
        "chunk_size": CHUNK_SIZE,
        "chunk_stride": CHUNK_STRIDE,
        "max_chunks_per_doc": MAX_CHUNKS_PER_DOC,
        "aggregation": AGGREGATION,
        "val_sample_size": VAL_SAMPLE_SIZE,
        "threshold_0_5": {
            "accuracy": result_05["accuracy"],
            "precision": result_05["precision"],
            "recall": result_05["recall"],
            "f1": result_05["f1"],
            "roc_auc": result_05["roc_auc"],
            "gini": result_05["gini"],
        },
        "best_threshold": {
            "threshold": best_result["threshold"],
            "accuracy": best_result["accuracy"],
            "precision": best_result["precision"],
            "recall": best_result["recall"],
            "f1": best_result["f1"],
            "roc_auc": best_result["roc_auc"],
            "gini": best_result["gini"],
        },
    }
output_path = MODELS_DIR / "chunked_judgment_metrics_max.json"

    with open(output_path, "w") as f:
        json.dump(metrics_to_save, f, indent=4)

    print("\nChunked metrics saved:")
    print(output_path)


if __name__ == "__main__":
    main()