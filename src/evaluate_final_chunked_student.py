import json
import numpy as np
import pandas as pd
import torch

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
    TEST_PATH,
    MODEL_NAME,
    DEVICE,
    MODELS_DIR,
    RANDOM_STATE,
)

from modeling import StudentModel


# ============================================================
# Evaluation settings
# ============================================================

# For final report, use None to evaluate the full test set.
# If it is too slow on Mac, temporarily set TEST_SAMPLE_SIZE = 300.
TEST_SAMPLE_SIZE = None

CHUNK_SIZE = 256
CHUNK_STRIDE = 224
MAX_CHUNKS_PER_DOC = 3

CHUNKED_STUDENT_MODEL_PATH = MODELS_DIR / "best_student_chunked_judgment.pt"
CHUNKED_THRESHOLD_PATH = MODELS_DIR / "chunked_judgment_threshold.json"
TEST_METRICS_PATH = MODELS_DIR / "final_chunked_test_metrics.json"


# ============================================================
# Data preparation
# ============================================================

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
    Select chunks from different parts of the judgment:
    beginning, middle, later part, end.
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


def predict_document(model, tokenizer, text):
    """
    Full judgment -> chunks -> Student logits -> max pooling -> probability.
    """
    chunk_bodies = make_chunk_bodies(text, tokenizer)

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

        # Max pooling over chunk logits.
        document_logit = torch.max(chunk_logits)

        document_probability = torch.sigmoid(document_logit).item()

    return document_probability, len(chunk_bodies)


# ============================================================
# Metrics
# ============================================================

def compute_metrics(y_true, y_probs, threshold):
    y_probs = np.array(y_probs)
    y_pred = (y_probs >= threshold).astype(int)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_probs)
    gini = 2 * roc_auc - 1

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
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


def load_saved_threshold():
    if not CHUNKED_THRESHOLD_PATH.exists():
        print("\nWARNING: chunked_judgment_threshold.json not found.")
        print("Using default threshold = 0.50")
        return 0.50

    with open(CHUNKED_THRESHOLD_PATH, "r") as f:
        threshold_data = json.load(f)

    threshold = float(threshold_data["threshold"])

    print("\nLoaded saved threshold:")
    print(f"Threshold: {threshold}")

    return threshold


# ============================================================
# Main evaluation
# ============================================================

def main():
    print("Final Test Evaluation: Chunked Judgment Student")
    print(f"Device: {DEVICE}")
    print(f"Test path: {TEST_PATH}")
    print(f"Model path: {CHUNKED_STUDENT_MODEL_PATH}")
    print(f"Chunk size: {CHUNK_SIZE}")
    print(f"Chunk stride: {CHUNK_STRIDE}")
    print(f"Max chunks per document: {MAX_CHUNKS_PER_DOC}")

    if not CHUNKED_STUDENT_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {CHUNKED_STUDENT_MODEL_PATH}"
        )

    df = prepare_dataframe(TEST_PATH, sample_size=TEST_SAMPLE_SIZE)

    print(f"\nTest rows: {len(df):,}")

    print("\nTest label distribution:")
    print(df["label_encoded"].value_counts())
    print(df["label_encoded"].value_counts(normalize=True))

    saved_threshold = load_saved_threshold()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Suppress long-text tokenizer warning.
    tokenizer.model_max_length = int(1e9)

    model = StudentModel().to(DEVICE)
    model.load_state_dict(
        torch.load(CHUNKED_STUDENT_MODEL_PATH, map_location=DEVICE)
    )
    model.eval()

    y_true = []
    y_probs = []
    chunks_used = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Final test inference"):
        text = row["judgement_masked"]
        label = int(row["label_encoded"])

        prob, n_chunks = predict_document(model, tokenizer, text)

        y_true.append(label)
        y_probs.append(prob)
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

    saved_threshold_result = compute_metrics(
        y_true,
        y_probs,
        threshold=saved_threshold
    )

    threshold_05_result = compute_metrics(
        y_true,
        y_probs,
        threshold=0.50
    )

    best_test_threshold_result = find_best_threshold(y_true, y_probs)

    print("\nTest Metrics at saved validation threshold:")
    print(f"Threshold: {saved_threshold_result['threshold']:.2f}")
    print(f"Accuracy:  {saved_threshold_result['accuracy']:.4f}")
    print(f"Precision: {saved_threshold_result['precision']:.4f}")
    print(f"Recall:    {saved_threshold_result['recall']:.4f}")
    print(f"F1-score:  {saved_threshold_result['f1']:.4f}")
    print(f"ROC-AUC:   {saved_threshold_result['roc_auc']:.4f}")
    print(f"Gini:      {saved_threshold_result['gini']:.4f}")

    print("\nConfusion Matrix at saved validation threshold:")
    print(confusion_matrix(y_true, saved_threshold_result["preds"]))

    print("\nClassification Report at saved validation threshold:")
    print(classification_report(
        y_true,
        saved_threshold_result["preds"],
        zero_division=0
    ))

    print("\nTest Metrics at threshold 0.50:")
    print(f"Accuracy:  {threshold_05_result['accuracy']:.4f}")
    print(f"Precision: {threshold_05_result['precision']:.4f}")
    print(f"Recall:    {threshold_05_result['recall']:.4f}")
    print(f"F1-score:  {threshold_05_result['f1']:.4f}")
    print(f"ROC-AUC:   {threshold_05_result['roc_auc']:.4f}")
    print(f"Gini:      {threshold_05_result['gini']:.4f}")

    print("\nBest threshold on test set by F1:")
    print(f"Threshold: {best_test_threshold_result['threshold']:.2f}")
    print(f"Accuracy:  {best_test_threshold_result['accuracy']:.4f}")
    print(f"Precision: {best_test_threshold_result['precision']:.4f}")
    print(f"Recall:    {best_test_threshold_result['recall']:.4f}")
    print(f"F1-score:  {best_test_threshold_result['f1']:.4f}")
    print(f"ROC-AUC:   {best_test_threshold_result['roc_auc']:.4f}")
    print(f"Gini:      {best_test_threshold_result['gini']:.4f}")

    metrics_to_save = {
        "model": "Final Chunked Judgment Student BERT",
        "model_path": str(CHUNKED_STUDENT_MODEL_PATH),
        "input_column": "judgement_masked",
        "test_sample_size": TEST_SAMPLE_SIZE,
        "chunk_size": CHUNK_SIZE,
        "chunk_stride": CHUNK_STRIDE,
        "max_chunks_per_doc": MAX_CHUNKS_PER_DOC,
        "pooling": "max",
        "saved_validation_threshold": saved_threshold,
        "test_metrics_at_saved_threshold": {
            "accuracy": saved_threshold_result["accuracy"],
            "precision": saved_threshold_result["precision"],
            "recall": saved_threshold_result["recall"],
            "f1": saved_threshold_result["f1"],
            "roc_auc": saved_threshold_result["roc_auc"],
            "gini": saved_threshold_result["gini"],
        },
        "test_metrics_at_threshold_0_5": {
            "accuracy": threshold_05_result["accuracy"],
            "precision": threshold_05_result["precision"],
            "recall": threshold_05_result["recall"],
            "f1": threshold_05_result["f1"],
            "roc_auc": threshold_05_result["roc_auc"],
            "gini": threshold_05_result["gini"],
        },
        "best_threshold_on_test_for_analysis_only": {
            "threshold": best_test_threshold_result["threshold"],
            "accuracy": best_test_threshold_result["accuracy"],
            "precision": best_test_threshold_result["precision"],
            "recall": best_test_threshold_result["recall"],
            "f1": best_test_threshold_result["f1"],
            "roc_auc": best_test_threshold_result["roc_auc"],
            "gini": best_test_threshold_result["gini"],
        },
    }

    with open(TEST_METRICS_PATH, "w") as f:
        json.dump(metrics_to_save, f, indent=4)

    print("\nFinal test metrics saved:")
    print(TEST_METRICS_PATH)


if __name__ == "__main__":
    main()