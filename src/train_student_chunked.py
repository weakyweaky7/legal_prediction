import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from torch.optim import AdamW
from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

from config import (
    TRAIN_PATH,
    VAL_PATH,
    MODEL_NAME,
    DEVICE,
    TEACHER_MODEL_PATH,
    MODELS_DIR,
    RANDOM_STATE,
)

from modeling import TeacherModel, StudentModel


# ============================================================
# Fast diagnostic settings for Mac CPU
# ============================================================

TRAIN_SAMPLE_SIZE = 500
VAL_SAMPLE_SIZE = 200

LOCAL_BATCH_SIZE = 2
LOCAL_EPOCHS = 1
LOCAL_LR = 5e-6

# loss = ALPHA * hard_loss + (1 - ALPHA) * soft_loss
# ALPHA = 0.9 means:
# 90% real labels, 10% teacher soft signal
ALPHA = 0.9
TEMPERATURE = 4.0

# Chunking settings
CHUNK_SIZE = 256
CHUNK_STRIDE = 224
MAX_CHUNKS_PER_DOC = 3

# Final model paths
CHUNKED_STUDENT_MODEL_PATH = MODELS_DIR / "best_student_chunked_judgment.pt"
CHUNKED_METRICS_PATH = MODELS_DIR / "student_chunked_judgment_metrics.json"
CHUNKED_THRESHOLD_PATH = MODELS_DIR / "chunked_judgment_threshold.json"


# ============================================================
# Data preparation
# ============================================================

def prepare_dataframe(path, sample_size=None):
    df = pd.read_csv(path)

    required_columns = [
        "proof_sentence_masked",
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
    Select chunks from different document parts.
    This is faster than using all chunks.
    """
    if len(chunks) <= max_chunks:
        return chunks

    indices = np.linspace(0, len(chunks) - 1, max_chunks, dtype=int)
    indices = sorted(set(indices.tolist()))

    return [chunks[i] for i in indices]


def make_chunk_bodies(text, tokenizer):
    """
    Converts long judgment text into selected chunk token bodies.
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
    Adds [CLS], [SEP], padding and attention mask.
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


class ChunkedJudgmentDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        proof_text = str(row["proof_sentence_masked"])
        judgment_text = str(row["judgement_masked"])
        label = float(row["label_encoded"])

        # Teacher input: proof_sentence_masked
        proof = self.tokenizer(
            proof_text,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # Student input: judgement_masked chunks
        chunk_bodies = make_chunk_bodies(judgment_text, self.tokenizer)

        chunk_input_ids = []
        chunk_attention_masks = []
        chunk_valid_mask = []

        for chunk_body in chunk_bodies:
            input_ids, attention_mask = encode_chunk_body(
                chunk_body,
                self.tokenizer
            )

            chunk_input_ids.append(input_ids)
            chunk_attention_masks.append(attention_mask)
            chunk_valid_mask.append(1)

        # Pad chunk slots to MAX_CHUNKS_PER_DOC
        while len(chunk_input_ids) < MAX_CHUNKS_PER_DOC:
            input_ids, attention_mask = encode_chunk_body([], self.tokenizer)

            chunk_input_ids.append(input_ids)
            chunk_attention_masks.append(attention_mask)
            chunk_valid_mask.append(0)

        # Safety cut if something went over the limit
        chunk_input_ids = chunk_input_ids[:MAX_CHUNKS_PER_DOC]
        chunk_attention_masks = chunk_attention_masks[:MAX_CHUNKS_PER_DOC]
        chunk_valid_mask = chunk_valid_mask[:MAX_CHUNKS_PER_DOC]

        return {
            "proof_input_ids": proof["input_ids"].squeeze(0),
            "proof_attention_mask": proof["attention_mask"].squeeze(0),

            "chunk_input_ids": torch.tensor(chunk_input_ids, dtype=torch.long),
            "chunk_attention_mask": torch.tensor(chunk_attention_masks, dtype=torch.long),
            "chunk_valid_mask": torch.tensor(chunk_valid_mask, dtype=torch.float),

            "label": torch.tensor(label, dtype=torch.float),
        }


# ============================================================
# Loss and evaluation
# ============================================================

def distillation_loss(student_doc_logits, teacher_logits, temperature):
    student_logits_t = student_doc_logits.view(-1) / temperature

    teacher_probs_t = torch.sigmoid(
        teacher_logits.view(-1) / temperature
    ).detach()

    loss = F.binary_cross_entropy_with_logits(
        student_logits_t,
        teacher_probs_t
    )

    return loss * (temperature ** 2)


def max_pool_chunks(chunk_logits, chunk_valid_mask):
    """
    chunk_logits shape: [batch_size, max_chunks]
    chunk_valid_mask shape: [batch_size, max_chunks]

    Invalid padded chunks are ignored.
    """
    masked_logits = chunk_logits.masked_fill(chunk_valid_mask == 0, -1e9)
    doc_logits = masked_logits.max(dim=1).values

    return doc_logits


def forward_student_on_chunks(student, chunk_input_ids, chunk_attention_mask, chunk_valid_mask):
    """
    Runs Student model over all chunks and returns document-level logits.
    """
    batch_size, max_chunks, seq_len = chunk_input_ids.shape

    flat_input_ids = chunk_input_ids.view(batch_size * max_chunks, seq_len)
    flat_attention_mask = chunk_attention_mask.view(batch_size * max_chunks, seq_len)

    flat_logits = student(
        flat_input_ids,
        flat_attention_mask
    ).view(batch_size, max_chunks)

    doc_logits = max_pool_chunks(flat_logits, chunk_valid_mask)

    return doc_logits, flat_logits


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


def evaluate_chunked_student(student, dataloader, loss_fn):
    student.eval()

    total_loss = 0
    y_true = []
    y_probs = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating chunked student"):
            chunk_input_ids = batch["chunk_input_ids"].to(DEVICE)
            chunk_attention_mask = batch["chunk_attention_mask"].to(DEVICE)
            chunk_valid_mask = batch["chunk_valid_mask"].to(DEVICE)

            labels = batch["label"].to(DEVICE).view(-1)

            doc_logits, _ = forward_student_on_chunks(
                student,
                chunk_input_ids,
                chunk_attention_mask,
                chunk_valid_mask
            )

            loss = loss_fn(doc_logits, labels)
            total_loss += loss.item()

            probs = torch.sigmoid(doc_logits)

            y_true.extend(labels.detach().cpu().numpy())
            y_probs.extend(probs.detach().cpu().numpy())

    avg_loss = total_loss / len(dataloader)

    result_05 = compute_metrics(y_true, y_probs, threshold=0.5)
    best_result = find_best_threshold(y_true, y_probs)

    return {
        "loss": avg_loss,
        "threshold_0_5": result_05,
        "best_threshold": best_result,
        "labels": y_true,
        "probs": y_probs,
    }


# ============================================================
# Main training
# ============================================================

def main():
    print("Training Chunked Judgment Student with Knowledge Distillation")
    print(f"Device: {DEVICE}")
    print(f"Train sample size: {TRAIN_SAMPLE_SIZE}")
    print(f"Validation sample size: {VAL_SAMPLE_SIZE}")
    print(f"Batch size: {LOCAL_BATCH_SIZE}")
    print(f"Epochs: {LOCAL_EPOCHS}")
    print(f"Learning rate: {LOCAL_LR}")
    print(f"Alpha: {ALPHA}")
    print(f"Temperature: {TEMPERATURE}")
    print(f"Chunk size: {CHUNK_SIZE}")
    print(f"Chunk stride: {CHUNK_STRIDE}")
    print(f"Max chunks per document: {MAX_CHUNKS_PER_DOC}")
    print("\nTeacher input: proof_sentence_masked")
    print("Student input: judgement_masked chunks")
    print("Document pooling: max over chunk logits")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    train_df = prepare_dataframe(TRAIN_PATH, sample_size=TRAIN_SAMPLE_SIZE)
    val_df = prepare_dataframe(VAL_PATH, sample_size=VAL_SAMPLE_SIZE)

    print(f"\nTrain rows: {len(train_df):,}")
    print(f"Validation rows: {len(val_df):,}")

    print("\nTrain label distribution:")
    print(train_df["label_encoded"].value_counts())
    print(train_df["label_encoded"].value_counts(normalize=True))

    print("\nValidation label distribution:")
    print(val_df["label_encoded"].value_counts())
    print(val_df["label_encoded"].value_counts(normalize=True))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Suppress long-text tokenizer warning.
    tokenizer.model_max_length = int(1e9)

    train_dataset = ChunkedJudgmentDataset(train_df, tokenizer)
    val_dataset = ChunkedJudgmentDataset(val_df, tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=LOCAL_BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=LOCAL_BATCH_SIZE,
        shuffle=False
    )

    print("\nLoading Teacher model...")
    teacher = TeacherModel().to(DEVICE)
    teacher.load_state_dict(
        torch.load(TEACHER_MODEL_PATH, map_location=DEVICE)
    )
    teacher.eval()

    for param in teacher.parameters():
        param.requires_grad = False

    print("Teacher loaded and frozen.")

    student = StudentModel().to(DEVICE)

    proof_student_path = MODELS_DIR / "best_student_proof_alpha08.pt"

    if proof_student_path.exists():
        print(f"\nLoading proof-trained student weights from: {proof_student_path}")
        student.load_state_dict(
            torch.load(proof_student_path, map_location=DEVICE)
        )
        print("Proof-trained student weights loaded.")
    else:
        print("\nWARNING: proof-trained student model not found.")
        print("Training chunked student from pretrained LegalBERT only.")

    optimizer = AdamW(
        student.parameters(),
        lr=LOCAL_LR
    )

    hard_loss_fn = nn.BCEWithLogitsLoss()

    best_val_auc = -1
    best_metrics = None

    for epoch in range(LOCAL_EPOCHS):
        student.train()

        total_train_loss = 0
        total_hard_loss = 0
        total_soft_loss = 0

        print(f"\nEpoch {epoch + 1}/{LOCAL_EPOCHS}")

        for batch in tqdm(train_loader, desc="Training chunked student"):
            proof_input_ids = batch["proof_input_ids"].to(DEVICE)
            proof_attention_mask = batch["proof_attention_mask"].to(DEVICE)

            chunk_input_ids = batch["chunk_input_ids"].to(DEVICE)
            chunk_attention_mask = batch["chunk_attention_mask"].to(DEVICE)
            chunk_valid_mask = batch["chunk_valid_mask"].to(DEVICE)

            labels = batch["label"].to(DEVICE).view(-1)

            optimizer.zero_grad()

            # Teacher is frozen.
            with torch.no_grad():
                teacher_logits = teacher(
                    proof_input_ids,
                    proof_attention_mask
                ).view(-1)

            # Student is trained on judgment chunks.
            student_doc_logits, _ = forward_student_on_chunks(
                student,
                chunk_input_ids,
                chunk_attention_mask,
                chunk_valid_mask
            )

            hard_loss = hard_loss_fn(student_doc_logits, labels)

            soft_loss = distillation_loss(
                student_doc_logits,
                teacher_logits,
                TEMPERATURE
            )

            loss = ALPHA * hard_loss + (1 - ALPHA) * soft_loss

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            total_hard_loss += hard_loss.item()
            total_soft_loss += soft_loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_hard_loss = total_hard_loss / len(train_loader)
        avg_soft_loss = total_soft_loss / len(train_loader)

        val_metrics = evaluate_chunked_student(
            student,
            val_loader,
            hard_loss_fn
        )

        threshold_05 = val_metrics["threshold_0_5"]
        best_threshold = val_metrics["best_threshold"]

        print(f"Train Loss:      {avg_train_loss:.4f}")
        print(f"Train Hard Loss: {avg_hard_loss:.4f}")
        print(f"Train Soft Loss: {avg_soft_loss:.4f}")

        print("\nValidation at threshold 0.50:")
        print(f"Val Loss:   {val_metrics['loss']:.4f}")
        print(f"Val Acc:    {threshold_05['accuracy']:.4f}")
        print(f"Val F1:     {threshold_05['f1']:.4f}")
        print(f"Val AUC:    {threshold_05['roc_auc']:.4f}")
        print(f"Val Gini:   {threshold_05['gini']:.4f}")

        print("\nBest threshold by F1:")
        print(f"Threshold: {best_threshold['threshold']:.2f}")
        print(f"Accuracy:  {best_threshold['accuracy']:.4f}")
        print(f"Precision: {best_threshold['precision']:.4f}")
        print(f"Recall:    {best_threshold['recall']:.4f}")
        print(f"F1-score:  {best_threshold['f1']:.4f}")
        print(f"ROC-AUC:   {best_threshold['roc_auc']:.4f}")
        print(f"Gini:      {best_threshold['gini']:.4f}")

        if best_threshold["roc_auc"] > best_val_auc:
            best_val_auc = best_threshold["roc_auc"]
            best_metrics = val_metrics

            torch.save(student.state_dict(), CHUNKED_STUDENT_MODEL_PATH)

            print(f"\nSaved best chunked student model with AUC={best_val_auc:.4f}")

    final_best = best_metrics["best_threshold"]

    print("\nFinal Best Chunked Student Validation Metrics:")
    print(f"Best Threshold: {final_best['threshold']:.2f}")
    print(f"Accuracy:       {final_best['accuracy']:.4f}")
    print(f"Precision:      {final_best['precision']:.4f}")
    print(f"Recall:         {final_best['recall']:.4f}")
    print(f"F1-score:       {final_best['f1']:.4f}")
    print(f"ROC-AUC:        {final_best['roc_auc']:.4f}")
    print(f"Gini:           {final_best['gini']:.4f}")

    print("\nConfusion Matrix at best threshold:")
    print(confusion_matrix(
        best_metrics["labels"],
        final_best["preds"]
    ))

    print("\nClassification Report at best threshold:")
    print(classification_report(
        best_metrics["labels"],
        final_best["preds"],
        zero_division=0
    ))

    metrics_to_save = {
        "model": "Chunked Judgment Student BERT with Knowledge Distillation",
        "model_path": str(CHUNKED_STUDENT_MODEL_PATH),
        "teacher_input_column": "proof_sentence_masked",
        "student_input_column": "judgement_masked",
        "training_strategy": (
            "Teacher is trained on proof-related legal sentences. "
            "Student is initialized from proof-trained weights and then "
            "fine-tuned on chunks from full judgment text. "
            "Document-level student prediction is obtained by max pooling "
            "over chunk logits."
        ),
        "train_sample_size": TRAIN_SAMPLE_SIZE,
        "val_sample_size": VAL_SAMPLE_SIZE,
        "batch_size": LOCAL_BATCH_SIZE,
        "epochs": LOCAL_EPOCHS,
        "learning_rate": LOCAL_LR,
        "alpha": ALPHA,
        "temperature": TEMPERATURE,
        "chunk_size": CHUNK_SIZE,
        "chunk_stride": CHUNK_STRIDE,
        "max_chunks_per_doc": MAX_CHUNKS_PER_DOC,
        "pooling": "max",
        "validation": {
            "threshold_0_5": {
                "accuracy": best_metrics["threshold_0_5"]["accuracy"],
                "precision": best_metrics["threshold_0_5"]["precision"],
                "recall": best_metrics["threshold_0_5"]["recall"],
                "f1": best_metrics["threshold_0_5"]["f1"],
                "roc_auc": best_metrics["threshold_0_5"]["roc_auc"],
                "gini": best_metrics["threshold_0_5"]["gini"],
            },
            "best_threshold": {
                "threshold": final_best["threshold"],
                "accuracy": final_best["accuracy"],
                "precision": final_best["precision"],
                "recall": final_best["recall"],
                "f1": final_best["f1"],
                "roc_auc": final_best["roc_auc"],
                "gini": final_best["gini"],
            },
        },
    }

    with open(CHUNKED_METRICS_PATH, "w") as f:
        json.dump(metrics_to_save, f, indent=4)

    threshold_to_save = {
        "mode": "chunked_judgment",
        "model_path": str(CHUNKED_STUDENT_MODEL_PATH),
        "threshold": final_best["threshold"],
        "aggregation": "max",
        "f1": final_best["f1"],
        "roc_auc": final_best["roc_auc"],
    }

    with open(CHUNKED_THRESHOLD_PATH, "w") as f:
        json.dump(threshold_to_save, f, indent=4)

    print("\nChunked student model saved:")
    print(CHUNKED_STUDENT_MODEL_PATH)

    print("\nChunked student metrics saved:")
    print(CHUNKED_METRICS_PATH)

    print("\nChunked threshold saved:")
    print(CHUNKED_THRESHOLD_PATH)


if __name__ == "__main__":
    main()