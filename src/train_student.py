import json
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
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
    BATCH_SIZE,
    EPOCHS,
    LR,
    TEMPERATURE,
    ALPHA,
    TEACHER_MODEL_PATH,
    STUDENT_MODEL_PATH,
    STUDENT_METRICS_PATH,
    MODELS_DIR,
    RANDOM_STATE,
)

from dataset import LegalDataset
from modeling import TeacherModel, StudentModel


# For diagnostic run, keep sample sizes smaller.
# Later you can change them to None for full training.
TRAIN_SAMPLE_SIZE = 1000
VAL_SAMPLE_SIZE = 300


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


def distillation_loss(student_logits, teacher_logits, temperature):
    student_logits_t = student_logits.view(-1) / temperature

    teacher_probs_t = torch.sigmoid(
        teacher_logits.view(-1) / temperature
    ).detach()

    loss = F.binary_cross_entropy_with_logits(
        student_logits_t,
        teacher_probs_t
    )

    return loss * (temperature ** 2)


def evaluate_student(model, dataloader, loss_fn):
    model.eval()

    total_loss = 0
    all_probs = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating student"):
            # Student is evaluated on full judgment text
            input_ids = batch["judgment_input_ids"].to(DEVICE)
            attention_mask = batch["judgment_attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE).view(-1)

            logits = model(input_ids, attention_mask).view(-1)

            loss = loss_fn(logits, labels)
            total_loss += loss.item()

            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).long()

            all_probs.extend(probs.detach().cpu().numpy())
            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())

    avg_loss = total_loss / len(dataloader)

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    roc_auc = roc_auc_score(all_labels, all_probs)
    gini = 2 * roc_auc - 1

    print("\nProbability stats:")
    print("min:", min(all_probs))
    print("max:", max(all_probs))
    print("mean:", sum(all_probs) / len(all_probs))

    print("\nPredicted class distribution:")
    print(pd.Series(all_preds).value_counts())

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "gini": gini,
        "labels": all_labels,
        "preds": all_preds,
        "probs": all_probs,
    }


def main():
    print("Training Student BERT with Knowledge Distillation")
    print(f"Device: {DEVICE}")
    print(f"ALPHA: {ALPHA}")
    print(f"TEMPERATURE: {TEMPERATURE}")
    print("Teacher input column: proof_sentence_masked")
    print("Student input column: judgement_masked")

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

    train_dataset = LegalDataset(train_df, tokenizer)
    val_dataset = LegalDataset(val_df, tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
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

    optimizer = AdamW(
        student.parameters(),
        lr=LR
    )

    hard_loss_fn = nn.BCEWithLogitsLoss()

    best_val_auc = -1
    best_metrics = None

    for epoch in range(EPOCHS):
        student.train()
        total_train_loss = 0
        total_hard_loss = 0
        total_soft_loss = 0

        print(f"\nEpoch {epoch + 1}/{EPOCHS}")

        for batch in tqdm(train_loader, desc="Training student"):
            proof_input_ids = batch["proof_input_ids"].to(DEVICE)
            proof_attention_mask = batch["proof_attention_mask"].to(DEVICE)

            judgment_input_ids = batch["judgment_input_ids"].to(DEVICE)
            judgment_attention_mask = batch["judgment_attention_mask"].to(DEVICE)

            labels = batch["label"].to(DEVICE).view(-1)

            optimizer.zero_grad()

            # Teacher is frozen, so it must be inside no_grad
            with torch.no_grad():
                teacher_logits = teacher(
                    proof_input_ids,
                    proof_attention_mask
                ).view(-1)

            # Student must be outside no_grad because we train it
            student_logits = student(
                judgment_input_ids,
                judgment_attention_mask
            ).view(-1)

            hard_loss = hard_loss_fn(student_logits, labels)

            soft_loss = distillation_loss(
                student_logits,
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

        val_metrics = evaluate_student(
            student,
            val_loader,
            hard_loss_fn
        )

        print(f"Train Loss:      {avg_train_loss:.4f}")
        print(f"Train Hard Loss: {avg_hard_loss:.4f}")
        print(f"Train Soft Loss: {avg_soft_loss:.4f}")
        print(f"Val Loss:        {val_metrics['loss']:.4f}")
        print(f"Val Acc:         {val_metrics['accuracy']:.4f}")
        print(f"Val F1:          {val_metrics['f1']:.4f}")
        print(f"Val AUC:         {val_metrics['roc_auc']:.4f}")
        print(f"Val Gini:        {val_metrics['gini']:.4f}")

        if val_metrics["roc_auc"] > best_val_auc:
            best_val_auc = val_metrics["roc_auc"]
            best_metrics = val_metrics

            torch.save(student.state_dict(), STUDENT_MODEL_PATH)

            print(f"Saved best student model with AUC={best_val_auc:.4f}")

    print("\nBest Student Validation Metrics:")
    print(f"Accuracy:  {best_metrics['accuracy']:.4f}")
    print(f"Precision: {best_metrics['precision']:.4f}")
    print(f"Recall:    {best_metrics['recall']:.4f}")
    print(f"F1-score:  {best_metrics['f1']:.4f}")
    print(f"ROC-AUC:   {best_metrics['roc_auc']:.4f}")
    print(f"Gini:      {best_metrics['gini']:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(best_metrics["labels"], best_metrics["preds"]))

    print("\nClassification Report:")
    print(classification_report(
        best_metrics["labels"],
        best_metrics["preds"],
        zero_division=0
    ))

    metrics_to_save = {
        "model": "Student BERT with Knowledge Distillation",
        "input_column": "judgement_masked",
        "teacher_input_column": "proof_sentence_masked",
        "note": (
            "Teacher uses proof-related legal text during training. "
            "Student uses full judgment text for deployment. "
            "Teacher is frozen and used only for knowledge distillation."
        ),
        "train_sample_size": TRAIN_SAMPLE_SIZE,
        "val_sample_size": VAL_SAMPLE_SIZE,
        "temperature": TEMPERATURE,
        "alpha": ALPHA,
        "validation": {
            "accuracy": best_metrics["accuracy"],
            "precision": best_metrics["precision"],
            "recall": best_metrics["recall"],
            "f1": best_metrics["f1"],
            "roc_auc": best_metrics["roc_auc"],
            "gini": best_metrics["gini"],
        }
    }

    with open(STUDENT_METRICS_PATH, "w") as f:
        json.dump(metrics_to_save, f, indent=4)

    print("\nStudent model saved:")
    print(STUDENT_MODEL_PATH)

    print("\nStudent metrics saved:")
    print(STUDENT_METRICS_PATH)


if __name__ == "__main__":
    main()