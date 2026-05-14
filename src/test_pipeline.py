import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from config import TRAIN_PATH, MODEL_NAME, DEVICE, BATCH_SIZE
from dataset import LegalDataset
from modeling import TeacherModel, StudentModel


def main():
    print("Testing full data-model pipeline")
    print(f"Device: {DEVICE}")

    train_df = pd.read_csv(TRAIN_PATH)

    print(f"\nTrain rows: {len(train_df):,}")
    print("Columns:")
    print(train_df.columns.tolist())

    required_columns = [
        "proof_sentence_masked",
        "judgement_masked",
        "label_encoded"
    ]

    for col in required_columns:
        if col not in train_df.columns:
            raise ValueError(f"Missing required column: {col}")

    print("\nRequired columns found.")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print("Tokenizer loaded.")

    sample_df = train_df.sample(
        n=min(8, len(train_df)),
        random_state=42
    )

    dataset = LegalDataset(sample_df, tokenizer)

    first_item = dataset[0]

    print("\nSingle item check:")
    print("proof_input_ids shape:", first_item["proof_input_ids"].shape)
    print("proof_attention_mask shape:", first_item["proof_attention_mask"].shape)
    print("judgment_input_ids shape:", first_item["judgment_input_ids"].shape)
    print("judgment_attention_mask shape:", first_item["judgment_attention_mask"].shape)
    print("label:", first_item["label"])

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    batch = next(iter(loader))

    print("\nBatch check:")
    print("proof_input_ids shape:", batch["proof_input_ids"].shape)
    print("proof_attention_mask shape:", batch["proof_attention_mask"].shape)
    print("judgment_input_ids shape:", batch["judgment_input_ids"].shape)
    print("judgment_attention_mask shape:", batch["judgment_attention_mask"].shape)
    print("label shape:", batch["label"].shape)

    teacher = TeacherModel().to(DEVICE)
    student = StudentModel().to(DEVICE)

    teacher.eval()
    student.eval()

    with torch.no_grad():
        teacher_logits = teacher(
            batch["proof_input_ids"].to(DEVICE),
            batch["proof_attention_mask"].to(DEVICE)
        )

        student_logits = student(
            batch["judgment_input_ids"].to(DEVICE),
            batch["judgment_attention_mask"].to(DEVICE)
        )

    print("\nModel output check:")
    print("Teacher logits shape:", teacher_logits.shape)
    print("Student logits shape:", student_logits.shape)

    teacher_probs = torch.sigmoid(teacher_logits)
    student_probs = torch.sigmoid(student_logits)

    print("\nExample probabilities:")
    print("Teacher probabilities:", teacher_probs.squeeze().detach().cpu().numpy())
    print("Student probabilities:", student_probs.squeeze().detach().cpu().numpy())

    print("\nPipeline test passed successfully.")


if __name__ == "__main__":
    main()