import torch
from torch.utils.data import Dataset

from config import MAX_PROOF_LEN, MAX_JUDGMENT_LEN


class LegalDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        proof_text = str(row["proof_sentence_masked"])
        judgment_text = str(row["judgement_masked"])
        label = row["label_encoded"]

        proof = self.tokenizer(
            proof_text,
            max_length=MAX_PROOF_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        judgment = self.tokenizer(
            judgment_text,
            max_length=MAX_JUDGMENT_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "proof_input_ids": proof["input_ids"].squeeze(0),
            "proof_attention_mask": proof["attention_mask"].squeeze(0),

            "judgment_input_ids": judgment["input_ids"].squeeze(0),
            "judgment_attention_mask": judgment["attention_mask"].squeeze(0),

            "label": torch.tensor(label, dtype=torch.float)
        }