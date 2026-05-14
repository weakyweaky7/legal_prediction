import re
import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split

from config import (
    RAW_DATA_PATH,
    DATABASE_PATH,
    CLEAN_DATA_PATH,
    TRAIN_PATH,
    VAL_PATH,
    TEST_PATH,
    RANDOM_STATE,
)


LEAKAGE_TERMS = [
    "appeal is dismissed",
    "appeal is allowed",
    "application is dismissed",
    "application is allowed",
    "claim is dismissed",
    "claim is allowed",
    "dismiss the appeal",
    "allow the appeal",
    "dismissed",
    "allowed",
    "granted",
    "rejected",
    "accepted",
    "satisfied",
    "denied",
]

LEAKAGE_TERMS_SORTED = sorted(LEAKAGE_TERMS, key=len, reverse=True)

LEAKAGE_PATTERNS = [
    r"\b" + re.escape(term) + r"\b"
    for term in LEAKAGE_TERMS_SORTED
]

COMBINED_PATTERN = "|".join(LEAKAGE_PATTERNS)


def create_folders():
    CLEAN_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)


def mask_outcome_words(text):
    text = str(text)

    return re.sub(
        COMBINED_PATTERN,
        "[OUTCOME_WORD]",
        text,
        flags=re.IGNORECASE
    )


def load_raw_data():
    df = pd.read_csv(RAW_DATA_PATH)

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    print("Raw dataset loaded")
    print(f"Rows: {df.shape[0]:,}")
    print(f"Columns: {df.shape[1]:,}")

    return df


def create_sqlite_view(df):
    conn = sqlite3.connect(DATABASE_PATH)

    df.to_sql(
        "case_files_total",
        conn,
        if_exists="replace",
        index=False
    )

    conn.execute("DROP VIEW IF EXISTS ml_training_view")

    conn.execute("""
    CREATE VIEW ml_training_view AS
    SELECT 
        name,
        case_category,
        case_type,
        case_info,
        judgement,
        proof_sentence,
        tokens,
        sentences,
        label
    FROM case_files_total
    WHERE label IN ('Accepted', 'Rejected')
      AND judgement IS NOT NULL
      AND proof_sentence IS NOT NULL
      AND label IS NOT NULL
    """)

    conn.commit()

    df_clean = pd.read_sql(
        "SELECT * FROM ml_training_view",
        conn
    )

    conn.close()

    return df_clean


def clean_dataset(df_clean):
    df_clean = df_clean.copy()

    df_clean["case_category"] = df_clean["case_category"].fillna("Unknown")
    df_clean["case_type"] = df_clean["case_type"].fillna("Unknown")
    df_clean["case_info"] = df_clean["case_info"].fillna("")
    df_clean["name"] = df_clean["name"].fillna("unknown_case")

    df_clean["judgement"] = (
        df_clean["judgement"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    df_clean["proof_sentence"] = (
        df_clean["proof_sentence"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    df_clean["tokens"] = df_clean["tokens"].fillna(
        df_clean["judgement"].astype(str).str.split().str.len()
    )

    df_clean["sentences"] = df_clean["sentences"].fillna(
        df_clean["judgement"].astype(str).str.count(r"[.!?]") + 1
    )

    df_clean["judgement_masked"] = df_clean["judgement"].apply(mask_outcome_words)
    df_clean["proof_sentence_masked"] = df_clean["proof_sentence"].apply(mask_outcome_words)

    df_clean["label_encoded"] = df_clean["label"].map({
        "Rejected": 0,
        "Accepted": 1
    })

    df_clean = df_clean.dropna(subset=[
        "judgement_masked",
        "proof_sentence_masked",
        "label_encoded"
    ])

    df_clean["label_encoded"] = df_clean["label_encoded"].astype(int)

    return df_clean


def split_dataset(df_clean):
    train_df, temp_df = train_test_split(
        df_clean,
        test_size=0.2,
        stratify=df_clean["label_encoded"],
        random_state=RANDOM_STATE
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        stratify=temp_df["label_encoded"],
        random_state=RANDOM_STATE
    )

    return train_df, val_df, test_df


def save_outputs(df_clean, train_df, val_df, test_df):
    df_clean.to_csv(CLEAN_DATA_PATH, index=False)
    train_df.to_csv(TRAIN_PATH, index=False)
    val_df.to_csv(VAL_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)

    print("\nFiles saved:")
    print(f"Clean data: {CLEAN_DATA_PATH}")
    print(f"Train data: {TRAIN_PATH}")
    print(f"Validation data: {VAL_PATH}")
    print(f"Test data: {TEST_PATH}")
    print(f"SQLite database: {DATABASE_PATH}")


def main():
    create_folders()

    df = load_raw_data()

    print("\nCreating SQLite data layer...")
    df_clean = create_sqlite_view(df)

    print("\nCleaning dataset and applying leakage-aware masking...")
    df_clean = clean_dataset(df_clean)

    print("\nSplitting dataset...")
    train_df, val_df, test_df = split_dataset(df_clean)

    save_outputs(df_clean, train_df, val_df, test_df)

    print("\nFinal summary:")
    print(f"Clean rows: {len(df_clean):,}")
    print(f"Train rows: {len(train_df):,}")
    print(f"Validation rows: {len(val_df):,}")
    print(f"Test rows: {len(test_df):,}")

    print("\nLabel distribution in clean data:")
    print(df_clean["label_encoded"].value_counts(normalize=True).round(3))

    print("\nColumns used for ML:")
    print("- proof_sentence_masked: Teacher input during training")
    print("- judgement_masked: Student input during training and inference")
    print("- label_encoded: target")


if __name__ == "__main__":
    main()