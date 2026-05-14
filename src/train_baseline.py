import json
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
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
    TEST_PATH,
    MODELS_DIR,
    BASELINE_MODEL_PATH,
    BASELINE_METRICS_PATH,
    RANDOM_STATE,
)


def evaluate_model(model, df, split_name):
    X = df["judgement_masked"].astype(str)
    y_true = df["label_encoded"].astype(int)

    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_prob)
    gini = 2 * roc_auc - 1

    print(f"\n===== {split_name.upper()} METRICS =====")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print(f"Gini:      {gini:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "gini": gini,
    }


def main():
    print("Training baseline model: TF-IDF + Logistic Regression")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)
    test_df = pd.read_csv(TEST_PATH)

    print(f"Train rows: {len(train_df):,}")
    print(f"Validation rows: {len(val_df):,}")
    print(f"Test rows: {len(test_df):,}")

    X_train = train_df["judgement_masked"].astype(str)
    y_train = train_df["label_encoded"].astype(int)

    baseline_model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                max_features=20000,
                ngram_range=(1, 2),
                min_df=3,
                max_df=0.9,
                stop_words="english"
            )
        ),
        (
            "logreg",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=RANDOM_STATE
            )
        )
    ])

    print("\nFitting baseline model...")
    baseline_model.fit(X_train, y_train)

    val_metrics = evaluate_model(
        baseline_model,
        val_df,
        split_name="validation"
    )

    test_metrics = evaluate_model(
        baseline_model,
        test_df,
        split_name="test"
    )

    joblib.dump(baseline_model, BASELINE_MODEL_PATH)

    metrics = {
        "model": "TF-IDF + Logistic Regression",
        "input_column": "judgement_masked",
        "validation": val_metrics,
        "test": test_metrics
    }

    with open(BASELINE_METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=4)

    print("\nBaseline model saved:")
    print(BASELINE_MODEL_PATH)

    print("\nBaseline metrics saved:")
    print(BASELINE_METRICS_PATH)


if __name__ == "__main__":
    main()