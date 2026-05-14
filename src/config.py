from pathlib import Path
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]

RAW_DATA_PATH = ROOT_DIR / "data" / "raw" / "case_files_total.csv"
DATABASE_PATH = ROOT_DIR / "data" / "legal_cases.db"

CLEAN_DATA_PATH = ROOT_DIR / "data" / "processed" / "clean_data.csv"
TRAIN_PATH = ROOT_DIR / "data" / "splits" / "train.csv"
VAL_PATH = ROOT_DIR / "data" / "splits" / "val.csv"
TEST_PATH = ROOT_DIR / "data" / "splits" / "test.csv"

MODELS_DIR = ROOT_DIR / "models"
TEACHER_MODEL_PATH = MODELS_DIR / "teacher.pt"
TEACHER_METRICS_PATH = MODELS_DIR / "teacher_metrics.json"
STUDENT_MODEL_PATH = MODELS_DIR / "best_student.pt"
STUDENT_METRICS_PATH = MODELS_DIR / "student_metrics.json"
THRESHOLD_PATH = MODELS_DIR / "threshold.json"
BASELINE_MODEL_PATH = MODELS_DIR / "baseline_tfidf_logreg.joblib"
BASELINE_METRICS_PATH = MODELS_DIR / "baseline_metrics.json"

MODEL_NAME = "nlpaueb/legal-bert-base-uncased"


DEVICE = "cpu"

RANDOM_STATE = 42
#BATCH_SIZE = 8
#EPOCHS = 3
LR = 2e-5


BATCH_SIZE = 2
EPOCHS = 1

MAX_PROOF_LEN = 96
MAX_JUDGMENT_LEN = 512



#MAX_PROOF_LEN = 128
#MAX_JUDGMENT_LEN = 512

TEMPERATURE = 4.0
ALPHA = 1.0
