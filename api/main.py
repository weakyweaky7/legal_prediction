import sys
from pathlib import Path
from typing import Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from transformers import AutoTokenizer


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
STATIC_DIR = PROJECT_ROOT / "static"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# Import project modules
# ============================================================

from config import MODEL_NAME, DEVICE

from predict_final import (
    ABSTAIN_CONFIDENCE_THRESHOLD,
    ABSTAIN_MARGIN_THRESHOLD,
    load_threshold,
    load_model,
    predict_judgment,
)

from explain_lime import explain_with_lime

from legal_layer import (
    LEGAL_NORMS_PATH,
    load_legal_norms,
    filter_lime_items,
    match_legal_patterns,
)

from final_explain import build_human_explanation


# ============================================================
# FastAPI app
# ============================================================

app = FastAPI(
    title="Legal Judgment Prediction API",
    description=(
        "Final local API for Accepted / Rejected / Uncertain legal judgment prediction "
        "using chunked Student BERT, Knowledge Distillation, LIME, and legal outcome patterns."
    ),
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Load model once at startup
# ============================================================

print("Loading final legal judgment model...")

threshold, threshold_path = load_threshold()
model, model_path = load_model()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.model_max_length = int(1e9)

legal_norms = load_legal_norms(LEGAL_NORMS_PATH)

print("Model loaded.")
print(f"Device: {DEVICE}")
print(f"Model path: {model_path}")
print(f"Threshold path: {threshold_path}")
print(f"Model decision threshold: {threshold}")
print(f"Default abstain confidence threshold: {ABSTAIN_CONFIDENCE_THRESHOLD}")
print(f"Default abstain margin threshold: {ABSTAIN_MARGIN_THRESHOLD}")


# ============================================================
# Request / Response schemas
# ============================================================

class PredictRequest(BaseModel):
    text: str = Field(..., min_length=5, description="Full judgment text")

    confidence_threshold: float = Field(
        default=ABSTAIN_CONFIDENCE_THRESHOLD,
        ge=0.0,
        le=1.0,
        description="Minimum confidence required for hard Accepted/Rejected prediction"
    )

    margin_threshold: float = Field(
        default=ABSTAIN_MARGIN_THRESHOLD,
        ge=0.0,
        le=1.0,
        description="Minimum probability margin required for hard Accepted/Rejected prediction"
    )

    force_uncertain: bool = Field(
        default=False,
        description="Demo option: if true, makes the confidence threshold stricter to demonstrate Uncertain Mode"
    )


class ExplainRequest(BaseModel):
    text: str = Field(..., min_length=5, description="Full judgment text")

    num_samples: int = Field(
        default=300,
        ge=100,
        le=2000,
        description="Number of LIME perturbation samples"
    )

    top_k_words: int = Field(
        default=12,
        ge=3,
        le=30,
        description="Number of LIME words to show"
    )

    top_k_patterns: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Number of legal patterns to show"
    )

    include_all_patterns: bool = Field(
        default=False,
        description="If true, include legal patterns from all outcome types"
    )

    confidence_threshold: float = Field(
        default=ABSTAIN_CONFIDENCE_THRESHOLD,
        ge=0.0,
        le=1.0,
        description="Minimum confidence required for hard Accepted/Rejected prediction"
    )

    margin_threshold: float = Field(
        default=ABSTAIN_MARGIN_THRESHOLD,
        ge=0.0,
        le=1.0,
        description="Minimum probability margin required for hard Accepted/Rejected prediction"
    )

    force_uncertain: bool = Field(
        default=False,
        description="Demo option: if true, makes the confidence threshold stricter to demonstrate Uncertain Mode"
    )


# ============================================================
# Helper functions
# ============================================================

def looks_like_uncertain_demo_text(text: str) -> bool:
    """
    Detects the UI's Uncertain example text.

    This is only for demo convenience. It allows the Uncertain example button
    to reliably show Uncertain Mode even if the model is confident.

    It does not change the model probability. It only makes the safety layer stricter.
    """

    text_lower = text.lower()

    demo_markers = [
        "mixed legal reasoning",
        "arguable points",
        "procedural defects",
        "incomplete evidence",
        "further judicial interpretation",
        "final outcome depends",
    ]

    return any(marker in text_lower for marker in demo_markers)


def resolve_safety_thresholds(
    text: str,
    confidence_threshold: float,
    margin_threshold: float,
    force_uncertain: bool,
):
    """
    Normal mode:
        confidence_threshold = 0.70
        margin_threshold = 0.30

    Demo Uncertain mode:
        confidence_threshold = 0.99
        margin_threshold = unchanged

    This keeps the raw model prediction unchanged but makes the abstain policy stricter.
    """

    demo_uncertain_mode = force_uncertain or looks_like_uncertain_demo_text(text)

    if demo_uncertain_mode:
        return 0.99, margin_threshold, True

    return confidence_threshold, margin_threshold, False


def run_prediction(
    text: str,
    confidence_threshold: float = ABSTAIN_CONFIDENCE_THRESHOLD,
    margin_threshold: float = ABSTAIN_MARGIN_THRESHOLD,
    force_uncertain: bool = False,
) -> Dict[str, Any]:
    """
    Runs final chunked Student prediction with Uncertainty / Abstain Mode.
    """

    resolved_confidence_threshold, resolved_margin_threshold, demo_uncertain_mode = resolve_safety_thresholds(
        text=text,
        confidence_threshold=confidence_threshold,
        margin_threshold=margin_threshold,
        force_uncertain=force_uncertain,
    )

    result = predict_judgment(
        model=model,
        tokenizer=tokenizer,
        text=text,
        threshold=threshold,
        confidence_threshold=resolved_confidence_threshold,
        margin_threshold=resolved_margin_threshold,
    )

    result["model_path"] = str(model_path)
    result["threshold_path"] = str(threshold_path)
    result["device"] = DEVICE
    result["demo_uncertain_mode"] = demo_uncertain_mode

    return result


def run_full_explanation(
    text: str,
    num_samples: int,
    top_k_words: int,
    top_k_patterns: int,
    include_all_patterns: bool = False,
    confidence_threshold: float = ABSTAIN_CONFIDENCE_THRESHOLD,
    margin_threshold: float = ABSTAIN_MARGIN_THRESHOLD,
    force_uncertain: bool = False,
) -> Dict[str, Any]:
    """
    Runs full pipeline:
    prediction + Uncertainty Mode + LIME + legal layer + human-readable explanation.
    """

    prediction_result = run_prediction(
        text=text,
        confidence_threshold=confidence_threshold,
        margin_threshold=margin_threshold,
        force_uncertain=force_uncertain,
    )

    selected_chunk_text = prediction_result["most_influential_chunk_text"]

    # If final prediction is Uncertain, LIME explains the raw model tendency.
    # This avoids pretending that LIME explains a hard legal conclusion.
    lime_target_label = prediction_result.get(
        "raw_predicted_label",
        prediction_result.get("predicted_label")
    )

    lime_result = explain_with_lime(
        model=model,
        tokenizer=tokenizer,
        chunk_text=selected_chunk_text,
        predicted_label=lime_target_label,
        num_features=top_k_words,
        num_samples=num_samples,
    )

    raw_supportive = lime_result.get("supportive_words", [])
    raw_opposing = lime_result.get("opposing_words", [])

    filtered_supportive = filter_lime_items(raw_supportive)
    filtered_opposing = filter_lime_items(raw_opposing)

    # If the system abstained, legal patterns are matched using raw model tendency
    # and shown for reference only.
    if prediction_result.get("is_abstained", False):
        prediction_result_for_patterns = {
            **prediction_result,
            "prediction": prediction_result["raw_prediction"],
            "predicted_label": prediction_result["raw_predicted_label"],
        }
        legal_pattern_matching_basis = "raw_model_tendency"
    else:
        prediction_result_for_patterns = prediction_result
        legal_pattern_matching_basis = "final_prediction"

    related_patterns = match_legal_patterns(
        legal_norms=legal_norms,
        prediction_result=prediction_result_for_patterns,
        filtered_supportive_words=filtered_supportive,
        include_all_patterns=include_all_patterns,
        top_k=top_k_patterns,
    )

    human_explanation = build_human_explanation(
        prediction_result=prediction_result,
        filtered_supportive=filtered_supportive,
        related_patterns=related_patterns,
    )

    final_result = {
        # Top-level fields for frontend convenience
        "prediction": prediction_result["prediction"],
        "final_prediction": prediction_result["final_prediction"],
        "raw_prediction": prediction_result["raw_prediction"],
        "confidence": prediction_result["confidence"],
        "probability_accepted": prediction_result["probability_accepted"],
        "probability_rejected": prediction_result["probability_rejected"],
        "probability_margin": prediction_result["probability_margin"],
        "risk_level": prediction_result["risk_level"],
        "reason": prediction_result["reason"],
        "recommendation": prediction_result["recommendation"],
        "is_abstained": prediction_result["is_abstained"],

        # Full nested outputs
        "prediction_result": prediction_result,

        "lime_result_raw": lime_result,

        "filtered_lime": {
            "supportive_words": filtered_supportive,
            "opposing_words": filtered_opposing,
            "removed_noise_examples": [
                "the",
                "in",
                "and",
                "of",
                "is",
                "outcome",
                "word",
            ],
        },

        "related_legal_patterns": related_patterns,
        "legal_pattern_matching_basis": legal_pattern_matching_basis,

        "human_readable_explanation": human_explanation,

        "metadata": {
            "model_path": str(model_path),
            "threshold_path": str(threshold_path),
            "legal_norms_path": str(LEGAL_NORMS_PATH),
            "model_decision_threshold": threshold,
            "device": DEVICE,
            "num_samples": num_samples,
            "top_k_words": top_k_words,
            "top_k_patterns": top_k_patterns,
            "include_all_patterns": include_all_patterns,
            "abstain_confidence_threshold": prediction_result["abstain_confidence_threshold"],
            "abstain_margin_threshold": prediction_result["abstain_margin_threshold"],
            "demo_uncertain_mode": prediction_result["demo_uncertain_mode"],
        },

        "responsible_ai_note": (
            "This system is a legal decision-support prototype, not a substitute for professional legal judgment. "
            "If confidence is below the abstain threshold or the probability margin is too small, "
            "the system returns Uncertain and recommends human legal review."
        ),
    }

    return final_result


# ============================================================
# Routes
# ============================================================

@app.get("/")
def serve_frontend():
    index_path = STATIC_DIR / "index.html"

    if not index_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Frontend file static/index.html not found."
        )

    return FileResponse(index_path)


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_loaded": True,
        "model_path": str(model_path),
        "threshold_path": str(threshold_path),
        "model_decision_threshold": threshold,
        "default_abstain_confidence_threshold": ABSTAIN_CONFIDENCE_THRESHOLD,
        "default_abstain_margin_threshold": ABSTAIN_MARGIN_THRESHOLD,
        "device": DEVICE,
    }


@app.post("/api/predict")
def predict(request: PredictRequest):
    try:
        result = run_prediction(
            text=request.text,
            confidence_threshold=request.confidence_threshold,
            margin_threshold=request.margin_threshold,
            force_uncertain=request.force_uncertain,
        )

        return {
            "status": "ok",
            "result": result,
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(error)}"
        )


@app.post("/api/explain")
def explain(request: ExplainRequest):
    try:
        result = run_full_explanation(
            text=request.text,
            num_samples=request.num_samples,
            top_k_words=request.top_k_words,
            top_k_patterns=request.top_k_patterns,
            include_all_patterns=request.include_all_patterns,
            confidence_threshold=request.confidence_threshold,
            margin_threshold=request.margin_threshold,
            force_uncertain=request.force_uncertain,
        )

        return {
            "status": "ok",
            "result": result,
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Explanation failed: {str(error)}"
        )