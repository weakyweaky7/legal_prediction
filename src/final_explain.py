import argparse
import json
from datetime import datetime

from config import MODELS_DIR

from predict_final import (
    LABELS,
    ABSTAIN_CONFIDENCE_THRESHOLD,
    ABSTAIN_MARGIN_THRESHOLD,
    load_threshold,
    load_model,
    predict_judgment,
    load_text_from_file,
    load_text_from_dataset,
)

from explain_lime import explain_with_lime

from legal_layer import (
    LEGAL_NORMS_PATH,
    load_legal_norms,
    filter_lime_items,
    match_legal_patterns,
)


# ============================================================
# Paths
# ============================================================

FINAL_OUTPUT_DIR = MODELS_DIR / "final_outputs"
LATEST_FINAL_OUTPUT_PATH = FINAL_OUTPUT_DIR / "latest_final_explanation.json"


# ============================================================
# Human-readable explanation
# ============================================================

def build_human_explanation(prediction_result, filtered_supportive, related_patterns):
    final_prediction = prediction_result["prediction"]
    raw_prediction = prediction_result.get("raw_prediction", final_prediction)

    probability_accepted = prediction_result["probability_accepted"]
    probability_rejected = prediction_result.get(
        "probability_rejected",
        1.0 - probability_accepted
    )

    confidence = prediction_result["confidence"]
    margin = prediction_result.get("probability_margin", prediction_result.get("margin", None))
    risk_level = prediction_result.get("risk_level", "Unknown")
    reason = prediction_result.get("reason", "")
    recommendation = prediction_result.get("recommendation", "")

    is_abstained = prediction_result.get("is_abstained", False)

    top_words = [item["word"] for item in filtered_supportive[:6]]

    # For legal pattern explanation:
    # If final prediction is Uncertain, legal patterns are explained according to raw model tendency.
    pattern_basis = raw_prediction if is_abstained else final_prediction

    outcome_patterns = [
        pattern["title"]
        for pattern in related_patterns
        if pattern["outcome_type"] == pattern_basis
    ]

    neutral_patterns = [
        pattern["title"]
        for pattern in related_patterns
        if pattern["outcome_type"] == "Neutral"
    ]

    if top_words:
        words_part = ", ".join(top_words)
    else:
        words_part = "no strong filtered LIME words"

    if outcome_patterns:
        outcome_patterns_part = ", ".join(outcome_patterns[:3])
    else:
        outcome_patterns_part = "no direct outcome-related patterns"

    if neutral_patterns:
        neutral_patterns_part = ", ".join(neutral_patterns[:3])
    else:
        neutral_patterns_part = "no additional neutral context patterns"

    if is_abstained:
        margin_part = f"{margin:.4f}" if margin is not None else "not available"

        explanation = (
            f"The system returned Uncertain instead of forcing an Accepted or Rejected result. "
            f"The raw model tendency was {raw_prediction}, but the confidence was {confidence:.4f} "
            f"and the probability margin was {margin_part}. "
            f"The probability of Accepted was {probability_accepted:.4f}, while the probability "
            f"of Rejected was {probability_rejected:.4f}. "
            f"The reason for abstaining was: {reason} "
            f"The main diagnostic LIME words were: {words_part}. "
            f"These words matched legal outcome patterns related to the raw model tendency, such as: "
            f"{outcome_patterns_part}. "
            f"Additional contextual legal patterns were: {neutral_patterns_part}. "
            f"Because this is a legal decision-support system, the related legal patterns are shown "
            f"for reference only, and the final conclusion should be made by a human legal expert. "
            f"Recommendation: {recommendation}"
        )

        return explanation

    if final_prediction == "Accepted":
        explanation = (
            f"The system predicted Accepted with confidence {confidence:.4f}. "
            f"The most influential chunk produced probability {probability_accepted:.4f} for Accepted. "
            f"The main supporting words were: {words_part}. "
            f"These words matched outcome-related patterns such as: {outcome_patterns_part}. "
            f"Additional contextual patterns found in the chunk were: {neutral_patterns_part}. "
            f"Overall, the selected part of the judgment contains language commonly associated "
            f"with an accepted or allowed appeal, especially phrases related to setting aside "
            f"a previous judgment. "
            f"Risk level: {risk_level}. "
            f"Recommendation: {recommendation}"
        )

    elif final_prediction == "Rejected":
        explanation = (
            f"The system predicted Rejected with confidence {confidence:.4f}. "
            f"The probability of Accepted was {probability_accepted:.4f}, which is below the model "
            f"decision threshold. "
            f"The main supporting words were: {words_part}. "
            f"These words matched outcome-related patterns such as: {outcome_patterns_part}. "
            f"Additional contextual patterns found in the chunk were: {neutral_patterns_part}. "
            f"Overall, the selected part of the judgment contains language commonly associated "
            f"with a rejected or dismissed appeal. "
            f"Risk level: {risk_level}. "
            f"Recommendation: {recommendation}"
        )

    else:
        explanation = (
            f"The system returned {final_prediction}. "
            f"Confidence: {confidence:.4f}. "
            f"The main supporting words were: {words_part}. "
            f"Related legal patterns: {outcome_patterns_part}. "
            f"Recommendation: {recommendation}"
        )

    return explanation


# ============================================================
# Saving
# ============================================================

def save_final_result(result):
    FINAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped_path = FINAL_OUTPUT_DIR / f"final_explanation_{timestamp}.json"

    with open(timestamped_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

    with open(LATEST_FINAL_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

    return timestamped_path, LATEST_FINAL_OUTPUT_PATH


# ============================================================
# Printing
# ============================================================

def print_final_output(result):
    prediction = result["prediction_result"]
    lime = result["filtered_lime"]
    patterns = result["related_legal_patterns"]
    human_explanation = result["human_readable_explanation"]

    is_abstained = prediction.get("is_abstained", False)

    print("\n================ FINAL LEGAL AI OUTPUT ================")

    print(f"Final prediction: {prediction['prediction']}")

    if is_abstained:
        print(f"Raw model tendency: {prediction['raw_prediction']}")
        print(f"Raw predicted label: {prediction['raw_predicted_label']}")
        print("Final predicted label: None because the system abstained.")
    else:
        print(f"Predicted label: {prediction.get('final_predicted_label', prediction.get('predicted_label'))}")

    print(f"Probability Accepted: {prediction['probability_accepted']:.4f}")
    print(f"Probability Rejected: {prediction['probability_rejected']:.4f}")
    print(f"Confidence: {prediction['confidence']:.4f}")
    print(f"Probability margin: {prediction['probability_margin']:.4f}")

    print(f"Model decision threshold: {prediction['decision_threshold']:.4f}")
    print(f"Abstain confidence threshold: {prediction['abstain_confidence_threshold']:.4f}")
    print(f"Abstain margin threshold: {prediction['abstain_margin_threshold']:.4f}")

    print(f"Risk level: {prediction['risk_level']}")
    print(f"Reason: {prediction['reason']}")
    print(f"Recommendation: {prediction['recommendation']}")

    print(f"Chunks analyzed: {prediction['chunks_analyzed']}")

    if result.get("true_label") is not None:
        true_label = result["true_label"]

        print("\nGround truth:")
        print(f"True label: {true_label}")
        print(f"True outcome: {LABELS[true_label]}")

        if is_abstained:
            raw_correctness = (
                "correct"
                if prediction["raw_predicted_label"] == true_label
                else "incorrect"
            )
            print("Final correctness: not evaluated because the system abstained.")
            print(f"Raw model correctness: {raw_correctness}")
        else:
            final_label = prediction.get("final_predicted_label", prediction.get("predicted_label"))
            final_correctness = "correct" if final_label == true_label else "incorrect"
            print(f"Final correctness: {final_correctness}")

    print("\nChunk probabilities:")
    for i, prob in enumerate(prediction["chunk_probabilities"]):
        print(f"Chunk {i}: P(Accepted) = {prob:.4f}")

    print("\nMost influential chunk:")
    print(f"Chunk index: {prediction['most_influential_chunk_index']}")
    print(f"Chunk probability: {prediction['most_influential_chunk_probability']:.4f}")
    print("Text preview:")
    print(prediction["most_influential_chunk_text"][:1200])

    if is_abstained:
        print("\nDiagnostic important words for raw model tendency:")
    else:
        print("\nFiltered important words supporting prediction:")

    if lime["supportive_words"]:
        for item in lime["supportive_words"]:
            print(f"- {item['word']}: {item['weight']:.4f}")
    else:
        print("- None")

    print("\nFiltered important words against prediction:")
    if lime["opposing_words"]:
        for item in lime["opposing_words"]:
            print(f"- {item['word']}: {item['weight']:.4f}")
    else:
        print("- None")

    print("\nRelated legal outcome patterns:")

    if is_abstained:
        print(
            "Note: because the final prediction is Uncertain, these patterns are shown "
            "for reference based on the raw model tendency."
        )

    if patterns:
        for i, pattern in enumerate(patterns, start=1):
            print(f"\n{i}. {pattern['title']}")
            print(f"   Outcome type: {pattern['outcome_type']}")
            print(f"   Score: {pattern['score']:.4f}")

            matched_keywords = pattern.get("matched_keywords", [])
            matched_lime_words = pattern.get("matched_lime_words", [])

            if matched_keywords:
                print(f"   Matched keywords: {', '.join(matched_keywords)}")
            else:
                print("   Matched keywords: None")

            if matched_lime_words:
                print(f"   Matched LIME words: {', '.join(matched_lime_words)}")
            else:
                print("   Matched LIME words: None")

            print(f"   Description: {pattern['description']}")
    else:
        print("- No related legal outcome patterns found.")

    print("\nHuman-readable explanation:")
    print(human_explanation)

    print("\nNote:")
    print("This system is a legal decision-support prototype, not a substitute for professional legal judgment.")
    print("=======================================================")


# ============================================================
# Main pipeline
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Final demo pipeline: prediction + uncertainty mode + LIME + legal layer."
    )

    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Raw judgment text."
    )

    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Path to a .txt file containing judgment text."
    )

    parser.add_argument(
        "--sample_index",
        type=int,
        default=None,
        help="Index of sample from test/val split."
    )

    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "val"],
        help="Dataset split for sample_index."
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=500,
        help="Number of LIME perturbation samples. Use 300 for speed, 1000 for stronger explanation."
    )

    parser.add_argument(
        "--top_k_words",
        type=int,
        default=12,
        help="Number of LIME words/features to show."
    )

    parser.add_argument(
        "--top_k_patterns",
        type=int,
        default=5,
        help="Number of related legal patterns to show."
    )

    parser.add_argument(
        "--include_all_patterns",
        action="store_true",
        help="If set, include legal patterns from all outcome types."
    )

    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=ABSTAIN_CONFIDENCE_THRESHOLD,
        help="Minimum confidence required for hard Accepted/Rejected prediction."
    )

    parser.add_argument(
        "--margin_threshold",
        type=float,
        default=ABSTAIN_MARGIN_THRESHOLD,
        help="Minimum probability margin required for hard Accepted/Rejected prediction."
    )

    args = parser.parse_args()

    print("Loading final model and threshold...")

    threshold, threshold_path = load_threshold()
    model, model_path = load_model()

    print(f"Model path: {model_path}")
    print(f"Threshold path: {threshold_path}")
    print(f"Model decision threshold: {threshold:.4f}")
    print(f"Abstain confidence threshold: {args.confidence_threshold:.4f}")
    print(f"Abstain margin threshold: {args.margin_threshold:.4f}")

    true_label = None

    if args.text is not None:
        text = args.text

    elif args.file is not None:
        text = load_text_from_file(args.file)

    elif args.sample_index is not None:
        text, true_label = load_text_from_dataset(args.split, args.sample_index)

    else:
        raise ValueError(
            "Provide one of: --text, --file, or --sample_index."
        )

    from transformers import AutoTokenizer
    from config import MODEL_NAME

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.model_max_length = int(1e9)

    print("\nStep 1/4: Running final chunked prediction with Uncertainty Mode...")

    prediction_result = predict_judgment(
        model=model,
        tokenizer=tokenizer,
        text=text,
        threshold=threshold,
        confidence_threshold=args.confidence_threshold,
        margin_threshold=args.margin_threshold,
    )

    selected_chunk_text = prediction_result["most_influential_chunk_text"]

    # If final prediction is Uncertain, LIME should still explain the raw model tendency.
    # This keeps the explanation diagnostic instead of pretending that the system made
    # a final hard legal decision.
    lime_target_label = prediction_result.get(
        "raw_predicted_label",
        prediction_result.get("predicted_label")
    )

    print("Step 2/4: Running LIME on the most influential chunk...")

    lime_result = explain_with_lime(
        model=model,
        tokenizer=tokenizer,
        chunk_text=selected_chunk_text,
        predicted_label=lime_target_label,
        num_features=args.top_k_words,
        num_samples=args.num_samples
    )

    print("Step 3/4: Filtering noisy LIME words...")

    raw_supportive = lime_result.get("supportive_words", [])
    raw_opposing = lime_result.get("opposing_words", [])

    filtered_supportive = filter_lime_items(raw_supportive)
    filtered_opposing = filter_lime_items(raw_opposing)

    print("Step 4/4: Matching legal outcome patterns...")

    legal_norms = load_legal_norms(LEGAL_NORMS_PATH)

    # Legal pattern matching should use the final prediction if it is hard prediction.
    # If the system abstained, patterns are matched using raw model tendency for reference only.
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
        include_all_patterns=args.include_all_patterns,
        top_k=args.top_k_patterns
    )

    human_explanation = build_human_explanation(
        prediction_result=prediction_result,
        filtered_supportive=filtered_supportive,
        related_patterns=related_patterns
    )

    final_result = {
        # Top-level fields for easier frontend use
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

        # Full nested output
        "prediction_result": prediction_result,
        "true_label": true_label,

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
                "word"
            ],
        },

        "related_legal_patterns": related_patterns,
        "legal_pattern_matching_basis": legal_pattern_matching_basis,

        "human_readable_explanation": human_explanation,

        "model_path": str(model_path),
        "threshold_path": str(threshold_path),
        "legal_norms_path": str(LEGAL_NORMS_PATH),

        "settings": {
            "lime_num_samples": args.num_samples,
            "top_k_words": args.top_k_words,
            "top_k_patterns": args.top_k_patterns,
            "include_all_patterns": args.include_all_patterns,
            "model_decision_threshold": threshold,
            "abstain_confidence_threshold": args.confidence_threshold,
            "abstain_margin_threshold": args.margin_threshold,
        },

        "responsible_ai_note": (
            "This system is a legal decision-support prototype. "
            "If confidence is below the abstain threshold or the probability margin is too small, "
            "the system returns Uncertain and recommends human legal review."
        )
    }

    timestamped_path, latest_path = save_final_result(final_result)

    print_final_output(final_result)

    print("\nFinal explanation saved:")
    print(timestamped_path)
    print("\nLatest final explanation saved:")
    print(latest_path)


if __name__ == "__main__":
    main()