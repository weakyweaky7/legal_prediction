def apply_abstain_mode(
    prob_accepted: float,
    confidence_threshold: float = 0.70,
    margin_threshold: float = 0.30
) -> dict:
    """
    Safety layer for legal AI predictions.

    The model still calculates probabilities for Accepted and Rejected,
    but the system refuses to give a hard legal prediction if confidence is low.
    """

    prob_accepted = float(prob_accepted)
    prob_rejected = 1.0 - prob_accepted

    confidence = max(prob_accepted, prob_rejected)
    margin = abs(prob_accepted - prob_rejected)

    raw_prediction = "Accepted" if prob_accepted >= 0.5 else "Rejected"

    if confidence < confidence_threshold or margin < margin_threshold:
        return {
            "prediction": "Uncertain",
            "raw_prediction": raw_prediction,
            "prob_accepted": round(prob_accepted, 4),
            "prob_rejected": round(prob_rejected, 4),
            "confidence": round(confidence, 4),
            "margin": round(margin, 4),
            "confidence_threshold": confidence_threshold,
            "margin_threshold": margin_threshold,
            "risk_level": "High",
            "reason": "The model confidence is below the legal decision threshold or the probability margin is too small.",
            "recommendation": "Human legal review is required before making any conclusion."
        }

    return {
        "prediction": raw_prediction,
        "raw_prediction": raw_prediction,
        "prob_accepted": round(prob_accepted, 4),
        "prob_rejected": round(prob_rejected, 4),
        "confidence": round(confidence, 4),
        "margin": round(margin, 4),
        "confidence_threshold": confidence_threshold,
        "margin_threshold": margin_threshold,
        "risk_level": "Low" if confidence >= 0.85 else "Medium",
        "reason": "The model confidence and probability margin are above the required thresholds.",
        "recommendation": "This result can be used as decision-support, but the final legal decision must remain human-driven."
    }