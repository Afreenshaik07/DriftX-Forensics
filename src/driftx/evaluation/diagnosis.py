def diagnose_origin(
    feature_score: float,
    prediction_score: float,
    residual_score: float,
    performance_score: float
):
    scores = {
        "feature": feature_score,
        "prediction": prediction_score,
        "residual": residual_score,
        "performance": performance_score
    }

    origin_stage = max(scores, key=scores.get)
    confidence = scores[origin_stage]

    return {
        "origin_stage": origin_stage,
        "confidence": round(float(confidence), 4),
        "evidence_scores": scores
    }
