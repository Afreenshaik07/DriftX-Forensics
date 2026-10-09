def calculate_forensic_score(
    feature_score: float,
    prediction_score: float,
    residual_score: float,
    performance_score: float
):
    weights = {
        "feature": 0.20,
        "prediction": 0.20,
        "residual": 0.25,
        "performance": 0.35
    }

    score = (
        feature_score * weights["feature"]
        + prediction_score * weights["prediction"]
        + residual_score * weights["residual"]
        + performance_score * weights["performance"]
    )

    return round(min(max(score, 0.0), 1.0), 4)
