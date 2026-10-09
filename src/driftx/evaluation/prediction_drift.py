from scipy.stats import ks_2samp


def detect_prediction_drift(
    reference_predictions,
    current_predictions,
    threshold: float = 0.05
):
    statistic, p_value = ks_2samp(
        reference_predictions,
        current_predictions
    )

    return {
        "stage": "prediction",
        "ks_statistic": float(statistic),
        "p_value": float(p_value),
        "drift_detected": bool(p_value < threshold)
    }
