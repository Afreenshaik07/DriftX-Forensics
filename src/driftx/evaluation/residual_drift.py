from scipy.stats import ks_2samp


def detect_residual_drift(
    reference_residuals,
    current_residuals,
    threshold: float = 0.05
):
    statistic, p_value = ks_2samp(
        reference_residuals,
        current_residuals
    )

    return {
        "stage": "residual",
        "ks_statistic": float(statistic),
        "p_value": float(p_value),
        "drift_detected": bool(p_value < threshold)
    }
