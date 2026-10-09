import numpy as np
from scipy.stats import ks_2samp


def ks_drift_test(
    reference,
    current,
    threshold: float = 0.05
):
    statistic, p_value = ks_2samp(reference, current)

    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "drift_detected": bool(p_value < threshold)
    }
