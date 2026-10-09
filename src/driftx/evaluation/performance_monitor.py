from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np


def evaluate_performance(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    return {
        "stage": "performance",
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2)
    }


def detect_performance_degradation(
    reference_metrics,
    current_metrics,
    mae_increase_threshold: float = 0.10,
    r2_drop_threshold: float = 0.10
):
    mae_change = (
        (current_metrics["mae"] - reference_metrics["mae"])
        / max(reference_metrics["mae"], 1e-12)
    )

    r2_change = (
        reference_metrics["r2"] - current_metrics["r2"]
    )

    degraded = (
        mae_change >= mae_increase_threshold
        or r2_change >= r2_drop_threshold
    )

    return {
        "stage": "performance",
        "mae_change": float(mae_change),
        "r2_drop": float(r2_change),
        "degradation_detected": bool(degraded)
    }
