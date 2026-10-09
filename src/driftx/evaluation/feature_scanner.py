import pandas as pd

from src.driftx.evaluation.drift_detector import ks_drift_test


def scan_feature_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    features,
    threshold: float = 0.05
) -> pd.DataFrame:

    results = []

    for feature in features:
        if feature not in reference.columns or feature not in current.columns:
            continue

        result = ks_drift_test(
            reference[feature].dropna(),
            current[feature].dropna(),
            threshold
        )

        results.append({
            "feature": feature,
            "ks_statistic": result["statistic"],
            "p_value": result["p_value"],
            "drift_detected": result["drift_detected"]
        })

    return pd.DataFrame(results).sort_values(
        "p_value",
        ascending=True
    ).reset_index(drop=True)
