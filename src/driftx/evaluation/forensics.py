from typing import List, Dict


def analyze_drift_propagation(
    feature_events: List[Dict],
    prediction_drift: bool,
    performance_degradation: bool
) -> Dict:

    affected_features = [
        event["feature"]
        for event in feature_events
        if event.get("drift_detected", False)
    ]

    if performance_degradation:
        final_stage = "performance"
    elif prediction_drift:
        final_stage = "prediction"
    elif affected_features:
        final_stage = "feature"
    else:
        final_stage = "none"

    if affected_features and performance_degradation:
        propagation_path = [
            "feature",
            "prediction",
            "performance"
        ]
    elif affected_features and prediction_drift:
        propagation_path = [
            "feature",
            "prediction"
        ]
    elif affected_features:
        propagation_path = ["feature"]
    else:
        propagation_path = []

    return {
        "origin_stage": "feature" if affected_features else "unknown",
        "final_stage": final_stage,
        "affected_features": affected_features,
        "propagation_path": propagation_path,
        "feature_count": len(affected_features)
    }
