def temporal_magnitude_attribution(
    drift_scores,
    onset_lags,
    epsilon: float = 1e-6
):
    scores = {}

    for stage, drift_score in drift_scores.items():
        lag = onset_lags.get(stage, 0)

        scores[stage] = float(
            drift_score / (lag + epsilon)
        )

    total = sum(scores.values())

    if total > 0:
        normalized = {
            stage: round(score / total, 4)
            for stage, score in scores.items()
        }
    else:
        normalized = {
            stage: 0.0
            for stage in scores
        }

    ranked = sorted(
        normalized.items(),
        key=lambda item: item[1],
        reverse=True
    )

    return {
        "stage_scores": normalized,
        "ranking": ranked,
        "predicted_origin": ranked[0][0] if ranked else "unknown"
    }
