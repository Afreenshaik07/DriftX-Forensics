def fuse_forensic_evidence(
    drift_scores,
    onset_lags,
    performance_impact
):
    fused = {}

    for stage, score in drift_scores.items():
        lag = onset_lags.get(stage, 0)

        temporal_consistency = 1.0 / (1.0 + lag)

        impact = performance_impact.get(stage, 0.0)

        fused[stage] = {
            "drift_score": float(score),
            "onset_lag": int(lag),
            "temporal_consistency": round(temporal_consistency, 4),
            "performance_impact": float(impact)
        }

    return fused
