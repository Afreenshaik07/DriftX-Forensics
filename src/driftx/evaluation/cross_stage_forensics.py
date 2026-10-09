from typing import Any

STAGE_ORDER = ["feature", "prediction", "residual", "performance"]


def _normalize(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _temporal_consistency(stage: str, onset_lags: dict[str, float]) -> float:
    if stage not in onset_lags:
        return 0.0
    downstream = [
        onset_lags[s]
        for s in STAGE_ORDER[STAGE_ORDER.index(stage) + 1:]
        if s in onset_lags
    ]
    if not downstream:
        return 1.0
    return sum(lag >= onset_lags[stage] for lag in downstream) / len(downstream)


def diagnose_cross_stage_origin(
    evidence: dict[str, float],
    onset_lags: dict[str, float],
    impact: dict[str, float],
) -> dict[str, Any]:
    normalized = {
        stage: _normalize(evidence.get(stage, 0.0))
        for stage in STAGE_ORDER
    }
    max_evidence = max(normalized.values(), default=0.0)

    if max_evidence < 0.10:
        return {
            "predicted_origin": "none",
            "confidence": round(1.0 - max_evidence, 4),
            "ranking": [],
            "propagation_path": [],
            "method": "evidence_gated_temporal_forensics",
        }

    support_floor = max(0.10, 0.35 * max_evidence)
    supported = [
        stage for stage in STAGE_ORDER
        if normalized[stage] >= support_floor
    ]
    timed = [stage for stage in supported if stage in onset_lags]

    if timed:
        earliest = min(
            timed,
            key=lambda stage: (onset_lags[stage], STAGE_ORDER.index(stage)),
        )
    else:
        earliest = max(supported, key=lambda stage: normalized[stage])

    ranking = []
    for stage in STAGE_ORDER:
        temporal = _temporal_consistency(stage, onset_lags)
        upstream = STAGE_ORDER[:STAGE_ORDER.index(stage)]
        isolation = (
            1.0 - sum(normalized[s] for s in upstream) / len(upstream)
            if upstream else 1.0
        )
        score = (
            0.55 * normalized[stage]
            + 0.20 * temporal
            + 0.15 * isolation
            + 0.10 * _normalize(impact.get(stage, 0.0))
        )
        ranking.append({
            "stage": stage,
            "local_evidence": round(normalized[stage], 4),
            "temporal_precedence": round(temporal, 4),
            "upstream_isolation": round(isolation, 4),
            "impact_evidence": round(_normalize(impact.get(stage, 0.0)), 4),
            "forensic_score": round(score, 4),
        })

    ranking.sort(key=lambda item: item["forensic_score"], reverse=True)
    propagation_path = [
        stage for stage in STAGE_ORDER
        if stage in onset_lags and normalized[stage] >= support_floor
    ]
    if earliest not in propagation_path:
        propagation_path.insert(0, earliest)

    confidence = 0.65 * normalized[earliest] + 0.35 * (
        1.0 if earliest in onset_lags else 0.0
    )

    return {
        "predicted_origin": earliest,
        "confidence": round(_normalize(confidence), 4),
        "ranking": ranking,
        "propagation_path": propagation_path,
        "method": "evidence_gated_temporal_forensics",
    }

