from typing import Dict


STAGE_ORDER = [
    "feature",
    "prediction",
    "residual",
    "performance"
]


def _normalize(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _temporal_consistency(stage: str, onset_lags: Dict[str, float]) -> float:
    if stage not in onset_lags:
        return 0.0

    origin_lag = onset_lags[stage]

    downstream = [
        onset_lags[s]
        for s in STAGE_ORDER
        if s in onset_lags
        and STAGE_ORDER.index(s) > STAGE_ORDER.index(stage)
    ]

    if not downstream:
        return 1.0

    valid = sum(1 for lag in downstream if lag >= origin_lag)

    return valid / len(downstream)


def _downstream_support(
    stage: str,
    evidence: Dict[str, float]
) -> float:
    index = STAGE_ORDER.index(stage)
    downstream = STAGE_ORDER[index + 1:]

    if not downstream:
        return 1.0

    values = [
        _normalize(evidence.get(s, 0.0))
        for s in downstream
    ]

    return sum(values) / len(values)


def _upstream_isolation(
    stage: str,
    evidence: Dict[str, float]
) -> float:
    index = STAGE_ORDER.index(stage)
    upstream = STAGE_ORDER[:index]

    if not upstream:
        return 1.0

    upstream_evidence = [
        _normalize(evidence.get(s, 0.0))
        for s in upstream
    ]

    return 1.0 - (
        sum(upstream_evidence) / len(upstream_evidence)
    )


def diagnose_cross_stage_origin(
    evidence: Dict[str, float],
    onset_lags: Dict[str, float],
    impact: Dict[str, float]
) -> Dict:
    """
    Cross-stage forensic diagnosis.

    Decision principle:
    1. Reject the case as NO_DRIFT when there is insufficient evidence.
    2. Build the observed propagation path.
    3. Select the earliest sufficiently supported stage.
    4. Treat later stages as propagated consequences rather than origins.

    This separates causal origin from downstream impact.
    """

    normalized = {
        stage: _normalize(evidence.get(stage, 0.0))
        for stage in STAGE_ORDER
    }

    max_evidence = max(normalized.values(), default=0.0)

    # Explicit no-drift / insufficient-evidence gate.
    if max_evidence < 0.10:
        return {
            "predicted_origin": "none",
            "confidence": round(1.0 - max_evidence, 4),
            "ranking": [],
            "propagation_path": [],
            "method": "causal_precedence_with_no_drift_gate"
        }

    # A stage is considered supported when its local evidence
    # is meaningful relative to the strongest observed signal.
    support_floor = max(0.20, 0.50 * max_evidence)

    supported = [
        stage
        for stage in STAGE_ORDER
        if normalized[stage] >= support_floor
    ]

    # Preserve temporal ordering when available.
    supported_with_onset = [
        stage
        for stage in supported
        if stage in onset_lags
    ]

    if supported_with_onset:
        earliest = min(
            supported_with_onset,
            key=lambda stage: (
                onset_lags[stage],
                STAGE_ORDER.index(stage)
            )
        )
    elif supported:
        earliest = supported[0]
    else:
        earliest = max(
            STAGE_ORDER,
            key=lambda stage: normalized[stage]
        )

    # Construct the observed propagation path from the earliest
    # supported stage forward.
    start_index = STAGE_ORDER.index(earliest)

    propagation_path = [
        stage
        for stage in STAGE_ORDER[start_index:]
        if normalized[stage] >= 0.10
    ]

    # Ranking is still reported for transparency, but ranking does
    # not override causal precedence.
    ranking = []

    for stage in STAGE_ORDER:
        temporal = _temporal_consistency(stage, onset_lags)
        downstream = _downstream_support(stage, evidence)
        upstream = _upstream_isolation(stage, evidence)

        forensic_score = (
            normalized[stage]
            * (
                0.40
                + 0.20 * temporal
                + 0.20 * downstream
                + 0.20 * upstream
            )
        )

        ranking.append({
            "stage": stage,
            "local_evidence": round(normalized[stage], 4),
            "temporal_precedence": round(temporal, 4),
            "downstream_support": round(downstream, 4),
            "upstream_isolation": round(upstream, 4),
            "impact_evidence": round(
                _normalize(impact.get(stage, 0.0)), 4
            ),
            "forensic_score": round(forensic_score, 4)
        })

    ranking.sort(
        key=lambda item: item["forensic_score"],
        reverse=True
    )

    origin_evidence = normalized[earliest]

    downstream_count = max(
        0,
        len(propagation_path) - 1
    )

    confidence = _normalize(
        0.60 * origin_evidence
        + 0.20 * (
            1.0 if earliest in onset_lags else 0.0
        )
        + 0.20 * min(
            1.0,
            downstream_count / 2.0
        )
    )

    return {
        "predicted_origin": earliest,
        "confidence": round(confidence, 4),
        "ranking": ranking,
        "propagation_path": propagation_path,
        "method": "causal_precedence_with_no_drift_gate"
    }
