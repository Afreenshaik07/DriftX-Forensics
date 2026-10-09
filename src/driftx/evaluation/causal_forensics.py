from typing import Dict, List


STAGE_ORDER = [
    "feature",
    "prediction",
    "residual",
    "performance",
]

# Minimum evidence required before a stage can be considered
# a meaningful drift signal.
MEANINGFUL_THRESHOLD = 0.20

# If there is no temporal evidence and no stage has strong
# evidence, classify the situation as NO_DRIFT/unknown.
NO_DRIFT_STRONG_EVIDENCE = 0.45


def _clip(value):
    return max(0.0, min(1.0, float(value)))


def _downstream_impact(stage, order, evidence):
    index = order.index(stage)
    downstream = order[index + 1:]

    if not downstream:
        return 0.0

    values = [
        _clip(evidence.get(s, 0.0))
        for s in downstream
    ]

    return sum(values) / len(values)


def _upstream_evidence(stage, order, evidence):
    index = order.index(stage)
    upstream = order[:index]

    if not upstream:
        return 0.0

    values = [
        _clip(evidence.get(s, 0.0))
        for s in upstream
    ]

    return sum(values) / len(values)


def _temporal_precedence(stage, order, onset_lags):
    if stage not in onset_lags:
        return 0.0

    stage_onset = onset_lags[stage]

    downstream = [
        s
        for s in order
        if s in onset_lags
        and order.index(s) > order.index(stage)
    ]

    if not downstream:
        return 1.0

    valid = [
        s
        for s in downstream
        if onset_lags[s] >= stage_onset
    ]

    return len(valid) / len(downstream)


def rank_causal_origins(
    evidence: Dict[str, float],
    onset_lags: Dict[str, int],
    impact: Dict[str, float],
    stage_order: List[str] = None,
):
    """
    Cross-stage forensic origin diagnosis.

    Important safety behavior:
    - Weak isolated evidence is not automatically treated as drift.
    - Temporal propagation is preferred when available.
    - Impact magnitude alone cannot determine the origin.
    - If there is insufficient evidence, return 'unknown'.
    """

    order = stage_order or STAGE_ORDER

    # ---------------------------------------------------------
    # 1. Clean and normalize evidence
    # ---------------------------------------------------------

    clean_evidence = {
        stage: _clip(evidence.get(stage, 0.0))
        for stage in order
    }

    # ---------------------------------------------------------
    # 2. NO-DRIFT SAFETY CHECK
    # ---------------------------------------------------------
    #
    # A real drift should normally produce either:
    #
    #   a) temporal evidence
    #   OR
    #   b) a sufficiently strong stage signal.
    #
    # If neither exists, do not force a causal diagnosis.
    # ---------------------------------------------------------

    max_evidence = max(
        clean_evidence.values(),
        default=0.0
    )

    has_temporal_evidence = bool(onset_lags)

    residual_performance_corroboration = (
        clean_evidence.get("residual", 0.0) >= MEANINGFUL_THRESHOLD
        and impact.get("performance", 0.0) >= MEANINGFUL_THRESHOLD
    )

    if (
        not has_temporal_evidence
        and max_evidence < NO_DRIFT_STRONG_EVIDENCE
        and not residual_performance_corroboration
    ):
        return {
            "predicted_origin": "unknown",
            "confidence": 0.0,
            "ranking": [],
            "propagation_path": [],
            "method": "origin_impact_separation_v6_no_drift_guard",
        }

    # ---------------------------------------------------------
    # 3. Build candidate stages
    # ---------------------------------------------------------

    candidates = []

    for stage in order:

        local = clean_evidence[stage]

        if local <= 0:
            continue

        temporal = _temporal_precedence(
            stage,
            order,
            onset_lags,
        )

        downstream = _downstream_impact(
            stage,
            order,
            clean_evidence,
        )

        upstream = _upstream_evidence(
            stage,
            order,
            clean_evidence,
        )

        impact_value = _clip(
            impact.get(stage, 0.0)
        )

        # -----------------------------------------------------
        # Origin score
        # -----------------------------------------------------
        #
        # Local evidence:
        #     Did this stage actually change?
        #
        # Temporal precedence:
        #     Did it happen before downstream stages?
        #
        # Downstream support:
        #     Can it explain later changes?
        #
        # Upstream penalty:
        #     If earlier stages already changed strongly,
        #     this stage is less likely to be the origin.
        # -----------------------------------------------------

        origin_score = (
            0.40 * local
            + 0.25 * temporal
            + 0.25 * downstream
            + 0.10 * (1.0 - upstream)
        )

        impact_score = (
            0.70 * local
            + 0.30 * impact_value
        )

        candidates.append({
            "stage": stage,
            "local_evidence": round(local, 4),
            "onset": onset_lags.get(stage),
            "onset_observed": stage in onset_lags,
            "temporal_precedence": round(
                temporal,
                4,
            ),
            "downstream_support": round(
                downstream,
                4,
            ),
            "upstream_evidence": round(
                upstream,
                4,
            ),
            "impact_evidence": round(
                impact_value,
                4,
            ),
            "origin_score": round(
                _clip(origin_score),
                4,
            ),
            "impact_score": round(
                _clip(impact_score),
                4,
            ),
        })

    # ---------------------------------------------------------
    # 4. No candidates
    # ---------------------------------------------------------

    if not candidates:
        return {
            "predicted_origin": "unknown",
            "confidence": 0.0,
            "ranking": [],
            "propagation_path": [],
            "method": "origin_impact_separation_v6_no_drift_guard",
        }

    # ---------------------------------------------------------
    # 5. Meaningful evidence
    # ---------------------------------------------------------

    meaningful = [
        c
        for c in candidates
        if c["local_evidence"] >= MEANINGFUL_THRESHOLD
    ]

    # If nothing reaches the meaningful threshold,
    # do not force a diagnosis.
    if not meaningful:
        return {
            "predicted_origin": "unknown",
            "confidence": 0.0,
            "ranking": candidates,
            "propagation_path": [],
            "method": "origin_impact_separation_v6_no_drift_guard",
        }

    # ---------------------------------------------------------
    # 6. Select origin
    # ---------------------------------------------------------

    observed_meaningful = [
        c
        for c in meaningful
        if c["onset_observed"]
    ]

    if observed_meaningful:

        # Temporal precedence is the strongest evidence
        # for deciding where propagation started.
        earliest_onset = min(
            c["onset"]
            for c in observed_meaningful
        )

        earliest = [
            c
            for c in observed_meaningful
            if c["onset"] == earliest_onset
        ]

        origin = max(
            earliest,
            key=lambda c: c["origin_score"],
        )

    else:

        # No onset information.
        # Use origin score rather than raw impact.
        origin = max(
            meaningful,
            key=lambda c: c["origin_score"],
        )

    # ---------------------------------------------------------
    # 7. Ranking
    # ---------------------------------------------------------

    ranking = sorted(
        candidates,
        key=lambda c: c["origin_score"],
        reverse=True,
    )

    # ---------------------------------------------------------
    # 8. Propagation path
    # ---------------------------------------------------------

    observed = [
        c
        for c in candidates
        if c["onset_observed"]
    ]

    propagation_path = [
        c["stage"]
        for c in sorted(
            observed,
            key=lambda c: c["onset"],
        )
    ]

    # If no temporal path exists, infer only from
    # meaningful evidence following the causal order.
    if not propagation_path:

        origin_index = order.index(
            origin["stage"]
        )

        propagation_path = [
            stage
            for stage in order[origin_index:]
            if clean_evidence.get(stage, 0.0)
            >= MEANINGFUL_THRESHOLD
        ]

    # ---------------------------------------------------------
    # 9. Confidence
    # ---------------------------------------------------------

    confidence = _clip(
        0.50 * origin["local_evidence"]
        + 0.30 * origin["downstream_support"]
        + 0.20 * origin["temporal_precedence"]
    )

    return {
        "predicted_origin": origin["stage"],
        "confidence": round(
            confidence,
            4,
        ),
        "ranking": ranking,
        "propagation_path": propagation_path,
        "method": "origin_impact_separation_v6_no_drift_guard",
    }