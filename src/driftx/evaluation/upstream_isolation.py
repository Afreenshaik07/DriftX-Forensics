from typing import Dict, List

STAGE_ORDER = ["feature", "prediction", "residual", "performance"]

def calculate_upstream_isolation(
    evidence: Dict[str, float],
    onset_lags: Dict[str, int],
    stage_order: List[str] = None
):
    order = stage_order or STAGE_ORDER
    results = {}

    for stage in order:
        local = float(evidence.get(stage, 0.0))
        onset = onset_lags.get(stage)

        if onset is None or local <= 0:
            results[stage] = {
                "local_evidence": 0.0,
                "upstream_stability": 0.0,
                "downstream_support": 0.0,
                "isolation_score": 0.0
            }
            continue

        upstream = order[:order.index(stage)]
        downstream = order[order.index(stage) + 1:]

        upstream_values = [
            float(evidence.get(s, 0.0))
            for s in upstream
            if onset_lags.get(s) is not None
            and onset_lags[s] <= onset
        ]

        downstream_values = [
            float(evidence.get(s, 0.0))
            for s in downstream
            if onset_lags.get(s) is not None
            and onset_lags[s] >= onset
        ]

        upstream_stability = (
            1.0 - max(upstream_values)
            if upstream_values
            else 1.0
        )

        downstream_support = (
            sum(downstream_values) / len(downstream_values)
            if downstream_values
            else 0.0
        )

        isolation = (
            0.50 * local
            + 0.30 * upstream_stability
            + 0.20 * downstream_support
        )

        results[stage] = {
            "local_evidence": round(local, 4),
            "upstream_stability": round(upstream_stability, 4),
            "downstream_support": round(downstream_support, 4),
            "isolation_score": round(
                max(0.0, min(1.0, isolation)), 4
            )
        }

    ranking = sorted(
        results.items(),
        key=lambda item: item[1]["isolation_score"],
        reverse=True
    )

    return {
        "ranking": [
            {"stage": stage, **details}
            for stage, details in ranking
        ],
        "predicted_origin": (
            ranking[0][0] if ranking else "unknown"
        )
    }
