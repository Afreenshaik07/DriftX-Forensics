from scipy.stats import ks_2samp
import numpy as np


def detect_stage_onset(
    reference,
    stream,
    window_size=200,
    step_size=100,
    alpha=0.01,
    min_consecutive=2,
):
    stream = np.asarray(stream, dtype=float)

    if len(stream) < 3 * window_size:
        return {
            "detected": False,
            "onset": None,
            "statistic": 0.0,
            "p_value": 1.0,
        }

    candidates = []

    for start in range(
        window_size,
        len(stream) - 2 * window_size + 1,
        step_size,
    ):
        previous_window = stream[start - window_size:start]
        current_window = stream[start:start + window_size]
        following_window = stream[
            start + window_size:start + 2 * window_size
        ]

        ks1, p1 = ks_2samp(previous_window, current_window)
        ks2, p2 = ks_2samp(previous_window, following_window)

        if p1 < alpha and p2 < alpha:
            candidates.append({
                "onset": start,
                "statistic": float((ks1 + ks2) / 2.0),
                "p_value": float(max(p1, p2)),
            })

    if not candidates:
        return {
            "detected": False,
            "onset": None,
            "statistic": 0.0,
            "p_value": 1.0,
        }

    # Earliest persistent change.
    # A later candidate is preferred only when the earlier one
    # is substantially weaker, preventing isolated false positives.
    strongest = max(
        candidates,
        key=lambda x: x["statistic"]
    )

    threshold = strongest["statistic"] * 0.75

    qualified = [
        candidate
        for candidate in candidates
        if candidate["statistic"] >= threshold
    ]

    best = min(
        qualified,
        key=lambda x: x["onset"]
    )

    return {
        "detected": True,
        "onset": int(best["onset"]),
        "statistic": float(best["statistic"]),
        "p_value": float(best["p_value"]),
    }


def detect_all_stage_onsets(
    stage_streams,
    references=None,
    window_size=200,
    step_size=100,
    alpha=0.01,
    min_consecutive=2,
):
    results = {}

    for stage, stream in stage_streams.items():
        reference = (
            references[stage]
            if references is not None and stage in references
            else None
        )

        results[stage] = detect_stage_onset(
            reference,
            stream,
            window_size,
            step_size,
            alpha,
            min_consecutive,
        )

    return results
