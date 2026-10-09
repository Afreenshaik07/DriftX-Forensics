from pathlib import Path

import pandas as pd
import numpy as np
import joblib

from scipy.stats import ks_2samp
from sklearn.metrics import mean_absolute_error

from src.driftx.evaluation.scenario_registry import get_scenarios
from src.driftx.evaluation.causal_forensics import rank_causal_origins
from src.driftx.evaluation.temporal_baseline import (
    temporal_magnitude_attribution
)


ROOT = Path(__file__).resolve().parents[1]

FEATURES = (
    ["cycle", "setting_1", "setting_2", "setting_3"]
    + [f"sensor_{i}" for i in range(1, 22)]
)

SENSORS = [f"sensor_{i}" for i in range(1, 22)]

STAGES = [
    "feature",
    "prediction",
    "residual",
    "performance"
]


def ks_statistic(a, b):
    return float(
        ks_2samp(
            a,
            b,
            method="asymp"
        ).statistic
    )


def stage_onset(
    reference,
    stream,
    window=200,
    step=50,
    threshold=0.20,
    persistence=2
):
    """
    Detect the first persistent local distribution transition.

    The known benchmark change point is NOT used by the detector.
    """

    reference = np.asarray(
        reference,
        dtype=float
    )

    stream = np.asarray(
        stream,
        dtype=float
    )

    if (
        len(reference) < window
        or len(stream) < (persistence + 1) * window
    ):
        return None

    baseline_window = reference[-window:]

    candidates = []

    for start in range(
        0,
        len(stream)
        - (persistence + 1) * window
        + 1,
        step
    ):

        previous_window = stream[
            start:start + window
        ]

        baseline_ks = ks_statistic(
            baseline_window,
            previous_window
        )

        if baseline_ks >= threshold:
            continue

        transition_scores = []

        for offset in range(persistence):

            left_start = (
                start
                + offset * window
            )

            right_start = (
                left_start
                + window
            )

            left_window = stream[
                left_start:left_start + window
            ]

            right_window = stream[
                right_start:right_start + window
            ]

            transition_scores.append(
                ks_statistic(
                    left_window,
                    right_window
                )
            )

        if all(
            score >= threshold
            for score in transition_scores
        ):
            candidates.append(
                (
                    start + window,
                    float(
                        np.mean(
                            transition_scores
                        )
                    )
                )
            )

    if not candidates:
        return None

    candidates.sort(
        key=lambda item: (
            item[0],
            -item[1]
        )
    )

    return candidates[0][0]


def build_onsets(
    data,
    cp,
    reference,
    current,
    reference_pred,
    current_pred
):
    """
    Build independently detected stage onsets.

    The known change point is NOT passed to the detector.

    Prediction onset uses a reference-vs-current sliding-window
    detector because prediction drift is a direct distribution shift.
    """

    reference_residual = (
        reference["RUL"].to_numpy()
        - reference_pred
    )

    current_residual = (
        current["RUL"].to_numpy()
        - current_pred
    )

    onset_candidates = {}

    # ---------------------------------------------------------
    # FEATURE STAGE
    # ---------------------------------------------------------

    feature_onsets = []

    for sensor in SENSORS:

        onset = stage_onset(
            reference[sensor].to_numpy(),
            current[sensor].to_numpy()
        )

        if onset is not None:
            feature_onsets.append(onset)

    if feature_onsets:
        onset_candidates["feature"] = min(
            feature_onsets
        )

    # ---------------------------------------------------------
    # PREDICTION STAGE
    # ---------------------------------------------------------

    prediction_window = 200
    prediction_step = 50
    prediction_threshold = 0.20
    prediction_persistence = 2

    reference_prediction_window = (
        reference_pred[-prediction_window:]
    )

    prediction_candidates = []

    for start in range(
        0,
        len(current_pred)
        - (
            prediction_persistence
            * prediction_window
        )
        + 1,
        prediction_step
    ):

        persistent_scores = []

        for offset in range(
            prediction_persistence
        ):

            window_start = (
                start
                + offset * prediction_window
            )

            window_end = (
                window_start
                + prediction_window
            )

            window = current_pred[
                window_start:window_end
            ]

            if len(window) < prediction_window:
                continue

            score = ks_statistic(
                reference_prediction_window,
                window
            )

            persistent_scores.append(score)

        if (
            len(persistent_scores)
            == prediction_persistence
            and all(
                score >= prediction_threshold
                for score in persistent_scores
            )
        ):
            prediction_candidates.append(
                (
                    start,
                    float(
                        np.mean(
                            persistent_scores
                        )
                    )
                )
            )

    if prediction_candidates:

        prediction_candidates.sort(
            key=lambda item: (
                item[0],
                -item[1]
            )
        )

        onset_candidates["prediction"] = (
            prediction_candidates[0][0]
        )

    # ---------------------------------------------------------
    # RESIDUAL STAGE
    # ---------------------------------------------------------

    residual_onset = stage_onset(
        reference_residual,
        current_residual
    )

    if residual_onset is not None:
        onset_candidates["residual"] = (
            residual_onset
        )

    # ---------------------------------------------------------
    # PERFORMANCE STAGE
    # ---------------------------------------------------------

    reference_error = np.abs(
        reference_residual
    )

    current_error = np.abs(
        current_residual
    )

    performance_onset = stage_onset(
        reference_error,
        current_error
    )

    if performance_onset is not None:
        onset_candidates["performance"] = (
            performance_onset
        )

    return onset_candidates


def build_evidence(
    feature_ks,
    prediction_ks,
    residual_ks,
    performance_score
):
    return {
        "feature": float(feature_ks),
        "prediction": float(prediction_ks),
        "residual": float(residual_ks),
        "performance": float(performance_score)
    }


def normalize_onsets(onsets):

    if not onsets:
        return {}

    earliest = min(
        onsets.values()
    )

    return {
        stage: max(
            0,
            int(onset - earliest)
        )
        for stage, onset in onsets.items()
    }


def infer_propagation(
    evidence,
    onset_lags
):

    ordered = sorted(
        [
            (
                stage,
                score,
                onset_lags.get(
                    stage,
                    10**9
                )
            )
            for stage, score in evidence.items()
            if score >= 0.20
        ],
        key=lambda item: item[2]
    )

    return [
        stage
        for stage, _, _ in ordered
    ]


def main():

    model = joblib.load(
        ROOT
        / "artifacts"
        / "baseline_rul_model.joblib"
    )

    results = []

    print(
        "=== DRIFTX-FORENSICS: "
        "INTEGRATED FORENSIC BENCHMARK ==="
    )

    for scenario in get_scenarios():

        scenario_id = scenario.scenario_id

        path = (
            ROOT
            / "data"
            / "interim"
            / f"{scenario_id}.csv"
        )

        if not path.exists():

            print(
                f"{scenario_id}: FILE NOT FOUND"
            )

            continue

        data = pd.read_csv(path)

        cp_value = data[
            "change_point"
        ].iloc[0]

        cp = (
            len(data) // 2
            if pd.isna(cp_value)
            else int(cp_value)
        )

        reference = data.iloc[:cp].copy()
        current = data.iloc[cp:].copy()

        # -----------------------------------------------------
        # PREDICTIONS
        # -----------------------------------------------------

        if "model_prediction" in data.columns:

            reference_pred = (
                reference[
                    "model_prediction"
                ].to_numpy()
            )

            current_pred = (
                current[
                    "model_prediction"
                ].to_numpy()
            )

        else:

            reference_pred = model.predict(
                reference[FEATURES]
            )

            current_pred = model.predict(
                current[FEATURES]
            )

        # -----------------------------------------------------
        # RESIDUALS
        # -----------------------------------------------------

        reference_residual = (
            reference["RUL"].to_numpy()
            - reference_pred
        )

        current_residual = (
            current["RUL"].to_numpy()
            - current_pred
        )

        # -----------------------------------------------------
        # FEATURE DRIFT
        # -----------------------------------------------------

        sensor_ks = {
            sensor: ks_statistic(
                reference[sensor],
                current[sensor]
            )
            for sensor in SENSORS
        }

        feature_ks = max(
            sensor_ks.values()
        )

        top_sensor_drift = sorted(
            sensor_ks.items(),
            key=lambda item: item[1],
            reverse=True
        )[:5]

        # -----------------------------------------------------
        # PREDICTION DRIFT
        # -----------------------------------------------------

        prediction_ks = ks_statistic(
            reference_pred,
            current_pred
        )

        # -----------------------------------------------------
        # RESIDUAL DRIFT
        # -----------------------------------------------------

        residual_ks = ks_statistic(
            reference_residual,
            current_residual
        )

        # -----------------------------------------------------
        # PERFORMANCE IMPACT
        # -----------------------------------------------------

        reference_mae = mean_absolute_error(
            reference["RUL"],
            reference_pred
        )

        current_mae = mean_absolute_error(
            current["RUL"],
            current_pred
        )

        mae_change = (
            current_mae - reference_mae
        ) / max(
            reference_mae,
            1e-12
        )

        performance_score = float(
            max(
                0.0,
                min(
                    1.0,
                    mae_change
                )
            )
        )

        # -----------------------------------------------------
        # EVIDENCE
        # -----------------------------------------------------

        evidence = build_evidence(
            feature_ks,
            prediction_ks,
            residual_ks,
            performance_score
        )

        # -----------------------------------------------------
        # ONSETS
        # -----------------------------------------------------

        raw_onsets = build_onsets(
            data=data,
            cp=cp,
            reference=reference,
            current=current,
            reference_pred=reference_pred,
            current_pred=current_pred
        )

        onset_lags = normalize_onsets(
            raw_onsets
        )

        # -----------------------------------------------------
        # IMPACT
        # -----------------------------------------------------

        impact = {
            "feature": 0.0,
            "prediction": 0.0,
            "residual": min(
                1.0,
                residual_ks
            ),
            "performance": performance_score
        }

        # -----------------------------------------------------
        # FORENSIC DIAGNOSIS
        # -----------------------------------------------------

        forensic = rank_causal_origins(
            evidence=evidence,
            onset_lags=onset_lags,
            impact=impact
        )

        propagation_path = (
            infer_propagation(
                evidence,
                onset_lags
            )
        )

        # -----------------------------------------------------
        # TEMPORAL BASELINE
        # -----------------------------------------------------

        baseline = (
            temporal_magnitude_attribution(
                drift_scores=evidence,
                onset_lags=onset_lags
            )
        )

        # -----------------------------------------------------
        # RESULTS
        # -----------------------------------------------------

        results.append({

            "scenario":
                scenario_id,

            "true_cause":
                scenario.cause,

            "predicted_origin":
                forensic[
                    "predicted_origin"
                ],

            "first_stage_correct":
                forensic[
                    "predicted_origin"
                ]
                == scenario.first_observable_stage,

            "true_first_stage":
                scenario.first_observable_stage,

            "predicted_first_stage":
                forensic[
                    "predicted_origin"
                ],

            "forensic_confidence":
                forensic[
                    "confidence"
                ],

            "forensic_path":
                "->".join(
                    propagation_path
                ),

            "feature_onset":
                raw_onsets.get(
                    "feature"
                ),

            "prediction_onset":
                raw_onsets.get(
                    "prediction"
                ),

            "residual_onset":
                raw_onsets.get(
                    "residual"
                ),

            "performance_onset":
                raw_onsets.get(
                    "performance"
                ),

            "baseline_origin":
                baseline[
                    "predicted_origin"
                ],

            "baseline_correct":
                baseline[
                    "predicted_origin"
                ]
                == scenario.first_observable_stage,

            "feature_ks":
                feature_ks,

            "top_sensor_drift":
                ";".join(
                    f"{sensor}:{score:.4f}"
                    for sensor, score
                    in top_sensor_drift
                ),

            "prediction_ks":
                prediction_ks,

            "residual_ks":
                residual_ks,

            "performance_score":
                performance_score,

            "reference_mae":
                reference_mae,

            "current_mae":
                current_mae,

            "mae_change_pct":
                mae_change * 100
        })

        print(
            f"{scenario_id:28s} "
            f"true={scenario.cause:18s} "
            f"forensic="
            f"{forensic['predicted_origin']:12s} "
            f"baseline="
            f"{baseline['predicted_origin']:12s}"
        )

    # ---------------------------------------------------------
    # SAVE RESULTS
    # ---------------------------------------------------------

    result_df = pd.DataFrame(
        results
    )

    output = (
        ROOT
        / "results"
        / "forensic_full_benchmark.csv"
    )

    result_df.to_csv(
        output,
        index=False
    )

    # ---------------------------------------------------------
    # AGGREGATE
    # ---------------------------------------------------------

    print(
        "\n=== AGGREGATE RESULTS ==="
    )

    print(
        f"Scenarios evaluated       : "
        f"{len(result_df)}"
    )

    print(
        f"Forensic first-stage acc. : "
        f"{result_df['first_stage_correct'].mean() * 100:.1f}%"
    )

    print(
        f"Temporal baseline acc.    : "
        f"{result_df['baseline_correct'].mean() * 100:.1f}%"
    )

    print(
        f"Mean forensic confidence  : "
        f"{result_df['forensic_confidence'].mean():.3f}"
    )

    print(
        f"Saved                     : "
        f"{output}"
    )


if __name__ == "__main__":
    main()