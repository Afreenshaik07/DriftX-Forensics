from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp

from src.driftx.config import CONFIG
from src.driftx.data.preprocessing import load_training_data, create_rul_target
from src.driftx.models.rul_model import RULModel
from src.driftx.evaluation.feature_scanner import scan_feature_drift
from src.driftx.evaluation.prediction_drift import detect_prediction_drift
from src.driftx.evaluation.residual_drift import detect_residual_drift
from src.driftx.evaluation.performance_monitor import evaluate_performance
from src.driftx.evaluation.causal_forensics import rank_causal_origins
from src.driftx.evaluation.experiment_result import ExperimentResult

ROOT = Path(__file__).resolve().parents[1]


def ks_statistic(a, b):
    return float(ks_2samp(a, b, method="asymp").statistic)


def detect_sequential_onset(reference, stream, window=200, step=50, threshold=0.20, persistence=2):
    reference = np.asarray(reference, dtype=float)
    stream = np.asarray(stream, dtype=float)

    if len(reference) < window or len(stream) < (persistence + 1) * window:
        return None

    baseline = reference[-window:]
    candidates = []

    for start in range(
        0,
        len(stream) - (persistence + 1) * window + 1,
        step
    ):
        transition_scores = []

        for offset in range(persistence):
            left_start = start + offset * window
            right_start = left_start + window

            left = stream[left_start:left_start + window]
            right = stream[right_start:right_start + window]

            transition_scores.append(
                ks_statistic(left, right)
            )

        baseline_score = ks_statistic(
            baseline,
            stream[start:start + window]
        )

        if baseline_score < threshold and all(
            score >= threshold for score in transition_scores
        ):
            candidates.append(
                (
                    start + window,
                    float(np.mean(transition_scores))
                )
            )

    if not candidates:
        return None

    return min(candidates, key=lambda x: x[0])[0]


def detect_stage_onsets(
    reference_df,
    stream_df,
    reference_predictions,
    stream_predictions,
    reference_residuals,
    stream_residuals,
    y_stream,
    features
):
    onset_candidates = []

    feature_onsets = []

    for feature in features:
        onset = detect_sequential_onset(
            reference_df[feature].to_numpy(),
            stream_df[feature].to_numpy()
        )

        if onset is not None:
            feature_onsets.append(onset)

    if feature_onsets:
        onset_candidates.append(("feature", min(feature_onsets)))

    prediction_onset = detect_sequential_onset(
        reference_predictions,
        stream_predictions
    )

    if prediction_onset is not None:
        onset_candidates.append(("prediction", prediction_onset))

    residual_onset = detect_sequential_onset(
        reference_residuals,
        stream_residuals
    )

    if residual_onset is not None:
        onset_candidates.append(("residual", residual_onset))

    performance_series = np.abs(
        y_stream.to_numpy() - stream_predictions
    )

    reference_error = np.abs(
        reference_df["RUL"].to_numpy() - reference_predictions
    )

    performance_onset = detect_sequential_onset(
        reference_error,
        performance_series
    )

    if performance_onset is not None:
        onset_candidates.append(("performance", performance_onset))

    return dict(onset_candidates)


def main():
    raw_dir = ROOT / CONFIG["paths"]["raw_data"]

    df = create_rul_target(load_training_data(raw_dir))

    features = [
        c for c in df.columns
        if c.startswith("sensor_")
    ]

    units = sorted(df["unit"].unique())
    split = int(len(units) * 0.8)

    train_units = units[:split]
    test_units = units[split:]

    train_df = df[df["unit"].isin(train_units)].copy()
    test_df = df[df["unit"].isin(test_units)].copy()

    X_train = train_df[features]
    y_train = train_df["RUL"]

    X_test = test_df[features]
    y_test = test_df["RUL"]

    model = RULModel(CONFIG["model"])
    model.fit(X_train, y_train)

    baseline_predictions = model.predict(X_test)
    baseline_residuals = y_test.to_numpy() - baseline_predictions

    midpoint = len(test_df) // 2

    reference = test_df.iloc[:midpoint].copy()
    current = test_df.iloc[midpoint:].copy()

    reference_predictions = baseline_predictions[:midpoint]
    current_predictions = baseline_predictions[midpoint:]

    reference_residuals = baseline_residuals[:midpoint]
    current_residuals = baseline_residuals[midpoint:]

    feature_results = scan_feature_drift(
        reference,
        current,
        features
    )

    prediction_result = detect_prediction_drift(
        reference_predictions,
        current_predictions
    )

    residual_result = detect_residual_drift(
        reference_residuals,
        current_residuals
    )

    reference_metrics = evaluate_performance(
        y_test.iloc[:midpoint],
        reference_predictions
    )

    current_metrics = evaluate_performance(
        y_test.iloc[midpoint:],
        current_predictions
    )

    feature_scores = {
        row["feature"]: float(row["ks_statistic"])
        for _, row in feature_results.iterrows()
        if row["drift_detected"]
    }

    evidence = {
        "feature": min(
            max(max(feature_scores.values(), default=0.0), 0.0),
            1.0
        ),
        "prediction": min(
            max(prediction_result["ks_statistic"], 0.0),
            1.0
        ),
        "residual": min(
            max(residual_result["ks_statistic"], 0.0),
            1.0
        ),
        "performance": min(
            max(
                (
                    current_metrics["mae"] /
                    max(reference_metrics["mae"], 1e-12)
                ) - 1.0,
                0.0
            ),
            1.0
        )
    }

    onset_lags = detect_stage_onsets(
        reference,
        current,
        reference_predictions,
        current_predictions,
        reference_residuals,
        current_residuals,
        y_test.iloc[midpoint:],
        features
    )

    impact = {
        "feature": evidence["feature"],
        "prediction": evidence["prediction"],
        "residual": evidence["residual"],
        "performance": evidence["performance"]
    }

    diagnosis = rank_causal_origins(
        evidence,
        onset_lags,
        impact
    )

    result = ExperimentResult(
        scenario_id="REAL_BASELINE",
        true_origin="unknown",
        predicted_origin=diagnosis["predicted_origin"],
        diagnosis_confidence=diagnosis["confidence"],
        detection_delay=0.0,
        false_alarm=False,
        propagation_path_true=[],
        propagation_path_predicted=diagnosis.get("propagation_path", []),
        metrics={
            "baseline_mae": float(reference_metrics["mae"]),
            "current_mae": float(current_metrics["mae"]),
            "baseline_rmse": float(reference_metrics["rmse"]),
            "current_rmse": float(current_metrics["rmse"]),
            "baseline_r2": float(reference_metrics["r2"]),
            "current_r2": float(current_metrics["r2"])
        }
    )

    output_path = ROOT / CONFIG["paths"]["results"] / "real_baseline_experiment.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([{
        "predicted_origin": diagnosis["predicted_origin"],
        "diagnosis_confidence": diagnosis["confidence"],
        "feature_onset": onset_lags.get("feature"),
        "prediction_onset": onset_lags.get("prediction"),
        "residual_onset": onset_lags.get("residual"),
        "performance_onset": onset_lags.get("performance"),
        "baseline_mae": reference_metrics["mae"],
        "current_mae": current_metrics["mae"],
        "baseline_rmse": reference_metrics["rmse"],
        "current_rmse": current_metrics["rmse"],
        "baseline_r2": reference_metrics["r2"],
        "current_r2": current_metrics["r2"],
        "prediction_drift": prediction_result["drift_detected"],
        "residual_drift": residual_result["drift_detected"]
    }]).to_csv(output_path, index=False)

    print("\n=== DriftX-Forensics Real Experiment ===")
    print(f"Train engines: {len(train_units)}")
    print(f"Test engines:  {len(test_units)}")
    print(f"Predicted origin: {diagnosis['predicted_origin']}")
    print(f"Confidence: {diagnosis['confidence']:.4f}")
    print(f"Observed onsets: {onset_lags}")
    print(f"Reference MAE: {reference_metrics['mae']:.4f}")
    print(f"Current MAE:   {current_metrics['mae']:.4f}")
    print(f"Prediction drift: {prediction_result['drift_detected']}")
    print(f"Residual drift:   {residual_result['drift_detected']}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
