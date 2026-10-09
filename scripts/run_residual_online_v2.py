from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.driftx.config import CONFIG
from src.driftx.models.rul_model import RULModel
from src.driftx.models.residual_adapter import ResidualIncrementalAdapter
from src.driftx.evaluation.safety_gate import evaluate_safety_gate


FEATURES = (
    ["cycle", "setting_1", "setting_2", "setting_3"]
    + [f"sensor_{i}" for i in range(1, 22)]
)

DATASET = (
    Path(CONFIG["paths"]["processed_data"])
    / "FD001_temporal_sensor2_drift.csv"
)

CHAMPION_PATH = (
    Path(CONFIG["paths"]["artifacts"])
    / "baseline_rul_model.joblib"
)

ADAPTER_PATH = (
    Path(CONFIG["paths"]["artifacts"])
    / "residual_online_adapter.joblib"
)

RESULT_PATH = (
    Path(CONFIG["paths"]["results"])
    / "residual_online_update_v2.csv"
)


def metrics(y_true, predictions):
    return {
        "mae": float(mean_absolute_error(y_true, predictions)),
        "rmse": float(mean_squared_error(y_true, predictions) ** 0.5),
        "r2": float(r2_score(y_true, predictions)),
    }


def main():

    print(
        "=== DRIFTX-FORENSICS: "
        "ONLINE RESIDUAL ADAPTATION V2 ==="
    )

    df = pd.read_csv(DATASET)

    units = sorted(df["unit"].unique())

    split_index = int(len(units) * 0.80)

    train_units = units[:split_index]
    test_units = units[split_index:]

    train_df = df[df["unit"].isin(train_units)].copy()
    test_df = df[df["unit"].isin(test_units)].copy()

    train_drifted = train_df[
        train_df["drift_active"] == 1
    ].copy()

    test_stable = test_df[
        test_df["drift_active"] == 0
    ].copy()

    test_drifted = test_df[
        test_df["drift_active"] == 1
    ].copy()

    print(f"Training engines : {len(train_units)}")
    print(f"Test engines     : {len(test_units)}")
    print(f"Training drift rows : {len(train_drifted)}")
    print(f"Test stable rows   : {len(test_stable)}")
    print(f"Test drift rows    : {len(test_drifted)}")

    # ---------------------------------------------------------
    # CHAMPION
    # ---------------------------------------------------------

    champion = RULModel()
    champion.load(CHAMPION_PATH)

    champion_stable_pred = champion.predict(
        test_stable[FEATURES]
    )

    champion_drift_pred = champion.predict(
        test_drifted[FEATURES]
    )

    champion_stable = metrics(
        test_stable["RUL"],
        champion_stable_pred,
    )

    champion_drifted = metrics(
        test_drifted["RUL"],
        champion_drift_pred,
    )

    print()
    print("=== CHAMPION ===")
    print(
        f"Stable MAE  : {champion_stable['mae']:.4f}"
    )
    print(
        f"Drift MAE   : {champion_drifted['mae']:.4f}"
    )

    # ---------------------------------------------------------
    # ZERO-INITIALIZED ADAPTER
    # ---------------------------------------------------------

    adapter = ResidualIncrementalAdapter()

    # We need to initialize the scaler/model without learning
    # a stable-regime correction.
    #
    # The first drift batch establishes the adapter itself.
    # This represents adaptation beginning only after drift
    # observations become available.

    X_update = train_drifted[
        FEATURES
    ].reset_index(drop=True)

    champion_update_pred = champion.predict(
        X_update
    )

    update_residuals = (
        train_drifted["RUL"].to_numpy()
        - champion_update_pred
    )

    batch_size = 512

    batches = 0

    print()
    print(
        "=== ONLINE DRIFT ADAPTATION ==="
    )

    # First batch initializes the adapter.
    first_end = min(
        batch_size,
        len(X_update),
    )

    adapter.fit(
        X_update.iloc[:first_end],
        update_residuals[:first_end],
    )

    batches += 1

    # Remaining batches are genuine partial_fit updates.
    for start in range(
        first_end,
        len(X_update),
        batch_size,
    ):

        end = min(
            start + batch_size,
            len(X_update),
        )

        adapter.partial_fit(
            X_update.iloc[start:end],
            update_residuals[start:end],
        )

        batches += 1

    adapter.save(
        ADAPTER_PATH
    )

    # ---------------------------------------------------------
    # ADAPTED CHALLENGER
    # ---------------------------------------------------------

    correction = adapter.predict_correction(
        test_drifted[FEATURES]
    )

    adapted_predictions = (
        champion_drift_pred
        + correction
    )

    adapted_metrics = metrics(
        test_drifted["RUL"],
        adapted_predictions,
    )

    print()
    print(
        "=== ADAPTED CHALLENGER ==="
    )

    print(
        f"MAE  : {adapted_metrics['mae']:.4f}"
    )

    print(
        f"RMSE : {adapted_metrics['rmse']:.4f}"
    )

    print(
        f"R2   : {adapted_metrics['r2']:.4f}"
    )

    print(
        f"Update batches : {batches}"
    )

    # ---------------------------------------------------------
    # SAFETY GATE
    # ---------------------------------------------------------

    print()
    print("=== SAFETY GATE ===")

    gate = evaluate_safety_gate(
        champion_metrics=champion_drifted,
        challenger_metrics=adapted_metrics,
        min_improvement=0.0,
        max_r2_drop=0.02,
        max_rmse_increase=0.0,
    )

    print(
        f"Decision : {gate.decision}"
    )

    print(
        f"MAE improvement : "
        f"{gate.improvement:.4f}"
    )

    print(
        f"Improvement ratio : "
        f"{gate.improvement_ratio:.4%}"
    )

    print(
        f"Reason : {gate.reason}"
    )

    # ---------------------------------------------------------
    # SAVE EXPERIMENT
    # ---------------------------------------------------------

    result = {
        "experiment":
            "online_residual_adaptation_v2",

        "drift_type":
            "FEATURE_DRIFT",

        "feature":
            "sensor_2",

        "magnitude":
            0.30,

        "champion_stable_mae":
            champion_stable["mae"],

        "champion_drifted_mae":
            champion_drifted["mae"],

        "champion_drifted_rmse":
            champion_drifted["rmse"],

        "champion_drifted_r2":
            champion_drifted["r2"],

        "adapted_mae":
            adapted_metrics["mae"],

        "adapted_rmse":
            adapted_metrics["rmse"],

        "adapted_r2":
            adapted_metrics["r2"],

        "mae_improvement":
            gate.improvement,

        "improvement_ratio":
            gate.improvement_ratio,

        "update_batches":
            batches,

        "safety_decision":
            gate.decision,

        "reason":
            gate.reason,
    }

    RESULT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.DataFrame(
        [result]
    ).to_csv(
        RESULT_PATH,
        index=False,
    )

    print()
    print("=== FINAL ===")
    print(
        f"Champion MAE : "
        f"{champion_drifted['mae']:.4f}"
    )
    print(
        f"Adapted MAE  : "
        f"{adapted_metrics['mae']:.4f}"
    )
    print(
        f"Safety       : "
        f"{gate.decision}"
    )
    print(
        f"Saved        : "
        f"{RESULT_PATH.resolve()}"
    )


if __name__ == "__main__":
    main()
