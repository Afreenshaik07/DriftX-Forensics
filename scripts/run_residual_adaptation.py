from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.driftx.config import CONFIG
from src.driftx.models.rul_model import RULModel
from src.driftx.models.model_registry import ModelRegistry
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
    / "residual_incremental_adapter.joblib"
)

RESULT_PATH = (
    Path(CONFIG["paths"]["results"])
    / "residual_incremental_experiment.csv"
)


def calculate_metrics(y_true, predictions):

    return {
        "mae": float(
            mean_absolute_error(
                y_true,
                predictions,
            )
        ),
        "rmse": float(
            mean_squared_error(
                y_true,
                predictions,
            ) ** 0.5
        ),
        "r2": float(
            r2_score(
                y_true,
                predictions,
            )
        ),
    }


def main():

    print(
        "=== DRIFTX-FORENSICS: "
        "RESIDUAL INCREMENTAL ADAPTATION ==="
    )

    if not DATASET.exists():
        raise FileNotFoundError(
            f"Temporal drift dataset not found: {DATASET}"
        )

    df = pd.read_csv(DATASET)

    units = sorted(
        df["unit"].unique()
    )

    split_index = int(
        len(units) * 0.80
    )

    train_units = units[:split_index]
    test_units = units[split_index:]

    train_df = df[
        df["unit"].isin(train_units)
    ].copy()

    test_df = df[
        df["unit"].isin(test_units)
    ].copy()

    print(
        f"Dataset rows       : {len(df)}"
    )

    print(
        f"Training engines    : {len(train_units)}"
    )

    print(
        f"Test engines        : {len(test_units)}"
    )

    # ---------------------------------------------------------
    # TRAINING REGIME
    # ---------------------------------------------------------

    train_stable = train_df[
        train_df["drift_active"] == 0
    ].copy()

    train_drifted = train_df[
        train_df["drift_active"] == 1
    ].copy()

    # ---------------------------------------------------------
    # TEST REGIMES
    # ---------------------------------------------------------

    test_stable = test_df[
        test_df["drift_active"] == 0
    ].copy()

    test_drifted = test_df[
        test_df["drift_active"] == 1
    ].copy()

    print()
    print("=== TEMPORAL REGIMES ===")

    print(
        f"Training stable rows : "
        f"{len(train_stable)}"
    )

    print(
        f"Training drift rows  : "
        f"{len(train_drifted)}"
    )

    print(
        f"Test stable rows     : "
        f"{len(test_stable)}"
    )

    print(
        f"Test drift rows      : "
        f"{len(test_drifted)}"
    )

    # ---------------------------------------------------------
    # CHAMPION
    # ---------------------------------------------------------

    champion = RULModel()

    champion.load(
        CHAMPION_PATH
    )

    stable_predictions = champion.predict(
        test_stable[FEATURES]
    )

    drifted_predictions = champion.predict(
        test_drifted[FEATURES]
    )

    champion_stable = calculate_metrics(
        test_stable["RUL"],
        stable_predictions,
    )

    champion_drifted = calculate_metrics(
        test_drifted["RUL"],
        drifted_predictions,
    )

    print()
    print("=== CHAMPION: STABLE REGIME ===")

    print(
        f"MAE  : {champion_stable['mae']:.4f}"
    )

    print(
        f"RMSE : {champion_stable['rmse']:.4f}"
    )

    print(
        f"R2   : {champion_stable['r2']:.4f}"
    )

    print()
    print("=== CHAMPION: DRIFTED REGIME ===")

    print(
        f"MAE  : {champion_drifted['mae']:.4f}"
    )

    print(
        f"RMSE : {champion_drifted['rmse']:.4f}"
    )

    print(
        f"R2   : {champion_drifted['r2']:.4f}"
    )

    # ---------------------------------------------------------
    # RESIDUAL ADAPTER
    # ---------------------------------------------------------

    adapter = ResidualIncrementalAdapter()

    # Initial residuals from the stable regime.
    stable_train_predictions = champion.predict(
        train_stable[FEATURES]
    )

    stable_residuals = (
        train_stable["RUL"].to_numpy()
        - stable_train_predictions
    )

    print()
    print("=== TRAINING RESIDUAL ADAPTER ===")

    adapter.fit(
        train_stable[FEATURES],
        stable_residuals,
    )

    # ---------------------------------------------------------
    # BEFORE ADAPTATION
    # ---------------------------------------------------------

    drift_predictions_before = champion.predict(
        test_drifted[FEATURES]
    )

    correction_before = adapter.predict_correction(
        test_drifted[FEATURES]
    )

    adapted_before = (
        drift_predictions_before
        + correction_before
    )

    before_update = calculate_metrics(
        test_drifted["RUL"],
        adapted_before,
    )

    print()
    print("=== BEFORE DRIFT UPDATE ===")

    print(
        f"MAE  : {before_update['mae']:.4f}"
    )

    print(
        f"RMSE : {before_update['rmse']:.4f}"
    )

    print(
        f"R2   : {before_update['r2']:.4f}"
    )

    # ---------------------------------------------------------
    # ONLINE ADAPTATION
    # ---------------------------------------------------------

    drift_predictions_train = champion.predict(
        train_drifted[FEATURES]
    )

    drift_residuals = (
        train_drifted["RUL"].to_numpy()
        - drift_predictions_train
    )

    print()
    print(
        "=== APPLYING RESIDUAL partial_fit() ==="
    )

    # Sequential batches simulate streaming observations.
    batch_size = 512

    X_update = train_drifted[
        FEATURES
    ].reset_index(drop=True)

    y_update = pd.Series(
        drift_residuals
    ).reset_index(drop=True)

    batches = 0

    for start in range(
        0,
        len(X_update),
        batch_size,
    ):

        end = min(
            start + batch_size,
            len(X_update),
        )

        adapter.partial_fit(
            X_update.iloc[start:end],
            y_update.iloc[start:end].to_numpy(),
        )

        batches += 1

    adapter.save(
        ADAPTER_PATH
    )

    # ---------------------------------------------------------
    # AFTER ADAPTATION
    # ---------------------------------------------------------

    drift_predictions_after = champion.predict(
        test_drifted[FEATURES]
    )

    correction_after = adapter.predict_correction(
        test_drifted[FEATURES]
    )

    adapted_after = (
        drift_predictions_after
        + correction_after
    )

    after_update = calculate_metrics(
        test_drifted["RUL"],
        adapted_after,
    )

    print()
    print("=== AFTER RESIDUAL UPDATE ===")

    print(
        f"MAE  : {after_update['mae']:.4f}"
    )

    print(
        f"RMSE : {after_update['rmse']:.4f}"
    )

    print(
        f"R2   : {after_update['r2']:.4f}"
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
        challenger_metrics=after_update,
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
    # REGISTRY
    # ---------------------------------------------------------

    registry_action = "KEEP_CHAMPION"

    if gate.decision == "PROMOTE":

        print()
        print(
            "NOTE: Residual adapter is stored "
            "separately from the champion."
        )

        registry_action = (
            "PROMOTION_PENDING_ADAPTER_WRAPPER"
        )

    print()
    print(
        f"Registry action : "
        f"{registry_action}"
    )

    # ---------------------------------------------------------
    # SAVE RESULTS
    # ---------------------------------------------------------

    result = {
        "experiment":
            "residual_incremental_adaptation",

        "drift_type":
            "FEATURE_DRIFT",

        "affected_feature":
            "sensor_2",

        "drift_magnitude":
            0.30,

        "champion_mae":
            champion_drifted["mae"],

        "champion_rmse":
            champion_drifted["rmse"],

        "champion_r2":
            champion_drifted["r2"],

        "before_update_mae":
            before_update["mae"],

        "before_update_rmse":
            before_update["rmse"],

        "before_update_r2":
            before_update["r2"],

        "after_update_mae":
            after_update["mae"],

        "after_update_rmse":
            after_update["rmse"],

        "after_update_r2":
            after_update["r2"],

        "update_improvement":
            before_update["mae"]
            - after_update["mae"],

        "champion_to_adapter_improvement":
            gate.improvement,

        "improvement_ratio":
            gate.improvement_ratio,

        "safety_decision":
            gate.decision,

        "registry_action":
            registry_action,

        "update_batches":
            batches,

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
    print("=== FINAL SUMMARY ===")

    print(
        f"Champion MAE      : "
        f"{champion_drifted['mae']:.4f}"
    )

    print(
        f"Before update MAE : "
        f"{before_update['mae']:.4f}"
    )

    print(
        f"After update MAE  : "
        f"{after_update['mae']:.4f}"
    )

    print(
        f"Adapter gain      : "
        f"{result['update_improvement']:.4f}"
    )

    print(
        f"Safety decision   : "
        f"{gate.decision}"
    )

    print(
        f"Saved             : "
        f"{RESULT_PATH.resolve()}"
    )

    print()
    print(
        "=== RESIDUAL ADAPTATION COMPLETE ==="
    )


if __name__ == "__main__":
    main()
