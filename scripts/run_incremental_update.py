from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.driftx.config import CONFIG
from src.driftx.data.preprocessing import load_training_data, create_rul_target
from src.driftx.data.drift_lab import inject_feature_drift
from src.driftx.models.incremental_model import IncrementalRULModel
from src.driftx.models.rul_model import RULModel
from src.driftx.evaluation.safety_gate import evaluate_safety_gate
from src.driftx.models.model_registry import ModelRegistry


FEATURES = (
    ["cycle", "setting_1", "setting_2", "setting_3"]
    + [f"sensor_{i}" for i in range(1, 22)]
)

CHAMPION_PATH = (
    Path(CONFIG["paths"]["artifacts"])
    / "baseline_rul_model.joblib"
)

INCREMENTAL_PATH = (
    Path(CONFIG["paths"]["artifacts"])
    / "incremental_update_model.joblib"
)

OUTPUT_FILE = (
    Path(CONFIG["paths"]["results"])
    / "incremental_update_experiment.csv"
)


def metrics(predictions, actual):
    return {
        "mae": float(mean_absolute_error(actual, predictions)),
        "rmse": float(mean_squared_error(actual, predictions) ** 0.5),
        "r2": float(r2_score(actual, predictions)),
    }


def apply_drift(df, magnitude=0.30):
    df = df.reset_index(drop=True)

    split = len(df) // 2

    stable = df.iloc[:split].copy()
    drifted = df.iloc[split:].copy()

    drifted = inject_feature_drift(
        drifted,
        features=["sensor_2"],
        magnitude=magnitude,
    )

    return stable, drifted


def main():

    print(
        "=== DRIFTX-FORENSICS: "
        "TRUE INCREMENTAL UPDATE EXPERIMENT ==="
    )

    raw_dir = Path(CONFIG["paths"]["raw_data"])

    df = create_rul_target(
        load_training_data(raw_dir)
    )

    units = sorted(df["unit"].unique())

    split = int(len(units) * 0.80)

    train_units = units[:split]
    test_units = units[split:]

    train_df = df[df["unit"].isin(train_units)].copy()
    test_df = df[df["unit"].isin(test_units)].copy()

    print(f"Dataset rows: {len(df)}")
    print(f"Training engines: {len(train_units)}")
    print(f"Test engines: {len(test_units)}")

    # ---------------------------------------------------------
    # Controlled drift
    # ---------------------------------------------------------

    test_stable, test_drifted = apply_drift(
        test_df,
        magnitude=0.30,
    )

    train_stable, train_drifted = apply_drift(
        train_df,
        magnitude=0.30,
    )

    print()
    print("=== CONTROLLED DRIFT ===")
    print("Feature  : sensor_2")
    print("Magnitude: 30%")
    print(f"Stable test rows : {len(test_stable)}")
    print(f"Drifted test rows: {len(test_drifted)}")

    # ---------------------------------------------------------
    # Champion
    # ---------------------------------------------------------

    champion = RULModel()
    champion.load(CHAMPION_PATH)

    champion_stable = metrics(
        champion.predict(test_stable[FEATURES]),
        test_stable["RUL"],
    )

    champion_drifted = metrics(
        champion.predict(test_drifted[FEATURES]),
        test_drifted["RUL"],
    )

    print()
    print("=== CHAMPION: STABLE ===")
    print(f"MAE  : {champion_stable['mae']:.4f}")
    print(f"RMSE : {champion_stable['rmse']:.4f}")
    print(f"R2   : {champion_stable['r2']:.4f}")

    print()
    print("=== CHAMPION: DRIFTED ===")
    print(f"MAE  : {champion_drifted['mae']:.4f}")
    print(f"RMSE : {champion_drifted['rmse']:.4f}")
    print(f"R2   : {champion_drifted['r2']:.4f}")

    # ---------------------------------------------------------
    # Incremental model
    # ---------------------------------------------------------

    model = IncrementalRULModel()

    print()
    print("=== INITIAL INCREMENTAL MODEL TRAINING ===")

    model.fit(
        train_stable[FEATURES],
        train_stable["RUL"],
    )

    before_update = model.evaluate(
        test_drifted[FEATURES],
        test_drifted["RUL"],
    )

    print()
    print("=== BEFORE UPDATE ===")
    print(f"MAE  : {before_update['mae']:.4f}")
    print(f"RMSE : {before_update['rmse']:.4f}")
    print(f"R2   : {before_update['r2']:.4f}")

    # ---------------------------------------------------------
    # TRUE incremental update
    # ---------------------------------------------------------

    print()
    print("=== APPLYING partial_fit() ===")

    model.partial_fit(
        train_drifted[FEATURES],
        train_drifted["RUL"],
    )

    model.save(INCREMENTAL_PATH)

    # ---------------------------------------------------------
    # Evaluate after update
    # ---------------------------------------------------------

    after_update = model.evaluate(
        test_drifted[FEATURES],
        test_drifted["RUL"],
    )

    print()
    print("=== AFTER INCREMENTAL UPDATE ===")
    print(f"MAE  : {after_update['mae']:.4f}")
    print(f"RMSE : {after_update['rmse']:.4f}")
    print(f"R2   : {after_update['r2']:.4f}")

    # ---------------------------------------------------------
    # Safety gate against champion
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

    print(f"Decision : {gate.decision}")
    print(f"MAE improvement : {gate.improvement:.4f}")
    print(f"Improvement ratio : {gate.improvement_ratio:.4%}")
    print(f"Reason : {gate.reason}")

    # ---------------------------------------------------------
    # Registry
    # ---------------------------------------------------------

    registry = ModelRegistry()

    registry_action = "KEEP_CHAMPION"

    if gate.decision == "PROMOTE":
        registry_result = registry.promote(
            challenger_path=INCREMENTAL_PATH,
            reason=(
                "Incremental partial_fit adaptation "
                "passed the safety gate."
            ),
        )

        registry_action = registry_result.get(
            "status",
            "PROMOTE",
        )

    print()
    print(f"Registry action : {registry_action}")

    # ---------------------------------------------------------
    # Save results
    # ---------------------------------------------------------

    result = {
        "experiment": "true_incremental_update",
        "drift_type": "FEATURE_DRIFT",
        "affected_feature": "sensor_2",
        "drift_magnitude": 0.30,

        "champion_mae": champion_drifted["mae"],
        "champion_rmse": champion_drifted["rmse"],
        "champion_r2": champion_drifted["r2"],

        "before_update_mae": before_update["mae"],
        "before_update_rmse": before_update["rmse"],
        "before_update_r2": before_update["r2"],

        "after_update_mae": after_update["mae"],
        "after_update_rmse": after_update["rmse"],
        "after_update_r2": after_update["r2"],

        "update_improvement": (
            before_update["mae"]
            - after_update["mae"]
        ),

        "champion_improvement": gate.improvement,
        "improvement_ratio": gate.improvement_ratio,

        "safety_decision": gate.decision,
        "registry_action": registry_action,
        "reason": gate.reason,
    }

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.DataFrame([result]).to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print("=== FINAL SUMMARY ===")
    print(f"Champion MAE      : {champion_drifted['mae']:.4f}")
    print(f"Before update MAE : {before_update['mae']:.4f}")
    print(f"After update MAE  : {after_update['mae']:.4f}")
    print(
        f"Update gain       : "
        f"{result['update_improvement']:.4f}"
    )
    print(f"Safety decision   : {gate.decision}")
    print(f"Registry action   : {registry_action}")

    print()
    print(f"Saved: {OUTPUT_FILE.resolve()}")
    print()
    print("=== INCREMENTAL UPDATE COMPLETE ===")


if __name__ == "__main__":
    main()
