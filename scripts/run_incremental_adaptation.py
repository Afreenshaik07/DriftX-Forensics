from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from src.driftx.config import CONFIG
from src.driftx.data.preprocessing import (
    load_training_data,
    create_rul_target,
)
from src.driftx.data.drift_lab import inject_feature_drift
from src.driftx.models.rul_model import RULModel
from src.driftx.models.challenger import ChallengerModel
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

CHALLENGER_PATH = (
    Path(CONFIG["paths"]["artifacts"])
    / "incremental_challenger_rul_model.joblib"
)

OUTPUT_FILE = (
    Path(CONFIG["paths"]["results"])
    / "incremental_adaptation.csv"
)


def calculate_metrics(predictions, actual):
    mae = mean_absolute_error(actual, predictions)
    rmse = mean_squared_error(actual, predictions) ** 0.5
    r2 = r2_score(actual, predictions)

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
    }


def create_drifted_regime(
    df: pd.DataFrame,
    magnitude: float = 0.30,
):
    df = df.reset_index(drop=True)

    change_point = len(df) // 2

    stable = df.iloc[:change_point].copy()

    drifted = df.iloc[change_point:].copy()

    drifted = inject_feature_drift(
        drifted,
        features=["sensor_2"],
        magnitude=magnitude,
    )

    combined = pd.concat(
        [stable, drifted],
        ignore_index=True,
    )

    return combined, stable, drifted


def main():

    print(
        "=== DRIFTX-FORENSICS: "
        "INCREMENTAL ADAPTATION EXPERIMENT ==="
    )

    # --------------------------------------------------
    # 1. Load dataset
    # --------------------------------------------------

    raw_dir = Path(
        CONFIG["paths"]["raw_data"]
    )

    df = create_rul_target(
        load_training_data(raw_dir)
    )

    print(
        f"Dataset rows: {len(df)}"
    )

    # --------------------------------------------------
    # 2. Engine-level split
    # --------------------------------------------------

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
        f"Training engines: {len(train_units)}"
    )

    print(
        f"Test engines: {len(test_units)}"
    )

    print(
        f"Training rows: {len(train_df)}"
    )

    print(
        f"Test rows: {len(test_df)}"
    )

    # --------------------------------------------------
    # 3. Controlled drift in unseen test engines
    # --------------------------------------------------

    (
        drifted_test,
        stable_test,
        current_test,
    ) = create_drifted_regime(
        test_df,
        magnitude=0.30,
    )

    print()
    print(
        "=== CONTROLLED DRIFT ==="
    )

    print(
        "Type     : FEATURE_DRIFT"
    )

    print(
        "Feature  : sensor_2"
    )

    print(
        "Magnitude: 30%"
    )

    # --------------------------------------------------
    # 4. Load champion
    # --------------------------------------------------

    if not CHAMPION_PATH.exists():
        raise FileNotFoundError(
            f"Champion model not found: "
            f"{CHAMPION_PATH}"
        )

    champion = RULModel()

    champion.load(
        CHAMPION_PATH
    )

    # --------------------------------------------------
    # 5. Champion on stable regime
    # --------------------------------------------------

    stable_predictions = champion.predict(
        stable_test[FEATURES]
    )

    stable_metrics = calculate_metrics(
        stable_predictions,
        stable_test["RUL"],
    )

    # --------------------------------------------------
    # 6. Champion on drifted regime
    # --------------------------------------------------

    champion_predictions = champion.predict(
        current_test[FEATURES]
    )

    champion_metrics = calculate_metrics(
        champion_predictions,
        current_test["RUL"],
    )

    print()
    print(
        "=== CHAMPION: STABLE ==="
    )

    print(
        f"MAE  : {stable_metrics['mae']:.4f}"
    )

    print(
        f"RMSE : {stable_metrics['rmse']:.4f}"
    )

    print(
        f"R2   : {stable_metrics['r2']:.4f}"
    )

    print()
    print(
        "=== CHAMPION: DRIFTED ==="
    )

    print(
        f"MAE  : {champion_metrics['mae']:.4f}"
    )

    print(
        f"RMSE : {champion_metrics['rmse']:.4f}"
    )

    print(
        f"R2   : {champion_metrics['r2']:.4f}"
    )

    # --------------------------------------------------
    # 7. Generate drifted adaptation data
    # --------------------------------------------------

    (
        drifted_train,
        historical_train,
        drifted_train_part,
    ) = create_drifted_regime(
        train_df,
        magnitude=0.30,
    )

    # --------------------------------------------------
    # 8. IMPORTANT:
    # Historical + drifted data
    # --------------------------------------------------

    incremental_train = pd.concat(
        [
            historical_train,
            drifted_train_part,
        ],
        ignore_index=True,
    )

    X_incremental = incremental_train[
        FEATURES
    ]

    y_incremental = incremental_train[
        "RUL"
    ]

    print()
    print(
        "=== INCREMENTAL ADAPTATION DATA ==="
    )

    print(
        f"Historical rows : "
        f"{len(historical_train)}"
    )

    print(
        f"Drifted rows    : "
        f"{len(drifted_train_part)}"
    )

    print(
        f"Total rows      : "
        f"{len(incremental_train)}"
    )

    # --------------------------------------------------
    # 9. Train challenger
    # --------------------------------------------------

    print()
    print(
        "=== TRAINING INCREMENTAL CHALLENGER ==="
    )

    challenger = ChallengerModel()

    challenger.train(
        X=X_incremental,
        y=y_incremental,
    )

    challenger.save(
        CHALLENGER_PATH
    )

    # --------------------------------------------------
    # 10. Evaluate challenger
    # --------------------------------------------------

    challenger_predictions = challenger.predict(
        current_test[FEATURES]
    )

    challenger_metrics = calculate_metrics(
        challenger_predictions,
        current_test["RUL"],
    )

    print()
    print(
        "=== CHALLENGER: DRIFTED ==="
    )

    print(
        f"MAE  : {challenger_metrics['mae']:.4f}"
    )

    print(
        f"RMSE : {challenger_metrics['rmse']:.4f}"
    )

    print(
        f"R2   : {challenger_metrics['r2']:.4f}"
    )

    # --------------------------------------------------
    # 11. Safety gate
    # --------------------------------------------------

    print()
    print(
        "=== SAFETY GATE ==="
    )

    gate_result = evaluate_safety_gate(
        champion_metrics=champion_metrics,
        challenger_metrics=challenger_metrics,
        min_improvement=0.0,
        max_r2_drop=0.02,
        max_rmse_increase=0.0,
    )

    print(
        f"Decision : "
        f"{gate_result.decision}"
    )

    print(
        f"MAE improvement : "
        f"{gate_result.improvement:.4f}"
    )

    print(
        f"Improvement ratio : "
        f"{gate_result.improvement_ratio:.4%}"
    )

    print(
        f"Reason : "
        f"{gate_result.reason}"
    )

    # --------------------------------------------------
    # 12. Registry
    # --------------------------------------------------

    registry = ModelRegistry()

    registry_action = "KEEP_CHAMPION"

    if gate_result.decision == "PROMOTE":

        registry_result = registry.promote(
            challenger_path=CHALLENGER_PATH,
            reason=(
                "Incremental adaptation challenger "
                "passed the safety gate."
            ),
        )

        registry_action = registry_result.get(
            "status",
            "PROMOTE",
        )

    print()
    print(
        f"Registry action : "
        f"{registry_action}"
    )

    # --------------------------------------------------
    # 13. Results
    # --------------------------------------------------

    improvement = (
        champion_metrics["mae"]
        - challenger_metrics["mae"]
    )

    result = {
        "experiment": "incremental_adaptation",
        "drift_type": "FEATURE_DRIFT",
        "affected_feature": "sensor_2",
        "drift_magnitude": 0.30,

        "stable_mae": stable_metrics["mae"],
        "champion_drifted_mae": champion_metrics["mae"],
        "challenger_drifted_mae": challenger_metrics["mae"],

        "champion_drifted_rmse": champion_metrics["rmse"],
        "challenger_drifted_rmse": challenger_metrics["rmse"],

        "champion_drifted_r2": champion_metrics["r2"],
        "challenger_drifted_r2": challenger_metrics["r2"],

        "adaptation_improvement": improvement,

        "improvement_ratio": (
            gate_result.improvement_ratio
        ),

        "safety_decision": (
            gate_result.decision
        ),

        "registry_action": registry_action,

        "reason": gate_result.reason,
    }

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.DataFrame(
        [result]
    ).to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print(
        "=== EXPERIMENT SUMMARY ==="
    )

    print(
        f"Champion MAE   : "
        f"{champion_metrics['mae']:.4f}"
    )

    print(
        f"Challenger MAE : "
        f"{challenger_metrics['mae']:.4f}"
    )

    print(
        f"Improvement    : "
        f"{improvement:.4f}"
    )

    print(
        f"Decision       : "
        f"{gate_result.decision}"
    )

    print(
        f"Registry       : "
        f"{registry_action}"
    )

    print()
    print(
        f"Saved: {OUTPUT_FILE.resolve()}"
    )

    print()
    print(
        "=== INCREMENTAL ADAPTATION "
        "EXPERIMENT COMPLETE ==="
    )


if __name__ == "__main__":
    main()