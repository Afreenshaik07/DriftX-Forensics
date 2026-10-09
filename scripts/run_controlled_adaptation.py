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


# ============================================================
# CONFIGURATION
# ============================================================

FEATURES = (
    [
        "cycle",
        "setting_1",
        "setting_2",
        "setting_3",
    ]
    + [f"sensor_{i}" for i in range(1, 22)]
)

CHAMPION_PATH = (
    Path(CONFIG["paths"]["artifacts"])
    / "baseline_rul_model.joblib"
)

CHALLENGER_PATH = (
    Path(CONFIG["paths"]["artifacts"])
    / "controlled_challenger_rul_model.joblib"
)

OUTPUT_FILE = (
    Path(CONFIG["paths"]["results"])
    / "controlled_adaptation.csv"
)


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(predictions, actual):
    """Calculate regression metrics."""

    mae = mean_absolute_error(
        actual,
        predictions,
    )

    rmse = mean_squared_error(
        actual,
        predictions,
    ) ** 0.5

    r2 = r2_score(
        actual,
        predictions,
    )

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
    }


# ============================================================
# APPLY CONTROLLED DRIFT
# ============================================================

def create_drifted_regime(
    df: pd.DataFrame,
    magnitude: float = 0.30,
):
    """
    Split a dataset into stable and drifted regimes.

    Drift is applied ONLY to the second half.
    """

    df = df.reset_index(drop=True)

    change_point = len(df) // 2

    stable_part = df.iloc[
        :change_point
    ].copy()

    drift_part = df.iloc[
        change_point:
    ].copy()

    drift_part = inject_feature_drift(
        drift_part,
        features=["sensor_2"],
        magnitude=magnitude,
    )

    combined = pd.concat(
        [
            stable_part,
            drift_part,
        ],
        ignore_index=True,
    )

    return (
        combined,
        stable_part,
        drift_part,
        change_point,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=== DRIFTX-FORENSICS: "
        "CONTROLLED ADAPTATION EXPERIMENT ==="
    )

    # --------------------------------------------------------
    # 1. Load CMAPSS
    # --------------------------------------------------------

    raw_dir = Path(
        CONFIG["paths"]["raw_data"]
    )

    df = load_training_data(
        raw_dir
    )

    df = create_rul_target(
        df
    )

    print(
        f"Dataset rows: {len(df)}"
    )

    # --------------------------------------------------------
    # 2. Engine-level split
    # --------------------------------------------------------

    units = sorted(
        df["unit"].unique()
    )

    split_index = int(
        len(units) * 0.80
    )

    train_units = units[
        :split_index
    ]

    test_units = units[
        split_index:
    ]

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

    # --------------------------------------------------------
    # 3. Create controlled drift in TEST data
    # --------------------------------------------------------

    (
        drifted_test,
        stable_test,
        current_test,
        test_change_point,
    ) = create_drifted_regime(
        test_df,
        magnitude=0.30,
    )

    print()
    print(
        "=== CONTROLLED DRIFT SCENARIO ==="
    )

    print(
        "Drift type       : FEATURE_DRIFT"
    )

    print(
        "Affected feature : sensor_2"
    )

    print(
        "Magnitude        : 30%"
    )

    print(
        f"Change point     : {test_change_point}"
    )

    print(
        f"Stable rows      : {len(stable_test)}"
    )

    print(
        f"Drifted rows     : {len(current_test)}"
    )

    # --------------------------------------------------------
    # 4. Load Champion
    # --------------------------------------------------------

    if not CHAMPION_PATH.exists():
        raise FileNotFoundError(
            f"Champion model not found: "
            f"{CHAMPION_PATH}"
        )

    champion = RULModel()

    champion.load(
        CHAMPION_PATH
    )

    # --------------------------------------------------------
    # 5. Champion on stable regime
    # --------------------------------------------------------

    stable_predictions = champion.predict(
        stable_test[FEATURES]
    )

    stable_metrics = calculate_metrics(
        stable_predictions,
        stable_test["RUL"],
    )

    print()
    print(
        "=== CHAMPION: STABLE REGIME ==="
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

    # --------------------------------------------------------
    # 6. Champion after drift
    # --------------------------------------------------------

    champion_predictions = champion.predict(
        current_test[FEATURES]
    )

    champion_metrics = calculate_metrics(
        champion_predictions,
        current_test["RUL"],
    )

    print()
    print(
        "=== CHAMPION: DRIFTED REGIME ==="
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

    # --------------------------------------------------------
    # 7. Create independent adaptation data
    # --------------------------------------------------------
    #
    # Use TRAINING engines.
    # Apply the same known drift mechanism.
    # Only the drifted half becomes adaptation data.
    # --------------------------------------------------------

    (
        drifted_train,
        train_stable,
        train_drift,
        train_change_point,
    ) = create_drifted_regime(
        train_df,
        magnitude=0.30,
    )

    adaptation_df = train_drift.copy()

    X_adaptation = adaptation_df[
        FEATURES
    ]

    y_adaptation = adaptation_df[
        "RUL"
    ]

    print()
    print(
        "=== ADAPTATION DATA ==="
    )

    print(
        f"Adaptation rows : "
        f"{len(adaptation_df)}"
    )

    print(
        "Source engines  : "
        f"{len(train_units)}"
    )

    print(
        "Regime          : DRIFTED"
    )

    # --------------------------------------------------------
    # 8. Train Challenger
    # --------------------------------------------------------

    print()
    print(
        "=== TRAINING CHALLENGER ==="
    )

    challenger = ChallengerModel()

    challenger.train(
        X=X_adaptation,
        y=y_adaptation,
    )

    challenger.save(
        CHALLENGER_PATH
    )

    # --------------------------------------------------------
    # 9. Evaluate Challenger on unseen test engines
    # --------------------------------------------------------

    challenger_predictions = challenger.predict(
        current_test[FEATURES]
    )

    challenger_metrics = calculate_metrics(
        challenger_predictions,
        current_test["RUL"],
    )

    print()
    print(
        "=== CHALLENGER: UNSEEN DRIFTED REGIME ==="
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

    # --------------------------------------------------------
    # 10. Safety Gate
    # --------------------------------------------------------

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
        f"Decision : {gate_result.decision}"
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
        f"Reason : {gate_result.reason}"
    )

    # --------------------------------------------------------
    # 11. Model Registry
    # --------------------------------------------------------

    registry = ModelRegistry()

    registry_action = "KEEP_CHAMPION"

    if gate_result.decision == "PROMOTE":

        registry_result = registry.promote(
            challenger_path=CHALLENGER_PATH,
            reason=(
                "Controlled feature-drift challenger "
                "passed the safety gate."
            ),
        )

        registry_action = registry_result.get(
            "status",
            "PROMOTE",
        )

    print()
    print(
        f"Registry action : {registry_action}"
    )

    # --------------------------------------------------------
    # 12. Calculate experiment effects
    # --------------------------------------------------------

    drift_degradation = (
        champion_metrics["mae"]
        - stable_metrics["mae"]
    )

    adaptation_improvement = (
        champion_metrics["mae"]
        - challenger_metrics["mae"]
    )

    # --------------------------------------------------------
    # 13. Save results
    # --------------------------------------------------------

    result = {
        "experiment": "controlled_adaptation",
        "drift_type": "FEATURE_DRIFT",
        "affected_feature": "sensor_2",
        "drift_magnitude": 0.30,

        "stable_mae": stable_metrics["mae"],
        "stable_rmse": stable_metrics["rmse"],
        "stable_r2": stable_metrics["r2"],

        "champion_drifted_mae": (
            champion_metrics["mae"]
        ),
        "champion_drifted_rmse": (
            champion_metrics["rmse"]
        ),
        "champion_drifted_r2": (
            champion_metrics["r2"]
        ),

        "challenger_drifted_mae": (
            challenger_metrics["mae"]
        ),
        "challenger_drifted_rmse": (
            challenger_metrics["rmse"]
        ),
        "challenger_drifted_r2": (
            challenger_metrics["r2"]
        ),

        "drift_degradation_mae": (
            drift_degradation
        ),

        "adaptation_improvement_mae": (
            adaptation_improvement
        ),

        "mae_improvement_ratio": (
            gate_result.improvement_ratio
        ),

        "safety_decision": (
            gate_result.decision
        ),

        "registry_action": registry_action,

        "reason": gate_result.reason,
    }

    result_df = pd.DataFrame(
        [result]
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # 14. Final summary
    # --------------------------------------------------------

    print()
    print(
        "=== EXPERIMENT SUMMARY ==="
    )

    print(
        f"Stable Champion MAE    : "
        f"{stable_metrics['mae']:.4f}"
    )

    print(
        f"Drifted Champion MAE   : "
        f"{champion_metrics['mae']:.4f}"
    )

    print(
        f"Drifted Challenger MAE : "
        f"{challenger_metrics['mae']:.4f}"
    )

    print(
        f"Adaptation improvement : "
        f"{adaptation_improvement:.4f}"
    )

    print(
        f"Safety decision         : "
        f"{gate_result.decision}"
    )

    print(
        f"Registry action         : "
        f"{registry_action}"
    )

    print()
    print(
        f"Saved: {OUTPUT_FILE.resolve()}"
    )

    print()
    print(
        "=== CONTROLLED ADAPTATION "
        "EXPERIMENT COMPLETE ==="
    )


if __name__ == "__main__":
    main()