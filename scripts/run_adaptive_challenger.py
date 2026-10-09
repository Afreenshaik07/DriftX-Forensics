from pathlib import Path

import pandas as pd

from src.driftx.config import CONFIG
from src.driftx.data.preprocessing import (
    create_rul_target,
    load_training_data,
)
from src.driftx.models.rul_model import RULModel
from src.driftx.models.challenger import ChallengerModel
from src.driftx.evaluation.safety_gate import (
    evaluate_safety_gate,
)
from src.driftx.models.model_registry import (
    ModelRegistry,
)


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
    / "adaptive_challenger_rul_model.joblib"
)

OUTPUT_FILE = (
    Path(CONFIG["paths"]["results"])
    / "adaptive_challenger_experiment.csv"
)


def calculate_metrics(predictions, actual):
    """
    Calculate MAE, RMSE and R2.
    """

    from sklearn.metrics import (
        mean_absolute_error,
        mean_squared_error,
        r2_score,
    )

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


def main():

    print(
        "=== DRIFTX-FORENSICS: "
        "ADAPTIVE CHALLENGER EXPERIMENT ==="
    )

    # ---------------------------------------------------------
    # 1. Load dataset
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # 2. Engine-based split
    # ---------------------------------------------------------

    units = sorted(
        df["unit"].unique()
    )

    split_index = int(
        len(units) * 0.8
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

    # ---------------------------------------------------------
    # 3. Simulate current regime
    # ---------------------------------------------------------
    #
    # The final portion of the historical training stream
    # represents the most recent environment observed by the
    # MLOps system.
    #
    # The challenger learns primarily from this recent regime.
    # ---------------------------------------------------------

    train_df = train_df.sort_values(
        ["unit", "cycle"]
    )

    recent_fraction = 0.30

    recent_start = int(
        len(train_df)
        * (1.0 - recent_fraction)
    )

    recent_df = train_df.iloc[
        recent_start:
    ].copy()

    X_recent = recent_df[
        FEATURES
    ]

    y_recent = recent_df[
        "RUL"
    ]

    X_test = test_df[
        FEATURES
    ]

    y_test = test_df[
        "RUL"
    ]

    print(
        f"Historical training rows: "
        f"{len(train_df)}"
    )

    print(
        f"Recent adaptation rows: "
        f"{len(recent_df)}"
    )

    print(
        f"Test rows: "
        f"{len(test_df)}"
    )

    # ---------------------------------------------------------
    # 4. Load Champion
    # ---------------------------------------------------------

    if not CHAMPION_PATH.exists():
        raise FileNotFoundError(
            "Champion model not found: "
            f"{CHAMPION_PATH}"
        )

    champion = RULModel()

    champion.load(
        CHAMPION_PATH
    )

    champion_predictions = champion.predict(
        X_test
    )

    champion_metrics = calculate_metrics(
        champion_predictions,
        y_test,
    )

    print()
    print("=== CHAMPION ===")

    print(
        f"MAE  : "
        f"{champion_metrics['mae']:.4f}"
    )

    print(
        f"RMSE : "
        f"{champion_metrics['rmse']:.4f}"
    )

    print(
        f"R2   : "
        f"{champion_metrics['r2']:.4f}"
    )

    # ---------------------------------------------------------
    # 5. Train adaptive Challenger
    # ---------------------------------------------------------

    print()
    print(
        "=== TRAINING ADAPTIVE CHALLENGER ==="
    )

    challenger = ChallengerModel()

    challenger.train(
        X=X_recent,
        y=y_recent,
    )

    challenger.save(
        CHALLENGER_PATH
    )

    challenger_predictions = (
        challenger.predict(
            X_test
        )
    )

    challenger_metrics = calculate_metrics(
        challenger_predictions,
        y_test,
    )

    print()
    print("=== ADAPTIVE CHALLENGER ===")

    print(
        f"MAE  : "
        f"{challenger_metrics['mae']:.4f}"
    )

    print(
        f"RMSE : "
        f"{challenger_metrics['rmse']:.4f}"
    )

    print(
        f"R2   : "
        f"{challenger_metrics['r2']:.4f}"
    )

    # ---------------------------------------------------------
    # 6. Safety Gate
    # ---------------------------------------------------------

    print()
    print("=== SAFETY GATE ===")

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

    # ---------------------------------------------------------
    # 7. Registry action
    # ---------------------------------------------------------

    registry = ModelRegistry()

    registry_action = "KEEP_CHAMPION"

    if gate_result.decision == "PROMOTE":

        registry_result = registry.promote(
            challenger_path=CHALLENGER_PATH,
            reason=(
                "Adaptive challenger passed "
                "the safety gate."
            ),
        )

        registry_action = registry_result[
            "status"
        ]

    print()
    print(
        f"Registry action : "
        f"{registry_action}"
    )

    # ---------------------------------------------------------
    # 8. Save experiment
    # ---------------------------------------------------------

    result = {
        "adaptation_training_fraction": (
            recent_fraction
        ),
        "historical_training_rows": len(
            train_df
        ),
        "adaptation_rows": len(
            recent_df
        ),
        "test_rows": len(
            test_df
        ),
        "champion_mae": (
            gate_result.champion_mae
        ),
        "challenger_mae": (
            gate_result.challenger_mae
        ),
        "mae_improvement": (
            gate_result.improvement
        ),
        "mae_improvement_ratio": (
            gate_result.improvement_ratio
        ),
        "champion_rmse": (
            gate_result.champion_rmse
        ),
        "challenger_rmse": (
            gate_result.challenger_rmse
        ),
        "champion_r2": (
            gate_result.champion_r2
        ),
        "challenger_r2": (
            gate_result.challenger_r2
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

    print()
    print(
        f"Saved: "
        f"{OUTPUT_FILE.resolve()}"
    )

    print()
    print(
        "=== ADAPTIVE CHALLENGER "
        "EXPERIMENT COMPLETE ==="
    )


if __name__ == "__main__":
    main()