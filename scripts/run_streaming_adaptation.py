from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.driftx.config import CONFIG
from src.driftx.models.rul_model import RULModel
from src.driftx.models.residual_adapter import ResidualIncrementalAdapter


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

RESULT_PATH = (
    Path(CONFIG["paths"]["results"])
    / "streaming_online_adaptation.csv"
)


def metrics(y_true, predictions):

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
        "STREAMING ONLINE ADAPTATION ==="
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

    champion = RULModel()
    champion.load(CHAMPION_PATH)

    # ---------------------------------------------------------
    # Train residual adapter only on TRAINING engines.
    # ---------------------------------------------------------

    train_drift = train_df[
        train_df["drift_active"] == 1
    ].copy()

    train_predictions = champion.predict(
        train_drift[FEATURES]
    )

    train_residuals = (
        train_drift["RUL"].to_numpy()
        - train_predictions
    )

    adapter = ResidualIncrementalAdapter()

    # Initialize using only a small initial drift window.
    init_size = min(
        512,
        len(train_drift),
    )

    adapter.fit(
        train_drift[
            FEATURES
        ].iloc[:init_size],
        train_residuals[:init_size],
    )

    for start in range(
        init_size,
        len(train_drift),
        512,
    ):

        end = min(
            start + 512,
            len(train_drift),
        )

        adapter.partial_fit(
            train_drift[
                FEATURES
            ].iloc[start:end],
            train_residuals[start:end],
        )

    # ---------------------------------------------------------
    # Streaming test protocol
    # ---------------------------------------------------------

    all_results = []

    for unit in test_units:

        engine = test_df[
            test_df["unit"] == unit
        ].sort_values("cycle").copy()

        drift_rows = engine[
            engine["drift_active"] == 1
        ].copy()

        if len(drift_rows) < 20:
            continue

        # First 25% of post-drift observations:
        # adaptation window.
        adaptation_size = max(
            1,
            int(len(drift_rows) * 0.25)
        )

        adaptation_window = drift_rows.iloc[
            :adaptation_size
        ]

        evaluation_window = drift_rows.iloc[
            adaptation_size:
        ]

        # -----------------------------------------------------
        # Adapt using only observations that arrived first.
        # -----------------------------------------------------

        champion_adapt_pred = champion.predict(
            adaptation_window[FEATURES]
        )

        adaptation_residuals = (
            adaptation_window["RUL"].to_numpy()
            - champion_adapt_pred
        )

        adapter.partial_fit(
            adaptation_window[FEATURES],
            adaptation_residuals,
        )

        # -----------------------------------------------------
        # Evaluate only on FUTURE observations.
        # -----------------------------------------------------

        champion_future_pred = champion.predict(
            evaluation_window[FEATURES]
        )

        correction_future = (
            adapter.predict_correction(
                evaluation_window[FEATURES]
            )
        )

        adapted_future_pred = (
            champion_future_pred
            + correction_future
        )

        champion_result = metrics(
            evaluation_window["RUL"],
            champion_future_pred,
        )

        adapted_result = metrics(
            evaluation_window["RUL"],
            adapted_future_pred,
        )

        all_results.append(
            {
                "unit": unit,
                "adaptation_rows": len(
                    adaptation_window
                ),
                "evaluation_rows": len(
                    evaluation_window
                ),
                "champion_mae":
                    champion_result["mae"],
                "champion_rmse":
                    champion_result["rmse"],
                "champion_r2":
                    champion_result["r2"],
                "adapted_mae":
                    adapted_result["mae"],
                "adapted_rmse":
                    adapted_result["rmse"],
                "adapted_r2":
                    adapted_result["r2"],
                "mae_improvement":
                    champion_result["mae"]
                    - adapted_result["mae"],
            }
        )

    # ---------------------------------------------------------
    # Aggregate
    # ---------------------------------------------------------

    results = pd.DataFrame(
        all_results
    )

    if results.empty:
        raise RuntimeError(
            "No valid streaming test engines."
        )

    champion_mae = results[
        "champion_mae"
    ].mean()

    adapted_mae = results[
        "adapted_mae"
    ].mean()

    champion_rmse = results[
        "champion_rmse"
    ].mean()

    adapted_rmse = results[
        "adapted_rmse"
    ].mean()

    champion_r2 = results[
        "champion_r2"
    ].mean()

    adapted_r2 = results[
        "adapted_r2"
    ].mean()

    improvement = (
        champion_mae
        - adapted_mae
    )

    improvement_ratio = (
        improvement / champion_mae
        if champion_mae > 0
        else 0.0
    )

    print()
    print("=== STREAMING RESULTS ===")

    print(
        f"Test engines       : "
        f"{len(results)}"
    )

    print(
        f"Champion MAE       : "
        f"{champion_mae:.4f}"
    )

    print(
        f"Adapted MAE        : "
        f"{adapted_mae:.4f}"
    )

    print(
        f"Champion RMSE      : "
        f"{champion_rmse:.4f}"
    )

    print(
        f"Adapted RMSE       : "
        f"{adapted_rmse:.4f}"
    )

    print(
        f"Champion R2        : "
        f"{champion_r2:.4f}"
    )

    print(
        f"Adapted R2         : "
        f"{adapted_r2:.4f}"
    )

    print(
        f"MAE improvement    : "
        f"{improvement:.4f}"
    )

    print(
        f"Improvement ratio  : "
        f"{improvement_ratio:.4%}"
    )

    # ---------------------------------------------------------
    # Safety decision
    # ---------------------------------------------------------

    safety = (
        improvement >= 0
        and adapted_rmse <= champion_rmse
        and adapted_r2 >= champion_r2 - 0.02
    )

    decision = (
        "PROMOTE"
        if safety
        else "REJECT"
    )

    print()
    print("=== SAFETY DECISION ===")
    print(f"Decision : {decision}")

    # ---------------------------------------------------------
    # Save per-engine results
    # ---------------------------------------------------------

    RESULT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        RESULT_PATH,
        index=False,
    )

    print()
    print(
        f"Saved: {RESULT_PATH.resolve()}"
    )

    print()
    print(
        "=== STREAMING EXPERIMENT COMPLETE ==="
    )


if __name__ == "__main__":
    main()

