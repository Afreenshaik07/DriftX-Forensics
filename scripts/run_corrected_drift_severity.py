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

ARTIFACT = Path(CONFIG["paths"]["artifacts"]) / "baseline_rul_model.joblib"
RESULT = Path(CONFIG["paths"]["results"]) / "corrected_drift_severity_benchmark.csv"


def metrics(y, p):
    return (
        mean_absolute_error(y, p),
        mean_squared_error(y, p) ** 0.5,
        r2_score(y, p),
    )


def inject_temporal_drift(df, magnitude):
    result = df.copy()
    result["drift_active"] = 0

    for unit in result["unit"].unique():
        idx = result[result.unit == unit].sort_values("cycle").index
        cut = int(len(idx) * 0.50)
        drift_idx = idx[cut:]

        result.loc[drift_idx, "sensor_2"] *= (1.0 + magnitude)
        result.loc[drift_idx, "drift_active"] = 1

    return result


def main():

    print("=== DRIFTX-FORENSICS: CORRECTED DRIFT SEVERITY BENCHMARK ===")

    base = pd.read_csv(
        Path(CONFIG["paths"]["processed_data"])
        / "FD001_temporal_sensor2_drift.csv"
    )

    units = sorted(base.unit.unique())
    split = int(len(units) * 0.80)

    train_units = units[:split]
    test_units = units[split:]

    clean = base.copy()

    results = []

    for magnitude in [0.10, 0.20, 0.30, 0.40]:

        print()
        print(f"--- {magnitude:.0%} DRIFT ---")

        train = inject_temporal_drift(
            clean[clean.unit.isin(train_units)],
            magnitude,
        )

        test = inject_temporal_drift(
            clean[clean.unit.isin(test_units)],
            magnitude,
        )

        champion = RULModel()
        champion.load(ARTIFACT)

        adapter = ResidualIncrementalAdapter()

        train_drift = train[
            train.drift_active == 1
        ]

        # Initialize only from an initial post-drift batch.
        initial = train_drift.iloc[:512]

        initial_pred = champion.predict(
            initial[FEATURES]
        )

        initial_residual = (
            initial.RUL.to_numpy()
            - initial_pred
        )

        adapter.fit(
            initial[FEATURES],
            initial_residual,
        )

        # Sequential online updates.
        for start in range(
            512,
            len(train_drift),
            512,
        ):

            end = min(
                start + 512,
                len(train_drift),
            )

            batch = train_drift.iloc[
                start:end
            ]

            pred = champion.predict(
                batch[FEATURES]
            )

            residual = (
                batch.RUL.to_numpy()
                - pred
            )

            adapter.partial_fit(
                batch[FEATURES],
                residual,
            )

        champion_mae = []
        adapted_mae = []
        champion_rmse = []
        adapted_rmse = []
        champion_r2 = []
        adapted_r2 = []

        for unit in test_units:

            engine = test[
                test.unit == unit
            ].sort_values("cycle")

            post = engine[
                engine.drift_active == 1
            ]

            if len(post) < 20:
                continue

            adaptation_size = max(
                1,
                int(len(post) * 0.25)
            )

            adaptation = post.iloc[
                :adaptation_size
            ]

            future = post.iloc[
                adaptation_size:
            ]

            if future.empty:
                continue

            adaptation_pred = champion.predict(
                adaptation[FEATURES]
            )

            adaptation_residual = (
                adaptation.RUL.to_numpy()
                - adaptation_pred
            )

            adapter.partial_fit(
                adaptation[FEATURES],
                adaptation_residual,
            )

            champion_pred = champion.predict(
                future[FEATURES]
            )

            correction = adapter.predict_correction(
                future[FEATURES]
            )

            adapted_pred = (
                champion_pred + correction
            )

            cm = metrics(
                future.RUL,
                champion_pred,
            )

            am = metrics(
                future.RUL,
                adapted_pred,
            )

            champion_mae.append(cm[0])
            champion_rmse.append(cm[1])
            champion_r2.append(cm[2])

            adapted_mae.append(am[0])
            adapted_rmse.append(am[1])
            adapted_r2.append(am[2])

        c_mae = sum(champion_mae) / len(champion_mae)
        a_mae = sum(adapted_mae) / len(adapted_mae)

        c_rmse = sum(champion_rmse) / len(champion_rmse)
        a_rmse = sum(adapted_rmse) / len(adapted_rmse)

        c_r2 = sum(champion_r2) / len(champion_r2)
        a_r2 = sum(adapted_r2) / len(adapted_r2)

        improvement = c_mae - a_mae
        ratio = improvement / c_mae

        decision = (
            "PROMOTE"
            if improvement >= 0
            and a_rmse <= c_rmse
            and a_r2 >= c_r2 - 0.02
            else "REJECT"
        )

        print(
            f"MAE {c_mae:.4f} -> {a_mae:.4f} | "
            f"Improvement {ratio:.2%} | "
            f"{decision}"
        )

        results.append({
            "drift_magnitude": magnitude,
            "test_engines": len(champion_mae),
            "champion_mae": c_mae,
            "adapted_mae": a_mae,
            "champion_rmse": c_rmse,
            "adapted_rmse": a_rmse,
            "champion_r2": c_r2,
            "adapted_r2": a_r2,
            "mae_improvement": improvement,
            "improvement_ratio": ratio,
            "decision": decision,
        })

    output = pd.DataFrame(results)

    RESULT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        RESULT,
        index=False,
    )

    print()
    print(f"Saved: {RESULT.resolve()}")


if __name__ == "__main__":
    main()
