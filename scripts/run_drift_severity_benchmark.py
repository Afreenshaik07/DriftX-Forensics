from pathlib import Path
import pandas as pd

from src.driftx.config import CONFIG
from src.driftx.data.drift_lab import inject_feature_drift
from src.driftx.data.preprocessing import load_training_data, create_rul_target
from src.driftx.models.rul_model import RULModel
from src.driftx.models.residual_adapter import ResidualIncrementalAdapter
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


FEATURES = (
    ["cycle", "setting_1", "setting_2", "setting_3"]
    + [f"sensor_{i}" for i in range(1, 22)]
)

ARTIFACT = Path(CONFIG["paths"]["artifacts"]) / "baseline_rul_model.joblib"
RESULT = Path(CONFIG["paths"]["results"]) / "drift_severity_benchmark.csv"


def metric(y, p):
    return (
        mean_absolute_error(y, p),
        mean_squared_error(y, p) ** 0.5,
        r2_score(y, p),
    )


def main():
    print("=== DRIFTX-FORENSICS: DRIFT SEVERITY BENCHMARK ===")

    df = create_rul_target(
        load_training_data(Path(CONFIG["paths"]["raw_data"]))
    )

    units = sorted(df["unit"].unique())
    split = int(len(units) * 0.80)

    train_units = units[:split]
    test_units = units[split:]

    train = df[df.unit.isin(train_units)].copy()
    test = df[df.unit.isin(test_units)].copy()

    champion = RULModel()
    champion.load(ARTIFACT)

    rows = []

    for magnitude in [0.10, 0.20, 0.30, 0.40]:

        drift_train = train.copy()
        drift_test = test.copy()

        for unit in drift_train.unit.unique():
            idx = drift_train[
                drift_train.unit == unit
            ].sort_values("cycle").index

            cut = int(len(idx) * 0.50)

            drift_train.loc[idx[cut:], "sensor_2"] *= (
                1.0 + magnitude
            )

        for unit in drift_test.unit.unique():
            idx = drift_test[
                drift_test.unit == unit
            ].sort_values("cycle").index

            cut = int(len(idx) * 0.50)

            drift_test.loc[idx[cut:], "sensor_2"] *= (
                1.0 + magnitude
            )

        train_drift = drift_train[
            [
                "unit",
                "cycle",
                "setting_1",
                "setting_2",
                "setting_3",
                *[f"sensor_{i}" for i in range(1, 22)],
                "RUL",
            ]
        ]

        train_drift = train_drift[
            train_drift.unit.isin(train_units)
        ]

        adapter = ResidualIncrementalAdapter()

        preds = champion.predict(
            train_drift[FEATURES]
        )

        residuals = (
            train_drift.RUL.to_numpy()
            - preds
        )

        init = min(512, len(train_drift))

        adapter.fit(
            train_drift[FEATURES].iloc[:init],
            residuals[:init],
        )

        for start in range(init, len(train_drift), 512):
            end = min(start + 512, len(train_drift))

            adapter.partial_fit(
                train_drift[FEATURES].iloc[start:end],
                residuals[start:end],
            )

        champion_mae = []
        adapted_mae = []
        champion_rmse = []
        adapted_rmse = []
        champion_r2 = []
        adapted_r2 = []

        for unit in test_units:

            engine = drift_test[
                drift_test.unit == unit
            ].sort_values("cycle")

            midpoint = int(len(engine) * 0.50)

            post = engine.iloc[midpoint:]

            if len(post) < 20:
                continue

            adaptation_size = max(
                1,
                int(len(post) * 0.25)
            )

            adaptation = post.iloc[:adaptation_size]
            future = post.iloc[adaptation_size:]

            if future.empty:
                continue

            cp = champion.predict(
                adaptation[FEATURES]
            )

            residual = (
                adaptation.RUL.to_numpy()
                - cp
            )

            adapter.partial_fit(
                adaptation[FEATURES],
                residual,
            )

            champion_pred = champion.predict(
                future[FEATURES]
            )

            adapted_pred = (
                champion_pred
                + adapter.predict_correction(
                    future[FEATURES]
                )
            )

            cm = metric(future.RUL, champion_pred)
            am = metric(future.RUL, adapted_pred)

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

        rows.append({
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

        print(
            f"{magnitude:.0%} drift | "
            f"MAE {c_mae:.4f} -> {a_mae:.4f} | "
            f"Improvement {ratio:.2%} | "
            f"{decision}"
        )

    result = pd.DataFrame(rows)

    RESULT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        RESULT,
        index=False,
    )

    print()
    print(f"Saved: {RESULT.resolve()}")


if __name__ == "__main__":
    main()
