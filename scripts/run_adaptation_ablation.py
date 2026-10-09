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

DATA = (
    Path(CONFIG["paths"]["processed_data"])
    / "FD001_temporal_sensor2_drift.csv"
)

MODEL = (
    Path(CONFIG["paths"]["artifacts"])
    / "baseline_rul_model.joblib"
)

RESULT = (
    Path(CONFIG["paths"]["results"])
    / "adaptation_ablation.csv"
)


def metrics(y, p):
    return {
        "mae": mean_absolute_error(y, p),
        "rmse": mean_squared_error(y, p) ** 0.5,
        "r2": r2_score(y, p),
    }


def main():

    print("=== DRIFTX-FORENSICS: ADAPTATION ABLATION ===")

    df = pd.read_csv(DATA)

    units = sorted(df.unit.unique())
    split = int(len(units) * 0.80)

    train_units = units[:split]
    test_units = units[split:]

    train = df[df.unit.isin(train_units)].copy()
    test = df[df.unit.isin(test_units)].copy()

    champion = RULModel()
    champion.load(MODEL)

    rows = []

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

        # -----------------------------
        # STATIC CHAMPION
        # -----------------------------

        static_pred = champion.predict(
            future[FEATURES]
        )

        static_metrics = metrics(
            future.RUL,
            static_pred,
        )

        # -----------------------------
        # TRAIN ONLINE ADAPTER
        # -----------------------------

        train_drift = train[
            train.drift_active == 1
        ]

        adapter = ResidualIncrementalAdapter()

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

        # -----------------------------
        # ADAPT USING THIS ENGINE
        # -----------------------------

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

        correction = adapter.predict_correction(
            future[FEATURES]
        )

        naive_pred = (
            static_pred + correction
        )

        naive_metrics = metrics(
            future.RUL,
            naive_pred,
        )

        # -----------------------------
        # SAFETY-GATED DRIFTX
        # -----------------------------

        promote = (
            naive_metrics["mae"]
            <= static_metrics["mae"]
            and
            naive_metrics["rmse"]
            <= static_metrics["rmse"]
            and
            naive_metrics["r2"]
            >= static_metrics["r2"] - 0.02
        )

        if promote:
            driftx_pred = naive_pred
            decision = "PROMOTE"
        else:
            driftx_pred = static_pred
            decision = "REJECT"

        driftx_metrics = metrics(
            future.RUL,
            driftx_pred,
        )

        rows.append({
            "unit": unit,

            "static_mae":
                static_metrics["mae"],
            "naive_mae":
                naive_metrics["mae"],
            "driftx_mae":
                driftx_metrics["mae"],

            "static_rmse":
                static_metrics["rmse"],
            "naive_rmse":
                naive_metrics["rmse"],
            "driftx_rmse":
                driftx_metrics["rmse"],

            "static_r2":
                static_metrics["r2"],
            "naive_r2":
                naive_metrics["r2"],
            "driftx_r2":
                driftx_metrics["r2"],

            "safety_decision":
                decision,
        })

    result = pd.DataFrame(rows)

    print()
    print("=== AGGREGATE RESULTS ===")

    for method in ["static", "naive", "driftx"]:

        print(
            f"{method.upper():8s} | "
            f"MAE  {result[method + '_mae'].mean():.4f} | "
            f"RMSE {result[method + '_rmse'].mean():.4f} | "
            f"R2   {result[method + '_r2'].mean():.4f}"
        )

    print()
    print(
        "Safety promotions:",
        (result.safety_decision == "PROMOTE").sum(),
        "/",
        len(result),
    )

    print(
        "Safety rejections :",
        (result.safety_decision == "REJECT").sum(),
        "/",
        len(result),
    )

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
