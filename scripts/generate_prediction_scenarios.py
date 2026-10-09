from pathlib import Path
import sys
import numpy as np
import pandas as pd

from src.driftx.config import CONFIG
from src.driftx.data.preprocessing import load_training_data, create_rul_target
from src.driftx.models.rul_model import RULModel


def main(seed=42, change_fraction=0.50):

    rng = np.random.default_rng(seed)

    raw_dir = Path(CONFIG["paths"]["raw_data"])
    model_path = (
        Path(CONFIG["paths"]["artifacts"])
        / "baseline_rul_model.joblib"
    )

    df = create_rul_target(
        load_training_data(raw_dir)
    )

    df = df.sample(
        frac=1.0,
        random_state=seed
    ).reset_index(drop=True)

    n = len(df)
    cp = int(n * change_fraction)

    features = (
        ["cycle", "setting_1", "setting_2", "setting_3"]
        + [f"sensor_{i}" for i in range(1, 22)]
    )

    model = RULModel()
    model.load(model_path)

    predictions = np.asarray(
        model.predict(df[features]),
        dtype=float
    )

    output_dir = Path("data/interim")
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # =========================================================
    # F05 - MODERATE PREDICTION DRIFT
    # =========================================================

    f05 = df.copy()

    f05["model_prediction"] = predictions

    f05.loc[
        cp:,
        "model_prediction"
    ] = (
        predictions[cp:] * 0.70
        + 3.0
    )

    f05["scenario"] = "F05_prediction"
    f05["change_point"] = cp
    f05["true_cause"] = "PREDICTION_DRIFT"
    f05["true_first_stage"] = "prediction"

    # =========================================================
    # F06 - STRONG PREDICTION DRIFT
    # =========================================================

    f06 = df.copy()

    f06["model_prediction"] = predictions

    f06.loc[
        cp:,
        "model_prediction"
    ] = (
        predictions[cp:] * 0.45
        + 8.0
    )

    f06["scenario"] = "F06_prediction_controlled"
    f06["change_point"] = cp
    f06["true_cause"] = "PREDICTION_DRIFT"
    f06["true_first_stage"] = "prediction"

    # =========================================================
    # F07 - PERFORMANCE-ONLY DRIFT
    # =========================================================

    f07 = df.copy(); f07["RUL"] = f07["RUL"].astype(float)

    # Keep the prediction distribution unchanged.
    # Instead, modify the target relationship so model error
    # increases while prediction distribution remains stable.
    f07["model_prediction"] = predictions

    target = f07["RUL"].astype(float).to_numpy()

    degradation = rng.normal(
        loc=0.0,
        scale=8.0,
        size=n - cp
    )

    f07.loc[
        cp:,
        "RUL"
    ] = target[cp:] + degradation

    f07["scenario"] = "F07_performance_only"
    f07["change_point"] = cp
    f07["true_cause"] = "PERFORMANCE_ONLY"
    f07["true_first_stage"] = "performance"

    # Save
    f05.to_csv(
        output_dir / "F05_prediction.csv",
        index=False
    )

    f06.to_csv(
        output_dir / "F06_prediction_controlled.csv",
        index=False
    )

    f07.to_csv(
        output_dir / "F07_performance_only.csv",
        index=False
    )

    print()
    print("=== UPDATED PREDICTION/PERFORMANCE LAB ===")
    print(f"Seed: {seed}")
    print(f"Change point: {cp}")
    print("F05: moderate prediction distribution shift")
    print("F06: strong prediction distribution shift")
    print("F07: prediction distribution retained; error degraded")
    print()
    print("Saved F05/F06/F07 successfully.")


if __name__ == "__main__":

    seed = (
        int(sys.argv[1])
        if len(sys.argv) > 1
        else 42
    )

    change_fraction = (
        float(sys.argv[2])
        if len(sys.argv) > 2
        else 0.50
    )

    main(
        seed,
        change_fraction
    )

