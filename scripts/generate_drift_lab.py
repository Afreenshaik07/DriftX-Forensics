from pathlib import Path
import sys

import pandas as pd
import numpy as np

from src.driftx.config import CONFIG
from src.driftx.data.preprocessing import load_training_data, create_rul_target


def save_scenario(df, name, change_point, cause, first_stage):
    output = Path("data/interim") / f"{name}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)

    result = df.copy()
    result["scenario"] = name
    result["change_point"] = change_point
    result["true_cause"] = cause
    result["true_first_stage"] = first_stage

    result.to_csv(output, index=False)


def main(seed=42):

    rng = np.random.default_rng(seed)

    raw_dir = Path(CONFIG["paths"]["raw_data"])

    df = create_rul_target(
        load_training_data(raw_dir)
    )

    df = df.copy()

    # IMPORTANT: allow controlled floating-point drift in RUL
    if "RUL" in df.columns:
        df["RUL"] = pd.to_numeric(
            df["RUL"],
            errors="coerce"
        ).astype(float)

    # Seed controls row ordering
    df = df.sample(
        frac=1.0,
        random_state=seed
    ).reset_index(drop=True)

    n = len(df)

    change_fraction = float(
        rng.choice([
            0.30,
            0.40,
            0.50,
            0.60,
            0.70
        ])
    )

    cp = int(n * change_fraction)

    magnitude = float(
        rng.choice([
            0.10,
            0.20,
            0.30,
            0.40
        ])
    )

    feature_pool = [
        "sensor_2",
        "sensor_3",
        "sensor_7",
        "sensor_8",
        "sensor_11"
    ]

    feature_a, feature_b, feature_c = rng.choice(
        feature_pool,
        size=3,
        replace=False
    )

    print()
    print("=== DRIFT LABORATORY ===")
    print(f"Seed: {seed}")
    print(f"Rows: {n}")
    print(f"Change point: {cp}")
    print(f"Magnitude: {magnitude:.2f}")
    print(
        f"Features: {feature_a}, "
        f"{feature_b}, "
        f"{feature_c}"
    )
    print()

    # =========================================================
    # F01 - ABRUPT FEATURE DRIFT
    # =========================================================

    f01 = df.copy()

    f01.loc[
        cp:,
        feature_a
    ] *= 1.0 + magnitude

    save_scenario(
        f01,
        "F01_feature_abrupt",
        cp,
        "FEATURE_DRIFT",
        "feature"
    )

    # =========================================================
    # F02 - GRADUAL FEATURE DRIFT
    # =========================================================

    f02 = df.copy()

    strength = np.zeros(n)

    strength[cp:] = np.linspace(
        0.0,
        magnitude,
        n - cp
    )

    f02[feature_b] *= 1.0 + strength

    save_scenario(
        f02,
        "F02_feature_gradual",
        cp,
        "FEATURE_DRIFT",
        "feature"
    )

    # =========================================================
    # F03 - MULTI FEATURE DRIFT
    # =========================================================

    f03 = df.copy()

    for feature in [
        feature_a,
        feature_b,
        feature_c
    ]:

        f03.loc[
            cp:,
            feature
        ] *= 1.0 + magnitude

    save_scenario(
        f03,
        "F03_multi_feature",
        cp,
        "FEATURE_DRIFT",
        "feature"
    )

    # =========================================================
    # F04 - CONCEPT DRIFT
    # =========================================================

    f04 = df.copy()

    f04["RUL"] = f04["RUL"].astype(float)

    original_rul = f04["RUL"].copy()

    concept_strength = rng.uniform(
        0.55,
        0.80
    )

    new_rul = (
        original_rul.loc[cp:] * concept_strength
        +
        f04.loc[
            cp:,
            feature_a
        ].rank(pct=True) * 10.0
    )

    f04.loc[
        cp:,
        "RUL"
    ] = new_rul.to_numpy()

    f04["concept_drift_active"] = 0

    f04.loc[
        cp:,
        "concept_drift_active"
    ] = 1

    save_scenario(
        f04,
        "F04_concept",
        cp,
        "CONCEPT_DRIFT",
        "residual"
    )

    # =========================================================
    # F05 - PREDICTION DRIFT
    # =========================================================

    f05 = df.copy()

    f05["RUL"] = f05["RUL"].astype(float)

    prediction_noise = rng.normal(
        loc=0.0,
        scale=max(
            1.0,
            magnitude * 8.0
        ),
        size=n - cp
    )

    f05.loc[
        cp:,
        "RUL"
    ] += prediction_noise

    save_scenario(
        f05,
        "F05_prediction",
        cp,
        "PREDICTION_DRIFT",
        "prediction"
    )

    # =========================================================
    # F06 - CONTROLLED PREDICTION DRIFT
    # =========================================================

    f06 = df.copy()

    f06["RUL"] = f06["RUL"].astype(float)

    strong_noise = rng.normal(
        loc=magnitude * 8.0,
        scale=max(
            1.5,
            magnitude * 5.0
        ),
        size=n - cp
    )

    f06.loc[
        cp:,
        "RUL"
    ] += strong_noise

    save_scenario(
        f06,
        "F06_prediction_controlled",
        cp,
        "PREDICTION_DRIFT",
        "prediction"
    )

    # =========================================================
    # F07 - PERFORMANCE ONLY
    # =========================================================

    f07 = df.copy()

    f07["RUL"] = f07["RUL"].astype(float)

    performance_noise = rng.normal(
        loc=0.0,
        scale=max(
            0.5,
            magnitude * 10.0
        ),
        size=n - cp
    )

    f07.loc[
        cp:,
        "RUL"
    ] += performance_noise

    save_scenario(
        f07,
        "F07_performance_only",
        cp,
        "PERFORMANCE_ONLY",
        "performance"
    )

    # =========================================================
    # F08 - RECURRENT FEATURE DRIFT
    # =========================================================

    f08 = df.copy()

    cp2 = int(
        cp
        +
        (n - cp)
        *
        rng.uniform(
            0.40,
            0.75
        )
    )

    f08.loc[
        cp:cp2,
        feature_a
    ] *= 1.0 + magnitude

    f08.loc[
        cp2:,
        feature_a
    ] /= 1.0 + magnitude

    save_scenario(
        f08,
        "F08_recurrent",
        cp,
        "RECURRENT_DRIFT",
        "feature"
    )

    # =========================================================
    # F09 - NO DRIFT
    # =========================================================

    f09 = df.copy()

    save_scenario(
        f09,
        "F09_no_drift",
        None,
        "NO_DRIFT",
        "none"
    )

    print()
    print(
        "Drift Laboratory generated successfully."
    )


if __name__ == "__main__":

    seed = 42

    if len(sys.argv) > 1:
        seed = int(sys.argv[1])

    main(seed)
