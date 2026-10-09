from pathlib import Path

import pandas as pd

from src.driftx.config import CONFIG
from src.driftx.data.preprocessing import load_training_data, create_rul_target


FEATURES = (
    ["cycle", "setting_1", "setting_2", "setting_3"]
    + [f"sensor_{i}" for i in range(1, 22)]
)


def inject_temporal_feature_drift(
    df: pd.DataFrame,
    feature: str = "sensor_2",
    magnitude: float = 0.30,
    drift_fraction: float = 0.50,
):
    """
    Apply feature drift after the lifecycle midpoint
    independently for every engine.

    Each engine therefore has:
        pre-drift regime  -> post-drift regime
    """

    result = df.copy()

    result["drift_active"] = 0

    for unit in result["unit"].unique():

        mask = result["unit"] == unit

        unit_rows = result.loc[mask].sort_values("cycle")

        if len(unit_rows) < 2:
            continue

        change_index = int(
            len(unit_rows) * drift_fraction
        )

        drift_indices = unit_rows.index[change_index:]

        result.loc[
            drift_indices,
            feature
        ] = (
            result.loc[
                drift_indices,
                feature
            ]
            * (1.0 + magnitude)
        )

        result.loc[
            drift_indices,
            "drift_active"
        ] = 1

    return result


def main():

    print(
        "=== DRIFTX TEMPORAL DRIFT LAB ==="
    )

    raw_dir = Path(
        CONFIG["paths"]["raw_data"]
    )

    df = create_rul_target(
        load_training_data(raw_dir)
    )

    drifted = inject_temporal_feature_drift(
        df,
        feature="sensor_2",
        magnitude=0.30,
        drift_fraction=0.50,
    )

    output_dir = Path(
        CONFIG["paths"]["processed_data"]
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / "FD001_temporal_sensor2_drift.csv"
    )

    drifted.to_csv(
        output_path,
        index=False,
    )

    print()
    print(
        f"Engines processed : "
        f"{drifted['unit'].nunique()}"
    )

    print(
        f"Total rows         : "
        f"{len(drifted)}"
    )

    print(
        f"Drifted rows       : "
        f"{int(drifted['drift_active'].sum())}"
    )

    print(
        f"Stable rows        : "
        f"{int((drifted['drift_active'] == 0).sum())}"
    )

    print()
    print(
        "Drift protocol     : "
        "50% lifecycle per engine"
    )

    print(
        "Affected feature   : sensor_2"
    )

    print(
        "Magnitude          : 30%"
    )

    print()
    print(
        f"Saved: {output_path.resolve()}"
    )


if __name__ == "__main__":
    main()
