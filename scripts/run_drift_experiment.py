from pathlib import Path
from src.driftx.config import CONFIG
from src.driftx.data.preprocessing import load_training_data
from src.driftx.data.drift_lab import inject_abrupt_drift


def main():
    raw_dir = Path(CONFIG["paths"]["raw_data"])

    df = load_training_data(raw_dir)

    feature = "sensor_2"

    drifted_df = inject_abrupt_drift(
        df,
        feature=feature,
        magnitude=0.30,
        change_point=0.50
    )

    print("Drift experiment created successfully.")
    print(f"Rows: {len(drifted_df)}")
    print(f"Injected feature: {feature}")
    print("Drift type: Abrupt")
    print("Magnitude: 30%")
    print("Change point: 50%")


if __name__ == "__main__":
    main()
