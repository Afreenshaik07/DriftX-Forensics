from pathlib import Path
import pandas as pd
import numpy as np
import joblib

ROOT = Path(__file__).resolve().parents[1]

def main():
    source = pd.read_csv(
        ROOT / "data" / "interim" / "F01_abrupt_sensor2.csv"
    )

    # Remove the previous feature-drift effect by using the original
    # feature values and preserving their order.
    df = source.copy()

    # Ensure deterministic ordering.
    df = df.reset_index(drop=True)

    midpoint = len(df) // 2

    # Keep ALL input features unchanged.
    # Only the target relationship changes after the known change point.
    original_rul = df["RUL"].to_numpy(dtype=float)

    changed_rul = original_rul.copy()

    post_drift = np.arange(len(df)) >= midpoint

    # Controlled concept drift:
    # the true target relationship changes while sensor distributions
    # remain unchanged.
    changed_rul[post_drift] = (
        original_rul[post_drift] * 0.65
        + df.loc[post_drift, "sensor_2"].rank(pct=True).to_numpy() * 10
    )

    df["RUL"] = changed_rul

    # Ground-truth metadata for the experiment.
    df["concept_drift_active"] = post_drift.astype(int)

    model = joblib.load(
        ROOT / "artifacts" / "baseline_rul_model.joblib"
    )

    features = (
        ["cycle", "setting_1", "setting_2", "setting_3"]
        + [c for c in df.columns if c.startswith("sensor_")]
    )

    df["model_prediction"] = model.predict(df[features])

    output = ROOT / "data" / "interim" / "F04_temporal_concept.csv"
    df.to_csv(output, index=False)

    print()
    print("=== F04 TEMPORAL CONCEPT-DRIFT LAB ===")
    print("Created:", output)
    print("Rows:", len(df))
    print("Ground-truth change point:", midpoint)
    print("Drift begins at row:", midpoint)
    print("Feature values changed: NO")
    print("Target relationship changed: YES")
    print("Model predictions added: YES")
    print("Ground-truth drift column added: YES")

if __name__ == "__main__":
    main()
