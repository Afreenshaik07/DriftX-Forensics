from pathlib import Path
import itertools
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "research_cases.csv"

SEEDS = [7, 21, 42, 84, 123, 156, 201, 256, 314, 512]
MAGNITUDES = [0.10, 0.20, 0.30, 0.40, 0.50]
ONSETS = [0.20, 0.35, 0.50, 0.65, 0.80]

SCENARIOS = [
    ("F01", "feature", "abrupt"),
    ("F02", "feature", "gradual"),
    ("F03", "feature", "multi_feature"),
    ("F04", "residual", "concept"),
    ("F05", "prediction", "prediction"),
    ("F06", "prediction", "prediction_controlled"),
    ("F07", "performance", "performance_only"),
    ("F08", "feature", "recurrent"),
    ("F09", "none", "no_drift"),
]

FEATURES = [
    "sensor_2",
    "sensor_3",
    "sensor_7",
    "sensor_8",
    "sensor_11",
]

rows = []

for seed, magnitude, onset, (scenario_id, origin, scenario_type) in itertools.product(
    SEEDS, MAGNITUDES, ONSETS, SCENARIOS
):
    rng = np.random.default_rng(seed + int(magnitude * 1000) + int(onset * 100))

    affected = (
        "none"
        if origin == "none"
        else str(rng.choice(FEATURES))
    )

    rows.append({
        "case_id": (
            f"{scenario_id}_S{seed}_"
            f"M{int(magnitude * 100):02d}_"
            f"O{int(onset * 100):02d}"
        ),
        "seed": seed,
        "scenario": scenario_id,
        "scenario_type": scenario_type,
        "true_origin": origin,
        "magnitude": magnitude,
        "onset_fraction": onset,
        "affected_feature": affected,
    })

df = pd.DataFrame(rows)

OUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT, index=False)

print("=== DRIFTX RESEARCH CASE GENERATOR ===")
print(f"Cases generated : {len(df)}")
print(f"Seeds           : {df['seed'].nunique()}")
print(f"Scenarios       : {df['scenario'].nunique()}")
print(f"Magnitudes      : {df['magnitude'].nunique()}")
print(f"Onsets          : {df['onset_fraction'].nunique()}")
print(f"Saved           : {OUT}")
print()
print(df.groupby(["scenario", "true_origin"]).size().to_string())
print()
print("RESEARCH CASES READY")
