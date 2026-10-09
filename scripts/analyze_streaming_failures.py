from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
src = ROOT / "results" / "streaming_online_adaptation.csv"
out = ROOT / "results" / "streaming_failure_analysis.csv"

df = pd.read_csv(src)

df["mae_change"] = df["champion_mae"] - df["adapted_mae"]
df["improvement_percent"] = (
    df["mae_change"] /
    df["champion_mae"].replace(0, np.nan)
) * 100

df["outcome"] = np.where(
    df["mae_change"] > 0,
    "IMPROVED",
    np.where(df["mae_change"] < 0, "WORSENED", "UNCHANGED")
)

summary = (
    df.groupby("outcome")
      .agg(
          engines=("outcome", "size"),
          mean_champion_mae=("champion_mae", "mean"),
          mean_adapted_mae=("adapted_mae", "mean"),
          mean_improvement=("mae_change", "mean"),
          mean_improvement_percent=("improvement_percent", "mean")
      )
      .reset_index()
)

df.to_csv(out, index=False)

print("=== DRIFTX STREAMING FAILURE ANALYSIS ===")
print()
print(summary.to_string(index=False))
print()
print("ENGINE-LEVEL RESULTS")
cols = [
    c for c in [
        "engine_id",
        "champion_mae",
        "adapted_mae",
        "mae_change",
        "improvement_percent",
        "outcome"
    ] if c in df.columns
]
print(df[cols].to_string(index=False))
print()
print(f"Saved: {out}")
