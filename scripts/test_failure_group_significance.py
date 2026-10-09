from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
src = ROOT / "results" / "streaming_failure_analysis.csv"

df = pd.read_csv(src)

improved = df.loc[df["outcome"] == "IMPROVED", "champion_mae"].to_numpy()
worsened = df.loc[df["outcome"] == "WORSENED", "champion_mae"].to_numpy()

observed = improved.mean() - worsened.mean()

rng = np.random.default_rng(42)
combined = np.concatenate([improved, worsened])
n_i = len(improved)

null = []

for _ in range(10000):
    shuffled = rng.permutation(combined)
    null.append(
        shuffled[:n_i].mean() -
        shuffled[n_i:].mean()
    )

p_value = np.mean(np.abs(null) >= abs(observed))

print("=== DRIFTX FAILURE-GROUP ANALYSIS ===")
print(f"Improved engines       : {len(improved)}")
print(f"Worsened engines       : {len(worsened)}")
print(f"Mean champion MAE      : {improved.mean():.4f} improved group")
print(f"Mean champion MAE      : {worsened.mean():.4f} worsened group")
print(f"Observed difference    : {observed:.4f}")
print(f"Permutation p-value    : {p_value:.4f}")

if p_value < 0.05:
    print("Finding: baseline error differs significantly between groups.")
else:
    print("Finding: insufficient evidence of a significant group difference.")
