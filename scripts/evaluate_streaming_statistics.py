from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
src = ROOT / "results" / "streaming_online_adaptation.csv"
out = ROOT / "results" / "streaming_statistical_evaluation.csv"

df = pd.read_csv(src)

champ = df["champion_mae"].astype(float).to_numpy()
adapt = df["adapted_mae"].astype(float).to_numpy()

diff = champ - adapt
pct = diff / np.maximum(champ, 1e-12) * 100.0

rng = np.random.default_rng(42)
boot = []

for _ in range(10000):
    sample = rng.choice(diff, size=len(diff), replace=True)
    boot.append(sample.mean())

ci_low, ci_high = np.percentile(boot, [2.5, 97.5])

result = pd.DataFrame([{
    "test_engines": len(diff),
    "mean_mae_improvement": diff.mean(),
    "median_mae_improvement": np.median(diff),
    "std_mae_improvement": diff.std(ddof=1),
    "mean_improvement_percent": pct.mean(),
    "improved_engines": int((diff > 0).sum()),
    "worsened_engines": int((diff < 0).sum()),
    "unchanged_engines": int((diff == 0).sum()),
    "bootstrap_95ci_low": ci_low,
    "bootstrap_95ci_high": ci_high
}])

out.parent.mkdir(parents=True, exist_ok=True)
result.to_csv(out, index=False)

print("=== DRIFTX STREAMING STATISTICAL EVALUATION ===")
print(f"Test engines              : {len(diff)}")
print(f"Improved engines          : {(diff > 0).sum()}")
print(f"Worsened engines          : {(diff < 0).sum()}")
print(f"Mean MAE improvement      : {diff.mean():.4f}")
print(f"Median MAE improvement    : {np.median(diff):.4f}")
print(f"Mean improvement          : {pct.mean():.2f}%")
print(f"Bootstrap 95% CI          : [{ci_low:.4f}, {ci_high:.4f}]")
print(f"Saved                     : {out}")
