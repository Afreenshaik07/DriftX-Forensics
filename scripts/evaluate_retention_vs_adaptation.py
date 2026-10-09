from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
src = ROOT / "results" / "streaming_failure_analysis.csv"
out = ROOT / "results" / "retention_vs_adaptation.csv"

df = pd.read_csv(src)

df["retain_mae"] = df["champion_mae"]
df["adapt_mae"] = df["adapted_mae"]

df["best_action"] = np.where(
    df["adapt_mae"] < df["retain_mae"],
    "ADAPT",
    "RETAIN"
)

df["avoidable_loss"] = np.maximum(
    df["adapt_mae"] - df["retain_mae"],
    0.0
)

summary = {
    "engines": len(df),
    "adapt_better": int((df["best_action"] == "ADAPT").sum()),
    "retain_better": int((df["best_action"] == "RETAIN").sum()),
    "mean_retain_mae": float(df["retain_mae"].mean()),
    "mean_adapt_mae": float(df["adapt_mae"].mean()),
    "mean_avoidable_loss": float(df["avoidable_loss"].mean())
}

df.to_csv(out, index=False)

print("=== CHAMPION RETENTION VS ADAPTATION ===")
print(f"Engines          : {summary['engines']}")
print(f"Adapt better     : {summary['adapt_better']}")
print(f"Retain better    : {summary['retain_better']}")
print(f"Mean retain MAE  : {summary['mean_retain_mae']:.4f}")
print(f"Mean adapt MAE   : {summary['mean_adapt_mae']:.4f}")
print(f"Avoidable loss   : {summary['mean_avoidable_loss']:.4f}")
print(f"Saved            : {out}")
