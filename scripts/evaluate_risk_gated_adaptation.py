from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
src = ROOT / "results" / "streaming_failure_analysis.csv"
out = ROOT / "results" / "risk_gated_adaptation.csv"

df = pd.read_csv(src)

# Current experiment does not contain the forensic risk score,
# so use baseline error as a conservative proxy for adaptation need.
thresholds = np.quantile(
    df["champion_mae"],
    [0.25, 0.50, 0.75]
)

rows = []

for threshold in thresholds:
    use_adaptation = df["champion_mae"] >= threshold

    final_mae = np.where(
        use_adaptation,
        df["adapted_mae"],
        df["champion_mae"]
    )

    baseline = df["champion_mae"].to_numpy()

    rows.append({
        "threshold": float(threshold),
        "adapted_engines": int(use_adaptation.sum()),
        "retained_engines": int((~use_adaptation).sum()),
        "final_mae": float(final_mae.mean()),
        "baseline_mae": float(baseline.mean()),
        "improvement_percent": float(
            (baseline.mean() - final_mae.mean())
            / baseline.mean() * 100
        ),
        "harmful_adaptations_allowed": int(
            (
                use_adaptation &
                (df["adapted_mae"] >= df["champion_mae"])
            ).sum()
        )
    })

result = pd.DataFrame(rows)
result.to_csv(out, index=False)

print("=== RISK-GATED ADAPTATION EXPERIMENT ===")
print(result.to_string(index=False))
print()
print(f"Saved: {out}")
