from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
src = ROOT / "results" / "final_forensic_benchmark.csv"
out = ROOT / "results" / "driftx_vs_temporal_baseline.csv"

df = pd.read_csv(src)

df["driftx_correct"] = df["first_stage_correct"].astype(bool)
df["baseline_correct"] = df["baseline_correct"].astype(bool)

driftx_accuracy = df["driftx_correct"].mean()
baseline_accuracy = df["baseline_correct"].mean()

summary = pd.DataFrame([
    {
        "method": "DriftX Forensics",
        "correct": int(df["driftx_correct"].sum()),
        "total": len(df),
        "accuracy": driftx_accuracy,
        "accuracy_percent": driftx_accuracy * 100,
    },
    {
        "method": "Temporal-Magnitude Baseline",
        "correct": int(df["baseline_correct"].sum()),
        "total": len(df),
        "accuracy": baseline_accuracy,
        "accuracy_percent": baseline_accuracy * 100,
    },
])

summary.to_csv(out, index=False)

print("=== VALID FORENSIC COMPARISON ===")
print(summary.to_string(index=False))

print()
print(
    f"DriftX advantage: "
    f"{(driftx_accuracy - baseline_accuracy) * 100:.2f} percentage points"
)

print()
print("=== PER-CASE RESULTS ===")
print(
    df[
        [
            "scenario",
            "true_first_stage",
            "predicted_first_stage",
            "baseline_origin",
            "driftx_correct",
            "baseline_correct",
        ]
    ].to_string(index=False)
)

print()
print(f"Saved: {out}")
