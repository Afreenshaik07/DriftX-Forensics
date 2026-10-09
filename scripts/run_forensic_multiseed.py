from pathlib import Path
import subprocess
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / "data" / "interim"

seeds = [7, 21, 42, 84, 123]
all_results = []

print("=== DRIFTX-FORENSICS: MULTI-SEED FORENSIC ROBUSTNESS ===")

for seed in seeds:
    print(f"\n--- SEED {seed} ---")

    # Re-run the existing benchmark deterministically.
    # The current benchmark itself uses fixed scenario files, so this first
    # establishes the repeated baseline before introducing seed-dependent
    # scenario generation.
    result = subprocess.run(
        [str(Path(__file__).resolve().parents[1] / ".venv" / "Scripts" / "python.exe"), "-m", "scripts.run_forensic_benchmark"],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(result.stderr)
        raise SystemExit(result.returncode)

    df = pd.read_csv(ROOT / "results" / "forensic_full_benchmark.csv")
    df["seed"] = seed
    all_results.append(df)

combined = pd.concat(all_results, ignore_index=True)

# Safety-aware scoring:
# NO_DRIFT + unknown is treated as a correct abstention.
combined["safety_aware_correct"] = (
    (combined["predicted_origin"] == combined["true_first_stage"])
    |
    (
        (combined["true_cause"] == "NO_DRIFT")
        & (combined["predicted_origin"] == "unknown")
    )
)

combined["baseline_safety_aware_correct"] = (
    (
        (combined["baseline_origin"] == combined["true_first_stage"])
        & (combined["true_cause"] != "NO_DRIFT")
    )
    |
    (
        (combined["true_cause"] == "NO_DRIFT")
        & (combined["baseline_origin"].isin(["unknown", "none"]))
    )
)

summary = combined.groupby("seed").agg(
    forensic_accuracy=("safety_aware_correct", "mean"),
    baseline_accuracy=("baseline_safety_aware_correct", "mean"),
    mean_confidence=("forensic_confidence", "mean"),
)

print("\n=== MULTI-SEED SUMMARY ===")
print(summary)

print("\n=== AGGREGATE ===")
print(
    f"DriftX mean accuracy : {summary.forensic_accuracy.mean():.2%}"
)
print(
    f"DriftX std           : {summary.forensic_accuracy.std(ddof=1):.2%}"
)
print(
    f"Baseline mean        : {summary.baseline_accuracy.mean():.2%}"
)
print(
    f"Baseline std         : {summary.baseline_accuracy.std(ddof=1):.2%}"
)

output = ROOT / "results" / "forensic_multiseed.csv"
combined.to_csv(output, index=False)
print(f"\nSaved: {output}")
