from pathlib import Path
import subprocess
import sys
import os
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

SEEDS = [7, 21, 42, 84, 123, 156, 201, 256, 314, 512]

rows = []

print("=== CORRECTED REAL MULTI-SEED FORENSIC BENCHMARK ===")

for seed in SEEDS:

    print(f"\n--- SEED {seed} ---")

    env = {**os.environ, "PYTHONPATH": "."}

    subprocess.run(
        [sys.executable, "scripts/generate_drift_lab.py", str(seed)],
        cwd=ROOT,
        env=env,
        check=True,
        stdout=subprocess.DEVNULL
    )

    subprocess.run(
        [sys.executable, "scripts/run_forensic_benchmark.py"],
        cwd=ROOT,
        env=env,
        check=True,
        stdout=subprocess.DEVNULL
    )

    source = RESULTS / "forensic_full_benchmark.csv"

    if not source.exists():
        raise FileNotFoundError(source)

    df = pd.read_csv(source)

    df["seed"] = seed

    df.to_csv(
        RESULTS / f"forensic_benchmark_seed_{seed}.csv",
        index=False
    )

    driftx = df["first_stage_correct"].astype(bool).mean()
    baseline = df["baseline_correct"].astype(bool).mean()

    rows.append({
        "seed": seed,
        "cases": len(df),
        "driftx_accuracy": driftx,
        "baseline_accuracy": baseline,
        "advantage_pp": (driftx - baseline) * 100
    })

    print(
        f"DriftX={driftx*100:.2f}% | "
        f"Baseline={baseline*100:.2f}% | "
        f"Advantage={(driftx-baseline)*100:+.2f} pp"
    )

summary = pd.DataFrame(rows)

summary.to_csv(
    RESULTS / "multi_seed_forensic_results.csv",
    index=False
)

print("\n=== FINAL REAL MULTI-SEED RESULT ===")
print(summary.to_string(index=False))

print()
print(f"Mean DriftX accuracy: {summary['driftx_accuracy'].mean()*100:.2f}%")
print(f"Mean baseline accuracy: {summary['baseline_accuracy'].mean()*100:.2f}%")
print(f"Mean advantage: {summary['advantage_pp'].mean():+.2f} pp")
print(f"Total real cases: {int(summary['cases'].sum())}")

print()
print("Saved:", RESULTS / "multi_seed_forensic_results.csv")
