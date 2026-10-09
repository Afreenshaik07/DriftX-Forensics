from pathlib import Path
import json
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

def load_csv(name):
    path = RESULTS / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)

def pct(x):
    return round(float(x) * 100, 2)

print("=" * 70)
print("DRIFTX-FORENSICS - RESEARCH EVIDENCE CONSOLIDATION")
print("=" * 70)

# ---------------------------------------------------------
# 1. FORENSIC BENCHMARK
# ---------------------------------------------------------
forensic = load_csv("forensic_full_benchmark.csv")

if not forensic.empty:
    if "first_stage_correct" in forensic:
        forensic_accuracy = forensic["first_stage_correct"].mean()
    elif "safety_aware_correct" in forensic:
        forensic_accuracy = forensic["safety_aware_correct"].mean()
    else:
        forensic_accuracy = np.nan

    if "baseline_correct" in forensic:
        baseline_accuracy = forensic["baseline_correct"].mean()
    elif "baseline_safety_aware_correct" in forensic:
        baseline_accuracy = forensic["baseline_safety_aware_correct"].mean()
    else:
        baseline_accuracy = np.nan

    print("\n[1] FORENSIC DIAGNOSIS")
    print(f"DriftX accuracy      : {pct(forensic_accuracy):.2f}%")
    print(f"Temporal baseline    : {pct(baseline_accuracy):.2f}%")
else:
    forensic_accuracy = np.nan
    baseline_accuracy = np.nan
    print("\n[1] FORENSIC DIAGNOSIS : data unavailable")

# ---------------------------------------------------------
# 2. ADAPTATION ABLATION
# ---------------------------------------------------------
adapt = load_csv("adaptation_ablation.csv")

if not adapt.empty:
    print("\n[2] ADAPTATION ABLATION")

    if "strategy" in adapt.columns:
        summary = (
            adapt.groupby("strategy")[["mae", "rmse", "r2"]]
            .mean()
            .sort_values("mae")
        )

        print(summary.to_string())

        if "DRIFTX" in summary.index and "STATIC" in summary.index:
            mae_gain = (
                (summary.loc["STATIC", "mae"] -
                 summary.loc["DRIFTX", "mae"])
                / summary.loc["STATIC", "mae"]
            )
            print(
                f"DriftX MAE improvement vs STATIC: "
                f"{pct(mae_gain):.2f}%"
            )

        if "DRIFTX" in summary.index and "NAIVE" in summary.index:
            mae_gain_naive = (
                (summary.loc["NAIVE", "mae"] -
                 summary.loc["DRIFTX", "mae"])
                / summary.loc["NAIVE", "mae"]
            )
            print(
                f"DriftX MAE improvement vs NAIVE : "
                f"{pct(mae_gain_naive):.2f}%"
            )
else:
    summary = pd.DataFrame()
    print("\n[2] ADAPTATION ABLATION : data unavailable")

# ---------------------------------------------------------
# 3. STREAMING ADAPTATION
# ---------------------------------------------------------
stream = load_csv("streaming_online_adaptation.csv")

if not stream.empty:
    print("\n[3] STREAMING ADAPTATION")

    numeric = stream.select_dtypes(include=np.number)

    if "champion_mae" in stream.columns and "adapted_mae" in stream.columns:
        champion_mae = stream["champion_mae"].mean()
        adapted_mae = stream["adapted_mae"].mean()

        improvement = (
            (champion_mae - adapted_mae)
            / champion_mae
        )

        print(f"Champion MAE : {champion_mae:.4f}")
        print(f"Adapted MAE  : {adapted_mae:.4f}")
        print(f"Improvement  : {pct(improvement):.2f}%")
else:
    champion_mae = adapted_mae = improvement = np.nan
    print("\n[3] STREAMING ADAPTATION : data unavailable")

# ---------------------------------------------------------
# 4. REGIME MEMORY
# ---------------------------------------------------------
memory_path = ROOT / "artifacts" / "regime_memory.json"

print("\n[4] REGIME MEMORY")

if memory_path.exists():
    memory = json.loads(memory_path.read_text())
    regimes = (
        memory.get("regimes", [])
        if isinstance(memory, dict)
        else memory
    )
    print(f"Stored regimes : {len(regimes)}")
else:
    print("Memory artifact unavailable")

# ---------------------------------------------------------
# 5. CREATE MASTER EVIDENCE FILE
# ---------------------------------------------------------
master = {
    "forensic_accuracy": None if np.isnan(forensic_accuracy) else float(forensic_accuracy),
    "temporal_baseline_accuracy": None if np.isnan(baseline_accuracy) else float(baseline_accuracy),
    "streaming_champion_mae": None if np.isnan(champion_mae) else float(champion_mae),
    "streaming_adapted_mae": None if np.isnan(adapted_mae) else float(adapted_mae),
    "streaming_mae_improvement": None if np.isnan(improvement) else float(improvement),
    "adaptation_ablation_available": not adapt.empty,
    "forensic_benchmark_available": not forensic.empty,
    "streaming_benchmark_available": not stream.empty,
    "regime_memory_available": memory_path.exists()
}

output = RESULTS / "MASTER_RESEARCH_EVIDENCE.json"
output.write_text(json.dumps(master, indent=2))

print("\n" + "=" * 70)
print(f"MASTER EVIDENCE SAVED: {output}")
print("RESEARCH EVIDENCE CONSOLIDATION COMPLETE")
print("=" * 70)

