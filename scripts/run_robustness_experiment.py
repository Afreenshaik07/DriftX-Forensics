import subprocess
import sys

SEEDS = [7, 21, 42]
for seed in SEEDS:
    print(f"\n{'='*18} ROBUSTNESS SEED {seed} {'='*18}")
    subprocess.run([sys.executable, "-m", "scripts.generate_drift_lab", str(seed)], check=True)
    subprocess.run([sys.executable, "-m", "scripts.generate_prediction_scenarios"], check=True)
    subprocess.run([sys.executable, "-m", "scripts.run_forensic_benchmark"], check=True)
