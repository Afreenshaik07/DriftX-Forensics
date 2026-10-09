from pathlib import Path
import pandas as pd
import numpy as np

from src.driftx.evaluation.safety_gate import evaluate_safety_gate

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "safety_gate_stress_test.csv"

champion = {
    "mae": 13.834849387728351,
    "rmse": 17.317460100929356,
    "r2": 0.028426118344511298,
}

tests = [
    {
        "test": "GOOD_CHALLENGER",
        "mae": 12.4474,
        "rmse": 15.8138,
        "r2": 0.2322,
    },
    {
        "test": "MAE_WORSE",
        "mae": 15.50,
        "rmse": 18.50,
        "r2": 0.02,
    },
    {
        "test": "RMSE_WORSE",
        "mae": 12.50,
        "rmse": 20.00,
        "r2": 0.20,
    },
    {
        "test": "R2_DROP",
        "mae": 12.50,
        "rmse": 16.00,
        "r2": -0.20,
    },
    {
        "test": "SEVERELY_BAD",
        "mae": 25.00,
        "rmse": 35.00,
        "r2": -0.50,
    },
]

rows = []

for case in tests:
    challenger = {
        "mae": case["mae"],
        "rmse": case["rmse"],
        "r2": case["r2"],
    }

    decision = evaluate_safety_gate(
        champion,
        challenger,
        min_improvement=0.0,
        max_r2_drop=0.02,
        max_rmse_increase=0.0,
    )

    rows.append({
        "test": case["test"],
        "champion_mae": champion["mae"],
        "challenger_mae": challenger["mae"],
        "champion_rmse": champion["rmse"],
        "challenger_rmse": challenger["rmse"],
        "champion_r2": champion["r2"],
        "challenger_r2": challenger["r2"],
        "decision": decision,
    })

result = pd.DataFrame(rows)
result.to_csv(OUT, index=False)

print("=== DRIFTX SAFETY-GATE STRESS TEST ===")
print(result[
    [
        "test",
        "challenger_mae",
        "challenger_rmse",
        "challenger_r2",
        "decision",
    ]
].to_string(index=False))

print()
print("PROMOTED:", int((result["decision"] == "PROMOTE").sum()))
print("REJECTED :", int((result["decision"] == "REJECT").sum()))
print(f"Saved: {OUT}")
