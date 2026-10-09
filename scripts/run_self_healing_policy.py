from pathlib import Path
import pandas as pd

from src.driftx.evaluation.adaptation_policy import decide_adaptation


def execute_policy(
    row,
    champion_predictions=None,
):
    decision = decide_adaptation(
        predicted_origin=str(row["predicted_origin"]),
        confidence=float(row["forensic_confidence"]),
        impact_score=float(row["performance_score"]),
        performance_degradation=float(row["mae_change_pct"]) / 100.0,
        drift_strength=max(
            float(row["feature_ks"]),
            float(row["prediction_ks"]),
            float(row["residual_ks"]),
        ),
        drift_type=str(row["true_cause"]),
    )

    if decision.action == "UPDATE":
        execution = "RESIDUAL_ADAPTER"

    elif decision.action == "RETRAIN":
        execution = "CHALLENGER_SAFETY_GATE"

    else:
        execution = "MONITOR_ONLY"

    return decision, execution


def main():
    path = Path("results/forensic_full_benchmark.csv")
    df = pd.read_csv(path)

    results = []

    print("\n=== DRIFTX: AUTONOMOUS SELF-HEALING POLICY ===")

    for _, row in df.iterrows():

        decision, execution = execute_policy(row)

        print(
            f"{row['scenario']:<28} "
            f"{decision.action:<8} "
            f"-> {execution}"
        )

        results.append({
            "scenario": row["scenario"],
            "origin": row["predicted_origin"],
            "drift_type": row["true_cause"],
            "action": decision.action,
            "severity": decision.severity,
            "risk_score": decision.risk_score,
            "execution": execution,
        })

    output = pd.DataFrame(results)

    output.to_csv(
        "results/self_healing_policy.csv",
        index=False,
    )

    print("\nSaved: results/self_healing_policy.csv")


if __name__ == "__main__":
    main()
