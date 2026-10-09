from pathlib import Path
import pandas as pd

from src.driftx.evaluation.adaptation_policy import decide_adaptation


def main():
    path = Path("results/forensic_full_benchmark.csv")
    df = pd.read_csv(path)

    results = []

    print("\n=== DRIFTX: FORENSICS -> ADAPTATION POLICY ===")

    for _, row in df.iterrows():
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

        print(
            f"{row['scenario']:<28} "
            f"origin={row['predicted_origin']:<12} "
            f"action={decision.action:<8} "
            f"severity={decision.severity}"
        )

        results.append({
            "scenario": row["scenario"],
            "origin": row["predicted_origin"],
            "confidence": row["forensic_confidence"],
            "action": decision.action,
            "severity": decision.severity,
            "risk_score": decision.risk_score,
            "reason": decision.reason,
        })

    pd.DataFrame(results).to_csv(
        "results/end_to_end_adaptation.csv",
        index=False,
    )

    print("\nSaved: results/end_to_end_adaptation.csv")


if __name__ == "__main__":
    main()
