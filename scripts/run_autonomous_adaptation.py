from pathlib import Path
import pandas as pd

from src.driftx.evaluation.adaptation_policy import decide_adaptation
from src.driftx.evaluation.adaptation_controller import execute_adaptation
from src.driftx.models.model_registry import ModelRegistry
from src.driftx.memory.regime_memory import RegimeMemory


def main():
    benchmark = Path("results/forensic_full_benchmark.csv")
    baseline_model = Path("artifacts/baseline_rul_model.joblib")

    if not benchmark.exists():
        raise FileNotFoundError(benchmark)

    if not baseline_model.exists():
        raise FileNotFoundError(baseline_model)

    df = pd.read_csv(benchmark)

    registry = ModelRegistry(
        registry_dir=Path("artifacts/registry")
    )

    memory = RegimeMemory(
        memory_path="artifacts/regime_memory.json"
    )

    init_result = registry.initialize(
        baseline_model,
        reason="Initial DriftX champion"
    )

    print("\n=== DRIFTX AUTONOMOUS ADAPTATION + REGIME MEMORY ===")
    print(f"Champion: {init_result['champion_path']}")
    print(f"Existing regimes: {memory.count()}\n")

    results = []

    for _, row in df.iterrows():

        drift_type = str(row["true_cause"])

        signature = {
            "feature_drift": float(row["feature_ks"]),
            "prediction_drift": float(row["prediction_ks"]),
            "residual_drift": float(row["residual_ks"]),
            "performance_degradation": float(row["mae_change_pct"]) / 100.0,
        }

        previous = memory.get_best_match(signature)

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
            drift_type=drift_type,
        )

        regime_id = f"REGIME_{len(memory.memory) + 1:03d}"

        memory.remember(
            regime_id=regime_id,
            signature=signature,
            origin=str(row["predicted_origin"]),
            drift_type=drift_type,
            action=decision.action,
            confidence=float(row["forensic_confidence"]),
            risk_score=float(decision.risk_score),
            model_version="champion",
            adapter="residual_adapter"
            if decision.action == "UPDATE"
            else "none",
        )

        results.append({
            "scenario": row["scenario"],
            "true_cause": drift_type,
            "predicted_origin": row["predicted_origin"],
            "confidence": row["forensic_confidence"],
            "action": decision.action,
            "severity": decision.severity,
            "risk_score": decision.risk_score,
            "reason": decision.reason,
            "previous_regime": (
                previous["regime_id"] if previous else "NONE"
            ),
            "regime_distance": (
                previous["distance"] if previous else None
            ),
        })

        memory_status = (
            f"MATCH={previous['regime_id']}"
            if previous
            else "NEW"
        )

        print(
            f"{row['scenario']:<28} "
            f"{decision.action:<8} "
            f"{decision.severity:<6} "
            f"risk={decision.risk_score:.3f} "
            f"{memory_status}"
        )

    output = pd.DataFrame(results)

    output.to_csv(
        "results/autonomous_adaptation_experiment.csv",
        index=False,
    )

    print("\n=== SUMMARY ===")
    print(output["action"].value_counts().to_string())
    print(f"\nRegimes stored: {memory.count()}")
    print("Saved: results/autonomous_adaptation_experiment.csv")
    print("Saved: artifacts/regime_memory.json")


if __name__ == "__main__":
    main()


