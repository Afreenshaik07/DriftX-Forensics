from pathlib import Path

import pandas as pd

from src.driftx.evaluation.adaptation_policy import (
    evaluate_forensic_result,
)


RESULT_FILE = Path("results/forensic_full_benchmark.csv")
OUTPUT_FILE = Path("results/adaptation_benchmark.csv")


def clip(value):
    return max(0.0, min(1.0, float(value)))


def normalize_percentage(value):
    """
    Convert percentage values such as 89.15 to 0.8915.
    Values already between 0 and 1 remain unchanged.
    """
    value = float(value)

    if value > 1.0:
        value /= 100.0

    return clip(value)


def calculate_drift_strength(row):
    """
    Estimate overall drift strength using the strongest
    observed stage-level drift signal.
    """

    stage_values = []

    for column in [
        "feature_ks",
        "prediction_ks",
        "residual_ks",
    ]:
        if column in row and pd.notna(row[column]):
            stage_values.append(
                normalize_percentage(row[column])
            )

    if not stage_values:
        return 0.0

    return clip(max(stage_values))


def calculate_performance_degradation(row):
    """
    Estimate normalized performance degradation.
    """

    if (
        "mae_change_pct" in row
        and pd.notna(row["mae_change_pct"])
    ):
        return normalize_percentage(
            row["mae_change_pct"]
        )

    if (
        "reference_mae" in row
        and "current_mae" in row
        and pd.notna(row["reference_mae"])
        and pd.notna(row["current_mae"])
    ):
        reference = float(row["reference_mae"])
        current = float(row["current_mae"])

        if reference <= 0:
            return 0.0

        degradation = (
            (current - reference)
            / reference
        )

        return clip(degradation)

    return 0.0


def calculate_impact(row):
    """
    Estimate operational impact.

    Performance degradation receives higher weight because
    actual model failure is the main operational consequence.
    """

    performance = (
        calculate_performance_degradation(row)
    )

    downstream_values = []

    for column in [
        "prediction_ks",
        "residual_ks",
        "performance_score",
    ]:

        if column not in row:
            continue

        if pd.isna(row[column]):
            continue

        value = float(row[column])

        if column == "performance_score":
            value = clip(value)
        else:
            value = normalize_percentage(value)

        downstream_values.append(value)

    downstream = (
        max(downstream_values)
        if downstream_values
        else 0.0
    )

    return clip(
        0.70 * performance
        + 0.30 * downstream
    )


def main():

    print(
        "=== DRIFTX-FORENSICS: ADAPTATION BENCHMARK ==="
    )

    if not RESULT_FILE.exists():
        raise FileNotFoundError(
            f"Benchmark result not found: {RESULT_FILE}"
        )

    df = pd.read_csv(RESULT_FILE)

    results = []

    for _, row in df.iterrows():

        # -----------------------------------------------------
        # Forensic diagnosis
        # -----------------------------------------------------

        predicted_origin = row.get(
            "predicted_first_stage",
            row.get(
                "predicted_origin",
                "unknown",
            ),
        )

        confidence = float(
            row.get(
                "forensic_confidence",
                0.0,
            )
        )

        forensic_result = {
            "predicted_origin": predicted_origin,
            "confidence": confidence,
        }

        # -----------------------------------------------------
        # Drift type / cause
        # -----------------------------------------------------

        drift_type = str(
            row.get(
                "true_cause",
                "UNKNOWN",
            )
        ).upper()

        # -----------------------------------------------------
        # Evidence calculations
        # -----------------------------------------------------

        drift_strength = (
            calculate_drift_strength(row)
        )

        performance_degradation = (
            calculate_performance_degradation(row)
        )

        impact_score = calculate_impact(row)

        # -----------------------------------------------------
        # Adaptation policy
        # -----------------------------------------------------

        decision = evaluate_forensic_result(
            forensic_result=forensic_result,
            impact_score=impact_score,
            performance_degradation=(
                performance_degradation
            ),
            drift_strength=drift_strength,
            drift_type=drift_type,
        )

        # -----------------------------------------------------
        # Store result
        # -----------------------------------------------------

        results.append(
            {
                "scenario": row["scenario"],

                "true_cause": drift_type,

                "true_first_stage": row.get(
                    "true_first_stage",
                    "unknown",
                ),

                "predicted_origin": predicted_origin,

                "forensic_confidence": round(
                    confidence,
                    4,
                ),

                "drift_strength": round(
                    drift_strength,
                    4,
                ),

                "performance_degradation": round(
                    performance_degradation,
                    4,
                ),

                "impact_score": round(
                    impact_score,
                    4,
                ),

                "adaptation_action": decision[
                    "action"
                ],

                "severity": decision[
                    "severity"
                ],

                "risk_score": decision[
                    "risk_score"
                ],

                "reason": decision[
                    "reason"
                ],
            }
        )

    result_df = pd.DataFrame(results)

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # ---------------------------------------------------------
    # Display
    # ---------------------------------------------------------

    print()
    print("=== ADAPTATION DECISIONS ===")

    for _, row in result_df.iterrows():

        print(
            f"{row['scenario']:<28} "
            f"cause={row['true_cause']:<18} "
            f"origin={row['predicted_origin']:<12} "
            f"action={row['adaptation_action']:<8} "
            f"severity={row['severity']:<6} "
            f"risk={row['risk_score']:.3f}"
        )

    print()
    print("=== ACTION SUMMARY ===")

    action_counts = (
        result_df["adaptation_action"]
        .value_counts()
        .to_dict()
    )

    for action in [
        "MONITOR",
        "UPDATE",
        "RETRAIN",
    ]:
        print(
            f"{action:<10}: "
            f"{action_counts.get(action, 0)}"
        )

    print()
    print(
        f"Saved: {OUTPUT_FILE.resolve()}"
    )


if __name__ == "__main__":
    main()