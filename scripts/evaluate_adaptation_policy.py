from pathlib import Path

import pandas as pd


INPUT_FILE = Path("results/adaptation_benchmark.csv")
OUTPUT_FILE = Path("results/adaptation_policy_evaluation.csv")


# Ground-truth adaptation actions for the controlled
# Drift Laboratory scenarios.
EXPECTED_ACTIONS = {
    "F01_feature_abrupt": "UPDATE",
    "F02_feature_gradual": "UPDATE",
    "F03_multi_feature": "UPDATE",
    "F04_concept": "RETRAIN",
    "F05_prediction": "UPDATE",
    "F06_prediction_controlled": "RETRAIN",
    "F07_performance_only": "RETRAIN",
    "F08_recurrent": "UPDATE",
    "F09_no_drift": "MONITOR",
}


def main():
    print(
        "=== DRIFTX-FORENSICS: ADAPTATION POLICY EVALUATION ==="
    )

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    df["expected_action"] = df["scenario"].map(
        EXPECTED_ACTIONS
    )

    # Make sure every scenario has a defined ground truth.
    missing = df.loc[
        df["expected_action"].isna(),
        "scenario",
    ].tolist()

    if missing:
        raise ValueError(
            "Missing expected actions for: "
            + ", ".join(missing)
        )

    df["action_correct"] = (
        df["adaptation_action"]
        == df["expected_action"]
    )

    # ---------------------------------------------------------
    # Overall accuracy
    # ---------------------------------------------------------

    accuracy = (
        df["action_correct"].mean() * 100
    )

    # ---------------------------------------------------------
    # Confusion matrix
    # ---------------------------------------------------------

    actions = [
        "MONITOR",
        "UPDATE",
        "RETRAIN",
    ]

    confusion = pd.crosstab(
        df["expected_action"],
        df["adaptation_action"],
        rownames=["Expected"],
        colnames=["Predicted"],
        dropna=False,
    )

    confusion = confusion.reindex(
        index=actions,
        columns=actions,
        fill_value=0,
    )

    # ---------------------------------------------------------
    # Per-action metrics
    # ---------------------------------------------------------

    action_metrics = []

    for action in actions:

        expected = (
            df["expected_action"] == action
        )

        predicted = (
            df["adaptation_action"] == action
        )

        true_positive = (
            expected & predicted
        ).sum()

        false_positive = (
            ~expected & predicted
        ).sum()

        false_negative = (
            expected & ~predicted
        ).sum()

        precision = (
            true_positive
            / (true_positive + false_positive)
            if true_positive + false_positive > 0
            else 0.0
        )

        recall = (
            true_positive
            / (true_positive + false_negative)
            if true_positive + false_negative > 0
            else 0.0
        )

        f1 = (
            2 * precision * recall
            / (precision + recall)
            if precision + recall > 0
            else 0.0
        )

        action_metrics.append(
            {
                "action": action,
                "precision": round(
                    precision,
                    4,
                ),
                "recall": round(
                    recall,
                    4,
                ),
                "f1": round(
                    f1,
                    4,
                ),
            }
        )

    # ---------------------------------------------------------
    # Save detailed evaluation
    # ---------------------------------------------------------

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # ---------------------------------------------------------
    # Print results
    # ---------------------------------------------------------

    print()
    print("=== SCENARIO RESULTS ===")

    for _, row in df.iterrows():

        status = (
            "CORRECT"
            if row["action_correct"]
            else "WRONG"
        )

        print(
            f"{row['scenario']:<28} "
            f"expected={row['expected_action']:<8} "
            f"predicted={row['adaptation_action']:<8} "
            f"{status}"
        )

    print()
    print("=== ADAPTATION POLICY ACCURACY ===")
    print(
        f"Scenarios evaluated : {len(df)}"
    )
    print(
        f"Correct decisions   : "
        f"{df['action_correct'].sum()}"
    )
    print(
        f"Policy accuracy     : "
        f"{accuracy:.1f}%"
    )

    print()
    print("=== CONFUSION MATRIX ===")
    print(confusion.to_string())

    print()
    print("=== ACTION METRICS ===")

    for metric in action_metrics:

        print(
            f"{metric['action']:<10} "
            f"precision={metric['precision']:.3f} "
            f"recall={metric['recall']:.3f} "
            f"F1={metric['f1']:.3f}"
        )

    print()
    print(
        f"Saved: {OUTPUT_FILE.resolve()}"
    )


if __name__ == "__main__":
    main()