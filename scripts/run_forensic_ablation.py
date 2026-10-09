import pandas as pd

p = pd.read_csv("results/forensic_full_benchmark.csv")

print("=== DRIFTX-FORENSICS: FORENSIC ABLATION ===")
print()

cols = [
    c for c in [
        "scenario",
        "true_cause",
        "true_first_stage",
        "forensic_origin",
        "baseline_origin"
    ]
    if c in p.columns
]

print(p[cols].to_string(index=False))

print()
print("=== FORENSIC ACCURACY ===")

if "forensic_origin" in p.columns:
    forensic = (
        p["forensic_origin"]
        == p["true_first_stage"]
    ).mean()

    print(
        f"DriftX forensic accuracy : "
        f"{forensic:.2%}"
    )

if "baseline_origin" in p.columns:
    baseline = (
        p["baseline_origin"]
        == p["true_first_stage"]
    ).mean()

    print(
        f"Temporal baseline accuracy: "
        f"{baseline:.2%}"
    )

print()
print("=== CONFUSION: DRIFTX ===")

if "forensic_origin" in p.columns:
    print(
        pd.crosstab(
            p["true_first_stage"],
            p["forensic_origin"],
            margins=True
        )
    )

print()
print("=== CONFUSION: TEMPORAL BASELINE ===")

if "baseline_origin" in p.columns:
    print(
        pd.crosstab(
            p["true_first_stage"],
            p["baseline_origin"],
            margins=True
        )
    )
