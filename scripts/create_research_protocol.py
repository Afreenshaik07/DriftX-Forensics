from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "research_benchmark_protocol.json"

protocol = {
    "name": "DriftX-Forensics Research Benchmark v1",
    "scenario_types": [
        "feature_abrupt",
        "feature_gradual",
        "multi_feature",
        "concept",
        "prediction",
        "prediction_controlled",
        "performance_only",
        "recurrent",
        "no_drift"
    ],
    "magnitudes": [0.10, 0.20, 0.30, 0.40, 0.50],
    "onset_fractions": [0.20, 0.35, 0.50, 0.65, 0.80],
    "seeds": [7, 21, 42, 84, 123, 156, 201, 256, 314, 512],
    "metrics": [
        "forensic_accuracy",
        "baseline_accuracy",
        "per_class_accuracy",
        "confusion_matrix",
        "confidence",
        "calibration"
    ],
    "rules": {
        "test_data_is_not_used_for_threshold_tuning": True,
        "ground_truth_is_generated_before_diagnosis": True,
        "same_cases_are_used_for_driftx_and_baseline": True,
        "random_seed_is_recorded": True,
        "magnitude_is_recorded": True,
        "onset_fraction_is_recorded": True,
        "existing_forensic_engine_is_unchanged": True
    }
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(protocol, indent=2))

total = (
    len(protocol["scenario_types"])
    * len(protocol["magnitudes"])
    * len(protocol["onset_fractions"])
    * len(protocol["seeds"])
)

print("=== DRIFTX RESEARCH BENCHMARK PROTOCOL ===")
print(f"Scenario types : {len(protocol['scenario_types'])}")
print(f"Magnitudes     : {len(protocol['magnitudes'])}")
print(f"Onsets         : {len(protocol['onset_fractions'])}")
print(f"Seeds          : {len(protocol['seeds'])}")
print(f"Maximum cases  : {total}")
print(f"Saved          : {OUT}")
print("PROTOCOL LOCKED")
