from dataclasses import dataclass
from typing import List, Dict


@dataclass
class DriftScenario:
    scenario_id: str
    drift_type: str
    cause: str
    first_observable_stage: str
    affected_features: List[str]
    magnitude: float
    description: str

    def ground_truth(self) -> Dict:
        return {
            "scenario_id": self.scenario_id,
            "drift_type": self.drift_type,
            "cause": self.cause,
            "first_observable_stage": self.first_observable_stage,
            "affected_features": self.affected_features,
            "magnitude": self.magnitude,
            "description": self.description,
        }


SCENARIOS = [
    DriftScenario("F01_feature_abrupt", "abrupt_feature", "FEATURE_DRIFT", "feature", ["sensor_2"], 0.30, "Abrupt 30% shift in sensor_2"),
    DriftScenario("F02_feature_gradual", "gradual_feature", "FEATURE_DRIFT", "feature", ["sensor_7"], 0.30, "Gradual 30% increase in sensor_7"),
    DriftScenario("F03_multi_feature", "multi_feature", "FEATURE_DRIFT", "feature", ["sensor_2", "sensor_7", "sensor_11"], 0.20, "Simultaneous drift across multiple sensors"),
    DriftScenario("F04_concept", "concept", "CONCEPT_DRIFT", "residual", [], 0.35, "Change in the target relationship causing residual and performance degradation"),
    DriftScenario("F05_prediction", "prediction", "PREDICTION_DRIFT", "prediction", [], 0.30, "Prediction-stage distribution shift"),
    DriftScenario("F06_prediction_controlled", "prediction_controlled", "PREDICTION_DRIFT", "prediction", [], 0.40, "Controlled prediction-stage distribution shift"),
    DriftScenario("F07_performance_only", "performance", "PERFORMANCE_ONLY", "performance", [], 1.00, "Prediction-target alignment failure causing performance degradation without prediction-distribution drift"),
    DriftScenario("F08_recurrent", "recurrent_feature", "RECURRENT_DRIFT", "feature", ["sensor_2"], 0.25, "Feature drift occurring in separated recurrent episodes"),
    DriftScenario("F09_no_drift", "none", "NO_DRIFT", "none", [], 0.00, "Control scenario with no injected drift"),
]


def get_scenarios() -> List[DriftScenario]:
    return SCENARIOS


def get_scenario(scenario_id: str) -> DriftScenario:
    for scenario in SCENARIOS:
        if scenario.scenario_id == scenario_id:
            return scenario

    raise ValueError(f"Unknown scenario: {scenario_id}")
