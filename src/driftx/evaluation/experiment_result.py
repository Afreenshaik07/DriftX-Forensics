from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class ExperimentResult:
    scenario_id: str
    true_origin: str
    predicted_origin: str
    diagnosis_confidence: float

    detection_delay: float = 0.0
    false_alarm: bool = False

    propagation_path_true: List[str] = field(default_factory=list)
    propagation_path_predicted: List[str] = field(default_factory=list)

    metrics: Dict[str, float] = field(default_factory=dict)

    def origin_correct(self) -> bool:
        return self.true_origin == self.predicted_origin

    def path_correct(self) -> bool:
        return (
            self.propagation_path_true
            == self.propagation_path_predicted
        )

    def to_dict(self):
        return {
            "scenario_id": self.scenario_id,
            "true_origin": self.true_origin,
            "predicted_origin": self.predicted_origin,
            "diagnosis_confidence": self.diagnosis_confidence,
            "detection_delay": self.detection_delay,
            "false_alarm": self.false_alarm,
            "origin_correct": self.origin_correct(),
            "path_correct": self.path_correct(),
            "propagation_path_true": self.propagation_path_true,
            "propagation_path_predicted": self.propagation_path_predicted,
            "metrics": self.metrics
        }
