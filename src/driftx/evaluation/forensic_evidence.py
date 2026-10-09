from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ForensicEvidence:
    feature_drift: bool = False
    prediction_drift: bool = False
    residual_drift: bool = False
    performance_degradation: bool = False

    affected_features: List[str] = field(default_factory=list)

    feature_score: float = 0.0
    prediction_score: float = 0.0
    residual_score: float = 0.0
    performance_score: float = 0.0

    metadata: Dict[str, float] = field(default_factory=dict)

    def to_dict(self):
        return {
            "feature_drift": self.feature_drift,
            "prediction_drift": self.prediction_drift,
            "residual_drift": self.residual_drift,
            "performance_degradation": self.performance_degradation,
            "affected_features": self.affected_features,
            "feature_score": self.feature_score,
            "prediction_score": self.prediction_score,
            "residual_score": self.residual_score,
            "performance_score": self.performance_score,
            "metadata": self.metadata
        }
