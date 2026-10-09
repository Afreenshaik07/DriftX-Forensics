from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class DriftEvent:
    stage: str
    drift_type: str
    affected_features: List[str] = field(default_factory=list)
    evidence: Dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0

    def to_dict(self):
        return {
            "stage": self.stage,
            "drift_type": self.drift_type,
            "affected_features": self.affected_features,
            "evidence": self.evidence,
            "confidence": self.confidence
        }
