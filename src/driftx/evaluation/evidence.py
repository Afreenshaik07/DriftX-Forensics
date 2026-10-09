from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class StageEvidence:
    stage: str
    score: float
    p_value: Optional[float] = None
    onset: Optional[int] = None
    persistence: float = 0.0
    impact: float = 0.0
    metadata: Dict = field(default_factory=dict)


@dataclass
class ForensicEvidence:
    stages: Dict[str, StageEvidence]
    feature_contributors: List[Dict] = field(default_factory=list)

    def stage_scores(self) -> Dict[str, float]:
        return {
            stage: evidence.score
            for stage, evidence in self.stages.items()
        }

    def onset_lags(self) -> Dict[str, int]:
        return {
            stage: evidence.onset
            for stage, evidence in self.stages.items()
            if evidence.onset is not None
        }

    def impact_scores(self) -> Dict[str, float]:
        return {
            stage: evidence.impact
            for stage, evidence in self.stages.items()
        }

    def to_dict(self) -> Dict:
        return {
            "stages": {
                stage: {
                    "score": evidence.score,
                    "p_value": evidence.p_value,
                    "onset": evidence.onset,
                    "persistence": evidence.persistence,
                    "impact": evidence.impact,
                    "metadata": evidence.metadata,
                }
                for stage, evidence in self.stages.items()
            },
            "feature_contributors": self.feature_contributors,
        }
