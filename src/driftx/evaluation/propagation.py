from dataclasses import dataclass, field
from typing import List


@dataclass
class PropagationPath:
    stages: List[str] = field(default_factory=list)

    def add_stage(self, stage: str):
        if stage not in self.stages:
            self.stages.append(stage)

    def to_dict(self):
        return {
            "stages": self.stages,
            "path_length": len(self.stages)
        }
