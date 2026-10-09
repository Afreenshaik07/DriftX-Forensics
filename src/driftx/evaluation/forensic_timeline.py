from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class ForensicEvent:
    stage: str
    timestamp: int
    evidence: Dict[str, float] = field(default_factory=dict)

    def to_dict(self):
        return {
            "stage": self.stage,
            "timestamp": self.timestamp,
            "evidence": self.evidence
        }


@dataclass
class ForensicTimeline:
    events: List[ForensicEvent] = field(default_factory=list)

    def add_event(self, event: ForensicEvent):
        self.events.append(event)
        self.events.sort(key=lambda item: item.timestamp)

    def to_dict(self):
        return {
            "events": [event.to_dict() for event in self.events],
            "event_count": len(self.events)
        }
