from pathlib import Path
import json
from datetime import datetime


class RegimeMemory:
    def __init__(self, memory_path="artifacts/regime_memory.json"):
        self.memory_path = Path(memory_path)
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)
        self.memory = self._load()

    def _load(self):
        if not self.memory_path.exists():
            return []
        try:
            with open(self.memory_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return []

    def _save(self):
        with open(self.memory_path, "w", encoding="utf-8") as f:
            json.dump(self.memory, f, indent=2)

    @staticmethod
    def _distance(a, b):
        keys = [
            "feature_drift",
            "prediction_drift",
            "residual_drift",
            "performance_degradation",
        ]

        values = []
        for key in keys:
            values.append(abs(float(a.get(key, 0.0)) - float(b.get(key, 0.0))))

        return sum(values) / len(values)

    def remember(self, regime_id, signature, origin, drift_type, action, confidence, risk_score, model_version="unknown", adapter="none"):
        record = {
            "regime_id": regime_id,
            "signature": signature,
            "origin": origin,
            "drift_type": drift_type,
            "action": action,
            "confidence": float(confidence),
            "risk_score": float(risk_score),
            "model_version": model_version,
            "adapter": adapter,
        }
        for i, existing in enumerate(self.memory):
            if self._distance(existing.get("signature", {}), signature) == 0:
                record["regime_id"] = existing.get("regime_id", regime_id)
                self.memory[i] = record
                self._save()
                return record
        self.memory.append(record)
        self._save()
        return record
    def find_similar(self, signature, threshold=0.15):
        matches = []

        for entry in self.memory:
            distance = self._distance(signature, entry["signature"])

            if distance <= threshold:
                matches.append(
                    {
                        "regime_id": entry["regime_id"],
                        "distance": distance,
                        "origin": entry["origin"],
                        "drift_type": entry["drift_type"],
                        "action": entry["action"],
                        "confidence": entry["confidence"],
                        "risk_score": entry["risk_score"],
                        "model_version": entry["model_version"],
                        "adapter": entry["adapter"],
                    }
                )

        return sorted(matches, key=lambda x: x["distance"])

    def get_best_match(self, signature, threshold=0.15):
        matches = self.find_similar(signature, threshold)

        if not matches:
            return None

        return matches[0]

    def count(self):
        return len(self.memory)

    def clear(self):
        self.memory = []
        self._save()

