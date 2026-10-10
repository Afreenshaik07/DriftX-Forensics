import unittest

from src.driftx.evaluation.incident_orchestrator import (
    process_incident,
    classify_incident_severity,
)


class IncidentOrchestratorTests(unittest.TestCase):
    def setUp(self):
        self.diagnosis = {
            "predicted_origin": "feature",
            "confidence": 0.85,
            "propagation_path": ["feature", "prediction"],
            "ranking": [],
        }
        self.evidence = {
            "feature": 0.90,
            "prediction": 0.65,
            "residual": 0.30,
            "performance": 0.15,
        }
        self.impact = dict(self.evidence)
        self.champion = {"mae": 12.0, "rmse": 18.0, "r2": 0.82}

    def run_incident(self, challenger):
        return process_incident(
            scenario="unit_test",
            diagnosis=self.diagnosis,
            evidence=self.evidence,
            impact=self.impact,
            champion_metrics=self.champion,
            challenger_metrics=challenger,
        )

    def test_safe_challenger_passes_gate(self):
        result = self.run_incident(
            {"mae": 10.5, "rmse": 17.0, "r2": 0.83}
        )
        self.assertEqual(result["safety_gate"]["decision"], "PROMOTE")
        self.assertFalse(result["deployment_authorized"])

    def test_bad_rmse_challenger_is_rejected(self):
        result = self.run_incident(
            {"mae": 10.0, "rmse": 20.0, "r2": 0.83}
        )
        self.assertEqual(result["safety_gate"]["decision"], "REJECT")
        self.assertFalse(result["deployment_authorized"])

    def test_missing_challenger_metrics_does_not_evaluate_gate(self):
        result = process_incident(
            scenario="diagnosis_only",
            diagnosis=self.diagnosis,
            evidence=self.evidence,
            impact=self.impact,
        )
        self.assertFalse(result["safety_gate"]["evaluated"])
        self.assertFalse(result["deployment_authorized"])

    def test_invalid_confidence_is_rejected(self):
        diagnosis = dict(self.diagnosis, confidence=1.5)
        with self.assertRaises(ValueError):
            process_incident(
                "invalid", diagnosis, self.evidence, self.impact
            )

    def test_severity_classification(self):
        result = classify_incident_severity(self.evidence, 0.85)
        self.assertEqual(result["severity"], "CRITICAL")

    def test_incomplete_metrics_are_rejected(self):
        with self.assertRaises(ValueError):
            self.run_incident({"mae": 10.0})

if __name__ == "__main__":
    unittest.main(verbosity=2)
