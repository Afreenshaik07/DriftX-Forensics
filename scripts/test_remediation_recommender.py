import unittest

from src.driftx.evaluation.remediation_recommender import recommend_remediation


class TestRemediationRecommender(unittest.TestCase):
    def test_high_confidence_feature_drift(self):
        result = recommend_remediation(
            "feature", 0.90, {"feature": 0.9}, {"feature": 0.7}
        )
        self.assertEqual(result["origin"], "feature")
        self.assertEqual(result["action"], "UPDATE")
        self.assertFalse(result["automatic_execution"])

    def test_low_confidence_uses_safe_fallback(self):
        result = recommend_remediation("residual", 0.30)
        self.assertEqual(result["origin"], "unknown")
        self.assertEqual(result["action"], "COLLECT_MORE_EVIDENCE")

    def test_no_drift_recommends_monitoring(self):
        result = recommend_remediation("none", 0.99)
        self.assertEqual(result["action"], "MONITOR")

    def test_unknown_origin_is_safe(self):
        result = recommend_remediation("unexpected_stage", 0.99)
        self.assertEqual(result["origin"], "unknown")
        self.assertFalse(result["automatic_execution"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
