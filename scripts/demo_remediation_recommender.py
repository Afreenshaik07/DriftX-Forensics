from src.driftx.evaluation.remediation_recommender import recommend_remediation

cases = [
    ("feature", 0.90, {"feature": 0.90}),
    ("prediction", 0.85, {"prediction": 0.85}),
    ("residual", 0.40, {"residual": 0.40}),
    ("performance", 0.92, {"performance": 0.92}),
    ("none", 0.99, {}),
]

for origin, confidence, evidence in cases:
    result = recommend_remediation(origin, confidence, evidence)
    print(f"\n{origin.upper()} | {result['action']}")
    print(f"Recommendation: {result['title']}")
    print(f"Confidence: {result['confidence']:.2f}")
    print(f"Verification: {result['verification']}")
