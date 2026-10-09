from typing import Any


REMEDIATION_PLAYBOOK: dict[str, dict[str, Any]] = {
    "feature": {
        "action": "UPDATE",
        "title": "Investigate and update input processing",
        "reason": "Feature-level drift indicates that incoming data distributions may have changed.",
        "steps": [
            "Validate the incoming schema and feature ranges.",
            "Check upstream data sources and recent pipeline changes.",
            "Compare reference and current feature distributions.",
            "Run the candidate model against a held-out validation set.",
        ],
        "verification": "Feature distributions return within configured tolerances without degrading model performance.",
    },
    "prediction": {
        "action": "RETRAIN_OR_CALIBRATE",
        "title": "Investigate prediction distribution changes",
        "reason": "Prediction drift may indicate changed model behavior or changing input populations.",
        "steps": [
            "Compare prediction distributions with the reference window.",
            "Check feature drift and prediction confidence.",
            "Evaluate calibration and subgroup performance.",
            "Validate any candidate update before promotion.",
        ],
        "verification": "Prediction behavior is validated against labeled holdout data and agreed performance limits.",
    },
    "residual": {
        "action": "INVESTIGATE_AND_RETRAIN",
        "title": "Investigate model error or concept drift",
        "reason": "Residual drift suggests that prediction errors may have changed.",
        "steps": [
            "Check label quality, freshness, and alignment.",
            "Compare residual distributions across time and important subgroups.",
            "Investigate changes in the relationship between features and targets.",
            "Evaluate retraining candidates on a temporal holdout.",
        ],
        "verification": "Holdout error improves or remains acceptable without violating safety constraints.",
    },
    "performance": {
        "action": "MONITOR_AND_INVESTIGATE",
        "title": "Investigate degraded model performance",
        "reason": "Performance deterioration requires investigation before changing the production model.",
        "steps": [
            "Verify that performance metrics use reliable, sufficiently recent labels.",
            "Compare MAE, RMSE, and R-squared with the champion baseline.",
            "Check data quality, subgroup failures, and recent deployment changes.",
            "Escalate to retraining only after confirming the cause.",
        ],
        "verification": "Performance recovers against the established baseline on a representative holdout.",
    },
    "none": {
        "action": "MONITOR",
        "title": "Continue baseline monitoring",
        "reason": "Available evidence does not currently justify a drift remediation.",
        "steps": [
            "Continue monitoring all stages.",
            "Collect more evidence if uncertainty remains.",
            "Reassess when new observations arrive.",
        ],
        "verification": "No new actionable drift signal appears within the monitoring window.",
    },
    "unknown": {
        "action": "COLLECT_MORE_EVIDENCE",
        "title": "Collect additional diagnostic evidence",
        "reason": "The current evidence is insufficient to confidently attribute drift to a stage.",
        "steps": [
            "Check missing telemetry and stage-level monitoring coverage.",
            "Collect comparable reference and current windows.",
            "Inspect feature, prediction, residual, and performance signals.",
            "Avoid automatic model promotion until evidence is sufficient.",
        ],
        "verification": "A stage attribution is supported by new evidence, or the case remains under monitoring.",
    },
}


def recommend_remediation(
    origin: str,
    confidence: float = 0.0,
    evidence: dict[str, float] | None = None,
    impact: dict[str, float] | None = None,
    confidence_threshold: float = 0.60,
) -> dict[str, Any]:
    """Return an evidence-aware remediation recommendation without executing changes."""
    confidence = max(0.0, min(1.0, float(confidence)))
    evidence = evidence or {}
    impact = impact or {}

    normalized_origin = str(origin).strip().lower()
    if normalized_origin not in REMEDIATION_PLAYBOOK:
        normalized_origin = "unknown"

    if normalized_origin not in {"none", "unknown"} and confidence < confidence_threshold:
        selected_origin = "unknown"
    else:
        selected_origin = normalized_origin

    playbook = REMEDIATION_PLAYBOOK[selected_origin]
    evidence_summary = [
        {
            "stage": stage,
            "evidence": round(max(0.0, min(1.0, float(value))), 4),
            "impact": round(max(0.0, min(1.0, float(impact.get(stage, 0.0)))), 4),
        }
        for stage, value in evidence.items()
    ]
    evidence_summary.sort(
        key=lambda item: item["evidence"] + item["impact"],
        reverse=True,
    )

    return {
        "origin": selected_origin,
        "confidence": round(confidence, 4),
        "action": playbook["action"],
        "title": playbook["title"],
        "reason": playbook["reason"],
        "steps": list(playbook["steps"]),
        "verification": playbook["verification"],
        "evidence_summary": evidence_summary,
        "automatic_execution": False,
    }
