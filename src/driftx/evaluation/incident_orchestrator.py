from typing import Any
import math

from src.driftx.evaluation.remediation_recommender import recommend_remediation
from src.driftx.evaluation.safety_gate import evaluate_safety_gate


def process_incident(
    scenario: str,
    diagnosis: dict[str, Any],
    evidence: dict[str, float],
    impact: dict[str, float],
    champion_metrics: dict[str, float] | None = None,
    challenger_metrics: dict[str, float] | None = None,
    min_improvement: float = 0.0,
    max_r2_drop: float = 0.02,
    max_rmse_increase: float = 0.0,
) -> dict[str, Any]:
    """Connect forensic diagnosis, remediation, and challenger safety evaluation."""

    origin = str(diagnosis.get("predicted_origin", "unknown"))
    confidence = float(diagnosis.get("confidence", 0.0))

    if not 0.0 <= confidence <= 1.0:
        raise ValueError("Diagnosis confidence must be between 0 and 1.")

    remediation = recommend_remediation(
        origin=origin,
        confidence=confidence,
        evidence=evidence,
        impact=impact,
    )

    result: dict[str, Any] = {
        "scenario": scenario,
        "diagnosis": diagnosis,
        "remediation": remediation,
        "safety_gate": {
            "evaluated": False,
            "decision": "NOT_EVALUATED",
            "reason": "Champion and challenger metrics have not both been supplied.",
        },
        "deployment_authorized": False,
        "status": "RECOMMENDATION_ONLY",
    }

    if champion_metrics is None and challenger_metrics is None:
        return result

    if champion_metrics is None or challenger_metrics is None:
        raise ValueError("Supply both champion_metrics and challenger_metrics.")

    required = ("mae", "rmse", "r2")
    for label, metrics in (
        ("champion", champion_metrics),
        ("challenger", challenger_metrics),
    ):
        missing = [key for key in required if key not in metrics]
        if missing:
            raise ValueError(f"{label} metrics missing: {missing}")

        for key in required:
            value = float(metrics[key])
            if not math.isfinite(value):
                raise ValueError(f"{label} {key} must be finite.")
            if key in ("mae", "rmse") and value < 0:
                raise ValueError(f"{label} {key} cannot be negative.")

    gate = evaluate_safety_gate(
        champion_metrics=champion_metrics,
        challenger_metrics=challenger_metrics,
        min_improvement=min_improvement,
        max_r2_drop=max_r2_drop,
        max_rmse_increase=max_rmse_increase,
    )

    result["safety_gate"] = gate.to_dict()
    result["safety_gate"]["evaluated"] = True
    result["status"] = "SAFETY_GATE_EVALUATED"

    # A passing gate is a recommendation, not permission to deploy automatically.
    result["deployment_authorized"] = False
    return result


def save_incident_record(incident: dict[str, Any], output_dir: str = "results") -> str:
    """Append exactly one valid JSON object per line to incident history."""
    import json
    import uuid
    from datetime import datetime, timezone
    from pathlib import Path

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "incident_history.jsonl"

    record = dict(incident)
    record.setdefault("incident_id", uuid.uuid4().hex)
    record.setdefault("recorded_at_utc", datetime.now(timezone.utc).isoformat())

    # Serialize completely before opening the file so invalid data cannot
    # partially write a JSON record.
    serialized = json.dumps(record, ensure_ascii=False, allow_nan=False, default=str)

    with path.open("a", encoding="utf-8") as handle:
        handle.write(serialized + "\n")

    return str(path)


def classify_incident_severity(
    evidence: dict[str, Any],
    confidence: float = 0.0,
) -> dict[str, Any]:
    """Classify incident severity from impact evidence and diagnosis confidence."""
    impact = evidence.get("impact", evidence)
    if not isinstance(impact, dict):
        impact = {}

    values = []
    for key, value in impact.items():
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            values.append(number)

    maximum_impact = max(values, default=0.0)

    if maximum_impact >= 0.8:
        severity = "CRITICAL"
    elif maximum_impact >= 0.5 or confidence < 0.4:
        severity = "HIGH"
    elif maximum_impact >= 0.2:
        severity = "MEDIUM"
    else:
        severity = "LOW"

    return {
        "severity": severity,
        "confidence": float(confidence),
        "maximum_impact": maximum_impact,
    }
