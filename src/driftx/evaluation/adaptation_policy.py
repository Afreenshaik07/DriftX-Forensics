from dataclasses import dataclass, asdict
from typing import Dict, Optional


@dataclass
class AdaptationDecision:
    action: str
    severity: str
    reason: str
    risk_score: float
    confidence: float
    origin: str
    memory_match: bool = False
    memory_regime: str = "NONE"

    def to_dict(self) -> Dict:
        return asdict(self)


def _clip(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _severity_from_risk(risk_score: float) -> str:
    if risk_score < 0.25:
        return "LOW"
    if risk_score < 0.60:
        return "MEDIUM"
    return "HIGH"


def decide_adaptation(
    predicted_origin: str,
    confidence: float,
    impact_score: float,
    performance_degradation: float,
    drift_strength: float,
    drift_type: str = "UNKNOWN",
    memory_match: Optional[Dict] = None,
) -> AdaptationDecision:

    confidence = _clip(confidence)
    impact_score = _clip(impact_score)
    performance_degradation = _clip(performance_degradation)
    drift_strength = _clip(drift_strength)

    drift_type = str(drift_type).upper()
    predicted_origin = str(predicted_origin).lower()

    memory_found = memory_match is not None
    memory_regime = (
        str(memory_match.get("regime_id", "NONE"))
        if memory_found
        else "NONE"
    )

    # ---------------------------------------------------------
    # 1. NO DRIFT / UNKNOWN
    # ---------------------------------------------------------

    if predicted_origin in {"unknown", "none"}:
        return AdaptationDecision(
            action="MONITOR",
            severity="LOW",
            reason="No reliable drift origin was established. The system remains in monitoring mode.",
            risk_score=0.0,
            confidence=confidence,
            origin=predicted_origin,
            memory_match=memory_found,
            memory_regime=memory_regime,
        )

    # ---------------------------------------------------------
    # 2. CURRENT EVIDENCE RISK
    # ---------------------------------------------------------

    risk_score = (
        0.30 * drift_strength
        + 0.30 * performance_degradation
        + 0.25 * impact_score
        + 0.15 * confidence
    )

    risk_score = _clip(risk_score)

    # ---------------------------------------------------------
    # 3. MEMORY-AUGMENTED EVIDENCE
    # ---------------------------------------------------------

    if memory_found:
        previous_action = str(
            memory_match.get("action", "MONITOR")
        ).upper()

        previous_confidence = _clip(
            memory_match.get("confidence", 0.0)
        )

        memory_distance = float(
            memory_match.get("distance", 1.0)
        )

        memory_strength = _clip(
            (1.0 - memory_distance)
            * previous_confidence
        )

        risk_score = _clip(
            risk_score + 0.10 * memory_strength
        )

    severity = _severity_from_risk(risk_score)

    # ---------------------------------------------------------
    # 4. HIGH-RISK CURRENT EVIDENCE ALWAYS WINS
    # ---------------------------------------------------------

    if (
        drift_type == "CONCEPT_DRIFT"
        and performance_degradation >= 0.20
        and confidence >= 0.30
    ):
        action = "RETRAIN"
        reason = (
            "Concept drift with meaningful performance degradation "
            "requires retraining."
        )

    elif (
        drift_type == "PERFORMANCE_ONLY"
        and performance_degradation >= 0.20
    ):
        action = "RETRAIN"
        reason = (
            "Significant performance degradation requires full retraining."
        )

    elif (
        drift_type == "RECURRENT_DRIFT"
        and drift_strength >= 0.15
        and confidence >= 0.20
    ):
        action = "UPDATE"
        reason = (
            "Recurrent drift supports a lightweight model update."
        )

    elif (
        risk_score >= 0.65
        and performance_degradation >= 0.40
        and confidence >= 0.50
    ):
        action = "RETRAIN"
        reason = (
            "Strong current drift evidence and performance degradation "
            "justify full retraining."
        )

    # ---------------------------------------------------------
    # 5. MEMORY CAN REINFORCE A MODERATE DECISION
    # ---------------------------------------------------------

    elif memory_found:
        previous_action = str(
            memory_match.get("action", "MONITOR")
        ).upper()

        if (
            previous_action in {"UPDATE", "RETRAIN"}
            and risk_score >= 0.25
        ):
            action = previous_action
            reason = (
                f"Current evidence resembles previously observed "
                f"regime {memory_regime}. Previous action "
                f"{previous_action} is reused as supporting evidence."
            )
        elif risk_score >= 0.30 and confidence >= 0.40:
            action = "UPDATE"
            reason = (
                "Meaningful drift detected; historical regime evidence "
                "supports a lightweight update."
            )
        else:
            action = "MONITOR"
            reason = (
                "Historical regime evidence exists, but current evidence "
                "is insufficient for safe adaptation."
            )

    elif risk_score >= 0.30 and confidence >= 0.40:
        action = "UPDATE"
        reason = (
            "Meaningful drift has been detected, but evidence does not "
            "justify full retraining."
        )

    else:
        action = "MONITOR"
        reason = (
            "Evidence is insufficient for safe model adaptation."
        )

    return AdaptationDecision(
        action=action,
        severity=_severity_from_risk(risk_score),
        reason=reason,
        risk_score=round(risk_score, 4),
        confidence=confidence,
        origin=predicted_origin,
        memory_match=memory_found,
        memory_regime=memory_regime,
    )


def evaluate_forensic_result(
    forensic_result: Dict,
    impact_score: float = 0.0,
    performance_degradation: float = 0.0,
    drift_strength: float = 0.0,
    drift_type: str = "UNKNOWN",
    memory_match: Optional[Dict] = None,
) -> Dict:

    decision = decide_adaptation(
        predicted_origin=forensic_result.get(
            "predicted_origin", "unknown"
        ),
        confidence=forensic_result.get("confidence", 0.0),
        impact_score=impact_score,
        performance_degradation=performance_degradation,
        drift_strength=drift_strength,
        drift_type=drift_type,
        memory_match=memory_match,
    )

    return decision.to_dict()
