from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict

from src.driftx.evaluation.safety_gate import (
    evaluate_safety_gate,
)
from src.driftx.models.model_registry import (
    ModelRegistry,
)


@dataclass
class AdaptationExecutionResult:
    """
    Final result of the autonomous adaptation controller.
    """

    adaptation_action: str
    safety_decision: str
    registry_action: str
    reason: str
    challenger_mae: float
    champion_mae: float
    mae_improvement: float

    def to_dict(self) -> Dict:
        return asdict(self)


def execute_adaptation(
    adaptation_action: str,
    champion_metrics: Dict[str, float],
    challenger_metrics: Dict[str, float],
    challenger_path: Path,
    registry: ModelRegistry,
    min_improvement: float = 0.0,
    max_r2_drop: float = 0.02,
    max_rmse_increase: float = 0.0,
) -> AdaptationExecutionResult:
    """
    Execute an adaptation decision safely.

    Flow:

        MONITOR
            -> no model change

        UPDATE / RETRAIN
            -> evaluate challenger
            -> safety gate
            -> promote or reject

    The champion model is never replaced without
    passing the safety gate.
    """

    adaptation_action = str(
        adaptation_action
    ).upper()

    # ---------------------------------------------------------
    # MONITOR
    # ---------------------------------------------------------

    if adaptation_action == "MONITOR":

        return AdaptationExecutionResult(
            adaptation_action="MONITOR",
            safety_decision="NOT_REQUIRED",
            registry_action="KEEP_CHAMPION",
            reason=(
                "Adaptation policy selected monitoring. "
                "No model replacement was attempted."
            ),
            challenger_mae=float(
                challenger_metrics.get(
                    "mae",
                    0.0,
                )
            ),
            champion_mae=float(
                champion_metrics.get(
                    "mae",
                    0.0,
                )
            ),
            mae_improvement=0.0,
        )

    # ---------------------------------------------------------
    # Validate adaptation action
    # ---------------------------------------------------------

    if adaptation_action not in {
        "UPDATE",
        "RETRAIN",
    }:
        raise ValueError(
            "Unsupported adaptation action: "
            f"{adaptation_action}"
        )

    # ---------------------------------------------------------
    # Safety Gate
    # ---------------------------------------------------------

    gate_result = evaluate_safety_gate(
        champion_metrics=champion_metrics,
        challenger_metrics=challenger_metrics,
        min_improvement=min_improvement,
        max_r2_drop=max_r2_drop,
        max_rmse_increase=max_rmse_increase,
    )

    # ---------------------------------------------------------
    # Challenger PASSED
    # ---------------------------------------------------------

    if gate_result.decision == "PROMOTE":

        registry_result = registry.promote(
            challenger_path=challenger_path,
            reason=(
                f"{adaptation_action} adaptation "
                "produced a challenger that passed "
                "the safety gate."
            ),
        )

        return AdaptationExecutionResult(
            adaptation_action=adaptation_action,
            safety_decision="PROMOTE",
            registry_action=registry_result[
                "status"
            ],
            reason=gate_result.reason,
            challenger_mae=gate_result.challenger_mae,
            champion_mae=gate_result.champion_mae,
            mae_improvement=gate_result.improvement,
        )

    # ---------------------------------------------------------
    # Challenger FAILED
    # ---------------------------------------------------------

    return AdaptationExecutionResult(
        adaptation_action=adaptation_action,
        safety_decision="REJECT",
        registry_action="KEEP_CHAMPION",
        reason=gate_result.reason,
        challenger_mae=gate_result.challenger_mae,
        champion_mae=gate_result.champion_mae,
        mae_improvement=gate_result.improvement,
    )