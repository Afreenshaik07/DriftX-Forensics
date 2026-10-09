from dataclasses import dataclass, asdict
from typing import Dict


@dataclass
class SafetyGateResult:
    """
    Result produced by the challenger safety gate.
    """

    decision: str
    reason: str
    champion_mae: float
    challenger_mae: float
    improvement: float
    improvement_ratio: float
    champion_rmse: float
    challenger_rmse: float
    champion_r2: float
    challenger_r2: float

    def to_dict(self) -> Dict:
        return asdict(self)


def _safe_float(value: float) -> float:
    return float(value)


def evaluate_safety_gate(
    champion_metrics: Dict[str, float],
    challenger_metrics: Dict[str, float],
    min_improvement: float = 0.0,
    max_r2_drop: float = 0.02,
    max_rmse_increase: float = 0.0,
) -> SafetyGateResult:
    """
    Compare a challenger model against the current champion.

    Promotion is allowed only when the challenger satisfies
    all safety conditions.

    Primary metric:
        MAE

    Safety conditions:
        1. Challenger MAE must improve sufficiently.
        2. Challenger RMSE must not increase beyond the limit.
        3. Challenger R2 must not fall beyond the allowed limit.

    Parameters
    ----------
    champion_metrics:
        Metrics of the current production/champion model.

    challenger_metrics:
        Metrics of the candidate/challenger model.

    min_improvement:
        Minimum absolute MAE improvement required.

    max_r2_drop:
        Maximum acceptable decrease in R2.

    max_rmse_increase:
        Maximum acceptable increase in RMSE.
    """

    champion_mae = _safe_float(
        champion_metrics["mae"]
    )

    challenger_mae = _safe_float(
        challenger_metrics["mae"]
    )

    champion_rmse = _safe_float(
        champion_metrics["rmse"]
    )

    challenger_rmse = _safe_float(
        challenger_metrics["rmse"]
    )

    champion_r2 = _safe_float(
        champion_metrics["r2"]
    )

    challenger_r2 = _safe_float(
        challenger_metrics["r2"]
    )

    # ---------------------------------------------------------
    # Calculate improvement
    # ---------------------------------------------------------

    improvement = (
        champion_mae
        - challenger_mae
    )

    if champion_mae > 0:
        improvement_ratio = (
            improvement / champion_mae
        )
    else:
        improvement_ratio = 0.0

    # ---------------------------------------------------------
    # Individual safety checks
    # ---------------------------------------------------------

    mae_pass = (
        improvement >= min_improvement
    )

    rmse_pass = (
        challenger_rmse
        <= champion_rmse + max_rmse_increase
    )

    r2_pass = (
        challenger_r2
        >= champion_r2 - max_r2_drop
    )

    # ---------------------------------------------------------
    # Final promotion decision
    # ---------------------------------------------------------

    if mae_pass and rmse_pass and r2_pass:

        decision = "PROMOTE"

        reason = (
            "Challenger passed all safety checks. "
            "MAE improved sufficiently while RMSE and "
            "R2 remained within the permitted safety limits."
        )

    else:

        decision = "REJECT"

        failed_checks = []

        if not mae_pass:
            failed_checks.append(
                "insufficient MAE improvement"
            )

        if not rmse_pass:
            failed_checks.append(
                "RMSE degradation"
            )

        if not r2_pass:
            failed_checks.append(
                "R2 degradation"
            )

        reason = (
            "Challenger rejected by safety gate: "
            + ", ".join(failed_checks)
            + "."
        )

    return SafetyGateResult(
        decision=decision,
        reason=reason,
        champion_mae=round(
            champion_mae,
            6,
        ),
        challenger_mae=round(
            challenger_mae,
            6,
        ),
        improvement=round(
            improvement,
            6,
        ),
        improvement_ratio=round(
            improvement_ratio,
            6,
        ),
        champion_rmse=round(
            champion_rmse,
            6,
        ),
        challenger_rmse=round(
            challenger_rmse,
            6,
        ),
        champion_r2=round(
            champion_r2,
            6,
        ),
        challenger_r2=round(
            challenger_r2,
            6,
        ),
    )


def promote_if_safe(
    champion_metrics: Dict[str, float],
    challenger_metrics: Dict[str, float],
    min_improvement: float = 0.0,
    max_r2_drop: float = 0.02,
    max_rmse_increase: float = 0.0,
) -> Dict:
    """
    Run the safety gate and return a promotion decision.
    """

    result = evaluate_safety_gate(
        champion_metrics=champion_metrics,
        challenger_metrics=challenger_metrics,
        min_improvement=min_improvement,
        max_r2_drop=max_r2_drop,
        max_rmse_increase=max_rmse_increase,
    )

    return result.to_dict()