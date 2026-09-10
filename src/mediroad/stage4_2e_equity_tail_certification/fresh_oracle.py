from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from mediroad.stage4_2_certification.backends import HighsBackend, SolveOptions
from mediroad.stage4_2_certification.formulation import CertificationFormulation
from mediroad.stage4_2_certification.types import Floor

Sense = Literal["max", "min"]


@dataclass
class FreshThresholdOracleResult:
    certified: bool
    method: str
    objective: str
    sense: Sense
    incumbent_value: float
    policy_bound: float
    strict_threshold: float
    relative_gap: float
    selected_indices: np.ndarray
    selected_source: str
    rounds: int
    history: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def policy_bound_and_strict_threshold(
    sense: Sense, incumbent: float, relative_gap: float
) -> tuple[float, float]:
    """Return the frozen policy boundary and the strict next-float threshold.

    The Stage4.2 policy says that an incumbent is certified when no solution is
    *strictly more than* ``relative_gap`` better.  The policy boundary is kept
    at exactly the requested ratio, while the feasibility row is moved one
    representable float beyond it so equality at 0.5% remains acceptable.
    """

    incumbent = float(incumbent)
    relative_gap = float(relative_gap)
    if not math.isfinite(incumbent) or not math.isfinite(relative_gap) or relative_gap < 0:
        raise ValueError("incumbent and relative_gap must be finite; gap must be non-negative")
    if sense == "max":
        boundary = math.fsum([incumbent, abs(incumbent) * relative_gap])
        return float(boundary), float(math.nextafter(boundary, math.inf))
    if sense == "min":
        boundary = math.fsum([incumbent, -abs(incumbent) * relative_gap])
        return float(boundary), float(math.nextafter(boundary, -math.inf))
    raise ValueError(f"Unsupported sense: {sense}")


def _better(sense: Sense, left: float, right: float) -> bool:
    return left > right if sense == "max" else left < right


def _reaches_threshold(sense: Sense, value: float, threshold: float) -> bool:
    tol = 1e-10 * max(1.0, abs(float(value)), abs(float(threshold)))
    if sense == "max":
        return float(value) >= float(threshold) - tol
    return float(value) <= float(threshold) + tol


def _oracle_options(
    base: SolveOptions,
    *,
    output_dir: Path,
    objective: str,
    round_index: int,
    time_limit_sec: float,
) -> SolveOptions:
    return SolveOptions(
        time_limit_sec=float(time_limit_sec),
        relative_gap=0.0,
        threads=int(base.threads),
        random_seed=int(base.random_seed) + 100_000 + int(round_index),
        parallel=bool(base.parallel),
        presolve=bool(base.presolve),
        heuristic_effort=float(base.heuristic_effort),
        mip_max_start_nodes=int(base.mip_max_start_nodes),
        log_to_console=bool(base.log_to_console),
        log_path=output_dir / f"oracle_{round_index}.highs.log",
        improving_solution_path=output_dir / f"oracle_{round_index}.highs.improving.sol",
        extra_options=dict(base.extra_options),
    )


def run_fresh_threshold_oracle(
    *,
    formulation: CertificationFormulation,
    backend: HighsBackend,
    objective: str,
    sense: Sense,
    floors: list[Floor],
    incumbent_selection: np.ndarray,
    incumbent_value: float,
    base_options: SolveOptions,
    output_dir: Path,
    relative_gap: float,
    max_rounds: int = 2,
    time_limit_sec: float | None = None,
) -> FreshThresholdOracleResult:
    """Fresh solver-native threshold certification for a frozen stage.

    Historical bounds are never read.  Each round asks only whether a plan
    strictly more than the frozen policy gap better than the current incumbent
    exists.  An INFEASIBLE result is therefore a fresh 0.5% certificate.  If a
    counterexample is found it becomes the next incumbent and the oracle is
    repeated.  Any timeout/no-incumbent state remains inconclusive.
    """

    selected = np.asarray(incumbent_selection, dtype=int)
    current_value = float(incumbent_value)
    history: list[dict[str, Any]] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    budget = float(base_options.time_limit_sec if time_limit_sec is None else time_limit_sec)

    for round_index in range(1, int(max_rounds) + 1):
        policy_bound, strict_threshold = policy_bound_and_strict_threshold(
            sense, current_value, relative_gap
        )
        threshold = Floor(objective, sense, strict_threshold)
        model = formulation.build(
            objective_name="feasibility",
            sense="min",
            floors=list(floors),
            threshold=threshold,
            name=f"stage42e__fresh_threshold__{objective}__round{round_index}",
        )
        options = _oracle_options(
            base_options,
            output_dir=output_dir,
            objective=objective,
            round_index=round_index,
            time_limit_sec=budget,
        )
        result = backend.solve(model, options, mip_start=None)
        record: dict[str, Any] = {
            "round": round_index,
            "objective": objective,
            "sense": sense,
            "incumbent_before": current_value,
            "policy_bound": policy_bound,
            "strict_threshold": strict_threshold,
            "status": result.status,
            "has_incumbent": bool(result.has_incumbent),
            "is_infeasible": bool(result.is_infeasible),
            "best_bound": result.best_bound,
            "relative_gap": result.relative_gap,
            "wall_time_sec": float(result.wall_time_sec),
            "mip_node_count": result.mip_node_count,
            "highs_version": result.highs_version,
            "message": result.message,
            "options": result.options,
            "log_path": str(result.log_path) if result.log_path else None,
        }
        history.append(record)

        if result.is_infeasible:
            return FreshThresholdOracleResult(
                certified=True,
                method="FRESH_THRESHOLD_INFEASIBILITY",
                objective=objective,
                sense=sense,
                incumbent_value=current_value,
                policy_bound=policy_bound,
                strict_threshold=strict_threshold,
                relative_gap=float(relative_gap),
                selected_indices=selected,
                selected_source=(
                    "FRESH_HIGHS_MAIN" if round_index == 1 else f"FRESH_THRESHOLD_COUNTEREXAMPLE_ROUND_{round_index - 1}"
                ),
                rounds=round_index,
                history=history,
            )

        if result.solution is None or not result.has_incumbent:
            return FreshThresholdOracleResult(
                certified=False,
                method="FRESH_THRESHOLD_INCONCLUSIVE",
                objective=objective,
                sense=sense,
                incumbent_value=current_value,
                policy_bound=policy_bound,
                strict_threshold=strict_threshold,
                relative_gap=math.inf,
                selected_indices=selected,
                selected_source="FRESH_THRESHOLD_INCONCLUSIVE",
                rounds=round_index,
                history=history,
            )

        proposed = formulation.selected_from_solution(model, result.solution)
        valid, reason = formulation.validate_selection(proposed)
        if not valid:
            record["counterexample_rejected"] = f"hard selection invalid: {reason}"
            return FreshThresholdOracleResult(
                certified=False,
                method="FRESH_THRESHOLD_INVALID_COUNTEREXAMPLE",
                objective=objective,
                sense=sense,
                incumbent_value=current_value,
                policy_bound=policy_bound,
                strict_threshold=strict_threshold,
                relative_gap=math.inf,
                selected_indices=selected,
                selected_source="FRESH_THRESHOLD_INVALID_COUNTEREXAMPLE",
                rounds=round_index,
                history=history,
            )
        full = formulation.complete_solution(model, proposed)
        violations = formulation.feasibility_violations(model, full, tol=1e-6)
        proposed_value = float(formulation.objective_for_selection(objective, proposed))
        record["counterexample_selected_indices"] = proposed.tolist()
        record["counterexample_cpu_objective"] = proposed_value
        record["counterexample_feasibility"] = violations
        if not bool(violations["feasible"]) or not _reaches_threshold(
            sense, proposed_value, strict_threshold
        ):
            record["counterexample_rejected"] = "CPU recomputation does not satisfy the strict threshold model"
            return FreshThresholdOracleResult(
                certified=False,
                method="FRESH_THRESHOLD_INVALID_COUNTEREXAMPLE",
                objective=objective,
                sense=sense,
                incumbent_value=current_value,
                policy_bound=policy_bound,
                strict_threshold=strict_threshold,
                relative_gap=math.inf,
                selected_indices=selected,
                selected_source="FRESH_THRESHOLD_INVALID_COUNTEREXAMPLE",
                rounds=round_index,
                history=history,
            )
        if not _better(sense, proposed_value, current_value):
            record["counterexample_rejected"] = "threshold solution did not improve the incumbent"
            return FreshThresholdOracleResult(
                certified=False,
                method="FRESH_THRESHOLD_NUMERICAL_INCONCLUSIVE",
                objective=objective,
                sense=sense,
                incumbent_value=current_value,
                policy_bound=policy_bound,
                strict_threshold=strict_threshold,
                relative_gap=math.inf,
                selected_indices=selected,
                selected_source="FRESH_THRESHOLD_NUMERICAL_INCONCLUSIVE",
                rounds=round_index,
                history=history,
            )
        selected = np.asarray(proposed, dtype=int)
        current_value = proposed_value

    policy_bound, strict_threshold = policy_bound_and_strict_threshold(
        sense, current_value, relative_gap
    )
    return FreshThresholdOracleResult(
        certified=False,
        method="FRESH_THRESHOLD_ROUND_LIMIT",
        objective=objective,
        sense=sense,
        incumbent_value=current_value,
        policy_bound=policy_bound,
        strict_threshold=strict_threshold,
        relative_gap=math.inf,
        selected_indices=selected,
        selected_source=f"FRESH_THRESHOLD_COUNTEREXAMPLE_ROUND_{max_rounds}",
        rounds=int(max_rounds),
        history=history,
    )


__all__ = [
    "FreshThresholdOracleResult",
    "policy_bound_and_strict_threshold",
    "run_fresh_threshold_oracle",
]
