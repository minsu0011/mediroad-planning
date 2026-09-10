"""Fail-closed certification for Stage 4 CP-SAT objective stages.

The optimizer uses a sequence of lexicographic CP-SAT solves.  A solver status
of ``FEASIBLE`` only proves that an incumbent exists; it says nothing about how
far that incumbent may be from the best known bound.  This module provides a
small, side-effect-free boundary between raw OR-Tools telemetry and release
language.

The default near-optimality contract is deliberately fixed in code at a 0.5%
relative gap.  A run may instead load an explicitly declared value from config,
but no API in this module estimates or tunes the threshold from solver results.
Missing or contradictory telemetry can never produce a certificate.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from numbers import Integral, Real
from typing import Any, Iterable, Mapping

import pandas as pd


def _require_finite_number(name: str, value: Any, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise SolverCertificationError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise SolverCertificationError(f"{name} must be a finite number")
    if minimum is not None and result < minimum:
        raise SolverCertificationError(f"{name} must be >= {minimum}")
    return result


class SolverCertificationError(ValueError):
    """Raised when solver telemetry or a certification contract is invalid."""


class CertificationClass(str, Enum):
    """The only release labels allowed for a CP-SAT objective stage or run."""

    OPTIMAL = "OPTIMAL"
    CERTIFIED_NEAR_OPTIMAL = "CERTIFIED_NEAR_OPTIMAL"
    FEASIBLE_UNCERTIFIED = "FEASIBLE_UNCERTIFIED"


@dataclass(frozen=True)
class NearOptimalityContract:
    """Immutable, predeclared rule used to certify a solver gap.

    Relative gap is computed directionally from the incumbent and the best
    objective bound, then divided by ``max(abs(objective_value),
    denominator_floor)``.  If ``absolute_gap_threshold`` is supplied, both the
    relative and absolute thresholds must pass.
    """

    contract_id: str
    relative_gap_threshold: float
    absolute_gap_threshold: float | None = None
    denominator_floor: float = 1.0
    numerical_tolerance: float = 1e-9
    prespecified: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.contract_id, str) or not self.contract_id.strip():
            raise SolverCertificationError("contract_id must be a non-empty string")
        _require_finite_number(
            "relative_gap_threshold", self.relative_gap_threshold, minimum=0.0
        )
        if float(self.relative_gap_threshold) >= 1.0:
            raise SolverCertificationError("relative_gap_threshold must be in [0, 1)")
        if self.absolute_gap_threshold is not None:
            _require_finite_number(
                "absolute_gap_threshold", self.absolute_gap_threshold, minimum=0.0
            )
        _require_finite_number("denominator_floor", self.denominator_floor, minimum=0.0)
        if float(self.denominator_floor) <= 0.0:
            raise SolverCertificationError("denominator_floor must be positive")
        _require_finite_number("numerical_tolerance", self.numerical_tolerance, minimum=0.0)
        if self.prespecified is not True:
            raise SolverCertificationError(
                "Near-optimality threshold must be prespecified before solver results are read"
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "contract_id": self.contract_id,
            "relative_gap_threshold": float(self.relative_gap_threshold),
            "absolute_gap_threshold": (
                None
                if self.absolute_gap_threshold is None
                else float(self.absolute_gap_threshold)
            ),
            "denominator_floor": float(self.denominator_floor),
            "numerical_tolerance": float(self.numerical_tolerance),
            "prespecified": True,
            "rule": "relative_gap_and_optional_absolute_gap",
        }


PREDECLARED_NEAR_OPTIMALITY_CONTRACT = NearOptimalityContract(
    contract_id="MEDIROAD_STAGE4_CP_SAT_GAP_V1_REL_0P005",
    relative_gap_threshold=0.005,
)
"""Project default: certify a complete FEASIBLE stage only at gap <= 0.5%."""

DEFAULT_NEAR_OPTIMALITY_CONTRACT = PREDECLARED_NEAR_OPTIMALITY_CONTRACT
DEFAULT_NEAR_OPTIMAL_RELATIVE_GAP = 0.005


def near_optimality_contract_from_config(
    config: Mapping[str, Any],
    *,
    contract_id: str | None = None,
) -> NearOptimalityContract:
    """Load a predeclared contract from ``solver.near_optimal_gap``.

    ``near_optimal_gap`` may be a scalar relative-gap threshold or a mapping
    containing ``relative_gap_threshold`` (``relative`` is also accepted),
    optional ``absolute_gap_threshold``, and optional denominator/tolerance.
    Missing configuration fails closed; this function never falls back to a
    threshold inferred from observed results.
    """

    if not isinstance(config, Mapping):
        raise SolverCertificationError("config must be a mapping")
    solver = config.get("solver")
    if not isinstance(solver, Mapping) or "near_optimal_gap" not in solver:
        raise SolverCertificationError(
            "config must predeclare solver.near_optimal_gap before certification"
        )
    raw = solver["near_optimal_gap"]
    if isinstance(raw, Mapping):
        relative = raw.get("relative_gap_threshold", raw.get("relative"))
        if relative is None:
            raise SolverCertificationError(
                "solver.near_optimal_gap mapping requires relative_gap_threshold"
            )
        absolute = raw.get("absolute_gap_threshold", raw.get("absolute"))
        denominator = raw.get("denominator_floor", 1.0)
        tolerance = raw.get("numerical_tolerance", 1e-9)
        configured_id = raw.get("contract_id")
    else:
        relative = raw
        absolute = None
        denominator = 1.0
        tolerance = 1e-9
        configured_id = None
    resolved_id = contract_id or configured_id or "MEDIROAD_STAGE4_CONFIGURED_GAP_V1"
    return NearOptimalityContract(
        contract_id=str(resolved_id),
        relative_gap_threshold=relative,
        absolute_gap_threshold=absolute,
        denominator_floor=denominator,
        numerical_tolerance=tolerance,
        prespecified=True,
    )


@dataclass(frozen=True)
class NormalizedSolverStage:
    """Canonical telemetry and conservative certification for one CP-SAT stage."""

    stage_order: int | None
    objective_name: str
    sense: str
    solver_status: str
    objective_value: float | None
    best_bound: float | None
    absolute_gap: float | None
    relative_gap: float | None
    wall_time_sec: float | None
    branches: int | None
    conflicts: int | None
    usable_incumbent: bool
    telemetry_complete: bool
    certification_class: CertificationClass
    contract_id: str
    relative_gap_threshold: float
    absolute_gap_threshold: float | None
    validation_errors: tuple[str, ...] = ()
    certification_reasons: tuple[str, ...] = ()

    @property
    def best_objective_bound(self) -> float | None:
        """Alias matching OR-Tools/report terminology."""

        return self.best_bound

    @property
    def certified(self) -> bool:
        return self.certification_class is not CertificationClass.FEASIBLE_UNCERTIFIED

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage_order": self.stage_order,
            "objective_name": self.objective_name,
            "sense": self.sense,
            "solver_status": self.solver_status,
            "objective_value": self.objective_value,
            "best_bound": self.best_bound,
            "absolute_gap": self.absolute_gap,
            "relative_gap": self.relative_gap,
            "wall_time_sec": self.wall_time_sec,
            "branches": self.branches,
            "conflicts": self.conflicts,
            "usable_incumbent": self.usable_incumbent,
            "telemetry_complete": self.telemetry_complete,
            "certification_class": self.certification_class.value,
            "certified": self.certified,
            "contract_id": self.contract_id,
            "relative_gap_threshold": self.relative_gap_threshold,
            "absolute_gap_threshold": self.absolute_gap_threshold,
            "validation_errors": "|".join(self.validation_errors),
            "certification_reasons": "|".join(self.certification_reasons),
        }


NORMALIZED_STAGE_COLUMNS = tuple(NormalizedSolverStage.__dataclass_fields__)


@dataclass(frozen=True)
class SolverRunCertification:
    """Certification of a complete lexicographic sequence."""

    stages: tuple[NormalizedSolverStage, ...]
    certification_class: CertificationClass
    all_stages_have_usable_incumbent: bool
    telemetry_complete: bool
    max_relative_gap: float | None
    contract_id: str
    reasons: tuple[str, ...] = ()

    @property
    def certified(self) -> bool:
        return self.certification_class is not CertificationClass.FEASIBLE_UNCERTIFIED

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage_count": len(self.stages),
            "certification_class": self.certification_class.value,
            "certified": self.certified,
            "all_stages_have_usable_incumbent": self.all_stages_have_usable_incumbent,
            "telemetry_complete": self.telemetry_complete,
            "max_relative_gap": self.max_relative_gap,
            "contract_id": self.contract_id,
            "reasons": "|".join(self.reasons),
        }


@dataclass(frozen=True)
class GreedyNoHarmResult:
    """Same-objective, same-constraint CP-SAT versus greedy comparison."""

    objective_name: str
    sense: str
    cp_sat_objective_value: float
    greedy_objective_value: float
    comparable: bool
    passed: bool
    improvement: float | None
    harm_amount: float | None
    allowed_tolerance: float
    reasons: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "objective_name": self.objective_name,
            "sense": self.sense,
            "cp_sat_objective_value": self.cp_sat_objective_value,
            "greedy_objective_value": self.greedy_objective_value,
            "comparable": self.comparable,
            "passed": self.passed,
            "improvement": self.improvement,
            "harm_amount": self.harm_amount,
            "allowed_tolerance": self.allowed_tolerance,
            "reasons": "|".join(self.reasons),
        }


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(missing) if isinstance(missing, bool) else False


def _record_mapping(record: Any) -> dict[str, Any]:
    if isinstance(record, pd.Series):
        return record.to_dict()
    if isinstance(record, Mapping):
        return dict(record)
    if is_dataclass(record) and not isinstance(record, type):
        return asdict(record)
    names = {
        "stage_order",
        "objective_name",
        "objective",
        "sense",
        "solver_status",
        "status",
        "objective_value",
        "best_bound",
        "best_objective_bound",
        "absolute_gap",
        "relative_gap",
        "wall_time_sec",
        "wall_time",
        "branches",
        "conflicts",
        "raw_solver_stats",
    }
    values = {name: getattr(record, name) for name in names if hasattr(record, name)}
    if not values:
        raise SolverCertificationError(
            "solver stage must be a mapping, pandas Series, dataclass, or attribute object"
        )
    return values


def _alias_value(
    record: Mapping[str, Any],
    aliases: tuple[str, ...],
    *,
    errors: list[str],
) -> Any:
    present = [
        (name, record[name])
        for name in aliases
        if name in record and not _is_missing(record[name])
    ]
    if not present:
        return None
    first_name, first_value = present[0]
    for name, value in present[1:]:
        same = value == first_value
        try:
            same = bool(same)
        except (TypeError, ValueError):
            same = False
        if not same:
            errors.append(f"conflicting_aliases:{first_name}:{name}")
    return first_value


def _coerce_number(
    name: str,
    value: Any,
    *,
    errors: list[str],
    minimum: float | None = None,
) -> float | None:
    if _is_missing(value):
        errors.append(f"missing:{name}")
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        errors.append(f"invalid:{name}:not_numeric")
        return None
    result = float(value)
    if not math.isfinite(result):
        errors.append(f"invalid:{name}:not_finite")
        return None
    if minimum is not None and result < minimum:
        errors.append(f"invalid:{name}:below_{minimum}")
        return None
    return result


def _coerce_count(name: str, value: Any, *, errors: list[str]) -> int | None:
    number = _coerce_number(name, value, errors=errors, minimum=0.0)
    if number is None:
        return None
    if not math.isclose(number, round(number), rel_tol=0.0, abs_tol=1e-9):
        errors.append(f"invalid:{name}:not_integer")
        return None
    return int(round(number))


def _normalise_sense(value: Any, errors: list[str]) -> str:
    if _is_missing(value):
        errors.append("missing:sense")
        return ""
    raw = str(value).strip().lower()
    if raw in {"max", "maximize", "maximise", "maximization", "maximisation"}:
        return "max"
    if raw in {"min", "minimize", "minimise", "minimization", "minimisation"}:
        return "min"
    errors.append(f"invalid:sense:{value}")
    return raw


_KNOWN_STATUSES = {"OPTIMAL", "FEASIBLE", "UNKNOWN", "INFEASIBLE", "MODEL_INVALID"}
_INCUMBENT_STATUSES = {"OPTIMAL", "FEASIBLE"}


def _normalise_status(value: Any, errors: list[str]) -> str:
    if _is_missing(value):
        errors.append("missing:solver_status")
        return ""
    status = str(value).strip().upper().split(".")[-1]
    if status not in _KNOWN_STATUSES:
        errors.append(f"invalid:solver_status:{value}")
    return status


def _gap(
    objective_value: float,
    best_bound: float,
    sense: str,
    contract: NearOptimalityContract,
    errors: list[str],
) -> tuple[float, float]:
    directional = (
        best_bound - objective_value if sense == "max" else objective_value - best_bound
    )
    scale = max(1.0, abs(objective_value), abs(best_bound))
    bound_tolerance = float(contract.numerical_tolerance) * scale
    if directional < -bound_tolerance:
        relation = "below" if sense == "max" else "above"
        errors.append(f"invalid:best_bound_{relation}_{sense}_incumbent")
    absolute = max(0.0, directional)
    denominator = max(abs(objective_value), float(contract.denominator_floor))
    return absolute, absolute / denominator


def _validate_reported_gap(
    name: str,
    raw_value: Any,
    computed: float | None,
    contract: NearOptimalityContract,
    errors: list[str],
) -> None:
    if _is_missing(raw_value):
        return
    value = _coerce_number(name, raw_value, errors=errors, minimum=0.0)
    if value is None or computed is None:
        return
    tolerance = float(contract.numerical_tolerance) * max(1.0, abs(computed), abs(value))
    if not math.isclose(value, computed, rel_tol=0.0, abs_tol=tolerance):
        errors.append(f"invalid:{name}:does_not_match_objective_and_bound")


def normalize_solver_stage(
    record: Any,
    contract: NearOptimalityContract = PREDECLARED_NEAR_OPTIMALITY_CONTRACT,
    *,
    strict: bool = True,
) -> NormalizedSolverStage:
    """Normalize and certify one raw CP-SAT stage.

    With the default ``strict=True``, missing or contradictory telemetry raises
    :class:`SolverCertificationError`.  ``strict=False`` is intended for audits
    of legacy outputs; validation errors remain attached to the result and
    force ``FEASIBLE_UNCERTIFIED``.
    """

    if not isinstance(contract, NearOptimalityContract):
        raise SolverCertificationError("contract must be a NearOptimalityContract")
    raw = _record_mapping(record)
    errors: list[str] = []
    reasons: list[str] = []

    stage_order_raw = _alias_value(raw, ("stage_order",), errors=errors)
    if _is_missing(stage_order_raw):
        stage_order = None
    else:
        stage_order = _coerce_count("stage_order", stage_order_raw, errors=errors)
        if stage_order is not None and stage_order <= 0:
            errors.append("invalid:stage_order:not_positive")

    objective_raw = _alias_value(raw, ("objective_name", "objective"), errors=errors)
    objective_name = "" if _is_missing(objective_raw) else str(objective_raw).strip()
    if not objective_name:
        errors.append("missing:objective_name")
    sense = _normalise_sense(_alias_value(raw, ("sense",), errors=errors), errors)
    status = _normalise_status(
        _alias_value(raw, ("solver_status", "status"), errors=errors), errors
    )

    raw_stats = raw.get("raw_solver_stats")
    if not isinstance(raw_stats, Mapping):
        raw_stats = {}
    branches_raw = _alias_value(raw, ("branches",), errors=errors)
    conflicts_raw = _alias_value(raw, ("conflicts",), errors=errors)
    if _is_missing(branches_raw):
        branches_raw = raw_stats.get("branches")
    elif "branches" in raw_stats and not _is_missing(raw_stats["branches"]):
        if branches_raw != raw_stats["branches"]:
            errors.append("conflicting_aliases:branches:raw_solver_stats.branches")
    if _is_missing(conflicts_raw):
        conflicts_raw = raw_stats.get("conflicts")
    elif "conflicts" in raw_stats and not _is_missing(raw_stats["conflicts"]):
        if conflicts_raw != raw_stats["conflicts"]:
            errors.append("conflicting_aliases:conflicts:raw_solver_stats.conflicts")

    wall_time = _coerce_number(
        "wall_time_sec",
        _alias_value(raw, ("wall_time_sec", "wall_time"), errors=errors),
        errors=errors,
        minimum=0.0,
    )
    branches = _coerce_count("branches", branches_raw, errors=errors)
    conflicts = _coerce_count("conflicts", conflicts_raw, errors=errors)

    objective_value_raw = _alias_value(raw, ("objective_value",), errors=errors)
    best_bound_raw = _alias_value(
        raw, ("best_bound", "best_objective_bound"), errors=errors
    )
    objective_value: float | None = None
    best_bound: float | None = None
    absolute_gap: float | None = None
    relative_gap: float | None = None
    usable_incumbent = status in _INCUMBENT_STATUSES
    if usable_incumbent:
        objective_value = _coerce_number(
            "objective_value", objective_value_raw, errors=errors
        )
        best_bound = _coerce_number("best_bound", best_bound_raw, errors=errors)
        if objective_value is not None and best_bound is not None and sense in {"max", "min"}:
            absolute_gap, relative_gap = _gap(
                objective_value, best_bound, sense, contract, errors
            )
    else:
        reasons.append(f"solver_status_{status or 'MISSING'}_has_no_usable_incumbent")
        if not _is_missing(objective_value_raw) or not _is_missing(best_bound_raw):
            errors.append("invalid:objective_or_bound_present_without_usable_incumbent")

    _validate_reported_gap(
        "absolute_gap", raw.get("absolute_gap"), absolute_gap, contract, errors
    )
    _validate_reported_gap(
        "relative_gap", raw.get("relative_gap"), relative_gap, contract, errors
    )

    if status == "OPTIMAL" and absolute_gap is not None:
        scale = max(1.0, abs(objective_value or 0.0), abs(best_bound or 0.0))
        if absolute_gap > float(contract.numerical_tolerance) * scale:
            errors.append("invalid:optimal_status_has_nonzero_gap")

    errors = list(dict.fromkeys(errors))
    telemetry_complete = not errors
    if errors or not usable_incumbent or relative_gap is None:
        certification = CertificationClass.FEASIBLE_UNCERTIFIED
    elif status == "OPTIMAL":
        certification = CertificationClass.OPTIMAL
    else:
        within_relative = relative_gap <= float(contract.relative_gap_threshold)
        within_absolute = (
            contract.absolute_gap_threshold is None
            or (absolute_gap is not None and absolute_gap <= float(contract.absolute_gap_threshold))
        )
        if within_relative and within_absolute:
            certification = CertificationClass.CERTIFIED_NEAR_OPTIMAL
        else:
            certification = CertificationClass.FEASIBLE_UNCERTIFIED
            if not within_relative:
                reasons.append("relative_gap_above_predeclared_threshold")
            if not within_absolute:
                reasons.append("absolute_gap_above_predeclared_threshold")

    if errors:
        reasons.append("telemetry_incomplete_or_inconsistent")
    reasons = list(dict.fromkeys(reasons))
    normalized = NormalizedSolverStage(
        stage_order=stage_order,
        objective_name=objective_name,
        sense=sense,
        solver_status=status,
        objective_value=objective_value,
        best_bound=best_bound,
        absolute_gap=absolute_gap,
        relative_gap=relative_gap,
        wall_time_sec=wall_time,
        branches=branches,
        conflicts=conflicts,
        usable_incumbent=usable_incumbent,
        telemetry_complete=telemetry_complete,
        certification_class=certification,
        contract_id=contract.contract_id,
        relative_gap_threshold=float(contract.relative_gap_threshold),
        absolute_gap_threshold=(
            None
            if contract.absolute_gap_threshold is None
            else float(contract.absolute_gap_threshold)
        ),
        validation_errors=tuple(errors),
        certification_reasons=tuple(reasons),
    )
    if strict and errors:
        raise SolverCertificationError(
            "Invalid CP-SAT stage telemetry: " + "; ".join(errors)
        )
    return normalized


def _records(records: Any) -> list[Any]:
    if isinstance(records, pd.DataFrame):
        return records.to_dict(orient="records")
    if isinstance(records, (Mapping, pd.Series)) or is_dataclass(records):
        return [records]
    if isinstance(records, Iterable) and not isinstance(records, (str, bytes)):
        return list(records)
    raise SolverCertificationError("records must be a stage record, iterable, or DataFrame")


def normalize_solver_stages(
    records: Any,
    contract: NearOptimalityContract = PREDECLARED_NEAR_OPTIMALITY_CONTRACT,
    *,
    strict: bool = True,
) -> pd.DataFrame:
    """Return a stable, CSV-ready table of canonical stage telemetry."""

    normalized = [
        normalize_solver_stage(record, contract, strict=strict) for record in _records(records)
    ]
    columns = list(normalized[0].as_dict()) if normalized else [
        "stage_order",
        "objective_name",
        "sense",
        "solver_status",
        "objective_value",
        "best_bound",
        "absolute_gap",
        "relative_gap",
        "wall_time_sec",
        "branches",
        "conflicts",
        "usable_incumbent",
        "telemetry_complete",
        "certification_class",
        "certified",
        "contract_id",
        "relative_gap_threshold",
        "absolute_gap_threshold",
        "validation_errors",
        "certification_reasons",
    ]
    return pd.DataFrame([stage.as_dict() for stage in normalized], columns=columns)


def certify_solver_run(
    records: Any,
    contract: NearOptimalityContract = PREDECLARED_NEAR_OPTIMALITY_CONTRACT,
    *,
    strict: bool = True,
) -> SolverRunCertification:
    """Certify a lexicographic run; every objective stage must be certified."""

    stages = tuple(
        normalize_solver_stage(record, contract, strict=strict) for record in _records(records)
    )
    if not stages:
        raise SolverCertificationError("Cannot certify an empty solver-stage sequence")
    orders = [stage.stage_order for stage in stages if stage.stage_order is not None]
    if orders and (len(orders) != len(stages) or len(set(orders)) != len(orders)):
        raise SolverCertificationError("stage_order must be present and unique for every stage")

    labels = {stage.certification_class for stage in stages}
    if labels == {CertificationClass.OPTIMAL}:
        classification = CertificationClass.OPTIMAL
    elif CertificationClass.FEASIBLE_UNCERTIFIED not in labels:
        classification = CertificationClass.CERTIFIED_NEAR_OPTIMAL
    else:
        classification = CertificationClass.FEASIBLE_UNCERTIFIED
    gaps = [stage.relative_gap for stage in stages if stage.relative_gap is not None]
    reasons = tuple(
        f"stage_{stage.stage_order or index}:{reason}"
        for index, stage in enumerate(stages, start=1)
        for reason in (*stage.validation_errors, *stage.certification_reasons)
    )
    return SolverRunCertification(
        stages=stages,
        certification_class=classification,
        all_stages_have_usable_incumbent=all(stage.usable_incumbent for stage in stages),
        telemetry_complete=all(stage.telemetry_complete for stage in stages),
        max_relative_gap=max(gaps) if gaps else None,
        contract_id=contract.contract_id,
        reasons=reasons,
    )


def compare_greedy_no_harm(
    *,
    cp_sat_objective_value: Real,
    greedy_objective_value: Real,
    objective_name: str,
    sense: str,
    greedy_objective_name: str,
    greedy_sense: str,
    same_constraints: bool,
    absolute_tolerance: float = 0.0,
    relative_tolerance: float = 0.0,
) -> GreedyNoHarmResult:
    """Check that CP-SAT does no harm versus a truly comparable greedy result.

    Comparability is fail-closed: objective name, objective sense, and the
    caller's explicit assertion of identical constraints must all agree.
    Positive ``improvement`` always means CP-SAT is better, for both max and min
    objectives.
    """

    cp_value = _require_finite_number("cp_sat_objective_value", cp_sat_objective_value)
    greedy_value = _require_finite_number("greedy_objective_value", greedy_objective_value)
    abs_tol = _require_finite_number("absolute_tolerance", absolute_tolerance, minimum=0.0)
    rel_tol = _require_finite_number("relative_tolerance", relative_tolerance, minimum=0.0)
    errors: list[str] = []
    normalized_sense = _normalise_sense(sense, errors)
    normalized_greedy_sense = _normalise_sense(greedy_sense, errors)
    if errors:
        raise SolverCertificationError("Invalid objective sense: " + "; ".join(errors))
    name = str(objective_name).strip()
    greedy_name = str(greedy_objective_name).strip()
    if not name or not greedy_name:
        raise SolverCertificationError("Both objective names must be non-empty")

    reasons: list[str] = []
    if name != greedy_name:
        reasons.append("objective_name_mismatch")
    if normalized_sense != normalized_greedy_sense:
        reasons.append("objective_sense_mismatch")
    if same_constraints is not True:
        reasons.append("identical_constraints_not_confirmed")
    comparable = not reasons
    allowed = max(abs_tol, rel_tol * max(abs(greedy_value), 1.0))
    improvement: float | None = None
    harm: float | None = None
    passed = False
    if comparable:
        improvement = (
            cp_value - greedy_value
            if normalized_sense == "max"
            else greedy_value - cp_value
        )
        harm = max(0.0, -improvement)
        passed = improvement >= -allowed
        if not passed:
            reasons.append("cp_sat_incumbent_worse_than_same_constraint_greedy")
    return GreedyNoHarmResult(
        objective_name=name,
        sense=normalized_sense,
        cp_sat_objective_value=cp_value,
        greedy_objective_value=greedy_value,
        comparable=comparable,
        passed=passed,
        improvement=improvement,
        harm_amount=harm,
        allowed_tolerance=allowed,
        reasons=tuple(reasons),
    )


def assert_greedy_no_harm(result: GreedyNoHarmResult) -> None:
    """Raise when a same-objective greedy no-harm gate is not satisfied."""

    if not isinstance(result, GreedyNoHarmResult):
        raise TypeError("result must be a GreedyNoHarmResult")
    if not result.passed:
        detail = "; ".join(result.reasons) or "greedy no-harm comparison failed"
        raise SolverCertificationError(detail)


__all__ = [
    "CertificationClass",
    "DEFAULT_NEAR_OPTIMALITY_CONTRACT",
    "DEFAULT_NEAR_OPTIMAL_RELATIVE_GAP",
    "GreedyNoHarmResult",
    "NearOptimalityContract",
    "NormalizedSolverStage",
    "PREDECLARED_NEAR_OPTIMALITY_CONTRACT",
    "SolverCertificationError",
    "SolverRunCertification",
    "assert_greedy_no_harm",
    "certify_solver_run",
    "compare_greedy_no_harm",
    "near_optimality_contract_from_config",
    "normalize_solver_stage",
    "normalize_solver_stages",
]
