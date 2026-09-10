"""Independent SciPy/HiGHS MILP backend for Stage 4 certification.

This backend deliberately does not share the CP-SAT model builder.  It does
reuse the lossless candidate-incidence + sigungu compression contract and the
public Stage 4 result types, allowing the two solvers to be compared without
changing downstream reporting.

Coverage variables are continuous in ``[0, 1]`` and constrained not to exceed
the number of selected candidates covering their group.  This is an exact
projection onto binary candidate selections for the Stage 4 model: all
coverage objective coefficients are non-negative, coverage floors cannot be
overstated, and every actually covered group can take value one.  The final
objective values are nevertheless recomputed from the selected binary venues
and checked against the uncompressed matrix before they are released.
"""

from __future__ import annotations

import math
import time
import warnings
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, OptimizeResult, milp

from .compressed_optimizer import (
    CompressedCoverageGroups,
    assert_compression_exact,
    build_compressed_coverage_groups,
)
from .optimizer import (
    ObjectiveStep,
    PreparedWeights,
    _candidate_constraint_data,
    _greedy_select,
    _metrics_from_selection,
    _objective_plan,
    prepare_weights,
)
from .types import OptimizationInput, ScenarioSolution, SolveStageResult
from .utils import ceil_fraction


class HighsCertificationError(RuntimeError):
    """Raised when compression or a returned HiGHS incumbent is inconsistent."""


def constrained_greedy_efficiency_floor(
    data: OptimizationInput,
    config: dict[str, Any],
    visit_count: int,
) -> int:
    """Return a validated same-base-constraint greedy population floor.

    The floor operationalizes the frozen no-harm rule for the *final*
    lexicographic Efficiency policy.  It is not a tolerance: every later
    objective stage must retain at least this exact deterministic-greedy value.
    """

    data.validate()
    weights = prepare_weights(data, config)
    selected, covered = _greedy_select(
        data, weights, int(visit_count), "efficiency", config
    )
    if len(selected) != int(visit_count) or len(set(selected)) != int(visit_count):
        raise HighsCertificationError(
            "Constrained greedy floor requires the exact requested visit count"
        )
    constraints = _candidate_constraint_data(data.candidates, config)
    chosen = np.asarray(selected, dtype=np.int64)
    if constraints["one_per_cluster"] and bool(
        config["optimization"]["cluster_constraint"]
    ):
        if len(set(constraints["clusters"][chosen])) != len(chosen):
            raise HighsCertificationError("Constrained greedy floor violates cluster cap")
    for labels, maximum, label in (
        (constraints["admins"], constraints["max_per_admin"], "admin"),
        (constraints["sigungu"], constraints["max_per_sigungu"], "sigungu"),
    ):
        counts = pd.Series(labels[chosen]).value_counts()
        if not counts.empty and int(counts.max()) > int(maximum):
            raise HighsCertificationError(
                f"Constrained greedy floor violates {label} cap"
            )
    if bool(config["optimization"]["exclude_known_overlap"]) and np.any(
        constraints["known_overlap"][chosen] == 1
    ):
        raise HighsCertificationError(
            "Constrained greedy floor violates known-overlap exclusion"
        )
    floor = int(weights.population[np.asarray(covered, dtype=bool)].sum())
    if floor <= 0:
        raise HighsCertificationError(
            "Constrained greedy floor must be a positive population value"
        )
    return floor


@dataclass(frozen=True)
class _VariableLayout:
    candidate_count: int
    group_count: int
    sigungus: tuple[str, ...]

    @property
    def x_start(self) -> int:
        return 0

    @property
    def y_start(self) -> int:
        return self.candidate_count

    @property
    def z_index(self) -> int:
        return self.candidate_count + self.group_count

    @property
    def deviation_start(self) -> int:
        return self.z_index + 1

    @property
    def variable_count(self) -> int:
        return self.deviation_start + len(self.sigungus)

    def x(self, index: int) -> int:
        return self.x_start + index

    def y(self, index: int) -> int:
        return self.y_start + index

    def deviation(self, sigungu: str) -> int:
        return self.deviation_start + self.sigungus.index(sigungu)


class _ConstraintBuilder:
    """Sparse row builder for SciPy's two-sided LinearConstraint."""

    def __init__(self, variable_count: int) -> None:
        self.variable_count = int(variable_count)
        self.rows: list[int] = []
        self.columns: list[int] = []
        self.values: list[float] = []
        self.lower: list[float] = []
        self.upper: list[float] = []

    def add(
        self,
        terms: Iterable[tuple[int, float]],
        *,
        lower: float = -math.inf,
        upper: float = math.inf,
    ) -> None:
        row = len(self.lower)
        seen = False
        for column, value in terms:
            if not value:
                continue
            if not 0 <= int(column) < self.variable_count:
                raise IndexError(f"MILP column {column} is outside the variable layout")
            self.rows.append(row)
            self.columns.append(int(column))
            self.values.append(float(value))
            seen = True
        if not seen and math.isinf(lower) and math.isinf(upper):
            raise ValueError("Empty unconstrained MILP row is not meaningful")
        self.lower.append(float(lower))
        self.upper.append(float(upper))

    def build(self) -> LinearConstraint:
        matrix = sparse.coo_matrix(
            (self.values, (self.rows, self.columns)),
            shape=(len(self.lower), self.variable_count),
            dtype=np.float64,
        ).tocsc()
        return LinearConstraint(
            matrix,
            np.asarray(self.lower, dtype=np.float64),
            np.asarray(self.upper, dtype=np.float64),
        )


@dataclass(frozen=True)
class _HighsModel:
    c: np.ndarray
    integrality: np.ndarray
    bounds: Bounds
    constraints: LinearConstraint
    layout: _VariableLayout
    candidate_constraints: dict[str, Any]
    target_visits_per_sigungu: int


def _arrays_equal(left: np.ndarray, right: np.ndarray) -> bool:
    return left.shape == right.shape and np.array_equal(left, right)


def _assert_groups_equal(
    actual: CompressedCoverageGroups,
    expected: CompressedCoverageGroups,
) -> None:
    failures: list[str] = []
    if actual.candidate_members != expected.candidate_members:
        failures.append("candidate_members")
    for field in ["sigungu", "population", "need_weighted", "high_need_population"]:
        if not _arrays_equal(getattr(actual, field), getattr(expected, field)):
            failures.append(field)
    for field in ["source_grid_count", "covered_source_grid_count", "source_nnz"]:
        if int(getattr(actual, field)) != int(getattr(expected, field)):
            failures.append(field)
    if failures:
        raise HighsCertificationError(
            "Provided coverage compression is not the lossless artifact for this input: "
            + ", ".join(failures)
        )


def _resolve_groups(
    data: OptimizationInput,
    weights: PreparedWeights,
    groups: CompressedCoverageGroups | None,
    *,
    validate_provided: bool,
) -> CompressedCoverageGroups:
    if groups is None:
        return build_compressed_coverage_groups(data, weights)
    if validate_provided:
        _assert_groups_equal(groups, build_compressed_coverage_groups(data, weights))
    return groups


def _expression_terms(
    name: str,
    layout: _VariableLayout,
    groups: CompressedCoverageGroups,
    candidate_constraints: dict[str, Any],
    target_visits_per_sigungu: int,
    config: dict[str, Any],
) -> list[tuple[int, float]]:
    if name == "total_population":
        return [
            (layout.y(index), int(groups.population[index]))
            for index in range(groups.group_count)
            if int(groups.population[index])
        ]
    if name == "need_weighted":
        return [
            (layout.y(index), int(groups.need_weighted[index]))
            for index in range(groups.group_count)
            if int(groups.need_weighted[index])
        ]
    if name == "high_need_population":
        return [
            (layout.y(index), int(groups.high_need_population[index]))
            for index in range(groups.group_count)
            if int(groups.high_need_population[index])
        ]
    if name == "min_sigungu_coverage":
        return [(layout.z_index, 1.0)]
    if name == "cost":
        travel_integer = np.rint(
            candidate_constraints["travel_minutes"] * 100
        ).astype(np.int64)
        travel_scale = int(config["optimization"]["travel_penalty_scale"])
        overlap_penalty = int(config["optimization"]["known_overlap_penalty"])
        terms: list[tuple[int, float]] = []
        for index in range(layout.candidate_count):
            coefficient = int(travel_integer[index] * travel_scale)
            if int(candidate_constraints["known_overlap"][index]) == 1:
                coefficient += overlap_penalty
            if coefficient:
                terms.append((layout.x(index), coefficient))
        terms.extend((layout.deviation(sigungu), 100.0) for sigungu in layout.sigungus)
        return terms
    raise KeyError(f"Unknown Stage 4 objective {name!r}")


def _build_highs_model(
    data: OptimizationInput,
    weights: PreparedWeights,
    groups: CompressedCoverageGroups,
    config: dict[str, Any],
    visit_count: int,
    step: ObjectiveStep,
    floors: dict[str, int],
    scenario: str,
    efficiency_reference: int | None,
) -> _HighsModel:
    candidate_constraints = _candidate_constraint_data(data.candidates, config)
    sigungus = tuple(sorted(set(candidate_constraints["sigungu"])))
    layout = _VariableLayout(len(data.candidates), groups.group_count, sigungus)
    if not sigungus:
        raise ValueError("HiGHS model requires at least one candidate sigungu")
    target = int(round(visit_count / len(sigungus)))

    lower = np.zeros(layout.variable_count, dtype=np.float64)
    upper = np.full(layout.variable_count, np.inf, dtype=np.float64)
    upper[: layout.candidate_count] = 1.0
    upper[layout.y_start : layout.z_index] = 1.0
    upper[layout.z_index] = 1000.0
    upper[layout.deviation_start :] = float(visit_count)
    if bool(config["optimization"]["exclude_known_overlap"]):
        excluded = np.flatnonzero(candidate_constraints["known_overlap"] == 1)
        upper[excluded] = 0.0

    integrality = np.zeros(layout.variable_count, dtype=np.uint8)
    integrality[: layout.candidate_count] = 1
    integrality[layout.z_index] = 1
    integrality[layout.deviation_start :] = 1
    builder = _ConstraintBuilder(layout.variable_count)

    builder.add(
        ((layout.x(index), 1.0) for index in range(layout.candidate_count)),
        lower=visit_count,
        upper=visit_count,
    )
    if candidate_constraints["one_per_cluster"] and bool(
        config["optimization"]["cluster_constraint"]
    ):
        for cluster in sorted(set(candidate_constraints["clusters"])):
            indices = np.flatnonzero(candidate_constraints["clusters"] == cluster)
            if len(indices) > 1:
                builder.add(
                    ((layout.x(int(index)), 1.0) for index in indices),
                    upper=1.0,
                )
    for admin in sorted(set(candidate_constraints["admins"])):
        indices = np.flatnonzero(candidate_constraints["admins"] == admin)
        builder.add(
            ((layout.x(int(index)), 1.0) for index in indices),
            upper=float(candidate_constraints["max_per_admin"]),
        )

    minimum_visits = 0
    if scenario == "equity_hard":
        minimum_visits = int(
            config["optimization"]["min_visits_per_sigungu_equity_hard"] or 1
        )
    for sigungu in sigungus:
        indices = np.flatnonzero(candidate_constraints["sigungu"] == sigungu)
        builder.add(
            ((layout.x(int(index)), 1.0) for index in indices),
            lower=float(minimum_visits) if minimum_visits else -math.inf,
            upper=float(candidate_constraints["max_per_sigungu"]),
        )

    # Continuous y cannot claim coverage without at least one selected member.
    # Positive objectives/floors make y=1 attainable exactly whenever covered.
    for group_index, members in enumerate(groups.candidate_members):
        builder.add(
            [
                (layout.y(group_index), 1.0),
                *((layout.x(int(index)), -1.0) for index in members),
            ],
            upper=0.0,
        )

    for sigungu, total in weights.total_by_sigungu.items():
        if total <= 0:
            continue
        relevant = np.flatnonzero(groups.sigungu == sigungu)
        terms = [
            (layout.y(int(index)), 1000.0 * int(groups.population[int(index)]))
            for index in relevant
            if int(groups.population[int(index)])
        ]
        terms.append((layout.z_index, -float(total)))
        builder.add(terms, lower=0.0)

    # d_s >= |sum(x in s) - round(visits / number_of_sigungus)|.
    for sigungu in sigungus:
        indices = np.flatnonzero(candidate_constraints["sigungu"] == sigungu)
        builder.add(
            [
                (layout.deviation(sigungu), 1.0),
                *((layout.x(int(index)), -1.0) for index in indices),
            ],
            lower=-float(target),
        )
        builder.add(
            [
                (layout.deviation(sigungu), 1.0),
                *((layout.x(int(index)), 1.0) for index in indices),
            ],
            lower=float(target),
        )

    for name, floor in floors.items():
        builder.add(
            _expression_terms(
                name,
                layout,
                groups,
                candidate_constraints,
                target,
                config,
            ),
            lower=float(floor),
        )
    if efficiency_reference is not None:
        if scenario == "balanced":
            fraction = float(
                config["optimization"]["balanced_min_efficiency_fraction"]
            )
        elif scenario in {"equity", "equity_hard"}:
            fraction = float(config["optimization"]["equity_min_efficiency_fraction"])
        else:
            fraction = 0.0
        if fraction:
            builder.add(
                _expression_terms(
                    "total_population",
                    layout,
                    groups,
                    candidate_constraints,
                    target,
                    config,
                ),
                lower=float(ceil_fraction(efficiency_reference, fraction)),
            )

    objective = np.zeros(layout.variable_count, dtype=np.float64)
    for index, coefficient in _expression_terms(
        step.name,
        layout,
        groups,
        candidate_constraints,
        target,
        config,
    ):
        objective[index] += float(coefficient)
    if step.sense == "max":
        objective *= -1.0
    elif step.sense != "min":
        raise ValueError(f"Unknown objective sense {step.sense!r}")
    return _HighsModel(
        c=objective,
        integrality=integrality,
        bounds=Bounds(lower, upper),
        constraints=builder.build(),
        layout=layout,
        candidate_constraints=candidate_constraints,
        target_visits_per_sigungu=target,
    )


def _covered_groups(
    groups: CompressedCoverageGroups,
    selected_indices: list[int],
) -> np.ndarray:
    selected = set(map(int, selected_indices))
    return np.fromiter(
        (
            any(candidate in selected for candidate in members)
            for members in groups.candidate_members
        ),
        dtype=bool,
        count=groups.group_count,
    )


def _exact_objective_values(
    groups: CompressedCoverageGroups,
    weights: PreparedWeights,
    selected_indices: list[int],
    model: _HighsModel,
    config: dict[str, Any],
) -> dict[str, int]:
    covered = _covered_groups(groups, selected_indices)
    values = {
        "total_population": int(groups.population[covered].sum()),
        "need_weighted": int(groups.need_weighted[covered].sum()),
        "high_need_population": int(groups.high_need_population[covered].sum()),
    }
    ratios: list[int] = []
    for sigungu, total in weights.total_by_sigungu.items():
        if total <= 0:
            continue
        numerator = int(
            groups.population[covered & (groups.sigungu == sigungu)].sum()
        )
        ratios.append((numerator * 1000) // int(total))
    values["min_sigungu_coverage"] = min(ratios) if ratios else 0

    selected = np.asarray(selected_indices, dtype=np.int64)
    constraints = model.candidate_constraints
    travel_integer = np.rint(constraints["travel_minutes"] * 100).astype(np.int64)
    travel_scale = int(config["optimization"]["travel_penalty_scale"])
    overlap_penalty = int(config["optimization"]["known_overlap_penalty"])
    cost = int(travel_integer[selected].sum() * travel_scale) if selected.size else 0
    if selected.size:
        cost += int((constraints["known_overlap"][selected] == 1).sum()) * overlap_penalty
    selected_sigungu = constraints["sigungu"][selected] if selected.size else np.asarray([])
    cost += 100 * sum(
        abs(int((selected_sigungu == sigungu).sum()) - model.target_visits_per_sigungu)
        for sigungu in model.layout.sigungus
    )
    values["cost"] = int(cost)
    return values


def _validate_selected(
    data: OptimizationInput,
    weights: PreparedWeights,
    groups: CompressedCoverageGroups,
    selected_indices: list[int],
    model: _HighsModel,
    config: dict[str, Any],
    visit_count: int,
    floors: dict[str, int],
    scenario: str,
    efficiency_reference: int | None,
) -> dict[str, int]:
    errors: list[str] = []
    if len(selected_indices) != visit_count or len(set(selected_indices)) != visit_count:
        errors.append(f"selected_count={len(selected_indices)} expected={visit_count}")
    constraints = model.candidate_constraints
    selected = np.asarray(selected_indices, dtype=np.int64)
    if selected.size and (selected.min() < 0 or selected.max() >= len(data.candidates)):
        errors.append("selected candidate index outside input")
    if selected.size:
        if constraints["one_per_cluster"] and bool(
            config["optimization"]["cluster_constraint"]
        ):
            if len(set(constraints["clusters"][selected])) != len(selected):
                errors.append("coverage-cluster cap violated")
        for labels, maximum, label in [
            (constraints["admins"], constraints["max_per_admin"], "admin"),
            (constraints["sigungu"], constraints["max_per_sigungu"], "sigungu"),
        ]:
            counts = pd.Series(labels[selected]).value_counts()
            if not counts.empty and int(counts.max()) > int(maximum):
                errors.append(f"{label} cap violated")
        if bool(config["optimization"]["exclude_known_overlap"]) and np.any(
            constraints["known_overlap"][selected] == 1
        ):
            errors.append("excluded known-overlap candidate selected")
        if scenario == "equity_hard":
            minimum = int(
                config["optimization"]["min_visits_per_sigungu_equity_hard"] or 1
            )
            counts = pd.Series(constraints["sigungu"][selected]).value_counts()
            for sigungu in model.layout.sigungus:
                if int(counts.get(sigungu, 0)) < minimum:
                    errors.append(f"equity-hard minimum violated for {sigungu}")

    if errors:
        raise HighsCertificationError("Invalid HiGHS incumbent: " + "; ".join(errors))
    assert_compression_exact(data, weights, groups, selected_indices)
    exact = _exact_objective_values(groups, weights, selected_indices, model, config)
    for name, floor in floors.items():
        if exact[name] < int(floor):
            errors.append(f"floor {name}: {exact[name]} < {floor}")
    if efficiency_reference is not None:
        if scenario == "balanced":
            required = ceil_fraction(
                efficiency_reference,
                float(config["optimization"]["balanced_min_efficiency_fraction"]),
            )
        elif scenario in {"equity", "equity_hard"}:
            required = ceil_fraction(
                efficiency_reference,
                float(config["optimization"]["equity_min_efficiency_fraction"]),
            )
        else:
            required = 0
        if exact["total_population"] < required:
            errors.append(
                f"efficiency floor: {exact['total_population']} < {required}"
            )
    if errors:
        raise HighsCertificationError("Invalid HiGHS incumbent: " + "; ".join(errors))
    return exact


def _finite_result_value(result: OptimizeResult, name: str) -> float | None:
    value = getattr(result, name, None)
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _raw_stats(
    result: OptimizeResult,
    *,
    step: ObjectiveStep,
    objective_value: int | None,
    best_bound: float | None,
    elapsed: float,
    groups: CompressedCoverageGroups,
) -> dict[str, Any]:
    absolute_gap: float | None = None
    relative_gap: float | None = None
    if objective_value is not None and best_bound is not None:
        directional = (
            best_bound - objective_value
            if step.sense == "max"
            else objective_value - best_bound
        )
        absolute_gap = float(max(0.0, directional))
        relative_gap = absolute_gap / max(1.0, abs(float(objective_value)))
    node_count = _finite_result_value(result, "mip_node_count")
    return {
        "backend": "scipy.optimize.milp/HiGHS",
        "scipy_status": int(getattr(result, "status", -1)),
        "scipy_success": bool(getattr(result, "success", False)),
        "message": str(getattr(result, "message", "")),
        "primal_objective": objective_value,
        "dual_bound": best_bound,
        "absolute_gap": absolute_gap,
        "relative_gap": relative_gap,
        "mip_gap": _finite_result_value(result, "mip_gap"),
        "mip_node_count": None if node_count is None else int(round(node_count)),
        "transformed_minimization_fun": _finite_result_value(result, "fun"),
        "transformed_minimization_dual_bound": _finite_result_value(
            result, "mip_dual_bound"
        ),
        "wall_time_sec": float(elapsed),
        "compressed_group_count": groups.group_count,
        "covered_source_grid_count": groups.covered_source_grid_count,
        "source_grid_count": groups.source_grid_count,
        "source_nnz": groups.source_nnz,
        "compression_ratio": groups.compression_ratio,
    }


def _empty_stage(
    step: ObjectiveStep,
    status: str,
    elapsed: float,
    result: OptimizeResult,
    groups: CompressedCoverageGroups,
    *,
    validation_error: str | None = None,
) -> SolveStageResult:
    stats = _raw_stats(
        result,
        step=step,
        objective_value=None,
        best_bound=None,
        elapsed=elapsed,
        groups=groups,
    )
    if validation_error:
        stats["solution_validation_error"] = validation_error
    return SolveStageResult(
        objective_name=step.name,
        sense=step.sense,
        status=status,
        objective_value=None,
        best_bound=None,
        wall_time_sec=float(elapsed),
        selected_indices=[],
        covered_grid_indices=[],
        raw_solver_stats=stats,
    )


def _solve_highs_stage(
    data: OptimizationInput,
    weights: PreparedWeights,
    groups: CompressedCoverageGroups,
    config: dict[str, Any],
    visit_count: int,
    step: ObjectiveStep,
    floors: dict[str, int],
    scenario: str,
    efficiency_reference: int | None,
    time_limit_sec: float,
    relative_gap_target: float,
    threads: int | None,
    random_seed: int | None,
) -> SolveStageResult:
    model = _build_highs_model(
        data,
        weights,
        groups,
        config,
        visit_count,
        step,
        floors,
        scenario,
        efficiency_reference,
    )
    started = time.perf_counter()
    options: dict[str, Any] = {
        "time_limit": float(time_limit_sec),
        "mip_rel_gap": float(relative_gap_target),
        "presolve": True,
        "disp": bool(config["runtime"].get("log_solver_progress", False)),
    }
    # SciPy forwards unknown options verbatim to HiGHS.  Explicitly passing
    # these two native HiGHS options makes the resource/determinism contract
    # executable instead of merely recording it in metadata.
    if threads is not None:
        options["threads"] = int(threads)
    if random_seed is not None:
        options["random_seed"] = int(random_seed)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Unrecognized options detected:.*passed to HiGHS verbatim.*",
            category=RuntimeWarning,
        )
        result = milp(
            model.c,
            integrality=model.integrality,
            bounds=model.bounds,
            constraints=model.constraints,
            options=options,
        )
    elapsed = time.perf_counter() - started
    scipy_status = int(getattr(result, "status", -1))
    x_values = getattr(result, "x", None)
    has_incumbent = (
        scipy_status in {0, 1}
        and x_values is not None
        and len(x_values) == model.layout.variable_count
        and np.isfinite(np.asarray(x_values, dtype=float)).all()
    )
    if not has_incumbent:
        status = {
            1: "UNKNOWN",
            2: "INFEASIBLE",
            3: "UNBOUNDED",
            4: "UNKNOWN",
        }.get(scipy_status, "MODEL_INVALID")
        empty = _empty_stage(step, status, elapsed, result, groups)
        empty.raw_solver_stats["requested_threads"] = threads
        empty.raw_solver_stats["requested_random_seed"] = random_seed
        return empty

    candidate_values = np.asarray(x_values[: model.layout.candidate_count], dtype=float)
    distance = np.abs(candidate_values - np.rint(candidate_values))
    if np.any(distance > 1e-5):
        empty = _empty_stage(
            step,
            "MODEL_INVALID",
            elapsed,
            result,
            groups,
            validation_error="HiGHS returned non-integral candidate variables",
        )
        empty.raw_solver_stats["requested_threads"] = threads
        empty.raw_solver_stats["requested_random_seed"] = random_seed
        return empty
    selected = np.flatnonzero(np.rint(candidate_values).astype(np.int8) == 1).tolist()
    try:
        exact = _validate_selected(
            data,
            weights,
            groups,
            selected,
            model,
            config,
            visit_count,
            floors,
            scenario,
            efficiency_reference,
        )
    except HighsCertificationError as exc:
        empty = _empty_stage(
            step,
            "MODEL_INVALID",
            elapsed,
            result,
            groups,
            validation_error=str(exc),
        )
        empty.raw_solver_stats["requested_threads"] = threads
        empty.raw_solver_stats["requested_random_seed"] = random_seed
        return empty

    objective_value = int(exact[step.name])
    transformed_bound = _finite_result_value(result, "mip_dual_bound")
    if transformed_bound is None:
        best_bound = None
    else:
        best_bound = -transformed_bound if step.sense == "max" else transformed_bound
    covered = np.asarray(data.coverage[selected].getnnz(axis=0)).ravel() > 0
    if best_bound is None:
        certified_relative_gap = math.inf
    else:
        directional_gap = (
            best_bound - objective_value
            if step.sense == "max"
            else objective_value - best_bound
        )
        certified_relative_gap = max(0.0, directional_gap) / max(
            abs(objective_value), 1.0
        )
    status = (
        "OPTIMAL"
        if scipy_status == 0 and certified_relative_gap <= 1e-12
        else "FEASIBLE"
    )
    stats = _raw_stats(
        result,
        step=step,
        objective_value=objective_value,
        best_bound=best_bound,
        elapsed=elapsed,
        groups=groups,
    )
    transformed_primal = _finite_result_value(result, "fun")
    if transformed_primal is not None:
        reported_original = (
            -transformed_primal if step.sense == "max" else transformed_primal
        )
        stats["reported_original_primal"] = reported_original
        stats["exact_primal_minus_reported"] = float(
            objective_value - reported_original
        )
    stats["requested_relative_gap_target"] = float(relative_gap_target)
    stats["requested_threads"] = threads
    stats["requested_random_seed"] = random_seed
    stats["certified_relative_gap_from_exact_primal"] = float(
        certified_relative_gap
    )
    return SolveStageResult(
        objective_name=step.name,
        sense=step.sense,
        status=status,
        objective_value=objective_value,
        best_bound=best_bound,
        wall_time_sec=float(elapsed),
        selected_indices=selected,
        covered_grid_indices=np.flatnonzero(covered).astype(int).tolist(),
        raw_solver_stats=stats,
    )


def _solve_highs_scenario(
    data: OptimizationInput,
    config: dict[str, Any],
    scenario: str,
    visit_count: int,
    *,
    efficiency_reference: int | None,
    time_limit_sec: float | None,
    groups: CompressedCoverageGroups | None,
    validate_provided_groups: bool,
    relative_gap_target: float,
    threads: int | None,
    random_seed: int | None,
    minimum_total_population_floor: int | None,
) -> ScenarioSolution:
    data.validate()
    if int(visit_count) <= 0:
        raise ValueError("visit_count must be positive")
    limit = (
        float(config["runtime"]["max_time_per_stage_sec"])
        if time_limit_sec is None
        else float(time_limit_sec)
    )
    if not math.isfinite(limit) or limit <= 0:
        raise ValueError("time_limit_sec must be finite and positive")
    if (
        not math.isfinite(float(relative_gap_target))
        or not 0.0 <= float(relative_gap_target) <= 1.0
    ):
        raise ValueError("relative_gap_target must be finite and in [0, 1]")
    if threads is not None and int(threads) <= 0:
        raise ValueError("threads must be positive when provided")
    if random_seed is not None and int(random_seed) < 0:
        raise ValueError("random_seed must be non-negative when provided")
    if minimum_total_population_floor is not None:
        if scenario != "efficiency":
            raise ValueError(
                "minimum_total_population_floor is only valid for efficiency"
            )
        if int(minimum_total_population_floor) <= 0:
            raise ValueError("minimum_total_population_floor must be positive")
    weights = prepare_weights(data, config)
    groups = _resolve_groups(
        data, weights, groups, validate_provided=validate_provided_groups
    )
    plan = _objective_plan(scenario, config)
    floors: dict[str, int] = {}
    if minimum_total_population_floor is not None:
        floors["total_population"] = int(minimum_total_population_floor)
    stages: list[SolveStageResult] = []
    final_selected: list[int] = []
    final_covered: list[int] = []
    for step in plan:
        result = _solve_highs_stage(
            data,
            weights,
            groups,
            config,
            int(visit_count),
            step,
            floors,
            scenario,
            efficiency_reference,
            limit,
            float(relative_gap_target),
            None if threads is None else int(threads),
            None if random_seed is None else int(random_seed),
        )
        stages.append(result)
        result.raw_solver_stats["applied_objective_floors"] = {
            str(name): int(value) for name, value in floors.items()
        }
        result.raw_solver_stats["minimum_total_population_floor"] = (
            None
            if minimum_total_population_floor is None
            else int(minimum_total_population_floor)
        )
        if result.objective_value is None:
            return ScenarioSolution(
                scenario=scenario,
                catchment_id=data.catchment_id,
                visit_count=int(visit_count),
                status=result.status,
                selected=pd.DataFrame(),
                covered_grid_mask=np.zeros(len(data.grids), dtype=bool),
                metrics={},
                stages=stages,
                notes=[
                    f"SciPy/HiGHS failed closed at objective stage {step.name}: "
                    f"{result.raw_solver_stats.get('message', '')}"
                ],
            )
        final_selected = result.selected_indices
        final_covered = result.covered_grid_indices
        if step.sense == "max" and step.retention is not None:
            retained = ceil_fraction(result.objective_value, step.retention)
            floors[step.name] = max(int(floors.get(step.name, retained)), retained)

    covered_mask = np.zeros(len(data.grids), dtype=bool)
    covered_mask[final_covered] = True
    metrics = _metrics_from_selection(
        data, weights, final_selected, covered_mask, int(visit_count)
    )
    metrics.update(
        {
            "compressed_group_count": groups.group_count,
            "covered_source_grid_count": groups.covered_source_grid_count,
            "coverage_compression_ratio": groups.compression_ratio,
            "solver_backend": "scipy.optimize.milp/HiGHS",
        }
    )
    if minimum_total_population_floor is not None:
        final_population = int(metrics["unique_elderly_population_scaled"])
        metrics["minimum_total_population_floor"] = int(
            minimum_total_population_floor
        )
        metrics["final_efficiency_no_harm_margin"] = int(
            final_population - int(minimum_total_population_floor)
        )
    status = "OPTIMAL" if all(stage.status == "OPTIMAL" for stage in stages) else "FEASIBLE"
    return ScenarioSolution(
        scenario=scenario,
        catchment_id=data.catchment_id,
        visit_count=int(visit_count),
        status=status,
        selected=data.candidates.iloc[final_selected].copy(),
        covered_grid_mask=covered_mask,
        metrics=metrics,
        stages=stages,
        notes=[
            "Independent SciPy/HiGHS MILP over lossless coverage groups.",
            "All released coverage objectives were recomputed from binary selections and "
            "asserted against the uncompressed matrix.",
            *(
                [
                    "The final Efficiency policy retained the exact same-base-constraint "
                    f"greedy population floor {int(minimum_total_population_floor)}."
                ]
                if minimum_total_population_floor is not None
                else []
            ),
        ],
    )


def solve_highs_scenario(
    data: OptimizationInput,
    config: dict[str, Any],
    scenario: str,
    visit_count: int,
    *,
    efficiency_reference: int | None = None,
    time_limit_sec: float | None = None,
    groups: CompressedCoverageGroups | None = None,
    solver_name: str | None = None,
    relative_gap_target: float = 0.0,
    threads: int | None = None,
    random_seed: int | None = None,
    minimum_total_population_floor: int | None = None,
) -> ScenarioSolution:
    """Solve one Stage 4 lexicographic scenario with independent HiGHS MILPs."""

    if solver_name not in {None, "highs", "scipy-highs"}:
        raise ValueError("solve_highs_scenario only supports solver_name='highs'")
    return _solve_highs_scenario(
        data,
        config,
        scenario,
        visit_count,
        efficiency_reference=efficiency_reference,
        time_limit_sec=time_limit_sec,
        groups=groups,
        validate_provided_groups=groups is not None,
        relative_gap_target=relative_gap_target,
        threads=threads,
        random_seed=random_seed,
        minimum_total_population_floor=minimum_total_population_floor,
    )


def solve_highs_primary_scenarios(
    data: OptimizationInput,
    config: dict[str, Any],
    visit_count: int,
    *,
    time_limit_sec: float | None = None,
    solver_name: str | None = None,
    relative_gap_target: float = 0.0,
    threads: int | None = None,
    random_seed: int | None = None,
) -> list[ScenarioSolution]:
    """Solve configured primary scenarios using one shared exact compression."""

    if solver_name not in {None, "highs", "scipy-highs"}:
        raise ValueError("solve_highs_primary_scenarios only supports solver_name='highs'")
    data.validate()
    weights = prepare_weights(data, config)
    groups = build_compressed_coverage_groups(data, weights)
    greedy_floor = constrained_greedy_efficiency_floor(data, config, visit_count)
    efficiency = _solve_highs_scenario(
        data,
        config,
        "efficiency",
        visit_count,
        efficiency_reference=None,
        time_limit_sec=time_limit_sec,
        groups=groups,
        validate_provided_groups=False,
        relative_gap_target=relative_gap_target,
        threads=threads,
        random_seed=random_seed,
        minimum_total_population_floor=greedy_floor,
    )
    if not efficiency.metrics:
        raise RuntimeError("HiGHS efficiency solve did not return a usable incumbent")
    efficiency_reference = int(efficiency.metrics["unique_elderly_population_scaled"])
    solutions = [efficiency]
    for configured_scenario in config["optimization"]["scenarios"]:
        scenario = str(configured_scenario)
        if scenario == "efficiency":
            continue
        solutions.append(
            _solve_highs_scenario(
                data,
                config,
                scenario,
                visit_count,
                efficiency_reference=efficiency_reference,
                time_limit_sec=time_limit_sec,
                groups=groups,
                validate_provided_groups=False,
                relative_gap_target=relative_gap_target,
                threads=threads,
                random_seed=random_seed,
                minimum_total_population_floor=None,
            )
        )
    return solutions


__all__ = [
    "HighsCertificationError",
    "constrained_greedy_efficiency_floor",
    "solve_highs_primary_scenarios",
    "solve_highs_scenario",
]
