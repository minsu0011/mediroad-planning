from __future__ import annotations

"""Exact CP-SAT optimizer over losslessly compressed coverage patterns.

Stage 4's production matrix has 58,311 grid columns, but many columns are
covered by exactly the same candidate set.  For every objective used by the
optimizer, columns with the same candidate-incidence pattern and sigungu can
be aggregated without changing any feasible solution or objective value.
"""

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

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


@dataclass(frozen=True)
class CompressedCoverageGroups:
    candidate_members: tuple[tuple[int, ...], ...]
    sigungu: np.ndarray
    population: np.ndarray
    need_weighted: np.ndarray
    high_need_population: np.ndarray
    source_grid_count: int
    covered_source_grid_count: int
    source_nnz: int

    @property
    def group_count(self) -> int:
        return len(self.candidate_members)

    @property
    def compression_ratio(self) -> float:
        return self.group_count / max(1, self.covered_source_grid_count)


def build_compressed_coverage_groups(
    data: OptimizationInput,
    weights: PreparedWeights | None = None,
) -> CompressedCoverageGroups:
    """Aggregate grids by exact candidate incidence and sigungu.

    Including sigungu in the grouping key preserves beneficiary-equity
    constraints exactly.  Empty-incidence grids are excluded from variables
    but remain in the sigungu denominators held by ``PreparedWeights``.
    """

    data.validate()
    weights = weights or prepare_weights(data, {
        "optimization": {
            "population_integer_scale": 1,
            "need_integer_scale": 1,
            "high_need_quantile": 0.8,
        }
    })
    binary = data.coverage.copy().tocsr()
    binary.data = np.ones_like(binary.data, dtype=np.int8)
    binary.eliminate_zeros()
    csc = binary.tocsc()
    grouped: dict[tuple[str, tuple[int, ...]], list[int]] = {}
    covered = 0
    for grid_idx in range(csc.shape[1]):
        members = tuple(
            int(value)
            for value in csc.indices[csc.indptr[grid_idx] : csc.indptr[grid_idx + 1]]
        )
        if not members:
            continue
        covered += 1
        key = (str(weights.grid_sigungu[grid_idx]), members)
        values = grouped.setdefault(key, [0, 0, 0])
        values[0] += int(weights.population[grid_idx])
        values[1] += int(weights.need_weighted[grid_idx])
        values[2] += int(weights.high_need_population[grid_idx])

    ordered = sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1]))
    artifact = CompressedCoverageGroups(
        candidate_members=tuple(key[1] for key, _ in ordered),
        sigungu=np.asarray([key[0] for key, _ in ordered], dtype=object),
        population=np.asarray([values[0] for _, values in ordered], dtype=np.int64),
        need_weighted=np.asarray([values[1] for _, values in ordered], dtype=np.int64),
        high_need_population=np.asarray([values[2] for _, values in ordered], dtype=np.int64),
        source_grid_count=int(csc.shape[1]),
        covered_source_grid_count=int(covered),
        source_nnz=int(csc.nnz),
    )
    if artifact.group_count == 0:
        raise ValueError("Compressed coverage contains no covered grid groups")
    if any(not members for members in artifact.candidate_members):
        raise AssertionError("Compressed coverage must not contain empty incidence patterns")
    return artifact


def compressed_objective_values(
    groups: CompressedCoverageGroups,
    selected_indices: list[int] | np.ndarray,
) -> dict[str, int]:
    selected = set(int(value) for value in selected_indices)
    covered = np.fromiter(
        (any(candidate in selected for candidate in members) for members in groups.candidate_members),
        count=groups.group_count,
        dtype=bool,
    )
    return {
        "total_population": int(groups.population[covered].sum()),
        "need_weighted": int(groups.need_weighted[covered].sum()),
        "high_need_population": int(groups.high_need_population[covered].sum()),
    }


def assert_compression_exact(
    data: OptimizationInput,
    weights: PreparedWeights,
    groups: CompressedCoverageGroups,
    selected_indices: list[int] | np.ndarray,
) -> None:
    selected = [int(value) for value in selected_indices]
    mask = np.zeros(len(data.grids), dtype=bool)
    if selected:
        mask = np.asarray(data.coverage[selected].getnnz(axis=0)).ravel() > 0
    direct = {
        "total_population": int(weights.population[mask].sum()),
        "need_weighted": int(weights.need_weighted[mask].sum()),
        "high_need_population": int(weights.high_need_population[mask].sum()),
    }
    compressed = compressed_objective_values(groups, selected)
    if direct != compressed:
        raise AssertionError(
            f"Lossless coverage compression contract failed: direct={direct}, compressed={compressed}"
        )


def _build_compressed_model(
    data: OptimizationInput,
    weights: PreparedWeights,
    groups: CompressedCoverageGroups,
    config: dict[str, Any],
    visit_count: int,
    objective_name: str,
    objective_sense: str,
    floors: dict[str, int],
    scenario: str,
    efficiency_reference: int | None,
    hint_indices: list[int] | None,
):
    from ortools.sat.python import cp_model

    model = cp_model.CpModel()
    n_candidates = len(data.candidates)
    x = [model.NewBoolVar(f"x_{idx}") for idx in range(n_candidates)]
    model.Add(sum(x) == int(visit_count))
    constraints = _candidate_constraint_data(data.candidates, config)

    if constraints["one_per_cluster"] and bool(config["optimization"]["cluster_constraint"]):
        for cluster in sorted(set(constraints["clusters"])):
            indices = np.flatnonzero(constraints["clusters"] == cluster).tolist()
            if len(indices) > 1:
                model.Add(sum(x[index] for index in indices) <= 1)
    for admin in sorted(set(constraints["admins"])):
        indices = np.flatnonzero(constraints["admins"] == admin).tolist()
        model.Add(sum(x[index] for index in indices) <= constraints["max_per_admin"])

    minimum_visits = 0
    if scenario == "equity_hard":
        minimum_visits = int(config["optimization"]["min_visits_per_sigungu_equity_hard"] or 1)
    count_sigungu: dict[str, Any] = {}
    for sigungu in sorted(set(constraints["sigungu"])):
        indices = np.flatnonzero(constraints["sigungu"] == sigungu).tolist()
        count = model.NewIntVar(0, visit_count, f"visit_count_sigungu_{sigungu}")
        model.Add(count == sum(x[index] for index in indices))
        model.Add(count <= constraints["max_per_sigungu"])
        if minimum_visits:
            model.Add(count >= minimum_visits)
        count_sigungu[sigungu] = count

    if bool(config["optimization"]["exclude_known_overlap"]):
        for index in np.flatnonzero(constraints["known_overlap"] == 1):
            model.Add(x[int(index)] == 0)

    y = []
    for group_index, members in enumerate(groups.candidate_members):
        variable = model.NewBoolVar(f"y_{group_index}")
        model.AddMaxEquality(variable, [x[index] for index in members])
        y.append(variable)

    total_population_expr = sum(
        int(groups.population[index]) * y[index] for index in range(groups.group_count)
    )
    need_weighted_expr = sum(
        int(groups.need_weighted[index]) * y[index] for index in range(groups.group_count)
    )
    high_need_expr = sum(
        int(groups.high_need_population[index]) * y[index]
        for index in range(groups.group_count)
    )

    min_coverage = model.NewIntVar(0, 1000, "min_sigungu_coverage_permille")
    for sigungu, total in weights.total_by_sigungu.items():
        relevant = np.flatnonzero(groups.sigungu == sigungu).tolist()
        upper = int(total)
        covered = model.NewIntVar(0, max(0, upper), f"covered_population_{sigungu}")
        if relevant:
            model.Add(
                covered
                == sum(int(groups.population[index]) * y[index] for index in relevant)
            )
        else:
            model.Add(covered == 0)
        if total > 0:
            model.Add(covered * 1000 >= min_coverage * int(total))

    target = int(round(visit_count / max(1, len(count_sigungu))))
    deviations = []
    for sigungu, count in count_sigungu.items():
        difference = model.NewIntVar(-visit_count, visit_count, f"count_diff_{sigungu}")
        deviation = model.NewIntVar(0, visit_count, f"count_dev_{sigungu}")
        model.Add(difference == count - target)
        model.AddAbsEquality(deviation, difference)
        deviations.append(deviation)

    travel_integer = np.rint(constraints["travel_minutes"] * 100).astype(np.int64)
    travel_scale = int(config["optimization"]["travel_penalty_scale"])
    overlap_penalty = int(config["optimization"]["known_overlap_penalty"])
    cost_expr = (
        sum(
            int(travel_integer[index] * travel_scale) * x[index]
            for index in range(n_candidates)
        )
        + sum(
            overlap_penalty * x[int(index)]
            for index in np.flatnonzero(constraints["known_overlap"] == 1)
        )
        + sum(100 * deviation for deviation in deviations)
    )
    expressions = {
        "total_population": total_population_expr,
        "need_weighted": need_weighted_expr,
        "high_need_population": high_need_expr,
        "min_sigungu_coverage": min_coverage,
        "cost": cost_expr,
    }
    for name, floor in floors.items():
        if name not in expressions:
            raise KeyError(f"Unknown objective floor {name}")
        model.Add(expressions[name] >= int(floor))

    if efficiency_reference is not None:
        if scenario == "balanced":
            floor = ceil_fraction(
                efficiency_reference,
                float(config["optimization"]["balanced_min_efficiency_fraction"]),
            )
            model.Add(total_population_expr >= floor)
        elif scenario in {"equity", "equity_hard"}:
            floor = ceil_fraction(
                efficiency_reference,
                float(config["optimization"]["equity_min_efficiency_fraction"]),
            )
            model.Add(total_population_expr >= floor)

    if objective_sense == "max":
        model.Maximize(expressions[objective_name])
    elif objective_sense == "min":
        model.Minimize(expressions[objective_name])
    else:
        raise ValueError(f"Unknown objective sense: {objective_sense}")
    if hint_indices:
        selected = set(int(index) for index in hint_indices)
        for index, variable in enumerate(x):
            model.AddHint(variable, 1 if index in selected else 0)
    return model, x, expressions


def _solve_compressed_stage(
    data: OptimizationInput,
    weights: PreparedWeights,
    groups: CompressedCoverageGroups,
    config: dict[str, Any],
    visit_count: int,
    step: ObjectiveStep,
    floors: dict[str, int],
    scenario: str,
    efficiency_reference: int | None,
    hint_indices: list[int] | None,
    time_limit_sec: float,
    seed: int,
) -> SolveStageResult:
    from ortools.sat.python import cp_model

    model, x, expressions = _build_compressed_model(
        data,
        weights,
        groups,
        config,
        visit_count,
        step.name,
        step.sense,
        floors,
        scenario,
        efficiency_reference,
        hint_indices,
    )
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_sec)
    solver.parameters.num_search_workers = int(config["runtime"]["jobs"])
    solver.parameters.random_seed = int(seed)
    solver.parameters.log_search_progress = bool(config["runtime"]["log_solver_progress"])
    started = time.time()
    status_code = solver.Solve(model)
    elapsed = time.time() - started
    status = solver.StatusName(status_code)
    common_stats = {
        "conflicts": int(solver.NumConflicts()),
        "branches": int(solver.NumBranches()),
        "response_stats": solver.ResponseStats(),
        "compressed_group_count": groups.group_count,
        "covered_source_grid_count": groups.covered_source_grid_count,
        "source_grid_count": groups.source_grid_count,
        "source_nnz": groups.source_nnz,
        "compression_ratio": groups.compression_ratio,
        "seed": int(seed),
    }
    if status_code not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return SolveStageResult(
            objective_name=step.name,
            sense=step.sense,
            status=status,
            objective_value=None,
            best_bound=None,
            wall_time_sec=elapsed,
            selected_indices=[],
            covered_grid_indices=[],
            raw_solver_stats=common_stats,
        )
    selected = [index for index, variable in enumerate(x) if solver.Value(variable) == 1]
    covered = np.asarray(data.coverage[selected].getnnz(axis=0)).ravel() > 0
    assert_compression_exact(data, weights, groups, selected)
    return SolveStageResult(
        objective_name=step.name,
        sense=step.sense,
        status=status,
        objective_value=int(solver.Value(expressions[step.name])),
        best_bound=float(solver.BestObjectiveBound()),
        wall_time_sec=elapsed,
        selected_indices=selected,
        covered_grid_indices=np.flatnonzero(covered).astype(int).tolist(),
        raw_solver_stats=common_stats,
    )


def solve_compressed_scenario(
    data: OptimizationInput,
    config: dict[str, Any],
    scenario: str,
    visit_count: int,
    *,
    efficiency_reference: int | None = None,
    time_limit_sec: float | None = None,
    seed: int | None = None,
    groups: CompressedCoverageGroups | None = None,
) -> ScenarioSolution:
    data.validate()
    weights = prepare_weights(data, config)
    groups = groups or build_compressed_coverage_groups(data, weights)
    plan = _objective_plan(scenario, config)
    floors: dict[str, int] = {}
    stages: list[SolveStageResult] = []
    warm_selected, _ = _greedy_select(data, weights, visit_count, scenario, config)
    hint_indices: list[int] | None = warm_selected
    limit = float(time_limit_sec or config["runtime"]["max_time_per_stage_sec"])
    actual_seed = int(config["seed"] if seed is None else seed)
    final_selected: list[int] = []
    final_covered: list[int] = []
    for step in plan:
        result = _solve_compressed_stage(
            data,
            weights,
            groups,
            config,
            visit_count,
            step,
            floors,
            scenario,
            efficiency_reference,
            hint_indices,
            limit,
            actual_seed,
        )
        stages.append(result)
        if result.objective_value is None:
            return ScenarioSolution(
                scenario=scenario,
                catchment_id=data.catchment_id,
                visit_count=visit_count,
                status=result.status,
                selected=pd.DataFrame(),
                covered_grid_mask=np.zeros(len(data.grids), dtype=bool),
                metrics={},
                stages=stages,
                notes=[f"Compressed CP-SAT failed at {step.name}"],
            )
        final_selected = result.selected_indices
        final_covered = result.covered_grid_indices
        hint_indices = final_selected
        if step.sense == "max" and step.retention is not None:
            floors[step.name] = ceil_fraction(result.objective_value, step.retention)

    covered_mask = np.zeros(len(data.grids), dtype=bool)
    covered_mask[final_covered] = True
    metrics = _metrics_from_selection(data, weights, final_selected, covered_mask, visit_count)
    metrics.update(
        {
            "compressed_group_count": groups.group_count,
            "covered_source_grid_count": groups.covered_source_grid_count,
            "coverage_compression_ratio": groups.compression_ratio,
            "solver_seed": actual_seed,
        }
    )
    status = "OPTIMAL" if all(stage.status == "OPTIMAL" for stage in stages) else "FEASIBLE"
    return ScenarioSolution(
        scenario=scenario,
        catchment_id=data.catchment_id,
        visit_count=visit_count,
        status=status,
        selected=data.candidates.iloc[final_selected].copy(),
        covered_grid_mask=covered_mask,
        metrics=metrics,
        stages=stages,
        notes=["Lossless candidate-incidence+sigungu coverage compression used."],
    )


def solve_compressed_primary_scenarios(
    data: OptimizationInput,
    config: dict[str, Any],
    visit_count: int,
    *,
    time_limit_sec: float | None = None,
    seed: int | None = None,
) -> list[ScenarioSolution]:
    weights = prepare_weights(data, config)
    groups = build_compressed_coverage_groups(data, weights)
    efficiency = solve_compressed_scenario(
        data,
        config,
        "efficiency",
        visit_count,
        time_limit_sec=time_limit_sec,
        seed=seed,
        groups=groups,
    )
    if not efficiency.metrics:
        raise RuntimeError("Compressed efficiency solve did not return a usable incumbent")
    reference = int(efficiency.metrics["unique_elderly_population_scaled"])
    solutions = [efficiency]
    for scenario in config["optimization"]["scenarios"]:
        if scenario == "efficiency":
            continue
        solutions.append(
            solve_compressed_scenario(
                data,
                config,
                str(scenario),
                visit_count,
                efficiency_reference=reference,
                time_limit_sec=time_limit_sec,
                seed=seed,
                groups=groups,
            )
        )
    return solutions


__all__ = [
    "CompressedCoverageGroups",
    "assert_compression_exact",
    "build_compressed_coverage_groups",
    "compressed_objective_values",
    "solve_compressed_primary_scenarios",
    "solve_compressed_scenario",
]
