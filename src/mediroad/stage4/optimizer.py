from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy import sparse

from .candidates import encode_known_overlap
from .types import OptimizationInput, ScenarioSolution, SolveStageResult
from .utils import ceil_fraction


@dataclass(frozen=True)
class ObjectiveStep:
    name: str
    sense: str  # max or min
    retention: float | None = None


@dataclass
class PreparedWeights:
    population: np.ndarray
    need_weighted: np.ndarray
    high_need_population: np.ndarray
    grid_sigungu: np.ndarray
    total_by_sigungu: dict[str, int]
    population_scale: int
    need_scale: int
    high_need_threshold: float


def prepare_weights(data: OptimizationInput, config: dict[str, Any]) -> PreparedWeights:
    cfg = config["optimization"]
    pop_scale = int(cfg["population_integer_scale"])
    need_scale = int(cfg["need_integer_scale"])
    pop_float = pd.to_numeric(data.grids["elderly65_population"], errors="coerce").fillna(0.0).to_numpy(float)
    need_raw = pd.to_numeric(data.grids["stage1_need_score"], errors="coerce").fillna(0.0).to_numpy(float)
    if np.nanmax(need_raw) > 1.000001:
        need_norm = np.clip(need_raw / 100.0, 0.0, 1.0)
    else:
        need_norm = np.clip(need_raw, 0.0, 1.0)
    population = np.rint(pop_float * pop_scale).astype(np.int64)
    need_weighted = np.rint(pop_float * need_norm * pop_scale * need_scale).astype(np.int64)
    threshold = float(np.quantile(need_norm, float(cfg["high_need_quantile"])))
    high_need = np.rint(pop_float * (need_norm >= threshold) * pop_scale).astype(np.int64)
    grid_sigungu = data.grids["sigungu"].astype(str).to_numpy()
    total_by_sigungu = {
        sigungu: int(population[grid_sigungu == sigungu].sum()) for sigungu in sorted(set(grid_sigungu))
    }
    return PreparedWeights(
        population=population,
        need_weighted=need_weighted,
        high_need_population=high_need,
        grid_sigungu=grid_sigungu,
        total_by_sigungu=total_by_sigungu,
        population_scale=pop_scale,
        need_scale=need_scale,
        high_need_threshold=threshold,
    )


def _objective_plan(scenario: str, config: dict[str, Any]) -> list[ObjectiveStep]:
    retention = config["optimization"]["objective_retention"]
    if scenario == "efficiency":
        return [
            ObjectiveStep("total_population", "max", float(retention["total_population"])),
            ObjectiveStep("need_weighted", "max", float(retention["need_weighted"])),
            ObjectiveStep("min_sigungu_coverage", "max", float(retention["min_sigungu_coverage"])),
            ObjectiveStep("cost", "min", None),
        ]
    if scenario == "balanced":
        return [
            ObjectiveStep("need_weighted", "max", float(retention["need_weighted"])),
            ObjectiveStep("min_sigungu_coverage", "max", float(retention["min_sigungu_coverage"])),
            ObjectiveStep("total_population", "max", float(retention["total_population"])),
            ObjectiveStep("cost", "min", None),
        ]
    if scenario == "equity":
        return [
            ObjectiveStep("min_sigungu_coverage", "max", float(retention["min_sigungu_coverage"])),
            ObjectiveStep("high_need_population", "max", float(retention["high_need"])),
            ObjectiveStep("need_weighted", "max", float(retention["need_weighted"])),
            ObjectiveStep("cost", "min", None),
        ]
    if scenario == "equity_hard":
        return [
            ObjectiveStep("min_sigungu_coverage", "max", 1.0),
            ObjectiveStep("high_need_population", "max", float(retention["high_need"])),
            ObjectiveStep("need_weighted", "max", float(retention["need_weighted"])),
            ObjectiveStep("cost", "min", None),
        ]
    raise ValueError(f"Unknown Stage 4 scenario: {scenario}")


def _cp_sat_available() -> bool:
    try:
        import ortools  # noqa: F401
        return True
    except ImportError:
        return False


def _candidate_constraint_data(candidates: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    cfg = config["candidate_reduction"]
    data: dict[str, Any] = {
        "clusters": candidates["cluster_id"].fillna(candidates["venue_id"]).astype(str).to_numpy(),
        "admins": candidates["admin_code"].astype(str).to_numpy(),
        "sigungu": candidates["sigungu"].astype(str).to_numpy(),
        "max_per_admin": int(cfg["max_visits_per_admin"]),
        "max_per_sigungu": int(cfg["max_visits_per_sigungu"]),
        "one_per_cluster": bool(cfg["one_per_coverage_cluster"]),
    }
    if "known_overlap" in candidates:
        data["known_overlap"] = encode_known_overlap(candidates["known_overlap"]).to_numpy(dtype=int)
    else:
        data["known_overlap"] = np.full(len(candidates), -1, dtype=int)
    travel = pd.to_numeric(candidates.get("travel_minutes", np.nan), errors="coerce")
    if not isinstance(travel, pd.Series):
        travel = pd.Series(np.full(len(candidates), np.nan), index=candidates.index)
    if travel.notna().any():
        fill = float(travel.median())
    else:
        fill = 0.0
    data["travel_minutes"] = travel.fillna(fill).clip(lower=0).to_numpy(dtype=float)
    return data


def _coverage_incidence(coverage: sparse.csr_matrix) -> tuple[sparse.csc_matrix, np.ndarray]:
    binary = coverage.copy().tocsr()
    binary.data = np.ones_like(binary.data, dtype=np.int8)
    binary.eliminate_zeros()
    csc = binary.tocsc()
    covered_columns = np.flatnonzero(np.diff(csc.indptr) > 0)
    return csc, covered_columns


def _build_cp_model(
    data: OptimizationInput,
    weights: PreparedWeights,
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
            idx = np.flatnonzero(constraints["clusters"] == cluster).tolist()
            if len(idx) > 1:
                model.Add(sum(x[i] for i in idx) <= 1)
    for admin in sorted(set(constraints["admins"])):
        idx = np.flatnonzero(constraints["admins"] == admin).tolist()
        model.Add(sum(x[i] for i in idx) <= constraints["max_per_admin"])
    min_sigungu = 0
    if scenario == "equity_hard":
        min_sigungu = int(config["optimization"]["min_visits_per_sigungu_equity_hard"] or 1)
    count_sigungu: dict[str, Any] = {}
    for sigungu in sorted(set(constraints["sigungu"])):
        idx = np.flatnonzero(constraints["sigungu"] == sigungu).tolist()
        count = model.NewIntVar(0, visit_count, f"visit_count_sigungu_{sigungu}")
        model.Add(count == sum(x[i] for i in idx))
        model.Add(count <= constraints["max_per_sigungu"])
        if min_sigungu:
            model.Add(count >= min_sigungu)
        count_sigungu[sigungu] = count

    if bool(config["optimization"]["exclude_known_overlap"]):
        for idx in np.flatnonzero(constraints["known_overlap"] == 1):
            model.Add(x[int(idx)] == 0)

    csc, covered_columns = _coverage_incidence(data.coverage)
    y: dict[int, Any] = {}
    for grid_idx in covered_columns.tolist():
        covering = csc.indices[csc.indptr[grid_idx] : csc.indptr[grid_idx + 1]].tolist()
        var = model.NewBoolVar(f"y_{grid_idx}")
        model.AddMaxEquality(var, [x[i] for i in covering])
        y[grid_idx] = var

    total_population_expr = sum(int(weights.population[g]) * y[g] for g in y)
    need_weighted_expr = sum(int(weights.need_weighted[g]) * y[g] for g in y)
    high_need_expr = sum(int(weights.high_need_population[g]) * y[g] for g in y)

    min_cov = model.NewIntVar(0, 1000, "min_sigungu_coverage_permille")
    covered_by_sigungu: dict[str, Any] = {}
    for sigungu, total in weights.total_by_sigungu.items():
        relevant = [g for g in y if weights.grid_sigungu[g] == sigungu]
        upper = int(total)
        covered = model.NewIntVar(0, max(0, upper), f"covered_pop_{sigungu}")
        if relevant:
            model.Add(covered == sum(int(weights.population[g]) * y[g] for g in relevant))
        else:
            model.Add(covered == 0)
        covered_by_sigungu[sigungu] = covered
        if total > 0:
            model.Add(covered * 1000 >= min_cov * int(total))

    n_sigungu = max(1, len(count_sigungu))
    target = int(round(visit_count / n_sigungu))
    deviations = []
    for sigungu, count in count_sigungu.items():
        diff = model.NewIntVar(-visit_count, visit_count, f"count_diff_{sigungu}")
        dev = model.NewIntVar(0, visit_count, f"count_dev_{sigungu}")
        model.Add(diff == count - target)
        model.AddAbsEquality(dev, diff)
        deviations.append(dev)

    travel_scale = int(config["optimization"]["travel_penalty_scale"])
    travel_int = np.rint(constraints["travel_minutes"] * 100).astype(np.int64)
    overlap_penalty = int(config["optimization"]["known_overlap_penalty"])
    cost_expr = (
        sum(int(travel_int[i] * travel_scale) * x[i] for i in range(n_candidates))
        + sum(int(overlap_penalty) * x[i] for i in np.flatnonzero(constraints["known_overlap"] == 1))
        + sum(100 * dev for dev in deviations)
    )

    expressions = {
        "total_population": total_population_expr,
        "need_weighted": need_weighted_expr,
        "high_need_population": high_need_expr,
        "min_sigungu_coverage": min_cov,
        "cost": cost_expr,
    }

    for name, floor in floors.items():
        if name not in expressions:
            raise KeyError(f"Unknown objective floor {name}")
        model.Add(expressions[name] >= int(floor))

    if efficiency_reference is not None:
        if scenario == "balanced":
            floor = ceil_fraction(efficiency_reference, float(config["optimization"]["balanced_min_efficiency_fraction"]))
            model.Add(total_population_expr >= floor)
        elif scenario in {"equity", "equity_hard"}:
            floor = ceil_fraction(efficiency_reference, float(config["optimization"]["equity_min_efficiency_fraction"]))
            model.Add(total_population_expr >= floor)

    if objective_sense == "max":
        model.Maximize(expressions[objective_name])
    elif objective_sense == "min":
        model.Minimize(expressions[objective_name])
    else:
        raise ValueError(objective_sense)

    if hint_indices:
        selected = set(hint_indices)
        for i, var in enumerate(x):
            model.AddHint(var, 1 if i in selected else 0)

    return model, x, y, expressions, covered_by_sigungu, count_sigungu


def _solve_cp_stage(
    data: OptimizationInput,
    weights: PreparedWeights,
    config: dict[str, Any],
    visit_count: int,
    step: ObjectiveStep,
    floors: dict[str, int],
    scenario: str,
    efficiency_reference: int | None,
    hint_indices: list[int] | None,
    time_limit_sec: float,
) -> SolveStageResult:
    from ortools.sat.python import cp_model

    model, x, y, expressions, _, _ = _build_cp_model(
        data,
        weights,
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
    solver.parameters.random_seed = int(config["seed"])
    solver.parameters.log_search_progress = bool(config["runtime"]["log_solver_progress"])
    started = time.time()
    status_code = solver.Solve(model)
    elapsed = time.time() - started
    status = solver.StatusName(status_code)
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
            raw_solver_stats={"response_stats": solver.ResponseStats()},
        )
    selected = [i for i, var in enumerate(x) if solver.Value(var) == 1]
    covered = [g for g, var in y.items() if solver.Value(var) == 1]
    value = solver.Value(expressions[step.name])
    return SolveStageResult(
        objective_name=step.name,
        sense=step.sense,
        status=status,
        objective_value=int(value),
        best_bound=float(solver.BestObjectiveBound()),
        wall_time_sec=elapsed,
        selected_indices=selected,
        covered_grid_indices=covered,
        raw_solver_stats={
            "conflicts": int(solver.NumConflicts()),
            "branches": int(solver.NumBranches()),
            "response_stats": solver.ResponseStats(),
        },
    )


def _greedy_select(
    data: OptimizationInput,
    weights: PreparedWeights,
    visit_count: int,
    scenario: str,
    config: dict[str, Any],
) -> tuple[list[int], np.ndarray]:
    coverage = data.coverage.tocsr().astype(bool)
    candidates = data.candidates.reset_index(drop=True)
    constraints = _candidate_constraint_data(candidates, config)
    selected: list[int] = []
    covered = np.zeros(len(data.grids), dtype=bool)
    admin_counts: dict[str, int] = {}
    sigungu_counts: dict[str, int] = {}
    clusters: set[str] = set()
    pop = weights.population.astype(float)
    need = weights.need_weighted.astype(float) / max(1, weights.need_scale)
    high = weights.high_need_population.astype(float)

    for _ in range(visit_count):
        best_idx: int | None = None
        best_score = -np.inf
        for i in range(len(candidates)):
            if i in selected:
                continue
            cluster = constraints["clusters"][i]
            admin = constraints["admins"][i]
            sigungu = constraints["sigungu"][i]
            if bool(config["optimization"]["exclude_known_overlap"]) and int(
                constraints["known_overlap"][i]
            ) == 1:
                continue
            if constraints["one_per_cluster"] and cluster in clusters:
                continue
            if admin_counts.get(admin, 0) >= constraints["max_per_admin"]:
                continue
            if sigungu_counts.get(sigungu, 0) >= constraints["max_per_sigungu"]:
                continue
            row = coverage.getrow(i)
            idx = row.indices
            marginal = idx[~covered[idx]]
            if scenario == "efficiency":
                score = pop[marginal].sum()
            elif scenario == "balanced":
                score = 0.45 * pop[marginal].sum() + 0.55 * need[marginal].sum()
            else:
                current_sigungu = sigungu_counts.get(sigungu, 0)
                equity_bonus = 1.0 / (1.0 + current_sigungu)
                score = high[marginal].sum() + 0.5 * need[marginal].sum() + 1000.0 * equity_bonus
            if score > best_score or (math.isclose(score, best_score) and str(candidates.iloc[i]["venue_id"]) < str(candidates.iloc[best_idx]["venue_id"]) if best_idx is not None else False):
                best_score = float(score)
                best_idx = i
        if best_idx is None:
            raise RuntimeError(f"Greedy solver could not choose {visit_count} feasible candidates")
        selected.append(best_idx)
        row = coverage.getrow(best_idx)
        covered[row.indices] = True
        admin = constraints["admins"][best_idx]
        sigungu = constraints["sigungu"][best_idx]
        cluster = constraints["clusters"][best_idx]
        admin_counts[admin] = admin_counts.get(admin, 0) + 1
        sigungu_counts[sigungu] = sigungu_counts.get(sigungu, 0) + 1
        clusters.add(cluster)
    return selected, covered


def _metrics_from_selection(
    data: OptimizationInput,
    weights: PreparedWeights,
    selected_indices: list[int],
    covered_mask: np.ndarray,
    visit_count: int,
) -> dict[str, Any]:
    selected = data.candidates.iloc[selected_indices]
    total = int(weights.population[covered_mask].sum())
    need = int(weights.need_weighted[covered_mask].sum())
    high = int(weights.high_need_population[covered_mask].sum())
    coverage_by_sigungu: dict[str, float] = {}
    for sigungu, denominator in weights.total_by_sigungu.items():
        mask = covered_mask & (weights.grid_sigungu == sigungu)
        numerator = int(weights.population[mask].sum())
        coverage_by_sigungu[sigungu] = numerator / denominator if denominator else 0.0
    visit_counts = selected["sigungu"].astype(str).value_counts().to_dict()
    naive = np.asarray(data.coverage[selected_indices] @ weights.population).ravel().sum()
    return {
        "visit_count": int(visit_count),
        "unique_elderly_population_scaled": total,
        "need_weighted_population_scaled": need,
        "high_need_population_scaled": high,
        "unique_elderly_population": total / weights.population_scale,
        "naive_sum_elderly_population": float(naive / weights.population_scale),
        "redundancy_ratio": float(1.0 - total / naive) if naive > 0 else 0.0,
        "covered_grid_count": int(covered_mask.sum()),
        "min_sigungu_coverage_ratio": float(min(coverage_by_sigungu.values()) if coverage_by_sigungu else 0.0),
        "mean_sigungu_coverage_ratio": float(np.mean(list(coverage_by_sigungu.values())) if coverage_by_sigungu else 0.0),
        "coverage_by_sigungu": coverage_by_sigungu,
        "visit_counts_by_sigungu": visit_counts,
    }


def solve_scenario(
    data: OptimizationInput,
    config: dict[str, Any],
    scenario: str,
    visit_count: int,
    *,
    solver_name: str | None = None,
    efficiency_reference: int | None = None,
    time_limit_sec: float | None = None,
) -> ScenarioSolution:
    data.validate()
    weights = prepare_weights(data, config)
    solver_name = solver_name or str(config["runtime"]["solver"])
    if solver_name == "cp-sat" and not _cp_sat_available():
        raise RuntimeError(
            "OR-Tools is not installed. Install requirements_stage4.txt, then rerun with --solver cp-sat. "
            "The greedy solver is diagnostic only."
        )
    if solver_name == "greedy":
        selected, covered = _greedy_select(data, weights, visit_count, scenario, config)
        metrics = _metrics_from_selection(data, weights, selected, covered, visit_count)
        return ScenarioSolution(
            scenario=scenario,
            catchment_id=data.catchment_id,
            visit_count=visit_count,
            status="GREEDY_DIAGNOSTIC",
            selected=data.candidates.iloc[selected].copy(),
            covered_grid_mask=covered,
            metrics=metrics,
            stages=[],
            notes=["Greedy solver is diagnostic and not the official Stage 4 result."],
        )

    plan = _objective_plan(scenario, config)
    floors: dict[str, int] = {}
    stages: list[SolveStageResult] = []
    # A deterministic greedy incumbent makes short CP-SAT stages productive and
    # avoids UNKNOWN/no-solution starts on the 58k-grid production model.
    warm_selected, _ = _greedy_select(data, weights, visit_count, scenario, config)
    hint_indices: list[int] | None = warm_selected
    time_limit = float(time_limit_sec or config["runtime"]["max_time_per_stage_sec"])
    final_selected: list[int] = []
    final_covered: list[int] = []
    for step in plan:
        result = _solve_cp_stage(
            data,
            weights,
            config,
            visit_count,
            step,
            floors,
            scenario,
            efficiency_reference,
            hint_indices,
            time_limit,
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
                notes=[f"CP-SAT failed at objective stage {step.name}"],
            )
        final_selected = result.selected_indices
        final_covered = result.covered_grid_indices
        hint_indices = final_selected
        if step.sense == "max" and step.retention is not None:
            floors[step.name] = ceil_fraction(result.objective_value, step.retention)

    covered_mask = np.zeros(len(data.grids), dtype=bool)
    covered_mask[final_covered] = True
    metrics = _metrics_from_selection(data, weights, final_selected, covered_mask, visit_count)
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
    )


def solve_primary_scenarios(
    data: OptimizationInput,
    config: dict[str, Any],
    visit_count: int,
    *,
    solver_name: str | None = None,
) -> list[ScenarioSolution]:
    scenarios = list(config["optimization"]["scenarios"])
    solutions: list[ScenarioSolution] = []
    efficiency = solve_scenario(data, config, "efficiency", visit_count, solver_name=solver_name)
    solutions.append(efficiency)
    efficiency_reference = int(efficiency.metrics.get("unique_elderly_population_scaled", 0))
    if efficiency_reference <= 0:
        raise RuntimeError("Efficiency solution did not produce a valid population reference")
    for scenario in scenarios:
        if scenario == "efficiency":
            continue
        solutions.append(
            solve_scenario(
                data,
                config,
                scenario,
                visit_count,
                solver_name=solver_name,
                efficiency_reference=efficiency_reference,
            )
        )
    return solutions
