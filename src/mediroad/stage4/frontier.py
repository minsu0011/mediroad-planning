from __future__ import annotations

from typing import Any

import pandas as pd

from .metrics import enrich_solution_metrics, solution_row
from .optimizer import solve_scenario
from .types import OptimizationInput, ScenarioSolution


def run_visit_frontier(
    data: OptimizationInput,
    config: dict[str, Any],
    *,
    solver_name: str,
) -> tuple[list[ScenarioSolution], pd.DataFrame]:
    visits_grid = [int(value) for value in config["robustness"]["frontier_visits"]]
    scenarios = list(config["robustness"]["frontier_scenarios"])
    solutions: list[ScenarioSolution] = []
    rows: list[dict[str, Any]] = []
    for visits in visits_grid:
        efficiency = solve_scenario(
            data,
            config,
            "efficiency",
            visits,
            solver_name=solver_name,
            time_limit_sec=float(config["runtime"]["frontier_time_per_stage_sec"]),
        )
        efficiency.metrics = enrich_solution_metrics(efficiency, data)
        solutions.append(efficiency)
        rows.append(solution_row(efficiency))
        reference = int(efficiency.metrics.get("unique_elderly_population_scaled", 0))
        for scenario in scenarios:
            if scenario == "efficiency":
                continue
            solution = solve_scenario(
                data,
                config,
                scenario,
                visits,
                solver_name=solver_name,
                efficiency_reference=reference,
                time_limit_sec=float(config["runtime"]["frontier_time_per_stage_sec"]),
            )
            solution.metrics = enrich_solution_metrics(solution, data)
            solutions.append(solution)
            rows.append(solution_row(solution))
    return solutions, pd.DataFrame(rows)

