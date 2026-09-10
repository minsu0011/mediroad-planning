from __future__ import annotations

from mediroad.stage4.bundle_assignment import assign_bundles
from mediroad.stage4.config import load_config
from mediroad.stage4.optimizer import solve_primary_scenarios
from mediroad.stage4.synthetic import make_synthetic_input


def test_greedy_synthetic_scenarios_select_exact_visits():
    data = make_synthetic_input()
    config = load_config(None)
    config["optimization"]["visit_count"] = 8
    config["candidate_reduction"]["max_visits_per_sigungu"] = 8
    solutions = solve_primary_scenarios(data, config, 8, solver_name="greedy")
    assert {s.scenario for s in solutions} == {"efficiency", "balanced", "equity"}
    for solution in solutions:
        assert len(solution.selected) == 8
        assert solution.selected["venue_id"].is_unique
        assert solution.selected["cluster_id"].is_unique
        assert solution.metrics["unique_elderly_population"] <= solution.metrics["naive_sum_elderly_population"]


def test_bundle_assignment_has_one_bundle_per_visit():
    data = make_synthetic_input()
    config = load_config(None)
    config["optimization"]["visit_count"] = 6
    config["candidate_reduction"]["max_visits_per_sigungu"] = 6
    solution = solve_primary_scenarios(data, config, 6, solver_name="greedy")[0]
    assigned = assign_bundles(solution.selected, data.bundles, config, solver_name="greedy")
    assert len(assigned) == 6
    assert assigned["bundle_id"].notna().all()

