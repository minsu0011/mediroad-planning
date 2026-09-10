from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from mediroad.stage4.compressed_optimizer import (
    build_compressed_coverage_groups,
    solve_compressed_scenario,
)
from mediroad.stage4.highs_certifier import (
    HighsCertificationError,
    constrained_greedy_efficiency_floor,
    solve_highs_primary_scenarios,
    solve_highs_scenario,
)
from mediroad.stage4.optimizer import prepare_weights
from mediroad.stage4.types import OptimizationInput


def _config() -> dict:
    return {
        "seed": 42,
        "runtime": {
            "jobs": 1,
            "log_solver_progress": False,
            "max_time_per_stage_sec": 10,
        },
        "candidate_reduction": {
            "one_per_coverage_cluster": True,
            "max_visits_per_admin": 1,
            "max_visits_per_sigungu": 2,
        },
        "optimization": {
            "population_integer_scale": 10,
            "need_integer_scale": 10,
            "high_need_quantile": 0.8,
            "cluster_constraint": True,
            "exclude_known_overlap": False,
            "known_overlap_penalty": 500,
            "travel_penalty_scale": 1,
            "min_visits_per_sigungu_equity_hard": 1,
            "balanced_min_efficiency_fraction": 0.8,
            "equity_min_efficiency_fraction": 0.6,
            "objective_retention": {
                "total_population": 1.0,
                "need_weighted": 1.0,
                "high_need": 1.0,
                "min_sigungu_coverage": 1.0,
            },
            "scenarios": ["efficiency", "balanced", "equity"],
        },
    }


def _data() -> OptimizationInput:
    candidates = pd.DataFrame(
        {
            "venue_id": ["v1", "v2", "v3", "v4"],
            "cluster_id": ["c1", "c2", "c3", "c4"],
            "admin_code": ["a1", "a2", "a3", "a4"],
            "sigungu": ["s1", "s1", "s2", "s2"],
            "known_overlap": ["no", "yes", "no", "no"],
            "travel_minutes": [5.0, 2.0, 1.0, 3.0],
        }
    )
    grids = pd.DataFrame(
        {
            "grid_id": ["g1", "g2", "g3", "g4", "g5", "g6"],
            "elderly65_population": [10.0, 20.0, 5.0, 30.0, 40.0, 10.0],
            "stage1_need_score": [10.0, 90.0, 50.0, 20.0, 100.0, 60.0],
            "sigungu": ["s1", "s1", "s1", "s2", "s2", "s2"],
        }
    )
    coverage = sparse.csr_matrix(
        np.asarray(
            [
                [1, 0, 1, 0, 0, 0],
                [0, 1, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 1],
                [0, 0, 0, 0, 1, 1],
            ],
            dtype=np.int8,
        )
    )
    bundles = pd.DataFrame(
        {
            "venue_id": candidates["venue_id"],
            "bundle_id": ["b", "b", "b", "b"],
            "bundle_value": [1, 1, 1, 1],
        }
    )
    return OptimizationInput(
        candidates=candidates,
        grids=grids,
        coverage=coverage,
        venue_ids=candidates["venue_id"].to_numpy(),
        grid_ids=grids["grid_id"].to_numpy(),
        bundles=bundles,
        catchment_id="synthetic_highs",
    )


@pytest.mark.parametrize("scenario", ["efficiency", "balanced", "equity", "equity_hard"])
def test_highs_matches_independent_compressed_cp_sat(scenario: str) -> None:
    pytest.importorskip("ortools")
    data = _data()
    config = _config()
    efficiency_reference = None
    if scenario != "efficiency":
        efficiency_reference = int(
            solve_highs_scenario(data, config, "efficiency", 2).metrics[
                "unique_elderly_population_scaled"
            ]
        )
    highs = solve_highs_scenario(
        data,
        config,
        scenario,
        2,
        efficiency_reference=efficiency_reference,
    )
    cp_sat = solve_compressed_scenario(
        data,
        config,
        scenario,
        2,
        efficiency_reference=efficiency_reference,
        time_limit_sec=10,
        seed=42,
    )
    assert highs.status == cp_sat.status == "OPTIMAL"
    assert [stage.objective_value for stage in highs.stages] == [
        stage.objective_value for stage in cp_sat.stages
    ]
    for key in [
        "unique_elderly_population_scaled",
        "need_weighted_population_scaled",
        "high_need_population_scaled",
        "min_sigungu_coverage_ratio",
    ]:
        assert highs.metrics[key] == pytest.approx(cp_sat.metrics[key])


def test_highs_records_primal_dual_gap_nodes_status_and_message() -> None:
    solution = solve_highs_scenario(
        _data(), _config(), "efficiency", 2, threads=4, random_seed=42
    )
    assert solution.status == "OPTIMAL"
    for stage in solution.stages:
        stats = stage.raw_solver_stats
        assert stats["backend"] == "scipy.optimize.milp/HiGHS"
        assert stats["scipy_status"] == 0
        assert isinstance(stats["message"], str) and stats["message"]
        assert stats["primal_objective"] == stage.objective_value
        assert stats["dual_bound"] == pytest.approx(stage.best_bound)
        assert stats["absolute_gap"] == pytest.approx(0.0)
        assert stats["relative_gap"] == pytest.approx(0.0)
        assert stats["mip_gap"] == pytest.approx(0.0)
        assert isinstance(stats["mip_node_count"], int)
        assert stats["requested_threads"] == 4
        assert stats["requested_random_seed"] == 42
        # Max objectives were sign-flipped for SciPy minimization and restored.
        if stage.sense == "max":
            assert stage.best_bound == pytest.approx(
                -stats["transformed_minimization_dual_bound"]
            )


def test_highs_primary_scenarios_share_compatible_public_result_types() -> None:
    solutions = solve_highs_primary_scenarios(_data(), _config(), 2)
    assert [solution.scenario for solution in solutions] == [
        "efficiency",
        "balanced",
        "equity",
    ]
    assert all(solution.status == "OPTIMAL" for solution in solutions)
    assert all(len(solution.selected) == 2 for solution in solutions)
    assert all(len(solution.stages) == 4 for solution in solutions)


def test_highs_enforces_known_overlap_exclusion() -> None:
    config = _config()
    config["optimization"]["exclude_known_overlap"] = True
    solution = solve_highs_scenario(_data(), config, "efficiency", 2)
    assert solution.status == "OPTIMAL"
    assert "v2" not in set(solution.selected["venue_id"])


def test_final_efficiency_retains_zero_tolerance_constrained_greedy_floor() -> None:
    data = _data()
    config = _config()
    config["optimization"]["objective_retention"]["total_population"] = 0.5
    floor = constrained_greedy_efficiency_floor(data, config, 2)
    solution = solve_highs_scenario(
        data,
        config,
        "efficiency",
        2,
        minimum_total_population_floor=floor,
    )
    assert solution.metrics["unique_elderly_population_scaled"] >= floor
    assert all(
        int(stage.raw_solver_stats["minimum_total_population_floor"]) == floor
        for stage in solution.stages
    )
    assert all(
        int(stage.raw_solver_stats["applied_objective_floors"]["total_population"])
        >= floor
        for stage in solution.stages
    )
    with pytest.raises(ValueError, match="only valid for efficiency"):
        solve_highs_scenario(
            data,
            config,
            "balanced",
            2,
            efficiency_reference=floor,
            minimum_total_population_floor=floor,
        )


def test_highs_infeasible_problem_returns_no_incumbent() -> None:
    solution = solve_highs_scenario(_data(), _config(), "efficiency", 5)
    assert solution.status == "INFEASIBLE"
    assert solution.selected.empty
    assert solution.metrics == {}
    assert solution.stages[0].objective_value is None


def test_highs_rejects_non_lossless_external_compression() -> None:
    data = _data()
    weights = prepare_weights(data, _config())
    groups = build_compressed_coverage_groups(data, weights)
    tampered = replace(groups, population=groups.population + 1)
    with pytest.raises(HighsCertificationError, match="not the lossless artifact"):
        solve_highs_scenario(data, _config(), "efficiency", 2, groups=tampered)


def test_highs_time_limit_without_incumbent_fails_closed(monkeypatch) -> None:
    from mediroad.stage4 import highs_certifier

    result = SimpleNamespace(
        status=1,
        success=False,
        message="Time limit reached without an incumbent",
        x=None,
        fun=None,
        mip_node_count=0,
        mip_dual_bound=None,
        mip_gap=None,
    )
    monkeypatch.setattr(highs_certifier, "milp", lambda *args, **kwargs: result)
    solution = solve_highs_scenario(_data(), _config(), "efficiency", 2)
    assert solution.status == "UNKNOWN"
    assert solution.metrics == {}
    assert solution.stages[0].raw_solver_stats["message"].startswith("Time limit")


def test_highs_forwards_threads_and_random_seed_and_records_them(monkeypatch) -> None:
    from mediroad.stage4 import highs_certifier

    captured: dict[str, object] = {}

    def fake_milp(*args, **kwargs):
        captured.update(kwargs["options"])
        return SimpleNamespace(
            status=1,
            success=False,
            message="Time limit reached without an incumbent",
            x=None,
            fun=None,
            mip_node_count=0,
            mip_dual_bound=None,
            mip_gap=None,
        )

    monkeypatch.setattr(highs_certifier, "milp", fake_milp)
    solution = solve_highs_scenario(
        _data(),
        _config(),
        "efficiency",
        2,
        threads=4,
        random_seed=42,
    )
    assert captured["threads"] == 4
    assert captured["random_seed"] == 42
    stats = solution.stages[0].raw_solver_stats
    assert stats["requested_threads"] == 4
    assert stats["requested_random_seed"] == 42


def test_highs_nonintegral_candidate_incumbent_fails_closed(monkeypatch) -> None:
    from mediroad.stage4 import highs_certifier

    def fractional(c, **kwargs):
        return SimpleNamespace(
            status=0,
            success=True,
            message="mock invalid incumbent",
            x=np.full(len(c), 0.5),
            fun=0.0,
            mip_node_count=1,
            mip_dual_bound=0.0,
            mip_gap=0.0,
        )

    monkeypatch.setattr(highs_certifier, "milp", fractional)
    solution = solve_highs_scenario(_data(), _config(), "efficiency", 2)
    assert solution.status == "MODEL_INVALID"
    assert solution.metrics == {}
    assert "non-integral" in solution.stages[0].raw_solver_stats[
        "solution_validation_error"
    ]
