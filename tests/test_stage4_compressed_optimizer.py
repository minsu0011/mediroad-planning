from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from mediroad.stage4.compressed_optimizer import (
    assert_compression_exact,
    build_compressed_coverage_groups,
    compressed_objective_values,
    solve_compressed_scenario,
)
from mediroad.stage4.optimizer import prepare_weights
from mediroad.stage4.types import OptimizationInput


def _config() -> dict:
    return {
        "seed": 42,
        "runtime": {"jobs": 1, "log_solver_progress": False, "max_time_per_stage_sec": 5},
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
            "known_overlap_penalty": 0,
            "travel_penalty_scale": 0,
            "min_visits_per_sigungu_equity_hard": 0,
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
            "venue_id": ["v1", "v2", "v3"],
            "cluster_id": ["c1", "c2", "c3"],
            "admin_code": ["a1", "a2", "a3"],
            "sigungu": ["s1", "s1", "s2"],
            "known_overlap": ["no", "no", "no"],
            "travel_minutes": [np.nan, np.nan, np.nan],
        }
    )
    grids = pd.DataFrame(
        {
            "grid_id": ["g1", "g2", "g3", "g4", "g5"],
            "elderly65_population": [10.0, 20.0, 30.0, 40.0, 50.0],
            "stage1_need_score": [10.0, 20.0, 30.0, 90.0, 100.0],
            "sigungu": ["s1", "s1", "s1", "s2", "s2"],
        }
    )
    # g1/g2 have the same incidence and sigungu and must aggregate exactly.
    matrix = sparse.csr_matrix(
        np.asarray(
            [
                [1, 1, 1, 0, 0],
                [1, 1, 0, 1, 0],
                [0, 0, 0, 1, 1],
            ],
            dtype=np.int8,
        )
    )
    bundles = pd.DataFrame(
        {"venue_id": ["v1", "v2", "v3"], "bundle_id": ["b", "b", "b"], "bundle_value": [1, 1, 1]}
    )
    return OptimizationInput(
        candidates=candidates,
        grids=grids,
        coverage=matrix,
        venue_ids=candidates["venue_id"].to_numpy(),
        grid_ids=grids["grid_id"].to_numpy(),
        bundles=bundles,
        catchment_id="synthetic",
    )


def test_compression_is_exact_for_multiple_selections() -> None:
    data = _data()
    weights = prepare_weights(data, _config())
    groups = build_compressed_coverage_groups(data, weights)
    assert groups.source_grid_count == 5
    assert groups.group_count == 4
    for selected in ([0], [1], [2], [0, 2], [1, 2]):
        assert_compression_exact(data, weights, groups, selected)


def test_compressed_values_preserve_sigungu_specific_incidence() -> None:
    data = _data()
    weights = prepare_weights(data, _config())
    groups = build_compressed_coverage_groups(data, weights)
    values = compressed_objective_values(groups, [2])
    assert values["total_population"] == 900


def test_compressed_solver_returns_exact_optimum() -> None:
    pytest.importorskip("ortools")
    solution = solve_compressed_scenario(
        _data(), _config(), "efficiency", 2, time_limit_sec=5, seed=42
    )
    assert solution.status == "OPTIMAL"
    assert solution.metrics["unique_elderly_population"] == pytest.approx(150.0)
    assert set(solution.selected["venue_id"]) == {"v1", "v3"}
    assert all(stage.best_bound == pytest.approx(stage.objective_value) for stage in solution.stages)

