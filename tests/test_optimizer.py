import numpy as np
import pandas as pd
from scipy import sparse

from mediroad.stage4_2.compression import compress_grid_patterns
from mediroad.stage4_2.optimizer import SpatialMILP, constrained_greedy


def config():
    return {
        "optimization": {
            "visit_count": 2,
            "objective_retention": {"total_population": 0.99, "need_weighted": 0.99, "high_need": 0.99, "min_sigungu_coverage": 1.0},
            "balanced_min_efficiency_fraction": 0.8,
            "equity_min_efficiency_fraction": 0.6,
            "known_overlap_penalty": 1000,
            "travel_penalty_scale": 1,
            "visit_location_deviation_penalty": 1,
            "deterministic_tiebreak_scale": 1e-6,
        },
        "candidate_expansion": {"max_visits_per_admin": 1, "max_visits_per_sigungu": 2, "one_per_coverage_cluster": True},
        "solver": {
            "min_sigungu_ratio_integer_scale": 1_000_000,
            "visit_deviation_exact_integer_formulation": True,
            "known_incumbent_objective_cutoff": False,
        },
    }


def problem():
    candidates = pd.DataFrame(
        [
            {"venue_id": "V1", "admin_code": "A1", "sigungu": "S1", "cluster_id": "C1", "known_overlap": 0},
            {"venue_id": "V2", "admin_code": "A2", "sigungu": "S1", "cluster_id": "C2", "known_overlap": 0},
            {"venue_id": "V3", "admin_code": "A3", "sigungu": "S2", "cluster_id": "C3", "known_overlap": 0},
            {"venue_id": "V4", "admin_code": "A4", "sigungu": "S2", "cluster_id": "C4", "known_overlap": 0},
        ]
    )
    coverage = sparse.csr_matrix(
        np.array(
            [
                [1, 1, 0, 0, 0, 0],
                [0, 1, 1, 0, 0, 0],
                [0, 0, 0, 1, 1, 0],
                [0, 0, 0, 0, 1, 1],
            ], dtype=np.int8
        )
    )
    grid = pd.DataFrame(
        {
            "grid_id": [f"G{i}" for i in range(6)],
            "sigungu": ["S1", "S1", "S1", "S2", "S2", "S2"],
            "elderly_population": [10, 20, 30, 40, 50, 60],
            "need_weighted_population": [9, 18, 27, 4, 5, 6],
            "high_need_population": [10, 20, 30, 0, 0, 0],
        }
    )
    return candidates, coverage, grid


def test_efficiency_and_balanced_solve_exact_visits():
    candidates, coverage, grid = problem()
    patterns = compress_grid_patterns(coverage, candidates["venue_id"].to_numpy(), grid)
    model = SpatialMILP(candidates, patterns, config(), candidate_set="synthetic")
    greedy_idx, greedy_pop = constrained_greedy(candidates, coverage, grid["elderly_population"].to_numpy(), config())
    eff = model.solve("efficiency", efficiency_reference=None, greedy_floor=greedy_pop, time_limit_per_stage=10, gap_threshold=0.005)
    bal = model.solve("balanced", efficiency_reference=eff.metrics["unique_elderly_population"], greedy_floor=None, time_limit_per_stage=10, gap_threshold=0.005)
    assert len(eff.selected_indices) == 2
    assert len(bal.selected_indices) == 2
    assert eff.metrics["unique_elderly_population"] >= greedy_pop - 1e-6
    assert eff.certified
    assert bal.certified


def test_discrete_ratio_and_deviation_variables_are_integral():
    candidates, coverage, grid = problem()
    patterns = compress_grid_patterns(coverage, candidates["venue_id"].to_numpy(), grid)
    model = SpatialMILP(candidates, patterns, config(), candidate_set="synthetic")
    assert model._integrality[model.z_index] == 1
    assert np.all(model._integrality[model.dev_slice] == 1)
    assert model._bounds.ub[model.z_index] == 1_000_000
    assert model._objective_vectors["min_sigungu_coverage"][model.z_index] == 1e-6
