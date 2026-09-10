from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

import mediroad.stage4_2.computational as computational
import mediroad.stage4_2.optimizer as optimizer_module
from mediroad.stage4_2.compression import compress_grid_patterns
from mediroad.stage4_2.errors import SolverCertificationError
from mediroad.stage4_2.optimizer import SpatialMILP
from mediroad.stage4_2.types import SolverStageTelemetry


def _config() -> dict:
    return {
        "solver": {
            "near_optimal_relative_gap_max": 0.005,
            "min_sigungu_ratio_integer_scale": 1_000_000,
            "visit_deviation_exact_integer_formulation": True,
            "known_incumbent_objective_cutoff": False,
        },
        "optimization": {
            "visit_count": 1,
            "objective_retention": {
                "total_population": 0.99,
                "need_weighted": 0.99,
                "high_need": 0.99,
                "min_sigungu_coverage": 1.0,
            },
            "balanced_min_efficiency_fraction": 0.8,
            "equity_min_efficiency_fraction": 0.6,
            "known_overlap_penalty": 1.0,
            "travel_penalty_scale": 1.0,
            "visit_location_deviation_penalty": 1.0,
            "deterministic_tiebreak_scale": 1e-6,
        },
        "candidate_expansion": {
            "max_visits_per_admin": 1,
            "max_visits_per_sigungu": 1,
            "one_per_coverage_cluster": True,
        },
        "bundle_assignment": {"primary_minimum_per_bundle": 1, "time_limit_sec": 5},
    }


def _tiny_problem() -> tuple[pd.DataFrame, sparse.csr_matrix, pd.DataFrame]:
    candidates = pd.DataFrame(
        [{"venue_id": "V1", "admin_code": "A1", "sigungu": "S1", "cluster_id": "C1", "known_overlap": 0}]
    )
    coverage = sparse.csr_matrix(np.ones((1, 1), dtype=np.int8))
    grid = pd.DataFrame(
        {
            "grid_id": ["G1"],
            "sigungu": ["S1"],
            "elderly_population": [1.0],
            "need_weighted_population": [1.0],
            "high_need_population": [1.0],
        }
    )
    return candidates, coverage, grid


def test_subunit_objective_uses_solver_reported_relative_gap(monkeypatch):
    candidates, coverage, grid = _tiny_problem()
    patterns = compress_grid_patterns(coverage, candidates["venue_id"].to_numpy(), grid)
    model = SpatialMILP(candidates, patterns, _config(), candidate_set="synthetic")

    def fake_milp(*args, **kwargs):
        return SimpleNamespace(
            status=1,
            x=np.zeros(model.nvars),
            fun=-0.30,
            mip_dual_bound=-0.39,
            mip_gap=0.30,
            mip_node_count=7,
            message="Time limit reached",
        )

    monkeypatch.setattr(optimizer_module, "milp", fake_milp)
    _, telemetry = model._solve_stage(
        scenario="equity",
        stage_index=1,
        objective_name="min_sigungu_coverage",
        sense="max",
        floors={},
        time_limit=1,
        relative_gap=0.005,
    )
    assert telemetry.absolute_gap == pytest.approx(0.09)
    assert telemetry.relative_gap == pytest.approx(0.30)
    assert telemetry.solver_reported_mip_gap == pytest.approx(0.30)
    assert not telemetry.certified


def test_candidate_no_incumbent_is_recorded_as_inconclusive(monkeypatch, tmp_path):
    candidates, coverage, grid = _tiny_problem()
    patterns = compress_grid_patterns(coverage, candidates["venue_id"].to_numpy(), grid)
    problem = {
        "candidates": candidates,
        "coverage": coverage,
        "grid": grid,
        "patterns": patterns,
        "bundles": pd.DataFrame(),
        "reduction_ledger": pd.DataFrame(),
    }
    monkeypatch.setattr(computational, "_prepare_candidate_problem", lambda *a, **k: problem)
    monkeypatch.setattr(computational, "constrained_greedy", lambda *a, **k: (np.array([0]), 1.0))

    failed = SolverStageTelemetry(
        scenario="efficiency",
        candidate_set="top5",
        stage_index=1,
        objective_name="total_population",
        sense="max",
        solver_status="LIMIT_OR_ITERATION",
        success=False,
        objective_value=None,
        best_bound=None,
        absolute_gap=None,
        relative_gap=None,
        solver_reported_mip_gap=None,
        mip_node_count=0,
        wall_time_sec=1.0,
        message="Time limit reached",
        time_limit_sec=1.0,
        solver_engine="scipy_milp_highs",
        threads_requested=4,
        random_seed=42,
        applied_objective_floors_json="{}",
        certified=False,
    )

    class FakeMILP:
        def __init__(self, *args, **kwargs):
            pass

        def solve(self, *args, **kwargs):
            raise SolverCertificationError("no incumbent", telemetry=[failed])

    monkeypatch.setattr(computational, "SpatialMILP", FakeMILP)
    result = computational.run_candidate_set(
        tmp_path,
        "top5",
        _config(),
        tmp_path / "out",
        scenarios=["efficiency", "balanced"],
        time_limits_by_scenario={"efficiency": 1.0, "balanced": 1.0},
    )
    assert result["results"] == []
    assert result["run_status"] == "COMPLETE_WITH_INCONCLUSIVE_NO_INCUMBENT"
    status = pd.read_csv(tmp_path / "out" / "scenario_run_status.csv")
    assert status["run_status"].tolist() == [
        "NO_INCUMBENT_INCONCLUSIVE",
        "SKIPPED_NO_EFFICIENCY_INCUMBENT",
    ]
    metrics = pd.read_csv(tmp_path / "out" / "scenario_metrics.csv")
    assert list(metrics.columns[:2]) == ["scenario", "candidate_set"]
    assert metrics.empty
