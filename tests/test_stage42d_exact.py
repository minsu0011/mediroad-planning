from __future__ import annotations

from pathlib import Path

import numpy as np

from mediroad.stage4_2_certification.synthetic import make_synthetic_formulation
from mediroad.stage4_2_certification.types import Floor
from mediroad.stage4_2d_equity_certification.exact import (
    EquityFrontStageCertifier,
    _seal_empty_solver_artifact,
    _solve_options,
)
from mediroad.stage4_2d_equity_certification.strengthening import (
    compute_safe_zero_fixings,
    generate_anchored_submodular_cuts,
    metric_targets,
)
from mediroad.stage4_2d_equity_certification.types import EvidenceSeed


def test_exact_certifier_proves_synthetic_min_sigungu(tmp_path: Path):
    f = make_synthetic_formulation()
    selected = np.asarray([0, 3])
    raw = f.metrics(selected)
    seed = EvidenceSeed(
        "seed",
        selected,
        {
            "total_population": raw["unique_elderly_population"],
            "min_sigungu_coverage": raw["min_sigungu_coverage_ratio"],
            "high_need_population": raw["high_need_population"],
            "need_weighted": raw["need_weighted_population"],
            "cost": raw["cost"],
        },
    )
    config = {
        "gate": {"near_optimal_relative_gap_max": 0.005},
        "solver": {
            "random_seed": 42,
            "max_certificate_rounds": 1,
            "local_branching_radii": [1],
            "local_seed_limit": 1,
            "log_to_console": False,
            "time_limit_sec": {"min_sigungu_coverage": {"direct": 10, "local": 2, "oracle": 10}},
            "extra_highs_options": {},
            "phase_options": {},
        },
    }
    targets = metric_targets(f, total_population_floor=10.0, min_sigungu_floor=None, high_need_target=None)
    fixings, _ = compute_safe_zero_fixings(f, targets, use_gpu=False)
    cuts, _ = generate_anchored_submodular_cuts(f, targets, [seed], anchor_sizes=[0, 1, 2], max_cuts=50)
    direct = f.build(
        objective_name="min_sigungu_coverage",
        sense="max",
        floors=[Floor("total_population", "max", 10.0)],
    )

    def oracle(target):
        return f.build(
            objective_name="min_sigungu_coverage",
            sense="max",
            floors=[Floor("total_population", "max", 10.0)],
            threshold=Floor("min_sigungu_coverage", "max", target),
        )

    def cut_builder(target, seeds):
        current = metric_targets(f, total_population_floor=10.0, min_sigungu_floor=target, high_need_target=None)
        fx, fi = compute_safe_zero_fixings(f, current, use_gpu=False)
        cu, ci = generate_anchored_submodular_cuts(f, current, seeds, anchor_sizes=[0, 1, 2], max_cuts=50)
        return cu, fx, {"fixing_diagnostics": fi, **ci}

    result = EquityFrontStageCertifier(f, config, threads=1, output_dir=tmp_path).certify(
        metric="min_sigungu_coverage",
        direct_model=direct,
        oracle_builder=oracle,
        seeds=[seed],
        base_cuts=cuts,
        base_fixings=fixings,
        retained_floor=10.0,
        cut_builder=cut_builder,
    )
    assert result.certified
    assert result.incumbent_value > 0.8


def test_exact_label_does_not_overstate_gap_tolerance_as_exact_optimal(tmp_path: Path):
    from mediroad.stage4_2d_equity_certification.types import SolveEvidence

    f = make_synthetic_formulation()
    config = {
        "gate": {"near_optimal_relative_gap_max": 0.005},
        "solver": {"extra_highs_options": {}, "phase_options": {}},
    }
    certifier = EquityFrontStageCertifier(f, config, threads=1, output_dir=tmp_path)
    evidence = SolveEvidence(
        backend="test",
        status="OPTIMAL",
        has_incumbent=True,
        is_infeasible=False,
        is_optimal=True,
        objective_value=100.0,
        best_bound=100.4,
        relative_gap=0.004,
        wall_time_sec=0.0,
        node_count=0,
        selected_indices=np.asarray([0, 3]),
        message="solver stopped at configured gap",
    )
    result = certifier._direct_certificate(
        evidence,
        metric="min_sigungu_coverage",
        retained_floor=0.0,
    )
    assert result is not None
    assert result.certificate == "DIRECT_MIP_GAP"


def test_empty_optional_solver_stream_gets_explicit_inventory_sentinel(tmp_path: Path):
    path = tmp_path / "no-incumbent.improving.sol"
    path.touch()
    _seal_empty_solver_artifact(path, "NO_IMPROVING_SOLUTION")
    assert path.stat().st_size > 0
    assert path.read_text(encoding="utf-8") == "# NO_IMPROVING_SOLUTION\n"


def test_min_sigungu_certificate_threshold_matches_stage42c_integer_oracle(tmp_path: Path):
    formulation = make_synthetic_formulation()
    certifier = EquityFrontStageCertifier(
        formulation,
        {"gate": {"near_optimal_relative_gap_max": 0.005}, "solver": {}},
        threads=1,
        output_dir=tmp_path,
    )
    assert certifier._certificate_threshold("min_sigungu_coverage", 0.5252179790669655) == 0.527844


def test_min_oracle_replays_verified_stage42c_search_controls(tmp_path: Path):
    config = {
        "gate": {"near_optimal_relative_gap_max": 0.005},
        "solver": {
            "use_objective_target": True,
            "oracle_heuristic_effort": 0.08,
            "mip_max_start_nodes": 30000,
            "extra_highs_options": {"mip_pool_soft_limit": 10000},
            "phase_options": {
                "oracle": {
                    "mip_heuristic_effort": 0.10,
                    "mip_max_start_nodes": 50000,
                }
            },
            "stage42c_oracle_replay": {
                "min_sigungu_coverage": {
                    "enabled": True,
                    "random_seed": 104752,
                    "heuristic_effort": 0.02,
                    "mip_max_start_nodes": 5000,
                    "use_objective_target": False,
                    "include_local_exclusion_cuts": False,
                    "extra_highs_options": {
                        "mip_pool_soft_limit": 20000,
                        "mip_lp_age_limit": 10,
                    },
                }
            },
        },
    }
    options = _solve_options(
        config,
        objective="min_sigungu_coverage",
        phase="oracle",
        time_limit_sec=240,
        threads=8,
        random_seed=1051,
        log_path=tmp_path / "oracle.highs.log",
        target=None,
    )
    assert options.random_seed == 104752
    assert options.heuristic_effort == 0.02
    assert options.mip_max_start_nodes == 5000
    assert options.extra_options["mip_pool_soft_limit"] == 20000
    assert options.extra_options["mip_lp_age_limit"] == 10
    assert "mip_heuristic_effort" not in options.extra_options
    assert "mip_max_start_nodes" not in options.extra_options
    assert "objective_target" not in options.extra_options
