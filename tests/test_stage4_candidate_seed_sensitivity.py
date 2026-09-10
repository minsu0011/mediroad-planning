from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mediroad.stage4.candidate_sensitivity import (
    CandidateBuildPolicy,
    CandidateSensitivityThresholds,
    build_candidate_sets,
    evaluate_candidate_sensitivity,
)
from mediroad.stage4.config import load_config
from mediroad.stage4.seed_stability import (
    DEFAULT_SEEDS,
    SeedStabilityThresholds,
    analyze_seed_stability,
)
from mediroad.stage4.synthetic import make_synthetic_input
from mediroad.stage4.types import ScenarioSolution, SolveStageResult


def _solution(
    seed: int,
    *,
    venue_prefix: str = "V",
    unique: float = 1000.0,
    need: float = 9000.0,
    high: float = 4000.0,
    minimum: float = 0.40,
    bound_fraction: float = 0.0,
    candidate_set: str | None = None,
) -> ScenarioSolution:
    rows = []
    bundles = []
    for index in range(4):
        venue = f"{venue_prefix}{index}"
        rows.append(
            {
                "venue_id": venue,
                "admin_code": f"A{index}",
                "sigungu": f"S{index % 2}",
                "cluster_id": f"C{index}",
            }
        )
        bundles.append({"venue_id": venue, "bundle_id": f"B{index % 2}"})
    selected = pd.DataFrame(rows)
    objective = unique
    stage = SolveStageResult(
        objective_name="total_population",
        sense="max",
        status="OPTIMAL" if bound_fraction == 0 else "FEASIBLE",
        objective_value=objective,
        best_bound=objective * (1.0 + bound_fraction),
        wall_time_sec=1.0,
        selected_indices=list(range(4)),
        covered_grid_indices=list(range(4)),
    )
    return ScenarioSolution(
        scenario="efficiency",
        catchment_id="hard_5000m",
        visit_count=4,
        status=stage.status,
        selected=selected,
        covered_grid_mask=np.ones(4, dtype=bool),
        metrics={
            "unique_elderly_population": unique,
            "need_weighted_population_scaled": need,
            "high_need_population_scaled": high,
            "min_sigungu_coverage_ratio": minimum,
            "candidate_set": candidate_set,
            "seed": seed,
        },
        stages=[stage],
        bundle_assignments=pd.DataFrame(bundles),
    )


def test_build_candidate_sets_are_aligned_and_preflighted():
    data = make_synthetic_input(n_admin=8, venues_per_admin=6, grids_per_admin=5)
    config = load_config(None)
    config["optimization"]["visit_count"] = 4
    config["candidate_reduction"]["max_visits_per_sigungu"] = 4
    build = build_candidate_sets(
        data,
        config,
        policy=CandidateBuildPolicy(max_candidates=40),
        visit_count=4,
    )
    assert set(build.inputs) == {"top3", "top5", "coarse_pareto", "coverage_cluster"}
    assert build.preflight["preflight_passed"].all()
    assert len(build.inputs["top3"].candidates) == 24
    assert len(build.inputs["top5"].candidates) == 40
    assert set(build.inputs["top3"].candidates["venue_id"]).issubset(
        set(build.inputs["coarse_pareto"].candidates["venue_id"])
    )
    for set_id, subset in build.inputs.items():
        assert np.array_equal(subset.venue_ids, subset.candidates["venue_id"].to_numpy(str))
        assert subset.coverage.shape == (len(subset.candidates), len(subset.grids))
        assert subset.candidates["admin_code"].nunique() == 8
        if set_id == "coverage_cluster":
            assert subset.candidates["cluster_id"].is_unique


def test_candidate_build_fails_closed_when_visit_capacity_is_too_small():
    data = make_synthetic_input(n_admin=2, venues_per_admin=2, grids_per_admin=5)
    config = load_config(None)
    config["optimization"]["visit_count"] = 5
    config["candidate_reduction"]["max_visits_per_admin"] = 1
    config["candidate_reduction"]["max_visits_per_sigungu"] = 1
    with pytest.raises(RuntimeError, match="preflight failed closed"):
        build_candidate_sets(data, config, strategies=("top3",), visit_count=5)


def test_candidate_source_alignment_fails_closed():
    data = make_synthetic_input()
    data.venue_ids = data.venue_ids[::-1].copy()
    with pytest.raises(ValueError, match="not aligned"):
        build_candidate_sets(data, load_config(None), strategies=("top3",), visit_count=4)


def test_optional_cluster_representative_reports_unavailable_without_silent_relaxation():
    data = make_synthetic_input(n_admin=2, venues_per_admin=2, grids_per_admin=5)
    data.candidates["cluster_id"] = "SHARED"
    config = load_config(None)
    config["optimization"]["visit_count"] = 1
    build = build_candidate_sets(
        data,
        config,
        strategies=("coverage_cluster",),
        visit_count=1,
    )
    assert not build.inputs
    assert build.membership.empty
    assert build.unavailable["candidate_set"].tolist() == ["coverage_cluster"]
    assert build.preflight.loc[0, "available"] == False  # noqa: E712
    assert build.preflight.loc[0, "preflight_passed"] == False  # noqa: E712


def test_top3_promotion_uses_predeclared_thresholds():
    solutions = {
        "top3": _solution(42, unique=995.0, need=8950.0, high=3960.0, minimum=0.395),
        "top5": _solution(42, unique=1000.0, need=9000.0, high=4000.0, minimum=0.400),
        "coarse_pareto": _solution(42, unique=1000.0, need=9000.0, high=4000.0, minimum=0.400),
    }
    result = evaluate_candidate_sensitivity(solutions)
    assert result.passed
    assert result.decision == "PASS_RETAIN_TOP3"
    assert result.recommended_candidate_set == "top3"
    assert result.comparisons["comparison_passed"].all()
    assert len(result.threshold_fingerprint) == 64


def test_candidate_sensitivity_is_inconclusive_when_solver_gap_is_large():
    solutions = {
        "top3": _solution(42, bound_fraction=0.20),
        "top5": _solution(42),
        "coarse_pareto": _solution(42),
    }
    result = evaluate_candidate_sensitivity(
        solutions,
        thresholds=CandidateSensitivityThresholds(max_solver_relative_gap=0.05),
    )
    assert not result.passed
    assert result.decision == "INCONCLUSIVE_CANDIDATE_SENSITIVITY_SOLVER_GAP"
    assert result.recommended_candidate_set is None


def test_candidate_sensitivity_uses_declared_fallback_not_posthoc_best():
    solutions = {
        "top3": _solution(42, unique=900.0),
        "top5": _solution(42, unique=1000.0),
        "coarse_pareto": _solution(42, unique=1100.0),
    }
    result = evaluate_candidate_sensitivity(solutions)
    assert result.passed
    assert result.decision == "PROMOTE_PREDECLARED_WIDER_CANDIDATE_SET"
    assert result.recommended_candidate_set == "top5"


def test_seed_contract_requires_exact_five_seeds():
    solutions = {seed: _solution(seed) for seed in DEFAULT_SEEDS[:-1]}
    with pytest.raises(ValueError, match="Seed set must match exactly"):
        analyze_seed_stability(solutions)


def test_seed_stability_distinguishes_cluster_from_exact_venue():
    solutions = {
        seed: _solution(seed, venue_prefix=f"SEED{seed}_")
        for seed in DEFAULT_SEEDS
    }
    result = analyze_seed_stability(solutions)
    assert result.passed
    assert result.classification == "PASS_CLUSTER_STABLE_VENUE_VARIABLE"
    assert result.pairwise["venue_jaccard"].eq(0.0).all()
    assert result.pairwise["coverage_cluster_jaccard"].eq(1.0).all()
    assert result.pairwise["bundle_jaccard"].eq(1.0).all()
    assert result.pairwise["cluster_bundle_jaccard"].eq(1.0).all()
    assert result.pairwise["venue_bundle_jaccard"].eq(0.0).all()
    assert set(result.selection_frequency["entity_type"]) == {
        "venue",
        "admin",
        "coverage_cluster",
        "bundle",
        "cluster_bundle",
        "venue_bundle",
    }


def test_seed_stability_exact_class_requires_stable_venue_bundle_allocation():
    solutions = {seed: _solution(seed) for seed in DEFAULT_SEEDS}
    result = analyze_seed_stability(solutions)
    assert result.passed
    assert result.classification == "PASS_SEED_STABLE"
    assert result.pairwise["cluster_bundle_jaccard"].eq(1.0).all()
    assert result.pairwise["venue_bundle_jaccard"].eq(1.0).all()


def test_seed_stability_does_not_gate_on_tautological_bundle_label_set():
    solutions = {seed: _solution(seed) for seed in DEFAULT_SEEDS}
    for seed in (42, 77, 101):
        assignment = solutions[seed].bundle_assignments.copy()
        assignment["bundle_id"] = assignment["bundle_id"].map({"B0": "B1", "B1": "B0"})
        solutions[seed].bundle_assignments = assignment

    result = analyze_seed_stability(solutions)
    assert not result.passed
    assert result.classification == "REVIEW_SEED_SELECTION_UNSTABLE"
    assert result.pairwise["bundle_jaccard"].eq(1.0).all()
    assert result.pairwise["cluster_bundle_jaccard"].median() == 0.0
    assert result.pairwise["venue_bundle_jaccard"].median() == 0.0

    # Preserve the existing public threshold name while applying it to actual
    # allocation pairs rather than to the fixed set of bundle labels.
    relaxed = analyze_seed_stability(
        solutions,
        thresholds=SeedStabilityThresholds(min_median_bundle_jaccard=0.0),
    )
    assert relaxed.passed
    assert relaxed.classification == "PASS_SEED_STABLE"
    assert "min_median_bundle_jaccard" in relaxed.thresholds


def test_seed_objective_spread_fails_promotion():
    solutions = {seed: _solution(seed) for seed in DEFAULT_SEEDS}
    solutions[101] = _solution(101, unique=800.0)
    result = analyze_seed_stability(solutions)
    assert not result.passed
    assert result.classification == "FAIL_SEED_OBJECTIVE_UNSTABLE"


def test_seed_stability_requires_complete_bundle_assignment():
    solutions = {seed: _solution(seed) for seed in DEFAULT_SEEDS}
    solutions[42].bundle_assignments = None
    with pytest.raises(ValueError, match="requires bundle assignments"):
        analyze_seed_stability(solutions)
