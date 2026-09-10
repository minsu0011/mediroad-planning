from __future__ import annotations

import math
from dataclasses import FrozenInstanceError

import pandas as pd
import pytest

from mediroad.stage4.solver_certification import (
    CertificationClass,
    NearOptimalityContract,
    SolverCertificationError,
    assert_greedy_no_harm,
    certify_solver_run,
    compare_greedy_no_harm,
    near_optimality_contract_from_config,
    normalize_solver_stage,
    normalize_solver_stages,
)
from mediroad.stage4.types import SolveStageResult


def _stage(**overrides):
    row = {
        "stage_order": 1,
        "objective": "total_population",
        "sense": "max",
        "status": "FEASIBLE",
        "objective_value": 1_000.0,
        "best_bound": 1_005.0,
        "wall_time_sec": 90.0,
        "branches": 120,
        "conflicts": 7,
    }
    row.update(overrides)
    return row


def test_predeclared_point_five_percent_contract_is_inclusive():
    stage = normalize_solver_stage(_stage())
    assert stage.absolute_gap == 5.0
    assert math.isclose(stage.relative_gap or -1.0, 0.005)
    assert stage.certification_class is CertificationClass.CERTIFIED_NEAR_OPTIMAL
    assert stage.certified
    assert stage.telemetry_complete


def test_minimization_gap_is_directional():
    stage = normalize_solver_stage(
        _stage(objective="cost", sense="min", objective_value=1_000.0, best_bound=995.0)
    )
    assert stage.absolute_gap == 5.0
    assert stage.relative_gap == 0.005
    assert stage.certification_class is CertificationClass.CERTIFIED_NEAR_OPTIMAL


def test_optimal_requires_bound_to_match_incumbent():
    optimal = normalize_solver_stage(_stage(status="OPTIMAL", best_bound=1_000.0))
    assert optimal.certification_class is CertificationClass.OPTIMAL
    with pytest.raises(SolverCertificationError, match="optimal_status_has_nonzero_gap"):
        normalize_solver_stage(_stage(status="OPTIMAL", best_bound=1_001.0))


def test_feasible_gap_above_contract_is_uncertified():
    stage = normalize_solver_stage(_stage(best_bound=1_006.0))
    assert stage.certification_class is CertificationClass.FEASIBLE_UNCERTIFIED
    assert "relative_gap_above_predeclared_threshold" in stage.certification_reasons


def test_solve_stage_result_reads_branches_and_conflicts_from_raw_stats():
    raw = SolveStageResult(
        objective_name="total_population",
        sense="max",
        status="FEASIBLE",
        objective_value=1000,
        best_bound=1004,
        wall_time_sec=10.0,
        selected_indices=[0],
        covered_grid_indices=[0, 1],
        raw_solver_stats={"branches": 11, "conflicts": 2},
    )
    stage = normalize_solver_stage(raw)
    assert stage.branches == 11
    assert stage.conflicts == 2
    assert stage.certification_class is CertificationClass.CERTIFIED_NEAR_OPTIMAL


def test_missing_legacy_telemetry_fails_closed():
    legacy = {key: value for key, value in _stage().items() if key not in {"branches", "conflicts"}}
    with pytest.raises(SolverCertificationError, match="missing:branches"):
        normalize_solver_stage(legacy)
    audited = normalize_solver_stage(legacy, strict=False)
    assert not audited.telemetry_complete
    assert not audited.certified
    assert audited.certification_class is CertificationClass.FEASIBLE_UNCERTIFIED


def test_contradictory_bound_and_reported_gap_fail_closed():
    with pytest.raises(SolverCertificationError, match="best_bound_below_max_incumbent"):
        normalize_solver_stage(_stage(best_bound=999.0))
    with pytest.raises(SolverCertificationError, match="does_not_match"):
        normalize_solver_stage(_stage(absolute_gap=4.0))


def test_no_incumbent_status_never_certifies():
    stage = normalize_solver_stage(
        _stage(status="UNKNOWN", objective_value=None, best_bound=None), strict=True
    )
    assert not stage.usable_incumbent
    assert stage.certification_class is CertificationClass.FEASIBLE_UNCERTIFIED


def test_dataframe_normalization_has_stable_audit_columns():
    frame = normalize_solver_stages(pd.DataFrame([_stage(), _stage(stage_order=2)]))
    required = {
        "objective_value",
        "best_bound",
        "absolute_gap",
        "relative_gap",
        "solver_status",
        "wall_time_sec",
        "branches",
        "conflicts",
        "certification_class",
    }
    assert required.issubset(frame.columns)
    assert frame["certified"].all()


def test_lexicographic_run_requires_every_stage_to_certify():
    certified = certify_solver_run(
        [_stage(), _stage(stage_order=2, status="OPTIMAL", best_bound=1000)]
    )
    assert certified.certification_class is CertificationClass.CERTIFIED_NEAR_OPTIMAL
    uncertified = certify_solver_run([_stage(), _stage(stage_order=2, best_bound=1200)])
    assert uncertified.certification_class is CertificationClass.FEASIBLE_UNCERTIFIED
    assert math.isclose(uncertified.max_relative_gap or -1.0, 0.2)


def test_config_contract_must_be_explicit_and_is_immutable():
    with pytest.raises(SolverCertificationError, match="predeclare"):
        near_optimality_contract_from_config({"solver": {}})
    contract = near_optimality_contract_from_config(
        {
            "solver": {
                "near_optimal_gap": {
                    "contract_id": "stage4-final-v1",
                    "relative_gap_threshold": 0.0025,
                    "absolute_gap_threshold": 10,
                }
            }
        }
    )
    assert contract.relative_gap_threshold == 0.0025
    assert contract.absolute_gap_threshold == 10
    with pytest.raises(FrozenInstanceError):
        contract.relative_gap_threshold = 0.5  # type: ignore[misc]
    with pytest.raises(SolverCertificationError, match="prespecified"):
        NearOptimalityContract("post-hoc", 0.005, prespecified=False)


def test_greedy_no_harm_is_direction_aware_and_fail_closed_on_comparability():
    maximum = compare_greedy_no_harm(
        cp_sat_objective_value=101,
        greedy_objective_value=100,
        objective_name="total_population",
        sense="max",
        greedy_objective_name="total_population",
        greedy_sense="max",
        same_constraints=True,
    )
    assert maximum.passed and maximum.improvement == 1
    minimum = compare_greedy_no_harm(
        cp_sat_objective_value=99,
        greedy_objective_value=100,
        objective_name="cost",
        sense="min",
        greedy_objective_name="cost",
        greedy_sense="min",
        same_constraints=True,
    )
    assert minimum.passed and minimum.improvement == 1
    incomparable = compare_greedy_no_harm(
        cp_sat_objective_value=200,
        greedy_objective_value=100,
        objective_name="need_weighted",
        sense="max",
        greedy_objective_name="total_population",
        greedy_sense="max",
        same_constraints=True,
    )
    assert not incomparable.comparable
    assert not incomparable.passed


def test_greedy_harm_raises_strong_gate_error():
    harmed = compare_greedy_no_harm(
        cp_sat_objective_value=99,
        greedy_objective_value=100,
        objective_name="total_population",
        sense="max",
        greedy_objective_name="total_population",
        greedy_sense="max",
        same_constraints=True,
    )
    assert harmed.harm_amount == 1
    with pytest.raises(SolverCertificationError, match="worse_than_same_constraint_greedy"):
        assert_greedy_no_harm(harmed)
