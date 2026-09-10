from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from mediroad.stage4_2.types import GridPatternData, ScenarioResult
from mediroad.stage4_operational_final import operational_optimizer
from mediroad.stage4_operational_final.operational_contract import OperationalTables
from mediroad.stage4_operational_final.operational_optimizer import (
    optimize_operational_scenario,
)


GRAPH_SHA = "a" * 64
BUNDLES = [f"BUNDLE_{index}" for index in range(1, 6)]


def _config() -> dict[str, object]:
    return {
        "optimization": {
            "visit_count": 20,
            "objective_retention": {
                "total_population": 1.0,
                "need_weighted": 1.0,
                "high_need": 1.0,
                "min_sigungu_coverage": 1.0,
            },
            "balanced_min_efficiency_fraction": 0.8,
            "equity_min_efficiency_fraction": 0.6,
            "visit_location_deviation_penalty": 0.0,
            "deterministic_tiebreak_scale": 0.0,
        },
        "candidate_constraints": {
            "max_visits_per_admin": 2,
            "max_visits_per_sigungu": 4,
            "one_per_coverage_cluster": True,
        },
        "bundle_assignment": {"primary_minimum_per_bundle": 1},
        "solver": {
            "near_optimal_relative_gap_max": 0.005,
            "min_sigungu_ratio_integer_scale": 1_000_000,
            "highs_threads_per_worker": 1,
            "highs_random_seed": 42,
        },
        "operational_optimizer": {
            "required_bundle_ids": BUNDLES,
            "require_field_verified": True,
            "required_vehicle_capacity": 1,
            "max_no_good_iterations": 4,
            "wall_time_limit_sec": 60.0,
            "spatial_time_limit_per_stage_sec": 5.0,
            "bundle_time_limit_sec": 5.0,
            "assignment_search_state_limit": 100_000,
        },
    }


def _spatial_problem(count: int) -> tuple[pd.DataFrame, GridPatternData]:
    venue_ids = [f"VENUE_{index:02d}" for index in range(count)]
    candidates = pd.DataFrame(
        {
            "venue_id": venue_ids,
            "admin_code": [f"ADMIN_{index:02d}" for index in range(count)],
            "sigungu": [f"SIGUNGU_{index % 6}" for index in range(count)],
            "cluster_id": [f"CLUSTER_{index:02d}" for index in range(count)],
            "field_verified": True,
            "known_overlap": "UNKNOWN",
        }
    )
    # One private grid pattern per venue makes the population objective's
    # ranking exact and makes the 21st candidate the deterministic alternative.
    population = np.arange(count, 0, -1, dtype=float)
    patterns = GridPatternData(
        candidate_ids=np.asarray(venue_ids, dtype=str),
        pattern_coverers=[np.asarray([index], dtype=int) for index in range(count)],
        population=population,
        need_weighted=population.copy(),
        high_need_population=population.copy(),
        sigungu=candidates["sigungu"].to_numpy(dtype=str),
        source_grid_count=np.ones(count, dtype=int),
        source_grid_indices=[np.asarray([index], dtype=int) for index in range(count)],
    )
    return candidates, patterns


def _bundle_values(candidates: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "venue_id": venue_id,
                "bundle_id": bundle_id,
                "bundle_value": float((venue_index + bundle_index) % len(BUNDLES)),
            }
            for venue_index, venue_id in enumerate(candidates["venue_id"].astype(str))
            for bundle_index, bundle_id in enumerate(BUNDLES)
        ]
    )


def _operational_tables(
    candidates: pd.DataFrame,
    *,
    common_date: bool = False,
    omit_venues: set[str] | None = None,
) -> OperationalTables:
    omitted = omit_venues or set()
    venue_ids = [
        value
        for value in candidates["venue_id"].astype(str).tolist()
        if value not in omitted
    ]
    date_by_venue = {
        venue_id: (
            "2026-09-01" if common_date else f"2026-09-{index + 1:02d}"
        )
        for index, venue_id in enumerate(venue_ids)
    }
    dates = sorted(set(date_by_venue.values()))

    bases = pd.DataFrame(
        [
            {
                "team_id": "TEAM_1",
                "base_id": "BASE_1",
                "base_name": "Confirmed synthetic base",
                "latitude": 36.8,
                "longitude": 127.7,
                "confirmed": True,
                "source_reference": "signed-base-evidence",
                "effective_date": "2026-08-20",
            }
        ]
    )
    capability = pd.DataFrame(
        [
            {
                "team_id": "TEAM_1",
                "bundle_id": bundle_id,
                "capable": True,
                "confirmed": True,
                "required_vehicle_type": "CLINIC_VAN",
                "minimum_vehicle_capacity": 1,
                "source_reference": f"signed-capability-{bundle_id}",
                "effective_date": "2026-08-20",
            }
            for bundle_id in BUNDLES
        ]
    )
    vehicles = pd.DataFrame(
        [
            {
                "vehicle_id": "VEHICLE_1",
                "team_id": "TEAM_1",
                "base_id": "BASE_1",
                "vehicle_type": "CLINIC_VAN",
                "capacity": 10,
                "operational": True,
                "confirmed": True,
                "source_reference": "signed-fleet-evidence",
                "effective_date": "2026-08-20",
            }
        ]
    )
    team_calendar = pd.DataFrame(
        [
            {
                "team_id": "TEAM_1",
                "date": date,
                "available": True,
                "confirmed": True,
                "source_reference": f"signed-team-calendar-{date}",
            }
            for date in dates
        ]
    )
    resource_calendar = pd.DataFrame(
        [
            {
                "resource_type": "VEHICLE",
                "resource_id": "VEHICLE_1",
                "date": date,
                "available": True,
                "confirmed": True,
                "source_reference": f"signed-fleet-calendar-{date}",
            }
            for date in dates
        ]
    )
    venue_calendar = pd.DataFrame(
        [
            {
                "venue_id": venue_id,
                "date": date_by_venue[venue_id],
                "available": True,
                "confirmed": True,
                "source_reference": f"signed-venue-calendar-{venue_id}",
            }
            for venue_id in venue_ids
        ]
    )
    travel = pd.DataFrame(
        [
            {
                "team_id": "TEAM_1",
                "venue_id": venue_id,
                "reachable": True,
                "distance_km": float(index + 1),
                "travel_minutes_proxy": float(index + 10),
                "graph_sha": GRAPH_SHA,
                "confirmed": True,
                "source_reference": f"frozen-routing-{venue_id}",
                "effective_date": "2026-08-20",
            }
            for index, venue_id in enumerate(venue_ids)
        ]
    )
    conflicts = pd.DataFrame(
        [
            {
                "team_id": "TEAM_1",
                "vehicle_id": "VEHICLE_1",
                "venue_id": venue_id,
                "date": date_by_venue[venue_id],
                "conflict": False,
                "confirmed": True,
                "source_reference": f"complete-conflict-check-{venue_id}",
            }
            for venue_id in venue_ids
        ]
    )
    return OperationalTables(
        bases=bases,
        team_bundle_capability=capability,
        vehicles=vehicles,
        team_calendar=team_calendar,
        resource_calendar=resource_calendar,
        venue_calendar=venue_calendar,
        travel=travel,
        existing_service_conflicts=conflicts,
    )


def test_joint_contract_blocks_independent_option_false_positive() -> None:
    candidates, patterns = _spatial_problem(20)

    result = optimize_operational_scenario(
        candidates,
        _bundle_values(candidates),
        _operational_tables(candidates, common_date=True),
        _config(),
        scenario="efficiency",
        patterns=patterns,
        expected_graph_sha=GRAPH_SHA,
    )

    # Every visit independently has 5 feasible bundle options, but one
    # team/vehicle on one date cannot jointly serve twenty visits.
    assert not result.ready
    assert result.certified
    assert result.status == "NO_OPERATIONALLY_FEASIBLE_PLAN"
    assert result.no_good_count == 1
    assert result.no_good_ledger[0].proof_kind == "JOINT_ASSIGNMENT_MILP_INFEASIBLE"


def test_exact_no_good_reoptimizes_to_feasible_spatial_alternative() -> None:
    candidates, patterns = _spatial_problem(21)
    tables = _operational_tables(candidates, omit_venues={"VENUE_19"})

    result = optimize_operational_scenario(
        candidates,
        _bundle_values(candidates),
        tables,
        _config(),
        scenario="efficiency",
        patterns=patterns,
        expected_graph_sha=GRAPH_SHA,
    )

    assert result.ready
    assert result.certified
    assert result.status == "OPERATIONALLY_FEASIBLE_CERTIFIED"
    assert result.iteration_count == 2
    assert result.no_good_count == 1
    assert result.no_good_ledger[0].selected_venue_ids == tuple(
        f"VENUE_{index:02d}" for index in range(20)
    )
    assert "VENUE_19" not in result.selected_venue_ids
    assert "VENUE_20" in result.selected_venue_ids
    assert len(result.selected_venue_ids) == 20
    assert set(result.bundle_counts) == set(BUNDLES)
    assert all(value >= 1 for value in result.bundle_counts.values())
    assert len(result.public_assignments) == 20
    assert "date" not in result.public_assignments.columns
    assert "date" not in result.selected_frame.columns
    assert len(result.availability_witness_sha256 or "") == 64
    assert result.gpu_is_proof is False


def test_visit_bundle_and_spatial_certification_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidates, patterns = _spatial_problem(20)
    bundle_values = _bundle_values(candidates)
    tables = _operational_tables(candidates)

    wrong_count = deepcopy(_config())
    wrong_count["optimization"]["visit_count"] = 19  # type: ignore[index]
    count_result = optimize_operational_scenario(
        candidates,
        bundle_values,
        tables,
        wrong_count,
        scenario="efficiency",
        patterns=patterns,
        expected_graph_sha=GRAPH_SHA,
    )
    assert not count_result.ready
    assert count_result.status == "INPUT_CONTRACT_INVALID"

    four_bundles = deepcopy(_config())
    four_bundles["operational_optimizer"]["required_bundle_ids"] = BUNDLES[:4]  # type: ignore[index]
    bundle_result = optimize_operational_scenario(
        candidates,
        bundle_values,
        tables,
        four_bundles,
        scenario="efficiency",
        patterns=patterns,
        expected_graph_sha=GRAPH_SHA,
    )
    assert not bundle_result.ready
    assert bundle_result.status == "INPUT_CONTRACT_INVALID"

    def uncertified_solve(
        model: object, scenario: str, **_: object
    ) -> ScenarioResult:
        selected = np.arange(20, dtype=int)
        frame = candidates.iloc[selected].copy().reset_index(drop=True)
        frame.insert(0, "candidate_set", "operational")
        frame.insert(0, "scenario", scenario)
        return ScenarioResult(
            scenario=scenario,
            candidate_set="operational",
            selected_indices=selected,
            selected_venue_ids=frame["venue_id"].astype(str).tolist(),
            solver_status="LIMIT_OR_ITERATION",
            certification_class="FEASIBLE_UNCERTIFIED",
            certified=False,
            max_relative_gap=0.10,
            telemetry=[],
            metrics={"selected_visit_count": 20},
            selected_frame=frame,
        )

    monkeypatch.setattr(
        operational_optimizer._NoGoodSpatialMILP, "solve", uncertified_solve
    )
    uncertified = optimize_operational_scenario(
        candidates,
        bundle_values,
        tables,
        _config(),
        scenario="efficiency",
        patterns=patterns,
        expected_graph_sha=GRAPH_SHA,
    )
    assert not uncertified.ready
    assert not uncertified.certified
    assert uncertified.status == "SPATIAL_CERTIFICATION_FAILED"
    assert uncertified.no_good_count == 0
