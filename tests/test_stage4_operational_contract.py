from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from mediroad.stage4_operational_final.operational_contract import (
    OperationalContractError,
    OperationalTables,
    audit_operational_contract,
    certify_operational_contract,
)


GRAPH_SHA = "a" * 64
DATE_1 = "2026-09-01"
DATE_2 = "2026-09-02"


def _tables(
    *,
    capability_team: str = "T1",
    vehicle_team: str = "T1",
    travel_team: str = "T1",
    team_date: str = DATE_1,
    vehicle_date: str = DATE_1,
    venue_date: str = DATE_1,
    conflict: bool = False,
    include_clearance: bool = True,
    capacity: int = 10,
    graph_sha: str = GRAPH_SHA,
) -> OperationalTables:
    bases = pd.DataFrame(
        [
            {
                "team_id": "T1",
                "base_id": "B1",
                "base_name": "Confirmed base",
                "latitude": 36.8,
                "longitude": 127.7,
                "confirmed": True,
                "source_reference": "signed-base-record",
                "effective_date": "2026-08-20",
            },
            {
                "team_id": "T2",
                "base_id": "B2",
                "base_name": "Other confirmed base",
                "latitude": 36.9,
                "longitude": 127.8,
                "confirmed": True,
                "source_reference": "signed-base-record-2",
                "effective_date": "2026-08-20",
            },
        ]
    )
    capability = pd.DataFrame(
        [
            {
                "team_id": team_id,
                "bundle_id": "chronic_primary",
                "capable": team_id == capability_team,
                "confirmed": True,
                "required_vehicle_type": "CLINIC_VAN",
                "minimum_vehicle_capacity": 5,
                "source_reference": f"signed-capability-record-{team_id}",
                "effective_date": "2026-08-20",
            }
            for team_id in ("T1", "T2")
        ]
    )
    vehicles = pd.DataFrame(
        [
            {
                "vehicle_id": "V1",
                "team_id": vehicle_team,
                "base_id": "B1" if vehicle_team == "T1" else "B2",
                "vehicle_type": "CLINIC_VAN",
                "capacity": capacity,
                "operational": True,
                "confirmed": True,
                "source_reference": "fleet-record",
                "effective_date": "2026-08-20",
            }
        ]
    )
    team_calendar = pd.DataFrame(
        [
            {
                "team_id": "T1",
                "date": team_date,
                "available": True,
                "confirmed": True,
                "source_reference": "team-calendar",
            }
        ]
    )
    resource_calendar = pd.DataFrame(
        [
            {
                "resource_type": "VEHICLE",
                "resource_id": "V1",
                "date": vehicle_date,
                "available": True,
                "confirmed": True,
                "source_reference": "fleet-calendar",
            }
        ]
    )
    venue_calendar = pd.DataFrame(
        [
            {
                "venue_id": "VENUE_1",
                "date": venue_date,
                "available": True,
                "confirmed": True,
                "source_reference": "venue-calendar",
            }
        ]
    )
    travel = pd.DataFrame(
        [
            {
                "team_id": travel_team,
                "venue_id": "VENUE_1",
                "reachable": True,
                "distance_km": 25.0,
                "travel_minutes_proxy": 40.0,
                "graph_sha": graph_sha,
                "confirmed": True,
                "source_reference": "routing-run",
                "effective_date": "2026-08-20",
            }
        ]
    )
    conflicts = pd.DataFrame(
        [
            {
                "team_id": "T1",
                "vehicle_id": "V1",
                "venue_id": "VENUE_1",
                "date": DATE_1,
                "conflict": conflict,
                "confirmed": True,
                "source_reference": "complete-existing-service-check",
            }
        ]
        if include_clearance
        else [],
        columns=[
            "team_id",
            "vehicle_id",
            "venue_id",
            "date",
            "conflict",
            "confirmed",
            "source_reference",
        ],
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


def _visits(count: int = 1, *, required_capacity: int = 5) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "visit_id": f"VISIT_{index + 1}",
                "venue_id": "VENUE_1",
                "bundle_id": "chronic_primary",
                "required_vehicle_capacity": required_capacity,
            }
            for index in range(count)
        ]
    )


def test_exact_joint_evidence_certifies_and_returns_assignment():
    audit = certify_operational_contract(
        _tables(),
        _visits(),
        required_visit_count=1,
        expected_graph_sha=GRAPH_SHA,
    )
    assert audit.ready
    assert audit.blockers == []
    assert len(audit.selected_assignments) == 1
    row = audit.selected_assignments.iloc[0]
    assert row["team_id"] == "T1"
    assert row["vehicle_id"] == "V1"
    assert row["venue_id"] == "VENUE_1"
    assert row["date"] == pd.Timestamp(DATE_1)


@pytest.mark.parametrize(
    ("kwargs", "blocker"),
    [
        ({"capability_team": "T2"}, "NO_CONFIRMED_COMPATIBLE_VEHICLE"),
        ({"vehicle_team": "T2"}, "NO_CONFIRMED_COMPATIBLE_VEHICLE"),
        ({"travel_team": "T2"}, "NO_CONFIRMED_REACHABLE_TEAM_VENUE_TRAVEL"),
        ({"vehicle_date": DATE_2}, "NO_COMMON_CONFIRMED_AVAILABILITY_DATE"),
        ({"venue_date": DATE_2}, "NO_COMMON_CONFIRMED_AVAILABILITY_DATE"),
        ({"capacity": 4}, "NO_CONFIRMED_COMPATIBLE_VEHICLE"),
    ],
)
def test_independently_valid_but_non_joining_inputs_fail_closed(kwargs, blocker):
    audit = audit_operational_contract(
        _tables(**kwargs),
        _visits(),
        required_visit_count=1,
        expected_graph_sha=GRAPH_SHA,
    )
    assert not audit.ready
    assert any(value.startswith(blocker) for value in audit.blockers)


def test_absent_conflict_row_does_not_mean_no_conflict():
    audit = audit_operational_contract(
        _tables(include_clearance=False),
        _visits(),
        required_visit_count=1,
        expected_graph_sha=GRAPH_SHA,
    )
    assert not audit.ready
    assert "EMPTY_TABLE:existing_service_conflicts" in audit.blockers


def test_confirmed_existing_service_conflict_blocks_tuple():
    audit = audit_operational_contract(
        _tables(conflict=True),
        _visits(),
        required_visit_count=1,
        expected_graph_sha=GRAPH_SHA,
    )
    assert not audit.ready
    assert any(
        value.startswith("NO_CONFIRMED_EXACT_NO_CONFLICT_CHECK") for value in audit.blockers
    )


def test_unknown_template_values_and_missing_files_fail_closed(tmp_path):
    tables = _tables()
    tables.team_calendar["available"] = ["UNKNOWN"]
    audit = audit_operational_contract(
        tables,
        _visits(),
        required_visit_count=1,
        expected_graph_sha=GRAPH_SHA,
    )
    assert not audit.ready
    assert "INVALID_OR_UNKNOWN_BOOLEAN:team_calendar.available:1" in audit.blockers

    with pytest.raises(OperationalContractError, match="missing"):
        from mediroad.stage4_operational_final.operational_contract import read_csv_table

        read_csv_table(tmp_path / "not-there.csv")


def test_graph_sha_must_match_frozen_provenance():
    audit = audit_operational_contract(
        _tables(graph_sha="b" * 64),
        _visits(),
        required_visit_count=1,
        expected_graph_sha=GRAPH_SHA,
    )
    assert not audit.ready
    assert "TRAVEL_GRAPH_SHA_MISMATCH:1" in audit.blockers


def test_mixed_graph_versions_and_placeholder_evidence_fail_closed():
    tables = _tables()
    extra = tables.travel.iloc[[0]].copy()
    extra["venue_id"] = "VENUE_2"
    extra["graph_sha"] = "b" * 64
    tables = replace(tables, travel=pd.concat([tables.travel, extra], ignore_index=True))
    audit = audit_operational_contract(tables, _visits(), required_visit_count=1)
    assert not audit.ready
    assert "MULTIPLE_TRAVEL_GRAPH_SHAS:2" in audit.blockers

    tables = _tables()
    tables.bases["source_reference"] = ["UNKNOWN", "signed-base-record-2"]
    audit = audit_operational_contract(
        tables,
        _visits(),
        required_visit_count=1,
        expected_graph_sha=GRAPH_SHA,
    )
    assert not audit.ready
    assert "UNKNOWN_REQUIRED_VALUE:bases.source_reference:1" in audit.blockers


def test_confirmed_team_bundle_matrix_must_be_complete():
    tables = _tables()
    tables = replace(
        tables,
        team_bundle_capability=tables.team_bundle_capability.loc[
            tables.team_bundle_capability["team_id"].eq("T1")
        ].copy(),
    )
    audit = audit_operational_contract(
        tables,
        _visits(),
        required_visit_count=1,
        expected_graph_sha=GRAPH_SHA,
    )
    assert not audit.ready
    assert "TEAM_BUNDLE_MATRIX_INCOMPLETE:1" in audit.blockers


def test_individual_options_do_not_prove_two_visit_joint_feasibility():
    # Both visits can independently use T1/V1/VENUE_1 on DATE_1, but those
    # resources cannot be double-booked.  A table-by-table audit would miss it.
    audit = audit_operational_contract(
        _tables(),
        _visits(2),
        required_visit_count=2,
        expected_graph_sha=GRAPH_SHA,
    )
    assert not audit.ready
    assert audit.summary["visits_with_options"] == 2
    assert "NO_CONFLICT_FREE_GLOBAL_ASSIGNMENT" in audit.blockers


def test_unconfirmed_vehicle_cannot_be_borrowed_by_confirmed_team():
    tables = _tables()
    tables.vehicles.loc[0, "confirmed"] = False
    audit = audit_operational_contract(
        tables,
        _visits(),
        required_visit_count=1,
        expected_graph_sha=GRAPH_SHA,
    )
    assert not audit.ready
    assert any(
        value.startswith("NO_CONFIRMED_COMPATIBLE_VEHICLE") for value in audit.blockers
    )


def test_visit_count_and_required_schema_are_hard_gates():
    audit = audit_operational_contract(
        _tables(),
        _visits(),
        required_visit_count=20,
        expected_graph_sha=GRAPH_SHA,
    )
    assert not audit.ready
    assert "VISIT_COUNT_MISMATCH:1:20" in audit.blockers

    bad = _tables().travel.drop(columns=["reachable"])
    tables = replace(_tables(), travel=bad)
    audit = audit_operational_contract(
        tables,
        _visits(),
        required_visit_count=1,
        expected_graph_sha=GRAPH_SHA,
    )
    assert not audit.ready
    assert any(value.startswith("MISSING_COLUMNS:travel:reachable") for value in audit.blockers)
