from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from mediroad.stage4.travel_sensitivity import (
    CONFIRMED_TRAVEL_EVIDENCE_STATUS,
    DIAGNOSTIC_TRAVEL_EVIDENCE_STATUS,
    TravelContractError,
    beneficiary_visit_location_equity_kpis,
    build_diagnostic_travel_scenarios,
    load_mobile_team_bases,
    prepare_final_travel_objective,
    validate_diagnostic_travel_scenarios,
    validate_mobile_team_bases,
)


def _confirmed_payload() -> dict:
    return {
        "schema_version": "1.0",
        "evidence_status": "CONFIRMED",
        "source_reference": "signed operations record OPS-2026-08",
        "verification_date": "2026-08-20",
        "teams": [
            {
                "team_id": "TEAM-01",
                "home_base": "verified depot",
                "latitude": 36.64,
                "longitude": 127.49,
                "available_bundles": ["chronic_primary", "neuro_mental"],
                "vehicle_type": "large_mobile_clinic",
                "max_daily_drive_minutes": 240,
                "max_daily_work_minutes": 480,
            }
        ],
    }


def test_missing_team_base_file_disables_and_nulls_final_travel(tmp_path):
    contract = load_mobile_team_bases(tmp_path / "mobile_team_bases.yaml")
    assert contract.evidence_status == "UNCONFIRMED"
    assert not contract.final_travel_objective_enabled
    assert contract.teams.empty

    venues = pd.DataFrame(
        {"venue_id": ["A", "B"], "travel_minutes": [12.0, 34.0]}
    )
    gated = prepare_final_travel_objective(venues, contract)
    assert gated["travel_minutes"].isna().all()
    assert gated["final_travel_minutes"].isna().all()
    assert not gated["final_travel_objective_enabled"].any()
    assert gated["final_travel_input_status"].eq(
        "DISABLED_UNCONFIRMED_TEAM_BASE"
    ).all()


def test_mobile_team_base_schema_requires_real_confirmation_evidence():
    contract = validate_mobile_team_bases(_confirmed_payload())
    assert contract.final_travel_objective_enabled
    assert len(contract.teams) == 1
    assert contract.teams.loc[0, "available_bundles"] == (
        "chronic_primary",
        "neuro_mental",
    )

    no_source = _confirmed_payload()
    no_source.pop("source_reference")
    with pytest.raises(TravelContractError, match="source_reference"):
        validate_mobile_team_bases(no_source)

    bad_hours = _confirmed_payload()
    bad_hours["teams"][0]["max_daily_drive_minutes"] = 600
    with pytest.raises(TravelContractError, match="cannot exceed"):
        validate_mobile_team_bases(bad_hours)


def test_confirmed_network_travel_can_pass_but_diagnostic_cannot():
    contract = validate_mobile_team_bases(_confirmed_payload())
    venues = pd.DataFrame(
        {
            "venue_id": ["A", "B"],
            "travel_minutes": [12.0, 34.0],
            "travel_evidence_status": CONFIRMED_TRAVEL_EVIDENCE_STATUS,
            "diagnostic_only": False,
        }
    )
    gated = prepare_final_travel_objective(venues, contract)
    assert gated["final_travel_minutes"].tolist() == [12.0, 34.0]
    assert gated["final_travel_objective_enabled"].all()

    venues["diagnostic_only"] = True
    venues["travel_evidence_status"] = DIAGNOSTIC_TRAVEL_EVIDENCE_STATUS
    with pytest.raises(TravelContractError, match="diagnostic_only"):
        prepare_final_travel_objective(venues, contract)


def test_three_assumed_base_scenarios_are_permanently_diagnostic():
    venues = pd.DataFrame(
        {
            "venue_id": ["A", "B", "C"],
            "osm_from_cheongju_medical_center_drive_min_v6": [20.0, 50.0, np.nan],
            "osm_from_chungju_medical_center_drive_min_v6": [30.0, 25.0, np.nan],
        }
    )
    scenarios = build_diagnostic_travel_scenarios(venues)
    assert len(scenarios) == 9
    assert set(scenarios["travel_scenario"]) == {
        "cheongju_medical_center",
        "chungju_medical_center",
        "dual_base_nearest",
    }
    dual = scenarios[scenarios["travel_scenario"].eq("dual_base_nearest")].set_index(
        "venue_id"
    )
    assert dual.loc["A", "travel_minutes"] == 20.0
    assert dual.loc["A", "assigned_diagnostic_base"] == "CHEONGJU_MEDICAL_CENTER"
    assert dual.loc["B", "travel_minutes"] == 25.0
    assert dual.loc["B", "assigned_diagnostic_base"] == "CHUNGJU_MEDICAL_CENTER"
    assert math.isnan(dual.loc["C", "travel_minutes"])
    assert scenarios["diagnostic_only"].all()
    assert not scenarios["final_promotion_allowed"].any()

    corrupted = scenarios.copy()
    corrupted.loc[0, "final_promotion_allowed"] = True
    with pytest.raises(TravelContractError, match="non-promotable"):
        validate_diagnostic_travel_scenarios(corrupted)

    missing_scenario = scenarios.drop(index=scenarios.index[0])
    with pytest.raises(TravelContractError, match="all three"):
        validate_diagnostic_travel_scenarios(missing_scenario)


def test_beneficiary_and_visit_location_equity_are_not_conflated():
    visits = pd.DataFrame({"sigungu": ["S1", "S1", "S1", "S2"]})
    coverage = pd.DataFrame(
        {
            "sigungu": ["S1", "S2", "S3"],
            "beneficiary_population": [100.0, 100.0, 100.0],
            "beneficiary_covered_population": [80.0, 20.0, 0.0],
        }
    )
    metrics = beneficiary_visit_location_equity_kpis(visits, coverage)
    assert metrics["visit_location_sigungu_count"] == 2
    assert metrics["visit_location_distribution"] == {"S1": 3, "S2": 1, "S3": 0}
    assert metrics["visit_location_share_distribution"] == {
        "S1": 0.75,
        "S2": 0.25,
        "S3": 0.0,
    }
    assert metrics["visit_location_gini"] == pytest.approx(0.5)
    assert metrics["beneficiary_min_sigungu_coverage"] == 0.0
    assert metrics["beneficiary_mean_sigungu_coverage"] == pytest.approx(1.0 / 3.0)
    assert metrics["beneficiary_coverage_gini"] == pytest.approx(0.5333333333333334)


def test_equity_helper_rejects_impossible_coverage():
    visits = pd.DataFrame({"sigungu": ["S1"]})
    impossible = pd.DataFrame(
        {
            "sigungu": ["S1"],
            "beneficiary_population": [10],
            "beneficiary_covered_population": [11],
        }
    )
    with pytest.raises(TravelContractError, match="cannot exceed"):
        beneficiary_visit_location_equity_kpis(visits, impossible)
