from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from mediroad.stage4.busproxy_resolution import (
    BusProxyResolutionError,
    actual_facility_fallbacks,
    resolve_final_physical_venues,
)
from mediroad.stage4.field_validation import (
    FIELD_VALIDATION_FIELDS,
    FIELD_VALIDATION_STATE_FIELDS,
    FieldValidationContractError,
    build_field_validation_queue,
    make_field_validation_form,
    robust_core_summary,
    validate_field_validation_schema,
)


def _completed_field_rows(venue_ids: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame({"venue_id": venue_ids})
    for column in FIELD_VALIDATION_STATE_FIELDS:
        frame[column] = "YES"
    frame["verification_date"] = "2026-08-20"
    return frame


def test_twelve_field_schema_is_explicit_and_fail_closed():
    assert len(FIELD_VALIDATION_FIELDS) == 12
    form = make_field_validation_form(
        pd.DataFrame({"venue_id": ["REAL_1"], "venue_name": ["facility"]})
    )
    assert form[list(FIELD_VALIDATION_STATE_FIELDS)].eq("UNKNOWN").all().all()
    assert form.loc[0, "verification_date"] == "UNKNOWN"
    assert not bool(form.loc[0, "field_verified"])

    completed = validate_field_validation_schema(
        _completed_field_rows(["REAL_1"]), as_of_date=date(2026, 8, 20)
    )
    assert bool(completed.loc[0, "field_verified"])

    invalid = _completed_field_rows(["REAL_1"])
    invalid.loc[0, "parking"] = "MAYBE"
    with pytest.raises(FieldValidationContractError, match="YES/NO/UNKNOWN"):
        validate_field_validation_schema(invalid)

    incomplete = _completed_field_rows(["REAL_1"])
    incomplete.loc[0, "toilet"] = "UNKNOWN"
    incomplete["field_verified"] = True
    with pytest.raises(FieldValidationContractError, match="disagrees"):
        validate_field_validation_schema(incomplete)


def test_robust_core_is_explicitly_none_when_no_frequency_reaches_point67():
    evidence = pd.DataFrame(
        {"venue_id": ["A", "B", "C"], "selection_frequency": [0.66, 0.50, 0.10]}
    )
    summary = robust_core_summary(evidence, threshold=0.67)
    assert summary["robust_core_status"] == "NONE"
    assert summary["robust_core_count"] == 0
    assert summary["robust_core_venue_ids"] == []


def test_priority_queue_is_bounded_deterministic_and_carries_all_evidence_axes():
    venue_ids = [f"REAL_{i:02d}" for i in range(36)]
    venue_ids[0] = "BUSPROXY_A"
    candidates = pd.DataFrame(
        {
            "venue_id": venue_ids,
            "venue_name": venue_ids,
            "venue_type": "community_facility",
            "cluster_id": [f"C{i // 3:02d}" for i in range(36)],
            "admin_code": [f"A{i:02d}" for i in range(36)],
            "sigungu": "S",
            "structural_need": np.arange(36, dtype=float),
            "raw_elderly_exposure": np.arange(36, 0, -1, dtype=float),
            "fallback_venue_1": pd.NA,
            "fallback_venue_2": pd.NA,
        }
    )
    # The BUSPROXY's Stage 3 fallbacks are real and in the same cluster.
    candidates.loc[0, "fallback_venue_1"] = "REAL_01"
    candidates.loc[0, "fallback_venue_2"] = "REAL_02"
    selection = pd.DataFrame(
        {
            "venue_id": venue_ids[:20],
            "selection_frequency": np.linspace(0.66, 0.10, 20),
        }
    )
    primary = candidates.iloc[:10][
        ["venue_id", "cluster_id", "fallback_venue_1", "fallback_venue_2"]
    ].copy()
    seed = pd.DataFrame(
        {"venue_id": venue_ids, "seed_selection_frequency": np.linspace(1.0, 0.0, 36)}
    )
    catchment = pd.DataFrame(
        {"venue_id": venue_ids, "catchment_stability": np.linspace(0.0, 1.0, 36)}
    )
    clusters = sorted(candidates["cluster_id"].unique())
    cluster = pd.DataFrame(
        {"cluster_id": clusters, "cluster_selection_frequency": np.linspace(0.2, 0.9, len(clusters))}
    )

    first = build_field_validation_queue(
        selection,
        candidates,
        primary_plan=primary,
        seed_stability=seed,
        catchment_stability=catchment,
        cluster_stability=cluster,
        min_size=30,
        max_size=60,
    )
    second = build_field_validation_queue(
        selection,
        candidates,
        primary_plan=primary,
        seed_stability=seed,
        catchment_stability=catchment,
        cluster_stability=cluster,
        min_size=30,
        max_size=60,
    )
    assert 30 <= len(first) <= 60
    assert first["venue_id"].is_unique
    assert set(primary["venue_id"]).issubset(set(first["venue_id"]))
    assert {"REAL_01", "REAL_02"}.issubset(set(first["venue_id"]))
    assert first["venue_id"].tolist() == second["venue_id"].tolist()
    assert first["robust_core_global_status"].eq("NONE").all()
    for column in (
        "policy_selection_frequency",
        "seed_stability",
        "catchment_stability",
        "cluster_stability",
        "need_relevance",
        "unique_coverage_contribution",
        "busproxy_risk",
        "fallback_risk",
    ):
        assert column in first
    assert first[list(FIELD_VALIDATION_STATE_FIELDS)].eq("UNKNOWN").all().all()


def test_queue_and_resolution_use_the_same_fallback_ranking():
    venue_ids = [f"REAL_{i:02d}" for i in range(30)]
    venue_ids[0] = "BUSPROXY_A"
    candidates = pd.DataFrame(
        {
            "venue_id": venue_ids,
            "cluster_id": ["C1", "C1", "C1", *[f"C{i:02d}" for i in range(3, 30)]],
            "structural_need": np.arange(30, dtype=float),
            "raw_elderly_exposure": np.arange(30, 0, -1, dtype=float),
            "need_weighted_exposure": np.arange(30, dtype=float),
            "venue_readiness_prior": [0.9, 0.6, 0.82, *([0.5] * 27)],
            "fallback_venue_1": pd.NA,
            "fallback_venue_2": pd.NA,
        }
    )
    plan = candidates.iloc[[0]][["venue_id", "cluster_id"]]
    frequency = pd.DataFrame(
        {"venue_id": ["BUSPROXY_A"], "selection_frequency": [0.5]}
    )
    fallback_table = actual_facility_fallbacks(plan, candidates)
    queue = build_field_validation_queue(
        frequency, candidates, primary_plan=plan, min_size=30, max_size=60
    )
    expected = set(
        fallback_table[["actual_facility_fallback_1", "actual_facility_fallback_2"]]
        .stack()
        .astype(str)
    )
    observed = set(queue.loc[queue["is_primary_fallback"], "venue_id"].astype(str))
    assert observed == expected
    assert fallback_table.loc[0, "actual_facility_fallback_1"] == "REAL_02"


def test_queue_preserves_inherited_top3_scope_and_marks_noneligible_rows():
    candidates = pd.DataFrame(
        {
            "venue_id": [f"V{i:02d}" for i in range(30)],
            "cluster_id": [f"C{i:02d}" for i in range(30)],
            "structural_need": np.arange(30, dtype=float),
            "raw_elderly_exposure": np.arange(30, dtype=float) + 1,
        }
    )
    selection = pd.DataFrame(
        {"venue_id": ["V00"], "selection_frequency": [1.0]}
    )
    catchment = pd.DataFrame(
        {
            "venue_id": ["V00", "V01", "V02"],
            "catchment_stability": [1.0, 0.5, 0.0],
            "evidence_scope": "INHERITED_PROVISIONAL_TOP3_DIAGNOSTIC",
            "eligible_candidate_set": "PARENT_TOP3",
            "evidence_universe_complete": True,
            "final_promotion_allowed": "False",
        }
    )
    queue = build_field_validation_queue(
        selection,
        candidates,
        catchment_stability=catchment,
        min_size=30,
        max_size=60,
    )
    assert queue["catchment_stability_evidence_status"].value_counts().to_dict() == {
        "NOT_ELIGIBLE_PARENT_TOP3": 27,
        "MEASURED": 3,
    }
    assert queue["catchment_stability_evidence_scope"].eq(
        "INHERITED_PROVISIONAL_TOP3_DIAGNOSTIC"
    ).all()
    assert not queue["catchment_stability_final_promotion_allowed"].any()


def test_busproxy_is_anchor_only_and_resolves_to_verified_real_same_cluster():
    candidates = pd.DataFrame(
        {
            "venue_id": ["BUSPROXY_A", "REAL_A1", "REAL_A2", "BUSPROXY_B"],
            "venue_name": ["stop", "hall", "clinic", "stop2"],
            "venue_type": ["bus_stop", "village_hall", "clinic", "school"],
            "cluster_id": ["C1", "C1", "C1", "C2"],
            "venue_readiness_prior": [0.9, 0.8, 0.7, 0.6],
            "raw_elderly_exposure": [100, 90, 80, 70],
            "need_weighted_exposure": [60, 50, 40, 30],
            "fallback_venue_1": ["REAL_A1", pd.NA, pd.NA, pd.NA],
            "fallback_venue_2": ["REAL_A2", pd.NA, pd.NA, pd.NA],
        }
    )
    anchors = candidates.loc[[0, 3], ["venue_id", "cluster_id", "fallback_venue_1", "fallback_venue_2"]]
    fallbacks = actual_facility_fallbacks(anchors, candidates)
    assert fallbacks.loc[0, "actual_facility_fallback_1"] == "REAL_A1"
    assert fallbacks.loc[0, "actual_facility_fallback_2"] == "REAL_A2"
    assert fallbacks.loc[1, "fallback_resolution_status"] == "NONE"

    # Even complete positive evidence for BUSPROXY_A cannot make it a physical venue.
    evidence = _completed_field_rows(["BUSPROXY_A", "REAL_A1", "REAL_A2"])
    evidence.loc[evidence["venue_id"].eq("REAL_A2"), "parking"] = "UNKNOWN"
    resolved = resolve_final_physical_venues(anchors, candidates, evidence)
    first = resolved.loc[resolved["venue_id"].eq("BUSPROXY_A")].iloc[0]
    second = resolved.loc[resolved["venue_id"].eq("BUSPROXY_B")].iloc[0]
    assert first["final_physical_venue_id"] == "REAL_A1"
    assert not bool(first["venue_unresolved"])
    assert bool(first["spatial_anchor_is_busproxy"])
    assert bool(second["location_selected"])
    assert bool(second["venue_unresolved"])
    assert pd.isna(second["final_physical_venue_id"])
    assert not resolved["final_physical_venue_id"].dropna().str.startswith("BUSPROXY_").any()


def test_busproxy_contract_rejects_truncated_or_inconsistent_universe():
    candidates = pd.DataFrame(
        {
            "venue_id": ["BUSPROXY_A", "REAL_A1"],
            "cluster_id": ["C1", "C1"],
            "fallback_venue_1": ["MISSING_REAL", pd.NA],
        }
    )
    anchors = candidates.iloc[[0]][["venue_id", "cluster_id", "fallback_venue_1"]]
    with pytest.raises(BusProxyResolutionError, match="absent"):
        actual_facility_fallbacks(anchors, candidates)

    inconsistent = anchors.copy()
    inconsistent["cluster_id"] = "OTHER"
    with pytest.raises(BusProxyResolutionError, match="disagrees"):
        actual_facility_fallbacks(inconsistent, candidates.drop(columns="fallback_venue_1"))


def test_cross_cluster_stage3_fallback_is_rejected_and_replaced_in_cluster():
    candidates = pd.DataFrame(
        {
            "venue_id": ["BUSPROXY_A", "REAL_SAME", "REAL_OTHER"],
            "cluster_id": ["C1", "C1", "C2"],
            "fallback_venue_1": ["REAL_OTHER", pd.NA, pd.NA],
            "raw_elderly_exposure": [10, 9, 100],
        }
    )
    result = actual_facility_fallbacks(
        candidates.iloc[[0]][["venue_id", "cluster_id", "fallback_venue_1"]],
        candidates,
    )
    assert result.loc[0, "actual_facility_fallback_1"] == "REAL_SAME"
    assert result.loc[0, "rejected_fallback_reference_count"] == 1
    assert "OUTSIDE_COVERAGE_CLUSTER" in result.loc[0, "rejected_fallback_references"]


def test_resolver_never_promotes_an_unqueued_third_cluster_candidate():
    candidates = pd.DataFrame(
        {
            "venue_id": ["BUSPROXY_A", "REAL_1", "REAL_2", "REAL_3"],
            "cluster_id": ["C1", "C1", "C1", "C1"],
            "structural_need": [4.0, 3.0, 2.0, 1.0],
            "raw_elderly_exposure": [4.0, 3.0, 2.0, 1.0],
            "venue_readiness_prior": [0.0, 0.9, 0.8, 0.7],
            "fallback_venue_1": ["REAL_1", pd.NA, pd.NA, pd.NA],
            "fallback_venue_2": ["REAL_2", pd.NA, pd.NA, pd.NA],
        }
    )
    anchor = candidates.iloc[[0]][
        ["venue_id", "cluster_id", "fallback_venue_1", "fallback_venue_2"]
    ]
    evidence = pd.DataFrame({"venue_id": candidates["venue_id"]})
    for column in FIELD_VALIDATION_STATE_FIELDS:
        evidence[column] = "NO"
    evidence["verification_date"] = "2026-08-20"
    third = evidence["venue_id"].eq("REAL_3")
    evidence.loc[third, list(FIELD_VALIDATION_STATE_FIELDS)] = "YES"
    result = resolve_final_physical_venues(anchor, candidates, evidence)
    assert bool(result.loc[0, "venue_unresolved"])
    assert pd.isna(result.loc[0, "final_physical_venue_id"])
