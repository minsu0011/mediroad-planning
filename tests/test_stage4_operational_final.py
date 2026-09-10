from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from mediroad.stage4_2.contracts import FIELD_STATUS_COLUMNS
from mediroad.stage4_2.field_validation import audit_field_validation
from mediroad.stage4_operational_final.field_evidence import audit_field_evidence
from mediroad.stage4_operational_final.runner import (
    NARROW_IDENTITY_EVIDENCE_RELATIVE_PATH,
    _apply_narrow_identity_evidence,
    _field_form_and_evidence,
    _manual_guide,
    _manual_required,
    _parse_highs_log,
    _snapshot_stage5,
    build_canonical_computational_summary,
    build_field_queue,
    build_selected_union,
    resolve_baseline,
)


ROOT = Path(__file__).resolve().parents[1]


def test_stage5_snapshot_ignores_its_own_audit_but_detects_real_namespace(
    tmp_path: Path,
) -> None:
    own_audit = (
        tmp_path
        / "outputs/model_v1/11_stage4_operational_final/runs/run/00_provenance/stage5_namespace_before.json"
    )
    own_audit.parent.mkdir(parents=True)
    own_audit.write_text("[]", encoding="utf-8")
    assert _snapshot_stage5(tmp_path) == []

    actual = tmp_path / "outputs/model_v1/12_stage5/runs/example.json"
    actual.parent.mkdir(parents=True)
    actual.write_text("{}", encoding="utf-8")
    snapshot = _snapshot_stage5(tmp_path)
    assert any(row["path"].endswith("12_stage5/runs/example.json") for row in snapshot)


@pytest.fixture(scope="module")
def frozen_operational_inputs():
    baseline, baseline_audit = resolve_baseline(ROOT)
    canonical = build_canonical_computational_summary(baseline)
    union, plans, catalog, contributions = build_selected_union(baseline)
    catalog = _apply_narrow_identity_evidence(
        catalog,
        pd.read_csv(ROOT / NARROW_IDENTITY_EVIDENCE_RELATIVE_PATH),
    )
    membership = pd.read_csv(baseline.membership_path, low_memory=False)
    queue, mapping, unresolved, _, field_summary = build_field_queue(
        union,
        catalog,
        membership,
        queue_min=30,
        queue_max=80,
    )
    return {
        "baseline": baseline,
        "baseline_audit": baseline_audit,
        "canonical": canonical,
        "union": union,
        "plans": plans,
        "contributions": contributions,
        "queue": queue,
        "mapping": mapping,
        "unresolved": unresolved,
        "field_summary": field_summary,
    }


def test_highs_log_parser_keeps_direct_gap_separate(tmp_path: Path) -> None:
    log = tmp_path / "direct.highs.log"
    log.write_text(
        "Status            Time limit reached\n"
        "Primal bound      654.5\n"
        "Dual bound        612.0\n"
        "Gap               6.49%\n",
        encoding="utf-8",
    )

    parsed = _parse_highs_log(log)

    assert parsed == {
        "direct_objective_value": 654.5,
        "direct_best_bound": 612.0,
        "direct_relative_gap": pytest.approx(0.0649),
        "direct_status": "Time limit reached",
    }


def test_frozen_parent_and_canonical_rescue_semantics(frozen_operational_inputs) -> None:
    audit = frozen_operational_inputs["baseline_audit"]
    canonical = frozen_operational_inputs["canonical"]

    assert audit["solver_candidate_count"] == 896
    assert audit["compressed_pattern_count"] == 10_838
    assert canonical["stage_count"] == canonical["certified_stage_count"] == 12
    assert all(row["final_certified"] for row in canonical["stages"])
    assert all(row["certified_gap_upper_bound"] <= 0.005 for row in canonical["stages"])

    rescued = [
        row
        for row in canonical["stages"]
        if row["certificate_kind"]
        not in {"OPTIMAL", "DIRECT_MIP_GAP"}
    ]
    assert rescued
    assert any(row["direct_relative_gap"] > 0.005 for row in rescued)
    assert all(row["direct_source_sha256"] for row in canonical["stages"])


def test_coarse_union_and_physical_queue_fail_closed(frozen_operational_inputs) -> None:
    union = frozen_operational_inputs["union"]
    queue = frozen_operational_inputs["queue"]
    mapping = frozen_operational_inputs["mapping"]
    unresolved = frozen_operational_inputs["unresolved"]

    assert len(union) == 38
    assert int(union["is_busproxy_spatial_anchor"].sum()) == 18
    assert set(union["selection_frequency"].value_counts().to_dict().items()) == {
        (1, 21),
        (2, 12),
        (3, 5),
    }
    assert 30 <= len(queue) <= 80
    assert not queue["venue_id"].astype(str).str.startswith("BUSPROXY_").any()

    bus_anchors = set(
        union.loc[union["is_busproxy_spatial_anchor"], "venue_id"].astype(str)
    )
    mapped_bus = set(
        mapping.loc[
            mapping["required_for_busproxy_resolution"], "source_anchor_id"
        ].astype(str)
    )
    assert mapped_bus == bus_anchors
    assert unresolved.empty
    assert mapping.loc[
        mapping["relationship"].eq("BUSPROXY_NEAREST_SAME_ADMIN_MANUAL_CANDIDATE"),
        "coverage_recompute_required",
    ].all()


def test_identity_evidence_never_promotes_operational_fields(frozen_operational_inputs) -> None:
    queue = frozen_operational_inputs["queue"]
    form, evidence = _field_form_and_evidence(queue)

    operational_statuses = [
        column for column in FIELD_STATUS_COLUMNS if column != "actual_facility_exists"
    ]
    assert form[operational_statuses].eq("UNKNOWN").all().all()
    assert form["verification_date"].eq("UNKNOWN").all()
    assert form["field_verified"].eq(False).all()
    evidenced = queue["identity_evidenced"].to_numpy(bool)
    assert evidence.loc[evidenced, "verification_notes"].str.contains(
        "operational attributes", case=False, regex=False
    ).all()
    narrow = queue["narrow_identity_evidence_applied"].to_numpy(bool)
    assert evidence.loc[~evidenced & ~narrow, "verification_notes"].eq("").all()
    assert evidence.loc[narrow, "verification_notes"].str.contains(
        "not an exact parking coordinate", case=False, regex=False
    ).all()

    audited = audit_field_validation(form)
    assert int(audited.normalized["field_verified"].sum()) == 0
    assert int(audited.normalized["field_validation_complete"].sum()) == 0

    proxy_rows = queue.loc[
        queue["coordinate_source"].astype(str).str.contains(
            "proxy|centroid|not_exact", case=False, regex=True
        )
    ]
    assert len(proxy_rows) == 1
    assert not proxy_rows["identity_evidenced"].any()
    assert proxy_rows["coverage_recompute_required"].all()

    maepo = queue.loc[queue["venue_id"].eq("mobile_clinic_MC2025-10_매포읍")]
    assert len(maepo) == 1
    assert maepo.iloc[0]["address"] == "충청북도 단양군 매포읍 평동로 111-11"
    assert maepo.iloc[0]["contact_if_known"] == "043-420-3604"
    assert maepo.iloc[0]["identity_address_evidenced"]
    assert maepo.iloc[0]["facility_exists_evidenced"]
    assert not maepo.iloc[0]["provisional_coordinate_exact_for_release"]
    assert maepo.iloc[0]["provisional_facility_latitude"] == pytest.approx(37.0324)
    assert maepo.iloc[0]["provisional_facility_longitude"] == pytest.approx(128.2992)

    maepo_form = form.loc[form["venue_id"].eq(maepo.iloc[0]["venue_id"])].iloc[0]
    assert "latitude" not in form.columns
    assert "longitude" not in form.columns
    assert maepo_form["catalog_anchor_latitude_not_for_release"] == pytest.approx(
        maepo.iloc[0]["latitude"]
    )
    assert maepo_form["catalog_anchor_longitude_not_for_release"] == pytest.approx(
        maepo.iloc[0]["longitude"]
    )
    assert "provisional_facility_latitude" not in form.columns
    assert "provisional_facility_longitude" not in form.columns
    assert maepo_form["provisional_facility_latitude_not_for_release"] == pytest.approx(
        37.0324
    )
    assert maepo_form["provisional_facility_longitude_not_for_release"] == pytest.approx(
        128.2992
    )
    assert maepo_form["verified_address"] == maepo.iloc[0]["address"]
    assert pd.isna(maepo_form["verified_latitude"])
    assert pd.isna(maepo_form["verified_longitude"])
    assert maepo_form["actual_facility_exists"] == "YES"
    assert not maepo_form["field_verified"]
    manual = _manual_required(queue, form, evidence)
    maepo_manual = manual.loc[
        manual["candidate_actual_facility_id"].eq(maepo.iloc[0]["venue_id"])
    ].iloc[0]
    assert "latitude" not in manual.columns
    assert "longitude" not in manual.columns
    assert maepo_manual["catalog_anchor_latitude_not_for_release"] == pytest.approx(
        maepo.iloc[0]["latitude"]
    )
    assert maepo_manual["catalog_anchor_longitude_not_for_release"] == pytest.approx(
        maepo.iloc[0]["longitude"]
    )
    assert maepo_manual["catalog_anchor_coordinate_source"] == (
        "admin_dong_centroid_proxy_not_exact_venue"
    )
    assert bool(maepo_manual["catalog_anchor_is_proxy_or_centroid"])
    assert int(manual["catalog_anchor_is_proxy_or_centroid"].map(bool).sum()) == 1
    assert "verified_latitude" in maepo_manual["fields_to_confirm"]
    assert "verified_longitude" in maepo_manual["fields_to_confirm"]
    assert "actual_facility_exists" not in maepo_manual["fields_to_confirm"]
    guide = _manual_guide()
    assert "catalog_anchor_latitude_not_for_release" in guide
    assert "내비게이션" in guide
    assert "verified_latitude" in guide


def test_narrow_identity_evidence_never_promotes_secondary_coordinate(
    frozen_operational_inputs,
) -> None:
    _, _, catalog, _ = build_selected_union(frozen_operational_inputs["baseline"])
    evidence = pd.read_csv(ROOT / NARROW_IDENTITY_EVIDENCE_RELATIVE_PATH)
    unsafe = evidence.copy()
    unsafe["provisional_coordinate_exact_for_release"] = True

    with pytest.raises(RuntimeError, match="cannot promote"):
        _apply_narrow_identity_evidence(catalog, unsafe)


def test_field_evidence_is_a_separate_release_gate(frozen_operational_inputs) -> None:
    queue = frozen_operational_inputs["queue"].head(2).copy()
    form, generated_evidence = _field_form_and_evidence(queue)
    for column in FIELD_STATUS_COLUMNS:
        form[column] = "YES"
    form["verification_date"] = "2026-08-22"
    form = audit_field_validation(form).normalized

    generated = audit_field_evidence(
        generated_evidence,
        form,
        expected_venue_ids=set(form["venue_id"]),
    )
    assert generated.summary["historical_field_verified_count"] == 2
    assert generated.summary["operational_release_verified_count"] == 0

    supplied = generated_evidence.copy()
    supplied["verification_source"] = "DIRECT_PHONE_AND_SITE_CHECK"
    supplied["source_reference"] = "field-log-20260822"
    supplied["contact_person_or_office"] = "venue office"
    supplied["contact_channel"] = "recorded phone call"
    supplied["verifier"] = "field-team-lead"
    checked = audit_field_evidence(
        supplied,
        form,
        expected_venue_ids=set(form["venue_id"]),
    )
    assert checked.summary["operational_release_verified_count"] == 2
