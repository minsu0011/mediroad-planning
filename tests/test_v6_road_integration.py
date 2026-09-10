"""Fast, read-only regression tests for the materialized V6 road upgrade."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
VALIDATOR_PATH = (
    PACKAGE_ROOT / "12_scripts" / "v6" / "validate_osm_road_integration_v6.py"
)


def _load_validator():
    spec = importlib.util.spec_from_file_location(
        "mediroad_validate_osm_road_integration_v6", VALIDATOR_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import validator: {VALIDATOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


VALIDATOR = _load_validator()


@pytest.fixture(scope="module")
def validation_result():
    # validate_package only reads the PBF hash, report and CSV outputs.  It does
    # not rebuild or mutate the road graph.
    return VALIDATOR.validate_package(PACKAGE_ROOT)


def _checks(result):
    return {item["name"]: item for item in result["checks"]}


def _assert_passed(result, names):
    checks = _checks(result)
    missing = [name for name in names if name not in checks]
    failed = [checks[name] for name in names if name in checks and not checks[name]["ok"]]
    assert not missing, f"validator did not emit checks: {missing}"
    assert not failed, f"road integration checks failed: {failed}"


def test_full_validator_accepts_existing_outputs(validation_result):
    failed = [item for item in validation_result["checks"] if not item["ok"]]
    assert validation_result["ok"], failed
    assert validation_result["summary"]["failed"] == 0


def test_required_artifacts_hashes_and_cardinality(validation_result):
    _assert_passed(
        validation_result,
        (
            "file_report_exists",
            "file_pbf_exists",
            "file_access_exists",
            "file_venue_exists",
            "file_integrated_exists",
            "pbf_sha256",
            "access_rows",
            "access_unique_admin",
            "facility_source_filtered_rows",
            "venue_rows",
            "integrated_rows",
            "integrated_columns",
            "integrated_unique_admin",
        ),
    )


def test_directed_graph_and_reachability_quality_gates(validation_result):
    _assert_passed(
        validation_result,
        (
            "directed_edges_exceed_raw",
            "graph_edge_count_consistent",
            "largest_scc_pct",
            "category_set_20",
            "all_20_categories_reachable_153",
        ),
    )


def test_all_materialized_road_times_are_finite_nonnegative(validation_result):
    _assert_passed(
        validation_result,
        (
            "access_time_columns_20",
            "access_road_times_finite_nonnegative",
            "venue_time_columns_3",
            "venue_road_times_finite_nonnegative",
            "integrated_time_columns_20",
            "integrated_road_times_finite_nonnegative",
            "integrated_road_availability_all_one",
            "access_integrated_values_consistent",
        ),
    )


def test_core_snap_and_canonical_safety_evidence(validation_result):
    _assert_passed(
        validation_result,
        (
            "report_admin_origin_count_and_snap",
            "report_admin_origin_core_resnap_count",
            "report_facility_count_and_snap",
            "report_facility_core_resnap_count",
            "report_venue_count_and_snap",
            "report_venue_core_resnap_count",
            "access_snap_within_limit",
            "venue_snap_within_limit",
            "integrated_snap_within_limit",
            "report_canonical_not_overwritten",
            "canonical_master_relative_path",
            "integrated_preserves_canonical_projection",
            "canonical_hash_evidence",
        ),
    )


def test_canonical_path_contract_is_explicit_about_legacy_evidence(validation_result):
    check = _checks(validation_result)["canonical_master_relative_path"]
    mode = check["actual"]["mode"]
    assert mode in {
        "strict_sha_report",
        "legacy_sha_less_report",
        "legacy_report_with_path",
    }
    if mode == "strict_sha_report":
        assert check["actual"]["value"] == "derived/mediroad_admin_dong_master_v6_final.csv"
        assert not any("LEGACY_EVIDENCE" in warning for warning in validation_result["warnings"])
    elif mode == "legacy_sha_less_report":
        assert any("LEGACY_EVIDENCE" in warning for warning in validation_result["warnings"])
