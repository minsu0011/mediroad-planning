from __future__ import annotations

import hashlib
import json

import pandas as pd
import pytest

from mediroad.stage4_operational_final.operational_contract import (
    OperationalContractError,
    OperationalTables,
    TABLE_SCHEMAS,
    audit_operational_contract,
)
from mediroad.stage4_operational_final.operational_io import (
    DEFAULT_BUNDLE_IDS,
    FROZEN_GRAPH_SOURCE_FILENAME,
    INPUT_MANIFEST_FILENAME,
    OPERATIONAL_INPUT_FILENAMES,
    TEMPLATE_GUIDE_FILENAME,
    assert_operational_inputs_unchanged,
    capture_operational_input_snapshot,
    compare_operational_input_snapshots,
    load_operational_input_dir,
    load_operational_tables,
    verify_operational_input_manifest,
    write_operational_input_manifest,
    write_operational_template_pack,
)


def _pbf(tmp_path, content: bytes = b"synthetic-pbf-for-io-contract"):
    path = tmp_path / "source.osm.pbf"
    path.write_bytes(content)
    return path


def _visits() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "visit_id": "VISIT_1",
                "venue_id": "OFFICIAL_FACILITY_1",
                "bundle_id": "chronic_primary",
                "required_vehicle_capacity": 5,
            }
        ]
    )


def test_template_pack_has_exact_schemas_and_no_fabricated_confirmation(tmp_path):
    pbf = _pbf(tmp_path)
    input_dir = tmp_path / "inputs"
    pack = write_operational_template_pack(
        input_dir,
        ["OFFICIAL_FACILITY_1", "HIRA_FACILITY_2"],
        osm_pbf_path=pbf,
    )

    assert set(path.name for path in pack.input_paths.values()) == set(
        OPERATIONAL_INPUT_FILENAMES.values()
    )
    assert (input_dir / FROZEN_GRAPH_SOURCE_FILENAME).is_file()
    assert (input_dir / TEMPLATE_GUIDE_FILENAME).is_file()
    assert (input_dir / INPUT_MANIFEST_FILENAME).is_file()
    assert pack.graph_sha256 == hashlib.sha256(pbf.read_bytes()).hexdigest()
    assert verify_operational_input_manifest(
        pack.manifest_path, input_dir=input_dir, osm_pbf_path=pbf
    ).identical

    loaded = load_operational_input_dir(input_dir)
    assert isinstance(loaded.tables, OperationalTables)
    assert loaded.issues == ()
    for key, expected in TABLE_SCHEMAS.items():
        assert loaded.tables.__getattribute__(key).columns.tolist() == list(expected)

    capability = loaded.tables.team_bundle_capability
    assert tuple(capability["bundle_id"]) == DEFAULT_BUNDLE_IDS
    assert not capability["confirmed"].astype(bool).any()
    assert capability["capable"].eq("UNKNOWN").all()
    assert capability["team_id"].fillna("").eq("").all()

    venues = loaded.tables.venue_calendar
    assert set(venues["venue_id"]) == {"OFFICIAL_FACILITY_1", "HIRA_FACILITY_2"}
    assert venues["available"].eq("UNKNOWN").all()
    assert not venues["confirmed"].astype(bool).any()

    travel = loaded.tables.travel
    assert travel["graph_sha"].eq(pack.graph_sha256).all()
    assert travel["reachable"].eq("UNKNOWN").all()
    assert not travel["confirmed"].astype(bool).any()

    conflicts = loaded.tables.existing_service_conflicts
    assert conflicts["conflict"].eq("UNKNOWN").all()
    assert not conflicts["confirmed"].astype(bool).any()

    audit = audit_operational_contract(
        loaded.tables,
        _visits(),
        required_visit_count=1,
        expected_graph_sha=pack.graph_sha256,
    )
    assert not audit.ready
    assert any("UNKNOWN" in blocker or "EMPTY_TABLE" in blocker for blocker in audit.blockers)


def test_bundle_guide_and_frozen_graph_source_are_explicit(tmp_path):
    pbf = _pbf(tmp_path)
    pack = write_operational_template_pack(
        tmp_path / "inputs", ["OFFICIAL_FACILITY_1"], osm_pbf_path=pbf
    )
    guide = pack.guide_path.read_text(encoding="utf-8")
    for bundle_id in DEFAULT_BUNDLE_IDS:
        assert bundle_id in guide
    assert "부재는 no-conflict가 아니다" in guide

    source = json.loads(pack.frozen_graph_source_path.read_text(encoding="utf-8"))
    assert source["sha256"] == hashlib.sha256(pbf.read_bytes()).hexdigest()
    assert source["graph_sha_semantics"] == "SHA256_OF_SOURCE_PBF_BYTES"
    assert source["operational_team_base_confirmed"] is False
    assert source["live_traffic"] is False


def test_header_only_zero_byte_and_missing_files_remain_auditable(tmp_path):
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    # A correct header-only file stays a correct empty DataFrame.
    pd.DataFrame(columns=TABLE_SCHEMAS["bases"]).to_csv(
        input_dir / OPERATIONAL_INPUT_FILENAMES["bases"], index=False
    )
    # A zero-byte exact file is converted to the correct empty schema with an issue.
    (input_dir / OPERATIONAL_INPUT_FILENAMES["vehicles"]).write_bytes(b"")

    loaded = load_operational_input_dir(input_dir)
    assert loaded.tables.bases.empty
    assert loaded.tables.bases.columns.tolist() == list(TABLE_SCHEMAS["bases"])
    assert loaded.tables.vehicles.empty
    assert loaded.tables.vehicles.columns.tolist() == list(TABLE_SCHEMAS["vehicles"])
    assert any(value.startswith("ZERO_BYTE_FILE:vehicles") for value in loaded.issues)
    assert any(value.startswith("MISSING_FILE:travel") for value in loaded.issues)

    # The strict contract reports blockers rather than the I/O layer inventing data.
    audit = audit_operational_contract(
        loaded.tables,
        _visits(),
        required_visit_count=1,
        expected_graph_sha="a" * 64,
    )
    assert not audit.ready
    assert "EMPTY_TABLE:bases" in audit.blockers
    assert "EMPTY_TABLE:vehicles" in audit.blockers


def test_loader_uses_exact_filenames_without_alias_fallback(tmp_path):
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    pd.DataFrame(columns=TABLE_SCHEMAS["bases"]).to_csv(
        input_dir / "bases.csv", index=False
    )
    loaded = load_operational_input_dir(input_dir)
    assert loaded.tables.bases.empty
    assert any(value.startswith("MISSING_FILE:bases") for value in loaded.issues)
    assert loaded.paths["bases"].name == "mobile_team_bases.csv"


def test_snapshot_detects_input_and_pbf_changes(tmp_path):
    pbf = _pbf(tmp_path)
    input_dir = tmp_path / "inputs"
    pack = write_operational_template_pack(
        input_dir, ["OFFICIAL_FACILITY_1"], osm_pbf_path=pbf
    )
    before = capture_operational_input_snapshot(input_dir, osm_pbf_path=pbf)
    same = capture_operational_input_snapshot(input_dir, osm_pbf_path=pbf)
    assert compare_operational_input_snapshots(before, same).identical
    assert assert_operational_inputs_unchanged(before, same).identical

    with pack.input_paths["venue_calendar"].open("a", encoding="utf-8") as handle:
        handle.write("\n")
    changed_input = capture_operational_input_snapshot(input_dir, osm_pbf_path=pbf)
    comparison = compare_operational_input_snapshots(before, changed_input)
    assert not comparison.identical
    assert any(item["kind"] == "INPUT_FILE_CHANGED" for item in comparison.differences)
    with pytest.raises(OperationalContractError, match="changed during execution"):
        assert_operational_inputs_unchanged(before, changed_input)

    # Restore a new baseline, then prove source-PBF mutation is independent.
    before_graph_change = capture_operational_input_snapshot(input_dir, osm_pbf_path=pbf)
    pbf.write_bytes(b"different-pbf-bytes")
    changed_graph = capture_operational_input_snapshot(input_dir, osm_pbf_path=pbf)
    comparison = compare_operational_input_snapshots(before_graph_change, changed_graph)
    assert not comparison.identical
    assert any(item["kind"] == "FROZEN_GRAPH_SOURCE_CHANGED" for item in comparison.differences)


def test_saved_manifest_verifies_every_exact_input_and_graph(tmp_path):
    pbf = _pbf(tmp_path)
    input_dir = tmp_path / "inputs"
    write_operational_template_pack(
        input_dir, ["OFFICIAL_FACILITY_1"], osm_pbf_path=pbf
    )
    snapshot = capture_operational_input_snapshot(input_dir, osm_pbf_path=pbf)
    execution_manifest = tmp_path / "before-run.json"
    write_operational_input_manifest(execution_manifest, snapshot)
    assert verify_operational_input_manifest(
        execution_manifest, input_dir=input_dir, osm_pbf_path=pbf
    ).identical

    (input_dir / OPERATIONAL_INPUT_FILENAMES["team_calendar"]).write_text(
        "corrupted", encoding="utf-8"
    )
    assert not verify_operational_input_manifest(
        execution_manifest, input_dir=input_dir, osm_pbf_path=pbf
    ).identical


def test_template_writer_refuses_busproxy_placeholders_and_overwrite(tmp_path):
    pbf = _pbf(tmp_path)
    with pytest.raises(OperationalContractError, match="BUSPROXY"):
        write_operational_template_pack(
            tmp_path / "bad", ["BUSPROXY_123"], osm_pbf_path=pbf
        )
    with pytest.raises(OperationalContractError, match="Placeholder"):
        write_operational_template_pack(
            tmp_path / "bad-placeholder", ["REPLACE_VENUE_ID"], osm_pbf_path=pbf
        )
    with pytest.raises(OperationalContractError, match="Placeholder"):
        write_operational_template_pack(
            tmp_path / "bad-null", [None], osm_pbf_path=pbf
        )

    target = tmp_path / "inputs"
    write_operational_template_pack(
        target, ["OFFICIAL_FACILITY_1"], osm_pbf_path=pbf
    )
    with pytest.raises(OperationalContractError, match="overwrite"):
        write_operational_template_pack(
            target, ["OFFICIAL_FACILITY_1"], osm_pbf_path=pbf
        )


def test_convenience_loader_returns_operational_tables(tmp_path):
    pbf = _pbf(tmp_path)
    input_dir = tmp_path / "inputs"
    write_operational_template_pack(
        input_dir, ["OFFICIAL_FACILITY_1"], osm_pbf_path=pbf
    )
    assert isinstance(load_operational_tables(input_dir), OperationalTables)
