from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ig = pytest.importorskip("igraph")
pytest.importorskip("pyarrow")

from mediroad.stage4_operational_final.operational_contract import TRAVEL_COLUMNS
import mediroad.stage4_operational_final.operational_travel as travel_module
from mediroad.stage4_operational_final.operational_travel import (
    GRAPH_DEFINITION,
    OperationalTravelError,
    recompute_operational_travel,
)


def _file_sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    bases = pd.DataFrame(
        [
            {
                "team_id": "TEAM_A",
                "base_id": "BASE_A",
                "base_name": "A",
                "latitude": 36.0000,
                "longitude": 127.0000,
                "confirmed": True,
                "source_reference": "confirmed-base-a",
                "effective_date": "2026-08-22",
            },
            {
                "team_id": "TEAM_B",
                "base_id": "BASE_B",
                "base_name": "B",
                "latitude": 36.0000,
                "longitude": 127.0010,
                "confirmed": True,
                "source_reference": "confirmed-base-b",
                "effective_date": "2026-08-22",
            },
        ]
    )
    venues = pd.DataFrame(
        [
            {
                "venue_id": "VENUE_A",
                "verified_latitude": 36.0000,
                "verified_longitude": 127.0000,
                "operational_release_verified": True,
                "verification_date": "2026-08-22",
                "source_reference": "field-evidence-a",
            },
            {
                "venue_id": "VENUE_B",
                "verified_latitude": 36.0000,
                "verified_longitude": 127.0010,
                "operational_release_verified": True,
                "verification_date": "2026-08-22",
                "source_reference": "field-evidence-b",
            },
            {
                "venue_id": "VENUE_ISOLATED",
                "verified_latitude": 36.0010,
                "verified_longitude": 127.0010,
                "operational_release_verified": True,
                "verification_date": "2026-08-22",
                "source_reference": "field-evidence-isolated",
            },
        ]
    )
    return bases, venues


def _write_graph_cache(
    root: Path,
    *,
    nan_time: bool = False,
    metadata_pbf_sha: str | None = None,
) -> tuple[Path, Path]:
    pbf = root / "network.osm.pbf"
    pbf.write_bytes(b"frozen synthetic pbf")
    pbf_sha = _file_sha(pbf)
    cache = root / "road_graph"
    cache.mkdir()
    graph = ig.Graph(n=3, edges=[(0, 1)], directed=True)
    graph.es["travel_time_min"] = [float("nan") if nan_time else 2.0]
    graph.es["length"] = [1_200.0]
    graph_path = cache / f"chungbuk_drive_core_{pbf_sha[:16]}.igraph.pickle"
    graph.write_pickle(str(graph_path))
    nodes = pd.DataFrame(
        {
            "node_id": ["n0", "n1", "n2"],
            "lon": [127.0000, 127.0010, 127.0010],
            "lat": [36.0000, 36.0000, 36.0010],
            "vertex_index": [0, 1, 2],
        }
    )
    nodes.to_parquet(
        cache / f"chungbuk_drive_core_nodes_{pbf_sha[:16]}.parquet", index=False
    )
    metadata = {
        "pbf_sha256": metadata_pbf_sha or pbf_sha,
        "definition": GRAPH_DEFINITION,
        "core_vertices": 3,
        "core_edges": 1,
    }
    (cache / f"chungbuk_drive_core_{pbf_sha[:16]}.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    return pbf, cache


def _run(root: Path, bases: pd.DataFrame, venues: pd.DataFrame, **kwargs: object):
    pbf = root / "network.osm.pbf"
    return recompute_operational_travel(
        bases,
        venues,
        package_root=root,
        pbf_path=pbf,
        graph_cache_dir=root / "road_graph",
        result_cache_dir=root / "travel_cache",
        max_snap_distance_m=100.0,
        **kwargs,
    )


def test_directed_oneway_unreachable_is_null_not_zero_and_schema_is_exact(tmp_path: Path) -> None:
    _write_graph_cache(tmp_path)
    bases, venues = _inputs()
    result = _run(tmp_path, bases, venues)

    assert tuple(result.travel.columns) == tuple(TRAVEL_COLUMNS)
    by_pair = result.travel.set_index(["team_id", "venue_id"])
    forward = by_pair.loc[("TEAM_A", "VENUE_B")]
    assert bool(forward["reachable"])
    assert forward["distance_km"] == pytest.approx(1.2)
    assert forward["travel_minutes_proxy"] == pytest.approx(2.0)

    reverse = by_pair.loc[("TEAM_B", "VENUE_A")]
    assert not bool(reverse["reachable"])
    assert pd.isna(reverse["distance_km"])
    assert pd.isna(reverse["travel_minutes_proxy"])
    isolated = by_pair.loc[("TEAM_A", "VENUE_ISOLATED")]
    assert not bool(isolated["reachable"])
    assert pd.isna(isolated["distance_km"])
    assert result.audit["computation"]["unreachable_numeric_values_are_null"]
    assert result.travel["graph_sha"].eq(_file_sha(tmp_path / "network.osm.pbf")).all()
    assert not result.cache_hit


def test_valid_cache_reuse_and_input_hash_invalidation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_graph_cache(tmp_path)
    bases, venues = _inputs()
    first = _run(tmp_path, bases, venues)
    original_loader = travel_module._load_graph_cache
    monkeypatch.setattr(
        travel_module,
        "_load_graph_cache",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("graph loaded on cache hit")),
    )
    second = _run(tmp_path, bases, venues)
    assert second.cache_hit
    assert second.cache_key == first.cache_key
    assert second.cache_path == first.cache_path
    pd.testing.assert_frame_equal(second.travel, first.travel, check_dtype=False)

    monkeypatch.setattr(travel_module, "_load_graph_cache", original_loader)

    changed = bases.copy()
    changed.loc[changed["team_id"].eq("TEAM_A"), "source_reference"] = "new-base-evidence"
    third = _run(tmp_path, changed, venues)
    assert not third.cache_hit
    assert third.cache_key != first.cache_key
    assert third.cache_path != first.cache_path

    changed_venues = venues.copy()
    changed_venues.loc[
        changed_venues["venue_id"].eq("VENUE_B"), "source_reference"
    ] = "new-field-evidence"
    fourth = _run(tmp_path, bases, changed_venues)
    assert fourth.cache_key not in {first.cache_key, third.cache_key}


def test_corrupted_cached_result_sha_fails_closed(tmp_path: Path) -> None:
    _write_graph_cache(tmp_path)
    bases, venues = _inputs()
    first = _run(tmp_path, bases, venues)
    travel_path = first.cache_path / "team_venue_travel.csv"
    travel_path.write_text(travel_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(OperationalTravelError, match="size mismatch|SHA mismatch"):
        _run(tmp_path, bases, venues)


def test_snap_distance_over_threshold_fails_closed(tmp_path: Path) -> None:
    _write_graph_cache(tmp_path)
    bases, venues = _inputs()
    venues.loc[venues["venue_id"].eq("VENUE_B"), "verified_longitude"] = 127.05
    with pytest.raises(OperationalTravelError, match="exceed the graph snap threshold"):
        _run(tmp_path, bases, venues)


def test_nan_edge_weight_fails_closed(tmp_path: Path) -> None:
    _write_graph_cache(tmp_path, nan_time=True)
    bases, venues = _inputs()
    with pytest.raises(OperationalTravelError, match="travel_time_min.*nonfinite"):
        _run(tmp_path, bases, venues)


def test_pbf_metadata_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    _write_graph_cache(tmp_path, metadata_pbf_sha="0" * 64)
    bases, venues = _inputs()
    with pytest.raises(OperationalTravelError, match="metadata PBF SHA"):
        _run(tmp_path, bases, venues)


def test_unknown_confirmation_or_non_exact_release_fails_closed(tmp_path: Path) -> None:
    _write_graph_cache(tmp_path)
    bases, venues = _inputs()
    bases["confirmed"] = bases["confirmed"].astype(object)
    bases.loc[0, "confirmed"] = "UNKNOWN"
    with pytest.raises(OperationalTravelError, match="bases.confirmed"):
        _run(tmp_path, bases, venues)

    bases, venues = _inputs()
    venues.loc[0, "operational_release_verified"] = False
    with pytest.raises(OperationalTravelError, match="operational_release_verified"):
        _run(tmp_path, bases, venues)


def test_mixed_reference_or_evidence_file_is_coalesced_per_venue(tmp_path: Path) -> None:
    _write_graph_cache(tmp_path)
    bases, venues = _inputs()
    venues["evidence_file_or_url"] = ""
    venues.loc[0, "evidence_file_or_url"] = "field-photo-a.jpg"
    venues.loc[0, "source_reference"] = ""
    result = _run(tmp_path, bases, venues)
    row = result.travel.loc[result.travel["venue_id"].eq("VENUE_A")].iloc[0]
    assert "VENUE_REFERENCE=field-photo-a.jpg" in row["source_reference"]
