"""Strict OSM travel recomputation for Stage 4 Operational Final.

This module turns confirmed team bases and independently verified, exact
physical venues into the ``team_venue_travel`` evidence table required by the
operational contract.  It deliberately does not construct visit dates or a
schedule.

The route model is the frozen Stage 4 graph: static OSM driving free-flow,
directed/oneway, retain-all before extraction of the largest strongly connected
core, and unsimplified edges.  ``travel_time_min`` and ``length`` are used as
independent shortest-path weights.  Missing, invalid, or unreachable values are
never changed to zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import shutil
from typing import Any, Mapping, Sequence
from uuid import uuid4

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree

from .operational_contract import TRAVEL_COLUMNS


TRAVEL_RECOMPUTE_SCHEMA_VERSION = "1.0"
GRAPH_DEFINITION = (
    "Static OSM driving car-profile free-flow; oneway restrictions preserved; "
    "largest SCC core."
)
DEFAULT_MAX_SNAP_DISTANCE_M = 1_000.0

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_TRUE_TEXT = frozenset({"TRUE", "1", "YES", "Y"})
_FALSE_TEXT = frozenset({"FALSE", "0", "NO", "N"})
_PLACEHOLDER = frozenset({"", "UNKNOWN", "NA", "N/A", "NONE", "NULL", "NAN"})


class OperationalTravelError(RuntimeError):
    """Raised when travel evidence cannot be derived without assumptions."""


@dataclass(frozen=True)
class OperationalTravelResult:
    """Travel table plus provenance needed by the runner and final audit."""

    travel: pd.DataFrame
    snap_audit: pd.DataFrame
    audit: Mapping[str, Any]
    cache_key: str
    cache_path: Path
    cache_hit: bool


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _json_scalar(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not math.isfinite(float(value)) else float(value)
    if pd.isna(value):
        return None
    return str(value)


def _frame_sha256(frame: pd.DataFrame, *, label: str) -> str:
    payload = {
        "label": label,
        "columns": list(frame.columns),
        "records": [
            {column: _json_scalar(row[column]) for column in frame.columns}
            for row in frame.to_dict("records")
        ],
    }
    return sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _as_text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def _strict_true(series: pd.Series, *, label: str) -> pd.Series:
    def parse(value: Any) -> bool | None:
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if value is None or (not isinstance(value, str) and pd.isna(value)):
            return None
        text = str(value).strip().upper()
        if text in _TRUE_TEXT:
            return True
        if text in _FALSE_TEXT:
            return False
        return None

    parsed = series.map(parse)
    if parsed.isna().any() or not bool(parsed.all()):
        raise OperationalTravelError(f"{label} must be explicitly true for every row")
    return parsed.astype(bool)


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], *, label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise OperationalTravelError(f"{label} missing required columns: {missing}")


def _validate_iso_dates(series: pd.Series, *, label: str) -> pd.Series:
    text = _as_text(series)
    invalid = ~text.str.match(_ISO_DATE_RE)
    parsed = pd.to_datetime(text, format="%Y-%m-%d", errors="coerce")
    if invalid.any() or parsed.isna().any():
        raise OperationalTravelError(f"{label} contains an invalid or unknown ISO date")
    return text


def _validate_wgs84(frame: pd.DataFrame, lat: str, lon: str, *, label: str) -> None:
    frame[lat] = pd.to_numeric(frame[lat], errors="coerce")
    frame[lon] = pd.to_numeric(frame[lon], errors="coerce")
    finite = np.isfinite(frame[lat].to_numpy(dtype=float)) & np.isfinite(
        frame[lon].to_numpy(dtype=float)
    )
    # Tight enough to reject swapped coordinates and non-Korean placeholders,
    # while covering the complete Korean peninsula and nearby graph buffer.
    korean = frame[lat].between(32.0, 39.5) & frame[lon].between(124.0, 132.0)
    if not bool(np.all(finite)) or not bool(korean.all()):
        raise OperationalTravelError(f"{label} coordinates must be finite Korean WGS84")


def _normalize_bases(bases: pd.DataFrame) -> pd.DataFrame:
    required = [
        "team_id",
        "base_id",
        "latitude",
        "longitude",
        "confirmed",
        "source_reference",
        "effective_date",
    ]
    _require_columns(bases, required, label="confirmed bases")
    work = bases[required].copy()
    for column in ("team_id", "base_id", "source_reference"):
        work[column] = _as_text(work[column])
        if work[column].str.upper().isin(_PLACEHOLDER).any():
            raise OperationalTravelError(f"confirmed bases contain missing {column}")
    _strict_true(work["confirmed"], label="bases.confirmed")
    work["confirmed"] = True
    work["effective_date"] = _validate_iso_dates(
        work["effective_date"], label="bases.effective_date"
    )
    _validate_wgs84(work, "latitude", "longitude", label="base")
    if work[["team_id", "base_id"]].duplicated().any():
        raise OperationalTravelError("confirmed bases contain duplicate team_id/base_id")
    if work["team_id"].duplicated().any():
        raise OperationalTravelError(
            "travel table is keyed by team_id/venue_id, so each team must have exactly one confirmed base"
        )
    return work.sort_values(["team_id", "base_id"], kind="stable").reset_index(drop=True)


def _normalize_venues(venues: pd.DataFrame) -> pd.DataFrame:
    required = [
        "venue_id",
        "verified_latitude",
        "verified_longitude",
        "operational_release_verified",
        "verification_date",
    ]
    _require_columns(venues, required, label="verified operational venues")
    reference_columns = [
        column for column in ("source_reference", "evidence_file_or_url") if column in venues
    ]
    if not reference_columns:
        raise OperationalTravelError(
            "verified venues require source_reference or evidence_file_or_url"
        )
    selected = required + reference_columns
    work = venues[selected].copy()
    work["venue_id"] = _as_text(work["venue_id"])
    for column in reference_columns:
        work[column] = _as_text(work[column])
    work["venue_source_reference"] = ""
    for column in reference_columns:
        usable = ~work[column].str.upper().isin(_PLACEHOLDER)
        fill = work["venue_source_reference"].eq("") & usable
        work.loc[fill, "venue_source_reference"] = work.loc[fill, column]
    if work["venue_id"].str.upper().isin(_PLACEHOLDER).any():
        raise OperationalTravelError("verified operational venues contain missing venue_id")
    if work["venue_source_reference"].str.upper().isin(_PLACEHOLDER).any():
        raise OperationalTravelError("verified operational venues contain missing evidence reference")
    _strict_true(
        work["operational_release_verified"],
        label="venues.operational_release_verified",
    )
    work["operational_release_verified"] = True
    work["verification_date"] = _validate_iso_dates(
        work["verification_date"], label="venues.verification_date"
    )
    _validate_wgs84(
        work,
        "verified_latitude",
        "verified_longitude",
        label="verified venue",
    )
    if work["venue_id"].duplicated().any():
        raise OperationalTravelError("verified operational venues contain duplicate venue_id")
    work = work.drop(columns=reference_columns)
    return work.sort_values("venue_id", kind="stable").reset_index(drop=True)


def _resolve_path(package_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else package_root / path


def _graph_cache_paths(cache_dir: Path, pbf_sha: str) -> tuple[Path, Path, Path]:
    prefix = pbf_sha[:16]
    return (
        cache_dir / f"chungbuk_drive_core_{prefix}.igraph.pickle",
        cache_dir / f"chungbuk_drive_core_nodes_{prefix}.parquet",
        cache_dir / f"chungbuk_drive_core_{prefix}.json",
    )


def _inspect_graph_cache_files(
    graph_cache_dir: Path,
    pbf_sha: str,
) -> tuple[dict[str, Any], dict[str, str]] | None:
    graph_path, nodes_path, meta_path = _graph_cache_paths(graph_cache_dir, pbf_sha)
    present = [path.exists() for path in (graph_path, nodes_path, meta_path)]
    if any(present) and not all(present):
        raise OperationalTravelError(
            "partial road-graph cache found; graph, node table, and metadata must all exist"
        )
    if not any(present):
        return None
    if not all(path.is_file() for path in (graph_path, nodes_path, meta_path)):
        raise OperationalTravelError("road graph cache entries must be regular files")
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise OperationalTravelError(f"road graph metadata is unreadable: {exc}") from exc
    if str(meta.get("pbf_sha256", "")).strip().lower() != pbf_sha:
        raise OperationalTravelError("road graph metadata PBF SHA does not match the supplied PBF")
    if str(meta.get("definition", "")).strip() != GRAPH_DEFINITION:
        raise OperationalTravelError("road graph definition does not match frozen Stage 4 semantics")
    file_hashes = {
        "graph_pickle_sha256": _sha256_file(graph_path),
        "graph_nodes_sha256": _sha256_file(nodes_path),
        "graph_metadata_sha256": _sha256_file(meta_path),
    }
    return dict(meta), file_hashes


def _load_graph_cache(
    package_root: Path,
    pbf_path: Path,
    graph_cache_dir: Path,
    pbf_sha: str,
    *,
    graph_bbox_buffer_deg: float,
    inspected: tuple[dict[str, Any], dict[str, str]] | None = None,
) -> tuple[Any, pd.DataFrame, dict[str, Any], dict[str, str]]:
    try:
        import igraph as ig
    except ImportError as exc:  # pragma: no cover - depends on execution environment
        raise OperationalTravelError(
            "python-igraph is required to read the frozen Stage 4 road graph"
        ) from exc

    inspection = inspected if inspected is not None else _inspect_graph_cache_files(
        graph_cache_dir, pbf_sha
    )
    if inspection is None:
        # Reuse the single frozen builder so driving, oneway, retain_all,
        # simplify=False, speed, and largest-SCC semantics cannot drift.
        from mediroad.stage4.network_validation import _build_or_load_graph

        try:
            _build_or_load_graph(
                package_root,
                pbf_path,
                graph_cache_dir,
                float(graph_bbox_buffer_deg),
            )
        except Exception as exc:
            raise OperationalTravelError(f"road graph build/load failed: {exc}") from exc
        inspection = _inspect_graph_cache_files(graph_cache_dir, pbf_sha)
        if inspection is None:  # pragma: no cover - defensive
            raise OperationalTravelError("road graph builder did not produce the expected cache")

    graph_path, nodes_path, meta_path = _graph_cache_paths(graph_cache_dir, pbf_sha)
    meta, file_hashes = inspection
    try:
        graph = ig.Graph.Read_Pickle(str(graph_path))
        nodes = pd.read_parquet(nodes_path)
    except Exception as exc:
        raise OperationalTravelError(f"road graph cache is unreadable: {exc}") from exc
    if not graph.is_directed():
        raise OperationalTravelError("road graph must be directed to preserve oneway restrictions")
    edge_attrs = set(graph.es.attributes())
    missing_weights = {"travel_time_min", "length"} - edge_attrs
    if missing_weights:
        raise OperationalTravelError(
            f"road graph lacks required independent weights: {sorted(missing_weights)}"
        )
    for attribute in ("travel_time_min", "length"):
        try:
            values = np.asarray(graph.es[attribute], dtype=float)
        except Exception as exc:
            raise OperationalTravelError(f"graph edge {attribute} is not numeric") from exc
        if len(values) != graph.ecount() or not np.isfinite(values).all() or np.any(values <= 0):
            raise OperationalTravelError(
                f"graph edge {attribute} contains missing, nonfinite, or nonpositive values"
            )

    required_nodes = {"node_id", "lon", "lat", "vertex_index"}
    if not required_nodes.issubset(nodes.columns):
        raise OperationalTravelError(
            f"road graph node table missing columns: {sorted(required_nodes - set(nodes.columns))}"
        )
    nodes = nodes[["node_id", "lon", "lat", "vertex_index"]].copy()
    nodes["vertex_index"] = pd.to_numeric(nodes["vertex_index"], errors="coerce")
    nodes["lon"] = pd.to_numeric(nodes["lon"], errors="coerce")
    nodes["lat"] = pd.to_numeric(nodes["lat"], errors="coerce")
    expected_index = np.arange(graph.vcount(), dtype=np.int64)
    if (
        len(nodes) != graph.vcount()
        or nodes["vertex_index"].isna().any()
        or not np.array_equal(nodes["vertex_index"].to_numpy(dtype=np.int64), expected_index)
        or not np.isfinite(nodes[["lon", "lat"]].to_numpy(dtype=float)).all()
    ):
        raise OperationalTravelError("road graph node table is not exactly aligned to graph vertices")

    return graph, nodes, dict(meta), file_hashes


def _snap(
    nodes: pd.DataFrame,
    points: pd.DataFrame,
    *,
    id_columns: Sequence[str],
    latitude: str,
    longitude: str,
    kind: str,
) -> pd.DataFrame:
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:5179", always_xy=True)
    node_x, node_y = transformer.transform(
        nodes["lon"].to_numpy(dtype=float), nodes["lat"].to_numpy(dtype=float)
    )
    point_x, point_y = transformer.transform(
        points[longitude].to_numpy(dtype=float), points[latitude].to_numpy(dtype=float)
    )
    if not np.isfinite(np.c_[node_x, node_y]).all() or not np.isfinite(
        np.c_[point_x, point_y]
    ).all():
        raise OperationalTravelError("EPSG:5179 projection produced nonfinite coordinates")
    distance, index = cKDTree(np.c_[node_x, node_y]).query(np.c_[point_x, point_y], k=1)
    snapped = points[list(id_columns)].copy()
    snapped["point_kind"] = kind
    snapped["input_latitude"] = points[latitude].to_numpy(dtype=float)
    snapped["input_longitude"] = points[longitude].to_numpy(dtype=float)
    snapped["vertex_index"] = nodes.iloc[index]["vertex_index"].to_numpy(dtype=np.int64)
    snapped["node_id"] = nodes.iloc[index]["node_id"].astype(str).to_numpy()
    snapped["node_latitude"] = nodes.iloc[index]["lat"].to_numpy(dtype=float)
    snapped["node_longitude"] = nodes.iloc[index]["lon"].to_numpy(dtype=float)
    snapped["snap_distance_m"] = np.asarray(distance, dtype=float)
    return snapped


def _atomic_write_cache(
    cache_path: Path,
    travel: pd.DataFrame,
    snap_audit: pd.DataFrame,
    audit: Mapping[str, Any],
    manifest_base: Mapping[str, Any],
) -> None:
    parent = cache_path.parent
    parent.mkdir(parents=True, exist_ok=True)
    temp = parent / f".{cache_path.name}.{uuid4().hex}.tmp"
    temp.mkdir(parents=False, exist_ok=False)
    try:
        travel_path = temp / "team_venue_travel.csv"
        snap_path = temp / "travel_snap_audit.csv"
        audit_path = temp / "TRAVEL_RECOMPUTE_AUDIT.json"
        travel.to_csv(travel_path, index=False, encoding="utf-8")
        snap_audit.to_csv(snap_path, index=False, encoding="utf-8")
        audit_path.write_text(
            json.dumps(dict(audit), ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        files = {
            path.name: {"sha256": _sha256_file(path), "size_bytes": path.stat().st_size}
            for path in (travel_path, snap_path, audit_path)
        }
        manifest = {**dict(manifest_base), "files": files}
        (temp / "CACHE_MANIFEST.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        try:
            os.replace(temp, cache_path)
        except OSError:
            # A concurrent identical computation won the race.  Its manifest
            # is verified by the normal cache reader before it can be used.
            if not cache_path.exists():
                raise
            shutil.rmtree(temp)
    finally:
        if temp.exists():
            shutil.rmtree(temp)


def _read_cache(cache_path: Path, expected: Mapping[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]] | None:
    if not cache_path.exists():
        return None
    if not cache_path.is_dir():
        raise OperationalTravelError("travel result cache path exists but is not a directory")
    manifest_path = cache_path / "CACHE_MANIFEST.json"
    if not manifest_path.is_file():
        raise OperationalTravelError("travel result cache is incomplete: manifest missing")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise OperationalTravelError(f"travel result cache manifest is unreadable: {exc}") from exc
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise OperationalTravelError(f"travel result cache manifest mismatch: {key}")
    expected_names = {
        "team_venue_travel.csv",
        "travel_snap_audit.csv",
        "TRAVEL_RECOMPUTE_AUDIT.json",
    }
    if set(manifest.get("files", {})) != expected_names:
        raise OperationalTravelError("travel result cache file set is not exact")
    actual_names = {path.name for path in cache_path.iterdir() if path.is_file()}
    if actual_names != expected_names | {"CACHE_MANIFEST.json"}:
        raise OperationalTravelError("travel result cache contains missing or unexpected files")
    for name in expected_names:
        path = cache_path / name
        record = manifest["files"][name]
        if not path.is_file() or path.stat().st_size != int(record.get("size_bytes", -1)):
            raise OperationalTravelError(f"travel result cache size mismatch: {name}")
        if _sha256_file(path) != str(record.get("sha256", "")):
            raise OperationalTravelError(f"travel result cache SHA mismatch: {name}")
    try:
        travel = pd.read_csv(cache_path / "team_venue_travel.csv", dtype={"team_id": str, "venue_id": str})
        snap = pd.read_csv(cache_path / "travel_snap_audit.csv")
        audit = json.loads((cache_path / "TRAVEL_RECOMPUTE_AUDIT.json").read_text(encoding="utf-8"))
    except Exception as exc:
        raise OperationalTravelError(f"travel result cache contents are unreadable: {exc}") from exc
    if tuple(travel.columns) != tuple(TRAVEL_COLUMNS):
        raise OperationalTravelError("cached travel table schema mismatch")
    if _frame_sha256(travel, label="TEAM_VENUE_TRAVEL") != audit.get("travel_table_sha256"):
        raise OperationalTravelError("cached travel table canonical SHA mismatch")
    if _frame_sha256(snap, label="TRAVEL_SNAP_AUDIT") != audit.get("snap_audit_sha256"):
        raise OperationalTravelError("cached snap audit canonical SHA mismatch")
    return travel, snap, audit


def _cache_identity(
    *,
    pbf_sha: str,
    base_sha: str,
    venue_sha: str,
    graph_hashes: Mapping[str, str],
    max_snap_distance_m: float,
    result_cache: Path,
) -> tuple[dict[str, Any], str, Path, dict[str, Any]]:
    inputs = {
        "schema_version": TRAVEL_RECOMPUTE_SCHEMA_VERSION,
        "pbf_sha256": pbf_sha,
        "base_input_sha256": base_sha,
        "venue_input_sha256": venue_sha,
        "graph_pickle_sha256": graph_hashes["graph_pickle_sha256"],
        "graph_nodes_sha256": graph_hashes["graph_nodes_sha256"],
        "graph_metadata_sha256": graph_hashes["graph_metadata_sha256"],
        "max_snap_distance_m": float(max_snap_distance_m),
        "time_weight": "travel_time_min",
        "distance_weight": "length",
        "route_mode": "OUT",
    }
    key = sha256(_canonical_json(inputs).encode("utf-8")).hexdigest()
    path = result_cache / f"operational_travel_{key[:20]}"
    return inputs, key, path, {**inputs, "cache_key": key}


def recompute_operational_travel(
    bases: pd.DataFrame,
    venues: pd.DataFrame,
    *,
    package_root: str | Path,
    pbf_path: str | Path,
    graph_cache_dir: str | Path | None = None,
    result_cache_dir: str | Path | None = None,
    max_snap_distance_m: float = DEFAULT_MAX_SNAP_DISTANCE_M,
    graph_bbox_buffer_deg: float = 0.10,
) -> OperationalTravelResult:
    """Compute unique confirmed ``team x venue`` directed shortest paths.

    ``bases`` must use the operational base schema and contain exactly one
    confirmed, referenced base per team.  ``venues`` is the candidate-universe
    table: it must contain release-verified exact coordinates, evidence, and a
    verification date.  The latter is carried as the travel effective date;
    it is evidence recency, not a proposed visit date.
    """

    root = Path(package_root).resolve()
    pbf = _resolve_path(root, pbf_path).resolve()
    if not pbf.is_file():
        raise OperationalTravelError(f"OSM PBF not found: {pbf}")
    pbf_sha = _sha256_file(pbf)
    if not _SHA256_RE.match(pbf_sha):  # defensive; hashlib always satisfies this
        raise OperationalTravelError("could not derive a valid OSM PBF SHA-256")
    graph_cache = (
        _resolve_path(root, graph_cache_dir).resolve()
        if graph_cache_dir is not None
        else (root / "outputs/model_v1/08_stage4/cache/road_graph").resolve()
    )
    result_cache = (
        _resolve_path(root, result_cache_dir).resolve()
        if result_cache_dir is not None
        else (root / "outputs/model_v1/08_stage4/cache/operational_travel").resolve()
    )
    if not math.isfinite(float(max_snap_distance_m)) or float(max_snap_distance_m) <= 0:
        raise OperationalTravelError("max_snap_distance_m must be finite and positive")

    normalized_bases = _normalize_bases(bases)
    normalized_venues = _normalize_venues(venues)
    if normalized_bases.empty:
        raise OperationalTravelError("no confirmed team bases were supplied")
    if normalized_venues.empty:
        raise OperationalTravelError("no release-verified exact venues were supplied")
    base_sha = _frame_sha256(normalized_bases, label="CONFIRMED_TEAM_BASES")
    venue_sha = _frame_sha256(normalized_venues, label="VERIFIED_OPERATIONAL_VENUES")

    inspection = _inspect_graph_cache_files(graph_cache, pbf_sha)
    if inspection is not None:
        graph_meta, graph_hashes = inspection
        cache_inputs, cache_key, cache_path, manifest_expected = _cache_identity(
            pbf_sha=pbf_sha,
            base_sha=base_sha,
            venue_sha=venue_sha,
            graph_hashes=graph_hashes,
            max_snap_distance_m=float(max_snap_distance_m),
            result_cache=result_cache,
        )
        cached = _read_cache(cache_path, manifest_expected)
        if cached is not None:
            travel, snap_audit, stored_audit = cached
            runtime_audit = {
                **stored_audit,
                "cache_hit": True,
                "cache_path": str(cache_path),
            }
            return OperationalTravelResult(
                travel=travel,
                snap_audit=snap_audit,
                audit=runtime_audit,
                cache_key=cache_key,
                cache_path=cache_path,
                cache_hit=True,
            )

    graph, nodes, graph_meta, graph_hashes = _load_graph_cache(
        root,
        pbf,
        graph_cache,
        pbf_sha,
        graph_bbox_buffer_deg=float(graph_bbox_buffer_deg),
        inspected=inspection,
    )
    cache_inputs, cache_key, cache_path, manifest_expected = _cache_identity(
        pbf_sha=pbf_sha,
        base_sha=base_sha,
        venue_sha=venue_sha,
        graph_hashes=graph_hashes,
        max_snap_distance_m=float(max_snap_distance_m),
        result_cache=result_cache,
    )

    base_snap = _snap(
        nodes,
        normalized_bases,
        id_columns=["team_id", "base_id"],
        latitude="latitude",
        longitude="longitude",
        kind="TEAM_BASE",
    )
    venue_snap = _snap(
        nodes,
        normalized_venues,
        id_columns=["venue_id"],
        latitude="verified_latitude",
        longitude="verified_longitude",
        kind="VERIFIED_VENUE",
    )
    snap_audit = pd.concat([base_snap, venue_snap], ignore_index=True, sort=False)
    snap_audit["within_threshold"] = snap_audit["snap_distance_m"].le(
        float(max_snap_distance_m)
    )
    snap_audit["max_snap_distance_m"] = float(max_snap_distance_m)
    if not bool(snap_audit["within_threshold"].all()):
        failed = snap_audit.loc[
            ~snap_audit["within_threshold"],
            [column for column in ("team_id", "base_id", "venue_id", "snap_distance_m") if column in snap_audit],
        ].to_dict("records")
        raise OperationalTravelError(
            f"one or more confirmed points exceed the graph snap threshold: {failed[:10]}"
        )

    base_vertices, base_inverse = np.unique(
        base_snap["vertex_index"].to_numpy(dtype=np.int64), return_inverse=True
    )
    venue_vertices, venue_inverse = np.unique(
        venue_snap["vertex_index"].to_numpy(dtype=np.int64), return_inverse=True
    )
    try:
        minutes_unique = np.asarray(
            graph.distances(
                source=base_vertices.tolist(),
                target=venue_vertices.tolist(),
                weights="travel_time_min",
                mode="OUT",
            ),
            dtype=float,
        )
        metres_unique = np.asarray(
            graph.distances(
                source=base_vertices.tolist(),
                target=venue_vertices.tolist(),
                weights="length",
                mode="OUT",
            ),
            dtype=float,
        )
    except Exception as exc:
        raise OperationalTravelError(f"directed shortest-path computation failed: {exc}") from exc
    expected_shape = (len(base_vertices), len(venue_vertices))
    if minutes_unique.shape != expected_shape or metres_unique.shape != expected_shape:
        raise OperationalTravelError("shortest-path matrix shape mismatch")
    minutes = minutes_unique[np.ix_(base_inverse, venue_inverse)]
    metres = metres_unique[np.ix_(base_inverse, venue_inverse)]
    if np.isnan(minutes).any() or np.isnan(metres).any():
        raise OperationalTravelError("shortest-path engine returned NaN")
    reachable = np.isfinite(minutes) & np.isfinite(metres)
    if np.any(np.isfinite(minutes) != np.isfinite(metres)):
        raise OperationalTravelError("time and distance reachability disagree")
    if np.any(reachable & ((minutes < 0) | (metres < 0))):
        raise OperationalTravelError("shortest-path engine returned a negative route weight")

    rows: list[dict[str, Any]] = []
    for base_index, base in normalized_bases.iterrows():
        for venue_index, venue in normalized_venues.iterrows():
            is_reachable = bool(reachable[base_index, venue_index])
            route_source = (
                "OSM_STATIC_FREE_FLOW_DIRECTED;"
                f"PBF_SHA256={pbf_sha};"
                f"BASE_REFERENCE={base['source_reference']};"
                f"VENUE_REFERENCE={venue['venue_source_reference']}"
            )
            rows.append(
                {
                    "team_id": base["team_id"],
                    "venue_id": venue["venue_id"],
                    "reachable": is_reachable,
                    "distance_km": (
                        float(metres[base_index, venue_index] / 1000.0)
                        if is_reachable
                        else np.nan
                    ),
                    "travel_minutes_proxy": (
                        float(minutes[base_index, venue_index]) if is_reachable else np.nan
                    ),
                    # The operational contract compares this field to the
                    # frozen source-graph/PBF digest, not the pickle encoding.
                    "graph_sha": pbf_sha,
                    "confirmed": True,
                    "source_reference": route_source,
                    "effective_date": venue["verification_date"],
                }
            )
    travel = pd.DataFrame(rows, columns=TRAVEL_COLUMNS).sort_values(
        ["team_id", "venue_id"], kind="stable"
    ).reset_index(drop=True)
    if travel[["team_id", "venue_id"]].duplicated().any():
        raise OperationalTravelError("derived travel table contains duplicate team_id/venue_id")

    travel_sha = _frame_sha256(travel, label="TEAM_VENUE_TRAVEL")
    snap_sha = _frame_sha256(snap_audit, label="TRAVEL_SNAP_AUDIT")
    audit = {
        "schema_version": TRAVEL_RECOMPUTE_SCHEMA_VERSION,
        "status": "PASS",
        "cache_hit": False,
        "cache_key": cache_key,
        "cache_path": str(cache_path),
        "interpretation": (
            "Static OSM driving free-flow proxy from confirmed team bases to field-verified "
            "physical venues; directed oneway restrictions preserved. Not observed traffic "
            "and not a visit schedule."
        ),
        "route_semantics": {
            "network_type": "driving",
            "direction": "oneway",
            "igraph_directed": True,
            "route_mode": "OUT_BASE_TO_VENUE",
            "retain_all_before_core": True,
            "core": "largest_strongly_connected_component",
            "simplify": False,
            "time_weight": "travel_time_min",
            "distance_weight": "length_metres",
            "traffic": "static_free_flow_not_live_or_observed",
        },
        "pbf": {"path": str(pbf), "sha256": pbf_sha, "size_bytes": pbf.stat().st_size},
        "graph_cache": {
            "directory": str(graph_cache),
            **graph_hashes,
            "metadata": graph_meta,
            "vertices": int(graph.vcount()),
            "edges": int(graph.ecount()),
        },
        "inputs": {
            "confirmed_team_base_count": int(len(normalized_bases)),
            "verified_operational_venue_count": int(len(normalized_venues)),
            "base_input_sha256": base_sha,
            "venue_input_sha256": venue_sha,
        },
        "snapping": {
            "crs": "EPSG:5179",
            "max_snap_distance_m": float(max_snap_distance_m),
            "base_max_snap_distance_m": float(base_snap["snap_distance_m"].max()),
            "venue_max_snap_distance_m": float(venue_snap["snap_distance_m"].max()),
            "snap_audit_sha256": snap_sha,
        },
        "computation": {
            "team_venue_pair_count": int(len(travel)),
            "unique_base_vertex_count": int(len(base_vertices)),
            "unique_venue_vertex_count": int(len(venue_vertices)),
            "reachable_count": int(travel["reachable"].sum()),
            "unreachable_count": int((~travel["reachable"]).sum()),
            "unreachable_numeric_values_are_null": bool(
                travel.loc[~travel["reachable"], ["distance_km", "travel_minutes_proxy"]]
                .isna()
                .all()
                .all()
            ),
        },
        "travel_table_sha256": travel_sha,
        "snap_audit_sha256": snap_sha,
    }
    _atomic_write_cache(cache_path, travel, snap_audit, audit, manifest_expected)
    verified_cache = _read_cache(cache_path, manifest_expected)
    if verified_cache is None:  # pragma: no cover - defensive
        raise OperationalTravelError("atomic travel cache write did not become visible")
    return OperationalTravelResult(
        travel=travel,
        snap_audit=snap_audit,
        audit=audit,
        cache_key=cache_key,
        cache_path=cache_path,
        cache_hit=False,
    )


__all__ = [
    "DEFAULT_MAX_SNAP_DISTANCE_M",
    "GRAPH_DEFINITION",
    "OperationalTravelError",
    "OperationalTravelResult",
    "TRAVEL_RECOMPUTE_SCHEMA_VERSION",
    "recompute_operational_travel",
]
