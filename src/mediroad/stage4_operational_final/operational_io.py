"""CSV-only I/O and provenance for the strict Stage 4 operational contract.

Templates generated here are deliberately *not* operational evidence.  They
contain no fabricated confirmation, availability, reachability, capability, or
no-conflict claim.  The only prefilled facts are caller-supplied physical venue
IDs, the five frozen service-bundle IDs, and the SHA-256 of an existing OSM PBF.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import pandas as pd

from .operational_contract import OperationalContractError, OperationalTables, TABLE_SCHEMAS


OPERATIONAL_IO_SCHEMA_VERSION = "1.0"

DEFAULT_BUNDLE_IDS = (
    "chronic_primary",
    "comprehensive_basic_referral",
    "musculoskeletal_rehab",
    "neuro_mental",
    "sensory_oral_skin",
)

OPERATIONAL_INPUT_FILENAMES: Mapping[str, str] = {
    "bases": "mobile_team_bases.csv",
    "team_bundle_capability": "team_bundle_capability.csv",
    "vehicles": "vehicles.csv",
    "team_calendar": "team_calendar.csv",
    "resource_calendar": "resource_calendar.csv",
    "venue_calendar": "venue_calendar.csv",
    "travel": "team_venue_travel.csv",
    "existing_service_conflicts": "existing_service_conflicts.csv",
}

FROZEN_GRAPH_SOURCE_FILENAME = "FROZEN_GRAPH_SOURCE.json"
TEMPLATE_GUIDE_FILENAME = "README_FILL_OPERATIONAL_INPUTS.md"
INPUT_MANIFEST_FILENAME = "OPERATIONAL_INPUT_MANIFEST.json"


@dataclass(frozen=True)
class OperationalInputLoad:
    tables: OperationalTables
    paths: dict[str, Path]
    issues: tuple[str, ...]

    @property
    def all_files_present(self) -> bool:
        return not any(value.startswith("MISSING_FILE:") for value in self.issues)


@dataclass(frozen=True)
class OperationalInputSnapshot:
    input_dir: str
    files: dict[str, dict[str, Any]]
    frozen_graph_source: dict[str, Any]
    fingerprint_sha256: str
    complete: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": OPERATIONAL_IO_SCHEMA_VERSION,
            "input_dir": self.input_dir,
            "files": self.files,
            "frozen_graph_source": self.frozen_graph_source,
            "fingerprint_sha256": self.fingerprint_sha256,
            "complete": bool(self.complete),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "OperationalInputSnapshot":
        return cls(
            input_dir=str(value.get("input_dir", "")),
            files={str(key): dict(item) for key, item in dict(value.get("files", {})).items()},
            frozen_graph_source=dict(value.get("frozen_graph_source", {})),
            fingerprint_sha256=str(value.get("fingerprint_sha256", "")),
            complete=bool(value.get("complete", False)),
        )


@dataclass(frozen=True)
class SnapshotComparison:
    identical: bool
    differences: tuple[dict[str, Any], ...]
    before_fingerprint: str
    after_fingerprint: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "identical": bool(self.identical),
            "differences": list(self.differences),
            "before_fingerprint": self.before_fingerprint,
            "after_fingerprint": self.after_fingerprint,
        }


@dataclass(frozen=True)
class TemplatePack:
    output_dir: Path
    input_paths: dict[str, Path]
    frozen_graph_source_path: Path
    guide_path: Path
    manifest_path: Path
    graph_sha256: str
    physical_venue_count: int


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(text, encoding=encoding)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_json(path: Path, value: Any) -> None:
    _atomic_write_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
    )


def _atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        frame.to_csv(temporary, index=False, encoding="utf-8-sig")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _path_record(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    resolved = path.resolve()
    exists = resolved.is_file()
    relative_path: str | None = None
    if relative_to is not None:
        try:
            relative_path = resolved.relative_to(relative_to.resolve()).as_posix()
        except ValueError:
            relative_path = None
    return {
        "path": str(resolved),
        "relative_path": relative_path,
        "exists": bool(exists),
        "size_bytes": int(resolved.stat().st_size) if exists else None,
        "sha256": _sha256_file(resolved) if exists else None,
    }


def _validate_physical_venue_ids(values: Any) -> list[str]:
    if isinstance(values, (str, bytes)):
        raise OperationalContractError("physical_venue_ids must be an iterable of IDs, not text")
    try:
        venue_ids = [str(value).strip() for value in values]
    except TypeError as exc:
        raise OperationalContractError("physical_venue_ids must be iterable") from exc
    if not venue_ids:
        raise OperationalContractError("At least one physical venue ID is required")
    if any(not value for value in venue_ids):
        raise OperationalContractError("Physical venue IDs cannot be blank")
    if len(venue_ids) != len(set(venue_ids)):
        raise OperationalContractError("Physical venue IDs must be unique")
    forbidden = [value for value in venue_ids if value.upper().startswith("BUSPROXY_")]
    if forbidden:
        raise OperationalContractError(
            f"BUSPROXY spatial anchors are not physical venue IDs: {forbidden[:5]}"
        )
    placeholders = [
        value
        for value in venue_ids
        if value.upper()
        in {"UNKNOWN", "REPLACE", "REPLACE_VENUE_ID", "N/A", "NA", "NAN", "NONE", "NULL"}
    ]
    if placeholders:
        raise OperationalContractError(f"Placeholder physical venue IDs are forbidden: {placeholders}")
    return venue_ids


def _template_frames(venue_ids: list[str], graph_sha256: str) -> dict[str, pd.DataFrame]:
    frames = {
        name: pd.DataFrame(columns=list(columns)) for name, columns in TABLE_SCHEMAS.items()
    }
    frames["team_bundle_capability"] = pd.DataFrame(
        [
            {
                "team_id": "",
                "bundle_id": bundle_id,
                "capable": "UNKNOWN",
                "confirmed": False,
                "required_vehicle_type": "UNKNOWN",
                "minimum_vehicle_capacity": "",
                "source_reference": "",
                "effective_date": "",
            }
            for bundle_id in DEFAULT_BUNDLE_IDS
        ],
        columns=list(TABLE_SCHEMAS["team_bundle_capability"]),
    )
    frames["venue_calendar"] = pd.DataFrame(
        [
            {
                "venue_id": venue_id,
                "date": "UNKNOWN",
                "available": "UNKNOWN",
                "confirmed": False,
                "source_reference": "",
            }
            for venue_id in venue_ids
        ],
        columns=list(TABLE_SCHEMAS["venue_calendar"]),
    )
    frames["travel"] = pd.DataFrame(
        [
            {
                "team_id": "",
                "venue_id": venue_id,
                "reachable": "UNKNOWN",
                "distance_km": "",
                "travel_minutes_proxy": "",
                "graph_sha": graph_sha256,
                "confirmed": False,
                "source_reference": "",
                "effective_date": "",
            }
            for venue_id in venue_ids
        ],
        columns=list(TABLE_SCHEMAS["travel"]),
    )
    frames["existing_service_conflicts"] = pd.DataFrame(
        [
            {
                "team_id": "",
                "vehicle_id": "",
                "venue_id": venue_id,
                "date": "UNKNOWN",
                "conflict": "UNKNOWN",
                "confirmed": False,
                "source_reference": "",
            }
            for venue_id in venue_ids
        ],
        columns=list(TABLE_SCHEMAS["existing_service_conflicts"]),
    )
    return frames


def _guide_text(venue_count: int, graph_sha256: str) -> str:
    bundle_lines = "\n".join(f"- `{bundle_id}`" for bundle_id in DEFAULT_BUNDLE_IDS)
    file_lines = "\n".join(
        f"- `{filename}`: `{key}` contract" for key, filename in OPERATIONAL_INPUT_FILENAMES.items()
    )
    return f"""# MEDIROAD Stage 4 Operational Input Pack

이 디렉터리는 사용자 입력용 템플릿이다. 현재 행은 확인된 운영 사실이 아니며,
`confirmed=false` 또는 `UNKNOWN`을 실제 값으로 승격하면 안 된다.

## 고정 bundle IDs

{bundle_lines}

## 정확한 입력 파일명

{file_lines}

## 입력 원칙

- `confirmed=true`는 비어 있지 않은 `source_reference`가 있는 실제 확인행에만 쓴다.
- availability, capability, reachability, no-conflict는 부재를 `true`로 해석하지 않는다.
- `existing_service_conflicts.csv`는 각 team×vehicle×venue×date에 대해 확인된
  `conflict=false` 행이 있어야 한다. 행 부재는 no-conflict가 아니다.
- OSM 시간은 정적 car-profile free-flow proxy이며 실시간 교통시간이 아니다.
- physical venue rows: `{venue_count}`
- frozen OSM PBF SHA-256: `{graph_sha256}`
"""


def write_operational_template_pack(
    output_dir: str | Path,
    physical_venue_ids: Any,
    *,
    osm_pbf_path: str | Path,
) -> TemplatePack:
    """Create a non-promotable, CSV-only user input pack.

    Existing target files are never overwritten.  The OSM PBF must already
    exist; its bytes are not copied into the pack.
    """

    destination = Path(output_dir).resolve()
    venue_ids = _validate_physical_venue_ids(physical_venue_ids)
    pbf = Path(osm_pbf_path).resolve()
    if not pbf.is_file():
        raise OperationalContractError(f"OSM PBF is missing: {pbf}")
    if pbf.suffix.lower() != ".pbf" or pbf.stat().st_size <= 0:
        raise OperationalContractError(f"OSM PBF must be a nonempty .pbf file: {pbf}")
    graph_record = _path_record(pbf)
    graph_sha256 = str(graph_record["sha256"])

    input_paths = {
        key: destination / filename for key, filename in OPERATIONAL_INPUT_FILENAMES.items()
    }
    frozen_graph_path = destination / FROZEN_GRAPH_SOURCE_FILENAME
    guide_path = destination / TEMPLATE_GUIDE_FILENAME
    manifest_path = destination / INPUT_MANIFEST_FILENAME
    targets = list(input_paths.values()) + [frozen_graph_path, guide_path, manifest_path]
    existing = [str(path) for path in targets if path.exists()]
    if existing:
        raise OperationalContractError(
            "Refusing to overwrite operational input pack files: " + ", ".join(existing)
        )
    destination.mkdir(parents=True, exist_ok=True)

    for key, frame in _template_frames(venue_ids, graph_sha256).items():
        _atomic_write_csv(input_paths[key], frame)

    frozen_graph = {
        "schema_version": OPERATIONAL_IO_SCHEMA_VERSION,
        "source_kind": "OSM_PBF_STATIC_CAR_PROFILE_FREE_FLOW",
        "graph_sha_semantics": "SHA256_OF_SOURCE_PBF_BYTES",
        **graph_record,
        "operational_team_base_confirmed": False,
        "live_traffic": False,
    }
    _atomic_write_json(frozen_graph_path, frozen_graph)
    _atomic_write_text(guide_path, _guide_text(len(venue_ids), graph_sha256))

    snapshot = capture_operational_input_snapshot(destination, osm_pbf_path=pbf)
    payload_records = {
        path.name: _path_record(path, relative_to=destination)
        for path in [*input_paths.values(), frozen_graph_path, guide_path]
    }
    manifest = {
        "schema_version": OPERATIONAL_IO_SCHEMA_VERSION,
        "manifest_kind": "OPERATIONAL_INPUT_TEMPLATE_PACK",
        "created_at_utc": _utc_now(),
        "self_excluded_from_payload_hashes": True,
        "template_is_operational_evidence": False,
        "default_bundle_ids": list(DEFAULT_BUNDLE_IDS),
        "physical_venue_count": len(venue_ids),
        "payload_files": payload_records,
        "input_snapshot": snapshot.as_dict(),
    }
    _atomic_write_json(manifest_path, manifest)
    return TemplatePack(
        output_dir=destination,
        input_paths=input_paths,
        frozen_graph_source_path=frozen_graph_path,
        guide_path=guide_path,
        manifest_path=manifest_path,
        graph_sha256=graph_sha256,
        physical_venue_count=len(venue_ids),
    )


def _read_exact_csv(path: Path, key: str) -> tuple[pd.DataFrame, list[str]]:
    issues: list[str] = []
    expected = list(TABLE_SCHEMAS[key])
    if not path.is_file():
        return pd.DataFrame(columns=expected), [f"MISSING_FILE:{key}:{path}"]
    if path.stat().st_size == 0:
        return pd.DataFrame(columns=expected), [f"ZERO_BYTE_FILE:{key}:{path}"]
    try:
        frame = pd.read_csv(path, low_memory=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=expected), [f"EMPTY_FILE:{key}:{path}"]
    except Exception as exc:
        return pd.DataFrame(columns=expected), [f"UNREADABLE_FILE:{key}:{type(exc).__name__}:{path}"]
    missing = [column for column in expected if column not in frame.columns]
    extra = [column for column in frame.columns if column not in expected]
    if missing:
        issues.append(f"MISSING_COLUMNS:{key}:{','.join(missing)}")
    if extra:
        issues.append(f"UNEXPECTED_COLUMNS:{key}:{','.join(map(str, extra))}")
    # Preserve malformed schemas so the strict contract can independently
    # report them.  Correct/header-only schemas retain canonical column order.
    if not missing:
        frame = frame.loc[:, expected].copy()
    return frame, issues


def load_operational_input_dir(input_dir: str | Path) -> OperationalInputLoad:
    """Load only the eight exact filenames without promoting placeholders."""

    directory = Path(input_dir).resolve()
    paths = {
        key: directory / filename for key, filename in OPERATIONAL_INPUT_FILENAMES.items()
    }
    frames: dict[str, pd.DataFrame] = {}
    issues: list[str] = []
    for key, path in paths.items():
        frames[key], found = _read_exact_csv(path, key)
        issues.extend(found)
    return OperationalInputLoad(
        tables=OperationalTables.from_mapping(frames),
        paths=paths,
        issues=tuple(issues),
    )


def load_operational_tables(input_dir: str | Path) -> OperationalTables:
    """Convenience wrapper retaining empty/header-only tables for audit."""

    return load_operational_input_dir(input_dir).tables


def capture_operational_input_snapshot(
    input_dir: str | Path,
    *,
    osm_pbf_path: str | Path,
) -> OperationalInputSnapshot:
    """Capture exact paths, byte sizes and SHA-256 before or after a run."""

    directory = Path(input_dir).resolve()
    files = {
        key: _path_record(directory / filename, relative_to=directory)
        for key, filename in OPERATIONAL_INPUT_FILENAMES.items()
    }
    graph = _path_record(Path(osm_pbf_path).resolve())
    graph.update(
        {
            "source_kind": "OSM_PBF_STATIC_CAR_PROFILE_FREE_FLOW",
            "graph_sha_semantics": "SHA256_OF_SOURCE_PBF_BYTES",
        }
    )
    fingerprint_payload = {
        "input_dir": str(directory),
        "files": files,
        "frozen_graph_source": graph,
    }
    complete = all(record["exists"] for record in files.values()) and bool(graph["exists"])
    return OperationalInputSnapshot(
        input_dir=str(directory),
        files=files,
        frozen_graph_source=graph,
        fingerprint_sha256=_sha256_json(fingerprint_payload),
        complete=complete,
    )


def compare_operational_input_snapshots(
    before: OperationalInputSnapshot,
    after: OperationalInputSnapshot,
) -> SnapshotComparison:
    """Compare path/existence/size/SHA evidence, including the source PBF."""

    differences: list[dict[str, Any]] = []
    all_keys = sorted(set(before.files) | set(after.files))
    for key in all_keys:
        left = before.files.get(key)
        right = after.files.get(key)
        if left != right:
            differences.append(
                {"kind": "INPUT_FILE_CHANGED", "key": key, "before": left, "after": right}
            )
    if before.input_dir != after.input_dir:
        differences.append(
            {
                "kind": "INPUT_DIRECTORY_CHANGED",
                "before": before.input_dir,
                "after": after.input_dir,
            }
        )
    if before.complete != after.complete:
        differences.append(
            {
                "kind": "SNAPSHOT_COMPLETENESS_CHANGED",
                "before": before.complete,
                "after": after.complete,
            }
        )
    if before.frozen_graph_source != after.frozen_graph_source:
        differences.append(
            {
                "kind": "FROZEN_GRAPH_SOURCE_CHANGED",
                "before": before.frozen_graph_source,
                "after": after.frozen_graph_source,
            }
        )
    return SnapshotComparison(
        identical=not differences and before.fingerprint_sha256 == after.fingerprint_sha256,
        differences=tuple(differences),
        before_fingerprint=before.fingerprint_sha256,
        after_fingerprint=after.fingerprint_sha256,
    )


def assert_operational_inputs_unchanged(
    before: OperationalInputSnapshot,
    after: OperationalInputSnapshot,
) -> SnapshotComparison:
    comparison = compare_operational_input_snapshots(before, after)
    if not comparison.identical:
        kinds = ",".join(str(item["kind"]) for item in comparison.differences)
        raise OperationalContractError(f"Operational inputs changed during execution: {kinds}")
    return comparison


def write_operational_input_manifest(
    path: str | Path,
    snapshot: OperationalInputSnapshot,
    *,
    manifest_kind: str = "OPERATIONAL_INPUT_EXECUTION_SNAPSHOT",
) -> Path:
    """Persist a snapshot manifest; the manifest never hashes itself."""

    target = Path(path).resolve()
    payload = {
        "schema_version": OPERATIONAL_IO_SCHEMA_VERSION,
        "manifest_kind": manifest_kind,
        "created_at_utc": _utc_now(),
        "self_excluded_from_hash": True,
        "input_snapshot": snapshot.as_dict(),
    }
    _atomic_write_json(target, payload)
    return target


def verify_operational_input_manifest(
    manifest_path: str | Path,
    *,
    input_dir: str | Path,
    osm_pbf_path: str | Path,
) -> SnapshotComparison:
    """Recompute every input/PBF hash and compare it with a saved manifest."""

    path = Path(manifest_path).resolve()
    if not path.is_file():
        raise OperationalContractError(f"Operational input manifest is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise OperationalContractError(f"Operational input manifest is unreadable: {path}") from exc
    snapshot_value = payload.get("input_snapshot") if isinstance(payload, dict) else None
    if not isinstance(snapshot_value, dict):
        raise OperationalContractError(f"Manifest has no input_snapshot object: {path}")
    expected = OperationalInputSnapshot.from_dict(snapshot_value)
    observed = capture_operational_input_snapshot(input_dir, osm_pbf_path=osm_pbf_path)
    return compare_operational_input_snapshots(expected, observed)
