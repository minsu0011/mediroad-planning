from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from .types import Stage3Paths


def _latest(paths: Iterable[Path]) -> Path | None:
    existing = [p for p in paths if p.exists()]
    if not existing:
        return None
    return max(existing, key=lambda p: (p.stat().st_mtime_ns, p.as_posix()))


def _glob_latest(root: Path, patterns: list[str]) -> Path | None:
    found: list[Path] = []
    for pattern in patterns:
        found.extend(root.glob(pattern))
    return _latest(found)


def _read_current_pointer(stage3_root: Path) -> Path | None:
    candidates = [stage3_root / "CURRENT", stage3_root / "CURRENT.txt", stage3_root / "CURRENT_RUN.txt"]
    for path in candidates:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        if not text:
            continue
        possible = Path(text)
        if possible.is_absolute() and possible.exists():
            return possible.resolve()
        for candidate in [stage3_root / possible, stage3_root / "runs" / possible]:
            if candidate.exists():
                return candidate.resolve()
    current_jsons = [stage3_root / "CURRENT_STAGE3_RUN.json", stage3_root / "CURRENT.json"]
    for current_json in current_jsons:
        if not current_json.exists():
            continue
        payload = json.loads(current_json.read_text(encoding="utf-8"))
        metadata_relative = payload.get("metadata_relative_path")
        if metadata_relative:
            # Stage 3 records the authoritative metadata relative to the project root.
            root_candidate = stage3_root
            while root_candidate.parent != root_candidate and not (root_candidate / "outputs").exists():
                root_candidate = root_candidate.parent
            metadata_path = root_candidate / str(metadata_relative)
            if metadata_path.exists():
                return metadata_path.parent.resolve()
        for key in ("run_root", "run_path", "current_run", "run_id"):
            value = payload.get(key)
            if not value:
                continue
            possible = Path(str(value))
            if not possible.is_absolute():
                possible = stage3_root / ("runs" if key == "run_id" else "") / possible
            if possible.exists():
                return possible.resolve()
    return None


def discover_stage3_paths(package_root: Path, config: dict) -> Stage3Paths:
    stage3_root = package_root / config["paths"]["stage3_root"]
    if not stage3_root.exists():
        raise FileNotFoundError(
            f"Stage 3 output root not found: {stage3_root}. Run/freeze Stage 3 before Stage 4."
        )

    run_root = _read_current_pointer(stage3_root)
    if run_root is None:
        run_root = _latest([p for p in (stage3_root / "runs").glob("stage3_*") if p.is_dir()])
    if run_root is None:
        run_root = stage3_root

    search_roots = [run_root, stage3_root]

    def find(patterns: list[str], required: bool = True) -> Path | None:
        for root in search_roots:
            path = _glob_latest(root, patterns)
            if path is not None:
                return path
        if required:
            raise FileNotFoundError(
                f"Could not discover Stage 3 artifact. Patterns={patterns}, roots={search_roots}"
            )
        return None

    interface = find([
        "**/stage3_to_stage4_interface.parquet",
        "**/stage3_to_stage4_interface.csv",
    ])
    grid_policy = find([
        "**/grid_policy_weights.parquet",
        "**/grid_policy_weights.csv",
        "**/grid_policy*.parquet",
    ])
    matrix_manifest = find([
        "**/venue_grid_matrix_manifest.csv",
        "**/*matrix*manifest*.csv",
    ])
    shortlist = find(["**/venue_shortlist.csv", "**/*shortlist*.parquet"], required=False)
    overlap = find(["**/venue_overlap_edges.parquet", "**/venue_overlap_edges.csv"], required=False)
    clusters = find([
        "**/venue_coverage_clusters.csv",
        "**/coverage_equivalent_clusters.csv",
        "**/*coverage*cluster*.parquet",
    ], required=False)
    qgate = find(["**/STAGE3_QUALITY_GATE.csv"], required=False)
    metadata = find(["**/STAGE3_RUN_METADATA.json"], required=False)
    inventory = find(["**/STAGE3_ARTIFACT_INVENTORY.csv"], required=False)

    return Stage3Paths(
        run_root=run_root,
        interface=interface,
        grid_policy=grid_policy,
        matrix_manifest=matrix_manifest,
        shortlist=shortlist,
        venue_overlap=overlap,
        coverage_clusters=clusters,
        quality_gate=qgate,
        metadata=metadata,
        inventory=inventory,
        package_root=package_root,
    )


def resolve_manifest_path(
    value: str | Path,
    manifest_path: Path,
    run_root: Path,
    package_root: Path | None = None,
) -> Path:
    path = Path(str(value))
    if path.is_absolute() and path.exists():
        return path
    bases = [manifest_path.parent, run_root, run_root.parent]
    if package_root is not None:
        bases.insert(0, package_root)
    for base in bases:
        candidate = (base / path).resolve()
        if candidate.exists():
            return candidate
    return (manifest_path.parent / path).resolve()


def find_matrix_row(manifest: pd.DataFrame, matrix_id: str) -> pd.Series:
    id_candidates = [c for c in ["matrix_id", "id", "name", "method"] if c in manifest.columns]
    if not id_candidates:
        raise KeyError("Matrix manifest needs one of matrix_id/id/name/method")
    matrix_id_lower = matrix_id.lower()
    for col in id_candidates:
        exact = manifest[manifest[col].astype(str).str.lower().eq(matrix_id_lower)]
        if len(exact) == 1:
            return exact.iloc[0]
    for col in id_candidates:
        partial = manifest[manifest[col].astype(str).str.lower().str.contains(matrix_id_lower, regex=False)]
        if len(partial) == 1:
            return partial.iloc[0]
    available = sorted(set(manifest[id_candidates[0]].astype(str)))
    raise KeyError(f"Matrix '{matrix_id}' not uniquely found. Available sample: {available[:30]}")
