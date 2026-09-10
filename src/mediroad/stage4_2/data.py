from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable
import hashlib

import numpy as np
import pandas as pd
from scipy import sparse

from .errors import ContractError
from .types import CoverageData, Stage3Artifacts
from .utils import ensure_columns, read_table, sha256_file


def first_existing_column(frame: pd.DataFrame, aliases: Iterable[str], *, label: str) -> str:
    for alias in aliases:
        if alias in frame.columns:
            return alias
    raise ContractError(f"Could not identify {label}; aliases={list(aliases)}, columns={list(frame.columns)}")


def normalize_stage3_interface(frame: pd.DataFrame) -> pd.DataFrame:
    aliases = {
        "venue_id": ["venue_id"],
        "bundle_id": ["bundle_id", "service_bundle_id"],
        "admin_code": ["admin_code", "admin_dong_code"],
        "sigungu": ["sigungu", "policy_sigungu", "policy_sigungu_name"],
        "cluster_id": ["coverage_cluster_id", "cluster_id"],
    }
    rename: dict[str, str] = {}
    for target, names in aliases.items():
        try:
            source = first_existing_column(frame, names, label=target)
        except ContractError:
            if target == "bundle_id":
                continue
            raise
        if source != target:
            rename[source] = target
    out = frame.rename(columns=rename).copy()
    out["venue_id"] = out["venue_id"].astype(str)
    out["admin_code"] = out["admin_code"].astype(str)
    out["sigungu"] = out["sigungu"].astype(str)
    if "cluster_id" not in out:
        out["cluster_id"] = out["venue_id"]
    out["cluster_id"] = out["cluster_id"].fillna(out["venue_id"]).astype(str)
    return out


def venue_table_from_interface(frame: pd.DataFrame) -> pd.DataFrame:
    work = normalize_stage3_interface(frame)
    venue_cols = [
        "venue_id",
        "venue_name",
        "venue_type",
        "admin_code",
        "sigungu",
        "cluster_id",
        "shortlist_rank",
        "pareto_tier",
        "known_overlap",
        "latitude",
        "longitude",
        "field_validation_required",
        "venue_readiness_prior",
        "transit_context",
        "transit_context_score",
        "road_context",
        "raw_elderly_exposure",
        "need_weighted_exposure",
        "high_need_elderly_exposure",
        "own_admin_exposure",
        "cross_admin_exposure",
        "cross_sigungu_exposure",
        "structural_need",
        "candidate_priority_tiebreak",
        "overlap_weighted_degree",
        "fallback_venue_1",
        "fallback_venue_2",
        "temporal_release_type",
        "temporal_confidence",
        "recommended_season",
        "fallback_season",
    ]
    venue_cols = [c for c in venue_cols if c in work.columns]
    venue = work[venue_cols].drop_duplicates("venue_id", keep="first").copy()
    for col in ["shortlist_rank", "pareto_tier", "candidate_priority_tiebreak"]:
        if col not in venue:
            venue[col] = np.nan
    for col in ["known_overlap", "venue_type", "venue_name"]:
        if col not in venue:
            venue[col] = "UNKNOWN"
    return venue.reset_index(drop=True)


def bundle_table_from_interface(frame: pd.DataFrame) -> pd.DataFrame:
    work = normalize_stage3_interface(frame)
    bundle_col = first_existing_column(work, ["bundle_id", "service_bundle_id"], label="bundle ID")
    value_col = first_existing_column(
        work,
        [
            "bundle_gap_weighted_exposure",
            "mean_bundle_gap_weighted_exposure",
            "specialty_gap_weighted_exposure",
            "specialty_gap",
        ],
        label="bundle value",
    )
    out = work[["venue_id", bundle_col, value_col]].copy()
    out.columns = ["venue_id", "bundle_id", "bundle_value"]
    out["venue_id"] = out["venue_id"].astype(str)
    out["bundle_id"] = out["bundle_id"].astype(str)
    out["bundle_value"] = pd.to_numeric(out["bundle_value"], errors="coerce").fillna(0.0)
    if out.duplicated(["venue_id", "bundle_id"]).any():
        raise ContractError("Duplicate venue_id × bundle_id rows in Stage 3 interface")
    return out


def normalize_grid_policy(frame: pd.DataFrame) -> pd.DataFrame:
    aliases = {
        # Stage 3's authoritative matrix column ID is the zero-padded
        # ``matrix_grid_id``.  Prefer it over the numeric point index so that
        # leading zeroes and the sealed matrix order survive round-trips.
        "grid_id": ["matrix_grid_id", "grid_id", "grid_point_index", "point_index", "gid"],
        "elderly_population": [
            "elderly65_calibrated_population",
            "elderly65_population",
            "elderly65_calibrated",
            "calibrated_elderly65",
            "elderly_population",
            "population65",
        ],
        "need_score": ["stage1_need_score", "need_score", "need_score_primary", "structural_need"],
        "sigungu": ["sigungu", "policy_sigungu", "policy_sigungu_name"],
        "admin_code": ["admin_code", "admin_dong_code"],
    }
    rename: dict[str, str] = {}
    for target, names in aliases.items():
        source = first_existing_column(frame, names, label=target)
        if source != target:
            rename[source] = target
    out = frame.rename(columns=rename).copy()
    out["grid_id"] = out["grid_id"].astype(str)
    out["sigungu"] = out["sigungu"].astype(str)
    out["admin_code"] = out["admin_code"].astype(str)
    out["elderly_population"] = pd.to_numeric(out["elderly_population"], errors="coerce")
    out["need_score"] = pd.to_numeric(out["need_score"], errors="coerce")
    if out[["elderly_population", "need_score"]].isna().any().any():
        raise ContractError("Grid policy contains missing elderly population or Need score")
    if (out["elderly_population"] < 0).any():
        raise ContractError("Grid policy contains negative elderly population")
    if out["need_score"].max() > 1.000001:
        out["need_norm"] = (out["need_score"] / 100.0).clip(0, 1)
    else:
        out["need_norm"] = out["need_score"].clip(0, 1)
    threshold = float(out["need_norm"].quantile(0.80))
    out["need_weighted_population"] = out["elderly_population"] * out["need_norm"]
    out["high_need_population"] = out["elderly_population"] * (out["need_norm"] >= threshold)
    if out["grid_id"].duplicated().any():
        raise ContractError("Duplicate grid_id in grid policy table")
    return out


def _resolve_path(value: Any, manifest: Path, run_root: Path, project_root: Path | None = None) -> Path:
    p = Path(str(value))
    candidates = [p]
    if not p.is_absolute():
        if project_root is not None:
            candidates.append(project_root / p)
        candidates.extend([manifest.parent / p, run_root / p])
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Could not resolve manifest path {value!r} from {manifest}")


def _read_order(path: Path, aliases: list[str]) -> np.ndarray:
    # CSV inference would turn ``000000`` into ``0``.  Stage 3 explicitly
    # seals these IDs as strings, so force a string read at this boundary.
    frame = pd.read_csv(path, dtype=str, low_memory=False) if path.suffix.lower() == ".csv" else read_table(path)
    col = next((c for c in aliases if c in frame.columns), None)
    if col is None and len(frame.columns) == 1:
        col = str(frame.columns[0])
    if col is None:
        raise ContractError(f"Could not identify order column in {path}: {list(frame.columns)}")
    return frame[col].astype(str).to_numpy()


def _order_sha256(values: Iterable[Any]) -> str:
    """Reproduce the MEDIROAD Stage 3 order seal exactly."""

    text = pd.Series(list(values), dtype="string")
    if text.isna().any() or text.duplicated().any():
        raise ContractError("Matrix order IDs must be non-null and unique")
    digest = hashlib.sha256(b"MEDIROAD_STAGE3_ORDER_V1\0")
    for value in text.astype(str):
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="little", signed=False))
        digest.update(encoded)
    return digest.hexdigest()


def load_coverage(stage3: Stage3Artifacts, matrix_id: str = "hard_5000m") -> CoverageData:
    manifest = pd.read_csv(stage3.matrix_manifest, low_memory=False)
    id_col = next((c for c in ["matrix_id", "id", "name"] if c in manifest), None)
    if id_col is None:
        raise ContractError(f"Matrix manifest missing matrix ID: {stage3.matrix_manifest}")
    normalized = manifest[id_col].astype(str).str.lower().str.replace("-", "_", regex=False)
    target = matrix_id.lower().replace("-", "_")
    rows = manifest.loc[normalized.eq(target)]
    if rows.empty:
        rows = manifest.loc[normalized.str.contains(target, regex=False)]
    if len(rows) != 1:
        raise ContractError(f"Expected exactly one matrix manifest row for {matrix_id}; got {len(rows)}")
    row = rows.iloc[0]
    matrix_col = next((c for c in ["matrix_path", "path", "file", "relative_path"] if c in row.index), None)
    if matrix_col is None or pd.isna(row[matrix_col]):
        files = list(stage3.run_root.rglob(f"*{matrix_id}*.npz"))
        if len(files) != 1:
            raise ContractError(f"Ambiguous matrix files for {matrix_id}: {files}")
        matrix_path = files[0]
    else:
        project_root = stage3.pointer.resolve().parents[3]
        matrix_path = _resolve_path(row[matrix_col], stage3.matrix_manifest, stage3.run_root, project_root)
    matrix = sparse.load_npz(matrix_path).tocsr()

    def sidecar(cols: list[str], canonical_names: list[str], patterns: list[str], label: str) -> Path:
        col = next((c for c in cols if c in row.index and pd.notna(row[c])), None)
        if col:
            return _resolve_path(row[col], stage3.matrix_manifest, stage3.run_root, project_root)
        for name in canonical_names:
            candidate = matrix_path.parent / name
            if candidate.is_file():
                return candidate
        found: list[Path] = []
        for pattern in patterns:
            found.extend(matrix_path.parent.glob(pattern))
        found = [p for p in found if p.is_file()]
        if len(found) != 1:
            raise ContractError(f"Could not uniquely locate {label} for {matrix_id}: {found}")
        return found[0]

    venue_order_path = sidecar(
        ["venue_order_path", "row_index_path", "row_order_path", "venue_index_path"],
        ["venue_grid_row_order.parquet", "venue_grid_row_order.csv"],
        [f"*{matrix_id}*venue*order*.parquet", f"*{matrix_id}*venue*order*.csv", "*row*order*.parquet", "*row*order*.csv"],
        "venue order",
    )
    grid_order_path = sidecar(
        ["grid_order_path", "column_index_path", "col_index_path", "grid_index_path"],
        ["venue_grid_column_order.parquet", "venue_grid_column_order.csv"],
        [f"*{matrix_id}*grid*order*.parquet", f"*{matrix_id}*grid*order*.csv", "*column*order*.parquet", "*column*order*.csv"],
        "grid order",
    )
    venue_ids = _read_order(venue_order_path, ["venue_id", "id"])
    grid_ids = _read_order(grid_order_path, ["matrix_grid_id", "grid_id", "grid_point_index", "point_index", "id"])
    cov = CoverageData(
        matrix_id=matrix_id,
        matrix=matrix,
        venue_ids=venue_ids,
        grid_ids=grid_ids,
        venue_order_path=venue_order_path,
        grid_order_path=grid_order_path,
        matrix_path=matrix_path,
        metadata=row.to_dict(),
    )
    cov.validate()
    for expected_col in ["file_sha256", "matrix_sha256"]:
        if expected_col in row.index and pd.notna(row[expected_col]) and str(row[expected_col]).strip():
            expected = str(row[expected_col]).strip().lower()
            actual = sha256_file(matrix_path).lower()
            if expected != actual:
                raise ContractError(f"{expected_col} mismatch: expected={expected}, actual={actual}")
    for expected_col, actual in [
        ("venue_order_sha", _order_sha256(venue_ids)),
        ("venue_order_id_sha256", _order_sha256(venue_ids)),
        ("grid_order_sha", _order_sha256(grid_ids)),
        ("grid_order_id_sha256", _order_sha256(grid_ids)),
    ]:
        if expected_col in row.index and pd.notna(row[expected_col]) and str(row[expected_col]).strip():
            expected = str(row[expected_col]).strip().lower()
            if expected != actual:
                raise ContractError(f"{expected_col} mismatch: expected={expected}, actual={actual}")
    # Legacy Stage 4/synthetic manifests use these names for sidecar file
    # hashes (the official Stage 3 manifest uses *_order_sha for ID-order
    # seals).  Keep both contracts explicit instead of guessing.
    for expected_col, path in [
        ("venue_order_sha256", venue_order_path),
        ("row_order_sha256", venue_order_path),
        ("grid_order_sha256", grid_order_path),
        ("column_order_sha256", grid_order_path),
    ]:
        if expected_col in row.index and pd.notna(row[expected_col]) and str(row[expected_col]).strip():
            expected = str(row[expected_col]).strip().lower()
            actual = sha256_file(path).lower()
            if expected != actual:
                raise ContractError(f"{expected_col} mismatch: expected={expected}, actual={actual}")
    for expected_col, actual in [
        ("n_venues", matrix.shape[0]),
        ("n_grids", matrix.shape[1]),
        ("nnz", matrix.nnz),
    ]:
        if expected_col in row.index and pd.notna(row[expected_col]) and int(row[expected_col]) != int(actual):
            raise ContractError(f"{expected_col} mismatch: expected={row[expected_col]}, actual={actual}")
    return cov


def align_coverage(coverage: CoverageData, venue_ids: list[str] | np.ndarray, grid_ids: list[str] | np.ndarray) -> sparse.csr_matrix:
    venue_ids = np.asarray(venue_ids, dtype=str)
    grid_ids = np.asarray(grid_ids, dtype=str)
    vpos = {v: i for i, v in enumerate(coverage.venue_ids.astype(str))}
    gpos = {g: i for i, g in enumerate(coverage.grid_ids.astype(str))}
    missing_v = [v for v in venue_ids if v not in vpos]
    missing_g = [g for g in grid_ids if g not in gpos]
    if missing_v:
        raise ContractError(f"Coverage matrix missing venues: {missing_v[:10]} ({len(missing_v)})")
    if missing_g:
        raise ContractError(f"Coverage matrix missing grids: {missing_g[:10]} ({len(missing_g)})")
    rows = np.fromiter((vpos[v] for v in venue_ids), dtype=np.int64, count=len(venue_ids))
    cols = np.fromiter((gpos[g] for g in grid_ids), dtype=np.int64, count=len(grid_ids))
    matrix = coverage.matrix[rows][:, cols].tocsr().copy()
    matrix.data = np.ones_like(matrix.data, dtype=np.int8)
    matrix.eliminate_zeros()
    return matrix
