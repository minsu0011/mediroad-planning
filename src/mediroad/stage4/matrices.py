from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from .discovery import find_matrix_row, resolve_manifest_path
from .types import CoverageMatrix, Stage3Paths
from .utils import coalesce_column, read_table, sha256_file


MATRIX_PATH_COLUMNS = ["matrix_path", "path", "file", "relative_path", "artifact_path"]
VENUE_ORDER_COLUMNS = ["venue_order_path", "row_index_path", "row_order_path", "venue_index_path"]
GRID_ORDER_COLUMNS = ["grid_order_path", "column_index_path", "col_index_path", "grid_index_path"]


def _value_from_row(row: pd.Series, candidates: list[str]) -> Any | None:
    for col in candidates:
        if col in row.index and pd.notna(row[col]) and str(row[col]).strip():
            return row[col]
    return None


def _discover_sidecar(base: Path, matrix_id: str, kind: str) -> Path | None:
    tokens = [matrix_id.lower().replace("-", "_"), base.stem.lower()]
    patterns = (
        ["*venue*order*", "*row*index*", "*venue*index*"]
        if kind == "venue"
        else ["*grid*order*", "*column*index*", "*col*index*", "*grid*index*"]
    )
    candidates: list[Path] = []
    for pattern in patterns:
        candidates.extend(base.parent.glob(pattern + ".csv"))
        candidates.extend(base.parent.glob(pattern + ".parquet"))
    if not candidates:
        return None
    scored: list[tuple[int, Path]] = []
    for path in candidates:
        name = path.stem.lower()
        score = sum(token in name for token in tokens)
        if kind == "venue":
            score += 4 * int("row_order" in name or "row_index" in name)
            score += 2 * int("venue_order" in name or "venue_index" in name)
            score -= 4 * int("column_order" in name or "col_index" in name)
        else:
            score += 4 * int("column_order" in name or "col_index" in name)
            score += 2 * int("grid_order" in name or "grid_index" in name)
            score -= 4 * int("row_order" in name or "row_index" in name)
        scored.append((score, path))
    # Prefer Parquet so string IDs such as 000000 retain leading zeroes.
    return max(
        scored,
        key=lambda x: (x[0], x[1].suffix.lower() == ".parquet", x[1].stat().st_mtime_ns),
    )[1]


def _read_order(path: Path, aliases: list[str]) -> np.ndarray:
    df = read_table(path)
    col = coalesce_column(df, aliases, required=False)
    if col is None:
        if len(df.columns) != 1:
            raise KeyError(f"Could not identify order ID column in {path}: {list(df.columns)}")
        col = str(df.columns[0])
    return df[col].astype(str).to_numpy()


def load_coverage_matrix(
    stage3: Stage3Paths,
    matrix_id: str,
    aliases: dict[str, list[str]],
) -> CoverageMatrix:
    manifest = pd.read_csv(stage3.matrix_manifest, low_memory=False)
    row = find_matrix_row(manifest, matrix_id)
    matrix_value = _value_from_row(row, MATRIX_PATH_COLUMNS)
    if matrix_value is None:
        candidates = list(stage3.run_root.rglob(f"*{matrix_id}*.npz"))
        if len(candidates) != 1:
            raise FileNotFoundError(f"No matrix path in manifest and ambiguous files for {matrix_id}")
        matrix_path = candidates[0]
    else:
        matrix_path = resolve_manifest_path(
            matrix_value, stage3.matrix_manifest, stage3.run_root, stage3.package_root
        )
    if not matrix_path.exists():
        raise FileNotFoundError(f"Coverage matrix not found: {matrix_path}")

    loaded = sparse.load_npz(matrix_path)
    matrix = loaded.tocsr()

    venue_order_value = _value_from_row(row, VENUE_ORDER_COLUMNS)
    grid_order_value = _value_from_row(row, GRID_ORDER_COLUMNS)
    venue_order_path = (
        resolve_manifest_path(
            venue_order_value, stage3.matrix_manifest, stage3.run_root, stage3.package_root
        )
        if venue_order_value is not None
        else _discover_sidecar(matrix_path, matrix_id, "venue")
    )
    grid_order_path = (
        resolve_manifest_path(
            grid_order_value, stage3.matrix_manifest, stage3.run_root, stage3.package_root
        )
        if grid_order_value is not None
        else _discover_sidecar(matrix_path, matrix_id, "grid")
    )
    if venue_order_path is None or grid_order_path is None:
        raise FileNotFoundError(
            f"Matrix {matrix_id} needs explicit venue/grid order sidecars. "
            f"matrix={matrix_path}, venue_order={venue_order_path}, grid_order={grid_order_path}"
        )

    venue_ids = _read_order(venue_order_path, aliases["venue_id"])
    grid_ids = _read_order(grid_order_path, aliases["grid_id"])
    method_value = _value_from_row(row, ["method", "matrix_method", "catchment_method"])
    radius_value = _value_from_row(
        row, ["radius_or_scale_m", "radius_or_scale", "radius_m", "scale_m", "threshold"]
    )
    radius = float(radius_value) if radius_value is not None else None
    actual_matrix_sha = sha256_file(matrix_path)
    actual_venue_sha = sha256_file(venue_order_path)
    actual_grid_sha = sha256_file(grid_order_path)
    for expected_col, actual, label in [
        ("file_sha256", actual_matrix_sha, "matrix"),
        ("matrix_sha256", actual_matrix_sha, "matrix"),
        ("venue_order_sha256", actual_venue_sha, "venue order"),
        ("grid_order_sha256", actual_grid_sha, "grid order"),
        ("row_order_sha256", actual_venue_sha, "venue order"),
        ("column_order_sha256", actual_grid_sha, "grid order"),
    ]:
        if expected_col in row.index and pd.notna(row[expected_col]) and str(row[expected_col]).strip():
            expected = str(row[expected_col]).strip().lower()
            if expected != actual.lower():
                raise RuntimeError(
                    f"Stage 3 {label} SHA mismatch for {matrix_id}: expected={expected}, actual={actual}"
                )
    metadata = row.to_dict()
    metadata.update(
        {
            "matrix_sha256": actual_matrix_sha,
            "venue_order_path": str(venue_order_path),
            "venue_order_sha256": actual_venue_sha,
            "grid_order_path": str(grid_order_path),
            "grid_order_sha256": actual_grid_sha,
        }
    )
    coverage = CoverageMatrix(
        matrix_id=matrix_id,
        matrix=matrix,
        venue_ids=venue_ids,
        grid_ids=grid_ids,
        method=str(method_value or matrix_id),
        radius_or_scale=radius,
        source_path=matrix_path,
        metadata=metadata,
    )
    coverage.validate()
    return coverage


def align_coverage(
    coverage: CoverageMatrix,
    venue_ids: pd.Series | np.ndarray,
    grid_ids: pd.Series | np.ndarray,
    *,
    binary: bool = True,
) -> sparse.csr_matrix:
    venue_ids = np.asarray(venue_ids, dtype=str)
    grid_ids = np.asarray(grid_ids, dtype=str)
    venue_pos = {value: idx for idx, value in enumerate(coverage.venue_ids.astype(str))}
    grid_pos = {value: idx for idx, value in enumerate(coverage.grid_ids.astype(str))}
    missing_v = [value for value in venue_ids if value not in venue_pos]
    missing_g = [value for value in grid_ids if value not in grid_pos]
    if missing_v:
        raise KeyError(f"Coverage matrix missing {len(missing_v)} venues, sample={missing_v[:10]}")
    if missing_g:
        raise KeyError(f"Coverage matrix missing {len(missing_g)} grids, sample={missing_g[:10]}")
    row_idx = np.fromiter((venue_pos[v] for v in venue_ids), dtype=np.int64, count=len(venue_ids))
    col_idx = np.fromiter((grid_pos[g] for g in grid_ids), dtype=np.int64, count=len(grid_ids))
    aligned = coverage.matrix[row_idx][:, col_idx].tocsr()
    if binary:
        aligned = aligned.copy()
        aligned.data = np.ones_like(aligned.data, dtype=np.int8)
        aligned.eliminate_zeros()
    return aligned


def save_coverage_matrix(
    coverage: sparse.csr_matrix,
    venue_ids: np.ndarray,
    grid_ids: np.ndarray,
    outdir: Path,
    matrix_id: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    outdir.mkdir(parents=True, exist_ok=True)
    matrix_path = outdir / f"{matrix_id}.npz"
    sparse.save_npz(matrix_path, coverage.tocsr(), compressed=True)
    venue_path = outdir / f"{matrix_id}__venue_order.csv"
    grid_path = outdir / f"{matrix_id}__grid_order.csv"
    pd.DataFrame({"venue_id": np.asarray(venue_ids, dtype=str)}).to_csv(
        venue_path, index=False, encoding="utf-8-sig"
    )
    pd.DataFrame({"grid_id": np.asarray(grid_ids, dtype=str)}).to_csv(
        grid_path, index=False, encoding="utf-8-sig"
    )
    payload = {
        "matrix_id": matrix_id,
        "matrix_path": matrix_path.name,
        "venue_order_path": venue_path.name,
        "grid_order_path": grid_path.name,
        "n_venues": int(coverage.shape[0]),
        "n_grids": int(coverage.shape[1]),
        "nnz": int(coverage.nnz),
        "density": float(coverage.nnz / max(1, coverage.shape[0] * coverage.shape[1])),
        "matrix_sha256": sha256_file(matrix_path),
        "venue_order_sha256": sha256_file(venue_path),
        "grid_order_sha256": sha256_file(grid_path),
        **metadata,
    }
    (outdir / f"{matrix_id}__metadata.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload
