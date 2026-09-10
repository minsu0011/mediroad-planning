"""Deterministic sparse spatial engine for Stage 3.

Stage 3 links candidate operating venues to calibrated 100 m elderly-grid
mass.  A grid row is an allocation unit, not an independent observation, and
all values produced here are *potential geographic exposure*, not patients,
attendance, demand, or operational feasibility.

The central safety contract is that a venue-by-grid dense matrix is never
accepted or created.  Neighbourhoods are discovered with :class:`cKDTree` and
stored as SciPy CSR matrices with sealed row/column orders.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import cKDTree


Method = Literal["hard", "exponential", "gaussian"]
_CRS_5179 = "EPSG:5179"


def _normalise_text(series: pd.Series, column: str) -> pd.Series:
    if series.isna().any():
        raise ValueError(f"{column} contains missing values")
    values = series.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    if values.eq("").any():
        raise ValueError(f"{column} contains blank values")
    return values


def _finite_numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        raise KeyError(f"Required spatial column is missing: {column}")
    values = pd.to_numeric(frame[column], errors="coerce").astype(float)
    array = values.to_numpy()
    if values.isna().any() or not np.isfinite(array).all():
        raise ValueError(f"{column} must contain only finite numeric values")
    return values


def _canonical_crs(crs: str | int) -> str:
    text = str(crs).strip().upper().replace(" ", "")
    if text == "5179":
        text = _CRS_5179
    if text != _CRS_5179:
        raise ValueError(f"Stage 3 metric coordinates require EPSG:5179, received {crs!r}")
    return _CRS_5179


def _validate_expected_admins(
    values: pd.Series,
    *,
    expected_admin_codes: Sequence[object] | None,
    expected_admin_count: int | None,
) -> None:
    observed = set(values.astype(str))
    if expected_admin_count is not None:
        count = int(expected_admin_count)
        if count <= 0:
            raise ValueError("expected_admin_count must be positive")
        if len(observed) != count:
            raise ValueError(f"Expected {count} admin codes, found {len(observed)}")
    if expected_admin_codes is not None:
        expected_series = _normalise_text(pd.Series(list(expected_admin_codes)), "expected_admin_codes")
        if expected_series.duplicated().any():
            raise ValueError("expected_admin_codes contains duplicates")
        expected = set(expected_series.astype(str))
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        if missing or extra:
            raise ValueError(
                "Admin-code coverage mismatch; "
                f"missing={missing[:10]}, extra={extra[:10]}"
            )


def _validate_bounds(
    x: np.ndarray,
    y: np.ndarray,
    bounds: tuple[float, float, float, float] | None,
) -> None:
    if bounds is None:
        return
    if len(bounds) != 4 or not np.isfinite(np.asarray(bounds, dtype=float)).all():
        raise ValueError("bounds must be four finite values: minx, miny, maxx, maxy")
    minx, miny, maxx, maxy = map(float, bounds)
    if minx > maxx or miny > maxy:
        raise ValueError("bounds minima cannot exceed maxima")
    outside = (x < minx) | (x > maxx) | (y < miny) | (y > maxy)
    if outside.any():
        raise ValueError(f"{int(outside.sum())} coordinates fall outside the allowed bounds")


def canonicalize_grid(
    frame: pd.DataFrame,
    *,
    crs: str | int,
    grid_id_col: str = "gid",
    x_col: str = "x_5179",
    y_col: str = "y_5179",
    population_col: str = "elderly65_calibrated_population",
    admin_col: str = "admin_dong_code",
    sigungu_col: str = "policy_sigungu_name",
    expected_rows: int | None = None,
    expected_admin_codes: Sequence[object] | None = None,
    expected_admin_count: int | None = None,
    bounds: tuple[float, float, float, float] | None = None,
    require_unique_coordinates: bool = True,
) -> pd.DataFrame:
    """Validate and deterministically order the canonical elderly grid.

    The caller must state the CRS explicitly even for a CSV source.  This
    prevents longitude/latitude or an unknown projected CRS from silently
    entering a metre-based catchment calculation.
    """

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("grid must be a pandas DataFrame")
    _canonical_crs(crs)
    required = {grid_id_col, x_col, y_col, population_col, admin_col, sigungu_col}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"Grid is missing required columns: {missing}")
    if expected_rows is not None and len(frame) != int(expected_rows):
        raise ValueError(f"Expected {int(expected_rows)} grid rows, found {len(frame)}")
    if frame.empty:
        raise ValueError("Grid cannot be empty")

    result = frame.copy()
    result[grid_id_col] = _normalise_text(result[grid_id_col], grid_id_col)
    result[admin_col] = _normalise_text(result[admin_col], admin_col)
    result[sigungu_col] = _normalise_text(result[sigungu_col], sigungu_col)
    if result[grid_id_col].duplicated().any():
        duplicated = result.loc[result[grid_id_col].duplicated(False), grid_id_col].iloc[0]
        raise ValueError(f"Grid ID is not unique: {duplicated}")
    result[x_col] = _finite_numeric(result, x_col)
    result[y_col] = _finite_numeric(result, y_col)
    result[population_col] = _finite_numeric(result, population_col)
    if result[population_col].lt(0).any():
        raise ValueError("Grid population must be non-negative")
    if require_unique_coordinates and result.duplicated([x_col, y_col]).any():
        raise ValueError("Grid coordinates are not unique")
    _validate_bounds(result[x_col].to_numpy(), result[y_col].to_numpy(), bounds)
    _validate_expected_admins(
        result[admin_col],
        expected_admin_codes=expected_admin_codes,
        expected_admin_count=expected_admin_count,
    )
    result = result.sort_values(grid_id_col, kind="stable").reset_index(drop=True)
    result.attrs["crs"] = _CRS_5179
    result.attrs["order_sha"] = order_sha256(result[grid_id_col])
    return result


def canonicalize_venues(
    frame: pd.DataFrame,
    *,
    crs: str | int,
    venue_id_col: str = "venue_id",
    x_col: str = "x_5179",
    y_col: str = "y_5179",
    admin_col: str = "admin_dong_code",
    sigungu_col: str = "policy_sigungu_name",
    expected_rows: int | None = None,
    expected_admin_codes: Sequence[object] | None = None,
    expected_admin_count: int | None = None,
    admin_to_sigungu: Mapping[object, object] | None = None,
    bounds: tuple[float, float, float, float] | None = None,
    require_unique_coordinates: bool = True,
) -> pd.DataFrame:
    """Validate and deterministically order canonical physical venues."""

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("venues must be a pandas DataFrame")
    _canonical_crs(crs)
    required = {venue_id_col, x_col, y_col, admin_col, sigungu_col}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"Venues are missing required columns: {missing}")
    if expected_rows is not None and len(frame) != int(expected_rows):
        raise ValueError(f"Expected {int(expected_rows)} venue rows, found {len(frame)}")
    if frame.empty:
        raise ValueError("Venues cannot be empty")

    result = frame.copy()
    result[venue_id_col] = _normalise_text(result[venue_id_col], venue_id_col)
    result[admin_col] = _normalise_text(result[admin_col], admin_col)
    result[sigungu_col] = _normalise_text(result[sigungu_col], sigungu_col)
    if result[venue_id_col].duplicated().any():
        duplicated = result.loc[result[venue_id_col].duplicated(False), venue_id_col].iloc[0]
        raise ValueError(f"Venue ID is not unique: {duplicated}")
    result[x_col] = _finite_numeric(result, x_col)
    result[y_col] = _finite_numeric(result, y_col)
    if require_unique_coordinates and result.duplicated([x_col, y_col]).any():
        raise ValueError("Venue coordinates are not unique physical locations")
    _validate_bounds(result[x_col].to_numpy(), result[y_col].to_numpy(), bounds)
    _validate_expected_admins(
        result[admin_col],
        expected_admin_codes=expected_admin_codes,
        expected_admin_count=expected_admin_count,
    )
    sigungu_mismatch_count = 0
    if admin_to_sigungu is not None:
        keys = _normalise_text(
            pd.Series(list(admin_to_sigungu.keys()), dtype="object"),
            "admin_to_sigungu keys",
        )
        values = _normalise_text(
            pd.Series(list(admin_to_sigungu.values()), dtype="object"),
            "admin_to_sigungu values",
        )
        if keys.duplicated().any():
            raise ValueError("admin_to_sigungu has duplicate normalised admin codes")
        mapping = dict(zip(keys.astype(str), values.astype(str), strict=True))
        derived = result[admin_col].astype(str).map(mapping)
        if derived.isna().any():
            missing_admins = sorted(result.loc[derived.isna(), admin_col].astype(str).unique())
            raise ValueError(
                "admin_to_sigungu does not cover venue admin codes: "
                f"{missing_admins[:10]}"
            )
        source_column = f"{sigungu_col}_source"
        if source_column in result.columns:
            raise ValueError(f"Cannot preserve source sigungu; column already exists: {source_column}")
        result[source_column] = result[sigungu_col]
        sigungu_mismatch_count = int(result[sigungu_col].astype(str).ne(derived).sum())
        # The venue source field is descriptive provenance only.  Boundary
        # logic must use the frozen admin-code mapping supplied by the caller.
        result[sigungu_col] = derived.astype("string")
    result = result.sort_values(venue_id_col, kind="stable").reset_index(drop=True)
    result.attrs["crs"] = _CRS_5179
    result.attrs["order_sha"] = order_sha256(result[venue_id_col])
    result.attrs["sigungu_rederived_from_admin"] = admin_to_sigungu is not None
    result.attrs["source_sigungu_mismatch_count"] = sigungu_mismatch_count
    return result


def order_sha256(values: Sequence[object] | pd.Series | pd.Index) -> str:
    """Seal an exact row/column ID order with an unambiguous SHA-256.

    Length-prefixed UTF-8 values avoid delimiter ambiguity (for example,
    ``["ab", "c"]`` versus ``["a", "bc"]``).
    """

    series = _normalise_text(pd.Series(list(values), dtype="object"), "order IDs")
    if series.duplicated().any():
        raise ValueError("Order IDs must be unique")
    digest = hashlib.sha256(b"MEDIROAD_STAGE3_ORDER_V1\0")
    for value in series.astype(str):
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="little", signed=False))
        digest.update(encoded)
    return digest.hexdigest()


@dataclass(frozen=True)
class SparseCatchment:
    """A CSR venue-by-grid matrix with an immutable ordering contract."""

    matrix_id: str
    method: Method
    parameter_m: float
    support_m: float
    matrix: sparse.csr_matrix
    venue_ids: tuple[str, ...]
    grid_ids: tuple[str, ...]
    venue_order_sha: str = field(init=False)
    grid_order_sha: str = field(init=False)

    def __post_init__(self) -> None:
        if self.method not in {"hard", "exponential", "gaussian"}:
            raise ValueError(f"Unknown catchment method: {self.method}")
        if not sparse.isspmatrix_csr(self.matrix):
            raise TypeError("Catchment matrix must be scipy.sparse.csr_matrix; dense is forbidden")
        parameter = float(self.parameter_m)
        support = float(self.support_m)
        if not math.isfinite(parameter) or parameter <= 0:
            raise ValueError("parameter_m must be positive and finite")
        if not math.isfinite(support) or support <= 0:
            raise ValueError("support_m must be positive and finite")
        if self.method == "hard" and not math.isclose(parameter, support):
            raise ValueError("A hard-radius artifact must have support_m == parameter_m")
        if support < parameter:
            raise ValueError("support_m cannot be smaller than parameter_m")
        if self.matrix.shape != (len(self.venue_ids), len(self.grid_ids)):
            raise ValueError("Catchment shape does not match sealed venue/grid orders")
        if len(set(self.venue_ids)) != len(self.venue_ids) or len(set(self.grid_ids)) != len(
            self.grid_ids
        ):
            raise ValueError("Catchment row/column IDs must be unique")
        data = self.matrix.data
        if not np.isfinite(data).all() or (data <= 0).any() or (data > 1 + 1e-7).any():
            raise ValueError("Catchment weights must be finite and in (0, 1]")
        if not self.matrix.has_sorted_indices:
            raise ValueError("Catchment CSR indices must be sorted")
        if self.method == "hard" and data.size and not np.equal(data, 1).all():
            raise ValueError("Hard-radius catchment weights must equal one")
        object.__setattr__(self, "venue_order_sha", order_sha256(self.venue_ids))
        object.__setattr__(self, "grid_order_sha", order_sha256(self.grid_ids))


def _default_decay_support() -> dict[str, dict[float, float]]:
    # Explicit finite support is essential: continuous kernels are otherwise
    # mathematically dense.  Three scales retains weights down to exp(-3) for
    # exponential and exp(-4.5) for Gaussian kernels.
    return {
        "exponential": {1000.0: 3000.0, 3000.0: 9000.0, 5000.0: 15000.0},
        "gaussian": {1000.0: 3000.0, 3000.0: 9000.0, 5000.0: 15000.0},
    }


@dataclass(frozen=True)
class CatchmentConfig:
    """Locked Stage 3 catchment scenarios and finite decay supports."""

    hard_radii_m: tuple[float, ...] = (1000.0, 3000.0, 5000.0, 10000.0)
    decay_scales_m: tuple[float, ...] = (1000.0, 3000.0, 5000.0)
    decay_max_support_m: Mapping[str, Mapping[float, float]] = field(
        default_factory=_default_decay_support
    )
    leafsize: int = 32
    dtype: Any = np.float32

    def validated(self) -> "CatchmentConfig":
        hard = _positive_unique_sorted(self.hard_radii_m, "hard_radii_m")
        scales = _positive_unique_sorted(self.decay_scales_m, "decay_scales_m")
        if int(self.leafsize) <= 0:
            raise ValueError("leafsize must be positive")
        dtype = np.dtype(self.dtype)
        if dtype.kind != "f":
            raise ValueError("Catchment dtype must be floating point")
        for method in ("exponential", "gaussian"):
            if method not in self.decay_max_support_m:
                raise ValueError(f"Missing finite support config for {method}")
            supports = _normalise_support_mapping(self.decay_max_support_m[method])
            for scale in scales:
                if scale not in supports:
                    raise ValueError(f"Missing {method} max support for scale {scale:g} m")
                support = supports[scale]
                if support < scale:
                    raise ValueError(f"{method} support cannot be below its scale")
                ratio = support / scale
                if ratio > 50 or (method == "gaussian" and ratio > 12):
                    raise ValueError(f"{method} support/scale is too large for stable sparse weights")
        # Normalisation is validated here; original tuples remain intact so the
        # dataclass can also serve as an exact user configuration record.
        _ = hard, scales
        return self


def _positive_unique_sorted(values: Sequence[float], name: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if not result or not np.isfinite(np.asarray(result)).all() or any(value <= 0 for value in result):
        raise ValueError(f"{name} must contain positive finite values")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} contains duplicates")
    if tuple(sorted(result)) != result:
        raise ValueError(f"{name} must be strictly increasing")
    return result


def _normalise_support_mapping(values: Mapping[float, float]) -> dict[float, float]:
    if not isinstance(values, Mapping):
        raise TypeError("Decay max support must be a mapping from scale to support")
    result: dict[float, float] = {}
    for key, value in values.items():
        scale = float(key)
        support = float(value)
        if scale in result:
            raise ValueError(f"Duplicate decay scale after numeric normalisation: {scale}")
        if not math.isfinite(scale) or scale <= 0 or not math.isfinite(support) or support <= 0:
            raise ValueError("Decay scales and supports must be positive and finite")
        result[scale] = support
    return result


def _ordered_spatial_inputs(
    frame: pd.DataFrame,
    *,
    id_col: str,
    x_col: str,
    y_col: str,
    kind: str,
) -> tuple[tuple[str, ...], np.ndarray]:
    required = {id_col, x_col, y_col}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"{kind} is missing required columns: {missing}")
    ids = _normalise_text(frame[id_col], id_col)
    if ids.duplicated().any():
        raise ValueError(f"{kind} IDs must be unique")
    x = _finite_numeric(frame, x_col)
    y = _finite_numeric(frame, y_col)
    order = np.argsort(ids.to_numpy(dtype=str), kind="stable")
    ordered_ids = tuple(ids.iloc[order].astype(str))
    coordinates = np.column_stack([x.iloc[order].to_numpy(), y.iloc[order].to_numpy()])
    return ordered_ids, np.ascontiguousarray(coordinates, dtype=np.float64)


def _distance_label(value_m: float) -> str:
    # Matrix IDs mirror the locked YAML contract (for example,
    # ``hard_5000m``); presentation layers may render the same value as 5 km.
    if float(value_m).is_integer():
        return f"{int(value_m)}m"
    return f"{float(value_m):g}m"


def _query_sparse_distances(
    venue_coordinates: np.ndarray,
    grid_coordinates: np.ndarray,
    *,
    max_support_m: float,
    leafsize: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if venue_coordinates.shape[0] == 0 or grid_coordinates.shape[0] == 0:
        raise ValueError("Venue and grid coordinate arrays cannot be empty")
    venue_tree = cKDTree(
        venue_coordinates,
        leafsize=int(leafsize),
        compact_nodes=True,
        balanced_tree=True,
    )
    grid_tree = cKDTree(
        grid_coordinates,
        leafsize=int(leafsize),
        compact_nodes=True,
        balanced_tree=True,
    )
    distances = venue_tree.sparse_distance_matrix(
        grid_tree,
        max_distance=float(max_support_m),
        p=2.0,
        output_type="coo_matrix",
    )
    row = np.asarray(distances.row, dtype=np.int64)
    col = np.asarray(distances.col, dtype=np.int64)
    data = np.asarray(distances.data, dtype=np.float64)
    if not np.isfinite(data).all() or (data < 0).any():
        raise RuntimeError("KD-tree returned invalid distances")
    if row.size:
        order = np.lexsort((col, row))
        row, col, data = row[order], col[order], data[order]
    return row, col, data


def _artifact_from_distances(
    *,
    rows: np.ndarray,
    columns: np.ndarray,
    distances: np.ndarray,
    shape: tuple[int, int],
    venue_ids: tuple[str, ...],
    grid_ids: tuple[str, ...],
    method: Method,
    parameter_m: float,
    support_m: float,
    dtype: Any,
) -> SparseCatchment:
    tolerance = max(1e-9, support_m * 1e-12)
    keep = distances <= support_m + tolerance
    selected_rows = rows[keep]
    selected_columns = columns[keep]
    selected_distances = distances[keep]
    if method == "hard":
        weights = np.ones(selected_distances.size, dtype=dtype)
        prefix = "hard"
    elif method == "exponential":
        weights = np.exp(-selected_distances / parameter_m).astype(dtype, copy=False)
        prefix = "exp"
    elif method == "gaussian":
        weights = np.exp(-0.5 * np.square(selected_distances / parameter_m)).astype(
            dtype, copy=False
        )
        prefix = "gaussian"
    else:  # pragma: no cover - Method and public validation make this unreachable.
        raise ValueError(f"Unknown method: {method}")
    if weights.size and (weights <= 0).any():
        raise ValueError("Configured support underflows the requested sparse dtype")
    matrix = sparse.csr_matrix(
        (weights, (selected_rows, selected_columns)), shape=shape, dtype=dtype
    )
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.sort_indices()
    return SparseCatchment(
        matrix_id=f"{prefix}_{_distance_label(parameter_m)}",
        method=method,
        parameter_m=float(parameter_m),
        support_m=float(support_m),
        matrix=matrix,
        venue_ids=venue_ids,
        grid_ids=grid_ids,
    )


def build_catchment_family(
    venues: pd.DataFrame,
    grid: pd.DataFrame,
    *,
    config: CatchmentConfig | None = None,
    venue_id_col: str = "venue_id",
    grid_id_col: str = "gid",
    venue_x_col: str = "x_5179",
    venue_y_col: str = "y_5179",
    grid_x_col: str = "x_5179",
    grid_y_col: str = "y_5179",
) -> dict[str, SparseCatchment]:
    """Build every hard/decay scenario from one maximum-radius KD-tree query."""

    resolved = (config or CatchmentConfig()).validated()
    hard = _positive_unique_sorted(resolved.hard_radii_m, "hard_radii_m")
    scales = _positive_unique_sorted(resolved.decay_scales_m, "decay_scales_m")
    supports = {
        method: _normalise_support_mapping(resolved.decay_max_support_m[method])
        for method in ("exponential", "gaussian")
    }
    venue_ids, venue_coordinates = _ordered_spatial_inputs(
        venues,
        id_col=venue_id_col,
        x_col=venue_x_col,
        y_col=venue_y_col,
        kind="Venues",
    )
    grid_ids, grid_coordinates = _ordered_spatial_inputs(
        grid,
        id_col=grid_id_col,
        x_col=grid_x_col,
        y_col=grid_y_col,
        kind="Grid",
    )
    max_support = max(
        max(hard),
        *(supports[method][scale] for method in supports for scale in scales),
    )
    rows, columns, distances = _query_sparse_distances(
        venue_coordinates,
        grid_coordinates,
        max_support_m=max_support,
        leafsize=resolved.leafsize,
    )
    shape = (len(venue_ids), len(grid_ids))
    result: dict[str, SparseCatchment] = {}
    for radius in hard:
        artifact = _artifact_from_distances(
            rows=rows,
            columns=columns,
            distances=distances,
            shape=shape,
            venue_ids=venue_ids,
            grid_ids=grid_ids,
            method="hard",
            parameter_m=radius,
            support_m=radius,
            dtype=resolved.dtype,
        )
        result[artifact.matrix_id] = artifact
    for method in ("exponential", "gaussian"):
        for scale in scales:
            artifact = _artifact_from_distances(
                rows=rows,
                columns=columns,
                distances=distances,
                shape=shape,
                venue_ids=venue_ids,
                grid_ids=grid_ids,
                method=method,  # type: ignore[arg-type]
                parameter_m=scale,
                support_m=supports[method][scale],
                dtype=resolved.dtype,
            )
            result[artifact.matrix_id] = artifact
    return result


def build_sparse_catchment(
    venues: pd.DataFrame,
    grid: pd.DataFrame,
    *,
    method: Method,
    radius_m: float | None = None,
    scale_m: float | None = None,
    max_support_m: float | None = None,
    venue_id_col: str = "venue_id",
    grid_id_col: str = "gid",
    venue_x_col: str = "x_5179",
    venue_y_col: str = "y_5179",
    grid_x_col: str = "x_5179",
    grid_y_col: str = "y_5179",
    leafsize: int = 32,
    dtype: Any = np.float32,
) -> SparseCatchment:
    """Build one sparse catchment; continuous kernels require finite support."""

    if method == "hard":
        if radius_m is None or scale_m is not None or max_support_m is not None:
            raise ValueError("Hard catchment requires only radius_m")
        parameter = support = float(radius_m)
    elif method in {"exponential", "gaussian"}:
        if scale_m is None or max_support_m is None or radius_m is not None:
            raise ValueError("Decay catchment requires scale_m and explicit max_support_m")
        parameter, support = float(scale_m), float(max_support_m)
    else:
        raise ValueError(f"Unknown catchment method: {method}")
    mapping = {
        "exponential": {parameter: support},
        "gaussian": {parameter: support},
    }
    if method == "hard":
        config = CatchmentConfig(
            hard_radii_m=(parameter,),
            decay_scales_m=(parameter,),
            decay_max_support_m={
                "exponential": {parameter: parameter},
                "gaussian": {parameter: parameter},
            },
            leafsize=leafsize,
            dtype=dtype,
        )
    else:
        config = CatchmentConfig(
            hard_radii_m=(support,),
            decay_scales_m=(parameter,),
            decay_max_support_m=mapping,
            leafsize=leafsize,
            dtype=dtype,
        )
    # Query once for the exact scenario, avoiding the extra family artifacts.
    config.validated()
    venue_ids, venue_coordinates = _ordered_spatial_inputs(
        venues,
        id_col=venue_id_col,
        x_col=venue_x_col,
        y_col=venue_y_col,
        kind="Venues",
    )
    grid_ids, grid_coordinates = _ordered_spatial_inputs(
        grid,
        id_col=grid_id_col,
        x_col=grid_x_col,
        y_col=grid_y_col,
        kind="Grid",
    )
    rows, columns, distances = _query_sparse_distances(
        venue_coordinates,
        grid_coordinates,
        max_support_m=support,
        leafsize=leafsize,
    )
    return _artifact_from_distances(
        rows=rows,
        columns=columns,
        distances=distances,
        shape=(len(venue_ids), len(grid_ids)),
        venue_ids=venue_ids,
        grid_ids=grid_ids,
        method=method,
        parameter_m=parameter,
        support_m=support,
        dtype=dtype,
    )


def matrix_manifest_record(
    artifact: SparseCatchment,
    *,
    source_sha: str | None = None,
) -> dict[str, Any]:
    """Return the manifest fields needed to save/reload a CSR artifact safely."""

    rows, columns = artifact.matrix.shape
    denominator = rows * columns
    record: dict[str, Any] = {
        "matrix_id": artifact.matrix_id,
        "method": artifact.method,
        "radius_or_scale_m": artifact.parameter_m,
        "max_support_m": artifact.support_m,
        "n_venues": rows,
        "n_grids": columns,
        "nnz": int(artifact.matrix.nnz),
        "density": float(artifact.matrix.nnz / denominator) if denominator else 0.0,
        "venue_order_sha": artifact.venue_order_sha,
        "grid_order_sha": artifact.grid_order_sha,
    }
    if source_sha is not None:
        text = str(source_sha).lower().strip()
        if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
            raise ValueError("source_sha must be a 64-character SHA-256 hex digest")
        record["source_sha"] = text
    return record


def _align_frame(frame: pd.DataFrame, id_col: str, expected: tuple[str, ...], kind: str) -> pd.DataFrame:
    if id_col not in frame:
        raise KeyError(f"{kind} lacks order column {id_col}")
    work = frame.copy()
    work[id_col] = _normalise_text(work[id_col], id_col)
    if work[id_col].duplicated().any():
        raise ValueError(f"{kind} order IDs are not unique")
    observed = set(work[id_col].astype(str))
    wanted = set(expected)
    if observed != wanted or len(work) != len(expected):
        raise ValueError(
            f"{kind} IDs do not match sealed matrix order; "
            f"missing={sorted(wanted - observed)[:10]}, extra={sorted(observed - wanted)[:10]}"
        )
    aligned = work.set_index(id_col, drop=False).loc[list(expected)].reset_index(drop=True)
    if order_sha256(aligned[id_col]) != order_sha256(expected):
        raise RuntimeError(f"{kind} order SHA verification failed")
    return aligned


def _vector(values: Sequence[float] | pd.Series, length: int, name: str) -> np.ndarray:
    result = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=np.float64)
    if result.shape != (length,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain {length} finite numeric values")
    return result


def compute_venue_exposures(
    artifact: SparseCatchment,
    venues: pd.DataFrame,
    grid: pd.DataFrame,
    *,
    venue_id_col: str = "venue_id",
    grid_id_col: str = "gid",
    population_col: str = "elderly65_calibrated_population",
    need_col: str = "normalized_need",
    admin_col: str = "admin_dong_code",
    sigungu_col: str = "policy_sigungu_name",
    high_need_mask: Sequence[bool] | pd.Series | None = None,
    high_need_quantile: float = 0.8,
) -> pd.DataFrame:
    """Compute total, need/high-need, and boundary-stratified exposure.

    ``need_col`` must already be a non-negative, frozen Stage 1 need weight.
    The routine never combines it with specialty gap, preserving the Stage 3
    efficiency/equity separation.
    """

    venue = _align_frame(venues, venue_id_col, artifact.venue_ids, "Venues")
    cells = _align_frame(grid, grid_id_col, artifact.grid_ids, "Grid")
    required_venue = {admin_col, sigungu_col}
    required_grid = {population_col, need_col, admin_col, sigungu_col}
    if missing := sorted(required_venue - set(venue.columns)):
        raise KeyError(f"Venues lack exposure columns: {missing}")
    if missing := sorted(required_grid - set(cells.columns)):
        raise KeyError(f"Grid lacks exposure columns: {missing}")

    population = _vector(cells[population_col], len(cells), population_col)
    need = _vector(cells[need_col], len(cells), need_col)
    if (population < 0).any() or (need < 0).any():
        raise ValueError("Population and need weights must be non-negative")
    if high_need_mask is None:
        quantile = float(high_need_quantile)
        if not math.isfinite(quantile) or not 0 < quantile < 1:
            raise ValueError("high_need_quantile must lie strictly between zero and one")
        threshold = float(np.quantile(need, quantile, method="linear"))
        high = need >= threshold
    else:
        raw_mask = np.asarray(high_need_mask)
        if raw_mask.shape != (len(cells),) or raw_mask.dtype.kind != "b":
            raise ValueError("high_need_mask must be a boolean vector in sealed grid order")
        high = raw_mask.astype(bool, copy=False)
        threshold = math.nan

    matrix = artifact.matrix
    raw = np.asarray(matrix @ population).reshape(-1)
    need_weighted = np.asarray(matrix @ (population * need)).reshape(-1)
    high_need = np.asarray(matrix @ (population * high.astype(np.float64))).reshape(-1)

    venue_admin = _normalise_text(venue[admin_col], admin_col).to_numpy(dtype=str)
    venue_sigungu = _normalise_text(venue[sigungu_col], sigungu_col).to_numpy(dtype=str)
    grid_admin = _normalise_text(cells[admin_col], admin_col).to_numpy(dtype=str)
    grid_sigungu = _normalise_text(cells[sigungu_col], sigungu_col).to_numpy(dtype=str)
    coo = matrix.tocoo(copy=False)
    contribution = coo.data.astype(np.float64, copy=False) * population[coo.col]
    need_contribution = contribution * need[coo.col]
    high_contribution = contribution * high[coo.col].astype(np.float64)
    own_mask = venue_admin[coo.row] == grid_admin[coo.col]
    same_sigungu_mask = venue_sigungu[coo.row] == grid_sigungu[coo.col]
    other_same_mask = same_sigungu_mask & ~own_mask
    cross_sigungu_mask = ~same_sigungu_mask

    def aggregate(mask: np.ndarray, weights: np.ndarray = contribution) -> np.ndarray:
        return np.bincount(
            coo.row[mask], weights=weights[mask], minlength=matrix.shape[0]
        ).astype(float)

    own = aggregate(own_mask)
    other_same = aggregate(other_same_mask)
    cross_sigungu = aggregate(cross_sigungu_mask)
    own_need = aggregate(own_mask, need_contribution)
    other_same_need = aggregate(other_same_mask, need_contribution)
    cross_sigungu_need = aggregate(cross_sigungu_mask, need_contribution)
    own_high = aggregate(own_mask, high_contribution)
    other_same_high = aggregate(other_same_mask, high_contribution)
    cross_sigungu_high = aggregate(cross_sigungu_mask, high_contribution)
    partition = own + other_same + cross_sigungu
    if not np.allclose(partition, raw, rtol=2e-6, atol=1e-6):
        raise RuntimeError("Own/same-sigungu/cross-sigungu exposure does not partition total")
    need_partition = own_need + other_same_need + cross_sigungu_need
    if not np.allclose(need_partition, need_weighted, rtol=2e-6, atol=1e-6):
        raise RuntimeError("Need-weighted boundary exposure does not partition total")
    high_partition = own_high + other_same_high + cross_sigungu_high
    if not np.allclose(high_partition, high_need, rtol=2e-6, atol=1e-6):
        raise RuntimeError("High-Need boundary exposure does not partition total")

    result = pd.DataFrame(
        {
            venue_id_col: list(artifact.venue_ids),
            "raw_exposure": raw,
            "need_weighted_exposure": need_weighted,
            "high_need_exposure": high_need,
            "own_admin_exposure": own,
            "other_admin_same_sigungu_exposure": other_same,
            "same_sigungu_exposure": own + other_same,
            "cross_admin_exposure": other_same + cross_sigungu,
            "cross_sigungu_exposure": cross_sigungu,
            "own_admin_need_weighted_exposure": own_need,
            "other_admin_same_sigungu_need_weighted_exposure": other_same_need,
            "cross_admin_need_weighted_exposure": other_same_need + cross_sigungu_need,
            "cross_sigungu_need_weighted_exposure": cross_sigungu_need,
            "own_admin_high_need_exposure": own_high,
            "other_admin_same_sigungu_high_need_exposure": other_same_high,
            "cross_admin_high_need_exposure": other_same_high + cross_sigungu_high,
            "cross_sigungu_high_need_exposure": cross_sigungu_high,
        }
    )
    numeric = result.drop(columns=venue_id_col).to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or (numeric < -1e-8).any():
        raise RuntimeError("Exposure calculation produced invalid values")
    result.attrs["matrix_id"] = artifact.matrix_id
    result.attrs["high_need_threshold"] = threshold
    result.attrs["interpretation"] = "potential beneficiary geographic exposure"
    return result


def _require_compatible(left: SparseCatchment, right: SparseCatchment) -> None:
    if left.matrix.shape != right.matrix.shape:
        raise ValueError("Catchment matrices have different shapes")
    if left.venue_order_sha != right.venue_order_sha or left.grid_order_sha != right.grid_order_sha:
        raise ValueError("Catchment matrices have different sealed orders")


def assert_hard_radius_monotonicity(
    artifacts: Sequence[SparseCatchment] | Mapping[str, SparseCatchment],
) -> pd.DataFrame:
    """Fail if any smaller hard catchment is not a subset of the next radius."""

    values = list(artifacts.values()) if isinstance(artifacts, Mapping) else list(artifacts)
    hard = sorted((item for item in values if item.method == "hard"), key=lambda x: x.parameter_m)
    if len(hard) < 2:
        raise ValueError("At least two hard-radius artifacts are required")
    records: list[dict[str, Any]] = []
    for smaller, larger in zip(hard, hard[1:]):
        _require_compatible(smaller, larger)
        retained = smaller.matrix.multiply(larger.matrix)
        missing = int(smaller.matrix.nnz - retained.nnz)
        if missing:
            raise ValueError(
                f"Hard-radius monotonicity failed: {missing} cells in "
                f"{smaller.matrix_id} are absent from {larger.matrix_id}"
            )
        records.append(
            {
                "smaller_matrix_id": smaller.matrix_id,
                "larger_matrix_id": larger.matrix_id,
                "smaller_nnz": int(smaller.matrix.nnz),
                "larger_nnz": int(larger.matrix.nnz),
                "missing_from_larger": missing,
                "pass": True,
            }
        )
    return pd.DataFrame(records)


def _require_sparse_matrix(matrix: sparse.spmatrix) -> sparse.csr_matrix:
    if not sparse.issparse(matrix):
        raise TypeError("A SciPy sparse matrix is required; dense venue-by-grid input is forbidden")
    result = matrix.tocsr(copy=False)
    if result.ndim != 2:
        raise ValueError("Sparse coverage matrix must be two-dimensional")
    if not np.isfinite(result.data).all() or (result.data < 0).any():
        raise ValueError("Sparse coverage values must be finite and non-negative")
    result.sort_indices()
    return result


def _validate_row_indices(row_indices: Sequence[int], n_rows: int) -> np.ndarray:
    raw = np.asarray(list(row_indices))
    if raw.ndim != 1 or raw.dtype.kind not in "iu":
        raise TypeError("row_indices must contain integers")
    rows = raw.astype(np.int64, copy=False)
    if rows.size and ((rows < 0).any() or (rows >= n_rows).any()):
        raise IndexError("row_indices contains an out-of-range venue row")
    if np.unique(rows).size != rows.size:
        raise ValueError("row_indices contains duplicates")
    return np.sort(rows)


def sparse_union(
    matrix: sparse.spmatrix,
    row_indices: Sequence[int],
    *,
    binary: bool = True,
) -> sparse.csr_matrix:
    """Return a sparse 1-by-grid union (OR for hard, max for decay weights)."""

    coverage = _require_sparse_matrix(matrix)
    rows = _validate_row_indices(row_indices, coverage.shape[0])
    if rows.size == 0:
        return sparse.csr_matrix((1, coverage.shape[1]), dtype=coverage.dtype)
    selected = coverage[rows].tocoo(copy=False)
    if selected.nnz == 0:
        return sparse.csr_matrix((1, coverage.shape[1]), dtype=coverage.dtype)
    order = np.argsort(selected.col, kind="stable")
    columns = selected.col[order]
    values = selected.data[order]
    starts = np.r_[0, np.flatnonzero(columns[1:] != columns[:-1]) + 1]
    unique_columns = columns[starts]
    if binary:
        union_values = np.ones(unique_columns.size, dtype=coverage.dtype)
    else:
        union_values = np.maximum.reduceat(values, starts).astype(coverage.dtype, copy=False)
    result = sparse.csr_matrix(
        (union_values, (np.zeros(unique_columns.size, dtype=np.int64), unique_columns)),
        shape=(1, coverage.shape[1]),
    )
    result.sort_indices()
    return result


@dataclass(frozen=True)
class UnionCoverage:
    covered_grid_count: int
    unique_weighted_coverage: float
    naive_weighted_sum: float
    redundancy_mass: float


def weighted_union_coverage(
    matrix: sparse.spmatrix,
    row_indices: Sequence[int],
    grid_weights: Sequence[float] | pd.Series,
    *,
    binary: bool = True,
) -> UnionCoverage:
    """Calculate unique union coverage and its naive double-counting baseline."""

    coverage = _require_sparse_matrix(matrix)
    rows = _validate_row_indices(row_indices, coverage.shape[0])
    weights = _vector(grid_weights, coverage.shape[1], "grid_weights")
    if (weights < 0).any():
        raise ValueError("grid_weights must be non-negative")
    union = sparse_union(coverage, rows, binary=binary)
    unique = float(np.dot(union.data.astype(float), weights[union.indices]))
    selected = coverage[rows].tocoo(copy=False)
    selected_values = np.ones(selected.nnz) if binary else selected.data.astype(float)
    naive = float(np.dot(selected_values, weights[selected.col]))
    tolerance = 1e-8 * max(1.0, naive)
    if unique > naive + tolerance:
        raise RuntimeError("Unique union coverage exceeds naive individual sum")
    return UnionCoverage(
        covered_grid_count=int(union.nnz),
        unique_weighted_coverage=unique,
        naive_weighted_sum=naive,
        redundancy_mass=max(0.0, naive - unique),
    )


def candidate_pairs_within_distance(
    venues: pd.DataFrame,
    max_distance_m: float,
    *,
    venue_id_col: str = "venue_id",
    x_col: str = "x_5179",
    y_col: str = "y_5179",
) -> pd.DataFrame:
    """Return deterministic venue pairs that can geometrically overlap."""

    distance = float(max_distance_m)
    if not math.isfinite(distance) or distance <= 0:
        raise ValueError("max_distance_m must be positive and finite")
    ids, coordinates = _ordered_spatial_inputs(
        venues, id_col=venue_id_col, x_col=x_col, y_col=y_col, kind="Venues"
    )
    pairs = cKDTree(coordinates).query_pairs(distance, output_type="ndarray")
    if pairs.size == 0:
        return pd.DataFrame(
            {
                "venue_i_index": pd.Series(dtype="int64"),
                "venue_j_index": pd.Series(dtype="int64"),
                "venue_i": pd.Series(dtype="string"),
                "venue_j": pd.Series(dtype="string"),
                "venue_distance_m": pd.Series(dtype="float64"),
            }
        )
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    pairs.sort(axis=1)
    order = np.lexsort((pairs[:, 1], pairs[:, 0]))
    pairs = pairs[order]
    delta = coordinates[pairs[:, 0]] - coordinates[pairs[:, 1]]
    pair_distance = np.sqrt(np.einsum("ij,ij->i", delta, delta))
    return pd.DataFrame(
        {
            "venue_i_index": pairs[:, 0],
            "venue_j_index": pairs[:, 1],
            "venue_i": [ids[index] for index in pairs[:, 0]],
            "venue_j": [ids[index] for index in pairs[:, 1]],
            "venue_distance_m": pair_distance,
        }
    )


def _sparse_pair_values(
    matrix: sparse.spmatrix,
    rows: np.ndarray,
    columns: np.ndarray,
    *,
    chunk_size: int = 250_000,
) -> np.ndarray:
    result = np.empty(rows.size, dtype=np.float64)
    csr = matrix.tocsr(copy=False)
    for start in range(0, rows.size, chunk_size):
        stop = min(rows.size, start + chunk_size)
        selected = csr[rows[start:stop], columns[start:stop]]
        if sparse.issparse(selected):
            values = selected.toarray().reshape(-1)
        else:
            values = np.asarray(selected).reshape(-1)
        result[start:stop] = values
    return result


def compute_pair_overlaps(
    artifact: SparseCatchment,
    venues: pd.DataFrame,
    grid_population: Sequence[float] | pd.Series,
    *,
    grid_need: Sequence[float] | pd.Series | None = None,
    venue_id_col: str = "venue_id",
    x_col: str = "x_5179",
    y_col: str = "y_5179",
    candidate_distance_m: float | None = None,
    min_jaccard: float = 0.0,
    min_population_weighted_jaccard: float = 0.0,
) -> pd.DataFrame:
    """Compute hard-catchment overlap only for plausible spatial pairs.

    Population- and need-weighted overlap are weighted Jaccard coefficients.
    This routine intentionally requires a hard/binary matrix; applying binary
    overlap semantics to truncated decay weights would be misleading.
    """

    if artifact.method != "hard" or (
        artifact.matrix.data.size and not np.equal(artifact.matrix.data, 1).all()
    ):
        raise ValueError("Pair overlap requires a hard/binary catchment artifact")
    if not 0 <= float(min_jaccard) <= 1 or not 0 <= float(
        min_population_weighted_jaccard
    ) <= 1:
        raise ValueError("Overlap thresholds must lie in [0, 1]")
    venue = _align_frame(venues, venue_id_col, artifact.venue_ids, "Venues")
    maximum = 2.0 * artifact.support_m
    candidate_distance = maximum if candidate_distance_m is None else float(candidate_distance_m)
    if not math.isfinite(candidate_distance) or candidate_distance <= 0:
        raise ValueError("candidate_distance_m must be positive and finite")
    candidates = candidate_pairs_within_distance(
        venue,
        candidate_distance,
        venue_id_col=venue_id_col,
        x_col=x_col,
        y_col=y_col,
    )
    output_columns = [
        "venue_i",
        "venue_j",
        "venue_distance_m",
        "intersection_grid_count",
        "union_grid_count",
        "jaccard",
        "population_overlap",
        "population_union",
        "population_weighted_jaccard",
        "need_weighted_overlap",
        "need_weighted_union",
        "need_weighted_jaccard",
    ]
    if candidates.empty:
        return pd.DataFrame(columns=output_columns)

    binary = artifact.matrix.astype(np.int32, copy=True)
    binary.data.fill(1)
    intersection_matrix = binary @ binary.T
    rows = candidates["venue_i_index"].to_numpy(dtype=np.int64)
    columns = candidates["venue_j_index"].to_numpy(dtype=np.int64)
    intersection = _sparse_pair_values(intersection_matrix, rows, columns)
    keep = intersection > 0
    candidates = candidates.loc[keep].reset_index(drop=True)
    rows, columns, intersection = rows[keep], columns[keep], intersection[keep]
    if candidates.empty:
        return pd.DataFrame(columns=output_columns)

    support_size = np.diff(binary.indptr).astype(np.float64)
    union_count = support_size[rows] + support_size[columns] - intersection
    jaccard = np.divide(intersection, union_count, out=np.zeros_like(intersection), where=union_count > 0)

    population = _vector(grid_population, artifact.matrix.shape[1], "grid_population")
    if (population < 0).any():
        raise ValueError("grid_population must be non-negative")
    weighted = binary.multiply(np.sqrt(population))
    population_intersection_matrix = weighted @ weighted.T
    population_intersection = _sparse_pair_values(
        population_intersection_matrix, rows, columns
    )
    row_population = np.asarray(binary @ population).reshape(-1)
    population_union = row_population[rows] + row_population[columns] - population_intersection
    population_jaccard = np.divide(
        population_intersection,
        population_union,
        out=np.zeros_like(population_intersection),
        where=population_union > 0,
    )

    if grid_need is None:
        need_intersection = np.full(rows.size, np.nan)
        need_union = np.full(rows.size, np.nan)
        need_jaccard = np.full(rows.size, np.nan)
    else:
        need = _vector(grid_need, artifact.matrix.shape[1], "grid_need")
        if (need < 0).any():
            raise ValueError("grid_need must be non-negative")
        need_mass = population * need
        need_weighted = binary.multiply(np.sqrt(need_mass))
        need_intersection_matrix = need_weighted @ need_weighted.T
        need_intersection = _sparse_pair_values(need_intersection_matrix, rows, columns)
        row_need = np.asarray(binary @ need_mass).reshape(-1)
        need_union = row_need[rows] + row_need[columns] - need_intersection
        need_jaccard = np.divide(
            need_intersection,
            need_union,
            out=np.zeros_like(need_intersection),
            where=need_union > 0,
        )

    result = pd.DataFrame(
        {
            "venue_i": candidates["venue_i"].astype(str),
            "venue_j": candidates["venue_j"].astype(str),
            "venue_distance_m": candidates["venue_distance_m"].to_numpy(dtype=float),
            "intersection_grid_count": intersection.astype(np.int64),
            "union_grid_count": union_count.astype(np.int64),
            "jaccard": jaccard,
            "population_overlap": population_intersection,
            "population_union": population_union,
            "population_weighted_jaccard": population_jaccard,
            "need_weighted_overlap": need_intersection,
            "need_weighted_union": need_union,
            "need_weighted_jaccard": need_jaccard,
        }
    )
    result = result.loc[
        result["jaccard"].ge(float(min_jaccard))
        & result["population_weighted_jaccard"].ge(float(min_population_weighted_jaccard))
    ]
    return result.sort_values(["venue_i", "venue_j"], kind="stable").reset_index(drop=True)


def connected_overlap_clusters(
    venue_ids: Sequence[object],
    edges: pd.DataFrame,
    *,
    threshold: float,
    metric_col: str = "jaccard",
    venue_i_col: str = "venue_i",
    venue_j_col: str = "venue_j",
    representative_scores: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """Build deterministic connected overlap clusters, retaining isolates."""

    cutoff = float(threshold)
    if not math.isfinite(cutoff) or not 0 <= cutoff <= 1:
        raise ValueError("Cluster threshold must lie in [0, 1]")
    ids = tuple(sorted(_normalise_text(pd.Series(list(venue_ids)), "venue_ids").astype(str)))
    if len(set(ids)) != len(ids):
        raise ValueError("venue_ids must be unique")
    if not ids:
        raise ValueError("venue_ids cannot be empty")
    required = {venue_i_col, venue_j_col, metric_col}
    if missing := sorted(required - set(edges.columns)):
        raise KeyError(f"Overlap edges lack columns: {missing}")
    index = {value: position for position, value in enumerate(ids)}
    parent = np.arange(len(ids), dtype=np.int64)
    size = np.ones(len(ids), dtype=np.int64)

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = int(parent[value])
        return value

    def union(left: int, right: int) -> None:
        root_left, root_right = find(left), find(right)
        if root_left == root_right:
            return
        if size[root_left] < size[root_right] or (
            size[root_left] == size[root_right] and ids[root_left] > ids[root_right]
        ):
            root_left, root_right = root_right, root_left
        parent[root_right] = root_left
        size[root_left] += size[root_right]

    seen_edges: set[tuple[str, str]] = set()
    for row in edges[[venue_i_col, venue_j_col, metric_col]].itertuples(index=False, name=None):
        left, right = str(row[0]).strip(), str(row[1]).strip()
        try:
            metric = float(row[2])
        except (TypeError, ValueError) as exc:
            raise ValueError("Overlap metric must be numeric") from exc
        if left not in index or right not in index:
            raise ValueError(f"Overlap edge references unknown venue: {left}, {right}")
        if left == right:
            raise ValueError("Overlap graph cannot contain self edges")
        if not math.isfinite(metric) or not 0 <= metric <= 1:
            raise ValueError("Overlap metric must be finite and in [0, 1]")
        edge = tuple(sorted((left, right)))
        if edge in seen_edges:
            raise ValueError(f"Duplicate undirected overlap edge: {edge}")
        seen_edges.add(edge)
        if metric >= cutoff:
            union(index[left], index[right])

    components: dict[int, list[str]] = {}
    for venue_id, position in index.items():
        components.setdefault(find(position), []).append(venue_id)
    groups = [sorted(members) for members in components.values()]
    groups.sort(key=lambda members: members[0])

    if representative_scores is not None:
        score_map = {str(key): float(value) for key, value in representative_scores.items()}
        if set(score_map) != set(ids) or not np.isfinite(list(score_map.values())).all():
            raise ValueError("representative_scores must provide one finite score per venue")
    else:
        score_map = {value: 0.0 for value in ids}

    records: list[dict[str, Any]] = []
    for cluster_number, members in enumerate(groups, start=1):
        representative = sorted(members, key=lambda value: (-score_map[value], value))[0]
        cluster_id = f"C{cluster_number:05d}"
        for venue_id in members:
            records.append(
                {
                    "venue_id": venue_id,
                    "coverage_cluster_id": cluster_id,
                    "representative_candidate": representative,
                    "cluster_size": len(members),
                    "alternate_candidate_count": len(members) - 1,
                    "is_representative": venue_id == representative,
                }
            )
    return pd.DataFrame(records).sort_values("venue_id", kind="stable").reset_index(drop=True)
