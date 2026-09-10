"""Build the field-verified physical candidate universe for Operational Final.

This module deliberately has no file-system side effects.  Its public builder
accepts already-loaded, audited tables and the frozen Stage 3 ``hard_5000m``
CSR matrix.  The caller remains responsible for persisting the returned
artifacts and their hashes.

Contract
--------
* ``field_evidence_release_ids`` must come from the independent field-evidence
  release gate, not merely from an all-YES form.
* Every released row must also have ``field_verified == True`` in the
  normalized form and must supply a non-placeholder ``verified_address`` plus
  finite Korean WGS84 ``verified_latitude``/``verified_longitude``.
* BUSPROXY is a spatial anchor only.  It is forbidden in the physical queue,
  physical side of the anchor mapping, release IDs, and returned universe.
* A Stage 3 row is reused only when its catalog WGS84 coordinates are unchanged.
  Changed coordinates are projected to EPSG:5179 and get a new 5 km binary row
  from a :class:`scipy.spatial.cKDTree` query against the frozen grid.
* A catalog coordinate described as a proxy/centroid/not-exact location is
  excluded unless field work supplies changed exact coordinates.
* In the returned operational-candidate table, conventional ``address``,
  ``latitude``, and ``longitude`` always mean the release-verified physical
  location.  Frozen catalog coordinates remain only in explicit
  ``*_not_for_release``/``stage3_catalog_*`` provenance columns.
* Coverage-equivalent clusters are recomputed over the resulting operational
  pool with Stage 3's locked unweighted Jaccard threshold, 0.80.  Conservatively,
  the original 10 km candidate-pair bound and 0.15 population-weighted output
  floor are retained as prerequisites.  The grouping is deterministic
  complete-link: every pair in a multi-venue cluster meets the threshold.  A
  transitive connected component is not called equivalent.

Rows that fail the exact-location requirements are recorded in ``exclusions``
and never receive a coverage row.  Structural contradictions (BUSPROXY,
unknown release IDs, a release ID whose normalized form is not verified, or a
misaligned frozen matrix) raise :class:`OperationalUniverseError` instead of
being silently repaired.

The returned SHA-256 values seal canonical *in-memory contents*, not source
files.  File SHA values should additionally be retained by the caller's normal
artifact manifest.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy import sparse
from scipy.spatial import cKDTree


HARD_5000M_RADIUS_M = 5_000.0
STAGE3_COVERAGE_CLUSTER_JACCARD = 0.80
STAGE3_OVERLAP_OUTPUT_MIN_JACCARD = 0.15
STAGE3_OVERLAP_OUTPUT_MIN_POPULATION_JACCARD = 0.15
STAGE3_PAIR_CENTER_DISTANCE_MAX_M = 10_000.0
METRIC_CRS = "EPSG:5179"
GEOGRAPHIC_CRS = "EPSG:4326"

_BUS_PREFIX = "BUSPROXY_"
_NON_EXACT_SOURCE_TOKENS = (
    "proxy",
    "centroid",
    "not_exact",
    "not exact",
    "unresolved",
    "approximate",
)
_PLACEHOLDER_TEXT = {"", "unknown", "nan", "none", "null", "n/a", "na", "미확인"}

_LEDGER_COLUMNS = [
    "venue_id",
    "stage3_venue_row",
    "stage3_catalog_latitude",
    "stage3_catalog_longitude",
    "verified_latitude",
    "verified_longitude",
    "verified_x_5179",
    "verified_y_5179",
    "catalog_coordinates_unchanged",
    "catalog_coordinate_is_proxy",
    "coverage_origin",
    "coverage_recomputed",
    "coverage_crs",
    "coverage_radius_m",
    "covered_grid_count",
    "operational_raw_elderly_exposure",
    "operational_need_weighted_exposure",
    "operational_high_need_exposure",
    "coverage_row_sha256",
]

_EXCLUSION_COLUMNS = [
    "venue_id",
    "exclusion_reason",
    "catalog_coordinate_source",
    "stage3_catalog_latitude",
    "stage3_catalog_longitude",
    "verified_address",
    "verified_latitude",
    "verified_longitude",
]

_OVERLAP_COLUMNS = [
    "venue_i",
    "venue_j",
    "intersection_grid_count",
    "union_grid_count",
    "jaccard",
    "population_overlap",
    "population_union",
    "population_weighted_jaccard",
    "venue_distance_m",
    "at_stage3_output_threshold",
    "at_cluster_threshold",
]

_CLUSTER_COLUMNS = [
    "venue_id",
    "operational_coverage_cluster_id",
    "representative_candidate",
    "cluster_size",
    "alternate_candidate_count",
    "is_representative",
    "cluster_min_pairwise_jaccard",
    "cluster_jaccard_threshold",
    "cluster_method",
]


class OperationalUniverseError(ValueError):
    """Raised when an input violates a fail-closed universe invariant."""


@dataclass(frozen=True)
class OperationalCandidateUniverse:
    """Auditable in-memory Operational Final candidate artifacts.

    ``coverage`` is aligned exactly to ``venue_ids`` and ``grid_ids``.
    ``unique_coverage`` is a 1-by-grid sparse logical OR over every operational
    row, so overlap is counted once.  ``source_anchor_provenance`` is the long
    mapping table; deterministic aggregate provenance is also attached to each
    row of ``candidates``.
    """

    candidates: pd.DataFrame
    coverage: sparse.csr_matrix
    venue_ids: np.ndarray
    grid_ids: np.ndarray
    grid: pd.DataFrame
    coverage_origin_ledger: pd.DataFrame
    source_anchor_provenance: pd.DataFrame
    release_evidence: pd.DataFrame
    audit_issues: pd.DataFrame
    coverage_overlap_edges: pd.DataFrame
    coverage_clusters: pd.DataFrame
    unique_coverage: sparse.csr_matrix
    exclusions: pd.DataFrame
    artifact_sha256: Mapping[str, str]
    summary: Mapping[str, Any]

    def grid_pattern_inputs(self) -> dict[str, Any]:
        """Return aligned inputs consumable by Stage 4 grid-pattern compression.

        The returned coverage is still sparse and its rows/columns follow the
        returned ID arrays exactly.  All extra population/need columns supplied
        in ``stage3_grid`` are retained in ``grid``.
        """

        return {
            "candidates": self.candidates,
            "grid": self.grid,
            "coverage": self.coverage,
            "venue_ids": self.venue_ids,
            "grid_ids": self.grid_ids,
        }

    def validate(self) -> None:
        """Recheck alignment and physical-only invariants."""

        if not sparse.isspmatrix_csr(self.coverage):
            raise OperationalUniverseError("operational coverage must be CSR")
        if not sparse.isspmatrix_csr(self.unique_coverage):
            raise OperationalUniverseError("unique coverage must be CSR")
        if self.coverage.shape != (len(self.venue_ids), len(self.grid_ids)):
            raise OperationalUniverseError("operational coverage/order shape mismatch")
        if self.unique_coverage.shape != (1, len(self.grid_ids)):
            raise OperationalUniverseError("unique coverage/grid shape mismatch")
        if len(self.grid) != len(self.grid_ids) or self.grid["grid_id"].astype(str).tolist() != self.grid_ids.astype(str).tolist():
            raise OperationalUniverseError("aligned grid rows do not match grid order")
        candidate_ids = self.candidates.get("venue_id", pd.Series(dtype=str)).astype(str).tolist()
        if candidate_ids != self.venue_ids.astype(str).tolist():
            raise OperationalUniverseError("candidate rows do not match venue order")
        if len(set(candidate_ids)) != len(candidate_ids):
            raise OperationalUniverseError("operational venue IDs are not unique")
        if _contains_busproxy(candidate_ids):
            raise OperationalUniverseError("BUSPROXY escaped into operational candidates")
        required_candidate_columns = {
            "verified_address",
            "verified_latitude",
            "verified_longitude",
            "field_verified",
            "coverage_reference",
            "coverage_row_sha256",
            "operational_raw_elderly_exposure",
            "operational_need_weighted_exposure",
            "operational_high_need_exposure",
            "cluster_id",
            "coverage_cluster_id",
        }
        if missing := sorted(required_candidate_columns - set(self.candidates.columns)):
            raise OperationalUniverseError(
                f"operational candidates lack required release columns: {missing}"
            )
        if len(self.candidates) and not self.candidates["field_verified"].map(_is_verified).all():
            raise OperationalUniverseError("an operational candidate is not field_verified")
        ledger_ids = self.coverage_origin_ledger.get(
            "venue_id", pd.Series(dtype=str)
        ).astype(str).tolist()
        if ledger_ids != candidate_ids:
            raise OperationalUniverseError("coverage origin ledger does not match venue order")
        cluster_ids = set(
            self.coverage_clusters.get("venue_id", pd.Series(dtype=str)).astype(str)
        )
        if cluster_ids != set(candidate_ids):
            raise OperationalUniverseError("coverage clusters do not cover the operational pool")
        if self.coverage.data.size and not np.equal(self.coverage.data, 1).all():
            raise OperationalUniverseError("operational hard coverage is not binary")
        if self.unique_coverage.data.size and not np.equal(self.unique_coverage.data, 1).all():
            raise OperationalUniverseError("unique coverage is not a sparse OR")
        expected_union = sparse_or_union(self.coverage)
        if (expected_union != self.unique_coverage).nnz:
            raise OperationalUniverseError("unique coverage differs from sparse OR")
        if any(len(str(value)) != 64 for value in self.artifact_sha256.values()):
            raise OperationalUniverseError("artifact SHA-256 ledger contains an invalid digest")


def _contains_busproxy(values: Iterable[Any]) -> bool:
    return any(str(value).strip().upper().startswith(_BUS_PREFIX) for value in values)


def _required_columns(frame: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise OperationalUniverseError(f"{label} missing required columns: {missing}")


def _normalize_ids(frame: pd.DataFrame, column: str, label: str) -> pd.DataFrame:
    _required_columns(frame, [column], label)
    result = frame.copy()
    if result[column].isna().any():
        raise OperationalUniverseError(f"{label} contains null {column}")
    result[column] = result[column].astype(str).str.strip()
    if result[column].eq("").any():
        raise OperationalUniverseError(f"{label} contains blank {column}")
    if result[column].duplicated().any():
        duplicates = result.loc[result[column].duplicated(False), column].unique().tolist()
        raise OperationalUniverseError(f"{label} contains duplicate IDs: {duplicates[:10]}")
    return result


def _normalized_release_ids(values: Iterable[str]) -> set[str]:
    raw = list(values)
    normalized = [str(value).strip() for value in raw]
    if any(not value for value in normalized):
        raise OperationalUniverseError("field evidence release IDs contain a blank value")
    if len(set(normalized)) != len(normalized):
        raise OperationalUniverseError("field evidence release IDs contain duplicates")
    if _contains_busproxy(normalized):
        raise OperationalUniverseError("BUSPROXY cannot be a field-evidence release ID")
    return set(normalized)


def _is_verified(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(math.isfinite(float(value)) and float(value) == 1.0)
    return str(value).strip().lower() in {"true", "1", "yes"}


def _binary_flag(value: Any) -> bool | None:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        if not math.isfinite(float(value)) or float(value) not in {0.0, 1.0}:
            return None
        return bool(float(value))
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


def _valid_address(value: Any) -> bool:
    if pd.isna(value):
        return False
    return str(value).strip().lower() not in _PLACEHOLDER_TEXT


def _numeric(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def _valid_korean_wgs84(latitude: float, longitude: float) -> bool:
    # EPSG:5179 is the Korean 2000 / Unified CS.  Tight national bounds also
    # catch the common latitude/longitude reversal before projection.
    return math.isfinite(latitude) and math.isfinite(longitude) and 30 <= latitude <= 40 and 120 <= longitude <= 135


def _is_proxy_coordinate_source(value: Any) -> bool:
    source = "" if pd.isna(value) else str(value).strip().lower()
    return any(token in source for token in _NON_EXACT_SOURCE_TOKENS)


def _json_scalar(value: Any) -> Any:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    return str(value)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _update_array_hash(digest: Any, array: np.ndarray) -> None:
    contiguous = np.ascontiguousarray(array)
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(np.asarray(contiguous.shape, dtype="<i8").tobytes())
    if contiguous.size:
        digest.update(memoryview(contiguous).cast("B"))


def _binary_csr_sha256(matrix: sparse.spmatrix) -> str:
    csr = matrix.tocsr(copy=False)
    digest = sha256(b"MEDIROAD_OPERATIONAL_BINARY_CSR_V1\0")
    _update_array_hash(digest, np.asarray(csr.shape, dtype="<i8"))
    _update_array_hash(digest, csr.indptr)
    _update_array_hash(digest, csr.indices)
    # Hard catchment values are all one, so the logical matrix is sealed by
    # its shape and canonical CSR structure without hashing dtype-dependent 1s.
    digest.update(b"LOGICAL_ONE_VALUES\0")
    digest.update(int(csr.nnz).to_bytes(8, byteorder="little", signed=False))
    return digest.hexdigest()


def _order_sha256(values: Sequence[Any], label: str) -> str:
    digest = sha256(f"MEDIROAD_OPERATIONAL_{label}_ORDER_V1\0".encode("utf-8"))
    for value in values:
        encoded = str(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="little", signed=False))
        digest.update(encoded)
    return digest.hexdigest()


def _frame_sha256(frame: pd.DataFrame, label: str) -> str:
    payload = frame.to_csv(
        index=False,
        lineterminator="\n",
        na_rep="<NA>",
        float_format="%.17g",
    ).encode("utf-8")
    return sha256(f"MEDIROAD_OPERATIONAL_{label}_V1\0".encode("utf-8") + payload).hexdigest()


def _canonical_binary_csr(matrix: sparse.spmatrix, label: str) -> sparse.csr_matrix:
    if not sparse.issparse(matrix):
        raise OperationalUniverseError(f"{label} must be a SciPy sparse matrix")
    result = matrix.tocsr(copy=False)
    if not result.has_canonical_format:
        result = result.copy()
        result.sum_duplicates()
        result.eliminate_zeros()
        result.sort_indices()
    if result.data.size and (
        not np.isfinite(result.data).all() or not np.equal(result.data, 1).all()
    ):
        raise OperationalUniverseError(f"{label} must be hard/binary with all stored values equal to one")
    return result


def sparse_or_union(
    coverage: sparse.spmatrix,
    row_indices: Sequence[int] | None = None,
) -> sparse.csr_matrix:
    """Return a 1-by-grid CSR logical OR without creating a dense grid vector."""

    matrix = _canonical_binary_csr(coverage, "coverage")
    if row_indices is None:
        rows = np.arange(matrix.shape[0], dtype=np.int64)
    else:
        raw = np.asarray(list(row_indices))
        if raw.ndim != 1 or raw.dtype.kind not in "iu":
            raise TypeError("row_indices must be a one-dimensional integer sequence")
        rows = raw.astype(np.int64, copy=False)
        if rows.size and ((rows < 0).any() or (rows >= matrix.shape[0]).any()):
            raise IndexError("row_indices contains an out-of-range coverage row")
        if np.unique(rows).size != rows.size:
            raise OperationalUniverseError("row_indices contains duplicates")
    if rows.size == 0:
        return sparse.csr_matrix((1, matrix.shape[1]), dtype=np.uint8)
    selected = matrix[rows]
    columns = np.unique(selected.indices)
    result = sparse.csr_matrix(
        (
            np.ones(columns.size, dtype=np.uint8),
            (np.zeros(columns.size, dtype=np.int64), columns),
        ),
        shape=(1, matrix.shape[1]),
        dtype=np.uint8,
    )
    result.sort_indices()
    return result


def _overlap_edges(
    coverage: sparse.csr_matrix,
    venue_ids: Sequence[str],
    coordinates_5179: np.ndarray,
    elderly_population: np.ndarray,
) -> pd.DataFrame:
    if coverage.shape[0] < 2:
        return pd.DataFrame(columns=_OVERLAP_COLUMNS)
    binary = coverage.astype(np.int32, copy=False)
    intersections = (binary @ binary.T).tocoo()
    sizes = np.diff(coverage.indptr).astype(np.int64)
    rows: list[dict[str, Any]] = []
    for left, right, intersection in zip(
        intersections.row, intersections.col, intersections.data
    ):
        if left >= right or intersection <= 0:
            continue
        union = int(sizes[left] + sizes[right] - intersection)
        jaccard = float(intersection / union) if union else 0.0
        left_columns = coverage.indices[
            coverage.indptr[left] : coverage.indptr[left + 1]
        ]
        right_columns = coverage.indices[
            coverage.indptr[right] : coverage.indptr[right + 1]
        ]
        intersection_columns = np.intersect1d(
            left_columns, right_columns, assume_unique=True
        )
        population_overlap = float(elderly_population[intersection_columns].sum())
        population_union = float(
            elderly_population[left_columns].sum()
            + elderly_population[right_columns].sum()
            - population_overlap
        )
        population_jaccard = (
            population_overlap / population_union if population_union > 0 else 0.0
        )
        delta = coordinates_5179[int(left)] - coordinates_5179[int(right)]
        distance = float(np.hypot(delta[0], delta[1]))
        at_output_threshold = (
            distance <= STAGE3_PAIR_CENTER_DISTANCE_MAX_M + 1e-9
            and jaccard >= STAGE3_OVERLAP_OUTPUT_MIN_JACCARD
            and population_jaccard
            >= STAGE3_OVERLAP_OUTPUT_MIN_POPULATION_JACCARD
        )
        rows.append(
            {
                "venue_i": str(venue_ids[int(left)]),
                "venue_j": str(venue_ids[int(right)]),
                "intersection_grid_count": int(intersection),
                "union_grid_count": union,
                "jaccard": jaccard,
                "population_overlap": population_overlap,
                "population_union": population_union,
                "population_weighted_jaccard": population_jaccard,
                "venue_distance_m": distance,
                "at_stage3_output_threshold": at_output_threshold,
                "at_cluster_threshold": (
                    at_output_threshold
                    and jaccard >= STAGE3_COVERAGE_CLUSTER_JACCARD
                ),
            }
        )
    return pd.DataFrame(rows, columns=_OVERLAP_COLUMNS).sort_values(
        ["venue_i", "venue_j"], kind="stable"
    ).reset_index(drop=True)


def _complete_link_clusters(
    venue_ids: Sequence[str],
    edges: pd.DataFrame,
    representative_scores: Mapping[str, float],
) -> pd.DataFrame:
    """Mirror Stage 3's deterministic greedy all-pairs clustering at 0.80."""

    identifiers = list(map(str, venue_ids))
    if not identifiers:
        return pd.DataFrame(columns=_CLUSTER_COLUMNS)
    adjacency = {venue_id: set() for venue_id in identifiers}
    strength: dict[tuple[str, str], float] = {}
    for row in edges.loc[edges["at_cluster_threshold"]].itertuples():
        left, right, value = str(row.venue_i), str(row.venue_j), float(row.jaccard)
        adjacency[left].add(right)
        adjacency[right].add(left)
        strength[tuple(sorted((left, right)))] = value

    def score(venue_id: str) -> float:
        value = _numeric(representative_scores.get(venue_id, math.nan))
        return value if math.isfinite(value) else -math.inf

    order = sorted(identifiers, key=lambda venue_id: (-score(venue_id), venue_id))
    assigned: set[str] = set()
    groups: list[list[str]] = []
    for seed in order:
        if seed in assigned:
            continue
        members = [seed]
        candidates = sorted(adjacency[seed] - assigned, key=lambda value: (-score(value), value))
        for candidate in candidates:
            if all(member in adjacency[candidate] for member in members):
                members.append(candidate)
        assigned.update(members)
        groups.append(members)

    records: list[dict[str, Any]] = []
    for number, members in enumerate(groups, start=1):
        pair_values = [
            strength[tuple(sorted((left, right)))]
            for index, left in enumerate(members)
            for right in members[index + 1 :]
        ]
        minimum = min(pair_values) if pair_values else 1.0
        if minimum < STAGE3_COVERAGE_CLUSTER_JACCARD - 1e-12:
            raise OperationalUniverseError("complete-link cluster violated the Stage 3 threshold")
        representative = members[0]
        cluster_id = f"OE{number:05d}"
        for venue_id in members:
            records.append(
                {
                    "venue_id": venue_id,
                    "operational_coverage_cluster_id": cluster_id,
                    "representative_candidate": representative,
                    "cluster_size": len(members),
                    "alternate_candidate_count": len(members) - 1,
                    "is_representative": venue_id == representative,
                    "cluster_min_pairwise_jaccard": minimum,
                    "cluster_jaccard_threshold": STAGE3_COVERAGE_CLUSTER_JACCARD,
                    "cluster_method": "GREEDY_COMPLETE_LINK_ALL_PAIRS_GE_THRESHOLD",
                }
            )
    result = pd.DataFrame(records, columns=_CLUSTER_COLUMNS)
    if len(result) != len(identifiers) or result["venue_id"].duplicated().any():
        raise OperationalUniverseError("coverage clustering lost or duplicated venues")
    return result.sort_values("venue_id", kind="stable").reset_index(drop=True)


def _prepare_provenance(
    mapping: pd.DataFrame,
    queue_ids: set[str],
    operational_ids: set[str],
) -> pd.DataFrame:
    _required_columns(
        mapping,
        ["physical_candidate_id", "source_anchor_id", "relationship"],
        "anchor mapping",
    )
    work = mapping.copy()
    for column in ("physical_candidate_id", "source_anchor_id", "relationship"):
        if work[column].isna().any():
            raise OperationalUniverseError(f"anchor mapping contains null {column}")
        work[column] = work[column].astype(str).str.strip()
        if work[column].eq("").any():
            raise OperationalUniverseError(f"anchor mapping contains blank {column}")
    if _contains_busproxy(work["physical_candidate_id"]):
        raise OperationalUniverseError("BUSPROXY appears on the physical side of anchor mapping")
    unknown = sorted(set(work["physical_candidate_id"]) - queue_ids)
    if unknown:
        raise OperationalUniverseError(
            f"anchor mapping references candidates outside the physical queue: {unknown[:10]}"
        )
    work = work.loc[work["physical_candidate_id"].isin(operational_ids)].copy()
    work.insert(0, "operational_venue_id", work["physical_candidate_id"].astype(str))
    work["is_fallback_relationship"] = ~work["relationship"].eq("DIRECT_SELECTED_PHYSICAL")
    sort_columns = [
        column
        for column in (
            "operational_venue_id",
            "source_anchor_id",
            "relationship_ordinal",
            "relationship",
        )
        if column in work.columns
    ]
    if sort_columns:
        work = work.sort_values(sort_columns, kind="stable").reset_index(drop=True)
    return work


def _aggregate_provenance(
    provenance: pd.DataFrame,
    queue_row: pd.Series,
    venue_id: str,
) -> dict[str, Any]:
    group = provenance.loc[provenance["operational_venue_id"].eq(venue_id)]
    if group.empty:
        relationships = [
            value
            for value in str(queue_row.get("anchor_relationships", "")).split("|")
            if value
        ]
        return {
            "operational_source_anchor_ids": "",
            "operational_anchor_relationships": "|".join(dict.fromkeys(relationships)),
            "source_anchor_count": 0,
            "busproxy_source_anchor_count": 0,
            "fallback_provenance_json": "[]",
            "provenance_kind": "OPERATIONAL_POOL_RESERVE",
        }
    source_ids = sorted(set(group["source_anchor_id"].astype(str)))
    relationships = sorted(set(group["relationship"].astype(str)))
    record_columns = [
        column
        for column in (
            "source_anchor_id",
            "source_anchor_is_busproxy",
            "source_anchor_cluster_id",
            "source_anchor_admin_code",
            "source_anchor_policies",
            "source_anchor_selection_frequency",
            "relationship",
            "relationship_ordinal",
            "same_coverage_cluster",
            "same_admin",
            "anchor_to_candidate_distance_km",
            "coverage_recompute_required",
            "required_for_busproxy_resolution",
            "priority",
        )
        if column in group.columns
    ]
    records = [
        {column: _json_scalar(row[column]) for column in record_columns}
        for row in group[record_columns].to_dict("records")
    ]
    if "source_anchor_is_busproxy" in group:
        bus_count = len(
            set(
                group.loc[
                    group["source_anchor_is_busproxy"].map(_is_verified),
                    "source_anchor_id",
                ].astype(str)
            )
        )
    else:
        bus_count = sum(value.upper().startswith(_BUS_PREFIX) for value in source_ids)
    return {
        "operational_source_anchor_ids": "|".join(source_ids),
        "operational_anchor_relationships": "|".join(relationships),
        "source_anchor_count": len(source_ids),
        "busproxy_source_anchor_count": bus_count,
        "fallback_provenance_json": _canonical_json(records),
        "provenance_kind": "SOURCE_ANCHOR_AND_FALLBACK_MAPPING",
    }


def build_verified_operational_universe(
    *,
    queue: pd.DataFrame,
    normalized_field_form: pd.DataFrame,
    field_evidence_release_ids: Iterable[str] | None = None,
    field_evidence: pd.DataFrame | None = None,
    stage3_catalog: pd.DataFrame,
    anchor_mapping: pd.DataFrame,
    stage3_hard_5000m: sparse.spmatrix,
    stage3_venue_ids: Sequence[str],
    stage3_grid: pd.DataFrame,
    stage3_grid_ids: Sequence[str],
    coordinate_equality_tolerance_degrees: float = 1e-9,
) -> OperationalCandidateUniverse:
    """Construct the verified, physical Operational Final candidate universe.

    Parameters are keyword-only to make matrix/order mistakes conspicuous.
    ``stage3_grid`` must cover exactly ``stage3_grid_ids`` and contain
    ``grid_id``, EPSG:5179 x/y, sigungu, and the three frozen population weight
    columns used by grid-pattern compression.  Extra grid columns are retained,
    notably ``is_rural_eup_myeon`` for rural/urban policy KPIs.
    ``stage3_catalog`` must be the catalog whose WGS84 coordinates generated
    the frozen matrix.  The normalized form must contain ``venue_id``,
    ``field_verified``, ``verified_address``, ``verified_latitude``, and
    ``verified_longitude``.  Supply exactly one of:
    (a) IDs already returned by the independent release gate, or (b) the raw
    ``field_evidence`` table, which this function audits against the whole queue
    before deriving those IDs.

    The coordinate tolerance only absorbs serialization noise.  It must be
    finite and no larger than 1e-6 degrees; it cannot be used to waive a material
    location change.
    """

    tolerance = float(coordinate_equality_tolerance_degrees)
    if not math.isfinite(tolerance) or tolerance < 0 or tolerance > 1e-6:
        raise OperationalUniverseError(
            "coordinate equality tolerance must be finite and in [0, 1e-6] degrees"
        )

    queue_work = _normalize_ids(queue, "venue_id", "physical queue")
    if _contains_busproxy(queue_work["venue_id"]):
        raise OperationalUniverseError("BUSPROXY appears in the physical queue")
    form = _normalize_ids(normalized_field_form, "venue_id", "normalized field form")
    _required_columns(
        form,
        [
            "field_verified",
            "verified_address",
            "verified_latitude",
            "verified_longitude",
        ],
        "normalized field form",
    )
    catalog = _normalize_ids(stage3_catalog, "venue_id", "Stage 3 catalog")
    _required_columns(
        catalog,
        ["latitude", "longitude", "coordinate_source"],
        "Stage 3 catalog",
    )
    if (field_evidence_release_ids is None) == (field_evidence is None):
        raise OperationalUniverseError(
            "supply exactly one of field_evidence_release_ids or field_evidence"
        )
    if field_evidence is not None:
        # Local import keeps the lower-level preaudited-ID API independent while
        # giving the runner a one-call evidence-backed integration boundary.
        from .field_evidence import audit_field_evidence

        evidence_audit = audit_field_evidence(
            field_evidence,
            form,
            expected_venue_ids=set(queue_work["venue_id"]),
        )
        release_ids = _normalized_release_ids(evidence_audit.release_verified_ids)
        release_evidence = evidence_audit.normalized.copy()
        audit_issues = evidence_audit.issues.copy()
        evidence_summary: dict[str, Any] = dict(evidence_audit.summary)
        evidence_summary["release_gate_mode"] = "AUDITED_FIELD_EVIDENCE_TABLE"
    else:
        assert field_evidence_release_ids is not None
        release_ids = _normalized_release_ids(field_evidence_release_ids)
        release_evidence = pd.DataFrame(
            {
                "venue_id": [
                    venue_id
                    for venue_id in queue_work["venue_id"]
                    if venue_id in release_ids
                ],
                "operational_release_verified": True,
                "release_gate_mode": "PREAUDITED_RELEASE_IDS",
            }
        )
        audit_issues = pd.DataFrame(columns=["venue_id", "field", "issue"])
        evidence_summary = {
            "release_gate_mode": "PREAUDITED_RELEASE_IDS",
            "operational_release_verified_count": len(release_ids),
        }

    queue_ids = set(queue_work["venue_id"])
    form_ids = set(form["venue_id"])
    catalog_ids = set(catalog["venue_id"])
    if form_ids != queue_ids:
        raise OperationalUniverseError(
            "normalized field form IDs do not exactly match the physical queue: "
            f"missing={sorted(queue_ids - form_ids)[:10]} "
            f"extra={sorted(form_ids - queue_ids)[:10]}"
        )
    for label, available in (
        ("physical queue", queue_ids),
        ("normalized field form", form_ids),
        ("Stage 3 catalog", catalog_ids),
    ):
        missing = sorted(release_ids - available)
        if missing:
            raise OperationalUniverseError(f"release IDs missing from {label}: {missing[:10]}")

    form_by_id = form.set_index("venue_id", drop=False)
    not_verified = sorted(
        venue_id
        for venue_id in release_ids
        if not _is_verified(form_by_id.loc[venue_id, "field_verified"])
    )
    if not_verified:
        raise OperationalUniverseError(
            "field-evidence release IDs are not field_verified in the normalized form: "
            f"{not_verified[:10]}"
        )

    matrix = _canonical_binary_csr(stage3_hard_5000m, "Stage 3 hard_5000m")
    venue_order = np.asarray([str(value).strip() for value in stage3_venue_ids], dtype=str)
    grid_order = np.asarray([str(value).strip() for value in stage3_grid_ids], dtype=str)
    if len(set(venue_order)) != len(venue_order) or any(not value for value in venue_order):
        raise OperationalUniverseError("Stage 3 venue order must be nonblank and unique")
    if len(set(grid_order)) != len(grid_order) or any(not value for value in grid_order):
        raise OperationalUniverseError("Stage 3 grid order must be nonblank and unique")
    if matrix.shape != (len(venue_order), len(grid_order)):
        raise OperationalUniverseError(
            "Stage 3 hard_5000m shape does not match venue/grid orders"
        )
    missing_matrix_rows = sorted(release_ids - set(venue_order))
    if missing_matrix_rows:
        raise OperationalUniverseError(
            f"release IDs missing from Stage 3 hard_5000m order: {missing_matrix_rows[:10]}"
        )

    grid = _normalize_ids(stage3_grid, "grid_id", "Stage 3 grid")
    _required_columns(
        grid,
        [
            "x_5179",
            "y_5179",
            "sigungu",
            "elderly_population",
            "need_weighted_population",
            "high_need_population",
        ],
        "Stage 3 grid",
    )
    if set(grid["grid_id"]) != set(grid_order) or len(grid) != len(grid_order):
        raise OperationalUniverseError("Stage 3 grid table does not exactly match grid order")
    grid = grid.set_index("grid_id", drop=False).loc[grid_order].reset_index(drop=True)
    grid_xy = grid[["x_5179", "y_5179"]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if not np.isfinite(grid_xy).all():
        raise OperationalUniverseError("Stage 3 EPSG:5179 grid coordinates are incomplete")
    grid_weights: dict[str, np.ndarray] = {}
    for column in (
        "elderly_population",
        "need_weighted_population",
        "high_need_population",
    ):
        values = pd.to_numeric(grid[column], errors="coerce").to_numpy(float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise OperationalUniverseError(
                f"Stage 3 grid contains invalid non-negative weights in {column}"
            )
        grid[column] = values
        grid_weights[column] = values

    queue_by_id = queue_work.set_index("venue_id", drop=False)
    catalog_by_id = catalog.set_index("venue_id", drop=False)
    venue_position = {venue_id: index for index, venue_id in enumerate(venue_order)}
    release_order = [venue_id for venue_id in queue_work["venue_id"] if venue_id in release_ids]

    exclusions: list[dict[str, Any]] = []
    accepted: list[dict[str, Any]] = []
    for venue_id in release_order:
        field_row = form_by_id.loc[venue_id]
        catalog_row = catalog_by_id.loc[venue_id]
        verified_address = field_row["verified_address"]
        verified_latitude = _numeric(field_row["verified_latitude"])
        verified_longitude = _numeric(field_row["verified_longitude"])
        catalog_latitude = _numeric(catalog_row["latitude"])
        catalog_longitude = _numeric(catalog_row["longitude"])
        coordinate_source = catalog_row["coordinate_source"]
        proxy_source = _is_proxy_coordinate_source(coordinate_source)
        valid_coordinates = _valid_korean_wgs84(verified_latitude, verified_longitude)
        unchanged = (
            valid_coordinates
            and _valid_korean_wgs84(catalog_latitude, catalog_longitude)
            and abs(verified_latitude - catalog_latitude) <= tolerance
            and abs(verified_longitude - catalog_longitude) <= tolerance
        )
        reasons: list[str] = []
        if not _valid_address(verified_address):
            reasons.append("MISSING_EXACT_VERIFIED_ADDRESS")
        if not valid_coordinates:
            reasons.append("MISSING_OR_INVALID_EXACT_VERIFIED_COORDINATES")
        if proxy_source and (not valid_coordinates or unchanged):
            reasons.append("CENTROID_OR_PROXY_REQUIRES_CORRECTED_EXACT_COORDINATES")
        if reasons:
            exclusions.append(
                {
                    "venue_id": venue_id,
                    "exclusion_reason": "|".join(reasons),
                    "catalog_coordinate_source": coordinate_source,
                    "stage3_catalog_latitude": catalog_latitude,
                    "stage3_catalog_longitude": catalog_longitude,
                    "verified_address": verified_address,
                    "verified_latitude": verified_latitude,
                    "verified_longitude": verified_longitude,
                }
            )
            continue
        accepted.append(
            {
                "venue_id": venue_id,
                "verified_address": str(verified_address).strip(),
                "verified_latitude": verified_latitude,
                "verified_longitude": verified_longitude,
                "stage3_catalog_latitude": catalog_latitude,
                "stage3_catalog_longitude": catalog_longitude,
                "stage3_catalog_address": catalog_row.get("address", ""),
                "stage3_catalog_coordinate_source": coordinate_source,
                "stage3_catalog_source_kind": catalog_row.get("source_kind", ""),
                "catalog_coordinates_unchanged": unchanged,
                "catalog_coordinate_is_proxy": proxy_source,
            }
        )

    accepted_frame = pd.DataFrame(accepted)
    accepted_ids = accepted_frame.get("venue_id", pd.Series(dtype=str)).astype(str).tolist()
    accepted_set = set(accepted_ids)
    provenance = _prepare_provenance(anchor_mapping, queue_ids, accepted_set)

    transformer = Transformer.from_crs(GEOGRAPHIC_CRS, METRIC_CRS, always_xy=True)
    if accepted:
        verified_lon = accepted_frame["verified_longitude"].to_numpy(float)
        verified_lat = accepted_frame["verified_latitude"].to_numpy(float)
        verified_x, verified_y = transformer.transform(verified_lon, verified_lat)
        verified_xy = np.column_stack([verified_x, verified_y]).astype(float, copy=False)
        if not np.isfinite(verified_xy).all():
            raise OperationalUniverseError("verified coordinates could not be projected to EPSG:5179")
    else:
        verified_xy = np.empty((0, 2), dtype=float)

    grid_tree = cKDTree(grid_xy, compact_nodes=True, balanced_tree=True)
    coverage_rows: list[sparse.csr_matrix] = []
    ledger_rows: list[dict[str, Any]] = []
    for index, item in accepted_frame.iterrows():
        venue_id = str(item["venue_id"])
        stage3_row = int(venue_position[venue_id])
        unchanged = bool(item["catalog_coordinates_unchanged"])
        if unchanged:
            row = matrix.getrow(stage3_row).astype(np.uint8, copy=True)
            row.data.fill(1)
            origin = "FROZEN_STAGE3_HARD_5000M_ROW"
            recomputed = False
        else:
            covered = np.asarray(
                grid_tree.query_ball_point(verified_xy[index], r=HARD_5000M_RADIUS_M),
                dtype=np.int64,
            )
            covered.sort()
            row = sparse.csr_matrix(
                (
                    np.ones(covered.size, dtype=np.uint8),
                    (np.zeros(covered.size, dtype=np.int64), covered),
                ),
                shape=(1, len(grid_order)),
                dtype=np.uint8,
            )
            origin = "RECOMPUTED_VERIFIED_EXACT_COORDINATES_EPSG5179_CKDTREE"
            recomputed = True
        row.sort_indices()
        coverage_rows.append(row)
        covered_columns = row.indices
        ledger_rows.append(
            {
                "venue_id": venue_id,
                "stage3_venue_row": stage3_row,
                "stage3_catalog_latitude": item["stage3_catalog_latitude"],
                "stage3_catalog_longitude": item["stage3_catalog_longitude"],
                "verified_latitude": item["verified_latitude"],
                "verified_longitude": item["verified_longitude"],
                "verified_x_5179": verified_xy[index, 0],
                "verified_y_5179": verified_xy[index, 1],
                "catalog_coordinates_unchanged": unchanged,
                "catalog_coordinate_is_proxy": bool(item["catalog_coordinate_is_proxy"]),
                "coverage_origin": origin,
                "coverage_recomputed": recomputed,
                "coverage_crs": METRIC_CRS,
                "coverage_radius_m": HARD_5000M_RADIUS_M,
                "covered_grid_count": int(row.nnz),
                "operational_raw_elderly_exposure": float(
                    grid_weights["elderly_population"][covered_columns].sum()
                ),
                "operational_need_weighted_exposure": float(
                    grid_weights["need_weighted_population"][covered_columns].sum()
                ),
                "operational_high_need_exposure": float(
                    grid_weights["high_need_population"][covered_columns].sum()
                ),
                "coverage_row_sha256": _binary_csr_sha256(row),
            }
        )

    operational_coverage = (
        sparse.vstack(coverage_rows, format="csr", dtype=np.uint8)
        if coverage_rows
        else sparse.csr_matrix((0, len(grid_order)), dtype=np.uint8)
    )
    operational_coverage.sort_indices()
    unique_coverage = sparse_or_union(operational_coverage)
    ledger = pd.DataFrame(ledger_rows, columns=_LEDGER_COLUMNS)

    if accepted:
        candidates = queue_by_id.loc[accepted_ids].reset_index(drop=True).copy()
        if "cluster_id" in candidates.columns:
            candidates["stage3_cluster_id"] = candidates["cluster_id"].astype(str)
        if "coverage_cluster_id" in candidates.columns:
            candidates["stage3_coverage_cluster_id"] = candidates[
                "coverage_cluster_id"
            ].astype(str)
        elif "stage3_cluster_id" in candidates.columns:
            candidates["stage3_coverage_cluster_id"] = candidates["stage3_cluster_id"]
        accepted_form = form_by_id.loc[accepted_ids].reset_index(drop=True)
        # Carry the normalized operational answers into the universe without
        # allowing pandas suffix columns to obscure which table was authoritative.
        # Queue context wins for duplicate descriptive fields; every other form
        # field (including field_verified and the eleven statuses) is retained.
        for column in accepted_form.columns:
            if column == "venue_id" or column in candidates.columns:
                continue
            candidates[column] = accepted_form[column].to_numpy()
        for column in accepted_frame.columns:
            if column == "venue_id":
                continue
            candidates[column] = accepted_frame[column].to_numpy()
        candidates["verified_x_5179"] = verified_xy[:, 0]
        candidates["verified_y_5179"] = verified_xy[:, 1]
        for column in (
            "stage3_venue_row",
            "coverage_origin",
            "coverage_recomputed",
            "coverage_crs",
            "coverage_radius_m",
            "covered_grid_count",
            "operational_raw_elderly_exposure",
            "operational_need_weighted_exposure",
            "operational_high_need_exposure",
            "coverage_row_sha256",
        ):
            candidates[column] = ledger[column].to_numpy()
        candidates["coverage_reference"] = [
            (
                f"hard_5000m:{origin}:stage3_row={row}:sha256={row_sha}"
                if not recomputed
                else f"hard_5000m:{origin}:sha256={row_sha}"
            )
            for origin, row, row_sha, recomputed in candidates[
                [
                    "coverage_origin",
                    "stage3_venue_row",
                    "coverage_row_sha256",
                    "coverage_recomputed",
                ]
            ].itertuples(index=False, name=None)
        ]
        aggregate_rows = [
            _aggregate_provenance(provenance, queue_by_id.loc[venue_id], venue_id)
            for venue_id in accepted_ids
        ]
        aggregate = pd.DataFrame(aggregate_rows)
        for column in aggregate.columns:
            candidates[column] = aggregate[column].to_numpy()
    else:
        candidates = queue_work.iloc[0:0].copy()
        for column in form.columns:
            if column != "venue_id" and column not in candidates.columns:
                candidates[column] = pd.Series(dtype=form[column].dtype)
        for column in (
            "verified_address",
            "verified_latitude",
            "verified_longitude",
            "stage3_catalog_latitude",
            "stage3_catalog_longitude",
            "stage3_catalog_address",
            "stage3_catalog_coordinate_source",
            "stage3_catalog_source_kind",
            "catalog_coordinates_unchanged",
            "catalog_coordinate_is_proxy",
            "verified_x_5179",
            "verified_y_5179",
            "stage3_venue_row",
            "coverage_origin",
            "coverage_recomputed",
            "coverage_crs",
            "coverage_radius_m",
            "covered_grid_count",
            "operational_raw_elderly_exposure",
            "operational_need_weighted_exposure",
            "operational_high_need_exposure",
            "coverage_row_sha256",
            "coverage_reference",
            "operational_source_anchor_ids",
            "operational_anchor_relationships",
            "source_anchor_count",
            "busproxy_source_anchor_count",
            "fallback_provenance_json",
            "provenance_kind",
            "stage3_cluster_id",
            "stage3_coverage_cluster_id",
        ):
            if column not in candidates.columns:
                candidates[column] = pd.Series(dtype="object")

    # Normalize the export contract after queue/form/catalog provenance has
    # been joined.  Queue latitude/longitude are frozen candidate anchors and
    # can differ from a field-corrected physical venue.  Never leave those old
    # values under conventional coordinate names in a release-facing table.
    candidates["catalog_anchor_latitude_not_for_release"] = candidates[
        "stage3_catalog_latitude"
    ]
    candidates["catalog_anchor_longitude_not_for_release"] = candidates[
        "stage3_catalog_longitude"
    ]
    candidates["catalog_anchor_coordinate_source"] = candidates[
        "stage3_catalog_coordinate_source"
    ]
    candidates["address"] = candidates["verified_address"]
    candidates["latitude"] = pd.to_numeric(
        candidates["verified_latitude"], errors="raise"
    )
    candidates["longitude"] = pd.to_numeric(
        candidates["verified_longitude"], errors="raise"
    )
    if "coordinate_source" in candidates.columns:
        candidates = candidates.drop(columns=["coordinate_source"])
    for old, new in (
        (
            "provisional_facility_latitude",
            "provisional_facility_latitude_not_for_release",
        ),
        (
            "provisional_facility_longitude",
            "provisional_facility_longitude_not_for_release",
        ),
    ):
        if old in candidates.columns:
            candidates[new] = candidates[old]
            candidates = candidates.drop(columns=[old])
        elif new not in candidates.columns:
            candidates[new] = pd.Series(dtype=float)

    overlaps = _overlap_edges(
        operational_coverage,
        accepted_ids,
        verified_xy,
        grid_weights["elderly_population"],
    )
    representative_scores = (
        candidates.set_index("venue_id")
        .get("operational_need_weighted_exposure", pd.Series(dtype=float))
        .to_dict()
    )
    clusters = _complete_link_clusters(accepted_ids, overlaps, representative_scores)
    if not clusters.empty:
        cluster_lookup = clusters.set_index("venue_id")
        for column in _CLUSTER_COLUMNS[1:]:
            candidates[column] = candidates["venue_id"].map(cluster_lookup[column])
        # Candidate-constraint consumers use ``cluster_id`` (and some adapters
        # use ``coverage_cluster_id``).  Point both at the recalculated
        # operational complete-link cluster; frozen Stage 3 values remain in
        # the explicit stage3_* provenance columns above.
        candidates["cluster_id"] = candidates["operational_coverage_cluster_id"]
        candidates["coverage_cluster_id"] = candidates[
            "operational_coverage_cluster_id"
        ]
    else:
        for column in _CLUSTER_COLUMNS[1:]:
            candidates[column] = pd.Series(dtype="object")
        candidates["cluster_id"] = pd.Series(dtype="object")
        candidates["coverage_cluster_id"] = pd.Series(dtype="object")

    exclusion_frame = pd.DataFrame(exclusions, columns=_EXCLUSION_COLUMNS)
    unique_columns = unique_coverage.indices
    unique_exposures = {
        "operational_unique_raw_elderly_exposure": float(
            grid_weights["elderly_population"][unique_columns].sum()
        ),
        "operational_unique_need_weighted_exposure": float(
            grid_weights["need_weighted_population"][unique_columns].sum()
        ),
        "operational_unique_high_need_exposure": float(
            grid_weights["high_need_population"][unique_columns].sum()
        ),
    }
    rural_column_preserved = "is_rural_eup_myeon" in grid.columns
    if rural_column_preserved:
        rural_flags = grid["is_rural_eup_myeon"].map(_binary_flag)
        if rural_flags.isna().any():
            raise OperationalUniverseError(
                "Stage 3 is_rural_eup_myeon contains a non-binary value"
            )
        rural = rural_flags.to_numpy(bool)
        unique_exposures.update(
            {
                "operational_unique_rural_elderly_exposure": float(
                    grid_weights["elderly_population"][unique_columns][
                        rural[unique_columns]
                    ].sum()
                ),
                "operational_unique_urban_elderly_exposure": float(
                    grid_weights["elderly_population"][unique_columns][
                        ~rural[unique_columns]
                    ].sum()
                ),
            }
        )
    hashes = {
        "source_stage3_hard_5000m_canonical_sha256": _binary_csr_sha256(matrix),
        "source_stage3_venue_order_sha256": _order_sha256(venue_order, "STAGE3_VENUE"),
        "source_stage3_grid_order_sha256": _order_sha256(grid_order, "STAGE3_GRID"),
        "operational_hard_5000m_canonical_sha256": _binary_csr_sha256(operational_coverage),
        "operational_venue_order_sha256": _order_sha256(accepted_ids, "VENUE"),
        "operational_unique_coverage_sha256": _binary_csr_sha256(unique_coverage),
        "coverage_origin_ledger_sha256": _frame_sha256(ledger, "COVERAGE_ORIGIN_LEDGER"),
        "source_anchor_provenance_sha256": _frame_sha256(provenance, "SOURCE_ANCHOR_PROVENANCE"),
        "coverage_overlap_edges_sha256": _frame_sha256(overlaps, "COVERAGE_OVERLAP_EDGES"),
        "coverage_clusters_sha256": _frame_sha256(clusters, "COVERAGE_CLUSTERS"),
        "exclusions_sha256": _frame_sha256(exclusion_frame, "EXCLUSIONS"),
        "aligned_grid_sha256": _frame_sha256(grid, "ALIGNED_GRID"),
        "operational_candidates_sha256": _frame_sha256(candidates, "CANDIDATES"),
        "release_evidence_sha256": _frame_sha256(release_evidence, "RELEASE_EVIDENCE"),
        "audit_issues_sha256": _frame_sha256(audit_issues, "AUDIT_ISSUES"),
    }
    hashes["operational_universe_sha256"] = sha256(
        _canonical_json(hashes).encode("utf-8")
    ).hexdigest()

    result = OperationalCandidateUniverse(
        candidates=candidates.reset_index(drop=True),
        coverage=operational_coverage,
        venue_ids=np.asarray(accepted_ids, dtype=str),
        grid_ids=grid_order,
        grid=grid,
        coverage_origin_ledger=ledger,
        source_anchor_provenance=provenance,
        release_evidence=release_evidence,
        audit_issues=audit_issues,
        coverage_overlap_edges=overlaps,
        coverage_clusters=clusters,
        unique_coverage=unique_coverage,
        exclusions=exclusion_frame,
        artifact_sha256=hashes,
        summary={
            "field_evidence_release_id_count": len(release_ids),
            "operational_candidate_count": len(accepted_ids),
            "excluded_release_id_count": len(exclusion_frame),
            "busproxy_operational_candidate_count": 0,
            "frozen_stage3_row_count": int((~ledger["coverage_recomputed"]).sum()) if len(ledger) else 0,
            "recomputed_coverage_row_count": int(ledger["coverage_recomputed"].sum()) if len(ledger) else 0,
            "coverage_cluster_count": int(clusters["operational_coverage_cluster_id"].nunique()) if len(clusters) else 0,
            "coverage_cluster_jaccard_threshold": STAGE3_COVERAGE_CLUSTER_JACCARD,
            "coverage_cluster_threshold_source": (
                "configs/model_v1/stage3_spatial.yaml::overlap.cluster_jaccard_min"
            ),
            "overlap_output_min_population_jaccard": (
                STAGE3_OVERLAP_OUTPUT_MIN_POPULATION_JACCARD
            ),
            "overlap_output_min_jaccard": STAGE3_OVERLAP_OUTPUT_MIN_JACCARD,
            "overlap_candidate_pair_center_distance_max_m": (
                STAGE3_PAIR_CENTER_DISTANCE_MAX_M
            ),
            "unique_covered_grid_count": int(unique_coverage.nnz),
            "coverage_radius_m": HARD_5000M_RADIUS_M,
            "coverage_crs": METRIC_CRS,
            "field_evidence_gate": evidence_summary,
            "is_rural_eup_myeon_grid_column_preserved": rural_column_preserved,
            **unique_exposures,
        },
    )
    result.validate()
    return result


__all__ = [
    "HARD_5000M_RADIUS_M",
    "METRIC_CRS",
    "OperationalCandidateUniverse",
    "OperationalUniverseError",
    "STAGE3_COVERAGE_CLUSTER_JACCARD",
    "STAGE3_OVERLAP_OUTPUT_MIN_JACCARD",
    "STAGE3_OVERLAP_OUTPUT_MIN_POPULATION_JACCARD",
    "STAGE3_PAIR_CENTER_DISTANCE_MAX_M",
    "build_verified_operational_universe",
    "sparse_or_union",
]
