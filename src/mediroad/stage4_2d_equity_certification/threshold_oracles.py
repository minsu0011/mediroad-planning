"""Fail-closed threshold oracles for the frozen Stage 4.2D equity front.

The ordinary Stage 4.2 certifier is useful for finding incumbents and closing a
MIP gap.  This module has a narrower job: rebuild two *decision* models and
accept a certificate only when a solver proves that the frozen improvement
threshold is infeasible.  Every transformation used by the proof model is an
outward relaxation, so infeasibility of the transported model is also
infeasibility of the policy model.

The functions are intentionally dependency-injectable.  Unit tests can use
small synthetic formulations and solver doubles, while official runs use the
native HiGHS and PySCIPOpt bindings in isolated spawned processes.
"""

from __future__ import annotations

import hashlib
import json
import math
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
from scipy import sparse

from mediroad.stage4_2_certification.backends import HighsBackend
from mediroad.stage4_2_certification.formulation import CertificationFormulation
from mediroad.stage4_2_certification.types import Floor, LinearMipModel

from .model_utils import append_cuts
from .strengthening import generate_anchored_submodular_cuts, metric_targets
from .types import CutRecord, EvidenceSeed, MetricCertificate


OFFICIAL_MIN_TOTAL_POPULATION_FLOOR = 167_282.84413449097
OFFICIAL_MIN_INCUMBENT = 0.5252179790669655
OFFICIAL_MIN_RETAINED_FLOOR = 0.525217
OFFICIAL_MIN_THRESHOLD = 0.527844
OFFICIAL_MIN_SELECTION_DIGEST = (
    "5440b1fb5d13530fcf5150c982e06f575182606490d3a7158549e7fd7ddf596e"
)

OFFICIAL_HIGH_INCUMBENT = 11_609.13711010601
OFFICIAL_HIGH_THRESHOLD = 11_667.182795656541
OFFICIAL_HIGH_SELECTION_DIGEST = (
    "cb3ef97672fc0d61f54c5c911783b033033d2570d86c47b5ca0bfdbf333d6435"
)
OFFICIAL_HIGH_CUT_SEED_COUNT = 26
OFFICIAL_HIGH_CUT_SEED_DIGEST = (
    "1561e262a0ab427db2b1f1e44b5e0cc550c699b1c0882297ea164e1200f8eecc"
)
OFFICIAL_SCIP_SEEDS = (11, 23, 42, 77, 101, 131, 197, 257)
OFFICIAL_POLICY_RELATIVE_GAP = 0.005
OFFICIAL_HIGHS_VERSION = "1.15.1"
OFFICIAL_PYSCIPOPT_VERSION = "6.2.1"
OFFICIAL_SCIP_VERSION_COMPONENTS = (10, 0, 2)
OFFICIAL_SCIP_TIME_LIMIT_SEC = 900.0
SCIP_INFINITY_SENTINEL_ABS_MIN = 1.0e19

OFFICIAL_MIN_HIGHS_OPTIONS: dict[str, Any] = {
    "output_flag": True,
    "log_to_console": False,
    "presolve": "on",
    "parallel": "on",
    "threads": 8,
    "time_limit": 900.0,
    "mip_rel_gap": 0.0,
    "mip_abs_gap": 0.0,
    "random_seed": 104752,
    "mip_detect_symmetry": True,
    "mip_allow_restart": True,
    "mip_heuristic_effort": 0.02,
    "mip_max_start_nodes": 5000,
    "mip_min_logging_interval": 5.0,
    "mip_pool_soft_limit": 20000,
    "mip_pscost_minreliable": 4,
    "mip_min_cliquetable_entries_for_parallelism": 10000,
    "mip_lp_age_limit": 10,
}


@dataclass
class ThresholdOracleModel:
    """A rebuilt exact model and the outward-relaxed model sent to a solver."""

    metric: str
    backend: str
    exact_model: LinearMipModel
    model: LinearMipModel
    threshold: float
    incumbent_value: float
    policy_relative_gap: float
    retained_floor: float
    total_population_floor: float
    min_sigungu_floor: float | None
    incumbent_selection: tuple[int, ...]
    selection_digest: str
    exact_model_digest: str
    proof_model_digest: str
    universe_digest: str
    cut_digest: str | None = None
    cut_seed_digest: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MpsArtifact:
    path: Path
    sha256: str
    size_bytes: int
    size_mib: float
    rows: int
    cols: int
    nnz: int
    transport_sufficient: bool
    transport_audit: dict[str, Any] = field(default_factory=dict)


@dataclass
class ThresholdOracleResult:
    metric: str
    backend: str
    status: str
    certified_threshold_infeasible: bool
    counterexample_found: bool
    threshold: float
    incumbent_value: float
    policy_relative_gap: float
    retained_floor: float
    incumbent_selection: tuple[int, ...]
    selection_digest: str
    best_bound: float | None
    relative_gap: float | None
    wall_time_sec: float
    node_count: int | None
    solver_version: str
    exact_model_digest: str
    proof_model_digest: str
    universe_digest: str
    cut_digest: str | None
    mps_sha256: str | None
    requested_options: dict[str, Any] = field(default_factory=dict)
    readback_options: dict[str, Any] = field(default_factory=dict)
    log_paths: list[Path] = field(default_factory=list)
    outcomes: list[dict[str, Any]] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)


def _canonical_indices(indices: Sequence[int] | np.ndarray) -> list[int]:
    raw = np.asarray(indices)
    if raw.ndim != 1 or raw.dtype.kind not in {"i", "u"}:
        raise ValueError("Selection must be a one-dimensional integer array")
    values = [int(value) for value in raw.tolist()]
    if len(values) != len(set(values)):
        raise ValueError("Selection contains duplicate candidate indices")
    return sorted(values)


def selection_digest(indices: Sequence[int] | np.ndarray) -> str:
    """Hash an order-independent selection using the frozen compact-JSON rule."""

    payload = json.dumps(_canonical_indices(indices), separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def seed_set_digest(seeds: Iterable[EvidenceSeed]) -> str:
    """Hash a multiset of selections while preserving repeated anchors."""

    canonical = [_canonical_indices(seed.selected_indices) for seed in seeds]
    canonical.sort()
    payload = json.dumps(canonical, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _metric_key(metric: str) -> str:
    mapping = {
        "total_population": "unique_elderly_population",
        "min_sigungu_coverage": "min_sigungu_coverage_ratio",
        "high_need_population": "high_need_population",
        "need_weighted": "need_weighted_population",
        "cost": "cost",
    }
    try:
        return mapping[metric]
    except KeyError as exc:
        raise ValueError(f"Unsupported threshold metric: {metric}") from exc


def _metric_tolerance(metric: str, value: float) -> float:
    if metric == "min_sigungu_coverage":
        return 2e-12
    # Recomputed metrics use the sealed float arrays and deterministic coverage
    # mask; this is a semantic check, not a solver feasibility tolerance.
    return 1e-8


def _require_exact_float(name: str, observed: float, expected: float) -> None:
    if float(observed).hex() != float(expected).hex():
        raise RuntimeError(
            f"Frozen {name} changed: expected={float(expected).hex()}, "
            f"observed={float(observed).hex()}"
        )


def _enforce_official_min_contract(
    formulation: CertificationFormulation,
    *,
    total_population_floor: float,
    threshold: float,
    incumbent_value: float,
    retained_floor: float,
    policy_relative_gap: float,
    expected_selection_digest: str,
) -> Mapping[str, int] | None:
    if str(formulation.candidate_set) != "top3":
        return None
    if formulation.nx != 452 or formulation.ny != 5242:
        raise RuntimeError("Frozen Top3 universe dimensions changed")
    _require_exact_float(
        "min total_population_floor",
        total_population_floor,
        OFFICIAL_MIN_TOTAL_POPULATION_FLOOR,
    )
    _require_exact_float("min threshold", threshold, OFFICIAL_MIN_THRESHOLD)
    _require_exact_float("min incumbent", incumbent_value, OFFICIAL_MIN_INCUMBENT)
    _require_exact_float(
        "min incoming retained_floor", retained_floor, OFFICIAL_MIN_TOTAL_POPULATION_FLOOR
    )
    _require_exact_float(
        "min policy_relative_gap", policy_relative_gap, OFFICIAL_POLICY_RELATIVE_GAP
    )
    if expected_selection_digest != OFFICIAL_MIN_SELECTION_DIGEST:
        raise RuntimeError("Frozen min selection digest argument changed")
    return {"rows": 5499, "cols": 5694, "nnz": 159935}


def _enforce_official_high_contract(
    formulation: CertificationFormulation,
    *,
    total_population_floor: float,
    min_sigungu_floor: float,
    threshold: float,
    incumbent_value: float,
    retained_floor: float,
    policy_relative_gap: float,
    expected_selection_digest: str,
    expected_cut_seed_count: int,
    expected_cut_seed_digest: str,
    anchor_sizes: Sequence[int],
    max_cuts: int,
) -> tuple[Mapping[str, int] | None, int | None, int | None]:
    if str(formulation.candidate_set) != "top3":
        return None, None, None
    if formulation.nx != 452 or formulation.ny != 5242:
        raise RuntimeError("Frozen Top3 universe dimensions changed")
    _require_exact_float(
        "high total_population_floor",
        total_population_floor,
        OFFICIAL_MIN_TOTAL_POPULATION_FLOOR,
    )
    _require_exact_float("high min_sigungu_floor", min_sigungu_floor, OFFICIAL_MIN_RETAINED_FLOOR)
    _require_exact_float("high threshold", threshold, OFFICIAL_HIGH_THRESHOLD)
    _require_exact_float("high incumbent", incumbent_value, OFFICIAL_HIGH_INCUMBENT)
    _require_exact_float("high retained_floor", retained_floor, OFFICIAL_MIN_RETAINED_FLOOR)
    _require_exact_float(
        "high policy_relative_gap", policy_relative_gap, OFFICIAL_POLICY_RELATIVE_GAP
    )
    if expected_selection_digest != OFFICIAL_HIGH_SELECTION_DIGEST:
        raise RuntimeError("Frozen high selection digest argument changed")
    if int(expected_cut_seed_count) != OFFICIAL_HIGH_CUT_SEED_COUNT:
        raise RuntimeError("Frozen high cut seed count argument changed")
    if expected_cut_seed_digest != OFFICIAL_HIGH_CUT_SEED_DIGEST:
        raise RuntimeError("Frozen high cut seed digest argument changed")
    if tuple(anchor_sizes) != tuple(range(21)) or int(max_cuts) != 4096:
        raise RuntimeError("Frozen high cut anchor/max-cut controls changed")
    return {"rows": 9377, "cols": 5694, "nnz": 1244380}, 3877, 3113


def validate_pinned_selection(
    formulation: CertificationFormulation,
    selected_indices: Sequence[int] | np.ndarray,
    *,
    expected_digest: str,
    metric: str,
    expected_value: float,
    total_population_floor: float,
    min_sigungu_floor: float | None = None,
    threshold: float | None = None,
) -> dict[str, Any]:
    """Validate a frozen incumbent against current semantics and its SHA pin."""

    selected = np.asarray(selected_indices)
    # Raw type, dimensionality, count, range, and uniqueness are checked before
    # hashing.  This prevents a malformed array from being normalized into a
    # digest that happens to match a valid selection.
    _canonical_indices(selected)
    valid, reason = formulation.validate_selection(selected)
    if not valid:
        raise RuntimeError(f"Pinned selection is invalid in the current universe: {reason}")
    observed_digest = selection_digest(selected)
    if observed_digest != str(expected_digest):
        raise RuntimeError(
            "Pinned selection digest changed: "
            f"expected={expected_digest}, observed={observed_digest}"
        )
    metrics = formulation.metrics(selected.astype(np.int64, copy=False))
    metric_key = _metric_key(metric)
    observed_value = float(metrics[metric_key])
    tolerance = _metric_tolerance(metric, expected_value)
    if not math.isclose(observed_value, float(expected_value), rel_tol=0.0, abs_tol=tolerance):
        raise RuntimeError(
            f"Pinned {metric} value changed: expected={expected_value:.17g}, "
            f"observed={observed_value:.17g}, tolerance={tolerance:.3g}"
        )
    total_value = float(metrics["unique_elderly_population"])
    total_tolerance = _metric_tolerance("total_population", total_population_floor)
    if total_value < float(total_population_floor) - total_tolerance:
        raise RuntimeError("Pinned selection violates the total-population floor")
    if min_sigungu_floor is not None:
        min_value = float(metrics["min_sigungu_coverage_ratio"])
        if min_value < float(min_sigungu_floor) - _metric_tolerance(
            "min_sigungu_coverage", min_sigungu_floor
        ):
            raise RuntimeError("Pinned selection violates the retained min-sigungu floor")
    if threshold is not None and observed_value >= float(threshold) - tolerance:
        raise RuntimeError(
            "Pinned incumbent already reaches the alleged improvement threshold; "
            "the threshold contract is contradictory"
        )
    return {
        "selection_digest": observed_digest,
        "selection_size": int(selected.size),
        "metric": metric,
        "metric_key": metric_key,
        "metric_value": observed_value,
        "metrics": metrics,
    }


def validate_cut_seed_set(
    formulation: CertificationFormulation,
    seeds: Sequence[EvidenceSeed],
    *,
    expected_count: int,
    expected_digest: str,
) -> str:
    if not seeds:
        raise RuntimeError("The high-need cut anchor seed set is missing")
    if len(seeds) != int(expected_count):
        raise RuntimeError(
            f"Cut seed count changed: expected={expected_count}, observed={len(seeds)}"
        )
    for index, seed in enumerate(seeds):
        valid, reason = formulation.validate_selection(np.asarray(seed.selected_indices))
        if not valid:
            raise RuntimeError(f"Cut seed {index} ({seed.source}) is invalid: {reason}")
    observed = seed_set_digest(seeds)
    if observed != str(expected_digest):
        raise RuntimeError(
            f"Cut seed digest changed: expected={expected_digest}, observed={observed}"
        )
    return observed


def _dimension_dict(model: LinearMipModel) -> dict[str, int]:
    return {"rows": model.n_row, "cols": model.n_col, "nnz": int(model.A.nnz)}


def _require_dimensions(
    model: LinearMipModel, expected: Mapping[str, int] | None, *, label: str
) -> None:
    if expected is None:
        return
    observed = _dimension_dict(model)
    wanted = {key: int(expected[key]) for key in ("rows", "cols", "nnz")}
    if observed != wanted:
        raise RuntimeError(f"{label} dimensions changed: expected={wanted}, observed={observed}")


def _canonical_model_digest(model: LinearMipModel) -> str:
    try:
        from .canonical import canonical_model_digest
    except ImportError as exc:  # pragma: no cover - installation contract
        raise RuntimeError("The canonical proof module is required for official oracles") from exc
    value = canonical_model_digest(model)
    if not isinstance(value, str) or len(value) != 64:
        raise RuntimeError("canonical_model_digest returned an invalid SHA-256 value")
    return value


def _canonical_universe_digest(
    formulation: CertificationFormulation, model: LinearMipModel
) -> str:
    try:
        from .canonical import canonical_universe_digest
    except ImportError as exc:  # pragma: no cover - installation contract
        raise RuntimeError("The canonical proof module is required for official oracles") from exc
    candidate_ids = formulation.candidates["venue_id"].astype(str).tolist()
    pattern_ids = list(range(formulation.ny))
    semantic_ids = {
        "candidate_admin_code": formulation.candidates["admin_code"].astype(str).tolist(),
        "candidate_sigungu": formulation.candidates["sigungu"].astype(str).tolist(),
        "candidate_cluster": formulation.candidates["cluster_id"].fillna(
            formulation.candidates["venue_id"]
        ).astype(str).tolist(),
        "pattern_sigungu": np.asarray(formulation.pattern_sigungu, dtype=str).tolist(),
    }
    value = canonical_universe_digest(
        model,
        candidate_ids=candidate_ids,
        pattern_ids=pattern_ids,
        semantic_ids=semantic_ids,
    )
    if not isinstance(value, str) or len(value) != 64:
        raise RuntimeError("canonical_universe_digest returned an invalid SHA-256 value")
    return value


def _outward_relax(model: LinearMipModel) -> tuple[LinearMipModel, dict[str, Any]]:
    try:
        from .canonical import make_outward_relaxation
    except ImportError as exc:  # pragma: no cover - installation contract
        raise RuntimeError("The canonical proof module is required for official oracles") from exc
    relaxation = make_outward_relaxation(
        model,
        absolute_margin=2e-6,
        relative_margin=1e-12,
        ulps=1,
    )
    relaxed = getattr(relaxation, "relaxed_model", None)
    manifest = getattr(relaxation, "manifest", None)
    if not isinstance(relaxed, LinearMipModel) or not isinstance(manifest, Mapping):
        raise RuntimeError("Canonical outward relaxation returned an invalid payload")
    # Every column bound in these decision models is an exactly transportable
    # integer (0 or 1).  Keeping it equal to the exact model is still an outward
    # relaxation, and preserves BINARY rather than general INTEGER typing when
    # HiGHS serializes the all-binary high-need model for SCIP.
    relaxed.col_lower = np.asarray(model.col_lower, dtype=np.float64).copy()
    relaxed.col_upper = np.asarray(model.col_upper, dtype=np.float64).copy()
    manifest = dict(manifest)
    manifest["exact_column_bounds_restored"] = True
    families = dict(manifest.get("families", {}))
    families["col_lower"] = {
        "direction": "exact_preserved",
        "reason": "integer-domain/MPS-binary transport",
        "realized_margin_min": 0.0,
        "realized_margin_max": 0.0,
    }
    families["col_upper"] = dict(families["col_lower"])
    manifest["families"] = families
    relaxed.validate()
    manifest["relaxed_model_sha256"] = _canonical_model_digest(relaxed)
    from .canonical import audit_relaxing_transport

    audit = audit_relaxing_transport(model, relaxed, proof_kind="infeasibility")
    if not bool(getattr(audit, "sufficient", False)):
        payload = audit.to_dict() if callable(getattr(audit, "to_dict", None)) else asdict(audit)
        raise RuntimeError(f"Exact-to-proof outward relaxation audit failed: {payload}")
    manifest["exact_to_proof_transport_audit"] = audit.to_dict()
    return relaxed, manifest


def _cut_digest(cuts: Sequence[CutRecord]) -> str:
    records: list[dict[str, Any]] = []
    for cut in cuts:
        records.append(
            {
                "name": cut.name,
                "kind": cut.kind,
                "indices": [int(v) for v in np.asarray(cut.indices).tolist()],
                "values_hex": [float(v).hex() for v in np.asarray(cut.values).tolist()],
                "lower_hex": float(cut.lower).hex(),
                "upper_hex": float(cut.upper).hex(),
                "metric": cut.metric,
                "anchor_size": cut.anchor_size,
                "source": cut.source,
                "outward_safe": bool(cut.metadata.get("outward_safe", False)),
            }
        )
    payload = json.dumps(records, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _binary_coverage_theorem_guard(
    formulation: CertificationFormulation, model: LinearMipModel
) -> dict[str, Any]:
    """Verify the hypotheses allowing continuous coverage variables to be binary.

    With binary venue variables, non-negative benefit floors, and one-link rows
    ``y_p <= sum_i x_i``, every feasible projected venue plan admits the maximal
    covered assignment ``y_p=1``.  Making every coverage variable binary is
    therefore exact in the venue projection for this high-need decision model.
    """

    model.validate()
    if model.objective_name != "high_need_population" or model.sense != "max":
        raise RuntimeError("Binary-coverage theorem is restricted to the high-need MAX model")
    if model.extra_slices:
        raise RuntimeError("Binary-coverage theorem does not allow auxiliary variable blocks")
    if model.x_slice != slice(0, formulation.nx) or model.y_slice != slice(
        formulation.nx, formulation.nx + formulation.ny
    ):
        raise RuntimeError("Unexpected x/y layout for binary-coverage theorem")
    if model.n_col != formulation.nx + formulation.ny:
        raise RuntimeError("Unexpected non-coverage columns in high-need theorem model")
    if not bool(model.metadata.get("one_link_pattern_formulation", False)):
        raise RuntimeError("High-need model is not marked as a one-link formulation")
    if np.any(model.integrality[model.x_slice] != 1):
        raise RuntimeError("Candidate variables are not all binary")
    if np.any(model.col_lower[model.y_slice] != 0.0) or np.any(
        model.col_upper[model.y_slice] > 1.0
    ):
        raise RuntimeError("Coverage variables are outside [0,1]")
    if np.any(~np.isfinite(formulation.population)) or np.any(formulation.population < 0.0):
        raise RuntimeError("Population weights violate theorem non-negativity")
    if np.any(~np.isfinite(formulation.high_need)) or np.any(formulation.high_need < 0.0):
        raise RuntimeError("High-need weights violate theorem non-negativity")
    row_names = list(model.metadata.get("row_names", []))
    if len(row_names) != model.n_row:
        raise RuntimeError("High-need theorem requires a complete row-name ledger")
    if np.any(model.c[model.x_slice] != 0.0) or not np.array_equal(
        model.c[model.y_slice], np.asarray(formulation.high_need, dtype=np.float64)
    ):
        raise RuntimeError("High-need objective vector violates theorem hypotheses")
    A = model.A.tocsr(copy=False)
    y_start, y_stop = model.y_slice.start, model.y_slice.stop
    observed_cover_patterns: set[int] = set()
    for row_index, row_name in enumerate(row_names):
        start, stop = A.indptr[row_index], A.indptr[row_index + 1]
        cols = A.indices[start:stop]
        values = A.data[start:stop]
        touches_y = bool(np.any((cols >= y_start) & (cols < y_stop)))
        name = str(row_name)
        if name.startswith("cover_link::"):
            try:
                pattern = int(name.rsplit("::", 1)[1])
            except (ValueError, IndexError) as exc:
                raise RuntimeError(f"Malformed cover-link row name: {name}") from exc
            if pattern < 0 or pattern >= formulation.ny or pattern in observed_cover_patterns:
                raise RuntimeError(f"Duplicate/out-of-range cover-link pattern: {pattern}")
            expected: dict[int, float] = {y_start + pattern: 1.0}
            expected.update(
                {int(candidate): -1.0 for candidate in formulation.pattern_coverers[pattern]}
            )
            observed = {int(column): float(value) for column, value in zip(cols, values)}
            if observed != expected:
                raise RuntimeError(f"Cover-link coefficients changed for pattern {pattern}")
            if not math.isinf(float(model.row_lower[row_index])) or model.row_lower[
                row_index
            ] > 0.0 or float(model.row_upper[row_index]) != 0.0:
                raise RuntimeError(f"Cover-link bounds changed for pattern {pattern}")
            observed_cover_patterns.add(pattern)
        elif name.startswith("floor::"):
            if not touches_y:
                continue
            if np.any(cols < y_start) or np.any(cols >= y_stop):
                raise RuntimeError(f"Benefit floor mixes non-coverage columns: {name}")
            if np.any(values < 0.0) or np.any(~np.isfinite(values)):
                raise RuntimeError(f"Benefit floor has a negative/non-finite coefficient: {name}")
            if not math.isfinite(float(model.row_lower[row_index])) or not math.isinf(
                float(model.row_upper[row_index])
            ) or model.row_upper[row_index] < 0.0:
                raise RuntimeError(f"Benefit floor is not a lower superlevel row: {name}")
        elif touches_y:
            raise RuntimeError(f"Unrecognized y-row invalidates coverage theorem: {row_name}")
    expected_cover_patterns = {
        pattern for pattern, coverers in enumerate(formulation.pattern_coverers) if len(coverers)
    }
    if observed_cover_patterns != expected_cover_patterns:
        raise RuntimeError("Cover-link row family is incomplete")
    return {
        "theorem": "BINARY_COVERAGE_MAXIMAL_EXTENSION",
        "projected_venue_feasible_region_preserved": True,
        "hypotheses": {
            "candidate_variables_binary": True,
            "coverage_bounds_zero_one": True,
            "one_link_rows_only": True,
            "benefit_weights_nonnegative": True,
            "no_auxiliary_columns": True,
        },
    }


def build_min_highs_oracle(
    formulation: CertificationFormulation,
    *,
    total_population_floor: float,
    threshold: float,
    incumbent_value: float,
    retained_floor: float,
    incumbent_selection: Sequence[int] | np.ndarray,
    expected_selection_digest: str,
    policy_relative_gap: float | None = None,
    expected_dimensions: Mapping[str, int] | None = None,
) -> ThresholdOracleModel:
    if policy_relative_gap is None:
        policy_relative_gap = (
            OFFICIAL_POLICY_RELATIVE_GAP
            if str(formulation.candidate_set) == "top3"
            else (float(threshold) - float(incumbent_value))
            / max(abs(float(incumbent_value)), 1e-12)
        )
    official_dimensions = _enforce_official_min_contract(
        formulation,
        total_population_floor=total_population_floor,
        threshold=threshold,
        incumbent_value=incumbent_value,
        retained_floor=retained_floor,
        policy_relative_gap=policy_relative_gap,
        expected_selection_digest=expected_selection_digest,
    )
    if official_dimensions is not None:
        if expected_dimensions is not None and {
            key: int(expected_dimensions[key]) for key in official_dimensions
        } != dict(official_dimensions):
            raise RuntimeError("Caller-supplied min dimensions contradict the frozen contract")
        expected_dimensions = official_dimensions
    pin = validate_pinned_selection(
        formulation,
        incumbent_selection,
        expected_digest=expected_selection_digest,
        metric="min_sigungu_coverage",
        expected_value=incumbent_value,
        total_population_floor=total_population_floor,
        threshold=threshold,
    )
    exact_model = formulation.build(
        objective_name="feasibility",
        sense="min",
        floors=[Floor("total_population", "max", float(total_population_floor))],
        threshold=Floor("min_sigungu_coverage", "max", float(threshold)),
        name="stage42d__official__min_sigungu_threshold",
    )
    _require_dimensions(exact_model, expected_dimensions, label="Min threshold oracle")
    if np.any(exact_model.c != 0.0):
        raise RuntimeError("Min threshold oracle must have a zero feasibility objective")
    if np.any(exact_model.integrality[exact_model.x_slice] != 1) or np.any(
        exact_model.integrality[exact_model.y_slice] != 0
    ):
        raise RuntimeError("Min oracle must have binary x and continuous y variables")
    proof_model, relaxation_manifest = _outward_relax(exact_model)
    _require_dimensions(proof_model, expected_dimensions, label="Relaxed min threshold oracle")
    return ThresholdOracleModel(
        metric="min_sigungu_coverage",
        backend="HIGHS",
        exact_model=exact_model,
        model=proof_model,
        threshold=float(threshold),
        incumbent_value=float(incumbent_value),
        policy_relative_gap=float(policy_relative_gap),
        retained_floor=float(retained_floor),
        total_population_floor=float(total_population_floor),
        min_sigungu_floor=None,
        incumbent_selection=tuple(int(v) for v in np.asarray(incumbent_selection).tolist()),
        selection_digest=pin["selection_digest"],
        exact_model_digest=_canonical_model_digest(exact_model),
        proof_model_digest=_canonical_model_digest(proof_model),
        universe_digest=_canonical_universe_digest(formulation, exact_model),
        metadata={
            "pinned_selection": pin,
            "proof_relaxation": relaxation_manifest,
            "coverage_columns_continuous": True,
            "candidate_columns_binary": True,
            "next_stage_min_sigungu_floor": (
                OFFICIAL_MIN_RETAINED_FLOOR
                if str(formulation.candidate_set) == "top3"
                else None
            ),
            "expected_dimensions": dict(expected_dimensions or _dimension_dict(exact_model)),
        },
    )


def build_high_scip_oracle(
    formulation: CertificationFormulation,
    *,
    total_population_floor: float,
    min_sigungu_floor: float,
    threshold: float,
    incumbent_value: float,
    retained_floor: float,
    incumbent_selection: Sequence[int] | np.ndarray,
    expected_selection_digest: str,
    cut_seeds: Sequence[EvidenceSeed],
    expected_cut_seed_count: int,
    expected_cut_seed_digest: str,
    policy_relative_gap: float | None = None,
    anchor_sizes: Iterable[int] = range(21),
    max_cuts: int = 4096,
    expected_cut_count: int | None = None,
    expected_anchor_count: int | None = None,
    expected_dimensions: Mapping[str, int] | None = None,
) -> ThresholdOracleModel:
    if policy_relative_gap is None:
        policy_relative_gap = (
            OFFICIAL_POLICY_RELATIVE_GAP
            if str(formulation.candidate_set) == "top3"
            else (float(threshold) - float(incumbent_value))
            / max(abs(float(incumbent_value)), 1e-12)
        )
    anchor_sizes = tuple(int(value) for value in anchor_sizes)
    official_dimensions, official_cut_count, official_anchor_count = _enforce_official_high_contract(
        formulation,
        total_population_floor=total_population_floor,
        min_sigungu_floor=min_sigungu_floor,
        threshold=threshold,
        incumbent_value=incumbent_value,
        retained_floor=retained_floor,
        policy_relative_gap=policy_relative_gap,
        expected_selection_digest=expected_selection_digest,
        expected_cut_seed_count=expected_cut_seed_count,
        expected_cut_seed_digest=expected_cut_seed_digest,
        anchor_sizes=anchor_sizes,
        max_cuts=max_cuts,
    )
    if official_dimensions is not None:
        if expected_dimensions is not None and {
            key: int(expected_dimensions[key]) for key in official_dimensions
        } != dict(official_dimensions):
            raise RuntimeError("Caller-supplied high dimensions contradict the frozen contract")
        if expected_cut_count is not None and int(expected_cut_count) != int(official_cut_count):
            raise RuntimeError("Caller-supplied high cut count contradicts the frozen contract")
        if expected_anchor_count is not None and int(expected_anchor_count) != int(
            official_anchor_count
        ):
            raise RuntimeError("Caller-supplied high anchor count contradicts the frozen contract")
        expected_dimensions = official_dimensions
        expected_cut_count = official_cut_count
        expected_anchor_count = official_anchor_count
    pin = validate_pinned_selection(
        formulation,
        incumbent_selection,
        expected_digest=expected_selection_digest,
        metric="high_need_population",
        expected_value=incumbent_value,
        total_population_floor=total_population_floor,
        min_sigungu_floor=min_sigungu_floor,
        threshold=threshold,
    )
    seeds_digest = validate_cut_seed_set(
        formulation,
        cut_seeds,
        expected_count=expected_cut_seed_count,
        expected_digest=expected_cut_seed_digest,
    )
    base = formulation.build(
        objective_name="high_need_population",
        sense="max",
        floors=[
            Floor("total_population", "max", float(total_population_floor)),
            Floor("min_sigungu_coverage", "max", float(min_sigungu_floor)),
        ],
        threshold=Floor("high_need_population", "max", float(threshold)),
        name="stage42d__official__high_need_threshold",
    )
    theorem = _binary_coverage_theorem_guard(formulation, base)
    base.integrality[base.y_slice] = 1
    base.metadata.setdefault("stage42d", {})["binary_coverage_theorem"] = theorem
    base.validate()

    targets = metric_targets(
        formulation,
        total_population_floor=float(total_population_floor),
        min_sigungu_floor=float(min_sigungu_floor),
        high_need_target=float(threshold),
    )
    try:
        cuts, cut_info = generate_anchored_submodular_cuts(
            formulation,
            targets,
            cut_seeds,
            anchor_sizes=anchor_sizes,
            max_cuts=int(max_cuts),
            outward_safe=True,
        )
    except TypeError as exc:
        if "outward_safe" in str(exc):
            raise RuntimeError(
                "Official high-need proof requires an outward_safe strengthening API"
            ) from exc
        raise
    if cut_info.get("outward_safe") is not True:
        raise RuntimeError("Strengthening engine did not attest outward-safe cuts")
    for cut in cuts:
        if cut.kind == "ANCHORED_SUBMODULAR_SUPERLEVEL" and cut.metadata.get(
            "outward_safe"
        ) is not True:
            raise RuntimeError(f"Cut lacks outward-safe provenance: {cut.name}")
    if expected_cut_count is not None and len(cuts) != int(expected_cut_count):
        raise RuntimeError(
            f"High cut count changed: expected={expected_cut_count}, observed={len(cuts)}"
        )
    if expected_anchor_count is not None and int(cut_info.get("unique_anchor_count", -1)) != int(
        expected_anchor_count
    ):
        raise RuntimeError(
            "High cut anchor count changed: "
            f"expected={expected_anchor_count}, observed={cut_info.get('unique_anchor_count')}"
        )
    exact_model = append_cuts(base, cuts)
    _require_dimensions(exact_model, expected_dimensions, label="High threshold oracle")
    if np.any(exact_model.integrality != 1):
        raise RuntimeError("SCIP high-need oracle must be all-binary")
    proof_model, relaxation_manifest = _outward_relax(exact_model)
    _require_dimensions(proof_model, expected_dimensions, label="Relaxed high threshold oracle")
    if np.any(proof_model.integrality != 1):
        raise RuntimeError("Outward transport changed all-binary integrality")
    digest = _cut_digest(cuts)
    return ThresholdOracleModel(
        metric="high_need_population",
        backend="SCIP_PORTFOLIO",
        exact_model=exact_model,
        model=proof_model,
        threshold=float(threshold),
        incumbent_value=float(incumbent_value),
        policy_relative_gap=float(policy_relative_gap),
        retained_floor=float(retained_floor),
        total_population_floor=float(total_population_floor),
        min_sigungu_floor=float(min_sigungu_floor),
        incumbent_selection=tuple(int(v) for v in np.asarray(incumbent_selection).tolist()),
        selection_digest=pin["selection_digest"],
        exact_model_digest=_canonical_model_digest(exact_model),
        proof_model_digest=_canonical_model_digest(proof_model),
        universe_digest=_canonical_universe_digest(formulation, exact_model),
        cut_digest=digest,
        cut_seed_digest=seeds_digest,
        metadata={
            "pinned_selection": pin,
            "binary_coverage_theorem": theorem,
            "cut_generation": cut_info,
            "cut_count": len(cuts),
            "proof_relaxation": relaxation_manifest,
            "all_columns_binary": True,
            "expected_dimensions": dict(expected_dimensions or _dimension_dict(exact_model)),
        },
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _status_error(status: Any) -> bool:
    return "error" in str(status).lower()


def _solver_index_names(highs: Any, lp: Any, *, kind: str, count: int) -> list[str]:
    attribute = "col_names_" if kind == "col" else "row_names_"
    raw_names = list(getattr(lp, attribute, []) or [])
    if len(raw_names) == count:
        return [str(value) for value in raw_names]
    getter_name = "getColName" if kind == "col" else "getRowName"
    getter = getattr(highs, getter_name, None)
    if not callable(getter):
        return []
    names: list[str] = []
    for index in range(count):
        result = getter(index)
        if not isinstance(result, tuple) or len(result) < 2 or _status_error(result[0]):
            return []
        names.append(str(result[1]))
    return names


def _highs_imported_model(
    template: LinearMipModel,
    highs: Any,
    *,
    require_index_names: bool = False,
) -> LinearMipModel:
    raw_model = highs.getModel()
    lp = raw_model.lp_
    n_col, n_row = int(lp.num_col_), int(lp.num_row_)
    if require_index_names:
        column_names = _solver_index_names(highs, lp, kind="col", count=n_col)
        row_names = _solver_index_names(highs, lp, kind="row", count=n_row)
        expected_columns = [f"c{index}" for index in range(n_col)]
        expected_rows = [f"r{index}" for index in range(n_row)]
        if column_names != expected_columns:
            raise RuntimeError(
                "MPS column identity/order is not the complete c0..cN map"
            )
        if row_names != expected_rows:
            raise RuntimeError("MPS row identity/order is not the complete r0..rN map")
    matrix = lp.a_matrix_
    starts = np.asarray(list(matrix.start_), dtype=np.int64)
    if starts.size == n_col:  # Some bindings omit the terminal pointer.
        starts = np.concatenate([starts, np.asarray([len(matrix.value_)], dtype=np.int64)])
    A = sparse.csc_matrix(
        (
            np.asarray(list(matrix.value_), dtype=np.float64),
            np.asarray(list(matrix.index_), dtype=np.int32),
            starts,
        ),
        shape=(n_row, n_col),
    )
    raw_integrality = list(getattr(lp, "integrality_", []))
    if len(raw_integrality) != n_col:
        raise RuntimeError("MPS import omitted the integrality vector")

    def integer_flag(value: Any) -> int:
        text = str(value).lower()
        if "continuous" in text:
            return 0
        if "integer" in text or "binary" in text or "impl" in text:
            return 1
        try:
            return int(int(value) != 0)
        except Exception as exc:
            raise RuntimeError(f"Unknown imported variable type: {value}") from exc

    imported = replace(
        template,
        c=np.asarray(list(lp.col_cost_), dtype=np.float64),
        A=A,
        row_lower=np.asarray(list(lp.row_lower_), dtype=np.float64),
        row_upper=np.asarray(list(lp.row_upper_), dtype=np.float64),
        col_lower=np.asarray(list(lp.col_lower_), dtype=np.float64),
        col_upper=np.asarray(list(lp.col_upper_), dtype=np.float64),
        integrality=np.asarray([integer_flag(value) for value in raw_integrality], dtype=np.int8),
    )
    imported.validate()
    return imported


def write_mps(
    oracle: ThresholdOracleModel,
    path: Path,
    *,
    backend: HighsBackend | None = None,
    audit_transport: bool = True,
) -> MpsArtifact:
    """Write and re-import the proof model, then audit serialization direction."""

    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    solver = backend or HighsBackend()
    try:
        solver.HighsClass.resetGlobalScheduler(True)
    except Exception:
        pass
    writer = solver.HighsClass()
    solver._pass_model(writer, oracle.model)
    status = writer.writeModel(str(path))
    if _status_error(status):
        raise RuntimeError(f"HiGHS could not write proof MPS: {status}")
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError("Proof MPS is missing or empty after write")
    audit_payload: dict[str, Any] = {"performed": False}
    sufficient = False
    if audit_transport:
        reader = solver.HighsClass()
        read_status = reader.readModel(str(path))
        if _status_error(read_status):
            raise RuntimeError(f"HiGHS could not re-import proof MPS: {read_status}")
        imported = _highs_imported_model(
            oracle.model, reader, require_index_names=True
        )
        try:
            from .canonical import audit_relaxing_transport
        except ImportError as exc:  # pragma: no cover - installation contract
            raise RuntimeError("Canonical transport audit is required") from exc
        audit = audit_relaxing_transport(
            oracle.exact_model,
            imported,
            proof_kind="infeasibility",
            intended_universe_digest=oracle.universe_digest,
            imported_universe_digest=oracle.universe_digest,
        )
        sufficient = bool(getattr(audit, "sufficient", False))
        to_dict = getattr(audit, "to_dict", None)
        audit_payload = dict(to_dict()) if callable(to_dict) else asdict(audit)
        audit_payload["solver_index_names_verified"] = True
        if not sufficient:
            raise RuntimeError(f"MPS transport is not an outward relaxation: {audit_payload}")
    size = int(path.stat().st_size)
    artifact = MpsArtifact(
        path=path,
        sha256=_sha256_file(path),
        size_bytes=size,
        size_mib=float(size / (2**20)),
        rows=oracle.model.n_row,
        cols=oracle.model.n_col,
        nnz=int(oracle.model.A.nnz),
        transport_sufficient=sufficient,
        transport_audit=audit_payload,
    )
    try:
        solver.HighsClass.resetGlobalScheduler(True)
    except Exception:
        pass
    return artifact


def _set_and_read_highs_options(highs: Any, options: Mapping[str, Any]) -> dict[str, Any]:
    for name, value in options.items():
        status = highs.setOptionValue(name, value)
        if _status_error(status):
            raise RuntimeError(f"HiGHS rejected required option {name}={value!r}: {status}")
    readback: dict[str, Any] = {}
    for name, expected in options.items():
        result = highs.getOptionValue(name)
        if not isinstance(result, tuple) or len(result) < 2 or _status_error(result[0]):
            raise RuntimeError(f"HiGHS could not read required option {name}: {result}")
        observed = result[1]
        readback[name] = observed
        if isinstance(expected, float):
            equal = math.isclose(float(observed), expected, rel_tol=0.0, abs_tol=1e-12)
        elif isinstance(expected, Path):
            equal = Path(str(observed)).resolve() == expected.resolve()
        else:
            equal = observed == expected
        if not equal:
            raise RuntimeError(
                f"HiGHS option readback mismatch for {name}: "
                f"requested={expected!r}, observed={observed!r}"
            )
    return readback


def _finite(value: Any) -> float | None:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


def _scip_finite(value: Any) -> float | None:
    """Normalize SCIP's finite infinity sentinels before persisting evidence."""

    converted = _finite(value)
    if converted is None or abs(converted) >= SCIP_INFINITY_SENTINEL_ABS_MIN:
        return None
    return converted


def _ensure_log(path: Path, sentinel: str) -> None:
    if path.is_file() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# {sentinel}\n", encoding="utf-8")


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(_jsonable(value), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _validate_solver_selection(
    oracle: ThresholdOracleModel,
    formulation: CertificationFormulation,
    selected_indices: Sequence[int] | np.ndarray,
) -> tuple[dict[str, Any], bool]:
    selected = np.asarray(selected_indices)
    valid, reason = formulation.validate_selection(selected)
    if not valid:
        raise RuntimeError(f"Solver returned an invalid incumbent: {reason}")
    metrics = formulation.metrics(selected.astype(np.int64, copy=False))
    if float(metrics["unique_elderly_population"]) < oracle.total_population_floor - _metric_tolerance(
        "total_population", oracle.total_population_floor
    ):
        raise RuntimeError("Solver incumbent violates the total-population floor")
    if oracle.min_sigungu_floor is not None and float(
        metrics["min_sigungu_coverage_ratio"]
    ) < oracle.min_sigungu_floor - _metric_tolerance(
        "min_sigungu_coverage", oracle.min_sigungu_floor
    ):
        raise RuntimeError("Solver incumbent violates the min-sigungu floor")
    metric_value = float(metrics[_metric_key(oracle.metric)])
    reaches = metric_value >= oracle.threshold - _metric_tolerance(oracle.metric, oracle.threshold)
    if not reaches:
        # Both official models explicitly contain the target row.  A solver
        # incumbent that fails recomputation is a semantic contradiction, not
        # evidence that may be silently ignored.
        raise RuntimeError("Solver incumbent does not satisfy the threshold row on recomputation")
    return metrics, reaches


def run_min_highs_oracle(
    oracle: ThresholdOracleModel,
    formulation: CertificationFormulation,
    *,
    output_dir: Path,
    option_overrides: Mapping[str, Any] | None = None,
    highs_factory: Callable[[], Any] | None = None,
    pass_model: Callable[[Any, LinearMipModel], None] | None = None,
) -> ThresholdOracleResult:
    if oracle.metric != "min_sigungu_coverage" or oracle.backend != "HIGHS":
        raise ValueError("run_min_highs_oracle received a non-min HiGHS oracle")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "min_threshold.highs.log"
    improving_path = output_dir / "min_threshold.highs.improving.sol"
    requested = dict(OFFICIAL_MIN_HIGHS_OPTIONS)
    if option_overrides:
        for name, value in option_overrides.items():
            if name in OFFICIAL_MIN_HIGHS_OPTIONS and value != OFFICIAL_MIN_HIGHS_OPTIONS[name]:
                raise ValueError(
                    f"Official min HiGHS option is frozen: {name}="
                    f"{OFFICIAL_MIN_HIGHS_OPTIONS[name]!r}"
                )
        requested.update(dict(option_overrides))
    # These evidence paths are required options too and are read back exactly.
    requested.update(
        {
            "log_file": str(log_path),
            "mip_improving_solution_save": True,
            "mip_improving_solution_report_sparse": True,
            "mip_improving_solution_file": str(improving_path),
        }
    )

    backend: HighsBackend | None = None
    if highs_factory is None:
        backend = HighsBackend()
        highs_factory = backend.HighsClass
        pass_model = backend._pass_model
        try:
            backend.HighsClass.resetGlobalScheduler(True)
        except Exception:
            pass
    highs = highs_factory()
    readback = _set_and_read_highs_options(highs, requested)
    if pass_model is None:
        raise ValueError("A pass_model callback is required with a custom HiGHS factory")
    pass_model(highs, oracle.model)
    loaded = _highs_imported_model(oracle.model, highs)
    if _dimension_dict(loaded) != _dimension_dict(oracle.model):
        raise RuntimeError("HiGHS passModel changed min proof dimensions")
    loaded_digest = _canonical_model_digest(loaded)
    if loaded_digest != oracle.proof_model_digest:
        raise RuntimeError(
            "In-memory HiGHS passModel changed the canonical proof model: "
            f"expected={oracle.proof_model_digest}, observed={loaded_digest}"
        )
    started = time.perf_counter()
    run_status = highs.run()
    measured_wall = time.perf_counter() - started
    if _status_error(run_status):
        raise RuntimeError(f"HiGHS min threshold solve failed: {run_status}")
    status = str(highs.modelStatusToString(highs.getModelStatus())).upper().replace(" ", "_")
    infeasible = status == "INFEASIBLE"
    info = highs.getInfo()
    solution = highs.getSolution()
    value_valid = bool(getattr(solution, "value_valid", False))
    values = np.asarray(list(getattr(solution, "col_value", [])), dtype=np.float64)
    has_incumbent = bool(value_valid and values.size == oracle.model.n_col and np.all(np.isfinite(values)))
    if infeasible and has_incumbent:
        raise RuntimeError("HiGHS status contradiction: INFEASIBLE with a valid incumbent")
    if "INFEASIBLE" in status and status != "INFEASIBLE":
        raise RuntimeError(f"Ambiguous HiGHS infeasibility status is not certifiable: {status}")
    counterexample = False
    counterexample_metrics: dict[str, Any] | None = None
    if has_incumbent:
        selected = formulation.selected_from_solution(oracle.model, values)
        counterexample_metrics, counterexample = _validate_solver_selection(
            oracle, formulation, selected
        )
    certified = bool(infeasible and not has_incumbent and not counterexample)
    if status == "OPTIMAL" and not has_incumbent:
        raise RuntimeError("HiGHS status contradiction: OPTIMAL without an incumbent")
    _ensure_log(log_path, "NO_HIGHS_LOG_OUTPUT")
    _ensure_log(improving_path, "NO_IMPROVING_SOLUTION")
    version = str(highs.version())
    if version != OFFICIAL_HIGHS_VERSION:
        raise RuntimeError(
            f"Official min oracle requires HiGHS {OFFICIAL_HIGHS_VERSION}; found {version}"
        )
    wall = _finite(getattr(highs, "getRunTime", lambda: None)())
    wall = measured_wall if wall is None else wall
    best_bound = _finite(getattr(info, "mip_dual_bound", None))
    nodes_raw = getattr(info, "mip_node_count", None)
    nodes = int(nodes_raw) if nodes_raw is not None else None
    result = ThresholdOracleResult(
        metric=oracle.metric,
        backend="HIGHS_THRESHOLD_DECISION",
        status=status,
        certified_threshold_infeasible=certified,
        counterexample_found=counterexample,
        threshold=oracle.threshold,
        incumbent_value=oracle.incumbent_value,
        policy_relative_gap=oracle.policy_relative_gap,
        retained_floor=oracle.retained_floor,
        incumbent_selection=oracle.incumbent_selection,
        selection_digest=oracle.selection_digest,
        best_bound=best_bound,
        relative_gap=oracle.policy_relative_gap,
        wall_time_sec=float(wall),
        node_count=nodes,
        solver_version=version,
        exact_model_digest=oracle.exact_model_digest,
        proof_model_digest=oracle.proof_model_digest,
        universe_digest=oracle.universe_digest,
        cut_digest=oracle.cut_digest,
        mps_sha256=None,
        requested_options=requested,
        readback_options=readback,
        log_paths=[log_path, improving_path],
        evidence={
            "run_status": str(run_status),
            "has_incumbent": has_incumbent,
            "counterexample_metrics": counterexample_metrics,
            "model_transport": {
                "kind": "IN_MEMORY_HIGHS_PASS_MODEL",
                "proof_model_digest": oracle.proof_model_digest,
            },
            "model_dimensions": _dimension_dict(oracle.model),
            "oracle_metadata": oracle.metadata,
            "strict_threshold_delta_ratio": (
                oracle.threshold - oracle.incumbent_value
            )
            / max(abs(oracle.incumbent_value), 1e-12),
        },
    )
    if backend is not None:
        try:
            backend.HighsClass.resetGlobalScheduler(True)
        except Exception:
            pass
    return result


def _safe_solver_call(function: Callable[[], Any], *, label: str) -> Any:
    try:
        return function()
    except Exception as exc:
        raise RuntimeError(f"SCIP could not provide required {label} evidence") from exc


def _scip_worker(payload: Mapping[str, Any]) -> dict[str, Any]:
    import pyscipopt  # type: ignore
    from pyscipopt import Model  # type: ignore

    mps_path = Path(str(payload["mps_path"]))
    expected_sha = str(payload["mps_sha256"])
    if _sha256_file(mps_path) != expected_sha:
        raise RuntimeError("Worker proof MPS does not match the pinned SHA")
    seed = int(payload["seed"])
    rows = int(payload["rows"])
    cols = int(payload["cols"])
    candidates = int(payload["candidates"])
    model = Model(f"MEDIROAD_STAGE42D_HIGH_SEED_{seed}")
    model.readProblem(str(mps_path))
    variables = {variable.name: variable for variable in model.getVars()}
    expected_names = {f"c{index}" for index in range(cols)}
    if set(variables) != expected_names:
        raise RuntimeError("SCIP MPS column map is not the complete c0..cN mapping")
    if int(model.getNVars()) != cols or int(model.getNConss()) != rows:
        raise RuntimeError("SCIP imported proof model dimensions changed")
    if any(str(variables[f"c{i}"].vtype()).upper() != "BINARY" for i in range(cols)):
        raise RuntimeError("SCIP high threshold model is not all-binary")

    time_limit = float(payload["time_limit_sec"])
    memory_limit_mib = float(payload["memory_limit_mib"])
    threads = int(payload["threads"])
    model.setRealParam("limits/time", time_limit)
    model.setRealParam("limits/gap", 0.0)
    model.setRealParam("limits/memory", memory_limit_mib)
    model.setIntParam("parallel/maxnthreads", threads)
    model.setIntParam("lp/threads", threads)
    model.setIntParam("randomization/randomseedshift", seed)
    model.setIntParam("display/verblevel", 3)
    log_path = Path(str(payload["log_path"]))
    model.setLogfile(str(log_path))
    model.hideOutput(True)
    params = model.getParams()
    names = (
        "limits/time",
        "limits/gap",
        "limits/memory",
        "parallel/maxnthreads",
        "lp/threads",
        "randomization/randomseedshift",
    )
    readback = {name: params[name] for name in names}
    requested = {
        "limits/time": time_limit,
        "limits/gap": 0.0,
        "limits/memory": memory_limit_mib,
        "parallel/maxnthreads": threads,
        "lp/threads": threads,
        "randomization/randomseedshift": seed,
    }
    for name, expected in requested.items():
        observed = readback[name]
        if isinstance(expected, float):
            equal = math.isclose(float(observed), expected, rel_tol=0.0, abs_tol=1e-12)
        else:
            equal = observed == expected
        if not equal:
            raise RuntimeError(
                f"SCIP parameter readback mismatch {name}: {expected!r} != {observed!r}"
            )

    model.optimize()
    solution = model.getBestSol()
    status = str(model.getStatus()).lower()
    selected: list[int] | None = None
    if solution is not None:
        x_values = np.asarray(
            [model.getSolVal(solution, variables[f"c{i}"]) for i in range(candidates)],
            dtype=np.float64,
        )
        if np.any(~np.isfinite(x_values)):
            raise RuntimeError("SCIP incumbent contains non-finite candidate values")
        selected = np.flatnonzero(x_values >= 0.5).astype(int).tolist()
    primal = _scip_finite(model.getPrimalbound())
    dual = _scip_finite(model.getDualbound())
    memory_bytes_raw = _safe_solver_call(lambda: model.getMemUsed(), label="memory usage")
    memory_bytes = int(memory_bytes_raw)
    if memory_bytes < 0:
        raise RuntimeError("SCIP reported negative memory usage")
    version = str(_safe_solver_call(lambda: model.version(), label="model.version"))
    version_components = {
        "major": int(_safe_solver_call(lambda: model.getMajorVersion(), label="major version")),
        "minor": int(_safe_solver_call(lambda: model.getMinorVersion(), label="minor version")),
        "tech": int(_safe_solver_call(lambda: model.getTechVersion(), label="tech version")),
    }
    if not version or version.lower() == "unknown":
        raise RuntimeError("SCIP model.version() is missing")
    pyscipopt_version = str(getattr(pyscipopt, "__version__", "UNKNOWN"))
    if pyscipopt_version != OFFICIAL_PYSCIPOPT_VERSION:
        raise RuntimeError(
            f"Official portfolio requires PySCIPOpt {OFFICIAL_PYSCIPOPT_VERSION}; "
            f"found {pyscipopt_version}"
        )
    observed_components = tuple(version_components[key] for key in ("major", "minor", "tech"))
    if observed_components != OFFICIAL_SCIP_VERSION_COMPONENTS:
        raise RuntimeError(
            "Official portfolio SCIP version changed: "
            f"expected={OFFICIAL_SCIP_VERSION_COMPONENTS}, observed={observed_components}"
        )
    outcome = {
        "seed": seed,
        "status": status,
        "has_solution": solution is not None,
        "primal_bound": primal,
        "dual_bound": dual,
        "gap": _scip_finite(model.getGap()),
        "nodes": int(model.getNNodes()),
        "lp_iterations": int(model.getNLPIterations()),
        "solving_time_sec": float(model.getSolvingTime()),
        "memory_used_bytes": memory_bytes,
        "memory_used_mib": float(memory_bytes / (2**20)),
        "selected_indices": selected,
        "column_map_verified": True,
        "all_columns_binary": True,
        "requested_parameters": requested,
        "readback_parameters": readback,
        "pyscipopt_version": pyscipopt_version,
        "scip_model_version": version,
        "scip_version_components": version_components,
        "log_path": str(log_path),
        "mps_sha256": expected_sha,
    }
    if _sha256_file(mps_path) != expected_sha:
        raise RuntimeError("Worker proof MPS changed during SCIP solve")
    return outcome


def validate_scip_portfolio_outcomes(
    oracle: ThresholdOracleModel,
    formulation: CertificationFormulation,
    outcomes: Sequence[Mapping[str, Any]],
    *,
    expected_seeds: Sequence[int],
    mps: MpsArtifact,
    requested_options: Mapping[str, Any] | None = None,
) -> ThresholdOracleResult:
    if oracle.metric != "high_need_population" or oracle.backend != "SCIP_PORTFOLIO":
        raise ValueError("SCIP portfolio validator received the wrong oracle")
    expected = tuple(int(seed) for seed in expected_seeds)
    if not expected or len(expected) != len(set(expected)):
        raise ValueError("Expected SCIP portfolio seeds must be non-empty and unique")
    if str(formulation.candidate_set) == "top3" and expected != OFFICIAL_SCIP_SEEDS:
        raise RuntimeError(
            f"Official Top3 portfolio seeds changed: expected={OFFICIAL_SCIP_SEEDS}, "
            f"observed={expected}"
        )
    if len(outcomes) != len(expected):
        raise RuntimeError(
            f"SCIP portfolio is incomplete: expected={len(expected)}, observed={len(outcomes)}"
        )
    observed_seeds = [int(outcome.get("seed")) for outcome in outcomes]
    if len(observed_seeds) != len(set(observed_seeds)) or set(observed_seeds) != set(expected):
        raise RuntimeError("SCIP portfolio seed set is missing, duplicated, or unexpected")

    expected_worker_base: dict[str, Any] | None = None
    if requested_options is not None:
        required_option_keys = {
            "seeds",
            "workers",
            "threads_per_worker",
            "time_limit_sec_per_seed",
            "memory_limit_mib_per_worker",
            "require_all_workers_infeasible",
        }
        if set(requested_options) != required_option_keys:
            raise RuntimeError("SCIP portfolio requested-option schema is incomplete or unexpected")
        if tuple(int(seed) for seed in requested_options["seeds"]) != expected:
            raise RuntimeError("SCIP portfolio requested seeds differ from the validated seed set")
        if int(requested_options["workers"]) != len(expected):
            raise RuntimeError("SCIP portfolio requested worker count differs from the seed count")
        if requested_options["require_all_workers_infeasible"] is not True:
            raise RuntimeError("SCIP portfolio requested options do not fail closed")
        expected_worker_base = {
            "limits/time": float(requested_options["time_limit_sec_per_seed"]),
            "limits/gap": 0.0,
            "limits/memory": float(requested_options["memory_limit_mib_per_worker"]),
            "parallel/maxnthreads": int(requested_options["threads_per_worker"]),
            "lp/threads": int(requested_options["threads_per_worker"]),
        }

    validated: list[dict[str, Any]] = []
    any_infeasible = False
    all_infeasible = True
    counterexample = False
    total_wall = 0.0
    total_nodes = 0
    best_upper_bound: float | None = None
    versions: set[str] = set()
    log_paths: list[Path] = []
    for raw in sorted(outcomes, key=lambda item: int(item["seed"])):
        outcome = dict(raw)
        status = str(outcome.get("status", "")).lower()
        if not status or "error" in status:
            raise RuntimeError(f"SCIP worker returned an error/empty status: {status!r}")
        has_solution = bool(outcome.get("has_solution", False))
        selected = outcome.get("selected_indices")
        if has_solution != (selected is not None):
            raise RuntimeError("SCIP incumbent presence contradicts selected_indices")
        if status == "infeasible" and has_solution:
            raise RuntimeError("SCIP status contradiction: INFEASIBLE with an incumbent")
        if status == "optimal" and not has_solution:
            raise RuntimeError("SCIP status contradiction: OPTIMAL without an incumbent")
        if outcome.get("column_map_verified") is not True or outcome.get(
            "all_columns_binary"
        ) is not True:
            raise RuntimeError("SCIP worker did not verify the all-binary column map")
        if str(outcome.get("mps_sha256", "")) != mps.sha256:
            raise RuntimeError("SCIP worker used an unpinned MPS")
        version = str(outcome.get("scip_model_version", ""))
        components = outcome.get("scip_version_components")
        if not version or version.lower() == "unknown" or not isinstance(components, Mapping):
            raise RuntimeError("SCIP worker version/model.version evidence is incomplete")
        for key in ("major", "minor", "tech"):
            if key not in components:
                raise RuntimeError("SCIP version component evidence is incomplete")
        observed_components = tuple(int(components[key]) for key in ("major", "minor", "tech"))
        if observed_components != OFFICIAL_SCIP_VERSION_COMPONENTS:
            raise RuntimeError(
                f"SCIP version components changed: {observed_components}"
            )
        if str(outcome.get("pyscipopt_version", "")) != OFFICIAL_PYSCIPOPT_VERSION:
            raise RuntimeError("PySCIPOpt version changed in a portfolio worker")
        versions.add(version)
        memory_bytes = outcome.get("memory_used_bytes")
        memory_mib = outcome.get("memory_used_mib")
        if memory_bytes is None or memory_mib is None or not math.isclose(
            float(memory_mib), float(memory_bytes) / (2**20), rel_tol=0.0, abs_tol=1e-12
        ):
            raise RuntimeError("SCIP memory evidence must convert bytes to MiB with 2**20")
        requested_parameters = outcome.get("requested_parameters")
        readback_parameters = outcome.get("readback_parameters")
        if requested_parameters != readback_parameters:
            raise RuntimeError("SCIP required parameter readback differs from the request")
        if expected_worker_base is not None:
            expected_worker_parameters = {
                **expected_worker_base,
                "randomization/randomseedshift": int(outcome["seed"]),
            }
            if requested_parameters != expected_worker_parameters:
                raise RuntimeError(
                    "SCIP worker parameters differ from the portfolio contract"
                )
        log_path = Path(str(outcome.get("log_path", "")))
        if not log_path.is_file() or log_path.stat().st_size <= 0:
            raise RuntimeError(f"SCIP worker log is missing or empty: {log_path}")
        log_paths.append(log_path)

        primal = _scip_finite(outcome.get("primal_bound"))
        dual = _scip_finite(outcome.get("dual_bound"))
        gap = _scip_finite(outcome.get("gap"))
        if gap is not None and gap < -1e-12:
            raise RuntimeError("SCIP reported a negative relative gap")
        outcome["primal_bound"] = primal
        outcome["dual_bound"] = dual
        outcome["gap"] = gap
        if has_solution and primal is None:
            raise RuntimeError("SCIP incumbent is missing a finite primal bound")
        if primal is not None and dual is not None and dual < primal - _metric_tolerance(
            "high_need_population", primal
        ):
            raise RuntimeError("SCIP MAX dual bound is below its primal bound")
        if dual is not None:
            best_upper_bound = dual if best_upper_bound is None else min(best_upper_bound, dual)
        metrics = None
        if has_solution:
            metrics, reaches = _validate_solver_selection(oracle, formulation, selected)
            counterexample = counterexample or reaches
        outcome["recomputed_metrics"] = metrics
        outcome["validated_counterexample"] = bool(metrics is not None)
        any_infeasible = any_infeasible or status == "infeasible"
        all_infeasible = all_infeasible and status == "infeasible" and not has_solution
        total_wall += float(outcome.get("solving_time_sec", 0.0))
        total_nodes += int(outcome.get("nodes", 0))
        validated.append(outcome)

    if any_infeasible and counterexample:
        raise RuntimeError("Portfolio contradiction: infeasibility proof and counterexample coexist")
    certified = bool(all_infeasible and not counterexample)
    solver_version = ",".join(sorted(versions))
    return ThresholdOracleResult(
        metric=oracle.metric,
        backend="SCIP_SPAWNED_PORTFOLIO",
        status="INFEASIBLE" if certified else ("COUNTEREXAMPLE" if counterexample else "INCONCLUSIVE"),
        certified_threshold_infeasible=certified,
        counterexample_found=counterexample,
        threshold=oracle.threshold,
        incumbent_value=oracle.incumbent_value,
        policy_relative_gap=oracle.policy_relative_gap,
        retained_floor=oracle.retained_floor,
        incumbent_selection=oracle.incumbent_selection,
        selection_digest=oracle.selection_digest,
        best_bound=best_upper_bound,
        relative_gap=oracle.policy_relative_gap,
        wall_time_sec=total_wall,
        node_count=total_nodes,
        solver_version=solver_version,
        exact_model_digest=oracle.exact_model_digest,
        proof_model_digest=oracle.proof_model_digest,
        universe_digest=oracle.universe_digest,
        cut_digest=oracle.cut_digest,
        mps_sha256=mps.sha256,
        requested_options=dict(requested_options or {}),
        readback_options={
            str(outcome["seed"]): dict(outcome["readback_parameters"])
            for outcome in validated
        },
        log_paths=log_paths,
        outcomes=validated,
        evidence={
            "all_workers_complete": True,
            "expected_seeds": list(expected),
            "all_workers_infeasible": all_infeasible,
            "mps": asdict(mps),
            "model_dimensions": _dimension_dict(oracle.model),
            "cut_seed_digest": oracle.cut_seed_digest,
            "oracle_metadata": oracle.metadata,
            "strict_threshold_delta_ratio": (
                oracle.threshold - oracle.incumbent_value
            )
            / max(abs(oracle.incumbent_value), 1e-12),
        },
    )


def run_scip_portfolio(
    oracle: ThresholdOracleModel,
    formulation: CertificationFormulation,
    mps: MpsArtifact,
    *,
    output_dir: Path,
    seeds: Sequence[int] = OFFICIAL_SCIP_SEEDS,
    workers: int = 8,
    threads_per_worker: int = 1,
    time_limit_sec: float = OFFICIAL_SCIP_TIME_LIMIT_SEC,
    memory_limit_mib: float = 2800.0,
    require_all_workers_infeasible: bool = True,
    worker_function: Callable[[Mapping[str, Any]], dict[str, Any]] = _scip_worker,
) -> ThresholdOracleResult:
    if not mps.transport_sufficient:
        raise RuntimeError("SCIP proof MPS has no sufficient outward-transport audit")
    if _sha256_file(mps.path) != mps.sha256:
        raise RuntimeError("SCIP proof MPS changed before portfolio")
    seeds = tuple(int(seed) for seed in seeds)
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("SCIP seeds must be non-empty and unique")
    if workers != len(seeds):
        raise ValueError("Official SCIP portfolio requires one spawned worker per seed")
    if threads_per_worker != 1:
        raise ValueError("Official SCIP portfolio workers must be single-threaded")
    if not bool(require_all_workers_infeasible):
        raise ValueError("Official SCIP portfolio must require all workers infeasible")
    if str(formulation.candidate_set) == "top3":
        if seeds != OFFICIAL_SCIP_SEEDS or workers != 8:
            raise RuntimeError("Frozen Top3 SCIP portfolio must run the exact 8-seed set")
        _require_exact_float(
            "SCIP time_limit_sec", time_limit_sec, OFFICIAL_SCIP_TIME_LIMIT_SEC
        )
        _require_exact_float("SCIP memory_limit_mib", memory_limit_mib, 2800.0)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    common = {
        "mps_path": str(mps.path),
        "mps_sha256": mps.sha256,
        "rows": oracle.model.n_row,
        "cols": oracle.model.n_col,
        "candidates": formulation.nx,
        "time_limit_sec": float(time_limit_sec),
        "memory_limit_mib": float(memory_limit_mib),
        "threads": int(threads_per_worker),
    }
    outcomes: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    worker_artifacts: dict[int, dict[str, Any]] = {}
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
        futures = {
            executor.submit(
                worker_function,
                {
                    **common,
                    "seed": seed,
                    "log_path": str(output_dir / f"high_threshold_seed_{seed}.scip.log"),
                },
            ): seed
            for seed in seeds
        }
        for future in as_completed(futures):
            seed = futures[future]
            try:
                outcome = future.result()
                outcomes.append(outcome)
                worker_path = output_dir / f"worker_seed_{seed}.json"
                _atomic_write_json(worker_path, outcome)
                worker_artifacts[seed] = {
                    "path": str(worker_path),
                    "size_bytes": worker_path.stat().st_size,
                    "sha256": _sha256_file(worker_path),
                    "kind": "OUTCOME",
                }
            except BaseException as exc:
                failure = {
                    "seed": seed,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                failures.append(failure)
                failure_path = output_dir / f"worker_seed_{seed}.failure.json"
                _atomic_write_json(failure_path, failure)
                worker_artifacts[seed] = {
                    "path": str(failure_path),
                    "size_bytes": failure_path.stat().st_size,
                    "sha256": _sha256_file(failure_path),
                    "kind": "FAILURE",
                }
            _atomic_write_json(
                output_dir / "portfolio_progress.json",
                {
                    "completed_seeds": sorted(int(item["seed"]) for item in outcomes),
                    "failed_seeds": sorted(int(item["seed"]) for item in failures),
                    "remaining_count": len(seeds) - len(outcomes) - len(failures),
                },
            )
    if failures:
        raise RuntimeError(f"SCIP portfolio worker failures: {failures}")
    if _sha256_file(mps.path) != mps.sha256:
        raise RuntimeError("SCIP proof MPS changed during portfolio")
    requested = {
        "seeds": list(seeds),
        "workers": workers,
        "threads_per_worker": threads_per_worker,
        "time_limit_sec_per_seed": float(time_limit_sec),
        "memory_limit_mib_per_worker": float(memory_limit_mib),
        "require_all_workers_infeasible": True,
    }
    result = validate_scip_portfolio_outcomes(
        oracle,
        formulation,
        outcomes,
        expected_seeds=seeds,
        mps=mps,
        requested_options=requested,
    )
    result.evidence["worker_result_artifacts"] = {
        str(seed): worker_artifacts[seed] for seed in sorted(worker_artifacts)
    }
    result.evidence["portfolio_progress_path"] = str(
        output_dir / "portfolio_progress.json"
    )
    return result


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def certificate_from_threshold_oracle(result: ThresholdOracleResult) -> MetricCertificate:
    """Convert a strict decision-oracle result into the existing certificate type."""

    if selection_digest(result.incumbent_selection) != result.selection_digest:
        raise RuntimeError("Threshold result incumbent selection is no longer pinned")
    if result.certified_threshold_infeasible and result.counterexample_found:
        raise RuntimeError("A threshold certificate cannot also contain a counterexample")
    if result.certified_threshold_infeasible:
        certificate = (
            "THRESHOLD_INFEASIBILITY_HIGHS"
            if result.backend.startswith("HIGHS")
            else "THRESHOLD_INFEASIBILITY_SCIP_PORTFOLIO"
        )
    elif result.counterexample_found:
        certificate = "THRESHOLD_COUNTEREXAMPLE_FOUND"
    else:
        certificate = "INCONCLUSIVE_THRESHOLD_ORACLE"
    return MetricCertificate(
        metric=result.metric,  # type: ignore[arg-type]
        certified=bool(result.certified_threshold_infeasible),
        certificate=certificate,
        incumbent_value=float(result.incumbent_value),
        best_bound=float(result.threshold) if result.certified_threshold_infeasible else result.best_bound,
        relative_gap=result.relative_gap,
        selected_indices=np.asarray(result.incumbent_selection, dtype=int),
        retained_floor=float(result.retained_floor),
        rounds=1,
        evidence=[_jsonable(asdict(result))],
    )


__all__ = [
    "MpsArtifact",
    "OFFICIAL_HIGH_CUT_SEED_COUNT",
    "OFFICIAL_HIGH_CUT_SEED_DIGEST",
    "OFFICIAL_HIGH_INCUMBENT",
    "OFFICIAL_HIGH_SELECTION_DIGEST",
    "OFFICIAL_HIGH_THRESHOLD",
    "OFFICIAL_MIN_HIGHS_OPTIONS",
    "OFFICIAL_MIN_INCUMBENT",
    "OFFICIAL_MIN_RETAINED_FLOOR",
    "OFFICIAL_MIN_SELECTION_DIGEST",
    "OFFICIAL_MIN_THRESHOLD",
    "OFFICIAL_MIN_TOTAL_POPULATION_FLOOR",
    "OFFICIAL_SCIP_SEEDS",
    "ThresholdOracleModel",
    "ThresholdOracleResult",
    "build_high_scip_oracle",
    "build_min_highs_oracle",
    "certificate_from_threshold_oracle",
    "run_min_highs_oracle",
    "run_scip_portfolio",
    "seed_set_digest",
    "selection_digest",
    "validate_cut_seed_set",
    "validate_pinned_selection",
    "validate_scip_portfolio_outcomes",
    "write_mps",
]
