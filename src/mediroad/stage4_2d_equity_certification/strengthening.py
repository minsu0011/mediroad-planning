from __future__ import annotations

import hashlib
import math
from dataclasses import asdict
from typing import Any, Iterable

import numpy as np

from mediroad.stage4_2_certification.formulation import CertificationFormulation

from .types import CutRecord, EvidenceSeed, FixingRecord, MetricTarget


def _dense_incidence(formulation: CertificationFormulation) -> np.ndarray:
    return formulation._candidate_pattern.toarray().astype(np.float64, copy=False)


def metric_targets(
    formulation: CertificationFormulation,
    *,
    total_population_floor: float,
    min_sigungu_floor: float | None,
    high_need_target: float | None,
) -> list[MetricTarget]:
    targets = [
        MetricTarget(
            name="total_population",
            target=float(total_population_floor),
            weights=np.asarray(formulation.population, dtype=np.float64),
            source="EQUITY_MIN_EFFICIENCY_FLOOR",
        )
    ]
    if min_sigungu_floor is not None:
        for sigungu, pattern_idx in formulation.sigungu_pattern_indices.items():
            weights = np.zeros(formulation.ny, dtype=np.float64)
            weights[pattern_idx] = formulation.population[pattern_idx]
            targets.append(
                MetricTarget(
                    name=f"sigungu::{sigungu}",
                    target=float(min_sigungu_floor * formulation.sigungu_population_total[sigungu]),
                    weights=weights,
                    source="EQUITY_RETAINED_MIN_SIGUNGU_FLOOR",
                )
            )
    if high_need_target is not None:
        targets.append(
            MetricTarget(
                name="high_need_population",
                target=float(high_need_target),
                weights=np.asarray(formulation.high_need, dtype=np.float64),
                source="EQUITY_HIGH_NEED_CERTIFICATION_THRESHOLD",
            )
        )
    return targets


def _pairwise_overlap(B: np.ndarray, weights: np.ndarray, use_gpu: bool) -> tuple[np.ndarray, str]:
    if use_gpu:
        try:
            import cupy as cp  # type: ignore

            Bg = cp.asarray(B, dtype=cp.float64)
            wg = cp.asarray(weights, dtype=cp.float64)
            singleton = Bg @ wg
            overlap = (Bg * wg[None, :]) @ Bg.T
            result = cp.asnumpy(overlap)
            single = cp.asnumpy(singleton)
            cp.cuda.Stream.null.synchronize()
            del Bg, wg, singleton, overlap
            cp.get_default_memory_pool().free_all_blocks()
            return np.vstack([single[None, :], result]), "CUPY_CUDA"
        except Exception:
            pass
    singleton = B @ weights
    overlap = (B * weights[None, :]) @ B.T
    return np.vstack([singleton[None, :], overlap]), "NUMPY_CPU"



def _group_relaxation_upper(
    gains: np.ndarray,
    *,
    remaining_slots: int,
    labels: np.ndarray,
    cap: int,
    forced_index: int | None,
) -> float:
    """Exact maximum of a single-group-cap relaxation plus cardinality."""

    pooled: list[float] = []
    forced_label = labels[forced_index] if forced_index is not None else None
    for label in np.unique(labels):
        residual = int(cap) - (1 if forced_index is not None and label == forced_label else 0)
        if residual <= 0:
            continue
        idx = np.flatnonzero(labels == label)
        if forced_index is not None:
            idx = idx[idx != forced_index]
        values = np.asarray(gains[idx], dtype=np.float64)
        values = values[np.isfinite(values) & (values > 0.0)]
        if values.size == 0:
            continue
        take = min(residual, values.size)
        pooled.extend(np.partition(values, -take)[-take:].tolist())
    if not pooled or remaining_slots <= 0:
        return 0.0
    values = np.asarray(pooled, dtype=np.float64)
    take = min(int(remaining_slots), values.size)
    return float(np.partition(values, -take)[-take:].sum())


def _resource_relaxed_addition_upper(
    formulation: CertificationFormulation,
    gains: np.ndarray,
    *,
    remaining_slots: int,
    forced_index: int | None,
) -> tuple[float, dict[str, float]]:
    work = np.maximum(np.asarray(gains, dtype=np.float64), 0.0).copy()
    if forced_index is not None:
        work[int(forced_index)] = -np.inf
    finite = work[np.isfinite(work) & (work > 0.0)]
    take = min(int(remaining_slots), finite.size)
    cardinality = float(np.partition(finite, -take)[-take:].sum()) if take > 0 else 0.0
    frame = formulation.candidates
    admin = frame["admin_code"].astype(str).to_numpy()
    sigungu = frame["sigungu"].astype(str).to_numpy()
    cluster = frame["cluster_id"].fillna(frame["venue_id"]).astype(str).to_numpy()
    bounds = {
        "cardinality": cardinality,
        "admin_relaxation": _group_relaxation_upper(
            work, remaining_slots=remaining_slots, labels=admin, cap=formulation.max_admin, forced_index=forced_index
        ),
        "sigungu_relaxation": _group_relaxation_upper(
            work, remaining_slots=remaining_slots, labels=sigungu, cap=formulation.max_sigungu, forced_index=forced_index
        ),
        "cluster_relaxation": _group_relaxation_upper(
            work, remaining_slots=remaining_slots, labels=cluster, cap=1 if formulation.one_cluster else formulation.visit_count, forced_index=forced_index
        ),
    }
    return float(min(bounds.values())), bounds

def compute_safe_zero_fixings(
    formulation: CertificationFormulation,
    targets: Iterable[MetricTarget],
    *,
    use_gpu: bool,
) -> tuple[list[FixingRecord], dict[str, Any]]:
    """Fix x_j=0 only when a cardinality-relaxed submodular upper bound fails.

    For any k-set T containing j, monotonicity and submodularity imply
    f(T) <= f({j}) + sum of the largest k-1 marginals Delta_i({j}).
    Resource caps and overlap among the added candidates are intentionally ignored,
    making the bound optimistic and therefore safe for zero fixing.
    """

    B = _dense_incidence(formulation)
    k = int(formulation.visit_count)
    by_candidate: dict[int, FixingRecord] = {}
    diagnostics: dict[str, Any] = {"metrics": {}, "backend": None}
    for target in targets:
        packed, backend = _pairwise_overlap(B, np.asarray(target.weights, dtype=np.float64), use_gpu)
        diagnostics["backend"] = backend
        singleton, overlap = packed[0], packed[1:]
        marginal = np.maximum(singleton[None, :] - overlap, 0.0)
        np.fill_diagonal(marginal, -np.inf)
        upper = np.empty(formulation.nx, dtype=np.float64)
        resource_details: list[dict[str, float]] = []
        for forced in range(formulation.nx):
            addition_upper, bounds = _resource_relaxed_addition_upper(
                formulation,
                marginal[forced],
                remaining_slots=k - 1,
                forced_index=forced,
            )
            upper[forced] = singleton[forced] + addition_upper
            resource_details.append(bounds)
        tol = 2e-8 * max(1.0, abs(float(target.target)))
        failed = np.flatnonzero(upper < float(target.target) - tol)
        diagnostics["metrics"][target.name] = {
            "target": float(target.target),
            "fixed_zero_count": int(failed.size),
            "minimum_optimistic_upper_bound": float(np.min(upper)),
            "maximum_optimistic_upper_bound": float(np.max(upper)),
        }
        for idx in failed.tolist():
            record = FixingRecord(
                candidate_index=int(idx),
                venue_id=str(formulation.candidates.iloc[idx]["venue_id"]),
                value=0,
                metric=target.name,
                target=float(target.target),
                optimistic_upper_bound=float(upper[idx]),
                proof_type="FORCED_IN_SUBMODULAR_CARDINALITY_UPPER_BOUND",
                metadata={
                    "visit_count": k,
                    "backend": backend,
                    "resource_relaxation_bounds": resource_details[idx],
                },
            )
            existing = by_candidate.get(int(idx))
            if existing is None or record.optimistic_upper_bound / max(record.target, 1e-12) < existing.optimistic_upper_bound / max(existing.target, 1e-12):
                by_candidate[int(idx)] = record
    fixings = [by_candidate[idx] for idx in sorted(by_candidate)]
    diagnostics["unique_fixed_zero_count"] = len(fixings)
    return fixings, diagnostics


def _anchor_order(formulation: CertificationFormulation, selected: np.ndarray, weights: np.ndarray) -> np.ndarray:
    selected = np.asarray(selected, dtype=int)
    B = formulation._candidate_pattern.toarray().astype(bool, copy=False)
    counts = B[selected].sum(axis=0)
    unique = np.asarray((B[selected] & (counts[None, :] == 1)) @ weights, dtype=np.float64)
    singleton = np.asarray(B[selected] @ weights, dtype=np.float64)
    order = np.lexsort((-singleton, -unique))
    return selected[order]


def _cut_hash(values: np.ndarray, lower: float, metric: str) -> str:
    h = hashlib.sha256()
    h.update(metric.encode("utf-8"))
    h.update(np.round(values, 10).tobytes())
    h.update(np.asarray([lower], dtype=np.float64).tobytes())
    return h.hexdigest()[:16]


def _nonnegative_sum_upper(value: float, term_count: int) -> tuple[float, float]:
    """Return an outward upper bound for a non-negative floating sum.

    ``numpy`` is free to use pairwise or vectorized accumulation.  The usual
    sequential ``gamma_n`` bound is therefore deliberately conservative for
    the dot products used here.  It lets the official infeasibility oracle use
    cuts that are a relaxation of the exact-real inequality represented by the
    stored float inputs, never an accidental strengthening caused by rounding.
    """

    observed = float(value)
    if not math.isfinite(observed) or observed < 0.0:
        raise ValueError("Outward-safe sums require finite non-negative values")
    if observed == 0.0 or term_count <= 0:
        return observed, 0.0
    unit_roundoff = np.finfo(np.float64).eps / 2.0
    product = float(term_count) * unit_roundoff
    if product >= 1.0:
        raise ValueError("Floating summation error bound is undefined")
    gamma = product / (1.0 - product)
    error = math.nextafter(gamma * observed / max(1.0 - gamma, np.finfo(float).tiny), math.inf)
    upper = math.nextafter(observed + error, math.inf)
    return upper, max(0.0, upper - observed)


def _power_of_two_scale(value: float) -> float:
    """Choose an exactly representable scale no smaller than ``value``."""

    if not math.isfinite(value) or value <= 1.0:
        return 1.0
    return math.ldexp(1.0, int(math.ceil(math.log2(value))))


def generate_anchored_submodular_cuts(
    formulation: CertificationFormulation,
    targets: Iterable[MetricTarget],
    seeds: Iterable[EvidenceSeed],
    *,
    anchor_sizes: Iterable[int],
    max_cuts: int,
    outward_safe: bool = False,
) -> tuple[list[CutRecord], dict[str, Any]]:
    """Generate valid superlevel cuts from submodular upper envelopes.

    f(T) <= f(S) + sum_{j in T\\S} Delta_j(S). Therefore f(T)>=R implies
    sum_{j notin S} Delta_j(S) x_j >= R-f(S).
    """

    B = formulation._candidate_pattern.toarray().astype(bool, copy=False)
    all_idx = np.arange(formulation.nx, dtype=np.int32)
    target_list = list(targets)
    seed_list = list(seeds)
    cuts: list[CutRecord] = []
    seen: set[str] = set()
    impossible: list[dict[str, Any]] = []
    maximum_sum_error = 0.0
    minimum_outward_rhs_relaxation = math.inf
    maximum_outward_rhs_relaxation = 0.0

    anchors: list[tuple[str, np.ndarray]] = [("EMPTY", np.empty(0, dtype=np.int32))]
    for seed in seed_list:
        for target in target_list:
            ordered = _anchor_order(formulation, seed.selected_indices, target.weights)
            for raw_size in anchor_sizes:
                size = max(0, min(int(raw_size), int(ordered.size)))
                anchors.append((f"{seed.source}::{target.name}::size{size}", ordered[:size].astype(np.int32)))
    # Deduplicate anchors independent of source.
    unique_anchors: list[tuple[str, np.ndarray]] = []
    anchor_seen: set[tuple[int, ...]] = set()
    for source, anchor in anchors:
        key = tuple(sorted(map(int, anchor)))
        if key in anchor_seen:
            continue
        anchor_seen.add(key)
        unique_anchors.append((source, np.asarray(key, dtype=np.int32)))

    for target in target_list:
        weights = np.asarray(target.weights, dtype=np.float64)
        if outward_safe and (
            np.any(~np.isfinite(weights)) or np.any(weights < 0.0)
        ):
            raise ValueError(
                "Outward-safe submodular cuts require finite non-negative metric weights"
            )
        for source, anchor in unique_anchors:
            covered = B[anchor].any(axis=0) if anchor.size else np.zeros(formulation.ny, dtype=bool)
            base = float(weights[covered].sum())
            legacy_rhs = float(target.target - base)
            if outward_safe:
                base_upper, base_error = _nonnegative_sum_upper(base, int(covered.sum()))
                rhs_exact_float = float(target.target - base_upper)
                rhs = math.nextafter(rhs_exact_float, -math.inf)
                maximum_sum_error = max(maximum_sum_error, base_error)
                rhs_relaxation = max(0.0, float(target.target - base) - rhs)
                minimum_outward_rhs_relaxation = min(
                    minimum_outward_rhs_relaxation, rhs_relaxation
                )
                maximum_outward_rhs_relaxation = max(
                    maximum_outward_rhs_relaxation, rhs_relaxation
                )
            else:
                rhs = legacy_rhs
            tol = 2e-8 * max(1.0, abs(float(target.target)))
            # The formal oracle must preserve the already-audited diagnostic
            # cut *identity and metric coverage*.  Outward arithmetic may add
            # tiny positive marginals or change a floating hash; using those
            # changes for cut selection can hit the global cap early and starve
            # later sigungu/high-need families.  Select anchors/cuts with the
            # frozen legacy rule, then relax every emitted coefficient/RHS.
            if legacy_rhs <= tol:
                continue
            gains_observed = np.asarray(
                B[:, ~covered] @ weights[~covered], dtype=np.float64
            )
            gains_observed[anchor] = 0.0
            legacy_positive = np.flatnonzero(gains_observed > tol)
            if legacy_positive.size == 0:
                impossible.append({"metric": target.name, "source": source, "base_value": base, "target": target.target})
                continue
            if outward_safe:
                gain_term_count = int((~covered).sum())
                gains_upper = np.zeros_like(gains_observed)
                for candidate_index, observed_gain in enumerate(gains_observed):
                    upper, error = _nonnegative_sum_upper(
                        float(observed_gain), gain_term_count
                    )
                    gains_upper[candidate_index] = upper
                    maximum_sum_error = max(maximum_sum_error, error)
                gains = gains_upper
                # Never discard a genuinely positive marginal in the official
                # proof model.  Zero dot products remain exactly zero above.
                positive = np.flatnonzero(gains > 0.0)
            else:
                gains = gains_observed
                positive = legacy_positive
            # The resource-cap relaxation can prove impossibility more strongly
            # than a cardinality-only sum while remaining an optimistic bound.
            if not outward_safe:
                resource_upper, resource_parts = _resource_relaxed_addition_upper(
                    formulation, gains, remaining_slots=formulation.visit_count, forced_index=None
                )
                if resource_upper + tol < rhs:
                    impossible.append(
                        {
                            "metric": target.name,
                            "source": source,
                            "base_value": base,
                            "target": target.target,
                            "resource_relaxed_upper": resource_upper,
                            "resource_relaxation_parts": resource_parts,
                            "required_gain": rhs,
                        }
                    )
            # Add an integer cardinality lower bound over candidates with positive
            # singleton/marginal contribution whenever it is nontrivial.
            sorted_positive = np.sort(gains[positive])[::-1]
            legacy_sorted_positive = np.sort(gains_observed[legacy_positive])[::-1]
            legacy_cumulative = np.cumsum(legacy_sorted_positive)
            legacy_required_count = int(
                np.searchsorted(legacy_cumulative + tol, legacy_rhs, side="left") + 1
            )
            if outward_safe:
                cumulative = np.asarray(
                    [
                        math.nextafter(math.fsum(map(float, sorted_positive[:count])), math.inf)
                        for count in range(1, int(sorted_positive.size) + 1)
                    ],
                    dtype=np.float64,
                )
                required_count = int(np.searchsorted(cumulative, rhs, side="left") + 1)
            else:
                required_count = legacy_required_count
            if (
                1 < legacy_required_count <= formulation.visit_count
                and 1 < required_count <= formulation.visit_count
            ):
                cardinality_key = f"card::{target.name}::{legacy_required_count}::{hashlib.sha1(legacy_positive.tobytes()).hexdigest()[:10]}"
                if cardinality_key not in seen:
                    seen.add(cardinality_key)
                    cuts.append(
                        CutRecord(
                            name=f"stage42d::metric_cardinality::{target.name}::{cardinality_key}",
                            indices=positive.astype(np.int32),
                            values=np.ones(positive.size, dtype=np.float64),
                            lower=float(required_count),
                            upper=math.inf,
                            kind="METRIC_CARDINALITY_LOWER_BOUND",
                            metric=target.name,
                            anchor_size=int(anchor.size),
                            base_value=base,
                            target=float(target.target),
                            source=source,
                            metadata={"required_count": required_count},
                        )
                    )
            # Scale every cut to O(1) coefficients for HiGHS numerical stability.
            if outward_safe:
                # A power-of-two divisor is exact in binary arithmetic.  Round
                # coefficients toward +inf and the lower side toward -inf so
                # both calculation and serialization can only relax the cut.
                scale = _power_of_two_scale(max(rhs, float(gains[positive].max()), 1.0))
                values = np.nextafter(gains[positive] / scale, math.inf)
                lower = math.nextafter(rhs / scale, -math.inf)
            else:
                scale = max(rhs, float(gains[positive].max()), 1.0)
                values = gains[positive] / scale
                lower = rhs / scale
            if outward_safe:
                legacy_scale = max(
                    legacy_rhs,
                    float(gains_observed[legacy_positive].max()),
                    1.0,
                )
                legacy_values = gains_observed[legacy_positive] / legacy_scale
                legacy_lower = legacy_rhs / legacy_scale
                key = _cut_hash(
                    legacy_values, legacy_lower, target.name
                ) + hashlib.sha1(legacy_positive.tobytes()).hexdigest()[:8]
            else:
                key = _cut_hash(values, lower, target.name) + hashlib.sha1(positive.tobytes()).hexdigest()[:8]
            if key in seen:
                continue
            seen.add(key)
            # A cardinality upper bound on this cut can analytically prove the
            # target impossible. This is checked without using policy caps.
            if not outward_safe:
                k = formulation.visit_count
                top = np.partition(gains, -min(k, gains.size))[-min(k, gains.size) :]
                if float(top.sum()) + tol < rhs:
                    impossible.append(
                        {
                            "metric": target.name,
                            "source": source,
                            "base_value": base,
                            "target": target.target,
                            "cut_cardinality_upper": float(top.sum()),
                            "required_gain": rhs,
                        }
                    )
            cuts.append(
                CutRecord(
                    name=f"stage42d::submodular::{target.name}::{key}",
                    indices=positive.astype(np.int32),
                    values=values.astype(np.float64),
                    lower=float(lower),
                    upper=math.inf,
                    kind="ANCHORED_SUBMODULAR_SUPERLEVEL",
                    metric=target.name,
                    anchor_size=int(anchor.size),
                    base_value=base,
                    target=float(target.target),
                    source=source,
                    metadata={
                        "scale": scale,
                        "raw_required_gain": rhs,
                        "outward_safe": bool(outward_safe),
                        "coefficient_rounding_direction": (
                            "POSITIVE_INFINITY" if outward_safe else "NEAREST"
                        ),
                        "rhs_rounding_direction": (
                            "NEGATIVE_INFINITY" if outward_safe else "NEAREST"
                        ),
                        "selection_identity_rule": (
                            "FROZEN_LEGACY_TOLERANCE__OUTWARD_EMISSION"
                            if outward_safe
                            else "LEGACY_TOLERANCE"
                        ),
                    },
                )
            )
            if len(cuts) >= max_cuts:
                break
        if len(cuts) >= max_cuts:
            break
    return cuts, {
        "cut_count": len(cuts),
        "unique_anchor_count": len(unique_anchors),
        "analytic_impossibility_witnesses": impossible,
        "outward_safe": bool(outward_safe),
        "summation_error_bound": "gamma_n_nonnegative_sequential_upper",
        "maximum_sum_error": float(maximum_sum_error),
        "minimum_outward_rhs_relaxation": (
            0.0 if math.isinf(minimum_outward_rhs_relaxation) else float(minimum_outward_rhs_relaxation)
        ),
        "maximum_outward_rhs_relaxation": float(maximum_outward_rhs_relaxation),
        "coefficient_rounding_direction": (
            "POSITIVE_INFINITY" if outward_safe else "NEAREST"
        ),
        "rhs_rounding_direction": (
            "NEGATIVE_INFINITY" if outward_safe else "NEAREST"
        ),
        "cut_records": [
            {
                "name": cut.name,
                "kind": cut.kind,
                "metric": cut.metric,
                "anchor_size": cut.anchor_size,
                "base_value": cut.base_value,
                "target": cut.target,
                "nnz": int(cut.indices.size),
            }
            for cut in cuts
        ],
    }


def serialize_fixings(fixings: Iterable[FixingRecord]) -> list[dict[str, Any]]:
    return [asdict(value) for value in fixings]
