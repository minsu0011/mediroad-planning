from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Literal

import numpy as np

from mediroad.stage4_2_certification.types import LinearMipModel
from mediroad.stage4_2d_equity_certification.canonical import (
    canonical_model_digest,
    canonicalize_csc,
)

Sense = Literal["max", "min"]


@dataclass(frozen=True)
class SubsetAudit:
    sufficient: bool
    proof: str
    parent_model_sha256: str
    child_model_sha256: str
    shape_equal: bool
    objective_equal: bool
    universe_equal: bool
    matrix_equal: bool
    variable_domain_equal: bool
    row_box_subset: bool
    tightened_row_count: int
    tightened_rows: list[dict[str, Any]]
    failures: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GapAudit:
    sense: Sense
    incumbent: float
    best_bound: float
    relative_gap: float
    bound_direction_valid: bool
    within_limit: bool
    exact: bool
    semantic_label: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _array_equal(left: Any, right: Any) -> bool:
    return bool(np.array_equal(np.asarray(left), np.asarray(right), equal_nan=False))


def _json_bound(value: float) -> float | str:
    value = float(value)
    if math.isinf(value):
        return "+INF" if value > 0 else "-INF"
    return value


def exact_feasible_set_subset(parent: LinearMipModel, child: LinearMipModel) -> SubsetAudit:
    """Prove ``feasible(child)`` is a subset of ``feasible(parent)``.

    This deliberately accepts only the simple, machine-auditable case needed by
    Stage4.2E: identical variables, domains and row coefficients, with every
    child row interval contained in the corresponding parent row interval.
    No numerical tolerance is used to manufacture a subset relation.
    """

    parent.validate()
    child.validate()
    p_matrix = canonicalize_csc(parent.A)
    c_matrix = canonicalize_csc(child.A)
    failures: list[str] = []
    shape_equal = (parent.n_row, parent.n_col) == (child.n_row, child.n_col)
    if not shape_equal:
        failures.append("shape")
    objective_equal = bool(
        parent.objective_name == child.objective_name
        and parent.sense == child.sense
        and _array_equal(parent.c, child.c)
        and float(parent.reported_objective_scale).hex()
        == float(child.reported_objective_scale).hex()
        and float(parent.objective_offset).hex() == float(child.objective_offset).hex()
    )
    if not objective_equal:
        failures.append("objective")
    universe_equal = bool(
        parent.variable_names == child.variable_names
        and parent.x_slice == child.x_slice
        and parent.y_slice == child.y_slice
        and parent.extra_slices == child.extra_slices
        and list(parent.metadata.get("row_names", []))
        == list(child.metadata.get("row_names", []))
    )
    if not universe_equal:
        failures.append("universe")
    matrix_equal = bool(
        p_matrix.shape == c_matrix.shape
        and _array_equal(p_matrix.indptr, c_matrix.indptr)
        and _array_equal(p_matrix.indices, c_matrix.indices)
        and _array_equal(p_matrix.data, c_matrix.data)
    )
    if not matrix_equal:
        failures.append("matrix")
    variable_domain_equal = bool(
        _array_equal(parent.col_lower, child.col_lower)
        and _array_equal(parent.col_upper, child.col_upper)
        and _array_equal(parent.integrality, child.integrality)
    )
    if not variable_domain_equal:
        failures.append("variable_domain")

    row_box_subset = False
    tightened: list[dict[str, Any]] = []
    if shape_equal:
        # Infinity comparisons have the desired set-inclusion semantics.
        lower_ok = np.asarray(child.row_lower) >= np.asarray(parent.row_lower)
        upper_ok = np.asarray(child.row_upper) <= np.asarray(parent.row_upper)
        row_box_subset = bool(np.all(lower_ok) and np.all(upper_ok))
        row_names = list(child.metadata.get("row_names", []))
        changed = np.flatnonzero(
            (np.asarray(child.row_lower) != np.asarray(parent.row_lower))
            | (np.asarray(child.row_upper) != np.asarray(parent.row_upper))
        )
        for row in changed:
            tightened.append(
                {
                    "row_index": int(row),
                    "row_name": row_names[int(row)] if int(row) < len(row_names) else None,
                    "parent_lower": _json_bound(parent.row_lower[row]),
                    "child_lower": _json_bound(child.row_lower[row]),
                    "parent_upper": _json_bound(parent.row_upper[row]),
                    "child_upper": _json_bound(child.row_upper[row]),
                    "contained": bool(lower_ok[row] and upper_ok[row]),
                }
            )
    if not row_box_subset:
        failures.append("row_box_subset")

    sufficient = bool(
        shape_equal
        and objective_equal
        and universe_equal
        and matrix_equal
        and variable_domain_equal
        and row_box_subset
    )
    return SubsetAudit(
        sufficient=sufficient,
        proof="IDENTICAL_LINEAR_SYSTEM_WITH_CHILD_ROW_INTERVALS_CONTAINED_IN_PARENT",
        parent_model_sha256=canonical_model_digest(parent),
        child_model_sha256=canonical_model_digest(child),
        shape_equal=shape_equal,
        objective_equal=objective_equal,
        universe_equal=universe_equal,
        matrix_equal=matrix_equal,
        variable_domain_equal=variable_domain_equal,
        row_box_subset=row_box_subset,
        tightened_row_count=len(tightened),
        tightened_rows=tightened,
        failures=failures,
    )


def inherited_bound_allowed(*, subset: SubsetAudit, artifact_sha_verified: bool) -> bool:
    return bool(subset.sufficient and artifact_sha_verified)


def combine_valid_bounds(sense: Sense, fresh: float | None, inherited: float | None) -> float | None:
    values = [float(value) for value in (fresh, inherited) if value is not None and math.isfinite(float(value))]
    if not values:
        return None
    # A maximization certificate wants the smallest valid upper bound. A
    # minimization certificate wants the largest valid lower bound.
    return min(values) if sense == "max" else max(values)


def choose_incumbent(
    sense: Sense,
    candidates: Iterable[tuple[str, float, np.ndarray]],
) -> tuple[str, float, np.ndarray]:
    values = [(str(source), float(value), np.asarray(selection, dtype=int)) for source, value, selection in candidates]
    if not values:
        raise RuntimeError("No feasible incumbent candidate")
    key = (lambda item: item[1])
    return (max(values, key=key) if sense == "max" else min(values, key=key))


def audit_gap(sense: Sense, incumbent: float, best_bound: float, limit: float) -> GapAudit:
    incumbent = float(incumbent)
    best_bound = float(best_bound)
    limit = float(limit)
    if not all(math.isfinite(value) for value in (incumbent, best_bound, limit)) or limit < 0:
        raise ValueError("Gap inputs must be finite and limit non-negative")
    scale = max(abs(incumbent), 1e-12)
    slack = 1e-10 * max(1.0, abs(incumbent), abs(best_bound))
    if sense == "max":
        direction = best_bound >= incumbent - slack
        gap = max(0.0, (best_bound - incumbent) / scale) if direction else math.inf
    elif sense == "min":
        direction = best_bound <= incumbent + slack
        gap = max(0.0, (incumbent - best_bound) / scale) if direction else math.inf
    else:
        raise ValueError(f"Unsupported sense: {sense}")
    exact = bool(direction and gap <= 1e-9)
    within = bool(direction and gap <= limit + 1e-12)
    label = "CERTIFIED_EXACT" if exact else ("CERTIFIED_NEAR_OPTIMAL" if within else "UNCERTIFIED")
    return GapAudit(
        sense=sense,
        incumbent=incumbent,
        best_bound=best_bound,
        relative_gap=float(gap),
        bound_direction_valid=direction,
        within_limit=within,
        exact=exact,
        semantic_label=label,
    )
