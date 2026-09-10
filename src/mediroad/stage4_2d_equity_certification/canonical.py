"""Fail-closed canonicalization and proof-transport checks.

This module deliberately separates two seals:

``canonical_model_digest``
    Seals the mathematical MIP (objective, canonical CSC matrix, bounds, and
    integrality).  Presentation-only model names and mutable metadata are not
    part of this digest.

``canonical_universe_digest``
    Seals the ordered semantic universe (variable/row identities and model
    slices).  Optional external identifiers can be supplied when the source
    data has stronger identities than solver row/column names.

The transport audit proves a sufficient, not necessary, condition.  It only
passes when the imported model is a relaxation of the intended model.  Row
coefficient drift is accounted for by computing exact binary-rational extrema
over the intended column box; no floating tolerance is used to turn an inward
round-trip into a pass.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

import numpy as np
from scipy import sparse

from mediroad.stage4_2_certification.types import LinearMipModel

from .model_utils import clone_model


_MODEL_DIGEST_VERSION = "mediroad-linear-mip-v1"
_UNIVERSE_DIGEST_VERSION = "mediroad-linear-mip-universe-v1"
ProofKind = Literal["infeasibility", "max_upper_bound", "min_lower_bound"]


def canonicalize_csc(matrix: sparse.spmatrix) -> sparse.csc_matrix:
    """Return an owned, sorted CSC matrix with duplicates and zeros removed.

    Sparse storage order, index integer width, explicit zero entries, and the
    ``-0.0`` spelling are transport details and therefore canonicalized away.
    Non-finite coefficients fail closed because they have no valid LP/MIP
    transport interpretation.
    """

    if not sparse.issparse(matrix):
        raise TypeError("matrix must be a scipy sparse matrix")
    output = matrix.astype(np.float64, copy=True).tocsc(copy=True)
    output.sum_duplicates()
    output.sort_indices()
    if output.data.size and not np.isfinite(output.data).all():
        raise ValueError("constraint matrix contains a non-finite coefficient")
    if output.data.size:
        output.data[output.data == 0.0] = 0.0
    output.eliminate_zeros()
    output.sort_indices()
    # Stable cross-platform integer widths are used by the digest routine.
    output.indices = np.asarray(output.indices, dtype=np.int64)
    output.indptr = np.asarray(output.indptr, dtype=np.int64)
    return output


def _checked_float_array(
    values: Any,
    *,
    name: str,
    allow_infinity: bool,
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if np.isnan(array).any():
        raise ValueError(f"{name} contains NaN")
    if not allow_infinity and not np.isfinite(array).all():
        raise ValueError(f"{name} contains a non-finite value")
    output = np.ascontiguousarray(array, dtype="<f8").copy()
    output[output == 0.0] = 0.0
    return output


def _hash_bytes(hasher: Any, label: str, payload: bytes) -> None:
    label_bytes = label.encode("utf-8")
    hasher.update(len(label_bytes).to_bytes(4, "big"))
    hasher.update(label_bytes)
    hasher.update(len(payload).to_bytes(8, "big"))
    hasher.update(payload)


def _hash_array(hasher: Any, label: str, values: np.ndarray) -> None:
    shape = np.asarray(values.shape, dtype=">i8").tobytes()
    _hash_bytes(hasher, f"{label}.shape", shape)
    _hash_bytes(hasher, f"{label}.data", np.ascontiguousarray(values).tobytes())


def canonical_model_digest(model: LinearMipModel) -> str:
    """Return the deterministic SHA-256 seal of the mathematical MIP.

    The digest includes the objective sense, raw objective vector, reported
    scale/offset, canonical CSC matrix, row/column bounds, and integrality.
    Variable and row identities are intentionally sealed separately by
    :func:`canonical_universe_digest`.
    """

    model.validate()
    matrix = canonicalize_csc(model.A)
    c = _checked_float_array(model.c, name="objective", allow_infinity=False)
    row_lower = _checked_float_array(
        model.row_lower, name="row_lower", allow_infinity=True
    )
    row_upper = _checked_float_array(
        model.row_upper, name="row_upper", allow_infinity=True
    )
    col_lower = _checked_float_array(
        model.col_lower, name="col_lower", allow_infinity=True
    )
    col_upper = _checked_float_array(
        model.col_upper, name="col_upper", allow_infinity=True
    )
    if np.any(row_lower > row_upper):
        raise ValueError("row lower bound exceeds upper bound")
    if np.any(col_lower > col_upper):
        raise ValueError("column lower bound exceeds upper bound")
    integrality = np.ascontiguousarray(model.integrality, dtype="<i8")
    if np.any((integrality != 0) & (integrality != 1)):
        raise ValueError("only continuous (0) and integer (1) columns are supported")
    scale = float(model.reported_objective_scale)
    offset = float(model.objective_offset)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("reported_objective_scale must be finite and positive")
    if not math.isfinite(offset):
        raise ValueError("objective_offset must be finite")

    hasher = hashlib.sha256()
    _hash_bytes(hasher, "format", _MODEL_DIGEST_VERSION.encode("ascii"))
    _hash_bytes(hasher, "objective_name", str(model.objective_name).encode("utf-8"))
    _hash_bytes(hasher, "sense", str(model.sense).encode("ascii"))
    _hash_array(hasher, "c", c)
    _hash_array(
        hasher,
        "reported_scale_offset",
        _checked_float_array([scale, offset], name="objective metadata", allow_infinity=False),
    )
    _hash_array(hasher, "A.indptr", np.asarray(matrix.indptr, dtype=">i8"))
    _hash_array(hasher, "A.indices", np.asarray(matrix.indices, dtype=">i8"))
    _hash_array(
        hasher,
        "A.data",
        _checked_float_array(matrix.data, name="A.data", allow_infinity=False),
    )
    _hash_array(hasher, "A.shape", np.asarray(matrix.shape, dtype=">i8"))
    _hash_array(hasher, "row_lower", row_lower)
    _hash_array(hasher, "row_upper", row_upper)
    _hash_array(hasher, "col_lower", col_lower)
    _hash_array(hasher, "col_upper", col_upper)
    _hash_array(hasher, "integrality", integrality)
    return hasher.hexdigest()


def _identifier(value: Any) -> list[Any]:
    if isinstance(value, np.generic):
        value = value.item()
    if value is None:
        return ["none", None]
    if isinstance(value, bool):
        return ["bool", value]
    if isinstance(value, int):
        return ["int", str(value)]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("universe identifiers cannot contain NaN or infinity")
        return ["float64", float(value).hex()]
    if isinstance(value, str):
        return ["str", value]
    raise TypeError(f"unsupported universe identifier type: {type(value).__name__}")


def _slice_payload(value: slice | int) -> list[Any]:
    if isinstance(value, slice):
        return ["slice", value.start, value.stop, value.step]
    return ["index", int(value)]


def canonical_universe_digest(
    model: LinearMipModel,
    *,
    candidate_ids: Sequence[Any] | None = None,
    pattern_ids: Sequence[Any] | None = None,
    semantic_ids: Mapping[str, Sequence[Any]] | None = None,
) -> str:
    """Seal ordered semantic identities independently of numeric coefficients.

    By default candidate and pattern identities are taken from the model's
    ``x_slice`` and ``y_slice`` variable names.  Callers should provide stronger
    source identifiers (for example sealed pattern/grid keys) when available.
    Mapping keys are sorted, while every identifier sequence remains ordered;
    consequently dictionary insertion order is irrelevant but universe order
    drift is detected.
    """

    model.validate()
    variable_names = list(model.variable_names)
    row_names_raw = model.metadata.get("row_names")
    if row_names_raw is None:
        row_names = [f"__position__::{index}" for index in range(model.n_row)]
    else:
        row_names = list(row_names_raw)
        if len(row_names) != model.n_row:
            raise ValueError("row_names length differs from model row count")

    default_candidate_ids = variable_names[model.x_slice]
    default_pattern_ids = variable_names[model.y_slice]
    candidates = list(default_candidate_ids if candidate_ids is None else candidate_ids)
    patterns = list(default_pattern_ids if pattern_ids is None else pattern_ids)
    if len(candidates) != len(default_candidate_ids):
        raise ValueError("candidate_ids length differs from x_slice width")
    if len(patterns) != len(default_pattern_ids):
        raise ValueError("pattern_ids length differs from y_slice width")

    external = {
        str(key): [_identifier(value) for value in values]
        for key, values in sorted((semantic_ids or {}).items(), key=lambda item: str(item[0]))
    }
    if len(external) != len(semantic_ids or {}):
        raise ValueError("semantic_ids keys collide after string conversion")
    payload = {
        "format": _UNIVERSE_DIGEST_VERSION,
        "shape": [model.n_row, model.n_col],
        "variable_names": [_identifier(value) for value in variable_names],
        "row_names": [_identifier(value) for value in row_names],
        "candidate_ids": [_identifier(value) for value in candidates],
        "pattern_ids": [_identifier(value) for value in patterns],
        "x_slice": _slice_payload(model.x_slice),
        "y_slice": _slice_payload(model.y_slice),
        "extra_slices": [
            [str(key), _slice_payload(value)]
            for key, value in sorted(model.extra_slices.items(), key=lambda item: str(item[0]))
        ],
        "semantic_ids": external,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ProofRelaxation:
    """Exact owned copy plus its explicitly outward proof relaxation."""

    exact_model: LinearMipModel
    relaxed_model: LinearMipModel
    absolute_margin: float
    relative_margin: float
    ulps: int
    manifest: dict[str, Any]


def _relax_bound_array(
    values: np.ndarray,
    *,
    direction: Literal["lower", "upper"],
    absolute_margin: float,
    relative_margin: float,
    ulps: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    source = _checked_float_array(values, name=direction, allow_infinity=True)
    output = source.copy()
    finite = np.isfinite(source)
    margins = absolute_margin + relative_margin * np.maximum(1.0, np.abs(source[finite]))
    if direction == "lower":
        output[finite] = source[finite] - margins
        toward = -math.inf
    else:
        output[finite] = source[finite] + margins
        toward = math.inf
    for _ in range(ulps):
        output[finite] = np.nextafter(output[finite], toward)
    realized = np.abs(output[finite] - source[finite])
    summary = {
        "finite_count": int(finite.sum()),
        "infinite_count": int((~finite).sum()),
        "requested_margin_min": float(margins.min(initial=math.inf))
        if margins.size
        else None,
        "requested_margin_max": float(margins.max(initial=-math.inf))
        if margins.size
        else None,
        "realized_margin_min": float(realized.min(initial=math.inf))
        if realized.size
        else None,
        "realized_margin_max": float(realized.max(initial=-math.inf))
        if realized.size
        else None,
        "direction": "downward" if direction == "lower" else "upward",
    }
    return output, summary


def relax_model_for_proof(
    model: LinearMipModel,
    *,
    absolute_margin: float = 0.0,
    relative_margin: float = 1e-12,
    ulps: int = 1,
    name: str | None = None,
) -> ProofRelaxation:
    """Copy ``model`` and relax every finite row/column bound outward.

    For a finite bound ``b`` the requested margin is
    ``absolute_margin + relative_margin * max(1, abs(b))``.  Lower bounds are
    moved down and upper bounds up, followed by ``ulps`` representable steps in
    the same direction.  The source is never mutated; ``exact_model`` and
    ``relaxed_model`` own independent arrays, sparse storage, and metadata.
    """

    absolute_margin = float(absolute_margin)
    relative_margin = float(relative_margin)
    if not math.isfinite(absolute_margin) or absolute_margin < 0.0:
        raise ValueError("absolute_margin must be finite and non-negative")
    if not math.isfinite(relative_margin) or relative_margin < 0.0:
        raise ValueError("relative_margin must be finite and non-negative")
    if isinstance(ulps, bool) or int(ulps) != ulps or int(ulps) < 1:
        raise ValueError("ulps must be a positive integer")
    ulps = int(ulps)

    exact = clone_model(model)
    relaxed = clone_model(exact, name=name or f"{model.name}__proof_relaxed")
    summaries: dict[str, Any] = {}
    relaxed.row_lower, summaries["row_lower"] = _relax_bound_array(
        exact.row_lower,
        direction="lower",
        absolute_margin=absolute_margin,
        relative_margin=relative_margin,
        ulps=ulps,
    )
    relaxed.row_upper, summaries["row_upper"] = _relax_bound_array(
        exact.row_upper,
        direction="upper",
        absolute_margin=absolute_margin,
        relative_margin=relative_margin,
        ulps=ulps,
    )
    relaxed.col_lower, summaries["col_lower"] = _relax_bound_array(
        exact.col_lower,
        direction="lower",
        absolute_margin=absolute_margin,
        relative_margin=relative_margin,
        ulps=ulps,
    )
    relaxed.col_upper, summaries["col_upper"] = _relax_bound_array(
        exact.col_upper,
        direction="upper",
        absolute_margin=absolute_margin,
        relative_margin=relative_margin,
        ulps=ulps,
    )
    manifest: dict[str, Any] = {
        "formula": "abs_margin + rel_margin * max(1, abs(bound)), then ulps outward",
        "absolute_margin": absolute_margin,
        "relative_margin": relative_margin,
        "ulps": ulps,
        "families": summaries,
        "exact_model_sha256": canonical_model_digest(exact),
    }
    relaxed.metadata.setdefault("stage42d", {})["proof_relaxation"] = manifest
    relaxed.validate()
    manifest["relaxed_model_sha256"] = canonical_model_digest(relaxed)
    return ProofRelaxation(
        exact_model=exact,
        relaxed_model=relaxed,
        absolute_margin=absolute_margin,
        relative_margin=relative_margin,
        ulps=ulps,
        manifest=manifest,
    )


# A dyadic is an exact integer times a power of two.  Every finite binary64
# value, and every sum/product of such values, has this representation.
_Dyad = tuple[int, int]
_ExtendedDyad = _Dyad | float


def _dyad(value: float) -> _Dyad:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("a finite value is required")
    if value == 0.0:
        return (0, 0)
    numerator, denominator = value.as_integer_ratio()
    return numerator, -(denominator.bit_length() - 1)


def _dyad_add(left: _Dyad, right: _Dyad) -> _Dyad:
    exponent = min(left[1], right[1])
    numerator = (left[0] << (left[1] - exponent)) + (
        right[0] << (right[1] - exponent)
    )
    return (0, 0) if numerator == 0 else (numerator, exponent)


def _dyad_sub(left: _Dyad, right: _Dyad) -> _Dyad:
    return _dyad_add(left, (-right[0], right[1]))


def _dyad_mul(left: _Dyad, right: _Dyad) -> _Dyad:
    numerator = left[0] * right[0]
    return (0, 0) if numerator == 0 else (numerator, left[1] + right[1])


def _dyad_compare(left: _Dyad, right: _Dyad) -> int:
    difference = _dyad_sub(left, right)[0]
    return (difference > 0) - (difference < 0)


def _extended_add(left: _ExtendedDyad, right: _ExtendedDyad) -> _ExtendedDyad:
    if isinstance(left, float) and math.isinf(left):
        return left
    if isinstance(right, float) and math.isinf(right):
        return right
    assert not isinstance(left, float) and not isinstance(right, float)
    return _dyad_add(left, right)


def _extended_le(left: _ExtendedDyad, right: _ExtendedDyad) -> bool:
    if isinstance(left, float) and math.isinf(left):
        if left < 0:
            return True
        return isinstance(right, float) and right > 0
    if isinstance(right, float) and math.isinf(right):
        return right > 0
    assert not isinstance(left, float) and not isinstance(right, float)
    return _dyad_compare(left, right) <= 0


def _exact_float_difference(left: float, right: float) -> _Dyad:
    return _dyad_sub(_dyad(left), _dyad(right))


def _product_with_bound(coefficient: _Dyad, bound: float) -> _ExtendedDyad:
    if coefficient[0] == 0:
        return (0, 0)
    if math.isinf(float(bound)):
        sign = (1 if coefficient[0] > 0 else -1) * (1 if bound > 0 else -1)
        return math.inf if sign > 0 else -math.inf
    return _dyad_mul(coefficient, _dyad(float(bound)))


def _exact_box_extrema(
    terms: Sequence[tuple[int, _Dyad]],
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[_ExtendedDyad, _ExtendedDyad]:
    minimum: _ExtendedDyad = (0, 0)
    maximum: _ExtendedDyad = (0, 0)
    for index, coefficient in terms:
        if coefficient[0] > 0:
            low_bound, high_bound = lower[index], upper[index]
        else:
            low_bound, high_bound = upper[index], lower[index]
        minimum = _extended_add(minimum, _product_with_bound(coefficient, low_bound))
        maximum = _extended_add(maximum, _product_with_bound(coefficient, high_bound))
    return minimum, maximum


def _dyad_to_outward_float(value: _ExtendedDyad, direction: Literal["lower", "upper"]) -> float:
    if isinstance(value, float):
        return value
    numerator, exponent = value
    try:
        if exponent >= 0:
            converted = float(numerator << exponent)
        else:
            converted = float(numerator / (1 << -exponent))
    except OverflowError:
        converted = math.copysign(math.inf, numerator)
    if math.isinf(converted):
        if direction == "lower" and numerator > 0:
            return np.finfo(np.float64).max
        if direction == "upper" and numerator < 0:
            return -np.finfo(np.float64).max
        return converted
    converted_dyad = _dyad(converted)
    comparison = _dyad_compare(converted_dyad, value)
    if direction == "lower" and comparison > 0:
        converted = math.nextafter(converted, -math.inf)
    elif direction == "upper" and comparison < 0:
        converted = math.nextafter(converted, math.inf)
    return converted


def box_linear_extrema(
    coefficients: Sequence[float] | np.ndarray,
    lower: Sequence[float] | np.ndarray,
    upper: Sequence[float] | np.ndarray,
) -> tuple[float, float]:
    """Return outward-rounded exact extrema of ``coefficients @ x`` on a box."""

    coeff = _checked_float_array(coefficients, name="coefficients", allow_infinity=False)
    low = _checked_float_array(lower, name="lower", allow_infinity=True)
    high = _checked_float_array(upper, name="upper", allow_infinity=True)
    if coeff.ndim != 1 or low.ndim != 1 or high.ndim != 1:
        raise ValueError("coefficients and bounds must be one-dimensional")
    if not (coeff.size == low.size == high.size):
        raise ValueError("coefficient and box lengths differ")
    if np.any(low > high):
        raise ValueError("box lower bound exceeds upper bound")
    terms = [(index, _dyad(float(value))) for index, value in enumerate(coeff) if value != 0.0]
    minimum, maximum = _exact_box_extrema(terms, low, high)
    return (
        _dyad_to_outward_float(minimum, "lower"),
        _dyad_to_outward_float(maximum, "upper"),
    )


def _row_difference_terms(
    intended: sparse.csr_matrix,
    imported: sparse.csr_matrix,
    row: int,
) -> list[tuple[int, _Dyad]]:
    left_start, left_stop = intended.indptr[row : row + 2]
    right_start, right_stop = imported.indptr[row : row + 2]
    left_indices = intended.indices[left_start:left_stop]
    right_indices = imported.indices[right_start:right_stop]
    left_data = intended.data[left_start:left_stop]
    right_data = imported.data[right_start:right_stop]
    li = ri = 0
    result: list[tuple[int, _Dyad]] = []
    while li < len(left_indices) or ri < len(right_indices):
        if ri >= len(right_indices) or (
            li < len(left_indices) and left_indices[li] < right_indices[ri]
        ):
            index = int(left_indices[li])
            difference = _exact_float_difference(0.0, float(left_data[li]))
            li += 1
        elif li >= len(left_indices) or right_indices[ri] < left_indices[li]:
            index = int(right_indices[ri])
            difference = _dyad(float(right_data[ri]))
            ri += 1
        else:
            index = int(left_indices[li])
            difference = _exact_float_difference(
                float(right_data[ri]), float(left_data[li])
            )
            li += 1
            ri += 1
        if difference[0] != 0:
            result.append((index, difference))
    return result


def _finite_or_infinity_dyad(value: float) -> _ExtendedDyad:
    return float(value) if math.isinf(float(value)) else _dyad(float(value))


def _bound_plus_extremum(bound: float, extremum: _ExtendedDyad) -> _ExtendedDyad:
    if math.isinf(float(bound)):
        return float(bound)
    return _extended_add(_dyad(float(bound)), extremum)


@dataclass(frozen=True)
class TransportAudit:
    """Machine-readable sufficient-condition report for proof transport."""

    sufficient: bool
    proof_kind: ProofKind
    shape_equal: bool
    universe_equal: bool
    column_box_superset: bool
    integrality_not_stricter: bool
    row_relaxation_superset: bool
    objective_dominance: bool
    intended_model_sha256: str
    imported_model_sha256: str
    intended_universe_sha256: str
    imported_universe_sha256: str
    canonical_nnz: tuple[int, int]
    row_box_extrema: tuple[tuple[float, float], ...]
    row_lower_failures: tuple[int, ...]
    row_upper_failures: tuple[int, ...]
    column_lower_failures: tuple[int, ...]
    column_upper_failures: tuple[int, ...]
    integrality_failures: tuple[int, ...]
    objective_box_extrema: tuple[float, float] | None
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "sufficient": self.sufficient,
            "proof_kind": self.proof_kind,
            "shape_equal": self.shape_equal,
            "universe_equal": self.universe_equal,
            "column_box_superset": self.column_box_superset,
            "integrality_not_stricter": self.integrality_not_stricter,
            "row_relaxation_superset": self.row_relaxation_superset,
            "objective_dominance": self.objective_dominance,
            "intended_model_sha256": self.intended_model_sha256,
            "imported_model_sha256": self.imported_model_sha256,
            "intended_universe_sha256": self.intended_universe_sha256,
            "imported_universe_sha256": self.imported_universe_sha256,
            "canonical_nnz": list(self.canonical_nnz),
            "row_box_extrema": [list(item) for item in self.row_box_extrema],
            "row_lower_failures": list(self.row_lower_failures),
            "row_upper_failures": list(self.row_upper_failures),
            "column_lower_failures": list(self.column_lower_failures),
            "column_upper_failures": list(self.column_upper_failures),
            "integrality_failures": list(self.integrality_failures),
            "objective_box_extrema": list(self.objective_box_extrema)
            if self.objective_box_extrema is not None
            else None,
            "reasons": list(self.reasons),
        }


def audit_model_transport(
    intended_relaxed: LinearMipModel,
    imported: LinearMipModel,
    *,
    proof_kind: ProofKind = "infeasibility",
    intended_universe_digest: str | None = None,
    imported_universe_digest: str | None = None,
) -> TransportAudit:
    """Audit whether an imported model safely relaxes an intended model.

    For each aligned row, let ``d = A_imported - A_intended``.  On the
    intended column box the exact extrema ``d_min`` and ``d_max`` are computed.
    The sufficient row conditions are then::

        imported_lower <= intended_lower + d_min
        imported_upper >= intended_upper + d_max

    Together with an outward imported column box and no stricter integrality,
    these conditions establish feasible-set containment.  Objective proof
    kinds additionally require pointwise objective dominance on that box.
    """

    if proof_kind not in {"infeasibility", "max_upper_bound", "min_lower_bound"}:
        raise ValueError(f"unsupported proof_kind: {proof_kind}")
    intended_relaxed.validate()
    imported.validate()
    intended_matrix = canonicalize_csc(intended_relaxed.A)
    imported_matrix = canonicalize_csc(imported.A)
    intended_digest = canonical_model_digest(intended_relaxed)
    imported_digest = canonical_model_digest(imported)
    intended_universe = intended_universe_digest or canonical_universe_digest(
        intended_relaxed
    )
    imported_universe = imported_universe_digest or canonical_universe_digest(imported)
    shape_equal = intended_matrix.shape == imported_matrix.shape
    universe_equal = intended_universe == imported_universe
    reasons: list[str] = []
    if not shape_equal:
        reasons.append("row/column shapes differ")
    if not universe_equal:
        reasons.append("ordered semantic universes differ")

    column_lower_failures: tuple[int, ...] = ()
    column_upper_failures: tuple[int, ...] = ()
    integrality_failures: tuple[int, ...] = ()
    row_lower_failures: tuple[int, ...] = ()
    row_upper_failures: tuple[int, ...] = ()
    row_extrema: list[tuple[float, float]] = []
    column_box_superset = False
    integrality_not_stricter = False
    row_relaxation_superset = False
    objective_dominance = proof_kind == "infeasibility"
    objective_extrema: tuple[float, float] | None = None

    if shape_equal:
        intended_low = _checked_float_array(
            intended_relaxed.col_lower, name="intended col_lower", allow_infinity=True
        )
        intended_high = _checked_float_array(
            intended_relaxed.col_upper, name="intended col_upper", allow_infinity=True
        )
        imported_low = _checked_float_array(
            imported.col_lower, name="imported col_lower", allow_infinity=True
        )
        imported_high = _checked_float_array(
            imported.col_upper, name="imported col_upper", allow_infinity=True
        )
        column_lower_failures = tuple(
            int(value) for value in np.flatnonzero(imported_low > intended_low)
        )
        column_upper_failures = tuple(
            int(value) for value in np.flatnonzero(imported_high < intended_high)
        )
        column_box_superset = not column_lower_failures and not column_upper_failures
        if not column_box_superset:
            reasons.append("imported column box is inward")

        intended_integer = np.asarray(intended_relaxed.integrality, dtype=np.int8)
        imported_integer = np.asarray(imported.integrality, dtype=np.int8)
        supported = np.all((intended_integer == 0) | (intended_integer == 1)) and np.all(
            (imported_integer == 0) | (imported_integer == 1)
        )
        if supported:
            # 0 (continuous) is a relaxation of 1 (integer); the reverse is not.
            integrality_failures = tuple(
                int(value)
                for value in np.flatnonzero(imported_integer > intended_integer)
            )
            integrality_not_stricter = not integrality_failures
        if not integrality_not_stricter:
            reasons.append("imported integrality is unsupported or stricter")

        intended_csr = intended_matrix.tocsr()
        imported_csr = imported_matrix.tocsr()
        lower_failures: list[int] = []
        upper_failures: list[int] = []
        for row in range(intended_relaxed.n_row):
            terms = _row_difference_terms(intended_csr, imported_csr, row)
            minimum, maximum = _exact_box_extrema(terms, intended_low, intended_high)
            row_extrema.append(
                (
                    _dyad_to_outward_float(minimum, "lower"),
                    _dyad_to_outward_float(maximum, "upper"),
                )
            )
            lower_rhs = _bound_plus_extremum(
                float(intended_relaxed.row_lower[row]), minimum
            )
            upper_rhs = _bound_plus_extremum(
                float(intended_relaxed.row_upper[row]), maximum
            )
            if not _extended_le(
                _finite_or_infinity_dyad(float(imported.row_lower[row])), lower_rhs
            ):
                lower_failures.append(row)
            if not _extended_le(
                upper_rhs, _finite_or_infinity_dyad(float(imported.row_upper[row]))
            ):
                upper_failures.append(row)
        row_lower_failures = tuple(lower_failures)
        row_upper_failures = tuple(upper_failures)
        row_relaxation_superset = not row_lower_failures and not row_upper_failures
        if not row_relaxation_superset:
            reasons.append("imported row system is not an outward transport")

        if proof_kind != "infeasibility":
            expected_sense = "max" if proof_kind == "max_upper_bound" else "min"
            same_scale = (
                float(imported.reported_objective_scale)
                == float(intended_relaxed.reported_objective_scale)
                and math.isfinite(float(imported.reported_objective_scale))
                and float(imported.reported_objective_scale) > 0.0
            )
            sense_ok = (
                imported.sense == intended_relaxed.sense == expected_sense
            )
            if same_scale and sense_ok:
                objective_terms = [
                    (index, difference)
                    for index, (left, right) in enumerate(
                        zip(imported.c, intended_relaxed.c, strict=True)
                    )
                    if (difference := _exact_float_difference(float(left), float(right)))[0]
                    != 0
                ]
                obj_min, obj_max = _exact_box_extrema(
                    objective_terms, intended_low, intended_high
                )
                objective_extrema = (
                    _dyad_to_outward_float(obj_min, "lower"),
                    _dyad_to_outward_float(obj_max, "upper"),
                )
                scale = _dyad(float(imported.reported_objective_scale))
                offset_delta = _exact_float_difference(
                    float(imported.objective_offset),
                    float(intended_relaxed.objective_offset),
                )
                # Multiplication by the shared positive scale avoids dividing
                # dyadics and preserves the sign of the reported difference.
                scaled_offset = _dyad_mul(scale, offset_delta)
                if proof_kind == "max_upper_bound":
                    dominance_witness = _extended_add(obj_min, scaled_offset)
                    objective_dominance = _extended_le((0, 0), dominance_witness)
                else:
                    dominance_witness = _extended_add(obj_max, scaled_offset)
                    objective_dominance = _extended_le(dominance_witness, (0, 0))
            if not objective_dominance:
                reasons.append(
                    "imported objective does not dominate the intended objective on the box"
                )

    sufficient = all(
        (
            shape_equal,
            universe_equal,
            column_box_superset,
            integrality_not_stricter,
            row_relaxation_superset,
            objective_dominance,
        )
    )
    return TransportAudit(
        sufficient=sufficient,
        proof_kind=proof_kind,
        shape_equal=shape_equal,
        universe_equal=universe_equal,
        column_box_superset=column_box_superset,
        integrality_not_stricter=integrality_not_stricter,
        row_relaxation_superset=row_relaxation_superset,
        objective_dominance=objective_dominance,
        intended_model_sha256=intended_digest,
        imported_model_sha256=imported_digest,
        intended_universe_sha256=intended_universe,
        imported_universe_sha256=imported_universe,
        canonical_nnz=(int(intended_matrix.nnz), int(imported_matrix.nnz)),
        row_box_extrema=tuple(row_extrema),
        row_lower_failures=row_lower_failures,
        row_upper_failures=row_upper_failures,
        column_lower_failures=column_lower_failures,
        column_upper_failures=column_upper_failures,
        integrality_failures=integrality_failures,
        objective_box_extrema=objective_extrema,
        reasons=tuple(reasons),
    )


# Integration-facing names spell out the safety property.  The shorter names
# remain available for existing callers and tests.
make_outward_relaxation = relax_model_for_proof
audit_relaxing_transport = audit_model_transport


__all__ = [
    "ProofRelaxation",
    "TransportAudit",
    "audit_model_transport",
    "audit_relaxing_transport",
    "box_linear_extrema",
    "canonical_model_digest",
    "canonical_universe_digest",
    "canonicalize_csc",
    "make_outward_relaxation",
    "relax_model_for_proof",
]
