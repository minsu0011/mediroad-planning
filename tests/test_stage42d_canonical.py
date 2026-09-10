from __future__ import annotations

import itertools
import math
from pathlib import Path

import mediroad
import numpy as np
import pytest
from scipy import sparse

# The versioned patch is intentionally overlay-shaped and has no second
# ``mediroad/__init__.py``.  Extend the installed package search path so this
# test exercises the patch source before installation as well as after it.
_PATCH_NAMESPACE = str(Path(__file__).resolve().parents[1] / "src" / "mediroad")
if _PATCH_NAMESPACE not in mediroad.__path__:
    mediroad.__path__.insert(0, _PATCH_NAMESPACE)

from mediroad.stage4_2_certification.types import LinearMipModel
from mediroad.stage4_2d_equity_certification.canonical import (
    audit_model_transport,
    box_linear_extrema,
    canonical_model_digest,
    canonical_universe_digest,
    canonicalize_csc,
    relax_model_for_proof,
)
from mediroad.stage4_2d_equity_certification.model_utils import clone_model


def _tiny_model() -> LinearMipModel:
    model = LinearMipModel(
        name="tiny",
        objective_name="high_need_population",
        sense="max",
        c=np.asarray([2.0, -0.25], dtype=np.float64),
        A=sparse.csc_matrix(
            np.asarray(
                [
                    [1.0, 2.0],
                    [-1.0, 0.5],
                ],
                dtype=np.float64,
            )
        ),
        row_lower=np.asarray([1.0, -math.inf], dtype=np.float64),
        row_upper=np.asarray([3.0, 2.0], dtype=np.float64),
        col_lower=np.asarray([0.0, -1.0], dtype=np.float64),
        col_upper=np.asarray([1.0, 2.0], dtype=np.float64),
        integrality=np.asarray([1, 0], dtype=np.int8),
        variable_names=["x::V0", "y::P0"],
        x_slice=slice(0, 1),
        y_slice=slice(1, 2),
        reported_objective_scale=2.0,
        objective_offset=0.5,
        metadata={"row_names": ["capacity", "coverage"]},
    )
    model.validate()
    return model


def _row_feasible(model: LinearMipModel, point: np.ndarray) -> bool:
    activity = np.asarray(model.A @ point).ravel()
    return bool(
        np.all(point >= model.col_lower)
        and np.all(point <= model.col_upper)
        and np.all(activity >= model.row_lower)
        and np.all(activity <= model.row_upper)
    )


def test_canonical_csc_and_model_digest_ignore_storage_noise_deterministically():
    model = _tiny_model()
    coo = model.A.tocoo()
    order = np.arange(coo.nnz)[::-1]
    noisy = sparse.coo_matrix(
        (
            np.concatenate([coo.data[order], [0.0, -0.0, 0.5, 0.5]]),
            (
                np.concatenate([coo.row[order], [0, 1, 0, 0]]),
                np.concatenate([coo.col[order], [0, 1, 0, 0]]),
            ),
        ),
        shape=coo.shape,
    ).tocsc()
    # The duplicated halves add one extra coefficient; remove the original
    # corresponding entry so the mathematical matrix remains unchanged.
    noisy = noisy.tolil()
    noisy[0, 0] -= 1.0
    noisy = noisy.tocsc()
    transported = clone_model(model)
    transported.A = noisy

    canonical = canonicalize_csc(transported.A)
    assert canonical.has_sorted_indices
    assert canonical.nnz == model.A.nnz
    assert canonical_model_digest(transported) == canonical_model_digest(model)
    assert canonical_model_digest(model) == canonical_model_digest(model)


@pytest.mark.parametrize(
    "mutation",
    ["coefficient", "objective", "row_bound", "column_bound", "integrality", "sense"],
)
def test_canonical_model_digest_detects_every_mathematical_drift(mutation: str):
    model = _tiny_model()
    changed = clone_model(model)
    if mutation == "coefficient":
        changed.A.data[0] = math.nextafter(float(changed.A.data[0]), math.inf)
    elif mutation == "objective":
        changed.c[0] = math.nextafter(float(changed.c[0]), math.inf)
    elif mutation == "row_bound":
        changed.row_upper[0] = math.nextafter(float(changed.row_upper[0]), math.inf)
    elif mutation == "column_bound":
        changed.col_lower[0] = math.nextafter(float(changed.col_lower[0]), -math.inf)
    elif mutation == "integrality":
        changed.integrality[0] = 0
    elif mutation == "sense":
        changed.sense = "min"
    assert canonical_model_digest(changed) != canonical_model_digest(model)


def test_universe_digest_is_mapping_order_independent_but_order_and_name_sensitive():
    model = _tiny_model()
    first = canonical_universe_digest(
        model,
        candidate_ids=["venue-001"],
        pattern_ids=["grid-001"],
        semantic_ids={"grid": ["g0", "g1"], "admin": ["a0"]},
    )
    second = canonical_universe_digest(
        model,
        candidate_ids=["venue-001"],
        pattern_ids=["grid-001"],
        semantic_ids={"admin": ["a0"], "grid": ["g0", "g1"]},
    )
    reordered = canonical_universe_digest(
        model,
        candidate_ids=["venue-001"],
        pattern_ids=["grid-001"],
        semantic_ids={"grid": ["g1", "g0"], "admin": ["a0"]},
    )
    renamed = clone_model(model)
    renamed.variable_names[0] = "x::OTHER"

    assert first == second
    assert reordered != first
    assert canonical_universe_digest(renamed) != canonical_universe_digest(model)
    # Presentation model names are intentionally outside both mathematical and
    # ordered-universe seals.
    renamed_only = clone_model(model, name="round_trip_name")
    assert canonical_model_digest(renamed_only) == canonical_model_digest(model)
    assert canonical_universe_digest(renamed_only) == canonical_universe_digest(model)


def test_proof_relaxation_moves_every_finite_bound_outward_and_keeps_exact_copy():
    source = _tiny_model()
    source_digest = canonical_model_digest(source)
    result = relax_model_for_proof(
        source, absolute_margin=1e-9, relative_margin=1e-6, ulps=2
    )
    exact = result.exact_model
    relaxed = result.relaxed_model

    for original, moved in (
        (exact.row_lower, relaxed.row_lower),
        (exact.col_lower, relaxed.col_lower),
    ):
        finite = np.isfinite(original)
        assert np.all(moved[finite] < original[finite])
        assert np.array_equal(np.isneginf(moved), np.isneginf(original))
    for original, moved in (
        (exact.row_upper, relaxed.row_upper),
        (exact.col_upper, relaxed.col_upper),
    ):
        finite = np.isfinite(original)
        assert np.all(moved[finite] > original[finite])
        assert np.array_equal(np.isposinf(moved), np.isposinf(original))

    assert canonical_model_digest(source) == source_digest
    assert canonical_model_digest(exact) == source_digest
    assert canonical_model_digest(relaxed) == result.manifest["relaxed_model_sha256"]
    assert result.manifest["exact_model_sha256"] == source_digest
    assert result.manifest["families"]["row_lower"]["direction"] == "downward"
    assert result.manifest["families"]["row_upper"]["direction"] == "upward"
    assert not np.shares_memory(source.row_lower, exact.row_lower)
    assert not np.shares_memory(exact.row_lower, relaxed.row_lower)
    assert source.A.data is not exact.A.data
    assert exact.A.data is not relaxed.A.data

    exact.row_lower[0] -= 1.0
    assert source.row_lower[0] == 1.0
    assert relaxed.row_lower[0] != exact.row_lower[0]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"absolute_margin": -1.0},
        {"relative_margin": -1.0},
        {"absolute_margin": math.inf},
        {"ulps": 0},
        {"ulps": 1.5},
    ],
)
def test_proof_relaxation_rejects_undocumented_or_invalid_margins(kwargs):
    with pytest.raises(ValueError):
        relax_model_for_proof(_tiny_model(), **kwargs)


def test_box_extrema_are_exactly_directional_including_unbounded_boxes():
    assert box_linear_extrema([2.0, -3.0, 0.0], [-1.0, 0.0, -math.inf], [4.0, 5.0, math.inf]) == (
        -17.0,
        8.0,
    )
    minimum, maximum = box_linear_extrema([1.0, -2.0], [-math.inf, 0.0], [1.0, math.inf])
    assert minimum == -math.inf
    assert maximum == 1.0


def test_transport_accepts_only_bounds_that_cover_coefficient_drift_extrema():
    intended = _tiny_model()
    imported = clone_model(intended)
    imported.A[0, 0] = 1.125
    imported.row_upper[0] = 3.125
    imported.integrality[0] = 0  # Integer-to-continuous is a valid relaxation.

    audit = audit_model_transport(intended, imported)
    assert audit.sufficient
    assert audit.row_box_extrema[0] == (0.0, 0.125)
    assert audit.integrality_not_stricter
    assert audit.to_dict()["sufficient"] is True

    inward = clone_model(imported)
    inward.row_upper[0] = math.nextafter(3.125, -math.inf)
    failed = audit_model_transport(intended, inward)
    assert not failed.sufficient
    assert failed.row_upper_failures == (0,)


def test_transport_exhaustively_implies_containment_on_small_integer_box():
    intended = _tiny_model()
    # Exercise all 3^4 coefficient-drift sign combinations.  The extrema-based
    # imported row bounds must contain every intended-feasible integer point.
    for drift_values in itertools.product((-0.25, 0.0, 0.25), repeat=4):
        imported = clone_model(intended)
        drift = np.asarray(drift_values, dtype=np.float64).reshape(2, 2)
        imported.A = sparse.csc_matrix(intended.A.toarray() + drift)
        for row in range(imported.n_row):
            minimum, maximum = box_linear_extrema(
                drift[row], intended.col_lower, intended.col_upper
            )
            if math.isfinite(intended.row_lower[row]):
                imported.row_lower[row] = intended.row_lower[row] + minimum
            if math.isfinite(intended.row_upper[row]):
                imported.row_upper[row] = intended.row_upper[row] + maximum
        audit = audit_model_transport(intended, imported)
        assert audit.sufficient, (drift_values, audit.reasons)
        for point_values in itertools.product((0.0, 1.0), (-1.0, 0.0, 1.0, 2.0)):
            point = np.asarray(point_values)
            if _row_feasible(intended, point):
                assert _row_feasible(imported, point), (drift_values, point_values)


def test_transport_fails_closed_on_box_integrality_and_universe_drift():
    intended = _tiny_model()

    inward_box = clone_model(intended)
    inward_box.col_upper[1] = math.nextafter(2.0, -math.inf)
    audit = audit_model_transport(intended, inward_box)
    assert not audit.sufficient
    assert audit.column_upper_failures == (1,)

    stricter_integer = clone_model(intended)
    stricter_integer.integrality[1] = 1
    audit = audit_model_transport(intended, stricter_integer)
    assert not audit.sufficient
    assert audit.integrality_failures == (1,)

    renamed = clone_model(intended)
    renamed.metadata["row_names"][0] = "different-row"
    audit = audit_model_transport(intended, renamed)
    assert not audit.sufficient
    assert not audit.universe_equal


def test_objective_transport_uses_box_extrema_and_correct_proof_direction():
    intended = _tiny_model()
    dominating = clone_model(intended)
    dominating.c[0] += 0.25  # x0 is non-negative, so max objective only rises.
    max_audit = audit_model_transport(
        intended, dominating, proof_kind="max_upper_bound"
    )
    assert max_audit.sufficient
    assert max_audit.objective_box_extrema == (0.0, 0.25)

    inward = clone_model(intended)
    inward.c[0] -= 0.25
    max_failed = audit_model_transport(
        intended, inward, proof_kind="max_upper_bound"
    )
    assert not max_failed.sufficient
    assert not max_failed.objective_dominance

    min_intended = clone_model(intended)
    min_intended.sense = "min"
    min_dominating = clone_model(min_intended)
    min_dominating.c[0] -= 0.25  # Imported lower objective is safe for min LB.
    min_audit = audit_model_transport(
        min_intended, min_dominating, proof_kind="min_lower_bound"
    )
    assert min_audit.sufficient


def test_transport_requires_equal_reported_scale_for_objective_proofs():
    intended = _tiny_model()
    imported = clone_model(intended)
    imported.reported_objective_scale = 1.0
    audit = audit_model_transport(intended, imported, proof_kind="max_upper_bound")
    assert not audit.sufficient
    assert not audit.objective_dominance
