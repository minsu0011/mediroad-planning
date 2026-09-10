from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
from scipy import sparse

from mediroad.stage4_2_certification.types import LinearMipModel
from mediroad.stage4_2e_equity_tail_certification.audit import (
    audit_gap,
    choose_incumbent,
    combine_valid_bounds,
    exact_feasible_set_subset,
    inherited_bound_allowed,
)


def _model() -> LinearMipModel:
    return LinearMipModel(
        name="parent",
        objective_name="need_weighted",
        sense="max",
        c=np.asarray([2.0, 1.0, 0.5], dtype=np.float64),
        A=sparse.csc_matrix(
            np.asarray(
                [
                    [1.0, 1.0, 0.0],
                    [0.0, 2.0, 1.0],
                ],
                dtype=np.float64,
            )
        ),
        row_lower=np.asarray([0.0, -math.inf], dtype=np.float64),
        row_upper=np.asarray([2.0, 3.0], dtype=np.float64),
        col_lower=np.zeros(3, dtype=np.float64),
        col_upper=np.ones(3, dtype=np.float64),
        integrality=np.asarray([1, 1, 0], dtype=np.int8),
        variable_names=["x::0", "x::1", "y::0"],
        x_slice=slice(0, 2),
        y_slice=slice(2, 3),
        reported_objective_scale=1.0,
        objective_offset=0.0,
        metadata={"row_names": ["r0", "r1"]},
    )


def test_exact_subset_accepts_only_exact_row_box_tightening() -> None:
    parent = _model()
    child = replace(
        parent,
        name="child",
        row_lower=np.asarray([0.25, -1.0], dtype=np.float64),
        row_upper=np.asarray([1.5, 2.5], dtype=np.float64),
    )

    audit = exact_feasible_set_subset(parent, child)

    assert audit.sufficient is True
    assert audit.failures == []
    assert audit.shape_equal
    assert audit.objective_equal
    assert audit.universe_equal
    assert audit.matrix_equal
    assert audit.variable_domain_equal
    assert audit.row_box_subset
    assert audit.tightened_row_count == 2
    assert [row["row_name"] for row in audit.tightened_rows] == ["r0", "r1"]
    assert all(row["contained"] for row in audit.tightened_rows)
    assert audit.parent_model_sha256 != audit.child_model_sha256


def test_exact_subset_rejects_even_one_ulp_of_row_relaxation() -> None:
    parent = _model()
    relaxed_lower = parent.row_lower.copy()
    relaxed_lower[0] = math.nextafter(float(relaxed_lower[0]), -math.inf)
    child = replace(parent, name="relaxed", row_lower=relaxed_lower)

    audit = exact_feasible_set_subset(parent, child)

    assert audit.sufficient is False
    assert audit.row_box_subset is False
    assert audit.failures == ["row_box_subset"]
    assert audit.tightened_rows[0]["contained"] is False


def test_exact_subset_rejects_matrix_drift() -> None:
    parent = _model()
    changed = parent.A.copy()
    changed.data = changed.data.copy()
    changed.data[0] = math.nextafter(float(changed.data[0]), math.inf)

    audit = exact_feasible_set_subset(parent, replace(parent, name="matrix", A=changed))

    assert audit.sufficient is False
    assert audit.matrix_equal is False
    assert "matrix" in audit.failures


@pytest.mark.parametrize(
    "changes",
    [
        {"objective_name": "cost"},
        {"sense": "min"},
        {"c": np.asarray([2.0, 1.0, 0.5000000000000001])},
        {"reported_objective_scale": 2.0},
        {"objective_offset": 1.0},
    ],
    ids=["name", "sense", "coefficients", "scale", "offset"],
)
def test_exact_subset_rejects_objective_semantic_drift(changes: dict[str, object]) -> None:
    parent = _model()

    audit = exact_feasible_set_subset(parent, replace(parent, name="objective", **changes))

    assert audit.sufficient is False
    assert audit.objective_equal is False
    assert "objective" in audit.failures


@pytest.mark.parametrize(
    "changes",
    [
        {"col_lower": np.asarray([0.0, 0.1, 0.0])},
        {"col_upper": np.asarray([1.0, 0.9, 1.0])},
        {"integrality": np.asarray([1, 0, 0], dtype=np.int8)},
    ],
    ids=["lower-bound", "upper-bound", "integrality"],
)
def test_exact_subset_rejects_variable_domain_drift(changes: dict[str, object]) -> None:
    parent = _model()

    audit = exact_feasible_set_subset(parent, replace(parent, name="domain", **changes))

    assert audit.sufficient is False
    assert audit.variable_domain_equal is False
    assert "variable_domain" in audit.failures


@pytest.mark.parametrize(
    "changes",
    [
        {"variable_names": ["x::renamed", "x::1", "y::0"]},
        {"x_slice": slice(0, 1)},
        {"y_slice": slice(1, 3)},
        {"extra_slices": {"unexpected": 2}},
        {"metadata": {"row_names": ["r0", "renamed"]}},
    ],
    ids=["names", "x-slice", "y-slice", "extra-layout", "row-names"],
)
def test_exact_subset_rejects_variable_or_row_universe_drift(
    changes: dict[str, object],
) -> None:
    parent = _model()

    audit = exact_feasible_set_subset(parent, replace(parent, name="universe", **changes))

    assert audit.sufficient is False
    assert audit.universe_equal is False
    assert "universe" in audit.failures


def test_bound_combination_uses_the_tightest_valid_direction() -> None:
    assert combine_valid_bounds("max", fresh=105.0, inherited=104.0) == 104.0
    assert combine_valid_bounds("min", fresh=95.0, inherited=96.0) == 96.0
    assert combine_valid_bounds("max", fresh=105.0, inherited=None) == 105.0
    assert combine_valid_bounds("min", fresh=None, inherited=96.0) == 96.0
    assert combine_valid_bounds("max", fresh=math.inf, inherited=None) is None
    assert combine_valid_bounds("min", fresh=math.nan, inherited=None) is None


def test_inherited_bound_requires_both_exact_subset_and_verified_sha() -> None:
    parent = _model()
    subset = exact_feasible_set_subset(
        parent,
        replace(parent, name="child", row_lower=np.asarray([0.25, -math.inf])),
    )
    failed_subset = exact_feasible_set_subset(
        parent,
        replace(parent, name="relaxed", row_lower=np.asarray([-0.25, -math.inf])),
    )

    assert inherited_bound_allowed(subset=subset, artifact_sha_verified=True)
    assert not inherited_bound_allowed(subset=subset, artifact_sha_verified=False)
    assert not inherited_bound_allowed(subset=failed_subset, artifact_sha_verified=True)


def test_choose_incumbent_obeys_objective_sense_and_preserves_source() -> None:
    low = np.asarray([0, 1], dtype=int)
    high = np.asarray([2, 3], dtype=int)

    max_source, max_value, max_selection = choose_incumbent(
        "max", [("old", 10.0, low), ("fresh", 12.0, high)]
    )
    min_source, min_value, min_selection = choose_incumbent(
        "min", [("old", 10.0, low), ("fresh", 8.0, high)]
    )

    assert (max_source, max_value) == ("fresh", 12.0)
    assert (min_source, min_value) == ("fresh", 8.0)
    np.testing.assert_array_equal(max_selection, high)
    np.testing.assert_array_equal(min_selection, high)
    with pytest.raises(RuntimeError, match="No feasible incumbent"):
        choose_incumbent("max", [])


@pytest.mark.parametrize(
    ("sense", "incumbent", "bound", "label", "direction", "within"),
    [
        ("max", 100.0, 100.0, "CERTIFIED_EXACT", True, True),
        ("max", 100.0, 100.4, "CERTIFIED_NEAR_OPTIMAL", True, True),
        ("max", 100.0, 101.0, "UNCERTIFIED", True, False),
        ("max", 100.0, 99.0, "UNCERTIFIED", False, False),
        ("min", 100.0, 100.0, "CERTIFIED_EXACT", True, True),
        ("min", 100.0, 99.6, "CERTIFIED_NEAR_OPTIMAL", True, True),
        ("min", 100.0, 99.0, "UNCERTIFIED", True, False),
        ("min", 100.0, 101.0, "UNCERTIFIED", False, False),
    ],
)
def test_gap_audit_direction_and_semantic_labels(
    sense: str,
    incumbent: float,
    bound: float,
    label: str,
    direction: bool,
    within: bool,
) -> None:
    result = audit_gap(sense, incumbent, bound, 0.005)  # type: ignore[arg-type]

    assert result.semantic_label == label
    assert result.bound_direction_valid is direction
    assert result.within_limit is within
    assert result.exact is (label == "CERTIFIED_EXACT")
    if not direction:
        assert math.isinf(result.relative_gap)


def test_gap_audit_rejects_invalid_inputs_and_unknown_sense() -> None:
    with pytest.raises(ValueError, match="finite"):
        audit_gap("max", math.nan, 1.0, 0.005)
    with pytest.raises(ValueError, match="non-negative"):
        audit_gap("max", 1.0, 1.0, -0.1)
    with pytest.raises(ValueError, match="Unsupported sense"):
        audit_gap("sideways", 1.0, 1.0, 0.005)  # type: ignore[arg-type]
