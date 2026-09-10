from __future__ import annotations

import numpy as np

from mediroad.stage4_2_certification.synthetic import make_synthetic_formulation
from mediroad.stage4_2d_equity_certification.model_utils import (
    append_cuts,
    apply_fixings,
    local_branching_cut,
)
from mediroad.stage4_2d_equity_certification.types import CutRecord, FixingRecord


def test_append_cut_and_fixing_preserve_sparse_model_contract():
    f = make_synthetic_formulation()
    model = f.build(objective_name="high_need_population", sense="max")
    fixing = FixingRecord(4, "V4", 0, "test", 1.0, 0.0, "UNIT_TEST")
    cut = CutRecord(
        name="test_cut",
        indices=np.asarray([0, 1]),
        values=np.asarray([1.0, 1.0]),
        lower=1.0,
        upper=np.inf,
        kind="TEST",
    )
    modified = append_cuts(apply_fixings(model, [fixing]), [cut])
    assert modified.n_row == model.n_row + 1
    assert modified.col_upper[4] == 0.0
    modified.validate()


def test_local_branching_inside_and_exclusion_are_complements_on_ball_boundary():
    seed = np.asarray([0, 1])
    inside = local_branching_cut(seed, visit_count=2, radius=1, mode="inside")
    outside = local_branching_cut(seed, visit_count=2, radius=1, mode="exclude")
    assert inside.lower == 1.0
    assert outside.upper == 0.0
