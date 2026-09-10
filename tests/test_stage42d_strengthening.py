from __future__ import annotations

import itertools

import numpy as np

from mediroad.stage4_2_certification.synthetic import make_synthetic_formulation
from mediroad.stage4_2d_equity_certification.strengthening import (
    compute_safe_zero_fixings,
    generate_anchored_submodular_cuts,
    metric_targets,
)
from mediroad.stage4_2d_equity_certification.types import EvidenceSeed
from mediroad.stage4_2d_equity_certification.types import MetricTarget


def all_valid_plans(formulation):
    for raw in itertools.combinations(range(formulation.nx), formulation.visit_count):
        selected = np.asarray(raw, dtype=int)
        valid, _ = formulation.validate_selection(selected)
        if valid:
            yield selected


def satisfies_targets(formulation, plan, targets):
    covered = formulation.covered_mask(plan)
    for target in targets:
        if float(target.weights[covered].sum()) < target.target - 1e-8:
            return False
    return True


def test_submodular_cuts_are_valid_for_every_feasible_plan():
    f = make_synthetic_formulation()
    seed = EvidenceSeed("seed", np.asarray([0, 3]), {})
    targets = metric_targets(
        f,
        total_population_floor=35.0,
        min_sigungu_floor=0.20,
        high_need_target=20.0,
    )
    cuts, _ = generate_anchored_submodular_cuts(
        f, targets, [seed], anchor_sizes=[0, 1, 2], max_cuts=200
    )
    for plan in all_valid_plans(f):
        if not satisfies_targets(f, plan, targets):
            continue
        x = np.zeros(f.nx)
        x[plan] = 1.0
        for cut in cuts:
            activity = float(np.dot(cut.values, x[cut.indices]))
            assert activity >= cut.lower - 1e-8


def test_safe_zero_fixings_never_remove_a_target_feasible_plan():
    f = make_synthetic_formulation()
    targets = metric_targets(
        f,
        total_population_floor=70.0,
        min_sigungu_floor=None,
        high_need_target=None,
    )
    fixings, _ = compute_safe_zero_fixings(f, targets, use_gpu=False)
    fixed = {value.candidate_index for value in fixings}
    for plan in all_valid_plans(f):
        if satisfies_targets(f, plan, targets):
            assert not fixed.intersection(set(map(int, plan)))


def test_outward_safe_cuts_never_exclude_a_target_feasible_plan():
    f = make_synthetic_formulation()
    # Decimal values exercise non-exact binary summation and the direction of
    # both coefficient and RHS rounding.
    weights = np.asarray([0.1 + index * 0.01 for index in range(f.ny)], dtype=float)
    feasible_values = []
    for plan in all_valid_plans(f):
        feasible_values.append(float(weights[f.covered_mask(plan)].sum()))
    target_value = float(np.quantile(feasible_values, 0.25))
    targets = [MetricTarget("decimal_metric", target_value, weights, "synthetic")]
    seed = EvidenceSeed("seed", np.asarray([0, 3]), {})
    cuts, info = generate_anchored_submodular_cuts(
        f,
        targets,
        [seed],
        anchor_sizes=[0, 1, 2],
        max_cuts=200,
        outward_safe=True,
    )
    assert info["outward_safe"] is True
    assert info["coefficient_rounding_direction"] == "POSITIVE_INFINITY"
    assert info["rhs_rounding_direction"] == "NEGATIVE_INFINITY"
    for plan in all_valid_plans(f):
        if float(weights[f.covered_mask(plan)].sum()) < target_value:
            continue
        x = np.zeros(f.nx)
        x[plan] = 1.0
        for cut in cuts:
            assert float(np.dot(cut.values, x[cut.indices])) >= cut.lower


def test_outward_safe_emission_preserves_frozen_cut_identity_and_metric_coverage():
    f = make_synthetic_formulation()
    weights = np.asarray([0.1 + index * 0.01 for index in range(f.ny)], dtype=float)
    target = MetricTarget("decimal_metric", 0.23, weights, "synthetic")
    seed = EvidenceSeed("seed", np.asarray([0, 3]), {})
    common = {
        "formulation": f,
        "targets": [target],
        "seeds": [seed],
        "anchor_sizes": [0, 1, 2],
        "max_cuts": 200,
    }
    legacy, _ = generate_anchored_submodular_cuts(**common, outward_safe=False)
    outward, _ = generate_anchored_submodular_cuts(**common, outward_safe=True)
    assert [cut.name for cut in outward] == [cut.name for cut in legacy]
    assert [cut.kind for cut in outward] == [cut.kind for cut in legacy]
    assert [cut.metric for cut in outward] == [cut.metric for cut in legacy]
    assert all(
        cut.metadata.get("selection_identity_rule")
        == "FROZEN_LEGACY_TOLERANCE__OUTWARD_EMISSION"
        for cut in outward
        if cut.kind == "ANCHORED_SUBMODULAR_SUPERLEVEL"
    )


def test_outward_safe_cuts_reject_negative_or_nonfinite_weights():
    f = make_synthetic_formulation()
    seed = EvidenceSeed("seed", np.asarray([0, 3]), {})
    for bad in (-1.0, np.nan):
        weights = np.ones(f.ny, dtype=float)
        weights[0] = bad
        target = MetricTarget("bad", 1.0, weights, "synthetic")
        with np.testing.assert_raises(ValueError):
            generate_anchored_submodular_cuts(
                f,
                [target],
                [seed],
                anchor_sizes=[0],
                max_cuts=10,
                outward_safe=True,
            )
