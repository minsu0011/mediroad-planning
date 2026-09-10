from __future__ import annotations

import itertools
import math
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from mediroad.stage4_2_certification.backends import ScipyBackend, SolveOptions
from mediroad.stage4_2_certification.formulation import CertificationFormulation
from mediroad.stage4_2_certification.synthetic import make_synthetic_formulation
from mediroad.stage4_2_certification.types import Floor


def _all_selections(formulation):
    for combo in itertools.combinations(range(formulation.nx), formulation.visit_count):
        selected = np.asarray(combo, dtype=int)
        valid, _ = formulation.validate_selection(selected)
        if valid:
            yield selected


def _best(formulation, objective, sense, floors=()):
    feasible = []
    for selected in _all_selections(formulation):
        metrics = {
            "total_population": formulation.objective_for_selection("total_population", selected),
            "need_weighted": formulation.objective_for_selection("need_weighted", selected),
            "high_need_population": formulation.objective_for_selection("high_need_population", selected),
            "min_sigungu_coverage": formulation.objective_for_selection("min_sigungu_coverage", selected),
            "cost": formulation.objective_for_selection("cost", selected),
        }
        if all(
            metrics[f.objective] >= f.value - 1e-7 if f.sense == "max" else metrics[f.objective] <= f.value + 1e-7
            for f in floors
        ):
            feasible.append((metrics[objective], selected))
    return (max(feasible, key=lambda row: row[0]) if sense == "max" else min(feasible, key=lambda row: row[0]))


def _solve(formulation, objective, sense, floors=()):
    model = formulation.build(objective_name=objective, sense=sense, floors=list(floors))
    result = ScipyBackend().solve(
        model,
        SolveOptions(time_limit_sec=30, relative_gap=0.0),
    )
    assert result.has_incumbent
    selected = formulation.selected_from_solution(model, result.solution)
    return model, result, selected


@pytest.fixture
def synthetic_formulation():
    return make_synthetic_formulation()


def test_one_link_total_population_matches_bruteforce(synthetic_formulation):
    f = synthetic_formulation
    expected, _ = _best(f, "total_population", "max")
    _, _, selected = _solve(f, "total_population", "max")
    assert math.isclose(f.objective_for_selection("total_population", selected), expected, abs_tol=1e-7)


@pytest.mark.parametrize(
    "selected",
    [np.asarray([-1, 3]), np.asarray([0, 999]), np.asarray([0.9, 3.0]), np.asarray([[0], [3]])],
)
def test_validate_selection_rejects_unbounded_or_noninteger_indices(synthetic_formulation, selected):
    valid, reason = synthetic_formulation.validate_selection(selected)
    assert not valid
    assert reason


def test_one_link_cost_with_coverage_floors_matches_bruteforce(synthetic_formulation):
    f = synthetic_formulation
    floors = [Floor("total_population", "max", 60.0), Floor("min_sigungu_coverage", "max", 0.30)]
    expected, _ = _best(f, "cost", "min", floors)
    model, _, selected = _solve(f, "cost", "min", floors)
    assert math.isclose(f.cost_for_selection(selected), expected, abs_tol=1e-6)
    assert "count_lambda" in model.extra_slices


def test_min_sigungu_continuous_auxiliary_matches_exact_ratio(synthetic_formulation):
    f = synthetic_formulation
    model, result, selected = _solve(f, "min_sigungu_coverage", "max")
    exact = f.objective_for_selection("min_sigungu_coverage", selected)
    assert math.isclose(result.objective_value, exact, abs_tol=1e-7)
    assert "q" in model.extra_slices
    q = int(model.extra_slices["q"])
    assert model.integrality[q] == 0


def test_removed_rows_equal_pattern_candidate_incidences(synthetic_formulation):
    f = synthetic_formulation
    model = f.build(objective_name="total_population", sense="max", floors=[])
    incidence = sum(len(x) for x in f.pattern_coverers)
    assert model.metadata["removed_redundant_or_rows"] == incidence
    link_rows = [name for name in model.metadata["row_names"] if name.startswith("cover_link::")]
    assert len(link_rows) == sum(bool(len(x)) for x in f.pattern_coverers)


def test_stage_specific_variables(synthetic_formulation):
    f = synthetic_formulation
    easy = f.build(objective_name="total_population", sense="max", floors=[])
    mincov = f.build(objective_name="min_sigungu_coverage", sense="max", floors=[])
    cost = f.build(objective_name="cost", sense="min", floors=[])
    assert "q" not in easy.extra_slices and "count_lambda" not in easy.extra_slices
    assert "q" in mincov.extra_slices and "count_lambda" not in mincov.extra_slices
    assert "q" not in cost.extra_slices and "count_lambda" in cost.extra_slices
    assert easy.n_col < mincov.n_col < cost.n_col
    count_lambda = cost.extra_slices["count_lambda"]
    assert isinstance(count_lambda, slice)
    assert np.all(cost.integrality[count_lambda] == 0)


def test_complete_solution_is_feasible(synthetic_formulation):
    f = synthetic_formulation
    selected = np.array([1, 3])
    floors = [Floor("total_population", "max", 50.0)]
    for objective, sense in [
        ("total_population", "max"),
        ("min_sigungu_coverage", "max"),
        ("cost", "min"),
    ]:
        model = f.build(objective_name=objective, sense=sense, floors=floors)
        values = f.complete_solution(model, selected)
        violations = f.feasibility_violations(model, values)
        assert bool(violations["feasible"]), violations


def test_one_link_rejects_negative_benefit_weights(synthetic_formulation):
    f = synthetic_formulation
    patterns = SimpleNamespace(
        n_candidates=f.nx,
        n_patterns=f.ny,
        pattern_coverers=f.pattern_coverers,
        population=f.population.copy(),
        need_weighted=f.need_weighted.copy(),
        high_need_population=f.high_need.copy(),
        sigungu=f.pattern_sigungu.copy(),
    )
    patterns.need_weighted[0] = -0.1
    with pytest.raises(ValueError, match="requires non-negative"):
        CertificationFormulation(
            f.candidates.copy(),
            patterns,
            deepcopy(f.base_config),
            deepcopy(f.cert_config),
            candidate_set="negative",
        )
