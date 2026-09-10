from __future__ import annotations

import pytest

from types import SimpleNamespace

import numpy as np
import pandas as pd

from mediroad.stage4_2_certification.synthetic import make_synthetic_formulation
from mediroad.stage4_2_certification.adapter import _compose_venue_alias_map
from mediroad.stage4_2_certification.backends import HighsBackend, ScipyBackend, SolveOptions
from mediroad.stage4_2_certification.formulation import CertificationFormulation

@pytest.fixture
def synthetic_formulation():
    return make_synthetic_formulation()


def test_direct_highs_backend_accepts_full_mip_start(synthetic_formulation):
    formulation = synthetic_formulation
    model = formulation.build(objective_name="total_population", sense="max", floors=[])
    start = formulation.complete_solution(model, np.array([0, 3], dtype=int))
    backend = HighsBackend(require_minimum_version=True)
    result = backend.solve(
        model,
        SolveOptions(
            time_limit_sec=15,
            relative_gap=0.0,
            threads=1,
            parallel=False,
            log_to_console=False,
        ),
        mip_start=start,
    )
    assert result.has_incumbent
    assert result.is_optimal
    assert result.solution is not None
    assert result.options["start_mode"] in {"FULL_HIGHS_SOLUTION", "SPARSE_COLUMN_START"}
    assert result.options["binding_source"] in {
        "public_highspy",
        "scipy_bundled_private_highspy",
    }


def test_direct_highs_exact_optimum_requires_closed_bound(synthetic_formulation):
    formulation = synthetic_formulation
    model = formulation.build(objective_name="total_population", sense="max", floors=[])
    backend = HighsBackend(require_minimum_version=True)
    result = backend.solve(
        model,
        SolveOptions(
            time_limit_sec=15,
            relative_gap=0.005,
            threads=1,
            parallel=False,
            log_to_console=False,
        ),
    )
    assert result.relative_gap is not None
    assert result.relative_gap <= 1e-9
    assert result.is_optimal


def test_integer_count_convex_hull_closes_fractional_equal_allocation_hole():
    candidates = pd.DataFrame(
        {
            "venue_id": ["A", "B", "C"],
            "admin_code": ["A", "B", "C"],
            "sigungu": ["S1", "S2", "S3"],
            "cluster_id": ["A", "B", "C"],
            "known_overlap": ["NO", "NO", "NO"],
            "travel_minutes": [0.0, 0.0, 0.0],
            "candidate_priority_tiebreak": [0.0, 0.0, 0.0],
        }
    )
    patterns = SimpleNamespace(
        n_candidates=3,
        n_patterns=3,
        pattern_coverers=[
            np.array([0], dtype=np.int32),
            np.array([1], dtype=np.int32),
            np.array([2], dtype=np.int32),
        ],
        population=np.ones(3),
        need_weighted=np.ones(3),
        high_need_population=np.zeros(3),
        sigungu=np.array(["S1", "S2", "S3"]),
    )
    base = {
        "optimization": {
            "visit_count": 2,
            "known_overlap_penalty": 0.0,
            "travel_penalty_scale": 0.0,
            "visit_location_deviation_penalty": 100.0,
            "deterministic_tiebreak_scale": 0.0,
        },
        "candidate_expansion": {
            "max_visits_per_admin": 2,
            "max_visits_per_sigungu": 2,
            "one_per_coverage_cluster": True,
        },
    }
    formulation = CertificationFormulation(
        candidates,
        patterns,
        base,
        {"formulation": {"min_sigungu_mode": "continuous_auxiliary_with_threshold_oracle"}},
        candidate_set="convex_hull",
    )
    model = formulation.build(objective_name="cost", sense="min", floors=[])
    # Inspect the root relaxation: the old continuous |count-2/3| formulation
    # admitted zero deviation at fractional equal allocation.  The integer-count
    # convex hull has a valid lower bound of 4/3 * 100 even with x relaxed.
    model.integrality[:] = 0
    result = ScipyBackend().solve(
        model,
        SolveOptions(time_limit_sec=15, relative_gap=0.0),
    )
    assert result.is_optimal
    assert result.objective_value is not None
    assert abs(result.objective_value - (400.0 / 3.0)) <= 1e-7


def test_sequential_candidate_aliases_resolve_to_final_representative():
    ledger = pd.DataFrame(
        {
            "venue_id": ["OLD", "MID", "KEEP"],
            "representative_venue_id": ["MID", "KEEP", "KEEP"],
        }
    )
    aliases = _compose_venue_alias_map(ledger, {"KEEP"})
    assert aliases["OLD"] == "KEEP"
    assert aliases["MID"] == "KEEP"
    assert aliases["KEEP"] == "KEEP"
