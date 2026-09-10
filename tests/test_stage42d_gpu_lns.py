from __future__ import annotations

import numpy as np

from mediroad.stage4_2_certification.synthetic import make_synthetic_formulation
from mediroad.stage4_2d_equity_certification.gpu_lns import HybridLargeNeighborhoodSearch
from mediroad.stage4_2d_equity_certification.types import EvidenceSeed


def test_cpu_lns_keeps_exact_feasible_elites():
    f = make_synthetic_formulation()
    raw = f.metrics(np.asarray([0, 3]))
    seed = EvidenceSeed(
        "seed",
        np.asarray([0, 3]),
        {
            "total_population": raw["unique_elderly_population"],
            "min_sigungu_coverage": raw["min_sigungu_coverage_ratio"],
            "high_need_population": raw["high_need_population"],
            "need_weighted": raw["need_weighted_population"],
            "cost": raw["cost"],
        },
    )
    config = {
        "hardware": {"gpu_lns": False},
        "search": {
            "rounds": 3,
            "gpu_batch_size": 64,
            "elite_pool_size": 4,
            "radius_schedule": [1],
            "ruin_sets_per_round": 2,
            "proposals_per_ruin": 4,
            "candidate_pool_size": 5,
            "random_seed": 1,
        },
    }
    result = HybridLargeNeighborhoodSearch(f, config).search(
        [seed],
        objective="high_need_population",
        total_population_floor=10.0,
        min_sigungu_floor=0.1,
    )
    assert result.elites
    for elite in result.elites:
        valid, _ = f.validate_selection(elite.selected_indices)
        assert valid
        assert elite.metrics["total_population"] >= 10.0
        assert elite.metrics["min_sigungu_coverage"] >= 0.1


def test_radius_two_recreation_refilters_selected_negative_infinity_priority():
    f = make_synthetic_formulation()
    search = HybridLargeNeighborhoodSearch(
        f,
        {
            "hardware": {"gpu_lns": False},
            "search": {"random_addition_probability": 0.0, "random_seed": 7},
        },
    )

    class TrackingRng:
        def __init__(self):
            self.inner = np.random.default_rng(7)
            self.integer_calls = 0

        def random(self):
            return 1.0

        def choice(self, *args, **kwargs):
            return self.inner.choice(*args, **kwargs)

        def integers(self, *args, **kwargs):
            self.integer_calls += 1
            return self.inner.integers(*args, **kwargs)

    rng = TrackingRng()
    priority = np.linspace(1.0, 2.0, f.nx)
    plan = search._recreate(np.asarray([], dtype=int), 2, priority, rng, f.nx)
    assert plan is not None
    assert len(set(map(int, plan))) == 2
    assert rng.integer_calls == 0
