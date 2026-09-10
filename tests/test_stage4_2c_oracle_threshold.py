from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from mediroad.stage4_2_certification.engine import CertificationEngine
from mediroad.stage4_2_certification.synthetic import make_synthetic_formulation


def _engine_stub():
    engine = object.__new__(CertificationEngine)
    engine.gap = 0.005
    engine.f = SimpleNamespace()
    return engine


def test_max_threshold_is_conservative():
    engine = _engine_stub()
    floor = engine._oracle_threshold("total_population", "max", 100.0)
    assert floor.sense == "max"
    assert floor.value > 100.5


def test_min_cost_threshold_is_strictly_below():
    engine = _engine_stub()
    floor = engine._oracle_threshold("cost", "min", 1000.0)
    assert floor.sense == "min"
    assert floor.value < 995.0


def test_min_sigungu_threshold_is_strictly_above_half_percent():
    engine = _engine_stub()
    floor = engine._oracle_threshold("min_sigungu_coverage", "max", 0.5)
    assert floor.value > 0.5025


def test_guided_oracle_uses_original_objective_target_and_specific_heuristic(tmp_path):
    formulation = make_synthetic_formulation()

    class InfeasibleBackend:
        def __init__(self):
            self.model = None
            self.options = None

        def solve(self, model, options, *, mip_start=None):
            self.model = model
            self.options = options
            return SimpleNamespace(
                status="INFEASIBLE",
                has_incumbent=False,
                is_infeasible=True,
                wall_time_sec=0.01,
                mip_node_count=1,
                message="synthetic infeasibility proof",
                options={"peak_rss_mb": 1.0},
                solution=None,
            )

    backend = InfeasibleBackend()
    engine = object.__new__(CertificationEngine)
    engine.gap = 0.005
    engine.f = formulation
    engine.backend = backend
    engine.highs_threads = 1
    engine.cfg = {
        "solver": {
            "random_seed": 42,
            "oracle_heuristic_effort": 0.02,
            "oracle_heuristic_effort_by_objective": {"high_need_population": 0.5},
        },
        "oracle": {
            "enabled": True,
            "max_rounds": 1,
            "guided_objective_enabled": True,
            "stop_at_objective_target": True,
            "time_limit_sec": {"high_need_population": 10},
        },
    }
    selected = np.array([0, 3], dtype=int)
    incumbent = formulation.objective_for_selection("high_need_population", selected)
    certified, certificate, *_ = engine._run_oracle(
        scenario="equity",
        candidate_set="synthetic",
        stage_index=2,
        objective_name="high_need_population",
        sense="max",
        floors=[],
        selected=selected,
        incumbent=incumbent,
        stage_dir=tmp_path,
    )
    assert certified
    assert certificate == "INFEASIBILITY_ORACLE"
    assert backend.model.objective_name == "high_need_population"
    assert backend.model.sense == "max"
    assert backend.options.heuristic_effort == 0.5
    assert backend.options.extra_options["objective_target"] > incumbent
