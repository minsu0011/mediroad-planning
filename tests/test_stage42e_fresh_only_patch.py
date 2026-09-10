from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mediroad.stage4_2_certification.backends import SolveOptions
from mediroad.stage4_2_certification.types import BackendResult
from mediroad.stage4_2e_equity_tail_certification.cli import _json_default
from mediroad.stage4_2e_equity_tail_certification.fresh_oracle import (
    policy_bound_and_strict_threshold,
    run_fresh_threshold_oracle,
)
from mediroad.stage4_2e_equity_tail_certification.runner import _final_plan_frame


def test_cli_json_default_handles_numpy_array_and_scalar() -> None:
    payload = {
        "indices": np.asarray([1, 2, 3], dtype=np.int64),
        "score": np.float64(1.25),
    }
    encoded = json.dumps(payload, default=_json_default)
    decoded = json.loads(encoded)
    assert decoded == {"indices": [1, 2, 3], "score": 1.25}


@pytest.mark.parametrize("existing", [False, True])
def test_final_plan_frame_is_idempotent_for_candidate_set_column(existing: bool) -> None:
    import pandas as pd

    candidates = pd.DataFrame({"venue_id": ["A", "B", "C"]})
    if existing:
        candidates["candidate_set"] = ["legacy", "legacy", "legacy"]
        candidates["scenario"] = ["legacy", "legacy", "legacy"]
    formulation = SimpleNamespace(candidates=candidates)
    plan = _final_plan_frame(  # type: ignore[arg-type]
        formulation, np.asarray([0, 2], dtype=int)
    )
    assert list(plan["venue_id"]) == ["A", "C"]
    assert list(plan["candidate_set"]) == ["top3", "top3"]
    assert list(plan["scenario"]) == ["equity", "equity"]
    assert list(plan.columns).count("candidate_set") == 1
    assert list(plan.columns).count("scenario") == 1


def test_policy_threshold_keeps_exact_half_percent_boundary_and_strict_next_float() -> None:
    bound, threshold = policy_bound_and_strict_threshold("max", 100.0, 0.005)
    assert bound == math.fsum([100.0, 0.5])
    assert threshold == math.nextafter(bound, math.inf)

    bound_min, threshold_min = policy_bound_and_strict_threshold("min", 100.0, 0.005)
    assert bound_min == math.fsum([100.0, -0.5])
    assert threshold_min == math.nextafter(bound_min, -math.inf)


class _TinyFormulation:
    def __init__(self) -> None:
        self.last_threshold = None

    def build(self, *, objective_name, sense, floors, threshold, name):
        self.last_threshold = threshold
        return SimpleNamespace(objective_name=objective_name, name=name)

    def selected_from_solution(self, model, solution):
        return np.asarray(solution, dtype=int)

    def validate_selection(self, selected):
        return True, None

    def complete_solution(self, model, selected):
        return np.asarray(selected, dtype=float)

    def feasibility_violations(self, model, full, tol=1e-6):
        return {"feasible": True}

    def objective_for_selection(self, objective, selected):
        return float(np.asarray(selected, dtype=float).sum())


class _OracleBackend:
    version = "1.15.1"

    def __init__(self, *, infeasible: bool, solution=None) -> None:
        self.infeasible = infeasible
        self.solution = solution
        self.calls = 0

    def solve(self, model, options, *, mip_start=None):
        self.calls += 1
        return BackendResult(
            model_name=model.name,
            status="INFEASIBLE" if self.infeasible else "TIME_LIMIT_WITH_INCUMBENT",
            has_incumbent=self.solution is not None,
            is_infeasible=self.infeasible,
            is_optimal=False,
            objective_value_raw=0.0 if self.solution is not None else None,
            objective_value=0.0 if self.solution is not None else None,
            best_bound_raw=None,
            best_bound=None,
            relative_gap=None,
            mip_node_count=1,
            wall_time_sec=0.01,
            solution=None if self.solution is None else np.asarray(self.solution, dtype=float),
            message="synthetic oracle",
            highs_version=self.version,
            options={"threads": 8, "parallel": True, "random_seed": options.random_seed},
            log_path=options.log_path,
        )


def _options(tmp_path: Path) -> SolveOptions:
    return SolveOptions(
        time_limit_sec=1.0,
        relative_gap=0.005,
        threads=8,
        random_seed=11,
        parallel=True,
        presolve=True,
        heuristic_effort=0.1,
        mip_max_start_nodes=10,
        log_to_console=False,
        log_path=tmp_path / "main.log",
        improving_solution_path=tmp_path / "main.sol",
        extra_options={},
    )


def test_fresh_threshold_oracle_certifies_only_on_fresh_infeasibility(tmp_path: Path) -> None:
    f = _TinyFormulation()
    backend = _OracleBackend(infeasible=True)
    selected = np.asarray([40, 60], dtype=int)  # objective = 100
    result = run_fresh_threshold_oracle(
        formulation=f,  # type: ignore[arg-type]
        backend=backend,  # type: ignore[arg-type]
        objective="need_weighted",
        sense="max",
        floors=[],
        incumbent_selection=selected,
        incumbent_value=100.0,
        base_options=_options(tmp_path),
        output_dir=tmp_path / "oracle",
        relative_gap=0.005,
        max_rounds=2,
        time_limit_sec=1.0,
    )
    assert result.certified is True
    assert result.method == "FRESH_THRESHOLD_INFEASIBILITY"
    assert result.relative_gap == 0.005
    assert result.policy_bound == 100.5
    assert result.strict_threshold == math.nextafter(100.5, math.inf)
    assert backend.calls == 1


def test_fresh_threshold_oracle_remains_inconclusive_without_solution_or_proof(tmp_path: Path) -> None:
    f = _TinyFormulation()
    backend = _OracleBackend(infeasible=False, solution=None)
    result = run_fresh_threshold_oracle(
        formulation=f,  # type: ignore[arg-type]
        backend=backend,  # type: ignore[arg-type]
        objective="cost",
        sense="min",
        floors=[],
        incumbent_selection=np.asarray([40, 60], dtype=int),
        incumbent_value=100.0,
        base_options=_options(tmp_path),
        output_dir=tmp_path / "oracle2",
        relative_gap=0.005,
        max_rounds=2,
        time_limit_sec=1.0,
    )
    assert result.certified is False
    assert result.method == "FRESH_THRESHOLD_INCONCLUSIVE"
    assert math.isinf(result.relative_gap)
