from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from mediroad.stage4_2_certification.adapter import (
    load_base_stage42_config,
    prepare_candidate_problem,
)
from mediroad.stage4_2_certification.config import validate_certification_config
from mediroad.stage4_2_certification.formulation import CertificationFormulation
from mediroad.stage4_2_certification.synthetic import make_synthetic_formulation
from mediroad.stage4_2_certification.types import BackendResult, Floor, LinearMipModel
from mediroad.stage4_2e_equity_tail_certification.audit import exact_feasible_set_subset
from mediroad.stage4_2e_equity_tail_certification.config import (
    FROZEN_C_RUN_ID,
    FROZEN_D_RUN_ID,
    FROZEN_HIGH_NEED_FLOOR,
    FROZEN_HIGH_NEED_INCUMBENT,
    FROZEN_MIN_SIGUNGU_FLOOR,
    FROZEN_OLD_COST_INCUMBENT,
    FROZEN_OLD_NEED_FLOOR,
    FROZEN_OLD_NEED_INCUMBENT,
    FROZEN_TOTAL_POPULATION_FLOOR,
    load_yaml,
    validate_config,
)
from mediroad.stage4_2e_equity_tail_certification.evidence import (
    TailEvidence,
    load_tail_evidence,
)
import mediroad.stage4_2e_equity_tail_certification.runner as tail_runner


ROOT = Path(__file__).resolve().parents[1]
OLD_HIGH_NEED_FLOOR = 11447.794835668963


@dataclass
class _ActualContext:
    formulation: CertificationFormulation
    cfg: dict[str, Any]
    evidence: TailEvidence


@pytest.fixture(scope="module")
def actual_context() -> _ActualContext:
    base_cfg = load_base_stage42_config(ROOT / "configs/model_v1/stage4_2.yaml")
    cert_cfg = load_yaml(ROOT / "configs/model_v1/stage4_2_certification.yaml")
    tail_cfg = load_yaml(
        ROOT / "configs/model_v1/stage4_2e_equity_tail_certification.yaml"
    )
    validate_certification_config(cert_cfg)
    validate_config(tail_cfg)
    problem = prepare_candidate_problem(ROOT, "top3", base_cfg, cache={})
    formulation = CertificationFormulation(
        problem.candidates,
        problem.patterns,
        base_cfg,
        cert_cfg,
        candidate_set="top3",
    )
    formulation.venue_alias_map = dict(problem.venue_alias_map)
    evidence = load_tail_evidence(ROOT, formulation, tail_cfg)
    return _ActualContext(formulation=formulation, cfg=tail_cfg, evidence=evidence)


def _actual_base_floors(high_floor: float) -> list[Floor]:
    return [
        Floor("total_population", "max", FROZEN_TOTAL_POPULATION_FLOOR),
        Floor("min_sigungu_coverage", "max", FROZEN_MIN_SIGUNGU_FLOOR),
        Floor("high_need_population", "max", high_floor),
    ]


def _assert_selection_feasible(
    formulation: CertificationFormulation,
    model: LinearMipModel,
    selection: np.ndarray,
) -> None:
    valid, reason = formulation.validate_selection(selection)
    assert valid, reason
    completed = formulation.complete_solution(model, selection)
    violations = formulation.feasibility_violations(model, completed, tol=1e-6)
    assert violations["feasible"] == 1.0, violations


def test_real_c_d_evidence_and_reconstructed_top3_subset_contract(
    actual_context: _ActualContext,
) -> None:
    formulation = actual_context.formulation
    evidence = actual_context.evidence

    assert (formulation.nx, formulation.ny, formulation.visit_count) == (452, 5242, 20)
    assert evidence.stage42d_run_id == FROZEN_D_RUN_ID
    assert evidence.stage42c_run_id == FROZEN_C_RUN_ID
    assert evidence.artifact_sha_verified is True
    assert len(evidence.records) == 11
    assert len({row["relative_path"] for row in evidence.records}) == len(evidence.records)
    assert all(len(str(row["sha256"])) == 64 for row in evidence.records)

    d_metrics = formulation.metrics(evidence.d_front_selection)
    assert float(d_metrics["high_need_population"]).hex() == FROZEN_HIGH_NEED_INCUMBENT.hex()
    assert float(d_metrics["min_sigungu_coverage_ratio"]) >= FROZEN_MIN_SIGUNGU_FLOOR

    old_floors = _actual_base_floors(OLD_HIGH_NEED_FLOOR)
    new_floors = _actual_base_floors(FROZEN_HIGH_NEED_FLOOR)
    old_need = formulation.build(
        objective_name="need_weighted", sense="max", floors=old_floors, name="old-need"
    )
    new_need = formulation.build(
        objective_name="need_weighted", sense="max", floors=new_floors, name="new-need"
    )
    need_subset = exact_feasible_set_subset(old_need, new_need)

    assert need_subset.sufficient is True
    assert need_subset.failures == []
    assert need_subset.tightened_row_count == 1
    assert need_subset.tightened_rows[0]["row_name"].endswith("high_need_population")
    assert new_need.metadata["floors"][2]["value"].hex() == FROZEN_HIGH_NEED_FLOOR.hex()
    _assert_selection_feasible(formulation, new_need, evidence.old_need_selection)
    assert (
        formulation.objective_for_selection("need_weighted", evidence.old_need_selection).hex()
        == FROZEN_OLD_NEED_INCUMBENT.hex()
    )

    # At the inherited incumbent, 99% is bit-for-bit the Stage4.2C retained
    # floor. Therefore the cost child differs only by the stricter D high-need
    # floor and is an exact subset of the historical cost model.
    new_need_floor = FROZEN_OLD_NEED_INCUMBENT * 0.99
    assert new_need_floor.hex() == FROZEN_OLD_NEED_FLOOR.hex()
    old_cost = formulation.build(
        objective_name="cost",
        sense="min",
        floors=[*old_floors, Floor("need_weighted", "max", FROZEN_OLD_NEED_FLOOR)],
        name="old-cost",
    )
    new_cost = formulation.build(
        objective_name="cost",
        sense="min",
        floors=[*new_floors, Floor("need_weighted", "max", new_need_floor)],
        name="new-cost",
    )
    cost_subset = exact_feasible_set_subset(old_cost, new_cost)

    assert cost_subset.sufficient is True
    assert cost_subset.failures == []
    assert cost_subset.tightened_row_count == 1
    assert cost_subset.tightened_rows[0]["row_name"].endswith("high_need_population")
    _assert_selection_feasible(formulation, new_cost, evidence.old_cost_selection)
    assert (
        formulation.objective_for_selection("cost", evidence.old_cost_selection).hex()
        == FROZEN_OLD_COST_INCUMBENT.hex()
    )


def test_real_evidence_rejects_a_tampered_parent_sha_before_use(
    actual_context: _ActualContext,
) -> None:
    cfg = copy.deepcopy(actual_context.cfg)
    cfg["stage42d_parent"]["artifact_sha256"][
        "CURRENT_STAGE4_2D_EQUITY_CERTIFICATION_RUN.json"
    ] = "0" * 64

    with pytest.raises(RuntimeError, match="SHA mismatch"):
        load_tail_evidence(ROOT, actual_context.formulation, cfg)


class _FakeBackend:
    version = "1.15.1"

    def __init__(self, formulation: CertificationFormulation) -> None:
        self.formulation = formulation
        self.calls: list[dict[str, Any]] = []

    def solve(
        self,
        model: LinearMipModel,
        options: Any,
        *,
        mip_start: np.ndarray | None = None,
    ) -> BackendResult:
        if model.objective_name == "need_weighted":
            selection = np.asarray([1, 3], dtype=int)
            bound_delta = 0.10
        elif model.objective_name == "cost":
            selection = np.asarray([2, 4], dtype=int)
            bound_delta = -0.01
        else:  # pragma: no cover - makes an unexpected third stage explicit
            raise AssertionError(model.objective_name)
        solution = self.formulation.complete_solution(model, selection)
        objective = self.formulation.objective_for_selection(model.objective_name, selection)
        bound = objective + bound_delta
        self.calls.append(
            {
                "objective": model.objective_name,
                "model": model,
                "options": options,
                "mip_start": None if mip_start is None else np.asarray(mip_start).copy(),
                "fresh_selection": selection.copy(),
            }
        )
        return BackendResult(
            model_name=model.name,
            status="TIME_LIMIT_WITH_INCUMBENT",
            has_incumbent=True,
            is_infeasible=False,
            is_optimal=False,
            objective_value_raw=objective,
            objective_value=objective,
            best_bound_raw=bound,
            best_bound=bound,
            relative_gap=abs(bound - objective) / abs(objective),
            mip_node_count=7,
            wall_time_sec=0.01,
            solution=solution,
            message="synthetic fresh solve",
            highs_version=self.version,
            options={
                "threads": options.threads,
                "parallel": options.parallel,
                "relative_gap": options.relative_gap,
                "random_seed": options.random_seed,
                "heuristic_effort": options.heuristic_effort,
                "skipped_options": {},
            },
            log_path=options.log_path,
        )


def _synthetic_cfg() -> dict[str, Any]:
    return {
        "contract": {"need_weighted_retention": 0.90},
        "solver": {
            "threads": 8,
            "relative_gap": 0.005,
            "parallel": True,
            "presolve": True,
            "time_limit_sec": {"need_weighted": 1.0, "cost": 1.0},
            "random_seed": {"need_weighted": 11, "cost": 22},
            "log_to_console": False,
        },
    }


def _synthetic_evidence(*, artifact_sha_verified: bool) -> TailEvidence:
    return TailEvidence(
        stage42d_run_id="synthetic-d",
        stage42c_run_id="synthetic-c",
        d_front_selection=np.asarray([1, 4], dtype=int),
        old_need_selection=np.asarray([2, 4], dtype=int),
        old_cost_selection=np.asarray([1, 3], dtype=int),
        old_need_incumbent=48.0,
        old_need_bound=52.05,
        old_need_floor=45.0,
        old_cost_incumbent=14.000006,
        old_cost_bound=10.995,
        artifact_sha_verified=artifact_sha_verified,
        records=[],
        d_pointer_payload={},
    )


@pytest.mark.parametrize("artifact_sha_verified", [True, False])
def test_solve_tail_models_runs_both_fresh_stages_prefers_better_incumbents_and_guards_inheritance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact_sha_verified: bool,
) -> None:
    formulation = make_synthetic_formulation()
    backend = _FakeBackend(formulation)
    monkeypatch.setattr(tail_runner, "FROZEN_TOTAL_POPULATION_FLOOR", 40.0)
    monkeypatch.setattr(tail_runner, "FROZEN_MIN_SIGUNGU_FLOOR", 0.40)
    monkeypatch.setattr(tail_runner, "FROZEN_HIGH_NEED_FLOOR", 44.0)
    monkeypatch.setattr(tail_runner, "FROZEN_OLD_NEED_INCUMBENT", 48.0)
    monkeypatch.setattr(tail_runner, "FROZEN_OLD_NEED_FLOOR", 45.0)

    def synthetic_floors(high_floor: float) -> list[Floor]:
        high = 43.0 if high_floor == OLD_HIGH_NEED_FLOOR else 44.0
        return [
            Floor("total_population", "max", 40.0),
            Floor("min_sigungu_coverage", "max", 0.40),
            Floor("high_need_population", "max", high),
        ]

    monkeypatch.setattr(tail_runner, "_base_floors", synthetic_floors)

    solved = tail_runner.solve_tail_models(
        formulation=formulation,
        evidence=_synthetic_evidence(artifact_sha_verified=artifact_sha_verified),
        cfg=_synthetic_cfg(),
        backend=backend,  # type: ignore[arg-type]
        run_dir=tmp_path,
    )

    assert [call["objective"] for call in backend.calls] == ["need_weighted", "cost"]
    assert all(call["mip_start"] is not None for call in backend.calls)
    assert all(call["options"].threads == 8 for call in backend.calls)
    assert backend.calls[0]["options"].random_seed == 11
    assert backend.calls[1]["options"].random_seed == 22

    assert solved.need.selected_source == "FRESH_HIGHS_1_15_1"
    assert solved.need.incumbent_value == 52.0
    np.testing.assert_array_equal(solved.need.selected_indices, np.asarray([1, 3]))
    assert solved.need.subset_audit["sufficient"] is True
    assert solved.need.inherited_bound_used is False
    assert solved.need.inherited_bound is None
    assert solved.need.historical_bound_diagnostic == 52.05
    assert solved.need.best_bound == 52.10
    assert solved.need.direct_best_bound == 52.10
    assert solved.need.fresh_incumbent_valid is True
    assert solved.need.fresh_bound_valid is True
    assert solved.need.certification_method == "FRESH_DIRECT_MIP_GAP"
    assert solved.need.semantic_label == "CERTIFIED_NEAR_OPTIMAL"
    assert solved.need.certified is True

    assert solved.cost.selected_source == "FRESH_HIGHS_1_15_1"
    np.testing.assert_array_equal(solved.cost.selected_indices, np.asarray([2, 4]))
    assert solved.cost.subset_audit["sufficient"] is True
    assert solved.cost.seed_audit["new_need_incumbent_not_below_old"] is True
    assert solved.cost.inherited_bound_used is False
    assert solved.cost.inherited_bound is None
    assert solved.cost.historical_bound_diagnostic == 10.995
    fresh_cost_bound = (
        formulation.objective_for_selection("cost", np.asarray([2, 4])) - 0.01
    )
    assert solved.cost.best_bound == fresh_cost_bound
    assert solved.cost.direct_best_bound == fresh_cost_bound
    assert solved.cost.fresh_incumbent_valid is True
    assert solved.cost.fresh_bound_valid is True
    assert solved.cost.certification_method == "FRESH_DIRECT_MIP_GAP"
    assert solved.cost.semantic_label == "CERTIFIED_NEAR_OPTIMAL"
    assert solved.cost.certified is True
    np.testing.assert_array_equal(solved.final_selection, np.asarray([2, 4]))
    assert solved.final_metrics["need_weighted_population"] == 48.0

    assert (tmp_path / "01_need_weighted/fresh.highs.log").is_file()
    assert (tmp_path / "02_cost/fresh.highs.log").is_file()

class _NoFreshIncumbentBackend(_FakeBackend):
    def solve(self, model, options, *, mip_start=None):  # type: ignore[override]
        if model.objective_name != "need_weighted":
            return super().solve(model, options, mip_start=mip_start)
        self.calls.append(
            {
                "objective": model.objective_name,
                "model": model,
                "options": options,
                "mip_start": None if mip_start is None else np.asarray(mip_start).copy(),
                "fresh_selection": None,
            }
        )
        return BackendResult(
            model_name=model.name,
            status="TIME_LIMIT_REACHED",
            has_incumbent=False,
            is_infeasible=False,
            is_optimal=False,
            objective_value_raw=None,
            objective_value=None,
            best_bound_raw=60.0,
            best_bound=60.0,
            relative_gap=None,
            mip_node_count=7,
            wall_time_sec=0.01,
            solution=None,
            message="synthetic no fresh incumbent",
            highs_version=self.version,
            options={
                "threads": options.threads,
                "parallel": options.parallel,
                "relative_gap": options.relative_gap,
                "random_seed": options.random_seed,
                "heuristic_effort": options.heuristic_effort,
                "skipped_options": {},
            },
            log_path=options.log_path,
        )


class _NoFreshBoundBackend(_FakeBackend):
    def solve(self, model, options, *, mip_start=None):  # type: ignore[override]
        result = super().solve(model, options, mip_start=mip_start)
        if model.objective_name == "need_weighted":
            result.best_bound = None
            result.best_bound_raw = None
        return result


def _patch_synthetic_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tail_runner, "FROZEN_TOTAL_POPULATION_FLOOR", 40.0)
    monkeypatch.setattr(tail_runner, "FROZEN_MIN_SIGUNGU_FLOOR", 0.40)
    monkeypatch.setattr(tail_runner, "FROZEN_HIGH_NEED_FLOOR", 44.0)
    monkeypatch.setattr(tail_runner, "FROZEN_OLD_NEED_INCUMBENT", 48.0)
    monkeypatch.setattr(tail_runner, "FROZEN_OLD_NEED_FLOOR", 45.0)

    def synthetic_floors(high_floor: float) -> list[Floor]:
        high = 43.0 if high_floor == OLD_HIGH_NEED_FLOOR else 44.0
        return [
            Floor("total_population", "max", 40.0),
            Floor("min_sigungu_coverage", "max", 0.40),
            Floor("high_need_population", "max", high),
        ]

    monkeypatch.setattr(tail_runner, "_base_floors", synthetic_floors)


def test_fresh_need_incumbent_is_mandatory_even_when_historical_evidence_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    formulation = make_synthetic_formulation()
    _patch_synthetic_contract(monkeypatch)
    with pytest.raises(RuntimeError, match="Fresh need-weighted solve did not return a CPU-valid incumbent"):
        tail_runner.solve_tail_models(
            formulation=formulation,
            evidence=_synthetic_evidence(artifact_sha_verified=True),
            cfg=_synthetic_cfg(),
            backend=_NoFreshIncumbentBackend(formulation),  # type: ignore[arg-type]
            run_dir=tmp_path,
        )


def test_fresh_need_bound_is_mandatory_even_when_historical_bound_is_verified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    formulation = make_synthetic_formulation()
    _patch_synthetic_contract(monkeypatch)
    with pytest.raises(RuntimeError, match="Fresh need-weighted solve did not return a valid upper bound"):
        tail_runner.solve_tail_models(
            formulation=formulation,
            evidence=_synthetic_evidence(artifact_sha_verified=True),
            cfg=_synthetic_cfg(),
            backend=_NoFreshBoundBackend(formulation),  # type: ignore[arg-type]
            run_dir=tmp_path,
        )

class _NeedOracleBackend(_FakeBackend):
    def solve(self, model, options, *, mip_start=None):  # type: ignore[override]
        if model.objective_name == "need_weighted":
            selection = np.asarray([1, 3], dtype=int)
            solution = self.formulation.complete_solution(model, selection)
            objective = self.formulation.objective_for_selection("need_weighted", selection)
            self.calls.append(
                {
                    "objective": model.objective_name,
                    "model": model,
                    "options": options,
                    "mip_start": None if mip_start is None else np.asarray(mip_start).copy(),
                    "fresh_selection": selection.copy(),
                }
            )
            return BackendResult(
                model_name=model.name,
                status="TIME_LIMIT_WITH_INCUMBENT",
                has_incumbent=True,
                is_infeasible=False,
                is_optimal=False,
                objective_value_raw=objective,
                objective_value=objective,
                best_bound_raw=60.0,
                best_bound=60.0,
                relative_gap=(60.0 - objective) / objective,
                mip_node_count=7,
                wall_time_sec=0.01,
                solution=solution,
                message="synthetic loose fresh bound",
                highs_version=self.version,
                options={
                    "threads": options.threads,
                    "parallel": options.parallel,
                    "relative_gap": options.relative_gap,
                    "random_seed": options.random_seed,
                    "heuristic_effort": options.heuristic_effort,
                    "skipped_options": {},
                },
                log_path=options.log_path,
            )
        if model.objective_name == "feasibility":
            self.calls.append(
                {
                    "objective": "feasibility",
                    "model": model,
                    "options": options,
                    "mip_start": mip_start,
                    "fresh_selection": None,
                }
            )
            return BackendResult(
                model_name=model.name,
                status="INFEASIBLE",
                has_incumbent=False,
                is_infeasible=True,
                is_optimal=False,
                objective_value_raw=None,
                objective_value=None,
                best_bound_raw=None,
                best_bound=None,
                relative_gap=None,
                mip_node_count=11,
                wall_time_sec=0.02,
                solution=None,
                message="synthetic threshold infeasibility proof",
                highs_version=self.version,
                options={
                    "threads": options.threads,
                    "parallel": options.parallel,
                    "relative_gap": options.relative_gap,
                    "random_seed": options.random_seed,
                    "heuristic_effort": options.heuristic_effort,
                    "skipped_options": {},
                },
                log_path=options.log_path,
            )
        return super().solve(model, options, mip_start=mip_start)


def test_need_threshold_oracle_can_certify_without_historical_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    formulation = make_synthetic_formulation()
    _patch_synthetic_contract(monkeypatch)
    solved = tail_runner.solve_tail_models(
        formulation=formulation,
        evidence=_synthetic_evidence(artifact_sha_verified=True),
        cfg=_synthetic_cfg(),
        backend=_NeedOracleBackend(formulation),  # type: ignore[arg-type]
        run_dir=tmp_path,
    )
    assert solved.need.certified is True
    assert solved.need.certification_method == "FRESH_THRESHOLD_INFEASIBILITY"
    assert solved.need.inherited_bound_used is False
    assert solved.need.fresh_incumbent_valid is True
    assert solved.need.fresh_bound_valid is True
    assert solved.need.direct_relative_gap > 0.005
    assert solved.need.relative_gap == 0.005
    assert solved.need.threshold_oracle is not None
    assert solved.need.threshold_oracle["certified"] is True
