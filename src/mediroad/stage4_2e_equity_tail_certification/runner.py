from __future__ import annotations

import hashlib
import json
import math
import platform
import sys
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mediroad.stage4_2_certification.adapter import (
    load_base_stage42_config,
    prepare_candidate_problem,
)
from mediroad.stage4_2_certification.backends import HighsBackend, SolveOptions
from mediroad.stage4_2_certification.config import validate_certification_config
from mediroad.stage4_2_certification.formulation import CertificationFormulation
from mediroad.stage4_2_certification.types import BackendResult, Floor, LinearMipModel
from mediroad.stage4_2d_equity_certification.config import (
    validate_config as validate_stage42d_config,
    validate_parent_contracts,
)
from mediroad.stage4_2d_equity_certification.io_utils import (
    atomic_write_csv,
    atomic_write_json,
    make_run_id,
    sha256_file,
    tree_inventory,
    utc_now,
)
from mediroad.stage4_2d_equity_certification.runner import (
    _atomic_write_text,
    _authoritative_input_snapshot,
    _exclusive_lock,
    _pointer_snapshot,
    _run_regression_tests,
    _source_snapshot,
    _stage5_snapshot,
    _verify_inventory,
    _verify_output_inventory,
)

from .audit import (
    GapAudit,
    SubsetAudit,
    audit_gap,
    choose_incumbent,
    exact_feasible_set_subset,
)
from .config import (
    FROZEN_GAP,
    FROZEN_HIGH_NEED_FLOOR,
    FROZEN_HIGHS_VERSION,
    FROZEN_MIN_SIGUNGU_FLOOR,
    FROZEN_OLD_HIGH_NEED_FLOOR,
    FROZEN_OLD_NEED_INCUMBENT,
    FROZEN_OLD_NEED_FLOOR,
    FROZEN_PATTERN_COUNT,
    FROZEN_TOP3_COUNT,
    FROZEN_TOTAL_POPULATION_FLOOR,
    load_yaml,
    set_thread_environment,
    validate_config,
)
from .evidence import TailEvidence, evidence_records_unchanged, load_tail_evidence
from .fresh_oracle import run_fresh_threshold_oracle
from .version import PACKAGE_NAME, VERSION


@dataclass
class TailStageResult:
    objective: str
    sense: str
    certified: bool
    semantic_label: str
    incumbent_value: float
    best_bound: float
    relative_gap: float
    selected_indices: np.ndarray
    selected_source: str
    retained_floor: Floor | None
    fresh_solve_completed: bool
    fresh_incumbent_valid: bool
    fresh_bound_valid: bool
    fresh_solver: dict[str, Any]
    certification_method: str
    direct_best_bound: float
    direct_relative_gap: float
    threshold_oracle: dict[str, Any] | None
    inherited_bound_used: bool
    inherited_bound: float | None
    historical_bound_diagnostic: float | None
    subset_audit: dict[str, Any]
    model: dict[str, Any]
    seed_audit: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TailSolveResult:
    need: TailStageResult
    cost: TailStageResult
    final_metrics: dict[str, Any]
    final_selection: np.ndarray


def _model_summary(model: LinearMipModel) -> dict[str, Any]:
    from mediroad.stage4_2d_equity_certification.canonical import canonical_model_digest

    return {
        "name": model.name,
        "objective": model.objective_name,
        "sense": model.sense,
        "rows": model.n_row,
        "cols": model.n_col,
        "nnz": int(model.A.nnz),
        "sha256": canonical_model_digest(model),
        "floors": model.metadata.get("floors", []),
    }


def _requested_options_payload(options: SolveOptions) -> dict[str, Any]:
    return {
        "time_limit_sec": float(options.time_limit_sec),
        "relative_gap": float(options.relative_gap),
        "threads": int(options.threads),
        "random_seed": int(options.random_seed),
        "parallel": bool(options.parallel),
        "presolve": bool(options.presolve),
        "heuristic_effort": float(options.heuristic_effort),
        "mip_max_start_nodes": int(options.mip_max_start_nodes),
        "log_to_console": bool(options.log_to_console),
        "extra_options": dict(options.extra_options),
    }


def _backend_payload(result: BackendResult, options: SolveOptions) -> dict[str, Any]:
    return {
        "model_name": result.model_name,
        "status": result.status,
        "has_incumbent": result.has_incumbent,
        "is_infeasible": result.is_infeasible,
        "is_optimal_exact": result.is_optimal,
        "objective_value_raw": result.objective_value_raw,
        "objective_value_reported": result.objective_value,
        "best_bound_raw": result.best_bound_raw,
        "best_bound_reported": result.best_bound,
        "solver_reported_relative_gap": result.relative_gap,
        "mip_node_count": result.mip_node_count,
        "wall_time_sec": result.wall_time_sec,
        "message": result.message,
        "highs_version": result.highs_version,
        "options_readback": result.options,
        "options_requested": _requested_options_payload(options),
        "log_path": str(result.log_path) if result.log_path else None,
    }


def _selection_feasibility(
    formulation: CertificationFormulation,
    model: LinearMipModel,
    selected: np.ndarray,
) -> tuple[bool, dict[str, Any], float]:
    selected = np.asarray(selected, dtype=int)
    valid, reason = formulation.validate_selection(selected)
    if not valid:
        return False, {"hard_selection_valid": False, "reason": reason}, math.nan
    full = formulation.complete_solution(model, selected)
    violations = formulation.feasibility_violations(model, full, tol=1e-6)
    value = formulation.objective_for_selection(model.objective_name, selected)
    return bool(violations["feasible"]), {
        "hard_selection_valid": True,
        "reason": None,
        "violations": violations,
        "cpu_recomputed_objective": value,
    }, float(value)


def _fresh_selection(
    formulation: CertificationFormulation,
    model: LinearMipModel,
    result: BackendResult,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    if not result.has_incumbent or result.solution is None:
        return None, {"available": False, "reason": "NO_FRESH_HIGHS_INCUMBENT"}
    selected = formulation.selected_from_solution(model, result.solution)
    feasible, audit, _ = _selection_feasibility(formulation, model, selected)
    audit["available"] = True
    audit["selected_indices"] = selected.tolist()
    if not feasible:
        return None, audit
    return selected, audit


def _solve_options(cfg: dict[str, Any], objective: str, stage_dir: Path) -> SolveOptions:
    solver = cfg["solver"]
    return SolveOptions(
        time_limit_sec=float(solver["time_limit_sec"][objective]),
        relative_gap=float(solver["relative_gap"]),
        threads=int(solver["threads"]),
        random_seed=int(solver["random_seed"][objective]),
        parallel=bool(solver["parallel"]),
        presolve=bool(solver["presolve"]),
        heuristic_effort=float(solver.get("heuristic_effort", 0.15)),
        mip_max_start_nodes=int(solver.get("mip_max_start_nodes", 10000)),
        log_to_console=bool(solver.get("log_to_console", True)),
        log_path=stage_dir / "fresh.highs.log",
        improving_solution_path=stage_dir / "fresh.highs.improving.sol",
        extra_options=dict(solver.get("extra_highs_options", {})),
    )


def _seal_solver_files(options: SolveOptions) -> None:
    for path, label in (
        (options.log_path, "NO_HIGHS_LOG_OUTPUT"),
        (options.improving_solution_path, "NO_IMPROVING_SOLUTION_OUTPUT"),
    ):
        if path is not None and (not path.exists() or path.stat().st_size == 0):
            _atomic_write_text(path, f"{label}\n")


def _valid_fresh_bound(sense: str, bound: float | None, incumbent: float) -> bool:
    if bound is None or not math.isfinite(float(bound)):
        return False
    slack = 1e-10 * max(1.0, abs(float(bound)), abs(float(incumbent)))
    return bool(float(bound) >= incumbent - slack) if sense == "max" else bool(
        float(bound) <= incumbent + slack
    )


def _fresh_contract_verified(result: BackendResult, options: SolveOptions) -> bool:
    skipped = result.options.get("skipped_options", {})
    frozen_extra_applied = isinstance(skipped, dict) and not any(
        key in skipped for key in options.extra_options
    )
    return bool(
        result.highs_version == FROZEN_HIGHS_VERSION
        and int(result.options.get("threads", -1)) == 8
        and result.options.get("parallel") is True
        and int(result.options.get("random_seed", -1)) == int(options.random_seed)
        and float(result.options.get("heuristic_effort", math.nan))
        == float(options.heuristic_effort)
        and frozen_extra_applied
        and not result.is_infeasible
        and "ERROR" not in str(result.status).upper()
    )


def _persisted_solver_contract_verified(
    payload: dict[str, Any], cfg: dict[str, Any], objective: str
) -> bool:
    solver = cfg["solver"]
    expected = {
        "time_limit_sec": float(solver["time_limit_sec"][objective]),
        "relative_gap": float(solver["relative_gap"]),
        "threads": int(solver["threads"]),
        "random_seed": int(solver["random_seed"][objective]),
        "parallel": bool(solver["parallel"]),
        "presolve": bool(solver["presolve"]),
        "heuristic_effort": float(solver["heuristic_effort"]),
        "mip_max_start_nodes": int(solver["mip_max_start_nodes"]),
        "log_to_console": bool(solver["log_to_console"]),
        "extra_options": dict(solver["extra_highs_options"]),
    }
    requested = payload.get("options_requested")
    readback = payload.get("options_readback")
    if requested != expected or not isinstance(readback, dict):
        return False
    skipped = readback.get("skipped_options", {})
    return bool(
        payload.get("highs_version") == FROZEN_HIGHS_VERSION
        and readback.get("threads") == expected["threads"]
        and readback.get("parallel") is expected["parallel"]
        and readback.get("random_seed") == expected["random_seed"]
        and readback.get("heuristic_effort") == expected["heuristic_effort"]
        and isinstance(skipped, dict)
        and not any(key in skipped for key in expected["extra_options"])
    )
def _base_floors(high_floor: float) -> list[Floor]:
    return [
        Floor("total_population", "max", FROZEN_TOTAL_POPULATION_FLOOR),
        Floor("min_sigungu_coverage", "max", FROZEN_MIN_SIGUNGU_FLOOR),
        Floor("high_need_population", "max", float(high_floor)),
    ]


def solve_tail_models(
    *,
    formulation: CertificationFormulation,
    evidence: TailEvidence,
    cfg: dict[str, Any],
    backend: HighsBackend,
    run_dir: Path,
) -> TailSolveResult:
    """Rebuild, freshly solve and certify the Equity need/cost tail."""

    if backend.version != FROZEN_HIGHS_VERSION:
        raise RuntimeError(
            f"Stage4.2E requires fresh HiGHS {FROZEN_HIGHS_VERSION}; found {backend.version}"
        )
    old_need_floors = _base_floors(FROZEN_OLD_HIGH_NEED_FLOOR)
    new_need_floors = _base_floors(FROZEN_HIGH_NEED_FLOOR)
    old_need_model = formulation.build(
        objective_name="need_weighted",
        sense="max",
        floors=old_need_floors,
        name="stage42e__old_feasible_set__need_weighted",
    )
    need_model = formulation.build(
        objective_name="need_weighted",
        sense="max",
        floors=new_need_floors,
        name="stage42e__fresh__need_weighted",
    )
    need_subset = exact_feasible_set_subset(old_need_model, need_model)
    old_need_feasible, old_need_audit, old_need_value = _selection_feasibility(
        formulation, need_model, evidence.old_need_selection
    )
    if not old_need_feasible or old_need_value.hex() != FROZEN_OLD_NEED_INCUMBENT.hex():
        raise RuntimeError(
            "The pinned old need plan is not a feasible exact incumbent floor in the new feasible set"
        )
    d_feasible, d_audit, d_need_value = _selection_feasibility(
        formulation, need_model, evidence.d_front_selection
    )
    need_seed = evidence.old_need_selection
    if d_feasible and d_need_value > old_need_value:
        need_seed = evidence.d_front_selection
    need_options = _solve_options(cfg, "need_weighted", run_dir / "01_need_weighted")
    need_fresh = backend.solve(
        need_model,
        need_options,
        mip_start=formulation.complete_solution(need_model, need_seed),
    )
    _seal_solver_files(need_options)
    if not _fresh_contract_verified(need_fresh, need_options):
        raise RuntimeError("Fresh need-weighted HiGHS solve violated the frozen solver contract")
    need_fresh_selected, need_fresh_audit = _fresh_selection(
        formulation, need_model, need_fresh
    )
    need_candidates: list[tuple[str, float, np.ndarray]] = [
        ("PINNED_STAGE42C_NEED_INCUMBENT", old_need_value, evidence.old_need_selection)
    ]
    if d_feasible:
        need_candidates.append(("PINNED_STAGE42D_FRONT_SEED", d_need_value, evidence.d_front_selection))
    if need_fresh_selected is not None:
        need_candidates.append(
            (
                "FRESH_HIGHS_1_15_1",
                formulation.objective_for_selection("need_weighted", need_fresh_selected),
                need_fresh_selected,
            )
        )
    if need_fresh_selected is None:
        raise RuntimeError("Fresh need-weighted solve did not return a CPU-valid incumbent")
    need_source, need_incumbent, need_selected = choose_incumbent("max", need_candidates)
    # Historical Stage4.2C bounds are diagnostics only.  The official certificate
    # must be built from this run's HiGHS solve and, if necessary, a fresh
    # threshold-infeasibility oracle.
    need_fresh_bound_valid = _valid_fresh_bound("max", need_fresh.best_bound, need_incumbent)
    if not need_fresh_bound_valid:
        raise RuntimeError("Fresh need-weighted solve did not return a valid upper bound")
    need_direct_bound = float(need_fresh.best_bound)
    need_direct_gap = audit_gap("max", need_incumbent, need_direct_bound, FROZEN_GAP)
    if not need_direct_gap.bound_direction_valid:
        raise RuntimeError("Fresh need-weighted bound contradicts a CPU-validated feasible incumbent")

    need_certified = bool(need_direct_gap.within_limit)
    need_semantic = need_direct_gap.semantic_label
    need_certificate_bound = need_direct_bound
    need_certificate_gap = need_direct_gap.relative_gap
    need_method = "FRESH_DIRECT_MIP_GAP"
    need_oracle_payload: dict[str, Any] | None = None
    if not need_certified:
        need_oracle = run_fresh_threshold_oracle(
            formulation=formulation,
            backend=backend,
            objective="need_weighted",
            sense="max",
            floors=new_need_floors,
            incumbent_selection=need_selected,
            incumbent_value=need_incumbent,
            base_options=need_options,
            output_dir=run_dir / "01_need_weighted" / "fresh_threshold_oracle",
            relative_gap=FROZEN_GAP,
            max_rounds=2,
            time_limit_sec=float(cfg["solver"]["time_limit_sec"]["need_weighted"]),
        )
        need_oracle_payload = need_oracle.to_dict()
        if need_oracle.incumbent_value != need_incumbent or not np.array_equal(
            need_oracle.selected_indices, need_selected
        ):
            need_incumbent = float(need_oracle.incumbent_value)
            need_selected = np.asarray(need_oracle.selected_indices, dtype=int)
            need_source = need_oracle.selected_source
            if not _valid_fresh_bound("max", need_fresh.best_bound, need_incumbent):
                raise RuntimeError("Fresh need-weighted bound is invalid after oracle incumbent improvement")
            need_direct_gap = audit_gap("max", need_incumbent, need_direct_bound, FROZEN_GAP)
        if need_oracle.certified:
            need_certified = True
            need_semantic = "CERTIFIED_NEAR_OPTIMAL"
            need_certificate_bound = float(need_oracle.policy_bound)
            need_certificate_gap = float(FROZEN_GAP)
            need_method = need_oracle.method
        else:
            need_method = need_oracle.method

    need_floor_value = float(need_incumbent * float(cfg["contract"]["need_weighted_retention"]))
    need_floor = Floor("need_weighted", "max", need_floor_value)
    need_stage = TailStageResult(
        objective="need_weighted",
        sense="max",
        certified=need_certified,
        semantic_label=need_semantic if need_certified else "UNCERTIFIED",
        incumbent_value=need_incumbent,
        best_bound=need_certificate_bound,
        relative_gap=need_certificate_gap if need_certified else need_direct_gap.relative_gap,
        selected_indices=need_selected,
        selected_source=need_source,
        retained_floor=need_floor,
        fresh_solve_completed=True,
        fresh_incumbent_valid=True,
        fresh_bound_valid=True,
        fresh_solver=_backend_payload(need_fresh, need_options),
        certification_method=need_method,
        direct_best_bound=need_direct_bound,
        direct_relative_gap=need_direct_gap.relative_gap,
        threshold_oracle=need_oracle_payload,
        inherited_bound_used=False,
        inherited_bound=None,
        historical_bound_diagnostic=evidence.old_need_bound,
        subset_audit=need_subset.to_dict(),
        model={
            "old": _model_summary(old_need_model),
            "new": _model_summary(need_model),
        },
        seed_audit={
            "old_need": old_need_audit,
            "stage42d_front": d_audit,
            "fresh": need_fresh_audit,
            "preference_rule": "best CPU-valid incumbent may seed search; certification never uses historical bounds",
            "historical_bound_diagnostic_only": evidence.old_need_bound,
        },
    )

    old_cost_model = formulation.build(
        objective_name="cost",
        sense="min",
        floors=[*old_need_floors, Floor("need_weighted", "max", FROZEN_OLD_NEED_FLOOR)],
        name="stage42e__old_feasible_set__cost",
    )
    cost_model = formulation.build(
        objective_name="cost",
        sense="min",
        floors=[*new_need_floors, need_floor],
        name="stage42e__fresh__cost",
    )
    cost_subset = exact_feasible_set_subset(old_cost_model, cost_model)
    new_need_not_below_old = bool(need_incumbent >= evidence.old_need_incumbent)
    old_cost_feasible, old_cost_audit, old_cost_value = _selection_feasibility(
        formulation, cost_model, evidence.old_cost_selection
    )
    need_cost_feasible, need_cost_audit, need_cost_value = _selection_feasibility(
        formulation, cost_model, need_selected
    )
    if not need_cost_feasible:
        raise RuntimeError("Selected new need incumbent does not satisfy its own retained cost-stage floor")
    cost_seed = evidence.old_cost_selection if old_cost_feasible else need_selected
    cost_options = _solve_options(cfg, "cost", run_dir / "02_cost")
    cost_fresh = backend.solve(
        cost_model,
        cost_options,
        mip_start=formulation.complete_solution(cost_model, cost_seed),
    )
    _seal_solver_files(cost_options)
    if not _fresh_contract_verified(cost_fresh, cost_options):
        raise RuntimeError("Fresh cost HiGHS solve violated the frozen solver contract")
    cost_fresh_selected, cost_fresh_audit = _fresh_selection(
        formulation, cost_model, cost_fresh
    )
    cost_candidates: list[tuple[str, float, np.ndarray]] = [
        ("NEW_NEED_INCUMBENT_AS_COST_SEED", need_cost_value, need_selected)
    ]
    if old_cost_feasible:
        cost_candidates.append(
            ("PINNED_STAGE42C_COST_INCUMBENT", old_cost_value, evidence.old_cost_selection)
        )
    if cost_fresh_selected is not None:
        cost_candidates.append(
            (
                "FRESH_HIGHS_1_15_1",
                formulation.objective_for_selection("cost", cost_fresh_selected),
                cost_fresh_selected,
            )
        )
    if cost_fresh_selected is None:
        raise RuntimeError("Fresh cost solve did not return a CPU-valid incumbent")
    cost_source, cost_incumbent, cost_selected = choose_incumbent("min", cost_candidates)
    cost_fresh_bound_valid = _valid_fresh_bound("min", cost_fresh.best_bound, cost_incumbent)
    if not cost_fresh_bound_valid:
        raise RuntimeError("Fresh cost solve did not return a valid lower bound")
    cost_direct_bound = float(cost_fresh.best_bound)
    cost_direct_gap = audit_gap("min", cost_incumbent, cost_direct_bound, FROZEN_GAP)
    if not cost_direct_gap.bound_direction_valid:
        raise RuntimeError("Fresh cost bound contradicts a CPU-validated feasible incumbent")

    cost_certified = bool(cost_direct_gap.within_limit)
    cost_semantic = cost_direct_gap.semantic_label
    cost_certificate_bound = cost_direct_bound
    cost_certificate_gap = cost_direct_gap.relative_gap
    cost_method = "FRESH_DIRECT_MIP_GAP"
    cost_oracle_payload: dict[str, Any] | None = None
    if not cost_certified:
        cost_oracle = run_fresh_threshold_oracle(
            formulation=formulation,
            backend=backend,
            objective="cost",
            sense="min",
            floors=[*new_need_floors, need_floor],
            incumbent_selection=cost_selected,
            incumbent_value=cost_incumbent,
            base_options=cost_options,
            output_dir=run_dir / "02_cost" / "fresh_threshold_oracle",
            relative_gap=FROZEN_GAP,
            max_rounds=2,
            time_limit_sec=float(cfg["solver"]["time_limit_sec"]["cost"]),
        )
        cost_oracle_payload = cost_oracle.to_dict()
        if cost_oracle.incumbent_value != cost_incumbent or not np.array_equal(
            cost_oracle.selected_indices, cost_selected
        ):
            cost_incumbent = float(cost_oracle.incumbent_value)
            cost_selected = np.asarray(cost_oracle.selected_indices, dtype=int)
            cost_source = cost_oracle.selected_source
            if not _valid_fresh_bound("min", cost_fresh.best_bound, cost_incumbent):
                raise RuntimeError("Fresh cost bound is invalid after oracle incumbent improvement")
            cost_direct_gap = audit_gap("min", cost_incumbent, cost_direct_bound, FROZEN_GAP)
        if cost_oracle.certified:
            cost_certified = True
            cost_semantic = "CERTIFIED_NEAR_OPTIMAL"
            cost_certificate_bound = float(cost_oracle.policy_bound)
            cost_certificate_gap = float(FROZEN_GAP)
            cost_method = cost_oracle.method
        else:
            cost_method = cost_oracle.method

    cost_stage = TailStageResult(
        objective="cost",
        sense="min",
        certified=cost_certified,
        semantic_label=cost_semantic if cost_certified else "UNCERTIFIED",
        incumbent_value=cost_incumbent,
        best_bound=cost_certificate_bound,
        relative_gap=cost_certificate_gap if cost_certified else cost_direct_gap.relative_gap,
        selected_indices=cost_selected,
        selected_source=cost_source,
        retained_floor=None,
        fresh_solve_completed=True,
        fresh_incumbent_valid=True,
        fresh_bound_valid=True,
        fresh_solver=_backend_payload(cost_fresh, cost_options),
        certification_method=cost_method,
        direct_best_bound=cost_direct_bound,
        direct_relative_gap=cost_direct_gap.relative_gap,
        threshold_oracle=cost_oracle_payload,
        inherited_bound_used=False,
        inherited_bound=None,
        historical_bound_diagnostic=evidence.old_cost_bound,
        subset_audit=cost_subset.to_dict(),
        model={
            "old": _model_summary(old_cost_model),
            "new": _model_summary(cost_model),
        },
        seed_audit={
            "old_cost": old_cost_audit,
            "old_cost_plan_feasible_in_new_model": old_cost_feasible,
            "fallback_seed_when_old_cost_infeasible": "NEW_NEED_INCUMBENT",
            "new_need_incumbent_as_cost_seed": need_cost_audit,
            "fresh": cost_fresh_audit,
            "new_need_incumbent_not_below_old": new_need_not_below_old,
            "historical_bound_diagnostic_only": evidence.old_cost_bound,
        },
    )
    return TailSolveResult(
        need=need_stage,
        cost=cost_stage,
        final_metrics=formulation.metrics(cost_selected),
        final_selection=cost_selected,
    )


def _source_snapshot_e(root: Path, config_paths: list[Path]) -> list[dict[str, Any]]:
    records = {row["relative_path"]: row for row in _source_snapshot(root, config_paths)}
    extra: set[Path] = set()
    package = root / "src/mediroad/stage4_2e_equity_tail_certification"
    if package.exists():
        extra.update(package.glob("*.py"))
    for relative in (
        "12_scripts/v6/run_model_v1_stage4_2e_equity_tail_certification.py",
        "run_model_v1_stage4_2e_equity_tail_certification.ps1",
    ):
        path = root / relative
        if path.is_file():
            extra.add(path)
    for path in extra:
        rel = path.resolve().relative_to(root.resolve()).as_posix()
        records[rel] = {
            "relative_path": rel,
            "size_bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
        }
    return [records[key] for key in sorted(records)]


def _parent_snapshot_e(root: Path) -> dict[str, dict[str, Any]]:
    result = _pointer_snapshot(root)
    path = root / "outputs/model_v1/10_stage4_2d_equity_certification/CURRENT_STAGE4_2D_EQUITY_CERTIFICATION_RUN.json"
    result[path.relative_to(root).as_posix()] = {
        "exists": path.exists(),
        "sha256": sha256_file(path) if path.exists() else None,
    }
    return result


def _authoritative_snapshot_e(root: Path) -> list[dict[str, Any]]:
    records = list(_authoritative_input_snapshot(root))
    pointer = root / "outputs/model_v1/10_stage4_2d_equity_certification/CURRENT_STAGE4_2D_EQUITY_CERTIFICATION_RUN.json"
    if not pointer.is_file():
        raise RuntimeError("Required Stage4.2D CURRENT pointer is missing")
    payload = json.loads(pointer.read_text(encoding="utf-8"))
    inventory = root / str(payload.get("inventory_relative_path", ""))
    verified = _verify_inventory(root, inventory)
    if verified["sha256"] != str(payload.get("inventory_sha256", "")).lower():
        raise RuntimeError("Stage4.2D CURRENT inventory binding failed")
    records.append(
        {
            "pointer_relative_path": pointer.relative_to(root).as_posix(),
            "pointer_sha256": sha256_file(pointer),
            "run_id": payload.get("run_id"),
            **verified,
        }
    )
    return sorted(records, key=lambda row: str(row["pointer_relative_path"]))


def _write_report(path: Path, run_id: str, decision: str, result: TailSolveResult) -> None:
    need, cost = result.need, result.cost
    lines = [
        "# MEDIROAD Stage 4.2E Equity Tail Certification",
        "",
        f"- Run: `{run_id}`",
        f"- Decision: `{decision}`",
        "- Scope: `TOP3_EQUITY_FULL_ONLY`",
        "- Candidate universe / visits: `Top3 / 20`",
        f"- High-need retained floor: `{FROZEN_HIGH_NEED_FLOOR!r}`",
        "",
        "| stage | semantic result | incumbent | valid bound | relative gap | inherited bound |",
        "|---|---|---:|---:|---:|---|",
        f"| need_weighted | {need.semantic_label} | {need.incumbent_value:.12f} | {need.best_bound:.12f} | {need.relative_gap:.8f} | {need.inherited_bound_used} |",
        f"| cost | {cost.semantic_label} | {cost.incumbent_value:.12f} | {cost.best_bound:.12f} | {cost.relative_gap:.8f} | {cost.inherited_bound_used} |",
        "",
        "Both stages were rebuilt and run through a fresh HiGHS 1.15.1 solve with 8 CPU threads. "
        "Historical bounds were used only after pinned source SHA verification and an exact feasible-set subset audit.",
        "",
        "This completes only the four-stage Top3 Equity lexicographic computational certification when PASS. "
        "Candidate expansion, full Stage4 computation, Operational Final, and Stage5 remain false.",
        "",
    ]
    _atomic_write_text(path, "\n".join(lines))


def _final_plan_frame(
    formulation: CertificationFormulation, selected: np.ndarray
) -> pd.DataFrame:
    """Return the canonical Equity plan without duplicating provenance columns."""

    plan = formulation.candidates.iloc[np.asarray(selected, dtype=int)].copy().reset_index(
        drop=True
    )
    # Candidate preparation now carries candidate_set itself.  Older frozen
    # inputs did not, so support both schemas while forcing the same values.
    if "candidate_set" in plan.columns:
        plan["candidate_set"] = "top3"
    else:
        plan.insert(0, "candidate_set", "top3")
    if "scenario" in plan.columns:
        plan["scenario"] = "equity"
    else:
        plan.insert(0, "scenario", "equity")
    return plan


def run_equity_tail_certification(
    *,
    project_root: Path,
    stage42_config_path: Path,
    stage42c_config_path: Path,
    stage42d_config_path: Path,
    stage42e_config_path: Path,
) -> dict[str, Any]:
    set_thread_environment(8)
    root = project_root.resolve()

    def resolved(path: Path) -> Path:
        return path.resolve() if path.is_absolute() else (root / path).resolve()

    bcfg_path = resolved(stage42_config_path)
    ccfg_path = resolved(stage42c_config_path)
    dcfg_path = resolved(stage42d_config_path)
    ecfg_path = resolved(stage42e_config_path)
    cfg = load_yaml(ecfg_path)
    validate_config(cfg)
    base_cfg = load_base_stage42_config(bcfg_path)
    cert_cfg = load_yaml(ccfg_path)
    d_cfg = load_yaml(dcfg_path)
    validate_certification_config(cert_cfg)
    validate_parent_contracts(base_cfg, cert_cfg)
    validate_stage42d_config(d_cfg)

    config_paths = [bcfg_path, ccfg_path, dcfg_path, ecfg_path]
    source_before = _source_snapshot_e(root, config_paths)
    parent_before = _parent_snapshot_e(root)
    authoritative_before = _authoritative_snapshot_e(root)
    stage5_before = _stage5_snapshot(root)
    fingerprint_payload = {
        "package": PACKAGE_NAME,
        "version": VERSION,
        "configs": {path.name: sha256_file(path) for path in config_paths},
        "source": source_before,
        "parents": parent_before,
        "authoritative_inputs": authoritative_before,
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    run_id = make_run_id("stage4_2e_equity_tail", fingerprint)
    output_root = root / str(cfg["paths"]["output_root"])
    report_root = root / str(cfg["paths"]["report_root"])
    run_dir = output_root / "runs" / run_id
    report_dir = report_root / "runs" / run_id
    lock = root / str(cfg["paths"]["lock"])
    running = run_dir / ".RUNNING"
    committed = run_dir / ".COMMITTED"
    failed = run_dir / ".FAILED"

    with _exclusive_lock(lock):
        run_dir.mkdir(parents=True, exist_ok=False)
        report_dir.mkdir(parents=True, exist_ok=False)
        atomic_write_json(running, {"run_id": run_id, "started_utc": utc_now()})
        try:
            regression = _run_regression_tests(root, run_dir / "00_regression")
            problem = prepare_candidate_problem(root, "top3", base_cfg, cache={})
            formulation = CertificationFormulation(
                problem.candidates,
                problem.patterns,
                base_cfg,
                cert_cfg,
                candidate_set="top3",
            )
            formulation.venue_alias_map = dict(problem.venue_alias_map)
            if (
                formulation.nx != FROZEN_TOP3_COUNT
                or formulation.ny != FROZEN_PATTERN_COUNT
                or formulation.visit_count != 20
            ):
                raise RuntimeError("Current candidate problem differs from frozen Top3/20 contract")
            evidence = load_tail_evidence(root, formulation, cfg)

            d_metadata_path = root / str(evidence.d_pointer_payload["metadata_relative_path"])
            d_metadata = json.loads(d_metadata_path.read_text(encoding="utf-8"))
            expected_config_hashes = {
                "stage42": sha256_file(bcfg_path),
                "stage42c": sha256_file(ccfg_path),
                "stage42d": sha256_file(dcfg_path),
            }
            observed_fingerprint = d_metadata.get("fingerprint_payload", {})
            if any(observed_fingerprint.get(key) != digest for key, digest in expected_config_hashes.items()):
                raise RuntimeError("Current Stage4.2/2C/2D config hashes do not match pinned D execution")

            backend = HighsBackend(require_minimum_version=True)
            solved = solve_tail_models(
                formulation=formulation,
                evidence=evidence,
                cfg=cfg,
                backend=backend,
                run_dir=run_dir,
            )
            for directory, stage in (
                (run_dir / "01_need_weighted", solved.need),
                (run_dir / "02_cost", solved.cost),
            ):
                atomic_write_json(directory / "old_to_new_feasible_set_subset.json", stage.subset_audit)
                atomic_write_json(directory / "fresh_solver_result.json", stage.fresh_solver)
                atomic_write_json(directory / "certificate.json", stage.to_dict())

            final_dir = run_dir / "03_final_incumbent"
            plan = _final_plan_frame(formulation, solved.final_selection)
            atomic_write_csv(final_dir / "plan__equity_full_certified.csv", plan)
            atomic_write_json(final_dir / "metrics.json", solved.final_metrics)
            atomic_write_csv(
                final_dir / "certification_summary.csv",
                pd.DataFrame(
                    [
                        {
                            "stage": 3,
                            "objective": solved.need.objective,
                            "semantic_label": solved.need.semantic_label,
                            "certified": solved.need.certified,
                            "incumbent": solved.need.incumbent_value,
                            "best_bound": solved.need.best_bound,
                            "relative_gap": solved.need.relative_gap,
                            "fresh_solve": solved.need.fresh_solve_completed,
                            "fresh_incumbent_valid": solved.need.fresh_incumbent_valid,
                            "fresh_bound_valid": solved.need.fresh_bound_valid,
                            "inherited_bound_used": solved.need.inherited_bound_used,
                            "certification_method": solved.need.certification_method,
                            "direct_best_bound": solved.need.direct_best_bound,
                            "direct_relative_gap": solved.need.direct_relative_gap,
                        },
                        {
                            "stage": 4,
                            "objective": solved.cost.objective,
                            "semantic_label": solved.cost.semantic_label,
                            "certified": solved.cost.certified,
                            "incumbent": solved.cost.incumbent_value,
                            "best_bound": solved.cost.best_bound,
                            "relative_gap": solved.cost.relative_gap,
                            "fresh_solve": solved.cost.fresh_solve_completed,
                            "fresh_incumbent_valid": solved.cost.fresh_incumbent_valid,
                            "fresh_bound_valid": solved.cost.fresh_bound_valid,
                            "inherited_bound_used": solved.cost.inherited_bound_used,
                            "certification_method": solved.cost.certification_method,
                            "direct_best_bound": solved.cost.direct_best_bound,
                            "direct_relative_gap": solved.cost.direct_relative_gap,
                        },
                    ]
                ),
            )

            parent_after = _parent_snapshot_e(root)
            authoritative_after = _authoritative_snapshot_e(root)
            stage5_after = _stage5_snapshot(root)
            source_after = _source_snapshot_e(root, config_paths)
            checks = {
                "parent_pointers_unchanged": parent_after == parent_before,
                "authoritative_input_inventories_unchanged": authoritative_after
                == authoritative_before,
                "stage5_namespace_absent_before": int(stage5_before.get("count", -1)) == 0,
                "stage5_namespace_unchanged": stage5_after == stage5_before,
                "tested_source_unchanged": source_after == source_before,
                "regression_suite_passed": regression.get("return_code") == 0
                and int(regression.get("failed_count", 0)) == 0,
                "pinned_d_and_c_artifact_sha_unchanged": evidence_records_unchanged(
                    root, evidence.records
                ),
                "top3_candidate_count_verified": formulation.nx == FROZEN_TOP3_COUNT,
                "pattern_count_verified": formulation.ny == FROZEN_PATTERN_COUNT,
                "visit_count_verified": formulation.visit_count == 20,
                "high_need_floor_binary64_verified": float(
                    cfg["contract"]["high_need_floor"]
                ).hex()
                == FROZEN_HIGH_NEED_FLOOR.hex(),
                "fresh_highs_version_verified": backend.version == FROZEN_HIGHS_VERSION,
                "fresh_need_solve_completed": solved.need.fresh_solve_completed,
                "fresh_cost_solve_completed": solved.cost.fresh_solve_completed,
                "fresh_need_solver_contract_verified": _persisted_solver_contract_verified(
                    solved.need.fresh_solver, cfg, "need_weighted"
                ),
                "fresh_cost_solver_contract_verified": _persisted_solver_contract_verified(
                    solved.cost.fresh_solver, cfg, "cost"
                ),
                "need_old_plan_feasible_and_incumbent_floor": bool(
                    solved.need.seed_audit["old_need"].get("violations", {}).get("feasible")
                ),
                "need_subset_audit_completed": "sufficient" in solved.need.subset_audit,
                "cost_subset_audit_completed": "sufficient" in solved.cost.subset_audit,
                "fresh_need_incumbent_valid": solved.need.fresh_incumbent_valid,
                "fresh_cost_incumbent_valid": solved.cost.fresh_incumbent_valid,
                "fresh_need_bound_valid": solved.need.fresh_bound_valid,
                "fresh_cost_bound_valid": solved.cost.fresh_bound_valid,
                "need_inherited_bound_forbidden": solved.need.inherited_bound_used is False,
                "cost_inherited_bound_forbidden": solved.cost.inherited_bound_used is False,
                "historical_need_bound_diagnostic_only": solved.need.historical_bound_diagnostic
                == evidence.old_need_bound and solved.need.inherited_bound is None,
                "historical_cost_bound_diagnostic_only": solved.cost.historical_bound_diagnostic
                == evidence.old_cost_bound and solved.cost.inherited_bound is None,
                "need_bound_direction_and_gap_certified": solved.need.certified
                and solved.need.relative_gap <= FROZEN_GAP + 1e-12,
                "cost_bound_direction_and_gap_certified": solved.cost.certified
                and solved.cost.relative_gap <= FROZEN_GAP + 1e-12,
                "need_semantic_label_valid": solved.need.semantic_label
                in {"CERTIFIED_EXACT", "CERTIFIED_NEAR_OPTIMAL"},
                "cost_semantic_label_valid": solved.cost.semantic_label
                in {"CERTIFIED_EXACT", "CERTIFIED_NEAR_OPTIMAL"},
            }
            passed = all(checks.values())
            decision = (
                "PASS_STAGE4_2E_TOP3_EQUITY_FULL_CERTIFIED"
                if passed
                else "FAIL_STAGE4_2E_TOP3_EQUITY_TAIL_UNCERTIFIED"
            )
            gate = {
                "decision": decision,
                "passed": passed,
                "promotable_scope": "TOP3_EQUITY_FULL_ONLY" if passed else "NONE",
                "top3_equity_full_lexicographic_certified": passed,
                "checks": checks,
                "candidate_expansion_certified": False,
                "stage4_full_computational_complete": False,
                "operational_final": False,
                "stage5_started": False,
                "stage5_release_allowed": False,
            }
            quality_dir = run_dir / "04_quality_and_provenance"
            atomic_write_json(quality_dir / "quality_gate.json", gate)
            atomic_write_json(quality_dir / "parent_pointers_before.json", parent_before)
            atomic_write_json(quality_dir / "parent_pointers_after.json", parent_after)
            atomic_write_json(quality_dir / "authoritative_inputs_before.json", authoritative_before)
            atomic_write_json(quality_dir / "authoritative_inputs_after.json", authoritative_after)
            atomic_write_json(
                quality_dir / "namespace_guard.json",
                {"stage5_before": stage5_before, "stage5_after": stage5_after},
            )
            atomic_write_json(quality_dir / "tested_source_before.json", source_before)
            atomic_write_json(quality_dir / "tested_source_after.json", source_after)
            atomic_write_json(quality_dir / "pinned_evidence.json", evidence.to_dict())

            metadata = {
                "run_id": run_id,
                "package": PACKAGE_NAME,
                "version": VERSION,
                "created_utc": utc_now(),
                "fingerprint": fingerprint,
                "fingerprint_payload": fingerprint_payload,
                "python_executable": sys.executable,
                "python_version": sys.version,
                "platform": platform.platform(),
                "highs_version": backend.version,
                "regression": regression,
                "quality_gate": gate,
                "need_weighted": solved.need.to_dict(),
                "cost": solved.cost.to_dict(),
                "final_metrics": solved.final_metrics,
                "scope": "TOP3_EQUITY_FULL_ONLY" if passed else "NONE",
                "candidate_expansion_certified": False,
                "stage4_full_computational_complete": False,
                "operational_final": False,
                "stage5_started": False,
                "stage5_release_allowed": False,
            }
            metadata_path = run_dir / "metadata.json"
            atomic_write_json(metadata_path, metadata)
            report_path = report_dir / "FINAL_REPORT.md"
            _write_report(report_path, run_id, decision, solved)
            atomic_write_json(report_dir / "quality_gate.json", gate)

            marker_names = {".RUNNING", ".FAILED", ".COMMITTED", "ARTIFACT_INVENTORY.csv"}
            artifacts = [
                path
                for path in run_dir.rglob("*")
                if path.is_file() and path.name not in marker_names
            ] + [path for path in report_dir.rglob("*") if path.is_file()]
            inventory_path = run_dir / "ARTIFACT_INVENTORY.csv"
            atomic_write_csv(inventory_path, pd.DataFrame(tree_inventory(artifacts, root)))
            _verify_output_inventory(root, inventory_path)

            # Final commit-time revalidation. CURRENT is written only after the
            # immutable marker, so readers can never observe a partial run.
            if _parent_snapshot_e(root) != parent_before:
                raise RuntimeError("Parent pointer changed before Stage4.2E commit")
            if _authoritative_snapshot_e(root) != authoritative_before:
                raise RuntimeError("Authoritative input changed before Stage4.2E commit")
            if _stage5_snapshot(root) != stage5_before:
                raise RuntimeError("Stage5 namespace changed before Stage4.2E commit")
            if _source_snapshot_e(root, config_paths) != source_before:
                raise RuntimeError("Tested Stage4.2E source changed before commit")
            if not evidence_records_unchanged(root, evidence.records):
                raise RuntimeError("Pinned Stage4.2C/2D evidence changed before commit")
            _verify_output_inventory(root, inventory_path)
            running.unlink(missing_ok=True)
            if passed:
                atomic_write_json(
                    committed,
                    {
                        "run_id": run_id,
                        "committed_utc": utc_now(),
                        "inventory_sha256": sha256_file(inventory_path),
                    },
                )
                pointer = output_root / "CURRENT_STAGE4_2E_EQUITY_TAIL_CERTIFICATION_RUN.json"
                atomic_write_json(
                    pointer,
                    {
                        "run_id": run_id,
                        "run_relative_path": run_dir.relative_to(root).as_posix(),
                        "metadata_relative_path": metadata_path.relative_to(root).as_posix(),
                        "metadata_sha256": sha256_file(metadata_path),
                        "inventory_relative_path": inventory_path.relative_to(root).as_posix(),
                        "inventory_sha256": sha256_file(inventory_path),
                        "report_relative_path": report_path.relative_to(root).as_posix(),
                        "report_sha256": sha256_file(report_path),
                        "quality_gate_relative_path": (
                            quality_dir / "quality_gate.json"
                        ).relative_to(root).as_posix(),
                        "quality_gate_sha256": sha256_file(quality_dir / "quality_gate.json"),
                        "need_certificate_relative_path": (
                            run_dir / "01_need_weighted/certificate.json"
                        ).relative_to(root).as_posix(),
                        "need_certificate_sha256": sha256_file(
                            run_dir / "01_need_weighted/certificate.json"
                        ),
                        "cost_certificate_relative_path": (
                            run_dir / "02_cost/certificate.json"
                        ).relative_to(root).as_posix(),
                        "cost_certificate_sha256": sha256_file(
                            run_dir / "02_cost/certificate.json"
                        ),
                        "decision": decision,
                        "scope": "TOP3_EQUITY_FULL_ONLY",
                        "top3_equity_full_lexicographic_certified": True,
                        "candidate_expansion_certified": False,
                        "stage4_full_computational_complete": False,
                        "operational_final": False,
                        "stage5_started": False,
                        "stage5_release_allowed": False,
                        "created_utc": utc_now(),
                    },
                )
            else:
                atomic_write_json(
                    failed,
                    {"run_id": run_id, "failed_utc": utc_now(), "decision": decision},
                )
                atomic_write_json(
                    output_root / "LATEST_FAILED_STAGE4_2E_EQUITY_TAIL_RUN.json",
                    {
                        "run_id": run_id,
                        "run_relative_path": run_dir.relative_to(root).as_posix(),
                        "decision": decision,
                    },
                )
            return {
                "run_id": run_id,
                "passed": passed,
                "decision": decision,
                "run_dir": str(run_dir),
                "report_dir": str(report_dir),
                "need_weighted": solved.need.to_dict(),
                "cost": solved.cost.to_dict(),
            }
        except Exception as exc:
            running.unlink(missing_ok=True)
            # A pointer is the sole authoritative promotion record.  If any
            # post-inventory step fails before that atomic write completes,
            # never leave this run simultaneously COMMITTED and FAILED.
            committed.unlink(missing_ok=True)
            atomic_write_json(
                failed,
                {
                    "run_id": run_id,
                    "failed_utc": utc_now(),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )
            atomic_write_json(
                output_root / "LATEST_FAILED_STAGE4_2E_EQUITY_TAIL_RUN.json",
                {
                    "run_id": run_id,
                    "run_relative_path": run_dir.relative_to(root).as_posix(),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            raise
