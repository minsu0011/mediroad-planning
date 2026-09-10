from __future__ import annotations

import copy, json, os, platform, sys, time, traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from mediroad.stage4_2_certification.adapter import constrained_greedy, load_base_stage42_config, prepare_candidate_problem
from mediroad.stage4_2_certification.backends import HighsBackend, SolveOptions
from mediroad.stage4_2_certification.config import resolve_hardware
from mediroad.stage4_2_certification.engine import CertificationEngine
from mediroad.stage4_2_certification.formulation import CertificationFormulation
from mediroad.stage4_2_certification.io_utils import atomic_write_csv, atomic_write_json, atomic_write_text, build_inventory, run_id, sha256_file, single_writer_lock, utc_now_iso, verify_inventory
from mediroad.stage4_2_certification.types import Floor, LinearMipModel, ScenarioCertificationResult
from mediroad.stage4_2d_equity_certification.model_utils import append_cuts
from mediroad.stage4_2d_equity_certification.strengthening import generate_anchored_submodular_cuts
from mediroad.stage4_2d_equity_certification.types import EvidenceSeed, MetricTarget

from .compare import evaluate_top3_reduction
from .config import CandidateThresholds, load_yaml, set_thread_environment, validate_config
from .evidence import load_frozen_candidate_decision, load_top3_parent
from .gpu_lns import ExpansionGpuLns
from .scip_portfolio import run_portfolio
from .version import __version__


def _snapshot_pointers(root: Path) -> dict[str, Any]:
    names=[
        "outputs/model_v1/07_stage3/CURRENT_STAGE3_RUN.json",
        "outputs/model_v1/09_stage4_finalization/CURRENT_STAGE4_FINALIZATION_RUN.json",
        "outputs/model_v1/10_stage4_2/CURRENT_STAGE4_2_RUN.json",
        "outputs/model_v1/10_stage4_2d_equity_certification/CURRENT_STAGE4_2D_EQUITY_CERTIFICATION_RUN.json",
        "outputs/model_v1/10_stage4_2e_equity_tail_certification/CURRENT_STAGE4_2E_EQUITY_TAIL_CERTIFICATION_RUN.json",
        "outputs/model_v1/10_stage4_2f_top3_aggregate/CURRENT_STAGE4_2F_TOP3_AGGREGATE_RUN.json",
    ]
    return {n:{"exists":(root/n).exists(),"sha256":sha256_file(root/n) if (root/n).exists() else None} for n in names}


def _stage5_snapshot(root: Path) -> list[str]:
    out=[]
    for base in [root/"outputs/model_v1",root/"reports/model_v1"]:
        if base.exists():
            out.extend(str(p.relative_to(root).as_posix()) for p in base.iterdir() if "stage5" in p.name.lower())
            out.extend(str(p.relative_to(root).as_posix()) for p in base.rglob("CURRENT_STAGE5*.json"))
    return sorted(set(out))


def _source_snapshot(root: Path) -> list[dict[str,Any]]:
    rows=[]
    for base in [
        root/"src/mediroad/stage4_2",
        root/"src/mediroad/stage4_2_certification",
        root/"src/mediroad/stage4_2f_top3_aggregate",
        root/"src/mediroad/stage4_2g_candidate_expansion",
    ]:
        if base.exists():
            for p in sorted(base.rglob("*.py")):
                rows.append({"path":p.relative_to(root).as_posix(),"sha256":sha256_file(p),"size":p.stat().st_size})
    for rel in [
        "configs/model_v1/stage4_2.yaml",
        "configs/model_v1/stage4_2_certification.yaml",
        "configs/model_v1/stage4_2f_top3_aggregate.yaml",
        "configs/model_v1/stage4_2g_candidate_expansion.yaml",
    ]:
        p=root/rel
        if p.exists(): rows.append({"path":rel,"sha256":sha256_file(p),"size":p.stat().st_size})
    return rows


def _evidence_snapshot(paths: list[Path], root: Path) -> list[dict[str, Any]]:
    rows=[]
    for p in paths:
        if not p.exists():
            rows.append({"path":str(p),"exists":False,"sha256":None,"size":None})
        else:
            rows.append({"path":p.relative_to(root).as_posix(),"exists":True,"sha256":sha256_file(p),"size":p.stat().st_size})
    return rows


def _map_plan(formulation: CertificationFormulation, plan: pd.DataFrame) -> np.ndarray:
    pos={v:i for i,v in enumerate(formulation.candidates["venue_id"].astype(str))}
    aliases=getattr(formulation,"venue_alias_map",{})
    selected=[]
    for raw in plan["venue_id"].astype(str):
        v=str(aliases.get(raw,raw))
        if v not in pos: raise RuntimeError(f"Top3 seed venue missing from expanded universe: {raw} -> {v}")
        selected.append(pos[v])
    arr=np.asarray(selected,dtype=int)
    valid,reason=formulation.validate_selection(arr)
    if not valid: raise RuntimeError(f"Mapped Top3 seed invalid in expanded universe: {reason}")
    return arr


def _previous_g_seed(
    output_root: Path,
    formulation: CertificationFormulation,
    candidate_set: str,
    scenario: str,
    *,
    objective: str,
    total_population_floor: float,
) -> tuple[np.ndarray | None, dict[str, Any] | None]:
    """Recover a feasible plan from an older failed/interrupted G run as a seed only.

    A previous plan never carries proof status.  It must map into the current
    candidate universe and pass the current CPU constraint validator.
    """
    runs=output_root/"runs"
    if not runs.exists(): return None,None
    best=None; best_value=-np.inf; best_meta=None
    dirs=sorted((p for p in runs.iterdir() if p.is_dir()), key=lambda p:p.stat().st_mtime, reverse=True)
    for old in dirs:
        for stem in [old/"candidate_sets"/candidate_set/f"plan__{scenario}.csv", old/"candidate_sets_rerun"/candidate_set/f"plan__{scenario}.csv"]:
            if not stem.exists(): continue
            try:
                plan=pd.read_csv(stem,low_memory=False)
                idx=_map_plan(formulation,plan)
                metrics=formulation.metrics(idx)
                total=float(metrics["unique_elderly_population"])
                if total < float(total_population_floor)-1e-6: continue
                value=float(total if objective=="total_population" else metrics["need_weighted_population"])
                if value>best_value:
                    best=idx; best_value=value
                    best_meta={"source":stem.as_posix(),"objective":objective,"objective_value":value,"total_population":total}
            except Exception:
                continue
    return best,best_meta


def _choose_best_seed(
    formulation: CertificationFormulation,
    seeds: list[tuple[str,np.ndarray | None]],
    *,
    objective: str,
    total_population_floor: float,
) -> tuple[np.ndarray,str,float]:
    best=None; best_name=""; best_value=-np.inf
    for name,idx in seeds:
        if idx is None: continue
        idx=np.asarray(idx,dtype=int)
        valid,_=formulation.validate_selection(idx)
        if not valid: continue
        m=formulation.metrics(idx); total=float(m["unique_elderly_population"])
        if total<float(total_population_floor)-1e-6: continue
        value=float(total if objective=="total_population" else m["need_weighted_population"])
        if value>best_value:
            best=idx.copy(); best_name=name; best_value=value
    if best is None: raise RuntimeError(f"No feasible seed for {objective}")
    return best,best_name,best_value


def _refresh_scenario(result: ScenarioCertificationResult, gap: float) -> None:
    result.certified=bool(len(result.stages)==4 and all(s.certified for s in result.stages))
    result.certification_class="CERTIFIED_NEAR_OPTIMAL" if result.certified else "FEASIBLE_UNCERTIFIED"
    result.max_relative_gap=max((float(s.relative_gap) for s in result.stages if s.relative_gap is not None),default=None)
    if result.certified and result.max_relative_gap is not None and result.max_relative_gap>gap+1e-12:
        raise RuntimeError("Certified scenario exposes a gap above the frozen threshold")


def _integerize_scip_coverage_projection(
    model: LinearMipModel,
    pattern_coverers: list[np.ndarray] | None = None,
) -> LinearMipModel:
    """Strengthen a threshold-feasibility MIP without changing its x projection.

    The one-link formulation has ``0 <= y_p <= 1`` and
    ``y_p <= sum(covering x_i)``.  Every other row containing ``y`` is a
    non-negative retained/threshold lower bound.  Consequently, for any
    integer x selection with a feasible fractional y, replacing y by the exact
    covered-pattern indicator can only increase those lower-bound left-hand
    sides and keeps every cover link feasible.  Requiring y to be binary is
    therefore projection-equivalent, while giving SCIP a substantially
    stronger all-binary proof model for the min/high/need threshold stages.
    """
    if model.objective_name != "feasibility":
        raise ValueError("SCIP projection strengthening is threshold-feasibility only")
    if not bool(model.metadata.get("one_link_pattern_formulation")):
        raise ValueError("SCIP projection strengthening requires the one-link formulation")
    y = model.y_slice
    if y.start is None or y.stop is None or y.stop <= y.start:
        raise ValueError("SCIP projection strengthening requires coverage variables")
    if np.any(np.abs(model.c[y]) > 1e-12):
        raise ValueError("Threshold-feasibility coverage variables must have zero objective")
    y_upper = np.asarray(model.col_upper[y], dtype=float)
    if (
        np.any(np.abs(model.col_lower[y]) > 1e-12)
        or np.any(~(np.isclose(y_upper, 0.0, atol=1e-12, rtol=0.0) | np.isclose(y_upper, 1.0, atol=1e-12, rtol=0.0)))
    ):
        raise ValueError("Coverage variables must retain canonical [0,1] or fixed-[0,0] bounds")

    row_names = list(model.metadata.get("row_names", []))
    if len(row_names) != model.n_row:
        raise ValueError("SCIP projection strengthening requires complete row names")
    matrix = model.A.tocsr()
    for row_index, row_name in enumerate(row_names):
        row = matrix.getrow(row_index)
        mask = (row.indices >= y.start) & (row.indices < y.stop)
        y_values = row.data[mask]
        if y_values.size == 0:
            continue
        if str(row_name).startswith("cover_link::"):
            if (
                y_values.size != 1
                or abs(float(y_values[0]) - 1.0) > 1e-12
                or not np.isneginf(model.row_lower[row_index])
                or abs(float(model.row_upper[row_index])) > 1e-12
            ):
                raise ValueError(f"Unexpected coverage-link structure: {row_name}")
        elif str(row_name).startswith("floor::"):
            if (
                np.any(y_values < -1e-12)
                or not np.isfinite(model.row_lower[row_index])
                or not np.isposinf(model.row_upper[row_index])
            ):
                raise ValueError(f"Coverage floor is not a non-negative lower bound: {row_name}")
        else:
            raise ValueError(f"Unrecognized coverage-variable row: {row_name}")

    model.integrality[y] = 1
    canonical_row_count = 0
    canonical_nnz = 0
    if pattern_coverers is not None:
        if len(pattern_coverers) != y.stop - y.start:
            raise ValueError("Pattern-coverer count does not match the coverage-variable slice")
        add_rows: list[int] = []
        add_cols: list[int] = []
        add_data: list[float] = []
        add_names: list[str] = []
        for pattern_index, raw_coverers in enumerate(pattern_coverers):
            coverers = np.unique(np.asarray(raw_coverers, dtype=int))
            upper = float(model.col_upper[y.start + pattern_index])
            if coverers.size == 0:
                if abs(upper) > 1e-12:
                    raise ValueError("Uncovered pattern is not fixed to zero")
                continue
            if abs(upper - 1.0) > 1e-12:
                raise ValueError("Covered pattern does not retain unit upper bound")
            if np.any(coverers < 0) or np.any(coverers >= model.x_slice.stop - model.x_slice.start):
                raise ValueError("Pattern coverer index escapes the candidate slice")
            row_index = canonical_row_count
            for candidate_index in coverers:
                add_rows.append(row_index)
                add_cols.append(model.x_slice.start + int(candidate_index))
                add_data.append(1.0)
            add_rows.append(row_index)
            add_cols.append(y.start + pattern_index)
            add_data.append(-float(coverers.size))
            add_names.append(f"scip_canonical_or::{pattern_index}")
            canonical_row_count += 1
            canonical_nnz += int(coverers.size + 1)
        if canonical_row_count:
            extra = sparse.coo_matrix(
                (add_data, (add_rows, add_cols)),
                shape=(canonical_row_count, model.n_col),
                dtype=np.float64,
            ).tocsc()
            model.A = sparse.vstack([model.A, extra], format="csc")
            model.row_lower = np.concatenate([model.row_lower, np.full(canonical_row_count, -np.inf)])
            model.row_upper = np.concatenate([model.row_upper, np.zeros(canonical_row_count)])
            row_names.extend(add_names)
            model.metadata["row_names"] = row_names
    model.metadata["scip_projection_strengthening"] = {
        "coverage_variables_binary": True,
        "coverage_variable_count": int(y.stop - y.start),
        "x_projection_equivalent": True,
        "canonical_or_rows": int(canonical_row_count),
        "canonical_or_nnz": int(canonical_nnz),
        "proof": "canonical covered-pattern lift; sum(x_coverers) <= M*y plus y <= sum(x_coverers)",
    }
    model.validate()
    return model


def _threshold_metric_targets(
    formulation: CertificationFormulation,
    floors: list[Floor],
    threshold: Floor,
) -> list[MetricTarget]:
    """Translate monotone coverage floors into submodular proof targets."""

    strongest: dict[str, float] = {}
    for floor in [*floors, threshold]:
        if floor.sense != "max":
            continue
        name = str(floor.objective)
        if name not in {
            "total_population", "need_weighted", "high_need_population",
            "min_sigungu_coverage",
        }:
            continue
        strongest[name] = max(float(floor.value), strongest.get(name, -np.inf))

    targets: list[MetricTarget] = []
    arrays = {
        "total_population": formulation.population,
        "need_weighted": formulation.need_weighted,
        "high_need_population": formulation.high_need,
    }
    for name, weights in arrays.items():
        if name in strongest:
            targets.append(MetricTarget(
                name=name, target=float(strongest[name]),
                weights=np.asarray(weights, dtype=np.float64),
                source="STAGE42H_FROZEN_THRESHOLD_OR_RETAINED_FLOOR",
            ))
    if "min_sigungu_coverage" in strongest:
        ratio = float(strongest["min_sigungu_coverage"])
        for sigungu, pattern_idx in formulation.sigungu_pattern_indices.items():
            weights = np.zeros(formulation.ny, dtype=np.float64)
            weights[pattern_idx] = formulation.population[pattern_idx]
            targets.append(MetricTarget(
                name=f"sigungu::{sigungu}",
                target=ratio * float(formulation.sigungu_population_total[sigungu]),
                weights=weights,
                source="STAGE42H_FROZEN_MIN_SIGUNGU_THRESHOLD_OR_FLOOR",
            ))
    return targets


def _rescue_seed_pool(
    result: ScenarioCertificationResult,
    engine: CertificationEngine,
    formulation: CertificationFormulation,
) -> list[EvidenceSeed]:
    raw: list[tuple[str, np.ndarray | None]] = [("ENGINE_GPU_SEED", engine.greedy_indices)]
    raw.extend(
        (f"ENGINE_GPU_PROOF_ELITE_{index:03d}", np.asarray(selected, dtype=int))
        for index, selected in enumerate(getattr(engine, "stage42h_evidence_indices", []), start=1)
    )
    raw.extend((f"DIRECT_STAGE_{stage.stage_index:02d}_{stage.objective_name}", stage.selected_indices) for stage in result.stages)
    seeds: list[EvidenceSeed] = []
    seen: set[tuple[int, ...]] = set()
    for source, selected in raw:
        if selected is None:
            continue
        indices = np.asarray(selected, dtype=int)
        valid, _ = formulation.validate_selection(indices)
        key = tuple(sorted(int(i) for i in indices))
        if not valid or key in seen:
            continue
        seen.add(key)
        metrics = {
            k: float(v) for k, v in formulation.metrics(indices).items()
            if isinstance(v, (int, float, np.integer, np.floating))
        }
        seeds.append(EvidenceSeed(source=source, selected_indices=indices.copy(), metrics=metrics))
    if not seeds:
        raise RuntimeError("Stage4.2H rescue has no valid evidence seed")
    proof_cfg = getattr(engine, "stage42h_proof_config", {})
    return _diversify_evidence_seeds(
        seeds, formulation,
        limit=int(proof_cfg.get("proof_anchor_seed_count", 96)),
        random_seed=int(proof_cfg.get("proof_anchor_random_seed", 42062)),
    )


def _diversify_evidence_seeds(
    seeds: list[EvidenceSeed],
    formulation: CertificationFormulation,
    *,
    limit: int,
    random_seed: int,
) -> list[EvidenceSeed]:
    """Create deterministic hard-feasible anchor-only plans for valid cuts."""

    if limit <= len(seeds):
        return seeds[:limit]
    admin = formulation.candidates["admin_code"].astype(str).to_numpy()
    sigungu = formulation.candidates["sigungu"].astype(str).to_numpy()
    cluster = formulation.candidates["cluster_id"].fillna(formulation.candidates["venue_id"]).astype(str).to_numpy()

    def hard_valid(selected: np.ndarray) -> bool:
        if len(np.unique(selected)) != formulation.visit_count:
            return False
        if np.unique(admin[selected], return_counts=True)[1].max(initial=0) > formulation.max_admin:
            return False
        if np.unique(sigungu[selected], return_counts=True)[1].max(initial=0) > formulation.max_sigungu:
            return False
        if formulation.one_cluster and len(np.unique(cluster[selected])) != formulation.visit_count:
            return False
        return True

    output = list(seeds)
    seen = {tuple(sorted(int(i) for i in seed.selected_indices)) for seed in output}
    rng = np.random.default_rng(int(random_seed))
    attempts = 0
    max_attempts = max(2000, limit * 400)
    while len(output) < limit and attempts < max_attempts:
        base = np.asarray(output[attempts % len(output)].selected_indices, dtype=int).copy()
        radius = 1 + (attempts % min(8, formulation.visit_count - 1))
        candidate = base.copy()
        for _ in range(radius):
            accepted = False
            for _ in range(32):
                position = int(rng.integers(0, formulation.visit_count))
                replacement = int(rng.integers(0, formulation.nx))
                trial = candidate.copy(); trial[position] = replacement
                if hard_valid(trial):
                    candidate = trial; accepted = True; break
            if not accepted:
                break
        attempts += 1
        key = tuple(sorted(int(i) for i in candidate))
        if key in seen or not hard_valid(candidate):
            continue
        seen.add(key)
        output.append(EvidenceSeed(
            source=f"DETERMINISTIC_HARD_FEASIBLE_ANCHOR_{len(output)+1:03d}",
            selected_indices=candidate,
            metrics={},
        ))
    if len(output) < min(limit, 32):
        raise RuntimeError(f"Could generate only {len(output)} diversified proof anchors")
    return output


def _strengthen_threshold_model(
    model: LinearMipModel,
    *,
    formulation: CertificationFormulation,
    floors: list[Floor],
    threshold: Floor,
    evidence_seeds: list[EvidenceSeed],
    cfg: dict[str, Any],
) -> tuple[LinearMipModel, dict[str, Any]]:
    """Add outward-safe submodular cuts while preserving exact x projection.

    The continuous one-link y formulation already has the exact integer-x
    projection for monotone coverage floors. Stage 4.2H therefore keeps y
    continuous for both proof objectives, avoiding more than 10k redundant
    branching variables. Binary coverage remains an explicit opt-in only.
    """

    strengthening = cfg.get("proof_strengthening", {})
    targets = _threshold_metric_targets(formulation, floors, threshold)
    anchor_sizes = [int(v) for v in strengthening.get("anchor_sizes", list(range(formulation.visit_count + 1)))]
    per_metric_cap = int(strengthening.get("max_cuts_per_metric", 256))
    cuts = []
    target_diagnostics: list[dict[str, Any]] = []
    for target in targets:
        target_cuts, info = generate_anchored_submodular_cuts(
            formulation, [target], evidence_seeds,
            anchor_sizes=anchor_sizes, max_cuts=per_metric_cap,
            outward_safe=True,
        )
        cuts.extend(target_cuts)
        target_diagnostics.append({
            "metric": target.name,
            "target": float(target.target),
            **{k: v for k, v in info.items() if k != "cut_records"},
        })
    strengthened = append_cuts(model, cuts)
    binary_objectives = {str(v) for v in strengthening.get("binary_coverage_objectives", [])}
    coverage_binary = str(threshold.objective) in binary_objectives
    if coverage_binary:
        strengthened = _integerize_scip_coverage_projection(strengthened, None)
    strengthened.metadata["stage42h_proof_strengthening"] = {
        "outward_safe": True,
        "x_projection_equivalent": True,
        "coverage_variables_binary": coverage_binary,
        "dense_cut_count": len(cuts),
        "evidence_seed_count": len(evidence_seeds),
        "anchor_sizes": anchor_sizes,
        "max_cuts_per_metric": per_metric_cap,
        "targets": target_diagnostics,
    }
    strengthened.validate()
    return strengthened, strengthened.metadata["stage42h_proof_strengthening"]


def _candidate_only_proof_relaxation(model: LinearMipModel) -> LinearMipModel:
    """Project an already-strengthened threshold model to candidate variables.

    Rows involving coverage auxiliaries are dropped, while hard candidate rows
    and valid submodular superlevel cuts are retained. Every feasible solution
    of the full model projects into this relaxation, so proving this smaller
    model infeasible is an exact certificate for the full threshold oracle.
    """

    if model.objective_name != "feasibility" or model.x_slice.start != 0:
        raise ValueError("Candidate-only proof projection requires a zero-objective threshold model")
    nx = int(model.x_slice.stop)
    outside_nnz = np.asarray(model.A[:, nx:].getnnz(axis=1)).ravel()
    keep = outside_nnz == 0
    row_names = list(model.metadata.get("row_names", []))
    if len(row_names) != model.n_row:
        raise ValueError("Candidate-only proof projection requires complete row names")
    metadata = copy.deepcopy(model.metadata)
    metadata["row_names"] = [name for name, retained in zip(row_names, keep) if retained]
    metadata["stage42h_candidate_only_relaxation"] = {
        "proof_direction": "FULL_MODEL_FEASIBLE_IMPLIES_PROJECTED_MODEL_FEASIBLE",
        "infeasible_is_authoritative": True,
        "feasible_requires_canonical_full_model_validation": True,
        "source_rows": int(model.n_row),
        "retained_rows": int(np.count_nonzero(keep)),
        "source_columns": int(model.n_col),
        "retained_candidate_columns": nx,
    }
    projected = LinearMipModel(
        name=f"{model.name}__candidate_only_relaxation",
        objective_name="feasibility",
        sense="min",
        c=np.asarray(model.c[:nx], dtype=np.float64).copy(),
        A=model.A[keep, :nx].tocsc(),
        row_lower=np.asarray(model.row_lower[keep], dtype=np.float64).copy(),
        row_upper=np.asarray(model.row_upper[keep], dtype=np.float64).copy(),
        col_lower=np.asarray(model.col_lower[:nx], dtype=np.float64).copy(),
        col_upper=np.asarray(model.col_upper[:nx], dtype=np.float64).copy(),
        integrality=np.asarray(model.integrality[:nx], dtype=np.int8).copy(),
        variable_names=list(model.variable_names[:nx]),
        x_slice=slice(0, nx),
        y_slice=slice(nx, nx),
        extra_slices={},
        metadata=metadata,
    )
    projected.validate()
    return projected


def _highs_threshold_rescue(
    model: LinearMipModel,
    *,
    engine: CertificationEngine,
    formulation: CertificationFormulation,
    validation_model: LinearMipModel | None = None,
    output_dir: Path,
    time_limit_sec: float,
    cfg: dict[str, Any],
) -> tuple[str, np.ndarray | None, dict[str, Any]]:
    authoritative_model = validation_model if validation_model is not None else model
    working_model = model
    started = time.perf_counter()
    rounds: list[dict[str, Any]] = []
    final_kind = "INCONCLUSIVE"
    final_selected: np.ndarray | None = None
    max_rounds = int(cfg.get("proof_strengthening", {}).get("outer_approximation_max_rounds", 16))
    for round_index in range(1, max_rounds + 1):
        elapsed = time.perf_counter() - started
        remaining = float(time_limit_sec) - elapsed
        if remaining <= 1.0:
            break
        stem = f"strengthened_round_{round_index:02d}"
        options = SolveOptions(
            time_limit_sec=remaining, relative_gap=0.0,
            threads=int(engine.highs_threads), random_seed=42052 + round_index - 1,
            heuristic_effort=float(cfg["solver"].get("oracle_heuristic_effort", 0.08)),
            presolve=True, parallel=True, log_to_console=False,
            log_path=output_dir / f"{stem}.highs.log",
            improving_solution_path=output_dir / f"{stem}.highs.improving.sol",
            mip_max_start_nodes=int(cfg["solver"].get("mip_max_start_nodes", 50000)),
            extra_options=copy.deepcopy(cfg["solver"].get("extra_highs_options", {})),
        )
        solved = engine.backend.solve(working_model, options)
        round_payload: dict[str, Any] = {
            "round": round_index,
            "status": solved.status,
            "is_infeasible": bool(solved.is_infeasible),
            "has_incumbent": bool(solved.has_incumbent),
            "best_bound": solved.best_bound,
            "relative_gap": solved.relative_gap,
            "mip_node_count": solved.mip_node_count,
            "wall_time_sec": solved.wall_time_sec,
            "message": solved.message,
            "highs_version": solved.highs_version,
            "options": solved.options,
            "model_rows": working_model.n_row,
            "model_cols": working_model.n_col,
            "model_nnz": int(working_model.A.nnz),
        }
        rounds.append(round_payload)
        if solved.is_infeasible:
            final_kind = "INFEASIBLE_PROOF"
            break
        if not solved.has_incumbent or solved.solution is None:
            break
        selected = formulation.selected_from_solution(working_model, solved.solution)
        valid, reason = formulation.validate_selection(selected)
        completed = formulation.complete_solution(authoritative_model, selected) if valid else None
        violations = formulation.feasibility_violations(authoritative_model, completed) if completed is not None else {"invalid_selection": 1.0, "feasible": 0.0}
        round_payload["candidate_selection_size"] = int(selected.size)
        round_payload["selection_validation_error"] = reason
        round_payload["canonical_completion_violations"] = violations
        if valid and bool(violations.get("feasible", 0.0)):
            final_kind = "FEASIBLE_WITNESS"
            final_selected = selected
            break

        floors = [Floor(str(row["objective"]), str(row["sense"]), float(row["value"])) for row in authoritative_model.metadata.get("floors", [])]
        threshold_row = authoritative_model.metadata.get("threshold")
        if not isinstance(threshold_row, dict):
            break
        threshold = Floor(str(threshold_row["objective"]), str(threshold_row["sense"]), float(threshold_row["value"]))
        targets = _threshold_metric_targets(formulation, floors, threshold)
        covered = formulation.covered_mask(selected)
        violated_targets = [
            target for target in targets
            if float(np.asarray(target.weights, dtype=float)[covered].sum())
            < float(target.target) - 1e-8 * max(1.0, abs(float(target.target)))
        ]
        if not violated_targets:
            break
        seed = EvidenceSeed(
            source=f"CANDIDATE_ONLY_FALSE_POSITIVE_ROUND_{round_index:02d}",
            selected_indices=selected,
            metrics={},
        )
        separation_cuts = []
        separation_rows = []
        for target in violated_targets:
            new_cuts, info = generate_anchored_submodular_cuts(
                formulation, [target], [seed],
                anchor_sizes=range(formulation.visit_count + 1),
                max_cuts=int(cfg.get("proof_strengthening", {}).get("max_cuts_per_metric", 256)),
                outward_safe=True,
            )
            separation_cuts.extend(new_cuts)
            separation_rows.append({"metric": target.name, "cut_count": len(new_cuts), "diagnostics": {k: v for k, v in info.items() if k != "cut_records"}})
        round_payload["outer_approximation_separation"] = separation_rows
        round_payload["added_cut_count"] = len(separation_cuts)
        if not separation_cuts:
            break
        working_model = append_cuts(working_model, separation_cuts)

    last = rounds[-1] if rounds else {}
    payload = {
        "status": last.get("status", "TIME_BUDGET_EXHAUSTED"),
        "is_infeasible": final_kind == "INFEASIBLE_PROOF",
        "has_incumbent": final_kind == "FEASIBLE_WITNESS",
        "best_bound": last.get("best_bound"),
        "relative_gap": last.get("relative_gap"),
        "mip_node_count": sum(int(row.get("mip_node_count") or 0) for row in rounds),
        "wall_time_sec": time.perf_counter() - started,
        "message": f"outer_approximation={final_kind}; rounds={len(rounds)}",
        "highs_version": last.get("highs_version"),
        "options": last.get("options", {}),
        "proof_model": copy.deepcopy(model.metadata.get("stage42h_candidate_only_relaxation")),
        "outer_approximation_rounds": rounds,
        "final_model_rows": working_model.n_row,
        "final_model_cols": working_model.n_col,
        "final_model_nnz": int(working_model.A.nnz),
    }
    atomic_write_json(output_dir / "strengthened_highs_result.json", payload)
    return final_kind, final_selected, payload


def _rescue_scenario(result: ScenarioCertificationResult, *, engine: CertificationEngine, formulation: CertificationFormulation, outdir: Path, cfg: dict[str,Any]) -> tuple[ScenarioCertificationResult,np.ndarray|None]:
    gap=float(cfg["solver"]["near_optimal_relative_gap_max"])
    witness=None
    evidence_seeds=_rescue_seed_pool(result,engine,formulation)
    for stage in result.stages:
        if stage.certified or stage.objective_value is None or stage.selected_indices is None: continue
        floors=[Floor(str(x["objective"]),str(x["sense"]),float(x["value"])) for x in stage.details.get("floors_before_stage",[])]
        threshold=engine._oracle_threshold(stage.objective_name,stage.sense,float(stage.objective_value))
        model=formulation.build(objective_name="feasibility",sense="min",floors=floors,threshold=threshold,name=f"stage42g__{formulation.candidate_set}__{stage.scenario}__{stage.stage_index}__scip_threshold")
        stage_dir=outdir/f"stage_{stage.stage_index:02d}_{stage.objective_name}"
        stage_dir.mkdir(parents=True,exist_ok=True)
        precheck_rows=[]
        for seed in evidence_seeds:
            completed=formulation.complete_solution(model,seed.selected_indices)
            violations=formulation.feasibility_violations(model,completed)
            precheck_rows.append({"source":seed.source,"feasible":bool(violations["feasible"]),"violations":violations})
            if bool(violations["feasible"]):
                witness=np.asarray(seed.selected_indices,dtype=int).copy()
                atomic_write_json(stage_dir/"proof_seed_precheck.json",{"counterexample_found":True,"rows":precheck_rows})
                break
        if witness is not None:
            break
        atomic_write_json(stage_dir/"proof_seed_precheck.json",{"counterexample_found":False,"rows":precheck_rows})
        model,strengthening=_strengthen_threshold_model(
            model,formulation=formulation,floors=floors,threshold=threshold,
            evidence_seeds=evidence_seeds,cfg=cfg,
        )
        atomic_write_json(stage_dir/"proof_strengthening.json",{
            **strengthening,
            "evidence_seeds":[{"source":s.source,"selected_indices":[int(i) for i in s.selected_indices]} for s in evidence_seeds],
        })
        highs_limits=cfg.get("proof_strengthening",{}).get("highs_first_time_limit_sec",{})
        highs_limit=float(highs_limits.get(stage.objective_name,0.0))
        validation_model=model
        proof_model=_candidate_only_proof_relaxation(model)
        atomic_write_json(stage_dir/"candidate_only_proof_model.json",proof_model.metadata["stage42h_candidate_only_relaxation"])
        if highs_limit>0.0:
            highs_kind,highs_witness,highs_payload=_highs_threshold_rescue(
                proof_model,engine=engine,formulation=formulation,validation_model=validation_model,output_dir=stage_dir,
                time_limit_sec=highs_limit,cfg=cfg,
            )
            stage.details["strengthened_highs_threshold_proof"]=highs_payload
            if highs_kind=="INFEASIBLE_PROOF":
                stage.certified=True; stage.certificate="HIGHS_CANDIDATE_ONLY_RELAXATION_INFEASIBILITY"; stage.relative_gap=gap
                continue
            if highs_kind=="FEASIBLE_WITNESS":
                witness=highs_witness
                break
        portfolio=run_portfolio(proof_model,stage_dir,seeds=[int(x) for x in cfg["scip_portfolio"]["seeds"]],workers=int(cfg["scip_portfolio"]["workers"]),time_limit_sec=float(cfg["scip_portfolio"]["time_limit_sec"]),memory_limit_mib=float(cfg["scip_portfolio"]["memory_limit_mib_per_worker"]))
        term=portfolio.get("terminal")
        if term and term.get("kind")=="INFEASIBLE_PROOF":
            stage.certified=True; stage.certificate="SCIP_THRESHOLD_INFEASIBILITY"; stage.relative_gap=gap
            stage.details["scip_threshold_proof"]=portfolio
        elif term and term.get("kind")=="FEASIBLE_WITNESS":
            idx=[int(i) for i in term["worker"].get("selected_indices",[])]
            if len(idx)==formulation.visit_count:
                candidate=np.asarray(idx,dtype=int); valid,_=formulation.validate_selection(candidate)
                completed=formulation.complete_solution(validation_model,candidate) if valid else None
                violations=formulation.feasibility_violations(validation_model,completed) if completed is not None else {"feasible":0.0}
                portfolio["canonical_full_model_violations"]=violations
                atomic_write_json(stage_dir/"portfolio_summary.json",portfolio)
                if valid and bool(violations.get("feasible",0.0)):
                    witness=candidate
                    break
        else:
            stage.details["scip_threshold_proof"]=portfolio
    _refresh_scenario(result,gap)
    return result,witness


def _scenario_rows(results: dict[str,dict[str,ScenarioCertificationResult]]) -> pd.DataFrame:
    rows=[]
    for cset,by in results.items():
        for scenario,res in by.items():
            rows.append({"candidate_set":cset,"scenario":scenario,"certified":res.certified,"certification_class":res.certification_class,"max_relative_gap":res.max_relative_gap,**{k:v for k,v in res.metrics.items() if not isinstance(v,dict)}})
    return pd.DataFrame(rows)


def _telemetry_rows(results: dict[str,dict[str,ScenarioCertificationResult]]) -> pd.DataFrame:
    rows=[]
    for cset,by in results.items():
        for scenario,res in by.items():
            for s in res.stages:
                rows.append({"candidate_set":cset,"scenario":scenario,"stage_index":s.stage_index,"objective_name":s.objective_name,"certificate":s.certificate,"certified":s.certified,"objective_value":s.objective_value,"best_bound":s.best_bound,"relative_gap":s.relative_gap,"wall_time_sec":s.wall_time_sec,"mip_node_count":s.mip_node_count,"seed_source":s.seed_source})
    return pd.DataFrame(rows)


def run(project_root: Path, config_path: Path) -> dict[str,Any]:
    project_root=project_root.resolve(); cfg=load_yaml(config_path); validate_config(cfg); set_thread_environment()
    base_cfg=load_base_stage42_config(project_root/"configs/model_v1/stage4_2.yaml")
    if int(base_cfg["optimization"]["visit_count"])!=20: raise RuntimeError("Frozen visit_count changed")
    c_cfg=load_yaml(project_root/"configs/model_v1/stage4_2_certification.yaml")
    cert=copy.deepcopy(c_cfg)
    cert["hardware"].update({"highs_threads":8,"memory_limit_gb":float(cfg["hardware"]["memory_soft_limit_gib"]),"gpu_seed_search":True,"native_windows_highs_threads":4})
    cert["solver"]["time_limit_sec"]=copy.deepcopy(cfg["solver"]["time_limit_sec"])
    cert["solver"]["mip_heuristic_effort"]=float(cfg["solver"].get("mip_heuristic_effort",0.20))
    cert["oracle"].update(copy.deepcopy(cfg["oracle"]))
    cert["seed_search"].update({"single_swap_enabled":True,"max_rounds":int(cfg["gpu_lns"].get("single_swap_rounds",50)),"cpu_max_rounds":4})
    cert["gate"]["near_optimal_relative_gap_max"]=0.005
    hardware=resolve_hardware(cert); backend=HighsBackend(require_minimum_version=True)
    top3=load_top3_parent(project_root); frozen=load_frozen_candidate_decision(project_root)
    thresholds=CandidateThresholds(**{k:float(v) for k,v in frozen["decision"]["thresholds"].items()})
    output_root=project_root/cfg["paths"]["output_root"]; report_root=project_root/cfg["paths"]["report_root"]; lock=project_root/cfg["paths"]["lock"]
    ident=run_id("stage4_2g_candidate_expansion"); run_root=output_root/"runs"/ident; report=report_root/"runs"/ident
    run_root.mkdir(parents=True,exist_ok=False); report.mkdir(parents=True,exist_ok=False); (run_root/".RUNNING").write_text(utc_now_iso()+"\n")
    before_ptr=_snapshot_pointers(project_root); before_s5=_stage5_snapshot(project_root); before_src=_source_snapshot(project_root)
    evidence_paths=[
        frozen["decision_path"],
        top3["source_root"]/"candidate_sets/top3/plan__efficiency.csv",
        top3["source_root"]/"candidate_sets/top3/metrics__efficiency.json",
        top3["source_root"]/"candidate_sets/top3/plan__balanced.csv",
        top3["source_root"]/"candidate_sets/top3/metrics__balanced.json",
        project_root/"outputs/model_v1/10_stage4_2f_top3_aggregate/CURRENT_STAGE4_2F_TOP3_AGGREGATE_RUN.json",
    ]
    before_evidence=_evidence_snapshot(evidence_paths,project_root)
    atomic_write_json(run_root/"parent_pointers_before.json",before_ptr)
    atomic_write_json(run_root/"stage5_namespace_before.json",before_s5)
    atomic_write_csv(run_root/"tested_source_before.csv",pd.DataFrame(before_src))
    atomic_write_csv(run_root/"frozen_evidence_before.csv",pd.DataFrame(before_evidence))
    frozen_contract={
        "reference_candidate_set":"top3",
        "candidate_sets":["top5","coarse_pareto"],
        "required_scenarios":["efficiency","balanced"],
        "visit_count":20,
        "near_optimal_relative_gap_max":0.005,
        "thresholds":asdict(thresholds),
        "threshold_fingerprint":thresholds.fingerprint,
        "stage5_started":False,
    }
    atomic_write_json(run_root/"frozen_contract.json",frozen_contract)
    metadata={"run_id":ident,"started_utc":utc_now_iso(),"version":__version__,"hardware":asdict(hardware),"environment":{"python":sys.version,"platform":platform.platform(),"highspy_version":backend.version},"parent_stage4_2f":top3["pointer"],"frozen_stage4_1_candidate_decision":str(frozen["decision_path"].relative_to(project_root).as_posix()),"threshold_fingerprint":thresholds.fingerprint,"parent_pointers_before":before_ptr}
    results:dict[str,dict[str,ScenarioCertificationResult]]={}; lns_rows=[]; summaries=[]; cache={}
    try:
      with single_writer_lock(lock):
        for cset in ["top5","coarse_pareto"]:
            problem=prepare_candidate_problem(project_root,cset,base_cfg,cache=cache)
            f=CertificationFormulation(problem.candidates,problem.patterns,base_cfg,cert,candidate_set=cset); f.venue_alias_map=dict(problem.venue_alias_map)
            greedy_idx,greedy_pop=constrained_greedy(problem.candidates,problem.coverage,problem.grid["elderly_population"].to_numpy(float),base_cfg)
            top3_eff_seed=_map_plan(f,top3["efficiency_plan"])
            top3_bal_seed=_map_plan(f,top3["balanced_plan"])
            prev_eff,prev_eff_meta=_previous_g_seed(output_root,f,cset,"efficiency",objective="total_population",total_population_floor=float(greedy_pop))
            seed_eff,seed_eff_name,seed_eff_value=_choose_best_seed(
                f,[("TOP3_CERTIFIED",top3_eff_seed),("CONSTRAINED_GREEDY",greedy_idx),("PREVIOUS_STAGE4_2G",prev_eff)],
                objective="total_population",total_population_floor=float(greedy_pop),
            )
            lns=ExpansionGpuLns(f,cfg["gpu_lns"])
            if bool(cfg["gpu_lns"].get("require_cuda_for_official",False)) and lns.backend!="CUPY_CUDA":
                raise RuntimeError(f"Official Stage4.2G requires RTX/CuPy CUDA, got {lns.backend}")
            eff_lns=lns.search(seed_eff,objective="total_population",total_population_floor=float(greedy_pop))
            lns_rows.append({"candidate_set":cset,"scenario":"efficiency","initial_seed_source":seed_eff_name,"initial_seed_value":seed_eff_value,"previous_seed_meta":json.dumps(prev_eff_meta,ensure_ascii=False) if prev_eff_meta else None,**{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in asdict(eff_lns).items() if k!="selected_indices"}})
            engine=CertificationEngine(project_root=project_root,formulation=f,base_config=base_cfg,cert_config=cert,stage41=problem.stage41,greedy_indices=eff_lns.selected_indices,backend=backend,highs_threads=hardware.highs_threads,output_dir=run_root/"candidate_sets")
            eff=engine.solve_scenario("efficiency",efficiency_reference=None,greedy_population=float(greedy_pop))
            eff,witness=_rescue_scenario(eff,engine=engine,formulation=f,outdir=run_root/"scip_rescue"/cset/"efficiency",cfg=cfg)
            if witness is not None:
                engine=CertificationEngine(project_root=project_root,formulation=f,base_config=base_cfg,cert_config=cert,stage41=problem.stage41,greedy_indices=witness,backend=backend,highs_threads=hardware.highs_threads,output_dir=run_root/"candidate_sets_rerun")
                eff=engine.solve_scenario("efficiency",efficiency_reference=None,greedy_population=float(greedy_pop)); eff,_=_rescue_scenario(eff,engine=engine,formulation=f,outdir=run_root/"scip_rescue_rerun"/cset/"efficiency",cfg=cfg)
            # Checkpoint each completed scenario immediately so an interrupted run
            # can reuse its feasible plan without inheriting any proof claim.
            out=run_root/"candidate_sets"/cset; out.mkdir(parents=True,exist_ok=True)
            atomic_write_csv(out/"plan__efficiency.csv",eff.selected_frame)
            atomic_write_json(out/"metrics__efficiency.json",eff.metrics)
            atomic_write_json(out/"stages__efficiency.json",[{k:v for k,v in asdict(s).items() if k!="solution"} for s in eff.stages])
            eff_ref=float(eff.metrics["unique_elderly_population"])
            bal_floor=eff_ref*float(base_cfg["optimization"]["balanced_min_efficiency_fraction"])
            prev_bal,prev_bal_meta=_previous_g_seed(output_root,f,cset,"balanced",objective="need_weighted",total_population_floor=bal_floor)
            seed_bal,seed_bal_name,seed_bal_value=_choose_best_seed(
                f,[("TOP3_CERTIFIED",top3_bal_seed),("PREVIOUS_STAGE4_2G",prev_bal)],
                objective="need_weighted",total_population_floor=bal_floor,
            )
            bal_lns=lns.search(seed_bal,objective="need_weighted",total_population_floor=bal_floor)
            lns_rows.append({"candidate_set":cset,"scenario":"balanced","initial_seed_source":seed_bal_name,"initial_seed_value":seed_bal_value,"previous_seed_meta":json.dumps(prev_bal_meta,ensure_ascii=False) if prev_bal_meta else None,**{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in asdict(bal_lns).items() if k!="selected_indices"}})
            engine_b=CertificationEngine(project_root=project_root,formulation=f,base_config=base_cfg,cert_config=cert,stage41=problem.stage41,greedy_indices=bal_lns.selected_indices,backend=backend,highs_threads=hardware.highs_threads,output_dir=run_root/"candidate_sets")
            bal=engine_b.solve_scenario("balanced",efficiency_reference=eff_ref,greedy_population=float(greedy_pop)); bal,witness=_rescue_scenario(bal,engine=engine_b,formulation=f,outdir=run_root/"scip_rescue"/cset/"balanced",cfg=cfg)
            if witness is not None:
                engine_b=CertificationEngine(project_root=project_root,formulation=f,base_config=base_cfg,cert_config=cert,stage41=problem.stage41,greedy_indices=witness,backend=backend,highs_threads=hardware.highs_threads,output_dir=run_root/"candidate_sets_rerun")
                bal=engine_b.solve_scenario("balanced",efficiency_reference=eff_ref,greedy_population=float(greedy_pop)); bal,_=_rescue_scenario(bal,engine=engine_b,formulation=f,outdir=run_root/"scip_rescue_rerun"/cset/"balanced",cfg=cfg)
            results[cset]={"efficiency":eff,"balanced":bal}
            atomic_write_csv(out/"plan__balanced.csv",bal.selected_frame)
            atomic_write_json(out/"metrics__balanced.json",bal.metrics)
            atomic_write_json(out/"stages__balanced.json",[{k:v for k,v in asdict(s).items() if k!="solution"} for s in bal.stages])
            summaries.append({"candidate_set":cset,"candidate_rows":len(problem.candidates),"compressed_patterns":problem.patterns.n_patterns,"coverage_nnz":int(problem.coverage.nnz),"greedy_population":float(greedy_pop),"gpu_backend":lns.backend})
        metrics=_scenario_rows(results); telemetry=_telemetry_rows(results); atomic_write_csv(run_root/"scenario_metrics.csv",metrics); atomic_write_csv(run_root/"solver_stage_telemetry.csv",telemetry); atomic_write_csv(run_root/"gpu_lns_summary.csv",pd.DataFrame(lns_rows)); atomic_write_csv(run_root/"candidate_problem_summary.csv",pd.DataFrame(summaries))
        alternatives={name:(by["balanced"].metrics,by["balanced"].selected_frame,float(by["balanced"].max_relative_gap or 1.0),bool(by["balanced"].certified and by["efficiency"].certified)) for name,by in results.items()}
        comp,decision=evaluate_top3_reduction(baseline_metrics=top3["balanced_metrics"],baseline_plan=top3["balanced_plan"],baseline_max_gap=float(top3["balanced_max_gap"]),alternatives=alternatives,thresholds=thresholds)
        atomic_write_csv(run_root/"candidate_comparison.csv",comp); atomic_write_json(run_root/"candidate_decision.json",decision)
        all_expanded=all(res.certified for by in results.values() for res in by.values())
        comparison_conclusive=bool(decision["passed"] and all_expanded)
        retain=decision.get("recommended_candidate_set")=="top3" and comparison_conclusive
        promoted=decision.get("recommended_candidate_set")=="top5" and comparison_conclusive
        if retain: final_decision="PASS_STAGE4_2G_CANDIDATE_REDUCTION_CERTIFIED_RETAIN_TOP3"
        elif promoted: final_decision="PASS_STAGE4_2G_EXPANSION_CERTIFIED_PROMOTE_TOP5_REQUIRES_EQUITY_RECERTIFICATION"
        else: final_decision="FAIL_STAGE4_2G_CANDIDATE_EXPANSION_INCONCLUSIVE"
        passed=bool(comparison_conclusive)
        after_ptr=_snapshot_pointers(project_root); after_s5=_stage5_snapshot(project_root); after_src=_source_snapshot(project_root); after_evidence=_evidence_snapshot(evidence_paths,project_root)
        atomic_write_json(run_root/"parent_pointers_after.json",after_ptr)
        atomic_write_json(run_root/"stage5_namespace_after.json",after_s5)
        atomic_write_csv(run_root/"tested_source_after.csv",pd.DataFrame(after_src))
        atomic_write_csv(run_root/"frozen_evidence_after.csv",pd.DataFrame(after_evidence))
        quality={"decision":final_decision,"passed":passed,"candidate_expansion_certified":comparison_conclusive,"candidate_reduction_certified":retain,"recommended_candidate_set":decision.get("recommended_candidate_set"),"top3_all_policy_scenarios_already_certified":True,"expanded_efficiency_balanced_all_certified":all_expanded,"ready_for_stage4_computational_final_gate":retain,"top5_equity_recertification_required":promoted,"operational_final":False,"stage5_started":False,"stage5_release_allowed":False,"checks":{"parent_pointers_unchanged":after_ptr==before_ptr,"stage5_namespace_unchanged":after_s5==before_s5,"tested_source_unchanged":after_src==before_src,"frozen_evidence_unchanged":after_evidence==before_evidence,"threshold_fingerprint_exact":thresholds.fingerprint==frozen["decision"]["threshold_fingerprint"]}}
        if not all(quality["checks"].values()): raise RuntimeError(f"Stage4.2G provenance gate failed: {quality['checks']}")
        atomic_write_json(run_root/"quality_gate.json",quality); metadata.update({"completed_utc":utc_now_iso(),"quality_gate":quality}); atomic_write_json(run_root/"metadata.json",metadata)
        text="# MEDIROAD Stage 4.2G Candidate Expansion Certification\n\n"+f"- Run: `{ident}`\n- Decision: `{final_decision}`\n- Threshold fingerprint: `{thresholds.fingerprint}`\n\n## Candidate comparison\n\n"+comp.to_markdown(index=False)+"\n\n## Expanded scenario certification\n\n"+metrics.to_markdown(index=False)+"\n"
        atomic_write_text(run_root/"FINAL_REPORT.md",text); report.mkdir(parents=True,exist_ok=True); atomic_write_text(report/"FINAL_REPORT.md",text)
        (run_root/".RUNNING").unlink(missing_ok=True); atomic_write_text(run_root/(".COMMITTED" if passed else ".FAILED"),utc_now_iso()+"\n"); inv=build_inventory(run_root,exclude={"ARTIFACT_INVENTORY.csv"}); atomic_write_csv(run_root/"ARTIFACT_INVENTORY.csv",inv); verify_inventory(run_root,inv)
        pointer={"run_id":ident,"decision":final_decision,"passed":passed,"run_relative_path":run_root.relative_to(project_root).as_posix(),"quality_gate_sha256":sha256_file(run_root/"quality_gate.json"),"inventory_sha256":sha256_file(run_root/"ARTIFACT_INVENTORY.csv"),"stage5_started":False}
        if passed: atomic_write_json(output_root/"CURRENT_STAGE4_2G_CANDIDATE_EXPANSION_RUN.json",pointer)
        else: atomic_write_json(output_root/f"FAILED_STAGE4_2G__{ident}.json",pointer)
        return {"run_root":str(run_root),"decision":final_decision,"passed":passed,"quality_gate":quality}
    except BaseException as exc:
        (run_root/".RUNNING").unlink(missing_ok=True); atomic_write_text(run_root/".FAILED",utc_now_iso()+"\n"); atomic_write_json(run_root/"UNHANDLED_EXCEPTION.json",{"type":type(exc).__name__,"message":str(exc),"traceback":traceback.format_exc()}); raise
