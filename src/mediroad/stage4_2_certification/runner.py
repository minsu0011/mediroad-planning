from __future__ import annotations

import json
import os
import platform
import sys
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .adapter import constrained_greedy, load_base_stage42_config, prepare_candidate_problem
from .backends import HighsBackend
from .config import (
    load_yaml,
    resolve_hardware,
    set_thread_environment,
    validate_certification_config,
)
from .engine import CertificationEngine
from .formulation import CertificationFormulation
from .io_utils import (
    atomic_write_csv,
    atomic_write_json,
    atomic_write_text,
    build_inventory,
    run_id,
    sha256_file,
    single_writer_lock,
    utc_now_iso,
    verify_inventory,
)
from .types import CandidateProblem, ScenarioCertificationResult, StageResult
from .version import __version__


def _path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _pointer_snapshot(root: Path) -> dict[str, Any]:
    pointers = [
        root / "outputs/model_v1/07_stage3/CURRENT_STAGE3_RUN.json",
        root / "outputs/model_v1/09_stage4_finalization/CURRENT_STAGE4_FINALIZATION_RUN.json",
        root / "outputs/model_v1/10_stage4_2/CURRENT_STAGE4_2_RUN.json",
    ]
    return {
        path.relative_to(root).as_posix(): {
            "exists": path.exists(),
            "sha256": sha256_file(path) if path.exists() else None,
        }
        for path in pointers
    }


def _candidate_expansion_prerequisite(
    top3_results: dict[str, ScenarioCertificationResult],
    cert_config: dict[str, Any],
) -> bool:
    """Fail closed before expensive expansion when a required Top3 axis failed."""

    efficiency = top3_results.get("efficiency")
    balanced = top3_results.get("balanced")
    equity = top3_results.get("equity")
    eb_certified = bool(
        efficiency is not None
        and efficiency.certified
        and balanced is not None
        and balanced.certified
    )
    fail_fast = bool(
        cert_config.get("gate", {}).get(
            "fail_fast_after_required_top3_uncertified", True
        )
    )
    if not fail_fast:
        return eb_certified
    if bool(cert_config.get("gate", {}).get("require_equity_certification", True)):
        return bool(eb_certified and equity is not None and equity.certified)
    return eb_certified


def _source_inventory(project_root: Path) -> list[dict[str, Any]]:
    package = Path(__file__).resolve().parent
    paths = set(package.glob("*.py"))
    paths.update((project_root / "src/mediroad/stage4_2").rglob("*.py"))
    for relative in [
        "src/mediroad/__init__.py",
        "12_scripts/v6/run_model_v1_stage4_2_certification.py",
        "run_model_v1_stage4_2_certification.ps1",
        "run_model_v1_stage4_2_certification.sh",
        "requirements_stage4_2_certification.txt",
        "requirements_stage4_2_certification_diagnostics.txt",
        "environment_stage4_2_certification.yml",
        "12_scripts/v6/probe_stage4_2c_scip_oracle.py",
    ]:
        path = project_root / relative
        if path.is_file():
            paths.add(path)
    rows: list[dict[str, Any]] = []
    for path in sorted(paths):
        try:
            name = path.resolve().relative_to(project_root).as_posix()
        except ValueError:
            name = str(path.resolve())
        rows.append({"name": name, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return rows


def _stage5_namespace_snapshot(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for base in [root / "outputs/model_v1", root / "reports/model_v1"]:
        if not base.exists():
            continue
        candidates = {
            path
            for path in base.iterdir()
            if "stage5" in path.name.lower()
        }
        candidates.update(base.rglob("CURRENT_STAGE5*.json"))
        for path in sorted(candidates):
            rows.append(
                {
                    "relative_path": path.relative_to(root).as_posix(),
                    "kind": "directory" if path.is_dir() else "file",
                    "sha256": sha256_file(path) if path.is_file() else None,
                }
            )
    return rows


def _save_scenario(
    outdir: Path,
    result: ScenarioCertificationResult,
    problem: CandidateProblem,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    plan = result.selected_frame.copy()
    try:
        from mediroad.stage4_2.bundles import assign_bundles

        plan, bundle_info = assign_bundles(
            plan,
            problem.bundles,
            minimum_per_bundle=1,
            time_limit_sec=120,
        )
        atomic_write_json(outdir / "bundle_assignment.json", bundle_info)
    except Exception as exc:
        atomic_write_json(outdir / "bundle_assignment_skipped.json", {"reason": repr(exc)})
    atomic_write_csv(outdir / f"plan__{result.scenario}.csv", plan)
    atomic_write_json(outdir / f"metrics__{result.scenario}.json", result.metrics)
    compact_stages: list[dict[str, Any]] = []
    for stage in result.stages:
        row = asdict(stage)
        # The complete y/q/deviation vector is deterministic from selected x and
        # the model contract. Omitting it prevents multi-megabyte JSON files.
        row.pop("solution", None)
        compact_stages.append(row)
    atomic_write_json(
        outdir / f"stages__{result.scenario}.json",
        compact_stages,
    )


def _stage_rows(results: list[ScenarioCertificationResult]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scenario in results:
        for stage in scenario.stages:
            direct = stage.details.get("direct", {})
            rows.append(
                {
                    "scenario": stage.scenario,
                    "candidate_set": stage.candidate_set,
                    "stage_index": stage.stage_index,
                    "objective_name": stage.objective_name,
                    "sense": stage.sense,
                    "status": stage.status,
                    "certificate": stage.certificate,
                    "certified": stage.certified,
                    "objective_value": stage.objective_value,
                    "best_bound": stage.best_bound,
                    "reported_relative_gap": stage.relative_gap,
                    "direct_relative_gap": direct.get("relative_gap"),
                    "wall_time_sec": stage.wall_time_sec,
                    "mip_node_count": stage.mip_node_count,
                    "seed_source": stage.seed_source,
                    "oracle_rounds": stage.oracle_rounds,
                    "model_rows": stage.model_rows,
                    "model_cols": stage.model_cols,
                    "model_nnz": stage.model_nnz,
                    "removed_redundant_or_rows": stage.details.get("model", {}).get(
                        "removed_redundant_or_rows"
                    ),
                    "peak_rss_mb": direct.get("options", {}).get("peak_rss_mb"),
                    "retained_floor_objective": stage.retained_floor.objective if stage.retained_floor else None,
                    "retained_floor_sense": stage.retained_floor.sense if stage.retained_floor else None,
                    "retained_floor_value": stage.retained_floor.value if stage.retained_floor else None,
                    "backend_message": stage.backend_message,
                }
            )
    return pd.DataFrame(rows)


def _metrics_rows(results: list[ScenarioCertificationResult]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for result in results:
        row = {k: v for k, v in result.metrics.items() if not isinstance(v, dict)}
        rows.append(row)
    return pd.DataFrame(rows)


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def _candidate_comparisons(
    all_results: dict[str, list[ScenarioCertificationResult]],
    cfg: dict[str, Any],
) -> pd.DataFrame:
    reference_set = str(cfg.get("candidate_expansion", {}).get("reference_candidate_set", "top3"))
    if reference_set not in all_results:
        return pd.DataFrame()
    ref = {result.scenario: result for result in all_results[reference_set]}
    rows: list[dict[str, Any]] = []
    for candidate_set, results in all_results.items():
        if candidate_set == reference_set:
            continue
        by_scenario = {result.scenario: result for result in results}
        for scenario in ["efficiency", "balanced"]:
            if scenario not in ref or scenario not in by_scenario:
                continue
            a, b = ref[scenario], by_scenario[scenario]
            a_clusters = set(a.selected_frame["cluster_id"].fillna(a.selected_frame["venue_id"]).astype(str))
            b_clusters = set(b.selected_frame["cluster_id"].fillna(b.selected_frame["venue_id"]).astype(str))
            a_admin = set(a.selected_frame["admin_code"].astype(str))
            b_admin = set(b.selected_frame["admin_code"].astype(str))
            rows.append(
                {
                    "reference_candidate_set": reference_set,
                    "candidate_set": candidate_set,
                    "scenario": scenario,
                    "reference_certified": a.certified,
                    "alternative_certified": b.certified,
                    "reference_unique_elderly": a.metrics["unique_elderly_population"],
                    "alternative_unique_elderly": b.metrics["unique_elderly_population"],
                    "unique_elderly_fraction_change": (
                        b.metrics["unique_elderly_population"] / a.metrics["unique_elderly_population"] - 1.0
                    ),
                    "reference_need_weighted": a.metrics["need_weighted_population"],
                    "alternative_need_weighted": b.metrics["need_weighted_population"],
                    "need_weighted_fraction_change": (
                        b.metrics["need_weighted_population"] / a.metrics["need_weighted_population"] - 1.0
                    ),
                    "reference_high_need": a.metrics["high_need_population"],
                    "alternative_high_need": b.metrics["high_need_population"],
                    "high_need_fraction_change": (
                        b.metrics["high_need_population"] / max(1e-12, a.metrics["high_need_population"]) - 1.0
                    ),
                    "reference_min_sigungu": a.metrics["min_sigungu_coverage_ratio"],
                    "alternative_min_sigungu": b.metrics["min_sigungu_coverage_ratio"],
                    "min_sigungu_change": (
                        b.metrics["min_sigungu_coverage_ratio"] - a.metrics["min_sigungu_coverage_ratio"]
                    ),
                    "venue_jaccard": _jaccard(set(a.selected_venue_ids), set(b.selected_venue_ids)),
                    "cluster_jaccard": _jaccard(a_clusters, b_clusters),
                    "admin_jaccard": _jaccard(a_admin, b_admin),
                }
            )
    return pd.DataFrame(rows)


def _promotion_decisions(comparison: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    if comparison.empty:
        return pd.DataFrame()
    rules = cfg.get("candidate_expansion", {})
    rows: list[dict[str, Any]] = []
    for candidate_set, group in comparison.groupby("candidate_set"):
        certified = bool(group["reference_certified"].all() and group["alternative_certified"].all())
        no_harm = bool(
            (group["unique_elderly_fraction_change"] >= -float(rules.get("max_unique_coverage_loss_fraction", 0.01))).all()
            and (group["need_weighted_fraction_change"] >= -float(rules.get("max_need_weighted_loss_fraction", 0.01))).all()
            and (group["high_need_fraction_change"] >= -float(rules.get("max_high_need_loss_fraction", 0.02))).all()
            and (group["min_sigungu_change"] >= -float(rules.get("max_min_sigungu_coverage_loss", 0.02))).all()
        )
        stable = bool(
            group.loc[group["scenario"].eq("balanced"), "cluster_jaccard"].min()
            >= float(rules.get("min_coverage_cluster_jaccard", 0.50))
        )
        promotion = certified and no_harm and stable
        rows.append(
            {
                "candidate_set": candidate_set,
                "both_solvers_certified": certified,
                "no_harm_pass": no_harm,
                "coverage_cluster_stability_pass": stable,
                "promotion_allowed": promotion,
                "decision": (
                    "PROMOTE_CERTIFIED_EXPANDED_SET"
                    if promotion
                    else "INCONCLUSIVE_NO_PROMOTION"
                    if not certified
                    else "DO_NOT_PROMOTE_POLICY_HARM_OR_INSTABILITY"
                ),
            }
        )
    return pd.DataFrame(rows)


def _report(
    metadata: dict[str, Any],
    metrics: pd.DataFrame,
    telemetry: pd.DataFrame,
    decisions: pd.DataFrame,
    quality: dict[str, Any],
) -> str:
    lines = [
        "# MEDIROAD Stage 4.2C Certification Hardening Report",
        "",
        f"- Run ID: `{metadata['run_id']}`",
        f"- Decision: `{quality['decision']}`",
        f"- highspy: `{metadata['environment']['highspy_version']}`",
        f"- HiGHS threads: `{metadata['hardware']['highs_threads']}`",
        f"- GPU seed search backend: `{metadata['hardware']['gpu_seed_backend']}`",
        "",
        "## Scenario summary",
        "",
    ]
    if metrics.empty:
        lines.append("No scenario result was produced.")
    else:
        lines.append(metrics.to_markdown(index=False))
    lines.extend(["", "## Stage certification", ""])
    if telemetry.empty:
        lines.append("No stage telemetry was produced.")
    else:
        cols = [
            "candidate_set",
            "scenario",
            "stage_index",
            "objective_name",
            "certificate",
            "certified",
            "objective_value",
            "direct_relative_gap",
            "reported_relative_gap",
            "wall_time_sec",
            "model_rows",
            "model_cols",
            "model_nnz",
            "seed_source",
            "oracle_rounds",
        ]
        lines.append(telemetry[[c for c in cols if c in telemetry]].to_markdown(index=False))
    if not decisions.empty:
        lines.extend(["", "## Candidate expansion decision", "", decisions.to_markdown(index=False)])
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The 0.5% certification threshold, lexicographic retention rates, candidate universes, and Stage 1–3 outputs were not relaxed.",
            "The RTX 3080 Ti is used only for optional incumbent single-swap evaluation when CuPy is available. HiGHS MIP proof remains CPU-bound.",
            "Stage 5 dates are not generated by this package.",
            "",
        ]
    )
    return "\n".join(lines)


def run_certification(
    *,
    project_root: Path,
    stage42_config_path: Path,
    certification_config_path: Path,
) -> dict[str, Any]:
    set_thread_environment()
    project_root = project_root.resolve()
    stage42_config_path = _path(project_root, stage42_config_path).resolve()
    certification_config_path = _path(project_root, certification_config_path).resolve()
    base_config = load_base_stage42_config(stage42_config_path)
    cert_config = load_yaml(certification_config_path)
    validate_certification_config(cert_config)
    hardware = resolve_hardware(cert_config)
    backend = HighsBackend(require_minimum_version=True)

    output_root = _path(project_root, cert_config["paths"]["output_root"])
    report_root = _path(project_root, cert_config["paths"]["report_root"])
    lock_path = _path(project_root, cert_config["paths"]["lock"])
    identifier = run_id("stage4_2c_certification")
    run_root = output_root / "runs" / identifier
    run_report_root = report_root / "runs" / identifier
    run_root.mkdir(parents=True, exist_ok=False)
    run_report_root.mkdir(parents=True, exist_ok=False)
    (run_root / ".RUNNING").write_text(utc_now_iso() + "\n", encoding="utf-8")
    parent_before = _pointer_snapshot(project_root)
    stage5_before = _stage5_namespace_snapshot(project_root)
    source_before = _source_inventory(project_root)
    metadata: dict[str, Any] = {
        "run_id": identifier,
        "started_utc": utc_now_iso(),
        "version": __version__,
        "project_root": str(project_root),
        "stage42_config": str(stage42_config_path),
        "stage42_config_sha256": sha256_file(stage42_config_path),
        "certification_config": str(certification_config_path),
        "certification_config_sha256": sha256_file(certification_config_path),
        "parent_pointers_before": parent_before,
        "source_files": source_before,
        "stage5_namespace_before": stage5_before,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "highspy_version": backend.version,
        },
        "hardware": asdict(hardware),
    }

    all_results: dict[str, list[ScenarioCertificationResult]] = {}
    problem_summaries: list[dict[str, Any]] = []
    cache: dict[str, Any] = {}
    exception: BaseException | None = None
    try:
        with single_writer_lock(lock_path):
            candidate_sets = list(cert_config.get("candidate_expansion", {}).get("candidate_sets", ["top3"]))
            reference_set = str(cert_config.get("candidate_expansion", {}).get("reference_candidate_set", "top3"))
            if reference_set not in candidate_sets:
                candidate_sets.insert(0, reference_set)

            sets_to_run = [reference_set]
            for candidate_set in sets_to_run:
                problem = prepare_candidate_problem(project_root, candidate_set, base_config, cache=cache)
                formulation = CertificationFormulation(
                    problem.candidates,
                    problem.patterns,
                    base_config,
                    cert_config,
                    candidate_set=candidate_set,
                )
                formulation.venue_alias_map = dict(problem.venue_alias_map)
                greedy_idx, greedy_population = constrained_greedy(
                    problem.candidates,
                    problem.coverage,
                    problem.grid["elderly_population"].to_numpy(float),
                    base_config,
                )
                engine = CertificationEngine(
                    project_root=project_root,
                    formulation=formulation,
                    base_config=base_config,
                    cert_config=cert_config,
                    stage41=problem.stage41,
                    greedy_indices=greedy_idx,
                    backend=backend,
                    highs_threads=hardware.highs_threads,
                    output_dir=run_root / "candidate_sets",
                )
                efficiency = engine.solve_scenario(
                    "efficiency", efficiency_reference=None, greedy_population=greedy_population
                )
                efficiency_reference = float(efficiency.metrics["unique_elderly_population"])
                balanced = engine.solve_scenario(
                    "balanced",
                    efficiency_reference=efficiency_reference,
                    greedy_population=greedy_population,
                )
                results = [efficiency, balanced]
                if bool(cert_config.get("gate", {}).get("run_equity_even_if_eb_uncertified", True)) or (
                    efficiency.certified and balanced.certified
                ):
                    equity = engine.solve_scenario(
                        "equity",
                        efficiency_reference=efficiency_reference,
                        greedy_population=greedy_population,
                    )
                    results.append(equity)
                all_results[candidate_set] = results
                for result in results:
                    _save_scenario(run_root / "candidate_sets" / candidate_set, result, problem)
                if not problem.reduction_ledger.empty:
                    atomic_write_csv(
                        run_root / "candidate_sets" / candidate_set / "candidate_reduction_ledger.csv",
                        problem.reduction_ledger,
                    )
                base_old_rows = (
                    1
                    + problem.candidates["admin_code"].astype(str).nunique()
                    + problem.candidates["sigungu"].astype(str).nunique()
                    + int(
                        problem.candidates["cluster_id"]
                        .fillna(problem.candidates["venue_id"])
                        .astype(str)
                        .value_counts()
                        .gt(1)
                        .sum()
                    )
                    + int(problem.patterns.n_patterns)
                    + int(sum(len(x) for x in problem.patterns.pattern_coverers))
                )
                sample_model = formulation.build(
                    objective_name="total_population", sense="max", floors=[]
                )
                problem_summaries.append(
                    {
                        "candidate_set": candidate_set,
                        "candidate_rows": len(problem.candidates),
                        "source_grid_rows": len(problem.grid),
                        "compressed_patterns": problem.patterns.n_patterns,
                        "coverage_nnz": int(problem.coverage.nnz),
                        "pattern_candidate_incidences": int(sum(len(x) for x in problem.patterns.pattern_coverers)),
                        "estimated_old_base_rows_before_z_dev": int(base_old_rows),
                        "new_easy_stage_rows": int(sample_model.n_row),
                        "new_easy_stage_cols": int(sample_model.n_col),
                        "new_easy_stage_nnz": int(sample_model.A.nnz),
                        "removed_redundant_y_ge_x_rows": int(
                            sum(len(x) for x in problem.patterns.pattern_coverers)
                        ),
                        "greedy_unique_elderly_population": greedy_population,
                    }
                )

            top3_results = {r.scenario: r for r in all_results[reference_set]}
            eb_certified = bool(
                top3_results.get("efficiency")
                and top3_results["efficiency"].certified
                and top3_results.get("balanced")
                and top3_results["balanced"].certified
            )
            expansion_prerequisite = _candidate_expansion_prerequisite(
                top3_results, cert_config
            )
            if expansion_prerequisite and bool(cert_config.get("candidate_expansion", {}).get("enabled", True)):
                for candidate_set in candidate_sets:
                    if candidate_set == reference_set:
                        continue
                    problem = prepare_candidate_problem(project_root, candidate_set, base_config, cache=cache)
                    formulation = CertificationFormulation(
                        problem.candidates,
                        problem.patterns,
                        base_config,
                        cert_config,
                        candidate_set=candidate_set,
                    )
                    formulation.venue_alias_map = dict(problem.venue_alias_map)
                    greedy_idx, greedy_population = constrained_greedy(
                        problem.candidates,
                        problem.coverage,
                        problem.grid["elderly_population"].to_numpy(float),
                        base_config,
                    )
                    engine = CertificationEngine(
                        project_root=project_root,
                        formulation=formulation,
                        base_config=base_config,
                        cert_config=cert_config,
                        stage41=problem.stage41,
                        greedy_indices=greedy_idx,
                        backend=backend,
                        highs_threads=hardware.highs_threads,
                        output_dir=run_root / "candidate_sets",
                    )
                    efficiency = engine.solve_scenario(
                        "efficiency", efficiency_reference=None, greedy_population=greedy_population
                    )
                    balanced = engine.solve_scenario(
                        "balanced",
                        efficiency_reference=float(efficiency.metrics["unique_elderly_population"]),
                        greedy_population=greedy_population,
                    )
                    results = [efficiency, balanced]
                    all_results[candidate_set] = results
                    for result in results:
                        _save_scenario(run_root / "candidate_sets" / candidate_set, result, problem)
                    problem_summaries.append(
                        {
                            "candidate_set": candidate_set,
                            "candidate_rows": len(problem.candidates),
                            "source_grid_rows": len(problem.grid),
                            "compressed_patterns": problem.patterns.n_patterns,
                            "coverage_nnz": int(problem.coverage.nnz),
                            "pattern_candidate_incidences": int(sum(len(x) for x in problem.patterns.pattern_coverers)),
                            "greedy_unique_elderly_population": greedy_population,
                        }
                    )

            flat_results = [result for results in all_results.values() for result in results]
            metrics = _metrics_rows(flat_results)
            telemetry = _stage_rows(flat_results)
            comparison = _candidate_comparisons(all_results, cert_config)
            decisions = _promotion_decisions(comparison, cert_config)
            atomic_write_csv(run_root / "scenario_metrics.csv", metrics)
            atomic_write_csv(run_root / "solver_stage_telemetry.csv", telemetry)
            atomic_write_csv(run_root / "candidate_problem_summary.csv", pd.DataFrame(problem_summaries))
            if not comparison.empty:
                atomic_write_csv(run_root / "candidate_comparison.csv", comparison)
            if not decisions.empty:
                atomic_write_csv(run_root / "candidate_decision.csv", decisions)

            top3 = {result.scenario: result for result in all_results[reference_set]}
            requirements = cert_config.get("gate", {})
            checks = {
                "top3_efficiency_certified": bool(top3.get("efficiency") and top3["efficiency"].certified),
                "top3_balanced_certified": bool(top3.get("balanced") and top3["balanced"].certified),
                "top3_equity_certified": bool(top3.get("equity") and top3["equity"].certified),
                "candidate_expansion_ran": bool(len(all_results) > 1),
                "candidate_expansion_certified_comparison": bool(
                    not decisions.empty and decisions["both_solvers_certified"].all()
                ),
                "stage5_namespace_created": (
                    _stage5_namespace_snapshot(project_root) != stage5_before
                    or bool(stage5_before)
                ),
                "parent_pointers_unchanged": _pointer_snapshot(project_root) == parent_before,
            }
            required_pass = (
                checks["top3_efficiency_certified"]
                and checks["top3_balanced_certified"]
                and (
                    checks["top3_equity_certified"]
                    if bool(requirements.get("require_equity_certification", True))
                    else True
                )
                and (
                    checks["candidate_expansion_certified_comparison"]
                    if bool(requirements.get("require_candidate_expansion_certification", True))
                    else True
                )
                and not checks["stage5_namespace_created"]
                and checks["parent_pointers_unchanged"]
            )
            decision = (
                "PASS_STAGE4_2C_COMPUTATIONAL_CERTIFICATION"
                if required_pass
                else "FAIL_STAGE4_2C_COMPUTATIONAL_CERTIFICATION"
            )
            quality = {
                "decision": decision,
                "passed": required_pass,
                "frozen_relative_gap": float(cert_config["gate"]["near_optimal_relative_gap_max"]),
                "checks": checks,
            }
            atomic_write_json(run_root / "quality_gate.json", quality)
            metadata["completed_utc"] = utc_now_iso()
            metadata["hardware"]["gpu_seed_backend"] = next(
                (
                    result.metrics.get("gpu_seed_backend")
                    for result in flat_results
                    if result.metrics.get("gpu_seed_backend")
                ),
                "UNKNOWN",
            )
            metadata["parent_pointers_after"] = _pointer_snapshot(project_root)
            metadata["quality_gate"] = quality
            atomic_write_json(run_root / "metadata.json", metadata)
            report = _report(metadata, metrics, telemetry, decisions, quality)
            atomic_write_text(run_report_root / "FINAL_REPORT.md", report)
            atomic_write_text(run_root / "FINAL_REPORT.md", report)
            if _pointer_snapshot(project_root) != parent_before:
                raise RuntimeError("Frozen parent pointer changed before commit")
            if _source_inventory(project_root) != source_before:
                raise RuntimeError("Tested/source code changed before commit")
            if _stage5_namespace_snapshot(project_root) != stage5_before:
                raise RuntimeError("Stage5 namespace changed before commit")
            (run_root / ".RUNNING").unlink(missing_ok=True)
            marker = ".COMMITTED" if required_pass else ".FAILED"
            atomic_write_text(run_root / marker, utc_now_iso() + "\n")
            inventory = build_inventory(run_root, exclude={"ARTIFACT_INVENTORY.csv"})
            atomic_write_csv(run_root / "ARTIFACT_INVENTORY.csv", inventory)
            verify_inventory(run_root, inventory)

            pointer = {
                "run_id": identifier,
                "decision": decision,
                "passed": required_pass,
                "run_relative_path": run_root.relative_to(project_root).as_posix(),
                "metadata_relative_path": (run_root / "metadata.json").relative_to(project_root).as_posix(),
                "metadata_sha256": sha256_file(run_root / "metadata.json"),
                "inventory_relative_path": (run_root / "ARTIFACT_INVENTORY.csv").relative_to(project_root).as_posix(),
                "inventory_sha256": sha256_file(run_root / "ARTIFACT_INVENTORY.csv"),
                "report_relative_path": (run_report_root / "FINAL_REPORT.md").relative_to(project_root).as_posix(),
                "report_sha256": sha256_file(run_report_root / "FINAL_REPORT.md"),
                "stage5_started": False,
            }
            if required_pass:
                atomic_write_json(output_root / "CURRENT_STAGE4_2_CERTIFICATION_RUN.json", pointer)
            else:
                atomic_write_json(output_root / f"FAILED_STAGE4_2_CERTIFICATION_RUN__{identifier}.json", pointer)
            return {
                "run_root": str(run_root),
                "report_root": str(run_report_root),
                "decision": decision,
                "passed": required_pass,
                "quality_gate": quality,
            }
    except BaseException as exc:
        exception = exc
        (run_root / ".RUNNING").unlink(missing_ok=True)
        atomic_write_text(run_root / ".FAILED", utc_now_iso() + "\n")
        atomic_write_json(
            run_root / "UNHANDLED_EXCEPTION.json",
            {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        try:
            inventory = build_inventory(run_root, exclude={"ARTIFACT_INVENTORY.csv"})
            atomic_write_csv(run_root / "ARTIFACT_INVENTORY.csv", inventory)
        except Exception:
            pass
        raise
    finally:
        if exception is not None:
            sys.stderr.write(f"Stage4.2C failed: {exception}\n")
