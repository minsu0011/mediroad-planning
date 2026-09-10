from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from .baselines import build_baselines
from .bundle_assignment import assign_bundles
from .candidates import canonicalize_interface, reduce_candidates
from .config import load_config
from .discovery import discover_stage3_paths
from .frontier import run_visit_frontier
from .matrices import align_coverage, load_coverage_matrix
from .metrics import compare_solutions, enrich_solution_metrics, selection_frequency, solution_row
from .network_validation import run_network_catchment_validation
from .optimizer import solve_primary_scenarios
from .preflight import FROZEN_CONTRACTS, run_preflight
from .reporting import build_markdown_report, create_figures, write_quality_gate, write_solution_artifacts
from .types import CoverageMatrix, OptimizationInput, QualityCheck, ScenarioSolution
from .utils import (
    atomic_write_csv,
    atomic_write_json,
    atomic_write_parquet,
    atomic_write_text,
    ensure_relative_to,
    inventory_files,
    read_table,
    sha256_file,
    single_writer_lock,
    stable_json_dumps,
    utc_now_iso,
)


def _canonicalize_grid(stage3, config: dict[str, Any]) -> pd.DataFrame:
    aliases = config["column_aliases"]
    raw = read_table(stage3.grid_policy)
    mapping = {
        "grid_id": next(c for c in aliases["grid_id"] if c in raw.columns),
        "elderly65_population": next(c for c in aliases["elderly_population"] if c in raw.columns),
        "stage1_need_score": next(c for c in aliases["need_score"] if c in raw.columns),
    }
    admin_col = next((c for c in aliases["admin_code"] if c in raw.columns), None)
    sigungu_col = next((c for c in aliases["sigungu"] if c in raw.columns), None)
    if admin_col is None or sigungu_col is None:
        raise KeyError("Grid policy table must contain admin and sigungu identifiers")
    mapping["admin_code"] = admin_col
    mapping["sigungu"] = sigungu_col
    out = raw.rename(columns={source: target for target, source in mapping.items()})[
        ["grid_id", "elderly65_population", "stage1_need_score", "admin_code", "sigungu"]
    ].copy()
    out["grid_id"] = out["grid_id"].astype(str)
    out["admin_code"] = out["admin_code"].astype(str)
    out["sigungu"] = out["sigungu"].astype(str)
    out["elderly65_population"] = pd.to_numeric(out["elderly65_population"], errors="raise")
    out["stage1_need_score"] = pd.to_numeric(out["stage1_need_score"], errors="raise")
    if out["grid_id"].duplicated().any():
        raise ValueError("Grid policy grid_id is not unique")
    return out.reset_index(drop=True)


def _attach_admin_need(candidates: pd.DataFrame, grids: pd.DataFrame) -> pd.DataFrame:
    out = candidates.copy()
    admin_need = grids.groupby("admin_code", as_index=True)["stage1_need_score"].mean()
    if "structural_need" not in out:
        out["structural_need"] = out["admin_code"].map(admin_need)
    else:
        out["structural_need"] = pd.to_numeric(out["structural_need"], errors="coerce").fillna(
            out["admin_code"].map(admin_need)
        )
    return out


def _snapshot_frozen(package_root: Path) -> dict[str, str]:
    return {
        relative: sha256_file(package_root / relative)
        for relative in FROZEN_CONTRACTS
        if (package_root / relative).exists()
    }


def _stage4_code_contract(package_root: Path) -> dict[str, str]:
    relatives = [
        "configs/model_v1/stage4.yaml",
        "requirements_stage4.txt",
        "12_scripts/v6/run_model_v1_stage4.py",
        "run_model_v1_stage4.ps1",
        "run_model_v1_stage4.sh",
        "run_model_v1_stage4_diagnostic.ps1",
        "src/mediroad/__init__.py",
        "src/mediroad/reporting/__init__.py",
        "src/mediroad/reporting/correlation.py",
        *[
            path.relative_to(package_root).as_posix()
            for path in sorted((package_root / "src/mediroad/stage4").glob("*.py"))
        ],
        *[
            path.relative_to(package_root).as_posix()
            for path in sorted((package_root / "tests").glob("test_*.py"))
        ],
    ]
    missing = [relative for relative in relatives if not (package_root / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"Stage 4 code contract files missing: {missing}")
    return {relative: sha256_file(package_root / relative) for relative in relatives}


def _run_regression_tests(package_root: Path) -> dict[str, Any]:
    import re

    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        "tests",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(package_root / "src")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["MPLBACKEND"] = "Agg"
    started = time.time()
    result = subprocess.run(
        command,
        cwd=package_root,
        env=env,
        text=True,
        capture_output=True,
        timeout=600,
        check=False,
    )
    match = re.search(r"(\d+) passed", result.stdout)
    return {
        "command": command,
        "returncode": int(result.returncode),
        "passed": int(match.group(1)) if match else 0,
        "failed": 0 if result.returncode == 0 else 1,
        "runtime_sec": float(time.time() - started),
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def _build_optimization_input(
    candidates: pd.DataFrame,
    grids: pd.DataFrame,
    bundles: pd.DataFrame,
    coverage_matrix: CoverageMatrix,
) -> OptimizationInput:
    matrix = align_coverage(
        coverage_matrix,
        candidates["venue_id"].astype(str).to_numpy(),
        grids["grid_id"].astype(str).to_numpy(),
        binary=True,
    )
    data = OptimizationInput(
        candidates=candidates.reset_index(drop=True),
        grids=grids.reset_index(drop=True),
        coverage=matrix,
        venue_ids=candidates["venue_id"].astype(str).to_numpy(),
        grid_ids=grids["grid_id"].astype(str).to_numpy(),
        bundles=bundles.copy(),
        catchment_id=coverage_matrix.matrix_id,
        metadata=coverage_matrix.metadata,
    )
    data.validate()
    return data


def _quality_checks(
    solutions: list[ScenarioSolution],
    candidates: pd.DataFrame,
    network_payload: dict[str, Any],
    frozen_before: dict[str, str],
    frozen_after: dict[str, str],
    visit_count: int,
    stage5_started: bool,
    official: bool,
    config: dict[str, Any],
    interface: pd.DataFrame,
    baselines: list[ScenarioSolution],
) -> list[QualityCheck]:
    checks: list[QualityCheck] = []
    checks.append(
        QualityCheck(
            "frozen_inputs_unchanged",
            frozen_before == frozen_after,
            "HARD",
            frozen_after,
            frozen_before,
            "Stage 1-3 frozen inputs must not change",
        )
    )
    if official:
        checks.append(
            QualityCheck(
                "network_catchment_validation_pass",
                network_payload.get("status") == "PASS",
                "HARD",
                network_payload.get("status"),
                "PASS",
                "The requested pre-Stage 4 network sensitivity must complete",
            )
        )
    else:
        checks.append(
            QualityCheck(
                "network_catchment_validation_diagnostic",
                network_payload.get("status") in {"PASS", "SKIPPED_DISABLED", "SKIPPED_DIAGNOSTIC"},
                "ADVISORY",
                network_payload.get("status"),
                "PASS or explicitly skipped in diagnostic mode",
                "Network sensitivity status",
            )
        )
    for solution in solutions:
        key = f"{solution.scenario}::{solution.catchment_id}"
        admin_max = int(solution.selected["admin_code"].astype(str).value_counts().max())
        sigungu_max = int(solution.selected["sigungu"].astype(str).value_counts().max())
        bundle_count = (
            0
            if solution.bundle_assignments is None
            else int(solution.bundle_assignments["bundle_id"].nunique())
        )
        checks.extend(
            [
                QualityCheck(
                    f"solver_returned_solution::{key}",
                    solution.status in {"OPTIMAL", "FEASIBLE", "GREEDY_DIAGNOSTIC"},
                    "HARD",
                    solution.status,
                    "OPTIMAL or FEASIBLE",
                    "Official CP-SAT stages must return a usable incumbent",
                ),
                QualityCheck(
                    f"selected_exactly_{visit_count}::{key}",
                    len(solution.selected) == visit_count,
                    "HARD",
                    len(solution.selected),
                    visit_count,
                    "Each Stage 4 scenario selects exactly the requested visits",
                ),
                QualityCheck(
                    f"selected_venue_unique::{key}",
                    not solution.selected["venue_id"].astype(str).duplicated().any(),
                    "HARD",
                    int(solution.selected["venue_id"].astype(str).duplicated().sum()),
                    0,
                    "No physical venue is selected twice",
                ),
                QualityCheck(
                    f"selected_cluster_unique::{key}",
                    not solution.selected["cluster_id"].astype(str).duplicated().any(),
                    "HARD",
                    int(solution.selected["cluster_id"].astype(str).duplicated().sum()),
                    0,
                    "At most one venue per coverage-equivalent cluster",
                ),
                QualityCheck(
                    f"selected_admin_cap::{key}",
                    admin_max <= int(config["candidate_reduction"]["max_visits_per_admin"]),
                    "HARD",
                    admin_max,
                    int(config["candidate_reduction"]["max_visits_per_admin"]),
                    "Selected visits respect the frozen per-admin cap",
                ),
                QualityCheck(
                    f"selected_sigungu_cap::{key}",
                    sigungu_max <= int(config["candidate_reduction"]["max_visits_per_sigungu"]),
                    "HARD",
                    sigungu_max,
                    int(config["candidate_reduction"]["max_visits_per_sigungu"]),
                    "Selected visits respect the frozen per-sigungu cap",
                ),
                QualityCheck(
                    f"unique_not_above_naive::{key}",
                    float(solution.metrics.get("unique_elderly_population", 0.0))
                    <= float(solution.metrics.get("naive_sum_elderly_population", 0.0)) + 1e-9,
                    "HARD",
                    solution.metrics.get("unique_elderly_population"),
                    f"<= {solution.metrics.get('naive_sum_elderly_population')}",
                    "Unique beneficiary coverage cannot exceed naive summed coverage",
                ),
                QualityCheck(
                    f"bundle_assignment_complete::{key}",
                    solution.bundle_assignments is not None
                    and len(solution.bundle_assignments) == visit_count
                    and solution.bundle_assignments["bundle_id"].notna().all(),
                    "HARD",
                    0 if solution.bundle_assignments is None else len(solution.bundle_assignments),
                    visit_count,
                    "Every provisional visit receives one frozen Stage 2A bundle",
                ),
                QualityCheck(
                    f"bundle_universe_represented::{key}",
                    bundle_count == 5,
                    "HARD",
                    bundle_count,
                    5,
                    "Every provisional scenario represents the five frozen Stage 2A bundles",
                ),
            ]
        )
    checks.append(
        QualityCheck(
            "stage5_not_started",
            not stage5_started,
            "HARD",
            stage5_started,
            False,
            "Exact month/date scheduling remains outside Stage 4",
        )
    )
    temporal_nonnull = int(
        interface[["recommended_month", "recommended_date", "fallback_date"]]
        .notna()
        .sum()
        .sum()
    )
    checks.extend(
        [
            QualityCheck(
                "stage4_interface_temporal_fields_null",
                temporal_nonnull == 0,
                "HARD",
                temporal_nonnull,
                0,
                "Stage 4 must not assign month/date fields",
            ),
            QualityCheck(
                "stage4_interface_stage5_false",
                bool(interface["stage5_started"].eq(False).all()),
                "HARD",
                int(interface["stage5_started"].astype(bool).sum()),
                0,
                "Stage 5 remains unstarted for every provisional visit",
            ),
        ]
    )
    constrained_baseline = next(
        (b for b in baselines if b.scenario == "BASELINE_CONSTRAINED_GREEDY_EFFICIENCY"),
        None,
    )
    primary_efficiency = next(
        (
            s
            for s in solutions
            if s.scenario == "efficiency" and s.catchment_id == "hard_5000m"
        ),
        None,
    )
    if constrained_baseline is not None and primary_efficiency is not None:
        observed = float(primary_efficiency.metrics["unique_elderly_population"])
        baseline_value = float(constrained_baseline.metrics["unique_elderly_population"])
        checks.append(
            QualityCheck(
                "primary_efficiency_vs_constrained_greedy_retention",
                observed + 1e-9 >= 0.99 * baseline_value,
                "HARD",
                observed / baseline_value if baseline_value else 0.0,
                ">=0.99",
                "Primary efficiency must retain at least 99% of the constraint-matched greedy baseline",
            )
        )
    # Catchment sensitivity remains an advisory, not a reason to tune Stage 3 retrospectively.
    if isinstance(network_payload.get("summary"), pd.DataFrame) and not network_payload["summary"].empty:
        best_retention = float(network_payload["summary"].iloc[0]["admin_best_venue_retention"])
        checks.append(
            QualityCheck(
                "network_vs_euclidean_best_venue_stability",
                best_retention >= 0.50,
                "ADVISORY",
                best_retention,
                ">=0.50",
                "Low retention means exact physical venue choice is catchment-sensitive; select robust cluster/fallbacks",
            )
        )
    return checks


def _field_validation_queue(
    frequency: pd.DataFrame,
    candidates: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    if frequency.empty:
        return pd.DataFrame()
    threshold = float(config["robustness"]["selection_frequency_core"])
    min_n = int(config["robustness"]["field_validation_queue_min"])
    max_n = int(config["robustness"]["field_validation_queue_max"])
    queue = frequency[frequency["selection_frequency"].ge(threshold)].copy()
    if len(queue) < min_n:
        queue = frequency.head(min_n).copy()
    queue = queue.head(max_n).copy()
    meta_cols = [
        c for c in [
            "venue_id", "venue_name", "venue_type", "admin_code", "sigungu", "cluster_id",
            "field_validation_required", "fallback_venue_1", "fallback_venue_2",
            "structural_need", "raw_elderly_exposure", "need_weighted_exposure",
        ] if c in candidates
    ]
    queue = queue.drop(columns=[c for c in meta_cols if c != "venue_id" and c in queue], errors="ignore")
    queue = queue.merge(candidates[meta_cols].drop_duplicates("venue_id"), on="venue_id", how="left")
    queue["validation_priority_reason"] = np.where(
        queue["selection_frequency"].ge(threshold),
        "robust_core_across_policy_or_catchment_scenarios",
        "highest_remaining_selection_frequency_to_reach_minimum_queue",
    )
    queue["operational_status"] = "UNVERIFIED"
    return queue


def _stage4_to_stage5_interface(
    primary_solution: ScenarioSolution,
    run_id: str,
) -> pd.DataFrame:
    selected = primary_solution.bundle_assignments.copy()
    selected = selected.reset_index(drop=True)
    selected.insert(0, "provisional_visit_id", [f"P{index:02d}" for index in range(1, len(selected) + 1)])
    selected.insert(0, "stage4_run_id", run_id)
    selected["plan_status"] = "PROVISIONAL_REQUIRES_FIELD_VALIDATION_AND_STAGE5_SCHEDULING"
    selected["recommended_month"] = pd.NA
    selected["recommended_date"] = pd.NA
    selected["fallback_date"] = pd.NA
    selected["stage5_started"] = False
    selected["temporal_evidence_status"] = selected.get(
        "temporal_release_type", "NO_STRONG_PREFERENCE"
    )
    return selected


def run_stage4(
    package_root: Path,
    *,
    config_path: Path | None = None,
    mode: str = "official",
    solver_name: str | None = None,
    force: bool = False,
    skip_network_validation: bool = False,
    run_frontier: bool = True,
) -> Path:
    package_root = package_root.resolve()
    config = load_config(config_path or package_root / "configs/model_v1/stage4.yaml")
    if solver_name is not None:
        config["runtime"]["solver"] = solver_name
    official = mode == "official"
    if mode not in {"official", "diagnostic"}:
        raise ValueError("mode must be official or diagnostic")
    if skip_network_validation:
        config["network_validation"]["enabled"] = False
        if official and config["network_validation"]["required_for_official"]:
            raise ValueError("Official Stage 4 cannot skip the requested network catchment validation")

    started_at = utc_now_iso()
    run_seed = stable_json_dumps({"started_at": started_at, "config": config})
    import hashlib

    run_id = f"stage4_{started_at.replace('-', '').replace(':', '')}_{hashlib.sha256(run_seed.encode()).hexdigest()[:12]}"
    stage4_root = package_root / config["paths"]["stage4_root"]
    runs_root = stage4_root / "runs"
    run_root = runs_root / run_id
    if run_root.exists():
        if not force:
            raise FileExistsError(f"Stage 4 run directory exists: {run_root}")
        shutil.rmtree(run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    lock_path = package_root / config["paths"]["lock"]
    frozen_before = _snapshot_frozen(package_root)
    code_before = _stage4_code_contract(package_root)
    pipeline_start = time.time()

    with single_writer_lock(lock_path):
        audit_dir = run_root / "00_preflight"
        audit_dir.mkdir(parents=True, exist_ok=True)
        test_result = (
            _run_regression_tests(package_root)
            if official
            else {
                "command": [],
                "returncode": 0,
                "passed": 0,
                "failed": 0,
                "runtime_sec": 0.0,
                "stdout": "Diagnostic mode: full regression suite not rerun.",
                "stderr": "",
            }
        )
        atomic_write_text(audit_dir / "regression_test_stdout.txt", test_result["stdout"])
        atomic_write_text(audit_dir / "regression_test_stderr.txt", test_result["stderr"])
        atomic_write_json(
            audit_dir / "regression_test_execution.json",
            {k: v for k, v in test_result.items() if k not in {"stdout", "stderr"}},
        )
        if official and test_result["returncode"] != 0:
            raise RuntimeError(
                "Stage 4 full regression suite failed before official computation. "
                f"See {audit_dir / 'regression_test_stdout.txt'}"
            )
        stage3, preflight_checks, preflight_meta = run_preflight(package_root, config)
        venue_all, bundle_values = canonicalize_interface(stage3, config)
        grids = _canonicalize_grid(stage3, config)
        venue_all = _attach_admin_need(venue_all, grids)
        candidates, excluded = reduce_candidates(venue_all, config)
        candidates = candidates.reset_index(drop=True)
        bundle_values = bundle_values[bundle_values["venue_id"].isin(candidates["venue_id"])].copy()

        atomic_write_csv(pd.DataFrame([c.as_dict() for c in preflight_checks]), audit_dir / "preflight_quality_checks.csv")
        atomic_write_csv(candidates, audit_dir / "stage4_candidate_set.csv")
        atomic_write_csv(excluded, audit_dir / "stage4_excluded_candidates.csv")
        atomic_write_csv(grids, audit_dir / "stage4_grid_policy.csv")
        atomic_write_csv(
            pd.DataFrame(
                [{"relative_path": path, "sha256": digest} for path, digest in code_before.items()]
            ),
            audit_dir / "stage4_code_contract.csv",
        )

        primary_matrix_id = str(config["network_validation"]["euclidean_reference_matrix_id"])
        primary_coverage = load_coverage_matrix(stage3, primary_matrix_id, config["column_aliases"])

        network_dir = run_root / "01_network_catchment_validation"
        if config["network_validation"]["enabled"]:
            network_payload = run_network_catchment_validation(
                package_root,
                stage3,
                candidates,
                grids,
                primary_coverage,
                config,
                network_dir,
            )
        else:
            network_payload = {
                "status": "SKIPPED_DIAGNOSTIC" if not official else "SKIPPED_DISABLED",
                "best_network_matrix_id": None,
                "matrices": {},
                "summary": pd.DataFrame(),
            }

        matrix_objects: dict[str, CoverageMatrix] = {primary_matrix_id: primary_coverage}
        for matrix_id in config["optimization"]["catchment_scenarios"]:
            if matrix_id in matrix_objects:
                continue
            try:
                matrix_objects[matrix_id] = load_coverage_matrix(stage3, matrix_id, config["column_aliases"])
            except (FileNotFoundError, KeyError):
                # Missing robustness matrix is advisory; the primary hard5 matrix remains mandatory.
                continue
        if config["optimization"]["include_best_network_catchment"]:
            best_network = network_payload.get("best_network_matrix_id")
            if best_network and best_network in network_payload.get("matrices", {}):
                matrix_objects[best_network] = network_payload["matrices"][best_network]

        visit_count = int(config["optimization"]["visit_count"])
        solver = str(config["runtime"]["solver"])
        all_solutions: list[ScenarioSolution] = []
        primary_data: OptimizationInput | None = None
        primary_solutions: list[ScenarioSolution] = []
        for matrix_id, coverage in matrix_objects.items():
            data = _build_optimization_input(candidates, grids, bundle_values, coverage)
            if matrix_id == primary_matrix_id:
                primary_data = data
            # Official run solves all policy scenarios for the primary and best network comparator.
            # Other catchments solve balanced only to control CPU time.
            if matrix_id == primary_matrix_id or matrix_id == network_payload.get("best_network_matrix_id"):
                solutions = solve_primary_scenarios(data, config, visit_count, solver_name=solver)
            else:
                efficiency = solve_primary_scenarios(data, {**config, "optimization": {**config["optimization"], "scenarios": ["efficiency", "balanced"]}}, visit_count, solver_name=solver)
                solutions = efficiency
            for solution in solutions:
                solution.bundle_assignments = assign_bundles(
                    solution.selected,
                    bundle_values,
                    config,
                    solver_name=solver,
                )
                solution.metrics = enrich_solution_metrics(solution, data)
            all_solutions.extend(solutions)
            if matrix_id == primary_matrix_id:
                primary_solutions = solutions

        if primary_data is None:
            raise RuntimeError("Primary Stage 4 optimization input was not constructed")

        solution_dir = run_root / "02_optimization"
        scenario_metrics = write_solution_artifacts(all_solutions, solution_dir)

        baselines = build_baselines(primary_data, config, visit_count)
        for baseline in baselines:
            baseline.bundle_assignments = assign_bundles(
                baseline.selected,
                bundle_values,
                config,
                solver_name="greedy" if solver != "cp-sat" else solver,
            )
            baseline.metrics = enrich_solution_metrics(baseline, primary_data)
        baseline_dir = run_root / "03_baselines"
        baseline_metrics = write_solution_artifacts(baselines, baseline_dir)

        comparisons = []
        for index, a in enumerate(all_solutions):
            for b in all_solutions[index + 1 :]:
                comparisons.append(compare_solutions(a, b))
        robust_dir = run_root / "04_robustness"
        robust_dir.mkdir(parents=True, exist_ok=True)
        if comparisons:
            atomic_write_csv(pd.DataFrame(comparisons), robust_dir / "solution_pairwise_stability.csv")
        frequency = selection_frequency(all_solutions, candidates)
        atomic_write_csv(frequency, robust_dir / "selection_frequency.csv")
        queue = _field_validation_queue(frequency, candidates, config)
        atomic_write_csv(queue, robust_dir / "robust_core_field_validation_queue.csv")

        frontier_solutions: list[ScenarioSolution] = []
        frontier_df = pd.DataFrame()
        if run_frontier:
            frontier_solutions, frontier_df = run_visit_frontier(primary_data, config, solver_name=solver)
            atomic_write_csv(frontier_df, robust_dir / "visit_count_frontier.csv")

        # Balanced primary is the provisional handoff; it is not a final annual schedule.
        balanced_primary = next(
            (s for s in primary_solutions if s.scenario == "balanced"),
            primary_solutions[0],
        )
        interface = _stage4_to_stage5_interface(balanced_primary, run_id)
        interface_dir = run_root / "05_interface"
        interface_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_parquet(interface, interface_dir / "stage4_to_stage5_interface.parquet")
        atomic_write_csv(interface, interface_dir / "stage4_to_stage5_interface.csv")

        frozen_after = _snapshot_frozen(package_root)
        code_after = _stage4_code_contract(package_root)
        checks = preflight_checks + _quality_checks(
            all_solutions,
            candidates,
            network_payload,
            frozen_before,
            frozen_after,
            visit_count,
            stage5_started=False,
            official=official,
            config=config,
            interface=interface,
            baselines=baselines,
        )
        checks.append(
            QualityCheck(
                "stage4_code_contract_unchanged",
                code_before == code_after,
                "HARD",
                code_after,
                code_before,
                "Stage 4 source/config/runner contract must not change during execution",
            )
        )
        checks.append(
            QualityCheck(
                "stage4_regression_tests_pass",
                (not official) or (
                    test_result["returncode"] == 0 and int(test_result["passed"]) > 0
                ),
                "HARD" if official else "DIAGNOSTIC",
                {
                    "passed": int(test_result["passed"]),
                    "failed": int(test_result["failed"]),
                    "returncode": int(test_result["returncode"]),
                },
                {"passed": ">0", "failed": 0, "returncode": 0},
                "Official Stage 4 requires a non-vacuous full regression run",
            )
        )
        gate_df = write_quality_gate(checks, run_root / "STAGE4_QUALITY_GATE.csv")
        hard_failed = gate_df[(gate_df["severity"] == "HARD") & (~gate_df["passed"].astype(bool))]
        if hard_failed.empty:
            status = "PASS_PROVISIONAL_STAGE4" if official else "PASS_DIAGNOSTIC_STAGE4"
        else:
            status = "FAIL_STAGE4"

        figures = create_figures(
            scenario_metrics,
            frequency,
            network_payload.get("summary", pd.DataFrame()),
            frontier_df,
            run_root / "figures",
        )
        report = build_markdown_report(
            run_id,
            status,
            preflight_meta,
            {
                "candidate_count": int(len(candidates)),
                "admin_count": int(candidates["admin_code"].nunique()),
                "sigungu_count": int(candidates["sigungu"].nunique()),
            },
            network_payload,
            scenario_metrics,
            baseline_metrics,
            frequency,
            frontier_df,
            checks,
        )
        atomic_write_text(run_root / "FINAL_REPORT.md", report)

        inventory = inventory_files(run_root, exclude_names={"STAGE4_RUN_METADATA.json", "STAGE4_ARTIFACT_INVENTORY.csv"})
        atomic_write_csv(inventory, run_root / "STAGE4_ARTIFACT_INVENTORY.csv")
        completed_at = utc_now_iso()
        metadata = {
            "version": config["version"],
            "run_id": run_id,
            "mode": mode,
            "status": status,
            "started_at": started_at,
            "completed_at": completed_at,
            "runtime_sec": float(time.time() - pipeline_start),
            "platform": platform.platform(),
            "python": sys.version,
            "solver": solver,
            "jobs": int(config["runtime"]["jobs"]),
            "tests_passed": int(test_result["passed"]),
            "tests_failed": int(test_result["failed"]),
            "tests_returncode": int(test_result["returncode"]),
            "tests_runtime_sec": float(test_result["runtime_sec"]),
            "stage3_run_root": ensure_relative_to(stage3.run_root, package_root),
            "candidate_count": int(len(candidates)),
            "grid_count": int(len(grids)),
            "bundle_count": int(bundle_values["bundle_id"].nunique()),
            "catchment_matrices": list(matrix_objects),
            "network_validation_status": network_payload.get("status"),
            "best_network_matrix_id": network_payload.get("best_network_matrix_id"),
            "hard_gate_total": int((gate_df["severity"] == "HARD").sum()),
            "hard_gate_passed": int(((gate_df["severity"] == "HARD") & gate_df["passed"].astype(bool)).sum()),
            "advisory_failed": int(((gate_df["severity"] != "HARD") & (~gate_df["passed"].astype(bool))).sum()),
            "artifact_count": int(len(inventory)),
            "artifact_inventory_sha256": sha256_file(run_root / "STAGE4_ARTIFACT_INVENTORY.csv"),
            "code_contract": code_after,
            "config_sha256": sha256_file(package_root / "configs/model_v1/stage4.yaml"),
            "frozen_before": frozen_before,
            "frozen_after": frozen_after,
            "stage1_frozen": frozen_before == frozen_after,
            "stage2a_frozen": True,
            "stage2b_frozen": True,
            "stage3_frozen": True,
            "stage4_complete": bool(official and hard_failed.empty),
            "diagnostic_complete": bool((not official) and hard_failed.empty),
            "stage4_provisional": True,
            "field_validation_required_before_final": True,
            "stage5_started": False,
            "exact_month_date_assigned": False,
            "interpretation": (
                "Provisional spatial allocation and frozen Stage 2A bundle assignment. "
                "Coverage is potential beneficiary mass, not expected patients."
            ),
        }
        atomic_write_json(run_root / "STAGE4_RUN_METADATA.json", metadata)

        if hard_failed.empty and official:
            stage4_root.mkdir(parents=True, exist_ok=True)
            report_root = package_root / config["paths"]["report_root"]
            report_root.mkdir(parents=True, exist_ok=True)
            report_alias = report_root / "CURRENT_STAGE4_GATE.md"
            atomic_write_text(report_alias, report)
            if sha256_file(report_alias) != sha256_file(run_root / "FINAL_REPORT.md"):
                raise RuntimeError("Stage 4 report alias is not byte-identical to the run report")
            atomic_write_text(stage4_root / "CURRENT.txt", f"runs/{run_id}\n")
            pointer = {
                "run_id": run_id,
                "metadata_relative_path": ensure_relative_to(
                    run_root / "STAGE4_RUN_METADATA.json", package_root
                ),
                "inventory_relative_path": ensure_relative_to(
                    run_root / "STAGE4_ARTIFACT_INVENTORY.csv", package_root
                ),
                "interface_relative_path": ensure_relative_to(
                    interface_dir / "stage4_to_stage5_interface.parquet", package_root
                ),
                "report_relative_path": ensure_relative_to(run_root / "FINAL_REPORT.md", package_root),
                "metadata_sha256": sha256_file(run_root / "STAGE4_RUN_METADATA.json"),
                "inventory_sha256": sha256_file(run_root / "STAGE4_ARTIFACT_INVENTORY.csv"),
                "interface_sha256": sha256_file(
                    interface_dir / "stage4_to_stage5_interface.parquet"
                ),
                "report_sha256": sha256_file(run_root / "FINAL_REPORT.md"),
                "stage4_complete": True,
                "stage5_started": False,
            }
            # Authoritative commit marker is written last.
            atomic_write_json(stage4_root / "CURRENT_STAGE4_RUN.json", pointer)
        else:
            raise RuntimeError(
                "Stage 4 hard gate failed:\n"
                + hard_failed[["check_id", "observed", "expected", "message"]].to_string(index=False)
            )
    return run_root
