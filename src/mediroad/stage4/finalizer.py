from __future__ import annotations

"""Stage 4.1 solver/candidate hardening and field-validation handoff.

This module deliberately writes to a new run-scoped tree.  Stage 1--3 and the
provisional Stage 4 CURRENT run are parents, never mutable work products.  The
highest release possible without field evidence and confirmed team bases is
``PASS_STAGE4_READY_FOR_FIELD_VALIDATION``; Stage 5 remains forbidden.
"""

from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from importlib import metadata as importlib_metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

try:  # Unix RSS accounting; Windows import must remain usable for audits/tests.
    import resource
except ModuleNotFoundError:  # pragma: no cover - exercised on Windows hosts
    resource = None  # type: ignore[assignment]

from .baselines import build_baselines
from .bundle_assignment import assign_bundles
from .busproxy_resolution import actual_facility_fallbacks, resolve_final_physical_venues
from .candidate_sensitivity import (
    CandidateBuildPolicy,
    CandidateSensitivityThresholds,
    build_candidate_sets,
    evaluate_candidate_sensitivity,
)
from .candidates import canonicalize_interface
from .compressed_optimizer import build_compressed_coverage_groups, solve_compressed_scenario
from .config import load_config
from .field_validation import (
    build_field_validation_queue,
    make_field_validation_form,
    robust_core_summary,
    validate_field_validation_schema,
)
from .highs_certifier import (
    constrained_greedy_efficiency_floor,
    solve_highs_scenario,
)
from .matrices import load_coverage_matrix
from .metrics import enrich_solution_metrics
from .optimizer import prepare_weights
from .pipeline import (
    _attach_admin_need,
    _build_optimization_input,
    _canonicalize_grid,
)
from .preflight import run_preflight
from .seed_stability import SeedStabilityThresholds, analyze_seed_stability
from .solver_certification import (
    NearOptimalityContract,
    certify_solver_run,
    compare_greedy_no_harm,
    normalize_solver_stages,
)
from .travel_sensitivity import (
    beneficiary_visit_location_equity_kpis,
    build_diagnostic_travel_scenarios,
    load_mobile_team_bases,
    prepare_final_travel_objective,
)
from .types import OptimizationInput, ScenarioSolution
from .utils import (
    atomic_write_csv,
    atomic_write_json,
    atomic_write_parquet,
    atomic_write_text,
    copy_atomic,
    ensure_relative_to,
    inventory_files,
    sha256_file,
    single_writer_lock,
    stable_json_dumps,
    utc_now_iso,
    ceil_fraction,
)


FINALIZATION_DECISIONS = {
    "PASS_STAGE4_SOLVER_CERTIFIED",
    "PASS_STAGE4_COMPUTATIONAL_FINAL",
    "PASS_STAGE4_READY_FOR_FIELD_VALIDATION",
    "PASS_STAGE4_OPERATIONAL_FINAL",
    "FAIL_STAGE4_FINALIZATION",
}

EXPECTED_LEXICOGRAPHIC_STAGES = {
    "efficiency": (
        ("total_population", "max"),
        ("need_weighted", "max"),
        ("min_sigungu_coverage", "max"),
        ("cost", "min"),
    ),
    "balanced": (
        ("need_weighted", "max"),
        ("min_sigungu_coverage", "max"),
        ("total_population", "max"),
        ("cost", "min"),
    ),
    "equity": (
        ("min_sigungu_coverage", "max"),
        ("high_need_population", "max"),
        ("need_weighted", "max"),
        ("cost", "min"),
    ),
}

CODE_CONTRACT_PATHS = (
    "configs/model_v1/stage4.yaml",
    "configs/model_v1/stage4_finalization.yaml",
    "requirements_stage4.txt",
    "12_scripts/v6/run_model_v1_stage4_finalization.py",
    "run_model_v1_stage4_finalization.ps1",
    "src/mediroad/__init__.py",
    "src/mediroad/stage4/__init__.py",
    "src/mediroad/stage4/types.py",
    "src/mediroad/stage4/utils.py",
    "src/mediroad/stage4/config.py",
    "src/mediroad/stage4/discovery.py",
    "src/mediroad/stage4/preflight.py",
    "src/mediroad/stage4/candidates.py",
    "src/mediroad/stage4/matrices.py",
    "src/mediroad/stage4/frontier.py",
    "src/mediroad/stage4/network_validation.py",
    "src/mediroad/stage4/reporting.py",
    "src/mediroad/stage4/pipeline.py",
    "src/mediroad/stage4/optimizer.py",
    "src/mediroad/stage4/compressed_optimizer.py",
    "src/mediroad/stage4/highs_certifier.py",
    "src/mediroad/stage4/solver_certification.py",
    "src/mediroad/stage4/candidate_sensitivity.py",
    "src/mediroad/stage4/seed_stability.py",
    "src/mediroad/stage4/bundle_assignment.py",
    "src/mediroad/stage4/baselines.py",
    "src/mediroad/stage4/metrics.py",
    "src/mediroad/stage4/field_validation.py",
    "src/mediroad/stage4/busproxy_resolution.py",
    "src/mediroad/stage4/travel_sensitivity.py",
    "src/mediroad/stage4/finalizer.py",
)


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def load_finalization_config(path: Path) -> dict[str, Any]:
    """Load the frozen Stage 4.1 contract and reject silent relaxations."""

    if not path.is_file():
        raise FileNotFoundError(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    config = dict(_require_mapping(raw, "Stage 4 finalization config"))
    if config.get("version") != "MEDIROAD_STAGE4_FINALIZATION_V1":
        raise ValueError("Unexpected Stage 4 finalization version")
    if config.get("contract_status") != (
        "post_quality_gate_and_greedy_floor_hardening_frozen_before_rerun"
    ):
        raise ValueError("Finalization contract status changed")
    if not str(config.get("contract_change_reason", "")).strip():
        raise ValueError("Post-diagnostic fail-closed disposition reason is required")
    if config.get("stage5_started") is not False:
        raise ValueError("Stage 5 must remain false")
    project = _require_mapping(config.get("project_contract"), "project_contract")
    for key in (
        "computational_final_is_not_operational_final",
        "stage1_to_stage3_frozen",
        "provisional_stage4_frozen",
        "temporal_selector_forbidden",
        "exact_month_date_forbidden",
        "potential_beneficiary_exposure_not_expected_patients",
    ):
        if project.get(key) is not True:
            raise ValueError(f"project_contract.{key} must remain true")
    if project.get("target_release") != "PASS_STAGE4_READY_FOR_FIELD_VALIDATION":
        raise ValueError("project target release changed")
    parent = _require_mapping(config.get("parent_contract"), "parent_contract")
    expected_parent_keys = {
        "stage3_current_pointer",
        "stage4_current_pointer",
        "stage4_parent_run_id",
        "stage4_parent_metadata",
        "stage4_parent_inventory",
        "stage4_parent_interface",
        "stage4_parent_report",
        "stage4_base_config",
        "direction_log",
    }
    if set(parent) != expected_parent_keys:
        raise ValueError("parent_contract key set changed")
    if parent.get("stage4_parent_run_id") != "stage4_20260819T052330Z_ec2f77ccc968":
        raise ValueError("frozen provisional Stage 4 parent run changed")
    for key in sorted(expected_parent_keys - {"stage4_parent_run_id"}):
        record = _require_mapping(parent.get(key), f"parent_contract.{key}")
        if set(record) != {"path", "sha256"}:
            raise ValueError(f"parent_contract.{key} schema changed")
        relative = str(record.get("path", ""))
        relative_path = Path(relative)
        if (
            not relative
            or relative_path.is_absolute()
            or ".." in relative_path.parts
            or relative_path == Path(".")
        ):
            raise ValueError(f"parent_contract.{key}.path must stay package-relative")
        digest = str(record.get("sha256", "")).lower()
        if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError(f"parent_contract.{key}.sha256 is invalid")
    environment = _require_mapping(config.get("runtime_environment"), "runtime_environment")
    if str(environment.get("python_major_minor")) != "3.11":
        raise ValueError("Stage 4.1 runtime must remain Python 3.11")
    observed_python = f"{sys.version_info.major}.{sys.version_info.minor}"
    if observed_python != "3.11":
        raise ValueError(
            f"Stage 4.1 must execute under Python 3.11, observed {observed_python}"
        )
    expected_packages = _require_mapping(
        environment.get("exact_package_versions"),
        "runtime_environment.exact_package_versions",
    )
    observed_packages = _dependency_versions(expected_packages)
    if observed_packages != {str(key): str(value) for key, value in expected_packages.items()}:
        raise ValueError(
            "Stage 4.1 dependency versions differ from the frozen runtime contract: "
            f"observed={observed_packages}"
        )
    runtime = _require_mapping(config.get("runtime"), "runtime")
    if int(runtime.get("jobs", 0)) != 8:
        raise ValueError("runtime.jobs must remain exactly 8")
    if int(runtime.get("highs_threads_per_task", 0)) != 4:
        raise ValueError("runtime.highs_threads_per_task must remain exactly 4")
    if int(runtime.get("frontier_retry_highs_threads", 0)) != 4:
        raise ValueError("runtime.frontier_retry_highs_threads must remain exactly 4")
    if int(runtime.get("highs_random_seed", -1)) != 42:
        raise ValueError("runtime.highs_random_seed must remain exactly 42")
    if int(runtime.get("candidate_parallel_workers", 0)) != 2:
        raise ValueError("runtime.candidate_parallel_workers must remain exactly 2")
    if int(runtime.get("frontier_parallel_workers", 0)) != 2:
        raise ValueError("runtime.frontier_parallel_workers must remain exactly 2")
    if int(runtime.get("frontier_retry_parallel_workers", 0)) != 1:
        raise ValueError(
            "runtime.frontier_retry_parallel_workers must remain exactly 1"
        )
    if float(runtime.get("memory_soft_limit_gib", 0)) != 24.0:
        raise ValueError("runtime memory soft limit must remain 24 GiB")
    if runtime.get("run_full_regression_suite") is not True:
        raise ValueError("full regression suite cannot be disabled")
    for key in (
        "diagnostic_time_per_stage_sec",
        "candidate_time_per_stage_sec",
        "final_time_per_stage_sec",
        "frontier_time_per_stage_sec",
        "frontier_retry_time_per_stage_sec",
    ):
        value = float(runtime.get(key, 0))
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"runtime.{key} must be positive")
    solver = _require_mapping(config.get("solver"), "solver")
    if list(solver.get("seeds", [])) != [11, 23, 42, 77, 101]:
        raise ValueError("solver.seeds must be exactly [11, 23, 42, 77, 101]")
    if float(solver.get("near_optimal_relative_gap_max", -1)) != 0.005:
        raise ValueError("near-optimality threshold must remain 0.5%")
    for key in (
        "require_objective_value",
        "require_best_bound",
        "require_branch_and_conflict_telemetry_for_cp_sat",
        "require_all_lexicographic_stages_certified_for_solver_certified_release",
        "greedy_no_harm_required_when_same_constraints",
        "final_efficiency_must_not_underperform_constrained_greedy",
        "high_level_release_may_be_ready_for_field_validation_when_uncertified",
        "highs_primary_requires_same_contract_recomputation",
        "compressed_cp_sat_is_diagnostic_only",
    ):
        if solver.get(key) is not True:
            raise ValueError(f"solver.{key} must remain true")
    if runtime.get("primary_solver") != "scipy_highs_lossless_group_milp":
        raise ValueError("runtime.primary_solver changed")
    if runtime.get("seed_diagnostic_solver") != "compressed_cp_sat_with_greedy_hints":
        raise ValueError("runtime.seed_diagnostic_solver changed")
    if runtime.get("independent_certifier") != "bound_based_scipy_highs_stage_telemetry":
        raise ValueError("runtime.independent_certifier changed")
    expected_budgets = {
        "diagnostic_time_per_stage_sec": 20.0,
        "candidate_time_per_stage_sec": 120.0,
        "final_time_per_stage_sec": 120.0,
        "frontier_time_per_stage_sec": 30.0,
        "frontier_retry_time_per_stage_sec": 120.0,
    }
    for key, expected in expected_budgets.items():
        if float(runtime.get(key, math.nan)) != expected:
            raise ValueError(f"runtime.{key} changed")
    frontier = _require_mapping(config.get("frontier"), "frontier")
    retry_contract = _require_mapping(
        frontier.get("no_incumbent_retry_contract"),
        "frontier.no_incumbent_retry_contract",
    )
    if retry_contract != {
        "trigger": "no_incumbent_only",
        "maximum_attempts": 2,
        "retry_is_sequential": True,
        "retry_changes_objectives_or_thresholds": False,
        "failed_run_id": "stage4_1_20260819T184615Z_b221621d4413",
        "failed_scenario": "efficiency",
        "failed_visit_count": 25,
        "failed_stage_budget_sec": 30,
    }:
        raise ValueError("frontier no-incumbent retry contract changed")
    candidate = _require_mapping(config.get("candidate_sets"), "candidate_sets")
    if list(candidate.get("strategies", [])) != [
        "top3",
        "top5",
        "coarse_pareto",
        "coverage_cluster",
    ]:
        raise ValueError("candidate-set ladder changed")
    if list(candidate.get("fallback_order", [])) != [
        "top5",
        "coarse_pareto",
        "coverage_cluster",
    ]:
        raise ValueError("candidate fallback order changed")
    if candidate.get("promotion_scenario") != "balanced":
        raise ValueError("candidate promotion scenario must remain balanced")
    if candidate.get("candidate_reduction_scope_balanced_only") is not True:
        raise ValueError("candidate reduction must remain a Balanced-only comparison")
    disposition = _require_mapping(
        candidate.get("uncertified_expansion_disposition"),
        "candidate_sets.uncertified_expansion_disposition",
    )
    if disposition != {
        "action": "retain_certified_frozen_top3",
        "promotion_allowed": False,
        "comparison_conclusive": False,
        "release_ceiling": "PASS_STAGE4_READY_FOR_FIELD_VALIDATION",
        "performance_claims_forbidden": True,
        "certified_comparator_threshold_breach_forbids_top3_retention": True,
    }:
        raise ValueError("uncertified expansion fail-closed disposition changed")
    if int(candidate.get("expected_admin_count", 0)) != 153:
        raise ValueError("candidate expected admin count must be 153")
    expected_candidate_values = {
        "max_candidates": 900,
        "coarse_pareto_tier_max": 3,
        "expected_coverage_cluster_maximum_matching": 149,
    }
    for key, expected in expected_candidate_values.items():
        if int(candidate.get(key, -1)) != expected:
            raise ValueError(f"candidate_sets.{key} changed")
    if candidate.get("require_all_admins") is not True:
        raise ValueError("candidate sets must retain all admins")
    if candidate.get("coverage_cluster_required") is not False:
        raise ValueError("infeasible coverage-cluster set cannot become mandatory")
    if candidate.get("coverage_cluster_unavailable_must_be_explicit") is not True:
        raise ValueError("coverage-cluster unavailability must remain explicit")
    expected_candidate_thresholds = {
        "top3_max_unique_coverage_loss_fraction": 0.01,
        "top3_max_need_weighted_loss_fraction": 0.01,
        "top3_max_high_need_loss_fraction": 0.02,
        "top3_max_min_sigungu_coverage_loss": 0.02,
        "min_selected_admin_jaccard": 0.50,
        "min_selected_coverage_cluster_jaccard": 0.50,
        "maximum_solver_relative_gap": 0.005,
    }
    for key, expected in expected_candidate_thresholds.items():
        if float(candidate.get(key, math.nan)) != expected:
            raise ValueError(f"candidate_sets.{key} changed")
    bundle = _require_mapping(config.get("bundle_sensitivity"), "bundle_sensitivity")
    if list(bundle.get("minimums_to_test", [])) != [0, 1]:
        raise ValueError("bundle minimum sensitivity must test 0 and 1")
    if int(bundle.get("primary_minimum_per_bundle", -1)) != 1:
        raise ValueError("primary bundle representation floor changed")
    if bundle.get("preserve_spatial_selection") is not True:
        raise ValueError("bundle sensitivity must preserve spatial anchors")
    if bundle.get("arbitrary_minimum_must_not_be_hidden") is not True:
        raise ValueError("bundle policy assumption must remain explicit")
    seed_stability = _require_mapping(config.get("seed_stability"), "seed_stability")
    expected_seed_thresholds = {
        "max_objective_relative_spread": 0.01,
        "max_unique_coverage_relative_spread": 0.01,
        "max_need_weighted_relative_spread": 0.01,
        "max_high_need_relative_spread": 0.02,
        "max_min_sigungu_coverage_absolute_spread": 0.02,
        "min_median_venue_jaccard": 0.67,
        "min_median_admin_jaccard": 0.67,
        "min_median_coverage_cluster_jaccard": 0.67,
        "min_median_bundle_jaccard": 0.67,
    }
    if set(seed_stability) != set(expected_seed_thresholds):
        raise ValueError("seed_stability threshold set changed")
    for key, expected in expected_seed_thresholds.items():
        if float(seed_stability.get(key, math.nan)) != expected:
            raise ValueError(f"seed_stability.{key} changed")
    field = _require_mapping(config.get("field_validation"), "field_validation")
    if not 30 <= int(field.get("queue_min", 0)) <= int(field.get("queue_max", 0)) <= 60:
        raise ValueError("field-validation queue bounds must be within 30--60")
    expected_field_columns = [
        "large_vehicle_access",
        "parking",
        "electricity",
        "toilet",
        "indoor_waiting",
        "heating_cooling",
        "barrier_free",
        "medical_equipment_space",
        "actual_facility_exists",
        "reservation_possible",
        "contact_verified",
        "verification_date",
    ]
    if list(field.get("expected_field_columns", [])) != expected_field_columns:
        raise ValueError("field-validation schema changed")
    if float(field.get("robust_frequency_threshold", math.nan)) != 0.67:
        raise ValueError("robust frequency threshold changed")
    busproxy = _require_mapping(config.get("busproxy"), "busproxy")
    if busproxy.get("allow_spatial_anchor") is not True:
        raise ValueError("BUSPROXY spatial-anchor contract changed")
    if busproxy.get("allow_unverified_final_physical_venue") is not False:
        raise ValueError("unverified BUSPROXY cannot become a final venue")
    if busproxy.get("unresolved_status_required") is not True:
        raise ValueError("unresolved BUSPROXY status must remain mandatory")
    if int(busproxy.get("fallback_count", -1)) != 2:
        raise ValueError("BUSPROXY fallback count changed")
    travel = _require_mapping(config.get("travel"), "travel")
    if travel.get("mode") != "unavailable":
        raise ValueError("travel must remain unavailable until confirmed team bases exist")
    if travel.get("allow_diagnostic_scenarios") is not True:
        raise ValueError("diagnostic travel scenarios must remain enabled")
    if list(travel.get("diagnostic_scenarios", [])) != [
        "cheongju_medical_center",
        "chungju_medical_center",
        "dual_base_nearest",
    ]:
        raise ValueError("diagnostic travel scenario contract changed")
    if travel.get("allow_diagnostic_promotion") is not False:
        raise ValueError("diagnostic travel cannot be promoted")
    if str(travel.get("mobile_team_bases_path")) != "configs/model_v1/mobile_team_bases.yaml":
        raise ValueError("mobile-team base input path changed")
    frontier = _require_mapping(config.get("frontier"), "frontier")
    if list(frontier.get("visit_counts", [])) != [5, 10, 15, 20, 25, 30, 40]:
        raise ValueError("visit-count frontier changed")
    if list(frontier.get("scenarios", [])) != ["efficiency", "balanced"]:
        raise ValueError("frontier scenarios changed")
    release = _require_mapping(config.get("release"), "release")
    if release.get("target") != "PASS_STAGE4_READY_FOR_FIELD_VALIDATION":
        raise ValueError("Stage 4.1 target release changed")
    if release.get("stage5_execution_forbidden") is not True:
        raise ValueError("Stage 5 execution must be forbidden")
    if release.get("operational_final_requires_verified_venues_and_confirmed_travel") is not True:
        raise ValueError("operational-final evidence requirements changed")
    if list(release.get("possible_decisions", [])) != [
        "PASS_STAGE4_SOLVER_CERTIFIED",
        "PASS_STAGE4_COMPUTATIONAL_FINAL",
        "PASS_STAGE4_READY_FOR_FIELD_VALIDATION",
        "PASS_STAGE4_OPERATIONAL_FINAL",
        "FAIL_STAGE4_FINALIZATION",
    ]:
        raise ValueError("release decision vocabulary changed")
    paths = _require_mapping(config.get("paths"), "paths")
    if paths != {
        "output_root": "outputs/model_v1/09_stage4_finalization",
        "report_root": "reports/model_v1/stage4_finalization",
        "lock": "outputs/model_v1/.stage4_finalization_writer.lock",
        "forbidden_stage5_roots": [
            "outputs/model_v1/10_stage5",
            "reports/model_v1/stage5",
        ],
    }:
        raise ValueError("Stage 4.1 output isolation paths changed")
    return config


def _dependency_versions(packages: Mapping[str, Any]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for distribution in sorted(map(str, packages)):
        try:
            versions[distribution] = importlib_metadata.version(distribution)
        except importlib_metadata.PackageNotFoundError as exc:
            raise RuntimeError(f"Required Stage 4.1 distribution is missing: {distribution}") from exc
    return versions


def _peak_rss_gib() -> float:
    """Return process peak RSS without making this module Unix-only."""

    if resource is not None:
        raw = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        # Linux/WSL reports KiB; macOS reports bytes.
        divisor = 1024.0**2 if sys.platform != "darwin" else 1024.0**3
        return raw / divisor
    if os.name == "nt":  # pragma: no cover - platform-specific audit path
        import ctypes
        from ctypes import wintypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.windll.kernel32
        psapi = ctypes.windll.psapi
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(ProcessMemoryCounters),
            wintypes.DWORD,
        ]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        handle = kernel32.GetCurrentProcess()
        ok = psapi.GetProcessMemoryInfo(
            handle, ctypes.byref(counters), counters.cb
        )
        if not ok:
            raise OSError("GetProcessMemoryInfo failed")
        return float(counters.PeakWorkingSetSize) / (1024.0**3)
    raise RuntimeError("Peak RSS accounting is unavailable on this platform")


def _sha_contract(package_root: Path, paths: Iterable[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in paths:
        path = package_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Code-contract file is missing: {relative}")
        result[relative] = sha256_file(path)
    return result


def _verify_declared_parents(package_root: Path, config: Mapping[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    contracts = _require_mapping(config.get("parent_contract"), "parent_contract")
    for contract_name, payload in contracts.items():
        if contract_name == "stage4_parent_run_id":
            continue
        record = _require_mapping(payload, f"parent_contract.{contract_name}")
        relative = str(record.get("path", ""))
        expected = str(record.get("sha256", "")).lower()
        path = (package_root / relative).resolve()
        try:
            path.relative_to(package_root.resolve())
        except ValueError as exc:
            raise RuntimeError(
                f"Declared parent path escapes the package root: {contract_name}"
            ) from exc
        actual = sha256_file(path) if path.is_file() else "MISSING"
        rows.append(
            {
                "contract_name": contract_name,
                "relative_path": relative,
                "expected_sha256": expected,
                "actual_sha256": actual,
                "size_bytes": path.stat().st_size if path.is_file() else -1,
                "passed": actual == expected,
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty or not frame["passed"].all():
        raise RuntimeError(
            "Frozen parent hash contract failed:\n"
            + frame.loc[~frame["passed"], ["contract_name", "relative_path", "expected_sha256", "actual_sha256"]].to_string(index=False)
        )
    pointer = json.loads(
        (package_root / contracts["stage4_current_pointer"]["path"]).read_text(encoding="utf-8")
    )
    if pointer.get("run_id") != contracts.get("stage4_parent_run_id"):
        raise RuntimeError("Stage 4 parent CURRENT run_id changed")
    if pointer.get("stage5_started") is not False:
        raise RuntimeError("Stage 4 parent unexpectedly reports Stage 5 started")
    return frame


def _inventory_tree(
    package_root: Path,
    pointer_relative_path: str,
    *,
    scope: str,
) -> pd.DataFrame:
    pointer_path = (package_root / pointer_relative_path).resolve()
    ensure_relative_to(pointer_path, package_root)
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    metadata_relative = str(pointer.get("metadata_relative_path", ""))
    metadata_path = (package_root / metadata_relative).resolve()
    ensure_relative_to(metadata_path, package_root)
    if not metadata_path.is_file():
        raise FileNotFoundError(f"{scope} metadata is missing: {metadata_relative}")
    if sha256_file(metadata_path) != str(pointer.get("metadata_sha256", "")).lower():
        raise RuntimeError(f"{scope} metadata SHA does not match its CURRENT pointer")

    inventory_relative = pointer.get("inventory_relative_path")
    inventory_paths_are_package_relative = not bool(inventory_relative)
    if inventory_relative:
        inventory_path = (package_root / str(inventory_relative)).resolve()
        expected_inventory_sha = str(
            pointer.get("inventory_sha256", pointer.get("artifact_inventory_sha256", ""))
        ).lower()
    else:
        # The frozen Stage 3 pointer predates inventory_relative_path.  Its
        # contract fixes the inventory name beside metadata and pins the hash
        # as artifact_inventory_sha256; no glob or first-match fallback is used.
        inventory_path = metadata_path.parent / "STAGE3_ARTIFACT_INVENTORY.csv"
        expected_inventory_sha = str(pointer.get("artifact_inventory_sha256", "")).lower()
    ensure_relative_to(inventory_path, package_root)
    if not inventory_path.is_file():
        raise FileNotFoundError(f"{scope} inventory is missing: {inventory_path}")
    if not expected_inventory_sha or sha256_file(inventory_path) != expected_inventory_sha:
        raise RuntimeError(f"{scope} inventory SHA does not match its CURRENT pointer")
    inventory = pd.read_csv(inventory_path, dtype={"relative_path": "string"})
    required = {"relative_path", "size_bytes", "sha256"}
    if not required.issubset(inventory.columns):
        raise RuntimeError(f"{scope} inventory is missing {sorted(required-set(inventory.columns))}")
    run_root = inventory_path.parent
    rows: list[dict[str, Any]] = []
    for row in inventory.itertuples(index=False):
        artifact = (
            package_root / str(row.relative_path)
            if inventory_paths_are_package_relative
            else run_root / str(row.relative_path)
        ).resolve()
        ensure_relative_to(artifact, package_root)
        actual_hash = sha256_file(artifact) if artifact.is_file() else "MISSING"
        actual_size = artifact.stat().st_size if artifact.is_file() else -1
        rows.append(
            {
                "scope": scope,
                "relative_path": ensure_relative_to(artifact, package_root),
                "expected_size_bytes": int(row.size_bytes),
                "actual_size_bytes": int(actual_size),
                "expected_sha256": str(row.sha256).lower(),
                "actual_sha256": actual_hash,
                "passed": bool(
                    actual_size == int(row.size_bytes)
                    and actual_hash == str(row.sha256).lower()
                ),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty or not frame["passed"].all():
        raise RuntimeError(f"{scope} parent inventory does not validate")
    return frame


def capture_parent_tree(package_root: Path, config: Mapping[str, Any]) -> pd.DataFrame:
    contracts = config["parent_contract"]
    frames = [
        _inventory_tree(
            package_root,
            str(contracts["stage3_current_pointer"]["path"]),
            scope="stage3_current_inventory",
        ),
        _inventory_tree(
            package_root,
            str(contracts["stage4_current_pointer"]["path"]),
            scope="stage4_parent_inventory",
        ),
    ]
    frame = pd.concat(frames, ignore_index=True)
    return frame.sort_values(["scope", "relative_path"]).reset_index(drop=True)


def _run_tests(package_root: Path) -> dict[str, Any]:
    started = time.perf_counter()
    command = [sys.executable, "-m", "pytest", "-q", "tests", "-p", "no:cacheprovider"]
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["MPLBACKEND"] = "Agg"
    completed = subprocess.run(
        command,
        cwd=package_root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    import re

    text = completed.stdout + "\n" + completed.stderr
    passed_matches = re.findall(r"(\d+) passed", text)
    failed_matches = re.findall(r"(\d+) failed", text)
    return {
        "command": command,
        "returncode": int(completed.returncode),
        "passed": int(passed_matches[-1]) if passed_matches else 0,
        "failed": int(failed_matches[-1]) if failed_matches else 0,
        "runtime_sec": float(time.perf_counter() - started),
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def _nonempty_test_stream(value: str, *, stream_name: str) -> str:
    """Keep successful empty pytest streams auditable without zero-byte artifacts."""
    if stream_name not in {"stdout", "stderr"}:
        raise ValueError("stream_name must be stdout or stderr")
    if not isinstance(value, str):
        raise TypeError(f"{stream_name} must be text")
    return value if value else f"NO_{stream_name.upper()}_OUTPUT\n"


def _test_manifest(package_root: Path) -> pd.DataFrame:
    paths = sorted((package_root / "tests").glob("test_*.py"))
    if not paths:
        raise RuntimeError("No regression tests found")
    return pd.DataFrame(
        [
            {
                "relative_path": ensure_relative_to(path, package_root),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in paths
        ]
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if pd.isna(value) if not isinstance(value, (dict, list, tuple, set)) else False:
        return None
    return value


def _atomic_yaml(path: Path, value: Mapping[str, Any]) -> None:
    atomic_write_text(path, yaml.safe_dump(dict(value), allow_unicode=True, sort_keys=False))


def _solution_metric_row(
    solution: ScenarioSolution,
    *,
    candidate_set: str,
    engine: str,
    seed: int | None = None,
    role: str,
) -> dict[str, Any]:
    scalar_metrics = {
        key: _json_safe(value)
        for key, value in solution.metrics.items()
        if not isinstance(value, (dict, list, tuple, set))
    }
    return {
        "candidate_set": candidate_set,
        "scenario": solution.scenario,
        "catchment_id": solution.catchment_id,
        "visit_count": int(solution.visit_count),
        "engine": engine,
        "seed": seed,
        "run_role": role,
        "solver_status": solution.status,
        "selected_venue_count": int(len(solution.selected)),
        **scalar_metrics,
    }


def _stage_rows(
    solution: ScenarioSolution,
    *,
    candidate_set: str,
    engine: str,
    seed: int | None,
    role: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for order, stage in enumerate(solution.stages, start=1):
        raw = stage.raw_solver_stats
        incumbent = None if stage.objective_value is None else float(stage.objective_value)
        bound = None if stage.best_bound is None else float(stage.best_bound)
        if incumbent is None or bound is None:
            absolute = relative = None
        else:
            directional = bound - incumbent if stage.sense == "max" else incumbent - bound
            absolute = max(0.0, directional)
            relative = absolute / max(abs(incumbent), 1.0)
        rows.append(
            {
                "candidate_set": candidate_set,
                "scenario": solution.scenario,
                "catchment_id": solution.catchment_id,
                "visit_count": int(solution.visit_count),
                "engine": engine,
                "seed": seed,
                "run_role": role,
                "stage_order": order,
                "objective_name": stage.objective_name,
                "sense": stage.sense,
                "solver_status": stage.status,
                "objective_value": incumbent,
                "best_bound": bound,
                "absolute_gap": absolute,
                "relative_gap": relative,
                "wall_time_sec": float(stage.wall_time_sec),
                "branches": raw.get("branches", raw.get("mip_node_count")),
                "conflicts": raw.get("conflicts"),
                "branches_status": (
                    "REPORTED"
                    if raw.get("branches", raw.get("mip_node_count")) is not None
                    else "MISSING_REQUIRED_TELEMETRY"
                ),
                "conflicts_status": (
                    "REPORTED_BY_CP_SAT"
                    if engine == "compressed_cp_sat" and raw.get("conflicts") is not None
                    else (
                        "MISSING_REQUIRED_CP_SAT_TELEMETRY"
                        if engine == "compressed_cp_sat"
                        else "NOT_EXPOSED_BY_SCIPY_HIGHS"
                    )
                ),
                "backend": raw.get("backend", engine),
                "requested_threads": raw.get("requested_threads"),
                "requested_random_seed": raw.get("requested_random_seed"),
                "minimum_total_population_floor": raw.get(
                    "minimum_total_population_floor"
                ),
                "applied_objective_floors_json": stable_json_dumps(
                    _json_safe(raw.get("applied_objective_floors", {}))
                ),
                "solver_message": raw.get("message", raw.get("response_stats", "")),
                "raw_solver_stats_json": stable_json_dumps(_json_safe(raw)),
            }
        )
    return rows


def _certify_stages(
    stage_frame: pd.DataFrame,
    contract: NearOptimalityContract,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    normalized_frames: list[pd.DataFrame] = []
    run_rows: list[dict[str, Any]] = []
    keys = ["candidate_set", "scenario", "catchment_id", "visit_count", "engine", "seed", "run_role"]
    for values, group in stage_frame.groupby(keys, dropna=False, sort=True):
        records = group.sort_values("stage_order").copy()
        scenario = str(records["scenario"].iloc[0])
        expected_plan = EXPECTED_LEXICOGRAPHIC_STAGES.get(scenario)
        if expected_plan is None:
            raise RuntimeError(f"Unsupported lexicographic scenario: {scenario}")
        observed_plan = tuple(
            zip(
                records["objective_name"].astype(str),
                records["sense"].astype(str).str.lower(),
            )
        )
        observed_order = tuple(pd.to_numeric(records["stage_order"]).astype(int))
        if observed_order != tuple(range(1, len(expected_plan) + 1)):
            raise RuntimeError(
                f"{scenario} lexicographic stage order is incomplete: {observed_order}"
            )
        if observed_plan != expected_plan:
            raise RuntimeError(
                f"{scenario} lexicographic plan mismatch: {observed_plan} != {expected_plan}"
            )
        adapted = records.copy()
        engines = set(adapted["engine"].astype(str))
        if len(engines) != 1:
            raise RuntimeError(f"A solver run mixes engines: {sorted(engines)}")
        engine = next(iter(engines))
        branches = pd.to_numeric(adapted["branches"], errors="coerce")
        conflicts = pd.to_numeric(adapted["conflicts"], errors="coerce")
        if branches.isna().any():
            raise RuntimeError(f"{engine} solver telemetry is missing branch/node counts")
        if engine == "compressed_cp_sat":
            if conflicts.isna().any():
                raise RuntimeError("CP-SAT solver telemetry is missing conflict counts")
            adapted["conflicts"] = conflicts.astype(int)
        elif engine == "scipy_highs":
            # SciPy/HiGHS exposes MIP nodes but not a CP-SAT conflict metric.
            # The adapter exists only inside the generic certifier; the public
            # source ledger retains a null and NOT_EXPOSED status.
            adapted["conflicts"] = 0
        else:
            raise RuntimeError(f"Unsupported certification engine: {engine}")
        adapted["branches"] = branches.astype(int)
        normalized = normalize_solver_stages(adapted.to_dict("records"), contract, strict=True)
        for key, value in zip(keys, values):
            normalized.insert(len(normalized.columns), key, value)
        normalized_frames.append(normalized)
        certification = certify_solver_run(adapted.to_dict("records"), contract, strict=True)
        run_rows.append({**dict(zip(keys, values)), **certification.as_dict()})
    return pd.concat(normalized_frames, ignore_index=True), pd.DataFrame(run_rows)


def _with_bundle_assignment(
    solution: ScenarioSolution,
    data: OptimizationInput,
    base_config: dict[str, Any],
    *,
    minimum: int,
) -> ScenarioSolution:
    local = deepcopy(base_config)
    local["bundle_assignment"] = deepcopy(base_config["bundle_assignment"])
    local["bundle_assignment"]["default_min"] = int(minimum)
    solution.bundle_assignments = assign_bundles(
        solution.selected,
        data.bundles,
        local,
        solver_name="cp-sat",
    )
    solution.metrics = enrich_solution_metrics(solution, data)
    return solution


def _candidate_thresholds(config: Mapping[str, Any]) -> CandidateSensitivityThresholds:
    cfg = config["candidate_sets"]
    return CandidateSensitivityThresholds(
        max_unique_coverage_loss_fraction=float(
            cfg["top3_max_unique_coverage_loss_fraction"]
        ),
        max_need_weighted_loss_fraction=float(
            cfg["top3_max_need_weighted_loss_fraction"]
        ),
        max_high_need_loss_fraction=float(cfg["top3_max_high_need_loss_fraction"]),
        max_min_sigungu_coverage_loss=float(
            cfg["top3_max_min_sigungu_coverage_loss"]
        ),
        min_selected_admin_jaccard=float(cfg["min_selected_admin_jaccard"]),
        min_selected_coverage_cluster_jaccard=float(
            cfg["min_selected_coverage_cluster_jaccard"]
        ),
        max_solver_relative_gap=float(cfg["maximum_solver_relative_gap"]),
    )


def _highs_runtime_options(
    final_config: Mapping[str, Any], *, frontier_retry: bool = False
) -> dict[str, int]:
    """Return the frozen native HiGHS resource and determinism settings."""

    return {
        "threads": int(
            final_config["runtime"][
                "frontier_retry_highs_threads"
                if frontier_retry
                else "highs_threads_per_task"
            ]
        ),
        "random_seed": int(final_config["runtime"]["highs_random_seed"]),
    }


def _candidate_top3_identity(build: Any, parent_run: Path) -> dict[str, Any]:
    parent_path = parent_run / "00_preflight/stage4_candidate_set.csv"
    if not parent_path.is_file():
        raise FileNotFoundError(parent_path)
    parent = pd.read_csv(parent_path, dtype={"venue_id": "string"})
    current_ids = set(build.inputs["top3"].candidates["venue_id"].astype(str))
    parent_ids = set(parent["venue_id"].astype(str))

    def digest(values: set[str]) -> str:
        payload = "".join(f"{value}\n" for value in sorted(values))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    union = current_ids | parent_ids
    return {
        "current_count": len(current_ids),
        "parent_count": len(parent_ids),
        "intersection_count": len(current_ids & parent_ids),
        "current_only_count": len(current_ids - parent_ids),
        "parent_only_count": len(parent_ids - current_ids),
        "venue_set_jaccard": len(current_ids & parent_ids) / max(1, len(union)),
        "current_venue_set_sha256": digest(current_ids),
        "parent_venue_set_sha256": digest(parent_ids),
        "parent_relative_path": "00_preflight/stage4_candidate_set.csv",
        "exact_set_equal": current_ids == parent_ids,
    }


def _seed_thresholds(config: Mapping[str, Any]) -> SeedStabilityThresholds:
    return SeedStabilityThresholds(**dict(config["seed_stability"]))


def _candidate_certification_disposition(
    certifications: pd.DataFrame,
    comparisons: pd.DataFrame,
    *,
    provisional_passed: bool,
    required_width: int,
) -> dict[str, Any]:
    """Resolve candidate promotion without treating an unproved expansion as evidence.

    A certified frozen Top3 may remain the field-validation universe when wider
    universes time out, but only if no *certified* comparator has already shown
    that Top3 violates the preregistered loss/Jaccard contract.  Point estimates
    from uncertified comparator incumbents are retained as diagnostics only.
    """

    expected_keys = {
        (candidate_set, scenario, role)
        for candidate_set in ("top3", "top5", "coarse_pareto")
        for scenario, role in (
            ("efficiency", "candidate_efficiency_reference"),
            ("balanced", "candidate_sensitivity"),
        )
    }
    observed_keys = set(
        zip(
            certifications["candidate_set"].astype(str),
            certifications["scenario"].astype(str),
            certifications["run_role"].astype(str),
        )
    )
    run_contract_complete = bool(
        len(certifications) == 6
        and observed_keys == expected_keys
        and certifications["engine"].astype(str).eq("scipy_highs").all()
        and certifications["catchment_id"].astype(str).eq("hard_5000m").all()
        and certifications["visit_count"].astype(int).eq(20).all()
        and certifications["stage_count"].astype(int).eq(4).all()
        and certifications["telemetry_complete"].astype(bool).all()
        and certifications["max_relative_gap"].notna().all()
    )
    certified_keys = set(
        zip(
            certifications.loc[
                certifications["certified"].astype(bool), "candidate_set"
            ].astype(str),
            certifications.loc[
                certifications["certified"].astype(bool), "scenario"
            ].astype(str),
            certifications.loc[
                certifications["certified"].astype(bool), "run_role"
            ].astype(str),
        )
    )
    baseline_expected = {
        ("top3", "efficiency", "candidate_efficiency_reference"),
        ("top3", "balanced", "candidate_sensitivity"),
    }
    baseline_certified = bool(baseline_expected <= certified_keys)
    certified_comparators = sorted(
        candidate_set
        for candidate_set in ("top5", "coarse_pareto")
        if {
            (candidate_set, "efficiency", "candidate_efficiency_reference"),
            (candidate_set, "balanced", "candidate_sensitivity"),
        }
        <= certified_keys
    )
    uncertified_comparators = sorted(
        set(("top5", "coarse_pareto")) - set(certified_comparators)
    )
    certified_harm_evidence: list[dict[str, Any]] = []
    for row in comparisons.to_dict("records"):
        reference = str(row.get("reference_candidate_set", ""))
        if (
            str(row.get("evaluation_baseline", "")) == "top3"
            and reference in certified_comparators
            and not bool(row.get("comparison_passed", False))
        ):
            certified_harm_evidence.append(row)

    common = {
        "candidate_solver_runs_complete": run_contract_complete,
        "selected_universe_solver_certified": baseline_certified,
        "candidate_reduction_certified": bool(
            run_contract_complete
            and baseline_certified
            and len(certified_comparators) == 2
        ),
        "expanded_comparison_conclusive": bool(
            run_contract_complete
            and baseline_certified
            and len(certified_comparators) == 2
        ),
        "certified_comparators": certified_comparators,
        "uncertified_comparators": uncertified_comparators,
        "certified_harm_evidence": certified_harm_evidence,
    }
    if not run_contract_complete or not baseline_certified:
        return {
            **common,
            "decision": "INCONCLUSIVE_CERTIFIED_TOP3_BASELINE_UNAVAILABLE",
            "selected_candidate_set": None,
            "selection_action": "NO_SELECTION",
            "expansion_promotion_allowed": False,
        }
    if uncertified_comparators:
        if certified_harm_evidence:
            return {
                **common,
                "decision": "INCONCLUSIVE_CERTIFIED_COMPARATOR_REJECTS_TOP3",
                "selected_candidate_set": None,
                "selection_action": "NO_SELECTION",
                "expansion_promotion_allowed": False,
            }
        return {
            **common,
            "decision": "RETAIN_CERTIFIED_TOP3__EXPANSION_COMPARISON_INCONCLUSIVE",
            "selected_candidate_set": "top3",
            "selection_action": "RETAIN_FROZEN_CERTIFIED_BASELINE",
            "retention_basis": "CERTIFIED_FROZEN_TOP3_FAIL_CLOSED",
            "expansion_promotion_allowed": False,
        }
    if not provisional_passed:
        return {
            **common,
            "decision": "INCONCLUSIVE_CERTIFIED_COMPARISON_NO_SELECTION",
            "selected_candidate_set": None,
            "selection_action": "NO_SELECTION",
            "expansion_promotion_allowed": False,
        }
    selected = ("top3", "top5", "coarse_pareto")[required_width]
    return {
        **common,
        "decision": {
            "top3": "PASS_RETAIN_TOP3",
            "top5": "PROMOTE_TOP5",
            "coarse_pareto": "PROMOTE_COARSE_PARETO",
        }[selected],
        "selected_candidate_set": selected,
        "selection_action": "CERTIFIED_PROMOTION_LADDER_DECISION",
        "selected_universe_solver_certified": True,
        "expansion_promotion_allowed": True,
    }


def _solve_candidate_universes(
    build: Any,
    base_config: dict[str, Any],
    final_config: Mapping[str, Any],
) -> tuple[dict[str, dict[str, ScenarioSolution]], pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    limit = float(final_config["runtime"]["candidate_time_per_stage_sec"])
    # Candidate promotion is intentionally one same-policy comparison.  The
    # certified efficiency solve supplies Balanced's frozen efficiency floor;
    # final Efficiency/Equity plans are rerun after the universe decision.
    by_scenario: dict[str, dict[str, ScenarioSolution]] = {
        "efficiency": {},
        "balanced": {},
    }
    telemetry: list[dict[str, Any]] = []

    def solve_candidate_set(
        set_id: str,
    ) -> tuple[str, ScenarioSolution, ScenarioSolution]:
        data = build.inputs[set_id]
        weights = prepare_weights(data, base_config)
        groups = build_compressed_coverage_groups(data, weights)
        greedy_floor = constrained_greedy_efficiency_floor(
            data, base_config, 20
        )
        efficiency = solve_highs_scenario(
            data,
            base_config,
            "efficiency",
            20,
            time_limit_sec=limit,
            groups=groups,
            relative_gap_target=float(
                final_config["solver"]["near_optimal_relative_gap_max"]
            ),
            **_highs_runtime_options(final_config),
            minimum_total_population_floor=greedy_floor,
        )
        if not efficiency.metrics:
            raise RuntimeError(f"HiGHS returned no candidate-sensitivity efficiency incumbent for {set_id}")
        reference = int(efficiency.metrics["unique_elderly_population_scaled"])
        solution = solve_highs_scenario(
            data,
            base_config,
            "balanced",
            20,
            efficiency_reference=reference,
            time_limit_sec=limit,
            groups=groups,
            relative_gap_target=float(
                final_config["solver"]["near_optimal_relative_gap_max"]
            ),
            **_highs_runtime_options(final_config),
        )
        if not solution.metrics:
            raise RuntimeError(
                f"HiGHS returned no candidate-sensitivity balanced incumbent for {set_id}"
            )
        return set_id, efficiency, solution

    # The frozen baseline is solved alone so wider, harder universes cannot
    # starve its proof. Only the two optional expansions share resources.
    top3_result = solve_candidate_set("top3")
    expansion_sets = ("top5", "coarse_pareto")
    worker_count = int(final_config["runtime"]["candidate_parallel_workers"])
    with ThreadPoolExecutor(
        max_workers=worker_count, thread_name_prefix="stage4-candidate"
    ) as executor:
        expansion_results = list(executor.map(solve_candidate_set, expansion_sets))
    solved_candidate_sets = [top3_result, *expansion_results]
    for set_id, efficiency, solution in solved_candidate_sets:
        data = build.inputs[set_id]
        _with_bundle_assignment(efficiency, data, base_config, minimum=1)
        by_scenario["efficiency"][set_id] = efficiency
        telemetry.extend(
            _stage_rows(
                efficiency,
                candidate_set=set_id,
                engine="scipy_highs",
                seed=None,
                role="candidate_efficiency_reference",
            )
        )
        _with_bundle_assignment(solution, data, base_config, minimum=1)
        by_scenario["balanced"][set_id] = solution
        telemetry.extend(
            _stage_rows(
                solution,
                candidate_set=set_id,
                engine="scipy_highs",
                seed=None,
                role="candidate_sensitivity",
            )
        )

    candidate_contract = NearOptimalityContract(
        contract_id="MEDIROAD_STAGE4_CANDIDATE_REL_0P005",
        relative_gap_threshold=float(
            final_config["solver"]["near_optimal_relative_gap_max"]
        ),
    )
    candidate_stage_frame = pd.DataFrame(telemetry)
    candidate_normalized, candidate_certifications = _certify_stages(
        candidate_stage_frame,
        candidate_contract,
    )
    candidate_uncertified = candidate_certifications[
        ~candidate_certifications["certified"].astype(bool)
    ]
    thresholds = _candidate_thresholds(final_config)
    summaries: list[pd.DataFrame] = []
    comparisons: list[pd.DataFrame] = []
    decisions: dict[str, Any] = {}
    width = {"top3": 0, "top5": 1, "coarse_pareto": 2}
    required_width = 0
    promotion_scenario = str(final_config["candidate_sets"]["promotion_scenario"])
    for scenario in ("balanced",):
        solutions = by_scenario[scenario]
        top3 = evaluate_candidate_sensitivity(
            solutions,
            thresholds=thresholds,
            baseline_set="top3",
            reference_sets=("top5", "coarse_pareto"),
            fallback_order=("top5", "coarse_pareto"),
            preflight=build.preflight,
        )
        top3.summary["evaluation_baseline"] = "top3"
        top3.comparisons["evaluation_baseline"] = "top3"
        summaries.append(top3.summary)
        comparisons.append(top3.comparisons)
        is_promotion_scenario = scenario == promotion_scenario
        if top3.decision == "INCONCLUSIVE_CANDIDATE_SENSITIVITY_SOLVER_GAP":
            decisions[scenario] = {
                "decision": top3.decision,
                "recommended_candidate_set": None,
                "passed": False,
                "threshold_fingerprint": top3.threshold_fingerprint,
                "notes": top3.notes,
                "decision_role": "PROMOTION" if is_promotion_scenario else "DIAGNOSTIC_ONLY",
            }
            continue
        if top3.recommended_candidate_set == "top3":
            recommended = "top3"
            decision_label = "PASS_RETAIN_TOP3"
        else:
            top5 = evaluate_candidate_sensitivity(
                solutions,
                thresholds=thresholds,
                baseline_set="top5",
                reference_sets=("coarse_pareto",),
                fallback_order=("coarse_pareto",),
                preflight=build.preflight,
            )
            top5.summary["evaluation_baseline"] = "top5"
            top5.comparisons["evaluation_baseline"] = "top5"
            summaries.append(top5.summary)
            comparisons.append(top5.comparisons)
            if top5.decision == "INCONCLUSIVE_CANDIDATE_SENSITIVITY_SOLVER_GAP":
                decisions[scenario] = {
                    "decision": top5.decision,
                    "recommended_candidate_set": None,
                    "passed": False,
                    "threshold_fingerprint": top5.threshold_fingerprint,
                    "notes": top5.notes,
                }
                continue
            recommended = str(top5.recommended_candidate_set)
            decision_label = (
                "PROMOTE_TOP5" if recommended == "top5" else "PROMOTE_COARSE_PARETO"
            )
        if is_promotion_scenario:
            required_width = max(required_width, width[recommended])
        decisions[scenario] = {
            "decision": decision_label,
            "recommended_candidate_set": recommended,
            "passed": True,
            "threshold_fingerprint": thresholds.fingerprint,
            "notes": top3.notes,
            "decision_role": "PROMOTION" if is_promotion_scenario else "DIAGNOSTIC_ONLY",
        }
    summary_frame = pd.concat(summaries, ignore_index=True)
    comparison_frame = pd.concat(comparisons, ignore_index=True)
    promotion_payload = decisions[promotion_scenario]
    disposition = _candidate_certification_disposition(
        candidate_certifications,
        comparison_frame,
        provisional_passed=bool(promotion_payload.get("passed")),
        required_width=required_width,
    )
    overall = disposition["selected_candidate_set"]
    overall_decision = str(disposition["decision"])
    comparison_certified = bool(disposition["candidate_reduction_certified"])
    promotion_payload.update(
        {
            "decision": overall_decision,
            "recommended_candidate_set": overall,
            "passed": bool(comparison_certified and overall is not None),
            "comparison_passed": bool(comparison_certified and overall is not None),
            "disposition_passed": bool(overall is not None),
            **disposition,
            "uncertified_solver_runs": candidate_uncertified[
                ["candidate_set", "scenario", "run_role", "max_relative_gap"]
            ].to_dict("records"),
        }
    )
    certified_comparators = set(map(str, disposition["certified_comparators"]))
    comparison_frame["evidence_status"] = np.where(
        comparison_frame["reference_candidate_set"].astype(str).isin(
            certified_comparators
        ),
        "CERTIFIED_SAME_CONTRACT",
        "DIAGNOSTIC_UNCERTIFIED_INCUMBENT",
    )
    comparison_frame["performance_claim_allowed"] = comparison_frame[
        "evidence_status"
    ].eq("CERTIFIED_SAME_CONTRACT")
    decision = {
        "baseline_set": "top3",
        "mandatory_reference_sets": ["top5", "coarse_pareto"],
        "promotion_ladder": ["top3", "top5", "coarse_pareto"],
        "promotion_scenario": promotion_scenario,
        "non_promotion_scenarios_role": "DIAGNOSTIC_ONLY",
        "per_scenario": decisions,
        "decision": overall_decision,
        "selected_candidate_set": overall,
        "thresholds": asdict(thresholds),
        "threshold_fingerprint": thresholds.fingerprint,
        "contract_status": final_config["contract_status"],
        "contract_change_reason": final_config["contract_change_reason"],
        "uncertified_expansion_disposition_contract": dict(
            final_config["candidate_sets"]["uncertified_expansion_disposition"]
        ),
        "candidate_solver_run_count": int(len(candidate_certifications)),
        "candidate_solver_uncertified_count": int(len(candidate_uncertified)),
        **disposition,
    }
    return (
        by_scenario,
        summary_frame,
        comparison_frame,
        {
            "decision": decision,
            "stage_rows": telemetry,
            "normalized_stages": candidate_normalized,
            "run_certifications": candidate_certifications,
        },
    )


def _solve_final_scenarios(
    data: OptimizationInput,
    base_config: dict[str, Any],
    final_config: Mapping[str, Any],
    candidate_set: str,
    reusable_solutions: Mapping[str, Mapping[str, ScenarioSolution]],
) -> tuple[dict[str, ScenarioSolution], list[dict[str, Any]]]:
    limit = float(final_config["runtime"]["final_time_per_stage_sec"])
    solutions: dict[str, ScenarioSolution] = {}
    stage_rows: list[dict[str, Any]] = []
    efficiency = deepcopy(reusable_solutions["efficiency"][candidate_set])
    if not efficiency.metrics:
        raise RuntimeError("Final HiGHS efficiency solve lacks an incumbent")
    efficiency.notes.append(
        "Reused the same-budget certified candidate-universe efficiency run."
    )
    solutions["efficiency"] = efficiency
    reference = int(efficiency.metrics["unique_elderly_population_scaled"])
    balanced = deepcopy(reusable_solutions["balanced"][candidate_set])
    if not balanced.metrics:
        raise RuntimeError("Final HiGHS balanced solve lacks an incumbent")
    balanced.notes.append(
        "Reused the same-budget certified candidate-universe balanced run."
    )
    solutions["balanced"] = balanced
    weights = prepare_weights(data, base_config)
    groups = build_compressed_coverage_groups(data, weights)
    equity = solve_highs_scenario(
        data,
        base_config,
        "equity",
        20,
        efficiency_reference=reference,
        time_limit_sec=limit,
        groups=groups,
        relative_gap_target=float(
            final_config["solver"]["near_optimal_relative_gap_max"]
        ),
        **_highs_runtime_options(final_config),
    )
    if not equity.metrics:
        raise RuntimeError("Final HiGHS equity solve lacks an incumbent")
    _with_bundle_assignment(equity, data, base_config, minimum=1)
    solutions["equity"] = equity
    for solution in solutions.values():
        stage_rows.extend(
            _stage_rows(
                solution,
                candidate_set=candidate_set,
                engine="scipy_highs",
                seed=None,
                role="final_n20_certification",
            )
        )
    return solutions, stage_rows


def _solve_seed_runs(
    data: OptimizationInput,
    base_config: dict[str, Any],
    final_config: Mapping[str, Any],
    candidate_set: str,
    efficiency_reference: int,
) -> tuple[dict[int, ScenarioSolution], list[dict[str, Any]]]:
    limit = float(final_config["runtime"]["diagnostic_time_per_stage_sec"])
    solutions: dict[int, ScenarioSolution] = {}
    stage_rows: list[dict[str, Any]] = []
    for seed in map(int, final_config["solver"]["seeds"]):
        local = deepcopy(base_config)
        local["seed"] = seed
        solution = solve_compressed_scenario(
            data,
            local,
            "balanced",
            20,
            efficiency_reference=int(efficiency_reference),
            time_limit_sec=limit,
            seed=seed,
        )
        if not solution.metrics:
            raise RuntimeError(f"Compressed CP-SAT seed {seed} returned no incumbent")
        _with_bundle_assignment(solution, data, local, minimum=1)
        solutions[seed] = solution
        stage_rows.extend(
            _stage_rows(
                solution,
                candidate_set=candidate_set,
                engine="compressed_cp_sat",
                seed=seed,
                role="balanced_seed_stability",
            )
        )
    return solutions, stage_rows


def _bundle_sensitivity(
    selected: pd.DataFrame,
    bundle_values: pd.DataFrame,
    base_config: dict[str, Any],
) -> tuple[dict[int, pd.DataFrame], pd.DataFrame, dict[str, Any]]:
    assignments: dict[int, pd.DataFrame] = {}
    rows: list[dict[str, Any]] = []
    for minimum in (0, 1):
        local = deepcopy(base_config)
        local["bundle_assignment"] = deepcopy(base_config["bundle_assignment"])
        local["bundle_assignment"]["default_min"] = minimum
        assignment = assign_bundles(selected, bundle_values, local, solver_name="cp-sat")
        assignments[minimum] = assignment
        counts = assignment["bundle_id"].astype(str).value_counts().sort_index().to_dict()
        assignment_complete = bool(
            len(assignment) == 20
            and assignment["venue_id"].astype(str).nunique() == 20
            and assignment["bundle_id"].notna().all()
        )
        minimum_count = min(map(int, counts.values()), default=0)
        rows.append(
            {
                "minimum_per_bundle": minimum,
                "selected_venue_count": int(assignment["venue_id"].nunique()),
                "bundle_count_unique": int(assignment["bundle_id"].nunique()),
                "total_bundle_value": float(assignment["bundle_value"].sum()),
                "mean_bundle_value_within_venue_pct": float(
                    assignment["bundle_value_within_venue_pct"].mean()
                ),
                "bundle_counts_json": stable_json_dumps(counts),
                "solver_statuses": "|".join(
                    sorted(assignment["bundle_assignment_status"].astype(str).unique())
                ),
                "assignment_complete": assignment_complete,
                "minimum_observed_bundle_count": minimum_count,
                "bundle_representation_floor_met": bool(minimum_count >= minimum),
                "spatial_sensitivity_status": "NOT_APPLICABLE_FIXED_ANCHORS",
            }
        )
    left = set(
        assignments[0]["venue_id"].astype(str)
        + "::"
        + assignments[0]["bundle_id"].astype(str)
    )
    right = set(
        assignments[1]["venue_id"].astype(str)
        + "::"
        + assignments[1]["bundle_id"].astype(str)
    )
    jaccard = len(left & right) / max(1, len(left | right))
    summary = pd.DataFrame(rows)
    summary["venue_bundle_jaccard_min0_vs_min1"] = jaccard
    total_by_minimum = summary.set_index("minimum_per_bundle")["total_bundle_value"]
    total_delta = float(total_by_minimum.loc[1] - total_by_minimum.loc[0])
    summary["total_bundle_value_delta_min1_minus_min0"] = total_delta
    decision = {
        "decision": "POLICY_CONFIRMED_MIN1",
        "primary_minimum_per_bundle": 1,
        "rationale": "predeclared representation floor for all five frozen service bundles",
        "spatial_sensitivity_status": "NOT_APPLICABLE_FIXED_ANCHORS",
        "spatial_selection_changed": None,
        "spatial_anchor_count": 20,
        "venue_bundle_jaccard": jaccard,
        "total_bundle_value_delta_min1_minus_min0": total_delta,
        "minimums_tested": [0, 1],
    }
    return assignments, summary, decision


def _policy_frequency(solutions: Mapping[str, ScenarioSolution]) -> pd.DataFrame:
    expected = {"efficiency", "balanced", "equity"}
    if set(solutions) != expected:
        raise ValueError("Policy frequency requires exactly three primary scenarios")
    rows = [
        {"venue_id": str(venue_id), "scenario": scenario}
        for scenario, solution in solutions.items()
        for venue_id in solution.selected["venue_id"].astype(str)
    ]
    frame = pd.DataFrame(rows)
    return (
        frame.groupby("venue_id", as_index=False)
        .agg(selection_count=("scenario", "nunique"))
        .assign(selection_frequency=lambda x: x["selection_count"] / 3.0)
    )


def _inherited_catchment_stability(parent_run: Path, candidates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    files = sorted((parent_run / "02_optimization").glob("provisional_plan__balanced__*__n20.csv"))
    if not files:
        raise RuntimeError("Parent Stage 4 has no balanced catchment comparator plans")
    records: list[dict[str, str]] = []
    for path in files:
        frame = pd.read_csv(path, dtype={"venue_id": "string"})
        for venue_id in frame["venue_id"].astype(str).unique():
            records.append({"venue_id": venue_id, "solution_id": path.stem})
    selected = pd.DataFrame(records)
    denominator = len(files)
    selected_venue = (
        selected.groupby("venue_id", as_index=False)
        .agg(catchment_count=("solution_id", "nunique"))
        .assign(catchment_stability=lambda x: x["catchment_count"] / denominator)
    )
    shortlist_rank = pd.to_numeric(candidates["shortlist_rank"], errors="coerce")
    eligible = candidates.loc[shortlist_rank.le(3), ["venue_id", "cluster_id"]].copy()
    eligible["venue_id"] = eligible["venue_id"].astype(str)
    eligible["cluster_id"] = eligible["cluster_id"].astype(str)
    if len(eligible) != 453 or eligible["venue_id"].nunique() != 453:
        raise RuntimeError("Inherited Top3 catchment evidence requires exactly 453 eligible venues")
    venue = eligible[["venue_id"]].merge(
        selected_venue,
        on="venue_id",
        how="left",
        validate="one_to_one",
    )
    venue["catchment_count"] = venue["catchment_count"].fillna(0).astype(int)
    venue["catchment_stability"] = venue["catchment_stability"].fillna(0.0)
    venue["evidence_scope"] = "INHERITED_PROVISIONAL_TOP3_DIAGNOSTIC"
    venue["eligible_candidate_set"] = "PARENT_TOP3"
    venue["evidence_universe_complete"] = True
    venue["final_promotion_allowed"] = False
    lookup = candidates[["venue_id", "cluster_id"]].copy()
    lookup["venue_id"] = lookup["venue_id"].astype(str)
    selected = selected.merge(lookup, on="venue_id", how="left", validate="many_to_one")
    if selected["cluster_id"].isna().any():
        raise RuntimeError("Parent catchment plan contains venue outside full universe")
    selected_cluster = (
        selected.groupby("cluster_id", as_index=False)
        .agg(cluster_count=("solution_id", "nunique"))
        .assign(cluster_stability=lambda x: x["cluster_count"] / denominator)
    )
    cluster = eligible[["cluster_id"]].drop_duplicates().merge(
        selected_cluster,
        on="cluster_id",
        how="left",
        validate="one_to_one",
    )
    cluster["cluster_count"] = cluster["cluster_count"].fillna(0).astype(int)
    cluster["cluster_stability"] = cluster["cluster_stability"].fillna(0.0)
    cluster["evidence_scope"] = "INHERITED_PROVISIONAL_TOP3_DIAGNOSTIC"
    cluster["eligible_candidate_set"] = "PARENT_TOP3"
    cluster["evidence_universe_complete"] = True
    cluster["final_promotion_allowed"] = False
    return venue, cluster


def _frontier_model_contract_sha256(
    data: OptimizationInput,
    base_config: Mapping[str, Any],
    final_config: Mapping[str, Any],
    candidate_set: str,
) -> str:
    """Seal the model definition shared by initial and retry frontier attempts."""

    venue_ids = sorted(data.candidates["venue_id"].astype(str).tolist())
    venue_membership_sha256 = hashlib.sha256(
        ("\n".join(venue_ids) + "\n").encode("utf-8")
    ).hexdigest()
    payload = {
        "candidate_set": str(candidate_set),
        "candidate_count": len(venue_ids),
        "candidate_membership_sha256": venue_membership_sha256,
        "catchment_id": str(data.catchment_id),
        "base_config": _json_safe(base_config),
        "objective_plans": _json_safe(EXPECTED_LEXICOGRAPHIC_STAGES),
        "frontier": _json_safe(final_config["frontier"]),
        "solver_relative_gap": float(
            final_config["solver"]["near_optimal_relative_gap_max"]
        ),
        "primary_solver": str(final_config["runtime"]["primary_solver"]),
        "highs_threads_per_task": int(
            final_config["runtime"]["highs_threads_per_task"]
        ),
        "frontier_retry_highs_threads": int(
            final_config["runtime"]["frontier_retry_highs_threads"]
        ),
        "highs_random_seed": int(final_config["runtime"]["highs_random_seed"]),
        "final_efficiency_greedy_no_harm_floor_required": bool(
            final_config["solver"][
                "final_efficiency_must_not_underperform_constrained_greedy"
            ]
        ),
        "greedy_total_population_floor_by_visit": {
            str(visit_count): constrained_greedy_efficiency_floor(
                data, base_config, int(visit_count)
            )
            for visit_count in final_config["frontier"]["visit_counts"]
        },
    }
    return hashlib.sha256(stable_json_dumps(payload).encode("utf-8")).hexdigest()


def _frontier_retry_eligible(record: Mapping[str, Any]) -> bool:
    """Only a time-limited UNKNOWN with no usable incumbent may be retried."""

    return bool(
        str(record.get("failure_status", "")) == "UNKNOWN"
        and "time limit" in str(record.get("failure_message", "")).lower()
    )


def _run_frontier(
    data: OptimizationInput,
    base_config: dict[str, Any],
    final_config: Mapping[str, Any],
    candidate_set: str,
    final_solutions: Mapping[str, ScenarioSolution],
    *,
    attempt_log_path: Path | None = None,
    attempt_stage_log_path: Path | None = None,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    list[dict[str, Any]],
    pd.DataFrame,
    pd.DataFrame,
]:
    initial_limit = float(final_config["runtime"]["frontier_time_per_stage_sec"])
    retry_limit = float(
        final_config["runtime"]["frontier_retry_time_per_stage_sec"]
    )
    retry_contract = final_config["frontier"]["no_incumbent_retry_contract"]
    if (
        retry_contract["trigger"] != "no_incumbent_only"
        or int(retry_contract["maximum_attempts"]) != 2
        or retry_contract["retry_is_sequential"] is not True
        or retry_contract["retry_changes_objectives_or_thresholds"] is not False
    ):
        raise RuntimeError("Frontier retry contract changed after config validation")
    metrics: list[dict[str, Any]] = []
    selections: list[pd.DataFrame] = []
    stages: list[dict[str, Any]] = []
    attempt_rows: list[dict[str, Any]] = []
    attempt_stage_rows: list[dict[str, Any]] = []
    weights = prepare_weights(data, base_config)
    groups = build_compressed_coverage_groups(data, weights)
    model_contract_sha256 = _frontier_model_contract_sha256(
        data, base_config, final_config, candidate_set
    )

    def failure_details(solution: ScenarioSolution) -> dict[str, Any]:
        if solution.metrics:
            return {
                "failure_scenario": "",
                "failure_objective": "",
                "failure_status": "",
                "failure_message": "",
                "failure_objective_value": None,
                "failure_best_bound": None,
                "failure_relative_gap": None,
                "failure_stage_wall_time_sec": None,
                "failure_nodes": None,
                "failure_raw_solver_stats_json": "",
            }
        stage = solution.stages[-1] if solution.stages else None
        raw = {} if stage is None else stage.raw_solver_stats
        return {
            "failure_scenario": str(solution.scenario),
            "failure_objective": "" if stage is None else str(stage.objective_name),
            "failure_status": str(solution.status if stage is None else stage.status),
            "failure_message": ""
            if stage is None
            else str(raw.get("message", "")),
            "failure_objective_value": None
            if stage is None or stage.objective_value is None
            else float(stage.objective_value),
            "failure_best_bound": None
            if stage is None or stage.best_bound is None
            else float(stage.best_bound),
            "failure_relative_gap": raw.get("relative_gap"),
            "failure_stage_wall_time_sec": None
            if stage is None
            else float(stage.wall_time_sec),
            "failure_nodes": raw.get("mip_node_count"),
            "failure_raw_solver_stats_json": ""
            if stage is None
            else stable_json_dumps(_json_safe(raw)),
        }

    def solve_visit_count(
        visit_count: int,
        *,
        attempt_number: int,
        time_limit_sec: float,
        attempt_kind: str,
        worker_count_for_attempt: int,
        highs_threads_for_attempt: int,
    ) -> tuple[int, bool, list[ScenarioSolution], dict[str, Any]]:
        attempt_started = time.perf_counter()
        reused_final = visit_count == 20
        if reused_final:
            solutions = [
                deepcopy(final_solutions["efficiency"]),
                deepcopy(final_solutions["balanced"]),
            ]
            failure = {
                "failure_scenario": "",
                "failure_objective": "",
                "failure_status": "",
                "failure_message": "",
                "failure_objective_value": None,
                "failure_best_bound": None,
                "failure_relative_gap": None,
                "failure_stage_wall_time_sec": None,
                "failure_nodes": None,
                "failure_raw_solver_stats_json": "",
            }
        else:
            greedy_floor = constrained_greedy_efficiency_floor(
                data, base_config, int(visit_count)
            )
            efficiency = solve_highs_scenario(
                data,
                base_config,
                "efficiency",
                visit_count,
                time_limit_sec=time_limit_sec,
                groups=groups,
                relative_gap_target=float(
                    final_config["solver"]["near_optimal_relative_gap_max"]
                ),
                threads=int(highs_threads_for_attempt),
                random_seed=int(final_config["runtime"]["highs_random_seed"]),
                minimum_total_population_floor=greedy_floor,
            )
            if not efficiency.metrics:
                solutions = [efficiency]
                failure = failure_details(efficiency)
            else:
                reference = int(efficiency.metrics["unique_elderly_population_scaled"])
                balanced = solve_highs_scenario(
                    data,
                    base_config,
                    "balanced",
                    visit_count,
                    efficiency_reference=reference,
                    time_limit_sec=time_limit_sec,
                    groups=groups,
                    relative_gap_target=float(
                        final_config["solver"]["near_optimal_relative_gap_max"]
                    ),
                    threads=int(highs_threads_for_attempt),
                    random_seed=int(final_config["runtime"]["highs_random_seed"]),
                )
                solutions = [efficiency, balanced]
                failure = failure_details(balanced)
        success = bool(len(solutions) == 2 and all(item.metrics for item in solutions))
        statuses = {item.scenario: item.status for item in solutions}
        record = {
            "visit_count": int(visit_count),
            "attempt_number": int(attempt_number),
            "attempt_kind": attempt_kind,
            "time_limit_per_stage_sec": float(time_limit_sec),
            "worker_count_for_attempt": int(worker_count_for_attempt),
            "requested_highs_threads": int(highs_threads_for_attempt),
            "reused_final_n20_solution": bool(reused_final),
            "success": success,
            "efficiency_status": str(statuses.get("efficiency", "NOT_RUN")),
            "balanced_status": str(statuses.get("balanced", "NOT_RUN")),
            **failure,
            "attempt_wall_time_sec": float(time.perf_counter() - attempt_started),
            "retry_trigger": "" if attempt_number == 1 else "NO_INCUMBENT",
            "model_contract_sha256": model_contract_sha256,
            "final_promotion_allowed": False,
        }
        return visit_count, reused_final, solutions, record

    def write_attempt_log() -> pd.DataFrame:
        frame = (
            pd.DataFrame(attempt_rows)
            .sort_values(["visit_count", "attempt_number"])
            .reset_index(drop=True)
        )
        if attempt_log_path is not None:
            atomic_write_csv(frame, attempt_log_path)
        return frame

    def append_attempt_stages(
        result: tuple[int, bool, list[ScenarioSolution], dict[str, Any]]
    ) -> None:
        visit_count, _reused, solutions, record = result
        for solution in solutions:
            for row in _stage_rows(
                solution,
                candidate_set=candidate_set,
                engine="scipy_highs",
                seed=None,
                role="diagnostic_visit_count_frontier_attempt",
            ):
                row.update(
                    {
                        "frontier_attempt_number": int(record["attempt_number"]),
                        "frontier_attempt_kind": str(record["attempt_kind"]),
                        "frontier_time_limit_per_stage_sec": float(
                            record["time_limit_per_stage_sec"]
                        ),
                        "frontier_attempt_success": bool(record["success"]),
                        "used_as_final_frontier_result": bool(record["success"]),
                        "model_contract_sha256": model_contract_sha256,
                        "final_promotion_allowed": False,
                        "frontier_visit_count": int(visit_count),
                    }
                )
                attempt_stage_rows.append(row)

    def write_attempt_stage_log() -> pd.DataFrame:
        frame = pd.DataFrame(attempt_stage_rows)
        if not frame.empty:
            frame = frame.sort_values(
                [
                    "visit_count",
                    "frontier_attempt_number",
                    "scenario",
                    "stage_order",
                ]
            ).reset_index(drop=True)
        if attempt_stage_log_path is not None:
            atomic_write_csv(frame, attempt_stage_log_path)
        return frame

    visit_counts = tuple(map(int, final_config["frontier"]["visit_counts"]))
    worker_count = int(final_config["runtime"]["frontier_parallel_workers"])
    initial_results: list[
        tuple[int, bool, list[ScenarioSolution], dict[str, Any]]
    ] = []
    with ThreadPoolExecutor(
        max_workers=worker_count, thread_name_prefix="stage4-frontier"
    ) as executor:
        futures = {
            executor.submit(
                solve_visit_count,
                count,
                attempt_number=1,
                time_limit_sec=0.0 if count == 20 else initial_limit,
                attempt_kind=(
                    "REUSED_FINAL_N20" if count == 20 else "INITIAL_PARALLEL"
                ),
                worker_count_for_attempt=worker_count,
                highs_threads_for_attempt=int(
                    final_config["runtime"]["highs_threads_per_task"]
                ),
            ): count
            for count in visit_counts
        }
        for future in as_completed(futures):
            result = future.result()
            initial_results.append(result)
            attempt_rows.append(result[3])
            append_attempt_stages(result)
            # Each completed visit count is checkpointed independently.  A
            # later worker exception therefore cannot erase earlier evidence.
            write_attempt_log()
            write_attempt_stage_log()
    solved_by_visit: dict[
        int, tuple[int, bool, list[ScenarioSolution], dict[str, Any]]
    ] = {result[0]: result for result in initial_results if result[3]["success"]}
    failed_counts = [result[0] for result in initial_results if not result[3]["success"]]
    failed_initial = {
        result[0]: result[3]
        for result in initial_results
        if not result[3]["success"]
    }
    non_retryable = [
        count
        for count, record in failed_initial.items()
        if not _frontier_retry_eligible(record)
    ]
    if non_retryable:
        first = failed_initial[sorted(non_retryable)[0]]
        raise RuntimeError(
            "Frontier failure is not retry-eligible under the no-incumbent "
            f"time-limit contract: scenario={first['failure_scenario']} "
            f"n={sorted(non_retryable)[0]} objective={first['failure_objective']} "
            f"status={first['failure_status']}"
        )
    retry_workers = int(final_config["runtime"]["frontier_retry_parallel_workers"])
    if retry_workers != 1:
        raise RuntimeError("Frontier no-incumbent retry must remain sequential")
    for visit_count in sorted(failed_counts):
        retry_result = solve_visit_count(
            visit_count,
            attempt_number=2,
            time_limit_sec=retry_limit,
            attempt_kind="SEQUENTIAL_NO_INCUMBENT_RETRY",
            worker_count_for_attempt=retry_workers,
            highs_threads_for_attempt=int(
                final_config["runtime"]["frontier_retry_highs_threads"]
            ),
        )
        attempt_rows.append(retry_result[3])
        append_attempt_stages(retry_result)
        write_attempt_log()
        write_attempt_stage_log()
        if not retry_result[3]["success"]:
            failure = retry_result[3]
            raise RuntimeError(
                "Frontier failed closed after the predeclared sequential "
                f"no-incumbent retry: scenario={failure['failure_scenario']} "
                f"n={visit_count} objective={failure['failure_objective']} "
                f"status={failure['failure_status']} initial_budget={initial_limit}s "
                f"retry_budget={retry_limit}s"
            )
        solved_by_visit[visit_count] = retry_result
    if set(solved_by_visit) != set(visit_counts):
        raise RuntimeError("Frontier did not produce a final result for every visit count")
    solved_frontier = [solved_by_visit[count] for count in visit_counts]
    for visit_count, reused_final, solutions, attempt_record in solved_frontier:
        for solution in solutions:
            if not solution.metrics:
                raise RuntimeError(
                    f"Frontier {solution.scenario} n={visit_count} has no incumbent"
                )
            _with_bundle_assignment(solution, data, base_config, minimum=1)
            assignment = solution.bundle_assignments
            assert assignment is not None
            row = _solution_metric_row(
                solution,
                candidate_set=candidate_set,
                engine="scipy_highs",
                role="diagnostic_visit_count_frontier",
            )
            row["total_specialty_gap_served"] = float(assignment["bundle_value"].sum())
            row["final_promotion_allowed"] = False
            row["reused_final_n20_solution"] = reused_final
            row["solver_attempt_number"] = int(attempt_record["attempt_number"])
            row["solver_time_limit_per_stage_sec"] = float(
                attempt_record["time_limit_per_stage_sec"]
            )
            row["frontier_retry_used"] = bool(
                int(attempt_record["attempt_number"]) > 1
            )
            metrics.append(row)
            selected = solution.selected.copy()
            selected.insert(0, "scenario", solution.scenario)
            selected.insert(1, "visit_count", visit_count)
            selected.insert(2, "candidate_set", candidate_set)
            selections.append(selected)
            stages.extend(
                _stage_rows(
                    solution,
                    candidate_set=candidate_set,
                    engine="scipy_highs",
                    seed=None,
                    role="diagnostic_visit_count_frontier",
                )
            )
    metric_frame = pd.DataFrame(metrics).sort_values(["scenario", "visit_count"])
    marginal_rows: list[dict[str, Any]] = []
    for scenario, group in metric_frame.groupby("scenario", sort=True):
        group = group.sort_values("visit_count")
        previous: pd.Series | None = None
        for _, row in group.iterrows():
            if previous is not None:
                delta_visits = int(row["visit_count"] - previous["visit_count"])
                marginal_rows.append(
                    {
                        "scenario": scenario,
                        "from_visit_count": int(previous["visit_count"]),
                        "to_visit_count": int(row["visit_count"]),
                        "additional_visits": delta_visits,
                        "unique_elderly_per_additional_visit": (
                            float(row["unique_elderly_population"])
                            - float(previous["unique_elderly_population"])
                        )
                        / delta_visits,
                        "high_need_scaled_per_additional_visit": (
                            float(row["high_need_population_scaled"])
                            - float(previous["high_need_population_scaled"])
                        )
                        / delta_visits,
                        "min_sigungu_coverage_change": float(
                            row["min_sigungu_coverage_ratio"]
                            - previous["min_sigungu_coverage_ratio"]
                        ),
                        "specialty_gap_per_additional_visit": (
                            float(row["total_specialty_gap_served"])
                            - float(previous["total_specialty_gap_served"])
                        )
                        / delta_visits,
                    }
                )
            previous = row
    return (
        metric_frame.reset_index(drop=True),
        pd.DataFrame(marginal_rows),
        pd.concat(selections, ignore_index=True),
        stages,
        write_attempt_log(),
        write_attempt_stage_log(),
    )


def _pareto_policy_frame(solutions: Mapping[str, ScenarioSolution]) -> pd.DataFrame:
    rows = [
        {
            "scenario": name,
            "unique_elderly_population": float(solution.metrics["unique_elderly_population"]),
            "need_weighted_population_scaled": float(
                solution.metrics["need_weighted_population_scaled"]
            ),
            "high_need_population_scaled": float(
                solution.metrics["high_need_population_scaled"]
            ),
            "min_sigungu_coverage_ratio": float(
                solution.metrics["min_sigungu_coverage_ratio"]
            ),
        }
        for name, solution in solutions.items()
    ]
    frame = pd.DataFrame(rows)
    values = frame[
        [
            "unique_elderly_population",
            "need_weighted_population_scaled",
            "high_need_population_scaled",
            "min_sigungu_coverage_ratio",
        ]
    ].to_numpy(float)
    dominated = []
    for index, value in enumerate(values):
        other = np.delete(values, index, axis=0)
        dominated.append(bool(np.any(np.all(other >= value, axis=1) & np.any(other > value, axis=1))))
    frame["pareto_nondominated"] = ~np.asarray(dominated, dtype=bool)
    return frame


def _annotate_frontier_marginal_certification(
    marginal: pd.DataFrame,
    frontier_run_certification: pd.DataFrame,
) -> pd.DataFrame:
    """Label marginal intervals by the certification of both endpoints."""

    required = {
        "scenario",
        "visit_count",
        "certification_class",
        "certified",
        "max_relative_gap",
    }
    missing = required - set(frontier_run_certification.columns)
    if missing:
        raise RuntimeError(
            f"Frontier certification is missing marginal endpoint fields: {sorted(missing)}"
        )
    lookup = frontier_run_certification[list(required)].copy()
    if lookup.duplicated(["scenario", "visit_count"]).any():
        raise RuntimeError("Frontier certification endpoint keys are duplicated")
    result = marginal.copy()
    for side, count_column in (
        ("from", "from_visit_count"),
        ("to", "to_visit_count"),
    ):
        renamed = lookup.rename(
            columns={
                "visit_count": count_column,
                "certification_class": f"{side}_certification_class",
                "certified": f"{side}_certified",
                "max_relative_gap": f"{side}_max_relative_gap",
            }
        )
        result = result.merge(
            renamed,
            on=["scenario", count_column],
            how="left",
            validate="many_to_one",
        )
    endpoint_columns = [
        "from_certification_class",
        "to_certification_class",
        "from_certified",
        "to_certified",
    ]
    if result[endpoint_columns].isna().any().any():
        raise RuntimeError("Frontier marginal interval has an unclassified endpoint")
    both = result["from_certified"].astype(bool) & result["to_certified"].astype(bool)
    result["interpretation_status"] = np.where(
        both, "CERTIFIED_ENDPOINTS", "INCONCLUSIVE_SOLVER_GAP"
    )
    result["diminishing_return_claim_allowed"] = both
    result["final_promotion_allowed"] = False
    return result


def _beneficiary_equity(
    solution: ScenarioSolution,
    data: OptimizationInput,
) -> dict[str, Any]:
    grid = data.grids[["sigungu", "elderly65_population"]].copy()
    population = pd.to_numeric(grid["elderly65_population"], errors="coerce")
    if population.isna().any():
        raise RuntimeError("Grid population is incomplete")
    grid["beneficiary_population"] = population
    grid["beneficiary_covered_population"] = np.where(
        solution.covered_grid_mask,
        population,
        0.0,
    )
    return beneficiary_visit_location_equity_kpis(
        solution.selected,
        grid[
            [
                "sigungu",
                "beneficiary_population",
                "beneficiary_covered_population",
            ]
        ],
    )


def _attach_diagnostic_travel_context(
    candidates: pd.DataFrame,
    stage3_run_root: Path,
) -> pd.DataFrame:
    """Attach frozen Stage 3 OSM drive times by venue ID.

    These columns are diagnostic-only and are intentionally absent from the
    Stage 3-to-4 interface.  Join them from the full exposure table, never
    from a shortlist and never by positional alignment.
    """

    travel_columns = [
        "osm_from_cheongju_medical_center_drive_min_v6",
        "osm_from_chungju_medical_center_drive_min_v6",
    ]
    source_path = stage3_run_root / "02_exposure/venue_exposure_summary.parquet"
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    source = pd.read_parquet(source_path, columns=["venue_id", *travel_columns])
    source["venue_id"] = source["venue_id"].astype("string")
    result = candidates.copy()
    result["venue_id"] = result["venue_id"].astype("string")
    if source["venue_id"].duplicated().any() or result["venue_id"].duplicated().any():
        raise RuntimeError("Diagnostic travel join requires unique venue_id values")
    if set(source["venue_id"].astype(str)) != set(result["venue_id"].astype(str)):
        raise RuntimeError(
            "Diagnostic travel source must cover the exact full Stage 3 venue universe"
        )

    for column in travel_columns:
        source[column] = pd.to_numeric(source[column], errors="coerce")
        if source[column].isna().any() or not np.isfinite(source[column]).all():
            raise RuntimeError(f"Diagnostic travel source column is incomplete: {column}")

    existing = [column for column in travel_columns if column in result.columns]
    if existing:
        aligned = source.set_index("venue_id").loc[result["venue_id"], travel_columns]
        aligned.index = result.index
        for column in existing:
            observed = pd.to_numeric(result[column], errors="coerce").to_numpy(dtype=float)
            expected = aligned[column].to_numpy(dtype=float)
            if not np.allclose(observed, expected, rtol=0.0, atol=1e-12):
                raise RuntimeError(f"Existing diagnostic travel values disagree: {column}")
        for column in travel_columns:
            if column not in result.columns:
                result[column] = aligned[column]
        return result

    return result.merge(source, on="venue_id", how="left", validate="one_to_one")


def _baseline_table(
    data: OptimizationInput,
    base_config: dict[str, Any],
    final_solutions: Mapping[str, ScenarioSolution],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    baselines = build_baselines(data, base_config, 20)
    rows = [
        _solution_metric_row(
            baseline,
            candidate_set=str(data.metadata.get("candidate_set_id", "final")),
            engine="deterministic_baseline",
            role="baseline_comparison",
        )
        for baseline in baselines
    ]
    rows.extend(
        _solution_metric_row(
            solution,
            candidate_set=str(data.metadata.get("candidate_set_id", "final")),
            engine="scipy_highs",
            role="final_policy",
        )
        for solution in final_solutions.values()
    )
    constrained = next(
        baseline
        for baseline in baselines
        if baseline.scenario == "BASELINE_CONSTRAINED_GREEDY_EFFICIENCY"
    )
    exact = final_solutions["efficiency"]
    no_harm = compare_greedy_no_harm(
        cp_sat_objective_value=float(exact.metrics["unique_elderly_population_scaled"]),
        greedy_objective_value=float(
            constrained.metrics["unique_elderly_population_scaled"]
        ),
        objective_name="total_population",
        sense="max",
        greedy_objective_name="total_population",
        greedy_sense="max",
        same_constraints=True,
    )
    evidence = no_harm.as_dict()
    if not exact.stages:
        raise RuntimeError("Final Efficiency solution lacks lexicographic stages")
    primary = exact.stages[0]
    if primary.objective_name != "total_population" or primary.sense != "max":
        raise RuntimeError("Final Efficiency primary stage contract changed")
    if primary.objective_value is None:
        raise RuntimeError("Final Efficiency primary stage lacks an incumbent")
    greedy_floor = int(constrained.metrics["unique_elderly_population_scaled"])
    retention_floor = ceil_fraction(
        int(primary.objective_value),
        float(base_config["optimization"]["objective_retention"]["total_population"]),
    )
    effective_floor = max(greedy_floor, retention_floor)
    final_value = int(exact.metrics["unique_elderly_population_scaled"])
    evidence.update(
        {
            "primary_stage_objective_value": int(primary.objective_value),
            "retention_floor": int(retention_floor),
            "constrained_greedy_floor": int(greedy_floor),
            "effective_total_population_floor": int(effective_floor),
            "final_efficiency_value": int(final_value),
            "floor_enforced": bool(final_value >= effective_floor),
            "floor_policy": "MAX_RETENTION_AND_CONSTRAINED_GREEDY_ZERO_TOLERANCE",
        }
    )
    evidence["passed"] = bool(evidence["passed"] and evidence["floor_enforced"])
    return pd.DataFrame(rows), evidence


def _field_readiness(
    final_solutions: Mapping[str, ScenarioSolution],
    all_candidates: pd.DataFrame,
    seed_result: Any,
    parent_run: Path,
    final_config: Mapping[str, Any],
    *,
    seed_stability_usable: bool,
) -> dict[str, Any]:
    policy = _policy_frequency(final_solutions)
    seed_venue = seed_result.selection_frequency[
        seed_result.selection_frequency["entity_type"].eq("venue")
    ][["entity_id", "seed_frequency"]].rename(
        columns={"entity_id": "venue_id", "seed_frequency": "seed_stability"}
    )
    seed_venue["solver_certified_for_queue"] = bool(seed_stability_usable)
    seed_venue["evidence_scope"] = (
        "MEASURED_CERTIFIED_CP_SAT_SEEDS"
        if seed_stability_usable
        else "DIAGNOSTIC_UNCERTIFIED_CP_SAT_SEEDS"
    )
    catchment, cluster = _inherited_catchment_stability(parent_run, all_candidates)
    primary = final_solutions["balanced"].selected.copy()
    fallbacks = actual_facility_fallbacks(
        primary,
        all_candidates,
        max_fallbacks=int(final_config["busproxy"]["fallback_count"]),
    )
    queue = build_field_validation_queue(
        policy,
        all_candidates,
        primary_plan=primary,
        seed_stability=seed_venue if seed_stability_usable else None,
        catchment_stability=catchment,
        cluster_stability=cluster,
        min_size=int(final_config["field_validation"]["queue_min"]),
        max_size=int(final_config["field_validation"]["queue_max"]),
        robust_threshold=float(final_config["field_validation"]["robust_frequency_threshold"]),
    )
    expected_fallback_ids = {
        str(value)
        for column in ("actual_facility_fallback_1", "actual_facility_fallback_2")
        for value in fallbacks[column].dropna().astype(str)
    }
    observed_fallback_ids = set(
        queue.loc[queue["is_primary_fallback"].astype(bool), "venue_id"].astype(str)
    )
    if expected_fallback_ids != observed_fallback_ids:
        raise RuntimeError("Field queue and physical fallback resolver disagree")
    form = make_field_validation_form(queue)
    validated_unknown = validate_field_validation_schema(form)
    resolution = resolve_final_physical_venues(
        primary,
        all_candidates,
        validated_unknown,
        max_fallbacks=int(final_config["busproxy"]["fallback_count"]),
    )
    core = robust_core_summary(
        policy,
        threshold=float(final_config["field_validation"]["robust_frequency_threshold"]),
    )
    return {
        "policy_frequency": policy,
        "seed_venue_stability": seed_venue,
        "catchment_stability": catchment,
        "cluster_stability": cluster,
        "fallbacks": fallbacks,
        "queue": queue,
        "form": form,
        "resolution": resolution,
        "robust_core": core,
    }


def _make_figures(
    candidate_comparisons: pd.DataFrame,
    seed_pairwise: pd.DataFrame,
    frontier: pd.DataFrame,
    bundle_summary: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    metrics = [
        ("unique_coverage_loss_fraction", "Unique coverage loss"),
        ("need_weighted_loss_fraction", "Need-weighted loss"),
        ("high_need_loss_fraction", "High-Need loss"),
        ("min_sigungu_coverage_loss", "Min-sigungu loss"),
    ]
    labels = (
        candidate_comparisons["evaluation_baseline"].astype(str)
        + "→"
        + candidate_comparisons["reference_candidate_set"].astype(str)
        + "\n"
        + candidate_comparisons["objective_metric"].astype(str)
    )
    for axis, (column, title) in zip(axes.ravel(), metrics):
        axis.barh(labels, pd.to_numeric(candidate_comparisons[column]))
        axis.set_title(title)
        axis.set_xlabel("loss")
    if (
        "evidence_status" in candidate_comparisons
        and candidate_comparisons["evidence_status"]
        .astype(str)
        .eq("DIAGNOSTIC_UNCERTIFIED_INCUMBENT")
        .any()
    ):
        fig.suptitle(
            "Candidate expansion: uncertified-incumbent diagnostics (no performance claim)",
            fontsize=11,
        )
    path = output_dir / "candidate_reduction_sensitivity.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    paths.append(path)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    if not seed_pairwise.empty:
        axes[0].hist(seed_pairwise["venue_jaccard"], bins=np.linspace(0, 1, 11))
        axes[1].hist(seed_pairwise["coverage_cluster_jaccard"], bins=np.linspace(0, 1, 11))
    axes[0].set_title("Seed pair venue Jaccard")
    axes[1].set_title("Seed pair coverage-cluster Jaccard")
    for axis in axes:
        axis.set_xlim(0, 1)
    path = output_dir / "seed_stability.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    paths.append(path)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for scenario, group in frontier.groupby("scenario"):
        axes[0].plot(group["visit_count"], group["unique_elderly_population"], marker="o", label=scenario)
        axes[1].plot(group["visit_count"], group["min_sigungu_coverage_ratio"], marker="o", label=scenario)
    axes[0].set_title("Unique elderly coverage frontier")
    axes[1].set_title("Minimum sigungu coverage frontier")
    for axis in axes:
        axis.set_xlabel("visits")
        axis.legend()
    path = output_dir / "visit_count_frontier.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    paths.append(path)

    fig, axis = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    axis.bar(bundle_summary["minimum_per_bundle"].astype(str), bundle_summary["total_bundle_value"])
    axis.set_xlabel("minimum visits per bundle")
    axis.set_ylabel("total specialty-gap value")
    axis.set_title("Bundle minimum sensitivity (same 20 anchors)")
    path = output_dir / "bundle_minimum_sensitivity.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    paths.append(path)
    return paths


def _gate(
    check_id: str,
    group: str,
    passed: bool,
    observed: Any,
    expected: Any,
    evidence: str,
    message: str,
    *,
    ready: bool = True,
    computational: bool = False,
    operational: bool = False,
    severity: str = "HARD",
) -> dict[str, Any]:
    return {
        "check_id": check_id,
        "gate_group": group,
        "severity": severity,
        "required_for_ready_field_validation": bool(ready),
        "required_for_computational_final": bool(computational),
        "required_for_operational_final": bool(operational),
        "passed": bool(passed),
        "observed_json": stable_json_dumps(_json_safe(observed)),
        "expected_json": stable_json_dumps(_json_safe(expected)),
        "evidence_relative_path": evidence,
        "message": message,
    }


def _diagnostic_travel_gate_evidence(
    diagnostic_travel: pd.DataFrame,
    *,
    expected_scenarios: Sequence[str],
    expected_venue_count: int,
) -> tuple[bool, dict[str, int]]:
    """Validate the canonical diagnostic-travel key without schema aliases.

    The travel module's public contract names this column ``travel_scenario``.
    Keeping that exact name here prevents the final quality gate from silently
    drifting to a generic ``scenario`` alias after all solver work completes.
    """

    required = {
        "venue_id",
        "travel_scenario",
        "travel_minutes",
        "assigned_diagnostic_base",
        "travel_value_status",
        "travel_evidence_status",
        "diagnostic_only",
        "final_promotion_allowed",
    }
    if not required.issubset(diagnostic_travel.columns):
        return False, {}
    counts = {
        str(key): int(value)
        for key, value in diagnostic_travel.groupby("travel_scenario")[
            "venue_id"
        ]
        .nunique()
        .to_dict()
        .items()
    }
    expected = {str(value) for value in expected_scenarios}
    diagnostic_flags_are_bool = diagnostic_travel["diagnostic_only"].map(
        lambda value: isinstance(value, (bool, np.bool_))
    ).all()
    promotion_flags_are_bool = diagnostic_travel["final_promotion_allowed"].map(
        lambda value: isinstance(value, (bool, np.bool_))
    ).all()
    ok = bool(
        len(diagnostic_travel) == int(expected_venue_count) * len(expected)
        and diagnostic_travel["venue_id"].nunique() == int(expected_venue_count)
        and not diagnostic_travel.duplicated(
            ["venue_id", "travel_scenario"]
        ).any()
        and set(counts) == expected
        and all(value == int(expected_venue_count) for value in counts.values())
        and diagnostic_flags_are_bool
        and promotion_flags_are_bool
        and diagnostic_travel["diagnostic_only"].eq(True).all()
        and diagnostic_travel["final_promotion_allowed"].eq(False).all()
    )
    return ok, counts


def _quality_gate(
    *,
    parent_contract: pd.DataFrame,
    parent_before: pd.DataFrame,
    parent_after: pd.DataFrame,
    stage5_before: pd.DataFrame,
    stage5_after: pd.DataFrame,
    code_unchanged: bool,
    tests: Mapping[str, Any],
    preflight_checks: Sequence[Any],
    all_candidates: pd.DataFrame,
    build: Any,
    top3_identity: Mapping[str, Any],
    candidate_decision: Mapping[str, Any],
    final_solutions: Mapping[str, ScenarioSolution],
    solver_stage_rows: pd.DataFrame,
    final_run_certification: pd.DataFrame,
    greedy: Mapping[str, Any],
    seed_result: Any,
    seed_run_certification: pd.DataFrame,
    bundle_summary: pd.DataFrame,
    field: Mapping[str, Any],
    travel_contract: Any,
    travel_gated: pd.DataFrame,
    diagnostic_travel: pd.DataFrame,
    equity: Mapping[str, Any],
    frontier: pd.DataFrame,
    frontier_attempts: pd.DataFrame,
    frontier_attempt_stages: pd.DataFrame,
    frontier_run_certification: pd.DataFrame,
    marginal: pd.DataFrame,
    interface: pd.DataFrame,
    peak_rss_gib: float,
    base_config: Mapping[str, Any],
    final_config: Mapping[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    expected_parent_contracts = {
        "stage3_current_pointer",
        "stage4_current_pointer",
        "stage4_parent_metadata",
        "stage4_parent_inventory",
        "stage4_parent_interface",
        "stage4_parent_report",
        "stage4_base_config",
        "direction_log",
    }
    parent_contract_ok = bool(
        len(parent_contract) == 8
        and set(parent_contract["contract_name"].astype(str))
        == expected_parent_contracts
        and parent_contract["passed"].astype(bool).all()
    )
    rows.append(_gate("parent_declared_hashes", "provenance", parent_contract_ok, {"passed": int(parent_contract["passed"].astype(bool).sum()), "total": len(parent_contract), "names": sorted(parent_contract["contract_name"].astype(str))}, {"passed": 8, "total": 8, "names": sorted(expected_parent_contracts)}, "00_provenance/declared_parent_contract.csv", "Every named predeclared parent hash must match; omission is a hard failure."))
    highs_rows = solver_stage_rows[
        solver_stage_rows["engine"].astype(str).eq("scipy_highs")
    ].copy()
    retry_visit_counts = set(
        frontier_attempts.loc[
            frontier_attempts["attempt_number"].astype(int).eq(2), "visit_count"
        ].astype(int)
    )
    # SciPy/HiGHS owns a process-global scheduler.  Reusing the same worker
    # process with a different thread count (4 -> 8) can return HiGHS status
    # "Not Set" before optimization starts, so both initial and retry solves
    # use the same frozen four-thread budget.
    expected_highs_threads = np.full(len(highs_rows), 4, dtype=int)
    highs_runtime_ok = bool(
        not highs_rows.empty
        and highs_rows["requested_threads"].notna().all()
        and np.array_equal(
            highs_rows["requested_threads"].astype(int).to_numpy(),
            expected_highs_threads,
        )
        and highs_rows["requested_random_seed"].notna().all()
        and highs_rows["requested_random_seed"].astype(int).eq(42).all()
    )
    rows.append(
        _gate(
            "highs_runtime_options_applied_to_every_stage",
            "solver",
            highs_runtime_ok,
            {
                "stage_rows": len(highs_rows),
                "threads": sorted(
                    highs_rows["requested_threads"].dropna().astype(int).unique()
                ),
                "random_seeds": sorted(
                    highs_rows["requested_random_seed"].dropna().astype(int).unique()
                ),
            },
            {"standard_threads": 4, "sequential_retry_threads": 4, "random_seeds": [42]},
            "01_solver_certification/solver_stage_telemetry.csv",
            "Every official HiGHS stage must receive the frozen deterministic seed and the same four-thread budget; changing the process-global HiGHS scheduler between attempts is forbidden.",
        )
    )
    efficiency_highs_rows = highs_rows[
        highs_rows["scenario"].astype(str).eq("efficiency")
    ]
    attempt_efficiency_rows = frontier_attempt_stages[
        frontier_attempt_stages["scenario"].astype(str).eq("efficiency")
    ]
    greedy_floor_telemetry_ok = bool(
        not efficiency_highs_rows.empty
        and efficiency_highs_rows["minimum_total_population_floor"].notna().all()
        and efficiency_highs_rows["minimum_total_population_floor"]
        .astype(float)
        .gt(0)
        .all()
        and not attempt_efficiency_rows.empty
        and attempt_efficiency_rows["minimum_total_population_floor"].notna().all()
        and attempt_efficiency_rows["minimum_total_population_floor"]
        .astype(float)
        .gt(0)
        .all()
    )
    rows.append(
        _gate(
            "efficiency_greedy_floor_recorded_on_every_highs_stage",
            "solver",
            greedy_floor_telemetry_ok,
            {
                "official_efficiency_stage_rows": len(efficiency_highs_rows),
                "frontier_attempt_efficiency_stage_rows": len(
                    attempt_efficiency_rows
                ),
                "official_floor_values": sorted(
                    efficiency_highs_rows["minimum_total_population_floor"]
                    .dropna()
                    .astype(int)
                    .unique()
                ),
                "attempt_floor_values": sorted(
                    attempt_efficiency_rows["minimum_total_population_floor"]
                    .dropna()
                    .astype(int)
                    .unique()
                ),
            },
            {
                "all_efficiency_stages_have_positive_zero_tolerance_greedy_floor": True
            },
            "01_solver_certification/solver_stage_telemetry.csv",
            "Every HiGHS Efficiency stage, including failed/retried frontier attempts, must record the exact constrained-greedy population floor.",
        )
    )
    parents_equal = parent_before.equals(parent_after)
    rows.append(_gate("parent_inventory_tree_unchanged", "provenance", parents_equal, parents_equal, True, "00_provenance/frozen_parent_after.csv", "Stage 3 and provisional Stage 4 inventories must remain byte-identical."))
    stage5_namespace_ok = bool(
        stage5_before.empty and stage5_after.empty and stage5_before.equals(stage5_after)
    )
    rows.append(_gate(
        "stage5_namespace_absent_and_unchanged",
        "provenance",
        stage5_namespace_ok,
        {"before": len(stage5_before), "after": len(stage5_after)},
        {"before": 0, "after": 0},
        "00_provenance/forbidden_stage5_namespace_after.csv",
        "No Stage 5 output root or CURRENT_STAGE5 pointer may exist before or after Stage 4.1.",
    ))
    rows.append(_gate("finalization_code_contract_unchanged", "provenance", code_unchanged, code_unchanged, True, "00_provenance/finalization_code_contract.csv", "Tested finalizer code must not mutate during the run."))
    test_pass = int(tests["returncode"]) == 0 and int(tests["passed"]) > 0 and int(tests["failed"]) == 0
    rows.append(_gate("full_regression_tests_pass", "tests", test_pass, {key: tests[key] for key in ("returncode", "passed", "failed")}, {"returncode": 0, "passed": ">0", "failed": 0}, "00_provenance/regression_test_execution.json", "Full non-vacuous regression suite is mandatory."))
    memory_limit = float(final_config["runtime"]["memory_soft_limit_gib"])
    rows.append(_gate(
        "runtime_peak_rss_within_soft_limit",
        "runtime",
        math.isfinite(peak_rss_gib) and peak_rss_gib <= memory_limit,
        peak_rss_gib,
        {"max_gib": memory_limit},
        "00_provenance/runtime_environment.json",
        "The optimized sparse/compressed execution must remain within the frozen 24 GiB soft limit.",
    ))
    preflight_hard = [check for check in preflight_checks if str(check.severity).upper() == "HARD"]
    rows.append(_gate("stage3_preflight_hard_pass", "provenance", all(check.passed for check in preflight_hard), sum(check.passed for check in preflight_hard), len(preflight_hard), "00_provenance/stage3_preflight_quality_gate.csv", "Frozen Stage 3 handoff must pass its hard checks."))
    universe_ok = (
        len(all_candidates) == 3909
        and all_candidates["venue_id"].astype(str).nunique() == 3909
        and all_candidates["admin_code"].astype(str).nunique() == 153
    )
    rows.append(_gate("full_stage3_venue_universe_exact", "provenance", universe_ok, {"rows": len(all_candidates), "venues": int(all_candidates["venue_id"].astype(str).nunique()), "admins": int(all_candidates["admin_code"].astype(str).nunique())}, {"rows": 3909, "venues": 3909, "admins": 153}, "00_provenance/stage3_preflight_metadata.json", "Field fallback and diagnostic travel must use the full frozen Stage 3 venue universe."))

    preflight = build.preflight.set_index("candidate_set")
    for candidate_set, expected_count in {"top3": 453, "top5": 715, "coarse_pareto": 900}.items():
        observed = int(preflight.loc[candidate_set, "candidate_count"])
        rows.append(_gate(f"candidate_{candidate_set}_preflight", "candidate", bool(preflight.loc[candidate_set, "preflight_passed"]) and observed == expected_count, observed, expected_count, "02_candidate_sensitivity/candidate_set_preflight.csv", f"{candidate_set} must preserve all 153 admins and exact capacity."))
    cluster = preflight.loc["coverage_cluster"]
    cluster_ok = (
        not bool(cluster["available"])
        and not bool(cluster["preflight_passed"])
        and "maximum_matching=149/153" in str(cluster["unavailable_reason"])
    )
    rows.append(_gate("coverage_cluster_disposition_explicit", "candidate", cluster_ok, str(cluster["unavailable_reason"]), "ALL_ADMIN_REPRESENTATION_INFEASIBLE:149/153", "02_candidate_sensitivity/unavailable_candidate_sets.csv", "One representative per cluster is structurally unavailable and must not be silently relaxed."))
    top3_identity_ok = bool(
        top3_identity.get("exact_set_equal") is True
        and int(top3_identity.get("current_count", -1)) == 453
        and int(top3_identity.get("parent_count", -1)) == 453
        and float(top3_identity.get("venue_set_jaccard", -1)) == 1.0
        and top3_identity.get("current_venue_set_sha256")
        == top3_identity.get("parent_venue_set_sha256")
    )
    rows.append(_gate(
        "candidate_top3_equals_frozen_parent_universe",
        "candidate",
        top3_identity_ok,
        top3_identity,
        {"rows": 453, "jaccard": 1.0, "set_sha_equal": True},
        "02_candidate_sensitivity/top3_frozen_parent_identity.json",
        "The fail-closed Top3 fallback must be exactly the frozen parent candidate universe, not a newly tuned reducer output.",
    ))
    candidate_attempts_complete = bool(
        candidate_decision.get("candidate_solver_runs_complete")
        and int(candidate_decision.get("candidate_solver_run_count", -1)) == 6
    )
    rows.append(_gate(
        "candidate_solver_attempts_complete",
        "candidate",
        candidate_attempts_complete,
        {
            "run_count": candidate_decision.get("candidate_solver_run_count"),
            "complete": candidate_decision.get("candidate_solver_runs_complete"),
        },
        {"efficiency_and_balanced_for_three_universes": 6, "four_stage_telemetry": True},
        "02_candidate_sensitivity/candidate_solver_run_certification.csv",
        "All three candidate universes must be attempted under the same public solver contract.",
    ))
    selected_candidate_ok = bool(
        candidate_decision.get("selected_candidate_set")
        in {"top3", "top5", "coarse_pareto"}
        and candidate_decision.get("selected_universe_solver_certified") is True
    )
    rows.append(_gate(
        "selected_candidate_universe_solver_certified",
        "candidate",
        selected_candidate_ok,
        candidate_decision,
        "one selected universe with all four-stage reference/policy runs <=0.5%",
        "02_candidate_sensitivity/candidate_promotion_decision.json",
        "The field-validation anchor universe itself must be certified even when wider-universe promotion is inconclusive.",
    ))
    safe_disposition = bool(
        candidate_decision.get("candidate_reduction_certified") is True
        or (
            candidate_decision.get("decision")
            == "RETAIN_CERTIFIED_TOP3__EXPANSION_COMPARISON_INCONCLUSIVE"
            and candidate_decision.get("selected_candidate_set") == "top3"
            and candidate_decision.get("expansion_promotion_allowed") is False
            and candidate_decision.get("selected_universe_solver_certified") is True
        )
    )
    rows.append(_gate(
        "candidate_fail_closed_disposition",
        "candidate",
        safe_disposition,
        candidate_decision,
        "certified comparison or certified frozen Top3 retained with expansion promotion forbidden",
        "02_candidate_sensitivity/candidate_promotion_decision.json",
        "An uncertified expansion is an inconclusive comparison, never a reason to promote or to claim Top3 equivalence.",
    ))
    rows.append(_gate(
        "candidate_reduction_comparison_certified",
        "candidate",
        bool(candidate_decision.get("candidate_reduction_certified")),
        {
            "certified": candidate_decision.get("candidate_reduction_certified"),
            "uncertified_runs": candidate_decision.get("candidate_solver_uncertified_count"),
        },
        {"all_six_candidate_runs_certified": True},
        "02_candidate_sensitivity/candidate_solver_run_certification.csv",
        "Without certified Top5/coarse solves, candidate-reduction loss and Jaccard are diagnostic only; computational-final status is withheld.",
        ready=False,
        computational=True,
    ))

    exact_twenty = all(len(solution.selected) == 20 and solution.selected["venue_id"].astype(str).nunique() == 20 for solution in final_solutions.values())
    rows.append(_gate("final_policy_exact_twenty_unique_anchors", "solver", exact_twenty, {key: len(value.selected) for key, value in final_solutions.items()}, {"efficiency": 20, "balanced": 20, "equity": 20}, "05_policy_and_frontier/final_policy_scenario_metrics.csv", "Every policy scenario must contain exactly 20 unique anchors."))
    final_contract_complete = (
        len(final_run_certification) == 3
        and set(final_run_certification["scenario"].astype(str))
        == {"efficiency", "balanced", "equity"}
        and final_run_certification["stage_count"].astype(int).eq(4).all()
        and final_run_certification["run_role"].astype(str).eq(
            "final_n20_certification"
        ).all()
    )
    rows.append(_gate("final_solver_contract_complete", "solver", final_contract_complete, final_run_certification[["scenario", "stage_count", "run_role"]].to_dict("records"), {"scenarios": ["efficiency", "balanced", "equity"], "stages_each": 4}, "01_solver_certification/solver_run_certification.csv", "Final certification requires all three exact four-stage lexicographic runs."))
    certified = bool(
        final_contract_complete
        and final_run_certification["certified"].astype(bool).all()
    )
    rows.append(_gate("final_solver_runs_certified", "solver", certified, final_run_certification[["scenario", "certification_class", "max_relative_gap"]].to_dict("records"), "all lexicographic runs <=0.5% or optimal", "01_solver_certification/solver_run_certification.csv", "All final lexicographic stages require bound-based certification for computational-final wording.", ready=False, computational=True))
    rows.append(_gate("solver_bound_and_gap_recorded", "solver", not final_run_certification.empty and final_run_certification["max_relative_gap"].notna().all(), int(final_run_certification["max_relative_gap"].notna().sum()), len(final_run_certification), "01_solver_certification/solver_run_certification.csv", "Objective, bound, and directional gap must be public."))
    greedy_gate_ok = bool(
        greedy["passed"]
        and greedy["comparable"]
        and greedy["floor_enforced"]
        and int(greedy["final_efficiency_value"])
        >= int(greedy["effective_total_population_floor"])
        and str(greedy["floor_policy"])
        == "MAX_RETENTION_AND_CONSTRAINED_GREEDY_ZERO_TOLERANCE"
    )
    rows.append(_gate("efficiency_same_constraint_greedy_no_harm", "solver", greedy_gate_ok, greedy, {"passed": True, "comparable": True, "floor_enforced": True, "zero_tolerance": True}, "01_solver_certification/greedy_no_harm.csv", "The final lexicographic Efficiency policy must retain the larger of its frozen retention floor and the exact same-base-constraint greedy value."))

    rows.append(_gate("five_seed_contract_exact", "seed", set(seed_result.per_seed["seed"].astype(int)) == {11, 23, 42, 77, 101}, sorted(seed_result.per_seed["seed"].astype(int).tolist()), [11, 23, 42, 77, 101], "03_seed_stability/seed_scenario_metrics.csv", "Five predeclared CP-SAT seeds must be present exactly."))
    seed_contract_complete = (
        len(seed_run_certification) == 5
        and set(seed_run_certification["seed"].astype(int)) == {11, 23, 42, 77, 101}
        and seed_run_certification["stage_count"].astype(int).eq(4).all()
        and seed_run_certification["telemetry_complete"].astype(bool).all()
        and seed_run_certification["max_relative_gap"].notna().all()
    )
    rows.append(_gate("five_seed_telemetry_complete", "seed", seed_contract_complete, seed_run_certification[["seed", "stage_count", "telemetry_complete", "max_relative_gap"]].to_dict("records"), {"seeds": [11, 23, 42, 77, 101], "stages_each": 4, "telemetry_complete": True}, "03_seed_stability/seed_solver_run_certification.csv", "Every seed diagnostic must expose a complete four-stage CP-SAT trace and gap."))
    rows.append(_gate("seed_stability_acceptable", "seed", bool(seed_result.passed), seed_result.classification, ["PASS_SEED_STABLE", "PASS_CLUSTER_STABLE_VENUE_VARIABLE"], "03_seed_stability/seed_stability_decision.json", "Uncertified CP-SAT seed frequencies are diagnostic and excluded from queue scoring because HiGHS is the primary solver.", ready=False, severity="ADVISORY"))

    bundle_min1 = bundle_summary[
        bundle_summary["minimum_per_bundle"].astype(int).eq(1)
    ]
    bundle_ok = bool(
        set(bundle_summary["minimum_per_bundle"].astype(int)) == {0, 1}
        and len(bundle_summary) == 2
        and bundle_summary["selected_venue_count"].astype(int).eq(20).all()
        and bundle_summary["assignment_complete"].astype(bool).all()
        and bundle_summary["bundle_representation_floor_met"].astype(bool).all()
        and bundle_summary["spatial_sensitivity_status"].astype(str).eq(
            "NOT_APPLICABLE_FIXED_ANCHORS"
        ).all()
        and len(bundle_min1) == 1
        and int(bundle_min1.iloc[0]["bundle_count_unique"]) == 5
        and int(bundle_min1.iloc[0]["minimum_observed_bundle_count"]) >= 1
        and np.isfinite(
            pd.to_numeric(
                bundle_summary["total_bundle_value_delta_min1_minus_min0"],
                errors="coerce",
            )
        ).all()
    )
    rows.append(_gate(
        "bundle_minimum_sensitivity_complete",
        "bundle",
        bundle_ok,
        bundle_summary.to_dict("records"),
        {
            "minimums": [0, 1],
            "fixed_anchor_scope": "NOT_APPLICABLE_FIXED_ANCHORS",
            "complete_assignments": 20,
            "min1_all_five_bundles": True,
        },
        "04_bundle_sensitivity/bundle_sensitivity_metrics.csv",
        "Bundle min=0/1 is an assignment-only test on the same 20 anchors; it must not be presented as spatial re-optimization.",
    ))

    queue = field["queue"]
    form = field["form"]
    fallbacks = field["fallbacks"]
    resolution = field["resolution"]
    core = field["robust_core"]
    queue_ok = 30 <= len(queue) <= 60 and queue["venue_id"].astype(str).nunique() == len(queue)
    rows.append(_gate("field_validation_queue_complete", "field", queue_ok, len(queue), "30..60 unique venues", "06_field_readiness/field_validation_priority_queue.csv", "The queue must contain anchors, usable same-cluster fallbacks, and priority evidence."))
    inherited_scope_columns = [
        "catchment_stability_evidence_scope",
        "cluster_stability_evidence_scope",
    ]
    inherited_promotion_columns = [
        "catchment_stability_final_promotion_allowed",
        "cluster_stability_final_promotion_allowed",
    ]
    inherited_scope_ok = bool(
        all(column in queue for column in inherited_scope_columns)
        and all(column in queue for column in inherited_promotion_columns)
        and all(
            set(queue[column].astype(str))
            <= {"INHERITED_PROVISIONAL_TOP3_DIAGNOSTIC", "NOT_MEASURED"}
            for column in inherited_scope_columns
        )
        and not queue[inherited_promotion_columns].astype(bool).any().any()
    )
    rows.append(_gate(
        "field_queue_inherited_stability_scope_explicit",
        "field",
        inherited_scope_ok,
        {
            column: sorted(queue[column].astype(str).unique())
            for column in inherited_scope_columns
            if column in queue
        },
        {
            "scope": "INHERITED_PROVISIONAL_TOP3_DIAGNOSTIC",
            "final_promotion_allowed": False,
        },
        "06_field_readiness/field_validation_priority_queue.csv",
        "Inherited Top3 catchment/cluster stability is a queue-priority diagnostic, never final promotion evidence.",
    ))
    primary_ids = set(final_solutions["balanced"].selected["venue_id"].astype(str))
    queue_ids = set(queue["venue_id"].astype(str))
    rows.append(_gate("all_primary_anchors_in_field_queue", "field", primary_ids <= queue_ids and len(primary_ids) == 20, len(primary_ids & queue_ids), 20, "06_field_readiness/field_validation_priority_queue.csv", "Every promoted spatial anchor must be queued for field verification."))
    expected_fallback_ids = {
        str(value)
        for column in ("actual_facility_fallback_1", "actual_facility_fallback_2")
        for value in fallbacks[column].dropna().astype(str)
    }
    observed_fallback_ids = set(
        queue.loc[queue["is_primary_fallback"].astype(bool), "venue_id"].astype(str)
    )
    rows.append(_gate("field_queue_fallback_set_exact", "field", expected_fallback_ids == observed_fallback_ids, {"resolver": len(expected_fallback_ids), "queue": len(observed_fallback_ids)}, "exact set equality", "06_field_readiness/actual_facility_fallbacks.csv", "Resolver fallbacks and mandatory queue fallbacks must be the same venue-ID set."))
    field_columns = list(final_config["field_validation"]["expected_field_columns"])
    response_columns = [column for column in field_columns if column != "verification_date"]
    unknown_ok = (
        all(column in form.columns for column in field_columns)
        and all(form[column].astype(str).eq("UNKNOWN").all() for column in response_columns)
        and form["verification_date"].astype(str).eq("UNKNOWN").all()
        and not form["field_verified"].astype(bool).any()
    )
    rows.append(_gate("field_form_unknown_fail_closed", "field", unknown_ok, {"rows": len(form), "verified": int(form["field_verified"].astype(bool).sum())}, {"all_responses": "UNKNOWN", "verified": 0}, "06_field_readiness/field_validation_form.csv", "An unreturned form is a request for evidence, never verified feasibility."))
    rows.append(_gate("robust_core_status_explicit", "field", core["robust_core_status"] in {"PRESENT", "NONE"}, core, "PRESENT or NONE", "06_field_readiness/robust_core_summary.json", "An empty robust core is a valid explicit result, not missing evidence."))
    unverified_final = int(resolution["final_physical_venue_id"].notna().sum())
    unresolved = int(resolution["venue_unresolved"].astype(bool).sum())
    bus_ok = unverified_final == 0 and unresolved == 20
    rows.append(_gate("busproxy_unverified_not_final", "field", bus_ok, {"resolved": unverified_final, "unresolved": unresolved}, {"resolved_without_field_evidence": 0, "unresolved": 20}, "06_field_readiness/physical_venue_resolution.csv", "No BUSPROXY or other unverified venue may become a final physical venue."))

    diagnostic_ok, diagnostic_counts = _diagnostic_travel_gate_evidence(
        diagnostic_travel,
        expected_scenarios=final_config["travel"]["diagnostic_scenarios"],
        expected_venue_count=3909,
    )
    travel_ok = (
        travel_contract.evidence_status == "UNCONFIRMED"
        and not travel_contract.final_travel_objective_enabled
        and diagnostic_ok
    )
    rows.append(_gate("travel_evidence_not_overclaimed", "travel", travel_ok, travel_contract.as_dict(), {"evidence_status": "UNCONFIRMED", "final_enabled": False}, "06_field_readiness/mobile_team_base_contract.json", "Assumed medical-center bases remain diagnostic and never enter final optimization."))
    gated_travel_ok = (
        len(travel_gated) == 20
        and travel_gated["travel_minutes"].isna().all()
        and travel_gated["final_travel_minutes"].isna().all()
        and not travel_gated["final_travel_objective_enabled"].astype(bool).any()
    )
    rows.append(_gate("unconfirmed_final_travel_disabled", "travel", gated_travel_ok, {"rows": len(travel_gated), "travel_nonnull": int(travel_gated["travel_minutes"].notna().sum()), "final_nonnull": int(travel_gated["final_travel_minutes"].notna().sum()), "enabled": int(travel_gated["final_travel_objective_enabled"].astype(bool).sum())}, {"rows": 20, "travel_nonnull": 0, "final_nonnull": 0, "enabled": 0}, "07_interface/stage4_1_to_field_validation.csv", "NaN travel must stay disabled rather than being converted to zero."))
    rows.append(_gate("diagnostic_travel_full_universe_only", "travel", diagnostic_ok, {"rows": len(diagnostic_travel), "scenario_venues": diagnostic_counts}, {"rows": 11727, "scenario_venues": 3909}, "06_field_readiness/diagnostic_travel_sensitivity.parquet", "All three assumed-base scenarios must cover the full venue universe and remain diagnostic-only."))
    beneficiary_values = list(map(float, equity["beneficiary_sigungu_coverage"].values()))
    equity_ok = (
        len(beneficiary_values) == 11
        and all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in beneficiary_values)
        and math.isfinite(float(equity["beneficiary_coverage_gini"]))
        and 0.0 <= float(equity["beneficiary_coverage_gini"]) <= 1.0
        and math.isfinite(float(equity["visit_location_gini"]))
        and 0.0 <= float(equity["visit_location_gini"]) <= 1.0
        and float(equity["beneficiary_covered_population"]) <= float(equity["beneficiary_total_population"]) + 1e-8
        and sum(map(int, equity["visit_location_distribution"].values())) == 20
    )
    rows.append(_gate("beneficiary_and_visit_location_equity_valid", "equity", equity_ok, equity, {"sigungu": 11, "visits": 20, "finite_ranges": True}, "06_field_readiness/beneficiary_visit_location_equity.json", "Beneficiary coverage and visit-location equity use the complete 11-sigungu universe, including zero-visit sigungu."))
    expected_visits = {5, 10, 15, 20, 25, 30, 40}
    frontier_pairs = set(
        zip(frontier["scenario"].astype(str), frontier["visit_count"].astype(int))
    )
    expected_pairs = {
        (scenario, visits)
        for scenario in ("efficiency", "balanced")
        for visits in expected_visits
    }
    frontier_ok = (
        len(frontier) == 14
        and frontier_pairs == expected_pairs
        and not frontier.duplicated(["scenario", "visit_count"]).any()
        and len(frontier_run_certification) == 14
        and frontier_run_certification["stage_count"].astype(int).eq(4).all()
        and frontier_run_certification["telemetry_complete"].astype(bool).all()
        and frontier_run_certification["max_relative_gap"].notna().all()
    )
    rows.append(_gate("visit_count_frontier_complete", "frontier", frontier_ok, {"rows": len(frontier), "pairs": sorted(frontier_pairs), "certification_rows": len(frontier_run_certification)}, {"rows": 14, "scenario_visit_pairs": 14, "four_stage_telemetry": True}, "05_policy_and_frontier/visit_count_frontier.csv", "Moderate-budget diagnostic frontier must contain both scenarios at every visit count with public bounds and gaps."))
    attempt_keys = set(
        zip(
            frontier_attempts["visit_count"].astype(int),
            frontier_attempts["attempt_number"].astype(int),
        )
    )
    first_attempts = frontier_attempts[
        frontier_attempts["attempt_number"].astype(int).eq(1)
    ]
    retry_attempts = frontier_attempts[
        frontier_attempts["attempt_number"].astype(int).eq(2)
    ]
    final_attempts = (
        frontier_attempts.sort_values("attempt_number")
        .groupby("visit_count", as_index=False)
        .tail(1)
    )
    retry_preceded_by_failure = True
    for retry in retry_attempts.itertuples(index=False):
        prior = frontier_attempts[
            frontier_attempts["visit_count"].astype(int).eq(int(retry.visit_count))
            & frontier_attempts["attempt_number"].astype(int).eq(1)
        ]
        retry_preceded_by_failure = retry_preceded_by_failure and bool(
            len(prior) == 1 and not bool(prior.iloc[0]["success"])
        )
    initial_non_n20 = first_attempts[
        ~first_attempts["visit_count"].astype(int).eq(20)
    ]
    initial_n20 = first_attempts[
        first_attempts["visit_count"].astype(int).eq(20)
    ]
    failed_attempts = frontier_attempts[~frontier_attempts["success"].astype(bool)]
    expected_model_contract_sha256 = _frontier_model_contract_sha256(
        build.inputs[str(candidate_decision["selected_candidate_set"])],
        base_config,
        final_config,
        str(candidate_decision["selected_candidate_set"]),
    )
    stage_attempt_keys = set(
        zip(
            frontier_attempt_stages["visit_count"].astype(int),
            frontier_attempt_stages["frontier_attempt_number"].astype(int),
        )
    )
    attempts_ok = bool(
        len(first_attempts) == 7
        and set(first_attempts["visit_count"].astype(int)) == expected_visits
        and len(attempt_keys) == len(frontier_attempts)
        and frontier_attempts["attempt_number"].astype(int).between(1, 2).all()
        and len(final_attempts) == 7
        and final_attempts["success"].astype(bool).all()
        and retry_attempts["attempt_kind"]
        .astype(str)
        .eq("SEQUENTIAL_NO_INCUMBENT_RETRY")
        .all()
        and retry_attempts["time_limit_per_stage_sec"]
        .astype(float)
        .eq(float(final_config["runtime"]["frontier_retry_time_per_stage_sec"]))
        .all()
        and retry_attempts["worker_count_for_attempt"].astype(int).eq(1).all()
        and retry_attempts["requested_highs_threads"].astype(int).eq(4).all()
        and retry_attempts["retry_trigger"].astype(str).eq("NO_INCUMBENT").all()
        and initial_non_n20["attempt_kind"].astype(str).eq("INITIAL_PARALLEL").all()
        and initial_non_n20["time_limit_per_stage_sec"].astype(float).eq(30.0).all()
        and initial_non_n20["worker_count_for_attempt"].astype(int).eq(2).all()
        and initial_non_n20["requested_highs_threads"].astype(int).eq(4).all()
        and not initial_non_n20["reused_final_n20_solution"].astype(bool).any()
        and len(initial_n20) == 1
        and str(initial_n20.iloc[0]["attempt_kind"]) == "REUSED_FINAL_N20"
        and float(initial_n20.iloc[0]["time_limit_per_stage_sec"]) == 0.0
        and bool(initial_n20.iloc[0]["reused_final_n20_solution"])
        and int(initial_n20.iloc[0]["requested_highs_threads"]) == 4
        and failed_attempts["failure_scenario"].astype(str).str.len().gt(0).all()
        and failed_attempts["failure_objective"].astype(str).str.len().gt(0).all()
        and failed_attempts["failure_status"].astype(str).eq("UNKNOWN").all()
        and failed_attempts["failure_message"]
        .astype(str)
        .str.lower()
        .str.contains("time limit")
        .all()
        and frontier_attempts["model_contract_sha256"]
        .astype(str)
        .eq(expected_model_contract_sha256)
        .all()
        and not frontier_attempts["final_promotion_allowed"].astype(bool).any()
        and retry_preceded_by_failure
        and attempt_keys.issubset(stage_attempt_keys)
        and frontier_attempt_stages["model_contract_sha256"]
        .astype(str)
        .eq(expected_model_contract_sha256)
        .all()
        and frontier_attempt_stages["requested_threads"].notna().all()
        and np.array_equal(
            frontier_attempt_stages["requested_threads"].astype(int).to_numpy(),
            np.full(len(frontier_attempt_stages), 4, dtype=int),
        )
        and frontier_attempt_stages["requested_random_seed"].notna().all()
        and frontier_attempt_stages["requested_random_seed"]
        .astype(int)
        .eq(42)
        .all()
        and not frontier_attempt_stages["final_promotion_allowed"].astype(bool).any()
    )
    rows.append(
        _gate(
            "frontier_no_incumbent_retry_contract",
            "frontier",
            attempts_ok,
            {
                "attempt_rows": len(frontier_attempts),
                "retry_rows": len(retry_attempts),
                "attempt_stage_rows": len(frontier_attempt_stages),
                "model_contract_sha256": expected_model_contract_sha256,
                "final_successful_visit_counts": sorted(
                    final_attempts.loc[
                        final_attempts["success"].astype(bool), "visit_count"
                    ].astype(int)
                ),
            },
            {
                "initial_visit_counts": sorted(expected_visits),
                "maximum_attempts": 2,
                "retry_trigger": "NO_INCUMBENT",
                "retry_budget_sec": 120,
                "retry_workers": 1,
                "initial_highs_threads_per_task": 4,
                "retry_highs_threads": 4,
                "single_model_contract_sha256": True,
                "failed_attempt_stage_telemetry": True,
                "final_successful_visit_counts": sorted(expected_visits),
            },
            "05_policy_and_frontier/visit_count_frontier_attempts.csv",
            "A no-incumbent frontier solve may receive one sequential 120-second retry without changing objectives, thresholds, or promotion status.",
        )
    )
    expected_intervals = {(5, 10), (10, 15), (15, 20), (20, 25), (25, 30), (30, 40)}
    marginal_ok = (
        len(marginal) == 12
        and set(zip(marginal["from_visit_count"].astype(int), marginal["to_visit_count"].astype(int))) == expected_intervals
        and set(marginal["scenario"].astype(str)) == {"efficiency", "balanced"}
        and not marginal.duplicated(["scenario", "from_visit_count", "to_visit_count"]).any()
        and marginal["interpretation_status"]
        .astype(str)
        .isin({"CERTIFIED_ENDPOINTS", "INCONCLUSIVE_SOLVER_GAP"})
        .all()
        and (
            marginal["diminishing_return_claim_allowed"].astype(bool)
            == (
                marginal["from_certified"].astype(bool)
                & marginal["to_certified"].astype(bool)
            )
        ).all()
        and not marginal["final_promotion_allowed"].astype(bool).any()
    )
    rows.append(_gate("visit_marginal_benefit_complete", "frontier", marginal_ok, {"rows": len(marginal), "intervals": sorted(expected_intervals), "claim_allowed_rows": int(marginal["diminishing_return_claim_allowed"].astype(bool).sum())}, {"rows": 12, "intervals_per_scenario": 6, "uncertified_endpoint_claims_forbidden": True}, "05_policy_and_frontier/visit_marginal_benefit.csv", "All six adjacent visit-count increments must be quantified for both policies; diminishing-return interpretation is allowed only when both endpoint solves are certified."))
    n20 = frontier[frontier["visit_count"].astype(int).eq(20)].set_index("scenario")
    n20_metrics = [
        "unique_elderly_population",
        "need_weighted_population_scaled",
        "high_need_population_scaled",
        "min_sigungu_coverage_ratio",
    ]
    n20_deltas: dict[str, float] = {}
    for scenario in ("efficiency", "balanced"):
        for metric in n20_metrics:
            n20_deltas[f"{scenario}:{metric}"] = abs(
                float(n20.loc[scenario, metric])
                - float(final_solutions[scenario].metrics[metric])
            )
    n20_ok = (
        len(n20) == 2
        and n20["reused_final_n20_solution"].astype(bool).all()
        and max(n20_deltas.values(), default=math.inf) <= 1e-12
    )
    rows.append(_gate("frontier_n20_reuses_final_solution_exactly", "frontier", n20_ok, n20_deltas, {"max_abs_delta": 0.0, "reused": True}, "05_policy_and_frontier/visit_count_frontier.csv", "The n=20 frontier point must be the final run, preventing solver-search variability from masquerading as policy sensitivity."))

    forbidden_nonnull = int(interface[["recommended_season", "fallback_season", "exact_month", "exact_date"]].notna().sum().sum())
    rows.append(_gate("temporal_and_exact_date_fields_null", "handoff", forbidden_nonnull == 0, forbidden_nonnull, 0, "07_interface/stage4_1_to_field_validation.csv", "Stage 2B remains abstention and Stage 5 scheduling has not begun."))
    interface_ok = (
        len(interface) == 20
        and interface["visit_id"].astype(str).nunique() == 20
        and interface["venue_id"].astype(str).nunique() == 20
        and not interface["stage5_release_allowed"].astype(bool).any()
        and not interface["stage5_started"].astype(bool).any()
    )
    rows.append(_gate("stage5_not_started", "handoff", final_config.get("stage5_started") is False and interface_ok, {"config_stage5_started": final_config.get("stage5_started"), "rows": len(interface), "unique_visits": int(interface["visit_id"].astype(str).nunique()), "release_allowed_count": int(interface["stage5_release_allowed"].astype(bool).sum()), "stage5_started_count": int(interface["stage5_started"].astype(bool).sum())}, {"stage5_started": False, "rows": 20, "unique_visits": 20, "release_allowed": 0}, "07_interface/stage5_readiness_checklist.json", "Stage 5 remains blocked pending verified venues, teams, vehicles, calendar, and availability."))

    # Operational-final checks are disclosed as expected failures, not hidden.
    rows.append(_gate("all_twenty_physical_venues_field_verified", "operational", unresolved == 0, {"unresolved": unresolved}, {"unresolved": 0}, "06_field_readiness/physical_venue_resolution.csv", "Operational final requires every anchor to resolve to a verified facility.", ready=False, computational=False, operational=True))
    rows.append(_gate("confirmed_team_base_travel_available", "operational", bool(travel_contract.final_travel_objective_enabled), travel_contract.as_dict(), {"final_travel_objective_enabled": True}, "06_field_readiness/mobile_team_base_contract.json", "Operational final requires confirmed team-base network travel.", ready=False, computational=False, operational=True))
    rows.append(_gate("vehicle_calendar_and_availability_confirmed", "operational", False, {"vehicle": "UNAVAILABLE", "calendar": "UNAVAILABLE", "venue_availability": "UNAVAILABLE"}, {"all": "CONFIRMED"}, "06_field_readiness/required_operational_data.json", "This Stage 4.1 runner cannot claim operational final without vehicles, calendar, and venue availability.", ready=False, computational=False, operational=True))
    return pd.DataFrame(rows)


def _forbidden_stage5_snapshot(
    package_root: Path, config: Mapping[str, Any]
) -> pd.DataFrame:
    """Inventory forbidden Stage 5 namespaces without flagging Stage 4 handoffs."""

    rows: list[dict[str, Any]] = []
    paths = _require_mapping(config.get("paths"), "paths")
    for relative in paths["forbidden_stage5_roots"]:
        root = (package_root / str(relative)).resolve()
        try:
            root.relative_to(package_root.resolve())
        except ValueError as exc:
            raise RuntimeError("Forbidden Stage 5 root escapes the package") from exc
        if root.is_file():
            files = [root]
        elif root.is_dir():
            files = sorted(path for path in root.rglob("*") if path.is_file())
        else:
            files = []
        for path in files:
            rows.append(
                {
                    "detection_rule": f"root:{relative}",
                    "relative_path": ensure_relative_to(path, package_root),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    for base in (package_root / "outputs/model_v1", package_root / "reports/model_v1"):
        if not base.exists():
            continue
        for path in sorted(base.rglob("CURRENT_STAGE5*.json")):
            if path.is_file():
                rows.append(
                    {
                        "detection_rule": "current_pointer_pattern",
                        "relative_path": ensure_relative_to(path, package_root),
                        "size_bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                )
    columns = ["detection_rule", "relative_path", "size_bytes", "sha256"]
    if not rows:
        return pd.DataFrame(columns=columns)
    return (
        pd.DataFrame(rows, columns=columns)
        .drop_duplicates("relative_path")
        .sort_values("relative_path")
        .reset_index(drop=True)
    )


def _release_decision(gates: pd.DataFrame) -> dict[str, Any]:
    def failures(column: str) -> list[str]:
        mask = gates[column].astype(bool) & ~gates["passed"].astype(bool)
        return gates.loc[mask, "check_id"].astype(str).tolist()

    ready_failures = failures("required_for_ready_field_validation")
    computational_failures = failures("required_for_computational_final")
    operational_failures = failures("required_for_operational_final")
    if ready_failures:
        decision = "FAIL_STAGE4_FINALIZATION"
    else:
        decision = "PASS_STAGE4_READY_FOR_FIELD_VALIDATION"
    candidate_expansion_inconclusive = (
        "candidate_reduction_comparison_certified" in computational_failures
    )
    return {
        "decision": decision,
        "stage4_provisional": True,
        "stage4_spatial_status": (
            (
                "READY_FOR_FIELD_VALIDATION__CANDIDATE_EXPANSION_INCONCLUSIVE"
                if candidate_expansion_inconclusive
                else "READY_FOR_FIELD_VALIDATION"
            )
            if not ready_failures
            else "FINALIZATION_FAILED"
        ),
        "stage4_operational_provisional": True,
        "stage4_computational_complete": not ready_failures and not computational_failures,
        "stage4_ready_for_field_validation": not ready_failures,
        "stage4_operational_final": (
            not ready_failures and not computational_failures and not operational_failures
        ),
        "stage5_started": False,
        "stage5_release_allowed": False,
        "ready_failures": ready_failures,
        "computational_failures": computational_failures,
        "operational_failures": operational_failures,
    }


def _report(
    *,
    run_id: str,
    release: Mapping[str, Any],
    final_solutions: Mapping[str, ScenarioSolution],
    final_certification: pd.DataFrame,
    greedy_no_harm: Mapping[str, Any],
    candidate_decision: Mapping[str, Any],
    candidate_comparisons: pd.DataFrame,
    seed_result: Any,
    bundle_summary: pd.DataFrame,
    robust_core: Mapping[str, Any],
    field_queue: pd.DataFrame,
    resolution: pd.DataFrame,
    travel_contract: Any,
    diagnostic_travel: pd.DataFrame,
    equity: Mapping[str, Any],
    frontier: pd.DataFrame,
    frontier_attempts: pd.DataFrame,
    marginal: pd.DataFrame,
    gates: pd.DataFrame,
) -> str:
    scenario_rows = pd.DataFrame(
        [
            {
                "정책": name,
                "상태": solution.status,
                "고유 고령인구": round(float(solution.metrics["unique_elderly_population"]), 2),
                "Need 가중": int(solution.metrics["need_weighted_population_scaled"]),
                "High-Need": int(solution.metrics["high_need_population_scaled"]),
                "최소 시군 coverage": round(float(solution.metrics["min_sigungu_coverage_ratio"]), 6),
                "중복률": round(float(solution.metrics["redundancy_ratio"]), 6),
            }
            for name, solution in final_solutions.items()
        ]
    )
    failed = gates[~gates["passed"].astype(bool)][["check_id", "gate_group", "severity", "message"]]
    diagnostic_selected = diagnostic_travel[
        diagnostic_travel["venue_id"].astype(str).isin(
            final_solutions["balanced"].selected["venue_id"].astype(str)
        )
    ]
    travel_summary = (
        diagnostic_selected.groupby("travel_scenario")["travel_minutes"]
        .median()
        .rename("median_minutes")
        .reset_index()
    )
    minimum_after_20 = marginal[marginal["from_visit_count"].eq(20)]
    return f"""# MEDIROAD MODEL V1 — Stage 4.1 Finalization Report

- run_id: `{run_id}`
- release: **{release['decision']}**
- computational complete: `{release['stage4_computational_complete']}`
- operational final: `{release['stage4_operational_final']}`
- Stage 5 release allowed: `{release['stage5_release_allowed']}`

## 1. 방향 수정

기존 Stage 4는 453개 Top3 후보에 대한 20초 CP-SAT feasible incumbent였습니다. 이번 finalization은 lossless coverage-pattern 압축, 독립 SciPy/HiGHS bound, Top3/Top5/coarse-Pareto 비교, 5-seed 분리, bundle min=0/1, BUSPROXY fail-closed 처리와 현장검증 queue를 추가했습니다. Stage 1–3과 기존 provisional Stage 4는 수정하지 않았습니다.

## 2. 최종 정책 시나리오

{scenario_rows.to_markdown(index=False)}

## 3. Solver 인증

{final_certification[['scenario','certification_class','certified','max_relative_gap','telemetry_complete']].to_markdown(index=False)}

- constrained-greedy population floor: `{int(greedy_no_harm['constrained_greedy_floor'])}`
- frozen retention floor: `{int(greedy_no_harm['retention_floor'])}`
- effective zero-tolerance floor: `{int(greedy_no_harm['effective_total_population_floor'])}`
- final Efficiency population: `{int(greedy_no_harm['final_efficiency_value'])}`
- final Efficiency greedy no-harm: `{bool(greedy_no_harm['passed'])}`

`OPTIMAL`은 실제 primal=dual일 때만, `CERTIFIED_NEAR_OPTIMAL`은 사전 고정 0.5% 이내일 때만 사용했습니다. 나머지는 `FEASIBLE_UNCERTIFIED`로 공개합니다.

## 4. Candidate reduction

- decision: `{candidate_decision['decision']}`
- selected universe: `{candidate_decision['selected_candidate_set']}`
- contract status: `{candidate_decision['contract_status']}`
- selected universe solver certified: `{candidate_decision['selected_universe_solver_certified']}`
- candidate reduction certified: `{candidate_decision['candidate_reduction_certified']}`
- expanded comparison conclusive: `{candidate_decision['expanded_comparison_conclusive']}`
- expansion promotion allowed: `{candidate_decision['expansion_promotion_allowed']}`
- strict one-representative coverage-cluster universe: unavailable (maximum matching 149/153)

`DIAGNOSTIC_UNCERTIFIED_INCUMBENT`로 표시된 Top5/coarse 결과는 탐색 병목을
공개하기 위한 값이며, Top3의 동등성·우월성 또는 사전 loss threshold 충족을
증명하지 않습니다. 이 경우에는 인증된 frozen Top3만 현장검증 기준안으로
유지하고 computational-final 표현을 차단합니다.

{candidate_comparisons.to_markdown(index=False)}

## 5. Seed stability

- classification: `{seed_result.classification}`
- passed: `{seed_result.passed}`
- exact seeds: `11, 23, 42, 77, 101`

{seed_result.metric_spread.to_markdown(index=False)}

## 6. Bundle min=0/1

공간 20곳은 고정하고 진료 bundle 배정만 비교했습니다. primary policy는 사전 선언한 min=1이며, 민감도 결과가 공간 선택의 근거로 역류하지 않습니다.

{bundle_summary.to_markdown(index=False)}

## 7. BUSPROXY와 현장검증

- robust core: `{robust_core['robust_core_status']}` (threshold `{robust_core['robust_core_threshold']}`, max frequency `{robust_core['maximum_selection_frequency']:.3f}`)
- field-validation queue: `{len(field_queue)}`곳
- unresolved physical venues: `{int(resolution['venue_unresolved'].sum())}/20`
- unverified final physical venues: `{int(resolution['final_physical_venue_id'].notna().sum())}`

BUSPROXY는 spatial anchor로만 남으며, 12개 현장필드가 모두 확인되기 전에는 어떤 anchor도 final physical venue로 확정하지 않았습니다.

## 8. Travel과 equity

- team-base evidence: `{travel_contract.evidence_status}`
- final travel objective enabled: `{travel_contract.final_travel_objective_enabled}`
- beneficiary min sigungu coverage: `{float(equity['beneficiary_min_sigungu_coverage']):.6f}`
- beneficiary coverage Gini: `{float(equity['beneficiary_coverage_gini']):.6f}`
- visit-location Gini (0회 시군 포함): `{float(equity['visit_location_gini']):.6f}`

가정한 두 의료원 출발지는 진단 전용입니다.

{travel_summary.to_markdown(index=False)}

## 9. Visit-count frontier

첫 공식 시도에서는 efficiency n=25가 30초 stage budget 안에 incumbent를
만들지 못해 전체 run이 fail-closed 되었습니다. 아래 ledger는 모든 최초
시도와, no-incumbent에만 허용한 1회 sequential 120초 retry를 공개합니다.
Retry는 objective plan·sense·retention formula·constraint family·후보군·
0.5% 인증 기준·승격 상태를 바꾸지 않습니다. 각 attempt에서 실제로 달라질
수 있는 incumbent와 그로부터 계산된 numeric floor는 stage telemetry에 그대로
공개하며 동일하다고 주장하지 않습니다.

{frontier_attempts.to_markdown(index=False)}

{frontier.to_markdown(index=False)}

20회 이후 첫 marginal 구간:

두 endpoint가 모두 solver-certified인 행만 diminishing-return 정책 해석을
허용합니다. `INCONCLUSIVE_SOLVER_GAP` 행은 수치를 공개하되 감소효과나 최적
방문횟수의 근거로 사용하지 않습니다.

{minimum_after_20.to_markdown(index=False)}

## 10. Gate와 남은 병목

- ready-for-field-validation failures: `{len(release['ready_failures'])}`
- computational-final failures: `{len(release['computational_failures'])}`
- operational-final failures: `{len(release['operational_failures'])}`

{failed.to_markdown(index=False)}

현 상태는 현장검증을 시작할 수 있는 계산·증거 package입니다. 실제 시설, 팀 출발지, 차량, 운영 calendar가 확인되지 않았으므로 operational final이나 Stage 5 일정으로 표현하지 않습니다.
"""


def run_stage4_finalization(
    package_root: Path,
    *,
    finalization_config_path: Path | None = None,
) -> Path:
    package_root = package_root.resolve()
    finalization_config_path = (
        finalization_config_path
        or package_root / "configs/model_v1/stage4_finalization.yaml"
    ).resolve()
    final_config = load_finalization_config(finalization_config_path)
    base_config = load_config(package_root / "configs/model_v1/stage4.yaml")
    runtime_versions = _dependency_versions(
        final_config["runtime_environment"]["exact_package_versions"]
    )
    code_before = _sha_contract(package_root, CODE_CONTRACT_PATHS)
    config_sha = sha256_file(finalization_config_path)
    started_at = utc_now_iso()
    run_token = hashlib.sha256(
        stable_json_dumps(
            {"started_at": started_at, "config_sha256": config_sha, "code": code_before}
        ).encode("utf-8")
    ).hexdigest()[:12]
    run_id = f"stage4_1_{started_at.replace('-', '').replace(':', '')}_{run_token}"
    output_root = package_root / str(final_config["paths"]["output_root"])
    run_root = output_root / "runs" / run_id
    running_marker = run_root / ".RUNNING"
    lock_path = package_root / str(final_config["paths"]["lock"])
    pipeline_started = time.perf_counter()
    run_started = False

    try:
        with single_writer_lock(lock_path):
            if run_root.exists():
                raise FileExistsError(run_root)
            run_root.mkdir(parents=True, exist_ok=False)
            atomic_write_json(
                running_marker,
                {"run_id": run_id, "pid": os.getpid(), "started_at": started_at},
            )
            run_started = True
            provenance_dir = run_root / "00_provenance"
            solver_dir = run_root / "01_solver_certification"
            candidate_dir = run_root / "02_candidate_sensitivity"
            seed_dir = run_root / "03_seed_stability"
            bundle_dir = run_root / "04_bundle_sensitivity"
            policy_dir = run_root / "05_policy_and_frontier"
            field_dir = run_root / "06_field_readiness"
            interface_dir = run_root / "07_interface"
            for directory in (
                provenance_dir,
                solver_dir,
                candidate_dir,
                seed_dir,
                bundle_dir,
                policy_dir,
                field_dir,
                interface_dir,
            ):
                directory.mkdir(parents=True, exist_ok=True)

            parent_contract = _verify_declared_parents(package_root, final_config)
            parent_before = capture_parent_tree(package_root, final_config)
            stage5_before = _forbidden_stage5_snapshot(package_root, final_config)
            atomic_write_csv(parent_contract, provenance_dir / "declared_parent_contract.csv")
            atomic_write_csv(parent_before, provenance_dir / "frozen_parent_before.csv")
            atomic_write_csv(
                stage5_before,
                provenance_dir / "forbidden_stage5_namespace_before.csv",
            )
            _atomic_yaml(provenance_dir / "stage4_finalization_config_snapshot.yaml", final_config)
            atomic_write_json(
                provenance_dir / "parent_stage4_contract.json",
                {
                    "parent_run_id": final_config["parent_contract"]["stage4_parent_run_id"],
                    "parent_pointer_path": final_config["parent_contract"]["stage4_current_pointer"]["path"],
                    "parent_pointer_sha256": final_config["parent_contract"]["stage4_current_pointer"]["sha256"],
                    "parent_artifact_count": int(
                        (parent_before["scope"] == "stage4_parent_inventory").sum()
                    ),
                    "verified_at_start": True,
                },
            )
            atomic_write_json(
                provenance_dir / "runtime_environment.json",
                {
                    "python": sys.version,
                    "platform": platform.platform(),
                    "package_versions": runtime_versions,
                    "concurrency": {
                        "logical_job_budget": int(final_config["runtime"]["jobs"]),
                        "highs_threads_per_task": int(
                            final_config["runtime"]["highs_threads_per_task"]
                        ),
                        "frontier_retry_highs_threads": int(
                            final_config["runtime"]["frontier_retry_highs_threads"]
                        ),
                        "highs_random_seed": int(
                            final_config["runtime"]["highs_random_seed"]
                        ),
                        "candidate_parallel_workers": int(
                            final_config["runtime"]["candidate_parallel_workers"]
                        ),
                        "frontier_parallel_workers": int(
                            final_config["runtime"]["frontier_parallel_workers"]
                        ),
                        "frontier_retry_parallel_workers": int(
                            final_config["runtime"][
                                "frontier_retry_parallel_workers"
                            ]
                        ),
                    },
                },
            )
            code_manifest = pd.DataFrame(
                [
                    {
                        "relative_path": relative,
                        "sha256": digest,
                        "size_bytes": (package_root / relative).stat().st_size,
                    }
                    for relative, digest in code_before.items()
                ]
            )
            atomic_write_csv(code_manifest, provenance_dir / "finalization_code_contract.csv")
            code_snapshot = provenance_dir / "code_snapshot"
            for relative in CODE_CONTRACT_PATHS:
                copy_atomic(package_root / relative, code_snapshot / relative)

            tests_before = _test_manifest(package_root)
            test_snapshot = provenance_dir / "test_snapshot"
            for row in tests_before.itertuples(index=False):
                copy_atomic(
                    package_root / str(row.relative_path),
                    test_snapshot / str(row.relative_path),
                )
            atomic_write_csv(tests_before, provenance_dir / "test_file_snapshot_manifest.csv")
            test_result = _run_tests(package_root)
            atomic_write_text(
                provenance_dir / "regression_test_stdout.txt",
                _nonempty_test_stream(test_result["stdout"], stream_name="stdout"),
            )
            atomic_write_text(
                provenance_dir / "regression_test_stderr.txt",
                _nonempty_test_stream(test_result["stderr"], stream_name="stderr"),
            )
            atomic_write_json(
                provenance_dir / "regression_test_execution.json",
                {key: value for key, value in test_result.items() if key not in {"stdout", "stderr"}},
            )
            if int(test_result["returncode"]) != 0 or int(test_result["passed"]) <= 0:
                raise RuntimeError("Full regression suite failed before Stage 4.1 computation")
            if not tests_before.equals(_test_manifest(package_root)):
                raise RuntimeError("Test sources changed during regression execution")
            if code_before != _sha_contract(package_root, CODE_CONTRACT_PATHS):
                raise RuntimeError("Finalization sources changed during regression execution")

            stage3, preflight_checks, preflight_meta = run_preflight(package_root, base_config)
            atomic_write_csv(
                pd.DataFrame([check.as_dict() for check in preflight_checks]),
                provenance_dir / "stage3_preflight_quality_gate.csv",
            )
            atomic_write_json(provenance_dir / "stage3_preflight_metadata.json", preflight_meta)
            all_candidates, bundle_values = canonicalize_interface(stage3, base_config)
            grids = _canonicalize_grid(stage3, base_config)
            all_candidates = _attach_admin_need(all_candidates, grids)
            primary_matrix_id = str(
                base_config["network_validation"]["euclidean_reference_matrix_id"]
            )
            coverage = load_coverage_matrix(
                stage3, primary_matrix_id, base_config["column_aliases"]
            )
            full_data = _build_optimization_input(
                all_candidates,
                grids,
                bundle_values,
                coverage,
            )
            build_policy = CandidateBuildPolicy(
                max_candidates=int(final_config["candidate_sets"]["max_candidates"]),
                coarse_pareto_tier_max=int(
                    final_config["candidate_sets"]["coarse_pareto_tier_max"]
                ),
                require_all_admins=bool(
                    final_config["candidate_sets"]["require_all_admins"]
                ),
                coverage_cluster_required=bool(
                    final_config["candidate_sets"]["coverage_cluster_required"]
                ),
            )
            build = build_candidate_sets(
                full_data,
                base_config,
                strategies=tuple(final_config["candidate_sets"]["strategies"]),
                policy=build_policy,
                visit_count=20,
            )
            parent_run = (
                package_root
                / "outputs/model_v1/08_stage4/runs"
                / str(final_config["parent_contract"]["stage4_parent_run_id"])
            )
            top3_identity = _candidate_top3_identity(build, parent_run)
            atomic_write_csv(build.preflight, candidate_dir / "candidate_set_preflight.csv")
            atomic_write_parquet(build.membership, candidate_dir / "candidate_set_membership.parquet")
            atomic_write_csv(build.membership, candidate_dir / "candidate_set_membership.csv")
            atomic_write_csv(build.unavailable, candidate_dir / "unavailable_candidate_sets.csv")
            atomic_write_json(
                candidate_dir / "top3_frozen_parent_identity.json", top3_identity
            )

            candidate_solutions, candidate_summary, candidate_comparisons, candidate_payload = _solve_candidate_universes(
                build, base_config, final_config
            )
            candidate_decision = candidate_payload["decision"]
            atomic_write_csv(
                pd.DataFrame(candidate_payload["stage_rows"]),
                candidate_dir / "candidate_solver_stage_telemetry.csv",
            )
            atomic_write_csv(
                candidate_payload["normalized_stages"],
                candidate_dir / "candidate_solver_stage_certification.csv",
            )
            atomic_write_csv(
                candidate_payload["run_certifications"],
                candidate_dir / "candidate_solver_run_certification.csv",
            )
            atomic_write_csv(candidate_summary, candidate_dir / "candidate_set_scenario_metrics.csv")
            atomic_write_csv(candidate_comparisons, candidate_dir / "candidate_set_comparisons.csv")
            atomic_write_json(candidate_dir / "candidate_promotion_decision.json", candidate_decision)
            selected_set = candidate_decision.get("selected_candidate_set")
            if selected_set not in build.inputs:
                raise RuntimeError("Candidate sensitivity is inconclusive; no final universe may be selected")
            final_data = build.inputs[str(selected_set)]

            final_solutions, final_stage_rows = _solve_final_scenarios(
                final_data,
                base_config,
                final_config,
                str(selected_set),
                candidate_solutions,
            )
            # Persist the selected-universe n=20 phase before optional seed and
            # frontier diagnostics.  This is a diagnostic checkpoint, not a
            # cross-run resume authorization; a later failure can no longer
            # erase the certified final-policy evidence.
            prefrontier_stage_rows = pd.DataFrame(final_stage_rows)
            prefrontier_contract = NearOptimalityContract(
                contract_id="MEDIROAD_STAGE4_FINAL_N20_PREFRONTIER_REL_0P005",
                relative_gap_threshold=float(
                    final_config["solver"]["near_optimal_relative_gap_max"]
                ),
            )
            (
                prefrontier_stage_certification,
                prefrontier_run_certification,
            ) = _certify_stages(prefrontier_stage_rows, prefrontier_contract)
            prefrontier_metrics = pd.DataFrame(
                [
                    _solution_metric_row(
                        solution,
                        candidate_set=str(selected_set),
                        engine="scipy_highs",
                        role="final_n20_prefrontier_checkpoint",
                    )
                    for solution in final_solutions.values()
                ]
            )
            checkpoint_paths = [
                solver_dir / "final_n20_prefrontier_stage_telemetry.csv",
                solver_dir / "final_n20_prefrontier_stage_certification.csv",
                solver_dir / "final_n20_prefrontier_run_certification.csv",
                solver_dir / "final_n20_prefrontier_scenario_metrics.csv",
            ]
            for frame, path in zip(
                (
                    prefrontier_stage_rows,
                    prefrontier_stage_certification,
                    prefrontier_run_certification,
                    prefrontier_metrics,
                ),
                checkpoint_paths,
                strict=True,
            ):
                atomic_write_csv(frame, path)
            solution_seals = []
            for scenario, solution in sorted(final_solutions.items()):
                selected_ids = sorted(solution.selected["venue_id"].astype(str))
                solution_seals.append(
                    {
                        "scenario": scenario,
                        "visit_count": int(solution.visit_count),
                        "selected_venue_ids_sha256": hashlib.sha256(
                            ("\n".join(selected_ids) + "\n").encode("utf-8")
                        ).hexdigest(),
                        "covered_grid_mask_sha256": hashlib.sha256(
                            np.packbits(
                                np.asarray(solution.covered_grid_mask, dtype=np.uint8)
                            ).tobytes()
                        ).hexdigest(),
                        "metrics_sha256": hashlib.sha256(
                            stable_json_dumps(_json_safe(solution.metrics)).encode(
                                "utf-8"
                            )
                        ).hexdigest(),
                    }
                )
            atomic_write_json(
                solver_dir / "FINAL_N20_PHASE_COMPLETE.json",
                {
                    "schema_version": "MEDIROAD_STAGE4_PHASE_CHECKPOINT_V1",
                    "phase_id": "selected_universe_final_n20",
                    "source_run_id": run_id,
                    "completed_at": utc_now_iso(),
                    "resumable": False,
                    "resume_status": "DIAGNOSTIC_CHECKPOINT_ONLY_NO_IMPORT_IMPLEMENTED",
                    "input_contract": {
                        "config_sha256": config_sha,
                        "code_contract_sha256": hashlib.sha256(
                            stable_json_dumps(code_before).encode("utf-8")
                        ).hexdigest(),
                        "declared_parent_contract_sha256": sha256_file(
                            provenance_dir / "declared_parent_contract.csv"
                        ),
                        "candidate_membership_sha256": sha256_file(
                            candidate_dir / "candidate_set_membership.csv"
                        ),
                        "candidate_set": str(selected_set),
                    },
                    "outputs": [
                        {
                            "relative_path": ensure_relative_to(path, run_root),
                            "size_bytes": path.stat().st_size,
                            "sha256": sha256_file(path),
                        }
                        for path in checkpoint_paths
                    ],
                    "solutions": solution_seals,
                    "semantic_checks": {
                        "scenario_count": len(final_solutions),
                        "exact_twenty_unique_each": all(
                            len(solution.selected) == 20
                            and solution.selected["venue_id"].astype(str).nunique()
                            == 20
                            for solution in final_solutions.values()
                        ),
                        "all_four_stage_telemetry": bool(
                            len(prefrontier_stage_rows) == 12
                            and prefrontier_run_certification["stage_count"]
                            .astype(int)
                            .eq(4)
                            .all()
                        ),
                    },
                },
            )
            efficiency_reference = int(
                final_solutions["efficiency"].metrics["unique_elderly_population_scaled"]
            )
            seed_solutions, seed_stage_rows = _solve_seed_runs(
                final_data,
                base_config,
                final_config,
                str(selected_set),
                efficiency_reference,
            )
            seed_result = analyze_seed_stability(
                seed_solutions,
                expected_seeds=tuple(final_config["solver"]["seeds"]),
                thresholds=_seed_thresholds(final_config),
            )
            seed_contract = NearOptimalityContract(
                contract_id="MEDIROAD_STAGE4_SEED_DIAGNOSTIC_REL_0P005",
                relative_gap_threshold=float(
                    final_config["solver"]["near_optimal_relative_gap_max"]
                ),
            )
            seed_normalized_stages, seed_run_certifications = _certify_stages(
                pd.DataFrame(seed_stage_rows),
                seed_contract,
            )
            seed_stability_usable = bool(
                len(seed_run_certifications) == 5
                and seed_run_certifications["certified"].astype(bool).all()
            )
            if not seed_stability_usable:
                seed_result.notes.append(
                    "CP-SAT seed runs were not all <=0.5% certified; exact-site frequency "
                    "is diagnostic and is excluded from field-queue scoring."
                )
                seed_result.classification = "INCONCLUSIVE_SEARCH_STABILITY"
                seed_result.passed = False
            atomic_write_csv(seed_result.per_seed, seed_dir / "seed_scenario_metrics.csv")
            atomic_write_csv(seed_result.pairwise, seed_dir / "seed_pairwise_jaccard.csv")
            atomic_write_csv(seed_result.metric_spread, seed_dir / "seed_metric_spread.csv")
            atomic_write_csv(seed_result.selection_frequency, seed_dir / "seed_selection_frequency.csv")
            atomic_write_csv(
                seed_normalized_stages,
                seed_dir / "seed_solver_stage_certification.csv",
            )
            atomic_write_csv(
                seed_run_certifications,
                seed_dir / "seed_solver_run_certification.csv",
            )
            atomic_write_json(
                seed_dir / "seed_stability_decision.json",
                {
                    "classification": seed_result.classification,
                    "passed": seed_result.passed,
                    "thresholds": seed_result.thresholds,
                    "threshold_fingerprint": seed_result.threshold_fingerprint,
                    "notes": seed_result.notes,
                },
            )

            bundle_assignments, bundle_summary, bundle_decision = _bundle_sensitivity(
                final_solutions["balanced"].selected,
                final_data.bundles,
                base_config,
            )
            for minimum, assignment in bundle_assignments.items():
                atomic_write_csv(assignment, bundle_dir / f"bundle_assignment_min{minimum}.csv")
            atomic_write_csv(bundle_summary, bundle_dir / "bundle_sensitivity_metrics.csv")
            atomic_write_json(bundle_dir / "bundle_sensitivity_decision.json", bundle_decision)
            final_solutions["balanced"].bundle_assignments = bundle_assignments[1]

            (
                frontier,
                marginal,
                frontier_selection,
                frontier_stage_rows,
                frontier_attempts,
                frontier_attempt_stages,
            ) = _run_frontier(
                final_data,
                base_config,
                final_config,
                str(selected_set),
                final_solutions,
                attempt_log_path=policy_dir / "visit_count_frontier_attempts.csv",
                attempt_stage_log_path=(
                    policy_dir / "visit_count_frontier_attempt_stage_telemetry.csv"
                ),
            )
            baseline_table, greedy_no_harm = _baseline_table(
                final_data, base_config, final_solutions
            )
            pareto = _pareto_policy_frame(final_solutions)
            final_metrics = pd.DataFrame(
                [
                    _solution_metric_row(
                        solution,
                        candidate_set=str(selected_set),
                        engine="scipy_highs",
                        role="final_policy",
                    )
                    for solution in final_solutions.values()
                ]
            )
            atomic_write_csv(final_metrics, policy_dir / "final_policy_scenario_metrics.csv")
            for scenario, solution in final_solutions.items():
                plan = solution.selected.copy()
                plan.insert(0, "scenario", scenario)
                plan.insert(1, "candidate_set", str(selected_set))
                atomic_write_csv(plan, policy_dir / f"final_plan__{scenario}.csv")
            atomic_write_csv(pareto, policy_dir / "pareto_frontier.csv")
            atomic_write_csv(frontier, policy_dir / "visit_count_frontier.csv")
            atomic_write_csv(marginal, policy_dir / "visit_marginal_benefit.csv")
            atomic_write_csv(
                frontier_attempts,
                policy_dir / "visit_count_frontier_attempts.csv",
            )
            atomic_write_csv(
                frontier_attempt_stages,
                policy_dir / "visit_count_frontier_attempt_stage_telemetry.csv",
            )
            atomic_write_parquet(frontier_selection, policy_dir / "visit_count_frontier_selections.parquet")
            atomic_write_csv(baseline_table, policy_dir / "baseline_comparison.csv")
            atomic_write_json(solver_dir / "greedy_no_harm.json", greedy_no_harm)
            atomic_write_csv(pd.DataFrame([greedy_no_harm]), solver_dir / "greedy_no_harm.csv")

            field = _field_readiness(
                final_solutions,
                all_candidates,
                seed_result,
                parent_run,
                final_config,
                seed_stability_usable=seed_stability_usable,
            )
            atomic_write_csv(field["policy_frequency"], field_dir / "policy_selection_frequency.csv")
            atomic_write_csv(field["seed_venue_stability"], field_dir / "seed_venue_stability.csv")
            atomic_write_csv(field["catchment_stability"], field_dir / "catchment_stability.csv")
            atomic_write_csv(field["cluster_stability"], field_dir / "cluster_stability.csv")
            atomic_write_csv(field["fallbacks"], field_dir / "actual_facility_fallbacks.csv")
            atomic_write_csv(field["queue"], field_dir / "field_validation_priority_queue.csv")
            atomic_write_csv(field["form"], field_dir / "field_validation_form.csv")
            atomic_write_csv(field["resolution"], field_dir / "physical_venue_resolution.csv")
            atomic_write_json(field_dir / "robust_core_summary.json", field["robust_core"])

            team_bases = load_mobile_team_bases(
                package_root / str(final_config["travel"]["mobile_team_bases_path"])
            )
            travel_gated = prepare_final_travel_objective(
                final_solutions["balanced"].selected,
                team_bases,
            )
            diagnostic_travel_input = _attach_diagnostic_travel_context(
                all_candidates,
                stage3.run_root,
            )
            diagnostic_travel = build_diagnostic_travel_scenarios(
                diagnostic_travel_input
            )
            equity = _beneficiary_equity(final_solutions["balanced"], final_data)
            atomic_write_json(field_dir / "mobile_team_base_contract.json", team_bases.as_dict())
            atomic_write_parquet(diagnostic_travel, field_dir / "diagnostic_travel_sensitivity.parquet")
            atomic_write_csv(
                diagnostic_travel[
                    diagnostic_travel["venue_id"].astype(str).isin(
                        final_solutions["balanced"].selected["venue_id"].astype(str)
                    )
                ],
                field_dir / "diagnostic_travel_selected20.csv",
            )
            atomic_write_json(field_dir / "beneficiary_visit_location_equity.json", equity)
            operational_requirements = {
                "field_validation": "PENDING",
                "mobile_team_bases": team_bases.evidence_status,
                "vehicle_information": "UNAVAILABLE",
                "operational_calendar": "UNAVAILABLE",
                "venue_availability": "UNAVAILABLE",
                "public_holidays": "NOT_IN_SCOPE_UNTIL_STAGE5",
                "stage5_started": False,
            }
            atomic_write_json(field_dir / "required_operational_data.json", operational_requirements)

            primary = final_solutions["balanced"].selected.copy()
            primary_assignment = bundle_assignments[1][
                ["venue_id", "bundle_id", "bundle_value", "bundle_assignment_status"]
            ]
            resolution_columns = [
                "venue_id",
                "spatial_anchor_venue_id",
                "spatial_anchor_is_busproxy",
                "actual_facility_fallback_1",
                "actual_facility_fallback_2",
                "final_physical_venue_id",
                "venue_unresolved",
                "resolution_status",
            ]
            interface = primary.merge(
                primary_assignment,
                on="venue_id",
                how="left",
                validate="one_to_one",
            ).merge(
                field["resolution"][resolution_columns],
                on="venue_id",
                how="left",
                validate="one_to_one",
            ).merge(
                travel_gated[
                    [
                        "venue_id",
                        "final_travel_minutes",
                        "final_travel_objective_enabled",
                        "final_travel_input_status",
                    ]
                ],
                on="venue_id",
                how="left",
                validate="one_to_one",
            )
            interface.insert(0, "visit_id", [f"VISIT_{index:02d}" for index in range(1, len(interface) + 1)])
            interface.insert(1, "stage4_1_run_id", run_id)
            interface["temporal_release_type"] = "NO_STRONG_PREFERENCE"
            interface["temporal_confidence"] = "LOW"
            interface["recommended_season"] = pd.NA
            interface["fallback_season"] = pd.NA
            interface["exact_month"] = pd.NA
            interface["exact_date"] = pd.NA
            interface["stage5_release_allowed"] = False
            interface["stage5_started"] = False
            atomic_write_csv(interface, interface_dir / "stage4_1_to_field_validation.csv")
            atomic_write_parquet(interface, interface_dir / "stage4_1_to_field_validation.parquet")
            atomic_write_json(
                interface_dir / "stage5_readiness_checklist.json",
                {
                    **operational_requirements,
                    "spatial_anchor_count": len(interface),
                    "unresolved_venue_count": int(interface["venue_unresolved"].sum()),
                    "exact_month_nonnull": int(interface["exact_month"].notna().sum()),
                    "exact_date_nonnull": int(interface["exact_date"].notna().sum()),
                    "stage5_release_allowed": False,
                },
            )

            all_stage_rows = pd.DataFrame(
                [
                    *candidate_payload["stage_rows"],
                    *final_stage_rows,
                    *seed_stage_rows,
                    *frontier_stage_rows,
                ]
            )
            atomic_write_csv(all_stage_rows, solver_dir / "solver_stage_telemetry.csv")
            contract = NearOptimalityContract(
                contract_id="MEDIROAD_STAGE4_FINALIZATION_REL_0P005",
                relative_gap_threshold=float(
                    final_config["solver"]["near_optimal_relative_gap_max"]
                ),
            )
            normalized_stages, run_certifications = _certify_stages(all_stage_rows, contract)
            atomic_write_csv(normalized_stages, solver_dir / "solver_stage_certification.csv")
            atomic_write_csv(run_certifications, solver_dir / "solver_run_certification.csv")
            final_certifications = run_certifications[
                run_certifications["run_role"].eq("final_n20_certification")
            ].copy()
            frontier_certifications = run_certifications[
                run_certifications["run_role"].eq("diagnostic_visit_count_frontier")
            ].copy()
            frontier = frontier.merge(
                frontier_certifications[
                    [
                        "candidate_set",
                        "scenario",
                        "catchment_id",
                        "visit_count",
                        "engine",
                        "certification_class",
                        "certified",
                        "telemetry_complete",
                        "max_relative_gap",
                    ]
                ],
                on=[
                    "candidate_set",
                    "scenario",
                    "catchment_id",
                    "visit_count",
                    "engine",
                ],
                how="left",
                validate="one_to_one",
            )
            marginal = _annotate_frontier_marginal_certification(
                marginal, frontier_certifications
            )
            atomic_write_csv(frontier, policy_dir / "visit_count_frontier.csv")
            atomic_write_csv(marginal, policy_dir / "visit_marginal_benefit.csv")

            parent_after = capture_parent_tree(package_root, final_config)
            stage5_after = _forbidden_stage5_snapshot(package_root, final_config)
            atomic_write_csv(parent_after, provenance_dir / "frozen_parent_after.csv")
            atomic_write_csv(
                stage5_after,
                provenance_dir / "forbidden_stage5_namespace_after.csv",
            )
            code_after = _sha_contract(package_root, CODE_CONTRACT_PATHS)
            code_unchanged = code_before == code_after
            if sha256_file(finalization_config_path) != config_sha:
                raise RuntimeError("Finalization config changed during execution")
            if not tests_before.equals(_test_manifest(package_root)):
                raise RuntimeError("Test sources changed after execution")

            gate_peak_rss_gib = _peak_rss_gib()
            gates = _quality_gate(
                parent_contract=parent_contract,
                parent_before=parent_before,
                parent_after=parent_after,
                stage5_before=stage5_before,
                stage5_after=stage5_after,
                code_unchanged=code_unchanged,
                tests=test_result,
                preflight_checks=preflight_checks,
                all_candidates=all_candidates,
                build=build,
                top3_identity=top3_identity,
                candidate_decision=candidate_decision,
                final_solutions=final_solutions,
                solver_stage_rows=all_stage_rows,
                final_run_certification=final_certifications,
                greedy=greedy_no_harm,
                seed_result=seed_result,
                seed_run_certification=seed_run_certifications,
                bundle_summary=bundle_summary,
                field=field,
                travel_contract=team_bases,
                travel_gated=travel_gated,
                diagnostic_travel=diagnostic_travel,
                equity=equity,
                frontier=frontier,
                frontier_attempts=frontier_attempts,
                frontier_attempt_stages=frontier_attempt_stages,
                frontier_run_certification=frontier_certifications,
                marginal=marginal,
                interface=interface,
                peak_rss_gib=gate_peak_rss_gib,
                base_config=base_config,
                final_config=final_config,
            )
            release = _release_decision(gates)
            atomic_write_csv(gates, run_root / "STAGE4_1_FINAL_QUALITY_GATE.csv")
            atomic_write_json(run_root / "STAGE4_1_RELEASE_DECISION.json", release)
            _make_figures(
                candidate_comparisons,
                seed_result.pairwise,
                frontier,
                bundle_summary,
                run_root / "figures",
            )
            report = _report(
                run_id=run_id,
                release=release,
                final_solutions=final_solutions,
                final_certification=final_certifications,
                greedy_no_harm=greedy_no_harm,
                candidate_decision=candidate_decision,
                candidate_comparisons=candidate_comparisons,
                seed_result=seed_result,
                bundle_summary=bundle_summary,
                robust_core=field["robust_core"],
                field_queue=field["queue"],
                resolution=field["resolution"],
                travel_contract=team_bases,
                diagnostic_travel=diagnostic_travel,
                equity=equity,
                frontier=frontier,
                frontier_attempts=frontier_attempts,
                marginal=marginal,
                gates=gates,
            )
            atomic_write_text(run_root / "FINAL_REPORT.md", report)

            inventory = inventory_files(
                run_root,
                exclude_names={
                    ".RUNNING",
                    "STAGE4_1_ARTIFACT_INVENTORY.csv",
                    "STAGE4_1_RUN_METADATA.json",
                },
            )
            if inventory.empty or (inventory["size_bytes"] <= 0).any():
                raise RuntimeError("Finalization artifact inventory is empty or contains empty files")
            atomic_write_csv(inventory, run_root / "STAGE4_1_ARTIFACT_INVENTORY.csv")
            for row in inventory.itertuples(index=False):
                artifact = run_root / str(row.relative_path)
                if artifact.stat().st_size != int(row.size_bytes) or sha256_file(artifact) != str(row.sha256):
                    raise RuntimeError(f"Final inventory mismatch: {row.relative_path}")

            completed_at = utc_now_iso()
            peak_rss_gib = _peak_rss_gib()
            if peak_rss_gib > float(final_config["runtime"]["memory_soft_limit_gib"]):
                raise RuntimeError(
                    f"Peak RSS {peak_rss_gib:.3f} GiB exceeded the 24 GiB soft limit"
                )
            metadata = {
                "version": final_config["version"],
                "run_id": run_id,
                "decision": release["decision"],
                "started_at": started_at,
                "completed_at": completed_at,
                "runtime_sec": float(time.perf_counter() - pipeline_started),
                "platform": platform.platform(),
                "python": sys.version,
                "dependency_versions": runtime_versions,
                "jobs": int(final_config["runtime"]["jobs"]),
                "highs_threads_per_task": int(
                    final_config["runtime"]["highs_threads_per_task"]
                ),
                "frontier_retry_highs_threads": int(
                    final_config["runtime"]["frontier_retry_highs_threads"]
                ),
                "highs_random_seed": int(
                    final_config["runtime"]["highs_random_seed"]
                ),
                "candidate_parallel_workers": int(
                    final_config["runtime"]["candidate_parallel_workers"]
                ),
                "frontier_parallel_workers": int(
                    final_config["runtime"]["frontier_parallel_workers"]
                ),
                "frontier_retry_parallel_workers": int(
                    final_config["runtime"]["frontier_retry_parallel_workers"]
                ),
                "frontier_attempt_count": int(len(frontier_attempts)),
                "frontier_retry_count": int(
                    frontier_attempts["attempt_number"].astype(int).eq(2).sum()
                ),
                "frontier_attempt_stage_count": int(len(frontier_attempt_stages)),
                "frontier_model_contract_sha256": str(
                    frontier_attempts["model_contract_sha256"].iloc[0]
                ),
                "peak_rss_gib": peak_rss_gib,
                "tests_passed": int(test_result["passed"]),
                "tests_failed": int(test_result["failed"]),
                "candidate_set": selected_set,
                "candidate_decision": candidate_decision["decision"],
                "seed_classification": seed_result.classification,
                "selected_universe_final_solver_certified": bool(
                    final_certifications["certified"].all()
                ),
                "candidate_reduction_certified": bool(
                    candidate_decision["candidate_reduction_certified"]
                ),
                "expanded_candidate_comparison_conclusive": bool(
                    candidate_decision["expanded_comparison_conclusive"]
                ),
                "solver_certified": bool(
                    final_certifications["certified"].all()
                    and candidate_decision["candidate_reduction_certified"]
                ),
                "stage4_provisional": release["stage4_provisional"],
                "stage4_spatial_status": release["stage4_spatial_status"],
                "stage4_operational_provisional": release[
                    "stage4_operational_provisional"
                ],
                "stage4_computational_complete": release["stage4_computational_complete"],
                "stage4_ready_for_field_validation": release["stage4_ready_for_field_validation"],
                "stage4_operational_final": release["stage4_operational_final"],
                "stage5_started": False,
                "stage5_release_allowed": False,
                "field_queue_count": int(len(field["queue"])),
                "unresolved_venue_count": int(field["resolution"]["venue_unresolved"].sum()),
                "travel_evidence_status": team_bases.evidence_status,
                "hard_ready_gate_total": int(gates["required_for_ready_field_validation"].sum()),
                "hard_ready_gate_failed": len(release["ready_failures"]),
                "computational_gate_failed": len(release["computational_failures"]),
                "operational_gate_failed": len(release["operational_failures"]),
                "artifact_count": int(len(inventory)),
                "artifact_inventory_sha256": sha256_file(
                    run_root / "STAGE4_1_ARTIFACT_INVENTORY.csv"
                ),
                "quality_gate_sha256": sha256_file(
                    run_root / "STAGE4_1_FINAL_QUALITY_GATE.csv"
                ),
                "interface_sha256": sha256_file(
                    interface_dir / "stage4_1_to_field_validation.parquet"
                ),
                "report_sha256": sha256_file(run_root / "FINAL_REPORT.md"),
                "config_sha256": config_sha,
                "code_contract": code_after,
                "parent_stage4_run_id": final_config["parent_contract"]["stage4_parent_run_id"],
                "parent_tree_unchanged": parent_before.equals(parent_after),
            }
            atomic_write_json(run_root / "STAGE4_1_RUN_METADATA.json", metadata)

            if release["decision"] == "FAIL_STAGE4_FINALIZATION":
                raise RuntimeError(
                    "Stage 4.1 ready-for-field-validation gates failed: "
                    + ", ".join(release["ready_failures"])
                )

            report_root = package_root / str(final_config["paths"]["report_root"])
            report_root.mkdir(parents=True, exist_ok=True)
            report_alias = report_root / "CURRENT_STAGE4_FINALIZATION.md"
            # Close every mutable-input race immediately before the authoritative
            # pointer commit. The mutable flat report is convenience-only, is
            # not referenced by the pointer, and is updated after commit.
            if not parent_before.equals(capture_parent_tree(package_root, final_config)):
                raise RuntimeError("Frozen parent tree changed before final commit")
            if not parent_contract.equals(
                _verify_declared_parents(package_root, final_config)
            ):
                raise RuntimeError("A declared parent hash changed before final commit")
            if code_before != _sha_contract(package_root, CODE_CONTRACT_PATHS):
                raise RuntimeError("Finalization code changed before final commit")
            if sha256_file(finalization_config_path) != config_sha:
                raise RuntimeError("Finalization config changed before final commit")
            if not tests_before.equals(_test_manifest(package_root)):
                raise RuntimeError("Test sources changed before final commit")
            if not _forbidden_stage5_snapshot(package_root, final_config).empty:
                raise RuntimeError("A forbidden Stage 5 namespace appeared before commit")
            running_marker.unlink(missing_ok=True)
            pointer = {
                "run_id": run_id,
                "decision": release["decision"],
                "metadata_relative_path": ensure_relative_to(
                    run_root / "STAGE4_1_RUN_METADATA.json", package_root
                ),
                "inventory_relative_path": ensure_relative_to(
                    run_root / "STAGE4_1_ARTIFACT_INVENTORY.csv", package_root
                ),
                "interface_relative_path": ensure_relative_to(
                    interface_dir / "stage4_1_to_field_validation.parquet", package_root
                ),
                "report_relative_path": ensure_relative_to(
                    run_root / "FINAL_REPORT.md", package_root
                ),
                "metadata_sha256": sha256_file(run_root / "STAGE4_1_RUN_METADATA.json"),
                "inventory_sha256": sha256_file(
                    run_root / "STAGE4_1_ARTIFACT_INVENTORY.csv"
                ),
                "interface_sha256": sha256_file(
                    interface_dir / "stage4_1_to_field_validation.parquet"
                ),
                "report_sha256": sha256_file(run_root / "FINAL_REPORT.md"),
                "stage4_computational_complete": release["stage4_computational_complete"],
                "stage4_ready_for_field_validation": True,
                "stage4_provisional": True,
                "stage4_spatial_status": release["stage4_spatial_status"],
                "stage4_operational_provisional": True,
                "stage4_operational_final": False,
                "candidate_reduction_certified": bool(
                    candidate_decision["candidate_reduction_certified"]
                ),
                "expanded_candidate_comparison_conclusive": bool(
                    candidate_decision["expanded_comparison_conclusive"]
                ),
                "stage5_started": False,
                "stage5_release_allowed": False,
            }
            atomic_write_json(output_root / "CURRENT_STAGE4_FINALIZATION_RUN.json", pointer)
            try:
                copy_atomic(run_root / "FINAL_REPORT.md", report_alias)
            except OSError:
                # The immutable run report and authoritative pointer are already
                # committed. A convenience alias failure cannot invalidate them.
                pass
            return run_root
    except BaseException as exc:
        if run_started:
            failed_marker = run_root / ".FAILED"
            if running_marker.exists():
                os.replace(running_marker, failed_marker)
            else:
                atomic_write_text(failed_marker, "FAILED\n")
            atomic_write_json(
                run_root / "STAGE4_1_RUN_FAILURE.json",
                {
                    "run_id": run_id,
                    "failed_at": utc_now_iso(),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
        raise


__all__ = [
    "FINALIZATION_DECISIONS",
    "capture_parent_tree",
    "load_finalization_config",
    "run_stage4_finalization",
]
