from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import yaml


FROZEN_GAP = 0.005
FROZEN_TOP3_COUNT = 452
FROZEN_PATTERN_COUNT = 5242
FROZEN_VISIT_COUNT = 20
FROZEN_TOTAL_POPULATION_FLOOR = 167282.84413449097
FROZEN_MIN_SIGUNGU_FLOOR = 0.525217
FROZEN_HIGH_NEED_INCUMBENT = 11609.13711010601
FROZEN_HIGH_NEED_RETENTION = 0.99
FROZEN_HIGH_NEED_FLOOR = 11493.045739004951
FROZEN_OLD_HIGH_NEED_FLOOR = 11447.794835668963
FROZEN_NEED_RETENTION = 0.99
FROZEN_D_RUN_ID = "stage4_2d_equity_20260821T014952Z_cc9fd641ddf0"
FROZEN_C_RUN_ID = "stage4_2c_certification_20260820T151657Z_56c5f1bad35c"
FROZEN_OLD_NEED_INCUMBENT = 104231.18010692896
FROZEN_OLD_NEED_BOUND = 104752.01728095711
FROZEN_OLD_NEED_FLOOR = 103188.86830585968
FROZEN_OLD_COST_INCUMBENT = 654.5488005454546
FROZEN_OLD_COST_BOUND = 651.8916526882051
FROZEN_HIGHS_VERSION = "1.15.1"
FROZEN_NEED_TIME_LIMIT_SEC = 900
FROZEN_COST_TIME_LIMIT_SEC = 1200
FROZEN_NEED_RANDOM_SEED = 5098
FROZEN_COST_RANDOM_SEED = 6107


def _same_binary64(value: Any, expected: float) -> bool:
    return (
        type(value) in (int, float)
        and math.isfinite(float(value))
        and float(value).hex() == float(expected).hex()
    )


def _same_typed_mapping(value: Any, expected: dict[str, Any]) -> bool:
    return bool(
        isinstance(value, dict)
        and set(value) == set(expected)
        and all(
            type(value[key]) is type(expected_value) and value[key] == expected_value
            for key, expected_value in expected.items()
        )
    )


def load_yaml(path: Path) -> dict[str, Any]:
    path = path.resolve()
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return payload


def set_thread_environment(threads: int = 8) -> None:
    """Reserve the frozen CPU budget for HiGHS, not helper BLAS libraries."""

    if int(threads) != 8:
        raise ValueError("Stage4.2E thread budget is frozen at 8")
    value = "1"
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        os.environ[name] = value


def _require_sha_map(section: Any, *, label: str, required: set[str]) -> None:
    if not isinstance(section, dict) or set(section) != required:
        raise ValueError(f"{label} must pin exactly {sorted(required)}")
    for relative, digest in section.items():
        if (
            not isinstance(relative, str)
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(ch not in "0123456789abcdef" for ch in digest)
        ):
            raise ValueError(f"Invalid SHA-256 pin in {label}: {relative!r}")


def validate_config(cfg: dict[str, Any]) -> None:
    required = {
        "version",
        "mode",
        "stage5_started",
        "paths",
        "hardware",
        "contract",
        "stage42d_parent",
        "stage42c_bound_evidence",
        "solver",
        "gate",
    }
    if set(cfg) != required:
        raise ValueError(
            "Stage4.2E config must contain exactly "
            f"{sorted(required)}; observed {sorted(cfg)}"
        )
    if cfg.get("version") != "MEDIROAD_STAGE4_2E_EQUITY_TAIL_EXACT_CERTIFICATION_V1":
        raise ValueError("Stage4.2E config version changed")
    if cfg.get("mode") != "official":
        raise ValueError("Stage4.2E currently permits official mode only")
    if cfg.get("stage5_started") is not False:
        raise ValueError("stage5_started must remain exactly false")

    paths = cfg["paths"]
    expected_paths = {
        "output_root": "outputs/model_v1/10_stage4_2e_equity_tail_certification",
        "report_root": "reports/model_v1/stage4_2e_equity_tail_certification",
        "lock": "outputs/model_v1/.stage4_2e_equity_tail_certification_writer.lock",
    }
    if not _same_typed_mapping(paths, expected_paths):
        raise ValueError("Stage4.2E output/report/lock paths are frozen")

    hardware = cfg["hardware"]
    expected_hardware = {
        "profile": "intel_core_ultra_7_258v_32gb_rtx_3080ti",
        "solver_threads": 8,
        "memory_soft_limit_gb": 27,
        "process_priority": "above_normal",
    }
    if not _same_typed_mapping(hardware, expected_hardware):
        raise ValueError("Stage4.2E hardware contract changed")

    contract = cfg["contract"]
    exact_values: dict[str, Any] = {
        "candidate_set": "top3",
        "visit_count": FROZEN_VISIT_COUNT,
        "candidate_count": FROZEN_TOP3_COUNT,
        "pattern_count": FROZEN_PATTERN_COUNT,
        "preserve_stage42c_objectives": True,
        "preserve_stage42c_constraints": True,
        "preserve_candidate_universe": True,
        "candidate_expansion_enabled": False,
    }
    for key, expected in exact_values.items():
        value = contract.get(key)
        if value != expected or type(value) is not type(expected):
            raise ValueError(f"contract.{key} must remain exactly {expected!r}")
    exact_floats = {
        "relative_gap": FROZEN_GAP,
        "total_population_floor": FROZEN_TOTAL_POPULATION_FLOOR,
        "min_sigungu_floor": FROZEN_MIN_SIGUNGU_FLOOR,
        "high_need_incumbent": FROZEN_HIGH_NEED_INCUMBENT,
        "high_need_retention": FROZEN_HIGH_NEED_RETENTION,
        "high_need_floor": FROZEN_HIGH_NEED_FLOOR,
        "need_weighted_retention": FROZEN_NEED_RETENTION,
    }
    for key, expected in exact_floats.items():
        if not _same_binary64(contract.get(key), expected):
            raise ValueError(f"contract.{key} must preserve binary64 {expected!r}")
    if set(contract) != set(exact_values) | set(exact_floats):
        raise ValueError("contract must contain exactly the frozen Stage4.2E fields")
    if (FROZEN_HIGH_NEED_INCUMBENT * FROZEN_HIGH_NEED_RETENTION).hex() != FROZEN_HIGH_NEED_FLOOR.hex():
        raise AssertionError("Internal frozen high-need floor is inconsistent")

    parent = cfg["stage42d_parent"]
    if set(parent) != {"run_id", "artifact_sha256"}:
        raise ValueError("stage42d_parent schema changed")
    if parent.get("run_id") != FROZEN_D_RUN_ID:
        raise ValueError("stage42d_parent.run_id changed")
    _require_sha_map(
        parent.get("artifact_sha256"),
        label="stage42d_parent.artifact_sha256",
        required={
            "CURRENT_STAGE4_2D_EQUITY_CERTIFICATION_RUN.json",
            "ARTIFACT_INVENTORY.csv",
            "01_min_threshold_oracle/certificate.json",
            "02_high_threshold_oracle/certificate.json",
            "03_final_incumbent/plan__equity_front_stage_seed.csv",
            "04_quality_and_provenance/quality_gate.json",
        },
    )

    old = cfg["stage42c_bound_evidence"]
    if set(old) != {
        "run_id",
        "artifact_sha256",
        "need_weighted_incumbent",
        "need_weighted_best_bound",
        "need_weighted_retained_floor",
        "cost_incumbent",
        "cost_best_bound",
    }:
        raise ValueError("stage42c_bound_evidence schema changed")
    if old.get("run_id") != FROZEN_C_RUN_ID:
        raise ValueError("stage42c_bound_evidence.run_id changed")
    _require_sha_map(
        old.get("artifact_sha256"),
        label="stage42c_bound_evidence.artifact_sha256",
        required={
            "ARTIFACT_INVENTORY.csv",
            "metadata.json",
            "quality_gate.json",
            "candidate_sets/top3/plan__equity.csv",
            "candidate_sets/top3/stages__equity.json",
        },
    )
    for key, expected in {
        "need_weighted_incumbent": FROZEN_OLD_NEED_INCUMBENT,
        "need_weighted_best_bound": FROZEN_OLD_NEED_BOUND,
        "need_weighted_retained_floor": FROZEN_OLD_NEED_FLOOR,
        "cost_incumbent": FROZEN_OLD_COST_INCUMBENT,
        "cost_best_bound": FROZEN_OLD_COST_BOUND,
    }.items():
        if not _same_binary64(old.get(key), expected):
            raise ValueError(f"stage42c_bound_evidence.{key} changed")

    solver = cfg["solver"]
    exact_solver: dict[str, Any] = {
        "backend": "direct_highspy",
        "required_highspy_version": FROZEN_HIGHS_VERSION,
        "threads": 8,
        "parallel": True,
        "presolve": True,
        "fresh_need_solve_required": True,
        "fresh_cost_solve_required": True,
        "allow_threshold_relaxation": False,
        "allow_objective_change": False,
        "allow_candidate_change": False,
        "allow_inherited_bounds_for_certification": False,
    }
    for key, expected in exact_solver.items():
        value = solver.get(key)
        if value != expected or type(value) is not type(expected):
            raise ValueError(f"solver.{key} must remain exactly {expected!r}")
    expected_solver_keys = set(exact_solver) | {
        "relative_gap",
        "time_limit_sec",
        "random_seed",
        "heuristic_effort",
        "mip_max_start_nodes",
        "log_to_console",
        "extra_highs_options",
    }
    if set(solver) != expected_solver_keys:
        raise ValueError("solver schema changed")
    if not _same_binary64(solver.get("relative_gap"), FROZEN_GAP):
        raise ValueError("solver.relative_gap changed")
    time_limits = solver.get("time_limit_sec")
    if not isinstance(time_limits, dict) or set(time_limits) != {"need_weighted", "cost"}:
        raise ValueError("solver.time_limit_sec must contain only need_weighted and cost")
    if not _same_typed_mapping(time_limits, {
        "need_weighted": FROZEN_NEED_TIME_LIMIT_SEC,
        "cost": FROZEN_COST_TIME_LIMIT_SEC,
    }):
        raise ValueError("Stage4.2E time limits changed")
    seeds = solver.get("random_seed")
    if not isinstance(seeds, dict) or set(seeds) != {"need_weighted", "cost"}:
        raise ValueError("solver.random_seed must contain only need_weighted and cost")
    if not _same_typed_mapping(seeds, {
        "need_weighted": FROZEN_NEED_RANDOM_SEED,
        "cost": FROZEN_COST_RANDOM_SEED,
    }):
        raise ValueError("Stage4.2E random seeds changed")
    if not _same_binary64(solver.get("heuristic_effort"), 0.10):
        raise ValueError("solver.heuristic_effort changed")
    if type(solver.get("mip_max_start_nodes")) is not int or solver["mip_max_start_nodes"] != 30000:
        raise ValueError("solver.mip_max_start_nodes changed")
    if solver.get("log_to_console") is not True:
        raise ValueError("solver.log_to_console must remain exactly true")
    extras = solver.get("extra_highs_options")
    expected_extras = {
        "mip_pool_soft_limit": 30000,
        "mip_pscost_minreliable": 4,
        "mip_min_cliquetable_entries_for_parallelism": 10000,
        "mip_detect_symmetry": True,
    }
    if not _same_typed_mapping(extras, expected_extras):
        raise ValueError("solver.extra_highs_options changed")

    gate = cfg["gate"]
    exact_gate = {
        "gap_threshold_must_not_be_relaxed": True,
        "require_fresh_need_solve": True,
        "require_fresh_cost_solve": True,
        "require_subset_and_sha_for_inherited_bounds": True,
        "require_fresh_incumbent_and_bound_each_stage": True,
        "historical_bounds_diagnostic_only": True,
        "require_need_weighted_certification": True,
        "require_cost_certification": True,
        "write_current_only_on_full_pass": True,
        "promotable_scope": "TOP3_EQUITY_FULL_ONLY",
        "candidate_expansion_certified": False,
        "stage4_full_computational_complete": False,
        "operational_final": False,
        "stage5_release_allowed": False,
    }
    for key, expected in exact_gate.items():
        value = gate.get(key)
        if value != expected or type(value) is not type(expected):
            raise ValueError(f"gate.{key} must remain exactly {expected!r}")
    if set(gate) != set(exact_gate):
        raise ValueError("gate schema changed")
