from __future__ import annotations

import os
import platform
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

FROZEN_GAP = 0.005
FROZEN_MIN_SIGUNGU_RETENTION = 1.0
FROZEN_HIGH_NEED_RETENTION = 0.99
FROZEN_PREFERRED_STAGE42C_RUN_ID = "stage4_2c_certification_20260820T151657Z_56c5f1bad35c"
FROZEN_PREFERRED_ARTIFACTS = {
    "ARTIFACT_INVENTORY.csv",
    "metadata.json",
    "quality_gate.json",
    "candidate_problem_summary.csv",
    "candidate_sets/top3/plan__efficiency.csv",
    "candidate_sets/top3/stages__efficiency.json",
    "candidate_sets/top3/plan__equity.csv",
    "candidate_sets/top3/stages__equity.json",
    "candidate_sets/top3/metrics__efficiency.json",
    "candidate_sets/top3/metrics__equity.json",
}
FROZEN_PREFERRED_INVENTORY_SHA256 = (
    "c1198cd711730871538ff536614f724ea64ee2634ced21d1aa9d144b00a74362"
)
FROZEN_IMPROVING_SOLUTION_ARTIFACTS = {
    "outputs/model_v1/10_stage4_2_certification/runs/"
    "stage4_2c_certification_20260820T154032Z_4a3af066364c/"
    "candidate_sets/top3/equity/stage_02_high_need_population/"
    "direct.highs.improving.sol": (
        "5e5361c7b8dbcae0f1b979176a1280324f6b09aae2e5d04b0ca3af0e3508d3b6"
    )
}
FROZEN_ORACLE_REFERENCES = {
    "total_population_floor": 167282.84413449097,
    "min_sigungu_incumbent": 0.5252179790669655,
    "min_sigungu_retained_floor": 0.525217,
    "min_sigungu_strict_threshold": 0.527844,
    "min_selection_sha256": "5440b1fb5d13530fcf5150c982e06f575182606490d3a7158549e7fd7ddf596e",
    "high_need_incumbent": 11609.13711010601,
    "high_need_strict_threshold": 11667.182795656541,
    "high_selection_sha256": "cb3ef97672fc0d61f54c5c911783b033033d2570d86c47b5ca0bfdbf333d6435",
    "anchor_seed_count": 26,
    "anchor_seed_manifest_sha256": "1561e262a0ab427db2b1f1e44b5e0cc550c699b1c0882297ea164e1200f8eecc",
}
FROZEN_MIN_HIGHS_ORACLE = {
    "expected_rows": 5499,
    "expected_cols": 5694,
    "expected_nnz": 159935,
    "expected_exact_model_sha256": "a8c5cb0f8b62771c4fcacbeaa9eec65d396ba460dcb693c25247f5ebe0aeda48",
    "expected_proof_model_sha256": "fe7715669123ed06ac1dbe0e297c1165dc282e31b142f993208fdd731f5f722a",
    "expected_universe_sha256": "a435792311692e5fd0b941c6b38ac3522c61df29025362c28e61cb6f0f8ac71a",
    "highspy_version": "1.15.1",
    "threads": 8,
    "time_limit_sec": 900,
    "random_seed": 104752,
    "heuristic_effort": 0.02,
    "mip_max_start_nodes": 5000,
    "mip_pool_soft_limit": 20000,
    "mip_pscost_minreliable": 4,
    "mip_min_cliquetable_entries_for_parallelism": 10000,
    "mip_lp_age_limit": 10,
    "mip_allow_restart": True,
    "mip_detect_symmetry": True,
}
FROZEN_HIGH_SCIP_ORACLE = {
    "expected_rows": 9377,
    "expected_cols": 5694,
    "expected_nnz": 1244380,
    "expected_exact_model_sha256": "e137612a8c69fde8cb7d8f2f7a1b36b3207d4712d6696fa9563e8f552919c2ab",
    "expected_proof_model_sha256": "52cefa254ba82fcf1607d31c0703f932f57dbb078d2164ce02f6552d05bfb75a",
    "expected_universe_sha256": "c7f3703d5168f464021ed7701877bf06cfe8dc176dc4be1892a1717179210c3c",
    "expected_cut_sha256": "6fd42e4e68b583acb1f4dd9a25276025d7f3519bc762a6d95e9856338f24867c",
    "expected_mps_sha256": "b06f0593a195d1ea3350a3f9ae62e782269ab91c8242ba803f365efb76494d50",
    "coverage_integrality": "binary_exact_projection",
    "outward_safe_cuts": True,
    "anchor_sizes": list(range(21)),
    "max_cuts": 4096,
    "expected_cut_count": 3877,
    "expected_unique_anchor_count": 3113,
    "portfolio_seeds": [11, 23, 42, 77, 101, 131, 197, 257],
    "workers": 8,
    "threads_per_worker": 1,
    "budget_strategy": "full_frozen_portfolio_rerun_same_mps_same_seeds",
    "budget_basis": {
        "run_id": "stage4_2d_equity_20260821T012542Z_394af1752be9",
        "prior_time_limit_sec_per_seed": 300,
        "proved_infeasible_seeds": [23, 77, 101, 197, 257],
        "timelimit_no_solution_seeds": [11, 42, 131],
    },
    "time_limit_sec_per_seed": 900,
    "memory_limit_mib_per_worker": 2800,
    "pyscipopt_version": "6.2.1",
    "scip_version": [10, 0, 2],
    "require_all_workers_complete": True,
    "require_all_workers_infeasible": True,
}


def _exact_number(value: Any, expected: float | int) -> bool:
    return (
        type(value) in (int, float)
        and math.isfinite(float(value))
        and abs(float(value) - float(expected)) <= 1e-12
    )


def _same_binary64(value: Any, expected: float) -> bool:
    """Require the exact configured binary64 value, rejecting coercive types."""

    return (
        type(value) in (int, float)
        and math.isfinite(float(value))
        and float(value).hex() == float(expected).hex()
    )


@dataclass(frozen=True)
class HardwareProfile:
    cpu_name: str
    logical_threads: int
    solver_threads: int
    portfolio_workers: int
    threads_per_portfolio_worker: int
    memory_total_gb: float | None
    memory_soft_limit_gb: float
    gpu_requested: bool
    gpu_available: bool
    gpu_backend: str
    gpu_name: str | None
    gpu_vram_gb: float | None
    native_windows: bool
    wsl: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_yaml(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return value


def is_wsl() -> bool:
    try:
        text = Path("/proc/version").read_text(encoding="utf-8", errors="ignore").lower()
    except OSError:
        return False
    return "microsoft" in text or "wsl" in text


def _cpu_name() -> str:
    value = platform.processor().strip()
    if value:
        return value
    if platform.system().lower() == "linux":
        try:
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="ignore").splitlines():
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return platform.machine() or "UNKNOWN"


def _memory_total_gb() -> float | None:
    try:
        import psutil  # type: ignore

        return float(psutil.virtual_memory().total) / (1024.0**3)
    except Exception:
        return None


def _gpu_probe(requested: bool) -> tuple[bool, str, str | None, float | None]:
    if not requested:
        return False, "DISABLED", None, None
    try:
        import cupy as cp  # type: ignore

        if int(cp.cuda.runtime.getDeviceCount()) <= 0:
            return False, "NO_CUDA_DEVICE", None, None
        cp.cuda.Device(0).use()
        props = cp.cuda.runtime.getDeviceProperties(0)
        raw_name = props.get("name", b"CUDA GPU") if isinstance(props, dict) else b"CUDA GPU"
        name = raw_name.decode(errors="replace") if isinstance(raw_name, (bytes, bytearray)) else str(raw_name)
        total = float(cp.cuda.runtime.memGetInfo()[1]) / (1024.0**3)
        probe = cp.arange(4096, dtype=cp.float32)
        value = float(cp.asnumpy((probe * probe).sum()))
        cp.cuda.Stream.null.synchronize()
        if value <= 0:
            raise RuntimeError("CUDA kernel verification failed")
        del probe
        cp.get_default_memory_pool().free_all_blocks()
        return True, "CUPY_CUDA", name, total
    except Exception as exc:
        return False, f"CPU_FALLBACK::{type(exc).__name__}", None, None


def resolve_hardware(cfg: dict[str, Any]) -> HardwareProfile:
    section = cfg.get("hardware", {})
    logical = max(1, int(os.cpu_count() or 8))
    solver_threads = min(logical, max(1, int(section.get("solver_threads", 8))))
    workers = min(logical, max(1, int(section.get("portfolio_workers", 2))))
    per_worker = max(1, int(section.get("threads_per_portfolio_worker", max(1, logical // workers))))
    if workers * per_worker > logical:
        per_worker = max(1, logical // workers)
    total_memory = _memory_total_gb()
    requested_memory = float(section.get("memory_soft_limit_gb", 27.0))
    memory_soft = requested_memory if total_memory is None else min(requested_memory, max(4.0, total_memory - 3.0))
    gpu_requested = bool(section.get("gpu_lns", True))
    gpu_available, gpu_backend, gpu_name, gpu_vram = _gpu_probe(gpu_requested)
    return HardwareProfile(
        cpu_name=_cpu_name(),
        logical_threads=logical,
        solver_threads=solver_threads,
        portfolio_workers=workers,
        threads_per_portfolio_worker=per_worker,
        memory_total_gb=total_memory,
        memory_soft_limit_gb=memory_soft,
        gpu_requested=gpu_requested,
        gpu_available=gpu_available,
        gpu_backend=gpu_backend,
        gpu_name=gpu_name,
        gpu_vram_gb=gpu_vram,
        native_windows=platform.system().lower() == "windows" and not is_wsl(),
        wsl=is_wsl(),
    )


def validate_config(cfg: dict[str, Any]) -> None:
    required = [
        "version",
        "paths",
        "hardware",
        "contract",
        "search",
        "solver",
        "gate",
        "oracle_certification",
    ]
    missing = [key for key in required if key not in cfg]
    if missing:
        raise ValueError(f"Stage4.2D config missing sections: {missing}")
    if "stage5_started" not in cfg or cfg["stage5_started"] is not False:
        raise ValueError("stage5_started must be present and exactly false")
    if cfg.get("mode") not in {"official", "diagnostic"}:
        raise ValueError("mode must be official or diagnostic")
    gap_value = cfg["gate"].get("near_optimal_relative_gap_max")
    if not _same_binary64(gap_value, FROZEN_GAP):
        raise ValueError(f"Frozen near-optimal gap must remain {FROZEN_GAP}")
    contract = cfg["contract"]
    if str(contract.get("candidate_set", "")) != "top3":
        raise ValueError("Stage4.2D is frozen to the Top3 candidate universe")
    if "visit_count" not in contract or type(contract["visit_count"]) is not int or contract["visit_count"] != 20:
        raise ValueError("Stage4.2D visit_count must remain 20")
    for name in ("preserve_stage42c_objectives", "preserve_stage42c_constraints"):
        if contract.get(name) is not True:
            raise ValueError(f"contract.{name} must be present and exactly true")
    exact_contract_flags = {
        "preserve_candidate_universe": True,
        "preserve_retention_rates": True,
        "candidate_expansion_enabled": False,
    }
    for name, expected in exact_contract_flags.items():
        if contract.get(name) is not expected:
            raise ValueError(f"contract.{name} must remain exactly {expected}")
    exact_retentions = {
        "min_sigungu_retention": FROZEN_MIN_SIGUNGU_RETENTION,
        "high_need_retention": FROZEN_HIGH_NEED_RETENTION,
    }
    for name, expected in exact_retentions.items():
        if not _same_binary64(contract.get(name), expected):
            raise ValueError(f"contract.{name} must remain exactly {expected}")
    gate_flags = {
        "gap_threshold_must_not_be_relaxed": True,
        "require_min_sigungu_certification": True,
        "require_high_need_certification": True,
        "write_current_only_on_full_front_stage_pass": True,
    }
    for name, expected in gate_flags.items():
        if cfg["gate"].get(name) is not expected:
            raise ValueError(f"gate.{name} must remain exactly {expected}")
    evidence = cfg.get("evidence", {})
    if evidence.get("preferred_stage42c_run_id") != FROZEN_PREFERRED_STAGE42C_RUN_ID:
        raise ValueError(
            "evidence.preferred_stage42c_run_id must remain the frozen verified source run"
        )
    preferred_artifacts = evidence.get("preferred_artifact_sha256")
    if not isinstance(preferred_artifacts, dict) or set(preferred_artifacts) != FROZEN_PREFERRED_ARTIFACTS:
        raise ValueError("evidence.preferred_artifact_sha256 must pin the exact required artifact set")
    if any(
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
        for value in preferred_artifacts.values()
    ):
        raise ValueError("Every preferred artifact must have a lowercase SHA-256 digest")
    if preferred_artifacts["ARTIFACT_INVENTORY.csv"] != FROZEN_PREFERRED_INVENTORY_SHA256:
        raise ValueError("Frozen preferred inventory trust anchor changed")
    if evidence.get("improving_solution_artifacts") != FROZEN_IMPROVING_SOLUTION_ARTIFACTS:
        raise ValueError("Pinned interrupted-solver evidence mapping changed")
    if int(cfg["hardware"].get("solver_threads", 8)) <= 0:
        raise ValueError("hardware.solver_threads must be positive")
    if float(cfg["hardware"].get("memory_soft_limit_gb", 27.0)) <= 0:
        raise ValueError("hardware.memory_soft_limit_gb must be positive")
    if int(cfg["search"].get("gpu_batch_size", 8192)) <= 0:
        raise ValueError("search.gpu_batch_size must be positive")
    if int(cfg["solver"].get("max_certificate_rounds", 4)) <= 0:
        raise ValueError("solver.max_certificate_rounds must be positive")
    replay_table = cfg["solver"].get("stage42c_oracle_replay", {})
    replay = replay_table.get("min_sigungu_coverage", {}) if isinstance(replay_table, dict) else {}
    expected_replay = {
        "enabled": True,
        "random_seed": 104752,
        "heuristic_effort": 0.02,
        "mip_max_start_nodes": 5000,
        "use_objective_target": False,
        "include_local_exclusion_cuts": False,
    }
    for name, expected in expected_replay.items():
        value = replay.get(name)
        if isinstance(expected, float):
            valid = _exact_number(value, expected)
        else:
            valid = value == expected and type(value) is type(expected)
        if not valid:
            raise ValueError(
                "solver.stage42c_oracle_replay.min_sigungu_coverage."
                f"{name} must remain exactly {expected}"
            )
    replay_extra = replay.get("extra_highs_options", {})
    expected_replay_extra = {
        "mip_pool_soft_limit": 20000,
        "mip_lp_age_limit": 10,
    }
    if not isinstance(replay_extra, dict) or replay_extra != expected_replay_extra:
        raise ValueError(
            "Stage4.2C min-coverage oracle replay extra options must remain exactly "
            f"{expected_replay_extra}"
        )
    oracle = cfg["oracle_certification"]
    if not isinstance(oracle, dict):
        raise ValueError("oracle_certification must be a mapping")
    exact_oracle_flags = {
        "enabled": True,
        "strategy": "rebuild_exact_threshold_models",
        "rebuild_model_in_official_run": True,
        "accept_diagnostic_artifacts": False,
        "candidate_count": 452,
        "pattern_count": 5242,
    }
    for name, expected in exact_oracle_flags.items():
        value = oracle.get(name)
        if value != expected or type(value) is not type(expected):
            raise ValueError(f"oracle_certification.{name} must remain exactly {expected!r}")

    references = oracle.get("frozen_references")
    if not isinstance(references, dict) or set(references) != set(FROZEN_ORACLE_REFERENCES):
        raise ValueError("oracle_certification.frozen_references must have the exact key set")
    for name, expected in FROZEN_ORACLE_REFERENCES.items():
        value = references.get(name)
        if type(expected) is float:
            valid = _same_binary64(value, expected)
        elif type(expected) is int:
            valid = type(value) is int and value == expected
        else:
            valid = value == expected
        if not valid or (type(expected) is int and type(value) is not int):
            raise ValueError(
                f"oracle_certification.frozen_references.{name} must remain exactly {expected!r}"
            )

    for section_name, expected_section in (
        ("min_highs", FROZEN_MIN_HIGHS_ORACLE),
        ("high_scip", FROZEN_HIGH_SCIP_ORACLE),
    ):
        observed = oracle.get(section_name)
        if not isinstance(observed, dict) or set(observed) != set(expected_section):
            raise ValueError(
                f"oracle_certification.{section_name} must have the exact frozen key set"
            )
        for name, expected in expected_section.items():
            value = observed.get(name)
            if type(expected) is float:
                valid = _same_binary64(value, expected)
            elif type(expected) is int:
                valid = type(value) is int and value == expected
            else:
                valid = value == expected and type(value) is type(expected)
            if not valid:
                raise ValueError(
                    f"oracle_certification.{section_name}.{name} must remain exactly {expected!r}"
                )

    hardware = cfg["hardware"]
    if type(hardware.get("solver_threads")) is not int or hardware["solver_threads"] != int(
        FROZEN_MIN_HIGHS_ORACLE["threads"]
    ):
        raise ValueError("hardware.solver_threads must match the frozen min-oracle thread count")
    if type(hardware.get("portfolio_workers")) is not int or hardware["portfolio_workers"] != int(
        FROZEN_HIGH_SCIP_ORACLE["workers"]
    ):
        raise ValueError("hardware.portfolio_workers must match the frozen SCIP portfolio")
    if (
        type(hardware.get("threads_per_portfolio_worker")) is not int
        or hardware["threads_per_portfolio_worker"]
        != int(FROZEN_HIGH_SCIP_ORACLE["threads_per_worker"])
    ):
        raise ValueError(
            "hardware.threads_per_portfolio_worker must match the frozen SCIP portfolio"
        )


def validate_parent_contracts(
    stage42: dict[str, Any],
    stage42c: dict[str, Any],
) -> None:
    """Fail closed if the frozen parent optimization semantics drift."""

    checks: list[tuple[str, Any, Any]] = [
        ("stage42.stage5_started", stage42.get("stage5_started"), False),
        ("stage42.matrix.primary_matrix_id", stage42.get("matrix", {}).get("primary_matrix_id"), "hard_5000m"),
        ("stage42.optimization.visit_count", stage42.get("optimization", {}).get("visit_count"), 20),
        (
            "stage42.optimization.equity_min_efficiency_fraction",
            stage42.get("optimization", {}).get("equity_min_efficiency_fraction"),
            0.60,
        ),
        (
            "stage42.optimization.known_overlap_penalty",
            stage42.get("optimization", {}).get("known_overlap_penalty"),
            1000.0,
        ),
        (
            "stage42.optimization.travel_penalty_scale",
            stage42.get("optimization", {}).get("travel_penalty_scale"),
            1.0,
        ),
        (
            "stage42.optimization.visit_location_deviation_penalty",
            stage42.get("optimization", {}).get("visit_location_deviation_penalty"),
            100.0,
        ),
        (
            "stage42.optimization.deterministic_tiebreak_scale",
            stage42.get("optimization", {}).get("deterministic_tiebreak_scale"),
            0.000001,
        ),
        (
            "stage42.candidate_expansion.reference_candidate_set",
            stage42.get("candidate_expansion", {}).get("reference_candidate_set"),
            "top3",
        ),
        (
            "stage42.candidate_expansion.compress_exact_equivalents",
            stage42.get("candidate_expansion", {}).get("compress_exact_equivalents"),
            True,
        ),
        (
            "stage42.candidate_expansion.safe_subset_dominance",
            stage42.get("candidate_expansion", {}).get("safe_subset_dominance"),
            True,
        ),
        (
            "stage42.candidate_expansion.dominance_max_group_size",
            stage42.get("candidate_expansion", {}).get("dominance_max_group_size"),
            80,
        ),
        (
            "stage42.candidate_expansion.max_visits_per_admin",
            stage42.get("candidate_expansion", {}).get("max_visits_per_admin"),
            2,
        ),
        (
            "stage42.candidate_expansion.max_visits_per_sigungu",
            stage42.get("candidate_expansion", {}).get("max_visits_per_sigungu"),
            4,
        ),
        (
            "stage42.candidate_expansion.one_per_coverage_cluster",
            stage42.get("candidate_expansion", {}).get("one_per_coverage_cluster"),
            True,
        ),
        (
            "stage42.solver.near_optimal_relative_gap_max",
            stage42.get("solver", {}).get("near_optimal_relative_gap_max"),
            FROZEN_GAP,
        ),
        (
            "stage42.solver.min_sigungu_ratio_integer_scale",
            stage42.get("solver", {}).get("min_sigungu_ratio_integer_scale"),
            1_000_000,
        ),
        (
            "stage42.solver.visit_deviation_exact_integer_formulation",
            stage42.get("solver", {}).get("visit_deviation_exact_integer_formulation"),
            True,
        ),
        (
            "stage42.solver.known_incumbent_objective_cutoff",
            stage42.get("solver", {}).get("known_incumbent_objective_cutoff"),
            False,
        ),
        ("stage42c.stage5_started", stage42c.get("stage5_started"), False),
        (
            "stage42c.solver.backend",
            stage42c.get("solver", {}).get("backend"),
            "direct_highspy",
        ),
        (
            "stage42c.solver.minimum_highs_version",
            stage42c.get("solver", {}).get("minimum_highs_version"),
            "1.8.0",
        ),
        (
            "stage42c.formulation.one_link_pattern_formulation",
            stage42c.get("formulation", {}).get("one_link_pattern_formulation"),
            True,
        ),
        (
            "stage42c.formulation.min_sigungu_mode",
            stage42c.get("formulation", {}).get("min_sigungu_mode"),
            "continuous_auxiliary_with_threshold_oracle",
        ),
        (
            "stage42c.formulation.integer_count_deviation_convex_hull",
            stage42c.get("formulation", {}).get("integer_count_deviation_convex_hull"),
            True,
        ),
        (
            "stage42c.formulation.stage_specific_variables",
            stage42c.get("formulation", {}).get("stage_specific_variables"),
            True,
        ),
        (
            "stage42c.formulation.min_sigungu_integer_scale",
            stage42c.get("formulation", {}).get("min_sigungu_integer_scale"),
            1_000_000,
        ),
        (
            "stage42c.formulation.preserve_candidate_universe",
            stage42c.get("formulation", {}).get("preserve_candidate_universe"),
            True,
        ),
        (
            "stage42c.formulation.preserve_lexicographic_retention",
            stage42c.get("formulation", {}).get("preserve_lexicographic_retention"),
            True,
        ),
        (
            "stage42c.gate.near_optimal_relative_gap_max",
            stage42c.get("gate", {}).get("near_optimal_relative_gap_max"),
            FROZEN_GAP,
        ),
        (
            "stage42c.gate.gap_threshold_must_not_be_relaxed",
            stage42c.get("gate", {}).get("gap_threshold_must_not_be_relaxed"),
            True,
        ),
    ]
    expected_retention = {
        "total_population": 0.99,
        "need_weighted": 0.99,
        "high_need": 0.99,
        "min_sigungu_coverage": 1.0,
    }
    retention = stage42.get("optimization", {}).get("objective_retention")
    if not isinstance(retention, dict) or set(retention) != set(expected_retention):
        raise ValueError("Frozen Stage4.2 objective_retention key set changed")
    checks.extend(
        (f"stage42.optimization.objective_retention.{name}", retention.get(name), expected)
        for name, expected in expected_retention.items()
    )
    for name, observed, expected in checks:
        if isinstance(expected, bool):
            valid = observed is expected
        elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
            valid = (
                isinstance(observed, (int, float))
                and not isinstance(observed, bool)
                and abs(float(observed) - float(expected)) <= 1e-12
            )
        else:
            valid = observed == expected and type(observed) is type(expected)
        if not valid:
            raise ValueError(f"Frozen parent contract mismatch: {name}={observed!r}, expected {expected!r}")


def set_thread_environment() -> None:
    """Give native CPU parallelism to HiGHS instead of helper BLAS libraries."""

    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        os.environ[key] = "1"
    os.environ.setdefault("PYTHONHASHSEED", "42")


def apply_process_priority(cfg: dict[str, Any]) -> str:
    requested = str(cfg.get("hardware", {}).get("process_priority", "normal")).lower()
    try:
        import psutil  # type: ignore

        process = psutil.Process()
        if platform.system().lower() == "windows":
            mapping = {
                "normal": psutil.NORMAL_PRIORITY_CLASS,
                "above_normal": psutil.ABOVE_NORMAL_PRIORITY_CLASS,
                "high": psutil.HIGH_PRIORITY_CLASS,
            }
            process.nice(mapping.get(requested, psutil.NORMAL_PRIORITY_CLASS))
        elif requested in {"above_normal", "high"}:
            # Negative nice values usually require elevated privileges; do not fail.
            try:
                process.nice(-5 if requested == "above_normal" else -10)
            except Exception:
                return "UNCHANGED_PERMISSION_DENIED"
        return requested.upper()
    except Exception as exc:
        return f"UNCHANGED::{type(exc).__name__}"
