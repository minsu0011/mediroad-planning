from __future__ import annotations

import os
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class HardwareProfile:
    logical_threads: int
    highs_threads: int
    memory_limit_gb: float
    gpu_seed_search: bool
    native_windows: bool
    wsl: bool


def load_yaml(path: Path) -> dict[str, Any]:
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


def resolve_hardware(cfg: dict[str, Any]) -> HardwareProfile:
    section = cfg.get("hardware", {})
    logical = int(os.cpu_count() or 8)
    configured_threads = int(section.get("highs_threads", 8))
    highs_threads = max(1, min(logical, configured_threads))
    native_windows = platform.system().lower() == "windows" and not is_wsl()
    # Native Windows is supported. Keep a separate cap because extension crashes are
    # harder to recover from than a WSL/Linux worker failure. The direct backend resets
    # the HiGHS global scheduler before and after each solve, so a fixed small parallel
    # count is safe when explicitly enabled in the frozen config.
    if native_windows:
        if bool(section.get("allow_native_windows_parallel", False)):
            native_cap = max(1, int(section.get("native_windows_highs_threads", 4)))
            highs_threads = min(highs_threads, native_cap)
        else:
            highs_threads = 1
    return HardwareProfile(
        logical_threads=logical,
        highs_threads=highs_threads,
        memory_limit_gb=float(section.get("memory_limit_gb", 28.0)),
        gpu_seed_search=bool(section.get("gpu_seed_search", True)),
        native_windows=native_windows,
        wsl=is_wsl(),
    )


def validate_certification_config(cfg: dict[str, Any]) -> None:
    required = ["version", "hardware", "solver", "formulation", "gate", "paths"]
    missing = [key for key in required if key not in cfg]
    if missing:
        raise ValueError(f"Certification config missing sections: {missing}")
    gap = float(cfg["gate"].get("near_optimal_relative_gap_max", -1.0))
    if not (0.0 <= gap < 1.0):
        raise ValueError("gate.near_optimal_relative_gap_max must be in [0,1)")
    if abs(gap - 0.005) > 1e-12:
        raise ValueError("The frozen MEDIROAD certification gap must remain 0.005")
    mode = str(
        cfg["formulation"].get(
            "min_sigungu_mode", "continuous_auxiliary_with_threshold_oracle"
        )
    )
    if mode != "continuous_auxiliary_with_threshold_oracle":
        raise ValueError("Unsupported formulation.min_sigungu_mode")
    if not bool(
        cfg["formulation"].get("integer_count_deviation_convex_hull", True)
    ):
        raise ValueError("Stage4.2C requires the integer-count deviation convex hull")
    if not bool(cfg["formulation"].get("one_link_pattern_formulation", True)):
        raise ValueError("Stage4.2C requires one-link pattern formulation")
    if int(cfg["formulation"].get("min_sigungu_integer_scale", 1_000_000)) <= 0:
        raise ValueError("formulation.min_sigungu_integer_scale must be positive")
    if int(cfg["hardware"].get("highs_threads", 8)) <= 0:
        raise ValueError("hardware.highs_threads must be positive")
    if int(cfg["hardware"].get("native_windows_highs_threads", 4)) <= 0:
        raise ValueError("hardware.native_windows_highs_threads must be positive")
    oracle = cfg.get("oracle", {})
    guided = oracle.get("guided_objective_enabled", False)
    target_stop = oracle.get("stop_at_objective_target", False)
    if not isinstance(guided, bool) or not isinstance(target_stop, bool):
        raise ValueError("Guided oracle switches must be booleans")
    if target_stop and not guided:
        raise ValueError("stop_at_objective_target requires guided_objective_enabled")
    solver = cfg.get("solver", {})
    efforts = solver.get("oracle_heuristic_effort_by_objective", {})
    if not isinstance(efforts, dict):
        raise ValueError("solver.oracle_heuristic_effort_by_objective must be a mapping")
    for name, value in efforts.items():
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"Invalid oracle heuristic effort for {name}: {value}")


def set_thread_environment() -> None:
    """Prevent BLAS/OpenMP oversubscription; HiGHS owns native CPU parallelism."""

    for key in [
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ]:
        os.environ[key] = "1"
    os.environ.setdefault("PYTHONHASHSEED", "42")
