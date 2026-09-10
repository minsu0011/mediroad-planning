from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

FROZEN_THRESHOLD_FINGERPRINT = "b0a1a2c6dc6f1d1e7c3ae1141700fbb3c54854b03746d2e45d2d0ba33c3b1ad4"


@dataclass(frozen=True)
class CandidateThresholds:
    max_unique_coverage_loss_fraction: float = 0.01
    max_need_weighted_loss_fraction: float = 0.01
    max_high_need_loss_fraction: float = 0.02
    max_min_sigungu_coverage_loss: float = 0.02
    min_selected_admin_jaccard: float = 0.50
    min_selected_coverage_cluster_jaccard: float = 0.50
    max_solver_relative_gap: float = 0.005

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected mapping: {path}")
    return value


def set_thread_environment() -> None:
    for key in [
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
    ]:
        os.environ[key] = "1"
    os.environ.setdefault("PYTHONHASHSEED", "42")


def validate_config(cfg: dict[str, Any]) -> None:
    required = {"version", "paths", "hardware", "candidate_expansion", "solver", "oracle", "scip_portfolio"}
    missing = sorted(required - set(cfg))
    if missing:
        raise ValueError(f"Stage4.2G config missing sections: {missing}")
    section = cfg["candidate_expansion"]
    if list(section.get("candidate_sets", [])) != ["top5", "coarse_pareto"]:
        raise ValueError("Stage4.2G candidate sets are frozen to top5 -> coarse_pareto")
    if str(section.get("reference_candidate_set")) != "top3":
        raise ValueError("Stage4.2G reference candidate set must remain top3")
    if list(section.get("required_scenarios", [])) != ["efficiency", "balanced"]:
        raise ValueError("Stage4.2G required scenarios must remain efficiency/balanced")
    thresholds = CandidateThresholds(
        max_unique_coverage_loss_fraction=float(section["max_unique_coverage_loss_fraction"]),
        max_need_weighted_loss_fraction=float(section["max_need_weighted_loss_fraction"]),
        max_high_need_loss_fraction=float(section["max_high_need_loss_fraction"]),
        max_min_sigungu_coverage_loss=float(section["max_min_sigungu_coverage_loss"]),
        min_selected_admin_jaccard=float(section["min_selected_admin_jaccard"]),
        min_selected_coverage_cluster_jaccard=float(section["min_selected_coverage_cluster_jaccard"]),
        max_solver_relative_gap=float(section["max_solver_relative_gap"]),
    )
    if thresholds.fingerprint != FROZEN_THRESHOLD_FINGERPRINT:
        raise ValueError(
            "Candidate sensitivity thresholds changed; expected frozen fingerprint "
            f"{FROZEN_THRESHOLD_FINGERPRINT}, got {thresholds.fingerprint}"
        )
    if abs(float(cfg["solver"]["near_optimal_relative_gap_max"]) - 0.005) > 1e-12:
        raise ValueError("Stage4.2G solver certification gap is frozen at 0.005")
    if int(cfg["hardware"].get("highs_threads", 0)) != 8:
        raise ValueError("Official Stage4.2G uses all 8 logical CPU threads for HiGHS")
    if not 20 <= float(cfg["hardware"].get("memory_soft_limit_gib", 0)) <= 28:
        raise ValueError("Stage4.2G RAM soft limit must stay within 20..28 GiB on the 32 GiB host")
    if int(cfg["scip_portfolio"].get("workers", 0)) < 1 or int(cfg["scip_portfolio"].get("workers", 0)) > 6:
        raise ValueError("SCIP portfolio workers must be in 1..6")
    if not bool(cfg.get("gpu_lns", {}).get("enabled", False)):
        raise ValueError("Official Stage4.2G requires GPU LNS to be enabled")
    if bool(cfg.get("stage5_started", False)):
        raise ValueError("Stage4.2G must not start Stage5")
