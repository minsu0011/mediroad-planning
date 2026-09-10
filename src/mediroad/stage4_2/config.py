from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .errors import ContractError
from .utils import sha256_file


@dataclass(frozen=True)
class LoadedConfig:
    path: Path
    data: dict[str, Any]
    sha256: str


def load_config(path: Path) -> LoadedConfig:
    if not path.exists():
        raise FileNotFoundError(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ContractError(f"Stage 4.2 config must be a mapping: {path}")
    validate_config(raw)
    return LoadedConfig(path=path, data=raw, sha256=sha256_file(path))


def validate_config(cfg: dict[str, Any]) -> None:
    required = ["version", "paths", "solver", "candidate_expansion", "field_validation", "operational_inputs"]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ContractError(f"Config missing sections: {missing}")
    solver = cfg["solver"]
    gap = float(solver.get("near_optimal_relative_gap_max", -1))
    if not 0 <= gap < 1:
        raise ContractError("solver.near_optimal_relative_gap_max must be in [0,1)")
    if int(solver.get("min_sigungu_ratio_integer_scale", 0)) != 1_000_000:
        raise ContractError("solver.min_sigungu_ratio_integer_scale must remain exactly 1000000")
    if solver.get("visit_deviation_exact_integer_formulation") is not True:
        raise ContractError("solver.visit_deviation_exact_integer_formulation must remain true")
    if solver.get("known_incumbent_objective_cutoff") is not False:
        raise ContractError("solver.known_incumbent_objective_cutoff must remain false after diagnostic evaluation")
    counts = cfg.get("optimization", {}).get("visit_count", 20)
    if int(counts) <= 0:
        raise ContractError("optimization.visit_count must be positive")
    allowed_sets = {"top3", "top5", "coarse_pareto", "verified", "resolved"}
    requested = set(cfg["candidate_expansion"].get("candidate_sets", []))
    bad = requested - allowed_sets
    if bad:
        raise ContractError(f"Unsupported candidate sets: {sorted(bad)}")
    statuses = cfg["field_validation"].get("allowed_statuses", ["YES", "NO", "UNKNOWN"])
    normalized_statuses = {
        "YES" if value is True else "NO" if value is False else str(value).strip().upper()
        for value in statuses
    }
    if normalized_statuses != {"YES", "NO", "UNKNOWN"}:
        raise ContractError("field_validation.allowed_statuses must be YES/NO/UNKNOWN")
    runtime = cfg.get("runtime", {})
    workers = int(runtime.get("candidate_workers", 1))
    threads = int(runtime.get("highs_threads_per_worker", 1))
    if workers < 1 or threads < 1 or workers * threads > 8:
        raise ContractError("runtime candidate_workers * highs_threads_per_worker must be in 1..8")
    int(runtime.get("highs_random_seed", 42))
