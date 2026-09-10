from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from .types import CandidateProblem


def load_base_stage42_config(path: Path) -> dict[str, Any]:
    try:
        from mediroad.stage4_2.config import load_config
    except ImportError:
        import yaml

        value = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"Expected mapping: {path}")
        return value
    return load_config(path).data


def _fallback_prepare(root: Path, candidate_set: str, config: dict[str, Any], cache: dict[str, Any]) -> dict[str, Any]:
    from mediroad.stage4_2.candidates import (
        build_candidate_set,
        compress_exact_equivalent_candidates,
        prune_safe_dominated_candidates,
    )
    from mediroad.stage4_2.compression import compress_grid_patterns
    from mediroad.stage4_2.data import (
        align_coverage,
        bundle_table_from_interface,
        load_coverage,
        normalize_grid_policy,
        normalize_stage3_interface,
        venue_table_from_interface,
    )
    from mediroad.stage4_2.discovery import discover_stage3, discover_stage41
    from mediroad.stage4_2.utils import read_table

    stage3 = cache.setdefault("stage3", discover_stage3(root))
    stage41 = cache.setdefault("stage41", discover_stage41(root))
    interface = cache.setdefault("interface", normalize_stage3_interface(read_table(stage3.interface)))
    venues = cache.setdefault("venues", venue_table_from_interface(interface))
    bundles = cache.setdefault("bundles", bundle_table_from_interface(interface))
    membership = cache.setdefault("membership", read_table(stage41.candidate_membership))
    coverage_all = cache.setdefault("coverage_all", load_coverage(stage3, config["matrix"]["primary_matrix_id"]))
    grid = cache.setdefault("grid", normalize_grid_policy(read_table(stage3.grid_policy)))
    grid = grid.set_index("grid_id").loc[coverage_all.grid_ids.astype(str)].reset_index()
    candidates = build_candidate_set(venues, membership, candidate_set)
    coverage = align_coverage(
        coverage_all,
        candidates["venue_id"].astype(str).to_numpy(),
        grid["grid_id"].astype(str).to_numpy(),
    )
    ledgers: list[pd.DataFrame] = []
    if bool(config.get("candidate_expansion", {}).get("compress_exact_equivalents", True)):
        result = compress_exact_equivalent_candidates(candidates, coverage)
        candidates, coverage = result.candidates, result.coverage
        ledger = result.ledger.copy()
        ledger.insert(0, "reduction_stage", "EXACT_EQUIVALENCE")
        ledgers.append(ledger)
    if bool(config.get("candidate_expansion", {}).get("safe_subset_dominance", True)):
        result = prune_safe_dominated_candidates(
            candidates,
            coverage,
            max_group_size=int(config.get("candidate_expansion", {}).get("dominance_max_group_size", 80)),
        )
        candidates, coverage = result.candidates, result.coverage
        ledger = result.ledger.copy()
        ledger.insert(0, "reduction_stage", "SAFE_SUBSET_DOMINANCE")
        ledgers.append(ledger)
    patterns = compress_grid_patterns(coverage, candidates["venue_id"].astype(str).to_numpy(), grid)
    return {
        "stage3": stage3,
        "stage41": stage41,
        "interface": interface,
        "venues": venues,
        "bundles": bundles,
        "membership": membership,
        "coverage_all": coverage_all,
        "grid": grid,
        "candidates": candidates,
        "coverage": coverage,
        "patterns": patterns,
        "reduction_ledger": pd.concat(ledgers, ignore_index=True) if ledgers else pd.DataFrame(),
    }




def _compose_venue_alias_map(ledger: pd.DataFrame, retained_ids: set[str]) -> dict[str, str]:
    """Compose sequential compression ledgers into original->final representative IDs."""

    aliases: dict[str, str] = {value: value for value in retained_ids}
    if ledger.empty or "venue_id" not in ledger or "representative_venue_id" not in ledger:
        return aliases
    for _, row in ledger.iterrows():
        source = str(row["venue_id"])
        target = str(row["representative_venue_id"])
        # Redirect aliases that pointed to an intermediate representative.
        for key, value in list(aliases.items()):
            if value == source:
                aliases[key] = target
        aliases[source] = target
        aliases.setdefault(target, target)
    # Chase any remaining chains and retain only resolvable final representatives.
    for key in list(aliases):
        value = aliases[key]
        seen: set[str] = set()
        while value in aliases and aliases[value] != value and value not in seen:
            seen.add(value)
            value = aliases[value]
        if value in retained_ids:
            aliases[key] = value
        elif key not in retained_ids:
            aliases.pop(key, None)
    return aliases

def prepare_candidate_problem(
    root: Path,
    candidate_set: str,
    config: dict[str, Any],
    *,
    cache: dict[str, Any] | None = None,
) -> CandidateProblem:
    cache = cache if cache is not None else {}
    try:
        from mediroad.stage4_2.computational import _prepare_candidate_problem

        signature = inspect.signature(_prepare_candidate_problem)
        kwargs: dict[str, Any] = {}
        if "stage3_cache" in signature.parameters:
            kwargs["stage3_cache"] = cache
        elif "cache" in signature.parameters:
            kwargs["cache"] = cache
        payload = _prepare_candidate_problem(root, candidate_set, config, **kwargs)
    except (ImportError, AttributeError):
        payload = _fallback_prepare(root, candidate_set, config, cache)
    required = ["candidates", "patterns", "coverage", "grid", "stage41", "bundles"]
    missing = [name for name in required if name not in payload]
    if missing:
        raise RuntimeError(f"Stage4.2 adapter missing fields: {missing}")
    candidates = payload["candidates"].reset_index(drop=True).copy()
    coverage = payload["coverage"].tocsr()
    if len(candidates) != coverage.shape[0]:
        raise RuntimeError("Candidate/coverage row mismatch")
    if len(candidates) != int(payload["patterns"].n_candidates):
        raise RuntimeError("Candidate/pattern row mismatch")
    reduction_ledger = payload.get("reduction_ledger", pd.DataFrame()).copy()
    retained_ids = set(candidates["venue_id"].astype(str))
    return CandidateProblem(
        candidate_set=candidate_set,
        candidates=candidates,
        patterns=payload["patterns"],
        coverage=coverage,
        grid=payload["grid"].reset_index(drop=True).copy(),
        stage41=payload["stage41"],
        bundles=payload["bundles"].copy(),
        reduction_ledger=reduction_ledger,
        venue_alias_map=_compose_venue_alias_map(reduction_ledger, retained_ids),
        source_payload=payload,
    )


def constrained_greedy(
    candidates: pd.DataFrame,
    coverage: sparse.csr_matrix,
    grid_population: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, float]:
    try:
        from mediroad.stage4_2.optimizer import constrained_greedy as current_greedy

        return current_greedy(candidates, coverage, grid_population, config)
    except Exception:
        pass
    visit_count = int(config.get("optimization", {}).get("visit_count", 20))
    max_admin = int(config.get("candidate_expansion", {}).get("max_visits_per_admin", 2))
    max_sigungu = int(config.get("candidate_expansion", {}).get("max_visits_per_sigungu", 4))
    one_cluster = bool(config.get("candidate_expansion", {}).get("one_per_coverage_cluster", True))
    selected: list[int] = []
    covered = np.zeros(coverage.shape[1], dtype=bool)
    admin_counts: dict[str, int] = {}
    sig_counts: dict[str, int] = {}
    clusters: set[str] = set()
    for _ in range(visit_count):
        best_i: int | None = None
        best_gain = -1.0
        for i, row in candidates.iterrows():
            if i in selected:
                continue
            admin, sig = str(row["admin_code"]), str(row["sigungu"])
            raw_cluster = row.get("cluster_id", row["venue_id"])
            cluster = str(row["venue_id"]) if pd.isna(raw_cluster) else str(raw_cluster)
            if admin_counts.get(admin, 0) >= max_admin or sig_counts.get(sig, 0) >= max_sigungu:
                continue
            if one_cluster and cluster in clusters:
                continue
            mask = np.asarray(coverage.getrow(i).toarray()).ravel() > 0
            gain = float(grid_population[mask & ~covered].sum())
            if gain > best_gain:
                best_i, best_gain = int(i), gain
        if best_i is None:
            raise RuntimeError("Constrained greedy could not fill visit_count")
        row = candidates.iloc[best_i]
        selected.append(best_i)
        covered |= np.asarray(coverage.getrow(best_i).toarray()).ravel() > 0
        admin = str(row["admin_code"])
        sig = str(row["sigungu"])
        raw_cluster = row.get("cluster_id", row["venue_id"])
        cluster = str(row["venue_id"]) if pd.isna(raw_cluster) else str(raw_cluster)
        admin_counts[admin] = admin_counts.get(admin, 0) + 1
        sig_counts[sig] = sig_counts.get(sig, 0) + 1
        clusters.add(cluster)
    return np.asarray(selected, dtype=int), float(grid_population[covered].sum())
