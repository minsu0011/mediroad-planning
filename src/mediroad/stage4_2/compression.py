from __future__ import annotations

import hashlib
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import sparse

from .errors import ContractError
from .types import GridPatternData


def compress_grid_patterns(
    coverage: sparse.csr_matrix,
    candidate_ids: np.ndarray,
    grid_policy: pd.DataFrame,
) -> GridPatternData:
    """Losslessly aggregate grids sharing candidate incidence and policy sigungu.

    Grouping by sigungu is required because minimum beneficiary coverage is a policy
    constraint. Objective weights are summed, not averaged. The original grid indices are
    retained for exact post-solve recomputation and audit.
    """
    if coverage.shape[0] != len(candidate_ids):
        raise ContractError("coverage candidate rows do not match candidate IDs")
    if coverage.shape[1] != len(grid_policy):
        raise ContractError("coverage grid columns do not match grid policy rows")
    csc = coverage.tocsc()
    groups: dict[tuple[str, bytes], list[int]] = defaultdict(list)
    for grid_idx in range(csc.shape[1]):
        start, end = csc.indptr[grid_idx], csc.indptr[grid_idx + 1]
        coverers = np.asarray(csc.indices[start:end], dtype=np.int32)
        sigungu = str(grid_policy.iloc[grid_idx]["sigungu"])
        digest = hashlib.sha256(coverers.tobytes()).digest()
        groups[(sigungu, digest)].append(grid_idx)

    pattern_coverers: list[np.ndarray] = []
    population: list[float] = []
    need_weighted: list[float] = []
    high_need: list[float] = []
    sigungu_values: list[str] = []
    source_counts: list[int] = []
    source_indices: list[np.ndarray] = []

    # Hash collisions are cryptographically negligible, but verify exact incidence in each group.
    for (sigungu, _), grid_indices_list in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[1][0])):
        by_exact: dict[tuple[int, ...], list[int]] = defaultdict(list)
        for grid_idx in grid_indices_list:
            start, end = csc.indptr[grid_idx], csc.indptr[grid_idx + 1]
            exact = tuple(int(x) for x in csc.indices[start:end])
            by_exact[exact].append(grid_idx)
        for exact, sub_indices in sorted(by_exact.items(), key=lambda kv: kv[1][0]):
            idx = np.asarray(sub_indices, dtype=np.int64)
            pattern_coverers.append(np.asarray(exact, dtype=np.int32))
            population.append(float(grid_policy.iloc[idx]["elderly_population"].sum()))
            need_weighted.append(float(grid_policy.iloc[idx]["need_weighted_population"].sum()))
            high_need.append(float(grid_policy.iloc[idx]["high_need_population"].sum()))
            sigungu_values.append(sigungu)
            source_counts.append(len(idx))
            source_indices.append(idx)

    return GridPatternData(
        candidate_ids=np.asarray(candidate_ids, dtype=str),
        pattern_coverers=pattern_coverers,
        population=np.asarray(population, dtype=float),
        need_weighted=np.asarray(need_weighted, dtype=float),
        high_need_population=np.asarray(high_need, dtype=float),
        sigungu=np.asarray(sigungu_values, dtype=str),
        source_grid_count=np.asarray(source_counts, dtype=np.int32),
        source_grid_indices=source_indices,
    )


def pattern_incidence_csr(patterns: GridPatternData) -> sparse.csr_matrix:
    """Return pattern × candidate binary incidence."""
    rows: list[int] = []
    cols: list[int] = []
    for p, coverers in enumerate(patterns.pattern_coverers):
        rows.extend([p] * len(coverers))
        cols.extend(int(x) for x in coverers)
    data = np.ones(len(rows), dtype=np.int8)
    return sparse.csr_matrix((data, (rows, cols)), shape=(patterns.n_patterns, patterns.n_candidates))


def verify_pattern_reconstruction(
    patterns: GridPatternData,
    coverage: sparse.csr_matrix,
    selected_indices: np.ndarray,
    grid_policy: pd.DataFrame,
    *,
    atol: float = 1e-6,
) -> dict[str, float | bool]:
    selected_indices = np.asarray(selected_indices, dtype=int)
    exact_mask = np.asarray(coverage[selected_indices].max(axis=0).toarray()).ravel() > 0 if len(selected_indices) else np.zeros(coverage.shape[1], dtype=bool)
    exact_pop = float(grid_policy.loc[exact_mask, "elderly_population"].sum())
    exact_need = float(grid_policy.loc[exact_mask, "need_weighted_population"].sum())
    exact_high = float(grid_policy.loc[exact_mask, "high_need_population"].sum())
    selected_set = set(int(x) for x in selected_indices)
    pattern_mask = np.fromiter(
        (any(int(v) in selected_set for v in coverers) for coverers in patterns.pattern_coverers),
        dtype=bool,
        count=patterns.n_patterns,
    )
    pattern_pop = float(patterns.population[pattern_mask].sum())
    pattern_need = float(patterns.need_weighted[pattern_mask].sum())
    pattern_high = float(patterns.high_need_population[pattern_mask].sum())
    return {
        "exact_population": exact_pop,
        "pattern_population": pattern_pop,
        "exact_need_weighted": exact_need,
        "pattern_need_weighted": pattern_need,
        "exact_high_need": exact_high,
        "pattern_high_need": pattern_high,
        "population_match": bool(abs(exact_pop - pattern_pop) <= atol),
        "need_match": bool(abs(exact_need - pattern_need) <= atol),
        "high_need_match": bool(abs(exact_high - pattern_high) <= atol),
    }
