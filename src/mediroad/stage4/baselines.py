from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .optimizer import _greedy_select, prepare_weights
from .types import OptimizationInput, ScenarioSolution


def _solution_from_indices(
    data: OptimizationInput,
    indices: list[int],
    name: str,
    config: dict[str, Any],
) -> ScenarioSolution:
    weights = prepare_weights(data, config)
    covered = np.zeros(len(data.grids), dtype=bool)
    for idx in indices:
        covered[data.coverage.getrow(idx).indices] = True
    selected = data.candidates.iloc[indices].copy()
    unique = int(weights.population[covered].sum())
    naive = int(np.asarray(data.coverage[indices] @ weights.population).ravel().sum())
    metrics = {
        "visit_count": len(indices),
        "unique_elderly_population_scaled": unique,
        "unique_elderly_population": unique / weights.population_scale,
        "need_weighted_population_scaled": int(weights.need_weighted[covered].sum()),
        "high_need_population_scaled": int(weights.high_need_population[covered].sum()),
        "naive_sum_elderly_population": naive / weights.population_scale,
        "redundancy_ratio": 1.0 - unique / naive if naive > 0 else 0.0,
        "covered_grid_count": int(covered.sum()),
    }
    return ScenarioSolution(
        scenario=name,
        catchment_id=data.catchment_id,
        visit_count=len(indices),
        status="BASELINE",
        selected=selected,
        covered_grid_mask=covered,
        metrics=metrics,
        stages=[],
    )


def _greedy_unique(data: OptimizationInput, weights: np.ndarray, visits: int) -> list[int]:
    selected: list[int] = []
    covered = np.zeros(len(data.grids), dtype=bool)
    used_cluster: set[str] = set()
    for _ in range(visits):
        best = None
        best_gain = -1.0
        for idx, row in data.candidates.iterrows():
            if idx in selected or str(row["cluster_id"]) in used_cluster:
                continue
            cols = data.coverage.getrow(idx).indices
            gain = float(weights[cols[~covered[cols]]].sum())
            if gain > best_gain:
                best_gain = gain
                best = int(idx)
        if best is None:
            break
        selected.append(best)
        used_cluster.add(str(data.candidates.iloc[best]["cluster_id"]))
        covered[data.coverage.getrow(best).indices] = True
    return selected


def build_baselines(data: OptimizationInput, config: dict[str, Any], visits: int) -> list[ScenarioSolution]:
    weights = prepare_weights(data, config)
    out: list[ScenarioSolution] = []
    constrained, _ = _greedy_select(data, weights, visits, "efficiency", config)
    out.append(
        _solution_from_indices(
            data,
            constrained,
            "BASELINE_CONSTRAINED_GREEDY_EFFICIENCY",
            config,
        )
    )
    raw = _greedy_unique(data, weights.population.astype(float), visits)
    out.append(_solution_from_indices(data, raw, "BASELINE_GREEDY_UNIQUE_ELDERLY", config))
    need = _greedy_unique(data, weights.need_weighted.astype(float), visits)
    out.append(_solution_from_indices(data, need, "BASELINE_GREEDY_NEED_WEIGHTED", config))

    # Need-first baseline: choose high-Need admin areas, then best raw-exposure venue within each.
    temp = data.candidates.copy()
    if "structural_need" in temp:
        temp["_need"] = pd.to_numeric(temp["structural_need"], errors="coerce").fillna(0.0)
    elif "stage1_need_score" in temp:
        temp["_need"] = pd.to_numeric(temp["stage1_need_score"], errors="coerce").fillna(0.0)
    else:
        admin_need = data.grids.groupby("admin_code")["stage1_need_score"].mean()
        temp["_need"] = temp["admin_code"].map(admin_need).fillna(0.0)
    temp["_raw"] = pd.to_numeric(temp.get("raw_elderly_exposure", 0.0), errors="coerce").fillna(0.0)
    chosen = (
        temp.sort_values(["_need", "_raw", "venue_id"], ascending=[False, False, True])
        .drop_duplicates("admin_code")
        .head(visits)
        .index.astype(int)
        .tolist()
    )
    out.append(_solution_from_indices(data, chosen, "BASELINE_NEED_TOP_ADMIN", config))

    # Equal-sigungu baseline: round-robin highest raw candidate per sigungu.
    groups = {
        key: group.sort_values(["_raw", "_need", "venue_id"], ascending=[False, False, True])
        for key, group in temp.groupby("sigungu")
    }
    equal: list[int] = []
    cursor = {key: 0 for key in groups}
    keys = sorted(groups)
    while len(equal) < visits:
        progressed = False
        for key in keys:
            group = groups[key]
            while cursor[key] < len(group):
                idx = int(group.index[cursor[key]])
                cursor[key] += 1
                if idx not in equal:
                    equal.append(idx)
                    progressed = True
                    break
            if len(equal) >= visits:
                break
        if not progressed:
            break
    out.append(_solution_from_indices(data, equal, "BASELINE_EQUAL_SIGUNGU_ROUND_ROBIN", config))
    return out
