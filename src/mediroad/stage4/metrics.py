from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .types import OptimizationInput, ScenarioSolution
from .utils import gini, hhi


def enrich_solution_metrics(solution: ScenarioSolution, data: OptimizationInput) -> dict[str, Any]:
    metrics = dict(solution.metrics)
    selected = solution.selected.copy()
    if selected.empty:
        return metrics
    counts = selected["sigungu"].astype(str).value_counts()
    metrics.update(
        {
            "visit_location_sigungu_count": int(len(counts)),
            "visit_location_hhi": hhi(counts.to_numpy()),
            "visit_location_gini": gini(counts.to_numpy()),
            "selected_admin_count": int(selected["admin_code"].astype(str).nunique()),
            "selected_cluster_count": int(selected["cluster_id"].astype(str).nunique()),
        }
    )
    travel = pd.to_numeric(selected.get("travel_minutes", np.nan), errors="coerce")
    if isinstance(travel, pd.Series) and travel.notna().any():
        metrics["travel_minutes_sum_proxy"] = float(travel.sum())
        metrics["travel_minutes_mean_proxy"] = float(travel.mean())
        metrics["travel_minutes_max_proxy"] = float(travel.max())
    else:
        metrics["travel_minutes_sum_proxy"] = None
        metrics["travel_minutes_mean_proxy"] = None
        metrics["travel_minutes_max_proxy"] = None
    if solution.bundle_assignments is not None and not solution.bundle_assignments.empty:
        bundle_counts = solution.bundle_assignments["bundle_id"].astype(str).value_counts().to_dict()
        metrics["bundle_counts"] = bundle_counts
        metrics["bundle_count_unique"] = len(bundle_counts)
    return metrics


def solution_row(solution: ScenarioSolution) -> dict[str, Any]:
    row: dict[str, Any] = {
        "scenario": solution.scenario,
        "catchment_id": solution.catchment_id,
        "visit_count": solution.visit_count,
        "status": solution.status,
    }
    for key, value in solution.metrics.items():
        if isinstance(value, (dict, list, tuple)):
            continue
        row[key] = value
    return row


def selection_frequency(solutions: list[ScenarioSolution], candidates: pd.DataFrame) -> pd.DataFrame:
    total = len(solutions)
    records: list[dict[str, Any]] = []
    for solution in solutions:
        for row in solution.selected.itertuples(index=False):
            records.append(
                {
                    "scenario": solution.scenario,
                    "catchment_id": solution.catchment_id,
                    "venue_id": str(row.venue_id),
                    "admin_code": str(row.admin_code),
                    "sigungu": str(row.sigungu),
                    "cluster_id": str(row.cluster_id),
                }
            )
    if not records:
        return pd.DataFrame()
    selected = pd.DataFrame(records)
    venue = (
        selected.groupby(["venue_id", "admin_code", "sigungu", "cluster_id"], as_index=False)
        .agg(selection_count=("scenario", "size"), scenario_count=("scenario", "nunique"), catchment_count=("catchment_id", "nunique"))
    )
    venue["selection_frequency"] = venue["selection_count"] / max(1, total)
    meta_cols = [c for c in ["venue_id", "venue_name", "venue_type", "fallback_venue_1", "fallback_venue_2"] if c in candidates]
    venue = venue.merge(candidates[meta_cols].drop_duplicates("venue_id"), on="venue_id", how="left")
    return venue.sort_values(["selection_frequency", "selection_count", "venue_id"], ascending=[False, False, True])


def compare_solutions(a: ScenarioSolution, b: ScenarioSolution) -> dict[str, Any]:
    a_ids = set(a.selected["venue_id"].astype(str))
    b_ids = set(b.selected["venue_id"].astype(str))
    union = a_ids | b_ids
    return {
        "scenario_a": a.scenario,
        "catchment_a": a.catchment_id,
        "scenario_b": b.scenario,
        "catchment_b": b.catchment_id,
        "venue_jaccard": len(a_ids & b_ids) / max(1, len(union)),
        "venue_overlap_count": len(a_ids & b_ids),
        "admin_jaccard": len(set(a.selected["admin_code"].astype(str)) & set(b.selected["admin_code"].astype(str)))
        / max(1, len(set(a.selected["admin_code"].astype(str)) | set(b.selected["admin_code"].astype(str)))),
    }

