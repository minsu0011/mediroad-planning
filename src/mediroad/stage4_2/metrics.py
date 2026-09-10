from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

from .types import ScenarioResult


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def compare_candidate_results(reference: ScenarioResult, alternative: ScenarioResult) -> dict[str, Any]:
    ref_admin = set(reference.selected_frame["admin_code"].astype(str))
    alt_admin = set(alternative.selected_frame["admin_code"].astype(str))
    ref_cluster = set(reference.selected_frame["cluster_id"].astype(str))
    alt_cluster = set(alternative.selected_frame["cluster_id"].astype(str))
    ref_venue = set(reference.selected_venue_ids)
    alt_venue = set(alternative.selected_venue_ids)
    row: dict[str, Any] = {
        "reference_candidate_set": reference.candidate_set,
        "alternative_candidate_set": alternative.candidate_set,
        "scenario": alternative.scenario,
        "reference_certified": reference.certified,
        "alternative_certified": alternative.certified,
        "venue_jaccard": jaccard(ref_venue, alt_venue),
        "admin_jaccard": jaccard(ref_admin, alt_admin),
        "coverage_cluster_jaccard": jaccard(ref_cluster, alt_cluster),
    }
    for metric in [
        "unique_elderly_population",
        "need_weighted_population",
        "high_need_population",
        "min_sigungu_coverage_ratio",
    ]:
        ref = float(reference.metrics.get(metric, np.nan))
        alt = float(alternative.metrics.get(metric, np.nan))
        row[f"reference_{metric}"] = ref
        row[f"alternative_{metric}"] = alt
        row[f"delta_{metric}"] = alt - ref
        row[f"relative_delta_{metric}"] = (alt - ref) / abs(ref) if np.isfinite(ref) and abs(ref) > 1e-12 else np.nan
    return row


def pairwise_plan_stability(results: list[ScenarioResult]) -> pd.DataFrame:
    rows = []
    for a, b in combinations(results, 2):
        rows.append(
            {
                "plan_a": f"{a.candidate_set}:{a.scenario}",
                "plan_b": f"{b.candidate_set}:{b.scenario}",
                "venue_jaccard": jaccard(set(a.selected_venue_ids), set(b.selected_venue_ids)),
                "admin_jaccard": jaccard(
                    set(a.selected_frame["admin_code"].astype(str)),
                    set(b.selected_frame["admin_code"].astype(str)),
                ),
                "cluster_jaccard": jaccard(
                    set(a.selected_frame["cluster_id"].astype(str)),
                    set(b.selected_frame["cluster_id"].astype(str)),
                ),
                "a_certified": a.certified,
                "b_certified": b.certified,
            }
        )
    return pd.DataFrame(rows)
