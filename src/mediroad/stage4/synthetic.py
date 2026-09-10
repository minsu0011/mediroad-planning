from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

from .types import OptimizationInput


def make_synthetic_input(
    n_admin: int = 12,
    venues_per_admin: int = 3,
    grids_per_admin: int = 20,
    bundles: int = 5,
    seed: int = 42,
) -> OptimizationInput:
    rng = np.random.default_rng(seed)
    n_venues = n_admin * venues_per_admin
    n_grids = n_admin * grids_per_admin
    admin_codes = [f"A{i:03d}" for i in range(n_admin)]
    sigungu_codes = [f"S{i % 4:02d}" for i in range(n_admin)]

    candidates = []
    for a, (admin, sigungu) in enumerate(zip(admin_codes, sigungu_codes)):
        for j in range(venues_per_admin):
            candidates.append(
                {
                    "venue_id": f"V{a:03d}_{j}",
                    "venue_name": f"Venue {a}-{j}",
                    "admin_code": admin,
                    "sigungu": sigungu,
                    "cluster_id": f"C{a:03d}_{j // 2}",
                    "shortlist_rank": j + 1,
                    "pareto_tier": 1 + (j > 0),
                    "travel_minutes": float(15 + a + j),
                    "known_overlap": "unknown",
                    "raw_elderly_exposure": float(1000 + 50 * a - 20 * j),
                    "need_weighted_exposure": float(800 + 30 * (n_admin - a) - 10 * j),
                    "structural_need": float(100 - 4 * a),
                }
            )
    candidates_df = pd.DataFrame(candidates)

    grids = []
    for a, (admin, sigungu) in enumerate(zip(admin_codes, sigungu_codes)):
        for g in range(grids_per_admin):
            grids.append(
                {
                    "grid_id": f"G{a:03d}_{g:03d}",
                    "admin_code": admin,
                    "sigungu": sigungu,
                    "elderly65_population": float(rng.uniform(1, 12)),
                    "stage1_need_score": float(90 - 5 * a + rng.normal(0, 1)),
                }
            )
    grids_df = pd.DataFrame(grids)

    rows: list[int] = []
    cols: list[int] = []
    for venue_idx in range(n_venues):
        admin_idx = venue_idx // venues_per_admin
        own_start = admin_idx * grids_per_admin
        own = np.arange(own_start, own_start + grids_per_admin)
        rows.extend([venue_idx] * len(own))
        cols.extend(own.tolist())
        # Cross-boundary coverage to neighbouring admin.
        if admin_idx + 1 < n_admin:
            nxt = np.arange((admin_idx + 1) * grids_per_admin, (admin_idx + 1) * grids_per_admin + 5)
            rows.extend([venue_idx] * len(nxt))
            cols.extend(nxt.tolist())
    coverage = sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.int8), (np.asarray(rows), np.asarray(cols))),
        shape=(n_venues, n_grids),
    )

    bundle_rows = []
    for venue in candidates_df["venue_id"]:
        for b in range(bundles):
            bundle_rows.append(
                {
                    "venue_id": venue,
                    "bundle_id": f"B{b}",
                    "bundle_value": float(rng.uniform(0, 100) + 10 * b),
                }
            )
    bundle_df = pd.DataFrame(bundle_rows)
    data = OptimizationInput(
        candidates=candidates_df,
        grids=grids_df,
        coverage=coverage,
        venue_ids=candidates_df["venue_id"].to_numpy(str),
        grid_ids=grids_df["grid_id"].to_numpy(str),
        bundles=bundle_df,
        catchment_id="synthetic_hard",
    )
    data.validate()
    return data

