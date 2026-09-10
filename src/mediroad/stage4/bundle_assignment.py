from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _ortools_available() -> bool:
    try:
        import ortools  # noqa: F401
        return True
    except ImportError:
        return False


def assign_bundles(
    selected: pd.DataFrame,
    bundle_values: pd.DataFrame,
    config: dict[str, Any],
    *,
    solver_name: str = "cp-sat",
) -> pd.DataFrame:
    """Assign exactly one frozen Stage 2A service bundle to each selected visit.

    Venue/catchment selection is already fixed. The assignment does not feed back into
    spatial selection, preserving the WHAT -> WHERE/WHO hierarchy observed in Stage 3.
    """
    if selected.empty:
        return pd.DataFrame()
    cfg = config["bundle_assignment"]
    work = bundle_values[bundle_values["venue_id"].astype(str).isin(selected["venue_id"].astype(str))].copy()
    expected = selected["venue_id"].astype(str).nunique()
    bundle_ids = sorted(work["bundle_id"].astype(str).unique())
    if not bundle_ids:
        raise ValueError("No Stage 2A bundle values are available for selected venues")
    counts = work.groupby("venue_id")["bundle_id"].nunique()
    if len(counts) != expected or counts.min() != len(bundle_ids):
        raise ValueError(
            f"Incomplete venue × bundle table: selected venues={expected}, rows per venue min={counts.min() if len(counts) else 0}, "
            f"bundle universe={len(bundle_ids)}"
        )

    work["bundle_value"] = pd.to_numeric(work["bundle_value"], errors="coerce").fillna(0.0)
    # Normalize within each venue so the assignment reflects relative WHAT, not raw venue population volume.
    work["bundle_value_within_venue_pct"] = work.groupby("venue_id")["bundle_value"].rank(
        method="average", pct=True
    )

    if solver_name != "cp-sat" or not _ortools_available():
        chosen = (
            work.sort_values(
                ["venue_id", "bundle_value_within_venue_pct", "bundle_value", "bundle_id"],
                ascending=[True, False, False, True],
            )
            .groupby("venue_id", as_index=False)
            .head(1)
        )
        chosen["bundle_assignment_status"] = "INDEPENDENT_TOP1_DIAGNOSTIC"
        return selected.merge(chosen, on="venue_id", how="left", validate="one_to_one")

    from ortools.sat.python import cp_model

    venues = selected["venue_id"].astype(str).tolist()
    venue_index = {venue: idx for idx, venue in enumerate(venues)}
    bundle_index = {bundle: idx for idx, bundle in enumerate(bundle_ids)}
    value_map = {
        (str(row.venue_id), str(row.bundle_id)): int(round(float(row.bundle_value_within_venue_pct) * 10000))
        for row in work.itertuples(index=False)
    }

    model = cp_model.CpModel()
    q: dict[tuple[int, int], Any] = {}
    for venue, vi in venue_index.items():
        for bundle, bi in bundle_index.items():
            q[(vi, bi)] = model.NewBoolVar(f"q_{vi}_{bi}")
        model.Add(sum(q[(vi, bi)] for bi in bundle_index.values()) == 1)

    min_per = cfg.get("min_per_bundle", {})
    max_per = cfg.get("max_per_bundle", {})
    default_min = int(cfg.get("default_min", 0))
    default_max = int(cfg.get("default_max", len(venues)))
    for bundle, bi in bundle_index.items():
        count = sum(q[(vi, bi)] for vi in venue_index.values())
        minimum = int(min_per.get(bundle, default_min))
        maximum = int(max_per.get(bundle, default_max))
        model.Add(count >= minimum)
        model.Add(count <= maximum)

    model.Maximize(
        sum(
            value_map[(venue, bundle)] * q[(venue_index[venue], bundle_index[bundle])]
            for venue in venues
            for bundle in bundle_ids
        )
    )
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = min(60.0, float(config["runtime"]["max_time_per_stage_sec"]))
    solver.parameters.num_search_workers = int(config["runtime"]["jobs"])
    solver.parameters.random_seed = int(config["seed"])
    status_code = solver.Solve(model)
    if status_code not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(f"Bundle assignment CP-SAT failed: {solver.StatusName(status_code)}")

    rows: list[dict[str, Any]] = []
    for venue, vi in venue_index.items():
        assigned = next(bundle for bundle, bi in bundle_index.items() if solver.Value(q[(vi, bi)]) == 1)
        record = work[(work["venue_id"].astype(str) == venue) & (work["bundle_id"].astype(str) == assigned)].iloc[0]
        rows.append(
            {
                "venue_id": venue,
                "bundle_id": assigned,
                "bundle_value": float(record["bundle_value"]),
                "bundle_value_within_venue_pct": float(record["bundle_value_within_venue_pct"]),
                "bundle_assignment_status": solver.StatusName(status_code),
            }
        )
    assignment = pd.DataFrame(rows)
    return selected.merge(assignment, on="venue_id", how="left", validate="one_to_one")

