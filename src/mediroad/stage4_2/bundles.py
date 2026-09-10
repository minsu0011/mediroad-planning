from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy import sparse

from .errors import ContractError, SolverCertificationError


def assign_bundles(
    selected: pd.DataFrame,
    bundle_values: pd.DataFrame,
    *,
    minimum_per_bundle: int = 1,
    maximum_per_bundle: int | None = None,
    allowed_bundle_ids: set[str] | None = None,
    time_limit_sec: float = 60.0,
) -> tuple[pd.DataFrame, dict[str, object]]:
    selected_ids = selected["venue_id"].astype(str).tolist()
    values = bundle_values.loc[bundle_values["venue_id"].astype(str).isin(selected_ids)].copy()
    values["bundle_id"] = values["bundle_id"].astype(str)
    if allowed_bundle_ids is not None:
        allowed = {str(value) for value in allowed_bundle_ids}
        values = values.loc[values["bundle_id"].isin(allowed)].copy()
    bundles = sorted(values["bundle_id"].astype(str).unique())
    if not bundles:
        raise ContractError("No bundle values for selected venues")
    full = pd.MultiIndex.from_product([selected_ids, bundles], names=["venue_id", "bundle_id"]).to_frame(index=False)
    full = full.merge(values, on=["venue_id", "bundle_id"], how="left")
    full["bundle_value"] = pd.to_numeric(full["bundle_value"], errors="coerce").fillna(0.0)
    # Within-venue percentile prevents raw population scale from redefining spatial selection.
    full["bundle_value_percentile"] = full.groupby("venue_id")["bundle_value"].rank(method="average", pct=True)
    n = len(full)
    c = -full["bundle_value_percentile"].to_numpy(float)
    integrality = np.ones(n, dtype=np.int8)
    bounds = Bounds(np.zeros(n), np.ones(n))
    rows, cols, vals, lb, ub = [], [], [], [], []
    r = 0
    for venue_id, idx in full.groupby("venue_id").groups.items():
        for i in idx:
            rows.append(r); cols.append(int(i)); vals.append(1.0)
        lb.append(1.0); ub.append(1.0); r += 1
    max_bundle = maximum_per_bundle or len(selected_ids)
    for bundle_id, idx in full.groupby("bundle_id").groups.items():
        for i in idx:
            rows.append(r); cols.append(int(i)); vals.append(1.0)
        lb.append(float(minimum_per_bundle)); ub.append(float(max_bundle)); r += 1
    A = sparse.coo_matrix((vals, (rows, cols)), shape=(r, n)).tocsr()
    result = milp(
        c,
        integrality=integrality,
        bounds=bounds,
        constraints=LinearConstraint(A, np.asarray(lb), np.asarray(ub)),
        options={"time_limit": float(time_limit_sec), "mip_rel_gap": 0.0, "presolve": True, "disp": False},
    )
    if result.x is None:
        raise SolverCertificationError(f"Bundle assignment failed: {result.message}")
    chosen = full.loc[np.asarray(result.x) >= 0.5].copy()
    if len(chosen) != len(selected_ids) or chosen["venue_id"].nunique() != len(selected_ids):
        raise SolverCertificationError("Bundle assignment did not assign exactly one bundle per venue")
    out = selected.merge(
        chosen[["venue_id", "bundle_id", "bundle_value", "bundle_value_percentile"]],
        on="venue_id",
        how="left",
        validate="one_to_one",
    )
    info = {
        "solver_status": "OPTIMAL" if int(result.status) == 0 else "FEASIBLE",
        "objective_value": float(-result.fun),
        "minimum_per_bundle": int(minimum_per_bundle),
        "bundle_counts": out["bundle_id"].value_counts().sort_index().to_dict(),
        "allowed_bundle_ids": bundles,
    }
    return out, info
