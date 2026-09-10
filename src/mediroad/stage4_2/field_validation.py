from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .contracts import (
    EVIDENCE_COLUMNS,
    FIELD_DATE_COLUMN,
    FIELD_DERIVED_COLUMNS,
    FIELD_KEY_COLUMNS,
    FIELD_REQUIRED_INPUT_COLUMNS,
    FIELD_STATUS_COLUMNS,
)
from .errors import ContractError
from .types import Stage41Artifacts
from .utils import ensure_columns, normalize_bool, normalize_yes_no_unknown, parse_date, read_table


@dataclass
class FieldValidationAudit:
    normalized: pd.DataFrame
    summary: dict[str, Any]
    issues: pd.DataFrame


def prepare_priority_queue(stage41: Stage41Artifacts) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    queue = pd.read_csv(stage41.field_queue, low_memory=False)
    form = pd.read_csv(stage41.field_form, low_memory=False)
    efficiency = pd.read_csv(stage41.plan_efficiency, low_memory=False)
    balanced = pd.read_csv(stage41.plan_balanced, low_memory=False)
    equity = pd.read_csv(stage41.plan_equity, low_memory=False)
    for frame in [queue, form, efficiency, balanced, equity]:
        frame["venue_id"] = frame["venue_id"].astype(str)
    eff = set(efficiency["venue_id"])
    bal = set(balanced["venue_id"])
    eq = set(equity["venue_id"])
    certified_intersection = eff & bal

    q = queue.copy()
    q["selected_efficiency"] = q["venue_id"].isin(eff)
    q["selected_balanced"] = q["venue_id"].isin(bal)
    q["selected_equity_diagnostic"] = q["venue_id"].isin(eq)
    q["certified_efficiency_balanced_intersection"] = q["venue_id"].isin(certified_intersection)
    bus_raw = q.get("is_busproxy_anchor", q["venue_id"].str.startswith("BUSPROXY_"))
    bus = pd.Series(bus_raw, index=q.index).map(normalize_bool).fillna(False).astype(bool)
    fallback_count = pd.to_numeric(q.get("actual_facility_fallback_count", 0), errors="coerce").fillna(0).astype(int)
    q["priority_reason"] = "FIELD_VALIDATION_FALLBACK_OR_OTHER"
    q["priority_tier"] = "D"
    q.loc[q["selected_efficiency"] | q["selected_balanced"], ["priority_tier", "priority_reason"]] = [
        "B",
        "CERTIFIED_POLICY_ANCHOR_OR_DIRECT_FALLBACK",
    ]
    q.loc[q["certified_efficiency_balanced_intersection"], ["priority_tier", "priority_reason"]] = [
        "A",
        "CERTIFIED_EFFICIENCY_BALANCED_INTERSECTION",
    ]
    q.loc[bus & fallback_count.eq(0), ["priority_tier", "priority_reason"]] = [
        "P0",
        "BUSPROXY_ANCHOR_WITHOUT_ACTUAL_FACILITY_FALLBACK",
    ]
    order = {"P0": 0, "A": 1, "B": 2, "C": 3, "D": 4}
    q["_tier_order"] = q["priority_tier"].map(order).fillna(9)
    score_col = "field_validation_priority_score" if "field_validation_priority_score" in q else None
    sort_cols = ["_tier_order"] + ([score_col] if score_col else []) + ["venue_id"]
    ascending = [True] + ([False] if score_col else []) + [True]
    q = q.sort_values(sort_cols, ascending=ascending).drop(columns=["_tier_order"]).reset_index(drop=True)
    q.insert(0, "stage4_2_field_rank", np.arange(1, len(q) + 1))

    # Preserve the exact official field contract, adding no derived evidence claims.
    required_form_cols = FIELD_KEY_COLUMNS + FIELD_REQUIRED_INPUT_COLUMNS + FIELD_DERIVED_COLUMNS
    ensure_columns(form, required_form_cols, label="Stage 4.1 field-validation form")
    updated_form = q[[c for c in ["venue_id", "priority_tier", "priority_reason", "stage4_2_field_rank"] if c in q]].merge(
        form[required_form_cols], on="venue_id", how="left", validate="one_to_one"
    )
    evidence = pd.DataFrame({"venue_id": updated_form["venue_id"].astype(str)})
    for col in EVIDENCE_COLUMNS[1:]:
        evidence[col] = ""
    return q, updated_form, evidence


def audit_field_validation(frame: pd.DataFrame) -> FieldValidationAudit:
    ensure_columns(frame, FIELD_KEY_COLUMNS + FIELD_REQUIRED_INPUT_COLUMNS, label="field validation")
    work = frame.copy()
    work["venue_id"] = work["venue_id"].astype(str)
    if work["venue_id"].duplicated().any():
        dup = work.loc[work["venue_id"].duplicated(False), "venue_id"].unique().tolist()
        raise ContractError(f"Duplicate field validation venue IDs: {dup[:10]}")

    issues: list[dict[str, Any]] = []
    for col in FIELD_STATUS_COLUMNS:
        work[col] = work[col].map(normalize_yes_no_unknown)
        invalid = ~work[col].isin(["YES", "NO", "UNKNOWN"])
        for venue_id, value in work.loc[invalid, ["venue_id", col]].itertuples(index=False):
            issues.append({"venue_id": venue_id, "field": col, "issue": "INVALID_STATUS", "value": value})
    # When every value is UNKNOWN, ``Series.map(parse_date)`` otherwise has
    # object dtype (all Python ``None``), so the later ``.dt`` accessor raises
    # instead of returning a fail-closed audit.  Coerce explicitly to a
    # datetimelike Series; unknown dates remain NaT and can never verify a row.
    work["_verification_ts"] = pd.to_datetime(
        work[FIELD_DATE_COLUMN].map(parse_date), errors="coerce"
    )
    invalid_date = work[FIELD_DATE_COLUMN].notna() & work[FIELD_DATE_COLUMN].astype(str).str.strip().ne("") & work["_verification_ts"].isna()
    for venue_id, value in work.loc[invalid_date, ["venue_id", FIELD_DATE_COLUMN]].itertuples(index=False):
        issues.append({"venue_id": venue_id, "field": FIELD_DATE_COLUMN, "issue": "INVALID_DATE", "value": value})

    all_known = work[FIELD_STATUS_COLUMNS].ne("UNKNOWN").all(axis=1)
    all_yes = work[FIELD_STATUS_COLUMNS].eq("YES").all(axis=1)
    has_date = work["_verification_ts"].notna()
    work["field_validation_complete"] = all_known & has_date
    work["field_verified"] = all_yes & has_date
    work["field_operationally_feasible"] = work["field_verified"]
    work[FIELD_DATE_COLUMN] = work["_verification_ts"].dt.strftime("%Y-%m-%d").fillna("UNKNOWN")
    work = work.drop(columns=["_verification_ts"])

    summary = {
        "rows": int(len(work)),
        "complete": int(work["field_validation_complete"].sum()),
        "verified": int(work["field_verified"].sum()),
        "operationally_feasible": int(work["field_operationally_feasible"].sum()),
        "unknown_rows": int((work[FIELD_STATUS_COLUMNS].eq("UNKNOWN").any(axis=1)).sum()),
        "invalid_issue_count": len(issues),
    }
    return FieldValidationAudit(normalized=work, summary=summary, issues=pd.DataFrame(issues))


def merge_field_results_with_queue(queue: pd.DataFrame, audit: FieldValidationAudit) -> pd.DataFrame:
    result_cols = ["venue_id", *FIELD_REQUIRED_INPUT_COLUMNS, *FIELD_DERIVED_COLUMNS]
    base = queue.drop(columns=[c for c in FIELD_REQUIRED_INPUT_COLUMNS + FIELD_DERIVED_COLUMNS if c in queue], errors="ignore")
    return base.merge(audit.normalized[result_cols], on="venue_id", how="left", validate="one_to_one")
