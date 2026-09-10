"""Fail-closed field-validation contract for Stage 4.1.

The Stage 4 optimizer selects spatial anchors.  This module deliberately does
not turn those anchors into operational venues.  It defines the evidence that
must be collected in the field and builds a bounded, deterministic priority
queue without converting missing stability evidence into a positive signal.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date
from typing import Any

import numpy as np
import pandas as pd


VALID_FIELD_STATES = frozenset({"YES", "NO", "UNKNOWN"})

# ``verification_date`` is part of the 12-field contract, but it is an ISO
# calendar date (or the explicit abstention value UNKNOWN), not a yes/no flag.
FIELD_VALIDATION_STATE_FIELDS = (
    "large_vehicle_access",
    "parking",
    "electricity",
    "toilet",
    "indoor_waiting",
    "heating_cooling",
    "barrier_free",
    "medical_equipment_space",
    "actual_facility_exists",
    "reservation_possible",
    "contact_verified",
)
FIELD_VALIDATION_FIELDS = (*FIELD_VALIDATION_STATE_FIELDS, "verification_date")

QUEUE_WEIGHTS = {
    "policy_selection_frequency": 0.30,
    "seed_stability": 0.15,
    "catchment_stability": 0.15,
    "cluster_stability": 0.10,
    "need_relevance": 0.10,
    "unique_coverage_contribution": 0.10,
    "busproxy_risk": 0.05,
    "fallback_risk": 0.05,
}


class FieldValidationContractError(ValueError):
    """Raised when field evidence could permit an unsupported promotion."""


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise FieldValidationContractError(f"{label} is missing required columns: {missing}")


def _validated_ids(frame: pd.DataFrame, label: str, key: str = "venue_id") -> pd.Series:
    _require_columns(frame, [key], label)
    raw = frame[key]
    if raw.isna().any():
        raise FieldValidationContractError(f"{label}.{key} contains null values")
    ids = raw.astype("string").str.strip()
    if ids.eq("").any():
        raise FieldValidationContractError(f"{label}.{key} contains blank values")
    duplicates = ids[ids.duplicated(keep=False)].unique().tolist()
    if duplicates:
        raise FieldValidationContractError(
            f"{label}.{key} must be unique; duplicate examples: {duplicates[:5]}"
        )
    return ids


def _strict_bool(series: pd.Series, label: str) -> pd.Series:
    def parse(value: Any) -> bool:
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, str) and value.strip().upper() in {"TRUE", "FALSE"}:
            return value.strip().upper() == "TRUE"
        raise FieldValidationContractError(f"{label} must contain only explicit booleans")

    return series.map(parse).astype(bool)


def _normalise_date(value: Any, *, as_of_date: date | None) -> tuple[str, bool]:
    if pd.isna(value) or str(value).strip().upper() in {"", "UNKNOWN"}:
        return "UNKNOWN", False
    text = str(value).strip()
    try:
        parsed = date.fromisoformat(text)
    except ValueError as exc:
        raise FieldValidationContractError(
            f"verification_date must be ISO YYYY-MM-DD or UNKNOWN; got {text!r}"
        ) from exc
    if parsed.isoformat() != text:
        raise FieldValidationContractError(
            f"verification_date must use canonical ISO YYYY-MM-DD; got {text!r}"
        )
    if as_of_date is not None and parsed > as_of_date:
        raise FieldValidationContractError(
            f"verification_date {text} is later than the audit date {as_of_date.isoformat()}"
        )
    return text, True


def validate_field_validation_schema(
    evidence: pd.DataFrame,
    *,
    as_of_date: date | None = None,
) -> pd.DataFrame:
    """Validate and normalise the 12 field checks.

    ``field_verified`` is derived conservatively: all eleven operational
    checks must be ``YES`` and a real verification date must be present.  If a
    caller supplies a ``field_verified`` column, it must exactly match that
    derivation.  Thus blanks, Python truthiness, and a partially completed form
    can never promote a physical venue.
    """

    _require_columns(evidence, ["venue_id", *FIELD_VALIDATION_FIELDS], "field evidence")
    out = evidence.copy()
    out["venue_id"] = _validated_ids(out, "field evidence")

    for column in FIELD_VALIDATION_STATE_FIELDS:
        values = out[column].astype("string").fillna("UNKNOWN").str.strip().str.upper()
        invalid = sorted(set(values.dropna().tolist()) - VALID_FIELD_STATES)
        if invalid:
            raise FieldValidationContractError(
                f"{column} contains values outside YES/NO/UNKNOWN: {invalid[:5]}"
            )
        out[column] = values

    normalised_dates: list[str] = []
    has_date: list[bool] = []
    for value in out["verification_date"].tolist():
        normalised, present = _normalise_date(value, as_of_date=as_of_date)
        normalised_dates.append(normalised)
        has_date.append(present)
    out["verification_date"] = pd.Series(normalised_dates, index=out.index, dtype="string")

    resolved = out[list(FIELD_VALIDATION_STATE_FIELDS)].ne("UNKNOWN").all(axis=1)
    all_positive = out[list(FIELD_VALIDATION_STATE_FIELDS)].eq("YES").all(axis=1)
    out["field_validation_complete"] = resolved & pd.Series(has_date, index=out.index)
    derived_verified = out["field_validation_complete"] & all_positive

    if "field_verified" in evidence.columns:
        supplied = _strict_bool(evidence["field_verified"], "field_verified")
        mismatch = supplied.ne(derived_verified)
        if mismatch.any():
            bad_ids = out.loc[mismatch, "venue_id"].head(5).tolist()
            raise FieldValidationContractError(
                "field_verified disagrees with the 12-field evidence for venue(s): "
                f"{bad_ids}"
            )
    out["field_verified"] = derived_verified.astype(bool)
    out["field_operationally_feasible"] = derived_verified.astype(bool)
    return out


def make_field_validation_form(venues: pd.DataFrame) -> pd.DataFrame:
    """Create an explicit-UNKNOWN field form for a unique venue list."""

    ids = _validated_ids(venues, "venues")
    identity = [
        column
        for column in ("venue_id", "venue_name", "venue_type", "cluster_id", "admin_code", "sigungu")
        if column in venues.columns
    ]
    out = venues[identity].copy()
    out["venue_id"] = ids
    for column in FIELD_VALIDATION_STATE_FIELDS:
        out[column] = "UNKNOWN"
    out["verification_date"] = "UNKNOWN"
    out["field_validation_complete"] = False
    out["field_verified"] = False
    out["field_operationally_feasible"] = False
    return out


def robust_core_summary(
    selection_evidence: pd.DataFrame,
    *,
    threshold: float = 0.67,
) -> dict[str, Any]:
    """Return an explicit PRESENT/NONE result for the robust-core rule."""

    if not 0 < float(threshold) <= 1:
        raise FieldValidationContractError("robust-core threshold must be in (0, 1]")
    ids = _validated_ids(selection_evidence, "selection evidence")
    _require_columns(selection_evidence, ["selection_frequency"], "selection evidence")
    frequency = pd.to_numeric(selection_evidence["selection_frequency"], errors="coerce")
    if frequency.isna().any() or (~np.isfinite(frequency)).any() or frequency.lt(0).any() or frequency.gt(1).any():
        raise FieldValidationContractError("selection_frequency must be finite and within [0, 1]")
    selected_ids = ids[frequency.ge(float(threshold))].tolist()
    return {
        "robust_core_threshold": float(threshold),
        "robust_core_count": len(selected_ids),
        "robust_core_status": "PRESENT" if selected_ids else "NONE",
        "robust_core_venue_ids": selected_ids,
        "maximum_selection_frequency": float(frequency.max()) if len(frequency) else 0.0,
    }


def _rank_fraction(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if values.notna().sum() == 0:
        return pd.Series(0.0, index=series.index, dtype=float)
    filled = values.fillna(values.min())
    if filled.nunique(dropna=True) <= 1:
        return pd.Series(0.5, index=series.index, dtype=float)
    return filled.rank(method="average", pct=True).astype(float)


def _metric_table(
    frame: pd.DataFrame | None,
    *,
    key: str,
    aliases: tuple[str, ...],
    output_name: str,
    label: str,
) -> tuple[pd.DataFrame | None, bool]:
    if frame is None:
        return None, False
    ids = _validated_ids(frame, label, key=key)
    source = next((column for column in aliases if column in frame.columns), None)
    if source is None:
        raise FieldValidationContractError(
            f"{label} needs one metric column from {list(aliases)}"
        )
    values = pd.to_numeric(frame[source], errors="coerce")
    if values.isna().any() or (~np.isfinite(values)).any() or values.lt(0).any() or values.gt(1).any():
        raise FieldValidationContractError(f"{label}.{source} must be finite and within [0, 1]")
    result = pd.DataFrame({key: ids, output_name: values.astype(float)})
    for metadata_column in (
        "evidence_scope",
        "eligible_candidate_set",
        "evidence_universe_complete",
        "final_promotion_allowed",
        "solver_certified_for_queue",
    ):
        if metadata_column in frame.columns:
            metadata = frame[metadata_column]
            if metadata_column in {
                "evidence_universe_complete",
                "final_promotion_allowed",
                "solver_certified_for_queue",
            }:
                metadata = _strict_bool(metadata, f"{label}.{metadata_column}")
            result[f"{output_name}_{metadata_column}"] = metadata.to_numpy()
    return result, True


def _is_busproxy(series: pd.Series) -> pd.Series:
    return series.astype("string").str.upper().str.startswith("BUSPROXY_")


def build_field_validation_queue(
    selection_evidence: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    primary_plan: pd.DataFrame | None = None,
    seed_stability: pd.DataFrame | None = None,
    catchment_stability: pd.DataFrame | None = None,
    cluster_stability: pd.DataFrame | None = None,
    min_size: int = 30,
    max_size: int = 60,
    robust_threshold: float = 0.67,
) -> pd.DataFrame:
    """Build the 30--60 row Stage 4.1 field-validation priority queue.

    Missing optional stability studies remain explicit ``NOT_MEASURED`` values
    scored as zero.  They are never inferred from a different denominator.
    Primary anchors, their one or two real-facility fallbacks, and every robust
    core venue are mandatory.  The function raises if those requirements cannot
    fit inside the declared maximum.
    """

    if not (30 <= int(min_size) <= int(max_size) <= 60):
        raise FieldValidationContractError("queue bounds must satisfy 30 <= min_size <= max_size <= 60")
    candidate_ids = _validated_ids(candidates, "candidates")
    _require_columns(candidates, ["cluster_id", "structural_need"], "candidates")
    unique_column = next(
        (
            column
            for column in ("unique_coverage_contribution", "raw_elderly_exposure")
            if column in candidates.columns
        ),
        None,
    )
    if unique_column is None:
        raise FieldValidationContractError(
            "candidates needs unique_coverage_contribution or raw_elderly_exposure"
        )
    if len(candidates) < int(min_size):
        raise FieldValidationContractError(
            f"candidate universe has {len(candidates)} rows and cannot fill a {min_size}-row queue"
        )

    work = candidates.copy()
    work["venue_id"] = candidate_ids
    work["cluster_id"] = work["cluster_id"].astype("string").str.strip()
    if work["cluster_id"].isna().any() or work["cluster_id"].eq("").any():
        raise FieldValidationContractError("candidates.cluster_id must be complete")

    selection_ids = _validated_ids(selection_evidence, "selection evidence")
    _require_columns(selection_evidence, ["selection_frequency"], "selection evidence")
    frequency = pd.to_numeric(selection_evidence["selection_frequency"], errors="coerce")
    if frequency.isna().any() or (~np.isfinite(frequency)).any() or frequency.lt(0).any() or frequency.gt(1).any():
        raise FieldValidationContractError("selection_frequency must be finite and within [0, 1]")
    unknown_selection_ids = sorted(set(selection_ids) - set(candidate_ids))
    if unknown_selection_ids:
        raise FieldValidationContractError(
            f"selection evidence references venues outside candidates: {unknown_selection_ids[:5]}"
        )
    selection = pd.DataFrame(
        {"venue_id": selection_ids, "policy_selection_frequency": frequency.astype(float)}
    )
    work = work.merge(selection, on="venue_id", how="left", validate="one_to_one")
    work["policy_selection_frequency"] = work["policy_selection_frequency"].fillna(0.0)

    metric_specs = [
        (
            seed_stability,
            "venue_id",
            ("seed_stability", "seed_selection_frequency"),
            "seed_stability",
            "seed stability",
        ),
        (
            catchment_stability,
            "venue_id",
            ("catchment_stability", "catchment_selection_frequency"),
            "catchment_stability",
            "catchment stability",
        ),
        (
            cluster_stability,
            "cluster_id",
            ("cluster_stability", "cluster_selection_frequency"),
            "cluster_stability",
            "cluster stability",
        ),
    ]
    evidence_availability: dict[str, bool] = {}
    for table, key, aliases, output_name, label in metric_specs:
        metric, available = _metric_table(
            table, key=key, aliases=aliases, output_name=output_name, label=label
        )
        evidence_availability[output_name] = available
        measured_mask = pd.Series(False, index=work.index)
        if metric is not None:
            unknown_keys = sorted(set(metric[key]) - set(work[key].astype("string")))
            if unknown_keys:
                raise FieldValidationContractError(
                    f"{label} references keys outside candidates: {unknown_keys[:5]}"
                )
            work = work.merge(metric, on=key, how="left", validate="many_to_one")
            measured_mask = work[output_name].notna()
        if metric is None:
            work[output_name] = 0.0
        elif output_name not in work.columns:
            work[output_name] = 0.0
        work[output_name] = work[output_name].fillna(0.0).astype(float)
        complete_column = f"{output_name}_evidence_universe_complete"
        eligible_column = f"{output_name}_eligible_candidate_set"
        complete_universe = bool(
            complete_column in work
            and work.loc[measured_mask, complete_column].fillna(False).astype(bool).all()
            and measured_mask.any()
        )
        eligible_label = (
            str(work.loc[measured_mask, eligible_column].dropna().iloc[0])
            if eligible_column in work
            and not work.loc[measured_mask, eligible_column].dropna().empty
            else ""
        )
        unmatched_status = (
            f"NOT_ELIGIBLE_{eligible_label.upper()}"
            if complete_universe and eligible_label
            else "NOT_MEASURED"
        )
        work[f"{output_name}_evidence_status"] = np.where(
            measured_mask, "MEASURED", unmatched_status
        )
        scope_column = f"{output_name}_evidence_scope"
        if scope_column not in work:
            work[scope_column] = "NOT_MEASURED"
        else:
            measured_scopes = work.loc[measured_mask, scope_column].dropna()
            default_scope = (
                str(measured_scopes.iloc[0]) if not measured_scopes.empty else "NOT_MEASURED"
            )
            work[scope_column] = work[scope_column].fillna(default_scope)
        promotion_column = f"{output_name}_final_promotion_allowed"
        if promotion_column not in work:
            work[promotion_column] = False
        else:
            work[promotion_column] = work[promotion_column].fillna(False).astype(bool)

    work["need_relevance"] = _rank_fraction(work["structural_need"])
    work["unique_coverage_contribution"] = _rank_fraction(work[unique_column])
    work["is_busproxy_anchor"] = _is_busproxy(work["venue_id"])
    work["busproxy_risk"] = work["is_busproxy_anchor"].astype(float)

    physical = ~work["is_busproxy_anchor"]
    physical_count = work.assign(_physical=physical.astype(int)).groupby("cluster_id")["_physical"].sum()
    work["actual_facility_fallback_count"] = work["cluster_id"].map(physical_count).astype(int)
    work.loc[physical, "actual_facility_fallback_count"] -= 1
    work["actual_facility_fallback_count"] = work["actual_facility_fallback_count"].clip(lower=0)
    work["fallback_availability"] = (
        work["actual_facility_fallback_count"].clip(upper=2).astype(float) / 2.0
    )
    work["fallback_risk"] = 1.0 - work["fallback_availability"]

    work["primary_plan_selected"] = False
    mandatory_ids: set[str] = set()
    fallback_ids: set[str] = set()
    if primary_plan is not None:
        plan_ids = _validated_ids(primary_plan, "primary plan")
        unknown_plan = sorted(set(plan_ids) - set(candidate_ids))
        if unknown_plan:
            raise FieldValidationContractError(
                f"primary plan references venues outside candidates: {unknown_plan[:5]}"
            )
        mandatory_ids.update(plan_ids.tolist())
        work.loc[work["venue_id"].isin(mandatory_ids), "primary_plan_selected"] = True

        indexed = work.set_index("venue_id", drop=False)
        for plan_id in plan_ids:
            anchor = indexed.loc[plan_id]
            cluster_id = str(anchor["cluster_id"])
            preferred: list[str] = []
            plan_row = primary_plan.loc[plan_ids.eq(plan_id)].iloc[0]
            for column in ("fallback_venue_1", "fallback_venue_2"):
                for source_row in (plan_row, anchor):
                    if column not in source_row.index or pd.isna(source_row[column]):
                        continue
                    fallback_id = str(source_row[column]).strip()
                    if not fallback_id or fallback_id.upper().startswith("BUSPROXY_"):
                        continue
                    if fallback_id not in indexed.index:
                        raise FieldValidationContractError(
                            f"fallback {fallback_id!r} is absent from the supplied candidate universe"
                        )
                    if str(indexed.loc[fallback_id, "cluster_id"]) != cluster_id:
                        # Stage 3 fallbacks were allowed to be nearby/high-Jaccard
                        # alternatives.  Stage 4.1 is stricter: reject that hint
                        # and safely refill from the exact coverage cluster.
                        continue
                    if fallback_id != plan_id and fallback_id not in preferred:
                        preferred.append(fallback_id)

            cluster_pool = work[
                work["cluster_id"].eq(cluster_id)
                & ~work["is_busproxy_anchor"]
                & work["venue_id"].ne(plan_id)
            ].copy()
            # Keep the queue and busproxy_resolution.actual_facility_fallbacks
            # on one deterministic fallback contract.  Readiness is an
            # operational tie-breaker only; every candidate still requires the
            # complete field form before it can become a physical venue.
            fallback_sort_columns: list[str] = []
            for column in (
                "venue_readiness_prior",
                "raw_elderly_exposure",
                "need_weighted_exposure",
            ):
                sort_column = f"_fallback_sort__{column}"
                if column in cluster_pool.columns:
                    cluster_pool[sort_column] = pd.to_numeric(
                        cluster_pool[column], errors="coerce"
                    ).fillna(float("-inf"))
                else:
                    cluster_pool[sort_column] = float("-inf")
                fallback_sort_columns.append(sort_column)
            cluster_pool = cluster_pool.sort_values(
                [*fallback_sort_columns, "venue_id"],
                ascending=[False, False, False, True],
                kind="stable",
            )
            for fallback_id in cluster_pool["venue_id"].tolist():
                if fallback_id not in preferred:
                    preferred.append(fallback_id)
            fallback_ids.update(preferred[:2])
        mandatory_ids.update(fallback_ids)

    core = robust_core_summary(selection_evidence, threshold=robust_threshold)
    mandatory_ids.update(core["robust_core_venue_ids"])
    if len(mandatory_ids) > int(max_size):
        raise FieldValidationContractError(
            f"{len(mandatory_ids)} mandatory anchors/fallbacks/robust venues exceed max_size={max_size}"
        )

    work["is_primary_fallback"] = work["venue_id"].isin(fallback_ids)
    work["is_robust_core"] = work["policy_selection_frequency"].ge(float(robust_threshold))
    work["queue_mandatory"] = work["venue_id"].isin(mandatory_ids)
    work["field_validation_priority_score"] = sum(
        float(weight) * work[component]
        for component, weight in QUEUE_WEIGHTS.items()
    )

    work = work.sort_values(
        [
            "queue_mandatory",
            "primary_plan_selected",
            "field_validation_priority_score",
            "policy_selection_frequency",
            "seed_stability",
            "catchment_stability",
            "cluster_stability",
            "venue_id",
        ],
        ascending=[False, False, False, False, False, False, False, True],
        kind="stable",
    )
    target_size = max(int(min_size), len(mandatory_ids))
    queue = work.head(target_size).copy().reset_index(drop=True)
    missing_mandatory = sorted(mandatory_ids - set(queue["venue_id"]))
    if missing_mandatory:
        raise FieldValidationContractError(
            f"queue truncation dropped mandatory venues: {missing_mandatory[:5]}"
        )

    def role(row: pd.Series) -> str:
        roles: list[str] = []
        if bool(row["primary_plan_selected"]):
            roles.append("PRIMARY_ANCHOR")
        if bool(row["is_primary_fallback"]):
            roles.append("PRIMARY_FALLBACK")
        if bool(row["is_robust_core"]):
            roles.append("ROBUST_CORE")
        return "+".join(roles) if roles else "PRIORITY_EVIDENCE"

    queue.insert(0, "field_validation_rank", np.arange(1, len(queue) + 1, dtype=int))
    queue["queue_role"] = queue.apply(role, axis=1)
    queue["robust_core_global_status"] = core["robust_core_status"]
    queue["robust_core_threshold"] = float(robust_threshold)
    queue["operational_status"] = "UNVERIFIED"
    for column in FIELD_VALIDATION_STATE_FIELDS:
        queue[column] = "UNKNOWN"
    queue["verification_date"] = "UNKNOWN"
    queue["field_validation_complete"] = False
    queue["field_verified"] = False
    queue["field_operationally_feasible"] = False

    if not (int(min_size) <= len(queue) <= int(max_size)):
        raise FieldValidationContractError("field-validation queue is outside its 30--60 row contract")
    if queue["venue_id"].duplicated().any():
        raise FieldValidationContractError("field-validation queue contains duplicate venue_id values")
    return queue


__all__ = [
    "FIELD_VALIDATION_FIELDS",
    "FIELD_VALIDATION_STATE_FIELDS",
    "FieldValidationContractError",
    "QUEUE_WEIGHTS",
    "VALID_FIELD_STATES",
    "build_field_validation_queue",
    "make_field_validation_form",
    "robust_core_summary",
    "validate_field_validation_schema",
]
