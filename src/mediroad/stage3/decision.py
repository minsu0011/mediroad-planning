"""Stage 3 decision, validation, and Stage 4 hand-off utilities.

The functions in this module deliberately do not create a single, synthetic
``venue_score``.  Potential beneficiary exposure, structural need, specialty
gap, operational context, redundancy, and field uncertainty remain separate
dimensions until Stage 4.  All helpers are deterministic and side-effect
free; the Stage 3 pipeline owns persistence and run-scoped provenance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


FIELD_VALIDATION_FIELDS: tuple[str, ...] = (
    "parking_verified",
    "power_verified",
    "indoor_waiting_verified",
    "toilet_verified",
    "vehicle_access_verified",
    "equipment_space_verified",
)

INTERFACE_REQUIRED_COLUMNS: tuple[str, ...] = (
    "venue_id",
    "venue_name",
    "venue_type",
    "admin_dong_code",
    "sigungu",
    "latitude",
    "longitude",
    "structural_need",
    "raw_elderly_exposure",
    "need_weighted_exposure",
    "high_need_elderly_exposure",
    "own_admin_exposure",
    "cross_admin_exposure",
    "cross_sigungu_exposure",
    "bundle_id",
    "specialty_gap",
    "bundle_gap_weighted_exposure",
    "venue_readiness_prior",
    "transit_context",
    "road_context",
    "known_outreach_overlap",
    "field_validation_required",
    "pareto_tier",
    "shortlist_rank",
    "coverage_cluster_id",
    "fallback_venue_1",
    "fallback_venue_2",
    "temporal_release_type",
    "temporal_confidence",
    "recommended_season",
    "fallback_season",
)

STAGE4_FORBIDDEN_COLUMNS: frozenset[str] = frozenset(
    {
        "selected_final",
        "visit_number",
        "assigned_month",
        "assigned_date",
        "final_team",
        "optimizer_objective",
    }
)

TEMPORAL_SELECTOR_FORBIDDEN_COLUMNS: frozenset[str] = frozenset(
    {
        "temporal_score",
        "season_score",
        "season_adjusted_exposure",
        "month_score",
        "date_score",
    }
)

ABSTENTION_RELEASE = "NO_STRONG_PREFERENCE"
ACTIONABLE_RELEASES: frozenset[str] = frozenset({"STRONG_SINGLE", "ROBUST_PAIR"})
VALID_SEASONS: frozenset[str] = frozenset({"spring", "summer", "autumn", "winter"})


@dataclass(frozen=True)
class AdminShortlistResult:
    """Complete candidates plus the three non-destructive shortlist views."""

    candidates: pd.DataFrame
    top3: pd.DataFrame
    top5: pd.DataFrame
    pareto_front: pd.DataFrame
    shortlist: pd.DataFrame


@dataclass(frozen=True)
class SpearmanAuditResult:
    """Descriptive venue-level and policy-unit-collapsed correlation audit."""

    all_venue_correlation: pd.DataFrame
    all_venue_counts: pd.DataFrame
    admin_collapsed_correlation: pd.DataFrame
    admin_collapsed_counts: pd.DataFrame
    admin_collapsed: pd.DataFrame
    comparison: pd.DataFrame


@dataclass(frozen=True)
class SensitivityResult:
    """Scenario-level and admin-level Stage 3 stability diagnostics."""

    summary: pd.DataFrame
    by_admin: pd.DataFrame


def _require_frame(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError(f"{label} must be a non-empty DataFrame")
    if not frame.columns.is_unique:
        duplicated = frame.columns[frame.columns.duplicated()].tolist()
        raise ValueError(f"{label} columns must be unique; duplicated={duplicated[:5]}")
    return frame.copy()


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"{label} missing columns: {missing}")


def _require_unique(frame: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    _require_columns(frame, columns, label)
    if frame[list(columns)].isna().any().any():
        raise ValueError(f"{label} keys cannot be missing: {list(columns)}")
    duplicated = frame.duplicated(list(columns), keep=False)
    if duplicated.any():
        examples = frame.loc[duplicated, list(columns)].head(5).to_dict("records")
        raise ValueError(f"{label} keys must be unique; examples={examples}")


def _finite_numeric(
    frame: pd.DataFrame,
    columns: Sequence[str],
    label: str,
    *,
    nonnegative: bool = False,
) -> pd.DataFrame:
    values = frame.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce")
    array = values.to_numpy(dtype=float)
    if not np.isfinite(array).all():
        raise ValueError(f"{label} must be finite numeric: {list(columns)}")
    if nonnegative and (array < 0).any():
        raise ValueError(f"{label} must be non-negative: {list(columns)}")
    return values


def _normalise_field_status(value: Any) -> str:
    if value is None or value is pd.NA:
        return "unknown"
    try:
        if bool(pd.isna(value)):
            return "unknown"
    except (TypeError, ValueError):
        pass
    if isinstance(value, (bool, np.bool_)):
        return "verified" if bool(value) else "not_verified"
    if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
        return "verified" if int(value) == 1 else "not_verified"
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if text in {"", "unknown", "na", "n/a", "none", "null", "미확인", "unchecked"}:
        return "unknown"
    if text in {"true", "yes", "y", "verified", "pass", "available", "확인"}:
        return "verified"
    if text in {"false", "no", "n", "not_verified", "fail", "unavailable", "불가"}:
        return "not_verified"
    # An unrecognised field note is not evidence that a condition was met.
    return "unknown"


def _coerce_bool(value: Any, *, unknown: bool = False) -> bool:
    if value is None or value is pd.NA:
        return unknown
    try:
        if bool(pd.isna(value)):
            return unknown
    except (TypeError, ValueError):
        pass
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer, float, np.floating)):
        if float(value) in (0.0, 1.0):
            return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "required"}:
        return True
    if text in {"false", "0", "no", "n", "complete"}:
        return False
    raise ValueError(f"cannot interpret boolean value: {value!r}")


def build_venue_feasibility_context(
    venues: pd.DataFrame,
    *,
    id_col: str = "venue_id",
    venue_type_col: str = "venue_type",
    verification_fields: Sequence[str] = FIELD_VALIDATION_FIELDS,
    readiness_prior_by_type: Mapping[str, float] | None = None,
    readiness_col: str = "venue_readiness_prior",
) -> pd.DataFrame:
    """Create honest venue context without inventing operational feasibility.

    Missing field-verification columns and values become the literal
    ``"unknown"``.  A supplied, config-owned facility-type prior may be
    attached, but no default weights are invented here.
    """

    frame = _require_frame(venues, "venues")
    _require_unique(frame, [id_col], "venues")
    if "venue_operationally_feasible" in frame.columns:
        raise ValueError(
            "venue_operationally_feasible is unsupported without field evidence; "
            "preserve verification fields instead"
        )

    fields = tuple(dict.fromkeys(str(field) for field in verification_fields))
    if not fields:
        raise ValueError("at least one verification field is required")
    for field in fields:
        if field not in frame.columns:
            frame[field] = "unknown"
        else:
            frame[field] = frame[field].map(_normalise_field_status)

    unknown_count = frame.loc[:, fields].eq("unknown").sum(axis=1).astype("int64")
    negative_count = frame.loc[:, fields].eq("not_verified").sum(axis=1).astype("int64")
    explicit_required = (
        frame["field_validation_required"].map(_coerce_bool)
        if "field_validation_required" in frame.columns
        else pd.Series(False, index=frame.index)
    )
    frame["field_validation_unknown_count"] = unknown_count
    frame["field_validation_negative_count"] = negative_count
    frame["field_validation_required"] = (
        explicit_required | unknown_count.gt(0) | negative_count.gt(0)
    ).astype(bool)
    frame["field_validation_status"] = np.select(
        [negative_count.gt(0), unknown_count.gt(0)],
        ["NEGATIVE_CONDITION_RECORDED", "UNKNOWN_REQUIRES_VALIDATION"],
        default="ALL_LISTED_FIELDS_VERIFIED",
    )

    if readiness_prior_by_type is not None:
        _require_columns(frame, [venue_type_col], "venues")
        mapping = {str(key): float(value) for key, value in readiness_prior_by_type.items()}
        if any(not np.isfinite(value) for value in mapping.values()):
            raise ValueError("readiness priors must be finite")
        mapped = frame[venue_type_col].astype(str).map(mapping)
        if readiness_col in frame.columns:
            existing = pd.to_numeric(frame[readiness_col], errors="coerce")
            conflict = existing.notna() & mapped.notna() & ~np.isclose(existing, mapped)
            if conflict.any():
                raise ValueError("existing readiness prior conflicts with supplied config")
            frame[readiness_col] = existing.fillna(mapped)
        else:
            frame[readiness_col] = mapped
        frame["venue_readiness_prior_source"] = np.where(
            frame[readiness_col].notna(), "CONFIG_VENUE_TYPE_PRIOR", "UNKNOWN"
        )
    elif readiness_col not in frame.columns:
        frame[readiness_col] = np.nan
        frame["venue_readiness_prior_source"] = "UNKNOWN"
    else:
        frame[readiness_col] = pd.to_numeric(frame[readiness_col], errors="coerce")
        frame["venue_readiness_prior_source"] = np.where(
            frame[readiness_col].notna(), "INPUT_CONTEXT", "UNKNOWN"
        )
    return frame


def _objective_directions(
    frame: pd.DataFrame,
    objectives: Mapping[str, str | int] | Sequence[str] | None,
) -> dict[str, int]:
    if objectives is None:
        defaults: tuple[tuple[str, int], ...] = (
            ("raw_elderly_exposure", 1),
            ("need_weighted_exposure", 1),
            ("bundle_gap_weighted_exposure", 1),
            ("venue_readiness_prior", 1),
            ("transit_accessibility", 1),
            ("known_outreach_overlap", -1),
            ("field_validation_unknown_count", -1),
        )
        directions = {
            column: direction
            for column, direction in defaults
            if column in frame.columns and pd.api.types.is_numeric_dtype(frame[column])
        }
    elif isinstance(objectives, Mapping):
        directions = {}
        for column, raw_direction in objectives.items():
            text = str(raw_direction).strip().lower()
            if raw_direction in (1, +1) or text in {"max", "maximize", "higher", "higher_is_better"}:
                directions[str(column)] = 1
            elif raw_direction in (-1,) or text in {"min", "minimize", "lower", "lower_is_better"}:
                directions[str(column)] = -1
            else:
                raise ValueError(f"unsupported objective direction for {column}: {raw_direction}")
    else:
        directions = {str(column): 1 for column in objectives}
    if not directions:
        raise ValueError("at least one finite numeric Pareto objective is required")
    _require_columns(frame, list(directions), "Pareto candidates")
    return directions


def _dominance_for_group(values: np.ndarray, atol: float) -> tuple[np.ndarray, np.ndarray]:
    size = values.shape[0]
    dominates = np.ones((size, size), dtype=bool)
    strict = np.zeros((size, size), dtype=bool)
    for index in range(values.shape[1]):
        left = values[:, index][:, None]
        right = values[:, index][None, :]
        dominates &= left >= right - atol
        strict |= left > right + atol
    dominates &= strict
    np.fill_diagonal(dominates, False)

    tiers = np.zeros(size, dtype=np.int64)
    remaining = np.ones(size, dtype=bool)
    tier = 1
    while remaining.any():
        positions = np.flatnonzero(remaining)
        subgraph = dominates[np.ix_(positions, positions)]
        front_local = ~subgraph.any(axis=0)
        if not front_local.any():  # Defensive: strict Pareto dominance is acyclic.
            raise RuntimeError("Pareto dominance graph unexpectedly contains a cycle")
        front = positions[front_local]
        tiers[front] = tier
        remaining[front] = False
        tier += 1
    return dominates, tiers


def compute_pareto_tiers(
    candidates: pd.DataFrame,
    objectives: Mapping[str, str | int] | Sequence[str] | None = None,
    *,
    group_col: str = "admin_dong_code",
    id_col: str = "venue_id",
    atol: float = 0.0,
) -> pd.DataFrame:
    """Assign strict multi-objective Pareto tiers independently per admin.

    ``dominance_count`` counts candidates that dominate the row;
    ``dominates_count`` counts candidates dominated by the row.  Tied vectors
    do not dominate one another.  Missing objectives fail closed rather than
    receiving an implicit best/worst imputation.
    """

    frame = _require_frame(candidates, "Pareto candidates")
    _require_unique(frame, [id_col], "Pareto candidates")
    _require_columns(frame, [group_col], "Pareto candidates")
    if frame[group_col].isna().any():
        raise ValueError("Pareto admin group cannot be missing")
    if not np.isfinite(float(atol)) or float(atol) < 0:
        raise ValueError("atol must be finite and non-negative")
    directions = _objective_directions(frame, objectives)
    numeric = _finite_numeric(frame, list(directions), "Pareto objectives")

    tier_result = pd.Series(index=frame.index, dtype="int64")
    dominated_by = pd.Series(index=frame.index, dtype="int64")
    dominates_count = pd.Series(index=frame.index, dtype="int64")
    for _, group in frame.groupby(group_col, sort=False, dropna=False):
        values = numeric.loc[group.index, list(directions)].to_numpy(
            dtype=float, copy=True
        )
        values *= np.asarray([directions[column] for column in directions], dtype=float)
        dominance, tiers = _dominance_for_group(values, float(atol))
        tier_result.loc[group.index] = tiers
        dominated_by.loc[group.index] = dominance.sum(axis=0).astype(np.int64)
        dominates_count.loc[group.index] = dominance.sum(axis=1).astype(np.int64)

    frame["pareto_tier"] = tier_result.astype("int64")
    frame["dominance_count"] = dominated_by.astype("int64")
    frame["dominates_count"] = dominates_count.astype("int64")
    frame["is_pareto_front"] = frame["pareto_tier"].eq(1)
    frame["is_dominated"] = frame["dominance_count"].gt(0)
    frame["dominance_flag"] = np.where(
        frame["is_pareto_front"], "NON_DOMINATED", "DOMINATED"
    )
    frame.attrs["pareto_objectives"] = directions
    frame.attrs["pareto_group_col"] = group_col
    frame.attrs["pareto_atol"] = float(atol)
    return frame


def _sort_spec(
    frame: pd.DataFrame,
    tie_breakers: Mapping[str, str | int] | None,
    id_col: str,
) -> tuple[list[str], list[bool]]:
    columns = ["pareto_tier"]
    ascending = [True]
    if tie_breakers is None:
        candidates = (
            ("need_weighted_exposure", False),
            ("raw_elderly_exposure", False),
            ("bundle_gap_weighted_exposure", False),
            ("field_validation_unknown_count", True),
        )
        for column, direction in candidates:
            if column in frame.columns:
                columns.append(column)
                ascending.append(direction)
    else:
        for column, raw_direction in tie_breakers.items():
            _require_columns(frame, [column], "shortlist candidates")
            text = str(raw_direction).lower().strip()
            if raw_direction in (1, +1) or text in {"asc", "min", "lower"}:
                is_ascending = True
            elif raw_direction in (-1,) or text in {"desc", "max", "higher"}:
                is_ascending = False
            else:
                raise ValueError(f"unsupported shortlist direction for {column}: {raw_direction}")
            columns.append(str(column))
            ascending.append(is_ascending)
    columns.append(id_col)
    ascending.append(True)
    return columns, ascending


def build_admin_shortlists(
    candidates: pd.DataFrame,
    *,
    group_col: str = "admin_dong_code",
    id_col: str = "venue_id",
    objectives: Mapping[str, str | int] | Sequence[str] | None = None,
    tie_breakers: Mapping[str, str | int] | None = None,
) -> AdminShortlistResult:
    """Build per-admin Top3, Top5, and Pareto-front views without deletion."""

    frame = _require_frame(candidates, "shortlist candidates")
    _require_unique(frame, [id_col], "shortlist candidates")
    _require_columns(frame, [group_col], "shortlist candidates")
    if "pareto_tier" not in frame.columns:
        frame = compute_pareto_tiers(
            frame, objectives, group_col=group_col, id_col=id_col
        )
    tiers = pd.to_numeric(frame["pareto_tier"], errors="coerce")
    if tiers.isna().any() or (tiers < 1).any() or ~(tiers % 1 == 0).all():
        raise ValueError("pareto_tier must contain positive integers")
    frame["pareto_tier"] = tiers.astype("int64")
    columns, ascending = _sort_spec(frame, tie_breakers, id_col)
    ordered = frame.sort_values(
        [group_col, *columns],
        ascending=[True, *ascending],
        kind="stable",
    ).copy()
    ordered["shortlist_rank"] = (
        ordered.groupby(group_col, sort=False).cumcount().add(1).astype("int64")
    )
    ordered["in_admin_top3"] = ordered["shortlist_rank"].le(3)
    ordered["in_admin_top5"] = ordered["shortlist_rank"].le(5)
    ordered["is_pareto_front"] = ordered["pareto_tier"].eq(1)
    ordered["shortlist_selected"] = ordered["in_admin_top5"] | ordered["is_pareto_front"]
    ordered["shortlist_membership"] = [
        ";".join(
            label
            for flag, label in (
                (top3, "TOP3"),
                (top5, "TOP5"),
                (front, "PARETO_FRONT"),
            )
            if flag
        )
        for top3, top5, front in zip(
            ordered["in_admin_top3"],
            ordered["in_admin_top5"],
            ordered["is_pareto_front"],
        )
    ]
    if not ordered.groupby(group_col)["shortlist_selected"].any().all():
        raise RuntimeError("every admin must retain at least one shortlist candidate")
    return AdminShortlistResult(
        candidates=ordered.reset_index(drop=True),
        top3=ordered.loc[ordered["in_admin_top3"]].reset_index(drop=True),
        top5=ordered.loc[ordered["in_admin_top5"]].reset_index(drop=True),
        pareto_front=ordered.loc[ordered["is_pareto_front"]].reset_index(drop=True),
        shortlist=ordered.loc[ordered["shortlist_selected"]].reset_index(drop=True),
    )


def build_admin_shortlist(*args: Any, **kwargs: Any) -> pd.DataFrame:
    """Convenience alias returning the non-destructive shortlist union."""

    return build_admin_shortlists(*args, **kwargs).shortlist


def _geographic_distance(
    primary: pd.Series,
    alternatives: pd.DataFrame,
    coordinate_cols: tuple[str, str] | None,
) -> np.ndarray:
    if coordinate_cols is None or not set(coordinate_cols).issubset(alternatives.columns):
        return np.full(len(alternatives), np.inf)
    first, second = coordinate_cols
    if first not in primary.index or second not in primary.index:
        return np.full(len(alternatives), np.inf)
    p1 = pd.to_numeric(pd.Series([primary[first]]), errors="coerce").iloc[0]
    p2 = pd.to_numeric(pd.Series([primary[second]]), errors="coerce").iloc[0]
    a1 = pd.to_numeric(alternatives[first], errors="coerce").to_numpy(dtype=float)
    a2 = pd.to_numeric(alternatives[second], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite([p1, p2]).all():
        return np.full(len(alternatives), np.inf)
    if "lat" in first.lower() or "lon" in second.lower():
        latitude_1 = np.radians(float(p1))
        longitude_1 = np.radians(float(p2))
        latitude_2 = np.radians(a1)
        longitude_2 = np.radians(a2)
        delta_latitude = latitude_2 - latitude_1
        delta_longitude = longitude_2 - longitude_1
        haversine = (
            np.sin(delta_latitude / 2.0) ** 2
            + np.cos(latitude_1)
            * np.cos(latitude_2)
            * np.sin(delta_longitude / 2.0) ** 2
        )
        return 2.0 * 6_371_008.8 * np.arcsin(np.sqrt(np.clip(haversine, 0.0, 1.0)))
    return np.hypot(a1 - float(p1), a2 - float(p2))


def build_coverage_cluster_fallbacks(
    shortlist: pd.DataFrame,
    all_candidates: pd.DataFrame | None = None,
    *,
    id_col: str = "venue_id",
    group_col: str = "admin_dong_code",
    sigungu_col: str = "sigungu",
    cluster_col: str = "coverage_cluster_id",
    exposure_col: str = "need_weighted_exposure",
    physical_id_col: str | None = None,
    coordinate_cols: tuple[str, str] | None = None,
    n_fallbacks: int = 2,
) -> pd.DataFrame:
    """Choose distinct fallbacks, prioritising coverage-equivalent clusters."""

    primary = _require_frame(shortlist, "shortlist")
    pool = _require_frame(
        all_candidates if all_candidates is not None else shortlist,
        "fallback candidate pool",
    )
    _require_unique(primary, [id_col], "shortlist")
    _require_unique(pool, [id_col], "fallback candidate pool")
    _require_columns(primary, [id_col, group_col, cluster_col], "shortlist")
    _require_columns(pool, [id_col, group_col, cluster_col], "fallback candidate pool")
    if int(n_fallbacks) < 1:
        raise ValueError("n_fallbacks must be at least one")
    if exposure_col in pool.columns:
        pool_exposure = pd.to_numeric(pool[exposure_col], errors="coerce")
    else:
        pool_exposure = pd.Series(np.nan, index=pool.index)
    effective_coordinate_cols = coordinate_cols
    if effective_coordinate_cols is None:
        if {"x_5179", "y_5179"}.issubset(pool.columns):
            effective_coordinate_cols = ("x_5179", "y_5179")
        elif {"latitude", "longitude"}.issubset(pool.columns):
            effective_coordinate_cols = ("latitude", "longitude")

    rows: list[dict[str, Any]] = []
    for _, venue in primary.iterrows():
        available = pool.loc[pool[id_col].astype(str).ne(str(venue[id_col]))].copy()
        if physical_id_col and physical_id_col in pool.columns and physical_id_col in venue.index:
            available = available.loc[
                available[physical_id_col].astype(str).ne(str(venue[physical_id_col]))
            ]
        cluster_known = pd.notna(venue[cluster_col]) and str(venue[cluster_col]).strip() != ""
        available["_same_cluster"] = (
            cluster_known
            & available[cluster_col].notna()
            & available[cluster_col].astype(str).eq(str(venue[cluster_col]))
        )
        available["_same_admin"] = available[group_col].astype(str).eq(str(venue[group_col]))
        if sigungu_col in available.columns and sigungu_col in venue.index:
            available["_same_sigungu"] = available[sigungu_col].astype(str).eq(
                str(venue[sigungu_col])
            )
        else:
            available["_same_sigungu"] = False
        # An unrelated, distant venue is not a meaningful operational
        # fallback.  Search the equivalent cluster first, then the same admin
        # or sigungu, and leave the slot null if no such candidate exists.
        available = available.loc[
            available["_same_cluster"]
            | available["_same_admin"]
            | available["_same_sigungu"]
        ].copy()
        if exposure_col in venue.index and pd.notna(venue[exposure_col]):
            reference = float(venue[exposure_col])
            available["_exposure_gap"] = (
                pool_exposure.loc[available.index] - reference
            ).abs().fillna(np.inf)
        else:
            available["_exposure_gap"] = np.inf
        available["_distance"] = _geographic_distance(
            venue, available, effective_coordinate_cols
        )
        if "pareto_tier" in available.columns:
            available["_pareto_tier"] = pd.to_numeric(
                available["pareto_tier"], errors="coerce"
            ).fillna(np.inf)
        else:
            available["_pareto_tier"] = np.inf
        available = available.sort_values(
            [
                "_same_cluster",
                "_same_admin",
                "_same_sigungu",
                "_exposure_gap",
                "_distance",
                "_pareto_tier",
                id_col,
            ],
            ascending=[False, False, False, True, True, True, True],
            kind="stable",
        )
        selected = available.head(int(n_fallbacks))
        record: dict[str, Any] = {id_col: venue[id_col]}
        for position in range(int(n_fallbacks)):
            number = position + 1
            if position < len(selected):
                alternative = selected.iloc[position]
                record[f"fallback_venue_{number}"] = alternative[id_col]
                record[f"fallback_{number}_same_cluster"] = bool(alternative["_same_cluster"])
                record[f"fallback_{number}_distance"] = (
                    float(alternative["_distance"])
                    if np.isfinite(alternative["_distance"])
                    else np.nan
                )
            else:
                record[f"fallback_venue_{number}"] = pd.NA
                record[f"fallback_{number}_same_cluster"] = False
                record[f"fallback_{number}_distance"] = np.nan
        rows.append(record)
    result = pd.DataFrame(rows)
    for number in range(1, int(n_fallbacks) + 1):
        invalid = result[f"fallback_venue_{number}"].astype("string").eq(
            result[id_col].astype("string")
        )
        if invalid.fillna(False).any():
            raise RuntimeError("a venue cannot be its own fallback")
    return result


def collapse_admin_venues(
    venues: pd.DataFrame,
    *,
    group_col: str = "admin_dong_code",
    id_col: str = "venue_id",
    strategy: str = "best",
    best_by: str = "need_weighted_exposure",
    metric_cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Collapse venue records to one independent policy record per admin."""

    frame = _require_frame(venues, "venues")
    _require_columns(frame, [group_col], "venues")
    if frame[group_col].isna().any():
        raise ValueError("admin group cannot be missing")
    strategy = str(strategy).strip().lower()
    counts = frame.groupby(group_col, sort=False).size().rename("candidate_count")
    if strategy in {"best", "admin_best"}:
        _require_columns(frame, [id_col, best_by], "venues")
        score = pd.to_numeric(frame[best_by], errors="coerce")
        if not np.isfinite(score).all():
            raise ValueError(f"best-by column must be finite numeric: {best_by}")
        ordered = frame.assign(_collapse_score=score).sort_values(
            [group_col, "_collapse_score", id_col],
            ascending=[True, False, True],
            kind="stable",
        )
        collapsed = ordered.groupby(group_col, sort=False, as_index=False).head(1)
        collapsed = collapsed.drop(columns="_collapse_score").copy()
        collapsed["admin_representative_method"] = f"BEST_BY_{best_by}"
    elif strategy in {"median", "admin_median"}:
        if metric_cols is None:
            metric_cols = [
                column
                for column in frame.select_dtypes(include=[np.number]).columns
                if column != group_col
            ]
        metric_cols = list(dict.fromkeys(metric_cols))
        _require_columns(frame, metric_cols, "venues")
        numeric = frame.loc[:, metric_cols].apply(pd.to_numeric, errors="coerce")
        if not np.isfinite(numeric.to_numpy(dtype=float)).all():
            raise ValueError("admin median metrics must be finite numeric")
        numeric[group_col] = frame[group_col].to_numpy()
        collapsed = numeric.groupby(group_col, sort=False, as_index=False).median()
        collapsed["admin_representative_method"] = "MEDIAN_OF_VENUES"
    else:
        raise ValueError("strategy must be 'best' or 'median'")
    collapsed["candidate_count"] = collapsed[group_col].map(counts).astype("int64")
    if collapsed[group_col].duplicated().any():
        raise RuntimeError("admin collapse did not produce one row per admin")
    return collapsed.reset_index(drop=True)


def build_need_exposure_quadrants(
    venues: pd.DataFrame,
    *,
    group_col: str = "admin_dong_code",
    id_col: str = "venue_id",
    need_col: str = "structural_need",
    exposure_col: str = "need_weighted_exposure",
    collapse: str = "best",
    need_threshold: float | str = "median",
    exposure_threshold: float | str = "median",
) -> pd.DataFrame:
    """Create the four policy quadrants after one-record-per-admin collapse."""

    collapsed = collapse_admin_venues(
        venues,
        group_col=group_col,
        id_col=id_col,
        strategy=collapse,
        best_by=exposure_col,
        metric_cols=[need_col, exposure_col],
    )
    _require_columns(collapsed, [need_col, exposure_col], "admin-collapsed venues")
    values = _finite_numeric(
        collapsed, [need_col, exposure_col], "Need-Exposure quadrant values"
    )

    def threshold(value: float | str, column: str) -> float:
        if isinstance(value, str):
            if value.strip().lower() != "median":
                raise ValueError("quadrant threshold string must be 'median'")
            return float(values[column].median())
        result = float(value)
        if not np.isfinite(result):
            raise ValueError("quadrant thresholds must be finite")
        return result

    need_cut = threshold(need_threshold, need_col)
    exposure_cut = threshold(exposure_threshold, exposure_col)
    high_need = values[need_col].ge(need_cut)
    high_exposure = values[exposure_col].ge(exposure_cut)
    collapsed["need_threshold"] = need_cut
    collapsed["exposure_threshold"] = exposure_cut
    collapsed["high_need"] = high_need.to_numpy()
    collapsed["high_exposure"] = high_exposure.to_numpy()
    collapsed["need_exposure_quadrant"] = np.select(
        [
            high_need & high_exposure,
            high_need & ~high_exposure,
            ~high_need & high_exposure,
        ],
        [
            "Q1_HIGH_NEED_HIGH_EXPOSURE",
            "Q2_HIGH_NEED_LOW_EXPOSURE",
            "Q3_LOW_NEED_HIGH_EXPOSURE",
        ],
        default="Q4_LOW_NEED_LOW_EXPOSURE",
    )
    collapsed["quadrant"] = collapsed["need_exposure_quadrant"]
    return collapsed


def _pairwise_spearman(
    frame: pd.DataFrame,
    metrics: Sequence[str],
    min_periods: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    numeric = frame.loc[:, list(metrics)].apply(pd.to_numeric, errors="coerce")
    corr = numeric.corr(method="spearman", min_periods=int(min_periods))
    valid = numeric.notna().to_numpy(dtype=np.int64)
    counts = pd.DataFrame(valid.T @ valid, index=metrics, columns=metrics)
    corr = corr.mask(counts < int(min_periods))
    return corr, counts


def stage3_spearman_audit(
    venues: pd.DataFrame,
    metrics: Sequence[str],
    *,
    group_col: str = "admin_dong_code",
    id_col: str = "venue_id",
    collapse: str = "best",
    best_by: str = "need_weighted_exposure",
    min_periods: int = 3,
) -> SpearmanAuditResult:
    """Contrast descriptive all-venue correlations with admin-level evidence."""

    frame = _require_frame(venues, "Stage 3 correlation input")
    metrics = tuple(dict.fromkeys(str(metric) for metric in metrics))
    if len(metrics) < 2:
        raise ValueError("at least two correlation metrics are required")
    _require_columns(frame, [group_col, *metrics], "Stage 3 correlation input")
    if int(min_periods) < 2:
        raise ValueError("min_periods must be at least two")
    all_corr, all_n = _pairwise_spearman(frame, metrics, int(min_periods))
    collapsed = collapse_admin_venues(
        frame,
        group_col=group_col,
        id_col=id_col,
        strategy=collapse,
        best_by=best_by,
        metric_cols=metrics,
    )
    if int(min_periods) > len(collapsed):
        raise ValueError("min_periods exceeds the number of independent admin records")
    admin_corr, admin_n = _pairwise_spearman(collapsed, metrics, int(min_periods))
    rows: list[dict[str, Any]] = []
    for left_position, left in enumerate(metrics):
        for right in metrics[left_position + 1 :]:
            rho_all = all_corr.loc[left, right]
            rho_admin = admin_corr.loc[left, right]
            rows.append(
                {
                    "metric_1": left,
                    "metric_2": right,
                    "rho_all_venue_descriptive": rho_all,
                    "n_all_venue": int(all_n.loc[left, right]),
                    "rho_admin_collapsed": rho_admin,
                    "n_admin_collapsed": int(admin_n.loc[left, right]),
                    "absolute_rho_difference": (
                        abs(float(rho_all) - float(rho_admin))
                        if pd.notna(rho_all) and pd.notna(rho_admin)
                        else np.nan
                    ),
                    "inference_level": "ADMIN_COLLAPSED_POLICY_UNITS",
                    "all_venue_role": "DESCRIPTIVE_ONLY",
                }
            )
    all_corr.attrs.update(
        {
            "analysis_level": "all_venue_descriptive",
            "sample_n": int(len(frame)),
            "independence_warning": "venues within an admin are not independent policy units",
        }
    )
    all_n.attrs.update(
        {
            "analysis_level": "all_venue_descriptive_pairwise_n",
            "sample_n": int(len(frame)),
        }
    )
    admin_corr.attrs.update(
        {
            "analysis_level": "admin_collapsed",
            "sample_n": int(len(collapsed)),
            "collapse_method": collapsed["admin_representative_method"].iloc[0],
        }
    )
    admin_n.attrs.update(
        {
            "analysis_level": "admin_collapsed_pairwise_n",
            "sample_n": int(len(collapsed)),
            "collapse_method": collapsed["admin_representative_method"].iloc[0],
        }
    )
    return SpearmanAuditResult(
        all_venue_correlation=all_corr,
        all_venue_counts=all_n,
        admin_collapsed_correlation=admin_corr,
        admin_collapsed_counts=admin_n,
        admin_collapsed=collapsed,
        comparison=pd.DataFrame(rows),
    )


def _ranked_ids(
    frame: pd.DataFrame,
    score_col: str,
    id_col: str,
    count: int,
) -> list[str]:
    working = frame.reset_index(drop=True)
    return (
        working.sort_values([score_col, id_col], ascending=[False, True], kind="stable")
        .head(min(int(count), len(frame)))[id_col]
        .astype(str)
        .tolist()
    )


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return float(len(left & right) / len(union)) if union else 1.0


def _median_relative_change(baseline: np.ndarray, alternative: np.ndarray) -> float:
    """Return the median venue-level relative change where baseline is positive."""

    positive = baseline > 0
    if not positive.any():
        return np.nan
    change = (alternative[positive] - baseline[positive]) / baseline[positive]
    return float(np.median(change))


def _sensitivity_exposure_change_metrics(
    baseline: pd.DataFrame,
    alternative: pd.DataFrame,
    *,
    raw_exposure_col: str,
    cross_admin_exposure_col: str,
    border_share_threshold: float,
    label: str,
) -> dict[str, Any]:
    """Summarise exposure and cross-boundary changes for one comparison scope.

    ``border_venue_frequency`` is a diagnostic proxy: the fraction of venue
    catchments whose cross-admin exposure share is at least the configured
    threshold.  It is not a geometric distance-to-boundary classification.
    Cross-admin share is deliberately derived from the scenario-specific raw
    and cross-admin exposures instead of trusting a potentially stale column.
    """

    metrics: dict[str, Any] = {
        "raw_exposure_median_baseline": np.nan,
        "raw_exposure_median_alternative": np.nan,
        "raw_exposure_median_absolute_change": np.nan,
        "raw_exposure_venue_relative_change_median": np.nan,
        "cross_admin_exposure_median_baseline": np.nan,
        "cross_admin_exposure_median_alternative": np.nan,
        "cross_admin_exposure_median_absolute_change": np.nan,
        "cross_admin_exposure_venue_relative_change_median": np.nan,
        "cross_admin_share_median_baseline": np.nan,
        "cross_admin_share_median_alternative": np.nan,
        "cross_admin_share_median_change": np.nan,
        "border_venue_count_baseline": np.nan,
        "border_venue_count_alternative": np.nan,
        "border_venue_frequency_baseline": np.nan,
        "border_venue_frequency_alternative": np.nan,
        "border_venue_frequency_change": np.nan,
        "border_venue_share_threshold": float(border_share_threshold),
        "border_venue_definition": "cross_admin_share_ge_threshold",
    }
    if raw_exposure_col not in baseline.columns or raw_exposure_col not in alternative.columns:
        return metrics

    base_raw = pd.to_numeric(baseline[raw_exposure_col], errors="coerce").to_numpy(
        dtype=float
    )
    alt_raw = pd.to_numeric(alternative[raw_exposure_col], errors="coerce").to_numpy(
        dtype=float
    )
    if not np.isfinite(base_raw).all() or not np.isfinite(alt_raw).all():
        raise ValueError(f"sensitivity raw exposure must be finite: {label}")
    if (base_raw < 0).any() or (alt_raw < 0).any():
        raise ValueError(f"sensitivity raw exposure must be non-negative: {label}")
    base_raw_median = float(np.median(base_raw))
    alt_raw_median = float(np.median(alt_raw))
    metrics.update(
        {
            "raw_exposure_median_baseline": base_raw_median,
            "raw_exposure_median_alternative": alt_raw_median,
            "raw_exposure_median_absolute_change": alt_raw_median - base_raw_median,
            "raw_exposure_venue_relative_change_median": _median_relative_change(
                base_raw, alt_raw
            ),
        }
    )

    if (
        cross_admin_exposure_col not in baseline.columns
        or cross_admin_exposure_col not in alternative.columns
    ):
        return metrics
    base_cross = pd.to_numeric(
        baseline[cross_admin_exposure_col], errors="coerce"
    ).to_numpy(dtype=float)
    alt_cross = pd.to_numeric(
        alternative[cross_admin_exposure_col], errors="coerce"
    ).to_numpy(dtype=float)
    if not np.isfinite(base_cross).all() or not np.isfinite(alt_cross).all():
        raise ValueError(f"sensitivity cross-admin exposure must be finite: {label}")
    if (base_cross < 0).any() or (alt_cross < 0).any():
        raise ValueError(
            f"sensitivity cross-admin exposure must be non-negative: {label}"
        )
    tolerance = 1e-8
    if (base_cross > base_raw + tolerance).any() or (
        alt_cross > alt_raw + tolerance
    ).any():
        raise ValueError(
            f"sensitivity cross-admin exposure cannot exceed raw exposure: {label}"
        )

    base_share = np.divide(
        base_cross,
        base_raw,
        out=np.zeros_like(base_cross),
        where=base_raw > 0,
    )
    alt_share = np.divide(
        alt_cross,
        alt_raw,
        out=np.zeros_like(alt_cross),
        where=alt_raw > 0,
    )
    base_cross_median = float(np.median(base_cross))
    alt_cross_median = float(np.median(alt_cross))
    base_share_median = float(np.median(base_share))
    alt_share_median = float(np.median(alt_share))
    base_border = base_share >= float(border_share_threshold)
    alt_border = alt_share >= float(border_share_threshold)
    base_frequency = float(base_border.mean())
    alt_frequency = float(alt_border.mean())
    metrics.update(
        {
            "cross_admin_exposure_median_baseline": base_cross_median,
            "cross_admin_exposure_median_alternative": alt_cross_median,
            "cross_admin_exposure_median_absolute_change": (
                alt_cross_median - base_cross_median
            ),
            "cross_admin_exposure_venue_relative_change_median": (
                _median_relative_change(base_cross, alt_cross)
            ),
            "cross_admin_share_median_baseline": base_share_median,
            "cross_admin_share_median_alternative": alt_share_median,
            "cross_admin_share_median_change": alt_share_median - base_share_median,
            "border_venue_count_baseline": int(base_border.sum()),
            "border_venue_count_alternative": int(alt_border.sum()),
            "border_venue_frequency_baseline": base_frequency,
            "border_venue_frequency_alternative": alt_frequency,
            "border_venue_frequency_change": alt_frequency - base_frequency,
        }
    )
    return metrics


def compute_sensitivity_metrics(
    baseline: pd.DataFrame,
    alternatives: Mapping[str, pd.DataFrame] | pd.DataFrame,
    *,
    id_col: str = "venue_id",
    group_col: str = "admin_dong_code",
    score_col: str = "need_weighted_exposure",
    pareto_col: str = "is_pareto_front",
    alternative_name: str = "alternative",
    raw_exposure_col: str = "raw_elderly_exposure",
    cross_admin_exposure_col: str = "cross_admin_exposure",
    border_share_threshold: float = 0.50,
    use_existing_shortlist_flags: bool = False,
) -> SensitivityResult:
    """Measure rank, Pareto, exposure, and cross-boundary stability.

    Exposure-change fields are emitted at both the whole-universe level and
    per administrative unit.  If raw exposure is absent, those optional
    fields remain missing; if raw exposure exists but cross-admin exposure is
    absent, only raw-exposure change is reported.
    """

    base = _require_frame(baseline, "sensitivity baseline")
    _require_unique(base, [id_col], "sensitivity baseline")
    _require_columns(base, [id_col, group_col, score_col], "sensitivity baseline")
    scenario_map = (
        {alternative_name: alternatives}
        if isinstance(alternatives, pd.DataFrame)
        else dict(alternatives)
    )
    if not scenario_map:
        raise ValueError("at least one sensitivity alternative is required")
    threshold = float(border_share_threshold)
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("border_share_threshold must lie in [0, 1]")
    base_score = pd.to_numeric(base[score_col], errors="coerce")
    if not np.isfinite(base_score).all():
        raise ValueError("baseline sensitivity score must be finite")
    base_indexed = base.assign(_score=base_score).set_index(id_col, drop=False)
    shortlist_columns = {"in_admin_top3", "in_admin_top5", "shortlist_rank"}
    if use_existing_shortlist_flags:
        _require_columns(base, sorted(shortlist_columns), "sensitivity baseline shortlist")
    base_ids = set(base_indexed.index.astype(str))
    # Normalise indexes to strings so identifiers such as 1 and "1" cannot
    # align inconsistently between set checks and correlation calculations.
    base_indexed.index = base_indexed.index.astype(str)
    base_ids = set(base_indexed.index)
    summary_rows: list[dict[str, Any]] = []
    admin_rows: list[dict[str, Any]] = []

    for scenario, raw_alternative in scenario_map.items():
        alternative = _require_frame(raw_alternative, f"sensitivity {scenario}")
        _require_unique(alternative, [id_col], f"sensitivity {scenario}")
        _require_columns(
            alternative,
            [id_col, group_col, score_col],
            f"sensitivity {scenario}",
        )
        alternative_score = pd.to_numeric(alternative[score_col], errors="coerce")
        if not np.isfinite(alternative_score).all():
            raise ValueError(f"sensitivity score must be finite: {scenario}")
        alternative_indexed = alternative.assign(_score=alternative_score).set_index(
            id_col, drop=False
        )
        if use_existing_shortlist_flags:
            _require_columns(
                alternative,
                sorted(shortlist_columns),
                f"sensitivity {scenario} shortlist",
            )
        alternative_indexed.index = alternative_indexed.index.astype(str)
        if set(alternative_indexed.index) != base_ids:
            raise ValueError(f"sensitivity candidate ids differ from baseline: {scenario}")
        alternative_indexed = alternative_indexed.reindex(base_indexed.index)
        if not alternative_indexed[group_col].astype(str).eq(
            base_indexed[group_col].astype(str)
        ).all():
            raise ValueError(f"sensitivity admin assignment changed: {scenario}")
        rho = base_indexed["_score"].corr(alternative_indexed["_score"], method="spearman")

        for admin, base_group in base_indexed.groupby(group_col, sort=True):
            ids = base_group.index
            alt_group = alternative_indexed.loc[ids]
            if use_existing_shortlist_flags:
                base_top3 = set(
                    base_group.loc[base_group["in_admin_top3"].map(_coerce_bool), id_col].astype(str)
                )
                alt_top3 = set(
                    alt_group.loc[alt_group["in_admin_top3"].map(_coerce_bool), id_col].astype(str)
                )
                base_top5 = set(
                    base_group.loc[base_group["in_admin_top5"].map(_coerce_bool), id_col].astype(str)
                )
                alt_top5 = set(
                    alt_group.loc[alt_group["in_admin_top5"].map(_coerce_bool), id_col].astype(str)
                )
                base_best = str(
                    base_group.reset_index(drop=True)
                    .sort_values(["shortlist_rank", id_col], kind="stable")[id_col]
                    .iloc[0]
                )
                alt_best = str(
                    alt_group.reset_index(drop=True)
                    .sort_values(["shortlist_rank", id_col], kind="stable")[id_col]
                    .iloc[0]
                )
            else:
                base_top3 = set(_ranked_ids(base_group, "_score", id_col, 3))
                alt_top3 = set(_ranked_ids(alt_group, "_score", id_col, 3))
                base_top5 = set(_ranked_ids(base_group, "_score", id_col, 5))
                alt_top5 = set(_ranked_ids(alt_group, "_score", id_col, 5))
                base_best = _ranked_ids(base_group, "_score", id_col, 1)[0]
                alt_best = _ranked_ids(alt_group, "_score", id_col, 1)[0]
            if pareto_col in base_group.columns:
                base_pareto = set(
                    base_group.loc[base_group[pareto_col].map(_coerce_bool), id_col].astype(str)
                )
            elif "pareto_tier" in base_group.columns:
                base_pareto = set(
                    base_group.loc[
                        pd.to_numeric(base_group["pareto_tier"], errors="coerce").eq(1),
                        id_col,
                    ].astype(str)
                )
            else:
                base_pareto = set()
            if pareto_col in alt_group.columns:
                alt_pareto = set(
                    alt_group.loc[alt_group[pareto_col].map(_coerce_bool), id_col].astype(str)
                )
            elif "pareto_tier" in alt_group.columns:
                alt_pareto = set(
                    alt_group.loc[
                        pd.to_numeric(alt_group["pareto_tier"], errors="coerce").eq(1),
                        id_col,
                    ].astype(str)
                )
            else:
                alt_pareto = set()
            retention = (
                float(len(base_pareto & alt_pareto) / len(base_pareto))
                if base_pareto
                else np.nan
            )
            exposure_changes = _sensitivity_exposure_change_metrics(
                base_group,
                alt_group,
                raw_exposure_col=raw_exposure_col,
                cross_admin_exposure_col=cross_admin_exposure_col,
                border_share_threshold=threshold,
                label=f"{scenario}/{admin}",
            )
            admin_rows.append(
                {
                    "scenario": str(scenario),
                    group_col: admin,
                    "candidate_count": int(len(base_group)),
                    "top3_jaccard": _jaccard(base_top3, alt_top3),
                    "top5_jaccard": _jaccard(base_top5, alt_top5),
                    "best_venue_stable": base_best == alt_best,
                    "pareto_retention": retention,
                    **exposure_changes,
                }
            )
        detail = pd.DataFrame([row for row in admin_rows if row["scenario"] == str(scenario)])
        base_pareto_all: set[str]
        alt_pareto_all: set[str]
        if pareto_col in base_indexed.columns and pareto_col in alternative_indexed.columns:
            base_pareto_all = set(
                base_indexed.loc[base_indexed[pareto_col].map(_coerce_bool), id_col].astype(str)
            )
            alt_pareto_all = set(
                alternative_indexed.loc[
                    alternative_indexed[pareto_col].map(_coerce_bool), id_col
                ].astype(str)
            )
        elif "pareto_tier" in base_indexed.columns and "pareto_tier" in alternative_indexed.columns:
            base_pareto_all = set(
                base_indexed.loc[
                    pd.to_numeric(base_indexed["pareto_tier"], errors="coerce").eq(1),
                    id_col,
                ].astype(str)
            )
            alt_pareto_all = set(
                alternative_indexed.loc[
                    pd.to_numeric(alternative_indexed["pareto_tier"], errors="coerce").eq(1),
                    id_col,
                ].astype(str)
            )
        else:
            base_pareto_all = set()
            alt_pareto_all = set()
        global_retention = (
            float(len(base_pareto_all & alt_pareto_all) / len(base_pareto_all))
            if base_pareto_all
            else np.nan
        )
        exposure_changes = _sensitivity_exposure_change_metrics(
            base_indexed,
            alternative_indexed,
            raw_exposure_col=raw_exposure_col,
            cross_admin_exposure_col=cross_admin_exposure_col,
            border_share_threshold=threshold,
            label=str(scenario),
        )
        summary_rows.append(
            {
                "scenario": str(scenario),
                "venue_count": int(len(base_indexed)),
                "admin_count": int(detail[group_col].nunique()),
                "venue_rank_spearman": float(rho) if pd.notna(rho) else np.nan,
                "admin_top3_jaccard_mean": float(detail["top3_jaccard"].mean()),
                "admin_top3_jaccard_min": float(detail["top3_jaccard"].min()),
                "admin_top5_jaccard_mean": float(detail["top5_jaccard"].mean()),
                "admin_top5_jaccard_min": float(detail["top5_jaccard"].min()),
                "region_best_venue_stability": float(detail["best_venue_stable"].mean()),
                "pareto_retention": global_retention,
                "admin_pareto_retention_mean": float(detail["pareto_retention"].mean())
                if detail["pareto_retention"].notna().any()
                else np.nan,
                **exposure_changes,
            }
        )
    return SensitivityResult(
        summary=pd.DataFrame(summary_rows),
        by_admin=pd.DataFrame(admin_rows),
    )


def build_sensitivity_metrics(*args: Any, **kwargs: Any) -> pd.DataFrame:
    """Convenience alias returning scenario-level sensitivity metrics."""

    return compute_sensitivity_metrics(*args, **kwargs).summary


def build_field_validation_queue(
    shortlist: pd.DataFrame,
    *,
    admin_col: str = "admin_dong_code",
    id_col: str = "venue_id",
    name_col: str = "venue_name",
    exposure_col: str = "raw_elderly_exposure",
    need_exposure_col: str = "need_weighted_exposure",
    bundle_relevance_col: str = "bundle_gap_weighted_exposure",
    top_k: int = 3,
    required_only: bool = True,
) -> pd.DataFrame:
    """Create a bounded phone/site validation queue from shortlist venues."""

    frame = _require_frame(shortlist, "field-validation shortlist")
    required = [
        admin_col,
        id_col,
        name_col,
        "pareto_tier",
        exposure_col,
        need_exposure_col,
        "field_validation_unknown_count",
    ]
    _require_columns(frame, required, "field-validation shortlist")
    _require_unique(frame, [id_col], "field-validation shortlist")
    if int(top_k) < 1:
        raise ValueError("top_k must be at least one")
    if "field_validation_required" not in frame.columns:
        frame["field_validation_required"] = frame[
            "field_validation_unknown_count"
        ].gt(0)
    else:
        frame["field_validation_required"] = frame["field_validation_required"].map(
            _coerce_bool
        )
    if required_only:
        frame = frame.loc[frame["field_validation_required"]].copy()
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "admin_dong",
                id_col,
                name_col,
                "pareto_tier",
                "exposure",
                "need_weighted_exposure",
                "bundle_relevance",
                "field_validation_unknown_count",
                "reason_to_validate",
                "field_validation_queue_rank",
            ]
        )
    numeric_columns = [
        "pareto_tier",
        exposure_col,
        need_exposure_col,
        "field_validation_unknown_count",
    ]
    _finite_numeric(frame, numeric_columns, "field-validation queue")
    if bundle_relevance_col not in frame.columns:
        frame[bundle_relevance_col] = np.nan
    shortlist_rank = (
        pd.to_numeric(frame["shortlist_rank"], errors="coerce")
        if "shortlist_rank" in frame.columns
        else pd.Series(np.inf, index=frame.index)
    )
    frame["_shortlist_rank"] = shortlist_rank.fillna(np.inf)
    frame = frame.sort_values(
        [
            admin_col,
            "pareto_tier",
            "_shortlist_rank",
            "field_validation_unknown_count",
            need_exposure_col,
            id_col,
        ],
        ascending=[True, True, True, False, False, True],
        kind="stable",
    )
    frame["field_validation_queue_rank"] = (
        frame.groupby(admin_col, sort=False).cumcount().add(1).astype("int64")
    )
    frame = frame.loc[frame["field_validation_queue_rank"].le(int(top_k))].copy()
    frame["reason_to_validate"] = [
        ";".join(
            [
                f"PARETO_TIER={int(tier)}",
                f"FIELD_UNKNOWN={int(unknown)}",
                "HIGH_POLICY_SHORTLIST_PRIORITY",
            ]
        )
        for tier, unknown in zip(
            frame["pareto_tier"], frame["field_validation_unknown_count"]
        )
    ]
    queue = pd.DataFrame(
        {
            "admin_dong": frame[admin_col].to_numpy(),
            id_col: frame[id_col].to_numpy(),
            name_col: frame[name_col].to_numpy(),
            "pareto_tier": frame["pareto_tier"].astype(int).to_numpy(),
            "exposure": pd.to_numeric(frame[exposure_col]).to_numpy(),
            "need_weighted_exposure": pd.to_numeric(frame[need_exposure_col]).to_numpy(),
            "bundle_relevance": pd.to_numeric(
                frame[bundle_relevance_col], errors="coerce"
            ).to_numpy(),
            "field_validation_unknown_count": frame[
                "field_validation_unknown_count"
            ].astype(int).to_numpy(),
            "reason_to_validate": frame["reason_to_validate"].to_numpy(),
            "field_validation_queue_rank": frame[
                "field_validation_queue_rank"
            ].astype(int).to_numpy(),
        }
    )
    return queue.reset_index(drop=True)


def validate_temporal_abstention(
    interface: pd.DataFrame,
    *,
    require_official_abstention: bool = True,
) -> None:
    """Enforce the official Stage 2B abstention in the Stage 3 interface.

    The default is deliberately tied to the current frozen release: all five
    bundles are ``NO_STRONG_PREFERENCE / LOW``.  A future pipeline may opt out
    only after Stage 2B itself has issued a new official release.
    """

    _require_columns(
        interface,
        [
            "temporal_release_type",
            "temporal_confidence",
            "recommended_season",
            "fallback_season",
        ],
        "Stage3-to-Stage4 interface",
    )
    release = interface["temporal_release_type"].astype(str).str.strip().str.upper()
    if release.eq("").any() or interface["temporal_release_type"].isna().any():
        raise ValueError("temporal_release_type must be populated")
    unknown_release = sorted(set(release) - ({ABSTENTION_RELEASE} | ACTIONABLE_RELEASES))
    if unknown_release:
        raise ValueError(f"unsupported temporal release types: {unknown_release}")
    confidence = interface["temporal_confidence"].astype(str).str.strip().str.upper()
    if interface["temporal_confidence"].isna().any() or confidence.eq("").any():
        raise ValueError("temporal_confidence must be populated")
    recommended = interface["recommended_season"].astype("string").str.strip().str.lower()
    fallback = interface["fallback_season"].astype("string").str.strip().str.lower()
    recommended = recommended.mask(recommended.eq(""))
    fallback = fallback.mask(fallback.eq(""))
    if require_official_abstention:
        if not release.eq(ABSTENTION_RELEASE).all():
            raise ValueError(
                "current frozen Stage 2B requires NO_STRONG_PREFERENCE for every row"
            )
        if not confidence.eq("LOW").all():
            raise ValueError("current frozen Stage 2B requires LOW temporal confidence")
    abstain = release.eq(ABSTENTION_RELEASE)
    if recommended.loc[abstain].notna().any() or fallback.loc[abstain].notna().any():
        raise ValueError(
            "NO_STRONG_PREFERENCE rows must keep recommended/fallback season null"
        )
    actionable = ~abstain
    if recommended.loc[actionable].isna().any() or fallback.loc[actionable].isna().any():
        raise ValueError("actionable temporal releases require two populated seasons")
    if not set(recommended.loc[actionable].dropna()).issubset(VALID_SEASONS):
        raise ValueError("recommended_season contains an unsupported value")
    if not set(fallback.loc[actionable].dropna()).issubset(VALID_SEASONS):
        raise ValueError("fallback_season contains an unsupported value")
    if recommended.loc[actionable].eq(fallback.loc[actionable]).any():
        raise ValueError("fallback_season must differ from recommended_season")
    for column in ("exact_month", "exact_date", "recommended_month", "recommended_date"):
        if column in interface.columns and interface[column].notna().any():
            raise ValueError(f"{column} must remain null at Stage 3")


def validate_stage3_to_stage4_interface(
    interface: pd.DataFrame,
    *,
    allow_extra_columns: bool = True,
    require_official_temporal_abstention: bool = True,
) -> pd.DataFrame:
    """Fail closed on schema, Stage 4 leakage, and temporal overclaiming."""

    frame = _require_frame(interface, "Stage3-to-Stage4 interface")
    _require_columns(frame, INTERFACE_REQUIRED_COLUMNS, "Stage3-to-Stage4 interface")
    lower_to_actual = {str(column).lower(): str(column) for column in frame.columns}
    forbidden = sorted(
        (STAGE4_FORBIDDEN_COLUMNS | TEMPORAL_SELECTOR_FORBIDDEN_COLUMNS)
        & set(lower_to_actual)
    )
    if forbidden:
        raise ValueError(f"Stage3 interface contains forbidden future columns: {forbidden}")
    if not allow_extra_columns:
        extra = sorted(set(frame.columns) - set(INTERFACE_REQUIRED_COLUMNS))
        if extra:
            raise ValueError(f"Stage3 interface contains unexpected columns: {extra}")
    _require_unique(frame, ["venue_id", "bundle_id"], "Stage3-to-Stage4 interface")
    if frame["venue_name"].isna().any() or frame["venue_name"].astype(str).str.strip().eq("").any():
        raise ValueError("venue_name must be populated")
    if frame["admin_dong_code"].isna().any() or frame["sigungu"].isna().any():
        raise ValueError("admin and sigungu mappings must be populated")
    numeric_nonnegative = (
        "structural_need",
        "raw_elderly_exposure",
        "need_weighted_exposure",
        "high_need_elderly_exposure",
        "own_admin_exposure",
        "cross_admin_exposure",
        "cross_sigungu_exposure",
        "specialty_gap",
        "bundle_gap_weighted_exposure",
        "venue_readiness_prior",
        "known_outreach_overlap",
    )
    numeric = _finite_numeric(
        frame, numeric_nonnegative, "Stage3 interface metrics", nonnegative=True
    )
    coordinates = _finite_numeric(
        frame, ["latitude", "longitude"], "Stage3 interface coordinates"
    )
    if not coordinates["latitude"].between(-90, 90).all() or not coordinates[
        "longitude"
    ].between(-180, 180).all():
        raise ValueError("Stage3 interface coordinates are outside geographic bounds")
    if (numeric["own_admin_exposure"] > numeric["raw_elderly_exposure"] + 1e-9).any():
        raise ValueError("own_admin_exposure cannot exceed raw_elderly_exposure")
    if (numeric["cross_admin_exposure"] > numeric["raw_elderly_exposure"] + 1e-9).any():
        raise ValueError("cross_admin_exposure cannot exceed raw_elderly_exposure")
    if (numeric["cross_sigungu_exposure"] > numeric["cross_admin_exposure"] + 1e-9).any():
        raise ValueError("cross_sigungu_exposure cannot exceed cross_admin_exposure")
    if not np.allclose(
        numeric["own_admin_exposure"] + numeric["cross_admin_exposure"],
        numeric["raw_elderly_exposure"],
        rtol=1e-9,
        atol=1e-8,
    ):
        raise ValueError(
            "own_admin_exposure + cross_admin_exposure must equal raw_elderly_exposure"
        )
    tier = pd.to_numeric(frame["pareto_tier"], errors="coerce")
    rank = pd.to_numeric(frame["shortlist_rank"], errors="coerce")
    if tier.isna().any() or rank.isna().any() or (tier < 1).any() or (rank < 1).any():
        raise ValueError("pareto_tier and shortlist_rank must be positive")
    frame["field_validation_required"] = frame["field_validation_required"].map(
        _coerce_bool
    )
    for fallback_column in ("fallback_venue_1", "fallback_venue_2"):
        self_reference = frame[fallback_column].astype("string").eq(
            frame["venue_id"].astype("string")
        )
        if self_reference.fillna(False).any():
            raise ValueError(f"{fallback_column} cannot reference the same venue")
    validate_temporal_abstention(
        frame,
        require_official_abstention=require_official_temporal_abstention,
    )
    return frame.loc[:, [*INTERFACE_REQUIRED_COLUMNS, *(
        column for column in frame.columns if column not in INTERFACE_REQUIRED_COLUMNS
    )]].reset_index(drop=True)


def build_stage3_to_stage4_interface(
    venue_bundle_rows: pd.DataFrame,
    temporal: pd.DataFrame | None = None,
    *,
    bundle_col: str = "bundle_id",
    allow_extra_columns: bool = True,
    require_official_temporal_abstention: bool = True,
) -> pd.DataFrame:
    """Attach a one-row-per-bundle frozen temporal contract and validate it."""

    frame = _require_frame(venue_bundle_rows, "Stage 3 venue-bundle rows")
    temporal_columns = (
        "temporal_release_type",
        "temporal_confidence",
        "recommended_season",
        "fallback_season",
    )
    if temporal is not None:
        time = _require_frame(temporal, "Stage 2B temporal interface")
        _require_unique(time, [bundle_col], "Stage 2B temporal interface")
        _require_columns(time, [bundle_col, *temporal_columns], "Stage 2B temporal interface")
        overlap = [column for column in temporal_columns if column in frame.columns]
        if overlap:
            frame = frame.drop(columns=overlap)
        frame = frame.merge(
            time.loc[:, [bundle_col, *temporal_columns]],
            on=bundle_col,
            how="left",
            validate="many_to_one",
        )
    return validate_stage3_to_stage4_interface(
        frame,
        allow_extra_columns=allow_extra_columns,
        require_official_temporal_abstention=require_official_temporal_abstention,
    )


__all__ = [
    "FIELD_VALIDATION_FIELDS",
    "INTERFACE_REQUIRED_COLUMNS",
    "STAGE4_FORBIDDEN_COLUMNS",
    "TEMPORAL_SELECTOR_FORBIDDEN_COLUMNS",
    "ABSTENTION_RELEASE",
    "AdminShortlistResult",
    "SpearmanAuditResult",
    "SensitivityResult",
    "build_venue_feasibility_context",
    "compute_pareto_tiers",
    "build_admin_shortlists",
    "build_admin_shortlist",
    "build_coverage_cluster_fallbacks",
    "collapse_admin_venues",
    "build_need_exposure_quadrants",
    "stage3_spearman_audit",
    "compute_sensitivity_metrics",
    "build_sensitivity_metrics",
    "build_field_validation_queue",
    "validate_temporal_abstention",
    "validate_stage3_to_stage4_interface",
    "build_stage3_to_stage4_interface",
]
