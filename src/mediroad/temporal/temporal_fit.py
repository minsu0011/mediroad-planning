"""Region x recommended-service-bundle x month Stage 2B fit.

Only evidence-backed numeric components enter the denominator.  Operator
calendar, future-service overlap, and team/vehicle/equipment availability are
explicit unknown status strings until supplied; they are never neutralized to
zero or silently treated as feasible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from .audit import assert_complete_bounded_scores, assert_unknown_not_zero


KEY_COLUMNS = ["admin_dong_code", "bundle_id", "month"]
DEFAULT_SOURCE_COMPONENTS = {
    "nhis_observed_utilization_seasonality": ("seasonal_service_fit",),
    "era5_land_operational_climate": ("climate_operational_fit",),
    "gtfs_2024_weekday_baseline": ("transit_time_fit",),
}

SEASON_MONTHS = {
    "winter": (12, 1, 2),
    "spring": (3, 4, 5),
    "summer": (6, 7, 8),
    "autumn": (9, 10, 11),
}
SEASON_ORDER = {name: index for index, name in enumerate(SEASON_MONTHS, start=1)}
HISTORICAL_CLIMATE_ROLE = "historical_operational_risk_advisory_not_primary_selector"
UNAVAILABLE_REPLAN_TRIGGER_STATUS = (
    "unavailable_requires_date_specific_forecast_and_operator_service_date"
)


@dataclass(frozen=True)
class TemporalAblationResult:
    summary: pd.DataFrame
    score_variants: pd.DataFrame


def _fit_section(config: Mapping[str, Any]) -> dict[str, Any]:
    value = config.get("temporal_fit", config)
    if not isinstance(value, Mapping):
        raise TypeError("temporal_fit config must be a mapping")
    return dict(value)


def _normalised_component_weights(config: Mapping[str, Any]) -> dict[str, float]:
    section = _fit_section(config)
    weights = {key: float(value) for key, value in section.get("component_weights", {}).items()}
    if not weights or any(value <= 0 for value in weights.values()):
        raise ValueError("Temporal fit requires positive component weights")
    total = sum(weights.values())
    if not np.isclose(total, 1.0, atol=1e-9):
        raise ValueError(f"Temporal component weights sum to {total}, expected 1")
    return weights


def _validate_unique(frame: pd.DataFrame, keys: list[str], label: str) -> None:
    missing = sorted(set(keys) - set(frame.columns))
    if missing:
        raise KeyError(f"{label} lacks key columns: {missing}")
    if frame.duplicated(keys).any():
        raise ValueError(f"{label} contains duplicate keys {keys}")


def build_region_bundle_month_fit(
    regions: pd.DataFrame,
    bundle_seasonality: pd.DataFrame,
    monthly_climate: pd.DataFrame,
    transit_fit: pd.DataFrame | pd.Series,
    config: Mapping[str, Any],
    *,
    enforce_config_contract: bool = False,
    planning_resolution_override: str | None = None,
) -> pd.DataFrame:
    """Build the complete region x bundle x month fit panel."""

    identity = ["admin_dong_code", "admin_dong_name", "policy_sigungu_name"]
    missing_identity = sorted(set(identity) - set(regions.columns))
    if missing_identity:
        raise KeyError(f"Region input lacks identity columns: {missing_identity}")
    region_frame = regions[identity].copy()
    region_frame["admin_dong_code"] = region_frame["admin_dong_code"].astype(str)
    _validate_unique(region_frame, ["admin_dong_code"], "Region input")

    season_required = [
        "bundle_id",
        "bundle_name_ko",
        "month",
        "seasonal_service_fit",
        "seasonality_modifier",
        "seasonality_reliability",
        "bundle_month_compatible",
    ]
    missing = sorted(set(season_required) - set(bundle_seasonality.columns))
    if missing:
        raise KeyError(f"Bundle seasonality lacks columns: {missing}")
    season = bundle_seasonality[season_required].copy()
    season["month"] = pd.to_numeric(season["month"], errors="raise").astype(int)
    _validate_unique(season, ["bundle_id", "month"], "Bundle seasonality")
    if not season["month"].between(1, 12).all():
        raise ValueError("Bundle seasonality months must be 1..12")
    bundle_month_count = season.groupby("bundle_id")["month"].nunique()
    if not bundle_month_count.eq(12).all():
        raise ValueError("Every service bundle requires all 12 months")

    climate_columns = [
        "admin_dong_code",
        "month",
        "climate_operational_fit",
        "climate_operational_risk",
    ]
    missing = sorted(set(climate_columns) - set(monthly_climate.columns))
    if missing:
        raise KeyError(f"Monthly climate input lacks columns: {missing}")
    climate = monthly_climate[climate_columns].copy()
    climate["admin_dong_code"] = climate["admin_dong_code"].astype(str)
    _validate_unique(climate, ["admin_dong_code", "month"], "Monthly climate")

    if isinstance(transit_fit, pd.Series):
        transit = transit_fit.rename("transit_time_fit").reset_index()
        transit = transit.rename(columns={transit.columns[0]: "admin_dong_code"})
    else:
        transit = transit_fit.copy()
    if not {"admin_dong_code", "transit_time_fit"}.issubset(transit.columns):
        raise KeyError("Transit fit requires admin_dong_code and transit_time_fit")
    transit = transit[["admin_dong_code", "transit_time_fit"]].copy()
    transit["admin_dong_code"] = transit["admin_dong_code"].astype(str)
    _validate_unique(transit, ["admin_dong_code"], "Transit fit")

    output = region_frame.merge(season, how="cross")
    output = output.merge(
        climate,
        on=["admin_dong_code", "month"],
        how="left",
        validate="many_to_one",
    ).merge(transit, on="admin_dong_code", how="left", validate="many_to_one")
    weights = _normalised_component_weights(config)
    assert_complete_bounded_scores(output, weights)
    output["temporal_fit_score"] = 0.0
    for component, weight in weights.items():
        contribution = f"contribution__{component}"
        output[contribution] = output[component] * weight
        output["temporal_fit_score"] += output[contribution]
    output["temporal_fit_score"] = output["temporal_fit_score"].clip(0.0, 100.0)

    section = _fit_section(config)
    unavailable = dict(section.get("unavailable_components", {}))
    for component, status in unavailable.items():
        output[component] = str(status)
    assert_unknown_not_zero(output, unavailable)
    output["exact_date_status"] = str(
        config.get("exact_date_status", "unavailable_requires_operator_calendar")
    )
    planning_resolution = str(
        planning_resolution_override or config.get("planning_resolution", "month")
    )
    if planning_resolution not in {"month", "season", "quarter"}:
        raise ValueError(f"Unsupported planning resolution: {planning_resolution}")
    output["planning_resolution"] = planning_resolution
    output["temporal_fit_interpretation"] = str(
        config.get(
            "target_definition",
            "relative_monthly_service_and_operational_fit_not_demand_prediction",
        )
    )
    _validate_unique(output, KEY_COLUMNS, "Temporal fit output")
    expected_rows = len(region_frame) * season["bundle_id"].nunique() * 12
    if len(output) != expected_rows:
        raise ValueError(f"Temporal fit has {len(output)} rows, expected {expected_rows}")
    if enforce_config_contract:
        configured = int(config.get("quality_gates", {}).get("expected_region_bundle_month_rows", expected_rows))
        if len(output) != configured:
            raise ValueError(f"Temporal fit rows {len(output)} != configured {configured}")
    assert_complete_bounded_scores(
        output,
        [*weights, "temporal_fit_score", "climate_operational_risk"],
    )
    return output.sort_values(KEY_COLUMNS, kind="mergesort").reset_index(drop=True)


def _quarter(month: int) -> int:
    return (int(month) - 1) // 3 + 1


def select_primary_fallback_months(
    temporal_fit: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    enforce_config_contract: bool = False,
) -> pd.DataFrame:
    """Select deterministic primary and different-quarter fallback months."""

    _validate_unique(temporal_fit, KEY_COLUMNS, "Temporal fit input")
    required = {*KEY_COLUMNS, "temporal_fit_score"}
    missing = sorted(required - set(temporal_fit.columns))
    if missing:
        raise KeyError(f"Temporal fit recommendation input lacks columns: {missing}")
    identity = [
        column
        for column in ("admin_dong_name", "policy_sigungu_name", "bundle_name_ko")
        if column in temporal_fit.columns
    ]
    rows: list[dict[str, Any]] = []
    for (admin_code, bundle_id), group in temporal_fit.groupby(
        ["admin_dong_code", "bundle_id"], sort=True
    ):
        if set(group["month"].astype(int)) != set(range(1, 13)):
            raise ValueError(f"Recommendation group {(admin_code, bundle_id)} lacks 12 months")
        ordered = group.sort_values(
            ["temporal_fit_score", "month"],
            ascending=[False, True],
            kind="mergesort",
        )
        primary = ordered.iloc[0]
        primary_month = int(primary["month"])
        different_quarter = ordered.loc[
            ordered["month"].astype(int).map(_quarter).ne(_quarter(primary_month))
        ]
        fallback = (different_quarter if not different_quarter.empty else ordered.iloc[1:]).iloc[0]
        row: dict[str, Any] = {
            "admin_dong_code": str(admin_code),
            "bundle_id": str(bundle_id),
            "primary_month": primary_month,
            "primary_quarter": _quarter(primary_month),
            "primary_temporal_fit_score": float(primary["temporal_fit_score"]),
            "fallback_month": int(fallback["month"]),
            "fallback_quarter": _quarter(int(fallback["month"])),
            "fallback_temporal_fit_score": float(fallback["temporal_fit_score"]),
            "fallback_distinct_from_primary": int(primary_month != int(fallback["month"])),
            "fallback_different_quarter": int(
                _quarter(primary_month) != _quarter(int(fallback["month"]))
            ),
            "fallback_rule": _fit_section(config).get(
                "fallback_rule",
                "highest_scoring_month_outside_primary_month_and_prefer_different_quarter",
            ),
            "exact_date_status": config.get(
                "exact_date_status", "unavailable_requires_operator_calendar"
            ),
        }
        for column in identity:
            row[column] = primary[column]
        rows.append(row)
    output = pd.DataFrame(rows)
    if not output["fallback_distinct_from_primary"].eq(1).all():
        raise ValueError("A fallback month equals its primary month")
    if enforce_config_contract:
        expected = int(config.get("quality_gates", {}).get("expected_recommendation_rows", len(output)))
        if len(output) != expected:
            raise ValueError(f"Recommendation rows {len(output)} != configured {expected}")
    return output.sort_values(["admin_dong_code", "bundle_id"]).reset_index(drop=True)


def aggregate_monthly_fit_to_season(temporal_fit: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the 12-month diagnostic substrate into four release seasons."""

    _validate_unique(temporal_fit, KEY_COLUMNS, "Temporal fit input")
    if not temporal_fit["month"].between(1, 12).all():
        raise ValueError("Temporal fit months must be 1..12")
    month_to_season = {
        month: season for season, months in SEASON_MONTHS.items() for month in months
    }
    frame = temporal_fit.copy()
    frame["season"] = frame["month"].map(month_to_season)
    frame["season_order"] = frame["season"].map(SEASON_ORDER)
    group_keys = ["admin_dong_code", "bundle_id", "season", "season_order"]
    identity = [
        column
        for column in ("admin_dong_name", "policy_sigungu_name", "bundle_name_ko")
        if column in frame.columns
    ]
    numeric_candidates = [
        column
        for column in (
            "temporal_fit_score",
            "seasonal_service_fit",
            "seasonality_modifier",
            "seasonality_reliability",
            "climate_operational_fit",
            "climate_operational_risk",
            "transit_time_fit",
        )
        if column in frame.columns
    ] + [column for column in frame.columns if column.startswith("contribution__")]
    aggregations: dict[str, tuple[str, str]] = {
        **{column: (column, "first") for column in identity},
        **{column: (column, "mean") for column in numeric_candidates},
        "month_count": ("month", "nunique"),
    }
    seasonal = frame.groupby(group_keys, as_index=False, sort=True).agg(**aggregations)
    if not seasonal["month_count"].eq(3).all():
        raise ValueError("Every released season must contain exactly three diagnostic months")
    seasonal["season_months"] = seasonal["season"].map(
        {name: "|".join(str(month) for month in months) for name, months in SEASON_MONTHS.items()}
    )
    seasonal["planning_resolution"] = "season"
    seasonal["monthly_output_status"] = "diagnostic_appendix_only_monthly_gate_failed"
    expected = temporal_fit.groupby(["admin_dong_code", "bundle_id"]).ngroups * 4
    if len(seasonal) != expected:
        raise ValueError(f"Seasonal fit has {len(seasonal)} rows, expected {expected}")
    return seasonal.sort_values(
        ["admin_dong_code", "bundle_id", "season_order"], kind="mergesort"
    ).reset_index(drop=True)


def select_primary_fallback_seasons(
    temporal_fit: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    enforce_config_contract: bool = False,
) -> pd.DataFrame:
    """Release season windows while guaranteeing exact month/date are null.

    Use this selector whenever the preregistered monthly-resolution gate fails.
    The 12-month scores remain available only as a diagnostic appendix.
    """

    seasonal = aggregate_monthly_fit_to_season(temporal_fit)
    identity = [
        column
        for column in ("admin_dong_name", "policy_sigungu_name", "bundle_name_ko")
        if column in seasonal.columns
    ]
    rows: list[dict[str, Any]] = []
    for (admin_code, bundle_id), group in seasonal.groupby(
        ["admin_dong_code", "bundle_id"], sort=True
    ):
        ordered = group.sort_values(
            ["temporal_fit_score", "season_order"],
            ascending=[False, True],
            kind="mergesort",
        )
        primary, fallback = ordered.iloc[0], ordered.iloc[1]
        row: dict[str, Any] = {
            "admin_dong_code": str(admin_code),
            "bundle_id": str(bundle_id),
            "planning_resolution": "season",
            "primary_season": str(primary["season"]),
            "primary_season_months": str(primary["season_months"]),
            "primary_temporal_fit_score": float(primary["temporal_fit_score"]),
            "fallback_season": str(fallback["season"]),
            "fallback_season_months": str(fallback["season_months"]),
            "fallback_temporal_fit_score": float(fallback["temporal_fit_score"]),
            "fallback_distinct_from_primary": int(primary["season"] != fallback["season"]),
            "primary_month": None,
            "fallback_month": None,
            "recommended_date": None,
            "fallback_date": None,
            "exact_month_status": "not_released_monthly_resolution_gate_failed",
            "exact_date_status": config.get(
                "exact_date_status", "unavailable_requires_operator_calendar"
            ),
            "monthly_output_status": "diagnostic_appendix_only_monthly_gate_failed",
        }
        for column in identity:
            row[column] = primary[column]
        rows.append(row)
    output = pd.DataFrame(rows)
    if not output["fallback_distinct_from_primary"].eq(1).all():
        raise ValueError("A fallback season equals its primary season")
    if output[["primary_month", "fallback_month", "recommended_date", "fallback_date"]].notna().any().any():
        raise ValueError("Season release must not contain exact month/date values")
    if enforce_config_contract:
        expected = int(config.get("quality_gates", {}).get("expected_recommendation_rows", len(output)))
        if len(output) != expected:
            raise ValueError(f"Season recommendation rows {len(output)} != configured {expected}")
    return output.sort_values(["admin_dong_code", "bundle_id"]).reset_index(drop=True)


def apply_season_release_contract(
    recommendations: pd.DataFrame,
    temporal_fit: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    enforce_config_contract: bool = False,
) -> pd.DataFrame:
    """Apply actionability and operational-evidence rules to season choices.

    The selector is deliberately deterministic even for tied or flat inputs so
    diagnostics are reproducible.  This release guard is the separate step
    that prevents those diagnostic tie-breaks from becoming unsupported policy
    recommendations.  A bundle must have positive reliability and a non-flat
    *season-level* released service-fit profile.  Historical reanalysis may be
    attached as an advisory, but it cannot imply that a date-specific weather
    or operator-calendar replan trigger is available.
    """

    keys = ["admin_dong_code", "bundle_id"]
    _validate_unique(recommendations, keys, "Season recommendations")
    required = {
        *keys,
        "primary_season",
        "fallback_season",
        "primary_temporal_fit_score",
        "fallback_temporal_fit_score",
    }
    missing = sorted(required - set(recommendations.columns))
    if missing:
        raise KeyError(f"Season recommendations lack columns: {missing}")
    weights = _normalised_component_weights(config)
    disallowed_primary = sorted(
        set(weights).intersection({"climate_operational_fit", "transit_time_fit"})
    )
    if disallowed_primary:
        raise ValueError(
            "Historical climate and static transit cannot be primary season selectors: "
            f"{disallowed_primary}"
        )

    seasonal = aggregate_monthly_fit_to_season(temporal_fit)
    evidence = seasonal.groupby(keys, as_index=False, sort=True).agg(
        bundle_season_fit_min=("seasonal_service_fit", "min"),
        bundle_season_fit_max=("seasonal_service_fit", "max"),
        bundle_seasonality_reliability=("seasonality_reliability", "mean"),
        decision_score_min=("temporal_fit_score", "min"),
        decision_score_max=("temporal_fit_score", "max"),
    )
    recommendation_frame = recommendations.copy()
    for key in keys:
        recommendation_frame[key] = recommendation_frame[key].astype(str)
        evidence[key] = evidence[key].astype(str)
    recommendation_keys = set(map(tuple, recommendation_frame[keys].to_numpy()))
    evidence_keys = set(map(tuple, evidence[keys].to_numpy()))
    if recommendation_keys != evidence_keys:
        raise ValueError("Season recommendations and temporal-fit groups do not match")
    evidence["bundle_season_fit_range"] = (
        evidence["bundle_season_fit_max"] - evidence["bundle_season_fit_min"]
    )
    evidence["decision_score_range"] = (
        evidence["decision_score_max"] - evidence["decision_score_min"]
    )
    minimum = float(
        config.get("quality_gates", {}).get(
            "minimum_bundle_season_fit_range_for_release", 1e-9
        )
    )
    evidence["temporal_signal_available"] = (
        evidence["bundle_season_fit_range"].gt(minimum)
        & evidence["bundle_seasonality_reliability"].gt(0)
    )

    output = recommendation_frame.drop(
        columns=[column for column in evidence.columns if column in recommendations.columns and column not in keys]
    ).merge(evidence, on=keys, how="left", validate="one_to_one")
    if output["temporal_signal_available"].isna().any():
        raise ValueError("Season recommendations and temporal-fit groups do not match")

    output["primary_fallback_score_margin"] = (
        output["primary_temporal_fit_score"] - output["fallback_temporal_fit_score"]
    )
    tie_tolerance = float(
        config.get("bundle_season_reliability", {}).get("comparison_tolerance", 1e-12)
    )
    tie_rows: list[dict[str, Any]] = []
    for (admin_code, bundle_id), group in seasonal.groupby(keys, sort=True):
        maximum = float(group["temporal_fit_score"].max())
        tie_rows.append(
            {
                "admin_dong_code": str(admin_code),
                "bundle_id": str(bundle_id),
                "primary_season_tie_count": int(
                    np.isclose(
                        group["temporal_fit_score"].to_numpy(float),
                        maximum,
                        rtol=0.0,
                        atol=tie_tolerance,
                    ).sum()
                ),
            }
        )
    output = output.merge(pd.DataFrame(tie_rows), on=keys, how="left", validate="one_to_one")
    output["primary_selection_rule"] = (
        "temporal_fit_score_desc_then_fixed_winter_spring_summer_autumn_order"
    )
    output["temporal_evidence_scope"] = "province_common_by_service_bundle"
    output["region_specific_clinical_seasonality_used"] = 0
    output["climate_role"] = HISTORICAL_CLIMATE_ROLE
    output["historical_climate_advisory_status"] = (
        "available_historical_reanalysis_not_date_specific_forecast"
        if "climate_operational_risk" in temporal_fit.columns
        else "unavailable"
    )
    output["replan_trigger_status"] = str(
        _fit_section(config).get(
            "replan_trigger_status", UNAVAILABLE_REPLAN_TRIGGER_STATUS
        )
    )
    output["date_specific_replan_trigger_available"] = 0
    output["transit_role"] = "static_context_not_temporal_selector"
    output["release_status"] = "provisional_macro_season_prior"

    unavailable = ~output["temporal_signal_available"].astype(bool)
    release_columns = [
        "primary_season",
        "primary_season_months",
        "primary_temporal_fit_score",
        "fallback_season",
        "fallback_season_months",
        "fallback_temporal_fit_score",
        "primary_fallback_score_margin",
        "primary_climate_risk_advisory",
        "primary_climate_fit_advisory",
        "fallback_climate_risk_advisory",
    ]
    output.loc[unavailable, [column for column in release_columns if column in output.columns]] = None
    if "fallback_distinct_from_primary" in output.columns:
        output["fallback_distinct_from_primary"] = output[
            "fallback_distinct_from_primary"
        ].astype("Int64")
        output.loc[unavailable, "fallback_distinct_from_primary"] = pd.NA
    output.loc[unavailable, "release_status"] = (
        "unavailable_no_reliable_bundle_season_signal"
    )

    exact_fields = [
        column
        for column in ("primary_month", "fallback_month", "recommended_date", "fallback_date")
        if column in output.columns
    ]
    if exact_fields and output[exact_fields].notna().any().any():
        raise ValueError("Season release must not contain exact month/date values")
    if not output["date_specific_replan_trigger_available"].eq(0).all():
        raise ValueError("A date-specific replan trigger was fabricated")

    if enforce_config_contract:
        gates = dict(config.get("quality_gates", {}))
        expected_rows = int(gates.get("expected_recommendation_rows", len(output)))
        actionable = int(output["temporal_signal_available"].sum())
        no_signal = int((~output["temporal_signal_available"]).sum())
        expected_actionable = int(
            gates.get("expected_actionable_recommendation_rows", actionable)
        )
        expected_no_signal = int(
            gates.get("expected_no_signal_recommendation_rows", no_signal)
        )
        expected_triggers = int(
            gates.get(
                "expected_available_date_specific_replan_trigger_rows",
                int(output["date_specific_replan_trigger_available"].sum()),
            )
        )
        observed = {
            "recommendation": len(output),
            "actionable": actionable,
            "no_signal": no_signal,
            "available_date_specific_replan_trigger": int(
                output["date_specific_replan_trigger_available"].sum()
            ),
        }
        expected = {
            "recommendation": expected_rows,
            "actionable": expected_actionable,
            "no_signal": expected_no_signal,
            "available_date_specific_replan_trigger": expected_triggers,
        }
        if observed != expected:
            raise ValueError(f"Season release contract {observed} != configured {expected}")
    return output.sort_values(keys, kind="mergesort").reset_index(drop=True)


def _safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    left_values = np.asarray(left, dtype=float)
    right_values = np.asarray(right, dtype=float)
    if np.ptp(left_values) <= 1e-15 or np.ptp(right_values) <= 1e-15:
        return 1.0 if np.allclose(left_values, right_values) else 0.0
    left_series, right_series = pd.Series(left_values), pd.Series(right_values)
    value = float(left_series.corr(right_series, method="spearman"))
    if np.isfinite(value):
        return value
    return 1.0 if np.allclose(left, right) else 0.0


def _top3_by_group(frame: pd.DataFrame, score_column: str) -> dict[tuple[str, str], set[int]]:
    output: dict[tuple[str, str], set[int]] = {}
    for key, group in frame.groupby(["admin_dong_code", "bundle_id"], sort=False):
        top = group.sort_values(
            [score_column, "month"], ascending=[False, True], kind="mergesort"
        ).head(3)
        output[(str(key[0]), str(key[1]))] = set(top["month"].astype(int))
    return output


def run_temporal_ablation(
    temporal_fit: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    source_components: Mapping[str, Iterable[str]] | None = None,
) -> TemporalAblationResult:
    """Remove every numeric component and source, renormalizing remaining weights."""

    _validate_unique(temporal_fit, KEY_COLUMNS, "Temporal ablation input")
    weights = _normalised_component_weights(config)
    assert_complete_bounded_scores(temporal_fit, [*weights, "temporal_fit_score"])
    source_map = {
        key: tuple(value)
        for key, value in (source_components or DEFAULT_SOURCE_COMPONENTS).items()
    }
    for source, components in source_map.items():
        unknown = sorted(set(components) - set(weights))
        if unknown:
            raise ValueError(f"Ablation source {source} references unknown components: {unknown}")

    experiments: list[tuple[str, str, set[str]]] = [
        ("component", component, {component}) for component in weights
    ] + [("source", source, set(components)) for source, components in source_map.items()]
    baseline_values = temporal_fit["temporal_fit_score"].to_numpy(float)
    baseline_top3 = _top3_by_group(temporal_fit, "temporal_fit_score")
    baseline_primary = {
        key: min(
            group.loc[group["temporal_fit_score"].eq(group["temporal_fit_score"].max()), "month"]
        )
        for key, group in temporal_fit.groupby(["admin_dong_code", "bundle_id"], sort=False)
    }
    summaries: list[dict[str, Any]] = []
    variants: list[pd.DataFrame] = []
    for ablation_type, variant, removed in experiments:
        remaining = {key: value for key, value in weights.items() if key not in removed}
        total = sum(remaining.values())
        destructive_no_supported_component = not remaining
        if destructive_no_supported_component:
            # A single evidence-backed temporal component is a legitimate
            # fail-closed release.  Its removal has no estimable replacement;
            # encode a neutral 50-point stress vector rather than inventing a
            # second source or silently skipping exhaustive coverage.
            variant_values = np.full(len(temporal_fit), 50.0, dtype=float)
        else:
            variant_values = sum(
                temporal_fit[component].to_numpy(float) * (weight / total)
                for component, weight in remaining.items()
            )
        scored = temporal_fit[KEY_COLUMNS].copy()
        scored["variant_score"] = variant_values
        variant_top3 = _top3_by_group(scored, "variant_score")
        recalls = [len(baseline_top3[key] & variant_top3[key]) / 3.0 for key in baseline_top3]
        jaccards = [
            len(baseline_top3[key] & variant_top3[key])
            / len(baseline_top3[key] | variant_top3[key])
            for key in baseline_top3
        ]
        variant_primary = {
            key: min(group.loc[group["variant_score"].eq(group["variant_score"].max()), "month"])
            for key, group in scored.groupby(["admin_dong_code", "bundle_id"], sort=False)
        }
        summaries.append(
            {
                "ablation_type": ablation_type,
                "variant": variant,
                "removed_components": "|".join(sorted(removed)),
                "remaining_weight_before_renormalization": total,
                "destructive_no_supported_component": destructive_no_supported_component,
                "spearman": _safe_spearman(baseline_values, variant_values),
                "median_abs_score_shift": float(np.median(np.abs(variant_values - baseline_values))),
                "top3_recall": float(np.mean(recalls)),
                "top3_jaccard": float(np.mean(jaccards)),
                "primary_month_change_fraction": float(
                    np.mean([baseline_primary[key] != variant_primary[key] for key in baseline_primary])
                ),
            }
        )
        scored.insert(0, "variant", variant)
        scored.insert(0, "ablation_type", ablation_type)
        variants.append(scored)
    summary = pd.DataFrame(summaries).sort_values(["ablation_type", "variant"]).reset_index(drop=True)
    score_variants = pd.concat(variants, ignore_index=True)
    if set(summary.loc[summary["ablation_type"].eq("component"), "variant"]) != set(weights):
        raise RuntimeError("Temporal component ablation coverage is incomplete")
    if set(summary.loc[summary["ablation_type"].eq("source"), "variant"]) != set(source_map):
        raise RuntimeError("Temporal source ablation coverage is incomplete")
    return TemporalAblationResult(summary, score_variants)
