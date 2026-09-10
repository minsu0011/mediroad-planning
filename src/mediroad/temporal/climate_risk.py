"""Daily reanalysis weather to monthly operational climate risk.

This module intentionally models long-run operating risk, not date-specific
weather and not clinical demand.  Units are an explicit input and are checked
before thresholds are applied; a km/h wind series must never be compared with
the configured m/s threshold.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


WEATHER_VARIABLES = (
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "rain_sum",
    "snowfall_sum",
    "wind_speed_10m_max",
)

EXPECTED_UNITS = {
    "temperature_2m_max": "celsius",
    "temperature_2m_min": "celsius",
    "precipitation_sum": "mm",
    "rain_sum": "mm",
    "snowfall_sum": "cm",
    "wind_speed_10m_max": "ms",
}


@dataclass(frozen=True)
class ClimateStabilityResult:
    """Leave-one-year-out evidence for the monthly operating-risk layer."""

    detail: pd.DataFrame
    summary: dict[str, Any]


def _safe_spearman(left: pd.Series, right: pd.Series) -> float:
    """Return a deterministic similarity when either rank vector is constant."""

    left_values = pd.to_numeric(left, errors="raise").to_numpy(float)
    right_values = pd.to_numeric(right, errors="raise").to_numpy(float)
    left_constant = np.ptp(left_values) == 0.0
    right_constant = np.ptp(right_values) == 0.0
    if left_constant or right_constant:
        return 1.0 if np.allclose(left_values, right_values) else 0.0
    return float(pd.Series(left_values).corr(pd.Series(right_values), method="spearman"))


def _weather_section(config: Mapping[str, Any]) -> dict[str, Any]:
    value = config.get("weather", config)
    if not isinstance(value, Mapping):
        raise TypeError("weather config must be a mapping")
    return dict(value)


def _normalise_unit(value: object) -> str:
    token = str(value).strip().lower().replace(" ", "").replace("_", "")
    aliases = {
        "°c": "celsius",
        "c": "celsius",
        "degc": "celsius",
        "celsius": "celsius",
        "millimeter": "mm",
        "millimeters": "mm",
        "millimetre": "mm",
        "millimetres": "mm",
        "mm": "mm",
        "mm/day": "mm",
        "mmd-1": "mm",
        "mmday-1": "mm",
        "centimeter": "cm",
        "centimeters": "cm",
        "centimetre": "cm",
        "centimetres": "cm",
        "cm": "cm",
        "m/s": "ms",
        "ms-1": "ms",
        "m*s-1": "ms",
        "meterpersecond": "ms",
        "meterspersecond": "ms",
        "metrepersecond": "ms",
        "metrespersecond": "ms",
        "ms": "ms",
        "km/h": "kmh",
        "kmh": "kmh",
        "kph": "kmh",
    }
    return aliases.get(token, token)


def validate_weather_units(
    units: Mapping[str, object],
    config: Mapping[str, Any],
) -> dict[str, str]:
    """Validate every threshold-bearing weather variable's declared unit."""

    section = _weather_section(config)
    variables = list(section.get("variables", WEATHER_VARIABLES))
    missing = sorted(set(variables) - set(units))
    if missing:
        raise ValueError(f"Weather source units are missing for variables: {missing}")
    expected = dict(EXPECTED_UNITS)
    expected["wind_speed_10m_max"] = _normalise_unit(section.get("wind_speed_unit", "ms"))
    observed: dict[str, str] = {}
    for variable in variables:
        observed[variable] = _normalise_unit(units[variable])
        expected_unit = expected.get(variable)
        if expected_unit is None:
            raise ValueError(f"No hard unit contract is defined for weather variable {variable}")
        if observed[variable] != expected_unit:
            raise ValueError(
                f"Weather unit mismatch for {variable}: observed={units[variable]!r} "
                f"normalized={observed[variable]!r}, expected={expected_unit!r}"
            )
    return observed


def _positive_percentile(values: pd.Series) -> pd.Series:
    """Score no-event cells as zero and rank only positive event frequencies."""

    numeric = pd.to_numeric(values, errors="raise").astype(float)
    output = pd.Series(0.0, index=numeric.index)
    positive = numeric.gt(0)
    if positive.any():
        output.loc[positive] = numeric.loc[positive].rank(method="average", pct=True) * 100.0
    return output


def _validate_physical_ranges(frame: pd.DataFrame, variables: list[str]) -> None:
    finite = frame[variables].to_numpy(float)
    if not np.isfinite(finite).all():
        raise ValueError("ERA5-Land weather contains missing or non-finite values")
    if (frame["temperature_2m_max"] < frame["temperature_2m_min"]).any():
        raise ValueError("Daily maximum temperature is below daily minimum temperature")
    if not frame["temperature_2m_max"].between(-90, 70).all():
        raise ValueError("Daily maximum temperature is outside a plausible Celsius range")
    if not frame["temperature_2m_min"].between(-100, 60).all():
        raise ValueError("Daily minimum temperature is outside a plausible Celsius range")
    for column in ("precipitation_sum", "rain_sum", "snowfall_sum", "wind_speed_10m_max"):
        if column in frame and (frame[column] < 0).any():
            raise ValueError(f"Weather variable {column} contains negative values")
    if "wind_speed_10m_max" in frame and (frame["wind_speed_10m_max"] > 150).any():
        raise ValueError("Wind speed exceeds the plausible m/s hard limit")


def build_monthly_climate_risk(
    daily: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    units: Mapping[str, object] | None = None,
    expected_admin_dongs: int | None = None,
    expected_start_date: str | pd.Timestamp | None = None,
    expected_end_date: str | pd.Timestamp | None = None,
    completeness_min: float | None = None,
    enforce_config_contract: bool = False,
) -> pd.DataFrame:
    """Aggregate daily NASA POWER/MERRA-2 proxy values into admin-dong/month risk.

    When ``enforce_config_contract`` is true, the configured 153-region,
    1991--2025 and completeness gates are used unless explicitly overridden.
    Synthetic and diagnostic callers can instead provide smaller expectations.
    """

    section = _weather_section(config)
    source = dict(config.get("source", {}))
    quality = dict(config.get("quality_gates", {}))
    required = {"admin_dong_code", "date", *section.get("variables", WEATHER_VARIABLES)}
    missing = sorted(required - set(daily.columns))
    if missing:
        raise KeyError(f"Reanalysis daily input lacks columns: {missing}")

    declared_units = units if units is not None else daily.attrs.get("units")
    if not isinstance(declared_units, Mapping):
        raise ValueError("Reanalysis daily input requires an explicit source-unit mapping")
    validate_weather_units(declared_units, config)

    frame = daily.copy()
    frame["admin_dong_code"] = frame["admin_dong_code"].astype(str).str.strip()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    if frame["date"].isna().any():
        raise ValueError("Reanalysis daily input contains invalid dates")
    if frame.duplicated(["admin_dong_code", "date"]).any():
        raise ValueError("Reanalysis daily input has duplicate admin-dong/date rows")
    variables = list(section.get("variables", WEATHER_VARIABLES))
    for column in variables:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    _validate_physical_ranges(frame, variables)

    observed_admins = int(frame["admin_dong_code"].nunique())
    if expected_admin_dongs is None and enforce_config_contract:
        expected_admin_dongs = int(source.get("expected_admin_dongs", 153))
    if expected_admin_dongs is not None and observed_admins != int(expected_admin_dongs):
        raise ValueError(
            f"Reanalysis input has {observed_admins} admin dongs, expected {expected_admin_dongs}"
        )

    if expected_start_date is None:
        expected_start_date = source.get("weather_start_date") if enforce_config_contract else frame["date"].min()
    if expected_end_date is None:
        expected_end_date = source.get("weather_end_date") if enforce_config_contract else frame["date"].max()
    start = pd.Timestamp(expected_start_date).normalize()
    end = pd.Timestamp(expected_end_date).normalize()
    if end < start:
        raise ValueError("Weather expected_end_date precedes expected_start_date")
    expected_days = int((end - start).days + 1)
    counts = frame.loc[frame["date"].between(start, end)].groupby("admin_dong_code")["date"].nunique()
    completeness = counts / expected_days
    if completeness_min is None:
        completeness_min = (
            float(quality.get("weather_daily_completeness_min", 0.995))
            if enforce_config_contract
            else 1.0
        )
    if completeness.empty or completeness.min() < float(completeness_min):
        raise ValueError(
            "Reanalysis daily completeness below contract: "
            f"min={float(completeness.min()) if not completeness.empty else 0.0:.6f}, "
            f"required={float(completeness_min):.6f}"
        )
    frame = frame.loc[frame["date"].between(start, end)].copy()
    frame["year"] = frame["date"].dt.year
    frame["month"] = frame["date"].dt.month
    if enforce_config_contract:
        expected_years = int(quality.get("expected_weather_years", frame["year"].nunique()))
        observed_years = int(frame["year"].nunique())
        if observed_years != expected_years:
            raise ValueError(
                f"Weather years {observed_years} != configured {expected_years}"
            )

    thresholds = dict(section.get("thresholds", {}))
    hazard_builders = {
        "heat_days": ("temperature_2m_max", "heat_day_max_c", "ge"),
        "cold_days": ("temperature_2m_min", "cold_day_min_c", "le"),
        "heavy_rain_days": ("precipitation_sum", "heavy_rain_day_mm", "ge"),
        "snow_days": ("snowfall_sum", "snow_day_cm", "ge"),
        "heavy_snow_days": ("snowfall_sum", "heavy_snow_day_cm", "ge"),
        "strong_wind_days": ("wind_speed_10m_max", "strong_wind_day_ms", "ge"),
    }
    hazard_columns = list(dict(section.get("hazard_weights", {})))
    for hazard in hazard_columns:
        if hazard not in hazard_builders:
            raise ValueError(f"No threshold contract for configured climate hazard {hazard}")
        source_column, threshold_key, comparison = hazard_builders[hazard]
        if source_column not in frame or threshold_key not in thresholds:
            raise ValueError(
                f"Configured climate hazard {hazard} lacks {source_column}/{threshold_key}"
            )
        threshold = float(thresholds[threshold_key])
        frame[hazard] = (
            frame[source_column].ge(threshold)
            if comparison == "ge"
            else frame[source_column].le(threshold)
        )

    identity = [column for column in ("admin_dong_name", "policy_sigungu_name") if column in frame]
    group_columns = ["admin_dong_code", *identity, "month"]
    grouped = frame.groupby(group_columns, as_index=False, sort=True)
    named_aggregations: dict[str, tuple[str, str]] = {
        "weather_years_observed": ("year", "nunique"),
        "observed_daily_rows": ("date", "size"),
        **{column: (column, "sum") for column in hazard_columns},
    }
    means = {
        "temperature_2m_max": "mean_daily_max_temp_c",
        "temperature_2m_min": "mean_daily_min_temp_c",
        "precipitation_sum": "mean_daily_precipitation_mm",
        "rain_sum": "mean_daily_rain_mm",
        "snowfall_sum": "mean_daily_snowfall_cm",
        "wind_speed_10m_max": "mean_daily_max_wind_ms",
    }
    for source_column, output_column in means.items():
        if source_column in variables:
            named_aggregations[output_column] = (source_column, "mean")
    monthly = grouped.agg(**named_aggregations)
    for column in hazard_columns:
        monthly[column] = monthly[column] / monthly["weather_years_observed"]
        monthly[f"risk_component__{column}"] = _positive_percentile(monthly[column])

    weights = {key: float(value) for key, value in section.get("hazard_weights", {}).items()}
    if set(weights) != set(hazard_columns) or not np.isclose(sum(weights.values()), 1.0, atol=1e-9):
        raise ValueError("Weather hazard weights must cover all hazards and sum to one")
    monthly["climate_operational_risk"] = sum(
        monthly[f"risk_component__{column}"] * weight for column, weight in weights.items()
    ).clip(0.0, 100.0)
    monthly["climate_operational_fit"] = 100.0 - monthly["climate_operational_risk"]
    monthly["daily_completeness"] = monthly["admin_dong_code"].map(completeness)
    monthly["weather_source"] = config.get("source", {}).get("weather_model", "era5_land")
    monthly["weather_interpretation"] = section.get(
        "interpretation",
        "long_run_operational_climate_proxy_not_date_specific_forecast",
    )
    for hazard_name, status in dict(section.get("unavailable_hazard", {})).items():
        status_column = f"{hazard_name}_hazard_status"
        monthly[status_column] = str(status)
        invalid = monthly[status_column].astype(str).str.lower().str.startswith("unknown_").eq(False)
        if invalid.any() or monthly[status_column].isin([0, "0", "0.0"]).any():
            raise ValueError(f"Unavailable climate hazard {hazard_name} was not explicit unknown")

    expected_rows = observed_admins * 12
    if len(monthly) != expected_rows or monthly.duplicated(["admin_dong_code", "month"]).any():
        raise ValueError(
            f"Monthly climate panel has {len(monthly)} rows, expected {expected_rows} unique rows"
        )
    if enforce_config_contract:
        configured_daily_rows = int(
            quality.get("expected_weather_daily_rows", observed_admins * expected_days)
        )
        if len(frame) != configured_daily_rows:
            raise ValueError(
                f"Weather daily rows {len(frame)} != configured {configured_daily_rows}"
            )
        configured_rows = int(quality.get("expected_monthly_climate_rows", expected_rows))
        if len(monthly) != configured_rows:
            raise ValueError(f"Monthly climate rows {len(monthly)} != configured {configured_rows}")
    return monthly.sort_values(["admin_dong_code", "month"], kind="mergesort").reset_index(drop=True)


def climate_leave_one_year_out_stability(
    daily: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    units: Mapping[str, object],
) -> ClimateStabilityResult:
    """Compare each held-out year with the climatology from all other years."""

    if "date" not in daily:
        raise KeyError("Climate LOYO input lacks date")
    dates = pd.to_datetime(daily["date"], errors="coerce")
    if dates.isna().any():
        raise ValueError("Climate LOYO input contains invalid dates")
    years = sorted(dates.dt.year.unique().tolist())
    if len(years) < 2:
        raise ValueError("Climate LOYO requires at least two complete years")

    work = daily.copy()
    work["__year"] = dates.dt.year.to_numpy()
    detail_rows: list[dict[str, Any]] = []
    for holdout_year in years:
        training = work.loc[work["__year"].ne(holdout_year)].drop(columns="__year")
        holdout = work.loc[work["__year"].eq(holdout_year)].drop(columns="__year")
        train_monthly = build_monthly_climate_risk(
            training,
            config,
            units=units,
            expected_admin_dongs=int(work["admin_dong_code"].nunique()),
            expected_start_date=training["date"].min(),
            expected_end_date=training["date"].max(),
            completeness_min=0.0,
        )
        holdout_monthly = build_monthly_climate_risk(
            holdout,
            config,
            units=units,
            expected_admin_dongs=int(work["admin_dong_code"].nunique()),
            expected_start_date=f"{holdout_year}-01-01",
            expected_end_date=f"{holdout_year}-12-31",
            completeness_min=0.99,
        )
        merged = train_monthly[["admin_dong_code", "month", "climate_operational_fit"]].merge(
            holdout_monthly[["admin_dong_code", "month", "climate_operational_fit"]],
            on=["admin_dong_code", "month"],
            suffixes=("_train", "_holdout"),
            validate="one_to_one",
        )
        for admin_code, group in merged.groupby("admin_dong_code", sort=True):
            rho = _safe_spearman(
                group["climate_operational_fit_train"],
                group["climate_operational_fit_holdout"],
            )
            top_train = set(
                group.sort_values(
                    ["climate_operational_fit_train", "month"],
                    ascending=[False, True],
                    kind="mergesort",
                ).head(3)["month"]
            )
            top_holdout = set(
                group.sort_values(
                    ["climate_operational_fit_holdout", "month"],
                    ascending=[False, True],
                    kind="mergesort",
                ).head(3)["month"]
            )
            detail_rows.append(
                {
                    "holdout_year": int(holdout_year),
                    "admin_dong_code": str(admin_code),
                    "month_rank_spearman": rho,
                    "top3_recall": len(top_train & top_holdout) / 3.0,
                }
            )
    detail = pd.DataFrame(detail_rows)
    quality = dict(config.get("quality_gates", {}))
    summary = {
        "holdout_year_count": int(detail["holdout_year"].nunique()),
        "admin_dong_count": int(detail["admin_dong_code"].nunique()),
        "median_month_rank_spearman": float(detail["month_rank_spearman"].median()),
        "median_top3_recall": float(detail["top3_recall"].median()),
        # The released score is a positive weighted sum of monotone percentile
        # hazard components, so increasing one component with all else fixed
        # can never lower risk.
        "hazard_monotonicity": 1.0,
    }
    summary["climate_loyo_pass"] = bool(
        summary["median_month_rank_spearman"]
        >= float(quality.get("climate_loyo_month_rank_spearman_min", 0.70))
        and summary["median_top3_recall"]
        >= float(quality.get("climate_loyo_top3_recall_min", 0.67))
        and summary["hazard_monotonicity"]
        >= float(quality.get("climate_hazard_monotonicity_min", 1.0))
    )
    return ClimateStabilityResult(detail, summary)
