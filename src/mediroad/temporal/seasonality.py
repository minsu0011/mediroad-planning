"""NHIS monthly-utilization seasonality for MEDIROAD Stage 2B.

The resulting modifier describes observed insured utilization seasonality.  It
is not unmet need, a patient-count forecast, or admin-dong-specific clinical
seasonality.  Policy sigungu patterns are normalized before an equal-sigungu
mean so Cheongju's population scale cannot dominate the common modifier.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


NHIS_COLUMN_MAP = {
    "진료년도": "year",
    "진료월": "month",
    "시도": "sido_name",
    "시군구": "source_sigungu_name",
    "진료과목코드": "specialty_code",
    "진료과목명": "specialty_name",
    "진료인원(명)": "persons",
    "진료건수(건)": "visits",
}

SERVICE_NAME_TO_ID = {
    "내과": "internal_medicine",
    "가정의학과": "family_medicine",
    "정형외과": "orthopedics",
    "재활의학과": "rehabilitation",
    "마취통증의학과": "anesthesiology_pain",
    "신경외과": "neurosurgery",
    "안과": "ophthalmology",
    "이비인후과": "otolaryngology",
    "치과": "dental",
    "피부과": "dermatology",
    "신경과": "neurology",
    "정신건강의학과": "psychiatry",
    "외과": "general_surgery",
    "산부인과": "obstetrics_gynecology",
    "비뇨의학과": "urology",
    "응급의학과": "emergency_medicine",
}

POLICY_SIGUNGU = {
    "청주시",
    "충주시",
    "제천시",
    "보은군",
    "옥천군",
    "영동군",
    "증평군",
    "진천군",
    "괴산군",
    "음성군",
    "단양군",
}

CHEONGJU_SOURCE_UNITS = {
    "청주시",
    "청주시 상당구",
    "청주시 서원구",
    "청주시 청원구",
    "청주시 흥덕구",
    "청원군",
}


@dataclass(frozen=True)
class SeasonalityResult:
    """Auditable outputs at service-month and supporting resolutions."""

    service_month: pd.DataFrame
    service_year_month: pd.DataFrame
    policy_sigungu_month: pd.DataFrame


@dataclass(frozen=True)
class ResolutionAssessment:
    """Per-service evidence plus the preregistered month/season decision."""

    detail: pd.DataFrame
    summary: dict[str, Any]


@dataclass(frozen=True)
class BundleSeasonalityResult:
    """Bundle-level profile whose validation target equals its release unit."""

    bundle_month: pd.DataFrame
    reliability_detail: pd.DataFrame
    summary: dict[str, Any]


def _nhis_section(config: Mapping[str, Any]) -> dict[str, Any]:
    value = config.get("nhis_seasonality", config)
    if not isinstance(value, Mapping):
        raise TypeError("nhis_seasonality config must be a mapping")
    return dict(value)


def _map_policy_sigungu(name: object) -> str:
    value = str(name).strip()
    if value in CHEONGJU_SOURCE_UNITS or value.startswith("청주시 "):
        return "청주시"
    return value


def _numeric_count(series: pd.Series, column: str) -> pd.Series:
    values = pd.to_numeric(series.astype(str).str.replace(",", "", regex=False), errors="coerce")
    if values.isna().any():
        raise ValueError(f"NHIS {column} has {int(values.isna().sum())} non-numeric values")
    if (values < 0).any():
        raise ValueError(f"NHIS {column} contains negative counts")
    return values.astype(float)


def read_nhis_monthly_cp949(
    path: str | Path,
    *,
    encoding: str = "cp949",
    province_name: str = "충청북도",
    expected_years: Iterable[int] = (2022, 2023),
    service_name_to_id: Mapping[str, str] | None = None,
    expected_policy_sigungu: int = 11,
) -> pd.DataFrame:
    """Read and normalize the official CP949 NHIS monthly extract.

    Four modern Cheongju districts and the legacy ``청원군`` rows are summed
    into one policy ``청주시``.  Unsupported specialties are excluded rather
    than being guessed into a service bundle.
    """

    if encoding.lower().replace("-", "") not in {"cp949", "ms949"}:
        raise ValueError("The frozen NHIS extract contract requires explicit CP949 decoding")
    frame = pd.read_csv(Path(path), encoding=encoding, dtype=str)
    missing = sorted(set(NHIS_COLUMN_MAP) - set(frame.columns))
    if missing:
        raise KeyError(f"NHIS monthly file lacks required columns: {missing}")
    frame = frame.rename(columns=NHIS_COLUMN_MAP)[list(NHIS_COLUMN_MAP.values())].copy()
    frame = frame.loc[frame["sido_name"].astype(str).str.strip().eq(province_name)].copy()
    if frame.empty:
        raise ValueError(f"NHIS monthly file has no rows for province {province_name}")

    frame["year"] = pd.to_numeric(frame["year"], errors="raise").astype(int)
    frame["month"] = pd.to_numeric(frame["month"], errors="raise").astype(int)
    if not frame["month"].between(1, 12).all():
        raise ValueError("NHIS month must be between 1 and 12")
    years = sorted(int(x) for x in expected_years)
    observed_years = sorted(frame["year"].unique().tolist())
    if observed_years != years:
        raise ValueError(f"NHIS years {observed_years} do not match expected {years}")

    frame["persons"] = _numeric_count(frame["persons"], "persons")
    frame["visits"] = _numeric_count(frame["visits"], "visits")
    frame["policy_sigungu_name"] = frame["source_sigungu_name"].map(_map_policy_sigungu)
    unknown_sigungu = sorted(set(frame["policy_sigungu_name"]) - POLICY_SIGUNGU)
    if unknown_sigungu:
        raise ValueError(f"Unmapped Chungbuk policy sigungu names: {unknown_sigungu}")

    mapping = dict(service_name_to_id or SERVICE_NAME_TO_ID)
    frame["service_id"] = frame["specialty_name"].map(mapping)
    frame = frame.loc[frame["service_id"].notna()].copy()
    if frame.empty:
        raise ValueError("No configured Stage 2 service specialties were found in NHIS data")

    group_columns = ["year", "month", "policy_sigungu_name", "service_id"]
    normalized = (
        frame.groupby(group_columns, as_index=False, sort=True)
        .agg(persons=("persons", "sum"), visits=("visits", "sum"))
        .sort_values(group_columns, kind="mergesort")
        .reset_index(drop=True)
    )
    count = int(normalized["policy_sigungu_name"].nunique())
    if count != expected_policy_sigungu:
        raise ValueError(
            f"NHIS policy mapping produced {count} sigungu, expected {expected_policy_sigungu}"
        )
    return normalized


def parse_public_holiday_api(payload: Mapping[str, Any] | list[Mapping[str, Any]]) -> pd.DataFrame:
    """Normalize official SpcdeInfoService JSON into unique holiday dates."""

    value: Any = payload
    if isinstance(value, Mapping) and "response" in value:
        value = value.get("response", {}).get("body", {}).get("items", {}).get("item", [])
    elif isinstance(value, Mapping) and "items" in value:
        value = value.get("items", {}).get("item", [])
    if isinstance(value, Mapping):
        value = [value]
    if not isinstance(value, list):
        raise TypeError("Holiday API payload must contain a list or mapping of item records")

    rows: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise TypeError("Holiday API item is not a mapping")
        raw_date = item.get("locdate", item.get("date"))
        if raw_date is None:
            raise KeyError("Holiday API item lacks locdate/date")
        parsed = pd.to_datetime(str(raw_date), format="%Y%m%d", errors="coerce")
        if pd.isna(parsed):
            parsed = pd.to_datetime(raw_date, errors="coerce")
        if pd.isna(parsed):
            raise ValueError(f"Invalid holiday date: {raw_date}")
        rows.append(
            {
                "date": pd.Timestamp(parsed).normalize(),
                "holiday_name": str(item.get("dateName", item.get("name", "official_holiday"))),
                "is_holiday": True,
                "source": "official_spcde_info_service",
            }
        )
    if not rows:
        return pd.DataFrame(columns=["date", "holiday_name", "is_holiday", "source"])
    return pd.DataFrame(rows).drop_duplicates("date").sort_values("date").reset_index(drop=True)


def build_working_day_calendar(
    years: Iterable[int],
    holiday_dates: Iterable[Any] | pd.Series | pd.DataFrame,
) -> pd.DataFrame:
    """Count Monday-Friday days after subtracting supplied official holidays."""

    year_values = sorted({int(year) for year in years})
    if not year_values:
        raise ValueError("At least one calendar year is required")
    if isinstance(holiday_dates, pd.DataFrame):
        if "date" not in holiday_dates:
            raise KeyError("Holiday DataFrame must contain date")
        holiday_values = holiday_dates["date"]
    else:
        holiday_values = pd.Series(list(holiday_dates))
    holidays = pd.to_datetime(holiday_values, errors="coerce").dropna().dt.normalize()
    holiday_set = set(holidays.tolist())

    dates = pd.DatetimeIndex(
        np.concatenate(
            [
                pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D").to_numpy()
                for year in year_values
            ]
        )
    )
    daily = pd.DataFrame({"date": dates})
    daily["year"] = daily["date"].dt.year
    daily["month"] = daily["date"].dt.month
    daily["is_weekday"] = daily["date"].dt.weekday.lt(5)
    daily["is_monday_saturday"] = daily["date"].dt.weekday.lt(6)
    daily["is_public_holiday"] = daily["date"].isin(holiday_set)
    daily["is_working_day"] = daily["is_weekday"] & ~daily["is_public_holiday"]
    daily["is_monday_saturday_working_day"] = (
        daily["is_monday_saturday"] & ~daily["is_public_holiday"]
    )
    monthly = (
        daily.groupby(["year", "month"], as_index=False)
        .agg(
            calendar_days=("date", "size"),
            weekday_days=("is_weekday", "sum"),
            monday_saturday_days=("is_monday_saturday", "sum"),
            public_holiday_days=("is_public_holiday", "sum"),
            public_holiday_weekday_days=(
                "is_public_holiday",
                lambda values: int(
                    (values.to_numpy(bool) & daily.loc[values.index, "is_weekday"].to_numpy(bool)).sum()
                ),
            ),
            working_days=("is_working_day", "sum"),
            monday_saturday_working_days=("is_monday_saturday_working_day", "sum"),
        )
        .sort_values(["year", "month"])
        .reset_index(drop=True)
    )
    if len(monthly) != len(year_values) * 12 or monthly["working_days"].le(0).any():
        raise ValueError("Working-day calendar is incomplete or has a zero-working-day month")
    return monthly


def shrink_yearly_month_modifiers(
    service_year_month: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    modifier_column: str = "visits_modifier_equal_sigungu",
) -> pd.DataFrame:
    """Shrink unstable year-to-year monthly patterns toward neutral 100."""

    section = _nhis_section(config)
    reliability = dict(section.get("reliability_shrinkage", {}))
    high_lambda = float(reliability.get("high_lambda", 0.50))
    moderate_lambda = float(reliability.get("moderate_lambda", 0.25))
    low_lambda = float(reliability.get("low_lambda", 0.0))
    if not (0.0 <= low_lambda <= moderate_lambda <= high_lambda <= 1.0):
        raise ValueError("Reliability tier lambdas must be ordered inside [0, 1]")

    required = {
        "service_id",
        "year",
        "month",
        modifier_column,
        "visits_modifier_equal_sigungu",
        "persons_modifier_equal_sigungu",
    }
    missing = sorted(required - set(service_year_month.columns))
    if missing:
        raise KeyError(f"Service-year seasonality lacks columns: {missing}")
    expected_years = sorted(int(x) for x in section.get("observed_years", []))
    if expected_years and sorted(service_year_month["year"].unique().tolist()) != expected_years:
        raise ValueError("Service-year seasonality does not contain the configured observed years")

    rows: list[dict[str, Any]] = []
    for service_id, group in service_year_month.groupby("service_id", sort=True):
        pivots = {
            measure: group.pivot(index="month", columns="year", values=column).sort_index()
            for measure, column in {
                "visits": "visits_modifier_equal_sigungu",
                "persons": "persons_modifier_equal_sigungu",
            }.items()
        }
        pivot = group.pivot(index="month", columns="year", values=modifier_column).sort_index()
        if (
            list(pivot.index) != list(range(1, 13))
            or pivot.isna().any().any()
            or any(value.isna().any().any() for value in pivots.values())
        ):
            raise ValueError(f"Service {service_id} lacks a complete 12-month/year panel")

        measure_correlations: dict[str, float] = {}
        measure_tvd: dict[str, float] = {}
        measure_peak_distance: dict[str, int] = {}
        for measure, measure_pivot in pivots.items():
            correlations: list[float] = []
            tvds: list[float] = []
            peak_distances: list[int] = []
            for left, right in combinations(measure_pivot.columns.tolist(), 2):
                rho_value = float(measure_pivot[left].corr(measure_pivot[right], method="spearman"))
                correlations.append(rho_value if np.isfinite(rho_value) else -1.0)
                left_profile = measure_pivot[left] / float(measure_pivot[left].sum())
                right_profile = measure_pivot[right] / float(measure_pivot[right].sum())
                tvds.append(float(0.5 * np.abs(left_profile - right_profile).sum()))
                left_peak = int(measure_pivot[left].idxmax())
                right_peak = int(measure_pivot[right].idxmax())
                raw_distance = abs(left_peak - right_peak)
                peak_distances.append(min(raw_distance, 12 - raw_distance))
            measure_correlations[measure] = float(np.median(correlations)) if correlations else -1.0
            measure_tvd[measure] = float(max(tvds)) if tvds else 1.0
            measure_peak_distance[measure] = int(max(peak_distances)) if peak_distances else 6

        patient_visit_correlations: list[float] = []
        for year in pivot.columns:
            value = float(pivots["persons"][year].corr(pivots["visits"][year], method="spearman"))
            patient_visit_correlations.append(value if np.isfinite(value) else -1.0)
        patient_visit_rho = float(min(patient_visit_correlations))
        min_measure_rho = float(min(measure_correlations.values()))
        median_measure_rho = float(np.median(list(measure_correlations.values())))
        max_tvd = float(max(measure_tvd.values()))
        max_peak_distance = int(max(measure_peak_distance.values()))

        high = (
            min_measure_rho
            >= float(reliability.get("high_min_both_measure_spearman", 0.70))
            and max_tvd <= float(reliability.get("high_max_total_variation", 0.05))
            and max_peak_distance
            <= int(reliability.get("high_max_peak_distance_months", 1))
            and patient_visit_rho
            >= float(reliability.get("high_min_patient_visit_profile_spearman", 0.70))
        )
        moderate = (
            median_measure_rho
            >= float(reliability.get("moderate_min_median_measure_spearman", 0.40))
            and max_tvd <= float(reliability.get("moderate_max_total_variation", 0.075))
        )
        if high:
            tier, rel = "high", high_lambda
        elif moderate:
            tier, rel = "moderate", moderate_lambda
        else:
            tier, rel = "low", low_lambda
        mean_modifier = pivot.mean(axis=1)
        shrunk = 1.0 + rel * (mean_modifier - 1.0)
        lower = float(section.get("lower_modifier", 0.85))
        upper = float(section.get("upper_modifier", 1.15))
        if not lower < 1.0 < upper:
            raise ValueError("Seasonality modifier bounds must contain neutral 1.0")
        shrunk = shrunk.clip(lower, upper)
        for month in range(1, 13):
            modifier = float(shrunk.loc[month])
            rows.append(
                {
                    "service_id": service_id,
                    "month": month,
                    "modifier_before_reliability_shrink": float(mean_modifier.loc[month]),
                    "year_to_year_spearman": measure_correlations[
                        "visits" if modifier_column.startswith("visits") else "persons"
                    ],
                    "visits_year_to_year_spearman": measure_correlations["visits"],
                    "persons_year_to_year_spearman": measure_correlations["persons"],
                    "patient_visit_profile_spearman_min": patient_visit_rho,
                    "year_profile_total_variation_max": max_tvd,
                    "peak_distance_months_max": max_peak_distance,
                    "seasonality_reliability_tier": tier,
                    "seasonality_reliability": rel,
                    "seasonality_modifier": modifier,
                    "seasonal_service_fit": (modifier - lower) / (upper - lower) * 100.0,
                    "observed_year_count": int(pivot.shape[1]),
                }
            )
    output = pd.DataFrame(rows)
    output["seasonal_service_fit"] = output["seasonal_service_fit"].clip(0.0, 100.0)
    output["seasonality_interpretation"] = section.get(
        "interpretation",
        "observed_insured_utilization_seasonality_not_patient_demand",
    )
    return output


def calculate_service_seasonality(
    nhis: pd.DataFrame,
    working_days: pd.DataFrame,
    config: Mapping[str, Any],
) -> SeasonalityResult:
    """Calculate equal-sigungu monthly modifiers and reliability shrinkage."""

    section = _nhis_section(config)
    required = {"year", "month", "policy_sigungu_name", "service_id", "persons", "visits"}
    missing = sorted(required - set(nhis.columns))
    if missing:
        raise KeyError(f"Normalized NHIS data lacks columns: {missing}")
    calendar_required = {
        "year",
        "month",
        "calendar_days",
        "working_days",
        "monday_saturday_working_days",
    }
    if not calendar_required.issubset(working_days.columns):
        raise KeyError(f"Working-day input lacks columns: {sorted(calendar_required - set(working_days.columns))}")

    merged = nhis.merge(
        working_days[
            ["year", "month", "calendar_days", "working_days", "monday_saturday_working_days"]
        ],
        on=["year", "month"],
        how="left",
        validate="many_to_one",
    )
    if merged["working_days"].isna().any() or merged["working_days"].le(0).any():
        raise ValueError("NHIS rows do not have valid official working-day denominators")
    merged["primary_exposure_days"] = np.where(
        merged["service_id"].eq("emergency_medicine"),
        merged["calendar_days"],
        merged["working_days"],
    )
    merged["primary_exposure_type"] = np.where(
        merged["service_id"].eq("emergency_medicine"),
        section.get("emergency_primary_exposure", "calendar_days"),
        section.get("outpatient_primary_exposure", "weekdays_minus_official_holidays"),
    )
    merged["sensitivity_exposure_days"] = merged["monday_saturday_working_days"]
    merged["exposure_sensitivity_type"] = section.get(
        "exposure_sensitivity", "monday_to_saturday_minus_official_holidays"
    )
    for measure in ("visits", "persons"):
        merged[f"{measure}_per_working_day"] = merged[measure] / merged["primary_exposure_days"]
        merged[f"{measure}_per_sensitivity_day"] = (
            merged[measure] / merged["sensitivity_exposure_days"]
        )
        baseline = merged.groupby(
            ["year", "policy_sigungu_name", "service_id"], sort=False
        )[f"{measure}_per_working_day"].transform("mean")
        if baseline.le(0).any():
            raise ValueError(f"NHIS {measure} annual monthly mean is zero")
        merged[f"{measure}_modifier_sigungu"] = merged[f"{measure}_per_working_day"] / baseline
        sensitivity_baseline = merged.groupby(
            ["year", "policy_sigungu_name", "service_id"], sort=False
        )[f"{measure}_per_sensitivity_day"].transform("mean")
        merged[f"{measure}_modifier_sensitivity_sigungu"] = (
            merged[f"{measure}_per_sensitivity_day"] / sensitivity_baseline
        )

    expected_years = sorted(int(x) for x in section.get("observed_years", []))
    expected_months = 12
    group_count = merged.groupby(["year", "policy_sigungu_name", "service_id"])["month"].nunique()
    if not group_count.eq(expected_months).all():
        bad = group_count[group_count.ne(expected_months)].head().to_dict()
        raise ValueError(f"NHIS policy-sigungu/service panels are incomplete: {bad}")
    if expected_years and sorted(merged["year"].unique().tolist()) != expected_years:
        raise ValueError("NHIS observed years do not match temporal config")

    service_year_month = (
        merged.groupby(["service_id", "year", "month"], as_index=False)
        .agg(
            visits_modifier_equal_sigungu=("visits_modifier_sigungu", "mean"),
            persons_modifier_equal_sigungu=("persons_modifier_sigungu", "mean"),
            visits_modifier_exposure_sensitivity=(
                "visits_modifier_sensitivity_sigungu",
                "mean",
            ),
            persons_modifier_exposure_sensitivity=(
                "persons_modifier_sensitivity_sigungu",
                "mean",
            ),
            policy_sigungu_count=("policy_sigungu_name", "nunique"),
        )
        .sort_values(["service_id", "year", "month"])
        .reset_index(drop=True)
    )
    expected_sigungu = int(config.get("source", {}).get("expected_policy_sigungu", 11))
    if not service_year_month["policy_sigungu_count"].eq(expected_sigungu).all():
        raise ValueError("Equal-sigungu seasonality does not have all configured policy sigungu")

    primary = str(section.get("primary_measure", "visits_per_working_day"))
    modifier_column = {
        "visits_per_working_day": "visits_modifier_equal_sigungu",
        "persons_per_working_day": "persons_modifier_equal_sigungu",
    }.get(primary)
    if modifier_column is None:
        raise ValueError(f"Unsupported NHIS primary measure: {primary}")
    service_month = shrink_yearly_month_modifiers(
        service_year_month,
        config,
        modifier_column=modifier_column,
    )
    return SeasonalityResult(service_month, service_year_month, merged)


def _safe_spearman(left: pd.Series, right: pd.Series) -> float:
    left_values = left.to_numpy(float)
    right_values = right.to_numpy(float)
    if np.ptp(left_values) <= 1e-15 or np.ptp(right_values) <= 1e-15:
        return 1.0 if np.allclose(left_values, right_values) else 0.0
    value = float(left.corr(right, method="spearman"))
    return value if np.isfinite(value) else 0.0


def _top_months(values: pd.Series, k: int = 3) -> set[int]:
    ordered = (
        pd.DataFrame({"month": values.index.astype(int), "value": values.to_numpy(float)})
        .sort_values(["value", "month"], ascending=[False, True], kind="mergesort")
        .head(k)
    )
    return set(ordered["month"].astype(int))


def _circular_month_distance(left: int, right: int) -> int:
    raw = abs(int(left) - int(right))
    return min(raw, 12 - raw)


def assess_monthly_resolution(
    service_year_month: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    value_column: str = "visits_modifier_equal_sigungu",
    sensitivity_column: str = "visits_modifier_exposure_sensitivity",
) -> ResolutionAssessment:
    """Apply preregistered evidence gates and downgrade month to season if needed.

    The decision uses raw year profiles, before reliability shrinkage.  LOW
    services therefore cannot manufacture monthly evidence merely because
    their released modifier is exactly neutral 1.0.
    """

    required = {"service_id", "year", "month", value_column, sensitivity_column}
    missing = sorted(required - set(service_year_month.columns))
    if missing:
        raise KeyError(f"Monthly-resolution input lacks columns: {missing}")
    quality = dict(config.get("quality_gates", {}))
    rows: list[dict[str, Any]] = []
    for service_id, group in service_year_month.groupby("service_id", sort=True):
        primary = group.pivot(index="month", columns="year", values=value_column).sort_index()
        sensitivity = group.pivot(
            index="month", columns="year", values=sensitivity_column
        ).sort_index()
        if (
            list(primary.index) != list(range(1, 13))
            or primary.shape[1] < 2
            or primary.isna().any().any()
            or sensitivity.isna().any().any()
        ):
            raise ValueError(f"Service {service_id} lacks complete monthly-resolution evidence")
        year_rhos: list[float] = []
        year_jaccards: list[float] = []
        peak_distances: list[int] = []
        for left, right in combinations(primary.columns.tolist(), 2):
            year_rhos.append(_safe_spearman(primary[left], primary[right]))
            left_top, right_top = _top_months(primary[left]), _top_months(primary[right])
            year_jaccards.append(len(left_top & right_top) / len(left_top | right_top))
            peak_distances.append(
                _circular_month_distance(primary[left].idxmax(), primary[right].idxmax())
            )
        primary_mean = primary.mean(axis=1)
        sensitivity_mean = sensitivity.mean(axis=1)
        exposure_rho = _safe_spearman(primary_mean, sensitivity_mean)
        primary_top = _top_months(primary_mean)
        sensitivity_top = _top_months(sensitivity_mean)
        exposure_jaccard = len(primary_top & sensitivity_top) / len(primary_top | sensitivity_top)
        row = {
            "service_id": service_id,
            "loyo_spearman": float(np.median(year_rhos)),
            "loyo_top3_jaccard": float(np.median(year_jaccards)),
            "exposure_sensitivity_spearman": exposure_rho,
            "exposure_sensitivity_top3_jaccard": exposure_jaccard,
            "peak_distance_months": int(max(peak_distances)),
        }
        row["monthly_resolution_pass"] = bool(
            row["loyo_spearman"]
            >= float(quality.get("monthly_resolution_loyo_spearman_min", 0.40))
            and row["loyo_top3_jaccard"]
            >= float(quality.get("monthly_resolution_top3_jaccard_min", 0.50))
            and row["exposure_sensitivity_spearman"]
            >= float(
                quality.get("monthly_resolution_exposure_sensitivity_spearman_min", 0.75)
            )
            and row["exposure_sensitivity_top3_jaccard"]
            >= float(quality.get("monthly_resolution_exposure_sensitivity_top3_jaccard_min", 0.50))
            and row["peak_distance_months"]
            <= int(quality.get("monthly_resolution_peak_distance_months_max", 2))
        )
        rows.append(row)

    detail = pd.DataFrame(rows)
    summary = {
        "service_count": int(len(detail)),
        "median_loyo_spearman": float(detail["loyo_spearman"].median()),
        "median_loyo_top3_jaccard": float(detail["loyo_top3_jaccard"].median()),
        "median_exposure_sensitivity_spearman": float(
            detail["exposure_sensitivity_spearman"].median()
        ),
        "median_exposure_sensitivity_top3_jaccard": float(
            detail["exposure_sensitivity_top3_jaccard"].median()
        ),
        "max_peak_distance_months": int(detail["peak_distance_months"].max()),
    }
    summary["monthly_resolution_pass"] = bool(
        summary["median_loyo_spearman"]
        >= float(quality.get("monthly_resolution_loyo_spearman_min", 0.40))
        and summary["median_loyo_top3_jaccard"]
        >= float(quality.get("monthly_resolution_top3_jaccard_min", 0.50))
        and summary["median_exposure_sensitivity_spearman"]
        >= float(quality.get("monthly_resolution_exposure_sensitivity_spearman_min", 0.75))
        and summary["median_exposure_sensitivity_top3_jaccard"]
        >= float(quality.get("monthly_resolution_exposure_sensitivity_top3_jaccard_min", 0.50))
        and summary["max_peak_distance_months"]
        <= int(quality.get("monthly_resolution_peak_distance_months_max", 2))
    )
    summary["recommended_planning_resolution"] = (
        "month" if summary["monthly_resolution_pass"] else "season"
    )
    detail["overall_recommended_planning_resolution"] = summary[
        "recommended_planning_resolution"
    ]
    return ResolutionAssessment(detail, summary)


def build_bundle_seasonality(
    service_month: pd.DataFrame,
    bundle_config: Mapping[str, Any],
) -> pd.DataFrame:
    """Aggregate supported service modifiers with explicit bundle weights."""

    required = {
        "service_id",
        "month",
        "seasonality_modifier",
        "seasonal_service_fit",
        "seasonality_reliability",
    }
    missing = sorted(required - set(service_month.columns))
    if missing:
        raise KeyError(f"Service-month seasonality lacks columns: {missing}")
    bundles = bundle_config.get("bundles", [])
    if not bundles:
        raise ValueError("Service bundle config contains no bundles")

    indexed = service_month.set_index(["service_id", "month"]).sort_index()
    rows: list[dict[str, Any]] = []
    for bundle in bundles:
        bundle_id = str(bundle["bundle_id"])
        services = dict(bundle.get("included_services", {}))
        if not services:
            raise ValueError(f"Bundle {bundle_id} has no included services")
        total = sum(float(value) for value in services.values())
        if not np.isclose(total, 1.0, atol=1e-9):
            raise ValueError(f"Bundle {bundle_id} service weights sum to {total}, expected 1")
        compatible = {int(value) for value in bundle.get("compatible_months", range(1, 13))}
        for month in range(1, 13):
            modifier = 0.0
            fit = 0.0
            reliability = 0.0
            for service_id, weight_value in services.items():
                key = (str(service_id), month)
                if key not in indexed.index:
                    raise KeyError(f"Bundle {bundle_id} service-month missing: {key}")
                source = indexed.loc[key]
                if isinstance(source, pd.DataFrame):
                    raise ValueError(f"Duplicate service-month seasonality row: {key}")
                weight = float(weight_value)
                modifier += weight * float(source["seasonality_modifier"])
                fit += weight * float(source["seasonal_service_fit"])
                reliability += weight * float(source["seasonality_reliability"])
            rows.append(
                {
                    "bundle_id": bundle_id,
                    "bundle_name_ko": str(bundle.get("bundle_name_ko", bundle_id)),
                    "month": month,
                    "seasonality_modifier": modifier,
                    "seasonal_service_fit": fit,
                    "seasonality_reliability": reliability,
                    "bundle_month_compatible": month in compatible,
                    "bundle_status": bundle_config.get(
                        "status", "recommended_service_bundles_not_confirmed_staffing_plan"
                    ),
                }
            )
    output = pd.DataFrame(rows).sort_values(["bundle_id", "month"]).reset_index(drop=True)
    if output.duplicated(["bundle_id", "month"]).any():
        raise ValueError("Bundle seasonality has duplicate bundle-month rows")
    return output


def build_bundle_seasonality_from_yearly_evidence(
    service_year_month: pd.DataFrame,
    bundle_config: Mapping[str, Any],
    config: Mapping[str, Any],
) -> BundleSeasonalityResult:
    """Validate and shrink the same service-bundle profiles that are released.

    Service-level reliability is valuable for service diagnostics, but applying
    it before bundle aggregation can reverse a bundle's validated seasonal
    ordering.  This path first builds each configured bundle within each year,
    evaluates the two year profiles at the released four-season resolution,
    and applies one reliability shrinkage factor to that same bundle profile.
    """

    required = {
        "service_id",
        "year",
        "month",
        "visits_modifier_equal_sigungu",
    }
    missing = sorted(required - set(service_year_month.columns))
    if missing:
        raise KeyError(f"Bundle yearly seasonality lacks columns: {missing}")
    evidence = service_year_month.copy()
    if evidence.duplicated(["service_id", "year", "month"]).any():
        raise ValueError("Service-year-month evidence key is not unique")
    evidence["year"] = pd.to_numeric(evidence["year"], errors="raise").astype(int)
    evidence["month"] = pd.to_numeric(evidence["month"], errors="raise").astype(int)
    evidence["visits_modifier_equal_sigungu"] = pd.to_numeric(
        evidence["visits_modifier_equal_sigungu"], errors="raise"
    ).astype(float)
    if not evidence["month"].between(1, 12).all():
        raise ValueError("Bundle yearly evidence months must be 1..12")
    modifier_values = evidence["visits_modifier_equal_sigungu"].to_numpy(float)
    if not np.isfinite(modifier_values).all() or (modifier_values <= 0).any():
        raise ValueError("Bundle yearly evidence modifiers must be finite and positive")
    years = sorted(evidence["year"].unique())
    expected_years = sorted(int(value) for value in config["nhis_seasonality"]["observed_years"])
    if years != expected_years or len(years) < 2:
        raise ValueError("Bundle reliability requires all configured observed years")
    indexed = evidence.set_index(["service_id", "year", "month"])
    bundles = bundle_config.get("bundles", [])
    if not bundles:
        raise ValueError("Service bundle config contains no bundles")
    raw_rows: list[dict[str, Any]] = []
    for bundle in bundles:
        bundle_id = str(bundle["bundle_id"])
        included = {
            str(service_id): float(weight)
            for service_id, weight in bundle.get("included_services", {}).items()
        }
        if not included or not np.isclose(sum(included.values()), 1.0, atol=1e-9):
            raise ValueError(f"Bundle {bundle_id} weights must be present and sum to one")
        for year in years:
            for month in range(1, 13):
                value = 0.0
                for service_id, weight in included.items():
                    key = (service_id, year, month)
                    if key not in indexed.index:
                        raise KeyError(f"Bundle yearly profile lacks {key}")
                    source = indexed.loc[key, "visits_modifier_equal_sigungu"]
                    if isinstance(source, pd.Series):
                        raise ValueError(f"Duplicate bundle yearly source row: {key}")
                    value += weight * float(source)
                raw_rows.append(
                    {
                        "bundle_id": bundle_id,
                        "bundle_name_ko": str(bundle.get("bundle_name_ko", bundle_id)),
                        "year": int(year),
                        "month": month,
                        "raw_bundle_modifier": value,
                    }
                )
    raw = pd.DataFrame(raw_rows)
    if raw.duplicated(["bundle_id", "year", "month"]).any():
        raise RuntimeError("Bundle-year-month construction is not unique")

    month_to_season = {
        month: season
        for season, months in {
            "winter": (12, 1, 2),
            "spring": (3, 4, 5),
            "summer": (6, 7, 8),
            "autumn": (9, 10, 11),
        }.items()
        for month in months
    }
    season_order = {"winter": 1, "spring": 2, "summer": 3, "autumn": 4}
    raw["season"] = raw["month"].map(month_to_season)
    reliability = dict(config.get("bundle_season_reliability", {}))
    high_min = float(reliability.get("high_min_season_rank_spearman", 0.70))
    moderate_min = float(reliability.get("moderate_min_season_rank_spearman", 0.40))
    high_lambda = float(reliability.get("high_lambda", 0.50))
    moderate_lambda = float(reliability.get("moderate_lambda", 0.25))
    low_lambda = float(reliability.get("low_lambda", 0.0))
    comparison_tolerance = float(reliability.get("comparison_tolerance", 1e-12))
    flat_signal_tolerance = float(reliability.get("flat_signal_tolerance", 1e-12))
    if comparison_tolerance < 0 or flat_signal_tolerance < 0:
        raise ValueError("Bundle reliability tolerances must be non-negative")
    if any(value < 0 or value > 1 for value in (high_lambda, moderate_lambda, low_lambda)):
        raise ValueError("Bundle reliability shrinkage lambdas must be in [0, 1]")
    detail_rows: list[dict[str, Any]] = []
    month_rows: list[dict[str, Any]] = []
    bundle_lookup = {str(row["bundle_id"]): row for row in bundles}
    lower = float(config["nhis_seasonality"]["lower_modifier"])
    upper = float(config["nhis_seasonality"]["upper_modifier"])
    for bundle_id, group in raw.groupby("bundle_id", sort=True):
        monthly = group.pivot(index="month", columns="year", values="raw_bundle_modifier").sort_index()
        seasonal = (
            group.groupby(["year", "season"], as_index=False)["raw_bundle_modifier"]
            .mean()
            .pivot(index="season", columns="year", values="raw_bundle_modifier")
            .reindex(season_order)
        )
        left_year, right_year = years[0], years[1]
        monthly_rho = _safe_spearman(monthly[left_year], monthly[right_year])
        left_top3 = _top_months(monthly[left_year], 3)
        right_top3 = _top_months(monthly[right_year], 3)
        monthly_top3_jaccard = len(left_top3 & right_top3) / len(left_top3 | right_top3)
        monthly_peak_distance = _circular_month_distance(
            int(monthly[left_year].idxmax()), int(monthly[right_year].idxmax())
        )
        seasonal_rho = _safe_spearman(seasonal[left_year], seasonal[right_year])

        def ranked_seasons(values: pd.Series) -> list[str]:
            ordered = pd.DataFrame(
                {
                    "season": values.index.astype(str),
                    "value": values.to_numpy(float),
                }
            )
            ordered["season_order"] = ordered["season"].map(season_order)
            return (
                ordered.sort_values(
                    ["value", "season_order"],
                    ascending=[False, True],
                    kind="mergesort",
                )["season"]
                .astype(str)
                .tolist()
            )

        left_ranked = ranked_seasons(seasonal[left_year])
        right_ranked = ranked_seasons(seasonal[right_year])
        top_left = left_ranked[0]
        top_right = right_ranked[0]
        top1_agreement = int(top_left == top_right)
        high_requires_top1 = bool(reliability.get("high_require_top1_agreement", True))
        average_profile = monthly.mean(axis=1)
        average_mean = float(average_profile.mean())
        if not np.isfinite(average_mean) or average_mean <= 0:
            raise ValueError(f"Bundle {bundle_id} has an invalid average profile mean")
        average_profile = average_profile / average_mean
        average_season_profile = pd.Series(
            {
                season: float(average_profile.loc[list(months)].mean())
                for season, months in {
                    "winter": (12, 1, 2),
                    "spring": (3, 4, 5),
                    "summer": (6, 7, 8),
                    "autumn": (9, 10, 11),
                }.items()
            }
        )
        raw_decision_season_range = float(
            average_season_profile.max() - average_season_profile.min()
        )
        if raw_decision_season_range <= flat_signal_tolerance:
            tier, shrink = "low", low_lambda
            reliability_decision_reason = "no_bundle_level_season_signal"
        elif seasonal_rho + comparison_tolerance >= high_min and (
            top1_agreement or not high_requires_top1
        ):
            tier, shrink = "high", high_lambda
            reliability_decision_reason = "high_season_rank_stability"
        elif seasonal_rho + comparison_tolerance >= moderate_min:
            tier, shrink = "moderate", moderate_lambda
            reliability_decision_reason = "moderate_season_rank_stability"
        else:
            tier, shrink = "low", low_lambda
            reliability_decision_reason = "insufficient_season_rank_stability"
        released_modifier = (1.0 + shrink * (average_profile - 1.0)).clip(lower, upper)
        released_fit = (released_modifier - lower) / (upper - lower) * 100.0
        released_season_fit = pd.Series(
            {
                season: float(released_fit.loc[list(months)].mean())
                for season, months in {
                    "winter": (12, 1, 2),
                    "spring": (3, 4, 5),
                    "summer": (6, 7, 8),
                    "autumn": (9, 10, 11),
                }.items()
            }
        )
        released_season_fit_range = float(
            released_season_fit.max() - released_season_fit.min()
        )
        release_minimum = float(
            config.get("quality_gates", {}).get(
                "minimum_bundle_season_fit_range_for_release", 1e-9
            )
        )
        temporal_signal_available = bool(
            shrink > 0 and released_season_fit_range > release_minimum
        )
        released_ranked = ranked_seasons(released_season_fit)
        released_primary, released_fallback = released_ranked[:2]
        released_primary_fallback_margin = float(
            released_season_fit.loc[released_primary]
            - released_season_fit.loc[released_fallback]
        )
        left_top2, right_top2 = set(left_ranked[:2]), set(right_ranked[:2])
        released_top2 = set(released_ranked[:2])
        released_top2_year_a_jaccard = len(released_top2 & left_top2) / len(
            released_top2 | left_top2
        )
        released_top2_year_b_jaccard = len(released_top2 & right_top2) / len(
            released_top2 | right_top2
        )
        released_primary_in_both_year_top2 = int(
            released_primary in left_top2 and released_primary in right_top2
        )
        if temporal_signal_available:
            if top1_agreement:
                direction_concordant = int(released_primary == top_left)
            else:
                direction_concordant = int(
                    released_primary in {top_left, top_right}
                    and released_primary_in_both_year_top2
                )
        else:
            direction_concordant = 1
        detail_rows.append(
            {
                "bundle_id": bundle_id,
                "monthly_year_spearman": monthly_rho,
                "monthly_top3_jaccard": monthly_top3_jaccard,
                "monthly_peak_distance": monthly_peak_distance,
                "season_rank_spearman": seasonal_rho,
                "top1_season_agreement": top1_agreement,
                "top1_season_year_a": top_left,
                "top1_season_year_b": top_right,
                "bundle_reliability_tier": tier,
                "bundle_reliability_lambda": shrink,
                "reliability_decision_reason": reliability_decision_reason,
                "raw_decision_season_range": raw_decision_season_range,
                "released_season_fit_range": released_season_fit_range,
                "released_primary_season": released_primary,
                "released_fallback_season": released_fallback,
                "released_primary_fallback_margin": released_primary_fallback_margin,
                "released_primary_matches_year_a_top1": int(released_primary == top_left),
                "released_primary_matches_year_b_top1": int(released_primary == top_right),
                "released_primary_matches_both_year_top1": int(
                    released_primary == top_left == top_right
                ),
                "released_primary_in_year_a_top2": int(released_primary in left_top2),
                "released_primary_in_year_b_top2": int(released_primary in right_top2),
                "released_primary_in_both_year_top2": released_primary_in_both_year_top2,
                "released_top2_year_a_jaccard": released_top2_year_a_jaccard,
                "released_top2_year_b_jaccard": released_top2_year_b_jaccard,
                "validation_release_direction_concordant": direction_concordant,
                "temporal_signal_available": temporal_signal_available,
                "validation_release_unit_match": 1,
            }
        )
        bundle = bundle_lookup[bundle_id]
        compatible = {int(value) for value in bundle.get("compatible_months", range(1, 13))}
        for month in range(1, 13):
            modifier = float(released_modifier.loc[month])
            month_rows.append(
                {
                    "bundle_id": bundle_id,
                    "bundle_name_ko": str(bundle.get("bundle_name_ko", bundle_id)),
                    "month": month,
                    "seasonality_modifier": modifier,
                    "seasonal_service_fit": float(released_fit.loc[month]),
                    "seasonality_reliability": shrink,
                    "seasonality_reliability_tier": tier,
                    "bundle_month_compatible": month in compatible,
                    "bundle_status": bundle_config.get(
                        "status", "recommended_service_bundles_not_confirmed_staffing_plan"
                    ),
                    "bundle_profile_definition": (
                        "bundle_weighted_raw_year_profile_then_bundle_level_reliability_shrink"
                    ),
                }
            )
    bundle_month = pd.DataFrame(month_rows).sort_values(["bundle_id", "month"]).reset_index(drop=True)
    detail = pd.DataFrame(detail_rows).sort_values("bundle_id").reset_index(drop=True)
    if len(bundle_month) != len(bundles) * 12 or bundle_month.duplicated(["bundle_id", "month"]).any():
        raise RuntimeError("Released bundle-month seasonality coverage is incomplete")
    actionable_detail = detail.loc[detail["temporal_signal_available"]].copy()
    direction_fraction = (
        float(actionable_detail["validation_release_direction_concordant"].mean())
        if not actionable_detail.empty
        else 1.0
    )
    both_year_top2_fraction = (
        float(actionable_detail["released_primary_in_both_year_top2"].mean())
        if not actionable_detail.empty
        else 1.0
    )
    profile_mismatch = (
        detail["validation_release_unit_match"].ne(1)
        | (
            detail["temporal_signal_available"]
            & detail["validation_release_direction_concordant"].ne(1)
        )
    )
    summary = {
        "bundle_count": int(len(detail)),
        "median_season_rank_spearman": float(detail["season_rank_spearman"].median()),
        "top1_season_agreement": float(detail["top1_season_agreement"].mean()),
        "high_reliability_bundle_count": int(detail["bundle_reliability_tier"].eq("high").sum()),
        "moderate_reliability_bundle_count": int(detail["bundle_reliability_tier"].eq("moderate").sum()),
        "low_reliability_bundle_count": int(detail["bundle_reliability_tier"].eq("low").sum()),
        "actionable_bundle_count": int(detail["temporal_signal_available"].sum()),
        "no_signal_bundle_count": int((~detail["temporal_signal_available"]).sum()),
        "validation_release_unit_match_fraction": float(
            detail["validation_release_unit_match"].mean()
        ),
        "validation_release_direction_concordance_fraction": direction_fraction,
        "validation_release_profile_mismatch_count": int(profile_mismatch.sum()),
        "released_primary_in_both_year_top2_fraction": both_year_top2_fraction,
        "released_top2_mean_year_jaccard": float(
            detail[
                ["released_top2_year_a_jaccard", "released_top2_year_b_jaccard"]
            ].to_numpy(float).mean()
        ),
        "released_primary_matches_both_year_top1_fraction": float(
            detail["released_primary_matches_both_year_top1"].mean()
        ),
    }
    quality = dict(config.get("quality_gates", {}))
    summary["seasonal_nhis_gate_pass"] = bool(
        summary["median_season_rank_spearman"]
        >= float(quality.get("seasonal_resolution_nhis_loyo_spearman_min", 0.40))
        and summary["top1_season_agreement"]
        >= float(quality.get("seasonal_resolution_nhis_top1_agreement_min", 0.50))
        and summary["validation_release_profile_mismatch_count"]
        <= int(quality.get("validation_release_profile_mismatch_count_max", 0))
        and summary["released_primary_in_both_year_top2_fraction"]
        >= float(quality.get("released_primary_in_both_year_top2_fraction_min", 1.0))
    )
    return BundleSeasonalityResult(bundle_month, detail, summary)
