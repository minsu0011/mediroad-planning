from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mediroad.temporal import (  # noqa: E402
    HISTORICAL_CLIMATE_ROLE,
    UNAVAILABLE_REPLAN_TRIGGER_STATUS,
    apply_season_release_contract,
    assert_unknown_not_zero,
    assess_monthly_resolution,
    build_bundle_seasonality,
    build_bundle_seasonality_from_yearly_evidence,
    build_monthly_climate_risk,
    build_region_bundle_month_fit,
    build_temporal_data_audit,
    build_working_day_calendar,
    climate_leave_one_year_out_stability,
    parse_public_holiday_api,
    read_nhis_monthly_cp949,
    run_temporal_ablation,
    select_primary_fallback_months,
    select_primary_fallback_seasons,
    shrink_yearly_month_modifiers,
    temporal_audit_row,
)


@pytest.fixture(scope="module")
def temporal_config() -> dict:
    return yaml.safe_load((ROOT / "configs/model_v1/temporal_model.yaml").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def bundle_config() -> dict:
    return yaml.safe_load((ROOT / "configs/model_v1/service_bundles.yaml").read_text(encoding="utf-8"))


def test_cp949_nhis_reader_merges_cheongju_policy_units(tmp_path: Path) -> None:
    ordinary = [
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
    ]
    cheongju = [
        "청주시 상당구",
        "청주시 서원구",
        "청주시 청원구",
        "청주시 흥덕구",
        "청원군",
    ]
    rows = []
    for year in (2022, 2023):
        for month in range(1, 13):
            for sigungu in ordinary + cheongju:
                rows.append(
                    {
                        "진료년도": year,
                        "진료월": f"{month:02d}",
                        "시도": "충청북도",
                        "시군구": sigungu,
                        "진료과목코드": 1,
                        "진료과목명": "내과",
                        "진료인원(명)": 1,
                        "진료건수(건)": 2,
                    }
                )
    path = tmp_path / "nhis_cp949.csv"
    pd.DataFrame(rows).to_csv(path, index=False, encoding="cp949")

    normalized = read_nhis_monthly_cp949(path)
    assert normalized["policy_sigungu_name"].nunique() == 11
    assert "청원군" not in set(normalized["policy_sigungu_name"])
    cheongju_row = normalized.loc[
        normalized["policy_sigungu_name"].eq("청주시")
        & normalized["year"].eq(2022)
        & normalized["month"].eq(1)
    ].iloc[0]
    assert cheongju_row["persons"] == 5
    assert cheongju_row["visits"] == 10
    with pytest.raises(ValueError, match="CP949"):
        read_nhis_monthly_cp949(path, encoding="utf-8")


def test_official_holiday_input_corrects_working_days() -> None:
    payload = {
        "response": {
            "body": {
                "items": {
                    "item": [
                        {"locdate": 20220101, "dateName": "weekend holiday"},
                        {"locdate": 20220103, "dateName": "weekday holiday"},
                    ]
                }
            }
        }
    }
    holidays = parse_public_holiday_api(payload)
    calendar = build_working_day_calendar([2022], holidays)
    january = calendar.loc[calendar["month"].eq(1)].iloc[0]
    assert january["working_days"] == january["weekday_days"] - 1
    assert january["monday_saturday_working_days"] == january["monday_saturday_days"] - 2
    assert len(calendar) == 12


def _year_profile_rows(stable: bool = True) -> pd.DataFrame:
    profile = np.linspace(0.8, 1.2, 12)
    profile = profile / profile.mean()
    rows = []
    for year in (2022, 2023):
        values = profile if stable or year == 2022 else profile[::-1]
        for month, value in enumerate(values, start=1):
            rows.append(
                {
                    "service_id": "internal_medicine",
                    "year": year,
                    "month": month,
                    "visits_modifier_equal_sigungu": value,
                    "persons_modifier_equal_sigungu": value,
                    "visits_modifier_exposure_sensitivity": value,
                    "persons_modifier_exposure_sensitivity": value,
                }
            )
    return pd.DataFrame(rows)


def test_reliability_tier_shrinks_and_low_is_exact_neutral(temporal_config: dict) -> None:
    stable = shrink_yearly_month_modifiers(_year_profile_rows(True), temporal_config)
    assert set(stable["seasonality_reliability_tier"]) == {"high"}
    assert set(stable["seasonality_reliability"]) == {0.5}
    original = _year_profile_rows(True).query("year == 2022").sort_values("month")
    expected = 1.0 + 0.5 * (original["visits_modifier_equal_sigungu"].to_numpy() - 1.0)
    np.testing.assert_allclose(stable.sort_values("month")["seasonality_modifier"], expected)

    unstable = shrink_yearly_month_modifiers(_year_profile_rows(False), temporal_config)
    assert set(unstable["seasonality_reliability_tier"]) == {"low"}
    assert set(unstable["seasonality_reliability"]) == {0.0}
    np.testing.assert_allclose(unstable["seasonality_modifier"], 1.0, atol=0.0)


def test_monthly_resolution_can_downgrade_to_season(temporal_config: dict) -> None:
    monthly = assess_monthly_resolution(_year_profile_rows(True), temporal_config)
    assert monthly.summary["monthly_resolution_pass"]
    assert monthly.summary["recommended_planning_resolution"] == "month"

    downgraded = assess_monthly_resolution(_year_profile_rows(False), temporal_config)
    assert not downgraded.summary["monthly_resolution_pass"]
    assert downgraded.summary["recommended_planning_resolution"] == "season"


def _weather_fixture() -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", "2020-12-31", freq="D")
    rows = []
    for admin_code in ("A", "B"):
        for date in dates:
            hot = admin_code == "A" and date.month == 7
            rows.append(
                {
                    "admin_dong_code": admin_code,
                    "date": date,
                    "temperature_2m_max": 35.0 if hot else 25.0,
                    "temperature_2m_min": 15.0,
                    "precipitation_sum": 35.0 if admin_code == "A" and date.month == 8 else 0.0,
                    "wind_speed_10m_max": 5.0,
                }
            )
    return pd.DataFrame(rows)


def _weather_units(wind: str = "m/s") -> dict[str, str]:
    return {
        "temperature_2m_max": "celsius",
        "temperature_2m_min": "celsius",
        "precipitation_sum": "mm",
        "wind_speed_10m_max": wind,
    }


def test_nasa_power_daily_to_monthly_risk_has_unit_hard_gate(temporal_config: dict) -> None:
    daily = _weather_fixture()
    monthly = build_monthly_climate_risk(
        daily,
        temporal_config,
        units=_weather_units(),
        expected_admin_dongs=2,
        expected_start_date="2020-01-01",
        expected_end_date="2020-12-31",
    )
    assert len(monthly) == 24
    hot = monthly.loc[monthly["admin_dong_code"].eq("A") & monthly["month"].eq(7)].iloc[0]
    control = monthly.loc[monthly["admin_dong_code"].eq("B") & monthly["month"].eq(7)].iloc[0]
    assert hot["climate_operational_risk"] > control["climate_operational_risk"]
    assert hot["climate_operational_fit"] < control["climate_operational_fit"]
    assert hot["snowfall_hazard_status"].startswith("unknown_")

    with pytest.raises(ValueError, match="unit mismatch"):
        build_monthly_climate_risk(
            daily,
            temporal_config,
            units=_weather_units("km/h"),
            expected_admin_dongs=2,
            expected_start_date="2020-01-01",
            expected_end_date="2020-12-31",
        )


def test_climate_leave_one_year_out_is_auditable(temporal_config: dict) -> None:
    frames = []
    for year in (2018, 2019, 2020):
        frame = _weather_fixture().copy()
        frame["date"] = frame["date"].map(
            lambda value: value.replace(year=year) if not (value.month == 2 and value.day == 29) else pd.NaT
        )
        frame = frame.dropna(subset=["date"])
        # Add leap day for 2020 so every held-out calendar is complete.
        if year == 2020:
            leap_rows = frame.loc[frame["date"].dt.strftime("%m-%d").eq("02-28")].copy()
            leap_rows["date"] = pd.Timestamp("2020-02-29")
            frame = pd.concat([frame, leap_rows], ignore_index=True)
        frames.append(frame)
    daily = pd.concat(frames, ignore_index=True)
    stability = climate_leave_one_year_out_stability(
        daily,
        temporal_config,
        units=_weather_units(),
    )
    assert stability.summary["holdout_year_count"] == 3
    assert stability.summary["admin_dong_count"] == 2
    assert stability.summary["hazard_monotonicity"] == 1.0
    assert len(stability.detail) == 6


def _bundle_month_fixture() -> pd.DataFrame:
    rows = []
    for bundle_id in ("b1", "b2"):
        for month in range(1, 13):
            rows.append(
                {
                    "bundle_id": bundle_id,
                    "bundle_name_ko": bundle_id,
                    "month": month,
                    "seasonal_service_fit": month / 12 * 100,
                    "seasonality_modifier": 0.85 + month / 12 * 0.30,
                    "seasonality_reliability": 0.5,
                    "bundle_month_compatible": True,
                }
            )
    return pd.DataFrame(rows)


def test_region_bundle_month_fit_recommendations_unknown_and_ablation(
    temporal_config: dict,
) -> None:
    regions = pd.DataFrame(
        {
            "admin_dong_code": ["A", "B"],
            "admin_dong_name": ["alpha", "beta"],
            "policy_sigungu_name": ["g1", "g2"],
        }
    )
    climate = pd.DataFrame(
        [
            {
                "admin_dong_code": admin,
                "month": month,
                "climate_operational_fit": month / 12 * 100,
                "climate_operational_risk": 100 - month / 12 * 100,
            }
            for admin in ("A", "B")
            for month in range(1, 13)
        ]
    )
    transit = pd.DataFrame(
        {"admin_dong_code": ["A", "B"], "transit_time_fit": [60.0, 40.0]}
    )
    fit = build_region_bundle_month_fit(
        regions,
        _bundle_month_fixture(),
        climate,
        transit,
        temporal_config,
    )
    assert len(fit) == 48
    assert fit["temporal_fit_score"].between(0, 100).all()
    unknown_columns = list(temporal_config["temporal_fit"]["unavailable_components"])
    assert_unknown_not_zero(fit, unknown_columns)
    assert fit[unknown_columns].apply(lambda column: column.str.startswith("unknown_").all()).all()

    recommendations = select_primary_fallback_months(fit, temporal_config)
    assert len(recommendations) == 4
    assert recommendations["fallback_distinct_from_primary"].eq(1).all()
    assert recommendations["fallback_different_quarter"].eq(1).all()
    assert recommendations["primary_month"].eq(12).all()
    assert recommendations["fallback_month"].eq(9).all()

    seasonal_recommendations = select_primary_fallback_seasons(fit, temporal_config)
    assert len(seasonal_recommendations) == 4
    assert seasonal_recommendations["planning_resolution"].eq("season").all()
    assert seasonal_recommendations["fallback_distinct_from_primary"].eq(1).all()
    assert seasonal_recommendations[
        ["primary_month", "fallback_month", "recommended_date", "fallback_date"]
    ].isna().all().all()
    assert seasonal_recommendations["monthly_output_status"].eq(
        "diagnostic_appendix_only_monthly_gate_failed"
    ).all()

    ablation = run_temporal_ablation(
        fit,
        temporal_config,
        source_components={
            "nhis_observed_utilization_seasonality": ("seasonal_service_fit",),
        },
    )
    assert set(ablation.summary["ablation_type"]) == {"component", "source"}
    assert ablation.summary.query("ablation_type == 'component'")["variant"].nunique() == 1
    assert ablation.summary.query("ablation_type == 'source'")["variant"].nunique() == 1
    assert ablation.summary["spearman"].between(-1, 1).all()
    assert ablation.summary["top3_recall"].between(0, 1).all()

    bad = fit.copy()
    bad.loc[0, unknown_columns[0]] = "0"
    with pytest.raises(ValueError, match="never as zero"):
        assert_unknown_not_zero(bad, unknown_columns)


def test_temporal_audit_schema_is_complete() -> None:
    frame = pd.DataFrame({"year": [2022, 2023], "value": [1.0, 2.0]})
    row = temporal_audit_row(
        frame,
        dataset="synthetic",
        time_resolution="year",
        spatial_resolution="policy_sigungu",
        specialty_resolution="service",
        source="unit_test",
        usable_for_model=True,
        usable_for_validation=False,
        confidence="test_only",
        limitations="synthetic",
        time_values=frame["year"],
        required_columns=["year", "value"],
    )
    audit = build_temporal_data_audit([row])
    assert len(audit) == 1
    assert audit.loc[0, "missingness"] == 0.0


def test_all_configured_bundle_weights_are_explicit(bundle_config: dict) -> None:
    services = sorted(
        {
            service
            for bundle in bundle_config["bundles"]
            for service in bundle["included_services"]
        }
    )
    service_rows = []
    for service in services:
        for month in range(1, 13):
            service_rows.append(
                {
                    "service_id": service,
                    "month": month,
                    "seasonality_modifier": 1.0,
                    "seasonal_service_fit": 50.0,
                    "seasonality_reliability": 0.0,
                }
            )
    bundles = build_bundle_seasonality(pd.DataFrame(service_rows), bundle_config)
    assert len(bundles) == len(bundle_config["bundles"]) * 12
    np.testing.assert_allclose(bundles["seasonality_modifier"], 1.0)


SEASON_TO_MONTHS = {
    "winter": (12, 1, 2),
    "spring": (3, 4, 5),
    "summer": (6, 7, 8),
    "autumn": (9, 10, 11),
}


def _monthly_profile_from_seasons(values: dict[str, float]) -> np.ndarray:
    profile = np.empty(12, dtype=float)
    for season, months in SEASON_TO_MONTHS.items():
        for month in months:
            profile[month - 1] = float(values[season])
    return profile / profile.mean()


def _bundle_year_rows(
    profiles: dict[str, dict[int, np.ndarray]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for service_id, yearly in profiles.items():
        for year, profile in yearly.items():
            for month, value in enumerate(profile, start=1):
                rows.append(
                    {
                        "service_id": service_id,
                        "year": year,
                        "month": month,
                        "visits_modifier_equal_sigungu": float(value),
                    }
                )
    return pd.DataFrame(rows)


def _one_bundle(weights: dict[str, float]) -> dict:
    return {
        "status": "synthetic_test",
        "bundles": [
            {
                "bundle_id": "bundle",
                "bundle_name_ko": "bundle",
                "included_services": weights,
                "compatible_months": list(range(1, 13)),
            }
        ],
    }


def test_bundle_validation_and_release_use_the_same_weighted_profile(
    temporal_config: dict,
) -> None:
    service_a = _monthly_profile_from_seasons(
        {"winter": 0.8, "spring": 1.3, "summer": 0.9, "autumn": 1.0}
    )
    service_b = _monthly_profile_from_seasons(
        {"winter": 0.9, "spring": 1.2, "summer": 0.8, "autumn": 1.1}
    )
    evidence = _bundle_year_rows(
        {
            "a": {2022: service_a, 2023: service_a},
            "b": {2022: service_b, 2023: service_b},
        }
    )
    result = build_bundle_seasonality_from_yearly_evidence(
        evidence, _one_bundle({"a": 0.25, "b": 0.75}), temporal_config
    )
    raw_bundle = 0.25 * service_a + 0.75 * service_b
    raw_bundle = raw_bundle / raw_bundle.mean()
    expected_modifier = 1.0 + 0.5 * (raw_bundle - 1.0)
    released = result.bundle_month.sort_values("month")
    np.testing.assert_allclose(released["seasonality_modifier"], expected_modifier)

    detail = result.reliability_detail.iloc[0]
    released_seasons = released.assign(
        season=released["month"].map(
            {month: season for season, months in SEASON_TO_MONTHS.items() for month in months}
        )
    ).groupby("season")["seasonal_service_fit"].mean()
    expected_order = sorted(
        SEASON_TO_MONTHS,
        key=lambda season: (-released_seasons.loc[season], list(SEASON_TO_MONTHS).index(season)),
    )
    assert detail["released_primary_season"] == expected_order[0]
    assert detail["released_fallback_season"] == expected_order[1]
    assert detail["validation_release_unit_match"] == 1
    assert detail["validation_release_direction_concordant"] == 1
    assert result.summary["validation_release_profile_mismatch_count"] == 0
    assert result.summary["validation_release_unit_match_fraction"] == 1.0


def test_bundle_threshold_tolerance_and_flat_signal_contract(
    temporal_config: dict,
) -> None:
    year_a = _monthly_profile_from_seasons(
        {"winter": 1.0, "spring": 2.0, "summer": 3.0, "autumn": 4.0}
    )
    # Four-season Spearman is exactly 0.4; floating representation must not
    # incorrectly drop it below the preregistered moderate threshold.
    year_b = _monthly_profile_from_seasons(
        {"winter": 2.0, "spring": 3.0, "summer": 1.0, "autumn": 4.0}
    )
    moderate = build_bundle_seasonality_from_yearly_evidence(
        _bundle_year_rows({"a": {2022: year_a, 2023: year_b}}),
        _one_bundle({"a": 1.0}),
        temporal_config,
    )
    row = moderate.reliability_detail.iloc[0]
    assert row["season_rank_spearman"] == pytest.approx(0.4)
    assert row["bundle_reliability_tier"] == "moderate"
    assert row["bundle_reliability_lambda"] == 0.25

    flat = build_bundle_seasonality_from_yearly_evidence(
        _bundle_year_rows({"a": {2022: np.ones(12), 2023: np.ones(12)}}),
        _one_bundle({"a": 1.0}),
        temporal_config,
    )
    flat_detail = flat.reliability_detail.iloc[0]
    assert flat_detail["bundle_reliability_tier"] == "low"
    assert flat_detail["bundle_reliability_lambda"] == 0.0
    assert flat_detail["reliability_decision_reason"] == "no_bundle_level_season_signal"
    assert not bool(flat_detail["temporal_signal_available"])
    np.testing.assert_allclose(flat.bundle_month["seasonal_service_fit"], 50.0)
    assert flat.summary["no_signal_bundle_count"] == 1


def _season_fit_panel(
    season_scores: dict[str, float],
    *,
    regions: tuple[str, ...] = ("A",),
    reliability: float = 0.5,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for region in regions:
        for season, months in SEASON_TO_MONTHS.items():
            for month in months:
                score = float(season_scores[season])
                rows.append(
                    {
                        "admin_dong_code": region,
                        "admin_dong_name": region,
                        "policy_sigungu_name": "g",
                        "bundle_id": "bundle",
                        "bundle_name_ko": "bundle",
                        "month": month,
                        "temporal_fit_score": score,
                        "seasonal_service_fit": score,
                        "seasonality_modifier": 0.85 + score / 100.0 * 0.30,
                        "seasonality_reliability": reliability,
                        "climate_operational_fit": 50.0,
                        "climate_operational_risk": 50.0,
                        "transit_time_fit": 50.0,
                        "contribution__seasonal_service_fit": score,
                    }
                )
    return pd.DataFrame(rows)


def test_season_selection_is_deterministic_and_release_counts_are_enforced(
    temporal_config: dict,
) -> None:
    fit = _season_fit_panel(
        {"winter": 80.0, "spring": 80.0, "summer": 40.0, "autumn": 20.0},
        regions=("A", "B"),
    )
    selected = select_primary_fallback_seasons(fit, temporal_config)
    shuffled = select_primary_fallback_seasons(
        fit.sample(frac=1.0, random_state=123), temporal_config
    )
    pd.testing.assert_frame_equal(selected, shuffled)
    assert selected["primary_season"].eq("winter").all()
    assert selected["fallback_season"].eq("spring").all()

    config = deepcopy(temporal_config)
    config["quality_gates"].update(
        {
            "expected_recommendation_rows": 2,
            "expected_actionable_recommendation_rows": 2,
            "expected_no_signal_recommendation_rows": 0,
            "expected_available_date_specific_replan_trigger_rows": 0,
        }
    )
    released = apply_season_release_contract(
        selected, fit, config, enforce_config_contract=True
    )
    assert released["temporal_signal_available"].all()
    assert released["primary_season_tie_count"].eq(2).all()
    assert released["climate_role"].eq(HISTORICAL_CLIMATE_ROLE).all()
    assert released["replan_trigger_status"].eq(
        UNAVAILABLE_REPLAN_TRIGGER_STATUS
    ).all()
    assert released["date_specific_replan_trigger_available"].eq(0).all()

    wrong = deepcopy(config)
    wrong["quality_gates"]["expected_actionable_recommendation_rows"] = 1
    with pytest.raises(ValueError, match="Season release contract"):
        apply_season_release_contract(
            selected, fit, wrong, enforce_config_contract=True
        )


def test_flat_bundle_does_not_release_arbitrary_tied_seasons(
    temporal_config: dict,
) -> None:
    fit = _season_fit_panel(
        {"winter": 50.0, "spring": 50.0, "summer": 50.0, "autumn": 50.0}
    )
    diagnostic = select_primary_fallback_seasons(fit, temporal_config)
    assert diagnostic.loc[0, "primary_season"] == "winter"
    assert diagnostic.loc[0, "fallback_season"] == "spring"

    config = deepcopy(temporal_config)
    config["quality_gates"].update(
        {
            "expected_recommendation_rows": 1,
            "expected_actionable_recommendation_rows": 0,
            "expected_no_signal_recommendation_rows": 1,
            "expected_available_date_specific_replan_trigger_rows": 0,
        }
    )
    released = apply_season_release_contract(
        diagnostic, fit, config, enforce_config_contract=True
    )
    assert not bool(released.loc[0, "temporal_signal_available"])
    assert released.loc[0, "release_status"] == (
        "unavailable_no_reliable_bundle_season_signal"
    )
    assert pd.isna(released.loc[0, "primary_season"])
    assert pd.isna(released.loc[0, "fallback_season"])
    assert pd.isna(released.loc[0, "fallback_distinct_from_primary"])
