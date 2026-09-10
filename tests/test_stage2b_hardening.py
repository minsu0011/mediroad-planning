from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from mediroad.temporal.hardening import (
    CONFIDENCE_LEVELS,
    RELEASE_TYPES,
    build_cross_year_consensus,
    leave_one_month_out,
    leave_one_sigungu_out,
    month_label_permutation_null,
    prepare_temporal_hardening_evidence,
    run_temporal_hardening,
    season_boundary_sensitivity,
    sigungu_block_bootstrap,
    strict_cross_year_transfer,
    working_day_normalization_sensitivity,
)


@pytest.fixture
def hardening_bundle_config() -> dict:
    return {
        "status": "synthetic",
        "bundles": [
            {
                "bundle_id": "stable_bundle",
                "bundle_name_ko": "stable",
                "included_services": {"service_a": 1.0},
            },
            {
                "bundle_id": "mixed_bundle",
                "bundle_name_ko": "mixed",
                "included_services": {"service_a": 0.4, "service_b": 0.6},
            },
        ],
    }


@pytest.fixture
def hardening_calendar() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "year": year,
                "month": month,
                "calendar_days": 30 + int(month in {1, 3, 5, 7, 8, 10, 12}),
                "working_days": 19 + (month % 4),
            }
            for year in (2022, 2023)
            for month in range(1, 13)
        ]
    )


@pytest.fixture
def hardening_nhis(hardening_calendar: pd.DataFrame) -> pd.DataFrame:
    exposure = hardening_calendar.set_index(["year", "month"])["working_days"]
    season_a = np.array([0.80, 0.85, 1.40, 1.35, 1.30, 0.90, 0.85, 0.80, 1.05, 1.00, 0.95, 0.75])
    season_b_2022 = np.array([0.85, 0.80, 0.90, 0.95, 1.00, 0.90, 0.85, 0.80, 1.35, 1.40, 1.30, 0.90])
    season_b_2023 = np.array([1.35, 1.30, 0.85, 0.90, 0.95, 0.85, 0.80, 0.75, 1.00, 1.05, 1.10, 1.40])
    rows: list[dict[str, object]] = []
    for sigungu_index, sigungu in enumerate(("g1", "g2", "g3"), start=1):
        for year in (2022, 2023):
            for service_id in ("service_a", "service_b"):
                profile = season_a if service_id == "service_a" else (
                    season_b_2022 if year == 2022 else season_b_2023
                )
                for month, modifier in enumerate(profile, start=1):
                    visits = float(exposure.loc[(year, month)] * 100 * sigungu_index * modifier)
                    rows.append(
                        {
                            "year": year,
                            "month": month,
                            "policy_sigungu_name": sigungu,
                            "service_id": service_id,
                            "persons": visits * 0.7,
                            "visits": visits,
                        }
                    )
    return pd.DataFrame(rows)


@pytest.fixture
def prepared(hardening_nhis, hardening_calendar, hardening_bundle_config):
    return prepare_temporal_hardening_evidence(
        hardening_nhis,
        hardening_calendar,
        hardening_bundle_config,
        expected_policy_sigungu=3,
    )


def test_evidence_declares_real_independent_n_and_normalization(prepared) -> None:
    assert prepared.audit["temporal_independent_n"] == 6
    assert prepared.audit["admin_dong_delivery_rows_are_independent_samples"] is False
    assert prepared.audit["observed_service_count"] == 2
    assert len(prepared.service_month) == 3 * 2 * 2 * 12
    assert len(prepared.bundle_month) == 3 * 2 * 2 * 12
    np.testing.assert_allclose(
        prepared.service_month["working_day_adjusted"],
        prepared.service_month["year_normalized_working_day"],
        atol=1e-12,
    )
    assert not prepared.bundle_month.duplicated(
        ["policy_sigungu_name", "bundle_id", "year", "month"]
    ).any()


def test_cross_year_is_strict_and_holdout_cannot_change_fit(prepared) -> None:
    baseline = strict_cross_year_transfer(prepared.bundle_month)
    assert len(baseline.detail) == 2 * 6
    assert baseline.detail["temporal_independent_n"].eq(6).all()
    assert baseline.detail["fit_holdout_overlap_row_count"].eq(0).all()
    assert baseline.detail["leakage_guard_pass"].all()
    assert len(baseline.summary) == 2 * 2

    changed = prepared.bundle_month.copy()
    changed.loc[changed["year"].eq(2023) & changed["month"].eq(7), "working_day_adjusted"] *= 100
    rerun = strict_cross_year_transfer(changed)
    left = baseline.detail.query("train_year == 2022").sort_values(
        ["policy_sigungu_name", "bundle_id"]
    )
    right = rerun.detail.query("train_year == 2022").sort_values(
        ["policy_sigungu_name", "bundle_id"]
    )
    assert left["train_primary"].tolist() == right["train_primary"].tolist()
    assert left["train_fallback"].tolist() == right["train_fallback"].tolist()

    leaked = pd.concat(
        [prepared.bundle_month, prepared.bundle_month.assign(year=2024)], ignore_index=True
    )
    with pytest.raises(ValueError, match="leakage guard"):
        strict_cross_year_transfer(leaked)
    with pytest.raises(ValueError, match="exactly two distinct"):
        strict_cross_year_transfer(prepared.bundle_month, years=(2022, 2022))


def test_consensus_lomo_loso_sensitivity_and_boundaries_are_complete(prepared) -> None:
    consensus = build_cross_year_consensus(prepared.bundle_month)
    assert len(consensus) == 2 * 4
    for rank_column in (
        "mean_rank",
        "median_rank",
        "borda_rank",
        "worst_year_rank",
        "cross_year_consensus_rank",
    ):
        assert consensus.groupby("bundle_id")[rank_column].apply(
            lambda values: set(values) == {1, 2, 3, 4}
        ).all()

    lomo = leave_one_month_out(prepared.bundle_month)
    scenarios = lomo.detail[["year", "removed_month"]].drop_duplicates()
    assert len(scenarios) == 24
    assert scenarios.groupby("year").size().eq(12).all()
    assert len(lomo.detail) == 24 * 2
    assert lomo.summary["lomo_unique_removals"].eq(24).all()

    loso = leave_one_sigungu_out(prepared.bundle_month)
    assert loso.detail["removed_sigungu"].nunique() == 3
    assert len(loso.detail) == 3 * 2
    assert loso.detail["training_sigungu_n"].eq(2).all()
    assert loso.detail["excluded_block_includes_all_months_years_services"].all()

    sensitivity = working_day_normalization_sensitivity(prepared.bundle_month)
    assert set(sensitivity["variant"]) >= {
        "working_day_adjusted",
        "raw_monthly_utilization",
        "year_normalized_working_day",
    }
    assert sensitivity["season_rank_spearman"].between(-1, 1).all()
    assert sensitivity["top2_agreement"].between(0, 1).all()

    boundaries = season_boundary_sensitivity(prepared.bundle_month)
    assert boundaries.detail["definition"].nunique() == 3
    assert len(boundaries.detail) == 3 * 2
    assert boundaries.detail["diagnostic_only_primary_definition_unchanged"].all()


def test_bootstrap_and_permutation_counts_and_parallel_determinism(
    prepared, hardening_bundle_config
) -> None:
    serial_boot = sigungu_block_bootstrap(
        prepared.bundle_month, n_boot=17, expected_n_boot=17, random_state=123, n_jobs=1
    )
    parallel_boot = sigungu_block_bootstrap(
        prepared.bundle_month, n_boot=17, expected_n_boot=17, random_state=123, n_jobs=2
    )
    pd.testing.assert_frame_equal(serial_boot.detail, parallel_boot.detail)
    assert serial_boot.detail["bootstrap_replicate"].nunique() == 17
    assert serial_boot.summary["n_valid"].eq(17).all()
    assert serial_boot.detail["complete_block_resampling"].all()
    with pytest.raises(ValueError, match="does not match"):
        sigungu_block_bootstrap(prepared.bundle_month, n_boot=16, expected_n_boot=17)

    serial_null = month_label_permutation_null(
        prepared.service_month,
        hardening_bundle_config,
        n_permutations=23,
        expected_n_permutations=23,
        random_state=456,
        n_jobs=1,
    )
    parallel_null = month_label_permutation_null(
        prepared.service_month,
        hardening_bundle_config,
        n_permutations=23,
        expected_n_permutations=23,
        random_state=456,
        n_jobs=2,
    )
    pd.testing.assert_frame_equal(serial_null.detail, parallel_null.detail)
    assert serial_null.detail["permutation_replicate"].nunique() == 23
    assert serial_null.summary["n_valid"].eq(23).all()
    assert serial_null.detail["annual_totals_preserved"].all()
    with pytest.raises(ValueError, match="does not match"):
        month_label_permutation_null(
            prepared.service_month,
            hardening_bundle_config,
            n_permutations=22,
            expected_n_permutations=23,
        )


def test_complete_runner_supports_explicit_abstention(
    hardening_nhis, hardening_calendar, hardening_bundle_config
) -> None:
    impossible = {
        "strong_single": {
            "cross_year_top1_agreement_min": 2.0,
            "bidirectional_primary_in_holdout_top2_min": 2.0,
            "bidirectional_selected_pair_holdout_top2_coverage_min": 2.0,
            "heldout_normalized_regret_max": -1.0,
            "pooled_normalized_top1_margin_min": 2.0,
            "year_normalized_top1_margin_min": 2.0,
            "bootstrap_primary_probability_min": 2.0,
            "lomo_primary_retention_min": 2.0,
            "loso_primary_retention_min": 2.0,
            "working_day_primary_retention_min": 2.0,
            "boundary_primary_retention_min": 2.0,
            "permutation_margin_percentile_min": 2.0,
        },
        "robust_pair": {
            "bidirectional_primary_in_holdout_top2_min": 2.0,
            "bidirectional_selected_pair_holdout_top2_coverage_min": 2.0,
            "year_top2_jaccard_min": 2.0,
            "heldout_normalized_regret_max": -1.0,
            "pooled_normalized_top1_margin_min": 2.0,
            "bootstrap_pair_coverage_probability_min": 2.0,
            "lomo_top2_retention_min": 2.0,
            "loso_top2_retention_min": 2.0,
            "working_day_top2_retention_min": 2.0,
            "boundary_top2_retention_min": 2.0,
            "permutation_margin_percentile_min": 2.0,
        },
    }
    result = run_temporal_hardening(
        hardening_nhis,
        hardening_calendar,
        hardening_bundle_config,
        n_boot=9,
        n_permutations=11,
        expected_policy_sigungu=3,
        random_state=789,
        n_jobs=2,
        release_thresholds=impossible,
    )
    summary = result.bundle_evidence_summary
    assert set(summary["release_type"]).issubset(RELEASE_TYPES)
    assert set(summary["confidence"]).issubset(CONFIDENCE_LEVELS)
    assert summary["release_type"].eq("NO_STRONG_PREFERENCE").all()
    assert summary[["final_primary", "final_fallback", "exact_month", "exact_date"]].isna().all().all()
    assert summary["temporal_independent_n"].eq(6).all()
