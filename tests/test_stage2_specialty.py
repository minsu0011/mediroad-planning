from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mediroad.scoring.transform import scale_need_direction  # noqa: E402
from mediroad.specialty import (  # noqa: E402
    COMPONENT_COLUMNS,
    FORMULA_COLUMNS,
    build_bundle_scores,
    build_specialty_gap,
    build_specialty_gap_from_root,
    combine_components,
    empirical_bayes_share,
    load_service_bundle_config,
    load_specialty_config,
    specialty_access_excess,
    validate_specialty_config,
    validate_specialty_results,
)


@pytest.fixture(scope="module")
def specialty_config() -> dict:
    return load_specialty_config(ROOT / "configs/model_v1/specialty_gap.yaml")


@pytest.fixture(scope="module")
def bundle_config() -> dict:
    return load_service_bundle_config(ROOT / "configs/model_v1/service_bundles.yaml")


@pytest.fixture(scope="module")
def specialty_scores(specialty_config: dict) -> pd.DataFrame:
    return build_specialty_gap_from_root(ROOT)


@pytest.fixture(scope="module")
def bundle_scores(specialty_scores: pd.DataFrame, bundle_config: dict) -> pd.DataFrame:
    return build_bundle_scores(specialty_scores, bundle_config)


def test_stage2a_config_freezes_15_focused_plus_dental_without_score_leakage(
    specialty_config: dict,
) -> None:
    result = validate_specialty_config(specialty_config)
    assert result["valid"], result["errors"]
    assert result["service_count"] == 16
    assert result["focused_specialty_count"] == 15
    services = specialty_config["services"]
    assert [value["service_id"] for value in services].count("dental") == 1
    assert specialty_config["model_contract"]["use_stage1_need_as_specialty_input"] is False
    assert specialty_config["model_contract"]["use_legacy_v0_as_input"] is False


def test_actual_hira_aggregation_has_complete_unique_153_by_16_key(
    specialty_scores: pd.DataFrame,
    specialty_config: dict,
) -> None:
    expected_rows = specialty_config["quality_gates"]["expected_specialty_rows"]
    assert specialty_scores.shape[0] == expected_rows == 153 * 16
    assert specialty_scores["admin_dong_code"].nunique() == 153
    assert specialty_scores["service_id"].nunique() == 16
    assert not specialty_scores.duplicated(["admin_dong_code", "service_id"]).any()
    assert specialty_scores.groupby("service_id").size().eq(153).all()
    assert specialty_scores["admin_dong_code"].str.fullmatch(r"\d{8}").all()
    assert int(
        specialty_scores.drop_duplicates(["admin_dong_code", "service_id"])[
            "local_specialty_specialist_count"
        ].isna().sum()
    ) == 0

    # The dental specialist count is taken from the HIRA facility staffing
    # table, while focused physician counts come from the specialty pivot.
    facility = pd.read_csv(
        ROOT / specialty_config["source"]["hira_facility_admin_join"], low_memory=False
    )
    dental = specialty_scores[specialty_scores["service_id"].eq("dental")]
    assert dental["local_specialty_specialist_count"].sum() == pytest.approx(
        pd.to_numeric(facility["dental_specialists"], errors="coerce").fillna(0).sum()
    )
    assert dental["local_specialty_facility_count"].sum() == pytest.approx(488.0)


def test_chs_and_sigungu_supply_are_scaled_at_n11_then_broadcast(
    specialty_scores: pd.DataFrame,
) -> None:
    assert specialty_scores["policy_sigungu_name"].nunique() == 11
    # Both fields are policy-sigungu context. They must be invariant within a
    # service × sigungu block rather than fitted on 153 repeated rows.
    for column in (
        "health_context_gap_score",
        "sigungu_specialist_rate_gap_score",
        "sigungu_specialists_per_1000_elderly",
    ):
        unique_within = specialty_scores.groupby(["service_id", "policy_sigungu_name"])[
            column
        ].nunique(dropna=False)
        assert unique_within.eq(1).all(), column


def test_common_access_and_specialty_excess_are_separate(
    specialty_scores: pd.DataFrame,
) -> None:
    # General access is a diagnostic common layer and may not replace the
    # specialty-excess component used in the Stage 2A formula.
    common_unique = specialty_scores.groupby("admin_dong_code")[
        "general_medical_access_gap_score"
    ].nunique()
    assert common_unique.eq(1).all()
    assert "general_medical_access_gap_score" not in COMPONENT_COLUMNS
    assert "specialty_excess_access_gap_score" in COMPONENT_COLUMNS
    assert (specialty_scores["specialty_access_excess_log1p"] >= 0).all()


def test_zero_total_supply_has_neutral_composition_gap(
    specialty_scores: pd.DataFrame,
) -> None:
    no_facilities = specialty_scores["local_total_medical_facility_count"].eq(0)
    no_specialists = specialty_scores["local_total_specialist_count"].eq(0)
    assert no_facilities.any()
    assert no_specialists.any()
    assert specialty_scores.loc[no_facilities, "facility_share_gap_score"].eq(50.0).all()
    assert specialty_scores.loc[no_specialists, "specialist_share_gap_score"].eq(50.0).all()


def test_four_formulas_contributions_and_scores_are_complete(
    specialty_scores: pd.DataFrame,
    specialty_config: dict,
) -> None:
    formula_columns = list(FORMULA_COLUMNS.values())
    assert specialty_scores[formula_columns].notna().all().all()
    assert np.isfinite(specialty_scores[formula_columns].to_numpy()).all()
    assert specialty_scores[formula_columns].ge(0).all().all()
    assert specialty_scores[formula_columns].le(100).all().all()
    assert np.allclose(
        specialty_scores["specialty_gap_score"],
        specialty_scores[FORMULA_COLUMNS[specialty_config["primary_formula"]]],
    )

    contribution_columns = [
        f"contribution_{column.removesuffix('_score')}" for column in COMPONENT_COLUMNS
    ]
    assert np.allclose(
        specialty_scores[contribution_columns].sum(axis=1),
        specialty_scores["gap_score_weighted_additive"],
        atol=1e-10,
    )


def test_supply_access_and_formula_directionality_is_monotone() -> None:
    denominator = pd.Series([10.0, 10.0, 10.0, 10.0])
    supply = empirical_bayes_share(
        pd.Series([0.0, 1.0, 3.0, 6.0]), denominator, prior_strength=5.0
    )
    assert supply.is_monotonic_increasing
    supply_gap = scale_need_direction(
        supply,
        method="rank_percentile",
        direction=-1,
        require_complete=True,
    )
    assert supply_gap.is_monotonic_decreasing

    access = specialty_access_excess(
        pd.Series([5.0, 10.0, 20.0]), pd.Series([5.0, 5.0, 5.0])
    )
    assert access.is_monotonic_increasing

    weights = {column: 0.2 for column in COMPONENT_COLUMNS}
    base = pd.DataFrame(
        {
            column: [20.0, 50.0, 80.0]
            for column in COMPONENT_COLUMNS
        }
    )
    increased = base.copy()
    increased.loc[1, COMPONENT_COLUMNS[0]] += 20.0
    formula_options = {
        "weighted_additive": {},
        "bounded_geometric": {"floor": 5.0},
        "minimum_bottleneck_hybrid": {"additive_share": 0.7, "minimum_share": 0.3},
        "rank_composite": {},
    }
    for formula, options in formula_options.items():
        before = combine_components(base, weights, formula=formula, formula_options=options)
        after = combine_components(increased, weights, formula=formula, formula_options=options)
        assert after.iloc[1] >= before.iloc[1], formula


def test_prior_strength_override_supports_declared_sensitivity_grid(
    specialty_config: dict,
) -> None:
    scores = build_specialty_gap_from_root(
        ROOT,
        facility_prior_strength=2.0,
        specialist_prior_strength=10.0,
    )
    assert scores["facility_share_prior_strength"].eq(2.0).all()
    assert scores["specialist_share_prior_strength"].eq(10.0).all()
    assert scores.shape[0] == specialty_config["quality_gates"]["expected_specialty_rows"]


def test_recommended_bundle_aggregation_has_complete_153_by_5_key(
    bundle_scores: pd.DataFrame,
    bundle_config: dict,
) -> None:
    assert len(bundle_scores) == 153 * 5
    assert bundle_scores["bundle_id"].nunique() == 5
    assert not bundle_scores.duplicated(["admin_dong_code", "bundle_id"]).any()
    assert bundle_scores["bundle_gap_score"].notna().all()
    assert bundle_scores["bundle_gap_score"].between(0, 100).all()
    excluded = {
        value["service_id"]
        for value in bundle_config["excluded_from_mobile_bundles"]
    }
    included = {
        service
        for bundle in bundle_config["bundles"]
        for service in bundle["included_services"]
    }
    assert excluded.isdisjoint(included)


def test_result_gate_checks_key_contributions_canonical_and_leakage(
    specialty_scores: pd.DataFrame,
    specialty_config: dict,
    bundle_scores: pd.DataFrame,
    bundle_config: dict,
) -> None:
    result = validate_specialty_results(
        specialty_scores,
        specialty_config,
        bundle_scores=bundle_scores,
        bundle_config=bundle_config,
        canonical_path=ROOT / "derived/mediroad_admin_dong_master_v6_final.csv",
    )
    assert result["valid"], [key for key, value in result["checks"].items() if not value]
    assert result["leaked_columns"] == []
    assert result["exact_duplicate_service_score_pairs"] == []
    assert result["canonical_sha256"] == specialty_config["source"][
        "frozen_canonical_sha256"
    ]
    forbidden = (
        "mediroad_need_score_v0",
        "recommended_specialties_v0",
        "recommended_specialty_scores_v0",
        "stage1_need_score",
        "legacy_gap",
    )
    assert not any(any(token in column for token in forbidden) for column in specialty_scores)


def test_build_is_invariant_to_source_row_order(specialty_config: dict) -> None:
    source = specialty_config["source"]
    master = pd.read_csv(
        ROOT / source["road_master"], dtype={"admin_dong_code": "string"}, low_memory=False
    )
    pivot = pd.read_csv(ROOT / source["hira_specialty_pivot"], low_memory=False)
    admin = pd.read_csv(
        ROOT / source["hira_facility_admin_join"],
        dtype={"admin_dong_code": "string"},
        low_memory=False,
    )
    shuffled = build_specialty_gap(
        master.sample(frac=1.0, random_state=17),
        pivot.sample(frac=1.0, random_state=18),
        admin.sample(frac=1.0, random_state=19),
        specialty_config,
    )
    baseline = build_specialty_gap_from_root(ROOT)
    key = ["service_id", "admin_dong_code"]
    columns = [*key, "specialty_gap_score", "specialty_gap_rank"]
    pd.testing.assert_frame_equal(
        baseline[columns].sort_values(key).reset_index(drop=True),
        shuffled[columns].sort_values(key).reset_index(drop=True),
        check_exact=True,
    )
