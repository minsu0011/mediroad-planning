from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mediroad.temporal.hardening_integration import (  # noqa: E402
    STAGE2_TO_STAGE3_COLUMNS,
    analyze_bundle_differentiation,
    build_stage2_to_stage3_interface,
    capture_frozen_integrity,
    classify_confidence_aware_release,
    compare_frozen_integrity,
    compare_what_when_models,
    dataframe_fingerprint,
    diagnose_need_replication,
    diagnose_temporal_spatial_leakage,
    file_sha256,
    stage2_to_stage3_schema,
    validate_confidence_release,
    validate_stage2_to_stage3_interface,
)


@pytest.fixture(scope="module")
def hardening_config() -> dict:
    return yaml.safe_load(
        (ROOT / "configs/model_v1/stage2b_hardening.yaml").read_text(encoding="utf-8")
    )


@pytest.fixture(scope="module")
def actual_bundle_scores() -> pd.DataFrame:
    return pd.read_parquet(
        ROOT / "outputs/model_v1/04_specialty/recommended_bundle_gap_scores.parquet"
    )


@pytest.fixture(scope="module")
def actual_season_scores() -> pd.DataFrame:
    return pd.read_parquet(
        ROOT / "outputs/model_v1/04_temporal/region_bundle_season_scores.parquet"
    )


@pytest.fixture(scope="module")
def actual_recommendations() -> pd.DataFrame:
    return pd.read_csv(
        ROOT / "outputs/model_v1/04_temporal/region_bundle_season_recommendations.csv",
        dtype={"admin_dong_code": str},
    )


@pytest.fixture(scope="module")
def actual_stage1_scores() -> pd.DataFrame:
    return pd.read_csv(
        ROOT / "outputs/model_v1/03_ablation/stage1_primary_need_score.csv",
        dtype={"admin_dong_code": str},
    )


def test_actual_hierarchical_what_then_when_is_exact_and_joint_is_diagnostic(
    actual_bundle_scores: pd.DataFrame,
    actual_season_scores: pd.DataFrame,
    hardening_config: dict,
) -> None:
    lambdas = hardening_config["analysis_contract"]["bounded_temporal_lambdas"]
    result = compare_what_when_models(
        actual_bundle_scores,
        actual_season_scores,
        lambdas=lambdas,
    )
    assert set(result.summary["model"]) == {
        "hierarchical_WHAT_then_WHEN",
        "joint_gap_times_temporal_fit",
        "bounded_lambda_0.10",
        "bounded_lambda_0.20",
        "bounded_lambda_0.30",
    }
    hierarchy = result.summary.set_index("model").loc["hierarchical_WHAT_then_WHEN"]
    assert hierarchy["top1_bundle_retention"] == 1.0
    assert hierarchy["top2_bundle_recall"] == 1.0
    assert hierarchy["mean_bundle_rank_spearman"] == pytest.approx(1.0)
    assert hierarchy["mean_normalized_rank_distortion"] == 0.0
    assert bool(hierarchy["hierarchical_what_invariant"])
    assert bool(hierarchy["release_eligible"])

    joint = result.summary.set_index("model").loc["joint_gap_times_temporal_fit"]
    assert joint["top1_bundle_retention"] < 0.90
    assert joint["mean_bundle_rank_spearman"] < hardening_config["advisory_thresholds"][
        "joint_model_bundle_rank_spearman_review"
    ]
    assert not bool(joint["release_eligible"])

    thresholds = hardening_config["integration_contract"]["specialty_intrusion"]
    for suffix, threshold_name in (
        ("0.10", "bounded_lambda_010_top1_retention_min"),
        ("0.20", "bounded_lambda_020_top1_retention_min"),
        ("0.30", "bounded_lambda_030_top1_retention_min"),
    ):
        observed = result.summary.set_index("model").loc[
            f"bounded_lambda_{suffix}", "top1_bundle_retention"
        ]
        assert observed >= thresholds[threshold_name]


def test_actual_temporal_delivery_is_not_spatial_pseudoreplication(
    actual_season_scores: pd.DataFrame,
    actual_recommendations: pd.DataFrame,
) -> None:
    result = diagnose_temporal_spatial_leakage(
        actual_season_scores,
        actual_recommendations,
    )
    summary = result.summary.iloc[0]
    assert summary["delivery_rows"] == 153 * 5
    assert summary["independent_temporal_n"] == 11 * 5 == 55
    assert not bool(summary["broadcast_rows_claimed_as_independent"])
    assert summary["unexpected_spatial_variation_count"] == 0
    assert not bool(summary["leakage_flag"])
    assert result.bundle_summary["unique_temporal_profiles"].eq(1).all()


def test_spatially_different_profile_without_evidence_fails_leakage_contract() -> None:
    scores = pd.DataFrame(
        [
            {
                "admin_dong_code": admin,
                "policy_sigungu_name": "S",
                "bundle_id": "b",
                "season": season,
                "temporal_fit_score": score + (10 if admin == "B" and season == "winter" else 0),
            }
            for admin in ("A", "B")
            for season, score in zip(
                ("spring", "summer", "autumn", "winter"),
                (60, 50, 40, 30),
                strict=True,
            )
        ]
    )
    recs = pd.DataFrame(
        {
            "admin_dong_code": ["A", "B"],
            "policy_sigungu_name": ["S", "S"],
            "bundle_id": ["b", "b"],
            "primary_season": ["spring", "spring"],
            "fallback_season": ["summer", "summer"],
            "region_specific_clinical_seasonality_used": [0, 0],
        }
    )
    result = diagnose_temporal_spatial_leakage(scores, recs)
    assert bool(result.summary.iloc[0]["leakage_flag"])
    assert bool(result.bundle_summary.iloc[0]["unsupported_spatial_variation"])


def test_actual_bundle_differentiation_is_measured_at_correct_levels(
    actual_bundle_scores: pd.DataFrame,
) -> None:
    result = analyze_bundle_differentiation(actual_bundle_scores)
    summary = result.summary.iloc[0]
    assert summary["admin_count"] == 153
    assert summary["bundle_count"] == 5
    assert summary["policy_sigungu_count"] == 11
    assert result.admin_summary.shape[0] == 153
    assert result.top_frequency["top1_count"].sum() == 153
    assert result.top_frequency["top2_position_count"].sum() == 153
    assert result.variance_decomposition.shape[0] == 5
    np.testing.assert_allclose(
        result.variance_decomposition["between_sigungu_variance_share"]
        + result.variance_decomposition["within_sigungu_variance_share"],
        1.0,
    )
    assert 0 <= summary["top1_choice_entropy"] <= 1
    assert 0 < summary["dominant_top1_fraction"] < 1
    assert bool(summary["all_bundles_ever_top1"])


def test_actual_need_replication_retains_bundle_specific_residual_signal(
    actual_stage1_scores: pd.DataFrame,
    actual_bundle_scores: pd.DataFrame,
    hardening_config: dict,
) -> None:
    result = diagnose_need_replication(actual_stage1_scores, actual_bundle_scores)
    assert result.bundle_summary.shape[0] == 5
    assert result.partial_correlations.shape[0] == 10
    assert result.residual_panel.shape[0] == 153 * 5
    assert result.bundle_summary["abs_stage1_need_bundle_gap_spearman"].max() < (
        hardening_config["advisory_thresholds"]["need_bundle_abs_spearman_review"]
    )
    assert result.bundle_summary["rank_variance_not_explained_by_need"].min() > 0.35
    assert result.bundle_summary["need_controlled_residual_sd"].gt(0).all()
    assert result.bundle_summary["residual_is_diagnostic_only"].all()


def _release_evidence_fixture() -> pd.DataFrame:
    common = {
        "bidirectional_primary_in_holdout_top2": 1.0,
        "bidirectional_selected_pair_holdout_top2_coverage": 1.0,
        "heldout_normalized_regret": 0.10,
        "bootstrap_pair_coverage_probability": 0.90,
        "lomo_top2_retention": 0.90,
        "loso_top2_retention": 0.90,
        "working_day_top2_retention": 1.0,
        "boundary_top2_retention": 1.0,
        "year_top2_jaccard": 1.0,
        "permutation_margin_percentile": 0.95,
    }
    strong = {
        **common,
        "bundle_id": "strong",
        "primary_season": "spring",
        "fallback_season": "winter",
        "cross_year_top1_agreement": 1.0,
        "pooled_normalized_top1_margin": 0.20,
        "year_normalized_top1_margin": 0.10,
        "bootstrap_primary_probability": 0.90,
        "lomo_primary_retention": 0.95,
        "loso_primary_retention": 0.90,
        "working_day_primary_retention": 1.0,
        "boundary_primary_retention": 1.0,
        "primary_temporal_fit_score": 70.0,
    }
    pair = {
        **strong,
        "bundle_id": "pair",
        "cross_year_top1_agreement": 0.0,
        "pooled_normalized_top1_margin": 0.05,
        "year_normalized_top1_margin": 0.0,
        "bootstrap_primary_probability": 0.50,
        "lomo_primary_retention": 0.70,
        "loso_primary_retention": 0.70,
    }
    abstain = {
        **pair,
        "bundle_id": "abstain",
        "bidirectional_primary_in_holdout_top2": 0.0,
        "bidirectional_selected_pair_holdout_top2_coverage": 0.0,
        "heldout_normalized_regret": 0.90,
        "bootstrap_pair_coverage_probability": 0.20,
        "lomo_top2_retention": 0.20,
        "loso_top2_retention": 0.20,
    }
    return pd.DataFrame([strong, pair, abstain])


def test_confidence_release_is_preregistered_and_fail_closed(hardening_config: dict) -> None:
    release = classify_confidence_aware_release(
        _release_evidence_fixture(),
        hardening_config["release_rules"],
    ).set_index("bundle_id")
    assert release.loc["strong", "release_type"] == "STRONG_SINGLE"
    assert release.loc["strong", "temporal_confidence"] == "HIGH"
    assert release.loc["pair", "release_type"] == "ROBUST_PAIR"
    assert release.loc["pair", "temporal_confidence"] == "MODERATE"
    assert release.loc["abstain", "release_type"] == "NO_STRONG_PREFERENCE"
    assert release.loc["abstain", "temporal_confidence"] == "LOW"
    assert pd.isna(release.loc["abstain", "primary_season"])
    assert pd.isna(release.loc["abstain", "fallback_season"])
    assert release[["exact_month", "exact_date"]].isna().all().all()
    assert validate_confidence_release(release.reset_index())["passed"].all()

    missing_metric = _release_evidence_fixture().drop(columns="permutation_margin_percentile")
    fail_closed = classify_confidence_aware_release(
        missing_metric,
        hardening_config["release_rules"],
    )
    assert fail_closed["release_type"].eq("NO_STRONG_PREFERENCE").all()
    assert fail_closed[["primary_season", "fallback_season"]].isna().all().all()


def test_stage2_to_stage3_interface_is_schema_only_and_blocks_false_precision(
    hardening_config: dict,
) -> None:
    stage1 = pd.DataFrame(
        {"admin_dong_code": ["A", "B"], "need_score": [80.0, 40.0]}
    )
    bundles = pd.DataFrame(
        [
            {"admin_dong_code": admin, "bundle_id": bundle, "bundle_gap_score": score}
            for admin, scores in (("A", (70.0, 60.0, 50.0)), ("B", (30.0, 40.0, 50.0)))
            for bundle, score in zip(("strong", "pair", "abstain"), scores, strict=True)
        ]
    )
    release = classify_confidence_aware_release(
        _release_evidence_fixture(), hardening_config["release_rules"]
    )
    interface = build_stage2_to_stage3_interface(
        stage1,
        bundles,
        release,
        temporal_independent_n=3,
        expected_admins=2,
        expected_bundles=3,
    )
    assert tuple(interface.columns) == STAGE2_TO_STAGE3_COLUMNS
    assert len(interface) == 6
    assert interface[["exact_month", "exact_date"]].isna().all().all()
    abstained = interface["release_type"].eq("NO_STRONG_PREFERENCE")
    assert interface.loc[abstained, ["recommended_season", "fallback_season"]].isna().all().all()
    gates = validate_stage2_to_stage3_interface(
        interface,
        expected_admins=2,
        expected_bundles=3,
        expected_independent_n=3,
    )
    assert gates["passed"].all(), gates.loc[~gates["passed"]].to_dict("records")
    assert stage2_to_stage3_schema()["stage3_execution"].eq(False).all()  # noqa: E712

    # The production hardening evidence table uses final_primary/final_fallback
    # and confidence.  The interface builder accepts that exact schema and can
    # resolve its diagnostic temporal score from the four-season table.
    hardening_style = release.rename(
        columns={
            "primary_season": "final_primary",
            "fallback_season": "final_fallback",
            "temporal_confidence": "confidence",
        }
    ).drop(columns=["recommended_season", "primary_temporal_fit_score"])
    season_scores = pd.DataFrame(
        [
            {"bundle_id": bundle, "season": season, "temporal_fit_score": score}
            for bundle in ("strong", "pair", "abstain")
            for season, score in zip(
                ("spring", "summer", "autumn", "winter"),
                (70.0, 50.0, 40.0, 60.0),
                strict=True,
            )
        ]
    )
    aliased = build_stage2_to_stage3_interface(
        stage1,
        bundles,
        hardening_style,
        season_scores=season_scores,
        temporal_independent_n=3,
        expected_admins=2,
        expected_bundles=3,
    )
    assert tuple(aliased.columns) == STAGE2_TO_STAGE3_COLUMNS
    assert aliased["temporal_fit_score"].eq(70.0).all()

    corrupted = interface.copy()
    corrupted.loc[0, "exact_month"] = 3
    failed = validate_stage2_to_stage3_interface(
        corrupted,
        expected_admins=2,
        expected_bundles=3,
        expected_independent_n=3,
    ).set_index("gate")
    assert not bool(failed.loc["interface_exact_month_null", "passed"])


def test_frozen_hash_and_semantic_fingerprint_contract(
    actual_stage1_scores: pd.DataFrame,
    actual_bundle_scores: pd.DataFrame,
    hardening_config: dict,
) -> None:
    paths = hardening_config["paths"]
    expected = hardening_config["frozen_contract"]
    artifact_paths = {
        "stage1_need_scores_file": ROOT / paths["stage1_need_scores"],
        "stage2a_bundle_scores_file": ROOT / paths["stage2a_bundle_scores"],
        "stage2a_specialty_scores_file": ROOT / paths["stage2a_specialty_scores"],
    }
    assert file_sha256(artifact_paths["stage1_need_scores_file"]) == expected[
        "stage1_need_scores_sha256"
    ]
    assert file_sha256(artifact_paths["stage2a_bundle_scores_file"]) == expected[
        "stage2a_bundle_scores_sha256"
    ]
    assert file_sha256(artifact_paths["stage2a_specialty_scores_file"]) == expected[
        "stage2a_specialty_scores_sha256"
    ]

    before = capture_frozen_integrity(
        actual_stage1_scores,
        actual_bundle_scores,
        artifact_paths,
    )
    reordered = capture_frozen_integrity(
        actual_stage1_scores.sample(frac=1.0, random_state=7),
        actual_bundle_scores.sample(frac=1.0, random_state=7),
        artifact_paths,
    )
    comparison = compare_frozen_integrity(before, reordered)
    assert comparison["unchanged"].all()
    assert dataframe_fingerprint(
        actual_stage1_scores,
        key_columns=["admin_dong_code"],
        columns=["need_score"],
    ) == dataframe_fingerprint(
        actual_stage1_scores.iloc[::-1],
        key_columns=["admin_dong_code"],
        columns=["need_score"],
    )

    changed_stage1 = actual_stage1_scores.copy()
    changed_stage1.loc[0, "need_score"] += 0.01
    after = capture_frozen_integrity(changed_stage1, actual_bundle_scores, artifact_paths)
    changed = compare_frozen_integrity(before, after).set_index("artifact")
    assert not bool(changed.loc["stage1_need_score_frame", "unchanged"])
    assert changed.drop(index="stage1_need_score_frame")["unchanged"].all()
