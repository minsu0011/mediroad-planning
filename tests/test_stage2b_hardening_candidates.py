from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from mediroad.temporal.hardening_candidates import (
    BASELINE_CANDIDATE_ID,
    EXPECTED_CANDIDATE_COUNT,
    EXPECTED_ITERATION2_CONTRACT_MAPPING_SHA256,
    ITERATION3_CANDIDATE_ID,
    NO_PROMOTION_DECISION,
    canonical_contract_mapping_sha256,
    evaluate_candidate_grid,
    evaluate_nested_candidate,
    run_hardening_candidates,
    validate_iteration2_config,
)


def _bundle_config() -> dict:
    return {
        "bundles": [
            {
                "bundle_id": "bundle_a",
                "included_services": {"s1": 0.6, "s2": 0.4},
            },
            {
                "bundle_id": "bundle_b",
                "included_services": {"s3": 0.5, "s4": 0.5},
            },
        ]
    }


def _iteration_config() -> dict:
    path = Path(__file__).resolve().parents[1] / "configs/model_v1/stage2b_iteration2.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    # Root owns the YAML edit.  In-memory insertion lets this isolated unit
    # test remain valid before and after that one-line provenance update.
    config["contract_mapping_sha256"] = EXPECTED_ITERATION2_CONTRACT_MAPPING_SHA256
    return config


def _service_panel() -> pd.DataFrame:
    rows = []
    month = np.arange(1, 13)
    patterns = {
        "s1": 1.0 + 0.18 * np.cos(2 * np.pi * (month - 4) / 12),
        "s2": 1.0 + 0.12 * np.cos(2 * np.pi * (month - 5) / 12),
        "s3": 1.0 + 0.20 * np.cos(2 * np.pi * (month - 9) / 12),
        "s4": 1.0 + 0.15 * np.sin(2 * np.pi * month / 12),
    }
    for sigungu_index, sigungu in enumerate(("g1", "g2", "g3")):
        for year in (2022, 2023):
            year_shift = 1 + 0.02 * (year - 2022)
            for service_index, service in enumerate(("s1", "s2", "s3", "s4")):
                level = 80 + 13 * service_index + 3 * sigungu_index
                local = 1 + 0.015 * sigungu_index * np.sin(2 * np.pi * month / 12)
                for month_index, month_number in enumerate(month):
                    rows.append(
                        {
                            "policy_sigungu_name": sigungu,
                            "service_id": service,
                            "year": year,
                            "month": int(month_number),
                            "working_day_rate": float(
                                level
                                * year_shift
                                * patterns[service][month_index]
                                * local[month_index]
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def test_locked_grid_has_72_candidates_and_complete_no_harm_ledger() -> None:
    result = evaluate_candidate_grid(
        _service_panel(),
        _bundle_config(),
        n_permutations=7,
        expected_policy_sigungu=3,
        random_state=17,
        n_jobs=2,
        iteration_config=_iteration_config(),
    )

    assert len(result.ledger) == EXPECTED_CANDIDATE_COUNT
    assert result.ledger["candidate_id"].is_unique
    assert len(result.bundle_ledger) == EXPECTED_CANDIDATE_COUNT * 2
    assert BASELINE_CANDIDATE_ID in set(result.ledger["candidate_id"])
    assert result.audit["candidate_count"] == 72
    assert result.audit["temporal_independent_n"] == 6
    assert result.audit["official_output_writer_available"] is False
    required = {
        "top1_agreement",
        "primary_in_holdout_top2",
        "selected_pair_holdout_top2_coverage",
        "rank_spearman_median",
        "normalized_regret_mean",
        "lomo_primary_retention",
        "lomo_top2_retention",
        "loso_primary_retention",
        "loso_top2_retention",
        "permutation_strength_mean",
        "global_no_harm_pass",
        "bundle_no_harm_pass",
        "eligible",
        "exploratory_no_harm_eligible",
        "eligible_for_official_promotion",
    }
    assert required.issubset(result.ledger.columns)
    assert np.isfinite(
        result.ledger[
            [column for column in required if result.ledger[column].dtype != bool]
        ].select_dtypes(include=[np.number])
    ).all().all()
    assert result.ledger["eligible"].equals(
        result.ledger["exploratory_no_harm_eligible"]
    )
    assert not result.ledger["eligible_for_official_promotion"].any()


def test_candidate_evaluation_is_deterministic_across_n_jobs() -> None:
    arguments = dict(
        service_month=_service_panel(),
        bundle_config=_bundle_config(),
        n_permutations=5,
        expected_policy_sigungu=3,
        random_state=23,
        iteration_config=_iteration_config(),
    )
    single = evaluate_candidate_grid(**arguments, n_jobs=1)
    parallel = evaluate_candidate_grid(**arguments, n_jobs=4)

    pd.testing.assert_frame_equal(single.ledger, parallel.ledger)
    pd.testing.assert_frame_equal(single.bundle_ledger, parallel.bundle_ledger)


def test_nested_candidate_counts_bootstrap_permutation_and_abstention() -> None:
    result = run_hardening_candidates(
        _service_panel(),
        _bundle_config(),
        n_boot=7,
        n_permutations=9,
        expected_policy_sigungu=3,
        random_state=31,
        n_jobs=3,
        iteration_config=_iteration_config(),
    )
    nested = result.nested

    assert len(nested.selection) == 2 * 3 * 2
    assert len(nested.cross_year_detail) == 2 * 3 * 2
    assert len(nested.lomo_detail) == 24 * 3 * 2
    assert nested.lomo_summary["lomo_unique_removals"].eq(24).all()
    assert len(nested.loso_detail) == 3 * 2
    assert nested.loso_summary["loso_unique_removals"].eq(3).all()
    assert len(nested.bootstrap_detail) == 7 * 2
    assert nested.bootstrap_summary["n_boot"].eq(7).all()
    assert len(nested.permutation_detail) == 9 * 2
    assert nested.permutation_summary["n_valid"].eq(9).all()
    assert nested.permutation_detail["annual_totals_preserved"].all()
    assert nested.release_evidence["release_type"].eq(
        "NO_STRONG_PREFERENCE"
    ).all()
    assert nested.release_evidence[["final_primary", "final_fallback"]].isna().all().all()
    assert nested.release_evidence["exploratory_only_due_no_third_year"].all()
    assert not bool(nested.comparison.iloc[0]["eligible_for_official_promotion"])
    assert result.promotion_decision in {
        NO_PROMOTION_DECISION,
        "EXPLORATORY_ONLY_DUE_NO_THIRD_YEAR",
    }
    assert result.audit["stage3_started"] is False
    assert result.audit["grid_nested_panel_seal_match"] is True
    assert (
        result.grid.audit["panel_contract_sha256"]
        == result.nested.audit["panel_contract_sha256"]
        == result.audit["combined_panel_contract_sha256"]
    )


def test_directional_selection_is_train_year_only() -> None:
    original = _service_panel()
    grid = evaluate_candidate_grid(
        original,
        _bundle_config(),
        n_permutations=3,
        expected_policy_sigungu=3,
        random_state=41,
        iteration_config=_iteration_config(),
    )
    first = evaluate_nested_candidate(
        original,
        _bundle_config(),
        grid,
        n_boot=3,
        n_permutations=3,
        expected_policy_sigungu=3,
        random_state=41,
        iteration_config=_iteration_config(),
    )
    changed = original.copy()
    holdout = changed["year"].eq(2023)
    changed.loc[holdout, "working_day_rate"] *= (
        0.6 + 0.08 * changed.loc[holdout, "month"].to_numpy()
    )
    changed_grid = evaluate_candidate_grid(
        changed,
        _bundle_config(),
        n_permutations=3,
        expected_policy_sigungu=3,
        random_state=41,
        iteration_config=_iteration_config(),
    )
    second = evaluate_nested_candidate(
        changed,
        _bundle_config(),
        changed_grid,
        n_boot=3,
        n_permutations=3,
        expected_policy_sigungu=3,
        random_state=41,
        iteration_config=_iteration_config(),
    )

    columns = [
        "policy_sigungu_name",
        "bundle_id",
        "common_monthly_variance_ratio",
        "selected_representation",
    ]
    left = first.selection.loc[first.selection["train_year"].eq(2022), columns]
    right = second.selection.loc[second.selection["train_year"].eq(2022), columns]
    pd.testing.assert_frame_equal(
        left.reset_index(drop=True), right.reset_index(drop=True)
    )
    assert not first.selection["holdout_labels_visible_during_selection"].any()
    assert not first.selection["threshold_grid_search_used"].any()


def test_nested_public_api_rejects_grid_from_a_different_panel() -> None:
    original = _service_panel()
    grid = evaluate_candidate_grid(
        original,
        _bundle_config(),
        n_permutations=3,
        expected_policy_sigungu=3,
        random_state=43,
        iteration_config=_iteration_config(),
    )
    changed = original.copy()
    changed.loc[0, "working_day_rate"] *= 1.01

    with pytest.raises(ValueError, match="panel SHA256 differs"):
        evaluate_nested_candidate(
            changed,
            _bundle_config(),
            grid,
            n_boot=3,
            n_permutations=3,
            expected_policy_sigungu=3,
            random_state=43,
            iteration_config=_iteration_config(),
        )


def test_fail_closed_leakage_counts_threshold_and_config_guards() -> None:
    config = _iteration_config()
    assert validate_iteration2_config(config)["candidate_count_expected"] == 72
    broken = copy.deepcopy(config)
    broken["candidate_grid"]["candidate_count_expected"] = 71
    with pytest.raises(ValueError):
        validate_iteration2_config(broken)

    panel = _service_panel()
    with pytest.raises(ValueError, match="exactly two|leakage"):
        evaluate_candidate_grid(
            panel.loc[panel["year"].eq(2022)],
            _bundle_config(),
            n_permutations=3,
            expected_policy_sigungu=3,
        )
    with pytest.raises(ValueError, match="positive integer"):
        evaluate_candidate_grid(
            panel,
            _bundle_config(),
            n_permutations=0,
            expected_policy_sigungu=3,
        )

    grid = evaluate_candidate_grid(
        panel,
        _bundle_config(),
        n_permutations=3,
        expected_policy_sigungu=3,
    )
    with pytest.raises(ValueError, match="locked at 0.50"):
        evaluate_nested_candidate(
            panel,
            _bundle_config(),
            grid,
            n_boot=3,
            n_permutations=3,
            expected_policy_sigungu=3,
            threshold=0.49,
        )


@pytest.mark.parametrize(
    ("path", "tampered_value"),
    [
        (("stage3_started",), True),
        (
            ("frozen_baseline", "official_output_tree_mutation_forbidden"),
            "outputs/model_v1/06_stage2b_iteration2",
        ),
        (("candidate_grid", "spatial_estimators", "local", "enabled"), False),
        (("candidate_grid", "fixed_before_full_grid_result"), False),
        (
            (
                "promotion_no_harm_contract",
                "bundle_harm_caps",
                "maximum_top2_retention_decline",
            ),
            0.051,
        ),
        (
            (
                "promotion_no_harm_contract",
                "bundle_harm_caps",
                "maximum_normalized_regret_increase",
            ),
            0.049,
        ),
        (
            (
                "promotion_no_harm_contract",
                "final_candidate_requirements",
                "bootstrap_replicates",
            ),
            999,
        ),
        (
            (
                "promotion_no_harm_contract",
                "final_candidate_requirements",
                "permutation_replicates",
            ),
            4999,
        ),
        (
            (
                "promotion_no_harm_contract",
                "final_candidate_requirements",
                "jobs",
            ),
            7,
        ),
        (("decision_contract", "stage3_execution_forbidden"), False),
        (
            (
                "output_contract",
                "official_hardening_tree_must_remain_unchanged",
            ),
            False,
        ),
        (
            (
                "iteration3_single_nested_candidate",
                "same_promotion_no_harm_contract_required",
            ),
            False,
        ),
        (
            (
                "iteration3_single_nested_candidate",
                "maximum_status_without_third_same_definition_year",
            ),
            "official",
        ),
        (
            (
                "promotion_no_harm_contract",
                "global_cross_year",
                "top1_agreement",
            ),
            "greater_than_or_equal_to_baseline",
        ),
        (
            (
                "promotion_no_harm_contract",
                "construct_integrity",
                "patient_and_provider_geography_mixing_count_max",
            ),
            1,
        ),
        (
            (
                "promotion_no_harm_contract",
                "final_candidate_requirements",
                "original_release_rules_must_be_reapplied",
            ),
            False,
        ),
        (
            (
                "decision_contract",
                "candidate_promotion_requires_all_no_harm_checks",
            ),
            False,
        ),
        (
            ("decision_contract", "no_eligible_candidate_decision"),
            "PROMOTE",
        ),
        (("output_contract", "output_dir"), "outputs/model_v1/05_stage2b_hardening"),
        (("output_contract", "atomic_metadata_last"), False),
    ],
)
def test_locked_config_fields_reject_tampering(
    path: tuple[str, ...], tampered_value: object
) -> None:
    config = copy.deepcopy(_iteration_config())
    target = config
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = tampered_value

    with pytest.raises(ValueError):
        validate_iteration2_config(config)


def test_canonical_mapping_seal_is_required_and_cannot_be_resealed_after_tamper() -> None:
    config = _iteration_config()
    assert (
        canonical_contract_mapping_sha256(config)
        == EXPECTED_ITERATION2_CONTRACT_MAPPING_SHA256
    )
    assert (
        validate_iteration2_config(config)["contract_mapping_sha256"]
        == EXPECTED_ITERATION2_CONTRACT_MAPPING_SHA256
    )

    missing = copy.deepcopy(config)
    missing.pop("contract_mapping_sha256")
    with pytest.raises(ValueError, match="SHA256 seal mismatch"):
        validate_iteration2_config(missing)

    forged = copy.deepcopy(config)
    forged["decision_contract"]["no_eligible_candidate_decision"] = "PROMOTE"
    forged["contract_mapping_sha256"] = canonical_contract_mapping_sha256(forged)
    assert forged["contract_mapping_sha256"] != EXPECTED_ITERATION2_CONTRACT_MAPPING_SHA256
    with pytest.raises(ValueError, match="SHA256 seal mismatch"):
        validate_iteration2_config(forged)
