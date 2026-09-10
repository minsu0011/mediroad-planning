from types import SimpleNamespace

import pandas as pd

from mediroad.reporting.stage2b_hardening import validate_temporal_evidence_summary
from mediroad.stage2b_hardening_pipeline import (
    _final_decision_from_release,
    _temporal_evidence_table,
)


def test_public_evidence_keeps_regret_and_abstention_contract() -> None:
    bundle = "musculoskeletal_rehab"
    release = pd.DataFrame(
        [
            {
                "bundle_id": bundle,
                "year2022_top1": "spring",
                "year2022_top2": "spring|autumn",
                "year2023_top1": "autumn",
                "year2023_top2": "spring|autumn",
                "pooled_top1": "spring",
                "pooled_top2": "spring|autumn",
                "consensus_primary_candidate": "spring",
                "consensus_fallback_candidate": "autumn",
                "cross_year_top1_agreement": 0.0,
                "bidirectional_primary_in_holdout_top2": 0.5,
                "bidirectional_selected_pair_holdout_top2_coverage": 1.0,
                "pooled_normalized_top1_margin": 0.01,
                "year_normalized_top1_margin": 0.01,
                "bootstrap_primary_probability": 0.55,
                "bootstrap_pair_coverage_probability": 0.9,
                "lomo_primary_retention": 0.7,
                "lomo_top2_retention": 0.9,
                "loso_primary_retention": 0.7,
                "loso_top2_retention": 0.9,
                "working_day_primary_retention": 0.5,
                "working_day_top2_retention": 1.0,
                "boundary_primary_retention": 0.5,
                "boundary_top2_retention": 1.0,
                "permutation_margin_percentile": 0.2,
                "permutation_p_value_margin": 0.8,
                "release_type": "NO_STRONG_PREFERENCE",
                "temporal_confidence": "LOW",
                "primary_season": pd.NA,
                "fallback_season": pd.NA,
                "temporal_independent_n": 55,
                "temporal_source": "synthetic_test",
                "temporal_spatial_resolution": "policy_sigungu_x_bundle",
                "release_reason": "explicit abstention",
                "limitations": "synthetic",
            }
        ]
    )
    transfer = pd.DataFrame(
        [
            {
                "bundle_id": bundle,
                "direction": "2022_to_2023",
                "heldout_regret": 0.4,
                "heldout_regret_normalized": 0.7,
            },
            {
                "bundle_id": bundle,
                "direction": "2023_to_2022",
                "heldout_regret": 0.2,
                "heldout_regret_normalized": 0.3,
            },
        ]
    )
    result = SimpleNamespace(cross_year=SimpleNamespace(summary=transfer))
    evidence = _temporal_evidence_table(result, release)
    assert evidence.loc[0, "heldout_regret"] == 0.4
    assert "temporal_confidence" in evidence
    checked = validate_temporal_evidence_summary(
        evidence, expected_bundles=(bundle,)
    )
    assert checked.loc[0, "release_type"] == "NO_STRONG_PREFERENCE"
    assert pd.isna(checked.loc[0, "final_primary"])


def test_final_decision_preserves_advisory_abstention() -> None:
    release = pd.DataFrame(
        {"release_type": ["STRONG_SINGLE", "NO_STRONG_PREFERENCE"]}
    )
    assert _final_decision_from_release(release) == "PASS_ADVISORY_TEMPORAL"
    assert (
        _final_decision_from_release(release, hard_failed=1)
        == "FAIL_TEMPORAL_RELEASE"
    )
