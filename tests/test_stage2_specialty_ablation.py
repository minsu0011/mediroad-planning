from __future__ import annotations

from collections import Counter
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mediroad.specialty.ablation import (  # noqa: E402
    _assert_deterministic_rank,
    compare_specialty_runs,
    expected_run_specs,
    run_exhaustive_specialty_ablation_from_root,
    run_specialty_ablation,
)
from mediroad.specialty.gap import (  # noqa: E402
    COMPONENT_COLUMNS,
    load_specialty_config,
)


@pytest.fixture(scope="module")
def specialty_config() -> dict:
    return load_specialty_config(ROOT / "configs/model_v1/specialty_gap.yaml")


def _synthetic_scores(admin_count: int = 35, service_count: int = 4) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for service_index in range(service_count):
        for admin_index in range(admin_count):
            rows.append(
                {
                    "admin_dong_code": f"43{admin_index:06d}",
                    "policy_sigungu_name": f"county_{admin_index % 5}",
                    "service_id": f"service_{service_index}",
                    "specialty_gap_score": float(
                        100 - admin_index + service_index / 10
                    ),
                }
            )
    return pd.DataFrame(rows)


def test_expected_ledger_exhaustively_covers_frozen_stage2a_contract(
    specialty_config: dict,
) -> None:
    specs = expected_run_specs(
        specialty_config,
        policy_sigungu_names=[f"county_{index}" for index in range(11)],
    )
    counts = Counter(spec.scope for spec in specs)
    assert len(specs) == 82
    assert len({spec.run_id for spec in specs}) == 82
    assert counts == {
        "baseline": 1,
        "component": 10,
        "scaler_formula": 16,
        "alpha": 16,
        "source_feature": 22,
        "supply_source": 4,
        "access": 2,
        "leave_one_sigungu_out": 11,
    }
    component_specs = [spec for spec in specs if spec.scope == "component"]
    assert {spec.variant for spec in component_specs} == set(COMPONENT_COLUMNS)
    assert {
        (spec.variant, spec.mode) for spec in component_specs
    } == {
        (component, mode)
        for component in COMPONENT_COLUMNS
        for mode in ("renormalize", "neutral")
    }


def test_common_stability_metrics_are_exact_for_identity_and_detect_disruption() -> None:
    baseline = _synthetic_scores()
    service, region, summary = compare_specialty_runs(
        baseline, baseline.sample(frac=1.0, random_state=7), run_id="identity"
    )
    assert np.allclose(service["rank_spearman"], 1.0)
    assert np.allclose(service["rank_kendall"], 1.0)
    assert np.allclose(service["top20_jaccard"], 1.0)
    assert np.allclose(service["symmetric_top20_in_top30"], 1.0)
    assert np.allclose(service["rank_displacement_p95"], 0.0)
    assert np.allclose(service["rank_displacement_max"], 0.0)
    assert np.allclose(service["sigungu_top20_tvd"], 0.0)
    assert np.allclose(region["service_top1_retained"], 1.0)
    assert np.allclose(region["service_top3_retention"], 1.0)
    assert np.allclose(region["service_ndcg_at3"], 1.0)
    assert summary["service_rank_spearman_min"] == pytest.approx(1.0)

    disrupted = baseline.copy()
    mask = disrupted["service_id"].eq("service_0")
    disrupted.loc[mask, "specialty_gap_score"] = disrupted.loc[
        mask, "specialty_gap_score"
    ].min() + disrupted.loc[mask, "specialty_gap_score"].max() - disrupted.loc[
        mask, "specialty_gap_score"
    ]
    changed_service, changed_region, _ = compare_specialty_runs(
        baseline, disrupted, run_id="disrupted"
    )
    service_zero = changed_service.set_index("service_id").loc["service_0"]
    assert service_zero["rank_spearman"] == pytest.approx(-1.0)
    assert service_zero["rank_kendall"] == pytest.approx(-1.0)
    assert service_zero["top20_jaccard"] < 1.0
    assert service_zero["rank_displacement_max"] > 0
    assert changed_region["service_top3_retention"].mean() < 1.0


def test_tie_break_order_is_fail_closed() -> None:
    invalid = pd.DataFrame(
        {
            "admin_dong_code": ["43000002", "43000001"],
            "service_id": ["service", "service"],
            "specialty_gap_score": [50.0, 50.0],
            "specialty_gap_rank": [1, 2],
        }
    )
    with pytest.raises(ValueError, match="deterministic"):
        _assert_deterministic_rank(invalid)


@pytest.fixture(scope="module")
def actual_core_ablation():
    return run_exhaustive_specialty_ablation_from_root(
        ROOT,
        scopes={
            "baseline",
            "component",
            "scaler_formula",
            "alpha",
            "supply_source",
            "access",
        },
        facility_prior_grid=[5.0],
        specialist_prior_grid=[10.0],
    )


def test_actual_build_api_smoke_has_complete_unique_run_ledger(
    actual_core_ablation,
) -> None:
    result = actual_core_ablation
    assert len(result.ledger) == 34
    assert not result.ledger["run_id"].duplicated().any()
    assert result.ledger["status"].eq("PASS").all()
    assert result.ledger["expected_score_rows"].eq(
        result.ledger["actual_score_rows"]
    ).all()
    assert result.ledger["actual_score_rows"].eq(153 * 16).all()
    assert result.run_summary["run_id"].nunique() == 34
    assert len(result.service_metrics) == 34 * 16
    assert len(result.region_service_metrics) == 34 * 153
    assert len(result.region_runs) == 34 * 153 * 16
    assert not result.region_runs.duplicated(
        ["run_id", "admin_dong_code", "service_id"]
    ).any()


def test_actual_baseline_metrics_and_deterministic_ranks_are_exact(
    actual_core_ablation,
) -> None:
    result = actual_core_ablation
    baseline_service = result.service_metrics[
        result.service_metrics["run_id"].eq("baseline::primary")
    ]
    baseline_region = result.region_service_metrics[
        result.region_service_metrics["run_id"].eq("baseline::primary")
    ]
    assert baseline_service["rank_spearman"].eq(1.0).all()
    assert baseline_service["rank_kendall"].eq(1.0).all()
    assert baseline_service["top20_jaccard"].eq(1.0).all()
    assert baseline_service["rank_displacement_max"].eq(0.0).all()
    assert baseline_region["service_top1_retained"].eq(1.0).all()
    assert baseline_region["service_top3_retention"].eq(1.0).all()
    assert baseline_region["service_ndcg_at3"].eq(1.0).all()
    _assert_deterministic_rank(result.baseline_scores)


def test_actual_grid_contains_every_scaler_formula_and_requested_alpha(
    actual_core_ablation,
    specialty_config: dict,
) -> None:
    ledger = actual_core_ablation.ledger
    formula = ledger[ledger["scope"].eq("scaler_formula")]
    assert set(formula["variant"]) == {
        f"{scaler}|{family}"
        for scaler in specialty_config["scalers"]
        for family in specialty_config["formula_families"]
    }
    alpha = ledger[ledger["scope"].eq("alpha")]
    assert alpha["parameters_json"].tolist() == [
        '{"facility_prior_strength": 5.0, "specialist_prior_strength": 10.0}'
    ]


def test_pipeline_alias_accepts_prebuilt_baseline_and_bundle_config(
    actual_core_ablation,
    specialty_config: dict,
) -> None:
    result = run_specialty_ablation(
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        specialty_config,
        actual_core_ablation.baseline_scores,
        bundle_config={},
        scopes={"baseline"},
    )
    assert result.ledger["run_id"].tolist() == ["baseline::primary"]
    assert result.ledger["status"].tolist() == ["PASS"]
