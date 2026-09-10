from __future__ import annotations

import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mediroad.reporting.stage2b_hardening import (  # noqa: E402
    FIGURE_IDS,
    _font_family,
    build_artifact_inventory,
    build_stage2b_quality_gate,
    render_stage2b_hardening_figures,
    validate_artifact_inventory,
    validate_stage2b_quality_gate,
    validate_temporal_evidence_summary,
)


BUNDLES = ["chronic", "musculoskeletal"]
SEASONS = ["spring", "summer", "autumn", "winter"]


def _synthetic_tables() -> dict[str, pd.DataFrame]:
    rank_rows = []
    bootstrap_rows = []
    for bundle_index, bundle in enumerate(BUNDLES):
        for season_index, season in enumerate(SEASONS):
            shifted = ((season_index + bundle_index) % 4) + 1
            rank_rows.append(
                {
                    "bundle_id": bundle,
                    "season": season,
                    "year2022_rank": season_index + 1,
                    "year2023_rank": shifted,
                    "pooled_rank": season_index + 1,
                    "cross_year_consensus_rank": season_index + 1,
                }
            )
            probabilities = (
                [0.70, 0.15, 0.10, 0.05]
                if bundle == "chronic"
                else [0.42, 0.08, 0.45, 0.05]
            )
            bootstrap_rows.append(
                {
                    "bundle_id": bundle,
                    "season": season,
                    "primary_probability": probabilities[season_index],
                }
            )

    transfer_rows = []
    margin_rows = []
    regret_rows = []
    for bundle_index, bundle in enumerate(BUNDLES):
        for direction_index, direction in enumerate(("2022_to_2023", "2023_to_2022")):
            transfer_rows.append(
                {
                    "bundle_id": bundle,
                    "direction": direction,
                    "top1_agreement": float(bundle == "chronic"),
                    "train_primary_in_holdout_top2": 1.0,
                    "pair_top2_coverage": 1.0,
                    "season_rank_spearman": 0.8 - 0.1 * bundle_index,
                    "season_rank_kendall": 0.67 - 0.1 * direction_index,
                }
            )
            margin_rows.append(
                {
                    "bundle_id": bundle,
                    "direction": direction,
                    "train_top1_margin": 0.30 - 0.08 * bundle_index,
                    "holdout_top1_margin": 0.25 - 0.08 * bundle_index,
                }
            )
            regret_rows.append(
                {
                    "bundle_id": bundle,
                    "direction": direction,
                    "heldout_regret_normalized": 0.03 + 0.04 * bundle_index,
                }
            )

    lomo_rows = []
    for bundle in BUNDLES:
        for year in (2022, 2023):
            for month in range(1, 13):
                lomo_rows.append(
                    {
                        "bundle_id": bundle,
                        "year": year,
                        "removed_month": month,
                        "primary_retained": not (
                            bundle == "musculoskeletal" and month in {4, 10}
                        ),
                    }
                )

    loso_rows = []
    sigungu = [f"정책시군 {index:02d}" for index in range(1, 12)]
    for bundle in BUNDLES:
        for index, name in enumerate(sigungu):
            loso_rows.append(
                {
                    "bundle_id": bundle,
                    "removed_sigungu": name,
                    "rank_spearman": 1.0 - 0.02 * index,
                }
            )

    null_rows = []
    for bundle_index, bundle in enumerate(BUNDLES):
        for metric, observed in (("season_margin", 0.28), ("concentration", 0.62)):
            null_rows.append(
                {
                    "bundle_id": bundle,
                    "metric": metric,
                    "observed": observed - 0.05 * bundle_index,
                    "null_mean": observed - 0.13,
                    "null_sd": 0.04,
                    "p_value_upper": 0.02 + 0.10 * bundle_index,
                    "standardized_strength": 3.25 - bundle_index,
                }
            )

    confidence = pd.DataFrame(
        {
            "bundle_id": BUNDLES,
            "release_type": ["STRONG_SINGLE", "ROBUST_PAIR"],
            "temporal_confidence": ["HIGH", "MODERATE"],
        }
    )
    return {
        "bundle_year_season_rank_heatmap": pd.DataFrame(rank_rows),
        "cross_year_transfer_matrix": pd.DataFrame(transfer_rows),
        "bundle_season_margin_chart": pd.DataFrame(margin_rows),
        "heldout_regret_chart": pd.DataFrame(regret_rows),
        "bootstrap_season_selection_probability": pd.DataFrame(bootstrap_rows),
        "leave_one_month_out_stability": pd.DataFrame(lomo_rows),
        "leave_one_sigungu_out_stability": pd.DataFrame(loso_rows),
        "observed_vs_permutation_null": pd.DataFrame(null_rows),
        "temporal_confidence_summary": confidence,
    }


def test_nine_figures_are_korean_safe_300dpi_complete_and_self_contained(
    tmp_path: Path,
) -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = render_stage2b_hardening_figures(
            _synthetic_tables(), tmp_path / "figures"
        )

    assert tuple(result.figures) == FIGURE_IDS
    assert len(result.artifact_manifest) == 27
    assert set(result.artifact_manifest["format"]) == {"png", "svg", "html"}
    png = result.artifact_manifest.loc[result.artifact_manifest["format"].eq("png")]
    assert png["dpi_x"].min() >= 299.0
    assert png["dpi_y"].min() >= 299.0
    assert png["width_px"].min() >= 1800
    assert png["height_px"].min() >= 1000
    html_rows = result.artifact_manifest.loc[
        result.artifact_manifest["format"].eq("html")
    ]
    assert html_rows["self_contained_html"].all()
    assert html_rows["external_script_count"].eq(0).all()
    assert Path(result.artifact_manifest_path).is_file()
    for figure_id, paths in result.figures.items():
        assert set(paths) == {"png", "svg", "html"}
        for path_text in paths.values():
            path = Path(path_text)
            assert path.is_file() and path.stat().st_size > 100
        html_text = Path(paths["html"]).read_text(encoding="utf-8")
        assert "plotly" in html_text.lower()
        assert not re.search(r"<script\b[^>]*\bsrc\s*=", html_text, flags=re.I)
    rank_html = Path(
        result.figures["bundle_year_season_rank_heatmap"]["html"]
    ).read_text(encoding="utf-8")
    confidence_html = Path(
        result.figures["temporal_confidence_summary"]["html"]
    ).read_text(encoding="utf-8")
    null_html = Path(
        result.figures["observed_vs_permutation_null"]["html"]
    ).read_text(encoding="utf-8")
    assert "만성질환" in rank_html and "근골격·재활" in rank_html
    assert "봄" in rank_html and "겨울" in rank_html
    assert "계절 우선순위 확정 없음" in confidence_html
    assert "canonical release_type" in confidence_html
    for field in (
        "observed=",
        "null_mean=",
        "null_sd=",
        "p_value_upper=",
        "standardized_strength=",
    ):
        assert field in null_html
    null_png = png.loc[
        png["figure_id"].eq("observed_vs_permutation_null")
    ].iloc[0]
    assert 2500 <= null_png["width_px"] < 6000
    assert null_png["height_px"] >= 1500
    confidence_table = result.normalized_tables["temporal_confidence_summary"]
    assert set(confidence_table["release_type"]) == {
        "STRONG_SINGLE",
        "ROBUST_PAIR",
    }
    assert list(
        result.normalized_tables[FIGURE_IDS[0]]["bundle_id"].astype(str).unique()
    ) == BUNDLES
    assert result.artifact_manifest.groupby("figure_id")["input_table_sha256"].nunique().eq(1).all()
    if _font_family() != "DejaVu Sans":
        assert not [warning for warning in caught if "Glyph" in str(warning.message)]


def test_figure_contract_fails_closed_and_explicit_aliases_work(tmp_path: Path) -> None:
    tables = _synthetic_tables()
    # The engine's cross-year summary publishes only train margin; the
    # reporter deterministically derives each holdout margin from the reverse
    # direction without mixing fit and evaluation records.
    tables["bundle_season_margin_chart"] = tables[
        "bundle_season_margin_chart"
    ].drop(columns="holdout_top1_margin")
    alias = tables["heldout_regret_chart"].rename(
        columns={"heldout_regret_normalized": "my_regret"}
    )
    tables["heldout_regret_chart"] = alias
    result = render_stage2b_hardening_figures(
        tables,
        tmp_path / "alias_figures",
        column_maps={
            "heldout_regret_chart": {"my_regret": "heldout_regret_normalized"}
        },
    )
    assert len(result.artifact_manifest) == 27

    incomplete = _synthetic_tables()
    incomplete["leave_one_month_out_stability"] = incomplete[
        "leave_one_month_out_stability"
    ].loc[lambda frame: ~((frame["year"] == 2022) & (frame["removed_month"] == 1))]
    with pytest.raises(ValueError, match="exactly 12 removals"):
        render_stage2b_hardening_figures(incomplete, tmp_path / "incomplete")

    ambiguous = _synthetic_tables()
    ambiguous["heldout_regret_chart"] = ambiguous["heldout_regret_chart"].rename(
        columns={"heldout_regret_normalized": "normalized_regret"}
    )
    ambiguous["heldout_regret_chart"]["heldout_regret"] = 0.1
    with pytest.raises(ValueError, match="ambiguous aliases"):
        render_stage2b_hardening_figures(ambiguous, tmp_path / "ambiguous")


def _evidence_summary() -> pd.DataFrame:
    rows = []
    for index, bundle in enumerate(
        ["chronic", "comprehensive", "musculoskeletal", "neuro", "sensory"]
    ):
        abstain = bundle == "sensory"
        pair = bundle == "musculoskeletal"
        rows.append(
            {
                "bundle_id": bundle,
                "2022_top1": "spring",
                "2022_top2": "spring|winter",
                "2023_top1": "autumn" if pair else "spring",
                "2023_top2": "spring|autumn" if pair else "spring|winter",
                "pooled_top1": "spring",
                "pooled_top2": "spring|winter",
                "cross_year_consistency": 0.65 if pair else 1.0,
                "heldout_regret": 0.02 + 0.01 * index,
                "bootstrap_primary_probability": 0.45 if pair else 0.85,
                "lomo_stability": 0.80 if pair else 1.0,
                "loso_stability": 0.90,
                "null_strength": 1.5,
                "release_type": (
                    "NO_STRONG_PREFERENCE"
                    if abstain
                    else "ROBUST_PAIR" if pair else "STRONG_SINGLE"
                ),
                "confidence": "LOW" if abstain else "MODERATE" if pair else "HIGH",
                "final_primary": pd.NA if abstain else "spring",
                "final_fallback": pd.NA if abstain else "autumn" if pair else "winter",
                "reason": "교차연도·재표집 증거를 함께 반영",
            }
        )
    return pd.DataFrame(rows)


def test_evidence_summary_permits_honest_abstention_and_rejects_false_precision() -> None:
    source = _evidence_summary().rename(columns={"confidence": "temporal_confidence"})
    validated = validate_temporal_evidence_summary(source)
    assert "confidence" in validated and "temporal_confidence" in validated
    assert validated["confidence"].equals(validated["temporal_confidence"])
    abstention = validated.loc[
        validated["release_type"].eq("NO_STRONG_PREFERENCE")
    ].iloc[0]
    assert pd.isna(abstention["final_primary"])
    assert pd.isna(abstention["final_fallback"])

    forced = _evidence_summary()
    forced.loc[forced["bundle_id"].eq("sensory"), "final_primary"] = "spring"
    with pytest.raises(ValueError, match="must not force"):
        validate_temporal_evidence_summary(forced)

    same_fallback = _evidence_summary()
    same_fallback.loc[same_fallback["bundle_id"].eq("chronic"), "final_fallback"] = "spring"
    with pytest.raises(ValueError, match="must differ"):
        validate_temporal_evidence_summary(same_fallback)


def test_quality_gate_preserves_severity_threshold_provenance_and_decision() -> None:
    records = [
        {
            "gate_id": "cross_year_complete",
            "severity": "hard",
            "passed": True,
            "observed": 10,
            "comparison": "==",
            "threshold": 10,
            "threshold_source": "stage2b_hardening.yaml config",
            "note": "5 bundles × 2 transfer directions",
        },
        {
            "gate_id": "weak_margin",
            "severity": "advisory",
            "passed": False,
            "observed": 0.145,
            "comparison": ">=",
            "threshold": 0.20,
            "threshold_source": "preregistered config threshold",
            "note": "musculoskeletal weakness is retained",
        },
        {
            "gate_id": "destructive_ablation",
            "severity": "diagnostic",
            "passed": False,
            "observed": 0,
            "comparison": "contract",
            "threshold": "expected failure",
            "threshold_source": "prompt contract section 30",
            "note": "sole supported signal removed",
        },
    ]
    result = build_stage2b_quality_gate(
        records, final_decision="PASS_COARSE_TEMPORAL"
    )
    assert result.hard_pass is True
    assert result.hard_total == 1 and result.hard_failed == 0
    assert result.advisory_failed == 1 and result.diagnostic_failed == 1
    assert result.gates.set_index("gate_id").loc["weak_margin", "threshold"] == 0.20
    assert "preregistered" in result.gates.set_index("gate_id").loc[
        "weak_margin", "threshold_source"
    ]
    round_trip = validate_stage2b_quality_gate(
        result.gates, final_decision="PASS_COARSE_TEMPORAL"
    )
    assert round_trip.hard_pass

    failed = pd.DataFrame(records)
    failed.loc[failed["severity"].eq("hard"), "passed"] = False
    with pytest.raises(ValueError, match="cannot pass"):
        build_stage2b_quality_gate(failed, final_decision="PASS_COARSE_TEMPORAL")

    posthoc = pd.DataFrame(records)
    posthoc.loc[0, "threshold_source"] = "chosen after results"
    with pytest.raises(ValueError, match="threshold_source"):
        build_stage2b_quality_gate(posthoc)


def test_artifact_inventory_hash_validation_detects_mutation(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    first = artifact_dir / "가이드.txt"
    second = artifact_dir / "evidence.csv"
    first.write_text("Stage 2B 계절 근거", encoding="utf-8")
    second.write_text("bundle,value\nchronic,1\n", encoding="utf-8")

    inventory = build_artifact_inventory(artifact_dir, base_dir=tmp_path)
    assert inventory["relative_path"].tolist() == [
        "artifacts/evidence.csv",
        "artifacts/가이드.txt",
    ]
    valid = validate_artifact_inventory(inventory, tmp_path)
    assert valid.valid and valid.checked_count == 2

    second.write_text("bundle,value\nchronic,2\n", encoding="utf-8")
    invalid = validate_artifact_inventory(inventory, tmp_path, raise_on_error=False)
    assert not invalid.valid
    assert invalid.sha_mismatch_count == 1
    with pytest.raises(ValueError, match="validation failed"):
        validate_artifact_inventory(inventory, tmp_path)
