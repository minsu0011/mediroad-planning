from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd
import pytest

import mediroad.reporting.stage3 as stage3_reporting
from mediroad.reporting.stage3 import (
    STAGE3_CORE_FIGURE_IDS,
    build_stage3_quality_gate,
    quality_gate_record,
    render_stage3_core_figures,
    validate_stage3_figure_manifest,
)
from mediroad.stage3.decision import (
    FIELD_VALIDATION_FIELDS,
    INTERFACE_REQUIRED_COLUMNS,
    build_admin_shortlists,
    build_coverage_cluster_fallbacks,
    build_field_validation_queue,
    build_need_exposure_quadrants,
    build_stage3_to_stage4_interface,
    build_venue_feasibility_context,
    collapse_admin_venues,
    compute_pareto_tiers,
    compute_sensitivity_metrics,
    stage3_spearman_audit,
    validate_stage3_to_stage4_interface,
    validate_temporal_abstention,
)


def _candidates() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for admin_position, admin in enumerate(("A", "B")):
        for position in range(6):
            raw_exposure = float(20 - position + admin_position)
            cross_admin_share = 0.2 + 0.1 * position
            rows.append(
                {
                    "venue_id": f"{admin}{position + 1}",
                    "venue_name": f"venue-{admin}{position + 1}",
                    "venue_type": "village_hall",
                    "admin_dong_code": admin,
                    "sigungu": "S1" if admin == "A" else "S2",
                    "raw_elderly_exposure": raw_exposure,
                    "cross_admin_exposure": raw_exposure * cross_admin_share,
                    "cross_admin_share": cross_admin_share,
                    "need_weighted_exposure": float(8 + (position % 3) - admin_position),
                    "bundle_gap_weighted_exposure": float(4 + ((position + 1) % 4)),
                    "known_outreach_overlap": float(position % 2),
                    "field_validation_unknown_count": int(position % 3),
                    "coverage_cluster_id": f"{admin}-c{position // 2}",
                    "latitude": 36.0 + admin_position * 0.2 + position * 0.001,
                    "longitude": 127.0 + admin_position * 0.2 + position * 0.001,
                }
            )
    return pd.DataFrame(rows)


def _pareto_objectives() -> dict[str, str]:
    return {
        "raw_elderly_exposure": "max",
        "need_weighted_exposure": "max",
        "bundle_gap_weighted_exposure": "max",
        "known_outreach_overlap": "min",
        "field_validation_unknown_count": "min",
    }


def _interface_row(venue_id: str = "v1", bundle_id: str = "B1") -> dict[str, object]:
    return {
        "venue_id": venue_id,
        "venue_name": "마을회관",
        "venue_type": "village_hall",
        "admin_dong_code": "A",
        "sigungu": "S1",
        "latitude": 36.5,
        "longitude": 127.5,
        "structural_need": 0.8,
        "raw_elderly_exposure": 100.0,
        "need_weighted_exposure": 80.0,
        "high_need_elderly_exposure": 45.0,
        "own_admin_exposure": 60.0,
        "cross_admin_exposure": 40.0,
        "cross_sigungu_exposure": 10.0,
        "bundle_id": bundle_id,
        "specialty_gap": 0.4,
        "bundle_gap_weighted_exposure": 40.0,
        "venue_readiness_prior": 0.5,
        "transit_context": "BASELINE_ONLY",
        "road_context": "METRIC_PROXY",
        "known_outreach_overlap": 5.0,
        "field_validation_required": True,
        "pareto_tier": 1,
        "shortlist_rank": 1,
        "coverage_cluster_id": "c1",
        "fallback_venue_1": "v2",
        "fallback_venue_2": pd.NA,
        "temporal_release_type": "NO_STRONG_PREFERENCE",
        "temporal_confidence": "LOW",
        "recommended_season": pd.NA,
        "fallback_season": pd.NA,
    }


def test_feasibility_preserves_unknown_and_does_not_invent_feasibility() -> None:
    venues = pd.DataFrame(
        {
            "venue_id": ["v1", "v2"],
            "venue_type": ["health_post", "village_hall"],
            "parking_verified": [True, None],
            "power_verified": ["no", "unexpected free text"],
        }
    )
    result = build_venue_feasibility_context(
        venues,
        readiness_prior_by_type={"health_post": 0.8, "village_hall": 0.4},
    )
    assert "venue_operationally_feasible" not in result
    assert result.loc[0, "parking_verified"] == "verified"
    assert result.loc[0, "power_verified"] == "not_verified"
    assert result.loc[1, "power_verified"] == "unknown"
    assert set(FIELD_VALIDATION_FIELDS).issubset(result.columns)
    assert result["field_validation_required"].all()
    assert result["field_validation_unknown_count"].min() >= 4
    assert result["venue_readiness_prior"].tolist() == [0.8, 0.4]


def test_feasibility_rejects_fabricated_operational_flag() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        build_venue_feasibility_context(
            pd.DataFrame(
                {
                    "venue_id": ["v1"],
                    "venue_type": ["hall"],
                    "venue_operationally_feasible": [True],
                }
            )
        )


def test_pareto_tiers_are_grouped_strict_and_ties_survive() -> None:
    candidates = pd.DataFrame(
        {
            "venue_id": ["a", "b", "c", "d", "x", "y"],
            "admin_dong_code": ["A", "A", "A", "A", "B", "B"],
            "benefit": [10, 8, 7, 10, 1, 2],
            "readiness": [5, 7, 4, 5, 1, 2],
            "uncertainty": [0, 0, 1, 0, 1, 0],
        }
    )
    result = compute_pareto_tiers(
        candidates,
        {"benefit": "max", "readiness": "max", "uncertainty": "min"},
    ).set_index("venue_id")
    assert result.loc["a", "pareto_tier"] == 1
    assert result.loc["b", "pareto_tier"] == 1
    assert result.loc["d", "pareto_tier"] == 1
    assert result.loc["c", "pareto_tier"] == 2
    assert result.loc["a", "dominance_count"] == 0
    assert not result.loc["a", "is_dominated"]
    assert result.loc["y", "pareto_tier"] == 1
    assert result.loc["x", "pareto_tier"] == 2


def test_pareto_fails_on_missing_or_nonfinite_objective() -> None:
    candidates = pd.DataFrame(
        {"venue_id": ["a"], "admin_dong_code": ["A"], "benefit": [np.nan]}
    )
    with pytest.raises(ValueError, match="finite"):
        compute_pareto_tiers(candidates, {"benefit": "max"})


def test_admin_shortlists_keep_top3_top5_pareto_and_all_candidates() -> None:
    candidates = compute_pareto_tiers(_candidates(), _pareto_objectives())
    result = build_admin_shortlists(candidates)
    assert len(result.candidates) == len(candidates)
    assert result.top3.groupby("admin_dong_code").size().eq(3).all()
    assert result.top5.groupby("admin_dong_code").size().eq(5).all()
    assert result.shortlist.groupby("admin_dong_code").size().ge(1).all()
    assert set(result.pareto_front["venue_id"]).issubset(set(result.shortlist["venue_id"]))
    assert result.candidates.groupby("admin_dong_code")["shortlist_rank"].min().eq(1).all()


def test_cluster_fallbacks_prioritize_equivalent_cluster_and_never_self() -> None:
    candidates = _candidates()
    primaries = candidates.loc[candidates["venue_id"].isin(["A1", "A3"])].copy()
    fallbacks = build_coverage_cluster_fallbacks(primaries, candidates).set_index("venue_id")
    assert fallbacks.loc["A1", "fallback_venue_1"] == "A2"
    assert bool(fallbacks.loc["A1", "fallback_1_same_cluster"])
    assert fallbacks.loc["A3", "fallback_venue_1"] == "A4"
    assert not (fallbacks["fallback_venue_1"].astype(str) == fallbacks.index.astype(str)).any()


def test_admin_collapse_and_four_quadrants_are_policy_unit_level() -> None:
    rows = []
    specs = {
        "A": (9.0, 10.0),
        "B": (8.0, 2.0),
        "C": (2.0, 8.0),
        "D": (1.0, 1.0),
    }
    for admin, (need, exposure) in specs.items():
        rows.extend(
            [
                {
                    "venue_id": f"{admin}1",
                    "admin_dong_code": admin,
                    "structural_need": need,
                    "need_weighted_exposure": exposure,
                },
                {
                    "venue_id": f"{admin}2",
                    "admin_dong_code": admin,
                    "structural_need": need,
                    "need_weighted_exposure": exposure - 0.5,
                },
            ]
        )
    venues = pd.DataFrame(rows)
    collapsed = collapse_admin_venues(venues)
    assert len(collapsed) == 4
    assert collapsed["candidate_count"].eq(2).all()
    quadrants = build_need_exposure_quadrants(
        venues, need_threshold=5.0, exposure_threshold=5.0
    ).set_index("admin_dong_code")
    assert quadrants.loc["A", "quadrant"].startswith("Q1")
    assert quadrants.loc["B", "quadrant"].startswith("Q2")
    assert quadrants.loc["C", "quadrant"].startswith("Q3")
    assert quadrants.loc["D", "quadrant"].startswith("Q4")


def test_spearman_audit_separates_descriptive_venues_from_admin_units() -> None:
    venues = _candidates().copy()
    venues["structural_need"] = venues["admin_dong_code"].map({"A": 0.8, "B": 0.3})
    result = stage3_spearman_audit(
        venues,
        ["raw_elderly_exposure", "need_weighted_exposure", "structural_need"],
        min_periods=2,
    )
    assert result.all_venue_correlation.shape == (3, 3)
    assert result.admin_collapsed_correlation.shape == (3, 3)
    assert len(result.admin_collapsed) == 2
    assert result.all_venue_counts.iloc[0, 1] == len(venues)
    assert result.admin_collapsed_counts.iloc[0, 1] == 2
    assert result.all_venue_correlation.attrs["analysis_level"] == "all_venue_descriptive"
    assert result.all_venue_correlation.attrs["sample_n"] == len(venues)
    assert result.all_venue_counts.attrs["sample_n"] == len(venues)
    assert result.admin_collapsed_correlation.attrs["sample_n"] == 2
    assert result.admin_collapsed_counts.attrs["sample_n"] == 2
    assert result.comparison["all_venue_role"].eq("DESCRIPTIVE_ONLY").all()


def test_sensitivity_metrics_identity_is_one_and_reports_admin_detail() -> None:
    baseline = compute_pareto_tiers(_candidates(), _pareto_objectives())
    identical = baseline.copy()
    result = compute_sensitivity_metrics(baseline, {"same": identical})
    row = result.summary.iloc[0]
    assert row["venue_rank_spearman"] == pytest.approx(1.0)
    assert row["admin_top3_jaccard_mean"] == pytest.approx(1.0)
    assert row["admin_top5_jaccard_mean"] == pytest.approx(1.0)
    assert row["pareto_retention"] == pytest.approx(1.0)
    assert row["raw_exposure_median_absolute_change"] == pytest.approx(0.0)
    assert row["cross_admin_exposure_median_absolute_change"] == pytest.approx(0.0)
    assert row["cross_admin_share_median_change"] == pytest.approx(0.0)
    assert row["border_venue_frequency_change"] == pytest.approx(0.0)
    assert row["border_venue_frequency_baseline"] == pytest.approx(0.5)
    assert result.by_admin["raw_exposure_median_absolute_change"].eq(0.0).all()
    assert result.by_admin["cross_admin_share_median_change"].eq(0.0).all()
    assert len(result.by_admin) == 2


def test_sensitivity_reports_exposure_and_border_frequency_changes() -> None:
    baseline = compute_pareto_tiers(_candidates(), _pareto_objectives())
    alternative = baseline.copy()
    alternative["raw_elderly_exposure"] *= 0.5
    alternative["cross_admin_exposure"] = 0.0
    # This stale value must be ignored in favour of scenario-specific exposure.
    alternative["cross_admin_share"] = baseline["cross_admin_share"]

    result = compute_sensitivity_metrics(
        baseline,
        {"own_admin_only": alternative},
        border_share_threshold=0.5,
    )
    row = result.summary.iloc[0]
    assert row["raw_exposure_venue_relative_change_median"] == pytest.approx(-0.5)
    assert row["cross_admin_exposure_median_alternative"] == pytest.approx(0.0)
    assert row["cross_admin_share_median_alternative"] == pytest.approx(0.0)
    assert row["border_venue_frequency_baseline"] == pytest.approx(0.5)
    assert row["border_venue_frequency_alternative"] == pytest.approx(0.0)
    assert row["border_venue_frequency_change"] == pytest.approx(-0.5)
    assert result.by_admin["border_venue_frequency_baseline"].eq(0.5).all()
    assert result.by_admin["border_venue_frequency_alternative"].eq(0.0).all()
    assert row["border_venue_definition"] == "cross_admin_share_ge_threshold"


def test_sensitivity_pareto_retention_falls_when_front_membership_changes() -> None:
    candidates = pd.DataFrame(
        {
            "venue_id": ["a", "b", "c"],
            "admin_dong_code": ["A", "A", "A"],
            "need_weighted_exposure": [10.0, 9.0, 8.0],
            "raw_elderly_exposure": [10.0, 9.0, 8.0],
            "cross_admin_exposure": [2.0, 2.0, 2.0],
            "field_burden": [5.0, 1.0, 6.0],
        }
    )
    objectives = {
        "need_weighted_exposure": "max",
        "raw_elderly_exposure": "max",
        "field_burden": "min",
    }
    baseline = compute_pareto_tiers(candidates, objectives)
    alternative_input = candidates.copy()
    alternative_input.loc[alternative_input["venue_id"].eq("b"), "field_burden"] = 20.0
    alternative = compute_pareto_tiers(alternative_input, objectives)

    assert set(baseline.loc[baseline["is_pareto_front"], "venue_id"]) == {"a", "b"}
    assert set(alternative.loc[alternative["is_pareto_front"], "venue_id"]) == {"a"}
    result = compute_sensitivity_metrics(baseline, {"changed_front": alternative})
    assert result.summary.iloc[0]["pareto_retention"] == pytest.approx(0.5)
    assert result.by_admin.iloc[0]["pareto_retention"] == pytest.approx(0.5)


def test_sensitivity_can_compare_recomputed_policy_shortlist_flags() -> None:
    baseline = compute_pareto_tiers(_candidates(), _pareto_objectives())
    baseline = build_admin_shortlists(
        baseline,
        objectives=_pareto_objectives(),
        tie_breakers={"need_weighted_exposure": "desc"},
    ).candidates
    alternative = baseline.copy()
    for _, group in alternative.groupby("admin_dong_code", sort=False):
        order = group.sort_values("shortlist_rank").index.tolist()
        # Keep the exposure score unchanged but replace one Top3 member and the
        # best venue.  A score-only comparison would falsely report identity.
        new_order = order[3:] + order[:3]
        alternative.loc[new_order, "shortlist_rank"] = range(1, len(new_order) + 1)
    alternative["in_admin_top3"] = alternative["shortlist_rank"].le(3)
    alternative["in_admin_top5"] = alternative["shortlist_rank"].le(5)
    result = compute_sensitivity_metrics(
        baseline,
        {"policy_change": alternative},
        use_existing_shortlist_flags=True,
    )
    row = result.summary.iloc[0]
    assert row["venue_rank_spearman"] == pytest.approx(1.0)
    assert row["admin_top3_jaccard_mean"] < 1.0
    assert row["admin_top5_jaccard_mean"] < 1.0
    assert row["region_best_venue_stability"] == pytest.approx(0.0)


def test_sensitivity_fails_if_candidate_universe_changes() -> None:
    baseline = _candidates()
    with pytest.raises(ValueError, match="candidate ids differ"):
        compute_sensitivity_metrics(baseline, baseline.iloc[:-1].copy())


def test_field_validation_queue_is_bounded_per_admin() -> None:
    candidates = compute_pareto_tiers(_candidates(), _pareto_objectives())
    candidates["field_validation_required"] = True
    candidates["shortlist_rank"] = candidates.groupby("admin_dong_code").cumcount() + 1
    queue = build_field_validation_queue(candidates, top_k=3)
    assert queue.groupby("admin_dong").size().eq(3).all()
    assert queue["field_validation_queue_rank"].between(1, 3).all()
    assert queue["reason_to_validate"].str.contains("FIELD_UNKNOWN").all()


def test_interface_schema_and_official_temporal_abstention_pass() -> None:
    frame = pd.DataFrame([_interface_row()])
    result = validate_stage3_to_stage4_interface(frame)
    assert tuple(result.columns[: len(INTERFACE_REQUIRED_COLUMNS)]) == INTERFACE_REQUIRED_COLUMNS
    assert result["recommended_season"].isna().all()
    assert result["fallback_season"].isna().all()


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("selected_final", True, "forbidden"),
        ("recommended_season", "spring", "recommended/fallback season null"),
        ("fallback_venue_1", "v1", "same venue"),
    ],
)
def test_interface_rejects_stage4_leakage_temporal_overclaim_and_self_fallback(
    column: str, value: object, message: str
) -> None:
    frame = pd.DataFrame([_interface_row()])
    frame[column] = value
    with pytest.raises(ValueError, match=message):
        validate_stage3_to_stage4_interface(frame)


def test_interface_requires_current_low_no_preference_contract() -> None:
    frame = pd.DataFrame([_interface_row()])
    frame["temporal_release_type"] = "STRONG_SINGLE"
    frame["temporal_confidence"] = "HIGH"
    frame["recommended_season"] = "spring"
    frame["fallback_season"] = "autumn"
    with pytest.raises(ValueError, match="current frozen"):
        validate_temporal_abstention(frame)
    # The generic validator remains reusable only through an explicit future-release opt-out.
    validate_temporal_abstention(frame, require_official_abstention=False)


def test_interface_requires_boundary_exposure_partition() -> None:
    frame = pd.DataFrame([_interface_row()])
    frame["cross_admin_exposure"] = 39.0
    with pytest.raises(ValueError, match="must equal"):
        validate_stage3_to_stage4_interface(frame)


def test_interface_builder_joins_temporal_many_to_one_without_creating_season() -> None:
    row = _interface_row()
    temporal_columns = [
        "temporal_release_type",
        "temporal_confidence",
        "recommended_season",
        "fallback_season",
    ]
    spatial = pd.DataFrame([{k: v for k, v in row.items() if k not in temporal_columns}])
    temporal = pd.DataFrame(
        {
            "bundle_id": ["B1"],
            "temporal_release_type": ["NO_STRONG_PREFERENCE"],
            "temporal_confidence": ["LOW"],
            "recommended_season": [pd.NA],
            "fallback_season": [pd.NA],
        }
    )
    interface = build_stage3_to_stage4_interface(spatial, temporal)
    assert interface["recommended_season"].isna().all()
    assert interface["fallback_season"].isna().all()


def test_quality_gate_helper_evaluates_and_summarizes() -> None:
    records = [
        quality_gate_record(
            "stage4_not_started",
            observed=False,
            comparison="==",
            threshold=False,
            note="Stage 4 remains outside this run.",
        ),
        quality_gate_record(
            "field_unknown_advisory",
            observed=2,
            comparison="<=",
            threshold=0,
            severity="advisory",
            note="Unknown fields are queued, not imputed.",
        ),
    ]
    result = build_stage3_quality_gate(records)
    assert result.hard_pass
    assert result.hard_total == 1
    assert result.advisory_failed == 1
    assert result.gates.set_index("gate_id").loc["stage4_not_started", "status"] == "PASS"
    with pytest.raises(ValueError, match="Stage 4"):
        build_stage3_quality_gate(records, stage4_started=True)


def test_core_figures_are_300dpi_svg_and_self_contained_html(tmp_path: Path) -> None:
    quadrant_rows = pd.DataFrame(
        {
            "admin_dong_code": ["A", "B", "C", "D"],
            "structural_need": [0.9, 0.8, 0.2, 0.1],
            "need_weighted_exposure": [0.9, 0.2, 0.8, 0.1],
            "need_threshold": [0.5] * 4,
            "exposure_threshold": [0.5] * 4,
            "quadrant": [
                "Q1_HIGH_NEED_HIGH_EXPOSURE",
                "Q2_HIGH_NEED_LOW_EXPOSURE",
                "Q3_LOW_NEED_HIGH_EXPOSURE",
                "Q4_LOW_NEED_LOW_EXPOSURE",
            ],
        }
    )
    labels = [
        "raw_elderly_exposure",
        "catchment_mean_need_intensity",
        "cross_admin_share",
    ]
    all_corr = pd.DataFrame(
        [[1.0, 0.8, 0.3], [0.8, 1.0, 0.4], [0.3, 0.4, 1.0]],
        index=labels,
        columns=labels,
    )
    admin_corr = pd.DataFrame(
        [[1.0, 0.5, 0.1], [0.5, 1.0, 0.2], [0.1, 0.2, 1.0]],
        index=labels,
        columns=labels,
    )
    all_corr.attrs.update({"analysis_level": "all_venue_descriptive", "sample_n": 3_909})
    admin_corr.attrs.update({"analysis_level": "admin_collapsed", "sample_n": 153})
    sensitivity = pd.DataFrame(
        {
            "scenario": ["3km", "5km"],
            "venue_rank_spearman": [0.8, 0.9],
            "admin_top3_jaccard_mean": [0.7, 0.85],
            "admin_top5_jaccard_mean": [0.8, 0.9],
            "pareto_retention": [0.75, 0.88],
            "region_best_venue_stability": [0.65, 0.82],
        }
    )
    pareto = pd.DataFrame(
        {
            "venue_id": ["v1", "v2", "v3", "v4"],
            "pareto_tier": [1, 1, 2, 3],
            "raw_elderly_exposure": [100, 80, 70, 40],
            "need_weighted_exposure": [60, 75, 50, 30],
        }
    )
    result = render_stage3_core_figures(
        {
            "need_exposure_quadrant": quadrant_rows,
            "correlation_all_venue": all_corr,
            "correlation_admin_collapsed": admin_corr,
            "sensitivity_stability": sensitivity,
            "pareto_shortlist_summary": pareto,
        },
        tmp_path,
    )
    validate_stage3_figure_manifest(result.artifact_manifest)
    assert len(result.artifact_manifest) == len(STAGE3_CORE_FIGURE_IDS) * 3
    assert result.artifact_manifest.loc[
        result.artifact_manifest["format"].eq("png"), "dpi_x"
    ].ge(299).all()
    html_rows = result.artifact_manifest.loc[
        result.artifact_manifest["format"].eq("html")
    ]
    assert html_rows["self_contained_html"].eq(True).all()
    for path in result.artifact_manifest["relative_path"]:
        assert (tmp_path / path).is_file()
    correlation_html = Path(
        result.figures["correlation_dependency_audit"]["html"]
    ).read_text(encoding="utf-8")
    correlation_svg = Path(
        result.figures["correlation_dependency_audit"]["svg"]
    ).read_text(encoding="utf-8")
    assert "n=3,909" in correlation_html
    assert "n=153" in correlation_html
    assert "n=3,909" in correlation_svg
    assert "n=153" in correlation_svg
    assert "Elderly exposure" in correlation_html
    assert "raw_elderly_exposure" in correlation_html


def test_correlation_layout_uses_dynamic_n_labels_and_canonical_hover() -> None:
    labels = [
        "raw_elderly_exposure",
        "catchment_mean_need_intensity",
        "overlap_weighted_degree",
    ]
    all_corr = pd.DataFrame(
        [[1.0, 0.8, 0.3], [0.8, 1.0, 0.4], [0.3, 0.4, 1.0]],
        index=labels,
        columns=labels,
    )
    admin_corr = pd.DataFrame(
        [[1.0, 0.5, 0.1], [0.5, 1.0, 0.2], [0.1, 0.2, 1.0]],
        index=labels,
        columns=labels,
    )
    all_corr.attrs["sample_n"] = 37
    admin_corr.attrs["sample_n"] = 4

    static, interactive, normalized = stage3_reporting._correlation_figures(
        all_corr, admin_corr
    )
    try:
        panel_titles = [axis.get_title() for axis in static.axes if axis.get_title()]
        assert any("n=37" in title for title in panel_titles)
        assert any("n=4" in title for title in panel_titles)
        tick_labels = {
            tick.get_text() for tick in static.axes[0].get_xticklabels()
        }
        assert "Elderly exposure" in tick_labels
        assert "raw_elderly_exposure" not in tick_labels
        width, height = static.get_size_inches()
        assert width <= 17.0
        assert height <= 7.0

        delta_trace = interactive.data[2]
        expected_delta_limit = float(
            np.abs((admin_corr - all_corr).to_numpy(dtype=float)).max()
        )
        assert float(delta_trace.zmax) == pytest.approx(expected_delta_limit)
        assert float(delta_trace.zmin) == pytest.approx(-expected_delta_limit)
        assert "Canonical:" in str(delta_trace.hovertemplate)
        assert "Δρ" in str(delta_trace.hovertemplate)
        canonical_hover = np.asarray(interactive.data[0].customdata, dtype=object)
        assert set(canonical_hover[..., 0].ravel()) == set(labels)
        assert set(canonical_hover[..., 1].ravel()) == set(labels)
        assert set(labels).issubset(normalized.columns)
        assert set(labels).issubset(set(normalized["metric"]))
        assert normalized["all_venue_sample_n"].eq(37).all()
        assert normalized["admin_collapsed_sample_n"].eq(4).all()
    finally:
        from matplotlib import pyplot as plt

        plt.close(static)

    missing_n = all_corr.copy()
    missing_n.attrs.clear()
    with pytest.raises(ValueError, match="sample_n attr"):
        stage3_reporting._correlation_figures(missing_n, admin_corr)


def test_sensitivity_layout_is_compact_faceted_and_preserves_canonical_ids() -> None:
    scenarios = [
        "custom_radius",
        "hard_3000m",
        "FULL_IDENTITY",
        "ablation_published_positive_only_suppressed_missing",
        "objective_loo__raw_elderly_exposure",
        "objective_loo__overlap_weighted_degree",
    ]
    sensitivity = pd.DataFrame(
        {
            "scenario": scenarios,
            "sensitivity_scope": [
                "catchment_definition",
                "catchment_definition",
                "identity_control",
                "population_calibration_shadow_with_77pct_cells_missing",
                "pareto_objective_leave_one_out",
                "pareto_objective_leave_one_out",
            ],
            "venue_rank_spearman": [-0.5, 0.9, 1.0, 0.8, 1.0, 1.0],
            "admin_top3_jaccard_mean": [0.4, 0.7, 1.0, 0.6, 0.9, 0.7],
            "admin_top5_jaccard_mean": [0.5, 0.8, 1.0, 0.7, 0.9, 0.8],
            "pareto_retention": [0.6, 0.8, 1.0, 0.7, 0.9, 0.5],
            "region_best_venue_stability": [0.3, 0.7, 1.0, 0.6, 1.0, 1.0],
        }
    )

    static, interactive, normalized = stage3_reporting._sensitivity_figures(
        sensitivity
    )
    try:
        width, height = static.get_size_inches()
        assert width <= 16.0
        assert height <= 9.0
        assert len(static.axes) == 3
        titles = [axis.get_title() for axis in static.axes]
        assert titles == [
            "Catchment definition\n(n=2)",
            "Model ablations / checks\n(n=2)",
            "Objective leave-one-out\n(n=2)",
        ]
        displayed = {
            tick.get_text() for axis in static.axes for tick in axis.get_yticklabels()
        }
        assert "Hard 3 km" in displayed
        assert "Published cells only" in displayed
        assert "LOO · Elderly exposure" in displayed
        assert not any("objective_loo__" in label for label in displayed)
        # Negative Spearman is valid by contract and must remain visible.
        assert all(axis.get_xlim()[0] <= -0.5 for axis in static.axes)

        assert float(interactive.layout.xaxis.range[0]) <= -0.5
        canonical_pairs = np.concatenate(
            [np.asarray(trace.customdata, dtype=object) for trace in interactive.data],
            axis=0,
        )
        assert set(scenarios).issubset(set(canonical_pairs[:, 0]))
        assert "venue_rank_spearman" in set(canonical_pairs[:, 1])
        assert all("Scenario ID=" in str(trace.hovertemplate) for trace in interactive.data)
        assert normalized["scenario"].tolist() == scenarios
        assert normalized["sensitivity_scope"].tolist() == sensitivity[
            "sensitivity_scope"
        ].tolist()
        html = interactive.to_html(include_plotlyjs=True, full_html=True)
        assert "objective_loo__overlap_weighted_degree" in html
        assert not re.search(r"<script\b[^>]*\bsrc\s*=", html, flags=re.I)
    finally:
        from matplotlib import pyplot as plt

        plt.close(static)
