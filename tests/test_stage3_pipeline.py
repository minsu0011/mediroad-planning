from __future__ import annotations

import re
from hashlib import sha256
from pathlib import Path

import pandas as pd
import pytest

from mediroad.reporting.stage3 import STAGE3_CORE_FIGURE_IDS
from mediroad.reporting.stage3_maps import STAGE3_SPATIAL_MAP_IDS
from mediroad.stage3_pipeline import (
    _build_sensitivity_alternative,
    _complete_link_overlap_clusters,
    _config_contract,
    _correlation_matrix_diagnostics,
    _id_set_sha,
    _is_temporal_selector_feature,
    _pareto_objectives,
    _stage3_report_figure_section,
    _snapshot_mutations,
    _validate_stage3_report_figure_links,
)
from mediroad.stage3.decision import compute_pareto_tiers, compute_sensitivity_metrics


def _contract() -> dict:
    # Keep mutation tests bound to the complete production contract.  A
    # hand-written minimal fixture can silently lag newly fail-closed fields and
    # then make the regression suite fail before the Stage 3 writer starts.
    import yaml

    root = Path(__file__).resolve().parents[1]
    return yaml.safe_load(
        (root / "configs/model_v1/stage3_spatial.yaml").read_text(encoding="utf-8")
    )


def test_stage3_config_is_fail_closed_against_stage4_and_posthoc_radius() -> None:
    _config_contract(_contract())
    altered = _contract()
    altered["stage4_started"] = True
    with pytest.raises(ValueError):
        _config_contract(altered)
    altered = _contract()
    altered["analysis_contract"]["primary_exposure_method"] = "hard_10000m"
    with pytest.raises(ValueError):
        _config_contract(altered)


def test_id_set_sha_is_order_invariant_and_identity_sensitive() -> None:
    assert _id_set_sha(["b", "a", "c"]) == _id_set_sha(["c", "b", "a"])
    assert _id_set_sha(["b", "a", "c"]) != _id_set_sha(["b", "a", "d"])


def test_frozen_snapshot_detects_size_and_hash_mutation() -> None:
    before = pd.DataFrame(
        [{"scope": "x", "relative_path": "a", "size_bytes": 1, "sha256": "a"}]
    )
    after = before.copy()
    assert _snapshot_mutations(before, after).empty
    after.loc[0, "sha256"] = "b"
    assert len(_snapshot_mutations(before, after)) == 1


def test_actual_stage3_config_remains_locked() -> None:
    root = Path(__file__).resolve().parents[1]
    import yaml

    config = yaml.safe_load(
        (root / "configs/model_v1/stage3_spatial.yaml").read_text(encoding="utf-8")
    )
    _config_contract(config)
    assert config["analysis_contract"]["expected_eligible_venue_rows"] == 3909
    assert config["analysis_contract"]["expected_interface_rows"] == 19545


def test_complete_link_overlap_clusters_do_not_merge_transitive_chain() -> None:
    # A--B and B--C are both strong, but A--C is not.  A connected-component
    # implementation would incorrectly call all three equivalent catchments.
    edges = pd.DataFrame(
        {
            "venue_i": ["A", "B"],
            "venue_j": ["B", "C"],
            "jaccard": [0.90, 0.90],
        }
    )
    clusters = _complete_link_overlap_clusters(
        ["A", "B", "C"],
        edges,
        {"A": 3.0, "B": 2.0, "C": 1.0},
        threshold=0.80,
    )
    assert clusters["coverage_cluster_id"].nunique() == 2
    assert clusters["cluster_size"].max() == 2
    assert clusters.loc[clusters["venue_id"].eq("C"), "cluster_size"].item() == 1
    assert clusters["cluster_min_pairwise_jaccard"].min() >= 0.80


def test_score_ablation_aligns_by_venue_id_and_recomputes_pareto() -> None:
    candidates = pd.DataFrame(
        {
            "venue_id": ["v2", "v1", "v3"],
            "admin_dong_code": ["A", "A", "A"],
            "need_weighted_exposure": [2.0, 3.0, 1.0],
        }
    )
    objectives = {"need_weighted_exposure": "higher"}
    baseline = compute_pareto_tiers(candidates, objectives)
    reversed_by_id = pd.Series({"v1": 1.0, "v2": 2.0, "v3": 3.0})
    alternative = _build_sensitivity_alternative(
        baseline,
        score_by_venue=reversed_by_id,
        objectives=objectives,
    )
    assert alternative.loc[alternative["is_pareto_front"], "venue_id"].tolist() == ["v3"]
    sensitivity = compute_sensitivity_metrics(baseline, {"reversed": alternative})
    assert sensitivity.summary.loc[0, "pareto_retention"] == 0.0
    with pytest.raises(ValueError, match="venue IDs"):
        _build_sensitivity_alternative(
            baseline,
            score_by_venue=pd.Series({"v1": 1.0, "v2": 2.0}),
            objectives=objectives,
        )


def test_pareto_objectives_and_temporal_selector_are_fail_closed() -> None:
    config = _contract()
    columns = (
        config["shortlist"]["pareto_metrics_higher"]
        + config["shortlist"]["pareto_metrics_lower"]
    )
    frame = pd.DataFrame({column: [1.0, 2.0] for column in columns})
    assert len(_pareto_objectives(frame, config)) == 8
    with pytest.raises(ValueError, match="lacks Pareto objectives"):
        _pareto_objectives(frame.drop(columns=[columns[0]]), config)
    for forbidden in (
        "season_adjusted_exposure",
        "spring_venue_score",
        "temporal_score",
        "exact_date",
    ):
        assert _is_temporal_selector_feature(forbidden)
    assert not _is_temporal_selector_feature("need_weighted_exposure")


def test_correlation_diagnostics_expose_constant_axis_nan() -> None:
    records = pd.DataFrame({"varying": [1.0, 2.0, 3.0], "constant": [1.0] * 3})
    matrix = records[["varying", "constant"]].corr(method="spearman")
    diagnostics = _correlation_matrix_diagnostics(
        matrix,
        records,
        ["varying", "constant"],
    )
    assert diagnostics["constant_metric_count"] == 1
    assert diagnostics["nonfinite_cell_count"] > 0


def _make_report_figure_manifests(
    report_dir: Path,
) -> tuple[Path, pd.DataFrame, pd.DataFrame]:
    figures_dir = report_dir / "figures"
    figures_dir.mkdir(parents=True)

    def build(figure_ids: tuple[str, ...]) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for figure_id in figure_ids:
            for artifact_format in ("png", "svg", "html"):
                artifact = figures_dir / f"{figure_id}.{artifact_format}"
                if artifact_format == "html":
                    payload = (
                        "<!doctype html><html><body><script>"
                        "window.Plotly={version:'test'};"
                        "</script></body></html>"
                    ).encode("utf-8")
                else:
                    payload = f"{figure_id}:{artifact_format}:publication".encode("utf-8")
                artifact.write_bytes(payload)
                rows.append(
                    {
                        "figure_id": figure_id,
                        "format": artifact_format,
                        "relative_path": artifact.name,
                        "size_bytes": len(payload),
                        "sha256": sha256(payload).hexdigest(),
                        "external_script_count": 0,
                        "self_contained_html": artifact_format == "html",
                    }
                )
        return pd.DataFrame(rows)

    core = build(STAGE3_CORE_FIGURE_IDS)
    spatial = build(STAGE3_SPATIAL_MAP_IDS)
    core.to_csv(report_dir / "STAGE3_CORE_FIGURE_MANIFEST.csv", index=False)
    spatial.to_csv(report_dir / "STAGE3_SPATIAL_MAP_MANIFEST.csv", index=False)
    return report_dir / "FINAL_REPORT.md", core, spatial


def test_canonical_report_figure_links_resolve_from_run_report(tmp_path: Path) -> None:
    report_path, core, spatial = _make_report_figure_manifests(
        tmp_path / "reports" / "model_v1" / "stage3" / "runs" / "run_001"
    )
    section = _stage3_report_figure_section(report_path, core, spatial)
    report_path.write_text(f"# Stage 3\n\n{section}\n", encoding="utf-8")

    _validate_stage3_report_figure_links(report_path, core, spatial)
    assert "run-scoped `FINAL_REPORT.md`가 권위 보고서" in section
    assert "`07_stage3_spatial_exposure_venue.md`는 byte-identical 편의 alias" in section
    assert "byte-identical 편의 alias" in section
    assert section.count("[HTML (self-contained)]") == 12
    assert all(f"`{figure_id}`" in section for figure_id in STAGE3_CORE_FIGURE_IDS)
    assert all(f"`{figure_id}`" in section for figure_id in STAGE3_SPATIAL_MAP_IDS)

    links = re.findall(r"\[[^\]\n]+\]\(([^)\n]+)\)", section)
    assert len(links) == 38
    assert len(set(links)) == 38
    assert all((report_path.parent / link).is_file() for link in links)


def test_canonical_report_figure_links_fail_closed(tmp_path: Path) -> None:
    report_path, core, spatial = _make_report_figure_manifests(
        tmp_path / "reports" / "model_v1" / "stage3" / "runs" / "run_002"
    )
    section = _stage3_report_figure_section(report_path, core, spatial)
    report_path.write_text(f"# Stage 3\n\n{section}\n", encoding="utf-8")

    missing = report_path.parent / "figures" / "candidate_venue_map.svg"
    missing.unlink()
    with pytest.raises(FileNotFoundError, match="does not resolve"):
        _validate_stage3_report_figure_links(report_path, core, spatial)

    missing.write_bytes(b"replacement")
    with pytest.raises(RuntimeError, match="does not match its manifest"):
        _validate_stage3_report_figure_links(report_path, core, spatial)


def test_canonical_report_rejects_external_html_and_missing_table_link(
    tmp_path: Path,
) -> None:
    report_path, core, spatial = _make_report_figure_manifests(
        tmp_path / "reports" / "model_v1" / "stage3" / "runs" / "run_003"
    )
    section = _stage3_report_figure_section(report_path, core, spatial)
    report_path.write_text(f"# Stage 3\n\n{section}\n", encoding="utf-8")
    report_path.write_text(
        report_path.read_text(encoding="utf-8").replace(
            "[PNG](figures/need_exposure_quadrant.png)", "PNG unavailable", 1
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="index is missing or changed"):
        _validate_stage3_report_figure_links(report_path, core, spatial)

    external = report_path.parent / "figures" / "elderly_grid_distribution.html"
    payload = b'<html><script src="https://example.invalid/plotly.js"></script></html>'
    external.write_bytes(payload)
    row = spatial["relative_path"].eq(external.name)
    spatial.loc[row, "size_bytes"] = len(payload)
    spatial.loc[row, "sha256"] = sha256(payload).hexdigest()
    with pytest.raises(ValueError, match="must be self-contained"):
        _stage3_report_figure_section(report_path, core, spatial)
