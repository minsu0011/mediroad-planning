from __future__ import annotations

import base64
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mediroad.reporting.stage2_correlation import (  # noqa: E402
    MixedSampleSizeError,
    assert_spectral_metrics_allowed,
    build_specialty_bundle_atlas,
    cross_bundle_separation_metrics,
    high_correlation_equivalence_ledger,
    render_temporal_month_profile_heatmap,
    safe_correlation_quality_metrics,
    same_granularity_correlation,
    spectral_correlation_metrics,
    stabilize_feature_manifest,
    stage1_overlap_metrics,
)
from mediroad.reporting.correlation import (  # noqa: E402
    _matplotlib_korean_font,
    block_bootstrap_quality,
)


def _score_manifest(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source_column": columns,
            "feature_id": [f"S{i:02d}" for i in range(1, len(columns) + 1)],
            "label_ko": [f"score {column}" for column in columns],
            "axis": ["bundle_a", "bundle_a", "bundle_b", "bundle_c"][: len(columns)],
            "bundle_id": ["A", "A", "B", "C"][: len(columns)],
            "granularity": ["admin_dong"] * len(columns),
            "role": ["core"] * len(columns),
        }
    )


def test_same_granularity_correlation_preserves_pairwise_n_and_rejects_mix() -> None:
    frame = pd.DataFrame(
        {
            "internal": [1, 2, 3, 4, 5, 6],
            "rehab": [6, 5, np.nan, 3, 2, 1],
            "sensory": [1, 1, 2, 2, 3, 3],
        },
        index=[f"r{i}" for i in range(6)],
    )
    manifest = _score_manifest(list(frame.columns))
    corr, pairwise_n = same_granularity_correlation(frame, manifest, min_periods=3)

    assert corr.attrs["stage2_same_granularity"] is True
    assert corr.attrs["stage2_granularity"] == "admin_dong"
    assert pairwise_n.loc["internal", "internal"] == 6
    assert pairwise_n.loc["internal", "rehab"] == 5
    assert pairwise_n.loc["rehab", "sensory"] == 5
    assert np.isclose(corr.loc["internal", "rehab"], -1.0)

    mixed = manifest.copy()
    mixed.loc[mixed.source_column.eq("sensory"), "granularity"] = "sigungu_repeated"
    with pytest.raises(ValueError, match="cannot mix spatial units"):
        same_granularity_correlation(frame, mixed, min_periods=3)

    with pytest.raises(MixedSampleSizeError, match="mixed-n"):
        safe_correlation_quality_metrics(
            corr, pairwise_n, manifest, include_spectral=True
        )


def test_spectral_metrics_fail_closed_for_mixed_n() -> None:
    corr = pd.DataFrame(
        [[1.0, 0.2, -0.1], [0.2, 1.0, 0.3], [-0.1, 0.3, 1.0]],
        index=list("abc"),
        columns=list("abc"),
    )
    common_n = pd.DataFrame(30, index=list("abc"), columns=list("abc"))
    assert assert_spectral_metrics_allowed(corr, common_n) == 30
    metrics = spectral_correlation_metrics(corr, common_n)
    assert metrics["common_sample_n"] == 30
    assert 0 < metrics["effective_rank_normalized"] <= 1
    assert metrics["positive_semidefinite"] is True

    mixed_n = common_n.copy()
    mixed_n.loc["a", "b"] = mixed_n.loc["b", "a"] = 11
    with pytest.raises(MixedSampleSizeError, match="mixed-n"):
        assert_spectral_metrics_allowed(corr, mixed_n)
    with pytest.raises(MixedSampleSizeError, match="mixed-n"):
        spectral_correlation_metrics(corr, mixed_n)

    # Equal off-diagonal values are still unsafe when the diagonal proves that
    # each feature used more rows than the intersections did.
    pairwise_deleted = pd.DataFrame(29, index=list("abc"), columns=list("abc"))
    for feature in pairwise_deleted.index:
        pairwise_deleted.loc[feature, feature] = 30
    with pytest.raises(MixedSampleSizeError, match="mixed-n"):
        assert_spectral_metrics_allowed(corr, pairwise_deleted)


def test_cross_bundle_separation_and_stage1_overlap_are_exact() -> None:
    corr = pd.DataFrame(
        [
            [1.0, 0.99, 0.20, -0.90],
            [0.99, 1.0, 0.85, 0.10],
            [0.20, 0.85, 1.0, 0.96],
            [-0.90, 0.10, 0.96, 1.0],
        ],
        index=list("abcd"),
        columns=list("abcd"),
    )
    bundles = {"a": "A", "b": "A", "c": "B", "d": "C"}
    metrics = cross_bundle_separation_metrics(corr, bundles)
    # The a-b within-bundle pair is excluded.  Cross-bundle absolute values:
    # .20, .90, .85, .10, .96 -> median .85.
    assert metrics["cross_bundle_candidate_pair_count"] == 5
    assert metrics["cross_bundle_median_abs_rho"] == pytest.approx(0.85)
    assert metrics["cross_bundle_abs_rho_ge_0_80_rate"] == pytest.approx(3 / 5)
    assert metrics["cross_bundle_abs_rho_ge_0_95_count"] == 1

    index = [f"r{i}" for i in range(10)]
    stage1 = pd.Series(np.arange(10), index=index, name="need_score")
    stage2 = pd.DataFrame(
        {
            "same": np.arange(10),
            "reverse": np.arange(9, -1, -1),
        },
        index=index,
    )
    detail, summary = stage1_overlap_metrics(stage2, stage1, top_fraction=0.20)
    same = detail.set_index("stage2_feature").loc["same"]
    reverse = detail.set_index("stage2_feature").loc["reverse"]
    assert same.spearman_rho == pytest.approx(1.0)
    assert same.top_overlap_rate == pytest.approx(1.0)
    assert reverse.spearman_rho == pytest.approx(-1.0)
    assert reverse.top_overlap_rate == pytest.approx(0.0)
    assert summary["stage1_overlap_pair_count"] == 2


def test_stable_feature_ids_survive_reordering_and_match_bootstrap_edges() -> None:
    features = ["svc_b::source", "svc_a::source", "svc_c::component"]
    manifest = pd.DataFrame(
        {
            "source_column": features,
            "label_ko": ["공급원 B", "공급원 A", "구성요소 C"],
            "axis": ["source", "source", "component"],
            "granularity": ["admin_dong"] * 3,
            "role": ["source", "source", "component"],
        }
    )
    stable = stabilize_feature_manifest(manifest, features)
    expected = dict(zip(features, ["F0001", "F0002", "F0003"]))
    assert stable.set_index("source_column")["feature_id"].to_dict() == expected

    reordered = stabilize_feature_manifest(stable, list(reversed(features)))
    assert reordered.set_index("source_column")["feature_id"].to_dict() == expected

    rng = np.random.default_rng(81)
    frame = pd.DataFrame(rng.normal(size=(18, 3)), columns=features)
    groups = pd.Series(np.repeat(["g1", "g2", "g3"], 6), index=frame.index)
    _, edges = block_bootstrap_quality(
        frame,
        groups,
        reordered,
        n_boot=4,
        seed=91,
        n_jobs=1,
    )
    observed = {}
    for row in edges.itertuples(index=False):
        observed[row.feature_a] = row.feature_id_a
        observed[row.feature_b] = row.feature_id_b
    assert observed == expected


def test_high_correlation_ledger_separates_expected_repetition_from_collapse() -> None:
    features = [
        "svc_a::shared_supply",
        "svc_b::shared_supply",
        "svc_a::specialty_gap_score",
        "svc_b::specialty_gap_score",
    ]
    corr = pd.DataFrame(
        [
            [1.0, 0.990, 0.10, 0.12],
            [0.990, 1.0, 0.11, 0.13],
            [0.10, 0.11, 1.0, 0.995],
            [0.12, 0.13, 0.995, 1.0],
        ],
        index=features,
        columns=features,
    )
    manifest = pd.DataFrame(
        {
            "source_column": features,
            "axis": ["supply", "supply", "output", "output"],
            "granularity": ["admin_dong"] * 4,
            "role": ["source", "source", "model_output", "model_output"],
        }
    )
    stable = stabilize_feature_manifest(manifest, features)
    ledger = high_correlation_equivalence_ledger(corr, stable)

    assert len(ledger) == 2
    expected = ledger.loc[ledger["expected_redundancy"]].iloc[0]
    unexplained = ledger.loc[~ledger["expected_redundancy"]].iloc[0]
    assert expected.explanation_code == "repeated_source_across_services"
    assert expected.semantic_key_a == "shared_supply"
    assert unexplained.explanation_code == "unexplained_high_correlation"
    assert "specialty_gap_score" in {
        unexplained.semantic_key_a,
        unexplained.semantic_key_b,
    }
    assert ledger.attrs["expected_redundancy_count"] == 1
    assert ledger.attrs["unexplained_pair_count"] == 1


def test_specialty_bundle_atlas_writes_high_resolution_self_contained_outputs(
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(20260818)
    n = 36
    latent_a = rng.normal(size=n)
    latent_b = rng.normal(size=n)
    scores = pd.DataFrame(
        {
            "internal": latent_a + rng.normal(scale=0.12, size=n),
            "chronic": latent_a + rng.normal(scale=0.12, size=n),
            "rehab": latent_b + rng.normal(scale=0.12, size=n),
            "sensory": rng.normal(size=n),
        },
        index=[f"region_{i:03d}" for i in range(n)],
    )
    manifest = _score_manifest(list(scores.columns))
    stage1 = pd.Series(latent_a, index=scores.index, name="stage1_need_score")
    manifest["label_ko"] = ["내과", "만성질환", "재활", "감각"]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = build_specialty_bundle_atlas(
            scores,
            manifest,
            tmp_path,
            stage1_scores=stage1,
            min_periods=20,
            stem="synthetic_atlas",
            title="합성 상관관계 지도",
        )

    assert set(result.ordered_features) == set(scores.columns)
    assert result.pairwise_n.to_numpy().min() == n
    assert result.stage1_overlap is not None
    assert result.stage1_overlap_summary is not None
    assert result.quality_metrics["spectral_metrics_applicable"] is False
    for kind in (
        "png",
        "svg",
        "html",
        "correlation_csv",
        "pairwise_n_csv",
        "manifest_csv",
        "high_correlation_ledger_csv",
    ):
        path = Path(result.artifacts[kind])
        assert path.exists() and path.stat().st_size > 100
    png = Path(result.artifacts["png"]).read_bytes()
    assert int.from_bytes(png[16:20], "big") >= 2000
    assert int.from_bytes(png[20:24], "big") >= 2000
    html = Path(result.artifacts["html"]).read_text(encoding="utf-8")
    assert "plotly" in html.lower()
    assert '<script src="https://cdn.plot.ly' not in html.lower()
    # Only Axis varies; constant granularity/proxy/role ribbons are omitted.
    assert '"y":["Axis"]' in html
    assert '"y":["Granularity"]' not in html
    assert '"y":["Proxy"]' not in html
    assert '"y":["Role"]' not in html

    stable_ids = result.manifest.set_index("source_column")["feature_id"]
    assert stable_ids.to_dict() == {
        "internal": "S01",
        "chronic": "S02",
        "rehab": "S03",
        "sensory": "S04",
    }
    saved_manifest = pd.read_csv(result.artifacts["manifest_csv"])
    assert saved_manifest.set_index("source_column")["feature_id"].to_dict() == stable_ids.to_dict()

    svg = Path(result.artifacts["svg"]).read_text(encoding="utf-8")
    embedded = re.findall(r"data:image/png;base64,\s*([^\"]+)", svg)
    assert embedded
    raster_sizes = []
    for payload in embedded:
        raster = base64.b64decode("".join(payload.split()))
        assert raster[:8] == b"\x89PNG\r\n\x1a\n"
        raster_sizes.append(
            (
                int.from_bytes(raster[16:20], "big"),
                int.from_bytes(raster[20:24], "big"),
            )
        )
    assert max(width for width, _ in raster_sizes) >= 2000
    assert max(height for _, height in raster_sizes) >= 2000
    if _matplotlib_korean_font() != "DejaVu Sans":
        assert not [warning for warning in caught if "Glyph" in str(warning.message)]


def test_temporal_month_profile_heatmap_validates_months_and_writes_all_formats(
    tmp_path: Path,
) -> None:
    months = np.arange(1, 13)
    profile = pd.DataFrame(
        {
            "chronic_bundle": 1.0 + 0.10 * np.sin(2 * np.pi * months / 12),
            "respiratory_bundle": 1.0 + 0.25 * np.cos(2 * np.pi * months / 12),
        },
        index=months,
    )
    manifest = pd.DataFrame(
        {
            "source_column": profile.columns,
            "label_ko": ["chronic", "respiratory"],
            "granularity": ["province_common_month", "province_common_month"],
            "bundle_id": ["B1", "B2"],
            "axis": ["clinical", "clinical"],
            "role": ["temporal_modifier", "temporal_modifier"],
        }
    )
    result = render_temporal_month_profile_heatmap(
        profile,
        manifest,
        tmp_path,
        stem="synthetic_month_profile",
    )
    assert result.granularity == "province"
    assert result.month_profile.index.tolist() == list(range(1, 13))
    for path_text in result.artifacts.values():
        path = Path(path_text)
        assert path.exists() and path.stat().st_size > 100
    png = Path(result.artifacts["png"]).read_bytes()
    assert int.from_bytes(png[16:20], "big") >= 2000
    assert int.from_bytes(png[20:24], "big") >= 1000
    html = Path(result.artifacts["html"]).read_text(encoding="utf-8")
    assert "plotly" in html.lower()
    assert '<script src="https://cdn.plot.ly' not in html.lower()

    with pytest.raises(ValueError, match="exactly months 1 through 12"):
        render_temporal_month_profile_heatmap(
            profile.iloc[:-1], manifest, tmp_path / "invalid"
        )
