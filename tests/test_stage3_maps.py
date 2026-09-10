from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mediroad.reporting.stage3_maps import (
    STAGE3_SPATIAL_MAP_IDS,
    render_stage3_spatial_maps,
    validate_stage3_spatial_map_manifest,
)


def _synthetic_tables() -> dict[str, pd.DataFrame]:
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import Polygon

    longitude = np.asarray([127.55, 127.60, 127.65, 127.70, 127.75, 127.80])
    latitude = np.asarray([36.55, 36.58, 36.61, 36.64, 36.67, 36.70])
    grid = gpd.GeoDataFrame(
        {
            "grid_point_index": np.arange(12),
            "elderly65_calibrated_population": np.asarray(
                [0.5, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144], dtype=float
            ),
            "is_rural_eup_myeon": [True, False] * 6,
        },
        geometry=gpd.points_from_xy(
            np.repeat(longitude, 2),
            np.tile(latitude[:2], 6) + np.repeat(np.arange(6) * 0.02, 2),
        ),
        crs=4326,
    )
    venue_points = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(longitude, latitude), crs=4326
    ).to_crs(5179)
    projected_grid = grid.to_crs(5179)
    distances = np.sqrt(
        (
            venue_points.geometry.x.to_numpy()[:, None]
            - projected_grid.geometry.x.to_numpy()[None, :]
        )
        ** 2
        + (
            venue_points.geometry.y.to_numpy()[:, None]
            - projected_grid.geometry.y.to_numpy()[None, :]
        )
        ** 2
    )
    catchment = distances <= 5000.0
    population = grid["elderly65_calibrated_population"].to_numpy(dtype=float)
    rural_flag = grid["is_rural_eup_myeon"].to_numpy(dtype=bool)
    raw_exposure = catchment @ population
    rural_exposure = catchment @ (population * rural_flag)
    urban_exposure = catchment @ (population * ~rural_flag)
    venues = pd.DataFrame(
        {
            "venue_id": [f"v{index}" for index in range(6)],
            "admin": ["A", "A", "B", "B", "C", "C"],
            "longitude": longitude,
            "latitude": latitude,
            "source_kind": [
                "official_village_hall_senior_center",
                "bus_stop_name_proxy",
                "hira_or_public_outreach",
                "official_dementia_center",
                "official_dementia_branch",
                "official_village_hall_senior_center",
            ],
            "raw_exposure": raw_exposure,
            "need_weighted_exposure": raw_exposure * [0.8, 0.9, 1.0, 1.1, 1.2, 1.3],
            "cross_admin_exposure": raw_exposure * [0.1, 0.2, 0.25, 0.35, 0.4, 0.5],
        }
    )
    assert np.allclose(raw_exposure, rural_exposure + urban_exposure)
    boundaries = gpd.GeoDataFrame(
        {
            "admin_dong_code": ["A", "B", "C"],
            "structural_need": [0.90, 0.80, 0.20],
            "need_weighted_exposure": [10.0, 2.0, 8.0],
            "need_exposure_quadrant": [
                "Q1_HIGH_NEED_HIGH_EXPOSURE",
                "Q2_HIGH_NEED_LOW_EXPOSURE",
                "Q3_LOW_NEED_HIGH_EXPOSURE",
            ],
        },
        geometry=[
            Polygon([(127.50, 36.50), (127.64, 36.50), (127.64, 36.62), (127.50, 36.62)]),
            Polygon([(127.62, 36.58), (127.74, 36.58), (127.74, 36.69), (127.62, 36.69)]),
            Polygon([(127.72, 36.64), (127.86, 36.64), (127.86, 36.76), (127.72, 36.76)]),
        ],
        crs=4326,
    )
    overlap = pd.DataFrame(
        {
            "venue_i": ["v0", "v1", "v2", "v3", "v4"],
            "venue_j": ["v1", "v2", "v3", "v4", "v5"],
            "population_weighted_jaccard": [0.82, 0.61, 0.47, 0.36, 0.22],
        }
    )
    bundle = pd.DataFrame(
        [
            {
                "venue_id": venue_id,
                "bundle_id": bundle_id,
                "bundle_gap_weighted_exposure": (venue_index + 1) * (bundle_index + 2),
            }
            for venue_index, venue_id in enumerate(venues["venue_id"])
            for bundle_index, bundle_id in enumerate(("B1", "B2", "B3"))
        ]
    )
    return {
        "grid_policy": grid,
        "candidate_venues": venues,
        "admin_boundaries": boundaries,
        "venue_overlap_edges": overlap,
        "bundle_gap_exposure": bundle,
    }


def test_render_all_spatial_map_triplets_offline_and_projected(tmp_path: Path) -> None:
    result = render_stage3_spatial_maps(_synthetic_tables(), tmp_path)

    assert tuple(result.figures) == STAGE3_SPATIAL_MAP_IDS
    assert len(result.artifact_manifest) == 8 * 3
    assert set(result.artifact_manifest["format"]) == {"png", "svg", "html"}
    assert Path(result.artifact_manifest_path).is_file()
    assert result.artifact_manifest["analysis_crs"].eq("EPSG:5179").all()
    assert result.artifact_manifest.loc[
        result.artifact_manifest["format"].eq("png"), "dpi_x"
    ].ge(299).all()
    assert result.artifact_manifest.loc[
        result.artifact_manifest["format"].eq("html"), "external_script_count"
    ].eq(0).all()
    assert result.artifact_manifest.loc[
        result.artifact_manifest["format"].eq("html"), "self_contained_html"
    ].eq(True).all()
    assert result.artifact_manifest.groupby("figure_id")["format"].agg(set).eq(
        {"png", "svg", "html"}
    ).all()
    validate_stage3_spatial_map_manifest(result.artifact_manifest, tmp_path)

    grid_normalized = result.normalized_tables["elderly_grid_distribution"]
    assert grid_normalized["x_5179"].between(800_000, 1_200_000).all()
    assert grid_normalized["y_5179"].between(1_700_000, 2_100_000).all()
    legend_labels = grid_normalized["population_class_label"].drop_duplicates()
    assert legend_labels.str.startswith("Q").all()
    assert legend_labels.nunique() == grid_normalized["population_class"].nunique()

    candidate_normalized = result.normalized_tables["candidate_venue_map"]
    assert {
        "공식 마을회관·경로당",
        "버스정류장명 기반 후보",
        "HIRA·공공 방문진료",
        "공식 치매안심센터",
        "공식 치매안심센터 분소",
    }.issubset(
        set(candidate_normalized["source_category"])
    )
    assert not candidate_normalized["source_category"].str.contains("_").any()

    admin_normalized = result.normalized_tables["admin_best_venue_map"]
    assert admin_normalized["map_geometry"].eq("ADMIN_POLYGON_CHOROPLETH").all()
    assert admin_normalized["label_strategy"].eq(
        "NO_STATIC_TEXT_LABELS_HTML_HOVER"
    ).all()
    assert set(admin_normalized["quadrant"]) == {
        "Q1_HIGH_NEED_HIGH_EXPOSURE",
        "Q2_HIGH_NEED_LOW_EXPOSURE",
        "Q3_LOW_NEED_HIGH_EXPOSURE",
    }

    overlap_normalized = result.normalized_tables["overlap_redundancy_map"]
    assert "hotspot_cluster" in set(overlap_normalized["record_type"])
    assert "zoom_edge" in set(overlap_normalized["record_type"])
    assert overlap_normalized["aggregation_cell_m"].between(1_000, 5_000).all()
    assert overlap_normalized["record_type"].eq("zoom_edge").sum() <= 120

    for stem in STAGE3_SPATIAL_MAP_IDS:
        for suffix in ("png", "svg", "html"):
            artifact = tmp_path / f"{stem}.{suffix}"
            assert artifact.is_file()
            assert artifact.stat().st_size > 100
        svg = (tmp_path / f"{stem}.svg").read_text(encoding="utf-8")
        html = (tmp_path / f"{stem}.html").read_text(encoding="utf-8")
        assert "<path" in svg
        assert "<text" not in svg.lower()
        assert "<script src=" not in html.lower()
        assert "Malgun Gothic" in html

    quadrant_html = (tmp_path / "admin_best_venue_map.html").read_text(
        encoding="utf-8"
    )
    assert '"fill":"toself"' in quadrant_html
    assert "행정동 Need–Exposure 정책 4분면" in quadrant_html


def test_manifest_validation_detects_hash_tampering(tmp_path: Path) -> None:
    result = render_stage3_spatial_maps(_synthetic_tables(), tmp_path)
    first = tmp_path / result.artifact_manifest.iloc[0]["relative_path"]
    first.write_bytes(first.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="size mismatch"):
        validate_stage3_spatial_map_manifest(result.artifact_manifest, tmp_path)


def test_renderer_fails_closed_on_low_dpi_and_missing_overlap(tmp_path: Path) -> None:
    tables = _synthetic_tables()
    with pytest.raises(ValueError, match="dpi"):
        render_stage3_spatial_maps(tables, tmp_path, dpi=299)
    tables.pop("venue_overlap_edges")
    with pytest.raises(KeyError, match="overlap"):
        render_stage3_spatial_maps(tables, tmp_path)
    tables = _synthetic_tables()
    tables.pop("admin_boundaries")
    with pytest.raises(KeyError, match="boundaries"):
        render_stage3_spatial_maps(tables, tmp_path)


def test_geometry_without_crs_is_rejected(tmp_path: Path) -> None:
    tables = _synthetic_tables()
    tables["grid_policy"] = tables["grid_policy"].set_crs(None, allow_override=True)
    with pytest.raises(ValueError, match="declare a CRS"):
        render_stage3_spatial_maps(tables, tmp_path)


def test_verified_hangul_font_is_fail_closed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import mediroad.reporting.stage3_maps as module

    def unavailable() -> str:
        raise RuntimeError("No verified Hangul-capable font found")

    monkeypatch.setattr(module, "_require_korean_font", unavailable)
    with pytest.raises(RuntimeError, match="Hangul-capable"):
        module.render_stage3_spatial_maps(_synthetic_tables(), tmp_path)
