"""Publication-grade spatial maps for MEDIROAD MODEL V1 Stage 3.

The Stage 3 core reporting module contains decision diagnostics.  This module
is deliberately separate: it renders the spatial/distribution evidence used
to inspect population allocation, venue coverage, overlap, and rural/urban
balance.  No basemap or web asset is requested, so every HTML remains usable
offline and reproducible from the supplied tables.

The public entry point is :func:`render_stage3_spatial_maps`.  It accepts a
mapping of GeoDataFrames/DataFrames, normalizes all spatial inputs to
EPSG:5179, writes eight exact PNG/SVG/HTML triplets, writes a checksummed
artifact manifest, and validates the result fail-closed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import html as html_module
import re

import numpy as np
import pandas as pd

from mediroad.reporting.stage2b_hardening import (
    _atomic_write_csv,
    _frame_sha256,
    _png_metadata,
    _save_figure_triplet,
    sha256_file,
)


STAGE3_SPATIAL_MAP_IDS: tuple[str, ...] = (
    "elderly_grid_distribution",
    "candidate_venue_map",
    "venue_exposure_distribution",
    "cross_admin_coverage_map",
    "overlap_redundancy_map",
    "admin_best_venue_map",
    "bundle_gap_weighted_venue_map",
    "rural_urban_exposure",
)

_TABLE_ALIASES: Mapping[str, tuple[str, ...]] = {
    "grid": ("grid", "grid_policy", "elderly_grid", "population_grid"),
    "venues": (
        "venues",
        "candidate_venues",
        "venue_exposure",
        "venue_exposures",
    ),
    "boundaries": (
        "admin_boundaries",
        "boundaries",
        "admin_boundary",
        "chungbuk_boundary",
    ),
    "overlap": ("overlap_edges", "venue_overlap_edges", "overlap"),
    "admin_best": ("admin_best_venues", "admin_best", "regional_best_venues"),
    "bundle": (
        "bundle_gap_exposure",
        "bundle_gap_weighted_venue",
        "venue_bundle_exposure",
        "bundle_exposure",
    ),
    "rural_urban": ("rural_urban_exposure", "venue_rural_urban_exposure"),
}

_BUNDLE_COLOURS: tuple[str, ...] = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#6B7280",
)

_QUADRANT_COLOURS: Mapping[str, str] = {
    "Q1_HIGH_NEED_HIGH_EXPOSURE": "#0F766E",
    "Q2_HIGH_NEED_LOW_EXPOSURE": "#DC2626",
    "Q3_LOW_NEED_HIGH_EXPOSURE": "#2563EB",
    "Q4_LOW_NEED_LOW_EXPOSURE": "#94A3B8",
}

_QUADRANT_LABELS: Mapping[str, str] = {
    "Q1_HIGH_NEED_HIGH_EXPOSURE": "Q1 · 높은 Need / 높은 노출",
    "Q2_HIGH_NEED_LOW_EXPOSURE": "Q2 · 높은 Need / 낮은 노출 (우선 병목)",
    "Q3_LOW_NEED_HIGH_EXPOSURE": "Q3 · 낮은 Need / 높은 노출",
    "Q4_LOW_NEED_LOW_EXPOSURE": "Q4 · 낮은 Need / 낮은 노출",
}

_SOURCE_LABELS: Mapping[str, str] = {
    "administrative_office": "행정복지센터",
    "bus_stop": "버스정류장",
    "bus_stop_name_proxy": "버스정류장명 기반 후보",
    "hira_or_public_outreach": "HIRA·공공 방문진료",
    "hospital": "병원",
    "medical_facility": "의료기관",
    "official_dementia_branch": "공식 치매안심센터 분소",
    "official_dementia_center": "공식 치매안심센터",
    "official_village_hall_senior_center": "공식 마을회관·경로당",
    "rail_station": "철도역",
    "senior_center": "경로당",
    "village_hall": "마을회관",
    "welfare_facility": "복지시설",
}


@dataclass(frozen=True)
class Stage3SpatialMapResult:
    """Paths, manifest, and normalized source tables for eight map triplets."""

    figures: Mapping[str, Mapping[str, str]]
    artifact_manifest: pd.DataFrame
    artifact_manifest_path: str
    normalized_tables: Mapping[str, pd.DataFrame]
    analysis_crs: str = "EPSG:5179"


def _require_korean_font() -> str:
    """Register and verify a Hangul-capable font; never silently fall back."""

    try:
        from matplotlib import font_manager
        from matplotlib.ft2font import FT2Font
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError("Stage 3 spatial maps require matplotlib") from exc

    candidates = (
        Path("C:/Windows/Fonts/malgun.ttf"),
        Path("/mnt/c/Windows/Fonts/malgun.ttf"),
        Path("C:/Windows/Fonts/NanumGothic.ttf"),
        Path("/usr/share/fonts/truetype/nanum/NanumGothic.ttf"),
        Path("/usr/share/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJKkr-Regular.otf"),
    )
    for path in candidates:
        if not path.is_file():
            continue
        try:
            charmap = FT2Font(str(path)).get_charmap()
        except (RuntimeError, OSError):
            continue
        if all(ord(character) in charmap for character in "가나다한글"):
            font_manager.fontManager.addfont(str(path))
            return font_manager.FontProperties(fname=str(path)).get_name()
    raise RuntimeError(
        "No verified Hangul-capable font found; install Malgun Gothic, "
        "NanumGothic, or Noto Sans CJK before rendering Stage 3 maps"
    )


def _matplotlib() -> tuple[Any, Any]:
    try:
        import matplotlib as mpl

        mpl.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError("Stage 3 spatial maps require matplotlib") from exc
    return mpl, plt


def _plotly() -> tuple[Any, Any]:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError("Stage 3 spatial maps require plotly") from exc
    return go, make_subplots


def _geopandas() -> Any:
    try:
        import geopandas as gpd
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError("Stage 3 spatial maps require geopandas") from exc
    return gpd


def _table(tables: Mapping[str, Any], role: str, *, required: bool) -> Any | None:
    for key in _TABLE_ALIASES[role]:
        if key in tables:
            value = tables[key]
            if value is None:
                continue
            if not isinstance(value, pd.DataFrame):
                raise TypeError(f"{key} must be a DataFrame or GeoDataFrame")
            if value.empty and required:
                raise ValueError(f"{key} must not be empty")
            return value.copy()
    if required:
        raise KeyError(
            f"missing Stage 3 spatial table for {role}; accepted keys="
            f"{list(_TABLE_ALIASES[role])}"
        )
    return None


def _first(frame: pd.DataFrame, aliases: Sequence[str], label: str) -> str:
    column = next((name for name in aliases if name in frame.columns), None)
    if column is None:
        raise KeyError(f"{label} requires one of {list(aliases)}")
    return column


def _finite_numeric(frame: pd.DataFrame, column: str, label: str) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError(f"{label} must be finite numeric")
    return values.astype(float)


def _as_points(frame: pd.DataFrame, label: str) -> Any:
    """Return non-empty point GeoDataFrame in EPSG:5179."""

    gpd = _geopandas()
    if isinstance(frame, gpd.GeoDataFrame):
        if frame.crs is None:
            raise ValueError(f"{label} GeoDataFrame must declare a CRS")
        result = frame.copy().to_crs(5179)
    else:
        x_col = next(
            (
                column
                for column in (
                    "x_5179",
                    "venue_x_5179",
                    "centroid_x_5179",
                    "x_coord_5179",
                )
                if column in frame.columns
            ),
            None,
        )
        y_col = next(
            (
                column
                for column in (
                    "y_5179",
                    "venue_y_5179",
                    "centroid_y_5179",
                    "y_coord_5179",
                )
                if column in frame.columns
            ),
            None,
        )
        if x_col is not None and y_col is not None:
            x = _finite_numeric(frame, x_col, f"{label}.{x_col}")
            y = _finite_numeric(frame, y_col, f"{label}.{y_col}")
            result = gpd.GeoDataFrame(
                frame.copy(), geometry=gpd.points_from_xy(x, y), crs=5179
            )
        else:
            lon_col = next(
                (column for column in ("longitude", "lon", "lng", "x_wgs84") if column in frame.columns),
                None,
            )
            lat_col = next(
                (column for column in ("latitude", "lat", "y_wgs84") if column in frame.columns),
                None,
            )
            if lon_col is None or lat_col is None:
                raise KeyError(
                    f"{label} requires geometry with CRS, EPSG:5179 x/y, or WGS84 lon/lat"
                )
            lon = _finite_numeric(frame, lon_col, f"{label}.{lon_col}")
            lat = _finite_numeric(frame, lat_col, f"{label}.{lat_col}")
            if not lon.between(120, 140).all() or not lat.between(30, 45).all():
                raise ValueError(f"{label} WGS84 coordinates are outside Korea bounds")
            result = gpd.GeoDataFrame(
                frame.copy(), geometry=gpd.points_from_xy(lon, lat), crs=4326
            ).to_crs(5179)
    if result.empty:
        raise ValueError(f"{label} must not be empty")
    if result.geometry.isna().any() or result.geometry.is_empty.any():
        raise ValueError(f"{label} contains null or empty geometry")
    if not result.geometry.geom_type.eq("Point").all():
        raise ValueError(f"{label} geometry must contain points only")
    if not result.geometry.is_valid.all():
        raise ValueError(f"{label} contains invalid geometry")
    result = result.reset_index(drop=True)
    result["_x_5179"] = result.geometry.x.astype(float)
    result["_y_5179"] = result.geometry.y.astype(float)
    return result


def _as_boundaries(frame: pd.DataFrame | None) -> Any | None:
    if frame is None or frame.empty:
        return None
    gpd = _geopandas()
    if not isinstance(frame, gpd.GeoDataFrame):
        raise TypeError("admin boundaries must be a GeoDataFrame")
    if frame.crs is None:
        raise ValueError("admin boundaries must declare a CRS")
    result = frame.copy().to_crs(5179)
    if result.geometry.isna().any() or result.geometry.is_empty.any():
        raise ValueError("admin boundaries contain null or empty geometry")
    if not result.geometry.geom_type.isin(("Polygon", "MultiPolygon")).all():
        raise ValueError("admin boundaries must contain Polygon/MultiPolygon geometry")
    if not result.geometry.is_valid.all():
        raise ValueError("admin boundaries contain invalid geometry")
    return result.reset_index(drop=True)


def _with_value(
    frame: Any,
    aliases: Sequence[str],
    canonical: str,
    label: str,
    *,
    nonnegative: bool = True,
) -> Any:
    result = frame.copy()
    column = _first(result, aliases, label)
    result[canonical] = _finite_numeric(result, column, f"{label}.{column}")
    if nonnegative and (result[canonical] < 0).any():
        raise ValueError(f"{label} must be non-negative")
    return result


def _venue_id_column(frame: pd.DataFrame) -> str:
    return _first(
        frame,
        ("venue_id", "candidate_venue_id", "venue_location_id"),
        "venue table",
    )


def _admin_column(frame: pd.DataFrame) -> str:
    return _first(
        frame,
        (
            "admin_dong_code",
            "venue_admin_dong_code",
            "policy_admin_dong_code",
            "admin_code",
            "admin",
            "admin_dong_name",
        ),
        "venue admin",
    )


def _normalize_sources(tables: Mapping[str, Any]) -> dict[str, Any]:
    grid = _as_points(_table(tables, "grid", required=True), "population grid")
    grid = _with_value(
        grid,
        (
            "elderly65_calibrated_population",
            "elderly_population",
            "population65",
            "calibrated_population",
            "population",
        ),
        "_elderly_population",
        "population grid",
    )
    venues = _as_points(_table(tables, "venues", required=True), "candidate venues")
    venue_id = _venue_id_column(venues)
    if venues[venue_id].isna().any() or venues[venue_id].astype(str).duplicated().any():
        raise ValueError("candidate venue identifiers must be non-null and unique")
    venues["_venue_id"] = venues[venue_id].astype(str)
    venues = _with_value(
        venues,
        (
            "raw_elderly_exposure",
            "raw_exposure",
            "raw_population_exposure",
            "reachable_elderly_population",
            "reachable_exposure",
            "exposure",
        ),
        "_raw_exposure",
        "venue exposure",
    )
    venues = _with_value(
        venues,
        (
            "need_weighted_exposure",
            "need_exposure",
            "structural_need_exposure",
        ),
        "_need_exposure",
        "venue need exposure",
    )
    boundaries = _as_boundaries(_table(tables, "boundaries", required=True))
    return {
        "grid": grid,
        "venues": venues,
        "boundaries": boundaries,
        "overlap": _table(tables, "overlap", required=False),
        "admin_best": _table(tables, "admin_best", required=False),
        "bundle": _table(tables, "bundle", required=False),
        "rural_urban": _table(tables, "rural_urban", required=False),
    }


def _plot_context(font: str) -> Any:
    mpl, _ = _matplotlib()
    return mpl.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [font],
            "axes.unicode_minus": False,
            "svg.fonttype": "path",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def _boundary_static(axis: Any, boundaries: Any | None) -> None:
    if boundaries is not None:
        boundaries.boundary.plot(
            ax=axis, color="#64748B", linewidth=0.45, alpha=0.75, zorder=1
        )


def _geometry_lines(geometry: Any) -> Iterable[tuple[list[float], list[float]]]:
    if geometry.geom_type == "Polygon":
        x, y = geometry.exterior.xy
        yield list(x), list(y)
        for ring in geometry.interiors:
            x, y = ring.xy
            yield list(x), list(y)
    elif geometry.geom_type == "MultiPolygon":
        for polygon in geometry.geoms:
            yield from _geometry_lines(polygon)


def _geometry_exteriors(geometry: Any) -> Iterable[tuple[list[float], list[float]]]:
    """Yield polygon shells for Plotly categorical fills."""

    if geometry.geom_type == "Polygon":
        x, y = geometry.exterior.xy
        yield list(x), list(y)
    elif geometry.geom_type == "MultiPolygon":
        for polygon in geometry.geoms:
            x, y = polygon.exterior.xy
            yield list(x), list(y)


def _boundary_interactive(figure: Any, boundaries: Any | None, *, row: int | None = None, col: int | None = None) -> None:
    if boundaries is None:
        return
    go, _ = _plotly()
    xs: list[float | None] = []
    ys: list[float | None] = []
    for geometry in boundaries.geometry:
        for x, y in _geometry_lines(geometry):
            xs.extend(x)
            xs.append(None)
            ys.extend(y)
            ys.append(None)
    trace = go.Scattergl(
        x=xs,
        y=ys,
        mode="lines",
        line={"color": "#64748B", "width": 0.7},
        hoverinfo="skip",
        showlegend=False,
        name="행정경계",
    )
    if row is None:
        figure.add_trace(trace)
    else:
        figure.add_trace(trace, row=row, col=col)


def _map_style(axis: Any, title: str) -> None:
    axis.set_title(title, loc="left", fontsize=15, fontweight="bold", pad=12)
    axis.set_aspect("equal", adjustable="datalim")
    axis.set_xlabel("EPSG:5179 X (m)")
    axis.set_ylabel("EPSG:5179 Y (m)")
    axis.grid(color="#E2E8F0", linewidth=0.35, alpha=0.65)
    axis.ticklabel_format(style="plain", useOffset=False)


def _add_north_scale(axis: Any) -> None:
    axis.annotate(
        "N",
        xy=(0.965, 0.97),
        xytext=(0.965, 0.88),
        xycoords="axes fraction",
        textcoords="axes fraction",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        arrowprops={"arrowstyle": "-|>", "color": "#0F172A", "lw": 1.2},
    )
    x0, x1 = axis.get_xlim()
    span = max(float(x1 - x0), 1.0)
    choices = np.asarray([1000, 2000, 5000, 10000, 20000, 50000, 100000])
    target = span * 0.18
    distance = int(choices[np.argmin(np.abs(choices - target))])
    start = x0 + span * 0.06
    y0, y1 = axis.get_ylim()
    y = y0 + (y1 - y0) * 0.055
    axis.plot([start, start + distance], [y, y], color="#0F172A", lw=2.4, zorder=20)
    axis.text(
        start + distance / 2,
        y + (y1 - y0) * 0.015,
        f"{distance / 1000:g} km",
        ha="center",
        va="bottom",
        fontsize=8,
        zorder=20,
    )


def _quantile_code(values: pd.Series, bins: int = 6) -> tuple[np.ndarray, list[str]]:
    raw = values.to_numpy(dtype=float)
    if not np.isfinite(raw).all():
        raise ValueError("map classification values must be finite")
    if np.allclose(raw, raw[0]):
        return np.zeros(len(raw), dtype=int), [f"{raw[0]:,.2f}"]
    quantiles = np.unique(np.quantile(raw, np.linspace(0.0, 1.0, bins + 1)))
    if len(quantiles) < 2:
        return np.zeros(len(raw), dtype=int), [f"{raw[0]:,.2f}"]
    code = np.searchsorted(quantiles[1:-1], raw, side="right")

    # Fixed one-decimal legends can collapse distinct adjacent breaks into
    # labels such as ``2.1–2.1``.  Use the least precision that preserves all
    # actual class boundaries.
    formatted: list[str] = []
    for precision in range(0, 9):
        candidate = [f"{value:,.{precision}f}" for value in quantiles]
        if len(set(candidate)) == len(candidate):
            formatted = candidate
            break
    if not formatted:  # exceptionally close floating-point breaks
        formatted = [f"{value:,.10g}" for value in quantiles]
    labels = [
        f"Q{index + 1} · {formatted[index]}–{formatted[index + 1]}"
        for index in range(len(quantiles) - 1)
    ]
    return code.astype(int), labels


def _source_categories(venues: pd.DataFrame) -> tuple[pd.Series, list[str]]:
    source = next(
        (
            column
            for column in ("source_kind", "venue_type", "source_type", "candidate_type")
            if column in venues.columns
        ),
        None,
    )
    if source is None:
        values = pd.Series("후보지", index=venues.index, dtype="string")
        return values, ["후보지"]
    raw_values = venues[source].fillna("미상").astype(str).str.strip()
    values = raw_values.map(lambda value: _SOURCE_LABELS.get(value.lower(), value))
    top = values.value_counts().head(6).index.tolist()
    collapsed = values.where(values.isin(top), "기타")
    order = [name for name in top if name in set(collapsed)]
    if collapsed.eq("기타").any():
        order.append("기타")
    return collapsed, order


def _plotly_layout(figure: Any, title: str, font: str) -> Any:
    figure.update_layout(
        title={"text": title, "x": 0.02, "xanchor": "left"},
        template="plotly_white",
        font={"family": f"{font}, Malgun Gothic, NanumGothic, sans-serif"},
        margin={"l": 60, "r": 35, "t": 75, "b": 55},
        hoverlabel={"font": {"family": f"{font}, sans-serif"}},
    )
    return figure


def _elderly_grid_figure(sources: Mapping[str, Any], font: str) -> tuple[Any, Any, pd.DataFrame]:
    grid = sources["grid"].copy()
    boundary = sources["boundaries"]
    codes, labels = _quantile_code(grid["_elderly_population"])
    grid["_population_class"] = codes
    normalized = pd.DataFrame(
        {
            "x_5179": grid["_x_5179"],
            "y_5179": grid["_y_5179"],
            "elderly_population": grid["_elderly_population"],
            "population_class": codes,
            "population_class_label": [labels[int(code)] for code in codes],
        }
    )
    if "grid_point_index" in grid.columns:
        normalized.insert(0, "grid_point_index", grid["grid_point_index"].astype(str))

    mpl, plt = _matplotlib()
    with _plot_context(font):
        static, axis = plt.subplots(figsize=(11.2, 8.2), constrained_layout=True)
        _boundary_static(axis, boundary)
        scatter = axis.scatter(
            grid["_x_5179"],
            grid["_y_5179"],
            c=codes,
            cmap="magma_r",
            s=5.0,
            marker="s",
            linewidths=0,
            alpha=0.88,
            rasterized=True,
            zorder=2,
        )
        colourbar = static.colorbar(scatter, ax=axis, fraction=0.035, pad=0.025)
        colourbar.set_label("고령인구 보정값 분위 등급")
        colourbar.set_ticks(np.arange(len(labels)))
        colourbar.set_ticklabels(labels)
        _map_style(axis, "100m 격자 고령인구 분포 · 보정 인구질량")
        _add_north_scale(axis)
        axis.text(
            0.0,
            -0.105,
            f"격자 {len(grid):,}개 · 합계 {grid['_elderly_population'].sum():,.0f}명 · EPSG:5179",
            transform=axis.transAxes,
            fontsize=9,
            color="#475569",
        )

    go, _ = _plotly()
    interactive = go.Figure()
    _boundary_interactive(interactive, boundary)
    hover = [
        f"고령인구 보정값: {value:,.2f}<br>분위: {labels[int(code)]}"
        for value, code in zip(grid["_elderly_population"], codes)
    ]
    interactive.add_trace(
        go.Scattergl(
            x=grid["_x_5179"],
            y=grid["_y_5179"],
            mode="markers",
            marker={
                "size": 4,
                "color": np.log1p(grid["_elderly_population"]),
                "colorscale": "Magma",
                "reversescale": True,
                "colorbar": {"title": "log(1+고령인구)"},
                "opacity": 0.82,
            },
            text=hover,
            hovertemplate="%{text}<extra></extra>",
            name="100m 격자",
        )
    )
    _plotly_layout(interactive, "100m 격자 고령인구 분포 · 보정 인구질량", font)
    interactive.update_yaxes(scaleanchor="x", scaleratio=1, title="EPSG:5179 Y (m)")
    interactive.update_xaxes(title="EPSG:5179 X (m)")
    return static, interactive, normalized


def _candidate_venue_figure(sources: Mapping[str, Any], font: str) -> tuple[Any, Any, pd.DataFrame]:
    venues = sources["venues"].copy()
    boundary = sources["boundaries"]
    categories, order = _source_categories(venues)
    venues["_source_category"] = categories
    palette = dict(zip(order, _BUNDLE_COLOURS * 2))
    normalized = pd.DataFrame(
        {
            "venue_id": venues["_venue_id"],
            "x_5179": venues["_x_5179"],
            "y_5179": venues["_y_5179"],
            "source_category": categories,
            "raw_exposure": venues["_raw_exposure"],
            "need_exposure": venues["_need_exposure"],
        }
    )

    _, plt = _matplotlib()
    with _plot_context(font):
        static, axis = plt.subplots(figsize=(11.2, 8.2), constrained_layout=True)
        _boundary_static(axis, boundary)
        for category in order:
            subset = venues.loc[venues["_source_category"].eq(category)]
            axis.scatter(
                subset["_x_5179"],
                subset["_y_5179"],
                s=12,
                c=palette[category],
                edgecolors="white",
                linewidths=0.18,
                alpha=0.8,
                label=f"{category} (n={len(subset):,})",
                zorder=3,
            )
        _map_style(axis, "Stage 3 분석대상 후보지 · 자료원별 공간 분포")
        _add_north_scale(axis)
        axis.legend(
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            frameon=False,
            fontsize=8.5,
            title="후보지 자료원",
        )

    go, _ = _plotly()
    interactive = go.Figure()
    _boundary_interactive(interactive, boundary)
    for category in order:
        subset = venues.loc[venues["_source_category"].eq(category)]
        interactive.add_trace(
            go.Scattergl(
                x=subset["_x_5179"],
                y=subset["_y_5179"],
                mode="markers",
                marker={"size": 7, "color": palette[category], "opacity": 0.78},
                customdata=np.column_stack(
                    [subset["_venue_id"], subset["_raw_exposure"], subset["_need_exposure"]]
                ),
                hovertemplate=(
                    "후보지: %{customdata[0]}<br>"
                    "원시 노출: %{customdata[1]:,.1f}<br>"
                    "Need 노출: %{customdata[2]:,.1f}<extra>" + html_module.escape(category) + "</extra>"
                ),
                name=f"{category} (n={len(subset):,})",
            )
        )
    _plotly_layout(interactive, "Stage 3 분석대상 후보지 · 자료원별 공간 분포", font)
    interactive.update_yaxes(scaleanchor="x", scaleratio=1, title="EPSG:5179 Y (m)")
    interactive.update_xaxes(title="EPSG:5179 X (m)")
    return static, interactive, normalized


def _venue_exposure_figure(sources: Mapping[str, Any], font: str) -> tuple[Any, Any, pd.DataFrame]:
    venues = sources["venues"].copy()
    normalized = pd.DataFrame(
        {
            "venue_id": venues["_venue_id"],
            "raw_exposure": venues["_raw_exposure"],
            "need_exposure": venues["_need_exposure"],
        }
    )
    raw_log = np.log1p(normalized["raw_exposure"])
    need_log = np.log1p(normalized["need_exposure"])
    rho = float(normalized[["raw_exposure", "need_exposure"]].corr(method="spearman").iloc[0, 1])

    _, plt = _matplotlib()
    with _plot_context(font):
        static, axes = plt.subplots(1, 2, figsize=(12.2, 6.7), constrained_layout=True)
        axes[0].hist(raw_log, bins=32, color="#2563EB", alpha=0.76, label="원시 고령인구")
        axes[0].hist(need_log, bins=32, color="#D97706", alpha=0.58, label="Need 가중")
        axes[0].axvline(raw_log.median(), color="#1D4ED8", lw=1.3, ls="--")
        axes[0].axvline(need_log.median(), color="#B45309", lw=1.3, ls="--")
        axes[0].set_title("분포와 중앙값", loc="left", fontweight="bold")
        axes[0].set_xlabel("log(1 + 도달 노출량)")
        axes[0].set_ylabel("후보지 수")
        axes[0].legend(frameon=False)
        hexbin = axes[1].hexbin(
            raw_log,
            need_log,
            gridsize=35,
            mincnt=1,
            cmap="viridis",
            linewidths=0.15,
        )
        static.colorbar(hexbin, ax=axes[1], fraction=0.045, pad=0.025, label="후보지 밀도")
        lower = min(float(raw_log.min()), float(need_log.min()))
        upper = max(float(raw_log.max()), float(need_log.max()))
        axes[1].plot([lower, upper], [lower, upper], ls="--", lw=1, color="#64748B")
        axes[1].set_title(f"원시–Need 노출 분리력 · Spearman ρ={rho:.3f}", loc="left", fontweight="bold")
        axes[1].set_xlabel("log(1 + 원시 고령인구 노출)")
        axes[1].set_ylabel("log(1 + Need 가중 노출)")
        static.suptitle("후보지 노출량 전체 분포", fontsize=16, fontweight="bold", x=0.01, ha="left")

    go, make_subplots = _plotly()
    interactive = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("노출량 히스토그램", f"원시–Need 노출 · ρ={rho:.3f}"),
    )
    interactive.add_trace(
        go.Histogram(x=raw_log, nbinsx=32, opacity=0.72, name="원시 고령인구", marker_color="#2563EB"),
        row=1,
        col=1,
    )
    interactive.add_trace(
        go.Histogram(x=need_log, nbinsx=32, opacity=0.58, name="Need 가중", marker_color="#D97706"),
        row=1,
        col=1,
    )
    interactive.add_trace(
        go.Scattergl(
            x=raw_log,
            y=need_log,
            mode="markers",
            marker={"size": 6, "color": need_log, "colorscale": "Viridis", "opacity": 0.6},
            customdata=np.column_stack([normalized["venue_id"], normalized["raw_exposure"], normalized["need_exposure"]]),
            hovertemplate=(
                "후보지: %{customdata[0]}<br>원시: %{customdata[1]:,.1f}<br>"
                "Need: %{customdata[2]:,.1f}<extra></extra>"
            ),
            name="후보지",
        ),
        row=1,
        col=2,
    )
    _plotly_layout(interactive, "후보지 노출량 전체 분포", font)
    interactive.update_layout(barmode="overlay")
    interactive.update_xaxes(title="log(1 + 노출량)", row=1, col=1)
    interactive.update_xaxes(title="log(1 + 원시 노출)", row=1, col=2)
    interactive.update_yaxes(title="후보지 수", row=1, col=1)
    interactive.update_yaxes(title="log(1 + Need 노출)", row=1, col=2)
    return static, interactive, normalized


def _cross_admin_values(venues: Any) -> Any:
    result = venues.copy()
    share_col = next(
        (
            column
            for column in (
                "cross_admin_share",
                "outside_admin_share",
                "cross_boundary_share",
                "cross_admin_coverage_share",
            )
            if column in result.columns
        ),
        None,
    )
    if share_col is not None:
        result["_cross_share"] = _finite_numeric(result, share_col, "cross-admin share")
        result["_cross_exposure"] = result["_cross_share"] * result["_raw_exposure"]
    else:
        direct = next(
            (
                column
                for column in ("cross_admin_exposure", "outside_admin_exposure", "cross_boundary_exposure")
                if column in result.columns
            ),
            None,
        )
        parts = [
            column
            for column in ("same_sigungu_other_admin_exposure", "cross_sigungu_exposure")
            if column in result.columns
        ]
        if direct is not None:
            result["_cross_exposure"] = _finite_numeric(result, direct, "cross-admin exposure")
        elif parts:
            values = result.loc[:, parts].apply(pd.to_numeric, errors="coerce")
            if values.isna().any().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
                raise ValueError("cross-admin exposure components must be finite")
            result["_cross_exposure"] = values.sum(axis=1)
        else:
            raise KeyError(
                "cross-admin map requires a cross-admin share, exposure, or exposure components"
            )
        denominator = result["_raw_exposure"].to_numpy(dtype=float)
        result["_cross_share"] = np.divide(
            result["_cross_exposure"].to_numpy(dtype=float),
            denominator,
            out=np.zeros(len(result), dtype=float),
            where=denominator > 0,
        )
    if (result["_cross_exposure"] < -1e-9).any() or not result["_cross_share"].between(0, 1 + 1e-6).all():
        raise ValueError("cross-admin exposure/share must be non-negative and share <= 1")
    result["_cross_share"] = result["_cross_share"].clip(0, 1)
    return result


def _cross_admin_figure(sources: Mapping[str, Any], font: str) -> tuple[Any, Any, pd.DataFrame]:
    venues = _cross_admin_values(sources["venues"])
    boundary = sources["boundaries"]
    normalized = pd.DataFrame(
        {
            "venue_id": venues["_venue_id"],
            "x_5179": venues["_x_5179"],
            "y_5179": venues["_y_5179"],
            "raw_exposure": venues["_raw_exposure"],
            "cross_admin_exposure": venues["_cross_exposure"],
            "cross_admin_share": venues["_cross_share"],
        }
    )
    sizes = 10 + 45 * np.sqrt(venues["_raw_exposure"] / max(float(venues["_raw_exposure"].max()), 1.0))

    _, plt = _matplotlib()
    with _plot_context(font):
        static, axis = plt.subplots(figsize=(11.2, 8.2), constrained_layout=True)
        _boundary_static(axis, boundary)
        scatter = axis.scatter(
            venues["_x_5179"],
            venues["_y_5179"],
            c=venues["_cross_share"],
            cmap="plasma",
            vmin=0,
            vmax=1,
            s=sizes,
            alpha=0.76,
            edgecolors="white",
            linewidths=0.2,
            zorder=3,
        )
        static.colorbar(scatter, ax=axis, fraction=0.035, pad=0.025, label="행정동 경계 밖 도달 비중")
        _map_style(axis, "행정경계 교차 도달 · 후보지별 외부 인구 비중")
        _add_north_scale(axis)
        axis.text(
            0,
            -0.105,
            "색=경계 밖 도달 비중 · 점 크기=전체 원시 노출량",
            transform=axis.transAxes,
            fontsize=9,
            color="#475569",
        )

    go, _ = _plotly()
    interactive = go.Figure()
    _boundary_interactive(interactive, boundary)
    interactive.add_trace(
        go.Scattergl(
            x=venues["_x_5179"],
            y=venues["_y_5179"],
            mode="markers",
            marker={
                "size": np.clip(sizes / 3, 5, 18),
                "color": venues["_cross_share"],
                "cmin": 0,
                "cmax": 1,
                "colorscale": "Plasma",
                "colorbar": {"title": "경계 밖 비중"},
                "opacity": 0.75,
            },
            customdata=np.column_stack(
                [venues["_venue_id"], venues["_cross_exposure"], venues["_cross_share"], venues["_raw_exposure"]]
            ),
            hovertemplate=(
                "후보지: %{customdata[0]}<br>경계 밖 노출: %{customdata[1]:,.1f}<br>"
                "경계 밖 비중: %{customdata[2]:.1%}<br>전체 노출: %{customdata[3]:,.1f}<extra></extra>"
            ),
            name="후보지",
        )
    )
    _plotly_layout(interactive, "행정경계 교차 도달 · 후보지별 외부 인구 비중", font)
    interactive.update_yaxes(scaleanchor="x", scaleratio=1, title="EPSG:5179 Y (m)")
    interactive.update_xaxes(title="EPSG:5179 X (m)")
    return static, interactive, normalized


def _prepare_overlap(sources: Mapping[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = sources["overlap"]
    if raw is None:
        raise KeyError(
            "overlap_redundancy_map requires overlap_edges/venue_overlap_edges"
        )
    if raw.empty:
        raise ValueError("overlap edge table must not be empty")
    source_col = _first(
        raw,
        ("venue_id_a", "venue_i", "source_venue_id", "venue_id_1", "from_venue_id"),
        "overlap source",
    )
    target_col = _first(
        raw,
        ("venue_id_b", "venue_j", "target_venue_id", "venue_id_2", "to_venue_id"),
        "overlap target",
    )
    score_col = _first(
        raw,
        (
            "weighted_overlap",
            "weighted_overlap_share",
            "population_weighted_jaccard",
            "need_weighted_jaccard",
            "jaccard_overlap",
            "jaccard",
            "overlap_score",
        ),
        "overlap score",
    )
    if raw[[source_col, target_col]].isna().any().any():
        raise ValueError("overlap endpoints must be non-null")
    edges = pd.DataFrame(
        {
            "venue_id_a": raw[source_col].astype(str),
            "venue_id_b": raw[target_col].astype(str),
            "overlap_score": _finite_numeric(raw, score_col, "overlap score"),
        }
    )
    if edges["venue_id_a"].eq(edges["venue_id_b"]).any():
        raise ValueError("overlap edges must not contain self-loops")
    if not edges["overlap_score"].between(0, 1 + 1e-6).all():
        raise ValueError("overlap scores must lie in [0, 1]")
    edges["overlap_score"] = edges["overlap_score"].clip(0, 1)
    undirected = np.sort(edges[["venue_id_a", "venue_id_b"]].to_numpy(dtype=str), axis=1)
    edges["_pair"] = undirected[:, 0] + "\x1f" + undirected[:, 1]
    if edges["_pair"].duplicated().any():
        raise ValueError("overlap table contains duplicate undirected venue pairs")
    edges = edges.drop(columns="_pair")

    venues = sources["venues"]
    coordinates = venues.set_index("_venue_id")[["_x_5179", "_y_5179"]]
    endpoints = set(edges["venue_id_a"]) | set(edges["venue_id_b"])
    missing = sorted(endpoints - set(coordinates.index))
    if missing:
        raise ValueError(f"overlap endpoints missing from candidate venues: {missing[:5]}")
    edges = edges.join(coordinates.add_suffix("_a"), on="venue_id_a")
    edges = edges.join(coordinates.add_suffix("_b"), on="venue_id_b")

    degree_a = edges.groupby("venue_id_a")["overlap_score"].agg(["count", "sum"])
    degree_b = edges.groupby("venue_id_b")["overlap_score"].agg(["count", "sum"])
    degree = degree_a.add(degree_b, fill_value=0).rename(
        columns={"count": "overlap_degree", "sum": "weighted_redundancy"}
    )
    nodes = venues[["_venue_id", "_x_5179", "_y_5179"]].copy()
    nodes = nodes.join(degree, on="_venue_id")
    nodes[["overlap_degree", "weighted_redundancy"]] = nodes[
        ["overlap_degree", "weighted_redundancy"]
    ].fillna(0.0)
    return edges.sort_values("overlap_score", ascending=False).reset_index(drop=True), nodes


def _overlap_hotspots(
    edges: pd.DataFrame,
    nodes: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, float]:
    """Aggregate dense local overlap links and select one auditable zoom area."""

    clustered_nodes = nodes.copy()
    x_span = float(clustered_nodes["_x_5179"].max() - clustered_nodes["_x_5179"].min())
    y_span = float(clustered_nodes["_y_5179"].max() - clustered_nodes["_y_5179"].min())
    cell_size = float(np.clip(max(x_span, y_span, 1.0) / 70.0, 1_000.0, 5_000.0))
    x_origin = float(np.floor(clustered_nodes["_x_5179"].min() / cell_size) * cell_size)
    y_origin = float(np.floor(clustered_nodes["_y_5179"].min() / cell_size) * cell_size)
    clustered_nodes["_cell_x"] = np.floor(
        (clustered_nodes["_x_5179"] - x_origin) / cell_size
    ).astype("int64")
    clustered_nodes["_cell_y"] = np.floor(
        (clustered_nodes["_y_5179"] - y_origin) / cell_size
    ).astype("int64")
    clustered_nodes["_cluster_id"] = (
        "C"
        + clustered_nodes["_cell_x"].astype(str)
        + "_"
        + clustered_nodes["_cell_y"].astype(str)
    )

    clusters = (
        clustered_nodes.groupby("_cluster_id", as_index=False, sort=True)
        .agg(
            x_5179=("_x_5179", "mean"),
            y_5179=("_y_5179", "mean"),
            venue_count=("_venue_id", "size"),
            weighted_redundancy=("weighted_redundancy", "sum"),
            mean_weighted_redundancy=("weighted_redundancy", "mean"),
            overlap_degree=("overlap_degree", "sum"),
        )
        .rename(columns={"_cluster_id": "cluster_id"})
    )
    clusters = clusters.sort_values(
        ["weighted_redundancy", "venue_count", "cluster_id"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    clusters["hotspot_rank"] = np.arange(1, len(clusters) + 1, dtype=np.int64)
    hotspot_limit = min(
        len(clusters), max(12, int(np.ceil(np.sqrt(max(len(clusters), 1)) * 3)))
    )
    clusters["displayed_hotspot"] = clusters["hotspot_rank"].le(hotspot_limit)

    node_cluster = clustered_nodes.set_index("_venue_id")["_cluster_id"]
    clustered_edges = edges.copy()
    clustered_edges["cluster_a"] = clustered_edges["venue_id_a"].map(node_cluster)
    clustered_edges["cluster_b"] = clustered_edges["venue_id_b"].map(node_cluster)
    pair = np.sort(
        clustered_edges[["cluster_a", "cluster_b"]].to_numpy(dtype=str), axis=1
    )
    clustered_edges["cluster_a"] = pair[:, 0]
    clustered_edges["cluster_b"] = pair[:, 1]
    cluster_edges = (
        clustered_edges.groupby(["cluster_a", "cluster_b"], as_index=False, sort=True)
        .agg(
            edge_count=("overlap_score", "size"),
            overlap_sum=("overlap_score", "sum"),
            overlap_mean=("overlap_score", "mean"),
            overlap_max=("overlap_score", "max"),
        )
    )
    coordinates = clusters.set_index("cluster_id")[["x_5179", "y_5179"]]
    cluster_edges = cluster_edges.join(coordinates.add_suffix("_a"), on="cluster_a")
    cluster_edges = cluster_edges.join(coordinates.add_suffix("_b"), on="cluster_b")
    hotspot_ids = set(
        clusters.loc[clusters["displayed_hotspot"], "cluster_id"].astype(str)
    )
    cluster_edges["displayed"] = (
        cluster_edges["cluster_a"].ne(cluster_edges["cluster_b"])
        & cluster_edges["cluster_a"].isin(hotspot_ids)
        & cluster_edges["cluster_b"].isin(hotspot_ids)
    )
    displayed_cluster_edges = (
        cluster_edges.loc[cluster_edges["displayed"]]
        .sort_values(["overlap_sum", "overlap_max"], ascending=False)
        .head(200)
        .copy()
    )

    top_cluster = str(clusters.iloc[0]["cluster_id"])
    seed_ids = set(
        clustered_nodes.loc[
            clustered_nodes["_cluster_id"].eq(top_cluster), "_venue_id"
        ].astype(str)
    )
    zoom_edges = edges.loc[
        edges["venue_id_a"].isin(seed_ids) | edges["venue_id_b"].isin(seed_ids)
    ].head(120).copy()
    if zoom_edges.empty:
        zoom_edges = edges.head(min(100, len(edges))).copy()
    zoom_ids = seed_ids | set(zoom_edges["venue_id_a"]) | set(zoom_edges["venue_id_b"])
    zoom_nodes = clustered_nodes.loc[
        clustered_nodes["_venue_id"].isin(zoom_ids)
    ].copy()
    return clusters, displayed_cluster_edges, zoom_edges, zoom_nodes, cell_size


def _overlap_figure(sources: Mapping[str, Any], font: str) -> tuple[Any, Any, pd.DataFrame]:
    edges, nodes = _prepare_overlap(sources)
    boundary = sources["boundaries"]
    clusters, cluster_edges, zoom_edges, zoom_nodes, cell_size = _overlap_hotspots(
        edges, nodes
    )
    hotspots = clusters.loc[clusters["displayed_hotspot"]].copy()

    _, plt = _matplotlib()
    from matplotlib.collections import LineCollection

    with _plot_context(font):
        static, axes = plt.subplots(1, 2, figsize=(13.8, 7.4), constrained_layout=True)
        _boundary_static(axes[0], boundary)
        if not cluster_edges.empty:
            province_segments = np.stack(
                [
                    cluster_edges[["x_5179_a", "y_5179_a"]].to_numpy(dtype=float),
                    cluster_edges[["x_5179_b", "y_5179_b"]].to_numpy(dtype=float),
                ],
                axis=1,
            )
            axes[0].add_collection(
                LineCollection(
                    province_segments,
                    colors="#64748B",
                    linewidths=np.clip(
                        0.35 + np.log1p(cluster_edges["edge_count"]) / 2.5,
                        0.4,
                        2.2,
                    ),
                    alpha=0.32,
                    zorder=2,
                )
            )
        axes[0].scatter(
            clusters["x_5179"],
            clusters["y_5179"],
            s=5,
            color="#CBD5E1",
            alpha=0.5,
            linewidths=0,
            zorder=2,
        )
        hotspot_strength = np.log1p(hotspots["weighted_redundancy"])
        points = axes[0].scatter(
            hotspots["x_5179"],
            hotspots["y_5179"],
            c=hotspot_strength,
            cmap="viridis",
            s=np.clip(22 + 8 * np.sqrt(hotspots["venue_count"]), 24, 95),
            edgecolors="white",
            linewidths=0.35,
            alpha=0.9,
            zorder=3,
        )
        colour_axis = axes[0].inset_axes([0.53, 0.025, 0.42, 0.022])
        colourbar = static.colorbar(
            points,
            cax=colour_axis,
            orientation="horizontal",
        )
        colourbar.set_label("log(1 + 셀 가중 중복도 합계)", fontsize=7)
        colourbar.ax.xaxis.set_label_position("top")
        colourbar.ax.tick_params(labelsize=7)
        top_hotspot = hotspots.iloc[0]
        axes[0].scatter(
            [top_hotspot["x_5179"]],
            [top_hotspot["y_5179"]],
            s=125,
            facecolors="none",
            edgecolors="#DC2626",
            linewidths=1.5,
            zorder=4,
        )
        _map_style(axes[0], "도 전체 공간집계 중복 hotspot")
        _add_north_scale(axes[0])
        axes[0].text(
            0,
            -0.105,
            f"전체 edge {len(edges):,}개 → {cell_size / 1000:.1f}km 셀 {len(clusters):,}개 · 상위 hotspot {len(hotspots):,}개 · 붉은 고리=H1",
            transform=axes[0].transAxes,
            fontsize=9,
            color="#475569",
        )

        _boundary_static(axes[1], boundary)
        zoom_segments = np.stack(
            [
                zoom_edges[["_x_5179_a", "_y_5179_a"]].to_numpy(dtype=float),
                zoom_edges[["_x_5179_b", "_y_5179_b"]].to_numpy(dtype=float),
            ],
            axis=1,
        )
        axes[1].add_collection(
            LineCollection(
                zoom_segments,
                colors="#7C3AED",
                linewidths=0.45 + 2.2 * zoom_edges["overlap_score"].to_numpy(dtype=float),
                alpha=0.48,
                zorder=2,
            )
        )
        axes[1].scatter(
            zoom_nodes["_x_5179"],
            zoom_nodes["_y_5179"],
            c=zoom_nodes["weighted_redundancy"],
            cmap="viridis",
            s=np.clip(18 + 4 * np.sqrt(zoom_nodes["overlap_degree"]), 18, 75),
            edgecolors="white",
            linewidths=0.35,
            alpha=0.92,
            zorder=3,
        )
        zoom_x = np.concatenate(
            [zoom_edges["_x_5179_a"].to_numpy(), zoom_edges["_x_5179_b"].to_numpy()]
        )
        zoom_y = np.concatenate(
            [zoom_edges["_y_5179_a"].to_numpy(), zoom_edges["_y_5179_b"].to_numpy()]
        )
        x_pad = max(float(np.ptp(zoom_x)) * 0.12, cell_size * 0.35)
        y_pad = max(float(np.ptp(zoom_y)) * 0.12, cell_size * 0.35)
        axes[1].set_xlim(float(zoom_x.min()) - x_pad, float(zoom_x.max()) + x_pad)
        axes[1].set_ylim(float(zoom_y.min()) - y_pad, float(zoom_y.max()) + y_pad)
        _map_style(axes[1], "H1 hotspot 상세 · 실제 후보지 연결")
        axes[1].set_aspect("equal", adjustable="box")
        _add_north_scale(axes[1])
        axes[1].text(
            0,
            -0.105,
            f"H1과 연결된 edge 중 상위 {len(zoom_edges):,}개 · 선 굵기=도달권 중복도",
            transform=axes[1].transAxes,
            fontsize=9,
            color="#475569",
        )
        static.suptitle(
            "후보지 도달권 중복·대체가능성: hotspot 요약과 국소 구조",
            fontsize=16,
            fontweight="bold",
            x=0.01,
            ha="left",
        )

    go, make_subplots = _plotly()
    interactive = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("도 전체 공간집계 hotspot", "H1 hotspot 실제 연결"),
    )
    _boundary_interactive(interactive, boundary, row=1, col=1)
    _boundary_interactive(interactive, boundary, row=1, col=2)
    line_x: list[float | None] = []
    line_y: list[float | None] = []
    for x_a, y_a, x_b, y_b in cluster_edges[
        ["x_5179_a", "y_5179_a", "x_5179_b", "y_5179_b"]
    ].itertuples(index=False, name=None):
        line_x.extend([x_a, x_b, None])
        line_y.extend([y_a, y_b, None])
    interactive.add_trace(
        go.Scattergl(
            x=line_x,
            y=line_y,
            mode="lines",
            line={"color": "rgba(100,116,139,0.30)", "width": 0.8},
            hoverinfo="skip",
            name=f"hotspot 간 집계 연결 (n={len(cluster_edges):,})",
        ),
        row=1,
        col=1,
    )
    interactive.add_trace(
        go.Scattergl(
            x=hotspots["x_5179"],
            y=hotspots["y_5179"],
            mode="markers",
            marker={
                "size": np.clip(7 + 2 * np.sqrt(hotspots["venue_count"]), 7, 18),
                "color": np.log1p(hotspots["weighted_redundancy"]),
                "colorscale": "Viridis",
                "colorbar": {"title": "log(1+셀 중복도)", "x": 0.46},
                "opacity": 0.86,
            },
            customdata=np.column_stack(
                [
                    hotspots["hotspot_rank"],
                    hotspots["venue_count"],
                    hotspots["weighted_redundancy"],
                    hotspots["mean_weighted_redundancy"],
                ]
            ),
            hovertemplate=(
                "hotspot H%{customdata[0]:.0f}<br>후보지 수: %{customdata[1]:,.0f}<br>"
                "가중 중복도 합계: %{customdata[2]:,.2f}<br>"
                "후보지 평균: %{customdata[3]:,.2f}<extra></extra>"
            ),
            name="공간집계 hotspot",
        ),
        row=1,
        col=1,
    )

    zoom_line_x: list[float | None] = []
    zoom_line_y: list[float | None] = []
    for x_a, y_a, x_b, y_b in zoom_edges[
        ["_x_5179_a", "_y_5179_a", "_x_5179_b", "_y_5179_b"]
    ].itertuples(index=False, name=None):
        zoom_line_x.extend([x_a, x_b, None])
        zoom_line_y.extend([y_a, y_b, None])
    interactive.add_trace(
        go.Scattergl(
            x=zoom_line_x,
            y=zoom_line_y,
            mode="lines",
            line={"color": "rgba(124,58,237,0.42)", "width": 1.15},
            hoverinfo="skip",
            name=f"H1 실제 edge (n={len(zoom_edges):,})",
        ),
        row=1,
        col=2,
    )
    interactive.add_trace(
        go.Scattergl(
            x=zoom_nodes["_x_5179"],
            y=zoom_nodes["_y_5179"],
            mode="markers",
            marker={
                "size": np.clip(6 + np.sqrt(zoom_nodes["overlap_degree"]), 6, 17),
                "color": zoom_nodes["weighted_redundancy"],
                "colorscale": "Viridis",
                "opacity": 0.86,
                "showscale": False,
            },
            customdata=np.column_stack(
                [zoom_nodes["_venue_id"], zoom_nodes["overlap_degree"], zoom_nodes["weighted_redundancy"]]
            ),
            hovertemplate=(
                "후보지: %{customdata[0]}<br>중복 연결수: %{customdata[1]:,.0f}<br>"
                "가중 중복도: %{customdata[2]:,.3f}<extra></extra>"
            ),
            name="H1 인접 후보지",
        ),
        row=1,
        col=2,
    )
    _plotly_layout(
        interactive,
        "후보지 도달권 중복·대체가능성: hotspot 요약과 국소 구조",
        font,
    )
    interactive.update_yaxes(scaleanchor="x", scaleratio=1, title="EPSG:5179 Y (m)", row=1, col=1)
    interactive.update_xaxes(title="EPSG:5179 X (m)", row=1, col=1)
    interactive.update_yaxes(scaleanchor="x2", scaleratio=1, title="EPSG:5179 Y (m)", row=1, col=2)
    interactive.update_xaxes(title="EPSG:5179 X (m)", row=1, col=2)

    cluster_normalized = clusters.assign(record_type="hotspot_cluster").rename(
        columns={"displayed_hotspot": "displayed"}
    )
    link_normalized = cluster_edges.assign(
        record_type="cluster_link",
        cluster_id=cluster_edges["cluster_a"] + "|" + cluster_edges["cluster_b"],
        x_5179=(cluster_edges["x_5179_a"] + cluster_edges["x_5179_b"]) / 2,
        y_5179=(cluster_edges["y_5179_a"] + cluster_edges["y_5179_b"]) / 2,
        displayed=True,
    )
    zoom_normalized = zoom_edges.assign(
        record_type="zoom_edge",
        cluster_id=zoom_edges["venue_id_a"] + "|" + zoom_edges["venue_id_b"],
        x_5179=(zoom_edges["_x_5179_a"] + zoom_edges["_x_5179_b"]) / 2,
        y_5179=(zoom_edges["_y_5179_a"] + zoom_edges["_y_5179_b"]) / 2,
        displayed=True,
    )
    normalized = pd.concat(
        [cluster_normalized, link_normalized, zoom_normalized],
        ignore_index=True,
        sort=False,
    )
    normalized["aggregation_cell_m"] = cell_size
    return static, interactive, normalized


def _prepare_admin_best(sources: Mapping[str, Any]) -> Any:
    venues = sources["venues"]
    supplied = sources["admin_best"]
    if supplied is None:
        admin_col = _admin_column(venues)
        best = venues.copy()
        best["_admin"] = best[admin_col].astype(str)
    else:
        venue_col = _venue_id_column(supplied)
        supplied = supplied.copy()
        supplied["_venue_id"] = supplied[venue_col].astype(str)
        if supplied["_venue_id"].duplicated().any():
            rank_col = next(
                (column for column in ("admin_rank", "regional_rank", "rank_within_admin") if column in supplied.columns),
                None,
            )
            flag_col = next(
                (column for column in ("is_admin_best", "admin_best_flag") if column in supplied.columns),
                None,
            )
            if rank_col is not None:
                supplied = supplied.loc[pd.to_numeric(supplied[rank_col], errors="coerce").eq(1)].copy()
            elif flag_col is not None:
                supplied = supplied.loc[supplied[flag_col].fillna(False).astype(bool)].copy()
        venue_columns = venues.set_index("_venue_id")
        missing = sorted(set(supplied["_venue_id"]) - set(venue_columns.index))
        if missing:
            raise ValueError(f"admin-best venues missing from candidate table: {missing[:5]}")
        keep = [
            column
            for column in ("_x_5179", "_y_5179", "_raw_exposure", "_need_exposure")
            if column not in supplied.columns
        ]
        best = supplied.join(venue_columns[keep], on="_venue_id")
        admin_col = _admin_column(best)
        best["_admin"] = best[admin_col].astype(str)
        if "_need_exposure" not in best.columns:
            need_col = _first(
                best,
                ("need_weighted_exposure", "need_exposure", "structural_need_exposure"),
                "admin-best need exposure",
            )
            best["_need_exposure"] = _finite_numeric(best, need_col, "admin-best need exposure")
        if "_raw_exposure" not in best.columns:
            raw_col = _first(
                best,
                ("raw_elderly_exposure", "raw_population_exposure", "reachable_exposure"),
                "admin-best raw exposure",
            )
            best["_raw_exposure"] = _finite_numeric(best, raw_col, "admin-best raw exposure")
    best = best.sort_values(
        ["_admin", "_need_exposure", "_venue_id"], ascending=[True, False, True]
    ).drop_duplicates("_admin", keep="first")
    if best.empty:
        raise ValueError("admin-best venue table has no rank-1 rows")
    return best.reset_index(drop=True)


def _prepare_admin_quadrants(sources: Mapping[str, Any], best: Any) -> Any:
    """Join one policy quadrant and its rank-1 venue to each admin polygon."""

    boundaries = sources["boundaries"]
    if boundaries is None:
        raise KeyError(
            "admin_best_venue_map requires admin_boundaries for a geographic "
            "Need-Exposure quadrant choropleth"
        )
    admin_col = _admin_column(boundaries)
    if boundaries[admin_col].isna().any():
        raise ValueError("admin boundary identifiers must be non-null")
    attributes = boundaries.drop(columns=boundaries.geometry.name).copy()
    attributes["_admin"] = attributes[admin_col].astype(str)

    quadrant_col = next(
        (
            column
            for column in ("need_exposure_quadrant", "quadrant")
            if column in attributes.columns
        ),
        None,
    )
    need_col = next(
        (
            column
            for column in ("structural_need", "need_score", "need")
            if column in attributes.columns
        ),
        None,
    )
    exposure_col = next(
        (
            column
            for column in (
                "need_weighted_exposure",
                "need_exposure",
                "structural_need_exposure",
            )
            if column in attributes.columns
        ),
        None,
    )

    # The production pipeline supplies pre-registered quadrant values on the
    # boundary table.  A deterministic venue-derived fallback keeps the public
    # table mapping compatible for callers that only supply raw venue fields.
    best_values = best[["_admin", "_need_exposure"]].copy()
    best_need_col = next(
        (
            column
            for column in ("structural_need", "need_score", "need")
            if column in best.columns
        ),
        None,
    )
    if best_need_col is not None:
        best_values["_structural_need"] = _finite_numeric(
            best, best_need_col, "admin-best structural need"
        )
    best_values = best_values.drop_duplicates("_admin", keep="first")

    selected = attributes[["_admin"]].drop_duplicates().copy()
    if selected["_admin"].duplicated().any():  # pragma: no cover - defensive
        raise RuntimeError("admin boundary collapse produced duplicate identifiers")

    def unique_admin_values(column: str, canonical: str) -> pd.DataFrame:
        frame = attributes[["_admin", column]].dropna(subset=[column]).copy()
        if frame.groupby("_admin")[column].nunique(dropna=True).gt(1).any():
            raise ValueError(f"admin boundaries contain conflicting {canonical} values")
        return frame.drop_duplicates("_admin", keep="first").rename(
            columns={column: canonical}
        )

    if need_col is not None:
        selected = selected.merge(
            unique_admin_values(need_col, "structural_need"),
            on="_admin",
            how="left",
            validate="one_to_one",
        )
    else:
        selected = selected.merge(
            best_values[["_admin", "_structural_need"]]
            if "_structural_need" in best_values.columns
            else pd.DataFrame(columns=["_admin", "_structural_need"]),
            on="_admin",
            how="left",
            validate="one_to_one",
        ).rename(columns={"_structural_need": "structural_need"})
    if exposure_col is not None:
        selected = selected.merge(
            unique_admin_values(exposure_col, "need_exposure"),
            on="_admin",
            how="left",
            validate="one_to_one",
        )
    else:
        selected = selected.merge(
            best_values[["_admin", "_need_exposure"]].rename(
                columns={"_need_exposure": "need_exposure"}
            ),
            on="_admin",
            how="left",
            validate="one_to_one",
        )
    for column in ("structural_need", "need_exposure"):
        selected[column] = pd.to_numeric(selected[column], errors="coerce")
        if selected[column].isna().any() or not np.isfinite(
            selected[column].to_numpy(dtype=float)
        ).all():
            raise ValueError(
                f"admin boundaries require finite {column} values for every polygon"
            )

    need_cut = float(selected["structural_need"].median())
    exposure_cut = float(selected["need_exposure"].median())
    if quadrant_col is not None:
        selected = selected.merge(
            unique_admin_values(quadrant_col, "quadrant"),
            on="_admin",
            how="left",
            validate="one_to_one",
        )
        selected["quadrant"] = selected["quadrant"].astype(str)
    else:
        high_need = selected["structural_need"].ge(need_cut)
        high_exposure = selected["need_exposure"].ge(exposure_cut)
        selected["quadrant"] = np.select(
            [
                high_need & high_exposure,
                high_need & ~high_exposure,
                ~high_need & high_exposure,
            ],
            [
                "Q1_HIGH_NEED_HIGH_EXPOSURE",
                "Q2_HIGH_NEED_LOW_EXPOSURE",
                "Q3_LOW_NEED_HIGH_EXPOSURE",
            ],
            default="Q4_LOW_NEED_LOW_EXPOSURE",
        )
    unsupported = sorted(set(selected["quadrant"]) - set(_QUADRANT_COLOURS))
    if unsupported:
        raise ValueError(f"unsupported Need-Exposure quadrants: {unsupported}")
    selected["quadrant_label"] = selected["quadrant"].map(_QUADRANT_LABELS)
    selected["need_threshold"] = need_cut
    selected["exposure_threshold"] = exposure_cut

    geometry = boundaries[[admin_col, boundaries.geometry.name]].copy()
    geometry["_admin"] = geometry[admin_col].astype(str)
    geometry = geometry[["_admin", boundaries.geometry.name]].dissolve(
        by="_admin", as_index=False
    )
    admin_geo = geometry.merge(selected, on="_admin", how="left", validate="one_to_one")
    best_overlay = best[
        ["_admin", "_venue_id", "_x_5179", "_y_5179", "_raw_exposure", "_need_exposure"]
    ].rename(
        columns={
            "_venue_id": "venue_id",
            "_x_5179": "venue_x_5179",
            "_y_5179": "venue_y_5179",
            "_raw_exposure": "raw_exposure",
            "_need_exposure": "best_need_exposure",
        }
    )
    return admin_geo.merge(
        best_overlay, on="_admin", how="left", validate="one_to_one"
    )


def _admin_best_figure(sources: Mapping[str, Any], font: str) -> tuple[Any, Any, pd.DataFrame]:
    best = _prepare_admin_best(sources)
    admin_geo = _prepare_admin_quadrants(sources, best)
    normalized = pd.DataFrame(admin_geo.drop(columns=admin_geo.geometry.name)).rename(
        columns={"_admin": "admin"}
    )
    normalized["map_geometry"] = "ADMIN_POLYGON_CHOROPLETH"
    normalized["label_strategy"] = "NO_STATIC_TEXT_LABELS_HTML_HOVER"
    point_rows = admin_geo.loc[admin_geo["venue_id"].notna()].copy()
    sizes = np.full(len(point_rows), 27.0, dtype=float)

    _, plt = _matplotlib()
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    with _plot_context(font):
        static, axis = plt.subplots(figsize=(11.2, 8.2), constrained_layout=True)
        for quadrant, colour in _QUADRANT_COLOURS.items():
            subset = admin_geo.loc[admin_geo["quadrant"].eq(quadrant)]
            if not subset.empty:
                subset.plot(
                    ax=axis,
                    color=colour,
                    edgecolor="white",
                    linewidth=0.55,
                    alpha=0.86,
                    zorder=2,
                )
        admin_geo.boundary.plot(
            ax=axis, color="#475569", linewidth=0.35, alpha=0.72, zorder=3
        )
        axis.scatter(
            point_rows["venue_x_5179"],
            point_rows["venue_y_5179"],
            color="#111827",
            marker="*",
            s=sizes,
            edgecolors="white",
            linewidths=0.35,
            alpha=0.88,
            zorder=4,
        )
        legend_handles = [
            Patch(facecolor=colour, edgecolor="white", label=_QUADRANT_LABELS[quadrant])
            for quadrant, colour in _QUADRANT_COLOURS.items()
        ]
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="*",
                color="none",
                markerfacecolor="#111827",
                markeredgecolor="white",
                markersize=9,
                label="행정동별 Need 노출 1위 후보지",
            )
        )
        axis.legend(
            handles=legend_handles,
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            frameon=False,
            fontsize=8.2,
            title="정책 4분면",
        )
        _map_style(axis, "행정동 Need–Exposure 정책 4분면 · 최상위 후보지")
        _add_north_scale(axis)
        axis.text(
            0,
            -0.105,
            f"행정동 {len(admin_geo):,}개 · 면 색=정책 4분면 · 별=행정동별 Need 노출 1위 (라벨은 HTML hover)",
            transform=axis.transAxes,
            fontsize=9,
            color="#475569",
        )

    go, _ = _plotly()
    interactive = go.Figure()
    legend_seen: set[str] = set()
    for admin, quadrant_value, structural_need, need_exposure, geometry_value in admin_geo[
        ["_admin", "quadrant", "structural_need", "need_exposure", admin_geo.geometry.name]
    ].itertuples(index=False, name=None):
        quadrant = str(quadrant_value)
        hover = (
            f"행정동: {html_module.escape(str(admin))}<br>"
            f"정책 분면: {html_module.escape(_QUADRANT_LABELS[quadrant])}<br>"
            f"구조적 Need: {float(structural_need):,.3f}<br>"
            f"Need 가중 노출: {float(need_exposure):,.1f}"
        )
        for x_coords, y_coords in _geometry_exteriors(geometry_value):
            interactive.add_trace(
                go.Scatter(
                    x=x_coords,
                    y=y_coords,
                    mode="lines",
                    fill="toself",
                    fillcolor=_QUADRANT_COLOURS[quadrant],
                    line={"color": "white", "width": 0.55},
                    opacity=0.86,
                    text=[hover] * len(x_coords),
                    hovertemplate="%{text}<extra></extra>",
                    name=_QUADRANT_LABELS[quadrant],
                    legendgroup=quadrant,
                    showlegend=quadrant not in legend_seen,
                )
            )
            legend_seen.add(quadrant)
    interactive.add_trace(
        go.Scattergl(
            x=point_rows["venue_x_5179"],
            y=point_rows["venue_y_5179"],
            mode="markers",
            marker={
                "size": np.clip(sizes / 5, 7, 22),
                "color": "#111827",
                "symbol": "star",
                "opacity": 0.86,
                "line": {"color": "white", "width": 0.5},
            },
            customdata=np.column_stack(
                [
                    point_rows["_admin"],
                    point_rows["venue_id"],
                    point_rows["best_need_exposure"],
                    point_rows["raw_exposure"],
                    point_rows["quadrant_label"],
                ]
            ),
            hovertemplate=(
                "행정동: %{customdata[0]}<br>최상위 후보지: %{customdata[1]}<br>"
                "Need 노출: %{customdata[2]:,.1f}<br>원시 노출: %{customdata[3]:,.1f}<br>"
                "%{customdata[4]}<extra></extra>"
            ),
            name="행정동별 Need 노출 1위",
        )
    )
    _plotly_layout(interactive, "행정동 Need–Exposure 정책 4분면 · 최상위 후보지", font)
    interactive.update_yaxes(scaleanchor="x", scaleratio=1, title="EPSG:5179 Y (m)")
    interactive.update_xaxes(title="EPSG:5179 X (m)")
    return static, interactive, normalized


def _prepare_bundle(sources: Mapping[str, Any]) -> pd.DataFrame:
    venues = sources["venues"]
    raw = sources["bundle"]
    if raw is None:
        raw = venues
    venue_col = _venue_id_column(raw)
    value_col = next(
        (
            column
            for column in (
                "bundle_gap_weighted_exposure",
                "bundle_gap_coverage",
                "gap_weighted_exposure",
                "weighted_bundle_gap_exposure",
                "bundle_gap_exposure",
            )
            if column in raw.columns
        ),
        None,
    )
    bundle_col = next(
        (column for column in ("bundle_id", "service_bundle_id", "dominant_bundle_id") if column in raw.columns),
        None,
    )
    long: pd.DataFrame
    if value_col is not None:
        if bundle_col is None:
            raise KeyError("bundle-gap exposure requires bundle_id/service_bundle_id")
        long = pd.DataFrame(
            {
                "venue_id": raw[venue_col].astype(str),
                "bundle_id": raw[bundle_col].fillna("미상").astype(str),
                "bundle_gap_exposure": _finite_numeric(raw, value_col, "bundle-gap exposure"),
            }
        )
    else:
        wide_columns = [
            column
            for column in raw.columns
            if "bundle" in column.lower()
            and ("gap" in column.lower() or "exposure" in column.lower())
            and pd.api.types.is_numeric_dtype(raw[column])
        ]
        if not wide_columns:
            raise KeyError(
                "bundle_gap_weighted_venue_map requires a long bundle-gap value or numeric bundle-gap columns"
            )
        long = raw[[venue_col, *wide_columns]].melt(
            id_vars=venue_col, var_name="bundle_id", value_name="bundle_gap_exposure"
        )
        long = long.rename(columns={venue_col: "venue_id"})
        long["venue_id"] = long["venue_id"].astype(str)
        long["bundle_id"] = long["bundle_id"].astype(str)
        long["bundle_gap_exposure"] = pd.to_numeric(long["bundle_gap_exposure"], errors="coerce")
        if long["bundle_gap_exposure"].isna().any():
            raise ValueError("wide bundle-gap values must be finite numeric")
    if (long["bundle_gap_exposure"] < 0).any() or not np.isfinite(
        long["bundle_gap_exposure"].to_numpy(dtype=float)
    ).all():
        raise ValueError("bundle-gap exposure must be finite and non-negative")
    if long.duplicated(["venue_id", "bundle_id"]).any():
        long = (
            long.groupby(["venue_id", "bundle_id"], as_index=False, sort=True)["bundle_gap_exposure"]
            .sum()
        )
    dominant = long.sort_values(
        ["venue_id", "bundle_gap_exposure", "bundle_id"], ascending=[True, False, True]
    ).drop_duplicates("venue_id")
    summary = long.groupby("venue_id", as_index=False).agg(
        total_bundle_gap_exposure=("bundle_gap_exposure", "sum"),
        active_bundle_count=("bundle_id", lambda values: int(pd.Series(values).nunique())),
    )
    summary = summary.merge(
        dominant[["venue_id", "bundle_id", "bundle_gap_exposure"]].rename(
            columns={
                "bundle_id": "dominant_bundle_id",
                "bundle_gap_exposure": "dominant_bundle_gap_exposure",
            }
        ),
        on="venue_id",
        how="left",
        validate="one_to_one",
    )
    coordinates = venues.set_index("_venue_id")[["_x_5179", "_y_5179"]]
    missing = sorted(set(summary["venue_id"]) - set(coordinates.index))
    if missing:
        raise ValueError(f"bundle-gap venues missing from candidate table: {missing[:5]}")
    return summary.join(coordinates, on="venue_id").reset_index(drop=True)


def _bundle_figure(sources: Mapping[str, Any], font: str) -> tuple[Any, Any, pd.DataFrame]:
    bundle = _prepare_bundle(sources)
    boundary = sources["boundaries"]
    counts = bundle["dominant_bundle_id"].value_counts()
    top = counts.head(6).index.tolist()
    bundle["_display_bundle"] = bundle["dominant_bundle_id"].where(
        bundle["dominant_bundle_id"].isin(top), "기타"
    )
    order = [value for value in top if value in set(bundle["_display_bundle"])]
    if bundle["_display_bundle"].eq("기타").any():
        order.append("기타")
    palette = dict(zip(order, _BUNDLE_COLOURS * 2))
    maximum = max(float(bundle["total_bundle_gap_exposure"].max()), 1.0)
    sizes = 14 + 68 * np.sqrt(bundle["total_bundle_gap_exposure"] / maximum)
    normalized = bundle[
        [
            "venue_id",
            "_x_5179",
            "_y_5179",
            "dominant_bundle_id",
            "dominant_bundle_gap_exposure",
            "total_bundle_gap_exposure",
            "active_bundle_count",
        ]
    ].rename(columns={"_x_5179": "x_5179", "_y_5179": "y_5179"})

    _, plt = _matplotlib()
    with _plot_context(font):
        static, axis = plt.subplots(figsize=(11.2, 8.2), constrained_layout=True)
        _boundary_static(axis, boundary)
        for category in order:
            mask = bundle["_display_bundle"].eq(category)
            axis.scatter(
                bundle.loc[mask, "_x_5179"],
                bundle.loc[mask, "_y_5179"],
                s=sizes[mask],
                color=palette[category],
                alpha=0.76,
                edgecolors="white",
                linewidths=0.32,
                label=f"{category} (n={int(mask.sum()):,})",
                zorder=3,
            )
        _map_style(axis, "진료 bundle gap 가중 도달량 · 후보지별 지배 bundle")
        _add_north_scale(axis)
        axis.legend(
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            frameon=False,
            fontsize=8.5,
            title="지배 bundle",
        )
        axis.text(
            0,
            -0.105,
            "색=가장 큰 gap 가중 bundle · 점 크기=5개 bundle gap 가중 도달량 합계",
            transform=axis.transAxes,
            fontsize=9,
            color="#475569",
        )

    go, _ = _plotly()
    interactive = go.Figure()
    _boundary_interactive(interactive, boundary)
    for category in order:
        mask = bundle["_display_bundle"].eq(category)
        subset = bundle.loc[mask]
        interactive.add_trace(
            go.Scattergl(
                x=subset["_x_5179"],
                y=subset["_y_5179"],
                mode="markers",
                marker={
                    "size": np.clip(sizes[mask] / 4, 6, 21),
                    "color": palette[category],
                    "opacity": 0.78,
                    "line": {"color": "white", "width": 0.4},
                },
                customdata=np.column_stack(
                    [
                        subset["venue_id"],
                        subset["dominant_bundle_id"],
                        subset["dominant_bundle_gap_exposure"],
                        subset["total_bundle_gap_exposure"],
                    ]
                ),
                hovertemplate=(
                    "후보지: %{customdata[0]}<br>지배 bundle: %{customdata[1]}<br>"
                    "지배 bundle gap 도달: %{customdata[2]:,.1f}<br>"
                    "전체 bundle gap 도달: %{customdata[3]:,.1f}<extra></extra>"
                ),
                name=f"{category} (n={int(mask.sum()):,})",
            )
        )
    _plotly_layout(interactive, "진료 bundle gap 가중 도달량 · 후보지별 지배 bundle", font)
    interactive.update_yaxes(scaleanchor="x", scaleratio=1, title="EPSG:5179 Y (m)")
    interactive.update_xaxes(title="EPSG:5179 X (m)")
    return static, interactive, normalized


def _prepare_rural_urban(sources: Mapping[str, Any]) -> pd.DataFrame:
    venues = sources["venues"]
    raw = sources["rural_urban"] if sources["rural_urban"] is not None else venues
    venue_col = _venue_id_column(raw)
    rural_col = next(
        (
            column
            for column in (
                "rural_exposure",
                "rural_elderly_exposure",
                "rural_raw_exposure",
                "reachable_rural_exposure",
            )
            if column in raw.columns
        ),
        None,
    )
    urban_col = next(
        (
            column
            for column in (
                "urban_exposure",
                "urban_elderly_exposure",
                "urban_raw_exposure",
                "reachable_urban_exposure",
            )
            if column in raw.columns
        ),
        None,
    )
    if (rural_col is None) != (urban_col is None):
        raise KeyError("rural and urban exposure columns must be supplied together")
    if rural_col is None:
        # The canonical pipeline keeps this split in the sealed grid rather
        # than denormalizing it into every candidate row.  Reproduce the
        # preregistered primary hard-5km partition exactly for this figure and
        # reconcile it against the already-computed raw exposure.
        grid = sources["grid"]
        rural_flag_col = _first(
            grid,
            ("is_rural_eup_myeon", "is_rural", "rural_flag"),
            "grid rural/urban classification",
        )
        flag = grid[rural_flag_col]
        if flag.isna().any():
            raise ValueError("grid rural flag must not contain missing values")
        if pd.api.types.is_bool_dtype(flag):
            rural_flag = flag.to_numpy(dtype=bool)
        else:
            normalized_flag = flag.astype("string").str.strip().str.upper()
            valid = normalized_flag.isin({"TRUE", "FALSE", "1", "0", "Y", "N"})
            if normalized_flag.isna().any() or not valid.all():
                raise ValueError("grid rural flag must be complete boolean-like data")
            rural_flag = normalized_flag.isin({"TRUE", "1", "Y"}).to_numpy(dtype=bool)
        try:
            from scipy.spatial import cKDTree
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ImportError("rural/urban Stage 3 exposure requires scipy") from exc
        coordinates = grid[["_x_5179", "_y_5179"]].to_numpy(dtype=float)
        venue_coordinates = venues[["_x_5179", "_y_5179"]].to_numpy(dtype=float)
        population = grid["_elderly_population"].to_numpy(dtype=float)
        tree = cKDTree(coordinates)
        rural_exposure = np.zeros(len(venues), dtype=float)
        urban_exposure = np.zeros(len(venues), dtype=float)
        for start in range(0, len(venues), 256):
            stop = min(start + 256, len(venues))
            neighbours = tree.query_ball_point(venue_coordinates[start:stop], r=5000.0)
            for offset, indices in enumerate(neighbours):
                selected = np.asarray(indices, dtype=np.int64)
                if selected.size:
                    rural_exposure[start + offset] = population[
                        selected[rural_flag[selected]]
                    ].sum()
                    urban_exposure[start + offset] = population[
                        selected[~rural_flag[selected]]
                    ].sum()
        computed_total = rural_exposure + urban_exposure
        expected_total = venues["_raw_exposure"].to_numpy(dtype=float)
        if not np.allclose(computed_total, expected_total, rtol=2e-6, atol=1e-6):
            maximum_error = float(np.max(np.abs(computed_total - expected_total)))
            raise ValueError(
                "reporting hard-5km rural/urban partition does not reconcile with "
                f"candidate raw exposure (max_abs_error={maximum_error:.6g})"
            )
        frame = pd.DataFrame(
            {
                "venue_id": venues["_venue_id"],
                "rural_exposure": rural_exposure,
                "urban_exposure": urban_exposure,
            }
        )
    else:
        frame = pd.DataFrame(
            {
                "venue_id": raw[venue_col].astype(str),
                "rural_exposure": _finite_numeric(raw, rural_col, "rural exposure"),
                "urban_exposure": _finite_numeric(raw, urban_col, "urban exposure"),
            }
        )
    if frame["venue_id"].duplicated().any():
        frame = frame.groupby("venue_id", as_index=False).agg(
            rural_exposure=("rural_exposure", "sum"),
            urban_exposure=("urban_exposure", "sum"),
        )
    if (frame[["rural_exposure", "urban_exposure"]] < 0).any().any():
        raise ValueError("rural/urban exposure must be non-negative")
    total = frame["rural_exposure"] + frame["urban_exposure"]
    frame["rural_share"] = np.divide(
        frame["rural_exposure"],
        total,
        out=np.zeros(len(frame), dtype=float),
        where=total.to_numpy(dtype=float) > 0,
    )
    return frame


def _rural_urban_figure(sources: Mapping[str, Any], font: str) -> tuple[Any, Any, pd.DataFrame]:
    frame = _prepare_rural_urban(sources)
    rural_log = np.log1p(frame["rural_exposure"])
    urban_log = np.log1p(frame["urban_exposure"])
    difference = rural_log - urban_log

    _, plt = _matplotlib()
    with _plot_context(font):
        static, axes = plt.subplots(1, 2, figsize=(12.2, 6.7), constrained_layout=True)
        scatter = axes[0].scatter(
            urban_log,
            rural_log,
            c=frame["rural_share"],
            cmap="coolwarm",
            vmin=0,
            vmax=1,
            s=16,
            alpha=0.65,
            linewidths=0,
            rasterized=True,
        )
        lower = min(float(rural_log.min()), float(urban_log.min()))
        upper = max(float(rural_log.max()), float(urban_log.max()))
        axes[0].plot([lower, upper], [lower, upper], color="#334155", ls="--", lw=1)
        axes[0].set_title("도시–농촌 노출 균형", loc="left", fontweight="bold")
        axes[0].set_xlabel("log(1 + 도시 고령인구 노출)")
        axes[0].set_ylabel("log(1 + 농촌 고령인구 노출)")
        static.colorbar(scatter, ax=axes[0], fraction=0.045, pad=0.025, label="농촌 노출 비중")
        axes[1].hist(difference, bins=32, color="#7C3AED", alpha=0.78)
        axes[1].axvline(0, color="#334155", ls="--", lw=1.2)
        axes[1].axvline(float(np.median(difference)), color="#C2410C", lw=1.4)
        axes[1].set_title("농촌 우세 ↔ 도시 우세 분리", loc="left", fontweight="bold")
        axes[1].set_xlabel("log 농촌 노출 - log 도시 노출")
        axes[1].set_ylabel("후보지 수")
        # Some Malgun Gothic builds omit U+2212, which Matplotlib may use for
        # negative numeric ticks even with Korean labels rendered correctly.
        for tick_label in axes[1].get_xticklabels():
            tick_label.set_fontfamily("DejaVu Sans")
        static.suptitle("후보지 농촌·도시 노출 이질성", fontsize=16, fontweight="bold", x=0.01, ha="left")

    go, make_subplots = _plotly()
    interactive = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("도시–농촌 노출 균형", "농촌 우세 ↔ 도시 우세"),
    )
    interactive.add_trace(
        go.Scattergl(
            x=urban_log,
            y=rural_log,
            mode="markers",
            marker={
                "size": 7,
                "color": frame["rural_share"],
                "colorscale": "RdBu",
                "cmin": 0,
                "cmax": 1,
                "colorbar": {"title": "농촌 비중", "x": 0.45},
                "opacity": 0.68,
            },
            customdata=np.column_stack(
                [frame["venue_id"], frame["rural_exposure"], frame["urban_exposure"], frame["rural_share"]]
            ),
            hovertemplate=(
                "후보지: %{customdata[0]}<br>농촌 노출: %{customdata[1]:,.1f}<br>"
                "도시 노출: %{customdata[2]:,.1f}<br>농촌 비중: %{customdata[3]:.1%}<extra></extra>"
            ),
            name="후보지",
        ),
        row=1,
        col=1,
    )
    interactive.add_trace(
        go.Histogram(x=difference, nbinsx=32, marker_color="#7C3AED", name="노출 차이"),
        row=1,
        col=2,
    )
    _plotly_layout(interactive, "후보지 농촌·도시 노출 이질성", font)
    interactive.update_xaxes(title="log(1 + 도시 노출)", row=1, col=1)
    interactive.update_yaxes(title="log(1 + 농촌 노출)", row=1, col=1)
    interactive.update_xaxes(title="log 농촌 − log 도시", row=1, col=2)
    interactive.update_yaxes(title="후보지 수", row=1, col=2)
    return static, interactive, frame


_FIGURE_BUILDERS: Mapping[str, Any] = {
    "elderly_grid_distribution": _elderly_grid_figure,
    "candidate_venue_map": _candidate_venue_figure,
    "venue_exposure_distribution": _venue_exposure_figure,
    "cross_admin_coverage_map": _cross_admin_figure,
    "overlap_redundancy_map": _overlap_figure,
    "admin_best_venue_map": _admin_best_figure,
    "bundle_gap_weighted_venue_map": _bundle_figure,
    "rural_urban_exposure": _rural_urban_figure,
}


def _manifest_rows(
    figures: Mapping[str, Mapping[str, str]],
    normalized: Mapping[str, pd.DataFrame],
    output_dir: Path,
    *,
    font: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for figure_id in STAGE3_SPATIAL_MAP_IDS:
        digest = _frame_sha256(normalized[figure_id])
        row_count = len(normalized[figure_id])
        for artifact_format in ("png", "svg", "html"):
            path = Path(figures[figure_id][artifact_format])
            width = height = dpi_x = dpi_y = None
            external_scripts = 0
            self_contained: bool | None = None
            if artifact_format == "png":
                width, height, dpi_x, dpi_y = _png_metadata(path)
            elif artifact_format == "html":
                content = path.read_text(encoding="utf-8")
                external_scripts = len(
                    re.findall(r"<script\b[^>]*\bsrc\s*=", content, flags=re.I)
                )
                external_styles = len(
                    re.findall(
                        r"<link\b[^>]*\b(?:href|src)\s*=\s*['\"](?:https?:)?//",
                        content,
                        flags=re.I,
                    )
                )
                self_contained = (
                    "plotly" in content.lower()
                    and external_scripts == 0
                    and external_styles == 0
                )
            rows.append(
                {
                    "figure_id": figure_id,
                    "format": artifact_format,
                    "relative_path": path.relative_to(output_dir).as_posix(),
                    "size_bytes": int(path.stat().st_size),
                    "sha256": sha256_file(path),
                    "input_table_sha256": digest,
                    "input_row_count": int(row_count),
                    "analysis_crs": "EPSG:5179",
                    "font_family": font,
                    "width_px": width,
                    "height_px": height,
                    "dpi_x": dpi_x,
                    "dpi_y": dpi_y,
                    "external_script_count": int(external_scripts),
                    "self_contained_html": self_contained,
                }
            )
    return pd.DataFrame(rows)


def validate_stage3_spatial_map_manifest(
    manifest: pd.DataFrame,
    output_dir: str | Path | None = None,
) -> None:
    """Fail closed on missing triplets, low resolution, or non-offline HTML.

    If ``output_dir`` is supplied, every relative path, byte count, and SHA256
    is also reconciled against the filesystem.
    """

    if not isinstance(manifest, pd.DataFrame) or manifest.empty:
        raise ValueError("Stage 3 spatial-map manifest must be non-empty")
    required = {
        "figure_id",
        "format",
        "relative_path",
        "size_bytes",
        "sha256",
        "input_table_sha256",
        "input_row_count",
        "analysis_crs",
        "font_family",
        "width_px",
        "height_px",
        "dpi_x",
        "dpi_y",
        "external_script_count",
        "self_contained_html",
    }
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise KeyError(f"Stage 3 spatial-map manifest missing columns: {missing}")
    expected = {
        (figure_id, artifact_format)
        for figure_id in STAGE3_SPATIAL_MAP_IDS
        for artifact_format in ("png", "svg", "html")
    }
    observed = set(zip(manifest["figure_id"], manifest["format"]))
    if observed != expected or len(manifest) != len(expected):
        raise ValueError("Stage 3 spatial-map manifest must contain exactly 8x3 artifacts")
    if not manifest["relative_path"].astype(str).str.fullmatch(
        r"[A-Za-z0-9_.\-/]+"
    ).all():
        raise ValueError("Stage 3 spatial-map manifest contains unsafe relative paths")
    if manifest["relative_path"].astype(str).str.contains(r"(?:^|/)\.\.(?:/|$)", regex=True).any():
        raise ValueError("Stage 3 spatial-map paths must not traverse parents")
    if (pd.to_numeric(manifest["size_bytes"], errors="coerce") <= 100).any():
        raise ValueError("Stage 3 spatial-map artifacts must be non-empty")
    if (pd.to_numeric(manifest["input_row_count"], errors="coerce") <= 0).any():
        raise ValueError("Stage 3 spatial maps require non-empty normalized inputs")
    for column in ("sha256", "input_table_sha256"):
        if not manifest[column].astype(str).str.fullmatch(r"[0-9a-f]{64}").all():
            raise ValueError(f"Stage 3 spatial-map {column} is invalid")
    if not manifest["analysis_crs"].eq("EPSG:5179").all():
        raise ValueError("Stage 3 spatial maps must use EPSG:5179 analysis coordinates")
    if manifest["font_family"].astype(str).str.strip().eq("").any():
        raise ValueError("Stage 3 spatial maps require a verified font family")
    png = manifest.loc[manifest["format"].eq("png")]
    if (pd.to_numeric(png["dpi_x"], errors="coerce") < 299).any() or (
        pd.to_numeric(png["dpi_y"], errors="coerce") < 299
    ).any():
        raise ValueError("Stage 3 spatial-map PNG files require at least 300-dpi metadata")
    if (pd.to_numeric(png["width_px"], errors="coerce") < 1800).any() or (
        pd.to_numeric(png["height_px"], errors="coerce") < 1000
    ).any():
        raise ValueError("Stage 3 spatial-map PNG files are below the publication floor")
    html_rows = manifest.loc[manifest["format"].eq("html")]
    if not html_rows["external_script_count"].eq(0).all() or not html_rows[
        "self_contained_html"
    ].eq(True).all():
        raise ValueError("Stage 3 spatial-map HTML files must be self-contained")

    if output_dir is not None:
        root = Path(output_dir).resolve()
        for row in manifest.itertuples(index=False):
            path = (root / str(row.relative_path)).resolve()
            try:
                path.relative_to(root)
            except ValueError as exc:
                raise ValueError(f"artifact path escapes output directory: {path}") from exc
            if not path.is_file():
                raise FileNotFoundError(f"Stage 3 spatial-map artifact missing: {path}")
            if path.stat().st_size != int(row.size_bytes):
                raise ValueError(f"Stage 3 spatial-map size mismatch: {path}")
            if sha256_file(path) != str(row.sha256):
                raise ValueError(f"Stage 3 spatial-map SHA256 mismatch: {path}")


def render_stage3_spatial_maps(
    tables: Mapping[str, pd.DataFrame],
    output_dir: str | Path,
    dpi: int = 300,
) -> Stage3SpatialMapResult:
    """Render and validate all eight Stage 3 spatial/distribution figures.

    Required logical inputs are a population grid, candidate venues, admin
    boundaries, venue overlap edges, bundle-gap exposure, and rural/urban
    exposure.  Admin polygons must carry Need/quadrant metrics or the venue
    table must carry enough fields to derive them.  An explicit admin-best
    table remains optional; otherwise rank-1 rows are deterministically
    derived from the venue table.  See :data:`_TABLE_ALIASES` for accepted
    mapping keys.
    """

    if not isinstance(tables, Mapping):
        raise TypeError("tables must be a mapping of DataFrames")
    if int(dpi) < 300:
        raise ValueError("Stage 3 publication spatial maps require dpi >= 300")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    font = _require_korean_font()
    sources = _normalize_sources(tables)
    figures: dict[str, Mapping[str, str]] = {}
    normalized: dict[str, pd.DataFrame] = {}
    for figure_id in STAGE3_SPATIAL_MAP_IDS:
        static, interactive, figure_table = _FIGURE_BUILDERS[figure_id](sources, font)
        if not isinstance(figure_table, pd.DataFrame) or figure_table.empty:
            raise ValueError(f"{figure_id} normalized source must be non-empty")
        normalized[figure_id] = pd.DataFrame(figure_table).reset_index(drop=True)
        figures[figure_id] = _save_figure_triplet(
            figure_id,
            static,
            interactive,
            destination,
            dpi=int(dpi),
        )
    manifest = _manifest_rows(figures, normalized, destination, font=font)
    validate_stage3_spatial_map_manifest(manifest, destination)
    manifest_path = destination / "STAGE3_SPATIAL_MAP_ARTIFACTS.csv"
    _atomic_write_csv(manifest, manifest_path)
    return Stage3SpatialMapResult(
        figures=figures,
        artifact_manifest=manifest,
        artifact_manifest_path=str(manifest_path),
        normalized_tables=normalized,
    )


__all__ = [
    "STAGE3_SPATIAL_MAP_IDS",
    "Stage3SpatialMapResult",
    "render_stage3_spatial_maps",
    "validate_stage3_spatial_map_manifest",
]
