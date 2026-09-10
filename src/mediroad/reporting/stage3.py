"""Publication figures and quality-gate helpers for MEDIROAD Stage 3.

This module contains reusable rendering functions only.  It does not select
an official run or update aliases.  Every rendered core figure is emitted as
a 300-dpi PNG, an SVG, and a Plotly HTML document with JavaScript embedded.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
import hashlib
import re

import numpy as np
import pandas as pd

from mediroad.reporting.correlation import hierarchical_order
from mediroad.reporting.stage2b_hardening import (
    _font_family,
    _frame_sha256,
    _png_metadata,
    _save_figure_triplet,
    sha256_file,
)


STAGE3_CORE_FIGURE_IDS: tuple[str, ...] = (
    "need_exposure_quadrant",
    "correlation_dependency_audit",
    "sensitivity_stability",
    "pareto_shortlist_summary",
)

QUALITY_GATE_COLUMNS: tuple[str, ...] = (
    "gate_id",
    "severity",
    "passed",
    "observed",
    "comparison",
    "threshold",
    "threshold_source",
    "note",
)

_QUADRANT_COLOURS: Mapping[str, str] = {
    "Q1_HIGH_NEED_HIGH_EXPOSURE": "#0F766E",
    "Q2_HIGH_NEED_LOW_EXPOSURE": "#DC2626",
    "Q3_LOW_NEED_HIGH_EXPOSURE": "#2563EB",
    "Q4_LOW_NEED_LOW_EXPOSURE": "#9CA3AF",
}
_TIER_COLOURS: tuple[str, ...] = (
    "#0F766E",
    "#2563EB",
    "#7C3AED",
    "#D97706",
    "#DC2626",
    "#6B7280",
)

_CORRELATION_DISPLAY_LABELS: Mapping[str, str] = {
    "raw_elderly_exposure": "Elderly exposure",
    "need_weighted_exposure": "Need-weighted exposure",
    "high_need_elderly_exposure": "High-Need exposure",
    "mean_bundle_gap_weighted_exposure": "Bundle-gap exposure",
    "catchment_mean_need_intensity": "Mean Need intensity",
    "high_need_population_share": "High-Need share",
    "catchment_mean_bundle_gap_intensity": "Mean bundle-gap intensity",
    "venue_readiness_prior": "Readiness prior",
    "transit_context_score": "Transit context",
    "structural_need": "Structural Need",
    "cross_admin_share": "Cross-admin share",
    "overlap_weighted_degree": "Overlap burden",
    "overlap_max_jaccard": "Max catchment overlap",
    "osm_nearest_mobile_team_base_drive_min_v6": "Nearest-team drive time",
}

_SENSITIVITY_METRIC_LABELS: Mapping[str, str] = {
    "venue_rank_spearman": "Venue rank ρ",
    "admin_top3_jaccard_mean": "Top-3 Jaccard",
    "admin_top5_jaccard_mean": "Top-5 Jaccard",
    "pareto_retention": "Pareto retention",
    "region_best_venue_stability": "Best-venue stability",
}

_SENSITIVITY_SCENARIO_LABELS: Mapping[str, str] = {
    "FULL_IDENTITY": "Reference",
    "ablation_no_stage1_need_weighting": "No Stage 1 Need",
    "ablation_own_admin_only": "Own admin only",
    "ablation_published_positive_only_suppressed_missing": "Published cells only",
    "ablation_without_bundle_gap_weighting": "No bundle-gap weight",
    "ablation_without_readiness_prior": "No readiness prior",
    "ablation_without_transit_context": "No transit context",
}

_SENSITIVITY_SCOPE_ORDER: tuple[str, ...] = (
    "Catchment definition",
    "Model ablations / checks",
    "Objective leave-one-out",
)


@dataclass(frozen=True)
class Stage3FigureResult:
    """Core figure triplets, their manifest, and validated source tables."""

    figures: Mapping[str, Mapping[str, str]]
    artifact_manifest: pd.DataFrame
    normalized_tables: Mapping[str, pd.DataFrame]


@dataclass(frozen=True)
class Stage3QualityGateResult:
    """Fail-closed Stage 3 quality-gate ledger."""

    gates: pd.DataFrame
    hard_pass: bool
    hard_total: int
    hard_failed: int
    advisory_failed: int
    diagnostic_failed: int


def _require_frame(table: pd.DataFrame, label: str) -> pd.DataFrame:
    if not isinstance(table, pd.DataFrame) or table.empty:
        raise ValueError(f"{label} must be a non-empty DataFrame")
    if not table.columns.is_unique:
        raise ValueError(f"{label} columns must be unique")
    return table.copy()


def _first_column(frame: pd.DataFrame, options: Sequence[str], label: str) -> str:
    match = next((column for column in options if column in frame.columns), None)
    if match is None:
        raise KeyError(f"{label} requires one of {list(options)}")
    return match


def _finite(frame: pd.DataFrame, columns: Sequence[str], label: str) -> pd.DataFrame:
    numeric = frame.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError(f"{label} values must be finite numeric")
    return numeric


def _matplotlib() -> tuple[Any, Any]:
    try:
        import matplotlib as mpl

        mpl.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Stage 3 figures require matplotlib") from exc
    return mpl, plt


def _plotly() -> tuple[Any, Any]:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Stage 3 figures require plotly") from exc
    return go, make_subplots


def _plot_context() -> Any:
    mpl, _ = _matplotlib()
    return mpl.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [_font_family(), "DejaVu Sans"],
            "axes.unicode_minus": False,
            "svg.fonttype": "path",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _quadrant_figures(frame: pd.DataFrame) -> tuple[Any, Any, pd.DataFrame]:
    data = _require_frame(frame, "Need-Exposure quadrant table")
    need_col = _first_column(data, ("structural_need", "need", "need_score"), "quadrant")
    exposure_col = _first_column(
        data,
        ("need_weighted_exposure", "reachable_exposure", "exposure"),
        "quadrant",
    )
    quadrant_col = _first_column(
        data, ("need_exposure_quadrant", "quadrant"), "quadrant"
    )
    admin_col = next(
        (
            column
            for column in ("admin_dong_code", "admin_dong", "admin_name")
            if column in data.columns
        ),
        None,
    )
    numeric = _finite(data, [need_col, exposure_col], "Need-Exposure quadrant")
    data[need_col] = numeric[need_col]
    data[exposure_col] = numeric[exposure_col]
    data[quadrant_col] = data[quadrant_col].astype(str)
    unknown = sorted(set(data[quadrant_col]) - set(_QUADRANT_COLOURS))
    if unknown:
        raise ValueError(f"unsupported Need-Exposure quadrants: {unknown}")
    need_cut = (
        float(pd.to_numeric(data["need_threshold"], errors="coerce").iloc[0])
        if "need_threshold" in data.columns
        else float(data[need_col].median())
    )
    exposure_cut = (
        float(pd.to_numeric(data["exposure_threshold"], errors="coerce").iloc[0])
        if "exposure_threshold" in data.columns
        else float(data[exposure_col].median())
    )

    _, plt = _matplotlib()
    with _plot_context():
        static, axis = plt.subplots(figsize=(10.5, 7.2), constrained_layout=True)
        for quadrant, colour in _QUADRANT_COLOURS.items():
            subset = data.loc[data[quadrant_col].eq(quadrant)]
            if subset.empty:
                continue
            axis.scatter(
                subset[need_col],
                subset[exposure_col],
                s=42,
                alpha=0.78,
                edgecolor="white",
                linewidth=0.5,
                color=colour,
                label=f"{quadrant.split('_')[0]} (n={len(subset)})",
            )
        axis.axvline(need_cut, color="#374151", linewidth=1.1, linestyle="--")
        axis.axhline(exposure_cut, color="#374151", linewidth=1.1, linestyle="--")
        axis.set_xlabel("Structural need")
        axis.set_ylabel("Reachable need-weighted exposure")
        axis.set_title("Need–Exposure policy quadrants (admin-collapsed)", fontweight="bold")
        axis.grid(alpha=0.16)
        axis.legend(frameon=False, loc="best")
        axis.text(
            0.01,
            0.01,
            "Q2 is a dispersed/high-need diagnostic, not a service-failure label.",
            transform=axis.transAxes,
            fontsize=8,
            color="#4B5563",
        )

    go, _ = _plotly()
    hover = data[admin_col].astype(str) if admin_col else pd.Series(data.index.astype(str))
    interactive = go.Figure()
    for quadrant, colour in _QUADRANT_COLOURS.items():
        subset = data.loc[data[quadrant_col].eq(quadrant)]
        if subset.empty:
            continue
        interactive.add_trace(
            go.Scatter(
                x=subset[need_col],
                y=subset[exposure_col],
                mode="markers",
                name=quadrant,
                text=hover.loc[subset.index],
                hovertemplate=(
                    "%{text}<br>Need=%{x:.4g}<br>Exposure=%{y:.4g}<extra></extra>"
                ),
                marker={"color": colour, "size": 9, "opacity": 0.8},
            )
        )
    interactive.add_vline(x=need_cut, line_dash="dash", line_color="#374151")
    interactive.add_hline(y=exposure_cut, line_dash="dash", line_color="#374151")
    interactive.update_layout(
        title="Need–Exposure policy quadrants (admin-collapsed)",
        xaxis_title="Structural need",
        yaxis_title="Reachable need-weighted exposure",
        template="plotly_white",
        font={"family": "Malgun Gothic, NanumGothic, sans-serif"},
    )
    return static, interactive, data


def _validate_square_correlation(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    corr = _require_frame(frame, label)
    corr.columns = corr.columns.map(str)
    corr.index = corr.index.map(str)
    if corr.shape[0] != corr.shape[1] or list(corr.index) != list(corr.columns):
        raise ValueError(f"{label} must be a labelled square matrix")
    numeric = corr.apply(pd.to_numeric, errors="coerce")
    numeric.attrs.update(corr.attrs)
    finite = numeric.to_numpy(dtype=float)
    if np.nanmin(finite) < -1.000001 or np.nanmax(finite) > 1.000001:
        raise ValueError(f"{label} contains values outside [-1, 1]")
    if not np.allclose(finite, finite.T, equal_nan=True, atol=1e-9):
        raise ValueError(f"{label} must be symmetric")
    return numeric


def _unique_display_labels(
    canonical_labels: Sequence[str],
    aliases: Mapping[str, str],
) -> list[str]:
    """Return concise, deterministic labels without losing one-to-one identity."""

    displayed: list[str] = []
    used: dict[str, int] = {}
    for canonical in canonical_labels:
        canonical = str(canonical)
        label = aliases.get(canonical, canonical.replace("_", " ").strip().title())
        used[label] = used.get(label, 0) + 1
        displayed.append(label if used[label] == 1 else f"{label} ({used[label]})")
    return displayed


def _correlation_sample_n(
    frame: pd.DataFrame,
) -> int:
    for key in ("sample_n", "pairwise_n", "n"):
        if key not in frame.attrs:
            continue
        value = pd.to_numeric(pd.Series([frame.attrs[key]]), errors="coerce").iloc[0]
        if pd.isna(value) or float(value) <= 0 or not float(value).is_integer():
            raise ValueError(f"correlation DataFrame attr {key} must be a positive integer")
        return int(value)
    raise ValueError(
        "correlation matrices require a positive sample_n attr from "
        "stage3_spearman_audit; sample size cannot be inferred from rho values"
    )


def _sample_n_suffix(sample_n: int) -> str:
    return f"n={sample_n:,}"


def _sensitivity_scope(scenario: str, declared_scope: Any = None) -> str:
    scenario = str(scenario)
    if declared_scope is not None and not pd.isna(declared_scope):
        declared = str(declared_scope).strip().lower()
        if "catchment" in declared:
            return "Catchment definition"
        if "leave_one_out" in declared or "objective_loo" in declared:
            return "Objective leave-one-out"
        return "Model ablations / checks"
    if re.fullmatch(r"(?:hard|exp|gaussian)_\d+m", scenario):
        return "Catchment definition"
    if scenario.startswith("objective_loo__"):
        return "Objective leave-one-out"
    return "Model ablations / checks"


def _sensitivity_scenario_label(scenario: str) -> str:
    scenario = str(scenario)
    if scenario in _SENSITIVITY_SCENARIO_LABELS:
        return _SENSITIVITY_SCENARIO_LABELS[scenario]
    catchment = re.fullmatch(r"(hard|exp|gaussian)_(\d+)m", scenario)
    if catchment:
        family, metres = catchment.groups()
        family_label = {"hard": "Hard", "exp": "Exponential", "gaussian": "Gaussian"}[
            family
        ]
        distance = int(metres)
        distance_label = (
            f"{distance // 1_000} km" if distance % 1_000 == 0 else f"{distance:,} m"
        )
        return f"{family_label} {distance_label}"
    if scenario.startswith("objective_loo__"):
        objective = scenario.removeprefix("objective_loo__")
        return f"LOO · {_CORRELATION_DISPLAY_LABELS.get(objective, objective.replace('_', ' ').title())}"
    return scenario.replace("_", " ").strip().title()


def _correlation_figures(
    all_venue: pd.DataFrame,
    admin_collapsed: pd.DataFrame,
) -> tuple[Any, Any, pd.DataFrame]:
    all_corr = _validate_square_correlation(all_venue, "all-venue correlation")
    admin_corr = _validate_square_correlation(admin_collapsed, "admin correlation")
    all_sample_n = _correlation_sample_n(all_corr)
    admin_sample_n = _correlation_sample_n(admin_corr)
    if set(all_corr.columns) != set(admin_corr.columns):
        raise ValueError("correlation views must contain identical metrics")
    admin_corr = admin_corr.reindex(index=all_corr.index, columns=all_corr.columns)
    # Use the independent-policy view to determine one common order.  Showing
    # both matrices in different orders would visually exaggerate separation.
    labels = hierarchical_order(admin_corr)
    all_corr = all_corr.reindex(index=labels, columns=labels)
    admin_corr = admin_corr.reindex(index=labels, columns=labels)
    delta_corr = admin_corr - all_corr
    display_labels = _unique_display_labels(labels, _CORRELATION_DISPLAY_LABELS)
    finite_delta = np.abs(delta_corr.to_numpy(dtype=float))
    finite_delta = finite_delta[np.isfinite(finite_delta)]
    delta_limit = min(
        1.0,
        max(0.05, float(finite_delta.max()) if finite_delta.size else 0.05),
    )
    _, plt = _matplotlib()
    with _plot_context():
        # Reserve explicit colour-bar columns.  Letting constrained_layout place
        # two shared colour bars can put the adaptive-delta scale beside the
        # wrong panel, which makes a careful comparison needlessly ambiguous.
        static = plt.figure(figsize=(16.5, 6.4), constrained_layout=True)
        grid = static.add_gridspec(
            1,
            5,
            width_ratios=(1.0, 1.0, 0.055, 1.0, 0.055),
            wspace=0.08,
        )
        axes = np.asarray(
            [
                static.add_subplot(grid[0, 0]),
                static.add_subplot(grid[0, 1]),
                static.add_subplot(grid[0, 3]),
            ],
            dtype=object,
        )
        raw_colour_axis = static.add_subplot(grid[0, 2])
        delta_colour_axis = static.add_subplot(grid[0, 4])
        raw_image = None
        delta_image = None
        for panel_index, (axis, matrix, title, colour_limit) in enumerate(
            (
                (
                    axes[0],
                    all_corr,
                    f"All venues — descriptive ({_sample_n_suffix(all_sample_n)})",
                    1.0,
                ),
                (
                    axes[1],
                    admin_corr,
                    "Admin-collapsed — policy units "
                    f"({_sample_n_suffix(admin_sample_n)})",
                    1.0,
                ),
                (axes[2], delta_corr, "Δρ: admin - all venues", delta_limit),
            )
        ):
            image = axis.imshow(
                matrix.to_numpy(),
                cmap="RdBu_r",
                vmin=-colour_limit,
                vmax=colour_limit,
            )
            if panel_index == 0:
                raw_image = image
            elif panel_index == 2:
                delta_image = image
            axis.set_xticks(
                range(len(labels)), display_labels, rotation=42, ha="right", fontsize=8
            )
            axis.set_yticks(range(len(labels)), display_labels, fontsize=8)
            axis.set_title(title, fontweight="bold")
            if len(labels) <= 10:
                for row in range(len(labels)):
                    for column in range(len(labels)):
                        value = matrix.iloc[row, column]
                        if pd.notna(value):
                            axis.text(
                                column,
                                row,
                                f"{value:.2f}",
                                ha="center",
                                va="center",
                                fontsize=7,
                                color=(
                                    "white"
                                    if abs(value) >= 0.62 * colour_limit
                                    else "#111827"
                                ),
                            )
        static.colorbar(
            raw_image,
            cax=raw_colour_axis,
            label="Spearman ρ",
        )
        static.colorbar(
            delta_image,
            cax=delta_colour_axis,
            label="Admin - all-venue Δρ",
        )
        static.suptitle(
            "Stage 3 dependency audit: repeated venues vs independent admins",
            fontweight="bold",
        )

    go, make_subplots = _plotly()
    interactive = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=(
            f"All venues — descriptive ({_sample_n_suffix(all_sample_n)})",
            "Admin-collapsed — policy units "
            f"({_sample_n_suffix(admin_sample_n)})",
            "Δρ: admin - all venues",
        ),
        horizontal_spacing=0.11,
    )
    canonical_y = np.repeat(np.asarray(labels, dtype=object)[:, None], len(labels), axis=1)
    canonical_x = np.repeat(np.asarray(labels, dtype=object)[None, :], len(labels), axis=0)
    canonical_pairs = np.stack((canonical_y, canonical_x), axis=-1)
    for column, matrix in ((1, all_corr), (2, admin_corr), (3, delta_corr)):
        is_delta = column == 3
        colour_limit = delta_limit if is_delta else 1.0
        colourbar = None
        if column == 2:
            colourbar = {"title": "ρ", "x": 0.665, "len": 0.72, "thickness": 12}
        elif is_delta:
            colourbar = {"title": "Δρ", "x": 1.02, "len": 0.72, "thickness": 12}
        interactive.add_trace(
            go.Heatmap(
                z=matrix.to_numpy(),
                x=display_labels,
                y=display_labels,
                customdata=canonical_pairs,
                zmin=-colour_limit,
                zmax=colour_limit,
                colorscale="RdBu",
                reversescale=True,
                colorbar=colourbar,
                showscale=column in {2, 3},
                hovertemplate=(
                    "%{y} × %{x}"
                    "<br>Canonical: %{customdata[0]} × %{customdata[1]}"
                    f"<br>{'Δρ' if is_delta else 'ρ'}=%{{z:.3f}}<extra></extra>"
                ),
            ),
            row=1,
            col=column,
        )
    interactive.update_layout(
        title="Stage 3 dependency audit",
        template="plotly_white",
        height=720,
        width=1_500,
        font={"family": "Malgun Gothic, NanumGothic, sans-serif"},
    )
    interactive.update_xaxes(tickangle=42, tickfont={"size": 10})
    interactive.update_yaxes(tickfont={"size": 10})
    digest_table = pd.concat(
        {
            "all_venue_descriptive": all_corr,
            "admin_collapsed": admin_corr,
            "admin_minus_all_delta": delta_corr,
        },
        names=["analysis_level", "metric"],
    ).reset_index()
    # Sample sizes change the publication claim and therefore belong in the
    # normalized digest used by the artifact manifest, not only in figure text.
    digest_table["all_venue_sample_n"] = int(all_sample_n)
    digest_table["admin_collapsed_sample_n"] = int(admin_sample_n)
    return static, interactive, digest_table


def _sensitivity_figures(frame: pd.DataFrame) -> tuple[Any, Any, pd.DataFrame]:
    data = _require_frame(frame, "sensitivity summary")
    scenario_col = _first_column(data, ("scenario", "variant", "method"), "sensitivity")
    metrics = [
        column
        for column in (
            "venue_rank_spearman",
            "admin_top3_jaccard_mean",
            "admin_top5_jaccard_mean",
            "pareto_retention",
            "region_best_venue_stability",
        )
        if column in data.columns
    ]
    if len(metrics) < 3:
        raise KeyError("sensitivity summary requires Spearman, Top3/Top5, and retention metrics")
    numeric = _finite(data, metrics, "sensitivity summary")
    for metric in metrics:
        lower = -1.0 if metric == "venue_rank_spearman" else 0.0
        if numeric[metric].lt(lower - 1e-9).any() or numeric[metric].gt(1 + 1e-9).any():
            interval = "[-1, 1]" if lower < 0 else "[0, 1]"
            raise ValueError(f"{metric} must be within {interval}")
    data.loc[:, metrics] = numeric
    canonical_scenarios = data[scenario_col].astype(str)
    declared_scope_col = next(
        (column for column in ("sensitivity_scope", "scope") if column in data.columns),
        None,
    )
    if declared_scope_col is None:
        scopes = canonical_scenarios.map(_sensitivity_scope)
    else:
        scopes = pd.Series(
            [
                _sensitivity_scope(scenario, declared)
                for scenario, declared in zip(
                    canonical_scenarios, data[declared_scope_col], strict=True
                )
            ],
            index=data.index,
            dtype="string",
        )
    x_lower = (
        -1.0
        if "venue_rank_spearman" in metrics
        and data["venue_rank_spearman"].lt(0).any()
        else 0.0
    )
    grouped: list[tuple[str, pd.DataFrame, list[str]]] = []
    for scope in _SENSITIVITY_SCOPE_ORDER:
        subset = data.loc[scopes.eq(scope)].copy()
        if subset.empty:
            continue
        labels = _unique_display_labels(
            subset[scenario_col].astype(str),
            {
                scenario: _sensitivity_scenario_label(scenario)
                for scenario in subset[scenario_col].astype(str)
            },
        )
        grouped.append((scope, subset, labels))
    if not grouped:  # pragma: no cover - guarded by the non-empty input check
        raise ValueError("sensitivity summary contains no scenarios")

    panel_count = len(grouped)
    maximum_rows = max(len(subset) for _, subset, _ in grouped)
    bar_height = 0.8 / len(metrics)
    _, plt = _matplotlib()
    with _plot_context():
        static, axes = plt.subplots(
            1,
            panel_count,
            figsize=(max(8.5, 5.25 * panel_count), max(6.2, 1.85 + maximum_rows * 0.72)),
            sharex=True,
            squeeze=False,
            constrained_layout=True,
        )
        axes_row = axes[0]
        legend_handles: list[Any] = []
        for panel_index, (scope, subset, scenario_labels) in enumerate(grouped):
            axis = axes_row[panel_index]
            positions = np.arange(len(subset))
            for metric_index, metric in enumerate(metrics):
                offset = (metric_index - (len(metrics) - 1) / 2) * bar_height
                bars = axis.barh(
                    positions + offset,
                    subset[metric].to_numpy(dtype=float),
                    height=bar_height,
                    label=_SENSITIVITY_METRIC_LABELS.get(
                        metric, metric.replace("_", " ")
                    ),
                    color=_TIER_COLOURS[metric_index % len(_TIER_COLOURS)],
                )
                if panel_index == 0:
                    legend_handles.append(bars)
            axis.set_yticks(positions, scenario_labels, fontsize=8.5)
            axis.invert_yaxis()
            axis.set_xlim(x_lower, 1.04)
            if x_lower < 0:
                axis.axvline(0, color="#6B7280", linewidth=0.8)
            axis.set_xlabel("Stability / retention")
            axis.set_title(f"{scope}\n(n={len(subset)})", fontweight="bold")
            axis.grid(axis="x", alpha=0.18)
            axis.set_axisbelow(True)
        static.suptitle("Stage 3 spatial sensitivity by scope", fontweight="bold")
        static.legend(
            legend_handles,
            [_SENSITIVITY_METRIC_LABELS.get(metric, metric) for metric in metrics],
            loc="outside lower center",
            ncol=min(3, len(metrics)),
            frameon=False,
            fontsize=8,
        )

    go, make_subplots = _plotly()
    interactive = make_subplots(
        rows=1,
        cols=panel_count,
        subplot_titles=tuple(
            f"{scope} (n={len(subset)})" for scope, subset, _ in grouped
        ),
        horizontal_spacing=0.08 if panel_count > 1 else 0.04,
    )
    for panel_index, (_, subset, scenario_labels) in enumerate(grouped, start=1):
        canonical_ids = subset[scenario_col].astype(str).to_numpy()
        for metric_index, metric in enumerate(metrics):
            customdata = np.column_stack(
                (canonical_ids, np.repeat(metric, len(subset)))
            )
            interactive.add_trace(
                go.Bar(
                    name=_SENSITIVITY_METRIC_LABELS.get(metric, metric),
                    legendgroup=metric,
                    showlegend=panel_index == 1,
                    x=subset[metric],
                    y=scenario_labels,
                    orientation="h",
                    customdata=customdata,
                    marker_color=_TIER_COLOURS[metric_index % len(_TIER_COLOURS)],
                    hovertemplate=(
                        "Scenario ID=%{customdata[0]}"
                        "<br>Metric ID=%{customdata[1]}"
                        "<br>%{y}: %{x:.3f}<extra></extra>"
                    ),
                ),
                row=1,
                col=panel_index,
            )
        interactive.update_xaxes(
            range=[x_lower, 1.04],
            title="Stability / retention",
            zeroline=x_lower < 0,
            row=1,
            col=panel_index,
        )
        interactive.update_yaxes(
            categoryorder="array",
            categoryarray=list(reversed(scenario_labels)),
            tickfont={"size": 10},
            row=1,
            col=panel_index,
        )
    interactive.update_layout(
        title="Stage 3 spatial sensitivity by scope",
        barmode="group",
        template="plotly_white",
        height=max(650, 215 + maximum_rows * 58),
        width=max(850, panel_count * 560),
        legend={"orientation": "h", "y": -0.14, "x": 0.5, "xanchor": "center"},
        font={"family": "Malgun Gothic, NanumGothic, sans-serif"},
    )
    return static, interactive, data


def _pareto_figures(frame: pd.DataFrame) -> tuple[Any, Any, pd.DataFrame]:
    data = _require_frame(frame, "Pareto shortlist")
    if "pareto_tier" not in data.columns:
        raise KeyError("Pareto shortlist missing pareto_tier")
    tiers = pd.to_numeric(data["pareto_tier"], errors="coerce")
    if tiers.isna().any() or (tiers < 1).any():
        raise ValueError("pareto_tier must be positive numeric")
    data["pareto_tier"] = tiers.astype(int)
    raw_col = next(
        (column for column in ("raw_elderly_exposure", "raw_exposure", "exposure") if column in data),
        None,
    )
    need_col = next(
        (column for column in ("need_weighted_exposure", "beneficiary_value") if column in data),
        None,
    )
    if raw_col and need_col:
        numeric = _finite(data, [raw_col, need_col], "Pareto shortlist")
        data[raw_col] = numeric[raw_col]
        data[need_col] = numeric[need_col]
    counts = data.groupby("pareto_tier", sort=True).size().rename("venue_count").reset_index()
    _, plt = _matplotlib()
    with _plot_context():
        static, axis = plt.subplots(figsize=(10.4, 6.4), constrained_layout=True)
        if raw_col and need_col:
            for tier, subset in data.groupby("pareto_tier", sort=True):
                colour = _TIER_COLOURS[min(int(tier) - 1, len(_TIER_COLOURS) - 1)]
                axis.scatter(
                    subset[raw_col],
                    subset[need_col],
                    label=f"Tier {tier} (n={len(subset)})",
                    color=colour,
                    s=40 if int(tier) == 1 else 24,
                    alpha=0.85 if int(tier) == 1 else 0.48,
                    edgecolor="white",
                    linewidth=0.35,
                )
            axis.set_xlabel("Raw elderly exposure")
            axis.set_ylabel("Need-weighted exposure")
        else:
            axis.bar(
                counts["pareto_tier"].astype(str),
                counts["venue_count"],
                color=[
                    _TIER_COLOURS[min(int(tier) - 1, len(_TIER_COLOURS) - 1)]
                    for tier in counts["pareto_tier"]
                ],
            )
            axis.set_xlabel("Pareto tier")
            axis.set_ylabel("Venue count")
        axis.set_title("Multi-objective Pareto venue summary", fontweight="bold")
        axis.grid(alpha=0.16)
        axis.legend(frameon=False, fontsize=8) if raw_col and need_col else None

    go, _ = _plotly()
    interactive = go.Figure()
    if raw_col and need_col:
        id_col = next((column for column in ("venue_id", "venue_name") if column in data), None)
        for tier, subset in data.groupby("pareto_tier", sort=True):
            interactive.add_trace(
                go.Scatter(
                    x=subset[raw_col],
                    y=subset[need_col],
                    mode="markers",
                    name=f"Tier {tier}",
                    text=subset[id_col].astype(str) if id_col else None,
                    marker={
                        "color": _TIER_COLOURS[min(int(tier) - 1, len(_TIER_COLOURS) - 1)],
                        "size": 10 if int(tier) == 1 else 7,
                        "opacity": 0.85 if int(tier) == 1 else 0.5,
                    },
                    hovertemplate="%{text}<br>Raw=%{x:.4g}<br>Need-weighted=%{y:.4g}<extra></extra>",
                )
            )
        interactive.update_xaxes(title=raw_col)
        interactive.update_yaxes(title=need_col)
    else:
        interactive.add_trace(
            go.Bar(
                x=counts["pareto_tier"].astype(str),
                y=counts["venue_count"],
                marker_color=[
                    _TIER_COLOURS[min(int(tier) - 1, len(_TIER_COLOURS) - 1)]
                    for tier in counts["pareto_tier"]
                ],
            )
        )
    interactive.update_layout(
        title="Multi-objective Pareto venue summary",
        template="plotly_white",
        font={"family": "Malgun Gothic, NanumGothic, sans-serif"},
    )
    return static, interactive, data


def _manifest_rows(
    figures: Mapping[str, Mapping[str, str]],
    normalized: Mapping[str, pd.DataFrame],
    output_dir: Path,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for figure_id in STAGE3_CORE_FIGURE_IDS:
        digest = _frame_sha256(normalized[figure_id])
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
                self_contained = "plotly" in content.lower() and external_scripts == 0
            rows.append(
                {
                    "figure_id": figure_id,
                    "format": artifact_format,
                    "relative_path": path.relative_to(output_dir).as_posix(),
                    "size_bytes": int(path.stat().st_size),
                    "sha256": sha256_file(path),
                    "input_table_sha256": digest,
                    "width_px": width,
                    "height_px": height,
                    "dpi_x": dpi_x,
                    "dpi_y": dpi_y,
                    "external_script_count": external_scripts,
                    "self_contained_html": self_contained,
                }
            )
    return pd.DataFrame(rows)


def validate_stage3_figure_manifest(manifest: pd.DataFrame) -> None:
    """Validate exact triplets, resolution, hashes, and offline HTML."""

    frame = _require_frame(manifest, "Stage 3 figure manifest")
    required = {
        "figure_id",
        "format",
        "relative_path",
        "size_bytes",
        "sha256",
        "input_table_sha256",
        "width_px",
        "height_px",
        "dpi_x",
        "dpi_y",
        "external_script_count",
        "self_contained_html",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"Stage 3 figure manifest missing columns: {missing}")
    expected = {
        (figure_id, artifact_format)
        for figure_id in STAGE3_CORE_FIGURE_IDS
        for artifact_format in ("png", "svg", "html")
    }
    observed = set(zip(frame["figure_id"], frame["format"]))
    if observed != expected or len(frame) != len(expected):
        raise ValueError("Stage 3 figure manifest must contain exactly four triplets")
    if (pd.to_numeric(frame["size_bytes"], errors="coerce") <= 100).any():
        raise ValueError("Stage 3 figure artifacts must be non-empty")
    if not frame["sha256"].astype(str).str.fullmatch(r"[0-9a-f]{64}").all():
        raise ValueError("Stage 3 figure artifact SHA256 is invalid")
    png = frame.loc[frame["format"].eq("png")]
    if (pd.to_numeric(png["dpi_x"], errors="coerce") < 299).any() or (
        pd.to_numeric(png["dpi_y"], errors="coerce") < 299
    ).any():
        raise ValueError("Stage 3 PNG figures require at least 300-dpi metadata")
    if (pd.to_numeric(png["width_px"], errors="coerce") < 1800).any() or (
        pd.to_numeric(png["height_px"], errors="coerce") < 1000
    ).any():
        raise ValueError("Stage 3 PNG figures are below the publication resolution floor")
    html = frame.loc[frame["format"].eq("html")]
    if not html["external_script_count"].eq(0).all() or not html[
        "self_contained_html"
    ].eq(True).all():
        raise ValueError("Stage 3 HTML figures must be self-contained")


def render_need_exposure_quadrant_figure(
    quadrants: pd.DataFrame,
    output_dir: str | Path,
    *,
    dpi: int = 300,
) -> tuple[Mapping[str, str], pd.DataFrame]:
    if int(dpi) < 300:
        raise ValueError("Stage 3 publication figures require dpi >= 300")
    static, interactive, normalized = _quadrant_figures(quadrants)
    paths = _save_figure_triplet(
        "need_exposure_quadrant", static, interactive, Path(output_dir), dpi=int(dpi)
    )
    return paths, normalized


def render_stage3_correlation_figure(
    all_venue_correlation: pd.DataFrame,
    admin_collapsed_correlation: pd.DataFrame,
    output_dir: str | Path,
    *,
    dpi: int = 300,
) -> tuple[Mapping[str, str], pd.DataFrame]:
    if int(dpi) < 300:
        raise ValueError("Stage 3 publication figures require dpi >= 300")
    static, interactive, normalized = _correlation_figures(
        all_venue_correlation, admin_collapsed_correlation
    )
    paths = _save_figure_triplet(
        "correlation_dependency_audit",
        static,
        interactive,
        Path(output_dir),
        dpi=int(dpi),
    )
    return paths, normalized


def render_sensitivity_figure(
    sensitivity_summary: pd.DataFrame,
    output_dir: str | Path,
    *,
    dpi: int = 300,
) -> tuple[Mapping[str, str], pd.DataFrame]:
    if int(dpi) < 300:
        raise ValueError("Stage 3 publication figures require dpi >= 300")
    static, interactive, normalized = _sensitivity_figures(sensitivity_summary)
    paths = _save_figure_triplet(
        "sensitivity_stability", static, interactive, Path(output_dir), dpi=int(dpi)
    )
    return paths, normalized


def render_pareto_summary_figure(
    pareto_candidates: pd.DataFrame,
    output_dir: str | Path,
    *,
    dpi: int = 300,
) -> tuple[Mapping[str, str], pd.DataFrame]:
    if int(dpi) < 300:
        raise ValueError("Stage 3 publication figures require dpi >= 300")
    static, interactive, normalized = _pareto_figures(pareto_candidates)
    paths = _save_figure_triplet(
        "pareto_shortlist_summary", static, interactive, Path(output_dir), dpi=int(dpi)
    )
    return paths, normalized


def render_stage3_core_figures(
    tables: Mapping[str, pd.DataFrame],
    output_dir: str | Path,
    *,
    dpi: int = 300,
) -> Stage3FigureResult:
    """Render the four decision-critical Stage 3 publication figures.

    Required keys are ``need_exposure_quadrant``, ``correlation_all_venue``,
    ``correlation_admin_collapsed``, ``sensitivity_stability``, and
    ``pareto_shortlist_summary``.
    """

    if int(dpi) < 300:
        raise ValueError("Stage 3 publication figures require dpi >= 300")
    required = {
        "need_exposure_quadrant",
        "correlation_all_venue",
        "correlation_admin_collapsed",
        "sensitivity_stability",
        "pareto_shortlist_summary",
    }
    missing = sorted(required - set(tables))
    if missing:
        raise KeyError(f"missing Stage 3 core figure tables: {missing}")
    destination = Path(output_dir)
    figures: dict[str, Mapping[str, str]] = {}
    normalized: dict[str, pd.DataFrame] = {}
    figures["need_exposure_quadrant"], normalized["need_exposure_quadrant"] = (
        render_need_exposure_quadrant_figure(
            tables["need_exposure_quadrant"], destination, dpi=int(dpi)
        )
    )
    figures["correlation_dependency_audit"], normalized[
        "correlation_dependency_audit"
    ] = render_stage3_correlation_figure(
        tables["correlation_all_venue"],
        tables["correlation_admin_collapsed"],
        destination,
        dpi=int(dpi),
    )
    figures["sensitivity_stability"], normalized["sensitivity_stability"] = (
        render_sensitivity_figure(
            tables["sensitivity_stability"], destination, dpi=int(dpi)
        )
    )
    figures["pareto_shortlist_summary"], normalized["pareto_shortlist_summary"] = (
        render_pareto_summary_figure(
            tables["pareto_shortlist_summary"], destination, dpi=int(dpi)
        )
    )
    manifest = _manifest_rows(figures, normalized, destination)
    validate_stage3_figure_manifest(manifest)
    return Stage3FigureResult(
        figures=figures,
        artifact_manifest=manifest,
        normalized_tables=normalized,
    )


def _evaluate_comparison(observed: Any, comparison: str, threshold: Any) -> bool:
    if comparison == "contract":
        if isinstance(observed, (bool, np.bool_)):
            return bool(observed)
        return str(observed).strip().upper() in {"PASS", "TRUE", "VALID", "UNCHANGED"}
    try:
        left = float(observed)
        right = float(threshold)
    except (TypeError, ValueError) as exc:
        raise ValueError("numeric quality-gate comparison requires numeric values") from exc
    if not np.isfinite([left, right]).all():
        raise ValueError("quality-gate comparison values must be finite")
    return {
        ">=": left >= right,
        ">": left > right,
        "<=": left <= right,
        "<": left < right,
        "==": left == right,
        "!=": left != right,
    }[comparison]


def quality_gate_record(
    gate_id: str,
    *,
    observed: Any,
    comparison: str = "contract",
    threshold: Any = "PASS",
    passed: bool | None = None,
    severity: str = "hard",
    threshold_source: str = "stage3_prompt_contract",
    note: str | None = None,
) -> dict[str, Any]:
    """Create one auditable quality-gate record with optional evaluation."""

    gate_id = str(gate_id).strip()
    severity = str(severity).strip().lower()
    comparison = str(comparison).strip()
    threshold_source = str(threshold_source).strip()
    if not gate_id:
        raise ValueError("gate_id cannot be blank")
    if severity not in {"hard", "advisory", "diagnostic"}:
        raise ValueError(f"unsupported quality-gate severity: {severity}")
    if comparison not in {">=", ">", "<=", "<", "==", "!=", "contract"}:
        raise ValueError(f"unsupported quality-gate comparison: {comparison}")
    if not re.search(r"config|prompt|contract|preregister", threshold_source, flags=re.I):
        raise ValueError("threshold_source must identify a config/prompt/contract")
    evaluated = _evaluate_comparison(observed, comparison, threshold)
    if passed is not None and bool(passed) != evaluated:
        raise ValueError("explicit passed flag conflicts with observed comparison")
    message = str(note).strip() if note is not None else f"{gate_id} evaluated by {comparison}"
    if not message:
        raise ValueError("quality-gate note cannot be blank")
    return {
        "gate_id": gate_id,
        "severity": severity,
        "passed": evaluated,
        "observed": observed,
        "comparison": comparison,
        "threshold": threshold,
        "threshold_source": threshold_source,
        "note": message,
    }


make_quality_gate_record = quality_gate_record


def build_stage3_quality_gate(
    records: pd.DataFrame | Sequence[Mapping[str, Any]],
    *,
    stage4_started: bool = False,
) -> Stage3QualityGateResult:
    """Validate and summarize a Stage 3 hard/advisory/diagnostic ledger."""

    frame = records.copy() if isinstance(records, pd.DataFrame) else pd.DataFrame(records)
    frame = _require_frame(frame, "Stage 3 quality-gate records")
    missing = sorted(set(QUALITY_GATE_COLUMNS) - set(frame.columns))
    if missing:
        raise KeyError(f"Stage 3 quality-gate records missing columns: {missing}")
    frame = frame.loc[:, QUALITY_GATE_COLUMNS].copy()
    frame["gate_id"] = frame["gate_id"].astype(str).str.strip()
    if frame["gate_id"].eq("").any() or frame["gate_id"].duplicated().any():
        raise ValueError("Stage 3 quality-gate ids must be non-blank and unique")
    frame["severity"] = frame["severity"].astype(str).str.strip().str.lower()
    invalid = sorted(set(frame["severity"]) - {"hard", "advisory", "diagnostic"})
    if invalid:
        raise ValueError(f"invalid Stage 3 quality-gate severities: {invalid}")
    if not frame["severity"].eq("hard").any():
        raise ValueError("Stage 3 quality gate requires at least one hard record")
    if not frame["passed"].map(lambda value: isinstance(value, (bool, np.bool_))).all():
        raise ValueError("Stage 3 quality-gate passed values must be boolean")
    frame["passed"] = frame["passed"].astype(bool)
    for column in ("comparison", "threshold_source", "note"):
        if frame[column].isna().any() or frame[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"Stage 3 quality-gate {column} must be populated")
    if bool(stage4_started):
        raise ValueError("Stage 4 must remain unstarted while finalizing Stage 3")
    severity_type = pd.CategoricalDtype(
        categories=["hard", "advisory", "diagnostic"], ordered=True
    )
    frame["severity"] = frame["severity"].astype(severity_type)
    frame["status"] = np.where(frame["passed"], "PASS", "FAIL")
    frame = frame.sort_values(["severity", "gate_id"], kind="stable").reset_index(drop=True)
    hard = frame["severity"].eq("hard")
    hard_failed = int((hard & ~frame["passed"]).sum())
    return Stage3QualityGateResult(
        gates=frame,
        hard_pass=hard_failed == 0,
        hard_total=int(hard.sum()),
        hard_failed=hard_failed,
        advisory_failed=int(
            (frame["severity"].eq("advisory") & ~frame["passed"]).sum()
        ),
        diagnostic_failed=int(
            (frame["severity"].eq("diagnostic") & ~frame["passed"]).sum()
        ),
    )


def validate_stage3_quality_gate(
    gates: pd.DataFrame,
    *,
    stage4_started: bool = False,
) -> Stage3QualityGateResult:
    """Validate a materialized gate table while ignoring derived status."""

    return build_stage3_quality_gate(
        gates.drop(columns="status", errors="ignore"),
        stage4_started=stage4_started,
    )


__all__ = [
    "STAGE3_CORE_FIGURE_IDS",
    "QUALITY_GATE_COLUMNS",
    "Stage3FigureResult",
    "Stage3QualityGateResult",
    "render_need_exposure_quadrant_figure",
    "render_stage3_correlation_figure",
    "render_sensitivity_figure",
    "render_pareto_summary_figure",
    "render_stage3_core_figures",
    "validate_stage3_figure_manifest",
    "quality_gate_record",
    "make_quality_gate_record",
    "build_stage3_quality_gate",
    "validate_stage3_quality_gate",
]
