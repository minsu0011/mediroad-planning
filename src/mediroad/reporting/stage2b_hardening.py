"""Publication and validation helpers for the Stage 2B hardening release.

The module is intentionally independent from the Stage 2B computation engine.
It accepts nine small, explicit evidence tables, validates their analysis-unit
contracts, and emits deterministic 300-dpi PNG, SVG, and self-contained HTML
figures.  It also provides the evidence-summary, quality-gate, and artifact-
inventory checks used by the final Stage 2 freeze decision.

These helpers do not infer patient demand, do not release an exact month/date,
and do not mutate Stage 1 or Stage 2A artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import re
import struct
from typing import Any, Mapping, Sequence
import uuid

import numpy as np
import pandas as pd


os.environ.setdefault("MPLBACKEND", "Agg")


FIGURE_IDS: tuple[str, ...] = (
    "bundle_year_season_rank_heatmap",
    "cross_year_transfer_matrix",
    "bundle_season_margin_chart",
    "heldout_regret_chart",
    "bootstrap_season_selection_probability",
    "leave_one_month_out_stability",
    "leave_one_sigungu_out_stability",
    "observed_vs_permutation_null",
    "temporal_confidence_summary",
)

FIGURE_REQUIRED_COLUMNS: Mapping[str, tuple[str, ...]] = {
    "bundle_year_season_rank_heatmap": (
        "bundle_id",
        "season",
        "year2022_rank",
        "year2023_rank",
        "pooled_rank",
        "cross_year_consensus_rank",
    ),
    "cross_year_transfer_matrix": (
        "bundle_id",
        "direction",
        "top1_agreement",
        "train_primary_in_holdout_top2",
        "pair_top2_coverage",
        "season_rank_spearman",
        "season_rank_kendall",
    ),
    "bundle_season_margin_chart": (
        "bundle_id",
        "direction",
        "train_top1_margin",
    ),
    "heldout_regret_chart": (
        "bundle_id",
        "direction",
        "heldout_regret_normalized",
    ),
    "bootstrap_season_selection_probability": (
        "bundle_id",
        "season",
        "primary_probability",
    ),
    "leave_one_month_out_stability": (
        "bundle_id",
        "year",
        "removed_month",
        "primary_retained",
    ),
    "leave_one_sigungu_out_stability": (
        "bundle_id",
        "removed_sigungu",
        "rank_spearman",
    ),
    "observed_vs_permutation_null": (
        "bundle_id",
        "metric",
        "observed",
        "null_mean",
        "null_sd",
        "p_value_upper",
        "standardized_strength",
    ),
    "temporal_confidence_summary": (
        "bundle_id",
        "release_type",
        "temporal_confidence",
    ),
}

# Aliases are deliberately explicit.  If more than one candidate is present,
# normalization fails closed and the caller must resolve it with column_maps.
COLUMN_ALIASES: Mapping[str, tuple[str, ...]] = {
    "bundle_id": ("bundle", "bundle_name", "service_bundle"),
    "season": ("season_id", "recommended_season"),
    "year2022_rank": ("2022_rank", "rank_2022"),
    "year2023_rank": ("2023_rank", "rank_2023"),
    "pooled_rank": ("rank_pooled",),
    "cross_year_consensus_rank": ("consensus_rank", "rank_consensus"),
    "direction": ("transfer_direction", "train_holdout_direction"),
    "top1_agreement": ("cross_year_top1_agreement",),
    "train_primary_in_holdout_top2": (
        "primary_in_holdout_top2",
        "top2_inclusion",
    ),
    "pair_top2_coverage": ("top2_pair_coverage",),
    "season_rank_spearman": ("rank_spearman", "spearman"),
    "season_rank_kendall": ("rank_kendall", "kendall"),
    "train_top1_margin": ("training_top1_margin",),
    "holdout_top1_margin": ("test_top1_margin",),
    "heldout_regret_normalized": ("normalized_regret", "heldout_regret"),
    "primary_probability": (
        "bootstrap_primary_probability",
        "selection_probability",
    ),
    "year": ("evaluation_year",),
    "removed_month": ("omitted_month", "left_out_month", "month"),
    "primary_retained": ("primary_retention", "primary_season_retained"),
    "removed_sigungu": ("omitted_sigungu", "left_out_sigungu"),
    "rank_spearman": ("season_rank_spearman", "rank_correlation"),
    "metric": ("null_metric",),
    "observed": ("observed_value",),
    "null_mean": ("permutation_mean",),
    "null_sd": ("permutation_sd",),
    "p_value_upper": ("permutation_p_value", "p_value"),
    "standardized_strength": ("permutation_strength", "z_strength"),
    "release_type": ("temporal_release_type",),
    "temporal_confidence": ("confidence",),
}

SEASON_ORDER: tuple[str, ...] = ("spring", "summer", "autumn", "winter")
SEASON_LABELS_KO: Mapping[str, str] = {
    "spring": "봄",
    "summer": "여름",
    "autumn": "가을",
    "winter": "겨울",
}
BUNDLE_ORDER: tuple[str, ...] = (
    "chronic",
    "comprehensive",
    "musculoskeletal",
    "neuro",
    "sensory",
)
BUNDLE_LABELS_KO: Mapping[str, str] = {
    "chronic": "만성질환",
    "comprehensive": "복합기본진료",
    "musculoskeletal": "근골격·재활",
    "neuro": "신경·정신",
    "sensory": "감각·구강·피부",
}
VALID_CONFIDENCE = frozenset({"HIGH", "MODERATE", "LOW"})
VALID_RELEASE_TYPES = frozenset(
    {"STRONG_SINGLE", "ROBUST_PAIR", "NO_STRONG_PREFERENCE"}
)
VALID_SEVERITIES = frozenset({"hard", "advisory", "diagnostic"})
VALID_FINAL_DECISIONS = frozenset(
    {
        "PASS_STRONG_TEMPORAL",
        "PASS_COARSE_TEMPORAL",
        "PASS_ADVISORY_TEMPORAL",
        "FAIL_TEMPORAL_RELEASE",
    }
)
CONFIDENCE_LABELS_KO: Mapping[str, str] = {
    "LOW": "낮음",
    "MODERATE": "중간",
    "HIGH": "높음",
}
RELEASE_TYPE_LABELS_KO: Mapping[str, str] = {
    "STRONG_SINGLE": "강한 단일 계절",
    "ROBUST_PAIR": "안정적 계절 쌍",
    "NO_STRONG_PREFERENCE": "계절 우선순위 확정 없음",
}
NULL_METRIC_LABELS_KO: Mapping[str, str] = {
    "bundle_seasonal_separation": "번들 계절 분리",
    "cross_year_rank_spearman": "교차연도 순위 ρ",
    "cross_year_top1_agreement": "교차연도 Top1 일치",
    "season_concentration": "계절 집중도",
    "top1_margin_normalized": "정규화 Top1 margin",
}


@dataclass(frozen=True)
class Stage2BHardeningReportingResult:
    """All nine figure triplets plus a deterministic publication manifest."""

    figures: Mapping[str, Mapping[str, str]]
    artifact_manifest: pd.DataFrame
    artifact_manifest_path: str
    normalized_tables: Mapping[str, pd.DataFrame]


@dataclass(frozen=True)
class Stage2BQualityGateResult:
    """Validated gate ledger and release-level counts."""

    gates: pd.DataFrame
    hard_pass: bool
    hard_total: int
    hard_failed: int
    advisory_failed: int
    diagnostic_failed: int
    final_decision: str | None


@dataclass(frozen=True)
class ArtifactInventoryValidation:
    """Result of checking a saved artifact inventory against the filesystem."""

    valid: bool
    checked_count: int
    missing_count: int
    size_mismatch_count: int
    sha_mismatch_count: int
    empty_count: int
    detail: pd.DataFrame


def _font_family() -> str:
    """Register a Korean-capable font on Windows or WSL and return its family."""

    try:
        from matplotlib import font_manager
    except ImportError:  # pragma: no cover
        return "DejaVu Sans"
    candidates = (
        (
            Path("/mnt/c/Windows/Fonts/malgun.ttf"),
            Path("/mnt/c/Windows/Fonts/malgunbd.ttf"),
        ),
        (
            Path("C:/Windows/Fonts/malgun.ttf"),
            Path("C:/Windows/Fonts/malgunbd.ttf"),
        ),
        (Path("/usr/share/fonts/truetype/nanum/NanumGothic.ttf"), None),
        (Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"), None),
    )
    for regular, bold in candidates:
        if regular.is_file():
            font_manager.fontManager.addfont(str(regular))
            if bold is not None and bold.is_file():
                font_manager.fontManager.addfont(str(bold))
            return font_manager.FontProperties(fname=str(regular)).get_name()
    return "DejaVu Sans"


def _atomic_path(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.with_name(f".{path.stem}.{uuid.uuid4().hex}.tmp{path.suffix}")


def _atomic_write_text(text: str, path: Path, *, encoding: str = "utf-8") -> None:
    temporary = _atomic_path(path)
    try:
        temporary.write_text(text, encoding=encoding)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = _atomic_path(path)
    try:
        frame.to_csv(temporary, index=False, encoding="utf-8-sig")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _frame_sha256(frame: pd.DataFrame) -> str:
    stable = frame.copy()
    stable.columns = stable.columns.map(str)
    payload = stable.to_csv(index=False, lineterminator="\n", na_rep="<NA>")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _normalize_columns(
    figure_id: str,
    table: pd.DataFrame,
    column_map: Mapping[str, str] | None,
) -> pd.DataFrame:
    if not isinstance(table, pd.DataFrame) or table.empty:
        raise ValueError(f"{figure_id} must be a non-empty DataFrame")
    frame = table.copy()
    frame.columns = frame.columns.map(str)
    if not frame.columns.is_unique:
        raise ValueError(f"{figure_id} columns must be unique")
    if column_map:
        unknown_sources = sorted(set(column_map) - set(frame.columns))
        if unknown_sources:
            raise KeyError(
                f"{figure_id} column_map sources not found: {unknown_sources}"
            )
        destinations = list(column_map.values())
        if len(destinations) != len(set(destinations)):
            raise ValueError(f"{figure_id} column_map destinations must be unique")
        frame = frame.rename(columns=dict(column_map))
    required = FIGURE_REQUIRED_COLUMNS[figure_id]
    for canonical in required:
        if canonical in frame.columns:
            continue
        candidates = [
            alias
            for alias in COLUMN_ALIASES.get(canonical, ())
            if alias in frame.columns
        ]
        if len(candidates) > 1:
            raise ValueError(
                f"{figure_id} has ambiguous aliases for {canonical}: {candidates}; "
                "resolve with column_maps"
            )
        if candidates:
            frame = frame.rename(columns={candidates[0]: canonical})
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise KeyError(f"{figure_id} missing required columns: {missing}")
    return frame


def _numeric(frame: pd.DataFrame, columns: Sequence[str], figure_id: str) -> None:
    for column in columns:
        converted = pd.to_numeric(frame[column], errors="coerce")
        invalid = converted.isna() | ~np.isfinite(converted.to_numpy(dtype=float))
        if invalid.any():
            raise ValueError(
                f"{figure_id}.{column} must contain only finite numeric values"
            )
        frame[column] = converted.astype(float)


def _probability(frame: pd.DataFrame, columns: Sequence[str], figure_id: str) -> None:
    _numeric(frame, columns, figure_id)
    for column in columns:
        if not frame[column].between(0.0, 1.0, inclusive="both").all():
            raise ValueError(f"{figure_id}.{column} must be in [0, 1]")


def _bool_numeric(series: pd.Series, label: str) -> pd.Series:
    mapping = {
        True: 1.0,
        False: 0.0,
        1: 1.0,
        0: 0.0,
        1.0: 1.0,
        0.0: 0.0,
        "true": 1.0,
        "false": 0.0,
        "1": 1.0,
        "0": 0.0,
    }
    normalized = series.map(
        lambda value: mapping.get(value.lower(), np.nan)
        if isinstance(value, str)
        else mapping.get(value, np.nan)
    )
    if normalized.isna().any():
        raise ValueError(f"{label} must contain only boolean/0/1 values")
    return normalized.astype(float)


def _bundle_sort_key(value: str) -> tuple[int, str]:
    try:
        return (BUNDLE_ORDER.index(value), value)
    except ValueError:
        return (len(BUNDLE_ORDER), value)


def _bundle_label(value: str) -> str:
    return f"{value} | {BUNDLE_LABELS_KO.get(value, value)}"


def _direction_label(value: Any) -> str:
    text = str(value).strip().replace("->", "→")
    aliases = {
        "2022_to_2023": "2022→2023",
        "2023_to_2022": "2023→2022",
        "22_to_23": "2022→2023",
        "23_to_22": "2023→2022",
    }
    return aliases.get(text, text)


def _assert_unique(frame: pd.DataFrame, keys: Sequence[str], figure_id: str) -> None:
    duplicated = frame.duplicated(list(keys), keep=False)
    if duplicated.any():
        sample = frame.loc[duplicated, list(keys)].head(5).to_dict("records")
        raise ValueError(f"{figure_id} duplicate keys {list(keys)}: {sample}")


def _normalize_figure_table(figure_id: str, frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["bundle_id"] = result["bundle_id"].astype(str).str.strip()
    if result["bundle_id"].eq("").any():
        raise ValueError(f"{figure_id}.bundle_id cannot be blank")

    if figure_id == "bundle_year_season_rank_heatmap":
        result["season"] = result["season"].astype(str).str.lower().str.strip()
        if not set(result["season"]).issubset(SEASON_ORDER):
            raise ValueError(f"{figure_id} contains an unsupported season")
        rank_columns = list(FIGURE_REQUIRED_COLUMNS[figure_id][2:])
        _numeric(result, rank_columns, figure_id)
        if (result[rank_columns] < 1.0).any().any():
            raise ValueError(f"{figure_id} ranks must be >= 1")
        _assert_unique(result, ["bundle_id", "season"], figure_id)
        coverage = result.groupby("bundle_id")["season"].agg(set)
        if not coverage.map(lambda values: values == set(SEASON_ORDER)).all():
            raise ValueError(f"{figure_id} requires all four seasons per bundle")
    elif figure_id == "cross_year_transfer_matrix":
        result["direction"] = result["direction"].map(_direction_label)
        if result["direction"].eq("").any():
            raise ValueError(f"{figure_id}.direction cannot be blank")
        metrics = list(FIGURE_REQUIRED_COLUMNS[figure_id][2:])
        _numeric(result, metrics, figure_id)
        if not result[metrics].apply(lambda column: column.between(-1, 1)).all().all():
            raise ValueError(f"{figure_id} transfer metrics must be in [-1, 1]")
        _assert_unique(result, ["bundle_id", "direction"], figure_id)
        coverage = result.groupby("bundle_id")["direction"].agg(set)
        expected_directions = {"2022→2023", "2023→2022"}
        if not coverage.map(lambda values: values == expected_directions).all():
            raise ValueError(f"{figure_id} requires both transfer directions per bundle")
    elif figure_id == "bundle_season_margin_chart":
        result["direction"] = result["direction"].map(_direction_label)
        if result["direction"].eq("").any():
            raise ValueError(f"{figure_id}.direction cannot be blank")
        metrics = ["train_top1_margin"]
        if "holdout_top1_margin" in result.columns:
            metrics.append("holdout_top1_margin")
        _numeric(result, metrics, figure_id)
        if (result[metrics] < 0).any().any():
            raise ValueError(f"{figure_id} margins must be non-negative")
        coverage = result.groupby("bundle_id")["direction"].agg(set)
        if not coverage.map(
            lambda values: values == {"2022→2023", "2023→2022"}
        ).all():
            raise ValueError(f"{figure_id} requires both transfer directions per bundle")
        if "holdout_top1_margin" not in result.columns:
            # The strict engine's summary publishes the training margin only.
            # With exactly two reciprocal directions, the holdout-year margin
            # is the reverse direction's training margin.  Derive it without
            # pooling the fit and evaluation years.
            direction_medians = result.groupby(
                ["bundle_id", "direction"], observed=True
            )["train_top1_margin"].median()
            reverse = {"2022→2023": "2023→2022", "2023→2022": "2022→2023"}
            result["holdout_top1_margin"] = [
                float(direction_medians.loc[(bundle, reverse[direction])])
                for bundle, direction in zip(
                    result["bundle_id"].astype(str),
                    result["direction"].astype(str),
                    strict=True,
                )
            ]
    elif figure_id == "heldout_regret_chart":
        result["direction"] = result["direction"].map(_direction_label)
        if result["direction"].eq("").any():
            raise ValueError(f"{figure_id}.direction cannot be blank")
        _numeric(result, ["heldout_regret_normalized"], figure_id)
        if (result["heldout_regret_normalized"] < 0).any():
            raise ValueError(f"{figure_id} regret must be non-negative")
        coverage = result.groupby("bundle_id")["direction"].agg(set)
        if not coverage.map(
            lambda values: values == {"2022→2023", "2023→2022"}
        ).all():
            raise ValueError(f"{figure_id} requires both transfer directions per bundle")
    elif figure_id == "bootstrap_season_selection_probability":
        result["season"] = result["season"].astype(str).str.lower().str.strip()
        if not set(result["season"]).issubset(SEASON_ORDER):
            raise ValueError(f"{figure_id} contains an unsupported season")
        _probability(result, ["primary_probability"], figure_id)
        _assert_unique(result, ["bundle_id", "season"], figure_id)
        coverage = result.groupby("bundle_id")["season"].agg(set)
        if not coverage.map(lambda values: values == set(SEASON_ORDER)).all():
            raise ValueError(f"{figure_id} requires all four seasons per bundle")
        totals = result.groupby("bundle_id")["primary_probability"].sum()
        if not np.allclose(totals.to_numpy(), 1.0, atol=0.02):
            raise ValueError(f"{figure_id} probabilities must sum to one per bundle")
    elif figure_id == "leave_one_month_out_stability":
        result["year"] = pd.to_numeric(result["year"], errors="coerce")
        result["removed_month"] = pd.to_numeric(
            result["removed_month"], errors="coerce"
        )
        if result[["year", "removed_month"]].isna().any().any():
            raise ValueError(f"{figure_id} year/month must be numeric")
        result["year"] = result["year"].astype(int)
        result["removed_month"] = result["removed_month"].astype(int)
        result["primary_retained"] = _bool_numeric(
            result["primary_retained"], f"{figure_id}.primary_retained"
        )
        _assert_unique(result, ["bundle_id", "year", "removed_month"], figure_id)
        if set(result["year"]) != {2022, 2023}:
            raise ValueError(f"{figure_id} requires years 2022 and 2023")
        coverage = result.groupby(["bundle_id", "year"])["removed_month"].agg(set)
        if not coverage.map(lambda values: values == set(range(1, 13))).all():
            raise ValueError(f"{figure_id} requires exactly 12 removals per bundle-year")
    elif figure_id == "leave_one_sigungu_out_stability":
        result["removed_sigungu"] = result["removed_sigungu"].astype(str).str.strip()
        _numeric(result, ["rank_spearman"], figure_id)
        if not result["rank_spearman"].between(-1, 1).all():
            raise ValueError(f"{figure_id}.rank_spearman must be in [-1, 1]")
        _assert_unique(result, ["bundle_id", "removed_sigungu"], figure_id)
        counts = result.groupby("bundle_id")["removed_sigungu"].nunique()
        if not counts.eq(11).all():
            raise ValueError(f"{figure_id} requires exactly 11 removals per bundle")
        sigungu_sets = result.groupby("bundle_id")["removed_sigungu"].agg(set)
        if len({tuple(sorted(values)) for values in sigungu_sets}) != 1:
            raise ValueError(f"{figure_id} must remove the same 11 policy sigungu")
    elif figure_id == "observed_vs_permutation_null":
        result["metric"] = result["metric"].astype(str).str.strip()
        if result["metric"].eq("").any():
            raise ValueError(f"{figure_id}.metric cannot be blank")
        numeric = [
            "observed",
            "null_mean",
            "null_sd",
            "p_value_upper",
            "standardized_strength",
        ]
        _numeric(result, numeric, figure_id)
        if (result["null_sd"] < 0).any():
            raise ValueError(f"{figure_id}.null_sd must be non-negative")
        if not result["p_value_upper"].between(0, 1).all():
            raise ValueError(f"{figure_id}.p_value_upper must be in [0, 1]")
        _assert_unique(result, ["bundle_id", "metric"], figure_id)
        metric_sets = result.groupby("bundle_id")["metric"].agg(set)
        if len({tuple(sorted(values)) for values in metric_sets}) != 1:
            raise ValueError(f"{figure_id} must report the same null metrics per bundle")
    elif figure_id == "temporal_confidence_summary":
        result["release_type"] = result["release_type"].astype(str).str.upper().str.strip()
        result["temporal_confidence"] = (
            result["temporal_confidence"].astype(str).str.upper().str.strip()
        )
        invalid_release = sorted(set(result["release_type"]) - VALID_RELEASE_TYPES)
        invalid_confidence = sorted(
            set(result["temporal_confidence"]) - VALID_CONFIDENCE
        )
        if invalid_release:
            raise ValueError(f"invalid release_type values: {invalid_release}")
        if invalid_confidence:
            raise ValueError(
                f"invalid temporal_confidence values: {invalid_confidence}"
            )
        _assert_unique(result, ["bundle_id"], figure_id)

    bundle_order = sorted(result["bundle_id"].unique(), key=_bundle_sort_key)
    result["bundle_id"] = pd.Categorical(
        result["bundle_id"], categories=bundle_order, ordered=True
    )
    sort_columns = [
        column
        for column in (
            "bundle_id",
            "year",
            "season",
            "direction",
            "removed_month",
            "removed_sigungu",
            "metric",
        )
        if column in result.columns
    ]
    return result.sort_values(sort_columns, kind="stable").reset_index(drop=True)


def _matplotlib_modules() -> tuple[Any, Any]:
    try:
        import matplotlib as mpl

        mpl.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Stage 2B hardening figures require matplotlib") from exc
    return mpl, plt


def _plotly_go() -> Any:
    try:
        import plotly.graph_objects as go
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Stage 2B hardening figures require plotly") from exc
    return go


def _save_figure_triplet(
    figure_id: str,
    mpl_figure: Any,
    plotly_figure: Any,
    output_dir: Path,
    *,
    dpi: int,
) -> Mapping[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "png": output_dir / f"{figure_id}.png",
        "svg": output_dir / f"{figure_id}.svg",
        "html": output_dir / f"{figure_id}.html",
    }
    mpl, plt = _matplotlib_modules()
    # Matplotlib resolves generic ``sans-serif`` at draw/save time, not when
    # Text objects are created.  Keep the Korean font context active for both
    # PNG and SVG rendering; otherwise a figure created correctly can silently
    # fall back to DejaVu Sans when it is materialized later.
    font = _font_family()
    with mpl.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [font, "DejaVu Sans"],
            "axes.unicode_minus": False,
            "svg.fonttype": "path",
        }
    ):
        for suffix in ("png", "svg"):
            temporary = _atomic_path(paths[suffix])
            try:
                mpl_figure.savefig(
                    temporary,
                    format=suffix,
                    dpi=dpi,
                    facecolor="white",
                    bbox_inches="tight",
                )
                os.replace(temporary, paths[suffix])
            finally:
                temporary.unlink(missing_ok=True)
    plt.close(mpl_figure)
    html_text = plotly_figure.to_html(
        include_plotlyjs=True,
        full_html=True,
        config={"displaylogo": False, "responsive": True},
    )
    _atomic_write_text(html_text, paths["html"])
    return {key: str(value) for key, value in paths.items()}


def _heatmap_figures(
    values: np.ndarray,
    row_labels: Sequence[str],
    column_labels: Sequence[str],
    *,
    title: str,
    subtitle: str,
    colour_map: str,
    plotly_scale: str | Sequence[Any],
    vmin: float,
    vmax: float,
    value_format: str = ".2f",
    colorbar_title: str = "값",
) -> tuple[Any, Any]:
    mpl, plt = _matplotlib_modules()
    font = _font_family()
    width = max(10.0, 0.8 * len(column_labels) + 4.0)
    height = max(5.0, 0.50 * len(row_labels) + 3.0)
    with mpl.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [font, "DejaVu Sans"],
            "axes.unicode_minus": False,
            "svg.fonttype": "path",
        }
    ):
        figure, axis = plt.subplots(figsize=(width, height), constrained_layout=True)
        image = axis.imshow(
            values,
            cmap=colour_map,
            vmin=vmin,
            vmax=vmax,
            aspect="auto",
            interpolation="nearest",
        )
        axis.set_xticks(np.arange(len(column_labels)), column_labels, rotation=35, ha="right")
        axis.set_yticks(np.arange(len(row_labels)), row_labels)
        axis.set_title(title, fontsize=14, fontweight="bold", pad=22)
        axis.text(
            0.5,
            1.01,
            subtitle,
            transform=axis.transAxes,
            ha="center",
            va="bottom",
            fontsize=8.5,
            color="#4B5563",
        )
        colour_bar = figure.colorbar(image, ax=axis, fraction=0.035, pad=0.02)
        colour_bar.set_label(colorbar_title)
        # Malgun Gothic reliably covers Korean but not U+2212 on every Windows
        # build.  Numeric colour-bar ticks use DejaVu explicitly so negative
        # transfer correlations never render as a missing-glyph box.
        for tick_label in colour_bar.ax.get_yticklabels():
            tick_label.set_fontfamily("DejaVu Sans")
        span = max(abs(vmax - vmin), 1e-12)
        for row in range(values.shape[0]):
            for column in range(values.shape[1]):
                value = values[row, column]
                if not np.isfinite(value):
                    continue
                scaled = abs((value - vmin) / span - 0.5) * 2
                axis.text(
                    column,
                    row,
                    format(float(value), value_format),
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if scaled > 0.58 else "#111827",
                )

    go = _plotly_go()
    interactive = go.Figure(
        go.Heatmap(
            z=values,
            x=list(column_labels),
            y=list(row_labels),
            zmin=vmin,
            zmax=vmax,
            colorscale=plotly_scale,
            colorbar={"title": colorbar_title},
            text=np.vectorize(lambda value: format(float(value), value_format))(values),
            texttemplate="%{text}",
            hovertemplate="행=%{y}<br>열=%{x}<br>값=%{z:.4f}<extra></extra>",
        )
    )
    interactive.update_layout(
        title={"text": f"{title}<br><sup>{subtitle}</sup>", "x": 0.5},
        width=max(1000, 105 * len(column_labels) + 420),
        height=max(650, 45 * len(row_labels) + 260),
        xaxis={"tickangle": -35},
        yaxis={"autorange": "reversed"},
        paper_bgcolor="white",
        plot_bgcolor="#E5E7EB",
        margin={"l": 230, "r": 100, "t": 110, "b": 130},
        font={"family": "Malgun Gothic, NanumGothic, sans-serif"},
    )
    return figure, interactive


def _grouped_bar_figures(
    frame: pd.DataFrame,
    *,
    category: str,
    series: str,
    value: str,
    title: str,
    subtitle: str,
    y_title: str,
    y_range: tuple[float, float] | None = None,
) -> tuple[Any, Any]:
    aggregated = (
        frame.groupby([category, series], observed=True, sort=False)[value]
        .median()
        .reset_index()
    )
    categories = list(dict.fromkeys(aggregated[category].astype(str)))
    series_values = list(dict.fromkeys(aggregated[series].astype(str)))
    x = np.arange(len(categories), dtype=float)
    width = 0.80 / max(len(series_values), 1)
    palette = ("#2563EB", "#D97706", "#059669", "#7C3AED", "#DC2626")
    mpl, plt = _matplotlib_modules()
    font = _font_family()
    with mpl.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [font, "DejaVu Sans"],
            "axes.unicode_minus": False,
            "svg.fonttype": "path",
        }
    ):
        figure, axis = plt.subplots(
            figsize=(max(10.0, 1.35 * len(categories) + 4.0), 6.2),
            constrained_layout=True,
        )
        for index, series_value in enumerate(series_values):
            subset = aggregated.loc[aggregated[series].astype(str).eq(series_value)]
            mapping = dict(zip(subset[category].astype(str), subset[value].astype(float)))
            heights = np.array([mapping.get(item, np.nan) for item in categories])
            offset = (index - (len(series_values) - 1) / 2) * width
            bars = axis.bar(
                x + offset,
                heights,
                width=width * 0.94,
                label=series_value,
                color=palette[index % len(palette)],
            )
            axis.bar_label(bars, fmt="%.3f", fontsize=7, padding=2)
        axis.set_xticks(x, categories, rotation=25, ha="right")
        axis.set_ylabel(y_title)
        if y_range is not None:
            axis.set_ylim(*y_range)
        axis.grid(axis="y", alpha=0.25)
        axis.legend(frameon=False, ncols=min(3, len(series_values)))
        axis.set_title(title, fontsize=14, fontweight="bold", pad=22)
        axis.text(
            0.5,
            1.01,
            subtitle,
            transform=axis.transAxes,
            ha="center",
            fontsize=8.5,
            color="#4B5563",
        )

    go = _plotly_go()
    interactive = go.Figure()
    for index, series_value in enumerate(series_values):
        subset = aggregated.loc[aggregated[series].astype(str).eq(series_value)]
        mapping = dict(zip(subset[category].astype(str), subset[value].astype(float)))
        interactive.add_bar(
            name=series_value,
            x=categories,
            y=[mapping.get(item, None) for item in categories],
            marker_color=palette[index % len(palette)],
            text=[
                f"{mapping[item]:.3f}" if item in mapping else ""
                for item in categories
            ],
            textposition="outside",
            hovertemplate="%{x}<br>%{fullData.name}: %{y:.4f}<extra></extra>",
        )
    interactive.update_layout(
        barmode="group",
        title={"text": f"{title}<br><sup>{subtitle}</sup>", "x": 0.5},
        width=max(1000, 150 * len(categories) + 350),
        height=650,
        yaxis={"title": y_title, "range": list(y_range) if y_range else None},
        xaxis={"tickangle": -25},
        paper_bgcolor="white",
        plot_bgcolor="white",
        font={"family": "Malgun Gothic, NanumGothic, sans-serif"},
        margin={"l": 90, "r": 60, "t": 110, "b": 130},
    )
    return figure, interactive


def _make_figure(figure_id: str, frame: pd.DataFrame) -> tuple[Any, Any]:
    bundle_values = sorted(frame["bundle_id"].astype(str).unique(), key=_bundle_sort_key)
    bundle_labels = [_bundle_label(value) for value in bundle_values]

    if figure_id == "bundle_year_season_rank_heatmap":
        sources = (
            ("year2022_rank", "2022 순위"),
            ("year2023_rank", "2023 순위"),
            ("pooled_rank", "통합 순위"),
            ("cross_year_consensus_rank", "연도합의 순위"),
        )
        rows: list[str] = []
        values: list[list[float]] = []
        for bundle in bundle_values:
            subset = frame.loc[frame["bundle_id"].astype(str).eq(bundle)].set_index("season")
            for column, source_label in sources:
                rows.append(f"{_bundle_label(bundle)} · {source_label}")
                values.append([float(subset.loc[season, column]) for season in SEASON_ORDER])
        array = np.asarray(values)
        return _heatmap_figures(
            array,
            rows,
            [SEASON_LABELS_KO[season] for season in SEASON_ORDER],
            title="번들·연도별 계절 순위",
            subtitle="1위가 가장 진하게 표시됨 · 2022/2023/통합/설명가능한 합의순위 비교",
            colour_map="viridis_r",
            plotly_scale="Viridis_r",
            vmin=1.0,
            vmax=max(4.0, float(np.nanmax(array))),
            value_format=".0f",
            colorbar_title="계절 순위",
        )

    if figure_id == "cross_year_transfer_matrix":
        metric_labels = (
            ("top1_agreement", "Top1 일치"),
            ("train_primary_in_holdout_top2", "Train Top1∈Holdout Top2"),
            ("pair_top2_coverage", "Train pair→Holdout Top2"),
            ("season_rank_spearman", "Spearman ρ"),
            ("season_rank_kendall", "Kendall τ"),
        )
        aggregate = frame.groupby(["bundle_id", "direction"], observed=True, sort=False)[
            [column for column, _ in metric_labels]
        ].mean()
        rows = []
        values = []
        for bundle in bundle_values:
            directions = sorted(
                frame.loc[frame["bundle_id"].astype(str).eq(bundle), "direction"].unique()
            )
            for direction in directions:
                rows.append(f"{_bundle_label(bundle)} · {direction}")
                values.append(
                    [
                        float(aggregate.loc[(bundle, direction), column])
                        for column, _ in metric_labels
                    ]
                )
        return _heatmap_figures(
            np.asarray(values),
            rows,
            [label for _, label in metric_labels],
            title="2022↔2023 교차연도 전이",
            subtitle="추천 생성 연도와 평가 연도를 분리한 bundle별 평균 · 독립단위 broadcast 금지",
            colour_map="RdYlGn",
            plotly_scale="RdYlGn",
            vmin=-1.0,
            vmax=1.0,
            colorbar_title="일치·순위 상관",
        )

    if figure_id == "bundle_season_margin_chart":
        long = frame.melt(
            id_vars=["bundle_id", "direction"],
            value_vars=["train_top1_margin", "holdout_top1_margin"],
            var_name="margin_scope",
            value_name="margin",
        )
        long["bundle_label"] = long["bundle_id"].astype(str).map(_bundle_label)
        long["series"] = long["direction"].astype(str) + " · " + long["margin_scope"].map(
            {"train_top1_margin": "학습", "holdout_top1_margin": "평가"}
        )
        return _grouped_bar_figures(
            long,
            category="bundle_label",
            series="series",
            value="margin",
            title="번들별 계절 Top1–Top2 margin",
            subtitle="작은 margin은 강한 단일계절 추천 근거가 아님",
            y_title="Top1–Top2 margin",
        )

    if figure_id == "heldout_regret_chart":
        prepared = frame.copy()
        prepared["bundle_label"] = prepared["bundle_id"].astype(str).map(_bundle_label)
        return _grouped_bar_figures(
            prepared,
            category="bundle_label",
            series="direction",
            value="heldout_regret_normalized",
            title="교차연도 평가 regret",
            subtitle="regret = 평가연도 최적점수 - 학습연도 선택계절의 평가점수 · 낮을수록 안정",
            y_title="정규화 held-out regret",
        )

    if figure_id == "bootstrap_season_selection_probability":
        prepared = frame.copy()
        prepared["bundle_label"] = prepared["bundle_id"].astype(str).map(_bundle_label)
        prepared["season_label"] = prepared["season"].astype(str).map(SEASON_LABELS_KO)
        return _grouped_bar_figures(
            prepared,
            category="bundle_label",
            series="season_label",
            value="primary_probability",
            title="시군 block-bootstrap 계절 선택확률",
            subtitle="11개 시군의 전체 월·진료과 sequence를 함께 재표집 · 월 독립 재표집 금지",
            y_title="Primary 계절 선택확률",
            y_range=(0.0, 1.08),
        )

    if figure_id == "leave_one_month_out_stability":
        rows = []
        values = []
        for bundle in bundle_values:
            for year in (2022, 2023):
                subset = frame.loc[
                    frame["bundle_id"].astype(str).eq(bundle) & frame["year"].eq(year)
                ].set_index("removed_month")
                rows.append(f"{_bundle_label(bundle)} · {year}")
                values.append(
                    [
                        float(subset.loc[month, "primary_retained"])
                        for month in range(1, 13)
                    ]
                )
        return _heatmap_figures(
            np.asarray(values),
            rows,
            [f"{month}월" for month in range(1, 13)],
            title="Leave-One-Month-Out 안정성",
            subtitle="각 연도에서 한 달씩 제거했을 때 기존 primary 계절 유지 여부 (1=유지)",
            colour_map="GnBu",
            plotly_scale="GnBu",
            vmin=0.0,
            vmax=1.0,
            value_format=".0f",
            colorbar_title="Primary 유지",
        )

    if figure_id == "leave_one_sigungu_out_stability":
        sigungu = sorted(frame["removed_sigungu"].astype(str).unique())
        values = []
        for bundle in bundle_values:
            subset = frame.loc[
                frame["bundle_id"].astype(str).eq(bundle)
            ].set_index("removed_sigungu")
            values.append([float(subset.loc[item, "rank_spearman"]) for item in sigungu])
        return _heatmap_figures(
            np.asarray(values),
            bundle_labels,
            sigungu,
            title="Leave-One-Sigungu-Out 안정성",
            subtitle="제외한 시군과 평가대상을 명시한 계절순위 Spearman ρ · 11개 정책 시군",
            colour_map="RdYlGn",
            plotly_scale="RdYlGn",
            vmin=-1.0,
            vmax=1.0,
            colorbar_title="순위 Spearman ρ",
        )

    if figure_id == "observed_vs_permutation_null":
        # The five null metrics have different raw scales.  Plot their signed
        # standardized strengths instead of putting incomparable raw values on
        # one y-axis.  Raw observed/null values remain visible in every static
        # cell and in the richer interactive hover contract.
        metric_values = list(dict.fromkeys(frame["metric"].astype(str)))
        metric_values = sorted(
            metric_values,
            key=lambda value: (
                list(NULL_METRIC_LABELS_KO).index(value)
                if value in NULL_METRIC_LABELS_KO
                else len(NULL_METRIC_LABELS_KO),
                value,
            ),
        )
        indexed = frame.set_index(["bundle_id", "metric"])
        strength = np.asarray(
            [
                [
                    float(indexed.loc[(bundle, metric), "standardized_strength"])
                    for metric in metric_values
                ]
                for bundle in bundle_values
            ],
            dtype=float,
        )
        observed = np.asarray(
            [
                [float(indexed.loc[(bundle, metric), "observed"]) for metric in metric_values]
                for bundle in bundle_values
            ],
            dtype=float,
        )
        null_mean = np.asarray(
            [
                [float(indexed.loc[(bundle, metric), "null_mean"]) for metric in metric_values]
                for bundle in bundle_values
            ],
            dtype=float,
        )
        null_sd = np.asarray(
            [
                [float(indexed.loc[(bundle, metric), "null_sd"]) for metric in metric_values]
                for bundle in bundle_values
            ],
            dtype=float,
        )
        p_value = np.asarray(
            [
                [float(indexed.loc[(bundle, metric), "p_value_upper"]) for metric in metric_values]
                for bundle in bundle_values
            ],
            dtype=float,
        )
        bound = max(2.0, float(np.nanmax(np.abs(strength))))
        metric_labels = [NULL_METRIC_LABELS_KO.get(value, value) for value in metric_values]
        mpl, plt = _matplotlib_modules()
        font = _font_family()
        with mpl.rc_context(
            {
                "font.family": "sans-serif",
                "font.sans-serif": [font, "DejaVu Sans"],
                "axes.unicode_minus": False,
                "svg.fonttype": "path",
            }
        ):
            figure, axis = plt.subplots(
                figsize=(max(12.0, 2.15 * len(metric_values) + 3.5), 7.5),
                constrained_layout=True,
            )
            image = axis.imshow(
                strength,
                cmap="RdBu_r",
                vmin=-bound,
                vmax=bound,
                aspect="auto",
                interpolation="nearest",
            )
            axis.set_xticks(
                np.arange(len(metric_labels)), metric_labels, rotation=24, ha="right"
            )
            axis.set_yticks(np.arange(len(bundle_labels)), bundle_labels)
            axis.set_title(
                "관측 계절신호와 month-label permutation null",
                fontsize=14,
                fontweight="bold",
                pad=22,
            )
            axis.text(
                0.5,
                1.01,
                "서로 다른 원척도는 혼합하지 않고 표준화 강도(z)를 비교 · p는 상측 경험 p값",
                transform=axis.transAxes,
                ha="center",
                fontsize=8.5,
                color="#4B5563",
            )
            colour_bar = figure.colorbar(image, ax=axis, fraction=0.035, pad=0.03)
            colour_bar.set_label("관측 - null 표준화 강도 (z)")
            for tick_label in colour_bar.ax.get_yticklabels():
                tick_label.set_fontfamily("DejaVu Sans")
            for row in range(strength.shape[0]):
                for column in range(strength.shape[1]):
                    text = (
                        f"obs {observed[row, column]:.3g}\n"
                        f"null {null_mean[row, column]:.3g}±{null_sd[row, column]:.2g}\n"
                        f"z {strength[row, column]:.2f} · p {p_value[row, column]:.3f}"
                    )
                    axis.text(
                        column,
                        row,
                        text,
                        ha="center",
                        va="center",
                        fontsize=6.8,
                        color=(
                            "white"
                            if abs(strength[row, column]) / bound >= 0.52
                            else "#111827"
                        ),
                    )

        custom = np.stack(
            [observed, null_mean, null_sd, p_value, strength], axis=-1
        )
        display_text = np.asarray(
            [
                [
                    f"z={strength[row, column]:.2f}<br>p={p_value[row, column]:.3f}"
                    for column in range(strength.shape[1])
                ]
                for row in range(strength.shape[0])
            ],
            dtype=object,
        )
        go = _plotly_go()
        interactive = go.Figure(
            go.Heatmap(
                z=strength,
                x=metric_labels,
                y=bundle_labels,
                zmin=-bound,
                zmax=bound,
                zmid=0.0,
                colorscale="RdBu",
                reversescale=True,
                customdata=custom,
                text=display_text,
                texttemplate="%{text}",
                colorbar={"title": "표준화 강도 z"},
                hovertemplate=(
                    "번들=%{y}<br>지표=%{x}"
                    "<br>observed=%{customdata[0]:.6g}"
                    "<br>null_mean=%{customdata[1]:.6g}"
                    "<br>null_sd=%{customdata[2]:.6g}"
                    "<br>p_value_upper=%{customdata[3]:.6g}"
                    "<br>standardized_strength=%{customdata[4]:.6g}<extra></extra>"
                ),
            )
        )
        interactive.update_layout(
            title={
                "text": (
                    "관측 계절신호와 month-label permutation null"
                    "<br><sup>서로 다른 원척도는 혼합하지 않고 표준화 강도(z)를 비교 · "
                    "hover에 관측/null/p 근거 보존</sup>"
                ),
                "x": 0.5,
            },
            width=max(1100, 175 * len(metric_values) + 420),
            height=max(650, 85 * len(bundle_labels) + 260),
            xaxis={"tickangle": -24},
            yaxis={"autorange": "reversed"},
            paper_bgcolor="white",
            plot_bgcolor="#E5E7EB",
            margin={"l": 250, "r": 110, "t": 115, "b": 150},
            font={"family": "Malgun Gothic, NanumGothic, sans-serif"},
        )
        return figure, interactive

    if figure_id == "temporal_confidence_summary":
        ordered = frame.copy()
        ordered["bundle_text"] = ordered["bundle_id"].astype(str).map(_bundle_label)
        confidence_score = ordered["temporal_confidence"].map(
            {"LOW": 1.0, "MODERATE": 2.0, "HIGH": 3.0}
        )
        values = confidence_score.to_numpy(dtype=float).reshape(-1, 1)
        annotations = [
            (
                f"{CONFIDENCE_LABELS_KO[confidence]} ({confidence})\n"
                f"{RELEASE_TYPE_LABELS_KO[release_type]}"
            )
            for confidence, release_type in zip(
                ordered["temporal_confidence"].astype(str),
                ordered["release_type"].astype(str),
                strict=True,
            )
        ]
        mpl, plt = _matplotlib_modules()
        font = _font_family()
        from matplotlib.colors import ListedColormap

        with mpl.rc_context(
            {
                "font.family": "sans-serif",
                "font.sans-serif": [font, "DejaVu Sans"],
                "axes.unicode_minus": False,
                "svg.fonttype": "path",
            }
        ):
            figure, axis = plt.subplots(
                figsize=(10.0, max(5.0, 0.75 * len(ordered) + 2.8)),
                constrained_layout=True,
            )
            image = axis.imshow(
                values,
                cmap=ListedColormap(["#D1D5DB", "#FBBF24", "#059669"]),
                vmin=0.5,
                vmax=3.5,
                aspect="auto",
            )
            axis.set_xticks([0], ["추천 신뢰도 / 공개 유형"])
            axis.set_yticks(np.arange(len(ordered)), ordered["bundle_text"])
            for index, text in enumerate(annotations):
                axis.text(0, index, text, ha="center", va="center", fontsize=9, color="#111827")
            axis.set_title("시간 추천 신뢰도 요약", fontsize=14, fontweight="bold", pad=22)
            axis.text(
                0.5,
                1.01,
                "단일 계절·안정적 계절 쌍·계절 우선순위 확정 없음의 증거수준을 번들별로 공개",
                transform=axis.transAxes,
                ha="center",
                fontsize=8.5,
                color="#4B5563",
            )
            colour_bar = figure.colorbar(image, ax=axis, fraction=0.035, pad=0.05, ticks=[1, 2, 3])
            colour_bar.ax.set_yticklabels(["낮음", "중간", "높음"])
            for tick_label in colour_bar.ax.get_yticklabels():
                tick_label.set_fontfamily(font)

        go = _plotly_go()
        interactive = go.Figure(
            go.Heatmap(
                z=values,
                x=["추천 신뢰도 / 공개 유형"],
                y=ordered["bundle_text"].tolist(),
                zmin=0.5,
                zmax=3.5,
                colorscale=[
                    [0.0, "#D1D5DB"],
                    [0.333, "#D1D5DB"],
                    [0.334, "#FBBF24"],
                    [0.666, "#FBBF24"],
                    [0.667, "#059669"],
                    [1.0, "#059669"],
                ],
                text=np.asarray(annotations, dtype=object).reshape(-1, 1),
                texttemplate="%{text}",
                customdata=np.asarray(
                    [
                        [confidence, release_type]
                        for confidence, release_type in zip(
                            ordered["temporal_confidence"].astype(str),
                            ordered["release_type"].astype(str),
                            strict=True,
                        )
                    ],
                    dtype=object,
                ).reshape(-1, 1, 2),
                hovertemplate=(
                    "%{y}<br>%{text}<br>canonical confidence=%{customdata[0]}"
                    "<br>canonical release_type=%{customdata[1]}<extra></extra>"
                ),
                colorbar={
                    "tickvals": [1, 2, 3],
                    "ticktext": ["낮음", "중간", "높음"],
                },
            )
        )
        interactive.update_layout(
            title={
                "text": (
                    "시간 추천 신뢰도 요약<br><sup>단일 계절·안정적 계절 쌍·"
                    "계절 우선순위 확정 없음의 증거수준을 번들별로 공개</sup>"
                ),
                "x": 0.5,
            },
            width=1050,
            height=max(600, 80 * len(ordered) + 260),
            yaxis={"autorange": "reversed"},
            font={"family": "Malgun Gothic, NanumGothic, sans-serif"},
            paper_bgcolor="white",
            margin={"l": 250, "r": 100, "t": 110, "b": 100},
        )
        return figure, interactive

    raise KeyError(f"unsupported figure_id: {figure_id}")


def _png_metadata(path: Path) -> tuple[int, int, float | None, float | None]:
    data = path.read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n" or len(data) < 33:
        raise ValueError(f"not a valid PNG: {path}")
    width, height = struct.unpack(">II", data[16:24])
    position = 8
    dpi_x: float | None = None
    dpi_y: float | None = None
    while position + 12 <= len(data):
        length = struct.unpack(">I", data[position : position + 4])[0]
        chunk_type = data[position + 4 : position + 8]
        payload = data[position + 8 : position + 8 + length]
        if chunk_type == b"pHYs" and len(payload) == 9:
            x_ppm, y_ppm, unit = struct.unpack(">IIB", payload)
            if unit == 1:
                dpi_x = x_ppm * 0.0254
                dpi_y = y_ppm * 0.0254
            break
        position += length + 12
    return int(width), int(height), dpi_x, dpi_y


def _artifact_rows(
    figures: Mapping[str, Mapping[str, str]],
    normalized: Mapping[str, pd.DataFrame],
    output_dir: Path,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for figure_id in FIGURE_IDS:
        input_digest = _frame_sha256(normalized[figure_id])
        for artifact_format in ("png", "svg", "html"):
            path = Path(figures[figure_id][artifact_format])
            size = path.stat().st_size
            width = height = None
            dpi_x = dpi_y = None
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
                    "size_bytes": int(size),
                    "sha256": sha256_file(path),
                    "input_table_sha256": input_digest,
                    "width_px": width,
                    "height_px": height,
                    "dpi_x": dpi_x,
                    "dpi_y": dpi_y,
                    "external_script_count": external_scripts,
                    "self_contained_html": self_contained,
                }
            )
    return pd.DataFrame(rows)


def validate_figure_artifact_manifest(manifest: pd.DataFrame) -> None:
    """Fail closed unless every required figure has PNG/SVG/self-contained HTML."""

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
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise KeyError(f"figure artifact manifest missing columns: {missing}")
    expected = {(figure_id, fmt) for figure_id in FIGURE_IDS for fmt in ("png", "svg", "html")}
    observed = set(zip(manifest["figure_id"], manifest["format"]))
    if observed != expected or len(manifest) != len(expected):
        raise ValueError("figure artifact manifest must contain exactly 9×3 artifacts")
    if (pd.to_numeric(manifest["size_bytes"], errors="coerce") <= 100).any():
        raise ValueError("all figure artifacts must be non-empty")
    if not manifest["sha256"].astype(str).str.fullmatch(r"[0-9a-f]{64}").all():
        raise ValueError("invalid artifact SHA256")
    png = manifest.loc[manifest["format"].eq("png")]
    if (pd.to_numeric(png["dpi_x"], errors="coerce") < 299.0).any() or (
        pd.to_numeric(png["dpi_y"], errors="coerce") < 299.0
    ).any():
        raise ValueError("all PNG figures must carry at least 300-dpi metadata")
    if (pd.to_numeric(png["width_px"], errors="coerce") < 1800).any() or (
        pd.to_numeric(png["height_px"], errors="coerce") < 1000
    ).any():
        raise ValueError("PNG figure resolution is below the publication floor")
    html_rows = manifest.loc[manifest["format"].eq("html")]
    if not html_rows["external_script_count"].eq(0).all() or not html_rows[
        "self_contained_html"
    ].eq(True).all():
        raise ValueError("HTML figures must be self-contained without external scripts")


def render_stage2b_hardening_figures(
    tables: Mapping[str, pd.DataFrame],
    output_dir: str | Path,
    *,
    column_maps: Mapping[str, Mapping[str, str]] | None = None,
    dpi: int = 300,
) -> Stage2BHardeningReportingResult:
    """Validate and render the nine preregistered Stage 2B hardening figures.

    ``column_maps`` is keyed by figure id and maps source column names to the
    canonical names in :data:`FIGURE_REQUIRED_COLUMNS`.  Unresolved aliases,
    duplicate keys, missing LOMO/LOSO removals, and invalid probabilities fail
    before any figures are written.
    """

    if int(dpi) < 300:
        raise ValueError("Stage 2B publication figures require dpi >= 300")
    missing_tables = [figure_id for figure_id in FIGURE_IDS if figure_id not in tables]
    if missing_tables:
        raise KeyError(f"missing Stage 2B figure tables: {missing_tables}")
    normalized: dict[str, pd.DataFrame] = {}
    for figure_id in FIGURE_IDS:
        raw = _normalize_columns(
            figure_id,
            tables[figure_id],
            (column_maps or {}).get(figure_id),
        )
        normalized[figure_id] = _normalize_figure_table(figure_id, raw)
    bundle_sets = {
        figure_id: set(frame["bundle_id"].astype(str))
        for figure_id, frame in normalized.items()
    }
    reference = bundle_sets[FIGURE_IDS[0]]
    inconsistent = {
        figure_id: sorted(values)
        for figure_id, values in bundle_sets.items()
        if values != reference
    }
    if inconsistent:
        raise ValueError(
            "all nine figure tables must cover the same bundles; "
            f"reference={sorted(reference)}, inconsistent={inconsistent}"
        )

    destination = Path(output_dir)
    figures: dict[str, Mapping[str, str]] = {}
    for figure_id in FIGURE_IDS:
        static, interactive = _make_figure(figure_id, normalized[figure_id])
        figures[figure_id] = _save_figure_triplet(
            figure_id,
            static,
            interactive,
            destination,
            dpi=int(dpi),
        )
    manifest = _artifact_rows(figures, normalized, destination)
    validate_figure_artifact_manifest(manifest)
    manifest_path = destination / "STAGE2B_HARDENING_FIGURE_ARTIFACTS.csv"
    _atomic_write_csv(manifest, manifest_path)
    return Stage2BHardeningReportingResult(
        figures=figures,
        artifact_manifest=manifest,
        artifact_manifest_path=str(manifest_path),
        normalized_tables=normalized,
    )


EVIDENCE_REQUIRED_COLUMNS: tuple[str, ...] = (
    "bundle_id",
    "2022_top1",
    "2022_top2",
    "2023_top1",
    "2023_top2",
    "pooled_top1",
    "pooled_top2",
    "cross_year_consistency",
    "heldout_regret",
    "bootstrap_primary_probability",
    "lomo_stability",
    "loso_stability",
    "null_strength",
    "release_type",
    "confidence",
    "final_primary",
    "final_fallback",
    "reason",
)


def validate_temporal_evidence_summary(
    evidence: pd.DataFrame,
    *,
    expected_bundles: Sequence[str] | None = BUNDLE_ORDER,
) -> pd.DataFrame:
    """Validate the one-row-per-bundle temporal evidence explanation table."""

    if not isinstance(evidence, pd.DataFrame) or evidence.empty:
        raise ValueError("temporal evidence summary must be a non-empty DataFrame")
    frame = evidence.copy()
    if "bundle_id" not in frame.columns and "bundle" in frame.columns:
        frame = frame.rename(columns={"bundle": "bundle_id"})
    if "confidence" not in frame.columns and "temporal_confidence" in frame.columns:
        # ``confidence`` is the compact evidence-table contract while
        # ``temporal_confidence`` is the public Stage 2B release contract.
        # Preserve the latter so validation is non-destructive for callers
        # that immediately build the 765-row delivery/interface tables.
        frame["confidence"] = frame["temporal_confidence"]
    if "temporal_confidence" not in frame.columns and "confidence" in frame.columns:
        frame["temporal_confidence"] = frame["confidence"]
    missing = sorted(set(EVIDENCE_REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise KeyError(f"temporal evidence summary missing columns: {missing}")
    frame["bundle_id"] = frame["bundle_id"].astype(str).str.strip()
    _assert_unique(frame, ["bundle_id"], "temporal_bundle_evidence_summary")
    if expected_bundles is not None and set(frame["bundle_id"]) != set(expected_bundles):
        raise ValueError(
            "temporal evidence bundles do not match the expected contract: "
            f"observed={sorted(frame['bundle_id'])}, expected={sorted(expected_bundles)}"
        )
    frame["release_type"] = frame["release_type"].astype(str).str.upper().str.strip()
    frame["confidence"] = frame["confidence"].astype(str).str.upper().str.strip()
    invalid_release = sorted(set(frame["release_type"]) - VALID_RELEASE_TYPES)
    invalid_confidence = sorted(set(frame["confidence"]) - VALID_CONFIDENCE)
    if invalid_release:
        raise ValueError(f"invalid release_type values: {invalid_release}")
    if invalid_confidence:
        raise ValueError(f"invalid confidence values: {invalid_confidence}")
    probability_columns = (
        "cross_year_consistency",
        "bootstrap_primary_probability",
        "lomo_stability",
        "loso_stability",
    )
    _probability(frame, probability_columns, "temporal_bundle_evidence_summary")
    _numeric(frame, ["heldout_regret", "null_strength"], "temporal_bundle_evidence_summary")
    if (frame["heldout_regret"] < 0).any():
        raise ValueError("heldout_regret must be non-negative")
    for column in ("2022_top1", "2023_top1", "pooled_top1", "final_primary", "final_fallback"):
        frame[column] = frame[column].astype("string").str.lower().str.strip()
    for column in ("2022_top2", "2023_top2", "pooled_top2"):
        if frame[column].isna().any() or frame[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"{column} must be populated")
    abstain = frame["release_type"].eq("NO_STRONG_PREFERENCE")
    if frame.loc[abstain, "final_primary"].notna().any() or frame.loc[
        abstain, "final_fallback"
    ].notna().any():
        raise ValueError("NO_STRONG_PREFERENCE must not force primary/fallback seasons")
    actionable = ~abstain
    if frame.loc[actionable, ["final_primary", "final_fallback"]].isna().any().any():
        raise ValueError("actionable releases require primary and fallback seasons")
    if (
        frame.loc[actionable, "final_primary"].to_numpy()
        == frame.loc[actionable, "final_fallback"].to_numpy()
    ).any():
        raise ValueError("final_fallback must differ from final_primary")
    seasons = set(SEASON_ORDER)
    for column in ("2022_top1", "2023_top1", "pooled_top1"):
        if not set(frame[column].dropna()).issubset(seasons):
            raise ValueError(f"{column} contains an unsupported season")
    for column in ("final_primary", "final_fallback"):
        if not set(frame.loc[actionable, column].dropna()).issubset(seasons):
            raise ValueError(f"{column} contains an unsupported season")
    if frame["reason"].isna().any() or frame["reason"].astype(str).str.strip().eq("").any():
        raise ValueError("every evidence row requires a non-blank reason")
    order = sorted(frame["bundle_id"].unique(), key=_bundle_sort_key)
    frame["bundle_id"] = pd.Categorical(frame["bundle_id"], categories=order, ordered=True)
    return frame.sort_values("bundle_id").reset_index(drop=True)


GATE_REQUIRED_COLUMNS: tuple[str, ...] = (
    "gate_id",
    "severity",
    "passed",
    "observed",
    "comparison",
    "threshold",
    "threshold_source",
    "note",
)


def build_stage2b_quality_gate(
    records: pd.DataFrame | Sequence[Mapping[str, Any]],
    *,
    final_decision: str | None = None,
) -> Stage2BQualityGateResult:
    """Build a fail-closed gate ledger while preserving preregistration provenance."""

    frame = records.copy() if isinstance(records, pd.DataFrame) else pd.DataFrame(list(records))
    if frame.empty:
        raise ValueError("quality-gate records cannot be empty")
    missing = sorted(set(GATE_REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise KeyError(f"quality-gate records missing columns: {missing}")
    frame = frame.copy()
    frame["gate_id"] = frame["gate_id"].astype(str).str.strip()
    if frame["gate_id"].eq("").any() or frame["gate_id"].duplicated().any():
        raise ValueError("quality gate ids must be non-blank and unique")
    frame["severity"] = frame["severity"].astype(str).str.lower().str.strip()
    invalid = sorted(set(frame["severity"]) - VALID_SEVERITIES)
    if invalid:
        raise ValueError(f"invalid quality-gate severities: {invalid}")
    frame["passed"] = _bool_numeric(frame["passed"], "quality_gate.passed").astype(bool)
    for column in ("comparison", "threshold_source", "note"):
        if frame[column].isna().any() or frame[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"quality_gate.{column} must be populated")
        frame[column] = frame[column].astype(str).str.strip()
    if not frame["comparison"].isin({">=", ">", "<=", "<", "==", "!=", "contract"}).all():
        raise ValueError("quality_gate.comparison contains an unsupported operator")
    if not frame["threshold_source"].str.contains(
        r"config|prompt|contract|preregister", case=False, regex=True
    ).all():
        raise ValueError(
            "threshold_source must identify a config, prompt, contract, or preregistration"
        )
    if not frame["severity"].eq("hard").any():
        raise ValueError("quality gate requires at least one hard check")
    severity_order = pd.CategoricalDtype(
        categories=["hard", "advisory", "diagnostic"], ordered=True
    )
    frame["severity"] = frame["severity"].astype(severity_order)
    frame["status"] = np.where(frame["passed"], "PASS", "FAIL")
    frame = frame.sort_values(["severity", "gate_id"], kind="stable").reset_index(drop=True)
    hard = frame["severity"].eq("hard")
    hard_failed = int((hard & ~frame["passed"]).sum())
    if final_decision is not None:
        final_decision = str(final_decision).upper().strip()
        if final_decision not in VALID_FINAL_DECISIONS:
            raise ValueError(f"invalid Stage 2B final decision: {final_decision}")
        if hard_failed and final_decision != "FAIL_TEMPORAL_RELEASE":
            raise ValueError("a release decision cannot pass while a hard gate fails")
        if not hard_failed and final_decision == "FAIL_TEMPORAL_RELEASE":
            # A scientifically weak but structurally valid model may be advisory;
            # outright failure must have an explicit hard-gate basis.
            raise ValueError("FAIL_TEMPORAL_RELEASE requires at least one failed hard gate")
    return Stage2BQualityGateResult(
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
        final_decision=final_decision,
    )


def validate_stage2b_quality_gate(
    gates: pd.DataFrame,
    *,
    final_decision: str | None = None,
) -> Stage2BQualityGateResult:
    """Validate a previously materialized quality-gate CSV."""

    frame = gates.drop(columns=["status"], errors="ignore")
    return build_stage2b_quality_gate(frame, final_decision=final_decision)


def build_artifact_inventory(
    roots: str | Path | Sequence[str | Path],
    *,
    base_dir: str | Path | None = None,
    exclude: Sequence[str | Path] = (),
) -> pd.DataFrame:
    """Hash files below one or more roots in deterministic path order."""

    root_list = [roots] if isinstance(roots, (str, Path)) else list(roots)
    if not root_list:
        raise ValueError("artifact inventory requires at least one root")
    base = Path(base_dir).resolve() if base_dir is not None else None
    excluded = {Path(path).resolve() for path in exclude}
    paths: set[Path] = set()
    for raw_root in root_list:
        root = Path(raw_root).resolve()
        if root.is_file():
            paths.add(root)
        elif root.is_dir():
            paths.update(path.resolve() for path in root.rglob("*") if path.is_file())
        else:
            raise FileNotFoundError(root)
    rows: list[dict[str, Any]] = []
    for path in sorted(paths - excluded, key=lambda item: item.as_posix()):
        if base is not None:
            try:
                relative = path.relative_to(base).as_posix()
            except ValueError as exc:
                raise ValueError(f"artifact is outside base_dir: {path}") from exc
        else:
            relative = path.name
        rows.append(
            {
                "relative_path": relative,
                "size_bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )
    if not rows:
        raise ValueError("artifact inventory is empty")
    inventory = pd.DataFrame(rows)
    if inventory["relative_path"].duplicated().any():
        raise ValueError("artifact relative paths are not unique; provide base_dir")
    inventory["content_alias_count"] = inventory.groupby("sha256")[
        "relative_path"
    ].transform("size")
    canonical = (
        inventory.sort_values(["sha256", "relative_path"], kind="stable")
        .groupby("sha256", sort=False)["relative_path"]
        .first()
    )
    inventory["canonical_content_path"] = inventory["sha256"].map(canonical)
    inventory["is_content_alias"] = (
        inventory["relative_path"] != inventory["canonical_content_path"]
    )
    return inventory.sort_values("relative_path", kind="stable").reset_index(drop=True)


def validate_artifact_inventory(
    inventory: pd.DataFrame,
    base_dir: str | Path,
    *,
    raise_on_error: bool = True,
) -> ArtifactInventoryValidation:
    """Recompute sizes and hashes and report every inventory mismatch."""

    required = {"relative_path", "size_bytes", "sha256"}
    missing = sorted(required - set(inventory.columns))
    if missing:
        raise KeyError(f"artifact inventory missing columns: {missing}")
    if inventory.empty or inventory["relative_path"].duplicated().any():
        raise ValueError("artifact inventory must be non-empty with unique paths")
    base = Path(base_dir).resolve()
    rows: list[dict[str, Any]] = []
    for record in inventory.itertuples(index=False):
        relative = str(record.relative_path)
        path = (base / relative).resolve()
        try:
            path.relative_to(base)
        except ValueError as exc:
            raise ValueError(f"inventory path escapes base_dir: {relative}") from exc
        exists = path.is_file()
        actual_size = int(path.stat().st_size) if exists else None
        actual_sha = sha256_file(path) if exists else None
        expected_size = int(record.size_bytes)
        expected_sha = str(record.sha256)
        rows.append(
            {
                "relative_path": relative,
                "exists": exists,
                "nonempty": bool(exists and actual_size and actual_size > 0),
                "size_matches": bool(exists and actual_size == expected_size),
                "sha_matches": bool(exists and actual_sha == expected_sha),
                "expected_size_bytes": expected_size,
                "actual_size_bytes": actual_size,
                "expected_sha256": expected_sha,
                "actual_sha256": actual_sha,
            }
        )
    detail = pd.DataFrame(rows)
    result = ArtifactInventoryValidation(
        valid=bool(
            detail[["exists", "nonempty", "size_matches", "sha_matches"]]
            .all(axis=1)
            .all()
        ),
        checked_count=int(len(detail)),
        missing_count=int((~detail["exists"]).sum()),
        size_mismatch_count=int((detail["exists"] & ~detail["size_matches"]).sum()),
        sha_mismatch_count=int((detail["exists"] & ~detail["sha_matches"]).sum()),
        empty_count=int((detail["exists"] & ~detail["nonempty"]).sum()),
        detail=detail,
    )
    if raise_on_error and not result.valid:
        failures = detail.loc[
            ~detail[["exists", "nonempty", "size_matches", "sha_matches"]].all(axis=1),
            "relative_path",
        ].tolist()
        raise ValueError(f"artifact inventory validation failed: {failures[:10]}")
    return result


__all__ = [
    "FIGURE_IDS",
    "FIGURE_REQUIRED_COLUMNS",
    "COLUMN_ALIASES",
    "EVIDENCE_REQUIRED_COLUMNS",
    "GATE_REQUIRED_COLUMNS",
    "Stage2BHardeningReportingResult",
    "Stage2BQualityGateResult",
    "ArtifactInventoryValidation",
    "render_stage2b_hardening_figures",
    "validate_figure_artifact_manifest",
    "validate_temporal_evidence_summary",
    "build_stage2b_quality_gate",
    "validate_stage2b_quality_gate",
    "build_artifact_inventory",
    "validate_artifact_inventory",
    "sha256_file",
]
