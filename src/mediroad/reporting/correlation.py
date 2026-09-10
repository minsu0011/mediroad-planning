"""Correlation analysis and publication-quality reporting utilities.

The MEDIROAD master mixes admin-dong measurements with values repeated from
only 11 policy sigungu.  This module therefore treats granularity as part of
the statistical contract rather than merely as plot decoration.  Callers can
attach the following metadata to a :class:`pandas.DataFrame` via ``df.attrs``:

``mediroad_granularity``
    Mapping from feature name to granularity label.
``mediroad_sigungu_repeated_features``
    Explicit sequence of repeated sigungu features.
``mediroad_analysis_level``
    One of ``admin_dong``, ``sigungu_collapsed``, ``within_sigungu`` or
    ``mixed_block_aware``.
``allow_sigungu_repeated_descriptive``
    Explicit opt-in for a raw descriptive atlas that repeats sigungu values.

Without the opt-in, :func:`spearman_with_counts` refuses to calculate a
153-row correlation for features declared to be sigungu-repeated.  The block
bootstrap function additionally uses group-level correlations for every pair
that contains such a feature.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import hashlib
import math
import os
import warnings

import numpy as np
import pandas as pd

# Correlation reporting only writes files.  A GUI backend is unnecessary and
# unsafe when block-bootstrap workers and WSL interpreter teardown overlap.
os.environ["MPLBACKEND"] = "Agg"


_FEATURE_NAME_FIELDS = (
    "source_column",
    "column_name",
    "feature_name",
    "featureName",
    "feature",
    "name",
)
_FEATURE_ID_FIELDS = ("feature_id", "featureId", "fid", "display_id")
_LABEL_FIELDS = (
    "label_ko",
    "display_label",
    "human_label",
    "feature_label",
    "label",
    "description",
)
_AXIS_FIELDS = ("axis", "feature_axis_v6", "model_axis", "score_axis")
_GRANULARITY_FIELDS = (
    "granularity",
    "native_granularity",
    "analysis_granularity",
)
_REPEATED_FLAG_FIELDS = (
    "sigungu_repeated_granularity_flag",
    "sigungu_repeated_flag",
    "group_repeated_flag",
)
_PROXY_FIELDS = (
    "proxy",
    "proxy_flag",
    "proxy_or_approximation_flag",
    "proxyFlag",
)
_ROLE_FIELDS = (
    "role",
    "analysis_role",
    "model_role",
    "data_role",
    "prior_data_role",
    "direct_score_eligibility_v6",
)

_SAFE_ANALYSIS_LEVELS = {
    "sigungu",
    "sigungu_collapsed",
    "within_sigungu",
    "mixed_block_aware",
    "group_aware",
    "cluster_aware",
}

_AXIS_PALETTE = {
    "demographic_vulnerability": "#4C78A8",
    "demographic": "#4C78A8",
    "health_burden_context": "#E45756",
    "health_burden": "#E45756",
    "health": "#E45756",
    "medical_supply_access": "#72B7B2",
    "medical_gap": "#72B7B2",
    "medical": "#72B7B2",
    "transport_access": "#F2CF5B",
    "transport": "#F2CF5B",
    "equity_outreach": "#B279A2",
    "equity_isolation": "#B279A2",
    "equity": "#B279A2",
    "candidate_venue": "#FF9DA6",
    "fine_grid_exposure": "#9D755D",
    "legacy_output": "#D62728",
    "quality_metadata": "#BAB0AC",
    "metadata": "#BAB0AC",
    "unlabeled": "#7F7F7F",
}
_GRANULARITY_PALETTE = {
    "admin_dong": "#4DAF4A",
    "admin-dong": "#4DAF4A",
    "sigungu_repeated": "#E41A1C",
    "sigungu": "#E41A1C",
    "policy_sigungu_repeated": "#E41A1C",
    "global": "#984EA3",
    "optimizer_record": "#FF7F00",
    "unspecified": "#999999",
}
_ROLE_PALETTE = {
    "core": "#1B9E77",
    "core_candidate": "#1B9E77",
    "candidate_input": "#1B9E77",
    "context": "#7570B3",
    "context_only": "#7570B3",
    "optimizer_only": "#D95F02",
    "validation_only": "#E7298A",
    "external_validation": "#E7298A",
    "output_leakage": "#E41A1C",
    "legacy_output": "#E41A1C",
    "identifier": "#666666",
    "id": "#666666",
    "quality_metadata": "#BDBDBD",
    "excluded": "#969696",
    "unlabeled": "#7F7F7F",
}


@dataclass(frozen=True)
class _GranularityInfo:
    repeated_columns: tuple[str, ...]
    analysis_level: str
    group_count: int | None
    handled: bool
    warning: str | None


def _first_existing(frame: pd.DataFrame, names: Iterable[str]) -> str | None:
    return next((name for name in names if name in frame.columns), None)


def _as_bool(value: Any) -> bool:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer, float, np.floating)):
        return bool(value)
    return str(value).strip().lower() in {
        "1",
        "true",
        "t",
        "yes",
        "y",
        "proxy",
        "repeated",
    }


def _normalise_text(value: Any, default: str) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return default
    text = str(value).strip()
    return text if text else default


def _manifest_frame(
    manifest: pd.DataFrame | Mapping[str, Any] | None,
    columns: Sequence[str],
) -> pd.DataFrame:
    """Return plotting/statistical metadata indexed in ``columns`` order."""

    columns = [str(column) for column in columns]
    if manifest is None:
        raw = pd.DataFrame(index=pd.Index(columns, name="feature_name"))
    elif isinstance(manifest, pd.DataFrame):
        raw = manifest.copy()
        key = _first_existing(raw, _FEATURE_NAME_FIELDS)
        if key is not None:
            raw[key] = raw[key].astype(str)
            raw = raw.drop_duplicates(key, keep="first").set_index(key, drop=False)
        else:
            raw.index = raw.index.map(str)
    elif isinstance(manifest, Mapping):
        if manifest and all(
            isinstance(value, Mapping) for value in manifest.values()
        ):
            raw = pd.DataFrame.from_dict(manifest, orient="index")
            raw.index = raw.index.map(str)
        else:
            raw = pd.DataFrame(manifest)
            key = _first_existing(raw, _FEATURE_NAME_FIELDS)
            if key is not None:
                raw[key] = raw[key].astype(str)
                raw = raw.drop_duplicates(key, keep="first").set_index(key, drop=False)
            else:
                raw.index = raw.index.map(str)
    else:
        raise TypeError("manifest must be a pandas DataFrame, mapping, or None")

    raw = raw.reindex(columns)
    result = pd.DataFrame(index=pd.Index(columns, name="feature_name"))
    result["feature_name"] = columns

    id_field = _first_existing(raw, _FEATURE_ID_FIELDS)
    if id_field is None:
        identifiers = pd.Series(index=columns, dtype="object")
    else:
        identifiers = raw[id_field].copy()
    fallback_ids = pd.Series(
        [f"F{position + 1:04d}" for position in range(len(columns))],
        index=columns,
    )
    identifiers = identifiers.reindex(columns)
    missing_id = identifiers.isna() | identifiers.astype(str).str.strip().eq("")
    identifiers = identifiers.astype("object")
    identifiers.loc[missing_id] = fallback_ids.loc[missing_id]
    result["feature_id"] = identifiers.astype(str).to_numpy()

    label_field = _first_existing(raw, _LABEL_FIELDS)
    labels = raw[label_field] if label_field is not None else pd.Series(columns, index=columns)
    labels = labels.reindex(columns).astype("object")
    missing_label = labels.isna() | labels.astype(str).str.strip().eq("")
    labels.loc[missing_label] = pd.Series(columns, index=columns).loc[missing_label]
    result["label"] = labels.astype(str).to_numpy()

    axis_field = _first_existing(raw, _AXIS_FIELDS)
    axis = raw[axis_field] if axis_field is not None else pd.Series("unlabeled", index=columns)
    result["axis"] = [
        _normalise_text(value, "unlabeled") for value in axis.reindex(columns)
    ]

    granularity_field = _first_existing(raw, _GRANULARITY_FIELDS)
    granularity = (
        raw[granularity_field]
        if granularity_field is not None
        else pd.Series("unspecified", index=columns)
    )
    granularity = granularity.reindex(columns).map(
        lambda value: _normalise_text(value, "unspecified")
    )
    repeated_field = _first_existing(raw, _REPEATED_FLAG_FIELDS)
    if repeated_field is not None:
        repeated_flags = raw[repeated_field].reindex(columns).map(_as_bool)
        granularity.loc[repeated_flags] = "sigungu_repeated"
    result["granularity"] = granularity.to_numpy()

    proxy_field = _first_existing(raw, _PROXY_FIELDS)
    proxy = (
        raw[proxy_field].reindex(columns).map(_as_bool)
        if proxy_field is not None
        else pd.Series(False, index=columns)
    )
    result["proxy"] = proxy.astype(bool).to_numpy()

    role_field = _first_existing(raw, _ROLE_FIELDS)
    role = raw[role_field] if role_field is not None else pd.Series("unlabeled", index=columns)
    result["role"] = [
        _normalise_text(value, "unlabeled") for value in role.reindex(columns)
    ]

    result["manifest_matched"] = raw.notna().any(axis=1).reindex(columns).fillna(False).to_numpy()
    return result


def _is_sigungu_label(value: Any) -> bool:
    text = str(value).strip().lower().replace("-", "_")
    return "sigungu" in text and (
        "repeat" in text or "policy" in text or text == "sigungu"
    )


def _declared_repeated_columns(df: pd.DataFrame) -> tuple[str, ...]:
    repeated: set[str] = set()
    for key in (
        "mediroad_sigungu_repeated_features",
        "sigungu_repeated_features",
    ):
        values = df.attrs.get(key)
        if values is not None:
            repeated.update(str(value) for value in values)

    for key in ("mediroad_granularity", "granularity"):
        values = df.attrs.get(key)
        if isinstance(values, Mapping):
            repeated.update(
                str(feature)
                for feature, level in values.items()
                if _is_sigungu_label(level)
            )

    attr_manifest = df.attrs.get("mediroad_manifest")
    if attr_manifest is None:
        attr_manifest = df.attrs.get("feature_manifest")
    if isinstance(attr_manifest, (pd.DataFrame, Mapping)):
        meta = _manifest_frame(attr_manifest, list(df.columns))
        repeated.update(
            meta.loc[meta["granularity"].map(_is_sigungu_label), "feature_name"]
        )
    return tuple(column for column in df.columns if str(column) in repeated)


def _granularity_guard(df: pd.DataFrame) -> _GranularityInfo:
    repeated = _declared_repeated_columns(df)
    level = str(df.attrs.get("mediroad_analysis_level", "admin_dong")).lower()
    group_count_raw = df.attrs.get("mediroad_group_count")
    try:
        group_count = int(group_count_raw) if group_count_raw is not None else None
    except (TypeError, ValueError):
        group_count = None

    if not repeated:
        return _GranularityInfo((), level, group_count, True, None)
    if level in _SAFE_ANALYSIS_LEVELS:
        return _GranularityInfo(repeated, level, group_count, True, None)

    message = (
        f"{len(repeated)} feature(s) are declared sigungu-repeated but the "
        f"analysis level is {level!r}. Collapse to one row per sigungu, use "
        "within-sigungu residuals, or call block_bootstrap_quality."
    )
    if _as_bool(df.attrs.get("allow_sigungu_repeated_descriptive")):
        warnings.warn(
            message + " Continuing only as an explicitly requested descriptive atlas.",
            RuntimeWarning,
            stacklevel=3,
        )
        return _GranularityInfo(repeated, level, group_count, False, message)
    raise ValueError(message)


def _numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if not df.columns.is_unique:
        duplicated = df.columns[df.columns.duplicated()].tolist()
        raise ValueError(f"feature columns must be unique; duplicated={duplicated[:5]}")
    numeric = df.apply(pd.to_numeric, errors="coerce")
    numeric.columns = numeric.columns.map(str)
    return numeric


def _spearman_core(
    df: pd.DataFrame,
    min_periods: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    numeric = _numeric_frame(df)
    if not isinstance(min_periods, (int, np.integer)) or int(min_periods) < 2:
        raise ValueError("min_periods must be an integer >= 2")
    min_periods = int(min_periods)
    if min_periods > len(numeric):
        raise ValueError("min_periods cannot exceed the number of rows")

    ranks = numeric.rank(axis=0, method="average", na_option="keep")
    corr = ranks.corr(method="pearson", min_periods=min_periods)
    valid = numeric.notna().to_numpy(dtype=np.int64, copy=False)
    count_values = valid.T @ valid
    counts = pd.DataFrame(count_values, index=numeric.columns, columns=numeric.columns)
    corr = corr.reindex(index=numeric.columns, columns=numeric.columns)
    corr = corr.mask(counts < min_periods)
    return corr, counts


def spearman_with_counts(
    df: pd.DataFrame,
    min_periods: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute pairwise Spearman correlations and pairwise sample counts.

    Missing values are handled pairwise and are never imputed.  Numeric-like
    strings are accepted; values that cannot be parsed become missing.  See
    the module docstring for the granularity metadata contract.
    """

    granularity = _granularity_guard(df)
    corr, counts = _spearman_core(df, min_periods)
    common_attrs = {
        "method": "spearman_pairwise_complete",
        "min_periods": int(min_periods),
        "n_observations": int(len(df)),
        "mediroad_analysis_level": granularity.analysis_level,
        "mediroad_group_count": granularity.group_count,
        "mediroad_sigungu_repeated_features": list(granularity.repeated_columns),
        "granularity_handled": granularity.handled,
        "granularity_warning": granularity.warning,
    }
    corr.attrs.update(common_attrs)
    counts.attrs.update(common_attrs)
    return corr, counts


def _controls_frame(
    df: pd.DataFrame,
    controls: pd.DataFrame | pd.Series | Sequence[str] | str,
) -> tuple[pd.DataFrame, list[str]]:
    if isinstance(controls, str):
        names = [controls]
        control_frame = df.loc[:, names]
        drop_from_output = names
    elif isinstance(controls, pd.Series):
        name = str(controls.name or "control_1")
        control_frame = controls.rename(name).to_frame().reindex(df.index)
        drop_from_output = []
    elif isinstance(controls, pd.DataFrame):
        if not controls.index.equals(df.index):
            control_frame = controls.reindex(df.index)
        else:
            control_frame = controls.copy()
        drop_from_output = [str(column) for column in controls.columns if column in df]
    elif isinstance(controls, Sequence):
        names = [str(name) for name in controls]
        missing = [name for name in names if name not in df.columns]
        if missing:
            raise KeyError(f"control columns not found in df: {missing}")
        control_frame = df.loc[:, names]
        drop_from_output = names
    else:
        raise TypeError("controls must be a DataFrame, Series, column name, or sequence")

    if control_frame.shape[1] == 0:
        raise ValueError("at least one control is required")
    if not control_frame.columns.is_unique:
        raise ValueError("control columns must be unique")
    control_frame = control_frame.apply(pd.to_numeric, errors="coerce")
    control_frame.columns = control_frame.columns.map(str)
    return control_frame, drop_from_output


def residualized_rank_frame(
    df: pd.DataFrame,
    controls: pd.DataFrame | pd.Series | Sequence[str] | str,
) -> pd.DataFrame:
    """Rank-transform features and return OLS residuals against ranked controls.

    A separate least-squares fit is performed for every feature using rows on
    which that feature and all controls are observed.  Rank transformation
    makes subsequent Pearson correlations partial Spearman correlations.
    Singular or collinear control matrices are handled by ``numpy.linalg.lstsq``.
    """

    granularity = _granularity_guard(df)
    numeric = _numeric_frame(df)
    control_frame, drop_from_output = _controls_frame(df, controls)
    features = numeric.drop(columns=drop_from_output, errors="ignore")
    ranked_features = features.rank(method="average", na_option="keep", pct=True)
    ranked_controls = control_frame.rank(method="average", na_option="keep", pct=True)

    residuals = pd.DataFrame(np.nan, index=df.index, columns=features.columns, dtype=float)
    effective_n: dict[str, int] = {}
    control_valid = ranked_controls.notna().all(axis=1)
    control_values = ranked_controls.to_numpy(dtype=float)
    parameter_count = ranked_controls.shape[1] + 1

    for column in features.columns:
        feature_values = ranked_features[column].to_numpy(dtype=float)
        valid = control_valid.to_numpy() & np.isfinite(feature_values)
        effective_n[column] = int(valid.sum())
        if valid.sum() <= parameter_count:
            continue
        y = feature_values[valid]
        if np.nanmax(y) == np.nanmin(y):
            continue
        design = np.column_stack(
            [np.ones(int(valid.sum()), dtype=float), control_values[valid, :]]
        )
        coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
        residuals.loc[valid, column] = y - design @ coefficients

    residuals.attrs.update(df.attrs)
    residuals.attrs.update(
        {
            "transform": "rank_then_ols_residual",
            "controls": list(ranked_controls.columns),
            "effective_n_by_feature": effective_n,
            "mediroad_analysis_level": granularity.analysis_level,
            "mediroad_sigungu_repeated_features": list(
                granularity.repeated_columns
            ),
            "granularity_handled": granularity.handled,
            "granularity_warning": granularity.warning,
        }
    )
    return residuals


def _aligned_square(corr: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(corr, pd.DataFrame):
        raise TypeError("corr must be a pandas DataFrame")
    if corr.shape[0] != corr.shape[1]:
        raise ValueError("corr must be square")
    if not corr.index.is_unique or not corr.columns.is_unique:
        raise ValueError("corr index and columns must be unique")
    corr = corr.copy()
    corr.index = corr.index.map(str)
    corr.columns = corr.columns.map(str)
    if set(corr.index) != set(corr.columns):
        raise ValueError("corr index and columns must contain the same features")
    corr = corr.reindex(index=corr.index, columns=corr.index)
    numeric = corr.apply(pd.to_numeric, errors="coerce")
    numeric.attrs.update(corr.attrs)
    return numeric


def _clean_correlation_array(corr: pd.DataFrame) -> np.ndarray:
    values = corr.to_numpy(dtype=float, copy=True)
    transpose = values.T
    both = np.isfinite(values) & np.isfinite(transpose)
    either = np.isfinite(values) | np.isfinite(transpose)
    averaged = np.zeros_like(values)
    averaged[both] = (values[both] + transpose[both]) / 2.0
    only_original = np.isfinite(values) & ~np.isfinite(transpose)
    only_transpose = ~np.isfinite(values) & np.isfinite(transpose)
    averaged[only_original] = values[only_original]
    averaged[only_transpose] = transpose[only_transpose]
    averaged[~either] = 0.0
    averaged = np.clip(averaged, -1.0, 1.0)
    np.fill_diagonal(averaged, 1.0)
    return averaged


def hierarchical_order(corr: pd.DataFrame) -> list[str]:
    """Return a deterministic average-linkage order using signed correlation distance."""

    corr = _aligned_square(corr)
    if len(corr) <= 1:
        return corr.index.tolist()

    try:
        from scipy.cluster.hierarchy import leaves_list, linkage
        from scipy.spatial.distance import squareform
    except ImportError as exc:  # pragma: no cover - dependency failure path
        raise ImportError("hierarchical_order requires scipy") from exc

    values = _clean_correlation_array(corr)
    distance = np.sqrt(np.maximum(0.0, (1.0 - values) / 2.0))
    distance = (distance + distance.T) / 2.0
    np.fill_diagonal(distance, 0.0)
    condensed = squareform(distance, checks=False)
    tree = linkage(condensed, method="average", optimal_ordering=True)
    leaves = leaves_list(tree)
    return corr.index.to_numpy()[leaves].tolist()


def _rank_auc(within: np.ndarray, between: np.ndarray) -> float:
    within = within[np.isfinite(within)]
    between = between[np.isfinite(between)]
    if len(within) == 0 or len(between) == 0:
        return float("nan")
    combined = pd.Series(np.concatenate([within, between]))
    ranks = combined.rank(method="average").to_numpy(dtype=float)
    n_within = len(within)
    u_statistic = ranks[:n_within].sum() - n_within * (n_within + 1) / 2.0
    return float(u_statistic / (n_within * len(between)))


def _silhouette_from_corr(values: np.ndarray, axes: np.ndarray) -> float:
    eligible = np.array(
        [bool(str(axis).strip()) and str(axis) != "unlabeled" for axis in axes]
    )
    if eligible.sum() < 3 or len(np.unique(axes[eligible])) < 2:
        return float("nan")
    indices = np.flatnonzero(eligible)
    distance = np.sqrt(np.maximum(0.0, (1.0 - values) / 2.0))
    silhouettes: list[float] = []
    for index in indices:
        same = indices[(axes[indices] == axes[index]) & (indices != index)]
        if len(same) == 0:
            continue
        a_value = float(np.mean(distance[index, same]))
        b_candidates = []
        for other_axis in np.unique(axes[indices]):
            if other_axis == axes[index]:
                continue
            other = indices[axes[indices] == other_axis]
            if len(other):
                b_candidates.append(float(np.mean(distance[index, other])))
        if not b_candidates:
            continue
        b_value = min(b_candidates)
        denominator = max(a_value, b_value)
        silhouettes.append(0.0 if denominator == 0 else (b_value - a_value) / denominator)
    return float(np.mean(silhouettes)) if silhouettes else float("nan")


def _quality_metric_values(values: np.ndarray, axes: np.ndarray) -> dict[str, float]:
    feature_count = values.shape[0]
    if feature_count < 2:
        return {
            "median_abs_corr": float("nan"),
            "median_abs_corr_within_axis": float("nan"),
            "median_abs_corr_between_axis": float("nan"),
            "axis_auc": float("nan"),
            "block_contrast": float("nan"),
            "axis_silhouette": float("nan"),
            "cross_axis_high_corr_rate_0_80": float("nan"),
            "cross_axis_high_corr_rate_0_90": float("nan"),
            "high_corr_pair_rate_0_90": float("nan"),
            "max_abs_corr_offdiag": float("nan"),
        }

    upper = np.triu_indices(feature_count, 1)
    raw_pairs = values[upper]
    valid = np.isfinite(raw_pairs)
    absolute = np.abs(raw_pairs)
    same_axis = (axes[:, None] == axes[None, :])[upper]
    labelled = (
        (axes[:, None] != "unlabeled")
        & (axes[None, :] != "unlabeled")
    )[upper]
    within = absolute[valid & same_axis & labelled]
    between = absolute[valid & ~same_axis & labelled]

    within_median = float(np.median(within)) if len(within) else float("nan")
    between_median = float(np.median(between)) if len(between) else float("nan")
    between_denominator = int((valid & ~same_axis & labelled).sum())
    all_denominator = int(valid.sum())
    clean = np.nan_to_num(values, nan=0.0)
    clean = np.clip((clean + clean.T) / 2.0, -1.0, 1.0)
    np.fill_diagonal(clean, 1.0)

    return {
        "median_abs_corr": float(np.median(absolute[valid])) if valid.any() else float("nan"),
        "median_abs_corr_within_axis": within_median,
        "median_abs_corr_between_axis": between_median,
        "axis_auc": _rank_auc(within, between),
        "block_contrast": within_median - between_median,
        "axis_silhouette": _silhouette_from_corr(clean, axes),
        "cross_axis_high_corr_rate_0_80": (
            float(np.sum(valid & ~same_axis & labelled & (absolute >= 0.80)))
            / between_denominator
            if between_denominator
            else float("nan")
        ),
        "cross_axis_high_corr_rate_0_90": (
            float(np.sum(valid & ~same_axis & labelled & (absolute >= 0.90)))
            / between_denominator
            if between_denominator
            else float("nan")
        ),
        "high_corr_pair_rate_0_90": (
            float(np.sum(valid & (absolute >= 0.90))) / all_denominator
            if all_denominator
            else float("nan")
        ),
        "max_abs_corr_offdiag": (
            float(np.max(absolute[valid])) if valid.any() else float("nan")
        ),
    }


def _finite_or_none(value: Any) -> float | int | bool | str | None:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def _population_columns(meta: pd.DataFrame) -> list[int]:
    explicit_names = {
        "population_total",
        "total_population",
        "resident_population_total",
    }
    positions = []
    for position, row in enumerate(meta.itertuples(index=False)):
        name = str(row.feature_name).lower()
        role = str(row.role).lower()
        label = str(row.label).lower()
        if name in explicit_names or "population_scale" in role or "총인구" in label:
            positions.append(position)
    return positions


def correlation_quality_metrics(
    corr: pd.DataFrame,
    manifest: pd.DataFrame | Mapping[str, Any] | None,
    *,
    spectral_metrics: bool | None = None,
) -> dict[str, Any]:
    """Calculate separation, redundancy, conditioning, and metadata metrics.

    Parameters
    ----------
    corr:
        Square correlation matrix.
    manifest:
        Feature metadata used for axes, granularity, roles, and plot IDs.
    spectral_metrics:
        ``None`` (default) automatically disables eigenvalue-, VIF-, and
        condition-number metrics for a ``mixed_granularity_spearman`` result.
        Such a matrix combines admin-level and group-level cells and is not a
        single Gram/PSD correlation matrix. ``False`` disables the metrics
        explicitly. ``True`` is an expert override and should only be used
        when the caller has independently established a common sample unit.
    """

    corr = _aligned_square(corr)
    meta = _manifest_frame(manifest, corr.index.tolist())
    raw_values = corr.to_numpy(dtype=float)
    values = _clean_correlation_array(corr)
    axes = meta["axis"].astype(str).to_numpy()
    base = _quality_metric_values(raw_values, axes)
    feature_count = len(corr)
    upper = np.triu_indices(feature_count, 1)
    pair_values = raw_values[upper] if feature_count >= 2 else np.array([])
    valid_pairs = np.isfinite(pair_values)

    repeated = meta["granularity"].map(_is_sigungu_label)
    handled = bool(corr.attrs.get("granularity_handled", not repeated.any()))
    analysis_level = str(corr.attrs.get("mediroad_analysis_level", "")).lower()
    method = str(corr.attrs.get("method", "")).lower()
    is_mixed_granularity = (
        analysis_level == "mixed_block_aware"
        or method == "mixed_granularity_spearman"
    )
    if spectral_metrics is None:
        spectral_applicable = not is_mixed_granularity
        spectral_mode = (
            "not_applicable_mixed_granularity"
            if is_mixed_granularity
            else "computed_common_sample_unit"
        )
    elif bool(spectral_metrics):
        spectral_applicable = True
        spectral_mode = (
            "computed_explicit_mixed_override"
            if is_mixed_granularity
            else "computed_explicit"
        )
    else:
        spectral_applicable = False
        spectral_mode = "disabled_by_caller"

    effective_rank = float("nan")
    effective_rank_normalized = float("nan")
    median_vif = float("nan")
    max_vif = float("nan")
    condition_number = float("nan")
    pc1_population_loading = float("nan")
    if spectral_applicable:
        eigenvalues, eigenvectors = np.linalg.eigh(values)
        positive_eigenvalues = np.clip(eigenvalues, 0.0, None)
        eigen_sum = positive_eigenvalues.sum()
        if eigen_sum > 0:
            eigen_probabilities = positive_eigenvalues[positive_eigenvalues > 0] / eigen_sum
            effective_rank = float(
                np.exp(-np.sum(eigen_probabilities * np.log(eigen_probabilities)))
            )
            effective_rank_normalized = (
                effective_rank / feature_count if feature_count else float("nan")
            )

        eigen_floor = 1e-8
        regularised_eigenvalues = np.maximum(eigenvalues, eigen_floor)
        inverse = (eigenvectors * (1.0 / regularised_eigenvalues)) @ eigenvectors.T
        vif = np.diag(inverse)
        median_vif = float(np.median(vif)) if len(vif) else float("nan")
        max_vif = float(np.max(vif)) if len(vif) else float("nan")
        positive = positive_eigenvalues[positive_eigenvalues > eigen_floor]
        condition_number = (
            float(positive.max() / positive.min()) if len(positive) else float("inf")
        )

        population_positions = _population_columns(meta)
        if feature_count and population_positions:
            largest = max(float(eigenvalues[-1]), 0.0)
            correlations_with_pc1 = np.sqrt(largest) * eigenvectors[:, -1]
            pc1_population_loading = float(
                np.max(np.abs(correlations_with_pc1[population_positions]))
            )

    manifest_coverage = float(meta["manifest_matched"].mean()) if feature_count else 1.0
    valid_pair_count = int(valid_pairs.sum())
    total_pairs = int(feature_count * (feature_count - 1) // 2)

    output: dict[str, Any] = {
        "n_features": feature_count,
        "n_total_pairs": total_pairs,
        "n_valid_pairs": valid_pair_count,
        "valid_pair_fraction": valid_pair_count / total_pairs if total_pairs else 1.0,
        **base,
        "n_high_corr_pairs_0_90": int(
            np.sum(valid_pairs & (np.abs(pair_values) >= 0.90))
        ),
        "n_high_corr_pairs_0_95": int(
            np.sum(valid_pairs & (np.abs(pair_values) >= 0.95))
        ),
        "n_effective_exact_duplicate_pairs": int(
            np.sum(valid_pairs & (np.abs(pair_values) >= 1.0 - 1e-12))
        ),
        "effective_rank": effective_rank,
        "effective_rank_normalized": effective_rank_normalized,
        "median_vif": median_vif,
        "max_vif": max_vif,
        "correlation_condition_number": condition_number,
        "pc1_population_loading": pc1_population_loading,
        "spectral_metrics_applicable": spectral_applicable,
        "spectral_metrics_mode": spectral_mode,
        "mixed_granularity_matrix": is_mixed_granularity,
        "manifest_coverage": manifest_coverage,
        "unlabeled_axis_count": int((meta["axis"] == "unlabeled").sum()),
        "unlabeled_role_count": int((meta["role"] == "unlabeled").sum()),
        "sigungu_repeated_feature_count": int(repeated.sum()),
        "proxy_feature_count": int(meta["proxy"].sum()),
        "granularity_handled": handled,
        "granularity_warning": corr.attrs.get("granularity_warning"),
    }

    separation_gate = (
        base["axis_auc"] >= 0.65
        and base["block_contrast"] >= 0.10
        and base["axis_silhouette"] >= 0.10
        and base["cross_axis_high_corr_rate_0_80"] <= 0.05
    )
    pairwise_redundancy_gate = output["n_effective_exact_duplicate_pairs"] == 0
    redundancy_gate = (
        bool(
            pairwise_redundancy_gate
            and output["effective_rank_normalized"] >= 0.50
            and output["median_vif"] < 5.0
            and output["max_vif"] < 10.0
        )
        if spectral_applicable
        else None
    )
    metadata_gate = (
        manifest_coverage == 1.0
        and output["unlabeled_axis_count"] == 0
        and output["unlabeled_role_count"] == 0
    )
    output.update(
        {
            "separation_gate_pass": bool(separation_gate),
            "pairwise_redundancy_gate_pass": bool(pairwise_redundancy_gate),
            "redundancy_gate_pass": redundancy_gate,
            "metadata_gate_pass": bool(metadata_gate),
            "granularity_gate_pass": handled,
            "overall_gate_pass": (
                bool(separation_gate and redundancy_gate and metadata_gate and handled)
                if redundancy_gate is not None
                else None
            ),
        }
    )
    return {key: _finite_or_none(value) for key, value in output.items()}


def _validate_groups(df: pd.DataFrame, groups: pd.Series | Sequence[Any]) -> pd.Series:
    if isinstance(groups, pd.Series):
        aligned = groups.reindex(df.index) if not groups.index.equals(df.index) else groups.copy()
    else:
        if len(groups) != len(df):
            raise ValueError("groups must have the same length as df")
        aligned = pd.Series(list(groups), index=df.index, name="bootstrap_group")
    if len(aligned) != len(df) or aligned.isna().any():
        raise ValueError("groups must align one-to-one with df and contain no missing values")
    if aligned.nunique(dropna=False) < 2:
        raise ValueError("block bootstrap requires at least two groups")
    return aligned


def _validate_repeated_within_groups(
    numeric: pd.DataFrame,
    groups: pd.Series,
    repeated: Sequence[str],
) -> None:
    violations = []
    for column in repeated:
        within_unique = numeric[column].groupby(groups, sort=False).nunique(dropna=True)
        if (within_unique > 1).any():
            violations.append(column)
    if violations:
        raise ValueError(
            "features declared sigungu-repeated vary within at least one group: "
            + ", ".join(violations[:10])
        )


def _group_summary(
    numeric: pd.DataFrame,
    group_instances: pd.Series,
    repeated: Sequence[str],
) -> pd.DataFrame:
    grouped = numeric.groupby(group_instances.to_numpy(), sort=False)
    summary = grouped.median(numeric_only=True)
    if repeated:
        first = grouped[list(repeated)].first()
        summary.loc[:, list(repeated)] = first.loc[:, list(repeated)]
    return summary.reindex(columns=numeric.columns)


def _granularity_aware_corr(
    numeric: pd.DataFrame,
    group_instances: pd.Series,
    repeated: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    row_minimum = max(3, int(math.ceil(len(numeric) * 0.50)))
    row_minimum = min(row_minimum, len(numeric))
    admin_corr, admin_n = _spearman_core(numeric, row_minimum)
    if not repeated:
        admin_corr.attrs.update(
            {
                "mediroad_analysis_level": "cluster_aware",
                "mediroad_group_count": int(group_instances.nunique()),
                "granularity_handled": True,
            }
        )
        return admin_corr, admin_n

    grouped = _group_summary(numeric, group_instances, repeated)
    group_minimum = max(3, int(math.ceil(len(grouped) * 0.70)))
    group_minimum = min(group_minimum, len(grouped))
    group_corr, group_n = _spearman_core(grouped, group_minimum)
    positions = [numeric.columns.get_loc(column) for column in repeated]
    mask = np.zeros(admin_corr.shape, dtype=bool)
    mask[positions, :] = True
    mask[:, positions] = True
    corr_values = admin_corr.to_numpy(dtype=float, copy=True)
    n_values = admin_n.to_numpy(dtype=np.int64, copy=True)
    grouped_values = group_corr.to_numpy(dtype=float)
    grouped_n_values = group_n.to_numpy(dtype=np.int64)
    corr_values[mask] = grouped_values[mask]
    n_values[mask] = grouped_n_values[mask]
    corr = pd.DataFrame(corr_values, index=numeric.columns, columns=numeric.columns)
    counts = pd.DataFrame(n_values, index=numeric.columns, columns=numeric.columns)
    attrs = {
        "method": "mixed_granularity_spearman",
        "mediroad_analysis_level": "mixed_block_aware",
        "mediroad_group_count": int(group_instances.nunique()),
        "mediroad_sigungu_repeated_features": list(repeated),
        "granularity_handled": True,
        "group_aggregation": "median_admin_features_first_repeated_features",
        "row_level_min_periods": row_minimum,
        "group_level_min_periods": group_minimum,
    }
    corr.attrs.update(attrs)
    counts.attrs.update(attrs)
    return corr, counts


def mixed_granularity_spearman(
    df: pd.DataFrame,
    groups: pd.Series | Sequence[Any],
    manifest: pd.DataFrame | Mapping[str, Any] | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute a correlation matrix without sigungu pseudo-replication.

    Admin-dong/admin-dong pairs use all admin-dong rows.  Every pair containing
    a feature labelled ``sigungu_repeated`` in ``manifest`` instead uses one
    group-level observation per sigungu: repeated features use their verified
    within-group value and admin-dong features use the within-group median.
    The returned pair-count matrix therefore makes the effective sample size
    (roughly 153 versus 11 in MEDIROAD) explicit cell by cell.
    """

    numeric = _numeric_frame(df)
    group_series = _validate_groups(numeric, groups)
    meta = _manifest_frame(manifest, numeric.columns.tolist())
    repeated = meta.loc[
        meta["granularity"].map(_is_sigungu_label), "feature_name"
    ].tolist()
    _validate_repeated_within_groups(numeric, group_series, repeated)
    group_values = pd.unique(group_series)
    group_instances = pd.Series(
        pd.Categorical(group_series, categories=group_values).codes,
        index=numeric.index,
    )
    corr, counts = _granularity_aware_corr(
        numeric,
        group_instances,
        repeated,
    )
    corr.attrs["manifest_coverage"] = float(meta["manifest_matched"].mean())
    counts.attrs["manifest_coverage"] = corr.attrs["manifest_coverage"]
    return corr, counts


def _safe_nan_stat(function: Any, values: np.ndarray, axis: int = 0) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return function(values, axis=axis)


def block_bootstrap_quality(
    df: pd.DataFrame,
    groups: pd.Series | Sequence[Any],
    manifest: pd.DataFrame | Mapping[str, Any] | None,
    n_boot: int = 1000,
    seed: int = 42,
    n_jobs: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate correlation-quality and edge stability by group bootstrap.

    Groups, rather than individual admin-dong rows, are sampled with
    replacement.  For a pair involving a sigungu-repeated feature, both
    variables are first reduced to one robust summary per sampled block.  This
    prevents the number of admin-dong rows in a sigungu from masquerading as
    independent evidence.
    """

    if not isinstance(n_boot, (int, np.integer)) or int(n_boot) < 1:
        raise ValueError("n_boot must be a positive integer")
    n_boot = int(n_boot)
    numeric = _numeric_frame(df)
    group_series = _validate_groups(numeric, groups)
    meta = _manifest_frame(manifest, numeric.columns.tolist())
    repeated = meta.loc[
        meta["granularity"].map(_is_sigungu_label), "feature_name"
    ].tolist()
    _validate_repeated_within_groups(numeric, group_series, repeated)

    group_values = pd.unique(group_series)
    group_positions = {
        group: np.flatnonzero(group_series.to_numpy() == group)
        for group in group_values
    }
    base_instances = pd.Series(
        pd.Categorical(group_series, categories=group_values).codes,
        index=numeric.index,
    )
    base_corr, base_counts = _granularity_aware_corr(
        numeric,
        base_instances,
        repeated,
    )
    base_values = base_corr.to_numpy(dtype=float)
    base_count_values = base_counts.to_numpy(dtype=np.int64)
    axes = meta["axis"].astype(str).to_numpy()
    metric_names = [
        "median_abs_corr",
        "median_abs_corr_within_axis",
        "median_abs_corr_between_axis",
        "axis_auc",
        "block_contrast",
        "axis_silhouette",
        "cross_axis_high_corr_rate_0_80",
        "cross_axis_high_corr_rate_0_90",
        "high_corr_pair_rate_0_90",
        "max_abs_corr_offdiag",
    ]
    base_metrics = _quality_metric_values(base_values, axes)
    upper = np.triu_indices(len(numeric.columns), 1)
    pair_count = len(upper[0])
    edge_samples = np.full((n_boot, pair_count), np.nan, dtype=np.float32)
    metric_samples = np.full((n_boot, len(metric_names)), np.nan, dtype=float)

    seed_sequence = np.random.SeedSequence(int(seed))
    child_seeds = [
        int(child.generate_state(1, dtype=np.uint64)[0])
        for child in seed_sequence.spawn(n_boot)
    ]

    def one_bootstrap(child_seed: int) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(child_seed)
        selected = rng.integers(0, len(group_values), size=len(group_values))
        pieces = []
        instance_values = []
        for instance, selected_position in enumerate(selected):
            positions = group_positions[group_values[selected_position]]
            pieces.append(numeric.iloc[positions])
            instance_values.extend([instance] * len(positions))
        sampled = pd.concat(pieces, axis=0, ignore_index=True)
        instances = pd.Series(instance_values, index=sampled.index)
        sampled_corr, _ = _granularity_aware_corr(sampled, instances, repeated)
        sampled_values = sampled_corr.to_numpy(dtype=float)
        metrics = _quality_metric_values(sampled_values, axes)
        metric_vector = np.array([metrics[name] for name in metric_names], dtype=float)
        return metric_vector, sampled_values[upper].astype(np.float32, copy=False)

    if n_jobs is None or int(n_jobs) == 0:
        worker_count = 1
    elif int(n_jobs) < 0:
        worker_count = max(1, os.cpu_count() or 1)
    else:
        worker_count = int(n_jobs)
    worker_count = min(worker_count, n_boot)

    if worker_count == 1:
        iterator = map(one_bootstrap, child_seeds)
        for bootstrap_index, (metrics, edges) in enumerate(iterator):
            metric_samples[bootstrap_index] = metrics
            edge_samples[bootstrap_index] = edges
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            for bootstrap_index, (metrics, edges) in enumerate(
                executor.map(one_bootstrap, child_seeds)
            ):
                metric_samples[bootstrap_index] = metrics
                edge_samples[bootstrap_index] = edges

    summary_rows = []
    for metric_index, metric_name in enumerate(metric_names):
        sample = metric_samples[:, metric_index]
        valid = sample[np.isfinite(sample)]
        summary_rows.append(
            {
                "metric": metric_name,
                "point_estimate": _finite_or_none(base_metrics[metric_name]),
                "bootstrap_mean": float(np.mean(valid)) if len(valid) else np.nan,
                "bootstrap_median": float(np.median(valid)) if len(valid) else np.nan,
                "bootstrap_sd": float(np.std(valid, ddof=1)) if len(valid) > 1 else np.nan,
                "ci05": float(np.quantile(valid, 0.05)) if len(valid) else np.nan,
                "ci95": float(np.quantile(valid, 0.95)) if len(valid) else np.nan,
                "n_valid": int(len(valid)),
                "n_boot": n_boot,
                "seed": int(seed),
                "block_count": int(len(group_values)),
                "repeated_feature_count": int(len(repeated)),
                "granularity_mode": "mixed_block_aware" if repeated else "block_aware",
            }
        )
    summary = pd.DataFrame(summary_rows)

    base_edges = base_values[upper]
    base_edge_counts = base_count_values[upper]
    valid_counts = np.isfinite(edge_samples).sum(axis=0)
    valid_fraction = valid_counts.astype(float) / float(n_boot)
    edge_mean = _safe_nan_stat(np.nanmean, edge_samples)
    edge_median = _safe_nan_stat(np.nanmedian, edge_samples)
    edge_sd = _safe_nan_stat(np.nanstd, edge_samples)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        edge_q05 = np.nanquantile(edge_samples, 0.05, axis=0)
        edge_q95 = np.nanquantile(edge_samples, 0.95, axis=0)
    base_sign = np.sign(base_edges)
    finite_sample = np.isfinite(edge_samples)
    same_sign = np.sign(edge_samples) == base_sign[None, :]
    # Undefined bootstrap correlations are conservatively counted as failures,
    # not removed from the denominator. This prevents low-variation binary or
    # group-context features from receiving inflated stability probabilities.
    sign_consistency = np.sum(same_sign & finite_sample, axis=0) / float(n_boot)
    probability_070 = np.sum(
        (np.abs(edge_samples) >= 0.70) & finite_sample,
        axis=0,
    ) / float(n_boot)
    probability_080 = np.sum(
        (np.abs(edge_samples) >= 0.80) & finite_sample,
        axis=0,
    ) / float(n_boot)

    left = upper[0]
    right = upper[1]
    edge_frame = pd.DataFrame(
        {
            "feature_a": meta.iloc[left]["feature_name"].to_numpy(),
            "feature_b": meta.iloc[right]["feature_name"].to_numpy(),
            "feature_id_a": meta.iloc[left]["feature_id"].to_numpy(),
            "feature_id_b": meta.iloc[right]["feature_id"].to_numpy(),
            "axis_a": meta.iloc[left]["axis"].to_numpy(),
            "axis_b": meta.iloc[right]["axis"].to_numpy(),
            "same_axis": axes[left] == axes[right],
            "granularity_a": meta.iloc[left]["granularity"].to_numpy(),
            "granularity_b": meta.iloc[right]["granularity"].to_numpy(),
            "base_rho": base_edges,
            "base_abs_rho": np.abs(base_edges),
            "base_pairwise_n": base_edge_counts,
            "bootstrap_mean": edge_mean,
            "bootstrap_median": edge_median,
            "bootstrap_sd": edge_sd,
            "ci05": edge_q05,
            "ci95": edge_q95,
            "valid_bootstraps": valid_counts,
            "valid_fraction": valid_fraction,
            "probability_denominator": n_boot,
            "sign_consistency": sign_consistency,
            "probability_abs_ge_070": probability_070,
            "probability_abs_ge_080": probability_080,
        }
    )
    edge_frame["stable_edge"] = (
        edge_frame["base_abs_rho"].ge(0.70)
        & edge_frame["valid_fraction"].ge(0.80)
        & edge_frame["sign_consistency"].ge(0.90)
        & edge_frame["probability_abs_ge_070"].ge(0.80)
    )
    edge_frame = edge_frame.sort_values(
        ["stable_edge", "base_abs_rho"], ascending=[False, False]
    ).reset_index(drop=True)
    summary.attrs.update(base_corr.attrs)
    edge_frame.attrs.update(base_corr.attrs)
    return summary, edge_frame


def _stable_colour(value: str, palette: Mapping[str, str]) -> str:
    text = _normalise_text(value, "unlabeled")
    lowered = text.lower().replace("-", "_").replace(" ", "_")
    if lowered in palette:
        return palette[lowered]
    for key, colour in palette.items():
        if key in lowered:
            return colour
    digest = hashlib.sha256(lowered.encode("utf-8")).digest()
    hue = int.from_bytes(digest[:2], "big") / 65535.0
    saturation = 0.48
    value_component = 0.78
    sector = int(hue * 6)
    fraction = hue * 6 - sector
    p_value = value_component * (1 - saturation)
    q_value = value_component * (1 - fraction * saturation)
    t_value = value_component * (1 - (1 - fraction) * saturation)
    sector %= 6
    rgb_options = (
        (value_component, t_value, p_value),
        (q_value, value_component, p_value),
        (p_value, value_component, t_value),
        (p_value, q_value, value_component),
        (t_value, p_value, value_component),
        (value_component, p_value, q_value),
    )
    red, green, blue = rgb_options[sector]
    return f"#{round(red * 255):02X}{round(green * 255):02X}{round(blue * 255):02X}"


def _hex_to_rgb(colour: str) -> tuple[float, float, float]:
    colour = colour.lstrip("#")
    return tuple(int(colour[position : position + 2], 16) / 255.0 for position in (0, 2, 4))


def _metadata_colours(meta: pd.DataFrame) -> tuple[np.ndarray, dict[str, dict[str, str]]]:
    categories: list[tuple[str, list[str], Mapping[str, str]]] = [
        ("Axis", meta["axis"].astype(str).tolist(), _AXIS_PALETTE),
        (
            "Granularity",
            meta["granularity"].astype(str).tolist(),
            _GRANULARITY_PALETTE,
        ),
        (
            "Proxy",
            ["proxy" if value else "direct" for value in meta["proxy"]],
            {"proxy": "#FF7F00", "direct": "#D9D9D9"},
        ),
        ("Role", meta["role"].astype(str).tolist(), _ROLE_PALETTE),
    ]
    colour_rows = []
    legend: dict[str, dict[str, str]] = {}
    for category, values, palette in categories:
        # A constant ribbon carries no separation information and consumes
        # scarce label space in large atlases.  Its value remains available
        # in the saved manifest and interactive cell tooltips.
        if len(set(values)) <= 1:
            continue
        mapping: dict[str, str] = {}
        row = []
        for value in values:
            mapping.setdefault(value, _stable_colour(value, palette))
            row.append(_hex_to_rgb(mapping[value]))
        colour_rows.append(row)
        legend[category] = mapping
    if not colour_rows:
        return np.empty((0, len(meta), 3), dtype=float), legend
    return np.asarray(colour_rows, dtype=float), legend


def _shorten(text: str, maximum: int = 26) -> str:
    text = str(text).replace("\n", " ").strip()
    return text if len(text) <= maximum else text[: maximum - 1] + "…"


def _aligned_counts(n: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    if not isinstance(n, pd.DataFrame):
        raise TypeError("n must be a pandas DataFrame")
    aligned = n.copy()
    aligned.index = aligned.index.map(str)
    aligned.columns = aligned.columns.map(str)
    if set(columns) - set(aligned.index) or set(columns) - set(aligned.columns):
        raise ValueError("n must contain all corr features on both axes")
    return aligned.reindex(index=columns, columns=columns).apply(
        pd.to_numeric, errors="coerce"
    )


def _path_without_known_suffix(path_base: str | Path) -> Path:
    path = Path(path_base)
    if path.suffix.lower() in {".png", ".svg", ".pdf", ".html"}:
        return path.with_suffix("")
    return path


def _matplotlib_korean_font() -> str:
    """Register a host Korean font in WSL before figures are rendered."""

    try:
        from matplotlib import font_manager
    except ImportError:  # pragma: no cover - handled by plotting caller
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
    )
    for regular, bold in candidates:
        if regular.is_file():
            font_manager.fontManager.addfont(str(regular))
            if bold is not None and bold.is_file():
                font_manager.fontManager.addfont(str(bold))
            return font_manager.FontProperties(fname=str(regular)).get_name()
    return "DejaVu Sans"


def plot_correlation_map(
    corr: pd.DataFrame,
    n: pd.DataFrame,
    manifest: pd.DataFrame | Mapping[str, Any] | None,
    path_base: str | Path,
    title: str,
    annotate: bool = False,
    mask_n: int | None = None,
) -> list[str]:
    """Save a signed, metadata-annotated correlation map as PNG and SVG."""

    corr = _aligned_square(corr)
    counts = _aligned_counts(n, corr.index.tolist())
    meta = _manifest_frame(manifest, corr.index.tolist())
    values = corr.to_numpy(dtype=float)
    count_values = counts.to_numpy(dtype=float)
    mask = ~np.isfinite(values)
    if mask_n is not None:
        if int(mask_n) < 1:
            raise ValueError("mask_n must be positive when supplied")
        mask |= ~np.isfinite(count_values) | (count_values < int(mask_n))

    try:
        import matplotlib as mpl
        mpl.use("Agg", force=True)
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except ImportError as exc:  # pragma: no cover - dependency failure path
        raise ImportError("plot_correlation_map requires matplotlib") from exc
    korean_font = _matplotlib_korean_font()

    feature_count = len(corr)
    if feature_count == 0:
        raise ValueError("cannot plot an empty correlation matrix")
    base = _path_without_known_suffix(path_base)
    base.parent.mkdir(parents=True, exist_ok=True)
    png_path = base.with_suffix(".png")
    svg_path = base.with_suffix(".svg")

    matrix_size = (
        max(10.0, min(22.0, 0.34 * feature_count))
        if feature_count <= 80
        else 22.0
    )
    figure_width = matrix_size + 5.0
    colour_rows, legends = _metadata_colours(meta)
    has_ribbons = bool(len(colour_rows))
    displayed_colours = (
        colour_rows
        if has_ribbons
        else np.ones((1, feature_count, 3), dtype=float)
    )
    left_colours = np.transpose(displayed_colours, (1, 0, 2))

    with mpl.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [korean_font, "DejaVu Sans"],
            "axes.unicode_minus": False,
        }
    ):
        figure = plt.figure(figsize=(figure_width, matrix_size), constrained_layout=False)
        grid = figure.add_gridspec(
            2,
            2,
            width_ratios=(0.035 if has_ribbons else 0.001, 1.0),
            height_ratios=(
                max(0.018, 0.014 * len(colour_rows)) if has_ribbons else 0.001,
                1.0,
            ),
            left=0.07,
            right=0.80,
            bottom=0.10,
            top=0.91,
            wspace=0.015,
            hspace=0.015,
        )
        top_axis = figure.add_subplot(grid[0, 1])
        left_axis = figure.add_subplot(grid[1, 0])
        main_axis = figure.add_subplot(grid[1, 1])
        top_axis.imshow(displayed_colours, aspect="auto", interpolation="nearest")
        top_axis.set_yticks(
            range(len(legends)),
            ["Gran." if value == "Granularity" else value for value in legends],
            fontsize=7,
        )
        top_axis.set_xticks([])
        top_axis.tick_params(length=0)
        for spine in top_axis.spines.values():
            spine.set_visible(False)
        left_axis.imshow(left_colours, aspect="auto", interpolation="nearest")
        left_axis.set_xticks([])
        left_axis.set_yticks([])
        for spine in left_axis.spines.values():
            spine.set_visible(False)
        if not has_ribbons:
            top_axis.set_axis_off()
            left_axis.set_axis_off()

        colour_map = mpl.colormaps["RdBu_r"].with_extremes(bad="#BDBDBD")
        image = main_axis.imshow(
            np.ma.array(values, mask=mask),
            vmin=-1.0,
            vmax=1.0,
            cmap=colour_map,
            aspect="equal",
            interpolation="nearest",
            rasterized=True,
        )

        if feature_count <= 60:
            labels = [
                f"{row.feature_id} · {_shorten(row.label)}"
                for row in meta.itertuples(index=False)
            ]
            tick_positions = np.arange(feature_count)
            x_font = max(5.0, min(8.0, 180.0 / max(feature_count, 1)))
            y_font = x_font
        else:
            tick_step = max(1, math.ceil(feature_count / 80))
            tick_positions = np.arange(0, feature_count, tick_step)
            labels = meta.iloc[tick_positions]["feature_id"].astype(str).tolist()
            x_font = 5.0
            y_font = 5.0
        main_axis.set_xticks(tick_positions, labels, rotation=90, fontsize=x_font)
        main_axis.set_yticks(tick_positions, labels, fontsize=y_font)
        main_axis.tick_params(length=2, pad=1)
        main_axis.set_xlabel("Feature ID (full labels and metadata are in the manifest)")
        main_axis.set_ylabel("Feature ID")

        if annotate:
            if feature_count > 60:
                warnings.warn(
                    "Cell annotations were disabled for more than 60 features to prevent collisions.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            else:
                threshold = -np.inf if feature_count <= 25 else 0.50
                font_size = 6.0 if feature_count <= 25 else 4.5
                for row in range(feature_count):
                    for column in range(feature_count):
                        value = values[row, column]
                        if mask[row, column] or abs(value) < threshold:
                            continue
                        main_axis.text(
                            column,
                            row,
                            f"{value:.2f}",
                            ha="center",
                            va="center",
                            fontsize=font_size,
                            color="white" if abs(value) >= 0.58 else "#222222",
                        )

        colour_bar = figure.colorbar(image, ax=main_axis, fraction=0.035, pad=0.02)
        colour_bar.set_label("Spearman ρ", rotation=90)
        warning_text = corr.attrs.get("granularity_warning")
        subtitle_parts = ["signed diverging scale; grey = unavailable"]
        if mask_n is not None:
            subtitle_parts.append(f"grey also marks pairwise n < {int(mask_n)}")
        if warning_text:
            subtitle_parts.append("WARNING: sigungu-repeated values shown descriptively")
        figure.suptitle(title, fontsize=14, fontweight="bold", y=0.975)
        figure.text(0.07, 0.035, " | ".join(subtitle_parts), fontsize=8, color="#444444")

        legend_y = 0.90
        for category, mapping in legends.items():
            handles = [
                Patch(facecolor=colour, edgecolor="none", label=_shorten(value, 32))
                for value, colour in mapping.items()
            ]
            legend = figure.legend(
                handles=handles,
                title=category,
                loc="upper left",
                bbox_to_anchor=(0.815, legend_y),
                frameon=False,
                fontsize=6.5,
                title_fontsize=7.5,
                borderaxespad=0,
                handlelength=1.2,
                labelspacing=0.3,
            )
            figure.add_artist(legend)
            legend_y -= min(0.22, 0.045 + 0.025 * len(handles))

        figure.savefig(png_path, dpi=300, facecolor="white", bbox_inches="tight")
        # dpi also controls the embedded raster resolution inside the otherwise
        # vector SVG (the heatmap artist is intentionally rasterized).
        figure.savefig(svg_path, dpi=300, facecolor="white", bbox_inches="tight")
        plt.close(figure)
    return [str(png_path), str(svg_path)]


def _discrete_plotly_scale(colours: Sequence[str]) -> list[list[Any]]:
    count = len(colours)
    if count == 1:
        return [[0.0, colours[0]], [1.0, colours[0]]]
    scale: list[list[Any]] = []
    for index, colour in enumerate(colours):
        start = index / count
        end = (index + 1) / count
        scale.extend([[start, colour], [end, colour]])
    return scale


def save_interactive_heatmap(
    corr: pd.DataFrame,
    n: pd.DataFrame,
    manifest: pd.DataFrame | Mapping[str, Any] | None,
    path: str | Path,
    title: str,
) -> str:
    """Save a self-contained Plotly heatmap with F-ID and metadata tooltips."""

    corr = _aligned_square(corr)
    counts = _aligned_counts(n, corr.index.tolist())
    meta = _manifest_frame(manifest, corr.index.tolist())
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as exc:  # pragma: no cover - dependency failure path
        raise ImportError("save_interactive_heatmap requires plotly") from exc

    output_path = Path(path)
    if output_path.suffix.lower() != ".html":
        output_path = output_path.with_suffix(".html")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mask_n_raw = corr.attrs.get("mask_n", corr.attrs.get("min_periods"))
    try:
        mask_n = int(mask_n_raw) if mask_n_raw is not None else None
    except (TypeError, ValueError):
        mask_n = None
    values = corr.to_numpy(dtype=float, copy=True)
    count_values = counts.to_numpy(dtype=float)
    if mask_n is not None:
        values[(~np.isfinite(count_values)) | (count_values < mask_n)] = np.nan

    hover_labels = []
    for row in meta.itertuples(index=False):
        hover_labels.append(
            f"{row.feature_id} · {row.label}"
            f"<br>axis={row.axis}; granularity={row.granularity}; "
            f"proxy={'yes' if row.proxy else 'no'}; role={row.role}"
        )

    ribbon_definitions = [
        ("Axis", meta["axis"].astype(str), _AXIS_PALETTE),
        ("Granularity", meta["granularity"].astype(str), _GRANULARITY_PALETTE),
        (
            "Proxy",
            meta["proxy"].map({True: "proxy", False: "direct"}),
            {"proxy": "#FF7F00", "direct": "#D9D9D9"},
        ),
        ("Role", meta["role"].astype(str), _ROLE_PALETTE),
    ]
    ribbon_definitions = [
        definition
        for definition in ribbon_definitions
        if definition[1].nunique(dropna=False) > 1
    ]
    heatmap_row = len(ribbon_definitions) + 1
    ribbon_height = 0.025
    figure = make_subplots(
        rows=heatmap_row,
        cols=1,
        shared_xaxes=True,
        row_heights=[ribbon_height] * len(ribbon_definitions)
        + [1.0 - ribbon_height * len(ribbon_definitions)],
        vertical_spacing=0.003,
    )
    for row_number, (category, series, palette) in enumerate(
        ribbon_definitions, start=1
    ):
        unique_values = list(dict.fromkeys(series.tolist()))
        mapping = {value: index for index, value in enumerate(unique_values)}
        colours = [_stable_colour(value, palette) for value in unique_values]
        codes = np.array([[mapping[value] for value in series]], dtype=float)
        custom = np.array([[value for value in series]], dtype=object)
        figure.add_trace(
            go.Heatmap(
                z=codes,
                x=hover_labels,
                y=[category],
                customdata=custom,
                colorscale=_discrete_plotly_scale(colours),
                zmin=-0.5,
                zmax=max(0.5, len(unique_values) - 0.5),
                showscale=False,
                hovertemplate=(
                    f"{category}: %{{customdata}}<br>Feature: %{{x}}<extra></extra>"
                ),
            ),
            row=row_number,
            col=1,
        )

    figure.add_trace(
        go.Heatmap(
            z=values,
            x=hover_labels,
            y=hover_labels,
            customdata=count_values,
            zmin=-1.0,
            zmax=1.0,
            zmid=0.0,
            colorscale=[
                [0.0, "#2166AC"],
                [0.5, "#F7F7F7"],
                [1.0, "#B2182B"],
            ],
            colorbar={"title": "Spearman ρ"},
            hoverongaps=True,
            hovertemplate=(
                "X: %{x}<br>Y: %{y}<br>Spearman ρ=%{z:.3f}"
                "<br>pairwise n=%{customdata:.0f}<extra></extra>"
            ),
        ),
        row=heatmap_row,
        col=1,
    )

    feature_count = len(corr)
    tick_step = max(1, math.ceil(feature_count / 80))
    selected = np.arange(0, feature_count, tick_step)
    tick_values = [hover_labels[position] for position in selected]
    tick_text = meta.iloc[selected]["feature_id"].astype(str).tolist()
    figure.update_xaxes(
        tickmode="array",
        tickvals=tick_values,
        ticktext=tick_text,
        tickangle=90,
        tickfont={"size": 8},
        row=heatmap_row,
        col=1,
    )
    figure.update_yaxes(
        tickmode="array",
        tickvals=tick_values,
        ticktext=tick_text,
        tickfont={"size": 8},
        autorange="reversed",
        row=heatmap_row,
        col=1,
    )
    for row_number in range(1, heatmap_row):
        figure.update_yaxes(showticklabels=True, tickfont={"size": 9}, row=row_number, col=1)

    warning_text = corr.attrs.get("granularity_warning")
    subtitle = "Grey cells are unavailable"
    if mask_n is not None:
        subtitle += f" or have pairwise n < {mask_n}"
    if warning_text:
        subtitle += "; WARNING: repeated sigungu values are descriptive only"
    figure.update_layout(
        title={"text": f"{title}<br><sup>{subtitle}</sup>", "x": 0.5},
        width=min(1800, max(900, 18 * feature_count)),
        height=min(1800, max(850, 18 * feature_count)),
        plot_bgcolor="#BDBDBD",
        paper_bgcolor="white",
        margin={"l": 110, "r": 100, "t": 100, "b": 120},
    )
    figure.write_html(
        str(output_path),
        include_plotlyjs=True,
        full_html=True,
        auto_open=False,
    )
    return str(output_path)


__all__ = [
    "spearman_with_counts",
    "residualized_rank_frame",
    "hierarchical_order",
    "correlation_quality_metrics",
    "mixed_granularity_spearman",
    "block_bootstrap_quality",
    "plot_correlation_map",
    "save_interactive_heatmap",
]
