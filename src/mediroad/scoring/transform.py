"""Deterministic, explainable transforms for the MODEL V1 policy score."""

from __future__ import annotations

import numpy as np
import pandas as pd


SCALERS = {
    "rank_percentile",
    "winsorized_percentile",
    "robust_z",
    "robust_minmax",
}

POLICY_SIGUNGU_REPEATED_GRANULARITY = "policy_sigungu_repeated_on_admin_dong"
EXPECTED_POLICY_SIGUNGU_COUNT = 11


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype(float)


def apply_transform(series: pd.Series, transform: str | None) -> pd.Series:
    """Apply a configured monotone transform without hiding assumptions."""

    values = _numeric(series)
    transform = (transform or "identity").lower()
    if transform in {"identity", "none"}:
        return values
    if transform == "log1p":
        if (values.dropna() < 0).any():
            raise ValueError(f"log1p received negative values for {series.name}")
        return np.log1p(values)
    if transform == "sqrt":
        if (values.dropna() < 0).any():
            raise ValueError(f"sqrt received negative values for {series.name}")
        return np.sqrt(values)
    if transform == "binary":
        observed = set(values.dropna().unique())
        if not observed.issubset({0.0, 1.0}):
            raise ValueError(f"binary feature {series.name} has values outside 0/1: {observed}")
        return values
    raise ValueError(f"Unsupported transform: {transform}")


def scale_need_direction(
    series: pd.Series,
    *,
    method: str,
    direction: int,
    transform: str | None = None,
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
    z_clip_lower: float = -3.0,
    z_clip_upper: float = 3.0,
    require_complete: bool = True,
) -> pd.Series:
    """Return a 0-100 score where larger always means greater policy need."""

    if method not in SCALERS:
        raise ValueError(f"Unknown scaler {method}; expected one of {sorted(SCALERS)}")
    if direction not in {-1, 1}:
        raise ValueError(f"Direction must be +1 or -1, got {direction} for {series.name}")

    values = apply_transform(series, transform)
    if require_complete and values.isna().any():
        raise ValueError(f"Active feature {series.name} has {int(values.isna().sum())} missing values")
    observed = values.dropna()
    if observed.nunique() <= 1:
        raise ValueError(f"Active feature {series.name} is constant")

    lo = float(observed.quantile(lower_quantile))
    hi = float(observed.quantile(upper_quantile))
    clipped = values.clip(lower=lo, upper=hi)

    if method == "rank_percentile":
        scaled = values.rank(method="average", pct=True) * 100.0
    elif method == "winsorized_percentile":
        scaled = clipped.rank(method="average", pct=True) * 100.0
    elif method == "robust_minmax":
        span = hi - lo
        scaled = pd.Series(50.0, index=values.index) if span <= 0 else (clipped - lo) / span * 100.0
    else:
        median = float(observed.median())
        mad = float(np.median(np.abs(observed.to_numpy() - median)))
        scale = mad * 1.4826
        if not np.isfinite(scale) or scale <= 1e-12:
            q25, q75 = observed.quantile([0.25, 0.75])
            scale = float((q75 - q25) / 1.349)
        if not np.isfinite(scale) or scale <= 1e-12:
            raise ValueError(f"Cannot robust-scale {series.name}")
        if z_clip_upper <= z_clip_lower:
            raise ValueError("z_clip_upper must exceed z_clip_lower")
        z_values = ((values - median) / scale).clip(z_clip_lower, z_clip_upper)
        scaled = (z_values - z_clip_lower) / (z_clip_upper - z_clip_lower) * 100.0

    if direction < 0:
        scaled = 100.0 - scaled
    return scaled.clip(0.0, 100.0)


def transformed_feature_matrix(
    master: pd.DataFrame,
    feature_specs: list[dict],
    scaler: str,
    scaler_options: dict | None = None,
) -> pd.DataFrame:
    """Build the auditable need-oriented feature matrix for one scaler."""

    columns: dict[str, pd.Series] = {}
    options = dict(scaler_options or {})
    for spec in feature_specs:
        name = spec["name"]
        if name not in master.columns:
            raise KeyError(f"Configured feature is missing from master: {name}")
        direction = spec.get("direction")
        if isinstance(direction, str):
            direction = 1 if direction.strip() in {"+", "+1", "higher_need"} else -1

        values = master[name]
        granularity = str(spec.get("granularity", "")).strip().lower()
        repeated_on_admin_dong = granularity == POLICY_SIGUNGU_REPEATED_GRANULARITY
        if repeated_on_admin_dong:
            group_column = "policy_sigungu_name"
            if group_column not in master.columns:
                raise KeyError(
                    f"{name} is {POLICY_SIGUNGU_REPEATED_GRANULARITY} but "
                    f"master lacks {group_column}"
                )
            groups = master[group_column]
            if groups.isna().any():
                raise ValueError(
                    f"{name} cannot be collapsed because {group_column} has "
                    f"{int(groups.isna().sum())} missing values"
                )
            group_count = int(groups.nunique(dropna=False))
            if group_count != EXPECTED_POLICY_SIGUNGU_COUNT:
                raise ValueError(
                    f"{name} requires exactly {EXPECTED_POLICY_SIGUNGU_COUNT} unique "
                    f"{group_column} values, found {group_count}"
                )

            grouped = pd.DataFrame(
                {group_column: groups.to_numpy(), "__feature_value": values.to_numpy()},
                index=master.index,
            )
            within_group_unique = grouped.groupby(group_column, sort=False, dropna=False)[
                "__feature_value"
            ].nunique(dropna=False)
            inconsistent = within_group_unique[within_group_unique.ne(1)].index.astype(str).tolist()
            if inconsistent:
                raise ValueError(
                    f"{name} is declared {POLICY_SIGUNGU_REPEATED_GRANULARITY} but varies "
                    f"within policy sigungu: {inconsistent}"
                )
            values = (
                grouped.drop_duplicates(group_column, keep="first")
                .set_index(group_column)["__feature_value"]
                .rename(name)
            )

        scaled = scale_need_direction(
            values,
            method=scaler,
            direction=int(direction),
            transform=spec.get("transform", "identity"),
            lower_quantile=float(options.get("lower_quantile", 0.01)),
            upper_quantile=float(options.get("upper_quantile", 0.99)),
            z_clip_lower=float(options.get("clip_lower", -3.0)),
            z_clip_upper=float(options.get("clip_upper", 3.0)),
            require_complete=spec.get("missing_policy", "error") == "error",
        )
        if repeated_on_admin_dong:
            scaled = master["policy_sigungu_name"].map(scaled)
            scaled.index = master.index
            scaled.name = name
        columns[name] = scaled
    return pd.DataFrame(columns, index=master.index)
