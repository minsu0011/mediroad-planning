"""Transparent Need Score construction and exhaustive ablation analysis.

This module deliberately evaluates rank and policy-allocation stability.  It
does not report predictive accuracy because MEDIROAD V6 has no complete patient
demand label.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

from .transform import transformed_feature_matrix


@dataclass(frozen=True)
class ScoreRun:
    scores: pd.DataFrame
    feature_matrix: pd.DataFrame
    group_scores: pd.DataFrame


def _normalise_weights(weights: dict[str, float], available: Iterable[str]) -> dict[str, float]:
    available = [x for x in available if x in weights and float(weights[x]) > 0]
    total = sum(float(weights[x]) for x in available)
    if not available or total <= 0:
        raise ValueError("No positive axis weights remain")
    return {x: float(weights[x]) / total for x in available}


def _feature_specs(config: dict) -> list[dict]:
    specs = [dict(x) for x in config.get("features", []) if bool(x.get("model_input", True))]
    for spec in specs:
        spec["name"] = spec.get("name") or spec.get("feature_id")
    names = [x.get("name") for x in specs]
    if not specs or any(not x for x in names):
        raise ValueError("Config must contain non-empty features with name fields")
    if len(names) != len(set(names)):
        raise ValueError("Configured active feature names are not unique")
    for spec in specs:
        if not spec.get("axis") or not spec.get("semantic_group"):
            raise ValueError(f"Feature lacks axis/semantic_group: {spec['name']}")
    return specs


def _profile_weights(config: dict, profile: str) -> dict[str, float]:
    value = config.get("weight_profiles", {}).get(profile)
    if value is None:
        raise KeyError(f"Unknown weight profile: {profile}")
    return dict(value.get("weights", value))


def _scaler_method(name: str) -> str:
    aliases = {"robust_zscore": "robust_z"}
    return aliases.get(name, name)


def _scaler_options(config: dict, name: str) -> dict:
    value = config.get("scalers", {}).get(name, {}) if isinstance(config.get("scalers"), dict) else {}
    return dict(value.get("parameters", value)) if isinstance(value, dict) else {}


def build_need_score(
    master: pd.DataFrame,
    config: dict,
    *,
    scaler: str,
    weight_profile: str,
    removed_features: set[str] | None = None,
    removed_groups: set[str] | None = None,
    removed_axes: set[str] | None = None,
    removal_mode: str = "renormalize",
    substitutions: dict[str, dict] | None = None,
    precomputed_matrix: pd.DataFrame | None = None,
) -> ScoreRun:
    """Build an explainable score with equal semantic-group weight per axis."""

    if removal_mode not in {"renormalize", "neutral"}:
        raise ValueError("removal_mode must be renormalize or neutral")
    removed_features = set(removed_features or ())
    removed_groups = set(removed_groups or ())
    removed_axes = set(removed_axes or ())
    substitutions = substitutions or {}
    specs = _feature_specs(config)

    effective_specs: list[dict] = []
    source = master.copy(deep=False)
    for spec in specs:
        spec = dict(spec)
        active_name = spec["name"]
        if active_name in substitutions:
            replacement = substitutions[active_name]
            shadow_name = replacement["shadow_feature"]
            if shadow_name not in master:
                raise KeyError(f"Substitution feature missing: {shadow_name}")
            source = source.copy()
            source[active_name] = master[shadow_name]
            spec["direction"] = replacement.get("direction", spec["direction"])
            spec["transform"] = replacement.get("transform", spec.get("transform", "identity"))
        effective_specs.append(spec)

    if precomputed_matrix is not None and not substitutions:
        expected = [x["name"] for x in effective_specs]
        if list(precomputed_matrix.columns) != expected or not precomputed_matrix.index.equals(master.index):
            raise ValueError("precomputed_matrix does not match configured features/master index")
        matrix = precomputed_matrix.copy()
    else:
        matrix = transformed_feature_matrix(
            source,
            effective_specs,
            _scaler_method(scaler),
            _scaler_options(config, scaler),
        )
    spec_by_name = {x["name"]: x for x in effective_specs}
    if removal_mode == "neutral":
        for name in removed_features:
            if name in matrix:
                matrix[name] = 50.0
    else:
        matrix = matrix.drop(columns=[x for x in removed_features if x in matrix])

    group_series: dict[str, pd.Series] = {}
    group_axis: dict[str, str] = {}
    for name in matrix.columns:
        spec = spec_by_name[name]
        group = spec["semantic_group"]
        axis = spec["axis"]
        if axis in removed_axes or group in removed_groups:
            continue
        group_axis[group] = axis
    for group, axis in group_axis.items():
        members = [
            name
            for name in matrix.columns
            if spec_by_name[name]["semantic_group"] == group and spec_by_name[name]["axis"] == axis
        ]
        if members:
            group_series[group] = matrix[members].mean(axis=1)
    if not group_series:
        raise ValueError("All feature groups were removed")
    group_scores = pd.DataFrame(group_series, index=master.index)

    axes = sorted(set(group_axis.values()) - removed_axes)
    axis_scores: dict[str, pd.Series] = {}
    for axis in axes:
        groups = [g for g in group_scores if group_axis[g] == axis]
        if groups:
            axis_scores[axis] = group_scores[groups].mean(axis=1)
    profile = _profile_weights(config, weight_profile)
    weights = _normalise_weights(profile, axis_scores)

    output = master[["admin_dong_code", "admin_dong_name", "policy_sigungu_name"]].copy()
    for axis, values in axis_scores.items():
        output[f"{axis}_score"] = values
    output["need_score"] = sum(output[f"{axis}_score"] * weight for axis, weight in weights.items())
    # Preserve the statistical tied rank, but use one explicit total order for
    # policy Top-K membership and every downstream stability metric.  Relying
    # on ``nsmallest`` over tied ranks makes membership depend on input row
    # order, which is unacceptable for a reproducible allocation score.
    output["need_rank_tied"] = output["need_score"].rank(method="min", ascending=False).astype(int)
    ordered_index = (
        output.assign(__admin_code=output["admin_dong_code"].astype(str))
        .sort_values(
            ["need_score", "__admin_code"],
            ascending=[False, True],
            kind="mergesort",
        )
        .index
    )
    deterministic_rank = pd.Series(
        np.arange(1, len(output) + 1, dtype=int),
        index=ordered_index,
    )
    output["need_rank"] = deterministic_rank.reindex(output.index).astype(int)
    output["need_score_tie_size"] = (
        output.groupby("need_score", dropna=False)["need_score"].transform("size").astype(int)
    )
    output["need_rank_tiebreaker"] = "need_score_desc_then_admin_dong_code_asc"
    output["scaler"] = scaler
    output["weight_profile"] = weight_profile
    return ScoreRun(output, matrix, group_scores)


def _gini(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if values.size == 0 or np.allclose(values.sum(), 0):
        return 0.0
    values = np.sort(np.maximum(values, 0))
    n = values.size
    return float((2 * np.sum((np.arange(n) + 1) * values) / (n * values.sum())) - (n + 1) / n)


def _top_codes(scores: pd.DataFrame, k: int) -> set[str]:
    if k < 1 or k > len(scores):
        raise ValueError(f"Top-K must be between 1 and {len(scores)}, got {k}")
    ordered = (
        scores.assign(__admin_code=scores["admin_dong_code"].astype(str))
        .sort_values(["need_rank", "__admin_code"], ascending=[True, True], kind="mergesort")
        .head(k)
    )
    return set(ordered["__admin_code"])


def compare_score_runs(
    baseline: pd.DataFrame,
    variant: pd.DataFrame,
    master: pd.DataFrame,
) -> dict[str, object]:
    """Calculate rank, Top-K, concentration, and coverage stability metrics."""

    left = baseline.set_index("admin_dong_code")
    right = variant.set_index("admin_dong_code").reindex(left.index)
    full_score = left["need_score"].to_numpy(float)
    alt_score = right["need_score"].to_numpy(float)
    full_rank = left["need_rank"].to_numpy(float)
    alt_rank = right["need_rank"].to_numpy(float)
    delta_score = alt_score - full_score
    delta_rank = alt_rank - full_rank
    score_sd = float(np.std(full_score, ddof=1))
    rho = float(spearmanr(full_score, alt_score).statistic)
    tau = float(kendalltau(full_score, alt_score).statistic)
    result: dict[str, object] = {
        "score_mae": float(np.mean(np.abs(delta_score))),
        "score_rmse": float(np.sqrt(np.mean(delta_score**2))),
        "score_rmse_over_full_sd": float(np.sqrt(np.mean(delta_score**2)) / score_sd) if score_sd else 0.0,
        "max_abs_score_shift": float(np.max(np.abs(delta_score))),
        "spearman": rho,
        "kendall": tau,
        "rank_shift_median": float(np.median(np.abs(delta_rank))),
        "rank_shift_p95": float(np.quantile(np.abs(delta_rank), 0.95)),
        "rank_shift_max": float(np.max(np.abs(delta_rank))),
        "mean_abs_rank_shift": float(np.mean(np.abs(delta_rank))),
    }
    for k in (10, 20, 30):
        a, b = _top_codes(baseline, k), _top_codes(variant, k)
        result[f"top{k}_jaccard"] = float(len(a & b) / len(a | b))
        result[f"top{k}_changed"] = int(k - len(a & b))
        if k == 20:
            result["top20_enter"] = "|".join(sorted(b - a))
            result["top20_exit"] = "|".join(sorted(a - b))

    # Exact Top-20 Jaccard is intentionally retained, but a policy boundary can
    # be crowded: a tiny score change may swap ranks 19 and 21 without changing
    # the substantive candidate pool.  This bidirectional containment metric
    # reports whether both Top-20 sets remain inside the other's Top-30 band.
    a20, b20 = _top_codes(baseline, 20), _top_codes(variant, 20)
    a30, b30 = _top_codes(baseline, 30), _top_codes(variant, 30)
    forward = len(a20 & b30) / 20.0
    reverse = len(b20 & a30) / 20.0
    result["top20_bidirectional_recall_at30"] = float(min(forward, reverse))
    result["top20_boundary_contained_at30"] = int(min(forward, reverse) == 1.0)
    baseline_rank = baseline.set_index("admin_dong_code")["need_rank"]
    variant_rank = variant.set_index("admin_dong_code")["need_rank"]
    outward = np.concatenate(
        [
            np.maximum(variant_rank.reindex(sorted(a20)).to_numpy(float) - 20.0, 0.0),
            np.maximum(baseline_rank.reindex(sorted(b20)).to_numpy(float) - 20.0, 0.0),
        ]
    )
    result["top20_outward_displacement_p90"] = float(np.quantile(outward, 0.90))
    result["top20_max_escape_beyond30"] = float(np.maximum(outward - 10.0, 0.0).max())

    meta = master.assign(admin_dong_code=master["admin_dong_code"].astype(str)).set_index("admin_dong_code")
    sigungu = sorted(meta["policy_sigungu_name"].dropna().unique())
    a_counts = meta.loc[list(a20), "policy_sigungu_name"].value_counts().reindex(sigungu, fill_value=0).to_numpy(float)
    b_counts = meta.loc[list(b20), "policy_sigungu_name"].value_counts().reindex(sigungu, fill_value=0).to_numpy(float)
    result["top20_sigungu_tvd"] = float(0.5 * np.abs(a_counts / 20.0 - b_counts / 20.0).sum())
    result["delta_hhi"] = float(np.sum((b_counts / 20.0) ** 2) - np.sum((a_counts / 20.0) ** 2))
    result["delta_gini"] = float(_gini(b_counts) - _gini(a_counts))
    result["top20_sigungu_coverage_delta"] = int(np.count_nonzero(b_counts) - np.count_nonzero(a_counts))

    for column in ("population_65plus", "population_75plus", "population_85plus"):
        if column in meta:
            a_value = float(meta.loc[list(a20), column].sum())
            b_value = float(meta.loc[list(b20), column].sum())
            result[f"top20_{column}_delta"] = b_value - a_value
    if "is_rural_eup_myeon" in meta:
        result["top20_rural_count_delta"] = float(
            meta.loc[list(b20), "is_rural_eup_myeon"].sum() - meta.loc[list(a20), "is_rural_eup_myeon"].sum()
        )
    return result


def _run_id(scope: str, variant: str, mode: str, scaler: str, profile: str) -> str:
    safe = variant.replace(" ", "_").replace("/", "_")
    return f"{scope}__{safe}__{mode}__{scaler}__{profile}"


def run_exhaustive_ablation(
    master: pd.DataFrame,
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run feature, broader-concept cluster, axis, and substitution tests.

    ``semantic_group`` controls score weighting.  ``ablation_cluster`` is a
    separate, broader grouping used only for joint-removal stress tests.  The
    separation prevents a singleton semantic-group design from silently making
    cluster ablation an exact duplicate of feature LOFO.
    """

    specs = _feature_specs(config)
    scalers = list(config.get("scalers", ["winsorized_percentile"]))
    profiles = list(config.get("weight_profiles", {}))
    selection = config.get("feature_selection", {})
    primary_scaler = config.get("primary_scaler", selection.get("default_scaler", scalers[0]))
    primary_profile = config.get(
        "primary_weight_profile", selection.get("default_weight_profile", profiles[0])
    )
    substitutions = list(config.get("substitutions", []))
    cluster_members: dict[str, set[str]] = {}
    for spec in specs:
        cluster = str(spec.get("ablation_cluster") or spec["semantic_group"])
        cluster_members.setdefault(cluster, set()).add(spec["name"])
    clusters = sorted(cluster_members)
    axes = sorted({x["axis"] for x in specs})

    metrics_rows: list[dict] = []
    region_rows: list[pd.DataFrame] = []
    baseline_rows: list[pd.DataFrame] = []
    for scaler in scalers:
        cached_matrix = transformed_feature_matrix(
            master,
            specs,
            _scaler_method(scaler),
            _scaler_options(config, scaler),
        )
        for profile in profiles:
            baseline = build_need_score(
                master,
                config,
                scaler=scaler,
                weight_profile=profile,
                precomputed_matrix=cached_matrix,
            )
            base_scores = baseline.scores
            base_id = _run_id("baseline", "full", "none", scaler, profile)
            base_region = base_scores[["admin_dong_code", "need_score", "need_rank"]].copy()
            base_region.insert(0, "run_id", base_id)
            base_region["scope"] = "baseline"
            base_region["variant"] = "full"
            base_region["top20"] = base_region["need_rank"].le(20).astype(int)
            baseline_rows.append(base_region)

            variants: list[tuple[str, str, str, dict]] = []
            for spec in specs:
                for mode in ("renormalize", "neutral"):
                    variants.append(("feature", spec["name"], mode, {"removed_features": {spec["name"]}}))
            for cluster in clusters:
                variants.append(
                    (
                        "cluster",
                        cluster,
                        "renormalize",
                        {"removed_features": cluster_members[cluster]},
                    )
                )
            for axis in axes:
                variants.append(("axis", axis, "renormalize", {"removed_axes": {axis}}))
            for item in substitutions:
                active_feature = item.get("active_feature") or item.get("selected_feature")
                if not active_feature:
                    raise ValueError(f"Substitution lacks active/selected feature: {item}")
                label = f"{active_feature}=>{item['shadow_feature']}"
                variants.append(
                    (
                        "substitution",
                        label,
                        "renormalize",
                        {"substitutions": {active_feature: item}},
                    )
                )

            for scope, variant, mode, kwargs in variants:
                run = build_need_score(
                    master,
                    config,
                    scaler=scaler,
                    weight_profile=profile,
                    removal_mode=mode,
                    precomputed_matrix=None if "substitutions" in kwargs else cached_matrix,
                    **kwargs,
                )
                run_id = _run_id(scope, variant, mode, scaler, profile)
                metrics = compare_score_runs(base_scores, run.scores, master)
                metrics_rows.append(
                    {
                        "run_id": run_id,
                        "scope": scope,
                        "variant": variant,
                        "removal_mode": mode,
                        "scaler": scaler,
                        "weight_profile": profile,
                        "is_primary": int(scaler == primary_scaler and profile == primary_profile),
                        **metrics,
                    }
                )
                region = run.scores[["admin_dong_code", "need_score", "need_rank"]].copy()
                region.insert(0, "run_id", run_id)
                region["scope"] = scope
                region["variant"] = variant
                region["top20"] = region["need_rank"].le(20).astype(int)
                region_rows.append(region)

    metrics_df = pd.DataFrame(metrics_rows)
    region_df = pd.concat(baseline_rows + region_rows, ignore_index=True)
    if metrics_df.empty:
        raise RuntimeError("Ablation produced no runs")
    metrics_df["normalized_influence_within_config"] = 0.0
    influence_groups = metrics_df.groupby(
        ["scope", "removal_mode", "scaler", "weight_profile"], sort=False
    ).groups
    for idx in influence_groups.values():
        total = float(metrics_df.loc[idx, "mean_abs_rank_shift"].sum())
        if total > 0:
            metrics_df.loc[idx, "normalized_influence_within_config"] = (
                metrics_df.loc[idx, "mean_abs_rank_shift"] / total
            )

    summary_rows: list[dict] = []
    for (scope, variant, mode), group in metrics_df.groupby(["scope", "variant", "removal_mode"], sort=True):
        primary = group[group["is_primary"].eq(1)]
        default_profile = group[group["weight_profile"].eq(primary_profile)]
        influence = float(group["mean_abs_rank_shift"].mean())
        summary_rows.append(
            {
                "scope": scope,
                "variant": variant,
                "removal_mode": mode,
                "run_count": int(len(group)),
                "primary_spearman": float(primary["spearman"].iloc[0]) if len(primary) else np.nan,
                "primary_kendall": float(primary["kendall"].iloc[0]) if len(primary) else np.nan,
                "primary_top20_jaccard": float(primary["top20_jaccard"].iloc[0]) if len(primary) else np.nan,
                "primary_top20_changed": int(primary["top20_changed"].iloc[0]) if len(primary) else np.nan,
                "primary_top20_bidirectional_recall_at30": float(
                    primary["top20_bidirectional_recall_at30"].iloc[0]
                ) if len(primary) else np.nan,
                "primary_rank_shift_p95": float(primary["rank_shift_p95"].iloc[0]) if len(primary) else np.nan,
                "default_profile_run_count": int(len(default_profile)),
                "default_profile_worst_spearman": float(default_profile["spearman"].min()),
                "default_profile_worst_kendall": float(default_profile["kendall"].min()),
                "default_profile_worst_top20_changed": int(default_profile["top20_changed"].max()),
                "default_profile_worst_top20_jaccard": float(default_profile["top20_jaccard"].min()),
                "default_profile_worst_top20_bidirectional_recall_at30": float(
                    default_profile["top20_bidirectional_recall_at30"].min()
                ),
                "default_profile_worst_rank_shift_p95": float(default_profile["rank_shift_p95"].max()),
                "default_profile_worst_top20_outward_displacement_p90": float(
                    default_profile["top20_outward_displacement_p90"].max()
                ),
                "default_profile_worst_top20_max_escape_beyond30": float(
                    default_profile["top20_max_escape_beyond30"].max()
                ),
                "default_profile_worst_top20_sigungu_tvd": float(
                    default_profile["top20_sigungu_tvd"].max()
                ),
                "default_profile_max_normalized_influence_within_config": float(
                    default_profile["normalized_influence_within_config"].max()
                ),
                "default_profile_min_normalized_influence_within_config": float(
                    default_profile["normalized_influence_within_config"].min()
                ),
                "all_profiles_max_normalized_influence_within_config": float(
                    group["normalized_influence_within_config"].max()
                ),
                "median_spearman": float(group["spearman"].median()),
                "worst_spearman": float(group["spearman"].min()),
                "worst_kendall": float(group["kendall"].min()),
                "median_top20_jaccard": float(group["top20_jaccard"].median()),
                "worst_top20_jaccard": float(group["top20_jaccard"].min()),
                "worst_top20_changed": int(group["top20_changed"].max()),
                "worst_top20_bidirectional_recall_at30": float(
                    group["top20_bidirectional_recall_at30"].min()
                ),
                "worst_top20_outward_displacement_p90": float(
                    group["top20_outward_displacement_p90"].max()
                ),
                "worst_top20_max_escape_beyond30": float(group["top20_max_escape_beyond30"].max()),
                "worst_top20_sigungu_tvd": float(group["top20_sigungu_tvd"].max()),
                "worst_rank_shift_p95": float(group["rank_shift_p95"].max()),
                "worst_rank_shift_max": float(group["rank_shift_max"].max()),
                "mean_abs_rank_shift_across_configs": influence,
            }
        )
    summary = pd.DataFrame(summary_rows)
    for (_scope, _mode), idx in summary.groupby(["scope", "removal_mode"]).groups.items():
        total = float(summary.loc[idx, "mean_abs_rank_shift_across_configs"].sum())
        summary.loc[idx, "normalized_influence"] = (
            summary.loc[idx, "mean_abs_rank_shift_across_configs"] / total if total > 0 else 0.0
        )
    return metrics_df, summary, region_df


def ablation_quality_gates(summary: pd.DataFrame) -> pd.DataFrame:
    """Apply completion and stress-diagnostic thresholds.

    Alternative policy profiles are deliberately different policy choices, so
    exact Top-20 identity across *every* profile/scaler is a stress diagnostic,
    not a valid completion requirement.  Completion gates focus on the declared
    default policy profile across all four scalers, normalized influence caps,
    tolerant Top-30 boundary containment, and the intended contribution of
    every axis.  The
    stricter worst-case exact-Jaccard checks remain visible as diagnostics.
    """

    checks: list[tuple[str, bool, str]] = []

    def add(name: str, passed: bool, gate_class: str = "completion") -> None:
        checks.append((name, bool(passed), gate_class))

    feature_all = summary[summary["scope"] == "feature"]
    feature = feature_all[feature_all["removal_mode"] == "renormalize"]
    add("feature_default4_spearman_ge_0_97", (feature_all["default_profile_worst_spearman"] >= 0.97).all())
    add("feature_default4_kendall_ge_0_90", (feature_all["default_profile_worst_kendall"] >= 0.90).all())
    add("feature_default4_top20_changed_le_3", (feature_all["default_profile_worst_top20_changed"] <= 3).all())
    add(
        "feature_default4_top20_recall_at30_ge_0_95",
        (feature_all["default_profile_worst_top20_bidirectional_recall_at30"] >= 0.95).all(),
    )
    add("feature_default4_p95_rank_shift_le_15", (feature_all["default_profile_worst_rank_shift_p95"] <= 15).all())
    add(
        "feature_default4_top20_outward_p90_le_5",
        (feature_all["default_profile_worst_top20_outward_displacement_p90"] <= 5).all(),
    )
    add(
        "feature_default4_max_escape_beyond30_le_10",
        (feature_all["default_profile_worst_top20_max_escape_beyond30"] <= 10).all(),
    )
    add("feature_default4_sigungu_tvd_le_0_15", (feature_all["default_profile_worst_top20_sigungu_tvd"] <= 0.1500001).all())
    add(
        "feature_default4_normalized_influence_le_0_10",
        feature_all["default_profile_max_normalized_influence_within_config"].max() <= 0.10,
    )
    add(
        "feature_any_profile_normalized_influence_le_0_20",
        feature_all["all_profiles_max_normalized_influence_within_config"].max() <= 0.20,
    )
    add("feature_all_profiles_top20_recall_at30_ge_0_90", (feature_all["worst_top20_bidirectional_recall_at30"] >= 0.90).all())
    add("feature_mean_normalized_influence_le_0_10", feature_all["normalized_influence"].max() <= 0.10)
    add("feature_worst_spearman_ge_0_97", (feature["worst_spearman"] >= 0.97).all(), "stress_diagnostic")
    add("feature_worst_top20_jaccard_ge_0_82", (feature["worst_top20_jaccard"] >= 0.82).all(), "stress_diagnostic")
    add("feature_worst_p95_rank_shift_le_10", (feature["worst_rank_shift_p95"] <= 10).all(), "stress_diagnostic")

    cluster = summary[summary["scope"] == "cluster"]
    add("cluster_default4_spearman_ge_0_90", (cluster["default_profile_worst_spearman"] >= 0.90).all())
    add("cluster_default4_top20_recall_at30_ge_0_85", (cluster["default_profile_worst_top20_bidirectional_recall_at30"] >= 0.85).all())
    add("cluster_default4_p95_rank_shift_le_25", (cluster["default_profile_worst_rank_shift_p95"] <= 25).all())
    add("cluster_default4_sigungu_tvd_le_0_25", (cluster["default_profile_worst_top20_sigungu_tvd"] <= 0.2500001).all())
    add("cluster_default4_max_escape_beyond30_le_20", (cluster["default_profile_worst_top20_max_escape_beyond30"] <= 20).all())
    add(
        "cluster_default4_normalized_influence_le_0_30",
        cluster["default_profile_max_normalized_influence_within_config"].max() <= 0.30,
    )
    add("cluster_mean_normalized_influence_le_0_30", cluster["normalized_influence"].max() <= 0.30)
    add("cluster_worst_top20_jaccard_ge_0_60", (cluster["worst_top20_jaccard"] >= 0.60).all(), "stress_diagnostic")
    add("cluster_worst_p95_rank_shift_le_25", (cluster["worst_rank_shift_p95"] <= 25).all(), "stress_diagnostic")

    substitution = summary[summary["scope"] == "substitution"]
    if len(substitution):
        add("substitution_worst_spearman_ge_0_95", (substitution["worst_spearman"] >= 0.95).all())
        add("substitution_worst_p95_rank_shift_le_15", (substitution["worst_rank_shift_p95"] <= 15).all())
        add("substitution_worst_top20_recall_at30_ge_0_90", (substitution["worst_top20_bidirectional_recall_at30"] >= 0.90).all())
        add("substitution_worst_max_escape_beyond30_le_10", (substitution["worst_top20_max_escape_beyond30"] <= 10).all())
        add(
            "substitution_worst_sigungu_tvd_le_0_10",
            (substitution["worst_top20_sigungu_tvd"] <= 0.1000001).all(),
            "stress_diagnostic",
        )
        add("substitution_worst_top20_jaccard_ge_0_75", (substitution["worst_top20_jaccard"] >= 0.75).all(), "stress_diagnostic")

    axis = summary[summary["scope"] == "axis"]
    if len(axis):
        add(
            "axis_default4_influence_each_between_0_05_and_0_40",
            (axis["default_profile_min_normalized_influence_within_config"] >= 0.05).all()
            and (axis["default_profile_max_normalized_influence_within_config"] <= 0.40).all(),
        )
        add("axis_default4_top20_jaccard_ge_0_40", (axis["default_profile_worst_top20_jaccard"] >= 0.40).all())
        add("axis_default4_top20_recall_at30_ge_0_60", (axis["default_profile_worst_top20_bidirectional_recall_at30"] >= 0.60).all())
        add("axis_default4_sigungu_tvd_le_0_35", (axis["default_profile_worst_top20_sigungu_tvd"] <= 0.3500001).all())
        add("axis_each_non_inert_in_primary", (axis["primary_top20_changed"] >= 1).all())
        add(
            "axis_worst_top20_jaccard_ge_0_40",
            (axis["worst_top20_jaccard"] >= 0.40).all(),
            "stress_diagnostic",
        )
        add(
            "axis_at_least_three_change_primary_top20_by_2",
            (axis["primary_top20_changed"] >= 2).sum() >= 3,
        )
    rows: list[dict] = []
    for name, passed, gate_class in checks:
        rows.append(
            {
                "gate": name,
                "gate_class": gate_class,
                "required_for_completion": int(gate_class == "completion"),
                "passed": int(passed),
                "status": "PASS" if passed else "FAIL",
            }
        )
    return pd.DataFrame(rows)


def certify_weight_profiles(
    metrics: pd.DataFrame,
    *,
    default_profile: str,
    expected_scalers: Iterable[str],
) -> pd.DataFrame:
    """Certify scaler robustness separately for each policy weight profile.

    Certification is deliberately fail-closed: a profile must contain a
    complete, duplicate-free scaler grid and at least one feature ablation.
    This prevents pandas' vacuous ``all()`` on an empty slice from producing a
    false ``CERTIFIED`` result.
    """

    expected_scaler_set = {str(value) for value in expected_scalers}
    if not expected_scaler_set:
        raise ValueError("expected_scalers must contain at least one scaler")
    required_columns = {
        "scope",
        "variant",
        "removal_mode",
        "scaler",
        "weight_profile",
        "spearman",
        "kendall",
        "top20_changed",
        "top20_bidirectional_recall_at30",
        "rank_shift_p95",
        "top20_outward_displacement_p90",
        "top20_max_escape_beyond30",
        "top20_sigungu_tvd",
        "normalized_influence_within_config",
    }
    missing_columns = sorted(required_columns - set(metrics.columns))
    if missing_columns:
        raise ValueError(f"Profile certification metrics lack columns: {missing_columns}")
    observed_profiles = set(metrics["weight_profile"].astype(str))
    if default_profile not in observed_profiles:
        raise ValueError(f"Default profile is absent from ablation metrics: {default_profile}")

    rows: list[dict[str, object]] = []
    for profile, profile_rows in metrics.groupby("weight_profile", sort=True):
        feature = profile_rows[profile_rows["scope"].eq("feature")]
        substitution = profile_rows[profile_rows["scope"].eq("substitution")]
        observed_scalers = set(profile_rows["scaler"].astype(str))
        grid_sizes = profile_rows.groupby(
            ["scope", "variant", "removal_mode"], dropna=False
        )["scaler"].agg(["size", "nunique"])
        grid_complete = bool(
            observed_scalers == expected_scaler_set
            and len(grid_sizes)
            and grid_sizes["size"].eq(len(expected_scaler_set)).all()
            and grid_sizes["nunique"].eq(len(expected_scaler_set)).all()
        )
        metric_columns = sorted(required_columns - {
            "scope", "variant", "removal_mode", "scaler", "weight_profile"
        })
        finite_metrics = bool(
            len(profile_rows)
            and np.isfinite(profile_rows[metric_columns].to_numpy(dtype=float)).all()
        )
        influence_limit = 0.10 if profile == default_profile else 0.20
        checks = {
            "feature_rows_nonempty": bool(len(feature)),
            "complete_expected_scaler_grid": grid_complete,
            "required_metrics_finite": finite_metrics,
            "feature_spearman_ge_0_97": bool(feature["spearman"].ge(0.97).all()),
            "feature_kendall_ge_0_90": bool(feature["kendall"].ge(0.90).all()),
            "feature_top20_changed_le_3": bool(feature["top20_changed"].le(3).all()),
            "feature_top20_recall_at30_ge_0_95": bool(
                feature["top20_bidirectional_recall_at30"].ge(0.95).all()
            ),
            "feature_rank_shift_p95_le_15": bool(feature["rank_shift_p95"].le(15).all()),
            "feature_outward_displacement_p90_le_5": bool(
                feature["top20_outward_displacement_p90"].le(5).all()
            ),
            "feature_max_escape_beyond30_le_10": bool(
                feature["top20_max_escape_beyond30"].le(10).all()
            ),
            "feature_sigungu_tvd_le_0_15": bool(feature["top20_sigungu_tvd"].le(0.1500001).all()),
            f"feature_config_influence_le_{influence_limit:.2f}": bool(
                feature["normalized_influence_within_config"].le(influence_limit).all()
            ),
        }
        if len(substitution):
            checks.update(
                {
                    "substitution_spearman_ge_0_95": bool(substitution["spearman"].ge(0.95).all()),
                    "substitution_rank_shift_p95_le_15": bool(substitution["rank_shift_p95"].le(15).all()),
                    "substitution_top20_recall_at30_ge_0_90": bool(
                        substitution["top20_bidirectional_recall_at30"].ge(0.90).all()
                    ),
                    "substitution_max_escape_beyond30_le_10": bool(
                        substitution["top20_max_escape_beyond30"].le(10).all()
                    ),
                }
            )
        failed = [name for name, passed in checks.items() if not passed]
        rows.append(
            {
                "weight_profile": profile,
                "is_default_profile": int(profile == default_profile),
                "scaler_count": int(len(observed_scalers)),
                "expected_scaler_count": int(len(expected_scaler_set)),
                "check_count": len(checks),
                "passed_count": sum(checks.values()),
                "profile_certified": int(not failed),
                "status": "CERTIFIED" if not failed else "EXPLORATORY_FRAGILE",
                "failed_checks": "|".join(failed),
                "max_substitution_sigungu_tvd_advisory": (
                    float(substitution["top20_sigungu_tvd"].max()) if len(substitution) else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def axis_priority_responsiveness(
    metrics: pd.DataFrame,
    *,
    default_profile: str,
    priority_profile_by_axis: dict[str, str],
    expected_scalers: Iterable[str],
) -> pd.DataFrame:
    """Check scaler-paired response to removal of a profile's priority axis."""

    axis_rows = metrics[metrics["scope"].eq("axis")].copy()
    expected_scaler_set = {str(value) for value in expected_scalers}
    if not expected_scaler_set:
        raise ValueError("expected_scalers must contain at least one scaler")
    rows: list[dict[str, object]] = []
    for axis, priority_profile in priority_profile_by_axis.items():
        baseline = axis_rows[
            axis_rows["variant"].eq(axis) & axis_rows["weight_profile"].eq(default_profile)
        ]
        priority = axis_rows[
            axis_rows["variant"].eq(axis) & axis_rows["weight_profile"].eq(priority_profile)
        ]
        if baseline.empty or priority.empty:
            raise ValueError(f"Missing axis/profile responsiveness rows for {axis}: {priority_profile}")
        if baseline["scaler"].duplicated().any() or priority["scaler"].duplicated().any():
            raise ValueError(f"Duplicate scaler rows for axis responsiveness: {axis}")
        if set(baseline["scaler"].astype(str)) != expected_scaler_set or set(
            priority["scaler"].astype(str)
        ) != expected_scaler_set:
            raise ValueError(f"Incomplete scaler grid for axis responsiveness: {axis}")
        paired = baseline[["scaler", "mean_abs_rank_shift"]].merge(
            priority[["scaler", "mean_abs_rank_shift"]],
            on="scaler",
            how="inner",
            validate="one_to_one",
            suffixes=("_default", "_priority"),
        )
        if not np.isfinite(
            paired[["mean_abs_rank_shift_default", "mean_abs_rank_shift_priority"]].to_numpy(float)
        ).all():
            raise ValueError(f"Non-finite axis responsiveness impact: {axis}")
        baseline_impact = float(paired["mean_abs_rank_shift_default"].mean())
        priority_impact = float(paired["mean_abs_rank_shift_priority"].mean())
        paired_pass = paired["mean_abs_rank_shift_priority"].ge(
            paired["mean_abs_rank_shift_default"] - 1e-12
        )
        ratios = np.divide(
            paired["mean_abs_rank_shift_priority"].to_numpy(float),
            paired["mean_abs_rank_shift_default"].to_numpy(float),
            out=np.full(len(paired), np.inf),
            where=paired["mean_abs_rank_shift_default"].to_numpy(float) > 0,
        )
        rows.append(
            {
                "axis": axis,
                "default_profile": default_profile,
                "priority_profile": priority_profile,
                "default_mean_abs_rank_shift": baseline_impact,
                "priority_mean_abs_rank_shift": priority_impact,
                "impact_ratio_priority_over_default": (
                    priority_impact / baseline_impact if baseline_impact > 0 else np.nan
                ),
                "minimum_scaler_impact_ratio": float(np.min(ratios)),
                "failing_scalers": "|".join(paired.loc[~paired_pass, "scaler"].astype(str)),
                "scaler_count": int(len(paired)),
                "monotonic_responsiveness_pass": int(paired_pass.all()),
            }
        )
    return pd.DataFrame(rows)
