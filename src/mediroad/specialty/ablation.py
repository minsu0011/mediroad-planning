"""Exhaustive Stage 2A specialty-gap sensitivity and ablation runner.

Every variant is compared against the frozen primary score with identical,
label-free policy-stability metrics.  The runner never treats a historical
visit, patient count, or Stage 1 Need score as an accuracy target.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

from mediroad.scoring.transform import scale_need_direction

from .gap import (
    COMPONENT_COLUMNS,
    FORMULA_COLUMNS,
    build_specialty_gap,
    combine_components,
    load_specialty_config,
)


DEFAULT_SCOPES = frozenset(
    {
        "baseline",
        "component",
        "scaler_formula",
        "alpha",
        "source_feature",
        "supply_source",
        "access",
        "leave_one_sigungu_out",
    }
)


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    scope: str
    variant: str
    mode: str
    parameters: Mapping[str, Any]
    expected_admin_count: int


@dataclass
class SpecialtyAblationResult:
    """In-memory, serialization-ready outputs of the exhaustive runner."""

    ledger: pd.DataFrame
    run_summary: pd.DataFrame
    service_metrics: pd.DataFrame
    region_service_metrics: pd.DataFrame
    region_runs: pd.DataFrame
    baseline_scores: pd.DataFrame


def _number_id(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def _component_weights(config: Mapping[str, Any]) -> dict[str, float]:
    raw = config["component_weights"]
    return {
        "health_context_gap_score": float(raw["health_context_gap"]),
        "age_structure_gap_score": float(raw["age_structure_gap"]),
        "specialty_supply_gap_score": float(raw["specialty_supply_gap"]),
        "specialty_excess_access_gap_score": float(raw["specialty_excess_access_gap"]),
        "public_support_gap_score": float(raw["public_support_gap"]),
    }


def _normalised_scopes(scopes: Iterable[str] | None) -> frozenset[str]:
    selected = DEFAULT_SCOPES if scopes is None else frozenset(map(str, scopes))
    unknown = sorted(selected - DEFAULT_SCOPES)
    if unknown:
        raise ValueError(f"Unknown specialty-ablation scopes: {unknown}")
    if "baseline" not in selected:
        selected = frozenset({*selected, "baseline"})
    return selected


def _source_feature_specs(config: Mapping[str, Any]) -> list[tuple[str, str]]:
    result: list[tuple[str, str]] = []
    for kind, key in (
        ("health", "health_features"),
        ("age", "age_features"),
        ("public", "public_support_column"),
    ):
        if kind == "public":
            values = {str(service[key]) for service in config["services"]}
        else:
            values = {
                str(value)
                for service in config["services"]
                for value in service[key]
            }
        result.extend((kind, value) for value in sorted(values))
    return result


def expected_run_specs(
    config: Mapping[str, Any],
    *,
    policy_sigungu_names: Sequence[str],
    scopes: Iterable[str] | None = None,
    facility_prior_grid: Sequence[float] | None = None,
    specialist_prior_grid: Sequence[float] | None = None,
) -> list[RunSpec]:
    """Build the deterministic expected-run ledger before any computation."""

    selected = _normalised_scopes(scopes)
    admin_count = int(config["source"]["expected_admin_dongs"])
    specs: list[RunSpec] = []
    if "baseline" in selected:
        specs.append(RunSpec("baseline::primary", "baseline", "primary", "baseline", {}, admin_count))

    if "component" in selected:
        for component in COMPONENT_COLUMNS:
            for mode in ("renormalize", "neutral"):
                specs.append(
                    RunSpec(
                        f"component::{component}::{mode}",
                        "component",
                        component,
                        mode,
                        {"component": component, "mode": mode},
                        admin_count,
                    )
                )

    if "scaler_formula" in selected:
        for scaler in config["scalers"]:
            for formula in config["formula_families"]:
                specs.append(
                    RunSpec(
                        f"scaler_formula::{scaler}::{formula}",
                        "scaler_formula",
                        f"{scaler}|{formula}",
                        "grid",
                        {"scaler": scaler, "formula": formula},
                        admin_count,
                    )
                )

    if "alpha" in selected:
        supply = config["supply"]
        prior_grid = supply.get("prior_sensitivity_grid", {})
        facility_values = list(
            facility_prior_grid
            if facility_prior_grid is not None
            else prior_grid["facility_share_prior_strength"]
        )
        specialist_values = list(
            specialist_prior_grid
            if specialist_prior_grid is not None
            else prior_grid["specialist_share_prior_strength"]
        )
        for facility in facility_values:
            for specialist in specialist_values:
                specs.append(
                    RunSpec(
                        f"alpha::F{_number_id(facility)}::D{_number_id(specialist)}",
                        "alpha",
                        f"F={float(facility):g}|D={float(specialist):g}",
                        "grid",
                        {
                            "facility_prior_strength": float(facility),
                            "specialist_prior_strength": float(specialist),
                        },
                        admin_count,
                    )
                )

    if "source_feature" in selected:
        for kind, column in _source_feature_specs(config):
            mode = "neutral" if kind == "public" else "loo_with_singleton_neutral"
            specs.append(
                RunSpec(
                    f"source_feature::{kind}::{column}",
                    "source_feature",
                    column,
                    mode,
                    {"source_kind": kind, "source_column": column},
                    admin_count,
                )
            )

    if "supply_source" in selected:
        for variant, mode in (
            ("facility_share", "neutral"),
            ("specialist_share", "neutral"),
            ("facility_share", "substitute_specialist"),
            ("specialist_share", "substitute_facility"),
        ):
            specs.append(
                RunSpec(
                    f"supply_source::{variant}::{mode}",
                    "supply_source",
                    variant,
                    mode,
                    {"supply_source": variant, "mode": mode},
                    admin_count,
                )
            )

    if "access" in selected:
        specs.extend(
            [
                RunSpec(
                    "access::osm_excess::neutral",
                    "access",
                    "osm_excess",
                    "neutral",
                    {"access_source": "osm_excess", "mode": "neutral"},
                    admin_count,
                ),
                RunSpec(
                    "access::absolute_specialty_time::shadow",
                    "access",
                    "absolute_specialty_time",
                    "shadow",
                    {"access_source": "absolute_specialty_time", "mode": "shadow"},
                    admin_count,
                ),
            ]
        )

    if "leave_one_sigungu_out" in selected:
        for sigungu in sorted(map(str, policy_sigungu_names)):
            specs.append(
                RunSpec(
                    f"leave_one_sigungu_out::{sigungu}",
                    "leave_one_sigungu_out",
                    sigungu,
                    "refit_without_block",
                    {"held_out_sigungu": sigungu},
                    -1,
                )
            )

    run_ids = [spec.run_id for spec in specs]
    if len(run_ids) != len(set(run_ids)):
        duplicates = sorted({value for value in run_ids if run_ids.count(value) > 1})
        raise ValueError(f"Expected run IDs are not unique: {duplicates}")
    return specs


def _rerank(scores: pd.DataFrame) -> pd.DataFrame:
    required = {"admin_dong_code", "service_id", "specialty_gap_score"}
    missing = sorted(required - set(scores))
    if missing:
        raise KeyError(f"Cannot rank specialty variant; missing {missing}")
    result = scores.copy()
    if result.duplicated(["admin_dong_code", "service_id"]).any():
        raise ValueError("Variant admin-service key is not unique")
    result["specialty_gap_rank_tied"] = result.groupby("service_id", sort=False)[
        "specialty_gap_score"
    ].rank(method="min", ascending=False)
    result["specialty_gap_rank"] = 0
    for _, indices in result.groupby("service_id", sort=False).groups.items():
        ordered = result.loc[list(indices)].sort_values(
            ["specialty_gap_score", "admin_dong_code"],
            ascending=[False, True],
            kind="stable",
        )
        result.loc[ordered.index, "specialty_gap_rank"] = np.arange(1, len(ordered) + 1)
    result["specialty_gap_rank"] = result["specialty_gap_rank"].astype(int)
    result["specialty_gap_rank_tied"] = result["specialty_gap_rank_tied"].astype(int)
    result["rank_tiebreaker"] = "specialty_gap_score_desc_then_admin_dong_code_asc"
    return result


def _assert_deterministic_rank(scores: pd.DataFrame) -> None:
    reranked = _rerank(scores)
    ordered = ["service_id", "admin_dong_code"]
    observed = scores.sort_values(ordered)
    expected = reranked.sort_values(ordered)
    for column in ("specialty_gap_rank", "specialty_gap_rank_tied"):
        if column not in observed or not np.array_equal(
            observed[column].to_numpy(), expected[column].to_numpy()
        ):
            raise ValueError(
                f"Variant {column} is not the deterministic score/code total order"
            )
    expected_tiebreaker = "specialty_gap_score_desc_then_admin_dong_code_asc"
    if "rank_tiebreaker" not in observed or not observed["rank_tiebreaker"].eq(
        expected_tiebreaker
    ).all():
        raise ValueError("Variant rank_tiebreaker contract is missing or inconsistent")


def _recompute_formula_scores(
    scores: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    weights: Mapping[str, float] | None = None,
    primary_formula: str | None = None,
) -> pd.DataFrame:
    result = scores.copy()
    score_weights = dict(weights or _component_weights(config))
    formula_name = str(primary_formula or config["primary_formula"])
    for component in COMPONENT_COLUMNS:
        contribution = f"contribution_{component.removesuffix('_score')}"
        result[contribution] = (
            result[component] * float(score_weights.get(component, 0.0))
        )
    for formula, options in config["formula_families"].items():
        output = pd.Series(index=result.index, dtype=float)
        # ``rank_composite`` is explicitly a within-service formula.  Applying
        # it to the stacked 16-service table would silently compare unlike
        # specialties, so every formula is recomputed on the same service
        # partitions used by ``build_specialty_gap``.
        for _, indices in result.groupby("service_id", sort=False).groups.items():
            group_indices = list(indices)
            output.loc[group_indices] = combine_components(
                result.loc[group_indices, list(score_weights)],
                score_weights,
                formula=str(formula),
                formula_options=options,
            ).to_numpy()
        result[FORMULA_COLUMNS[str(formula)]] = output
    result["primary_formula"] = formula_name
    result["specialty_gap_score"] = result[FORMULA_COLUMNS[formula_name]]
    return _rerank(result)


def _component_variant(
    baseline: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    component: str,
    mode: str,
) -> pd.DataFrame:
    result = baseline.copy()
    weights = _component_weights(config)
    if mode == "neutral":
        result[component] = 50.0
    elif mode == "renormalize":
        weights.pop(component)
        total = float(sum(weights.values()))
        weights = {key: value / total for key, value in weights.items()}
    else:
        raise ValueError(f"Unknown component-ablation mode: {mode}")
    return _recompute_formula_scores(result, config, weights=weights)


def _source_feature_variant(
    master: pd.DataFrame,
    specialty_pivot: pd.DataFrame,
    facility_admin_join: pd.DataFrame,
    config: Mapping[str, Any],
    baseline: pd.DataFrame,
    *,
    kind: str,
    column: str,
) -> pd.DataFrame:
    modified = deepcopy(dict(config))
    singleton_services: set[str] = set()
    if kind in {"health", "age"}:
        key = f"{kind}_features"
        for service in modified["services"]:
            values = list(map(str, service[key]))
            if column not in values:
                continue
            if len(values) == 1:
                singleton_services.add(str(service["service_id"]))
            else:
                service[key] = [value for value in values if value != column]
        result = build_specialty_gap(
            master, specialty_pivot, facility_admin_join, modified
        )
        if singleton_services:
            component = f"{kind}_context_gap_score" if kind == "health" else "age_structure_gap_score"
            if kind == "health":
                component = "health_context_gap_score"
            result.loc[result["service_id"].isin(singleton_services), component] = 50.0
            result = _recompute_formula_scores(result, config)
        return result
    if kind == "public":
        result = baseline.copy()
        affected = {
            str(service["service_id"])
            for service in config["services"]
            if str(service["public_support_column"]) == column
        }
        result.loc[result["service_id"].isin(affected), "public_support_gap_score"] = 50.0
        return _recompute_formula_scores(result, config)
    raise ValueError(f"Unknown source-feature kind: {kind}")


def _supply_source_variant(
    baseline: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    source: str,
    mode: str,
) -> pd.DataFrame:
    result = baseline.copy()
    if source == "facility_share":
        target = "facility_share_gap_score"
        substitute = "specialist_share_gap_score"
    elif source == "specialist_share":
        target = "specialist_share_gap_score"
        substitute = "facility_share_gap_score"
    else:
        raise ValueError(f"Unknown supply source: {source}")
    if mode == "neutral":
        result[target] = 50.0
    elif mode.startswith("substitute_"):
        result[target] = result[substitute]
    else:
        raise ValueError(f"Unknown supply-source mode: {mode}")
    supply = config["supply"]
    weights = pd.Series(
        {
            "facility_share_gap_score": float(supply["local_facility_share_weight"]),
            "specialist_share_gap_score": float(supply["local_specialist_share_weight"]),
            "sigungu_specialist_rate_gap_score": float(
                supply["sigungu_specialist_rate_weight"]
            ),
        }
    )
    weights /= float(weights.sum())
    result["specialty_supply_gap_score"] = (
        result[list(weights.index)].mul(weights, axis=1).sum(axis=1)
    )
    return _recompute_formula_scores(result, config)


def _access_variant(
    baseline: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    mode: str,
) -> pd.DataFrame:
    result = baseline.copy()
    if mode == "neutral":
        result["specialty_excess_access_gap_score"] = 50.0
    elif mode == "absolute_time_shadow":
        scaler = str(config["primary_scaler"])
        options = dict(config["scalers"][scaler])
        for _, indices in result.groupby("service_id", sort=False).groups.items():
            values = result.loc[list(indices), "specialty_drive_min_v6"].astype(float)
            scaled = scale_need_direction(
                values,
                method=scaler,
                direction=1,
                transform="identity",
                lower_quantile=float(options.get("lower_quantile", 0.01)),
                upper_quantile=float(options.get("upper_quantile", 0.99)),
                z_clip_lower=float(options.get("clip_lower", -3.0)),
                z_clip_upper=float(options.get("clip_upper", 3.0)),
                require_complete=True,
            )
            result.loc[list(indices), "specialty_excess_access_gap_score"] = scaled.to_numpy()
    else:
        raise ValueError(f"Unknown access variant mode: {mode}")
    return _recompute_formula_scores(result, config)


def _deterministic_top_codes(frame: pd.DataFrame, k: int) -> list[str]:
    ordered = frame.sort_values(
        ["specialty_gap_score", "admin_dong_code"],
        ascending=[False, True],
        kind="stable",
    )
    return ordered["admin_dong_code"].astype(str).head(min(k, len(ordered))).tolist()


def _safe_rank_correlation(left: np.ndarray, right: np.ndarray, kind: str) -> float:
    if len(left) < 2:
        return float("nan")
    value = spearmanr(left, right).statistic if kind == "spearman" else kendalltau(left, right).statistic
    return float(value) if value is not None and np.isfinite(value) else float("nan")


def compare_specialty_runs(
    baseline: pd.DataFrame,
    variant: pd.DataFrame,
    *,
    run_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Compare service-wise region ranks and region-wise service portfolios."""

    key = ["admin_dong_code", "service_id"]
    required = {*key, "policy_sigungu_name", "specialty_gap_score"}
    for name, frame in (("baseline", baseline), ("variant", variant)):
        missing = sorted(required - set(frame))
        if missing:
            raise KeyError(f"{name} comparison frame lacks {missing}")
        if frame.duplicated(key).any():
            raise ValueError(f"{name} comparison key is not unique")
    shared = baseline[key].merge(variant[key], on=key, how="inner")
    if shared.empty:
        raise ValueError("Baseline and variant have no shared admin-service rows")
    base = baseline.merge(shared, on=key, how="inner", validate="one_to_one")
    alt = variant.merge(shared, on=key, how="inner", validate="one_to_one")
    base = _rerank(base)
    alt = _rerank(alt)

    service_rows: list[dict[str, Any]] = []
    for service_id in sorted(base["service_id"].unique()):
        left = base[base["service_id"].eq(service_id)].sort_values("admin_dong_code")
        right = alt[alt["service_id"].eq(service_id)].sort_values("admin_dong_code")
        if not np.array_equal(left["admin_dong_code"].to_numpy(), right["admin_dong_code"].to_numpy()):
            raise ValueError(f"Admin alignment failed for service {service_id}")
        base_rank = left["specialty_gap_rank"].to_numpy(dtype=float)
        alt_rank = right["specialty_gap_rank"].to_numpy(dtype=float)
        displacement = np.abs(base_rank - alt_rank)
        base20 = set(_deterministic_top_codes(left, 20))
        alt20 = set(_deterministic_top_codes(right, 20))
        base30 = set(_deterministic_top_codes(left, 30))
        alt30 = set(_deterministic_top_codes(right, 30))
        union = base20 | alt20
        jaccard = len(base20 & alt20) / len(union) if union else 1.0
        forward = len(base20 & alt30) / len(base20) if base20 else 1.0
        reverse = len(alt20 & base30) / len(alt20) if alt20 else 1.0

        base_county = left[left["admin_dong_code"].isin(base20)][
            "policy_sigungu_name"
        ].value_counts(normalize=True)
        alt_county = right[right["admin_dong_code"].isin(alt20)][
            "policy_sigungu_name"
        ].value_counts(normalize=True)
        counties = base_county.index.union(alt_county.index)
        tvd = 0.5 * float(
            np.abs(
                base_county.reindex(counties, fill_value=0).to_numpy()
                - alt_county.reindex(counties, fill_value=0).to_numpy()
            ).sum()
        )
        service_rows.append(
            {
                "run_id": run_id,
                "service_id": service_id,
                "comparison_admin_count": len(left),
                "rank_spearman": _safe_rank_correlation(base_rank, alt_rank, "spearman"),
                "rank_kendall": _safe_rank_correlation(base_rank, alt_rank, "kendall"),
                "top20_jaccard": jaccard,
                "baseline_top20_in_variant_top30": forward,
                "variant_top20_in_baseline_top30": reverse,
                "symmetric_top20_in_top30": (forward + reverse) / 2.0,
                "bidirectional_top20_in_top30_min": min(forward, reverse),
                "rank_displacement_p95": float(np.quantile(displacement, 0.95)),
                "rank_displacement_max": float(displacement.max(initial=0)),
                "sigungu_top20_tvd": tvd,
            }
        )
    service_metrics = pd.DataFrame(service_rows)

    region_rows: list[dict[str, Any]] = []
    shared_admins = sorted(set(base["admin_dong_code"]) & set(alt["admin_dong_code"]))
    for admin_code in shared_admins:
        left = base[base["admin_dong_code"].eq(admin_code)]
        right = alt[alt["admin_dong_code"].eq(admin_code)]
        if set(left["service_id"]) != set(right["service_id"]):
            raise ValueError(f"Service alignment failed for admin {admin_code}")
        left_order = (
            left.sort_values(
                ["specialty_gap_score", "service_id"], ascending=[False, True], kind="stable"
            )["service_id"]
            .astype(str)
            .tolist()
        )
        right_order = (
            right.sort_values(
                ["specialty_gap_score", "service_id"], ascending=[False, True], kind="stable"
            )["service_id"]
            .astype(str)
            .tolist()
        )
        base_top3 = left_order[:3]
        alt_top3 = right_order[:3]
        relevance = {service: 3 - index for index, service in enumerate(base_top3)}
        dcg = sum(
            (2 ** relevance.get(service, 0) - 1) / np.log2(position + 2)
            for position, service in enumerate(alt_top3)
        )
        idcg = sum((2 ** relevance[service] - 1) / np.log2(position + 2) for position, service in enumerate(base_top3))
        region_rows.append(
            {
                "run_id": run_id,
                "admin_dong_code": admin_code,
                "policy_sigungu_name": str(left["policy_sigungu_name"].iloc[0]),
                "service_top1_retained": float(left_order[0] == right_order[0]),
                "service_top3_retention": len(set(base_top3) & set(alt_top3)) / 3.0,
                "service_ndcg_at3": float(dcg / idcg) if idcg > 0 else 1.0,
                "baseline_top1_service": left_order[0],
                "variant_top1_service": right_order[0],
                "baseline_top3_services": "|".join(base_top3),
                "variant_top3_services": "|".join(alt_top3),
            }
        )
    region_metrics = pd.DataFrame(region_rows)

    summary = {
        "comparison_admin_count": float(len(shared_admins)),
        "service_rank_spearman_median": float(service_metrics["rank_spearman"].median()),
        "service_rank_spearman_min": float(service_metrics["rank_spearman"].min()),
        "service_rank_kendall_median": float(service_metrics["rank_kendall"].median()),
        "service_top20_jaccard_median": float(service_metrics["top20_jaccard"].median()),
        "service_top20_jaccard_min": float(service_metrics["top20_jaccard"].min()),
        "service_symmetric_top20_in_top30_median": float(
            service_metrics["symmetric_top20_in_top30"].median()
        ),
        "service_symmetric_top20_in_top30_min": float(
            service_metrics["bidirectional_top20_in_top30_min"].min()
        ),
        "service_rank_displacement_p95_median": float(
            service_metrics["rank_displacement_p95"].median()
        ),
        "service_rank_displacement_max": float(service_metrics["rank_displacement_max"].max()),
        "service_sigungu_top20_tvd_median": float(service_metrics["sigungu_top20_tvd"].median()),
        "service_sigungu_top20_tvd_max": float(service_metrics["sigungu_top20_tvd"].max()),
        "region_service_top1_retention": float(region_metrics["service_top1_retained"].mean()),
        "region_service_top3_retention": float(region_metrics["service_top3_retention"].mean()),
        "region_service_ndcg_at3": float(region_metrics["service_ndcg_at3"].mean()),
    }
    return service_metrics, region_metrics, summary


def _minimal_region_run(scores: pd.DataFrame, spec: RunSpec) -> pd.DataFrame:
    columns = [
        "admin_dong_code",
        "admin_dong_name",
        "policy_sigungu_name",
        "service_id",
        "service_name_ko",
        "service_type",
        "specialty_gap_score",
        "specialty_gap_rank",
        "specialty_gap_rank_tied",
        "scaler",
        "primary_formula",
    ]
    result = scores[columns].copy()
    result.insert(0, "run_id", spec.run_id)
    result.insert(1, "scope", spec.scope)
    result.insert(2, "variant", spec.variant)
    result.insert(3, "mode", spec.mode)
    return result


def run_exhaustive_specialty_ablation(
    master: pd.DataFrame,
    specialty_pivot: pd.DataFrame,
    facility_admin_join: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    baseline_scores: pd.DataFrame | None = None,
    scopes: Iterable[str] | None = None,
    facility_prior_grid: Sequence[float] | None = None,
    specialist_prior_grid: Sequence[float] | None = None,
    fail_closed: bool = True,
) -> SpecialtyAblationResult:
    """Run the configured Stage 2A exhaustive ablation and strict run ledger."""

    selected = _normalised_scopes(scopes)
    baseline = (
        build_specialty_gap(master, specialty_pivot, facility_admin_join, config)
        if baseline_scores is None
        else baseline_scores.copy()
    )
    policy_sigungu_names = sorted(baseline["policy_sigungu_name"].astype(str).unique())
    specs = expected_run_specs(
        config,
        policy_sigungu_names=policy_sigungu_names,
        scopes=selected,
        facility_prior_grid=facility_prior_grid,
        specialist_prior_grid=specialist_prior_grid,
    )
    spec_by_id = {spec.run_id: spec for spec in specs}
    actual: dict[str, pd.DataFrame] = {}
    errors: dict[str, str] = {}

    scaler_cache: dict[str, pd.DataFrame] = {
        str(config["primary_scaler"]): baseline.copy()
    }
    for spec in specs:
        try:
            if spec.scope == "baseline":
                variant = baseline.copy()
            elif spec.scope == "component":
                variant = _component_variant(
                    baseline,
                    config,
                    component=str(spec.parameters["component"]),
                    mode=str(spec.parameters["mode"]),
                )
            elif spec.scope == "scaler_formula":
                scaler = str(spec.parameters["scaler"])
                formula = str(spec.parameters["formula"])
                if scaler not in scaler_cache:
                    scaler_cache[scaler] = build_specialty_gap(
                        master,
                        specialty_pivot,
                        facility_admin_join,
                        config,
                        scaler=scaler,
                    )
                variant = scaler_cache[scaler].copy()
                variant["specialty_gap_score"] = variant[FORMULA_COLUMNS[formula]]
                variant["primary_formula"] = formula
                variant = _rerank(variant)
            elif spec.scope == "alpha":
                facility_prior = float(spec.parameters["facility_prior_strength"])
                specialist_prior = float(spec.parameters["specialist_prior_strength"])
                if np.isclose(facility_prior, float(config["supply"]["facility_share_prior_strength"])) and np.isclose(
                    specialist_prior,
                    float(config["supply"]["specialist_share_prior_strength"]),
                ):
                    variant = baseline.copy()
                else:
                    variant = build_specialty_gap(
                        master,
                        specialty_pivot,
                        facility_admin_join,
                        config,
                        facility_prior_strength=facility_prior,
                        specialist_prior_strength=specialist_prior,
                    )
            elif spec.scope == "source_feature":
                variant = _source_feature_variant(
                    master,
                    specialty_pivot,
                    facility_admin_join,
                    config,
                    baseline,
                    kind=str(spec.parameters["source_kind"]),
                    column=str(spec.parameters["source_column"]),
                )
            elif spec.scope == "supply_source":
                variant = _supply_source_variant(
                    baseline,
                    config,
                    source=str(spec.parameters["supply_source"]),
                    mode=str(spec.parameters["mode"]),
                )
            elif spec.scope == "access":
                variant = _access_variant(
                    baseline,
                    config,
                    mode=(
                        "absolute_time_shadow"
                        if spec.mode == "shadow"
                        else str(spec.parameters["mode"])
                    ),
                )
            elif spec.scope == "leave_one_sigungu_out":
                held_out = str(spec.parameters["held_out_sigungu"])
                reduced_master = master.loc[
                    master["policy_sigungu_name"].astype(str).ne(held_out)
                ].copy()
                reduced = deepcopy(dict(config))
                reduced_admin_count = int(reduced_master["admin_dong_code"].nunique())
                reduced["source"]["expected_admin_dongs"] = reduced_admin_count
                reduced["source"]["expected_policy_sigungu"] = len(policy_sigungu_names) - 1
                reduced["quality_gates"]["expected_specialty_rows"] = reduced_admin_count * len(
                    config["services"]
                )
                reduced["quality_gates"]["expected_primary_focused_rows"] = (
                    reduced_admin_count
                    * int(config["quality_gates"]["expected_focused_physician_specialty_count"])
                )
                reduced["quality_gates"]["expected_adjunct_dental_rows"] = reduced_admin_count
                variant = build_specialty_gap(
                    reduced_master, specialty_pivot, facility_admin_join, reduced
                )
                spec_by_id[spec.run_id] = RunSpec(
                    spec.run_id,
                    spec.scope,
                    spec.variant,
                    spec.mode,
                    spec.parameters,
                    reduced_admin_count,
                )
            else:
                raise ValueError(f"Unhandled ablation scope: {spec.scope}")

            expected_admin = spec_by_id[spec.run_id].expected_admin_count
            expected_rows = expected_admin * len(config["services"])
            if len(variant) != expected_rows:
                raise ValueError(
                    f"{spec.run_id} produced {len(variant)} rows; expected {expected_rows}"
                )
            if variant["admin_dong_code"].nunique() != expected_admin:
                raise ValueError(f"{spec.run_id} admin cardinality mismatch")
            if variant["service_id"].nunique() != len(config["services"]):
                raise ValueError(f"{spec.run_id} service cardinality mismatch")
            if variant["specialty_gap_score"].isna().any() or not np.isfinite(
                variant["specialty_gap_score"].to_numpy()
            ).all():
                raise ValueError(f"{spec.run_id} has incomplete scores")
            _assert_deterministic_rank(variant)
            actual[spec.run_id] = variant
        except Exception as exc:
            errors[spec.run_id] = f"{type(exc).__name__}: {exc}"
            if fail_closed:
                raise RuntimeError(f"Specialty ablation failed at {spec.run_id}") from exc

    expected_ids = set(spec_by_id)
    actual_ids = set(actual)
    if expected_ids != actual_ids:
        missing = sorted(expected_ids - actual_ids)
        extra = sorted(actual_ids - expected_ids)
        raise RuntimeError(f"Specialty ablation run coverage mismatch; missing={missing}, extra={extra}")

    ledger_rows: list[dict[str, Any]] = []
    service_frames: list[pd.DataFrame] = []
    region_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    region_run_frames: list[pd.DataFrame] = []
    for run_id in [spec.run_id for spec in specs]:
        spec = spec_by_id[run_id]
        variant = actual[run_id]
        service_metrics, region_metrics, summary = compare_specialty_runs(
            baseline, variant, run_id=run_id
        )
        service_frames.append(service_metrics)
        region_frames.append(region_metrics)
        summary_rows.append(
            {
                "run_id": run_id,
                "scope": spec.scope,
                "variant": spec.variant,
                "mode": spec.mode,
                **summary,
            }
        )
        region_run_frames.append(_minimal_region_run(variant, spec))
        ledger_rows.append(
            {
                "run_id": run_id,
                "scope": spec.scope,
                "variant": spec.variant,
                "mode": spec.mode,
                "parameters_json": json.dumps(
                    dict(spec.parameters), ensure_ascii=False, sort_keys=True
                ),
                "expected_admin_count": spec.expected_admin_count,
                "expected_score_rows": spec.expected_admin_count * len(config["services"]),
                "actual_admin_count": int(variant["admin_dong_code"].nunique()),
                "actual_score_rows": len(variant),
                "status": "PASS",
                "error": errors.get(run_id, ""),
            }
        )

    ledger = pd.DataFrame(ledger_rows)
    if ledger["run_id"].duplicated().any() or not ledger["status"].eq("PASS").all():
        raise RuntimeError("Specialty ablation ledger is not complete and unique")
    if not ledger["expected_score_rows"].eq(ledger["actual_score_rows"]).all():
        raise RuntimeError("Specialty ablation ledger row counts do not match")
    return SpecialtyAblationResult(
        ledger=ledger,
        run_summary=pd.DataFrame(summary_rows),
        service_metrics=pd.concat(service_frames, ignore_index=True),
        region_service_metrics=pd.concat(region_frames, ignore_index=True),
        region_runs=pd.concat(region_run_frames, ignore_index=True),
        baseline_scores=baseline,
    )


def run_specialty_ablation(
    master: pd.DataFrame,
    specialty_pivot: pd.DataFrame,
    facility_admin_join: pd.DataFrame,
    config: Mapping[str, Any],
    baseline_scores: pd.DataFrame | None = None,
    *,
    bundle_config: Mapping[str, Any] | None = None,
    scopes: Iterable[str] | None = None,
    facility_prior_grid: Sequence[float] | None = None,
    specialist_prior_grid: Sequence[float] | None = None,
    fail_closed: bool = True,
) -> SpecialtyAblationResult:
    """Pipeline-friendly alias with an optional prebuilt primary score.

    ``bundle_config`` is accepted to keep the Stage 2A/B pipeline call site
    stable; specialty-level ablations deliberately do not use bundle labels.
    """

    if bundle_config is not None and not isinstance(bundle_config, Mapping):
        raise TypeError("bundle_config must be a mapping when provided")
    return run_exhaustive_specialty_ablation(
        master,
        specialty_pivot,
        facility_admin_join,
        config,
        baseline_scores=baseline_scores,
        scopes=scopes,
        facility_prior_grid=facility_prior_grid,
        specialist_prior_grid=specialist_prior_grid,
        fail_closed=fail_closed,
    )


def run_exhaustive_specialty_ablation_from_root(
    package_root: str | Path,
    *,
    config_path: str | Path | None = None,
    scopes: Iterable[str] | None = None,
    facility_prior_grid: Sequence[float] | None = None,
    specialist_prior_grid: Sequence[float] | None = None,
    fail_closed: bool = True,
) -> SpecialtyAblationResult:
    """Load approved sources and run the Stage 2A ablation without writing files."""

    root = Path(package_root).resolve()
    path = Path(config_path) if config_path is not None else root / "configs/model_v1/specialty_gap.yaml"
    if not path.is_absolute():
        path = root / path
    config = load_specialty_config(path)
    source = config["source"]
    master = pd.read_csv(
        root / str(source["road_master"]),
        dtype={"admin_dong_code": "string"},
        low_memory=False,
    )
    specialty_pivot = pd.read_csv(
        root / str(source["hira_specialty_pivot"]), low_memory=False
    )
    facility_admin_join = pd.read_csv(
        root / str(source["hira_facility_admin_join"]),
        dtype={"admin_dong_code": "string"},
        low_memory=False,
    )
    return run_exhaustive_specialty_ablation(
        master,
        specialty_pivot,
        facility_admin_join,
        config,
        scopes=scopes,
        facility_prior_grid=facility_prior_grid,
        specialist_prior_grid=specialist_prior_grid,
        fail_closed=fail_closed,
    )
