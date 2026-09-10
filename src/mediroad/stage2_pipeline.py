"""Reproducible MEDIROAD MODEL V1 Stage 2A/2B build and quality gate.

This entry point deliberately stops before Stage 3.  Specialty Gap and
Temporal Fit are transparent policy MCDA layers, not patient-demand,
attendance, causal-effect, or individual-risk predictions.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
import platform
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import yaml

from mediroad.reporting.correlation import (
    block_bootstrap_quality,
    plot_correlation_map,
    save_interactive_heatmap,
)
from mediroad.reporting.stage2_correlation import (
    build_specialty_bundle_atlas,
    render_temporal_month_profile_heatmap,
    same_granularity_correlation,
    stabilize_feature_manifest,
)
from mediroad.specialty import (
    COMPONENT_COLUMNS,
    FORMULA_COLUMNS,
    build_bundle_scores,
    build_specialty_gap_from_root,
    load_service_bundle_config,
    load_specialty_config,
    validate_specialty_results,
)
from mediroad.specialty.ablation import expected_run_specs, run_specialty_ablation
from mediroad.temporal import (
    aggregate_monthly_fit_to_season,
    apply_season_release_contract,
    assess_monthly_resolution,
    build_bundle_seasonality,
    build_bundle_seasonality_from_yearly_evidence,
    build_monthly_climate_risk,
    build_region_bundle_month_fit,
    build_temporal_data_audit,
    build_working_day_calendar,
    calculate_service_seasonality,
    climate_leave_one_year_out_stability,
    read_nhis_monthly_cp949,
    run_temporal_ablation,
    select_primary_fallback_seasons,
    temporal_audit_row,
)


FROZEN_CANONICAL = Path("derived/mediroad_admin_dong_master_v6_final.csv")
SPECIALTY_OUTPUT = Path("outputs/model_v1/04_specialty")
TEMPORAL_OUTPUT = Path("outputs/model_v1/04_temporal")
STAGE2_OUTPUT = Path("outputs/model_v1/04_stage2")
STAGE2_REPORT = Path("reports/model_v1/stage2")
WEATHER_UNITS = {
    "temperature_2m_max": "C",
    "temperature_2m_min": "C",
    "precipitation_sum": "mm/day",
    "wind_speed_10m_max": "m/s",
}
CORRELATION_BLOCK = {
    "internal_medicine": "chronic_primary",
    "family_medicine": "chronic_primary",
    "orthopedics": "musculoskeletal_rehab",
    "rehabilitation": "musculoskeletal_rehab",
    "neurosurgery": "musculoskeletal_rehab",
    "anesthesiology_pain": "musculoskeletal_rehab",
    "neurology": "neuro_mental",
    "psychiatry": "neuro_mental",
    "ophthalmology": "sensory_oral_skin",
    "otolaryngology": "sensory_oral_skin",
    "dermatology": "sensory_oral_skin",
    "dental": "sensory_oral_skin",
    "obstetrics_gynecology": "basic_referral",
    "emergency_medicine": "basic_referral",
    "general_surgery": "basic_referral",
    "urology": "basic_referral",
}
SEASON_MONTHS = {
    "winter": (12, 1, 2),
    "spring": (3, 4, 5),
    "summer": (6, 7, 8),
    "autumn": (9, 10, 11),
}


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot JSON-encode {type(value)!r}")


def _atomic_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.tmp-{os.getpid()}")


def write_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _atomic_path(path)
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    temporary.replace(path)


def write_text(value: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _atomic_path(path)
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def write_csv(frame: pd.DataFrame, path: Path, *, index: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _atomic_path(path)
    frame.to_csv(temporary, index=index, encoding="utf-8-sig")
    temporary.replace(path)


def write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _atomic_path(path)
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stage2_artifact_inventory(root: Path) -> pd.DataFrame:
    """Hash materialized Stage 2 artifacts, excluding the inventory itself."""

    roots = [
        root / SPECIALTY_OUTPUT,
        root / TEMPORAL_OUTPUT,
        root / STAGE2_OUTPUT,
        root / STAGE2_REPORT,
    ]
    direct = [
        root / "reports/model_v1/04_specialty_gap.md",
        root / "reports/model_v1/05_temporal_layer.md",
    ]
    excluded = {
        (root / STAGE2_OUTPUT / "STAGE2_ARTIFACT_INVENTORY.csv").resolve(),
        (root / STAGE2_OUTPUT / "STAGE2_RUN_METADATA.json").resolve(),
    }
    paths: set[Path] = set()
    for directory in roots:
        if directory.exists():
            paths.update(path.resolve() for path in directory.rglob("*") if path.is_file())
    paths.update(path.resolve() for path in direct if path.is_file())
    rows = []
    for path in sorted(paths - excluded, key=lambda value: value.as_posix()):
        rows.append(
            {
                "relative_path": path.relative_to(root).as_posix(),
                "size_bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )
    if not rows:
        raise RuntimeError("Stage 2 artifact inventory is empty")
    inventory = pd.DataFrame(rows)
    inventory["content_alias_count"] = inventory.groupby("sha256")[
        "relative_path"
    ].transform("size")
    canonical_by_hash: dict[str, str] = {}
    for digest, group in inventory.groupby("sha256", sort=False):
        ordered = sorted(
            group["relative_path"].astype(str),
            key=lambda value: (
                0 if value.startswith("outputs/") else 1,
                len(value),
                value,
            ),
        )
        canonical_by_hash[str(digest)] = ordered[0]
    inventory["canonical_content_path"] = inventory["sha256"].map(
        canonical_by_hash
    )
    inventory["is_content_alias"] = (
        inventory["relative_path"] != inventory["canonical_content_path"]
    )
    return inventory


def _manifest(
    features: Iterable[str],
    *,
    labels: Mapping[str, str] | None = None,
    blocks: Mapping[str, str] | None = None,
    granularity: str,
    roles: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    labels = labels or {}
    blocks = blocks or {}
    roles = roles or {}
    rows = []
    for order, feature in enumerate(features, start=1):
        feature_text = str(feature)
        if "::" in feature_text:
            service_id, feature_family = feature_text.split("::", 1)
        elif feature_text in CORRELATION_BLOCK:
            service_id, feature_family = feature_text, "specialty_gap_score"
        else:
            service_id, feature_family = "", str(blocks.get(feature, "unassigned"))
        rows.append(
            {
                # The identifier is assigned before any hierarchical ordering.
                # Rendering code must preserve it rather than renumbering the
                # clustered matrix position.
                "feature_id": f"F{order:04d}",
                "source_column": feature_text,
                "label_ko": str(labels.get(feature, feature)),
                "granularity": granularity,
                "axis": str(blocks.get(feature, "unassigned")),
                "bundle_id": str(blocks.get(feature, "unassigned")),
                "role": str(roles.get(feature, "model_output")),
                "service_id": service_id,
                "clinical_bundle_id": str(
                    CORRELATION_BLOCK.get(service_id, "not_applicable")
                ),
                "feature_family": feature_family,
                "proxy": False,
                "display_order": order,
            }
        )
    return pd.DataFrame(rows)


def _pivot_long(
    frame: pd.DataFrame,
    value_columns: Iterable[str],
    *,
    feature_key: str = "service_id",
) -> pd.DataFrame:
    pieces = []
    for value in value_columns:
        pivot = frame.pivot(index="admin_dong_code", columns=feature_key, values=value)
        pivot.columns = [f"{service}::{value}" for service in pivot.columns]
        pieces.append(pivot)
    return pd.concat(pieces, axis=1)


def _select_discriminating_atlas_features(
    full_wide: pd.DataFrame,
    full_manifest: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Remove only documented formula lineage and exact semantic aliases.

    The complete 160-column atlas remains the lineage audit.  This companion
    view improves visual separation without deleting evidence from the model:
    raw excess-access inputs defer to their normalized component, and exact
    component aliases sharing one explicit lineage keep their first stable ID.
    """

    meta = full_manifest.set_index("source_column", drop=False).reindex(
        full_wide.columns
    )
    if meta["feature_id"].isna().any() or not meta["feature_id"].is_unique:
        raise ValueError("Representative atlas requires complete stable feature IDs")
    selection = meta.reset_index(drop=True).copy()
    selection["included_in_discriminating_atlas"] = True
    selection["representative_column"] = selection["source_column"]
    selection["selection_reason"] = "retained_distinct_feature"

    for service_id in sorted(
        value for value in selection["service_id"].astype(str).unique() if value
    ):
        raw_column = f"{service_id}::specialty_access_excess_log1p"
        component_column = f"{service_id}::specialty_excess_access_gap_score"
        if raw_column in meta.index and component_column in meta.index:
            mask = selection["source_column"].eq(raw_column)
            selection.loc[mask, "included_in_discriminating_atlas"] = False
            selection.loc[mask, "representative_column"] = component_column
            selection.loc[mask, "selection_reason"] = (
                "deterministic_monotone_raw_lineage_represented_by_normalized_component"
            )

    retained_representatives: list[str] = []
    for column in full_wide.columns:
        row_index = selection.index[selection["source_column"].eq(column)][0]
        if not bool(selection.at[row_index, "included_in_discriminating_atlas"]):
            continue
        lineage_key = str(meta.loc[column, "equivalence_key"])
        duplicate_of: str | None = None
        for representative in retained_representatives:
            if str(meta.loc[representative, "equivalence_key"]) != lineage_key:
                continue
            if np.allclose(
                full_wide[column].to_numpy(float),
                full_wide[representative].to_numpy(float),
                rtol=0.0,
                atol=1e-12,
                equal_nan=True,
            ):
                duplicate_of = representative
                break
        if duplicate_of is None:
            retained_representatives.append(str(column))
        else:
            selection.at[row_index, "included_in_discriminating_atlas"] = False
            selection.at[row_index, "representative_column"] = duplicate_of
            selection.at[row_index, "selection_reason"] = (
                "exact_value_alias_same_declared_lineage"
            )

    retained = selection.loc[
        selection["included_in_discriminating_atlas"], "source_column"
    ].astype(str).tolist()
    representative_wide = full_wide[retained].copy()
    representative_manifest = full_manifest.loc[
        full_manifest["source_column"].isin(retained)
    ].copy()
    representative_manifest = representative_manifest.set_index(
        "source_column", drop=False
    ).reindex(retained).reset_index(drop=True)
    if representative_wide.shape[1] != len(representative_manifest):
        raise RuntimeError("Discriminating atlas selection coverage mismatch")
    return representative_wide, representative_manifest, selection


def _review_high_correlation_ledger(
    ledger: pd.DataFrame,
    *,
    atlas_name: str,
) -> pd.DataFrame:
    """Turn every high-correlation edge into an explicit review decision."""

    reviewed = ledger.copy()
    if reviewed.empty:
        for column, default in (
            ("atlas_name", atlas_name),
            ("review_status", "not_applicable_no_high_correlation_pairs"),
            ("unexplained_after_review", False),
        ):
            reviewed[column] = pd.Series(dtype=type(default))
        return reviewed
    reviewed.insert(0, "atlas_name", atlas_name)
    reviewed["review_status"] = np.where(
        reviewed["expected_redundancy"].astype(bool),
        "explained_structural_or_deterministic_lineage",
        "unreviewed_high_correlation",
    )
    key_a = reviewed["semantic_key_a"].astype(str)
    key_b = reviewed["semantic_key_b"].astype(str)
    distinct_public_proxy = (
        ~reviewed["expected_redundancy"].astype(bool)
        & key_a.str.contains("public_support_gap_score", regex=False)
        & key_b.str.contains("public_support_gap_score", regex=False)
        & (
            (
                key_a.str.contains("dementia_home_visit", regex=False)
                & key_b.str.contains("rehabilitation_medical_institution", regex=False)
            )
            | (
                key_b.str.contains("dementia_home_visit", regex=False)
                & key_a.str.contains("rehabilitation_medical_institution", regex=False)
            )
        )
    )
    reviewed.loc[distinct_public_proxy, "review_status"] = (
        "reviewed_distinct_source_empirical_collinearity_not_duplicate"
    )
    reviewed.loc[distinct_public_proxy, "explanation"] = (
        "Distinct dementia-home-visit and rehabilitation-access proxies share "
        "a rural geography surface; retained as separate meanings and flagged "
        "for sensitivity review."
    )
    reviewed["unexplained_after_review"] = reviewed["review_status"].eq(
        "unreviewed_high_correlation"
    )
    return reviewed


def _service_metadata(config: Mapping[str, Any]) -> tuple[dict[str, str], dict[str, str]]:
    labels = {str(row["service_id"]): str(row["name_ko"]) for row in config["services"]}
    blocks = {
        service_id: CORRELATION_BLOCK.get(service_id, "unassigned")
        for service_id in labels
    }
    return labels, blocks


def _affected_source_feature_summary(
    ablation: Any,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """Aggregate source-feature shocks only over services that use the source.

    A median over all 16 services is structurally optimistic because most
    source-feature variants do not directly target most services. Public-source
    variants leave non-target services unchanged. Health and age variants can
    also move non-target services indirectly because the preregistered
    within-region common effect is re-estimated after multi-feature LOO. This
    table separates direct dependency, expected centering propagation, and
    exact invariance instead of conflating them.
    """

    services = {str(row["service_id"]): row for row in config["services"]}
    expected_service_ids = set(services)
    region_runs = ablation.region_runs
    score_keys = ["admin_dong_code", "service_id"]
    baseline_region = region_runs.loc[
        region_runs["run_id"].eq("baseline::primary"),
        [*score_keys, "specialty_gap_score"],
    ].rename(columns={"specialty_gap_score": "baseline_score"})
    if len(baseline_region) != len(expected_service_ids) * int(
        config["source"]["expected_admin_dongs"]
    ):
        raise ValueError("Source-feature audit baseline coverage is incomplete")
    component_by_kind = {
        "health": "health_context_gap_score",
        "age": "age_structure_gap_score",
        "public": "public_support_gap_score",
    }
    metric_columns = (
        "rank_spearman",
        "rank_kendall",
        "top20_jaccard",
        "symmetric_top20_in_top30",
        "rank_displacement_p95",
        "rank_displacement_max",
        "sigungu_top20_tvd",
    )
    rows: list[dict[str, Any]] = []
    ledger = ablation.ledger.loc[ablation.ledger["scope"].eq("source_feature")]
    for run in ledger.itertuples(index=False):
        parameters = json.loads(str(run.parameters_json))
        source_kind = str(parameters["source_kind"])
        source_column = str(parameters["source_column"])
        if source_kind == "health":
            affected = {
                service_id
                for service_id, service in services.items()
                if source_column in {str(value) for value in service["health_features"]}
            }
        elif source_kind == "age":
            affected = {
                service_id
                for service_id, service in services.items()
                if source_column in {str(value) for value in service["age_features"]}
            }
        elif source_kind == "public":
            affected = {
                service_id
                for service_id, service in services.items()
                if source_column == str(service["public_support_column"])
            }
        else:
            raise ValueError(f"Unknown source-feature kind: {source_kind}")
        if not affected:
            raise ValueError(
                f"Source-feature run has no affected services: {run.run_id}"
            )

        metrics = ablation.service_metrics.loc[
            ablation.service_metrics["run_id"].eq(str(run.run_id))
        ].copy()
        observed_service_ids = set(metrics["service_id"].astype(str))
        if observed_service_ids != expected_service_ids:
            raise ValueError(
                f"Source-feature service coverage mismatch for {run.run_id}: "
                f"expected={sorted(expected_service_ids)}, observed={sorted(observed_service_ids)}"
            )
        for column in metric_columns:
            metrics[column] = pd.to_numeric(metrics[column], errors="coerce")
        if metrics[list(metric_columns)].isna().any().any():
            raise ValueError(f"Non-finite source-feature metric in {run.run_id}")

        affected_metrics = metrics.loc[metrics["service_id"].astype(str).isin(affected)]
        unaffected_metrics = metrics.loc[
            ~metrics["service_id"].astype(str).isin(affected)
        ]
        unaffected_invariant = bool(
            unaffected_metrics.empty
            or (
                np.isclose(unaffected_metrics["rank_spearman"], 1.0, atol=1e-12).all()
                and np.isclose(
                    unaffected_metrics["symmetric_top20_in_top30"], 1.0, atol=1e-12
                ).all()
                and np.isclose(
                    unaffected_metrics["rank_displacement_max"], 0.0, atol=1e-12
                ).all()
            )
        )
        unaffected_changed_count = int(
            (
                ~np.isclose(
                    unaffected_metrics["rank_spearman"], 1.0, atol=1e-12
                )
            ).sum()
        )
        region_variant = region_runs.loc[
            region_runs["run_id"].eq(str(run.run_id)),
            [*score_keys, "specialty_gap_score"],
        ].rename(columns={"specialty_gap_score": "variant_score"})
        component_run_id = (
            "component::" + component_by_kind[source_kind] + "::neutral"
        )
        component_variant = region_runs.loc[
            region_runs["run_id"].eq(component_run_id),
            [*score_keys, "specialty_gap_score"],
        ].rename(columns={"specialty_gap_score": "component_neutral_score"})
        all_score_comparison = (
            baseline_region.merge(region_variant, on=score_keys, validate="one_to_one")
            .merge(component_variant, on=score_keys, validate="one_to_one")
        )
        score_comparison = all_score_comparison.loc[
            all_score_comparison["service_id"].astype(str).isin(affected)
        ]
        indirect_score_comparison = all_score_comparison.loc[
            ~all_score_comparison["service_id"].astype(str).isin(affected)
        ]
        direct_abs_delta = (
            score_comparison["variant_score"] - score_comparison["baseline_score"]
        ).abs()
        indirect_abs_delta = (
            indirect_score_comparison["variant_score"]
            - indirect_score_comparison["baseline_score"]
        ).abs()
        source_mean_abs_delta = float(
            direct_abs_delta.mean()
        )
        component_mean_abs_delta = float(
            (
                score_comparison["component_neutral_score"]
                - score_comparison["baseline_score"]
            )
            .abs()
            .mean()
        )
        delta_ratio = (
            source_mean_abs_delta / component_mean_abs_delta
            if component_mean_abs_delta > 0
            else (0.0 if source_mean_abs_delta == 0 else float("inf"))
        )
        rows.append(
            {
                "run_id": str(run.run_id),
                "source_kind": source_kind,
                "source_column": source_column,
                "variant_semantics": (
                    "full_refit_loo_for_multi_feature_services_else_parent_component_neutral_for_singletons"
                ),
                "affected_service_count": int(len(affected)),
                "affected_service_ids": json.dumps(
                    sorted(affected), ensure_ascii=False
                ),
                "affected_changed_service_count": int(
                    (~np.isclose(affected_metrics["rank_spearman"], 1.0, atol=1e-12)).sum()
                ),
                "unaffected_service_count": int(len(expected_service_ids - affected)),
                "unaffected_services_invariant": unaffected_invariant,
                "unaffected_invariance_expected": source_kind == "public",
                "indirect_common_centering_coupling_expected": source_kind
                in {"health", "age"},
                "unaffected_changed_service_count": unaffected_changed_count,
                "indirect_mean_abs_score_delta": float(
                    indirect_abs_delta.mean() if len(indirect_abs_delta) else 0.0
                ),
                "indirect_to_direct_mean_abs_delta_ratio": float(
                    (indirect_abs_delta.mean() / direct_abs_delta.mean())
                    if len(indirect_abs_delta) and direct_abs_delta.mean() > 0
                    else 0.0
                ),
                "direct_share_of_total_abs_score_movement": float(
                    direct_abs_delta.sum()
                    / (direct_abs_delta.sum() + indirect_abs_delta.sum())
                    if direct_abs_delta.sum() + indirect_abs_delta.sum() > 0
                    else 1.0
                ),
                "indirect_rank_displacement_max": float(
                    unaffected_metrics["rank_displacement_max"].max()
                    if not unaffected_metrics.empty
                    else 0.0
                ),
                "affected_rank_spearman_median": float(
                    affected_metrics["rank_spearman"].median()
                ),
                "affected_rank_spearman_min": float(
                    affected_metrics["rank_spearman"].min()
                ),
                "affected_rank_kendall_median": float(
                    affected_metrics["rank_kendall"].median()
                ),
                "affected_top20_jaccard_median": float(
                    affected_metrics["top20_jaccard"].median()
                ),
                "affected_symmetric_top20_in_top30_median": float(
                    affected_metrics["symmetric_top20_in_top30"].median()
                ),
                "direct_dependency_attention": bool(
                    affected_metrics["rank_spearman"].median() + 1e-12
                    < float(
                        config["quality_gates"][
                            "default_feature_median_spearman_min"
                        ]
                    )
                    or affected_metrics["symmetric_top20_in_top30"].median()
                    + 1e-12
                    < float(
                        config["quality_gates"][
                            "default_feature_median_top20_in_top30_min"
                        ]
                    )
                ),
                "affected_rank_displacement_p95_max": float(
                    affected_metrics["rank_displacement_p95"].max()
                ),
                "affected_rank_displacement_max": float(
                    affected_metrics["rank_displacement_max"].max()
                ),
                "affected_sigungu_top20_tvd_max": float(
                    affected_metrics["sigungu_top20_tvd"].max()
                ),
                "affected_mean_abs_score_delta": source_mean_abs_delta,
                "affected_component_neutral_mean_abs_score_delta": (
                    component_mean_abs_delta
                ),
                "affected_delta_to_component_neutral_ratio": float(delta_ratio),
            }
        )
    result = pd.DataFrame(rows).sort_values(
        ["affected_rank_spearman_median", "source_kind", "source_column"],
        kind="stable",
    )
    if result.empty or len(result) != len(ledger) or not result["run_id"].is_unique:
        raise ValueError("Affected-service ablation summary is incomplete")
    return result.reset_index(drop=True)


def _weather_grid_representatives(
    daily: pd.DataFrame,
    variables: Iterable[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return one admin per exact weather series plus a transparent mapping."""

    variables = list(variables)
    hashes = pd.util.hash_pandas_object(daily[variables], index=False).to_numpy()
    rows = []
    signatures: dict[str, str] = {}
    for admin_code, indices in daily.groupby("admin_dong_code", sort=True).indices.items():
        signature = hashlib.sha256(hashes[indices].tobytes()).hexdigest()
        signatures[str(admin_code)] = signature
    for grid_order, signature in enumerate(sorted(set(signatures.values())), start=1):
        members = sorted(code for code, value in signatures.items() if value == signature)
        for code in members:
            rows.append(
                {
                    "admin_dong_code": code,
                    "weather_series_id": f"WG{grid_order:02d}",
                    "representative_admin_dong_code": members[0],
                    "weather_series_member_count": len(members),
                    "daily_series_sha256": signature,
                }
            )
    mapping = pd.DataFrame(rows)
    representatives = daily.loc[
        daily["admin_dong_code"].astype(str).isin(
            mapping["representative_admin_dong_code"].unique()
        )
    ].copy()
    return representatives, mapping


def _static_transit_fit(master: pd.DataFrame) -> pd.DataFrame:
    source = "gtfs_weekday_departures_per_1000_elderly"
    values = pd.to_numeric(master[source], errors="coerce")
    if values.isna().any():
        raise ValueError(f"Static transit baseline has missing values: {source}")
    fit = values.rank(method="average", pct=True) * 100.0
    return pd.DataFrame(
        {
            "admin_dong_code": master["admin_dong_code"].astype(str),
            "transit_time_fit": fit,
            "transit_evidence_status": "static_2024_weekday_baseline_no_monthly_variation",
        }
    )


def _seasonal_ablation_metrics(
    temporal_fit: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """Evaluate every temporal component at the released four-season resolution."""

    weights = {
        str(key): float(value)
        for key, value in config["temporal_fit"]["component_weights"].items()
    }
    month_to_season = {
        month: season for season, months in SEASON_MONTHS.items() for month in months
    }
    work = temporal_fit.copy()
    work["season"] = work["month"].map(month_to_season)
    baseline = (
        work.groupby(["admin_dong_code", "bundle_id", "season"], as_index=False)[
            "temporal_fit_score"
        ]
        .mean()
    )
    rows = []
    for removed in weights:
        remaining = {key: value for key, value in weights.items() if key != removed}
        total = sum(remaining.values())
        variant_column = f"variant_without__{removed}"
        work[variant_column] = (
            50.0
            if not remaining
            else sum(work[key] * (value / total) for key, value in remaining.items())
        )
        variant = (
            work.groupby(["admin_dong_code", "bundle_id", "season"], as_index=False)[
                variant_column
            ]
            .mean()
        )
        merged = baseline.merge(
            variant,
            on=["admin_dong_code", "bundle_id", "season"],
            validate="one_to_one",
        )
        group_metrics = []
        for _, group in merged.groupby(["admin_dong_code", "bundle_id"], sort=False):
            rho = _safe_spearman(
                group["temporal_fit_score"], group[variant_column]
            )
            base_top = group.sort_values(
                ["temporal_fit_score", "season"], ascending=[False, True], kind="stable"
            ).head(2)["season"]
            variant_top = group.sort_values(
                [variant_column, "season"], ascending=[False, True], kind="stable"
            ).head(2)["season"]
            base_set, variant_set = set(base_top), set(variant_top)
            group_metrics.append(
                (
                    float(rho),
                    len(base_set & variant_set) / 2.0,
                    int(base_top.iloc[0] == variant_top.iloc[0]),
                )
            )
        rows.append(
            {
                "removed_component": removed,
                "median_season_rank_spearman": float(np.median([x[0] for x in group_metrics])),
                "median_top2_season_recall": float(np.median([x[1] for x in group_metrics])),
                "primary_season_retention": float(np.mean([x[2] for x in group_metrics])),
                "destructive_no_supported_component": not bool(remaining),
            }
        )
    return pd.DataFrame(rows)


def _md_table(frame: pd.DataFrame, columns: Iterable[str] | None = None, limit: int = 30) -> str:
    selected = frame if columns is None else frame[list(columns)]
    selected = selected.head(limit).copy()
    if selected.empty:
        return "_(no rows)_"
    columns_list = [str(column) for column in selected.columns]

    def clean(value: Any) -> str:
        if value is None or (isinstance(value, float) and not np.isfinite(value)):
            return "NA"
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(columns_list) + " |",
        "| " + " | ".join("---" for _ in columns_list) + " |",
    ]
    for row in selected.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(clean(value) for value in row) + " |")
    return "\n".join(lines)


def _run_correlation_suite(
    root: Path,
    scores: pd.DataFrame,
    bundle_scores: pd.DataFrame,
    master: pd.DataFrame,
    specialty_config: Mapping[str, Any],
    bundle_config: Mapping[str, Any],
    *,
    n_boot: int,
    n_jobs: int,
) -> dict[str, Any]:
    correlation_dir = root / SPECIALTY_OUTPUT / "02_correlation"
    figure_dir = root / STAGE2_REPORT / "figures"
    correlation_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    service_labels, service_blocks = _service_metadata(specialty_config)
    stage1 = pd.read_csv(
        root / specialty_config["source"]["stage1_score"],
        dtype={"admin_dong_code": str},
    ).set_index("admin_dong_code")["need_score"]

    score_wide = scores.pivot(
        index="admin_dong_code", columns="service_id", values="specialty_gap_score"
    )
    admin_to_sigungu = (
        scores[["admin_dong_code", "policy_sigungu_name"]]
        .drop_duplicates("admin_dong_code")
        .set_index("admin_dong_code")["policy_sigungu_name"]
        .reindex(score_wide.index)
    )
    score_manifest = _manifest(
        score_wide.columns,
        labels=service_labels,
        blocks=service_blocks,
        granularity="admin_dong",
    )
    score_manifest = stabilize_feature_manifest(score_manifest, score_wide.columns)
    primary_atlas = build_specialty_bundle_atlas(
        score_wide,
        score_manifest,
        figure_dir,
        bundle_by_feature=service_blocks,
        stage1_scores=stage1,
        min_periods=107,
        include_spectral=True,
        stem="stage2a_specialty_gap_differential",
        title="Stage 2A Specialty Differential Gap — 153 Admin Dongs",
        annotate=True,
    )
    write_csv(score_manifest, correlation_dir / "specialty_score_manifest.csv")
    write_csv(primary_atlas.correlation, correlation_dir / "specialty_score_correlation.csv", index=True)
    write_csv(primary_atlas.pairwise_n, correlation_dir / "specialty_score_pairwise_n.csv", index=True)
    if primary_atlas.stage1_overlap is not None:
        write_csv(
            primary_atlas.stage1_overlap,
            correlation_dir / "specialty_stage1_need_overlap.csv",
        )

    bootstrap_quality, bootstrap_edges = block_bootstrap_quality(
        score_wide,
        admin_to_sigungu,
        score_manifest,
        n_boot=n_boot,
        seed=int(specialty_config.get("random_seed", 42)),
        n_jobs=n_jobs,
    )
    write_csv(bootstrap_quality, correlation_dir / "specialty_score_block_bootstrap_quality.csv")
    write_csv(bootstrap_edges, correlation_dir / "specialty_score_block_bootstrap_edges.csv")

    within = score_wide - score_wide.groupby(admin_to_sigungu).transform("mean")
    within_atlas = build_specialty_bundle_atlas(
        within,
        score_manifest,
        figure_dir,
        bundle_by_feature=service_blocks,
        min_periods=107,
        include_spectral=True,
        stem="stage2a_specialty_gap_within_sigungu",
        title="Stage 2A Specialty Gap — Within-Sigungu Residual Correlation",
        annotate=True,
    )
    write_csv(
        within_atlas.correlation,
        correlation_dir / "specialty_within_sigungu_correlation.csv",
        index=True,
    )
    write_csv(
        within_atlas.pairwise_n,
        correlation_dir / "specialty_within_sigungu_pairwise_n.csv",
        index=True,
    )
    within_bootstrap_quality, within_bootstrap_edges = block_bootstrap_quality(
        within,
        admin_to_sigungu,
        score_manifest,
        n_boot=n_boot,
        seed=int(specialty_config.get("random_seed", 42)) + 1,
        n_jobs=n_jobs,
    )
    write_csv(
        within_bootstrap_quality,
        correlation_dir / "specialty_within_sigungu_block_bootstrap_quality.csv",
    )
    write_csv(
        within_bootstrap_edges,
        correlation_dir / "specialty_within_sigungu_block_bootstrap_edges.csv",
    )

    bundle_wide = bundle_scores.pivot(
        index="admin_dong_code", columns="bundle_id", values="bundle_gap_score"
    )
    bundle_labels = {
        str(row["bundle_id"]): str(row["bundle_name_ko"])
        for row in bundle_config["bundles"]
    }
    bundle_blocks = {feature: feature for feature in bundle_wide.columns}
    bundle_manifest = _manifest(
        bundle_wide.columns,
        labels=bundle_labels,
        blocks=bundle_blocks,
        granularity="admin_dong",
    )
    bundle_manifest = stabilize_feature_manifest(bundle_manifest, bundle_wide.columns)
    bundle_atlas = build_specialty_bundle_atlas(
        bundle_wide,
        bundle_manifest,
        figure_dir,
        bundle_by_feature=bundle_blocks,
        stage1_scores=stage1,
        min_periods=107,
        include_spectral=True,
        stem="stage2a_recommended_bundle_gap",
        title="Stage 2A Recommended Service-Bundle Gap — Shared Services Explicit",
        annotate=True,
    )
    write_csv(bundle_atlas.correlation, correlation_dir / "bundle_score_correlation.csv", index=True)
    write_csv(bundle_atlas.pairwise_n, correlation_dir / "bundle_score_pairwise_n.csv", index=True)

    source_columns = [
        "local_specialty_facility_count",
        "local_specialty_specialist_count",
        "specialty_drive_min_v6",
        "specialty_access_excess_log1p",
    ]
    source_wide = _pivot_long(scores, source_columns)
    source_blocks = {feature: feature.split("::", 1)[1] for feature in source_wide.columns}
    source_labels = {
        feature: f"{service_labels[feature.split('::', 1)[0]]} | {feature.split('::', 1)[1]}"
        for feature in source_wide.columns
    }
    source_manifest = _manifest(
        source_wide.columns,
        labels=source_labels,
        blocks=source_blocks,
        granularity="admin_dong",
        roles={feature: "source_or_derived_component" for feature in source_wide.columns},
    )
    source_manifest = stabilize_feature_manifest(source_manifest, source_wide.columns)
    # Raw sources are service-specific observations.  Giving each a distinct
    # lineage key prevents a shared suffix from falsely explaining empirical
    # cross-service collinearity.
    source_manifest["equivalence_key"] = source_manifest["source_column"]
    source_manifest["equivalence_kind"] = "source"
    source_atlas = build_specialty_bundle_atlas(
        source_wide,
        source_manifest,
        figure_dir,
        bundle_by_feature=source_blocks,
        min_periods=107,
        include_spectral=False,
        stem="stage2a_source_feature_atlas",
        title="Stage 2A Full Source Atlas — Supply, Absolute Access, Excess Access",
        annotate=False,
    )
    write_csv(source_manifest, correlation_dir / "source_feature_manifest.csv")
    write_csv(source_atlas.correlation, correlation_dir / "source_feature_correlation.csv", index=True)
    write_csv(source_atlas.pairwise_n, correlation_dir / "source_feature_pairwise_n.csv", index=True)

    component_wide = _pivot_long(scores, COMPONENT_COLUMNS)
    component_blocks = {
        feature: feature.split("::", 1)[1] for feature in component_wide.columns
    }
    component_labels = {
        feature: f"{service_labels[feature.split('::', 1)[0]]} | {feature.split('::', 1)[1]}"
        for feature in component_wide.columns
    }
    component_manifest = _manifest(
        component_wide.columns,
        labels=component_labels,
        blocks=component_blocks,
        granularity="admin_dong",
        roles={feature: "primary_component" for feature in component_wide.columns},
    )
    component_manifest = stabilize_feature_manifest(
        component_manifest, component_wide.columns
    )
    service_definitions = {
        str(row["service_id"]): row for row in specialty_config["services"]
    }

    def component_lineage(feature: str) -> str:
        service_id, component = str(feature).split("::", 1)
        service = service_definitions[service_id]
        if component == "health_context_gap_score":
            source_key = "|".join(sorted(map(str, service["health_features"])))
        elif component == "age_structure_gap_score":
            source_key = "|".join(sorted(map(str, service["age_features"])))
        elif component == "public_support_gap_score":
            source_key = str(service["public_support_column"])
        else:
            source_key = service_id
        return f"{component}::{source_key}"

    component_manifest["equivalence_key"] = component_manifest[
        "source_column"
    ].map(component_lineage)
    component_manifest["equivalence_kind"] = "component"
    component_atlas = build_specialty_bundle_atlas(
        component_wide,
        component_manifest,
        figure_dir,
        bundle_by_feature=component_blocks,
        min_periods=107,
        include_spectral=False,
        stem="stage2a_all_component_atlas",
        title="Stage 2A Complete 16-Service × 5-Component Correlation Atlas",
        annotate=False,
    )
    write_csv(component_manifest, correlation_dir / "component_feature_manifest.csv")
    write_csv(component_atlas.correlation, correlation_dir / "component_feature_correlation.csv", index=True)
    write_csv(component_atlas.pairwise_n, correlation_dir / "component_feature_pairwise_n.csv", index=True)

    full_wide = pd.concat(
        [
            source_wide,
            component_wide,
            score_wide.rename(columns=lambda value: f"{value}::specialty_gap_score"),
        ],
        axis=1,
    )
    full_blocks = {
        feature: feature.split("::", 1)[1] for feature in full_wide.columns
    }
    full_labels = {
        feature: f"{service_labels[feature.split('::', 1)[0]]} | {feature.split('::', 1)[1]}"
        for feature in full_wide.columns
    }
    full_manifest = _manifest(
        full_wide.columns,
        labels=full_labels,
        blocks=full_blocks,
        granularity="admin_dong",
        roles={
            feature: (
                "source_or_derived_component"
                if feature in source_wide.columns
                else "primary_component"
                if feature in component_wide.columns
                else "model_output"
            )
            for feature in full_wide.columns
        },
    )
    full_manifest = stabilize_feature_manifest(full_manifest, full_wide.columns)
    full_manifest["equivalence_key"] = full_manifest["source_column"]
    full_manifest["equivalence_kind"] = full_manifest["role"].map(
        {
            "source_or_derived_component": "source",
            "primary_component": "component",
            "model_output": "model_output",
        }
    )
    for service_id in service_labels:
        lineage_key = f"{service_id}::specialty_excess_access_construct"
        lineage_columns = {
            f"{service_id}::specialty_access_excess_log1p",
            f"{service_id}::specialty_excess_access_gap_score",
        }
        full_manifest.loc[
            full_manifest["source_column"].isin(lineage_columns),
            "equivalence_key",
        ] = lineage_key
    component_lineage_by_column = component_manifest.set_index("source_column")[
        "equivalence_key"
    ]
    component_rows = full_manifest["source_column"].isin(
        component_lineage_by_column.index
    )
    full_manifest.loc[component_rows, "equivalence_key"] = full_manifest.loc[
        component_rows, "source_column"
    ].map(component_lineage_by_column)
    # The excess-access component and its raw log-excess source are a known
    # monotone lineage and intentionally share an explicit key.
    for service_id in service_labels:
        lineage_key = f"{service_id}::specialty_excess_access_construct"
        lineage_columns = {
            f"{service_id}::specialty_access_excess_log1p",
            f"{service_id}::specialty_excess_access_gap_score",
        }
        full_manifest.loc[
            full_manifest["source_column"].isin(lineage_columns),
            "equivalence_key",
        ] = lineage_key
    full_atlas = build_specialty_bundle_atlas(
        full_wide,
        full_manifest,
        figure_dir,
        bundle_by_feature=full_blocks,
        min_periods=107,
        include_spectral=False,
        stem="stage2a_full_160_feature_atlas",
        title="Stage 2A Full 160-Feature Atlas — Interactive IDs Preserve Readability",
        annotate=False,
    )
    write_csv(full_manifest, correlation_dir / "full_stage2a_manifest.csv")
    write_csv(full_atlas.correlation, correlation_dir / "full_stage2a_correlation.csv", index=True)
    write_csv(full_atlas.pairwise_n, correlation_dir / "full_stage2a_pairwise_n.csv", index=True)

    representative_wide, representative_manifest, representative_selection = (
        _select_discriminating_atlas_features(full_wide, full_manifest)
    )
    representative_blocks = representative_manifest.set_index("source_column")[
        "feature_family"
    ].to_dict()
    representative_atlas = build_specialty_bundle_atlas(
        representative_wide,
        representative_manifest,
        figure_dir,
        bundle_by_feature=representative_blocks,
        min_periods=107,
        include_spectral=False,
        stem="stage2a_discriminating_representative_atlas",
        title=(
            "Stage 2A Discriminating Atlas — Exact Aliases and Raw Formula "
            "Lineage Collapsed"
        ),
        annotate=False,
    )
    write_csv(
        representative_manifest,
        correlation_dir / "discriminating_representative_manifest.csv",
    )
    write_csv(
        representative_selection,
        correlation_dir / "discriminating_atlas_selection_ledger.csv",
    )
    write_csv(
        representative_atlas.correlation,
        correlation_dir / "discriminating_representative_correlation.csv",
        index=True,
    )
    write_csv(
        representative_atlas.pairwise_n,
        correlation_dir / "discriminating_representative_pairwise_n.csv",
        index=True,
    )

    reviewed_ledgers: list[pd.DataFrame] = []
    for atlas_name, atlas in (
        ("primary_16", primary_atlas),
        ("within_sigungu_16", within_atlas),
        ("recommended_bundle_5", bundle_atlas),
        ("source_64", source_atlas),
        ("component_80", component_atlas),
        ("full_160", full_atlas),
        ("discriminating_representative", representative_atlas),
    ):
        reviewed = _review_high_correlation_ledger(
            atlas.high_correlation_ledger,
            atlas_name=atlas_name,
        )
        reviewed_ledgers.append(reviewed)
        write_csv(
            reviewed,
            Path(atlas.artifacts["high_correlation_ledger_csv"]),
        )
    high_correlation_review = pd.concat(
        reviewed_ledgers, ignore_index=True, sort=False
    )
    write_csv(
        high_correlation_review,
        correlation_dir / "stage2a_high_correlation_explanation_ledger.csv",
    )

    chs_columns = [column for column in master.columns if column.startswith("chs65p_")]
    health = (
        master[["policy_sigungu_name", *chs_columns]]
        .drop_duplicates("policy_sigungu_name")
        .set_index("policy_sigungu_name")
        .sort_index()
    )
    if len(health) != 11:
        raise ValueError(f"CHS correlation must use 11 policy sigungu, found {len(health)}")
    health_blocks = {
        column: (
            "chronic"
            if any(token in column for token in ("hypertension", "diabetes"))
            else "functional"
            if any(token in column for token in ("mobility", "usual_activity", "fall", "accident"))
            else "psychosocial"
            if any(token in column for token in ("depressive", "stress"))
            else "care_access"
            if any(token in column for token in ("unmet", "dissatisfaction"))
            else "general_health"
        )
        for column in chs_columns
    }
    health_manifest = _manifest(
        chs_columns,
        blocks=health_blocks,
        granularity="policy_sigungu",
        roles={column: "context_n11" for column in chs_columns},
    )
    health_manifest = stabilize_feature_manifest(health_manifest, chs_columns)
    health_corr, health_n = same_granularity_correlation(health, health_manifest, min_periods=8)
    health_paths = plot_correlation_map(
        health_corr,
        health_n,
        health_manifest,
        figure_dir / "stage2a_health_context_n11",
        "Stage 2A Health Context — 11 Independent Policy Sigungu",
        annotate=True,
        mask_n=8,
    )
    health_html = save_interactive_heatmap(
        health_corr,
        health_n,
        health_manifest,
        figure_dir / "stage2a_health_context_n11.html",
        "Stage 2A Health Context — n=11, No 153-Row Pseudoreplication",
    )
    write_csv(health_corr, correlation_dir / "health_context_n11_correlation.csv", index=True)
    write_csv(health_n, correlation_dir / "health_context_n11_pairwise_n.csv", index=True)

    formula_rows = []
    formula_columns = list(FORMULA_COLUMNS.values())
    for service_id, group in scores.groupby("service_id", sort=True):
        corr = group[formula_columns].corr(method="spearman")
        pair_values = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack()
        formula_rows.append(
            {
                "service_id": service_id,
                "formula_pair_median_spearman": float(pair_values.median()),
                "formula_pair_min_spearman": float(pair_values.min()),
                "formula_pair_max_spearman": float(pair_values.max()),
            }
        )
    formula_stability = pd.DataFrame(formula_rows)
    write_csv(formula_stability, correlation_dir / "formula_family_stability.csv")

    bundle_ids = [str(row["bundle_id"]) for row in bundle_config["bundles"]]
    service_ids = [str(row["service_id"]) for row in specialty_config["services"]]
    weight_vectors = pd.DataFrame(0.0, index=bundle_ids, columns=service_ids)
    for bundle in bundle_config["bundles"]:
        for service_id, weight in bundle["included_services"].items():
            weight_vectors.loc[str(bundle["bundle_id"]), str(service_id)] = float(weight)
    overlap = pd.DataFrame(index=bundle_ids, columns=bundle_ids, dtype=float)
    for left in bundle_ids:
        for right in bundle_ids:
            overlap.loc[left, right] = float(
                np.minimum(weight_vectors.loc[left], weight_vectors.loc[right]).sum()
            )
    write_csv(weight_vectors.reset_index(names="bundle_id"), correlation_dir / "bundle_service_weights.csv")
    write_csv(overlap, correlation_dir / "bundle_shared_weight_overlap.csv", index=True)

    score_id_map = score_manifest.set_index("source_column")["feature_id"].to_dict()
    bootstrap_id_contract = bool(
        (
            bootstrap_edges["feature_id_a"]
            == bootstrap_edges["feature_a"].map(score_id_map)
        ).all()
        and (
            bootstrap_edges["feature_id_b"]
            == bootstrap_edges["feature_b"].map(score_id_map)
        ).all()
        and (
            within_bootstrap_edges["feature_id_a"]
            == within_bootstrap_edges["feature_a"].map(score_id_map)
        ).all()
        and (
            within_bootstrap_edges["feature_id_b"]
            == within_bootstrap_edges["feature_b"].map(score_id_map)
        ).all()
    )
    publication_atlases = (
        primary_atlas,
        within_atlas,
        bundle_atlas,
        source_atlas,
        component_atlas,
        full_atlas,
        representative_atlas,
    )
    required_artifact_keys = {
        "png",
        "svg",
        "html",
        "correlation_csv",
        "pairwise_n_csv",
        "manifest_csv",
        "high_correlation_ledger_csv",
    }
    publication_artifacts_complete = all(
        required_artifact_keys.issubset(atlas.artifacts)
        and all(
            Path(atlas.artifacts[key]).is_file()
            and Path(atlas.artifacts[key]).stat().st_size > 0
            for key in required_artifact_keys
        )
        for atlas in publication_atlases
    )
    publication_manifest_label_coverage = float(
        np.mean(
            [
                bool(
                    atlas.manifest["label_ko"].notna().all()
                    and atlas.manifest["label_ko"].astype(str).str.strip().ne("").all()
                    and atlas.manifest["feature_id"].is_unique
                )
                for atlas in publication_atlases
            ]
        )
    )
    external_script_count = 0
    for atlas in publication_atlases:
        html_text = Path(atlas.artifacts["html"]).read_text(
            encoding="utf-8", errors="replace"
        ).lower()
        external_script_count += html_text.count("<script src=")
        external_script_count += html_text.count("<script src =")
    pairwise_matrices = (
        primary_atlas.pairwise_n,
        within_atlas.pairwise_n,
        bundle_atlas.pairwise_n,
        source_atlas.pairwise_n,
        component_atlas.pairwise_n,
        full_atlas.pairwise_n,
        representative_atlas.pairwise_n,
    )
    pairwise_n_min = int(
        min(matrix.to_numpy(dtype=int).min() for matrix in pairwise_matrices)
    )
    pairwise_n_max = int(
        max(matrix.to_numpy(dtype=int).max() for matrix in pairwise_matrices)
    )

    return {
        "primary_separation": dict(primary_atlas.separation_metrics),
        "primary_stage1_overlap": dict(primary_atlas.stage1_overlap_summary or {}),
        "primary_quality": dict(primary_atlas.quality_metrics),
        "within_separation": dict(within_atlas.separation_metrics),
        "within_quality": dict(within_atlas.quality_metrics),
        "bundle_correlation_max_off_diagonal": float(
            bundle_atlas.correlation.where(
                ~np.eye(len(bundle_atlas.correlation), dtype=bool)
            ).abs().max().max()
        ),
        "bundle_shared_weight_overlap_max_off_diagonal": float(
            overlap.where(~np.eye(len(overlap), dtype=bool)).max().max()
        ),
        "health_context_abs_rho_095_pairs": int(
            (
                health_corr.where(np.triu(np.ones(health_corr.shape), k=1).astype(bool))
                .stack()
                .abs()
                >= 0.95
            ).sum()
        ),
        "formula_pair_min_spearman": float(formula_stability["formula_pair_min_spearman"].min()),
        "score_abs_rho_098_pairs": int(
            (
                primary_atlas.correlation.where(
                    np.triu(np.ones(primary_atlas.correlation.shape), k=1).astype(bool)
                )
                .stack()
                .abs()
                >= 0.98
            ).sum()
        ),
        "full_atlas_feature_count": int(full_wide.shape[1]),
        "discriminating_atlas_feature_count": int(representative_wide.shape[1]),
        "discriminating_atlas_excluded_feature_count": int(
            (~representative_selection["included_in_discriminating_atlas"]).sum()
        ),
        "discriminating_separation": dict(
            representative_atlas.separation_metrics
        ),
        "high_correlation_pair_count": int(len(high_correlation_review)),
        "high_correlation_unexplained_after_review_count": int(
            high_correlation_review["unexplained_after_review"].sum()
        ),
        "high_correlation_empirical_review_pair_count": int(
            high_correlation_review["review_status"]
            .eq("reviewed_distinct_source_empirical_collinearity_not_duplicate")
            .sum()
        ),
        "full_atlas_high_correlation_pair_count": int(
            high_correlation_review["atlas_name"].eq("full_160").sum()
        ),
        "full_atlas_unexplained_after_review_count": int(
            high_correlation_review.loc[
                high_correlation_review["atlas_name"].eq("full_160"),
                "unexplained_after_review",
            ].sum()
        ),
        "representative_high_correlation_pair_count": int(
            high_correlation_review["atlas_name"]
            .eq("discriminating_representative")
            .sum()
        ),
        "representative_unexplained_after_review_count": int(
            high_correlation_review.loc[
                high_correlation_review["atlas_name"].eq(
                    "discriminating_representative"
                ),
                "unexplained_after_review",
            ].sum()
        ),
        "health_effective_n": int(len(health)),
        "stable_feature_id_contract_pass": bootstrap_id_contract,
        "publication_artifacts_complete": publication_artifacts_complete,
        "publication_manifest_label_coverage": publication_manifest_label_coverage,
        "interactive_external_script_count": int(external_script_count),
        "admin_atlas_pairwise_n_min": pairwise_n_min,
        "admin_atlas_pairwise_n_max": pairwise_n_max,
        "bootstrap_n": int(n_boot),
        "bootstrap_edge_count": int(len(bootstrap_edges)),
        "bootstrap_min_valid_fraction": float(bootstrap_edges["valid_fraction"].min()),
        "bootstrap_quality_min_valid_fraction": float(
            (bootstrap_quality["n_valid"] / bootstrap_quality["n_boot"]).min()
        ),
        "bootstrap_stable_edge_count": int(bootstrap_edges["stable_edge"].sum()),
        "within_bootstrap_min_valid_fraction": float(
            within_bootstrap_edges["valid_fraction"].min()
        ),
        "within_bootstrap_quality_min_valid_fraction": float(
            (
                within_bootstrap_quality["n_valid"]
                / within_bootstrap_quality["n_boot"]
            ).min()
        ),
        "within_bootstrap_stable_edge_count": int(
            within_bootstrap_edges["stable_edge"].sum()
        ),
        "within_bootstrap_edge_count": int(len(within_bootstrap_edges)),
        "within_bootstrap_axis_auc_ci05": float(
            within_bootstrap_quality.loc[
                within_bootstrap_quality["metric"].eq("axis_auc"), "ci05"
            ].iloc[0]
        ),
        "within_bootstrap_block_contrast_ci05": float(
            within_bootstrap_quality.loc[
                within_bootstrap_quality["metric"].eq("block_contrast"), "ci05"
            ].iloc[0]
        ),
        "artifacts": {
            "primary": dict(primary_atlas.artifacts),
            "within_sigungu": dict(within_atlas.artifacts),
            "bundle": dict(bundle_atlas.artifacts),
            "source": dict(source_atlas.artifacts),
            "components": dict(component_atlas.artifacts),
            "full": dict(full_atlas.artifacts),
            "discriminating_representative": dict(
                representative_atlas.artifacts
            ),
            "health_png": health_paths[0],
            "health_svg": health_paths[1],
            "health_html": health_html,
        },
    }


def _safe_spearman(left: pd.Series, right: pd.Series) -> float:
    left_values = pd.to_numeric(left, errors="coerce")
    right_values = pd.to_numeric(right, errors="coerce")
    valid = left_values.notna() & right_values.notna()
    if int(valid.sum()) < 3:
        return float("nan")
    if left_values[valid].nunique() < 2 or right_values[valid].nunique() < 2:
        return 1.0 if np.allclose(left_values[valid], right_values[valid]) else 0.0
    return float(left_values[valid].corr(right_values[valid], method="spearman"))


def _healthmap_external_validation(
    root: Path,
    scores: pd.DataFrame,
    master: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Use four independent 2023 HealthMap access measures as weak holdouts."""

    path = root / str(config["source"]["healthmap_holdout"])
    holdout = pd.read_csv(path, low_memory=False)
    service_tokens = {
        "internal_medicine": "접근성취약인구율_내과_60분",
        "orthopedics": "접근성취약인구율_정형외과_60분",
        "emergency_medicine": "접근성취약인구율_응급실_30분",
        "rehabilitation": "접근성취약인구율_재활_60분",
    }
    resolved: dict[str, str] = {}
    for service_id, token in service_tokens.items():
        matches = [column for column in holdout.columns if token in str(column)]
        if len(matches) != 1:
            raise ValueError(
                f"HealthMap field resolution failed for {service_id}: {matches}"
            )
        resolved[service_id] = matches[0]

    population = master.set_index("admin_dong_code")["population_65plus"]
    work = scores.loc[scores["service_id"].isin(resolved)].copy()
    work["population_65plus"] = work["admin_dong_code"].map(population)
    if work["population_65plus"].isna().any():
        raise ValueError("HealthMap validation cannot align population weights")
    aggregate_rows: list[dict[str, Any]] = []
    for (service_id, sigungu), group in work.groupby(
        ["service_id", "policy_sigungu_name"], sort=True
    ):
        weight = group["population_65plus"].to_numpy(float)
        score = group["specialty_gap_score"].to_numpy(float)
        aggregate_rows.append(
            {
                "service_id": str(service_id),
                "policy_sigungu_name": str(sigungu),
                "stage2_gap_population65_weighted_mean": float(
                    np.average(score, weights=weight)
                ),
            }
        )
    aggregate = pd.DataFrame(aggregate_rows)
    detail_rows: list[dict[str, Any]] = []
    for service_id, column in resolved.items():
        comparison = aggregate.loc[aggregate["service_id"].eq(service_id)].merge(
            holdout[["policy_sigungu_name", column]],
            on="policy_sigungu_name",
            how="inner",
            validate="one_to_one",
        )
        comparison[column] = pd.to_numeric(comparison[column], errors="coerce")
        rho = _safe_spearman(
            comparison["stage2_gap_population65_weighted_mean"], comparison[column]
        )
        detail_rows.append(
            {
                "service_id": service_id,
                "healthmap_column": column,
                "effective_n_policy_sigungu": int(
                    comparison[
                        ["stage2_gap_population65_weighted_mean", column]
                    ].dropna().shape[0]
                ),
                "spearman_rho": rho,
                "direction_concordant": bool(np.isfinite(rho) and rho > 0),
                "interpretation": (
                    "weak_external_direction_check_not_accuracy_or_demand_validation"
                ),
            }
        )
    detail = pd.DataFrame(detail_rows)
    summary = {
        "service_count": int(len(detail)),
        "effective_n": 11,
        "directional_concordance": float(detail["direction_concordant"].mean()),
        "median_spearman_rho": float(detail["spearman_rho"].median()),
        "minimum_spearman_rho": float(detail["spearman_rho"].min()),
    }
    return detail, summary


def _nhis_seasonal_stability(
    service_year_month: pd.DataFrame,
    bundle_config: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    month_to_season = {
        month: season for season, months in SEASON_MONTHS.items() for month in months
    }
    work = service_year_month.copy()
    work["season"] = work["month"].map(month_to_season)
    service_seasonal = (
        work.groupby(["service_id", "year", "season"], as_index=False)[
            "visits_modifier_equal_sigungu"
        ]
        .mean()
    )
    bundle_rows: list[dict[str, Any]] = []
    indexed = service_seasonal.set_index(["service_id", "year", "season"])
    years = sorted(service_seasonal["year"].unique().tolist())
    for bundle in bundle_config["bundles"]:
        included = {
            str(service_id): float(weight)
            for service_id, weight in bundle["included_services"].items()
        }
        for year in years:
            for season in SEASON_MONTHS:
                value = sum(
                    float(
                        indexed.loc[
                            (service_id, year, season),
                            "visits_modifier_equal_sigungu",
                        ]
                    )
                    * weight
                    for service_id, weight in included.items()
                )
                bundle_rows.append(
                    {
                        "bundle_id": str(bundle["bundle_id"]),
                        "year": int(year),
                        "season": season,
                        "bundle_visits_modifier": value,
                    }
                )
    seasonal = pd.DataFrame(bundle_rows)
    rows: list[dict[str, Any]] = []
    for bundle_id, group in seasonal.groupby("bundle_id", sort=True):
        pivot = group.pivot(
            index="season", columns="year", values="bundle_visits_modifier"
        ).reindex(SEASON_MONTHS)
        if pivot.shape[1] != 2 or pivot.isna().any().any():
            raise ValueError(f"NHIS seasonal stability panel incomplete: {bundle_id}")
        left, right = pivot.columns.tolist()
        top_left = str(
            pivot[left].sort_values(ascending=False, kind="mergesort").index[0]
        )
        top_right = str(
            pivot[right].sort_values(ascending=False, kind="mergesort").index[0]
        )
        rows.append(
            {
                "bundle_id": str(bundle_id),
                "year_a": int(left),
                "year_b": int(right),
                "season_rank_spearman": _safe_spearman(pivot[left], pivot[right]),
                "top1_season_agreement": int(top_left == top_right),
                "top1_season_year_a": top_left,
                "top1_season_year_b": top_right,
            }
        )
    detail = pd.DataFrame(rows)
    gates = config["quality_gates"]
    summary = {
        "bundle_count": int(len(detail)),
        "median_season_rank_spearman": float(detail["season_rank_spearman"].median()),
        "top1_season_agreement": float(detail["top1_season_agreement"].mean()),
    }
    summary["seasonal_nhis_gate_pass"] = bool(
        summary["median_season_rank_spearman"]
        >= float(gates["seasonal_resolution_nhis_loyo_spearman_min"])
        and summary["top1_season_agreement"]
        >= float(gates["seasonal_resolution_nhis_top1_agreement_min"])
    )
    return detail, summary


def _climate_seasonal_stability(
    daily_representatives: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """LOYO stability at the released four-season, not monthly, resolution."""

    work = daily_representatives.copy()
    work["date"] = pd.to_datetime(work["date"], errors="raise")
    work["__year"] = work["date"].dt.year
    years = sorted(work["__year"].unique().tolist())
    rows: list[dict[str, Any]] = []
    for holdout_year in years:
        training = work.loc[work["__year"].ne(holdout_year)].drop(columns="__year")
        holdout = work.loc[work["__year"].eq(holdout_year)].drop(columns="__year")
        train_month = build_monthly_climate_risk(
            training,
            config,
            units=WEATHER_UNITS,
            expected_admin_dongs=training["admin_dong_code"].nunique(),
            expected_start_date=training["date"].min(),
            expected_end_date=training["date"].max(),
            completeness_min=0.0,
        )
        holdout_month = build_monthly_climate_risk(
            holdout,
            config,
            units=WEATHER_UNITS,
            expected_admin_dongs=holdout["admin_dong_code"].nunique(),
            expected_start_date=f"{holdout_year}-01-01",
            expected_end_date=f"{holdout_year}-12-31",
            completeness_min=0.99,
        )
        for frame in (train_month, holdout_month):
            frame["season"] = frame["month"].map(
                {
                    month: season
                    for season, months in SEASON_MONTHS.items()
                    for month in months
                }
            )
        train_season = train_month.groupby(
            ["admin_dong_code", "season"], as_index=False
        )["climate_operational_fit"].mean()
        holdout_season = holdout_month.groupby(
            ["admin_dong_code", "season"], as_index=False
        )["climate_operational_fit"].mean()
        merged = train_season.merge(
            holdout_season,
            on=["admin_dong_code", "season"],
            suffixes=("_train", "_holdout"),
            validate="one_to_one",
        )
        for admin_code, group in merged.groupby("admin_dong_code", sort=True):
            ordered_train = group.sort_values(
                ["climate_operational_fit_train", "season"],
                ascending=[False, True],
                kind="mergesort",
            )
            ordered_holdout = group.sort_values(
                ["climate_operational_fit_holdout", "season"],
                ascending=[False, True],
                kind="mergesort",
            )
            top2_train = set(ordered_train.head(2)["season"])
            top2_holdout = set(ordered_holdout.head(2)["season"])
            rows.append(
                {
                    "holdout_year": int(holdout_year),
                    "representative_admin_dong_code": str(admin_code),
                    "season_rank_spearman": _safe_spearman(
                        group["climate_operational_fit_train"],
                        group["climate_operational_fit_holdout"],
                    ),
                    "top2_season_recall": len(top2_train & top2_holdout) / 2.0,
                    "top1_season_agreement": int(
                        ordered_train.iloc[0]["season"]
                        == ordered_holdout.iloc[0]["season"]
                    ),
                }
            )
    detail = pd.DataFrame(rows)
    gates = config["quality_gates"]
    summary = {
        "holdout_year_count": int(detail["holdout_year"].nunique()),
        "representative_weather_series_count": int(
            detail["representative_admin_dong_code"].nunique()
        ),
        "median_season_rank_spearman": float(detail["season_rank_spearman"].median()),
        "median_top2_season_recall": float(detail["top2_season_recall"].median()),
        "top1_season_agreement": float(detail["top1_season_agreement"].mean()),
    }
    summary["seasonal_climate_gate_pass"] = bool(
        summary["median_season_rank_spearman"]
        >= float(gates["seasonal_resolution_climate_loyo_spearman_min"])
        and summary["median_top2_season_recall"]
        >= float(gates["seasonal_resolution_climate_top2_recall_min"])
        and summary["top1_season_agreement"]
        >= float(gates["seasonal_resolution_climate_top1_agreement_min"])
    )
    return detail, summary


def _audit_record(
    frame: pd.DataFrame,
    *,
    dataset: str,
    time_resolution: str,
    spatial_resolution: str,
    specialty_resolution: str,
    source: str,
    usable_for_model: bool,
    usable_for_validation: bool,
    confidence: str,
    limitations: str,
    required_columns: Iterable[str],
    time_values: pd.Series | None = None,
    path: Path | None = None,
) -> dict[str, Any]:
    row = temporal_audit_row(
        frame,
        dataset=dataset,
        time_resolution=time_resolution,
        spatial_resolution=spatial_resolution,
        specialty_resolution=specialty_resolution,
        source=source,
        usable_for_model=usable_for_model,
        usable_for_validation=usable_for_validation,
        confidence=confidence,
        limitations=limitations,
        required_columns=required_columns,
        time_values=time_values,
    )
    if path is not None:
        row["relative_path"] = str(path).replace("\\", "/")
        row["sha256"] = sha256_file(path)
    return row


def _build_temporal_audit(
    root: Path,
    *,
    master: pd.DataFrame,
    scores: pd.DataFrame,
    nhis: pd.DataFrame,
    holidays: pd.DataFrame,
    service_month: pd.DataFrame,
    weather_daily: pd.DataFrame,
    monthly_climate: pd.DataFrame,
    temporal_fit: pd.DataFrame,
    specialty_config: Mapping[str, Any],
    temporal_config: Mapping[str, Any],
) -> pd.DataFrame:
    specialty_source = specialty_config["source"]
    temporal_source = temporal_config["source"]
    road_path = root / str(specialty_source["road_master"])
    nhis_path = root / str(temporal_source["nhis_monthly_raw"])
    holiday_path = root / "14_v6_external_data/stage2_temporal/raw/official_public_holidays_2022_2023.csv"
    weather_path = root / "14_v6_external_data/stage2_temporal/raw/nasa_power_admin_dong_daily_1991_2025.parquet"
    healthmap_path = root / str(specialty_source["healthmap_holdout"])
    healthmap = pd.read_csv(healthmap_path, low_memory=False)
    mobile_path = root / "07_outreach_overlap/mobile_clinic_public/mobile_clinic_events_publicly_verified_2025_2026.csv"
    staffing_path = root / "13_mobile_clinic_actual/cheongju_medical_center_foi_staffing_sessions_2025_2026.csv"
    rural_path = root / "07_outreach_overlap/rural_medical_bus/rural_medical_bus_known_events_2025_2026.csv"
    mobile = pd.read_csv(mobile_path, low_memory=False)
    staffing = pd.read_csv(staffing_path, low_memory=False)
    rural = pd.read_csv(rural_path, low_memory=False)

    records = [
        _audit_record(
            master,
            dataset="road_integrated_admin_master",
            time_resolution="cross_section_2026_sources_mixed",
            spatial_resolution="153_admin_dong",
            specialty_resolution="16_service_access_supported",
            source="MEDIROAD_V6 road-integrated master",
            usable_for_model=True,
            usable_for_validation=True,
            confidence="high_static_proxy",
            limitations="OSM free-flow routing is not observed congestion",
            required_columns=("admin_dong_code", "policy_sigungu_name"),
            path=road_path,
        ),
        _audit_record(
            scores,
            dataset="stage2a_specialty_gap",
            time_resolution="structural_cross_section",
            spatial_resolution="153_admin_dong",
            specialty_resolution="16_services",
            source="Stage2A transparent MCDA",
            usable_for_model=True,
            usable_for_validation=False,
            confidence="model_derived",
            limitations="relative policy gap; not patient demand",
            required_columns=("admin_dong_code", "service_id", "specialty_gap_score"),
        ),
        _audit_record(
            nhis,
            dataset="nhis_sigungu_specialty_month",
            time_resolution="monthly_2022_2023",
            spatial_resolution="11_policy_sigungu_equal_weight",
            specialty_resolution="16_mapped_services",
            source="NHIS official file data 15129861",
            usable_for_model=True,
            usable_for_validation=True,
            confidence="moderate_two_years",
            limitations="insured observed utilization; not unmet need; all ages",
            required_columns=("year", "month", "policy_sigungu_name", "service_id", "visits"),
            time_values=nhis["year"],
            path=nhis_path,
        ),
        _audit_record(
            holidays,
            dataset="official_public_holidays",
            time_resolution="daily_2022_2023",
            spatial_resolution="national_calendar",
            specialty_resolution="none",
            source="Korean Astronomy and Space Science Institute holiday API",
            usable_for_model=True,
            usable_for_validation=True,
            confidence="high",
            limitations="does not encode provider-specific closures",
            required_columns=("date",),
            time_values=holidays["date"],
            path=holiday_path,
        ),
        _audit_record(
            service_month,
            dataset="nhis_reliability_shrunk_service_month_profile",
            time_resolution="monthly_common_profile",
            spatial_resolution="province_common_from_11_sigungu",
            specialty_resolution="16_services",
            source="derived from official NHIS panel",
            usable_for_model=True,
            usable_for_validation=True,
            confidence="service_specific_tier",
            limitations="monthly gate failed; release only at season resolution",
            required_columns=("service_id", "month", "seasonality_modifier"),
        ),
        _audit_record(
            weather_daily,
            dataset="nasa_power_merra2_daily",
            time_resolution="daily_1991_2025",
            spatial_resolution="153_admin_dong_mapped_to_coarse_grid",
            specialty_resolution="operational_common",
            source="NASA POWER MERRA-2 API",
            usable_for_model=True,
            usable_for_validation=True,
            confidence="moderate_reanalysis_proxy",
            limitations="153 admin dongs collapse to coarse weather series; snowfall unavailable",
            required_columns=("admin_dong_code", "date", *WEATHER_UNITS),
            time_values=weather_daily["date"],
            path=weather_path,
        ),
        _audit_record(
            monthly_climate,
            dataset="monthly_operational_climate_fit",
            time_resolution="long_run_monthly_climatology",
            spatial_resolution="153_admin_dong_coarse_proxy",
            specialty_resolution="operational_common",
            source="derived NASA POWER hazard rates",
            usable_for_model=True,
            usable_for_validation=True,
            confidence="moderate",
            limitations="not a date-specific forecast; snowfall unknown-not-zero",
            required_columns=("admin_dong_code", "month", "climate_operational_fit"),
        ),
        _audit_record(
            temporal_fit,
            dataset="stage2b_region_bundle_month_diagnostic",
            time_resolution="monthly_diagnostic_season_release",
            spatial_resolution="153_admin_dong",
            specialty_resolution="5_recommended_bundles",
            source="Stage2B transparent temporal MCDA",
            usable_for_model=True,
            usable_for_validation=False,
            confidence="season_only",
            limitations="calendar/team/vehicle/venue availability unknown",
            required_columns=("admin_dong_code", "bundle_id", "month", "temporal_fit_score"),
        ),
        _audit_record(
            healthmap,
            dataset="healthmap_2023_external_holdout",
            time_resolution="annual_2023",
            spatial_resolution="11_policy_sigungu",
            specialty_resolution="selected_access_categories",
            source="HealthMap public validation extract",
            usable_for_model=False,
            usable_for_validation=True,
            confidence="moderate_n11",
            limitations="coarse independent directional check only",
            required_columns=("policy_sigungu_name",),
            time_values=healthmap["healthmap_reference_year"],
            path=healthmap_path,
        ),
        _audit_record(
            mobile,
            dataset="public_mobile_clinic_events",
            time_resolution="event_2025_2026",
            spatial_resolution="named_event_locations",
            specialty_resolution="free_text_partial",
            source="publicly verified event ledger",
            usable_for_model=False,
            usable_for_validation=True,
            confidence="low_sparse",
            limitations="8 events; participant counts mostly unavailable",
            required_columns=("event_date", "city_county", "eup_myeon_dong"),
            time_values=mobile["event_date"],
            path=mobile_path,
        ),
        _audit_record(
            staffing,
            dataset="foi_staffing_sessions",
            time_resolution="event_2025_2026",
            spatial_resolution="session_locations",
            specialty_resolution="staffing_only",
            source="Cheongju Medical Center FOI",
            usable_for_model=False,
            usable_for_validation=True,
            confidence="moderate_sparse",
            limitations="six staffing sessions; no patient demand",
            required_columns=("session_date", "sigungu", "eup_myeon_dong"),
            time_values=staffing["session_date"],
            path=staffing_path,
        ),
        _audit_record(
            rural,
            dataset="rural_medical_bus_known_events",
            time_resolution="event_2025_2026",
            spatial_resolution="three_sigungu_partial",
            specialty_resolution="program_event",
            source="public rural medical bus ledger",
            usable_for_model=False,
            usable_for_validation=True,
            confidence="low_sparse",
            limitations="descriptive operational prior, not full Chungbuk calendar",
            required_columns=tuple(rural.columns[:1]),
            time_values=(
                rural[next((c for c in rural.columns if "date" in c.lower()), rural.columns[0])]
            ),
            path=rural_path,
        ),
    ]
    return build_temporal_data_audit(records)


def _operational_advisory_sensitivity(
    temporal_fit: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """Quantify rejected climate/static-transit overlays without releasing them."""

    rows: list[dict[str, Any]] = []
    scenarios = [float(value) for value in config["temporal_fit"]["weather_weight_scenarios"]]
    experiments = [
        ("climate_overlay", weight, "climate_operational_fit")
        for weight in scenarios
    ] + [("static_transit_overlay", 0.10, "transit_time_fit")]
    baseline = temporal_fit.copy()
    month_to_season = {
        month: season for season, months in SEASON_MONTHS.items() for month in months
    }
    baseline["season"] = baseline["month"].map(month_to_season)
    for experiment, overlay_weight, column in experiments:
        work = baseline.copy()
        work["variant_score"] = (
            (1.0 - overlay_weight) * work["temporal_fit_score"]
            + overlay_weight * work[column]
        )
        month_rhos: list[float] = []
        month_top3: list[float] = []
        month_top1: list[int] = []
        season_rhos: list[float] = []
        season_top2: list[float] = []
        season_top1: list[int] = []
        for _, group in work.groupby(["admin_dong_code", "bundle_id"], sort=False):
            month_rhos.append(
                _safe_spearman(group["temporal_fit_score"], group["variant_score"])
            )
            base_order = group.sort_values(
                ["temporal_fit_score", "month"],
                ascending=[False, True],
                kind="mergesort",
            )
            variant_order = group.sort_values(
                ["variant_score", "month"],
                ascending=[False, True],
                kind="mergesort",
            )
            month_top3.append(
                len(set(base_order.head(3)["month"]) & set(variant_order.head(3)["month"]))
                / 3.0
            )
            month_top1.append(int(base_order.iloc[0]["month"] == variant_order.iloc[0]["month"]))
            seasonal = group.groupby("season", as_index=False)[
                ["temporal_fit_score", "variant_score"]
            ].mean()
            season_rhos.append(
                _safe_spearman(seasonal["temporal_fit_score"], seasonal["variant_score"])
            )
            base_season = seasonal.sort_values(
                ["temporal_fit_score", "season"],
                ascending=[False, True],
                kind="mergesort",
            )
            variant_season = seasonal.sort_values(
                ["variant_score", "season"],
                ascending=[False, True],
                kind="mergesort",
            )
            season_top2.append(
                len(
                    set(base_season.head(2)["season"])
                    & set(variant_season.head(2)["season"])
                )
                / 2.0
            )
            season_top1.append(
                int(base_season.iloc[0]["season"] == variant_season.iloc[0]["season"])
            )
        rows.append(
            {
                "experiment": experiment,
                "overlay_column": column,
                "overlay_weight": overlay_weight,
                "median_month_rank_spearman": float(np.median(month_rhos)),
                "mean_month_top3_recall": float(np.mean(month_top3)),
                "month_top1_retention": float(np.mean(month_top1)),
                "median_season_rank_spearman": float(np.median(season_rhos)),
                "mean_season_top2_recall": float(np.mean(season_top2)),
                "season_top1_retention": float(np.mean(season_top1)),
                "released_as_primary": False,
                "reason": (
                    "climate_LOYO_failed"
                    if experiment == "climate_overlay"
                    else "static_transit_has_no_month_variation"
                ),
            }
        )
    return pd.DataFrame(rows)


def _seasonality_strength_sensitivity(
    temporal_fit: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """Test every preregistered signal-strength scenario around neutral 50."""

    strengths = [
        float(value)
        for value in config["temporal_fit"]["seasonality_strength_scenarios"]
    ]
    if not strengths or any(value < 0 for value in strengths):
        raise ValueError("Seasonality strength scenarios must be non-negative")
    baseline = temporal_fit.copy()
    baseline["season"] = baseline["month"].map(
        {month: season for season, months in SEASON_MONTHS.items() for month in months}
    )
    rows: list[dict[str, Any]] = []
    for strength in strengths:
        work = baseline.copy()
        work["variant_score"] = 50.0 + strength * (
            work["temporal_fit_score"] - 50.0
        )
        month_rhos: list[float] = []
        month_top3: list[float] = []
        month_top1: list[int] = []
        season_rhos: list[float] = []
        season_top2: list[float] = []
        season_top1: list[int] = []
        for _, group in work.groupby(["admin_dong_code", "bundle_id"], sort=False):
            month_rhos.append(
                _safe_spearman(group["temporal_fit_score"], group["variant_score"])
            )
            base_order = group.sort_values(
                ["temporal_fit_score", "month"],
                ascending=[False, True],
                kind="mergesort",
            )
            variant_order = group.sort_values(
                ["variant_score", "month"],
                ascending=[False, True],
                kind="mergesort",
            )
            month_top3.append(
                len(set(base_order.head(3)["month"]) & set(variant_order.head(3)["month"]))
                / 3.0
            )
            month_top1.append(int(base_order.iloc[0]["month"] == variant_order.iloc[0]["month"]))
            seasonal = group.groupby("season", as_index=False).agg(
                baseline_score=("temporal_fit_score", "mean"),
                variant_score=("variant_score", "mean"),
            )
            season_rhos.append(
                _safe_spearman(seasonal["baseline_score"], seasonal["variant_score"])
            )
            base_seasons = seasonal.sort_values(
                ["baseline_score", "season"], ascending=[False, True], kind="mergesort"
            )
            variant_seasons = seasonal.sort_values(
                ["variant_score", "season"], ascending=[False, True], kind="mergesort"
            )
            season_top2.append(
                len(
                    set(base_seasons.head(2)["season"])
                    & set(variant_seasons.head(2)["season"])
                )
                / 2.0
            )
            season_top1.append(
                int(base_seasons.iloc[0]["season"] == variant_seasons.iloc[0]["season"])
            )
        rows.append(
            {
                "seasonality_strength": strength,
                "destructive_neutral_signal": bool(np.isclose(strength, 0.0)),
                "median_month_rank_spearman": float(np.median(month_rhos)),
                "mean_month_top3_recall": float(np.mean(month_top3)),
                "month_top1_retention": float(np.mean(month_top1)),
                "median_season_rank_spearman": float(np.median(season_rhos)),
                "mean_season_top2_recall": float(np.mean(season_top2)),
                "season_top1_retention": float(np.mean(season_top1)),
                "interpretation": (
                    "destructive_no_signal_stress"
                    if np.isclose(strength, 0.0)
                    else "positive_amplitude_rank_invariance_check"
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("seasonality_strength").reset_index(drop=True)


def _run_temporal_layer(
    root: Path,
    master: pd.DataFrame,
    scores: pd.DataFrame,
    specialty_config: Mapping[str, Any],
    bundle_config: Mapping[str, Any],
) -> dict[str, Any]:
    output_dir = root / TEMPORAL_OUTPUT
    figure_dir = root / STAGE2_REPORT / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    temporal_config_path = root / "configs/model_v1/temporal_model.yaml"
    temporal_config = yaml.safe_load(temporal_config_path.read_text(encoding="utf-8"))
    if not isinstance(temporal_config, dict):
        raise ValueError("Temporal config must be a YAML mapping")

    nhis_path = root / str(temporal_config["source"]["nhis_monthly_raw"])
    nhis = read_nhis_monthly_cp949(
        nhis_path,
        encoding=str(temporal_config["source"]["nhis_encoding"]),
        expected_years=temporal_config["nhis_seasonality"]["observed_years"],
        expected_policy_sigungu=int(temporal_config["source"]["expected_policy_sigungu"]),
    )
    holiday_path = root / "14_v6_external_data/stage2_temporal/raw/official_public_holidays_2022_2023.csv"
    holidays = pd.read_csv(holiday_path, low_memory=False)
    holidays["date"] = pd.to_datetime(holidays["date"], errors="raise")
    working_days = build_working_day_calendar(
        temporal_config["nhis_seasonality"]["observed_years"], holidays
    )
    seasonality = calculate_service_seasonality(nhis, working_days, temporal_config)
    monthly_resolution = assess_monthly_resolution(
        seasonality.service_year_month, temporal_config
    )
    if monthly_resolution.summary["monthly_resolution_pass"]:
        raise RuntimeError(
            "Current release contract expects the preregistered monthly gate to fail; "
            "review evidence before changing resolution"
        )
    bundle_seasonality_result = build_bundle_seasonality_from_yearly_evidence(
        seasonality.service_year_month, bundle_config, temporal_config
    )
    bundle_seasonality = bundle_seasonality_result.bundle_month

    weather_path = root / "14_v6_external_data/stage2_temporal/raw/nasa_power_admin_dong_daily_1991_2025.parquet"
    weather_daily = pd.read_parquet(weather_path)
    weather_daily["admin_dong_code"] = (
        weather_daily["admin_dong_code"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(8)
    )
    monthly_climate = build_monthly_climate_risk(
        weather_daily,
        temporal_config,
        units=WEATHER_UNITS,
        enforce_config_contract=True,
    )
    representatives, weather_mapping = _weather_grid_representatives(
        weather_daily, WEATHER_UNITS
    )
    climate_monthly_stability = climate_leave_one_year_out_stability(
        representatives, temporal_config, units=WEATHER_UNITS
    )
    climate_season_detail, climate_season_summary = _climate_seasonal_stability(
        representatives, temporal_config
    )
    nhis_season_detail = bundle_seasonality_result.reliability_detail
    nhis_season_summary = dict(bundle_seasonality_result.summary)

    transit = _static_transit_fit(master)
    temporal_fit = build_region_bundle_month_fit(
        master,
        bundle_seasonality,
        monthly_climate,
        transit,
        temporal_config,
        enforce_config_contract=True,
        planning_resolution_override="season",
    )
    recommendations = select_primary_fallback_seasons(
        temporal_fit, temporal_config, enforce_config_contract=True
    )
    seasonal_fit = aggregate_monthly_fit_to_season(temporal_fit)
    primary_context = seasonal_fit[
        [
            "admin_dong_code",
            "bundle_id",
            "season",
            "climate_operational_risk",
            "climate_operational_fit",
            "transit_time_fit",
        ]
    ].rename(
        columns={
            "season": "primary_season",
            "climate_operational_risk": "primary_climate_risk_advisory",
            "climate_operational_fit": "primary_climate_fit_advisory",
            "transit_time_fit": "static_transit_context",
        }
    )
    recommendations = recommendations.merge(
        primary_context,
        on=["admin_dong_code", "bundle_id", "primary_season"],
        how="left",
        validate="one_to_one",
    )
    fallback_context = seasonal_fit[
        ["admin_dong_code", "bundle_id", "season", "climate_operational_risk"]
    ].rename(
        columns={
            "season": "fallback_season",
            "climate_operational_risk": "fallback_climate_risk_advisory",
        }
    )
    recommendations = recommendations.merge(
        fallback_context,
        on=["admin_dong_code", "bundle_id", "fallback_season"],
        how="left",
        validate="one_to_one",
    )
    recommendations = recommendations.merge(
        nhis_season_detail[
            [
                "bundle_id",
                "season_rank_spearman",
                "top1_season_agreement",
                "bundle_reliability_tier",
                "bundle_reliability_lambda",
                "released_primary_season",
                "released_fallback_season",
                "released_primary_fallback_margin",
                "released_primary_matches_both_year_top1",
                "released_primary_in_both_year_top2",
                "validation_release_direction_concordant",
                "temporal_signal_available",
            ]
        ].rename(
            columns={
                "season_rank_spearman": "nhis_bundle_season_loyo_spearman",
                "top1_season_agreement": "nhis_bundle_top1_agreement",
                "temporal_signal_available": (
                    "validated_bundle_temporal_signal_available"
                ),
            }
        ),
        on="bundle_id",
        how="left",
        validate="many_to_one",
    )
    recommendations["selection_matches_validated_bundle_profile"] = (
        recommendations["primary_season"].eq(
            recommendations["released_primary_season"]
        )
        & recommendations["fallback_season"].eq(
            recommendations["released_fallback_season"]
        )
    )
    if not recommendations["selection_matches_validated_bundle_profile"].all():
        mismatched = sorted(
            recommendations.loc[
                ~recommendations["selection_matches_validated_bundle_profile"],
                "bundle_id",
            ].unique()
        )
        raise RuntimeError(
            "Released season selector diverges from validated bundle profile: "
            + ", ".join(map(str, mismatched))
        )
    recommendations["season_evidence_confidence"] = np.where(
        recommendations["released_primary_matches_both_year_top1"].eq(1),
        "high_two_year_top1_concordant",
        np.where(
            recommendations["released_primary_in_both_year_top2"].eq(1),
            "moderate_two_year_top2_concordant",
            "low_release_not_supported",
        ),
    )
    recommendations = apply_season_release_contract(
        recommendations,
        temporal_fit,
        temporal_config,
        enforce_config_contract=True,
    )
    recommendations["primary_season_decision_status"] = np.where(
        recommendations["released_primary_matches_both_year_top1"].eq(1),
        "two_year_top1_concordant_macro_prior",
        "provisional_primary_fallback_pair_year_top1_disagrees",
    )
    recommendations.loc[
        recommendations["temporal_signal_available"]
        & recommendations["released_primary_matches_both_year_top1"].eq(1),
        "release_status",
    ] = "provisional_macro_season_prior_two_year_top1_concordant"
    recommendations.loc[
        recommendations["temporal_signal_available"]
        & recommendations["released_primary_matches_both_year_top1"].ne(1),
        "release_status",
    ] = "provisional_primary_fallback_pair_year_top1_disagrees"
    unavailable = ~recommendations["temporal_signal_available"]
    exact_fields = ["primary_month", "fallback_month", "recommended_date", "fallback_date"]
    if recommendations[exact_fields].notna().any().any():
        raise RuntimeError("Exact month/date leaked into the season-only release")
    temporal_ablation = run_temporal_ablation(
        temporal_fit,
        temporal_config,
        source_components={
            "nhis_observed_utilization_seasonality": ("seasonal_service_fit",),
        },
    )
    seasonal_ablation = _seasonal_ablation_metrics(temporal_fit, temporal_config)
    advisory_sensitivity = _operational_advisory_sensitivity(
        temporal_fit, temporal_config
    )
    seasonality_strength_sensitivity = _seasonality_strength_sensitivity(
        temporal_fit, temporal_config
    )
    temporal_audit = _build_temporal_audit(
        root,
        master=master,
        scores=scores,
        nhis=nhis,
        holidays=holidays,
        service_month=seasonality.service_month,
        weather_daily=weather_daily,
        monthly_climate=monthly_climate,
        temporal_fit=temporal_fit,
        specialty_config=specialty_config,
        temporal_config=temporal_config,
    )

    write_csv(temporal_audit, output_dir / "temporal_data_audit.csv")
    write_parquet(nhis, output_dir / "nhis_policy_sigungu_service_month.parquet")
    write_csv(working_days, output_dir / "official_working_day_calendar.csv")
    write_csv(seasonality.service_month, output_dir / "service_month_seasonality.csv")
    write_csv(seasonality.service_year_month, output_dir / "service_year_month_evidence.csv")
    write_csv(monthly_resolution.detail, output_dir / "monthly_resolution_gate_by_service.csv")
    write_json(monthly_resolution.summary, output_dir / "monthly_resolution_gate_summary.json")
    write_csv(bundle_seasonality, output_dir / "bundle_month_seasonality.csv")
    write_csv(monthly_climate, output_dir / "monthly_climate_operational_risk.csv")
    write_csv(weather_mapping, output_dir / "weather_series_mapping.csv")
    write_csv(climate_monthly_stability.detail, output_dir / "climate_monthly_loyo.csv")
    write_json(climate_monthly_stability.summary, output_dir / "climate_monthly_loyo_summary.json")
    write_csv(climate_season_detail, output_dir / "climate_seasonal_loyo.csv")
    write_json(climate_season_summary, output_dir / "climate_seasonal_loyo_summary.json")
    write_csv(nhis_season_detail, output_dir / "nhis_seasonal_loyo.csv")
    write_json(nhis_season_summary, output_dir / "nhis_seasonal_loyo_summary.json")
    write_parquet(temporal_fit, output_dir / "region_bundle_month_scores_diagnostic.parquet")
    write_parquet(seasonal_fit, output_dir / "region_bundle_season_scores.parquet")
    write_csv(recommendations, output_dir / "region_bundle_season_recommendations.csv")
    write_csv(
        _column_manifest(recommendations, "stage2b_recommendation"),
        output_dir / "stage2b_recommendation_column_manifest.csv",
    )
    write_csv(temporal_ablation.summary, output_dir / "temporal_ablation_summary.csv")
    write_parquet(temporal_ablation.score_variants, output_dir / "temporal_ablation_runs.parquet")
    write_csv(seasonal_ablation, output_dir / "temporal_seasonal_ablation_summary.csv")
    write_csv(
        advisory_sensitivity,
        output_dir / "operational_advisory_overlay_sensitivity.csv",
    )
    write_csv(
        seasonality_strength_sensitivity,
        output_dir / "seasonality_strength_sensitivity.csv",
    )

    service_labels, _ = _service_metadata(specialty_config)
    service_profile = seasonality.service_month.pivot(
        index="month", columns="service_id", values="seasonality_modifier"
    ).sort_index()
    service_profile_manifest = _manifest(
        service_profile.columns,
        labels=service_labels,
        granularity="province_common",
    )
    service_figure = render_temporal_month_profile_heatmap(
        service_profile,
        service_profile_manifest,
        figure_dir,
        center=1.0,
        stem="stage2b_service_month_profile_diagnostic",
        title="Stage 2B Common Service-Month Profile - Diagnostic Only",
    )
    bundle_profile = bundle_seasonality.pivot(
        index="month", columns="bundle_id", values="seasonality_modifier"
    ).sort_index()
    bundle_labels = {
        str(row["bundle_id"]): str(row["bundle_name_ko"])
        for row in bundle_config["bundles"]
    }
    bundle_profile_manifest = _manifest(
        bundle_profile.columns,
        labels=bundle_labels,
        granularity="province_common",
    )
    bundle_figure = render_temporal_month_profile_heatmap(
        bundle_profile,
        bundle_profile_manifest,
        figure_dir,
        center=1.0,
        stem="stage2b_bundle_month_profile_diagnostic",
        title="Stage 2B Recommended-Bundle Month Profile - Diagnostic Only",
    )
    climate_with_region = monthly_climate.merge(
        master[["admin_dong_code", "policy_sigungu_name"]],
        on="admin_dong_code",
        validate="many_to_one",
    )
    climate_profile = climate_with_region.pivot_table(
        index="month",
        columns="policy_sigungu_name",
        values="climate_operational_fit",
        aggfunc="mean",
    ).sort_index()
    climate_manifest = _manifest(
        climate_profile.columns,
        granularity="policy_sigungu",
    )
    climate_figure = render_temporal_month_profile_heatmap(
        climate_profile,
        climate_manifest,
        figure_dir,
        center=50.0,
        stem="stage2b_climate_month_fit_by_sigungu_diagnostic",
        title="Stage 2B Long-Run Climate Fit by Policy Sigungu - Diagnostic Only",
        annotate=False,
    )
    temporal_profile = temporal_fit.pivot_table(
        index="month", columns="bundle_id", values="temporal_fit_score", aggfunc="mean"
    ).sort_index()
    temporal_profile_manifest = _manifest(
        temporal_profile.columns,
        labels=bundle_labels,
        granularity="province_common",
    )
    temporal_figure = render_temporal_month_profile_heatmap(
        temporal_profile,
        temporal_profile_manifest,
        figure_dir,
        center=50.0,
        stem="stage2b_temporal_fit_month_diagnostic",
        title="Stage 2B Integrated Month Fit - Diagnostic Substrate, Season Release",
    )

    return {
        "config": temporal_config,
        "temporal_audit_rows": int(len(temporal_audit)),
        "nhis_rows": int(len(nhis)),
        "service_month_rows": int(len(seasonality.service_month)),
        "seasonality_reliability_tiers": seasonality.service_month[
            "seasonality_reliability_tier"
        ].value_counts().to_dict(),
        "monthly_resolution": dict(monthly_resolution.summary),
        "monthly_resolution_service_pass_count": int(
            monthly_resolution.detail["monthly_resolution_pass"].sum()
        ),
        "weather_daily_rows": int(len(weather_daily)),
        "weather_series_count": int(len(weather_mapping)),
        "unique_weather_series_count": int(weather_mapping["weather_series_id"].nunique()),
        "monthly_climate_rows": int(len(monthly_climate)),
        "climate_risk_min": float(monthly_climate["climate_operational_risk"].min()),
        "climate_risk_max": float(monthly_climate["climate_operational_risk"].max()),
        "climate_monthly_stability": dict(climate_monthly_stability.summary),
        "climate_seasonal_stability": climate_season_summary,
        "nhis_seasonal_stability": nhis_season_summary,
        "nhis_bundle_reliability": nhis_season_detail.to_dict(orient="records"),
        "temporal_fit_rows": int(len(temporal_fit)),
        "seasonal_fit_rows": int(len(seasonal_fit)),
        "recommendation_rows": int(len(recommendations)),
        "actionable_recommendation_rows": int(
            recommendations["temporal_signal_available"].sum()
        ),
        "no_signal_recommendation_rows": int(unavailable.sum()),
        "unavailable_rows_with_fabricated_season_count": int(
            recommendations.loc[
                unavailable, ["primary_season", "fallback_season"]
            ].notna().sum().sum()
        ),
        "exact_month_date_non_null_count": int(
            recommendations[exact_fields].notna().sum().sum()
        ),
        "replan_trigger_status_values": sorted(
            recommendations["replan_trigger_status"].dropna().astype(str).unique()
        ),
        "date_specific_replan_trigger_available_count": int(
            recommendations["date_specific_replan_trigger_available"].sum()
        ),
        "fallback_distinct_fraction": float(
            recommendations.loc[
                recommendations["temporal_signal_available"],
                "fallback_distinct_from_primary",
            ].mean()
        ),
        "selection_matches_validated_bundle_profile_fraction": float(
            recommendations["selection_matches_validated_bundle_profile"].mean()
        ),
        "low_evidence_release_count": int(
            (
                recommendations["temporal_signal_available"]
                & recommendations["season_evidence_confidence"].eq(
                    "low_release_not_supported"
                )
            ).sum()
        ),
        "year_top1_disagreement_release_count": int(
            recommendations["primary_season_decision_status"]
            .eq("provisional_primary_fallback_pair_year_top1_disagrees")
            .sum()
        ),
        "temporal_ablation_run_count": int(len(temporal_ablation.summary)),
        "temporal_ablation": temporal_ablation.summary.to_dict(orient="records"),
        "seasonal_ablation": seasonal_ablation.to_dict(orient="records"),
        "operational_advisory_sensitivity": advisory_sensitivity.to_dict(orient="records"),
        "seasonality_strength_sensitivity": seasonality_strength_sensitivity.to_dict(
            orient="records"
        ),
        "primary_component_names": list(
            temporal_config["temporal_fit"]["component_weights"]
        ),
        "climate_is_primary_selector": bool(
            "climate_operational_fit"
            in temporal_config["temporal_fit"]["component_weights"]
        ),
        "transit_is_primary_selector": bool(
            "transit_time_fit" in temporal_config["temporal_fit"]["component_weights"]
        ),
        "artifacts": {
            "service_profile": dict(service_figure.artifacts),
            "bundle_profile": dict(bundle_figure.artifacts),
            "climate_profile": dict(climate_figure.artifacts),
            "temporal_profile": dict(temporal_figure.artifacts),
        },
    }


def _column_manifest(frame: pd.DataFrame, layer: str) -> pd.DataFrame:
    labels_ko = {
        "admin_dong_code": "행정동 코드",
        "admin_dong_name": "행정동명",
        "policy_sigungu_name": "정책 시군명",
        "service_id": "진료 서비스 ID",
        "service_name_ko": "진료 서비스명",
        "specialty_gap_score": "진료과별 상대 격차 점수",
        "specialty_gap_rank": "진료과별 결정적 순위",
        "health_context_gap_score": "진료과 특이 건강부담 격차",
        "age_structure_gap_score": "진료과 특이 연령구조 격차",
        "specialty_supply_gap_score": "진료과 공급구성 격차",
        "specialty_excess_access_gap_score": "일반의료 대비 추가 접근격차",
        "public_support_gap_score": "일반의료 대비 공공지원 접근격차",
        "bundle_id": "권고 서비스 번들 ID",
        "bundle_name_ko": "권고 서비스 번들명",
        "bundle_gap_score": "권고 서비스 번들 격차 점수",
        "month": "진단용 월",
        "season": "배포 계절",
        "temporal_fit_score": "상대적 계절·운영 적합도",
        "primary_season": "우선 권고 계절",
        "fallback_season": "대체 권고 계절",
        "primary_season_decision_status": "우선 계절 근거 판정 상태",
        "season_evidence_confidence": "계절 근거 신뢰 등급",
        "temporal_signal_available": "배포 가능한 계절 신호 여부",
        "release_status": "계절 권고 배포 상태",
        "replan_trigger_status": "재계획 트리거 가용 상태",
        "date_specific_replan_trigger_available": "날짜별 재계획 트리거 가용 여부",
        "climate_role": "기후 자료의 모델 역할",
        "transit_role": "대중교통 자료의 모델 역할",
        "bundle_reliability_tier": "번들 계절 신뢰 등급",
        "bundle_reliability_lambda": "번들 계절 축소 계수",
        "released_primary_fallback_margin": "검증 프로필 우선·대체 계절 점수차",
        "primary_fallback_score_margin": "배포 우선·대체 계절 점수차",
        "nhis_bundle_season_loyo_spearman": "건보 번들 계절 연도간 순위상관",
        "nhis_bundle_top1_agreement": "건보 번들 연도간 최우선 계절 일치",
        "climate_operational_fit": "장기 기후 운영 적합도",
        "seasonal_service_fit": "공통 진료이용 계절 적합도",
        "transit_time_fit": "정적 대중교통 기반 적합도",
    }
    rows = []
    for order, column in enumerate(frame.columns, start=1):
        lowered = str(column).lower()
        if lowered in {"admin_dong_code", "service_id", "bundle_id"}:
            role = "key"
        elif "rank" in lowered or "score" in lowered or "recommend" in lowered:
            role = "model_output"
        elif "status" in lowered or "source" in lowered or "definition" in lowered:
            role = "audit_metadata"
        elif "count" in lowered or "time" in lowered or "rate" in lowered:
            role = "source_or_derived_feature"
        else:
            role = "model_component_or_metadata"
        rows.append(
            {
                "feature_id": f"{layer.upper()}_{order:03d}",
                "source_column": str(column),
                "label_ko": labels_ko.get(str(column), str(column)),
                "label_en": str(column).replace("_", " "),
                "layer": layer,
                "role": role,
                "dtype": str(frame[column].dtype),
                "missing_count": int(frame[column].isna().sum()),
                "unique_count": int(frame[column].nunique(dropna=True)),
                "labeled": True,
            }
        )
    return pd.DataFrame(rows)


def _gate_frame(
    *,
    specialty_config: Mapping[str, Any],
    temporal_config: Mapping[str, Any],
    specialty_validation: Mapping[str, Any],
    correlation: Mapping[str, Any],
    healthmap: Mapping[str, Any],
    ablation: Any,
    temporal: Mapping[str, Any],
    hashes: Mapping[str, str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add(
        stage: str,
        gate: str,
        value: Any,
        rule: str,
        threshold: Any,
        passed: bool,
        *,
        severity: str = "hard",
        note: str = "",
    ) -> None:
        rows.append(
            {
                "stage": stage,
                "gate": gate,
                "value": value,
                "rule": rule,
                "threshold": threshold,
                "passed": bool(passed),
                "severity": severity,
                "note": note,
            }
        )

    sg = specialty_config["quality_gates"]
    tg = temporal_config["quality_gates"]
    add(
        "contract",
        "frozen_canonical_sha256",
        hashes["frozen_canonical"],
        "==",
        specialty_config["source"]["frozen_canonical_sha256"],
        hashes["frozen_canonical"]
        == specialty_config["source"]["frozen_canonical_sha256"],
    )
    add(
        "contract",
        "road_master_sha256",
        hashes["road_master"],
        "==",
        specialty_config["source"]["road_master_sha256"],
        hashes["road_master"] == specialty_config["source"]["road_master_sha256"],
    )
    add(
        "contract",
        "stage1_config_sha256",
        hashes["stage1_config"],
        "==",
        specialty_config["source"]["stage1_config_sha256"],
        hashes["stage1_config"] == specialty_config["source"]["stage1_config_sha256"],
    )
    add(
        "2A",
        "specialty_result_validation",
        specialty_validation["valid"],
        "is",
        True,
        bool(specialty_validation["valid"]),
    )
    for name, value, threshold in (
        (
            "focused_physician_rows",
            specialty_validation["focused_physician_rows"],
            sg["expected_primary_focused_rows"],
        ),
        (
            "adjunct_dental_rows",
            specialty_validation["adjunct_dental_rows"],
            sg["expected_adjunct_dental_rows"],
        ),
        (
            "component_missing_count",
            specialty_validation["component_missing_count"],
            sg["component_missing_count_max"],
        ),
    ):
        rule = "<=" if name == "component_missing_count" else "=="
        passed = value <= threshold if rule == "<=" else value == threshold
        add("2A", name, value, rule, threshold, passed)
    primary_sep = correlation["primary_separation"]
    primary_overlap = correlation["primary_stage1_overlap"]
    for name, key, rule, threshold, passed in (
        (
            "cross_bundle_median_abs_rho",
            "cross_bundle_median_abs_rho",
            "<=",
            float(sg["cross_bundle_abs_rho_median_max"]),
            float(primary_sep["cross_bundle_median_abs_rho"])
            <= float(sg["cross_bundle_abs_rho_median_max"]),
        ),
        (
            "cross_bundle_abs_rho_ge_080_rate",
            "cross_bundle_abs_rho_ge_0_80_rate",
            "<=",
            float(sg["cross_bundle_abs_rho_080_rate_max"]),
            float(primary_sep["cross_bundle_abs_rho_ge_0_80_rate"])
            <= float(sg["cross_bundle_abs_rho_080_rate_max"]),
        ),
        (
            "cross_bundle_abs_rho_ge_095_pairs",
            "cross_bundle_abs_rho_ge_0_95_count",
            "<=",
            int(sg["cross_bundle_abs_rho_095_pairs_max"]),
            int(primary_sep["cross_bundle_abs_rho_ge_0_95_count"])
            <= int(sg["cross_bundle_abs_rho_095_pairs_max"]),
        ),
    ):
        add("2A-correlation", name, primary_sep[key], rule, threshold, passed)
    add(
        "2A-correlation",
        "stage1_need_median_abs_rho",
        primary_overlap["stage1_overlap_median_abs_rho"],
        "<=",
        sg["stage1_need_abs_rho_median_max"],
        float(primary_overlap["stage1_overlap_median_abs_rho"])
        <= float(sg["stage1_need_abs_rho_median_max"]),
    )
    add(
        "2A-correlation",
        "stage1_need_max_abs_rho",
        primary_overlap["stage1_overlap_max_abs_rho"],
        "<=",
        sg["stage1_need_abs_rho_max"],
        float(primary_overlap["stage1_overlap_max_abs_rho"])
        <= float(sg["stage1_need_abs_rho_max"]),
    )
    within_quality = correlation["within_quality"]
    add(
        "2A-correlation",
        "within_sigungu_axis_auc",
        within_quality.get("axis_auc"),
        ">=",
        sg["within_sigungu_bundle_axis_auc_min"],
        float(within_quality.get("axis_auc") or -np.inf)
        >= float(sg["within_sigungu_bundle_axis_auc_min"]),
        note="preregistered exclusive clinical-block separation gate",
    )
    add(
        "2A-correlation",
        "within_sigungu_block_contrast",
        within_quality.get("block_contrast"),
        ">=",
        sg["within_sigungu_bundle_block_contrast_min"],
        float(within_quality.get("block_contrast") or -np.inf)
        >= float(sg["within_sigungu_bundle_block_contrast_min"]),
        note="preregistered exclusive clinical-block separation gate",
    )
    add(
        "2A-correlation",
        "within_sigungu_axis_auc_bootstrap_ci05",
        correlation["within_bootstrap_axis_auc_ci05"],
        ">=",
        sg["within_sigungu_bundle_axis_auc_min"],
        float(correlation["within_bootstrap_axis_auc_ci05"])
        >= float(sg["within_sigungu_bundle_axis_auc_min"]),
        severity="advisory",
        note=(
            "uncertainty diagnostic for only 11 independent sigungu blocks; "
            "the preregistered hard gate is the point estimate"
        ),
    )
    add(
        "2A-correlation",
        "within_sigungu_block_contrast_bootstrap_ci05",
        correlation["within_bootstrap_block_contrast_ci05"],
        ">=",
        sg["within_sigungu_bundle_block_contrast_min"],
        float(correlation["within_bootstrap_block_contrast_ci05"])
        >= float(sg["within_sigungu_bundle_block_contrast_min"]),
        severity="advisory",
        note="small-n block-bootstrap uncertainty is reported, not tuned away",
    )
    add(
        "2A-correlation",
        "health_context_abs_rho_095_pairs",
        correlation["health_context_abs_rho_095_pairs"],
        "<=",
        sg["health_context_abs_rho_095_pairs_max"],
        int(correlation["health_context_abs_rho_095_pairs"])
        <= int(sg["health_context_abs_rho_095_pairs_max"]),
    )
    add(
        "2A-correlation",
        "bootstrap_replicates",
        correlation["bootstrap_n"],
        ">=",
        sg["bootstrap_replicates_min"],
        int(correlation["bootstrap_n"]) >= int(sg["bootstrap_replicates_min"]),
        note="final release precision; smoke runs below this value remain incomplete",
    )
    expected_bootstrap_edges = 16 * 15 // 2
    add(
        "2A-correlation",
        "bootstrap_all_service_pairs_complete",
        {
            "primary": correlation["bootstrap_edge_count"],
            "within_sigungu": correlation["within_bootstrap_edge_count"],
        },
        "==",
        {"primary": expected_bootstrap_edges, "within_sigungu": expected_bootstrap_edges},
        int(correlation["bootstrap_edge_count"]) == expected_bootstrap_edges
        and int(correlation["within_bootstrap_edge_count"])
        == expected_bootstrap_edges,
        note="all unordered pairs among the 16 service scores must be present",
    )
    quality_valid_fraction = min(
        correlation["bootstrap_quality_min_valid_fraction"],
        correlation["within_bootstrap_quality_min_valid_fraction"],
    )
    add(
        "2A-correlation",
        "bootstrap_quality_metric_valid_fraction",
        quality_valid_fraction,
        ">=",
        sg["bootstrap_metric_valid_fraction_min"],
        float(quality_valid_fraction)
        >= float(sg["bootstrap_metric_valid_fraction_min"]),
    )
    add(
        "2A-correlation",
        "bootstrap_valid_fraction",
        min(
            correlation["bootstrap_min_valid_fraction"],
            correlation["within_bootstrap_min_valid_fraction"],
        ),
        ">=",
        0.80,
        min(
            correlation["bootstrap_min_valid_fraction"],
            correlation["within_bootstrap_min_valid_fraction"],
        )
        >= 0.80,
        severity="diagnostic",
        note="edge coverage diagnostic; threshold is not a YAML completion gate",
    )
    add(
        "2A-correlation",
        "final_service_score_abs_rho_098_pairs",
        correlation["score_abs_rho_098_pairs"],
        "<=",
        sg["exact_duplicate_service_score_pairs_max"],
        int(correlation["score_abs_rho_098_pairs"])
        <= int(sg["exact_duplicate_service_score_pairs_max"]),
    )
    add(
        "2A-correlation",
        "full_atlas_unexplained_abs_rho_098_pairs",
        correlation["full_atlas_unexplained_after_review_count"],
        "<=",
        sg["unexplained_abs_rho_098_pairs_max"],
        int(correlation["full_atlas_unexplained_after_review_count"])
        <= int(sg["unexplained_abs_rho_098_pairs_max"]),
        note="structural aliases, deterministic lineage, and distinct-source review remain in the ledger",
    )
    add(
        "2A-correlation",
        "representative_atlas_unexplained_abs_rho_098_pairs",
        correlation["representative_unexplained_after_review_count"],
        "<=",
        sg["unexplained_abs_rho_098_pairs_max"],
        int(correlation["representative_unexplained_after_review_count"])
        <= int(sg["unexplained_abs_rho_098_pairs_max"]),
    )
    add(
        "2A-correlation",
        "full_atlas_feature_count",
        correlation["full_atlas_feature_count"],
        "==",
        sg["expected_full_stage2a_atlas_features"],
        int(correlation["full_atlas_feature_count"])
        == int(sg["expected_full_stage2a_atlas_features"]),
    )
    add(
        "2A-correlation",
        "discriminating_atlas_feature_count",
        correlation["discriminating_atlas_feature_count"],
        "==",
        sg["expected_discriminating_atlas_features"],
        int(correlation["discriminating_atlas_feature_count"])
        == int(sg["expected_discriminating_atlas_features"]),
    )
    discriminating_sep = correlation["discriminating_separation"]
    add(
        "2A-correlation",
        "discriminating_atlas_cross_family_median_abs_rho",
        discriminating_sep["cross_bundle_median_abs_rho"],
        "<=",
        sg["cross_bundle_abs_rho_median_max"],
        float(discriminating_sep["cross_bundle_median_abs_rho"])
        <= float(sg["cross_bundle_abs_rho_median_max"]),
        note="the 120-axis separation view must meet at least the preregistered primary-atlas floor",
    )
    add(
        "2A-correlation",
        "discriminating_atlas_cross_family_abs_rho_ge_080_rate",
        discriminating_sep["cross_bundle_abs_rho_ge_0_80_rate"],
        "<=",
        sg["cross_bundle_abs_rho_080_rate_max"],
        float(discriminating_sep["cross_bundle_abs_rho_ge_0_80_rate"])
        <= float(sg["cross_bundle_abs_rho_080_rate_max"]),
    )
    add(
        "2A-correlation",
        "discriminating_atlas_cross_family_abs_rho_ge_095_pairs",
        discriminating_sep["cross_bundle_abs_rho_ge_0_95_count"],
        "<=",
        sg["cross_bundle_abs_rho_095_pairs_max"],
        int(discriminating_sep["cross_bundle_abs_rho_ge_0_95_count"])
        <= int(sg["cross_bundle_abs_rho_095_pairs_max"]),
    )
    add(
        "2A-correlation",
        "stable_feature_id_contract",
        correlation["stable_feature_id_contract_pass"],
        "is",
        True,
        bool(correlation["stable_feature_id_contract_pass"]),
    )
    add(
        "2A-correlation",
        "publication_artifact_formats_complete",
        correlation["publication_artifacts_complete"],
        "is",
        True,
        bool(correlation["publication_artifacts_complete"]),
    )
    add(
        "2A-correlation",
        "publication_manifest_label_and_id_coverage",
        correlation["publication_manifest_label_coverage"],
        "==",
        1.0,
        np.isclose(
            float(correlation["publication_manifest_label_coverage"]),
            1.0,
            atol=1e-12,
        ),
    )
    add(
        "2A-correlation",
        "interactive_external_script_count",
        correlation["interactive_external_script_count"],
        "==",
        0,
        int(correlation["interactive_external_script_count"]) == 0,
    )
    add(
        "2A-correlation",
        "admin_atlas_pairwise_n_range",
        {
            "min": correlation["admin_atlas_pairwise_n_min"],
            "max": correlation["admin_atlas_pairwise_n_max"],
        },
        "==",
        {
            "min": sg["expected_admin_atlas_pairwise_n"],
            "max": sg["expected_admin_atlas_pairwise_n"],
        },
        int(correlation["admin_atlas_pairwise_n_min"])
        == int(sg["expected_admin_atlas_pairwise_n"])
        and int(correlation["admin_atlas_pairwise_n_max"])
        == int(sg["expected_admin_atlas_pairwise_n"]),
    )
    add(
        "2A-correlation",
        "reviewed_distinct_source_abs_rho_098_pairs",
        correlation["high_correlation_empirical_review_pair_count"],
        "reported",
        "all retained with distinct meanings",
        True,
        severity="advisory",
        note="empirical collinearity is not disguised as exact duplication",
    )
    add(
        "2A-validation",
        "healthmap_directional_concordance",
        healthmap["directional_concordance"],
        ">=",
        sg["healthmap_directional_concordance_min"],
        float(healthmap["directional_concordance"])
        >= float(sg["healthmap_directional_concordance_min"]),
        severity="advisory",
        note="n=11 weak external directional check; never an optimization target",
    )

    ledger = ablation.ledger
    summary = ablation.run_summary
    policy_names = sorted(
        ledger.loc[
            ledger["scope"].eq("leave_one_sigungu_out"), "variant"
        ].astype(str)
    )
    expected_specs = expected_run_specs(
        specialty_config, policy_sigungu_names=policy_names
    )
    expected_ids = {spec.run_id for spec in expected_specs}
    observed_ids = set(ledger["run_id"].astype(str))
    add(
        "2A-ablation",
        "run_coverage",
        int(ledger["status"].eq("PASS").sum()),
        "==",
        int(len(ledger)),
        bool(
            ledger["status"].eq("PASS").all()
            and ledger["run_id"].is_unique
            and ledger["expected_score_rows"].eq(ledger["actual_score_rows"]).all()
            and observed_ids == expected_ids
        ),
    )
    for scope, expected_count, configured_threshold in (
        ("component", 10, sg["component_ablation_coverage"]),
        ("source_feature", 22, sg["feature_ablation_coverage"]),
    ):
        observed_pass = int(
            (ledger["scope"].eq(scope) & ledger["status"].eq("PASS")).sum()
        )
        coverage = observed_pass / expected_count
        add(
            "2A-ablation",
            f"{scope}_coverage",
            coverage,
            ">=",
            configured_threshold,
            coverage >= float(configured_threshold),
        )
    alpha = summary.loc[summary["scope"].eq("alpha")]
    add(
        "2A-ablation",
        "alpha_min_service_spearman",
        float(alpha["service_rank_spearman_min"].min()),
        ">=",
        sg["alpha_grid_specialty_spearman_min"],
        float(alpha["service_rank_spearman_min"].min())
        >= float(sg["alpha_grid_specialty_spearman_min"]),
    )
    add(
        "2A-ablation",
        "alpha_min_median_top20_jaccard",
        float(alpha["service_top20_jaccard_median"].min()),
        ">=",
        sg["alpha_grid_top20_jaccard_median_min"],
        float(alpha["service_top20_jaccard_median"].min())
        >= float(sg["alpha_grid_top20_jaccard_median_min"]),
    )
    add(
        "2A-ablation",
        "alpha_min_top20_jaccard",
        float(alpha["service_top20_jaccard_min"].min()),
        ">=",
        sg["alpha_grid_top20_jaccard_min"],
        float(alpha["service_top20_jaccard_min"].min())
        >= float(sg["alpha_grid_top20_jaccard_min"]),
    )
    add(
        "2A-ablation",
        "alpha_region_top3_retention",
        float(alpha["region_service_top3_retention"].min()),
        ">=",
        sg["alpha_grid_within_region_top3_retention_min"],
        float(alpha["region_service_top3_retention"].min())
        >= float(sg["alpha_grid_within_region_top3_retention_min"]),
    )
    source_feature = summary.loc[summary["scope"].eq("source_feature")]
    add(
        "2A-ablation",
        "source_feature_all_service_median_spearman",
        float(source_feature["service_rank_spearman_median"].min()),
        ">=",
        sg["default_feature_median_spearman_min"],
        float(source_feature["service_rank_spearman_median"].min())
        >= float(sg["default_feature_median_spearman_min"]),
        note=(
            "global-refit stability across all services; direct dependency "
            "stress is reported separately and is not diluted into this meaning"
        ),
    )
    add(
        "2A-ablation",
        "source_feature_all_service_top20_in_top30",
        float(source_feature["service_symmetric_top20_in_top30_median"].min()),
        ">=",
        sg["default_feature_median_top20_in_top30_min"],
        float(source_feature["service_symmetric_top20_in_top30_median"].min())
        >= float(sg["default_feature_median_top20_in_top30_min"]),
        note=(
            "global-refit boundary stability; affected-only stress remains advisory"
        ),
    )
    affected = ablation.affected_source_feature_summary
    add(
        "2A-ablation",
        "affected_source_feature_mapping_complete",
        {
            "run_count": int(len(affected)),
            "nonempty_affected": int(affected["affected_service_count"].gt(0).sum()),
            "direct_service_feature_instances": int(
                affected["affected_service_count"].sum()
            ),
        },
        "==",
        {
            "run_count": 22,
            "nonempty_affected": 22,
            "direct_service_feature_instances": int(
                sg["expected_direct_source_service_instances"]
            ),
        },
        len(affected) == 22
        and affected["affected_service_count"].gt(0).all()
        and int(affected["affected_service_count"].sum())
        == int(sg["expected_direct_source_service_instances"])
        and affected["run_id"].is_unique,
    )
    invariance_expected = affected.loc[
        affected["unaffected_invariance_expected"]
    ]
    add(
        "2A-ablation",
        "uncoupled_unaffected_services_invariant",
        int(invariance_expected["unaffected_services_invariant"].sum()),
        "==",
        int(len(invariance_expected)),
        bool(invariance_expected["unaffected_services_invariant"].all()),
        note=(
            "public-source shocks must not spill over; health/age shocks are "
            "explicitly coupled through the preregistered common-effect centering"
        ),
    )
    direct_changed = int(affected["affected_changed_service_count"].sum())
    add(
        "2A-ablation",
        "direct_source_service_instances_exercised",
        direct_changed,
        "==",
        sg["expected_direct_source_service_instances"],
        direct_changed == int(sg["expected_direct_source_service_instances"]),
        note="every mapped direct dependency must be exercised by its ablation",
    )
    max_source_component_ratio = float(
        affected["affected_delta_to_component_neutral_ratio"].max()
    )
    add(
        "2A-ablation",
        "source_feature_delta_not_greater_than_component_neutral_shock",
        max_source_component_ratio,
        "<=",
        sg["source_feature_to_component_neutral_delta_ratio_max"],
        max_source_component_ratio
        <= float(sg["source_feature_to_component_neutral_delta_ratio_max"])
        + 1e-12,
        note=(
            "a single source removal may be sensitive but must not exceed "
            "neutralising its entire parent component"
        ),
    )
    add(
        "2A-ablation",
        "affected_source_feature_median_spearman",
        float(affected["affected_rank_spearman_median"].min()),
        ">=",
        sg["default_feature_median_spearman_min"],
        float(affected["affected_rank_spearman_median"].min())
        >= float(sg["default_feature_median_spearman_min"]),
        severity="advisory",
        note=(
            "meaningful feature removal is allowed to change services that use it; "
            "reported as policy sensitivity, never diluted by unaffected services"
        ),
    )
    add(
        "2A-ablation",
        "affected_source_feature_top20_in_top30",
        float(affected["affected_symmetric_top20_in_top30_median"].min()),
        ">=",
        sg["default_feature_median_top20_in_top30_min"],
        float(affected["affected_symmetric_top20_in_top30_median"].min())
        >= float(sg["default_feature_median_top20_in_top30_min"]),
        severity="advisory",
        note="affected-service boundary sensitivity is disclosed without post-hoc tuning",
    )
    formula = summary.loc[summary["scope"].eq("scaler_formula")]
    add(
        "2A-ablation",
        "formula_scaler_median_spearman",
        float(formula["service_rank_spearman_median"].min()),
        ">=",
        sg["formula_median_spearman_min"],
        float(formula["service_rank_spearman_median"].min())
        >= float(sg["formula_median_spearman_min"]),
    )
    add(
        "2A-ablation",
        "formula_scaler_top20_in_top30",
        float(formula["service_symmetric_top20_in_top30_median"].min()),
        ">=",
        sg["formula_median_top20_in_top30_min"],
        float(formula["service_symmetric_top20_in_top30_median"].min())
        >= float(sg["formula_median_top20_in_top30_min"]),
    )

    monthly = temporal["monthly_resolution"]
    add(
        "2B-resolution",
        "safe_month_to_season_downgrade",
        monthly["recommended_planning_resolution"],
        "==",
        "season_after_month_gate_fail",
        (not bool(monthly["monthly_resolution_pass"]))
        and monthly["recommended_planning_resolution"] == "season",
    )
    add(
        "2B-evidence",
        "nhis_seasonal_loyo",
        temporal["nhis_seasonal_stability"],
        "pass",
        True,
        bool(temporal["nhis_seasonal_stability"]["seasonal_nhis_gate_pass"]),
    )
    nhis_bundle = temporal["nhis_seasonal_stability"]
    for gate_name, summary_key, rule, threshold in (
        (
            "bundle_validation_release_unit_match",
            "validation_release_unit_match_fraction",
            ">=",
            tg["bundle_validation_release_unit_match_min"],
        ),
        (
            "bundle_validation_release_direction_concordance",
            "validation_release_direction_concordance_fraction",
            ">=",
            tg["bundle_validation_release_direction_concordance_min"],
        ),
        (
            "bundle_release_profile_mismatch_count",
            "validation_release_profile_mismatch_count",
            "<=",
            tg["bundle_release_profile_mismatch_max"],
        ),
        (
            "bundle_released_primary_in_both_year_top2",
            "released_primary_in_both_year_top2_fraction",
            ">=",
            tg["bundle_released_primary_in_both_year_top2_min"],
        ),
    ):
        value = nhis_bundle[summary_key]
        passed = (
            float(value) >= float(threshold)
            if rule == ">="
            else float(value) <= float(threshold)
        )
        add("2B-evidence", gate_name, value, rule, threshold, passed)
    add(
        "2B-evidence",
        "climate_seasonal_loyo",
        temporal["climate_seasonal_stability"],
        "pass",
        True,
        bool(temporal["climate_seasonal_stability"]["seasonal_climate_gate_pass"]),
        severity="advisory",
        note="if failed, climate cannot remain a primary macro selector",
    )
    add(
        "2B-contract",
        "supported_primary_component_only",
        temporal["primary_component_names"],
        "==",
        ["seasonal_service_fit"],
        temporal["primary_component_names"] == ["seasonal_service_fit"],
        note="climate and static transit remain visible advisory/context fields",
    )
    add(
        "2B-contract",
        "unsupported_climate_transit_not_primary",
        {
            "climate": temporal["climate_is_primary_selector"],
            "transit": temporal["transit_is_primary_selector"],
        },
        "==",
        {"climate": False, "transit": False},
        not temporal["climate_is_primary_selector"]
        and not temporal["transit_is_primary_selector"],
    )
    for name, value, threshold in (
        ("temporal_audit_rows", temporal["temporal_audit_rows"], tg["expected_temporal_audit_min_rows"]),
        ("weather_daily_rows", temporal["weather_daily_rows"], tg["expected_weather_daily_rows"]),
        ("monthly_climate_rows", temporal["monthly_climate_rows"], tg["expected_monthly_climate_rows"]),
        ("temporal_fit_rows", temporal["temporal_fit_rows"], tg["expected_region_bundle_month_rows"]),
        ("recommendation_rows", temporal["recommendation_rows"], tg["expected_recommendation_rows"]),
    ):
        rule = ">=" if name == "temporal_audit_rows" else "=="
        passed = value >= threshold if rule == ">=" else value == threshold
        add("2B-contract", name, value, rule, threshold, passed)
    for name, value, threshold in (
        (
            "actionable_recommendation_rows",
            temporal["actionable_recommendation_rows"],
            tg["expected_actionable_recommendation_rows"],
        ),
        (
            "no_signal_recommendation_rows",
            temporal["no_signal_recommendation_rows"],
            tg["expected_no_signal_recommendation_rows"],
        ),
        (
            "unavailable_rows_with_fabricated_season_count",
            temporal["unavailable_rows_with_fabricated_season_count"],
            tg["expected_unavailable_rows_with_fabricated_season_count"],
        ),
    ):
        add("2B-contract", name, value, "==", threshold, value == threshold)
    expected_replan_status = str(
        temporal_config["temporal_fit"]["replan_trigger_status"]
    )
    add(
        "2B-contract",
        "replan_trigger_unavailable_without_forecast_and_operator_date",
        temporal["replan_trigger_status_values"],
        "==",
        [expected_replan_status],
        temporal["replan_trigger_status_values"] == [expected_replan_status],
    )
    add(
        "2B-contract",
        "date_specific_replan_trigger_available_rows",
        temporal["date_specific_replan_trigger_available_count"],
        "==",
        tg["expected_available_date_specific_replan_trigger_rows"],
        int(temporal["date_specific_replan_trigger_available_count"])
        == int(tg["expected_available_date_specific_replan_trigger_rows"]),
    )
    add(
        "2B-contract",
        "selector_matches_validated_bundle_profile",
        temporal["selection_matches_validated_bundle_profile_fraction"],
        ">=",
        tg["bundle_selector_validation_match_fraction_min"],
        float(temporal["selection_matches_validated_bundle_profile_fraction"])
        >= float(tg["bundle_selector_validation_match_fraction_min"]),
    )
    add(
        "2B-contract",
        "low_evidence_release_rows",
        temporal["low_evidence_release_count"],
        "<=",
        tg["low_evidence_release_count_max"],
        int(temporal["low_evidence_release_count"])
        <= int(tg["low_evidence_release_count_max"]),
    )
    add(
        "2B-evidence",
        "year_top1_disagreement_release_rows",
        temporal["year_top1_disagreement_release_count"],
        "==",
        0,
        int(temporal["year_top1_disagreement_release_count"]) == 0,
        severity="advisory",
        note=(
            "these rows release a primary/fallback season pair, not a robust "
            "single-season claim"
        ),
    )
    add(
        "2B-contract",
        "exact_month_date_non_null",
        temporal["exact_month_date_non_null_count"],
        "==",
        0,
        temporal["exact_month_date_non_null_count"] == 0,
    )
    add(
        "2B-contract",
        "fallback_distinct_fraction",
        temporal["fallback_distinct_fraction"],
        ">=",
        tg["fallback_distinct_from_primary_fraction_min"],
        float(temporal["fallback_distinct_fraction"])
        >= float(tg["fallback_distinct_from_primary_fraction_min"]),
    )
    temporal_ablation = pd.DataFrame(temporal["temporal_ablation"])
    advisory_sensitivity = pd.DataFrame(temporal["operational_advisory_sensitivity"])
    strength_sensitivity = pd.DataFrame(
        temporal["seasonality_strength_sensitivity"]
    )
    expected_strengths = sorted(
        float(value)
        for value in temporal_config["temporal_fit"][
            "seasonality_strength_scenarios"
        ]
    )
    add(
        "2B-ablation",
        "seasonality_strength_scenario_coverage",
        sorted(strength_sensitivity["seasonality_strength"].astype(float)),
        "==",
        expected_strengths,
        sorted(strength_sensitivity["seasonality_strength"].astype(float))
        == expected_strengths,
    )
    positive_strength = strength_sensitivity.loc[
        strength_sensitivity["seasonality_strength"].gt(0)
    ]
    positive_invariant = bool(
        not positive_strength.empty
        and positive_strength[
            [
                "median_month_rank_spearman",
                "mean_month_top3_recall",
                "month_top1_retention",
                "median_season_rank_spearman",
                "mean_season_top2_recall",
                "season_top1_retention",
            ]
        ]
        .ge(1.0 - 1e-12)
        .all()
        .all()
    )
    add(
        "2B-ablation",
        "positive_seasonality_strength_rank_invariance",
        positive_invariant,
        "is",
        True,
        positive_invariant,
        note="positive amplitude changes must not alter the released ordering",
    )
    expected_advisory_count = len(
        temporal_config["temporal_fit"]["weather_weight_scenarios"]
    ) + 1
    add(
        "2B-ablation",
        "operational_advisory_scenario_coverage_and_quarantine",
        {
            "rows": int(len(advisory_sensitivity)),
            "released_as_primary": int(
                advisory_sensitivity["released_as_primary"].astype(bool).sum()
            ),
        },
        "==",
        {"rows": expected_advisory_count, "released_as_primary": 0},
        len(advisory_sensitivity) == expected_advisory_count
        and not advisory_sensitivity["released_as_primary"].astype(bool).any(),
    )
    add(
        "2B-ablation",
        "component_source_coverage",
        int(len(temporal_ablation)),
        "==",
        2,
        len(temporal_ablation) == 2
        and int(temporal_ablation["ablation_type"].eq("component").sum()) == 1
        and int(temporal_ablation["ablation_type"].eq("source").sum()) == 1,
    )
    add(
        "2B-ablation",
        "minimum_rank_spearman",
        float(temporal_ablation["spearman"].min()),
        ">=",
        tg["default_ablation_median_spearman_min"],
        float(temporal_ablation["spearman"].min())
        >= float(tg["default_ablation_median_spearman_min"]),
        severity="diagnostic",
        note="destructive removal of the sole supported seasonal signal is expected to fail",
    )
    add(
        "2B-ablation",
        "minimum_top3_recall",
        float(temporal_ablation["top3_recall"].min()),
        ">=",
        tg["default_ablation_top3_recall_min"],
        float(temporal_ablation["top3_recall"].min())
        >= float(tg["default_ablation_top3_recall_min"]),
        severity="diagnostic",
        note="destructive removal of the sole supported seasonal signal is expected to fail",
    )
    return pd.DataFrame(rows)


def _write_stage2_reports(
    root: Path,
    *,
    gate: pd.DataFrame,
    correlation: Mapping[str, Any],
    healthmap_detail: pd.DataFrame,
    healthmap_summary: Mapping[str, Any],
    ablation: Any,
    temporal: Mapping[str, Any],
    timings: Mapping[str, float],
) -> None:
    report_dir = root / STAGE2_REPORT
    report_dir.mkdir(parents=True, exist_ok=True)
    ablation_summary = ablation.run_summary.sort_values(
        ["service_rank_spearman_min", "service_top20_jaccard_min"],
        ascending=[True, True],
    )
    affected_attention = ablation.affected_source_feature_summary.loc[
        ablation.affected_source_feature_summary["direct_dependency_attention"]
    ].sort_values("affected_rank_spearman_median")
    specialty_text = f"""# MEDIROAD MODEL V1 - Stage 2A Specialty Gap

## Outcome

Stage 2A contains 16 transparent service-gap scores for 153 admin dongs and five recommended, not staffing-confirmed, service bundles. It is not a patient-demand or causal model.

## Correlation separation

- Cross-clinical-block median |rho|: `{correlation['primary_separation']['cross_bundle_median_abs_rho']:.4f}`
- Cross-block |rho| >= .80 rate: `{correlation['primary_separation']['cross_bundle_abs_rho_ge_0_80_rate']:.4f}`
- Stage 1 Need median/max |rho|: `{correlation['primary_stage1_overlap']['stage1_overlap_median_abs_rho']:.4f}` / `{correlation['primary_stage1_overlap']['stage1_overlap_max_abs_rho']:.4f}`
- Within-sigungu high-correlation structure is reported separately; CHS uses exactly n=11.
- Within-sigungu bootstrap CI05 for AUC/contrast: `{correlation['within_bootstrap_axis_auc_ci05']:.4f}` / `{correlation['within_bootstrap_block_contrast_ci05']:.4f}`; these small-n uncertainty diagnostics are not tuned away.
- Bundle-score correlation is overlap-aware because bundles intentionally share services; its maximum is not optimized away.
- The complete lineage atlas has `{correlation['full_atlas_feature_count']}` axes; the discriminating companion has `{correlation['discriminating_atlas_feature_count']}` after collapsing only exact aliases and deterministic raw-to-component lineage.
- Discriminating-atlas cross-family median |rho| / |rho| >= .80 rate / |rho| >= .95 count: `{correlation['discriminating_separation']['cross_bundle_median_abs_rho']:.4f}` / `{correlation['discriminating_separation']['cross_bundle_abs_rho_ge_0_80_rate']:.4f}` / `{correlation['discriminating_separation']['cross_bundle_abs_rho_ge_0_95_count']}`.
- Unexplained |rho| >= .98 pairs after review: `{correlation['high_correlation_unexplained_after_review_count']}`.

## External directional check

{_md_table(healthmap_detail)}

Directional concordance: `{healthmap_summary['directional_concordance']:.3f}`. This is a weak n=11 holdout, never an optimization target.

## Exhaustive ablation

All `{len(ablation.ledger)}` preregistered runs completed. The most disruptive diagnostics are:

{_md_table(ablation_summary, ['run_id', 'scope', 'service_rank_spearman_min', 'service_top20_jaccard_min', 'region_service_top3_retention'], 15)}

The excess-access removal shock is intentionally retained: it demonstrates that specialty-specific travel burden supplies real differentiation rather than repeating general access.

## Direct source-dependency stress

Global refit stability remains a hard gate. The table below deliberately uses only services that directly depend on each source, so meaningful sensitivity is not hidden by unrelated services. It is advisory because deleting a genuine signal should change the services that use it. A single-source shock is separately bounded so it never exceeds neutralising its whole parent component.

The source-feature family is an explicit hybrid stress contract: services with multiple health/age inputs are rebuilt after leave-one-feature-out, including re-estimation of the within-region 16-service common effect; singleton services use a neutral parent-component outage because a one-feature model cannot be refit after deletion. Therefore health/age variants may propagate smaller indirect changes to non-target services, while public-source non-target services must remain exactly invariant. The machine-readable affected-source ledger reports both paths separately.

{_md_table(affected_attention, ['source_kind', 'source_column', 'affected_service_count', 'affected_rank_spearman_median', 'affected_symmetric_top20_in_top30_median', 'affected_delta_to_component_neutral_ratio', 'affected_rank_displacement_max'], 22)}
"""
    write_text(specialty_text, root / "reports/model_v1/04_specialty_gap.md")

    temporal_ablation = pd.DataFrame(temporal["temporal_ablation"])
    bundle_reliability = pd.DataFrame(temporal["nhis_bundle_reliability"])
    advisory_sensitivity = pd.DataFrame(
        temporal["operational_advisory_sensitivity"]
    )
    strength_sensitivity = pd.DataFrame(
        temporal["seasonality_strength_sensitivity"]
    )
    temporal_text = f"""# MEDIROAD MODEL V1 - Stage 2B Temporal Layer

## Outcome

The preregistered monthly gate failed and the released planning resolution is therefore **season**. All exact month/date fields are null. The 12-month panel remains a diagnostic substrate only.

- NHIS monthly median LOYO rho: `{temporal['monthly_resolution']['median_loyo_spearman']:.4f}`
- NHIS monthly Top-3 Jaccard: `{temporal['monthly_resolution']['median_loyo_top3_jaccard']:.4f}`
- NHIS season-scale evidence: `{json.dumps(temporal['nhis_seasonal_stability'], ensure_ascii=False, default=_json_default)}`
- Climate season-scale evidence: `{json.dumps(temporal['climate_seasonal_stability'], ensure_ascii=False, default=_json_default)}`
- Distinct fallback seasons: `{temporal['fallback_distinct_fraction']:.3f}`
- Exact month/date non-null count: `{temporal['exact_month_date_non_null_count']}`
- Rows released as a primary/fallback pair because the two years disagree on Top-1: `{temporal['year_top1_disagreement_release_count']}`

## Bundle-aligned release evidence

Validation, reliability shrinkage, and season selection all use the same five service-bundle profiles. A low margin remains visible rather than being inflated into confidence.

{_md_table(bundle_reliability, ['bundle_id', 'bundle_reliability_tier', 'bundle_reliability_lambda', 'season_rank_spearman', 'top1_season_agreement', 'released_primary_season', 'released_fallback_season', 'released_primary_fallback_margin', 'released_primary_in_both_year_top2'], 10)}

## Temporal ablation

{_md_table(temporal_ablation, ['ablation_type', 'variant', 'spearman', 'top3_recall', 'primary_month_change_fraction'])}

Removing the sole supported NHIS seasonal signal is a deliberately destructive stress test, not a stability requirement. Exhaustive coverage and the explicit neutral fallback are the completion checks.

## Seasonal-signal strength sensitivity

{_md_table(strength_sensitivity, ['seasonality_strength', 'destructive_neutral_signal', 'median_month_rank_spearman', 'mean_month_top3_recall', 'median_season_rank_spearman', 'mean_season_top2_recall', 'season_top1_retention', 'interpretation'])}

Every positive amplitude must preserve the full monthly and seasonal ordering exactly. Strength zero is retained as the explicit destructive no-signal stress.

## Rejected operational overlays

{_md_table(advisory_sensitivity, ['experiment', 'overlay_weight', 'median_season_rank_spearman', 'mean_season_top2_recall', 'season_top1_retention', 'released_as_primary', 'reason'])}

NHIS expresses coarse observed insured utilization seasonality, not local demand. Because the climate seasonal LOYO gate failed, NASA POWER is retained only as a historical long-run operational-risk annotation, not as the macro season selector or a live replan trigger. A future replan trigger requires a date-specific forecast and an operator-confirmed service date. Static GTFS is spatial context only. Team, vehicle, equipment and venue calendars remain explicit unknowns.
"""
    write_text(temporal_text, root / "reports/model_v1/05_temporal_layer.md")

    hard = gate.loc[gate["severity"].eq("hard")]
    failures = hard.loc[~hard["passed"]]
    current_text = f"""# MEDIROAD Stage 2 Current Quality Gate

- Hard gates passed: `{int(hard['passed'].sum())}/{len(hard)}`
- Stage 2 completion: `{'PASS' if failures.empty else 'FAIL'}`
- Runtime seconds: `{sum(timings.values()):.2f}`

## Failed hard gates

{_md_table(failures, ['stage', 'gate', 'value', 'rule', 'threshold', 'note'], 50)}

## All gates

{_md_table(gate, ['stage', 'gate', 'value', 'rule', 'threshold', 'passed', 'severity'], 100)}

Stage 3 and later optimization have not been executed.
"""
    write_text(current_text, report_dir / "CURRENT_STAGE2_GATE.md")


def run_stage2(
    package_root: str | Path,
    *,
    n_boot: int = 1000,
    n_jobs: int = 8,
    fail_on_gate: bool = True,
) -> dict[str, Any]:
    root = Path(package_root).resolve()
    started_at_utc = datetime.now(timezone.utc)
    started = time.perf_counter()
    timings: dict[str, float] = {}
    specialty_config_path = root / "configs/model_v1/specialty_gap.yaml"
    bundle_config_path = root / "configs/model_v1/service_bundles.yaml"
    stage1_config_path = root / "configs/model_v1/stage1_features.yaml"
    specialty_config = load_specialty_config(specialty_config_path)
    bundle_config = load_service_bundle_config(bundle_config_path)
    source = specialty_config["source"]
    road_path = root / str(source["road_master"])
    hashes = {
        "frozen_canonical": sha256_file(root / FROZEN_CANONICAL),
        "road_master": sha256_file(road_path),
        "stage1_config": sha256_file(stage1_config_path),
        "specialty_config": sha256_file(specialty_config_path),
        "bundle_config": sha256_file(bundle_config_path),
        "temporal_config": sha256_file(root / "configs/model_v1/temporal_model.yaml"),
        "stage2_pipeline_code": sha256_file(root / "src/mediroad/stage2_pipeline.py"),
        "specialty_gap_code": sha256_file(root / "src/mediroad/specialty/gap.py"),
        "specialty_ablation_code": sha256_file(
            root / "src/mediroad/specialty/ablation.py"
        ),
        "temporal_seasonality_code": sha256_file(
            root / "src/mediroad/temporal/seasonality.py"
        ),
        "temporal_fit_code": sha256_file(
            root / "src/mediroad/temporal/temporal_fit.py"
        ),
        "stage2_correlation_code": sha256_file(
            root / "src/mediroad/reporting/stage2_correlation.py"
        ),
    }
    for actual_key, expected_key in (
        ("frozen_canonical", "frozen_canonical_sha256"),
        ("road_master", "road_master_sha256"),
        ("stage1_config", "stage1_config_sha256"),
    ):
        if hashes[actual_key] != str(source[expected_key]):
            raise RuntimeError(f"Frozen input contract failed: {actual_key}")

    load_started = time.perf_counter()
    master = pd.read_csv(
        road_path, dtype={"admin_dong_code": str}, low_memory=False
    )
    master["admin_dong_code"] = master["admin_dong_code"].str.zfill(8)
    specialty_pivot = pd.read_csv(root / str(source["hira_specialty_pivot"]), low_memory=False)
    facility_admin = pd.read_csv(
        root / str(source["hira_facility_admin_join"]),
        dtype={"admin_dong_code": str},
        low_memory=False,
    )
    timings["input_load"] = time.perf_counter() - load_started

    specialty_started = time.perf_counter()
    scores = build_specialty_gap_from_root(root)
    bundle_scores = build_bundle_scores(scores, bundle_config)
    specialty_validation = validate_specialty_results(
        scores,
        specialty_config,
        bundle_scores=bundle_scores,
        bundle_config=bundle_config,
        canonical_path=root / FROZEN_CANONICAL,
    )
    specialty_validation["focused_physician_rows"] = int(
        scores["service_type"].eq("focused_physician_specialty").sum()
    )
    specialty_validation["adjunct_dental_rows"] = int(
        scores["service_id"].eq("dental").sum()
    )
    specialty_validation["component_missing_count"] = int(
        scores[list(COMPONENT_COLUMNS)].isna().sum().sum()
    )
    if not specialty_validation["valid"]:
        raise RuntimeError(
            "Stage 2A result validation failed: "
            + ", ".join(
                key for key, value in specialty_validation["checks"].items() if not value
            )
        )
    specialty_dir = root / SPECIALTY_OUTPUT
    specialty_dir.mkdir(parents=True, exist_ok=True)
    write_parquet(scores, specialty_dir / "specialty_gap_scores.parquet")
    write_csv(scores, specialty_dir / "specialty_gap_scores.csv")
    write_parquet(bundle_scores, specialty_dir / "recommended_bundle_gap_scores.parquet")
    write_csv(bundle_scores, specialty_dir / "recommended_bundle_gap_scores.csv")
    write_csv(_column_manifest(scores, "stage2a"), specialty_dir / "stage2a_output_column_manifest.csv")
    write_csv(_column_manifest(bundle_scores, "stage2a_bundle"), specialty_dir / "bundle_output_column_manifest.csv")
    service_registry = pd.DataFrame(specialty_config["services"])
    for column in service_registry.columns:
        service_registry[column] = service_registry[column].map(
            lambda value: json.dumps(value, ensure_ascii=False, sort_keys=True)
            if isinstance(value, (list, dict))
            else value
        )
    write_csv(service_registry, specialty_dir / "service_definition_registry.csv")
    write_json(specialty_validation, specialty_dir / "specialty_validation.json")
    timings["specialty_primary"] = time.perf_counter() - specialty_started

    correlation_started = time.perf_counter()
    correlation = _run_correlation_suite(
        root,
        scores,
        bundle_scores,
        master,
        specialty_config,
        bundle_config,
        n_boot=n_boot,
        n_jobs=n_jobs,
    )
    healthmap_detail, healthmap_summary = _healthmap_external_validation(
        root, scores, master, specialty_config
    )
    write_csv(healthmap_detail, specialty_dir / "healthmap_external_validation.csv")
    write_json(healthmap_summary, specialty_dir / "healthmap_external_validation_summary.json")
    timings["correlation_and_validation"] = time.perf_counter() - correlation_started

    ablation_started = time.perf_counter()
    ablation = run_specialty_ablation(
        master,
        specialty_pivot,
        facility_admin,
        specialty_config,
        scores,
        bundle_config=bundle_config,
        fail_closed=True,
    )
    ablation.ledger["execution_semantics"] = np.where(
        ablation.ledger["scope"].eq("source_feature"),
        (
            "hybrid_refit_stress: multi-feature services refit after LOO; "
            "singleton services use neutral parent-component outage"
        ),
        ablation.ledger["scope"].astype(str)
        + "::"
        + ablation.ledger["mode"].astype(str),
    )
    ablation_dir = specialty_dir / "03_ablation"
    write_csv(ablation.ledger, ablation_dir / "ablation_run_ledger.csv")
    write_csv(ablation.run_summary, ablation_dir / "ablation_run_summary.csv")
    write_csv(ablation.service_metrics, ablation_dir / "ablation_service_metrics.csv")
    write_parquet(
        ablation.region_service_metrics,
        ablation_dir / "ablation_region_service_metrics.parquet",
    )
    write_parquet(ablation.region_runs, ablation_dir / "ablation_region_runs.parquet")
    ablation.affected_source_feature_summary = _affected_source_feature_summary(
        ablation, specialty_config
    )
    write_csv(
        ablation.affected_source_feature_summary,
        ablation_dir / "affected_service_feature_ablation_summary.csv",
    )
    timings["specialty_ablation"] = time.perf_counter() - ablation_started

    temporal_started = time.perf_counter()
    temporal = _run_temporal_layer(
        root, master, scores, specialty_config, bundle_config
    )
    write_csv(
        _column_manifest(
            pd.read_parquet(root / TEMPORAL_OUTPUT / "region_bundle_season_scores.parquet"),
            "stage2b",
        ),
        root / TEMPORAL_OUTPUT / "stage2b_output_column_manifest.csv",
    )
    timings["temporal"] = time.perf_counter() - temporal_started

    gate = _gate_frame(
        specialty_config=specialty_config,
        temporal_config=temporal["config"],
        specialty_validation=specialty_validation,
        correlation=correlation,
        healthmap=healthmap_summary,
        ablation=ablation,
        temporal=temporal,
        hashes=hashes,
    )
    hard = gate.loc[gate["severity"].eq("hard")]
    hard_failures = hard.loc[~hard["passed"]]
    stage_complete = bool(hard_failures.empty)
    write_csv(gate, root / STAGE2_OUTPUT / "STAGE2_QUALITY_GATE.csv")
    gate_payload = {
        "stage_complete": stage_complete,
        "hard_gate_pass_count": int(hard["passed"].sum()),
        "hard_gate_count": int(len(hard)),
        "hard_fail_count": int(len(hard_failures)),
        "advisory_or_diagnostic_fail_count": int(
            (~gate.loc[~gate["severity"].eq("hard"), "passed"]).sum()
        ),
        "failed_hard_gates": hard_failures.to_dict(orient="records"),
        "stage3_executed": False,
        "interpretation": "policy_mcda_not_patient_demand_or_causal_prediction",
    }
    write_json(gate_payload, root / STAGE2_REPORT / "STAGE2_QUALITY_GATE.json")
    _write_stage2_reports(
        root,
        gate=gate,
        correlation=correlation,
        healthmap_detail=healthmap_detail,
        healthmap_summary=healthmap_summary,
        ablation=ablation,
        temporal=temporal,
        timings=timings,
    )
    hashes_after = {
        "frozen_canonical": sha256_file(root / FROZEN_CANONICAL),
        "road_master": sha256_file(road_path),
        "stage1_config": sha256_file(stage1_config_path),
    }
    if any(hashes_after[key] != hashes[key] for key in hashes_after):
        raise RuntimeError("A frozen Stage 1 or road input changed during Stage 2 execution")
    artifact_inventory = _stage2_artifact_inventory(root)
    artifact_inventory_path = root / STAGE2_OUTPUT / "STAGE2_ARTIFACT_INVENTORY.csv"
    write_csv(artifact_inventory, artifact_inventory_path)
    timings["total"] = time.perf_counter() - started
    ended_at_utc = datetime.now(timezone.utc)
    run_id = (
        "stage2_"
        + started_at_utc.strftime("%Y%m%dT%H%M%SZ")
        + "_"
        + hashes["stage2_pipeline_code"][:12]
    )
    metadata = {
        "run_id": run_id,
        "started_at_utc": started_at_utc.isoformat(),
        "ended_at_utc": ended_at_utc.isoformat(),
        "run_status": "PASS" if stage_complete else "QUALITY_GATE_FAIL",
        "stage_complete": stage_complete,
        "stage_scope": "Stage2A_and_Stage2B_only",
        "stage3_executed": False,
        "package_root": str(root),
        "input_sha256": hashes,
        "input_sha256_after": hashes_after,
        "bootstrap_replicates": int(n_boot),
        "worker_count": int(n_jobs),
        "timings_seconds": timings,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "specialty_rows": int(len(scores)),
        "bundle_rows": int(len(bundle_scores)),
        "ablation_runs": int(len(ablation.ledger)),
        "temporal_fit_rows": temporal["temporal_fit_rows"],
        "recommendation_rows": temporal["recommendation_rows"],
        "artifact_count": int(len(artifact_inventory)),
        "artifact_inventory_relative_path": artifact_inventory_path.relative_to(root).as_posix(),
        "artifact_inventory_sha256": sha256_file(artifact_inventory_path),
        "quality_gate": gate_payload,
    }
    write_json(metadata, root / STAGE2_OUTPUT / "STAGE2_RUN_METADATA.json")
    if fail_on_gate and not stage_complete:
        failed_names = ", ".join(hard_failures["gate"].astype(str).tolist())
        raise RuntimeError(f"Stage 2 quality gate failed: {failed_names}")
    return metadata


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, default=Path.cwd())
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument(
        "--allow-gate-fail",
        action="store_true",
        help="Materialize diagnostics and return normally even when a hard quality gate fails.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    result = run_stage2(
        args.package_root,
        n_boot=args.bootstrap,
        n_jobs=args.jobs,
        fail_on_gate=not args.allow_gate_fail,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=_json_default))
    return 0 if result["stage_complete"] or args.allow_gate_fail else 2


if __name__ == "__main__":
    raise SystemExit(main())
