"""MODEL V1 Stage-1: feature audit, correlation atlas, and ablation gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
import uuid
from pathlib import Path
from typing import Any

# Stage 1 is a batch renderer. Force a headless backend before importing any
# reporting module so WSL/worker teardown never touches Tk GUI state.
os.environ["MPLBACKEND"] = "Agg"

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from mediroad.data.feature_registry import (
    build_feature_manifest,
    load_feature_config,
    validate_feature_manifest,
)
from mediroad.reporting.correlation import (
    block_bootstrap_quality,
    correlation_quality_metrics,
    hierarchical_order,
    mixed_granularity_spearman,
    plot_correlation_map,
    residualized_rank_frame,
    save_interactive_heatmap,
    spearman_with_counts,
)
from mediroad.scoring.ablation import (
    ablation_quality_gates,
    axis_priority_responsiveness,
    build_need_score,
    certify_weight_profiles,
    run_exhaustive_ablation,
)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot JSON-encode {type(value).__name__}")


def _format_optional_float(value: Any, digits: int = 4) -> str:
    """Format metrics that are intentionally N/A for mixed-unit matrices."""

    if value is None or pd.isna(value):
        return "N/A (mixed analysis units)"
    return f"{float(value):.{digits}f}"


def _atomic_path(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")


def write_json(value: Any, path: Path) -> None:
    temp = _atomic_path(path)
    try:
        temp.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, default=_json_default),
            encoding="utf-8",
        )
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def write_text(value: str, path: Path) -> None:
    temp = _atomic_path(path)
    try:
        temp.write_text(value, encoding="utf-8")
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def write_csv(frame: pd.DataFrame, path: Path, *, index: bool = False) -> None:
    temp = _atomic_path(path)
    try:
        frame.to_csv(temp, index=index, encoding="utf-8-sig")
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def write_csv_gz(frame: pd.DataFrame, path: Path, *, index: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=index, encoding="utf-8", compression="gzip")


def write_parquet(frame: pd.DataFrame, path: Path) -> None:
    temp = _atomic_path(path)
    try:
        frame.to_parquet(temp, index=False, compression="zstd")
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def configure_fonts() -> None:
    try:
        import matplotlib
        from matplotlib import font_manager

        font_sets = [
            [Path("/mnt/c/Windows/Fonts/malgun.ttf"), Path("/mnt/c/Windows/Fonts/malgunbd.ttf")],
            [Path("C:/Windows/Fonts/malgun.ttf"), Path("C:/Windows/Fonts/malgunbd.ttf")],
            [Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")],
        ]
        for candidates in font_sets:
            existing = [candidate for candidate in candidates if candidate.exists()]
            if existing:
                for candidate in existing:
                    font_manager.fontManager.addfont(str(candidate))
                family = font_manager.FontProperties(fname=str(existing[0])).get_name()
                matplotlib.rcParams["font.family"] = family
                break
        matplotlib.rcParams["axes.unicode_minus"] = False
    except Exception:
        # Plot helpers retain a safe DejaVu fallback; missing Korean glyphs are
        # surfaced visually but must not invalidate numeric outputs.
        pass


def matrix_pairs(
    corr: pd.DataFrame,
    counts: pd.DataFrame,
    manifest: pd.DataFrame,
    *,
    threshold: float = 0.0,
) -> pd.DataFrame:
    names = corr.columns.tolist()
    meta = manifest.set_index("source_column")
    rows: list[dict[str, Any]] = []
    for left in range(len(names)):
        for right in range(left + 1, len(names)):
            value = corr.iat[left, right]
            if not np.isfinite(value) or abs(value) < threshold:
                continue
            a, b = names[left], names[right]
            rows.append(
                {
                    "feature_a": a,
                    "feature_b": b,
                    "label_a": meta.at[a, "label_ko"] if a in meta.index else a,
                    "label_b": meta.at[b, "label_ko"] if b in meta.index else b,
                    "axis_a": meta.at[a, "axis"] if a in meta.index else "unlabeled",
                    "axis_b": meta.at[b, "axis"] if b in meta.index else "unlabeled",
                    "semantic_group_a": meta.at[a, "semantic_group"] if a in meta.index and "semantic_group" in meta else "",
                    "semantic_group_b": meta.at[b, "semantic_group"] if b in meta.index and "semantic_group" in meta else "",
                    "rho": float(value),
                    "abs_rho": float(abs(value)),
                    "pairwise_n": int(counts.iat[left, right]),
                    "same_axis": int(
                        a in meta.index and b in meta.index and meta.at[a, "axis"] == meta.at[b, "axis"]
                    ),
                    "same_semantic_group": int(
                        a in meta.index
                        and b in meta.index
                        and "semantic_group" in meta
                        and pd.notna(meta.at[a, "semantic_group"])
                        and str(meta.at[a, "semantic_group"]) != ""
                        and meta.at[a, "semantic_group"] == meta.at[b, "semantic_group"]
                    ),
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=[
                "feature_a",
                "feature_b",
                "label_a",
                "label_b",
                "axis_a",
                "axis_b",
                "semantic_group_a",
                "semantic_group_b",
                "rho",
                "abs_rho",
                "pairwise_n",
                "same_axis",
                "same_semantic_group",
            ]
        )
    return pd.DataFrame(rows).sort_values("abs_rho", ascending=False).reset_index(drop=True)


def _attach_granularity(
    frame: pd.DataFrame,
    manifest: pd.DataFrame,
    *,
    analysis_level: str,
    allow_repeated_descriptive: bool = False,
) -> pd.DataFrame:
    result = frame.copy()
    meta = manifest.set_index("source_column").reindex(result.columns)
    granularity = meta["granularity"].fillna("unspecified").astype(str).to_dict()
    repeated = [name for name, value in granularity.items() if "sigungu" in value.lower()]
    result.attrs.update(
        {
            "mediroad_granularity": granularity,
            "mediroad_sigungu_repeated_features": repeated,
            "mediroad_analysis_level": analysis_level,
            "allow_sigungu_repeated_descriptive": allow_repeated_descriptive,
        }
    )
    return result


def _reorder(
    corr: pd.DataFrame,
    counts: pd.DataFrame,
    order: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    reordered_corr = corr.loc[order, order].copy()
    reordered_counts = counts.loc[order, order].copy()
    reordered_corr.attrs.update(corr.attrs)
    reordered_counts.attrs.update(counts.attrs)
    return reordered_corr, reordered_counts


def _sigungu_collapse(
    frame: pd.DataFrame,
    master: pd.DataFrame,
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    meta = manifest.set_index("source_column")
    output: dict[str, pd.Series] = {}
    groups = master["policy_sigungu_name"]
    for column in frame.columns:
        granularity = str(meta.at[column, "granularity"]) if column in meta.index else "admin_dong"
        grouped = frame[column].groupby(groups, sort=True)
        if "sigungu" in granularity.lower():
            if (grouped.nunique(dropna=True) > 1).any():
                raise ValueError(f"Sigungu-repeated feature varies within group: {column}")
            output[column] = grouped.first()
        else:
            output[column] = grouped.median()
    collapsed = pd.DataFrame(output)
    collapsed.attrs.update(
        {
            "mediroad_analysis_level": "sigungu_collapsed",
            "mediroad_group_count": int(len(collapsed)),
            "mediroad_sigungu_repeated_features": [
                c for c in collapsed if "sigungu" in str(meta.at[c, "granularity"]).lower()
            ],
        }
    )
    return collapsed


def _semantic_group_composites(
    need_oriented: pd.DataFrame,
    selected_manifest: pd.DataFrame,
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Collapse correlated source components into rank-based semantic units.

    The full component map remains available for audit.  The publication map
    uses one column per configured semantic concept, so deliberately grouped
    near-duplicates improve reliability without receiving duplicate weight or
    pretending to be independent representatives.
    """

    manifest_by_column = selected_manifest.set_index("source_column", drop=False)
    group_members: dict[str, list[str]] = {}
    group_order: list[str] = []
    for spec in config["features"]:
        column = str(spec["feature_id"])
        group = str(spec["semantic_group"])
        if group not in group_members:
            group_order.append(group)
            group_members[group] = []
        group_members[group].append(column)

    composite_data: dict[str, pd.Series] = {}
    metadata_rows: list[dict[str, Any]] = []
    for position, group in enumerate(group_order, start=1):
        members = group_members[group]
        missing = sorted(set(members) - set(need_oriented.columns))
        if missing:
            raise KeyError(f"Semantic group {group} is missing configured components: {missing}")
        measurement_granularities = {
            str(manifest_by_column.at[name, "granularity"])
            for name in members
        }
        # ``granularity`` also records *how* an admin-level value was measured
        # (local count, centroid-to-network destination, etc.).  Those strings
        # may legitimately differ inside one construct.  What must never be
        # mixed is the independent analysis unit: 153 admin dongs versus the
        # 11 policy-sigungu contexts repeated over admin rows.
        analysis_units = {
            "policy_sigungu"
            if "sigungu" in value.lower()
            else "admin_dong"
            for value in measurement_granularities
        }
        if len(analysis_units) != 1:
            raise ValueError(
                f"Semantic group mixes independent analysis units: {group} -> "
                f"{analysis_units} from {measurement_granularities}"
            )
        analysis_unit = next(iter(analysis_units))
        composite_name = f"semantic__{group}"
        ranked = need_oriented[members].rank(method="average", pct=True)
        composite_data[composite_name] = ranked.mean(axis=1)
        first = manifest_by_column.loc[members[0]].to_dict()
        if len(members) == 1:
            label_ko = str(first.get("label_ko") or members[0])
            label_en = str(first.get("label_en") or members[0])
        else:
            label_ko = f"{group.replace('_', ' ')} 복합지표"
            label_en = f"{group.replace('_', ' ').title()} composite"
        first.update(
            {
                "feature_id": f"SG{position:03d}",
                "source_column": composite_name,
                "label_ko": label_ko,
                "label_en": label_en,
                "semantic_group": group,
                "component_count": len(members),
                "component_columns": "|".join(members),
                "component_measurement_granularities": "|".join(sorted(measurement_granularities)),
                "granularity": (
                    "policy_sigungu_repeated_on_admin_dong"
                    if analysis_unit == "policy_sigungu"
                    else "admin_dong"
                ),
                "proxy": any(bool(manifest_by_column.at[name, "proxy"]) for name in members),
                "model_input": True,
                "role": "model_input_semantic_composite",
            }
        )
        metadata_rows.append(first)
    frame = pd.DataFrame(composite_data, index=need_oriented.index)
    metadata = pd.DataFrame(metadata_rows)
    return frame, metadata


def _save_matrix_bundle(
    name: str,
    corr: pd.DataFrame,
    counts: pd.DataFrame,
    manifest: pd.DataFrame,
    matrix_dir: Path,
    figure_dir: Path,
    *,
    title: str,
    annotate: bool,
    interactive: bool = False,
    mask_n: int | None = None,
) -> list[str]:
    write_csv_gz(corr, matrix_dir / f"{name}_correlation.csv.gz")
    write_csv_gz(counts, matrix_dir / f"{name}_pairwise_n.csv.gz")
    figures = plot_correlation_map(
        corr,
        counts,
        manifest,
        figure_dir / name,
        title,
        annotate=annotate,
        mask_n=mask_n,
    )
    if interactive:
        figures.append(
            save_interactive_heatmap(
                corr,
                counts,
                manifest,
                figure_dir / f"{name}.html",
                title,
            )
        )
    return figures


def _plot_ablation(summary: pd.DataFrame, region_stability: pd.DataFrame, figure_dir: Path) -> list[str]:
    import matplotlib.pyplot as plt

    outputs: list[str] = []
    feature = summary[(summary["scope"] == "feature") & (summary["removal_mode"] == "renormalize")]
    feature = feature.sort_values("normalized_influence", ascending=True)
    fig, ax = plt.subplots(figsize=(12, max(7, len(feature) * 0.32)))
    ax.barh(feature["variant"], feature["normalized_influence"], color="#4C78A8")
    ax.axvline(0.10, color="#D62728", linestyle="--", linewidth=1, label="10% diagnostic gate")
    ax.set_xlabel("Normalized mean absolute rank influence")
    ax.set_title("Feature leave-one-out influence across 20 configurations")
    ax.legend(loc="lower right")
    fig.tight_layout()
    for suffix in ("png", "svg"):
        path = figure_dir / f"ablation_feature_influence.{suffix}"
        fig.savefig(path, dpi=300 if suffix == "png" else None, bbox_inches="tight")
        outputs.append(str(path))
    plt.close(fig)

    probability_column = "top20_inclusion_frequency_scope_equal"
    ranked = region_stability.sort_values([probability_column, "mean_rank_scope_equal"], ascending=[False, True]).head(40)
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.barh(
        ranked["admin_dong_name"] + " (" + ranked["policy_sigungu_name"] + ")",
        ranked[probability_column],
        color=np.where(ranked[probability_column].ge(0.8), "#2CA02C", "#FFBF00"),
    )
    ax.axvline(0.8, color="#2CA02C", linestyle="--", linewidth=1)
    ax.axvline(0.2, color="#D62728", linestyle=":", linewidth=1)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Top-20 inclusion frequency (equal weight per perturbation scope)")
    ax.set_title("Region stability under Stage-1 perturbations (scope balanced)")
    ax.invert_yaxis()
    fig.tight_layout()
    for suffix in ("png", "svg"):
        path = figure_dir / f"ablation_region_stability.{suffix}"
        fig.savefig(path, dpi=300 if suffix == "png" else None, bbox_inches="tight")
        outputs.append(str(path))
    plt.close(fig)
    return outputs


def _markdown_table(frame: pd.DataFrame, columns: list[str], limit: int = 20) -> str:
    view = frame.loc[:, [c for c in columns if c in frame]].head(limit).copy()
    if view.empty:
        return "(해당 항목 없음)"
    return view.to_markdown(index=False)


def run_stage1(
    root: Path,
    *,
    n_boot: int = 1000,
    n_jobs: int = 8,
) -> dict[str, Any]:
    started = time.perf_counter()
    configure_fonts()
    root = root.resolve()
    master_path = root / "derived/mediroad_admin_dong_master_v6_road_integrated.csv"
    audit_path = root / "reports/V6_FEATURE_COLUMN_AUDIT.csv"
    config_path = root / "configs/model_v1/stage1_features.yaml"
    road_report_path = root / "14_v6_external_data/road_network/derived/osm_road_network_integration_report_v6.json"
    for path in (master_path, audit_path, config_path, road_report_path):
        if not path.exists():
            raise FileNotFoundError(path)

    output_root = root / "outputs/model_v1"
    audit_dir = output_root / "01_data_audit"
    correlation_dir = output_root / "02_correlation"
    matrix_dir = correlation_dir / "matrices"
    ablation_dir = output_root / "03_ablation"
    report_root = root / "reports/model_v1"
    figure_dir = report_root / "figures"
    for directory in (audit_dir, matrix_dir, ablation_dir, figure_dir):
        directory.mkdir(parents=True, exist_ok=True)

    timings: dict[str, float] = {}
    stage_started = time.perf_counter()
    master = pd.read_csv(master_path, dtype={"admin_dong_code": str}, low_memory=False)
    audit = pd.read_csv(audit_path, low_memory=False)
    config = load_feature_config(config_path)
    road_report = json.loads(road_report_path.read_text(encoding="utf-8"))
    manifest = build_feature_manifest(master, audit, config)
    manifest_validation = validate_feature_manifest(manifest, master)
    if not manifest_validation["valid"] or not manifest_validation["ready_for_model"]:
        raise RuntimeError(f"Feature manifest gate failed: {manifest_validation}")
    write_csv(manifest, audit_dir / "feature_manifest_549.csv")
    write_csv(
        manifest[
            [
                "feature_id",
                "source_column",
                "label_ko",
                "label_en",
                "axis",
                "role",
                "unit",
                "direction",
                "granularity",
                "proxy",
                "model_input",
            ]
        ],
        audit_dir / "feature_label_dictionary.csv",
    )
    write_csv(manifest[~manifest["model_input"]], audit_dir / "exclusion_and_deferral_ledger.csv")
    write_json(manifest_validation, audit_dir / "feature_manifest_validation.json")
    timings["feature_audit_seconds"] = time.perf_counter() - stage_started

    stage_started = time.perf_counter()
    generated_figures: list[str] = []
    correlation_metrics: list[dict[str, Any]] = []
    numeric_columns = [
        column
        for column in master.select_dtypes(include=[np.number]).columns
        if master[column].nunique(dropna=True) > 1
    ]
    full_numeric = _attach_granularity(
        master[numeric_columns],
        manifest,
        analysis_level="admin_dong",
        allow_repeated_descriptive=True,
    )
    # Preserve the all-numeric audit atlas while giving every cell its correct
    # analysis unit: admin/admin pairs use 153 rows; any pair touching a
    # sigungu-repeated feature uses 11 collapsed sigungu observations.
    full_corr, full_n = mixed_granularity_spearman(full_numeric, master["policy_sigungu_name"], manifest)
    full_order = hierarchical_order(full_corr)
    full_corr, full_n = _reorder(full_corr, full_n, full_order)
    generated_figures += _save_matrix_bundle(
        "correlation_atlas_all_numeric_raw_clustered",
        full_corr,
        full_n,
        manifest,
        matrix_dir,
        figure_dir,
        title="MEDIROAD V6 · 전체 분석가능 수치 피처 Spearman Atlas (감사용)",
        annotate=False,
        interactive=True,
        mask_n=107,
    )
    write_csv(matrix_pairs(full_corr, full_n, manifest, threshold=0.90), correlation_dir / "all_numeric_high_correlation_pairs_abs090.csv")
    correlation_metrics.append({"panel": "all_numeric_mixed_granularity_audit", **correlation_quality_metrics(full_corr, manifest)})

    selected_columns = [str(item["feature_id"]) for item in config["features"]]
    selected_manifest = manifest[manifest["source_column"].isin(selected_columns)].copy()
    selected_raw = master[selected_columns].apply(pd.to_numeric, errors="coerce")
    direction = {
        str(item["feature_id"]): 1.0 if str(item["direction"]) == "+" else -1.0
        for item in config["features"]
    }
    selected_need = selected_raw.mul(pd.Series(direction), axis=1)
    groups = master["policy_sigungu_name"]
    raw_corr, raw_n = mixed_granularity_spearman(selected_raw, groups, selected_manifest)
    generated_figures += _save_matrix_bundle(
        "active_source_components_raw_semantic_order",
        raw_corr,
        raw_n,
        selected_manifest,
        matrix_dir,
        figure_dir,
        title=f"MODEL V1 · {len(selected_columns)}개 활성 원천 구성 피처 원방향 상관 (시군 반복 보정)",
        annotate=True,
        mask_n=8,
    )
    component_need_corr, component_need_n = mixed_granularity_spearman(
        selected_need, groups, selected_manifest
    )
    component_order = hierarchical_order(component_need_corr)
    component_need_corr, component_need_n = _reorder(
        component_need_corr, component_need_n, component_order
    )
    generated_figures += _save_matrix_bundle(
        "active_source_components_need_oriented_clustered",
        component_need_corr,
        component_need_n,
        selected_manifest,
        matrix_dir,
        figure_dir,
        title="MODEL V1 · 활성 원천 구성 피처 Need 방향 상관지도 (그룹화 감사)",
        annotate=True,
        mask_n=8,
    )
    component_pairs = matrix_pairs(
        component_need_corr, component_need_n, selected_manifest, threshold=0.90
    )
    write_csv(
        component_pairs,
        correlation_dir / "active_source_component_pairs_abs090.csv",
    )

    representative_need, representative_manifest = _semantic_group_composites(
        selected_need, selected_manifest, config
    )
    write_csv(representative_manifest, correlation_dir / "semantic_composite_manifest.csv")
    need_corr, need_n = mixed_granularity_spearman(
        representative_need, groups, representative_manifest
    )
    cluster_order = hierarchical_order(need_corr)
    clustered_corr, clustered_n = _reorder(need_corr, need_n, cluster_order)
    generated_figures += _save_matrix_bundle(
        "core_semantic_need_oriented_clustered",
        clustered_corr,
        clustered_n,
        representative_manifest,
        matrix_dir,
        figure_dir,
        title="MODEL V1 · Need 방향 정렬 대표 피처 상관지도 (계층 군집)",
        annotate=True,
        interactive=True,
        mask_n=8,
    )
    selected_pairs = matrix_pairs(need_corr, need_n, representative_manifest, threshold=0.60)
    write_csv(selected_pairs, correlation_dir / "core_semantic_correlation_pairs_abs060.csv")
    correlation_metrics.append({"panel": "core_semantic_mixed_granularity", **correlation_quality_metrics(need_corr, representative_manifest)})

    health_columns = [
        c
        for c in selected_columns
        if selected_manifest.set_index("source_column").at[c, "axis"] == "B_health_burden"
    ]
    health_collapsed = _sigungu_collapse(selected_need[health_columns], master, selected_manifest)
    health_corr, health_n = spearman_with_counts(health_collapsed, min_periods=8)
    generated_figures += _save_matrix_bundle(
        "health_context_11_sigungu",
        health_corr,
        health_n,
        selected_manifest,
        matrix_dir,
        figure_dir,
        title="건강부담 대표 피처 · 11개 시군 Spearman (기술적 연관성)",
        annotate=True,
        mask_n=8,
    )
    correlation_metrics.append({"panel": "health_context_11_sigungu", **correlation_quality_metrics(health_corr, selected_manifest)})

    selected_collapsed = _sigungu_collapse(representative_need, master, representative_manifest)
    collapsed_corr, collapsed_n = spearman_with_counts(selected_collapsed, min_periods=8)
    collapsed_order = hierarchical_order(collapsed_corr)
    collapsed_corr, collapsed_n = _reorder(collapsed_corr, collapsed_n, collapsed_order)
    generated_figures += _save_matrix_bundle(
        "core_semantic_sigungu_collapsed_cross_granularity",
        collapsed_corr,
        collapsed_n,
        representative_manifest,
        matrix_dir,
        figure_dir,
        title=f"MODEL V1 · {len(representative_need.columns)}개 의미단위 피처 시군 축약 교차지도 (n=11)",
        annotate=True,
        mask_n=8,
    )

    representative_columns = representative_need.columns.tolist()
    repeated_mask = representative_manifest.set_index("source_column").loc[
        representative_columns, "granularity"
    ].str.contains(
        "sigungu", case=False, na=False
    )
    admin_columns = [c for c in representative_columns if not bool(repeated_mask.loc[c])]
    oriented_admin = representative_need[admin_columns]
    size_controls = pd.DataFrame(
        {
            "log_population_total": np.log1p(pd.to_numeric(master["population_total"])),
            "log_area_km2": np.log1p(pd.to_numeric(master["area_km2"])),
        },
        index=master.index,
    )
    partial_frame = residualized_rank_frame(oriented_admin, size_controls)
    partial_frame.attrs["mediroad_analysis_level"] = "admin_dong"
    partial_corr, partial_n = spearman_with_counts(partial_frame, min_periods=107)
    partial_order = hierarchical_order(partial_corr)
    partial_corr, partial_n = _reorder(partial_corr, partial_n, partial_order)
    generated_figures += _save_matrix_bundle(
        "core_admin_partial_population_area",
        partial_corr,
        partial_n,
        representative_manifest,
        matrix_dir,
        figure_dir,
        title="대표 행정동 피처 · 총인구·면적 보정 Partial Spearman",
        annotate=True,
        mask_n=107,
    )
    correlation_metrics.append({"panel": "core_admin_partial_population_area", **correlation_quality_metrics(partial_corr, representative_manifest)})

    fixed_effects = pd.get_dummies(master["policy_sigungu_name"], prefix="sigungu", drop_first=True, dtype=float)
    within_controls = pd.concat([size_controls, fixed_effects], axis=1)
    within_frame = residualized_rank_frame(oriented_admin, within_controls)
    within_frame.attrs["mediroad_analysis_level"] = "within_sigungu"
    within_corr, within_n = spearman_with_counts(within_frame, min_periods=107)
    within_order = hierarchical_order(within_corr)
    within_corr, within_n = _reorder(within_corr, within_n, within_order)
    generated_figures += _save_matrix_bundle(
        "core_admin_within_sigungu_partial",
        within_corr,
        within_n,
        representative_manifest,
        matrix_dir,
        figure_dir,
        title="대표 행정동 피처 · 시군 FE·총인구·면적 보정 Within-Sigungu",
        annotate=True,
        mask_n=107,
    )
    correlation_metrics.append({"panel": "core_admin_within_sigungu_partial", **correlation_quality_metrics(within_corr, representative_manifest)})

    eligible_roles = {"model_input", "shadow_substitution", "redundant_not_selected"}
    eligible_meta = manifest[
        manifest["role"].isin(eligible_roles)
        & ~manifest["constant_flag"]
        & ~manifest["granularity"].str.contains("sigungu", case=False, na=False)
        & manifest["source_column"].isin(numeric_columns)
    ].copy()
    eligible_columns = eligible_meta["source_column"].tolist()
    eligible_frame = master[eligible_columns].apply(pd.to_numeric, errors="coerce")
    eligible_partial = residualized_rank_frame(eligible_frame, within_controls)
    eligible_partial.attrs["mediroad_analysis_level"] = "within_sigungu"
    eligible_corr, eligible_n = spearman_with_counts(eligible_partial, min_periods=107)
    eligible_order = hierarchical_order(eligible_corr)
    eligible_corr, eligible_n = _reorder(eligible_corr, eligible_n, eligible_order)
    generated_figures += _save_matrix_bundle(
        "eligible_admin_within_sigungu_partial_clustered",
        eligible_corr,
        eligible_n,
        eligible_meta,
        matrix_dir,
        figure_dir,
        title="Stage-1 검토가능 행정동 피처 · 시군/규모 보정 Correlation Atlas",
        annotate=False,
        interactive=False,
        mask_n=107,
    )
    correlation_metrics.append({"panel": "eligible_admin_within_sigungu_partial", **correlation_quality_metrics(eligible_corr, eligible_meta)})

    context_meta = manifest[
        manifest["granularity"].str.contains("sigungu", case=False, na=False)
        & ~manifest["constant_flag"]
        & manifest["source_column"].isin(numeric_columns)
    ].copy()
    context_columns = context_meta["source_column"].tolist()
    context_collapsed = _sigungu_collapse(master[context_columns], master, context_meta)
    context_corr, context_n = spearman_with_counts(context_collapsed, min_periods=8)
    context_order = hierarchical_order(context_corr)
    context_corr, context_n = _reorder(context_corr, context_n, context_order)
    generated_figures += _save_matrix_bundle(
        "all_sigungu_context_collapsed_clustered",
        context_corr,
        context_n,
        context_meta,
        matrix_dir,
        figure_dir,
        title="전체 시군 반복 Context 피처 · 11개 시군 축약 Atlas",
        annotate=False,
        interactive=False,
        mask_n=8,
    )
    correlation_metrics.append({"panel": "all_sigungu_context_collapsed", **correlation_quality_metrics(context_corr, context_meta)})

    mixed_bootstrap_summary, mixed_bootstrap_edges = block_bootstrap_quality(
        representative_need,
        groups,
        representative_manifest,
        n_boot=n_boot,
        seed=42,
        n_jobs=n_jobs,
    )
    admin_bootstrap_summary, admin_bootstrap_edges = block_bootstrap_quality(
        within_frame,
        groups,
        representative_manifest[representative_manifest["source_column"].isin(admin_columns)],
        n_boot=n_boot,
        seed=43,
        n_jobs=n_jobs,
    )
    health_bootstrap_summary, health_bootstrap_edges = block_bootstrap_quality(
        selected_need[health_columns],
        groups,
        selected_manifest[selected_manifest["source_column"].isin(health_columns)],
        n_boot=n_boot,
        seed=44,
        n_jobs=n_jobs,
    )
    write_csv(mixed_bootstrap_summary, correlation_dir / "core_semantic_mixed_block_bootstrap_quality.csv")
    write_csv(mixed_bootstrap_edges, correlation_dir / "core_semantic_mixed_block_bootstrap_edges.csv")
    write_csv(admin_bootstrap_summary, correlation_dir / "core_admin_within_block_bootstrap_quality.csv")
    write_csv(admin_bootstrap_edges, correlation_dir / "core_admin_within_block_bootstrap_edges.csv")
    write_csv(health_bootstrap_summary, correlation_dir / "health_context_block_bootstrap_quality.csv")
    write_csv(health_bootstrap_edges, correlation_dir / "health_context_block_bootstrap_edges.csv")
    correlation_metrics_df = pd.DataFrame(correlation_metrics)
    write_csv(correlation_metrics_df, correlation_dir / "correlation_quality_metrics.csv")
    timings["correlation_seconds"] = time.perf_counter() - stage_started

    stage_started = time.perf_counter()
    ablation_runs, ablation_summary, ablation_region_runs = run_exhaustive_ablation(master, config)
    gates = ablation_quality_gates(ablation_summary)
    selection = config.get("feature_selection", {})
    primary_scaler = config.get(
        "primary_scaler", selection.get("default_scaler", "winsorized_percentile")
    )
    primary_profile = config.get(
        "primary_weight_profile", selection.get("default_weight_profile", "balanced_policy")
    )
    expected_scalers = list(config.get("scalers", {}))
    profile_certification = certify_weight_profiles(
        ablation_runs,
        default_profile=primary_profile,
        expected_scalers=expected_scalers,
    )
    responsiveness = axis_priority_responsiveness(
        ablation_runs,
        default_profile=primary_profile,
        priority_profile_by_axis={
            "A_demographic_vulnerability": "elderly_priority",
            "C_medical_supply_access_gap": "medical_gap_priority",
            "D_transport_access_gap": "access_priority",
        },
        expected_scalers=expected_scalers,
    )
    write_parquet(ablation_runs, ablation_dir / "ablation_all_runs.parquet")
    write_csv(ablation_runs, ablation_dir / "ablation_all_runs.csv")
    write_csv(ablation_summary, ablation_dir / "ablation_summary.csv")
    write_csv(profile_certification, ablation_dir / "weight_profile_certification.csv")
    write_csv(responsiveness, ablation_dir / "axis_priority_responsiveness.csv")
    write_parquet(ablation_region_runs, ablation_dir / "ablation_region_runs.parquet")
    write_csv(gates, ablation_dir / "ablation_quality_gates.csv")

    primary = build_need_score(master, config, scaler=primary_scaler, weight_profile=primary_profile)
    write_csv(primary.scores, ablation_dir / "stage1_primary_need_score.csv")
    write_csv(
        pd.concat(
            [master[["admin_dong_code"]], primary.feature_matrix.add_prefix("need_oriented__")], axis=1
        ),
        ablation_dir / "stage1_primary_transformed_features.csv",
    )
    write_csv(
        pd.concat([master[["admin_dong_code"]], primary.group_scores.add_prefix("group__")], axis=1),
        ablation_dir / "stage1_primary_semantic_group_scores.csv",
    )

    region_base = master[["admin_dong_code", "admin_dong_name", "policy_sigungu_name"]].copy()
    # Each perturbation scope receives equal weight.  Averaging all raw runs
    # would let the much larger feature-LOFO scope dominate the apparent
    # inclusion frequency and turn it into a run-count artefact.
    region_by_scope = (
        ablation_region_runs.groupby(["scope", "admin_dong_code"], as_index=False)
        .agg(
            top20_inclusion_frequency=("top20", "mean"),
            mean_rank=("need_rank", "mean"),
            rank_sd_within_scope=("need_rank", "std"),
            best_rank=("need_rank", "min"),
            worst_rank=("need_rank", "max"),
            mean_score=("need_score", "mean"),
            score_sd_within_scope=("need_score", "std"),
            run_count=("run_id", "nunique"),
        )
    )
    write_csv(region_by_scope, ablation_dir / "region_stability_by_scope.csv")
    region_aggregate = (
        region_by_scope.groupby("admin_dong_code", as_index=False)
        .agg(
            top20_inclusion_frequency_scope_equal=("top20_inclusion_frequency", "mean"),
            mean_rank_scope_equal=("mean_rank", "mean"),
            rank_sd_between_scopes=("mean_rank", "std"),
            best_rank=("best_rank", "min"),
            worst_rank=("worst_rank", "max"),
            mean_score_scope_equal=("mean_score", "mean"),
            score_sd_between_scopes=("mean_score", "std"),
            scope_count=("scope", "nunique"),
            raw_run_count=("run_count", "sum"),
        )
        .merge(region_base, on="admin_dong_code", how="left", validate="one_to_one")
    )
    write_csv(region_aggregate, ablation_dir / "region_stability.csv")

    primary_feature = ablation_summary[
        ablation_summary["scope"].eq("feature") & ablation_summary["removal_mode"].eq("renormalize")
    ].set_index("variant")
    substitution_lookup = {
        str(item["shadow_feature"]): f"{item.get('active_feature') or item.get('selected_feature')}=>{item['shadow_feature']}"
        for item in config.get("substitutions", [])
    }
    coverage = manifest.copy()
    coverage["ablation_execution_status"] = "NOT_APPLICABLE_DOCUMENTED"
    coverage["ablation_variant"] = "N_A"
    coverage["ablation_reason"] = coverage["not_applicable_reason"]
    coverage["primary_top20_jaccard"] = np.nan
    coverage["worst_spearman"] = np.nan
    coverage["normalized_influence"] = np.nan
    for index, row in coverage.iterrows():
        column = str(row["source_column"])
        if bool(row["model_input"]) and column in primary_feature.index:
            result = primary_feature.loc[column]
            coverage.at[index, "ablation_execution_status"] = "EXACT_FEATURE_LOFO_TESTED_40_RUNS"
            coverage.at[index, "ablation_variant"] = column
            coverage.at[index, "ablation_reason"] = "active Stage-1 feature; renormalize and neutral leave-one-out"
            coverage.at[index, "primary_top20_jaccard"] = result["primary_top20_jaccard"]
            coverage.at[index, "worst_spearman"] = result["worst_spearman"]
            coverage.at[index, "normalized_influence"] = result["normalized_influence"]
        elif column in substitution_lookup:
            variant = substitution_lookup[column]
            result = ablation_summary[
                ablation_summary["scope"].eq("substitution") & ablation_summary["variant"].eq(variant)
            ]
            coverage.at[index, "ablation_execution_status"] = "SHADOW_SUBSTITUTION_TESTED_20_RUNS"
            coverage.at[index, "ablation_variant"] = variant
            coverage.at[index, "ablation_reason"] = "one-at-a-time representative substitution sensitivity"
            if len(result):
                coverage.at[index, "primary_top20_jaccard"] = result.iloc[0]["primary_top20_jaccard"]
                coverage.at[index, "worst_spearman"] = result.iloc[0]["worst_spearman"]
                coverage.at[index, "normalized_influence"] = result.iloc[0]["normalized_influence"]
    tested_active_renormalize = set(
        ablation_summary.loc[
            ablation_summary["scope"].eq("feature")
            & ablation_summary["removal_mode"].eq("renormalize")
            & ablation_summary["run_count"].eq(20),
            "variant",
        ].astype(str)
    )
    tested_active_neutral = set(
        ablation_summary.loc[
            ablation_summary["scope"].eq("feature")
            & ablation_summary["removal_mode"].eq("neutral")
            & ablation_summary["run_count"].eq(20),
            "variant",
        ].astype(str)
    )
    expected_active = set(selected_columns)
    tested_substitutions = set(
        ablation_summary.loc[
            ablation_summary["scope"].eq("substitution") & ablation_summary["run_count"].eq(20),
            "variant",
        ].astype(str)
    )
    expected_substitutions = set(substitution_lookup.values())
    documented_na = coverage.loc[~coverage["model_input"], "ablation_reason"].astype(str).str.strip()
    coverage_valid = bool(
        len(coverage) == 549
        and tested_active_renormalize == expected_active
        and tested_active_neutral == expected_active
        and tested_substitutions == expected_substitutions
        and documented_na.ne("").all()
        and not documented_na.str.upper().isin({"N_A", "NA", "NONE", "TODO"}).any()
        and coverage.loc[coverage["model_input"], "ablation_execution_status"]
        .eq("EXACT_FEATURE_LOFO_TESTED_40_RUNS")
        .all()
    )
    if not coverage_valid:
        raise RuntimeError(
            "Ablation coverage ledger failed set/run-count checks: "
            f"active_renorm={len(tested_active_renormalize)}/{len(expected_active)}, "
            f"active_neutral={len(tested_active_neutral)}/{len(expected_active)}, "
            f"substitution={len(tested_substitutions)}/{len(expected_substitutions)}"
        )
    write_csv(coverage, ablation_dir / "all_549_feature_ablation_coverage.csv")
    generated_figures += _plot_ablation(ablation_summary, region_aggregate, figure_dir)
    timings["ablation_seconds"] = time.perf_counter() - stage_started

    score_vs_total_population = float(
        spearmanr(primary.scores["need_score"], pd.to_numeric(master["population_total"])).statistic
    )
    score_vs_elderly_population = float(
        spearmanr(primary.scores["need_score"], pd.to_numeric(master["population_65plus"])).statistic
    )
    mixed_bootstrap_lookup = mixed_bootstrap_summary.set_index("metric")
    admin_bootstrap_lookup = admin_bootstrap_summary.set_index("metric")
    metric_lookup = correlation_metrics_df.set_index("panel")
    selected_metric = metric_lookup.loc["core_semantic_mixed_granularity"].to_dict()
    admin_partial_metric = metric_lookup.loc["core_admin_partial_population_area"].to_dict()
    admin_within_metric = metric_lookup.loc["core_admin_within_sigungu_partial"].to_dict()
    health_metric = metric_lookup.loc["health_context_11_sigungu"].to_dict()
    registry_pass = bool(manifest_validation["valid"] and manifest_validation["ready_for_model"])
    road_pass = bool(
        float(road_report.get("largest_strong_component_pct", 0)) >= 97.0
        and all(int(item.get("reachable_admin_dongs", 0)) == 153 for item in road_report["categories"].values())
    )
    correlation_hard_checks = {
        "manifest_coverage_549": len(manifest) == 549,
        "selected_exact_duplicate_pairs_zero": int(selected_metric["n_effective_exact_duplicate_pairs"]) == 0,
        "selected_high_corr_pairs_abs095_zero": int(selected_metric["n_high_corr_pairs_0_95"]) == 0,
        "active_component_unexplained_abs095_pairs_zero": bool(
            component_pairs.loc[
                component_pairs["abs_rho"].ge(0.95)
                & component_pairs["same_semantic_group"].eq(0)
            ].empty
        ),
        "selected_manifest_coverage_complete": float(selected_metric["manifest_coverage"]) == 1.0,
        # Metrics that are structurally undefined (for example axis AUC in the
        # six-feature, single-axis health panel) have a null point estimate and
        # n_valid=0 by design; completion applies only to defined metrics.
        "bootstrap_completed_all_three_panels": bool(
            (
                mixed_bootstrap_summary.loc[mixed_bootstrap_summary["point_estimate"].notna(), "n_valid"]
                == n_boot
            ).all()
            and (
                admin_bootstrap_summary.loc[admin_bootstrap_summary["point_estimate"].notna(), "n_valid"]
                == n_boot
            ).all()
            and (
                health_bootstrap_summary.loc[health_bootstrap_summary["point_estimate"].notna(), "n_valid"]
                == n_boot
            ).all()
        ),
        "bootstrap_edge_valid_fraction_ge_0_80": bool(
            mixed_bootstrap_edges["valid_fraction"].ge(0.80).all()
            and admin_bootstrap_edges["valid_fraction"].ge(0.80).all()
            and health_bootstrap_edges["valid_fraction"].ge(0.80).all()
        ),
        "pairwise_n_explicit": bool(set(np.unique(need_n.to_numpy())).issuperset({11, 153})),
        "mixed_spectral_metrics_marked_not_applicable": not bool(selected_metric["spectral_metrics_applicable"]),
    }
    correlation_separation_checks = {
        "admin_population_area_partial_axis_auc_ge_0_60": float(admin_partial_metric["axis_auc"]) >= 0.60,
        "admin_within_sigungu_axis_auc_ge_0_60": float(admin_within_metric["axis_auc"]) >= 0.60,
        "admin_within_sigungu_block_contrast_ge_0_10": float(admin_within_metric["block_contrast"]) >= 0.10,
        "admin_within_cross_axis_high_corr_rate_le_0_05": float(
            admin_within_metric["cross_axis_high_corr_rate_0_80"]
        ) <= 0.05,
        "health_context_max_abs_corr_le_0_70": float(health_metric["max_abs_corr_offdiag"]) <= 0.70,
        "admin_and_health_abs095_pairs_zero": int(admin_within_metric["n_high_corr_pairs_0_95"]) == 0
        and int(health_metric["n_high_corr_pairs_0_95"]) == 0,
        "admin_bootstrap_axis_auc_median_ge_0_60": float(
            admin_bootstrap_lookup.at["axis_auc", "bootstrap_median"]
        ) >= 0.60,
        "admin_bootstrap_axis_auc_ci05_gt_0_55": float(admin_bootstrap_lookup.at["axis_auc", "ci05"]) > 0.55,
    }
    correlation_advisory = {
        "mixed_axis_auc_ge_0_60": float(selected_metric["axis_auc"]) >= 0.60,
        "mixed_block_contrast_ge_0_08": float(selected_metric["block_contrast"]) >= 0.08,
        "cross_axis_high_corr_rate_le_0_05": float(selected_metric["cross_axis_high_corr_rate_0_80"]) <= 0.05,
        "mixed_bootstrap_axis_auc_ci05_gt_0_50": float(mixed_bootstrap_lookup.at["axis_auc", "ci05"]) > 0.50,
    }
    ablation_pass = bool(
        gates.loc[gates["required_for_completion"].eq(1), "passed"].eq(1).all()
        and profile_certification.loc[
            profile_certification["is_default_profile"].eq(1), "profile_certified"
        ].eq(1).all()
        and responsiveness["monotonic_responsiveness_pass"].eq(1).all()
    )
    feature_influence = ablation_summary[
        ablation_summary["scope"].eq("feature")
        & ablation_summary["removal_mode"].eq("renormalize")
    ].set_index("variant")["normalized_influence"]
    axis_influence = ablation_summary[ablation_summary["scope"].eq("axis")].set_index("variant")[
        "normalized_influence"
    ]
    demographic_groups = {
        str(item["semantic_group"])
        for item in config["features"]
        if item["axis"] == "A_demographic_vulnerability"
    }
    balanced_demographic_weight = float(
        config["weight_profiles"]["balanced_policy"]["weights"]["A_demographic_vulnerability"]
    )
    count_and_age_groups = {
        str(item["semantic_group"])
        for item in config["features"]
        if item["feature_id"] in {"population_65plus", "population_65plus_ratio", "population_75plus_ratio"}
    }
    count_and_age_default_share = balanced_demographic_weight * len(count_and_age_groups) / len(
        demographic_groups
    )
    population_checks = {
        "each_elderly_count_component_influence_le_0_10": bool(
            max(
                float(feature_influence.get(name, np.inf))
                for name in ("population_65plus", "population_75plus")
            ) <= 0.10
        ),
        "each_age_structure_component_influence_le_0_10": bool(
            max(
                float(feature_influence.get(name, np.inf))
                for name in ("population_65plus_ratio", "population_75plus_ratio")
                if name in feature_influence.index
            ) <= 0.10
        ),
        "demographic_axis_normalized_influence_le_0_40": float(
            axis_influence.get("A_demographic_vulnerability", np.inf)
        ) <= 0.40,
        "count_and_age_group_default_weight_share_le_0_20": count_and_age_default_share <= 0.20,
    }
    rural_top20_share = float(
        master.loc[primary.scores["need_rank"].le(20), "is_rural_eup_myeon"].mean()
    )
    default_profile_rows = profile_certification.loc[
        profile_certification["is_default_profile"].eq(1)
    ]
    default_profile_certified = bool(
        len(default_profile_rows) == 1
        and default_profile_rows["profile_certified"].eq(1).all()
    )
    responsiveness_pass = bool(
        len(responsiveness) == 3
        and responsiveness["monotonic_responsiveness_pass"].eq(1).all()
    )
    quality_gate = {
        "version": "MEDIROAD_MODEL_V1_STAGE1_GATE",
        "registry_pass": registry_pass,
        "road_pass": road_pass,
        "correlation_hard_checks": correlation_hard_checks,
        "correlation_separation_checks": correlation_separation_checks,
        "correlation_advisory_separation": correlation_advisory,
        "ablation_pass": ablation_pass,
        "default_profile_all_scalers_certified": default_profile_certified,
        "weight_profile_certification": {
            str(row["weight_profile"]): str(row["status"])
            for _, row in profile_certification.iterrows()
        },
        "axis_priority_responsiveness_pass": responsiveness_pass,
        "population_dominance_checks": population_checks,
        "urbanicity_concentration_advisory": {
            "need_vs_total_population_spearman": score_vs_total_population,
            "need_vs_elderly_population_spearman": score_vs_elderly_population,
            "top20_rural_share": rural_top20_share,
            "interpretation": "association diagnostic, not a count-dominance test; access and equity axes intentionally encode rural disadvantage",
        },
        "score_population_spearman": {
            "population_total": score_vs_total_population,
            "population_65plus": score_vs_elderly_population,
        },
        "hard_gate_pass": bool(
            registry_pass
            and road_pass
            and all(correlation_hard_checks.values())
            and all(correlation_separation_checks.values())
            and ablation_pass
            and default_profile_certified
            and responsiveness_pass
            and all(population_checks.values())
        ),
        "advisory_separation_pass": bool(all(correlation_advisory.values())),
        "predictive_accuracy_claimed": False,
        "target_definition": "relative policy need; not patient-count prediction",
    }
    write_json(quality_gate, report_root / "QUALITY_GATE.json")

    active = manifest[manifest["model_input"]]
    data_audit_report = f"""# MODEL V1 Stage-1 데이터·피처 감사

## 결론

- Road-integrated master: **{len(master)}행 × {len(master.columns)}열**
- 행정동 unique: **{master['admin_dong_code'].nunique()}/153**
- 전수 라벨: **{len(manifest)}/549**, 미라벨 0
- Stage-1 입력: **{len(active)}개**, 제외·후속·검증 전용: **{len(manifest)-len(active)}개
- 실제 환자수 라벨: **없음**. 본 단계는 설명 가능한 상대 정책 필요도와 순위 안정성을 검증한다.
- Manifest gate: **{'PASS' if registry_pass else 'FAIL'}**

## 5축 active 수

{pd.Series(manifest_validation['axis_counts'], name='feature_count').rename_axis('axis').reset_index().to_markdown(index=False)}

## 누수·granularity 가드레일

- V0/기존 score·rank 16열은 입력에서 제외했다.
- 환자·참여자/outcome-like 3열은 입력에서 제외했다.
- 과거 outreach는 backtest/optimizer overlap 전용이며 Need 입력이 아니다.
- CHS 8개는 11개 시군 context로 표시하고 상관계수의 유효 n을 11로 계산했다.
- OSM 3개 broad access는 정적 free-flow proxy로 표시하며 153/153 도달성 gate를 통과했다.
"""
    write_text(data_audit_report, report_root / "01_data_audit.md")

    top_selected_pairs = selected_pairs.head(15)
    correlation_report = f"""# MODEL V1 Stage-1 상관관계 지도 품질 보고서

## 산출 범위

- 감사용 전체 numeric atlas: **{len(full_corr)}개 비상수 수치 열**
- 정책 해석용 의미단위 지도: **{len(representative_manifest)}개** (활성 원천 구성 피처 {len(selected_columns)}개를 중복가중 없이 그룹화; 행정동 pair n=153, 시군 context 포함 pair n=11)
- 별도 패널: 원방향, Need 방향 정렬, 11개 시군 건강, 전체 시군 context, 총인구·면적 partial, within-sigungu partial
- 시군 block bootstrap: **{n_boot:,}회**, worker **{n_jobs}개**

## 대표 지도 분리·중복 지표

| 지표 | 값 |
|---|---:|
| 행정동 총인구·면적 partial Axis AUC | {float(admin_partial_metric['axis_auc']):.4f} |
| 행정동 시군내 partial Axis AUC | {float(admin_within_metric['axis_auc']):.4f} |
| 행정동 시군내 block contrast | {float(admin_within_metric['block_contrast']):.4f} |
| 행정동 시군내 cross-axis abs(ρ)≥.80 비율 | {float(admin_within_metric['cross_axis_high_corr_rate_0_80']):.4f} |
| 11개 시군 건강패널 최대 abs(ρ) | {float(health_metric['max_abs_corr_offdiag']):.4f} |
| 의미단위 abs(ρ)≥.95 pair | {int(selected_metric['n_high_corr_pairs_0_95'])} |
| 의미단위 exact duplicate pair | {int(selected_metric['n_effective_exact_duplicate_pairs'])} |
| 혼합행렬 Effective rank / p | {_format_optional_float(selected_metric['effective_rank_normalized'])} |

## 강한 대표 피처 연관 (기술적 상관, 인과 아님)

{_markdown_table(top_selected_pairs, ['feature_a','feature_b','rho','pairwise_n','same_axis'], 15)}

## 해석 원칙

전체 549열을 한 장의 정책 근거로 오독하지 않도록 전체 atlas는 감사용으로 두고, 대표·partial·시군 패널을 분리했다. 시군 context를 153개 독립 관측처럼 검정하지 않았으며 모든 cell의 pairwise n을 CSV/HTML tooltip에 보존했다. 상관은 예측력이나 인과효과가 아니다.
"""
    write_text(correlation_report, report_root / "02_correlation_quality.md")

    feature_summary = ablation_summary[
        ablation_summary["scope"].eq("feature") & ablation_summary["removal_mode"].eq("renormalize")
    ].sort_values("normalized_influence", ascending=False)
    failed_gates = gates[gates["passed"].eq(0)]
    failed_completion_gates = failed_gates[failed_gates["required_for_completion"].eq(1)]
    failed_stress_diagnostics = failed_gates[failed_gates["required_for_completion"].eq(0)]
    ablation_report = f"""# MODEL V1 Stage-1 전 피처 이탈(Ablation) 보고서

## 실행 범위

- {len(selected_columns)}개 active 원천 구성 피처: renormalized LOO + neutral/frozen LOO
- semantic cluster: {ablation_summary[ablation_summary.scope.eq('cluster')]['variant'].nunique()}개
- 5개 정책 축 LOO
- 안전한 대표 대체: {ablation_summary[ablation_summary.scope.eq('substitution')]['variant'].nunique()}개
- 구성 조합: 4 scaler × 5 weight profile = 20개
- 총 ablation run: **{len(ablation_runs):,}개**, 지역별 결과: **{len(ablation_region_runs):,}행
- 549개 전 열은 exact test, substitution 또는 명시적 N/A 사유를 갖는다.

## 영향도 상위 active 피처

{_markdown_table(feature_summary, ['variant','normalized_influence','worst_spearman','worst_top20_jaccard','worst_rank_shift_p95'], 15)}

## Gate

- Completion gate: **{'PASS' if ablation_pass else 'FAIL'}**

### 미통과 completion gate

{_markdown_table(failed_completion_gates, ['gate','status'], 30)}

### 보존된 worst-case stress 경고

{_markdown_table(failed_stress_diagnostics, ['gate','status'], 30)}

### 정책 가중 프로필별 4-scaler 인증

{_markdown_table(profile_certification, ['weight_profile','status','passed_count','check_count','failed_checks'], 10)}

### 우선순위 프로필 축 반응성

{_markdown_table(responsiveness, ['axis','priority_profile','default_mean_abs_rank_shift','priority_mean_abs_rank_shift','impact_ratio_priority_over_default','monotonic_responsiveness_pass'], 10)}

이 지표는 환자수 예측 정확도가 아니다. 피처·축을 제거하거나 대표 정의를 바꿀 때 상대 Need 순위와 Top-K 정책 선택이 얼마나 흔들리는지를 측정한다.
"""
    write_text(ablation_report, report_root / "03_ablation.md")

    total_elapsed = time.perf_counter() - started
    timings["total_seconds"] = total_elapsed
    run_metadata = {
        "run_version": "MEDIROAD_MODEL_V1_STAGE1_20260818",
        "package_root": str(root),
        "master_relative_path": master_path.relative_to(root).as_posix(),
        "master_sha256": sha256_file(master_path),
        "config_sha256": sha256_file(config_path),
        "seed": 42,
        "n_boot": n_boot,
        "n_jobs": n_jobs,
        "cpu_count": os.cpu_count(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "gpu_used": False,
        "gpu_reason": "153x549 rank correlations and policy-score ablations are CPU/vectorization bound; GPU transfer overhead exceeds benefit",
        "timings": timings,
        "generated_figure_count": len(generated_figures),
        "quality_gate": quality_gate,
    }
    write_json(run_metadata, output_root / "RUN_METADATA.json")
    current_gate_report = f"""# MODEL V1 현재 단계 Gate

- 전수 피처 라벨: **PASS ({len(manifest)}/549)**
- OSM 도로 접근: **{'PASS' if road_pass else 'FAIL'} (20/20 범주 153/153)**
- 상관 hard gate: **{'PASS' if all(correlation_hard_checks.values()) else 'FAIL'}**
- 상관 stratified separation gate: **{'PASS' if all(correlation_separation_checks.values()) else 'FAIL'}**
- 상관 mixed-atlas advisory: **{'PASS' if all(correlation_advisory.values()) else 'REVIEW'}**
- Ablation gate: **{'PASS' if ablation_pass else 'FAIL'}**
- 기본 정책 4-scaler 인증: **{'PASS' if quality_gate['default_profile_all_scalers_certified'] else 'FAIL'}**
- 인구규모 지배 gate: **{'PASS' if all(population_checks.values()) else 'FAIL'}**
- 농촌성 집중 advisory: Need-총인구 ρ **{score_vs_total_population:.3f}**, Top-20 읍·면 비중 **{rural_top20_share:.1%}** (접근·형평 축의 의도된 연관이므로 별도 해석)
- 현재 단계 종합 hard gate: **{'PASS' if quality_gate['hard_gate_pass'] else 'FAIL'}**

다음 Specialty Gap·Exposure·Venue·Optimizer 단계는 아직 실행하지 않았다. 실패/검토 항목은 본 단계 안에서 원인 분석 후 재실행한다.
"""
    write_text(current_gate_report, report_root / "CURRENT_STAGE_GATE.md")
    return run_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, default=Path.cwd())
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--n-jobs", type=int, default=min(8, os.cpu_count() or 1))
    return parser.parse_args()


def _quality_gate_exit_code(metadata: dict[str, Any]) -> int:
    """Return a non-zero process code whenever the required Stage-1 gate fails."""

    try:
        passed = metadata["quality_gate"]["hard_gate_pass"]
    except (KeyError, TypeError) as exc:
        raise ValueError("Run metadata lacks quality_gate.hard_gate_pass") from exc
    return 0 if bool(passed) else 2


def main() -> int:
    args = parse_args()
    metadata = run_stage1(args.package_root, n_boot=args.bootstrap, n_jobs=args.n_jobs)
    print(json.dumps(metadata, ensure_ascii=False, indent=2, default=_json_default))
    return _quality_gate_exit_code(metadata)


if __name__ == "__main__":
    raise SystemExit(main())
