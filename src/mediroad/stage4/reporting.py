from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .types import QualityCheck, ScenarioSolution
from .utils import atomic_write_csv, atomic_write_json, atomic_write_text


def write_solution_artifacts(
    solutions: list[ScenarioSolution],
    outdir: Path,
) -> pd.DataFrame:
    outdir.mkdir(parents=True, exist_ok=True)
    metric_rows: list[dict[str, Any]] = []
    for solution in solutions:
        safe = f"{solution.scenario}__{solution.catchment_id}__n{solution.visit_count}".replace("/", "_")
        selected = solution.selected.copy()
        selected.insert(0, "scenario", solution.scenario)
        selected.insert(1, "catchment_id", solution.catchment_id)
        selected.insert(2, "visit_count", solution.visit_count)
        if solution.bundle_assignments is not None and not solution.bundle_assignments.empty:
            assignment_cols = [
                c for c in ["venue_id", "bundle_id", "bundle_value", "bundle_value_within_venue_pct", "bundle_assignment_status"]
                if c in solution.bundle_assignments
            ]
            selected = selected.drop(columns=[c for c in assignment_cols if c != "venue_id" and c in selected], errors="ignore")
            selected = selected.merge(
                solution.bundle_assignments[assignment_cols], on="venue_id", how="left", validate="one_to_one"
            )
        atomic_write_csv(selected, outdir / f"provisional_plan__{safe}.csv")
        stage_rows = []
        for index, stage in enumerate(solution.stages, start=1):
            stage_rows.append(
                {
                    "stage_order": index,
                    "objective": stage.objective_name,
                    "sense": stage.sense,
                    "status": stage.status,
                    "objective_value": stage.objective_value,
                    "best_bound": stage.best_bound,
                    "wall_time_sec": stage.wall_time_sec,
                    "selected_count": len(stage.selected_indices),
                    "covered_grid_count": len(stage.covered_grid_indices),
                }
            )
        if stage_rows:
            atomic_write_csv(pd.DataFrame(stage_rows), outdir / f"solver_stages__{safe}.csv")
        row: dict[str, Any] = {
            "scenario": solution.scenario,
            "catchment_id": solution.catchment_id,
            "visit_count": solution.visit_count,
            "status": solution.status,
        }
        for key, value in solution.metrics.items():
            if isinstance(value, (dict, list, tuple)):
                row[key] = json.dumps(value, ensure_ascii=False)
            else:
                row[key] = value
        metric_rows.append(row)
    metrics = pd.DataFrame(metric_rows)
    atomic_write_csv(metrics, outdir / "scenario_metrics.csv")
    return metrics


def write_quality_gate(checks: list[QualityCheck], path: Path) -> pd.DataFrame:
    df = pd.DataFrame([check.as_dict() for check in checks])
    atomic_write_csv(df, path)
    return df


def create_figures(
    metrics: pd.DataFrame,
    selection_frequency: pd.DataFrame,
    network_summary: pd.DataFrame,
    frontier: pd.DataFrame,
    outdir: Path,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mediroad.reporting.correlation import _matplotlib_korean_font

    korean_font = _matplotlib_korean_font()

    outdir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    style = {
        "font.family": "sans-serif",
        "font.sans-serif": [korean_font, "DejaVu Sans"],
        "axes.unicode_minus": False,
    }

    if not metrics.empty and "unique_elderly_population" in metrics:
        with plt.rc_context(style):
            fig, ax = plt.subplots(figsize=(10, 6))
            labels = metrics["scenario"].astype(str) + "\n" + metrics["catchment_id"].astype(str)
            ax.bar(labels, metrics["unique_elderly_population"].astype(float))
            ax.set_ylabel("Potential beneficiary elderly coverage")
            ax.set_title("Stage 4 provisional scenario comparison")
            ax.tick_params(axis="x", rotation=45)
            fig.tight_layout()
            path = outdir / "scenario_unique_coverage.png"
            fig.savefig(path, dpi=300)
            plt.close(fig)
            paths.append(path)

    if not selection_frequency.empty:
        top = selection_frequency.head(30).copy()
        with plt.rc_context(style):
            fig, ax = plt.subplots(figsize=(10, 8))
            labels = top.get("venue_name", top["venue_id"]).astype(str)
            ax.barh(labels[::-1], top["selection_frequency"].astype(float).to_numpy()[::-1])
            ax.set_xlabel("Selection frequency across scenario/catchment runs")
            ax.set_title("Robust provisional venue candidates")
            fig.tight_layout()
            path = outdir / "robust_selection_frequency.png"
            fig.savefig(path, dpi=300)
            plt.close(fig)
        paths.append(path)

    if not network_summary.empty:
        with plt.rc_context(style):
            fig, ax = plt.subplots(figsize=(9, 6))
            ax.plot(
                network_summary["threshold_minutes"].astype(float),
                network_summary["admin_best_venue_retention"].astype(float),
                marker="o",
                label="Best venue retention",
            )
            ax.plot(
                network_summary["threshold_minutes"].astype(float),
                network_summary["admin_top3_jaccard"].astype(float),
                marker="o",
                label="Admin Top3 Jaccard",
            )
            ax.set_xlabel("Static OSM inbound travel-time threshold (minutes)")
            ax.set_ylabel("Stability vs Euclidean 5km")
            ax.set_ylim(0, 1)
            ax.legend()
            ax.set_title("Pre-Stage 4 network catchment sensitivity")
            fig.tight_layout()
            path = outdir / "network_catchment_sensitivity.png"
            fig.savefig(path, dpi=300)
            plt.close(fig)
            paths.append(path)

    if not frontier.empty and "unique_elderly_population" in frontier:
        with plt.rc_context(style):
            fig, ax = plt.subplots(figsize=(9, 6))
            for scenario, group in frontier.groupby("scenario"):
                group = group.sort_values("visit_count")
                ax.plot(
                    group["visit_count"].astype(int),
                    group["unique_elderly_population"].astype(float),
                    marker="o",
                    label=str(scenario),
                )
            ax.set_xlabel("Visit count")
            ax.set_ylabel("Potential beneficiary elderly coverage")
            ax.set_title("Visit-count coverage frontier")
            ax.legend()
            fig.tight_layout()
            path = outdir / "visit_count_frontier.png"
            fig.savefig(path, dpi=300)
            plt.close(fig)
        paths.append(path)
    return paths


def build_markdown_report(
    run_id: str,
    status: str,
    preflight_meta: dict[str, Any],
    candidate_summary: dict[str, Any],
    network_payload: dict[str, Any],
    metrics: pd.DataFrame,
    baseline_metrics: pd.DataFrame,
    selection_frequency: pd.DataFrame,
    frontier: pd.DataFrame,
    checks: list[QualityCheck],
) -> str:
    hard = [c for c in checks if c.severity == "HARD"]
    hard_pass = sum(c.passed for c in hard)
    advisory = [c for c in checks if c.severity != "HARD"]
    advisory_fail = sum(not c.passed for c in advisory)
    lines = [
        "# MEDIROAD MODEL V1 Stage 4 — Provisional Multi-objective Optimization Report",
        "",
        f"- run_id: `{run_id}`",
        f"- status: **{status}**",
        f"- hard gates: `{hard_pass}/{len(hard)}` PASS",
        f"- advisory failures: `{advisory_fail}`",
        "- Stage 5 scheduling started: `false`",
        "",
        "## 1. Project direction reflected in this stage",
        "",
        "Stage 4 does not rank venues with a single Need×Exposure×Specialty score. It selects a provisional set of visit catchments by exact unique grid coverage, keeps efficiency and equity as separate policy objectives, and assigns the Stage 2A service bundle only after the spatial set is selected. Stage 2B remains advisory because no strong seasonal preference was established.",
        "",
        "## 2. Frozen input and Stage 3 interface",
        "",
        f"- Stage 3 run root: `{preflight_meta.get('stage3_run_root')}`",
        f"- grid rows: `{preflight_meta.get('n_grids')}`",
        f"- venue universe: `{preflight_meta.get('n_venues')}`",
        f"- bundle universe: `{preflight_meta.get('n_bundles')}`",
        f"- provisional optimization candidates: `{candidate_summary.get('candidate_count')}`",
        f"- admin coverage: `{candidate_summary.get('admin_count')}`",
        "",
        "## 3. Pre-Stage 4 road-network catchment validation",
        "",
        f"- status: `{network_payload.get('status')}`",
        f"- selected robustness comparator: `{network_payload.get('best_network_matrix_id')}`",
        "- interpretation: static OSM inbound car-profile free-flow proxy; not observed elderly travel, real service area, or live traffic.",
        "",
        "## 4. Provisional scenarios",
        "",
    ]
    if metrics.empty:
        lines.append("No valid scenario solution was produced.")
    else:
        columns = [
            c for c in [
                "scenario", "catchment_id", "visit_count", "status", "unique_elderly_population",
                "redundancy_ratio", "min_sigungu_coverage_ratio", "visit_location_sigungu_count",
                "travel_minutes_sum_proxy",
            ] if c in metrics
        ]
        lines.append(metrics[columns].to_markdown(index=False))
    lines.extend(["", "## 5. Baseline comparison", ""])
    if baseline_metrics.empty:
        lines.append("No baseline result.")
    else:
        columns = [c for c in ["scenario", "catchment_id", "unique_elderly_population", "redundancy_ratio"] if c in baseline_metrics]
        lines.append(baseline_metrics[columns].to_markdown(index=False))
        lines.append(
            "\nBASELINE_CONSTRAINED_GREEDY_EFFICIENCY uses the same admin, "
            "sigungu, and coverage-cluster caps as CP-SAT. Other baselines are "
            "diagnostic comparators and may violate one or more policy caps."
        )
    lines.extend(["", "## 6. Robust core and field-validation queue", ""])
    if selection_frequency.empty:
        lines.append("No robust-core frequency table.")
    else:
        cols = [c for c in ["venue_id", "venue_name", "admin_code", "sigungu", "selection_frequency"] if c in selection_frequency]
        lines.append(selection_frequency.head(20)[cols].to_markdown(index=False))
    lines.extend(["", "## 7. Visit-count frontier", ""])
    if frontier.empty:
        lines.append("Frontier was not executed in this mode.")
    else:
        cols = [c for c in ["scenario", "visit_count", "unique_elderly_population", "min_sigungu_coverage_ratio"] if c in frontier]
        lines.append(frontier[cols].to_markdown(index=False))
    lines.extend(
        [
            "",
            "## 8. Interpretation contract",
            "",
            "- Coverage is potential beneficiary elderly population mass under a spatial or static-network catchment proxy, not expected patients.",
            "- Selected venues are provisional candidates. Parking, power, toilets, indoor waiting space, vehicle access, equipment space, and actual availability require field/operator confirmation.",
            "- Exact month/date and live weather scheduling remain null and belong to Stage 5.",
            "- The provisional optimizer must be rerun after the robust-core field-validation queue is checked.",
            "",
            "## 9. Next action",
            "",
            "Validate the robust core and fallback venues, update operational feasibility and team/vehicle constraints, then rerun Stage 4 final. Do not treat this provisional run as the final annual operating schedule.",
        ]
    )
    return "\n".join(lines) + "\n"
