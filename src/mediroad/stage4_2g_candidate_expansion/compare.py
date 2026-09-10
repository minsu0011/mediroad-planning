from __future__ import annotations

from dataclasses import asdict
from typing import Any

import pandas as pd

from .config import CandidateThresholds


def _relative_loss(base: float, reference: float) -> float:
    return max(0.0, (float(reference) - float(base)) / max(abs(float(reference)), 1e-12))


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def evaluate_top3_reduction(
    *,
    baseline_metrics: dict[str, float],
    baseline_plan: pd.DataFrame,
    baseline_max_gap: float,
    alternatives: dict[str, tuple[dict[str, float], pd.DataFrame, float, bool]],
    thresholds: CandidateThresholds,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in ["top5", "coarse_pareto"]:
        if name not in alternatives:
            raise KeyError(f"Missing mandatory comparator {name}")
        metrics, plan, max_gap, certified = alternatives[name]
        unique_loss = _relative_loss(baseline_metrics["unique_elderly_population"], metrics["unique_elderly_population"])
        need_loss = _relative_loss(baseline_metrics["need_weighted_population"], metrics["need_weighted_population"])
        high_loss = _relative_loss(baseline_metrics["high_need_population"], metrics["high_need_population"])
        min_loss = max(0.0, float(metrics["min_sigungu_coverage_ratio"]) - float(baseline_metrics["min_sigungu_coverage_ratio"]))
        admin_j = _jaccard(set(baseline_plan["admin_code"].astype(str)), set(plan["admin_code"].astype(str)))
        base_cluster = baseline_plan["cluster_id"].fillna(baseline_plan["venue_id"]).astype(str)
        alt_cluster = plan["cluster_id"].fillna(plan["venue_id"]).astype(str)
        cluster_j = _jaccard(set(base_cluster), set(alt_cluster))
        combined_gap = max(float(baseline_max_gap), float(max_gap))
        checks = {
            "unique_coverage_loss_passed": unique_loss <= thresholds.max_unique_coverage_loss_fraction + 1e-15,
            "need_weighted_loss_passed": need_loss <= thresholds.max_need_weighted_loss_fraction + 1e-15,
            "high_need_loss_passed": high_loss <= thresholds.max_high_need_loss_fraction + 1e-15,
            "min_sigungu_loss_passed": min_loss <= thresholds.max_min_sigungu_coverage_loss + 1e-15,
            "admin_jaccard_passed": admin_j >= thresholds.min_selected_admin_jaccard - 1e-15,
            "coverage_cluster_jaccard_passed": cluster_j >= thresholds.min_selected_coverage_cluster_jaccard - 1e-15,
            "solver_gap_passed": bool(certified and combined_gap <= thresholds.max_solver_relative_gap + 1e-12),
        }
        rows.append({
            "candidate_set": "top3",
            "reference_candidate_set": name,
            "objective_metric": "need_weighted_population",
            "objective_loss_fraction": need_loss,
            "unique_coverage_loss_fraction": unique_loss,
            "need_weighted_loss_fraction": need_loss,
            "high_need_loss_fraction": high_loss,
            "min_sigungu_coverage_loss": min_loss,
            "selected_admin_jaccard": admin_j,
            "selected_coverage_cluster_jaccard": cluster_j,
            "max_solver_relative_gap": combined_gap,
            **checks,
            "comparison_passed": bool(all(checks.values())),
        })
    frame = pd.DataFrame(rows)
    solver_conclusive = bool(frame["solver_gap_passed"].all())
    top3_passed = bool(frame["comparison_passed"].all())
    if not solver_conclusive:
        decision = "INCONCLUSIVE_CANDIDATE_SENSITIVITY_SOLVER_GAP"
        recommended = None
        passed = False
    elif top3_passed:
        decision = "PASS_RETAIN_TOP3"
        recommended = "top3"
        passed = True
    else:
        decision = "PROMOTE_PREDECLARED_WIDER_CANDIDATE_SET"
        recommended = "top5"
        passed = True
    summary = {
        "decision": decision,
        "passed": passed,
        "recommended_candidate_set": recommended,
        "thresholds": asdict(thresholds),
        "threshold_fingerprint": thresholds.fingerprint,
        "mandatory_comparators": ["top5", "coarse_pareto"],
        "fallback_order": ["top5", "coarse_pareto"],
    }
    return frame, summary
