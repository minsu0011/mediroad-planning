from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from .config import CandidateThresholds


@dataclass(frozen=True)
class CandidateView:
    name: str
    metrics: dict[str, float]
    plan: pd.DataFrame
    max_gap: float
    certified: bool


def _loss(base: float, reference: float) -> float:
    return max(0.0, (float(reference) - float(base)) / max(abs(float(reference)), 1e-12))


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def compare_pair(base: CandidateView, reference: CandidateView, thresholds: CandidateThresholds) -> dict[str, Any]:
    unique_loss = _loss(base.metrics['unique_elderly_population'], reference.metrics['unique_elderly_population'])
    need_loss = _loss(base.metrics['need_weighted_population'], reference.metrics['need_weighted_population'])
    high_loss = _loss(base.metrics['high_need_population'], reference.metrics['high_need_population'])
    min_loss = max(0.0, float(reference.metrics['min_sigungu_coverage_ratio']) - float(base.metrics['min_sigungu_coverage_ratio']))
    admin_j = _jaccard(set(base.plan['admin_code'].astype(str)), set(reference.plan['admin_code'].astype(str)))
    base_cluster = base.plan['cluster_id'].fillna(base.plan['venue_id']).astype(str)
    ref_cluster = reference.plan['cluster_id'].fillna(reference.plan['venue_id']).astype(str)
    cluster_j = _jaccard(set(base_cluster), set(ref_cluster))
    combined_gap = max(float(base.max_gap), float(reference.max_gap))
    checks = {
        'unique_coverage_loss_passed': unique_loss <= thresholds.max_unique_coverage_loss_fraction + 1e-15,
        'need_weighted_loss_passed': need_loss <= thresholds.max_need_weighted_loss_fraction + 1e-15,
        'high_need_loss_passed': high_loss <= thresholds.max_high_need_loss_fraction + 1e-15,
        'min_sigungu_loss_passed': min_loss <= thresholds.max_min_sigungu_coverage_loss + 1e-15,
        'admin_jaccard_passed': admin_j >= thresholds.min_selected_admin_jaccard - 1e-15,
        'coverage_cluster_jaccard_passed': cluster_j >= thresholds.min_selected_coverage_cluster_jaccard - 1e-15,
        'solver_gap_passed': bool(base.certified and reference.certified and combined_gap <= thresholds.max_solver_relative_gap + 1e-12),
    }
    return {
        'candidate_set': base.name,
        'reference_candidate_set': reference.name,
        'objective_metric': 'need_weighted_population',
        'objective_loss_fraction': need_loss,
        'unique_coverage_loss_fraction': unique_loss,
        'need_weighted_loss_fraction': need_loss,
        'high_need_loss_fraction': high_loss,
        'min_sigungu_coverage_loss': min_loss,
        'selected_admin_jaccard': admin_j,
        'selected_coverage_cluster_jaccard': cluster_j,
        'max_solver_relative_gap': combined_gap,
        **checks,
        'comparison_passed': bool(all(checks.values())),
    }


def evaluate_sequential_ladder(
    views: dict[str, CandidateView],
    *,
    ladder: list[str],
    thresholds: CandidateThresholds,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if ladder != ['top3', 'top5', 'coarse_pareto']:
        raise ValueError(f'Unexpected ladder: {ladder}')
    for name in ladder:
        if name not in views:
            raise KeyError(f'Missing candidate view: {name}')
        if not views[name].certified or views[name].max_gap > thresholds.max_solver_relative_gap + 1e-12:
            return pd.DataFrame(), pd.DataFrame(), {
                'decision': 'INCONCLUSIVE_CANDIDATE_LADDER_SOLVER_GAP',
                'passed': False,
                'recommended_candidate_set': None,
                'thresholds': asdict(thresholds),
                'threshold_fingerprint': thresholds.fingerprint,
            }

    comparisons: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []
    selected = ladder[0]
    for index, current in enumerate(ladder):
        later = ladder[index + 1 :]
        if not later:
            selected = current
            transitions.append({
                'from_candidate_set': current,
                'action': 'FINAL_RUNG_REACHED',
                'to_candidate_set': current,
                'all_wider_comparisons_passed': True,
                'failed_wider_sets': '',
            })
            break
        rows = [compare_pair(views[current], views[other], thresholds) for other in later]
        comparisons.extend(rows)
        failed = [row['reference_candidate_set'] for row in rows if not row['comparison_passed']]
        if not failed:
            selected = current
            transitions.append({
                'from_candidate_set': current,
                'action': 'RETAIN_CURRENT_RUNG',
                'to_candidate_set': current,
                'all_wider_comparisons_passed': True,
                'failed_wider_sets': '',
            })
            break
        next_rung = ladder[index + 1]
        transitions.append({
            'from_candidate_set': current,
            'action': 'PROMOTE_ONE_RUNG',
            'to_candidate_set': next_rung,
            'all_wider_comparisons_passed': False,
            'failed_wider_sets': ';'.join(failed),
        })
        selected = next_rung

    final_rung = selected == ladder[-1]
    decision = (
        'PASS_SEQUENTIAL_LADDER_FINAL_RUNG_COARSE_PARETO'
        if final_rung
        else f'PASS_SEQUENTIAL_LADDER_RETAIN_{selected.upper()}'
    )
    summary = {
        'decision': decision,
        'passed': True,
        'recommended_candidate_set': selected,
        'final_rung_reached': final_rung,
        'promotion_ladder': ladder,
        'thresholds': asdict(thresholds),
        'threshold_fingerprint': thresholds.fingerprint,
    }
    return pd.DataFrame(comparisons), pd.DataFrame(transitions), summary
