from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

FROZEN_THRESHOLD_FINGERPRINT = 'b0a1a2c6dc6f1d1e7c3ae1141700fbb3c54854b03746d2e45d2d0ba33c3b1ad4'


@dataclass(frozen=True)
class CandidateThresholds:
    max_unique_coverage_loss_fraction: float = 0.01
    max_need_weighted_loss_fraction: float = 0.01
    max_high_need_loss_fraction: float = 0.02
    max_min_sigungu_coverage_loss: float = 0.02
    min_selected_admin_jaccard: float = 0.50
    min_selected_coverage_cluster_jaccard: float = 0.50
    max_solver_relative_gap: float = 0.005

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError(f'Expected mapping: {path}')
    return value


def set_thread_environment() -> None:
    # Give the exact MIP solver the CPU. Avoid BLAS/OpenMP oversubscription.
    for key in (
        'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
        'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS',
    ):
        os.environ[key] = '1'
    os.environ.setdefault('PYTHONHASHSEED', '42')


def thresholds_from_config(cfg: dict[str, Any]) -> CandidateThresholds:
    section = cfg['ladder']
    return CandidateThresholds(
        max_unique_coverage_loss_fraction=float(section['max_unique_coverage_loss_fraction']),
        max_need_weighted_loss_fraction=float(section['max_need_weighted_loss_fraction']),
        max_high_need_loss_fraction=float(section['max_high_need_loss_fraction']),
        max_min_sigungu_coverage_loss=float(section['max_min_sigungu_coverage_loss']),
        min_selected_admin_jaccard=float(section['min_selected_admin_jaccard']),
        min_selected_coverage_cluster_jaccard=float(section['min_selected_coverage_cluster_jaccard']),
        max_solver_relative_gap=float(section['max_solver_relative_gap']),
    )


def validate_config(cfg: dict[str, Any]) -> None:
    required = {'version', 'paths', 'hardware', 'parent_stage42g', 'ladder', 'equity', 'solver', 'oracle', 'gpu_lns', 'proof_strengthening', 'scip_portfolio'}
    missing = sorted(required - set(cfg))
    if missing:
        raise ValueError(f'Stage4.2H config missing sections: {missing}')
    ladder = list(cfg['ladder'].get('promotion_ladder', []))
    if ladder != ['top3', 'top5', 'coarse_pareto']:
        raise ValueError('Promotion ladder is frozen to top3 -> top5 -> coarse_pareto')
    thresholds = thresholds_from_config(cfg)
    if thresholds.fingerprint != FROZEN_THRESHOLD_FINGERPRINT:
        raise ValueError(
            'Candidate sensitivity thresholds changed; expected frozen fingerprint '
            f'{FROZEN_THRESHOLD_FINGERPRINT}, got {thresholds.fingerprint}'
        )
    if str(cfg['ladder'].get('expected_final_candidate_set')) != 'coarse_pareto':
        raise ValueError('This repair package is pinned to the observed G evidence: final rung must be coarse_pareto')
    if str(cfg['equity'].get('candidate_set')) != 'coarse_pareto':
        raise ValueError('Stage4.2H recertifies Equity only on coarse_pareto')
    if int(cfg['equity'].get('visit_count', 0)) != 20:
        raise ValueError('visit_count is frozen at 20')
    equity = cfg['equity']
    exact_equity_contract = {
        'total_population_efficiency_fraction': 0.60,
        'min_sigungu_retention': 1.00,
        'high_need_retention': 0.99,
        'need_weighted_retention': 0.99,
    }
    for key, expected in exact_equity_contract.items():
        if abs(float(equity.get(key, float('nan'))) - expected) > 1e-12:
            raise ValueError(f'Frozen Equity contract changed: {key}={equity.get(key)!r}, expected {expected}')
    if not bool(equity.get('preserve_stage42_objectives', False)) or not bool(equity.get('preserve_stage42_constraints', False)):
        raise ValueError('Stage4.2 objectives/constraints must remain frozen')
    if int(equity.get('expected_candidate_count', 0)) != 896 or int(equity.get('expected_pattern_count', 0)) != 10838:
        raise ValueError('Pinned coarse-Pareto reduction dimensions changed')
    if abs(float(cfg['solver']['near_optimal_relative_gap_max']) - 0.005) > 1e-12:
        raise ValueError('near-optimality threshold is frozen at 0.005')
    if int(cfg['hardware'].get('highs_threads', 0)) != 8:
        raise ValueError('Official Stage4.2H uses all 8 logical CPU threads for HiGHS')
    mem = float(cfg['hardware'].get('memory_soft_limit_gib', 0))
    if not 24 <= mem <= 28:
        raise ValueError('32GiB host requires a 24..28GiB solver soft limit')
    if not bool(cfg['gpu_lns'].get('enabled', False)):
        raise ValueError('Official Stage4.2H requires GPU LNS')
    if not bool(cfg['gpu_lns'].get('require_cuda_for_official', False)):
        raise ValueError('Official Stage4.2H must fail closed if CUDA is unavailable')
    workers = int(cfg['scip_portfolio'].get('workers', 0))
    if not 1 <= workers <= 6:
        raise ValueError('SCIP portfolio workers must stay in 1..6 on the 32GiB host')
    proof = cfg['proof_strengthening']
    if not bool(proof.get('enabled', False)):
        raise ValueError('Stage4.2H outward-safe proof strengthening is required')
    if list(proof.get('anchor_sizes', [])) != list(range(21)):
        raise ValueError('Stage4.2H proof anchors are frozen to all prefix sizes 0..20')
    if int(proof.get('proof_anchor_seed_count', 0)) != 96 or int(proof.get('proof_anchor_random_seed', 0)) != 42062:
        raise ValueError('Stage4.2H diversified proof-anchor contract changed')
    if int(proof.get('max_cuts_per_metric', 0)) != 256:
        raise ValueError('Stage4.2H per-metric cut cap changed')
    if int(proof.get('outer_approximation_max_rounds', 0)) != 16:
        raise ValueError('Stage4.2H outer-approximation round contract changed')
    if list(proof.get('binary_coverage_objectives', [])):
        raise ValueError('Coarse-Pareto proof must retain the exact continuous-y projection')
    highs_limits = proof.get('highs_first_time_limit_sec', {})
    if float(highs_limits.get('min_sigungu_coverage', 0)) < 1200 or float(highs_limits.get('high_need_population', 0)) < 1200:
        raise ValueError('Stage4.2H strengthened HiGHS front-stage budgets must be at least 1200 seconds')
    if bool(cfg.get('stage5_started', False)):
        raise ValueError('Stage4.2H must not start Stage5')
