from __future__ import annotations

import copy
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mediroad.stage4_2_certification.adapter import load_base_stage42_config, prepare_candidate_problem
from mediroad.stage4_2_certification.backends import HighsBackend
from mediroad.stage4_2_certification.config import resolve_hardware
from mediroad.stage4_2_certification.engine import CertificationEngine
from mediroad.stage4_2_certification.formulation import CertificationFormulation
from mediroad.stage4_2_certification.io_utils import atomic_write_json
from mediroad.stage4_2_certification.types import ScenarioCertificationResult
from mediroad.stage4_2g_candidate_expansion.runner import _rescue_scenario

from .gpu_lns import CoarseEquityGpuLns, GpuSearchResult


def _map_plan(formulation: CertificationFormulation, plan: pd.DataFrame) -> np.ndarray:
    positions = {v: i for i, v in enumerate(formulation.candidates['venue_id'].astype(str))}
    aliases = getattr(formulation, 'venue_alias_map', {})
    selected: list[int] = []
    for raw in plan['venue_id'].astype(str):
        canonical = str(aliases.get(raw, raw))
        if canonical not in positions:
            raise RuntimeError(f'Seed venue missing from coarse universe: {raw} -> {canonical}')
        selected.append(int(positions[canonical]))
    arr = np.asarray(selected, dtype=int)
    valid, reason = formulation.validate_selection(arr)
    if not valid:
        raise RuntimeError(f'Mapped seed invalid: {reason}')
    return arr


def _result_score(result: ScenarioCertificationResult) -> tuple[int, float, float, float, float]:
    # Prefer certification first, then lexicographic stage metrics.
    m = result.metrics
    return (
        1 if result.certified else 0,
        float(m.get('min_sigungu_coverage_ratio', -np.inf)),
        float(m.get('high_need_population', -np.inf)),
        float(m.get('need_weighted_population', -np.inf)),
        -float(m.get('cost', np.inf)),
    )


def _stage_retained_min_floor(result: ScenarioCertificationResult) -> float:
    if not result.stages:
        raise RuntimeError('Equity result has no stages')
    floor = result.stages[0].retained_floor
    if floor is not None and floor.objective == 'min_sigungu_coverage':
        return float(floor.value)
    # Fallback mirrors the frozen 1.0 retention and 1e-6 objective units.
    value = float(result.stages[0].objective_value)
    scale = 1_000_000
    return int(np.floor(value * scale + 1e-9)) / scale


def _gpu_payload(result: GpuSearchResult) -> dict[str, Any]:
    payload = asdict(result)
    payload['selected_indices'] = [int(x) for x in result.selected_indices]
    return payload


def solve_coarse_equity(
    *,
    project_root: Path,
    cfg: dict[str, Any],
    g_efficiency_metrics: dict[str, Any],
    top3_equity_plan: pd.DataFrame,
    output_dir: Path,
) -> tuple[ScenarioCertificationResult, CertificationFormulation, dict[str, Any]]:
    root = project_root.resolve()
    base_cfg = load_base_stage42_config(root / 'configs/model_v1/stage4_2.yaml')
    frozen_opt = base_cfg.get('optimization', {})
    expected_contract = {
        'visit_count': 20,
        'equity_min_efficiency_fraction': 0.60,
    }
    for key, expected in expected_contract.items():
        actual = frozen_opt.get(key)
        if actual is None or abs(float(actual) - float(expected)) > 1e-12:
            raise RuntimeError(f'Base Stage4.2 Equity contract changed: {key}={actual!r}, expected {expected}')
    retention = frozen_opt.get('objective_retention', {})
    for key, expected in {'min_sigungu_coverage': 1.0, 'high_need': 0.99, 'need_weighted': 0.99}.items():
        actual = retention.get(key)
        if actual is None or abs(float(actual) - expected) > 1e-12:
            raise RuntimeError(f'Base Stage4.2 retention changed: {key}={actual!r}, expected {expected}')
    cert_path = root / 'configs/model_v1/stage4_2_certification.yaml'
    import yaml
    cert = yaml.safe_load(cert_path.read_text(encoding='utf-8'))
    if not isinstance(cert, dict):
        raise RuntimeError('Invalid Stage4.2C certification config')
    cert = copy.deepcopy(cert)
    cert['hardware'].update({
        'highs_threads': 8,
        'memory_limit_gb': float(cfg['hardware']['memory_soft_limit_gib']),
        'gpu_seed_search': False,  # H owns GPU LNS explicitly.
        'native_windows_highs_threads': 4,
    })
    cert['solver']['time_limit_sec'] = copy.deepcopy(cfg['solver']['time_limit_sec'])
    cert['solver']['mip_heuristic_effort'] = float(cfg['solver'].get('mip_heuristic_effort', 0.20))
    cert['solver']['oracle_heuristic_effort'] = float(cfg['solver'].get('oracle_heuristic_effort', 0.08))
    cert['solver']['mip_max_start_nodes'] = int(cfg['solver'].get('mip_max_start_nodes', 50000))
    cert['solver']['extra_highs_options'] = copy.deepcopy(cfg['solver'].get('extra_highs_options', {}))
    cert['oracle'].update(copy.deepcopy(cfg['oracle']))
    cert['seed_search'].update({'single_swap_enabled': True, 'max_rounds': 16, 'cpu_max_rounds': 4})
    cert['gate']['near_optimal_relative_gap_max'] = 0.005

    problem = prepare_candidate_problem(root, 'coarse_pareto', base_cfg, cache={})
    formulation = CertificationFormulation(
        problem.candidates, problem.patterns, base_cfg, cert, candidate_set='coarse_pareto'
    )
    formulation.venue_alias_map = dict(problem.venue_alias_map)
    if len(problem.candidates) != int(cfg['equity']['expected_candidate_count']):
        raise RuntimeError(f"coarse candidate count changed: {len(problem.candidates)}")
    if int(problem.patterns.n_patterns) != int(cfg['equity']['expected_pattern_count']):
        raise RuntimeError(f"coarse pattern count changed: {problem.patterns.n_patterns}")

    mapped_top3 = _map_plan(formulation, top3_equity_plan)
    efficiency_reference = float(g_efficiency_metrics['unique_elderly_population'])
    total_floor = efficiency_reference * float(base_cfg['optimization']['equity_min_efficiency_fraction'])
    mapped_metrics = formulation.metrics(mapped_top3)
    if float(mapped_metrics['unique_elderly_population']) < total_floor - 1e-6:
        raise RuntimeError('Certified Top3 Equity seed violates the coarse Equity total-population floor')

    lns = CoarseEquityGpuLns(formulation, cfg['gpu_lns'])
    if bool(cfg['gpu_lns'].get('require_cuda_for_official', True)) and lns.backend != 'CUPY_CUDA':
        raise RuntimeError(f'Official Stage4.2H requires RTX/CuPy CUDA, got {lns.backend}')

    min_search = lns.search(
        mapped_top3,
        objective='min_sigungu_coverage',
        total_floor=total_floor,
        min_floor=None,
        rounds=int(cfg['gpu_lns'].get('min_sigungu_rounds', 192)),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_dir / 'gpu_lns_min_sigungu.json', _gpu_payload(min_search))

    backend = HighsBackend(require_minimum_version=True)
    hardware = resolve_hardware(cert)
    engine = CertificationEngine(
        project_root=root,
        formulation=formulation,
        base_config=base_cfg,
        cert_config=cert,
        stage41=problem.stage41,
        greedy_indices=min_search.selected_indices,
        backend=backend,
        highs_threads=hardware.highs_threads,
        output_dir=output_dir / 'solve_pass_1',
    )
    engine.stage42h_proof_config = dict(cfg.get('proof_strengthening', {}))
    engine.stage42h_evidence_indices = [
        np.asarray(selected, dtype=int) for selected in min_search.proof_elite_indices
    ]
    first = engine.solve_scenario(
        'equity', efficiency_reference=efficiency_reference, greedy_population=efficiency_reference
    )
    first, witness = _rescue_scenario(
        first, engine=engine, formulation=formulation,
        outdir=output_dir / 'scip_rescue_pass_1', cfg=cfg,
    )
    candidates = [first]
    seed_for_high = witness if witness is not None else np.asarray(first.stages[0].selected_indices, dtype=int)
    min_floor = _stage_retained_min_floor(first)

    high_search = lns.search(
        seed_for_high,
        objective='high_need_population',
        total_floor=total_floor,
        min_floor=min_floor,
        rounds=int(cfg['gpu_lns'].get('high_need_rounds', 320)),
    )
    atomic_write_json(output_dir / 'gpu_lns_high_need.json', _gpu_payload(high_search))

    should_rerun = (
        not first.certified
        or witness is not None
    )
    if should_rerun:
        engine2 = CertificationEngine(
            project_root=root,
            formulation=formulation,
            base_config=base_cfg,
            cert_config=cert,
            stage41=problem.stage41,
            greedy_indices=high_search.selected_indices,
            backend=backend,
            highs_threads=hardware.highs_threads,
            output_dir=output_dir / 'solve_pass_2',
        )
        engine2.stage42h_proof_config = dict(cfg.get('proof_strengthening', {}))
        engine2.stage42h_evidence_indices = [
            np.asarray(selected, dtype=int)
            for selected in [*min_search.proof_elite_indices, *high_search.proof_elite_indices]
        ]
        second = engine2.solve_scenario(
            'equity', efficiency_reference=efficiency_reference, greedy_population=efficiency_reference
        )
        second, witness2 = _rescue_scenario(
            second, engine=engine2, formulation=formulation,
            outdir=output_dir / 'scip_rescue_pass_2', cfg=cfg,
        )
        candidates.append(second)
        if witness2 is not None:
            engine3 = CertificationEngine(
                project_root=root, formulation=formulation, base_config=base_cfg,
                cert_config=cert, stage41=problem.stage41, greedy_indices=witness2,
                backend=backend, highs_threads=hardware.highs_threads,
                output_dir=output_dir / 'solve_pass_3',
            )
            engine3.stage42h_proof_config = dict(cfg.get('proof_strengthening', {}))
            engine3.stage42h_evidence_indices = [
                np.asarray(selected, dtype=int)
                for selected in [*min_search.proof_elite_indices, *high_search.proof_elite_indices]
            ]
            third = engine3.solve_scenario(
                'equity', efficiency_reference=efficiency_reference, greedy_population=efficiency_reference
            )
            third, _ = _rescue_scenario(
                third, engine=engine3, formulation=formulation,
                outdir=output_dir / 'scip_rescue_pass_3', cfg=cfg,
            )
            candidates.append(third)

    best = max(candidates, key=_result_score)
    audit = {
        'candidate_count': int(len(problem.candidates)),
        'pattern_count': int(problem.patterns.n_patterns),
        'coverage_nnz': int(problem.coverage.nnz),
        'efficiency_reference': efficiency_reference,
        'equity_total_population_floor': total_floor,
        'top3_seed_metrics_on_coarse': {
            k: v for k, v in mapped_metrics.items() if not isinstance(v, dict)
        },
        'gpu_backend': lns.backend,
        'solve_pass_count': len(candidates),
        'best_certified': bool(best.certified),
        'best_max_relative_gap': best.max_relative_gap,
    }
    atomic_write_json(output_dir / 'coarse_equity_solver_audit.json', audit)
    return best, formulation, audit
