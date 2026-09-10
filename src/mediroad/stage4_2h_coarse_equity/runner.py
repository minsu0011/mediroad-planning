from __future__ import annotations

import json
import platform
import sys
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mediroad.stage4_2_certification.io_utils import (
    atomic_write_csv,
    atomic_write_json,
    atomic_write_text,
    build_inventory,
    run_id,
    sha256_file,
    utc_now_iso,
    verify_inventory,
)

from mediroad.stage4_2d_equity_certification.runner import _run_regression_tests

from .config import load_yaml, set_thread_environment, thresholds_from_config, validate_config
from .evidence import load_stage42g_parent, load_top3_equity_seed, load_top3_view
from .ladder import CandidateView, evaluate_sequential_ladder
from .solver import solve_coarse_equity
from .version import PACKAGE_NAME, VERSION


DECISION_PASS = 'PASS_STAGE4_2H_COMPUTATIONAL_FINAL_COARSE_PARETO__FIELD_VALIDATION_PENDING'
DECISION_FAIL = 'FAIL_STAGE4_2H_COARSE_EQUITY_OR_LADDER_UNCERTIFIED'


def _stage5_snapshot(root: Path) -> list[str]:
    out: list[str] = []
    for base in (root / 'outputs/model_v1', root / 'reports/model_v1'):
        if not base.exists():
            continue
        out.extend(str(p.relative_to(root).as_posix()) for p in base.iterdir() if 'stage5' in p.name.lower())
        out.extend(str(p.relative_to(root).as_posix()) for p in base.rglob('CURRENT_STAGE5*.json'))
    return sorted(set(out))


def _pointer_snapshot(root: Path) -> dict[str, Any]:
    names = [
        'outputs/model_v1/07_stage3/CURRENT_STAGE3_RUN.json',
        'outputs/model_v1/09_stage4_finalization/CURRENT_STAGE4_FINALIZATION_RUN.json',
        'outputs/model_v1/10_stage4_2/CURRENT_STAGE4_2_RUN.json',
        'outputs/model_v1/10_stage4_2d_equity_certification/CURRENT_STAGE4_2D_EQUITY_CERTIFICATION_RUN.json',
        'outputs/model_v1/10_stage4_2e_equity_tail_certification/CURRENT_STAGE4_2E_EQUITY_TAIL_CERTIFICATION_RUN.json',
        'outputs/model_v1/10_stage4_2f_top3_aggregate/CURRENT_STAGE4_2F_TOP3_AGGREGATE_RUN.json',
        'outputs/model_v1/10_stage4_2g_candidate_expansion/CURRENT_STAGE4_2G_CANDIDATE_EXPANSION_RUN.json',
    ]
    return {n: {'exists': (root / n).exists(), 'sha256': sha256_file(root / n) if (root / n).exists() else None} for n in names}


def _source_snapshot(root: Path) -> list[dict[str, Any]]:
    paths: set[Path] = set()
    for rel in (
        'src/mediroad/stage4_2',
        'src/mediroad/stage4_2_certification',
        'src/mediroad/stage4_2d_equity_certification',
        'src/mediroad/stage4_2e_equity_tail_certification',
        'src/mediroad/stage4_2f_top3_aggregate',
        'src/mediroad/stage4_2g_candidate_expansion',
        'src/mediroad/stage4_2h_coarse_equity',
    ):
        base = root / rel
        if base.is_dir():
            paths.update(base.rglob('*.py'))
    for rel in (
        'configs/model_v1/stage4_2.yaml',
        'configs/model_v1/stage4_2_certification.yaml',
        'configs/model_v1/stage4_2g_candidate_expansion.yaml',
        'configs/model_v1/stage4_2h_coarse_equity.yaml',
        'run_model_v1_stage4_2h_coarse_equity.ps1',
    ):
        p = root / rel
        if p.is_file(): paths.add(p)
    tests = root / 'tests'
    if tests.is_dir(): paths.update(tests.glob('test_*.py'))
    return [
        {'path': p.relative_to(root).as_posix(), 'size': int(p.stat().st_size), 'sha256': sha256_file(p)}
        for p in sorted(paths)
    ]


def _map_top3_view(raw: dict[str, Any]) -> CandidateView:
    return CandidateView(
        'top3', raw['balanced_metrics'], raw['balanced_plan'],
        float(raw['balanced_max_gap']), True,
    )


def _g_views(g: Any) -> dict[str, CandidateView]:
    out: dict[str, CandidateView] = {}
    for name in ('top5', 'coarse_pareto'):
        metric = g.metrics[name]['balanced']
        out[name] = CandidateView(
            name=name,
            metrics=metric,
            plan=g.plans[name]['balanced'],
            max_gap=float(metric['max_relative_gap']),
            certified=bool(metric['certified']),
        )
    return out


def _serialize_stage(stage: Any, source: str) -> dict[str, Any]:
    return {
        'candidate_set': stage.candidate_set,
        'scenario': stage.scenario,
        'stage_index': int(stage.stage_index),
        'objective_name': stage.objective_name,
        'sense': stage.sense,
        'certificate': stage.certificate,
        'certified': bool(stage.certified),
        'objective_value': stage.objective_value,
        'best_bound': stage.best_bound,
        'relative_gap': stage.relative_gap,
        'wall_time_sec': stage.wall_time_sec,
        'mip_node_count': stage.mip_node_count,
        'seed_source': stage.seed_source,
        'source': source,
    }


def _parent_stage_rows(g: Any, candidate_set: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scenario in ('efficiency', 'balanced'):
        for raw in g.stages[candidate_set][scenario]:
            rows.append({
                'candidate_set': candidate_set,
                'scenario': scenario,
                'stage_index': int(raw['stage_index']),
                'objective_name': str(raw['objective_name']),
                'sense': str(raw['sense']),
                'certificate': str(raw['certificate']),
                'certified': bool(raw['certified']),
                'objective_value': raw.get('objective_value'),
                'best_bound': raw.get('best_bound'),
                'relative_gap': raw.get('relative_gap'),
                'wall_time_sec': raw.get('wall_time_sec'),
                'mip_node_count': raw.get('mip_node_count'),
                'seed_source': raw.get('seed_source'),
                'source': 'STAGE4_2G_CERTIFIED_PARENT',
            })
    return rows


def run(project_root: Path, config_path: Path) -> dict[str, Any]:
    root = project_root.resolve()
    cfg = load_yaml(config_path.resolve() if config_path.is_absolute() else (root / config_path).resolve())
    validate_config(cfg); set_thread_environment()
    output_root = root / str(cfg['paths']['output_root'])
    report_root = root / str(cfg['paths']['report_root'])
    lock = root / str(cfg['paths']['lock'])
    if lock.exists():
        raise RuntimeError(f'Stage4.2H writer lock exists: {lock}')
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text(f'pid={__import__("os").getpid()}\n', encoding='utf-8')
    ident = run_id('stage4_2h_coarse_equity')
    run_root = output_root / 'runs' / ident
    report_root_run = report_root / 'runs' / ident
    run_root.mkdir(parents=True, exist_ok=False); report_root_run.mkdir(parents=True, exist_ok=False)
    (run_root / '.RUNNING').write_text(utc_now_iso() + '\n', encoding='utf-8')
    before_ptr = _pointer_snapshot(root); before_s5 = _stage5_snapshot(root); before_src = _source_snapshot(root)
    atomic_write_json(run_root / 'parent_pointers_before.json', before_ptr)
    atomic_write_json(run_root / 'stage5_namespace_before.json', before_s5)
    atomic_write_csv(run_root / 'tested_source_before.csv', pd.DataFrame(before_src))
    metadata = {
        'run_id': ident, 'started_utc': utc_now_iso(), 'package': PACKAGE_NAME, 'version': VERSION,
        'python': sys.version, 'platform': platform.platform(), 'hardware_profile': cfg['hardware'],
    }
    atomic_write_json(run_root / 'metadata.json', metadata)
    try:
        regression = _run_regression_tests(root, run_root / '00_regression')
        metadata['regression'] = regression
        atomic_write_json(run_root / 'metadata.json', metadata)
        g = load_stage42g_parent(root, cfg)
        top3 = load_top3_view(root)
        top3_equity_plan, top3_equity_metrics, top3_equity_root = load_top3_equity_seed(root)
        views = {'top3': _map_top3_view(top3), **_g_views(g)}
        thresholds = thresholds_from_config(cfg)
        comparisons, transitions, ladder = evaluate_sequential_ladder(
            views, ladder=list(cfg['ladder']['promotion_ladder']), thresholds=thresholds,
        )
        atomic_write_csv(run_root / 'candidate_ladder_comparisons.csv', comparisons)
        atomic_write_csv(run_root / 'candidate_ladder_transitions.csv', transitions)
        atomic_write_json(run_root / 'candidate_ladder_decision.json', ladder)
        if not ladder['passed'] or ladder['recommended_candidate_set'] != str(cfg['ladder']['expected_final_candidate_set']):
            raise RuntimeError(f'Candidate ladder did not resolve to the pinned final rung: {ladder}')

        equity, formulation, solver_audit = solve_coarse_equity(
            project_root=root, cfg=cfg,
            g_efficiency_metrics=g.metrics['coarse_pareto']['efficiency'],
            top3_equity_plan=top3_equity_plan,
            output_dir=run_root / 'coarse_equity',
        )
        atomic_write_csv(run_root / 'coarse_equity/plan__equity.csv', equity.selected_frame)
        atomic_write_json(run_root / 'coarse_equity/metrics__equity.json', equity.metrics)
        atomic_write_json(run_root / 'coarse_equity/stages__equity.json', [
            {**_serialize_stage(s, 'STAGE4_2H_FRESH_COARSE_EQUITY'), 'selected_indices': [int(x) for x in s.selected_indices] if s.selected_indices is not None else None}
            for s in equity.stages
        ])

        stage_rows = _parent_stage_rows(g, 'coarse_pareto') + [_serialize_stage(s, 'STAGE4_2H_FRESH_COARSE_EQUITY') for s in equity.stages]
        stage_frame = pd.DataFrame(stage_rows)
        atomic_write_csv(run_root / 'stage4_computational_stage_certification.csv', stage_frame)
        scenario_rows = [
            {'candidate_set': 'coarse_pareto', 'scenario': 'efficiency', **g.metrics['coarse_pareto']['efficiency'], 'source': 'STAGE4_2G_CERTIFIED_PARENT'},
            {'candidate_set': 'coarse_pareto', 'scenario': 'balanced', **g.metrics['coarse_pareto']['balanced'], 'source': 'STAGE4_2G_CERTIFIED_PARENT'},
            {'candidate_set': 'coarse_pareto', 'scenario': 'equity', **equity.metrics, 'certified': equity.certified, 'certification_class': equity.certification_class, 'max_relative_gap': equity.max_relative_gap, 'source': 'STAGE4_2H_FRESH_COARSE_EQUITY'},
        ]
        scenario_frame = pd.DataFrame(scenario_rows)
        atomic_write_csv(run_root / 'stage4_computational_scenario_metrics.csv', scenario_frame)

        all_12 = len(stage_frame) == 12 and bool(stage_frame['certified'].all()) and bool((pd.to_numeric(stage_frame['relative_gap'], errors='coerce').fillna(1.0) <= 0.005 + 1e-12).all())
        coarse_eb = bool(g.metrics['coarse_pareto']['efficiency']['certified'] and g.metrics['coarse_pareto']['balanced']['certified'])
        equity_ok = bool(equity.certified and len(equity.stages) == 4)
        after_ptr = _pointer_snapshot(root); after_s5 = _stage5_snapshot(root); after_src = _source_snapshot(root)
        atomic_write_json(run_root / 'parent_pointers_after.json', after_ptr)
        atomic_write_json(run_root / 'stage5_namespace_after.json', after_s5)
        atomic_write_csv(run_root / 'tested_source_after.csv', pd.DataFrame(after_src))
        checks = {
            'parent_pointers_unchanged': before_ptr == after_ptr,
            'stage5_namespace_unchanged': before_s5 == after_s5,
            'tested_source_unchanged': before_src == after_src,
            'threshold_fingerprint_exact': thresholds.fingerprint == str(cfg['ladder']['threshold_fingerprint']),
            'g_parent_inventory_verified': True,
            'sequential_ladder_finalized': bool(ladder['final_rung_reached']),
            'selected_candidate_set_is_coarse_pareto': ladder['recommended_candidate_set'] == 'coarse_pareto',
            'coarse_efficiency_balanced_certified': coarse_eb,
            'coarse_equity_four_stages_certified': equity_ok,
            'all_12_required_stages_certified': all_12,
        }
        passed = bool(all(checks.values()))
        decision = DECISION_PASS if passed else DECISION_FAIL
        quality = {
            'decision': decision,
            'passed': passed,
            'selected_candidate_set': 'coarse_pareto' if passed else ladder.get('recommended_candidate_set'),
            'candidate_ladder_finalized': bool(ladder.get('final_rung_reached')),
            'candidate_reduction_top3_certified': False,
            'coarse_pareto_efficiency_balanced_certified': coarse_eb,
            'coarse_pareto_equity_certified': equity_ok,
            'stage4_required_scenarios_certified': all_12,
            'stage4_computational_final': passed,
            'field_validation_pending': True,
            'operational_final': False,
            'stage5_started': False,
            'stage5_release_allowed': False,
            'checks': checks,
        }
        atomic_write_json(run_root / 'quality_gate.json', quality)
        metadata.update({'completed_utc': utc_now_iso(), 'quality_gate': quality, 'parent_stage42g_run_id': g.pointer['run_id'], 'top3_equity_seed_run': str(top3_equity_root), 'solver_audit': solver_audit})
        atomic_write_json(run_root / 'metadata.json', metadata)
        report_lines = [
            '# MEDIROAD Stage 4.2H — Sequential Candidate Ladder + Coarse Equity Certification', '',
            f'- Run: `{ident}`', f'- Decision: `{decision}`',
            f"- Selected candidate universe: `{quality['selected_candidate_set']}`", '',
            '## Sequential ladder', '', comparisons.to_markdown(index=False), '', transitions.to_markdown(index=False), '',
            '## Coarse-Pareto required scenarios', '', scenario_frame.to_markdown(index=False), '',
            'Top3/Top5/coarse candidate thresholds were not changed. Stage4.2G E/B proof is reused only after pointer-bound inventory verification. Coarse Equity is solved fresh in this run.', '',
            'This is a computational-final decision only. Physical venue validation, confirmed team/vehicle/calendar/travel inputs, Operational Final, and Stage5 remain blocked.', '',
        ]
        text = '\n'.join(report_lines)
        atomic_write_text(run_root / 'FINAL_REPORT.md', text); atomic_write_text(report_root_run / 'FINAL_REPORT.md', text)
        (run_root / '.RUNNING').unlink(missing_ok=True)
        atomic_write_text(run_root / ('.COMMITTED' if passed else '.FAILED'), utc_now_iso() + '\n')
        inv = build_inventory(run_root, exclude={'ARTIFACT_INVENTORY.csv'})
        atomic_write_csv(run_root / 'ARTIFACT_INVENTORY.csv', inv); verify_inventory(run_root, inv)
        pointer = {
            'run_id': ident, 'decision': decision, 'passed': passed,
            'run_relative_path': run_root.relative_to(root).as_posix(),
            'quality_gate_sha256': sha256_file(run_root / 'quality_gate.json'),
            'inventory_sha256': sha256_file(run_root / 'ARTIFACT_INVENTORY.csv'),
            'selected_candidate_set': quality['selected_candidate_set'],
            'stage4_computational_final': passed,
            'operational_final': False, 'stage5_started': False,
        }
        if passed:
            output_root.mkdir(parents=True, exist_ok=True)
            atomic_write_json(output_root / 'CURRENT_STAGE4_2H_COMPUTATIONAL_FINAL_RUN.json', pointer)
        else:
            atomic_write_json(output_root / f'FAILED_STAGE4_2H__{ident}.json', pointer)
        return {'run_root': str(run_root), 'decision': decision, 'passed': passed, 'quality_gate': quality}
    except BaseException as exc:
        (run_root / '.RUNNING').unlink(missing_ok=True)
        atomic_write_text(run_root / '.FAILED', utc_now_iso() + '\n')
        atomic_write_json(run_root / 'UNHANDLED_EXCEPTION.json', {'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()})
        raise
    finally:
        try: lock.unlink()
        except FileNotFoundError: pass
