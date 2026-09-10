from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from mediroad.stage4_2_certification.io_utils import sha256_file
from mediroad.stage4_2g_candidate_expansion.evidence import load_top3_parent


STAGE42E_CURRENT_SHA256 = '769d36d45e49d4d881cea35c6faca738188b71e2401f9ceeb0c877eb1c0b7294'


@dataclass
class Stage42GParent:
    pointer_path: Path
    run_root: Path
    pointer: dict[str, Any]
    quality: dict[str, Any]
    metrics: dict[str, dict[str, dict[str, Any]]]
    plans: dict[str, dict[str, pd.DataFrame]]
    stages: dict[str, dict[str, list[dict[str, Any]]]]


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding='utf-8-sig'))
    if not isinstance(value, dict):
        raise RuntimeError(f'Expected JSON object: {path}')
    return value


def _verify_inventory(project_root: Path, run_root: Path, inventory_path: Path) -> None:
    frame = pd.read_csv(inventory_path, encoding='utf-8-sig')
    required = {'relative_path', 'size_bytes', 'sha256'}
    if not required.issubset(frame.columns):
        raise RuntimeError(f'Invalid inventory schema: {inventory_path}')
    failures: list[str] = []
    for row in frame.itertuples(index=False):
        relative = Path(str(row.relative_path))
        if relative.is_absolute():
            failures.append(f'ABSOLUTE:{row.relative_path}')
            continue
        candidates = (
            run_root / relative,
            project_root / relative,
            inventory_path.parent / relative,
        )
        existing: list[Path] = []
        for candidate in candidates:
            resolved = candidate.resolve()
            try:
                resolved.relative_to(project_root.resolve())
            except ValueError:
                continue
            if resolved.is_file() and resolved not in existing:
                existing.append(resolved)
        if not existing:
            failures.append(f'MISSING:{row.relative_path}')
            continue
        path = existing[0]
        if int(path.stat().st_size) != int(row.size_bytes):
            failures.append(f'SIZE:{row.relative_path}')
        if sha256_file(path) != str(row.sha256):
            failures.append(f'SHA:{row.relative_path}')
    if failures:
        raise RuntimeError(f'Parent inventory mismatch ({len(failures)}): {failures[:10]}')


def load_stage42g_parent(root: Path, cfg: dict[str, Any]) -> Stage42GParent:
    parent_cfg = cfg['parent_stage42g']
    pointer_path = root / 'outputs/model_v1/10_stage4_2g_candidate_expansion/CURRENT_STAGE4_2G_CANDIDATE_EXPANSION_RUN.json'
    if not pointer_path.is_file():
        raise RuntimeError('Authoritative Stage4.2G CURRENT pointer is missing')
    expected_current = str(parent_cfg.get('current_sha256', '')).strip()
    if expected_current and sha256_file(pointer_path) != expected_current:
        raise RuntimeError('Stage4.2G CURRENT SHA differs from the pinned handoff')
    pointer = _read_json(pointer_path)
    if str(pointer.get('run_id')) != str(parent_cfg['run_id']):
        raise RuntimeError(f"Unexpected Stage4.2G run: {pointer.get('run_id')}")
    if not bool(pointer.get('passed')):
        raise RuntimeError('Stage4.2G parent is not passed')
    run_root = (root / str(pointer['run_relative_path'])).resolve()
    run_root.relative_to(root.resolve())
    inventory = run_root / 'ARTIFACT_INVENTORY.csv'
    quality_path = run_root / 'quality_gate.json'
    if sha256_file(inventory) != str(pointer['inventory_sha256']):
        raise RuntimeError('Stage4.2G pointer-bound inventory SHA mismatch')
    if sha256_file(quality_path) != str(pointer['quality_gate_sha256']):
        raise RuntimeError('Stage4.2G pointer-bound quality SHA mismatch')
    expected_inventory = str(parent_cfg.get('inventory_sha256', '')).strip()
    expected_quality = str(parent_cfg.get('quality_gate_sha256', '')).strip()
    if expected_inventory and sha256_file(inventory) != expected_inventory:
        raise RuntimeError('Stage4.2G inventory does not match pinned handoff')
    if expected_quality and sha256_file(quality_path) != expected_quality:
        raise RuntimeError('Stage4.2G quality gate does not match pinned handoff')
    _verify_inventory(root, run_root, inventory)
    quality = _read_json(quality_path)
    if not bool(quality.get('expanded_efficiency_balanced_all_certified')):
        raise RuntimeError('Stage4.2G did not certify all expanded E/B scenarios')
    metrics: dict[str, dict[str, dict[str, Any]]] = {}
    plans: dict[str, dict[str, pd.DataFrame]] = {}
    stages: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for candidate_set in ('top5', 'coarse_pareto'):
        metrics[candidate_set] = {}
        plans[candidate_set] = {}
        stages[candidate_set] = {}
        for scenario in ('efficiency', 'balanced'):
            base = run_root / 'candidate_sets' / candidate_set
            metric_path = base / f'metrics__{scenario}.json'
            plan_path = base / f'plan__{scenario}.csv'
            stage_path = base / f'stages__{scenario}.json'
            metric = _read_json(metric_path)
            if not bool(metric.get('certified')) or float(metric.get('max_relative_gap', 1.0)) > 0.005 + 1e-12:
                raise RuntimeError(f'Uncertified parent scenario: {candidate_set}/{scenario}')
            metrics[candidate_set][scenario] = metric
            plans[candidate_set][scenario] = pd.read_csv(plan_path, encoding='utf-8-sig', low_memory=False)
            stage_value = json.loads(stage_path.read_text(encoding='utf-8-sig'))
            if not isinstance(stage_value, list) or len(stage_value) != 4 or not all(bool(x.get('certified')) for x in stage_value):
                raise RuntimeError(f'Invalid Stage4.2G stage ledger: {candidate_set}/{scenario}')
            stages[candidate_set][scenario] = stage_value
    return Stage42GParent(pointer_path, run_root, pointer, quality, metrics, plans, stages)


def load_top3_equity_seed(root: Path) -> tuple[pd.DataFrame, dict[str, Any], Path]:
    pointer_path = root / 'outputs/model_v1/10_stage4_2e_equity_tail_certification/CURRENT_STAGE4_2E_EQUITY_TAIL_CERTIFICATION_RUN.json'
    if not pointer_path.is_file():
        raise RuntimeError('Top3 Equity full-certification CURRENT is missing')
    if sha256_file(pointer_path) != STAGE42E_CURRENT_SHA256:
        raise RuntimeError('Top3 Equity CURRENT SHA differs from the frozen official parent')
    pointer = _read_json(pointer_path)
    if str(pointer.get('decision')) != 'PASS_STAGE4_2E_TOP3_EQUITY_FULL_CERTIFIED':
        raise RuntimeError('Unexpected Top3 Equity parent decision')
    if str(pointer.get('scope')) != 'TOP3_EQUITY_FULL_ONLY':
        raise RuntimeError('Unexpected Top3 Equity parent scope')
    if not bool(pointer.get('top3_equity_full_lexicographic_certified')):
        raise RuntimeError('Top3 Equity seed parent is not fully certified')
    for key in (
        'candidate_expansion_certified',
        'stage4_full_computational_complete',
        'operational_final',
        'stage5_started',
        'stage5_release_allowed',
    ):
        if bool(pointer.get(key)):
            raise RuntimeError(f'Top3 Equity parent scope flag must remain false: {key}')
    run_root = (root / str(pointer['run_relative_path'])).resolve()
    inventory = (root / str(pointer['inventory_relative_path'])).resolve()
    run_root.relative_to(root.resolve())
    inventory.relative_to(root.resolve())
    if not (run_root / '.COMMITTED').is_file():
        raise RuntimeError('Top3 Equity parent is not committed')
    if sha256_file(inventory) != str(pointer['inventory_sha256']):
        raise RuntimeError('Top3 Equity parent inventory SHA mismatch')
    _verify_inventory(root, run_root, inventory)
    plan = pd.read_csv(run_root / '03_final_incumbent/plan__equity_full_certified.csv', encoding='utf-8-sig', low_memory=False)
    metrics = _read_json(run_root / '03_final_incumbent/metrics.json')
    return plan, metrics, run_root


def load_top3_view(root: Path) -> dict[str, Any]:
    # Reuse the already-hardened C/F evidence loader rather than reconstructing
    # historical Top3 provenance in a second implementation.
    return load_top3_parent(root)
