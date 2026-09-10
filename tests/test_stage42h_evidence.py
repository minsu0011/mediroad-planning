from __future__ import annotations

from pathlib import Path

from mediroad.stage4_2h_coarse_equity.config import load_yaml
from mediroad.stage4_2h_coarse_equity.evidence import (
    load_stage42g_parent,
    load_top3_equity_seed,
)


ROOT = Path(__file__).resolve().parents[1]


def test_official_stage42g_parent_inventory_and_expanded_scenarios_load() -> None:
    cfg = load_yaml(ROOT / 'configs/model_v1/stage4_2h_coarse_equity.yaml')
    parent = load_stage42g_parent(ROOT, cfg)
    assert parent.pointer['run_id'] == (
        'stage4_2g_candidate_expansion_20260821T162221Z_55d8276df85e'
    )
    assert len(parent.plans['coarse_pareto']['efficiency']) == 20
    assert len(parent.plans['coarse_pareto']['balanced']) == 20


def test_official_stage42e_project_relative_inventory_and_seed_load() -> None:
    plan, metrics, run_root = load_top3_equity_seed(ROOT)
    assert run_root.name == 'stage4_2e_equity_tail_20260821T110943Z_e1352e4b7adc'
    assert len(plan) == 20
    assert float(metrics['min_sigungu_coverage_ratio']) > 0.5
