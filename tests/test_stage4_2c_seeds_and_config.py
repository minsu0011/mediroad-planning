from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml

from mediroad.stage4_2_certification.synthetic import make_synthetic_formulation
from mediroad.stage4_2_certification import config as config_module
from mediroad.stage4_2_certification.adapter import load_base_stage42_config
from mediroad.stage4_2_certification.config import resolve_hardware, validate_certification_config
from mediroad.stage4_2_certification.seeds import SingleSwapImprover
from mediroad.stage4_2_certification.types import Floor
from mediroad.stage4_2_certification.runner import _candidate_expansion_prerequisite


def _config():
    return {
        "version": "test",
        "hardware": {"highs_threads": 8},
        "solver": {},
        "formulation": {
            "one_link_pattern_formulation": True,
            "integer_count_deviation_convex_hull": True,
        },
        "gate": {"near_optimal_relative_gap_max": 0.005},
        "paths": {},
    }

@pytest.fixture
def synthetic_formulation():
    return make_synthetic_formulation()


def test_frozen_gap_cannot_be_relaxed():
    cfg = _config()
    validate_certification_config(cfg)
    bad = deepcopy(cfg)
    bad["gate"]["near_optimal_relative_gap_max"] = 0.01
    with pytest.raises(ValueError, match="must remain 0.005"):
        validate_certification_config(bad)


def test_stage42_loader_never_swallows_contract_validation_error(tmp_path: Path):
    source = Path(__file__).parents[1] / "configs/model_v1/stage4_2.yaml"
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    config["solver"]["min_sigungu_ratio_integer_scale"] = 999
    target = tmp_path / "stage4_2.yaml"
    target.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    with pytest.raises(Exception):
        load_base_stage42_config(target)


def test_single_swap_improves_or_preserves(synthetic_formulation):
    f = synthetic_formulation
    start = np.array([0, 2])
    search = SingleSwapImprover(f, allow_gpu=False)
    result = search.improve(
        start,
        objective_name="total_population",
        sense="max",
        floors=[],
        max_rounds=10,
    )
    assert result.objective_value >= f.objective_for_selection("total_population", start)
    valid, reason = f.validate_selection(result.selected_indices)
    assert valid, reason


def test_single_swap_honors_floor(synthetic_formulation):
    f = synthetic_formulation
    start = np.array([1, 3])
    floor = Floor("min_sigungu_coverage", "max", 0.30)
    search = SingleSwapImprover(f, allow_gpu=False)
    result = search.improve(
        start,
        objective_name="cost",
        sense="min",
        floors=[floor],
        max_rounds=10,
    )
    assert f.objective_for_selection("min_sigungu_coverage", result.selected_indices) >= 0.30 - 1e-7


def test_native_windows_parallel_cap(monkeypatch):
    monkeypatch.setattr(config_module.platform, "system", lambda: "Windows")
    monkeypatch.setattr(config_module, "is_wsl", lambda: False)
    monkeypatch.setattr(config_module.os, "cpu_count", lambda: 16)
    profile = resolve_hardware(
        {
            "hardware": {
                "highs_threads": 8,
                "allow_native_windows_parallel": True,
                "native_windows_highs_threads": 4,
            }
        }
    )
    assert profile.native_windows
    assert profile.highs_threads == 4


def test_candidate_expansion_fails_fast_when_required_equity_is_uncertified():
    from types import SimpleNamespace

    results = {
        "efficiency": SimpleNamespace(certified=True),
        "balanced": SimpleNamespace(certified=True),
        "equity": SimpleNamespace(certified=False),
    }
    config = {
        "gate": {
            "require_equity_certification": True,
            "fail_fast_after_required_top3_uncertified": True,
        }
    }
    assert not _candidate_expansion_prerequisite(results, config)
    config["gate"]["fail_fast_after_required_top3_uncertified"] = False
    assert _candidate_expansion_prerequisite(results, config)
