from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

import pytest

from mediroad.stage4_2e_equity_tail_certification.config import (
    FROZEN_HIGH_NEED_FLOOR,
    FROZEN_HIGH_NEED_INCUMBENT,
    FROZEN_HIGH_NEED_RETENTION,
    load_yaml,
    validate_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/model_v1/stage4_2e_equity_tail_certification.yaml"


def _config() -> dict[str, Any]:
    return load_yaml(CONFIG_PATH)


def _set(cfg: dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    cursor: dict[str, Any] = cfg
    for part in parts[:-1]:
        cursor = cursor[part]
    cursor[parts[-1]] = value


def test_official_config_validates_and_high_need_floor_is_exact_binary64() -> None:
    cfg = _config()

    validate_config(cfg)

    configured = float(cfg["contract"]["high_need_floor"])
    assert configured.hex() == FROZEN_HIGH_NEED_FLOOR.hex()
    assert configured.hex() == (
        FROZEN_HIGH_NEED_INCUMBENT * FROZEN_HIGH_NEED_RETENTION
    ).hex()


def test_load_yaml_requires_a_mapping(tmp_path: Path) -> None:
    sequence = tmp_path / "sequence.yaml"
    sequence.write_text("- one\n- two\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Expected YAML mapping"):
        load_yaml(sequence)


@pytest.mark.parametrize(
    ("dotted", "value"),
    [
        ("mode", "diagnostic"),
        ("stage5_started", True),
        ("hardware.solver_threads", 7),
        ("contract.candidate_set", "top2"),
        ("contract.visit_count", 20.0),
        ("contract.candidate_count", 451),
        ("contract.pattern_count", 5241),
        ("contract.preserve_stage42c_objectives", 1),
        ("contract.preserve_stage42c_constraints", False),
        ("contract.preserve_candidate_universe", False),
        ("contract.candidate_expansion_enabled", True),
        ("contract.relative_gap", math.nextafter(0.005, math.inf)),
        (
            "contract.total_population_floor",
            math.nextafter(167282.84413449097, math.inf),
        ),
        ("contract.min_sigungu_floor", math.nextafter(0.525217, math.inf)),
        (
            "contract.high_need_incumbent",
            math.nextafter(11609.13711010601, math.inf),
        ),
        ("contract.high_need_retention", math.nextafter(0.99, math.inf)),
        (
            "contract.high_need_floor",
            math.nextafter(FROZEN_HIGH_NEED_FLOOR, math.inf),
        ),
        ("contract.need_weighted_retention", math.nextafter(0.99, math.inf)),
        ("stage42d_parent.run_id", "different-d-run"),
        ("stage42c_bound_evidence.run_id", "different-c-run"),
        (
            "stage42c_bound_evidence.need_weighted_best_bound",
            math.nextafter(104752.01728095711, math.inf),
        ),
        ("solver.backend", "scipy_milp"),
        ("solver.required_highspy_version", "1.14.0"),
        ("solver.threads", 4),
        ("solver.parallel", 1),
        ("solver.fresh_need_solve_required", False),
        ("solver.fresh_cost_solve_required", False),
        ("solver.allow_threshold_relaxation", True),
        ("solver.allow_objective_change", True),
        ("solver.allow_candidate_change", True),
        ("solver.allow_inherited_bounds_for_certification", True),
        ("solver.relative_gap", math.nextafter(0.005, math.inf)),
        ("solver.time_limit_sec.need_weighted", 0),
        ("solver.random_seed.cost", -1),
        ("gate.require_subset_and_sha_for_inherited_bounds", False),
        ("gate.require_fresh_incumbent_and_bound_each_stage", False),
        ("gate.historical_bounds_diagnostic_only", False),
        ("gate.require_need_weighted_certification", False),
        ("gate.promotable_scope", "STAGE4_FULL"),
        ("gate.candidate_expansion_certified", True),
        ("gate.stage4_full_computational_complete", True),
        ("gate.operational_final", True),
        ("gate.stage5_release_allowed", True),
    ],
)
def test_frozen_config_mutations_fail_closed(dotted: str, value: Any) -> None:
    cfg = copy.deepcopy(_config())
    _set(cfg, dotted, value)

    with pytest.raises(ValueError):
        validate_config(cfg)


def test_missing_required_section_fails_closed() -> None:
    cfg = copy.deepcopy(_config())
    del cfg["gate"]

    with pytest.raises(ValueError, match="must contain exactly|missing sections"):
        validate_config(cfg)


@pytest.mark.parametrize("section", ["stage42d_parent", "stage42c_bound_evidence"])
def test_artifact_pin_manifest_rejects_missing_extra_or_noncanonical_sha(
    section: str,
) -> None:
    base = _config()

    missing = copy.deepcopy(base)
    pins = missing[section]["artifact_sha256"]
    del pins[next(iter(pins))]
    with pytest.raises(ValueError, match="must pin exactly"):
        validate_config(missing)

    extra = copy.deepcopy(base)
    extra[section]["artifact_sha256"]["unexpected.json"] = "0" * 64
    with pytest.raises(ValueError, match="must pin exactly"):
        validate_config(extra)

    uppercase = copy.deepcopy(base)
    pins = uppercase[section]["artifact_sha256"]
    first = next(iter(pins))
    pins[first] = "A" * 64
    with pytest.raises(ValueError, match="Invalid SHA-256"):
        validate_config(uppercase)


def test_solver_time_limit_and_seed_key_sets_are_exact() -> None:
    cfg = copy.deepcopy(_config())
    cfg["solver"]["time_limit_sec"]["extra"] = 1
    with pytest.raises(ValueError, match="contain only"):
        validate_config(cfg)

    cfg = copy.deepcopy(_config())
    cfg["solver"]["random_seed"]["extra"] = 1
    with pytest.raises(ValueError, match="contain only"):
        validate_config(cfg)
