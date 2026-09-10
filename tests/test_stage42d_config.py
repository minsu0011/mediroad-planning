from __future__ import annotations

import copy
import math
from types import SimpleNamespace
from pathlib import Path

import pytest

from mediroad.stage4_2d_equity_certification.config import (
    load_yaml,
    validate_config,
    validate_parent_contracts,
)
from mediroad.stage4_2d_equity_certification.runner import (
    _run_regression_tests,
    _source_snapshot,
    _stage5_snapshot,
)


def test_regression_preflight_records_nonempty_streams_and_count(tmp_path: Path, monkeypatch):
    def fake_run(*args, **kwargs):
        assert kwargs["cwd"] == tmp_path
        assert kwargs["env"]["PYTHONDONTWRITEBYTECODE"] == "1"
        return SimpleNamespace(returncode=0, stdout="412 passed in 1.23s\n", stderr="")

    monkeypatch.setattr("mediroad.stage4_2d_equity_certification.runner.subprocess.run", fake_run)
    result = _run_regression_tests(tmp_path, tmp_path / "audit")
    assert result["passed_count"] == 412
    assert result["failed_count"] == 0
    assert (tmp_path / "audit/regression_test_stderr.txt").read_text(encoding="utf-8") == (
        "NO_STDERR_OUTPUT\n"
    )


def test_regression_preflight_fails_closed(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        "mediroad.stage4_2d_equity_certification.runner.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=1,
            stdout="1 failed, 411 passed in 1.23s\n",
            stderr="failure\n",
        ),
    )
    with pytest.raises(RuntimeError, match="regression suite failed"):
        _run_regression_tests(tmp_path, tmp_path / "audit")


def test_stage5_snapshot_recurses_but_allows_frozen_stage4_evidence(tmp_path: Path):
    allowed = tmp_path / "outputs/model_v1/08_stage4/runs/frozen/05_interface"
    allowed.mkdir(parents=True)
    (allowed / "stage4_to_stage5_interface.csv").write_text("frozen\n", encoding="utf-8")
    (allowed / "stage5_readiness_checklist.json").write_text("{}\n", encoding="utf-8")
    # A readiness checklist is allowed only at its Stage 4.1 contractual path,
    # so placing it in the Stage 4 handoff directory must be detected.
    misplaced = _stage5_snapshot(tmp_path)
    assert any(row["relative_path"].endswith("stage5_readiness_checklist.json") for row in misplaced["records"])
    (allowed / "stage5_readiness_checklist.json").unlink()
    assert _stage5_snapshot(tmp_path) == {"records": [], "count": 0}

    readiness = tmp_path / "outputs/model_v1/09_stage4_finalization/runs/frozen/07_interface"
    readiness.mkdir(parents=True)
    (readiness / "stage5_readiness_checklist.json").write_text("{}\n", encoding="utf-8")
    assert _stage5_snapshot(tmp_path) == {"records": [], "count": 0}

    forbidden = tmp_path / "outputs/model_v1/archive/nested/10_stage5/runs/r1"
    forbidden.mkdir(parents=True)
    payload = forbidden / "result.json"
    payload.write_text('{"started": true}\n', encoding="utf-8")
    snapshot = _stage5_snapshot(tmp_path)
    relative_paths = {row["relative_path"] for row in snapshot["records"]}
    assert "outputs/model_v1/archive/nested/10_stage5" in relative_paths
    assert "outputs/model_v1/archive/nested/10_stage5/runs/r1/result.json" in relative_paths
    file_row = next(row for row in snapshot["records"] if row["relative_path"].endswith("result.json"))
    assert file_row["size_bytes"] > 0
    assert len(file_row["sha256"]) == 64


def test_source_snapshot_seals_all_regression_tests_and_stage42_environments(tmp_path: Path):
    tests = tmp_path / "tests"
    tests.mkdir(parents=True)
    for name in ("test_stage42d_config.py", "test_stage4_2c_formulation.py", "test_unrelated.py"):
        (tests / name).write_text("def test_ok(): pass\n", encoding="utf-8")
    (tmp_path / "requirements_stage4_2_certification.txt").write_text("highspy==1.15.1\n", encoding="utf-8")
    (tmp_path / "environment_stage4_2.yml").write_text("name: frozen\n", encoding="utf-8")
    rows = _source_snapshot(tmp_path, [])
    sealed = {row["relative_path"] for row in rows}
    assert {
        "tests/test_stage42d_config.py",
        "tests/test_stage4_2c_formulation.py",
        "tests/test_unrelated.py",
        "requirements_stage4_2_certification.txt",
        "environment_stage4_2.yml",
    }.issubset(sealed)


def test_official_config_preserves_frozen_gap():
    path = Path(__file__).parents[1] / "configs/model_v1/stage4_2d_equity_certification.yaml"
    config = load_yaml(path)
    validate_config(config)
    bad = copy.deepcopy(config)
    bad["gate"]["near_optimal_relative_gap_max"] = 0.01
    with pytest.raises(ValueError):
        validate_config(bad)


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("contract", "min_sigungu_retention", 0.999),
        ("contract", "high_need_retention", 0.98),
        ("contract", "preserve_candidate_universe", False),
        ("contract", "preserve_retention_rates", False),
        ("contract", "candidate_expansion_enabled", True),
        ("gate", "require_min_sigungu_certification", False),
        ("gate", "require_high_need_certification", False),
        ("gate", "write_current_only_on_full_front_stage_pass", False),
    ],
)
def test_frozen_contract_flags_cannot_be_relaxed(section, key, value):
    path = Path(__file__).parents[1] / "configs/model_v1/stage4_2d_equity_certification.yaml"
    config = load_yaml(path)
    config[section][key] = value
    with pytest.raises(ValueError):
        validate_config(config)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("enabled", False),
        ("random_seed", 1051),
        ("heuristic_effort", 0.08),
        ("use_objective_target", True),
        ("include_local_exclusion_cuts", True),
    ],
)
def test_verified_stage42c_min_oracle_replay_cannot_drift(key, value):
    path = Path(__file__).parents[1] / "configs/model_v1/stage4_2d_equity_certification.yaml"
    config = load_yaml(path)
    config["solver"]["stage42c_oracle_replay"]["min_sigungu_coverage"][key] = value
    with pytest.raises(ValueError):
        validate_config(config)


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("root", "mode", None),
        ("root", "stage5_started", None),
        ("root", "stage5_started", "false"),
        ("contract", "visit_count", None),
        ("contract", "preserve_stage42c_objectives", "true"),
        ("contract", "candidate_expansion_enabled", None),
        ("gate", "require_high_need_certification", "true"),
        ("evidence", "preferred_stage42c_run_id", "different"),
    ],
)
def test_missing_or_coercive_frozen_values_are_rejected(section, key, value):
    path = Path(__file__).parents[1] / "configs/model_v1/stage4_2d_equity_certification.yaml"
    config = load_yaml(path)
    target = config if section == "root" else config[section]
    if value is None:
        target.pop(key, None)
    else:
        target[key] = value
    with pytest.raises(ValueError):
        validate_config(config)


def test_parent_contract_exact_semantics_are_frozen():
    root = Path(__file__).parents[1]
    stage42 = load_yaml(root / "configs/model_v1/stage4_2.yaml")
    stage42c = load_yaml(root / "configs/model_v1/stage4_2_certification.yaml")
    validate_parent_contracts(stage42, stage42c)
    stage42["optimization"]["objective_retention"]["min_sigungu_coverage"] = 0.99
    with pytest.raises(ValueError, match="parent contract mismatch"):
        validate_parent_contracts(stage42, stage42c)


def test_preferred_inventory_and_improving_solution_pins_are_exact():
    path = Path(__file__).parents[1] / "configs/model_v1/stage4_2d_equity_certification.yaml"
    config = load_yaml(path)
    config["evidence"]["preferred_artifact_sha256"]["ARTIFACT_INVENTORY.csv"] = "0" * 64
    with pytest.raises(ValueError, match="trust anchor"):
        validate_config(config)

    config = load_yaml(path)
    config["evidence"]["improving_solution_artifacts"] = {}
    with pytest.raises(ValueError, match="interrupted-solver evidence"):
        validate_config(config)


def test_numeric_strings_do_not_satisfy_frozen_numeric_contract():
    path = Path(__file__).parents[1] / "configs/model_v1/stage4_2d_equity_certification.yaml"
    config = load_yaml(path)
    config["gate"]["near_optimal_relative_gap_max"] = "0.005"
    with pytest.raises(ValueError):
        validate_config(config)


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("root", "enabled", False),
        ("root", "accept_diagnostic_artifacts", True),
        ("root", "candidate_count", "452"),
        ("references", "min_sigungu_strict_threshold", 0.5278),
        ("references", "high_need_strict_threshold", 11660.0),
        ("references", "anchor_seed_count", "26"),
        ("min", "threads", 4),
        ("min", "time_limit_sec", "900"),
        ("min", "mip_allow_restart", "true"),
        ("min", "expected_exact_model_sha256", "0" * 64),
        ("min", "expected_universe_sha256", "0" * 64),
        ("high", "outward_safe_cuts", False),
        ("high", "portfolio_seeds", [42]),
        ("high", "budget_strategy", "retry_only_three_seeds"),
        ("high", "time_limit_sec_per_seed", 300),
        ("high", "memory_limit_mib_per_worker", "2800"),
        ("high", "expected_cut_sha256", "0" * 64),
        ("high", "expected_mps_sha256", "0" * 64),
    ],
)
def test_formal_oracle_contract_cannot_drift_or_coerce(section, key, value):
    path = Path(__file__).parents[1] / "configs/model_v1/stage4_2d_equity_certification.yaml"
    config = load_yaml(path)
    oracle = config["oracle_certification"]
    target = {
        "root": oracle,
        "references": oracle["frozen_references"],
        "min": oracle["min_highs"],
        "high": oracle["high_scip"],
    }[section]
    target[key] = value
    with pytest.raises(ValueError):
        validate_config(config)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("solver_threads", 4),
        ("portfolio_workers", 2),
        ("threads_per_portfolio_worker", 4),
    ],
)
def test_hardware_profile_matches_frozen_oracle_parallelism(key, value):
    path = Path(__file__).parents[1] / "configs/model_v1/stage4_2d_equity_certification.yaml"
    config = load_yaml(path)
    config["hardware"][key] = value
    with pytest.raises(ValueError):
        validate_config(config)


def test_formal_oracle_sections_and_keys_are_mandatory():
    path = Path(__file__).parents[1] / "configs/model_v1/stage4_2d_equity_certification.yaml"
    for mutation in ("whole", "reference_key", "min_key", "high_key"):
        config = load_yaml(path)
        if mutation == "whole":
            config.pop("oracle_certification")
        elif mutation == "reference_key":
            config["oracle_certification"]["frozen_references"].pop("min_selection_sha256")
        elif mutation == "min_key":
            config["oracle_certification"]["min_highs"].pop("random_seed")
        else:
            config["oracle_certification"]["high_scip"].pop("scip_version")
        with pytest.raises(ValueError):
            validate_config(config)


def test_official_and_diagnostic_configs_share_frozen_oracle_contract():
    config_dir = Path(__file__).parents[1] / "configs/model_v1"
    official = load_yaml(config_dir / "stage4_2d_equity_certification.yaml")
    diagnostic = load_yaml(
        config_dir / "stage4_2d_equity_certification_diagnostic.yaml"
    )
    validate_config(official)
    validate_config(diagnostic)
    assert diagnostic["oracle_certification"] == official["oracle_certification"]


@pytest.mark.parametrize(
    ("section", "key"),
    [
        ("gate", "near_optimal_relative_gap_max"),
        ("contract", "min_sigungu_retention"),
        ("contract", "high_need_retention"),
        ("references", "min_sigungu_strict_threshold"),
        ("references", "high_need_strict_threshold"),
    ],
)
def test_frozen_float_contract_rejects_one_ulp_drift(section, key):
    path = Path(__file__).parents[1] / "configs/model_v1/stage4_2d_equity_certification.yaml"
    config = load_yaml(path)
    if section == "references":
        target = config["oracle_certification"]["frozen_references"]
    else:
        target = config[section]
    target[key] = math.nextafter(float(target[key]), math.inf)
    with pytest.raises(ValueError):
        validate_config(config)
