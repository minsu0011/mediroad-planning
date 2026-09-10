from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

from mediroad.stage4_2_certification.adapter import (
    load_base_stage42_config,
    prepare_candidate_problem,
)
from mediroad.stage4_2_certification.config import validate_certification_config
from mediroad.stage4_2_certification.formulation import CertificationFormulation
from mediroad.stage4_2d_equity_certification.config import validate_parent_contracts
from mediroad.stage4_2d_equity_certification.io_utils import sha256_file
from mediroad.stage4_2f_top3_aggregate.audit import compatibility_adoption_audit
from mediroad.stage4_2f_top3_aggregate.config import (
    EXPECTED_ADOPTION_STAGES,
    FROZEN_C_INVENTORY_SHA256,
    FROZEN_D_POINTER_SHA256,
    PendingStage42E,
    load_yaml,
    validate_config,
)
from mediroad.stage4_2f_top3_aggregate.runner import run_top3_semantic_aggregate
from mediroad.stage4_2f_top3_aggregate.parents import _stage, load_parent_evidence


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/model_v1/stage4_2f_top3_aggregate.yaml"


def _future_ready_config() -> dict:
    cfg = copy.deepcopy(load_yaml(CONFIG))
    cfg["stage42e_parent"]["run_id"] = "stage4_2e_equity_tail_20990101T000000Z_deadbeef0000"
    for index, key in enumerate(cfg["stage42e_parent"]["artifact_sha256"]):
        cfg["stage42e_parent"]["artifact_sha256"][key] = f"{index + 1:064x}"
    return cfg


def _formulation() -> CertificationFormulation:
    base = load_base_stage42_config(ROOT / "configs/model_v1/stage4_2.yaml")
    cert = yaml.safe_load(
        (ROOT / "configs/model_v1/stage4_2_certification.yaml").read_text(
            encoding="utf-8"
        )
    )
    validate_certification_config(cert)
    validate_parent_contracts(base, cert)
    problem = prepare_candidate_problem(ROOT, "top3", base, cache={})
    formulation = CertificationFormulation(
        problem.candidates,
        problem.patterns,
        base,
        cert,
        candidate_set="top3",
    )
    formulation.venue_alias_map = dict(problem.venue_alias_map)
    return formulation


def test_checked_in_config_is_officially_pinned_and_valid() -> None:
    cfg = load_yaml(CONFIG)
    validate_config(cfg)
    assert cfg["stage42e_parent"]["run_id"] == (
        "stage4_2e_equity_tail_20260821T110943Z_e1352e4b7adc"
    )
    assert all(
        len(value) == 64
        for value in cfg["stage42e_parent"]["artifact_sha256"].values()
    )


def test_future_exact_e_pins_make_template_valid_but_stage_pin_drift_fails() -> None:
    cfg = _future_ready_config()
    validate_config(cfg)
    cfg["adoption_stage_pins"]["balanced:4"]["selection_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="eight-stage"):
        validate_config(cfg)


def test_d_current_pointer_is_exactly_pinned() -> None:
    pointer = (
        ROOT
        / "outputs/model_v1/10_stage4_2d_equity_certification"
        / "CURRENT_STAGE4_2D_EQUITY_CERTIFICATION_RUN.json"
    )
    assert sha256_file(pointer) == FROZEN_D_POINTER_SHA256
    payload = json.loads(pointer.read_text(encoding="utf-8"))
    assert payload["decision"] == "PASS_STAGE4_2D_EQUITY_FRONT_STAGES_CERTIFIED"
    assert payload["stage4_full_computational_complete"] is False
    assert payload["operational_final"] is False
    assert payload["stage5_started"] is False


def test_pinned_official_d_and_e_parent_evidence_loads() -> None:
    parents = load_parent_evidence(ROOT, load_yaml(CONFIG))
    assert parents.stage42d_run_id == "stage4_2d_equity_20260821T014952Z_cc9fd641ddf0"
    assert parents.stage42e_run_id == (
        "stage4_2e_equity_tail_20260821T110943Z_e1352e4b7adc"
    )
    assert len(parents.equity_stages) == 4
    assert all(
        stage.semantic_label == "CERTIFIED_NEAR_OPTIMAL"
        for stage in parents.equity_stages
    )


def test_exact_current_eight_stage_compatibility_recomputation() -> None:
    cfg = load_yaml(CONFIG)
    audit = compatibility_adoption_audit(ROOT, _formulation(), cfg)
    assert audit.passed is True
    assert audit.source_inventory_sha256 == FROZEN_C_INVENTORY_SHA256
    assert audit.stage_count == 8
    assert audit.historical_source_preimage_verified is True
    assert audit.candidate_count == 452
    assert audit.pattern_count == 5242
    assert audit.visit_count == 20
    assert audit.scenario_classification == {
        "efficiency": "CERTIFIED_NEAR_OPTIMAL",
        "balanced": "CERTIFIED_NEAR_OPTIMAL",
    }
    observed = {
        f"{stage.scenario}:{stage.stage_index}": {
            "model_sha256": stage.model_sha256,
            "universe_sha256": stage.universe_sha256,
            "selection_sha256": stage.selection_sha256,
        }
        for stage in audit.stages
    }
    for key, expected in EXPECTED_ADOPTION_STAGES.items():
        assert observed[key] == {
            field: expected[field]
            for field in ("model_sha256", "universe_sha256", "selection_sha256")
        }
    assert all(stage.current_selection_feasible for stage in audit.stages)
    assert all(stage.historical_preimage_selection_feasible for stage in audit.stages)
    assert all(stage.historical_current_model_equal for stage in audit.stages)
    assert all(
        stage.historical_preimage_model_sha256 == stage.model_sha256
        and stage.historical_preimage_universe_sha256 == stage.universe_sha256
        and stage.historical_preimage_selection_sha256 == stage.selection_sha256
        for stage in audit.stages
    )
    assert all(stage.max_constraint_violation <= stage.feasibility_tolerance for stage in audit.stages)
    assert all(stage.semantic_label == "CERTIFIED_NEAR_OPTIMAL" for stage in audit.stages)
    assert all(stage.legacy_solver_claims_used is False for stage in audit.stages)
    assert all(stage.legacy_run_pass_used is False for stage in audit.stages)
    encoded = json.dumps(audit.to_dict(), ensure_ascii=False)
    assert '"semantic_label": "OPTIMAL"' not in encoded
    assert '"status"' not in encoded


def test_pending_parent_aborts_before_creating_any_stage42f_output(
    tmp_path: Path,
) -> None:
    pending_cfg = copy.deepcopy(load_yaml(CONFIG))
    pending_cfg["stage42e_parent"]["run_id"] = (
        "PENDING_AFTER_STAGE4_2E_OFFICIAL_SUCCESS"
    )
    for key in pending_cfg["stage42e_parent"]["artifact_sha256"]:
        pending_cfg["stage42e_parent"]["artifact_sha256"][key] = (
            "PENDING_AFTER_STAGE4_2E_OFFICIAL_SUCCESS"
        )
    pending_path = tmp_path / "stage4_2f_pending.yaml"
    pending_path.write_text(
        yaml.safe_dump(pending_cfg, sort_keys=False),
        encoding="utf-8",
    )
    output = ROOT / "outputs/model_v1/10_stage4_2f_top3_aggregate"
    before = sorted(path.as_posix() for path in output.rglob("*")) if output.exists() else []
    with pytest.raises(PendingStage42E):
        run_top3_semantic_aggregate(
            project_root=ROOT,
            stage42_config_path=Path("configs/model_v1/stage4_2.yaml"),
            stage42c_config_path=Path("configs/model_v1/stage4_2_certification.yaml"),
            stage42d_config_path=Path(
                "configs/model_v1/stage4_2d_equity_certification.yaml"
            ),
            stage42e_config_path=Path(
                "configs/model_v1/stage4_2e_equity_tail_certification.yaml"
            ),
            stage42f_config_path=pending_path,
        )
    after = sorted(path.as_posix() for path in output.rglob("*")) if output.exists() else []
    assert after == before


def test_aggregate_implementation_has_no_solver_backend_or_solve_call() -> None:
    package = ROOT / "src/mediroad/stage4_2f_top3_aggregate"
    implementation = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(package.glob("*.py"))
        if path.name not in {"__init__.py", "__main__.py"}
    )
    assert "HighsBackend" not in implementation
    assert "Scip" not in implementation
    assert ".solve(" not in implementation


def test_parent_solver_status_is_normalized_and_never_copied(tmp_path: Path) -> None:
    source = tmp_path / "certificate.json"
    source.write_text("{}\n", encoding="utf-8")
    stage = _stage(
        certificate={
            "metric": "need_weighted",
            "certified": True,
            "status": "OPTIMAL",
            "semantic_label": "CERTIFIED_EXACT",
            "incumbent_value": 100.0,
            "best_bound": 100.4,
            "relative_gap": 0.004,
        },
        scenario_index=3,
        objective="need_weighted",
        sense="max",
        source_stage="TEST_PARENT",
        source_run_id="test",
        source_path=source,
        root=tmp_path,
    )
    assert stage.semantic_label == "CERTIFIED_NEAR_OPTIMAL"
    encoded = json.dumps(stage.to_dict())
    assert '"semantic_label": "OPTIMAL"' not in encoded
    assert '"semantic_label": "CERTIFIED_EXACT"' not in encoded
    assert "status" not in encoded


def test_equity_tail_parent_must_be_fresh_bound_only(tmp_path: Path) -> None:
    source = tmp_path / "certificate.json"
    source.write_text("{}\n", encoding="utf-8")
    certificate = {
        "objective": "cost",
        "sense": "min",
        "certified": True,
        "incumbent_value": 100.0,
        "best_bound": 99.6,
        "relative_gap": 0.004,
        "fresh_solve_completed": True,
        "fresh_incumbent_valid": True,
        "fresh_bound_valid": True,
        "inherited_bound_used": True,
    }
    with pytest.raises(RuntimeError, match="fresh-incumbent/fresh-bound-only"):
        _stage(
            certificate=certificate,
            scenario_index=4,
            objective="cost",
            sense="min",
            source_stage="STAGE4_2E_TAIL_4",
            source_run_id="test",
            source_path=source,
            root=tmp_path,
            require_fresh_only=True,
        )
    certificate["inherited_bound_used"] = False
    stage = _stage(
        certificate=certificate,
        scenario_index=4,
        objective="cost",
        sense="min",
        source_stage="STAGE4_2E_TAIL_4",
        source_run_id="test",
        source_path=source,
        root=tmp_path,
        require_fresh_only=True,
    )
    assert stage.semantic_label == "CERTIFIED_NEAR_OPTIMAL"


def test_runner_commits_inventory_before_writing_current() -> None:
    source = (
        ROOT / "src/mediroad/stage4_2f_top3_aggregate/runner.py"
    ).read_text(encoding="utf-8")
    inventory = source.index("_verify_output_inventory(root, inventory_path)")
    committed = source.index("atomic_write_json(\n                    committed")
    current = source.index(
        'output_root / "CURRENT_STAGE4_2F_TOP3_AGGREGATE_RUN.json"'
    )
    assert inventory < committed < current
    for seal in (
        "_source_snapshot",
        "_authoritative_input_snapshot",
        "_stage5_snapshot",
        "parent_records_unchanged",
    ):
        assert seal in source


def test_scope_flags_never_overclaim_stage4_or_stage5() -> None:
    cfg = load_yaml(CONFIG)
    assert cfg["contract"]["scope"] == "TOP3_REQUIRED_SCENARIOS_ONLY"
    assert cfg["gate"]["candidate_expansion_certified"] is False
    assert cfg["gate"]["stage4_full_computational_complete"] is False
    assert cfg["gate"]["operational_final"] is False
    assert cfg["stage5_started"] is False
    assert cfg["gate"]["stage5_release_allowed"] is False
