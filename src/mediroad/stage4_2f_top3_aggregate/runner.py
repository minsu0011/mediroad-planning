from __future__ import annotations

import hashlib
import json
import platform
import sys
import traceback
from pathlib import Path
from typing import Any

import pandas as pd

from mediroad.stage4_2_certification.adapter import (
    load_base_stage42_config,
    prepare_candidate_problem,
)
from mediroad.stage4_2_certification.config import validate_certification_config
from mediroad.stage4_2_certification.formulation import CertificationFormulation
from mediroad.stage4_2d_equity_certification.config import (
    validate_config as validate_stage42d_config,
    validate_parent_contracts,
)
from mediroad.stage4_2d_equity_certification.io_utils import (
    atomic_write_csv,
    atomic_write_json,
    make_run_id,
    sha256_file,
    tree_inventory,
    utc_now,
)
from mediroad.stage4_2d_equity_certification.runner import (
    _atomic_write_text,
    _authoritative_input_snapshot,
    _exclusive_lock,
    _run_regression_tests,
    _stage5_snapshot,
    _verify_output_inventory,
)
from mediroad.stage4_2e_equity_tail_certification.config import (
    validate_config as validate_stage42e_config,
)

from .audit import compatibility_adoption_audit, evidence_records_unchanged
from .config import load_yaml, validate_config
from .parents import load_parent_evidence, parent_records_unchanged
from .version import PACKAGE_NAME, VERSION


DECISION_PASS = "PASS_STAGE4_2F_TOP3_REQUIRED_SCENARIOS_CERTIFIED"
DECISION_FAIL = "FAIL_STAGE4_2F_TOP3_REQUIRED_SCENARIOS_UNCERTIFIED"
SCOPE = "TOP3_REQUIRED_SCENARIOS_ONLY"


def _snapshot(paths: list[Path], root: Path) -> list[dict[str, Any]]:
    root = root.resolve()
    output: list[dict[str, Any]] = []
    for path in sorted({path.resolve() for path in paths}):
        path.relative_to(root)
        output.append(
            {
                "relative_path": path.relative_to(root).as_posix(),
                "exists": path.is_file(),
                "size_bytes": int(path.stat().st_size) if path.is_file() else None,
                "sha256": sha256_file(path) if path.is_file() else None,
            }
        )
    return output


def _source_snapshot(root: Path, config_paths: list[Path]) -> list[dict[str, Any]]:
    paths: set[Path] = set(config_paths)
    for relative in (
        "src/mediroad/stage4_2",
        "src/mediroad/stage4_2_certification",
        "src/mediroad/stage4_2d_equity_certification",
        "src/mediroad/stage4_2e_equity_tail_certification",
        "src/mediroad/stage4_2f_top3_aggregate",
    ):
        package = root / relative
        if package.is_dir():
            paths.update(package.rglob("*.py"))
    tests = root / "tests"
    if tests.is_dir():
        paths.update(tests.glob("test_*.py"))
    for pattern in (
        "requirements_stage4_2*.txt",
        "environment_stage4_2*.yml",
        "run_model_v1_stage4_2f_top3_aggregate.ps1",
        "12_scripts/v6/run_model_v1_stage4_2f_top3_aggregate.py",
    ):
        paths.update(path for path in root.glob(pattern) if path.is_file())
    return _snapshot([path for path in paths if path.is_file()], root)


def _parent_paths(root: Path, cfg: dict[str, Any]) -> list[Path]:
    c_run = (
        root
        / "outputs/model_v1/10_stage4_2_certification_diagnostic/runs"
        / str(cfg["preferred_stage42c"]["run_id"])
    )
    d_run = (
        root
        / "outputs/model_v1/10_stage4_2d_equity_certification/runs"
        / str(cfg["stage42d_parent"]["run_id"])
    )
    e_run = (
        root
        / "outputs/model_v1/10_stage4_2e_equity_tail_certification/runs"
        / str(cfg["stage42e_parent"]["run_id"])
    )
    paths = [
        c_run / relative
        for relative in cfg["preferred_stage42c"]["artifact_sha256"]
    ]
    paths.extend(
        root / relative
        for relative in cfg["preferred_stage42c"]["source_preimage_sha256"]
    )
    paths.extend(
        root / relative
        for relative in cfg["preferred_stage42c"]["historical_parent_pointer_sha256"]
    )
    paths.extend(
        d_run / relative
        for relative in cfg["stage42d_parent"]["artifact_sha256"]
        if not relative.startswith("CURRENT_")
    )
    paths.append(
        root
        / "outputs/model_v1/10_stage4_2d_equity_certification"
        / "CURRENT_STAGE4_2D_EQUITY_CERTIFICATION_RUN.json"
    )
    paths.extend(
        e_run / relative
        for relative in cfg["stage42e_parent"]["artifact_sha256"]
        if not relative.startswith("CURRENT_")
    )
    paths.append(
        root
        / "outputs/model_v1/10_stage4_2e_equity_tail_certification"
        / "CURRENT_STAGE4_2E_EQUITY_TAIL_CERTIFICATION_RUN.json"
    )
    return paths


def _report(
    path: Path,
    *,
    run_id: str,
    decision: str,
    scenario_summary: list[dict[str, Any]],
    stages: list[dict[str, Any]],
) -> None:
    lines = [
        "# MEDIROAD Stage 4.2F — Top3 Required-Scenario Semantic Aggregate",
        "",
        f"- Run: `{run_id}`",
        f"- Decision: `{decision}`",
        f"- Scope: `{SCOPE}`",
        "- Frozen relative gap: `0.005`",
        "",
        "## Scenario certification",
        "",
        "| Scenario | Stages | Semantic classification | Source method |",
        "|---|---:|---|---|",
    ]
    for row in scenario_summary:
        lines.append(
            f"| {row['scenario']} | {row['stage_count']} | {row['semantic_label']} | {row['basis']} |"
        )
    lines.extend(
        [
            "",
            "## Stage ledger",
            "",
            "| Scenario | Stage | Objective | Sense | Label | Gap | Source |",
            "|---|---:|---|---|---|---:|---|",
        ]
    )
    for row in stages:
        source = row.get("source_stage", "STAGE4_2C_COMPATIBILITY_ADOPTION")
        lines.append(
            f"| {row['scenario']} | {row['stage_index']} | {row['objective']} | "
            f"{row['sense']} | {row['semantic_label']} | {float(row['relative_gap']):.12g} | {source} |"
        )
    lines.extend(
        [
            "",
            "Efficiency와 Balanced는 선호 Stage4.2C 실행의 solver 상태나 실행 전체 PASS를 복사하지 않았습니다. "
            "고정된 incumbent/bound, C 실행 당시 source preimage, 현재 모델의 8개 model/universe/selection digest와 "
            "양쪽 CPU 재계산이 모두 같을 때만 호환 채택했습니다.",
            "",
            "이 결과는 Top3 필수 3개 시나리오의 계산 인증만 집계합니다. 후보 확장, 전체 Stage4 계산 완료, "
            "현장 검증, Operational Final, Stage5는 인증하거나 시작하지 않습니다.",
            "",
        ]
    )
    _atomic_write_text(path, "\n".join(lines))


def run_top3_semantic_aggregate(
    *,
    project_root: Path,
    stage42_config_path: Path,
    stage42c_config_path: Path,
    stage42d_config_path: Path,
    stage42e_config_path: Path,
    stage42f_config_path: Path,
) -> dict[str, Any]:
    root = project_root.resolve()

    def resolved(path: Path) -> Path:
        return path.resolve() if path.is_absolute() else (root / path).resolve()

    bcfg_path = resolved(stage42_config_path)
    ccfg_path = resolved(stage42c_config_path)
    dcfg_path = resolved(stage42d_config_path)
    ecfg_path = resolved(stage42e_config_path)
    fcfg_path = resolved(stage42f_config_path)
    cfg = load_yaml(fcfg_path)
    # This occurs before creating an output directory.  The checked-in PENDING
    # template therefore cannot leave a misleading failed or CURRENT run.
    validate_config(cfg)
    base_cfg = load_base_stage42_config(bcfg_path)
    cert_cfg = load_yaml(ccfg_path)
    d_cfg = load_yaml(dcfg_path)
    e_cfg = load_yaml(ecfg_path)
    validate_certification_config(cert_cfg)
    validate_parent_contracts(base_cfg, cert_cfg)
    validate_stage42d_config(d_cfg)
    validate_stage42e_config(e_cfg)

    config_paths = [bcfg_path, ccfg_path, dcfg_path, ecfg_path, fcfg_path]
    source_before = _source_snapshot(root, config_paths)
    parent_before = _snapshot(_parent_paths(root, cfg), root)
    data_before = _authoritative_input_snapshot(root)
    stage5_before = _stage5_snapshot(root)
    fingerprint_payload = {
        "package": PACKAGE_NAME,
        "version": VERSION,
        "source": source_before,
        "parents": parent_before,
        "authoritative_data": data_before,
        "stage5": stage5_before,
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    run_id = make_run_id("stage4_2f_top3_aggregate", fingerprint)
    output_root = root / str(cfg["paths"]["output_root"])
    report_root = root / str(cfg["paths"]["report_root"])
    run_dir = output_root / "runs" / run_id
    report_dir = report_root / "runs" / run_id
    lock = root / str(cfg["paths"]["lock"])
    running = run_dir / ".RUNNING"
    committed = run_dir / ".COMMITTED"
    failed = run_dir / ".FAILED"

    with _exclusive_lock(lock):
        run_dir.mkdir(parents=True, exist_ok=False)
        report_dir.mkdir(parents=True, exist_ok=False)
        atomic_write_json(running, {"run_id": run_id, "started_utc": utc_now()})
        try:
            regression = _run_regression_tests(root, run_dir / "00_regression")
            problem = prepare_candidate_problem(root, "top3", base_cfg, cache={})
            formulation = CertificationFormulation(
                problem.candidates,
                problem.patterns,
                base_cfg,
                cert_cfg,
                candidate_set="top3",
            )
            formulation.venue_alias_map = dict(problem.venue_alias_map)
            compatibility = compatibility_adoption_audit(root, formulation, cfg)
            parents = load_parent_evidence(root, cfg)

            compatibility_payload = compatibility.to_dict()
            parent_payload = parents.to_dict()
            atomic_write_json(
                run_dir / "01_efficiency_balanced_adoption/compatibility_audit.json",
                compatibility_payload,
            )
            atomic_write_json(
                run_dir / "02_equity_parent_aggregation/parent_evidence.json",
                parent_payload,
            )

            stages = [stage.to_dict() for stage in compatibility.stages]
            stages.extend(stage.to_dict() for stage in parents.equity_stages)
            scenario_summary = [
                {
                    "scenario": "efficiency",
                    "stage_count": 4,
                    "semantic_label": "CERTIFIED_NEAR_OPTIMAL",
                    "basis": "EXACT_CURRENT_COMPATIBILITY_ADOPTION",
                },
                {
                    "scenario": "balanced",
                    "stage_count": 4,
                    "semantic_label": "CERTIFIED_NEAR_OPTIMAL",
                    "basis": "EXACT_CURRENT_COMPATIBILITY_ADOPTION",
                },
                {
                    "scenario": "equity",
                    "stage_count": 4,
                    "semantic_label": "CERTIFIED_NEAR_OPTIMAL",
                    "basis": "PINNED_STAGE4_2D_PLUS_STAGE4_2E_OFFICIAL_CERTIFICATES",
                },
            ]
            atomic_write_json(
                run_dir / "03_top3_semantic_aggregate/scenario_certification.json",
                {"scope": SCOPE, "scenarios": scenario_summary, "stages": stages},
            )
            flat_rows = []
            for row in stages:
                flat_rows.append(
                    {
                        key: row.get(key)
                        for key in (
                            "scenario",
                            "stage_index",
                            "objective",
                            "sense",
                            "semantic_label",
                            "certificate_basis",
                            "source_run_id",
                            "source_relative_path",
                            "source_artifact_sha256",
                            "incumbent",
                            "best_bound",
                            "relative_gap",
                            "model_sha256",
                            "universe_sha256",
                            "selection_sha256",
                        )
                    }
                )
            atomic_write_csv(
                run_dir / "03_top3_semantic_aggregate/stage_certification.csv",
                pd.DataFrame(flat_rows),
            )

            parent_after = _snapshot(_parent_paths(root, cfg), root)
            data_after = _authoritative_input_snapshot(root)
            stage5_after = _stage5_snapshot(root)
            source_after = _source_snapshot(root, config_paths)
            checks = {
                "regression_suite_passed": regression.get("return_code") == 0
                and int(regression.get("failed_count", 0)) == 0,
                "preferred_stage42c_inventory_exact": compatibility.source_inventory_sha256
                == cfg["preferred_stage42c"]["artifact_sha256"]["ARTIFACT_INVENTORY.csv"],
                "exact_eight_stage_model_universe_selection_digests_verified": compatibility.passed
                and compatibility.stage_count == 8,
                "current_cpu_recomputation_verified": all(
                    stage.current_selection_feasible for stage in compatibility.stages
                ),
                "historical_source_preimage_verified": compatibility.historical_source_preimage_verified,
                "historical_preimage_equals_current_eight_models": all(
                    stage.historical_preimage_selection_feasible
                    and stage.historical_current_model_equal
                    for stage in compatibility.stages
                ),
                "legacy_optimal_claim_not_copied": all(
                    not stage.legacy_solver_claims_used for stage in compatibility.stages
                ),
                "legacy_run_pass_not_copied": all(
                    not stage.legacy_run_pass_used for stage in compatibility.stages
                ),
                "efficiency_relabelled_certified_near_optimal": compatibility.scenario_classification.get(
                    "efficiency"
                )
                == "CERTIFIED_NEAR_OPTIMAL",
                "balanced_relabelled_certified_near_optimal": compatibility.scenario_classification.get(
                    "balanced"
                )
                == "CERTIFIED_NEAR_OPTIMAL",
                "official_stage42d_current_exact": parents.stage42d_run_id
                == cfg["stage42d_parent"]["run_id"],
                "official_stage42e_current_exact": parents.stage42e_run_id
                == cfg["stage42e_parent"]["run_id"],
                "equity_four_stages_certified": len(parents.equity_stages) == 4
                and all(
                    stage.semantic_label == "CERTIFIED_NEAR_OPTIMAL"
                    and stage.relative_gap <= 0.005 + 1e-12
                    for stage in parents.equity_stages
                ),
                "all_three_required_scenarios_have_four_stages": len(stages) == 12
                and all(row["stage_count"] == 4 for row in scenario_summary),
                "all_exposed_semantic_labels_are_conservative": all(
                    row["semantic_label"] == "CERTIFIED_NEAR_OPTIMAL" for row in stages
                ),
                "parent_artifacts_unchanged": parent_after == parent_before
                and parent_records_unchanged(root, parents.evidence_records),
                "compatibility_evidence_unchanged": evidence_records_unchanged(
                    root, compatibility.evidence_records
                ),
                "authoritative_data_inventories_unchanged": data_after == data_before,
                "tested_source_unchanged": source_after == source_before,
                "stage5_namespace_absent_before": int(stage5_before.get("count", -1)) == 0,
                "stage5_namespace_unchanged": stage5_after == stage5_before,
                "aggregate_invoked_no_solver": True,
                "candidate_expansion_certified": False,
                "stage4_full_computational_complete": False,
                "operational_final": False,
                "stage5_started": False,
                "stage5_release_allowed": False,
            }
            passed = all(
                value is True
                for key, value in checks.items()
                if key
                not in {
                    "candidate_expansion_certified",
                    "stage4_full_computational_complete",
                    "operational_final",
                    "stage5_started",
                    "stage5_release_allowed",
                }
            ) and all(
                checks[key] is False
                for key in (
                    "candidate_expansion_certified",
                    "stage4_full_computational_complete",
                    "operational_final",
                    "stage5_started",
                    "stage5_release_allowed",
                )
            )
            decision = DECISION_PASS if passed else DECISION_FAIL
            gate = {
                "decision": decision,
                "passed": passed,
                "promotable_scope": SCOPE if passed else "NONE",
                "top3_required_scenarios_certified": passed,
                "scenario_classification": {
                    row["scenario"]: row["semantic_label"] for row in scenario_summary
                },
                "checks": checks,
                "candidate_expansion_certified": False,
                "stage4_full_computational_complete": False,
                "operational_final": False,
                "stage5_started": False,
                "stage5_release_allowed": False,
            }
            quality_dir = run_dir / "04_quality_and_provenance"
            atomic_write_json(quality_dir / "quality_gate.json", gate)
            atomic_write_json(quality_dir / "parent_artifacts_before.json", parent_before)
            atomic_write_json(quality_dir / "parent_artifacts_after.json", parent_after)
            atomic_write_json(quality_dir / "authoritative_data_before.json", data_before)
            atomic_write_json(quality_dir / "authoritative_data_after.json", data_after)
            atomic_write_json(quality_dir / "tested_source_before.json", source_before)
            atomic_write_json(quality_dir / "tested_source_after.json", source_after)
            atomic_write_json(
                quality_dir / "namespace_guard.json",
                {"before": stage5_before, "after": stage5_after},
            )

            metadata = {
                "run_id": run_id,
                "package": PACKAGE_NAME,
                "version": VERSION,
                "created_utc": utc_now(),
                "fingerprint": fingerprint,
                "fingerprint_payload": fingerprint_payload,
                "python_executable": sys.executable,
                "python_version": sys.version,
                "platform": platform.platform(),
                "solver_invoked": False,
                "compatibility_adoption": compatibility_payload,
                "parent_evidence": parent_payload,
                "scenario_summary": scenario_summary,
                "quality_gate": gate,
                "scope": SCOPE if passed else "NONE",
                "candidate_expansion_certified": False,
                "stage4_full_computational_complete": False,
                "operational_final": False,
                "stage5_started": False,
                "stage5_release_allowed": False,
            }
            metadata_path = run_dir / "metadata.json"
            atomic_write_json(metadata_path, metadata)
            report_path = report_dir / "FINAL_REPORT.md"
            _report(
                report_path,
                run_id=run_id,
                decision=decision,
                scenario_summary=scenario_summary,
                stages=stages,
            )
            atomic_write_json(report_dir / "quality_gate.json", gate)

            excluded = {".RUNNING", ".FAILED", ".COMMITTED", "ARTIFACT_INVENTORY.csv"}
            artifacts = [
                path
                for path in run_dir.rglob("*")
                if path.is_file() and path.name not in excluded
            ] + [path for path in report_dir.rglob("*") if path.is_file()]
            inventory_path = run_dir / "ARTIFACT_INVENTORY.csv"
            atomic_write_csv(inventory_path, pd.DataFrame(tree_inventory(artifacts, root)))
            _verify_output_inventory(root, inventory_path)

            if _snapshot(_parent_paths(root, cfg), root) != parent_before:
                raise RuntimeError("Stage4.2C/2D/2E parent changed before Stage4.2F commit")
            if _authoritative_input_snapshot(root) != data_before:
                raise RuntimeError("Authoritative data changed before Stage4.2F commit")
            if _stage5_snapshot(root) != stage5_before:
                raise RuntimeError("Stage5 namespace changed before Stage4.2F commit")
            if _source_snapshot(root, config_paths) != source_before:
                raise RuntimeError("Tested Stage4.2F source changed before commit")
            if not evidence_records_unchanged(root, compatibility.evidence_records):
                raise RuntimeError("Compatibility evidence changed before Stage4.2F commit")
            if not parent_records_unchanged(root, parents.evidence_records):
                raise RuntimeError("Official D/E evidence changed before Stage4.2F commit")
            _verify_output_inventory(root, inventory_path)

            running.unlink(missing_ok=True)
            if passed:
                atomic_write_json(
                    committed,
                    {
                        "run_id": run_id,
                        "committed_utc": utc_now(),
                        "inventory_sha256": sha256_file(inventory_path),
                    },
                )
                # CURRENT is the final write.  It can only point to a fully
                # inventoried, immutable PASS run.
                atomic_write_json(
                    output_root / "CURRENT_STAGE4_2F_TOP3_AGGREGATE_RUN.json",
                    {
                        "run_id": run_id,
                        "run_relative_path": run_dir.relative_to(root).as_posix(),
                        "metadata_relative_path": metadata_path.relative_to(root).as_posix(),
                        "metadata_sha256": sha256_file(metadata_path),
                        "inventory_relative_path": inventory_path.relative_to(root).as_posix(),
                        "inventory_sha256": sha256_file(inventory_path),
                        "report_relative_path": report_path.relative_to(root).as_posix(),
                        "report_sha256": sha256_file(report_path),
                        "quality_gate_relative_path": (
                            quality_dir / "quality_gate.json"
                        ).relative_to(root).as_posix(),
                        "quality_gate_sha256": sha256_file(
                            quality_dir / "quality_gate.json"
                        ),
                        "decision": decision,
                        "scope": SCOPE,
                        "top3_required_scenarios_certified": True,
                        "candidate_expansion_certified": False,
                        "stage4_full_computational_complete": False,
                        "operational_final": False,
                        "stage5_started": False,
                        "stage5_release_allowed": False,
                        "created_utc": utc_now(),
                    },
                )
            else:
                atomic_write_json(
                    failed,
                    {"run_id": run_id, "failed_utc": utc_now(), "decision": decision},
                )
                atomic_write_json(
                    output_root / "LATEST_FAILED_STAGE4_2F_TOP3_AGGREGATE_RUN.json",
                    {
                        "run_id": run_id,
                        "run_relative_path": run_dir.relative_to(root).as_posix(),
                        "decision": decision,
                    },
                )
            return {
                "run_id": run_id,
                "passed": passed,
                "decision": decision,
                "scope": SCOPE if passed else "NONE",
                "run_dir": str(run_dir),
                "report_dir": str(report_dir),
            }
        except Exception as exc:
            running.unlink(missing_ok=True)
            atomic_write_json(
                failed,
                {
                    "run_id": run_id,
                    "failed_utc": utc_now(),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )
            atomic_write_json(
                output_root / "LATEST_FAILED_STAGE4_2F_TOP3_AGGREGATE_RUN.json",
                {
                    "run_id": run_id,
                    "run_relative_path": run_dir.relative_to(root).as_posix(),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            raise
