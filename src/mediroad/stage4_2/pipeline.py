from __future__ import annotations

import json
import os
import platform
import sys
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .computational import run_computational_hardening
from .config import LoadedConfig
from .data import normalize_stage3_interface, venue_table_from_interface
from .discovery import discover_stage3, discover_stage41, verify_inventory
from .errors import InputNotReadyError
from .field_validation import audit_field_validation, merge_field_results_with_queue, prepare_priority_queue
from .operational_final import run_operational_final
from .operational_inputs import audit_operational_inputs, build_operational_manifest, write_operational_templates
from .quality import save_quality_gate
from .reporting import write_computational_report, write_field_audit_report, write_prepare_report
from .utils import (
    atomic_write_csv,
    atomic_write_json,
    atomic_write_text,
    build_inventory,
    deterministic_run_id,
    read_json,
    read_table,
    relative_to,
    sha256_file,
    sha256_json,
    single_writer_lock,
    utc_now_iso,
)
from .venue_resolution import resolve_physical_venues


class Stage42Pipeline:
    def __init__(self, root: Path, config: LoadedConfig) -> None:
        self.root = root.resolve()
        self.config = config
        self.output_root = self.root / config.data["paths"].get("output_root", "outputs/model_v1/10_stage4_2")
        self.report_root = self.root / config.data["paths"].get("report_root", "reports/model_v1/stage4_2")
        self.lock_path = self.root / config.data["paths"].get("lock", "outputs/model_v1/.stage4_2_writer.lock")

    def _new_run(self, mode: str) -> tuple[str, Path, Path]:
        contract = {"mode": mode, "config_sha256": self.config.sha256, "version": self.config.data["version"]}
        run_id = deterministic_run_id(f"stage4_2_{mode}", contract)
        run_root = self.output_root / "runs" / run_id
        report_dir = self.report_root / "runs" / run_id
        run_root.mkdir(parents=True, exist_ok=False)
        report_dir.mkdir(parents=True, exist_ok=False)
        atomic_write_text(run_root / ".RUNNING", f"run_id={run_id}\nmode={mode}\npid={os.getpid()}\nstarted={utc_now_iso()}\n")
        return run_id, run_root, report_dir

    def _stage5_snapshot(self) -> list[dict[str, Any]]:
        candidates: set[Path] = set()
        output_parent = self.root / "outputs/model_v1"
        report_parent = self.root / "reports/model_v1"
        if output_parent.exists():
            candidates.update(p for p in output_parent.rglob("CURRENT_STAGE5*.json") if p.is_file())
            candidates.update(p for p in output_parent.iterdir() if p.name.lower().startswith("stage5"))
        if report_parent.exists():
            candidates.update(p for p in report_parent.iterdir() if p.name.lower().startswith("stage5"))
        rows: list[dict[str, Any]] = []
        for path in sorted(candidates):
            if path.is_dir():
                rows.append({"path": relative_to(path, self.root), "kind": "directory"})
            else:
                rows.append(
                    {
                        "path": relative_to(path, self.root),
                        "kind": "file",
                        "size_bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                )
        return rows

    def _source_contract(self) -> list[dict[str, Any]]:
        code_root = self.root if (self.root / "src/mediroad/stage4_2").is_dir() else Path(__file__).resolve().parents[3]
        paths = list((code_root / "src/mediroad/stage4_2").glob("*.py"))
        paths += [
            code_root / "12_scripts/v6/run_model_v1_stage4_2.py",
            code_root / "run_model_v1_stage4_2.sh",
            code_root / "run_model_v1_stage4_2_prepare.ps1",
            code_root / "run_model_v1_stage4_2_computational.ps1",
            self.config.path,
        ]
        paths += list((code_root / "tests").glob("test_stage4_2*.py"))
        paths += [
            code_root / "tests/test_bundles.py",
            code_root / "tests/test_candidate_compression.py",
            code_root / "tests/test_compression.py",
            code_root / "tests/test_field_validation.py",
            code_root / "tests/test_optimizer.py",
            code_root / "tests/test_venue_resolution.py",
        ]
        unique = sorted({path.resolve() for path in paths})
        missing = [str(path) for path in unique if not path.is_file()]
        if missing:
            raise RuntimeError(f"Stage 4.2 source contract files missing: {missing}")
        return [
            {
                "relative_path": relative_to(path, self.root),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in unique
        ]

    @contextmanager
    def _managed_run(self, mode: str):
        with single_writer_lock(self.lock_path):
            run_id, run_root, report_dir = self._new_run(mode)
            metadata = self._base_metadata(run_id, mode, utc_now_iso())
            try:
                yield run_id, run_root, report_dir, metadata
            except BaseException as exc:
                failure = {
                    "run_id": run_id,
                    "mode": mode,
                    "failed_at": utc_now_iso(),
                    "exception_type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                }
                atomic_write_json(run_root / "STAGE4_2_RUN_FAILURE.json", failure)
                source = run_root / ".RUNNING"
                if not source.exists():
                    source = run_root / ".COMMITTED"
                if source.exists():
                    os.replace(source, run_root / ".FAILED")
                mode_pointer = self.output_root / f"CURRENT_STAGE4_2_{mode.upper()}_RUN.json"
                if mode_pointer.is_file():
                    try:
                        if read_json(mode_pointer).get("run_id") == run_id:
                            mode_pointer.unlink()
                    except Exception:
                        pass
                raise

    def _base_metadata(self, run_id: str, mode: str, started: str) -> dict[str, Any]:
        stage3 = discover_stage3(self.root)
        stage41 = discover_stage41(self.root)
        source_contract = self._source_contract()
        return {
            "version": self.config.data["version"],
            "run_id": run_id,
            "mode": mode,
            "started_at": started,
            "completed_at": None,
            "project_root": str(self.root),
            "python": sys.version,
            "platform": platform.platform(),
            "config_relative_path": relative_to(self.config.path, self.root),
            "config_sha256": self.config.sha256,
            "stage3_pointer_sha256": sha256_file(stage3.pointer),
            "stage4_1_pointer_sha256": sha256_file(stage41.pointer),
            "source_contract": source_contract,
            "source_contract_sha256": sha256_json(source_contract),
            "stage5_namespace_before": self._stage5_snapshot(),
            "stage5_started": False,
        }

    def _finish(self, metadata: dict[str, Any], run_root: Path, report_path: Path, decision: dict[str, Any], *, success: bool) -> dict[str, Any]:
        stage3 = discover_stage3(self.root)
        stage41 = discover_stage41(self.root)
        verify_inventory(stage41.run_root, stage41.inventory)
        if sha256_file(self.config.path) != metadata["config_sha256"]:
            raise RuntimeError("Stage 4.2 config changed during the run")
        if sha256_file(stage3.pointer) != metadata["stage3_pointer_sha256"]:
            raise RuntimeError("Stage 3 CURRENT pointer changed during the run")
        if sha256_file(stage41.pointer) != metadata["stage4_1_pointer_sha256"]:
            raise RuntimeError("Stage 4.1 CURRENT pointer changed during the run")
        source_contract = self._source_contract()
        if sha256_json(source_contract) != metadata["source_contract_sha256"]:
            raise RuntimeError("Stage 4.2 source/test contract changed during the run")
        stage5_after = self._stage5_snapshot()
        if stage5_after != metadata["stage5_namespace_before"]:
            raise RuntimeError("Forbidden Stage 5 namespace changed during Stage 4.2")
        metadata["stage5_namespace_after"] = stage5_after
        if not report_path.is_file() or report_path.stat().st_size <= 0:
            raise RuntimeError("Stage 4.2 report is missing or empty")
        metadata.update(
            {
                "completed_at": utc_now_iso(),
                "success": bool(success),
                "decision": decision,
                "report_relative_path": relative_to(report_path, self.root),
            }
        )
        inventory_path = run_root / "STAGE4_2_ARTIFACT_INVENTORY.csv"
        inventory = build_inventory(
            run_root,
            exclude_names={inventory_path.name, "STAGE4_2_RUN_METADATA.json", ".RUNNING", ".COMMITTED", ".FAILED"},
        )
        if inventory.empty or (inventory["size_bytes"] <= 0).any():
            raise RuntimeError("Stage 4.2 inventory contains no artifacts or zero-byte artifacts")
        atomic_write_csv(inventory_path, inventory)
        metadata["artifact_count"] = int(len(inventory))
        metadata["artifact_inventory_sha256"] = sha256_file(inventory_path)
        metadata_path = run_root / "STAGE4_2_RUN_METADATA.json"
        atomic_write_json(metadata_path, metadata)
        for row in inventory.itertuples(index=False):
            artifact = run_root / str(row.relative_path)
            if not artifact.is_file() or artifact.stat().st_size != int(row.size_bytes) or sha256_file(artifact) != row.sha256:
                raise RuntimeError(f"Stage 4.2 artifact changed before commit: {row.relative_path}")
        pointer = {
            "run_id": metadata["run_id"],
            "mode": metadata["mode"],
            "decision": decision,
            "metadata_relative_path": relative_to(metadata_path, self.root),
            "metadata_sha256": sha256_file(metadata_path),
            "inventory_relative_path": relative_to(inventory_path, self.root),
            "inventory_sha256": sha256_file(inventory_path),
            "report_relative_path": relative_to(report_path, self.root),
            "report_sha256": sha256_file(report_path),
            "stage5_started": False,
        }
        if success:
            os.replace(run_root / ".RUNNING", run_root / ".COMMITTED")
            atomic_write_json(self.output_root / f"CURRENT_STAGE4_2_{str(metadata['mode']).upper()}_RUN.json", pointer)
            # The generic CURRENT pointer is the authoritative commit marker and
            # must be written last.
            atomic_write_json(self.output_root / "CURRENT_STAGE4_2_RUN.json", pointer)
        else:
            os.replace(run_root / ".RUNNING", run_root / ".FAILED")
        return pointer

    def prepare(self) -> dict[str, Any]:
        with self._managed_run("prepare") as (run_id, run_root, report_dir, metadata):
            stage41 = discover_stage41(self.root)
            verify_inventory(stage41.run_root, stage41.inventory)
            queue, form, evidence = prepare_priority_queue(stage41)
            atomic_write_csv(run_root / "field_validation_priority_queue.csv", queue)
            atomic_write_csv(run_root / "field_validation_form.csv", form)
            atomic_write_csv(run_root / "field_validation_evidence.csv", evidence)
            bundle_ids = sorted(normalize_stage3_interface(read_table(discover_stage3(self.root).interface))["bundle_id"].astype(str).unique())
            template_paths = write_operational_templates(run_root / "operational_templates", bundle_ids)
            summary = {
                "queue_rows": int(len(queue)),
                "anchor_rows": 20,
                "busproxy_without_fallback": int(
                    (
                        queue.get("is_busproxy_anchor", pd.Series(False, index=queue.index)).astype(bool)
                        & pd.to_numeric(queue.get("actual_facility_fallback_count", 0), errors="coerce").fillna(0).eq(0)
                    ).sum()
                ),
                "certified_intersection": int(queue["certified_efficiency_balanced_intersection"].sum()),
                "template_files": {k: relative_to(v, self.root) for k, v in template_paths.items()},
            }
            atomic_write_json(run_root / "PREPARE_SUMMARY.json", summary)
            report_path = report_dir / "FINAL_REPORT.md"
            write_prepare_report(report_path, summary, queue)
            checks = [
                {"gate_id": "queue_nonempty", "gate_type": "HARD", "passed": len(queue) > 0, "observed": len(queue), "expected": ">0", "message": "Field queue generated"},
                {"gate_id": "stage5_not_started", "gate_type": "HARD", "passed": True, "observed": False, "expected": False, "message": "No Stage 5 outputs generated"},
            ]
            save_quality_gate(run_root / "STAGE4_2_QUALITY_GATE.csv", checks)
            return self._finish(metadata, run_root, report_path, {"decision": "PASS_STAGE4_2_READY_FOR_FIELD_VALIDATION"}, success=True)

    def computational(self) -> dict[str, Any]:
        with self._managed_run("computational") as (run_id, run_root, report_dir, metadata):
            stage41 = discover_stage41(self.root)
            verify_inventory(stage41.run_root, stage41.inventory)
            payload = run_computational_hardening(self.root, self.config.data, run_root)
            metrics_frames = []
            for candidate_set in payload["runs"]:
                p = run_root / "candidate_sets" / candidate_set / "scenario_metrics.csv"
                metrics_frames.append(pd.read_csv(p))
            metrics = pd.concat(metrics_frames, ignore_index=True)
            report_path = report_dir / "FINAL_REPORT.md"
            write_computational_report(report_path, payload["decision"], payload["comparisons"], metrics)
            certified_series = metrics["certified"].map(lambda value: bool(value) if isinstance(value, (bool, np.bool_)) else str(value).strip().lower() in {"true", "1", "yes"})
            metrics = metrics.assign(_certified_bool=certified_series)
            top3_eff = metrics.loc[(metrics.candidate_set == "top3") & (metrics.scenario == "efficiency"), "_certified_bool"]
            top3_bal = metrics.loc[(metrics.candidate_set == "top3") & (metrics.scenario == "balanced"), "_certified_bool"]
            checks = [
                {"gate_id": "top3_efficiency_certified", "gate_type": "HARD", "passed": bool(len(top3_eff) == 1 and top3_eff.iloc[0]), "observed": top3_eff.tolist(), "expected": [True], "message": "Reference efficiency certified"},
                {"gate_id": "top3_balanced_certified", "gate_type": "HARD", "passed": bool(len(top3_bal) == 1 and top3_bal.iloc[0]), "observed": top3_bal.tolist(), "expected": [True], "message": "Reference balanced certified"},
                {"gate_id": "candidate_expansion_conclusive", "gate_type": "ADVISORY", "passed": bool(payload['decision']['candidate_expansion_comparison_conclusive']), "observed": payload['decision']['candidate_expansion_comparison_conclusive'], "expected": True, "message": "Expanded-set comparison requires both sides certified"},
                {"gate_id": "equity_certified", "gate_type": "ADVISORY", "passed": bool(payload['decision']['equity_long_solve_certified']), "observed": payload['decision']['equity_max_relative_gap'], "expected": f"<={self.config.data['solver']['near_optimal_relative_gap_max']}", "message": "Long Equity solve"},
                {"gate_id": "stage5_not_started", "gate_type": "HARD", "passed": True, "observed": False, "expected": False, "message": "No Stage 5 outputs generated"},
            ]
            gate = save_quality_gate(run_root / "STAGE4_2_QUALITY_GATE.csv", checks)
            success = bool(gate.loc[gate.gate_type.eq("HARD"), "passed"].all())
            decision = {
                "decision": "PASS_STAGE4_2_COMPUTATIONAL_HARDENING" if success else "FAIL_STAGE4_2",
                **payload["decision"],
            }
            return self._finish(metadata, run_root, report_path, decision, success=success)

    def audit_field(self, field_validation_path: Path, operational_paths: dict[str, Path] | None = None) -> dict[str, Any]:
        with self._managed_run("field_audit") as (run_id, run_root, report_dir, metadata):
            stage3 = discover_stage3(self.root)
            stage41 = discover_stage41(self.root)
            queue = pd.read_csv(stage41.field_queue, low_memory=False)
            audit = audit_field_validation(read_table(field_validation_path))
            merged = merge_field_results_with_queue(queue, audit)
            atomic_write_csv(run_root / "field_validation_normalized.csv", audit.normalized)
            atomic_write_csv(run_root / "field_validation_issues.csv", audit.issues)
            atomic_write_csv(run_root / "field_validation_queue_merged.csv", merged)
            venue_catalog = venue_table_from_interface(normalize_stage3_interface(read_table(stage3.interface)))
            resolution = resolve_physical_venues(read_table(stage41.field_interface), audit.normalized, venue_catalog)
            atomic_write_csv(run_root / "physical_venue_resolution.csv", resolution.resolution)
            atomic_write_csv(run_root / "unresolved_anchors.csv", resolution.unresolved)
            blockers = []
            advisories = []
            op_summary = {}
            verified_ids = set(audit.normalized.loc[audit.normalized["field_verified"], "venue_id"].astype(str))
            verified_non_bus = {venue_id for venue_id in verified_ids if not venue_id.startswith("BUSPROXY_")}
            required_visits = int(self.config.data.get("optimization", {}).get("visit_count", 20))
            if operational_paths:
                op = audit_operational_inputs(operational_paths, verified_ids)
                blockers.extend(op.blockers)
                op_summary = op.summaries
                atomic_write_json(run_root / "operational_input_manifest.json", build_operational_manifest(operational_paths))
            if not resolution.summary["all_resolved"]:
                advisories.append(f"STAGE4_1_UNRESOLVED_ANCHORS:{resolution.summary['unresolved']}:REOPTIMIZATION_CAN_REPLACE")
            if len(verified_non_bus) < required_visits:
                blockers.append(f"INSUFFICIENT_VERIFIED_NON_BUS_POOL:{len(verified_non_bus)}<{required_visits}")
            report_path = report_dir / "FINAL_REPORT.md"
            write_field_audit_report(report_path, audit.summary, resolution.summary, [*blockers, *advisories])
            decision_name = "PASS_STAGE4_2_READY_FOR_OPERATIONAL_FINAL" if not blockers else "PASS_STAGE4_2_READY_FOR_FIELD_VALIDATION"
            decision = {
                "decision": decision_name,
                "blockers": blockers,
                "advisories": advisories,
                "verified_non_bus_pool": len(verified_non_bus),
                "field": audit.summary,
                "resolution": resolution.summary,
                "operational": op_summary,
            }
            checks = [
                {"gate_id": "field_schema_valid", "gate_type": "HARD", "passed": audit.summary["invalid_issue_count"] == 0, "observed": audit.summary["invalid_issue_count"], "expected": 0, "message": "Field form schema and statuses"},
                {"gate_id": "verified_non_bus_pool_sufficient", "gate_type": "OPERATIONAL", "passed": len(verified_non_bus) >= required_visits, "observed": len(verified_non_bus), "expected": f">={required_visits}", "message": "Reoptimization has enough verified physical venues"},
                {"gate_id": "all_stage4_1_anchors_resolved", "gate_type": "ADVISORY", "passed": resolution.summary["all_resolved"], "observed": resolution.summary["unresolved"], "expected": 0, "message": "Old provisional anchors may be replaced by reoptimization"},
                {"gate_id": "stage5_not_started", "gate_type": "HARD", "passed": True, "observed": False, "expected": False, "message": "No Stage 5 outputs generated"},
            ]
            save_quality_gate(run_root / "STAGE4_2_QUALITY_GATE.csv", checks)
            return self._finish(metadata, run_root, report_path, decision, success=True)

    def operational_final(self, field_validation_path: Path, operational_paths: dict[str, Path]) -> dict[str, Any]:
        with self._managed_run("operational_final") as (run_id, run_root, report_dir, metadata):
            report_path = report_dir / "FINAL_REPORT.md"
            try:
                payload = run_operational_final(
                    self.root,
                    self.config.data,
                    run_root,
                    field_validation_path=field_validation_path,
                    operational_paths=operational_paths,
                )
                decision = payload["decision"]
                write_field_audit_report(report_path, {"status": "COMPLETE"}, payload["resolution"].summary, [])
                checks = [
                    {"gate_id": "operational_final_decision", "gate_type": "HARD", "passed": decision["decision"] == "PASS_STAGE4_OPERATIONAL_FINAL", "observed": decision["decision"], "expected": "PASS_STAGE4_OPERATIONAL_FINAL", "message": "All spatial/field/travel policy runs certified"},
                    {"gate_id": "stage5_not_started", "gate_type": "HARD", "passed": True, "observed": False, "expected": False, "message": "Stage 5 still not executed"},
                ]
                gate = save_quality_gate(run_root / "STAGE4_2_QUALITY_GATE.csv", checks)
                success = bool(gate.loc[gate.gate_type.eq("HARD"), "passed"].all())
                return self._finish(metadata, run_root, report_path, decision, success=success)
            except InputNotReadyError as exc:
                decision = {"decision": "PASS_STAGE4_2_READY_FOR_FIELD_VALIDATION", "blocker": str(exc), "stage5_started": False}
                write_field_audit_report(report_path, {"status": "INCOMPLETE"}, {}, [str(exc)])
                save_quality_gate(
                    run_root / "STAGE4_2_QUALITY_GATE.csv",
                    [
                        {"gate_id": "operational_inputs_ready", "gate_type": "OPERATIONAL", "passed": False, "observed": str(exc), "expected": "all confirmed", "message": "Operational final blocked without inventing data"},
                        {"gate_id": "stage5_not_started", "gate_type": "HARD", "passed": True, "observed": False, "expected": False, "message": "Stage 5 not executed"},
                    ],
                )
                return self._finish(metadata, run_root, report_path, decision, success=True)
