from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import traceback
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml
from scipy import sparse

from mediroad.stage4_2.contracts import (
    EVIDENCE_COLUMNS,
    FIELD_DATE_COLUMN,
    FIELD_DERIVED_COLUMNS,
    FIELD_KEY_COLUMNS,
    FIELD_REQUIRED_INPUT_COLUMNS,
    FIELD_STATUS_COLUMNS,
)
from mediroad.stage4_2.data import (
    align_coverage,
    bundle_table_from_interface,
    load_coverage,
    normalize_grid_policy,
    normalize_stage3_interface,
    venue_table_from_interface,
)
from mediroad.stage4_2.discovery import discover_stage3
from mediroad.stage4_2.field_validation import audit_field_validation
from mediroad.stage4_2.utils import read_table
from mediroad.stage4_operational_final.field_evidence import audit_field_evidence
from mediroad.stage4_operational_final.candidate_universe import (
    OperationalCandidateUniverse,
    build_verified_operational_universe,
)
from mediroad.stage4_operational_final.operational_io import (
    OPERATIONAL_INPUT_FILENAMES,
    capture_operational_input_snapshot,
    load_operational_input_dir,
    write_operational_template_pack,
)
from mediroad.stage4_operational_final.operational_optimizer import (
    OperationalOptimizationResult,
    optimize_operational_scenario,
)
from mediroad.stage4_operational_final.operational_contract import OperationalTables
from mediroad.stage4_operational_final.operational_travel import (
    OperationalTravelError,
    recompute_operational_travel,
)


EXPECTED_H_DECISION = "PASS_STAGE4_2H_COMPUTATIONAL_FINAL_COARSE_PARETO__FIELD_VALIDATION_PENDING"
BLOCKED_DECISION = "BLOCKED_STAGE4_OPERATIONAL_FINAL_REAL_WORLD_INPUTS_REQUIRED"
THRESHOLD_FINGERPRINT = "b0a1a2c6dc6f1d1e7c3ae1141700fbb3c54854b03746d2e45d2d0ba33c3b1ad4"
POLICIES = ("efficiency", "balanced", "equity")
PRIORITY_RANK = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
NARROW_IDENTITY_EVIDENCE_RELATIVE_PATH = (
    "14_v6_external_data/operational_final/"
    "narrow_venue_identity_evidence_20260822.csv"
)
ALLOWED_CERTIFICATES = {
    "OPTIMAL",
    "DIRECT_MIP_GAP",
    "INFEASIBILITY_ORACLE",
    "HIGHS_CANDIDATE_ONLY_RELAXATION_INFEASIBILITY",
    "SCIP_THRESHOLD_INFEASIBILITY",
}
EXPECTED_STAGE_OBJECTIVES = {
    "efficiency": ["total_population", "need_weighted", "min_sigungu_coverage", "cost"],
    "balanced": ["need_weighted", "min_sigungu_coverage", "total_population", "cost"],
    "equity": ["min_sigungu_coverage", "high_need_population", "need_weighted", "cost"],
}


@dataclass(frozen=True)
class Baseline:
    root: Path
    h_pointer_path: Path
    h_pointer: dict[str, Any]
    h_run: Path
    g_pointer_path: Path
    g_pointer: dict[str, Any]
    g_run: Path
    stage3_pointer_path: Path
    stage41_pointer_path: Path
    stage41_run: Path
    plan_paths: dict[str, Path]
    membership_path: Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    os.replace(temporary, path)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(text.rstrip() + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    shutil.copyfile(source, temporary)
    os.replace(temporary, destination)


def _write_sparse_npz(path: Path, matrix: sparse.spmatrix) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp.npz"
    )
    try:
        sparse.save_npz(temporary, matrix.tocsr(), compressed=True)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _truth(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _verify_inventory(
    run_root: Path,
    inventory_path: Path,
    *,
    require_complete: bool = False,
) -> dict[str, Any]:
    rows = list(csv.DictReader(inventory_path.open("r", encoding="utf-8-sig", newline="")))
    required_columns = {"relative_path", "size_bytes", "sha256"}
    if not rows and inventory_path.stat().st_size > 0:
        header = next(csv.reader(inventory_path.open("r", encoding="utf-8-sig", newline="")), [])
        if not required_columns.issubset(header):
            raise RuntimeError(f"Inventory schema is invalid: {inventory_path}")
    if rows and not required_columns.issubset(rows[0]):
        raise RuntimeError(f"Inventory schema is invalid: {inventory_path}")
    relative_paths = [str(row["relative_path"]).replace("\\", "/") for row in rows]
    if len(relative_paths) != len(set(relative_paths)):
        raise RuntimeError(f"Inventory has duplicate paths: {inventory_path}")
    failures: list[dict[str, Any]] = []
    root_resolved = run_root.resolve()
    for row, relative_path in zip(rows, relative_paths):
        candidate = Path(relative_path)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise RuntimeError(f"Inventory path escapes run root: {relative_path}")
        artifact = (run_root / candidate).resolve()
        try:
            artifact.relative_to(root_resolved)
        except ValueError as exc:
            raise RuntimeError(f"Inventory path escapes run root: {relative_path}") from exc
        exists = artifact.is_file()
        actual_size = artifact.stat().st_size if exists else None
        actual_sha = _sha256(artifact) if exists else None
        if (
            not exists
            or actual_size != int(row["size_bytes"])
            or actual_sha != str(row["sha256"]).lower()
        ):
            failures.append(
                {
                    "relative_path": relative_path,
                    "exists": exists,
                    "expected_size": int(row["size_bytes"]),
                    "actual_size": actual_size,
                    "expected_sha256": row["sha256"],
                    "actual_sha256": actual_sha,
                }
            )
    if failures:
        raise RuntimeError(f"Inventory verification failed for {inventory_path}: {failures[:3]}")
    unlisted: list[str] = []
    if require_complete:
        ignored = {
            "ARTIFACT_INVENTORY.csv",
            "ARTIFACT_INVENTORY_AUDIT.json",
            ".RUNNING",
            ".FAILED",
            ".COMMITTED",
        }
        actual = {
            path.relative_to(run_root).as_posix()
            for path in run_root.rglob("*")
            if path.is_file() and path.name not in ignored
        }
        unlisted = sorted(actual - set(relative_paths))
        missing_from_disk = sorted(set(relative_paths) - actual)
        if unlisted or missing_from_disk:
            raise RuntimeError(
                "Inventory file-set is incomplete: "
                f"unlisted={unlisted[:10]} missing={missing_from_disk[:10]}"
            )
    return {
        "rows": len(rows),
        "mismatches": 0,
        "unlisted_files": unlisted,
        "inventory_sha256": _sha256(inventory_path),
    }


def resolve_baseline(root: Path) -> tuple[Baseline, dict[str, Any]]:
    h_pointer_path = root / "outputs/model_v1/10_stage4_2h_coarse_equity/CURRENT_STAGE4_2H_COMPUTATIONAL_FINAL_RUN.json"
    g_pointer_path = root / "outputs/model_v1/10_stage4_2g_candidate_expansion/CURRENT_STAGE4_2G_CANDIDATE_EXPANSION_RUN.json"
    stage3_pointer_path = root / "outputs/model_v1/07_stage3/CURRENT_STAGE3_RUN.json"
    stage41_pointer_path = root / "outputs/model_v1/09_stage4_finalization/CURRENT_STAGE4_FINALIZATION_RUN.json"
    for path in (h_pointer_path, g_pointer_path, stage3_pointer_path, stage41_pointer_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    h = _json(h_pointer_path)
    if h.get("decision") != EXPECTED_H_DECISION or not _truth(h.get("passed")):
        raise RuntimeError(f"Stage4.2H CURRENT is not the required passed baseline: {h}")
    if h.get("selected_candidate_set") != "coarse_pareto" or _truth(h.get("stage5_started")):
        raise RuntimeError("Stage4.2H candidate/scope contract mismatch")
    h_run = root / str(h["run_relative_path"])
    if not (h_run / ".COMMITTED").is_file() or (h_run / ".RUNNING").exists():
        raise RuntimeError("Stage4.2H baseline is not atomically committed")
    if h_run.name != str(h.get("run_id")):
        raise RuntimeError("Stage4.2H run path and run_id differ")
    h_quality = h_run / "quality_gate.json"
    h_inventory = h_run / "ARTIFACT_INVENTORY.csv"
    if _sha256(h_quality) != h["quality_gate_sha256"] or _sha256(h_inventory) != h["inventory_sha256"]:
        raise RuntimeError("Stage4.2H CURRENT hash binding failed")
    h_inventory_audit = _verify_inventory(h_run, h_inventory)
    h_parents_before = _json(h_run / "parent_pointers_before.json")
    h_parents_after = _json(h_run / "parent_pointers_after.json")
    if h_parents_before != h_parents_after:
        raise RuntimeError("Stage4.2H sealed parent snapshots differ")
    for pointer in (stage3_pointer_path, stage41_pointer_path, g_pointer_path):
        key = _relative(pointer, root)
        captured = h_parents_before.get(key)
        if not captured or not _truth(captured.get("exists")):
            raise RuntimeError(f"Stage4.2H did not seal required parent pointer: {key}")
        if _sha256(pointer) != str(captured.get("sha256", "")).lower():
            raise RuntimeError(f"Live parent pointer differs from Stage4.2H seal: {key}")
    quality = _json(h_quality)
    if not _truth(quality.get("stage4_computational_final")) or not _truth(quality.get("field_validation_pending")):
        raise RuntimeError("Stage4.2H quality scope is not computational-final/field-pending")
    ladder = _json(h_run / "candidate_ladder_decision.json")
    if (
        ladder.get("recommended_candidate_set") != "coarse_pareto"
        or ladder.get("threshold_fingerprint") != THRESHOLD_FINGERPRINT
        or not _truth(ladder.get("final_rung_reached"))
    ):
        raise RuntimeError("Frozen candidate ladder contract mismatch")

    g = _json(g_pointer_path)
    h_metadata = _json(h_run / "metadata.json")
    if g.get("run_id") != h_metadata.get("parent_stage42g_run_id") or not _truth(g.get("passed")):
        raise RuntimeError("Stage4.2G parent binding mismatch")
    g_run = root / str(g["run_relative_path"])
    if g_run.name != str(g.get("run_id")) or not (g_run / ".COMMITTED").is_file():
        raise RuntimeError("Stage4.2G run identity/commit contract failed")
    if (g_run / ".RUNNING").exists() or (g_run / ".FAILED").exists():
        raise RuntimeError("Stage4.2G parent has an invalid terminal marker")
    g_quality = g_run / "quality_gate.json"
    g_inventory = g_run / "ARTIFACT_INVENTORY.csv"
    if _sha256(g_quality) != g["quality_gate_sha256"] or _sha256(g_inventory) != g["inventory_sha256"]:
        raise RuntimeError("Stage4.2G CURRENT hash binding failed")
    g_inventory_audit = _verify_inventory(g_run, g_inventory)

    stage41_pointer = _json(stage41_pointer_path)
    stage41_run = (root / stage41_pointer["metadata_relative_path"]).parent
    stage41_metadata = root / str(stage41_pointer["metadata_relative_path"])
    stage41_inventory = root / str(stage41_pointer["inventory_relative_path"])
    stage41_interface = root / str(stage41_pointer["interface_relative_path"])
    if stage41_run.name != str(stage41_pointer.get("run_id")):
        raise RuntimeError("Stage4.1 run identity contract failed")
    for label, path, field in (
        ("metadata", stage41_metadata, "metadata_sha256"),
        ("inventory", stage41_inventory, "inventory_sha256"),
        ("interface", stage41_interface, "interface_sha256"),
    ):
        if not path.is_file() or _sha256(path) != str(stage41_pointer.get(field, "")).lower():
            raise RuntimeError(f"Stage4.1 {label} pointer hash binding failed")
    stage41_inventory_audit = _verify_inventory(stage41_run, stage41_inventory)
    membership_path = stage41_run / "02_candidate_sensitivity/candidate_set_membership.csv"
    plans = {
        "efficiency": g_run / "candidate_sets/coarse_pareto/plan__efficiency.csv",
        "balanced": g_run / "candidate_sets/coarse_pareto/plan__balanced.csv",
        "equity": h_run / "coarse_equity/plan__equity.csv",
    }
    for policy, path in plans.items():
        frame = pd.read_csv(path, low_memory=False)
        if len(frame) != 20 or frame["venue_id"].astype(str).nunique() != 20:
            raise RuntimeError(f"{policy} plan is not an exact 20-venue plan")
        if not frame["candidate_set"].astype(str).eq("coarse_pareto").all():
            raise RuntimeError(f"{policy} plan is not coarse_pareto")

    membership = pd.read_csv(membership_path, low_memory=False)
    coarse_ids = set(membership.loc[membership["candidate_set"].eq("coarse_pareto"), "venue_id"].astype(str))
    if len(coarse_ids) != 900:
        raise RuntimeError(f"Expected 900 preregistered coarse members, got {len(coarse_ids)}")
    for policy, path in plans.items():
        selected = set(pd.read_csv(path, usecols=["venue_id"])["venue_id"].astype(str))
        if not selected.issubset(coarse_ids):
            raise RuntimeError(f"{policy} plan escapes coarse membership")

    stage3_pointer = _json(stage3_pointer_path)
    stage3_run = (root / str(stage3_pointer["metadata_relative_path"])).parent
    stage3_metadata = root / str(stage3_pointer["metadata_relative_path"])
    stage3_interface = root / str(stage3_pointer["interface_relative_path"])
    stage3_inventory = stage3_run / "STAGE3_ARTIFACT_INVENTORY.csv"
    if stage3_run.name != str(stage3_pointer.get("run_id")):
        raise RuntimeError("Stage3 run identity contract failed")
    if _sha256(stage3_metadata) != str(stage3_pointer.get("metadata_sha256", "")).lower():
        raise RuntimeError("Stage3 metadata hash binding failed")
    if _sha256(stage3_interface) != str(stage3_pointer.get("interface_sha256", "")).lower():
        raise RuntimeError("Stage3 interface hash binding failed")
    if _sha256(stage3_inventory) != str(stage3_pointer.get("artifact_inventory_sha256", "")).lower():
        raise RuntimeError("Stage3 inventory hash binding failed")
    # Stage3's historical inventory stores project-root-relative paths, unlike
    # the later run-relative inventories.
    stage3_inventory_audit = _verify_inventory(root, stage3_inventory)

    solver_audit = _json(h_run / "coarse_equity/coarse_equity_solver_audit.json")
    if int(solver_audit.get("candidate_count", -1)) != 896:
        raise RuntimeError("Frozen solver candidate count is not 896")
    if int(solver_audit.get("pattern_count", -1)) != 10_838:
        raise RuntimeError("Frozen compressed pattern count is not 10,838")

    baseline = Baseline(
        root=root,
        h_pointer_path=h_pointer_path,
        h_pointer=h,
        h_run=h_run,
        g_pointer_path=g_pointer_path,
        g_pointer=g,
        g_run=g_run,
        stage3_pointer_path=stage3_pointer_path,
        stage41_pointer_path=stage41_pointer_path,
        stage41_run=stage41_run,
        plan_paths=plans,
        membership_path=membership_path,
    )
    audit = {
        "stage42h": {
            "run_id": h["run_id"],
            "decision": h["decision"],
            "current_sha256": _sha256(h_pointer_path),
            "quality_gate_sha256": _sha256(h_quality),
            "inventory": h_inventory_audit,
        },
        "stage42g": {
            "run_id": g["run_id"],
            "decision": g["decision"],
            "current_sha256": _sha256(g_pointer_path),
            "quality_gate_sha256": _sha256(g_quality),
            "inventory": g_inventory_audit,
        },
        "stage3_pointer_sha256": _sha256(stage3_pointer_path),
        "stage3_inventory": stage3_inventory_audit,
        "stage41_pointer_sha256": _sha256(stage41_pointer_path),
        "stage41_inventory": stage41_inventory_audit,
        "h_sealed_parent_snapshot_sha256": _sha256(h_run / "parent_pointers_before.json"),
        "candidate_membership_sha256": _sha256(membership_path),
        "preregistered_coarse_member_count": len(coarse_ids),
        "solver_candidate_count": int(solver_audit["candidate_count"]),
        "compressed_pattern_count": int(solver_audit["pattern_count"]),
        "coverage_nnz": int(solver_audit["coverage_nnz"]),
        "threshold_fingerprint": ladder["threshold_fingerprint"],
        "plan_sha256": {policy: _sha256(path) for policy, path in plans.items()},
    }
    return baseline, audit


def _parse_highs_log(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    text = path.read_text(encoding="utf-8", errors="replace")
    result: dict[str, Any] = {}
    for target, pattern in (
        ("direct_objective_value", r"Primal bound\s+([-+0-9.eE]+)"),
        ("direct_best_bound", r"Dual bound\s+([-+0-9.eE]+)"),
        ("direct_relative_gap", r"Gap\s+([-+0-9.eE]+)%"),
    ):
        matches = re.findall(pattern, text)
        if matches:
            value = float(matches[-1])
            result[target] = value / 100.0 if target == "direct_relative_gap" else value
    status = re.findall(r"Status\s+([^\r\n]+)", text)
    if status:
        result["direct_status"] = status[-1].strip()
    return result


def _direct_gap(objective: float, bound: float, sense: str) -> float:
    if str(sense).lower() == "min":
        directional = objective - bound
    else:
        directional = bound - objective
    return max(0.0, float(directional)) / max(abs(float(objective)), 1e-12)


def _certificate_proof(
    baseline: Baseline,
    *,
    certificate: str,
    scenario: str,
    stage_index: int,
    objective: str,
    source_artifact: Path,
    source_stage: dict[str, Any],
) -> dict[str, Any]:
    if certificate in {"OPTIMAL", "DIRECT_MIP_GAP"}:
        return {
            "certificate_proof_status": "DIRECT_SOLVER_BOUND",
            "certificate_proof_artifact": None,
            "certificate_proof_sha256": None,
            "certificate_threshold": None,
        }
    if certificate == "INFEASIBILITY_ORACLE":
        history = source_stage.get("details", {}).get("oracle_history", [])
        if not history or not bool(history[-1].get("is_infeasible")):
            raise RuntimeError(f"Missing infeasibility-oracle proof for {scenario} stage {stage_index}")
        proof_path = source_artifact
        status = str(history[-1].get("status"))
        threshold = history[-1].get("threshold")
    elif certificate == "HIGHS_CANDIDATE_ONLY_RELAXATION_INFEASIBILITY":
        proof_path = (
            baseline.h_run
            / f"coarse_equity/scip_rescue_pass_1/stage_{stage_index:02d}_{objective}/strengthened_highs_result.json"
        )
        proof = _json(proof_path)
        if str(proof.get("status")) != "INFEASIBLE" or not _truth(proof.get("is_infeasible")):
            raise RuntimeError(f"Invalid strengthened HiGHS proof for {scenario} stage {stage_index}")
        status = "INFEASIBLE"
        threshold = None
    elif certificate == "SCIP_THRESHOLD_INFEASIBILITY":
        proof_path = (
            baseline.h_run
            / f"coarse_equity/scip_rescue_pass_1/stage_{stage_index:02d}_{objective}/portfolio_summary.json"
        )
        proof = _json(proof_path)
        terminal = proof.get("terminal", {})
        worker = terminal.get("worker", {})
        if terminal.get("kind") != "INFEASIBLE_PROOF" or worker.get("status") != "INFEASIBLE":
            raise RuntimeError(f"Invalid SCIP threshold proof for {scenario} stage {stage_index}")
        status = "INFEASIBLE_PROOF"
        threshold = None
    else:  # defensive: callers also enforce the allowlist
        raise RuntimeError(f"Unsupported certificate {certificate!r}")
    return {
        "certificate_proof_status": status,
        "certificate_proof_artifact": _relative(proof_path, baseline.root),
        "certificate_proof_sha256": _sha256(proof_path),
        "certificate_threshold": threshold,
    }


def build_canonical_computational_summary(baseline: Baseline) -> dict[str, Any]:
    certification_path = baseline.h_run / "stage4_computational_stage_certification.csv"
    stages = pd.read_csv(certification_path, low_memory=False)
    if len(stages) != 12 or not stages["certified"].map(_truth).all():
        raise RuntimeError("The authoritative Stage 4 certification ledger is not 12/12")
    if set(stages["certificate"].astype(str)) - ALLOWED_CERTIFICATES:
        raise RuntimeError(
            "Unknown certificate kinds: "
            f"{sorted(set(stages['certificate'].astype(str)) - ALLOWED_CERTIFICATES)}"
        )
    for scenario, objectives in EXPECTED_STAGE_OBJECTIVES.items():
        group = stages.loc[stages["scenario"].astype(str).eq(scenario)].sort_values("stage_index")
        if group["stage_index"].astype(int).tolist() != [1, 2, 3, 4]:
            raise RuntimeError(f"{scenario} certification ledger is not exactly stages 1..4")
        if group["objective_name"].astype(str).tolist() != objectives:
            raise RuntimeError(f"{scenario} objective order differs from the frozen contract")
    rows: list[dict[str, Any]] = []
    stage_sources: dict[Path, list[dict[str, Any]]] = {}
    for row in stages.to_dict("records"):
        scenario = str(row["scenario"])
        index = int(row["stage_index"])
        objective = str(row["objective_name"])
        if scenario == "equity":
            source_run = baseline.h_pointer["run_id"]
            source_artifact = baseline.h_run / "coarse_equity/stages__equity.json"
            direct_log = baseline.h_run / f"coarse_equity/solve_pass_1/coarse_pareto/equity/stage_{index:02d}_{objective}/direct.highs.log"
        else:
            source_run = baseline.g_pointer["run_id"]
            source_artifact = baseline.g_run / f"candidate_sets/coarse_pareto/stages__{scenario}.json"
            direct_log = baseline.g_run / f"candidate_sets/coarse_pareto/{scenario}/stage_{index:02d}_{objective}/direct.highs.log"
        if source_artifact not in stage_sources:
            payload = json.loads(source_artifact.read_text(encoding="utf-8-sig"))
            if not isinstance(payload, list):
                raise RuntimeError(f"Stage source is not a list: {source_artifact}")
            stage_sources[source_artifact] = payload
        matches = [
            value
            for value in stage_sources[source_artifact]
            if int(value.get("stage_index", -1)) == index
            and str(value.get("objective_name")) == objective
        ]
        if len(matches) != 1:
            raise RuntimeError(f"Cannot bind source stage for {scenario}/{index}/{objective}")
        source_stage = matches[0]
        if str(source_stage.get("certificate")) != str(row["certificate"]):
            raise RuntimeError(f"Certificate ledger/source mismatch for {scenario} stage {index}")

        displayed_direct = _parse_highs_log(direct_log)
        exact_direct = source_stage.get("details", {}).get("direct", {})
        direct_objective = float(
            exact_direct.get(
                "objective_value",
                displayed_direct.get("direct_objective_value", row["objective_value"]),
            )
        )
        direct_bound = float(
            exact_direct.get(
                "best_bound",
                displayed_direct.get("direct_best_bound", row["best_bound"]),
            )
        )
        direct_gap = float(
            exact_direct.get(
                "authoritative_gap_after_seed_improvement",
                exact_direct.get(
                    "relative_gap",
                    _direct_gap(direct_objective, direct_bound, str(row["sense"])),
                ),
            )
        )
        certificate = str(row["certificate"])
        row_gap = float(row["relative_gap"])
        rescue = certificate in {
            "INFEASIBILITY_ORACLE",
            "HIGHS_CANDIDATE_ONLY_RELAXATION_INFEASIBILITY",
            "SCIP_THRESHOLD_INFEASIBILITY",
        }
        if row_gap > 0.005 + 1e-12:
            raise RuntimeError(f"Certified gap exceeds 0.005 for {scenario} stage {index}")
        certified_bound = 0.0 if certificate == "OPTIMAL" else row_gap
        proof = _certificate_proof(
            baseline,
            certificate=certificate,
            scenario=scenario,
            stage_index=index,
            objective=objective,
            source_artifact=source_artifact,
            source_stage=source_stage,
        )
        payload = {
                "candidate_set": "coarse_pareto",
                "scenario": scenario,
                "stage_index": index,
                "objective": objective,
                "sense": row["sense"],
                "objective_value": float(row["objective_value"]),
                "direct_objective_value": float(direct_objective),
                "direct_best_bound": float(direct_bound),
                "direct_relative_gap": float(direct_gap),
                "direct_reported_gap": displayed_direct.get("direct_relative_gap"),
                "direct_status": exact_direct.get(
                    "status", displayed_direct.get("direct_status", "RECORDED_STAGE_TELEMETRY")
                ),
                "certified_gap_upper_bound": float(certified_bound),
                "certificate_kind": certificate,
                "final_certified": True,
                "final_certification_class": "OPTIMAL" if certificate == "OPTIMAL" else "CERTIFIED_NEAR_OPTIMAL",
                "source_run_id": source_run,
                "source_artifact": _relative(source_artifact, baseline.root),
                "source_sha256": _sha256(source_artifact),
                "direct_source_artifact": _relative(direct_log, baseline.root),
                "direct_source_sha256": _sha256(direct_log),
                "rescue_certificate_used": rescue,
            }
        payload.update(proof)
        rows.append(payload)
    return {
        "schema": "MEDIROAD_STAGE4_COMPUTATIONAL_FINAL_CANONICAL_V1",
        "decision": EXPECTED_H_DECISION,
        "candidate_universe": "coarse_pareto",
        "stage_count": 12,
        "certified_stage_count": 12,
        "stage4_computational_final": True,
        "field_validation_pending": True,
        "operational_final": False,
        "stage5_started": False,
        "authoritative_ledger": _relative(certification_path, baseline.root),
        "authoritative_ledger_sha256": _sha256(certification_path),
        "stages": sorted(rows, key=lambda x: (POLICIES.index(x["scenario"]), x["stage_index"])),
    }


def _coverage_contributions(
    baseline: Baseline,
    plans: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stage3 = discover_stage3(baseline.root)
    interface = normalize_stage3_interface(read_table(stage3.interface))
    venues = venue_table_from_interface(interface)
    coverage_all = load_coverage(stage3, "hard_5000m")
    grid = normalize_grid_policy(read_table(stage3.grid_policy))
    grid = grid.set_index("grid_id").loc[coverage_all.grid_ids.astype(str)].reset_index()
    population = grid["elderly_population"].to_numpy(float)
    need = grid["need_weighted_population"].to_numpy(float)
    high = grid["high_need_population"].to_numpy(float)
    records: list[dict[str, Any]] = []
    for policy, plan in plans.items():
        ids = plan["venue_id"].astype(str).to_numpy()
        matrix = align_coverage(coverage_all, ids, grid["grid_id"].astype(str).to_numpy())
        counts = np.asarray(matrix.sum(axis=0)).ravel()
        for index, venue_id in enumerate(ids):
            cols = matrix.getrow(index).indices
            unique_cols = cols[counts[cols] == 1]
            records.append(
                {
                    "venue_id": venue_id,
                    "policy": policy,
                    "unique_coverage_contribution": float(population[unique_cols].sum()),
                    "unique_need_weighted_contribution": float(need[unique_cols].sum()),
                    "unique_high_need_contribution": float(high[unique_cols].sum()),
                    "covered_grid_count": int(len(cols)),
                    "uniquely_covered_grid_count": int(len(unique_cols)),
                }
            )
    bundle = (
        interface.groupby("venue_id", as_index=False)
        .agg(
            specialty_gap_max=("specialty_gap", "max"),
            specialty_gap_mean=("specialty_gap", "mean"),
            bundle_gap_weighted_exposure_max=("bundle_gap_weighted_exposure", "max"),
        )
    )
    return pd.DataFrame(records), venues, bundle


def _clean_values(values: Iterable[Any]) -> list[str]:
    return sorted({str(value).strip() for value in values if pd.notna(value) and str(value).strip() and str(value).strip().lower() != "nan"})


def _ordered_clean_values(values: Iterable[Any]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if pd.isna(value):
            continue
        text = str(value).strip()
        if not text or text.lower() == "nan" or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def build_selected_union(baseline: Baseline) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    plans = {policy: pd.read_csv(path, low_memory=False) for policy, path in baseline.plan_paths.items()}
    contributions, venues, bundle = _coverage_contributions(baseline, plans)
    master_path = baseline.root / "08_candidate_venues/candidate_venues_model_ready_final_v6.csv"
    master = pd.read_csv(master_path, low_memory=False).drop_duplicates("venue_id")
    master["venue_id"] = master["venue_id"].astype(str)
    master_cols = [
        "venue_id",
        "address",
        "coordinate_source",
        "source_kind",
        "previous_outreach_evidence",
        "operational_suitability_verified",
        "operational_suitability_status",
        "official_location_flag",
        "official_village_senior_flag",
        "previous_operation_evidence_flag",
    ]
    master = master[[col for col in master_cols if col in master.columns]]
    all_rows = pd.concat(plans.values(), ignore_index=True)
    rows: list[dict[str, Any]] = []
    for venue_id, group in all_rows.groupby(all_rows["venue_id"].astype(str), sort=True):
        first = group.iloc[0]
        selected = sorted(set(group["scenario"].astype(str)), key=POLICIES.index)
        ordered_group = group.assign(
            _policy_order=group["scenario"].astype(str).map(POLICIES.index)
        ).sort_values("_policy_order")
        fallback_values: list[Any] = []
        for fallback_row in ordered_group.to_dict("records"):
            fallback_values.extend(
                [fallback_row.get("fallback_venue_1"), fallback_row.get("fallback_venue_2")]
            )
        payload = first.to_dict()
        payload.update(
            {
                "venue_id": venue_id,
                "selected_efficiency": "efficiency" in selected,
                "selected_balanced": "balanced" in selected,
                "selected_equity": "equity" in selected,
                "selected_policies": "|".join(selected),
                "selection_frequency": len(selected),
                "policy_intersection_class": "ALL_THREE" if len(selected) == 3 else ("TWO_POLICIES" if len(selected) == 2 else "SINGLE_POLICY"),
                "is_busproxy_spatial_anchor": venue_id.startswith("BUSPROXY_"),
                "physical_venue_resolution_status": "SPATIAL_ANCHOR_REQUIRES_PHYSICAL_RESOLUTION" if venue_id.startswith("BUSPROXY_") else "PHYSICAL_IDENTITY_FIELD_VALIDATION_PENDING",
                "current_field_validation_status": "UNKNOWN_NOT_FIELD_VERIFIED",
                "fallback_venue_ids": "|".join(_ordered_clean_values(fallback_values)),
            }
        )
        for policy in POLICIES:
            row = contributions.loc[(contributions["venue_id"].eq(venue_id)) & (contributions["policy"].eq(policy))]
            for metric in ("unique_coverage_contribution", "unique_need_weighted_contribution", "unique_high_need_contribution"):
                payload[f"{metric}__{policy}"] = float(row.iloc[0][metric]) if len(row) else 0.0
        rows.append(payload)
    union = pd.DataFrame(rows).merge(bundle, on="venue_id", how="left", validate="one_to_one")
    union = union.merge(master, on="venue_id", how="left", validate="one_to_one")
    union = union.sort_values(["selection_frequency", "is_busproxy_spatial_anchor", "venue_id"], ascending=[False, True, True]).reset_index(drop=True)
    catalog = venues.merge(master, on="venue_id", how="left", validate="one_to_one")
    contact_lookup: dict[str, str] = {}
    hira_path = baseline.root / "01_public_sources/v2_full/derived/chungbuk_hira_medical_facilities_master_202606.csv"
    if hira_path.is_file():
        hira = pd.read_csv(hira_path, usecols=["facility_id", "phone"], dtype=str)
        contact_lookup.update(
            {
                f"HIRA_{facility_id}": str(phone).strip()
                for facility_id, phone in hira[["facility_id", "phone"]].itertuples(index=False, name=None)
                if pd.notna(phone) and str(phone).strip()
            }
        )
    vhsc_path = baseline.root / "14_v5_external_data/venues/chungbuk_official_village_halls_senior_centers_v5_2.csv"
    if vhsc_path.is_file():
        vhsc = pd.read_csv(vhsc_path, usecols=["venue_id", "telephone"], dtype=str)
        contact_lookup.update(
            {
                str(venue_id): str(phone).strip()
                for venue_id, phone in vhsc[["venue_id", "telephone"]].itertuples(index=False, name=None)
                if pd.notna(phone) and str(phone).strip()
            }
        )
    catalog["contact_if_known"] = catalog["venue_id"].map(contact_lookup).fillna("")
    return union, plans, catalog, contributions


def _haversine_km(lat1: float, lon1: float, lat2: pd.Series, lon2: pd.Series) -> pd.Series:
    radius = 6371.0088
    p1 = math.radians(float(lat1))
    p2 = np.radians(pd.to_numeric(lat2, errors="coerce").to_numpy(float))
    dlat = p2 - p1
    dlon = np.radians(pd.to_numeric(lon2, errors="coerce").to_numpy(float) - float(lon1))
    value = np.sin(dlat / 2.0) ** 2 + math.cos(p1) * np.cos(p2) * np.sin(dlon / 2.0) ** 2
    return pd.Series(2.0 * radius * np.arcsin(np.sqrt(np.clip(value, 0, 1))), index=lat2.index)


def _identity_class(venue_id: str, row: pd.Series | dict[str, Any]) -> tuple[str, bool]:
    source = str(row.get("source_kind", ""))
    coordinate_source = str(row.get("coordinate_source", "")).strip().lower()
    address = str(row.get("address", "")).strip()
    non_exact_location = (
        not address
        or "proxy" in coordinate_source
        or "centroid" in coordinate_source
        or "not_exact" in coordinate_source
    )
    if non_exact_location:
        if venue_id.startswith("mobile_clinic_") or venue_id.startswith("rural_"):
            return "EXISTING_SERVICE_LOCATION_UNRESOLVED", False
        return "AMBIGUOUS", False
    if venue_id.startswith("OFFICIAL_VHSC_"):
        return "OFFICIAL_ID_EXACT", True
    if venue_id.startswith("HIRA_") or venue_id.startswith("NMC_"):
        return "OFFICIAL_ID_EXACT", True
    if venue_id.startswith("mobile_clinic_") or venue_id.startswith("rural_"):
        return "EXISTING_SERVICE_CONFIRMED", True
    if source.startswith("official_") or _truth(row.get("official_location_flag", False)):
        return "OFFICIAL_ID_EXACT", True
    return "NAME_COORDINATE_STRONG_MATCH", False


def _apply_narrow_identity_evidence(
    catalog: pd.DataFrame,
    evidence: pd.DataFrame,
    *,
    evidence_relative_path: str = NARROW_IDENTITY_EVIDENCE_RELATIVE_PATH,
) -> pd.DataFrame:
    """Overlay narrowly researched identity facts without promoting coordinates.

    The overlay can establish a facility's current official name/address/contact
    listing and existence.  A secondary facility point is retained only as a
    field-work hint: it never replaces the frozen candidate coordinate and can
    never satisfy the exact-coordinate or operational release gates.
    """

    required = {
        "venue_id",
        "venue_name",
        "official_address",
        "contact_if_known",
        "address_evidenced",
        "facility_exists_evidenced",
        "official_facility_source_url",
        "official_dataset_source_url",
        "official_public_toilet_source_url",
        "provisional_latitude",
        "provisional_longitude",
        "provisional_coordinate_source",
        "provisional_coordinate_source_url",
        "provisional_coordinate_exact_for_release",
        "source_retrieved_date",
        "limitations",
    }
    missing = sorted(required - set(evidence.columns))
    if missing:
        raise RuntimeError(f"Narrow identity evidence missing columns: {missing}")
    if evidence.empty:
        raise RuntimeError("Narrow identity evidence must not be empty")

    work = evidence.copy()
    work["venue_id"] = work["venue_id"].fillna("").astype(str).str.strip()
    if work["venue_id"].eq("").any() or work["venue_id"].duplicated().any():
        raise RuntimeError("Narrow identity evidence venue IDs must be nonblank and unique")
    result = catalog.drop_duplicates("venue_id").copy()
    result["venue_id"] = result["venue_id"].astype(str)
    unknown = sorted(set(work["venue_id"]) - set(result["venue_id"]))
    if unknown:
        raise RuntimeError(f"Narrow identity evidence references unknown venue IDs: {unknown}")

    defaults: dict[str, Any] = {
        "narrow_identity_evidence_applied": False,
        "identity_address_evidenced": False,
        "facility_exists_evidenced": False,
        "identity_source_reference": "",
        "identity_source_urls": "",
        "provisional_facility_latitude": np.nan,
        "provisional_facility_longitude": np.nan,
        "provisional_coordinate_source": "",
        "provisional_coordinate_source_url": "",
        "provisional_coordinate_exact_for_release": False,
        "identity_evidence_limitations": "",
    }
    for column, default in defaults.items():
        if column not in result.columns:
            result[column] = default

    for _, item in work.iterrows():
        venue_id = str(item["venue_id"])
        row_index = result.index[result["venue_id"].eq(venue_id)]
        if len(row_index) != 1:
            raise RuntimeError(f"Narrow identity evidence target is not one-to-one: {venue_id}")
        index = row_index[0]
        if str(result.at[index, "venue_name"]).strip() != str(item["venue_name"]).strip():
            raise RuntimeError(f"Narrow identity evidence venue name mismatch: {venue_id}")

        address = str(item["official_address"]).strip()
        contact = str(item["contact_if_known"]).strip()
        official_urls = [
            str(item["official_facility_source_url"]).strip(),
            str(item["official_dataset_source_url"]).strip(),
            str(item["official_public_toilet_source_url"]).strip(),
        ]
        coordinate_url = str(item["provisional_coordinate_source_url"]).strip()
        if not address or not all(value.startswith("https://") for value in official_urls):
            raise RuntimeError(f"Narrow identity evidence lacks official address/source: {venue_id}")
        if not coordinate_url.startswith("https://"):
            raise RuntimeError(f"Narrow identity evidence lacks coordinate provenance: {venue_id}")
        latitude = pd.to_numeric(pd.Series([item["provisional_latitude"]]), errors="coerce").iloc[0]
        longitude = pd.to_numeric(pd.Series([item["provisional_longitude"]]), errors="coerce").iloc[0]
        if not (np.isfinite(latitude) and np.isfinite(longitude)):
            raise RuntimeError(f"Narrow identity evidence provisional coordinate is invalid: {venue_id}")
        if not (33.0 <= float(latitude) <= 39.5 and 124.0 <= float(longitude) <= 132.0):
            raise RuntimeError(f"Narrow identity evidence coordinate is outside Korea: {venue_id}")
        if _truth(item["provisional_coordinate_exact_for_release"]):
            raise RuntimeError(
                "Narrow identity evidence cannot promote a secondary facility point "
                f"to an exact release coordinate: {venue_id}"
            )

        result.at[index, "address"] = address
        result.at[index, "contact_if_known"] = contact
        result.at[index, "narrow_identity_evidence_applied"] = True
        result.at[index, "identity_address_evidenced"] = _truth(item["address_evidenced"])
        result.at[index, "facility_exists_evidenced"] = _truth(
            item["facility_exists_evidenced"]
        )
        result.at[index, "identity_source_reference"] = evidence_relative_path
        result.at[index, "identity_source_urls"] = "|".join([*official_urls, coordinate_url])
        result.at[index, "provisional_facility_latitude"] = float(latitude)
        result.at[index, "provisional_facility_longitude"] = float(longitude)
        result.at[index, "provisional_coordinate_source"] = str(
            item["provisional_coordinate_source"]
        ).strip()
        result.at[index, "provisional_coordinate_source_url"] = coordinate_url
        result.at[index, "provisional_coordinate_exact_for_release"] = False
        result.at[index, "identity_evidence_limitations"] = str(item["limitations"]).strip()
    return result


def _priority(frequency: int, relation: str, ordinal: int) -> str:
    if relation == "DIRECT_SELECTED_PHYSICAL":
        return "P0" if frequency == 3 else ("P1" if frequency == 2 else "P2")
    if relation.startswith("BUSPROXY_"):
        base = 0 if frequency == 3 else (1 if frequency == 2 else 2)
        return f"P{min(3, base + max(0, ordinal - 1))}"
    return "P2" if frequency >= 2 and ordinal == 1 else "P3"


def build_field_queue(
    union: pd.DataFrame,
    catalog: pd.DataFrame,
    membership: pd.DataFrame,
    *,
    queue_min: int,
    queue_max: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    catalog = catalog.drop_duplicates("venue_id").copy()
    catalog["venue_id"] = catalog["venue_id"].astype(str)
    by_id = catalog.set_index("venue_id", drop=False)
    coarse = set(membership.loc[membership["candidate_set"].eq("coarse_pareto"), "venue_id"].astype(str))
    relationships: list[dict[str, Any]] = []

    def add(anchor: pd.Series, candidate_id: str, relation: str, ordinal: int, required_for_bus: bool) -> bool:
        if not candidate_id or candidate_id.startswith("BUSPROXY_") or candidate_id not in by_id.index:
            return False
        candidate = by_id.loc[candidate_id]
        # A BUS spatial anchor may only point at a facility whose physical
        # identity and coordinates are exact enough to investigate.  Historical
        # service rows located at an admin centroid remain useful context, but
        # are not physical fallback candidates.
        if required_for_bus and not _identity_class(candidate_id, candidate)[1]:
            return False
        distance = float(
            _haversine_km(
                float(anchor["latitude"]),
                float(anchor["longitude"]),
                pd.Series([candidate["latitude"]]),
                pd.Series([candidate["longitude"]]),
            ).iloc[0]
        )
        relationships.append(
            {
                "source_anchor_id": str(anchor["venue_id"]),
                "source_anchor_is_busproxy": bool(anchor["is_busproxy_spatial_anchor"]),
                "source_anchor_cluster_id": str(anchor["cluster_id"]),
                "source_anchor_admin_code": str(anchor["admin_code"]),
                "source_anchor_policies": str(anchor["selected_policies"]),
                "source_anchor_selection_frequency": int(anchor["selection_frequency"]),
                "physical_candidate_id": candidate_id,
                "relationship": relation,
                "relationship_ordinal": ordinal,
                "same_coverage_cluster": str(candidate.get("cluster_id", "")) == str(anchor["cluster_id"]),
                "same_admin": str(candidate.get("admin_code", "")) == str(anchor["admin_code"]),
                "anchor_to_candidate_distance_km": distance,
                "coverage_recompute_required": candidate_id != str(anchor["venue_id"]),
                "required_for_busproxy_resolution": required_for_bus,
                "priority": _priority(int(anchor["selection_frequency"]), relation, ordinal),
            }
        )
        return True

    for _, anchor in union.iterrows():
        anchor_id = str(anchor["venue_id"])
        is_bus = bool(anchor["is_busproxy_spatial_anchor"])
        fallbacks = str(anchor.get("fallback_venue_ids", "")).split("|") if str(anchor.get("fallback_venue_ids", "")) else []
        nonbus_fallbacks = [value for value in fallbacks if value and not value.startswith("BUSPROXY_") and value in by_id.index]
        same_cluster = [value for value in nonbus_fallbacks if str(by_id.loc[value].get("cluster_id", "")) == str(anchor["cluster_id"])]
        if not is_bus:
            add(anchor, anchor_id, "DIRECT_SELECTED_PHYSICAL", 1, False)
            fallback = same_cluster[0] if same_cluster else (nonbus_fallbacks[0] if nonbus_fallbacks else None)
            if fallback:
                add(anchor, fallback, "NONBUS_SELECTED_DECLARED_FALLBACK", 1, False)
            continue

        added: list[str] = []
        for candidate_id in same_cluster[:2]:
            if add(anchor, candidate_id, "BUSPROXY_DECLARED_SAME_CLUSTER_FALLBACK", len(added) + 1, True):
                added.append(candidate_id)
        if len(added) < 2:
            alternatives = catalog.loc[
                ~catalog["venue_id"].str.upper().str.startswith("BUSPROXY_")
                & catalog["cluster_id"].astype(str).eq(str(anchor["cluster_id"]))
                & ~catalog["venue_id"].isin(added)
            ].copy()
            alternatives = alternatives.sort_values(
                ["venue_readiness_prior", "raw_elderly_exposure", "venue_id"],
                ascending=[False, False, True],
            )
            for candidate_id in alternatives["venue_id"]:
                if add(anchor, str(candidate_id), "BUSPROXY_COARSE_SAME_CLUSTER_ALTERNATIVE", len(added) + 1, True):
                    added.append(str(candidate_id))
                if len(added) >= 2:
                    break
        if len(added) < 2:
            alternatives = catalog.loc[
                ~catalog["venue_id"].str.upper().str.startswith("BUSPROXY_")
                & catalog["admin_code"].astype(str).eq(str(anchor["admin_code"]))
                & ~catalog["venue_id"].isin(added)
            ].copy()
            if len(alternatives):
                alternatives["_distance"] = _haversine_km(
                    float(anchor["latitude"]), float(anchor["longitude"]), alternatives["latitude"], alternatives["longitude"]
                )
                alternatives = alternatives.loc[alternatives["_distance"].le(5.0)].sort_values(
                    ["venue_readiness_prior", "_distance", "raw_elderly_exposure", "venue_id"],
                    ascending=[False, True, False, True],
                )
                for candidate_id in alternatives["venue_id"]:
                    if add(anchor, str(candidate_id), "BUSPROXY_NEAREST_SAME_ADMIN_MANUAL_CANDIDATE", len(added) + 1, True):
                        added.append(str(candidate_id))
                    if len(added) >= 2:
                        break

    mapping = pd.DataFrame(relationships)
    if mapping.empty:
        raise RuntimeError("No physical resolution candidates could be generated")
    mapping["_rank"] = mapping["priority"].map(PRIORITY_RANK)
    mapping = mapping.sort_values(
        ["_rank", "source_anchor_selection_frequency", "relationship_ordinal", "physical_candidate_id"],
        ascending=[True, False, True, True],
    ).drop(columns="_rank").reset_index(drop=True)

    queue_rows: list[dict[str, Any]] = []
    for candidate_id, group in mapping.groupby("physical_candidate_id", sort=False):
        candidate = by_id.loc[candidate_id]
        priority = min(group["priority"], key=PRIORITY_RANK.get)
        resolution_class, identity_evidenced = _identity_class(candidate_id, candidate)
        coordinate_recompute_required = resolution_class in {
            "EXISTING_SERVICE_LOCATION_UNRESOLVED",
            "AMBIGUOUS",
        }
        queue_rows.append(
            {
                "priority": priority,
                "operational_candidate_id": candidate_id,
                "venue_id": candidate_id,
                "venue_name": candidate.get("venue_name", "UNKNOWN"),
                "venue_type": candidate.get("venue_type", "UNKNOWN"),
                "address": candidate.get("address", ""),
                "latitude": candidate.get("latitude", np.nan),
                "longitude": candidate.get("longitude", np.nan),
                "admin_code": candidate.get("admin_code", ""),
                "sigungu": candidate.get("sigungu", ""),
                "cluster_id": candidate.get("cluster_id", ""),
                "source_anchor_ids": "|".join(sorted(set(group["source_anchor_id"].astype(str)))),
                "policies_impacted": "|".join(_clean_values("|".join(group["source_anchor_policies"]).split("|"))),
                "anchor_relationships": "|".join(sorted(set(group["relationship"].astype(str)))),
                "max_anchor_selection_frequency": int(group["source_anchor_selection_frequency"].max()),
                "required_for_busproxy_resolution": bool(group["required_for_busproxy_resolution"].any()),
                "resolution_class": resolution_class,
                "identity_evidenced": identity_evidenced,
                "resolution_status": "IDENTITY_EVIDENCED_FIELD_VALIDATION_PENDING" if identity_evidenced else "MANUAL_IDENTITY_AND_FIELD_VALIDATION_REQUIRED",
                "coverage_recompute_required": bool(
                    group["coverage_recompute_required"].any()
                    or coordinate_recompute_required
                ),
                "minimum_anchor_distance_km": float(group["anchor_to_candidate_distance_km"].min()),
                "source_kind": candidate.get("source_kind", ""),
                "coordinate_source": candidate.get("coordinate_source", ""),
                "contact_if_known": candidate.get("contact_if_known", ""),
                "narrow_identity_evidence_applied": bool(
                    _truth(candidate.get("narrow_identity_evidence_applied", False))
                ),
                "identity_address_evidenced": bool(
                    _truth(candidate.get("identity_address_evidenced", False))
                ),
                "facility_exists_evidenced": bool(
                    _truth(candidate.get("facility_exists_evidenced", False))
                ),
                "identity_source_reference": candidate.get("identity_source_reference", ""),
                "identity_source_urls": candidate.get("identity_source_urls", ""),
                "provisional_facility_latitude": candidate.get(
                    "provisional_facility_latitude", np.nan
                ),
                "provisional_facility_longitude": candidate.get(
                    "provisional_facility_longitude", np.nan
                ),
                "provisional_coordinate_source": candidate.get(
                    "provisional_coordinate_source", ""
                ),
                "provisional_coordinate_source_url": candidate.get(
                    "provisional_coordinate_source_url", ""
                ),
                "provisional_coordinate_exact_for_release": False,
                "identity_evidence_limitations": candidate.get(
                    "identity_evidence_limitations", ""
                ),
                "venue_readiness_prior": candidate.get("venue_readiness_prior", np.nan),
                "raw_elderly_exposure": candidate.get("raw_elderly_exposure", np.nan),
                "need_weighted_exposure": candidate.get("need_weighted_exposure", np.nan),
                "high_need_elderly_exposure": candidate.get("high_need_elderly_exposure", np.nan),
                "structural_need": candidate.get("structural_need", np.nan),
            }
        )
    queue = pd.DataFrame(queue_rows)
    queue["_rank"] = queue["priority"].map(PRIORITY_RANK)
    queue = queue.sort_values(
        ["_rank", "required_for_busproxy_resolution", "max_anchor_selection_frequency", "venue_readiness_prior", "venue_id"],
        ascending=[True, False, False, False, True],
    ).drop(columns="_rank").reset_index(drop=True)

    protected_ids = set(
        mapping.loc[
            mapping["relationship"].eq("DIRECT_SELECTED_PHYSICAL")
            | (mapping["required_for_busproxy_resolution"] & mapping["relationship_ordinal"].eq(1)),
            "physical_candidate_id",
        ]
    )
    if len(queue) > queue_max:
        protected = queue.loc[queue["venue_id"].isin(protected_ids)]
        if len(protected) > queue_max:
            raise RuntimeError(
                f"Protected physical candidates exceed queue_max: {len(protected)}>{queue_max}"
            )
        remainder = queue.loc[~queue["venue_id"].isin(protected_ids)].head(max(0, queue_max - len(protected)))
        queue = pd.concat([protected, remainder], ignore_index=True)
    if len(queue) < queue_min:
        extras = catalog.loc[
            catalog["venue_id"].isin(coarse)
            & ~catalog["venue_id"].str.upper().str.startswith("BUSPROXY_")
            & ~catalog["venue_id"].isin(queue["venue_id"])
            & catalog["sigungu"].isin(union["sigungu"])
        ].sort_values(["venue_readiness_prior", "raw_elderly_exposure", "venue_id"], ascending=[False, False, True])
        extra_rows = []
        for _, candidate in extras.head(queue_min - len(queue)).iterrows():
            resolution_class, identity_evidenced = _identity_class(str(candidate["venue_id"]), candidate)
            extra_rows.append(
                {
                    "priority": "P3",
                    "operational_candidate_id": candidate["venue_id"],
                    "venue_id": candidate["venue_id"],
                    "venue_name": candidate.get("venue_name", "UNKNOWN"),
                    "venue_type": candidate.get("venue_type", "UNKNOWN"),
                    "address": candidate.get("address", ""),
                    "latitude": candidate.get("latitude", np.nan),
                    "longitude": candidate.get("longitude", np.nan),
                    "admin_code": candidate.get("admin_code", ""),
                    "sigungu": candidate.get("sigungu", ""),
                    "cluster_id": candidate.get("cluster_id", ""),
                    "source_anchor_ids": "",
                    "policies_impacted": "OPERATIONAL_POOL_RESERVE",
                    "anchor_relationships": "COARSE_PHYSICAL_RESERVE",
                    "max_anchor_selection_frequency": 0,
                    "required_for_busproxy_resolution": False,
                    "resolution_class": resolution_class,
                    "identity_evidenced": identity_evidenced,
                    "resolution_status": "IDENTITY_EVIDENCED_FIELD_VALIDATION_PENDING" if identity_evidenced else "MANUAL_IDENTITY_AND_FIELD_VALIDATION_REQUIRED",
                    "coverage_recompute_required": False,
                    "minimum_anchor_distance_km": np.nan,
                    "source_kind": candidate.get("source_kind", ""),
                    "coordinate_source": candidate.get("coordinate_source", ""),
                    "contact_if_known": candidate.get("contact_if_known", ""),
                    "narrow_identity_evidence_applied": bool(
                        _truth(candidate.get("narrow_identity_evidence_applied", False))
                    ),
                    "identity_address_evidenced": bool(
                        _truth(candidate.get("identity_address_evidenced", False))
                    ),
                    "facility_exists_evidenced": bool(
                        _truth(candidate.get("facility_exists_evidenced", False))
                    ),
                    "identity_source_reference": candidate.get(
                        "identity_source_reference", ""
                    ),
                    "identity_source_urls": candidate.get("identity_source_urls", ""),
                    "provisional_facility_latitude": candidate.get(
                        "provisional_facility_latitude", np.nan
                    ),
                    "provisional_facility_longitude": candidate.get(
                        "provisional_facility_longitude", np.nan
                    ),
                    "provisional_coordinate_source": candidate.get(
                        "provisional_coordinate_source", ""
                    ),
                    "provisional_coordinate_source_url": candidate.get(
                        "provisional_coordinate_source_url", ""
                    ),
                    "provisional_coordinate_exact_for_release": False,
                    "identity_evidence_limitations": candidate.get(
                        "identity_evidence_limitations", ""
                    ),
                    "venue_readiness_prior": candidate.get("venue_readiness_prior", np.nan),
                    "raw_elderly_exposure": candidate.get("raw_elderly_exposure", np.nan),
                    "need_weighted_exposure": candidate.get("need_weighted_exposure", np.nan),
                    "high_need_elderly_exposure": candidate.get("high_need_elderly_exposure", np.nan),
                    "structural_need": candidate.get("structural_need", np.nan),
                }
            )
        queue = pd.concat([queue, pd.DataFrame(extra_rows)], ignore_index=True)
    if not queue_min <= len(queue) <= queue_max:
        raise RuntimeError(f"Physical queue size outside contract: {len(queue)} not in {queue_min}..{queue_max}")
    if queue["venue_id"].astype(str).str.upper().str.startswith("BUSPROXY_").any():
        raise RuntimeError("BUSPROXY escaped into the physical field queue")
    if queue["venue_id"].duplicated().any():
        raise RuntimeError("Physical field queue contains duplicate venue IDs")
    mapping = mapping.loc[mapping["physical_candidate_id"].isin(set(queue["venue_id"].astype(str)))].copy()
    queue.insert(0, "field_validation_rank", np.arange(1, len(queue) + 1))

    bus_anchors = set(union.loc[union["is_busproxy_spatial_anchor"], "venue_id"].astype(str))
    mapped_bus = set(mapping.loc[mapping["required_for_busproxy_resolution"], "source_anchor_id"].astype(str))
    same_cluster_bus = set(
        mapping.loc[
            mapping["required_for_busproxy_resolution"]
            & mapping["same_coverage_cluster"],
            "source_anchor_id",
        ].astype(str)
    )
    cross_cluster_manual_only = mapped_bus - same_cluster_bus
    unresolved_bus = sorted(bus_anchors - mapped_bus)
    unresolved = union.loc[union["venue_id"].isin(unresolved_bus)].copy()
    if unresolved.empty:
        unresolved = pd.DataFrame(columns=list(union.columns) + ["unresolved_reason"])
    else:
        unresolved["unresolved_reason"] = "NO_NON_BUS_PHYSICAL_CANDIDATE_WITHIN_REGISTERED_FALLBACK_OR_5KM_SAME_ADMIN_SEARCH"

    form, evidence = _field_form_and_evidence(queue)
    summary = {
        "selected_union_count": int(len(union)),
        "selected_union_busproxy_count": int(union["is_busproxy_spatial_anchor"].sum()),
        "relationship_rows": int(len(mapping)),
        "physical_queue_count": int(len(queue)),
        "queue_min": queue_min,
        "queue_max": queue_max,
        "identity_evidenced_count": int(queue["identity_evidenced"].sum()),
        "narrow_identity_evidence_count": int(
            queue["narrow_identity_evidence_applied"].sum()
        ),
        "address_evidenced_count": int(
            (queue["identity_evidenced"] | queue["identity_address_evidenced"]).sum()
        ),
        "provisional_coordinate_not_release_exact_count": int(
            queue["provisional_facility_latitude"].notna().sum()
        ),
        "field_verified_count": 0,
        "busproxy_anchor_count": len(bus_anchors),
        "busproxy_anchor_with_physical_candidate_count": len(mapped_bus),
        "busproxy_anchor_without_physical_candidate_count": len(unresolved_bus),
        "busproxy_same_cluster_candidate_ready_count": len(same_cluster_bus),
        "busproxy_cross_cluster_manual_only_count": len(cross_cluster_manual_only),
        "busproxy_cross_cluster_manual_only_ids": sorted(cross_cluster_manual_only),
        "busproxy_operationally_resolved_count": 0,
        "selected_final_busproxy_count": None,
        "note": (
            "BUSPROXY anchors are never rows in the physical field-validation form. "
            "A candidate mapping is not an operational resolution; cross-cluster "
            "candidates require field verification and 5 km coverage recomputation."
        ),
    }
    return queue, mapping, unresolved, form.merge(evidence, on="venue_id", how="left"), summary


def _field_form_and_evidence(queue: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    context_columns = [
        "venue_id",
        "venue_name",
        "venue_type",
        "cluster_id",
        "admin_code",
        "sigungu",
        "address",
        "latitude",
        "longitude",
        "resolution_class",
        "coordinate_source",
        "source_anchor_ids",
        "anchor_relationships",
        "coverage_recompute_required",
        "narrow_identity_evidence_applied",
        "identity_address_evidenced",
        "facility_exists_evidenced",
        "identity_source_reference",
        "identity_source_urls",
        "provisional_facility_latitude",
        "provisional_facility_longitude",
        "provisional_coordinate_source",
        "provisional_coordinate_source_url",
        "provisional_coordinate_exact_for_release",
        "identity_evidence_limitations",
    ]
    form = queue[context_columns].copy()
    # These two values are deliberately only search hints.  Give the exported
    # form fail-closed names so a spreadsheet user cannot mistake them for the
    # independently verified coordinates consumed by the release gate.
    form = form.rename(
        columns={
            "latitude": "catalog_anchor_latitude_not_for_release",
            "longitude": "catalog_anchor_longitude_not_for_release",
            "provisional_facility_latitude": "provisional_facility_latitude_not_for_release",
            "provisional_facility_longitude": "provisional_facility_longitude_not_for_release",
        }
    )
    address_evidenced = queue["identity_evidenced"].map(_truth) | queue[
        "identity_address_evidenced"
    ].map(_truth)
    facility_exists_evidenced = queue["identity_evidenced"].map(_truth) | queue[
        "facility_exists_evidenced"
    ].map(_truth)
    form["verified_address"] = np.where(address_evidenced, queue["address"], "")
    form["verified_latitude"] = np.where(queue["identity_evidenced"], queue["latitude"], np.nan)
    form["verified_longitude"] = np.where(queue["identity_evidenced"], queue["longitude"], np.nan)
    for column in FIELD_STATUS_COLUMNS:
        form[column] = "UNKNOWN"
    form["actual_facility_exists"] = np.where(
        facility_exists_evidenced, "YES", "UNKNOWN"
    )
    form[FIELD_DATE_COLUMN] = "UNKNOWN"
    for column in FIELD_DERIVED_COLUMNS:
        form[column] = False
    evidence = pd.DataFrame({"venue_id": queue["venue_id"].astype(str)})
    for column in EVIDENCE_COLUMNS[1:]:
        evidence[column] = ""
    identity_or_narrow = queue["identity_evidenced"].map(_truth) | queue[
        "narrow_identity_evidence_applied"
    ].map(_truth)
    evidence["verification_source"] = np.where(
        identity_or_narrow, "PROJECT_INTERNAL_OFFICIAL_IDENTITY_SOURCE", ""
    )
    source_reference = np.select(
        [
            queue["venue_id"].astype(str).str.startswith("HIRA_"),
            queue["venue_id"].astype(str).str.startswith("OFFICIAL_VHSC_"),
        ],
        [
            "01_public_sources/v2_full/derived/chungbuk_hira_medical_facilities_master_202606.csv",
            "14_v5_external_data/venues/chungbuk_official_village_halls_senior_centers_v5_2.csv",
        ],
        default="08_candidate_venues/candidate_venues_model_ready_final_v6.csv",
    )
    source_reference = pd.Series(source_reference, index=queue.index, dtype="object")
    narrow_reference = queue["identity_source_reference"].fillna("").astype(str).str.strip()
    source_reference = source_reference.mask(narrow_reference.ne(""), narrow_reference)
    evidence["source_reference"] = np.where(
        identity_or_narrow, source_reference, ""
    )
    evidence["verification_notes"] = ""
    evidence.loc[queue["identity_evidenced"].map(_truth), "verification_notes"] = (
        "Identity/existence only; all operational attributes and direct contact remain unverified."
    )
    evidence.loc[
        queue["narrow_identity_evidence_applied"].map(_truth), "verification_notes"
    ] = (
        "Official facility/address/contact listing only. The secondary facility point is not "
        "an exact parking coordinate and is forbidden for release; direct field coordinates and "
        "all operational attributes remain unverified."
    )
    return form, evidence


def _manual_required(
    queue: pd.DataFrame,
    form: pd.DataFrame,
    evidence: pd.DataFrame | None = None,
) -> pd.DataFrame:
    status = form.set_index("venue_id")
    evidence_by_id = evidence.set_index("venue_id") if evidence is not None and len(evidence) else None
    rows: list[dict[str, Any]] = []
    for _, candidate in queue.iterrows():
        values = status.loc[str(candidate["venue_id"])]
        catalog_anchor_coordinate_source = str(
            candidate.get("coordinate_source", "")
        ).strip()
        catalog_anchor_is_proxy_or_centroid = bool(
            re.search(
                r"proxy|centroid|not_exact",
                catalog_anchor_coordinate_source,
                flags=re.IGNORECASE,
            )
        )
        missing = [column for column in FIELD_STATUS_COLUMNS if str(values[column]) != "YES"]
        if not str(values.get("verified_address", "")).strip():
            missing.append("verified_address")
        for coordinate_column in ("verified_latitude", "verified_longitude"):
            coordinate_value = pd.to_numeric(
                pd.Series([values.get(coordinate_column, np.nan)]), errors="coerce"
            ).iloc[0]
            if not np.isfinite(coordinate_value):
                missing.append(coordinate_column)
        if str(values[FIELD_DATE_COLUMN]) in {"", "UNKNOWN", "nan", "NaT"}:
            missing.append(FIELD_DATE_COLUMN)
        evidence_missing: list[str] = []
        if evidence_by_id is not None:
            evidence_row = evidence_by_id.loc[str(candidate["venue_id"])]
            if not _truth(evidence_row.get("field_evidence_complete", False)):
                evidence_missing = [
                    "verification_source",
                    "source_reference_or_evidence_file",
                    "verifier",
                    "contact_channel_or_evidence_file",
                ]
        rows.append(
            {
                "priority": candidate["priority"],
                "field_validation_rank": candidate["field_validation_rank"],
                "anchor": candidate["source_anchor_ids"],
                "candidate_actual_facility_id": candidate["venue_id"],
                "candidate_actual_facility_name": candidate["venue_name"],
                "address": candidate["address"],
                "catalog_anchor_latitude_not_for_release": candidate["latitude"],
                "catalog_anchor_longitude_not_for_release": candidate["longitude"],
                "catalog_anchor_coordinate_source": catalog_anchor_coordinate_source,
                "catalog_anchor_is_proxy_or_centroid": catalog_anchor_is_proxy_or_centroid,
                "contact_if_known": candidate.get("contact_if_known", ""),
                "provisional_facility_latitude_not_for_release": candidate.get(
                    "provisional_facility_latitude", np.nan
                ),
                "provisional_facility_longitude_not_for_release": candidate.get(
                    "provisional_facility_longitude", np.nan
                ),
                "provisional_coordinate_source": candidate.get(
                    "provisional_coordinate_source", ""
                ),
                "identity_source_reference": candidate.get(
                    "identity_source_reference", ""
                ),
                "identity_source_urls": candidate.get("identity_source_urls", ""),
                "identity_evidence_limitations": candidate.get(
                    "identity_evidence_limitations", ""
                ),
                "fields_to_confirm": "|".join(
                    [*missing, *[f"EVIDENCE:{value}" for value in evidence_missing]]
                ),
                "reason": candidate["anchor_relationships"],
                "current_evidence": f"{candidate['resolution_class']}|{candidate['source_kind']}|{candidate['coordinate_source']}",
                "fallback_candidate": bool("FALLBACK" in str(candidate["anchor_relationships"]) or candidate["required_for_busproxy_resolution"]),
                "policy_impact": candidate["policies_impacted"],
                "coverage_recompute_required": candidate["coverage_recompute_required"],
            }
        )
    return pd.DataFrame(rows)


def _audit_operational_sources(root: Path, input_dir: Path | None, verified_ids: set[str]) -> tuple[dict[str, Any], bool, list[str]]:
    previous_contract = root / "outputs/model_v1/09_stage4_finalization/runs/stage4_1_20260819T203433Z_754bf1271229/06_field_readiness/mobile_team_base_contract.json"
    pbf = root / "14_v6_external_data/road_network/raw/chungcheong-non-military_20260817.osm.pbf"
    blockers: list[str] = []
    details: dict[str, Any] = {
        "input_dir": str(input_dir) if input_dir else None,
        "previous_team_base_contract": _json(previous_contract) if previous_contract.is_file() else None,
        "internal_evidence": {
            "staffing_sessions": "13_mobile_clinic_actual/cheongju_medical_center_foi_staffing_sessions_2025_2026.csv",
            "staffing_sessions_disposition": "STAFFING_EVIDENCE_ONLY_NOT_TEAM_BASE_OR_TEAM_CAPABILITY",
            "osm_graph": "14_v6_external_data/road_network/raw/chungcheong-non-military_20260817.osm.pbf",
            "osm_graph_available": pbf.is_file(),
            "osm_pbf_sha256": _sha256(pbf) if pbf.is_file() else None,
            "travel_disposition": "ROUTING_GRAPH_AVAILABLE_BUT_CONFIRMED_TEAM_BASES_ABSENT",
        },
        "files": {},
    }
    if input_dir is None:
        blockers.extend([f"OPERATIONAL_INPUT_PENDING:{key}" for key in OPERATIONAL_INPUT_FILENAMES])
        return details, False, blockers
    loaded = load_operational_input_dir(input_dir)
    details["loader_issues"] = list(loaded.issues)
    blockers.extend(loaded.issues)
    for key, path in loaded.paths.items():
        table = getattr(loaded.tables, key)
        details["files"][key] = {
            "path": str(path),
            "exists": path.is_file(),
            "sha256": _sha256(path) if path.is_file() else None,
            "rows": int(len(table)),
            "columns": list(table.columns),
        }
        if table.empty:
            blockers.append(f"EMPTY_OPERATIONAL_INPUT:{key}")
    if len(verified_ids) < 20:
        blockers.append(f"OPERATIONAL_PHYSICAL_POOL_INSUFFICIENT:{len(verified_ids)}<20")
    details["release_verified_physical_venue_count"] = int(len(verified_ids))
    details["strict_contract_disposition"] = (
        "ATTEMPT_ONLY_AFTER_OPERATIONAL_PLAN_AND_BUNDLE_REQUIREMENTS_EXIST"
    )
    blockers = list(dict.fromkeys(blockers))
    return details, not blockers, blockers


def _prepare_operational_tables_and_travel(
    root: Path,
    run_root: Path,
    input_dir: Path,
    universe: OperationalCandidateUniverse,
    osm_pbf: Path,
) -> tuple[OperationalTables, dict[str, Any], list[str]]:
    loaded = load_operational_input_dir(input_dir)
    tables = loaded.tables
    blockers = list(loaded.issues)
    travel_details: dict[str, Any] = {
        "routing_contract": (
            "driving; direction=oneway; retain_all=True before largest-strong-core; "
            "static free-flow proxy"
        ),
        "osm_pbf_sha256": _sha256(osm_pbf),
        "source": "USER_SUPPLIED",
        "generated": False,
    }
    if tables.travel.empty:
        other_empty = [
            key
            for key in OPERATIONAL_INPUT_FILENAMES
            if key != "travel" and getattr(tables, key).empty
        ]
        if other_empty:
            blockers.extend(
                f"TRAVEL_RECOMPUTE_PREREQUISITE_EMPTY:{key}" for key in other_empty
            )
            return tables, travel_details, list(dict.fromkeys(blockers))
        confirmed_bases = tables.bases.loc[
            tables.bases["confirmed"].map(_truth)
        ].copy()
        venues = universe.candidates.copy()
        evidence_columns = [
            value
            for value in ("venue_id", "source_reference", "evidence_file_or_url")
            if value in universe.release_evidence.columns
        ]
        if len(evidence_columns) > 1:
            venues = venues.merge(
                universe.release_evidence[evidence_columns],
                on="venue_id",
                how="left",
                validate="one_to_one",
            )
        venues["operational_release_verified"] = True
        try:
            computed = recompute_operational_travel(
                confirmed_bases,
                venues,
                package_root=root,
                pbf_path=osm_pbf,
                graph_cache_dir=root / "outputs/model_v1/08_stage4/cache/road_graph",
                result_cache_dir=run_root / "05_travel/cache",
                max_snap_distance_m=1000.0,
            )
        except (OperationalTravelError, ImportError) as exc:
            blockers.append(f"TRAVEL_RECOMPUTE_FAILED:{type(exc).__name__}:{exc}")
            travel_details.update(
                {
                    "source": "AUTOMATIC_OSM_RECOMPUTE_FAILED",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            return tables, travel_details, list(dict.fromkeys(blockers))
        table_mapping = {
            key: getattr(tables, key) for key in OPERATIONAL_INPUT_FILENAMES
        }
        table_mapping["travel"] = computed.travel
        tables = OperationalTables(**table_mapping)
        _write_csv(
            run_root / "05_travel/team_venue_travel_operational.csv",
            computed.travel,
        )
        _write_csv(run_root / "05_travel/travel_snap_audit.csv", computed.snap_audit)
        _write_json(run_root / "05_travel/travel_recompute_audit.json", computed.audit)
        travel_details.update(
            {
                "source": "AUTOMATIC_OSM_RECOMPUTE",
                "generated": True,
                "cache_key": computed.cache_key,
                "cache_hit": computed.cache_hit,
                "rows": len(computed.travel),
                "reachable_rows": int(computed.travel["reachable"].map(_truth).sum()),
                "unreachable_rows": int((~computed.travel["reachable"].map(_truth)).sum()),
            }
        )
    else:
        _write_csv(
            run_root / "05_travel/team_venue_travel_operational.csv",
            tables.travel,
        )
        travel_details.update(
            {
                "rows": len(tables.travel),
                "input_sha256": _sha256(
                    loaded.paths["travel"]
                ) if loaded.paths["travel"].is_file() else None,
                "strict_validation": "DEFERRED_TO_POLICY_JOINT_OPERATIONAL_AUDIT",
            }
        )
    return tables, travel_details, list(dict.fromkeys(blockers))


def _build_operational_universe(
    baseline: Baseline,
    queue: pd.DataFrame,
    form: pd.DataFrame,
    evidence: pd.DataFrame,
    catalog: pd.DataFrame,
    mapping: pd.DataFrame,
) -> OperationalCandidateUniverse:
    stage3 = discover_stage3(baseline.root)
    coverage = load_coverage(stage3, "hard_5000m")
    grid = normalize_grid_policy(read_table(stage3.grid_policy))
    return build_verified_operational_universe(
        queue=queue,
        normalized_field_form=form,
        field_evidence=evidence,
        stage3_catalog=catalog,
        anchor_mapping=mapping,
        stage3_hard_5000m=coverage.matrix,
        stage3_venue_ids=coverage.venue_ids,
        stage3_grid=grid,
        stage3_grid_ids=coverage.grid_ids,
    )


def _write_operational_universe(
    run_root: Path,
    universe: OperationalCandidateUniverse,
) -> None:
    directory = run_root / "04_operational_inputs"
    _write_csv(directory / "operational_candidate_universe.csv", universe.candidates)
    _write_sparse_npz(directory / "operational_hard_5000m.npz", universe.coverage)
    _write_csv(
        directory / "operational_venue_order.csv",
        pd.DataFrame({"matrix_row": np.arange(len(universe.venue_ids)), "venue_id": universe.venue_ids}),
    )
    _write_csv(
        directory / "operational_grid_order.csv",
        pd.DataFrame({"matrix_column": np.arange(len(universe.grid_ids)), "grid_id": universe.grid_ids}),
    )
    _write_csv(directory / "coverage_origin_ledger.csv", universe.coverage_origin_ledger)
    _write_csv(directory / "source_anchor_provenance.csv", universe.source_anchor_provenance)
    _write_csv(directory / "operational_release_evidence.csv", universe.release_evidence)
    _write_csv(directory / "operational_universe_audit_issues.csv", universe.audit_issues)
    _write_csv(directory / "operational_coverage_overlap_edges.csv", universe.coverage_overlap_edges)
    _write_csv(directory / "operational_coverage_clusters.csv", universe.coverage_clusters)
    _write_csv(directory / "operational_candidate_exclusions.csv", universe.exclusions)
    _write_json(
        directory / "operational_candidate_universe_summary.json",
        {"summary": dict(universe.summary), "artifact_sha256": dict(universe.artifact_sha256)},
    )


def _gini(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0 or float(array.sum()) <= 0:
        return 0.0
    array.sort()
    n = len(array)
    return float(
        2.0 * np.sum(np.arange(1, n + 1) * array) / (n * array.sum())
        - (n + 1) / n
    )


def _plan_coverage_metrics(
    universe: OperationalCandidateUniverse,
    selected_indices: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    selected_indices = np.asarray(selected_indices, dtype=int)
    selected = universe.coverage[selected_indices].tocsr()
    counts = np.asarray(selected.sum(axis=0)).ravel()
    covered = counts > 0
    grid = universe.grid.reset_index(drop=True)
    population = pd.to_numeric(grid["elderly_population"], errors="raise").to_numpy(float)
    need = pd.to_numeric(grid["need_weighted_population"], errors="raise").to_numpy(float)
    high = pd.to_numeric(grid["high_need_population"], errors="raise").to_numpy(float)
    rows: list[dict[str, Any]] = []
    for local_index, candidate_index in enumerate(selected_indices):
        columns = selected.getrow(local_index).indices
        unique = columns[counts[columns] == 1]
        rows.append(
            {
                "venue_id": str(universe.venue_ids[candidate_index]),
                "unique_coverage_contribution": float(population[unique].sum()),
                "unique_need_weighted_contribution": float(need[unique].sum()),
                "unique_high_need_contribution": float(high[unique].sum()),
                "covered_grid_count_in_final_plan": int(len(columns)),
                "uniquely_covered_grid_count_in_final_plan": int(len(unique)),
            }
        )
    covered_by_sigungu = {
        str(sigungu): float(population[covered & grid["sigungu"].astype(str).eq(str(sigungu)).to_numpy()].sum())
        for sigungu in sorted(grid["sigungu"].astype(str).unique())
    }
    total_repeated = float(
        sum(population[selected.getrow(index).indices].sum() for index in range(selected.shape[0]))
    )
    unique_population = float(population[covered].sum())
    rural = (
        grid["is_rural_eup_myeon"].map(_truth).to_numpy(bool)
        if "is_rural_eup_myeon" in grid
        else np.zeros(len(grid), dtype=bool)
    )
    metrics = {
        "unique_elderly_population": unique_population,
        "need_weighted_population": float(need[covered].sum()),
        "high_need_population": float(high[covered].sum()),
        "beneficiary_gini_across_sigungu": _gini(covered_by_sigungu.values()),
        "beneficiary_gini_definition": "Gini of unique covered elderly population across 11 sigungu",
        "redundancy_population_fraction": (
            float(1.0 - unique_population / total_repeated) if total_repeated > 0 else 0.0
        ),
        "rural_unique_elderly_population": float(population[covered & rural].sum()),
        "urban_unique_elderly_population": float(population[covered & ~rural].sum()),
        "rural_unique_elderly_share": (
            float(population[covered & rural].sum() / unique_population)
            if unique_population > 0
            else 0.0
        ),
        "sigungu_unique_covered_population_json": json.dumps(
            covered_by_sigungu, ensure_ascii=False, sort_keys=True
        ),
    }
    return pd.DataFrame(rows), metrics


def _write_operational_optimizer_result(
    run_root: Path,
    result: OperationalOptimizationResult,
    universe: OperationalCandidateUniverse,
    bundle_values: pd.DataFrame,
) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    policy = result.scenario
    summary = result.as_dict()
    _write_json(run_root / f"06_optimizer/optimizer_result__{policy}.json", summary)
    _write_csv(
        run_root / f"06_optimizer/no_good_ledger__{policy}.csv",
        pd.DataFrame([asdict(value) for value in result.no_good_ledger]),
    )
    if result.spatial_result is not None:
        telemetry = pd.DataFrame(
            [asdict(value) for value in result.spatial_result.telemetry]
        )
    else:
        telemetry = pd.DataFrame()
    _write_csv(run_root / f"06_optimizer/solver_stage_telemetry__{policy}.csv", telemetry)
    if not result.ready:
        return None, {
            "policy": policy,
            "ready": False,
            "certified": bool(result.certified),
            "certification_class": result.certification_class,
            "status": result.status,
            "blockers": "|".join(result.blockers),
        }

    plan = result.selected_frame.copy()
    contributions, recomputed = _plan_coverage_metrics(
        universe, result.selected_indices
    )
    plan = plan.merge(contributions, on="venue_id", how="left", validate="one_to_one")
    chosen_bundle_values = bundle_values.rename(
        columns={"bundle_value": "specialty_gap_weighted_exposure"}
    )
    plan = plan.merge(
        chosen_bundle_values,
        on=["venue_id", "bundle_id"],
        how="left",
        validate="one_to_one",
    )
    evidence_columns = [
        column
        for column in (
            "venue_id",
            "verification_source",
            "source_reference",
            "evidence_file_or_url",
            "contact_person_or_office",
            "contact_channel",
            "verifier",
            "verification_notes",
            "operational_release_verified",
        )
        if column in universe.release_evidence.columns
    ]
    if len(evidence_columns) > 1:
        plan = plan.merge(
            universe.release_evidence[evidence_columns],
            on="venue_id",
            how="left",
            validate="one_to_one",
        )
    plan["policy"] = policy
    plan["actual_venue_id"] = plan["venue_id"].astype(str)
    plan["actual_venue_name"] = plan.get("venue_name", "UNKNOWN")
    plan["address"] = plan["verified_address"]
    plan["latitude"] = plan["verified_latitude"]
    plan["longitude"] = plan["verified_longitude"]
    plan["original_spatial_anchor"] = plan.get(
        "operational_source_anchor_ids", ""
    )
    plan["fallback_information"] = plan.get("fallback_provenance_json", "[]")
    relationships = (
        plan["operational_anchor_relationships"].astype(str)
        if "operational_anchor_relationships" in plan
        else pd.Series("", index=plan.index, dtype=str)
    )
    plan["selected_via_fallback"] = ~relationships.str.contains(
        "DIRECT_SELECTED_PHYSICAL", regex=False
    )
    plan["bundle"] = plan["bundle_id"]
    plan["potential_beneficiary_exposure"] = plan.get(
        "operational_raw_elderly_exposure",
        plan.get("raw_elderly_exposure", np.nan),
    )
    plan["high_need_contribution"] = plan["unique_high_need_contribution"]
    plan["vehicle_compatibility"] = (
        plan["vehicle_type"].astype(str)
        + "|capacity="
        + plan["capacity"].astype(str)
        + "|required="
        + plan["required_vehicle_capacity"].astype(str)
    )
    plan["travel_proxy"] = plan["travel_minutes_proxy"]
    plan["selection_reason"] = (
        "frozen_" + policy + "_lexicographic_objectives_plus_proven_operational_feasibility"
    )
    if len(plan) != 20 or plan["venue_id"].nunique() != 20:
        raise RuntimeError(f"{policy} final plan is not exactly 20 unique venues")
    if plan["venue_id"].astype(str).str.upper().str.startswith("BUSPROXY_").any():
        raise RuntimeError(f"{policy} final plan contains BUSPROXY")
    if not plan["field_verified"].map(_truth).all():
        raise RuntimeError(f"{policy} final plan contains a non-field-verified venue")
    forbidden_visit_columns = {
        "date",
        "exact_date",
        "exact_month",
        "visit_date",
    }
    present_forbidden = sorted(forbidden_visit_columns & set(plan.columns))
    if present_forbidden:
        raise RuntimeError(
            f"Stage 5 visit-date columns escaped into {policy} final plan: {present_forbidden}"
        )
    preferred = [
        "visit_id",
        "policy",
        "sigungu",
        "admin_code",
        "actual_venue_id",
        "actual_venue_name",
        "address",
        "latitude",
        "longitude",
        "original_spatial_anchor",
        "fallback_information",
        "selected_via_fallback",
        "field_verified",
        "verification_source",
        "source_reference",
        "evidence_file_or_url",
        "verifier",
        "bundle",
        "structural_need",
        "specialty_gap_weighted_exposure",
        "potential_beneficiary_exposure",
        "unique_coverage_contribution",
        "unique_need_weighted_contribution",
        "unique_high_need_contribution",
        "team_id",
        "vehicle_id",
        "vehicle_compatibility",
        "distance_km",
        "travel_proxy",
        "selection_reason",
    ]
    plan = plan[
        [
            *[value for value in preferred if value in plan.columns],
            *[value for value in plan.columns if value not in preferred],
        ]
    ]
    _write_csv(run_root / f"07_final_plans/final_plan__{policy}.csv", plan)
    _write_csv(
        run_root / f"07_final_plans/public_operational_assignments__{policy}.csv",
        result.public_assignments,
    )
    _write_json(
        run_root / f"07_final_plans/availability_witness__{policy}.json",
        {
            "availability_witness_sha256": result.availability_witness_sha256,
            "dates_exposed": False,
            "assignment_count": len(result.public_assignments),
        },
    )
    metrics = dict(result.spatial_metrics)
    metrics.update(recomputed)
    alternate_count = pd.to_numeric(
        plan["alternate_candidate_count"]
        if "alternate_candidate_count" in plan
        else pd.Series(0, index=plan.index),
        errors="coerce",
    ).fillna(0)
    metrics.update(
        {
            "policy": policy,
            "ready": True,
            "certified": True,
            "certification_class": result.certification_class,
            "status": result.status,
            "bundle_counts_json": json.dumps(result.bundle_counts, sort_keys=True),
            "verified_venue_rate": float(plan["field_verified"].map(_truth).mean()),
            "selected_busproxy_count": 0,
            "fallback_availability_rate": float(alternate_count.gt(0).mean()),
            "mean_travel_minutes_proxy": float(
                pd.to_numeric(plan["travel_minutes_proxy"], errors="raise").mean()
            ),
            "max_travel_minutes_proxy": float(
                pd.to_numeric(plan["travel_minutes_proxy"], errors="raise").max()
            ),
            "availability_witness_sha256": result.availability_witness_sha256,
        }
    )
    return plan, metrics


def _snapshot_stage5(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for parent in (root / "outputs/model_v1", root / "reports/model_v1"):
        if not parent.exists():
            continue
        excluded_operational_final = (
            parent / "11_stage4_operational_final"
            if parent.name == "model_v1" and parent.parent.name == "outputs"
            else parent / "stage4_operational_final"
        ).resolve()

        # Do not use ``Path.rglob`` here.  The output tree contains historical
        # provenance snapshots, and rglob still walks the whole Operational
        # Final subtree before we can discard it.  ``os.walk(topdown=True)``
        # lets us prune that subtree at its parent while preserving the exact
        # Stage5-name detection semantics for every other directory and file.
        def _raise_walk_error(error: OSError) -> None:
            raise error

        for directory, directory_names, file_names in os.walk(
            parent, topdown=True, followlinks=False, onerror=_raise_walk_error
        ):
            directory_path = Path(directory)
            directory_names.sort()
            file_names.sort()
            directory_names[:] = [
                name
                for name in directory_names
                if (directory_path / name).resolve() != excluded_operational_final
            ]

            for name in directory_names:
                path = directory_path / name
                relative_parts = [part.lower() for part in path.relative_to(parent).parts]
                if any("stage5" in part for part in relative_parts):
                    rows.append({"path": _relative(path, root), "kind": "directory"})

            for name in file_names:
                path = directory_path / name
                relative_parts = [part.lower() for part in path.relative_to(parent).parts]
                if not any("stage5" in part for part in relative_parts):
                    continue
                rows.append(
                    {
                        "path": _relative(path, root),
                        "kind": "file",
                        "size_bytes": path.stat().st_size,
                        "sha256": _sha256(path),
                    }
                )
    return sorted(rows, key=lambda row: str(row["path"]))


def _source_contract(root: Path, config_path: Path) -> pd.DataFrame:
    paths = sorted((root / "src/mediroad/stage4_operational_final").glob("*.py"))
    paths += [
        config_path,
        root / "scripts/run_stage4_operational_final.py",
        root / "run_model_v1_stage4_operational_final.ps1",
        root / NARROW_IDENTITY_EVIDENCE_RELATIVE_PATH,
    ]
    paths += sorted((root / "tests").glob("test_stage4_operational_final*.py"))
    paths += sorted((root / "tests").glob("test_stage4_operational_*.py"))
    paths += [root / "tests/test_field_validation.py"]
    paths += [
        root / f"src/mediroad/stage4_2/{name}.py"
        for name in (
            "bundles",
            "candidates",
            "compression",
            "contracts",
            "data",
            "discovery",
            "errors",
            "field_validation",
            "optimizer",
            "types",
            "utils",
        )
    ]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Operational source contract missing: {missing}")
    return pd.DataFrame(
        [{"relative_path": _relative(path, root), "size_bytes": path.stat().st_size, "sha256": _sha256(path)} for path in sorted(set(paths))]
    )


def _input_manifest(
    field_validation_path: Path | None,
    field_evidence_path: Path | None,
    operational_input_dir: Path | None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    direct = {
        "field_validation": field_validation_path,
        "field_evidence": field_evidence_path,
    }
    for role, path in direct.items():
        if path is None:
            continue
        if not path.is_file():
            raise FileNotFoundError(path)
        rows.append(
            {
                "role": role,
                "source_path": str(path.resolve()),
                "relative_name": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    if operational_input_dir is not None:
        if not operational_input_dir.is_dir():
            raise FileNotFoundError(operational_input_dir)
        for path in sorted(value for value in operational_input_dir.rglob("*") if value.is_file()):
            rows.append(
                {
                    "role": "operational_input",
                    "source_path": str(path.resolve()),
                    "relative_name": path.relative_to(operational_input_dir).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    return pd.DataFrame(
        rows,
        columns=["role", "source_path", "relative_name", "size_bytes", "sha256"],
    )


def _snapshot_inputs(run_root: Path, manifest: pd.DataFrame) -> None:
    for index, row in enumerate(manifest.to_dict("records"), start=1):
        name = str(row["relative_name"]).replace("..", "_").replace("\\", "/")
        destination = run_root / "00_provenance/input_snapshot" / str(row["role"]) / name
        _copy_file(Path(str(row["source_path"])), destination)


def _hardware_audit(config: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "requested_profile": config.get("hardware", {}),
        "logical_cpu_count": os.cpu_count(),
        "platform": platform.platform(),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "environment_threads": {
            key: os.environ.get(key)
            for key in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "CUDA_VISIBLE_DEVICES",
            )
        },
    }
    try:
        import psutil  # type: ignore

        memory = psutil.virtual_memory()
        result["ram_total_bytes"] = int(memory.total)
        result["ram_available_bytes_at_start"] = int(memory.available)
    except Exception as exc:  # pragma: no cover - environment diagnostic only
        result["psutil_error"] = f"{type(exc).__name__}:{exc}"
    try:
        probe = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
        result["nvidia_smi_return_code"] = int(probe.returncode)
        result["nvidia_smi"] = probe.stdout.strip()
        result["nvidia_smi_stderr"] = probe.stderr.strip()
    except Exception as exc:  # pragma: no cover - environment diagnostic only
        result["nvidia_smi_error"] = f"{type(exc).__name__}:{exc}"
    return result


def _acquire_writer_lock(output_root: Path) -> tuple[Path, str]:
    """Acquire the single-writer lock without overwriting another run's state."""

    output_root.mkdir(parents=True, exist_ok=True)
    lock_path = output_root / ".STAGE4_OPERATIONAL_FINAL_WRITER.lock"
    token = uuid.uuid4().hex
    payload = json.dumps(
        {
            "token": token,
            "pid": os.getpid(),
            "hostname": platform.node(),
            "created_utc": datetime.now(timezone.utc).isoformat(),
        },
        ensure_ascii=False,
        indent=2,
    ) + "\n"
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        owner = lock_path.read_text(encoding="utf-8", errors="replace") if lock_path.is_file() else "UNKNOWN"
        raise RuntimeError(
            "Another Stage 4 Operational Final writer owns the output namespace: "
            f"{lock_path}\n{owner}"
        ) from exc
    try:
        os.write(descriptor, payload.encode("utf-8"))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return lock_path, token


def _release_writer_lock(lock_path: Path, token: str) -> None:
    """Release only the lock created by this process."""

    if not lock_path.is_file():
        return
    try:
        recorded = json.loads(lock_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return
    if recorded.get("token") == token:
        lock_path.unlink()


def _build_inventory(run_root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(run_root.rglob("*")):
        if not path.is_file() or path.name in {"ARTIFACT_INVENTORY.csv", ".RUNNING", ".FAILED", ".COMMITTED"}:
            continue
        rows.append({"relative_path": path.relative_to(run_root).as_posix(), "size_bytes": path.stat().st_size, "sha256": _sha256(path)})
    return pd.DataFrame(rows)


def _manual_guide() -> str:
    fields = "\n".join(f"- `{column}`" for column in FIELD_STATUS_COLUMNS)
    return f"""# MEDIROAD Stage 4 수동 현장검증 안내

`MANUAL_FIELD_VALIDATION_REQUIRED.csv`의 우선순서대로 확인한 뒤
`field_validation_form.csv`와 `field_validation_evidence.csv`를 함께 갱신한다.

허용값은 `YES`, `NO`, `UNKNOWN`뿐이다. 추측은 `UNKNOWN`으로 남긴다. 다음 11개 상태가 모두 `YES`이고 `verification_date`가 실제 확인일 `YYYY-MM-DD`일 때만 `field_verified=true`가 된다.

{fields}

`actual_facility_exists=YES`가 미리 적힌 행은 프로젝트 내부 공식 ID 자료가 시설 정체성/존재를 지원한다는 뜻일 뿐이다. 주차·전기·예약·직접 연락 등 운영 적합성이 확인됐다는 뜻이 아니다.

`verified_address`, `verified_latitude`, `verified_longitude`도 실제 물리 장소와
일치해야 한다. 수동팩과 `field_validation_form.csv`의
`catalog_anchor_latitude_not_for_release` 및
`catalog_anchor_longitude_not_for_release`는 동결 계산 anchor의 참조값이지 현장
관측 좌표가 아니다. 수동팩에서 `catalog_anchor_is_proxy_or_centroid=true`인
행은 특히 실제 시설 위치나 내비게이션 좌표가 아니므로 `verified_*`에 복사하면
안 된다. 공식 주소와 현장 탐색 정보로 장소를 찾은 뒤 독립 확인한 좌표만 기록한다.

`field_validation_form.csv`의 `provisional_facility_*_not_for_release` 값도 현장
탐색용 힌트일 뿐이며 그대로 복사해 coverage/release 좌표로 사용할 수 없다.

증거는 `field_validation_evidence.csv`에 출처, 담당 기관, 확인 채널, 검증자, 메모를 기록한다. BUSPROXY는 공간 anchor이므로 이 양식의 물리 후보로 승격할 수 없다.
"""


def _validate_config(root: Path, config: dict[str, Any]) -> Path:
    if config.get("version") != "MEDIROAD_STAGE4_OPERATIONAL_FINAL_V1":
        raise RuntimeError("Unexpected Operational Final config version")
    if _truth(config.get("stage5_started")):
        raise RuntimeError("Operational Final config must keep stage5_started=false")
    parent = config.get("computational_parent", {})
    expected_parent = {
        "required_decision": EXPECTED_H_DECISION,
        "candidate_universe": "coarse_pareto",
        "expected_solver_candidate_count": 896,
        "expected_pattern_count": 10_838,
        "threshold_fingerprint": THRESHOLD_FINGERPRINT,
    }
    for key, expected in expected_parent.items():
        if parent.get(key) != expected:
            raise RuntimeError(f"Config drift at computational_parent.{key}")
    if not _truth(parent.get("frozen")):
        raise RuntimeError("Computational parent must be frozen")
    field = config.get("field_validation", {})
    queue_min = int(field.get("queue_min", 0))
    queue_max = int(field.get("queue_max", 0))
    if not (20 <= queue_min <= queue_max <= 896):
        raise RuntimeError("Field queue bounds are invalid")
    if not all(
        _truth(field.get(key))
        for key in (
            "verified_requires_all_yes",
            "verified_requires_valid_date",
            "busproxy_final_forbidden",
        )
    ):
        raise RuntimeError("Field fail-closed contract was relaxed")
    if field.get("narrow_identity_evidence_relative_path") != NARROW_IDENTITY_EVIDENCE_RELATIVE_PATH:
        raise RuntimeError("Narrow identity evidence path drifted")
    if not _truth(field.get("provisional_coordinate_release_forbidden")):
        raise RuntimeError("Provisional coordinates must remain forbidden for release")
    optimization = config.get("optimization", {})
    if int(optimization.get("visit_count", -1)) != 20:
        raise RuntimeError("Operational visit count must remain 20")
    if list(optimization.get("policies", [])) != list(POLICIES):
        raise RuntimeError("Operational policies/order changed")
    if abs(float(optimization.get("near_optimal_relative_gap_max", -1)) - 0.005) > 1e-12:
        raise RuntimeError("Near-optimality threshold must remain 0.005")
    if not _truth(optimization.get("exact_month_date_forbidden")):
        raise RuntimeError("Exact Stage5 dates must remain forbidden")
    expected_optimization = {
        "objective_retention": {
            "total_population": 0.99,
            "need_weighted": 0.99,
            "high_need": 0.99,
            "min_sigungu_coverage": 1.0,
        },
        "balanced_min_efficiency_fraction": 0.80,
        "equity_min_efficiency_fraction": 0.60,
        "high_need_quantile": 0.80,
        "known_overlap_penalty": 1000.0,
        "travel_penalty_scale": 1.0,
        "visit_location_deviation_penalty": 100.0,
        "deterministic_tiebreak_scale": 0.000001,
    }
    for key, expected in expected_optimization.items():
        observed = optimization.get(key)
        if isinstance(expected, dict):
            if observed != expected:
                raise RuntimeError(f"Frozen optimization contract drifted at {key}")
        elif abs(float(observed) - expected) > 1e-12:
            raise RuntimeError(f"Frozen optimization contract drifted at {key}")
    if config.get("matrix", {}).get("primary_matrix_id") != "hard_5000m":
        raise RuntimeError("Primary coverage matrix must remain hard_5000m")
    solver = config.get("solver", {})
    expected_solver = {
        "near_optimal_relative_gap_max": 0.005,
        "min_sigungu_ratio_integer_scale": 1_000_000,
        "visit_deviation_exact_integer_formulation": True,
        "known_incumbent_objective_cutoff": False,
        "operational_final_time_per_stage_sec": 600,
        "highs_threads": 8,
        "random_seed": 42,
    }
    for key, expected in expected_solver.items():
        if solver.get(key) != expected:
            raise RuntimeError(f"Frozen solver contract drifted at {key}")
    if config.get("runtime", {}) != {
        "candidate_workers": 1,
        "highs_threads_per_worker": 8,
        "highs_random_seed": 42,
        "logical_cpu_budget": 8,
        "memory_soft_limit_gib": 27,
    }:
        raise RuntimeError("Runtime resource contract drifted")
    candidate_constraints = config.get("candidate_constraints", {})
    if candidate_constraints != {
        "one_per_coverage_cluster": True,
        "max_visits_per_admin": 2,
        "max_visits_per_sigungu": 4,
    }:
        raise RuntimeError("Frozen candidate constraints drifted")
    bundle_assignment = config.get("bundle_assignment", {})
    if int(bundle_assignment.get("primary_minimum_per_bundle", -1)) != 1:
        raise RuntimeError("Every frozen bundle must appear at least once")
    if config.get("operational_optimizer", {}) != {
        "max_no_good_iterations": 100,
        "wall_time_limit_sec": 7200,
        "spatial_time_limit_per_stage_sec": 600,
        "bundle_time_limit_sec": 120,
        "assignment_search_state_limit": 500_000,
    }:
        raise RuntimeError("Operational decomposition contract drifted")
    release = config.get("release", {})
    if release.get("success_decision") != "PASS_STAGE4_OPERATIONAL_FINAL":
        raise RuntimeError("Success decision contract changed")
    if release.get("blocked_decision") != BLOCKED_DECISION:
        raise RuntimeError("Blocked decision contract changed")
    if not _truth(release.get("stage5_release_only_after_success")):
        raise RuntimeError("Stage5 release contract changed")

    output_root = (root / str(config.get("paths", {}).get("output_root", ""))).resolve()
    expected_root = (root / "outputs/model_v1/11_stage4_operational_final").resolve()
    if output_root != expected_root:
        raise RuntimeError(f"Output namespace must be exactly {expected_root}")
    return output_root


def _final_report(
    baseline_audit: dict[str, Any],
    canonical: dict[str, Any],
    union: pd.DataFrame,
    field_summary: dict[str, Any],
    blockers: list[str],
    gates: list[dict[str, Any]],
    *,
    decision_name: str,
    operational_audit: dict[str, Any],
    universe_summary: dict[str, Any],
    policy_metrics: list[dict[str, Any]],
    optimization_results: dict[str, OperationalOptimizationResult],
    travel_details: dict[str, Any],
) -> str:
    failed = [gate for gate in gates if not gate["passed"]]
    exact_location_evaluated = int(
        universe_summary.get("field_evidence_release_id_count", 0)
    )
    exact_location_accepted = int(
        universe_summary.get("operational_candidate_count", 0)
    )
    exact_location_excluded = int(
        universe_summary.get("excluded_release_id_count", 0)
    )
    if exact_location_evaluated != exact_location_accepted + exact_location_excluded:
        raise RuntimeError(
            "Exact-location partition mismatch: "
            f"evaluated={exact_location_evaluated}, "
            f"accepted={exact_location_accepted}, excluded={exact_location_excluded}"
        )
    exact_location_result = (
        "N/A — 0 release IDs entered evaluation "
        "(accepted 0; exclusion-ledger rows 0)"
        if exact_location_evaluated == 0
        else (
            f"accepted {exact_location_accepted}, excluded {exact_location_excluded} "
            f"/ evaluated {exact_location_evaluated}"
        )
    )
    metric_lines: list[str] = []
    for policy in POLICIES:
        rows = [value for value in policy_metrics if value.get("policy") == policy]
        if not rows:
            metric_lines.append(f"- {policy}: 운영 최적화 미실행")
            continue
        value = rows[0]
        if not value.get("ready"):
            metric_lines.append(
                f"- {policy}: `{value.get('status', 'NOT_READY')}` / `{value.get('blockers', '')}`"
            )
            continue
        metric_lines.append(
            "- "
            f"{policy}: unique={float(value.get('unique_elderly_population', 0)):,.3f}, "
            f"need={float(value.get('need_weighted_population', 0)):,.3f}, "
            f"high-need={float(value.get('high_need_population', 0)):,.3f}, "
            f"min-sigungu={float(value.get('min_sigungu_coverage_ratio', 0)):.6f}, "
            f"mean travel={float(value.get('mean_travel_minutes_proxy', 0)):.3f}분"
        )
    blocker_text = (
        "\n".join(f"- `{value}`" for value in blockers)
        if blockers
        else "- 없음"
    )
    optimizer_lines = [
        f"- {policy}: `{result.status}`, certification=`{result.certification_class}`, "
        f"no-good={result.no_good_count}"
        for policy, result in optimization_results.items()
    ] or ["- 실제 운영입력이 완성되지 않아 solver를 호출하지 않음"]
    stage5_state = "STAGE5_READY 인터페이스만 허용" if decision_name == "PASS_STAGE4_OPERATIONAL_FINAL" else "Stage 5 진입 금지"
    return f"""# MEDIROAD Stage 4 Operational Final

## Decision

`{decision_name}`

Quality gate {len(gates) - len(failed)}/{len(gates)} 통과. 실제 확인되지 않은 현장·팀·차량·가용성 사실은 승격하지 않았다.

## 1. Computational parent

- Stage4.2H run: `{baseline_audit['stage42h']['run_id']}`
- Decision: `{baseline_audit['stage42h']['decision']}`
- Solver universe: {baseline_audit['solver_candidate_count']} candidates / {baseline_audit['compressed_pattern_count']} lossless patterns
- Canonical stages: {canonical['certified_stage_count']}/{canonical['stage_count']} certified
- Direct gap과 rescue certificate의 certified upper bound를 별도 필드로 보존했다.

## 2. Candidate ladder history

사전등록 순서는 `top3 → top5 → coarse_pareto`이며 최종 rung은 `coarse_pareto`다. 이는 충북 전체 3,909개 unrestricted optimum 주장이 아니다.

## 3. Final coarse-Pareto computational plan

- E/B/Q 계획: 20 + 20 + 20
- 고유 spatial anchor: {len(union)}
- 선택 빈도 3: {int(union['selection_frequency'].eq(3).sum())}
- BUSPROXY spatial anchor: {int(union['is_busproxy_spatial_anchor'].sum())}

## 4. Venue resolution methodology

공식 ID·주소·정확 좌표만 physical identity 근거로 사용했다. centroid/proxy 좌표는 현장에서 정확 좌표가 새로 확인되지 않으면 operational universe에서 제외한다. 좌표가 바뀌면 EPSG:5179에서 hard 5 km CSR을 다시 계산한다.

## 5. Field validation

- 새 coarse queue: {field_summary['physical_queue_count']}
- 내부 identity evidence: {field_summary['identity_evidenced_count']}
- 제한 공식 lookup 보강: {field_summary['narrow_identity_evidence_count']}
- 주소 근거 확보: {field_summary['address_evidenced_count']}/{field_summary['physical_queue_count']}
- historical all-YES field gate: {field_summary['field_verified_count']}
- 독립 evidence gate까지 통과: {field_summary['operational_release_verified_count']}
- 현장 검증 대기: {field_summary['pending_field_validation_count']}
- 독립 evidence 대기: {field_summary['pending_evidence_count']}
- exact operational candidates: {exact_location_accepted}
- exact-location result: {exact_location_result}
- queue 내 비정확 proxy/centroid 위치: {field_summary['queue_proxy_not_exact_count']}
- release 금지 provisional facility point: {field_summary['provisional_coordinate_not_release_exact_count']}

## 6. BUSPROXY resolution

- 물리 후보가 있는 BUSPROXY: {field_summary['busproxy_anchor_with_physical_candidate_count']}/{field_summary['busproxy_anchor_count']}
- same-cluster 후보 보유: {field_summary['busproxy_same_cluster_candidate_ready_count']}/{field_summary['busproxy_anchor_count']}
- cross-cluster manual-only: {field_summary['busproxy_cross_cluster_manual_only_count']}
- 후보 mapping은 운영 확정이 아니며 BUSPROXY 자체는 final plan에 절대 들어가지 않는다.

## 7. Operational input provenance

- 입력 디렉터리: `{operational_audit.get('input_dir')}`
- 운영 preflight: `{operational_audit.get('preflight_ready', False)}`
- release-verified pool: {operational_audit.get('release_verified_physical_venue_count', 0)}
- 진단용 청주/충주 base 가정은 실제 base로 승격하지 않았다.

## 8. Travel methodology

- 상태: `{travel_details.get('status')}`
- 출처: `{travel_details.get('source', 'NOT_GENERATED')}`
- PBF SHA-256: `{travel_details.get('osm_pbf_sha256', operational_audit.get('internal_evidence', {}).get('osm_pbf_sha256'))}`
- OSM driving/oneway 정적 free-flow proxy이며 실시간 실제 소요시간이 아니다. Base×operational venue만 계산하고 unreachable/NaN을 0으로 바꾸지 않는다.

## 9. Operational optimizer formulation

기존 4단계 lexicographic objective/retention, 20회, admin≤2, sigungu≤4, coverage cluster≤1을 동결했다. 각 공간해에 대해 5개 bundle(각 최소 1), team×bundle×vehicle×venue×availability×명시적 no-conflict 공동 할당을 증명한다. 증명된 infeasible 20-set만 exact no-good으로 제외한다.

{chr(10).join(optimizer_lines)}

## 10. Efficiency / Balanced / Equity results

{chr(10).join(metric_lines)}

## 11. Solver certification

Authoritative proof는 CPU HiGHS이며 threshold는 `relative gap ≤ 0.005`다. GPU는 proof가 아니다. 운영 telemetry는 `06_optimizer/operational_solver_stage_certification.csv`에 direct bound/gap/certificate semantics를 분리해 기록했다.

## 12. Coverage/equity trade-off

Unique coverage는 실제 physical 좌표의 sparse OR로 한 번만 집계한다. 정책별 beneficiary Gini, visit-location Gini, redundancy, rural/urban unique exposure, bundle 및 travel 지표는 `07_final_plans/operational_policy_metrics.csv`에 있다.

## 13. Final 20-visit plan

세 정책이 모두 인증된 경우에만 `final_plan__efficiency.csv`, `final_plan__balanced.csv`, `final_plan__equity.csv`를 release한다. 공개 plan에는 방문 날짜를 넣지 않고 내부 availability witness는 SHA-256으로 결박한다.

## 14. Hard blockers and limitations

{blocker_text}

- 100m population grid는 calibrated proxy다.
- Stage2B는 `NO_STRONG_PREFERENCE`다.
- field evidence 출처와 operational input uncertainty를 숨기지 않는다.
- 범위는 사전등록 coarse-Pareto universe다.

## 15. Stage 5 interface

`{stage5_state}`. exact month/date/annual schedule은 생성하지 않았고 Stage5 namespace는 그대로 보존했다.
"""


def run_readiness(
    root: Path,
    config_path: Path,
    *,
    field_validation_path: Path | None = None,
    field_evidence_path: Path | None = None,
    operational_input_dir: Path | None = None,
) -> dict[str, Any]:
    root = root.resolve()
    config_path = config_path.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_root = _validate_config(root, config)
    if field_evidence_path is not None and field_validation_path is None:
        raise RuntimeError("--field-evidence requires --field-validation")
    baseline, baseline_audit = resolve_baseline(root)
    source_before = _source_contract(root, config_path)
    input_before = _input_manifest(
        field_validation_path,
        field_evidence_path,
        operational_input_dir,
    )
    stage5_before = _snapshot_stage5(root)
    parent_before = {
        "stage3": {"path": _relative(baseline.stage3_pointer_path, root), "sha256": _sha256(baseline.stage3_pointer_path)},
        "stage41": {"path": _relative(baseline.stage41_pointer_path, root), "sha256": _sha256(baseline.stage41_pointer_path)},
        "stage42g": {"path": _relative(baseline.g_pointer_path, root), "sha256": _sha256(baseline.g_pointer_path)},
        "stage42h": {"path": _relative(baseline.h_pointer_path, root), "sha256": _sha256(baseline.h_pointer_path)},
    }
    started = datetime.now(timezone.utc)
    contract_hash = hashlib.sha256(
        json.dumps(
            {
                "config_sha256": _sha256(config_path),
                "h_current_sha256": parent_before["stage42h"]["sha256"],
                "input_manifest": input_before.to_dict("records"),
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()[:12]
    run_id = (
        f"stage4_operational_final_{started.strftime('%Y%m%dT%H%M%SZ')}_"
        f"{contract_hash}_{uuid.uuid4().hex[:8]}"
    )
    run_root = output_root / "runs" / run_id
    lock_path, lock_token = _acquire_writer_lock(output_root)
    try:
        run_root.mkdir(parents=True, exist_ok=False)
        _write_text(
            run_root / ".RUNNING",
            f"run_id={run_id}\nstarted_utc={started.isoformat()}\npid={os.getpid()}",
        )
    except BaseException:
        _release_writer_lock(lock_path, lock_token)
        raise

    try:
        _write_json(run_root / "00_provenance/baseline_verification.json", baseline_audit)
        _write_json(run_root / "00_provenance/parent_pointers_before.json", parent_before)
        _write_json(run_root / "00_provenance/stage5_namespace_before.json", stage5_before)
        _write_csv(run_root / "00_provenance/source_contract_before.csv", source_before)
        _write_csv(run_root / "00_provenance/input_manifest_before.csv", input_before)
        _snapshot_inputs(run_root, input_before)
        _write_json(run_root / "00_provenance/hardware_audit.json", _hardware_audit(config))

        canonical = build_canonical_computational_summary(baseline)
        _write_json(run_root / "01_computational_parent/stage4_computational_final_canonical.json", canonical)
        union, plans, catalog, contributions = build_selected_union(baseline)
        narrow_identity_path = root / NARROW_IDENTITY_EVIDENCE_RELATIVE_PATH
        narrow_identity_evidence = read_table(narrow_identity_path)
        catalog = _apply_narrow_identity_evidence(
            catalog,
            narrow_identity_evidence,
            evidence_relative_path=NARROW_IDENTITY_EVIDENCE_RELATIVE_PATH,
        )
        _write_csv(run_root / "02_venue_resolution/coarse_final_selected_union.csv", union)
        _write_csv(run_root / "02_venue_resolution/policy_unique_coverage_contributions.csv", contributions)
        for policy, plan in plans.items():
            # Preserve the frozen parent's byte identity; ``plans`` is only the
            # normalized in-memory view used for downstream joins.
            _copy_file(
                baseline.plan_paths[policy],
                run_root / f"01_computational_parent/plan__{policy}.csv",
            )

        membership = pd.read_csv(baseline.membership_path, low_memory=False)
        queue, mapping, unresolved, _, field_summary = build_field_queue(
            union,
            catalog,
            membership,
            queue_min=int(config["field_validation"]["queue_min"]),
            queue_max=int(config["field_validation"]["queue_max"]),
        )
        field_summary["narrow_identity_evidence_relative_path"] = (
            NARROW_IDENTITY_EVIDENCE_RELATIVE_PATH
        )
        field_summary["narrow_identity_evidence_sha256"] = _sha256(
            narrow_identity_path
        )
        form, evidence = _field_form_and_evidence(queue)
        if field_validation_path:
            returned = audit_field_validation(read_table(field_validation_path))
            if set(returned.normalized["venue_id"].astype(str)) != set(queue["venue_id"].astype(str)):
                raise RuntimeError("Returned field-validation venue IDs do not exactly match the current coarse queue")
            form = returned.normalized
            field_summary["field_verified_count"] = int(form["field_verified"].sum())
            field_summary["field_validation_complete_count"] = int(form["field_validation_complete"].sum())
            field_summary["invalid_issue_count"] = int(returned.summary["invalid_issue_count"])
        else:
            field_summary["field_validation_complete_count"] = 0
            field_summary["invalid_issue_count"] = 0
        if field_evidence_path:
            evidence = read_table(field_evidence_path)
        evidence_audit = audit_field_evidence(
            evidence,
            form,
            expected_venue_ids=set(queue["venue_id"].astype(str)),
        )
        evidence = evidence_audit.normalized
        manual = _manual_required(queue, form, evidence)
        evidence_summary = dict(evidence_audit.summary)
        evidence_summary.update(
            {
                "pending_field_validation_count": int(
                    len(form) - form["field_validation_complete"].map(_truth).sum()
                ),
                "pending_evidence_count": int(
                    len(evidence) - evidence["field_evidence_complete"].map(_truth).sum()
                ),
                "manual_validation_required_count": int(len(manual)),
                "queue_proxy_not_exact_count": int(
                    queue["coordinate_source"]
                    .astype(str)
                    .str.contains("proxy|centroid|not_exact", case=False, regex=True, na=False)
                    .sum()
                ),
                "issue_count_semantics": (
                    "Issues count invalid evidence claims; pending UNKNOWN rows are reported separately"
                ),
            }
        )
        field_summary.update(evidence_summary)
        field_summary["field_verified_count"] = int(form["field_verified"].map(_truth).sum())
        field_summary["operational_release_verified_count"] = int(
            len(evidence_audit.release_verified_ids)
        )
        _write_csv(run_root / "02_venue_resolution/anchor_to_physical_candidate_mapping.csv", mapping)
        _write_csv(run_root / "02_venue_resolution/unresolved_busproxy_anchors.csv", unresolved)
        _write_json(run_root / "02_venue_resolution/venue_resolution_summary.json", field_summary)
        _write_csv(run_root / "03_field_validation/field_validation_priority_queue.csv", queue)
        _write_csv(run_root / "03_field_validation/field_validation_form.csv", form)
        _write_csv(run_root / "03_field_validation/field_validation_evidence.csv", evidence)
        _write_csv(run_root / "03_field_validation/field_validation_evidence_issues.csv", evidence_audit.issues)
        _write_json(run_root / "03_field_validation/field_validation_evidence_summary.json", evidence_summary)
        _write_csv(run_root / "03_field_validation/MANUAL_FIELD_VALIDATION_REQUIRED.csv", manual)
        _write_text(run_root / "03_field_validation/MANUAL_FIELD_VALIDATION_GUIDE.md", _manual_guide())

        verified_ids = set(evidence_audit.release_verified_ids)
        operational_universe = _build_operational_universe(
            baseline,
            queue,
            form,
            evidence,
            catalog,
            mapping,
        )
        _write_operational_universe(run_root, operational_universe)
        field_summary["operational_candidate_count"] = int(
            len(operational_universe.candidates)
        )
        field_summary["operational_candidate_exclusion_count"] = int(
            len(operational_universe.exclusions)
        )
        _write_json(run_root / "02_venue_resolution/venue_resolution_summary.json", field_summary)

        osm_pbf = root / "14_v6_external_data/road_network/raw/chungcheong-non-military_20260817.osm.pbf"
        template_venue_ids = (
            operational_universe.candidates["venue_id"].astype(str).tolist()
            if len(operational_universe.candidates)
            else queue["venue_id"].astype(str).tolist()
        )
        template_pack = write_operational_template_pack(
            run_root / "04_operational_inputs/templates_to_complete",
            template_venue_ids,
            osm_pbf_path=osm_pbf,
        )
        templates = {
            **template_pack.input_paths,
            "frozen_graph_source": template_pack.frozen_graph_source_path,
            "guide": template_pack.guide_path,
            "template_manifest": template_pack.manifest_path,
        }
        operational_audit, operational_ready, operational_blockers = _audit_operational_sources(
            root, operational_input_dir, verified_ids
        )
        operational_audit["generated_templates"] = {key: _relative(path, root) for key, path in templates.items()}
        candidate_ready = len(operational_universe.candidates) >= 20
        operational_tables: OperationalTables | None = None
        travel_details: dict[str, Any] = {
            "status": "BLOCKED_CONFIRMED_TEAM_BASES_AND_FIELD_VERIFIED_VENUES_REQUIRED",
            "osm_graph_available": operational_audit["internal_evidence"]["osm_graph_available"],
            "routing_contract": "driving; direction=oneway; retain_all=True; static free-flow proxy",
            "matrix_not_generated": True,
            "reason": "Do not fabricate origins or promote diagnostic bases.",
        }
        if operational_input_dir is not None and candidate_ready:
            operational_tables, prepared_travel, preparation_blockers = (
                _prepare_operational_tables_and_travel(
                    root,
                    run_root,
                    operational_input_dir,
                    operational_universe,
                    osm_pbf,
                )
            )
            if prepared_travel.get("generated"):
                operational_blockers = [
                    value
                    for value in operational_blockers
                    if value != "EMPTY_OPERATIONAL_INPUT:travel"
                ]
            operational_blockers.extend(preparation_blockers)
            operational_blockers = list(dict.fromkeys(operational_blockers))
            operational_ready = not operational_blockers and all(
                not getattr(operational_tables, key).empty
                for key in OPERATIONAL_INPUT_FILENAMES
            )
            travel_details = {
                **prepared_travel,
                "status": "READY_FOR_STRICT_POLICY_AUDIT" if operational_ready else "BLOCKED",
                "matrix_not_generated": not bool(
                    prepared_travel.get("generated")
                    or (operational_tables is not None and not operational_tables.travel.empty)
                ),
            }
        operational_audit["effective_blockers"] = operational_blockers
        operational_audit["preflight_ready"] = operational_ready
        operational_audit["operational_candidate_count"] = len(
            operational_universe.candidates
        )
        _write_json(run_root / "04_operational_inputs/operational_input_audit.json", operational_audit)
        _write_json(run_root / "05_travel/travel_status.json", travel_details)

        optimization_results: dict[str, OperationalOptimizationResult] = {}
        final_plans: dict[str, pd.DataFrame] = {}
        policy_metrics: list[dict[str, Any]] = []
        certification_rows: list[dict[str, Any]] = []
        optimization_attempted = bool(
            candidate_ready and operational_ready and operational_tables is not None
        )
        optimization_blockers: list[str] = []
        bundle_values = pd.DataFrame()
        if optimization_attempted:
            stage3 = discover_stage3(root)
            stage3_interface = normalize_stage3_interface(read_table(stage3.interface))
            bundle_values = bundle_table_from_interface(stage3_interface)
            expected_graph_sha = _sha256(osm_pbf)
            efficiency = optimize_operational_scenario(
                operational_universe.candidates,
                bundle_values,
                operational_tables,
                config,
                scenario="efficiency",
                coverage=operational_universe.coverage,
                grid=operational_universe.grid,
                expected_graph_sha=expected_graph_sha,
            )
            optimization_results["efficiency"] = efficiency
            if efficiency.ready:
                efficiency_reference = float(
                    efficiency.spatial_metrics["unique_elderly_population"]
                )
                for policy in ("balanced", "equity"):
                    optimization_results[policy] = optimize_operational_scenario(
                        operational_universe.candidates,
                        bundle_values,
                        operational_tables,
                        config,
                        scenario=policy,
                        coverage=operational_universe.coverage,
                        grid=operational_universe.grid,
                        efficiency_reference=efficiency_reference,
                        expected_graph_sha=expected_graph_sha,
                    )
            else:
                optimization_blockers.extend(efficiency.blockers)
                optimization_blockers.append(
                    f"EFFICIENCY_OPERATIONAL_OPTIMIZER_STATUS:{efficiency.status}"
                )
            for policy, result in optimization_results.items():
                plan, metrics = _write_operational_optimizer_result(
                    run_root,
                    result,
                    operational_universe,
                    bundle_values,
                )
                policy_metrics.append(metrics)
                if plan is not None:
                    final_plans[policy] = plan
                if not result.ready:
                    optimization_blockers.extend(result.blockers)
                    optimization_blockers.append(
                        f"{policy.upper()}_OPERATIONAL_OPTIMIZER_STATUS:{result.status}"
                    )
                if result.spatial_result is not None:
                    for telemetry in result.spatial_result.telemetry:
                        row = asdict(telemetry)
                        row.update(
                            {
                                "certificate_kind": (
                                    "OPTIMAL"
                                    if telemetry.solver_status == "OPTIMAL"
                                    else "DIRECT_MIP_GAP"
                                ),
                                "direct_relative_gap": telemetry.relative_gap,
                                "direct_best_bound": telemetry.best_bound,
                                "certified_gap_upper_bound": telemetry.relative_gap,
                                "final_certified": telemetry.certified,
                            }
                        )
                        certification_rows.append(row)
                _write_json(
                    run_root / f"06_optimizer/checkpoint__{policy}.json",
                    {
                        "config_sha256": _sha256(config_path),
                        "input_contract_sha256": contract_hash,
                        "operational_universe_sha256": operational_universe.artifact_sha256[
                            "operational_universe_sha256"
                        ],
                        "result": result.as_dict(),
                        "completed_policy_checkpoint_only": True,
                        "mid_mip_resume_supported": False,
                    },
                )
            for policy in POLICIES:
                if policy not in optimization_results:
                    _write_json(
                        run_root / f"06_optimizer/optimizer_result__{policy}.json",
                        {
                            "scenario": policy,
                            "status": "NOT_RUN_AFTER_EFFICIENCY_FAILURE",
                            "ready": False,
                            "certified": False,
                            "blockers": ["EFFICIENCY_REFERENCE_NOT_OPERATIONALLY_CERTIFIED"],
                        },
                    )
                    optimization_blockers.append(
                        f"{policy.upper()}_NOT_RUN_AFTER_EFFICIENCY_FAILURE"
                    )
        else:
            optimization_blockers.extend(
                [
                    "OPERATIONAL_REOPTIMIZATION_NOT_RUN",
                    "EFFICIENCY_BALANCED_EQUITY_OPERATIONAL_CERTIFICATION_PENDING",
                ]
            )

        all_policy_certified = bool(
            len(optimization_results) == 3
            and all(
                optimization_results[policy].ready
                and optimization_results[policy].certified
                for policy in POLICIES
            )
        )
        _write_csv(
            run_root / "06_optimizer/operational_solver_stage_certification.csv",
            pd.DataFrame(certification_rows),
        )
        _write_csv(
            run_root / "07_final_plans/operational_policy_metrics.csv",
            pd.DataFrame(policy_metrics),
        )
        _write_json(
            run_root / "06_optimizer/optimizer_status.json",
            {
                "status": (
                    "THREE_POLICY_OPERATIONALLY_CERTIFIED"
                    if all_policy_certified
                    else ("RUN_FAIL_CLOSED" if optimization_attempted else "NOT_RUN_FAIL_CLOSED")
                ),
                "visit_count": 20,
                "policies": list(POLICIES),
                "near_optimal_relative_gap_max": 0.005,
                "attempted": optimization_attempted,
                "certified_policies": [
                    policy
                    for policy, result in optimization_results.items()
                    if result.ready and result.certified
                ],
                "blockers": list(dict.fromkeys(optimization_blockers)),
                "gpu_is_not_proof": True,
            },
        )
        _write_json(
            run_root / "07_final_plans/final_plan_status.json",
            {
                "created": all_policy_certified,
                "created_policy_files": sorted(final_plans),
                "all_three_policy_plans_certified": all_policy_certified,
                "stage5_started": False,
                "exact_visit_dates_exposed": False,
                "reason": (
                    "All frozen policies passed spatial and joint operational certification."
                    if all_policy_certified
                    else "No release before all three policies and operational inputs pass."
                ),
            },
        )

        blockers = []
        if field_summary["operational_release_verified_count"] < 20:
            blockers.append(
                "FIELD_VALIDATION_OR_EVIDENCE_PENDING:"
                f"{field_summary['operational_release_verified_count']}_RELEASE_VERIFIED<{20}"
            )
        if len(operational_universe.candidates) < 20:
            blockers.append(
                "OPERATIONAL_EXACT_PHYSICAL_CANDIDATE_POOL_INSUFFICIENT:"
                f"{len(operational_universe.candidates)}<20"
            )
        if field_summary["busproxy_anchor_without_physical_candidate_count"]:
            blockers.append(
                f"BUSPROXY_PHYSICAL_CANDIDATE_MAPPING_PENDING:{field_summary['busproxy_anchor_without_physical_candidate_count']}"
            )
        blockers.extend(operational_blockers)
        blockers.extend(optimization_blockers)
        blockers = list(dict.fromkeys(blockers))
        plans_exact_twenty = bool(
            len(final_plans) == 3
            and all(
                len(final_plans[policy]) == 20
                and final_plans[policy]["venue_id"].astype(str).nunique() == 20
                for policy in POLICIES
            )
        )
        selected_busproxy_count = int(
            sum(
                plan["venue_id"]
                .astype(str)
                .str.upper()
                .str.startswith("BUSPROXY_")
                .sum()
                for plan in final_plans.values()
            )
        )
        selected_all_field_verified = bool(
            len(final_plans) == 3
            and all(plan["field_verified"].map(_truth).all() for plan in final_plans.values())
        )
        bundle_complete = bool(
            len(optimization_results) == 3
            and all(
                len(result.bundle_counts) == 5
                and min(result.bundle_counts.values(), default=0) >= 1
                for result in optimization_results.values()
            )
        )
        stage4_pass = bool(
            canonical["certified_stage_count"] == 12
            and candidate_ready
            and operational_ready
            and all_policy_certified
            and plans_exact_twenty
            and selected_busproxy_count == 0
            and selected_all_field_verified
            and bundle_complete
            and not blockers
        )
        decision_name = (
            "PASS_STAGE4_OPERATIONAL_FINAL"
            if stage4_pass
            else (
                "BLOCKED_STAGE4_OPERATIONAL_FINAL_SOLVER_OR_FEASIBILITY_NOT_CERTIFIED"
                if optimization_attempted
                else BLOCKED_DECISION
            )
        )
        gates = [
            {"gate_id": "computational_parent_sha_verified", "gate_type": "HARD", "passed": True, "observed": baseline.h_pointer["run_id"], "expected": "verified", "message": "Stage4.2H parent and inventory are sealed"},
            {"gate_id": "coarse_candidate_parent_verified", "gate_type": "HARD", "passed": True, "observed": baseline_audit["solver_candidate_count"], "expected": 896, "message": "Final candidate ladder rung"},
            {"gate_id": "canonical_12_stage_summary", "gate_type": "HARD", "passed": canonical["certified_stage_count"] == 12, "observed": canonical["certified_stage_count"], "expected": 12, "message": "Direct and rescue semantics separated"},
            {"gate_id": "coarse_selected_union_generated", "gate_type": "HARD", "passed": len(union) == 38, "observed": len(union), "expected": 38, "message": "E/B/Equity selected union"},
            {"gate_id": "field_queue_size", "gate_type": "HARD", "passed": int(config["field_validation"]["queue_min"]) <= len(queue) <= int(config["field_validation"]["queue_max"]), "observed": len(queue), "expected": f"{config['field_validation']['queue_min']}..{config['field_validation']['queue_max']}", "message": "High-value physical candidate queue"},
            {"gate_id": "field_verified_pool_sufficient", "gate_type": "OPERATIONAL", "passed": field_summary["operational_release_verified_count"] >= 20, "observed": field_summary["operational_release_verified_count"], "expected": ">=20", "message": "All final candidates must satisfy the unchanged field gate and evidence provenance gate"},
            {"gate_id": "operational_candidate_universe_generated", "gate_type": "OPERATIONAL", "passed": candidate_ready, "observed": len(operational_universe.candidates), "expected": ">=20", "message": "Only exact release-verified physical candidates with hard-5km coverage"},
            {"gate_id": "busproxy_resolution_candidates_complete", "gate_type": "OPERATIONAL", "passed": field_summary["busproxy_anchor_without_physical_candidate_count"] == 0, "observed": {"anchors": field_summary["busproxy_anchor_count"], "mapped": field_summary["busproxy_anchor_with_physical_candidate_count"], "unresolved": field_summary["busproxy_anchor_without_physical_candidate_count"], "same_cluster_ready": field_summary["busproxy_same_cluster_candidate_ready_count"], "cross_cluster_manual_only": field_summary["busproxy_cross_cluster_manual_only_count"]}, "expected": {"mapped_equals_anchors": True, "unresolved": 0}, "message": "Every BUSPROXY has physical validation candidates; mapping is not operational resolution"},
            {"gate_id": "operational_inputs_complete", "gate_type": "OPERATIONAL", "passed": operational_ready, "observed": operational_blockers, "expected": [], "message": "Team, vehicle, availability, travel, conflict inputs"},
            {"gate_id": "selected_final_venues_physical", "gate_type": "RELEASE", "passed": plans_exact_twenty and selected_busproxy_count == 0, "observed": {"plans_exact_twenty": plans_exact_twenty, "selected_busproxy_count": selected_busproxy_count}, "expected": {"plans_exact_twenty": True, "selected_busproxy_count": 0}, "message": "All final visits are physical venues"},
            {"gate_id": "selected_final_venues_field_verified", "gate_type": "RELEASE", "passed": selected_all_field_verified, "observed": selected_all_field_verified, "expected": True, "message": "Every released venue passed field and evidence gates"},
            {"gate_id": "operational_three_policy_certification", "gate_type": "RELEASE", "passed": all_policy_certified, "observed": [policy for policy, result in optimization_results.items() if result.ready and result.certified], "expected": list(POLICIES), "message": "Efficiency/Balanced/Equity each require four certified stages plus a joint operational witness"},
            {"gate_id": "bundle_assignment_complete", "gate_type": "RELEASE", "passed": bundle_complete, "observed": {policy: result.bundle_counts for policy, result in optimization_results.items()}, "expected": "five frozen bundles each >=1 in every policy", "message": "Team-compatible bundle assignment"},
            {"gate_id": "unique_coverage_recomputed", "gate_type": "HARD", "passed": True, "observed": {"sha256": operational_universe.artifact_sha256["operational_hard_5000m_canonical_sha256"], "candidate_rows": int(operational_universe.coverage.shape[0]), "grid_columns": int(operational_universe.coverage.shape[1]), "nnz": int(operational_universe.coverage.nnz), "applicable": bool(len(operational_universe.candidates)), "artifact_integrity_verified": True, "coverage_execution": "RECOMPUTED_OR_EXACT_ROW_REUSED" if len(operational_universe.candidates) else "NOT_RUN_EMPTY_RELEASE_POOL", "disposition": "RECOMPUTED_OR_EXACT_ROW_REUSED" if len(operational_universe.candidates) else "NOT_APPLICABLE_EMPTY_RELEASE_POOL_STRUCTURALLY_SEALED"}, "expected": "content-sealed sparse hard-5km matrix; empty release pool explicitly labeled", "message": "Actual coordinates reuse exact rows or trigger EPSG:5179 recomputation. N/A for an empty release pool: PASS verifies only CSR schema/content integrity, not venue coverage."},
            {"gate_id": "provenance_and_inventory_enforced_before_commit", "gate_type": "HARD", "passed": True, "observed": "POSTWRITE_FAIL_CLOSED_CHECK", "expected": "verified before .COMMITTED", "message": "Input/source/parent immutability and complete artifact inventory are enforced after report generation"},
            {"gate_id": "stage4_operational_final", "gate_type": "RELEASE", "passed": stage4_pass, "observed": decision_name, "expected": "PASS_STAGE4_OPERATIONAL_FINAL", "message": "Release remains fail-closed unless every gate passes"},
            {"gate_id": "stage5_namespace_untouched", "gate_type": "HARD", "passed": True, "observed": False, "expected": False, "message": "Stage5 was not started"},
        ]
        gate_frame = pd.DataFrame(gates)
        _write_csv(run_root / "08_quality_gate/quality_gate.csv", gate_frame)
        decision = {
            "decision": decision_name,
            "passed": stage4_pass,
            "stage4_computational_final": True,
            "stage4_operational_final": stage4_pass,
            "stage5_started": False,
            "stage5_release_allowed": stage4_pass,
            "blockers": blockers,
            "manual_field_validation_file": "03_field_validation/MANUAL_FIELD_VALIDATION_REQUIRED.csv",
            "field_validation_form": "03_field_validation/field_validation_form.csv",
            "field_evidence_file": "03_field_validation/field_validation_evidence.csv",
            "operational_input_template_dir": "04_operational_inputs/templates_to_complete",
            "final_plan_files": {
                policy: f"07_final_plans/final_plan__{policy}.csv"
                for policy in final_plans
            },
        }
        _write_json(run_root / "08_quality_gate/operational_final_decision.json", decision)
        if stage4_pass:
            _write_json(
                run_root / "07_final_plans/NEXT_STAGE_INTERFACE.json",
                {
                    "state": "STAGE5_READY",
                    "stage4_operational_final_decision": decision_name,
                    "stage5_started": False,
                    "exact_month_or_date_present": False,
                    "final_plan_sha256": {
                        policy: _sha256(
                            run_root / f"07_final_plans/final_plan__{policy}.csv"
                        )
                        for policy in POLICIES
                    },
                    "instruction": "Stage 5 may consume these plans but was not executed by this run.",
                },
            )
        ledger = pd.DataFrame(
            [
                {
                    "experiment_id": "E01_CANONICAL_PARENT",
                    "problem": "Raw rescue-era metrics can show certified=false or loose direct bounds",
                    "existing_method": "Downstream readers inspect raw per-scenario JSON",
                    "observed_blocker": "Direct gap and rescue certification semantics are conflated",
                    "code_change": "Created a 12-stage canonical summary with direct and certified gaps separated",
                    "policy_contract_changed": False,
                    "test": "12 rows; source hashes; direct log parsing",
                    "result": "PASS",
                    "improvement": "Unambiguous computational parent",
                    "regression": "None; parents remain immutable",
                    "final_decision": "ADOPT",
                },
                {
                    "experiment_id": "E02_COARSE_FIELD_QUEUE",
                    "problem": "Historical 47-row queue is tied to Top3",
                    "existing_method": "Reuse Stage4.1 queue",
                    "observed_blocker": "Coarse plan union contains different anchors and 18 BUSPROXY IDs",
                    "code_change": "Built a coarse 3-policy union and bounded physical candidate queue",
                    "policy_contract_changed": False,
                    "test": "38-union; queue bounds; BUSPROXY excluded from form",
                    "result": "PASS",
                    "improvement": "Current-plan field work is actionable",
                    "regression": "Field gate remains all-YES plus date",
                    "final_decision": "ADOPT",
                },
                {
                    "experiment_id": "E03_OPERATIONAL_SOLVE_GATE",
                    "problem": "Operational facts may be missing or mutually infeasible",
                    "existing_method": "Diagnostic bases or templates could be mistaken for facts",
                    "observed_blocker": "Real-world constraints cannot be inferred",
                    "code_change": "Eight-table fail-closed contract and exact user-fill templates",
                    "policy_contract_changed": False,
                    "test": "Joint team×bundle×vehicle×venue×availability×no-conflict audit",
                    "result": "PASS" if operational_ready else "BLOCKED_AS_DESIGNED",
                    "improvement": "Prevents fabricated or independently-feasible-only PASS",
                    "regression": "Requires explicit evidence for every operational tuple",
                    "final_decision": "ADOPT_FAIL_CLOSED_CONTRACT",
                },
                {
                    "experiment_id": "E04_EXACT_PHYSICAL_COVERAGE",
                    "problem": "Fallback coordinates can differ from frozen spatial anchors",
                    "existing_method": "Reuse anchor exposure and coverage row",
                    "observed_blocker": (
                        f"{len(verified_ids)} release IDs entered exact-location evaluation; "
                        f"{len(operational_universe.exclusions)} lacked exact usable locations"
                    ),
                    "code_change": "Exact-coordinate gate plus EPSG:5179 hard-5km sparse recomputation and cluster rebuild",
                    "policy_contract_changed": False,
                    "test": "CSR/order/content SHA, sparse OR unique coverage, BUSPROXY exclusion",
                    "result": (
                        "PASS"
                        if verified_ids
                        else "NOT_APPLICABLE_EMPTY_RELEASE_POOL"
                    ),
                    "improvement": f"{len(operational_universe.candidates)} audited operational candidates",
                    "regression": "Proxy/centroid candidates remain excluded until corrected",
                    "final_decision": (
                        "ADOPT"
                        if verified_ids
                        else "ADOPT_IMPLEMENTATION_NOT_EXERCISED_BY_REAL_RELEASE_ROWS"
                    ),
                },
                {
                    "experiment_id": "E05_OPERATIONAL_DECOMPOSITION",
                    "problem": "A certified spatial plan may lack a feasible joint resource assignment",
                    "existing_method": "Independent nonempty input checks",
                    "observed_blocker": "|".join(optimization_blockers),
                    "code_change": "Frozen spatial MILP plus exact infeasible-set no-good decomposition",
                    "policy_contract_changed": False,
                    "test": "Four stages per policy, gap≤0.005, five bundles min1, hidden availability witness",
                    "result": "PASS" if all_policy_certified else "BLOCKED_OR_NOT_RUN",
                    "improvement": "Operational feasibility and solver certification share one release gate",
                    "regression": "Can require repeated exact solves when many 20-sets are infeasible",
                    "final_decision": "ADOPT" if all_policy_certified else "KEEP_FAIL_CLOSED",
                },
            ]
        )
        _write_csv(run_root / "EXPERIMENT_LEDGER.csv", ledger)
        _write_text(
            run_root / "FINAL_REPORT.md",
            _final_report(
                baseline_audit,
                canonical,
                union,
                field_summary,
                blockers,
                gates,
                decision_name=decision_name,
                operational_audit=operational_audit,
                universe_summary=dict(operational_universe.summary),
                policy_metrics=policy_metrics,
                optimization_results=optimization_results,
                travel_details=travel_details,
            ),
        )

        parent_after = {
            key: {"path": value["path"], "sha256": _sha256(root / value["path"])} for key, value in parent_before.items()
        }
        source_after = _source_contract(root, config_path)
        input_after = _input_manifest(
            field_validation_path,
            field_evidence_path,
            operational_input_dir,
        )
        stage5_after = _snapshot_stage5(root)
        if parent_before != parent_after:
            raise RuntimeError("Authoritative parent pointer changed during readiness run")
        if not source_before.equals(source_after):
            raise RuntimeError("Operational source contract changed during readiness run")
        if not input_before.equals(input_after):
            raise RuntimeError("User-supplied field or operational inputs changed during run")
        if stage5_before != stage5_after:
            raise RuntimeError("Stage5 namespace changed during readiness run")
        _write_json(run_root / "00_provenance/parent_pointers_after.json", parent_after)
        _write_json(run_root / "00_provenance/stage5_namespace_after.json", stage5_after)
        _write_csv(run_root / "00_provenance/source_contract_after.csv", source_after)
        _write_csv(run_root / "00_provenance/input_manifest_after.csv", input_after)

        metadata = {
            "run_id": run_id,
            "started_utc": started.isoformat(),
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.version,
            "platform": platform.platform(),
            "config_relative_path": _relative(config_path, root),
            "config_sha256": _sha256(config_path),
            "hardware_profile": config["hardware"],
            "decision": decision["decision"],
            "decision_record": decision,
            "authoritative_parent_unchanged": True,
            "source_contract_unchanged": True,
            "input_manifest_unchanged": True,
            "stage5_namespace_unchanged": True,
        }
        _write_json(run_root / "metadata.json", metadata)
        inventory = _build_inventory(run_root)
        _write_csv(run_root / "ARTIFACT_INVENTORY.csv", inventory)
        inventory_audit = _verify_inventory(
            run_root, run_root / "ARTIFACT_INVENTORY.csv", require_complete=True
        )
        _write_json(run_root / "ARTIFACT_INVENTORY_AUDIT.json", inventory_audit)
        # The audit file is intentionally outside its own inventory; verify all inventoried rows again.
        _verify_inventory(
            run_root, run_root / "ARTIFACT_INVENTORY.csv", require_complete=True
        )
        current_written = False
        terminal_marker = ".FAILED"
        if stage4_pass:
            terminal_marker = ".COMMITTED"
            os.replace(run_root / ".RUNNING", run_root / terminal_marker)
            current_path = output_root / "CURRENT_STAGE4_OPERATIONAL_FINAL_RUN.json"
            current_payload = {
                "schema": "MEDIROAD_STAGE4_OPERATIONAL_FINAL_CURRENT_V1",
                "run_id": run_id,
                "run_relative_path": _relative(run_root, root),
                "decision": decision_name,
                "passed": True,
                "stage4_operational_final": True,
                "stage5_started": False,
                "stage5_release_allowed": True,
                "metadata_sha256": _sha256(run_root / "metadata.json"),
                "decision_sha256": _sha256(
                    run_root / "08_quality_gate/operational_final_decision.json"
                ),
                "quality_gate_sha256": _sha256(
                    run_root / "08_quality_gate/quality_gate.csv"
                ),
                "inventory_sha256": _sha256(
                    run_root / "ARTIFACT_INVENTORY.csv"
                ),
                "inventory_audit_sha256": _sha256(
                    run_root / "ARTIFACT_INVENTORY_AUDIT.json"
                ),
                "parent_stage42h_run_id": baseline.h_pointer["run_id"],
                "parent_stage42h_current_sha256": parent_before["stage42h"]["sha256"],
                "completed_utc": metadata["completed_utc"],
            }
            try:
                _write_json(current_path, current_payload)
            except BaseException:
                os.replace(run_root / ".COMMITTED", run_root / ".FAILED")
                raise
            current_written = True
        else:
            os.replace(run_root / ".RUNNING", run_root / terminal_marker)
        return {
            "run_id": run_id,
            "run_root": str(run_root),
            "decision": decision,
            "field_queue_count": int(len(queue)),
            "field_verified_count": int(field_summary["field_verified_count"]),
            "operational_release_verified_count": int(
                field_summary["operational_release_verified_count"]
            ),
            "artifact_inventory_rows": int(len(inventory)),
            "terminal_marker": terminal_marker,
            "current_operational_final_pointer_written": current_written,
        }
    except BaseException as exc:
        _write_json(
            run_root / "RUN_FAILURE.json",
            {
                "exception_type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
                "failed_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        if (run_root / ".RUNNING").exists():
            os.replace(run_root / ".RUNNING", run_root / ".FAILED")
        raise
    finally:
        _release_writer_lock(lock_path, lock_token)
