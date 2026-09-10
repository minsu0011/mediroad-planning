from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from mediroad.stage4_2d_equity_certification.io_utils import sha256_file
from mediroad.stage4_2d_equity_certification.runner import _verify_inventory

from .config import (
    EXPECTED_D_ARTIFACTS,
    FROZEN_D_RUN_ID,
    FROZEN_GAP,
)


@dataclass(frozen=True)
class AggregatedStage:
    scenario: str
    stage_index: int
    objective: str
    sense: str
    semantic_label: str
    certificate_basis: str
    source_stage: str
    source_run_id: str
    source_relative_path: str
    source_artifact_sha256: str
    incumbent: float
    best_bound: float
    relative_gap: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ParentEvidence:
    stage42d_run_id: str
    stage42e_run_id: str
    stage42d_pointer: dict[str, Any]
    stage42e_pointer: dict[str, Any]
    equity_stages: list[AggregatedStage]
    evidence_records: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "equity_stages": [stage.to_dict() for stage in self.equity_stages],
        }


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _record(root: Path, path: Path, expected: str) -> dict[str, Any]:
    root = root.resolve()
    path = path.resolve()
    path.relative_to(root)
    if not path.is_file():
        raise RuntimeError(f"Pinned Stage4.2F parent artifact is missing: {path}")
    observed = sha256_file(path)
    if observed != expected:
        raise RuntimeError(f"Pinned Stage4.2F parent artifact SHA mismatch: {path}")
    return {
        "relative_path": path.relative_to(root).as_posix(),
        "size_bytes": int(path.stat().st_size),
        "sha256": observed,
    }


def parent_records_unchanged(root: Path, records: list[dict[str, Any]]) -> bool:
    root = root.resolve()
    for row in records:
        path = (root / str(row["relative_path"])).resolve()
        try:
            path.relative_to(root)
        except ValueError:
            return False
        if (
            not path.is_file()
            or int(path.stat().st_size) != int(row["size_bytes"])
            or sha256_file(path) != str(row["sha256"])
        ):
            return False
    return True


def _pointer_artifact(root: Path, pointer: dict[str, Any], path_key: str, sha_key: str) -> Path:
    path = (root / str(pointer.get(path_key, ""))).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise RuntimeError(f"Parent pointer escapes project root: {path_key}") from exc
    if not path.is_file() or sha256_file(path) != str(pointer.get(sha_key, "")).lower():
        raise RuntimeError(f"Parent pointer artifact binding failed: {path_key}")
    return path


def _require_false_scope(
    payload: dict[str, Any], label: str, *, require_candidate_expansion: bool = True
) -> None:
    keys = [
        "stage4_full_computational_complete",
        "operational_final",
        "stage5_started",
        "stage5_release_allowed",
    ]
    if require_candidate_expansion:
        keys.insert(0, "candidate_expansion_certified")
    for key in keys:
        if payload.get(key) is not False:
            raise RuntimeError(f"{label} unexpectedly broadens scope: {key}")


def _stage(
    *,
    certificate: dict[str, Any],
    scenario_index: int,
    objective: str,
    sense: str,
    source_stage: str,
    source_run_id: str,
    source_path: Path,
    root: Path,
    require_fresh_only: bool = False,
) -> AggregatedStage:
    if certificate.get("certified") is not True:
        raise RuntimeError(f"Uncertified parent stage: {source_stage}")
    observed_objective = certificate.get("objective", certificate.get("metric"))
    if observed_objective != objective:
        raise RuntimeError(f"Parent objective changed: {source_stage}")
    if certificate.get("sense", sense) != sense:
        raise RuntimeError(f"Parent objective sense changed: {source_stage}")
    if require_fresh_only and (
        certificate.get("fresh_solve_completed") is not True
        or certificate.get("fresh_incumbent_valid") is not True
        or certificate.get("fresh_bound_valid") is not True
        or certificate.get("inherited_bound_used") is not False
    ):
        raise RuntimeError(
            f"Stage4.2E parent is not a fresh-incumbent/fresh-bound-only certificate: {source_stage}"
        )
    incumbent = float(certificate.get("incumbent_value"))
    bound = float(certificate.get("best_bound"))
    gap = float(certificate.get("relative_gap"))
    if not all(math.isfinite(value) for value in (incumbent, bound, gap)):
        raise RuntimeError(f"Parent stage has a non-finite certificate: {source_stage}")
    if gap < 0.0 or gap > FROZEN_GAP + 1e-12:
        raise RuntimeError(f"Parent stage exceeds frozen gap: {source_stage}")
    slack = 1e-10 * max(1.0, abs(incumbent), abs(bound))
    if (sense == "max" and bound < incumbent - slack) or (
        sense == "min" and bound > incumbent + slack
    ):
        raise RuntimeError(f"Parent stage bound direction is invalid: {source_stage}")
    return AggregatedStage(
        scenario="equity",
        stage_index=scenario_index,
        objective=objective,
        sense=sense,
        # Stage4.2F exposes one conservative semantic class.  It does not copy
        # a solver-specific status or upgrade a threshold proof to exactness.
        semantic_label="CERTIFIED_NEAR_OPTIMAL",
        certificate_basis="PINNED_OFFICIAL_PARENT_CERTIFICATE_NORMALIZED_TO_FROZEN_GAP",
        source_stage=source_stage,
        source_run_id=source_run_id,
        source_relative_path=source_path.relative_to(root.resolve()).as_posix(),
        source_artifact_sha256=sha256_file(source_path),
        incumbent=incumbent,
        best_bound=bound,
        relative_gap=gap,
    )


def load_parent_evidence(root: Path, cfg: dict[str, Any]) -> ParentEvidence:
    root = root.resolve()
    records: list[dict[str, Any]] = []

    d_pointer_path = (
        root
        / "outputs/model_v1/10_stage4_2d_equity_certification"
        / "CURRENT_STAGE4_2D_EQUITY_CERTIFICATION_RUN.json"
    )
    d_pins = cfg["stage42d_parent"]["artifact_sha256"]
    if d_pins != EXPECTED_D_ARTIFACTS:
        raise RuntimeError("Stage4.2D parent pins changed")
    records.append(
        _record(
            root,
            d_pointer_path,
            d_pins["CURRENT_STAGE4_2D_EQUITY_CERTIFICATION_RUN.json"],
        )
    )
    d_pointer = _json(d_pointer_path)
    if not isinstance(d_pointer, dict) or d_pointer.get("run_id") != FROZEN_D_RUN_ID:
        raise RuntimeError("Stage4.2D CURRENT does not name the exact frozen official run")
    if (
        d_pointer.get("decision") != "PASS_STAGE4_2D_EQUITY_FRONT_STAGES_CERTIFIED"
        or d_pointer.get("scope") != "TOP3_EQUITY_FRONT_STAGES_ONLY"
        or d_pointer.get("stage4_2d_front_stages_certified") is not True
    ):
        raise RuntimeError("Stage4.2D CURRENT is not the required front-stage PASS")
    _require_false_scope(
        d_pointer, "Stage4.2D CURRENT", require_candidate_expansion=False
    )
    d_run = (
        root
        / "outputs/model_v1/10_stage4_2d_equity_certification/runs"
        / FROZEN_D_RUN_ID
    )
    if not (d_run / ".COMMITTED").is_file():
        raise RuntimeError("Stage4.2D CURRENT run is not committed")
    for relative, digest in d_pins.items():
        if relative == "CURRENT_STAGE4_2D_EQUITY_CERTIFICATION_RUN.json":
            continue
        records.append(_record(root, d_run / relative, digest))
    verified_d_inventory = _verify_inventory(root, d_run / "ARTIFACT_INVENTORY.csv")
    if verified_d_inventory["sha256"] != d_pins["ARTIFACT_INVENTORY.csv"]:
        raise RuntimeError("Stage4.2D inventory binding changed")
    for path_key, sha_key in (
        ("inventory_relative_path", "inventory_sha256"),
        ("metadata_relative_path", "metadata_sha256"),
        ("report_relative_path", "report_sha256"),
        ("quality_gate_relative_path", "quality_gate_sha256"),
        ("min_certificate_relative_path", "min_certificate_sha256"),
        ("high_certificate_relative_path", "high_certificate_sha256"),
    ):
        _pointer_artifact(root, d_pointer, path_key, sha_key)
    d_gate = _json(d_run / "04_quality_and_provenance/quality_gate.json")
    d_gate_checks = d_gate.get("checks") if isinstance(d_gate, dict) else None
    if (
        not isinstance(d_gate, dict)
        or d_gate.get("decision")
        != "PASS_STAGE4_2D_EQUITY_FRONT_STAGES_CERTIFIED"
        or d_gate.get("passed") is not True
        or d_gate.get("computational_certified") is not True
        or d_gate.get("promotable_scope") != "TOP3_EQUITY_FRONT_STAGES_ONLY"
        or not isinstance(d_gate_checks, dict)
        or any(
            d_gate_checks.get(key) is not False
            for key in (
                "candidate_expansion_certified",
                "stage4_full_computational_complete",
                "operational_final",
                "stage5_started",
                "stage5_release_allowed",
            )
        )
    ):
        raise RuntimeError("Stage4.2D official quality gate is not PASS")

    e_parent = cfg["stage42e_parent"]
    e_run_id = str(e_parent["run_id"])
    e_pins = dict(e_parent["artifact_sha256"])
    e_pointer_path = (
        root
        / "outputs/model_v1/10_stage4_2e_equity_tail_certification"
        / "CURRENT_STAGE4_2E_EQUITY_TAIL_CERTIFICATION_RUN.json"
    )
    records.append(
        _record(
            root,
            e_pointer_path,
            e_pins["CURRENT_STAGE4_2E_EQUITY_TAIL_CERTIFICATION_RUN.json"],
        )
    )
    e_pointer = _json(e_pointer_path)
    if not isinstance(e_pointer, dict) or e_pointer.get("run_id") != e_run_id:
        raise RuntimeError("Stage4.2E CURRENT does not name the exact config-pinned run")
    if (
        e_pointer.get("decision") != "PASS_STAGE4_2E_TOP3_EQUITY_FULL_CERTIFIED"
        or e_pointer.get("scope") != "TOP3_EQUITY_FULL_ONLY"
        or e_pointer.get("top3_equity_full_lexicographic_certified") is not True
    ):
        raise RuntimeError("Stage4.2E CURRENT is not the required Equity-full PASS")
    _require_false_scope(e_pointer, "Stage4.2E CURRENT")
    e_run = (
        root
        / "outputs/model_v1/10_stage4_2e_equity_tail_certification/runs"
        / e_run_id
    )
    if not (e_run / ".COMMITTED").is_file():
        raise RuntimeError("Stage4.2E CURRENT run is not committed")
    for relative, digest in e_pins.items():
        if relative == "CURRENT_STAGE4_2E_EQUITY_TAIL_CERTIFICATION_RUN.json":
            continue
        records.append(_record(root, e_run / relative, digest))
    verified_e_inventory = _verify_inventory(root, e_run / "ARTIFACT_INVENTORY.csv")
    if verified_e_inventory["sha256"] != e_pins["ARTIFACT_INVENTORY.csv"]:
        raise RuntimeError("Stage4.2E inventory binding changed")
    for path_key, sha_key in (
        ("inventory_relative_path", "inventory_sha256"),
        ("metadata_relative_path", "metadata_sha256"),
        ("report_relative_path", "report_sha256"),
        ("quality_gate_relative_path", "quality_gate_sha256"),
        ("need_certificate_relative_path", "need_certificate_sha256"),
        ("cost_certificate_relative_path", "cost_certificate_sha256"),
    ):
        _pointer_artifact(root, e_pointer, path_key, sha_key)
    e_gate = _json(e_run / "04_quality_and_provenance/quality_gate.json")
    if (
        not isinstance(e_gate, dict)
        or e_gate.get("passed") is not True
        or e_gate.get("top3_equity_full_lexicographic_certified") is not True
    ):
        raise RuntimeError("Stage4.2E official quality gate is not Equity-full PASS")

    d_min_path = d_run / "01_min_threshold_oracle/certificate.json"
    d_high_path = d_run / "02_high_threshold_oracle/certificate.json"
    e_need_path = e_run / "01_need_weighted/certificate.json"
    e_cost_path = e_run / "02_cost/certificate.json"
    stages = [
        _stage(
            certificate=_json(d_min_path),
            scenario_index=1,
            objective="min_sigungu_coverage",
            sense="max",
            source_stage="STAGE4_2D_FRONT_1",
            source_run_id=FROZEN_D_RUN_ID,
            source_path=d_min_path,
            root=root,
        ),
        _stage(
            certificate=_json(d_high_path),
            scenario_index=2,
            objective="high_need_population",
            sense="max",
            source_stage="STAGE4_2D_FRONT_2",
            source_run_id=FROZEN_D_RUN_ID,
            source_path=d_high_path,
            root=root,
        ),
        _stage(
            certificate=_json(e_need_path),
            scenario_index=3,
            objective="need_weighted",
            sense="max",
            source_stage="STAGE4_2E_TAIL_3",
            source_run_id=e_run_id,
            source_path=e_need_path,
            root=root,
            require_fresh_only=True,
        ),
        _stage(
            certificate=_json(e_cost_path),
            scenario_index=4,
            objective="cost",
            sense="min",
            source_stage="STAGE4_2E_TAIL_4",
            source_run_id=e_run_id,
            source_path=e_cost_path,
            root=root,
            require_fresh_only=True,
        ),
    ]
    return ParentEvidence(
        stage42d_run_id=FROZEN_D_RUN_ID,
        stage42e_run_id=e_run_id,
        stage42d_pointer=d_pointer,
        stage42e_pointer=e_pointer,
        equity_stages=stages,
        evidence_records=records,
    )
