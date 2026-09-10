from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mediroad.stage4_2_certification.formulation import CertificationFormulation
from mediroad.stage4_2d_equity_certification.io_utils import sha256_file
from mediroad.stage4_2d_equity_certification.runner import _verify_inventory

from .config import (
    FROZEN_C_RUN_ID,
    FROZEN_D_RUN_ID,
    FROZEN_HIGH_NEED_FLOOR,
    FROZEN_HIGH_NEED_INCUMBENT,
    FROZEN_MIN_SIGUNGU_FLOOR,
    FROZEN_OLD_COST_BOUND,
    FROZEN_OLD_COST_INCUMBENT,
    FROZEN_OLD_HIGH_NEED_FLOOR,
    FROZEN_OLD_NEED_BOUND,
    FROZEN_OLD_NEED_FLOOR,
    FROZEN_OLD_NEED_INCUMBENT,
    FROZEN_TOTAL_POPULATION_FLOOR,
)


@dataclass(frozen=True)
class TailEvidence:
    stage42d_run_id: str
    stage42c_run_id: str
    d_front_selection: np.ndarray
    old_need_selection: np.ndarray
    old_cost_selection: np.ndarray
    old_need_incumbent: float
    old_need_bound: float
    old_need_floor: float
    old_cost_incumbent: float
    old_cost_bound: float
    artifact_sha_verified: bool
    records: list[dict[str, Any]]
    d_pointer_payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _same(value: Any, expected: float) -> bool:
    return type(value) in (int, float) and float(value).hex() == float(expected).hex()


def _record(root: Path, path: Path, expected: str) -> dict[str, Any]:
    path = path.resolve()
    path.relative_to(root.resolve())
    if not path.is_file():
        raise RuntimeError(f"Pinned evidence is missing: {path}")
    actual = sha256_file(path)
    if actual != str(expected).lower():
        raise RuntimeError(f"Pinned evidence SHA mismatch: {path}")
    return {
        "relative_path": path.relative_to(root.resolve()).as_posix(),
        "size_bytes": int(path.stat().st_size),
        "sha256": actual,
    }


def _plan_selection(path: Path, formulation: CertificationFormulation) -> np.ndarray:
    frame = pd.read_csv(path)
    if "venue_id" not in frame or len(frame) != formulation.visit_count:
        raise RuntimeError(f"Invalid pinned plan: {path}")
    lookup = {
        str(venue): int(index)
        for index, venue in enumerate(formulation.candidates["venue_id"].astype(str))
    }
    selected: list[int] = []
    for raw in frame["venue_id"].astype(str):
        venue = formulation.venue_alias_map.get(raw, raw)
        if venue not in lookup:
            raise RuntimeError(f"Pinned plan venue is outside frozen Top3 universe: {venue}")
        selected.append(lookup[venue])
    result = np.asarray(selected, dtype=int)
    valid, reason = formulation.validate_selection(result)
    if not valid:
        raise RuntimeError(f"Pinned plan violates current hard constraints: {reason}")
    return result


def _stage_selection(row: dict[str, Any], formulation: CertificationFormulation) -> np.ndarray:
    raw = row.get("selected_indices")
    if not isinstance(raw, list) or any(type(value) is not int for value in raw):
        raise RuntimeError("Pinned Stage4.2C stage has no exact integer selection")
    selected = np.asarray(raw, dtype=int)
    valid, reason = formulation.validate_selection(selected)
    if not valid:
        raise RuntimeError(f"Pinned Stage4.2C selection violates hard constraints: {reason}")
    return selected


def _assert_floor_list(rows: Any, expected: list[tuple[str, str, float]]) -> None:
    if not isinstance(rows, list) or len(rows) != len(expected):
        raise RuntimeError("Pinned Stage4.2C floor list changed")
    for row, (objective, sense, value) in zip(rows, expected):
        if (
            not isinstance(row, dict)
            or row.get("objective") != objective
            or row.get("sense") != sense
            or not _same(row.get("value"), value)
        ):
            raise RuntimeError("Pinned Stage4.2C floor semantics changed")


def load_tail_evidence(
    root: Path,
    formulation: CertificationFormulation,
    cfg: dict[str, Any],
) -> TailEvidence:
    """Load and fully revalidate the D parent and C bound evidence."""

    root = root.resolve()
    records: list[dict[str, Any]] = []
    parent_cfg = cfg["stage42d_parent"]
    pointer = root / "outputs/model_v1/10_stage4_2d_equity_certification/CURRENT_STAGE4_2D_EQUITY_CERTIFICATION_RUN.json"
    d_pins = dict(parent_cfg["artifact_sha256"])
    records.append(
        _record(root, pointer, d_pins["CURRENT_STAGE4_2D_EQUITY_CERTIFICATION_RUN.json"])
    )
    pointer_payload = _json(pointer)
    if not isinstance(pointer_payload, dict) or pointer_payload.get("run_id") != FROZEN_D_RUN_ID:
        raise RuntimeError("CURRENT Stage4.2D pointer is not the frozen PASS run")
    if pointer_payload.get("stage4_2d_front_stages_certified") is not True:
        raise RuntimeError("Pinned Stage4.2D pointer is not front-stage certified")
    if pointer_payload.get("scope") != "TOP3_EQUITY_FRONT_STAGES_ONLY":
        raise RuntimeError("Pinned Stage4.2D pointer scope changed")
    if pointer_payload.get("stage5_started") is not False:
        raise RuntimeError("Pinned Stage4.2D pointer claims Stage5")
    d_run = root / "outputs/model_v1/10_stage4_2d_equity_certification/runs" / FROZEN_D_RUN_ID
    if not (d_run / ".COMMITTED").is_file():
        raise RuntimeError("Pinned Stage4.2D run is not committed")
    for relative, digest in d_pins.items():
        if relative == "CURRENT_STAGE4_2D_EQUITY_CERTIFICATION_RUN.json":
            continue
        records.append(_record(root, d_run / relative, digest))
    inventory = _verify_inventory(root, d_run / "ARTIFACT_INVENTORY.csv")
    if inventory["sha256"] != d_pins["ARTIFACT_INVENTORY.csv"]:
        raise RuntimeError("Pinned Stage4.2D inventory trust anchor changed")
    for path_key, sha_key in (
        ("metadata_relative_path", "metadata_sha256"),
        ("inventory_relative_path", "inventory_sha256"),
        ("report_relative_path", "report_sha256"),
        ("quality_gate_relative_path", "quality_gate_sha256"),
        ("min_certificate_relative_path", "min_certificate_sha256"),
        ("high_certificate_relative_path", "high_certificate_sha256"),
    ):
        path = root / str(pointer_payload.get(path_key, ""))
        if not path.is_file() or sha256_file(path) != str(pointer_payload.get(sha_key, "")).lower():
            raise RuntimeError(f"Stage4.2D pointer artifact binding failed: {path_key}")

    min_cert = _json(d_run / "01_min_threshold_oracle/certificate.json")
    high_cert = _json(d_run / "02_high_threshold_oracle/certificate.json")
    quality = _json(d_run / "04_quality_and_provenance/quality_gate.json")
    if not isinstance(quality, dict) or quality.get("passed") is not True:
        raise RuntimeError("Pinned Stage4.2D quality gate is not PASS")
    if not isinstance(min_cert, dict) or min_cert.get("certified") is not True:
        raise RuntimeError("Pinned Stage4.2D min certificate is invalid")
    if not isinstance(high_cert, dict) or high_cert.get("certified") is not True:
        raise RuntimeError("Pinned Stage4.2D high certificate is invalid")
    if not _same(min_cert.get("incumbent_value"), 0.5252179790669655):
        raise RuntimeError("Pinned Stage4.2D min incumbent changed")
    if not _same(high_cert.get("incumbent_value"), FROZEN_HIGH_NEED_INCUMBENT):
        raise RuntimeError("Pinned Stage4.2D high incumbent changed")
    d_front = _plan_selection(
        d_run / "03_final_incumbent/plan__equity_front_stage_seed.csv", formulation
    )
    high_raw = high_cert.get("selected_indices")
    if not isinstance(high_raw, list) or set(map(int, high_raw)) != set(map(int, d_front)):
        raise RuntimeError("Pinned Stage4.2D front plan is not the high certificate selection")
    d_metrics = formulation.metrics(d_front)
    if not _same(d_metrics["high_need_population"], FROZEN_HIGH_NEED_INCUMBENT):
        raise RuntimeError("CPU recomputation of pinned D high incumbent changed")
    if float(d_metrics["min_sigungu_coverage_ratio"]) < FROZEN_MIN_SIGUNGU_FLOOR:
        raise RuntimeError("Pinned D plan violates its retained min-sigungu floor")

    old_cfg = cfg["stage42c_bound_evidence"]
    c_run = root / "outputs/model_v1/10_stage4_2_certification_diagnostic/runs" / FROZEN_C_RUN_ID
    if not (c_run / ".COMMITTED").is_file():
        raise RuntimeError("Pinned Stage4.2C evidence run is not committed")
    c_pins = dict(old_cfg["artifact_sha256"])
    for relative, digest in c_pins.items():
        records.append(_record(root, c_run / relative, digest))
    c_inventory = _verify_inventory(root, c_run / "ARTIFACT_INVENTORY.csv")
    if c_inventory["sha256"] != c_pins["ARTIFACT_INVENTORY.csv"]:
        raise RuntimeError("Pinned Stage4.2C inventory trust anchor changed")
    stages = _json(c_run / "candidate_sets/top3/stages__equity.json")
    if not isinstance(stages, list) or len(stages) != 4:
        raise RuntimeError("Pinned Stage4.2C Equity stage trace changed")
    expected_order = [
        (1, "min_sigungu_coverage", "max"),
        (2, "high_need_population", "max"),
        (3, "need_weighted", "max"),
        (4, "cost", "min"),
    ]
    for row, expected in zip(stages, expected_order):
        if (
            not isinstance(row, dict)
            or (row.get("stage_index"), row.get("objective_name"), row.get("sense")) != expected
            or row.get("scenario") != "equity"
            or row.get("candidate_set") != "top3"
        ):
            raise RuntimeError("Pinned Stage4.2C Equity stage order changed")
    need_row, cost_row = stages[2], stages[3]
    for row, incumbent, bound in (
        (need_row, FROZEN_OLD_NEED_INCUMBENT, FROZEN_OLD_NEED_BOUND),
        (cost_row, FROZEN_OLD_COST_INCUMBENT, FROZEN_OLD_COST_BOUND),
    ):
        if row.get("certified") is not True or not _same(row.get("objective_value"), incumbent) or not _same(
            row.get("best_bound"), bound
        ):
            raise RuntimeError("Pinned Stage4.2C inherited bound evidence changed")
    if not isinstance(need_row.get("retained_floor"), dict) or not _same(
        need_row["retained_floor"].get("value"), FROZEN_OLD_NEED_FLOOR
    ):
        raise RuntimeError("Pinned Stage4.2C need retained floor changed")
    old_need_floors = [
        ("total_population", "max", FROZEN_TOTAL_POPULATION_FLOOR),
        ("min_sigungu_coverage", "max", FROZEN_MIN_SIGUNGU_FLOOR),
        ("high_need_population", "max", FROZEN_OLD_HIGH_NEED_FLOOR),
    ]
    _assert_floor_list(need_row.get("details", {}).get("floors_before_stage"), old_need_floors)
    _assert_floor_list(
        cost_row.get("details", {}).get("floors_before_stage"),
        [*old_need_floors, ("need_weighted", "max", FROZEN_OLD_NEED_FLOOR)],
    )
    old_need = _stage_selection(need_row, formulation)
    old_cost = _stage_selection(cost_row, formulation)
    if not _same(formulation.objective_for_selection("need_weighted", old_need), FROZEN_OLD_NEED_INCUMBENT):
        raise RuntimeError("CPU recomputation of old need incumbent changed")
    if not _same(formulation.objective_for_selection("cost", old_cost), FROZEN_OLD_COST_INCUMBENT):
        raise RuntimeError("CPU recomputation of old cost incumbent changed")

    return TailEvidence(
        stage42d_run_id=FROZEN_D_RUN_ID,
        stage42c_run_id=FROZEN_C_RUN_ID,
        d_front_selection=d_front,
        old_need_selection=old_need,
        old_cost_selection=old_cost,
        old_need_incumbent=FROZEN_OLD_NEED_INCUMBENT,
        old_need_bound=FROZEN_OLD_NEED_BOUND,
        old_need_floor=FROZEN_OLD_NEED_FLOOR,
        old_cost_incumbent=FROZEN_OLD_COST_INCUMBENT,
        old_cost_bound=FROZEN_OLD_COST_BOUND,
        artifact_sha_verified=True,
        records=records,
        d_pointer_payload=pointer_payload,
    )


def evidence_records_unchanged(root: Path, records: list[dict[str, Any]]) -> bool:
    root = root.resolve()
    for row in records:
        path = root / str(row["relative_path"])
        if (
            not path.is_file()
            or int(path.stat().st_size) != int(row["size_bytes"])
            or sha256_file(path) != str(row["sha256"])
        ):
            return False
    return True
