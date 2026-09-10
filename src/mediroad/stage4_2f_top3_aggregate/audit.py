from __future__ import annotations

import importlib.util
import json
import math
import sys
import types as module_types
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from mediroad.stage4_2_certification.formulation import CertificationFormulation
from mediroad.stage4_2_certification.types import Floor
from mediroad.stage4_2d_equity_certification.canonical import (
    canonical_model_digest,
    canonical_universe_digest,
)
from mediroad.stage4_2d_equity_certification.io_utils import sha256_file
from mediroad.stage4_2d_equity_certification.runner import _verify_inventory
from mediroad.stage4_2d_equity_certification.threshold_oracles import selection_digest

from .config import (
    EXPECTED_ADOPTION_STAGES,
    EXPECTED_C_ARTIFACTS,
    EXPECTED_C_PARENT_POINTERS,
    EXPECTED_C_SOURCE_PREIMAGE,
    FROZEN_C_INVENTORY_SHA256,
    FROZEN_C_RUN_ID,
    FROZEN_GAP,
)


@dataclass(frozen=True)
class AdoptedStage:
    scenario: str
    stage_index: int
    objective: str
    sense: str
    semantic_label: str
    certificate_basis: str
    source_run_id: str
    source_relative_path: str
    source_artifact_sha256: str
    incumbent: float
    best_bound: float
    relative_gap: float
    current_recomputed_objective: float
    current_selection_feasible: bool
    feasibility_tolerance: float
    max_constraint_violation: float
    historical_preimage_recomputed_objective: float
    historical_preimage_selection_feasible: bool
    historical_preimage_model_sha256: str
    historical_preimage_universe_sha256: str
    historical_preimage_selection_sha256: str
    historical_current_model_equal: bool
    model_sha256: str
    universe_sha256: str
    selection_sha256: str
    model_rows: int
    model_cols: int
    model_nnz: int
    floors: list[dict[str, Any]]
    legacy_solver_claims_used: bool
    legacy_run_pass_used: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CompatibilityAudit:
    passed: bool
    source_run_id: str
    source_inventory_sha256: str
    candidate_count: int
    pattern_count: int
    visit_count: int
    stage_count: int
    stages: list[AdoptedStage]
    scenario_classification: dict[str, str]
    historical_source_preimage_verified: bool
    evidence_records: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "stages": [stage.to_dict() for stage in self.stages],
        }


def _same(value: Any, expected: float) -> bool:
    return bool(
        type(value) in (int, float)
        and math.isfinite(float(value))
        and float(value).hex() == float(expected).hex()
    )


def _record(root: Path, path: Path, expected: str) -> dict[str, Any]:
    root = root.resolve()
    path = path.resolve()
    path.relative_to(root)
    if not path.is_file():
        raise RuntimeError(f"Pinned compatibility evidence is missing: {path}")
    observed = sha256_file(path)
    if observed != expected:
        raise RuntimeError(f"Pinned compatibility evidence SHA mismatch: {path}")
    return {
        "relative_path": path.relative_to(root).as_posix(),
        "size_bytes": int(path.stat().st_size),
        "sha256": observed,
    }


def evidence_records_unchanged(root: Path, records: list[dict[str, Any]]) -> bool:
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


def _universe_digest(
    formulation: CertificationFormulation, model: Any
) -> str:
    candidate_ids = formulation.candidates["venue_id"].astype(str).tolist()
    pattern_ids = list(range(formulation.ny))
    semantic_ids = {
        "candidate_admin_code": formulation.candidates["admin_code"].astype(str).tolist(),
        "candidate_sigungu": formulation.candidates["sigungu"].astype(str).tolist(),
        "candidate_cluster": formulation.candidates["cluster_id"].fillna(
            formulation.candidates["venue_id"]
        ).astype(str).tolist(),
        "pattern_sigungu": np.asarray(formulation.pattern_sigungu, dtype=str).tolist(),
    }
    return canonical_universe_digest(
        model,
        candidate_ids=candidate_ids,
        pattern_ids=pattern_ids,
        semantic_ids=semantic_ids,
    )


def _historical_formulation(root: Path) -> tuple[Any, Any]:
    """Load the C-run source preimage under an isolated module namespace."""

    source = (
        root
        / "MEDIROAD_STAGE4_2C_CERTIFICATION_BOTTLENECK_PATCH_20260820"
        / "src/mediroad/stage4_2_certification"
    )
    package_name = "_mediroad_stage42c_frozen_for_stage42f"
    for name in (
        f"{package_name}.formulation",
        f"{package_name}.adapter",
        f"{package_name}.types",
        package_name,
    ):
        sys.modules.pop(name, None)
    package = module_types.ModuleType(package_name)
    package.__path__ = [str(source)]  # type: ignore[attr-defined]
    sys.modules[package_name] = package
    for module_name in ("types", "adapter", "formulation"):
        qualified = f"{package_name}.{module_name}"
        spec = importlib.util.spec_from_file_location(qualified, source / f"{module_name}.py")
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load frozen Stage4.2C source module: {module_name}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[qualified] = module
        spec.loader.exec_module(module)
    adapter = sys.modules[f"{package_name}.adapter"]
    formulation_module = sys.modules[f"{package_name}.formulation"]
    types_module = sys.modules[f"{package_name}.types"]
    base = adapter.load_base_stage42_config(root / "configs/model_v1/stage4_2.yaml")
    diagnostic_config = yaml.safe_load(
        (
            root
            / "MEDIROAD_STAGE4_2C_CERTIFICATION_BOTTLENECK_PATCH_20260820"
            / "configs/model_v1/stage4_2_certification_diagnostic.yaml"
        ).read_text(encoding="utf-8")
    )
    if not isinstance(diagnostic_config, dict):
        raise RuntimeError("Frozen Stage4.2C diagnostic config is not a mapping")
    problem = adapter.prepare_candidate_problem(root, "top3", base, cache={})
    formulation = formulation_module.CertificationFormulation(
        problem.candidates,
        problem.patterns,
        base,
        diagnostic_config,
        candidate_set="top3",
    )
    formulation.venue_alias_map = dict(problem.venue_alias_map)
    return formulation, types_module.Floor


def _floors(raw: Any) -> list[Floor]:
    if not isinstance(raw, list):
        raise RuntimeError("Compatibility source has no exact floor list")
    output: list[Floor] = []
    for row in raw:
        if (
            not isinstance(row, dict)
            or set(row) != {"objective", "sense", "value"}
            or row.get("sense") not in {"max", "min"}
            or not isinstance(row.get("objective"), str)
            or type(row.get("value")) not in (int, float)
            or not math.isfinite(float(row["value"]))
        ):
            raise RuntimeError("Compatibility source floor semantics changed")
        output.append(
            Floor(
                objective=str(row["objective"]),
                sense=str(row["sense"]),  # type: ignore[arg-type]
                value=float(row["value"]),
            )
        )
    return output


def _bound_gap(sense: str, incumbent: float, bound: float) -> tuple[bool, float]:
    scale = max(abs(incumbent), 1e-12)
    # HiGHS may report a bound a few floating additions inside an incumbent.
    # The slack only validates direction; it never enlarges the reported gap.
    slack = 1e-10 * max(1.0, abs(incumbent), abs(bound))
    if sense == "max":
        direction = bound >= incumbent - slack
        gap = max(0.0, (bound - incumbent) / scale) if direction else math.inf
    elif sense == "min":
        direction = bound <= incumbent + slack
        gap = max(0.0, (incumbent - bound) / scale) if direction else math.inf
    else:
        raise RuntimeError(f"Unsupported adoption sense: {sense}")
    return direction, float(gap)


def _selection_from_plan(
    path: Path, formulation: CertificationFormulation
) -> np.ndarray:
    frame = pd.read_csv(path)
    if "venue_id" not in frame or len(frame) != formulation.visit_count:
        raise RuntimeError(f"Pinned final plan schema changed: {path}")
    lookup = {
        str(venue): int(index)
        for index, venue in enumerate(formulation.candidates["venue_id"].astype(str))
    }
    selected: list[int] = []
    for raw in frame["venue_id"].astype(str):
        venue = formulation.venue_alias_map.get(raw, raw)
        if venue not in lookup:
            raise RuntimeError(f"Pinned final plan is outside current Top3 universe: {venue}")
        selected.append(lookup[venue])
    result = np.asarray(selected, dtype=np.int64)
    valid, reason = formulation.validate_selection(result)
    if not valid:
        raise RuntimeError(f"Pinned final plan violates current hard constraints: {reason}")
    return result


def compatibility_adoption_audit(
    root: Path,
    formulation: CertificationFormulation,
    cfg: dict[str, Any],
) -> CompatibilityAudit:
    """Rebuild and seal the eight Efficiency/Balanced models without solving.

    Historical solver status strings and the historical run-level decision are
    deliberately neither read as certification evidence nor copied to the
    result.  Certification is adopted from the pinned incumbent/bound pair only
    after the exact current model, universe and selection seals match and the
    current CPU recomputation is feasible.
    """

    root = root.resolve()
    source = (
        root
        / "outputs/model_v1/10_stage4_2_certification_diagnostic/runs"
        / FROZEN_C_RUN_ID
    )
    if not (source / ".COMMITTED").is_file():
        raise RuntimeError("Preferred Stage4.2C compatibility source is not committed")
    configured = cfg["preferred_stage42c"]["artifact_sha256"]
    if configured != EXPECTED_C_ARTIFACTS:
        raise RuntimeError("Preferred Stage4.2C trust anchors changed")
    records = [
        _record(root, source / relative, digest)
        for relative, digest in EXPECTED_C_ARTIFACTS.items()
    ]
    for relative, digest in EXPECTED_C_SOURCE_PREIMAGE.items():
        records.append(_record(root, root / relative, digest))
    for relative, digest in EXPECTED_C_PARENT_POINTERS.items():
        records.append(_record(root, root / relative, digest))
    verified_inventory = _verify_inventory(root, source / "ARTIFACT_INVENTORY.csv")
    if verified_inventory["sha256"] != FROZEN_C_INVENTORY_SHA256:
        raise RuntimeError("Preferred Stage4.2C inventory trust anchor changed")
    metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    if not isinstance(metadata, dict) or metadata.get("run_id") != FROZEN_C_RUN_ID:
        raise RuntimeError("Preferred Stage4.2C metadata run identity changed")
    if (
        metadata.get("stage42_config_sha256")
        != EXPECTED_C_SOURCE_PREIMAGE["configs/model_v1/stage4_2.yaml"]
        or metadata.get("certification_config_sha256")
        != EXPECTED_C_SOURCE_PREIMAGE[
            "MEDIROAD_STAGE4_2C_CERTIFICATION_BOTTLENECK_PATCH_20260820/configs/model_v1/stage4_2_certification_diagnostic.yaml"
        ]
    ):
        raise RuntimeError("Preferred Stage4.2C metadata/config preimage binding changed")
    metadata_sources = {
        str(row.get("name")): str(row.get("sha256"))
        for row in metadata.get("source_files", [])
        if isinstance(row, dict)
    }
    for name in ("types.py", "adapter.py", "formulation.py"):
        relative = next(path for path in EXPECTED_C_SOURCE_PREIMAGE if path.endswith(f"/{name}"))
        if metadata_sources.get(name) != EXPECTED_C_SOURCE_PREIMAGE[relative]:
            raise RuntimeError(f"Preferred Stage4.2C metadata/source preimage changed: {name}")
    before_pointers = metadata.get("parent_pointers_before")
    after_pointers = metadata.get("parent_pointers_after")
    for relative, digest in EXPECTED_C_PARENT_POINTERS.items():
        if (
            not isinstance(before_pointers, dict)
            or not isinstance(after_pointers, dict)
            or before_pointers.get(relative) != {"exists": True, "sha256": digest}
            or after_pointers.get(relative) != {"exists": True, "sha256": digest}
        ):
            raise RuntimeError(f"Historical parent/data pointer binding changed: {relative}")
    if (
        formulation.nx != 452
        or formulation.ny != 5242
        or formulation.visit_count != 20
    ):
        raise RuntimeError("Current problem is not the frozen Top3/452/5242/20 universe")
    historical, historical_floor_type = _historical_formulation(root)
    if (
        historical.nx != formulation.nx
        or historical.ny != formulation.ny
        or historical.visit_count != formulation.visit_count
    ):
        raise RuntimeError("Historical source preimage and current Top3 universe dimensions differ")

    adopted: list[AdoptedStage] = []
    for scenario in ("efficiency", "balanced"):
        relative = f"candidate_sets/top3/stages__{scenario}.json"
        rows = json.loads((source / relative).read_text(encoding="utf-8"))
        if not isinstance(rows, list) or len(rows) != 4:
            raise RuntimeError(f"Preferred Stage4.2C {scenario} stage trace changed")
        for expected_index, row in enumerate(rows, start=1):
            if not isinstance(row, dict):
                raise RuntimeError("Compatibility source stage must be a mapping")
            key = f"{scenario}:{expected_index}"
            expected = EXPECTED_ADOPTION_STAGES[key]
            identity = (
                row.get("scenario"),
                row.get("candidate_set"),
                row.get("stage_index"),
                row.get("objective_name"),
                row.get("sense"),
            )
            if identity != (
                scenario,
                "top3",
                expected_index,
                expected["objective"],
                expected["sense"],
            ):
                raise RuntimeError(f"Compatibility stage identity changed: {key}")
            details = row.get("details")
            if not isinstance(details, dict):
                raise RuntimeError(f"Compatibility stage details changed: {key}")
            floors = _floors(details.get("floors_before_stage"))
            model = formulation.build(
                objective_name=str(expected["objective"]),
                sense=str(expected["sense"]),  # type: ignore[arg-type]
                floors=floors,
                name=f"stage4_2f_adoption::{scenario}::{expected_index}",
            )
            historical_floors = [
                historical_floor_type(floor.objective, floor.sense, floor.value)
                for floor in floors
            ]
            historical_model = historical.build(
                objective_name=str(expected["objective"]),
                sense=str(expected["sense"]),
                floors=historical_floors,
                name=f"stage4_2f_historical_preimage::{scenario}::{expected_index}",
            )
            raw_selection = row.get("selected_indices")
            if (
                not isinstance(raw_selection, list)
                or any(type(value) is not int for value in raw_selection)
            ):
                raise RuntimeError(f"Compatibility selection changed: {key}")
            selected = np.asarray(raw_selection, dtype=np.int64)
            valid, reason = formulation.validate_selection(selected)
            if not valid:
                raise RuntimeError(f"Compatibility selection is invalid ({key}): {reason}")
            completed = formulation.complete_solution(model, selected)
            violations = formulation.feasibility_violations(model, completed, tol=1e-7)
            historical_completed = historical.complete_solution(historical_model, selected)
            historical_violations = historical.feasibility_violations(
                historical_model, historical_completed, tol=1e-7
            )
            max_violation = max(
                float(violations[name])
                for name in (
                    "max_row_lower",
                    "max_row_upper",
                    "max_col_lower",
                    "max_col_upper",
                    "max_integrality",
                )
            )
            incumbent = float(row.get("objective_value"))
            bound = float(row.get("best_bound"))
            source_gap = float(row.get("relative_gap"))
            current_objective = formulation.objective_for_selection(
                str(expected["objective"]), selected
            )
            historical_objective = historical.objective_for_selection(
                str(expected["objective"]), selected
            )
            direction, recomputed_gap = _bound_gap(str(expected["sense"]), incumbent, bound)
            observed = {
                "objective": expected["objective"],
                "sense": expected["sense"],
                "model_sha256": canonical_model_digest(model),
                "universe_sha256": _universe_digest(formulation, model),
                "selection_sha256": selection_digest(selected),
                "rows": model.n_row,
                "cols": model.n_col,
                "nnz": int(model.A.nnz),
                "incumbent": incumbent,
                "best_bound": bound,
                "relative_gap": source_gap,
            }
            historical_observed = {
                "model_sha256": canonical_model_digest(historical_model),
                "universe_sha256": _universe_digest(historical, historical_model),
                "selection_sha256": selection_digest(selected),
                "rows": historical_model.n_row,
                "cols": historical_model.n_col,
                "nnz": int(historical_model.A.nnz),
            }
            for field in (
                "objective",
                "sense",
                "model_sha256",
                "universe_sha256",
                "selection_sha256",
                "rows",
                "cols",
                "nnz",
            ):
                if observed[field] != expected[field]:
                    raise RuntimeError(f"Exact {key} {field} seal changed")
                if field in historical_observed and historical_observed[field] != expected[field]:
                    raise RuntimeError(f"Historical-preimage {key} {field} seal changed")
            for field in ("incumbent", "best_bound", "relative_gap"):
                if not _same(observed[field], float(expected[field])):
                    raise RuntimeError(f"Exact {key} {field} binary64 value changed")
            if not _same(current_objective, incumbent):
                raise RuntimeError(f"Current CPU objective recomputation changed: {key}")
            if not _same(historical_objective, incumbent):
                raise RuntimeError(f"Historical-preimage CPU objective changed: {key}")
            if not direction or not math.isfinite(recomputed_gap):
                raise RuntimeError(f"Incumbent/bound direction is invalid: {key}")
            if abs(recomputed_gap - source_gap) > 1e-12 or recomputed_gap > FROZEN_GAP:
                raise RuntimeError(f"Current bound-gap recomputation is not certified: {key}")
            if violations.get("feasible") != 1.0 or max_violation > 1e-7:
                raise RuntimeError(f"Current model feasibility recomputation failed: {key}")
            if historical_violations.get("feasible") != 1.0:
                raise RuntimeError(f"Historical-preimage model feasibility failed: {key}")
            historical_current_equal = bool(
                historical_observed["model_sha256"] == observed["model_sha256"]
                and historical_observed["universe_sha256"] == observed["universe_sha256"]
                and historical_observed["selection_sha256"] == observed["selection_sha256"]
            )
            if not historical_current_equal:
                raise RuntimeError(f"Historical/current compatibility proof failed: {key}")

            adopted.append(
                AdoptedStage(
                    scenario=scenario,
                    stage_index=expected_index,
                    objective=str(expected["objective"]),
                    sense=str(expected["sense"]),
                    semantic_label="CERTIFIED_NEAR_OPTIMAL",
                    certificate_basis=(
                        "PINNED_BOUND_PLUS_HISTORICAL_PREIMAGE_EQUALS_CURRENT_MODEL_UNIVERSE_SELECTION_RECOMPUTATION"
                    ),
                    source_run_id=FROZEN_C_RUN_ID,
                    source_relative_path=(source / relative).relative_to(root).as_posix(),
                    source_artifact_sha256=EXPECTED_C_ARTIFACTS[relative],
                    incumbent=incumbent,
                    best_bound=bound,
                    relative_gap=recomputed_gap,
                    current_recomputed_objective=current_objective,
                    current_selection_feasible=True,
                    feasibility_tolerance=1e-7,
                    max_constraint_violation=max_violation,
                    historical_preimage_recomputed_objective=historical_objective,
                    historical_preimage_selection_feasible=True,
                    historical_preimage_model_sha256=str(
                        historical_observed["model_sha256"]
                    ),
                    historical_preimage_universe_sha256=str(
                        historical_observed["universe_sha256"]
                    ),
                    historical_preimage_selection_sha256=str(
                        historical_observed["selection_sha256"]
                    ),
                    historical_current_model_equal=historical_current_equal,
                    model_sha256=str(observed["model_sha256"]),
                    universe_sha256=str(observed["universe_sha256"]),
                    selection_sha256=str(observed["selection_sha256"]),
                    model_rows=int(model.n_row),
                    model_cols=int(model.n_col),
                    model_nnz=int(model.A.nnz),
                    floors=[
                        {
                            "objective": floor.objective,
                            "sense": floor.sense,
                            "value": floor.value,
                        }
                        for floor in floors
                    ],
                    legacy_solver_claims_used=False,
                    legacy_run_pass_used=False,
                )
            )

        plan_relative = f"candidate_sets/top3/plan__{scenario}.csv"
        final_plan = _selection_from_plan(source / plan_relative, formulation)
        if selection_digest(final_plan) != adopted[-1].selection_sha256:
            raise RuntimeError(f"Pinned {scenario} final plan and stage-4 selection differ")

    classifications = {
        scenario: "CERTIFIED_NEAR_OPTIMAL" for scenario in ("efficiency", "balanced")
    }
    passed = bool(
        len(adopted) == 8
        and all(stage.current_selection_feasible for stage in adopted)
        and all(stage.relative_gap <= FROZEN_GAP for stage in adopted)
        and all(stage.semantic_label == "CERTIFIED_NEAR_OPTIMAL" for stage in adopted)
        and all(not stage.legacy_solver_claims_used for stage in adopted)
        and all(not stage.legacy_run_pass_used for stage in adopted)
        and all(stage.historical_preimage_selection_feasible for stage in adopted)
        and all(stage.historical_current_model_equal for stage in adopted)
    )
    return CompatibilityAudit(
        passed=passed,
        source_run_id=FROZEN_C_RUN_ID,
        source_inventory_sha256=FROZEN_C_INVENTORY_SHA256,
        candidate_count=formulation.nx,
        pattern_count=formulation.ny,
        visit_count=formulation.visit_count,
        stage_count=len(adopted),
        stages=adopted,
        scenario_classification=classifications,
        historical_source_preimage_verified=True,
        evidence_records=records,
    )
