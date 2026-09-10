from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from .errors import ContractError
from .types import Stage3Artifacts, Stage41Artifacts
from .utils import read_json, sha256_file


def find_project_root(start: Path | None = None) -> Path:
    start = (start or Path.cwd()).resolve()
    candidates = [start, *start.parents]
    for root in candidates:
        if (root / "outputs/model_v1").exists() and (root / "configs/model_v1").exists():
            return root
    raise ContractError(
        f"Could not find MEDIROAD project root from {start}. Expected outputs/model_v1 and configs/model_v1."
    )


def _resolve(root: Path, value: str | Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else root / p


def _find_one(root: Path, patterns: Iterable[str], *, label: str) -> Path:
    found: list[Path] = []
    for pattern in patterns:
        found.extend(root.rglob(pattern))
    found = sorted({p.resolve() for p in found if p.is_file()})
    if not found:
        raise FileNotFoundError(f"Could not locate {label} under {root}; patterns={list(patterns)}")
    if len(found) > 1:
        # Prefer paths inside the currently pointed run and then shortest paths.
        found.sort(key=lambda p: (len(p.parts), p.as_posix()))
    return found[0]


def _verify_pointer_reference(root: Path, pointer: dict, key: str, sha_key: str | None = None) -> Path:
    if key not in pointer:
        raise ContractError(f"Pointer missing {key}")
    path = _resolve(root, pointer[key])
    if not path.exists():
        raise FileNotFoundError(path)
    if sha_key and pointer.get(sha_key):
        actual = sha256_file(path)
        expected = str(pointer[sha_key]).lower()
        if actual.lower() != expected:
            raise ContractError(f"Pointer SHA mismatch for {path}: expected={expected}, actual={actual}")
    return path


def discover_stage3(root: Path) -> Stage3Artifacts:
    pointer_path = root / "outputs/model_v1/07_stage3/CURRENT_STAGE3_RUN.json"
    pointer = read_json(pointer_path)
    if not bool(pointer.get("stage3_complete")):
        raise ContractError("Stage 3 CURRENT pointer is not complete")
    metadata = _verify_pointer_reference(root, pointer, "metadata_relative_path", "metadata_sha256")
    interface = _verify_pointer_reference(root, pointer, "interface_relative_path", "interface_sha256")
    run_root = metadata.parent
    manifest = _find_one(
        run_root,
        ["venue_grid_matrix_manifest.csv", "*matrix_manifest*.csv", "*matrix*manifest*.csv"],
        label="Stage 3 venue-grid matrix manifest",
    )
    grid_policy = _find_one(
        run_root,
        [
            "grid_policy_weights.parquet",
            "grid_policy_weights.csv",
            "*grid*policy*.parquet",
            "*grid*policy*.csv",
            "*grid*weights*.parquet",
            "*grid*weights*.csv",
        ],
        label="Stage 3 grid-policy table",
    )
    return Stage3Artifacts(
        pointer=pointer_path,
        run_root=run_root,
        metadata=metadata,
        interface=interface,
        matrix_manifest=manifest,
        grid_policy=grid_policy,
    )


def discover_stage41(root: Path) -> Stage41Artifacts:
    pointer_path = root / "outputs/model_v1/09_stage4_finalization/CURRENT_STAGE4_FINALIZATION_RUN.json"
    pointer = read_json(pointer_path)
    if not bool(pointer.get("stage4_ready_for_field_validation")):
        raise ContractError("Stage 4.1 CURRENT pointer is not ready for field validation")
    metadata = _verify_pointer_reference(root, pointer, "metadata_relative_path", "metadata_sha256")
    inventory = _verify_pointer_reference(root, pointer, "inventory_relative_path", "inventory_sha256")
    report = _verify_pointer_reference(root, pointer, "report_relative_path", "report_sha256")
    run_root = metadata.parent

    def one(rel: str, patterns: list[str], label: str) -> Path:
        direct = run_root / rel
        return direct if direct.exists() else _find_one(run_root, patterns, label=label)

    field_form = one(
        "06_field_readiness/field_validation_form.csv",
        ["field_validation_form.csv"],
        "field-validation form",
    )
    field_queue = one(
        "06_field_readiness/field_validation_priority_queue.csv",
        ["field_validation_priority_queue.csv"],
        "field-validation queue",
    )
    field_interface = one(
        "07_interface/stage4_1_to_field_validation.parquet",
        ["stage4_1_to_field_validation.parquet", "stage4_1_to_field_validation.csv"],
        "Stage 4.1 field interface",
    )
    physical_resolution = one(
        "06_field_readiness/physical_venue_resolution.csv",
        ["physical_venue_resolution.csv"],
        "physical venue resolution",
    )
    candidate_membership = one(
        "02_candidate_sensitivity/candidate_set_membership.parquet",
        ["candidate_set_membership.parquet", "candidate_set_membership.csv"],
        "candidate-set membership",
    )
    plan_efficiency = one(
        "05_policy_and_frontier/final_plan__efficiency.csv",
        ["final_plan__efficiency.csv"],
        "efficiency plan",
    )
    plan_balanced = one(
        "05_policy_and_frontier/final_plan__balanced.csv",
        ["final_plan__balanced.csv"],
        "balanced plan",
    )
    plan_equity = one(
        "05_policy_and_frontier/final_plan__equity.csv",
        ["final_plan__equity.csv"],
        "equity plan",
    )
    policy_metrics = one(
        "05_policy_and_frontier/final_policy_scenario_metrics.csv",
        ["final_policy_scenario_metrics.csv"],
        "policy metrics",
    )
    solver_certification = one(
        "01_solver_certification/final_n20_prefrontier_run_certification.csv",
        ["final_n20_prefrontier_run_certification.csv", "solver_run_certification.csv"],
        "solver certification",
    )
    return Stage41Artifacts(
        pointer=pointer_path,
        run_root=run_root,
        metadata=metadata,
        inventory=inventory,
        report=report,
        field_form=field_form,
        field_queue=field_queue,
        field_interface=field_interface,
        physical_resolution=physical_resolution,
        candidate_membership=candidate_membership,
        plan_efficiency=plan_efficiency,
        plan_balanced=plan_balanced,
        plan_equity=plan_equity,
        policy_metrics=policy_metrics,
        solver_certification=solver_certification,
    )


def verify_inventory(run_root: Path, inventory_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(inventory_path, low_memory=False)
    path_col = next((c for c in ["relative_path", "artifact_relative_path", "path"] if c in frame), None)
    size_col = next((c for c in ["size_bytes", "artifact_size_bytes", "size"] if c in frame), None)
    sha_col = next((c for c in ["sha256", "artifact_sha256"] if c in frame), None)
    if not path_col or not size_col or not sha_col:
        raise ContractError(f"Unrecognized inventory schema: {list(frame.columns)}")
    problems: list[str] = []
    for _, row in frame.iterrows():
        rel = str(row[path_col])
        path = run_root / rel
        if not path.exists():
            # Some inventories are rooted one directory above their metadata.
            candidate = run_root.parent / rel
            if candidate.exists():
                path = candidate
            else:
                problems.append(f"missing:{rel}")
                continue
        expected_size = int(row[size_col])
        if path.stat().st_size != expected_size:
            problems.append(f"size:{rel}")
            continue
        expected_sha = str(row[sha_col]).lower()
        if sha256_file(path).lower() != expected_sha:
            problems.append(f"sha:{rel}")
    if problems:
        raise ContractError(f"Stage 4.1 inventory verification failed ({len(problems)}): {problems[:10]}")
    return frame
