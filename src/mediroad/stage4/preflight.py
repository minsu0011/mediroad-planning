from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .discovery import discover_stage3_paths
from .types import QualityCheck, Stage3Paths
from .utils import coalesce_column, read_table, sha256_file


FROZEN_CONTRACTS = {
    "derived/mediroad_admin_dong_master_v6_final.csv": {
        "sha256": "3bcff13063c741edf1492df22b774bd6d37424d729c15423e31ba8ea576adf7a",
        "rows": 153,
    },
    "derived/mediroad_admin_dong_master_v6_road_integrated.csv": {
        "sha256": "faed8c4d024dfff89602eebf3229c1eaeea955ca2d14fb88f0e2cbc4f76e078a",
        "rows": 153,
    },
    "configs/model_v1/stage1_features.yaml": {
        "sha256": "3fe95a55e47b80ff1b0cb8c4819f21ed0dc855dcd696486e10e29025294def8a",
        "rows": None,
    },
}


def _check_contract(path: Path, contract: dict[str, Any]) -> list[QualityCheck]:
    checks: list[QualityCheck] = []
    checks.append(
        QualityCheck(
            check_id=f"exists::{path.as_posix()}",
            passed=path.exists(),
            severity="HARD",
            observed=path.exists(),
            expected=True,
            message="Frozen input exists",
        )
    )
    if not path.exists():
        return checks
    observed_sha = sha256_file(path)
    checks.append(
        QualityCheck(
            check_id=f"sha::{path.as_posix()}",
            passed=observed_sha == contract["sha256"],
            severity="HARD",
            observed=observed_sha,
            expected=contract["sha256"],
            message="Frozen SHA must match the Stage 1-3 contract",
        )
    )
    if contract.get("rows") is not None:
        try:
            rows = len(pd.read_csv(path, low_memory=False))
        except Exception as exc:
            checks.append(
                QualityCheck(
                    check_id=f"rows::{path.as_posix()}",
                    passed=False,
                    severity="HARD",
                    observed=str(exc),
                    expected=contract["rows"],
                    message="Could not count rows in frozen table",
                )
            )
        else:
            checks.append(
                QualityCheck(
                    check_id=f"rows::{path.as_posix()}",
                    passed=rows == contract["rows"],
                    severity="HARD",
                    observed=rows,
                    expected=contract["rows"],
                    message="Frozen row count",
                )
            )
    return checks



def _validate_stage3_inventory(stage3: Stage3Paths) -> tuple[list[QualityCheck], dict[str, Any]]:
    checks: list[QualityCheck] = []
    summary: dict[str, Any] = {"inventory_present": False}
    if stage3.inventory is None or not stage3.inventory.exists():
        checks.append(
            QualityCheck(
                "stage3_artifact_inventory_present",
                False,
                "HARD",
                None,
                "STAGE3_ARTIFACT_INVENTORY.csv",
                "Frozen Stage 3 must expose its artifact inventory",
            )
        )
        return checks, summary
    inventory = pd.read_csv(stage3.inventory, low_memory=False)
    path_col = next((c for c in ["relative_path", "path", "artifact_path", "file"] if c in inventory.columns), None)
    sha_col = next((c for c in ["sha256", "file_sha256", "artifact_sha256"] if c in inventory.columns), None)
    size_col = next((c for c in ["size_bytes", "bytes", "file_size"] if c in inventory.columns), None)
    if path_col is None or sha_col is None:
        checks.append(
            QualityCheck(
                "stage3_artifact_inventory_schema",
                False,
                "HARD",
                list(inventory.columns),
                "path + sha256 columns",
                "Stage 3 inventory schema must permit deterministic verification",
            )
        )
        return checks, summary
    failures: list[dict[str, Any]] = []
    checked = 0
    for _, values in inventory.iterrows():
        rel = Path(str(values[path_col]))
        candidates = [stage3.run_root / rel, stage3.inventory.parent / rel]
        if stage3.package_root is not None:
            candidates.insert(0, stage3.package_root / rel)
        file_path = next((candidate for candidate in candidates if candidate.exists()), None)
        if file_path is None:
            failures.append({"path": str(rel), "reason": "missing"})
            continue
        actual_sha = sha256_file(file_path)
        expected_sha = str(values[sha_col]).strip().lower()
        if actual_sha.lower() != expected_sha:
            failures.append({"path": str(rel), "reason": "sha", "expected": expected_sha, "actual": actual_sha})
            continue
        if size_col is not None and pd.notna(values[size_col]):
            expected_size = int(values[size_col])
            if file_path.stat().st_size != expected_size:
                failures.append({
                    "path": str(rel),
                    "reason": "size",
                    "expected": expected_size,
                    "actual": file_path.stat().st_size,
                })
                continue
        checked += 1
    summary = {
        "inventory_present": True,
        "inventory_path": str(stage3.inventory),
        "inventory_rows": int(len(inventory)),
        "inventory_verified": int(checked),
        "inventory_failures": failures[:20],
    }
    checks.append(
        QualityCheck(
            "stage3_artifact_inventory_verified",
            len(failures) == 0 and checked == len(inventory),
            "HARD",
            {"verified": checked, "failed": len(failures)},
            {"verified": len(inventory), "failed": 0},
            "Every Stage 3 artifact listed in the frozen inventory must match size/SHA",
        )
    )
    return checks, summary

def validate_stage3_artifacts(stage3: Stage3Paths, config: dict[str, Any]) -> tuple[list[QualityCheck], dict[str, Any]]:
    aliases = config["column_aliases"]
    checks: list[QualityCheck] = []
    interface = read_table(stage3.interface)
    grid = read_table(stage3.grid_policy)

    venue_col = coalesce_column(interface, aliases["venue_id"])
    bundle_col = coalesce_column(interface, aliases["bundle_id"])
    grid_col = coalesce_column(grid, aliases["grid_id"])
    pop_col = coalesce_column(grid, aliases["elderly_population"])
    need_col = coalesce_column(grid, aliases["need_score"])

    n_venue = interface[venue_col].astype(str).nunique()
    n_bundle = interface[bundle_col].astype(str).nunique()
    n_grid = grid[grid_col].astype(str).nunique()
    pop = pd.to_numeric(grid[pop_col], errors="coerce")
    need = pd.to_numeric(grid[need_col], errors="coerce")

    checks.extend(
        [
            QualityCheck("stage3_interface_nonempty", len(interface) > 0, "HARD", len(interface), ">0", "Stage 3 interface rows"),
            QualityCheck("stage3_venue_count", n_venue >= 100, "HARD", n_venue, ">=100", "Venue coverage interface"),
            QualityCheck("stage3_bundle_count", n_bundle == 5, "HARD", n_bundle, 5, "Frozen Stage 2A bundle universe"),
            QualityCheck("stage3_grid_count", n_grid >= 50000, "HARD", n_grid, ">=50000", "100m grid universe"),
            QualityCheck("stage3_grid_unique", not grid[grid_col].astype(str).duplicated().any(), "HARD", int(grid[grid_col].astype(str).duplicated().sum()), 0, "Grid IDs unique"),
            QualityCheck("stage3_population_nonnegative", bool((pop.fillna(-1) >= 0).all()), "HARD", float(pop.min()), ">=0", "Calibrated population non-negative"),
            QualityCheck("stage3_population_finite", bool(np.isfinite(pop).all()), "HARD", int((~np.isfinite(pop)).sum()), 0, "Calibrated population finite"),
            QualityCheck("stage3_need_finite", bool(np.isfinite(need).all()), "HARD", int((~np.isfinite(need)).sum()), 0, "Need values finite"),
        ]
    )

    stage3_advisory_failures: list[dict[str, Any]] = []
    if stage3.quality_gate and stage3.quality_gate.exists():
        qgate = pd.read_csv(stage3.quality_gate, low_memory=False)
        pass_col = next((c for c in ["passed", "pass", "status"] if c in qgate.columns), None)
        severity_col = next((c for c in ["severity", "gate_type", "type"] if c in qgate.columns), None)
        hard_flag_col = next((c for c in ["is_hard", "hard", "required"] if c in qgate.columns), None)
        if pass_col is not None:
            if pass_col == "status":
                passed = qgate[pass_col].astype(str).str.upper().isin(["PASS", "PASSED", "TRUE", "1"])
            else:
                passed = qgate[pass_col].astype(str).str.lower().isin(["true", "1", "pass", "passed"])
            if severity_col is not None:
                severity = qgate[severity_col].astype(str).str.upper()
                hard_mask = severity.isin(["HARD", "REQUIRED", "CRITICAL"])
                advisory_mask = ~hard_mask
            elif hard_flag_col is not None:
                hard_mask = qgate[hard_flag_col].astype(str).str.lower().isin(["true", "1", "yes", "hard", "required"])
                advisory_mask = ~hard_mask
            else:
                # If Stage 3 did not label severity, preserve the historical contract: all listed gates are hard.
                hard_mask = pd.Series(True, index=qgate.index)
                advisory_mask = pd.Series(False, index=qgate.index)
            hard_passed = passed[hard_mask]
            checks.append(
                QualityCheck(
                    "stage3_quality_gate_all_hard_pass",
                    bool(hard_passed.all()),
                    "HARD",
                    int(hard_passed.sum()),
                    len(hard_passed),
                    "Stage 3 hard quality gates must remain passed; advisory failures are preserved separately",
                )
            )
            advisory_failed = qgate[advisory_mask & ~passed].copy()
            stage3_advisory_failures = advisory_failed.to_dict(orient="records")
            checks.append(
                QualityCheck(
                    "stage3_advisory_failures_preserved",
                    len(advisory_failed) == 0,
                    "ADVISORY",
                    len(advisory_failed),
                    0,
                    "Stage 3 advisory failures are not promoted to hard failures or hidden",
                )
            )

    metadata: dict[str, Any] = {
        "stage3_run_root": str(stage3.run_root),
        "stage3_interface": str(stage3.interface),
        "stage3_grid_policy": str(stage3.grid_policy),
        "stage3_matrix_manifest": str(stage3.matrix_manifest),
        "n_interface_rows": int(len(interface)),
        "n_venues": int(n_venue),
        "n_bundles": int(n_bundle),
        "n_grids": int(n_grid),
        "total_elderly65": float(pop.sum()),
        "stage3_advisory_failures": stage3_advisory_failures,
    }
    if stage3.metadata and stage3.metadata.exists():
        try:
            metadata["stage3_metadata"] = json.loads(stage3.metadata.read_text(encoding="utf-8"))
        except Exception:
            metadata["stage3_metadata"] = {"unparsed_path": str(stage3.metadata)}
    return checks, metadata


def run_preflight(package_root: Path, config: dict[str, Any]) -> tuple[Stage3Paths, list[QualityCheck], dict[str, Any]]:
    checks: list[QualityCheck] = []
    for relative, contract in FROZEN_CONTRACTS.items():
        checks.extend(_check_contract(package_root / relative, contract))
    stage3 = discover_stage3_paths(package_root, config)
    stage3_checks, metadata = validate_stage3_artifacts(stage3, config)
    checks.extend(stage3_checks)
    inventory_checks, inventory_summary = _validate_stage3_inventory(stage3)
    checks.extend(inventory_checks)
    metadata["stage3_inventory_verification"] = inventory_summary
    hard_failures = [c for c in checks if c.severity == "HARD" and not c.passed]
    if hard_failures:
        details = "\n".join(f"- {c.check_id}: observed={c.observed}, expected={c.expected}" for c in hard_failures)
        raise RuntimeError(f"Stage 4 preflight failed:\n{details}")
    return stage3, checks, metadata
