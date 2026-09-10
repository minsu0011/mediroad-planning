from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import traceback
from fnmatch import fnmatch
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd

from mediroad.stage4_2_certification.adapter import (
    constrained_greedy,
    load_base_stage42_config,
    prepare_candidate_problem,
)
from mediroad.stage4_2_certification.config import validate_certification_config
from mediroad.stage4_2_certification.formulation import CertificationFormulation
from mediroad.stage4_2_certification.types import Floor

from .config import (
    apply_process_priority,
    load_yaml,
    resolve_hardware,
    set_thread_environment,
    validate_config,
    validate_parent_contracts,
)
from .evidence import reconstruct_contract, retained_min_sigungu_floor
from .exact import EquityFrontStageCertifier
from .gpu_lns import HybridLargeNeighborhoodSearch
from .io_utils import atomic_write_csv, atomic_write_json, make_run_id, sha256_file, tree_inventory, utc_now
from .strengthening import (
    compute_safe_zero_fixings,
    generate_anchored_submodular_cuts,
    metric_targets,
    serialize_fixings,
)
from .threshold_oracles import (
    ThresholdOracleModel,
    ThresholdOracleResult,
    build_high_scip_oracle,
    build_min_highs_oracle,
    certificate_from_threshold_oracle,
    run_min_highs_oracle,
    run_scip_portfolio,
    selection_digest,
    write_mps,
)
from .types import EvidenceSeed, MetricCertificate
from .version import PACKAGE_NAME, VERSION


def _pointer_paths(root: Path) -> list[Path]:
    return [
        root / "outputs/model_v1/07_stage3/CURRENT_STAGE3_RUN.json",
        root / "outputs/model_v1/08_stage4/CURRENT_STAGE4_RUN.json",
        root / "outputs/model_v1/09_stage4_finalization/CURRENT_STAGE4_FINALIZATION_RUN.json",
        root / "outputs/model_v1/10_stage4_2/CURRENT_STAGE4_2_PREPARE_RUN.json",
        root / "outputs/model_v1/10_stage4_2/CURRENT_STAGE4_2_RUN.json",
        root / "outputs/model_v1/10_stage4_2_certification/CURRENT_STAGE4_2_CERTIFICATION_RUN.json",
        root / "outputs/model_v1/10_stage4_2_certification_diagnostic/CURRENT_STAGE4_2_CERTIFICATION_RUN.json",
    ]


def _pointer_snapshot(root: Path) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for path in _pointer_paths(root):
        rel = path.relative_to(root).as_posix()
        output[rel] = {
            "exists": path.exists(),
            "sha256": sha256_file(path) if path.exists() else None,
        }
    return output


def _stage5_snapshot(root: Path) -> dict[str, Any]:
    # Stage 4.2D must not create or mutate a Stage 5 namespace.  Searching only
    # the immediate children of outputs/reports misses a nested Stage 5 root,
    # while a blind ``*stage5*`` search also catches the frozen Stage 4 handoff
    # and our own audit evidence.  Keep that historical evidence on an exact
    # allow-list and inventory every other matching namespace recursively.
    allowed_evidence_paths = (
        "outputs/model_v1/08_stage4/runs/*/05_interface/stage4_to_stage5_interface.csv",
        "outputs/model_v1/08_stage4/runs/*/05_interface/stage4_to_stage5_interface.parquet",
        "outputs/model_v1/09_stage4_finalization/runs/*/00_provenance/forbidden_stage5_namespace_before.csv",
        "outputs/model_v1/09_stage4_finalization/runs/*/00_provenance/forbidden_stage5_namespace_after.csv",
        "outputs/model_v1/09_stage4_finalization/runs/*/07_interface/stage5_readiness_checklist.json",
        "outputs/model_v1/10_stage4_2d_equity_certification/runs/*/stage5_namespace_before_after.json",
        "outputs/model_v1/10_stage4_2d_equity_certification_diagnostic/runs/*/stage5_namespace_before_after.json",
    )

    def is_allowed_evidence(path: Path) -> bool:
        relative = path.relative_to(root).as_posix().lower()
        return any(fnmatch(relative, pattern) for pattern in allowed_evidence_paths)

    matched: set[Path] = set()
    for base in (root / "outputs/model_v1", root / "reports/model_v1"):
        if not base.exists():
            continue
        namespace_roots: set[Path] = set()
        for path in base.rglob("*"):
            if "stage5" not in path.name.lower() or is_allowed_evidence(path):
                continue
            namespace_roots.add(path)
        for namespace_root in namespace_roots:
            matched.add(namespace_root)
            if namespace_root.is_dir():
                matched.update(namespace_root.rglob("*"))

    records: list[dict[str, Any]] = []
    for path in sorted(matched):
        records.append(
            {
                "relative_path": path.relative_to(root).as_posix(),
                "kind": "directory" if path.is_dir() else "file",
                "size_bytes": int(path.stat().st_size) if path.is_file() else None,
                "sha256": sha256_file(path) if path.is_file() else None,
            }
        )
    return {"records": records, "count": len(records)}


def _source_snapshot(root: Path, config_paths: list[Path]) -> list[dict[str, Any]]:
    paths: set[Path] = set()
    for package in (
        root / "src/mediroad/stage4_2",
        root / "src/mediroad/stage4_2_certification",
        root / "src/mediroad/stage4_2d_equity_certification",
    ):
        if package.exists():
            paths.update(package.rglob("*.py"))
    for relative in (
        "src/mediroad/__init__.py",
        "12_scripts/v6/run_model_v1_stage4_2d_equity_certification.py",
        "run_model_v1_stage4_2d_equity_certification.ps1",
        "run_model_v1_stage4_2d_equity_certification_diagnostic.ps1",
        "run_stage4_2d_smoke.ps1",
        "scripts/run_stage4_2d_smoke.py",
        "scripts/audit_stage4_2d_contract.py",
        "scripts/probe_stage4_2d_repair.py",
        "requirements_stage4_2d_equity.txt",
        "requirements_stage4_2d_gpu_optional.txt",
        "environment_stage4_2d_equity.yml",
    ):
        path = root / relative
        if path.is_file():
            paths.add(path)
    paths.update(path for path in config_paths if path.is_file())
    tests_root = root / "tests"
    if tests_root.exists():
        # The regression command is project-wide; sealing only the Stage 4.2D
        # test names permits Stage 4.2/4.2C (or another imported regression)
        # to change between preflight and commit.
        paths.update(tests_root.glob("test_*.py"))
    for pattern in (
        "requirements_stage4_2*.txt",
        "environment_stage4_2*.yml",
    ):
        paths.update(path for path in root.glob(pattern) if path.is_file())
    return [
        {
            "relative_path": path.resolve().relative_to(root).as_posix(),
            "size_bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
        }
        for path in sorted(paths)
    ]


def _atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(value)
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _run_regression_tests(root: Path, output_dir: Path) -> dict[str, Any]:
    """Execute and seal the project-wide regression suite before optimization."""

    command = [sys.executable, "-m", "pytest", "-q", "tests", "-p", "no:cacheprovider"]
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONPATH"] = str(root / "src")
    completed = subprocess.run(
        command,
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=1200,
        check=False,
    )
    stdout = completed.stdout if completed.stdout else "NO_STDOUT_OUTPUT\n"
    stderr = completed.stderr if completed.stderr else "NO_STDERR_OUTPUT\n"
    _atomic_write_text(output_dir / "regression_test_stdout.txt", stdout)
    _atomic_write_text(output_dir / "regression_test_stderr.txt", stderr)
    match = re.search(r"(?P<passed>\d+) passed", completed.stdout or "")
    failures = re.search(r"(?P<failed>\d+) failed", completed.stdout or "")
    result = {
        "command": command,
        "return_code": int(completed.returncode),
        "passed_count": int(match.group("passed")) if match else None,
        "failed_count": int(failures.group("failed")) if failures else 0,
        "stdout_sha256": sha256_file(output_dir / "regression_test_stdout.txt"),
        "stderr_sha256": sha256_file(output_dir / "regression_test_stderr.txt"),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "cacheprovider_disabled": True,
        "bytecode_disabled": True,
    }
    atomic_write_json(output_dir / "regression_test_execution.json", result)
    if completed.returncode != 0 or result["passed_count"] is None or result["failed_count"] != 0:
        raise RuntimeError(f"Stage4.2D regression suite failed: {result}")
    return result


def _inventory_path_from_pointer(root: Path, pointer_path: Path, payload: dict[str, Any]) -> Path:
    raw = payload.get("inventory_relative_path")
    if raw:
        path = root / str(raw)
    elif pointer_path.name == "CURRENT_STAGE3_RUN.json" and payload.get("metadata_relative_path"):
        path = (root / str(payload["metadata_relative_path"])).parent / "STAGE3_ARTIFACT_INVENTORY.csv"
    else:
        raise RuntimeError(f"Authoritative pointer has no inventory path: {pointer_path}")
    resolved = path.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise RuntimeError(f"Inventory path escapes project root: {resolved}") from exc
    return resolved


def _verify_inventory(root: Path, inventory_path: Path) -> dict[str, Any]:
    if not inventory_path.is_file():
        raise RuntimeError(f"Authoritative inventory is missing: {inventory_path}")
    frame = pd.read_csv(inventory_path, dtype={"relative_path": str, "sha256": str})
    required = {"relative_path", "size_bytes", "sha256"}
    if frame.empty or not required.issubset(frame.columns) or frame["relative_path"].duplicated().any():
        raise RuntimeError(f"Invalid authoritative inventory: {inventory_path}")
    for row in frame.to_dict("records"):
        relative = Path(str(row["relative_path"]))
        candidates = (root / relative, inventory_path.parent / relative)
        existing = [path for path in candidates if path.is_file()]
        if not existing:
            raise RuntimeError(f"Authoritative artifact is missing: {relative}")
        if not any(
            int(path.stat().st_size) == int(row["size_bytes"])
            and sha256_file(path) == str(row["sha256"]).lower()
            for path in existing
        ):
            raise RuntimeError(f"Authoritative artifact hash/size mismatch: {relative}")
    return {
        "relative_path": inventory_path.relative_to(root.resolve()).as_posix(),
        "sha256": sha256_file(inventory_path),
        "row_count": int(len(frame)),
    }


def _authoritative_input_snapshot(root: Path) -> list[dict[str, Any]]:
    """Verify every declared parent inventory, not just its CURRENT pointer."""

    records: list[dict[str, Any]] = []
    for pointer_path in _pointer_paths(root):
        if not pointer_path.is_file():
            continue
        payload = json.loads(pointer_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise RuntimeError(f"Invalid CURRENT pointer payload: {pointer_path}")
        inventory_path = _inventory_path_from_pointer(root, pointer_path, payload)
        inventory_record = _verify_inventory(root, inventory_path)
        expected_inventory_hash = payload.get("inventory_sha256") or payload.get("artifact_inventory_sha256")
        if expected_inventory_hash and str(expected_inventory_hash).lower() != inventory_record["sha256"]:
            raise RuntimeError(f"CURRENT pointer inventory SHA mismatch: {pointer_path}")
        for path_key, sha_key in (
            ("metadata_relative_path", "metadata_sha256"),
            ("interface_relative_path", "interface_sha256"),
            ("report_relative_path", "report_sha256"),
        ):
            if payload.get(path_key) and payload.get(sha_key):
                artifact = (root / str(payload[path_key])).resolve()
                try:
                    artifact.relative_to(root.resolve())
                except ValueError as exc:
                    raise RuntimeError(f"Pointer artifact escapes project root: {artifact}") from exc
                if not artifact.is_file() or sha256_file(artifact) != str(payload[sha_key]).lower():
                    raise RuntimeError(f"CURRENT pointer artifact SHA mismatch: {artifact}")
        records.append(
            {
                "pointer_relative_path": pointer_path.relative_to(root).as_posix(),
                "pointer_sha256": sha256_file(pointer_path),
                "run_id": payload.get("run_id"),
                **inventory_record,
            }
        )
    required_names = {
        "CURRENT_STAGE3_RUN.json",
        "CURRENT_STAGE4_RUN.json",
        "CURRENT_STAGE4_FINALIZATION_RUN.json",
        "CURRENT_STAGE4_2_PREPARE_RUN.json",
    }
    observed = {Path(record["pointer_relative_path"]).name for record in records}
    missing = sorted(required_names - observed)
    if missing:
        raise RuntimeError(f"Required authoritative parent pointers are missing: {missing}")
    return records


def _verify_output_inventory(root: Path, inventory_path: Path) -> None:
    frame = pd.read_csv(inventory_path, dtype={"relative_path": str, "sha256": str})
    if frame.empty or frame["relative_path"].duplicated().any():
        raise RuntimeError("Stage4.2D output inventory is empty or duplicated")
    if (pd.to_numeric(frame["size_bytes"], errors="coerce") <= 0).any():
        raise RuntimeError("Stage4.2D output inventory contains an empty artifact")
    for row in frame.to_dict("records"):
        path = root / str(row["relative_path"])
        if not path.is_file():
            raise RuntimeError(f"Inventoried Stage4.2D artifact is missing: {path}")
        if int(path.stat().st_size) != int(row["size_bytes"]) or sha256_file(path) != str(row["sha256"]).lower():
            raise RuntimeError(f"Inventoried Stage4.2D artifact mismatch: {path}")


def _verify_evidence_records(root: Path, records: list[dict[str, Any]]) -> bool:
    for record in records:
        path = (root / str(record["relative_path"])).resolve()
        try:
            path.relative_to(root.resolve())
        except ValueError:
            return False
        if not path.is_file():
            return False
        if int(path.stat().st_size) != int(record["size_bytes"]):
            return False
        if sha256_file(path) != str(record["sha256"]).lower():
            return False
    return True


@contextmanager
def _exclusive_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd: int | None = None
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, f"pid={os.getpid()} utc={utc_now()}\n".encode())
        yield
    except FileExistsError as exc:
        raise RuntimeError(f"Stage4.2D writer lock already exists: {path}") from exc
    finally:
        if fd is not None:
            os.close(fd)
            try:
                path.unlink()
            except FileNotFoundError:
                pass


def _exact_seed(formulation: CertificationFormulation, source: str, selected: np.ndarray) -> EvidenceSeed:
    raw = formulation.metrics(np.asarray(selected, dtype=int))
    return EvidenceSeed(
        source=source,
        selected_indices=np.asarray(selected, dtype=int),
        metrics={
            "total_population": float(raw["unique_elderly_population"]),
            "min_sigungu_coverage": float(raw["min_sigungu_coverage_ratio"]),
            "high_need_population": float(raw["high_need_population"]),
            "need_weighted": float(raw["need_weighted_population"]),
            "cost": float(raw["cost"]),
        },
    )


def _dedup_seeds(seeds: list[EvidenceSeed], objective: str) -> list[EvidenceSeed]:
    values: dict[tuple[int, ...], EvidenceSeed] = {}
    for seed in seeds:
        key = tuple(sorted(map(int, seed.selected_indices)))
        old = values.get(key)
        if old is None or seed.metrics.get(objective, float("-inf")) > old.metrics.get(objective, float("-inf")):
            values[key] = seed
    return sorted(values.values(), key=lambda seed: seed.metrics.get(objective, float("-inf")), reverse=True)


def _certificate_row(certificate: MetricCertificate) -> dict[str, Any]:
    return {
        "metric": certificate.metric,
        "certified": certificate.certified,
        "certificate": certificate.certificate,
        "incumbent_value": certificate.incumbent_value,
        "best_bound": certificate.best_bound,
        "relative_gap": certificate.relative_gap,
        "retained_floor": certificate.retained_floor,
        "rounds": certificate.rounds,
    }


def _write_report(
    path: Path,
    *,
    run_id: str,
    decision: str,
    contract: Any,
    hardware: Any,
    min_cert: MetricCertificate,
    high_cert: MetricCertificate | None,
    final_metrics: dict[str, Any],
) -> None:
    rows = [
        "# MEDIROAD Stage 4.2D — Equity Front-Stage Certification",
        "",
        f"- Run: `{run_id}`",
        f"- Decision: `{decision}`",
        f"- Frozen gap: `{contract.near_optimal_gap:.6f}`",
        f"- Candidate universe: `{contract.candidate_set}`",
        f"- Source Stage4.2C run: `{contract.source_run_id}`",
        f"- CPU solver threads: `{hardware.solver_threads}`",
        f"- GPU backend: `{hardware.gpu_backend}`",
        "",
        "## Certification",
        "",
        "| Metric | Certified | Certificate | Incumbent | Bound | Gap |",
        "|---|---:|---|---:|---:|---:|",
        f"| min_sigungu_coverage | {min_cert.certified} | {min_cert.certificate} | {min_cert.incumbent_value:.12g} | {min_cert.best_bound if min_cert.best_bound is not None else ''} | {min_cert.relative_gap if min_cert.relative_gap is not None else ''} |",
    ]
    if high_cert is not None:
        rows.append(
            f"| high_need_population | {high_cert.certified} | {high_cert.certificate} | {high_cert.incumbent_value:.12g} | {high_cert.best_bound if high_cert.best_bound is not None else ''} | {high_cert.relative_gap if high_cert.relative_gap is not None else ''} |"
        )
    rows.extend(
        [
            "",
            "## Final incumbent metrics",
            "",
            "```json",
            json.dumps(final_metrics, ensure_ascii=False, indent=2),
            "```",
            "",
            (
                "This run certifies only the frozen Top3 Equity front stages. It does not certify candidate expansion, field validation, Operational Final, or Stage5."
                if min_cert.certified and high_cert is not None and high_cert.certified
                else "This run did not certify both frozen Top3 Equity front stages. No downstream promotion is allowed."
            ),
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _oracle_model_summary(oracle: ThresholdOracleModel) -> dict[str, Any]:
    """Return the serializable proof contract without embedding sparse matrices."""

    return {
        "metric": oracle.metric,
        "backend": oracle.backend,
        "threshold": oracle.threshold,
        "incumbent_value": oracle.incumbent_value,
        "policy_relative_gap": oracle.policy_relative_gap,
        "strict_threshold_delta_ratio": (
            oracle.threshold - oracle.incumbent_value
        )
        / max(abs(oracle.incumbent_value), 1e-12),
        "retained_floor": oracle.retained_floor,
        "total_population_floor": oracle.total_population_floor,
        "min_sigungu_floor": oracle.min_sigungu_floor,
        "incumbent_selection": list(oracle.incumbent_selection),
        "selection_digest": oracle.selection_digest,
        "exact_model": {
            "rows": oracle.exact_model.n_row,
            "cols": oracle.exact_model.n_col,
            "nnz": int(oracle.exact_model.A.nnz),
            "sha256": oracle.exact_model_digest,
        },
        "proof_model": {
            "rows": oracle.model.n_row,
            "cols": oracle.model.n_col,
            "nnz": int(oracle.model.A.nnz),
            "sha256": oracle.proof_model_digest,
        },
        "universe_sha256": oracle.universe_digest,
        "cut_sha256": oracle.cut_digest,
        "cut_seed_manifest_sha256": oracle.cut_seed_digest,
        "metadata": oracle.metadata,
    }


def _pinned_selection(
    seeds: list[EvidenceSeed],
    expected_digest: str,
    *,
    label: str,
) -> np.ndarray:
    """Select one frozen incumbent by its order-independent exact digest."""

    matches = [
        np.asarray(seed.selected_indices, dtype=np.int64)
        for seed in seeds
        if selection_digest(seed.selected_indices) == expected_digest
    ]
    if not matches:
        raise RuntimeError(f"Frozen {label} selection is absent from the sealed evidence set")
    canonical = {tuple(sorted(map(int, selected.tolist()))) for selected in matches}
    if len(canonical) != 1:
        raise RuntimeError(f"Frozen {label} selection digest is ambiguous")
    return np.asarray(next(iter(canonical)), dtype=np.int64)


def _exact_to_proof_transport_sufficient(oracle: ThresholdOracleModel) -> bool:
    audit = (
        oracle.metadata.get("proof_relaxation", {})
        .get("exact_to_proof_transport_audit", {})
    )
    return isinstance(audit, dict) and audit.get("sufficient") is True


def _write_formal_oracle_report(
    path: Path,
    *,
    run_id: str,
    decision: str,
    contract: Any,
    hardware: Any,
    min_oracle: ThresholdOracleModel,
    min_result: ThresholdOracleResult,
    high_oracle: ThresholdOracleModel | None,
    high_result: ThresholdOracleResult | None,
    high_mps: Any | None,
    final_metrics: dict[str, Any],
) -> None:
    high_status = high_result.status if high_result is not None else "PREREQUISITE_SKIPPED"
    high_certified = bool(
        high_result is not None and high_result.certified_threshold_infeasible
    )
    rows = [
        "# MEDIROAD Stage 4.2D Equity Front-Stage Certification",
        "",
        f"- Run: `{run_id}`",
        f"- Decision: `{decision}`",
        f"- Scope: frozen `{contract.candidate_set}` Equity front stages only",
        f"- Policy near-optimality gap: `{contract.near_optimal_gap:.6f}`",
        f"- Source Stage 4.2C run: `{contract.source_run_id}`",
        "",
        "## Solver-native threshold certificates",
        "",
        "These are solver-native computational infeasibility certificates, not independently checkable formal proof logs.",
        "",
        "| Metric | Solver | Status | Incumbent | Strict threshold | Certified |",
        "|---|---|---|---:|---:|---:|",
        (
            f"| min_sigungu_coverage | HiGHS {min_result.solver_version} | "
            f"{min_result.status} | {min_result.incumbent_value:.15g} | "
            f"{min_result.threshold:.15g} | {min_result.certified_threshold_infeasible} |"
        ),
        (
            f"| high_need_population | SCIP spawned portfolio | {high_status} | "
            f"{high_result.incumbent_value if high_result is not None else ''} | "
            f"{high_result.threshold if high_result is not None else ''} | {high_certified} |"
        ),
        "",
        "## Exact reconstruction and numerical transport",
        "",
        (
            f"- Min exact/proof model: `{min_oracle.exact_model.n_row}` rows, "
            f"`{min_oracle.exact_model.n_col}` columns, `{min_oracle.exact_model.A.nnz}` nonzeros; "
            "passed in memory to HiGHS with canonical digest verification."
        ),
    ]
    if high_oracle is not None and high_mps is not None:
        rows.extend(
            [
                (
                    f"- High exact/proof model: `{high_oracle.exact_model.n_row}` rows, "
                    f"`{high_oracle.exact_model.n_col}` columns, "
                    f"`{high_oracle.exact_model.A.nnz}` nonzeros."
                ),
                "- Binary coverage variables preserve the projected venue-plan feasible region under the verified one-link/nonnegative-benefit hypotheses.",
                "- Exact-to-proof and proof serialization were outward-relaxed and machine-audited before SCIP execution.",
                f"- Proof MPS SHA-256: `{high_mps.sha256}`.",
            ]
        )
    rows.extend(
        [
            "",
            "## Hardware actually used by the proof path",
            "",
            f"- Min oracle: HiGHS with `{hardware.solver_threads}` CPU threads.",
            (
                f"- High oracle: `{hardware.portfolio_workers}` independent SCIP workers x "
                f"`{hardware.threads_per_portfolio_worker}` CPU thread."
            ),
            "- RTX GPU acceleration was used in prior diagnostic primal exploration, but these exact solver-native proof engines are CPU-bound; no GPU proof acceleration is claimed.",
            "",
            "## Pinned incumbent metrics",
            "",
            "```json",
            json.dumps(final_metrics, ensure_ascii=False, indent=2),
            "```",
            "",
            "## Scope ceiling",
            "",
            (
                "Both frozen Top3 Equity front-stage thresholds were certified. "
                "Candidate expansion, later Equity objectives, field validation, Operational Final, and Stage 5 remain uncertified or prohibited."
                if min_result.certified_threshold_infeasible and high_certified
                else "Both frozen Top3 Equity front stages were not certified; no promotion is allowed."
            ),
        ]
    )
    _atomic_write_text(path, "\n".join(rows) + "\n")


def _run_formal_oracle_pipeline(
    *,
    root: Path,
    config: dict[str, Any],
    run_mode: str,
    is_official: bool,
    hardware: Any,
    process_priority_status: Any,
    fingerprint: str,
    fingerprint_payload: dict[str, Any],
    run_id: str,
    output_root: Path,
    run_dir: Path,
    report_dir: Path,
    bcfg_path: Path,
    ccfg_path: Path,
    dcfg_path: Path,
    formulation: CertificationFormulation,
    contract: Any,
    greedy_population: float,
    parent_before: dict[str, Any],
    authoritative_inputs_before: list[dict[str, Any]],
    stage5_before: dict[str, Any],
    source_before: list[dict[str, Any]],
    evidence_records: list[dict[str, Any]],
    regression: dict[str, Any],
    regression_passed: bool,
    running: Path,
    failed: Path,
    committed: Path,
) -> dict[str, Any]:
    """Run the frozen, solver-native threshold decision models and commit last."""

    oracle_cfg = config["oracle_certification"]
    references = oracle_cfg["frozen_references"]
    min_cfg = oracle_cfg["min_highs"]
    high_cfg = oracle_cfg["high_scip"]

    min_selected = _pinned_selection(
        contract.seeds,
        str(references["min_selection_sha256"]),
        label="min-sigungu incumbent",
    )
    high_selected = _pinned_selection(
        contract.seeds,
        str(references["high_selection_sha256"]),
        label="high-need incumbent",
    )

    min_dir = run_dir / "01_min_threshold_oracle"
    min_oracle = build_min_highs_oracle(
        formulation,
        total_population_floor=float(references["total_population_floor"]),
        threshold=float(references["min_sigungu_strict_threshold"]),
        incumbent_value=float(references["min_sigungu_incumbent"]),
        retained_floor=float(references["total_population_floor"]),
        incumbent_selection=min_selected,
        expected_selection_digest=str(references["min_selection_sha256"]),
        policy_relative_gap=float(contract.near_optimal_gap),
        expected_dimensions={
            "rows": int(min_cfg["expected_rows"]),
            "cols": int(min_cfg["expected_cols"]),
            "nnz": int(min_cfg["expected_nnz"]),
        },
    )
    if (
        min_oracle.exact_model_digest != str(min_cfg["expected_exact_model_sha256"])
        or min_oracle.proof_model_digest != str(min_cfg["expected_proof_model_sha256"])
        or min_oracle.universe_digest != str(min_cfg["expected_universe_sha256"])
    ):
        raise RuntimeError("Frozen min threshold model/universe digest changed")
    atomic_write_json(min_dir / "oracle_model_summary.json", _oracle_model_summary(min_oracle))
    min_result = run_min_highs_oracle(min_oracle, formulation, output_dir=min_dir)
    min_certificate = certificate_from_threshold_oracle(min_result)
    atomic_write_json(min_dir / "threshold_result.json", asdict(min_result))
    atomic_write_json(min_dir / "certificate.json", asdict(min_certificate))

    high_oracle: ThresholdOracleModel | None = None
    high_result: ThresholdOracleResult | None = None
    high_certificate: MetricCertificate | None = None
    high_mps: Any | None = None
    high_dir = run_dir / "02_high_threshold_oracle"
    if min_result.certified_threshold_infeasible:
        high_oracle = build_high_scip_oracle(
            formulation,
            total_population_floor=float(references["total_population_floor"]),
            min_sigungu_floor=float(references["min_sigungu_retained_floor"]),
            threshold=float(references["high_need_strict_threshold"]),
            incumbent_value=float(references["high_need_incumbent"]),
            retained_floor=float(references["min_sigungu_retained_floor"]),
            incumbent_selection=high_selected,
            expected_selection_digest=str(references["high_selection_sha256"]),
            cut_seeds=contract.seeds,
            expected_cut_seed_count=int(references["anchor_seed_count"]),
            expected_cut_seed_digest=str(references["anchor_seed_manifest_sha256"]),
            policy_relative_gap=float(contract.near_optimal_gap),
            anchor_sizes=tuple(int(value) for value in high_cfg["anchor_sizes"]),
            max_cuts=int(high_cfg["max_cuts"]),
            expected_cut_count=int(high_cfg["expected_cut_count"]),
            expected_anchor_count=int(high_cfg["expected_unique_anchor_count"]),
            expected_dimensions={
                "rows": int(high_cfg["expected_rows"]),
                "cols": int(high_cfg["expected_cols"]),
                "nnz": int(high_cfg["expected_nnz"]),
            },
        )
        if (
            high_oracle.exact_model_digest
            != str(high_cfg["expected_exact_model_sha256"])
            or high_oracle.proof_model_digest
            != str(high_cfg["expected_proof_model_sha256"])
            or high_oracle.universe_digest
            != str(high_cfg["expected_universe_sha256"])
            or high_oracle.cut_digest != str(high_cfg["expected_cut_sha256"])
        ):
            raise RuntimeError("Frozen high threshold model/universe/cut digest changed")
        atomic_write_json(
            high_dir / "oracle_model_summary.json", _oracle_model_summary(high_oracle)
        )
        high_mps = write_mps(high_oracle, high_dir / "high_threshold_oracle.mps")
        if high_mps.sha256 != str(high_cfg["expected_mps_sha256"]):
            raise RuntimeError("Frozen high threshold proof MPS digest changed")
        atomic_write_json(high_dir / "mps_artifact.json", asdict(high_mps))
        high_result = run_scip_portfolio(
            high_oracle,
            formulation,
            high_mps,
            output_dir=high_dir / "portfolio",
            seeds=tuple(int(value) for value in high_cfg["portfolio_seeds"]),
            workers=int(high_cfg["workers"]),
            threads_per_worker=int(high_cfg["threads_per_worker"]),
            time_limit_sec=float(high_cfg["time_limit_sec_per_seed"]),
            memory_limit_mib=float(high_cfg["memory_limit_mib_per_worker"]),
            require_all_workers_infeasible=bool(
                high_cfg["require_all_workers_infeasible"]
            ),
        )
        high_certificate = certificate_from_threshold_oracle(high_result)
        atomic_write_json(high_dir / "threshold_result.json", asdict(high_result))
        atomic_write_json(high_dir / "certificate.json", asdict(high_certificate))
    else:
        atomic_write_json(
            high_dir / "PREREQUISITE_SKIPPED.json",
            {
                "reason": "MIN_SIGUNGU_THRESHOLD_NOT_CERTIFIED",
                "high_oracle_executed": False,
                "promotion_allowed": False,
            },
        )

    final_selected = np.sort(
        np.asarray(
            high_result.incumbent_selection if high_result is not None else min_result.incumbent_selection,
            dtype=np.int64,
        )
    )
    final_metrics = formulation.metrics(final_selected)
    final_plan = formulation.candidates.iloc[final_selected].copy().reset_index(drop=True)
    final_plan.insert(0, "selection_order", np.arange(1, len(final_plan) + 1))
    final_plan.insert(1, "candidate_index", final_selected)
    final_dir = run_dir / "03_final_incumbent"
    atomic_write_csv(final_dir / "plan__equity_front_stage_seed.csv", final_plan)
    atomic_write_json(final_dir / "metrics.json", final_metrics)
    atomic_write_csv(
        final_dir / "certification_summary.csv",
        pd.DataFrame(
            [
                _certificate_row(min_certificate),
                *(
                    [_certificate_row(high_certificate)]
                    if high_certificate is not None
                    else []
                ),
            ]
        ),
    )

    parent_after = _pointer_snapshot(root)
    authoritative_inputs_after = _authoritative_input_snapshot(root)
    stage5_after = _stage5_snapshot(root)
    source_after = _source_snapshot(root, [bcfg_path, ccfg_path, dcfg_path])
    parent_unchanged = parent_before == parent_after
    authoritative_inputs_unchanged = authoritative_inputs_before == authoritative_inputs_after
    stage5_unchanged = stage5_before == stage5_after
    stage5_absent_before = int(stage5_before.get("count", 0)) == 0
    source_unchanged = source_before == source_after
    evidence_unchanged = _verify_evidence_records(root, evidence_records)

    min_dimensions_verified = (
        min_oracle.exact_model.n_row == int(min_cfg["expected_rows"])
        and min_oracle.exact_model.n_col == int(min_cfg["expected_cols"])
        and int(min_oracle.exact_model.A.nnz) == int(min_cfg["expected_nnz"])
    )
    min_frozen_digests_verified = bool(
        min_oracle.exact_model_digest == str(min_cfg["expected_exact_model_sha256"])
        and min_oracle.proof_model_digest == str(min_cfg["expected_proof_model_sha256"])
        and min_oracle.universe_digest == str(min_cfg["expected_universe_sha256"])
    )
    min_in_memory_digest_verified = bool(
        min_result.exact_model_digest == min_oracle.exact_model_digest
        and min_result.proof_model_digest == min_oracle.proof_model_digest
        and min_result.evidence.get("model_transport", {}).get("proof_model_digest")
        == min_oracle.proof_model_digest
    )
    high_dimensions_verified = bool(
        high_oracle is not None
        and high_oracle.exact_model.n_row == int(high_cfg["expected_rows"])
        and high_oracle.exact_model.n_col == int(high_cfg["expected_cols"])
        and int(high_oracle.exact_model.A.nnz) == int(high_cfg["expected_nnz"])
    )
    high_frozen_digests_verified = bool(
        high_oracle is not None
        and high_mps is not None
        and high_oracle.exact_model_digest == str(high_cfg["expected_exact_model_sha256"])
        and high_oracle.proof_model_digest == str(high_cfg["expected_proof_model_sha256"])
        and high_oracle.universe_digest == str(high_cfg["expected_universe_sha256"])
        and high_oracle.cut_digest == str(high_cfg["expected_cut_sha256"])
        and high_mps.sha256 == str(high_cfg["expected_mps_sha256"])
    )
    high_exact_transport = bool(
        high_oracle is not None and _exact_to_proof_transport_sufficient(high_oracle)
    )
    high_mps_transport = bool(
        high_mps is not None
        and high_mps.transport_sufficient
        and isinstance(high_mps.transport_audit, dict)
        and high_mps.transport_audit.get("sufficient") is True
    )
    high_workers_complete = bool(
        high_result is not None
        and high_result.evidence.get("all_workers_complete") is True
        and len(high_result.outcomes) == int(high_cfg["workers"])
    )
    high_workers_infeasible = bool(
        high_result is not None
        and high_result.evidence.get("all_workers_infeasible") is True
        and all(
            str(outcome.get("status", "")).lower() == "infeasible"
            and outcome.get("has_solution") is False
            for outcome in high_result.outcomes
        )
    )
    frozen_selections_verified = bool(
        min_oracle.selection_digest == str(references["min_selection_sha256"])
        and (
            high_oracle is not None
            and high_oracle.selection_digest == str(references["high_selection_sha256"])
        )
    )
    frozen_seed_manifest_verified = bool(
        high_oracle is not None
        and high_oracle.cut_seed_digest
        == str(references["anchor_seed_manifest_sha256"])
    )
    min_result_bound_to_model = bool(
        min_result.selection_digest == min_oracle.selection_digest
        and min_result.exact_model_digest == min_oracle.exact_model_digest
        and min_result.proof_model_digest == min_oracle.proof_model_digest
        and min_result.universe_digest == min_oracle.universe_digest
        and min_result.mps_sha256 is None
    )
    high_result_bound_to_model = bool(
        high_result is not None
        and high_oracle is not None
        and high_mps is not None
        and high_result.selection_digest == high_oracle.selection_digest
        and high_result.exact_model_digest == high_oracle.exact_model_digest
        and high_result.proof_model_digest == high_oracle.proof_model_digest
        and high_result.universe_digest == high_oracle.universe_digest
        and high_result.cut_digest == high_oracle.cut_digest
        and high_result.mps_sha256 == high_mps.sha256
    )
    high_mps_current_bytes_bound = bool(
        high_mps is not None
        and high_mps.path.is_file()
        and int(high_mps.path.stat().st_size) == int(high_mps.size_bytes)
        and sha256_file(high_mps.path) == high_mps.sha256
    )
    min_solver_version_and_options_verified = bool(
        min_result.solver_version == str(min_cfg["highspy_version"])
        and min_result.requested_options == min_result.readback_options
    )
    expected_high_portfolio_options = (
        {
            "seeds": [int(value) for value in high_cfg["portfolio_seeds"]],
            "workers": int(high_cfg["workers"]),
            "threads_per_worker": int(high_cfg["threads_per_worker"]),
            "time_limit_sec_per_seed": float(high_cfg["time_limit_sec_per_seed"]),
            "memory_limit_mib_per_worker": float(high_cfg["memory_limit_mib_per_worker"]),
            "require_all_workers_infeasible": True,
        }
        if high_result is not None
        else None
    )
    high_solver_versions_verified = bool(
        high_result is not None
        and len(high_result.outcomes) == int(high_cfg["workers"])
        and high_result.requested_options == expected_high_portfolio_options
        and all(
            str(outcome.get("pyscipopt_version")) == str(high_cfg["pyscipopt_version"])
            and tuple(
                int(outcome.get("scip_version_components", {}).get(name, -1))
                for name in ("major", "minor", "tech")
            )
            == tuple(int(value) for value in high_cfg["scip_version"])
            and outcome.get("requested_parameters") == outcome.get("readback_parameters")
            and outcome.get("requested_parameters")
            == {
                "limits/time": float(high_cfg["time_limit_sec_per_seed"]),
                "limits/gap": 0.0,
                "limits/memory": float(high_cfg["memory_limit_mib_per_worker"]),
                "parallel/maxnthreads": int(high_cfg["threads_per_worker"]),
                "lp/threads": int(high_cfg["threads_per_worker"]),
                "randomization/randomseedshift": int(outcome.get("seed", -1)),
            }
            for outcome in high_result.outcomes
        )
    )
    high_cut_contract_verified = bool(
        high_oracle is not None
        and int(high_oracle.metadata.get("cut_count", -1))
        == int(high_cfg["expected_cut_count"])
        and int(
            high_oracle.metadata.get("cut_generation", {}).get(
                "unique_anchor_count", -1
            )
        )
        == int(high_cfg["expected_unique_anchor_count"])
    )
    oracle_models_rebuilt = bool(
        oracle_cfg["rebuild_model_in_official_run"] is True
        and min_oracle.exact_model_digest
        and high_oracle is not None
        and high_oracle.exact_model_digest
    )
    diagnostic_artifacts_not_accepted = oracle_cfg["accept_diagnostic_artifacts"] is False
    resolved_parallelism_verified = bool(
        int(hardware.solver_threads) == int(min_cfg["threads"])
        and int(hardware.portfolio_workers) == int(high_cfg["workers"])
        and int(hardware.threads_per_portfolio_worker)
        == int(high_cfg["threads_per_worker"])
    )
    computational_certified = bool(
        min_result.certified_threshold_infeasible
        and high_result is not None
        and high_result.certified_threshold_infeasible
    )
    oracle_checks = {
        "oracle_models_rebuilt_in_current_run": oracle_models_rebuilt,
        "diagnostic_artifacts_not_accepted_as_certificates": diagnostic_artifacts_not_accepted,
        "resolved_cpu_parallelism_matches_frozen_contract": resolved_parallelism_verified,
        "min_dimensions_verified": min_dimensions_verified,
        "min_frozen_model_and_universe_digests_verified": min_frozen_digests_verified,
        "min_exact_to_proof_transport_sufficient": _exact_to_proof_transport_sufficient(min_oracle),
        "min_in_memory_model_digest_verified": min_in_memory_digest_verified,
        "min_result_bound_to_rebuilt_model": min_result_bound_to_model,
        "min_solver_version_and_options_verified": min_solver_version_and_options_verified,
        "min_threshold_infeasible": bool(min_result.certified_threshold_infeasible),
        "high_dimensions_verified": high_dimensions_verified,
        "high_frozen_model_universe_cut_and_mps_digests_verified": high_frozen_digests_verified,
        "high_exact_to_proof_transport_sufficient": high_exact_transport,
        "high_mps_transport_sufficient": high_mps_transport,
        "high_all_workers_complete": high_workers_complete,
        "high_all_workers_infeasible": high_workers_infeasible,
        "high_result_bound_to_rebuilt_model_and_mps": high_result_bound_to_model,
        "high_mps_current_bytes_bound_to_result": high_mps_current_bytes_bound,
        "high_solver_versions_and_options_verified": high_solver_versions_verified,
        "high_cut_and_anchor_counts_verified": high_cut_contract_verified,
        "high_threshold_infeasible": bool(
            high_result is not None and high_result.certified_threshold_infeasible
        ),
        "frozen_incumbent_selections_verified": frozen_selections_verified,
        "frozen_cut_seed_manifest_verified": frozen_seed_manifest_verified,
        "frozen_policy_gap_exact": float(contract.near_optimal_gap).hex()
        == float(0.005).hex(),
    }
    integrity_checks = {
        "parent_pointers_unchanged": parent_unchanged,
        "authoritative_input_inventories_unchanged": authoritative_inputs_unchanged,
        "stage5_namespace_absent_before": stage5_absent_before,
        "stage5_namespace_unchanged": stage5_unchanged,
        "tested_source_unchanged": source_unchanged,
        "regression_suite_passed": regression_passed,
        "frozen_stage42c_evidence_unchanged": evidence_unchanged,
    }
    integrity_passed = all(integrity_checks.values())
    formal_passed = all(oracle_checks.values())
    if is_official:
        gate_passed = bool(computational_certified and formal_passed and integrity_passed)
        decision = (
            "PASS_STAGE4_2D_EQUITY_FRONT_STAGES_CERTIFIED"
            if gate_passed
            else "FAIL_STAGE4_2D_EQUITY_FRONT_STAGES_UNCERTIFIED"
        )
        promotable_scope = "TOP3_EQUITY_FRONT_STAGES_ONLY" if gate_passed else "NONE"
    else:
        gate_passed = bool(integrity_passed and formal_passed)
        decision = (
            "PASS_STAGE4_2D_DIAGNOSTIC_NOT_PROMOTABLE"
            if gate_passed
            else "FAIL_STAGE4_2D_DIAGNOSTIC_INTEGRITY_OR_ORACLE_GATE"
        )
        promotable_scope = "NONE"
    gate = {
        "mode": run_mode,
        "decision": decision,
        "passed": gate_passed,
        "computational_certified": computational_certified,
        "promotable_scope": promotable_scope,
        "frozen_relative_gap": contract.near_optimal_gap,
        "certificate_kind": "SOLVER_NATIVE_THRESHOLD_INFEASIBILITY",
        "independently_checkable_formal_proof": False,
        "checks": {
            **integrity_checks,
            **oracle_checks,
            "candidate_expansion_certified": False,
            "stage4_full_computational_complete": False,
            "operational_final": False,
            "stage5_started": False,
            "stage5_release_allowed": False,
        },
    }
    gate_dir = run_dir / "04_quality_and_provenance"
    atomic_write_json(gate_dir / "quality_gate.json", gate)
    atomic_write_json(gate_dir / "parent_pointers_after.json", parent_after)
    atomic_write_json(
        gate_dir / "authoritative_inputs_after.json", authoritative_inputs_after
    )
    atomic_write_json(
        gate_dir / "stage5_namespace_before_after.json",
        {"before": stage5_before, "after": stage5_after},
    )
    atomic_write_json(gate_dir / "tested_source_before.json", source_before)
    atomic_write_json(gate_dir / "tested_source_after.json", source_after)

    evidence_paths = [
        min_dir / "oracle_model_summary.json",
        min_dir / "threshold_result.json",
        min_dir / "certificate.json",
        final_dir / "plan__equity_front_stage_seed.csv",
        gate_dir / "quality_gate.json",
    ]
    if high_result is not None and high_mps is not None:
        evidence_paths.extend(
            [
                high_dir / "oracle_model_summary.json",
                high_dir / "high_threshold_oracle.mps",
                high_dir / "mps_artifact.json",
                high_dir / "threshold_result.json",
                high_dir / "certificate.json",
            ]
        )
    metadata = {
        "run_id": run_id,
        "package": PACKAGE_NAME,
        "version": VERSION,
        "created_utc": utc_now(),
        "fingerprint": fingerprint,
        "fingerprint_payload": fingerprint_payload,
        "hardware": hardware.to_dict(),
        "process_priority_status": process_priority_status,
        "regression": regression,
        "candidate_count": formulation.nx,
        "pattern_count": formulation.ny,
        "coverage_incidence_count": int(formulation._candidate_pattern.nnz),
        "greedy_population": float(greedy_population),
        "quality_gate": gate,
        "min_oracle": _oracle_model_summary(min_oracle),
        "high_oracle": _oracle_model_summary(high_oracle) if high_oracle is not None else None,
        "evidence_artifacts": [
            {
                "relative_path": path.relative_to(root).as_posix(),
                "size_bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
            for path in evidence_paths
        ],
        "stage4_2d_front_stages_certified": computational_certified,
        "stage4_full_computational_complete": False,
        "operational_final": False,
        "stage5_started": False,
        "stage5_release_allowed": False,
    }
    metadata_path = run_dir / "metadata.json"
    atomic_write_json(metadata_path, metadata)
    report_path = report_dir / "FINAL_REPORT.md"
    _write_formal_oracle_report(
        report_path,
        run_id=run_id,
        decision=decision,
        contract=contract,
        hardware=hardware,
        min_oracle=min_oracle,
        min_result=min_result,
        high_oracle=high_oracle,
        high_result=high_result,
        high_mps=high_mps,
        final_metrics=final_metrics,
    )
    atomic_write_json(report_dir / "quality_gate.json", gate)

    marker_names = {".RUNNING", ".FAILED", ".COMMITTED", "ARTIFACT_INVENTORY.csv"}
    artifacts = [
        path for path in run_dir.rglob("*") if path.is_file() and path.name not in marker_names
    ] + [path for path in report_dir.rglob("*") if path.is_file()]
    inventory_path = run_dir / "ARTIFACT_INVENTORY.csv"
    atomic_write_csv(inventory_path, pd.DataFrame(tree_inventory(artifacts, root)))
    _verify_output_inventory(root, inventory_path)

    if _pointer_snapshot(root) != parent_before:
        raise RuntimeError("Authoritative parent pointer changed before Stage4.2D commit")
    if _authoritative_input_snapshot(root) != authoritative_inputs_before:
        raise RuntimeError("Authoritative parent inventory changed before Stage4.2D commit")
    if _source_snapshot(root, [bcfg_path, ccfg_path, dcfg_path]) != source_before:
        raise RuntimeError("Tested Stage4.2D source changed before commit")
    if _stage5_snapshot(root) != stage5_before:
        raise RuntimeError("Stage5 namespace changed before Stage4.2D commit")
    if not _verify_evidence_records(root, evidence_records):
        raise RuntimeError("Frozen Stage4.2C evidence changed before Stage4.2D commit")
    _verify_output_inventory(root, inventory_path)
    if high_mps is not None and (
        not high_mps.path.is_file()
        or int(high_mps.path.stat().st_size) != int(high_mps.size_bytes)
        or sha256_file(high_mps.path) != high_mps.sha256
        or high_result is None
        or high_result.mps_sha256 != high_mps.sha256
    ):
        raise RuntimeError("High-threshold proof MPS changed before Stage4.2D commit")

    running.unlink(missing_ok=True)
    if gate_passed:
        atomic_write_json(
            committed,
            {
                "run_id": run_id,
                "committed_utc": utc_now(),
                "mode": run_mode,
                "inventory_sha256": sha256_file(inventory_path),
            },
        )
        if is_official and computational_certified:
            quality_path = gate_dir / "quality_gate.json"
            min_certificate_path = min_dir / "certificate.json"
            high_certificate_path = high_dir / "certificate.json"
            high_result_path = high_dir / "threshold_result.json"
            high_mps_path = high_dir / "high_threshold_oracle.mps"
            pointer = output_root / "CURRENT_STAGE4_2D_EQUITY_CERTIFICATION_RUN.json"
            atomic_write_json(
                pointer,
                {
                    "run_id": run_id,
                    "run_relative_path": run_dir.relative_to(root).as_posix(),
                    "metadata_relative_path": metadata_path.relative_to(root).as_posix(),
                    "metadata_sha256": sha256_file(metadata_path),
                    "inventory_relative_path": inventory_path.relative_to(root).as_posix(),
                    "inventory_sha256": sha256_file(inventory_path),
                    "report_relative_path": report_path.relative_to(root).as_posix(),
                    "report_sha256": sha256_file(report_path),
                    "quality_gate_relative_path": quality_path.relative_to(root).as_posix(),
                    "quality_gate_sha256": sha256_file(quality_path),
                    "frozen_contract_relative_path": (run_dir / "frozen_contract.json").relative_to(root).as_posix(),
                    "frozen_contract_sha256": sha256_file(run_dir / "frozen_contract.json"),
                    "min_certificate_relative_path": min_certificate_path.relative_to(root).as_posix(),
                    "min_certificate_sha256": sha256_file(min_certificate_path),
                    "high_certificate_relative_path": high_certificate_path.relative_to(root).as_posix(),
                    "high_certificate_sha256": sha256_file(high_certificate_path),
                    "high_portfolio_result_relative_path": high_result_path.relative_to(root).as_posix(),
                    "high_portfolio_result_sha256": sha256_file(high_result_path),
                    "high_proof_mps_relative_path": high_mps_path.relative_to(root).as_posix(),
                    "high_proof_mps_sha256": sha256_file(high_mps_path),
                    "decision": decision,
                    "scope": promotable_scope,
                    "certificate_kind": "SOLVER_NATIVE_THRESHOLD_INFEASIBILITY",
                    "stage4_2d_front_stages_certified": True,
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
            output_root / "LATEST_FAILED_STAGE4_2D_EQUITY_RUN.json",
            {"run_id": run_id, "run_dir": run_dir.relative_to(root).as_posix(), "decision": decision},
        )
    return {
        "run_id": run_id,
        "passed": bool(gate_passed),
        "decision": decision,
        "run_dir": str(run_dir),
        "report_dir": str(report_dir),
        "min_sigungu": _certificate_row(min_certificate),
        "high_need": _certificate_row(high_certificate) if high_certificate is not None else None,
    }


def run_equity_certification(
    *,
    project_root: Path,
    stage42_config_path: Path,
    stage42c_config_path: Path,
    stage42d_config_path: Path,
) -> dict[str, Any]:
    set_thread_environment()
    root = project_root.resolve()
    dcfg_path = stage42d_config_path if stage42d_config_path.is_absolute() else root / stage42d_config_path
    ccfg_path = stage42c_config_path if stage42c_config_path.is_absolute() else root / stage42c_config_path
    bcfg_path = stage42_config_path if stage42_config_path.is_absolute() else root / stage42_config_path
    config = load_yaml(dcfg_path)
    validate_config(config)
    run_mode = str(config.get("mode", "official"))
    is_official = run_mode == "official"
    base_config = load_base_stage42_config(bcfg_path)
    cert_config = load_yaml(ccfg_path)
    validate_certification_config(cert_config)
    validate_parent_contracts(base_config, cert_config)
    hardware = resolve_hardware(config)
    process_priority_status = apply_process_priority(config)
    source_before = _source_snapshot(root, [bcfg_path, ccfg_path, dcfg_path])
    authoritative_inputs_before = _authoritative_input_snapshot(root)

    fingerprint_payload = {
        "package": PACKAGE_NAME,
        "version": VERSION,
        "stage42": sha256_file(bcfg_path),
        "stage42c": sha256_file(ccfg_path),
        "stage42d": sha256_file(dcfg_path),
        "source_snapshot": source_before,
        "parents": _pointer_snapshot(root),
        "authoritative_inputs": authoritative_inputs_before,
    }
    import hashlib

    fingerprint = hashlib.sha256(json.dumps(fingerprint_payload, sort_keys=True).encode()).hexdigest()
    run_id = make_run_id("stage4_2d_equity", fingerprint)
    output_root = root / str(config["paths"]["output_root"])
    report_root = root / str(config["paths"]["report_root"])
    run_dir = output_root / "runs" / run_id
    report_dir = report_root / "runs" / run_id
    lock_path = root / str(config["paths"]["lock"])
    parent_before = _pointer_snapshot(root)
    stage5_before = _stage5_snapshot(root)
    running = run_dir / ".RUNNING"
    failed = run_dir / ".FAILED"
    committed = run_dir / ".COMMITTED"

    with _exclusive_lock(lock_path):
        run_dir.mkdir(parents=True, exist_ok=False)
        report_dir.mkdir(parents=True, exist_ok=False)
        atomic_write_json(running, {"run_id": run_id, "started_utc": utc_now()})
        try:
            regression = _run_regression_tests(root, run_dir / "00_regression")
            regression_passed = bool(
                regression.get("return_code") == 0
                and int(regression.get("failed_count", 0)) == 0
                and int(regression.get("passed_count", 0)) > 0
            )
            problem = prepare_candidate_problem(root, "top3", base_config, cache={})
            formulation = CertificationFormulation(
                problem.candidates,
                problem.patterns,
                base_config,
                cert_config,
                candidate_set="top3",
            )
            formulation.venue_alias_map = dict(problem.venue_alias_map)
            if formulation.visit_count != int(config["contract"]["visit_count"]):
                raise RuntimeError("Stage4.2 and Stage4.2D visit-count contracts differ")
            greedy_indices, greedy_population = constrained_greedy(
                problem.candidates,
                problem.coverage,
                problem.grid["elderly_population"].to_numpy(float),
                base_config,
            )
            contract = reconstruct_contract(
                root,
                formulation,
                problem.stage41,
                base_config,
                config,
                greedy_indices=greedy_indices,
            )
            atomic_write_json(run_dir / "hardware_profile.json", hardware.to_dict())
            atomic_write_json(run_dir / "frozen_contract.json", asdict(contract))
            atomic_write_json(run_dir / "parent_pointers_before.json", parent_before)
            atomic_write_json(run_dir / "authoritative_inputs_before.json", authoritative_inputs_before)
            evidence_records = [
                *list(contract.evidence.get("preferred_artifacts", [])),
                *list(contract.evidence.get("improving_solution_artifacts", [])),
            ]
            if not _verify_evidence_records(root, evidence_records):
                raise RuntimeError("Frozen Stage4.2C evidence changed before optimization")

            if config["oracle_certification"]["enabled"] is True:
                return _run_formal_oracle_pipeline(
                    root=root,
                    config=config,
                    run_mode=run_mode,
                    is_official=is_official,
                    hardware=hardware,
                    process_priority_status=process_priority_status,
                    fingerprint=fingerprint,
                    fingerprint_payload=fingerprint_payload,
                    run_id=run_id,
                    output_root=output_root,
                    run_dir=run_dir,
                    report_dir=report_dir,
                    bcfg_path=bcfg_path,
                    ccfg_path=ccfg_path,
                    dcfg_path=dcfg_path,
                    formulation=formulation,
                    contract=contract,
                    greedy_population=float(greedy_population),
                    parent_before=parent_before,
                    authoritative_inputs_before=authoritative_inputs_before,
                    stage5_before=stage5_before,
                    source_before=source_before,
                    evidence_records=evidence_records,
                    regression=regression,
                    regression_passed=regression_passed,
                    running=running,
                    failed=failed,
                    committed=committed,
                )

            lns = HybridLargeNeighborhoodSearch(formulation, config)
            min_lns = lns.search(
                contract.seeds,
                objective="min_sigungu_coverage",
                total_population_floor=contract.total_population_floor,
                min_sigungu_floor=None,
                rounds=int(config.get("search", {}).get("min_sigungu_rounds", config.get("search", {}).get("rounds", 80))),
            )
            seeds_min = _dedup_seeds([*min_lns.elites, *contract.seeds], "min_sigungu_coverage")
            atomic_write_json(
                run_dir / "gpu_lns_min_sigungu.json",
                {
                    **asdict(min_lns),
                    "elites": [
                        {"source": seed.source, "selected_indices": seed.selected_indices, "metrics": seed.metrics}
                        for seed in min_lns.elites
                    ],
                },
            )

            base_targets_min = metric_targets(
                formulation,
                total_population_floor=contract.total_population_floor,
                min_sigungu_floor=None,
                high_need_target=None,
            )
            base_fix_min, base_fix_min_info = compute_safe_zero_fixings(
                formulation, base_targets_min, use_gpu=hardware.gpu_available
            )
            base_cuts_min, base_cut_min_info = generate_anchored_submodular_cuts(
                formulation,
                base_targets_min,
                seeds_min,
                anchor_sizes=config.get("strengthening", {}).get("anchor_sizes", [0, 5, 10, 15, 20]),
                max_cuts=int(config.get("strengthening", {}).get("max_base_cuts", 256)),
            )
            atomic_write_json(run_dir / "strengthening_min_base.json", {"fixings": serialize_fixings(base_fix_min), "fixing_diagnostics": base_fix_min_info, "cuts": base_cut_min_info})

            min_direct = formulation.build(
                objective_name="min_sigungu_coverage",
                sense="max",
                floors=[Floor("total_population", "max", contract.total_population_floor)],
                name="stage42d__top3__equity__min_sigungu_direct",
            )

            def min_oracle_builder(target: float) -> Any:
                return formulation.build(
                    objective_name="feasibility",
                    sense="min",
                    floors=[Floor("total_population", "max", contract.total_population_floor)],
                    threshold=Floor("min_sigungu_coverage", "max", target),
                    name=f"stage42d__top3__equity__min_sigungu_threshold_{target:.12g}",
                )

            def min_cut_builder(target: float, seeds: list[EvidenceSeed]) -> tuple[list[Any], list[Any], dict[str, Any]]:
                targets = metric_targets(
                    formulation,
                    total_population_floor=contract.total_population_floor,
                    min_sigungu_floor=target,
                    high_need_target=None,
                )
                fixings, fix_info = compute_safe_zero_fixings(formulation, targets, use_gpu=hardware.gpu_available)
                cuts, cut_info = generate_anchored_submodular_cuts(
                    formulation,
                    targets,
                    seeds,
                    anchor_sizes=config.get("strengthening", {}).get("anchor_sizes", [0, 5, 10, 15, 20]),
                    max_cuts=int(config.get("strengthening", {}).get("max_target_cuts", 1024)),
                )
                return cuts, fixings, {"fixings": serialize_fixings(fixings), "fixing_diagnostics": fix_info, **cut_info}

            certifier = EquityFrontStageCertifier(formulation, config, threads=hardware.solver_threads, output_dir=run_dir)
            min_cert = certifier.certify(
                metric="min_sigungu_coverage",
                direct_model=min_direct,
                oracle_builder=min_oracle_builder,
                seeds=seeds_min,
                base_cuts=base_cuts_min,
                base_fixings=base_fix_min,
                retained_floor=contract.total_population_floor,
                cut_builder=min_cut_builder,
            )
            atomic_write_json(run_dir / "certificate_min_sigungu.json", asdict(min_cert))

            high_cert: MetricCertificate | None = None
            final_selected = np.asarray(min_cert.selected_indices, dtype=int)
            if min_cert.certified:
                min_floor = retained_min_sigungu_floor(
                    float(min_cert.incumbent_value),
                    float(config["contract"]["min_sigungu_retention"]),
                    scale=int(getattr(formulation, "min_sigungu_scale", 1_000_000)),
                )
                stage1_seed = _exact_seed(formulation, "STAGE42D_CERTIFIED_MIN_SIGUNGU", min_cert.selected_indices)
                high_seed_pool = _dedup_seeds([stage1_seed, *seeds_min], "high_need_population")
                high_lns = lns.search(
                    high_seed_pool,
                    objective="high_need_population",
                    total_population_floor=contract.total_population_floor,
                    min_sigungu_floor=min_floor,
                    rounds=int(config.get("search", {}).get("high_need_rounds", config.get("search", {}).get("rounds", 120))),
                )
                seeds_high = _dedup_seeds([*high_lns.elites, *high_seed_pool], "high_need_population")
                atomic_write_json(
                    run_dir / "gpu_lns_high_need.json",
                    {
                        **asdict(high_lns),
                        "elites": [
                            {"source": seed.source, "selected_indices": seed.selected_indices, "metrics": seed.metrics}
                            for seed in high_lns.elites
                        ],
                    },
                )
                base_targets_high = metric_targets(
                    formulation,
                    total_population_floor=contract.total_population_floor,
                    min_sigungu_floor=min_floor,
                    high_need_target=None,
                )
                base_fix_high, base_fix_high_info = compute_safe_zero_fixings(
                    formulation, base_targets_high, use_gpu=hardware.gpu_available
                )
                base_cuts_high, base_cut_high_info = generate_anchored_submodular_cuts(
                    formulation,
                    base_targets_high,
                    seeds_high,
                    anchor_sizes=config.get("strengthening", {}).get("anchor_sizes", [0, 5, 10, 15, 20]),
                    max_cuts=int(config.get("strengthening", {}).get("max_base_cuts", 256)),
                )
                atomic_write_json(run_dir / "strengthening_high_base.json", {"fixings": serialize_fixings(base_fix_high), "fixing_diagnostics": base_fix_high_info, "cuts": base_cut_high_info})
                high_direct = formulation.build(
                    objective_name="high_need_population",
                    sense="max",
                    floors=[
                        Floor("total_population", "max", contract.total_population_floor),
                        Floor("min_sigungu_coverage", "max", min_floor),
                    ],
                    name="stage42d__top3__equity__high_need_direct",
                )

                def high_oracle_builder(target: float) -> Any:
                    return formulation.build(
                        objective_name="feasibility",
                        sense="min",
                        floors=[
                            Floor("total_population", "max", contract.total_population_floor),
                            Floor("min_sigungu_coverage", "max", min_floor),
                        ],
                        threshold=Floor("high_need_population", "max", target),
                        name=f"stage42d__top3__equity__high_need_threshold_{target:.12g}",
                    )

                def high_cut_builder(target: float, seeds: list[EvidenceSeed]) -> tuple[list[Any], list[Any], dict[str, Any]]:
                    targets = metric_targets(
                        formulation,
                        total_population_floor=contract.total_population_floor,
                        min_sigungu_floor=min_floor,
                        high_need_target=target,
                    )
                    fixings, fix_info = compute_safe_zero_fixings(formulation, targets, use_gpu=hardware.gpu_available)
                    cuts, cut_info = generate_anchored_submodular_cuts(
                        formulation,
                        targets,
                        seeds,
                        anchor_sizes=config.get("strengthening", {}).get("anchor_sizes", [0, 5, 10, 15, 20]),
                        max_cuts=int(config.get("strengthening", {}).get("max_target_cuts", 1024)),
                    )
                    return cuts, fixings, {"fixings": serialize_fixings(fixings), "fixing_diagnostics": fix_info, **cut_info}

                high_cert = certifier.certify(
                    metric="high_need_population",
                    direct_model=high_direct,
                    oracle_builder=high_oracle_builder,
                    seeds=seeds_high,
                    base_cuts=base_cuts_high,
                    base_fixings=base_fix_high,
                    retained_floor=min_floor,
                    cut_builder=high_cut_builder,
                )
                atomic_write_json(run_dir / "certificate_high_need.json", asdict(high_cert))
                final_selected = np.asarray(high_cert.selected_indices, dtype=int)

            computational_certified = bool(min_cert.certified and high_cert is not None and high_cert.certified)
            final_metrics = formulation.metrics(final_selected)
            final_plan = formulation.candidates.iloc[final_selected].copy().reset_index(drop=True)
            final_plan.insert(0, "selection_order", np.arange(1, len(final_plan) + 1))
            atomic_write_csv(run_dir / "plan__equity_front_stage_seed.csv", final_plan)
            atomic_write_json(run_dir / "final_incumbent_metrics.json", final_metrics)
            atomic_write_csv(
                run_dir / "certification_summary.csv",
                pd.DataFrame([_certificate_row(min_cert), *([_certificate_row(high_cert)] if high_cert is not None else [])]),
            )

            parent_after = _pointer_snapshot(root)
            authoritative_inputs_after = _authoritative_input_snapshot(root)
            stage5_after = _stage5_snapshot(root)
            source_after = _source_snapshot(root, [bcfg_path, ccfg_path, dcfg_path])
            parent_unchanged = parent_before == parent_after
            authoritative_inputs_unchanged = authoritative_inputs_before == authoritative_inputs_after
            stage5_unchanged = stage5_before == stage5_after
            stage5_absent_before = int(stage5_before.get("count", 0)) == 0
            source_unchanged = source_before == source_after
            evidence_unchanged = _verify_evidence_records(root, evidence_records)
            if is_official:
                gate_passed = (
                    computational_certified
                    and parent_unchanged
                    and authoritative_inputs_unchanged
                    and stage5_unchanged
                    and stage5_absent_before
                    and source_unchanged
                    and evidence_unchanged
                    and regression_passed
                )
                decision = (
                    "PASS_STAGE4_2D_EQUITY_FRONT_STAGES_CERTIFIED"
                    if gate_passed
                    else "FAIL_STAGE4_2D_EQUITY_FRONT_STAGES_UNCERTIFIED"
                )
                promotable_scope = "TOP3_EQUITY_FRONT_STAGES_ONLY" if gate_passed else "NONE"
            else:
                gate_passed = (
                    parent_unchanged
                    and authoritative_inputs_unchanged
                    and stage5_unchanged
                    and stage5_absent_before
                    and source_unchanged
                    and evidence_unchanged
                    and regression_passed
                )
                decision = (
                    "PASS_STAGE4_2D_DIAGNOSTIC_NOT_PROMOTABLE"
                    if gate_passed
                    else "FAIL_STAGE4_2D_DIAGNOSTIC_INTEGRITY_GATE"
                )
                promotable_scope = "NONE"
            gate = {
                "mode": run_mode,
                "decision": decision,
                "passed": gate_passed,
                "computational_certified": computational_certified,
                "promotable_scope": promotable_scope,
                "frozen_relative_gap": contract.near_optimal_gap,
                "checks": {
                    "min_sigungu_certified": bool(min_cert.certified),
                    "high_need_certified": bool(high_cert is not None and high_cert.certified),
                    "parent_pointers_unchanged": parent_unchanged,
                    "authoritative_input_inventories_unchanged": authoritative_inputs_unchanged,
                    "stage5_namespace_absent_before": stage5_absent_before,
                    "stage5_namespace_unchanged": stage5_unchanged,
                    "tested_source_unchanged": source_unchanged,
                    "regression_suite_passed": regression_passed,
                    "frozen_stage42c_evidence_unchanged": evidence_unchanged,
                    "candidate_expansion_certified": False,
                    "operational_final": False,
                },
            }
            if (
                not parent_unchanged
                or not authoritative_inputs_unchanged
                or not stage5_unchanged
                or not stage5_absent_before
                or not source_unchanged
                or not evidence_unchanged
                or not regression_passed
            ):
                gate["passed"] = False
                gate["decision"] = "FAIL_STAGE4_2D_INTEGRITY_OR_STAGE5_GATE"
                gate["promotable_scope"] = "NONE"
            atomic_write_json(run_dir / "quality_gate.json", gate)
            atomic_write_json(run_dir / "parent_pointers_after.json", parent_after)
            atomic_write_json(run_dir / "authoritative_inputs_after.json", authoritative_inputs_after)
            atomic_write_json(run_dir / "stage5_namespace_before_after.json", {"before": stage5_before, "after": stage5_after})
            atomic_write_json(run_dir / "tested_source_before.json", source_before)
            atomic_write_json(run_dir / "tested_source_after.json", source_after)
            metadata = {
                "run_id": run_id,
                "package": PACKAGE_NAME,
                "version": VERSION,
                "created_utc": utc_now(),
                "fingerprint": fingerprint,
                "fingerprint_payload": fingerprint_payload,
                "hardware": hardware.to_dict(),
                "process_priority_status": process_priority_status,
                "regression": regression,
                "candidate_count": formulation.nx,
                "pattern_count": formulation.ny,
                "coverage_incidence_count": int(formulation._candidate_pattern.nnz),
                "greedy_population": float(greedy_population),
                "quality_gate": gate,
            }
            atomic_write_json(run_dir / "metadata.json", metadata)
            _write_report(
                report_dir / "FINAL_REPORT.md",
                run_id=run_id,
                decision=str(gate["decision"]),
                contract=contract,
                hardware=hardware,
                min_cert=min_cert,
                high_cert=high_cert,
                final_metrics=final_metrics,
            )
            atomic_write_json(report_dir / "quality_gate.json", gate)
            marker_names = {".RUNNING", ".FAILED", ".COMMITTED", "ARTIFACT_INVENTORY.csv"}
            artifacts = [
                path
                for path in run_dir.rglob("*")
                if path.is_file() and path.name not in marker_names
            ] + [path for path in report_dir.rglob("*") if path.is_file()]
            inventory_path = run_dir / "ARTIFACT_INVENTORY.csv"
            atomic_write_csv(inventory_path, pd.DataFrame(tree_inventory(artifacts, root)))
            _verify_output_inventory(root, inventory_path)

            # Close the last mutation window after all immutable artifacts are
            # materialized. CURRENT is the sole promotion commit and is written
            # only after these exact rechecks succeed.
            if _pointer_snapshot(root) != parent_before:
                raise RuntimeError("Authoritative parent pointer changed before Stage4.2D commit")
            if _authoritative_input_snapshot(root) != authoritative_inputs_before:
                raise RuntimeError("Authoritative parent inventory changed before Stage4.2D commit")
            if _source_snapshot(root, [bcfg_path, ccfg_path, dcfg_path]) != source_before:
                raise RuntimeError("Tested Stage4.2D source changed before commit")
            if _stage5_snapshot(root) != stage5_before:
                raise RuntimeError("Stage5 namespace changed before Stage4.2D commit")
            if not _verify_evidence_records(root, evidence_records):
                raise RuntimeError("Frozen Stage4.2C evidence changed before Stage4.2D commit")

            running.unlink(missing_ok=True)
            if gate["passed"]:
                atomic_write_json(committed, {"run_id": run_id, "committed_utc": utc_now(), "mode": run_mode})
                if is_official and computational_certified:
                    metadata_path = run_dir / "metadata.json"
                    report_path = report_dir / "FINAL_REPORT.md"
                    quality_path = run_dir / "quality_gate.json"
                    pointer = output_root / "CURRENT_STAGE4_2D_EQUITY_CERTIFICATION_RUN.json"
                    atomic_write_json(
                        pointer,
                        {
                            "run_id": run_id,
                            "run_relative_path": str(run_dir.relative_to(root)),
                            "metadata_relative_path": str(metadata_path.relative_to(root)),
                            "metadata_sha256": sha256_file(metadata_path),
                            "inventory_relative_path": str(inventory_path.relative_to(root)),
                            "inventory_sha256": sha256_file(inventory_path),
                            "report_relative_path": str(report_path.relative_to(root)),
                            "report_sha256": sha256_file(report_path),
                            "quality_gate_relative_path": str(quality_path.relative_to(root)),
                            "quality_gate_sha256": sha256_file(quality_path),
                            "frozen_contract_relative_path": str((run_dir / "frozen_contract.json").relative_to(root)),
                            "frozen_contract_sha256": sha256_file(run_dir / "frozen_contract.json"),
                            "authoritative_inputs_relative_path": str(
                                (run_dir / "authoritative_inputs_before.json").relative_to(root)
                            ),
                            "authoritative_inputs_sha256": sha256_file(
                                run_dir / "authoritative_inputs_before.json"
                            ),
                            "decision": gate["decision"],
                            "scope": gate["promotable_scope"],
                            "stage4_2d_front_stages_certified": True,
                            "stage4_full_computational_complete": False,
                            "operational_final": False,
                            "stage5_started": False,
                            "stage5_release_allowed": False,
                            "created_utc": utc_now(),
                        },
                    )
            else:
                atomic_write_json(failed, {"run_id": run_id, "failed_utc": utc_now(), "decision": gate["decision"]})
                atomic_write_json(
                    output_root / "LATEST_FAILED_STAGE4_2D_EQUITY_RUN.json",
                    {"run_id": run_id, "run_dir": str(run_dir.relative_to(root)), "decision": gate["decision"]},
                )
            return {
                "run_id": run_id,
                "passed": bool(gate["passed"]),
                "decision": gate["decision"],
                "run_dir": str(run_dir),
                "report_dir": str(report_dir),
                "min_sigungu": _certificate_row(min_cert),
                "high_need": _certificate_row(high_cert) if high_cert is not None else None,
            }
        except Exception as exc:
            running.unlink(missing_ok=True)
            committed.unlink(missing_ok=True)
            atomic_write_json(
                failed,
                {
                    "run_id": run_id,
                    "failed_utc": utc_now(),
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                },
            )
            atomic_write_json(
                output_root / "LATEST_FAILED_STAGE4_2D_EQUITY_RUN.json",
                {"run_id": run_id, "run_dir": str(run_dir.relative_to(root)), "error": repr(exc)},
            )
            raise
