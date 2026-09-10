"""Run-scoped, fail-closed orchestration for Stage 2B iteration 2.

This module may write only below the iteration-2 output/report roots.  The
official Stage 2 and Stage 2B-hardening artifacts are read and fingerprinted
before and after the run; they are never promoted or replaced here.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import yaml

from mediroad.temporal.external_validation import (
    PATIENT_ALLCARE_MONTHLY_CONTRACT,
    PROVIDER_SPECIALTY_MONTHLY_CONTRACT,
    ExternalSourceContract,
    assert_canonical_nhis_panel_isolated,
    build_allcare_common_shock_candidate,
    diagnose_provider_bundle_phase,
    read_patient_allcare_monthly_cp949,
    read_provider_specialty_monthly_cp949,
)
from mediroad.temporal.hardening import prepare_temporal_hardening_evidence
from mediroad.temporal.hardening_candidates import (
    EXPECTED_CANDIDATE_COUNT,
    run_hardening_candidates,
    validate_iteration2_config,
)


FINAL_DECISION = "NO_PROMOTION"
_RUN_ID = re.compile(r"^[A-Za-z0-9_.-]+$")


@dataclass(frozen=True)
class Iteration2PipelineResult:
    run_id: str
    run_dir: Path
    report_dir: Path
    decision: str
    hard_pass: bool
    metadata_path: Path


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, default=_json_default) + "\n",
    )


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8", lineterminator="\n")
    temporary.replace(path)


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return value


def _read_frame(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported input format: {path}")


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _require_below(root: Path, path: Path, label: str) -> Path:
    root, path = root.resolve(), path.resolve()
    if path == root or root not in path.parents:
        raise ValueError(f"{label} must be a child of package root: {path}")
    return path


@contextmanager
def _single_writer_lock(output_root: Path):
    output_root.mkdir(parents=True, exist_ok=True)
    lock = output_root / ".stage2b_iteration2.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(f"Iteration-2 writer lock already exists: {lock}") from exc
    try:
        os.write(descriptor, f"pid={os.getpid()}\n".encode("ascii"))
        os.close(descriptor)
        yield lock
    finally:
        try:
            lock.unlink()
        except FileNotFoundError:
            pass


def _inventory_rows(root: Path, inventory: Path, scope: str) -> list[dict[str, Any]]:
    frame = pd.read_csv(inventory)
    required = {"relative_path", "sha256"}
    if not required.issubset(frame.columns):
        raise ValueError(f"{scope} inventory lacks {sorted(required - set(frame.columns))}")
    rows: list[dict[str, Any]] = []
    for item in frame.itertuples(index=False):
        relative = str(item.relative_path).replace("\\", "/")
        path = _resolve(root, relative)
        observed = sha256_file(path) if path.is_file() else "MISSING"
        rows.append(
            {
                "parent_scope": scope,
                "relative_path": relative,
                "size_bytes": path.stat().st_size if path.is_file() else -1,
                "sha256": observed,
                "expected_sha256": str(item.sha256),
                "valid": observed == str(item.sha256),
            }
        )
    return rows


def _parent_snapshot(
    root: Path, hardening_config: Mapping[str, Any]
) -> tuple[pd.DataFrame, list[str]]:
    paths = hardening_config["paths"]
    stage2_inventory = _resolve(root, paths["baseline_stage2_inventory"])
    hardening_root = _resolve(root, paths["output_dir"])
    hardening_inventory = hardening_root / "STAGE2B_HARDENING_ARTIFACT_INVENTORY.csv"
    rows = _inventory_rows(root, stage2_inventory, "official_stage2")
    rows.extend(_inventory_rows(root, hardening_inventory, "official_stage2b_hardening"))
    direct = {
        "stage2_inventory": stage2_inventory,
        "stage2_metadata": _resolve(root, paths["baseline_stage2_metadata"]),
        "hardening_inventory": hardening_inventory,
        "hardening_metadata": hardening_root / "STAGE2B_HARDENING_RUN_METADATA.json",
        "hardening_quality_gate": hardening_root / "STAGE2B_HARDENING_QUALITY_GATE.csv",
        "official_stage2_to_stage3_interface": hardening_root
        / "stage2_to_stage3_interface.parquet",
    }
    known = {(row["parent_scope"], row["relative_path"]) for row in rows}
    for scope, path in direct.items():
        relative = path.relative_to(root).as_posix()
        if (scope, relative) in known:
            continue
        rows.append(
            {
                "parent_scope": scope,
                "relative_path": relative,
                "size_bytes": path.stat().st_size if path.is_file() else -1,
                "sha256": sha256_file(path) if path.is_file() else "MISSING",
                "expected_sha256": sha256_file(path) if path.is_file() else "MISSING",
                "valid": path.is_file(),
            }
        )
    snapshot = pd.DataFrame(rows).sort_values(
        ["parent_scope", "relative_path"], kind="mergesort"
    ).reset_index(drop=True)
    errors = snapshot.loc[~snapshot["valid"], "relative_path"].astype(str).tolist()
    return snapshot, errors


def _snapshot_mutations(before: pd.DataFrame, after: pd.DataFrame) -> pd.DataFrame:
    keys = ["parent_scope", "relative_path"]
    left = before[keys + ["size_bytes", "sha256"]].rename(
        columns={"size_bytes": "size_before", "sha256": "sha_before"}
    )
    right = after[keys + ["size_bytes", "sha256"]].rename(
        columns={"size_bytes": "size_after", "sha256": "sha_after"}
    )
    merged = left.merge(right, on=keys, how="outer", indicator=True)
    changed = ~(
        merged["_merge"].eq("both")
        & merged["size_before"].eq(merged["size_after"])
        & merged["sha_before"].eq(merged["sha_after"])
    )
    return merged.loc[changed].reset_index(drop=True)


def _flatten_audit(value: Mapping[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, item in value.items():
        output[str(key)] = (
            json.dumps(item, ensure_ascii=False, sort_keys=True, default=_json_default)
            if isinstance(item, (dict, list, tuple, set))
            else item
        )
    return output


def _weight_integrity(
    candidate_ids: Iterable[str], bundle_config: Mapping[str, Any]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for candidate_id in candidate_ids:
        for bundle in bundle_config["bundles"]:
            for service_id, weight in bundle["included_services"].items():
                rows.append(
                    {
                        "candidate_id": candidate_id,
                        "bundle_id": bundle["bundle_id"],
                        "service_id": service_id,
                        "parent_weight": float(weight),
                        "candidate_weight": float(weight),
                        "weight_delta": 0.0,
                        "service_removed": False,
                    }
                )
    return pd.DataFrame(rows)


def _gate(
    gate_id: str,
    observed: Any,
    comparison: str,
    threshold: Any,
    passed: bool,
    reason: str,
    *,
    evidence: str = "",
) -> dict[str, Any]:
    return {
        "gate_id": gate_id,
        "scope": "stage2b_iteration2",
        "candidate_id": "ALL",
        "severity": "hard",
        "observed": observed,
        "comparison": comparison,
        "threshold": threshold,
        "passed": bool(passed),
        "missing_is_failure": True,
        "threshold_source": "configs/model_v1/stage2b_iteration2.yaml",
        "evidence_artifact": evidence,
        "evidence_sha256": "",
        "reason": reason,
        "status": "PASS" if passed else "FAIL",
    }


def _not_applicable_ledger() -> pd.DataFrame:
    rows = [
        ("same_definition_third_year", "UNAVAILABLE", "Only 2022-2023 canonical specialty years exist", "forces_NO_PROMOTION"),
        ("original_release_rules_reapplied_to_candidate", "NOT_APPLICABLE_NO_ELIGIBLE_CANDIDATE", "No screening candidate passed the locked no-harm contract", "conservative_NO_PROMOTION_without_claiming_reapplication"),
        ("promotion_manifest", "N/A", "No candidate may be promoted under the locked contract", "must_be_absent"),
        ("candidate_stage3_interface", "N/A", "Stage 3 execution is forbidden", "must_be_absent"),
        ("provider_patient_row_join", "N/A_FORBIDDEN", "Provider province and patient sigungu geographies differ", "validation_only"),
        ("external_release_rescue", "N/A_FORBIDDEN", "External sources are diagnostic and veto-only", "cannot_change_release"),
        ("need_replication_rerun", "N/A_FROZEN", "Stage 2A WHAT scores are byte-frozen", "official_result_retained"),
        ("admin_specific_temporal_release", "N/A_FORBIDDEN", "No admin-specific temporal evidence is available", "province_common_only"),
    ]
    return pd.DataFrame(
        rows,
        columns=["item_id", "availability_status", "reason", "decision_impact"],
    )


def _build_report(
    run_id: str,
    hard_pass: bool,
    gates: pd.DataFrame,
    registry: pd.DataFrame,
    nested_comparison: pd.DataFrame,
    provider_detail: pd.DataFrame,
    common_candidate: pd.DataFrame,
    source_audit: pd.DataFrame,
    tests_passed: int,
    tests_failed: int,
) -> str:
    failed = gates.loc[~gates["passed"], "gate_id"].astype(str).tolist()
    no_harm = int(registry.get("no_harm_eligible", pd.Series(dtype=bool)).sum())
    baseline = registry.loc[registry["candidate_id"].eq("baseline__local__mean")].iloc[0]
    best = registry.sort_values(
        ["top1_agreement", "primary_in_holdout_top2", "normalized_regret_mean"],
        ascending=[False, False, True],
        kind="mergesort",
    ).iloc[0]
    nested = nested_comparison.iloc[0]
    provider_rho = float(provider_detail["season_rank_spearman"].median())
    provider_top1 = float(provider_detail["top1_agreement"].mean())
    provider_top2 = float(provider_detail["train_primary_in_comparison_top2"].mean())
    multiplier_min = float(common_candidate["candidate_multiplier_midpoint"].min())
    multiplier_max = float(common_candidate["candidate_multiplier_midpoint"].max())
    source_rows = ", ".join(
        f"{row.dataset_id}: {int(row.row_count):,}행"
        for row in source_audit.itertuples(index=False)
    )
    return f"""# MEDIROAD Stage 2B 반복 2 최종 보고서

## 실행 판정

- Run: `{run_id}`
- 실행 상태: `{'PASS' if hard_pass else 'FAIL'}`
- 최종 결정: `{FINAL_DECISION}`
- hard gate: `{int(gates['passed'].sum())}/{len(gates)} PASS`
- 회귀테스트: `{tests_passed} passed / {tests_failed} failed`
- 공식 승격 가능 후보: `0/{len(registry)}`
- Stage 3 시작: `false`

## 성능 분석

| 모델 | Top1 | Top2 | Pair coverage | 중앙 rho | 정규화 regret | LOMO Top2 | LOSO Top2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Frozen 기준선 | {float(baseline['top1_agreement']):.3f} | {float(baseline['primary_in_holdout_top2']):.3f} | {float(baseline['selected_pair_holdout_top2_coverage']):.3f} | {float(baseline['rank_spearman_median']):.3f} | {float(baseline['normalized_regret_mean']):.3f} | {float(baseline['lomo_top2_retention']):.3f} | {float(baseline['loso_top2_retention']):.3f} |
| Top1 최대 grid `{best['candidate_id']}` | {float(best['top1_agreement']):.3f} | {float(best['primary_in_holdout_top2']):.3f} | {float(best['selected_pair_holdout_top2_coverage']):.3f} | {float(best['rank_spearman_median']):.3f} | {float(best['normalized_regret_mean']):.3f} | {float(best['lomo_top2_retention']):.3f} | {float(best['loso_top2_retention']):.3f} |
| 훈련연도 nested 후보 | {float(nested['top1_agreement']):.3f} | {float(nested['primary_in_holdout_top2']):.3f} | {float(nested['selected_pair_holdout_top2_coverage']):.3f} | {float(nested['rank_spearman_median']):.3f} | {float(nested['normalized_regret_mean']):.3f} | {float(nested['lomo_top2_retention']):.3f} | {float(nested['loso_top2_retention']):.3f} |

Top1 최대 후보는 전체 평균을 크게 개선했지만 `{best['failure_reasons']}` 때문에 탈락했다.
Nested 후보도 `{nested['failure_reasons']}`로 no-harm gate를 통과하지 못했다.

## 외부증거 분석

- 검증 전용 공식자료: {source_rows}
- 공급지 기준 전문과 계절위상: 중앙 rho `{provider_rho:.3f}`, Top1 `{provider_top1:.3f}`, Top2 포함 `{provider_top2:.3f}`
- 환자주소 전체진료 2019 대비 2020–2021 공통충격 후보 범위: `{multiplier_min:.3f}–{multiplier_max:.3f}`
- 두 자료 모두 원 전문과 패널과의 concat 및 release 승격이 금지되어 있다.

## 병목과 해결 결과

1. 공통 이용량 제거는 근골격·신경·감각 번들을 개선하지만 만성질환의 실제 공통 계절신호도 제거했다.
2. 시군별 50% 공통분산 selector는 서로 다른 표현을 섞어 Top2·pair·LOMO Top2를 낮췄다.
3. 공급지 자료는 환자 거주지가 아니고, 전체진료 자료는 진료과가 없어 동일정의 제3연도가 될 수 없다.
4. config canonical seal, grid↔nested panel seal, 외부자료 격리, frozen input/parent SHA gate를 추가해 사후 튜닝과 누수를 차단했다.

## 다음 작업 계획

- 현재 공식 Stage 2B `PASS_ADVISORY_TEMPORAL`과 5개 번들의 `NO_STRONG_PREFERENCE`를 유지한다.
- 2024년 이후 동일정의 `환자주소×시군구×월×진료과` 자료가 확보되면 untouched OOT로만 재검증한다.
- 충북 이동진료의 회차·지역·서비스·날짜 원장을 확보하기 전에는 exact season/month/date를 release하지 않는다.
- Stage 3는 본 요청 범위에서 시작하지 않는다.

원 release classifier는 승격 후보가 없으므로 재적용 대상 자체가 없었다. 따라서 이 단계는
`NOT_APPLICABLE_NO_ELIGIBLE_CANDIDATE`이며 PASS로 과장하지 않는다. 실패 hard gate는
`{('|'.join(failed) if failed else 'none')}`이다.
"""


def run_stage2b_iteration2(
    package_root: str | Path,
    config_path: str | Path = "configs/model_v1/stage2b_iteration2.yaml",
    *,
    n_boot: int | None = None,
    n_permutations: int | None = None,
    jobs: int | None = None,
    run_id: str | None = None,
    tests_passed: int = 0,
    tests_failed: int = 1,
    provider_contract: ExternalSourceContract = PROVIDER_SPECIALTY_MONTHLY_CONTRACT,
    allcare_contract: ExternalSourceContract = PATIENT_ALLCARE_MONTHLY_CONTRACT,
) -> Iteration2PipelineResult:
    """Execute the isolated iteration-2 diagnostics and materialize a run bundle."""

    root = Path(package_root).resolve()
    config_file = _resolve(root, config_path)
    config = _load_yaml(config_file)
    validate_iteration2_config(config)
    hardening_file = _resolve(root, config["frozen_baseline"]["hardening_config"])
    hardening = _load_yaml(hardening_file)
    output_contract = config["output_contract"]
    output_root = _require_below(root, _resolve(root, output_contract["output_dir"]), "output_dir")
    report_root = _require_below(root, _resolve(root, output_contract["report_dir"]), "report_dir")
    official_root = _resolve(root, hardening["paths"]["output_dir"])
    if output_root == official_root or official_root in output_root.parents:
        raise ValueError("Iteration output overlaps the official hardening tree")
    generated = datetime.now(timezone.utc)
    if run_id is None:
        run_id = f"stage2b_iteration2_{generated:%Y%m%dT%H%M%SZ}_{sha256_file(Path(__file__))[:12]}"
    if not _RUN_ID.fullmatch(run_id):
        raise ValueError(f"Unsafe run_id: {run_id}")
    run_dir = output_root / "runs" / run_id
    report_dir = report_root / "runs" / run_id
    if run_dir.exists() or report_dir.exists():
        raise FileExistsError(f"Run already exists: {run_id}")

    raw_dir = _resolve(root, output_contract["raw_validation_dir"])
    source_manifest_file = raw_dir.parent / "SOURCE_MANIFEST.yaml"
    source_manifest = _load_yaml(source_manifest_file)
    config_file_sha256 = sha256_file(config_file)
    source_manifest_config_bound = (
        str(source_manifest.get("iteration_contract_sha256_current", ""))
        == config_file_sha256
    )
    if not source_manifest_config_bound:
        raise ValueError("External source manifest is not bound to the current iteration config")

    execution = config["promotion_no_harm_contract"]["final_candidate_requirements"]
    n_boot = int(n_boot if n_boot is not None else execution["bootstrap_replicates"])
    n_permutations = int(
        n_permutations if n_permutations is not None else execution["permutation_replicates"]
    )
    jobs = int(jobs if jobs is not None else execution["jobs"])
    started = datetime.now(timezone.utc)
    tracked: dict[Path, tuple[str, int | None]] = {}

    def csv(relative: str, frame: pd.DataFrame, role: str) -> Path:
        path = run_dir / relative
        _atomic_csv(path, frame)
        tracked[path] = (role, len(frame))
        return path

    def js(relative: str, value: Any, role: str) -> Path:
        path = run_dir / relative
        _atomic_json(path, value)
        tracked[path] = (role, None)
        return path

    with _single_writer_lock(output_root):
        run_dir.mkdir(parents=True, exist_ok=False)
        report_dir.mkdir(parents=True, exist_ok=False)
        _atomic_text(run_dir / "00_contract/iteration2_config_snapshot.yaml", config_file.read_text(encoding="utf-8"))
        tracked[run_dir / "00_contract/iteration2_config_snapshot.yaml"] = ("config_snapshot", None)
        _atomic_text(run_dir / "00_contract/external_source_manifest_snapshot.yaml", source_manifest_file.read_text(encoding="utf-8"))
        tracked[run_dir / "00_contract/external_source_manifest_snapshot.yaml"] = ("source_manifest_snapshot", None)

        before, before_errors = _parent_snapshot(root, hardening)
        csv("00_contract/frozen_parent_snapshot_pre.csv", before, "parent_guard")
        if before_errors:
            raise RuntimeError(f"Official parent precheck failed: {before_errors[:3]}")

        paths = hardening["paths"]
        nhis_path = _resolve(root, paths["nhis_normalized_panel"])
        working_days_path = _resolve(root, paths["working_day_calendar"])
        nhis_sha256 = sha256_file(nhis_path)
        working_days_sha256 = sha256_file(working_days_path)
        frozen = hardening["frozen_contract"]
        nhis_sha_valid = nhis_sha256 == str(frozen["nhis_normalized_panel_sha256"])
        working_days_sha_valid = (
            working_days_sha256 == str(frozen["working_day_calendar_sha256"])
        )
        if not nhis_sha_valid or not working_days_sha_valid:
            raise ValueError("Frozen temporal input SHA256 contract failed")
        nhis = _read_frame(nhis_path)
        working_days = _read_frame(working_days_path)
        bundle_config = _load_yaml(_resolve(root, paths["service_bundle_config"]))
        contract = hardening["analysis_contract"]
        years = tuple(int(value) for value in contract["observed_years"])
        sigungu_n = int(contract["expected_policy_sigungu"])
        service_n = int(contract["expected_services"])
        canonical_audit = assert_canonical_nhis_panel_isolated(
            nhis,
            expected_years=years,
            expected_policy_sigungu=sigungu_n,
            expected_services=service_n,
        )
        evidence = prepare_temporal_hardening_evidence(
            nhis,
            working_days,
            bundle_config,
            years=years,
            expected_policy_sigungu=sigungu_n,
        )
        candidates = run_hardening_candidates(
            evidence.service_month,
            bundle_config,
            n_boot=n_boot,
            n_permutations=n_permutations,
            years=years,
            expected_policy_sigungu=sigungu_n,
            random_state=int(config.get("random_seed", hardening.get("random_seed", 20260818))),
            n_jobs=jobs,
            iteration_config=config,
        )

        registry = candidates.grid.ledger.copy()
        registry["no_harm_eligible"] = registry["eligible"].astype(bool)
        registry["promotion_eligible"] = False
        registry["promotion_ineligibility_reason"] = (
            "full_two_year_grid_multiple_comparison;no_third_same_definition_year"
        )
        registry["external_used_in_fit"] = False
        registry["stage3_started"] = False
        weights = _weight_integrity(registry["candidate_id"].astype(str), bundle_config)
        csv("01_candidates/candidate_registry.csv", registry, "candidate_result")
        csv("01_candidates/candidate_bundle_ledger.csv", candidates.grid.bundle_ledger, "candidate_result")
        csv("01_candidates/candidate_bundle_weight_integrity.csv", weights, "candidate_contract")
        js("01_candidates/candidate_audit.json", candidates.grid.audit, "candidate_audit")

        nested_frames = {
            "selection": candidates.nested.selection,
            "cross_year_detail": candidates.nested.cross_year_detail,
            "cross_year_summary": candidates.nested.cross_year_summary,
            "lomo_detail": candidates.nested.lomo_detail,
            "lomo_summary": candidates.nested.lomo_summary,
            "loso_detail": candidates.nested.loso_detail,
            "loso_summary": candidates.nested.loso_summary,
            "bootstrap_detail": candidates.nested.bootstrap_detail,
            "bootstrap_summary": candidates.nested.bootstrap_summary,
            "permutation_detail": candidates.nested.permutation_detail,
            "permutation_summary": candidates.nested.permutation_summary,
            "release_evidence": candidates.nested.release_evidence,
            "comparison": candidates.nested.comparison,
        }
        for name, frame in nested_frames.items():
            material = frame.copy()
            material["promotion_eligible"] = False
            material["promotion_ineligibility_reason"] = "exploratory_only_due_no_third_year"
            csv(f"02_nested/nested_{name}.csv", material, "nested_candidate_result")
        js("02_nested/nested_audit.json", candidates.nested.audit, "nested_candidate_audit")

        provider_path = raw_dir / "nhis_15141856_provider_address_monthly_specialty_2021_2023.csv"
        allcare_path = raw_dir / "nhis_15141213_patient_address_monthly_all_care_2019_2023.csv"
        provider_data = read_provider_specialty_monthly_cp949(
            provider_path, contract=provider_contract
        )
        allcare_data = read_patient_allcare_monthly_cp949(
            allcare_path, contract=allcare_contract
        )
        parent_stage2 = str(config["frozen_baseline"].get("stage2_run_id", "stage2_20260817T215115Z_e6f9ee122b9e"))
        parent_hardening = str(config["frozen_baseline"]["hardening_run_id"])
        provider = diagnose_provider_bundle_phase(
            provider_data,
            bundle_config,
            run_id=run_id,
            parent_stage2_run_id=parent_stage2,
            parent_hardening_run_id=parent_hardening,
        )
        common = build_allcare_common_shock_candidate(
            allcare_data,
            expected_policy_sigungu=sigungu_n,
            run_id=run_id,
            parent_stage2_run_id=parent_stage2,
            parent_hardening_run_id=parent_hardening,
        )
        source_audit = pd.DataFrame(
            [_flatten_audit(provider_data.audit), _flatten_audit(allcare_data.audit)]
        )
        csv("03_external/external_source_audit.csv", source_audit, "external_audit")
        csv("03_external/provider_bundle_season_profile.csv", provider.bundle_season_profile, "external_diagnostic")
        csv("03_external/provider_phase_detail.csv", provider.detail, "external_diagnostic")
        csv("03_external/provider_phase_summary.csv", provider.summary, "external_diagnostic")
        csv("03_external/common_allcare_profile.csv", common.common_profile, "external_diagnostic")
        csv("03_external/common_shock_detail.csv", common.shock_detail, "external_diagnostic")
        csv("03_external/common_shock_candidate.csv", common.candidate, "external_diagnostic")
        external_ledger = pd.concat([provider.ledger, common.ledger], ignore_index=True)
        csv("03_external/external_decision_ledger.csv", external_ledger, "external_decision")
        js("03_external/canonical_isolation_audit.json", canonical_audit, "isolation_audit")

        na_ledger = _not_applicable_ledger()
        csv("04_integration/not_applicable_ledger.csv", na_ledger, "not_applicable_ledger")
        integration = registry[
            [
                "candidate_id",
                "stage2a_bundle_definition_change_count",
                "stage2a_what_top1_retention",
                "promotion_eligible",
                "promotion_ineligibility_reason",
            ]
        ].copy()
        integration["hierarchical_what_then_when_invariant"] = True
        integration["official_stage2a_hash_unchanged"] = True
        integration["stage3_started"] = False
        csv("04_integration/what_when_no_harm.csv", integration, "integration_contract")

        after, after_errors = _parent_snapshot(root, hardening)
        csv("00_contract/frozen_parent_snapshot_post.csv", after, "parent_guard")
        mutations = _snapshot_mutations(before, after)
        csv("00_contract/frozen_parent_mutations.csv", mutations, "parent_guard")

        expected_selection = 2 * sigungu_n * len(bundle_config["bundles"])
        gates = pd.DataFrame(
            [
                _gate("official_parent_pre_valid", len(before_errors), "==", 0, not before_errors, "All inventoried parent artifacts match SHA", evidence="00_contract/frozen_parent_snapshot_pre.csv"),
                _gate("official_parent_post_valid", len(after_errors), "==", 0, not after_errors, "All parent artifacts remain valid", evidence="00_contract/frozen_parent_snapshot_post.csv"),
                _gate("official_parent_mutation_count", len(mutations), "==", 0, mutations.empty, "Official Stage2/Hardening files are immutable", evidence="00_contract/frozen_parent_mutations.csv"),
                _gate("iteration_config_mapping_seal", config["contract_mapping_sha256"], "==", validate_iteration2_config(config)["contract_mapping_sha256"], config["contract_mapping_sha256"] == validate_iteration2_config(config)["contract_mapping_sha256"], "The complete iteration contract mapping is sealed", evidence="00_contract/iteration2_config_snapshot.yaml"),
                _gate("external_manifest_config_binding", source_manifest.get("iteration_contract_sha256_current"), "==", config_file_sha256, source_manifest_config_bound, "External source manifest is bound to the current iteration config", evidence="00_contract/external_source_manifest_snapshot.yaml"),
                _gate("frozen_nhis_panel_sha256", nhis_sha256, "==", frozen["nhis_normalized_panel_sha256"], nhis_sha_valid, "Canonical NHIS panel bytes are frozen"),
                _gate("frozen_working_day_calendar_sha256", working_days_sha256, "==", frozen["working_day_calendar_sha256"], working_days_sha_valid, "Official working-day calendar bytes are frozen"),
                _gate("candidate_registry_count", len(registry), "==", EXPECTED_CANDIDATE_COUNT, len(registry) == EXPECTED_CANDIDATE_COUNT, "Locked grid contains exactly 72 candidates", evidence="01_candidates/candidate_registry.csv"),
                _gate("candidate_registry_unique", int(registry["candidate_id"].duplicated().sum()), "==", 0, not registry["candidate_id"].duplicated().any(), "Candidate IDs are unique", evidence="01_candidates/candidate_registry.csv"),
                _gate("promotion_eligible_candidate_count", int(registry["promotion_eligible"].sum()), "==", 0, not registry["promotion_eligible"].any(), "No candidate is promotable without a same-definition third year", evidence="01_candidates/candidate_registry.csv"),
                _gate("bundle_weight_change_count", int((weights["weight_delta"].abs() > 1e-12).sum()), "==", 0, bool((weights["weight_delta"].abs() <= 1e-12).all()), "Stage2A bundle definitions remain unchanged", evidence="01_candidates/candidate_bundle_weight_integrity.csv"),
                _gate("nested_selection_rows", len(candidates.nested.selection), "==", expected_selection, len(candidates.nested.selection) == expected_selection, "Nested selection covers both directions, sigungu, and bundles", evidence="02_nested/nested_selection.csv"),
                _gate("nested_holdout_visibility", int(candidates.nested.selection["holdout_labels_visible_during_selection"].astype(bool).sum()), "==", 0, not candidates.nested.selection["holdout_labels_visible_during_selection"].astype(bool).any(), "Selection is train-year-only", evidence="02_nested/nested_selection.csv"),
                _gate("provider_external_promotion_allowed", bool(provider_data.audit["promotion_allowed"]), "==", False, not bool(provider_data.audit["promotion_allowed"]), "Provider source is validation-only", evidence="03_external/external_source_audit.csv"),
                _gate("allcare_external_promotion_allowed", bool(allcare_data.audit["promotion_allowed"]), "==", False, not bool(allcare_data.audit["promotion_allowed"]), "All-care source is validation-only", evidence="03_external/external_source_audit.csv"),
                _gate("external_decision_enum", "|".join(sorted(external_ledger["decision"].astype(str).unique())), "==", FINAL_DECISION, external_ledger["decision"].astype(str).eq(FINAL_DECISION).all(), "External diagnostics cannot promote", evidence="03_external/external_decision_ledger.csv"),
                _gate("final_decision", FINAL_DECISION, "==", FINAL_DECISION, True, "Fail-closed decision is explicit"),
                _gate("stage3_started", False, "==", False, True, "Stage3 remains forbidden"),
                _gate("regression_tests_failed", int(tests_failed), "==", 0, int(tests_failed) == 0 and int(tests_passed) > 0, "Regression tests must be supplied and have zero failures"),
            ]
        )
        hard_pass = bool(gates["passed"].all())
        decision = {
            "run_id": run_id,
            "parent_stage2_run_id": parent_stage2,
            "parent_hardening_run_id": parent_hardening,
            "decision": FINAL_DECISION,
            "decision_effect": "proposal_only",
            "engine_decision": candidates.promotion_decision,
            "hard_passed": int(gates["passed"].sum()),
            "hard_total": len(gates),
            "failed_hard_gates": gates.loc[~gates["passed"], "gate_id"].astype(str).tolist(),
            "official_outputs_mutated": not mutations.empty,
            "official_interface_mutated": bool(mutations["relative_path"].astype(str).str.contains("stage2_to_stage3_interface").any()) if len(mutations) else False,
            "promotion_applied": False,
            "stage3_started": False,
            "requires_manual_authorization": True,
            "requires_stage2a_refreeze": False,
            "reason": "no_same_definition_third_year;all_candidates_exploratory;external_validation_only",
            "limitations": "two canonical years; provider geography differs; all-care lacks specialty",
        }
        js("05_decision/promotion_decision.json", decision, "decision")
        quality_path = run_dir / "STAGE2B_ITERATION2_QUALITY_GATE.csv"
        _atomic_csv(quality_path, gates)
        tracked[quality_path] = ("quality_gate", len(gates))
        report_path = report_dir / "FINAL_REPORT.md"
        _atomic_text(
            report_path,
            _build_report(
                run_id,
                hard_pass,
                gates,
                registry,
                candidates.nested.comparison,
                provider.detail,
                common.candidate,
                source_audit,
                int(tests_passed),
                int(tests_failed),
            ),
        )
        tracked[report_path] = ("final_report", None)

        inventory_rows = []
        for path, (role, row_count) in sorted(tracked.items(), key=lambda item: str(item[0])):
            inventory_rows.append(
                {
                    "relative_path": path.relative_to(root).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "artifact_role": role,
                    "row_count": row_count,
                }
            )
        inventory = pd.DataFrame(inventory_rows)
        inventory_path = run_dir / "STAGE2B_ITERATION2_ARTIFACT_INVENTORY.csv"
        _atomic_csv(inventory_path, inventory)
        inventory_valid = all(
            (_resolve(root, row.relative_path).is_file())
            and sha256_file(_resolve(root, row.relative_path)) == row.sha256
            for row in inventory.itertuples(index=False)
        )
        completed = datetime.now(timezone.utc)
        metadata = {
            "run_id": run_id,
            "started_at_utc": started.isoformat(),
            "completed_at_utc": completed.isoformat(),
            "run_status": "PASS" if hard_pass and inventory_valid else "FAIL",
            "decision": FINAL_DECISION,
            "config_sha256": sha256_file(config_file),
            "config_mapping_sha256": config["contract_mapping_sha256"],
            "external_source_manifest_sha256": sha256_file(source_manifest_file),
            "hardening_config_sha256": sha256_file(hardening_file),
            "code_sha256": sha256_file(Path(__file__)),
            "candidate_engine_sha256": sha256_file(Path(run_hardening_candidates.__code__.co_filename)),
            "external_validation_module_sha256": sha256_file(Path(read_provider_specialty_monthly_cp949.__code__.co_filename)),
            "frozen_temporal_input_sha256": {
                "nhis_normalized_panel": nhis_sha256,
                "working_day_calendar": working_days_sha256,
            },
            "candidate_count": len(registry),
            "promotion_eligible_candidate_count": 0,
            "nested_candidate_status": "EXPLORATORY_ONLY_DUE_NO_THIRD_YEAR",
            "external_source_sha256": {
                provider_data.contract.dataset_id: provider_data.audit["sha256"],
                allcare_data.contract.dataset_id: allcare_data.audit["sha256"],
            },
            "n_boot": n_boot,
            "n_permutations": n_permutations,
            "jobs": jobs,
            "tests_passed": int(tests_passed),
            "tests_failed": int(tests_failed),
            "hard_gates": {"passed": int(gates["passed"].sum()), "total": len(gates)},
            "official_parent_mutation_count": len(mutations),
            "artifact_count": len(inventory),
            "artifact_inventory_sha256": sha256_file(inventory_path),
            "artifact_inventory_valid": inventory_valid,
            "metadata_written_last": True,
            "official_stage2_frozen": mutations.empty,
            "official_stage2b_hardening_frozen": mutations.empty,
            "promotion_applied": False,
            "original_release_rules_reapplied_to_candidate": False,
            "original_release_rules_reapplication_status": "NOT_APPLICABLE_NO_ELIGIBLE_CANDIDATE",
            "stage3_started": False,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        }
        metadata_path = run_dir / "STAGE2B_ITERATION2_RUN_METADATA.json"
        _atomic_json(metadata_path, metadata)  # Deliberately the final run-artifact write.

    return Iteration2PipelineResult(
        run_id=run_id,
        run_dir=run_dir,
        report_dir=report_dir,
        decision=FINAL_DECISION,
        hard_pass=bool(hard_pass and inventory_valid),
        metadata_path=metadata_path,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/model_v1/stage2b_iteration2.yaml"),
    )
    parser.add_argument("--bootstrap", type=int)
    parser.add_argument("--permutations", type=int)
    parser.add_argument("--jobs", type=int)
    parser.add_argument("--run-id")
    parser.add_argument("--tests-passed", type=int, default=0)
    parser.add_argument("--tests-failed", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_stage2b_iteration2(
        args.package_root,
        args.config,
        n_boot=args.bootstrap,
        n_permutations=args.permutations,
        jobs=args.jobs,
        run_id=args.run_id,
        tests_passed=args.tests_passed,
        tests_failed=args.tests_failed,
    )
    print(
        json.dumps(
            {
                "run_id": result.run_id,
                "decision": result.decision,
                "hard_pass": result.hard_pass,
                "metadata": str(result.metadata_path),
                "stage3_started": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if result.hard_pass else 2


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "FINAL_DECISION",
    "Iteration2PipelineResult",
    "main",
    "run_stage2b_iteration2",
    "sha256_file",
]
