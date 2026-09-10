"""Fail-closed, validation-only NHIS evidence for Stage 2B.

The two sources in this module deliberately do not share the canonical
``patient-sigungu x specialty x month`` contract.  They may diagnose temporal
phase or a possible common shock, but they must never be appended to the
canonical NHIS panel and can never promote a Stage 2B release automatically.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping, NoReturn, Sequence

import numpy as np
import pandas as pd

from .seasonality import CHEONGJU_SOURCE_UNITS, POLICY_SIGUNGU, SERVICE_NAME_TO_ID


VALIDATION_ONLY_ROLE = "validation_only"
NO_PROMOTION_DECISION = "NO_PROMOTION"
CANONICAL_NHIS_COLUMNS = (
    "year",
    "month",
    "policy_sigungu_name",
    "service_id",
    "persons",
    "visits",
)
SEASON_MONTHS: dict[str, tuple[int, ...]] = {
    "winter": (12, 1, 2),
    "spring": (3, 4, 5),
    "summer": (6, 7, 8),
    "autumn": (9, 10, 11),
}
SEASON_ORDER = tuple(SEASON_MONTHS)
NO_PROMOTION_LEDGER_COLUMNS = (
    "run_id",
    "parent_stage2_run_id",
    "parent_hardening_run_id",
    "candidate_id",
    "decision",
    "decision_effect",
    "hard_passed",
    "hard_total",
    "failed_hard_gates",
    "official_outputs_mutated",
    "official_interface_mutated",
    "promotion_applied",
    "stage3_started",
    "requires_manual_authorization",
    "requires_stage2a_refreeze",
    "reason",
    "limitations",
)


@dataclass(frozen=True)
class ExternalSourceContract:
    """Immutable byte, schema, and semantic contract for one official file."""

    dataset_id: str
    kind: str
    source_page: str
    direct_download_url: str
    expected_sha256: str
    expected_rows: int
    expected_duplicate_dimensional_key_count: int
    raw_columns: tuple[str, ...]
    key_columns: tuple[str, ...]
    metric_columns: tuple[tuple[str, str], ...]
    suppression_allowed_metrics: tuple[str, ...]
    expected_years: tuple[int, ...]
    geography_basis: str
    expected_duplicate_geographies: tuple[str, ...] = ()
    license_name: str = "이용허락범위 제한 없음"
    encoding: str = "cp949"


PROVIDER_SPECIALTY_MONTHLY_CONTRACT = ExternalSourceContract(
    dataset_id="15141856",
    kind="provider_specialty_monthly",
    source_page="https://www.data.go.kr/data/15141856/fileData.do",
    direct_download_url=(
        "https://www.data.go.kr/cmm/cmm/fileDownload.do?"
        "atchFileId=FILE_000000003098435&fileDetailSn=1&insertDataPrcus=N"
    ),
    expected_sha256="f2b68da6d9ed3782d84df3965c0b2a6a3ceeb1a73861603ef3b3d30721dcc069",
    expected_rows=194_951,
    expected_duplicate_dimensional_key_count=0,
    raw_columns=(
        "진료년월",
        "요양기관종별",
        "진료과목코드",
        "진료과목명",
        "진료형태",
        "요양기관 주소지",
        "진료인원(명)",
        "진료건수(건)",
    ),
    key_columns=(
        "진료년월",
        "요양기관종별",
        "진료과목코드",
        "진료과목명",
        "진료형태",
        "요양기관 주소지",
    ),
    metric_columns=(("persons", "진료인원(명)"), ("visits", "진료건수(건)")),
    suppression_allowed_metrics=("persons", "visits"),
    expected_years=(2021, 2022, 2023),
    geography_basis="provider_address_province",
)


PATIENT_ALLCARE_MONTHLY_CONTRACT = ExternalSourceContract(
    dataset_id="15141213",
    kind="patient_allcare_monthly",
    source_page="https://www.data.go.kr/data/15141213/fileData.do",
    direct_download_url=(
        "https://www.data.go.kr/cmm/cmm/fileDownload.do?"
        "atchFileId=FILE_000000003090612&fileDetailSn=1&insertDataPrcus=N"
    ),
    expected_sha256="e78a93d6240454b0f6cdd084be8044a46a74a499935b5c83362a770aad8925cd",
    expected_rows=228_207,
    expected_duplicate_dimensional_key_count=52,
    raw_columns=(
        "진료년도",
        "진료월",
        "요양기관종별",
        "시도",
        "시군구",
        "진료건수(건)",
        "진료비(천원)",
    ),
    key_columns=("진료년도", "진료월", "요양기관종별", "시도", "시군구"),
    metric_columns=(("visits", "진료건수(건)"), ("cost_thousand_krw", "진료비(천원)")),
    suppression_allowed_metrics=("visits",),
    expected_years=(2019, 2020, 2021, 2022, 2023),
    geography_basis="patient_address_sigungu",
    expected_duplicate_geographies=("경기도",),
)


@dataclass(frozen=True)
class ValidatedExternalData:
    """Normalized validation-only rows plus their immutable source audit."""

    contract: ExternalSourceContract
    frame: pd.DataFrame
    audit: dict[str, Any]


@dataclass(frozen=True)
class ProviderPhaseDiagnosticResult:
    """Provider-side 2021 -> 2022/2023 seasonal phase diagnostic."""

    bundle_season_profile: pd.DataFrame
    detail: pd.DataFrame
    summary: pd.DataFrame
    ledger: pd.DataFrame


@dataclass(frozen=True)
class CommonShockCandidateResult:
    """Unpromoted 2019-baseline common-shock adjustment candidate."""

    common_profile: pd.DataFrame
    shock_detail: pd.DataFrame
    candidate: pd.DataFrame
    ledger: pd.DataFrame


class ExternalValidationIsolationError(RuntimeError):
    """Raised when validation-only evidence is offered for canonical concat."""


def _file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_contract(contract: ExternalSourceContract, expected_kind: str) -> None:
    if contract.kind != expected_kind:
        raise ValueError(f"Expected source kind {expected_kind}, got {contract.kind}")
    if contract.encoding.lower().replace("-", "") not in {"cp949", "ms949"}:
        raise ValueError("Official external-validation contracts require CP949")
    if contract.expected_rows <= 0:
        raise ValueError("Expected source row count must be positive")
    if contract.expected_duplicate_dimensional_key_count < 0:
        raise ValueError("Expected duplicate dimensional-key count cannot be negative")
    digest = contract.expected_sha256.lower()
    if len(digest) != 64 or any(value not in "0123456789abcdef" for value in digest):
        raise ValueError("Expected SHA256 must be a 64-character hexadecimal digest")
    if len(set(contract.raw_columns)) != len(contract.raw_columns):
        raise ValueError("Source contract contains duplicate column names")
    if not set(contract.key_columns).issubset(contract.raw_columns):
        raise ValueError("Source key columns are not a subset of raw columns")


def _parse_metric(
    series: pd.Series,
    *,
    metric: str,
    allow_suppression: bool,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    text = series.astype("string").str.strip()
    suppressed = text.eq("*")
    if suppressed.any() and not allow_suppression:
        raise ValueError(f"Metric {metric} contains unexpected '*' suppression")
    cleaned = text.str.replace(",", "", regex=False).mask(suppressed)
    values = pd.to_numeric(cleaned, errors="coerce").astype(float)
    invalid = values.isna() & ~suppressed
    if invalid.any():
        examples = sorted(text.loc[invalid].drop_duplicates().astype(str).tolist())[:5]
        raise ValueError(f"Metric {metric} contains non-numeric values: {examples}")
    if (values.dropna() < 0).any():
        raise ValueError(f"Metric {metric} contains negative values")
    lower = values.fillna(0.0)
    upper = values.fillna(4.0)
    return values, lower, upper, suppressed.astype(bool)


def _validate_source_frame(
    path: str | Path,
    contract: ExternalSourceContract,
    *,
    expected_kind: str,
    encoding: str,
) -> tuple[pd.DataFrame, str, int, int]:
    _validate_contract(contract, expected_kind)
    if encoding.lower().replace("-", "") not in {"cp949", "ms949"}:
        raise ValueError("External NHIS files must be decoded explicitly as CP949")
    source_path = Path(path)
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    actual_sha = _file_sha256(source_path)
    if actual_sha != contract.expected_sha256.lower():
        raise ValueError(
            f"{contract.dataset_id} SHA256 mismatch: "
            f"expected={contract.expected_sha256.lower()}, observed={actual_sha}"
        )
    frame = pd.read_csv(
        source_path,
        encoding=encoding,
        dtype="string",
        keep_default_na=False,
    )
    if tuple(frame.columns) != contract.raw_columns:
        raise KeyError(
            f"{contract.dataset_id} schema mismatch: "
            f"expected={list(contract.raw_columns)}, observed={list(frame.columns)}"
        )
    if len(frame) != contract.expected_rows:
        raise ValueError(
            f"{contract.dataset_id} row-count mismatch: "
            f"expected={contract.expected_rows}, observed={len(frame)}"
        )
    blank_counts = {
        column: int(frame[column].astype("string").str.strip().eq("").sum())
        for column in contract.raw_columns
    }
    blank_counts = {column: count for column, count in blank_counts.items() if count}
    if blank_counts:
        raise ValueError(f"{contract.dataset_id} contains blank cells: {blank_counts}")
    duplicate_count = int(frame.duplicated(list(contract.key_columns)).sum())
    if duplicate_count != contract.expected_duplicate_dimensional_key_count:
        raise ValueError(
            f"{contract.dataset_id} duplicate dimensional-key count mismatch: "
            f"expected={contract.expected_duplicate_dimensional_key_count}, "
            f"observed={duplicate_count}"
        )
    return frame, actual_sha, source_path.stat().st_size, duplicate_count


def _base_audit(
    contract: ExternalSourceContract,
    *,
    actual_sha: str,
    byte_count: int,
    row_count: int,
    years: Sequence[int],
    suppression: Mapping[str, int],
    duplicate_dimensional_key_count: int,
) -> dict[str, Any]:
    audit: dict[str, Any] = {
        "dataset_id": contract.dataset_id,
        "source_page": contract.source_page,
        "direct_download_url": contract.direct_download_url,
        "evidence_role": VALIDATION_ONLY_ROLE,
        "geography_basis": contract.geography_basis,
        "license": contract.license_name,
        "encoding": contract.encoding,
        "sha256": actual_sha,
        "byte_count": int(byte_count),
        "row_count": int(row_count),
        "column_count": len(contract.raw_columns),
        "observed_years": "|".join(map(str, years)),
        "schema_validated": True,
        "row_count_validated": True,
        "sha256_validated": True,
        "duplicate_dimensional_key_count": int(duplicate_dimensional_key_count),
        "concat_allowed": False,
        "promotion_allowed": False,
        "official_output_write_allowed": False,
    }
    for metric, count in sorted(suppression.items()):
        audit[f"{metric}_suppressed_count"] = int(count)
        audit[f"{metric}_suppressed_fraction"] = float(count / row_count)
    return audit


def read_provider_specialty_monthly_cp949(
    path: str | Path,
    *,
    contract: ExternalSourceContract = PROVIDER_SPECIALTY_MONTHLY_CONTRACT,
    encoding: str = "cp949",
) -> ValidatedExternalData:
    """Validate and normalize official provider-address specialty-month data."""

    raw, actual_sha, byte_count, duplicate_count = _validate_source_frame(
        path,
        contract,
        expected_kind="provider_specialty_monthly",
        encoding=encoding,
    )
    year_month = raw["진료년월"].astype("string").str.extract(r"^(\d{4})-(\d{2})$")
    if year_month.isna().any(axis=None):
        raise ValueError("Provider source 진료년월 must use YYYY-MM")
    years = year_month[0].astype(int)
    months = year_month[1].astype(int)
    observed_years = tuple(sorted(years.unique().tolist()))
    if observed_years != contract.expected_years:
        raise ValueError(
            f"Provider source years {observed_years} do not match {contract.expected_years}"
        )
    if not months.between(1, 12).all():
        raise ValueError("Provider source contains a month outside 1..12")
    for year in contract.expected_years:
        if set(months.loc[years.eq(year)].unique()) != set(range(1, 13)):
            raise ValueError(f"Provider source year {year} lacks complete month coverage")

    normalized = pd.DataFrame(
        {
            "source_dataset_id": contract.dataset_id,
            "evidence_role": VALIDATION_ONLY_ROLE,
            "promotion_allowed": False,
            "source_row_number": np.arange(2, len(raw) + 2, dtype=int),
            "year": years.to_numpy(int),
            "month": months.to_numpy(int),
            "institution_type": raw["요양기관종별"].str.strip(),
            "specialty_code": raw["진료과목코드"].str.strip(),
            "specialty_name": raw["진료과목명"].str.strip(),
            "care_type": raw["진료형태"].str.strip(),
            "provider_sido_name": raw["요양기관 주소지"].str.strip(),
        }
    )
    suppression: dict[str, int] = {}
    for metric, raw_column in contract.metric_columns:
        values, lower, upper, suppressed = _parse_metric(
            raw[raw_column],
            metric=metric,
            allow_suppression=metric in contract.suppression_allowed_metrics,
        )
        normalized[metric] = values.to_numpy(float)
        normalized[f"{metric}_lower"] = lower.to_numpy(float)
        normalized[f"{metric}_upper"] = upper.to_numpy(float)
        normalized[f"{metric}_suppressed"] = suppressed.to_numpy(bool)
        suppression[metric] = int(suppressed.sum())
    audit = _base_audit(
        contract,
        actual_sha=actual_sha,
        byte_count=byte_count,
        row_count=len(raw),
        years=observed_years,
        suppression=suppression,
        duplicate_dimensional_key_count=duplicate_count,
    )
    audit["observed_year_month_count"] = int(
        normalized[["year", "month"]].drop_duplicates().shape[0]
    )
    audit["provider_sido_count"] = int(normalized["provider_sido_name"].nunique())
    return ValidatedExternalData(contract=contract, frame=normalized, audit=audit)


def read_patient_allcare_monthly_cp949(
    path: str | Path,
    *,
    contract: ExternalSourceContract = PATIENT_ALLCARE_MONTHLY_CONTRACT,
    encoding: str = "cp949",
) -> ValidatedExternalData:
    """Validate and normalize official patient-address all-care monthly data."""

    raw, actual_sha, byte_count, duplicate_count = _validate_source_frame(
        path,
        contract,
        expected_kind="patient_allcare_monthly",
        encoding=encoding,
    )
    duplicate_rows = raw.loc[
        raw.duplicated(list(contract.key_columns), keep=False)
    ]
    duplicate_geographies = tuple(sorted(duplicate_rows["시도"].astype(str).unique()))
    if duplicate_geographies != contract.expected_duplicate_geographies:
        raise ValueError(
            f"{contract.dataset_id} duplicate geography mismatch: "
            f"expected={contract.expected_duplicate_geographies}, "
            f"observed={duplicate_geographies}"
        )
    years = pd.to_numeric(raw["진료년도"], errors="coerce")
    months = pd.to_numeric(raw["진료월"], errors="coerce")
    if years.isna().any() or months.isna().any():
        raise ValueError("All-care source year/month columns must be numeric")
    years = years.astype(int)
    months = months.astype(int)
    observed_years = tuple(sorted(years.unique().tolist()))
    if observed_years != contract.expected_years:
        raise ValueError(
            f"All-care source years {observed_years} do not match {contract.expected_years}"
        )
    if not months.between(1, 12).all():
        raise ValueError("All-care source contains a month outside 1..12")
    for year in contract.expected_years:
        if set(months.loc[years.eq(year)].unique()) != set(range(1, 13)):
            raise ValueError(f"All-care source year {year} lacks complete month coverage")

    normalized = pd.DataFrame(
        {
            "source_dataset_id": contract.dataset_id,
            "evidence_role": VALIDATION_ONLY_ROLE,
            "promotion_allowed": False,
            "source_row_number": np.arange(2, len(raw) + 2, dtype=int),
            "year": years.to_numpy(int),
            "month": months.to_numpy(int),
            "institution_type": raw["요양기관종별"].str.strip(),
            "patient_sido_name": raw["시도"].str.strip(),
            "source_sigungu_name": raw["시군구"].str.strip(),
        }
    )
    suppression: dict[str, int] = {}
    for metric, raw_column in contract.metric_columns:
        values, lower, upper, suppressed = _parse_metric(
            raw[raw_column],
            metric=metric,
            allow_suppression=metric in contract.suppression_allowed_metrics,
        )
        normalized[metric] = values.to_numpy(float)
        normalized[f"{metric}_lower"] = lower.to_numpy(float)
        normalized[f"{metric}_upper"] = upper.to_numpy(float)
        normalized[f"{metric}_suppressed"] = suppressed.to_numpy(bool)
        suppression[metric] = int(suppressed.sum())
    audit = _base_audit(
        contract,
        actual_sha=actual_sha,
        byte_count=byte_count,
        row_count=len(raw),
        years=observed_years,
        suppression=suppression,
        duplicate_dimensional_key_count=duplicate_count,
    )
    audit["duplicate_dimensional_key_geographies"] = "|".join(duplicate_geographies)
    audit["observed_year_month_count"] = int(
        normalized[["year", "month"]].drop_duplicates().shape[0]
    )
    audit["patient_sido_count"] = int(normalized["patient_sido_name"].nunique())
    return ValidatedExternalData(contract=contract, frame=normalized, audit=audit)


def _require_validation_only(data: ValidatedExternalData) -> None:
    frame = data.frame
    required = {"source_dataset_id", "evidence_role", "promotion_allowed"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ExternalValidationIsolationError(
            f"External frame lacks validation-only markers: {missing}"
        )
    if not frame["source_dataset_id"].astype(str).eq(data.contract.dataset_id).all():
        raise ExternalValidationIsolationError("External source_dataset_id marker changed")
    if not frame["evidence_role"].astype(str).eq(VALIDATION_ONLY_ROLE).all():
        raise ExternalValidationIsolationError("External evidence role is not validation_only")
    if frame["promotion_allowed"].astype(bool).any():
        raise ExternalValidationIsolationError("External frame improperly permits promotion")


def assert_canonical_nhis_panel_isolated(
    panel: pd.DataFrame,
    *,
    expected_years: Sequence[int] = (2022, 2023),
    expected_policy_sigungu: int = 11,
    expected_services: int = 16,
) -> dict[str, int]:
    """Fail closed if a canonical panel shows any external-row contamination."""

    if tuple(panel.columns) != CANONICAL_NHIS_COLUMNS:
        raise ExternalValidationIsolationError(
            "Canonical NHIS schema changed or contains external-validation columns; "
            f"expected={list(CANONICAL_NHIS_COLUMNS)}, observed={list(panel.columns)}"
        )
    if panel.empty:
        raise ExternalValidationIsolationError("Canonical NHIS panel is empty")
    years = tuple(sorted(pd.to_numeric(panel["year"], errors="raise").astype(int).unique()))
    expected_years = tuple(sorted(int(value) for value in expected_years))
    if years != expected_years:
        raise ExternalValidationIsolationError(
            f"Canonical years changed: expected={expected_years}, observed={years}"
        )
    months = pd.to_numeric(panel["month"], errors="raise").astype(int)
    if not months.between(1, 12).all():
        raise ExternalValidationIsolationError("Canonical month is outside 1..12")
    if panel.isna().any(axis=None):
        raise ExternalValidationIsolationError("Canonical NHIS panel contains missing values")
    key = ["year", "month", "policy_sigungu_name", "service_id"]
    if panel.duplicated(key).any():
        raise ExternalValidationIsolationError("Canonical NHIS panel has duplicate keys")
    sigungu_n = int(panel["policy_sigungu_name"].nunique())
    service_n = int(panel["service_id"].nunique())
    if sigungu_n != expected_policy_sigungu or service_n != expected_services:
        raise ExternalValidationIsolationError(
            "Canonical dimensions changed: "
            f"sigungu={sigungu_n}/{expected_policy_sigungu}, "
            f"services={service_n}/{expected_services}"
        )
    expected_rows = len(expected_years) * 12 * expected_policy_sigungu * expected_services
    if len(panel) != expected_rows:
        raise ExternalValidationIsolationError(
            f"Canonical grid is incomplete: expected={expected_rows}, observed={len(panel)}"
        )
    for value_column in ("persons", "visits"):
        values = pd.to_numeric(panel[value_column], errors="coerce").to_numpy(float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ExternalValidationIsolationError(
                f"Canonical {value_column} contains invalid values"
            )
    return {
        "row_count": int(len(panel)),
        "year_count": len(years),
        "policy_sigungu_count": sigungu_n,
        "service_count": service_n,
        "external_validation_row_count": 0,
    }


def forbid_external_validation_concat(
    canonical_panel: pd.DataFrame,
    external: ValidatedExternalData,
    *,
    expected_policy_sigungu: int = 11,
    expected_services: int = 16,
) -> NoReturn:
    """Explicitly reject attempts to concatenate validation evidence."""

    assert_canonical_nhis_panel_isolated(
        canonical_panel,
        expected_policy_sigungu=expected_policy_sigungu,
        expected_services=expected_services,
    )
    _require_validation_only(external)
    raise ExternalValidationIsolationError(
        f"Dataset {external.contract.dataset_id} is validation_only and cannot be "
        "concatenated with the canonical patient-sigungu x specialty x month panel"
    )


def _bundle_contract(bundle_config: Mapping[str, Any]) -> list[tuple[str, str, dict[str, float]]]:
    bundles = bundle_config.get("bundles", [])
    if not isinstance(bundles, list) or not bundles:
        raise ValueError("Bundle config contains no bundles")
    output: list[tuple[str, str, dict[str, float]]] = []
    observed_ids: set[str] = set()
    for bundle in bundles:
        bundle_id = str(bundle.get("bundle_id", "")).strip()
        if not bundle_id or bundle_id in observed_ids:
            raise ValueError("Bundle ids must be nonblank and unique")
        observed_ids.add(bundle_id)
        included = {
            str(service_id): float(weight)
            for service_id, weight in dict(bundle.get("included_services", {})).items()
        }
        if not included or not all(np.isfinite(value) and value > 0 for value in included.values()):
            raise ValueError(f"Bundle {bundle_id} has invalid service weights")
        if not np.isclose(sum(included.values()), 1.0, atol=1e-12):
            raise ValueError(f"Bundle {bundle_id} weights do not sum to one")
        output.append(
            (bundle_id, str(bundle.get("bundle_name_ko", bundle_id)), included)
        )
    return output


def _ranked_seasons(values: pd.Series) -> list[str]:
    frame = pd.DataFrame(
        {"season": list(SEASON_ORDER), "value": [float(values.loc[x]) for x in SEASON_ORDER]}
    )
    frame["season_order"] = frame["season"].map(
        {season: index for index, season in enumerate(SEASON_ORDER)}
    )
    return frame.sort_values(
        ["value", "season_order"], ascending=[False, True], kind="mergesort"
    )["season"].tolist()


def _spearman(left: pd.Series, right: pd.Series) -> float:
    x = left.reindex(SEASON_ORDER).rank(method="average").to_numpy(float)
    y = right.reindex(SEASON_ORDER).rank(method="average").to_numpy(float)
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _phase_distance(left: str, right: str) -> int:
    a, b = SEASON_ORDER.index(left), SEASON_ORDER.index(right)
    distance = abs(a - b)
    return int(min(distance, len(SEASON_ORDER) - distance))


def _no_promotion_ledger(
    *,
    candidate_id: str,
    evidence_sha256: str,
    reason: str,
    limitations: str,
    run_id: str,
    parent_stage2_run_id: str,
    parent_hardening_run_id: str,
) -> pd.DataFrame:
    row = {
        "run_id": str(run_id),
        "parent_stage2_run_id": str(parent_stage2_run_id),
        "parent_hardening_run_id": str(parent_hardening_run_id),
        "candidate_id": candidate_id,
        "decision": NO_PROMOTION_DECISION,
        "decision_effect": "proposal_only",
        "hard_passed": 0,
        "hard_total": 0,
        "failed_hard_gates": "not_applicable_diagnostic_only",
        "official_outputs_mutated": False,
        "official_interface_mutated": False,
        "promotion_applied": False,
        "stage3_started": False,
        "requires_manual_authorization": True,
        "requires_stage2a_refreeze": True,
        "reason": reason,
        "limitations": f"evidence_sha256={evidence_sha256}; {limitations}",
    }
    return pd.DataFrame([row], columns=NO_PROMOTION_LEDGER_COLUMNS)


def diagnose_provider_bundle_phase(
    data: ValidatedExternalData,
    bundle_config: Mapping[str, Any],
    *,
    province_name: str = "충청북도",
    train_year: int = 2021,
    holdout_years: Sequence[int] = (2022, 2023),
    service_name_to_id: Mapping[str, str] | None = None,
    run_id: str = "external_validation_unpersisted",
    parent_stage2_run_id: str = "unbound",
    parent_hardening_run_id: str = "unbound",
) -> ProviderPhaseDiagnosticResult:
    """Diagnose provider-side bundle phase without changing any release."""

    _require_validation_only(data)
    if data.contract.kind != "provider_specialty_monthly":
        raise ValueError("Provider phase diagnostic requires dataset 15141856 contract")
    bundles = _bundle_contract(bundle_config)
    years = (int(train_year), *(int(value) for value in holdout_years))
    if len(set(years)) != len(years) or set(years) - set(data.contract.expected_years):
        raise ValueError("Provider phase years must be distinct source years")
    mapping = dict(service_name_to_id or SERVICE_NAME_TO_ID)
    configured_services = sorted(
        {service_id for _, _, included in bundles for service_id in included}
    )
    work = data.frame.loc[
        data.frame["provider_sido_name"].astype(str).eq(province_name)
        & data.frame["year"].isin(years)
    ].copy()
    if work.empty:
        raise ValueError(f"Provider source has no rows for {province_name}")
    work["service_id"] = work["specialty_name"].astype(str).map(mapping)
    work = work.loc[work["service_id"].isin(configured_services)].copy()
    missing_services = sorted(set(configured_services) - set(work["service_id"]))
    if missing_services:
        raise ValueError(f"Provider source lacks configured bundle services: {missing_services}")

    scenario_columns = {
        "suppressed_as_0": "visits_lower",
        "suppressed_as_4": "visits_upper",
    }
    bundle_month_rows: list[dict[str, Any]] = []
    for scenario, value_column in scenario_columns.items():
        service_month = (
            work.groupby(["year", "month", "service_id"], as_index=False)[value_column]
            .sum()
            .rename(columns={value_column: "visits_scenario"})
        )
        expected_grid = pd.MultiIndex.from_product(
            [years, range(1, 13), configured_services],
            names=["year", "month", "service_id"],
        )
        observed_grid = pd.MultiIndex.from_frame(
            service_month[["year", "month", "service_id"]]
        )
        missing_grid = expected_grid.difference(observed_grid)
        if len(missing_grid):
            raise ValueError(
                f"Provider service-month grid is incomplete; first missing={missing_grid[0]}"
            )
        service_month["annual_month_mean"] = service_month.groupby(
            ["year", "service_id"], sort=False
        )["visits_scenario"].transform("mean")
        if (service_month["annual_month_mean"] <= 0).any():
            raise ValueError("Provider service-year has a non-positive monthly mean")
        service_month["service_month_modifier"] = (
            service_month["visits_scenario"] / service_month["annual_month_mean"]
        )
        indexed = service_month.set_index(["year", "month", "service_id"])
        for bundle_id, bundle_name, included in bundles:
            for year in years:
                for month in range(1, 13):
                    modifier = sum(
                        weight
                        * float(indexed.loc[(year, month, service_id), "service_month_modifier"])
                        for service_id, weight in included.items()
                    )
                    bundle_month_rows.append(
                        {
                            "suppression_endpoint": scenario,
                            "bundle_id": bundle_id,
                            "bundle_name_ko": bundle_name,
                            "year": year,
                            "month": month,
                            "bundle_month_modifier": modifier,
                        }
                    )
    bundle_month = pd.DataFrame(bundle_month_rows)
    bundle_month["season"] = bundle_month["month"].map(
        {month: season for season, months in SEASON_MONTHS.items() for month in months}
    )
    seasonal = (
        bundle_month.groupby(
            ["suppression_endpoint", "bundle_id", "bundle_name_ko", "year", "season"],
            as_index=False,
        )["bundle_month_modifier"]
        .mean()
        .rename(columns={"bundle_month_modifier": "bundle_season_modifier"})
    )
    seasonal["evidence_role"] = VALIDATION_ONLY_ROLE
    seasonal["promotion_allowed"] = False

    detail_rows: list[dict[str, Any]] = []
    for scenario, scenario_group in seasonal.groupby("suppression_endpoint", sort=True):
        for bundle_id, bundle_group in scenario_group.groupby("bundle_id", sort=True):
            profiles = {
                int(year): group.set_index("season")["bundle_season_modifier"].reindex(
                    SEASON_ORDER
                )
                for year, group in bundle_group.groupby("year", sort=True)
            }
            train = profiles[int(train_year)]
            comparisons: list[tuple[str, pd.Series]] = [
                (str(year), profiles[int(year)]) for year in holdout_years
            ]
            pooled = pd.concat(
                [profiles[int(year)].rename(str(year)) for year in holdout_years], axis=1
            ).mean(axis=1)
            comparisons.append(("pooled_" + "_".join(map(str, holdout_years)), pooled))
            train_rank = _ranked_seasons(train)
            for comparison, holdout in comparisons:
                holdout_rank = _ranked_seasons(holdout)
                detail_rows.append(
                    {
                        "bundle_id": bundle_id,
                        "suppression_endpoint": scenario,
                        "train_year": int(train_year),
                        "comparison": comparison,
                        "train_primary_season": train_rank[0],
                        "comparison_primary_season": holdout_rank[0],
                        "top1_agreement": train_rank[0] == holdout_rank[0],
                        "train_primary_in_comparison_top2": train_rank[0] in holdout_rank[:2],
                        "top2_overlap": len(set(train_rank[:2]) & set(holdout_rank[:2])) / 2.0,
                        "season_rank_spearman": _spearman(train, holdout),
                        "phase_distance_seasons": _phase_distance(
                            train_rank[0], holdout_rank[0]
                        ),
                        "evidence_role": VALIDATION_ONLY_ROLE,
                        "promotion_allowed": False,
                    }
                )
    detail = pd.DataFrame(detail_rows).sort_values(
        ["bundle_id", "comparison", "suppression_endpoint"], kind="mergesort"
    ).reset_index(drop=True)
    endpoint_invariance = (
        detail.groupby(["bundle_id", "comparison"])
        .agg(
            train_endpoint_n=("train_primary_season", "nunique"),
            comparison_endpoint_n=("comparison_primary_season", "nunique"),
        )
        .assign(endpoint_invariant=lambda value: value.max(axis=1).eq(1))
        .groupby("bundle_id")["endpoint_invariant"]
        .all()
    )
    summary = (
        detail.groupby("bundle_id", as_index=False)
        .agg(
            comparison_count=("comparison", "size"),
            min_season_rank_spearman=("season_rank_spearman", "min"),
            mean_season_rank_spearman=("season_rank_spearman", "mean"),
            top1_agreement_fraction=("top1_agreement", "mean"),
            train_primary_in_comparison_top2_fraction=(
                "train_primary_in_comparison_top2",
                "mean",
            ),
            max_phase_distance_seasons=("phase_distance_seasons", "max"),
        )
        .sort_values("bundle_id", kind="mergesort")
        .reset_index(drop=True)
    )
    summary["suppression_endpoint_decision_invariant"] = summary["bundle_id"].map(
        endpoint_invariance
    ).astype(bool)
    summary["diagnostic_status"] = "DIAGNOSTIC_ONLY"
    summary["promotion_allowed"] = False
    ledger = _no_promotion_ledger(
        candidate_id="provider_phase_2021_to_2022_2023",
        evidence_sha256=str(data.audit["sha256"]),
        reason=(
            "Provider-address province evidence can diagnose phase concordance but cannot "
            "replace or promote patient-address sigungu evidence"
        ),
        limitations=(
            "provider_address_not_patient_address; province_not_sigungu; overlapping_2022_2023_"
            "claims_not_independent; persons_not_summed; suppressed_cells_endpoint_sensitivity"
        ),
        run_id=run_id,
        parent_stage2_run_id=parent_stage2_run_id,
        parent_hardening_run_id=parent_hardening_run_id,
    )
    return ProviderPhaseDiagnosticResult(
        bundle_season_profile=seasonal.sort_values(
            ["bundle_id", "suppression_endpoint", "year", "season"], kind="mergesort"
        ).reset_index(drop=True),
        detail=detail,
        summary=summary,
        ledger=ledger,
    )


def _map_policy_sigungu(value: object) -> str:
    name = str(value).strip()
    if name in CHEONGJU_SOURCE_UNITS or name.startswith("청주시 "):
        return "청주시"
    return name


def build_allcare_common_shock_candidate(
    data: ValidatedExternalData,
    *,
    province_name: str = "충청북도",
    baseline_year: int = 2019,
    shock_years: Sequence[int] = (2020, 2021),
    expected_policy_sigungu: int = 11,
    run_id: str = "external_validation_unpersisted",
    parent_stage2_run_id: str = "unbound",
    parent_hardening_run_id: str = "unbound",
) -> CommonShockCandidateResult:
    """Build, but never apply, a 2019-baseline common-shock correction."""

    _require_validation_only(data)
    if data.contract.kind != "patient_allcare_monthly":
        raise ValueError("Common-shock diagnostic requires dataset 15141213 contract")
    if int(baseline_year) != 2019 or tuple(int(value) for value in shock_years) != (2020, 2021):
        raise ValueError(
            "Common-shock candidate is locked to baseline_year=2019 and "
            "shock_years=(2020, 2021)"
        )
    fit_years = (int(baseline_year), *(int(value) for value in shock_years))
    if len(set(fit_years)) != len(fit_years):
        raise ValueError("Baseline and shock years must be distinct")
    if set(fit_years) - set(data.contract.expected_years):
        raise ValueError("Common-shock fit years are not covered by the source")
    if max(fit_years) > 2021:
        raise ValueError("Common-shock candidate is frozen to pre-2022 fit years")
    work = data.frame.loc[
        data.frame["patient_sido_name"].astype(str).eq(province_name)
        & data.frame["year"].isin(fit_years)
    ].copy()
    if work.empty:
        raise ValueError(f"All-care source has no rows for {province_name}")
    work["policy_sigungu_name"] = work["source_sigungu_name"].map(_map_policy_sigungu)
    observed_sigungu = set(work["policy_sigungu_name"].astype(str))
    if len(observed_sigungu) != expected_policy_sigungu:
        raise ValueError(
            f"All-care mapping produced {len(observed_sigungu)} policy sigungu, "
            f"expected {expected_policy_sigungu}"
        )
    if expected_policy_sigungu == len(POLICY_SIGUNGU) and observed_sigungu != POLICY_SIGUNGU:
        raise ValueError(
            "All-care policy sigungu names differ from the canonical Chungbuk contract"
        )

    scenario_columns = {
        "suppressed_as_0": "visits_lower",
        "suppressed_as_4": "visits_upper",
    }
    profile_rows: list[pd.DataFrame] = []
    for scenario, value_column in scenario_columns.items():
        sigungu_month = (
            work.groupby(
                ["year", "month", "policy_sigungu_name"], as_index=False
            )[value_column]
            .sum()
            .rename(columns={value_column: "allcare_visits_scenario"})
        )
        expected_rows = len(fit_years) * 12 * expected_policy_sigungu
        if len(sigungu_month) != expected_rows or sigungu_month.duplicated(
            ["year", "month", "policy_sigungu_name"]
        ).any():
            raise ValueError("All-care fit grid is incomplete after policy-sigungu mapping")
        sigungu_month["sigungu_year_month_mean"] = sigungu_month.groupby(
            ["year", "policy_sigungu_name"], sort=False
        )["allcare_visits_scenario"].transform("mean")
        if (sigungu_month["sigungu_year_month_mean"] <= 0).any():
            raise ValueError("All-care sigungu-year has a non-positive monthly mean")
        sigungu_month["sigungu_year_normalized"] = (
            sigungu_month["allcare_visits_scenario"]
            / sigungu_month["sigungu_year_month_mean"]
        )
        common = (
            sigungu_month.groupby(["year", "month"], as_index=False)[
                "sigungu_year_normalized"
            ]
            .mean()
            .rename(columns={"sigungu_year_normalized": "equal_sigungu_common_index"})
        )
        common["suppression_endpoint"] = scenario
        profile_rows.append(common)
    common_profile = pd.concat(profile_rows, ignore_index=True).sort_values(
        ["suppression_endpoint", "year", "month"], kind="mergesort"
    ).reset_index(drop=True)
    common_profile["fit_scope"] = "2019_2021_only"
    common_profile["evidence_role"] = VALIDATION_ONLY_ROLE
    common_profile["promotion_allowed"] = False

    shock_rows: list[dict[str, Any]] = []
    for scenario, scenario_group in common_profile.groupby("suppression_endpoint", sort=True):
        indexed = scenario_group.set_index(["year", "month"])["equal_sigungu_common_index"]
        for shock_year in shock_years:
            for month in range(1, 13):
                baseline = float(indexed.loc[(int(baseline_year), month)])
                observed = float(indexed.loc[(int(shock_year), month)])
                if baseline <= 0 or observed <= 0:
                    raise ValueError("Common-shock index must remain positive")
                ratio = observed / baseline
                shock_rows.append(
                    {
                        "suppression_endpoint": scenario,
                        "baseline_year": int(baseline_year),
                        "shock_year": int(shock_year),
                        "month": month,
                        "baseline_common_index": baseline,
                        "observed_common_index": observed,
                        "shock_ratio": ratio,
                        "log_shock_deviation": float(np.log(ratio)),
                        "candidate_adjustment_multiplier": 1.0 / ratio,
                        "fit_uses_2022_2023_rows": False,
                        "candidate_only": True,
                        "promotion_allowed": False,
                    }
                )
    shock_detail = pd.DataFrame(shock_rows).sort_values(
        ["month", "shock_year", "suppression_endpoint"], kind="mergesort"
    ).reset_index(drop=True)
    pooled = (
        shock_detail.groupby(["suppression_endpoint", "month"], as_index=False)
        .agg(
            candidate_adjustment_multiplier=(
                "candidate_adjustment_multiplier",
                lambda values: float(np.exp(np.log(values.astype(float)).mean())),
            ),
            max_abs_log_shock=("log_shock_deviation", lambda values: float(np.abs(values).max())),
        )
    )
    multiplier = pooled.pivot(
        index="month",
        columns="suppression_endpoint",
        values="candidate_adjustment_multiplier",
    ).sort_index()
    shock_strength = pooled.groupby("month")["max_abs_log_shock"].max()
    candidate = pd.DataFrame(
        {
            "month": multiplier.index.astype(int),
            "candidate_multiplier_suppressed_as_0": multiplier["suppressed_as_0"].to_numpy(float),
            "candidate_multiplier_suppressed_as_4": multiplier["suppressed_as_4"].to_numpy(float),
        }
    )
    candidate["candidate_multiplier_midpoint"] = np.sqrt(
        candidate["candidate_multiplier_suppressed_as_0"]
        * candidate["candidate_multiplier_suppressed_as_4"]
    )
    candidate["suppression_endpoint_spread"] = (
        candidate[
            [
                "candidate_multiplier_suppressed_as_0",
                "candidate_multiplier_suppressed_as_4",
            ]
        ].max(axis=1)
        - candidate[
            [
                "candidate_multiplier_suppressed_as_0",
                "candidate_multiplier_suppressed_as_4",
            ]
        ].min(axis=1)
    )
    candidate["max_abs_log_shock"] = candidate["month"].map(shock_strength).astype(float)
    candidate["baseline_year"] = int(baseline_year)
    candidate["shock_years"] = "|".join(map(str, shock_years))
    candidate["fit_scope"] = "2019_2021_only"
    candidate["candidate_only"] = True
    candidate["promotion_allowed"] = False
    candidate["apply_to_canonical_panel"] = False
    ledger = _no_promotion_ledger(
        candidate_id="allcare_common_shock_2019_2021",
        evidence_sha256=str(data.audit["sha256"]),
        reason=(
            "The all-care series can expose a pre-2022 common shock candidate, but it lacks "
            "specialty resolution and is not an adjustment authorized for the canonical panel"
        ),
        limitations=(
            "all_care_includes_provider_types_outside_canonical_scope; specialty_absent; "
            "pandemic_and_year_effects_conflated; suppressed_cells_endpoint_sensitivity; "
            "fit_rows_2022_2023=0"
        ),
        run_id=run_id,
        parent_stage2_run_id=parent_stage2_run_id,
        parent_hardening_run_id=parent_hardening_run_id,
    )
    return CommonShockCandidateResult(
        common_profile=common_profile,
        shock_detail=shock_detail,
        candidate=candidate,
        ledger=ledger,
    )
