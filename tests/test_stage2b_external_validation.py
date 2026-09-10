from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mediroad.temporal.external_validation import (
    CANONICAL_NHIS_COLUMNS,
    NO_PROMOTION_LEDGER_COLUMNS,
    PATIENT_ALLCARE_MONTHLY_CONTRACT,
    PROVIDER_SPECIALTY_MONTHLY_CONTRACT,
    ExternalValidationIsolationError,
    assert_canonical_nhis_panel_isolated,
    build_allcare_common_shock_candidate,
    diagnose_provider_bundle_phase,
    forbid_external_validation_concat,
    read_patient_allcare_monthly_cp949,
    read_provider_specialty_monthly_cp949,
)


def _write_cp949(frame: pd.DataFrame, path: Path) -> str:
    frame.to_csv(path, index=False, encoding="cp949", lineterminator="\n")
    return sha256(path.read_bytes()).hexdigest()


def _provider_raw() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    internal = np.array([80, 85, 140, 135, 130, 90, 85, 80, 105, 100, 95, 75])
    family = np.array([85, 80, 90, 95, 100, 90, 85, 80, 135, 140, 130, 90])
    for year in (2021, 2022, 2023):
        year_scale = 1.0 + (year - 2021) * 0.1
        for month in range(1, 13):
            for code, name, profile in (
                ("1", "내과", internal),
                ("23", "가정의학과", family),
            ):
                visits: object = str(int(profile[month - 1] * year_scale))
                persons: object = str(max(5, int(float(visits) * 0.7)))
                if year == 2021 and month == 7 and name == "가정의학과":
                    persons = "*"
                    visits = "*"
                rows.append(
                    {
                        "진료년월": f"{year}-{month:02d}",
                        "요양기관종별": "의원",
                        "진료과목코드": code,
                        "진료과목명": name,
                        "진료형태": "외래",
                        "요양기관 주소지": "충청북도",
                        "진료인원(명)": persons,
                        "진료건수(건)": visits,
                    }
                )
    return pd.DataFrame(rows, columns=PROVIDER_SPECIALTY_MONTHLY_CONTRACT.raw_columns)


def _allcare_raw() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    source_units = (("괴산군", 1.0), ("청주시 상당구", 3.0))
    for year in (2019, 2020, 2021, 2022, 2023):
        for month in range(1, 13):
            for sigungu, scale in source_units:
                base = 100.0
                if year in (2020, 2021) and month == 3:
                    base = 50.0
                visits: object = str(int(base * scale))
                if year == 2020 and month == 4 and sigungu == "괴산군":
                    visits = "*"
                rows.append(
                    {
                        "진료년도": str(year),
                        "진료월": f"{month:02d}",
                        "요양기관종별": "의원",
                        "시도": "충청북도",
                        "시군구": sigungu,
                        "진료건수(건)": visits,
                        "진료비(천원)": str(int(base * scale * 10)),
                    }
                )
    return pd.DataFrame(rows, columns=PATIENT_ALLCARE_MONTHLY_CONTRACT.raw_columns)


@pytest.fixture
def provider_source(tmp_path: Path):
    path = tmp_path / "provider.csv"
    raw = _provider_raw()
    digest = _write_cp949(raw, path)
    contract = replace(
        PROVIDER_SPECIALTY_MONTHLY_CONTRACT,
        expected_sha256=digest,
        expected_rows=len(raw),
    )
    return path, contract


@pytest.fixture
def allcare_source(tmp_path: Path):
    path = tmp_path / "allcare.csv"
    raw = _allcare_raw()
    digest = _write_cp949(raw, path)
    contract = replace(
        PATIENT_ALLCARE_MONTHLY_CONTRACT,
        expected_sha256=digest,
        expected_rows=len(raw),
        expected_duplicate_dimensional_key_count=0,
        expected_duplicate_geographies=(),
    )
    return path, contract


@pytest.fixture
def bundle_config() -> dict:
    return {
        "status": "synthetic_validation_only",
        "bundles": [
            {
                "bundle_id": "internal_only",
                "bundle_name_ko": "내과 단독",
                "included_services": {"internal_medicine": 1.0},
            },
            {
                "bundle_id": "primary_mix",
                "bundle_name_ko": "일차진료 혼합",
                "included_services": {
                    "internal_medicine": 0.5,
                    "family_medicine": 0.5,
                },
            },
        ],
    }


def _canonical_panel() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "year": year,
                "month": month,
                "policy_sigungu_name": "괴산군",
                "service_id": "internal_medicine",
                "persons": float(50 + month),
                "visits": float(100 + month),
            }
            for year in (2022, 2023)
            for month in range(1, 13)
        ],
        columns=CANONICAL_NHIS_COLUMNS,
    )


def test_provider_parser_validates_bytes_schema_rows_and_suppression(provider_source) -> None:
    path, contract = provider_source
    data = read_provider_specialty_monthly_cp949(path, contract=contract)

    assert data.audit["sha256_validated"] is True
    assert data.audit["schema_validated"] is True
    assert data.audit["row_count"] == 72
    assert data.audit["visits_suppressed_count"] == 1
    assert data.audit["persons_suppressed_count"] == 1
    assert data.audit["concat_allowed"] is False
    assert data.audit["promotion_allowed"] is False
    suppressed = data.frame.loc[data.frame["visits_suppressed"]].iloc[0]
    assert np.isnan(suppressed["visits"])
    assert suppressed["visits_lower"] == 0.0
    assert suppressed["visits_upper"] == 4.0
    assert data.frame["evidence_role"].eq("validation_only").all()


def test_allcare_parser_validates_contract_and_never_coerces_star_to_zero(allcare_source) -> None:
    path, contract = allcare_source
    data = read_patient_allcare_monthly_cp949(path, contract=contract)

    assert data.audit["row_count"] == 120
    assert data.audit["visits_suppressed_count"] == 1
    assert data.audit["cost_thousand_krw_suppressed_count"] == 0
    row = data.frame.loc[data.frame["visits_suppressed"]].iloc[0]
    assert np.isnan(row["visits"])
    assert row["visits_lower"] == 0.0
    assert row["visits_upper"] == 4.0


def test_parsers_fail_closed_on_sha_row_schema_and_encoding(provider_source, tmp_path) -> None:
    path, contract = provider_source
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        read_provider_specialty_monthly_cp949(
            path, contract=replace(contract, expected_sha256="0" * 64)
        )
    with pytest.raises(ValueError, match="row-count mismatch"):
        read_provider_specialty_monthly_cp949(
            path, contract=replace(contract, expected_rows=contract.expected_rows + 1)
        )
    with pytest.raises(ValueError, match="CP949"):
        read_provider_specialty_monthly_cp949(path, contract=contract, encoding="utf-8")

    bad_path = tmp_path / "bad_schema.csv"
    bad = _provider_raw().drop(columns="진료건수(건)")
    digest = _write_cp949(bad, bad_path)
    bad_contract = replace(
        contract,
        expected_sha256=digest,
        expected_rows=len(bad),
    )
    with pytest.raises(KeyError, match="schema mismatch"):
        read_provider_specialty_monthly_cp949(bad_path, contract=bad_contract)


def test_canonical_concat_is_rejected_and_panel_contract_is_complete(provider_source) -> None:
    path, contract = provider_source
    external = read_provider_specialty_monthly_cp949(path, contract=contract)
    canonical = _canonical_panel()
    audit = assert_canonical_nhis_panel_isolated(
        canonical, expected_policy_sigungu=1, expected_services=1
    )
    assert audit["external_validation_row_count"] == 0

    with pytest.raises(ExternalValidationIsolationError, match="validation_only"):
        forbid_external_validation_concat(
            canonical,
            external,
            expected_policy_sigungu=1,
            expected_services=1,
        )
    contaminated = pd.concat([canonical, external.frame], ignore_index=True, sort=False)
    with pytest.raises(ExternalValidationIsolationError, match="schema changed"):
        assert_canonical_nhis_panel_isolated(
            contaminated, expected_policy_sigungu=1, expected_services=1
        )


def test_provider_phase_is_deterministic_and_no_promotion(
    provider_source, bundle_config
) -> None:
    path, contract = provider_source
    data = read_provider_specialty_monthly_cp949(path, contract=contract)
    original = data.frame.copy(deep=True)

    result = diagnose_provider_bundle_phase(data, bundle_config)

    pd.testing.assert_frame_equal(data.frame, original)
    assert len(result.bundle_season_profile) == 2 * 2 * 3 * 4
    assert len(result.detail) == 2 * 2 * 3
    assert result.detail["season_rank_spearman"].between(-1, 1).all()
    assert result.detail["promotion_allowed"].eq(False).all()
    assert result.summary["diagnostic_status"].eq("DIAGNOSTIC_ONLY").all()
    assert result.summary["promotion_allowed"].eq(False).all()
    assert tuple(result.ledger.columns) == NO_PROMOTION_LEDGER_COLUMNS
    ledger = result.ledger.iloc[0]
    assert ledger["decision"] == "NO_PROMOTION"
    assert ledger["decision_effect"] == "proposal_only"
    assert bool(ledger["official_outputs_mutated"]) is False
    assert bool(ledger["promotion_applied"]) is False
    assert bool(ledger["stage3_started"]) is False

    repeated = diagnose_provider_bundle_phase(data, bundle_config)
    pd.testing.assert_frame_equal(result.detail, repeated.detail)
    pd.testing.assert_frame_equal(result.summary, repeated.summary)


def test_allcare_candidate_uses_only_2019_2021_and_is_not_applied(allcare_source) -> None:
    path, contract = allcare_source
    data = read_patient_allcare_monthly_cp949(path, contract=contract)
    original = data.frame.copy(deep=True)

    result = build_allcare_common_shock_candidate(
        data,
        expected_policy_sigungu=2,
    )

    pd.testing.assert_frame_equal(data.frame, original)
    assert len(result.common_profile) == 2 * 3 * 12
    assert set(result.common_profile["year"]) == {2019, 2020, 2021}
    assert len(result.shock_detail) == 2 * 2 * 12
    assert not result.shock_detail["fit_uses_2022_2023_rows"].any()
    assert len(result.candidate) == 12
    march = result.candidate.set_index("month").loc[3]
    assert march["candidate_multiplier_midpoint"] > 1.5
    assert result.candidate["candidate_only"].all()
    assert not result.candidate["promotion_allowed"].any()
    assert not result.candidate["apply_to_canonical_panel"].any()
    ledger = result.ledger.iloc[0]
    assert ledger["decision"] == "NO_PROMOTION"
    assert bool(ledger["requires_manual_authorization"]) is True
    assert "fit_rows_2022_2023=0" in ledger["limitations"]


def test_allcare_common_shock_rejects_post_2021_fit(allcare_source) -> None:
    path, contract = allcare_source
    data = read_patient_allcare_monthly_cp949(path, contract=contract)
    with pytest.raises(ValueError, match="locked to baseline_year=2019"):
        build_allcare_common_shock_candidate(
            data,
            baseline_year=2020,
            shock_years=(2021, 2022),
            expected_policy_sigungu=2,
        )
    with pytest.raises(ValueError, match="locked to baseline_year=2019"):
        build_allcare_common_shock_candidate(
            data,
            baseline_year=2018,
            shock_years=(2019, 2020),
            expected_policy_sigungu=2,
        )
    with pytest.raises(ValueError, match="locked to baseline_year=2019"):
        build_allcare_common_shock_candidate(
            data,
            baseline_year=2019,
            shock_years=(2020,),
            expected_policy_sigungu=2,
        )
