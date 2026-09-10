from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from mediroad.stage4_2.contracts import EVIDENCE_COLUMNS
from mediroad.stage4_2.errors import ContractError
from mediroad.stage4_2.utils import normalize_bool


IDENTITY_ONLY_SOURCES = {
    "PROJECT_INTERNAL_OFFICIAL_IDENTITY_SOURCE",
    "PROJECT_INTERNAL_IDENTITY_ONLY",
}


@dataclass(frozen=True)
class FieldEvidenceAudit:
    normalized: pd.DataFrame
    issues: pd.DataFrame
    summary: dict[str, Any]
    release_verified_ids: set[str]


def _text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def audit_field_evidence(
    evidence: pd.DataFrame,
    field_form: pd.DataFrame,
    *,
    expected_venue_ids: set[str],
) -> FieldEvidenceAudit:
    """Audit evidence provenance independently from the all-YES field gate.

    A form can satisfy the historical 11-status-plus-date contract while still
    lacking any auditable source.  Such a row remains ``field_verified`` in the
    historical sense, but it cannot enter the Operational Final universe until
    this evidence gate also passes.
    """

    missing = [column for column in EVIDENCE_COLUMNS if column not in evidence.columns]
    if missing:
        raise ContractError(f"field evidence missing columns: {missing}")
    if "venue_id" not in field_form or "field_verified" not in field_form:
        raise ContractError("normalized field form must contain venue_id and field_verified")

    work = evidence[EVIDENCE_COLUMNS].copy()
    work["venue_id"] = _text(work["venue_id"])
    if work["venue_id"].duplicated().any():
        duplicates = work.loc[work["venue_id"].duplicated(False), "venue_id"].unique().tolist()
        raise ContractError(f"duplicate field evidence venue IDs: {duplicates[:10]}")
    actual_ids = set(work["venue_id"])
    if actual_ids != expected_venue_ids:
        missing_ids = sorted(expected_venue_ids - actual_ids)
        extra_ids = sorted(actual_ids - expected_venue_ids)
        raise ContractError(
            "field evidence venue IDs do not exactly match the active queue: "
            f"missing={missing_ids[:10]} extra={extra_ids[:10]}"
        )

    for column in EVIDENCE_COLUMNS[1:]:
        work[column] = _text(work[column])

    source_present = work["verification_source"].ne("")
    source_not_identity_only = ~work["verification_source"].str.upper().isin(IDENTITY_ONLY_SOURCES)
    reference_present = work["source_reference"].ne("") | work["evidence_file_or_url"].ne("")
    verifier_present = work["verifier"].ne("")
    contact_or_file_present = (
        work["contact_person_or_office"].ne("")
        | work["contact_channel"].ne("")
        | work["evidence_file_or_url"].ne("")
    )
    work["field_evidence_complete"] = (
        source_present
        & source_not_identity_only
        & reference_present
        & verifier_present
        & contact_or_file_present
    )

    verified = field_form[["venue_id", "field_verified"]].copy()
    verified["venue_id"] = _text(verified["venue_id"])
    verified["field_verified"] = (
        verified["field_verified"].map(normalize_bool).fillna(False).astype(bool)
    )
    work = work.merge(verified, on="venue_id", how="left", validate="one_to_one")
    work["field_verified"] = work["field_verified"].fillna(False).astype(bool)
    work["operational_release_verified"] = (
        work["field_verified"] & work["field_evidence_complete"]
    )

    issue_rows: list[dict[str, str]] = []
    requirements = {
        "verification_source": source_present & source_not_identity_only,
        "source_reference_or_evidence_file": reference_present,
        "verifier": verifier_present,
        "contact_channel_or_evidence_file": contact_or_file_present,
    }
    for label, present in requirements.items():
        failed = work["field_verified"] & ~present
        for venue_id in work.loc[failed, "venue_id"]:
            issue_rows.append(
                {
                    "venue_id": str(venue_id),
                    "field": label,
                    "issue": "MISSING_RELEASE_EVIDENCE",
                }
            )

    release_ids = set(work.loc[work["operational_release_verified"], "venue_id"])
    summary = {
        "rows": int(len(work)),
        "historical_field_verified_count": int(work["field_verified"].sum()),
        "field_evidence_complete_count": int(work["field_evidence_complete"].sum()),
        "operational_release_verified_count": int(work["operational_release_verified"].sum()),
        "verified_without_release_evidence_count": int(
            (work["field_verified"] & ~work["field_evidence_complete"]).sum()
        ),
        "issue_count": int(len(issue_rows)),
    }
    return FieldEvidenceAudit(
        normalized=work,
        issues=pd.DataFrame(
            issue_rows,
            columns=["venue_id", "field", "issue"],
        ),
        summary=summary,
        release_verified_ids=release_ids,
    )
