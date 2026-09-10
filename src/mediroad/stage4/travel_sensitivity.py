"""Fail-closed travel evidence and equity helpers for Stage 4.1.

Stage 3 contains useful *diagnostic* drive-time surfaces from two medical
centres.  Those surfaces do not prove where an operational mobile team starts.
This module keeps the diagnostic analysis available while preventing it from
silently becoming a final Stage 4 travel objective.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml

from .utils import gini


MOBILE_TEAM_BASE_SCHEMA_VERSION = "1.0"
BASE_EVIDENCE_STATUSES = frozenset({"CONFIRMED", "UNCONFIRMED"})
CONFIRMED_TRAVEL_EVIDENCE_STATUS = "CONFIRMED_TEAM_BASE_NETWORK"
DIAGNOSTIC_TRAVEL_EVIDENCE_STATUS = "DIAGNOSTIC_ONLY_ASSUMED_BASE"

TEAM_FIELDS = (
    "team_id",
    "home_base",
    "latitude",
    "longitude",
    "available_bundles",
    "vehicle_type",
    "max_daily_drive_minutes",
    "max_daily_work_minutes",
)

DIAGNOSTIC_SCENARIOS = {
    "cheongju_medical_center": {
        "column": "osm_from_cheongju_medical_center_drive_min_v6",
        "base_label": "CHEONGJU_MEDICAL_CENTER",
    },
    "chungju_medical_center": {
        "column": "osm_from_chungju_medical_center_drive_min_v6",
        "base_label": "CHUNGJU_MEDICAL_CENTER",
    },
    "dual_base_nearest": {
        "column": None,
        "base_label": "NEAREST_OF_CHEONGJU_OR_CHUNGJU_MEDICAL_CENTER",
    },
}


class TravelContractError(ValueError):
    """Raised when travel evidence could be mistaken for operational truth."""


@dataclass
class MobileTeamBaseContract:
    """Validated state of ``configs/model_v1/mobile_team_bases.yaml``."""

    schema_version: str
    evidence_status: str
    teams: pd.DataFrame
    source_reference: str | None
    verification_date: str | None
    source_path: Path | None
    final_travel_objective_enabled: bool
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "evidence_status": self.evidence_status,
            "team_count": int(len(self.teams)),
            "source_reference": self.source_reference,
            "verification_date": self.verification_date,
            "source_path": str(self.source_path) if self.source_path is not None else None,
            "final_travel_objective_enabled": bool(self.final_travel_objective_enabled),
            "reason": self.reason,
        }


def _require_mapping(payload: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise TravelContractError(f"{label} must be a mapping")
    return payload


def _nonblank(value: Any, label: str) -> str:
    if value is None or not isinstance(value, str) or not value.strip():
        raise TravelContractError(f"{label} must be a nonblank string")
    return value.strip()


def _positive_number(value: Any, label: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TravelContractError(f"{label} must be a positive number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise TravelContractError(f"{label} must be a positive number") from exc
    if not np.isfinite(number) or number <= 0:
        raise TravelContractError(f"{label} must be a positive finite number")
    return number


def _canonical_iso_date(value: Any, label: str) -> str:
    text = _nonblank(value, label)
    try:
        parsed = date.fromisoformat(text)
    except ValueError as exc:
        raise TravelContractError(f"{label} must use ISO YYYY-MM-DD") from exc
    if parsed.isoformat() != text:
        raise TravelContractError(f"{label} must use canonical ISO YYYY-MM-DD")
    return text


def _validate_team_rows(rows: Any) -> pd.DataFrame:
    if not isinstance(rows, list):
        raise TravelContractError("teams must be a list")
    records: list[dict[str, Any]] = []
    for index, raw in enumerate(rows):
        team = _require_mapping(raw, f"teams[{index}]")
        missing = [field for field in TEAM_FIELDS if field not in team]
        if missing:
            raise TravelContractError(f"teams[{index}] is missing required fields: {missing}")
        team_id = _nonblank(team["team_id"], f"teams[{index}].team_id")
        home_base = _nonblank(team["home_base"], f"teams[{index}].home_base")
        vehicle_type = _nonblank(team["vehicle_type"], f"teams[{index}].vehicle_type")

        try:
            latitude = float(team["latitude"])
            longitude = float(team["longitude"])
        except (TypeError, ValueError) as exc:
            raise TravelContractError(f"teams[{index}] coordinates must be numeric") from exc
        if not np.isfinite(latitude) or not -90 <= latitude <= 90:
            raise TravelContractError(f"teams[{index}].latitude is outside [-90, 90]")
        if not np.isfinite(longitude) or not -180 <= longitude <= 180:
            raise TravelContractError(f"teams[{index}].longitude is outside [-180, 180]")

        bundles = team["available_bundles"]
        if not isinstance(bundles, list) or not bundles:
            raise TravelContractError(f"teams[{index}].available_bundles must be a nonempty list")
        bundle_ids = [_nonblank(value, f"teams[{index}].available_bundles") for value in bundles]
        if len(bundle_ids) != len(set(bundle_ids)):
            raise TravelContractError(f"teams[{index}].available_bundles contains duplicates")

        drive = _positive_number(
            team["max_daily_drive_minutes"], f"teams[{index}].max_daily_drive_minutes"
        )
        work = _positive_number(
            team["max_daily_work_minutes"], f"teams[{index}].max_daily_work_minutes"
        )
        if drive > work:
            raise TravelContractError(
                f"teams[{index}].max_daily_drive_minutes cannot exceed max_daily_work_minutes"
            )
        records.append(
            {
                "team_id": team_id,
                "home_base": home_base,
                "latitude": latitude,
                "longitude": longitude,
                "available_bundles": tuple(bundle_ids),
                "vehicle_type": vehicle_type,
                "max_daily_drive_minutes": drive,
                "max_daily_work_minutes": work,
            }
        )

    frame = pd.DataFrame(records, columns=TEAM_FIELDS)
    if not frame.empty and frame["team_id"].duplicated().any():
        duplicates = frame.loc[frame["team_id"].duplicated(keep=False), "team_id"].unique().tolist()
        raise TravelContractError(f"team_id must be unique; duplicate examples: {duplicates[:5]}")
    return frame


def validate_mobile_team_bases(
    payload: Mapping[str, Any],
    *,
    source_path: Path | None = None,
) -> MobileTeamBaseContract:
    """Validate a future mobile-team-base YAML payload.

    A ``CONFIRMED`` payload requires both a source reference and a verification
    date.  ``UNCONFIRMED`` may preserve tentative rows for discussion, but the
    returned contract always disables their use in a final objective.
    """

    data = _require_mapping(payload, "mobile team base payload")
    version = str(data.get("schema_version", "")).strip()
    if version != MOBILE_TEAM_BASE_SCHEMA_VERSION:
        raise TravelContractError(
            f"schema_version must be {MOBILE_TEAM_BASE_SCHEMA_VERSION!r}; got {version!r}"
        )
    status = str(data.get("evidence_status", "")).strip().upper()
    if status not in BASE_EVIDENCE_STATUSES:
        raise TravelContractError(
            f"evidence_status must be one of {sorted(BASE_EVIDENCE_STATUSES)}"
        )
    teams = _validate_team_rows(data.get("teams", []))

    source_reference_raw = data.get("source_reference")
    verification_date_raw = data.get("verification_date")
    source_reference = None
    verification_date = None
    if status == "CONFIRMED":
        if teams.empty:
            raise TravelContractError("CONFIRMED base evidence requires at least one team")
        source_reference = _nonblank(source_reference_raw, "source_reference")
        verification_date = _canonical_iso_date(verification_date_raw, "verification_date")
        enabled = True
        reason = "confirmed operational team bases may accept separately verified network travel times"
    else:
        if source_reference_raw is not None and str(source_reference_raw).strip():
            source_reference = str(source_reference_raw).strip()
        if verification_date_raw is not None and str(verification_date_raw).strip():
            verification_date = _canonical_iso_date(verification_date_raw, "verification_date")
        enabled = False
        reason = "team home-base evidence is UNCONFIRMED; final travel objective is disabled"

    return MobileTeamBaseContract(
        schema_version=version,
        evidence_status=status,
        teams=teams,
        source_reference=source_reference,
        verification_date=verification_date,
        source_path=source_path.resolve() if source_path is not None else None,
        final_travel_objective_enabled=enabled,
        reason=reason,
    )


def load_mobile_team_bases(path: Path) -> MobileTeamBaseContract:
    """Load base evidence, returning an explicit UNCONFIRMED gate if absent."""

    resolved = path.resolve()
    if not resolved.exists():
        return MobileTeamBaseContract(
            schema_version=MOBILE_TEAM_BASE_SCHEMA_VERSION,
            evidence_status="UNCONFIRMED",
            teams=pd.DataFrame(columns=TEAM_FIELDS),
            source_reference=None,
            verification_date=None,
            source_path=resolved,
            final_travel_objective_enabled=False,
            reason="mobile_team_bases.yaml is absent; no operational team home base is evidenced",
        )
    if not resolved.is_file():
        raise TravelContractError(f"mobile team base path is not a file: {resolved}")
    try:
        payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise TravelContractError(f"could not parse mobile team base YAML: {resolved}") from exc
    return validate_mobile_team_bases(payload, source_path=resolved)


def _explicit_bool(series: pd.Series, label: str) -> pd.Series:
    def parse(value: Any) -> bool:
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, str) and value.strip().upper() in {"TRUE", "FALSE"}:
            return value.strip().upper() == "TRUE"
        raise TravelContractError(f"{label} must contain explicit booleans")

    return series.map(parse).astype(bool)


def prepare_final_travel_objective(
    venues: pd.DataFrame,
    base_contract: MobileTeamBaseContract,
    *,
    travel_column: str = "travel_minutes",
    evidence_status_column: str = "travel_evidence_status",
    diagnostic_only_column: str = "diagnostic_only",
) -> pd.DataFrame:
    """Gate candidate travel values before any final optimizer can use them.

    Under ``UNCONFIRMED``, both the requested source column and
    ``final_travel_minutes`` are forced to NaN.  Under ``CONFIRMED``, travel
    values additionally need the explicit ``CONFIRMED_TEAM_BASE_NETWORK``
    provenance marker and must not be diagnostic rows.
    """

    if "venue_id" not in venues.columns:
        raise TravelContractError("venues is missing venue_id")
    if venues["venue_id"].isna().any() or venues["venue_id"].astype("string").str.strip().eq("").any():
        raise TravelContractError("venues.venue_id must be complete")
    if venues["venue_id"].astype("string").duplicated().any():
        raise TravelContractError("venues.venue_id must be unique")

    out = venues.copy()
    if not base_contract.final_travel_objective_enabled:
        out[travel_column] = np.nan
        out["final_travel_minutes"] = np.nan
        out["final_travel_objective_enabled"] = False
        out["final_travel_input_status"] = "DISABLED_UNCONFIRMED_TEAM_BASE"
        return out

    missing = [
        column
        for column in (travel_column, evidence_status_column)
        if column not in out.columns
    ]
    if missing:
        raise TravelContractError(f"confirmed travel input is missing columns: {missing}")
    if diagnostic_only_column in out.columns:
        diagnostic = _explicit_bool(out[diagnostic_only_column], diagnostic_only_column)
        if diagnostic.any():
            raise TravelContractError("diagnostic_only travel can never be promoted to a final objective")
    status = out[evidence_status_column].astype("string").str.strip().str.upper()
    if not status.eq(CONFIRMED_TRAVEL_EVIDENCE_STATUS).all():
        raise TravelContractError(
            "confirmed travel input requires CONFIRMED_TEAM_BASE_NETWORK provenance on every venue"
        )
    travel = pd.to_numeric(out[travel_column], errors="coerce")
    if travel.isna().any() or (~np.isfinite(travel)).any() or travel.lt(0).any():
        raise TravelContractError("confirmed final travel minutes must be complete, finite, and nonnegative")
    out[travel_column] = travel.astype(float)
    out["final_travel_minutes"] = travel.astype(float)
    out["final_travel_objective_enabled"] = True
    out["final_travel_input_status"] = "ENABLED_CONFIRMED_TEAM_BASE_NETWORK"
    return out


def _validated_diagnostic_minutes(series: pd.Series, label: str) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    invalid = values.notna() & ((~np.isfinite(values)) | values.lt(0))
    if invalid.any():
        raise TravelContractError(f"{label} contains negative or non-finite drive minutes")
    return values.astype(float)


def build_diagnostic_travel_scenarios(venues: pd.DataFrame) -> pd.DataFrame:
    """Build Cheongju, Chungju, and nearest-dual-base sensitivity rows.

    The output is permanently marked ``diagnostic_only=True`` and
    ``final_promotion_allowed=False``.  These labels are invariant, independent
    of whether the diagnostic travel values are complete.
    """

    required = [
        "venue_id",
        DIAGNOSTIC_SCENARIOS["cheongju_medical_center"]["column"],
        DIAGNOSTIC_SCENARIOS["chungju_medical_center"]["column"],
    ]
    missing = [column for column in required if column not in venues.columns]
    if missing:
        raise TravelContractError(f"diagnostic venue input is missing columns: {missing}")
    if venues["venue_id"].isna().any() or venues["venue_id"].astype("string").duplicated().any():
        raise TravelContractError("diagnostic venues require unique, non-null venue_id values")

    venue_ids = venues["venue_id"].astype("string")
    cheongju = _validated_diagnostic_minutes(
        venues[DIAGNOSTIC_SCENARIOS["cheongju_medical_center"]["column"]],
        "Cheongju diagnostic travel",
    )
    chungju = _validated_diagnostic_minutes(
        venues[DIAGNOSTIC_SCENARIOS["chungju_medical_center"]["column"]],
        "Chungju diagnostic travel",
    )
    dual = pd.concat([cheongju, chungju], axis=1).min(axis=1, skipna=True)
    dual[pd.concat([cheongju, chungju], axis=1).isna().all(axis=1)] = np.nan

    assigned = pd.Series(pd.NA, index=venues.index, dtype="string")
    assigned.loc[cheongju.notna() & (chungju.isna() | cheongju.le(chungju))] = (
        "CHEONGJU_MEDICAL_CENTER"
    )
    assigned.loc[chungju.notna() & (cheongju.isna() | chungju.lt(cheongju))] = (
        "CHUNGJU_MEDICAL_CENTER"
    )

    scenarios = [
        ("cheongju_medical_center", cheongju, "CHEONGJU_MEDICAL_CENTER"),
        ("chungju_medical_center", chungju, "CHUNGJU_MEDICAL_CENTER"),
        ("dual_base_nearest", dual, assigned),
    ]
    frames: list[pd.DataFrame] = []
    for scenario, minutes, base in scenarios:
        frame = pd.DataFrame(
            {
                "venue_id": venue_ids,
                "travel_scenario": scenario,
                "travel_minutes": minutes,
                "assigned_diagnostic_base": base,
                "travel_value_status": np.where(minutes.notna(), "AVAILABLE", "UNAVAILABLE"),
                "travel_evidence_status": DIAGNOSTIC_TRAVEL_EVIDENCE_STATUS,
                "diagnostic_only": True,
                "final_promotion_allowed": False,
            }
        )
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    return validate_diagnostic_travel_scenarios(result)


def validate_diagnostic_travel_scenarios(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate that diagnostic travel has not acquired final semantics."""

    required = [
        "venue_id",
        "travel_scenario",
        "travel_minutes",
        "travel_evidence_status",
        "diagnostic_only",
        "final_promotion_allowed",
    ]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise TravelContractError(f"diagnostic travel frame is missing columns: {missing}")
    out = frame.copy()
    diagnostic = _explicit_bool(out["diagnostic_only"], "diagnostic_only")
    promotable = _explicit_bool(out["final_promotion_allowed"], "final_promotion_allowed")
    if not diagnostic.all() or promotable.any():
        raise TravelContractError("diagnostic travel must remain diagnostic-only and non-promotable")
    status = out["travel_evidence_status"].astype("string").str.strip().str.upper()
    if not status.eq(DIAGNOSTIC_TRAVEL_EVIDENCE_STATUS).all():
        raise TravelContractError("diagnostic travel has an invalid provenance marker")
    if not out["travel_scenario"].isin(DIAGNOSTIC_SCENARIOS).all():
        raise TravelContractError("diagnostic travel contains an unknown scenario")
    if out["venue_id"].isna().any() or out["venue_id"].astype("string").str.strip().eq("").any():
        raise TravelContractError("diagnostic travel venue_id values must be complete")
    key_duplicates = out.duplicated(["venue_id", "travel_scenario"], keep=False)
    if key_duplicates.any():
        raise TravelContractError("diagnostic travel venue_id x scenario keys must be unique")
    expected = frozenset(DIAGNOSTIC_SCENARIOS)
    observed = out.groupby("venue_id")["travel_scenario"].agg(lambda values: frozenset(values))
    if not observed.map(lambda values: values == expected).all():
        raise TravelContractError("each diagnostic venue must have all three assumed-base scenarios")
    out["travel_minutes"] = _validated_diagnostic_minutes(
        out["travel_minutes"], "diagnostic travel"
    )
    return out


def beneficiary_visit_location_equity_kpis(
    selected_visits: pd.DataFrame,
    beneficiary_coverage: pd.DataFrame,
    *,
    sigungu_column: str = "sigungu",
    beneficiary_total_column: str = "beneficiary_population",
    beneficiary_covered_column: str = "beneficiary_covered_population",
) -> dict[str, Any]:
    """Compute separate beneficiary and visit-location equity KPIs.

    The beneficiary table may be grid-level; values are summed by sigungu.  The
    complete beneficiary sigungu universe is also used for visit-location Gini,
    so locations with zero selected visits are not silently dropped.
    """

    if sigungu_column not in selected_visits.columns:
        raise TravelContractError(f"selected_visits is missing {sigungu_column}")
    required = [sigungu_column, beneficiary_total_column, beneficiary_covered_column]
    missing = [column for column in required if column not in beneficiary_coverage.columns]
    if missing:
        raise TravelContractError(f"beneficiary_coverage is missing columns: {missing}")

    coverage = beneficiary_coverage[required].copy()
    if coverage[sigungu_column].isna().any() or coverage[sigungu_column].astype("string").str.strip().eq("").any():
        raise TravelContractError("beneficiary sigungu values must be complete")
    coverage[sigungu_column] = coverage[sigungu_column].astype("string").str.strip()
    for column in (beneficiary_total_column, beneficiary_covered_column):
        coverage[column] = pd.to_numeric(coverage[column], errors="coerce")
        if (
            coverage[column].isna().any()
            or (~np.isfinite(coverage[column])).any()
            or coverage[column].lt(0).any()
        ):
            raise TravelContractError(f"{column} must be finite and nonnegative")
    grouped = coverage.groupby(sigungu_column, sort=True, as_index=True)[
        [beneficiary_total_column, beneficiary_covered_column]
    ].sum()
    if grouped.empty or grouped[beneficiary_total_column].le(0).any():
        raise TravelContractError("every beneficiary sigungu needs positive total population")
    if grouped[beneficiary_covered_column].gt(grouped[beneficiary_total_column] + 1e-9).any():
        raise TravelContractError("covered beneficiary population cannot exceed total population")

    visit_sigungu = selected_visits[sigungu_column]
    if visit_sigungu.isna().any() or visit_sigungu.astype("string").str.strip().eq("").any():
        raise TravelContractError("selected visit sigungu values must be complete")
    visit_sigungu = visit_sigungu.astype("string").str.strip()
    unknown = sorted(set(visit_sigungu) - set(grouped.index.astype(str)))
    if unknown:
        raise TravelContractError(
            f"selected visits reference sigungu outside beneficiary universe: {unknown[:5]}"
        )
    visit_counts = visit_sigungu.value_counts().reindex(grouped.index, fill_value=0).astype(int)
    total_visits = int(visit_counts.sum())
    visit_shares = (
        visit_counts.astype(float) / total_visits
        if total_visits > 0
        else pd.Series(0.0, index=visit_counts.index)
    )
    coverage_ratio = grouped[beneficiary_covered_column] / grouped[beneficiary_total_column]

    return {
        "visit_location_sigungu_count": int((visit_counts > 0).sum()),
        "visit_location_distribution": {str(k): int(v) for k, v in visit_counts.items()},
        "visit_location_share_distribution": {str(k): float(v) for k, v in visit_shares.items()},
        "visit_location_gini": float(gini(visit_counts.to_numpy(dtype=float))),
        "beneficiary_sigungu_coverage": {str(k): float(v) for k, v in coverage_ratio.items()},
        "beneficiary_min_sigungu_coverage": float(coverage_ratio.min()),
        "beneficiary_mean_sigungu_coverage": float(coverage_ratio.mean()),
        "beneficiary_coverage_gini": float(gini(coverage_ratio.to_numpy(dtype=float))),
        "beneficiary_total_population": float(grouped[beneficiary_total_column].sum()),
        "beneficiary_covered_population": float(grouped[beneficiary_covered_column].sum()),
    }


__all__ = [
    "BASE_EVIDENCE_STATUSES",
    "CONFIRMED_TRAVEL_EVIDENCE_STATUS",
    "DIAGNOSTIC_SCENARIOS",
    "DIAGNOSTIC_TRAVEL_EVIDENCE_STATUS",
    "MOBILE_TEAM_BASE_SCHEMA_VERSION",
    "MobileTeamBaseContract",
    "TEAM_FIELDS",
    "TravelContractError",
    "beneficiary_visit_location_equity_kpis",
    "build_diagnostic_travel_scenarios",
    "load_mobile_team_bases",
    "prepare_final_travel_objective",
    "validate_diagnostic_travel_scenarios",
    "validate_mobile_team_bases",
]
