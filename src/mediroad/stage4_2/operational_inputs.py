from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .contracts import (
    CALENDAR_REQUIRED_COLUMNS,
    TEAM_BASE_REQUIRED_COLUMNS,
    TEAM_CAPABILITY_REQUIRED_COLUMNS,
    TRAVEL_REQUIRED_COLUMNS,
    VEHICLE_REQUIRED_COLUMNS,
)
from .errors import ContractError
from .utils import ensure_columns, normalize_bool, parse_date, read_table, sha256_file


@dataclass
class OperationalAudit:
    ready: bool
    summaries: dict[str, Any]
    blockers: list[str]
    normalized_tables: dict[str, pd.DataFrame]


def write_operational_templates(output_dir: Path, bundle_ids: list[str]) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    base_payload = {
        "status": "UNCONFIRMED_TEMPLATE",
        "effective_date": None,
        "source_reference": None,
        "teams": [
            {
                "team_id": "TEAM_01",
                "base_id": "BASE_01",
                "base_name": "REPLACE_WITH_CONFIRMED_BASE",
                "latitude": None,
                "longitude": None,
                "confirmed": False,
                "source_reference": None,
                "effective_date": None,
            }
        ],
    }
    base_path = output_dir / "mobile_team_bases.template.yaml"
    base_path.write_text(yaml.safe_dump(base_payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
    paths["team_bases"] = base_path

    capability = pd.DataFrame(
        [
            {
                "team_id": "TEAM_01",
                "bundle_id": bundle,
                "capable": "UNKNOWN",
                "confirmed": False,
                "source_reference": "",
            }
            for bundle in bundle_ids
        ]
    )
    cap_path = output_dir / "team_bundle_capability.template.csv"
    capability.to_csv(cap_path, index=False, encoding="utf-8-sig")
    paths["team_capability"] = cap_path

    vehicles = pd.DataFrame(
        [
            {
                "vehicle_id": "VEHICLE_01",
                "team_id": "TEAM_01",
                "base_id": "BASE_01",
                "vehicle_type": "REPLACE",
                "confirmed": False,
                "max_daily_drive_minutes": "",
                "max_daily_work_minutes": "",
                "source_reference": "",
                "effective_date": "",
            }
        ]
    )
    vehicle_path = output_dir / "vehicles.template.csv"
    vehicles.to_csv(vehicle_path, index=False, encoding="utf-8-sig")
    paths["vehicles"] = vehicle_path

    calendar = pd.DataFrame(
        [
            {
                "resource_type": "TEAM",
                "resource_id": "TEAM_01",
                "date": "YYYY-MM-DD",
                "available": "UNKNOWN",
                "source_reference": "",
            },
            {
                "resource_type": "VEHICLE",
                "resource_id": "VEHICLE_01",
                "date": "YYYY-MM-DD",
                "available": "UNKNOWN",
                "source_reference": "",
            },
        ]
    )
    cal_path = output_dir / "resource_calendar.template.csv"
    calendar.to_csv(cal_path, index=False, encoding="utf-8-sig")
    paths["calendar"] = cal_path

    venue_calendar = pd.DataFrame(
        [{"venue_id": "REPLACE_VENUE_ID", "date": "YYYY-MM-DD", "available": "UNKNOWN", "source_reference": ""}]
    )
    venue_cal_path = output_dir / "venue_calendar.template.csv"
    venue_calendar.to_csv(venue_cal_path, index=False, encoding="utf-8-sig")
    paths["venue_calendar"] = venue_cal_path

    travel = pd.DataFrame(
        [
            {
                "team_id": "TEAM_01",
                "venue_id": "REPLACE_VERIFIED_VENUE_ID",
                "travel_minutes": "",
                "routing_method": "OSM_STATIC_FREE_FLOW",
                "confirmed": False,
                "source_reference": "",
                "effective_date": "",
            }
        ]
    )
    travel_path = output_dir / "team_venue_travel.template.csv"
    travel.to_csv(travel_path, index=False, encoding="utf-8-sig")
    paths["travel"] = travel_path

    existing = pd.DataFrame(
        [
            {
                "service_id": "REPLACE",
                "date": "YYYY-MM-DD",
                "venue_id": "",
                "admin_code": "",
                "bundle_id": "",
                "status": "KNOWN_OR_UNKNOWN",
                "source_reference": "",
            }
        ]
    )
    existing_path = output_dir / "existing_service_calendar.template.csv"
    existing.to_csv(existing_path, index=False, encoding="utf-8-sig")
    paths["existing_services"] = existing_path
    return paths


def _load_team_bases(path: Path) -> pd.DataFrame:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("teams"), list):
        raise ContractError("mobile_team_bases.yaml must contain a teams list")
    frame = pd.DataFrame(data["teams"])
    ensure_columns(frame, TEAM_BASE_REQUIRED_COLUMNS, label="mobile team bases")
    return frame


def audit_operational_inputs(paths: dict[str, Path], verified_venue_ids: set[str]) -> OperationalAudit:
    blockers: list[str] = []
    normalized: dict[str, pd.DataFrame] = {}
    summaries: dict[str, Any] = {}

    for key, path in paths.items():
        if not path.exists():
            blockers.append(f"MISSING_FILE:{key}:{path}")

    if blockers:
        return OperationalAudit(False, summaries, blockers, normalized)

    bases = _load_team_bases(paths["team_bases"])
    bases["confirmed"] = bases["confirmed"].map(normalize_bool).fillna(False)
    bases["latitude"] = pd.to_numeric(bases["latitude"], errors="coerce")
    bases["longitude"] = pd.to_numeric(bases["longitude"], errors="coerce")
    bases["effective_ts"] = bases["effective_date"].map(parse_date)
    confirmed_bases = bases["confirmed"] & bases["latitude"].between(30, 40) & bases["longitude"].between(120, 135) & bases["effective_ts"].notna()
    if not confirmed_bases.all() or len(bases) == 0:
        blockers.append("TEAM_BASES_NOT_FULLY_CONFIRMED")
    normalized["team_bases"] = bases
    confirmed_team_ids = set(bases.loc[confirmed_bases, "team_id"].astype(str))
    confirmed_base_ids = set(bases.loc[confirmed_bases, "base_id"].astype(str))
    summaries["confirmed_team_count"] = int(confirmed_bases.sum())

    capability = read_table(paths["team_capability"])
    ensure_columns(capability, TEAM_CAPABILITY_REQUIRED_COLUMNS, label="team bundle capability")
    capability["capable"] = capability["capable"].map(normalize_bool)
    capability["confirmed"] = capability["confirmed"].map(normalize_bool).fillna(False)
    capability["team_id"] = capability["team_id"].astype(str)
    capability["bundle_id"] = capability["bundle_id"].astype(str)
    if capability["capable"].isna().any() or not capability["confirmed"].all():
        blockers.append("TEAM_BUNDLE_CAPABILITY_NOT_CONFIRMED")
    unknown_teams = set(capability["team_id"]) - confirmed_team_ids
    if unknown_teams:
        blockers.append(f"CAPABILITY_REFERENCES_UNCONFIRMED_TEAMS:{len(unknown_teams)}")
    capable_rows = capability.loc[capability["confirmed"] & capability["capable"].fillna(False)].copy()
    available_bundle_ids = sorted(capable_rows["bundle_id"].unique())
    if not available_bundle_ids:
        blockers.append("NO_CONFIRMED_CAPABLE_SERVICE_BUNDLE")
    normalized["team_capability"] = capability
    summaries["available_bundle_ids"] = available_bundle_ids
    summaries["available_bundle_count"] = int(len(available_bundle_ids))

    vehicles = read_table(paths["vehicles"])
    ensure_columns(vehicles, VEHICLE_REQUIRED_COLUMNS, label="vehicles")
    vehicles["confirmed"] = vehicles["confirmed"].map(normalize_bool).fillna(False)
    vehicles["effective_ts"] = vehicles["effective_date"].map(parse_date)
    for col in ["max_daily_drive_minutes", "max_daily_work_minutes"]:
        vehicles[col] = pd.to_numeric(vehicles[col], errors="coerce")
    vehicles["team_id"] = vehicles["team_id"].astype(str)
    vehicles["base_id"] = vehicles["base_id"].astype(str)
    vehicle_refs_valid = set(vehicles["team_id"]).issubset(confirmed_team_ids) and set(vehicles["base_id"]).issubset(confirmed_base_ids)
    if len(vehicles) == 0 or not vehicles["confirmed"].all() or vehicles[["max_daily_drive_minutes", "max_daily_work_minutes"]].isna().any().any() or vehicles["effective_ts"].isna().any() or not vehicle_refs_valid:
        blockers.append("VEHICLE_CONTRACT_NOT_CONFIRMED")
    normalized["vehicles"] = vehicles
    confirmed_vehicle_ids = set(vehicles.loc[vehicles["confirmed"], "vehicle_id"].astype(str))

    calendar = read_table(paths["calendar"])
    ensure_columns(calendar, CALENDAR_REQUIRED_COLUMNS, label="resource calendar")
    calendar["available"] = calendar["available"].map(normalize_bool)
    calendar["date_ts"] = calendar["date"].map(parse_date)
    calendar["resource_type"] = calendar["resource_type"].astype(str).str.upper()
    calendar["resource_id"] = calendar["resource_id"].astype(str)
    valid_resource = (
        (calendar["resource_type"].eq("TEAM") & calendar["resource_id"].isin(confirmed_team_ids))
        | (calendar["resource_type"].eq("VEHICLE") & calendar["resource_id"].isin(confirmed_vehicle_ids))
    )
    available_true = calendar["available"].fillna(False)
    required_resources = confirmed_team_ids | confirmed_vehicle_ids
    resources_with_availability = set(calendar.loc[available_true, "resource_id"])
    if calendar["available"].isna().any() or calendar["date_ts"].isna().any() or not valid_resource.all() or not required_resources.issubset(resources_with_availability):
        blockers.append("RESOURCE_CALENDAR_INCOMPLETE")
    normalized["calendar"] = calendar

    venue_calendar = read_table(paths["venue_calendar"])
    ensure_columns(venue_calendar, ["venue_id", "date", "available", "source_reference"], label="venue calendar")
    venue_calendar["venue_id"] = venue_calendar["venue_id"].astype(str)
    venue_calendar["available"] = venue_calendar["available"].map(normalize_bool)
    venue_calendar["date_ts"] = venue_calendar["date"].map(parse_date)
    missing_verified = verified_venue_ids - set(venue_calendar["venue_id"])
    venue_with_available_date = set(venue_calendar.loc[venue_calendar["available"].fillna(False), "venue_id"])
    missing_available = verified_venue_ids - venue_with_available_date
    if missing_verified or missing_available or venue_calendar["available"].isna().any() or venue_calendar["date_ts"].isna().any():
        blockers.append(
            f"VENUE_CALENDAR_INCOMPLETE:{len(missing_verified)}_MISSING_ROWS:{len(missing_available)}_WITHOUT_AVAILABLE_DATE"
        )
    normalized["venue_calendar"] = venue_calendar

    travel = read_table(paths["travel"])
    ensure_columns(travel, TRAVEL_REQUIRED_COLUMNS, label="team-venue travel")
    travel["team_id"] = travel["team_id"].astype(str)
    travel["venue_id"] = travel["venue_id"].astype(str)
    travel["confirmed"] = travel["confirmed"].map(normalize_bool).fillna(False)
    travel["travel_minutes"] = pd.to_numeric(travel["travel_minutes"], errors="coerce")
    travel["effective_ts"] = travel["effective_date"].map(parse_date)
    travel_verified = travel.loc[travel["venue_id"].isin(verified_venue_ids)].copy()
    confirmed_travel = travel_verified.loc[travel_verified["confirmed"] & travel_verified["team_id"].isin(confirmed_team_ids)].copy()
    missing_travel = verified_venue_ids - set(confirmed_travel["venue_id"])
    invalid_travel_rows = (
        confirmed_travel["travel_minutes"].isna()
        | confirmed_travel["travel_minutes"].lt(0)
        | confirmed_travel["effective_ts"].isna()
    )
    if missing_travel or invalid_travel_rows.any():
        blockers.append(f"CONFIRMED_TEAM_VENUE_TRAVEL_INCOMPLETE:{len(missing_travel)}_VENUES_MISSING")
    normalized["travel"] = travel

    summaries.update(
        {
            "vehicle_count": int(len(vehicles)),
            "calendar_rows": int(len(calendar)),
            "venue_calendar_rows": int(len(venue_calendar)),
            "travel_rows": int(len(travel)),
            "verified_venue_count": int(len(verified_venue_ids)),
        }
    )
    return OperationalAudit(ready=not blockers, summaries=summaries, blockers=blockers, normalized_tables=normalized)


def build_operational_manifest(paths: dict[str, Path]) -> dict[str, Any]:
    return {
        "files": {
            key: {"path": str(path), "exists": path.exists(), "sha256": sha256_file(path) if path.exists() else None}
            for key, path in paths.items()
        }
    }
