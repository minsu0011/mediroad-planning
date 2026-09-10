"""Fail-closed operational evidence contract for Stage 4.

The computational Stage 4 plans are spatial plans.  They must not become an
operational plan merely because several independent input tables are nonempty.
This module proves a stronger statement: every requested visit has a single,
evidence-backed ``team x bundle x vehicle x venue x date`` assignment and all
requested visits can be assigned together without double-booking a team,
vehicle, or venue.

Absence is never interpreted as availability.  In particular, an absent row in
the existing-service table is *not* evidence that no conflict exists.  A
confirmed, exact negative conflict check is required for every feasible tuple.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


OPERATIONAL_CONTRACT_SCHEMA_VERSION = "1.0"

BASE_COLUMNS = (
    "team_id",
    "base_id",
    "base_name",
    "latitude",
    "longitude",
    "confirmed",
    "source_reference",
    "effective_date",
)
CAPABILITY_COLUMNS = (
    "team_id",
    "bundle_id",
    "capable",
    "confirmed",
    "required_vehicle_type",
    "minimum_vehicle_capacity",
    "source_reference",
    "effective_date",
)
VEHICLE_COLUMNS = (
    "vehicle_id",
    "team_id",
    "base_id",
    "vehicle_type",
    "capacity",
    "operational",
    "confirmed",
    "source_reference",
    "effective_date",
)
TEAM_CALENDAR_COLUMNS = (
    "team_id",
    "date",
    "available",
    "confirmed",
    "source_reference",
)
RESOURCE_CALENDAR_COLUMNS = (
    "resource_type",
    "resource_id",
    "date",
    "available",
    "confirmed",
    "source_reference",
)
VENUE_CALENDAR_COLUMNS = (
    "venue_id",
    "date",
    "available",
    "confirmed",
    "source_reference",
)
TRAVEL_COLUMNS = (
    "team_id",
    "venue_id",
    "reachable",
    "distance_km",
    "travel_minutes_proxy",
    "graph_sha",
    "confirmed",
    "source_reference",
    "effective_date",
)
EXISTING_SERVICE_CONFLICT_COLUMNS = (
    "team_id",
    "vehicle_id",
    "venue_id",
    "date",
    "conflict",
    "confirmed",
    "source_reference",
)
VISIT_COLUMNS = (
    "visit_id",
    "venue_id",
    "bundle_id",
    "required_vehicle_capacity",
)

TABLE_SCHEMAS: Mapping[str, tuple[str, ...]] = {
    "bases": BASE_COLUMNS,
    "team_bundle_capability": CAPABILITY_COLUMNS,
    "vehicles": VEHICLE_COLUMNS,
    "team_calendar": TEAM_CALENDAR_COLUMNS,
    "resource_calendar": RESOURCE_CALENDAR_COLUMNS,
    "venue_calendar": VENUE_CALENDAR_COLUMNS,
    "travel": TRAVEL_COLUMNS,
    "existing_service_conflicts": EXISTING_SERVICE_CONFLICT_COLUMNS,
}

_TRUE_VALUES = frozenset({"TRUE", "1", "YES", "Y", "예", "네"})
_FALSE_VALUES = frozenset({"FALSE", "0", "NO", "N", "아니오", "아니요"})
_UNKNOWN_VALUES = frozenset({"", "UNKNOWN", "NA", "N/A", "NONE", "NULL", "미확인"})
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class OperationalContractError(ValueError):
    """Raised when an operational plan cannot be proven from confirmed inputs."""


@dataclass(frozen=True)
class OperationalTables:
    """The eight evidence tables needed by the operational contract."""

    bases: pd.DataFrame
    team_bundle_capability: pd.DataFrame
    vehicles: pd.DataFrame
    team_calendar: pd.DataFrame
    resource_calendar: pd.DataFrame
    venue_calendar: pd.DataFrame
    travel: pd.DataFrame
    existing_service_conflicts: pd.DataFrame

    @classmethod
    def from_mapping(cls, value: Mapping[str, pd.DataFrame]) -> "OperationalTables":
        missing = [name for name in TABLE_SCHEMAS if name not in value]
        if missing:
            raise OperationalContractError(f"Missing operational tables: {missing}")
        invalid = [name for name in TABLE_SCHEMAS if not isinstance(value[name], pd.DataFrame)]
        if invalid:
            raise OperationalContractError(f"Operational tables must be DataFrames: {invalid}")
        return cls(**{name: value[name] for name in TABLE_SCHEMAS})


@dataclass
class OperationalContractAudit:
    """Machine-readable result of a fail-closed operational contract audit."""

    ready: bool
    blockers: list[str]
    summary: dict[str, Any]
    feasible_options: pd.DataFrame = field(default_factory=pd.DataFrame)
    selected_assignments: pd.DataFrame = field(default_factory=pd.DataFrame)
    normalized_tables: dict[str, pd.DataFrame] = field(default_factory=dict, repr=False)

    def raise_if_not_ready(self) -> "OperationalContractAudit":
        if not self.ready:
            rendered = "; ".join(self.blockers) if self.blockers else "UNKNOWN_BLOCKER"
            raise OperationalContractError(f"Operational contract is not ready: {rendered}")
        return self

    def as_dict(self) -> dict[str, Any]:
        return {
            "ready": bool(self.ready),
            "blockers": list(self.blockers),
            "summary": dict(self.summary),
            "feasible_option_count": int(len(self.feasible_options)),
            "selected_assignment_count": int(len(self.selected_assignments)),
        }


def _as_text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def _parse_bool(value: Any) -> bool | None:
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().upper()
    if text in _TRUE_VALUES:
        return True
    if text in _FALSE_VALUES:
        return False
    return None


def _normalize_bool_column(
    frame: pd.DataFrame,
    column: str,
    *,
    table: str,
    blockers: list[str],
) -> None:
    parsed = frame[column].map(_parse_bool)
    invalid = int(parsed.isna().sum())
    if invalid:
        blockers.append(f"INVALID_OR_UNKNOWN_BOOLEAN:{table}.{column}:{invalid}")
    frame[column] = parsed.astype("boolean")


def _normalize_date_column(
    frame: pd.DataFrame,
    column: str,
    *,
    table: str,
    blockers: list[str],
) -> None:
    raw = _as_text(frame[column])
    parsed = pd.to_datetime(raw, format="%Y-%m-%d", errors="coerce")
    canonical = raw.str.match(_ISO_DATE_RE)
    invalid = int((parsed.isna() | ~canonical).sum())
    if invalid:
        blockers.append(f"INVALID_OR_UNKNOWN_DATE:{table}.{column}:{invalid}")
    frame[column] = parsed.dt.normalize()


def _require_nonblank(
    frame: pd.DataFrame,
    columns: Iterable[str],
    *,
    table: str,
    blockers: list[str],
) -> None:
    for column in columns:
        frame[column] = _as_text(frame[column])
        blank = frame[column].eq("")
        if blank.any():
            blockers.append(f"BLANK_REQUIRED_VALUE:{table}.{column}:{int(blank.sum())}")
        upper = frame[column].str.upper()
        placeholder = (
            upper.isin(_UNKNOWN_VALUES - {""})
            | upper.str.match(r"^(?:UNKNOWN|REPLACE)(?:$|[_\s-])")
        ) & ~blank
        if placeholder.any():
            blockers.append(
                f"UNKNOWN_REQUIRED_VALUE:{table}.{column}:{int(placeholder.sum())}"
            )


def _require_unique(
    frame: pd.DataFrame,
    columns: list[str],
    *,
    table: str,
    blockers: list[str],
) -> None:
    duplicates = int(frame.duplicated(columns, keep=False).sum())
    if duplicates:
        blockers.append(f"DUPLICATE_KEY:{table}:{'+'.join(columns)}:{duplicates}")


def _numeric(
    frame: pd.DataFrame,
    column: str,
    *,
    table: str,
    blockers: list[str],
    minimum: float,
    strictly_positive: bool = False,
) -> None:
    values = pd.to_numeric(frame[column], errors="coerce")
    invalid = ~np.isfinite(values)
    if strictly_positive:
        invalid |= values.le(minimum)
    else:
        invalid |= values.lt(minimum)
    count = int(invalid.sum())
    if count:
        blockers.append(f"INVALID_NUMERIC:{table}.{column}:{count}")
    frame[column] = values


def _normalize_table(
    name: str,
    raw: pd.DataFrame,
    blockers: list[str],
) -> pd.DataFrame:
    required = TABLE_SCHEMAS[name]
    frame = raw.copy(deep=True)
    missing = [column for column in required if column not in frame.columns]
    if missing:
        blockers.append(f"MISSING_COLUMNS:{name}:{','.join(missing)}")
        return pd.DataFrame(columns=required)
    frame = frame.loc[:, list(required)].copy()
    if frame.empty:
        blockers.append(f"EMPTY_TABLE:{name}")
        return frame

    source_columns = ["source_reference"]
    id_columns: dict[str, list[str]] = {
        "bases": ["team_id", "base_id", "base_name"],
        "team_bundle_capability": ["team_id", "bundle_id", "required_vehicle_type"],
        "vehicles": ["vehicle_id", "team_id", "base_id", "vehicle_type"],
        "team_calendar": ["team_id"],
        "resource_calendar": ["resource_type", "resource_id"],
        "venue_calendar": ["venue_id"],
        "travel": ["team_id", "venue_id", "graph_sha"],
        "existing_service_conflicts": ["team_id", "vehicle_id", "venue_id"],
    }
    _require_nonblank(frame, id_columns[name] + source_columns, table=name, blockers=blockers)

    bool_columns: dict[str, tuple[str, ...]] = {
        "bases": ("confirmed",),
        "team_bundle_capability": ("capable", "confirmed"),
        "vehicles": ("operational", "confirmed"),
        "team_calendar": ("available", "confirmed"),
        "resource_calendar": ("available", "confirmed"),
        "venue_calendar": ("available", "confirmed"),
        "travel": ("reachable", "confirmed"),
        "existing_service_conflicts": ("conflict", "confirmed"),
    }
    for column in bool_columns[name]:
        _normalize_bool_column(frame, column, table=name, blockers=blockers)

    if "effective_date" in frame.columns:
        _normalize_date_column(frame, "effective_date", table=name, blockers=blockers)
    if "date" in frame.columns:
        _normalize_date_column(frame, "date", table=name, blockers=blockers)

    if name == "bases":
        _numeric(frame, "latitude", table=name, blockers=blockers, minimum=-90)
        _numeric(frame, "longitude", table=name, blockers=blockers, minimum=-180)
        invalid_geo = ~frame["latitude"].between(30, 40) | ~frame["longitude"].between(120, 135)
        if invalid_geo.any():
            blockers.append(f"BASE_COORDINATE_OUTSIDE_KOREA_BOUNDS:{int(invalid_geo.sum())}")
        _require_unique(frame, ["team_id"], table=name, blockers=blockers)
        _require_unique(frame, ["base_id"], table=name, blockers=blockers)
    elif name == "team_bundle_capability":
        _numeric(
            frame,
            "minimum_vehicle_capacity",
            table=name,
            blockers=blockers,
            minimum=0,
        )
        _require_unique(frame, ["team_id", "bundle_id"], table=name, blockers=blockers)
    elif name == "vehicles":
        _numeric(frame, "capacity", table=name, blockers=blockers, minimum=0, strictly_positive=True)
        _require_unique(frame, ["vehicle_id"], table=name, blockers=blockers)
    elif name == "team_calendar":
        _require_unique(frame, ["team_id", "date"], table=name, blockers=blockers)
    elif name == "resource_calendar":
        frame["resource_type"] = frame["resource_type"].str.upper()
        unsupported = ~frame["resource_type"].eq("VEHICLE")
        if unsupported.any():
            blockers.append(f"UNSUPPORTED_RESOURCE_TYPE:{int(unsupported.sum())}")
        _require_unique(frame, ["resource_type", "resource_id", "date"], table=name, blockers=blockers)
    elif name == "venue_calendar":
        _require_unique(frame, ["venue_id", "date"], table=name, blockers=blockers)
    elif name == "travel":
        frame["graph_sha"] = frame["graph_sha"].str.lower()
        invalid_sha = ~frame["graph_sha"].str.match(_SHA256_RE)
        if invalid_sha.any():
            blockers.append(f"INVALID_GRAPH_SHA:travel:{int(invalid_sha.sum())}")
        for column in ("distance_km", "travel_minutes_proxy"):
            values = pd.to_numeric(frame[column], errors="coerce")
            required_numeric = frame["reachable"].fillna(False)
            invalid = required_numeric & (~np.isfinite(values) | values.lt(0))
            if invalid.any():
                blockers.append(f"INVALID_REACHABLE_TRAVEL_VALUE:{column}:{int(invalid.sum())}")
            frame[column] = values
        _require_unique(frame, ["team_id", "venue_id"], table=name, blockers=blockers)
    elif name == "existing_service_conflicts":
        keys = ["team_id", "vehicle_id", "venue_id", "date"]
        contradictions = (
            frame.groupby(keys, dropna=False)["conflict"].nunique(dropna=False).gt(1)
        )
        if contradictions.any():
            blockers.append(f"CONTRADICTORY_CONFLICT_EVIDENCE:{int(contradictions.sum())}")
        _require_unique(frame, keys, table=name, blockers=blockers)
    return frame


def _normalize_visits(
    visits: pd.DataFrame,
    blockers: list[str],
    required_visit_count: int | None,
) -> pd.DataFrame:
    if not isinstance(visits, pd.DataFrame):
        blockers.append("VISIT_REQUIREMENTS_NOT_A_DATAFRAME")
        return pd.DataFrame(columns=VISIT_COLUMNS)
    missing = [column for column in VISIT_COLUMNS if column not in visits.columns]
    if missing:
        blockers.append(f"MISSING_COLUMNS:visit_requirements:{','.join(missing)}")
        return pd.DataFrame(columns=VISIT_COLUMNS)
    frame = visits.loc[:, list(VISIT_COLUMNS)].copy(deep=True)
    if frame.empty:
        blockers.append("EMPTY_TABLE:visit_requirements")
        return frame
    _require_nonblank(
        frame,
        ["visit_id", "venue_id", "bundle_id"],
        table="visit_requirements",
        blockers=blockers,
    )
    _numeric(
        frame,
        "required_vehicle_capacity",
        table="visit_requirements",
        blockers=blockers,
        minimum=0,
        strictly_positive=True,
    )
    _require_unique(frame, ["visit_id"], table="visit_requirements", blockers=blockers)
    if required_visit_count is not None and len(frame) != int(required_visit_count):
        blockers.append(f"VISIT_COUNT_MISMATCH:{len(frame)}:{int(required_visit_count)}")
    return frame


def _active(frame: pd.DataFrame, *columns: str) -> pd.DataFrame:
    mask = pd.Series(True, index=frame.index, dtype=bool)
    for column in columns:
        mask &= frame[column].fillna(False).astype(bool)
    return frame.loc[mask].copy()


def _vehicle_type_matches(actual: pd.Series, required: pd.Series) -> pd.Series:
    wanted = required.astype(str).str.strip().str.upper()
    return wanted.eq("ANY") | actual.astype(str).str.strip().str.upper().eq(wanted)


def _options_for_visit(
    visit: pd.Series,
    tables: Mapping[str, pd.DataFrame],
) -> tuple[pd.DataFrame, str | None]:
    venue_id = str(visit["venue_id"])
    bundle_id = str(visit["bundle_id"])
    visit_id = str(visit["visit_id"])
    requested_capacity = float(visit["required_vehicle_capacity"])

    bases = _active(tables["bases"], "confirmed")
    capability = _active(tables["team_bundle_capability"], "confirmed", "capable")
    capability = capability.loc[capability["bundle_id"].eq(bundle_id)].copy()
    capability = capability.merge(
        bases[["team_id", "base_id"]], on="team_id", how="inner", validate="many_to_one"
    )
    if capability.empty:
        return pd.DataFrame(), f"NO_CONFIRMED_CAPABLE_TEAM:{visit_id}:{bundle_id}"

    vehicles = _active(tables["vehicles"], "confirmed", "operational")
    candidates = capability.merge(
        vehicles,
        on=["team_id", "base_id"],
        how="inner",
        suffixes=("_capability", "_vehicle"),
        validate="one_to_many",
    )
    required_capacity = np.maximum(
        candidates["minimum_vehicle_capacity"].astype(float), requested_capacity
    )
    candidates = candidates.loc[
        candidates["capacity"].astype(float).ge(required_capacity)
        & _vehicle_type_matches(
            candidates["vehicle_type"], candidates["required_vehicle_type"]
        )
    ].copy()
    if candidates.empty:
        return pd.DataFrame(), f"NO_CONFIRMED_COMPATIBLE_VEHICLE:{visit_id}"

    travel = _active(tables["travel"], "confirmed", "reachable")
    travel = travel.loc[travel["venue_id"].eq(venue_id)].copy()
    candidates = candidates.merge(
        travel[
            [
                "team_id",
                "venue_id",
                "distance_km",
                "travel_minutes_proxy",
                "graph_sha",
            ]
        ],
        on="team_id",
        how="inner",
        validate="many_to_one",
    )
    if candidates.empty:
        return pd.DataFrame(), f"NO_CONFIRMED_REACHABLE_TEAM_VENUE_TRAVEL:{visit_id}"

    team_dates = _active(tables["team_calendar"], "confirmed", "available")
    vehicle_dates = _active(tables["resource_calendar"], "confirmed", "available")
    vehicle_dates = vehicle_dates.loc[vehicle_dates["resource_type"].eq("VEHICLE")].rename(
        columns={"resource_id": "vehicle_id"}
    )
    venue_dates = _active(tables["venue_calendar"], "confirmed", "available")
    venue_dates = venue_dates.loc[venue_dates["venue_id"].eq(venue_id)].copy()

    options = candidates.merge(
        team_dates[["team_id", "date"]], on="team_id", how="inner", validate="many_to_many"
    )
    options = options.merge(
        vehicle_dates[["vehicle_id", "date"]],
        on=["vehicle_id", "date"],
        how="inner",
        validate="many_to_one",
    )
    options = options.merge(
        venue_dates[["venue_id", "date"]],
        on=["venue_id", "date"],
        how="inner",
        validate="many_to_one",
    )
    if options.empty:
        return pd.DataFrame(), f"NO_COMMON_CONFIRMED_AVAILABILITY_DATE:{visit_id}"

    clearances = _active(tables["existing_service_conflicts"], "confirmed")
    clearances = clearances.loc[~clearances["conflict"].fillna(True).astype(bool)].copy()
    options = options.merge(
        clearances[["team_id", "vehicle_id", "venue_id", "date"]],
        on=["team_id", "vehicle_id", "venue_id", "date"],
        how="inner",
        validate="many_to_one",
    )
    if options.empty:
        return pd.DataFrame(), f"NO_CONFIRMED_EXACT_NO_CONFLICT_CHECK:{visit_id}"

    options.insert(0, "visit_id", visit_id)
    # ``bundle_id`` is already carried through the capability join.  Assigning
    # it makes the frozen visit requirement explicit without creating a
    # duplicate-labelled pandas column.
    options["bundle_id"] = bundle_id
    options["required_vehicle_capacity"] = requested_capacity
    output_columns = [
        "visit_id",
        "bundle_id",
        "venue_id",
        "date",
        "team_id",
        "base_id",
        "vehicle_id",
        "vehicle_type",
        "capacity",
        "required_vehicle_capacity",
        "distance_km",
        "travel_minutes_proxy",
        "graph_sha",
    ]
    return (
        options.loc[:, output_columns]
        .drop_duplicates()
        .sort_values(["date", "team_id", "vehicle_id"], kind="stable")
        .reset_index(drop=True),
        None,
    )


def _find_joint_assignment(
    options: pd.DataFrame,
    visit_ids: list[str],
    *,
    state_limit: int,
) -> tuple[pd.DataFrame, int, bool]:
    by_visit = {
        visit_id: options.loc[options["visit_id"].eq(visit_id)].to_dict("records")
        for visit_id in visit_ids
    }
    order = sorted(visit_ids, key=lambda value: (len(by_visit[value]), value))
    used_team_dates: set[tuple[str, pd.Timestamp]] = set()
    used_vehicle_dates: set[tuple[str, pd.Timestamp]] = set()
    used_venue_dates: set[tuple[str, pd.Timestamp]] = set()
    chosen: list[dict[str, Any]] = []
    states = 0
    exhausted = False

    def search(position: int) -> bool:
        nonlocal states, exhausted
        states += 1
        if states > state_limit:
            exhausted = True
            return False
        if position == len(order):
            return True
        visit_id = order[position]
        for option in by_visit[visit_id]:
            date = pd.Timestamp(option["date"])
            team_key = (str(option["team_id"]), date)
            vehicle_key = (str(option["vehicle_id"]), date)
            venue_key = (str(option["venue_id"]), date)
            if (
                team_key in used_team_dates
                or vehicle_key in used_vehicle_dates
                or venue_key in used_venue_dates
            ):
                continue
            used_team_dates.add(team_key)
            used_vehicle_dates.add(vehicle_key)
            used_venue_dates.add(venue_key)
            chosen.append(option)
            if search(position + 1):
                return True
            chosen.pop()
            used_team_dates.remove(team_key)
            used_vehicle_dates.remove(vehicle_key)
            used_venue_dates.remove(venue_key)
            if exhausted:
                return False
        return False

    found = search(0)
    if not found:
        return pd.DataFrame(columns=options.columns), states, exhausted
    selected = pd.DataFrame(chosen, columns=options.columns)
    selected = selected.set_index("visit_id").loc[visit_ids].reset_index()
    return selected, states, exhausted


def audit_operational_contract(
    tables: OperationalTables | Mapping[str, pd.DataFrame],
    visit_requirements: pd.DataFrame,
    *,
    required_visit_count: int | None = 20,
    required_bundle_ids: Iterable[str] | None = None,
    expected_graph_sha: str | None = None,
    assignment_search_state_limit: int = 500_000,
) -> OperationalContractAudit:
    """Audit evidence and prove a joint operational assignment.

    ``ready`` is true only if all schemas and evidence values are valid, every
    visit has at least one exact feasible tuple, and one non-double-booked set
    of tuples covers all visits.  ``expected_graph_sha`` should be supplied by
    the caller's frozen OSM provenance; a mismatch is a hard blocker.
    """

    blockers: list[str] = []
    if isinstance(tables, Mapping):
        try:
            tables = OperationalTables.from_mapping(tables)
        except OperationalContractError as exc:
            return OperationalContractAudit(
                ready=False,
                blockers=[f"TABLE_COLLECTION_INVALID:{exc}"],
                summary={"required_visit_count": required_visit_count},
            )
    if not isinstance(tables, OperationalTables):
        return OperationalContractAudit(
            ready=False,
            blockers=["TABLE_COLLECTION_INVALID:expected OperationalTables or mapping"],
            summary={"required_visit_count": required_visit_count},
        )
    if assignment_search_state_limit <= 0:
        blockers.append("INVALID_ASSIGNMENT_SEARCH_STATE_LIMIT")

    normalized: dict[str, pd.DataFrame] = {}
    for name in TABLE_SCHEMAS:
        normalized[name] = _normalize_table(name, getattr(tables, name), blockers)
    visits = _normalize_visits(visit_requirements, blockers, required_visit_count)

    if required_bundle_ids is None:
        required_bundles = sorted(set(visits.get("bundle_id", pd.Series(dtype=str)).astype(str)))
    else:
        required_bundles = sorted({str(value).strip() for value in required_bundle_ids})
        if not required_bundles or any(not value for value in required_bundles):
            blockers.append("REQUIRED_BUNDLE_IDS_INVALID")

    if not normalized["travel"].empty:
        graph_shas = normalized["travel"]["graph_sha"]
        graph_shas = graph_shas.loc[graph_shas.str.match(_SHA256_RE, na=False)]
        if graph_shas.nunique() > 1:
            blockers.append(f"MULTIPLE_TRAVEL_GRAPH_SHAS:{int(graph_shas.nunique())}")

    if expected_graph_sha is not None:
        expected = str(expected_graph_sha).strip().lower()
        if not _SHA256_RE.match(expected):
            blockers.append("EXPECTED_GRAPH_SHA_INVALID")
        elif not normalized["travel"].empty:
            mismatches = normalized["travel"]["graph_sha"].ne(expected)
            if mismatches.any():
                blockers.append(f"TRAVEL_GRAPH_SHA_MISMATCH:{int(mismatches.sum())}")

    # Referential integrity is checked before constructing tuples.  This also
    # prevents a valid-looking vehicle or travel row from being borrowed from
    # an unrelated team.
    if not blockers:
        bases = normalized["bases"]
        capabilities = normalized["team_bundle_capability"]
        vehicles = normalized["vehicles"]
        base_teams = set(bases["team_id"])
        base_pairs = set(zip(bases["team_id"], bases["base_id"]))
        unknown_cap_teams = set(capabilities["team_id"]) - base_teams
        if unknown_cap_teams:
            blockers.append(f"CAPABILITY_UNKNOWN_TEAM:{len(unknown_cap_teams)}")
        confirmed_teams = set(bases.loc[bases["confirmed"].fillna(False), "team_id"])
        confirmed_capability_pairs = set(
            zip(
                capabilities.loc[capabilities["confirmed"].fillna(False), "team_id"],
                capabilities.loc[capabilities["confirmed"].fillna(False), "bundle_id"],
            )
        )
        expected_capability_pairs = {
            (team_id, bundle_id)
            for team_id in confirmed_teams
            for bundle_id in required_bundles
        }
        missing_capability_pairs = expected_capability_pairs - confirmed_capability_pairs
        if missing_capability_pairs:
            blockers.append(
                f"TEAM_BUNDLE_MATRIX_INCOMPLETE:{len(missing_capability_pairs)}"
            )
        invalid_vehicle_pairs = {
            pair
            for pair in zip(vehicles["team_id"], vehicles["base_id"])
            if pair not in base_pairs
        }
        if invalid_vehicle_pairs:
            blockers.append(f"VEHICLE_TEAM_BASE_MISMATCH:{len(invalid_vehicle_pairs)}")
        unknown_calendar_teams = set(normalized["team_calendar"]["team_id"]) - base_teams
        if unknown_calendar_teams:
            blockers.append(f"TEAM_CALENDAR_UNKNOWN_TEAM:{len(unknown_calendar_teams)}")
        vehicle_ids = set(vehicles["vehicle_id"])
        calendar_vehicle_ids = set(
            normalized["resource_calendar"].loc[
                normalized["resource_calendar"]["resource_type"].eq("VEHICLE"), "resource_id"
            ]
        )
        unknown_resources = calendar_vehicle_ids - vehicle_ids
        if unknown_resources:
            blockers.append(f"RESOURCE_CALENDAR_UNKNOWN_VEHICLE:{len(unknown_resources)}")
        unknown_travel_teams = set(normalized["travel"]["team_id"]) - base_teams
        if unknown_travel_teams:
            blockers.append(f"TRAVEL_UNKNOWN_TEAM:{len(unknown_travel_teams)}")

    option_frames: list[pd.DataFrame] = []
    if not blockers:
        for _, visit in visits.iterrows():
            options, reason = _options_for_visit(visit, normalized)
            if reason is not None:
                blockers.append(reason)
            else:
                option_frames.append(options)
    feasible_options = (
        pd.concat(option_frames, ignore_index=True)
        if option_frames
        else pd.DataFrame()
    )

    selected = pd.DataFrame()
    search_states = 0
    search_inconclusive = False
    if not blockers and len(option_frames) == len(visits):
        selected, search_states, search_inconclusive = _find_joint_assignment(
            feasible_options,
            visits["visit_id"].astype(str).tolist(),
            state_limit=int(assignment_search_state_limit),
        )
        if search_inconclusive:
            blockers.append(
                f"ASSIGNMENT_SEARCH_INCONCLUSIVE_STATE_LIMIT:{assignment_search_state_limit}"
            )
        elif len(selected) != len(visits):
            blockers.append("NO_CONFLICT_FREE_GLOBAL_ASSIGNMENT")

    summary = {
        "contract_schema_version": OPERATIONAL_CONTRACT_SCHEMA_VERSION,
        "required_visit_count": required_visit_count,
        "required_bundle_ids": required_bundles,
        "visit_requirement_rows": int(len(visits)),
        "table_rows": {name: int(len(frame)) for name, frame in normalized.items()},
        "feasible_option_count": int(len(feasible_options)),
        "visits_with_options": int(feasible_options["visit_id"].nunique())
        if not feasible_options.empty
        else 0,
        "selected_assignment_count": int(len(selected)),
        "assignment_search_states": int(search_states),
        "assignment_search_inconclusive": bool(search_inconclusive),
        "expected_graph_sha": expected_graph_sha,
        "absence_means_no_conflict": False,
    }
    blockers = list(dict.fromkeys(blockers))
    return OperationalContractAudit(
        ready=not blockers and len(selected) == len(visits) and len(visits) > 0,
        blockers=blockers,
        summary=summary,
        feasible_options=feasible_options,
        selected_assignments=selected,
        normalized_tables=normalized,
    )


def certify_operational_contract(
    tables: OperationalTables | Mapping[str, pd.DataFrame],
    visit_requirements: pd.DataFrame,
    **kwargs: Any,
) -> OperationalContractAudit:
    """Return a ready audit or raise :class:`OperationalContractError`."""

    return audit_operational_contract(tables, visit_requirements, **kwargs).raise_if_not_ready()


def read_csv_table(path: str | Path) -> pd.DataFrame:
    """Small explicit loader used by callers; missing files remain hard errors."""

    resolved = Path(path)
    if not resolved.is_file():
        raise OperationalContractError(f"Operational input file is missing: {resolved}")
    try:
        return pd.read_csv(resolved, low_memory=False)
    except Exception as exc:  # pragma: no cover - pandas supplies the useful detail
        raise OperationalContractError(f"Could not read operational input: {resolved}") from exc
