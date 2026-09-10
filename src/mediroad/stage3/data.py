"""Stage 3 data contracts and deterministic spatial-context preparation."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping, Sequence

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree


ADMIN_KEY = "admin_dong_code"
VENUE_KEY = "venue_location_id"


@dataclass(frozen=True)
class VenueEligibilityResult:
    all_venues: pd.DataFrame
    eligible: pd.DataFrame
    exclusions: pd.DataFrame


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_string_key(values: pd.Series) -> pd.Series:
    """Return numeric administrative identifiers as zero-free-decimal strings."""

    text = values.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    if text.isna().any() or text.eq("").any():
        raise ValueError("Administrative key contains missing or empty values")
    return text


def verify_frozen_files(
    package_root: str | Path,
    contracts: Mapping[str, tuple[str | Path, str]],
) -> pd.DataFrame:
    """Verify immutable input files and return a machine-readable ledger."""

    root = Path(package_root).resolve()
    rows: list[dict[str, Any]] = []
    for source_id, (relative, expected) in contracts.items():
        path = (root / Path(relative)).resolve()
        if root not in path.parents:
            raise ValueError(f"Frozen path leaves package root: {path}")
        observed = sha256_file(path) if path.is_file() else "MISSING"
        rows.append(
            {
                "source_id": str(source_id),
                "relative_path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size if path.is_file() else -1,
                "expected_sha256": str(expected).lower(),
                "observed_sha256": observed.lower(),
                "valid": observed.lower() == str(expected).lower(),
            }
        )
    ledger = pd.DataFrame(rows).sort_values("source_id", kind="mergesort")
    if not bool(ledger["valid"].all()):
        bad = ledger.loc[~ledger["valid"], "source_id"].astype(str).tolist()
        raise ValueError(f"Frozen Stage 3 input mismatch: {bad}")
    return ledger.reset_index(drop=True)


def validate_grid_source(
    grid: pd.DataFrame,
    *,
    expected_rows: int,
    expected_admins: int,
) -> dict[str, Any]:
    required = {
        "grid_point_index",
        "gid",
        ADMIN_KEY,
        "policy_sigungu_name",
        "x_5179",
        "y_5179",
        "elderly65_calibrated_population",
        "raw_grid_value_202410",
        "grid_state",
    }
    missing = required - set(grid.columns)
    if missing:
        raise ValueError(f"Grid lacks required columns: {sorted(missing)}")
    frame = grid.copy()
    frame[ADMIN_KEY] = canonical_string_key(frame[ADMIN_KEY])
    population = pd.to_numeric(frame["elderly65_calibrated_population"], errors="coerce")
    audit = {
        "row_count": int(len(frame)),
        "admin_count": int(frame[ADMIN_KEY].nunique()),
        "sigungu_count": int(frame["policy_sigungu_name"].nunique()),
        "grid_index_duplicate_count": int(frame["grid_point_index"].duplicated().sum()),
        "gid_duplicate_count": int(frame["gid"].duplicated().sum()),
        "coordinate_duplicate_count": int(frame[["x_5179", "y_5179"]].duplicated().sum()),
        "population_missing_count": int(population.isna().sum()),
        "population_negative_count": int((population < 0).sum()),
        "population_total": float(population.sum()),
    }
    failures = [
        audit["row_count"] != int(expected_rows),
        audit["admin_count"] != int(expected_admins),
        audit["grid_index_duplicate_count"] != 0,
        audit["gid_duplicate_count"] != 0,
        audit["coordinate_duplicate_count"] != 0,
        audit["population_missing_count"] != 0,
        audit["population_negative_count"] != 0,
    ]
    if any(failures):
        raise ValueError(f"Grid source contract failed: {audit}")
    return audit


def reconcile_grid_population(
    grid: pd.DataFrame,
    master: pd.DataFrame,
    *,
    tolerance: float = 1e-8,
) -> pd.DataFrame:
    left = grid.copy()
    right = master.copy()
    left[ADMIN_KEY] = canonical_string_key(left[ADMIN_KEY])
    right[ADMIN_KEY] = canonical_string_key(right[ADMIN_KEY])
    if right[ADMIN_KEY].duplicated().any():
        raise ValueError("Master administrative key is not unique")
    observed = (
        left.groupby(ADMIN_KEY, sort=True)["elderly65_calibrated_population"]
        .sum()
        .rename("grid_elderly65_total")
    )
    expected = right.set_index(ADMIN_KEY)["population_65plus"].rename(
        "official_elderly65_total"
    )
    result = pd.concat([observed, expected], axis=1, join="outer").reset_index()
    result["difference"] = (
        result["grid_elderly65_total"] - result["official_elderly65_total"]
    )
    result["absolute_difference"] = result["difference"].abs()
    result["within_tolerance"] = result["absolute_difference"].le(float(tolerance))
    if result[["grid_elderly65_total", "official_elderly65_total"]].isna().any().any():
        raise ValueError("Grid/master administrative coverage mismatch")
    if not bool(result["within_tolerance"].all()):
        raise ValueError("Calibrated grid does not reconcile to official elderly totals")
    return result.sort_values(ADMIN_KEY, kind="mergesort").reset_index(drop=True)


def filter_venues_to_boundary(
    venues: pd.DataFrame,
    boundary: gpd.GeoDataFrame,
    *,
    metric_crs: str = "EPSG:5179",
) -> VenueEligibilityResult:
    required = {VENUE_KEY, "venue_id", ADMIN_KEY, "x_5179", "y_5179"}
    missing = required - set(venues.columns)
    if missing:
        raise ValueError(f"Venue table lacks required columns: {sorted(missing)}")
    frame = venues.copy()
    frame[ADMIN_KEY] = canonical_string_key(frame[ADMIN_KEY])
    if frame[VENUE_KEY].isna().any() or frame[VENUE_KEY].duplicated().any():
        raise ValueError("Venue location key must be complete and unique")
    if frame["venue_id"].isna().any() or frame["venue_id"].duplicated().any():
        raise ValueError("Venue source key must be complete and unique")
    xy = frame[["x_5179", "y_5179"]].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(xy.to_numpy(dtype=float)).all():
        raise ValueError("Venue coordinates contain non-finite values")
    if xy.duplicated().any():
        raise ValueError("Canonical physical venue coordinates are duplicated")

    borders = boundary.to_crs(metric_crs)
    if borders.empty or (~borders.geometry.is_valid).any():
        raise ValueError("Boundary geometry is empty or invalid")
    province = borders.geometry.union_all()
    points = gpd.GeoSeries(
        gpd.points_from_xy(xy["x_5179"], xy["y_5179"]), crs=metric_crs
    )
    inside = points.covered_by(province).to_numpy(dtype=bool)
    distance = np.zeros(len(frame), dtype=float)
    if (~inside).any():
        distance[~inside] = points.loc[~inside].distance(province.boundary).to_numpy()
    frame["inside_chungbuk_boundary"] = inside
    frame["distance_outside_chungbuk_boundary_m"] = distance
    frame["stage3_spatial_eligible"] = inside
    frame["stage3_spatial_exclusion_reason"] = np.where(
        inside, "", "outside_chungbuk_boundary"
    )
    frame = frame.sort_values(VENUE_KEY, kind="mergesort").reset_index(drop=True)
    eligible = frame.loc[frame["stage3_spatial_eligible"]].reset_index(drop=True)
    exclusions = frame.loc[~frame["stage3_spatial_eligible"]].reset_index(drop=True)
    return VenueEligibilityResult(frame, eligible, exclusions)


def build_grid_policy_weights(
    grid: pd.DataFrame,
    need_scores: pd.DataFrame,
    bundle_scores: pd.DataFrame,
    master: pd.DataFrame,
    *,
    high_need_quantile: float = 0.80,
) -> pd.DataFrame:
    frame = pd.DataFrame(grid.drop(columns=["geometry"], errors="ignore")).copy()
    need = need_scores.copy()
    bundles = bundle_scores.copy()
    admin = master.copy()
    for item in (frame, need, bundles, admin):
        item[ADMIN_KEY] = canonical_string_key(item[ADMIN_KEY])
    if need[ADMIN_KEY].duplicated().any():
        raise ValueError("Need table administrative key must be unique")
    if bundles.duplicated([ADMIN_KEY, "bundle_id"]).any():
        raise ValueError("Bundle score key must be unique")

    need_columns = [
        ADMIN_KEY,
        "A_demographic_vulnerability_score",
        "B_health_burden_score",
        "C_medical_supply_access_gap_score",
        "D_transport_access_gap_score",
        "E_isolation_equity_score",
        "need_score",
        "need_rank",
    ]
    missing_need = set(need_columns) - set(need.columns)
    if missing_need:
        raise ValueError(f"Need score table lacks {sorted(missing_need)}")
    rural_columns = [ADMIN_KEY, "is_rural_eup_myeon"]
    if not set(rural_columns).issubset(admin.columns):
        raise ValueError("Master lacks rural classification")
    bundle_wide = bundles.pivot(index=ADMIN_KEY, columns="bundle_id", values="bundle_gap_score")
    bundle_wide.columns = [f"bundle_gap__{column}" for column in bundle_wide.columns]
    bundle_wide = bundle_wide.reset_index()

    frame = frame.merge(need[need_columns], on=ADMIN_KEY, how="left", validate="many_to_one")
    frame = frame.merge(
        admin[rural_columns], on=ADMIN_KEY, how="left", validate="many_to_one"
    )
    frame = frame.merge(bundle_wide, on=ADMIN_KEY, how="left", validate="many_to_one")
    joined = need_columns[1:] + ["is_rural_eup_myeon"] + [
        column for column in frame.columns if column.startswith("bundle_gap__")
    ]
    if frame[joined].isna().any().any():
        raise ValueError("Grid policy broadcast contains missing policy values")
    frame["normalized_need"] = frame["need_score"].astype(float).div(100.0)
    admin_threshold = float(need["need_score"].astype(float).quantile(high_need_quantile))
    frame["high_need_admin_flag"] = frame["need_score"].astype(float).ge(admin_threshold)
    frame["grid_policy_role"] = "population_mass_spatial_allocation_not_independent_sample"
    return frame.sort_values("grid_point_index", kind="mergesort").reset_index(drop=True)


def _project_lon_lat(
    longitude: pd.Series, latitude: pd.Series, target_crs: str
) -> np.ndarray:
    lon = pd.to_numeric(longitude, errors="coerce").to_numpy(dtype=float)
    lat = pd.to_numeric(latitude, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(lon) & np.isfinite(lat)
    output = np.full((len(lon), 2), np.nan, dtype=float)
    transformer = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
    if valid.any():
        x, y = transformer.transform(lon[valid], lat[valid])
        output[valid, 0] = x
        output[valid, 1] = y
    return output


def build_transport_and_road_context(
    venues: pd.DataFrame,
    static_bus: pd.DataFrame,
    current_bus: pd.DataFrame,
    gtfs: pd.DataFrame,
    road_context: pd.DataFrame,
    *,
    metric_crs: str = "EPSG:5179",
    nearby_radius_m: float = 1000.0,
    jobs: int = 1,
) -> pd.DataFrame:
    """Attach clearly dated transit and road context without implying feasibility."""

    result = venues.copy()
    result[ADMIN_KEY] = canonical_string_key(result[ADMIN_KEY])
    coordinates = result[["x_5179", "y_5179"]].to_numpy(dtype=float)

    static = static_bus.copy()
    # The joined 2025 snapshot retains out-of-province points for provenance.
    # Only rows that were assigned inside a Chungbuk admin polygon are valid
    # distance references; a nearest external stop would silently understate
    # local access.  ``boundary_distance_m`` is populated only for fallback
    # assignments outside the polygon and unmapped rows have a null admin key.
    if {ADMIN_KEY, "boundary_distance_m"}.issubset(static.columns):
        inside_static = static[ADMIN_KEY].notna() & static["boundary_distance_m"].isna()
        static = static.loc[inside_static].reset_index(drop=True)
    bus_xy = _project_lon_lat(static["longitude"], static["latitude"], metric_crs)
    bus_valid = np.isfinite(bus_xy).all(axis=1)
    if not bus_valid.any():
        raise ValueError("No valid static bus-stop coordinates")
    bus_tree = cKDTree(bus_xy[bus_valid])
    bus_distance, _ = bus_tree.query(coordinates, k=1, workers=max(1, int(jobs)))
    result["distance_to_static_bus_stop_20251209_m"] = bus_distance

    gtfs_xy = _project_lon_lat(gtfs["stop_lon"], gtfs["stop_lat"], metric_crs)
    gtfs_valid = np.isfinite(gtfs_xy).all(axis=1)
    if not gtfs_valid.any():
        raise ValueError("No valid GTFS stop coordinates")
    gtfs_clean = gtfs.loc[gtfs_valid].reset_index(drop=True)
    gtfs_tree = cKDTree(gtfs_xy[gtfs_valid])
    gtfs_distance, _ = gtfs_tree.query(coordinates, k=1, workers=max(1, int(jobs)))
    neighbors = gtfs_tree.query_ball_point(
        coordinates, r=float(nearby_radius_m), workers=max(1, int(jobs))
    )
    departures = pd.to_numeric(
        gtfs_clean["weekday_departure_events"], errors="coerce"
    ).fillna(0.0).to_numpy(dtype=float)
    route_counts = pd.to_numeric(
        gtfs_clean["weekday_unique_routes"], errors="coerce"
    ).fillna(0.0).to_numpy(dtype=float)
    result["distance_to_gtfs_served_stop_202403_m"] = gtfs_distance
    result["nearby_gtfs_stop_count_202403"] = [len(index) for index in neighbors]
    result["nearby_gtfs_departures_baseline_202403"] = [
        float(departures[index].sum()) for index in neighbors
    ]
    result["nearby_gtfs_stop_route_count_sum_baseline_202403"] = [
        float(route_counts[index].sum()) for index in neighbors
    ]

    current = current_bus.copy()
    mapped = current["mapped_admin_dong_code"].dropna()
    current_counts = canonical_string_key(mapped).value_counts()
    result["current_bus_stop_records_admin_20260722"] = (
        result[ADMIN_KEY].map(current_counts).fillna(0).astype(int)
    )
    result["distance_to_current_bus_stop_20260722_status"] = (
        "unavailable_source_has_no_coordinates"
    )

    road = road_context.copy()
    if road[VENUE_KEY].duplicated().any():
        raise ValueError("Venue road context key is duplicated")
    road_columns = [
        VENUE_KEY,
        "osm_node_id_v6",
        "osm_snap_distance_m_v6",
        "osm_snap_valid_v6",
        "osm_from_cheongju_medical_center_drive_min_v6",
        "osm_from_chungju_medical_center_drive_min_v6",
        "osm_nearest_mobile_team_base_drive_min_v6",
    ]
    result = result.merge(road[road_columns], on=VENUE_KEY, how="left", validate="one_to_one")
    if result[road_columns[1:]].isna().any().any():
        raise ValueError("Venue road context is incomplete")

    static_closeness = 1.0 / (1.0 + result["distance_to_static_bus_stop_20251209_m"] / 1000.0)
    gtfs_closeness = 1.0 / (1.0 + result["distance_to_gtfs_served_stop_202403_m"] / 1000.0)
    departure_signal = np.log1p(result["nearby_gtfs_departures_baseline_202403"])
    departure_rank = departure_signal.rank(method="average", pct=True)
    result["transit_context_score"] = (
        static_closeness.rank(method="average", pct=True)
        + gtfs_closeness.rank(method="average", pct=True)
        + departure_rank
    ) / 3.0
    result["transit_context_interpretation"] = (
        "2025_static_coordinates_plus_2024_GTFS_baseline_not_current_frequency"
    )
    return result


__all__ = [
    "ADMIN_KEY",
    "VENUE_KEY",
    "VenueEligibilityResult",
    "build_grid_policy_weights",
    "build_transport_and_road_context",
    "canonical_string_key",
    "filter_venues_to_boundary",
    "reconcile_grid_population",
    "sha256_file",
    "validate_grid_source",
    "verify_frozen_files",
]
