from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point, box

from mediroad.stage3.data import (
    build_grid_policy_weights,
    filter_venues_to_boundary,
    reconcile_grid_population,
    validate_grid_source,
)


def _grid() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            "grid_point_index": [0, 1, 2],
            "gid": ["g0", "g1", "g2"],
            "admin_dong_code": ["01", "01", "02"],
            "admin_dong_name": ["a", "a", "b"],
            "policy_sigungu_name": ["s1", "s1", "s2"],
            "x_5179": [10.0, 20.0, 80.0],
            "y_5179": [10.0, 20.0, 80.0],
            "elderly65_calibrated_population": [4.0, 6.0, 20.0],
            "raw_grid_value_202410": [2.0, 3.0, 10.0],
            "grid_state": ["suppressed_1to5", "suppressed_1to5", "observed_6plus"],
        },
        geometry=[Point(10, 10), Point(20, 20), Point(80, 80)],
        crs="EPSG:5179",
    )


def _master() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "admin_dong_code": ["01", "02"],
            "population_65plus": [10.0, 20.0],
            "is_rural_eup_myeon": [1, 0],
        }
    )


def test_grid_contract_and_population_reconciliation() -> None:
    grid = _grid()
    audit = validate_grid_source(grid, expected_rows=3, expected_admins=2)
    assert audit["population_total"] == pytest.approx(30.0)
    reconciliation = reconcile_grid_population(grid, _master())
    assert reconciliation["within_tolerance"].all()
    assert reconciliation["absolute_difference"].max() == pytest.approx(0.0)


def test_grid_contract_rejects_duplicate_coordinates() -> None:
    grid = _grid()
    grid.loc[2, ["x_5179", "y_5179"]] = [20.0, 20.0]
    with pytest.raises(ValueError, match="Grid source contract failed"):
        validate_grid_source(grid, expected_rows=3, expected_admins=2)


def test_boundary_filter_excludes_outside_without_mutating_source() -> None:
    venues = pd.DataFrame(
        {
            "venue_location_id": ["v1", "v2"],
            "venue_id": ["source1", "source2"],
            "admin_dong_code": ["01", "02"],
            "x_5179": [50.0, 120.0],
            "y_5179": [50.0, 50.0],
        }
    )
    original = venues.copy(deep=True)
    boundary = gpd.GeoDataFrame(
        {"name": ["province"]}, geometry=[box(0, 0, 100, 100)], crs="EPSG:5179"
    )
    result = filter_venues_to_boundary(venues, boundary)
    assert len(result.eligible) == 1
    assert len(result.exclusions) == 1
    assert result.exclusions.loc[0, "distance_outside_chungbuk_boundary_m"] == pytest.approx(20.0)
    pd.testing.assert_frame_equal(venues, original)


def test_grid_policy_broadcast_preserves_population_mass_and_bundle_axes() -> None:
    need = pd.DataFrame(
        {
            "admin_dong_code": ["01", "02"],
            "A_demographic_vulnerability_score": [20.0, 80.0],
            "B_health_burden_score": [30.0, 70.0],
            "C_medical_supply_access_gap_score": [40.0, 60.0],
            "D_transport_access_gap_score": [50.0, 50.0],
            "E_isolation_equity_score": [60.0, 40.0],
            "need_score": [25.0, 75.0],
            "need_rank": [2, 1],
        }
    )
    bundles = pd.DataFrame(
        {
            "admin_dong_code": ["01", "01", "02", "02"],
            "bundle_id": ["b1", "b2", "b1", "b2"],
            "bundle_gap_score": [10.0, 30.0, 70.0, 90.0],
        }
    )
    result = build_grid_policy_weights(
        _grid(), need, bundles, _master(), high_need_quantile=0.5
    )
    assert len(result) == 3
    assert result["elderly65_calibrated_population"].sum() == pytest.approx(30.0)
    assert set(result.filter(like="bundle_gap__").columns) == {
        "bundle_gap__b1",
        "bundle_gap__b2",
    }
    assert np.allclose(result["normalized_need"], result["need_score"] / 100.0)
    assert result.groupby("admin_dong_code")["high_need_admin_flag"].nunique().max() == 1

