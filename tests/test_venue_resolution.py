import pandas as pd

from mediroad.stage4_2.venue_resolution import resolve_physical_venues


def test_busproxy_resolves_to_verified_declared_fallback():
    anchor = pd.DataFrame(
        [
            {
                "venue_id": "BUSPROXY_A",
                "cluster_id": "C1",
                "spatial_anchor_venue_id": "BUSPROXY_A",
                "spatial_anchor_is_busproxy": True,
                "actual_facility_fallback_1": "REAL_1",
                "actual_facility_fallback_2": "REAL_2",
            }
        ]
    )
    field = pd.DataFrame(
        [
            {"venue_id": "BUSPROXY_A", "cluster_id": "C1", "field_verified": False},
            {"venue_id": "REAL_1", "cluster_id": "C1", "field_verified": True},
            {"venue_id": "REAL_2", "cluster_id": "C1", "field_verified": False},
        ]
    )
    catalog = pd.DataFrame(
        [
            {"venue_id": "BUSPROXY_A", "cluster_id": "C1", "venue_name": "proxy", "venue_type": "proxy"},
            {"venue_id": "REAL_1", "cluster_id": "C1", "venue_name": "hall", "venue_type": "senior_center"},
            {"venue_id": "REAL_2", "cluster_id": "C1", "venue_name": "office", "venue_type": "office"},
        ]
    )
    result = resolve_physical_venues(anchor, field, catalog)
    assert result.summary["all_resolved"]
    assert result.resolution.loc[0, "final_physical_venue_id"] == "REAL_1"
    assert result.summary["busproxy_final_count"] == 0


def test_unverified_direct_anchor_stays_unresolved():
    anchor = pd.DataFrame(
        [
            {
                "venue_id": "REAL_A",
                "cluster_id": "C1",
                "spatial_anchor_venue_id": "REAL_A",
                "spatial_anchor_is_busproxy": False,
                "actual_facility_fallback_1": None,
                "actual_facility_fallback_2": None,
            }
        ]
    )
    field = pd.DataFrame([{"venue_id": "REAL_A", "cluster_id": "C1", "field_verified": False}])
    catalog = pd.DataFrame([{"venue_id": "REAL_A", "cluster_id": "C1", "venue_name": "x", "venue_type": "hall"}])
    result = resolve_physical_venues(anchor, field, catalog)
    assert result.summary["unresolved"] == 1
