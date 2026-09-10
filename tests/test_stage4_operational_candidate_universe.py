from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pyproj import Transformer
from scipy import sparse

from mediroad.stage4_2.compression import compress_grid_patterns
from mediroad.stage4_operational_final.candidate_universe import (
    HARD_5000M_RADIUS_M,
    STAGE3_COVERAGE_CLUSTER_JACCARD,
    OperationalUniverseError,
    build_verified_operational_universe,
    sparse_or_union,
)


def _project(longitude: float, latitude: float) -> tuple[float, float]:
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:5179", always_xy=True)
    x, y = transformer.transform(longitude, latitude)
    return float(x), float(y)


def _pattern_ready_grid(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    count = len(result)
    elderly = np.arange(1, count + 1, dtype=float) * 10.0
    result["sigungu"] = "synthetic"
    result["elderly_population"] = elderly
    result["need_weighted_population"] = elderly / 2.0
    result["high_need_population"] = np.where(
        np.arange(count) % 2 == 0, elderly, 0.0
    )
    result["is_rural_eup_myeon"] = np.arange(count) % 2 == 0
    return result


def _mapping(rows: list[dict[str, object]]) -> pd.DataFrame:
    defaults = {
        "source_anchor_is_busproxy": False,
        "source_anchor_cluster_id": "S3C",
        "source_anchor_admin_code": "A",
        "source_anchor_policies": "efficiency",
        "source_anchor_selection_frequency": 1,
        "relationship_ordinal": 1,
        "same_coverage_cluster": True,
        "same_admin": True,
        "anchor_to_candidate_distance_km": 0.0,
        "coverage_recompute_required": False,
        "required_for_busproxy_resolution": False,
        "priority": "P1",
    }
    return pd.DataFrame([{**defaults, **row} for row in rows])


def _form(ids: list[str], coordinates: dict[str, tuple[str, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "venue_id": venue_id,
                "field_verified": True,
                "verification_date": "2026-08-22",
                "parking": "YES",
                "verified_address": coordinates[venue_id][0],
                "verified_latitude": coordinates[venue_id][1],
                "verified_longitude": coordinates[venue_id][2],
            }
            for venue_id in ids
        ]
    )


def test_frozen_rows_are_reused_and_changed_coordinates_are_recomputed() -> None:
    reuse_id, moved_id = "FAC_REUSE", "FAC_MOVED"
    reuse_lat_lon = (36.0, 127.0)
    old_moved_lat_lon = (36.0, 127.10)
    new_moved_lat_lon = (36.0, 127.20)
    reuse_x, reuse_y = _project(reuse_lat_lon[1], reuse_lat_lon[0])
    moved_x, moved_y = _project(new_moved_lat_lon[1], new_moved_lat_lon[0])

    grid_ids = np.array(["g0", "g1", "g2", "g3"])
    grid = _pattern_ready_grid(pd.DataFrame(
        {
            "grid_id": grid_ids,
            "x_5179": [moved_x, moved_x + 4_999.0, moved_x + 5_001.0, reuse_x],
            "y_5179": [moved_y, moved_y, moved_y, reuse_y],
        }
    ))
    frozen = sparse.csr_matrix(
        np.array(
            [
                [0, 0, 0, 1],
                [0, 0, 1, 0],
            ],
            dtype=np.uint8,
        )
    )
    queue = pd.DataFrame(
        {
            "venue_id": [reuse_id, moved_id],
            "venue_name": ["Reuse", "Moved"],
            "cluster_id": ["S3_REUSE", "S3_MOVED"],
            "need_weighted_exposure": [2.0, 1.0],
            "anchor_relationships": ["DIRECT_SELECTED_PHYSICAL", "BUSPROXY_DECLARED_SAME_CLUSTER_FALLBACK"],
        }
    )
    catalog = pd.DataFrame(
        {
            "venue_id": [reuse_id, moved_id],
            "address": ["stage3 reuse address", "stage3 old address"],
            "latitude": [reuse_lat_lon[0], old_moved_lat_lon[0]],
            "longitude": [reuse_lat_lon[1], old_moved_lat_lon[1]],
            "coordinate_source": ["official_exact", "official_exact"],
            "source_kind": ["official_hall", "official_hall"],
        }
    )
    form = _form(
        [reuse_id, moved_id],
        {
            reuse_id: ("verified reuse address", *reuse_lat_lon),
            moved_id: ("verified corrected address", *new_moved_lat_lon),
        },
    )
    mapping = _mapping(
        [
            {
                "source_anchor_id": reuse_id,
                "physical_candidate_id": reuse_id,
                "relationship": "DIRECT_SELECTED_PHYSICAL",
            },
            {
                "source_anchor_id": "BUSPROXY_A",
                "source_anchor_is_busproxy": True,
                "physical_candidate_id": moved_id,
                "relationship": "BUSPROXY_DECLARED_SAME_CLUSTER_FALLBACK",
                "coverage_recompute_required": True,
                "required_for_busproxy_resolution": True,
            },
        ]
    )

    result = build_verified_operational_universe(
        queue=queue,
        normalized_field_form=form,
        field_evidence_release_ids={reuse_id, moved_id},
        stage3_catalog=catalog,
        anchor_mapping=mapping,
        stage3_hard_5000m=frozen,
        stage3_venue_ids=[reuse_id, moved_id],
        stage3_grid=grid,
        stage3_grid_ids=grid_ids,
    )

    assert result.venue_ids.tolist() == [reuse_id, moved_id]
    assert sparse.isspmatrix_csr(result.coverage)
    assert result.coverage.getrow(0).indices.tolist() == [3]
    assert result.coverage.getrow(1).indices.tolist() == [0, 1]
    ledger = result.coverage_origin_ledger.set_index("venue_id")
    assert ledger.loc[reuse_id, "coverage_origin"] == "FROZEN_STAGE3_HARD_5000M_ROW"
    assert not bool(ledger.loc[reuse_id, "coverage_recomputed"])
    assert bool(ledger.loc[moved_id, "coverage_recomputed"])
    assert "CKDTREE" in ledger.loc[moved_id, "coverage_origin"]
    assert result.summary["frozen_stage3_row_count"] == 1
    assert result.summary["recomputed_coverage_row_count"] == 1
    assert result.summary["busproxy_operational_candidate_count"] == 0
    assert result.candidates["field_verified"].eq(True).all()
    by_id = result.candidates.set_index("venue_id")
    reuse = by_id.loc[reuse_id]
    moved = by_id.loc[moved_id]
    assert reuse["operational_raw_elderly_exposure"] == pytest.approx(40.0)
    assert reuse["operational_need_weighted_exposure"] == pytest.approx(20.0)
    assert reuse["operational_high_need_exposure"] == pytest.approx(0.0)
    assert moved["operational_source_anchor_ids"] == "BUSPROXY_A"
    assert moved["busproxy_source_anchor_count"] == 1
    assert "BUSPROXY_DECLARED_SAME_CLUSTER_FALLBACK" in moved["fallback_provenance_json"]
    assert moved["stage3_cluster_id"] == "S3_MOVED"
    assert moved["address"] == "verified corrected address"
    assert moved["latitude"] == pytest.approx(new_moved_lat_lon[0])
    assert moved["longitude"] == pytest.approx(new_moved_lat_lon[1])
    assert moved["catalog_anchor_latitude_not_for_release"] == pytest.approx(
        old_moved_lat_lon[0]
    )
    assert moved["catalog_anchor_longitude_not_for_release"] == pytest.approx(
        old_moved_lat_lon[1]
    )
    assert moved["catalog_anchor_coordinate_source"] == "official_exact"
    assert "coordinate_source" not in result.candidates.columns
    assert moved["cluster_id"].startswith("OE")
    assert moved["coverage_cluster_id"] == moved["cluster_id"]
    assert moved["coverage_row_sha256"] in moved["coverage_reference"]
    assert moved["operational_raw_elderly_exposure"] == pytest.approx(30.0)
    assert moved["operational_need_weighted_exposure"] == pytest.approx(15.0)
    assert moved["operational_high_need_exposure"] == pytest.approx(10.0)
    assert result.unique_coverage.indices.tolist() == [0, 1, 3]
    assert result.summary["operational_unique_raw_elderly_exposure"] == pytest.approx(70.0)
    assert result.summary["operational_unique_need_weighted_exposure"] == pytest.approx(35.0)
    assert result.summary["operational_unique_high_need_exposure"] == pytest.approx(10.0)
    assert result.summary["operational_unique_rural_elderly_exposure"] == pytest.approx(10.0)
    assert result.summary["operational_unique_urban_elderly_exposure"] == pytest.approx(60.0)
    pattern_inputs = result.grid_pattern_inputs()
    assert pattern_inputs["coverage"] is result.coverage
    assert pattern_inputs["grid"]["grid_id"].tolist() == grid_ids.tolist()
    assert "is_rural_eup_myeon" in pattern_inputs["grid"]
    patterns = compress_grid_patterns(
        pattern_inputs["coverage"],
        pattern_inputs["venue_ids"],
        pattern_inputs["grid"],
    )
    assert int(patterns.source_grid_count.sum()) == len(grid_ids)
    assert all(len(value) == 64 for value in result.artifact_sha256.values())


def test_centroid_proxy_is_excluded_until_exact_coordinates_are_corrected() -> None:
    venue_id = "mobile_clinic_PROXY"
    old_lat, old_lon = 36.1, 127.1
    new_lat, new_lon = 36.11, 127.11
    new_x, new_y = _project(new_lon, new_lat)
    queue = pd.DataFrame(
        {
            "venue_id": [venue_id],
            "need_weighted_exposure": [1.0],
            "anchor_relationships": ["DIRECT_SELECTED_PHYSICAL"],
            "provisional_facility_latitude": [36.105],
            "provisional_facility_longitude": [127.105],
        }
    )
    catalog = pd.DataFrame(
        {
            "venue_id": [venue_id],
            "address": [""],
            "latitude": [old_lat],
            "longitude": [old_lon],
            "coordinate_source": ["admin_dong_centroid_proxy_not_exact_venue"],
            "source_kind": ["historical_mobile_service"],
        }
    )
    mapping = _mapping(
        [
            {
                "source_anchor_id": venue_id,
                "physical_candidate_id": venue_id,
                "relationship": "DIRECT_SELECTED_PHYSICAL",
            }
        ]
    )
    grid = _pattern_ready_grid(
        pd.DataFrame({"grid_id": ["g0"], "x_5179": [new_x], "y_5179": [new_y]})
    )
    frozen = sparse.csr_matrix([[1]], dtype=np.uint8)

    uncorrected = build_verified_operational_universe(
        queue=queue,
        normalized_field_form=_form(
            [venue_id], {venue_id: ("field supplied address", old_lat, old_lon)}
        ),
        field_evidence_release_ids={venue_id},
        stage3_catalog=catalog,
        anchor_mapping=mapping,
        stage3_hard_5000m=frozen,
        stage3_venue_ids=[venue_id],
        stage3_grid=grid,
        stage3_grid_ids=["g0"],
    )
    assert uncorrected.candidates.empty
    assert uncorrected.coverage.shape == (0, 1)
    assert uncorrected.exclusions["exclusion_reason"].str.contains(
        "CENTROID_OR_PROXY_REQUIRES_CORRECTED_EXACT_COORDINATES", regex=False
    ).all()

    corrected = build_verified_operational_universe(
        queue=queue,
        normalized_field_form=_form(
            [venue_id], {venue_id: ("field corrected exact address", new_lat, new_lon)}
        ),
        field_evidence_release_ids={venue_id},
        stage3_catalog=catalog,
        anchor_mapping=mapping,
        stage3_hard_5000m=frozen,
        stage3_venue_ids=[venue_id],
        stage3_grid=grid,
        stage3_grid_ids=["g0"],
    )
    assert corrected.exclusions.empty
    assert corrected.venue_ids.tolist() == [venue_id]
    assert corrected.coverage.indices.tolist() == [0]
    assert corrected.summary["recomputed_coverage_row_count"] == 1
    candidate = corrected.candidates.iloc[0]
    assert candidate["address"] == "field corrected exact address"
    assert candidate["latitude"] == pytest.approx(new_lat)
    assert candidate["longitude"] == pytest.approx(new_lon)
    assert candidate["catalog_anchor_latitude_not_for_release"] == pytest.approx(old_lat)
    assert candidate["catalog_anchor_longitude_not_for_release"] == pytest.approx(old_lon)
    assert candidate["provisional_facility_latitude_not_for_release"] == pytest.approx(
        36.105
    )
    assert candidate["provisional_facility_longitude_not_for_release"] == pytest.approx(
        127.105
    )
    assert "provisional_facility_latitude" not in corrected.candidates.columns
    assert "provisional_facility_longitude" not in corrected.candidates.columns


def test_busproxy_is_a_case_insensitive_hard_invariant() -> None:
    venue_id = "busproxy_bad"
    x, y = _project(127.0, 36.0)
    with pytest.raises(OperationalUniverseError, match="BUSPROXY appears in the physical queue"):
        build_verified_operational_universe(
            queue=pd.DataFrame({"venue_id": [venue_id]}),
            normalized_field_form=_form(
                [venue_id], {venue_id: ("verified address", 36.0, 127.0)}
            ),
            field_evidence_release_ids=set(),
            stage3_catalog=pd.DataFrame(
                {
                    "venue_id": [venue_id],
                    "latitude": [36.0],
                    "longitude": [127.0],
                    "coordinate_source": ["exact"],
                }
            ),
            anchor_mapping=pd.DataFrame(
                columns=["physical_candidate_id", "source_anchor_id", "relationship"]
            ),
            stage3_hard_5000m=sparse.csr_matrix([[1]], dtype=np.uint8),
            stage3_venue_ids=[venue_id],
            stage3_grid=_pattern_ready_grid(
                pd.DataFrame(
                    {"grid_id": ["g0"], "x_5179": [x], "y_5179": [y]}
                )
            ),
            stage3_grid_ids=["g0"],
        )


def test_operational_clusters_use_stage3_complete_link_threshold_and_sparse_or() -> None:
    venue_ids = ["A", "B", "C"]
    # A--B and B--C are >= 0.80, but A--C is below 0.80.  Complete-link
    # therefore may group only one edge.  Operational need exposure makes B
    # the representative seed and C the stronger alternate, so A stays out.
    frozen = sparse.csr_matrix(
        np.array(
            [
                [1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1, 1],
                [0, 1, 1, 1, 1, 1],
            ],
            dtype=np.uint8,
        )
    )
    coordinates = {
        "A": (36.0, 127.0),
        "B": (36.0, 127.01),
        "C": (36.0, 127.02),
    }
    queue = pd.DataFrame(
        {
            "venue_id": venue_ids,
            "need_weighted_exposure": [3.0, 2.0, 1.0],
            "anchor_relationships": ["DIRECT_SELECTED_PHYSICAL"] * 3,
        }
    )
    catalog = pd.DataFrame(
        {
            "venue_id": venue_ids,
            "address": [f"catalog {value}" for value in venue_ids],
            "latitude": [coordinates[value][0] for value in venue_ids],
            "longitude": [coordinates[value][1] for value in venue_ids],
            "coordinate_source": ["official_exact"] * 3,
        }
    )
    form = _form(
        venue_ids,
        {
            value: (f"verified {value}", coordinates[value][0], coordinates[value][1])
            for value in venue_ids
        },
    )
    mapping = _mapping(
        [
            {
                "source_anchor_id": value,
                "physical_candidate_id": value,
                "relationship": "DIRECT_SELECTED_PHYSICAL",
            }
            for value in venue_ids
        ]
    )
    ax, ay = _project(127.0, 36.0)
    grid_ids = [f"g{index}" for index in range(6)]
    grid = _pattern_ready_grid(pd.DataFrame(
        {
            "grid_id": grid_ids,
            "x_5179": [ax + index * 100 for index in range(6)],
            "y_5179": [ay] * 6,
        }
    ))

    result = build_verified_operational_universe(
        queue=queue,
        normalized_field_form=form,
        field_evidence_release_ids=set(venue_ids),
        stage3_catalog=catalog,
        anchor_mapping=mapping,
        stage3_hard_5000m=frozen,
        stage3_venue_ids=venue_ids,
        stage3_grid=grid,
        stage3_grid_ids=grid_ids,
    )

    assert STAGE3_COVERAGE_CLUSTER_JACCARD == 0.80
    assert HARD_5000M_RADIUS_M == 5_000.0
    overlap = result.coverage_overlap_edges.set_index(["venue_i", "venue_j"])
    assert overlap.loc[("A", "B"), "jaccard"] == pytest.approx(5 / 6)
    assert overlap.loc[("B", "C"), "jaccard"] == pytest.approx(5 / 6)
    assert overlap.loc[("A", "C"), "jaccard"] == pytest.approx(4 / 6)
    clusters = result.coverage_clusters.set_index("venue_id")
    assert clusters.loc["B", "operational_coverage_cluster_id"] == clusters.loc[
        "C", "operational_coverage_cluster_id"
    ]
    assert clusters.loc["A", "operational_coverage_cluster_id"] != clusters.loc[
        "B", "operational_coverage_cluster_id"
    ]
    assert clusters["cluster_min_pairwise_jaccard"].min() >= 0.80

    union = sparse_or_union(frozen)
    assert sparse.isspmatrix_csr(union)
    assert union.indices.tolist() == list(range(6))
    assert union.data.tolist() == [1] * 6
    assert result.unique_coverage.nnz == 6
    assert result.coverage.nnz == 16
    assert result.summary["unique_covered_grid_count"] == 6

    # Stage 3 first filters overlap output at population-weighted Jaccard 0.15.
    # Retaining that prerequisite is conservative: a high unweighted overlap
    # consisting only of zero-population grids must not form one policy cluster.
    low_weight_overlap_grid = grid.copy()
    low_weight_overlap_grid["elderly_population"] = [0, 0, 0, 0, 0, 100]
    low_weight_overlap_grid["need_weighted_population"] = [0, 0, 0, 0, 0, 50]
    low_weight_overlap_grid["high_need_population"] = [0, 0, 0, 0, 0, 100]
    conservative = build_verified_operational_universe(
        queue=queue.iloc[:2].copy(),
        normalized_field_form=form.iloc[:2].copy(),
        field_evidence_release_ids={"A", "B"},
        stage3_catalog=catalog.iloc[:2].copy(),
        anchor_mapping=mapping.loc[
            mapping["physical_candidate_id"].isin({"A", "B"})
        ].copy(),
        stage3_hard_5000m=frozen[:2],
        stage3_venue_ids=["A", "B"],
        stage3_grid=low_weight_overlap_grid,
        stage3_grid_ids=grid_ids,
    )
    conservative_edge = conservative.coverage_overlap_edges.iloc[0]
    assert conservative_edge["jaccard"] == pytest.approx(5 / 6)
    assert conservative_edge["population_weighted_jaccard"] == 0.0
    assert not bool(conservative_edge["at_cluster_threshold"])
    assert conservative.coverage_clusters["operational_coverage_cluster_id"].nunique() == 2


def test_release_id_requires_exact_location_and_a_verified_normalized_form() -> None:
    venue_id = "FAC_A"
    x, y = _project(127.0, 36.0)
    queue = pd.DataFrame({"venue_id": [venue_id]})
    catalog = pd.DataFrame(
        {
            "venue_id": [venue_id],
            "latitude": [36.0],
            "longitude": [127.0],
            "coordinate_source": ["official_exact"],
        }
    )
    mapping = _mapping(
        [
            {
                "source_anchor_id": venue_id,
                "physical_candidate_id": venue_id,
                "relationship": "DIRECT_SELECTED_PHYSICAL",
            }
        ]
    )
    grid = _pattern_ready_grid(
        pd.DataFrame({"grid_id": ["g0"], "x_5179": [x], "y_5179": [y]})
    )

    invalid_location = _form([venue_id], {venue_id: ("UNKNOWN", 36.0, 127.0)})
    excluded = build_verified_operational_universe(
        queue=queue,
        normalized_field_form=invalid_location,
        field_evidence_release_ids={venue_id},
        stage3_catalog=catalog,
        anchor_mapping=mapping,
        stage3_hard_5000m=sparse.csr_matrix([[1]], dtype=np.uint8),
        stage3_venue_ids=[venue_id],
        stage3_grid=grid,
        stage3_grid_ids=["g0"],
    )
    assert excluded.candidates.empty
    assert excluded.exclusions.loc[0, "exclusion_reason"] == "MISSING_EXACT_VERIFIED_ADDRESS"

    not_verified = _form([venue_id], {venue_id: ("exact address", 36.0, 127.0)})
    not_verified["field_verified"] = False
    with pytest.raises(OperationalUniverseError, match="not field_verified"):
        build_verified_operational_universe(
            queue=queue,
            normalized_field_form=not_verified,
            field_evidence_release_ids={venue_id},
            stage3_catalog=catalog,
            anchor_mapping=mapping,
            stage3_hard_5000m=sparse.csr_matrix([[1]], dtype=np.uint8),
            stage3_venue_ids=[venue_id],
            stage3_grid=grid,
            stage3_grid_ids=["g0"],
        )


def test_runner_facing_api_can_audit_evidence_and_derive_release_ids() -> None:
    venue_id = "FAC_EVIDENCED"
    x, y = _project(127.0, 36.0)
    queue = pd.DataFrame({"venue_id": [venue_id], "cluster_id": ["S3C"]})
    form = _form([venue_id], {venue_id: ("exact address", 36.0, 127.0)})
    evidence = pd.DataFrame(
        {
            "venue_id": [venue_id],
            "verification_source": ["DIRECT_PHONE_AND_SITE_CHECK"],
            "source_reference": ["field-log-20260822"],
            "contact_person_or_office": ["venue office"],
            "contact_channel": ["recorded phone call"],
            "verifier": ["field-team-lead"],
            "verification_notes": ["exact coordinate confirmed"],
            "evidence_file_or_url": ["evidence/FAC_EVIDENCED.json"],
        }
    )
    result = build_verified_operational_universe(
        queue=queue,
        normalized_field_form=form,
        field_evidence=evidence,
        stage3_catalog=pd.DataFrame(
            {
                "venue_id": [venue_id],
                "address": ["catalog address"],
                "latitude": [36.0],
                "longitude": [127.0],
                "coordinate_source": ["official_exact"],
            }
        ),
        anchor_mapping=_mapping(
            [
                {
                    "source_anchor_id": venue_id,
                    "physical_candidate_id": venue_id,
                    "relationship": "DIRECT_SELECTED_PHYSICAL",
                }
            ]
        ),
        stage3_hard_5000m=sparse.csr_matrix([[1]], dtype=np.uint8),
        stage3_venue_ids=[venue_id],
        stage3_grid=_pattern_ready_grid(
            pd.DataFrame({"grid_id": ["g0"], "x_5179": [x], "y_5179": [y]})
        ),
        stage3_grid_ids=["g0"],
    )

    assert result.venue_ids.tolist() == [venue_id]
    assert result.release_evidence["operational_release_verified"].eq(True).all()
    assert result.audit_issues.empty
    assert result.summary["field_evidence_gate"]["release_gate_mode"] == (
        "AUDITED_FIELD_EVIDENCE_TABLE"
    )
