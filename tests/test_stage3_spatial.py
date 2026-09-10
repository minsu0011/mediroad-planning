from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mediroad.stage3 import (  # noqa: E402
    CatchmentConfig,
    SparseCatchment,
    assert_hard_radius_monotonicity,
    build_catchment_family,
    build_sparse_catchment,
    candidate_pairs_within_distance,
    canonicalize_grid,
    canonicalize_venues,
    compute_pair_overlaps,
    compute_venue_exposures,
    connected_overlap_clusters,
    matrix_manifest_record,
    order_sha256,
    sparse_union,
    weighted_union_coverage,
)


@pytest.fixture()
def raw_grid() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "gid": ["g4", "g1", "g3", "g2", "g5"],
            "x_5179": [1003500.0, 1000000.0, 1002500.0, 1000500.0, 1007000.0],
            "y_5179": [1800000.0] * 5,
            "elderly65_calibrated_population": [40.0, 10.0, 30.0, 20.0, 50.0],
            "normalized_need": [0.8, 0.1, 0.9, 0.5, 1.0],
            "admin_dong_code": ["A2", "A1", "A2", "A1", "A3"],
            "policy_sigungu_name": ["S1", "S1", "S1", "S1", "S2"],
        }
    )


@pytest.fixture()
def raw_venues() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "venue_id": ["v3", "v1", "v2", "v4"],
            "x_5179": [1007000.0, 1000000.0, 1003000.0, 1015000.0],
            "y_5179": [1800000.0] * 4,
            "admin_dong_code": ["A3", "A1", "A2", "A3"],
            "policy_sigungu_name": ["S2", "S1", "S1", "S2"],
        }
    )


@pytest.fixture()
def canonical_inputs(raw_grid: pd.DataFrame, raw_venues: pd.DataFrame):
    grid = canonicalize_grid(
        raw_grid,
        crs="EPSG:5179",
        expected_rows=5,
        expected_admin_codes=["A1", "A2", "A3"],
    )
    venues = canonicalize_venues(
        raw_venues,
        crs=5179,
        expected_rows=4,
        expected_admin_codes=["A1", "A2", "A3"],
    )
    return grid, venues


def test_grid_and_venue_validation_is_canonical_and_sealed(canonical_inputs) -> None:
    grid, venues = canonical_inputs
    assert grid["gid"].tolist() == ["g1", "g2", "g3", "g4", "g5"]
    assert venues["venue_id"].tolist() == ["v1", "v2", "v3", "v4"]
    assert grid.attrs["crs"] == venues.attrs["crs"] == "EPSG:5179"
    assert grid.attrs["order_sha"] == order_sha256(grid["gid"])
    assert venues.attrs["order_sha"] == order_sha256(venues["venue_id"])


@pytest.mark.parametrize(
    ("kind", "mutation", "message"),
    [
        ("grid", lambda x: x.assign(gid=["g1"] * len(x)), "not unique"),
        (
            "grid",
            lambda x: x.assign(elderly65_calibrated_population=[1, 2, -1, 4, 5]),
            "non-negative",
        ),
        ("grid", lambda x: x.assign(x_5179=[np.nan] + [1.0] * 4), "finite"),
        ("venue", lambda x: x.assign(venue_id=["v1"] * len(x)), "not unique"),
        ("venue", lambda x: x.assign(x_5179=[1.0] * len(x)), "not unique physical"),
    ],
)
def test_canonical_validation_fails_closed(raw_grid, raw_venues, kind, mutation, message) -> None:
    function = canonicalize_grid if kind == "grid" else canonicalize_venues
    frame = mutation(raw_grid.copy() if kind == "grid" else raw_venues.copy())
    with pytest.raises(ValueError, match=message):
        function(frame, crs="EPSG:5179")


def test_wrong_or_unstated_metric_crs_fails(raw_grid) -> None:
    with pytest.raises(ValueError, match="EPSG:5179"):
        canonicalize_grid(raw_grid, crs="EPSG:4326")


def test_venue_sigungu_is_rederived_from_frozen_admin_mapping(raw_venues) -> None:
    wrong = raw_venues.copy()
    wrong["policy_sigungu_name"] = "UNTRUSTED_SOURCE_VALUE"
    result = canonicalize_venues(
        wrong,
        crs=5179,
        admin_to_sigungu={"A1": "S1", "A2": "S1", "A3": "S2"},
    )
    assert result.set_index("admin_dong_code")["policy_sigungu_name"].to_dict() == {
        "A1": "S1",
        "A2": "S1",
        "A3": "S2",
    }
    assert result["policy_sigungu_name_source"].eq("UNTRUSTED_SOURCE_VALUE").all()
    assert result.attrs["sigungu_rederived_from_admin"] is True
    assert result.attrs["source_sigungu_mismatch_count"] == 4
    with pytest.raises(ValueError, match="does not cover"):
        canonicalize_venues(wrong, crs=5179, admin_to_sigungu={"A1": "S1"})


def test_order_sha_is_order_sensitive_unambiguous_and_rejects_duplicates() -> None:
    assert order_sha256(["ab", "c"]) != order_sha256(["a", "bc"])
    assert order_sha256(["a", "b"]) != order_sha256(["b", "a"])
    with pytest.raises(ValueError, match="unique"):
        order_sha256(["a", "a"])


def _small_config() -> CatchmentConfig:
    return CatchmentConfig(
        hard_radii_m=(1000.0, 3000.0, 5000.0, 10000.0),
        decay_scales_m=(1000.0, 3000.0, 5000.0),
        decay_max_support_m={
            "exponential": {1000.0: 3000.0, 3000.0: 5000.0, 5000.0: 10000.0},
            "gaussian": {1000.0: 3000.0, 3000.0: 5000.0, 5000.0: 10000.0},
        },
    )


def test_family_builds_required_scenarios_as_csr_without_dense(canonical_inputs) -> None:
    grid, venues = canonical_inputs
    family = build_catchment_family(venues.sample(frac=1), grid.sample(frac=1), config=_small_config())
    assert set(family) == {
        "hard_1000m",
        "hard_3000m",
        "hard_5000m",
        "hard_10000m",
        "exp_1000m",
        "exp_3000m",
        "exp_5000m",
        "gaussian_1000m",
        "gaussian_3000m",
        "gaussian_5000m",
    }
    for artifact in family.values():
        assert sparse.isspmatrix_csr(artifact.matrix)
        assert artifact.matrix.shape == (4, 5)
        assert artifact.venue_ids == ("v1", "v2", "v3", "v4")
        assert artifact.grid_ids == ("g1", "g2", "g3", "g4", "g5")
        assert (artifact.matrix.data > 0).all()
        assert (artifact.matrix.data <= 1).all()


def test_hard_radius_is_inclusive_and_monotone(canonical_inputs) -> None:
    grid, venues = canonical_inputs
    family = build_catchment_family(venues, grid, config=_small_config())
    # v1 to g2 is exactly 500 m and v1 to g3 is exactly 2,500 m.
    assert family["hard_1000m"].matrix[0, 1] == 1
    assert family["hard_1000m"].matrix[0, 2] == 0
    assert family["hard_3000m"].matrix[0, 2] == 1
    audit = assert_hard_radius_monotonicity(family)
    assert len(audit) == 3
    assert audit["pass"].all()
    assert audit["larger_nnz"].ge(audit["smaller_nnz"]).all()


def test_monotonicity_detects_a_missing_small_radius_cell(canonical_inputs) -> None:
    grid, venues = canonical_inputs
    small = build_sparse_catchment(venues, grid, method="hard", radius_m=3000)
    broken_matrix = build_sparse_catchment(venues, grid, method="hard", radius_m=5000).matrix.copy()
    broken_matrix[0, small.matrix[0].indices[0]] = 0
    broken_matrix.eliminate_zeros()
    broken = SparseCatchment(
        matrix_id="hard_5km_broken",
        method="hard",
        parameter_m=5000,
        support_m=5000,
        matrix=broken_matrix,
        venue_ids=small.venue_ids,
        grid_ids=small.grid_ids,
    )
    with pytest.raises(ValueError, match="monotonicity failed"):
        assert_hard_radius_monotonicity([small, broken])


def test_decay_matches_formula_and_requires_finite_support(canonical_inputs) -> None:
    grid, venues = canonical_inputs
    exponential = build_sparse_catchment(
        venues, grid, method="exponential", scale_m=1000, max_support_m=3000
    )
    gaussian = build_sparse_catchment(
        venues, grid, method="gaussian", scale_m=1000, max_support_m=3000
    )
    assert exponential.matrix[0, 1] == pytest.approx(np.exp(-0.5), rel=1e-6)
    assert gaussian.matrix[0, 1] == pytest.approx(np.exp(-0.5 * 0.5**2), rel=1e-6)
    with pytest.raises(ValueError, match="explicit max_support"):
        build_sparse_catchment(venues, grid, method="exponential", scale_m=1000)


def test_manifest_seals_shape_density_and_orders(canonical_inputs) -> None:
    grid, venues = canonical_inputs
    artifact = build_sparse_catchment(venues, grid, method="hard", radius_m=3000)
    record = matrix_manifest_record(artifact, source_sha="a" * 64)
    assert record["n_venues"] == 4
    assert record["n_grids"] == 5
    assert record["density"] == pytest.approx(artifact.matrix.nnz / 20)
    assert record["venue_order_sha"] == order_sha256(artifact.venue_ids)
    assert record["grid_order_sha"] == order_sha256(artifact.grid_ids)
    with pytest.raises(ValueError, match="64-character"):
        matrix_manifest_record(artifact, source_sha="bad")


def test_exposure_partitions_cross_boundaries_and_respects_need(canonical_inputs) -> None:
    grid, venues = canonical_inputs
    artifact = build_sparse_catchment(venues, grid, method="hard", radius_m=5000)
    result = compute_venue_exposures(
        artifact,
        venues.sample(frac=1),
        grid.sample(frac=1),
        high_need_mask=np.array([False, False, True, True, True]),
    )
    assert np.allclose(
        result["own_admin_exposure"]
        + result["other_admin_same_sigungu_exposure"]
        + result["cross_sigungu_exposure"],
        result["raw_exposure"],
    )
    assert np.allclose(
        result["cross_admin_exposure"],
        result["other_admin_same_sigungu_exposure"] + result["cross_sigungu_exposure"],
    )
    assert result["cross_sigungu_exposure"].gt(0).any()
    assert result["need_weighted_exposure"].le(result["raw_exposure"] + 1e-9).all()
    assert np.allclose(
        result["own_admin_need_weighted_exposure"]
        + result["other_admin_same_sigungu_need_weighted_exposure"]
        + result["cross_sigungu_need_weighted_exposure"],
        result["need_weighted_exposure"],
    )
    assert np.allclose(
        result["cross_admin_need_weighted_exposure"],
        result["other_admin_same_sigungu_need_weighted_exposure"]
        + result["cross_sigungu_need_weighted_exposure"],
    )
    assert np.allclose(
        result["own_admin_high_need_exposure"]
        + result["other_admin_same_sigungu_high_need_exposure"]
        + result["cross_sigungu_high_need_exposure"],
        result["high_need_exposure"],
    )


def test_exposure_rejects_matrix_order_or_invalid_weights(canonical_inputs) -> None:
    grid, venues = canonical_inputs
    artifact = build_sparse_catchment(venues, grid, method="hard", radius_m=3000)
    with pytest.raises(ValueError, match="do not match sealed"):
        compute_venue_exposures(artifact, venues.iloc[:-1], grid)
    bad = grid.copy()
    bad.loc[0, "normalized_need"] = -1
    with pytest.raises(ValueError, match="non-negative"):
        compute_venue_exposures(artifact, venues, bad)


def test_sparse_union_matches_bruteforce_and_never_exceeds_naive(canonical_inputs) -> None:
    grid, venues = canonical_inputs
    artifact = build_sparse_catchment(venues, grid, method="hard", radius_m=3000)
    union = sparse_union(artifact.matrix, [0, 1])
    brute = np.any(artifact.matrix[[0, 1]].toarray() > 0, axis=0)
    assert np.array_equal(union.indices, np.flatnonzero(brute))
    population = grid.sort_values("gid")["elderly65_calibrated_population"].to_numpy()
    summary = weighted_union_coverage(artifact.matrix, [0, 1], population)
    assert summary.unique_weighted_coverage == pytest.approx(population[brute].sum())
    assert summary.unique_weighted_coverage <= summary.naive_weighted_sum
    assert summary.redundancy_mass == pytest.approx(
        summary.naive_weighted_sum - summary.unique_weighted_coverage
    )
    with pytest.raises(TypeError, match="dense"):
        sparse_union(artifact.matrix.toarray(), [0, 1])


def test_decay_sparse_union_uses_max_not_sum(canonical_inputs) -> None:
    grid, venues = canonical_inputs
    artifact = build_sparse_catchment(
        venues, grid, method="exponential", scale_m=3000, max_support_m=5000
    )
    union = sparse_union(artifact.matrix, [0, 1], binary=False)
    brute = np.max(artifact.matrix[[0, 1]].toarray(), axis=0)
    assert np.allclose(union.toarray().ravel(), brute)


def test_candidate_pairs_are_distance_bounded_and_deterministic(canonical_inputs) -> None:
    _, venues = canonical_inputs
    pairs = candidate_pairs_within_distance(venues.sample(frac=1), 4000)
    assert pairs[["venue_i", "venue_j"]].values.tolist() == [["v1", "v2"], ["v2", "v3"]]
    assert pairs.loc[0, "venue_distance_m"] == pytest.approx(3000)
    assert pairs.loc[1, "venue_distance_m"] == pytest.approx(4000)


def test_pair_overlap_matches_manual_jaccard_and_weighted_overlap(canonical_inputs) -> None:
    grid, venues = canonical_inputs
    artifact = build_sparse_catchment(venues, grid, method="hard", radius_m=3000)
    ordered_grid = grid.sort_values("gid")
    population = ordered_grid["elderly65_calibrated_population"].to_numpy()
    need = ordered_grid["normalized_need"].to_numpy()
    edges = compute_pair_overlaps(
        artifact,
        venues,
        population,
        grid_need=need,
        candidate_distance_m=6000,
    )
    edge = edges[(edges["venue_i"] == "v1") & (edges["venue_j"] == "v2")].iloc[0]
    left = set(artifact.matrix[0].indices)
    right = set(artifact.matrix[1].indices)
    intersection = left & right
    union = left | right
    assert edge["intersection_grid_count"] == len(intersection)
    assert edge["jaccard"] == pytest.approx(len(intersection) / len(union))
    assert edge["population_overlap"] == pytest.approx(population[list(intersection)].sum())
    assert edge["population_weighted_jaccard"] == pytest.approx(
        population[list(intersection)].sum() / population[list(union)].sum()
    )
    assert 0 <= edge["need_weighted_jaccard"] <= 1


def test_pair_overlap_rejects_decay_semantics(canonical_inputs) -> None:
    grid, venues = canonical_inputs
    artifact = build_sparse_catchment(
        venues, grid, method="gaussian", scale_m=1000, max_support_m=3000
    )
    with pytest.raises(ValueError, match="hard/binary"):
        compute_pair_overlaps(artifact, venues, np.ones(len(grid)))


def test_connected_clusters_include_isolates_and_choose_stable_representative() -> None:
    edges = pd.DataFrame(
        {
            "venue_i": ["v1", "v2", "v3"],
            "venue_j": ["v2", "v3", "v4"],
            "jaccard": [0.9, 0.85, 0.2],
        }
    )
    clusters = connected_overlap_clusters(
        ["v4", "v3", "v2", "v1", "v5"],
        edges,
        threshold=0.8,
        representative_scores={"v1": 1, "v2": 3, "v3": 2, "v4": 5, "v5": 4},
    )
    first = clusters.set_index("venue_id")
    assert first.loc["v1", "coverage_cluster_id"] == first.loc["v3", "coverage_cluster_id"]
    assert first.loc["v1", "representative_candidate"] == "v2"
    assert first.loc["v1", "alternate_candidate_count"] == 2
    assert first.loc["v4", "cluster_size"] == first.loc["v5", "cluster_size"] == 1


def test_cluster_graph_fails_on_unknown_or_duplicate_edges() -> None:
    unknown = pd.DataFrame({"venue_i": ["v1"], "venue_j": ["bad"], "jaccard": [0.9]})
    with pytest.raises(ValueError, match="unknown"):
        connected_overlap_clusters(["v1", "v2"], unknown, threshold=0.8)
    duplicate = pd.DataFrame(
        {"venue_i": ["v1", "v2"], "venue_j": ["v2", "v1"], "jaccard": [0.9, 0.9]}
    )
    with pytest.raises(ValueError, match="Duplicate"):
        connected_overlap_clusters(["v1", "v2"], duplicate, threshold=0.8)


def test_default_contract_declares_required_stage3_scales() -> None:
    config = CatchmentConfig().validated()
    assert config.hard_radii_m == (1000.0, 3000.0, 5000.0, 10000.0)
    assert config.decay_scales_m == (1000.0, 3000.0, 5000.0)
    broken = CatchmentConfig(decay_max_support_m={"exponential": {}, "gaussian": {}})
    with pytest.raises(ValueError, match="Missing exponential"):
        broken.validated()
