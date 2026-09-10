import numpy as np
import pandas as pd
from scipy import sparse

from mediroad.stage4_2.candidates import compress_exact_equivalent_candidates, prune_safe_dominated_candidates


def test_exact_equivalent_candidates_collapse():
    candidates = pd.DataFrame(
        [
            {"venue_id": "BUSPROXY_1", "admin_code": "A", "sigungu": "S", "cluster_id": "C", "known_overlap": 0, "venue_readiness_prior": 0.3},
            {"venue_id": "REAL_1", "admin_code": "A", "sigungu": "S", "cluster_id": "C", "known_overlap": 0, "venue_readiness_prior": 0.6},
        ]
    )
    coverage = sparse.csr_matrix(np.array([[1, 0, 1], [1, 0, 1]], dtype=np.int8))
    result = compress_exact_equivalent_candidates(
        candidates, coverage, {"optimization": {"deterministic_tiebreak_scale": 0.0}}
    )
    assert len(result.candidates) == 1
    assert result.candidates.iloc[0]["venue_id"] == "REAL_1"


def test_safe_subset_dominance():
    candidates = pd.DataFrame(
        [
            {"venue_id": "A", "admin_code": "X", "sigungu": "S", "cluster_id": "C", "known_overlap": 0, "travel_minutes": 5},
            {"venue_id": "B", "admin_code": "X", "sigungu": "S", "cluster_id": "C", "known_overlap": 0, "travel_minutes": 5},
        ]
    )
    coverage = sparse.csr_matrix(np.array([[1, 0, 0], [1, 1, 0]], dtype=np.int8))
    result = prune_safe_dominated_candidates(
        candidates, coverage, {"optimization": {"deterministic_tiebreak_scale": 0.0}}
    )
    assert result.candidates["venue_id"].tolist() == ["B"]


def test_subset_is_not_pruned_when_replacement_has_higher_optimizer_cost():
    candidates = pd.DataFrame(
        [
            {"venue_id": "A", "admin_code": "X", "sigungu": "S", "cluster_id": "C", "known_overlap": 0, "travel_minutes": 5},
            {"venue_id": "B", "admin_code": "X", "sigungu": "S", "cluster_id": "C", "known_overlap": 0, "travel_minutes": 10},
        ]
    )
    coverage = sparse.csr_matrix(np.array([[1, 0, 0], [1, 1, 0]], dtype=np.int8))
    result = prune_safe_dominated_candidates(
        candidates,
        coverage,
        {"optimization": {"travel_penalty_scale": 1.0, "deterministic_tiebreak_scale": 0.0}},
    )
    assert result.candidates["venue_id"].tolist() == ["A", "B"]
