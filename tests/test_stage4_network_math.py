from __future__ import annotations

import numpy as np
import pandas as pd

from mediroad.stage4.network_validation import (
    _admin_rank_stability,
    _binary_network_matrices,
    _topk_jaccard_by_admin,
)


def test_topk_jaccard_and_best_retention():
    candidates = pd.DataFrame(
        {
            "venue_id": ["a", "b", "c", "d"],
            "admin_code": ["x", "x", "y", "y"],
        }
    )
    ref = np.array([10, 9, 7, 6], dtype=float)
    alt = np.array([9, 10, 7, 6], dtype=float)
    top, best = _topk_jaccard_by_admin(candidates, ref, alt, k=2)
    assert top == 1.0
    assert best == 0.5


def test_network_matrix_deduplicates_snapped_vertices_and_restores_rows():
    class Graph:
        def distances(self, *, source, target, weights, mode):
            assert len(source) == len(set(source))
            assert len(target) == len(set(target))
            lookup = {(0, 2): 5.0, (0, 3): 12.0, (1, 2): 20.0, (1, 3): 3.0}
            return [[lookup[(s, t)] for t in target] for s in source]

    matrices = _binary_network_matrices(
        Graph(),
        venue_vertices=np.array([0, 0, 1]),
        grid_vertices=np.array([2, 2, 3, 3]),
        valid_venues=np.ones(3, dtype=bool),
        valid_grids=np.ones(4, dtype=bool),
        thresholds=[10],
        batch_size=2,
    )
    assert matrices[10].toarray().tolist() == [
        [1, 1, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
    ]


def test_admin_top3_scope_is_disclosed_as_uninformative():
    candidates = pd.DataFrame(
        {
            "admin_code": ["a", "a", "a", "b", "b", "b"],
            "venue_id": [f"v{i}" for i in range(6)],
        }
    )
    median_rho, informative = _admin_rank_stability(
        candidates,
        np.array([3, 2, 1, 3, 2, 1], dtype=float),
        np.array([1, 2, 3, 3, 2, 1], dtype=float),
        top_k=3,
    )
    assert median_rho == 0.0
    assert informative == 0.0
