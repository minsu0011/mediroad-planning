import numpy as np
import pandas as pd
from scipy import sparse

from mediroad.stage4_2.compression import compress_grid_patterns, verify_pattern_reconstruction


def test_grid_pattern_compression_is_lossless():
    coverage = sparse.csr_matrix(
        np.array(
            [
                [1, 1, 0, 0, 1, 0],
                [0, 0, 1, 1, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ],
            dtype=np.int8,
        )
    )
    grid = pd.DataFrame(
        {
            "grid_id": [f"G{i}" for i in range(6)],
            "sigungu": ["S1", "S1", "S1", "S1", "S1", "S2"],
            "elderly_population": [10, 20, 30, 40, 50, 60],
            "need_weighted_population": [1, 2, 3, 4, 5, 6],
            "high_need_population": [0, 0, 30, 40, 0, 60],
        }
    )
    patterns = compress_grid_patterns(coverage, np.array(["V1", "V2", "V3"]), grid)
    assert patterns.n_patterns < 6
    check = verify_pattern_reconstruction(patterns, coverage, np.array([0, 2]), grid)
    assert check["population_match"]
    assert check["need_match"]
    assert check["high_need_match"]
