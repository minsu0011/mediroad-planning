from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from mediroad.stage4_2.data import _order_sha256, load_coverage, normalize_grid_policy
from mediroad.stage4_2.types import Stage3Artifacts
from mediroad.stage4_2.utils import sha256_file


def test_load_coverage_uses_canonical_stage3_orders_and_preserves_zero_padding(tmp_path: Path) -> None:
    root = tmp_path
    run_root = root / "outputs/model_v1/07_stage3/runs/RUN"
    exposure = run_root / "02_exposure"
    exposure.mkdir(parents=True)
    pointer = root / "outputs/model_v1/07_stage3/CURRENT_STAGE3_RUN.json"
    pointer.parent.mkdir(parents=True, exist_ok=True)
    pointer.write_text("{}", encoding="utf-8")

    matrix_path = exposure / "venue_grid_hard_5000m.npz"
    sparse.save_npz(matrix_path, sparse.csr_matrix(np.array([[1, 0], [1, 1]], dtype=np.uint8)))
    row_order = pd.DataFrame({"matrix_row": [0, 1], "venue_id": ["V1", "V2"]})
    col_order = pd.DataFrame({"matrix_column": [0, 1], "matrix_grid_id": ["000000", "000001"]})
    row_order.to_parquet(exposure / "venue_grid_row_order.parquet", index=False)
    col_order.to_parquet(exposure / "venue_grid_column_order.parquet", index=False)

    manifest = exposure / "venue_grid_matrix_manifest.csv"
    pd.DataFrame(
        [
            {
                "matrix_id": "hard_5000m",
                "relative_path": matrix_path.relative_to(root).as_posix(),
                "file_sha256": sha256_file(matrix_path),
                "venue_order_sha": _order_sha256(row_order["venue_id"]),
                "grid_order_sha": _order_sha256(col_order["matrix_grid_id"]),
                "n_venues": 2,
                "n_grids": 2,
                "nnz": 3,
            }
        ]
    ).to_csv(manifest, index=False)
    grid_policy = run_root / "01_grid_policy/grid_policy_weights.parquet"
    grid_policy.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "matrix_grid_id": ["000000", "000001"],
            "grid_point_index": [0, 1],
            "elderly65_calibrated_population": [1.0, 2.0],
            "need_score": [50.0, 60.0],
            "policy_sigungu_name": ["S", "S"],
            "admin_dong_code": ["A", "A"],
        }
    ).to_parquet(grid_policy, index=False)

    artifacts = Stage3Artifacts(pointer, run_root, run_root / "metadata.json", run_root / "interface.parquet", manifest, grid_policy)
    coverage = load_coverage(artifacts)
    assert coverage.venue_ids.tolist() == ["V1", "V2"]
    assert coverage.grid_ids.tolist() == ["000000", "000001"]
    normalized = normalize_grid_policy(pd.read_parquet(grid_policy))
    assert normalized["grid_id"].tolist() == ["000000", "000001"]
