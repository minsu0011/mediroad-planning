from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from mediroad.stage4.config import load_config
from mediroad.stage4.discovery import _read_current_pointer
from mediroad.stage4.matrices import _discover_sidecar


def test_current_stage3_pointer_resolves_project_relative_metadata(tmp_path: Path) -> None:
    stage3_root = tmp_path / "outputs" / "model_v1" / "07_stage3"
    run_root = stage3_root / "runs" / "stage3_test"
    run_root.mkdir(parents=True)
    metadata = run_root / "STAGE3_RUN_METADATA.json"
    metadata.write_text("{}", encoding="utf-8")
    pointer = {
        "run_id": "stage3_test",
        "metadata_relative_path": str(metadata.relative_to(tmp_path)).replace("\\", "/"),
    }
    (stage3_root / "CURRENT_STAGE3_RUN.json").write_text(
        json.dumps(pointer), encoding="utf-8"
    )

    assert _read_current_pointer(stage3_root) == run_root.resolve()


def test_sidecar_discovery_does_not_swap_venue_rows_and_grid_columns(
    tmp_path: Path,
) -> None:
    matrix = tmp_path / "venue_grid_hard_5000m.npz"
    matrix.touch()
    row = tmp_path / "venue_grid_row_order.parquet"
    column = tmp_path / "venue_grid_column_order.parquet"
    pd.DataFrame({"venue_id": ["V1"]}).to_parquet(row, index=False)
    pd.DataFrame({"matrix_grid_id": ["000000"]}).to_parquet(column, index=False)

    assert _discover_sidecar(matrix, "hard_5000m", "venue") == row
    assert _discover_sidecar(matrix, "hard_5000m", "grid") == column


def test_stage3_grid_population_alias_matches_frozen_stage3_schema() -> None:
    config = load_config(None)
    assert "elderly65_calibrated_population" in config["column_aliases"]["elderly_population"]
