from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse


@dataclass(frozen=True)
class Stage3Paths:
    run_root: Path
    interface: Path
    grid_policy: Path
    matrix_manifest: Path
    shortlist: Path | None = None
    venue_overlap: Path | None = None
    coverage_clusters: Path | None = None
    quality_gate: Path | None = None
    metadata: Path | None = None
    inventory: Path | None = None
    package_root: Path | None = None


@dataclass
class CoverageMatrix:
    matrix_id: str
    matrix: sparse.csr_matrix
    venue_ids: np.ndarray
    grid_ids: np.ndarray
    method: str
    radius_or_scale: float | None
    source_path: Path
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not sparse.isspmatrix_csr(self.matrix):
            self.matrix = self.matrix.tocsr()
        if self.matrix.shape != (len(self.venue_ids), len(self.grid_ids)):
            raise ValueError(
                f"Coverage matrix shape {self.matrix.shape} does not match "
                f"venue/grid order {(len(self.venue_ids), len(self.grid_ids))}."
            )
        if self.matrix.data.size and np.nanmin(self.matrix.data) < 0:
            raise ValueError(f"Coverage matrix {self.matrix_id} contains negative values")


@dataclass
class OptimizationInput:
    candidates: pd.DataFrame
    grids: pd.DataFrame
    coverage: sparse.csr_matrix
    venue_ids: np.ndarray
    grid_ids: np.ndarray
    bundles: pd.DataFrame
    catchment_id: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.coverage.shape != (len(self.candidates), len(self.grids)):
            raise ValueError(
                "Optimization coverage shape does not align to candidate/grid tables: "
                f"{self.coverage.shape} vs {(len(self.candidates), len(self.grids))}"
            )
        if self.candidates["venue_id"].astype(str).duplicated().any():
            raise ValueError("Candidate venue_id must be unique")
        if self.grids["grid_id"].astype(str).duplicated().any():
            raise ValueError("Grid grid_id must be unique")
        if (pd.to_numeric(self.grids["elderly65_population"], errors="coerce") < 0).any():
            raise ValueError("Grid population cannot be negative")


@dataclass
class SolveStageResult:
    objective_name: str
    sense: str
    status: str
    objective_value: int | float | None
    best_bound: int | float | None
    wall_time_sec: float
    selected_indices: list[int]
    covered_grid_indices: list[int]
    raw_solver_stats: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScenarioSolution:
    scenario: str
    catchment_id: str
    visit_count: int
    status: str
    selected: pd.DataFrame
    covered_grid_mask: np.ndarray
    metrics: dict[str, Any]
    stages: list[SolveStageResult]
    bundle_assignments: pd.DataFrame | None = None
    notes: list[str] = field(default_factory=list)


@dataclass
class QualityCheck:
    check_id: str
    passed: bool
    severity: str
    observed: Any
    expected: Any
    message: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "check_id": self.check_id,
            "passed": bool(self.passed),
            "severity": self.severity,
            "observed": self.observed,
            "expected": self.expected,
            "message": self.message,
        }
