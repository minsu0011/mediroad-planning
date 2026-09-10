from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import sparse

Sense = Literal["max", "min"]


@dataclass(frozen=True)
class Floor:
    """A retained lexicographic objective constraint in reported units."""

    objective: str
    sense: Sense
    value: float


@dataclass
class LinearMipModel:
    """Solver-neutral sparse mixed-integer model."""

    name: str
    objective_name: str
    sense: Sense
    c: np.ndarray
    A: sparse.csc_matrix
    row_lower: np.ndarray
    row_upper: np.ndarray
    col_lower: np.ndarray
    col_upper: np.ndarray
    integrality: np.ndarray
    variable_names: list[str]
    x_slice: slice
    y_slice: slice
    extra_slices: dict[str, slice | int] = field(default_factory=dict)
    reported_objective_scale: float = 1.0
    objective_offset: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_col(self) -> int:
        return int(self.c.size)

    @property
    def n_row(self) -> int:
        return int(self.row_lower.size)

    def reported_objective(self, raw_value: float | None) -> float | None:
        if raw_value is None:
            return None
        return float(raw_value / self.reported_objective_scale + self.objective_offset)

    def validate(self) -> None:
        if self.A.shape != (self.n_row, self.n_col):
            raise ValueError(f"A shape {self.A.shape} != ({self.n_row}, {self.n_col})")
        for name, arr, expected in [
            ("row_lower", self.row_lower, self.n_row),
            ("row_upper", self.row_upper, self.n_row),
            ("col_lower", self.col_lower, self.n_col),
            ("col_upper", self.col_upper, self.n_col),
            ("integrality", self.integrality, self.n_col),
        ]:
            if len(arr) != expected:
                raise ValueError(f"{name} length {len(arr)} != {expected}")
        if len(self.variable_names) != self.n_col:
            raise ValueError("variable_names length mismatch")
        if np.any(self.row_lower > self.row_upper):
            raise ValueError("row lower bound exceeds upper bound")
        if np.any(self.col_lower > self.col_upper):
            raise ValueError("column lower bound exceeds upper bound")
        if not sparse.isspmatrix_csc(self.A):
            raise TypeError("A must be CSC")


@dataclass
class Seed:
    source: str
    selected_indices: np.ndarray
    full_values: np.ndarray | None = None
    feasible: bool = False
    objective_value: float | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    rejection_reason: str | None = None


@dataclass
class BackendResult:
    model_name: str
    status: str
    has_incumbent: bool
    is_infeasible: bool
    is_optimal: bool
    objective_value_raw: float | None
    objective_value: float | None
    best_bound_raw: float | None
    best_bound: float | None
    relative_gap: float | None
    mip_node_count: int | None
    wall_time_sec: float
    solution: np.ndarray | None
    message: str
    highs_version: str | None = None
    options: dict[str, Any] = field(default_factory=dict)
    log_path: Path | None = None


@dataclass
class StageResult:
    scenario: str
    candidate_set: str
    stage_index: int
    objective_name: str
    sense: Sense
    status: str
    certificate: str
    certified: bool
    objective_value: float | None
    best_bound: float | None
    relative_gap: float | None
    wall_time_sec: float
    mip_node_count: int | None
    selected_indices: np.ndarray | None
    solution: np.ndarray | None
    retained_floor: Floor | None
    seed_source: str | None
    oracle_rounds: int
    model_rows: int
    model_cols: int
    model_nnz: int
    backend_message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScenarioCertificationResult:
    scenario: str
    candidate_set: str
    certified: bool
    certification_class: str
    selected_indices: np.ndarray
    selected_venue_ids: list[str]
    selected_frame: pd.DataFrame
    metrics: dict[str, Any]
    stages: list[StageResult]
    max_relative_gap: float | None


@dataclass
class CandidateProblem:
    candidate_set: str
    candidates: pd.DataFrame
    patterns: Any
    coverage: sparse.csr_matrix
    grid: pd.DataFrame
    stage41: Any
    bundles: pd.DataFrame
    reduction_ledger: pd.DataFrame
    venue_alias_map: dict[str, str] = field(default_factory=dict)
    source_payload: dict[str, Any] = field(default_factory=dict)
