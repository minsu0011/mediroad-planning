from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse


@dataclass(frozen=True)
class Stage3Artifacts:
    pointer: Path
    run_root: Path
    metadata: Path
    interface: Path
    matrix_manifest: Path
    grid_policy: Path


@dataclass(frozen=True)
class Stage41Artifacts:
    pointer: Path
    run_root: Path
    metadata: Path
    inventory: Path
    report: Path
    field_form: Path
    field_queue: Path
    field_interface: Path
    physical_resolution: Path
    candidate_membership: Path
    plan_efficiency: Path
    plan_balanced: Path
    plan_equity: Path
    policy_metrics: Path
    solver_certification: Path


@dataclass
class CoverageData:
    matrix_id: str
    matrix: sparse.csr_matrix
    venue_ids: np.ndarray
    grid_ids: np.ndarray
    venue_order_path: Path
    grid_order_path: Path
    matrix_path: Path
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.matrix.shape != (len(self.venue_ids), len(self.grid_ids)):
            raise ValueError(
                f"Coverage shape {self.matrix.shape} does not match venue/grid orders "
                f"({len(self.venue_ids)}, {len(self.grid_ids)})"
            )
        if len(set(map(str, self.venue_ids))) != len(self.venue_ids):
            raise ValueError("Duplicate venue IDs in coverage order")
        if len(set(map(str, self.grid_ids))) != len(self.grid_ids):
            raise ValueError("Duplicate grid IDs in coverage order")


@dataclass
class GridPatternData:
    candidate_ids: np.ndarray
    pattern_coverers: list[np.ndarray]
    population: np.ndarray
    need_weighted: np.ndarray
    high_need_population: np.ndarray
    sigungu: np.ndarray
    source_grid_count: np.ndarray
    source_grid_indices: list[np.ndarray]

    @property
    def n_candidates(self) -> int:
        return len(self.candidate_ids)

    @property
    def n_patterns(self) -> int:
        return len(self.pattern_coverers)


@dataclass
class SolverStageTelemetry:
    scenario: str
    candidate_set: str
    stage_index: int
    objective_name: str
    sense: str
    solver_status: str
    success: bool
    objective_value: float | None
    best_bound: float | None
    absolute_gap: float | None
    relative_gap: float | None
    solver_reported_mip_gap: float | None
    mip_node_count: int | None
    wall_time_sec: float
    message: str
    time_limit_sec: float
    solver_engine: str
    threads_requested: int
    random_seed: int
    applied_objective_floors_json: str
    certified: bool


@dataclass
class ScenarioResult:
    scenario: str
    candidate_set: str
    selected_indices: np.ndarray
    selected_venue_ids: list[str]
    solver_status: str
    certification_class: str
    certified: bool
    max_relative_gap: float | None
    telemetry: list[SolverStageTelemetry]
    metrics: dict[str, Any]
    selected_frame: pd.DataFrame
