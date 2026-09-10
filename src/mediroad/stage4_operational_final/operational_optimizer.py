"""Operationally constrained Stage 4 spatial optimization.

The frozen Stage 4.2 spatial MILP remains the master problem.  Operational
feasibility is checked by an exact subproblem over bundle, team, vehicle,
venue, and availability witnesses.  If (and only if) the subproblem proves a
selected 20-venue set infeasible, the master receives the exact no-good cut::

    sum(x[i] for i in selected_set) <= 19

Because the frozen master also enforces ``sum(x) == 20``, this removes exactly
one selected set and no other spatial plan.  Limits or uncertified solver
results never produce a no-good cut and never produce an operational PASS.

Dates are used internally to prove joint availability.  Public result objects
expose only a deterministic witness hash and the assigned bundle/team/vehicle/
travel fields; they deliberately do not expose visit dates.  GPU work, if a
caller performs any, is never proof in this module.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import time
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, milp

from mediroad.stage4_2.compression import compress_grid_patterns
from mediroad.stage4_2.errors import ContractError, SolverCertificationError
from mediroad.stage4_2.optimizer import SpatialMILP
from mediroad.stage4_2.types import GridPatternData, ScenarioResult

from .operational_contract import (
    OperationalContractAudit,
    OperationalTables,
    audit_operational_contract,
)


REQUIRED_VISIT_COUNT = 20
REQUIRED_BUNDLE_COUNT = 5
REQUIRED_BUNDLE_MINIMUM = 1
MAX_CERTIFIED_RELATIVE_GAP = 0.005
GPU_IS_PROOF = False

_FROZEN_CANDIDATE_CONSTRAINTS: dict[str, Any] = {
    "one_per_coverage_cluster": True,
    "max_visits_per_admin": 2,
    "max_visits_per_sigungu": 4,
}

_LOCAL_OPTION_BLOCKER_PREFIXES = (
    "NO_CONFIRMED_CAPABLE_TEAM:",
    "NO_CONFIRMED_COMPATIBLE_VEHICLE:",
    "NO_CONFIRMED_REACHABLE_TEAM_VENUE_TRAVEL:",
    "NO_COMMON_CONFIRMED_AVAILABILITY_DATE:",
    "NO_CONFIRMED_EXACT_NO_CONFLICT_CHECK:",
)
_OPTION_ENUMERATION_ONLY_BLOCKER_PREFIXES = (
    "NO_CONFLICT_FREE_GLOBAL_ASSIGNMENT",
    "ASSIGNMENT_SEARCH_INCONCLUSIVE_STATE_LIMIT:",
)


@dataclass(frozen=True)
class NoGoodRecord:
    """One exact master cut backed by a proven-infeasible operational set."""

    iteration: int
    selected_indices: tuple[int, ...]
    selected_venue_ids: tuple[str, ...]
    rhs: int
    proof_kind: str
    proof_blockers: tuple[str, ...] = ()


@dataclass
class OperationalOptimizationResult:
    """Fail-closed result of one policy's operational decomposition."""

    scenario: str
    status: str
    ready: bool
    certified: bool
    certification_class: str
    blockers: list[str]
    iteration_count: int
    no_good_ledger: list[NoGoodRecord] = field(default_factory=list)
    selected_indices: np.ndarray = field(
        default_factory=lambda: np.asarray([], dtype=int)
    )
    selected_venue_ids: list[str] = field(default_factory=list)
    selected_frame: pd.DataFrame = field(default_factory=pd.DataFrame)
    public_assignments: pd.DataFrame = field(default_factory=pd.DataFrame)
    availability_witness_sha256: str | None = None
    bundle_counts: dict[str, int] = field(default_factory=dict)
    spatial_metrics: dict[str, Any] = field(default_factory=dict)
    operational_audit_summary: dict[str, Any] = field(default_factory=dict)
    spatial_result: ScenarioResult | None = field(default=None, repr=False)
    gpu_is_proof: bool = GPU_IS_PROOF
    dates_exposed: bool = False

    @property
    def no_good_count(self) -> int:
        return len(self.no_good_ledger)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-ready public summary without availability dates."""

        return {
            "scenario": self.scenario,
            "status": self.status,
            "ready": bool(self.ready),
            "certified": bool(self.certified),
            "certification_class": self.certification_class,
            "blockers": list(self.blockers),
            "iteration_count": int(self.iteration_count),
            "no_good_count": int(self.no_good_count),
            "no_good_ledger": [asdict(value) for value in self.no_good_ledger],
            "selected_venue_ids": list(self.selected_venue_ids),
            "availability_witness_sha256": self.availability_witness_sha256,
            "bundle_counts": dict(self.bundle_counts),
            "spatial_metrics": dict(self.spatial_metrics),
            "operational_audit_summary": dict(self.operational_audit_summary),
            "public_assignment_count": int(len(self.public_assignments)),
            "gpu_is_proof": False,
            "dates_exposed": False,
        }


@dataclass
class _JointBundleResult:
    status: str
    blockers: list[str]
    visit_requirements: pd.DataFrame = field(default_factory=pd.DataFrame)
    chosen_options: pd.DataFrame = field(default_factory=pd.DataFrame)
    option_count: int = 0
    solver_status: int | None = None
    solver_message: str = ""

    @property
    def proven_infeasible(self) -> bool:
        return self.status == "PROVEN_INFEASIBLE"


class _NoGoodSpatialMILP(SpatialMILP):
    """Frozen SpatialMILP with additive exact selected-set exclusions."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._operational_no_goods: list[tuple[int, ...]] = []
        super().__init__(*args, **kwargs)

    @property
    def operational_no_goods(self) -> tuple[tuple[int, ...], ...]:
        return tuple(self._operational_no_goods)

    def add_exact_no_good(self, selected_indices: Iterable[int]) -> tuple[int, ...]:
        selected = tuple(sorted({int(value) for value in selected_indices}))
        visit_count = int(
            self.config.get("optimization", {}).get("visit_count", REQUIRED_VISIT_COUNT)
        )
        if len(selected) != visit_count:
            raise ContractError(
                f"Exact no-good requires {visit_count} distinct selected indices; "
                f"got {len(selected)}"
            )
        if selected and (selected[0] < 0 or selected[-1] >= self.nx):
            raise ContractError("Exact no-good contains an out-of-range candidate index")
        if selected not in self._operational_no_goods:
            self._operational_no_goods.append(selected)
        return selected

    def _constraint_with_floors(
        self, floors: dict[str, tuple[str, float]]
    ) -> LinearConstraint:
        constraint = super()._constraint_with_floors(floors)
        if not self._operational_no_goods:
            return constraint
        rows: list[int] = []
        cols: list[int] = []
        for row_index, selected in enumerate(self._operational_no_goods):
            rows.extend([row_index] * len(selected))
            cols.extend(selected)
        values = np.ones(len(cols), dtype=float)
        cuts = sparse.coo_matrix(
            (values, (rows, cols)),
            shape=(len(self._operational_no_goods), self.nvars),
        ).tocsr()
        visit_count = int(
            self.config.get("optimization", {}).get("visit_count", REQUIRED_VISIT_COUNT)
        )
        return LinearConstraint(
            sparse.vstack([constraint.A, cuts], format="csr"),
            np.concatenate(
                [
                    np.asarray(constraint.lb, dtype=float),
                    np.full(len(self._operational_no_goods), -np.inf),
                ]
            ),
            np.concatenate(
                [
                    np.asarray(constraint.ub, dtype=float),
                    np.full(len(self._operational_no_goods), visit_count - 1.0),
                ]
            ),
        )


def _failure_result(
    scenario: str,
    status: str,
    blockers: Iterable[str],
    *,
    iteration_count: int = 0,
    no_goods: Iterable[NoGoodRecord] = (),
    spatial_result: ScenarioResult | None = None,
) -> OperationalOptimizationResult:
    return OperationalOptimizationResult(
        scenario=scenario,
        status=status,
        ready=False,
        certified=False,
        certification_class="NOT_CERTIFIED",
        blockers=list(dict.fromkeys(str(value) for value in blockers)),
        iteration_count=int(iteration_count),
        no_good_ledger=list(no_goods),
        spatial_result=spatial_result,
        spatial_metrics=dict(spatial_result.metrics) if spatial_result is not None else {},
    )


def _truth(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().upper() in {"TRUE", "1", "YES", "Y"}


def prepare_spatial_patterns(
    candidates: pd.DataFrame,
    *,
    patterns: GridPatternData | None = None,
    coverage: sparse.spmatrix | None = None,
    grid: pd.DataFrame | None = None,
) -> GridPatternData:
    """Validate prebuilt patterns or losslessly compress sparse coverage/grid."""

    if "venue_id" not in candidates.columns:
        raise ContractError("Operational candidates are missing venue_id")
    candidate_ids = candidates["venue_id"].astype(str).to_numpy()
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ContractError("Operational candidates contain duplicate venue_id values")
    if patterns is not None:
        if coverage is not None or grid is not None:
            raise ContractError(
                "Supply either prebuilt patterns or sparse coverage+grid, not both"
            )
        if not np.array_equal(
            np.asarray(patterns.candidate_ids, dtype=str), candidate_ids
        ):
            raise ContractError(
                "Prebuilt pattern candidate order does not exactly match candidates"
            )
        return patterns
    if coverage is None or grid is None:
        raise ContractError("Sparse coverage and grid are both required without patterns")
    if not sparse.issparse(coverage):
        raise ContractError("Coverage must remain a scipy sparse matrix")
    required_grid = {
        "sigungu",
        "elderly_population",
        "need_weighted_population",
        "high_need_population",
    }
    missing = sorted(required_grid - set(grid.columns))
    if missing:
        raise ContractError(f"Grid is missing policy columns: {missing}")
    matrix = coverage.tocsr(copy=True)
    if matrix.shape != (len(candidates), len(grid)):
        raise ContractError(
            f"Coverage shape {matrix.shape} does not match "
            f"candidates/grid {(len(candidates), len(grid))}"
        )
    if matrix.data.size and (
        not np.isfinite(matrix.data).all() or (matrix.data <= 0).any()
    ):
        raise ContractError("Sparse coverage contains nonpositive or nonfinite entries")
    return compress_grid_patterns(matrix, candidate_ids, grid.reset_index(drop=True))


def _required_bundles(
    bundle_values: pd.DataFrame, config: Mapping[str, Any]
) -> tuple[list[str], pd.DataFrame]:
    required_columns = {"venue_id", "bundle_id", "bundle_value"}
    missing = sorted(required_columns - set(bundle_values.columns))
    if missing:
        raise ContractError(f"Bundle values are missing columns: {missing}")
    values = bundle_values.loc[:, ["venue_id", "bundle_id", "bundle_value"]].copy()
    values["venue_id"] = values["venue_id"].astype(str)
    values["bundle_id"] = values["bundle_id"].astype(str).str.strip()
    if values["bundle_id"].eq("").any():
        raise ContractError("Bundle values contain blank bundle IDs")
    if values.duplicated(["venue_id", "bundle_id"]).any():
        raise ContractError("Bundle values contain duplicate venue_id x bundle_id rows")
    values["bundle_value"] = pd.to_numeric(
        values["bundle_value"], errors="coerce"
    ).fillna(0.0)
    section = config.get("operational_optimizer", {})
    configured = section.get("required_bundle_ids")
    bundles = (
        [str(value).strip() for value in configured]
        if configured is not None
        else sorted(values["bundle_id"].unique())
    )
    if len(bundles) != REQUIRED_BUNDLE_COUNT or len(set(bundles)) != len(bundles):
        raise ContractError(
            f"Operational optimizer requires exactly {REQUIRED_BUNDLE_COUNT} "
            f"distinct bundles; got {bundles}"
        )
    if any(not value for value in bundles):
        raise ContractError("Required bundle IDs must be nonblank")
    minimum = int(
        config.get("bundle_assignment", {}).get(
            "primary_minimum_per_bundle", REQUIRED_BUNDLE_MINIMUM
        )
    )
    if minimum != REQUIRED_BUNDLE_MINIMUM:
        raise ContractError("The frozen five-bundle minimum must remain exactly one")
    return bundles, values.loc[values["bundle_id"].isin(bundles)].reset_index(
        drop=True
    )


def _normalize_spatial_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Map the operational config's frozen constraints to SpatialMILP keys.

    Stage 4 Operational Final names the section ``candidate_constraints``;
    the inherited Stage 4.2 solver reads ``candidate_expansion``.  Keeping the
    translation here lets the runner pass its frozen config unchanged.  If a
    caller supplies both spellings, disagreement is a hard input error rather
    than an implicit precedence rule.
    """

    normalized = deepcopy(dict(config))
    current = config.get("candidate_constraints")
    legacy = config.get("candidate_expansion")
    if current is not None and not isinstance(current, Mapping):
        raise ContractError("candidate_constraints must be a mapping")
    if legacy is not None and not isinstance(legacy, Mapping):
        raise ContractError("candidate_expansion must be a mapping")
    if current is None and legacy is None:
        raise ContractError("Frozen candidate constraints are missing")

    resolved: dict[str, Any] = {}
    for key, expected in _FROZEN_CANDIDATE_CONSTRAINTS.items():
        current_has = isinstance(current, Mapping) and key in current
        legacy_has = isinstance(legacy, Mapping) and key in legacy
        if not current_has and not legacy_has:
            raise ContractError(f"Frozen candidate constraint is missing: {key}")
        current_value = current[key] if current_has else None
        legacy_value = legacy[key] if legacy_has else None
        if current_has and legacy_has and current_value != legacy_value:
            raise ContractError(f"Conflicting candidate constraint aliases: {key}")
        value = current_value if current_has else legacy_value
        if isinstance(expected, bool):
            value = _truth(value)
        else:
            try:
                value = int(value)
            except (TypeError, ValueError) as exc:
                raise ContractError(
                    f"Frozen candidate constraint is not an integer: {key}"
                ) from exc
        if value != expected:
            raise ContractError(
                f"Frozen candidate constraint drifted: {key}={value!r}; "
                f"expected {expected!r}"
            )
        resolved[key] = value

    expansion = dict(legacy) if isinstance(legacy, Mapping) else {}
    expansion.update(resolved)
    normalized["candidate_expansion"] = expansion
    return normalized


def _validate_master_contract(
    candidates: pd.DataFrame,
    config: Mapping[str, Any],
    gap_threshold: float,
) -> None:
    visit_count = int(
        config.get("optimization", {}).get("visit_count", REQUIRED_VISIT_COUNT)
    )
    if visit_count != REQUIRED_VISIT_COUNT:
        raise ContractError(
            f"Operational Stage 4 visit_count must remain {REQUIRED_VISIT_COUNT}; "
            f"got {visit_count}"
        )
    if not (0.0 <= gap_threshold <= MAX_CERTIFIED_RELATIVE_GAP):
        raise ContractError(
            f"Certified relative gap must be within [0,{MAX_CERTIFIED_RELATIVE_GAP}]"
        )
    required_candidate_columns = {
        "venue_id",
        "admin_code",
        "sigungu",
        "cluster_id",
    }
    missing = sorted(required_candidate_columns - set(candidates.columns))
    if missing:
        raise ContractError(f"Operational candidates are missing columns: {missing}")
    ids = candidates["venue_id"].astype(str)
    if ids.duplicated().any():
        raise ContractError("Operational candidates contain duplicate venue IDs")
    if ids.str.startswith("BUSPROXY_").any():
        raise ContractError("BUSPROXY spatial anchors cannot enter the operational MILP")
    require_verified = bool(
        config.get("operational_optimizer", {}).get(
            "require_field_verified", True
        )
    )
    if require_verified:
        if "field_verified" not in candidates.columns:
            raise ContractError("Operational candidates are missing field_verified")
        if not candidates["field_verified"].map(_truth).all():
            raise ContractError("Every operational candidate must be field_verified")


def _visit_requirements(
    selected: pd.DataFrame, config: Mapping[str, Any]
) -> pd.DataFrame:
    selected = selected.reset_index(drop=True)
    default_capacity = config.get("operational_optimizer", {}).get(
        "required_vehicle_capacity"
    )
    if "required_vehicle_capacity" in selected.columns:
        capacities = pd.to_numeric(
            selected["required_vehicle_capacity"], errors="coerce"
        )
    elif default_capacity is not None:
        capacities = pd.Series(float(default_capacity), index=selected.index)
    else:
        raise ContractError(
            "required_vehicle_capacity must be present on candidates or in "
            "operational_optimizer config"
        )
    if (~np.isfinite(capacities) | capacities.le(0)).any():
        raise ContractError("Selected visits have invalid required vehicle capacity")
    return pd.DataFrame(
        {
            "visit_id": [f"OP_VISIT_{index + 1:02d}" for index in range(len(selected))],
            "venue_id": selected["venue_id"].astype(str).tolist(),
            "required_vehicle_capacity": capacities.astype(float).tolist(),
        }
    )


def _is_option_enumeration_blocker(value: str) -> bool:
    return value.startswith(_LOCAL_OPTION_BLOCKER_PREFIXES) or value.startswith(
        _OPTION_ENUMERATION_ONLY_BLOCKER_PREFIXES
    )


def _enumerate_exact_options(
    visits: pd.DataFrame,
    bundles: list[str],
    bundle_values: pd.DataFrame,
    tables: OperationalTables | Mapping[str, pd.DataFrame],
    *,
    expected_graph_sha: str | None,
) -> _JointBundleResult:
    option_frames: list[pd.DataFrame] = []
    local_blockers: list[str] = []
    audit_summaries: list[dict[str, Any]] = []
    for bundle_id in bundles:
        fixed = visits.copy()
        fixed["bundle_id"] = bundle_id
        audit = audit_operational_contract(
            tables,
            fixed[
                [
                    "visit_id",
                    "venue_id",
                    "bundle_id",
                    "required_vehicle_capacity",
                ]
            ],
            required_visit_count=REQUIRED_VISIT_COUNT,
            required_bundle_ids=[bundle_id],
            expected_graph_sha=expected_graph_sha,
            # Only exact option generation is needed here.  The joint MILP and
            # final full audit below establish global feasibility.
            assignment_search_state_limit=1,
        )
        audit_summaries.append(dict(audit.summary))
        invalid = [
            value
            for value in audit.blockers
            if not _is_option_enumeration_blocker(str(value))
        ]
        if invalid:
            return _JointBundleResult(
                status="INPUT_INVALID",
                blockers=invalid,
                option_count=sum(len(frame) for frame in option_frames),
            )
        local_blockers.extend(str(value) for value in audit.blockers)
        if not audit.feasible_options.empty:
            frame = audit.feasible_options.copy()
            frame["bundle_id"] = bundle_id
            option_frames.append(frame)

    options = (
        pd.concat(option_frames, ignore_index=True).drop_duplicates()
        if option_frames
        else pd.DataFrame()
    )
    visits_with_options = (
        set(options["visit_id"].astype(str)) if not options.empty else set()
    )
    missing_visits = sorted(set(visits["visit_id"].astype(str)) - visits_with_options)
    if missing_visits:
        return _JointBundleResult(
            status="PROVEN_INFEASIBLE",
            blockers=[
                "NO_OPERATIONAL_OPTION_ACROSS_ALL_FIVE_BUNDLES:"
                + ",".join(missing_visits),
                *local_blockers,
            ],
            option_count=int(len(options)),
        )

    full_values = pd.MultiIndex.from_product(
        [visits["venue_id"].astype(str).tolist(), bundles],
        names=["venue_id", "bundle_id"],
    ).to_frame(index=False)
    full_values = full_values.merge(
        bundle_values,
        on=["venue_id", "bundle_id"],
        how="left",
        validate="one_to_one",
    )
    full_values["bundle_value"] = pd.to_numeric(
        full_values["bundle_value"], errors="coerce"
    ).fillna(0.0)
    full_values["bundle_value_percentile"] = full_values.groupby("venue_id")[
        "bundle_value"
    ].rank(method="average", pct=True)
    options = options.merge(
        full_values,
        on=["venue_id", "bundle_id"],
        how="left",
        validate="many_to_one",
    )
    return _JointBundleResult(
        status="OPTIONS_READY",
        blockers=[],
        visit_requirements=visits.copy(),
        chosen_options=options.reset_index(drop=True),
        option_count=int(len(options)),
    )


def _add_group_constraint(
    frame: pd.DataFrame,
    groups: Iterable[tuple[Any, pd.Index]],
    *,
    lower: float,
    upper: float,
    rows: list[int],
    cols: list[int],
    vals: list[float],
    lbs: list[float],
    ubs: list[float],
) -> None:
    del frame  # The grouped indices are the only data required here.
    for _, indices in groups:
        row = len(lbs)
        for index in indices:
            rows.append(row)
            cols.append(int(index))
            vals.append(1.0)
        lbs.append(float(lower))
        ubs.append(float(upper))


def _solve_joint_bundle_assignment(
    option_result: _JointBundleResult,
    bundles: list[str],
    *,
    time_limit_sec: float,
) -> _JointBundleResult:
    options = option_result.chosen_options.reset_index(drop=True).copy()
    if options.empty:
        return _JointBundleResult(
            status="PROVEN_INFEASIBLE",
            blockers=["JOINT_OPERATIONAL_OPTION_SET_EMPTY"],
        )
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    lbs: list[float] = []
    ubs: list[float] = []
    _add_group_constraint(
        options,
        options.groupby("visit_id", sort=False).groups.items(),
        lower=1.0,
        upper=1.0,
        rows=rows,
        cols=cols,
        vals=vals,
        lbs=lbs,
        ubs=ubs,
    )
    for bundle_id in bundles:
        indices = options.index[options["bundle_id"].eq(bundle_id)]
        if len(indices) == 0:
            return _JointBundleResult(
                status="PROVEN_INFEASIBLE",
                blockers=[f"NO_OPERATIONAL_OPTION_FOR_REQUIRED_BUNDLE:{bundle_id}"],
                option_count=len(options),
            )
        _add_group_constraint(
            options,
            [(bundle_id, indices)],
            lower=float(REQUIRED_BUNDLE_MINIMUM),
            upper=float(REQUIRED_VISIT_COUNT),
            rows=rows,
            cols=cols,
            vals=vals,
            lbs=lbs,
            ubs=ubs,
        )
    for keys in (
        ["team_id", "date"],
        ["vehicle_id", "date"],
        ["venue_id", "date"],
    ):
        _add_group_constraint(
            options,
            options.groupby(keys, sort=False).groups.items(),
            lower=-np.inf,
            upper=1.0,
            rows=rows,
            cols=cols,
            vals=vals,
            lbs=lbs,
            ubs=ubs,
        )
    matrix = sparse.coo_matrix(
        (vals, (rows, cols)), shape=(len(lbs), len(options))
    ).tocsr()
    objective = -options["bundle_value_percentile"].to_numpy(float)
    result = milp(
        objective,
        integrality=np.ones(len(options), dtype=np.int8),
        bounds=Bounds(np.zeros(len(options)), np.ones(len(options))),
        constraints=LinearConstraint(
            matrix, np.asarray(lbs, dtype=float), np.asarray(ubs, dtype=float)
        ),
        options={
            "time_limit": max(0.001, float(time_limit_sec)),
            "mip_rel_gap": 0.0,
            "presolve": True,
            "disp": False,
        },
    )
    status = int(result.status)
    message = str(result.message)
    if status == 2:
        return _JointBundleResult(
            status="PROVEN_INFEASIBLE",
            blockers=["JOINT_BUNDLE_OPERATIONAL_ASSIGNMENT_INFEASIBLE"],
            option_count=len(options),
            solver_status=status,
            solver_message=message,
        )
    if status != 0 or result.x is None:
        return _JointBundleResult(
            status="INCONCLUSIVE",
            blockers=[f"JOINT_ASSIGNMENT_SOLVER_NOT_CERTIFIED:{status}:{message}"],
            option_count=len(options),
            solver_status=status,
            solver_message=message,
        )
    chosen = options.loc[np.asarray(result.x, dtype=float) >= 0.5].copy()
    if (
        len(chosen) != REQUIRED_VISIT_COUNT
        or chosen["visit_id"].nunique() != REQUIRED_VISIT_COUNT
        or any(int(chosen["bundle_id"].eq(value).sum()) < 1 for value in bundles)
    ):
        return _JointBundleResult(
            status="INCONCLUSIVE",
            blockers=["JOINT_ASSIGNMENT_SOLUTION_CONTRACT_MISMATCH"],
            option_count=len(options),
            solver_status=status,
            solver_message=message,
        )
    visits = chosen[
        ["visit_id", "venue_id", "bundle_id", "required_vehicle_capacity"]
    ].sort_values("visit_id", kind="stable")
    return _JointBundleResult(
        status="FEASIBLE_CERTIFIED",
        blockers=[],
        visit_requirements=visits.reset_index(drop=True),
        chosen_options=chosen.sort_values("visit_id", kind="stable").reset_index(
            drop=True
        ),
        option_count=len(options),
        solver_status=status,
        solver_message=message,
    )


def _canonical_witness_hash(assignments: pd.DataFrame) -> str:
    columns = [
        "visit_id",
        "venue_id",
        "bundle_id",
        "date",
        "team_id",
        "base_id",
        "vehicle_id",
        "vehicle_type",
        "capacity",
        "required_vehicle_capacity",
        "distance_km",
        "travel_minutes_proxy",
        "graph_sha",
    ]
    frame = assignments.loc[:, columns].sort_values("visit_id", kind="stable").copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.strftime(
        "%Y-%m-%d"
    )
    for column in frame.columns:
        if column == "date":
            continue
        if pd.api.types.is_numeric_dtype(frame[column]):
            frame[column] = frame[column].map(
                lambda value: None if pd.isna(value) else float(value)
            )
        else:
            frame[column] = frame[column].astype(str)
    payload = json.dumps(
        frame.to_dict("records"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _public_assignments(assignments: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "visit_id",
        "venue_id",
        "bundle_id",
        "team_id",
        "base_id",
        "vehicle_id",
        "vehicle_type",
        "capacity",
        "required_vehicle_capacity",
        "distance_km",
        "travel_minutes_proxy",
        "graph_sha",
    ]
    public = assignments.loc[:, columns].copy()
    if any("date" in str(column).lower() for column in public.columns):
        raise RuntimeError("Public operational assignments must not expose dates")
    return public.sort_values("visit_id", kind="stable").reset_index(drop=True)


def _spatial_infeasibility_is_certified(exc: SolverCertificationError) -> bool:
    telemetry = list(getattr(exc, "telemetry", []) or [])
    return bool(
        telemetry
        and telemetry[-1].stage_index == 1
        and telemetry[-1].solver_status == "INFEASIBLE"
    )


def _spatial_certification_blockers(
    result: ScenarioResult, *, scenario: str, gap_threshold: float
) -> list[str]:
    blockers: list[str] = []
    if not result.certified:
        blockers.append(f"SPATIAL_CERTIFICATION_CLASS:{result.certification_class}")
    if result.scenario != scenario:
        blockers.append(f"SPATIAL_SCENARIO_MISMATCH:{result.scenario}:{scenario}")
    if result.candidate_set != "operational":
        blockers.append(f"SPATIAL_CANDIDATE_SET_MISMATCH:{result.candidate_set}")
    telemetry = list(result.telemetry)
    if len(telemetry) != 4 or [value.stage_index for value in telemetry] != [1, 2, 3, 4]:
        blockers.append(f"SPATIAL_LEXICOGRAPHIC_STAGE_COUNT:{len(telemetry)}")
    if telemetry and not all(value.certified for value in telemetry):
        blockers.append("SPATIAL_LEXICOGRAPHIC_STAGE_UNCERTIFIED")
    gap = result.max_relative_gap
    if gap is None or not math.isfinite(float(gap)):
        blockers.append("SPATIAL_MAX_RELATIVE_GAP_MISSING")
    elif float(gap) > gap_threshold + 1e-12:
        blockers.append(f"SPATIAL_MAX_RELATIVE_GAP:{gap}")
    return blockers


def optimize_operational_scenario(
    candidates: pd.DataFrame,
    bundle_values: pd.DataFrame,
    operational_tables: OperationalTables | Mapping[str, pd.DataFrame],
    config: dict[str, Any],
    *,
    scenario: str,
    patterns: GridPatternData | None = None,
    coverage: sparse.spmatrix | None = None,
    grid: pd.DataFrame | None = None,
    efficiency_reference: float | None = None,
    greedy_floor: float | None = None,
    expected_graph_sha: str | None = None,
) -> OperationalOptimizationResult:
    """Optimize one frozen policy with exact operational no-good decomposition.

    A returned ``ready=True`` result has both a certified four-stage spatial
    result and a successful final :func:`audit_operational_contract`.  An
    iteration, wall-time, state, bundle, or solver-certification limit returns
    an inconclusive fail-closed result.  Only exact subproblem infeasibility is
    allowed to add a no-good cut.
    """

    started = time.monotonic()
    section = config.get("operational_optimizer", {})
    max_iterations = int(section.get("max_no_good_iterations", 100))
    wall_time_limit = float(section.get("wall_time_limit_sec", 7200.0))
    spatial_stage_limit = float(
        section.get(
            "spatial_time_limit_per_stage_sec",
            config.get("solver", {}).get(
                "operational_final_time_per_stage_sec", 600.0
            ),
        )
    )
    bundle_time_limit = float(
        section.get(
            "bundle_time_limit_sec",
            config.get("bundle_assignment", {}).get("time_limit_sec", 120.0),
        )
    )
    assignment_state_limit = int(
        section.get("assignment_search_state_limit", 500_000)
    )
    gap_threshold = float(
        config.get("solver", {}).get(
            "near_optimal_relative_gap_max", MAX_CERTIFIED_RELATIVE_GAP
        )
    )
    no_goods: list[NoGoodRecord] = []
    if scenario not in {"efficiency", "balanced", "equity"}:
        return _failure_result(
            scenario, "INPUT_CONTRACT_INVALID", [f"UNKNOWN_SCENARIO:{scenario}"]
        )
    try:
        if max_iterations <= 0 or wall_time_limit <= 0 or spatial_stage_limit <= 0:
            raise ContractError("Operational decomposition limits must be positive")
        if bundle_time_limit <= 0 or assignment_state_limit <= 0:
            raise ContractError("Operational subproblem limits must be positive")
        spatial_config = _normalize_spatial_config(config)
        _validate_master_contract(candidates, spatial_config, gap_threshold)
        bundles, normalized_bundle_values = _required_bundles(
            bundle_values, spatial_config
        )
        prepared_patterns = prepare_spatial_patterns(
            candidates,
            patterns=patterns,
            coverage=coverage,
            grid=grid,
        )
        model = _NoGoodSpatialMILP(
            candidates,
            prepared_patterns,
            spatial_config,
            candidate_set="operational",
        )
    except (ContractError, ValueError, TypeError) as exc:
        return _failure_result(
            scenario, "INPUT_CONTRACT_INVALID", [f"{type(exc).__name__}:{exc}"]
        )

    for iteration in range(1, max_iterations + 1):
        remaining = wall_time_limit - (time.monotonic() - started)
        if remaining <= 0:
            return _failure_result(
                scenario,
                "INCONCLUSIVE_WALL_TIME_LIMIT",
                ["OPERATIONAL_DECOMPOSITION_WALL_TIME_LIMIT"],
                iteration_count=iteration - 1,
                no_goods=no_goods,
            )
        stage_limit = min(spatial_stage_limit, max(0.001, remaining / 4.0))
        try:
            spatial_result = model.solve(
                scenario,
                efficiency_reference=efficiency_reference,
                greedy_floor=greedy_floor,
                time_limit_per_stage=stage_limit,
                gap_threshold=gap_threshold,
            )
        except SolverCertificationError as exc:
            if no_goods and _spatial_infeasibility_is_certified(exc):
                return OperationalOptimizationResult(
                    scenario=scenario,
                    status="NO_OPERATIONALLY_FEASIBLE_PLAN",
                    ready=False,
                    certified=True,
                    certification_class="CERTIFIED_INFEASIBLE_AFTER_EXACT_NO_GOODS",
                    blockers=["NO_OPERATIONALLY_FEASIBLE_20_VISIT_PLAN"],
                    iteration_count=iteration,
                    no_good_ledger=no_goods,
                )
            return _failure_result(
                scenario,
                "SPATIAL_SOLVER_INCONCLUSIVE",
                [f"{type(exc).__name__}:{exc}"],
                iteration_count=iteration,
                no_goods=no_goods,
            )
        certification_blockers = _spatial_certification_blockers(
            spatial_result,
            scenario=scenario,
            gap_threshold=gap_threshold,
        )
        if certification_blockers:
            return _failure_result(
                scenario,
                "SPATIAL_CERTIFICATION_FAILED",
                certification_blockers,
                iteration_count=iteration,
                no_goods=no_goods,
                spatial_result=spatial_result,
            )
        if len(spatial_result.selected_indices) != REQUIRED_VISIT_COUNT:
            return _failure_result(
                scenario,
                "SPATIAL_SELECTION_CONTRACT_FAILED",
                [
                    f"SPATIAL_VISIT_COUNT:{len(spatial_result.selected_indices)}"
                ],
                iteration_count=iteration,
                no_goods=no_goods,
                spatial_result=spatial_result,
            )

        selected = candidates.iloc[spatial_result.selected_indices].copy().reset_index(
            drop=True
        )
        try:
            visits = _visit_requirements(selected, spatial_config)
            option_result = _enumerate_exact_options(
                visits,
                bundles,
                normalized_bundle_values,
                operational_tables,
                expected_graph_sha=expected_graph_sha,
            )
        except (ContractError, ValueError, TypeError) as exc:
            return _failure_result(
                scenario,
                "OPERATIONAL_INPUT_INVALID",
                [f"{type(exc).__name__}:{exc}"],
                iteration_count=iteration,
                no_goods=no_goods,
                spatial_result=spatial_result,
            )
        if option_result.status == "INPUT_INVALID":
            return _failure_result(
                scenario,
                "OPERATIONAL_INPUT_INVALID",
                option_result.blockers,
                iteration_count=iteration,
                no_goods=no_goods,
                spatial_result=spatial_result,
            )
        remaining = wall_time_limit - (time.monotonic() - started)
        if remaining <= 0:
            return _failure_result(
                scenario,
                "INCONCLUSIVE_WALL_TIME_LIMIT",
                ["OPERATIONAL_DECOMPOSITION_WALL_TIME_LIMIT"],
                iteration_count=iteration,
                no_goods=no_goods,
                spatial_result=spatial_result,
            )
        joint = (
            option_result
            if option_result.proven_infeasible
            else _solve_joint_bundle_assignment(
                option_result,
                bundles,
                time_limit_sec=min(bundle_time_limit, remaining),
            )
        )
        if joint.proven_infeasible:
            exact = model.add_exact_no_good(spatial_result.selected_indices)
            no_goods.append(
                NoGoodRecord(
                    iteration=iteration,
                    selected_indices=exact,
                    selected_venue_ids=tuple(
                        sorted(spatial_result.selected_venue_ids)
                    ),
                    rhs=REQUIRED_VISIT_COUNT - 1,
                    proof_kind=(
                        "NO_OPTIONS_ACROSS_ALL_BUNDLES"
                        if option_result.proven_infeasible
                        else "JOINT_ASSIGNMENT_MILP_INFEASIBLE"
                    ),
                    proof_blockers=tuple(joint.blockers),
                )
            )
            continue
        if joint.status != "FEASIBLE_CERTIFIED":
            return _failure_result(
                scenario,
                "OPERATIONAL_SUBPROBLEM_INCONCLUSIVE",
                joint.blockers,
                iteration_count=iteration,
                no_goods=no_goods,
                spatial_result=spatial_result,
            )

        final_audit: OperationalContractAudit = audit_operational_contract(
            operational_tables,
            joint.visit_requirements,
            required_visit_count=REQUIRED_VISIT_COUNT,
            required_bundle_ids=bundles,
            expected_graph_sha=expected_graph_sha,
            assignment_search_state_limit=assignment_state_limit,
        )
        if not final_audit.ready:
            return _failure_result(
                scenario,
                "FINAL_OPERATIONAL_AUDIT_FAILED_OR_INCONCLUSIVE",
                final_audit.blockers,
                iteration_count=iteration,
                no_goods=no_goods,
                spatial_result=spatial_result,
            )
        if time.monotonic() - started > wall_time_limit:
            return _failure_result(
                scenario,
                "INCONCLUSIVE_WALL_TIME_LIMIT",
                ["OPERATIONAL_DECOMPOSITION_WALL_TIME_LIMIT"],
                iteration_count=iteration,
                no_goods=no_goods,
                spatial_result=spatial_result,
            )
        assignments = final_audit.selected_assignments.copy()
        witness_sha = _canonical_witness_hash(assignments)
        public = _public_assignments(assignments)
        plan = spatial_result.selected_frame.copy()
        plan = plan.merge(
            public,
            on="venue_id",
            how="left",
            validate="one_to_one",
            suffixes=("", "_operational"),
        )
        if len(plan) != REQUIRED_VISIT_COUNT or plan["team_id"].isna().any():
            return _failure_result(
                scenario,
                "FINAL_PLAN_JOIN_FAILED",
                ["PUBLIC_OPERATIONAL_ASSIGNMENT_JOIN_MISMATCH"],
                iteration_count=iteration,
                no_goods=no_goods,
                spatial_result=spatial_result,
            )
        bundle_counts = {
            str(key): int(value)
            for key, value in public["bundle_id"].value_counts().sort_index().items()
        }
        if set(bundle_counts) != set(bundles) or any(
            value < REQUIRED_BUNDLE_MINIMUM for value in bundle_counts.values()
        ):
            return _failure_result(
                scenario,
                "FINAL_BUNDLE_CONTRACT_FAILED",
                [f"FINAL_BUNDLE_COUNTS:{bundle_counts}"],
                iteration_count=iteration,
                no_goods=no_goods,
                spatial_result=spatial_result,
            )
        metrics = dict(spatial_result.metrics)
        metrics.update(
            {
                "operational_assignment_count": REQUIRED_VISIT_COUNT,
                "bundle_counts": bundle_counts,
                "availability_witness_sha256": witness_sha,
                "selected_busproxy_count": 0,
                "gpu_is_proof": False,
            }
        )
        return OperationalOptimizationResult(
            scenario=scenario,
            status="OPERATIONALLY_FEASIBLE_CERTIFIED",
            ready=True,
            certified=True,
            certification_class=(
                "OPERATIONALLY_FEASIBLE__" + spatial_result.certification_class
            ),
            blockers=[],
            iteration_count=iteration,
            no_good_ledger=no_goods,
            selected_indices=np.asarray(spatial_result.selected_indices, dtype=int),
            selected_venue_ids=list(spatial_result.selected_venue_ids),
            selected_frame=plan,
            public_assignments=public,
            availability_witness_sha256=witness_sha,
            bundle_counts=bundle_counts,
            spatial_metrics=metrics,
            operational_audit_summary=dict(final_audit.summary),
            spatial_result=spatial_result,
        )

    return _failure_result(
        scenario,
        "INCONCLUSIVE_NO_GOOD_ITERATION_LIMIT",
        [f"NO_GOOD_ITERATION_LIMIT:{max_iterations}"],
        iteration_count=max_iterations,
        no_goods=no_goods,
    )


__all__ = [
    "NoGoodRecord",
    "OperationalOptimizationResult",
    "optimize_operational_scenario",
    "prepare_spatial_patterns",
]
