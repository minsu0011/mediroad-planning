"""Fail-closed candidate-reduction sensitivity for Stage 4.1.

The production Stage 4 run used three candidates per administrative dong.  That
is a computational choice, not a policy truth.  This module constructs the four
pre-declared candidate universes requested by the Stage 4 finalisation contract
and compares solutions without selecting thresholds after seeing the results.

Nothing in this module writes artifacts or changes a CURRENT pointer.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from .candidates import add_candidate_scores, encode_known_overlap
from .types import OptimizationInput, ScenarioSolution, SolveStageResult


DEFAULT_CANDIDATE_SETS = ("top3", "top5", "coarse_pareto", "coverage_cluster")
MANDATORY_COMPARATORS = ("top5", "coarse_pareto")


@dataclass(frozen=True)
class CandidateBuildPolicy:
    """Candidate-set rules declared before any optimisation result is observed."""

    max_candidates: int = 900
    coarse_pareto_tier_max: int = 3
    require_all_admins: bool = True
    coverage_cluster_required: bool = False

    def __post_init__(self) -> None:
        if self.max_candidates <= 0:
            raise ValueError("max_candidates must be positive")
        if self.coarse_pareto_tier_max <= 0:
            raise ValueError("coarse_pareto_tier_max must be positive")


@dataclass(frozen=True)
class CandidateSensitivityThresholds:
    """Pre-registered promotion thresholds for retaining admin Top3.

    Coverage thresholds are deliberately stricter than exact-venue agreement:
    the Stage 3 coverage-equivalent cluster is the relevant spatial unit.
    """

    max_unique_coverage_loss_fraction: float = 0.01
    max_need_weighted_loss_fraction: float = 0.01
    max_high_need_loss_fraction: float = 0.02
    max_min_sigungu_coverage_loss: float = 0.02
    min_selected_admin_jaccard: float = 0.50
    min_selected_coverage_cluster_jaccard: float = 0.50
    max_solver_relative_gap: float = 0.005

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class CandidateSetBuild:
    """In-memory candidate universes and their auditable membership schema."""

    inputs: dict[str, OptimizationInput]
    preflight: pd.DataFrame
    membership: pd.DataFrame
    policy: CandidateBuildPolicy
    unavailable: pd.DataFrame


@dataclass
class CandidateSensitivityResult:
    """Tabular result suitable for a later Stage 4.1 writer."""

    decision: str
    passed: bool
    recommended_candidate_set: str | None
    summary: pd.DataFrame
    comparisons: pd.DataFrame
    thresholds: dict[str, float]
    threshold_fingerprint: str
    notes: list[str]


def _required_candidate_columns() -> set[str]:
    return {
        "venue_id",
        "admin_code",
        "sigungu",
        "cluster_id",
        "shortlist_rank",
        "pareto_tier",
    }


def _validate_source(data: OptimizationInput) -> pd.DataFrame:
    data.validate()
    missing = sorted(_required_candidate_columns() - set(data.candidates.columns))
    if missing:
        raise KeyError(f"Candidate sensitivity missing required columns: {missing}")
    candidates = data.candidates.reset_index(drop=True).copy()
    for column in ["venue_id", "admin_code", "sigungu", "cluster_id"]:
        if candidates[column].isna().any() or candidates[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"Candidate column {column} contains null/blank identifiers")
        candidates[column] = candidates[column].astype(str)
    expected_ids = candidates["venue_id"].to_numpy(str)
    if not np.array_equal(expected_ids, np.asarray(data.venue_ids, dtype=str)):
        raise ValueError("OptimizationInput.venue_ids is not aligned to candidate row order")
    if data.coverage.shape[0] != len(candidates):
        raise ValueError("Coverage rows are not aligned to the full candidate universe")
    bundle_ids = set(data.bundles["venue_id"].astype(str))
    missing_bundles = sorted(set(expected_ids) - bundle_ids)
    if missing_bundles:
        raise ValueError(
            f"Bundle table is missing {len(missing_bundles)} candidate venues; "
            f"sample={missing_bundles[:10]}"
        )
    candidates["_source_row_index"] = np.arange(len(candidates), dtype=np.int64)
    return add_candidate_scores(candidates)


def _sort_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work["_rank_numeric"] = pd.to_numeric(work["shortlist_rank"], errors="coerce")
    work["_pareto_numeric"] = pd.to_numeric(work["pareto_tier"], errors="coerce")
    return work.sort_values(
        [
            "_pareto_numeric",
            "_rank_numeric",
            "candidate_priority_tiebreak",
            "venue_id",
        ],
        ascending=[True, True, False, True],
        na_position="last",
        kind="mergesort",
    )


def _top_n_per_admin(candidates: pd.DataFrame, n: int) -> pd.DataFrame:
    if n <= 0:
        raise ValueError("top-n must be positive")
    ranked = _sort_candidates(candidates)
    return ranked.groupby("admin_code", sort=True, group_keys=False).head(n).copy()


def _cap_preserving_admins(frame: pd.DataFrame, max_candidates: int) -> pd.DataFrame:
    """Apply a deterministic cap while retaining one candidate per admin."""

    if len(frame) <= max_candidates:
        return frame.copy()
    ranked = _sort_candidates(frame)
    required = ranked.groupby("admin_code", sort=True, group_keys=False).head(1)
    if len(required) > max_candidates:
        raise ValueError(
            f"Candidate cap {max_candidates} is below the {len(required)}-admin preservation floor"
        )
    remaining = ranked[~ranked["venue_id"].isin(required["venue_id"])]
    return pd.concat(
        [required, remaining.head(max_candidates - len(required))], ignore_index=True
    )


def _coarse_pareto(candidates: pd.DataFrame, policy: CandidateBuildPolicy) -> pd.DataFrame:
    tier = pd.to_numeric(candidates["pareto_tier"], errors="coerce")
    eligible = candidates[tier.le(policy.coarse_pareto_tier_max)].copy()
    if eligible.empty:
        raise ValueError("coarse_pareto has no candidates within the declared Pareto tiers")
    # A missing admin is repaired from the full universe before the pre-declared
    # cap is applied.  This is not outcome-dependent; it preserves input scope.
    missing = sorted(set(candidates["admin_code"]) - set(eligible["admin_code"]))
    if missing:
        fallback = _sort_candidates(candidates[candidates["admin_code"].isin(missing)])
        fallback = fallback.groupby("admin_code", sort=True, group_keys=False).head(1)
        eligible = pd.concat([eligible, fallback], ignore_index=True).drop_duplicates("venue_id")
    if len(eligible) <= policy.max_candidates:
        return eligible
    # The wider comparator must contain Top3.  Otherwise a difference would
    # mix two reductions and could not identify loss attributable to Top3.
    mandatory = _top_n_per_admin(candidates, 3)
    if len(mandatory) > policy.max_candidates:
        raise ValueError(
            f"coarse Pareto cap {policy.max_candidates} cannot retain the Top3 contract "
            f"({len(mandatory)} rows)"
        )
    eligible = pd.concat([eligible, mandatory], ignore_index=True).drop_duplicates("venue_id")
    remaining = _sort_candidates(
        eligible[~eligible["venue_id"].isin(mandatory["venue_id"])]
    )
    return pd.concat(
        [mandatory, remaining.head(policy.max_candidates - len(mandatory))],
        ignore_index=True,
    )


def _coverage_cluster_representatives(
    candidates: pd.DataFrame, policy: CandidateBuildPolicy
) -> pd.DataFrame:
    """Choose one representative per cluster and repair administrative coverage.

    A representative may be swapped only within its own coverage cluster.  This
    keeps the one-row-per-cluster contract exact while avoiding arbitrary loss
    of an administrative dong whenever a feasible representative assignment
    exists.
    """

    ranked = _sort_candidates(candidates)
    source_admins = sorted(candidates["admin_code"].astype(str).unique())

    if not policy.require_all_admins:
        representatives = ranked.drop_duplicates("cluster_id", keep="first").copy()
        return _cap_preserving_admins(representatives, policy.max_candidates)

    # Reserve a distinct cluster for every admin via deterministic bipartite
    # matching.  A local swap heuristic can strand an admin even when a global
    # representative assignment exists, so use an augmenting-path proof here.
    option_rows: dict[tuple[str, str], pd.Series] = {}
    admin_options: dict[str, list[str]] = {}
    for admin in source_admins:
        admin_rows = ranked[ranked["admin_code"].astype(str).eq(admin)]
        best_by_cluster = admin_rows.drop_duplicates("cluster_id", keep="first")
        admin_options[admin] = best_by_cluster["cluster_id"].astype(str).tolist()
        for _, row in best_by_cluster.iterrows():
            option_rows[(admin, str(row["cluster_id"]))] = row

    matched_cluster: dict[str, str] = {}

    def augment(admin: str, seen: set[str]) -> bool:
        for cluster in admin_options[admin]:
            if cluster in seen:
                continue
            seen.add(cluster)
            previous = matched_cluster.get(cluster)
            if previous is None or augment(previous, seen):
                matched_cluster[cluster] = admin
                return True
        return False

    # Constrained admins first improves determinism and avoids needless deep
    # augmenting paths; the matching result itself is still exact.
    for admin in sorted(source_admins, key=lambda value: (len(admin_options[value]), value)):
        augment(admin, set())
    matched_admins = set(matched_cluster.values())
    unmatched_admins = sorted(set(source_admins) - matched_admins)
    if unmatched_admins:
        raise RuntimeError(
            "No one-representative-per-cluster assignment can retain every admin: "
            f"maximum_matching={len(matched_admins)}/{len(source_admins)}, "
            f"unmatched_sample={unmatched_admins[:10]}"
        )

    reserved = {
        cluster: option_rows[(admin, cluster)]
        for cluster, admin in matched_cluster.items()
    }
    rows: list[pd.Series] = []
    for cluster, group in ranked.groupby("cluster_id", sort=True):
        rows.append(reserved.get(str(cluster), group.iloc[0]))
    representatives = pd.DataFrame(rows).reset_index(drop=True)

    if representatives["cluster_id"].astype(str).duplicated().any():
        raise RuntimeError("coverage_cluster construction produced duplicate clusters")
    return _cap_preserving_admins(representatives, policy.max_candidates)


def _subset_input(data: OptimizationInput, selected: pd.DataFrame, set_id: str) -> OptimizationInput:
    selected = selected.copy()
    if selected["venue_id"].astype(str).duplicated().any():
        raise ValueError(f"{set_id} contains duplicate venue IDs")
    selected["candidate_set_id"] = set_id
    selected = selected.sort_values("_source_row_index", kind="mergesort")
    indices = selected["_source_row_index"].to_numpy(dtype=np.int64)
    selected = selected.drop(columns=["_rank_numeric", "_pareto_numeric", "_source_row_index"], errors="ignore")
    selected = selected.reset_index(drop=True)
    ids = selected["venue_id"].astype(str)
    bundles = data.bundles[data.bundles["venue_id"].astype(str).isin(set(ids))].copy()
    subset = OptimizationInput(
        candidates=selected,
        grids=data.grids.copy(),
        coverage=data.coverage.tocsr()[indices, :].copy(),
        venue_ids=ids.to_numpy(str),
        grid_ids=np.asarray(data.grid_ids, dtype=str).copy(),
        bundles=bundles,
        catchment_id=data.catchment_id,
        metadata={**data.metadata, "candidate_set_id": set_id},
    )
    subset.validate()
    return subset


class _Dinic:
    """Small integer max-flow implementation used for exact feasibility checks."""

    def __init__(self, n: int) -> None:
        self.graph: list[list[list[int]]] = [[] for _ in range(n)]

    def add_edge(self, source: int, target: int, capacity: int) -> None:
        forward = [target, capacity, len(self.graph[target])]
        reverse = [source, 0, len(self.graph[source])]
        self.graph[source].append(forward)
        self.graph[target].append(reverse)

    def max_flow(self, source: int, sink: int) -> int:
        total = 0
        n = len(self.graph)
        while True:
            level = [-1] * n
            level[source] = 0
            queue = [source]
            for node in queue:
                for target, capacity, _ in self.graph[node]:
                    if capacity > 0 and level[target] < 0:
                        level[target] = level[node] + 1
                        queue.append(target)
            if level[sink] < 0:
                return total
            cursor = [0] * n

            def push(node: int, amount: int) -> int:
                if node == sink:
                    return amount
                while cursor[node] < len(self.graph[node]):
                    edge = self.graph[node][cursor[node]]
                    target, capacity, reverse_index = edge
                    if capacity > 0 and level[target] == level[node] + 1:
                        sent = push(target, min(amount, capacity))
                        if sent:
                            edge[1] -= sent
                            self.graph[target][reverse_index][1] += sent
                            return sent
                    cursor[node] += 1
                return 0

            while True:
                sent = push(source, 10**9)
                if not sent:
                    break
                total += sent


def _max_feasible_visits(candidates: pd.DataFrame, config: Mapping[str, Any]) -> int:
    cfg = config["candidate_reduction"]
    work = candidates.copy()
    if bool(config["optimization"].get("exclude_known_overlap", False)):
        overlap = work.get("known_overlap", pd.Series("unknown", index=work.index))
        work = work[encode_known_overlap(overlap).ne(1)].copy()
    if work.empty:
        return 0

    admin_to_sigungu = work.groupby("admin_code")["sigungu"].nunique()
    if (admin_to_sigungu > 1).any():
        bad = admin_to_sigungu[admin_to_sigungu > 1].index.astype(str).tolist()
        raise ValueError(f"Admin codes map to multiple sigungu values: {bad[:10]}")

    sigungus = sorted(work["sigungu"].astype(str).unique())
    admins = sorted(work["admin_code"].astype(str).unique())
    cluster_active = bool(cfg.get("one_per_coverage_cluster", True)) and bool(
        config["optimization"].get("cluster_constraint", True)
    )
    unit_column = "cluster_id" if cluster_active else "venue_id"
    units = sorted(work[unit_column].astype(str).unique())
    labels = ["SOURCE", *[f"S::{x}" for x in sigungus], *[f"A::{x}" for x in admins], *[f"U::{x}" for x in units], "SINK"]
    node = {label: idx for idx, label in enumerate(labels)}
    flow = _Dinic(len(labels))
    source, sink = node["SOURCE"], node["SINK"]
    sigungu_cap = int(cfg["max_visits_per_sigungu"])
    admin_cap = int(cfg["max_visits_per_admin"])
    for sigungu in sigungus:
        flow.add_edge(source, node[f"S::{sigungu}"], sigungu_cap)
    admin_sigungu = work.drop_duplicates("admin_code").set_index("admin_code")["sigungu"].astype(str)
    for admin in admins:
        flow.add_edge(node[f"S::{admin_sigungu.loc[admin]}"], node[f"A::{admin}"], admin_cap)
    for row in work[["admin_code", unit_column]].astype(str).drop_duplicates().itertuples(index=False):
        flow.add_edge(node[f"A::{row[0]}"], node[f"U::{row[1]}"], 1)
    for unit in units:
        flow.add_edge(node[f"U::{unit}"], sink, 1)
    return flow.max_flow(source, sink)


def candidate_set_preflight(
    candidate_inputs: Mapping[str, OptimizationInput],
    source_admins: Sequence[str],
    config: Mapping[str, Any],
    visit_count: int,
) -> pd.DataFrame:
    """Prove basic feasibility before spending solver time."""

    if visit_count <= 0:
        raise ValueError("visit_count must be positive")
    source_admin_set = set(map(str, source_admins))
    rows: list[dict[str, Any]] = []
    require_all = bool(config["candidate_reduction"].get("require_all_admins", True))
    for set_id, data in candidate_inputs.items():
        data.validate()
        candidates = data.candidates
        admins = set(candidates["admin_code"].astype(str))
        missing = sorted(source_admin_set - admins)
        capacity = _max_feasible_visits(candidates, config)
        passed = capacity >= visit_count and (not require_all or not missing)
        rows.append(
            {
                "candidate_set": str(set_id),
                "candidate_count": int(len(candidates)),
                "admin_count": int(len(admins)),
                "missing_admin_count": int(len(missing)),
                "missing_admin_sample": "|".join(missing[:10]),
                "sigungu_count": int(candidates["sigungu"].astype(str).nunique()),
                "coverage_cluster_count": int(candidates["cluster_id"].astype(str).nunique()),
                "max_feasible_visits_exact": int(capacity),
                "requested_visits": int(visit_count),
                "preflight_passed": bool(passed),
            }
        )
    return pd.DataFrame(rows).sort_values("candidate_set").reset_index(drop=True)


def build_candidate_sets(
    data: OptimizationInput,
    config: Mapping[str, Any],
    *,
    strategies: Sequence[str] = DEFAULT_CANDIDATE_SETS,
    policy: CandidateBuildPolicy | None = None,
    visit_count: int | None = None,
) -> CandidateSetBuild:
    """Build aligned Top3/Top5/Pareto/cluster inputs and fail on ambiguity."""

    policy = policy or CandidateBuildPolicy()
    requested = tuple(dict.fromkeys(map(str, strategies)))
    unknown = sorted(set(requested) - set(DEFAULT_CANDIDATE_SETS))
    if unknown:
        raise ValueError(f"Unknown candidate-set strategies: {unknown}")
    if not requested:
        raise ValueError("At least one candidate-set strategy is required")
    candidates = _validate_source(data)
    selected_frames: dict[str, pd.DataFrame] = {}
    unavailable_rows: list[dict[str, Any]] = []
    for strategy in requested:
        if strategy == "top3":
            selected = _top_n_per_admin(candidates, 3)
        elif strategy == "top5":
            selected = _top_n_per_admin(candidates, 5)
        elif strategy == "coarse_pareto":
            selected = _coarse_pareto(candidates, policy)
        else:
            try:
                selected = _coverage_cluster_representatives(candidates, policy)
            except RuntimeError as exc:
                if policy.coverage_cluster_required:
                    raise
                unavailable_rows.append(
                    {
                        "candidate_set": strategy,
                        "available": False,
                        "preflight_passed": False,
                        "unavailable_reason": str(exc),
                    }
                )
                continue
        selected = _cap_preserving_admins(selected, policy.max_candidates)
        selected_frames[strategy] = selected

    inputs = {
        set_id: _subset_input(data, selected, set_id)
        for set_id, selected in selected_frames.items()
    }
    effective_config = deepcopy(dict(config))
    effective_config["candidate_reduction"] = dict(config["candidate_reduction"])
    effective_config["candidate_reduction"]["require_all_admins"] = policy.require_all_admins
    visits = int(visit_count or config["optimization"]["visit_count"])
    if inputs:
        preflight = candidate_set_preflight(
            inputs, candidates["admin_code"].astype(str).unique(), effective_config, visits
        )
    else:
        preflight = pd.DataFrame(columns=["candidate_set", "preflight_passed"])
    preflight["available"] = True
    preflight["unavailable_reason"] = ""
    if unavailable_rows:
        preflight = pd.concat(
            [preflight, pd.DataFrame(unavailable_rows)], ignore_index=True, sort=False
        ).sort_values("candidate_set").reset_index(drop=True)
    failed = preflight[
        preflight["available"].eq(True)
        & ~preflight["preflight_passed"].fillna(False).astype(bool)
    ]
    if not failed.empty:
        details = failed[["candidate_set", "missing_admin_count", "max_feasible_visits_exact"]].to_dict("records")
        raise RuntimeError(f"Candidate-set preflight failed closed: {details}")

    membership_columns = [
        "candidate_set_id",
        "venue_id",
        "admin_code",
        "sigungu",
        "cluster_id",
        "shortlist_rank",
        "pareto_tier",
        "candidate_priority_tiebreak",
    ]
    if inputs:
        membership = pd.concat(
            [subset.candidates[membership_columns] for subset in inputs.values()],
            ignore_index=True,
        ).rename(columns={"candidate_set_id": "candidate_set"})
    else:
        membership = pd.DataFrame(
            columns=["candidate_set", *membership_columns[1:]]
        )
    unavailable = pd.DataFrame(
        unavailable_rows,
        columns=["candidate_set", "available", "preflight_passed", "unavailable_reason"],
    )
    return CandidateSetBuild(
        inputs=inputs,
        preflight=preflight,
        membership=membership,
        policy=policy,
        unavailable=unavailable,
    )


def solve_candidate_sets(
    build: CandidateSetBuild,
    config: Mapping[str, Any],
    scenario: str,
    visit_count: int,
    *,
    solver_name: str = "cp-sat",
    time_limit_sec: float,
    efficiency_reference_by_set: Mapping[str, int] | None = None,
    solve_fn: Callable[..., ScenarioSolution] | None = None,
) -> dict[str, ScenarioSolution]:
    """Run all sets with an identical scenario, seed, and per-stage budget."""

    if time_limit_sec <= 0:
        raise ValueError("time_limit_sec must be positive")
    if scenario != "efficiency":
        missing_references = sorted(
            set(build.inputs) - set(efficiency_reference_by_set or {})
        )
        if missing_references:
            raise ValueError(
                "Non-efficiency candidate sensitivity requires a separately derived "
                f"efficiency reference for every set; missing={missing_references}"
            )
    if solve_fn is None:
        from .optimizer import solve_scenario

        solve_fn = solve_scenario
    solutions: dict[str, ScenarioSolution] = {}
    for set_id in build.inputs:
        local_config = deepcopy(dict(config))
        solve_kwargs: dict[str, Any] = {
            "solver_name": solver_name,
            "time_limit_sec": float(time_limit_sec),
        }
        if scenario != "efficiency":
            solve_kwargs["efficiency_reference"] = int(
                (efficiency_reference_by_set or {})[set_id]
            )
        solutions[set_id] = solve_fn(
            build.inputs[set_id],
            local_config,
            scenario,
            int(visit_count),
            **solve_kwargs,
        )
    return solutions


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return float(len(left & right) / len(union)) if union else 1.0


def _relative_loss(value: float, reference: float) -> float:
    denominator = max(abs(reference), 1e-12)
    return float(max(0.0, reference - value) / denominator)


def _stage_relative_gap(stage: SolveStageResult) -> float:
    if stage.objective_value is None or stage.best_bound is None:
        raise ValueError(f"Solver stage {stage.objective_name} lacks objective/bound")
    value = float(stage.objective_value)
    bound = float(stage.best_bound)
    scale = max(abs(value), abs(bound), 1.0)
    if stage.sense == "max":
        directional = bound - value
    elif stage.sense == "min":
        directional = value - bound
    else:
        raise ValueError(f"Unknown solver objective sense: {stage.sense}")
    if directional < -1e-9 * scale:
        raise ValueError(
            f"Solver stage {stage.objective_name} has a bound inconsistent with {stage.sense}"
        )
    # Match solver_certification.py and OR-Tools reporting: the incumbent is
    # the relative-gap denominator, with a unit floor.
    return float(max(0.0, directional) / max(abs(value), 1.0))


def _objective_metric(scenario: str) -> str:
    if scenario == "efficiency":
        return "unique_elderly_population"
    if scenario == "balanced":
        return "need_weighted_population_scaled"
    if scenario in {"equity", "equity_hard"}:
        return "min_sigungu_coverage_ratio"
    raise ValueError(f"No pre-declared primary objective metric for scenario {scenario!r}")


def _validate_solution_set(
    solutions: Mapping[str, ScenarioSolution], expected_sets: Sequence[str]
) -> tuple[str, str, int]:
    missing = sorted(set(expected_sets) - set(solutions))
    if missing:
        raise ValueError(f"Candidate sensitivity missing required solution sets: {missing}")
    scenario_values = {solutions[key].scenario for key in expected_sets}
    catchment_values = {solutions[key].catchment_id for key in expected_sets}
    visit_values = {int(solutions[key].visit_count) for key in expected_sets}
    if len(scenario_values) != 1 or len(catchment_values) != 1 or len(visit_values) != 1:
        raise ValueError(
            "Candidate solutions must use the same scenario, catchment, and visit count"
        )
    for key in expected_sets:
        solution = solutions[key]
        if solution.status not in {
            "OPTIMAL",
            "FEASIBLE",
            "NEAR_OPTIMAL_CERTIFIED",
            "BEST_KNOWN_FEASIBLE",
        }:
            raise ValueError(f"Candidate set {key} has unusable solver status {solution.status}")
        if len(solution.selected) != solution.visit_count:
            raise ValueError(f"Candidate set {key} does not contain exactly {solution.visit_count} visits")
        if solution.selected["venue_id"].astype(str).duplicated().any():
            raise ValueError(f"Candidate set {key} selected duplicate venues")
        for column in ["admin_code", "sigungu", "cluster_id"]:
            if column not in solution.selected:
                raise KeyError(f"Candidate set {key} selection lacks {column}")
        if not solution.stages:
            raise ValueError(f"Candidate set {key} lacks solver stages/bounds")
    return next(iter(scenario_values)), next(iter(catchment_values)), next(iter(visit_values))


def evaluate_candidate_sensitivity(
    solutions: Mapping[str, ScenarioSolution],
    *,
    thresholds: CandidateSensitivityThresholds | None = None,
    baseline_set: str = "top3",
    reference_sets: Sequence[str] = MANDATORY_COMPARATORS,
    fallback_order: Sequence[str] = ("top5", "coarse_pareto", "coverage_cluster"),
    preflight: pd.DataFrame | None = None,
) -> CandidateSensitivityResult:
    """Evaluate Top3 against pre-declared wider universes.

    The function never chooses the numerically nicest fallback.  If Top3 fails,
    the first usable set in ``fallback_order`` is recommended, making the policy
    independent of the observed result.
    """

    thresholds = thresholds or CandidateSensitivityThresholds()
    expected = (baseline_set, *tuple(reference_sets))
    scenario, catchment, visit_count = _validate_solution_set(solutions, expected)
    if preflight is not None:
        required_columns = {"candidate_set", "preflight_passed"}
        if not required_columns.issubset(preflight.columns):
            raise KeyError(f"Preflight lacks columns {sorted(required_columns - set(preflight.columns))}")
        indexed = preflight.set_index("candidate_set")
        for set_id in expected:
            if set_id not in indexed.index or not bool(indexed.loc[set_id, "preflight_passed"]):
                raise RuntimeError(f"Candidate set {set_id} did not pass preflight")

    metric_names = [
        "unique_elderly_population",
        "high_need_population_scaled",
        "need_weighted_population_scaled",
        "min_sigungu_coverage_ratio",
    ]
    primary = _objective_metric(scenario)
    summary_rows: list[dict[str, Any]] = []
    for set_id, solution in solutions.items():
        missing_metrics = [metric for metric in metric_names if metric not in solution.metrics]
        if missing_metrics:
            raise KeyError(f"Candidate set {set_id} lacks metrics: {missing_metrics}")
        gaps = [_stage_relative_gap(stage) for stage in solution.stages]
        summary_rows.append(
            {
                "candidate_set": set_id,
                "scenario": scenario,
                "catchment_id": catchment,
                "visit_count": visit_count,
                "solver_status": solution.status,
                **{metric: float(solution.metrics[metric]) for metric in metric_names},
                "visit_location_sigungu_count": int(solution.selected["sigungu"].astype(str).nunique()),
                "selected_admin_count": int(solution.selected["admin_code"].astype(str).nunique()),
                "selected_coverage_cluster_count": int(solution.selected["cluster_id"].astype(str).nunique()),
                "objective_metric": primary,
                "objective_value": float(solution.metrics[primary]),
                "max_stage_relative_gap": float(max(gaps)),
                "solver_runtime_sec": float(sum(stage.wall_time_sec for stage in solution.stages)),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values("candidate_set").reset_index(drop=True)
    summary_index = summary.set_index("candidate_set")

    comparisons: list[dict[str, Any]] = []
    base = solutions[baseline_set]
    for reference_set in reference_sets:
        reference = solutions[reference_set]
        base_row = summary_index.loc[baseline_set]
        ref_row = summary_index.loc[reference_set]
        unique_loss = _relative_loss(
            float(base_row["unique_elderly_population"]),
            float(ref_row["unique_elderly_population"]),
        )
        need_loss = _relative_loss(
            float(base_row["need_weighted_population_scaled"]),
            float(ref_row["need_weighted_population_scaled"]),
        )
        high_loss = _relative_loss(
            float(base_row["high_need_population_scaled"]),
            float(ref_row["high_need_population_scaled"]),
        )
        min_sigungu_loss = max(
            0.0,
            float(ref_row["min_sigungu_coverage_ratio"])
            - float(base_row["min_sigungu_coverage_ratio"]),
        )
        admin_jaccard = _jaccard(
            set(base.selected["admin_code"].astype(str)),
            set(reference.selected["admin_code"].astype(str)),
        )
        cluster_jaccard = _jaccard(
            set(base.selected["cluster_id"].astype(str)),
            set(reference.selected["cluster_id"].astype(str)),
        )
        solver_gap = max(
            float(base_row["max_stage_relative_gap"]),
            float(ref_row["max_stage_relative_gap"]),
        )
        checks = {
            "unique_coverage_loss_passed": unique_loss <= thresholds.max_unique_coverage_loss_fraction,
            "need_weighted_loss_passed": need_loss <= thresholds.max_need_weighted_loss_fraction,
            "high_need_loss_passed": high_loss <= thresholds.max_high_need_loss_fraction,
            "min_sigungu_loss_passed": min_sigungu_loss <= thresholds.max_min_sigungu_coverage_loss,
            "admin_jaccard_passed": admin_jaccard >= thresholds.min_selected_admin_jaccard,
            "coverage_cluster_jaccard_passed": cluster_jaccard >= thresholds.min_selected_coverage_cluster_jaccard,
            "solver_gap_passed": solver_gap <= thresholds.max_solver_relative_gap,
        }
        comparisons.append(
            {
                "candidate_set": baseline_set,
                "reference_candidate_set": reference_set,
                "objective_metric": primary,
                "objective_loss_fraction": _relative_loss(
                    float(base_row["objective_value"]), float(ref_row["objective_value"])
                ),
                "unique_coverage_loss_fraction": unique_loss,
                "need_weighted_loss_fraction": need_loss,
                "high_need_loss_fraction": high_loss,
                "min_sigungu_coverage_loss": float(min_sigungu_loss),
                "selected_admin_jaccard": admin_jaccard,
                "selected_coverage_cluster_jaccard": cluster_jaccard,
                "max_solver_relative_gap": solver_gap,
                "runtime_ratio_top3_to_reference": float(base_row["solver_runtime_sec"])
                / max(float(ref_row["solver_runtime_sec"]), 1e-12),
                **checks,
                "comparison_passed": bool(all(checks.values())),
            }
        )
    comparison_frame = pd.DataFrame(comparisons)
    solver_conclusive = bool(comparison_frame["solver_gap_passed"].all())
    top3_passed = bool(comparison_frame["comparison_passed"].all())
    notes = [
        "Thresholds are fingerprinted before comparison; no post-result threshold selection is allowed.",
        "Exact venue identity is diagnostic; coverage-cluster and admin agreement are promotion gates.",
    ]
    if not solver_conclusive:
        decision = "INCONCLUSIVE_CANDIDATE_SENSITIVITY_SOLVER_GAP"
        recommended: str | None = None
        passed = False
        notes.append("At least one compared solve exceeds the pre-registered solver-gap ceiling.")
    elif top3_passed:
        decision = "PASS_RETAIN_TOP3"
        recommended = baseline_set
        passed = True
    else:
        available = [name for name in fallback_order if name in solutions]
        if not available:
            decision = "FAIL_NO_PREDECLARED_FALLBACK"
            recommended = None
            passed = False
        else:
            decision = "PROMOTE_PREDECLARED_WIDER_CANDIDATE_SET"
            recommended = available[0]
            passed = True
            notes.append(
                f"Top3 breached a promotion threshold; fallback order selected {recommended}, not a post-hoc best set."
            )
    return CandidateSensitivityResult(
        decision=decision,
        passed=passed,
        recommended_candidate_set=recommended,
        summary=summary,
        comparisons=comparison_frame,
        thresholds={key: float(value) for key, value in asdict(thresholds).items()},
        threshold_fingerprint=thresholds.fingerprint,
        notes=notes,
    )
