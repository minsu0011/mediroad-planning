"""Multi-seed stability aggregation for Stage 4.1.

The module separates stable policy value from unstable physical venue identity.
It is intentionally read-only and accepts completed :class:`ScenarioSolution`
objects so a caller can use certified solver runs without duplicating them.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import itertools
import json
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .types import ScenarioSolution


DEFAULT_SEEDS = (11, 23, 42, 77, 101)


@dataclass(frozen=True)
class SeedStabilityThresholds:
    max_objective_relative_spread: float = 0.01
    max_unique_coverage_relative_spread: float = 0.01
    max_need_weighted_relative_spread: float = 0.01
    max_high_need_relative_spread: float = 0.02
    max_min_sigungu_coverage_absolute_spread: float = 0.02
    min_median_venue_jaccard: float = 0.67
    min_median_admin_jaccard: float = 0.67
    min_median_coverage_cluster_jaccard: float = 0.67
    # Backward-compatible public name.  The threshold is applied to allocation
    # pairs (cluster->bundle and venue->bundle), not to the usually tautological
    # set of bundle labels when every run enforces the same representation floor.
    min_median_bundle_jaccard: float = 0.67

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class SeedStabilityResult:
    classification: str
    passed: bool
    per_seed: pd.DataFrame
    pairwise: pd.DataFrame
    metric_spread: pd.DataFrame
    selection_frequency: pd.DataFrame
    thresholds: dict[str, float]
    threshold_fingerprint: str
    notes: list[str]


def _objective_metric(scenario: str) -> str:
    if scenario == "efficiency":
        return "unique_elderly_population"
    if scenario == "balanced":
        return "need_weighted_population_scaled"
    if scenario in {"equity", "equity_hard"}:
        return "min_sigungu_coverage_ratio"
    raise ValueError(f"No pre-declared seed objective for scenario {scenario!r}")


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return float(len(left & right) / len(union)) if union else 1.0


def _bundle_frame(solution: ScenarioSolution) -> pd.DataFrame:
    source = solution.bundle_assignments
    if source is None or source.empty:
        if "bundle_id" not in solution.selected:
            raise ValueError(
                "Seed stability requires bundle assignments; none are present in the solution"
            )
        source = solution.selected
    required = {"venue_id", "bundle_id"}
    if not required.issubset(source.columns):
        raise KeyError(f"Bundle assignment lacks columns {sorted(required - set(source.columns))}")
    frame = source[["venue_id", "bundle_id"]].copy()
    if frame.isna().any().any():
        raise ValueError("Bundle assignment contains null venue or bundle identifiers")
    frame = frame.astype(str)
    if frame["venue_id"].duplicated().any():
        raise ValueError("Bundle assignment must contain exactly one bundle per selected venue")
    selected_ids = set(solution.selected["venue_id"].astype(str))
    if set(frame["venue_id"]) != selected_ids:
        raise ValueError("Bundle assignment venue set does not equal the selected venue set")
    return frame


def _validate_runs(
    solutions_by_seed: Mapping[int, ScenarioSolution], expected_seeds: Sequence[int]
) -> tuple[list[int], str, str, int, str]:
    expected = [int(seed) for seed in expected_seeds]
    if len(expected) != len(set(expected)):
        raise ValueError("expected_seeds contains duplicates")
    observed = {int(seed) for seed in solutions_by_seed}
    if observed != set(expected):
        raise ValueError(
            f"Seed set must match exactly; missing={sorted(set(expected)-observed)}, "
            f"unexpected={sorted(observed-set(expected))}"
        )
    scenarios = {solutions_by_seed[seed].scenario for seed in expected}
    catchments = {solutions_by_seed[seed].catchment_id for seed in expected}
    visits = {int(solutions_by_seed[seed].visit_count) for seed in expected}
    if len(scenarios) != 1 or len(catchments) != 1 or len(visits) != 1:
        raise ValueError("Seed runs must share scenario, catchment, and visit count")
    scenario = next(iter(scenarios))
    catchment = next(iter(catchments))
    visit_count = next(iter(visits))
    objective = _objective_metric(scenario)
    required_metrics = {
        objective,
        "unique_elderly_population",
        "need_weighted_population_scaled",
        "high_need_population_scaled",
        "min_sigungu_coverage_ratio",
    }
    usable = {"OPTIMAL", "FEASIBLE", "NEAR_OPTIMAL_CERTIFIED", "BEST_KNOWN_FEASIBLE"}
    for seed in expected:
        solution = solutions_by_seed[seed]
        if solution.status not in usable:
            raise ValueError(f"Seed {seed} has unusable status {solution.status}")
        if len(solution.selected) != visit_count:
            raise ValueError(f"Seed {seed} selected {len(solution.selected)} rather than {visit_count}")
        if solution.selected["venue_id"].astype(str).duplicated().any():
            raise ValueError(f"Seed {seed} selected duplicate venues")
        missing_columns = {"venue_id", "admin_code", "cluster_id"} - set(solution.selected.columns)
        if missing_columns:
            raise KeyError(f"Seed {seed} selection lacks columns {sorted(missing_columns)}")
        missing_metrics = required_metrics - set(solution.metrics)
        if missing_metrics:
            raise KeyError(f"Seed {seed} lacks metrics {sorted(missing_metrics)}")
        _bundle_frame(solution)
    return expected, scenario, catchment, visit_count, objective


def _sets(solution: ScenarioSolution) -> dict[str, set[str]]:
    bundles = _bundle_frame(solution)
    selected_clusters = solution.selected[["venue_id", "cluster_id"]].copy()
    if selected_clusters.isna().any().any():
        raise ValueError("Selected venue-to-cluster mapping contains null identifiers")
    selected_clusters = selected_clusters.astype("string")
    if selected_clusters.apply(lambda column: column.str.strip().eq("")).any().any():
        raise ValueError("Selected venue-to-cluster mapping contains blank identifiers")
    selected_clusters = selected_clusters.astype(str)
    allocation = bundles.merge(
        selected_clusters,
        on="venue_id",
        how="left",
        validate="one_to_one",
    )
    if allocation["cluster_id"].isna().any():
        raise ValueError("Bundle assignment could not be aligned to coverage clusters")
    return {
        "venue": set(solution.selected["venue_id"].astype(str)),
        "admin": set(solution.selected["admin_code"].astype(str)),
        "coverage_cluster": set(solution.selected["cluster_id"].astype(str)),
        # Retain the bundle-label set as a diagnostic only.  With a mandatory
        # min=1 floor it is commonly identical across every run and therefore
        # cannot establish allocation stability by itself.
        "bundle": set(bundles["bundle_id"].astype(str)),
        "cluster_bundle": set(
            allocation["cluster_id"].astype(str)
            + "::"
            + allocation["bundle_id"].astype(str)
        ),
        "venue_bundle": set(
            bundles["venue_id"].astype(str) + "::" + bundles["bundle_id"].astype(str)
        ),
    }


def _spread_row(metric: str, values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    if not np.isfinite(array).all():
        raise ValueError(f"Metric {metric} contains non-finite seed values")
    minimum = float(array.min())
    maximum = float(array.max())
    absolute = float(maximum - minimum)
    relative = float(absolute / max(abs(maximum), abs(minimum), 1e-12))
    return {
        "metric": metric,
        "minimum": minimum,
        "maximum": maximum,
        "mean": float(array.mean()),
        "standard_deviation": float(array.std(ddof=0)),
        "absolute_spread": absolute,
        "relative_spread": relative,
    }


def analyze_seed_stability(
    solutions_by_seed: Mapping[int, ScenarioSolution],
    *,
    expected_seeds: Sequence[int] = DEFAULT_SEEDS,
    thresholds: SeedStabilityThresholds | None = None,
) -> SeedStabilityResult:
    """Aggregate the exact five-seed contract and classify its stability."""

    thresholds = thresholds or SeedStabilityThresholds()
    seeds, scenario, catchment, visit_count, objective = _validate_runs(
        solutions_by_seed, expected_seeds
    )
    run_sets = {seed: _sets(solutions_by_seed[seed]) for seed in seeds}
    stability_kinds = [
        "venue",
        "admin",
        "coverage_cluster",
        "bundle",
        "cluster_bundle",
        "venue_bundle",
    ]
    metric_names = [
        "unique_elderly_population",
        "need_weighted_population_scaled",
        "high_need_population_scaled",
        "min_sigungu_coverage_ratio",
    ]

    per_seed_rows: list[dict[str, Any]] = []
    for seed in seeds:
        solution = solutions_by_seed[seed]
        per_seed_rows.append(
            {
                "seed": seed,
                "scenario": scenario,
                "catchment_id": catchment,
                "visit_count": visit_count,
                "solver_status": solution.status,
                "objective_metric": objective,
                "objective_value": float(solution.metrics[objective]),
                **{metric: float(solution.metrics[metric]) for metric in metric_names},
                "selected_venue_count": len(run_sets[seed]["venue"]),
                "selected_admin_count": len(run_sets[seed]["admin"]),
                "selected_coverage_cluster_count": len(run_sets[seed]["coverage_cluster"]),
                "selected_bundle_count": len(run_sets[seed]["bundle"]),
                "selected_cluster_bundle_count": len(run_sets[seed]["cluster_bundle"]),
                "selected_venue_bundle_count": len(run_sets[seed]["venue_bundle"]),
            }
        )
    per_seed = pd.DataFrame(per_seed_rows).sort_values("seed").reset_index(drop=True)

    pair_rows: list[dict[str, Any]] = []
    for left, right in itertools.combinations(seeds, 2):
        pair_rows.append(
            {
                "seed_a": left,
                "seed_b": right,
                **{
                    f"{kind}_jaccard": _jaccard(run_sets[left][kind], run_sets[right][kind])
                    for kind in stability_kinds
                },
            }
        )
    pairwise = pd.DataFrame(pair_rows).sort_values(["seed_a", "seed_b"]).reset_index(drop=True)

    spread_metrics = list(dict.fromkeys([objective, *metric_names]))
    spread = pd.DataFrame(
        [
            _spread_row(metric, [float(solutions_by_seed[seed].metrics[metric]) for seed in seeds])
            for metric in spread_metrics
        ]
    )
    spread_index = spread.set_index("metric")

    frequency_rows: list[dict[str, Any]] = []
    for kind in stability_kinds:
        universe = sorted(set().union(*(run_sets[seed][kind] for seed in seeds)))
        for identifier in universe:
            count = sum(identifier in run_sets[seed][kind] for seed in seeds)
            frequency_rows.append(
                {
                    "entity_type": kind,
                    "entity_id": identifier,
                    "seed_count": int(count),
                    "seed_frequency": float(count / len(seeds)),
                }
            )
    selection_frequency = pd.DataFrame(frequency_rows).sort_values(
        ["entity_type", "seed_frequency", "entity_id"], ascending=[True, False, True]
    ).reset_index(drop=True)

    median_jaccard = {
        kind: float(pairwise[f"{kind}_jaccard"].median())
        for kind in stability_kinds
    }
    objective_stable = bool(
        float(spread_index.loc[objective, "relative_spread"])
        <= thresholds.max_objective_relative_spread
        and float(spread_index.loc["unique_elderly_population", "relative_spread"])
        <= thresholds.max_unique_coverage_relative_spread
        and float(spread_index.loc["need_weighted_population_scaled", "relative_spread"])
        <= thresholds.max_need_weighted_relative_spread
        and float(spread_index.loc["high_need_population_scaled", "relative_spread"])
        <= thresholds.max_high_need_relative_spread
        and float(spread_index.loc["min_sigungu_coverage_ratio", "absolute_spread"])
        <= thresholds.max_min_sigungu_coverage_absolute_spread
    )
    structural_stable = bool(
        median_jaccard["admin"] >= thresholds.min_median_admin_jaccard
        and median_jaccard["coverage_cluster"]
        >= thresholds.min_median_coverage_cluster_jaccard
        and median_jaccard["cluster_bundle"]
        >= thresholds.min_median_bundle_jaccard
    )
    exact_stable = bool(
        median_jaccard["venue"] >= thresholds.min_median_venue_jaccard
        and median_jaccard["venue_bundle"]
        >= thresholds.min_median_bundle_jaccard
    )

    notes = [
        "Bundle-set Jaccard is diagnostic only because a mandatory bundle universe can make it vacuously one.",
        "The backward-compatible min_median_bundle_jaccard threshold gates cluster-bundle allocation for structural stability and venue-bundle allocation for exact stability.",
        "A stable objective with variable exact venues is not represented as a unique physical-site truth.",
    ]
    if not objective_stable:
        classification = "FAIL_SEED_OBJECTIVE_UNSTABLE"
        passed = False
    elif structural_stable and exact_stable:
        classification = "PASS_SEED_STABLE"
        passed = True
    elif structural_stable:
        classification = "PASS_CLUSTER_STABLE_VENUE_VARIABLE"
        passed = True
        notes.append("Use coverage cluster plus representative/fallback venues as the policy unit.")
    else:
        classification = "REVIEW_SEED_SELECTION_UNSTABLE"
        passed = False
        notes.append("Seed variability reaches admin/coverage-cluster structure; do not promote a final candidate universe.")

    # Put decision evidence directly in the spread table without hiding the raw
    # pairwise distribution.
    spread["threshold_fingerprint"] = thresholds.fingerprint
    return SeedStabilityResult(
        classification=classification,
        passed=passed,
        per_seed=per_seed,
        pairwise=pairwise,
        metric_spread=spread,
        selection_frequency=selection_frequency,
        thresholds={key: float(value) for key, value in asdict(thresholds).items()},
        threshold_fingerprint=thresholds.fingerprint,
        notes=notes,
    )
