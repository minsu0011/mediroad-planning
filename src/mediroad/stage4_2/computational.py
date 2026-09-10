from __future__ import annotations

from dataclasses import asdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .bundles import assign_bundles
from .candidates import (
    build_candidate_set,
    compress_exact_equivalent_candidates,
    prune_safe_dominated_candidates,
)
from .compression import compress_grid_patterns, verify_pattern_reconstruction
from .data import (
    align_coverage,
    bundle_table_from_interface,
    load_coverage,
    normalize_grid_policy,
    normalize_stage3_interface,
    venue_table_from_interface,
)
from .discovery import discover_stage3, discover_stage41
from .contracts import FIELD_DERIVED_COLUMNS, FIELD_KEY_COLUMNS, FIELD_REQUIRED_INPUT_COLUMNS, FIELD_STATUS_COLUMNS
from .errors import ContractError, SolverCertificationError
from .metrics import compare_candidate_results, pairwise_plan_stability
from .optimizer import SpatialMILP, constrained_greedy, telemetry_frame
from .types import ScenarioResult
from .utils import atomic_write_csv, atomic_write_json, read_json, read_table, sha256_file


def _prepare_candidate_problem(
    root: Path,
    candidate_set: str,
    config: dict[str, Any],
    *,
    stage3_cache: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cache = stage3_cache if stage3_cache is not None else {}
    stage3 = cache.setdefault("stage3", discover_stage3(root))
    stage41 = cache.setdefault("stage41", discover_stage41(root))
    interface = cache.setdefault("interface", normalize_stage3_interface(read_table(stage3.interface)))
    venues = cache.setdefault("venues", venue_table_from_interface(interface))
    bundles = cache.setdefault("bundles", bundle_table_from_interface(interface))
    membership = cache.setdefault("membership", read_table(stage41.candidate_membership))
    coverage_all = cache.setdefault("coverage_all", load_coverage(stage3, config["matrix"]["primary_matrix_id"]))
    grid = cache.setdefault("grid", normalize_grid_policy(read_table(stage3.grid_policy)))
    grid = grid.set_index("grid_id").loc[coverage_all.grid_ids.astype(str)].reset_index()

    candidates = build_candidate_set(venues, membership, candidate_set)
    coverage = align_coverage(coverage_all, candidates["venue_id"].astype(str).to_numpy(), grid["grid_id"].astype(str).to_numpy())
    ledgers: list[pd.DataFrame] = []
    if bool(config["candidate_expansion"].get("compress_exact_equivalents", True)):
        comp = compress_exact_equivalent_candidates(candidates, coverage, config)
        candidates, coverage = comp.candidates, comp.coverage
        comp.ledger.insert(0, "reduction_stage", "EXACT_EQUIVALENCE")
        ledgers.append(comp.ledger)
    if bool(config["candidate_expansion"].get("safe_subset_dominance", True)):
        dom = prune_safe_dominated_candidates(
            candidates,
            coverage,
            config,
            max_group_size=int(config["candidate_expansion"].get("dominance_max_group_size", 80)),
        )
        candidates, coverage = dom.candidates, dom.coverage
        dom.ledger.insert(0, "reduction_stage", "SAFE_SUBSET_DOMINANCE")
        ledgers.append(dom.ledger)
    patterns = compress_grid_patterns(coverage, candidates["venue_id"].astype(str).to_numpy(), grid)
    ledger = pd.concat(ledgers, ignore_index=True) if ledgers else pd.DataFrame()
    return {
        "stage3": stage3,
        "stage41": stage41,
        "interface": interface,
        "venues": venues,
        "bundles": bundles,
        "membership": membership,
        "coverage_all": coverage_all,
        "grid": grid,
        "candidates": candidates,
        "coverage": coverage,
        "patterns": patterns,
        "reduction_ledger": ledger,
    }


def run_candidate_set(
    root: Path,
    candidate_set: str,
    config: dict[str, Any],
    outdir: Path,
    *,
    scenarios: list[str],
    time_limits_by_scenario: dict[str, float],
    cache: dict[str, Any] | None = None,
) -> dict[str, Any]:
    problem = _prepare_candidate_problem(root, candidate_set, config, stage3_cache=cache)
    candidates = problem["candidates"]
    coverage = problem["coverage"]
    grid = problem["grid"]
    patterns = problem["patterns"]
    bundle_values = problem["bundles"]
    outdir.mkdir(parents=True, exist_ok=True)
    if not problem["reduction_ledger"].empty:
        atomic_write_csv(outdir / "candidate_reduction_ledger.csv", problem["reduction_ledger"])

    greedy_idx, greedy_pop = constrained_greedy(
        candidates,
        coverage,
        grid["elderly_population"].to_numpy(float),
        config,
    )
    greedy_plan = candidates.iloc[greedy_idx].copy()
    greedy_plan.insert(0, "scenario", "CONSTRAINED_GREEDY_EFFICIENCY")
    atomic_write_csv(outdir / "constrained_greedy_plan.csv", greedy_plan)
    atomic_write_json(outdir / "constrained_greedy_metrics.json", {"unique_elderly_population": greedy_pop})

    model = SpatialMILP(candidates, patterns, config, candidate_set=candidate_set)
    gap = float(config["solver"]["near_optimal_relative_gap_max"])
    results: list[ScenarioResult] = []
    failed_scenarios: list[dict[str, Any]] = []
    failed_telemetry = []
    scenario_status: list[dict[str, Any]] = []

    def solve_or_record(
        scenario: str,
        *,
        efficiency_reference: float | None,
        greedy_floor: float | None,
        initial_incumbent_indices: np.ndarray | None = None,
    ) -> ScenarioResult | None:
        try:
            result = model.solve(
                scenario,
                efficiency_reference=efficiency_reference,
                greedy_floor=greedy_floor,
                initial_incumbent_indices=initial_incumbent_indices,
                time_limit_per_stage=float(time_limits_by_scenario[scenario]),
                gap_threshold=gap,
            )
        except SolverCertificationError as exc:
            # A time-limited/no-incumbent expanded universe is valid evidence
            # that the comparison is inconclusive; it must never be promoted or
            # fabricated as a feasible plan.  Structural/post-solve contract
            # errors do not have a failed telemetry row and remain fatal.
            telemetry = list(getattr(exc, "telemetry", []))
            if not telemetry or bool(telemetry[-1].success):
                raise
            failed_telemetry.extend(telemetry)
            failure = {
                "scenario": scenario,
                "candidate_set": candidate_set,
                "run_status": "NO_INCUMBENT_INCONCLUSIVE",
                "comparison_eligible": False,
                "promotion_allowed": False,
                "error": str(exc),
                "failed_objective": telemetry[-1].objective_name,
                "failed_stage_index": telemetry[-1].stage_index,
                "solver_status": telemetry[-1].solver_status,
            }
            failed_scenarios.append(failure)
            scenario_status.append(failure)
            atomic_write_json(outdir / f"scenario_failure__{scenario}.json", failure)
            return None
        scenario_status.append(
            {
                "scenario": scenario,
                "candidate_set": candidate_set,
                "run_status": "COMPLETED_WITH_INCUMBENT",
                "comparison_eligible": bool(result.certified),
                "promotion_allowed": False,
                "error": "",
                "failed_objective": "",
                "failed_stage_index": None,
                "solver_status": result.solver_status,
            }
        )
        return result

    efficiency_reference: float | None = None
    if "efficiency" in scenarios or any(s in scenarios for s in ["balanced", "equity"]):
        efficiency = solve_or_record(
            "efficiency",
            efficiency_reference=None,
            greedy_floor=greedy_pop,
        )
        if efficiency is not None:
            results.append(efficiency)
            efficiency_reference = float(efficiency.metrics["unique_elderly_population"])
    for scenario in scenarios:
        if scenario == "efficiency":
            continue
        if efficiency_reference is None:
            scenario_status.append(
                {
                    "scenario": scenario,
                    "candidate_set": candidate_set,
                    "run_status": "SKIPPED_NO_EFFICIENCY_INCUMBENT",
                    "comparison_eligible": False,
                    "promotion_allowed": False,
                    "error": "Efficiency scenario produced no incumbent",
                    "failed_objective": "",
                    "failed_stage_index": None,
                    "solver_status": "NOT_RUN_DEPENDENCY",
                }
            )
            continue
        result = solve_or_record(
            scenario,
            efficiency_reference=efficiency_reference,
            greedy_floor=None,
            initial_incumbent_indices=efficiency.selected_indices if efficiency is not None else None,
        )
        if result is not None:
            results.append(result)

    metrics_rows: list[dict[str, Any]] = []
    reconstruction_rows: list[dict[str, Any]] = []
    for result in results:
        plan = result.selected_frame.copy()
        bundle_min1, info1 = assign_bundles(
            plan,
            bundle_values,
            minimum_per_bundle=int(config.get("bundle_assignment", {}).get("primary_minimum_per_bundle", 1)),
            time_limit_sec=float(config.get("bundle_assignment", {}).get("time_limit_sec", 60)),
        )
        atomic_write_csv(outdir / f"plan__{result.scenario}.csv", bundle_min1)
        if result.scenario == "balanced":
            bundle_min0, info0 = assign_bundles(
                plan,
                bundle_values,
                minimum_per_bundle=0,
                time_limit_sec=float(config.get("bundle_assignment", {}).get("time_limit_sec", 60)),
            )
            atomic_write_csv(outdir / "bundle_assignment_min0.csv", bundle_min0)
            atomic_write_json(outdir / "bundle_minimum_sensitivity.json", {"min0": info0, "min1": info1})
        row = dict(result.metrics)
        row["greedy_floor"] = greedy_pop
        row["greedy_no_harm_margin"] = float(row["unique_elderly_population"] - greedy_pop) if result.scenario == "efficiency" else np.nan
        metrics_rows.append(row)
        check = verify_pattern_reconstruction(patterns, coverage, result.selected_indices, grid)
        check.update({"scenario": result.scenario, "candidate_set": candidate_set})
        reconstruction_rows.append(check)

    metrics_frame = pd.DataFrame(metrics_rows)
    if metrics_frame.empty:
        metrics_frame = pd.DataFrame(columns=["scenario", "candidate_set", "certification_class", "certified", "max_relative_gap"])
    telemetry = telemetry_frame(results)
    if failed_telemetry:
        telemetry = pd.concat([telemetry, pd.DataFrame([asdict(row) for row in failed_telemetry])], ignore_index=True)
    reconstruction_frame = pd.DataFrame(reconstruction_rows)
    if reconstruction_frame.empty:
        reconstruction_frame = pd.DataFrame(columns=["scenario", "candidate_set"])
    atomic_write_csv(outdir / "scenario_metrics.csv", metrics_frame)
    atomic_write_csv(outdir / "solver_stage_telemetry.csv", telemetry)
    atomic_write_csv(outdir / "scenario_run_status.csv", pd.DataFrame(scenario_status))
    atomic_write_csv(outdir / "pattern_reconstruction_check.csv", reconstruction_frame)
    atomic_write_csv(outdir / "pairwise_plan_stability.csv", pairwise_plan_stability(results))
    atomic_write_json(
        outdir / "candidate_problem_summary.json",
        {
            "candidate_set": candidate_set,
            "candidate_rows_after_safe_reduction": int(len(candidates)),
            "coverage_nnz": int(coverage.nnz),
            "source_grid_rows": int(len(grid)),
            "compressed_pattern_rows": int(patterns.n_patterns),
            "grid_compression_ratio": float(patterns.n_patterns / max(1, len(grid))),
            "greedy_unique_elderly_population": greedy_pop,
            "run_status": "COMPLETE" if not failed_scenarios else "COMPLETE_WITH_INCONCLUSIVE_NO_INCUMBENT",
            "failed_scenarios": failed_scenarios,
            "results": [
                {
                    "scenario": r.scenario,
                    "certified": r.certified,
                    "certification_class": r.certification_class,
                    "max_relative_gap": r.max_relative_gap,
                }
                for r in results
            ],
        },
    )
    return {
        "results": results,
        "greedy_pop": greedy_pop,
        "failed_scenarios": failed_scenarios,
        "run_status": "COMPLETE" if not failed_scenarios else "COMPLETE_WITH_INCONCLUSIVE_NO_INCUMBENT",
    }


def _candidate_set_job(
    root: Path,
    candidate_set: str,
    config: dict[str, Any],
    outdir: Path,
    scenarios: list[str],
    time_limits_by_scenario: dict[str, float],
) -> tuple[str, dict[str, Any]]:
    """Process-pool entry point; each worker owns one native HiGHS scheduler."""

    return candidate_set, run_candidate_set(
        root,
        candidate_set,
        config,
        outdir,
        scenarios=scenarios,
        time_limits_by_scenario=time_limits_by_scenario,
        cache=None,
    )



def _build_computational_field_universe(
    run_root: Path,
    config: dict[str, Any],
    stage41: Any,
    venue_catalog: pd.DataFrame,
    all_runs: dict[str, dict[str, Any]],
    decisions: list[dict[str, Any]],
    reference_set: str,
) -> dict[str, Any]:
    """Create an updated field-validation universe from certified/promoted plans.

    Stage 4.1's 47-row queue is preserved. Certified reference plans and any certified
    promoted expanded-set plans can add anchors/fallbacks. No field fact is invented.
    """
    existing_queue = read_table(stage41.field_queue).copy()
    existing_form = read_table(stage41.field_form).copy()
    existing_queue["venue_id"] = existing_queue["venue_id"].astype(str)
    existing_form["venue_id"] = existing_form["venue_id"].astype(str)
    catalog = venue_catalog.drop_duplicates("venue_id").copy()
    catalog["venue_id"] = catalog["venue_id"].astype(str)
    catalog_by_id = catalog.set_index("venue_id", drop=False)

    roles: dict[str, set[str]] = {}
    certified_eff: set[str] = set()
    certified_bal: set[str] = set()

    def add_result(candidate_set: str, result: ScenarioResult, prefix: str) -> None:
        if not result.certified:
            return
        role = f"{prefix}_{result.scenario.upper()}"
        for venue_id in result.selected_venue_ids:
            roles.setdefault(str(venue_id), set()).add(role)
        if result.scenario == "efficiency":
            certified_eff.update(map(str, result.selected_venue_ids))
        if result.scenario == "balanced":
            certified_bal.update(map(str, result.selected_venue_ids))

    for result in all_runs[reference_set]["results"]:
        add_result(reference_set, result, "REFERENCE_CERTIFIED")
    promoted_sets = {str(row["candidate_set"]) for row in decisions if bool(row["promotion_allowed"])}
    for candidate_set in promoted_sets:
        for result in all_runs[candidate_set]["results"]:
            add_result(candidate_set, result, "PROMOTED_EXPANDED_CERTIFIED")

    mandatory_ids = set(roles)
    fallback_roles: dict[str, set[str]] = {}
    for venue_id in sorted(mandatory_ids):
        if venue_id not in catalog_by_id.index:
            continue
        row = catalog_by_id.loc[venue_id]
        for col in ["fallback_venue_1", "fallback_venue_2"]:
            value = row.get(col)
            if pd.notna(value) and str(value).strip() and str(value) in catalog_by_id.index:
                fallback_roles.setdefault(str(value), set()).add(f"FALLBACK_FOR_{venue_id}")
    roles.update({k: roles.get(k, set()) | v for k, v in fallback_roles.items()})

    universe_ids = list(dict.fromkeys([*existing_queue["venue_id"].tolist(), *sorted(roles)]))
    max_rows = int(config.get("field_validation", {}).get("queue_max", 60))
    mandatory_plus_fallback = set(roles)
    if len(universe_ids) > max_rows:
        # Never drop newly selected anchors/fallbacks. Trim only old non-mandatory tail rows.
        old_optional = [v for v in existing_queue["venue_id"] if v not in mandatory_plus_fallback]
        keep_optional = max(0, max_rows - len(mandatory_plus_fallback))
        universe_ids = list(dict.fromkeys([*sorted(mandatory_plus_fallback), *old_optional[:keep_optional]]))

    base = catalog.loc[catalog["venue_id"].isin(universe_ids)].copy()
    queue_meta = existing_queue.drop_duplicates("venue_id")
    queue = base.merge(queue_meta, on="venue_id", how="left", suffixes=("", "_stage41"))
    for col in ["venue_name", "venue_type", "admin_code", "sigungu", "cluster_id"]:
        old = f"{col}_stage41"
        if old in queue:
            queue[col] = queue[old].combine_first(queue[col])
            queue = queue.drop(columns=[old])
    queue["stage4_2_selection_roles"] = queue["venue_id"].map(lambda v: "|".join(sorted(roles.get(str(v), set()))))
    queue["certified_efficiency_balanced_intersection"] = queue["venue_id"].isin(certified_eff & certified_bal)
    queue["stage4_2_selected_anchor"] = queue["venue_id"].isin(mandatory_ids)
    queue["stage4_2_fallback_candidate"] = queue["venue_id"].isin(set(fallback_roles))
    fallback_counts = {}
    for venue_id in mandatory_ids:
        if venue_id in catalog_by_id.index:
            row = catalog_by_id.loc[venue_id]
            vals = [str(row.get(c)) for c in ["fallback_venue_1", "fallback_venue_2"] if pd.notna(row.get(c)) and str(row.get(c)).strip()]
            fallback_counts[venue_id] = sum(not value.startswith("BUSPROXY_") for value in vals)
    is_bus = queue["venue_id"].str.startswith("BUSPROXY_")
    queue["stage4_2_priority_tier"] = "D"
    queue.loc[queue["stage4_2_selected_anchor"], "stage4_2_priority_tier"] = "B"
    queue.loc[queue["certified_efficiency_balanced_intersection"], "stage4_2_priority_tier"] = "A"
    p0 = is_bus & queue["venue_id"].map(lambda v: fallback_counts.get(str(v), 0)).eq(0) & queue["stage4_2_selected_anchor"]
    queue.loc[p0, "stage4_2_priority_tier"] = "P0"
    tier_order = {"P0": 0, "A": 1, "B": 2, "C": 3, "D": 4}
    queue["_tier_order"] = queue["stage4_2_priority_tier"].map(tier_order).fillna(9)
    queue = queue.sort_values(["_tier_order", "venue_id"]).drop(columns="_tier_order").reset_index(drop=True)
    queue.insert(0, "stage4_2_field_rank", np.arange(1, len(queue) + 1))

    old_form = existing_form.set_index("venue_id", drop=False)
    form_rows: list[dict[str, Any]] = []
    for _, row in queue.iterrows():
        venue_id = str(row["venue_id"])
        if venue_id in old_form.index:
            payload = old_form.loc[venue_id].to_dict()
        else:
            payload = {key: row.get(key, "UNKNOWN") for key in FIELD_KEY_COLUMNS}
            payload.update({col: "UNKNOWN" for col in FIELD_REQUIRED_INPUT_COLUMNS})
            payload.update({col: False for col in FIELD_DERIVED_COLUMNS})
        payload.update(
            {
                "stage4_2_field_rank": int(row["stage4_2_field_rank"]),
                "stage4_2_priority_tier": row["stage4_2_priority_tier"],
                "stage4_2_selection_roles": row["stage4_2_selection_roles"],
            }
        )
        form_rows.append(payload)
    form = pd.DataFrame(form_rows)
    evidence = pd.DataFrame({"venue_id": form["venue_id"].astype(str)})
    for col in ["verification_source", "source_reference", "contact_person_or_office", "contact_channel", "verifier", "verification_notes", "evidence_file_or_url"]:
        evidence[col] = ""

    outdir = run_root / "field_validation_update"
    outdir.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(outdir / "field_validation_priority_queue.csv", queue)
    atomic_write_csv(outdir / "field_validation_form.csv", form)
    atomic_write_csv(outdir / "field_validation_evidence.csv", evidence)
    summary = {
        "rows": int(len(queue)),
        "new_rows_vs_stage4_1": int(len(set(queue["venue_id"]) - set(existing_queue["venue_id"]))),
        "selected_anchor_rows": int(queue["stage4_2_selected_anchor"].sum()),
        "certified_intersection_rows": int(queue["certified_efficiency_balanced_intersection"].sum()),
        "p0_busproxy_without_nonbus_fallback": int((queue["stage4_2_priority_tier"] == "P0").sum()),
        "queue_max": max_rows,
    }
    atomic_write_json(outdir / "field_validation_update_summary.json", summary)
    return summary

def run_computational_hardening(root: Path, config: dict[str, Any], run_root: Path) -> dict[str, Any]:
    candidate_sets = list(config["candidate_expansion"].get("candidate_sets", ["top3", "top5", "coarse_pareto"]))
    reference_set = str(config["candidate_expansion"].get("reference_candidate_set", "top3"))
    all_runs: dict[str, dict[str, Any]] = {}
    candidate_budget = float(config["solver"].get("candidate_time_per_stage_sec", 300))
    equity_budget = float(config.get("equity_long_solve", {}).get("time_limit_per_stage_sec", 600))
    jobs: list[tuple[str, list[str], dict[str, float]]] = []
    for candidate_set in candidate_sets:
        scenarios = ["efficiency", "balanced"]
        if candidate_set == reference_set and bool(config.get("equity_long_solve", {}).get("enabled", True)):
            scenarios.append("equity")
        time_limits = {scenario: (equity_budget if scenario == "equity" else candidate_budget) for scenario in scenarios}
        jobs.append((candidate_set, scenarios, time_limits))

    workers = min(int(config.get("runtime", {}).get("candidate_workers", 1)), len(jobs))
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _candidate_set_job,
                    root,
                    candidate_set,
                    config,
                    run_root / "candidate_sets" / candidate_set,
                    scenarios,
                    time_limits,
                ): candidate_set
                for candidate_set, scenarios, time_limits in jobs
            }
            for future in as_completed(futures):
                candidate_set, payload = future.result()
                all_runs[candidate_set] = payload
    else:
        cache: dict[str, Any] = {}
        for candidate_set, scenarios, time_limits in jobs:
            all_runs[candidate_set] = run_candidate_set(
                root,
                candidate_set,
                config,
                run_root / "candidate_sets" / candidate_set,
                scenarios=scenarios,
                time_limits_by_scenario=time_limits,
                cache=cache,
            )
    all_runs = {candidate_set: all_runs[candidate_set] for candidate_set in candidate_sets}

    reference_balanced = next(
        (r for r in all_runs[reference_set]["results"] if r.scenario == "balanced"),
        None,
    )
    comparisons: list[dict[str, Any]] = []
    for candidate_set, payload in all_runs.items():
        if candidate_set == reference_set:
            continue
        alternative = next((r for r in payload["results"] if r.scenario == "balanced"), None)
        if reference_balanced is not None and alternative is not None:
            row = compare_candidate_results(reference_balanced, alternative)
            row["comparison_evidence_status"] = (
                "CERTIFIED_COMPARISON"
                if reference_balanced.certified and alternative.certified
                else "DIAGNOSTIC_UNCERTIFIED_INCUMBENT"
            )
        else:
            row = {
                "reference_candidate_set": reference_set,
                "alternative_candidate_set": candidate_set,
                "scenario": "balanced",
                "reference_certified": bool(reference_balanced and reference_balanced.certified),
                "alternative_certified": bool(alternative and alternative.certified),
                "venue_jaccard": np.nan,
                "admin_jaccard": np.nan,
                "coverage_cluster_jaccard": np.nan,
                "comparison_evidence_status": "NO_INCUMBENT_INCONCLUSIVE",
            }
            for metric in [
                "unique_elderly_population",
                "need_weighted_population",
                "high_need_population",
                "min_sigungu_coverage_ratio",
            ]:
                row[f"reference_{metric}"] = (
                    float(reference_balanced.metrics.get(metric, np.nan)) if reference_balanced is not None else np.nan
                )
                row[f"alternative_{metric}"] = (
                    float(alternative.metrics.get(metric, np.nan)) if alternative is not None else np.nan
                )
                row[f"delta_{metric}"] = np.nan
                row[f"relative_delta_{metric}"] = np.nan
        comparisons.append(row)
    comparison_frame = pd.DataFrame(comparisons)
    atomic_write_csv(run_root / "candidate_expansion_comparison.csv", comparison_frame)

    thresholds = config["candidate_expansion"]
    decisions = []
    for row in comparisons:
        conclusive = bool(row["reference_certified"] and row["alternative_certified"])
        promotion = False
        reason = "INCONCLUSIVE_NO_PROMOTION"
        if conclusive:
            loss_pop = max(0.0, -float(row["relative_delta_unique_elderly_population"]))
            loss_need = max(0.0, -float(row["relative_delta_need_weighted_population"]))
            loss_high = max(0.0, -float(row["relative_delta_high_need_population"]))
            loss_min = max(0.0, -float(row["delta_min_sigungu_coverage_ratio"]))
            stability = float(row["coverage_cluster_jaccard"])
            promotion = bool(
                loss_pop <= float(thresholds.get("max_unique_coverage_loss_fraction", 0.01))
                and loss_need <= float(thresholds.get("max_need_weighted_loss_fraction", 0.01))
                and loss_high <= float(thresholds.get("max_high_need_loss_fraction", 0.02))
                and loss_min <= float(thresholds.get("max_min_sigungu_coverage_loss", 0.02))
                and stability >= float(thresholds.get("min_coverage_cluster_jaccard", 0.50))
            )
            reason = "PROMOTE_EXPANDED_SET" if promotion else "CERTIFIED_BUT_NO_CLEAR_PROMOTION"
        decisions.append(
            {
                "candidate_set": row["alternative_candidate_set"],
                "comparison_conclusive": conclusive,
                "promotion_allowed": promotion,
                "decision": reason,
                "comparison_evidence_status": row.get("comparison_evidence_status", "UNKNOWN"),
            }
        )
    atomic_write_csv(run_root / "candidate_expansion_decision.csv", pd.DataFrame(decisions))

    reference_results = all_runs[reference_set]["results"]
    equity = next((r for r in reference_results if r.scenario == "equity"), None)
    decision = {
        "reference_candidate_set": reference_set,
        "candidate_expansion_comparison_conclusive": bool(decisions and all(d["comparison_conclusive"] for d in decisions)),
        "candidate_expansion_any_promotion": bool(any(d["promotion_allowed"] for d in decisions)),
        "equity_long_solve_certified": bool(equity and equity.certified),
        "equity_certification_class": equity.certification_class if equity else "NOT_RUN",
        "equity_max_relative_gap": equity.max_relative_gap if equity else None,
        "candidate_set_run_status": {name: payload["run_status"] for name, payload in all_runs.items()},
        "candidate_set_failed_scenarios": {name: payload["failed_scenarios"] for name, payload in all_runs.items()},
        "stage5_started": False,
    }
    field_update = _build_computational_field_universe(
        run_root,
        config,
        discover_stage41(root),
        venue_table_from_interface(normalize_stage3_interface(read_table(discover_stage3(root).interface))),
        all_runs,
        decisions,
        reference_set,
    )
    decision["field_validation_update"] = field_update
    atomic_write_json(run_root / "computational_hardening_decision.json", decision)
    return {"runs": all_runs, "comparisons": comparison_frame, "decision": decision}
