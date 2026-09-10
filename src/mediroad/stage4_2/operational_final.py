from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .bundles import assign_bundles
from .compression import compress_grid_patterns
from .data import align_coverage, bundle_table_from_interface, load_coverage, normalize_grid_policy, normalize_stage3_interface, venue_table_from_interface
from .discovery import discover_stage3, discover_stage41
from .errors import ContractError, InputNotReadyError
from .field_validation import audit_field_validation
from .operational_inputs import audit_operational_inputs
from .optimizer import SpatialMILP, constrained_greedy, telemetry_frame
from .utils import atomic_write_csv, atomic_write_json, read_table
from .venue_resolution import resolve_physical_venues


def run_operational_final(
    root: Path,
    config: dict[str, Any],
    run_root: Path,
    *,
    field_validation_path: Path,
    operational_paths: dict[str, Path],
) -> dict[str, Any]:
    stage3 = discover_stage3(root)
    stage41 = discover_stage41(root)
    interface = normalize_stage3_interface(read_table(stage3.interface))
    venue_catalog = venue_table_from_interface(interface)
    bundle_values = bundle_table_from_interface(interface)
    anchor_interface = read_table(stage41.field_interface)
    field_raw = read_table(field_validation_path)
    field_audit = audit_field_validation(field_raw)
    atomic_write_csv(run_root / "field_validation_normalized.csv", field_audit.normalized)
    atomic_write_csv(run_root / "field_validation_issues.csv", field_audit.issues)
    atomic_write_json(run_root / "field_validation_summary.json", field_audit.summary)

    resolution = resolve_physical_venues(anchor_interface, field_audit.normalized, venue_catalog)
    atomic_write_csv(run_root / "physical_venue_resolution.csv", resolution.resolution)
    atomic_write_csv(run_root / "unresolved_anchors.csv", resolution.unresolved)
    atomic_write_json(run_root / "physical_venue_resolution_summary.json", resolution.summary)

    verified_ids = set(field_audit.normalized.loc[field_audit.normalized["field_verified"], "venue_id"].astype(str))
    op_audit = audit_operational_inputs(operational_paths, verified_ids)
    atomic_write_json(
        run_root / "operational_input_audit.json",
        {"ready": op_audit.ready, "summaries": op_audit.summaries, "blockers": op_audit.blockers},
    )
    # Stage 4.1 anchors are provisional. Failed anchors do not block the new optimization
    # when a sufficiently large verified non-BUS candidate pool exists; the model can replace them.
    if not op_audit.ready:
        raise InputNotReadyError(f"Operational inputs not ready: {op_audit.blockers}")

    # Use every verified non-BUS physical facility, not just the previously selected anchors,
    # so failed anchors can be replaced without silently reverting to an unverified proxy.
    verified_non_bus = sorted(v for v in verified_ids if not v.startswith("BUSPROXY_") and v in set(venue_catalog["venue_id"]))
    if len(verified_non_bus) < int(config.get("optimization", {}).get("visit_count", 20)):
        raise InputNotReadyError(f"Only {len(verified_non_bus)} verified non-BUS venues; at least 20 are required")
    candidates = venue_catalog.loc[venue_catalog["venue_id"].isin(verified_non_bus)].copy().reset_index(drop=True)
    travel = op_audit.normalized_tables["travel"]
    travel = travel.loc[travel["confirmed"] & travel["venue_id"].isin(verified_non_bus)].copy()
    min_travel = travel.groupby("venue_id", as_index=False)["travel_minutes"].min()
    candidates = candidates.merge(min_travel, on="venue_id", how="left", suffixes=("", "_confirmed"))
    if "travel_minutes_confirmed" in candidates:
        candidates["travel_minutes"] = candidates["travel_minutes_confirmed"]
    if candidates["travel_minutes"].isna().any():
        missing = candidates.loc[candidates["travel_minutes"].isna(), "venue_id"].tolist()
        raise InputNotReadyError(f"Confirmed travel missing for verified venues: {missing[:10]}")

    coverage_all = load_coverage(stage3, config["matrix"]["primary_matrix_id"])
    grid = normalize_grid_policy(read_table(stage3.grid_policy)).set_index("grid_id").loc[coverage_all.grid_ids.astype(str)].reset_index()
    coverage = align_coverage(coverage_all, candidates["venue_id"].to_numpy(str), grid["grid_id"].to_numpy(str))
    patterns = compress_grid_patterns(coverage, candidates["venue_id"].to_numpy(str), grid)
    model = SpatialMILP(candidates, patterns, config, candidate_set="verified")
    greedy_idx, greedy_pop = constrained_greedy(candidates, coverage, grid["elderly_population"].to_numpy(float), config)
    gap = float(config["solver"]["near_optimal_relative_gap_max"])
    time_limit = float(config["solver"].get("operational_final_time_per_stage_sec", 600))
    efficiency = model.solve(
        "efficiency",
        efficiency_reference=None,
        greedy_floor=greedy_pop,
        time_limit_per_stage=time_limit,
        gap_threshold=gap,
    )
    balanced = model.solve(
        "balanced",
        efficiency_reference=float(efficiency.metrics["unique_elderly_population"]),
        greedy_floor=None,
        time_limit_per_stage=time_limit,
        gap_threshold=gap,
    )
    equity = model.solve(
        "equity",
        efficiency_reference=float(efficiency.metrics["unique_elderly_population"]),
        greedy_floor=None,
        time_limit_per_stage=time_limit,
        gap_threshold=gap,
    )
    results = [efficiency, balanced, equity]
    all_certified = all(r.certified for r in results)
    allowed_bundle_ids = set(op_audit.summaries.get("available_bundle_ids", []))
    if not allowed_bundle_ids:
        raise InputNotReadyError("No confirmed service bundle is deliverable by an available team")
    for result in results:
        assigned, info = assign_bundles(
            result.selected_frame,
            bundle_values,
            minimum_per_bundle=int(config.get("bundle_assignment", {}).get("primary_minimum_per_bundle", 1)),
            allowed_bundle_ids=allowed_bundle_ids,
            time_limit_sec=float(config.get("bundle_assignment", {}).get("time_limit_sec", 60)),
        )
        assigned["exact_month"] = np.nan
        assigned["exact_date"] = np.nan
        assigned["stage5_started"] = False
        # No individual plan can authorize Stage 5.  Release eligibility is a
        # run-level property and requires every frozen policy scenario.
        assigned["stage5_release_allowed"] = bool(all_certified and result.scenario == "balanced")
        atomic_write_csv(run_root / f"operational_plan__{result.scenario}.csv", assigned)
        atomic_write_json(run_root / f"bundle_assignment__{result.scenario}.json", info)
    atomic_write_csv(run_root / "solver_stage_telemetry.csv", telemetry_frame(results))
    metrics = pd.DataFrame([r.metrics for r in results])
    atomic_write_csv(run_root / "operational_scenario_metrics.csv", metrics)
    decision = {
        "decision": "PASS_STAGE4_OPERATIONAL_FINAL" if all_certified else "PASS_STAGE4_2_READY_FOR_OPERATIONAL_FINAL",
        "all_twenty_physical_venues_field_verified": True,
        "stage4_1_anchor_resolution": resolution.summary,
        "stage4_1_unresolved_anchors_replaced_by_reoptimization": int(resolution.summary.get("unresolved", 0)),
        "confirmed_team_base_travel_available": True,
        "vehicle_calendar_and_availability_confirmed": True,
        "all_policy_scenarios_certified": all_certified,
        "stage5_started": False,
        "stage5_release_allowed": bool(all_certified),
    }
    atomic_write_json(run_root / "operational_final_decision.json", decision)
    return {"decision": decision, "results": results, "resolution": resolution, "operational_audit": op_audit}
