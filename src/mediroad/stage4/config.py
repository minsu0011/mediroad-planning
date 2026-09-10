from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "version": "MEDIROAD_STAGE4_V1",
    "seed": 42,
    "runtime": {
        "jobs": 8,
        "solver": "cp-sat",
        "max_time_per_stage_sec": 20,
        "frontier_time_per_stage_sec": 15,
        "log_solver_progress": False,
        "max_memory_soft_gb": 24,
    },
    "candidate_reduction": {
        "top_n_per_admin": 3,
        "max_candidates": 600,
        "pareto_tier_max": 3,
        "require_all_admins": True,
        "one_per_coverage_cluster": True,
        "max_visits_per_admin": 2,
        "max_visits_per_sigungu": 4,
    },
    "network_validation": {
        "enabled": True,
        "required_for_official": True,
        "candidate_scope": "admin_top3",
        "threshold_minutes": [10, 15, 20],
        "max_snap_distance_m": 5000,
        "graph_bbox_buffer_deg": 0.15,
        "batch_size": 24,
        "pbf_relative_path": "14_v6_external_data/road_network/raw/chungcheong-non-military_20260817.osm.pbf",
        "pbf_sha256": "cfcf02addc9a0453f2e0fcffd3b27e027d131cbb3b2c63dd68e4313325724d2e",
        "cache_graph": True,
        "euclidean_reference_matrix_id": "hard_5000m",
    },
    "optimization": {
        "visit_count": 20,
        "population_integer_scale": 100,
        "need_integer_scale": 100,
        "objective_retention": {
            "total_population": 0.99,
            "need_weighted": 0.99,
            "high_need": 0.99,
            "min_sigungu_coverage": 1.0,
        },
        "balanced_min_efficiency_fraction": 0.80,
        "equity_min_efficiency_fraction": 0.60,
        "high_need_quantile": 0.80,
        "exclude_known_overlap": False,
        "known_overlap_penalty": 1000,
        "travel_penalty_scale": 1,
        "allow_unknown_travel": True,
        "cluster_constraint": True,
        "scenarios": ["efficiency", "balanced", "equity"],
        "catchment_scenarios": ["hard_5000m", "hard_3000m"],
        "include_best_network_catchment": True,
        "min_visits_per_sigungu_equity_hard": 0,
    },
    "bundle_assignment": {
        "enabled": True,
        "value_column_priority": [
            "bundle_gap_weighted_exposure",
            "mean_bundle_gap_weighted_exposure",
            "specialty_gap",
        ],
        "min_per_bundle": {},
        "max_per_bundle": {},
        "default_min": 1,
        "default_max": 20,
    },
    "robustness": {
        "selection_frequency_core": 0.67,
        "field_validation_queue_min": 30,
        "field_validation_queue_max": 60,
        "frontier_visits": [5, 10, 15, 20, 25, 30, 40],
        "frontier_scenarios": ["efficiency", "balanced"],
    },
    "paths": {
        "stage3_root": "outputs/model_v1/07_stage3",
        "stage4_root": "outputs/model_v1/08_stage4",
        "report_root": "reports/model_v1/stage4",
        "lock": "outputs/model_v1/.stage4_writer.lock",
    },
    "column_aliases": {
        "venue_id": ["venue_id", "candidate_id", "physical_venue_id"],
        "venue_name": ["venue_name", "facility_name", "candidate_name"],
        "admin_code": ["admin_dong_code", "emd_code", "admin_code"],
        "admin_name": ["admin_dong_name", "emd_name", "admin_name"],
        "sigungu": ["sigungu", "policy_sigungu_name", "sigungu_name"],
        "bundle_id": ["bundle_id", "service_bundle", "bundle"],
        "grid_id": ["grid_id", "gid", "grid_code"],
        "elderly_population": [
            "elderly65_calibrated_population",
            "elderly65_population",
            "elderly65_calibrated",
            "calibrated_elderly65",
            "elderly_population",
            "population65",
        ],
        "need_score": ["stage1_need_score", "structural_need", "need_score", "need_score_v1"],
        "cluster_id": ["coverage_cluster_id", "coverage_equivalent_cluster_id", "cluster_id"],
        "shortlist_rank": ["shortlist_rank", "admin_shortlist_rank", "rank_within_admin"],
        "pareto_tier": ["pareto_tier", "pareto_rank"],
        "travel_minutes": [
            "osm_nearest_mobile_team_base_drive_min_v6",
            "travel_time_proxy_min",
            "road_context_minutes",
            "team_base_drive_min",
        ],
        "known_overlap": ["known_outreach_overlap", "outreach_overlap_known", "overlap_status"],
    },
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: Path | None = None) -> dict[str, Any]:
    config = deepcopy(DEFAULT_CONFIG)
    if path is not None:
        if not path.exists():
            raise FileNotFoundError(f"Stage 4 config not found: {path}")
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise TypeError("Stage 4 YAML root must be a mapping")
        config = _deep_merge(config, loaded)
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    visits = int(config["optimization"]["visit_count"])
    if visits <= 0:
        raise ValueError("optimization.visit_count must be positive")
    if int(config["runtime"]["jobs"]) <= 0:
        raise ValueError("runtime.jobs must be positive")
    for key in ["balanced_min_efficiency_fraction", "equity_min_efficiency_fraction"]:
        val = float(config["optimization"][key])
        if not 0 < val <= 1:
            raise ValueError(f"optimization.{key} must be in (0, 1]")
    for value in config["optimization"]["objective_retention"].values():
        if not 0 < float(value) <= 1:
            raise ValueError("All objective retention values must be in (0, 1]")
    q = float(config["optimization"]["high_need_quantile"])
    if not 0 < q < 1:
        raise ValueError("optimization.high_need_quantile must be in (0, 1)")
