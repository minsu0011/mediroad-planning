"""Run-scoped MEDIROAD MODEL V1 Stage 3 spatial exposure pipeline.

Stage 3 prepares potential-beneficiary geographic coverage and venue context.
It does not select visits, assign dates, infer patient demand, or start Stage 4.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

from pyproj import Transformer
from pyproj import datadir as _pyproj_datadir

# GeoPandas/GDAL inside the isolated WSL environment may otherwise discover
# the system PROJ directory instead of pyproj's bundled, version-matched
# database.  Set both accepted variable names before GeoPandas is imported.
os.environ.setdefault("PROJ_DATA", _pyproj_datadir.get_data_dir())
os.environ.setdefault("PROJ_LIB", _pyproj_datadir.get_data_dir())

import geopandas as gpd
import numpy as np
import pandas as pd
import yaml
from scipy import sparse

from mediroad.reporting.stage3 import (
    STAGE3_CORE_FIGURE_IDS,
    build_stage3_quality_gate,
    quality_gate_record,
    render_stage3_core_figures,
)
from mediroad.stage3.data import (
    build_grid_policy_weights,
    build_transport_and_road_context,
    canonical_string_key,
    filter_venues_to_boundary,
    reconcile_grid_population,
    sha256_file,
    validate_grid_source,
)
from mediroad.stage3.decision import (
    build_admin_shortlists,
    build_coverage_cluster_fallbacks,
    build_field_validation_queue,
    build_need_exposure_quadrants,
    build_stage3_to_stage4_interface,
    build_venue_feasibility_context,
    compute_pareto_tiers,
    compute_sensitivity_metrics,
    stage3_spearman_audit,
    validate_stage3_to_stage4_interface,
)
from mediroad.stage3.spatial import (
    CatchmentConfig,
    SparseCatchment,
    assert_hard_radius_monotonicity,
    build_catchment_family,
    canonicalize_grid,
    canonicalize_venues,
    compute_pair_overlaps,
    compute_venue_exposures,
    connected_overlap_clusters,
    matrix_manifest_record,
    weighted_union_coverage,
)


_RUN_ID = re.compile(r"^[A-Za-z0-9_.-]+$")
ADMIN = "admin_dong_code"
VENUE = "venue_id"
BUNDLE = "bundle_id"

TEMPORAL_SELECTOR_FORBIDDEN_COLUMNS = frozenset(
    {
        "recommended_season", "fallback_season", "exact_month", "exact_date",
        "temporal_fit_score", "temporal_score", "season_score",
        "season_adjusted_exposure", "month_score", "date_score",
        "temporal_confidence", "temporal_release_type",
        "spring_venue_score", "summer_venue_score", "autumn_venue_score",
        "winter_venue_score",
    }
)
TEMPORAL_SELECTOR_FORBIDDEN_PATTERN = re.compile(
    r"(?:^|_)(?:temporal|season|spring|summer|autumn|winter|month|date)(?:_|$)",
    flags=re.IGNORECASE,
)


def _is_temporal_selector_feature(value: object) -> bool:
    feature = str(value).strip().lower()
    return feature in TEMPORAL_SELECTOR_FORBIDDEN_COLUMNS or bool(
        TEMPORAL_SELECTOR_FORBIDDEN_PATTERN.search(feature)
    )


@dataclass(frozen=True)
class Stage3PipelineResult:
    run_id: str
    run_dir: Path
    report_dir: Path
    metadata_path: Path
    hard_pass: bool
    stage3_complete: bool


@dataclass(frozen=True)
class Stage3TestResult:
    passed: int
    failed: int
    returncode: int | None
    command: tuple[str, ...]
    duration_seconds: float
    test_file_count: int
    test_files_sha256: str
    stdout: str
    stderr: str
    source_contract_sha256: str = "NOT_RECORDED"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _peak_rss_gib() -> float:
    """Best-effort process peak RSS with platform-correct units."""

    try:
        import resource

        value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        divisor = 1024.0**3 if sys.platform == "darwin" else 1024.0**2
        return value / divisor
    except (ImportError, OSError, ValueError):
        try:
            import psutil

            process = psutil.Process()
            peak = getattr(process.memory_info(), "peak_wset", process.memory_info().rss)
            return float(peak) / 1024.0**3
        except (ImportError, OSError, ValueError):
            return float("nan")


def _dependency_versions() -> dict[str, str]:
    packages = (
        "numpy",
        "pandas",
        "scipy",
        "geopandas",
        "shapely",
        "pyproj",
        "matplotlib",
        "plotly",
        "pyarrow",
        "PyYAML",
        "Pillow",
        "pytest",
        "pyogrio",
        "Fiona",
        "GDAL",
    )
    result: dict[str, str] = {}
    for package in packages:
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = "MISSING"
    return result


def _test_source_contract_files(root: Path) -> tuple[Path, ...]:
    relative_paths = (
        "configs/model_v1/stage3_spatial.yaml",
        "requirements-model-v1.txt",
        "run_model_v1_stage3.py",
        "run_model_v1_stage3.ps1",
        "src/mediroad/__init__.py",
        "src/mediroad/stage3_pipeline.py",
        "src/mediroad/stage3/__init__.py",
        "src/mediroad/stage3/data.py",
        "src/mediroad/stage3/spatial.py",
        "src/mediroad/stage3/decision.py",
        "src/mediroad/reporting/__init__.py",
        "src/mediroad/reporting/stage3.py",
        "src/mediroad/reporting/stage3_maps.py",
        "src/mediroad/reporting/correlation.py",
        "src/mediroad/reporting/stage2b_hardening.py",
    )
    paths = tuple((root / relative).resolve() for relative in relative_paths)
    if any(not path.is_file() for path in paths):
        missing = [str(path) for path in paths if not path.is_file()]
        raise FileNotFoundError(f"Stage 3 test-source contract files missing: {missing}")
    return paths


def _file_contract_sha256(root: Path, paths: Sequence[Path]) -> str:
    digest = sha256(b"MEDIROAD_STAGE3_TEST_SOURCE_CONTRACT_V1\0")
    for path in sorted(paths, key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, default=_json_default) + "\n",
    )


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8", lineterminator="\n")
    temporary.replace(path)


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _atomic_sparse(path: Path, matrix: sparse.csr_matrix) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
    sparse.save_npz(temporary, matrix, compressed=True)
    temporary.replace(path)


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return value


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _require_below(root: Path, path: Path, label: str) -> Path:
    root, path = root.resolve(), path.resolve()
    if root not in path.parents:
        raise ValueError(f"{label} must be below the package root: {path}")
    return path


def _read_frame(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv" or path.name.lower().endswith(".csv.gz"):
        return pd.read_csv(path, low_memory=False)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported table format: {path}")


def _id_set_sha(values: Iterable[object]) -> str:
    payload = "".join(f"{value}\n" for value in sorted(map(str, values))).encode("utf-8")
    return sha256(payload).hexdigest()


def _config_contract(config: Mapping[str, Any]) -> None:
    if config.get("status") != "post_diagnostic_hardening_frozen_before_final_rerun":
        raise ValueError("Stage 3 post-diagnostic contract status changed")
    if config.get("target_definition") != (
        "potential_elderly_beneficiary_geographic_coverage_not_patient_demand_or_actual_service_area"
    ):
        raise ValueError("Stage 3 target definition changed")
    if config.get("random_seed") != 20260819:
        raise ValueError("Stage 3 deterministic seed contract changed")
    if config.get("stage_scope") != "Stage3A_grid_exposure_Stage3B_venue_context_Stage3C_spatial_integration_only":
        raise ValueError("Stage 3 scope contract changed")
    if config.get("stage3_started") is not True or config.get("stage4_started") is not False:
        raise ValueError("Stage 3 must be started and Stage 4 must remain false")
    analysis = config.get("analysis_contract", {})
    if analysis.get("stage4_execution_forbidden") is not True:
        raise ValueError("Stage 4 execution must be forbidden")
    if analysis.get("primary_exposure_method") != "hard_5000m":
        raise ValueError("Primary exposure method must remain preregistered hard_5000m")
    if analysis.get("grid_role") != "population_mass_spatial_allocation_unit_not_independent_sample":
        raise ValueError("Grid statistical-role contract changed")
    expected_analysis = {
        "expected_admin_dongs": 153,
        "expected_policy_sigungu": 11,
        "expected_grid_rows": 58311,
        "expected_input_venue_rows": 3938,
        "expected_eligible_venue_rows": 3909,
        "expected_out_of_boundary_exclusions": 29,
        "expected_bundles": 5,
        "expected_interface_rows": 19545,
        "metric_crs": "EPSG:5179",
        "venue_role": "candidate_physical_location_not_verified_operational_site",
        "primary_population": "elderly65_calibrated_population",
        "high_need_admin_quantile": 0.80,
        "quadrant_quantile": 0.50,
        "temporal_role": "provenance_only_not_primary_selector",
        "exact_month_must_be_null": True,
        "exact_date_must_be_null": True,
    }
    for key, expected in expected_analysis.items():
        if analysis.get(key) != expected:
            raise ValueError(f"Stage 3 analysis contract changed: {key}")
    catchments = config.get("catchments", {})
    expected_catchments = {
        "hard_radii_m": [1000, 3000, 5000, 10000],
        "decay_methods": ["exponential", "gaussian"],
        "decay_scales_m": [1000, 3000, 5000],
        "decay_max_distance_m": 10000,
        "matrix_dtype_hard": "uint8",
        "matrix_dtype_decay": "float32",
        "deterministic_venue_order": "venue_id_ascending",
        "deterministic_grid_order": "grid_point_index_ascending",
        "dense_distance_matrix_forbidden": True,
    }
    for key, expected in expected_catchments.items():
        if catchments.get(key) != expected:
            raise ValueError(f"Stage 3 catchment contract changed: {key}")
    overlap = config.get("overlap", {})
    expected_overlap = {
        "primary_matrix": "hard_5000m",
        "candidate_pair_center_distance_max_m": 10000,
        "output_min_jaccard": 0.15,
        "output_min_weighted_overlap_share": 0.15,
        "cluster_jaccard_min": 0.80,
        "fallback_jaccard_min": 0.60,
        "fallback_center_distance_max_m": 5000,
    }
    for key, expected in expected_overlap.items():
        if overlap.get(key) != expected:
            raise ValueError(f"Stage 3 overlap contract changed: {key}")
    transit = config.get("transit_context", {})
    expected_transit = {
        "static_bus_reference": "2025-12-09_coordinate_snapshot",
        "current_bus_reference": "2026-07-22_structure_without_coordinates",
        "gtfs_reference": "2024-03_service_baseline",
        "nearby_radius_m": 1000,
        "current_bus_distance_status": "unavailable_source_has_no_coordinates",
    }
    for key, expected in expected_transit.items():
        if transit.get(key) != expected:
            raise ValueError(f"Stage 3 transit provenance contract changed: {key}")
    shortlist = config.get("shortlist", {})
    if shortlist.get("pareto_definition") != "strict_raw_objective_dominance_primary":
        raise ValueError("Strict raw Pareto must remain primary")
    if shortlist.get("coarse_policy_screen_definition") != (
        "within_admin_3_bin_ordinal_same_objectives_no_weighted_sum"
    ) or shortlist.get("exact_raw_pareto_retained_as_primary") is not True:
        raise ValueError("Stage 3 auxiliary Pareto screen contract changed")
    if shortlist.get("pareto_quantile_bins") != 3 or shortlist.get(
        "pareto_quantile_bin_sensitivity"
    ) != [2, 3, 4, 5]:
        raise ValueError("Coarse Pareto screen/sensitivity contract changed")
    if shortlist.get("retain_all_candidates_in_interface") is not True or shortlist.get(
        "dominated_candidates_flag_only"
    ) is not True:
        raise ValueError("Stage 3 non-destructive shortlist contract changed")
    sensitivity = config.get("sensitivity", {})
    if sensitivity.get("reference_method") != "hard_5000m" or sensitivity.get(
        "own_admin_only_shadow"
    ) is not True:
        raise ValueError("Stage 3 sensitivity reference contract changed")
    execution = config.get("execution", {})
    if execution.get("single_writer_lock") is not True or execution.get("atomic_metadata_last") is not True:
        raise ValueError("Stage 3 writer/provenance contract changed")
    if execution.get("stage4_started") is not False:
        raise ValueError("Stage 4 cannot be started by Stage 3 config")
    if execution.get("run_scoped_output") is not True or execution.get("jobs") != 8:
        raise ValueError("Stage 3 run-scope or resource contract changed")
    if execution.get("memory_limit_advisory_gib") != 24:
        raise ValueError("Stage 3 memory advisory changed")
    reporting = config.get("reporting", {})
    expected_reporting = {
        "png_dpi": 300,
        "svg_required": True,
        "html_self_contained": True,
        "korean_font_fail_closed": True,
        "all_venue_correlation_is_descriptive": True,
        "admin_collapsed_correlation_required": True,
    }
    for key, expected in expected_reporting.items():
        if reporting.get(key) != expected:
            raise ValueError(f"Stage 3 reporting contract changed: {key}")
    quality = config.get("quality_gates", {})
    if quality.get("stage4_started_must_equal") is not False or quality.get(
        "tests_must_pass"
    ) is not True:
        raise ValueError("Stage 3 completion gate contract changed")
    expected_quality_counts = {
        "matrix_count_exact": 10,
        "bundle_rows_exact": 19545,
        "interface_rows_exact": 19545,
        "fallback_venue_1_coverage_exact": 3909,
        "correlation_nonfinite_cell_max": 0,
        "discriminating_constant_metric_max": 0,
        "current_bus_distance_status_mismatch_max": 0,
        "current_bus_numeric_distance_column_max": 0,
        "bundle_sensitivity_executed_rows_exact": 50,
        "pareto_objective_loo_classified_rows_exact": 8,
        "pareto_objective_loo_executed_rows_exact": 7,
        "pareto_objective_loo_not_identifiable_rows_exact": 1,
        "shortlist_tie_breaker_loo_classified_rows_exact": 4,
        "shortlist_tie_breaker_loo_executed_rows_exact": 3,
        "shortlist_tie_breaker_loo_not_identifiable_rows_exact": 1,
        "sensitivity_scope_missing_max": 0,
        "stage3_tests_passed_min": 1,
    }
    for key, expected in expected_quality_counts.items():
        if quality.get(key) != expected:
            raise ValueError(f"Stage 3 completion gate contract changed: {key}")


@contextmanager
def _single_writer_lock(output_root: Path):
    output_root.mkdir(parents=True, exist_ok=True)
    lock = output_root / ".stage3.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(f"Stage 3 writer lock already exists: {lock}") from exc
    try:
        os.write(descriptor, f"pid={os.getpid()}\n".encode("ascii"))
        os.close(descriptor)
        yield lock
    finally:
        try:
            lock.unlink()
        except FileNotFoundError:
            pass


def _input_contracts(config: Mapping[str, Any]) -> dict[str, tuple[str, str]]:
    paths = config["paths"]
    frozen = config["frozen_contract"]
    mapping = {
        "canonical_master": ("frozen_canonical_master", "canonical_master_sha256"),
        "road_integrated_master": ("road_integrated_master", "road_integrated_master_sha256"),
        "stage1_need_scores": ("stage1_need_scores", "stage1_need_scores_sha256"),
        "stage2a_bundle_scores": ("stage2a_bundle_scores", "stage2a_bundle_scores_sha256"),
        "stage2a_specialty_scores": ("stage2a_specialty_scores", "stage2a_specialty_scores_sha256"),
        "stage2b_interface": ("stage2b_interface", "stage2b_interface_sha256"),
        "stage2_inventory": ("stage2_inventory", "stage2_inventory_sha256"),
        "stage2b_inventory": ("stage2b_inventory", "stage2b_inventory_sha256"),
        "stage1_run_metadata": ("stage1_run_metadata", "stage1_run_metadata_sha256"),
        "stage2_run_metadata": ("stage2_run_metadata", "stage2_run_metadata_sha256"),
        "stage2b_run_metadata": ("stage2b_run_metadata", "stage2b_run_metadata_sha256"),
        "stage2b_iteration2_metadata": ("stage2b_iteration2_metadata", "stage2b_iteration2_metadata_sha256"),
        "stage2b_iteration2_inventory": ("stage2b_iteration2_inventory", "stage2b_iteration2_inventory_sha256"),
        "grid_gpkg": ("grid_gpkg", "grid_gpkg_sha256"),
        "grid_csv": ("grid_csv", "grid_csv_sha256"),
        "venues": ("venues", "venues_sha256"),
        "venue_road_context": ("venue_road_context", "venue_road_context_sha256"),
        "static_bus_stops": ("static_bus_stops", "static_bus_stops_sha256"),
        "current_bus_structure": ("current_bus_structure", "current_bus_structure_sha256"),
        "gtfs_stop_service": ("gtfs_stop_service", "gtfs_stop_service_sha256"),
        "admin_boundaries": ("admin_boundaries", "admin_boundaries_sha256"),
        "service_bundle_config": ("service_bundle_config", "service_bundle_config_sha256"),
    }
    return {
        source: (str(paths[path_key]), str(frozen[sha_key]).lower())
        for source, (path_key, sha_key) in mapping.items()
    }


def _inventory_snapshot(root: Path, inventory_path: Path, scope: str) -> pd.DataFrame:
    inventory = pd.read_csv(inventory_path)
    required = {"relative_path", "sha256"}
    if not required.issubset(inventory.columns):
        raise ValueError(f"{scope} inventory lacks {sorted(required - set(inventory.columns))}")
    rows: list[dict[str, Any]] = []
    for item in inventory.itertuples(index=False):
        relative = str(item.relative_path).replace("\\", "/")
        path = _resolve(root, relative)
        observed = sha256_file(path) if path.is_file() else "MISSING"
        rows.append(
            {
                "scope": scope,
                "relative_path": relative,
                "size_bytes": path.stat().st_size if path.is_file() else -1,
                "sha256": observed,
                "expected_sha256": str(item.sha256),
                "valid": observed == str(item.sha256),
            }
        )
    result = pd.DataFrame(rows)
    if not result["valid"].all():
        raise ValueError(f"{scope} inventory contains mutated or missing artifacts")
    return result


def _frozen_snapshot(root: Path, config: Mapping[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for source, (relative, expected) in _input_contracts(config).items():
        path = _resolve(root, relative)
        observed = sha256_file(path) if path.is_file() else "MISSING"
        rows.append(
            {
                "scope": f"frozen:{source}",
                "relative_path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size if path.is_file() else -1,
                "sha256": observed,
                "expected_sha256": expected,
                "valid": observed == expected,
            }
        )
    result = pd.DataFrame(rows)
    if not result["valid"].all():
        bad = result.loc[~result["valid"], "scope"].tolist()
        raise ValueError(f"Frozen Stage 3 inputs changed: {bad}")
    result = pd.concat(
        [
            result,
            _inventory_snapshot(root, _resolve(root, config["paths"]["stage2_inventory"]), "stage2"),
            _inventory_snapshot(root, _resolve(root, config["paths"]["stage2b_inventory"]), "stage2b"),
        ],
        ignore_index=True,
    )
    return result.sort_values(["scope", "relative_path"], kind="stable").reset_index(drop=True)


def _snapshot_mutations(before: pd.DataFrame, after: pd.DataFrame) -> pd.DataFrame:
    key = ["scope", "relative_path"]
    left = before[key + ["size_bytes", "sha256"]].rename(
        columns={"size_bytes": "size_before", "sha256": "sha_before"}
    )
    right = after[key + ["size_bytes", "sha256"]].rename(
        columns={"size_bytes": "size_after", "sha256": "sha_after"}
    )
    merged = left.merge(right, on=key, how="outer", indicator=True)
    changed = (merged["_merge"] != "both") | (merged["size_before"] != merged["size_after"]) | (
        merged["sha_before"] != merged["sha_after"]
    )
    return merged.loc[changed].drop(columns="_merge").reset_index(drop=True)


def _readiness_prior(venues: pd.DataFrame, config: Mapping[str, Any]) -> pd.Series:
    rules = config["readiness_prior"]
    values = np.full(len(venues), float(rules["default"]), dtype=float)

    def lift(mask: pd.Series, key: str) -> None:
        nonlocal values
        values = np.where(mask.fillna(False).to_numpy(dtype=bool), np.maximum(values, float(rules[key])), values)

    lift(pd.to_numeric(venues.get("operational_suitability_verified", 0), errors="coerce").fillna(0).eq(1), "operational_suitability_verified")
    lift(pd.to_numeric(venues.get("previous_operation_evidence_flag", 0), errors="coerce").fillna(0).eq(1), "previous_operation_evidence")
    lift(pd.to_numeric(venues.get("public_health_or_outreach_flag", 0), errors="coerce").fillna(0).eq(1), "public_health_or_outreach")
    dementia = pd.to_numeric(venues.get("dementia_service_linkage_flag", 0), errors="coerce").fillna(0).eq(1)
    lift(dementia, "official_dementia")
    lift(pd.to_numeric(venues.get("official_village_senior_flag", 0), errors="coerce").fillna(0).eq(1), "official_village_or_senior")
    lift(pd.to_numeric(venues.get("official_location_flag", 0), errors="coerce").fillna(0).eq(1), "official_location")
    return pd.Series(values, index=venues.index, name="venue_readiness_prior")


def _pareto_objectives(frame: pd.DataFrame, config: Mapping[str, Any]) -> dict[str, str]:
    high = [str(column) for column in config["shortlist"]["pareto_metrics_higher"]]
    low = [str(column) for column in config["shortlist"]["pareto_metrics_lower"]]
    configured = high + low
    if len(configured) != 8 or len(set(configured)) != len(configured):
        raise ValueError("Stage 3 requires eight unique configured Pareto objectives")
    missing = sorted(set(configured) - set(map(str, frame.columns)))
    if missing:
        raise ValueError(f"Stage 3 candidate table lacks Pareto objectives: {missing}")
    return {**{column: "higher" for column in high}, **{column: "lower" for column in low}}


def _correlation_matrix_diagnostics(
    matrix: pd.DataFrame,
    records: pd.DataFrame,
    metrics: Sequence[str],
) -> dict[str, float | int]:
    expected = list(map(str, metrics))
    if list(map(str, matrix.index)) != expected or list(map(str, matrix.columns)) != expected:
        raise ValueError("Correlation matrix axes do not match the locked metric order")
    missing = sorted(set(expected) - set(map(str, records.columns)))
    if missing:
        raise ValueError(f"Correlation records lack locked metrics: {missing}")
    values = matrix.to_numpy(dtype=float)
    return {
        "nonfinite_cell_count": int((~np.isfinite(values)).sum()),
        "symmetry_max_abs_error": float(
            np.nanmax(np.abs(values - values.T))
        ),
        "diagonal_max_abs_error": float(
            np.nanmax(np.abs(np.diag(values) - 1.0))
        ),
        "constant_metric_count": int(
            sum(records[column].nunique(dropna=True) <= 1 for column in expected)
        ),
    }


def _compute_policy_pareto_tiers(
    candidates: pd.DataFrame,
    objectives: Mapping[str, str],
    *,
    group_col: str,
    id_col: str,
    quantile_bins: int,
) -> pd.DataFrame:
    """Use strict raw-objective Pareto tiers as primary and add an auxiliary screen.

    The strict eight-axis result remains the non-destructive policy lineage.
    Within-admin ordinal bins use the same objectives without a weighted sum and
    are reported only as a coarse false-precision diagnostic; they never replace
    the strict front or remove candidates from the interface.
    """

    bins = int(quantile_bins)
    if bins < 2:
        raise ValueError("pareto quantile_bins must be at least two")
    exact = compute_pareto_tiers(
        candidates, objectives, group_col=group_col, id_col=id_col
    )
    working = exact.copy()
    quantile_objectives: dict[str, str] = {}
    quantile_columns: list[str] = []
    for column, direction in objectives.items():
        quantile_column = f"__pareto_ordinal__{column}"
        higher_is_better = str(direction).lower() in {"higher", "high", "max", "maximize", "desc", "-1"}
        lower_is_better = str(direction).lower() in {"lower", "low", "min", "minimize", "asc", "1"}
        if not (higher_is_better or lower_is_better):
            raise ValueError(f"Unsupported Pareto direction: {column}={direction}")
        percent_rank = working.groupby(group_col, sort=False)[column].rank(
            method="average",
            pct=True,
            ascending=higher_is_better,
        )
        working[quantile_column] = np.ceil(percent_rank * bins).clip(1, bins).astype(int)
        quantile_objectives[quantile_column] = "higher"
        quantile_columns.append(quantile_column)
    policy = compute_pareto_tiers(
        working,
        quantile_objectives,
        group_col=group_col,
        id_col=id_col,
    )
    result = exact.copy()
    result["exact_pareto_tier"] = result["pareto_tier"]
    result["is_exact_pareto_front"] = result["is_pareto_front"]
    result["exact_pareto_definition"] = "STRICT_RAW_OBJECTIVE_DOMINANCE_PRIMARY"
    result["coarse_policy_pareto_tier"] = policy["pareto_tier"].to_numpy()
    result["is_coarse_policy_pareto_front"] = policy["is_pareto_front"].to_numpy()
    result["coarse_policy_pareto_definition"] = (
        f"WITHIN_ADMIN_{bins}_BIN_ORDINAL_SAME_OBJECTIVES_NO_WEIGHTED_SUM"
    )
    result["pareto_definition"] = "STRICT_RAW_OBJECTIVE_DOMINANCE_PRIMARY"
    return result


def _venue_indexed_values(frame: pd.DataFrame, column: str) -> pd.Series:
    """Return a finite venue-ID keyed vector; positional assignment is forbidden."""

    if VENUE not in frame or column not in frame:
        raise KeyError(f"Cannot index sensitivity values: {VENUE}, {column}")
    result = pd.Series(
        pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=float),
        index=frame[VENUE].astype(str),
        name=column,
    )
    if result.index.duplicated().any() or not np.isfinite(result).all():
        raise ValueError(f"Sensitivity values are duplicated or nonfinite: {column}")
    return result


def _build_sensitivity_alternative(
    candidates: pd.DataFrame,
    *,
    score_by_venue: pd.Series,
    objectives: Mapping[str, str],
    overrides_by_venue: Mapping[str, pd.Series] | None = None,
    drop_objectives: Iterable[str] = (),
    quantile_bins: int | None = None,
    tie_breakers: Mapping[str, str | int] | None = None,
) -> pd.DataFrame:
    """Build one ID-aligned ablation and recompute Pareto and shortlist state."""

    venue_ids = candidates[VENUE].astype(str)
    expected = set(venue_ids)

    def align(values: pd.Series, label: str) -> np.ndarray:
        keyed = pd.Series(values.copy())
        keyed.index = keyed.index.astype(str)
        if keyed.index.duplicated().any() or set(keyed.index) != expected:
            raise ValueError(f"Sensitivity {label} venue IDs do not match candidates")
        numeric = pd.to_numeric(keyed.reindex(venue_ids), errors="raise").to_numpy(dtype=float)
        if not np.isfinite(numeric).all():
            raise ValueError(f"Sensitivity {label} contains nonfinite values")
        return numeric

    alternative = candidates.copy()
    alternative["need_weighted_exposure"] = align(score_by_venue, "score")
    for column, values in (overrides_by_venue or {}).items():
        alternative[str(column)] = align(values, str(column))
    if {"raw_elderly_exposure", "cross_admin_exposure"}.issubset(alternative.columns):
        denominator = alternative["raw_elderly_exposure"].to_numpy(dtype=float)
        alternative["cross_admin_share"] = np.divide(
            alternative["cross_admin_exposure"],
            denominator,
            out=np.zeros(len(alternative), dtype=float),
            where=denominator > 0,
        )
    excluded = {str(value) for value in drop_objectives}
    directions = {
        column: direction for column, direction in objectives.items() if column not in excluded
    }
    if not directions:
        raise ValueError("Sensitivity ablation removed every Pareto objective")
    clean = alternative.drop(
            columns=[
                "pareto_tier",
                "dominance_count",
                "dominates_count",
                "is_pareto_front",
                "is_dominated",
                "dominance_flag",
                "exact_pareto_tier",
                "is_exact_pareto_front",
                "coarse_policy_pareto_tier",
                "is_coarse_policy_pareto_front",
                "coarse_policy_pareto_definition",
                "pareto_definition",
                "exact_pareto_definition",
            ],
            errors="ignore",
        )
    if quantile_bins is None:
        pareto = compute_pareto_tiers(
            clean, directions, group_col=ADMIN, id_col=VENUE
        )
    else:
        pareto = _compute_policy_pareto_tiers(
            clean,
            directions,
            group_col=ADMIN,
            id_col=VENUE,
            quantile_bins=int(quantile_bins),
        )
    return build_admin_shortlists(
        pareto,
        group_col=ADMIN,
        id_col=VENUE,
        objectives=directions,
        tie_breakers=tie_breakers,
    ).candidates


def _bundle_exposure(
    artifact: SparseCatchment,
    venues: pd.DataFrame,
    grid: pd.DataFrame,
    *,
    grid_id_col: str,
    population_col: str = "elderly65_calibrated_population",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    aligned = grid.set_index(grid_id_col, drop=False).loc[list(artifact.grid_ids)]
    aligned_venues = venues.set_index(VENUE, drop=False).loc[list(artifact.venue_ids)]
    population = pd.to_numeric(aligned[population_col], errors="raise").to_numpy(dtype=float)
    bundle_columns = sorted(column for column in aligned if column.startswith("bundle_gap__"))
    if len(bundle_columns) != 5:
        raise ValueError(f"Expected five Stage 2A bundle columns, found {bundle_columns}")
    long_rows: list[pd.DataFrame] = []
    values: list[np.ndarray] = []
    own_values: list[np.ndarray] = []
    coo = artifact.matrix.tocoo(copy=False)
    venue_admin = aligned_venues[ADMIN].astype(str).to_numpy()
    grid_admin = aligned[ADMIN].astype(str).to_numpy()
    own_mask = venue_admin[coo.row] == grid_admin[coo.col]
    for column in bundle_columns:
        bundle_id = column.removeprefix("bundle_gap__")
        gap = pd.to_numeric(aligned[column], errors="raise").to_numpy(dtype=float) / 100.0
        exposure = np.asarray(artifact.matrix @ (population * gap)).reshape(-1)
        contribution = (
            coo.data.astype(np.float64, copy=False)
            * population[coo.col]
            * gap[coo.col]
        )
        own = np.bincount(
            coo.row[own_mask],
            weights=contribution[own_mask],
            minlength=artifact.matrix.shape[0],
        ).astype(float)
        cross = exposure - own
        if (cross < -1e-7).any():
            raise RuntimeError("Bundle own-admin exposure exceeds total exposure")
        cross = np.maximum(cross, 0.0)
        values.append(exposure)
        own_values.append(own)
        long_rows.append(
            pd.DataFrame(
                {
                    VENUE: list(artifact.venue_ids),
                    BUNDLE: bundle_id,
                    "bundle_gap_weighted_exposure": exposure,
                    "own_admin_bundle_gap_weighted_exposure": own,
                    "cross_admin_bundle_gap_weighted_exposure": cross,
                    "matrix_id": artifact.matrix_id,
                }
            )
        )
    long = pd.concat(long_rows, ignore_index=True)
    wide = pd.DataFrame(
        {
            VENUE: list(artifact.venue_ids),
            "mean_bundle_gap_weighted_exposure": np.mean(np.column_stack(values), axis=1),
            "mean_own_admin_bundle_gap_weighted_exposure": np.mean(
                np.column_stack(own_values), axis=1
            ),
        }
    )
    wide["mean_cross_admin_bundle_gap_weighted_exposure"] = (
        wide["mean_bundle_gap_weighted_exposure"]
        - wide["mean_own_admin_bundle_gap_weighted_exposure"]
    ).clip(lower=0.0)
    return long, wide


def _edge_degrees(edges: pd.DataFrame, venue_ids: Sequence[str]) -> pd.DataFrame:
    base = pd.DataFrame({VENUE: list(map(str, venue_ids))})
    if edges.empty:
        base["overlap_degree"] = 0
        base["overlap_weighted_degree"] = 0.0
        base["overlap_max_jaccard"] = 0.0
        return base
    left = edges[["venue_i", "population_weighted_jaccard", "jaccard"]].rename(
        columns={"venue_i": VENUE}
    )
    right = edges[["venue_j", "population_weighted_jaccard", "jaccard"]].rename(
        columns={"venue_j": VENUE}
    )
    joined = pd.concat([left, right], ignore_index=True)
    degree = joined.groupby(VENUE, sort=True).agg(
        overlap_degree=("jaccard", "size"),
        overlap_weighted_degree=("population_weighted_jaccard", "sum"),
        overlap_max_jaccard=("jaccard", "max"),
    )
    base = base.merge(degree, on=VENUE, how="left", validate="one_to_one")
    base["overlap_degree"] = base["overlap_degree"].fillna(0).astype(int)
    for column in ("overlap_weighted_degree", "overlap_max_jaccard"):
        base[column] = base[column].fillna(0.0)
    return base


def _complete_link_overlap_clusters(
    venue_ids: Sequence[str],
    edges: pd.DataFrame,
    representative_scores: Mapping[str, float],
    *,
    threshold: float,
) -> pd.DataFrame:
    """Greedy deterministic complete-link groups for true coverage equivalence.

    Connected components are retained separately as an overlap-network
    diagnostic because transitive chains do not imply pairwise-equivalent
    catchments.  Every multi-member cluster returned here is a clique at the
    locked Jaccard threshold.
    """

    identifiers = list(map(str, venue_ids))
    adjacency = {value: set() for value in identifiers}
    strength: dict[tuple[str, str], float] = {}
    for item in edges.loc[edges["jaccard"].ge(float(threshold))].itertuples(index=False):
        left, right, value = str(item.venue_i), str(item.venue_j), float(item.jaccard)
        adjacency[left].add(right)
        adjacency[right].add(left)
        strength[tuple(sorted((left, right)))] = value
    order = sorted(
        identifiers,
        key=lambda value: (-float(representative_scores.get(value, -np.inf)), value),
    )
    assigned: set[str] = set()
    clusters: list[list[str]] = []
    for seed in order:
        if seed in assigned:
            continue
        members = [seed]
        candidates = sorted(
            adjacency[seed] - assigned,
            key=lambda value: (-float(representative_scores.get(value, -np.inf)), value),
        )
        for candidate in candidates:
            if all(member in adjacency[candidate] for member in members):
                members.append(candidate)
        assigned.update(members)
        clusters.append(members)
    rows: list[dict[str, Any]] = []
    for number, members in enumerate(clusters, 1):
        cluster_id = f"E{number:05d}"
        representative = members[0]
        pair_values = [
            strength[tuple(sorted((left, right)))]
            for index, left in enumerate(members)
            for right in members[index + 1 :]
        ]
        minimum = min(pair_values) if pair_values else 1.0
        for venue_id in members:
            rows.append(
                {
                    VENUE: venue_id,
                    "coverage_cluster_id": cluster_id,
                    "representative_candidate": representative,
                    "cluster_size": len(members),
                    "alternate_candidate_count": len(members) - 1,
                    "is_representative": venue_id == representative,
                    "cluster_min_pairwise_jaccard": minimum,
                    "cluster_method": "GREEDY_COMPLETE_LINK_ALL_PAIRS_GE_THRESHOLD",
                }
            )
    result = pd.DataFrame(rows)
    if len(result) != len(identifiers) or result[VENUE].duplicated().any():
        raise RuntimeError("Complete-link coverage clustering lost or duplicated venues")
    if (result["cluster_min_pairwise_jaccard"] < float(threshold) - 1e-12).any():
        raise RuntimeError("Coverage-equivalent cluster violates pairwise threshold")
    return result.sort_values(VENUE, kind="stable").reset_index(drop=True)


def _direct_overlap_fallbacks(
    candidates: pd.DataFrame,
    edges: pd.DataFrame,
    *,
    minimum_jaccard: float,
    maximum_distance_m: float,
) -> pd.DataFrame:
    """Prefer directly coverage-similar physical venues, then safe context fallbacks."""

    generic = build_coverage_cluster_fallbacks(
        candidates,
        candidates,
        id_col=VENUE,
        group_col=ADMIN,
        sigungu_col="sigungu",
        cluster_col="coverage_cluster_id",
        exposure_col="need_weighted_exposure",
        physical_id_col="venue_location_id",
        coordinate_cols=("latitude", "longitude"),
        n_fallbacks=2,
    ).set_index(VENUE)
    adjacency: dict[str, list[tuple[str, float, float]]] = {}
    selected_edges = edges.loc[
        edges["jaccard"].ge(float(minimum_jaccard))
        & edges["venue_distance_m"].le(float(maximum_distance_m))
    ]
    for item in selected_edges.itertuples(index=False):
        adjacency.setdefault(str(item.venue_i), []).append(
            (str(item.venue_j), float(item.jaccard), float(item.venue_distance_m))
        )
        adjacency.setdefault(str(item.venue_j), []).append(
            (str(item.venue_i), float(item.jaccard), float(item.venue_distance_m))
        )
    physical = candidates.set_index(VENUE)["venue_location_id"].astype(str).to_dict()
    rows: list[dict[str, Any]] = []
    for venue_id in candidates[VENUE].astype(str):
        options = sorted(adjacency.get(venue_id, []), key=lambda value: (-value[1], value[2], value[0]))
        chosen: list[tuple[str, float, float, str]] = []
        for alternative, jaccard, distance in options:
            if alternative == venue_id or physical.get(alternative) == physical.get(venue_id):
                continue
            if alternative not in {item[0] for item in chosen}:
                chosen.append((alternative, jaccard, distance, "DIRECT_JACCARD_GE_060"))
            if len(chosen) == 2:
                break
        # Always scan generic candidates from rank 1.  When one direct edge is
        # already selected, starting at generic rank 2 would silently skip the
        # best distinct context fallback.
        for generic_number in (1, 2):
            if len(chosen) >= 2:
                break
            fallback = generic.loc[venue_id, f"fallback_venue_{generic_number}"]
            if pd.notna(fallback) and str(fallback) not in {item[0] for item in chosen}:
                chosen.append(
                    (
                        str(fallback),
                        np.nan,
                        float(
                            generic.loc[
                                venue_id, f"fallback_{generic_number}_distance"
                            ]
                        ),
                        "CONTEXT_FALLBACK",
                    )
                )
        record: dict[str, Any] = {VENUE: venue_id}
        for number in (1, 2):
            if len(chosen) >= number:
                alternative, jaccard, distance, reason = chosen[number - 1]
                record[f"fallback_venue_{number}"] = alternative
                record[f"fallback_{number}_jaccard"] = jaccard
                record[f"fallback_{number}_distance_m"] = distance
                record[f"fallback_{number}_reason"] = reason
            else:
                record[f"fallback_venue_{number}"] = pd.NA
                record[f"fallback_{number}_jaccard"] = np.nan
                record[f"fallback_{number}_distance_m"] = np.nan
                record[f"fallback_{number}_reason"] = "UNAVAILABLE"
        rows.append(record)
    return pd.DataFrame(rows)


def _union_diagnostics(
    artifact: SparseCatchment,
    candidates: pd.DataFrame,
    grid: pd.DataFrame,
    *,
    grid_id_col: str,
) -> pd.DataFrame:
    population = (
        grid.set_index(grid_id_col).loc[list(artifact.grid_ids), "elderly65_calibrated_population"]
    )
    row_by_id = {venue_id: index for index, venue_id in enumerate(artifact.venue_ids)}
    rows: list[dict[str, Any]] = []
    for admin, group in candidates.loc[candidates["in_admin_top5"]].groupby(ADMIN, sort=True):
        indices = [row_by_id[str(value)] for value in group[VENUE]]
        coverage = weighted_union_coverage(artifact.matrix, indices, population, binary=True)
        rows.append(
            {
                ADMIN: admin,
                "selected_venue_count": len(indices),
                "covered_grid_count": coverage.covered_grid_count,
                "unique_elderly_coverage": coverage.unique_weighted_coverage,
                "naive_elderly_coverage_sum": coverage.naive_weighted_sum,
                "redundancy_mass": coverage.redundancy_mass,
                "unique_not_above_naive": coverage.unique_weighted_coverage <= coverage.naive_weighted_sum + 1e-8,
            }
        )
    return pd.DataFrame(rows)


def _candidate_baselines(candidates: pd.DataFrame) -> pd.DataFrame:
    required = {
        ADMIN,
        VENUE,
        "admin_centroid_distance_m",
        "raw_elderly_exposure",
        "need_weighted_exposure",
        "cross_admin_exposure",
        "field_validation_unknown_count",
        "transit_context_score",
        "overlap_weighted_degree",
        "shortlist_rank",
    }
    if missing := sorted(required - set(candidates.columns)):
        raise KeyError(f"Candidate baseline input lacks columns: {missing}")
    rows: list[pd.Series] = []
    for admin, group in candidates.groupby(ADMIN, sort=True):
        definitions: dict[str, tuple[pd.DataFrame, str, bool]] = {
            "A_ADMIN_CENTROID_NEAREST": (
                group,
                "admin_centroid_distance_m",
                True,
            ),
            "B_MAX_RAW_EXPOSURE": (group, "raw_elderly_exposure", False),
            "C_MAX_NEED_EXPOSURE": (group, "need_weighted_exposure", False),
            "D_LEXICOGRAPHIC_NEED_RANK1_DIAGNOSTIC_NOT_FINAL": (
                group.loc[group["shortlist_rank"].eq(1)],
                "shortlist_rank",
                True,
            ),
        }
        chosen_by_method: dict[str, pd.Series] = {}
        for method, (pool, score_column, ascending) in definitions.items():
            if pool.empty:
                raise RuntimeError(f"Baseline method is unavailable for {admin}: {method}")
            chosen = pool.sort_values(
                [score_column, VENUE], ascending=[ascending, True], kind="stable"
            ).iloc[0].copy()
            chosen["baseline_method"] = method
            chosen["baseline_available"] = True
            chosen_by_method[method] = chosen
        proposed = chosen_by_method[
            "D_LEXICOGRAPHIC_NEED_RANK1_DIAGNOSTIC_NOT_FINAL"
        ]
        for method, chosen in chosen_by_method.items():
            for metric in (
                "raw_elderly_exposure",
                "need_weighted_exposure",
                "cross_admin_exposure",
                "field_validation_unknown_count",
                "transit_context_score",
                "overlap_weighted_degree",
            ):
                chosen[f"proposed_minus_method__{metric}"] = float(proposed[metric]) - float(
                    chosen[metric]
                )
            rows.append(chosen)
    selected = pd.DataFrame(rows)
    keep = [
        "baseline_method",
        ADMIN,
        VENUE,
        "venue_name",
        "baseline_available",
        "admin_centroid_distance_m",
        "raw_elderly_exposure",
        "need_weighted_exposure",
        "cross_admin_exposure",
        "field_validation_unknown_count",
        "transit_context_score",
        "overlap_weighted_degree",
        "proposed_minus_method__raw_elderly_exposure",
        "proposed_minus_method__need_weighted_exposure",
        "proposed_minus_method__cross_admin_exposure",
        "proposed_minus_method__field_validation_unknown_count",
        "proposed_minus_method__transit_context_score",
        "proposed_minus_method__overlap_weighted_degree",
    ]
    return selected.loc[:, keep].reset_index(drop=True)


def _bundle_rank_sensitivity(
    baseline: pd.DataFrame,
    alternatives: Mapping[str, pd.DataFrame],
    venue_admin: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare bundle-specific venue ranks without treating venues as IID."""

    key = [VENUE, BUNDLE]
    score = "bundle_gap_weighted_exposure"
    if baseline.duplicated(key).any() or baseline[key + [score]].isna().any().any():
        raise ValueError("Baseline bundle exposure key is duplicated or incomplete")
    admin_map = venue_admin[[VENUE, ADMIN]].copy()
    admin_map[VENUE] = admin_map[VENUE].astype(str)
    if admin_map[VENUE].duplicated().any() or admin_map[ADMIN].isna().any():
        raise ValueError("Venue-admin mapping is not one-to-one")
    baseline_values = baseline[key + [score]].copy()
    baseline_values[VENUE] = baseline_values[VENUE].astype(str)
    expected_keys = set(map(tuple, baseline_values[key].itertuples(index=False, name=None)))
    summaries: list[dict[str, Any]] = []
    admin_rows: list[dict[str, Any]] = []
    for scenario, raw_alternative in alternatives.items():
        alternative = raw_alternative[key + [score]].copy()
        alternative[VENUE] = alternative[VENUE].astype(str)
        if alternative.duplicated(key).any():
            raise ValueError(f"Bundle sensitivity key duplicated: {scenario}")
        observed_keys = set(map(tuple, alternative[key].itertuples(index=False, name=None)))
        if observed_keys != expected_keys:
            raise ValueError(f"Bundle sensitivity keys changed: {scenario}")
        paired = baseline_values.merge(
            alternative,
            on=key,
            how="inner",
            validate="one_to_one",
            suffixes=("_baseline", "_alternative"),
        ).merge(admin_map, on=VENUE, how="left", validate="many_to_one")
        for bundle_id, bundle_rows in paired.groupby(BUNDLE, sort=True):
            rho = bundle_rows[f"{score}_baseline"].corr(
                bundle_rows[f"{score}_alternative"], method="spearman"
            )
            bundle_admin: list[dict[str, Any]] = []
            for admin, group in bundle_rows.groupby(ADMIN, sort=True):
                base_order = group.sort_values(
                    [f"{score}_baseline", VENUE],
                    ascending=[False, True],
                    kind="stable",
                )[VENUE].tolist()
                alternative_order = group.sort_values(
                    [f"{score}_alternative", VENUE],
                    ascending=[False, True],
                    kind="stable",
                )[VENUE].tolist()
                count = min(3, len(group))
                base_top = set(base_order[:count])
                alternative_top = set(alternative_order[:count])
                union = base_top | alternative_top
                record = {
                    "scenario": str(scenario),
                    BUNDLE: str(bundle_id),
                    ADMIN: str(admin),
                    "candidate_count": int(len(group)),
                    "top3_jaccard": len(base_top & alternative_top) / len(union),
                    "best_venue_stable": base_order[0] == alternative_order[0],
                }
                admin_rows.append(record)
                bundle_admin.append(record)
            admin_frame = pd.DataFrame(bundle_admin)
            delta = (
                bundle_rows[f"{score}_alternative"]
                - bundle_rows[f"{score}_baseline"]
            ).abs()
            summaries.append(
                {
                    "scenario": str(scenario),
                    BUNDLE: str(bundle_id),
                    "venue_count": int(len(bundle_rows)),
                    "admin_count": int(admin_frame[ADMIN].nunique()),
                    "venue_rank_spearman": float(rho) if pd.notna(rho) else np.nan,
                    "admin_top3_jaccard_mean": float(admin_frame["top3_jaccard"].mean()),
                    "admin_top3_jaccard_min": float(admin_frame["top3_jaccard"].min()),
                    "region_best_venue_stability": float(
                        admin_frame["best_venue_stable"].mean()
                    ),
                    "score_mean_absolute_change": float(delta.mean()),
                    "availability_status": "EXECUTED",
                    "interpretation": "bundle_specific_spatial_rank_stability_not_accuracy",
                }
            )
    return pd.DataFrame(summaries), pd.DataFrame(admin_rows)


def _artifact_inventory(root: Path, paths: Sequence[Path]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for parent in paths:
        if not parent.exists():
            continue
        files = [parent] if parent.is_file() else [item for item in parent.rglob("*") if item.is_file()]
        for path in files:
            if path.name in {"STAGE3_ARTIFACT_INVENTORY.csv", "STAGE3_RUN_METADATA.json", ".RUNNING"}:
                continue
            relative = path.relative_to(root).as_posix()
            if relative in seen:
                continue
            seen.add(relative)
            records.append(
                {
                    "relative_path": relative,
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    result = pd.DataFrame(records).sort_values("relative_path", kind="stable").reset_index(drop=True)
    if result.empty or result["relative_path"].duplicated().any() or (result["size_bytes"] <= 0).any():
        raise ValueError("Stage 3 artifact inventory is empty, duplicated, or contains empty files")
    for item in result.itertuples(index=False):
        path = root / item.relative_path
        if path.stat().st_size != item.size_bytes or sha256_file(path) != item.sha256:
            raise RuntimeError(f"Stage 3 inventory verification failed: {item.relative_path}")
    return result


def _resolve_report_link(report_path: Path, relative_link: str) -> Path:
    """Resolve one repository-local Markdown link without allowing traversal."""

    value = str(relative_link).strip()
    if not re.fullmatch(r"[A-Za-z0-9_.\-/]+", value):
        raise ValueError(f"Stage 3 report contains an unsafe relative link: {value!r}")
    raw_parts = value.split("/")
    if any(part in {"", ".", ".."} for part in raw_parts):
        raise ValueError(f"Stage 3 report link must be canonical and relative: {value!r}")
    relative = PurePosixPath(value)
    if relative.is_absolute():
        raise ValueError(f"Stage 3 report link must be relative: {value!r}")
    report_dir = report_path.parent.resolve()
    target = report_dir.joinpath(*relative.parts).resolve()
    try:
        target.relative_to(report_dir)
    except ValueError as exc:
        raise ValueError(f"Stage 3 report link escapes its run directory: {value!r}") from exc
    return target


def _stage3_report_figure_section(
    report_path: Path,
    core_manifest: pd.DataFrame,
    spatial_manifest: pd.DataFrame,
) -> str:
    """Build and verify the canonical report's complete 12-figure link index."""

    # Keep the map module optional until the publication layer is requested.
    # A canonical report, however, is complete only with all eight spatial maps.
    from mediroad.reporting.stage3_maps import STAGE3_SPATIAL_MAP_IDS

    formats = ("png", "svg", "html")
    specifications = (
        (
            "Core diagnostic",
            "STAGE3_CORE_FIGURE_MANIFEST.csv",
            tuple(STAGE3_CORE_FIGURE_IDS),
            core_manifest,
        ),
        (
            "Spatial map",
            "STAGE3_SPATIAL_MAP_MANIFEST.csv",
            tuple(STAGE3_SPATIAL_MAP_IDS),
            spatial_manifest,
        ),
    )
    required = {
        "figure_id",
        "format",
        "relative_path",
        "size_bytes",
        "sha256",
        "external_script_count",
        "self_contained_html",
    }
    figure_rows: list[str] = []
    manifest_rows: list[str] = []
    all_figure_ids: set[str] = set()

    for layer, manifest_name, expected_ids, manifest in specifications:
        if not isinstance(manifest, pd.DataFrame) or manifest.empty:
            raise ValueError(f"Stage 3 {layer} manifest must be non-empty")
        missing = sorted(required - set(manifest.columns))
        if missing:
            raise KeyError(f"Stage 3 {layer} manifest missing columns: {missing}")

        frame = manifest.copy()
        frame["figure_id"] = frame["figure_id"].astype(str)
        frame["format"] = frame["format"].astype(str).str.lower()
        expected_pairs = {
            (figure_id, artifact_format)
            for figure_id in expected_ids
            for artifact_format in formats
        }
        observed_pairs = set(zip(frame["figure_id"], frame["format"]))
        if len(frame) != len(expected_pairs) or observed_pairs != expected_pairs:
            raise ValueError(
                f"Stage 3 {layer} report manifest must contain exact PNG/SVG/HTML triplets"
            )
        if all_figure_ids & set(expected_ids):
            raise ValueError("Stage 3 core and spatial figure IDs must be disjoint")
        all_figure_ids.update(expected_ids)

        manifest_target = _resolve_report_link(report_path, manifest_name)
        if not manifest_target.is_file() or manifest_target.stat().st_size <= 0:
            raise FileNotFoundError(
                f"Stage 3 report manifest link does not resolve: {manifest_target}"
            )
        manifest_rows.append(f"| {layer} | [{manifest_name}]({manifest_name}) |")

        for figure_id in expected_ids:
            links: dict[str, str] = {}
            for artifact_format in formats:
                row = frame.loc[
                    frame["figure_id"].eq(figure_id)
                    & frame["format"].eq(artifact_format)
                ].iloc[0]
                manifest_relative = str(row["relative_path"]).strip()
                if not manifest_relative.lower().endswith(f".{artifact_format}"):
                    raise ValueError(
                        f"Stage 3 figure extension/format mismatch: {manifest_relative}"
                    )
                relative_link = f"figures/{manifest_relative}"
                target = _resolve_report_link(report_path, relative_link)
                figures_root = (report_path.parent / "figures").resolve()
                try:
                    target.relative_to(figures_root)
                except ValueError as exc:
                    raise ValueError(
                        f"Stage 3 figure link escapes figures directory: {relative_link}"
                    ) from exc
                if not target.is_file():
                    raise FileNotFoundError(
                        f"Stage 3 report figure link does not resolve: {target}"
                    )
                try:
                    expected_size = int(row["size_bytes"])
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Stage 3 figure size is invalid: {relative_link}"
                    ) from exc
                expected_sha = str(row["sha256"])
                if (
                    expected_size <= 0
                    or target.stat().st_size != expected_size
                    or not re.fullmatch(r"[0-9a-f]{64}", expected_sha)
                    or sha256_file(target) != expected_sha
                ):
                    raise RuntimeError(
                        f"Stage 3 report figure does not match its manifest: {target}"
                    )
                if artifact_format == "html":
                    self_contained = str(row["self_contained_html"]).strip().lower()
                    external_scripts = pd.to_numeric(
                        pd.Series([row["external_script_count"]]), errors="coerce"
                    ).iloc[0]
                    html_text = target.read_text(encoding="utf-8")
                    has_external_asset = bool(
                        re.search(r"<script\b[^>]*\bsrc\s*=", html_text, flags=re.I)
                        or re.search(
                            r"<link\b[^>]*\bhref\s*=\s*['\"](?:https?:)?//",
                            html_text,
                            flags=re.I,
                        )
                    )
                    if (
                        self_contained != "true"
                        or not np.isfinite(external_scripts)
                        or external_scripts != 0
                        or "plotly" not in html_text.lower()
                        or has_external_asset
                    ):
                        raise ValueError(
                            f"Stage 3 report HTML must be self-contained: {target}"
                        )
                links[artifact_format] = relative_link
            figure_rows.append(
                f"| {layer} | `{figure_id}` | [PNG]({links['png']}) | "
                f"[SVG]({links['svg']}) | "
                f"[HTML (self-contained)]({links['html']}) |"
            )

    if len(all_figure_ids) != 12 or len(figure_rows) != 12:
        raise RuntimeError("Stage 3 canonical report requires exactly 12 figure IDs")

    return "\n".join(
        [
            "이 run-scoped `FINAL_REPORT.md`가 권위 보고서다. 상위 경로의 flat "
            "`07_stage3_spatial_exposure_venue.md`는 byte-identical 편의 alias이며, 아래 상대 링크는 "
            "CURRENT가 지정한 run-scoped 보고서 위치에서 연다.",
            "",
            "### Figure manifests",
            "",
            "| 구분 | Manifest |",
            "|---|---|",
            *manifest_rows,
            "",
            "### Figures",
            "",
            "| 구분 | Figure ID | PNG | SVG | Self-contained HTML |",
            "|---|---|---|---|---|",
            *figure_rows,
        ]
    )


def _validate_stage3_report_figure_links(
    report_path: Path,
    core_manifest: pd.DataFrame,
    spatial_manifest: pd.DataFrame,
) -> None:
    """Fail closed unless the materialized canonical report contains all links."""

    expected_section = _stage3_report_figure_section(
        report_path, core_manifest, spatial_manifest
    )
    if not report_path.is_file():
        raise FileNotFoundError(f"Stage 3 canonical report is missing: {report_path}")
    report_text = report_path.read_text(encoding="utf-8")
    if expected_section not in report_text:
        raise RuntimeError("Stage 3 canonical report figure index is missing or changed")
    relative_links = re.findall(r"\[[^\]\n]+\]\(([^)\n]+)\)", expected_section)
    if len(relative_links) != 38 or len(set(relative_links)) != 38:
        raise RuntimeError("Stage 3 canonical report must expose 36 figures and 2 manifests")
    for relative_link in relative_links:
        target = _resolve_report_link(report_path, relative_link)
        if not target.is_file() or target.stat().st_size <= 0:
            raise FileNotFoundError(
                f"Stage 3 canonical report link does not resolve: {target}"
            )


def _write_report(
    path: Path,
    *,
    run_id: str,
    stats: Mapping[str, Any],
    gate: pd.DataFrame,
    sensitivity: pd.DataFrame,
    quadrants: pd.DataFrame,
    radius_distribution: pd.DataFrame,
    venue_type_exposure: pd.DataFrame,
    bundle_sensitivity: pd.DataFrame,
    baselines: pd.DataFrame,
    boundary_detail: pd.DataFrame,
    calibration_detail: pd.DataFrame,
    pareto_granularity: pd.DataFrame,
    tie_breaker_sensitivity: pd.DataFrame,
    core_figure_manifest: pd.DataFrame,
    spatial_figure_manifest: pd.DataFrame,
) -> None:
    hard_failed = gate.loc[(gate["severity"].astype(str) == "hard") & ~gate["passed"]]
    advisory_failed = gate.loc[(gate["severity"].astype(str) == "advisory") & ~gate["passed"]]
    scenario_lines = "\n".join(
        f"- `{row.scenario}`: ρ={row.venue_rank_spearman:.3f}, Top3={row.admin_top3_jaccard_mean:.3f}, Top5={row.admin_top5_jaccard_mean:.3f}, best 유지={row.region_best_venue_stability:.3f}, Pareto 유지={row.pareto_retention:.3f}"
        for row in sensitivity.sort_values("scenario").itertuples(index=False)
    )
    quadrant_counts = quadrants["need_exposure_quadrant"].value_counts().to_dict()
    radius_lines = "\n".join(
        f"- {int(row.radius_m) // 1000}km: median={row.median:,.1f}, p75={row.p75:,.1f}, p90={row.p90:,.1f}, max={row.max:,.1f}"
        for row in radius_distribution.sort_values("radius_m").itertuples(index=False)
    )
    name_column = "admin_dong_name" if "admin_dong_name" in quadrants else ADMIN
    q1_names = ", ".join(
        quadrants.loc[quadrants["need_exposure_quadrant"].eq("Q1_HIGH_NEED_HIGH_EXPOSURE"), name_column]
        .astype(str)
        .sort_values()
        .tolist()
    ) or "없음"
    q2_names = ", ".join(
        quadrants.loc[quadrants["need_exposure_quadrant"].eq("Q2_HIGH_NEED_LOW_EXPOSURE"), name_column]
        .astype(str)
        .sort_values()
        .tolist()
    ) or "없음"
    venue_type_lines = "\n".join(
        f"- `{row.venue_type}`: n={int(row.venue_count):,}, raw median={row.raw_exposure_median:,.1f}, Need median={row.need_exposure_median:,.1f}, cross share={row.cross_admin_share_median:.1%}"
        for row in venue_type_exposure.sort_values(
            ["venue_count", "venue_type"], ascending=[False, True]
        ).head(20).itertuples(index=False)
    )
    bundle_executed = bundle_sensitivity.loc[
        bundle_sensitivity["availability_status"].eq("EXECUTED")
        & ~bundle_sensitivity["scenario"].eq("FULL_IDENTITY")
    ]
    bundle_lines = "\n".join(
        f"- `{row.bundle_id}` worst: scenario=`{row.scenario}`, ρ={row.venue_rank_spearman:.3f}, Top3={row.admin_top3_jaccard_mean:.3f}, best 유지={row.region_best_venue_stability:.3f}"
        for row in bundle_executed.sort_values("venue_rank_spearman").groupby(BUNDLE, sort=True).head(1).itertuples(index=False)
    )
    baseline_lines = "\n".join(
        f"- `{method}`: {count}개 행정동"
        for method, count in baselines["baseline_method"].value_counts().sort_index().items()
    )
    def _change_summary(frame: pd.DataFrame) -> tuple[float, float, float]:
        admin = frame.sort_values([ADMIN, VENUE]).groupby(ADMIN, sort=False).head(1)
        return (
            float(frame["need_weighted_exposure_relative_change"].abs().median()),
            float(frame["cross_admin_share_change"].abs().median()),
            float(admin["admin_best_venue_changed"].mean()),
        )
    boundary_need, boundary_cross, boundary_best = _change_summary(boundary_detail)
    calibration_need, calibration_cross, calibration_best = _change_summary(calibration_detail)
    pareto_lines = "\n".join(
        f"- {int(row.quantile_bins)} bins: coarse front={int(row.coarse_front_rows):,} ({row.coarse_front_share:.1%}), exact front={int(row.exact_front_rows):,}"
        for row in pareto_granularity.sort_values("quantile_bins").itertuples(index=False)
    )
    tie_lines = "\n".join(
        f"- remove `{row.removed_tie_breaker}`: status=`{row.availability_status}`, unique={int(row.unique_value_count)}, Top3={row.admin_top3_jaccard_mean:.3f}, Top5={row.admin_top5_jaccard_mean:.3f}, best 유지={row.region_best_venue_stability:.3f}"
        for row in tie_breaker_sensitivity.sort_values("removed_tie_breaker").itertuples(index=False)
    )
    objective_loo = sensitivity.loc[
        sensitivity["scenario"].astype(str).str.startswith("objective_loo__")
    ]
    objective_identifiable = int(objective_loo["availability_status"].eq("EXECUTED").sum())
    objective_not_identifiable = int(
        objective_loo["availability_status"].eq(
            "NOT_IDENTIFIABLE_CONSTANT_OBJECTIVE"
        ).sum()
    )
    tie_identifiable = int(tie_breaker_sensitivity["availability_status"].eq("EXECUTED").sum())
    tie_not_identifiable = int(
        tie_breaker_sensitivity["availability_status"].eq(
            "NOT_IDENTIFIABLE_CONSTANT_TIE_BREAKER"
        ).sum()
    )
    figure_link_section = _stage3_report_figure_section(
        path, core_figure_manifest, spatial_figure_manifest
    )
    content = f"""# MEDIROAD MODEL V1 Stage 3 공간수혜·후보장소 최종 보고서

- run_id: `{run_id}`
- 판정: **{'PASS' if hard_failed.empty else 'FAIL'}**
- Stage 3 complete: `{str(hard_failed.empty).lower()}`
- Stage 4 started: `false`

## 1. Stage 3 목적

100m 고령인구 격자를 독립 표본이 아닌 인구질량 배분 단위로 사용해 후보장소별 잠재 수혜 Coverage를 계산했다. 이 값은 환자수·실제 이동거리·확정 서비스권역이 아니다.

## 2. 변경된 MEDIROAD 구조

Stage 1 Need, Stage 2A 진료구성 Gap, Stage 3 beneficiary exposure를 별도 축으로 보존했다. 하나의 임의 Venue Score로 합치지 않았고 Stage 4의 다목적 최적화 입력만 준비했다.

## 3. Frozen input verification

동결 입력 및 Stage 2/2B inventory를 실행 전후 검증했다. 변경 파일은 `{stats['frozen_mutation_count']}`개다.

## 4. 100m elderly grid audit

- 격자 `{stats['grid_rows']:,}`개, 행정동 `{stats['grid_admins']}`개
- 보정 65+ 합계 `{stats['grid_population_total']:,.0f}`명
- 읍면동 정본 재조정 최대 오차 `{stats['grid_reconciliation_max_error']:.3e}`
- privacy-suppressed cell은 실제 0으로 해석하지 않았다.

## 5. Candidate venue audit

- 원본 `{stats['venue_input_rows']:,}`개
- 공간 계산 eligible `{stats['venue_eligible_rows']:,}`개
- 충북 경계 밖 bus proxy 제외 `{stats['venue_excluded_rows']}`개
- 원 시군 provenance mismatch `{stats['venue_sigungu_source_mismatch']}`개는 행정동 코드 정본으로 재파생했다.

## 6. Catchment methodology

EPSG:5179 직선거리 proxy로 hard 1/3/5/10km와 exponential·Gaussian scale 1/3/5km, 총 `{stats['matrix_count']}`개 CSR을 만들었다. primary는 결과 확인 전에 고정한 `hard_5000m`이다.

## 7. Beneficiary Exposure

5km 장소별 잠재 65+ exposure: median `{stats['exposure_median']:,.1f}`, p75 `{stats['exposure_p75']:,.1f}`, p90 `{stats['exposure_p90']:,.1f}`, max `{stats['exposure_max']:,.1f}`.

## 8. Cross-boundary coverage

cross-admin exposure가 양수인 장소 `{stats['cross_admin_positive']:,}`개, cross-sigungu 양수 `{stats['cross_sigungu_positive']:,}`개다. 전체 exposure 중 cross-admin 비중 중앙값은 `{stats['cross_admin_share_median']:.1%}`다.

## 9. Need-weighted coverage

Need는 인구량과 별도 축이다. Need 상위 20개 행정동 중 High Exposure 비율 `{stats['need_top20_high_exposure_share']:.1%}`, Exposure 상위 20개 중 Need 상위 20 포함 비율 `{stats['exposure_top20_need_overlap']:.1%}`다. 총량 계열의 높은 공통상관을 숨기지 않고 lineage atlas에 남겼으며, 정책용 상관지도는 raw volume과 catchment 평균 Need/Gap intensity 및 High-Need share를 분해했다. 분해 지도 |ρ|≥.95 쌍은 `{stats['discriminating_rho095_pairs']}`개다.

## 10. Specialty-gap-weighted coverage

5개 service bundle, 총 `{stats['bundle_rows']:,}`행(각 `{stats['venue_eligible_rows']:,}` venue)을 만들었고 Stage 1 Need와 다시 곱하지 않았다.

## 11. Venue feasibility context

현장 전력·주차·화장실·차량진입 등은 unknown을 유지했다. 제한된 검증 queue는 `{stats['field_queue_rows']}`개다. readiness는 시설유형/기존 증거 prior이며 운영가능 확정값이 아니다.

## 12. Coverage redundancy

primary overlap edge `{stats['overlap_edges']:,}`개다. 단순 연결요소는 transitive chain이라 최대 `{stats['overlap_component_max_size']}`개까지 커졌지만, 이를 동등 catchment라고 부르지 않았다. 모든 구성원 쌍 Jaccard≥0.80을 만족하는 complete-link coverage-equivalent cluster는 `{stats['coverage_clusters']:,}`개, 최대 `{stats['coverage_cluster_max_size']}`개다. unique union > naive sum 위반은 `{stats['union_violation_count']}`개다.

## 13. Pareto shortlist

Pareto front `{stats['pareto_front_rows']:,}`개(`{stats['pareto_front_share']:.1%}`), 행정동 Top3 `{stats['top3_rows']:,}`개, Top5 `{stats['top5_rows']:,}`개를 비파괴 flag로 보존했다. 큰 Pareto front는 다목적 불확실성을 뜻하며 임의 단일점수로 억지 축소하지 않았다.

strict raw-objective Pareto가 primary이며 coarse screen은 보조 진단이다.

{pareto_lines}

## 14. Sensitivity / 전체 이탈테스트

{scenario_lines}

8개 Pareto objective는 전부 분류했다. 비상수 `{objective_identifiable}`개는 식별 가능한 LOO를 실행했고, `{objective_not_identifiable}`개(`field_validation_unknown_count`)는 전 후보가 6/6 unknown이라 `NOT_IDENTIFIABLE_CONSTANT_OBJECTIVE`로 공개했다. 4개 tie-breaker도 `{tie_identifiable}`개 식별 가능, `{tie_not_identifiable}`개 상수로 분리했다.

{tie_lines}

`published_positive_only` shadow는 privacy-suppressed cell 77.4%를 실제 0으로 대치한 모델이 아니라 **missing으로 제외한 불완전 관측 진단**이다. 따라서 낮은 ρ는 보정 민감성 경고이지 primary 교체 근거가 아니다.

## 15. Stage 1 rural advisory 재검사

농촌/도시 exposure 차이는 정책 trade-off로 보고하며 Stage 1을 재학습하거나 조정하지 않았다.

## 16. Stage 2A integration

동결 5개 bundle Gap을 catchment 내 65+ 질량으로 각각 적분했다. bundle 간 우선순위는 Stage 4까지 분리 보존한다.

## 17. Stage 2B treatment

전 행 `NO_STRONG_PREFERENCE / LOW`; season/month/date는 null이다. temporal은 provenance/interface일 뿐 primary venue selector 기여는 0이다.

## 18. Stage 4 interface

`{stats['interface_rows']:,}`행 = `{stats['venue_eligible_rows']:,}` venues × 5 bundles. 최종 20회 선택·방문번호·월·날짜·팀·optimizer objective 컬럼은 없다.

## 19. Need–Exposure 정책 4분면과 한계

행정동 수: `{json.dumps(quadrant_counts, ensure_ascii=False)}`. Q2는 “버려야 할 곳”이 아니라 높은 Need에도 분산·고립 때문에 단일 장소에서 많은 인구를 포착하기 어려운 곳이다. Euclidean catchment는 실제 도로시간이나 진료수요가 아니다.

- Q1 목록: {q1_names}
- Q2 목록: {q2_names}

## 20. 반경·유형·번들·경계 민감도 상세

### 반경별 exposure 분포

{radius_lines}

### venue 유형별 exposure(상위 20개 유형)

{venue_type_lines}

### bundle별 최약 공간 시나리오

{bundle_lines}

### 기준선 비교의 정확한 의미

{baseline_lines}

`D_LEXICOGRAPHIC_NEED_RANK1_DIAGNOSTIC_NOT_FINAL`은 최종 제안 장소가 아니라, Pareto tier 뒤 Need를 우선한 진단용 한 점이다. Stage 3은 후보 집합을 보존하며 최종 장소를 선택하지 않는다.

- own-admin-only: Need 상대변화 절대값 중앙값 `{boundary_need:.1%}`, cross-share 변화 절대값 중앙값 `{boundary_cross:.1%}`, 행정동 best 변경 `{boundary_best:.1%}`
- published-positive-only shadow: Need 상대변화 절대값 중앙값 `{calibration_need:.1%}`, cross-share 변화 절대값 중앙값 `{calibration_cross:.1%}`, 행정동 best 변경 `{calibration_best:.1%}`

## 21. 상관·중복 해석

raw exposure, Need-weighted exposure, bundle exposure의 높은 상관은 공통 인구량 표면에서 생긴다. 그래서 completeness lineage 지도와 volume-adjusted intensity 지도를 분리했다. Stage 4에서는 이 세 총량축을 독립 full-weight로 다시 더하면 중복 가중이 되므로 금지한다.

## 22. Stage 3 Quality Gate

- hard: `{int((gate['severity'].astype(str) == 'hard').sum()) - len(hard_failed)}/{int((gate['severity'].astype(str) == 'hard').sum())}` {'PASS' if hard_failed.empty else 'FAIL'}
- advisory failed: `{len(advisory_failed)}`

## 23. 병목, 해결 조치, 다음 계획

- 병목: cross-admin 수혜비중이 크고 반경 정의에 따라 best venue가 변한다. 해결: own/cross 분해와 10개 catchment 전수 민감도, bundle별 순위 이탈을 모두 봉인했다.
- 병목: 모든 후보의 현장 전력·주차·화장실·차량진입 등이 미검증이다. 해결: 값을 만들지 않고 unknown으로 유지하며 현장검증 queue를 제공했다.
- 병목: strict Pareto front가 넓다. 해결: 원 front를 숨기거나 삭제하지 않고 2/3/4/5-bin 보조 screen을 함께 제시했다.
- 다음 계획: Stage 4로 넘어가기 전에 현장검증 자료와 실제 차량 운행 제약을 수집한다. **이번 실행에서는 Stage 4를 시작하지 않는다.**

## 24. Reproducibility

희소행렬·행/열 order SHA·config/code/input SHA·artifact inventory를 봉인했다. Stage 3 hard gate가 모두 통과하면 Stage 4가 이 frozen interface를 사용할 수 있으나, **이번 실행에서 Stage 4 자체는 시작하지 않았다.**

## 25. Figure·manifest direct links

{figure_link_section}
"""
    _atomic_text(path, content)
    _validate_stage3_report_figure_links(
        path, core_figure_manifest, spatial_figure_manifest
    )


def run_stage3(
    package_root: str | Path,
    *,
    config_path: str | Path = "configs/model_v1/stage3_spatial.yaml",
    jobs: int | None = None,
    promote: bool = False,
    run_id: str | None = None,
    tests_passed: int = 0,
    tests_failed: int = 0,
    test_result: Stage3TestResult | None = None,
) -> Stage3PipelineResult:
    started = _utc_now()
    started_perf = time.perf_counter()
    root = Path(package_root).resolve()
    config_file = _resolve(root, config_path)
    config = _load_yaml(config_file)
    _config_contract(config)
    if test_result is not None and (
        int(tests_passed) != int(test_result.passed)
        or int(tests_failed) != int(test_result.failed)
    ):
        raise ValueError("Stage 3 test counts disagree with sealed test evidence")
    paths = config["paths"]
    analysis = config["analysis_contract"]
    resolved_jobs = int(jobs if jobs is not None else config["execution"]["jobs"])
    if not 1 <= resolved_jobs <= (os.cpu_count() or 1):
        raise ValueError("jobs must be between 1 and available logical CPU count")
    output_root = _require_below(root, _resolve(root, paths["output_root"]), "output root")
    report_root = _require_below(root, _resolve(root, paths["report_root"]), "report root")
    code_path = Path(__file__).resolve()
    module_paths = {
        "pipeline": code_path,
        "mediroad_init": Path(__file__).with_name("__init__.py"),
        "data": Path(__file__).with_name("stage3") / "data.py",
        "spatial": Path(__file__).with_name("stage3") / "spatial.py",
        "decision": Path(__file__).with_name("stage3") / "decision.py",
        "stage3_init": Path(__file__).with_name("stage3") / "__init__.py",
        "reporting_init": Path(__file__).with_name("reporting") / "__init__.py",
        "reporting_core": Path(__file__).with_name("reporting") / "stage3.py",
        "reporting_maps": Path(__file__).with_name("reporting") / "stage3_maps.py",
        "reporting_correlation_dependency": Path(__file__).with_name("reporting") / "correlation.py",
        "reporting_triplet_dependency": Path(__file__).with_name("reporting") / "stage2b_hardening.py",
        "root_runner": root / "run_model_v1_stage3.py",
        "powershell_wrapper": root / "run_model_v1_stage3.ps1",
        "requirements_model_v1": root / "requirements-model-v1.txt",
    }
    if any(not path.is_file() for path in module_paths.values()):
        missing = [name for name, path in module_paths.items() if not path.is_file()]
        raise FileNotFoundError(f"Stage 3 code provenance files are missing: {missing}")
    config_sha = sha256_file(config_file)
    code_sha = sha256_file(code_path)
    if run_id is None:
        run_id = f"stage3_{started.strftime('%Y%m%dT%H%M%SZ')}_{code_sha[:12]}"
    if not _RUN_ID.fullmatch(run_id):
        raise ValueError(f"Invalid run_id: {run_id}")
    run_dir = output_root / "runs" / run_id
    report_dir = report_root / "runs" / run_id
    if run_dir.exists() or report_dir.exists():
        raise FileExistsError(f"Stage 3 run already exists: {run_id}")

    with _single_writer_lock(output_root):
        run_dir.mkdir(parents=True)
        report_dir.mkdir(parents=True)
        running = run_dir / ".RUNNING"
        _atomic_text(running, f"run_id={run_id}\nstarted_at_utc={started.isoformat()}\n")
        for name in (
            "00_audit",
            "01_grid_policy",
            "02_exposure",
            "03_overlap",
            "04_bundle",
            "05_feasibility",
            "06_shortlist",
            "07_validation",
            "08_interface",
        ):
            (run_dir / name).mkdir()
        (report_dir / "figures").mkdir()

        before = _frozen_snapshot(root, config)
        _atomic_csv(run_dir / "00_audit" / "frozen_snapshot_before.csv", before)
        shutil.copy2(config_file, run_dir / "00_audit" / "stage3_config_snapshot.yaml")
        code_snapshot_dir = run_dir / "00_audit" / "code_snapshot"
        code_snapshot_dir.mkdir()
        code_snapshot_rows: list[dict[str, Any]] = []
        for name, source in module_paths.items():
            snapshot = code_snapshot_dir / f"{name}{source.suffix}"
            shutil.copy2(source, snapshot)
            digest = sha256_file(source)
            if sha256_file(snapshot) != digest:
                raise RuntimeError(f"Stage 3 code snapshot copy mismatch: {name}")
            code_snapshot_rows.append(
                {
                    "module": name,
                    "source_relative_path": source.relative_to(root).as_posix(),
                    "snapshot_relative_path": snapshot.relative_to(root).as_posix(),
                    "sha256": digest,
                    "size_bytes": source.stat().st_size,
                }
            )
        code_snapshot_manifest = pd.DataFrame(code_snapshot_rows)
        _atomic_csv(
            run_dir / "00_audit" / "code_snapshot_manifest.csv",
            code_snapshot_manifest,
        )
        test_snapshot_rows: list[dict[str, Any]] = []
        test_digest = sha256()
        test_snapshot_dir = run_dir / "00_audit" / "test_snapshot"
        test_snapshot_dir.mkdir()
        for test_path in sorted((root / "tests").glob("test_*.py")):
            relative = test_path.relative_to(root).as_posix()
            digest = sha256_file(test_path)
            snapshot = test_snapshot_dir / test_path.name
            shutil.copy2(test_path, snapshot)
            if sha256_file(snapshot) != digest:
                raise RuntimeError(f"Stage 3 test snapshot copy mismatch: {relative}")
            test_digest.update(relative.encode("utf-8"))
            test_digest.update(b"\0")
            test_digest.update(bytes.fromhex(digest))
            test_snapshot_rows.append(
                {
                    "relative_path": relative,
                    "snapshot_relative_path": snapshot.relative_to(root).as_posix(),
                    "sha256": digest,
                    "size_bytes": test_path.stat().st_size,
                }
            )
        test_snapshot_manifest = pd.DataFrame(test_snapshot_rows)
        if test_result is not None and (
            len(test_snapshot_manifest) != int(test_result.test_file_count)
            or test_digest.hexdigest() != str(test_result.test_files_sha256)
            or _file_contract_sha256(root, _test_source_contract_files(root))
            != str(test_result.source_contract_sha256)
        ):
            raise RuntimeError("Repository tests or source contract changed after execution")
        _atomic_csv(
            run_dir / "00_audit" / "test_file_snapshot_manifest.csv",
            test_snapshot_manifest,
        )
        effective_test_result = test_result or Stage3TestResult(
            passed=int(tests_passed),
            failed=int(tests_failed),
            returncode=None,
            command=(),
            duration_seconds=0.0,
            test_file_count=0,
            test_files_sha256="NOT_EXECUTED",
            stdout="",
            stderr="",
        )
        _atomic_json(
            run_dir / "00_audit" / "stage3_test_execution.json",
            {
                "passed": effective_test_result.passed,
                "failed": effective_test_result.failed,
                "returncode": effective_test_result.returncode,
                "command": list(effective_test_result.command),
                "duration_seconds": effective_test_result.duration_seconds,
                "test_file_count": effective_test_result.test_file_count,
                "test_files_sha256": effective_test_result.test_files_sha256,
                "source_contract_sha256": effective_test_result.source_contract_sha256,
                "stdout": effective_test_result.stdout,
                "stderr": effective_test_result.stderr,
            },
        )

        master = _read_frame(_resolve(root, paths["frozen_canonical_master"]))
        need = _read_frame(_resolve(root, paths["stage1_need_scores"]))
        bundles = _read_frame(_resolve(root, paths["stage2a_bundle_scores"]))
        temporal = _read_frame(_resolve(root, paths["stage2b_interface"]))
        venues_raw = _read_frame(_resolve(root, paths["venues"]))
        road = _read_frame(_resolve(root, paths["venue_road_context"]))
        static_bus = _read_frame(_resolve(root, paths["static_bus_stops"]))
        current_bus = _read_frame(_resolve(root, paths["current_bus_structure"]))
        gtfs = _read_frame(_resolve(root, paths["gtfs_stop_service"]))
        grid_gdf = gpd.read_file(_resolve(root, paths["grid_gpkg"]))
        boundary = gpd.read_file(_resolve(root, paths["admin_boundaries"]))
        for frame in (master, need, bundles, temporal, venues_raw, road):
            frame[ADMIN] = canonical_string_key(frame[ADMIN])

        grid_audit = validate_grid_source(
            grid_gdf,
            expected_rows=int(analysis["expected_grid_rows"]),
            expected_admins=int(analysis["expected_admin_dongs"]),
        )
        reconciliation = reconcile_grid_population(grid_gdf, master)
        _atomic_csv(run_dir / "00_audit" / "grid_population_reconciliation.csv", reconciliation)
        boundary_result = filter_venues_to_boundary(venues_raw, boundary)
        if len(boundary_result.all_venues) != int(analysis["expected_input_venue_rows"]):
            raise ValueError("Venue input count changed")
        if len(boundary_result.eligible) != int(analysis["expected_eligible_venue_rows"]):
            raise ValueError("Eligible venue count changed")
        if len(boundary_result.exclusions) != int(analysis["expected_out_of_boundary_exclusions"]):
            raise ValueError("Out-of-boundary venue exclusion count changed")
        if _id_set_sha(boundary_result.exclusions[VENUE]) != analysis["outside_venue_id_set_sha256"]:
            raise ValueError("Out-of-boundary venue identity set changed")
        if _id_set_sha(boundary_result.eligible[VENUE]) != analysis["eligible_venue_id_set_sha256"]:
            raise ValueError("Eligible venue identity set changed")
        _atomic_parquet(run_dir / "00_audit" / "venue_audit_all.parquet", boundary_result.all_venues)
        _atomic_csv(run_dir / "00_audit" / "venue_exclusion_ledger.csv", boundary_result.exclusions)

        master_map = master.set_index(ADMIN)["policy_sigungu_name"].astype(str).to_dict()
        venue_eligible = boundary_result.eligible.copy()
        source_sigungu = venue_eligible["policy_sigungu_name"].astype(str)
        resolved_sigungu = venue_eligible[ADMIN].map(master_map)
        mismatch = source_sigungu.ne(resolved_sigungu)
        if int(mismatch.sum()) != int(analysis["eligible_sigungu_source_mismatch_count"]):
            raise ValueError("Eligible venue source-sigungu mismatch count changed")
        if _id_set_sha(venue_eligible.loc[mismatch, VENUE]) != analysis["eligible_sigungu_mismatch_id_set_sha256"]:
            raise ValueError("Eligible venue source-sigungu mismatch identity set changed")
        if resolved_sigungu.isna().any():
            raise ValueError("Eligible venue lacks frozen admin-to-sigungu mapping")
        venue_eligible["policy_sigungu_source_mismatch"] = mismatch

        grid_policy = build_grid_policy_weights(
            grid_gdf,
            need,
            bundles,
            master,
            high_need_quantile=float(analysis["high_need_admin_quantile"]),
        )
        grid_policy["matrix_grid_id"] = grid_policy["grid_point_index"].map(
            lambda value: f"{int(value):06d}"
        )
        admin_codes = master[ADMIN].astype(str).tolist()
        grid = canonicalize_grid(
            grid_policy,
            crs=str(analysis["metric_crs"]),
            grid_id_col="matrix_grid_id",
            expected_rows=int(analysis["expected_grid_rows"]),
            expected_admin_codes=admin_codes,
            expected_admin_count=int(analysis["expected_admin_dongs"]),
        )
        venue = canonicalize_venues(
            venue_eligible,
            crs=str(analysis["metric_crs"]),
            venue_id_col=VENUE,
            expected_rows=int(analysis["expected_eligible_venue_rows"]),
            expected_admin_codes=admin_codes,
            expected_admin_count=int(analysis["expected_admin_dongs"]),
            admin_to_sigungu=master_map,
        )
        _atomic_parquet(run_dir / "01_grid_policy" / "grid_policy_weights.parquet", grid)
        _atomic_parquet(run_dir / "00_audit" / "eligible_venue_universe.parquet", venue)

        catchments = config["catchments"]
        supports = {
            method: {
                float(scale): float(catchments["decay_max_distance_m"])
                for scale in catchments["decay_scales_m"]
            }
            for method in ("exponential", "gaussian")
        }
        family = build_catchment_family(
            venue,
            grid,
            config=CatchmentConfig(
                hard_radii_m=tuple(map(float, catchments["hard_radii_m"])),
                decay_scales_m=tuple(map(float, catchments["decay_scales_m"])),
                decay_max_support_m=supports,
                leafsize=32,
                dtype=np.float32,
            ),
            venue_id_col=VENUE,
            grid_id_col="matrix_grid_id",
        )
        # Binary hard matrices are stored compactly while retaining sealed order.
        for matrix_id, artifact in list(family.items()):
            if artifact.method == "hard":
                family[matrix_id] = SparseCatchment(
                    matrix_id=artifact.matrix_id,
                    method=artifact.method,
                    parameter_m=artifact.parameter_m,
                    support_m=artifact.support_m,
                    matrix=artifact.matrix.astype(np.uint8),
                    venue_ids=artifact.venue_ids,
                    grid_ids=artifact.grid_ids,
                )
        if len(family) != int(config["quality_gates"]["matrix_count_exact"]):
            raise ValueError("Unexpected Stage 3 matrix count")
        monotonic = assert_hard_radius_monotonicity(family)
        _atomic_csv(run_dir / "02_exposure" / "hard_radius_monotonicity.csv", monotonic)
        matrix_rows: list[dict[str, Any]] = []
        for matrix_id, artifact in family.items():
            matrix_path = run_dir / "02_exposure" / f"venue_grid_{matrix_id}.npz"
            _atomic_sparse(matrix_path, artifact.matrix)
            record = matrix_manifest_record(artifact, source_sha=config["frozen_contract"]["grid_gpkg_sha256"])
            record.update(
                {
                    "relative_path": matrix_path.relative_to(root).as_posix(),
                    "file_sha256": sha256_file(matrix_path),
                    "file_size_bytes": matrix_path.stat().st_size,
                    "matrix_dtype": str(artifact.matrix.dtype),
                }
            )
            matrix_rows.append(record)
        matrix_manifest = pd.DataFrame(matrix_rows).sort_values("matrix_id", kind="stable")
        _atomic_csv(run_dir / "02_exposure" / "venue_grid_matrix_manifest.csv", matrix_manifest)
        primary = family[str(analysis["primary_exposure_method"])]
        venue_order = venue.set_index(VENUE).loc[list(primary.venue_ids)].reset_index()
        venue_order.insert(0, "matrix_row", np.arange(len(venue_order), dtype=int))
        grid_order = grid.set_index("matrix_grid_id").loc[list(primary.grid_ids)].reset_index()
        grid_order.insert(0, "matrix_column", np.arange(len(grid_order), dtype=int))
        row_order_manifest = venue_order[
            ["matrix_row", VENUE, "venue_location_id", ADMIN, "policy_sigungu_name"]
        ].copy()
        column_order_manifest = grid_order[
            ["matrix_column", "matrix_grid_id", "grid_point_index", "gid", ADMIN]
        ].copy()
        if not column_order_manifest["matrix_grid_id"].astype(str).str.fullmatch(
            r"\d{6}"
        ).all():
            raise ValueError("matrix_grid_id must remain a zero-padded six-character string")
        _atomic_csv(run_dir / "02_exposure" / "venue_grid_row_order.csv", row_order_manifest)
        _atomic_csv(run_dir / "02_exposure" / "venue_grid_column_order.csv", column_order_manifest)
        _atomic_parquet(
            run_dir / "02_exposure" / "venue_grid_row_order.parquet",
            row_order_manifest,
        )
        _atomic_parquet(
            run_dir / "02_exposure" / "venue_grid_column_order.parquet",
            column_order_manifest,
        )
        _atomic_json(
            run_dir / "02_exposure" / "venue_grid_order_schema.json",
            {
                "algorithm": "MEDIROAD_STAGE3_ORDER_V1; 8-byte little-endian length + UTF-8 value",
                "venue_id_dtype": "string",
                "matrix_grid_id_dtype": "six_character_zero_padded_string",
                "csv_read_instruction": {"dtype": {"matrix_grid_id": "string"}},
                "venue_order_sha256": primary.venue_order_sha,
                "grid_order_sha256": primary.grid_order_sha,
                "parquet_manifests_are_authoritative_for_dtype_preservation": True,
            },
        )

        scenario_exposures: dict[str, pd.DataFrame] = {}
        scenario_bundle_means: dict[str, pd.DataFrame] = {}
        scenario_bundle_long: dict[str, pd.DataFrame] = {}
        primary_bundle_long: pd.DataFrame | None = None
        high_mask = grid_order["high_need_admin_flag"].to_numpy(dtype=bool)
        population = grid_order["elderly65_calibrated_population"].to_numpy(dtype=float)
        rural = grid_order["is_rural_eup_myeon"].to_numpy(dtype=float)
        for matrix_id, artifact in family.items():
            exposure = compute_venue_exposures(
                artifact,
                venue_order,
                grid_order,
                venue_id_col=VENUE,
                grid_id_col="matrix_grid_id",
                high_need_mask=high_mask,
            ).rename(
                columns={
                    "raw_exposure": "raw_elderly_exposure",
                    "high_need_exposure": "high_need_elderly_exposure",
                }
            )
            exposure["rural_elderly_exposure"] = np.asarray(artifact.matrix @ (population * rural)).reshape(-1)
            exposure["urban_elderly_exposure"] = np.asarray(artifact.matrix @ (population * (1.0 - rural))).reshape(-1)
            exposure["matrix_id"] = matrix_id
            scenario_exposures[matrix_id] = exposure
            bundle_long, bundle_mean = _bundle_exposure(
                artifact, venue_order, grid_order, grid_id_col="matrix_grid_id"
            )
            scenario_bundle_long[matrix_id] = bundle_long
            scenario_bundle_means[matrix_id] = bundle_mean
            if matrix_id == analysis["primary_exposure_method"]:
                primary_bundle_long = bundle_long
        all_exposure = pd.concat(scenario_exposures.values(), ignore_index=True)
        _atomic_parquet(run_dir / "02_exposure" / "venue_exposure_all_scenarios.parquet", all_exposure)
        if primary_bundle_long is None:
            raise RuntimeError("Primary bundle coverage was not generated")
        _atomic_parquet(run_dir / "04_bundle" / "venue_bundle_coverage.parquet", primary_bundle_long)

        transport = build_transport_and_road_context(
            venue_order,
            static_bus,
            current_bus,
            gtfs,
            road,
            metric_crs=str(analysis["metric_crs"]),
            nearby_radius_m=float(config["transit_context"]["nearby_radius_m"]),
            jobs=resolved_jobs,
        )
        transport["venue_readiness_prior"] = _readiness_prior(transport, config)
        feasibility = build_venue_feasibility_context(
            transport,
            id_col=VENUE,
            readiness_col="venue_readiness_prior",
        )
        feasibility["sigungu"] = feasibility["policy_sigungu_name"].astype(str)
        feasibility["known_outreach_overlap"] = pd.to_numeric(
            feasibility.get("previous_operation_evidence_flag", 0), errors="coerce"
        ).fillna(0.0)
        feasibility["transit_context"] = feasibility["transit_context_score"]
        team_drive = pd.to_numeric(
            feasibility["osm_nearest_mobile_team_base_drive_min_v6"], errors="raise"
        )
        feasibility["road_context"] = 1.0 / (1.0 + team_drive / 60.0)
        feasibility["road_context_interpretation"] = (
            "inverse_free_flow_OSM_team_base_drive_time_context_not_operational_schedule"
        )
        need_map = need.set_index(ADMIN)["need_score"].astype(float)
        feasibility["structural_need"] = feasibility[ADMIN].map(need_map)
        if feasibility["structural_need"].isna().any():
            raise ValueError("Venue structural Need broadcast is incomplete")

        primary_exposure = scenario_exposures[str(analysis["primary_exposure_method"])].drop(
            columns="matrix_id"
        )
        candidates = feasibility.merge(primary_exposure, on=VENUE, how="left", validate="one_to_one")
        candidates = candidates.merge(
            scenario_bundle_means[str(analysis["primary_exposure_method"])],
            on=VENUE,
            how="left",
            validate="one_to_one",
        )
        candidates["bundle_gap_weighted_exposure"] = candidates[
            "mean_bundle_gap_weighted_exposure"
        ]
        rural_map = master.set_index(ADMIN)["is_rural_eup_myeon"]
        candidates["is_rural_eup_myeon"] = candidates[ADMIN].map(rural_map)
        if candidates["is_rural_eup_myeon"].isna().any():
            raise ValueError("Venue rural/urban context broadcast is incomplete")
        candidates["cross_admin_share"] = np.divide(
            candidates["cross_admin_exposure"],
            candidates["raw_elderly_exposure"],
            out=np.zeros(len(candidates), dtype=float),
            where=candidates["raw_elderly_exposure"].to_numpy(dtype=float) > 0,
        )
        candidates["cross_sigungu_share"] = np.divide(
            candidates["cross_sigungu_exposure"],
            candidates["raw_elderly_exposure"],
            out=np.zeros(len(candidates), dtype=float),
            where=candidates["raw_elderly_exposure"].to_numpy(dtype=float) > 0,
        )
        positive_exposure = candidates["raw_elderly_exposure"].to_numpy(dtype=float) > 0
        candidates["catchment_mean_need_intensity"] = np.divide(
            candidates["need_weighted_exposure"],
            candidates["raw_elderly_exposure"],
            out=np.zeros(len(candidates), dtype=float),
            where=positive_exposure,
        )
        candidates["high_need_population_share"] = np.divide(
            candidates["high_need_elderly_exposure"],
            candidates["raw_elderly_exposure"],
            out=np.zeros(len(candidates), dtype=float),
            where=positive_exposure,
        )
        candidates["catchment_mean_bundle_gap_intensity"] = np.divide(
            candidates["mean_bundle_gap_weighted_exposure"],
            candidates["raw_elderly_exposure"],
            out=np.zeros(len(candidates), dtype=float),
            where=positive_exposure,
        )
        centroid_transformer = Transformer.from_crs(
            "EPSG:4326", str(analysis["metric_crs"]), always_xy=True
        )
        centroid_x, centroid_y = centroid_transformer.transform(
            pd.to_numeric(master["centroid_lon"], errors="raise").to_numpy(dtype=float),
            pd.to_numeric(master["centroid_lat"], errors="raise").to_numpy(dtype=float),
        )
        centroid_lookup = pd.DataFrame(
            {
                ADMIN: master[ADMIN].astype(str),
                "admin_centroid_x_5179": centroid_x,
                "admin_centroid_y_5179": centroid_y,
            }
        ).set_index(ADMIN)
        candidates["admin_centroid_x_5179"] = candidates[ADMIN].map(
            centroid_lookup["admin_centroid_x_5179"]
        )
        candidates["admin_centroid_y_5179"] = candidates[ADMIN].map(
            centroid_lookup["admin_centroid_y_5179"]
        )
        candidates["admin_centroid_distance_m"] = np.hypot(
            candidates["x_5179"] - candidates["admin_centroid_x_5179"],
            candidates["y_5179"] - candidates["admin_centroid_y_5179"],
        )
        if not np.isfinite(candidates["admin_centroid_distance_m"]).all():
            raise ValueError("Admin-centroid venue baseline distance is incomplete")
        _atomic_parquet(
            run_dir / "05_feasibility" / "venue_feasibility_context.parquet", feasibility
        )

        overlap_config = config["overlap"]
        edges = compute_pair_overlaps(
            primary,
            venue_order,
            population,
            grid_need=grid_order["normalized_need"].to_numpy(dtype=float),
            venue_id_col=VENUE,
            candidate_distance_m=float(overlap_config["candidate_pair_center_distance_max_m"]),
            min_jaccard=float(overlap_config["output_min_jaccard"]),
            min_population_weighted_jaccard=float(
                overlap_config["output_min_weighted_overlap_share"]
            ),
        )
        _atomic_parquet(run_dir / "03_overlap" / "venue_overlap_edges.parquet", edges)
        degrees = _edge_degrees(edges, primary.venue_ids)
        candidates = candidates.merge(degrees, on=VENUE, how="left", validate="one_to_one")
        representative_scores = candidates.set_index(VENUE)["need_weighted_exposure"].to_dict()
        overlap_components = connected_overlap_clusters(
            primary.venue_ids,
            edges,
            threshold=float(overlap_config["cluster_jaccard_min"]),
            representative_scores=representative_scores,
        )
        overlap_components = overlap_components.rename(
            columns={
                "coverage_cluster_id": "overlap_component_id",
                "representative_candidate": "component_representative_candidate",
                "cluster_size": "component_size",
                "alternate_candidate_count": "component_alternate_candidate_count",
                "is_representative": "is_component_representative",
            }
        )
        _atomic_csv(
            run_dir / "03_overlap" / "venue_overlap_connected_components.csv",
            overlap_components,
        )
        clusters = _complete_link_overlap_clusters(
            primary.venue_ids,
            edges,
            representative_scores,
            threshold=float(overlap_config["cluster_jaccard_min"]),
        )
        _atomic_csv(run_dir / "03_overlap" / "venue_coverage_clusters.csv", clusters)
        candidates = candidates.merge(clusters, on=VENUE, how="left", validate="one_to_one")
        candidates = candidates.merge(
            overlap_components[[VENUE, "overlap_component_id", "component_size"]],
            on=VENUE,
            how="left",
            validate="one_to_one",
        )
        if "coverage_cluster_id" not in candidates:
            raise ValueError("Coverage cluster output lacks coverage_cluster_id")

        objectives = _pareto_objectives(candidates, config)
        shortlist_tie_breakers = {
            "need_weighted_exposure": "desc",
            "raw_elderly_exposure": "desc",
            "mean_bundle_gap_weighted_exposure": "desc",
            "field_validation_unknown_count": "asc",
        }
        selector_rows = [
            {
                "consumer": "primary_exposure",
                "feature": "hard_5000m_spatial_catchment",
                "source_stage": "Stage3",
                "role": "preregistered_primary_spatial_definition",
            },
            *[
                {
                    "consumer": "pareto_objective",
                    "feature": column,
                    "source_stage": (
                        "Stage2A" if "bundle_gap" in column else "Stage1" if "need" in column else "Stage3"
                    ),
                    "role": direction,
                }
                for column, direction in objectives.items()
            ],
            *[
                {
                    "consumer": "shortlist_tie_breaker",
                    "feature": column,
                    "source_stage": (
                        "Stage2A" if "bundle_gap" in column else "Stage1" if "need" in column else "Stage3"
                    ),
                    "role": direction,
                }
                for column, direction in shortlist_tie_breakers.items()
            ],
        ]
        selector_manifest = pd.DataFrame(selector_rows)
        selector_manifest["temporal_selector_forbidden"] = selector_manifest[
            "feature"
        ].map(_is_temporal_selector_feature)
        temporal_primary_selector_count = int(
            selector_manifest["temporal_selector_forbidden"].sum()
        )
        _atomic_csv(
            run_dir / "00_audit" / "selector_feature_manifest.csv",
            selector_manifest,
        )
        pareto_bins = int(config["shortlist"]["pareto_quantile_bins"])
        candidates = _compute_policy_pareto_tiers(
            candidates,
            objectives,
            group_col=ADMIN,
            id_col=VENUE,
            quantile_bins=pareto_bins,
        )
        shortlist_result = build_admin_shortlists(
            candidates,
            group_col=ADMIN,
            id_col=VENUE,
            objectives=objectives,
            tie_breakers=shortlist_tie_breakers,
        )
        candidates = shortlist_result.candidates
        pareto_objective_audit = pd.DataFrame(
            [
                {
                    "objective": column,
                    "direction": direction,
                    "non_null_count": int(candidates[column].notna().sum()),
                    "unique_value_count": int(candidates[column].nunique(dropna=True)),
                    "constant_objective": bool(candidates[column].nunique(dropna=True) <= 1),
                    "role": "strict_exact_pareto_primary_and_coarse_screen_same_objectives",
                }
                for column, direction in objectives.items()
            ]
        )
        _atomic_csv(
            run_dir / "06_shortlist" / "pareto_objective_audit.csv",
            pareto_objective_audit,
        )
        pareto_clean = candidates.drop(
            columns=[
                "pareto_tier",
                "dominance_count",
                "dominates_count",
                "is_pareto_front",
                "is_dominated",
                "dominance_flag",
                "exact_pareto_tier",
                "is_exact_pareto_front",
                "exact_pareto_definition",
                "coarse_policy_pareto_tier",
                "is_coarse_policy_pareto_front",
                "coarse_policy_pareto_definition",
                "pareto_definition",
            ],
            errors="ignore",
        )
        exact_front_ids = set(
            candidates.loc[candidates["is_pareto_front"], VENUE].astype(str)
        )
        pareto_bin_rows: list[dict[str, Any]] = []
        for bins in config["shortlist"]["pareto_quantile_bin_sensitivity"]:
            probe = _compute_policy_pareto_tiers(
                pareto_clean,
                objectives,
                group_col=ADMIN,
                id_col=VENUE,
                quantile_bins=int(bins),
            )
            coarse_ids = set(
                probe.loc[probe["is_coarse_policy_pareto_front"], VENUE].astype(str)
            )
            admin_counts = probe.groupby(ADMIN)["is_coarse_policy_pareto_front"].sum()
            pareto_bin_rows.append(
                {
                    "quantile_bins": int(bins),
                    "coarse_front_rows": len(coarse_ids),
                    "coarse_front_share": len(coarse_ids) / len(probe),
                    "exact_front_rows": len(exact_front_ids),
                    "coarse_within_exact_fraction": (
                        len(coarse_ids & exact_front_ids) / len(coarse_ids)
                        if coarse_ids
                        else 1.0
                    ),
                    "coarse_front_admin_median": float(admin_counts.median()),
                    "coarse_front_admin_p90": float(admin_counts.quantile(0.90)),
                    "coarse_front_admin_max": int(admin_counts.max()),
                    "role": "diagnostic_coarse_screen_sensitivity_not_exact_pareto_replacement",
                }
            )
        pareto_bin_sensitivity = pd.DataFrame(pareto_bin_rows)
        _atomic_csv(
            run_dir / "06_shortlist" / "pareto_granularity_sensitivity.csv",
            pareto_bin_sensitivity,
        )
        fallbacks = _direct_overlap_fallbacks(
            candidates,
            edges,
            minimum_jaccard=float(overlap_config["fallback_jaccard_min"]),
            maximum_distance_m=float(overlap_config["fallback_center_distance_max_m"]),
        )
        candidates = candidates.merge(fallbacks, on=VENUE, how="left", validate="one_to_one")
        _atomic_parquet(run_dir / "02_exposure" / "venue_exposure_summary.parquet", candidates)
        _atomic_parquet(
            run_dir / "06_shortlist" / "venue_pareto_front.parquet",
            candidates.loc[candidates["is_pareto_front"]],
        )
        _atomic_parquet(
            run_dir / "06_shortlist" / "venue_exact_pareto_front_lineage.parquet",
            candidates.loc[candidates["is_exact_pareto_front"]],
        )
        _atomic_parquet(
            run_dir / "06_shortlist" / "venue_coarse_policy_pareto_screen.parquet",
            candidates.loc[candidates["is_coarse_policy_pareto_front"]],
        )
        _atomic_csv(
            run_dir / "06_shortlist" / "venue_shortlist.csv",
            candidates.loc[candidates["shortlist_selected"]],
        )
        _atomic_csv(run_dir / "06_shortlist" / "venue_fallbacks.csv", fallbacks)
        field_queue = build_field_validation_queue(
            candidates,
            admin_col=ADMIN,
            id_col=VENUE,
            top_k=int(config["shortlist"]["field_validation_top_n_per_admin"]),
        )
        _atomic_csv(run_dir / "06_shortlist" / "field_validation_priority.csv", field_queue)
        union = _union_diagnostics(primary, candidates, grid_order, grid_id_col="matrix_grid_id")
        _atomic_csv(run_dir / "03_overlap" / "admin_top5_union_coverage.csv", union)
        baselines = _candidate_baselines(candidates)
        _atomic_csv(run_dir / "07_validation" / "candidate_baseline_comparison.csv", baselines)

        quadrants = build_need_exposure_quadrants(
            candidates,
            group_col=ADMIN,
            id_col=VENUE,
            need_col="structural_need",
            exposure_col="need_weighted_exposure",
            collapse="best",
        )
        _atomic_csv(run_dir / "07_validation" / "need_exposure_quadrants.csv", quadrants)
        correlation_metrics = [
            "raw_elderly_exposure",
            "need_weighted_exposure",
            "high_need_elderly_exposure",
            "venue_readiness_prior",
            "transit_context_score",
            "structural_need",
            "mean_bundle_gap_weighted_exposure",
            "cross_admin_share",
            "overlap_weighted_degree",
            "osm_nearest_mobile_team_base_drive_min_v6",
        ]
        lineage_correlation = stage3_spearman_audit(
            candidates,
            correlation_metrics,
            group_col=ADMIN,
            id_col=VENUE,
            collapse="best",
            best_by="need_weighted_exposure",
            min_periods=10,
        )
        discriminating_metrics = [
            "raw_elderly_exposure",
            "catchment_mean_need_intensity",
            "high_need_population_share",
            "venue_readiness_prior",
            "transit_context_score",
            "structural_need",
            "catchment_mean_bundle_gap_intensity",
            "cross_admin_share",
            "overlap_max_jaccard",
            "osm_nearest_mobile_team_base_drive_min_v6",
        ]
        correlation = stage3_spearman_audit(
            candidates,
            discriminating_metrics,
            group_col=ADMIN,
            id_col=VENUE,
            collapse="best",
            best_by="need_weighted_exposure",
            min_periods=10,
        )
        for name, table in (
            ("all_venue_correlation", correlation.all_venue_correlation),
            ("all_venue_pairwise_n", correlation.all_venue_counts),
            ("admin_collapsed_correlation", correlation.admin_collapsed_correlation),
            ("admin_collapsed_pairwise_n", correlation.admin_collapsed_counts),
            ("admin_collapsed_records", correlation.admin_collapsed),
            ("correlation_level_comparison", correlation.comparison),
        ):
            if name.endswith("correlation") or name.endswith("pairwise_n"):
                table.to_csv(run_dir / "07_validation" / f"{name}.csv", encoding="utf-8")
            else:
                _atomic_csv(run_dir / "07_validation" / f"{name}.csv", table)
        for name, table in (
            ("lineage_all_venue_correlation", lineage_correlation.all_venue_correlation),
            ("lineage_all_venue_pairwise_n", lineage_correlation.all_venue_counts),
            ("lineage_admin_collapsed_correlation", lineage_correlation.admin_collapsed_correlation),
            ("lineage_admin_collapsed_pairwise_n", lineage_correlation.admin_collapsed_counts),
            ("lineage_correlation_level_comparison", lineage_correlation.comparison),
        ):
            if name.endswith("correlation") or name.endswith("pairwise_n"):
                table.to_csv(run_dir / "07_validation" / f"{name}.csv", encoding="utf-8")
            else:
                _atomic_csv(run_dir / "07_validation" / f"{name}.csv", table)

        scenario_frames: dict[str, pd.DataFrame] = {}
        base_context = candidates.drop(
            columns=[
                "raw_elderly_exposure",
                "need_weighted_exposure",
                "high_need_elderly_exposure",
                "own_admin_exposure",
                "other_admin_same_sigungu_exposure",
                "same_sigungu_exposure",
                "cross_admin_exposure",
                "cross_sigungu_exposure",
                "rural_elderly_exposure",
                "urban_elderly_exposure",
                "mean_bundle_gap_weighted_exposure",
                "bundle_gap_weighted_exposure",
                "pareto_tier",
                "dominance_count",
                "dominates_count",
                "is_pareto_front",
                "is_dominated",
                "dominance_flag",
                "shortlist_rank",
                "in_admin_top3",
                "in_admin_top5",
                "shortlist_selected",
                "shortlist_membership",
                "exact_pareto_tier",
                "is_exact_pareto_front",
                "exact_pareto_definition",
                "coarse_policy_pareto_tier",
                "is_coarse_policy_pareto_front",
                "coarse_policy_pareto_definition",
                "pareto_definition",
            ],
            errors="ignore",
        )
        for matrix_id in config["sensitivity"]["compare_methods"]:
            exposure = scenario_exposures[matrix_id].drop(columns="matrix_id")
            alternative = base_context.merge(exposure, on=VENUE, how="left", validate="one_to_one")
            alternative = alternative.merge(
                scenario_bundle_means[matrix_id], on=VENUE, how="left", validate="one_to_one"
            )
            alternative["bundle_gap_weighted_exposure"] = alternative[
                "mean_bundle_gap_weighted_exposure"
            ]
            alternative = _compute_policy_pareto_tiers(
                alternative,
                _pareto_objectives(alternative, config),
                group_col=ADMIN,
                id_col=VENUE,
                quantile_bins=pareto_bins,
            )
            scenario_frames[matrix_id] = build_admin_shortlists(
                alternative,
                group_col=ADMIN,
                id_col=VENUE,
                objectives=objectives,
                tie_breakers=shortlist_tie_breakers,
            ).candidates

        def score_alternative(
            name: str,
            score: pd.Series,
            *,
            drop_objectives: Iterable[str] = (),
            drop_tie_breakers: Iterable[str] = (),
            overrides: Mapping[str, pd.Series] | None = None,
        ) -> None:
            removed_ties = {str(value) for value in drop_tie_breakers}
            scenario_tie_breakers = {
                column: direction
                for column, direction in shortlist_tie_breakers.items()
                if column not in removed_ties
            }
            scenario_frames[name] = _build_sensitivity_alternative(
                candidates,
                score_by_venue=score,
                objectives=objectives,
                overrides_by_venue=overrides,
                drop_objectives=drop_objectives,
                quantile_bins=pareto_bins,
                tie_breakers=scenario_tie_breakers,
            )

        score_alternative(
            "FULL_IDENTITY",
            _venue_indexed_values(candidates, "need_weighted_exposure"),
        )

        # Exhaust every locked Pareto objective one at a time.  These are
        # non-destructive decision-structure shocks: the spatial score remains
        # fixed while Pareto membership is recomputed without exactly one axis.
        for objective in objectives:
            score_alternative(
                f"objective_loo__{objective}",
                _venue_indexed_values(candidates, "need_weighted_exposure"),
                drop_objectives=(objective,),
            )

        score_alternative(
            "ablation_no_stage1_need_weighting",
            _venue_indexed_values(candidates, "raw_elderly_exposure"),
            drop_objectives=("need_weighted_exposure", "high_need_elderly_exposure"),
            drop_tie_breakers=("need_weighted_exposure",),
        )
        zero_by_venue = pd.Series(
            np.zeros(len(candidates), dtype=float), index=candidates[VENUE].astype(str)
        )
        score_alternative(
            "ablation_own_admin_only",
            _venue_indexed_values(candidates, "own_admin_need_weighted_exposure"),
            overrides={
                "raw_elderly_exposure": _venue_indexed_values(
                    candidates, "own_admin_exposure"
                ),
                "high_need_elderly_exposure": _venue_indexed_values(
                    candidates, "own_admin_high_need_exposure"
                ),
                "mean_bundle_gap_weighted_exposure": _venue_indexed_values(
                    candidates, "mean_own_admin_bundle_gap_weighted_exposure"
                ),
                "bundle_gap_weighted_exposure": _venue_indexed_values(
                    candidates, "mean_own_admin_bundle_gap_weighted_exposure"
                ),
                "cross_admin_exposure": zero_by_venue,
                "cross_sigungu_exposure": zero_by_venue,
            },
        )
        raw_observed = np.where(
            grid_order["privacy_suppressed_flag"].astype(bool).to_numpy(),
            0.0,
            pd.to_numeric(grid_order["raw_grid_value_202410"], errors="raise").to_numpy(dtype=float),
        )
        shadow_grid = grid_order.copy()
        shadow_grid["published_positive_population_shadow"] = raw_observed
        shadow_exposure = compute_venue_exposures(
            primary,
            venue_order,
            shadow_grid,
            venue_id_col=VENUE,
            grid_id_col="matrix_grid_id",
            population_col="published_positive_population_shadow",
            high_need_mask=high_mask,
        ).rename(
            columns={
                "raw_exposure": "raw_elderly_exposure",
                "high_need_exposure": "high_need_elderly_exposure",
            }
        )
        shadow_bundle_long, shadow_bundle_mean = _bundle_exposure(
            primary,
            venue_order,
            shadow_grid,
            grid_id_col="matrix_grid_id",
            population_col="published_positive_population_shadow",
        )
        shadow_by_id = shadow_exposure.set_index(VENUE)
        shadow_bundle_by_id = shadow_bundle_mean.set_index(VENUE)
        score_alternative(
            "ablation_published_positive_only_suppressed_missing",
            shadow_by_id["need_weighted_exposure"],
            overrides={
                column: shadow_by_id[column]
                for column in (
                    "raw_elderly_exposure",
                    "high_need_elderly_exposure",
                    "own_admin_exposure",
                    "other_admin_same_sigungu_exposure",
                    "same_sigungu_exposure",
                    "cross_admin_exposure",
                    "cross_sigungu_exposure",
                )
            }
            | {
                "mean_bundle_gap_weighted_exposure": shadow_bundle_by_id[
                    "mean_bundle_gap_weighted_exposure"
                ],
                "bundle_gap_weighted_exposure": shadow_bundle_by_id[
                    "mean_bundle_gap_weighted_exposure"
                ],
            },
        )
        score_alternative(
            "ablation_without_transit_context",
            _venue_indexed_values(candidates, "need_weighted_exposure"),
            drop_objectives=("transit_context_score",),
        )
        score_alternative(
            "ablation_without_readiness_prior",
            _venue_indexed_values(candidates, "need_weighted_exposure"),
            drop_objectives=("venue_readiness_prior",),
        )
        score_alternative(
            "ablation_without_bundle_gap_weighting",
            _venue_indexed_values(candidates, "need_weighted_exposure"),
            drop_objectives=("mean_bundle_gap_weighted_exposure",),
            drop_tie_breakers=("mean_bundle_gap_weighted_exposure",),
        )
        sensitivity = compute_sensitivity_metrics(
            candidates,
            scenario_frames,
            id_col=VENUE,
            group_col=ADMIN,
            score_col="need_weighted_exposure",
            use_existing_shortlist_flags=True,
        )
        tie_breaker_rows: list[dict[str, Any]] = []
        for removed_tie_breaker in shortlist_tie_breakers:
            tie_breaker_unique_count = int(
                candidates[removed_tie_breaker].nunique(dropna=True)
            )
            tie_breaker_identifiable = tie_breaker_unique_count > 1
            remaining_tie_breakers = {
                column: direction
                for column, direction in shortlist_tie_breakers.items()
                if column != removed_tie_breaker
            }
            alternative_shortlist = build_admin_shortlists(
                candidates,
                group_col=ADMIN,
                id_col=VENUE,
                objectives=objectives,
                tie_breakers=remaining_tie_breakers,
            ).candidates
            admin_metrics: list[dict[str, float]] = []
            for admin in sorted(candidates[ADMIN].astype(str).unique()):
                base_group = candidates.loc[candidates[ADMIN].astype(str).eq(admin)]
                alt_group = alternative_shortlist.loc[
                    alternative_shortlist[ADMIN].astype(str).eq(admin)
                ]
                base_top3 = set(base_group.loc[base_group["in_admin_top3"], VENUE].astype(str))
                alt_top3 = set(alt_group.loc[alt_group["in_admin_top3"], VENUE].astype(str))
                base_top5 = set(base_group.loc[base_group["in_admin_top5"], VENUE].astype(str))
                alt_top5 = set(alt_group.loc[alt_group["in_admin_top5"], VENUE].astype(str))
                top3_union = base_top3 | alt_top3
                top5_union = base_top5 | alt_top5
                base_best = str(base_group.nsmallest(1, "shortlist_rank")[VENUE].iloc[0])
                alt_best = str(alt_group.nsmallest(1, "shortlist_rank")[VENUE].iloc[0])
                admin_metrics.append(
                    {
                        "top3_jaccard": len(base_top3 & alt_top3) / len(top3_union),
                        "top5_jaccard": len(base_top5 & alt_top5) / len(top5_union),
                        "best_stable": float(base_best == alt_best),
                    }
                )
            admin_metrics_frame = pd.DataFrame(admin_metrics)
            tie_breaker_rows.append(
                {
                    "removed_tie_breaker": removed_tie_breaker,
                    "unique_value_count": tie_breaker_unique_count,
                    "remaining_tie_breaker_count": len(remaining_tie_breakers),
                    "admin_count": len(admin_metrics_frame),
                    "admin_top3_jaccard_mean": float(admin_metrics_frame["top3_jaccard"].mean()),
                    "admin_top5_jaccard_mean": float(admin_metrics_frame["top5_jaccard"].mean()),
                    "region_best_venue_stability": float(admin_metrics_frame["best_stable"].mean()),
                    "availability_status": (
                        "EXECUTED"
                        if tie_breaker_identifiable
                        else "NOT_IDENTIFIABLE_CONSTANT_TIE_BREAKER"
                    ),
                    "interpretation": (
                        "lexicographic_tie_breaker_sensitivity_not_final_selection_accuracy"
                        if tie_breaker_identifiable
                        else "all_candidates_share_the_same_unknown_count_no_rank_effect_identifiable"
                    ),
                }
            )
        tie_breaker_sensitivity = pd.DataFrame(tie_breaker_rows)
        scope_map = {
            "FULL_IDENTITY": "identity_control",
            **{matrix_id: "catchment_definition" for matrix_id in config["sensitivity"]["compare_methods"]},
            **{
                f"objective_loo__{objective}": "pareto_objective_leave_one_out"
                for objective in objectives
            },
            "ablation_no_stage1_need_weighting": "policy_weight_ablation",
            "ablation_own_admin_only": "boundary_ablation",
            "ablation_published_positive_only_suppressed_missing": "population_calibration_shadow_with_77pct_cells_missing",
            "ablation_without_transit_context": "pareto_context_ablation",
            "ablation_without_readiness_prior": "pareto_context_ablation",
            "ablation_without_bundle_gap_weighting": "pareto_context_ablation",
        }
        sensitivity.summary["sensitivity_scope"] = sensitivity.summary["scenario"].map(scope_map)
        objective_unique_counts = {
            str(row.objective): int(row.unique_value_count)
            for row in pareto_objective_audit.itertuples(index=False)
        }
        sensitivity.summary["objective_unique_value_count"] = sensitivity.summary[
            "scenario"
        ].map(
            lambda scenario: objective_unique_counts.get(
                str(scenario).removeprefix("objective_loo__")
            )
            if str(scenario).startswith("objective_loo__")
            else np.nan
        )
        sensitivity.summary["availability_status"] = "EXECUTED"
        constant_objective_mask = (
            sensitivity.summary["scenario"].astype(str).str.startswith("objective_loo__")
            & sensitivity.summary["objective_unique_value_count"].eq(1)
        )
        sensitivity.summary.loc[
            constant_objective_mask, "availability_status"
        ] = "NOT_IDENTIFIABLE_CONSTANT_OBJECTIVE"
        sensitivity.summary["primary_method_changed"] = False
        sensitivity.summary["interpretation"] = np.where(
            sensitivity.summary["scenario"].eq(
                "ablation_published_positive_only_suppressed_missing"
            ),
            "diagnostic_only_suppressed_cells_are_missing_not_zero_low_rank_stability_expected",
            "diagnostic_stability_not_prediction_accuracy",
        )
        sensitivity.by_admin["sensitivity_scope"] = sensitivity.by_admin["scenario"].map(scope_map)
        _atomic_csv(run_dir / "07_validation" / "exposure_sensitivity.csv", sensitivity.summary)
        _atomic_parquet(run_dir / "07_validation" / "exposure_sensitivity_by_admin.parquet", sensitivity.by_admin)
        _atomic_csv(
            run_dir / "07_validation" / "shortlist_tie_breaker_sensitivity.csv",
            tie_breaker_sensitivity,
        )

        bundle_alternatives = {
            "FULL_IDENTITY": primary_bundle_long,
            **{
                matrix_id: scenario_bundle_long[matrix_id]
                for matrix_id in config["sensitivity"]["compare_methods"]
            },
        }
        bundle_sensitivity, bundle_sensitivity_admin = _bundle_rank_sensitivity(
            primary_bundle_long,
            bundle_alternatives,
            candidates[[VENUE, ADMIN]],
        )
        bundle_ids = sorted(primary_bundle_long[BUNDLE].astype(str).unique())
        bundle_not_applicable = pd.DataFrame(
            {
                "scenario": "ablation_without_bundle_gap_weighting",
                BUNDLE: bundle_ids,
                "venue_count": len(candidates),
                "admin_count": candidates[ADMIN].nunique(),
                "venue_rank_spearman": np.nan,
                "admin_top3_jaccard_mean": np.nan,
                "admin_top3_jaccard_min": np.nan,
                "region_best_venue_stability": np.nan,
                "score_mean_absolute_change": np.nan,
                "availability_status": "NOT_APPLICABLE_COMPONENT_REMOVED",
                "interpretation": "bundle rank is undefined when bundle-gap weighting is removed",
            }
        )
        bundle_sensitivity = pd.concat(
            [bundle_sensitivity, bundle_not_applicable], ignore_index=True
        )
        _atomic_csv(
            run_dir / "07_validation" / "bundle_rank_sensitivity.csv",
            bundle_sensitivity,
        )
        _atomic_parquet(
            run_dir / "07_validation" / "bundle_rank_sensitivity_by_admin.parquet",
            bundle_sensitivity_admin,
        )

        def exposure_change_panel(scenario: str) -> pd.DataFrame:
            alternative = scenario_frames[scenario]
            columns = [
                VENUE,
                ADMIN,
                "raw_elderly_exposure",
                "need_weighted_exposure",
                "cross_admin_exposure",
                "cross_admin_share",
            ]
            paired = candidates[columns].merge(
                alternative[columns],
                on=[VENUE, ADMIN],
                how="inner",
                validate="one_to_one",
                suffixes=("_baseline", "_alternative"),
            )
            if len(paired) != len(candidates):
                raise RuntimeError(f"Exposure change panel is incomplete: {scenario}")
            for metric in (
                "raw_elderly_exposure",
                "need_weighted_exposure",
                "cross_admin_exposure",
                "cross_admin_share",
            ):
                before_values = paired[f"{metric}_baseline"].to_numpy(dtype=float)
                after_values = paired[f"{metric}_alternative"].to_numpy(dtype=float)
                paired[f"{metric}_change"] = after_values - before_values
                paired[f"{metric}_relative_change"] = np.divide(
                    after_values - before_values,
                    before_values,
                    out=np.full(len(paired), np.nan, dtype=float),
                    where=np.abs(before_values) > 1e-12,
                )
            paired["cross_admin_positive_baseline"] = paired[
                "cross_admin_exposure_baseline"
            ].gt(0)
            paired["cross_admin_positive_alternative"] = paired[
                "cross_admin_exposure_alternative"
            ].gt(0)
            paired["cross_admin_majority_baseline"] = paired[
                "cross_admin_share_baseline"
            ].gt(0.5)
            paired["cross_admin_majority_alternative"] = paired[
                "cross_admin_share_alternative"
            ].gt(0.5)
            baseline_best = (
                paired.sort_values(
                    [ADMIN, "need_weighted_exposure_baseline", VENUE],
                    ascending=[True, False, True],
                    kind="stable",
                )
                .groupby(ADMIN, sort=False)
                .head(1)
                .set_index(ADMIN)[VENUE]
            )
            alternative_best = (
                paired.sort_values(
                    [ADMIN, "need_weighted_exposure_alternative", VENUE],
                    ascending=[True, False, True],
                    kind="stable",
                )
                .groupby(ADMIN, sort=False)
                .head(1)
                .set_index(ADMIN)[VENUE]
            )
            paired["baseline_best_venue"] = paired[ADMIN].map(baseline_best)
            paired["alternative_best_venue"] = paired[ADMIN].map(alternative_best)
            paired["admin_best_venue_changed"] = paired[
                "baseline_best_venue"
            ].ne(paired["alternative_best_venue"])
            paired.insert(0, "scenario", scenario)
            return paired

        boundary_detail = exposure_change_panel("ablation_own_admin_only")
        calibration_detail = exposure_change_panel(
            "ablation_published_positive_only_suppressed_missing"
        )
        _atomic_csv(
            run_dir / "07_validation" / "boundary_sensitivity.csv",
            boundary_detail,
        )
        _atomic_csv(
            run_dir / "07_validation" / "calibration_sensitivity.csv",
            calibration_detail,
        )
        radius_rows: list[dict[str, Any]] = []
        for matrix_id in ("hard_1000m", "hard_3000m", "hard_5000m", "hard_10000m"):
            values = scenario_exposures[matrix_id]["raw_elderly_exposure"].astype(float)
            radius_rows.append(
                {
                    "matrix_id": matrix_id,
                    "radius_m": int(matrix_id.removeprefix("hard_").removesuffix("m")),
                    "venue_count": len(values),
                    "median": float(values.median()),
                    "p75": float(values.quantile(0.75)),
                    "p90": float(values.quantile(0.90)),
                    "max": float(values.max()),
                }
            )
        radius_distribution = pd.DataFrame(radius_rows)
        _atomic_csv(
            run_dir / "07_validation" / "radius_exposure_distribution.csv",
            radius_distribution,
        )
        venue_type_exposure = (
            candidates.groupby("venue_type", dropna=False, sort=True)
            .agg(
                venue_count=(VENUE, "size"),
                admin_count=(ADMIN, "nunique"),
                raw_exposure_median=("raw_elderly_exposure", "median"),
                raw_exposure_p75=("raw_elderly_exposure", lambda value: value.quantile(0.75)),
                raw_exposure_p90=("raw_elderly_exposure", lambda value: value.quantile(0.90)),
                raw_exposure_max=("raw_elderly_exposure", "max"),
                need_exposure_median=("need_weighted_exposure", "median"),
                cross_admin_share_median=("cross_admin_share", "median"),
                readiness_prior_median=("venue_readiness_prior", "median"),
            )
            .reset_index()
        )
        _atomic_csv(
            run_dir / "07_validation" / "venue_type_exposure.csv",
            venue_type_exposure,
        )
        rural_diagnostic = candidates.groupby("is_rural_eup_myeon", as_index=False).agg(
            venue_count=(VENUE, "size"),
            raw_exposure_median=("raw_elderly_exposure", "median"),
            need_exposure_median=("need_weighted_exposure", "median"),
            high_need_exposure_median=("high_need_elderly_exposure", "median"),
            cross_admin_share_median=("cross_admin_share", "median"),
        )
        _atomic_csv(run_dir / "07_validation" / "rural_exposure_diagnostic.csv", rural_diagnostic)

        interface = candidates.drop(columns="bundle_gap_weighted_exposure").merge(
            primary_bundle_long.drop(columns="matrix_id"),
            on=VENUE,
            how="left",
            validate="one_to_many",
        )
        temporal_fields = temporal[
            [
                ADMIN,
                BUNDLE,
                "specialty_gap_score",
                "bundle_rank",
                "recommended_season",
                "fallback_season",
                "release_type",
                "temporal_confidence",
                "temporal_fit_score",
                "temporal_independent_n",
                "exact_month",
                "exact_date",
            ]
        ].copy()
        if temporal_fields.duplicated([ADMIN, BUNDLE]).any():
            raise ValueError("Stage 2B interface key is duplicated")
        interface = interface.merge(
            temporal_fields,
            on=[ADMIN, BUNDLE],
            how="left",
            validate="many_to_one",
        )
        interface = interface.rename(
            columns={
                "policy_sigungu_name": "sigungu_source_resolved",
                "specialty_gap_score": "specialty_gap",
                "release_type": "temporal_release_type",
            }
        )
        interface["sigungu"] = interface["sigungu_source_resolved"].astype(str)
        required_interface = [
            VENUE,
            "venue_name",
            "venue_type",
            ADMIN,
            "sigungu",
            "latitude",
            "longitude",
            "structural_need",
            "raw_elderly_exposure",
            "need_weighted_exposure",
            "high_need_elderly_exposure",
            "own_admin_exposure",
            "cross_admin_exposure",
            "cross_sigungu_exposure",
            BUNDLE,
            "specialty_gap",
            "bundle_gap_weighted_exposure",
            "venue_readiness_prior",
            "transit_context",
            "road_context",
            "known_outreach_overlap",
            "field_validation_required",
            "pareto_tier",
            "shortlist_rank",
            "coverage_cluster_id",
            "fallback_venue_1",
            "fallback_venue_2",
            "temporal_release_type",
            "temporal_confidence",
            "recommended_season",
            "fallback_season",
        ]
        extras = [
            "venue_location_id",
            "x_5179",
            "y_5179",
            "is_pareto_front",
            "exact_pareto_tier",
            "is_exact_pareto_front",
            "coarse_policy_pareto_tier",
            "is_coarse_policy_pareto_front",
            "coarse_policy_pareto_definition",
            "pareto_definition",
            "exact_pareto_definition",
            "in_admin_top3",
            "in_admin_top5",
            "shortlist_selected",
            "field_validation_unknown_count",
            "cross_admin_share",
            "cross_sigungu_share",
            "bundle_rank",
            "temporal_fit_score",
            "temporal_independent_n",
            "exact_month",
            "exact_date",
            "distance_to_current_bus_stop_20260722_status",
            "transit_context_interpretation",
            "road_context_interpretation",
        ]
        interface = interface.loc[:, required_interface + extras]
        interface = build_stage3_to_stage4_interface(
            interface,
            temporal=None,
            require_official_temporal_abstention=True,
        )
        interface = validate_stage3_to_stage4_interface(
            interface,
            require_official_temporal_abstention=True,
        )
        if len(interface) != int(analysis["expected_interface_rows"]):
            raise ValueError("Stage 3 interface row count changed")
        _atomic_parquet(
            run_dir / "08_interface" / "stage3_to_stage4_interface.parquet", interface
        )

        figure_result = render_stage3_core_figures(
            {
                "need_exposure_quadrant": quadrants,
                "correlation_all_venue": correlation.all_venue_correlation,
                "correlation_admin_collapsed": correlation.admin_collapsed_correlation,
                "sensitivity_stability": sensitivity.summary,
                "pareto_shortlist_summary": candidates,
            },
            report_dir / "figures",
            dpi=int(config["reporting"]["png_dpi"]),
        )
        _atomic_csv(report_dir / "STAGE3_CORE_FIGURE_MANIFEST.csv", figure_result.artifact_manifest)

        # Optional extended map module is a separate, tested publication layer.
        extended_figure_manifest = pd.DataFrame()
        try:
            from mediroad.reporting.stage3_maps import render_stage3_spatial_maps

            boundary_metric = boundary.to_crs(str(analysis["metric_crs"]))
            grid_map = gpd.GeoDataFrame(
                grid_order.copy(),
                geometry=gpd.points_from_xy(grid_order["x_5179"], grid_order["y_5179"]),
                crs=str(analysis["metric_crs"]),
            )
            venue_map = gpd.GeoDataFrame(
                candidates.copy(),
                geometry=gpd.points_from_xy(candidates["x_5179"], candidates["y_5179"]),
                crs=str(analysis["metric_crs"]),
            )
            admin_map = boundary_metric.merge(quadrants, on=ADMIN, how="left", validate="one_to_one")
            extended = render_stage3_spatial_maps(
                {
                    "grid_policy": grid_map,
                    "candidate_venues": venue_map,
                    "admin_boundaries": admin_map,
                    "venue_overlap_edges": edges,
                    "bundle_gap_exposure": primary_bundle_long,
                    "rural_urban_exposure": candidates,
                },
                report_dir / "figures",
                dpi=int(config["reporting"]["png_dpi"]),
            )
            extended_figure_manifest = extended.artifact_manifest
            _atomic_csv(
                report_dir / "STAGE3_SPATIAL_MAP_MANIFEST.csv", extended_figure_manifest
            )
        except ImportError:
            # Development runs before the optional publication module lands
            # remain possible, but final quality gate records the missing layer.
            extended_figure_manifest = pd.DataFrame()

        after = _frozen_snapshot(root, config)
        mutations = _snapshot_mutations(before, after)
        _atomic_csv(run_dir / "00_audit" / "frozen_snapshot_after.csv", after)
        _atomic_csv(run_dir / "00_audit" / "frozen_mutations.csv", mutations)

        forbidden = {
            "selected_final",
            "visit_number",
            "assigned_month",
            "assigned_date",
            "final_team",
            "optimizer_objective",
        }
        interface_forbidden = len(forbidden & set(map(str.lower, interface.columns)))
        exposure_numeric = all_exposure.select_dtypes(include=[np.number]).to_numpy(dtype=float)
        matrix_shape_ok = all(
            artifact.matrix.shape
            == (
                int(analysis["expected_eligible_venue_rows"]),
                int(analysis["expected_grid_rows"]),
            )
            for artifact in family.values()
        )
        order_ok = all(
            artifact.venue_order_sha == primary.venue_order_sha
            and artifact.grid_order_sha == primary.grid_order_sha
            for artifact in family.values()
        )
        decay_values = np.concatenate(
            [artifact.matrix.data for artifact in family.values() if artifact.method != "hard"]
        )
        current_coordinate_columns = {
            column.lower()
            for column in current_bus.columns
            if column.lower() in {"latitude", "longitude", "lat", "lon", "x", "y"}
        }
        current_bus_numeric_distance_columns = {
            column
            for column in feasibility.columns
            if str(column).startswith("distance_to_current_bus_stop_20260722")
            and not str(column).endswith("_status")
        }
        current_bus_status_mismatch = int(
            feasibility["distance_to_current_bus_stop_20260722_status"]
            .astype(str)
            .ne(config["transit_context"]["current_bus_distance_status"])
            .sum()
        )
        correlation_values = correlation.admin_collapsed_correlation.to_numpy(dtype=float)
        correlation_all_diagnostics = _correlation_matrix_diagnostics(
            correlation.all_venue_correlation,
            candidates,
            discriminating_metrics,
        )
        correlation_admin_diagnostics = _correlation_matrix_diagnostics(
            correlation.admin_collapsed_correlation,
            correlation.admin_collapsed,
            discriminating_metrics,
        )
        correlation_upper = np.abs(
            correlation_values[np.triu_indices_from(correlation_values, k=1)]
        )
        discriminating_rho095_pairs = int((correlation_upper >= 0.95).sum())
        discriminating_exact_pairs = int(np.isclose(correlation_upper, 1.0, atol=1e-12).sum())
        preliminary_inventory = _artifact_inventory(root, [run_dir, report_dir])
        gates: list[dict[str, Any]] = []

        def hard(gate_id: str, observed: Any, comparison: str, threshold: Any, note: str) -> None:
            gates.append(
                quality_gate_record(
                    gate_id,
                    observed=observed,
                    comparison=comparison,
                    threshold=threshold,
                    severity="hard",
                    threshold_source="stage3 prompt, frozen-input, and post-diagnostic hardening contract",
                    note=note,
                )
            )

        def advisory(gate_id: str, observed: Any, comparison: str, threshold: Any, note: str) -> None:
            gates.append(
                quality_gate_record(
                    gate_id,
                    observed=observed,
                    comparison=comparison,
                    threshold=threshold,
                    severity="advisory",
                    threshold_source="stage3 prompt advisory contract",
                    note=note,
                )
            )

        q = config["quality_gates"]
        hard("frozen_artifact_mutation_count", len(mutations), "<=", q["frozen_artifact_mutation_count_max"], "Stage 1/2 and source artifacts must remain byte-identical")
        hard("grid_rows", len(grid), "==", q["grid_rows_exact"], "100m grid row contract")
        hard("grid_id_duplicates", int(grid["matrix_grid_id"].duplicated().sum()), "<=", q["grid_id_duplicate_max"], "Sealed matrix grid IDs are unique")
        hard("grid_admin_coverage", grid[ADMIN].nunique(), "==", q["grid_admin_count_exact"], "All 153 admins require grid population")
        hard("grid_sigungu_coverage", grid["policy_sigungu_name"].nunique(), "==", analysis["expected_policy_sigungu"], "Grid retains all 11 policy sigungu")
        hard("grid_geometry_invalid", int((~grid_gdf.geometry.is_valid | grid_gdf.geometry.is_empty).sum()), "<=", q["grid_geometry_invalid_max"], "Grid point geometry validity")
        hard("grid_population_missing", int(grid["elderly65_calibrated_population"].isna().sum()), "<=", q["grid_population_missing_max"], "Population mass must be complete")
        hard("grid_population_negative", int((grid["elderly65_calibrated_population"] < 0).sum()), "<=", q["grid_population_negative_max"], "Population mass must be nonnegative")
        hard("grid_calibration_admin_max_abs_error", float(reconciliation["absolute_difference"].max()), "<=", q["grid_calibration_admin_max_abs_error"], "Grid totals reconcile to frozen admin totals")
        hard("venue_input_rows", len(boundary_result.all_venues), "==", q["venue_input_rows_exact"], "Canonical venue source count")
        hard("venue_eligible_rows", len(venue), "==", q["venue_eligible_rows_exact"], "Boundary-safe venue universe")
        hard("venue_outside_exclusion_rows", len(boundary_result.exclusions), "==", q["venue_out_of_boundary_exclusion_rows_exact"], "Known external bus proxies excluded")
        hard("venue_eligible_outside_boundary", int((~venue["inside_chungbuk_boundary"].astype(bool)).sum()), "<=", q["venue_eligible_out_of_boundary_max"], "No eligible venue lies outside Chungbuk")
        hard("venue_id_duplicates", int(venue[VENUE].duplicated().sum()), "<=", q["venue_id_duplicate_max"], "Venue IDs are unique")
        hard("venue_location_id_duplicates", int(venue["venue_location_id"].duplicated().sum()), "<=", q["venue_location_id_duplicate_max"], "Physical location IDs are unique")
        hard("venue_coordinate_duplicates", int(venue.duplicated(["x_5179", "y_5179"]).sum()), "<=", q["venue_coordinate_duplicate_max"], "Physical candidate coordinates are unique")
        hard("venue_coordinate_invalid", int((~np.isfinite(venue[["x_5179", "y_5179"]].to_numpy(dtype=float))).sum()), "<=", q["venue_coordinate_invalid_max"], "Venue coordinates are finite")
        hard("venue_admin_coverage", venue[ADMIN].nunique(), "==", q["venue_admin_count_exact"], "Every admin retains an eligible venue")
        hard("venue_sigungu_coverage", venue["policy_sigungu_name"].nunique(), "==", analysis["expected_policy_sigungu"], "Resolved venues retain all 11 policy sigungu")
        hard("venue_source_sigungu_mismatch", int(mismatch.sum()), "==", q["venue_eligible_sigungu_source_mismatch_exact"], "Known source provenance mismatch remains explicit")
        hard("venue_resolved_sigungu_mismatch", int(venue["policy_sigungu_name"].ne(venue[ADMIN].map(master_map)).sum()), "<=", q["venue_resolved_sigungu_mismatch_max"], "Frozen admin mapping resolves every venue")
        hard("matrix_count", len(family), "==", q["matrix_count_exact"], "Four hard and six decay scenarios")
        hard("matrix_shape", int(matrix_shape_ok), "==", 1, "Every CSR has sealed 3909x58311 shape")
        hard("matrix_shape_config_contract", int(list(primary.matrix.shape) == list(q["matrix_shape_exact"])), "==", 1, "Primary sparse matrix matches the explicit config dimensions")
        hard("matrix_order_sha", int(order_ok), "==", 1, "Every CSR shares sealed row/column order")
        hard("hard_matrix_dtype", int(all(str(item.matrix.dtype) == catchments["matrix_dtype_hard"] for item in family.values() if item.method == "hard")), "==", 1, "Hard matrices use compact uint8 CSR")
        hard("decay_matrix_dtype", int(all(str(item.matrix.dtype) == catchments["matrix_dtype_decay"] for item in family.values() if item.method != "hard")), "==", 1, "Decay matrices use float32 CSR")
        hard("sparse_matrix_only", int(all(sparse.issparse(item.matrix) for item in family.values())), "==", 1, "Dense venue-grid matrices are forbidden")
        hard("hard_radius_monotonicity", int(monotonic["missing_from_larger"].sum()), "<=", q["hard_radius_monotonic_violation_max"], "Hard catchments must be nested")
        hard("decay_weight_below_zero", int((decay_values < 0.0).sum()), "<=", q["decay_weight_below_zero_max"], "Decay weights cannot be negative")
        hard("decay_weight_above_one", int((decay_values > 1.0).sum()), "<=", q["decay_weight_above_one_max"], "Decay weights cannot exceed one")
        hard("exposure_nonfinite", int((~np.isfinite(exposure_numeric)).sum()), "<=", q["exposure_nonfinite_max"], "All exposure outputs finite")
        hard("exposure_negative", int((exposure_numeric < -1e-8).sum()), "<=", q["exposure_negative_max"], "All exposure outputs nonnegative")
        hard("cross_boundary_positive_venue", int((candidates["cross_admin_exposure"] > 0).sum()), ">=", q["cross_boundary_positive_venue_min"], "Cross-admin catchments must be enabled")
        hard("own_plus_cross_partition", float((candidates["own_admin_exposure"] + candidates["cross_admin_exposure"] - candidates["raw_elderly_exposure"]).abs().max()), "<=", 1e-5, "Own plus cross-admin exposure equals total")
        hard("unique_union_above_naive", int((~union["unique_not_above_naive"]).sum()), "<=", q["unique_union_exceeds_naive_sum_max"], "Unique coverage cannot double-count above naive sum")
        hard("bundle_rows", len(primary_bundle_long), "==", q["bundle_rows_exact"], "Five bundle exposures for each eligible venue")
        hard("bundle_count", primary_bundle_long[BUNDLE].nunique(), "==", analysis["expected_bundles"], "All five Stage 2A bundles remain separate")
        hard("interface_rows", len(interface), "==", q["interface_rows_exact"], "Complete venue-bundle handoff")
        hard("interface_key_duplicates", int(interface.duplicated([VENUE, BUNDLE]).sum()), "<=", q["interface_key_duplicate_max"], "Interface key unique")
        hard("interface_forbidden_stage4_columns", interface_forbidden, "<=", q["interface_forbidden_stage4_column_max"], "No Stage 4 selection or schedule fields")
        hard("temporal_recommended_season_nonnull", int(interface["recommended_season"].notna().sum()), "<=", q["interface_recommended_season_nonnull_max"], "Official temporal abstention retained")
        hard("temporal_fallback_season_nonnull", int(interface["fallback_season"].notna().sum()), "<=", q["interface_fallback_season_nonnull_max"], "Official temporal abstention retained")
        hard("temporal_exact_month_nonnull", int(interface["exact_month"].notna().sum()), "<=", q["interface_exact_month_nonnull_max"], "Exact month remains unavailable")
        hard("temporal_exact_date_nonnull", int(interface["exact_date"].notna().sum()), "<=", q["interface_exact_date_nonnull_max"], "Exact date remains unavailable")
        hard("temporal_primary_selector_count", temporal_primary_selector_count, "<=", q["temporal_primary_selector_count_max"], "Selector manifest contains no temporal input")
        hard("current_bus_coordinate_columns", len(current_coordinate_columns), "==", 0, "2026 current bus source has no coordinates; distance is unavailable")
        hard("current_bus_distance_status_mismatch", current_bus_status_mismatch, "<=", q["current_bus_distance_status_mismatch_max"], "Every candidate explicitly records that current-bus distance is unavailable")
        hard("current_bus_numeric_distance_columns", len(current_bus_numeric_distance_columns), "<=", q["current_bus_numeric_distance_column_max"], "No numeric 2026 current-bus distance may be fabricated")
        hard("admin_shortlist_coverage", candidates.loc[candidates["shortlist_selected"], ADMIN].nunique(), "==", q["admin_shortlist_coverage_exact"], "Every admin has shortlist options")
        hard("field_validation_queue_admin_coverage", field_queue["admin_dong"].nunique(), "==", q["field_validation_queue_admin_coverage_exact"], "Bounded field queue covers every admin")
        hard("fallback_venue_1_coverage", int(fallbacks["fallback_venue_1"].notna().sum()), "==", q["fallback_venue_1_coverage_exact"], "Every eligible venue has at least one distinct fallback")
        hard("coverage_cluster_min_pairwise_jaccard", float(clusters["cluster_min_pairwise_jaccard"].min()), ">=", q["coverage_cluster_min_pairwise_jaccard"], "Coverage-equivalent groups require all-pairs overlap at the locked threshold")
        hard("correlation_all_venue_pairwise_n", int(correlation.all_venue_counts.to_numpy().min()), "==", q["correlation_all_venue_pairwise_n_exact"], "All-venue panel is descriptive and complete")
        hard("correlation_admin_collapsed_pairwise_n", int(correlation.admin_collapsed_counts.to_numpy().min()), "==", q["correlation_admin_collapsed_pairwise_n_exact"], "Policy evidence uses one record per admin")
        hard("correlation_nonfinite_cells", int(correlation_all_diagnostics["nonfinite_cell_count"] + correlation_admin_diagnostics["nonfinite_cell_count"]), "<=", q["correlation_nonfinite_cell_max"], "All discriminating correlation cells must be finite")
        hard("correlation_symmetry_max_abs_error", max(float(correlation_all_diagnostics["symmetry_max_abs_error"]), float(correlation_admin_diagnostics["symmetry_max_abs_error"])), "<=", q["correlation_symmetry_max_abs_error"], "Both discriminating correlation matrices must be symmetric")
        hard("correlation_diagonal_max_abs_error", max(float(correlation_all_diagnostics["diagonal_max_abs_error"]), float(correlation_admin_diagnostics["diagonal_max_abs_error"])), "<=", q["correlation_diagonal_max_abs_error"], "Both discriminating correlation matrices must have unit diagonal")
        hard("discriminating_constant_metrics", max(int(correlation_all_diagnostics["constant_metric_count"]), int(correlation_admin_diagnostics["constant_metric_count"])), "<=", q["discriminating_constant_metric_max"], "Every discriminating atlas metric varies at both venue and admin levels")
        hard("discriminating_abs_rho_095_pairs", discriminating_rho095_pairs, "<=", q["discriminating_abs_rho_095_pairs_max"], "Volume/intensity decomposition limits near-duplicate axes")
        hard("discriminating_exact_duplicate_pairs", discriminating_exact_pairs, "<=", q["discriminating_exact_duplicate_pairs_max"], "Discriminating correlation atlas contains no exact duplicate axes")
        bundle_executed = bundle_sensitivity["availability_status"].eq("EXECUTED")
        hard("bundle_sensitivity_executed_rows", int(bundle_executed.sum()), "==", q["bundle_sensitivity_executed_rows_exact"], "Ten spatial definitions times five bundles are tested")
        objective_loo_rows = sensitivity.summary["scenario"].astype(str).str.startswith(
            "objective_loo__"
        )
        hard("pareto_objective_loo_classified_rows", int(objective_loo_rows.sum()), "==", q["pareto_objective_loo_classified_rows_exact"], "Every locked Pareto objective is explicitly classified")
        hard("pareto_objective_loo_executed_rows", int(sensitivity.summary.loc[objective_loo_rows, "availability_status"].eq("EXECUTED").sum()), "==", q["pareto_objective_loo_executed_rows_exact"], "Every nonconstant Pareto objective has an identifiable leave-one-out rerun")
        hard("pareto_objective_loo_not_identifiable_rows", int(sensitivity.summary.loc[objective_loo_rows, "availability_status"].eq("NOT_IDENTIFIABLE_CONSTANT_OBJECTIVE").sum()), "==", q["pareto_objective_loo_not_identifiable_rows_exact"], "Constant Pareto objectives are disclosed rather than presented as informative shocks")
        hard("shortlist_tie_breaker_loo_classified_rows", len(tie_breaker_sensitivity), "==", q["shortlist_tie_breaker_loo_classified_rows_exact"], "Every lexicographic shortlist tie-breaker is explicitly classified")
        hard("shortlist_tie_breaker_loo_executed_rows", int(tie_breaker_sensitivity["availability_status"].eq("EXECUTED").sum()), "==", q["shortlist_tie_breaker_loo_executed_rows_exact"], "Every nonconstant shortlist tie-breaker has an identifiable leave-one-out rerun")
        hard("shortlist_tie_breaker_loo_not_identifiable_rows", int(tie_breaker_sensitivity["availability_status"].eq("NOT_IDENTIFIABLE_CONSTANT_TIE_BREAKER").sum()), "==", q["shortlist_tie_breaker_loo_not_identifiable_rows_exact"], "Constant tie-breakers are disclosed rather than presented as informative shocks")
        hard("sensitivity_scope_missing", int(sensitivity.summary["sensitivity_scope"].isna().sum()), "<=", q["sensitivity_scope_missing_max"], "Every sensitivity run has explicit execution semantics")
        identity_rows = sensitivity.summary.loc[sensitivity.summary["scenario"].eq("FULL_IDENTITY")]
        hard("full_identity_sensitivity_rows", len(identity_rows), "==", q["full_identity_sensitivity_rows_exact"], "A full identity control is mandatory")
        identity_min = float(identity_rows[["venue_rank_spearman", "admin_top3_jaccard_mean", "admin_top5_jaccard_mean", "region_best_venue_stability", "pareto_retention"]].min(axis=1).min()) if len(identity_rows) else -1.0
        hard("full_identity_sensitivity_exact", identity_min, ">=", 1.0 - 1e-12, "Identity ablation reproduces ranks, shortlist, and Pareto membership")
        stage4_evidence_count = int(config.get("stage4_started") is not False) + int(config["execution"].get("stage4_started") is not False) + int(not config["analysis_contract"].get("stage4_execution_forbidden", False)) + int(interface_forbidden > 0)
        hard("stage4_started_evidence_count", stage4_evidence_count, "==", 0, "Config, execution contract, and interface all show Stage 4 unstarted")
        hard("stage3_regression_tests_failed", int(tests_failed), "==", 0, "Stage 3 wrapper regression suite")
        hard("stage3_regression_tests_passed", int(tests_passed), ">=", q["stage3_tests_passed_min"], "At least one collected Stage 3 regression test must execute")
        hard(
            "stage3_regression_test_returncode",
            effective_test_result.returncode if effective_test_result.returncode is not None else -1,
            "==",
            0,
            "A completed pytest process with exit code zero is required",
        )
        hard("artifact_preinventory_invalid", int(preliminary_inventory.empty), "==", q["artifact_sha_mismatch_count_max"], "Nonempty hashed artifacts before final report")
        hard("core_figure_triplets", len(figure_result.artifact_manifest), "==", 12, "Four decision-critical PNG/SVG/HTML triplets")
        hard("extended_map_triplets", len(extended_figure_manifest), "==", 24, "Eight requested spatial map PNG/SVG/HTML triplets")
        advisory("minimum_scenario_rank_spearman", float(sensitivity.summary["venue_rank_spearman"].min()), ">=", 0.60, "Low value indicates catchment or weighting sensitivity; it does not tune the primary")
        advisory("field_validation_unknown_venues", int(candidates["field_validation_required"].sum()), "<=", len(candidates), "Unknown field feasibility is expected and must be validated")
        advisory("high_overlap_edges", int((edges["jaccard"] >= 0.8).sum()), "<=", len(edges), "High overlap is an optimizer trade-off, not a hard failure")
        advisory("fallback_venue_2_coverage", int(fallbacks["fallback_venue_2"].notna().sum()), ">=", len(fallbacks) - 4, "Second fallback is retained wherever a distinct physical alternative exists")
        advisory("exact_pareto_front_share", float(candidates["is_pareto_front"].mean()), "<=", 0.50, "The strict prompt-defined Pareto front remains primary even when wide")
        advisory("coarse_policy_pareto_front_share", float(candidates["is_coarse_policy_pareto_front"].mean()), "<=", q["coarse_policy_pareto_front_share_advisory_max"], "Three-bin ordinal front is an auxiliary no-false-precision screen, not a replacement")
        peak_rss_gib = _peak_rss_gib()
        advisory("peak_rss_gib", peak_rss_gib, "<=", float(config["execution"]["memory_limit_advisory_gib"]), "Peak process memory stays within the preregistered 32GB workstation budget")
        gate_result = build_stage3_quality_gate(gates, stage4_started=False)
        _atomic_csv(run_dir / "STAGE3_QUALITY_GATE.csv", gate_result.gates)

        best_admin = quadrants.set_index(ADMIN)
        need_top20 = set(need.nsmallest(20, "need_rank")[ADMIN].astype(str))
        exposure_top20 = set(
            quadrants.nlargest(20, "need_weighted_exposure")[ADMIN].astype(str)
        )
        high_exposure_admins = set(
            quadrants.loc[quadrants["high_exposure"], ADMIN].astype(str)
        )
        raw_primary = candidates["raw_elderly_exposure"]
        stats = {
            "frozen_mutation_count": len(mutations),
            "grid_rows": len(grid),
            "grid_admins": grid[ADMIN].nunique(),
            "grid_population_total": float(grid["elderly65_calibrated_population"].sum()),
            "grid_reconciliation_max_error": float(reconciliation["absolute_difference"].max()),
            "venue_input_rows": len(boundary_result.all_venues),
            "venue_eligible_rows": len(venue),
            "venue_excluded_rows": len(boundary_result.exclusions),
            "venue_sigungu_source_mismatch": int(mismatch.sum()),
            "matrix_count": len(family),
            "exposure_median": float(raw_primary.median()),
            "exposure_p75": float(raw_primary.quantile(0.75)),
            "exposure_p90": float(raw_primary.quantile(0.90)),
            "exposure_max": float(raw_primary.max()),
            "cross_admin_positive": int((candidates["cross_admin_exposure"] > 0).sum()),
            "cross_sigungu_positive": int((candidates["cross_sigungu_exposure"] > 0).sum()),
            "cross_admin_share_median": float(candidates["cross_admin_share"].median()),
            "need_top20_high_exposure_share": len(need_top20 & high_exposure_admins) / 20.0,
            "exposure_top20_need_overlap": len(exposure_top20 & need_top20) / 20.0,
            "bundle_rows": len(primary_bundle_long),
            "field_queue_rows": len(field_queue),
            "overlap_edges": len(edges),
            "overlap_component_max_size": int(overlap_components["component_size"].max()),
            "coverage_clusters": clusters["coverage_cluster_id"].nunique(),
            "coverage_cluster_max_size": int(clusters["cluster_size"].max()),
            "union_violation_count": int((~union["unique_not_above_naive"]).sum()),
            "pareto_front_rows": int(candidates["is_pareto_front"].sum()),
            "pareto_front_share": float(candidates["is_pareto_front"].mean()),
            "coarse_policy_pareto_front_rows": int(
                candidates["is_coarse_policy_pareto_front"].sum()
            ),
            "coarse_policy_pareto_front_share": float(
                candidates["is_coarse_policy_pareto_front"].mean()
            ),
            "discriminating_rho095_pairs": discriminating_rho095_pairs,
            "top3_rows": int(candidates["in_admin_top3"].sum()),
            "top5_rows": int(candidates["in_admin_top5"].sum()),
            "interface_rows": len(interface),
        }
        report_path = report_dir / "FINAL_REPORT.md"
        _write_report(
            report_path,
            run_id=run_id,
            stats=stats,
            gate=gate_result.gates,
            sensitivity=sensitivity.summary,
            quadrants=quadrants,
            radius_distribution=radius_distribution,
            venue_type_exposure=venue_type_exposure,
            bundle_sensitivity=bundle_sensitivity,
            baselines=baselines,
            boundary_detail=boundary_detail,
            calibration_detail=calibration_detail,
            pareto_granularity=pareto_bin_sensitivity,
            tie_breaker_sensitivity=tie_breaker_sensitivity,
            core_figure_manifest=figure_result.artifact_manifest,
            spatial_figure_manifest=extended_figure_manifest,
        )

        inventory = _artifact_inventory(root, [run_dir, report_dir])
        inventory_path = run_dir / "STAGE3_ARTIFACT_INVENTORY.csv"
        _atomic_csv(inventory_path, inventory)
        # Verify the final inventory independently after materialization.
        materialized = pd.read_csv(inventory_path)
        for item in materialized.itertuples(index=False):
            artifact = root / item.relative_path
            if artifact.stat().st_size != item.size_bytes or sha256_file(artifact) != item.sha256:
                raise RuntimeError(f"Final Stage 3 artifact inventory mismatch: {item.relative_path}")
        ended = _utc_now()
        for item in code_snapshot_manifest.itertuples(index=False):
            source = root / item.source_relative_path
            if sha256_file(source) != item.sha256:
                raise RuntimeError(f"Stage 3 source code changed during execution: {item.module}")
        for item in test_snapshot_manifest.itertuples(index=False):
            source = root / item.relative_path
            snapshot = root / item.snapshot_relative_path
            if (
                not source.is_file()
                or not snapshot.is_file()
                or sha256_file(source) != item.sha256
                or sha256_file(snapshot) != item.sha256
            ):
                raise RuntimeError(f"Stage 3 regression test changed during execution: {item.relative_path}")
        if _file_contract_sha256(root, _test_source_contract_files(root)) != str(
            effective_test_result.source_contract_sha256
        ):
            raise RuntimeError("Stage 3 tested source contract changed during execution")
        metadata = {
            "run_id": run_id,
            "started_at_utc": started.isoformat(),
            "ended_at_utc": ended.isoformat(),
            "runtime_seconds": time.perf_counter() - started_perf,
            "package_root": str(root),
            "python": sys.version,
            "platform": platform.platform(),
            "dependency_versions": _dependency_versions(),
            "jobs": resolved_jobs,
            "peak_rss_gib": peak_rss_gib,
            "config_relative_path": config_file.relative_to(root).as_posix(),
            "config_sha256": config_sha,
            "code_sha256": {
                name: sha256_file(path) if path.is_file() else "MISSING"
                for name, path in module_paths.items()
            },
            "input_sha256": {source: expected for source, (_, expected) in _input_contracts(config).items()},
            "grid_rows": len(grid),
            "venue_input_rows": len(boundary_result.all_venues),
            "venue_eligible_rows": len(venue),
            "matrix_dimensions": list(primary.matrix.shape),
            "matrix_nnz": {matrix_id: int(item.matrix.nnz) for matrix_id, item in family.items()},
            "venue_order_sha256": primary.venue_order_sha,
            "grid_order_sha256": primary.grid_order_sha,
            "tests_passed": int(tests_passed),
            "tests_failed": int(tests_failed),
            "tests_executed": effective_test_result.returncode is not None,
            "tests_returncode": effective_test_result.returncode,
            "test_file_count": effective_test_result.test_file_count,
            "test_files_sha256": effective_test_result.test_files_sha256,
            "test_source_contract_sha256": effective_test_result.source_contract_sha256,
            "hard_gate_passed": int(gate_result.hard_total - gate_result.hard_failed),
            "hard_gate_total": int(gate_result.hard_total),
            "hard_gate_failed": int(gate_result.hard_failed),
            "advisory_failed": int(gate_result.advisory_failed),
            "artifact_count": len(inventory),
            "artifact_inventory_sha256": sha256_file(inventory_path),
            "artifact_sha_mismatch_count": 0,
            "stage1_frozen": len(mutations.loc[mutations["scope"].astype(str).str.contains("stage1")]) == 0 if not mutations.empty else True,
            "stage2a_frozen": len(mutations.loc[mutations["scope"].astype(str).str.contains("stage2")]) == 0 if not mutations.empty else True,
            "stage2b_frozen": len(mutations.loc[mutations["scope"].astype(str).str.contains("stage2b")]) == 0 if not mutations.empty else True,
            "stage3_complete": bool(gate_result.hard_pass),
            "run_mode": "promotion_candidate" if promote else "diagnostic",
            "promotion_requested": bool(promote),
            "promotion_eligible": bool(promote and gate_result.hard_pass),
            "promotion_commit_marker": "outputs/model_v1/07_stage3/CURRENT_STAGE3_RUN.json_is_authoritative",
            "tests_skipped": effective_test_result.returncode is None,
            "stage4_started": False,
            "temporal_primary_selector_count": temporal_primary_selector_count,
            "report_relative_path": report_path.relative_to(root).as_posix(),
        }
        metadata_path = run_dir / "STAGE3_RUN_METADATA.json"
        _atomic_json(metadata_path, metadata)

        if promote and gate_result.hard_pass:
            # The CURRENT pointer is the commit marker and is always written
            # last.  Consumers must resolve the run-scoped report/interface
            # through this pointer; the flat report is only a convenience alias.
            current_report = _resolve(root, paths["current_report"])
            _atomic_text(current_report, report_path.read_text(encoding="utf-8"))
            if sha256_file(current_report) != sha256_file(report_path):
                raise RuntimeError(
                    "Stage 3 flat report alias must be byte-identical to the canonical report"
                )
            # The run itself is complete before promotion.  CURRENT is the
            # final, sole promotion commit marker.
            running.unlink()
            _atomic_json(
                output_root / "CURRENT_STAGE3_RUN.json",
                {
                    "run_id": run_id,
                    "metadata_relative_path": metadata_path.relative_to(root).as_posix(),
                    "interface_relative_path": (
                        run_dir / "08_interface" / "stage3_to_stage4_interface.parquet"
                    ).relative_to(root).as_posix(),
                    "metadata_sha256": sha256_file(metadata_path),
                    "artifact_inventory_sha256": sha256_file(inventory_path),
                    "interface_sha256": sha256_file(
                        run_dir / "08_interface" / "stage3_to_stage4_interface.parquet"
                    ),
                    "report_sha256": sha256_file(report_path),
                    "stage3_complete": True,
                    "stage4_started": False,
                },
            )
        else:
            # Diagnostic or non-passing runs have no promotion pointer.
            running.unlink()

        return Stage3PipelineResult(
            run_id=run_id,
            run_dir=run_dir,
            report_dir=report_dir,
            metadata_path=metadata_path,
            hard_pass=bool(gate_result.hard_pass),
            stage3_complete=bool(gate_result.hard_pass),
        )


def _run_stage3_tests(root: Path) -> Stage3TestResult:
    test_files = sorted((root / "tests").glob("test_*.py"))
    if not test_files:
        raise RuntimeError("No repository regression tests were collected")
    source_contract_sha = _file_contract_sha256(
        root, _test_source_contract_files(root)
    )
    digest = sha256()
    for path in test_files:
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(path)))
    command = [sys.executable, "-m", "pytest", "-q", "tests"]
    environment = os.environ.copy()
    source_path = str((root / "src").resolve())
    prior_pythonpath = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = os.pathsep.join(
        value for value in (source_path, prior_pythonpath) if value
    )
    started = time.perf_counter()
    result = subprocess.run(
        command,
        cwd=root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    passed_match = re.search(r"(\d+) passed", result.stdout)
    failed_match = re.search(r"(\d+) failed", result.stdout)
    passed = int(passed_match.group(1)) if passed_match else 0
    failed = int(failed_match.group(1)) if failed_match else (0 if result.returncode == 0 else 1)
    effective_returncode = int(result.returncode)
    source_contract_after = _file_contract_sha256(
        root, _test_source_contract_files(root)
    )
    stderr = result.stderr
    if source_contract_after != source_contract_sha:
        effective_returncode = 3
        failed = max(failed, 1)
        stderr += "\nStage 3 source/config contract changed while pytest was running.\n"
    if effective_returncode != 0 and failed == 0:
        failed = 1
    return Stage3TestResult(
        passed=passed,
        failed=failed,
        returncode=effective_returncode,
        command=tuple(command),
        duration_seconds=time.perf_counter() - started,
        test_file_count=len(test_files),
        test_files_sha256=digest.hexdigest(),
        stdout=result.stdout,
        stderr=stderr,
        source_contract_sha256=source_contract_sha,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", default=".")
    parser.add_argument("--config", default="configs/model_v1/stage3_spatial.yaml")
    parser.add_argument("--jobs", type=int, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--promote", action="store_true")
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args(argv)
    root = Path(args.package_root).resolve()
    test_result = (
        Stage3TestResult(0, 0, None, (), 0.0, 0, "NOT_EXECUTED", "", "")
        if args.skip_tests
        else _run_stage3_tests(root)
    )
    tests_passed, tests_failed = test_result.passed, test_result.failed
    if tests_failed:
        return 2
    resolved_run_id = args.run_id or (
        f"stage3_{_utc_now().strftime('%Y%m%dT%H%M%SZ')}_{sha256_file(Path(__file__))[:12]}"
    )
    if not _RUN_ID.fullmatch(resolved_run_id):
        raise ValueError(f"Invalid run_id: {resolved_run_id}")
    try:
        result = run_stage3(
            root,
            config_path=args.config,
            jobs=args.jobs,
            promote=args.promote,
            run_id=resolved_run_id,
            tests_passed=tests_passed,
            tests_failed=tests_failed,
            test_result=test_result,
        )
    except (Exception, KeyboardInterrupt) as exc:
        # Preserve partial artifacts for diagnosis, but never leave them looking
        # like a finished generation.  The writer lock is released by the
        # context manager before this marker transition.
        try:
            failure_config = _load_yaml(_resolve(root, args.config))
            failure_output = _require_below(
                root,
                _resolve(root, failure_config["paths"]["output_root"]),
                "output root",
            )
            failure_run_dir = _require_below(
                root,
                failure_output / "runs" / resolved_run_id,
                "failure run directory",
            )
            running_marker = failure_run_dir / ".RUNNING"
            if running_marker.exists():
                failed_marker = failure_run_dir / ".FAILED"
                os.replace(running_marker, failed_marker)
                _atomic_json(
                    failure_run_dir / "STAGE3_RUN_FAILURE.json",
                    {
                        "run_id": resolved_run_id,
                        "failed_at_utc": _utc_now().isoformat(),
                        "exception_type": type(exc).__name__,
                        "exception_message": str(exc),
                        "traceback": traceback.format_exc(),
                        "stage3_complete": False,
                        "stage4_started": False,
                    },
                )
        except Exception:
            pass
        raise
    print(f"Stage 3 run: {result.run_id}")
    print(f"Metadata: {result.metadata_path}")
    print(f"Hard gate: {'PASS' if result.hard_pass else 'FAIL'}")
    return 0 if result.hard_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
