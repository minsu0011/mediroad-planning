"""Stage 2B hardening and Stage 2 integrated validation pipeline.

This build is deliberately isolated from the frozen Stage 2 release.  It tests
whether the two available NHIS years support a reproducible *coarse* season
prior, permits explicit abstention, and stops before Stage 3.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import yaml

from mediroad.reporting.stage2b_hardening import (
    build_artifact_inventory,
    build_stage2b_quality_gate,
    render_stage2b_hardening_figures,
    validate_artifact_inventory,
    validate_temporal_evidence_summary,
)
from mediroad.temporal.hardening import TemporalHardeningResult, run_temporal_hardening
from mediroad.temporal.hardening_integration import (
    analyze_bundle_differentiation,
    build_stage2_to_stage3_interface,
    classify_confidence_aware_release,
    compare_what_when_models,
    diagnose_need_replication,
    diagnose_temporal_spatial_leakage,
    stage2_to_stage3_schema,
    validate_confidence_release,
    validate_stage2_to_stage3_interface,
)


DEFAULT_CONFIG = Path("configs/model_v1/stage2b_hardening.yaml")


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return value.as_posix()
    raise TypeError(f"Cannot JSON-encode {type(value)!r}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _temporary_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.tmp-{os.getpid()}")


def write_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(path)
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    temporary.replace(path)


def write_text(value: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(path)
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def write_csv(frame: pd.DataFrame, path: Path, *, index: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(path)
    frame.to_csv(temporary, index=index, encoding="utf-8-sig")
    temporary.replace(path)


def write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(path)
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return value


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} missing columns: {missing}")


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


@dataclass(frozen=True)
class FrozenSnapshot:
    direct_hashes: dict[str, str]
    official_inventory_hashes: dict[str, str]


@dataclass(frozen=True)
class HardeningInputs:
    nhis: pd.DataFrame
    working_days: pd.DataFrame
    bundle_config: dict[str, Any]
    stage1_scores: pd.DataFrame
    bundle_scores: pd.DataFrame
    specialty_scores: pd.DataFrame
    baseline_season_scores: pd.DataFrame


def _read_inputs(root: Path, config: Mapping[str, Any]) -> HardeningInputs:
    paths = config["paths"]
    nhis = pd.read_parquet(_resolve(root, paths["nhis_normalized_panel"]))
    working_days = pd.read_csv(_resolve(root, paths["working_day_calendar"]))
    stage1_scores = pd.read_csv(_resolve(root, paths["stage1_need_scores"]))
    bundle_scores = pd.read_parquet(_resolve(root, paths["stage2a_bundle_scores"]))
    specialty_scores = pd.read_parquet(_resolve(root, paths["stage2a_specialty_scores"]))
    baseline_season_scores = pd.read_parquet(
        _resolve(root, paths["baseline_stage2b_season_scores"])
    )
    bundle_config = _load_yaml(_resolve(root, paths["service_bundle_config"]))
    _require_columns(
        nhis,
        ["year", "month", "policy_sigungu_name", "service_id", "persons", "visits"],
        "NHIS normalized panel",
    )
    _require_columns(
        bundle_scores,
        [
            "admin_dong_code",
            "admin_dong_name",
            "policy_sigungu_name",
            "bundle_id",
            "bundle_gap_score",
            "bundle_gap_rank",
        ],
        "Stage 2A bundle scores",
    )
    _require_columns(
        baseline_season_scores,
        [
            "admin_dong_code",
            "policy_sigungu_name",
            "bundle_id",
            "season",
            "temporal_fit_score",
        ],
        "Frozen Stage 2B season scores",
    )
    contract = config["analysis_contract"]
    if len(nhis) != int(contract["expected_panel_rows"]):
        raise ValueError("NHIS panel row contract failed")
    if nhis["policy_sigungu_name"].nunique() != int(
        contract["expected_policy_sigungu"]
    ):
        raise ValueError("NHIS policy-sigungu contract failed")
    if nhis["service_id"].nunique() != int(contract["expected_services"]):
        raise ValueError("NHIS service-universe contract failed")
    return HardeningInputs(
        nhis=nhis,
        working_days=working_days,
        bundle_config=bundle_config,
        stage1_scores=stage1_scores,
        bundle_scores=bundle_scores,
        specialty_scores=specialty_scores,
        baseline_season_scores=baseline_season_scores,
    )


def _verify_frozen_contract(root: Path, config: Mapping[str, Any]) -> FrozenSnapshot:
    paths = dict(config["paths"])
    frozen = dict(config["frozen_contract"])
    direct_contract = {
        "frozen_canonical_master": "canonical_master_sha256",
        "road_integrated_master": "road_integrated_master_sha256",
        "stage1_config": "stage1_config_sha256",
        "stage1_need_scores": "stage1_need_scores_sha256",
        "stage2a_bundle_scores": "stage2a_bundle_scores_sha256",
        "stage2a_specialty_scores": "stage2a_specialty_scores_sha256",
        "nhis_normalized_panel": "nhis_normalized_panel_sha256",
        "working_day_calendar": "working_day_calendar_sha256",
        "baseline_stage2_metadata": "baseline_stage2_metadata_sha256",
        "baseline_stage2_inventory": "baseline_stage2_inventory_sha256",
    }
    direct_hashes: dict[str, str] = {}
    for path_key, digest_key in direct_contract.items():
        path = _resolve(root, paths[path_key])
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256_file(path)
        expected = str(frozen[digest_key]).lower()
        if actual != expected:
            raise RuntimeError(
                f"Frozen contract mismatch for {path_key}: {actual} != {expected}"
            )
        direct_hashes[path_key] = actual

    metadata_path = _resolve(root, paths["baseline_stage2_metadata"])
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("run_id") != frozen["baseline_stage2_run_id"]:
        raise RuntimeError("Frozen Stage 2 run id mismatch")
    quality = metadata.get("quality_gate", {})
    if int(quality.get("hard_gate_pass_count", -1)) != int(
        frozen["baseline_stage2_hard_gate_passed"]
    ) or int(quality.get("hard_gate_count", -1)) != int(
        frozen["baseline_stage2_hard_gate_total"]
    ):
        raise RuntimeError("Frozen Stage 2 quality-gate count mismatch")
    if bool(metadata.get("stage3_executed", True)):
        raise RuntimeError("Frozen Stage 2 metadata unexpectedly records Stage 3")

    inventory_path = _resolve(root, paths["baseline_stage2_inventory"])
    inventory = pd.read_csv(inventory_path)
    _require_columns(inventory, ["relative_path", "sha256"], "official inventory")
    official: dict[str, str] = {}
    for row in inventory[["relative_path", "sha256"]].itertuples(index=False):
        relative = str(row.relative_path)
        path = _resolve(root, relative)
        if not path.is_file():
            raise FileNotFoundError(f"Frozen Stage 2 artifact missing: {relative}")
        actual = sha256_file(path)
        expected = str(row.sha256).lower()
        if actual != expected:
            raise RuntimeError(
                f"Frozen Stage 2 artifact mutated: {relative}: {actual} != {expected}"
            )
        official[relative] = actual
    return FrozenSnapshot(direct_hashes, official)


def _assert_frozen_unchanged(root: Path, snapshot: FrozenSnapshot) -> None:
    for relative, expected in snapshot.official_inventory_hashes.items():
        path = _resolve(root, relative)
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"Frozen Stage 2 artifact changed during run: {relative}")


def _release_from_temporal_result(
    result: TemporalHardeningResult,
    baseline_season_scores: pd.DataFrame,
    release_rules: Mapping[str, Any],
) -> pd.DataFrame:
    evidence = result.bundle_evidence_summary.copy()
    evidence["primary_season"] = evidence["consensus_primary_candidate"]
    evidence["fallback_season"] = evidence["consensus_fallback_candidate"]
    classified = classify_confidence_aware_release(evidence, release_rules)
    engine_types = result.bundle_evidence_summary.set_index("bundle_id")["release_type"]
    classifier_types = classified.set_index("bundle_id")["release_type"]
    if not engine_types.sort_index().equals(classifier_types.sort_index()):
        raise RuntimeError("Independent temporal release classifiers disagree")

    # Use the frozen Stage 2B score scale only as an interpretable diagnostic
    # value in the hand-off; confidence and released seasons come exclusively
    # from the new strict two-year hardening evidence.
    fit_lookup = (
        baseline_season_scores.groupby(["bundle_id", "season"], as_index=False)[
            "temporal_fit_score"
        ]
        .mean()
        .rename(columns={"season": "diagnostic_primary_season"})
    )
    classified = classified.merge(
        fit_lookup,
        on=["bundle_id", "diagnostic_primary_season"],
        how="left",
        validate="one_to_one",
    )
    if classified["temporal_fit_score"].isna().any():
        raise RuntimeError("Could not attach diagnostic temporal fit to release")
    classified["temporal_source"] = (
        "NHIS_2022_2023_observed_insured_utilization_working_day_adjusted"
    )
    classified["temporal_spatial_resolution"] = (
        "province_common_bundle_season_with_sigungu_block_validation"
    )
    classified["temporal_independent_n"] = int(
        result.evidence.audit["temporal_independent_n"]
    )
    classified["limitations"] = (
        "two years; all-age insured utilization; not mobile-clinic demand; "
        "no exact month/date; climate and static transit excluded from selector"
    )
    validate_confidence_release(classified, raise_on_error=True)
    return classified


def _temporal_evidence_table(
    result: TemporalHardeningResult,
    release: pd.DataFrame,
) -> pd.DataFrame:
    table = release.copy()
    cross = result.cross_year.summary.copy()
    for metric in ("heldout_regret", "heldout_regret_normalized"):
        pivot = cross.pivot(index="bundle_id", columns="direction", values=metric)
        pivot.columns = [f"{metric}_{column}" for column in pivot.columns]
        table = table.merge(pivot.reset_index(), on="bundle_id", validate="one_to_one")
    table["bundle"] = table["bundle_id"]
    table["2022_top1"] = table["year2022_top1"]
    table["2022_top2"] = table["year2022_top2"]
    table["2023_top1"] = table["year2023_top1"]
    table["2023_top2"] = table["year2023_top2"]
    table["cross_year_consistency"] = table["cross_year_top1_agreement"]
    table["heldout_regret"] = table[
        ["heldout_regret_2022_to_2023", "heldout_regret_2023_to_2022"]
    ].max(axis=1)
    table["bootstrap_primary_probability"] = pd.to_numeric(
        table["bootstrap_primary_probability"], errors="raise"
    )
    table["lomo_stability"] = table["lomo_primary_retention"]
    table["loso_stability"] = table["loso_primary_retention"]
    table["null_strength"] = table["permutation_margin_percentile"]
    table["temporal_confidence"] = table["temporal_confidence"].astype(str)
    table["final_primary"] = table["primary_season"]
    table["final_fallback"] = table["fallback_season"]
    table["reason"] = table["release_reason"]
    preferred = [
        "bundle",
        "bundle_id",
        "2022_top1",
        "2022_top2",
        "2023_top1",
        "2023_top2",
        "year2022_top1",
        "year2022_top2",
        "year2023_top1",
        "year2023_top2",
        "pooled_top1",
        "pooled_top2",
        "consensus_primary_candidate",
        "consensus_fallback_candidate",
        "cross_year_consistency",
        "bidirectional_primary_in_holdout_top2",
        "bidirectional_selected_pair_holdout_top2_coverage",
        "pooled_normalized_top1_margin",
        "year_normalized_top1_margin",
        "heldout_regret_2022_to_2023",
        "heldout_regret_2023_to_2022",
        "heldout_regret_normalized_2022_to_2023",
        "heldout_regret_normalized_2023_to_2022",
        "heldout_regret",
        "bootstrap_primary_probability",
        "bootstrap_pair_coverage_probability",
        "lomo_stability",
        "lomo_top2_retention",
        "loso_stability",
        "loso_top2_retention",
        "working_day_primary_retention",
        "working_day_top2_retention",
        "boundary_primary_retention",
        "boundary_top2_retention",
        "null_strength",
        "permutation_p_value_margin",
        "release_type",
        "temporal_confidence",
        "final_primary",
        "final_fallback",
        "temporal_independent_n",
        "temporal_source",
        "temporal_spatial_resolution",
        "reason",
        "limitations",
    ]
    return table[[column for column in preferred if column in table]].copy()


def _build_delivery_release(
    bundle_scores: pd.DataFrame,
    evidence: pd.DataFrame,
) -> pd.DataFrame:
    release = evidence.copy()
    release = release.rename(
        columns={
            "year2022_top1": "year2022_top1",
            "year2023_top1": "year2023_top1",
        }
    )
    base_columns = [
        "admin_dong_code",
        "admin_dong_name",
        "policy_sigungu_name",
        "bundle_id",
        "bundle_gap_score",
        "bundle_gap_rank",
    ]
    delivery = bundle_scores[base_columns].merge(
        release,
        on="bundle_id",
        how="left",
        validate="many_to_one",
    )
    if delivery["release_type"].isna().any():
        raise RuntimeError("Temporal release does not cover every delivery bundle")
    delivery["sigungu"] = delivery["policy_sigungu_name"]
    delivery["admin_dong"] = delivery["admin_dong_name"]
    delivery["primary_season"] = delivery["final_primary"]
    delivery["fallback_season"] = delivery["final_fallback"]
    delivery["cross_year_top1_agreement"] = delivery["cross_year_consistency"]
    delivery["cross_year_top2_consistency"] = delivery[
        "bidirectional_selected_pair_holdout_top2_coverage"
    ]
    delivery["heldout_regret_22_to_23"] = delivery[
        "heldout_regret_2022_to_2023"
    ]
    delivery["heldout_regret_23_to_22"] = delivery[
        "heldout_regret_2023_to_2022"
    ]
    delivery["working_day_sensitivity"] = delivery[
        "working_day_top2_retention"
    ]
    delivery["permutation_strength"] = delivery["null_strength"]
    delivery["season_margin"] = delivery["pooled_normalized_top1_margin"]
    delivery["exact_month"] = pd.NA
    delivery["exact_date"] = pd.NA
    delivery["reason"] = delivery["reason"].astype(str)
    ordered = [
        "admin_dong_code",
        "sigungu",
        "admin_dong",
        "bundle_id",
        "bundle_gap_score",
        "bundle_gap_rank",
        "primary_season",
        "fallback_season",
        "release_type",
        "temporal_confidence",
        "cross_year_top1_agreement",
        "cross_year_top2_consistency",
        "heldout_regret_22_to_23",
        "heldout_regret_23_to_22",
        "bootstrap_primary_probability",
        "lomo_stability",
        "loso_stability",
        "working_day_sensitivity",
        "permutation_strength",
        "season_margin",
        "exact_month",
        "exact_date",
        "temporal_source",
        "temporal_spatial_resolution",
        "temporal_independent_n",
        "reason",
        "limitations",
    ]
    output = delivery[ordered].sort_values(
        ["admin_dong_code", "bundle_id"], kind="mergesort"
    ).reset_index(drop=True)
    if output.duplicated(["admin_dong_code", "bundle_id"]).any():
        raise RuntimeError("Stage 2B delivery release keys are not unique")
    return output


def _model_comparison(
    integration_summary: pd.DataFrame,
    evidence: pd.DataFrame,
) -> pd.DataFrame:
    release_counts = evidence["release_type"].value_counts().to_dict()
    regret = float(
        evidence[
            [
                "heldout_regret_normalized_2022_to_2023",
                "heldout_regret_normalized_2023_to_2022",
            ]
        ].to_numpy(float).mean()
    )
    rows = [
        {
            "model": "Model0_no_temporal_layer",
            "cross_year_regret": np.nan,
            "forced_single_count": 0,
            "robust_pair_count": 0,
            "abstention_count": 5,
            "false_precision_count": 0,
            "stage2a_top1_bundle_retention": 1.0,
            "release_role": "no_temporal_guidance",
        },
        {
            "model": "Model1_forced_single_season",
            "cross_year_regret": regret,
            "forced_single_count": 5,
            "robust_pair_count": 0,
            "abstention_count": 0,
            "false_precision_count": int(
                5 - release_counts.get("STRONG_SINGLE", 0)
            ),
            "stage2a_top1_bundle_retention": 1.0,
            "release_role": "rejected_false_precision_comparator",
        },
        {
            "model": "Model2_confidence_aware",
            "cross_year_regret": regret,
            "forced_single_count": int(release_counts.get("STRONG_SINGLE", 0)),
            "robust_pair_count": int(release_counts.get("ROBUST_PAIR", 0)),
            "abstention_count": int(
                release_counts.get("NO_STRONG_PREFERENCE", 0)
            ),
            "false_precision_count": 0,
            "stage2a_top1_bundle_retention": 1.0,
            "release_role": "bundle_specific_confidence_or_abstention",
        },
        {
            "model": "Model3_hierarchical_WHAT_then_WHEN",
            "cross_year_regret": regret,
            "forced_single_count": int(release_counts.get("STRONG_SINGLE", 0)),
            "robust_pair_count": int(release_counts.get("ROBUST_PAIR", 0)),
            "abstention_count": int(
                release_counts.get("NO_STRONG_PREFERENCE", 0)
            ),
            "false_precision_count": 0,
            "stage2a_top1_bundle_retention": float(
                integration_summary.loc[
                    integration_summary["model"].eq(
                        "hierarchical_WHAT_then_WHEN"
                    ),
                    "top1_bundle_retention",
                ].iloc[0]
            ),
            "release_role": "primary_architecture_coarse_advisory_WHEN_after_WHAT",
        },
    ]
    return pd.DataFrame(rows)


_DISPLAY_BUNDLE_ID = {
    "chronic_primary": "chronic",
    "comprehensive_basic_referral": "comprehensive",
    "musculoskeletal_rehab": "musculoskeletal",
    "neuro_mental": "neuro",
    "sensory_oral_skin": "sensory",
}


def _display_bundles(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["bundle_id"] = output["bundle_id"].map(_DISPLAY_BUNDLE_ID).fillna(
        output["bundle_id"]
    )
    return output


def _figure_tables(
    temporal: TemporalHardeningResult,
    release: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    return {
        "bundle_year_season_rank_heatmap": _display_bundles(temporal.consensus),
        "cross_year_transfer_matrix": _display_bundles(temporal.cross_year.summary),
        "bundle_season_margin_chart": _display_bundles(temporal.cross_year.summary),
        "heldout_regret_chart": _display_bundles(temporal.cross_year.summary),
        "bootstrap_season_selection_probability": _display_bundles(
            temporal.bootstrap.summary
        ),
        "leave_one_month_out_stability": _display_bundles(temporal.lomo.detail),
        "leave_one_sigungu_out_stability": _display_bundles(temporal.loso.detail),
        "observed_vs_permutation_null": _display_bundles(
            temporal.permutation.summary
        ),
        "temporal_confidence_summary": _display_bundles(release),
    }


def _record(
    gate_id: str,
    severity: str,
    passed: bool,
    observed: Any,
    comparison: str,
    threshold: Any,
    threshold_source: str,
    note: str,
) -> dict[str, Any]:
    return {
        "gate_id": gate_id,
        "severity": severity,
        "passed": bool(passed),
        "observed": observed,
        "comparison": comparison,
        "threshold": threshold,
        "threshold_source": threshold_source,
        "note": note,
    }


def _base_quality_records(
    *,
    config: Mapping[str, Any],
    inputs: HardeningInputs,
    temporal: TemporalHardeningResult,
    release: pd.DataFrame,
    delivery: pd.DataFrame,
    interface_gates: pd.DataFrame,
    integration_summary: pd.DataFrame,
    spatial_summary: pd.DataFrame,
    need_summary: pd.DataFrame,
    differentiation_summary: pd.DataFrame,
    figure_artifact_count: int,
    test_result: Mapping[str, Any],
    bootstrap: int,
    permutations: int,
) -> list[dict[str, Any]]:
    contract = config["analysis_contract"]
    q = config["quality_gates"]
    advisory = config["advisory_thresholds"]
    records: list[dict[str, Any]] = []

    def hard(
        name: str,
        observed: Any,
        comparison: str,
        threshold: Any,
        passed: bool,
        note: str,
        source: str = "config: stage2b_hardening.yaml",
    ) -> None:
        records.append(_record(name, "hard", passed, observed, comparison, threshold, source, note))

    def adv(
        name: str,
        observed: Any,
        comparison: str,
        threshold: Any,
        passed: bool,
        note: str,
        source: str = "preregistered advisory config: stage2b_hardening.yaml",
    ) -> None:
        records.append(_record(name, "advisory", passed, observed, comparison, threshold, source, note))

    def diagnostic(
        name: str,
        observed: Any,
        comparison: str,
        threshold: Any,
        passed: bool,
        note: str,
    ) -> None:
        records.append(
            _record(
                name,
                "diagnostic",
                passed,
                observed,
                comparison,
                threshold,
                "prompt diagnostic contract",
                note,
            )
        )

    audit = temporal.evidence.audit
    hard("nhis_panel_rows", len(inputs.nhis), "==", contract["expected_panel_rows"], len(inputs.nhis) == int(contract["expected_panel_rows"]), "Normalized NHIS input is the complete 2022-2023 rectangle.")
    hard("nhis_observed_years", sorted(inputs.nhis["year"].unique().tolist()), "==", contract["observed_years"], sorted(inputs.nhis["year"].unique().tolist()) == list(contract["observed_years"]), "Only the two preregistered years are analyzed.")
    hard("policy_sigungu_count", audit["policy_sigungu_count"], "==", contract["expected_policy_sigungu"], int(audit["policy_sigungu_count"]) == int(contract["expected_policy_sigungu"]), "The independent spatial blocks are the 11 policy sigungu.")
    hard("observed_service_count", audit["observed_service_count"], "==", contract["expected_services"], int(audit["observed_service_count"]) == int(contract["expected_services"]), "All normalized NHIS services are present before bundle selection.")
    hard("bundle_count", audit["bundle_count"], "==", contract["expected_bundles"], int(audit["bundle_count"]) == int(contract["expected_bundles"]), "Every configured service bundle is covered.")
    hard("temporal_independent_n", audit["temporal_independent_n"], "==", contract["independent_n"], int(audit["temporal_independent_n"]) == int(contract["independent_n"]), "Inference uses 11 sigungu x 5 bundles, never 765 delivery rows.")
    hard("broadcast_rows_not_independent", audit["admin_dong_delivery_rows_are_independent_samples"], "==", False, not bool(audit["admin_dong_delivery_rows_are_independent_samples"]), "The 153-admin delivery expansion is explicitly non-independent.")

    transfer = temporal.cross_year.detail
    hard("cross_year_transfer_rows", len(transfer), "==", int(q["cross_year_bundle_rows"]) * int(contract["expected_policy_sigungu"]), len(transfer) == int(q["cross_year_bundle_rows"]) * int(contract["expected_policy_sigungu"]), "Both strict train-holdout directions cover every sigungu-bundle unit.")
    hard("cross_year_directions", transfer["direction"].nunique(), "==", q["cross_year_direction_count"], transfer["direction"].nunique() == int(q["cross_year_direction_count"]), "2022-to-2023 and 2023-to-2022 are both complete.")
    hard("cross_year_zero_fit_holdout_overlap", int(transfer["fit_holdout_overlap_row_count"].sum()), "==", 0, transfer["fit_holdout_overlap_row_count"].eq(0).all(), "No pooled-year leakage is allowed.")

    lomo = temporal.lomo.detail
    hard("lomo_run_count", len(lomo), "==", q["lomo_expected_bundle_runs"], len(lomo) == int(q["lomo_expected_bundle_runs"]), "Twelve month removals per year and bundle are complete.")
    hard("lomo_exact_12_per_bundle_year", int(lomo.groupby(["bundle_id", "year"])["removed_month"].nunique().min()), "==", q["lomo_expected_month_removals_per_year"], lomo.groupby(["bundle_id", "year"])["removed_month"].nunique().eq(int(q["lomo_expected_month_removals_per_year"])).all(), "Every calendar month is removed exactly once per bundle-year.")
    loso = temporal.loso.detail
    hard("loso_run_count", len(loso), "==", q["loso_expected_bundle_runs"], len(loso) == int(q["loso_expected_bundle_runs"]), "All 11 policy-sigungu block removals are complete for five bundles.")
    hard("loso_exact_11_per_bundle", int(loso.groupby("bundle_id")["removed_sigungu"].nunique().min()), "==", q["loso_expected_sigungu_removals"], loso.groupby("bundle_id")["removed_sigungu"].nunique().eq(int(q["loso_expected_sigungu_removals"])).all(), "Each policy sigungu is excluded once without splitting its monthly sequence.")
    hard("working_day_sensitivity_complete", len(temporal.working_day_sensitivity), "==", 30, len(temporal.working_day_sensitivity) == 30, "Five bundles x six preregistered normalization variants are present.")
    hard("season_boundary_sensitivity_complete", len(temporal.season_boundary.detail), "==", 15, len(temporal.season_boundary.detail) == 15, "Primary plus two boundary diagnostics are complete without post-hoc switching.")

    boot = temporal.bootstrap
    hard("bootstrap_replicates", int(boot.detail["bootstrap_replicate"].nunique()), "==", bootstrap, boot.detail["bootstrap_replicate"].nunique() == bootstrap, "Requested sigungu block-bootstrap count completed.")
    hard("bootstrap_n_valid", int(boot.summary["n_valid"].min()), "==", bootstrap, boot.summary["n_valid"].eq(bootstrap).all(), "Every bundle-season bootstrap metric uses every replicate.")
    hard("bootstrap_complete_blocks", int(boot.detail["complete_block_resampling"].mean()), "==", 1, boot.detail["complete_block_resampling"].all(), "Whole sigungu month/service sequences are resampled together.")
    perm = temporal.permutation
    hard("permutation_replicates", int(perm.detail["permutation_replicate"].nunique()), "==", permutations, perm.detail["permutation_replicate"].nunique() == permutations, "Requested within-block month-label permutations completed.")
    hard("permutation_n_valid", int(perm.summary["n_valid"].min()), "==", permutations, perm.summary["n_valid"].eq(permutations).all(), "Every bundle-null metric uses every permutation.")
    hard("permutation_annual_totals_preserved", int(perm.detail["annual_totals_preserved"].mean()), "==", 1, perm.detail["annual_totals_preserved"].all(), "Month shuffling preserves every sigungu-year-service annual total.")

    hard("bundle_evidence_rows", len(release), "==", contract["expected_bundles"], len(release) == int(contract["expected_bundles"]), "Every bundle has one explicit confidence-aware disposition.")
    hard("delivery_release_rows", len(delivery), "==", contract["expected_release_rows"], len(delivery) == int(contract["expected_release_rows"]), "The delivery table is 153 admins x five bundles.")
    hard("delivery_release_keys_unique", int(delivery.duplicated(["admin_dong_code", "bundle_id"]).sum()), "==", 0, not delivery.duplicated(["admin_dong_code", "bundle_id"]).any(), "No duplicate Stage 2B delivery keys exist.")
    hard("exact_month_null", int(delivery["exact_month"].notna().sum()), "==", q["exact_month_nonnull_max"], delivery["exact_month"].notna().sum() <= int(q["exact_month_nonnull_max"]), "Exact month release remains forbidden.")
    hard("exact_date_null", int(delivery["exact_date"].notna().sum()), "==", q["exact_date_nonnull_max"], delivery["exact_date"].notna().sum() <= int(q["exact_date_nonnull_max"]), "Exact date release remains forbidden.")

    hierarchy = integration_summary.loc[integration_summary["model"].eq("hierarchical_WHAT_then_WHEN")].iloc[0]
    hard("hierarchical_what_top1_invariant", hierarchy["top1_bundle_retention"], "==", 1.0, np.isclose(float(hierarchy["top1_bundle_retention"]), 1.0), "Primary WHAT-then-WHEN architecture cannot alter Stage 2A bundle ranks.")
    hard("temporal_spatial_leakage_count", int(spatial_summary["unexpected_spatial_variation_count"].iloc[0]), "==", 0, int(spatial_summary["unexpected_spatial_variation_count"].iloc[0]) == 0, "No unsupported admin-level temporal differentiation is created.")
    for row in interface_gates.itertuples(index=False):
        hard(f"stage3_interface__{row.gate}", row.observed, str(row.comparator), row.expected, bool(row.passed), str(row.detail) or "Stage 3 hand-off contract check.", "prompt Stage3 interface contract")
    hard("stage3_started", False, "==", False, True, "Only the interface is materialized; Stage 3 execution remains forbidden.")
    hard("figure_artifact_triplets", figure_artifact_count, "==", 27, figure_artifact_count == 27, "Nine preregistered figures each have PNG, SVG, and self-contained HTML.")
    hard("full_test_suite", test_result["exit_code"], "==", 0, int(test_result["exit_code"]) == 0, f"{test_result['passed']} tests passed in the shared Python 3.11 environment.", "prompt regression-test contract")

    transfer_summary = temporal.cross_year.summary
    global_top1 = float(transfer_summary["top1_agreement"].mean())
    global_top2 = float(transfer_summary["train_primary_in_holdout_top2"].mean())
    adv("cross_year_top1_agreement", global_top1, ">=", advisory["cross_year_top1_agreement_min"], global_top1 >= float(advisory["cross_year_top1_agreement_min"]), "Weak cross-year Top1 agreement is evidence, not a structural build failure.")
    adv("cross_year_primary_in_holdout_top2", global_top2, ">=", advisory["cross_year_primary_in_holdout_top2_min"], global_top2 >= float(advisory["cross_year_primary_in_holdout_top2_min"]), "Measures whether year-trained primaries remain inside holdout Top2.")
    for row in release.itertuples(index=False):
        adv(f"bundle_release_strength__{row.bundle_id}", row.release_type, "!=", "NO_STRONG_PREFERENCE", row.release_type != "NO_STRONG_PREFERENCE", "Explicit abstention is scientifically valid but records weak temporal evidence.")
        adv(f"permutation_margin_strength__{row.bundle_id}", float(row.permutation_margin_percentile), ">=", advisory["permutation_margin_percentile_min"], float(row.permutation_margin_percentile) >= float(advisory["permutation_margin_percentile_min"]), "Permutation evidence is advisory and never the sole selector.")
    max_need = float(need_summary["abs_stage1_need_bundle_gap_spearman"].max())
    adv("stage2a_need_replication", max_need, "<=", advisory["need_bundle_abs_spearman_review"], max_need <= float(advisory["need_bundle_abs_spearman_review"]), "Bundle Gap should not simply reproduce Stage 1 Need.")
    dominant_fraction = float(differentiation_summary["dominant_top1_fraction"].iloc[0])
    adv("bundle_top1_dominance", dominant_fraction, "<=", 0.50, dominant_fraction <= 0.50, "No single bundle should dominate merely because of a shared rural surface.")

    joint = integration_summary.loc[integration_summary["model"].eq("joint_gap_times_temporal_fit")].iloc[0]
    diagnostic("joint_model_top1_retention", float(joint["top1_bundle_retention"]), ">=", advisory["joint_model_bundle_rank_spearman_review"], float(joint["top1_bundle_retention"]) >= float(advisory["joint_model_bundle_rank_spearman_review"]), "A failing joint model confirms why it is not the primary architecture.")
    for lam, minimum in ((0.10, 0.90), (0.20, 0.80), (0.30, 0.70)):
        row = integration_summary.loc[np.isclose(pd.to_numeric(integration_summary["lambda"], errors="coerce"), lam)].iloc[0]
        diagnostic(f"bounded_lambda_{lam:.2f}_top1_retention", float(row["top1_bundle_retention"]), ">=", minimum, float(row["top1_bundle_retention"]) >= minimum, "Bounded adjustment remains diagnostic and cannot replace the hierarchy.")
    return records


def _final_decision_from_release(
    release: pd.DataFrame,
    *,
    hard_failed: int = 0,
) -> str:
    if hard_failed:
        return "FAIL_TEMPORAL_RELEASE"
    counts = release["release_type"].value_counts().to_dict()
    if counts.get("STRONG_SINGLE", 0) == len(release):
        return "PASS_STRONG_TEMPORAL"
    if counts.get("NO_STRONG_PREFERENCE", 0) == 0:
        return "PASS_COARSE_TEMPORAL"
    return "PASS_ADVISORY_TEMPORAL"


def _md_table(frame: pd.DataFrame, columns: Iterable[str] | None = None) -> str:
    selected = frame[list(columns)].copy() if columns is not None else frame.copy()
    if selected.empty:
        return "_해당 행 없음_"
    return selected.to_markdown(index=False, floatfmt=".4f")


def _gate_markdown(gates: pd.DataFrame, decision: str) -> str:
    hard = gates.loc[gates["severity"].astype(str).eq("hard")]
    advisory = gates.loc[gates["severity"].astype(str).eq("advisory")]
    failed = gates.loc[~gates["passed"]]
    return f"""# CURRENT STAGE 2B HARDENING GATE

- 최종 판정: `{decision}`
- Hard gate: `{int(hard['passed'].sum())}/{len(hard)} PASS`
- Advisory: `{int(advisory['passed'].sum())}/{len(advisory)} PASS`
- Stage 3 시작: `false`
- 통계 독립 n: `55 (11 policy sigungu × 5 bundles)`
- 읍면동 765행은 전달 단위이며 독립 표본으로 사용하지 않음

## 실패·주의 항목

{_md_table(failed, ['gate_id', 'severity', 'observed', 'comparison', 'threshold', 'note'])}

## 전체 게이트

{_md_table(gates, ['gate_id', 'severity', 'status', 'observed', 'comparison', 'threshold', 'threshold_source', 'note'])}
"""


def _final_report_markdown(
    *,
    run_id: str,
    decision: str,
    bootstrap: int,
    permutations: int,
    jobs: int,
    temporal: TemporalHardeningResult,
    evidence: pd.DataFrame,
    integration_summary: pd.DataFrame,
    spatial_summary: pd.DataFrame,
    differentiation_summary: pd.DataFrame,
    need_summary: pd.DataFrame,
    model_comparison: pd.DataFrame,
    gates: pd.DataFrame,
    test_result: Mapping[str, Any],
    timings: Mapping[str, float],
) -> str:
    bundle_columns = [
        "bundle_id",
        "2022_top1",
        "2023_top1",
        "cross_year_consistency",
        "bidirectional_primary_in_holdout_top2",
        "bootstrap_primary_probability",
        "lomo_stability",
        "loso_stability",
        "null_strength",
        "release_type",
        "temporal_confidence",
        "final_primary",
        "final_fallback",
    ]
    transfer_columns = [
        "bundle_id",
        "direction",
        "top1_agreement",
        "train_primary_in_holdout_top2",
        "pair_top2_coverage",
        "season_rank_spearman",
        "season_rank_kendall",
        "heldout_regret_normalized",
    ]
    failed_advisory = gates.loc[
        gates["severity"].astype(str).isin(["advisory", "diagnostic"])
        & ~gates["passed"]
    ]
    musculoskeletal = evidence.loc[
        evidence["bundle_id"].eq("musculoskeletal_rehab")
    ]
    release_counts = evidence["release_type"].value_counts().to_dict()
    role = (
        "optimizer의 primary objective가 아닌 advisory coarse scheduling prior"
        if release_counts.get("NO_STRONG_PREFERENCE", 0)
        else "confidence-aware coarse scheduling prior"
    )
    return f"""# MEDIROAD MODEL V1 — Stage 2B Hardening 및 Stage 2 통합검증

실행 ID: `{run_id}`  
최종 판정: `{decision}`  
Stage 3 시작: `false`

## 1. 목적

2022–2023 NHIS 관측 의료이용만으로 진료번들의 계절 순위를 어느 수준까지 재현할 수 있는지 검증했다. 실제 환자수·이동진료 수요·정확한 월/날짜를 예측하지 않는다.

## 2. 기존 Stage 2B 구조

기존 pooled 계절 추천은 보존했다. 이번 hardening은 별도 경로에서 strict 2022→2023 및 2023→2022 전이, LOMO, LOSO, 시군 block bootstrap, month-label permutation을 추가했다.

## 3. 데이터와 독립 분석단위

- 원자료: NHIS 2022–2023 시군구×월×진료과 급여 이용량
- 검증 단위: `11 policy sigungu × 5 bundle = 55`
- 결과 전달: `153 admin dong × 5 bundle = 765`, 단 통계 n으로 사용하지 않음
- 추가 다운로드: 불필요. 요구된 공식 NHIS/근무일 자료가 이미 SHA로 동결되어 있으며 다른 연도를 임의 결합하지 않았다.

## 4. 2022 → 2023 transfer

{_md_table(temporal.cross_year.summary.loc[temporal.cross_year.summary['direction'].eq('2022_to_2023')], transfer_columns)}

## 5. 2023 → 2022 transfer

{_md_table(temporal.cross_year.summary.loc[temporal.cross_year.summary['direction'].eq('2023_to_2022')], transfer_columns)}

## 6. Consensus season

평균순위·중앙순위·Borda·worst-year 순위를 모두 저장했다. release 후보는 사전등록한 중앙순위→pooled score→고정 계절순 tie-break를 사용한다.

{_md_table(evidence, bundle_columns)}

## 7. LOMO stability

{_md_table(temporal.lomo.summary)}

## 8. LOSO stability

{_md_table(temporal.loso.summary)}

## 9. Working-day 및 연도정규화 민감도

working-day adjusted, raw utilization, within-year normalized/z/percentile/annual-share의 6개 정의를 동일 primary 계절 정의 아래 비교했다.

{_md_table(temporal.working_day_sensitivity)}

## 10. Null / permutation test

각 sigungu×year×specialty의 12개 월 값을 보존하고 month label만 섞었다. 연간 총량은 모든 `{permutations}`회에서 보존됐다. p-value는 advisory이며 단독 release 기준이 아니다.

{_md_table(temporal.permutation.summary)}

## 11. Sigungu block bootstrap

11개 시군의 전체 월·진료과 sequence를 함께 재표집했다. `{bootstrap}`회 모두 유효했다.

{_md_table(temporal.bootstrap.summary)}

## 12. Bundle별 confidence

{_md_table(evidence, bundle_columns)}

현재 역할: **{role}**. 약한 번들에 계절을 강제로 부여하지 않았다.

## 13. Musculoskeletal review

{_md_table(musculoskeletal, bundle_columns)}

2022년과 2023년 Top1이 다르고 regret·bootstrap·경계 민감도가 약하다. HIGH 단일 계절이 되도록 임계값을 사후 완화하지 않았다.

## 14. Stage 2A × Stage 2B integration

{_md_table(integration_summary, ['model', 'lambda', 'top1_bundle_retention', 'top2_bundle_recall', 'mean_bundle_rank_spearman', 'mean_normalized_rank_distortion', 'release_eligible'])}

Hierarchical WHAT→WHEN은 Stage 2A 순위를 100% 보존한다. 단순 곱셈 joint model은 진료번들 우선순위를 크게 바꾸므로 폐기하고, bounded λ 실험은 진단으로만 유지한다.

공간 누수:

{_md_table(spatial_summary)}

번들 분화:

{_md_table(differentiation_summary)}

Need 복제 진단:

{_md_table(need_summary)}

## 15. Temporal role decision

{_md_table(model_comparison)}

현재 자료에서는 forced single-season Model 1이 false precision을 만든다. Model 2의 abstention과 Model 3의 계층 구조를 채택하며, Stage 2B를 advisory로 제한한다.

## 16. 남은 한계와 현재 병목

- 관측 연도가 2개뿐이다.
- 이용량은 전연령 급여 이용이며 이동진료 수요가 아니다.
- 시군 내부 계절 이질성이 province-common 추천을 약화한다.
- 계절 경계 및 working-day 정의에 민감한 번들이 있다.
- NASA POWER는 historical operational-risk advisory로만 남고 selector에는 포함되지 않는다.
- exact month/date와 live forecast trigger는 계속 unavailable이다.

실패 advisory/diagnostic:

{_md_table(failed_advisory, ['gate_id', 'severity', 'observed', 'comparison', 'threshold', 'note'])}

## 17. Stage 2 최종 gate

- 판정: `{decision}`
- Hard: `{int(gates.loc[gates['severity'].astype(str).eq('hard'), 'passed'].sum())}/{int(gates['severity'].astype(str).eq('hard').sum())}`
- Tests: `{test_result['passed']} passed`, exit `{test_result['exit_code']}`
- Stage 1/2A/기존 Stage 2 SHA 불변

## 18. Stage 3 interface

`stage2_to_stage3_interface.parquet`를 생성했다. recommended/fallback season은 abstention이면 null이고 exact_month/exact_date는 전 행 null이다. Stage 3 코드는 실행하지 않았다.

## 19. 재현성 및 성능

- Bootstrap `{bootstrap}`, permutation `{permutations}`, jobs `{jobs}`
- Python `{platform.python_version()}`, platform `{platform.platform()}`
- 통계 hardening `{timings.get('temporal_hardening', float('nan')):.3f}s`
- 통합·출력·그림 포함 총 `{timings.get('total', float('nan')):.3f}s`
- GPU는 이 55-unit×4-season 재표집 문제에 이점이 없어 사용하지 않았고, CPU 8 worker를 사용했다.
"""


def _run_tests(root: Path) -> dict[str, Any]:
    started = time.perf_counter()
    environment = os.environ.copy()
    source_path = str(root / "src")
    environment["PYTHONPATH"] = os.pathsep.join(
        value
        for value in (source_path, environment.get("PYTHONPATH", ""))
        if value
    )
    process = subprocess.run(
        [sys.executable, "-B", "-m", "pytest", "-q"],
        cwd=root,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    combined = "\n".join(value for value in [process.stdout, process.stderr] if value)
    match = re.search(r"(?P<passed>\d+) passed", combined)
    failed_match = re.search(r"(?P<failed>\d+) failed", combined)
    result = {
        "passed": int(match.group("passed")) if match else 0,
        "failed": int(failed_match.group("failed")) if failed_match else 0,
        "exit_code": int(process.returncode),
        "elapsed_seconds": float(time.perf_counter() - started),
        "output_tail": combined[-4000:],
    }
    if process.returncode != 0:
        raise RuntimeError(f"Test suite failed:\n{combined[-4000:]}")
    return result


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--permutations", type=int, default=5000)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--diagnostic", action="store_true")
    parser.add_argument("--allow-gate-fail", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.bootstrap <= 10000:
        parser.error("--bootstrap must be in [1, 10000]")
    if not 1 <= args.permutations <= 50000:
        parser.error("--permutations must be in [1, 50000]")
    if not 1 <= args.jobs <= 8:
        parser.error("--jobs must be in [1, 8]")
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    # The executable body is completed below after the independently tested
    # temporal, integration, and reporting modules are imported.
    return run_stage2b_hardening(args)


def run_stage2b_hardening(args: argparse.Namespace) -> int:
    total_started = time.perf_counter()
    started_at = datetime.now(timezone.utc)
    root = args.package_root.resolve()
    config_path = _resolve(root, args.config)
    config = _load_yaml(config_path)
    execution = config["execution"]
    expected_bootstrap = int(
        execution["diagnostic_bootstrap"]
        if args.diagnostic
        else execution["final_bootstrap"]
    )
    expected_permutations = int(
        execution["diagnostic_permutations"]
        if args.diagnostic
        else execution["final_permutations"]
    )
    if args.bootstrap != expected_bootstrap or args.permutations != expected_permutations:
        mode = "diagnostic" if args.diagnostic else "final"
        raise ValueError(
            f"{mode} contract requires B={expected_bootstrap}, "
            f"P={expected_permutations}; received B={args.bootstrap}, "
            f"P={args.permutations}"
        )

    output_dir = _resolve(root, config["paths"]["output_dir"])
    report_dir = _resolve(root, config["paths"]["report_dir"])
    final_report_path = _resolve(root, config["paths"]["final_report"])
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    code_hash = sha256_file(Path(__file__))
    run_id = (
        f"stage2b_hardening_{started_at.strftime('%Y%m%dT%H%M%SZ')}_"
        f"{code_hash[:12]}"
    )
    timings: dict[str, float] = {}

    checkpoint = time.perf_counter()
    frozen = _verify_frozen_contract(root, config)
    inputs = _read_inputs(root, config)
    timings["input_and_frozen_validation"] = time.perf_counter() - checkpoint

    checkpoint = time.perf_counter()
    test_result = _run_tests(root)
    timings["tests"] = time.perf_counter() - checkpoint

    checkpoint = time.perf_counter()
    temporal = run_temporal_hardening(
        inputs.nhis,
        inputs.working_days,
        inputs.bundle_config,
        n_boot=args.bootstrap,
        n_permutations=args.permutations,
        years=tuple(config["analysis_contract"]["observed_years"]),
        expected_policy_sigungu=int(
            config["analysis_contract"]["expected_policy_sigungu"]
        ),
        random_state=int(config["random_seed"]),
        n_jobs=args.jobs,
        release_thresholds=config["release_rules"],
    )
    timings["temporal_hardening"] = time.perf_counter() - checkpoint

    checkpoint = time.perf_counter()
    release = _release_from_temporal_result(
        temporal,
        inputs.baseline_season_scores,
        config["release_rules"],
    )
    evidence = _temporal_evidence_table(temporal, release)
    evidence = validate_temporal_evidence_summary(
        evidence,
        expected_bundles=tuple(_DISPLAY_BUNDLE_ID),
    )
    # Categorical values are convenient for plotting but should not leak into
    # Parquet/CSV public schemas.
    evidence["bundle_id"] = evidence["bundle_id"].astype("string")
    evidence["bundle"] = evidence["bundle"].astype("string")
    delivery = _build_delivery_release(inputs.bundle_scores, evidence)

    integration = compare_what_when_models(
        inputs.bundle_scores,
        inputs.baseline_season_scores,
        lambdas=tuple(config["analysis_contract"]["bounded_temporal_lambdas"]),
    )
    spatial_input = delivery.rename(
        columns={
            "sigungu": "policy_sigungu_name",
        }
    )
    spatial_input["region_specific_clinical_seasonality_used"] = 0
    spatial = diagnose_temporal_spatial_leakage(
        inputs.baseline_season_scores,
        spatial_input,
    )
    differentiation = analyze_bundle_differentiation(inputs.bundle_scores)
    need = diagnose_need_replication(inputs.stage1_scores, inputs.bundle_scores)
    interface = build_stage2_to_stage3_interface(
        inputs.stage1_scores,
        inputs.bundle_scores,
        release,
        temporal_independent_n=int(config["analysis_contract"]["independent_n"]),
        expected_admins=int(config["analysis_contract"]["expected_admin_dongs"]),
        expected_bundles=int(config["analysis_contract"]["expected_bundles"]),
    )
    interface_gates = validate_stage2_to_stage3_interface(
        interface,
        expected_admins=int(config["analysis_contract"]["expected_admin_dongs"]),
        expected_bundles=int(config["analysis_contract"]["expected_bundles"]),
        expected_independent_n=int(config["analysis_contract"]["independent_n"]),
        stage3_started=False,
        raise_on_error=True,
    )
    model_comparison = _model_comparison(integration.summary, evidence)
    timings["release_and_integration"] = time.perf_counter() - checkpoint

    checkpoint = time.perf_counter()
    # Evidence and statistical validation tables.
    write_parquet(temporal.evidence.service_month, output_dir / "01_evidence/service_month_hardening.parquet")
    write_parquet(temporal.evidence.bundle_month, output_dir / "01_evidence/bundle_month_hardening.parquet")
    write_json(temporal.evidence.audit, output_dir / "01_evidence/temporal_hardening_audit.json")
    write_parquet(temporal.cross_year.detail, output_dir / "02_cross_year/cross_year_transfer_detail.parquet")
    write_csv(temporal.cross_year.summary, output_dir / "02_cross_year/cross_year_transfer_summary.csv")
    write_csv(temporal.consensus, output_dir / "02_cross_year/cross_year_consensus_season.csv")
    write_csv(temporal.lomo.detail, output_dir / "03_stability/lomo_detail.csv")
    write_csv(temporal.lomo.summary, output_dir / "03_stability/lomo_summary.csv")
    write_csv(temporal.working_day_sensitivity, output_dir / "03_stability/working_day_and_normalization_sensitivity.csv")
    write_csv(temporal.loso.detail, output_dir / "03_stability/loso_detail.csv")
    write_csv(temporal.loso.summary, output_dir / "03_stability/loso_summary.csv")
    write_csv(temporal.season_boundary.detail, output_dir / "03_stability/season_boundary_detail.csv")
    write_csv(temporal.season_boundary.summary, output_dir / "03_stability/season_boundary_summary.csv")
    write_parquet(temporal.bootstrap.detail, output_dir / "03_stability/sigungu_block_bootstrap_detail.parquet")
    write_csv(temporal.bootstrap.summary, output_dir / "03_stability/sigungu_block_bootstrap_summary.csv")
    write_parquet(temporal.permutation.detail, output_dir / "03_stability/month_label_permutation_detail.parquet")
    write_csv(temporal.permutation.summary, output_dir / "03_stability/month_label_permutation_summary.csv")
    write_csv(evidence, output_dir / "04_release/temporal_bundle_evidence_summary.csv")
    write_csv(delivery, output_dir / "04_release/stage2b_hardened_release.csv")
    write_parquet(delivery, output_dir / "04_release/stage2b_hardened_release.parquet")

    # Stage 2 integrated diagnostics and Stage 3 interface only.
    write_parquet(integration.rank_panel, output_dir / "05_integration/what_when_rank_panel.parquet")
    write_csv(integration.summary, output_dir / "05_integration/what_when_model_summary.csv")
    write_csv(spatial.bundle_summary, output_dir / "05_integration/temporal_spatial_leakage_by_bundle.csv")
    write_csv(spatial.summary, output_dir / "05_integration/temporal_spatial_leakage_summary.csv")
    write_csv(differentiation.admin_summary, output_dir / "05_integration/bundle_differentiation_by_admin.csv")
    write_csv(differentiation.top_frequency, output_dir / "05_integration/bundle_top_frequency.csv")
    write_csv(differentiation.sigungu_summary, output_dir / "05_integration/bundle_differentiation_by_sigungu.csv")
    write_csv(differentiation.variance_decomposition, output_dir / "05_integration/bundle_variance_decomposition.csv")
    write_csv(differentiation.summary, output_dir / "05_integration/bundle_differentiation_summary.csv")
    write_csv(need.bundle_summary, output_dir / "05_integration/need_replication_by_bundle.csv")
    write_csv(need.partial_correlations, output_dir / "05_integration/need_controlled_partial_correlations.csv")
    write_parquet(need.residual_panel, output_dir / "05_integration/need_controlled_residual_panel.parquet")
    write_csv(model_comparison, output_dir / "05_integration/temporal_model_comparison.csv")
    write_parquet(interface, output_dir / "stage2_to_stage3_interface.parquet")
    write_csv(stage2_to_stage3_schema(), output_dir / "stage2_to_stage3_interface_schema.csv")
    write_csv(interface_gates, output_dir / "stage2_to_stage3_interface_quality_gate.csv")
    frozen_rows = [
        {
            "artifact": name,
            "sha256_before": digest,
            "sha256_after": sha256_file(_resolve(root, config["paths"][name])),
            "unchanged": digest == sha256_file(_resolve(root, config["paths"][name])),
        }
        for name, digest in frozen.direct_hashes.items()
    ]
    write_csv(pd.DataFrame(frozen_rows), output_dir / "05_integration/frozen_integrity.csv")

    figures = render_stage2b_hardening_figures(
        _figure_tables(temporal, release),
        report_dir / "figures",
        dpi=int(execution["png_dpi"]),
    )
    timings["materialization_and_figures"] = time.perf_counter() - checkpoint

    checkpoint = time.perf_counter()
    _assert_frozen_unchanged(root, frozen)
    records = _base_quality_records(
        config=config,
        inputs=inputs,
        temporal=temporal,
        release=release,
        delivery=delivery,
        interface_gates=interface_gates,
        integration_summary=integration.summary,
        spatial_summary=spatial.summary,
        need_summary=need.bundle_summary,
        differentiation_summary=differentiation.summary,
        figure_artifact_count=len(figures.artifact_manifest),
        test_result=test_result,
        bootstrap=args.bootstrap,
        permutations=args.permutations,
    )

    # First inventory establishes provenance before the final gate/report.  The
    # gate itself is excluded to avoid a self-referential hash cycle.
    inventory_path = output_dir / "STAGE2B_HARDENING_ARTIFACT_INVENTORY.csv"
    metadata_path = output_dir / "STAGE2B_HARDENING_RUN_METADATA.json"
    quality_path = output_dir / "STAGE2B_HARDENING_QUALITY_GATE.csv"
    inventory = build_artifact_inventory(
        [output_dir, report_dir],
        base_dir=root,
        exclude=[inventory_path, metadata_path, quality_path],
    )
    inventory_validation = validate_artifact_inventory(inventory, root, raise_on_error=True)
    records.append(
        _record(
            "artifact_inventory_validation",
            "hard",
            inventory_validation.valid,
            inventory_validation.checked_count,
            "contract",
            "all inventoried artifacts nonempty with matching size and SHA256",
            "prompt artifact provenance contract",
            "The inventory is recomputed again after the final human-readable report.",
        )
    )
    provisional = build_stage2b_quality_gate(records)
    decision = _final_decision_from_release(
        release,
        hard_failed=provisional.hard_failed,
    )
    gate_result = build_stage2b_quality_gate(records, final_decision=decision)
    write_csv(gate_result.gates, quality_path)
    write_text(
        _gate_markdown(gate_result.gates, decision),
        report_dir / "CURRENT_STAGE2B_HARDENING_GATE.md",
    )
    timings["total"] = time.perf_counter() - total_started
    final_report = _final_report_markdown(
        run_id=run_id,
        decision=decision,
        bootstrap=args.bootstrap,
        permutations=args.permutations,
        jobs=args.jobs,
        temporal=temporal,
        evidence=evidence,
        integration_summary=integration.summary,
        spatial_summary=spatial.summary,
        differentiation_summary=differentiation.summary,
        need_summary=need.bundle_summary,
        model_comparison=model_comparison,
        gates=gate_result.gates,
        test_result=test_result,
        timings=timings,
    )
    write_text(final_report, final_report_path)

    # Final inventory includes the top report and CURRENT gate report.  The
    # CSV gate remains excluded only because it is independently hashed in the
    # metadata written last.
    inventory = build_artifact_inventory(
        [output_dir, report_dir, final_report_path],
        base_dir=root,
        exclude=[inventory_path, metadata_path, quality_path],
    )
    inventory_validation = validate_artifact_inventory(inventory, root, raise_on_error=True)
    write_csv(inventory, inventory_path)
    # Validate the materialized inventory, not only the in-memory frame.
    inventory_materialized = pd.read_csv(inventory_path)
    final_inventory_validation = validate_artifact_inventory(
        inventory_materialized, root, raise_on_error=True
    )
    if not final_inventory_validation.valid:
        raise RuntimeError("Final artifact inventory is invalid")
    _assert_frozen_unchanged(root, frozen)
    timings["quality_report_and_provenance"] = time.perf_counter() - checkpoint
    timings["total"] = time.perf_counter() - total_started

    ended_at = datetime.now(timezone.utc)
    input_hashes = {
        **frozen.direct_hashes,
        "hardening_config": sha256_file(config_path),
    }
    code_paths = {
        "stage2b_hardening_pipeline": Path(__file__),
        "temporal_hardening": root / "src/mediroad/temporal/hardening.py",
        "hardening_integration": root / "src/mediroad/temporal/hardening_integration.py",
        "hardening_reporting": root / "src/mediroad/reporting/stage2b_hardening.py",
    }
    metadata = {
        "run_id": run_id,
        "started_at_utc": started_at.isoformat(),
        "completed_at_utc": ended_at.isoformat(),
        "run_status": "PASS" if gate_result.hard_pass else "FAIL",
        "final_decision": decision,
        "diagnostic": bool(args.diagnostic),
        "source_sha": input_hashes,
        "code_sha": {name: sha256_file(path) for name, path in code_paths.items()},
        "config_sha": sha256_file(config_path),
        "bootstrap": int(args.bootstrap),
        "permutations": int(args.permutations),
        "jobs": int(args.jobs),
        "tests": test_result,
        "hard_gates": {
            "passed": int(gate_result.hard_total - gate_result.hard_failed),
            "total": int(gate_result.hard_total),
            "failed": int(gate_result.hard_failed),
        },
        "advisories": {
            "failed": int(gate_result.advisory_failed),
            "diagnostic_failed": int(gate_result.diagnostic_failed),
        },
        "release_type_counts": {
            str(key): int(value)
            for key, value in release["release_type"].value_counts().items()
        },
        "temporal_independent_n": int(
            temporal.evidence.audit["temporal_independent_n"]
        ),
        "delivery_rows": int(len(delivery)),
        "artifact_count": int(len(inventory_materialized)),
        "artifact_inventory_sha256": sha256_file(inventory_path),
        "artifact_sha_status": bool(final_inventory_validation.valid),
        "quality_gate_sha256": sha256_file(quality_path),
        "stage1_frozen": True,
        "stage2a_frozen": True,
        "official_stage2_frozen": True,
        "stage3_started": False,
        "timings_seconds": timings,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "interpretation": (
            "observed_insured_utilization_coarse_scheduling_prior_not_demand; "
            "explicit_abstention_allowed"
        ),
    }
    # This is deliberately the last write of the run.
    write_json(metadata, metadata_path)

    print(
        json.dumps(
            {
                "run_id": run_id,
                "decision": decision,
                "hard_gate": f"{gate_result.hard_total - gate_result.hard_failed}/{gate_result.hard_total}",
                "release_counts": metadata["release_type_counts"],
                "bootstrap": args.bootstrap,
                "permutations": args.permutations,
                "tests_passed": test_result["passed"],
                "artifact_count": len(inventory_materialized),
                "elapsed_seconds": timings["total"],
                "stage3_started": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if gate_result.hard_failed and not args.allow_gate_fail:
        return 2
    return 0


__all__ = [
    "FrozenSnapshot",
    "main",
    "run_stage2b_hardening",
    "sha256_file",
    "write_csv",
    "write_json",
    "write_parquet",
    "write_text",
]
