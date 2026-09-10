"""Recommended service-bundle aggregation for Stage 2A."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .gap import COMPONENT_COLUMNS, FORMULA_COLUMNS


def load_service_bundle_config(path: str | Path) -> dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Service-bundle config must be a YAML mapping")
    return config


def _bundle_contract(
    bundle_config: Mapping[str, Any],
    *,
    service_ids: set[str],
) -> list[dict[str, Any]]:
    bundles = bundle_config.get("bundles")
    if not isinstance(bundles, list) or not bundles:
        raise ValueError("Service-bundle config requires a non-empty bundles list")
    bundle_ids = [str(bundle.get("bundle_id", "")) for bundle in bundles]
    if any(not value for value in bundle_ids) or len(bundle_ids) != len(set(bundle_ids)):
        raise ValueError("bundle_id values must be nonblank and unique")
    excluded = {
        str(value.get("service_id"))
        for value in bundle_config.get("excluded_from_mobile_bundles", [])
    }
    for bundle in bundles:
        included = bundle.get("included_services")
        if not isinstance(included, Mapping) or not included:
            raise ValueError(f"{bundle['bundle_id']} has no included_services")
        unknown = sorted(set(map(str, included)) - service_ids)
        if unknown:
            raise ValueError(f"{bundle['bundle_id']} references unknown services: {unknown}")
        forbidden = sorted(set(map(str, included)) & excluded)
        if forbidden:
            raise ValueError(f"{bundle['bundle_id']} includes excluded services: {forbidden}")
        weights = np.asarray([float(value) for value in included.values()], dtype=float)
        if not np.isfinite(weights).all() or (weights <= 0).any() or not np.isclose(
            weights.sum(), 1.0
        ):
            raise ValueError(f"{bundle['bundle_id']} weights must be positive and sum to one")
    return [dict(bundle) for bundle in bundles]


def build_bundle_scores(
    specialty_scores: pd.DataFrame,
    bundle_config: Mapping[str, Any],
    *,
    score_column: str = "specialty_gap_score",
) -> pd.DataFrame:
    """Aggregate service scores to five auditable recommended bundles."""

    required = {
        "admin_dong_code",
        "admin_dong_name",
        "policy_sigungu_name",
        "service_id",
        score_column,
    }
    missing = sorted(required - set(specialty_scores.columns))
    if missing:
        raise KeyError(f"Specialty scores lack bundle inputs: {missing}")
    if specialty_scores.duplicated(["admin_dong_code", "service_id"]).any():
        raise ValueError("Specialty score admin-service key is not unique")
    service_ids = set(specialty_scores["service_id"].astype(str))
    bundles = _bundle_contract(bundle_config, service_ids=service_ids)
    admin_count = specialty_scores["admin_dong_code"].nunique()

    aggregate_sources = [score_column]
    aggregate_sources.extend(column for column in COMPONENT_COLUMNS if column in specialty_scores)
    aggregate_sources.extend(
        column for column in FORMULA_COLUMNS.values() if column in specialty_scores
    )
    aggregate_sources = list(dict.fromkeys(aggregate_sources))

    base = (
        specialty_scores[
            ["admin_dong_code", "admin_dong_name", "policy_sigungu_name"]
        ]
        .drop_duplicates("admin_dong_code")
        .sort_values("admin_dong_code", kind="stable")
        .reset_index(drop=True)
    )
    if len(base) != admin_count:
        raise ValueError("Admin descriptive labels are inconsistent across services")

    frames: list[pd.DataFrame] = []
    for bundle_order, bundle in enumerate(bundles, start=1):
        included = {str(key): float(value) for key, value in bundle["included_services"].items()}
        selected = specialty_scores.loc[
            specialty_scores["service_id"].isin(included),
            ["admin_dong_code", "service_id", *aggregate_sources],
        ].copy()
        if len(selected) != admin_count * len(included):
            raise ValueError(f"{bundle['bundle_id']} lacks one or more admin-service rows")
        selected["__bundle_weight"] = selected["service_id"].map(included)
        result = base.copy()
        for column in aggregate_sources:
            weighted = (
                selected.assign(__weighted=selected[column] * selected["__bundle_weight"])
                .groupby("admin_dong_code", sort=False)["__weighted"]
                .sum()
            )
            output_column = (
                "bundle_gap_score"
                if column == score_column
                else f"bundle_{column}"
            )
            result[output_column] = result["admin_dong_code"].map(weighted).astype(float)
        result["bundle_order"] = bundle_order
        result["bundle_id"] = str(bundle["bundle_id"])
        result["bundle_name_ko"] = str(bundle["bundle_name_ko"])
        result["included_service_ids"] = "|".join(included)
        result["included_service_weights"] = "|".join(
            f"{key}:{value:.8g}" for key, value in included.items()
        )
        for column in (
            "mobile_service_suitability",
            "required_staff",
            "required_equipment",
            "required_venue_conditions",
            "incompatible_conditions",
            "estimated_duration",
            "confidence",
        ):
            result[column] = str(bundle.get(column, "unknown"))
        result["compatible_months"] = "|".join(
            str(value) for value in bundle.get("compatible_months", [])
        )
        result["availability_source"] = str(
            bundle_config.get("availability_source", "unknown")
        )
        result["bundle_config_version"] = str(bundle_config.get("version", "unknown"))
        frames.append(result)

    output = pd.concat(frames, ignore_index=True)
    output = output.sort_values(["bundle_order", "admin_dong_code"], kind="stable").reset_index(
        drop=True
    )
    output["bundle_gap_rank_tied"] = output.groupby("bundle_id", sort=False)[
        "bundle_gap_score"
    ].rank(method="min", ascending=False)
    output["bundle_gap_rank"] = 0
    for _, indices in output.groupby("bundle_id", sort=False).groups.items():
        ordered = output.loc[list(indices)].sort_values(
            ["bundle_gap_score", "admin_dong_code"],
            ascending=[False, True],
            kind="stable",
        )
        output.loc[ordered.index, "bundle_gap_rank"] = np.arange(1, len(ordered) + 1)
    output["bundle_gap_rank"] = output["bundle_gap_rank"].astype(int)
    output["bundle_gap_rank_tied"] = output["bundle_gap_rank_tied"].astype(int)
    output["rank_tiebreaker"] = "bundle_gap_score_desc_then_admin_dong_code_asc"
    return output
