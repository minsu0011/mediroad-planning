"""Fail-closed contracts for MEDIROAD Stage 2A specialty outputs."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any
import hashlib

import numpy as np
import pandas as pd

from .gap import COMPONENT_COLUMNS, FORMULA_COLUMNS


class SpecialtyValidationError(ValueError):
    """Raised when a Stage 2A config or result violates its frozen contract."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_specialty_config(config: Mapping[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    required = {
        "version",
        "target_definition",
        "primary_formula",
        "primary_scaler",
        "source",
        "model_contract",
        "component_weights",
        "supply",
        "scalers",
        "formula_families",
        "services",
        "quality_gates",
    }
    missing = sorted(required - set(config))
    if missing:
        errors.append(f"missing top-level config fields: {missing}")
        return {"valid": False, "errors": errors}

    services = config.get("services")
    if not isinstance(services, list) or not services:
        errors.append("services must be a non-empty list")
        services = []
    service_ids = [str(value.get("service_id", "")) for value in services]
    service_names = [str(value.get("name_ko", "")) for value in services]
    if any(not value for value in service_ids) or len(service_ids) != len(set(service_ids)):
        errors.append("service_id values must be nonblank and unique")
    if any(not value for value in service_names) or len(service_names) != len(set(service_names)):
        errors.append("name_ko values must be nonblank and unique")
    expected_services = int(config["quality_gates"].get("expected_service_count", -1))
    if len(services) != expected_services:
        errors.append(f"expected {expected_services} services, found {len(services)}")
    focused = sum(
        str(value.get("service_type")) == "focused_physician_specialty" for value in services
    )
    expected_focused = int(
        config["quality_gates"].get("expected_focused_physician_specialty_count", -1)
    )
    if focused != expected_focused:
        errors.append(f"expected {expected_focused} focused specialties, found {focused}")
    if service_ids.count("dental") != 1:
        errors.append("exactly one dental adjunct service is required")
    for service in services:
        for field in (
            "service_id",
            "name_ko",
            "service_type",
            "health_features",
            "health_evidence",
            "age_features",
            "public_support_column",
            "public_support_evidence",
            "mobile_service_role",
        ):
            if field not in service:
                errors.append(f"{service.get('service_id', '<unknown>')} lacks {field}")
        if not service.get("health_features") or not service.get("age_features"):
            errors.append(f"{service.get('service_id')} requires health and age features")

    weights = config.get("component_weights", {})
    expected_weight_keys = {
        "health_context_gap",
        "age_structure_gap",
        "specialty_supply_gap",
        "specialty_excess_access_gap",
        "public_support_gap",
    }
    if set(weights) != expected_weight_keys:
        errors.append("component_weights do not match the five Stage 2A components")
    else:
        weight_values = np.asarray([float(value) for value in weights.values()], dtype=float)
        if not np.isfinite(weight_values).all() or (weight_values <= 0).any():
            errors.append("component weights must be positive and finite")
        if not np.isclose(weight_values.sum(), 1.0):
            errors.append("component weights must sum to one")
        maximum = float(config["quality_gates"].get("primary_weight_max", 1.0))
        if float(weight_values.max()) > maximum + 1e-12:
            errors.append("a component weight exceeds primary_weight_max")

    formulas = set(config.get("formula_families", {}))
    if formulas != set(FORMULA_COLUMNS):
        errors.append(f"formula families must be exactly {sorted(FORMULA_COLUMNS)}")
    if config.get("primary_formula") not in formulas:
        errors.append("primary_formula is not a configured formula family")
    if config.get("primary_scaler") not in config.get("scalers", {}):
        errors.append("primary_scaler is not configured")
    contract = config.get("model_contract", {})
    if contract.get("use_stage1_need_as_specialty_input") is not False:
        errors.append("Stage 1 Need must not be a Specialty Gap input")
    if contract.get("use_legacy_v0_as_input") is not False:
        errors.append("Legacy V0 gap must not be a Specialty Gap input")
    expected_rows = int(config["source"].get("expected_admin_dongs", 0)) * len(services)
    if int(config["quality_gates"].get("expected_specialty_rows", -1)) != expected_rows:
        errors.append("expected_specialty_rows does not equal admins × services")
    return {
        "valid": not errors,
        "errors": errors,
        "service_count": len(services),
        "focused_specialty_count": focused,
    }


def require_valid_specialty_config(config: Mapping[str, Any]) -> None:
    result = validate_specialty_config(config)
    if not result["valid"]:
        raise SpecialtyValidationError("; ".join(result["errors"]))


def validate_specialty_results(
    scores: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    bundle_scores: pd.DataFrame | None = None,
    bundle_config: Mapping[str, Any] | None = None,
    canonical_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate cardinality, key, boundedness, contributions, and leakage."""

    require_valid_specialty_config(config)
    gates = config["quality_gates"]
    expected_rows = int(gates["expected_specialty_rows"])
    expected_admins = int(config["source"]["expected_admin_dongs"])
    expected_services = int(gates["expected_service_count"])
    component_columns = list(COMPONENT_COLUMNS)
    formula_columns = list(FORMULA_COLUMNS.values())
    contribution_columns = [
        f"contribution_{column.removesuffix('_score')}" for column in component_columns
    ]
    required = {
        "admin_dong_code",
        "service_id",
        "service_type",
        "specialty_gap_score",
        "specialty_gap_rank",
        *component_columns,
        *formula_columns,
        *contribution_columns,
    }
    missing_columns = sorted(required - set(scores.columns))
    checks: dict[str, bool] = {
        "required_columns": not missing_columns,
        "row_count": len(scores) == expected_rows,
        "admin_count": scores.get("admin_dong_code", pd.Series(dtype=object)).nunique()
        == expected_admins,
        "service_count": scores.get("service_id", pd.Series(dtype=object)).nunique()
        == expected_services,
        "admin_service_key_unique": not scores.duplicated(
            [column for column in ("admin_dong_code", "service_id") if column in scores]
        ).any()
        if {"admin_dong_code", "service_id"}.issubset(scores)
        else False,
    }
    numeric_columns = [
        column for column in ["specialty_gap_score", *component_columns, *formula_columns] if column in scores
    ]
    if numeric_columns:
        numeric = scores[numeric_columns].apply(pd.to_numeric, errors="coerce")
        checks["scores_complete_finite"] = (
            int(numeric.isna().sum().sum()) <= int(gates["score_missing_count_max"])
            and np.isfinite(numeric.to_numpy()).all()
        )
        checks["scores_bounded_0_100"] = bool(
            (numeric.to_numpy() >= -1e-9).all() and (numeric.to_numpy() <= 100 + 1e-9).all()
        )
    else:
        checks["scores_complete_finite"] = False
        checks["scores_bounded_0_100"] = False

    if all(column in scores for column in contribution_columns):
        contributions = scores[contribution_columns].apply(pd.to_numeric, errors="coerce")
        difference = (contributions.sum(axis=1) - scores["gap_score_weighted_additive"]).abs()
        checks["weighted_contributions_sum_to_additive"] = bool(difference.max() <= 1e-8)
        denominator = scores["gap_score_weighted_additive"].replace(0, np.nan)
        shares = contributions.div(denominator, axis=0).replace([np.inf, -np.inf], np.nan)
        p95 = float(shares.max(axis=1).quantile(0.95))
        checks["contribution_p95_share"] = p95 <= float(gates["contribution_p95_share_max"])
    else:
        p95 = float("nan")
        checks["weighted_contributions_sum_to_additive"] = False
        checks["contribution_p95_share"] = False

    forbidden_fragments = (
        "mediroad_need_score_v0",
        "recommended_specialties_v0",
        "recommended_specialty_scores_v0",
        "weighted_elderly_need_exposure_index_v0",
        "stage1_need_score",
        "legacy_gap",
    )
    leaked = [
        column for column in scores.columns if any(value in column for value in forbidden_fragments)
    ]
    checks["no_stage1_or_legacy_input_leakage"] = not leaked

    duplicate_pairs: list[str] = []
    if {"admin_dong_code", "service_id", "specialty_gap_score"}.issubset(scores):
        wide = scores.pivot(
            index="admin_dong_code", columns="service_id", values="specialty_gap_score"
        )
        names = list(wide.columns)
        for left_index, left in enumerate(names):
            for right in names[left_index + 1 :]:
                if np.array_equal(wide[left].to_numpy(), wide[right].to_numpy()):
                    duplicate_pairs.append(f"{left}|{right}")
    checks["exact_duplicate_service_score_pairs"] = len(duplicate_pairs) <= int(
        gates["exact_duplicate_service_score_pairs_max"]
    )

    canonical_sha = None
    if canonical_path is not None:
        canonical_sha = sha256_file(canonical_path)
        checks["canonical_sha_unchanged"] = (
            canonical_sha == str(config["source"]["frozen_canonical_sha256"])
        )

    bundle_summary: dict[str, Any] | None = None
    if bundle_scores is not None:
        expected_bundle_count = int(gates["expected_bundle_count"])
        expected_bundle_rows = int(gates["expected_bundle_rows"])
        bundle_checks = {
            "row_count": len(bundle_scores) == expected_bundle_rows,
            "bundle_count": bundle_scores.get("bundle_id", pd.Series(dtype=object)).nunique()
            == expected_bundle_count,
            "key_unique": not bundle_scores.duplicated(
                [column for column in ("admin_dong_code", "bundle_id") if column in bundle_scores]
            ).any()
            if {"admin_dong_code", "bundle_id"}.issubset(bundle_scores)
            else False,
            "score_complete": "bundle_gap_score" in bundle_scores
            and bundle_scores["bundle_gap_score"].notna().all()
            and np.isfinite(bundle_scores["bundle_gap_score"].to_numpy()).all(),
        }
        if bundle_config is not None:
            excluded = {
                str(value.get("service_id"))
                for value in bundle_config.get("excluded_from_mobile_bundles", [])
            }
            included = {
                str(service)
                for bundle in bundle_config.get("bundles", [])
                for service in bundle.get("included_services", {})
            }
            bundle_checks["excluded_services_absent"] = not bool(excluded & included)
        checks.update({f"bundle_{key}": value for key, value in bundle_checks.items()})
        bundle_summary = bundle_checks

    return {
        "valid": all(checks.values()),
        "checks": checks,
        "missing_columns": missing_columns,
        "leaked_columns": leaked,
        "exact_duplicate_service_score_pairs": duplicate_pairs,
        "contribution_p95_max_share": p95,
        "canonical_sha256": canonical_sha,
        "bundle_summary": bundle_summary,
        "summary": {
            "rows": len(scores),
            "admin_dongs": int(scores["admin_dong_code"].nunique())
            if "admin_dong_code" in scores
            else 0,
            "services": int(scores["service_id"].nunique()) if "service_id" in scores else 0,
        },
    }


def require_valid_specialty_results(
    scores: pd.DataFrame,
    config: Mapping[str, Any],
    **kwargs: Any,
) -> dict[str, Any]:
    result = validate_specialty_results(scores, config, **kwargs)
    if not result["valid"]:
        failed = [key for key, value in result["checks"].items() if not value]
        raise SpecialtyValidationError(f"Stage 2A result gate failed: {failed}")
    return result
