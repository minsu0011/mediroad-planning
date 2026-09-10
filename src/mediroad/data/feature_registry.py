"""Feature-registry utilities for MEDIROAD MODEL V1 Stage 1.

The frozen V6 audit describes 525 columns, while the road-integrated master has
24 additional OSM columns (549 total).  This module deliberately joins the audit
by column name and classifies the appended road columns itself, so an omitted
audit row can never silently become a model input.

Only the 27 fields declared in ``configs/model_v1/stage1_features.yaml`` are
Stage-1 inputs.  Historical outreach, V0 outputs, identifiers/coordinates,
candidate-venue/optimizer fields, potential-beneficiary exposure fields, and
specialty/capacity fields are excluded by deterministic guardrails.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import re
from typing import Any

import pandas as pd


AXIS_FROM_AUDIT = {
    "demographic_vulnerability": "A_demographic_vulnerability",
    "health_burden_context": "B_health_burden",
    "medical_supply_access": "C_medical_supply_access_gap",
    "transport_access": "D_transport_access_gap",
    "equity_outreach": "E_isolation_equity",
}

CORE_AXES = tuple(AXIS_FROM_AUDIT.values())

REQUIRED_CONFIG_FEATURE_FIELDS = {
    "feature_id",
    "label_ko",
    "label_en",
    "axis",
    "direction",
    "granularity",
    "proxy",
    "semantic_group",
    "transform",
    "unit",
    "value_type",
    "missing_semantics",
    "confidence",
    "description",
}

REQUIRED_MANIFEST_FIELDS = (
    "feature_id",
    "source_column",
    "label_ko",
    "label_en",
    "axis",
    "role",
    "granularity",
    "direction",
    "unit",
    "value_type",
    "missing_semantics",
    "proxy",
    "model_input",
    "semantic_group",
    "transform",
    "confidence",
    "ablation_scope",
    "status",
    "not_applicable_reason",
    "description",
)

OUTREACH_RE = re.compile(
    r"(?:known_(?:mobile_clinic|rural_bus)|previous_outreach|previous_mobile_clinic|"
    r"previous_rural_bus|candidate_previous_operation|known_service_overlap|"
    r"hira_home_visit_overlap|known_hira_home_visit|operationally_(?:proven|verified))",
    flags=re.IGNORECASE,
)

OUTCOME_RE = re.compile(
    r"(?:actual_(?:mobile_clinic_)?demand|mobile_clinic_participant|patient_(?:count|demand)|"
    r"actual_demand_label)",
    flags=re.IGNORECASE,
)

LEGACY_OUTPUT_RE = re.compile(
    r"(?:_score_v0$|need_score_(?:rank|decile)_v0$|mediroad_need_score_v0$|"
    r"weighted_elderly_need_exposure_index_v0$|recommended_specialt(?:y|ies).*_v0$|"
    r"score_(?:version|target)$|need_score_v0_unchanged)",
    flags=re.IGNORECASE,
)

IDENTIFIER_RE = re.compile(
    r"(?:^admin_dong_code$|^legal_dong_code$|^admin_dong_name$|^policy_sigungu_name$|"
    r"(?:^|_)(?:event_ids?|node_id|facility_name|center_name)(?:_|$)|"
    r"nearest_.+_(?:name|facility_name|facility_type)$)",
    flags=re.IGNORECASE,
)

COORDINATE_RE = re.compile(
    r"(?:^centroid_(?:lon|lat)$|(?:^|_)(?:longitude|latitude)$|"
    r"nearest_.+_facility_(?:lon|lat)$|(?:^|_)coord(?:inate)?(?:_|$))",
    flags=re.IGNORECASE,
)

TEMPORAL_METADATA_RE = re.compile(
    r"(?:reference_date|reference_ym|snapshot_date|latest_year|population_date|"
    r"hira_capacity_reference_date|data_freeze_version|data_release|^score_version$)",
    flags=re.IGNORECASE,
)

VENUE_OPTIMIZER_RE = re.compile(
    r"(?:^candidate_|candidate_venue|candidate_location|^venue_type_count__|"
    r"venue_count|official_village_hall|bus_stop_name_proxy_venue|"
    r"dementia_branch_count|field_verification)",
    flags=re.IGNORECASE,
)

FINE_GRID_RE = re.compile(
    r"(?:^elderly_grid_|fine_grid_beneficiary|potential_beneficiary)",
    flags=re.IGNORECASE,
)

SPECIALTY_RE = re.compile(
    r"(?:specialty_|nearest_(?:내과|가정의학과|정형외과|재활의학과|신경과|안과|"
    r"산부인과|정신건강의학과|응급의학과|이비인후과|외과|신경외과|"
    r"마취통증의학과|비뇨의학과|피부과))",
    flags=re.IGNORECASE,
)

CAPACITY_RE = re.compile(
    r"(?:_beds?$|bed_|^equipment_|^allied_staff_|^special_care_|"
    r"inpatient_capable|facility_count__(?:CT|MRI|골밀도|양전자|유방|종양|체외|초음파|콘빔|혈액)|"
    r"nearest_(?:CT|MRI|골밀도|양전자|유방|종양|체외|초음파|콘빔|혈액|"
    r"primary_care_home_visit|korean_medicine_home_visit|home_nursing|"
    r"disability_home_visit|dementia_home_visit|chronic_disease_management|"
    r"emergency_medical_institution|rehabilitation_medical_institution)|"
    r"rehabilitation_allied_staff)",
    flags=re.IGNORECASE,
)

EXTERNAL_CONTEXT_RE = re.compile(
    r"(?:^nhis_|^mobility_support_|dementia_safety_center|healthmap|cancer_cases)",
    flags=re.IGNORECASE,
)

CURRENT_SNAPSHOT_RE = re.compile(r"(?:^molit_|alternative_transport)", flags=re.IGNORECASE)

ROAD_QUALITY_RE = re.compile(
    r"(?:^osm_origin_|^road_network_|road_network_time_definition)",
    flags=re.IGNORECASE,
)


def _require_nonblank(value: Any, field: str) -> None:
    if value is None or (isinstance(value, str) and not value.strip()):
        raise ValueError(f"Configuration field '{field}' must not be blank")


def _validate_config(config: Mapping[str, Any]) -> None:
    for key in (
        "schema_version",
        "model_version",
        "stage",
        "axes",
        "features",
        "substitutions",
        "weight_profiles",
        "scalers",
    ):
        if key not in config:
            raise ValueError(f"Missing top-level configuration key: {key}")

    axes = config["axes"]
    if not isinstance(axes, Mapping) or set(axes) != set(CORE_AXES):
        raise ValueError(f"axes must contain exactly {list(CORE_AXES)}")

    features = config["features"]
    if not isinstance(features, list):
        raise TypeError("features must be a list")
    expected = int(config.get("feature_selection", {}).get("expected_active_feature_count", 25))
    if len(features) != expected:
        raise ValueError(f"Expected {expected} active features, found {len(features)}")

    ids: list[str] = []
    for index, feature in enumerate(features):
        if not isinstance(feature, Mapping):
            raise TypeError(f"features[{index}] must be a mapping")
        missing = REQUIRED_CONFIG_FEATURE_FIELDS - set(feature)
        if missing:
            raise ValueError(f"features[{index}] is missing fields: {sorted(missing)}")
        for field in REQUIRED_CONFIG_FEATURE_FIELDS - {"proxy"}:
            _require_nonblank(feature[field], f"features[{index}].{field}")
        if not isinstance(feature["proxy"], bool):
            raise TypeError(f"features[{index}].proxy must be boolean")
        if feature["axis"] not in axes:
            raise ValueError(f"Unknown axis for {feature['feature_id']}: {feature['axis']}")
        if feature["direction"] not in {"+", "-"}:
            raise ValueError(f"Direction must be '+' or '-' for {feature['feature_id']}")
        ids.append(str(feature["feature_id"]))
    if len(ids) != len(set(ids)):
        duplicates = sorted({x for x in ids if ids.count(x) > 1})
        raise ValueError(f"Duplicate active feature_id values: {duplicates}")

    substitutions = config["substitutions"]
    if not isinstance(substitutions, list) or not substitutions:
        raise ValueError("substitutions must be a nonempty list")
    required_substitution_fields = {"selected_feature", "shadow_feature", "direction", "transform", "reason"}
    shadow_ids: list[str] = []
    for index, substitution in enumerate(substitutions):
        if not isinstance(substitution, Mapping):
            raise TypeError(f"substitutions[{index}] must be a mapping")
        missing = required_substitution_fields - set(substitution)
        if missing:
            raise ValueError(f"substitutions[{index}] is missing fields: {sorted(missing)}")
        for field in required_substitution_fields:
            _require_nonblank(substitution[field], f"substitutions[{index}].{field}")
        if substitution["selected_feature"] not in ids:
            raise ValueError(
                f"substitutions[{index}].selected_feature is not active: {substitution['selected_feature']}"
            )
        if substitution["shadow_feature"] in ids:
            raise ValueError(
                f"substitutions[{index}].shadow_feature is already active: {substitution['shadow_feature']}"
            )
        if substitution["direction"] not in {"+", "-"}:
            raise ValueError(f"Invalid substitution direction at index {index}")
        shadow_ids.append(str(substitution["shadow_feature"]))
    if len(shadow_ids) != len(set(shadow_ids)):
        duplicates = sorted({x for x in shadow_ids if shadow_ids.count(x) > 1})
        raise ValueError(f"Duplicate shadow_feature values: {duplicates}")

    profiles = config["weight_profiles"]
    if not isinstance(profiles, Mapping) or len(profiles) != 5:
        raise ValueError("Exactly five weight profiles are required")
    for name, profile in profiles.items():
        weights = profile.get("weights") if isinstance(profile, Mapping) else None
        if not isinstance(weights, Mapping) or set(weights) != set(axes):
            raise ValueError(f"Weight profile '{name}' must define every axis exactly once")
        values = [float(weights[axis]) for axis in axes]
        if any(value < 0 for value in values) or abs(sum(values) - 1.0) > 1e-9:
            raise ValueError(f"Weight profile '{name}' must contain nonnegative weights summing to 1")

    scalers = config["scalers"]
    if not isinstance(scalers, Mapping) or len(scalers) != 4:
        raise ValueError("Exactly four scaler definitions are required")
    default_profile = config.get("feature_selection", {}).get("default_weight_profile")
    default_scaler = config.get("feature_selection", {}).get("default_scaler")
    if default_profile not in profiles:
        raise ValueError(f"Unknown default_weight_profile: {default_profile}")
    if default_scaler not in scalers:
        raise ValueError(f"Unknown default_scaler: {default_scaler}")


def load_feature_config(path: str | Path) -> dict[str, Any]:
    """Load and validate the Stage-1 YAML configuration.

    Parameters
    ----------
    path:
        Path to ``stage1_features.yaml``.

    Returns
    -------
    dict
        A validated configuration mapping.
    """

    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - environment diagnostic
        raise RuntimeError("PyYAML is required to load the feature registry configuration") from exc

    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Feature configuration not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, Mapping):
        raise TypeError("Feature configuration root must be a mapping")
    config = dict(config)
    _validate_config(config)
    return config


def _as_dataframe(value: pd.DataFrame | str | Path, *, kind: str) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    path = Path(value)
    if not path.is_file():
        raise FileNotFoundError(f"{kind} file not found: {path}")
    dtype = {"admin_dong_code": "string"} if kind == "master" else {"column_name": "string"}
    return pd.read_csv(path, dtype=dtype, low_memory=False)


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _audit_text(row: Mapping[str, Any], key: str, fallback: str) -> str:
    value = row.get(key)
    if value is None or pd.isna(value) or not str(value).strip():
        return fallback
    return str(value).strip()


def _humanize_identifier(column: str) -> str:
    return re.sub(r"\s+", " ", column.replace("__", " / ").replace("_", " ")).strip()


def _generic_labels(column: str) -> tuple[str, str]:
    explicit = {
        "admin_dong_code": ("행정동 코드", "Administrative-dong code"),
        "admin_dong_name": ("행정동명", "Administrative-dong name"),
        "policy_sigungu_name": ("정책 시군명", "Policy sigungu name"),
        "legal_dong_code": ("법정동 코드", "Legal-dong code"),
        "centroid_lon": ("행정동 중심 경도", "Administrative-dong centroid longitude"),
        "centroid_lat": ("행정동 중심 위도", "Administrative-dong centroid latitude"),
        "area_km2": ("행정동 면적", "Administrative-dong area"),
        "population_total": ("총인구", "Total population"),
        "road_network_status_v6": ("V6 도로망 상태", "V6 road-network status"),
        "road_network_time_definition_v6": ("V6 도로시간 정의", "V6 road-time definition"),
    }
    if column in explicit:
        return explicit[column]

    match = re.fullmatch(r"osm_nearest_(.+)_drive_min_v6", column)
    if match:
        target = match.group(1).replace("specialty_", "진료과_").replace("_", " ")
        return f"OSM 최근접 {target} 정적 주행시간", f"OSM static drive time to nearest {match.group(1)}"
    match = re.fullmatch(r"nearest_(.+)_drive_min_proxy", column)
    if match:
        target = match.group(1).replace("_", " ")
        return f"최근접 {target} 주행시간 근사치", f"Approximate drive time to nearest {match.group(1)}"
    match = re.fullmatch(r"nearest_(.+)_straight_km(?:_v5)?", column)
    if match:
        target = match.group(1).replace("_", " ")
        return f"최근접 {target} 직선거리", f"Straight-line distance to nearest {match.group(1)}"
    match = re.fullmatch(r"specialty_(.+)_facility_count_local", column)
    if match:
        return f"읍면동 내 {match.group(1)} 진료기관 수", f"Local {match.group(1)} facility count"
    match = re.fullmatch(r"equipment_units__(.+)", column)
    if match:
        return f"{match.group(1)} 장비 대수", f"{match.group(1)} equipment units"
    match = re.fullmatch(r"equipment_facility_count__(.+)", column)
    if match:
        return f"{match.group(1)} 보유기관 수", f"Facilities with {match.group(1)}"
    match = re.fullmatch(r"allied_staff_count__(.+)", column)
    if match:
        return f"{match.group(1)} 인력 수", f"{match.group(1)} staff count"
    match = re.fullmatch(r"allied_staff_facility_count__(.+)", column)
    if match:
        return f"{match.group(1)} 배치기관 수", f"Facilities with {match.group(1)} staff"
    match = re.fullmatch(r"venue_type_count__(.+)", column)
    if match:
        return f"후보장소 유형 수: {match.group(1)}", f"Venue type count: {match.group(1)}"
    match = re.fullmatch(r"special_care_facility_count__(.+)", column)
    if match:
        return f"특수서비스 기관 수: {match.group(1)}", f"Special-care facility count: {match.group(1)}"

    human = _humanize_identifier(column)
    return f"원천 필드: {human}", human


def _axis_for(column: str, audit_row: Mapping[str, Any]) -> str:
    if column.startswith("osm_nearest_"):
        return "C_medical_supply_access_gap"
    audit_axis = _audit_text(audit_row, "feature_axis_v6", "N_A")
    return AXIS_FROM_AUDIT.get(audit_axis, "N_A")


def _granularity_for(column: str, audit_row: Mapping[str, Any]) -> str:
    if _to_bool(audit_row.get("sigungu_repeated_granularity_flag")) or re.search(
        r"(?:^chs65p_|_sigungu(?:_|$))", column, flags=re.IGNORECASE
    ):
        return "policy_sigungu_repeated_on_admin_dong"
    if column.startswith("osm_nearest_"):
        return "admin_dong_centroid_to_hira_facility_network"
    if column.startswith("osm_origin_"):
        return "admin_dong_centroid_to_osm_node"
    if FINE_GRID_RE.search(column):
        return "admin_dong_aggregate_of_100m_grid"
    if VENUE_OPTIMIZER_RE.search(column):
        return "admin_dong_aggregate_of_candidate_locations"
    if OUTREACH_RE.search(column) or OUTCOME_RE.search(column):
        return "admin_dong_public_event_evidence"
    if TEMPORAL_METADATA_RE.search(column) or column.startswith("road_network_"):
        return "package_or_source_metadata"
    if re.search(r"^nearest_", column):
        return "admin_dong_centroid_to_nearest_target"
    return "admin_dong"


def _unit_and_type(column: str, series: pd.Series) -> tuple[str, str]:
    lower = column.lower()
    dtype = series.dtype
    if IDENTIFIER_RE.search(column):
        return "identifier_or_name", "identifier_or_text"
    if COORDINATE_RE.search(column):
        return "decimal_degrees_or_coordinate", "spatial_coordinate"
    if re.search(r"(?:date|reference_ym|snapshot|latest_year|release|version)$", lower):
        return "date_or_version", "temporal_or_version_metadata"
    if "per_1000" in lower or "per_10000" in lower:
        denominator = "10000" if "per_10000" in lower else "1000"
        return f"rate_per_{denominator}", "continuous_rate"
    if lower.endswith("_pct") or "_pct_" in lower:
        return "percent", "continuous"
    if lower.endswith("_km2") and "density" not in lower and "per_km2" not in lower:
        return "square_kilometres", "continuous"
    if "per_km2" in lower or "density" in lower:
        return "count_per_square_kilometre", "continuous_rate"
    if lower.endswith("_km") or "straight_km" in lower or "distance_km" in lower:
        return "kilometres", "continuous"
    if "drive_min" in lower or lower.endswith("_min"):
        return "minutes", "continuous"
    if lower.endswith("_hour"):
        return "hour_of_day", "continuous_time"
    if lower.endswith("_sec"):
        return "seconds_after_midnight", "continuous_time"
    if lower.endswith("_flag") or lower.startswith("has_") or lower.endswith("_available") or lower.endswith("_valid_v6"):
        return "binary_flag", "binary"
    if lower.endswith("_ratio") or "_ratio_" in lower or lower.endswith("_share") or "_share_" in lower:
        return "proportion", "continuous"
    if "entropy" in lower or "hhi" in lower or "index" in lower or "score" in lower:
        return "index_or_score", "continuous_index"
    if "rank" in lower or "decile" in lower:
        return "rank", "ordinal"
    if (
        lower.endswith("_count")
        or "_count_" in lower
        or lower.endswith("_sum")
        or "_sum_" in lower
        or "population_" in lower
        or "households_" in lower
        or "_beds" in lower
        or "_units" in lower
    ):
        return "count", "count"
    if pd.api.types.is_bool_dtype(dtype) or series.dropna().nunique() <= 2 and pd.api.types.is_numeric_dtype(dtype):
        return "binary_or_two_level", "binary"
    if pd.api.types.is_integer_dtype(dtype):
        return "source_defined_count_or_integer", "integer"
    if pd.api.types.is_numeric_dtype(dtype):
        return "source_defined_numeric", "continuous"
    return "source_defined_text", "categorical_or_text"


def _semantic_group_for(column: str, axis: str, role: str) -> str:
    lower = column.lower()
    if role == "legacy_output_leakage":
        return "legacy_generated_output"
    if role == "outcome_label_leakage":
        return "actual_or_post_event_outcome"
    if role == "historical_validation_only":
        return "historical_outreach_evidence"
    if role == "identifier_join_only":
        return "identifier_or_join_key"
    if role == "spatial_mapping_only":
        return "spatial_coordinate_or_node"
    if role == "optimizer_venue_only":
        return "candidate_venue_or_optimizer"
    if role == "future_exposure_only":
        return "potential_beneficiary_exposure"
    if role == "future_specialty_capacity_only":
        return "specialty_or_capacity"
    if role == "external_validation_only":
        return "external_health_or_mobility_context"
    if role in {"quality_metadata_only", "temporal_metadata_only"}:
        return "data_quality_or_provenance"
    if lower.startswith("chs65p_"):
        return "health_burden_context"
    if axis == "A_demographic_vulnerability":
        if "single_household" in lower or "1person" in lower:
            return "household_isolation"
        if "density" in lower or lower == "area_km2":
            return "settlement_structure"
        return "population_structure"
    if axis == "B_health_burden":
        return "health_burden_context"
    if axis == "C_medical_supply_access_gap":
        return "medical_access" if re.search(r"nearest_|drive_min|straight_km", lower) else "medical_supply"
    if axis == "D_transport_access_gap":
        return "transport_access" if "nearest" in lower else "scheduled_transport_supply"
    if axis == "E_isolation_equity":
        return "equity_or_isolation_context"
    return "other_context_or_metadata"


def _direction_for(column: str, axis: str, role: str) -> str:
    lower = column.lower()
    if role in {
        "legacy_output_leakage",
        "outcome_label_leakage",
        "historical_validation_only",
        "identifier_join_only",
        "spatial_mapping_only",
        "temporal_metadata_only",
        "optimizer_venue_only",
        "future_exposure_only",
        "external_validation_only",
        "quality_metadata_only",
        "context_only",
    }:
        return "N_A"
    if re.search(r"nearest_|drive_min|straight_km|distance", lower):
        return "+"
    if "first_departure" in lower:
        return "+"
    if "last_departure" in lower:
        return "-"
    if axis == "B_health_burden":
        return "+"
    if axis == "C_medical_supply_access_gap":
        return "-"
    if axis == "D_transport_access_gap":
        return "-"
    if axis == "A_demographic_vulnerability":
        return "-" if "density" in lower else "+"
    if axis == "E_isolation_equity":
        return "-" if "density" in lower else "+"
    return "N_A"


def _missing_semantics_for(
    column: str,
    role: str,
    missing_count: int,
    all_null: bool,
) -> str:
    lower = column.lower()
    if all_null:
        return "not_collected_all_null; preserve_na"
    if OUTREACH_RE.search(column) or "alternative_transport" in lower:
        return "unknown_is_not_zero; preserve_non_disclosure"
    if column.startswith("osm_nearest_"):
        return (
            "complete_after_road_reachability_gate"
            if missing_count == 0
            else "unreachable_or_graph_quality_failure; preserve_na_and_block_model"
        )
    if re.search(r"nearest_.+_(?:name|facility_type|facility_name)$", column):
        return "not_available_when_no_matched_target; preserve_na"
    if missing_count > 0:
        return "source_missing_unknown_or_not_applicable; preserve_na"
    if re.search(r"(?:_count|_sum|_beds|_units)(?:_|$)", lower):
        return "complete; observed_zero_is_zero_records"
    return "complete_no_missing"


def _confidence_for(
    role: str,
    audit_row: Mapping[str, Any],
    proxy: bool,
    missing_count: int,
    sigungu_repeated: bool,
) -> str:
    if role in {"legacy_output_leakage", "outcome_label_leakage"}:
        return "not_applicable_excluded_leakage"
    if role in {
        "identifier_join_only",
        "spatial_mapping_only",
        "temporal_metadata_only",
        "quality_metadata_only",
    }:
        return "not_applicable_metadata"
    if missing_count > 0:
        return "low_or_conditional_missingness"
    if sigungu_repeated:
        return "medium_group_context"
    quality = _audit_text(audit_row, "quality_status_v6", "NOT_AUDITED_POST_ROAD_APPEND")
    if proxy:
        return "medium_proxy"
    if quality.startswith("COMPLETE"):
        return "high_observed_aggregate"
    if role in {"external_validation_only", "context_only"}:
        return "medium_context_only"
    return "medium_reviewed"


def _classification_for(column: str, audit_row: Mapping[str, Any]) -> dict[str, Any]:
    legacy_flag = _to_bool(audit_row.get("legacy_output_leakage_flag"))
    direct = _audit_text(audit_row, "direct_score_eligibility_v6", "NOT_IN_FROZEN_AUDIT")
    prior_role = _audit_text(audit_row, "prior_data_role", "NOT_IN_FROZEN_AUDIT")

    if legacy_flag or LEGACY_OUTPUT_RE.search(column):
        return {
            "role": "legacy_output_leakage",
            "status": "EXCLUDED_LEAKAGE",
            "ablation_scope": "not_applicable",
            "reason": "V0 score/rank/recommendation or generated output; prohibited as a V1 input",
        }
    if OUTCOME_RE.search(column):
        return {
            "role": "outcome_label_leakage",
            "status": "EXCLUDED_OUTCOME_LEAKAGE",
            "ablation_scope": "historical_validation_only",
            "reason": "actual/post-event demand or participant information; prohibited as a Need Score input",
        }
    if OUTREACH_RE.search(column):
        return {
            "role": "historical_validation_only",
            "status": "EXCLUDED_HISTORICAL_OUTREACH",
            "ablation_scope": "historical_backtest_or_optimizer_overlap_only",
            "reason": "past outreach evidence is reserved for backtest or overlap penalty and is not ground truth",
        }
    if ROAD_QUALITY_RE.search(column):
        return {
            "role": "quality_metadata_only",
            "status": "EXCLUDED_ROAD_QUALITY_METADATA",
            "ablation_scope": "road_quality_gate_only",
            "reason": "OSM node/snap/status metadata is used for the road quality gate, not scoring",
        }
    if IDENTIFIER_RE.search(column) or direct == "EXCLUDE_IDENTIFIER":
        return {
            "role": "identifier_join_only",
            "status": "EXCLUDED_IDENTIFIER",
            "ablation_scope": "not_applicable",
            "reason": "identifier/name field retained only for joins and reporting",
        }
    if COORDINATE_RE.search(column):
        return {
            "role": "spatial_mapping_only",
            "status": "EXCLUDED_COORDINATE",
            "ablation_scope": "not_applicable",
            "reason": "coordinate or snapped-node metadata retained only for mapping/routing quality control",
        }
    if TEMPORAL_METADATA_RE.search(column):
        return {
            "role": "temporal_metadata_only",
            "status": "EXCLUDED_TEMPORAL_METADATA",
            "ablation_scope": "not_applicable",
            "reason": "reference date/version metadata is not a policy-need feature",
        }
    if VENUE_OPTIMIZER_RE.search(column) or direct == "OPTIMIZER_OR_VENUE_SELECTION_ONLY":
        return {
            "role": "optimizer_venue_only",
            "status": "DEFERRED_VENUE_OPTIMIZER",
            "ablation_scope": "future_venue_or_optimizer_stage",
            "reason": "candidate-location or venue-supply field is deferred to venue scoring and optimization",
        }
    if FINE_GRID_RE.search(column) or direct == "COVERAGE_OR_ABLATION_REQUIRED":
        return {
            "role": "future_exposure_only",
            "status": "DEFERRED_POTENTIAL_EXPOSURE",
            "ablation_scope": "future_exposure_sensitivity_only",
            "reason": "100m calibrated elderly-grid field is reserved for Potential Beneficiary Exposure",
        }
    if SPECIALTY_RE.search(column) or CAPACITY_RE.search(column):
        return {
            "role": "future_specialty_capacity_only",
            "status": "DEFERRED_SPECIALTY_CAPACITY",
            "ablation_scope": "future_specialty_or_capacity_stage",
            "reason": "specialty, equipment, bed, allied-staff, or special-care detail is deferred to Specialty Gap/Capacity",
        }
    if column.startswith("osm_nearest_"):
        return {
            "role": "redundant_not_selected",
            "status": "EXCLUDED_NOT_SELECTED_STAGE1",
            "ablation_scope": "representative_substitution_only",
            "reason": "non-specialty OSM access field was not selected after broad-access redundancy review",
        }
    if EXTERNAL_CONTEXT_RE.search(column) or "external_validation" in prior_role:
        return {
            "role": "external_validation_only",
            "status": "EXCLUDED_EXTERNAL_VALIDATION",
            "ablation_scope": "external_validation_only",
            "reason": "NHIS/HealthMap/mobility context is reserved for external validation or grouped sensitivity",
        }
    if CURRENT_SNAPSHOT_RE.search(column) or direct in {
        "CONTEXT_ONLY_CURRENT_SNAPSHOT",
        "CONTEXT_UNKNOWN_NOT_ZERO",
    }:
        return {
            "role": "context_only",
            "status": "EXCLUDED_CONTEXT_ONLY",
            "ablation_scope": "context_or_freshness_check_only",
            "reason": "current snapshot/partial evidence cannot replace GTFS service intensity or unknown with zero",
        }
    if direct in {"EXCLUDE_CONSTANT_METADATA", "CONTEXT_ONLY_TEXT"} or prior_role in {
        "quality_flag",
        "data_governance",
        "context_or_metadata",
        "road_network_status",
    }:
        return {
            "role": "quality_metadata_only",
            "status": "EXCLUDED_METADATA",
            "ablation_scope": "not_applicable",
            "reason": "constant, text, quality, or governance metadata is retained for provenance only",
        }
    if direct == "GROUP_LEVEL_CONTEXT_OR_GROUPED_MODEL":
        return {
            "role": "redundant_not_selected",
            "status": "EXCLUDED_GROUP_CONTEXT_NOT_SELECTED",
            "ablation_scope": "representative_substitution_only",
            "reason": "group-level context was not selected among the strict Stage-1 representatives",
        }
    if direct in {"CANDIDATE_INPUT_REVIEW_CORRELATION", "ELIGIBLE_PROXY_WITH_GUARDRAIL"}:
        return {
            "role": "redundant_not_selected",
            "status": "EXCLUDED_NOT_SELECTED_STAGE1",
            "ablation_scope": "representative_substitution_only",
            "reason": "eligible source feature excluded after semantic and correlation-based representative selection",
        }
    return {
        "role": "context_only",
        "status": "EXCLUDED_UNCLASSIFIED_CONTEXT",
        "ablation_scope": "manual_review_only",
        "reason": "not selected for Stage 1 and lacks evidence for safe automatic scoring",
    }


def build_feature_manifest(
    master: pd.DataFrame | str | Path,
    audit: pd.DataFrame | str | Path,
    config: Mapping[str, Any] | str | Path,
) -> pd.DataFrame:
    """Build a complete, leakage-guarded manifest for every master column.

    ``master`` and ``audit`` may be DataFrames or CSV paths. ``config`` may be a
    validated mapping or YAML path. The returned row order exactly matches the
    master column order.
    """

    master_df = _as_dataframe(master, kind="master")
    audit_df = _as_dataframe(audit, kind="audit")
    if isinstance(config, (str, Path)):
        config_dict = load_feature_config(config)
    elif isinstance(config, Mapping):
        config_dict = dict(config)
        _validate_config(config_dict)
    else:
        raise TypeError("config must be a mapping or YAML path")

    if master_df.columns.duplicated().any():
        duplicates = master_df.columns[master_df.columns.duplicated()].tolist()
        raise ValueError(f"Master contains duplicate columns: {duplicates}")
    if "column_name" not in audit_df.columns:
        raise ValueError("Audit must contain a column_name field")
    if audit_df["column_name"].duplicated().any():
        duplicates = audit_df.loc[audit_df["column_name"].duplicated(), "column_name"].tolist()
        raise ValueError(f"Audit contains duplicate column_name rows: {duplicates}")

    expected_rows = int(config_dict.get("source", {}).get("expected_admin_dong_rows", len(master_df)))
    if len(master_df) != expected_rows:
        raise ValueError(f"Expected {expected_rows} master rows, found {len(master_df)}")

    configured_features = {str(item["feature_id"]): dict(item) for item in config_dict["features"]}
    missing_configured = sorted(set(configured_features) - set(master_df.columns))
    if missing_configured:
        raise ValueError(f"Configured Stage-1 features missing from master: {missing_configured}")

    substitution_lookup: dict[str, dict[str, Any]] = {}
    selected_to_shadows: dict[str, list[str]] = {}
    for substitution in config_dict["substitutions"]:
        shadow = str(substitution["shadow_feature"])
        selected = str(substitution["selected_feature"])
        substitution_lookup[shadow] = dict(substitution)
        selected_to_shadows.setdefault(selected, []).append(shadow)
    missing_shadows = sorted(set(substitution_lookup) - set(master_df.columns))
    if missing_shadows:
        raise ValueError(f"Configured substitution shadows missing from master: {missing_shadows}")

    audit_lookup = audit_df.set_index("column_name", drop=False).to_dict(orient="index")
    rows: list[dict[str, Any]] = []

    for order, column in enumerate(master_df.columns, start=1):
        series = master_df[column]
        audit_row = audit_lookup.get(column, {})
        non_null_count = int(series.notna().sum())
        missing_count = int(series.isna().sum())
        unique_count = int(series.nunique(dropna=True))
        all_null = non_null_count == 0
        constant = unique_count <= 1
        sigungu_repeated = _to_bool(audit_row.get("sigungu_repeated_granularity_flag"))

        if column in configured_features:
            feature = configured_features[column]
            label_ko = str(feature["label_ko"])
            label_en = str(feature["label_en"])
            axis = str(feature["axis"])
            role = "model_input"
            granularity = str(feature["granularity"])
            direction = str(feature["direction"])
            unit = str(feature["unit"])
            value_type = str(feature["value_type"])
            missing_semantics = str(feature["missing_semantics"])
            proxy = bool(feature["proxy"])
            model_input = True
            semantic_group = str(feature["semantic_group"])
            transform = str(feature["transform"])
            confidence = str(feature["confidence"])
            ablation_scope = "stage1_feature_leave_one_out_and_axis_leave_one_out"
            status = "ACTIVE" if missing_count == 0 and not constant else "ACTIVE_BLOCKED_QUALITY_GATE"
            not_applicable_reason = "N_A_ACTIVE_STAGE1_MODEL_INPUT"
            description = str(feature["description"])
            substitution_for = "N_A_NOT_SHADOW_FEATURE"
            shadow_candidates = "|".join(selected_to_shadows.get(column, [])) or "N_A_NO_CONFIGURED_SHADOW"
        elif column in substitution_lookup:
            substitution = substitution_lookup[column]
            selected_id = str(substitution["selected_feature"])
            selected_feature = configured_features[selected_id]
            label_ko, label_en = _generic_labels(column)
            axis = str(selected_feature["axis"])
            role = "shadow_substitution"
            granularity = _granularity_for(column, audit_row)
            unit, value_type = _unit_and_type(column, series)
            proxy = _to_bool(audit_row.get("proxy_or_approximation_flag")) or bool(
                "_drive_min_proxy" in column or "_straight_km" in column
            )
            direction = str(substitution["direction"])
            missing_semantics = _missing_semantics_for(column, role, missing_count, all_null)
            model_input = False
            semantic_group = str(selected_feature["semantic_group"])
            transform = str(substitution["transform"])
            confidence = _confidence_for(role, audit_row, proxy, missing_count, sigungu_repeated)
            ablation_scope = f"one_at_a_time_substitution_for:{selected_id}"
            status = "SHADOW_SUBSTITUTION_ONLY"
            not_applicable_reason = (
                f"Not a simultaneous input; replace {selected_id} only during substitution ablation"
            )
            description = str(substitution["reason"])
            substitution_for = selected_id
            shadow_candidates = "N_A_SHADOW_FEATURE"
        else:
            classification = _classification_for(column, audit_row)
            label_ko, label_en = _generic_labels(column)
            axis = _axis_for(column, audit_row)
            role = str(classification["role"])
            granularity = _granularity_for(column, audit_row)
            unit, value_type = _unit_and_type(column, series)
            proxy = _to_bool(audit_row.get("proxy_or_approximation_flag")) or bool(
                column.startswith("osm_nearest_") or "_drive_min_proxy" in column or "_straight_km" in column
            )
            direction = _direction_for(column, axis, role)
            missing_semantics = _missing_semantics_for(column, role, missing_count, all_null)
            model_input = False
            semantic_group = _semantic_group_for(column, axis, role)
            transform = "none_not_stage1_model_input"
            confidence = _confidence_for(role, audit_row, proxy, missing_count, sigungu_repeated)
            ablation_scope = str(classification["ablation_scope"])
            status = str(classification["status"])
            not_applicable_reason = str(classification["reason"])
            warning = _audit_text(audit_row, "description_or_warning", "no frozen-audit description")
            if warning == "Defined in V4 data dictionary.":
                warning = "retained V4 field reviewed in the V6 column audit"
            description = f"{not_applicable_reason}; {warning}"
            substitution_for = "N_A_NOT_SHADOW_FEATURE"
            shadow_candidates = "N_A_NO_CONFIGURED_SHADOW"

        source_dataset = _audit_text(
            audit_row,
            "source_dataset",
            "OSM_ROAD_NETWORK_V6_POST_AUDIT_APPEND" if column.startswith("osm_") or column == "road_network_time_definition_v6" else "MASTER_COLUMN_NOT_IN_FROZEN_AUDIT",
        )
        audit_coverage = "AUDITED_FROZEN_525" if column in audit_lookup else "POST_AUDIT_OSM_APPEND"

        rows.append(
            {
                "manifest_order": order,
                "feature_id": column,
                "source_column": column,
                "label_ko": label_ko,
                "label_en": label_en,
                "axis": axis,
                "role": role,
                "granularity": granularity,
                "direction": direction,
                "unit": unit,
                "value_type": value_type,
                "missing_semantics": missing_semantics,
                "proxy": proxy,
                "model_input": model_input,
                "semantic_group": semantic_group,
                "transform": transform,
                "confidence": confidence,
                "ablation_scope": ablation_scope,
                "status": status,
                "not_applicable_reason": not_applicable_reason,
                "description": description,
                "substitution_for": substitution_for,
                "shadow_candidates": shadow_candidates,
                "source_dataset": source_dataset,
                "audit_coverage": audit_coverage,
                "dtype_observed": str(series.dtype),
                "non_null_count": non_null_count,
                "missing_count": missing_count,
                "missing_pct": round(missing_count / len(master_df) * 100.0, 6),
                "unique_count": unique_count,
                "constant_flag": constant,
                "all_null_flag": all_null,
                "sigungu_repeated_granularity_flag": sigungu_repeated,
                "legacy_output_leakage_flag": _to_bool(audit_row.get("legacy_output_leakage_flag")),
                "audit_quality_status": _audit_text(audit_row, "quality_status_v6", "N_A_POST_AUDIT_APPEND"),
                "audit_direct_score_eligibility": _audit_text(
                    audit_row, "direct_score_eligibility_v6", "N_A_POST_AUDIT_APPEND"
                ),
                "audit_recommended_treatment": _audit_text(
                    audit_row, "recommended_v1_treatment_v6", "N_A_POST_AUDIT_APPEND"
                ),
            }
        )

    manifest = pd.DataFrame(rows)
    string_columns = [
        column
        for column in manifest.columns
        if column not in {
            "manifest_order",
            "non_null_count",
            "missing_count",
            "missing_pct",
            "unique_count",
            "proxy",
            "model_input",
            "constant_flag",
            "all_null_flag",
            "sigungu_repeated_granularity_flag",
            "legacy_output_leakage_flag",
        }
    ]
    for column in string_columns:
        manifest[column] = manifest[column].fillna("N_A_UNSPECIFIED").astype(str)
        manifest.loc[manifest[column].str.strip().eq(""), column] = "N_A_UNSPECIFIED"
    manifest.attrs["expected_active_feature_count"] = int(
        config_dict.get("feature_selection", {}).get("expected_active_feature_count", 27)
    )
    return manifest


def _forbidden_active_reason(column: str, role: str) -> str | None:
    if role != "model_input":
        return f"role_is_{role}"
    if LEGACY_OUTPUT_RE.search(column) or "_v0" in column.lower():
        return "legacy_or_v0_output"
    if OUTCOME_RE.search(column):
        return "actual_or_post_event_outcome"
    if OUTREACH_RE.search(column):
        return "historical_outreach"
    if IDENTIFIER_RE.search(column):
        return "identifier_or_name"
    if COORDINATE_RE.search(column):
        return "coordinate"
    if VENUE_OPTIMIZER_RE.search(column):
        return "candidate_venue_or_optimizer"
    if FINE_GRID_RE.search(column):
        return "future_potential_exposure"
    if SPECIALTY_RE.search(column) or CAPACITY_RE.search(column):
        return "future_specialty_or_capacity"
    return None


def validate_feature_manifest(
    manifest: pd.DataFrame,
    master: pd.DataFrame | str | Path,
) -> dict[str, Any]:
    """Validate manifest completeness, active-set quality, and leakage rules.

    Returns a serializable dictionary. ``valid`` is false for structural or
    leakage failures. ``ready_for_model`` additionally requires every active
    feature to be nonconstant and complete; this intentionally blocks use of an
    old road-integrated master with unreachable OSM values.
    """

    if not isinstance(manifest, pd.DataFrame):
        raise TypeError("manifest must be a pandas DataFrame")
    master_df = _as_dataframe(master, kind="master")
    errors: list[str] = []
    blockers: list[str] = []
    warnings: list[str] = []

    missing_fields = [field for field in REQUIRED_MANIFEST_FIELDS if field not in manifest.columns]
    if missing_fields:
        errors.append(f"missing required manifest fields: {missing_fields}")

    if len(master_df.columns) != 549:
        errors.append(f"road-integrated master must have 549 columns, found {len(master_df.columns)}")
    if len(manifest) != len(master_df.columns):
        errors.append(f"manifest row count {len(manifest)} != master column count {len(master_df.columns)}")

    if "feature_id" in manifest:
        duplicated_ids = manifest.loc[manifest["feature_id"].duplicated(), "feature_id"].astype(str).tolist()
        if duplicated_ids:
            errors.append(f"duplicate feature_id values: {duplicated_ids}")
    if "source_column" in manifest:
        manifest_columns = manifest["source_column"].astype(str).tolist()
        master_columns = [str(column) for column in master_df.columns]
        missing_columns = sorted(set(master_columns) - set(manifest_columns))
        extra_columns = sorted(set(manifest_columns) - set(master_columns))
        if missing_columns:
            errors.append(f"master columns missing from manifest: {missing_columns}")
        if extra_columns:
            errors.append(f"manifest columns absent from master: {extra_columns}")
        if not missing_columns and not extra_columns and manifest_columns != master_columns:
            warnings.append("manifest covers all columns but does not preserve master column order")

    blank_by_field: dict[str, int] = {}
    for field in REQUIRED_MANIFEST_FIELDS:
        if field not in manifest:
            continue
        series = manifest[field]
        if field in {"proxy", "model_input"}:
            blank_count = int(series.isna().sum())
        else:
            blank_count = int(series.isna().sum() + series.fillna("").astype(str).str.strip().eq("").sum())
        if blank_count:
            blank_by_field[field] = blank_count
    if blank_by_field:
        errors.append(f"blank required manifest values: {blank_by_field}")

    if "audit_coverage" in manifest:
        audit_counts = manifest["audit_coverage"].value_counts(dropna=False).to_dict()
        if audit_counts.get("AUDITED_FROZEN_525", 0) != 525:
            errors.append(f"expected 525 frozen-audited fields, found {audit_counts.get('AUDITED_FROZEN_525', 0)}")
        if audit_counts.get("POST_AUDIT_OSM_APPEND", 0) != 24:
            errors.append(f"expected 24 post-audit OSM fields, found {audit_counts.get('POST_AUDIT_OSM_APPEND', 0)}")
    else:
        errors.append("manifest is missing audit_coverage")

    if "model_input" in manifest:
        active = manifest[manifest["model_input"].map(_to_bool)].copy()
    else:
        active = manifest.iloc[0:0].copy()
    expected_active = int(manifest.attrs.get("expected_active_feature_count", 27))
    if len(active) != expected_active:
        errors.append(
            f"expected {expected_active} Stage-1 model inputs, found {len(active)}"
        )

    axis_counts = active["axis"].value_counts().sort_index().to_dict() if "axis" in active else {}
    missing_axes = [axis for axis in CORE_AXES if axis_counts.get(axis, 0) == 0]
    if missing_axes:
        errors.append(f"active set does not cover all five axes: {missing_axes}")
    unexpected_axes = sorted(set(axis_counts) - set(CORE_AXES))
    if unexpected_axes:
        errors.append(f"active set contains non-core axes: {unexpected_axes}")

    invalid_directions = (
        active.loc[~active["direction"].isin(["+", "-"]), "source_column"].astype(str).tolist()
        if "direction" in active
        else []
    )
    if invalid_directions:
        errors.append(f"active features with invalid direction: {invalid_directions}")

    forbidden_active: dict[str, str] = {}
    if {"source_column", "role"}.issubset(active.columns):
        for row in active[["source_column", "role"]].itertuples(index=False):
            reason = _forbidden_active_reason(str(row.source_column), str(row.role))
            if reason:
                forbidden_active[str(row.source_column)] = reason
    if forbidden_active:
        errors.append(f"forbidden active features: {forbidden_active}")

    active_missing: dict[str, int] = {}
    active_constant: list[str] = []
    for column in active.get("source_column", pd.Series(dtype=str)).astype(str):
        if column not in master_df:
            continue
        missing_count = int(master_df[column].isna().sum())
        if missing_count:
            active_missing[column] = missing_count
        if int(master_df[column].nunique(dropna=True)) <= 1:
            active_constant.append(column)
    if active_missing:
        blockers.append(f"active features contain semantic/quality missingness: {active_missing}")
    if active_constant:
        blockers.append(f"active features are constant or all-null: {active_constant}")

    if "legacy_output_leakage_flag" in active and active["legacy_output_leakage_flag"].map(_to_bool).any():
        errors.append("an audit-flagged legacy output is active")
    if "status" in active:
        blocked_status = active.loc[active["status"].ne("ACTIVE"), "source_column"].astype(str).tolist()
        if blocked_status:
            blockers.append(f"active features failing manifest quality status: {blocked_status}")

    structural_valid = not errors
    ready_for_model = structural_valid and not blockers
    return {
        "valid": structural_valid,
        "ready_for_model": ready_for_model,
        "errors": errors,
        "blockers": blockers,
        "warnings": warnings,
        "summary": {
            "master_rows": int(len(master_df)),
            "master_columns": int(len(master_df.columns)),
            "manifest_rows": int(len(manifest)),
            "active_feature_count": int(len(active)),
            "excluded_or_deferred_count": int(len(manifest) - len(active)),
            "active_proxy_count": int(active["proxy"].map(_to_bool).sum()) if "proxy" in active else 0,
            "active_sigungu_context_count": int(
                active["granularity"].astype(str).str.contains("sigungu", case=False, na=False).sum()
            )
            if "granularity" in active
            else 0,
            "active_missing_feature_count": int(len(active_missing)),
            "active_constant_feature_count": int(len(active_constant)),
            "blank_required_field_counts": blank_by_field,
        },
        "axis_counts": {str(key): int(value) for key, value in axis_counts.items()},
        "status_counts": {
            str(key): int(value) for key, value in manifest["status"].value_counts().sort_index().items()
        }
        if "status" in manifest
        else {},
        "role_counts": {
            str(key): int(value) for key, value in manifest["role"].value_counts().sort_index().items()
        }
        if "role" in manifest
        else {},
        "active_missing": active_missing,
        "active_constant": active_constant,
        "forbidden_active": forbidden_active,
    }


__all__ = [
    "load_feature_config",
    "build_feature_manifest",
    "validate_feature_manifest",
]
