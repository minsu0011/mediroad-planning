"""Stage 2A regional specialty-service gap construction.

The score produced here is a relative policy gap.  It is not a patient-count,
attendance, diagnosis, or individual-demand prediction.  All joins use the
eight-character ``admin_dong_code``; names are descriptive labels only.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from mediroad.scoring.transform import scale_need_direction


COMPONENT_COLUMNS = (
    "health_context_gap_score",
    "age_structure_gap_score",
    "specialty_supply_gap_score",
    "specialty_excess_access_gap_score",
    "public_support_gap_score",
)

FORMULA_COLUMNS = {
    "weighted_additive": "gap_score_weighted_additive",
    "bounded_geometric": "gap_score_bounded_geometric",
    "minimum_bottleneck_hybrid": "gap_score_minimum_bottleneck_hybrid",
    "rank_composite": "gap_score_rank_composite",
}


def load_specialty_config(path: str | Path) -> dict[str, Any]:
    """Load the Stage 2A YAML contract and fail on a malformed top level."""

    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Specialty config must be a YAML mapping")
    from .validation import require_valid_specialty_config

    require_valid_specialty_config(config)
    return config


def _normalise_admin_code(series: pd.Series) -> pd.Series:
    values = series.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    if values.isna().any() or values.eq("").any():
        raise ValueError("admin_dong_code contains missing or blank values")
    return values


def _normalise_facility_id(series: pd.Series) -> pd.Series:
    values = series.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    if values.isna().any() or values.eq("").any():
        raise ValueError("facility_id contains missing or blank values")
    return values


def _numeric_complete(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        raise KeyError(f"Required specialty source column is missing: {column}")
    values = pd.to_numeric(frame[column], errors="coerce").astype(float)
    if values.isna().any():
        raise ValueError(f"{column} has {int(values.isna().sum())} non-numeric or missing values")
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError(f"{column} contains non-finite values")
    return values


def _scaler_options(config: Mapping[str, Any], scaler: str) -> dict[str, Any]:
    scalers = config.get("scalers", {})
    if scaler not in scalers:
        raise KeyError(f"Unknown configured scaler: {scaler}")
    value = scalers[scaler]
    if not isinstance(value, Mapping):
        raise ValueError(f"Scaler options must be a mapping: {scaler}")
    return dict(value)


def _scale(
    series: pd.Series,
    *,
    scaler: str,
    direction: int,
    options: Mapping[str, Any],
) -> pd.Series:
    return scale_need_direction(
        series,
        method=scaler,
        direction=direction,
        transform="identity",
        lower_quantile=float(options.get("lower_quantile", 0.01)),
        upper_quantile=float(options.get("upper_quantile", 0.99)),
        z_clip_lower=float(options.get("clip_lower", -3.0)),
        z_clip_upper=float(options.get("clip_upper", 3.0)),
        require_complete=True,
    )


def empirical_bayes_share(
    numerator: pd.Series | Sequence[float],
    denominator: pd.Series | Sequence[float],
    *,
    prior_strength: float,
) -> pd.Series:
    """Return a globally shrunk binomial share.

    The global share supplies the prior mean.  A locality with no general
    supply therefore receives the neutral global share rather than a fabricated
    zero.  General medical absence is already represented in Stage 1.
    """

    num = pd.Series(numerator, copy=False, dtype=float)
    den = pd.Series(denominator, copy=False, dtype=float)
    if len(num) != len(den):
        raise ValueError("EB numerator and denominator lengths differ")
    if not np.isfinite(num.to_numpy()).all() or not np.isfinite(den.to_numpy()).all():
        raise ValueError("EB share inputs must be finite")
    if (num < 0).any() or (den < 0).any():
        raise ValueError("EB share inputs must be non-negative")
    if (num > den + 1e-9).any():
        raise ValueError("EB numerator cannot exceed its general-supply denominator")
    strength = float(prior_strength)
    if not np.isfinite(strength) or strength <= 0:
        raise ValueError("EB prior_strength must be positive and finite")
    denominator_total = float(den.sum())
    if denominator_total <= 0:
        raise ValueError("EB denominator has no observed supply")
    prior_mean = float(num.sum() / denominator_total)
    result = (num + strength * prior_mean) / (den + strength)
    result.index = num.index
    return result.clip(0.0, 1.0)


def specialty_access_excess(
    specialty_minutes: pd.Series | Sequence[float],
    general_medical_minutes: pd.Series | Sequence[float],
) -> pd.Series:
    """Compute log1p specialty time minus log1p general-medical time."""

    specialty = pd.Series(specialty_minutes, copy=False, dtype=float)
    general = pd.Series(general_medical_minutes, copy=False, dtype=float)
    if len(specialty) != len(general):
        raise ValueError("Specialty and general travel-time lengths differ")
    values = np.concatenate([specialty.to_numpy(), general.to_numpy()])
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Travel-time inputs must be finite and non-negative")
    excess = np.log1p(specialty) - np.log1p(general)
    # The nearest-general set contains every specialty destination.  Tiny
    # negatives may only be floating-point noise; material negatives indicate
    # a broken destination or routing contract.
    if (excess < -1e-9).any():
        raise ValueError("Specialty travel time is materially below nearest-general time")
    excess = excess.clip(lower=0.0)
    excess.index = specialty.index
    return excess


def _validate_master(master: pd.DataFrame, expected_rows: int) -> pd.DataFrame:
    required = {
        "admin_dong_code",
        "admin_dong_name",
        "policy_sigungu_name",
        "population_65plus",
        "medical_facility_count_local",
        "medical_specialist_count_local",
        "osm_nearest_medical_any_drive_min_v6",
    }
    missing = sorted(required - set(master.columns))
    if missing:
        raise KeyError(f"Road master lacks required Stage 2A columns: {missing}")
    result = master.copy()
    result["admin_dong_code"] = _normalise_admin_code(result["admin_dong_code"])
    if len(result) != expected_rows:
        raise ValueError(f"Expected {expected_rows} admin dongs, found {len(result)}")
    if result["admin_dong_code"].duplicated().any():
        raise ValueError("Road master admin_dong_code is not unique")
    if result["policy_sigungu_name"].isna().any():
        raise ValueError("Road master has missing policy_sigungu_name")
    return result.sort_values("admin_dong_code", kind="stable").reset_index(drop=True)


def aggregate_hira_supply(
    master: pd.DataFrame,
    specialty_pivot: pd.DataFrame,
    facility_admin_join: pd.DataFrame,
    services: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    """Aggregate 15 physician specialties plus dental to 153 admin dongs.

    Physician institution and specialist counts come from the HIRA specialty
    pivot after an exact facility-ID spatial join.  Dental institution count is
    the audited ``dental_service_count_local`` master field, while dental-board
    specialists come from HIRA facility staffing because the generic ``치과``
    pivot flag does not represent all dental specialty departments.
    """

    if "facility_id" not in specialty_pivot or "facility_id" not in facility_admin_join:
        raise KeyError("Both HIRA inputs require facility_id")
    if "admin_dong_code" not in facility_admin_join:
        raise KeyError("HIRA facility-admin join lacks admin_dong_code")

    master_work = master.copy()
    master_work["admin_dong_code"] = _normalise_admin_code(master_work["admin_dong_code"])
    if master_work["admin_dong_code"].duplicated().any():
        raise ValueError("Master key is not unique during HIRA aggregation")

    pivot = specialty_pivot.copy()
    admin = facility_admin_join.copy()
    pivot["facility_id"] = _normalise_facility_id(pivot["facility_id"])
    admin["facility_id"] = _normalise_facility_id(admin["facility_id"])
    admin["admin_dong_code"] = _normalise_admin_code(admin["admin_dong_code"])
    if pivot["facility_id"].duplicated().any() or admin["facility_id"].duplicated().any():
        raise ValueError("HIRA facility_id must be unique in both source tables")

    admin_columns = ["facility_id", "admin_dong_code"]
    for column in ("medical_specialists", "dental_specialists"):
        if column not in admin.columns:
            raise KeyError(f"HIRA facility-admin join lacks {column}")
        admin[column] = pd.to_numeric(admin[column], errors="coerce").fillna(0.0).astype(float)
        admin_columns.append(column)

    merged = pivot.merge(
        admin[admin_columns],
        on="facility_id",
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    unmatched = int(merged["_merge"].ne("both").sum())
    if unmatched:
        raise ValueError(f"{unmatched} HIRA specialty facilities lack an admin-dong join")
    merged = merged.drop(columns="_merge")

    codes = master_work["admin_dong_code"]
    admin_index = pd.Index(codes, name="admin_dong_code")
    total_specialists = (
        admin.assign(
            __total_specialists=admin["medical_specialists"] + admin["dental_specialists"]
        )
        .groupby("admin_dong_code", sort=False)["__total_specialists"]
        .sum()
        .reindex(admin_index, fill_value=0.0)
    )
    total_facilities = pd.Series(
        _numeric_complete(master_work, "medical_facility_count_local").to_numpy(),
        index=admin_index,
    )

    base = master_work.set_index("admin_dong_code", drop=False)
    rows: list[pd.DataFrame] = []
    for service_order, service in enumerate(services, start=1):
        service_id = str(service["service_id"])
        name_ko = str(service["name_ko"])
        service_type = str(service["service_type"])
        if service_type == "focused_physician_specialty":
            presence_column = f"specialty_present__{name_ko}"
            specialist_column = f"specialist_count__{name_ko}"
            presence = _numeric_complete(merged, presence_column)
            specialist = _numeric_complete(merged, specialist_column)
            facility_count = (
                merged.assign(__value=presence)
                .groupby("admin_dong_code", sort=False)["__value"]
                .sum()
                .reindex(admin_index, fill_value=0.0)
            )
            specialist_count = (
                merged.assign(__value=specialist)
                .groupby("admin_dong_code", sort=False)["__value"]
                .sum()
                .reindex(admin_index, fill_value=0.0)
            )
            audited_column = f"specialty_{name_ko}_facility_count_local"
        elif service_id == "dental" and service_type == "adjunct_dental_service":
            audited_column = "dental_service_count_local"
            facility_count = pd.Series(
                _numeric_complete(master_work, audited_column).to_numpy(), index=admin_index
            )
            specialist_count = (
                admin.groupby("admin_dong_code", sort=False)["dental_specialists"]
                .sum()
                .reindex(admin_index, fill_value=0.0)
            )
        else:
            raise ValueError(f"Unsupported service contract: {service_id}/{service_type}")

        audited = pd.Series(
            _numeric_complete(master_work, audited_column).to_numpy(), index=admin_index
        )
        if not np.allclose(facility_count.to_numpy(), audited.to_numpy(), atol=0.0, rtol=0.0):
            maximum = float(np.max(np.abs(facility_count.to_numpy() - audited.to_numpy())))
            raise ValueError(
                f"HIRA facility aggregation disagrees with audited master for {service_id}; "
                f"max difference={maximum}"
            )
        if (facility_count > total_facilities + 1e-9).any():
            raise ValueError(f"{service_id} facility count exceeds local general facility supply")
        if (specialist_count > total_specialists + 1e-9).any():
            raise ValueError(f"{service_id} specialist count exceeds local total specialists")

        frame = pd.DataFrame(
            {
                "admin_dong_code": admin_index,
                "admin_dong_name": base.loc[admin_index, "admin_dong_name"].to_numpy(),
                "policy_sigungu_name": base.loc[admin_index, "policy_sigungu_name"].to_numpy(),
                "population_65plus": _numeric_complete(base.loc[admin_index], "population_65plus").to_numpy(),
                "service_order": service_order,
                "service_id": service_id,
                "service_name_ko": name_ko,
                "service_type": service_type,
                "mobile_service_role": str(service["mobile_service_role"]),
                "local_specialty_facility_count": facility_count.to_numpy(dtype=float),
                "local_specialty_specialist_count": specialist_count.to_numpy(dtype=float),
                "local_total_medical_facility_count": total_facilities.to_numpy(dtype=float),
                "local_total_specialist_count": total_specialists.to_numpy(dtype=float),
            }
        )
        rows.append(frame)

    result = pd.concat(rows, ignore_index=True)
    return result.sort_values(["service_order", "admin_dong_code"], kind="stable").reset_index(
        drop=True
    )


def _policy_context_scaled(
    master: pd.DataFrame,
    column: str,
    *,
    scaler: str,
    direction: int,
    options: Mapping[str, Any],
    expected_groups: int,
) -> pd.Series:
    values = _numeric_complete(master, column)
    context = pd.DataFrame(
        {
            "policy_sigungu_name": master["policy_sigungu_name"].to_numpy(),
            "value": values.to_numpy(),
        },
        index=master.index,
    )
    within_unique = context.groupby("policy_sigungu_name", sort=False)["value"].nunique(
        dropna=False
    )
    inconsistent = within_unique[within_unique.ne(1)]
    if not inconsistent.empty:
        raise ValueError(f"{column} varies within policy sigungu: {inconsistent.index.tolist()}")
    unique = context.drop_duplicates("policy_sigungu_name", keep="first").set_index(
        "policy_sigungu_name"
    )["value"]
    if len(unique) != expected_groups:
        raise ValueError(
            f"{column} must scale at n={expected_groups} policy sigungu, found {len(unique)}"
        )
    unique.name = column
    scaled = _scale(unique, scaler=scaler, direction=direction, options=options)
    mapped = master["policy_sigungu_name"].map(scaled)
    mapped.index = master.index
    return mapped.astype(float)


def _sigungu_supply_rate_gap(
    service_supply: pd.DataFrame,
    *,
    scaler: str,
    options: Mapping[str, Any],
    expected_groups: int,
) -> tuple[pd.Series, pd.Series]:
    grouped = service_supply.groupby("policy_sigungu_name", sort=False).agg(
        specialist_count=("local_specialty_specialist_count", "sum"),
        population_65plus=("population_65plus", "sum"),
    )
    if len(grouped) != expected_groups:
        raise ValueError(
            f"Sigungu specialist-rate scaling expected n={expected_groups}, found {len(grouped)}"
        )
    if (grouped["population_65plus"] <= 0).any():
        raise ValueError("Sigungu older population must be positive")
    rate = grouped["specialist_count"] / grouped["population_65plus"] * 1000.0
    rate.name = "sigungu_specialists_per_1000_elderly"
    gap = _scale(rate, scaler=scaler, direction=-1, options=options)
    mapped_rate = service_supply["policy_sigungu_name"].map(rate).astype(float)
    mapped_gap = service_supply["policy_sigungu_name"].map(gap).astype(float)
    mapped_rate.index = service_supply.index
    mapped_gap.index = service_supply.index
    return mapped_rate, mapped_gap


def combine_components(
    components: pd.DataFrame,
    weights: Mapping[str, float],
    *,
    formula: str,
    formula_options: Mapping[str, Any] | None = None,
) -> pd.Series:
    """Combine 0–100 gap components with one configured formula family."""

    columns = list(weights)
    missing = sorted(set(columns) - set(components.columns))
    if missing:
        raise KeyError(f"Formula components are missing: {missing}")
    values = components[columns].astype(float)
    array = values.to_numpy()
    if not np.isfinite(array).all() or (array < -1e-9).any() or (array > 100 + 1e-9).any():
        raise ValueError("Formula components must be finite and bounded to 0–100")
    weight = pd.Series(weights, dtype=float).reindex(columns)
    if (weight <= 0).any() or not np.isfinite(weight.to_numpy()).all():
        raise ValueError("Formula weights must be positive and finite")
    weight = weight / float(weight.sum())
    options = dict(formula_options or {})

    additive = values.mul(weight, axis=1).sum(axis=1)
    if formula == "weighted_additive":
        score = additive
    elif formula == "bounded_geometric":
        floor = float(options.get("floor", 5.0))
        if not 0 < floor <= 100:
            raise ValueError("bounded_geometric floor must be in (0, 100]")
        score = np.exp(np.log(values.clip(lower=floor)).mul(weight, axis=1).sum(axis=1))
    elif formula == "minimum_bottleneck_hybrid":
        additive_share = float(options.get("additive_share", 0.70))
        minimum_share = float(options.get("minimum_share", 0.30))
        if not np.isclose(additive_share + minimum_share, 1.0) or min(
            additive_share, minimum_share
        ) < 0:
            raise ValueError("Hybrid formula shares must be non-negative and sum to one")
        score = additive_share * additive + minimum_share * values.min(axis=1)
    elif formula == "rank_composite":
        ranks = values.rank(axis=0, method="average", pct=True) * 100.0
        score = ranks.mul(weight, axis=1).sum(axis=1)
    else:
        raise KeyError(f"Unsupported specialty formula: {formula}")
    return pd.Series(score, index=components.index, name=FORMULA_COLUMNS.get(formula, formula)).clip(
        0.0, 100.0
    )


def build_specialty_gap(
    master: pd.DataFrame,
    specialty_pivot: pd.DataFrame,
    facility_admin_join: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    scaler: str | None = None,
    primary_formula: str | None = None,
    facility_prior_strength: float | None = None,
    specialist_prior_strength: float | None = None,
) -> pd.DataFrame:
    """Build all 16 Stage 2A service scores and auditable contributions."""

    from .validation import require_valid_specialty_config

    require_valid_specialty_config(config)
    expected_rows = int(config["source"]["expected_admin_dongs"])
    expected_groups = int(config["source"]["expected_policy_sigungu"])
    master_work = _validate_master(master, expected_rows)
    if master_work["policy_sigungu_name"].nunique() != expected_groups:
        raise ValueError(
            f"Expected {expected_groups} policy sigungu, found "
            f"{master_work['policy_sigungu_name'].nunique()}"
        )
    services = list(config["services"])
    supply = aggregate_hira_supply(master_work, specialty_pivot, facility_admin_join, services)

    scaler_name = str(scaler or config["primary_scaler"])
    scaler_options = _scaler_options(config, scaler_name)
    primary = str(primary_formula or config["primary_formula"])
    if primary not in config["formula_families"]:
        raise KeyError(f"Primary formula is not configured: {primary}")

    # Cache features whose analysis unit is shared across services.  CHS is
    # fitted on exactly 11 policy-sigungu observations and only then broadcast.
    health_cache: dict[str, pd.Series] = {}
    age_cache: dict[str, pd.Series] = {}
    public_cache: dict[str, pd.Series] = {}
    for service in services:
        for column in service["health_features"]:
            if column not in health_cache:
                health_cache[column] = _policy_context_scaled(
                    master_work,
                    str(column),
                    scaler=scaler_name,
                    direction=1,
                    options=scaler_options,
                    expected_groups=expected_groups,
                )
        for column in service["age_features"]:
            if column not in age_cache:
                age_cache[column] = _scale(
                    _numeric_complete(master_work, str(column)),
                    scaler=scaler_name,
                    direction=1,
                    options=scaler_options,
                )
        public_column = str(service["public_support_column"])
        if public_column not in public_cache:
            public_cache[public_column] = _scale(
                _numeric_complete(master_work, public_column),
                scaler=scaler_name,
                direction=1,
                options=scaler_options,
            )

    # Stage 2A estimates a specialty *differential*, not another copy of the
    # common elderly/rural Need construct.  First assemble each service's raw
    # context fit, then remove the within-region mean across all services.
    # CHS-derived health deltas are re-scaled on the 11 independent policy
    # sigungu observations before broadcasting; age deltas remain native
    # admin-dong evidence.  Public-support distance is likewise expressed as
    # excess log travel time above nearest general medical care.
    health_raw_by_service: dict[str, pd.Series] = {}
    age_raw_by_service: dict[str, pd.Series] = {}
    public_raw_by_service: dict[str, pd.Series] = {}
    public_excess_by_service: dict[str, pd.Series] = {}
    general_minutes_all = _numeric_complete(
        master_work, "osm_nearest_medical_any_drive_min_v6"
    ).reset_index(drop=True)
    for service in services:
        service_id = str(service["service_id"])
        health_columns = [str(value) for value in service["health_features"]]
        age_columns = [str(value) for value in service["age_features"]]
        health_raw_by_service[service_id] = pd.concat(
            [health_cache[column].reset_index(drop=True) for column in health_columns],
            axis=1,
        ).mean(axis=1)
        age_raw_by_service[service_id] = pd.concat(
            [age_cache[column].reset_index(drop=True) for column in age_columns],
            axis=1,
        ).mean(axis=1)
        public_column = str(service["public_support_column"])
        public_raw = _numeric_complete(master_work, public_column).reset_index(drop=True)
        public_raw_by_service[service_id] = public_raw
        public_excess_by_service[service_id] = np.log1p(public_raw) - np.log1p(
            general_minutes_all
        )

    health_raw_matrix = pd.DataFrame(health_raw_by_service)
    age_raw_matrix = pd.DataFrame(age_raw_by_service)
    health_common = health_raw_matrix.mean(axis=1)
    age_common = age_raw_matrix.mean(axis=1)
    health_score_by_service: dict[str, pd.Series] = {}
    age_score_by_service: dict[str, pd.Series] = {}
    public_score_by_service: dict[str, pd.Series] = {}
    policy = master_work["policy_sigungu_name"].reset_index(drop=True)
    if policy.nunique() != expected_groups:
        raise ValueError("Policy context does not contain the configured 11 sigungu")
    for service in services:
        service_id = str(service["service_id"])
        health_delta = health_raw_matrix[service_id] - health_common
        policy_delta = pd.DataFrame(
            {"policy_sigungu_name": policy, "health_delta": health_delta}
        )
        if policy_delta.groupby("policy_sigungu_name")["health_delta"].nunique().gt(1).any():
            raise ValueError(
                f"Health differential for {service_id} is not constant within policy sigungu"
            )
        independent = (
            policy_delta.drop_duplicates("policy_sigungu_name")
            .sort_values("policy_sigungu_name", kind="stable")
            .reset_index(drop=True)
        )
        if len(independent) != expected_groups:
            raise ValueError(
                f"Health differential for {service_id} was not fitted on 11 sigungu"
            )
        scaled_independent = _scale(
            independent["health_delta"],
            scaler=scaler_name,
            direction=1,
            options=scaler_options,
        )
        health_score_by_service[service_id] = policy.map(
            dict(zip(independent["policy_sigungu_name"], scaled_independent))
        ).astype(float)
        age_score_by_service[service_id] = _scale(
            (age_raw_matrix[service_id] - age_common).rename("age_context_delta"),
            scaler=scaler_name,
            direction=1,
            options=scaler_options,
        )
        public_score_by_service[service_id] = _scale(
            public_excess_by_service[service_id].rename("public_support_excess_log1p"),
            scaler=scaler_name,
            direction=1,
            options=scaler_options,
        )

    component_weights = {str(k): float(v) for k, v in config["component_weights"].items()}
    score_weights = {
        "health_context_gap_score": component_weights["health_context_gap"],
        "age_structure_gap_score": component_weights["age_structure_gap"],
        "specialty_supply_gap_score": component_weights["specialty_supply_gap"],
        "specialty_excess_access_gap_score": component_weights[
            "specialty_excess_access_gap"
        ],
        "public_support_gap_score": component_weights["public_support_gap"],
    }
    supply_config = config["supply"]
    effective_facility_prior = float(
        supply_config["facility_share_prior_strength"]
        if facility_prior_strength is None
        else facility_prior_strength
    )
    effective_specialist_prior = float(
        supply_config["specialist_share_prior_strength"]
        if specialist_prior_strength is None
        else specialist_prior_strength
    )
    if min(effective_facility_prior, effective_specialist_prior) <= 0 or not np.isfinite(
        [effective_facility_prior, effective_specialist_prior]
    ).all():
        raise ValueError("Supply-prior strengths must be positive and finite")
    supply_weights = pd.Series(
        {
            "facility_share_gap_score": float(supply_config["local_facility_share_weight"]),
            "specialist_share_gap_score": float(
                supply_config["local_specialist_share_weight"]
            ),
            "sigungu_specialist_rate_gap_score": float(
                supply_config["sigungu_specialist_rate_weight"]
            ),
        }
    )
    supply_weights = supply_weights / float(supply_weights.sum())

    master_indexed = master_work.set_index("admin_dong_code", drop=False)
    service_frames: list[pd.DataFrame] = []
    for service in services:
        service_id = str(service["service_id"])
        service_supply = supply.loc[supply["service_id"].eq(service_id)].copy()
        service_supply = service_supply.sort_values("admin_dong_code", kind="stable").reset_index(
            drop=True
        )
        service_master = master_indexed.loc[service_supply["admin_dong_code"]].reset_index(drop=True)

        facility_share = empirical_bayes_share(
            service_supply["local_specialty_facility_count"],
            service_supply["local_total_medical_facility_count"],
            prior_strength=effective_facility_prior,
        )
        specialist_share = empirical_bayes_share(
            service_supply["local_specialty_specialist_count"],
            service_supply["local_total_specialist_count"],
            prior_strength=effective_specialist_prior,
        )
        facility_gap = _scale(
            facility_share.rename("eb_local_facility_share"),
            scaler=scaler_name,
            direction=-1,
            options=scaler_options,
        )
        specialist_gap = _scale(
            specialist_share.rename("eb_local_specialist_share"),
            scaler=scaler_name,
            direction=-1,
            options=scaler_options,
        )
        # A composition share is undefined when its total supply is zero.  The
        # EB estimate remains available for audit, but the composition-gap
        # score is explicitly neutral; Stage 1 already handles general absence.
        facility_gap = facility_gap.mask(
            service_supply["local_total_medical_facility_count"].eq(0), 50.0
        )
        specialist_gap = specialist_gap.mask(
            service_supply["local_total_specialist_count"].eq(0), 50.0
        )
        sigungu_rate, sigungu_rate_gap = _sigungu_supply_rate_gap(
            service_supply,
            scaler=scaler_name,
            options=scaler_options,
            expected_groups=expected_groups,
        )

        health_columns = [str(value) for value in service["health_features"]]
        age_columns = [str(value) for value in service["age_features"]]
        health_score = health_score_by_service[service_id]
        age_score = age_score_by_service[service_id]

        specialty_column = (
            "osm_nearest_dental_drive_min_v6"
            if service_id == "dental"
            else f"osm_nearest_specialty_{service['name_ko']}_drive_min_v6"
        )
        specialty_minutes = _numeric_complete(service_master, specialty_column)
        general_minutes = _numeric_complete(
            service_master, "osm_nearest_medical_any_drive_min_v6"
        )
        access_excess = specialty_access_excess(specialty_minutes, general_minutes)
        access_gap = _scale(
            access_excess.rename("specialty_access_excess_log1p"),
            scaler=scaler_name,
            direction=1,
            options=scaler_options,
        )
        general_access_gap = _scale(
            general_minutes.rename("general_medical_drive_min_v6"),
            scaler=scaler_name,
            direction=1,
            options=scaler_options,
        )
        public_column = str(service["public_support_column"])
        public_raw = _numeric_complete(service_master, public_column)
        public_gap = public_score_by_service[service_id].reset_index(drop=True)

        service_supply["eb_local_facility_share"] = facility_share.to_numpy()
        service_supply["eb_local_specialist_share"] = specialist_share.to_numpy()
        service_supply["sigungu_specialists_per_1000_elderly"] = sigungu_rate.to_numpy()
        service_supply["facility_share_gap_score"] = facility_gap.to_numpy()
        service_supply["specialist_share_gap_score"] = specialist_gap.to_numpy()
        service_supply["sigungu_specialist_rate_gap_score"] = sigungu_rate_gap.to_numpy()
        service_supply["specialty_supply_gap_score"] = (
            service_supply[list(supply_weights.index)].mul(supply_weights, axis=1).sum(axis=1)
        )
        service_supply["health_context_raw_score"] = health_raw_by_service[
            service_id
        ].to_numpy()
        service_supply["health_context_common_score"] = health_common.to_numpy()
        service_supply["health_context_differential_raw"] = (
            health_raw_matrix[service_id] - health_common
        ).to_numpy()
        service_supply["health_context_gap_score"] = health_score.to_numpy()
        service_supply["age_structure_raw_score"] = age_raw_by_service[
            service_id
        ].to_numpy()
        service_supply["age_structure_common_score"] = age_common.to_numpy()
        service_supply["age_structure_differential_raw"] = (
            age_raw_matrix[service_id] - age_common
        ).to_numpy()
        service_supply["age_structure_gap_score"] = age_score.to_numpy()
        service_supply["specialty_drive_min_v6"] = specialty_minutes.to_numpy()
        service_supply["general_medical_drive_min_v6"] = general_minutes.to_numpy()
        service_supply["specialty_access_excess_log1p"] = access_excess.to_numpy()
        service_supply["specialty_excess_access_gap_score"] = access_gap.to_numpy()
        service_supply["general_medical_access_gap_score"] = general_access_gap.to_numpy()
        service_supply["public_support_column"] = public_column
        service_supply["public_support_raw"] = public_raw.to_numpy()
        service_supply["public_support_raw_gap_score"] = public_cache[
            public_column
        ].reset_index(drop=True).to_numpy()
        service_supply["public_support_excess_log1p"] = public_excess_by_service[
            service_id
        ].to_numpy()
        service_supply["public_support_gap_score"] = public_gap.to_numpy()
        service_supply["health_feature_ids"] = "|".join(health_columns)
        service_supply["health_evidence"] = str(service["health_evidence"])
        service_supply["age_feature_ids"] = "|".join(age_columns)
        service_supply["public_support_evidence"] = str(service["public_support_evidence"])
        service_supply["scaler"] = scaler_name
        service_supply["facility_share_prior_strength"] = effective_facility_prior
        service_supply["specialist_share_prior_strength"] = effective_specialist_prior

        for formula, options in config["formula_families"].items():
            formula_name = str(formula)
            service_supply[FORMULA_COLUMNS[formula_name]] = combine_components(
                service_supply[list(score_weights)],
                score_weights,
                formula=formula_name,
                formula_options=options,
            )
        service_supply["primary_formula"] = primary
        service_supply["specialty_gap_score"] = service_supply[FORMULA_COLUMNS[primary]]
        for component, weight in score_weights.items():
            contribution_name = f"contribution_{component.removesuffix('_score')}"
            service_supply[contribution_name] = service_supply[component] * float(weight)
        service_frames.append(service_supply)

    result = pd.concat(service_frames, ignore_index=True)
    result = result.sort_values(["service_order", "admin_dong_code"], kind="stable").reset_index(
        drop=True
    )
    result["specialty_gap_rank_tied"] = result.groupby("service_id", sort=False)[
        "specialty_gap_score"
    ].rank(method="min", ascending=False)
    result["specialty_gap_score_tie_size"] = result.groupby(
        ["service_id", "specialty_gap_score"], sort=False
    )["admin_dong_code"].transform("size")
    result["specialty_gap_rank"] = 0
    for _, indices in result.groupby("service_id", sort=False).groups.items():
        ordered = result.loc[list(indices)].sort_values(
            ["specialty_gap_score", "admin_dong_code"],
            ascending=[False, True],
            kind="stable",
        )
        result.loc[ordered.index, "specialty_gap_rank"] = np.arange(1, len(ordered) + 1)
    result["specialty_gap_rank"] = result["specialty_gap_rank"].astype(int)
    result["specialty_gap_rank_tied"] = result["specialty_gap_rank_tied"].astype(int)
    result["specialty_gap_score_tie_size"] = result["specialty_gap_score_tie_size"].astype(int)
    result["rank_tiebreaker"] = "specialty_gap_score_desc_then_admin_dong_code_asc"
    result["score_target"] = str(config["target_definition"])
    result["model_version"] = str(config["version"])
    return result


def build_specialty_gap_from_root(
    package_root: str | Path,
    *,
    config_path: str | Path | None = None,
    scaler: str | None = None,
    primary_formula: str | None = None,
    facility_prior_strength: float | None = None,
    specialist_prior_strength: float | None = None,
) -> pd.DataFrame:
    """Load only approved Stage 2A inputs from a package and build scores."""

    root = Path(package_root).resolve()
    path = Path(config_path) if config_path is not None else root / "configs/model_v1/specialty_gap.yaml"
    if not path.is_absolute():
        path = root / path
    config = load_specialty_config(path)
    source = config["source"]
    master = pd.read_csv(
        root / str(source["road_master"]),
        dtype={"admin_dong_code": "string"},
        low_memory=False,
    )
    pivot = pd.read_csv(root / str(source["hira_specialty_pivot"]), low_memory=False)
    admin = pd.read_csv(
        root / str(source["hira_facility_admin_join"]),
        dtype={"admin_dong_code": "string"},
        low_memory=False,
    )
    # Stage 1 score and legacy V0 gap are deliberately not loaded.  They are
    # downstream context and a comparison baseline, never Stage 2A inputs.
    return build_specialty_gap(
        master,
        pivot,
        admin,
        config,
        scaler=scaler,
        primary_formula=primary_formula,
        facility_prior_strength=facility_prior_strength,
        specialist_prior_strength=specialist_prior_strength,
    )
