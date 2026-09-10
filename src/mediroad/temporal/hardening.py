"""Strict, abstention-friendly temporal validation for MEDIROAD Stage 2B.

This module deliberately lives beside, rather than inside, the frozen Stage 2
pipeline.  It evaluates the two observed NHIS years without treating the 153
admin-dong delivery rows as independent observations.  The statistical unit is
the policy-sigungu/service-bundle pair (11 x 5 = 55 in the production data).

The input is observed insured utilisation.  Nothing in this module estimates
mobile-clinic demand, patient counts, an exact month, or an exact date.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence, TypeVar

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr


OBSERVED_YEARS = (2022, 2023)
SEASON_ORDER = ("spring", "summer", "autumn", "winter")
PRIMARY_SEASON_DEFINITION: dict[str, tuple[int, ...]] = {
    "spring": (3, 4, 5),
    "summer": (6, 7, 8),
    "autumn": (9, 10, 11),
    "winter": (12, 1, 2),
}
SEASON_BOUNDARY_DEFINITIONS: dict[str, dict[str, tuple[int, ...]]] = {
    "primary_meteorological": PRIMARY_SEASON_DEFINITION,
    "calendar_quarters": {
        "spring": (1, 2, 3),
        "summer": (4, 5, 6),
        "autumn": (7, 8, 9),
        "winter": (10, 11, 12),
    },
    "shifted_plus_one_month": {
        "spring": (4, 5, 6),
        "summer": (7, 8, 9),
        "autumn": (10, 11, 12),
        "winter": (1, 2, 3),
    },
}
PRIMARY_VALUE_COLUMN = "working_day_adjusted"
SENSITIVITY_VALUE_COLUMNS = (
    PRIMARY_VALUE_COLUMN,
    "raw_monthly_utilization",
    "year_normalized_working_day",
    "within_year_z_working_day",
    "within_year_percentile_working_day",
    "annual_share_working_day",
)
TEMPORAL_INDEPENDENT_UNIT = "policy_sigungu_x_service_bundle"
RELEASE_TYPES = (
    "STRONG_SINGLE",
    "ROBUST_PAIR",
    "NO_STRONG_PREFERENCE",
)
CONFIDENCE_LEVELS = ("HIGH", "MODERATE", "LOW")

# These defaults are declared before looking at bundle outcomes.  Callers may
# supply another pre-registered mapping; the evidence tables remain unchanged.
DEFAULT_RELEASE_THRESHOLDS: dict[str, float] = {
    "strong_unit_top1_agreement_min": 0.65,
    "strong_pair_coverage_min": 0.85,
    "strong_regret_max": 0.15,
    "strong_bootstrap_primary_min": 0.70,
    "strong_lomo_primary_min": 0.80,
    "strong_loso_primary_min": 0.80,
    "strong_margin_normalized_min": 0.10,
    "strong_permutation_strength_min": 0.80,
    "pair_coverage_min": 0.75,
    "pair_regret_max": 0.25,
    "pair_bootstrap_top2_min": 0.65,
    "pair_lomo_top2_min": 0.75,
    "pair_loso_top2_min": 0.75,
}


@dataclass(frozen=True)
class TemporalHardeningEvidence:
    """Normalized service evidence and derived sigungu-bundle evidence."""

    service_month: pd.DataFrame
    bundle_month: pd.DataFrame
    audit: dict[str, Any]


@dataclass(frozen=True)
class CrossYearTransferResult:
    detail: pd.DataFrame
    summary: pd.DataFrame


@dataclass(frozen=True)
class DiagnosticResult:
    detail: pd.DataFrame
    summary: pd.DataFrame


@dataclass(frozen=True)
class TemporalHardeningResult:
    evidence: TemporalHardeningEvidence
    cross_year: CrossYearTransferResult
    consensus: pd.DataFrame
    lomo: DiagnosticResult
    working_day_sensitivity: pd.DataFrame
    loso: DiagnosticResult
    season_boundary: DiagnosticResult
    bootstrap: DiagnosticResult
    permutation: DiagnosticResult
    bundle_evidence_summary: pd.DataFrame


def _require_columns(frame: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise KeyError(f"{label} lacks required columns: {missing}")


def _strict_two_year_contract(
    frame: pd.DataFrame,
    years: Sequence[int] = OBSERVED_YEARS,
) -> tuple[int, int]:
    configured = tuple(int(year) for year in years)
    if len(configured) != 2 or len(set(configured)) != 2:
        raise ValueError("Temporal hardening requires exactly two distinct train/holdout years")
    observed = tuple(sorted(pd.to_numeric(frame["year"], errors="raise").astype(int).unique()))
    if observed != tuple(sorted(configured)):
        raise ValueError(
            f"Two-year leakage guard rejected observed years {observed}; expected {tuple(sorted(configured))}"
        )
    return tuple(sorted(configured))  # type: ignore[return-value]


def _validate_positive_integer(value: int, label: str) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return int(value)


def _validate_n_jobs(n_jobs: int) -> int:
    if isinstance(n_jobs, bool) or int(n_jobs) != n_jobs or int(n_jobs) <= 0:
        raise ValueError("n_jobs must be a positive integer")
    return int(n_jobs)


def _bundle_weights(bundle_config: Mapping[str, Any]) -> tuple[list[str], list[str], np.ndarray]:
    bundles = list(bundle_config.get("bundles", []))
    if not bundles:
        raise ValueError("Service bundle config contains no bundles")
    bundle_ids = [str(bundle["bundle_id"]) for bundle in bundles]
    if len(bundle_ids) != len(set(bundle_ids)):
        raise ValueError("Service bundle IDs are not unique")
    service_ids = sorted(
        {
            str(service_id)
            for bundle in bundles
            for service_id in dict(bundle.get("included_services", {}))
        }
    )
    weights = np.zeros((len(bundle_ids), len(service_ids)), dtype=float)
    service_index = {service_id: index for index, service_id in enumerate(service_ids)}
    for bundle_index, bundle in enumerate(bundles):
        included = {
            str(service_id): float(weight)
            for service_id, weight in dict(bundle.get("included_services", {})).items()
        }
        if not included or not np.isclose(sum(included.values()), 1.0, atol=1e-9):
            raise ValueError(f"Bundle {bundle_ids[bundle_index]} weights must sum to one")
        if any(not np.isfinite(weight) or weight < 0 for weight in included.values()):
            raise ValueError(f"Bundle {bundle_ids[bundle_index]} has invalid service weights")
        for service_id, weight in included.items():
            weights[bundle_index, service_index[service_id]] = weight
    return bundle_ids, service_ids, weights


def _season_matrix(definition: Mapping[str, Sequence[int]]) -> np.ndarray:
    if tuple(definition) != SEASON_ORDER:
        raise ValueError(f"Season definition keys must be ordered as {SEASON_ORDER}")
    flat = [int(month) for season in SEASON_ORDER for month in definition[season]]
    if sorted(flat) != list(range(1, 13)) or len(flat) != 12:
        raise ValueError("A season definition must partition months 1..12 exactly once")
    matrix = np.zeros((4, 12), dtype=float)
    for season_index, season in enumerate(SEASON_ORDER):
        months = tuple(int(month) for month in definition[season])
        matrix[season_index, np.asarray(months, dtype=int) - 1] = 1.0 / len(months)
    return matrix


def _ordered_indices(values: Sequence[float]) -> np.ndarray:
    scores = np.asarray(values, dtype=float)
    if scores.shape != (4,) or not np.isfinite(scores).all():
        raise ValueError("A season profile must contain four finite scores")
    return np.lexsort((np.arange(4), -scores))


def _rank_vector(values: Sequence[float]) -> np.ndarray:
    order = _ordered_indices(values)
    ranks = np.empty(4, dtype=int)
    ranks[order] = np.arange(1, 5)
    return ranks


def _safe_correlations(left: Sequence[float], right: Sequence[float]) -> tuple[float, float]:
    left_values = np.asarray(left, dtype=float)
    right_values = np.asarray(right, dtype=float)
    if not np.isfinite(left_values).all() or not np.isfinite(right_values).all():
        raise ValueError("Rank correlation inputs must be finite")
    left_flat = np.ptp(left_values) <= 1e-15
    right_flat = np.ptp(right_values) <= 1e-15
    if left_flat or right_flat:
        value = 1.0 if left_flat and right_flat and np.allclose(left_values, right_values) else 0.0
        return value, value
    rho = float(spearmanr(left_values, right_values).statistic)
    tau = float(kendalltau(left_values, right_values).statistic)
    return (rho if np.isfinite(rho) else 0.0, tau if np.isfinite(tau) else 0.0)


def _profile(values: Sequence[float]) -> dict[str, Any]:
    scores = np.asarray(values, dtype=float)
    order = _ordered_indices(scores)
    ranks = _rank_vector(scores)
    score_range = float(np.ptp(scores))
    top1_margin = float(scores[order[0]] - scores[order[1]])
    top2_margin = float(scores[order[1]] - scores[order[2]])
    denominator = score_range if score_range > 1e-15 else 1.0
    return {
        "scores": scores,
        "order": order,
        "ranks": ranks,
        "primary": SEASON_ORDER[int(order[0])],
        "fallback": SEASON_ORDER[int(order[1])],
        "top2": {SEASON_ORDER[int(order[0])], SEASON_ORDER[int(order[1])]},
        "range": score_range,
        "top1_margin": top1_margin,
        "top2_margin": top2_margin,
        "top1_margin_normalized": top1_margin / denominator,
        "top2_margin_normalized": top2_margin / denominator,
    }


T = TypeVar("T")


def _parallel_map(function: Callable[[int], T], count: int, n_jobs: int) -> list[T]:
    workers = min(_validate_n_jobs(n_jobs), count)
    if workers == 1:
        return [function(index) for index in range(count)]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(function, range(count)))


def prepare_temporal_hardening_evidence(
    nhis: pd.DataFrame,
    working_days: pd.DataFrame,
    bundle_config: Mapping[str, Any],
    *,
    years: Sequence[int] = OBSERVED_YEARS,
    expected_policy_sigungu: int = 11,
) -> TemporalHardeningEvidence:
    """Build auditable service and bundle panels from normalized NHIS counts.

    ``working_day_adjusted`` reproduces the frozen Stage 2 construction: each
    sigungu-service-year working-day rate is divided by its own 12-month mean
    before explicit bundle weighting.  The raw and alternative normalizations
    are retained only for sensitivity diagnostics.
    """

    required = {
        "year",
        "month",
        "policy_sigungu_name",
        "service_id",
        "persons",
        "visits",
    }
    _require_columns(nhis, required, "Normalized NHIS panel")
    _require_columns(
        working_days,
        {"year", "month", "calendar_days", "working_days"},
        "Working-day calendar",
    )
    expected_policy_sigungu = _validate_positive_integer(
        expected_policy_sigungu, "expected_policy_sigungu"
    )
    observed_years = _strict_two_year_contract(nhis, years)
    calendar_years = _strict_two_year_contract(working_days, years)
    if observed_years != calendar_years:
        raise ValueError("NHIS and working-day calendar year contracts differ")

    work = nhis.copy()
    work["year"] = pd.to_numeric(work["year"], errors="raise").astype(int)
    work["month"] = pd.to_numeric(work["month"], errors="raise").astype(int)
    if not work["month"].between(1, 12).all():
        raise ValueError("NHIS months must be in 1..12")
    for column in ("persons", "visits"):
        work[column] = pd.to_numeric(work[column], errors="raise").astype(float)
        if not np.isfinite(work[column]).all() or work[column].lt(0).any():
            raise ValueError(f"NHIS {column} must be finite and non-negative")
    keys = ["policy_sigungu_name", "service_id", "year", "month"]
    if work.duplicated(keys).any():
        raise ValueError("Normalized NHIS panel has duplicate sigungu-service-year-month keys")
    sigungu = sorted(work["policy_sigungu_name"].astype(str).unique())
    if len(sigungu) != expected_policy_sigungu:
        raise ValueError(
            f"NHIS contains {len(sigungu)} policy sigungu, expected {expected_policy_sigungu}"
        )

    bundle_ids, included_services, weights = _bundle_weights(bundle_config)
    missing_services = sorted(set(included_services) - set(work["service_id"].astype(str)))
    if missing_services:
        raise KeyError(f"NHIS lacks configured bundle services: {missing_services}")
    observed_services = sorted(work["service_id"].astype(str).unique())
    panel_counts = work.groupby(keys[:-1], sort=False)["month"].nunique()
    if not panel_counts.eq(12).all():
        raise ValueError("NHIS has incomplete 12-month sigungu-service-year panels")

    calendar = working_days[["year", "month", "calendar_days", "working_days"]].copy()
    if calendar.duplicated(["year", "month"]).any() or len(calendar) != 24:
        raise ValueError("Working-day calendar must contain 24 unique year-month rows")
    for column in ("calendar_days", "working_days"):
        calendar[column] = pd.to_numeric(calendar[column], errors="raise").astype(float)
        if calendar[column].le(0).any():
            raise ValueError(f"Working-day calendar {column} must be positive")
    work = work.merge(calendar, on=["year", "month"], validate="many_to_one")
    if len(work) != expected_policy_sigungu * len(observed_services) * 2 * 12:
        raise ValueError("NHIS service panel has incomplete rectangular coverage")

    exposure = np.where(
        work["service_id"].eq("emergency_medicine"),
        work["calendar_days"],
        work["working_days"],
    )
    work["working_day_rate"] = work["visits"].to_numpy(float) / exposure
    group_keys = ["policy_sigungu_name", "service_id", "year"]
    raw_mean = work.groupby(group_keys, sort=False)["visits"].transform("mean")
    rate_mean = work.groupby(group_keys, sort=False)["working_day_rate"].transform("mean")
    rate_sum = work.groupby(group_keys, sort=False)["working_day_rate"].transform("sum")
    if raw_mean.le(0).any() or rate_mean.le(0).any() or rate_sum.le(0).any():
        raise ValueError("At least one NHIS annual service profile has no observed utilisation")
    work["raw_monthly_utilization"] = work["visits"] / raw_mean
    work[PRIMARY_VALUE_COLUMN] = work["working_day_rate"] / rate_mean
    work["year_normalized_working_day"] = 12.0 * work["working_day_rate"] / rate_sum
    rate_std = work.groupby(group_keys, sort=False)["working_day_rate"].transform(
        lambda values: float(values.std(ddof=0))
    )
    work["within_year_z_working_day"] = np.where(
        rate_std.gt(1e-15), (work["working_day_rate"] - rate_mean) / rate_std, 0.0
    )
    work["within_year_percentile_working_day"] = work.groupby(
        group_keys, sort=False
    )["working_day_rate"].rank(method="average", pct=True)
    work["annual_share_working_day"] = work["working_day_rate"] / rate_sum
    if not np.allclose(
        work[PRIMARY_VALUE_COLUMN], work["year_normalized_working_day"], atol=1e-12
    ):
        raise RuntimeError("Algebraic year-normalization audit failed")

    bundle_name = {
        str(bundle["bundle_id"]): str(bundle.get("bundle_name_ko", bundle["bundle_id"]))
        for bundle in bundle_config["bundles"]
    }
    bundle_work = work.loc[work["service_id"].astype(str).isin(included_services)].copy()
    indexed = bundle_work.set_index(
        ["policy_sigungu_name", "year", "month", "service_id"]
    ).sort_index()
    bundle_rows: list[dict[str, Any]] = []
    for policy_sigungu_name in sigungu:
        for year in observed_years:
            for month in range(1, 13):
                slice_frame = indexed.loc[(policy_sigungu_name, year, month)]
                slice_frame = slice_frame.reindex(included_services)
                if slice_frame[list(SENSITIVITY_VALUE_COLUMNS)].isna().any().any():
                    raise ValueError("Bundle source slice is incomplete after service reindex")
                matrix = slice_frame[list(SENSITIVITY_VALUE_COLUMNS)].to_numpy(float)
                for bundle_index, bundle_id in enumerate(bundle_ids):
                    row: dict[str, Any] = {
                        "policy_sigungu_name": policy_sigungu_name,
                        "year": year,
                        "month": month,
                        "bundle_id": bundle_id,
                        "bundle_name_ko": bundle_name[bundle_id],
                    }
                    for column_index, column in enumerate(SENSITIVITY_VALUE_COLUMNS):
                        row[column] = float(weights[bundle_index] @ matrix[:, column_index])
                    bundle_rows.append(row)
    bundle_month = pd.DataFrame(bundle_rows).sort_values(
        ["policy_sigungu_name", "bundle_id", "year", "month"], kind="mergesort"
    ).reset_index(drop=True)
    bundle_keys = ["policy_sigungu_name", "bundle_id", "year", "month"]
    expected_bundle_rows = expected_policy_sigungu * len(bundle_ids) * 2 * 12
    if len(bundle_month) != expected_bundle_rows or bundle_month.duplicated(bundle_keys).any():
        raise RuntimeError("Sigungu-bundle evidence is not a complete unique rectangle")
    numeric = bundle_month[list(SENSITIVITY_VALUE_COLUMNS)].to_numpy(float)
    if not np.isfinite(numeric).all():
        raise RuntimeError("Sigungu-bundle evidence contains non-finite values")

    service_month = work.sort_values(keys, kind="mergesort").reset_index(drop=True)
    audit = {
        "observed_years": list(observed_years),
        "policy_sigungu_count": len(sigungu),
        "observed_service_count": len(observed_services),
        "included_service_count": len(included_services),
        "bundle_count": len(bundle_ids),
        "service_month_rows": int(len(service_month)),
        "bundle_month_rows": int(len(bundle_month)),
        "temporal_independent_unit": TEMPORAL_INDEPENDENT_UNIT,
        "temporal_independent_n": int(len(sigungu) * len(bundle_ids)),
        "expected_production_independent_n": 55,
        "admin_dong_delivery_rows_are_independent_samples": False,
        "two_year_leakage_guard": "strict_distinct_fit_and_holdout_years",
        "primary_value_column": PRIMARY_VALUE_COLUMN,
        "interpretation": "observed_insured_utilization_coarse_scheduling_prior_not_demand",
    }
    return TemporalHardeningEvidence(service_month, bundle_month, audit)


def _season_scores(
    frame: pd.DataFrame,
    *,
    value_column: str = PRIMARY_VALUE_COLUMN,
    definition: Mapping[str, Sequence[int]] = PRIMARY_SEASON_DEFINITION,
    group_columns: Sequence[str] = ("policy_sigungu_name", "bundle_id", "year"),
) -> pd.DataFrame:
    _require_columns(
        frame,
        {*group_columns, "month", value_column},
        "Bundle-month evidence",
    )
    month_to_season = {
        int(month): season for season, months in definition.items() for month in months
    }
    work = frame.copy()
    work["season"] = work["month"].map(month_to_season)
    if work["season"].isna().any():
        raise ValueError("Season definition did not map every observed month")
    output = (
        work.groupby([*group_columns, "season"], as_index=False, sort=True)[value_column]
        .mean()
        .rename(columns={value_column: "season_score"})
    )
    expected = work[list(group_columns)].drop_duplicates().shape[0] * 4
    if len(output) != expected:
        raise ValueError("Season score coverage is incomplete")
    output["season_order"] = output["season"].map(
        {season: index for index, season in enumerate(SEASON_ORDER)}
    )
    return output.sort_values([*group_columns, "season_order"], kind="mergesort").reset_index(
        drop=True
    )


def strict_cross_year_transfer(
    bundle_month: pd.DataFrame,
    *,
    years: Sequence[int] = OBSERVED_YEARS,
    value_column: str = PRIMARY_VALUE_COLUMN,
) -> CrossYearTransferResult:
    """Run 2022->2023 and 2023->2022 without pooled fitting leakage."""

    left_year, right_year = _strict_two_year_contract(bundle_month, years)
    seasonal = _season_scores(bundle_month, value_column=value_column)
    sigungu = sorted(bundle_month["policy_sigungu_name"].astype(str).unique())
    bundles = sorted(bundle_month["bundle_id"].astype(str).unique())
    expected_independent_n = len(sigungu) * len(bundles)
    indexed = seasonal.set_index(["policy_sigungu_name", "bundle_id", "year", "season"])[
        "season_score"
    ]
    rows: list[dict[str, Any]] = []
    for train_year, holdout_year in ((left_year, right_year), (right_year, left_year)):
        direction = f"{train_year}_to_{holdout_year}"
        for policy_sigungu_name in sigungu:
            for bundle_id in bundles:
                train = np.array(
                    [indexed.loc[(policy_sigungu_name, bundle_id, train_year, season)] for season in SEASON_ORDER],
                    dtype=float,
                )
                holdout = np.array(
                    [indexed.loc[(policy_sigungu_name, bundle_id, holdout_year, season)] for season in SEASON_ORDER],
                    dtype=float,
                )
                train_profile, holdout_profile = _profile(train), _profile(holdout)
                rho, tau = _safe_correlations(train, holdout)
                selected_index = SEASON_ORDER.index(train_profile["primary"])
                fallback_index = SEASON_ORDER.index(train_profile["fallback"])
                regret = float(holdout.max() - holdout[selected_index])
                denominator = holdout_profile["range"] if holdout_profile["range"] > 1e-15 else 1.0
                overlap = len(train_profile["top2"] & holdout_profile["top2"])
                rows.append(
                    {
                        "direction": direction,
                        "train_year": train_year,
                        "holdout_year": holdout_year,
                        "policy_sigungu_name": policy_sigungu_name,
                        "bundle_id": bundle_id,
                        "train_primary": train_profile["primary"],
                        "train_fallback": train_profile["fallback"],
                        "holdout_primary": holdout_profile["primary"],
                        "holdout_fallback": holdout_profile["fallback"],
                        "top1_agreement": int(train_profile["primary"] == holdout_profile["primary"]),
                        "train_primary_in_holdout_top2": int(
                            train_profile["primary"] in holdout_profile["top2"]
                        ),
                        "pair_top2_coverage": overlap / 2.0,
                        "pair_covers_holdout_top2": int(overlap == 2),
                        "season_rank_spearman": rho,
                        "season_rank_kendall": tau,
                        "heldout_score_of_train_primary": float(holdout[selected_index]),
                        "heldout_best_score": float(holdout.max()),
                        "heldout_regret": regret,
                        "heldout_regret_normalized": regret / denominator,
                        "train_top1_margin": train_profile["top1_margin"],
                        "train_top2_margin": train_profile["top2_margin"],
                        "train_top1_margin_normalized": train_profile["top1_margin_normalized"],
                        "train_top2_margin_normalized": train_profile["top2_margin_normalized"],
                        "holdout_top1_margin": holdout_profile["top1_margin"],
                        "holdout_top2_margin": holdout_profile["top2_margin"],
                        "margin_sign_consistent": int(
                            holdout[selected_index] >= holdout[fallback_index]
                        ),
                        "temporal_independent_unit": TEMPORAL_INDEPENDENT_UNIT,
                        "temporal_independent_n": expected_independent_n,
                        "fit_year_record_count": int(
                            bundle_month["year"].eq(train_year).sum()
                        ),
                        "holdout_year_record_count": int(
                            bundle_month["year"].eq(holdout_year).sum()
                        ),
                        "fit_holdout_overlap_row_count": 0,
                        "leakage_guard_pass": True,
                    }
                )
    detail = pd.DataFrame(rows).sort_values(
        ["direction", "bundle_id", "policy_sigungu_name"], kind="mergesort"
    ).reset_index(drop=True)
    if (
        len(detail) != expected_independent_n * 2
        or detail.groupby("direction").size().ne(expected_independent_n).any()
        or detail["fit_holdout_overlap_row_count"].ne(0).any()
    ):
        raise RuntimeError("Strict cross-year transfer coverage/leakage contract failed")
    summary = (
        detail.groupby(["direction", "train_year", "holdout_year", "bundle_id"], as_index=False)
        .agg(
            top1_agreement=("top1_agreement", "mean"),
            train_primary_in_holdout_top2=("train_primary_in_holdout_top2", "mean"),
            pair_top2_coverage=("pair_top2_coverage", "mean"),
            pair_covers_holdout_top2=("pair_covers_holdout_top2", "mean"),
            season_rank_spearman=("season_rank_spearman", "median"),
            season_rank_kendall=("season_rank_kendall", "median"),
            heldout_regret=("heldout_regret", "mean"),
            heldout_regret_normalized=("heldout_regret_normalized", "mean"),
            train_top1_margin=("train_top1_margin", "median"),
            train_top2_margin=("train_top2_margin", "median"),
            margin_sign_consistency=("margin_sign_consistent", "mean"),
            temporal_independent_n=("policy_sigungu_name", "nunique"),
        )
        .sort_values(["bundle_id", "direction"], kind="mergesort")
        .reset_index(drop=True)
    )
    summary["temporal_independent_n_total"] = expected_independent_n
    summary["admin_dong_broadcast_n_used"] = 0
    return CrossYearTransferResult(detail, summary)


def build_cross_year_consensus(
    bundle_month: pd.DataFrame,
    *,
    years: Sequence[int] = OBSERVED_YEARS,
    value_column: str = PRIMARY_VALUE_COLUMN,
) -> pd.DataFrame:
    """Create transparent year, pooled, mean/median/Borda/worst ranks."""

    left_year, right_year = _strict_two_year_contract(bundle_month, years)
    provincial = (
        bundle_month.groupby(["bundle_id", "year", "month"], as_index=False)[value_column]
        .mean()
    )
    seasonal = _season_scores(
        provincial,
        value_column=value_column,
        group_columns=("bundle_id", "year"),
    )
    indexed = seasonal.set_index(["bundle_id", "year", "season"])["season_score"]
    rows: list[dict[str, Any]] = []
    for bundle_id in sorted(provincial["bundle_id"].astype(str).unique()):
        scores = {
            year: np.array(
                [indexed.loc[(bundle_id, year, season)] for season in SEASON_ORDER], dtype=float
            )
            for year in (left_year, right_year)
        }
        year_ranks = {year: _rank_vector(values) for year, values in scores.items()}
        pooled = np.mean(np.stack([scores[left_year], scores[right_year]]), axis=0)
        pooled_ranks = _rank_vector(pooled)
        mean_ranks = np.mean(np.stack(list(year_ranks.values())), axis=0)
        median_ranks = np.median(np.stack(list(year_ranks.values())), axis=0)
        worst_ranks = np.max(np.stack(list(year_ranks.values())), axis=0)
        borda_points = np.sum(5 - np.stack(list(year_ranks.values())), axis=0)

        def ordinal(values: np.ndarray, *, descending: bool = False) -> np.ndarray:
            primary = -values if descending else values
            order = np.lexsort((np.arange(4), primary))
            output = np.empty(4, dtype=int)
            output[order] = np.arange(1, 5)
            return output

        mean_method_rank = ordinal(mean_ranks)
        median_method_rank = ordinal(median_ranks)
        borda_method_rank = ordinal(borda_points, descending=True)
        worst_method_rank = ordinal(worst_ranks)
        consensus_order = np.lexsort(
            (np.arange(4), pooled_ranks, worst_ranks, mean_ranks, median_ranks)
        )
        consensus_rank = np.empty(4, dtype=int)
        consensus_rank[consensus_order] = np.arange(1, 5)
        for season_index, season in enumerate(SEASON_ORDER):
            rows.append(
                {
                    "bundle_id": bundle_id,
                    "season": season,
                    f"year{left_year}_score": float(scores[left_year][season_index]),
                    f"year{right_year}_score": float(scores[right_year][season_index]),
                    "pooled_score": float(pooled[season_index]),
                    f"year{left_year}_rank": int(year_ranks[left_year][season_index]),
                    f"year{right_year}_rank": int(year_ranks[right_year][season_index]),
                    "pooled_rank": int(pooled_ranks[season_index]),
                    "mean_rank_value": float(mean_ranks[season_index]),
                    "median_rank_value": float(median_ranks[season_index]),
                    "borda_points": float(borda_points[season_index]),
                    "worst_year_rank_value": int(worst_ranks[season_index]),
                    "mean_rank": int(mean_method_rank[season_index]),
                    "median_rank": int(median_method_rank[season_index]),
                    "borda_rank": int(borda_method_rank[season_index]),
                    "worst_year_rank": int(worst_method_rank[season_index]),
                    "cross_year_consensus_rank": int(consensus_rank[season_index]),
                    "consensus_method": "lexicographic_median_mean_worst_pooled",
                }
            )
    output = pd.DataFrame(rows).sort_values(
        ["bundle_id", "cross_year_consensus_rank"], kind="mergesort"
    ).reset_index(drop=True)
    if output.groupby("bundle_id").size().ne(4).any():
        raise RuntimeError("Cross-year consensus does not contain four seasons per bundle")
    return output


def _provincial_profile(
    frame: pd.DataFrame,
    *,
    value_column: str = PRIMARY_VALUE_COLUMN,
    definition: Mapping[str, Sequence[int]] = PRIMARY_SEASON_DEFINITION,
    years_separate: bool = False,
) -> pd.DataFrame:
    group_columns = ["bundle_id", "year", "month"] if years_separate else ["bundle_id", "month"]
    provincial = frame.groupby(group_columns, as_index=False)[value_column].mean()
    season_groups = ("bundle_id", "year") if years_separate else ("bundle_id",)
    return _season_scores(
        provincial,
        value_column=value_column,
        definition=definition,
        group_columns=season_groups,
    )


def leave_one_month_out(
    bundle_month: pd.DataFrame,
    *,
    years: Sequence[int] = OBSERVED_YEARS,
    value_column: str = PRIMARY_VALUE_COLUMN,
) -> DiagnosticResult:
    """Run exactly 12 month removals in each of the two observed years."""

    observed_years = _strict_two_year_contract(bundle_month, years)
    full = _provincial_profile(bundle_month, value_column=value_column, years_separate=True)
    full_indexed = full.set_index(["bundle_id", "year", "season"])["season_score"]
    bundles = sorted(bundle_month["bundle_id"].astype(str).unique())
    rows: list[dict[str, Any]] = []
    for year in observed_years:
        year_frame = bundle_month.loc[bundle_month["year"].eq(year)].copy()
        for removed_month in range(1, 13):
            removal_id = f"{year}_remove_{removed_month:02d}"
            variant_frame = year_frame.loc[year_frame["month"].ne(removed_month)].copy()
            variant = _provincial_profile(
                variant_frame,
                value_column=value_column,
                years_separate=True,
            )
            variant_indexed = variant.set_index(["bundle_id", "year", "season"])[
                "season_score"
            ]
            for bundle_id in bundles:
                full_values = np.array(
                    [full_indexed.loc[(bundle_id, year, season)] for season in SEASON_ORDER],
                    dtype=float,
                )
                variant_values = np.array(
                    [variant_indexed.loc[(bundle_id, year, season)] for season in SEASON_ORDER],
                    dtype=float,
                )
                reference, changed = _profile(full_values), _profile(variant_values)
                rho, tau = _safe_correlations(full_values, variant_values)
                selected_index = SEASON_ORDER.index(changed["primary"])
                regret = float(full_values.max() - full_values[selected_index])
                denominator = reference["range"] if reference["range"] > 1e-15 else 1.0
                overlap = len(reference["top2"] & changed["top2"])
                rows.append(
                    {
                        "removal_id": removal_id,
                        "year": year,
                        "removed_month": removed_month,
                        "bundle_id": bundle_id,
                        "full_primary": reference["primary"],
                        "full_fallback": reference["fallback"],
                        "variant_primary": changed["primary"],
                        "variant_fallback": changed["fallback"],
                        "primary_retained": int(reference["primary"] == changed["primary"]),
                        "top2_retention": overlap / 2.0,
                        "top2_pair_retained": int(overlap == 2),
                        "rank_spearman": rho,
                        "rank_kendall": tau,
                        "full_margin": reference["top1_margin"],
                        "variant_margin": changed["top1_margin"],
                        "margin_change": float(changed["top1_margin"] - reference["top1_margin"]),
                        "absolute_margin_change": abs(
                            float(changed["top1_margin"] - reference["top1_margin"])
                        ),
                        "maximum_regret": regret,
                        "maximum_regret_normalized": regret / denominator,
                        "removed_month_was_in_full_primary_season": int(
                            removed_month in PRIMARY_SEASON_DEFINITION[reference["primary"]]
                        ),
                    }
                )
    detail = pd.DataFrame(rows).sort_values(
        ["year", "removed_month", "bundle_id"], kind="mergesort"
    ).reset_index(drop=True)
    scenario_count = detail[["year", "removed_month"]].drop_duplicates().shape[0]
    per_year = detail[["year", "removed_month"]].drop_duplicates().groupby("year").size()
    if scenario_count != 24 or per_year.ne(12).any() or len(detail) != 24 * len(bundles):
        raise RuntimeError("LOMO must contain exactly 12 removal scenarios per observed year")
    summary = (
        detail.groupby("bundle_id", as_index=False)
        .agg(
            lomo_primary_retention=("primary_retained", "mean"),
            lomo_top2_retention=("top2_retention", "mean"),
            lomo_top2_pair_retention=("top2_pair_retained", "mean"),
            lomo_rank_spearman=("rank_spearman", "median"),
            lomo_max_absolute_margin_change=("absolute_margin_change", "max"),
            lomo_maximum_regret=("maximum_regret", "max"),
            lomo_maximum_regret_normalized=("maximum_regret_normalized", "max"),
            lomo_diagnostic_rows=("removal_id", "size"),
            lomo_unique_removals=("removal_id", "nunique"),
        )
        .sort_values("bundle_id")
        .reset_index(drop=True)
    )
    return DiagnosticResult(detail, summary)


def working_day_normalization_sensitivity(
    bundle_month: pd.DataFrame,
    *,
    years: Sequence[int] = OBSERVED_YEARS,
    baseline_column: str = PRIMARY_VALUE_COLUMN,
    value_columns: Sequence[str] = SENSITIVITY_VALUE_COLUMNS,
) -> pd.DataFrame:
    """Compare raw, working-day and within-year normalization choices."""

    _strict_two_year_contract(bundle_month, years)
    columns = tuple(dict.fromkeys(str(column) for column in value_columns))
    _require_columns(bundle_month, {baseline_column, *columns}, "Sensitivity bundle panel")
    baseline_seasonal = _provincial_profile(bundle_month, value_column=baseline_column)
    baseline_indexed = baseline_seasonal.set_index(["bundle_id", "season"])["season_score"]
    bundles = sorted(bundle_month["bundle_id"].astype(str).unique())
    rows: list[dict[str, Any]] = []
    for variant_column in columns:
        seasonal = _provincial_profile(bundle_month, value_column=variant_column)
        indexed = seasonal.set_index(["bundle_id", "season"])["season_score"]
        for bundle_id in bundles:
            baseline = np.array(
                [baseline_indexed.loc[(bundle_id, season)] for season in SEASON_ORDER], dtype=float
            )
            variant = np.array(
                [indexed.loc[(bundle_id, season)] for season in SEASON_ORDER], dtype=float
            )
            reference, changed = _profile(baseline), _profile(variant)
            rho, tau = _safe_correlations(baseline, variant)
            overlap = len(reference["top2"] & changed["top2"])
            rows.append(
                {
                    "bundle_id": bundle_id,
                    "baseline_variant": baseline_column,
                    "variant": variant_column,
                    "season_rank_spearman": rho,
                    "season_rank_kendall": tau,
                    "top1_agreement": int(reference["primary"] == changed["primary"]),
                    "top2_agreement": overlap / 2.0,
                    "top2_pair_agreement": int(overlap == 2),
                    "baseline_primary": reference["primary"],
                    "variant_primary": changed["primary"],
                    "baseline_fallback": reference["fallback"],
                    "variant_fallback": changed["fallback"],
                    "baseline_margin": reference["top1_margin"],
                    "variant_margin": changed["top1_margin"],
                    "margin_change": float(changed["top1_margin"] - reference["top1_margin"]),
                    "absolute_margin_change": abs(
                        float(changed["top1_margin"] - reference["top1_margin"])
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values(["bundle_id", "variant"], kind="mergesort").reset_index(
        drop=True
    )


def leave_one_sigungu_out(
    bundle_month: pd.DataFrame,
    *,
    years: Sequence[int] = OBSERVED_YEARS,
    value_column: str = PRIMARY_VALUE_COLUMN,
) -> DiagnosticResult:
    """Remove one entire sigungu block, retaining all years/months/bundles."""

    _strict_two_year_contract(bundle_month, years)
    sigungu = sorted(bundle_month["policy_sigungu_name"].astype(str).unique())
    if len(sigungu) < 3:
        raise ValueError("LOSO requires at least three policy sigungu")
    bundles = sorted(bundle_month["bundle_id"].astype(str).unique())
    full = _provincial_profile(bundle_month, value_column=value_column)
    full_indexed = full.set_index(["bundle_id", "season"])["season_score"]
    rows: list[dict[str, Any]] = []
    for removed_sigungu in sigungu:
        training = bundle_month.loc[
            bundle_month["policy_sigungu_name"].astype(str).ne(removed_sigungu)
        ].copy()
        heldout = bundle_month.loc[
            bundle_month["policy_sigungu_name"].astype(str).eq(removed_sigungu)
        ].copy()
        variant = _provincial_profile(training, value_column=value_column)
        heldout_seasonal = _provincial_profile(heldout, value_column=value_column)
        variant_indexed = variant.set_index(["bundle_id", "season"])["season_score"]
        heldout_indexed = heldout_seasonal.set_index(["bundle_id", "season"])["season_score"]
        for bundle_id in bundles:
            reference_values = np.array(
                [full_indexed.loc[(bundle_id, season)] for season in SEASON_ORDER], dtype=float
            )
            variant_values = np.array(
                [variant_indexed.loc[(bundle_id, season)] for season in SEASON_ORDER], dtype=float
            )
            heldout_values = np.array(
                [heldout_indexed.loc[(bundle_id, season)] for season in SEASON_ORDER], dtype=float
            )
            reference, changed, evaluation = (
                _profile(reference_values),
                _profile(variant_values),
                _profile(heldout_values),
            )
            rho, tau = _safe_correlations(reference_values, variant_values)
            selected_index = SEASON_ORDER.index(changed["primary"])
            regret = float(heldout_values.max() - heldout_values[selected_index])
            denominator = evaluation["range"] if evaluation["range"] > 1e-15 else 1.0
            overlap = len(reference["top2"] & changed["top2"])
            rows.append(
                {
                    "removed_sigungu": removed_sigungu,
                    "bundle_id": bundle_id,
                    "full_primary": reference["primary"],
                    "full_fallback": reference["fallback"],
                    "variant_primary": changed["primary"],
                    "variant_fallback": changed["fallback"],
                    "heldout_primary": evaluation["primary"],
                    "primary_retained": int(reference["primary"] == changed["primary"]),
                    "top2_retention": overlap / 2.0,
                    "top2_pair_retained": int(overlap == 2),
                    "rank_spearman": rho,
                    "rank_kendall": tau,
                    "full_margin": reference["top1_margin"],
                    "variant_margin": changed["top1_margin"],
                    "margin_change": float(changed["top1_margin"] - reference["top1_margin"]),
                    "heldout_regret": regret,
                    "heldout_regret_normalized": regret / denominator,
                    "training_sigungu_n": len(sigungu) - 1,
                    "heldout_sigungu_n": 1,
                    "excluded_block_includes_all_months_years_services": True,
                }
            )
    detail = pd.DataFrame(rows).sort_values(
        ["removed_sigungu", "bundle_id"], kind="mergesort"
    ).reset_index(drop=True)
    if detail["removed_sigungu"].nunique() != len(sigungu) or len(detail) != len(sigungu) * len(bundles):
        raise RuntimeError("LOSO scenario coverage is incomplete")
    summary = (
        detail.groupby("bundle_id", as_index=False)
        .agg(
            loso_primary_retention=("primary_retained", "mean"),
            loso_top2_retention=("top2_retention", "mean"),
            loso_top2_pair_retention=("top2_pair_retained", "mean"),
            loso_rank_spearman=("rank_spearman", "median"),
            loso_max_absolute_margin_change=("margin_change", lambda values: float(np.abs(values).max())),
            loso_max_heldout_regret=("heldout_regret", "max"),
            loso_max_heldout_regret_normalized=("heldout_regret_normalized", "max"),
            loso_unique_removals=("removed_sigungu", "nunique"),
        )
        .sort_values("bundle_id")
        .reset_index(drop=True)
    )
    return DiagnosticResult(detail, summary)


def season_boundary_sensitivity(
    bundle_month: pd.DataFrame,
    *,
    years: Sequence[int] = OBSERVED_YEARS,
    value_column: str = PRIMARY_VALUE_COLUMN,
    definitions: Mapping[str, Mapping[str, Sequence[int]]] = SEASON_BOUNDARY_DEFINITIONS,
) -> DiagnosticResult:
    """Diagnose reasonable boundary alternatives without changing the primary."""

    _strict_two_year_contract(bundle_month, years)
    if "primary_meteorological" not in definitions:
        raise ValueError("Season-boundary diagnostics must retain the frozen primary definition")
    profiles: dict[str, pd.DataFrame] = {}
    for name, definition in definitions.items():
        _season_matrix(definition)
        profiles[str(name)] = _provincial_profile(
            bundle_month,
            value_column=value_column,
            definition=definition,
        )
    baseline_indexed = profiles["primary_meteorological"].set_index(["bundle_id", "season"])[
        "season_score"
    ]
    bundles = sorted(bundle_month["bundle_id"].astype(str).unique())
    rows: list[dict[str, Any]] = []
    for definition_name, seasonal in profiles.items():
        indexed = seasonal.set_index(["bundle_id", "season"])["season_score"]
        for bundle_id in bundles:
            baseline = np.array(
                [baseline_indexed.loc[(bundle_id, season)] for season in SEASON_ORDER], dtype=float
            )
            variant = np.array(
                [indexed.loc[(bundle_id, season)] for season in SEASON_ORDER], dtype=float
            )
            reference, changed = _profile(baseline), _profile(variant)
            rho, tau = _safe_correlations(baseline, variant)
            overlap = len(reference["top2"] & changed["top2"])
            rows.append(
                {
                    "definition": definition_name,
                    "bundle_id": bundle_id,
                    "primary_season": changed["primary"],
                    "fallback_season": changed["fallback"],
                    "baseline_primary": reference["primary"],
                    "baseline_fallback": reference["fallback"],
                    "baseline_primary_retained": int(reference["primary"] == changed["primary"]),
                    "top2_retention": overlap / 2.0,
                    "top2_pair_retained": int(overlap == 2),
                    "rank_spearman": rho,
                    "rank_kendall": tau,
                    "margin": changed["top1_margin"],
                    "margin_normalized": changed["top1_margin_normalized"],
                    "diagnostic_only_primary_definition_unchanged": True,
                }
            )
    detail = pd.DataFrame(rows).sort_values(["bundle_id", "definition"]).reset_index(drop=True)
    if len(detail) != len(bundles) * len(definitions):
        raise RuntimeError("Season-boundary diagnostic coverage is incomplete")
    alternatives = detail.loc[detail["definition"].ne("primary_meteorological")]
    summary = (
        alternatives.groupby("bundle_id", as_index=False)
        .agg(
            boundary_primary_retention=("baseline_primary_retained", "mean"),
            boundary_top2_retention=("top2_retention", "mean"),
            boundary_top2_pair_retention=("top2_pair_retained", "mean"),
            boundary_rank_spearman=("rank_spearman", "median"),
            boundary_definition_count=("definition", "nunique"),
        )
        .sort_values("bundle_id")
        .reset_index(drop=True)
    )
    return DiagnosticResult(detail, summary)


def _bundle_array(
    bundle_month: pd.DataFrame,
    value_column: str,
) -> tuple[list[str], list[str], list[int], np.ndarray]:
    sigungu = sorted(bundle_month["policy_sigungu_name"].astype(str).unique())
    bundles = sorted(bundle_month["bundle_id"].astype(str).unique())
    years = sorted(pd.to_numeric(bundle_month["year"], errors="raise").astype(int).unique())
    indexed = bundle_month.set_index(
        ["policy_sigungu_name", "bundle_id", "year", "month"]
    )[value_column]
    expected = len(sigungu) * len(bundles) * len(years) * 12
    if len(indexed) != expected or not indexed.index.is_unique:
        raise ValueError("Bundle panel is not rectangular for array construction")
    values = np.empty((len(sigungu), len(bundles), len(years), 12), dtype=float)
    for sigungu_index, policy_sigungu_name in enumerate(sigungu):
        for bundle_index, bundle_id in enumerate(bundles):
            for year_index, year in enumerate(years):
                values[sigungu_index, bundle_index, year_index] = [
                    float(indexed.loc[(policy_sigungu_name, bundle_id, year, month)])
                    for month in range(1, 13)
                ]
    if not np.isfinite(values).all():
        raise ValueError("Bundle array contains non-finite values")
    return sigungu, bundles, years, values


def sigungu_block_bootstrap(
    bundle_month: pd.DataFrame,
    *,
    n_boot: int,
    years: Sequence[int] = OBSERVED_YEARS,
    value_column: str = PRIMARY_VALUE_COLUMN,
    random_state: int = 42,
    n_jobs: int = 1,
    expected_n_boot: int | None = None,
) -> DiagnosticResult:
    """Resample whole sigungu blocks, preserving all month/year sequences."""

    n_boot = _validate_positive_integer(n_boot, "n_boot")
    if expected_n_boot is not None and n_boot != _validate_positive_integer(
        expected_n_boot, "expected_n_boot"
    ):
        raise ValueError(f"n_boot={n_boot} does not match expected_n_boot={expected_n_boot}")
    _strict_two_year_contract(bundle_month, years)
    sigungu, bundles, observed_years, values = _bundle_array(bundle_month, value_column)
    if tuple(observed_years) != tuple(sorted(int(year) for year in years)):
        raise RuntimeError("Bootstrap year order does not match strict contract")
    season_matrix = _season_matrix(PRIMARY_SEASON_DEFINITION)
    baseline_month = values.mean(axis=(0, 2))
    baseline_season = np.einsum("bm,sm->bs", baseline_month, season_matrix)
    baseline_profiles = [_profile(baseline_season[index]) for index in range(len(bundles))]
    seed_sequence = np.random.SeedSequence(int(random_state))
    seeds = [int(seed.generate_state(1, dtype=np.uint64)[0]) for seed in seed_sequence.spawn(n_boot)]

    def one_bootstrap(replicate: int) -> list[dict[str, Any]]:
        rng = np.random.default_rng(seeds[replicate])
        sampled = rng.integers(0, len(sigungu), size=len(sigungu))
        # One index selects the complete 2-year x 12-month x all-bundle block.
        month_profile = values[sampled].mean(axis=(0, 2))
        season_scores = np.einsum("bm,sm->bs", month_profile, season_matrix)
        output: list[dict[str, Any]] = []
        for bundle_index, bundle_id in enumerate(bundles):
            profile = _profile(season_scores[bundle_index])
            row: dict[str, Any] = {
                "bootstrap_replicate": replicate + 1,
                "bundle_id": bundle_id,
                "primary_season": profile["primary"],
                "fallback_season": profile["fallback"],
                "top1_margin": profile["top1_margin"],
                "top1_margin_normalized": profile["top1_margin_normalized"],
                "baseline_primary_retained": int(
                    profile["primary"] == baseline_profiles[bundle_index]["primary"]
                ),
                "baseline_top2_coverage": len(
                    profile["top2"] & baseline_profiles[bundle_index]["top2"]
                )
                / 2.0,
                "sampled_unique_sigungu": int(np.unique(sampled).size),
                "complete_block_resampling": True,
            }
            for season_index, season in enumerate(SEASON_ORDER):
                row[f"score__{season}"] = float(season_scores[bundle_index, season_index])
                row[f"rank__{season}"] = int(profile["ranks"][season_index])
            output.append(row)
        return output

    nested = _parallel_map(one_bootstrap, n_boot, n_jobs)
    detail = pd.DataFrame([row for block in nested for row in block]).sort_values(
        ["bootstrap_replicate", "bundle_id"], kind="mergesort"
    ).reset_index(drop=True)
    replicate_counts = detail.groupby("bundle_id")["bootstrap_replicate"].nunique()
    if (
        detail["bootstrap_replicate"].nunique() != n_boot
        or replicate_counts.ne(n_boot).any()
        or len(detail) != n_boot * len(bundles)
        or not detail["complete_block_resampling"].all()
    ):
        raise RuntimeError("Bootstrap n_valid does not equal requested n_boot")
    summary_rows: list[dict[str, Any]] = []
    for bundle_index, bundle_id in enumerate(bundles):
        group = detail.loc[detail["bundle_id"].eq(bundle_id)]
        baseline = baseline_profiles[bundle_index]
        margin_values = group["top1_margin"].to_numpy(float)
        for season in SEASON_ORDER:
            summary_rows.append(
                {
                    "bundle_id": bundle_id,
                    "season": season,
                    "is_baseline_primary": int(season == baseline["primary"]),
                    "is_baseline_fallback": int(season == baseline["fallback"]),
                    "primary_probability": float(group["primary_season"].eq(season).mean()),
                    "fallback_probability": float(group["fallback_season"].eq(season).mean()),
                    "top2_probability": float(
                        (group["primary_season"].eq(season) | group["fallback_season"].eq(season)).mean()
                    ),
                    "mean_rank": float(group[f"rank__{season}"].mean()),
                    "sd_rank": float(group[f"rank__{season}"].std(ddof=0)),
                    "baseline_primary_probability": float(
                        group["baseline_primary_retained"].mean()
                    ),
                    "baseline_pair_coverage_probability": float(
                        group["baseline_top2_coverage"].mean()
                    ),
                    "margin_q025": float(np.quantile(margin_values, 0.025)),
                    "margin_median": float(np.median(margin_values)),
                    "margin_q975": float(np.quantile(margin_values, 0.975)),
                    "n_boot": n_boot,
                    "n_valid": int(len(group)),
                    "random_state": int(random_state),
                    "temporal_block_n": len(sigungu),
                }
            )
    summary = pd.DataFrame(summary_rows).sort_values(
        ["bundle_id", "season"], kind="mergesort"
    ).reset_index(drop=True)
    if summary["n_valid"].ne(n_boot).any():
        raise RuntimeError("Bootstrap summary contains incomplete replicates")
    return DiagnosticResult(detail, summary)


def _service_array(
    service_month: pd.DataFrame,
    service_ids: Sequence[str],
    value_column: str,
) -> tuple[list[str], list[int], np.ndarray]:
    sigungu = sorted(service_month["policy_sigungu_name"].astype(str).unique())
    years = sorted(pd.to_numeric(service_month["year"], errors="raise").astype(int).unique())
    subset = service_month.loc[service_month["service_id"].astype(str).isin(service_ids)]
    indexed = subset.set_index(["policy_sigungu_name", "year", "service_id", "month"])[
        value_column
    ]
    expected = len(sigungu) * len(years) * len(service_ids) * 12
    if len(indexed) != expected or not indexed.index.is_unique:
        raise ValueError("Service panel is not rectangular for month-label permutation")
    values = np.empty((len(sigungu), len(years), len(service_ids), 12), dtype=float)
    for sigungu_index, policy_sigungu_name in enumerate(sigungu):
        for year_index, year in enumerate(years):
            for service_index, service_id in enumerate(service_ids):
                values[sigungu_index, year_index, service_index] = [
                    float(indexed.loc[(policy_sigungu_name, year, service_id, month)])
                    for month in range(1, 13)
                ]
    if not np.isfinite(values).all():
        raise ValueError("Service permutation array contains non-finite values")
    return sigungu, years, values


def _null_statistics(year_season: np.ndarray) -> dict[str, np.ndarray]:
    """Metrics for an array shaped year x bundle x season."""

    if year_season.ndim != 3 or year_season.shape[0] != 2 or year_season.shape[2] != 4:
        raise ValueError("Null statistic input must be year x bundle x four seasons")
    pooled = year_season.mean(axis=0)
    bundle_count = pooled.shape[0]
    output = {
        "top1_margin_normalized": np.empty(bundle_count, dtype=float),
        "season_concentration": np.empty(bundle_count, dtype=float),
        "cross_year_rank_spearman": np.empty(bundle_count, dtype=float),
        "cross_year_top1_agreement": np.empty(bundle_count, dtype=float),
        "bundle_seasonal_separation": np.empty(bundle_count, dtype=float),
    }
    for bundle_index in range(bundle_count):
        pooled_profile = _profile(pooled[bundle_index])
        year_a = _profile(year_season[0, bundle_index])
        year_b = _profile(year_season[1, bundle_index])
        rho, _ = _safe_correlations(
            year_season[0, bundle_index], year_season[1, bundle_index]
        )
        mean_abs = float(np.mean(np.abs(pooled[bundle_index])))
        denominator = mean_abs if mean_abs > 1e-15 else 1.0
        output["top1_margin_normalized"][bundle_index] = pooled_profile[
            "top1_margin_normalized"
        ]
        output["season_concentration"][bundle_index] = pooled_profile["range"] / denominator
        output["cross_year_rank_spearman"][bundle_index] = rho
        output["cross_year_top1_agreement"][bundle_index] = int(
            year_a["primary"] == year_b["primary"]
        )
        output["bundle_seasonal_separation"][bundle_index] = float(
            np.std(pooled[bundle_index], ddof=0) / denominator
        )
    return output


def month_label_permutation_null(
    service_month: pd.DataFrame,
    bundle_config: Mapping[str, Any],
    *,
    n_permutations: int,
    years: Sequence[int] = OBSERVED_YEARS,
    value_column: str = PRIMARY_VALUE_COLUMN,
    random_state: int = 42,
    n_jobs: int = 1,
    expected_n_permutations: int | None = None,
) -> DiagnosticResult:
    """Shuffle month labels within sigungu x year x specialty blocks.

    The operation preserves every block's annual total and its twelve observed
    values while breaking calendar alignment.  The p-values are advisory and
    are never the sole release criterion.
    """

    n_permutations = _validate_positive_integer(n_permutations, "n_permutations")
    if expected_n_permutations is not None and n_permutations != _validate_positive_integer(
        expected_n_permutations, "expected_n_permutations"
    ):
        raise ValueError(
            f"n_permutations={n_permutations} does not match "
            f"expected_n_permutations={expected_n_permutations}"
        )
    _strict_two_year_contract(service_month, years)
    bundle_ids, service_ids, weights = _bundle_weights(bundle_config)
    sigungu, observed_years, values = _service_array(service_month, service_ids, value_column)
    if tuple(observed_years) != tuple(sorted(int(year) for year in years)):
        raise RuntimeError("Permutation year order does not match strict contract")
    season_matrix = _season_matrix(PRIMARY_SEASON_DEFINITION)

    def to_year_season(service_values: np.ndarray) -> np.ndarray:
        # g x y x service x month, bundle weights b x service
        bundle_values = np.einsum("gysm,bs->gybm", service_values, weights)
        province_year_month = bundle_values.mean(axis=0)
        return np.einsum("ybm,sm->ybs", province_year_month, season_matrix)

    observed = _null_statistics(to_year_season(values))
    original_totals = values.sum(axis=-1)
    seed_sequence = np.random.SeedSequence(int(random_state))
    seeds = [
        int(seed.generate_state(1, dtype=np.uint64)[0])
        for seed in seed_sequence.spawn(n_permutations)
    ]

    def one_permutation(replicate: int) -> list[dict[str, Any]]:
        rng = np.random.default_rng(seeds[replicate])
        order = np.argsort(rng.random(values.shape), axis=-1, kind="stable")
        permuted = np.take_along_axis(values, order, axis=-1)
        totals_preserved = bool(np.allclose(permuted.sum(axis=-1), original_totals, atol=1e-12))
        metrics = _null_statistics(to_year_season(permuted))
        output: list[dict[str, Any]] = []
        for bundle_index, bundle_id in enumerate(bundle_ids):
            row: dict[str, Any] = {
                "permutation_replicate": replicate + 1,
                "bundle_id": bundle_id,
                "annual_totals_preserved": totals_preserved,
                "month_labels_shuffled_within_sigungu_year_specialty": True,
            }
            for metric, metric_values in metrics.items():
                row[metric] = float(metric_values[bundle_index])
            output.append(row)
        return output

    nested = _parallel_map(one_permutation, n_permutations, n_jobs)
    detail = pd.DataFrame([row for block in nested for row in block]).sort_values(
        ["permutation_replicate", "bundle_id"], kind="mergesort"
    ).reset_index(drop=True)
    valid_counts = detail.groupby("bundle_id")["permutation_replicate"].nunique()
    if (
        detail["permutation_replicate"].nunique() != n_permutations
        or valid_counts.ne(n_permutations).any()
        or len(detail) != n_permutations * len(bundle_ids)
        or not detail["annual_totals_preserved"].all()
    ):
        raise RuntimeError("Permutation count or annual-total preservation contract failed")
    summary_rows: list[dict[str, Any]] = []
    for bundle_index, bundle_id in enumerate(bundle_ids):
        group = detail.loc[detail["bundle_id"].eq(bundle_id)]
        for metric, observed_values in observed.items():
            null_values = group[metric].to_numpy(float)
            observed_value = float(observed_values[bundle_index])
            null_mean = float(null_values.mean())
            null_sd = float(null_values.std(ddof=0))
            p_value = float((1 + np.count_nonzero(null_values >= observed_value)) / (n_permutations + 1))
            standardized = (
                float((observed_value - null_mean) / null_sd)
                if null_sd > 1e-15
                else (0.0 if np.isclose(observed_value, null_mean) else np.inf)
            )
            summary_rows.append(
                {
                    "bundle_id": bundle_id,
                    "metric": metric,
                    "observed": observed_value,
                    "null_mean": null_mean,
                    "null_sd": null_sd,
                    "null_q025": float(np.quantile(null_values, 0.025)),
                    "null_q975": float(np.quantile(null_values, 0.975)),
                    "p_value_upper": p_value,
                    "permutation_strength": 1.0 - p_value,
                    "standardized_strength": standardized,
                    "n_permutations": n_permutations,
                    "n_valid": int(len(group)),
                    "random_state": int(random_state),
                    "null_is_advisory_not_sole_release_gate": True,
                    "annual_totals_preserved": True,
                }
            )
    summary = pd.DataFrame(summary_rows).sort_values(
        ["bundle_id", "metric"], kind="mergesort"
    ).reset_index(drop=True)
    if summary["n_valid"].ne(n_permutations).any():
        raise RuntimeError("Permutation summary contains incomplete replicates")
    return DiagnosticResult(detail, summary)


def _release_rule_sections(
    rules: Mapping[str, Any] | None,
) -> tuple[dict[str, float], dict[str, float]]:
    if rules is None:
        defaults = DEFAULT_RELEASE_THRESHOLDS
        strong = {
            "cross_year_top1_agreement_min": 1.0,
            "bidirectional_primary_in_holdout_top2_min": defaults[
                "strong_pair_coverage_min"
            ],
            "bidirectional_selected_pair_holdout_top2_coverage_min": defaults[
                "strong_pair_coverage_min"
            ],
            "heldout_normalized_regret_max": defaults["strong_regret_max"],
            "pooled_normalized_top1_margin_min": defaults[
                "strong_margin_normalized_min"
            ],
            "year_normalized_top1_margin_min": 0.05,
            "bootstrap_primary_probability_min": defaults[
                "strong_bootstrap_primary_min"
            ],
            "lomo_primary_retention_min": defaults["strong_lomo_primary_min"],
            "loso_primary_retention_min": defaults["strong_loso_primary_min"],
            "working_day_primary_retention_min": 1.0,
            "boundary_primary_retention_min": 0.67,
            "permutation_margin_percentile_min": defaults[
                "strong_permutation_strength_min"
            ],
        }
        pair = {
            "bidirectional_primary_in_holdout_top2_min": defaults["pair_coverage_min"],
            "bidirectional_selected_pair_holdout_top2_coverage_min": defaults[
                "pair_coverage_min"
            ],
            "year_top2_jaccard_min": 0.50,
            "heldout_normalized_regret_max": defaults["pair_regret_max"],
            "pooled_normalized_top1_margin_min": 0.02,
            "bootstrap_pair_coverage_probability_min": defaults[
                "pair_bootstrap_top2_min"
            ],
            "lomo_top2_retention_min": defaults["pair_lomo_top2_min"],
            "loso_top2_retention_min": defaults["pair_loso_top2_min"],
            "working_day_top2_retention_min": 0.75,
            "boundary_top2_retention_min": 0.67,
            "permutation_margin_percentile_min": 0.50,
        }
        return strong, pair
    if "strong_single" not in rules or "robust_pair" not in rules:
        raise KeyError("Release rules require strong_single and robust_pair sections")
    strong = {key: float(value) for key, value in dict(rules["strong_single"]).items()}
    pair = {key: float(value) for key, value in dict(rules["robust_pair"]).items()}
    return strong, pair


def build_abstention_evidence_summary(
    cross_year: CrossYearTransferResult,
    consensus: pd.DataFrame,
    lomo: DiagnosticResult,
    working_day_sensitivity: pd.DataFrame,
    loso: DiagnosticResult,
    season_boundary: DiagnosticResult,
    bootstrap: DiagnosticResult,
    permutation: DiagnosticResult,
    *,
    release_thresholds: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Combine diagnostics into transparent single/pair/abstention evidence.

    The function does not tune thresholds to observed outcomes.  It consumes a
    pre-registered ``strong_single``/``robust_pair`` mapping or documented
    conservative defaults.
    """

    strong_rules, pair_rules = _release_rule_sections(release_thresholds)
    bundles = sorted(consensus["bundle_id"].astype(str).unique())
    year_rank_columns = sorted(
        [column for column in consensus.columns if column.startswith("year") and column.endswith("_rank")]
    )
    year_score_columns = sorted(
        [column for column in consensus.columns if column.startswith("year") and column.endswith("_score")]
    )
    if len(year_rank_columns) != 2 or len(year_score_columns) != 2:
        raise ValueError("Consensus must expose exactly two year-specific ranks and scores")

    rows: list[dict[str, Any]] = []
    for bundle_id in bundles:
        consensus_bundle = consensus.loc[consensus["bundle_id"].eq(bundle_id)].copy()
        consensus_bundle = consensus_bundle.sort_values("cross_year_consensus_rank")
        candidate_primary = str(consensus_bundle.iloc[0]["season"])
        candidate_fallback = str(consensus_bundle.iloc[1]["season"])
        year_primary = [
            str(consensus_bundle.sort_values(column).iloc[0]["season"])
            for column in year_rank_columns
        ]
        year_top2 = [
            set(consensus_bundle.sort_values(column).head(2)["season"].astype(str))
            for column in year_rank_columns
        ]
        cross_year_top1_agreement = float(year_primary[0] == year_primary[1])
        year_top2_jaccard = len(year_top2[0] & year_top2[1]) / len(year_top2[0] | year_top2[1])
        year_margin_normalized: list[float] = []
        for score_column in year_score_columns:
            values = np.array(
                [
                    float(
                        consensus_bundle.loc[
                            consensus_bundle["season"].eq(season), score_column
                        ].iloc[0]
                    )
                    for season in SEASON_ORDER
                ]
            )
            year_margin_normalized.append(float(_profile(values)["top1_margin_normalized"]))
        pooled_values = np.array(
            [
                float(
                    consensus_bundle.loc[
                        consensus_bundle["season"].eq(season), "pooled_score"
                    ].iloc[0]
                )
                for season in SEASON_ORDER
            ]
        )
        pooled_profile = _profile(pooled_values)

        transfer = cross_year.detail.loc[cross_year.detail["bundle_id"].eq(bundle_id)]
        lomo_row = lomo.summary.loc[lomo.summary["bundle_id"].eq(bundle_id)].iloc[0]
        loso_row = loso.summary.loc[loso.summary["bundle_id"].eq(bundle_id)].iloc[0]
        boundary_row = season_boundary.summary.loc[
            season_boundary.summary["bundle_id"].eq(bundle_id)
        ].iloc[0]
        bootstrap_bundle = bootstrap.summary.loc[bootstrap.summary["bundle_id"].eq(bundle_id)]
        bootstrap_primary = float(
            bootstrap_bundle.loc[
                bootstrap_bundle["season"].eq(candidate_primary), "primary_probability"
            ].iloc[0]
        )
        bootstrap_pair = float(
            bootstrap_bundle.loc[
                bootstrap_bundle["season"].isin([candidate_primary, candidate_fallback]),
                "top2_probability",
            ].min()
        )
        permutation_row = permutation.summary.loc[
            permutation.summary["bundle_id"].eq(bundle_id)
            & permutation.summary["metric"].eq("top1_margin_normalized")
        ].iloc[0]
        sensitivity_rows = working_day_sensitivity.loc[
            working_day_sensitivity["bundle_id"].eq(bundle_id)
            & working_day_sensitivity["variant"].isin(
                ["raw_monthly_utilization", "year_normalized_working_day"]
            )
        ]
        working_primary_retention = float(sensitivity_rows["top1_agreement"].min())
        working_top2_retention = float(sensitivity_rows["top2_agreement"].min())
        bidirectional_primary_top2 = float(transfer["train_primary_in_holdout_top2"].mean())
        bidirectional_pair_coverage = float(transfer["pair_top2_coverage"].mean())
        regret = float(transfer["heldout_regret_normalized"].median())
        permutation_strength = float(permutation_row["permutation_strength"])

        metrics = {
            "cross_year_top1_agreement": cross_year_top1_agreement,
            "sigungu_unit_top1_agreement": float(transfer["top1_agreement"].mean()),
            "bidirectional_primary_in_holdout_top2": bidirectional_primary_top2,
            "bidirectional_selected_pair_holdout_top2_coverage": bidirectional_pair_coverage,
            "year_top2_jaccard": year_top2_jaccard,
            "heldout_normalized_regret": regret,
            "pooled_normalized_top1_margin": float(pooled_profile["top1_margin_normalized"]),
            "year_normalized_top1_margin": float(min(year_margin_normalized)),
            "bootstrap_primary_probability": bootstrap_primary,
            "bootstrap_pair_coverage_probability": bootstrap_pair,
            "lomo_primary_retention": float(lomo_row["lomo_primary_retention"]),
            "lomo_top2_retention": float(lomo_row["lomo_top2_retention"]),
            "loso_primary_retention": float(loso_row["loso_primary_retention"]),
            "loso_top2_retention": float(loso_row["loso_top2_retention"]),
            "working_day_primary_retention": working_primary_retention,
            "working_day_top2_retention": working_top2_retention,
            "boundary_primary_retention": float(boundary_row["boundary_primary_retention"]),
            "boundary_top2_retention": float(boundary_row["boundary_top2_retention"]),
            "permutation_margin_percentile": permutation_strength,
        }

        def meets(rule: Mapping[str, float], names: Mapping[str, str]) -> tuple[bool, list[str]]:
            failures: list[str] = []
            for rule_name, metric_name in names.items():
                if rule_name not in rule:
                    raise KeyError(f"Release rule lacks {rule_name}")
                threshold = float(rule[rule_name])
                value = metrics[metric_name]
                if rule_name.endswith("_max"):
                    passed = value <= threshold + 1e-12
                else:
                    passed = value + 1e-12 >= threshold
                if not passed:
                    failures.append(f"{metric_name}={value:.4f} vs {threshold:.4f}")
            return not failures, failures

        strong_names = {
            "cross_year_top1_agreement_min": "cross_year_top1_agreement",
            "bidirectional_primary_in_holdout_top2_min": "bidirectional_primary_in_holdout_top2",
            "bidirectional_selected_pair_holdout_top2_coverage_min": "bidirectional_selected_pair_holdout_top2_coverage",
            "heldout_normalized_regret_max": "heldout_normalized_regret",
            "pooled_normalized_top1_margin_min": "pooled_normalized_top1_margin",
            "year_normalized_top1_margin_min": "year_normalized_top1_margin",
            "bootstrap_primary_probability_min": "bootstrap_primary_probability",
            "lomo_primary_retention_min": "lomo_primary_retention",
            "loso_primary_retention_min": "loso_primary_retention",
            "working_day_primary_retention_min": "working_day_primary_retention",
            "boundary_primary_retention_min": "boundary_primary_retention",
            "permutation_margin_percentile_min": "permutation_margin_percentile",
        }
        pair_names = {
            "bidirectional_primary_in_holdout_top2_min": "bidirectional_primary_in_holdout_top2",
            "bidirectional_selected_pair_holdout_top2_coverage_min": "bidirectional_selected_pair_holdout_top2_coverage",
            "year_top2_jaccard_min": "year_top2_jaccard",
            "heldout_normalized_regret_max": "heldout_normalized_regret",
            "pooled_normalized_top1_margin_min": "pooled_normalized_top1_margin",
            "bootstrap_pair_coverage_probability_min": "bootstrap_pair_coverage_probability",
            "lomo_top2_retention_min": "lomo_top2_retention",
            "loso_top2_retention_min": "loso_top2_retention",
            "working_day_top2_retention_min": "working_day_top2_retention",
            "boundary_top2_retention_min": "boundary_top2_retention",
            "permutation_margin_percentile_min": "permutation_margin_percentile",
        }
        strong_pass, strong_failures = meets(strong_rules, strong_names)
        pair_pass, pair_failures = meets(pair_rules, pair_names)
        if strong_pass:
            release_type, confidence = "STRONG_SINGLE", "HIGH"
            final_primary: object = candidate_primary
            final_fallback: object = candidate_fallback
            reason = "all preregistered strong-single evidence rules passed"
        elif pair_pass:
            release_type, confidence = "ROBUST_PAIR", "MODERATE"
            final_primary = candidate_primary
            final_fallback = candidate_fallback
            reason = "strong-single failed; all preregistered robust-pair rules passed"
        else:
            release_type, confidence = "NO_STRONG_PREFERENCE", "LOW"
            final_primary = pd.NA
            final_fallback = pd.NA
            reason = "abstained; robust-pair failures: " + "; ".join(pair_failures)
        rows.append(
            {
                "bundle_id": bundle_id,
                "year2022_top1": year_primary[0],
                "year2022_top2": "|".join(sorted(year_top2[0], key=SEASON_ORDER.index)),
                "year2023_top1": year_primary[1],
                "year2023_top2": "|".join(sorted(year_top2[1], key=SEASON_ORDER.index)),
                "pooled_top1": pooled_profile["primary"],
                "pooled_top2": "|".join(
                    sorted(pooled_profile["top2"], key=SEASON_ORDER.index)
                ),
                "consensus_primary_candidate": candidate_primary,
                "consensus_fallback_candidate": candidate_fallback,
                **metrics,
                "permutation_p_value_margin": float(permutation_row["p_value_upper"]),
                "release_type": release_type,
                "confidence": confidence,
                "final_primary": final_primary,
                "final_fallback": final_fallback,
                "exact_month": pd.NA,
                "exact_date": pd.NA,
                "temporal_independent_n": int(
                    cross_year.detail["temporal_independent_n"].iloc[0]
                ),
                "temporal_independent_unit": TEMPORAL_INDEPENDENT_UNIT,
                "strong_single_gate_pass": strong_pass,
                "robust_pair_gate_pass": pair_pass,
                "strong_single_failures": "; ".join(strong_failures),
                "robust_pair_failures": "; ".join(pair_failures),
                "reason": reason,
                "limitations": (
                    "two observed NHIS years; insured utilisation is not mobile-clinic demand; "
                    "season-only prior; no exact month/date"
                ),
            }
        )
    output = pd.DataFrame(rows).sort_values("bundle_id").reset_index(drop=True)
    if (
        not set(output["release_type"]).issubset(RELEASE_TYPES)
        or not set(output["confidence"]).issubset(CONFIDENCE_LEVELS)
        or output.loc[
            output["release_type"].eq("NO_STRONG_PREFERENCE"),
            ["final_primary", "final_fallback"],
        ].notna().any().any()
        or output[["exact_month", "exact_date"]].notna().any().any()
    ):
        raise RuntimeError("Abstention/release contract failed")
    return output


def run_temporal_hardening(
    nhis: pd.DataFrame,
    working_days: pd.DataFrame,
    bundle_config: Mapping[str, Any],
    *,
    n_boot: int,
    n_permutations: int,
    years: Sequence[int] = OBSERVED_YEARS,
    expected_policy_sigungu: int = 11,
    random_state: int = 42,
    n_jobs: int = 1,
    release_thresholds: Mapping[str, Any] | None = None,
) -> TemporalHardeningResult:
    """Execute the complete deterministic Stage 2B statistical hardening suite."""

    evidence = prepare_temporal_hardening_evidence(
        nhis,
        working_days,
        bundle_config,
        years=years,
        expected_policy_sigungu=expected_policy_sigungu,
    )
    cross_year = strict_cross_year_transfer(evidence.bundle_month, years=years)
    consensus = build_cross_year_consensus(evidence.bundle_month, years=years)
    lomo = leave_one_month_out(evidence.bundle_month, years=years)
    working_sensitivity = working_day_normalization_sensitivity(
        evidence.bundle_month, years=years
    )
    loso = leave_one_sigungu_out(evidence.bundle_month, years=years)
    boundary = season_boundary_sensitivity(evidence.bundle_month, years=years)
    bootstrap = sigungu_block_bootstrap(
        evidence.bundle_month,
        n_boot=n_boot,
        years=years,
        random_state=random_state,
        n_jobs=n_jobs,
        expected_n_boot=n_boot,
    )
    permutation = month_label_permutation_null(
        evidence.service_month,
        bundle_config,
        n_permutations=n_permutations,
        years=years,
        random_state=random_state,
        n_jobs=n_jobs,
        expected_n_permutations=n_permutations,
    )
    bundle_summary = build_abstention_evidence_summary(
        cross_year,
        consensus,
        lomo,
        working_sensitivity,
        loso,
        boundary,
        bootstrap,
        permutation,
        release_thresholds=release_thresholds,
    )
    return TemporalHardeningResult(
        evidence=evidence,
        cross_year=cross_year,
        consensus=consensus,
        lomo=lomo,
        working_day_sensitivity=working_sensitivity,
        loso=loso,
        season_boundary=boundary,
        bootstrap=bootstrap,
        permutation=permutation,
        bundle_evidence_summary=bundle_summary,
    )


__all__ = [
    "CONFIDENCE_LEVELS",
    "CrossYearTransferResult",
    "DiagnosticResult",
    "OBSERVED_YEARS",
    "PRIMARY_SEASON_DEFINITION",
    "PRIMARY_VALUE_COLUMN",
    "RELEASE_TYPES",
    "SEASON_BOUNDARY_DEFINITIONS",
    "SEASON_ORDER",
    "SENSITIVITY_VALUE_COLUMNS",
    "TEMPORAL_INDEPENDENT_UNIT",
    "TemporalHardeningEvidence",
    "TemporalHardeningResult",
    "build_abstention_evidence_summary",
    "build_cross_year_consensus",
    "leave_one_month_out",
    "leave_one_sigungu_out",
    "month_label_permutation_null",
    "prepare_temporal_hardening_evidence",
    "run_temporal_hardening",
    "season_boundary_sensitivity",
    "sigungu_block_bootstrap",
    "strict_cross_year_transfer",
    "working_day_normalization_sensitivity",
]
