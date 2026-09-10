"""Exploratory Stage 2B hardening candidates with fail-closed promotion gates.

This module is intentionally separate from :mod:`mediroad.temporal.hardening`.
It never writes artifacts and cannot mutate or replace the official Stage 2B
release.  It preserves the pre-registered 72-candidate iteration-2 ledger and
evaluates one iteration-3 rule whose representation is selected using the
training year only.

The observed statistical unit remains policy-sigungu x service-bundle (n=55
in production).  The 153 admin-dong delivery rows are never treated as
independent temporal observations.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .hardening import OBSERVED_YEARS, PRIMARY_SEASON_DEFINITION, SEASON_ORDER


REPRESENTATIONS = (
    "baseline",
    "composition_share",
    "log_common_mean_residual",
    "log_common_median_residual",
)
SEASONAL_AGGREGATORS = ("mean", "median")
EB_CENTERS = ("mean", "median")
EB_NOISE_MULTIPLIERS = (0.5, 1.0, 2.0, 4.0)
EXPECTED_CANDIDATE_COUNT = 72
ITERATION3_CANDIDATE_ID = "train_year_common_variance_gated_composition"
ITERATION3_THRESHOLD = 0.50
BASELINE_CANDIDATE_ID = "baseline__local__mean"
NO_PROMOTION_DECISION = "NO_PROMOTION_KEEP_PASS_ADVISORY_TEMPORAL"
EXPLORATORY_DECISION = "EXPLORATORY_ONLY_DUE_NO_THIRD_YEAR"
OFFICIAL_HARDENING_OUTPUT_PATH = "outputs/model_v1/05_stage2b_hardening"
EXPECTED_ITERATION2_CONTRACT_MAPPING_SHA256 = (
    "ab243b9ec1e40986aa9712cc59e6cd2d3f33ff99930314fa173d450e9ff28abb"
)
_TOLERANCE = 1e-12


@dataclass(frozen=True)
class CandidateGridResult:
    """The locked 72-grid and its no-harm comparison to the local baseline."""

    ledger: pd.DataFrame
    bundle_ledger: pd.DataFrame
    audit: dict[str, Any]


@dataclass(frozen=True)
class NestedCandidateResult:
    """Evidence for the single train-year-only iteration-3 candidate."""

    selection: pd.DataFrame
    cross_year_detail: pd.DataFrame
    cross_year_summary: pd.DataFrame
    lomo_detail: pd.DataFrame
    lomo_summary: pd.DataFrame
    loso_detail: pd.DataFrame
    loso_summary: pd.DataFrame
    bootstrap_detail: pd.DataFrame
    bootstrap_summary: pd.DataFrame
    permutation_detail: pd.DataFrame
    permutation_summary: pd.DataFrame
    release_evidence: pd.DataFrame
    comparison: pd.DataFrame
    audit: dict[str, Any]


@dataclass(frozen=True)
class HardeningCandidatesResult:
    """Complete read-only candidate evaluation."""

    grid: CandidateGridResult
    nested: NestedCandidateResult
    promotion_decision: str
    audit: dict[str, Any]


@dataclass(frozen=True)
class _Panel:
    sigungu: tuple[str, ...]
    years: tuple[int, int]
    services: tuple[str, ...]
    bundles: tuple[str, ...]
    rates: np.ndarray  # sigungu x year x service x month
    weights: np.ndarray  # bundle x service


def _positive_integer(value: int, label: str) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return int(value)


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise KeyError(f"{label} lacks required columns: {missing}")


def _bundle_spec(bundle_config: Mapping[str, Any]) -> tuple[tuple[str, ...], tuple[str, ...], np.ndarray]:
    bundles = list(bundle_config.get("bundles", []))
    if not bundles:
        raise ValueError("Service bundle config contains no bundles")
    bundle_ids = tuple(str(bundle["bundle_id"]) for bundle in bundles)
    if len(bundle_ids) != len(set(bundle_ids)):
        raise ValueError("Service bundle IDs must be unique")
    service_ids = tuple(
        sorted(
            {
                str(service_id)
                for bundle in bundles
                for service_id in dict(bundle.get("included_services", {}))
            }
        )
    )
    weights = np.zeros((len(bundle_ids), len(service_ids)), dtype=float)
    index = {service_id: position for position, service_id in enumerate(service_ids)}
    for bundle_index, bundle in enumerate(bundles):
        included = {
            str(service_id): float(weight)
            for service_id, weight in dict(bundle.get("included_services", {})).items()
        }
        if not included or not np.isclose(sum(included.values()), 1.0, atol=1e-12):
            raise ValueError(f"Bundle {bundle_ids[bundle_index]} weights must sum to one")
        if any(not np.isfinite(weight) or weight < 0 for weight in included.values()):
            raise ValueError(f"Bundle {bundle_ids[bundle_index]} has invalid weights")
        for service_id, weight in included.items():
            weights[bundle_index, index[service_id]] = weight
    return bundle_ids, service_ids, weights


def _panel(
    service_month: pd.DataFrame,
    bundle_config: Mapping[str, Any],
    *,
    years: Sequence[int],
    expected_policy_sigungu: int,
) -> _Panel:
    _require_columns(
        service_month,
        {"policy_sigungu_name", "service_id", "year", "month", "working_day_rate"},
        "Service-month evidence",
    )
    configured_years = tuple(sorted(int(year) for year in years))
    if len(configured_years) != 2 or len(set(configured_years)) != 2:
        raise ValueError("Candidate hardening requires exactly two distinct years")
    observed_years = tuple(
        sorted(pd.to_numeric(service_month["year"], errors="raise").astype(int).unique())
    )
    if observed_years != configured_years:
        raise ValueError(
            f"Two-year leakage guard rejected {observed_years}; expected {configured_years}"
        )
    expected_policy_sigungu = _positive_integer(
        expected_policy_sigungu, "expected_policy_sigungu"
    )
    sigungu = tuple(sorted(service_month["policy_sigungu_name"].astype(str).unique()))
    if len(sigungu) != expected_policy_sigungu:
        raise ValueError(
            f"Expected {expected_policy_sigungu} policy sigungu, observed {len(sigungu)}"
        )
    bundle_ids, service_ids, weights = _bundle_spec(bundle_config)
    missing = sorted(set(service_ids) - set(service_month["service_id"].astype(str)))
    if missing:
        raise KeyError(f"Service-month evidence lacks configured services: {missing}")
    work = service_month.loc[
        service_month["service_id"].astype(str).isin(service_ids),
        ["policy_sigungu_name", "year", "service_id", "month", "working_day_rate"],
    ].copy()
    work["policy_sigungu_name"] = work["policy_sigungu_name"].astype(str)
    work["service_id"] = work["service_id"].astype(str)
    work["year"] = pd.to_numeric(work["year"], errors="raise").astype(int)
    work["month"] = pd.to_numeric(work["month"], errors="raise").astype(int)
    work["working_day_rate"] = pd.to_numeric(
        work["working_day_rate"], errors="raise"
    ).astype(float)
    keys = ["policy_sigungu_name", "year", "service_id", "month"]
    expected_rows = len(sigungu) * 2 * len(service_ids) * 12
    if (
        len(work) != expected_rows
        or work.duplicated(keys).any()
        or not work["month"].between(1, 12).all()
    ):
        raise ValueError("Service-month evidence is not a unique rectangular panel")
    values = work["working_day_rate"].to_numpy(float)
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("working_day_rate must be finite and strictly positive")
    indexed = work.set_index(keys)["working_day_rate"]
    rates = np.empty((len(sigungu), 2, len(service_ids), 12), dtype=float)
    for sigungu_index, name in enumerate(sigungu):
        for year_index, year in enumerate(configured_years):
            for service_index, service_id in enumerate(service_ids):
                rates[sigungu_index, year_index, service_index] = [
                    float(indexed.loc[(name, year, service_id, month)])
                    for month in range(1, 13)
                ]
    return _Panel(sigungu, configured_years, service_ids, bundle_ids, rates, weights)


def canonical_contract_mapping_sha256(config: Mapping[str, Any]) -> str:
    """Hash the complete canonical config mapping except its seal field."""

    if not isinstance(config, Mapping):
        raise TypeError("Iteration-2 config must be a mapping")
    unsealed = dict(config)
    unsealed.pop("contract_mapping_sha256", None)
    try:
        payload = json.dumps(
            unsealed,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError("Iteration-2 config is not canonical-JSON serializable") from error
    return hashlib.sha256(payload).hexdigest()


def _panel_contract_sha256(panel: _Panel) -> str:
    """Seal the exact normalized panel axes, rates, bundles, and weights."""

    digest = hashlib.sha256()
    axes = {
        "sigungu": panel.sigungu,
        "years": panel.years,
        "services": panel.services,
        "bundles": panel.bundles,
        "rate_shape": panel.rates.shape,
        "weight_shape": panel.weights.shape,
    }
    digest.update(
        json.dumps(
            axes,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    )
    digest.update(np.ascontiguousarray(panel.rates, dtype="<f8").tobytes())
    digest.update(np.ascontiguousarray(panel.weights, dtype="<f8").tobytes())
    return digest.hexdigest()


def validate_iteration2_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed if the pre-registered iteration-2/3 grid was changed."""

    configured_seal = config.get("contract_mapping_sha256")
    recomputed_seal = canonical_contract_mapping_sha256(config)
    if (
        type(configured_seal) is not str
        or configured_seal != EXPECTED_ITERATION2_CONTRACT_MAPPING_SHA256
        or recomputed_seal != EXPECTED_ITERATION2_CONTRACT_MAPPING_SHA256
    ):
        raise ValueError(
            "Iteration-2 canonical contract mapping SHA256 seal mismatch"
        )

    def exact_bool(
        section: Mapping[str, Any], key: str, expected: bool, label: str
    ) -> None:
        value = section.get(key)
        if type(value) is not bool or value is not expected:
            raise ValueError(f"{label} must be exactly {expected}")

    def exact_integer(
        section: Mapping[str, Any], key: str, expected: int, label: str
    ) -> None:
        value = section.get(key)
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or int(value) != expected
        ):
            raise ValueError(f"{label} must be exactly {expected}")

    def exact_number(
        section: Mapping[str, Any], key: str, expected: float, label: str
    ) -> None:
        value = section.get(key)
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, float, np.integer, np.floating))
            or not np.isclose(float(value), expected, rtol=0, atol=0)
        ):
            raise ValueError(f"{label} must be exactly {expected}")

    exact_bool(config, "stage3_started", False, "Stage3 started guard")
    frozen = dict(config.get("frozen_baseline", {}))
    if (
        type(frozen.get("official_output_tree_mutation_forbidden")) is not str
        or frozen.get("official_output_tree_mutation_forbidden")
        != OFFICIAL_HARDENING_OUTPUT_PATH
    ):
        raise ValueError(
            "Frozen official output mutation-forbidden path differs from the locked path"
        )

    grid = dict(config.get("candidate_grid", {}))
    representations = tuple(str(value) for value in grid.get("representations", []))
    aggregators = tuple(str(value) for value in grid.get("seasonal_aggregators", []))
    spatial_estimators = dict(grid.get("spatial_estimators", {}))
    local = dict(spatial_estimators.get("local", {}))
    exact_bool(local, "enabled", True, "Iteration-2 local estimator enabled guard")
    empirical_bayes = dict(spatial_estimators.get("empirical_bayes", {}))
    centers = tuple(str(value) for value in empirical_bayes.get("centers", []))
    multipliers = tuple(float(value) for value in empirical_bayes.get("noise_multipliers", []))
    exact_integer(
        grid,
        "candidate_count_expected",
        EXPECTED_CANDIDATE_COUNT,
        "Iteration-2 candidate count",
    )
    expected_count = EXPECTED_CANDIDATE_COUNT
    exact_bool(
        grid,
        "fixed_before_full_grid_result",
        True,
        "Iteration-2 pre-result grid lock",
    )
    if representations != REPRESENTATIONS:
        raise ValueError("Iteration-2 representations differ from the locked grid")
    if aggregators != SEASONAL_AGGREGATORS:
        raise ValueError("Iteration-2 seasonal aggregators differ from the locked grid")
    if centers != EB_CENTERS or multipliers != EB_NOISE_MULTIPLIERS:
        raise ValueError("Iteration-2 empirical-Bayes grid differs from the locked grid")
    if expected_count != EXPECTED_CANDIDATE_COUNT:
        raise ValueError("Iteration-2 candidate_count_expected must equal 72")
    nested = dict(config.get("iteration3_single_nested_candidate", {}))
    exact_bool(
        nested,
        "threshold_grid_search_forbidden",
        True,
        "Iteration-3 threshold search guard",
    )
    exact_bool(
        nested,
        "original_bundle_weights_unchanged",
        True,
        "Iteration-3 bundle-weight integrity guard",
    )
    if (
        str(nested.get("candidate_id")) != ITERATION3_CANDIDATE_ID
        or str(nested.get("selection_data")) != "train_year_only"
        or not np.isclose(
            float(nested.get("common_monthly_variance_ratio_threshold", np.nan)),
            ITERATION3_THRESHOLD,
            atol=0,
        )
    ):
        raise ValueError("Iteration-3 single nested candidate is not the locked rule")

    no_harm = dict(config.get("promotion_no_harm_contract", {}))
    bundle_caps = dict(no_harm.get("bundle_harm_caps", {}))
    exact_number(
        bundle_caps,
        "maximum_top2_retention_decline",
        0.05,
        "Bundle Top2 harm cap",
    )
    exact_number(
        bundle_caps,
        "maximum_normalized_regret_increase",
        0.05,
        "Bundle normalized-regret harm cap",
    )
    final_requirements = dict(no_harm.get("final_candidate_requirements", {}))
    exact_integer(
        final_requirements,
        "bootstrap_replicates",
        1000,
        "Final candidate bootstrap contract",
    )
    exact_integer(
        final_requirements,
        "permutation_replicates",
        5000,
        "Final candidate permutation contract",
    )
    exact_integer(
        final_requirements,
        "jobs",
        8,
        "Final candidate jobs contract",
    )

    decision = dict(config.get("decision_contract", {}))
    exact_bool(
        decision,
        "stage3_execution_forbidden",
        True,
        "Decision-contract Stage3 execution guard",
    )
    output = dict(config.get("output_contract", {}))
    exact_bool(
        output,
        "official_hardening_tree_must_remain_unchanged",
        True,
        "Official hardening output-tree immutability guard",
    )
    return {
        "contract_mapping_sha256": recomputed_seal,
        "candidate_count_expected": expected_count,
        "iteration3_threshold": ITERATION3_THRESHOLD,
        "threshold_grid_search_forbidden": True,
        "stage3_started": False,
        "official_output_tree_mutation_forbidden": OFFICIAL_HARDENING_OUTPUT_PATH,
        "maximum_top2_retention_decline": 0.05,
        "maximum_normalized_regret_increase": 0.05,
        "final_bootstrap_replicates": 1000,
        "final_permutation_replicates": 5000,
        "final_jobs": 8,
        "stage3_execution_forbidden": True,
    }


def _representation(rates: np.ndarray, name: str) -> np.ndarray:
    """Return annual-mean-one service profiles; service is the penultimate axis."""

    if not np.isfinite(rates).all() or np.any(rates <= 0):
        raise ValueError("Candidate representations require positive finite rates")
    if name == "baseline":
        values = rates / rates.mean(axis=-1, keepdims=True)
    elif name == "composition_share":
        values = rates / rates.sum(axis=-2, keepdims=True)
        values = values / values.mean(axis=-1, keepdims=True)
    elif name in {"log_common_mean_residual", "log_common_median_residual"}:
        logged = np.log(rates)
        if name == "log_common_mean_residual":
            common = logged.mean(axis=-2, keepdims=True)
        else:
            common = np.median(logged, axis=-2, keepdims=True)
        values = np.exp(logged - common)
        values = values / values.mean(axis=-1, keepdims=True)
    else:
        raise ValueError(f"Unknown representation: {name}")
    if not np.isfinite(values).all():
        raise RuntimeError(f"Representation {name} produced non-finite values")
    return values


def _bundle_values(service_values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    # prefix x sigungu x year x service x month -> prefix x sigungu x year x bundle x month
    return np.einsum("bs,...gysm->...gybm", weights, service_values, optimize=True)


def _season_month_indices() -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(int(month) - 1 for month in PRIMARY_SEASON_DEFINITION[season])
        for season in SEASON_ORDER
    )


def _seasonal(
    bundle_values: np.ndarray,
    aggregator: str,
    *,
    removed_month: int | None = None,
    definition: Mapping[str, Sequence[int]] = PRIMARY_SEASON_DEFINITION,
) -> np.ndarray:
    if aggregator not in SEASONAL_AGGREGATORS:
        raise ValueError(f"Unknown seasonal aggregator: {aggregator}")
    outputs: list[np.ndarray] = []
    for season in SEASON_ORDER:
        indices = [
            int(month) - 1
            for month in definition[season]
            if removed_month is None or int(month) != int(removed_month)
        ]
        if not indices:
            raise ValueError("Month removal emptied a season")
        values = bundle_values[..., indices]
        function = np.mean if aggregator == "mean" else np.median
        outputs.append(function(values, axis=-1))
    return np.stack(outputs, axis=-1)


def _grid_profiles(
    bundle_values: np.ndarray,
    aggregator: str,
    *,
    removed_month: int | None = None,
) -> tuple[np.ndarray, tuple[tuple[str, str | None, float | None], ...]]:
    """Profiles for local plus the eight fixed EB estimators.

    Reliability is estimated from the training-year panel only.  The EB center
    is leave-one-sigungu-out; the scalar reliability uses between-sigungu
    seasonal variance and within-season monthly noise for that year/bundle.
    """

    seasonal = _seasonal(bundle_values, aggregator, removed_month=removed_month)
    # seasonal axes: prefix, sigungu, year, bundle, season
    sigungu_axis = seasonal.ndim - 4
    sigungu_n = seasonal.shape[sigungu_axis]
    if sigungu_n < 2:
        raise ValueError("Empirical-Bayes candidates require at least two sigungu")
    summed = seasonal.sum(axis=sigungu_axis, keepdims=True)
    loo_mean = (summed - seasonal) / (sigungu_n - 1)
    loo_median = np.empty_like(seasonal)
    for sigungu_index in range(sigungu_n):
        target = [slice(None)] * seasonal.ndim
        target[sigungu_axis] = sigungu_index
        loo_median[tuple(target)] = np.median(
            np.delete(seasonal, sigungu_index, axis=sigungu_axis),
            axis=sigungu_axis,
        )

    # Shared train-year reliability avoids using holdout outcomes and keeps the
    # same fixed estimator for every sigungu in a direction.
    between = np.var(seasonal, axis=sigungu_axis, ddof=0).mean(axis=-1, keepdims=True)
    residual_blocks: list[np.ndarray] = []
    for season_index, months in enumerate(_season_month_indices()):
        retained = [
            month
            for month in months
            if removed_month is None or month != int(removed_month) - 1
        ]
        residual_blocks.append(
            bundle_values[..., retained] - seasonal[..., season_index, None]
        )
    within = np.mean(
        np.concatenate(residual_blocks, axis=-1) ** 2,
        axis=(sigungu_axis, -1),
    )[..., None]
    insert_sigungu = np.expand_dims
    profiles = [seasonal]
    definitions: list[tuple[str, str | None, float | None]] = [("local", None, None)]
    for center_name, center in (("mean", loo_mean), ("median", loo_median)):
        for multiplier in EB_NOISE_MULTIPLIERS:
            reliability = between / (between + multiplier * within + 1e-15)
            reliability = insert_sigungu(reliability, axis=sigungu_axis)
            profiles.append(reliability * seasonal + (1.0 - reliability) * center)
            definitions.append(("empirical_bayes", center_name, multiplier))
    return np.stack(profiles, axis=sigungu_axis), tuple(definitions)


def _orders(values: np.ndarray) -> np.ndarray:
    return np.argsort(-values, axis=-1, kind="stable")


def _rank_vectors(values: np.ndarray) -> np.ndarray:
    return np.argsort(_orders(values), axis=-1, kind="stable") + 1


def _unit_metrics(predicted: np.ndarray, holdout: np.ndarray) -> dict[str, np.ndarray]:
    predicted_order = _orders(predicted)
    holdout_order = _orders(holdout)
    top1 = (predicted_order[..., 0] == holdout_order[..., 0]).astype(float)
    top2 = np.any(
        predicted_order[..., 0, None] == holdout_order[..., :2], axis=-1
    ).astype(float)
    pair = np.sum(
        predicted_order[..., :2, None] == holdout_order[..., None, :2],
        axis=(-2, -1),
    ) / 2.0
    predicted_rank = _rank_vectors(predicted).astype(float)
    holdout_rank = _rank_vectors(holdout).astype(float)
    centered_predicted = predicted_rank - predicted_rank.mean(axis=-1, keepdims=True)
    centered_holdout = holdout_rank - holdout_rank.mean(axis=-1, keepdims=True)
    denominator = np.sqrt(
        np.sum(centered_predicted**2, axis=-1)
        * np.sum(centered_holdout**2, axis=-1)
    )
    rho = np.sum(centered_predicted * centered_holdout, axis=-1) / np.where(
        denominator > 1e-15, denominator, 1.0
    )
    score_range = np.ptp(holdout, axis=-1)
    chosen = np.take_along_axis(holdout, predicted_order[..., :1], axis=-1)[..., 0]
    regret = (np.max(holdout, axis=-1) - chosen) / np.where(
        score_range > 1e-15, score_range, 1.0
    )
    return {
        "top1_agreement": top1,
        "primary_in_holdout_top2": top2,
        "selected_pair_holdout_top2_coverage": pair,
        "rank_spearman": rho,
        "normalized_regret": regret,
    }


def _retention(full: np.ndarray, variant: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    full_order = _orders(full)
    variant_order = _orders(variant)
    primary = (full_order[..., 0] == variant_order[..., 0]).astype(float)
    top2 = np.sum(
        full_order[..., :2, None] == variant_order[..., None, :2], axis=(-2, -1)
    ) / 2.0
    return primary, top2


def _normalized_margin(values: np.ndarray) -> np.ndarray:
    ordered = np.sort(values, axis=-1)
    score_range = np.ptp(values, axis=-1)
    return (ordered[..., -1] - ordered[..., -2]) / np.where(
        score_range > 1e-15, score_range, 1.0
    )


def _candidate_id(
    representation: str,
    aggregator: str,
    estimator: str,
    center: str | None,
    multiplier: float | None,
) -> str:
    if estimator == "local":
        return f"{representation}__local__{aggregator}"
    return f"{representation}__eb_{center}_k{multiplier:g}__{aggregator}"


def _candidate_definitions() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for representation in REPRESENTATIONS:
        for aggregator in SEASONAL_AGGREGATORS:
            for estimator, center, multiplier in (
                [("local", None, None)]
                + [
                    ("empirical_bayes", center_name, multiplier_value)
                    for center_name in EB_CENTERS
                    for multiplier_value in EB_NOISE_MULTIPLIERS
                ]
            ):
                rows.append(
                    {
                        "candidate_id": _candidate_id(
                            representation, aggregator, estimator, center, multiplier
                        ),
                        "representation": representation,
                        "seasonal_aggregator": aggregator,
                        "spatial_estimator": estimator,
                        "eb_center": center,
                        "noise_multiplier": multiplier,
                    }
                )
    output = pd.DataFrame(rows)
    if len(output) != EXPECTED_CANDIDATE_COUNT or not output["candidate_id"].is_unique:
        raise RuntimeError("Locked candidate grid did not produce 72 unique candidates")
    return output


def _permutation_seeds(count: int, random_state: int) -> tuple[int, ...]:
    sequence = np.random.SeedSequence(int(random_state))
    return tuple(
        int(child.generate_state(1, dtype=np.uint64)[0])
        for child in sequence.spawn(count)
    )


def _grid_permutation_strength(
    panel: _Panel,
    definitions: pd.DataFrame,
    *,
    n_permutations: int,
    random_state: int,
    batch_size: int,
) -> dict[str, np.ndarray]:
    """Advisory month-label null for every fixed candidate.

    The shuffle is within sigungu x year x service and therefore preserves
    each source block's twelve values and annual total.  Nonlinear
    representations are recomputed after shuffling.
    """

    observed: dict[tuple[str, str], tuple[np.ndarray, tuple[Any, ...]]] = {}
    exceed: dict[tuple[str, str], np.ndarray] = {}
    for representation in REPRESENTATIONS:
        bundle = _bundle_values(_representation(panel.rates, representation), panel.weights)
        for aggregator in SEASONAL_AGGREGATORS:
            profiles, estimator_definitions = _grid_profiles(bundle, aggregator)
            margins = _normalized_margin(profiles.mean(axis=(1, 2)))
            observed[(representation, aggregator)] = (margins, estimator_definitions)
            exceed[(representation, aggregator)] = np.zeros_like(margins, dtype=np.int64)

    seeds = _permutation_seeds(n_permutations, random_state)
    original_totals = panel.rates.sum(axis=-1)
    annual_totals_preserved = True
    for start in range(0, n_permutations, batch_size):
        stop = min(start + batch_size, n_permutations)
        batch = np.empty((stop - start, *panel.rates.shape), dtype=float)
        for offset, seed in enumerate(seeds[start:stop]):
            rng = np.random.default_rng(seed)
            order = np.argsort(
                rng.random(panel.rates.shape), axis=-1, kind="stable"
            )
            batch[offset] = np.take_along_axis(panel.rates, order, axis=-1)
        annual_totals_preserved = annual_totals_preserved and bool(
            np.allclose(batch.sum(axis=-1), original_totals[None, ...], atol=1e-12)
        )
        for representation in REPRESENTATIONS:
            bundle = _bundle_values(_representation(batch, representation), panel.weights)
            for aggregator in SEASONAL_AGGREGATORS:
                profiles, _ = _grid_profiles(bundle, aggregator)
                null_margin = _normalized_margin(profiles.mean(axis=(2, 3)))
                observed_margin = observed[(representation, aggregator)][0]
                exceed[(representation, aggregator)] += np.sum(
                    null_margin >= observed_margin[None, ...], axis=0
                )
    if not annual_totals_preserved:
        raise RuntimeError("Candidate month-label permutation changed annual totals")

    output: dict[str, np.ndarray] = {}
    for (representation, aggregator), counts in exceed.items():
        _, estimator_definitions = observed[(representation, aggregator)]
        strength = 1.0 - (1.0 + counts) / (n_permutations + 1.0)
        for estimator_index, (estimator, center, multiplier) in enumerate(
            estimator_definitions
        ):
            output[
                _candidate_id(
                    representation, aggregator, estimator, center, multiplier
                )
            ] = strength[estimator_index]
    if set(output) != set(definitions["candidate_id"]):
        raise RuntimeError("Permutation results do not cover the locked grid")
    return output


def evaluate_candidate_grid(
    service_month: pd.DataFrame,
    bundle_config: Mapping[str, Any],
    *,
    n_permutations: int,
    years: Sequence[int] = OBSERVED_YEARS,
    expected_policy_sigungu: int = 11,
    random_state: int = 20260818,
    n_jobs: int = 1,
    permutation_batch_size: int = 64,
    iteration_config: Mapping[str, Any] | None = None,
) -> CandidateGridResult:
    """Evaluate the complete locked 72-grid without writing official outputs."""

    n_permutations = _positive_integer(n_permutations, "n_permutations")
    n_jobs = _positive_integer(n_jobs, "n_jobs")
    permutation_batch_size = _positive_integer(
        permutation_batch_size, "permutation_batch_size"
    )
    config_audit = (
        validate_iteration2_config(iteration_config)
        if iteration_config is not None
        else {"candidate_count_expected": EXPECTED_CANDIDATE_COUNT}
    )
    panel = _panel(
        service_month,
        bundle_config,
        years=years,
        expected_policy_sigungu=expected_policy_sigungu,
    )
    panel_seal = _panel_contract_sha256(panel)
    definitions = _candidate_definitions()
    metric_rows: list[dict[str, Any]] = []
    bundle_rows: list[dict[str, Any]] = []
    cached: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, tuple[Any, ...]]] = {}

    for representation in REPRESENTATIONS:
        bundle = _bundle_values(_representation(panel.rates, representation), panel.weights)
        for aggregator in SEASONAL_AGGREGATORS:
            profiles, estimator_definitions = _grid_profiles(bundle, aggregator)
            seasonal = _seasonal(bundle, aggregator)
            cached[(representation, aggregator)] = (
                bundle,
                profiles,
                estimator_definitions,
            )
            directional = []
            for train_index, holdout_index in ((0, 1), (1, 0)):
                holdout = np.broadcast_to(
                    seasonal[:, holdout_index][None, ...],
                    profiles[:, :, train_index].shape,
                )
                directional.append(
                    _unit_metrics(profiles[:, :, train_index], holdout)
                )
            for estimator_index, (estimator, center, multiplier) in enumerate(
                estimator_definitions
            ):
                candidate_id = _candidate_id(
                    representation, aggregator, estimator, center, multiplier
                )
                concatenated = {
                    name: np.concatenate(
                        [direction[name][estimator_index].reshape(-1) for direction in directional]
                    )
                    for name in directional[0]
                }
                metric_rows.append(
                    {
                        "candidate_id": candidate_id,
                        "top1_agreement": float(concatenated["top1_agreement"].mean()),
                        "primary_in_holdout_top2": float(
                            concatenated["primary_in_holdout_top2"].mean()
                        ),
                        "selected_pair_holdout_top2_coverage": float(
                            concatenated[
                                "selected_pair_holdout_top2_coverage"
                            ].mean()
                        ),
                        "rank_spearman_median": float(
                            np.median(concatenated["rank_spearman"])
                        ),
                        "normalized_regret_mean": float(
                            concatenated["normalized_regret"].mean()
                        ),
                    }
                )
                for bundle_index, bundle_id in enumerate(panel.bundles):
                    by_bundle = {
                        name: np.concatenate(
                            [
                                direction[name][estimator_index, :, bundle_index]
                                for direction in directional
                            ]
                        )
                        for name in directional[0]
                    }
                    bundle_rows.append(
                        {
                            "candidate_id": candidate_id,
                            "bundle_id": bundle_id,
                            "top1_agreement": float(by_bundle["top1_agreement"].mean()),
                            "primary_in_holdout_top2": float(
                                by_bundle["primary_in_holdout_top2"].mean()
                            ),
                            "selected_pair_holdout_top2_coverage": float(
                                by_bundle[
                                    "selected_pair_holdout_top2_coverage"
                                ].mean()
                            ),
                            "rank_spearman_median": float(
                                np.median(by_bundle["rank_spearman"])
                            ),
                            "normalized_regret_mean": float(
                                by_bundle["normalized_regret"].mean()
                            ),
                            "directional_unit_n": 2 * len(panel.sigungu),
                        }
                    )

    metrics = pd.DataFrame(metric_rows).set_index("candidate_id")
    lomo_totals: dict[str, list[float]] = {
        candidate_id: [0.0, 0.0, 0.0]
        for candidate_id in definitions["candidate_id"]
    }
    loso_totals: dict[str, list[float]] = {
        candidate_id: [0.0, 0.0, 0.0]
        for candidate_id in definitions["candidate_id"]
    }
    for (representation, aggregator), (bundle, profiles, estimator_definitions) in cached.items():
        for year_index in range(2):
            for removed_month in range(1, 13):
                variant, _ = _grid_profiles(
                    bundle[:, year_index : year_index + 1],
                    aggregator,
                    removed_month=removed_month,
                )
                primary, top2 = _retention(
                    profiles[:, :, year_index], variant[:, :, 0]
                )
                for estimator_index, (estimator, center, multiplier) in enumerate(
                    estimator_definitions
                ):
                    candidate_id = _candidate_id(
                        representation, aggregator, estimator, center, multiplier
                    )
                    lomo_totals[candidate_id][0] += float(primary[estimator_index].sum())
                    lomo_totals[candidate_id][1] += float(top2[estimator_index].sum())
                    lomo_totals[candidate_id][2] += float(primary[estimator_index].size)
        full_release = profiles.mean(axis=(1, 2))
        for removed_sigungu in range(len(panel.sigungu)):
            keep = [
                index
                for index in range(len(panel.sigungu))
                if index != removed_sigungu
            ]
            variant, _ = _grid_profiles(bundle[keep], aggregator)
            variant_release = variant.mean(axis=(1, 2))
            primary, top2 = _retention(full_release, variant_release)
            for estimator_index, (estimator, center, multiplier) in enumerate(
                estimator_definitions
            ):
                candidate_id = _candidate_id(
                    representation, aggregator, estimator, center, multiplier
                )
                loso_totals[candidate_id][0] += float(primary[estimator_index].sum())
                loso_totals[candidate_id][1] += float(top2[estimator_index].sum())
                loso_totals[candidate_id][2] += float(primary[estimator_index].size)

    permutation = _grid_permutation_strength(
        panel,
        definitions,
        n_permutations=n_permutations,
        random_state=random_state,
        batch_size=permutation_batch_size,
    )
    for candidate_id in definitions["candidate_id"]:
        metrics.loc[candidate_id, "lomo_primary_retention"] = (
            lomo_totals[candidate_id][0] / lomo_totals[candidate_id][2]
        )
        metrics.loc[candidate_id, "lomo_top2_retention"] = (
            lomo_totals[candidate_id][1] / lomo_totals[candidate_id][2]
        )
        metrics.loc[candidate_id, "loso_primary_retention"] = (
            loso_totals[candidate_id][0] / loso_totals[candidate_id][2]
        )
        metrics.loc[candidate_id, "loso_top2_retention"] = (
            loso_totals[candidate_id][1] / loso_totals[candidate_id][2]
        )
        metrics.loc[candidate_id, "permutation_strength_mean"] = float(
            permutation[candidate_id].mean()
        )
        metrics.loc[candidate_id, "permutation_strength_min"] = float(
            permutation[candidate_id].min()
        )

    ledger = definitions.merge(metrics.reset_index(), on="candidate_id", validate="one_to_one")
    bundle_ledger = pd.DataFrame(bundle_rows)
    baseline = ledger.loc[ledger["candidate_id"].eq(BASELINE_CANDIDATE_ID)].iloc[0]
    baseline_bundle = bundle_ledger.loc[
        bundle_ledger["candidate_id"].eq(BASELINE_CANDIDATE_ID)
    ].set_index("bundle_id")
    bundle_ledger["top2_change_vs_baseline"] = np.nan
    bundle_ledger["regret_change_vs_baseline"] = np.nan
    bundle_ledger["bundle_no_harm_pass"] = False
    for index, row in bundle_ledger.iterrows():
        reference = baseline_bundle.loc[row["bundle_id"]]
        top2_change = float(
            row["primary_in_holdout_top2"]
            - reference["primary_in_holdout_top2"]
        )
        regret_change = float(
            row["normalized_regret_mean"] - reference["normalized_regret_mean"]
        )
        bundle_ledger.loc[index, "top2_change_vs_baseline"] = top2_change
        bundle_ledger.loc[index, "regret_change_vs_baseline"] = regret_change
        bundle_ledger.loc[index, "bundle_no_harm_pass"] = bool(
            top2_change + _TOLERANCE >= -0.05
            and regret_change <= 0.05 + _TOLERANCE
        )

    ledger["global_no_harm_pass"] = False
    ledger["bundle_no_harm_pass"] = False
    ledger["stage2a_bundle_definition_change_count"] = 0
    ledger["stage2a_what_top1_retention"] = 1.0
    ledger["eligible"] = False
    ledger["failure_reasons"] = ""
    comparison_names = (
        "top1_agreement",
        "primary_in_holdout_top2",
        "selected_pair_holdout_top2_coverage",
    )
    for index, row in ledger.iterrows():
        checks = {
            "top1_not_strictly_better": row[comparison_names[0]]
            <= baseline[comparison_names[0]] + _TOLERANCE,
            "top2_not_strictly_better": row[comparison_names[1]]
            <= baseline[comparison_names[1]] + _TOLERANCE,
            "pair_not_strictly_better": row[comparison_names[2]]
            <= baseline[comparison_names[2]] + _TOLERANCE,
            "rank_spearman_worse": row["rank_spearman_median"]
            < baseline["rank_spearman_median"] - _TOLERANCE,
            "regret_worse": row["normalized_regret_mean"]
            > baseline["normalized_regret_mean"] + _TOLERANCE,
            "lomo_primary_worse": row["lomo_primary_retention"]
            < baseline["lomo_primary_retention"] - _TOLERANCE,
            "lomo_top2_worse": row["lomo_top2_retention"]
            < baseline["lomo_top2_retention"] - _TOLERANCE,
            "loso_primary_worse": row["loso_primary_retention"]
            < baseline["loso_primary_retention"] - _TOLERANCE,
            "loso_top2_worse": row["loso_top2_retention"]
            < baseline["loso_top2_retention"] - _TOLERANCE,
            "permutation_strength_worse": row["permutation_strength_mean"]
            < baseline["permutation_strength_mean"] - _TOLERANCE,
        }
        global_pass = not any(checks.values())
        candidate_bundle = bundle_ledger.loc[
            bundle_ledger["candidate_id"].eq(row["candidate_id"])
        ]
        bundle_pass = bool(candidate_bundle["bundle_no_harm_pass"].all())
        failures = [name for name, failed in checks.items() if failed]
        if not bundle_pass:
            harmed = "|".join(
                candidate_bundle.loc[
                    ~candidate_bundle["bundle_no_harm_pass"], "bundle_id"
                ].astype(str)
            )
            failures.append(f"bundle_harm_caps:{harmed}")
        ledger.loc[index, "global_no_harm_pass"] = global_pass
        ledger.loc[index, "bundle_no_harm_pass"] = bundle_pass
        ledger.loc[index, "eligible"] = bool(global_pass and bundle_pass)
        ledger.loc[index, "failure_reasons"] = ";".join(failures)

    # Backward-compatible ``eligible`` means only that the exploratory no-harm
    # ledger passed.  It can never authorize mutation or official promotion.
    ledger["exploratory_no_harm_eligible"] = ledger["eligible"].astype(bool)
    ledger["eligible"] = ledger["exploratory_no_harm_eligible"]
    ledger["eligible_for_official_promotion"] = False
    ledger["temporal_independent_n"] = len(panel.sigungu) * len(panel.bundles)
    ledger["directional_unit_n"] = 2 * len(panel.sigungu) * len(panel.bundles)
    ledger["pseudo_replication_n"] = 55
    ledger["exploratory_only_due_no_third_year"] = True
    ledger = ledger.sort_values("candidate_id", kind="mergesort").reset_index(drop=True)
    bundle_ledger = bundle_ledger.sort_values(
        ["candidate_id", "bundle_id"], kind="mergesort"
    ).reset_index(drop=True)
    if len(ledger) != EXPECTED_CANDIDATE_COUNT:
        raise RuntimeError("Candidate ledger row count is not 72")
    audit = {
        **config_audit,
        "candidate_count": len(ledger),
        "bundle_ledger_rows": len(bundle_ledger),
        "observed_years": list(panel.years),
        "policy_sigungu_count": len(panel.sigungu),
        "bundle_count": len(panel.bundles),
        "temporal_independent_unit": "policy_sigungu_x_service_bundle",
        "temporal_independent_n": len(panel.sigungu) * len(panel.bundles),
        "directional_unit_n": 2 * len(panel.sigungu) * len(panel.bundles),
        "admin_dong_delivery_rows_are_independent_samples": False,
        "n_permutations": n_permutations,
        "random_state": int(random_state),
        "n_jobs": n_jobs,
        "vectorized_deterministic_independent_of_n_jobs": True,
        "official_output_writer_available": False,
        "panel_contract_sha256": panel_seal,
        "stage2a_bundle_definition_change_count": 0,
        "exploratory_no_harm_eligible_count": int(
            ledger["exploratory_no_harm_eligible"].sum()
        ),
        "eligible_candidate_count": int(
            ledger["exploratory_no_harm_eligible"].sum()
        ),
        "eligible_for_official_promotion_count": 0,
        "eligible_column_is_exploratory_no_harm_alias": True,
    }
    return CandidateGridResult(ledger, bundle_ledger, audit)


def _common_variance_ratio(
    baseline_services: np.ndarray,
    weights: np.ndarray,
    *,
    retained_months: Sequence[int] | None = None,
) -> np.ndarray:
    """Weighted common-month variance divided by weighted service variance.

    The result has axes prefix x sigungu x year x bundle and is bounded in
    [0, 1] by convexity.  Only values from the candidate's training year enter
    a directional selection decision.
    """

    values = baseline_services
    if retained_months is not None:
        values = values[..., list(retained_months)]
    common = np.einsum("bs,...gysm->...gybm", weights, values, optimize=True)
    numerator = np.var(common, axis=-1, ddof=0)
    service_variance = np.var(values, axis=-1, ddof=0)
    denominator = np.einsum(
        "bs,...gys->...gyb", weights, service_variance, optimize=True
    )
    ratio = numerator / np.where(denominator > 1e-15, denominator, 1.0)
    return np.clip(ratio, 0.0, 1.0)


def _nested_components(
    rates: np.ndarray,
    weights: np.ndarray,
    *,
    removed_month: int | None = None,
    definition: Mapping[str, Sequence[int]] = PRIMARY_SEASON_DEFINITION,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    baseline_services = _representation(rates, "baseline")
    composition_services = _representation(rates, "composition_share")
    baseline_bundle = _bundle_values(baseline_services, weights)
    composition_bundle = _bundle_values(composition_services, weights)
    retained = None
    if removed_month is not None:
        retained = [month for month in range(12) if month != int(removed_month) - 1]
    ratio = _common_variance_ratio(
        baseline_services, weights, retained_months=retained
    )
    baseline_season = _seasonal(
        baseline_bundle,
        "mean",
        removed_month=removed_month,
        definition=definition,
    )
    composition_season = _seasonal(
        composition_bundle,
        "mean",
        removed_month=removed_month,
        definition=definition,
    )
    return ratio, baseline_season, composition_season


def _select(
    ratio: np.ndarray,
    baseline: np.ndarray,
    composition: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    use_baseline = ratio >= threshold
    selected = np.where(use_baseline[..., None], baseline, composition)
    return selected, use_baseline


def _nested_cross_year(
    panel: _Panel,
    *,
    threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    ratio, baseline, composition = _nested_components(panel.rates, panel.weights)
    own_year_profiles, own_year_baseline = _select(
        ratio, baseline, composition, threshold
    )
    selection_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    for train_index, holdout_index in ((0, 1), (1, 0)):
        train_year = panel.years[train_index]
        holdout_year = panel.years[holdout_index]
        train_mask = own_year_baseline[:, train_index]
        predicted = np.where(
            train_mask[..., None],
            baseline[:, train_index],
            composition[:, train_index],
        )
        # The representation is chosen with train data and then applied to the
        # holdout; holdout labels cannot alter the selection.
        holdout = np.where(
            train_mask[..., None],
            baseline[:, holdout_index],
            composition[:, holdout_index],
        )
        metrics = _unit_metrics(predicted, holdout)
        predicted_order = _orders(predicted)
        holdout_order = _orders(holdout)
        for sigungu_index, sigungu in enumerate(panel.sigungu):
            for bundle_index, bundle_id in enumerate(panel.bundles):
                selected_representation = (
                    "baseline"
                    if bool(train_mask[sigungu_index, bundle_index])
                    else "composition_share"
                )
                selection_rows.append(
                    {
                        "candidate_id": ITERATION3_CANDIDATE_ID,
                        "train_year": train_year,
                        "holdout_year": holdout_year,
                        "policy_sigungu_name": sigungu,
                        "bundle_id": bundle_id,
                        "common_monthly_variance_ratio": float(
                            ratio[sigungu_index, train_index, bundle_index]
                        ),
                        "threshold": threshold,
                        "selected_representation": selected_representation,
                        "selection_data": "train_year_only",
                        "holdout_labels_visible_during_selection": False,
                        "threshold_grid_search_used": False,
                    }
                )
                detail_rows.append(
                    {
                        "candidate_id": ITERATION3_CANDIDATE_ID,
                        "train_year": train_year,
                        "holdout_year": holdout_year,
                        "policy_sigungu_name": sigungu,
                        "bundle_id": bundle_id,
                        "selected_representation": selected_representation,
                        "train_primary": SEASON_ORDER[
                            int(predicted_order[sigungu_index, bundle_index, 0])
                        ],
                        "holdout_primary": SEASON_ORDER[
                            int(holdout_order[sigungu_index, bundle_index, 0])
                        ],
                        **{
                            name: float(values[sigungu_index, bundle_index])
                            for name, values in metrics.items()
                        },
                        "temporal_independent_unit": "policy_sigungu_x_service_bundle",
                        "temporal_independent_n": len(panel.sigungu)
                        * len(panel.bundles),
                    }
                )
    selection = pd.DataFrame(selection_rows).sort_values(
        ["train_year", "policy_sigungu_name", "bundle_id"], kind="mergesort"
    ).reset_index(drop=True)
    detail = pd.DataFrame(detail_rows).sort_values(
        ["train_year", "policy_sigungu_name", "bundle_id"], kind="mergesort"
    ).reset_index(drop=True)
    summary = (
        detail.groupby("bundle_id", as_index=False, sort=True)
        .agg(
            top1_agreement=("top1_agreement", "mean"),
            primary_in_holdout_top2=("primary_in_holdout_top2", "mean"),
            selected_pair_holdout_top2_coverage=(
                "selected_pair_holdout_top2_coverage",
                "mean",
            ),
            rank_spearman_median=("rank_spearman", "median"),
            normalized_regret_mean=("normalized_regret", "mean"),
            directional_unit_n=("bundle_id", "size"),
        )
        .reset_index(drop=True)
    )
    summary["direction_count"] = 2
    return selection, detail, summary, own_year_profiles, own_year_baseline


def _nested_lomo(
    panel: _Panel,
    full_profiles: np.ndarray,
    *,
    threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for year_index, year in enumerate(panel.years):
        for removed_month in range(1, 13):
            ratio, baseline, composition = _nested_components(
                panel.rates[:, year_index : year_index + 1],
                panel.weights,
                removed_month=removed_month,
            )
            variant, _ = _select(ratio, baseline, composition, threshold)
            primary, top2 = _retention(
                full_profiles[:, year_index], variant[:, 0]
            )
            for sigungu_index, sigungu in enumerate(panel.sigungu):
                for bundle_index, bundle_id in enumerate(panel.bundles):
                    rows.append(
                        {
                            "year": year,
                            "removed_month": removed_month,
                            "policy_sigungu_name": sigungu,
                            "bundle_id": bundle_id,
                            "primary_retained": int(
                                primary[sigungu_index, bundle_index]
                            ),
                            "top2_retention": float(
                                top2[sigungu_index, bundle_index]
                            ),
                            "selection_recomputed_without_removed_month": True,
                        }
                    )
    detail = pd.DataFrame(rows).sort_values(
        ["year", "removed_month", "policy_sigungu_name", "bundle_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    summary = (
        detail.groupby("bundle_id", as_index=False, sort=True)
        .agg(
            lomo_primary_retention=("primary_retained", "mean"),
            lomo_top2_retention=("top2_retention", "mean"),
            lomo_diagnostic_rows=("bundle_id", "size"),
        )
        .reset_index(drop=True)
    )
    summary["lomo_unique_removals"] = detail[
        ["year", "removed_month"]
    ].drop_duplicates().shape[0]
    return detail, summary


def _nested_loso(
    panel: _Panel,
    full_profiles: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    full_release = full_profiles.mean(axis=(0, 1))
    rows: list[dict[str, Any]] = []
    for removed_index, removed_sigungu in enumerate(panel.sigungu):
        keep = [index for index in range(len(panel.sigungu)) if index != removed_index]
        variant = full_profiles[keep].mean(axis=(0, 1))
        primary, top2 = _retention(full_release, variant)
        full_order = _orders(full_release)
        variant_order = _orders(variant)
        for bundle_index, bundle_id in enumerate(panel.bundles):
            rows.append(
                {
                    "removed_sigungu": removed_sigungu,
                    "bundle_id": bundle_id,
                    "full_primary": SEASON_ORDER[
                        int(full_order[bundle_index, 0])
                    ],
                    "variant_primary": SEASON_ORDER[
                        int(variant_order[bundle_index, 0])
                    ],
                    "primary_retained": int(primary[bundle_index]),
                    "top2_retention": float(top2[bundle_index]),
                    "excluded_block_includes_all_months_years_services": True,
                }
            )
    detail = pd.DataFrame(rows).sort_values(
        ["removed_sigungu", "bundle_id"], kind="mergesort"
    ).reset_index(drop=True)
    summary = (
        detail.groupby("bundle_id", as_index=False, sort=True)
        .agg(
            loso_primary_retention=("primary_retained", "mean"),
            loso_top2_retention=("top2_retention", "mean"),
            loso_unique_removals=("removed_sigungu", "nunique"),
        )
        .reset_index(drop=True)
    )
    return detail, summary


def _nested_bootstrap(
    panel: _Panel,
    profiles: np.ndarray,
    *,
    n_boot: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    full = profiles.mean(axis=(0, 1))
    full_order = _orders(full)
    seeds = _permutation_seeds(n_boot, random_state + 1)
    rows: list[dict[str, Any]] = []
    for replicate, seed in enumerate(seeds, start=1):
        rng = np.random.default_rng(seed)
        sampled = rng.integers(0, len(panel.sigungu), size=len(panel.sigungu))
        variant = profiles[sampled].mean(axis=(0, 1))
        order = _orders(variant)
        primary, top2 = _retention(full, variant)
        for bundle_index, bundle_id in enumerate(panel.bundles):
            rows.append(
                {
                    "bootstrap_replicate": replicate,
                    "bundle_id": bundle_id,
                    "primary_season": SEASON_ORDER[int(order[bundle_index, 0])],
                    "fallback_season": SEASON_ORDER[int(order[bundle_index, 1])],
                    "full_primary": SEASON_ORDER[
                        int(full_order[bundle_index, 0])
                    ],
                    "primary_retained": int(primary[bundle_index]),
                    "top2_retention": float(top2[bundle_index]),
                    "top1_margin_normalized": float(
                        _normalized_margin(variant)[bundle_index]
                    ),
                    "sampled_sigungu_blocks": len(panel.sigungu),
                }
            )
    detail = pd.DataFrame(rows).sort_values(
        ["bootstrap_replicate", "bundle_id"], kind="mergesort"
    ).reset_index(drop=True)
    summary = (
        detail.groupby("bundle_id", as_index=False, sort=True)
        .agg(
            bootstrap_primary_probability=("primary_retained", "mean"),
            bootstrap_top2_retention=("top2_retention", "mean"),
            bootstrap_margin_median=("top1_margin_normalized", "median"),
            n_boot=("bootstrap_replicate", "nunique"),
        )
        .reset_index(drop=True)
    )
    if summary["n_boot"].ne(n_boot).any() or len(detail) != n_boot * len(panel.bundles):
        raise RuntimeError("Nested bootstrap replicate contract failed")
    return detail, summary


def _nested_permutation(
    panel: _Panel,
    observed_profiles: np.ndarray,
    *,
    threshold: float,
    n_permutations: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    observed_margin = _normalized_margin(observed_profiles.mean(axis=(0, 1)))
    original_totals = panel.rates.sum(axis=-1)
    rows: list[dict[str, Any]] = []
    for replicate, seed in enumerate(
        _permutation_seeds(n_permutations, random_state), start=1
    ):
        rng = np.random.default_rng(seed)
        order = np.argsort(
            rng.random(panel.rates.shape), axis=-1, kind="stable"
        )
        permuted = np.take_along_axis(panel.rates, order, axis=-1)
        preserved = bool(
            np.allclose(permuted.sum(axis=-1), original_totals, atol=1e-12)
        )
        ratio, baseline, composition = _nested_components(permuted, panel.weights)
        profiles, _ = _select(ratio, baseline, composition, threshold)
        null_margin = _normalized_margin(profiles.mean(axis=(0, 1)))
        for bundle_index, bundle_id in enumerate(panel.bundles):
            rows.append(
                {
                    "permutation_replicate": replicate,
                    "bundle_id": bundle_id,
                    "top1_margin_normalized": float(null_margin[bundle_index]),
                    "annual_totals_preserved": preserved,
                    "month_labels_shuffled_within_sigungu_year_service": True,
                    "representation_selection_recomputed": True,
                }
            )
    detail = pd.DataFrame(rows).sort_values(
        ["permutation_replicate", "bundle_id"], kind="mergesort"
    ).reset_index(drop=True)
    if (
        len(detail) != n_permutations * len(panel.bundles)
        or not detail["annual_totals_preserved"].all()
    ):
        raise RuntimeError("Nested permutation count/annual-total contract failed")
    summary_rows: list[dict[str, Any]] = []
    for bundle_index, bundle_id in enumerate(panel.bundles):
        null = detail.loc[
            detail["bundle_id"].eq(bundle_id), "top1_margin_normalized"
        ].to_numpy(float)
        observed = float(observed_margin[bundle_index])
        p_value = float(
            (1 + np.count_nonzero(null >= observed)) / (n_permutations + 1)
        )
        summary_rows.append(
            {
                "bundle_id": bundle_id,
                "metric": "top1_margin_normalized",
                "observed": observed,
                "null_mean": float(null.mean()),
                "null_q025": float(np.quantile(null, 0.025)),
                "null_q975": float(np.quantile(null, 0.975)),
                "p_value_upper": p_value,
                "permutation_strength": 1.0 - p_value,
                "n_permutations": n_permutations,
                "n_valid": len(null),
                "annual_totals_preserved": True,
                "null_is_advisory_not_sole_release_gate": True,
            }
        )
    return detail, pd.DataFrame(summary_rows).sort_values("bundle_id").reset_index(drop=True)


def _nested_boundary(
    panel: _Panel,
    primary_profiles: np.ndarray,
    primary_mask: np.ndarray,
) -> pd.DataFrame:
    full = primary_profiles.mean(axis=(0, 1))
    rows: list[dict[str, Any]] = []
    alternatives = {
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
    baseline_services = _representation(panel.rates, "baseline")
    composition_services = _representation(panel.rates, "composition_share")
    baseline_bundle = _bundle_values(baseline_services, panel.weights)
    composition_bundle = _bundle_values(composition_services, panel.weights)
    for definition_name, definition in alternatives.items():
        baseline = _seasonal(baseline_bundle, "mean", definition=definition)
        composition = _seasonal(composition_bundle, "mean", definition=definition)
        variant = np.where(primary_mask[..., None], baseline, composition).mean(
            axis=(0, 1)
        )
        primary, top2 = _retention(full, variant)
        for bundle_index, bundle_id in enumerate(panel.bundles):
            rows.append(
                {
                    "definition": definition_name,
                    "bundle_id": bundle_id,
                    "primary_retained": float(primary[bundle_index]),
                    "top2_retention": float(top2[bundle_index]),
                }
            )
    detail = pd.DataFrame(rows)
    return (
        detail.groupby("bundle_id", as_index=False, sort=True)
        .agg(
            boundary_primary_retention=("primary_retained", "mean"),
            boundary_top2_retention=("top2_retention", "mean"),
            boundary_definition_count=("definition", "nunique"),
        )
        .reset_index(drop=True)
    )


def _nested_release_evidence(
    panel: _Panel,
    cross_summary: pd.DataFrame,
    profiles: np.ndarray,
    lomo_summary: pd.DataFrame,
    loso_summary: pd.DataFrame,
    bootstrap_summary: pd.DataFrame,
    permutation_summary: pd.DataFrame,
    primary_mask: np.ndarray,
) -> pd.DataFrame:
    """Return abstention-friendly evidence without relaxing official rules."""

    year_profiles = profiles.mean(axis=0)  # year x bundle x season
    pooled = year_profiles.mean(axis=0)
    pooled_order = _orders(pooled)
    boundary = _nested_boundary(panel, profiles, primary_mask).set_index("bundle_id")
    cross = cross_summary.set_index("bundle_id")
    lomo = lomo_summary.set_index("bundle_id")
    loso = loso_summary.set_index("bundle_id")
    bootstrap = bootstrap_summary.set_index("bundle_id")
    permutation = permutation_summary.set_index("bundle_id")
    rows: list[dict[str, Any]] = []
    for bundle_index, bundle_id in enumerate(panel.bundles):
        year_orders = _orders(year_profiles[:, bundle_index])
        year_top2 = [set(order[:2]) for order in year_orders]
        jaccard = len(year_top2[0] & year_top2[1]) / len(
            year_top2[0] | year_top2[1]
        )
        rows.append(
            {
                "candidate_id": ITERATION3_CANDIDATE_ID,
                "bundle_id": bundle_id,
                "year2022_top1": SEASON_ORDER[int(year_orders[0, 0])],
                "year2023_top1": SEASON_ORDER[int(year_orders[1, 0])],
                "cross_year_top1_agreement": float(
                    year_orders[0, 0] == year_orders[1, 0]
                ),
                "year_top2_jaccard": float(jaccard),
                "consensus_primary_candidate": SEASON_ORDER[
                    int(pooled_order[bundle_index, 0])
                ],
                "consensus_fallback_candidate": SEASON_ORDER[
                    int(pooled_order[bundle_index, 1])
                ],
                "pooled_normalized_top1_margin": float(
                    _normalized_margin(pooled)[bundle_index]
                ),
                "bidirectional_primary_in_holdout_top2": float(
                    cross.loc[bundle_id, "primary_in_holdout_top2"]
                ),
                "bidirectional_selected_pair_holdout_top2_coverage": float(
                    cross.loc[
                        bundle_id, "selected_pair_holdout_top2_coverage"
                    ]
                ),
                "heldout_normalized_regret": float(
                    cross.loc[bundle_id, "normalized_regret_mean"]
                ),
                "lomo_primary_retention": float(
                    lomo.loc[bundle_id, "lomo_primary_retention"]
                ),
                "lomo_top2_retention": float(
                    lomo.loc[bundle_id, "lomo_top2_retention"]
                ),
                "loso_primary_retention": float(
                    loso.loc[bundle_id, "loso_primary_retention"]
                ),
                "loso_top2_retention": float(
                    loso.loc[bundle_id, "loso_top2_retention"]
                ),
                "bootstrap_primary_probability": float(
                    bootstrap.loc[bundle_id, "bootstrap_primary_probability"]
                ),
                "bootstrap_top2_retention": float(
                    bootstrap.loc[bundle_id, "bootstrap_top2_retention"]
                ),
                "boundary_primary_retention": float(
                    boundary.loc[bundle_id, "boundary_primary_retention"]
                ),
                "boundary_top2_retention": float(
                    boundary.loc[bundle_id, "boundary_top2_retention"]
                ),
                "permutation_margin_percentile": float(
                    permutation.loc[bundle_id, "permutation_strength"]
                ),
                "release_type": "NO_STRONG_PREFERENCE",
                "confidence": "LOW",
                "final_primary": pd.NA,
                "final_fallback": pd.NA,
                "exact_month": pd.NA,
                "exact_date": pd.NA,
                "original_release_thresholds_relaxed": False,
                "official_release_unchanged": True,
                "exploratory_only_due_no_third_year": True,
                "reason": (
                    "candidate remains exploratory; no third same-definition year; "
                    "official Stage2B abstention is unchanged"
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("bundle_id").reset_index(drop=True)


def _nested_comparison(
    grid: CandidateGridResult,
    cross_detail: pd.DataFrame,
    cross_summary: pd.DataFrame,
    lomo_summary: pd.DataFrame,
    loso_summary: pd.DataFrame,
    permutation_summary: pd.DataFrame,
) -> pd.DataFrame:
    baseline = grid.ledger.loc[
        grid.ledger["candidate_id"].eq(BASELINE_CANDIDATE_ID)
    ].iloc[0]
    metrics = {
        "top1_agreement": float(cross_detail["top1_agreement"].mean()),
        "primary_in_holdout_top2": float(
            cross_detail["primary_in_holdout_top2"].mean()
        ),
        "selected_pair_holdout_top2_coverage": float(
            cross_detail["selected_pair_holdout_top2_coverage"].mean()
        ),
        "rank_spearman_median": float(
            cross_detail["rank_spearman"].median()
        ),
        "normalized_regret_mean": float(
            cross_detail["normalized_regret"].mean()
        ),
        "lomo_primary_retention": float(
            np.average(
                lomo_summary["lomo_primary_retention"],
                weights=lomo_summary["lomo_diagnostic_rows"],
            )
        ),
        "lomo_top2_retention": float(
            np.average(
                lomo_summary["lomo_top2_retention"],
                weights=lomo_summary["lomo_diagnostic_rows"],
            )
        ),
        "loso_primary_retention": float(
            loso_summary["loso_primary_retention"].mean()
        ),
        "loso_top2_retention": float(loso_summary["loso_top2_retention"].mean()),
        "permutation_strength_mean": float(
            permutation_summary["permutation_strength"].mean()
        ),
    }
    failure_checks = {
        "top1_not_strictly_better": metrics["top1_agreement"]
        <= baseline["top1_agreement"] + _TOLERANCE,
        "top2_not_strictly_better": metrics["primary_in_holdout_top2"]
        <= baseline["primary_in_holdout_top2"] + _TOLERANCE,
        "pair_not_strictly_better": metrics[
            "selected_pair_holdout_top2_coverage"
        ]
        <= baseline["selected_pair_holdout_top2_coverage"] + _TOLERANCE,
        "rank_spearman_worse": metrics["rank_spearman_median"]
        < baseline["rank_spearman_median"] - _TOLERANCE,
        "regret_worse": metrics["normalized_regret_mean"]
        > baseline["normalized_regret_mean"] + _TOLERANCE,
        "lomo_primary_worse": metrics["lomo_primary_retention"]
        < baseline["lomo_primary_retention"] - _TOLERANCE,
        "lomo_top2_worse": metrics["lomo_top2_retention"]
        < baseline["lomo_top2_retention"] - _TOLERANCE,
        "loso_primary_worse": metrics["loso_primary_retention"]
        < baseline["loso_primary_retention"] - _TOLERANCE,
        "loso_top2_worse": metrics["loso_top2_retention"]
        < baseline["loso_top2_retention"] - _TOLERANCE,
        "permutation_strength_worse": metrics["permutation_strength_mean"]
        < baseline["permutation_strength_mean"] - _TOLERANCE,
    }
    baseline_bundle = grid.bundle_ledger.loc[
        grid.bundle_ledger["candidate_id"].eq(BASELINE_CANDIDATE_ID)
    ].set_index("bundle_id")
    candidate_bundle = cross_summary.set_index("bundle_id")
    harmed: list[str] = []
    for bundle_id, row in candidate_bundle.iterrows():
        reference = baseline_bundle.loc[bundle_id]
        if (
            row["primary_in_holdout_top2"]
            < reference["primary_in_holdout_top2"] - 0.05 - _TOLERANCE
            or row["normalized_regret_mean"]
            > reference["normalized_regret_mean"] + 0.05 + _TOLERANCE
        ):
            harmed.append(str(bundle_id))
    global_pass = not any(failure_checks.values())
    bundle_pass = not harmed
    no_harm_pass = global_pass and bundle_pass
    failures = [name for name, failed in failure_checks.items() if failed]
    if harmed:
        failures.append("bundle_harm_caps:" + "|".join(harmed))
    return pd.DataFrame(
        [
            {
                "candidate_id": ITERATION3_CANDIDATE_ID,
                **metrics,
                "global_no_harm_pass": global_pass,
                "bundle_no_harm_pass": bundle_pass,
                "no_harm_pass": no_harm_pass,
                "failure_reasons": ";".join(failures),
                "stage2a_bundle_definition_change_count": 0,
                "stage2a_what_top1_retention": 1.0,
                "original_release_rules_relaxed": False,
                "eligible_for_official_promotion": False,
                "exploratory_only_due_no_third_year": True,
                "decision": (
                    EXPLORATORY_DECISION if no_harm_pass else NO_PROMOTION_DECISION
                ),
            }
        ]
    )


def evaluate_nested_candidate(
    service_month: pd.DataFrame,
    bundle_config: Mapping[str, Any],
    grid: CandidateGridResult,
    *,
    n_boot: int,
    n_permutations: int,
    years: Sequence[int] = OBSERVED_YEARS,
    expected_policy_sigungu: int = 11,
    random_state: int = 20260818,
    n_jobs: int = 1,
    threshold: float = ITERATION3_THRESHOLD,
    iteration_config: Mapping[str, Any] | None = None,
) -> NestedCandidateResult:
    """Evaluate the single train-year-only common-variance gated candidate."""

    n_boot = _positive_integer(n_boot, "n_boot")
    n_permutations = _positive_integer(n_permutations, "n_permutations")
    n_jobs = _positive_integer(n_jobs, "n_jobs")
    if not np.isclose(float(threshold), ITERATION3_THRESHOLD, atol=0):
        raise ValueError("Iteration-3 threshold is locked at 0.50; tuning is forbidden")
    if iteration_config is not None:
        validate_iteration2_config(iteration_config)
    panel = _panel(
        service_month,
        bundle_config,
        years=years,
        expected_policy_sigungu=expected_policy_sigungu,
    )
    panel_seal = _panel_contract_sha256(panel)
    grid_panel_seal = grid.audit.get("panel_contract_sha256")
    if type(grid_panel_seal) is not str or grid_panel_seal != panel_seal:
        raise ValueError(
            "Nested candidate panel SHA256 differs from the candidate-grid panel"
        )
    selection, cross_detail, cross_summary, profiles, primary_mask = _nested_cross_year(
        panel, threshold=threshold
    )
    lomo_detail, lomo_summary = _nested_lomo(
        panel, profiles, threshold=threshold
    )
    loso_detail, loso_summary = _nested_loso(panel, profiles)
    bootstrap_detail, bootstrap_summary = _nested_bootstrap(
        panel, profiles, n_boot=n_boot, random_state=random_state
    )
    permutation_detail, permutation_summary = _nested_permutation(
        panel,
        profiles,
        threshold=threshold,
        n_permutations=n_permutations,
        random_state=random_state,
    )
    release = _nested_release_evidence(
        panel,
        cross_summary,
        profiles,
        lomo_summary,
        loso_summary,
        bootstrap_summary,
        permutation_summary,
        primary_mask,
    )
    comparison = _nested_comparison(
        grid,
        cross_detail,
        cross_summary,
        lomo_summary,
        loso_summary,
        permutation_summary,
    )
    audit = {
        "candidate_id": ITERATION3_CANDIDATE_ID,
        "threshold": threshold,
        "threshold_grid_search_used": False,
        "selection_data": "train_year_only",
        "selection_rows": len(selection),
        "cross_year_direction_count": 2,
        "cross_year_detail_rows": len(cross_detail),
        "temporal_independent_unit": "policy_sigungu_x_service_bundle",
        "temporal_independent_n": len(panel.sigungu) * len(panel.bundles),
        "production_temporal_independent_n": 55,
        "admin_dong_delivery_rows_are_independent_samples": False,
        "lomo_unique_removals": 24,
        "lomo_detail_rows": len(lomo_detail),
        "loso_unique_removals": len(panel.sigungu),
        "loso_detail_rows": len(loso_detail),
        "n_boot": n_boot,
        "n_permutations": n_permutations,
        "random_state": int(random_state),
        "n_jobs": n_jobs,
        "deterministic_independent_of_n_jobs": True,
        "stage2a_bundle_definition_change_count": 0,
        "official_output_writer_available": False,
        "panel_contract_sha256": panel_seal,
        "official_release_unchanged": True,
        "exploratory_only_due_no_third_year": True,
        "promotion_decision": str(comparison.iloc[0]["decision"]),
    }
    return NestedCandidateResult(
        selection,
        cross_detail,
        cross_summary,
        lomo_detail,
        lomo_summary,
        loso_detail,
        loso_summary,
        bootstrap_detail,
        bootstrap_summary,
        permutation_detail,
        permutation_summary,
        release,
        comparison,
        audit,
    )


def run_hardening_candidates(
    service_month: pd.DataFrame,
    bundle_config: Mapping[str, Any],
    *,
    n_boot: int,
    n_permutations: int,
    years: Sequence[int] = OBSERVED_YEARS,
    expected_policy_sigungu: int = 11,
    random_state: int = 20260818,
    n_jobs: int = 1,
    permutation_batch_size: int = 64,
    iteration_config: Mapping[str, Any] | None = None,
) -> HardeningCandidatesResult:
    """Run both locked iterations in memory; no writer is intentionally exposed."""

    grid = evaluate_candidate_grid(
        service_month,
        bundle_config,
        n_permutations=n_permutations,
        years=years,
        expected_policy_sigungu=expected_policy_sigungu,
        random_state=random_state,
        n_jobs=n_jobs,
        permutation_batch_size=permutation_batch_size,
        iteration_config=iteration_config,
    )
    nested = evaluate_nested_candidate(
        service_month,
        bundle_config,
        grid,
        n_boot=n_boot,
        n_permutations=n_permutations,
        years=years,
        expected_policy_sigungu=expected_policy_sigungu,
        random_state=random_state,
        n_jobs=n_jobs,
        iteration_config=iteration_config,
    )
    decision = str(nested.comparison.iloc[0]["decision"])
    if grid.audit["panel_contract_sha256"] != nested.audit["panel_contract_sha256"]:
        raise RuntimeError("Grid and nested candidate panels have different seals")
    audit = {
        "candidate_count": int(grid.audit["candidate_count"]),
        "iteration3_candidate_count": 1,
        "official_output_writer_available": False,
        "official_hardening_tree_mutated": False,
        "combined_panel_contract_sha256": grid.audit["panel_contract_sha256"],
        "grid_nested_panel_seal_match": True,
        "stage2a_bundle_definition_change_count": 0,
        "stage3_started": False,
        "promotion_decision": decision,
    }
    return HardeningCandidatesResult(grid, nested, decision, audit)


__all__ = [
    "BASELINE_CANDIDATE_ID",
    "CandidateGridResult",
    "EB_CENTERS",
    "EB_NOISE_MULTIPLIERS",
    "EXPECTED_CANDIDATE_COUNT",
    "EXPECTED_ITERATION2_CONTRACT_MAPPING_SHA256",
    "EXPLORATORY_DECISION",
    "HardeningCandidatesResult",
    "ITERATION3_CANDIDATE_ID",
    "ITERATION3_THRESHOLD",
    "NO_PROMOTION_DECISION",
    "NestedCandidateResult",
    "REPRESENTATIONS",
    "SEASONAL_AGGREGATORS",
    "canonical_contract_mapping_sha256",
    "evaluate_candidate_grid",
    "evaluate_nested_candidate",
    "run_hardening_candidates",
    "validate_iteration2_config",
]
