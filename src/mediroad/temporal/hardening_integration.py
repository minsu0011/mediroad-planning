"""Stage 2A x 2B integration and fail-closed Stage 3 interface contracts.

This module deliberately contains no Stage 3 modelling.  It checks that the
Stage 2B coarse scheduling prior is attached *after* the Stage 2A service
bundle decision (WHAT -> WHEN), quantifies the distortion produced by
experimental joint alternatives, and builds the read-only hand-off table that
a later Stage 3 run may consume.

The statistical unit of the temporal evidence is policy-sigungu x bundle
(currently 11 x 5 = 55).  The 153 x 5 delivery table is never treated as 765
independent temporal observations.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


VALID_SEASONS = ("spring", "summer", "autumn", "winter")
VALID_RELEASE_TYPES = (
    "STRONG_SINGLE",
    "ROBUST_PAIR",
    "NO_STRONG_PREFERENCE",
)
VALID_TEMPORAL_CONFIDENCE = ("HIGH", "MODERATE", "LOW")
STAGE2_TO_STAGE3_COLUMNS = (
    "admin_dong_code",
    "stage1_need_score",
    "bundle_id",
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
)


@dataclass(frozen=True)
class IntegrationComparisonResult:
    """Experimental WHAT/WHEN model comparison."""

    rank_panel: pd.DataFrame
    summary: pd.DataFrame


@dataclass(frozen=True)
class SpatialLeakageResult:
    """Temporal profile multiplicity and leakage audit."""

    bundle_summary: pd.DataFrame
    summary: pd.DataFrame


@dataclass(frozen=True)
class FrozenIntegritySnapshot:
    """Named immutable artifact and canonical-frame fingerprints."""

    values: dict[str, str]


@dataclass(frozen=True)
class BundleDifferentiationResult:
    """Bundle diversity and within/between-sigungu diagnostics."""

    admin_summary: pd.DataFrame
    top_frequency: pd.DataFrame
    sigungu_summary: pd.DataFrame
    variance_decomposition: pd.DataFrame
    summary: pd.DataFrame


@dataclass(frozen=True)
class NeedReplicationResult:
    """Stage 1 Need overlap and Need-controlled bundle differentiation."""

    bundle_summary: pd.DataFrame
    partial_correlations: pd.DataFrame
    residual_panel: pd.DataFrame


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise KeyError(f"{label} lacks required columns: {missing}")


def _validate_unique(frame: pd.DataFrame, keys: Sequence[str], label: str) -> None:
    _require_columns(frame, keys, label)
    if frame.duplicated(list(keys)).any():
        raise ValueError(f"{label} contains duplicate keys: {list(keys)}")


def _clean_code(series: pd.Series) -> pd.Series:
    return series.astype("string").str.replace(r"\.0$", "", regex=True)


def _finite_numeric(series: pd.Series, label: str) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError(f"{label} contains missing or non-finite values")
    return values


def _safe_correlation(left: Sequence[float], right: Sequence[float]) -> float:
    x = np.asarray(left, dtype=float)
    y = np.asarray(right, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(left: Sequence[float], right: Sequence[float]) -> float:
    x = pd.Series(left).rank(method="average").to_numpy(float)
    y = pd.Series(right).rank(method="average").to_numpy(float)
    return _safe_correlation(x, y)


def _deterministic_rank(
    frame: pd.DataFrame,
    *,
    group: str,
    item: str,
    score: str,
    output: str,
) -> pd.DataFrame:
    ranked = frame.copy()
    ranked[output] = 0
    for _, indices in ranked.groupby(group, sort=True).groups.items():
        ordered = ranked.loc[list(indices)].sort_values(
            [score, item], ascending=[False, True], kind="mergesort"
        )
        ranked.loc[ordered.index, output] = np.arange(1, len(ordered) + 1)
    ranked[output] = ranked[output].astype(int)
    return ranked


def file_sha256(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return the byte-level SHA256 of an existing file."""

    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_scalar(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        if value == 0:
            value = 0.0
        return format(value, ".17g")
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    return str(value)


def dataframe_fingerprint(
    frame: pd.DataFrame,
    *,
    key_columns: Sequence[str],
    columns: Sequence[str] | None = None,
) -> str:
    """Hash a canonical, row-order-independent dataframe representation.

    File SHA protects the frozen byte artifact while this fingerprint protects
    the semantically important keys and values after deserialisation.
    """

    _validate_unique(frame, key_columns, "Fingerprint frame")
    selected = list(columns) if columns is not None else list(frame.columns)
    _require_columns(frame, [*key_columns, *selected], "Fingerprint frame")
    selected = list(dict.fromkeys([*key_columns, *selected]))
    data = frame[selected].sort_values(list(key_columns), kind="mergesort").reset_index(drop=True)
    payload = {
        "columns": selected,
        "dtypes": [str(data[column].dtype) for column in selected],
        "data": [
            [_canonical_scalar(value) for value in row]
            for row in data.itertuples(index=False, name=None)
        ],
    }
    encoded = json.dumps(
        payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def capture_frozen_integrity(
    stage1_scores: pd.DataFrame,
    stage2a_bundle_scores: pd.DataFrame,
    artifact_paths: Mapping[str, str | Path] | None = None,
) -> FrozenIntegritySnapshot:
    """Capture Stage 1/2A semantic fingerprints plus supplied file hashes."""

    stage1_score = "stage1_need_score" if "stage1_need_score" in stage1_scores else "need_score"
    _require_columns(stage1_scores, ["admin_dong_code", stage1_score], "Stage 1 scores")
    _require_columns(
        stage2a_bundle_scores,
        ["admin_dong_code", "bundle_id", "bundle_gap_score"],
        "Stage 2A bundle scores",
    )
    values = {
        "stage1_need_score_frame": dataframe_fingerprint(
            stage1_scores,
            key_columns=["admin_dong_code"],
            columns=[stage1_score],
        ),
        "stage2a_bundle_score_frame": dataframe_fingerprint(
            stage2a_bundle_scores,
            key_columns=["admin_dong_code", "bundle_id"],
        ),
    }
    for name, path in sorted((artifact_paths or {}).items()):
        if name in values:
            raise ValueError(f"Duplicate frozen integrity name: {name}")
        values[str(name)] = file_sha256(path)
    return FrozenIntegritySnapshot(values)


def compare_frozen_integrity(
    before: FrozenIntegritySnapshot | Mapping[str, str],
    after: FrozenIntegritySnapshot | Mapping[str, str],
    *,
    require_same_names: bool = True,
) -> pd.DataFrame:
    """Compare pre/post snapshots without mutating either artifact set."""

    left = before.values if isinstance(before, FrozenIntegritySnapshot) else dict(before)
    right = after.values if isinstance(after, FrozenIntegritySnapshot) else dict(after)
    names = sorted(set(left) | set(right))
    if require_same_names and set(left) != set(right):
        missing_before = sorted(set(right) - set(left))
        missing_after = sorted(set(left) - set(right))
        raise ValueError(
            "Frozen snapshot names differ; "
            f"missing_before={missing_before}, missing_after={missing_after}"
        )
    return pd.DataFrame(
        [
            {
                "artifact": name,
                "before_sha256": left.get(name),
                "after_sha256": right.get(name),
                "unchanged": bool(left.get(name) is not None and left.get(name) == right.get(name)),
                "severity": "hard",
            }
            for name in names
        ]
    )


def validate_expected_frozen_hashes(
    observed: FrozenIntegritySnapshot | Mapping[str, str],
    expected: Mapping[str, str],
) -> pd.DataFrame:
    """Compare an observed snapshot with preregistered SHA/fingerprint values."""

    values = observed.values if isinstance(observed, FrozenIntegritySnapshot) else dict(observed)
    rows = []
    for name, expected_hash in sorted(expected.items()):
        actual = values.get(name)
        rows.append(
            {
                "artifact": name,
                "observed_sha256": actual,
                "expected_sha256": str(expected_hash),
                "matches": bool(actual == str(expected_hash)),
                "severity": "hard",
            }
        )
    return pd.DataFrame(rows)


def compare_what_when_models(
    bundle_scores: pd.DataFrame,
    season_scores: pd.DataFrame,
    *,
    lambdas: Sequence[float] = (0.10, 0.20, 0.30),
) -> IntegrationComparisonResult:
    """Compare hierarchical, joint, and bounded temporal adjustment models.

    The released hierarchy ranks bundles only by ``bundle_gap_score``.  Joint
    and bounded scores use the best season for each bundle solely to quantify
    specialty intrusion; they are diagnostic and are not release selectors.
    ``temporal_centered_fit`` is fixed to (fit - 50) / 50 and clipped to [-1,1],
    so lambda is an auditable upper bound on proportional adjustment.
    """

    bundle_keys = ["admin_dong_code", "bundle_id"]
    season_keys = [*bundle_keys, "season"]
    _require_columns(bundle_scores, [*bundle_keys, "bundle_gap_score"], "Bundle scores")
    _require_columns(
        season_scores,
        [*season_keys, "temporal_fit_score"],
        "Season scores",
    )
    bundles = bundle_scores[[*bundle_keys, "bundle_gap_score"]].copy()
    seasons = season_scores[[*season_keys, "temporal_fit_score"]].copy()
    bundles["admin_dong_code"] = _clean_code(bundles["admin_dong_code"])
    seasons["admin_dong_code"] = _clean_code(seasons["admin_dong_code"])
    bundles["bundle_id"] = bundles["bundle_id"].astype(str)
    seasons["bundle_id"] = seasons["bundle_id"].astype(str)
    _validate_unique(bundles, bundle_keys, "Bundle scores")
    _validate_unique(seasons, season_keys, "Season scores")
    bundles["bundle_gap_score"] = _finite_numeric(
        bundles["bundle_gap_score"], "bundle_gap_score"
    )
    seasons["temporal_fit_score"] = _finite_numeric(
        seasons["temporal_fit_score"], "temporal_fit_score"
    )
    if not seasons["temporal_fit_score"].between(0, 100).all():
        raise ValueError("temporal_fit_score must be bounded to [0, 100]")
    if (np.asarray(lambdas, dtype=float) <= 0).any() or (
        np.asarray(lambdas, dtype=float) >= 1
    ).any():
        raise ValueError("Every bounded lambda must lie strictly between 0 and 1")

    bundle_sets = bundles.groupby("admin_dong_code")["bundle_id"].agg(lambda x: tuple(sorted(x)))
    if bundle_sets.nunique() != 1:
        raise ValueError("Every admin_dong must have the same Stage 2A bundle set")
    season_bundle_keys = set(map(tuple, seasons[bundle_keys].drop_duplicates().to_numpy()))
    bundle_key_set = set(map(tuple, bundles[bundle_keys].to_numpy()))
    if season_bundle_keys != bundle_key_set:
        raise ValueError("Season scores and Stage 2A bundle keys do not match")
    season_counts = seasons.groupby(bundle_keys)["season"].nunique()
    if not season_counts.eq(len(VALID_SEASONS)).all():
        raise ValueError("Every admin_dong x bundle requires four distinct seasons")

    order_map = {season: index for index, season in enumerate(VALID_SEASONS)}
    seasons["__season_order"] = seasons["season"].map(order_map).fillna(len(order_map)).astype(int)
    ordered = seasons.sort_values(
        [*bundle_keys, "temporal_fit_score", "__season_order"],
        ascending=[True, True, False, True],
        kind="mergesort",
    )
    best = ordered.drop_duplicates(bundle_keys, keep="first").rename(
        columns={
            "season": "temporal_top_season",
            "temporal_fit_score": "temporal_top_score",
        }
    )
    base = bundles.merge(
        best[[*bundle_keys, "temporal_top_season", "temporal_top_score"]],
        on=bundle_keys,
        how="left",
        validate="one_to_one",
    )
    base["temporal_centered_fit"] = (
        (base["temporal_top_score"] - 50.0) / 50.0
    ).clip(-1.0, 1.0)
    base = _deterministic_rank(
        base,
        group="admin_dong_code",
        item="bundle_id",
        score="bundle_gap_score",
        output="stage2a_bundle_rank",
    )

    scenarios: list[tuple[str, float | None, np.ndarray]] = [
        ("hierarchical_WHAT_then_WHEN", None, base["bundle_gap_score"].to_numpy(float)),
        (
            "joint_gap_times_temporal_fit",
            None,
            base["bundle_gap_score"].to_numpy(float)
            * base["temporal_top_score"].to_numpy(float)
            / 100.0,
        ),
    ]
    for value in lambdas:
        lam = float(value)
        scenarios.append(
            (
                f"bounded_lambda_{lam:.2f}",
                lam,
                base["bundle_gap_score"].to_numpy(float)
                * (1.0 + lam * base["temporal_centered_fit"].to_numpy(float)),
            )
        )

    panels: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    baseline_top1: dict[str, str] = {}
    baseline_top2: dict[str, set[str]] = {}
    for admin, group in base.groupby("admin_dong_code", sort=True):
        ranked_group = group.sort_values("stage2a_bundle_rank")
        baseline_top1[str(admin)] = str(ranked_group.iloc[0]["bundle_id"])
        baseline_top2[str(admin)] = set(ranked_group.iloc[:2]["bundle_id"].astype(str))

    for model, lam, scores in scenarios:
        panel = base.copy()
        panel["model"] = model
        panel["lambda"] = lam
        panel["integrated_bundle_score"] = scores
        panel = _deterministic_rank(
            panel,
            group="admin_dong_code",
            item="bundle_id",
            score="integrated_bundle_score",
            output="integrated_bundle_rank",
        )
        panel["rank_shift"] = panel["integrated_bundle_rank"] - panel["stage2a_bundle_rank"]
        denominator = panel.groupby("admin_dong_code")["bundle_id"].transform("size") - 1
        panel["normalized_abs_rank_distortion"] = (
            panel["rank_shift"].abs() / denominator.replace(0, np.nan)
        ).fillna(0.0)
        base_score = panel["bundle_gap_score"].abs().replace(0, np.nan)
        panel["abs_relative_score_distortion"] = (
            (panel["integrated_bundle_score"] - panel["bundle_gap_score"]).abs()
            / base_score
        ).fillna(0.0)

        top1_match: list[float] = []
        top2_recall: list[float] = []
        top2_exact: list[float] = []
        correlations: list[float] = []
        for admin, group in panel.groupby("admin_dong_code", sort=True):
            scenario_order = group.sort_values("integrated_bundle_rank")
            scenario_top1 = str(scenario_order.iloc[0]["bundle_id"])
            scenario_top2 = set(scenario_order.iloc[:2]["bundle_id"].astype(str))
            top1_match.append(float(scenario_top1 == baseline_top1[str(admin)]))
            top2_recall.append(len(scenario_top2 & baseline_top2[str(admin)]) / 2.0)
            top2_exact.append(float(scenario_top2 == baseline_top2[str(admin)]))
            by_bundle = group.sort_values("bundle_id")
            correlations.append(
                _spearman(
                    by_bundle["stage2a_bundle_rank"],
                    by_bundle["integrated_bundle_rank"],
                )
            )
        invariant = bool(
            panel["integrated_bundle_rank"].eq(panel["stage2a_bundle_rank"]).all()
            and np.allclose(
                panel["integrated_bundle_score"], panel["bundle_gap_score"], atol=0, rtol=0
            )
        )
        summaries.append(
            {
                "model": model,
                "lambda": lam,
                "admin_count": int(panel["admin_dong_code"].nunique()),
                "bundle_count": int(panel["bundle_id"].nunique()),
                "top1_bundle_retention": float(np.mean(top1_match)),
                "top2_bundle_recall": float(np.mean(top2_recall)),
                "top2_exact_set_retention": float(np.mean(top2_exact)),
                "mean_bundle_rank_spearman": float(np.mean(correlations)),
                "median_bundle_rank_spearman": float(np.median(correlations)),
                "minimum_bundle_rank_spearman": float(np.min(correlations)),
                "mean_normalized_rank_distortion": float(
                    panel["normalized_abs_rank_distortion"].mean()
                ),
                "maximum_normalized_rank_distortion": float(
                    panel["normalized_abs_rank_distortion"].max()
                ),
                "mean_abs_relative_score_distortion": float(
                    panel["abs_relative_score_distortion"].mean()
                ),
                "maximum_abs_relative_score_distortion": float(
                    panel["abs_relative_score_distortion"].max()
                ),
                "bundle_change_rate": float(1.0 - np.mean(top1_match)),
                "hierarchical_what_invariant": invariant,
                "release_eligible": bool(model == "hierarchical_WHAT_then_WHEN" and invariant),
            }
        )
        panels.append(panel)

    summary = pd.DataFrame(summaries)
    hierarchy = summary.loc[summary["model"].eq("hierarchical_WHAT_then_WHEN")].iloc[0]
    if not bool(hierarchy["hierarchical_what_invariant"]):
        raise AssertionError("Hierarchical WHAT -> WHEN unexpectedly changed Stage 2A bundle ranks")
    return IntegrationComparisonResult(
        pd.concat(panels, ignore_index=True),
        summary,
    )


def diagnose_temporal_spatial_leakage(
    season_scores: pd.DataFrame,
    recommendations: pd.DataFrame,
) -> SpatialLeakageResult:
    """Flag unsupported admin-level temporal differentiation.

    A region-specific temporal profile is permitted only when the input
    explicitly says that admin-level temporal evidence was used.  Climate and
    static transit advisory columns do not grant that permission.
    """

    season_keys = ["admin_dong_code", "bundle_id", "season"]
    recommendation_keys = ["admin_dong_code", "bundle_id"]
    _require_columns(
        season_scores,
        [*season_keys, "temporal_fit_score", "policy_sigungu_name"],
        "Season scores",
    )
    _require_columns(
        recommendations,
        [*recommendation_keys, "primary_season", "fallback_season", "policy_sigungu_name"],
        "Temporal recommendations",
    )
    scores = season_scores.copy()
    recs = recommendations.copy()
    for frame in (scores, recs):
        frame["admin_dong_code"] = _clean_code(frame["admin_dong_code"])
        frame["bundle_id"] = frame["bundle_id"].astype(str)
    _validate_unique(scores, season_keys, "Season scores")
    _validate_unique(recs, recommendation_keys, "Temporal recommendations")

    evidence_flag_column = "region_specific_clinical_seasonality_used"
    if evidence_flag_column in recs:
        evidence_used = pd.to_numeric(
            recs[evidence_flag_column], errors="coerce"
        ).fillna(0).astype(int).ne(0)
    else:
        evidence_used = pd.Series(False, index=recs.index)
    recs["__admin_temporal_evidence"] = evidence_used

    score_order = {season: index for index, season in enumerate(VALID_SEASONS)}
    scores["__order"] = scores["season"].map(score_order).fillna(99).astype(int)
    signature_rows = []
    for (admin, bundle), group in scores.groupby(recommendation_keys, sort=True):
        ordered = group.sort_values(["__order", "season"], kind="mergesort")
        signature_rows.append(
            {
                "admin_dong_code": str(admin),
                "bundle_id": str(bundle),
                "temporal_profile_signature": "|".join(
                    f"{season}:{float(score):.12g}"
                    for season, score in zip(
                        ordered["season"], ordered["temporal_fit_score"], strict=True
                    )
                ),
            }
        )
    signatures = pd.DataFrame(signature_rows)
    recs = recs.merge(signatures, on=recommendation_keys, how="left", validate="one_to_one")
    if recs["temporal_profile_signature"].isna().any():
        raise ValueError("Recommendation and season-score keys do not match")
    recs["released_pair_signature"] = (
        recs["primary_season"].astype("string").fillna("<ABSTAIN>")
        + "|"
        + recs["fallback_season"].astype("string").fillna("<ABSTAIN>")
    )

    rows: list[dict[str, Any]] = []
    for bundle, group in recs.groupby("bundle_id", sort=True):
        max_profile_within_sigungu = int(
            group.groupby("policy_sigungu_name")["temporal_profile_signature"].nunique().max()
        )
        max_release_within_sigungu = int(
            group.groupby("policy_sigungu_name")["released_pair_signature"].nunique().max()
        )
        admin_evidence = bool(group["__admin_temporal_evidence"].any())
        profile_count = int(group["temporal_profile_signature"].nunique())
        release_count = int(group["released_pair_signature"].nunique())
        unsupported = bool(
            not admin_evidence
            and (profile_count > 1 or release_count > 1 or max_profile_within_sigungu > 1)
        )
        rows.append(
            {
                "bundle_id": bundle,
                "admin_count": int(group["admin_dong_code"].nunique()),
                "policy_sigungu_count": int(group["policy_sigungu_name"].nunique()),
                "unique_temporal_profiles": profile_count,
                "unique_released_pairs": release_count,
                "max_profiles_within_sigungu": max_profile_within_sigungu,
                "max_released_pairs_within_sigungu": max_release_within_sigungu,
                "admin_level_temporal_evidence_used": admin_evidence,
                "unsupported_spatial_variation": unsupported,
            }
        )
    bundle_summary = pd.DataFrame(rows)
    sigungu_count = int(recs["policy_sigungu_name"].nunique())
    bundle_count = int(recs["bundle_id"].nunique())
    independent_n = sigungu_count * bundle_count
    summary = pd.DataFrame(
        [
            {
                "delivery_rows": int(len(recs)),
                "admin_count": int(recs["admin_dong_code"].nunique()),
                "bundle_count": bundle_count,
                "policy_sigungu_count": sigungu_count,
                "independent_temporal_n": independent_n,
                "broadcast_rows_claimed_as_independent": bool(len(recs) == independent_n),
                "unexpected_spatial_variation_count": int(
                    bundle_summary["unsupported_spatial_variation"].sum()
                ),
                "leakage_flag": bool(bundle_summary["unsupported_spatial_variation"].any()),
                "temporal_spatial_resolution": (
                    "province_common_bundle_season_with_sigungu_block_validation"
                ),
            }
        ]
    )
    return SpatialLeakageResult(bundle_summary, summary)


def _normalised_entropy(values: Sequence[float]) -> float:
    probabilities = np.asarray(values, dtype=float)
    probabilities = probabilities[np.isfinite(probabilities) & (probabilities > 0)]
    if len(probabilities) <= 1:
        return 0.0
    probabilities = probabilities / probabilities.sum()
    return float(-(probabilities * np.log(probabilities)).sum() / math.log(len(probabilities)))


def analyze_bundle_differentiation(
    bundle_scores: pd.DataFrame,
) -> BundleDifferentiationResult:
    """Describe service-bundle separation without inventing a pass threshold."""

    required = [
        "admin_dong_code",
        "policy_sigungu_name",
        "bundle_id",
        "bundle_gap_score",
    ]
    _require_columns(bundle_scores, required, "Bundle scores")
    frame = bundle_scores[required].copy()
    frame["admin_dong_code"] = _clean_code(frame["admin_dong_code"])
    frame["bundle_id"] = frame["bundle_id"].astype(str)
    frame["bundle_gap_score"] = _finite_numeric(
        frame["bundle_gap_score"], "bundle_gap_score"
    )
    _validate_unique(frame, ["admin_dong_code", "bundle_id"], "Bundle scores")
    bundle_sets = frame.groupby("admin_dong_code")["bundle_id"].agg(lambda x: tuple(sorted(x)))
    if bundle_sets.nunique() != 1:
        raise ValueError("Every admin_dong must have the same bundle set")
    frame = _deterministic_rank(
        frame,
        group="admin_dong_code",
        item="bundle_id",
        score="bundle_gap_score",
        output="within_admin_bundle_rank",
    )

    admin_rows: list[dict[str, Any]] = []
    for admin, group in frame.groupby("admin_dong_code", sort=True):
        ordered = group.sort_values("within_admin_bundle_rank")
        scores = ordered["bundle_gap_score"].to_numpy(float)
        shifted = scores - min(float(scores.min()), 0.0)
        if shifted.sum() <= 0:
            shifted = np.ones(len(shifted), dtype=float)
        admin_rows.append(
            {
                "admin_dong_code": str(admin),
                "policy_sigungu_name": str(ordered.iloc[0]["policy_sigungu_name"]),
                "top1_bundle_id": str(ordered.iloc[0]["bundle_id"]),
                "top2_bundle_id": str(ordered.iloc[1]["bundle_id"]),
                "top1_score": float(scores[0]),
                "top2_score": float(scores[1]),
                "top1_top2_margin": float(scores[0] - scores[1]),
                "top1_top2_relative_margin": float(
                    (scores[0] - scores[1]) / max(abs(scores[0]), 1e-12)
                ),
                "within_admin_bundle_sd": float(np.std(scores, ddof=0)),
                "within_admin_bundle_range": float(np.ptp(scores)),
                "within_admin_bundle_score_share_entropy": _normalised_entropy(shifted),
            }
        )
    admin_summary = pd.DataFrame(admin_rows)

    bundle_ids = sorted(frame["bundle_id"].unique())
    frequency_rows = []
    for bundle in bundle_ids:
        top1_count = int(admin_summary["top1_bundle_id"].eq(bundle).sum())
        top2_count = int(admin_summary["top2_bundle_id"].eq(bundle).sum())
        top2_set_count = int(
            (admin_summary["top1_bundle_id"].eq(bundle) | admin_summary["top2_bundle_id"].eq(bundle)).sum()
        )
        frequency_rows.append(
            {
                "bundle_id": bundle,
                "top1_count": top1_count,
                "top1_fraction": top1_count / len(admin_summary),
                "top2_position_count": top2_count,
                "top2_position_fraction": top2_count / len(admin_summary),
                "top2_set_count": top2_set_count,
                "top2_set_fraction": top2_set_count / len(admin_summary),
            }
        )
    top_frequency = pd.DataFrame(frequency_rows)

    sigungu_summary = (
        admin_summary.groupby("policy_sigungu_name", as_index=False, sort=True)
        .agg(
            admin_count=("admin_dong_code", "nunique"),
            distinct_top1_bundles=("top1_bundle_id", "nunique"),
            mean_top1_top2_margin=("top1_top2_margin", "mean"),
            mean_within_admin_bundle_sd=("within_admin_bundle_sd", "mean"),
            mean_within_admin_bundle_range=("within_admin_bundle_range", "mean"),
            mean_within_admin_bundle_entropy=(
                "within_admin_bundle_score_share_entropy",
                "mean",
            ),
        )
    )

    variance_rows = []
    for bundle, group in frame.groupby("bundle_id", sort=True):
        values = group["bundle_gap_score"].to_numpy(float)
        grand_mean = float(values.mean())
        total_ss = float(np.square(values - grand_mean).sum())
        sigungu_stats = group.groupby("policy_sigungu_name")["bundle_gap_score"].agg(
            ["mean", "count"]
        )
        between_ss = float(
            (sigungu_stats["count"] * np.square(sigungu_stats["mean"] - grand_mean)).sum()
        )
        within_ss = max(total_ss - between_ss, 0.0)
        variance_rows.append(
            {
                "bundle_id": bundle,
                "admin_count": int(len(group)),
                "policy_sigungu_count": int(group["policy_sigungu_name"].nunique()),
                "total_sum_squares": total_ss,
                "between_sigungu_sum_squares": between_ss,
                "within_sigungu_sum_squares": within_ss,
                "between_sigungu_variance_share": (
                    between_ss / total_ss if total_ss > 0 else 0.0
                ),
                "within_sigungu_variance_share": (
                    within_ss / total_ss if total_ss > 0 else 0.0
                ),
            }
        )
    variance_decomposition = pd.DataFrame(variance_rows)
    top1_probabilities = top_frequency["top1_count"].to_numpy(float)
    summary = pd.DataFrame(
        [
            {
                "admin_count": int(frame["admin_dong_code"].nunique()),
                "bundle_count": int(frame["bundle_id"].nunique()),
                "policy_sigungu_count": int(frame["policy_sigungu_name"].nunique()),
                "dominant_top1_bundle": str(
                    top_frequency.sort_values(
                        ["top1_count", "bundle_id"], ascending=[False, True]
                    ).iloc[0]["bundle_id"]
                ),
                "dominant_top1_fraction": float(top_frequency["top1_fraction"].max()),
                "top1_choice_entropy": _normalised_entropy(top1_probabilities),
                "median_top1_top2_margin": float(admin_summary["top1_top2_margin"].median()),
                "median_top1_top2_relative_margin": float(
                    admin_summary["top1_top2_relative_margin"].median()
                ),
                "median_within_admin_bundle_sd": float(
                    admin_summary["within_admin_bundle_sd"].median()
                ),
                "median_within_admin_bundle_entropy": float(
                    admin_summary["within_admin_bundle_score_share_entropy"].median()
                ),
                "mean_between_sigungu_variance_share": float(
                    variance_decomposition["between_sigungu_variance_share"].mean()
                ),
                "all_bundles_ever_top1": bool(top_frequency["top1_count"].gt(0).all()),
            }
        ]
    )
    return BundleDifferentiationResult(
        admin_summary,
        top_frequency,
        sigungu_summary,
        variance_decomposition,
        summary,
    )


def _residualise(y: np.ndarray, control: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(control)), control])
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    return y - design @ coefficients


def diagnose_need_replication(
    stage1_scores: pd.DataFrame,
    bundle_scores: pd.DataFrame,
) -> NeedReplicationResult:
    """Measure Stage 1 Need replication and residual bundle differentiation.

    Residuals and partial correlations are diagnostics only.  They must not
    silently replace the transparent Stage 2A bundle gap score.
    """

    need_column = "stage1_need_score" if "stage1_need_score" in stage1_scores else "need_score"
    _require_columns(stage1_scores, ["admin_dong_code", need_column], "Stage 1 scores")
    _require_columns(
        bundle_scores,
        ["admin_dong_code", "bundle_id", "bundle_gap_score"],
        "Bundle scores",
    )
    need = stage1_scores[["admin_dong_code", need_column]].copy().rename(
        columns={need_column: "stage1_need_score"}
    )
    bundles = bundle_scores[["admin_dong_code", "bundle_id", "bundle_gap_score"]].copy()
    for frame in (need, bundles):
        frame["admin_dong_code"] = _clean_code(frame["admin_dong_code"])
    bundles["bundle_id"] = bundles["bundle_id"].astype(str)
    _validate_unique(need, ["admin_dong_code"], "Stage 1 scores")
    _validate_unique(bundles, ["admin_dong_code", "bundle_id"], "Bundle scores")
    need["stage1_need_score"] = _finite_numeric(need["stage1_need_score"], "stage1_need_score")
    bundles["bundle_gap_score"] = _finite_numeric(bundles["bundle_gap_score"], "bundle_gap_score")
    panel = bundles.merge(need, on="admin_dong_code", how="left", validate="many_to_one")
    if panel["stage1_need_score"].isna().any():
        raise ValueError("Some Stage 2A admins have no Stage 1 Need score")

    summary_rows = []
    residual_frames = []
    top_fraction = 0.20
    for bundle, group in panel.groupby("bundle_id", sort=True):
        ordered = group.sort_values("admin_dong_code").copy()
        need_rank = ordered["stage1_need_score"].rank(method="average").to_numpy(float)
        gap_rank = ordered["bundle_gap_score"].rank(method="average").to_numpy(float)
        rho = _safe_correlation(need_rank, gap_rank)
        residual = _residualise(gap_rank, need_rank)
        ordered["stage1_need_rank"] = need_rank
        ordered["bundle_gap_rank_ascending"] = gap_rank
        ordered["need_controlled_bundle_rank_residual"] = residual
        residual_frames.append(ordered)
        top_n = max(1, int(math.ceil(len(ordered) * top_fraction)))
        need_top = set(
            ordered.nlargest(top_n, "stage1_need_score")["admin_dong_code"].astype(str)
        )
        gap_top = set(
            ordered.nlargest(top_n, "bundle_gap_score")["admin_dong_code"].astype(str)
        )
        summary_rows.append(
            {
                "bundle_id": bundle,
                "n_admin": int(len(ordered)),
                "stage1_need_bundle_gap_spearman": rho,
                "abs_stage1_need_bundle_gap_spearman": abs(rho),
                "rank_variance_explained_by_need": float(rho**2),
                "rank_variance_not_explained_by_need": float(max(0.0, 1.0 - rho**2)),
                "need_controlled_residual_sd": float(np.std(residual, ddof=0)),
                "top20_overlap_count": int(len(need_top & gap_top)),
                "top20_overlap_fraction": float(len(need_top & gap_top) / top_n),
                "residual_is_diagnostic_only": True,
            }
        )
    bundle_summary = pd.DataFrame(summary_rows)
    residual_panel = pd.concat(residual_frames, ignore_index=True)

    residual_wide = residual_panel.pivot(
        index="admin_dong_code",
        columns="bundle_id",
        values="need_controlled_bundle_rank_residual",
    ).sort_index()
    partial_rows = []
    bundle_ids = list(residual_wide.columns)
    for index, left in enumerate(bundle_ids):
        for right in bundle_ids[index + 1 :]:
            values = residual_wide[[left, right]].dropna()
            partial_rows.append(
                {
                    "bundle_id_left": left,
                    "bundle_id_right": right,
                    "n_admin": int(len(values)),
                    "partial_spearman_controlling_stage1_need": _safe_correlation(
                        values[left], values[right]
                    ),
                    "interpretation": "diagnostic_residual_differentiation_not_primary_score",
                }
            )
    partial_correlations = pd.DataFrame(partial_rows)
    return NeedReplicationResult(bundle_summary, partial_correlations, residual_panel)


def _coalesce_metric(frame: pd.DataFrame, target: str, candidates: Sequence[str]) -> None:
    if target in frame:
        return
    available = [column for column in candidates if column in frame]
    if not available:
        return
    values = frame[available].apply(pd.to_numeric, errors="coerce")
    if "regret" in target:
        frame[target] = values.max(axis=1)
    elif "margin" in target and "year_" in target:
        frame[target] = values.min(axis=1)
    else:
        frame[target] = values.min(axis=1)


def _normalise_release_columns(release: pd.DataFrame) -> pd.DataFrame:
    """Accept the hardening evidence table and canonical release spellings."""

    frame = release.copy()
    aliases = {
        "primary_season": ("final_primary", "recommended_season"),
        "fallback_season": ("final_fallback",),
        "temporal_confidence": ("confidence",),
        "diagnostic_primary_season": (
            "consensus_primary_candidate",
            "pooled_top1",
        ),
        "diagnostic_fallback_season": ("consensus_fallback_candidate",),
    }
    for target, candidates in aliases.items():
        if target in frame:
            continue
        for candidate in candidates:
            if candidate in frame:
                frame[target] = frame[candidate]
                break
    if "recommended_season" not in frame and "primary_season" in frame:
        frame["recommended_season"] = frame["primary_season"]
    return frame


def _materialise_release_metrics(evidence: pd.DataFrame) -> pd.DataFrame:
    frame = evidence.copy()
    aliases = {
        "bidirectional_primary_in_holdout_top2": (
            "primary_in_holdout_top2_22_to_23",
            "primary_in_holdout_top2_23_to_22",
            "holdout_top2_22_to_23",
            "holdout_top2_23_to_22",
        ),
        "bidirectional_selected_pair_holdout_top2_coverage": (
            "selected_pair_holdout_top2_coverage_22_to_23",
            "selected_pair_holdout_top2_coverage_23_to_22",
        ),
        "heldout_normalized_regret": (
            "heldout_normalized_regret_22_to_23",
            "heldout_normalized_regret_23_to_22",
            "heldout_regret_22_to_23",
            "heldout_regret_23_to_22",
        ),
        "year_normalized_top1_margin": (
            "year2022_normalized_top1_margin",
            "year2023_normalized_top1_margin",
            "normalized_top1_margin_2022",
            "normalized_top1_margin_2023",
        ),
    }
    for target, candidates in aliases.items():
        _coalesce_metric(frame, target, candidates)
    return frame


def _evaluate_threshold_section(
    frame: pd.DataFrame,
    rules: Mapping[str, Any],
) -> tuple[pd.Series, list[list[str]], list[list[str]]]:
    passed = pd.Series(True, index=frame.index, dtype=bool)
    failed: list[list[str]] = [[] for _ in range(len(frame))]
    missing: list[list[str]] = [[] for _ in range(len(frame))]
    for key, raw_threshold in rules.items():
        if key.endswith("_min"):
            metric, comparator = key[:-4], "min"
        elif key.endswith("_max"):
            metric, comparator = key[:-4], "max"
        else:
            continue
        if metric not in frame:
            passed[:] = False
            for values in missing:
                values.append(metric)
            continue
        values = pd.to_numeric(frame[metric], errors="coerce")
        threshold = float(raw_threshold)
        metric_pass = values.ge(threshold) if comparator == "min" else values.le(threshold)
        metric_pass = metric_pass & np.isfinite(values)
        passed &= metric_pass
        for position, is_pass in enumerate(metric_pass.to_numpy(bool)):
            if not is_pass:
                failed[position].append(key)
    return passed, failed, missing


def classify_confidence_aware_release(
    evidence: pd.DataFrame,
    release_rules: Mapping[str, Any],
) -> pd.DataFrame:
    """Apply preregistered strong/pair/abstention thresholds fail-closed.

    Missing or non-finite evidence can never produce a forced season.  The
    diagnostic primary/fallback are retained in separate audit columns, while
    an abstention has null released seasons.
    """

    frame = _materialise_release_metrics(evidence)
    if "primary_season" not in frame and "consensus_primary_candidate" in frame:
        frame["primary_season"] = frame["consensus_primary_candidate"]
    if "fallback_season" not in frame and "consensus_fallback_candidate" in frame:
        frame["fallback_season"] = frame["consensus_fallback_candidate"]
    _require_columns(frame, ["bundle_id", "primary_season", "fallback_season"], "Evidence")
    if frame.empty:
        raise ValueError("Temporal evidence is empty")
    if "admin_dong_code" in frame:
        frame["admin_dong_code"] = _clean_code(frame["admin_dong_code"])
        keys = ["admin_dong_code", "bundle_id"]
    else:
        keys = ["bundle_id"]
    frame["bundle_id"] = frame["bundle_id"].astype(str)
    _validate_unique(frame, keys, "Temporal release evidence")
    strong_rules = release_rules.get("strong_single", {})
    pair_rules = release_rules.get("robust_pair", {})
    if not isinstance(strong_rules, Mapping) or not isinstance(pair_rules, Mapping):
        raise TypeError("release_rules must contain strong_single and robust_pair mappings")
    strong_pass, strong_failed, strong_missing = _evaluate_threshold_section(frame, strong_rules)
    pair_pass, pair_failed, pair_missing = _evaluate_threshold_section(frame, pair_rules)
    valid_pair = (
        frame["primary_season"].isin(VALID_SEASONS)
        & frame["fallback_season"].isin(VALID_SEASONS)
        & frame["primary_season"].ne(frame["fallback_season"])
    )
    strong_pass &= valid_pair
    pair_pass &= valid_pair

    frame["diagnostic_primary_season"] = frame["primary_season"]
    frame["diagnostic_fallback_season"] = frame["fallback_season"]
    release_type = np.where(
        strong_pass,
        "STRONG_SINGLE",
        np.where(pair_pass, "ROBUST_PAIR", "NO_STRONG_PREFERENCE"),
    )
    confidence = np.where(strong_pass, "HIGH", np.where(pair_pass, "MODERATE", "LOW"))
    frame["release_type"] = release_type
    frame["temporal_confidence"] = confidence
    abstain = frame["release_type"].eq("NO_STRONG_PREFERENCE")
    frame.loc[abstain, ["primary_season", "fallback_season"]] = pd.NA
    frame["recommended_season"] = frame["primary_season"]
    frame["exact_month"] = pd.NA
    frame["exact_date"] = pd.NA
    frame["missing_strong_metrics"] = ["|".join(values) for values in strong_missing]
    frame["missing_pair_metrics"] = ["|".join(values) for values in pair_missing]
    reasons = []
    for position, kind in enumerate(release_type):
        if kind == "STRONG_SINGLE":
            reasons.append("all_preregistered_strong_single_thresholds_passed")
        elif kind == "ROBUST_PAIR":
            reasons.append(
                "strong_single_not_supported;preregistered_robust_pair_thresholds_passed;"
                + "|".join(strong_failed[position])
            )
        else:
            failures = sorted(set(pair_failed[position] + pair_missing[position]))
            reasons.append(
                "insufficient_for_forced_season;explicit_abstention;" + "|".join(failures)
            )
    frame["release_reason"] = reasons
    frame["threshold_contract_pass"] = True
    validate_confidence_release(frame, raise_on_error=True)
    return frame


def validate_confidence_release(
    release: pd.DataFrame,
    *,
    raise_on_error: bool = False,
) -> pd.DataFrame:
    """Return hard gates for release enums, abstention, and false precision."""

    frame = _normalise_release_columns(release)
    required = [
        "bundle_id",
        "primary_season",
        "fallback_season",
        "release_type",
        "temporal_confidence",
        "exact_month",
        "exact_date",
    ]
    missing = sorted(set(required) - set(frame.columns))
    rows: list[dict[str, Any]] = []

    def add(name: str, observed: Any, expected: Any, passed: bool, detail: str = "") -> None:
        rows.append(
            {
                "gate": name,
                "observed": observed,
                "comparator": "==",
                "expected": expected,
                "passed": bool(passed),
                "severity": "hard",
                "detail": detail,
            }
        )

    add("release_required_columns_present", len(missing), 0, not missing, "|".join(missing))
    if not missing:
        add(
            "release_type_valid_enum",
            int((~frame["release_type"].isin(VALID_RELEASE_TYPES)).sum()),
            0,
            frame["release_type"].isin(VALID_RELEASE_TYPES).all(),
        )
        add(
            "temporal_confidence_valid_enum",
            int((~frame["temporal_confidence"].isin(VALID_TEMPORAL_CONFIDENCE)).sum()),
            0,
            frame["temporal_confidence"].isin(VALID_TEMPORAL_CONFIDENCE).all(),
        )
        abstain = frame["release_type"].eq("NO_STRONG_PREFERENCE")
        actionable = ~abstain
        add(
            "abstention_has_null_seasons",
            int(frame.loc[abstain, ["primary_season", "fallback_season"]].notna().sum().sum()),
            0,
            frame.loc[abstain, ["primary_season", "fallback_season"]].isna().all().all(),
        )
        actionable_valid = (
            frame.loc[actionable, "primary_season"].isin(VALID_SEASONS)
            & frame.loc[actionable, "fallback_season"].isin(VALID_SEASONS)
            & frame.loc[actionable, "primary_season"].ne(
                frame.loc[actionable, "fallback_season"]
            )
        )
        add(
            "actionable_seasons_valid_and_distinct",
            int(actionable_valid.sum()),
            int(actionable.sum()),
            actionable_valid.all(),
        )
        confidence_mapping = {
            "STRONG_SINGLE": "HIGH",
            "ROBUST_PAIR": "MODERATE",
            "NO_STRONG_PREFERENCE": "LOW",
        }
        expected_confidence = frame["release_type"].map(confidence_mapping)
        add(
            "release_confidence_mapping",
            int(frame["temporal_confidence"].eq(expected_confidence).sum()),
            len(frame),
            frame["temporal_confidence"].eq(expected_confidence).all(),
        )
        add(
            "exact_month_all_null",
            int(frame["exact_month"].notna().sum()),
            0,
            frame["exact_month"].isna().all(),
        )
        add(
            "exact_date_all_null",
            int(frame["exact_date"].notna().sum()),
            0,
            frame["exact_date"].isna().all(),
        )
    gates = pd.DataFrame(rows)
    if raise_on_error and not gates["passed"].all():
        failures = gates.loc[~gates["passed"], "gate"].tolist()
        raise ValueError(f"Temporal release contract failed: {failures}")
    return gates


def build_stage2_to_stage3_interface(
    stage1_scores: pd.DataFrame,
    bundle_scores: pd.DataFrame,
    temporal_release: pd.DataFrame,
    *,
    season_scores: pd.DataFrame | None = None,
    temporal_independent_n: int = 55,
    expected_admins: int | None = 153,
    expected_bundles: int | None = 5,
) -> pd.DataFrame:
    """Build the Stage 2 hand-off schema without running any Stage 3 model."""

    need_column = "stage1_need_score" if "stage1_need_score" in stage1_scores else "need_score"
    _require_columns(stage1_scores, ["admin_dong_code", need_column], "Stage 1 scores")
    _require_columns(
        bundle_scores,
        ["admin_dong_code", "bundle_id", "bundle_gap_score"],
        "Stage 2A bundle scores",
    )
    release = _normalise_release_columns(temporal_release)
    _require_columns(
        release,
        ["bundle_id", "release_type", "temporal_confidence"],
        "Temporal release",
    )
    validate_confidence_release(release, raise_on_error=True)
    if temporal_independent_n <= 0:
        raise ValueError("temporal_independent_n must be positive")

    need = stage1_scores[["admin_dong_code", need_column]].copy().rename(
        columns={need_column: "stage1_need_score"}
    )
    bundles = bundle_scores[["admin_dong_code", "bundle_id", "bundle_gap_score"]].copy()
    need["admin_dong_code"] = _clean_code(need["admin_dong_code"])
    bundles["admin_dong_code"] = _clean_code(bundles["admin_dong_code"])
    bundles["bundle_id"] = bundles["bundle_id"].astype(str)
    _validate_unique(need, ["admin_dong_code"], "Stage 1 scores")
    _validate_unique(bundles, ["admin_dong_code", "bundle_id"], "Stage 2A bundle scores")
    bundles = _deterministic_rank(
        bundles,
        group="admin_dong_code",
        item="bundle_id",
        score="bundle_gap_score",
        output="bundle_rank",
    )
    panel = bundles.merge(need, on="admin_dong_code", how="left", validate="many_to_one")
    if panel["stage1_need_score"].isna().any():
        raise ValueError("Stage 1 Need is missing for one or more Stage 2A admins")

    release["bundle_id"] = release["bundle_id"].astype(str)
    if "recommended_season" not in release:
        release["recommended_season"] = release["primary_season"]
    if "temporal_fit_score" not in release:
        for candidate in (
            "primary_temporal_fit_score",
            "pooled_primary_temporal_fit_score",
            "diagnostic_primary_temporal_fit_score",
        ):
            if candidate in release:
                release["temporal_fit_score"] = release[candidate]
                break
    if "temporal_fit_score" not in release and season_scores is not None:
        score_frame = season_scores.copy()
        _require_columns(
            score_frame,
            ["bundle_id", "season", "temporal_fit_score"],
            "Season scores for interface",
        )
        score_frame["bundle_id"] = score_frame["bundle_id"].astype(str)
        selection_column = (
            "diagnostic_primary_season"
            if "diagnostic_primary_season" in release
            else "recommended_season"
        )
        selection = release[["bundle_id", selection_column]].rename(
            columns={selection_column: "__selected_season"}
        )
        if "admin_dong_code" in release:
            selection["admin_dong_code"] = _clean_code(release["admin_dong_code"])
        if "admin_dong_code" in score_frame:
            score_frame["admin_dong_code"] = _clean_code(score_frame["admin_dong_code"])
            if "admin_dong_code" not in selection:
                profile_counts = score_frame.groupby(
                    ["bundle_id", "season"]
                )["temporal_fit_score"].nunique(dropna=False)
                if not profile_counts.eq(1).all():
                    raise ValueError(
                        "Bundle-level release cannot use spatially varying season scores"
                    )
                score_frame = score_frame.drop_duplicates(["bundle_id", "season"])
                score_keys = ["bundle_id"]
            else:
                score_keys = ["admin_dong_code", "bundle_id"]
        else:
            if "admin_dong_code" in selection:
                raise ValueError(
                    "Admin-level release requires admin-level season scores for interface"
                )
            score_keys = ["bundle_id"]
        lookup = selection.merge(
            score_frame,
            left_on=[*score_keys, "__selected_season"],
            right_on=[*score_keys, "season"],
            how="left",
            validate="one_to_one",
        )
        if lookup["temporal_fit_score"].isna().any():
            raise ValueError("Could not resolve temporal fit for every release row")
        release = release.merge(
            lookup[[*score_keys, "temporal_fit_score"]],
            on=score_keys,
            how="left",
            validate="one_to_one",
        )
    _require_columns(
        release,
        ["recommended_season", "fallback_season", "temporal_fit_score"],
        "Temporal release",
    )
    release_columns = [
        "bundle_id",
        "recommended_season",
        "fallback_season",
        "release_type",
        "temporal_confidence",
        "temporal_fit_score",
    ]
    if "admin_dong_code" in release:
        release["admin_dong_code"] = _clean_code(release["admin_dong_code"])
        _validate_unique(release, ["admin_dong_code", "bundle_id"], "Temporal release")
        panel = panel.merge(
            release[["admin_dong_code", *release_columns]],
            on=["admin_dong_code", "bundle_id"],
            how="left",
            validate="one_to_one",
        )
    else:
        _validate_unique(release, ["bundle_id"], "Temporal release")
        panel = panel.merge(
            release[release_columns], on="bundle_id", how="left", validate="many_to_one"
        )
    if panel["release_type"].isna().any():
        raise ValueError("Temporal release does not cover every Stage 2A bundle key")
    panel = panel.rename(columns={"bundle_gap_score": "specialty_gap_score"})
    panel["temporal_independent_n"] = int(temporal_independent_n)
    panel["exact_month"] = pd.NA
    panel["exact_date"] = pd.NA
    interface = panel[list(STAGE2_TO_STAGE3_COLUMNS)].sort_values(
        ["admin_dong_code", "bundle_rank"], kind="mergesort"
    ).reset_index(drop=True)
    validate_stage2_to_stage3_interface(
        interface,
        expected_admins=expected_admins,
        expected_bundles=expected_bundles,
        expected_independent_n=temporal_independent_n,
        stage3_started=False,
        raise_on_error=True,
    )
    return interface


def validate_stage2_to_stage3_interface(
    interface: pd.DataFrame,
    *,
    expected_admins: int | None = 153,
    expected_bundles: int | None = 5,
    expected_independent_n: int | None = 55,
    stage3_started: bool = False,
    raise_on_error: bool = False,
) -> pd.DataFrame:
    """Validate the hand-off schema and explicitly assert Stage 3 is unstarted."""

    rows: list[dict[str, Any]] = []

    def add(
        name: str,
        observed: Any,
        comparator: str,
        expected: Any,
        passed: bool,
        detail: str = "",
    ) -> None:
        rows.append(
            {
                "gate": name,
                "observed": observed,
                "comparator": comparator,
                "expected": expected,
                "passed": bool(passed),
                "severity": "hard",
                "detail": detail,
            }
        )

    missing = sorted(set(STAGE2_TO_STAGE3_COLUMNS) - set(interface.columns))
    add("interface_required_columns", len(missing), "==", 0, not missing, "|".join(missing))
    add("stage3_started", bool(stage3_started), "==", False, not stage3_started)
    if not missing:
        frame = interface.copy()
        frame["admin_dong_code"] = _clean_code(frame["admin_dong_code"])
        admin_count = int(frame["admin_dong_code"].nunique())
        bundle_count = int(frame["bundle_id"].nunique())
        if expected_admins is not None:
            add("interface_admin_count", admin_count, "==", expected_admins, admin_count == expected_admins)
        if expected_bundles is not None:
            add(
                "interface_bundle_count",
                bundle_count,
                "==",
                expected_bundles,
                bundle_count == expected_bundles,
            )
        expected_rows = (
            expected_admins * expected_bundles
            if expected_admins is not None and expected_bundles is not None
            else len(frame)
        )
        add("interface_row_count", len(frame), "==", expected_rows, len(frame) == expected_rows)
        duplicate_count = int(frame.duplicated(["admin_dong_code", "bundle_id"]).sum())
        add("interface_keys_unique", duplicate_count, "==", 0, duplicate_count == 0)
        nonfinite = 0
        for column in ("stage1_need_score", "specialty_gap_score", "temporal_fit_score"):
            values = pd.to_numeric(frame[column], errors="coerce")
            nonfinite += int((~np.isfinite(values)).sum())
        add("interface_required_metrics_finite", nonfinite, "==", 0, nonfinite == 0)
        rank_contract = True
        if expected_bundles is not None:
            expected_ranks = set(range(1, expected_bundles + 1))
            rank_contract = all(
                set(pd.to_numeric(group["bundle_rank"], errors="coerce")) == expected_ranks
                for _, group in frame.groupby("admin_dong_code")
            )
        add("interface_bundle_rank_permutation", rank_contract, "==", True, rank_contract)
        add(
            "interface_release_type_enum",
            int((~frame["release_type"].isin(VALID_RELEASE_TYPES)).sum()),
            "==",
            0,
            frame["release_type"].isin(VALID_RELEASE_TYPES).all(),
        )
        add(
            "interface_confidence_enum",
            int((~frame["temporal_confidence"].isin(VALID_TEMPORAL_CONFIDENCE)).sum()),
            "==",
            0,
            frame["temporal_confidence"].isin(VALID_TEMPORAL_CONFIDENCE).all(),
        )
        abstain = frame["release_type"].eq("NO_STRONG_PREFERENCE")
        abstain_nonnull = int(
            frame.loc[abstain, ["recommended_season", "fallback_season"]].notna().sum().sum()
        )
        add("interface_abstention_seasons_null", abstain_nonnull, "==", 0, abstain_nonnull == 0)
        actionable = ~abstain
        actionable_valid = (
            frame.loc[actionable, "recommended_season"].isin(VALID_SEASONS)
            & frame.loc[actionable, "fallback_season"].isin(VALID_SEASONS)
            & frame.loc[actionable, "recommended_season"].ne(
                frame.loc[actionable, "fallback_season"]
            )
        )
        add(
            "interface_actionable_seasons",
            int(actionable_valid.sum()),
            "==",
            int(actionable.sum()),
            actionable_valid.all(),
        )
        exact_month_count = int(frame["exact_month"].notna().sum())
        exact_date_count = int(frame["exact_date"].notna().sum())
        add("interface_exact_month_null", exact_month_count, "==", 0, exact_month_count == 0)
        add("interface_exact_date_null", exact_date_count, "==", 0, exact_date_count == 0)
        observed_n = pd.to_numeric(frame["temporal_independent_n"], errors="coerce")
        if expected_independent_n is not None:
            n_match = observed_n.eq(expected_independent_n).all()
            add(
                "interface_temporal_independent_n",
                sorted(observed_n.dropna().unique().tolist()),
                "==",
                [expected_independent_n],
                n_match,
            )
        pseudo_replication = bool(observed_n.eq(len(frame)).any())
        add(
            "interface_no_delivery_row_pseudoreplication",
            pseudo_replication,
            "==",
            False,
            not pseudo_replication,
        )
        need_unique = frame.groupby("admin_dong_code")["stage1_need_score"].nunique(dropna=False)
        add(
            "interface_stage1_need_constant_across_bundles",
            int(need_unique.max()),
            "==",
            1,
            need_unique.eq(1).all(),
        )
        forbidden_prefixes = ("venue_", "exposure_", "candidate_", "optimizer_")
        forbidden = sorted(
            column for column in frame.columns if column.lower().startswith(forbidden_prefixes)
        )
        add(
            "interface_contains_no_stage3_outputs",
            len(forbidden),
            "==",
            0,
            not forbidden,
            "|".join(forbidden),
        )
    gates = pd.DataFrame(rows)
    if raise_on_error and not gates["passed"].all():
        failures = gates.loc[~gates["passed"], "gate"].tolist()
        raise ValueError(f"Stage 2 -> Stage 3 interface contract failed: {failures}")
    return gates


def stage2_to_stage3_schema() -> pd.DataFrame:
    """Return the human/audit-readable Stage 3 hand-off schema."""

    descriptions = {
        "admin_dong_code": "153-row delivery geography key; not a temporal replicate",
        "stage1_need_score": "frozen Stage 1 WHERE NEED score",
        "bundle_id": "Stage 2A WHAT service-bundle key",
        "specialty_gap_score": "frozen Stage 2A bundle gap score",
        "bundle_rank": "within-admin Stage 2A WHAT rank",
        "recommended_season": "coarse season or null on abstention",
        "fallback_season": "different coarse season or null on abstention",
        "release_type": "STRONG_SINGLE, ROBUST_PAIR, or NO_STRONG_PREFERENCE",
        "temporal_confidence": "HIGH, MODERATE, or LOW",
        "temporal_fit_score": "diagnostic coarse scheduling fit, not patient demand",
        "temporal_independent_n": "policy-sigungu x bundle units, currently 55",
        "exact_month": "must remain null",
        "exact_date": "must remain null",
    }
    nullable = {"recommended_season", "fallback_season", "exact_month", "exact_date"}
    return pd.DataFrame(
        [
            {
                "column": column,
                "order": index,
                "nullable": column in nullable,
                "description": descriptions[column],
                "stage3_execution": False,
            }
            for index, column in enumerate(STAGE2_TO_STAGE3_COLUMNS, start=1)
        ]
    )
