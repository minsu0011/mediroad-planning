"""Stage 2 correlation diagnostics and publication-quality atlases.

The Stage 2 specialty and temporal layers contain measurements at several
spatial resolutions.  This module deliberately exposes only a
same-granularity correlation path.  A caller must supply feature metadata and
mixed spatial resolutions are rejected instead of being silently pooled.

The plotting functions wrap the mature Stage 1 correlation reporter for
specialty/bundle score atlases and provide a dedicated month-profile plot for
the temporal layer.  Every interactive artifact embeds Plotly JavaScript so
that it can be opened without a network connection.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from mediroad.reporting.correlation import (
    correlation_quality_metrics,
    hierarchical_order,
    plot_correlation_map,
    save_interactive_heatmap,
    spearman_with_counts,
)


os.environ.setdefault("MPLBACKEND", "Agg")


def _matplotlib_korean_font() -> str:
    """Register Malgun Gothic from the Windows host when running under WSL."""

    try:
        from matplotlib import font_manager
    except ImportError:  # pragma: no cover - handled by plotting caller
        return "DejaVu Sans"
    for regular, bold in (
        (
            Path("/mnt/c/Windows/Fonts/malgun.ttf"),
            Path("/mnt/c/Windows/Fonts/malgunbd.ttf"),
        ),
        (
            Path("C:/Windows/Fonts/malgun.ttf"),
            Path("C:/Windows/Fonts/malgunbd.ttf"),
        ),
        (Path("/usr/share/fonts/truetype/nanum/NanumGothic.ttf"), None),
    ):
        if regular.is_file():
            font_manager.fontManager.addfont(str(regular))
            if bold is not None and bold.is_file():
                font_manager.fontManager.addfont(str(bold))
            return font_manager.FontProperties(fname=str(regular)).get_name()
    return "DejaVu Sans"


_FEATURE_FIELDS = (
    "source_column",
    "column_name",
    "feature_name",
    "featureName",
    "feature",
    "name",
)
_GRANULARITY_FIELDS = (
    "granularity",
    "native_granularity",
    "analysis_granularity",
)
_LABEL_FIELDS = (
    "label_ko",
    "display_label",
    "human_label",
    "feature_label",
    "label",
    "description",
)
_BUNDLE_FIELDS = (
    "bundle_id",
    "service_bundle",
    "bundle",
    "bundle_name",
    "semantic_group",
    "axis",
)
_FEATURE_ID_FIELDS = ("feature_id", "featureId", "fid", "display_id")
_EQUIVALENCE_KEY_FIELDS = (
    "equivalence_key",
    "shared_source_key",
    "semantic_source_key",
    "component_id",
)
_EQUIVALENCE_KIND_FIELDS = (
    "equivalence_kind",
    "feature_kind",
    "source_kind",
)
_SERVICE_FIELDS = ("service_id", "specialty_id", "service")

ManifestLike = (
    pd.DataFrame
    | Mapping[str, Any]
    | Sequence[Mapping[str, Any]]
)


class MixedSampleSizeError(ValueError):
    """Raised when spectral statistics are requested for a mixed-n matrix."""


@dataclass(frozen=True)
class SpecialtyBundleAtlasResult:
    """In-memory diagnostics and paths produced by a score-atlas run."""

    correlation: pd.DataFrame
    pairwise_n: pd.DataFrame
    ordered_features: tuple[str, ...]
    granularity: str
    separation_metrics: Mapping[str, float | int]
    stage1_overlap: pd.DataFrame | None
    stage1_overlap_summary: Mapping[str, float | int] | None
    quality_metrics: Mapping[str, Any]
    manifest: pd.DataFrame
    high_correlation_ledger: pd.DataFrame
    artifacts: Mapping[str, str]


@dataclass(frozen=True)
class TemporalMonthProfileResult:
    """Validated month-by-profile matrix and rendered artifact paths."""

    month_profile: pd.DataFrame
    granularity: str
    artifacts: Mapping[str, str]


def _first_field(frame: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    return next((field for field in candidates if field in frame.columns), None)


def _manifest_table(manifest: ManifestLike, features: Sequence[str]) -> pd.DataFrame:
    """Normalize supported manifest shapes and require complete feature coverage."""

    if isinstance(manifest, pd.DataFrame):
        raw = manifest.copy()
    elif isinstance(manifest, Mapping):
        nested_features = manifest.get("features")
        if isinstance(nested_features, Sequence) and not isinstance(
            nested_features, (str, bytes)
        ):
            raw = pd.DataFrame(list(nested_features))
        elif manifest and all(isinstance(value, Mapping) for value in manifest.values()):
            raw = pd.DataFrame.from_dict(manifest, orient="index")
            if _first_field(raw, _FEATURE_FIELDS) is None:
                raw.insert(0, "source_column", raw.index.map(str))
        else:
            raw = pd.DataFrame(manifest)
    elif isinstance(manifest, Sequence) and not isinstance(manifest, (str, bytes)):
        raw = pd.DataFrame(list(manifest))
    else:
        raise TypeError("manifest must be a DataFrame, mapping, or sequence of mappings")

    key = _first_field(raw, _FEATURE_FIELDS)
    if key is None:
        if isinstance(raw.index, pd.RangeIndex):
            raise ValueError("manifest requires an explicit feature-name field")
        raw = raw.copy()
        raw.insert(0, "source_column", raw.index.map(str))
        key = "source_column"
    raw[key] = raw[key].astype(str)
    duplicated = raw.loc[raw[key].duplicated(keep=False), key].unique().tolist()
    if duplicated:
        raise ValueError(f"manifest feature names must be unique; duplicated={duplicated[:5]}")

    feature_names = [str(feature) for feature in features]
    indexed = raw.set_index(key, drop=False)
    missing = [feature for feature in feature_names if feature not in indexed.index]
    if missing:
        raise ValueError(
            "manifest does not describe every score feature; missing="
            + ", ".join(missing[:10])
        )
    selected = indexed.reindex(feature_names).copy()
    if "source_column" not in selected.columns:
        selected["source_column"] = feature_names
    else:
        selected["source_column"] = feature_names
    selected.index = pd.Index(feature_names, name="feature_name")
    return selected


def stabilize_feature_manifest(
    manifest: ManifestLike,
    features: Sequence[str],
    *,
    id_prefix: str = "F",
    minimum_width: int = 4,
) -> pd.DataFrame:
    """Return complete metadata with IDs assigned before any clustering.

    Existing non-empty IDs are preserved.  Missing IDs are assigned from the
    supplied pre-clustering feature order, so the same normalized manifest can
    be passed to the atlas, interactive renderer, and bootstrap reporter.
    Callers should keep and reuse the returned frame; plotting-order-specific
    fallback IDs are deliberately avoided.
    """

    feature_names = [str(feature) for feature in features]
    if len(set(feature_names)) != len(feature_names):
        raise ValueError("features must be unique before stable IDs are assigned")
    prefix = str(id_prefix).strip()
    if not prefix:
        raise ValueError("id_prefix must be non-empty")
    if not isinstance(minimum_width, (int, np.integer)) or int(minimum_width) < 1:
        raise ValueError("minimum_width must be a positive integer")

    meta = _manifest_table(manifest, feature_names).copy()
    id_field = _first_field(meta, _FEATURE_ID_FIELDS)
    identifiers = (
        meta[id_field].astype("object").copy()
        if id_field is not None
        else pd.Series(index=meta.index, dtype="object")
    )
    missing = identifiers.isna() | identifiers.astype(str).str.strip().eq("")
    identifiers.loc[~missing] = identifiers.loc[~missing].astype(str).str.strip()
    duplicate_existing = identifiers.loc[~missing][
        identifiers.loc[~missing].duplicated(keep=False)
    ].unique().tolist()
    if duplicate_existing:
        raise ValueError(
            "manifest feature IDs must be unique; duplicated="
            + ", ".join(map(str, duplicate_existing[:5]))
        )

    width = max(int(minimum_width), len(str(max(len(feature_names), 1))))
    used = set(identifiers.loc[~missing].astype(str))
    for position, feature in enumerate(feature_names, start=1):
        if not bool(missing.loc[feature]):
            continue
        candidate_position = position
        candidate = f"{prefix}{candidate_position:0{width}d}"
        while candidate in used:
            candidate_position += len(feature_names)
            candidate = f"{prefix}{candidate_position:0{width}d}"
        identifiers.loc[feature] = candidate
        used.add(candidate)

    if id_field is not None and id_field != "feature_id":
        meta = meta.drop(columns=[id_field])
    meta["feature_id"] = identifiers.astype(str)
    meta["precluster_order"] = np.arange(1, len(meta) + 1, dtype=int)
    preferred = ["source_column", "feature_id", "precluster_order"]
    remaining = [column for column in meta.columns if column not in preferred]
    return meta[preferred + remaining].reset_index(drop=True)


def _nonempty_text(value: Any) -> str:
    try:
        if value is None or bool(pd.isna(value)):
            return ""
    except (TypeError, ValueError):
        if value is None:
            return ""
    return str(value).strip()


def _feature_semantics(row: pd.Series) -> tuple[str, str, str, bool]:
    """Return service, semantic key, kind, and explicit-key flag."""

    feature = str(row["source_column"])
    service_field = next(
        (field for field in _SERVICE_FIELDS if field in row.index), None
    )
    service = _nonempty_text(row.get(service_field)) if service_field else ""
    parsed_service, separator, parsed_key = feature.partition("::")
    if not service and separator:
        service = parsed_service.strip()

    explicit_field = next(
        (
            field
            for field in _EQUIVALENCE_KEY_FIELDS
            if field in row.index and _nonempty_text(row.get(field))
        ),
        None,
    )
    semantic_key = (
        _nonempty_text(row.get(explicit_field))
        if explicit_field is not None
        else parsed_key.strip() if separator else ""
    )
    explicit_kind_field = next(
        (
            field
            for field in _EQUIVALENCE_KIND_FIELDS
            if field in row.index and _nonempty_text(row.get(field))
        ),
        None,
    )
    if explicit_kind_field is not None:
        kind = _nonempty_text(row.get(explicit_kind_field)).lower()
    else:
        role = _nonempty_text(row.get("role")).lower()
        key_lower = semantic_key.lower()
        final_outputs = {
            "specialty_gap_score",
            "bundle_gap_score",
            "temporal_fit_score",
            "need_score",
            "priority_score",
            "final_score",
        }
        if key_lower in final_outputs:
            kind = "model_output"
        elif "component" in role or key_lower.endswith("_gap_score"):
            kind = "component"
        elif "source" in role or separator:
            kind = "source"
        else:
            kind = "unknown"
    return service, semantic_key, kind, explicit_field is not None


def high_correlation_equivalence_ledger(
    corr: pd.DataFrame,
    manifest: ManifestLike,
    *,
    threshold: float = 0.98,
) -> pd.DataFrame:
    """Explain every high-absolute-correlation pair in an atlas.

    Service-prefixed features of the form ``service::source_or_component`` are
    classified as expected structural repetition only when both sides share
    the same non-output semantic key and refer to different services.  An
    explicit ``equivalence_key`` (or compatible manifest field) takes
    precedence.  All other high-correlation pairs remain visibly unexplained.
    """

    if not isinstance(threshold, (int, float, np.integer, np.floating)):
        raise TypeError("threshold must be numeric")
    threshold = float(threshold)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must lie in [0, 1]")
    if not isinstance(corr, pd.DataFrame) or corr.empty:
        raise ValueError("corr must be a non-empty DataFrame")
    if corr.shape[0] != corr.shape[1]:
        raise ValueError("corr must be square")
    aligned = corr.copy()
    aligned.index = aligned.index.map(str)
    aligned.columns = aligned.columns.map(str)
    if not aligned.index.is_unique or not aligned.columns.is_unique:
        raise ValueError("corr axes must be unique")
    if set(aligned.index) != set(aligned.columns):
        raise ValueError("corr rows and columns must contain the same features")
    aligned = aligned.reindex(index=aligned.index, columns=aligned.index).apply(
        pd.to_numeric, errors="coerce"
    )
    values = aligned.to_numpy(dtype=float)
    if not np.allclose(values, values.T, equal_nan=True, atol=1e-10):
        raise ValueError("corr must be symmetric")

    stable = stabilize_feature_manifest(manifest, aligned.index.tolist())
    meta = stable.set_index("source_column", drop=False).reindex(aligned.index)
    semantics = {
        feature: _feature_semantics(meta.loc[feature]) for feature in aligned.index
    }
    columns = [
        "feature_a",
        "feature_b",
        "feature_id_a",
        "feature_id_b",
        "rho",
        "abs_rho",
        "service_a",
        "service_b",
        "semantic_key_a",
        "semantic_key_b",
        "semantic_kind_a",
        "semantic_kind_b",
        "expected_redundancy",
        "explanation_code",
        "explanation",
    ]
    rows: list[dict[str, Any]] = []
    for left_position, left in enumerate(aligned.index):
        for right_position in range(left_position + 1, len(aligned.index)):
            right = aligned.index[right_position]
            rho = float(aligned.iloc[left_position, right_position])
            if not np.isfinite(rho) or abs(rho) + 1e-12 < threshold:
                continue
            service_a, key_a, kind_a, explicit_a = semantics[left]
            service_b, key_b, kind_b, explicit_b = semantics[right]
            same_key = bool(key_a and key_b and key_a == key_b)
            different_services = bool(
                service_a and service_b and service_a != service_b
            )
            explicit_equivalence = same_key and explicit_a and explicit_b
            repeated_semantic = (
                same_key
                and different_services
                and kind_a in {"source", "component"}
                and kind_b in {"source", "component"}
            )
            expected = bool(explicit_equivalence or repeated_semantic)
            if explicit_equivalence:
                code = "explicit_equivalence_key"
                explanation = (
                    f"Manifest-declared equivalence key '{key_a}' is shared."
                )
            elif repeated_semantic:
                code = f"repeated_{kind_a}_across_services"
                explanation = (
                    f"The same {kind_a} '{key_a}' is repeated for distinct "
                    f"services ({service_a}, {service_b})."
                )
            else:
                code = "unexplained_high_correlation"
                explanation = (
                    "No shared non-output source/component equivalence was "
                    "established; review for duplication or model collapse."
                )
            rows.append(
                {
                    "feature_a": left,
                    "feature_b": right,
                    "feature_id_a": meta.loc[left, "feature_id"],
                    "feature_id_b": meta.loc[right, "feature_id"],
                    "rho": rho,
                    "abs_rho": abs(rho),
                    "service_a": service_a,
                    "service_b": service_b,
                    "semantic_key_a": key_a,
                    "semantic_key_b": key_b,
                    "semantic_kind_a": kind_a,
                    "semantic_kind_b": kind_b,
                    "expected_redundancy": expected,
                    "explanation_code": code,
                    "explanation": explanation,
                }
            )
    ledger = pd.DataFrame(rows, columns=columns)
    if not ledger.empty:
        ledger = ledger.sort_values(
            ["expected_redundancy", "abs_rho", "feature_id_a", "feature_id_b"],
            ascending=[True, False, True, True],
            kind="stable",
        ).reset_index(drop=True)
    ledger.attrs.update(
        {
            "threshold": threshold,
            "pair_count": int(len(ledger)),
            "expected_redundancy_count": int(
                ledger["expected_redundancy"].sum() if len(ledger) else 0
            ),
            "unexplained_pair_count": int(
                (~ledger["expected_redundancy"]).sum() if len(ledger) else 0
            ),
        }
    )
    return ledger


def _granularity_family(value: Any) -> str:
    if value is None:
        return ""
    try:
        if bool(pd.isna(value)):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if not text or text in {"na", "n/a", "none", "unknown", "unspecified"}:
        return ""
    # Check sigungu first: labels such as policy_sigungu_repeated_on_admin_dong
    # contain both tokens but are statistically sigungu-level evidence.
    if "sigungu" in text or "si_gun_gu" in text:
        return "sigungu"
    if "admin_dong" in text or "administrative_dong" in text or text == "emd":
        return "admin_dong"
    if any(token in text for token in ("province", "provincial", "chungbuk")):
        return "province"
    if any(token in text for token in ("national", "nationwide", "country")):
        return "national"
    if text in {"global", "common"}:
        return text
    return text


def _single_granularity(meta: pd.DataFrame) -> str:
    field = _first_field(meta, _GRANULARITY_FIELDS)
    if field is None:
        raise ValueError(
            "manifest must declare granularity for every Stage 2 score feature"
        )
    granularities = meta[field].map(_granularity_family)
    missing = meta.index[granularities.eq("")].tolist()
    if missing:
        raise ValueError(
            "manifest has missing/unknown granularity; features="
            + ", ".join(map(str, missing[:10]))
        )
    unique = tuple(dict.fromkeys(granularities.tolist()))
    if len(unique) != 1:
        details = ", ".join(
            f"{feature}={granularity}"
            for feature, granularity in granularities.items()
        )
        raise ValueError(
            "same-granularity correlation cannot mix spatial units; " + details
        )
    return unique[0]


def _numeric_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    if frame.empty or frame.shape[1] < 2:
        raise ValueError("at least two score features and one observation are required")
    if not frame.columns.is_unique:
        duplicated = frame.columns[frame.columns.duplicated()].tolist()
        raise ValueError(f"score feature columns must be unique; duplicated={duplicated[:5]}")
    if not frame.index.is_unique:
        raise ValueError("observation index must be unique")
    numeric = frame.copy()
    numeric.columns = numeric.columns.map(str)
    if not numeric.columns.is_unique:
        raise ValueError("score feature names must remain unique after string conversion")
    numeric = numeric.apply(pd.to_numeric, errors="coerce")
    empty_columns = numeric.columns[numeric.notna().sum().eq(0)].tolist()
    if empty_columns:
        raise ValueError(
            "score features contain no numeric observations: "
            + ", ".join(empty_columns[:10])
        )
    return numeric


def same_granularity_correlation(
    frame: pd.DataFrame,
    manifest: ManifestLike,
    *,
    min_periods: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return a Spearman matrix and exact pairwise-n matrix for one native unit.

    The function does not collapse or broadcast observations.  ``frame`` must
    already contain one independent row per unit declared by the manifest.
    Mixed manifest granularities are rejected before any statistic is
    calculated.
    """

    numeric = _numeric_feature_frame(frame)
    meta = _manifest_table(manifest, numeric.columns)
    granularity = _single_granularity(meta)
    if not isinstance(min_periods, (int, np.integer)) or int(min_periods) < 2:
        raise ValueError("min_periods must be an integer >= 2")

    numeric.attrs.update(frame.attrs)
    granularity_field = _first_field(meta, _GRANULARITY_FIELDS)
    assert granularity_field is not None  # established by _single_granularity
    numeric.attrs["mediroad_granularity"] = {
        feature: meta.loc[feature, granularity_field]
        for feature in numeric.columns
    }
    numeric.attrs["mediroad_manifest"] = meta.reset_index(drop=True)
    if granularity == "sigungu":
        numeric.attrs["mediroad_analysis_level"] = "sigungu_collapsed"
        numeric.attrs["mediroad_group_count"] = len(numeric)
        numeric.attrs["mediroad_sigungu_repeated_features"] = list(numeric.columns)
    else:
        numeric.attrs["mediroad_analysis_level"] = granularity
        numeric.attrs["mediroad_sigungu_repeated_features"] = []

    # Reuse the Stage 1 guard/count contract.  Stage 1 ranks each whole column
    # before correlating, which differs slightly from exact pairwise Spearman
    # when the two features have different missingness.  pandas recalculates
    # ranks on each pairwise-complete subset, so use it for the Stage 2 rho
    # values while retaining the audited Stage 1 pairwise-n matrix.
    stage1_corr, pairwise_n = spearman_with_counts(
        numeric, min_periods=int(min_periods)
    )
    corr = numeric.corr(method="spearman", min_periods=int(min_periods))
    corr = corr.reindex(index=numeric.columns, columns=numeric.columns)
    corr = corr.mask(pairwise_n < int(min_periods))
    corr.attrs.update(stage1_corr.attrs)
    corr.attrs["method"] = "spearman_pairwise_complete_exact"
    extra_attrs = {
        "stage2_same_granularity": True,
        "stage2_granularity": granularity,
        "stage2_manifest_complete": True,
    }
    corr.attrs.update(extra_attrs)
    pairwise_n.attrs.update(extra_attrs)
    return corr, pairwise_n


def _aligned_corr_and_counts(
    corr: pd.DataFrame,
    pairwise_n: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not isinstance(corr, pd.DataFrame) or not isinstance(pairwise_n, pd.DataFrame):
        raise TypeError("corr and pairwise_n must be pandas DataFrames")
    if corr.shape[0] != corr.shape[1] or corr.empty:
        raise ValueError("corr must be a non-empty square matrix")
    if not corr.index.is_unique or not corr.columns.is_unique:
        raise ValueError("corr axes must be unique")
    aligned_corr = corr.copy()
    aligned_corr.index = aligned_corr.index.map(str)
    aligned_corr.columns = aligned_corr.columns.map(str)
    if set(aligned_corr.index) != set(aligned_corr.columns):
        raise ValueError("corr rows and columns must contain the same features")
    aligned_corr = aligned_corr.reindex(
        index=aligned_corr.index, columns=aligned_corr.index
    ).apply(pd.to_numeric, errors="coerce")

    aligned_n = pairwise_n.copy()
    aligned_n.index = aligned_n.index.map(str)
    aligned_n.columns = aligned_n.columns.map(str)
    missing_rows = set(aligned_corr.index) - set(aligned_n.index)
    missing_columns = set(aligned_corr.columns) - set(aligned_n.columns)
    if missing_rows or missing_columns:
        raise ValueError("pairwise_n must contain every correlation feature")
    aligned_n = aligned_n.reindex(
        index=aligned_corr.index, columns=aligned_corr.columns
    ).apply(pd.to_numeric, errors="coerce")

    corr_values = aligned_corr.to_numpy(dtype=float)
    count_values = aligned_n.to_numpy(dtype=float)
    if not np.allclose(corr_values, corr_values.T, equal_nan=True, atol=1e-10):
        raise ValueError("corr must be symmetric")
    if not np.allclose(count_values, count_values.T, equal_nan=True, atol=0):
        raise ValueError("pairwise_n must be symmetric")
    finite_corr = corr_values[np.isfinite(corr_values)]
    if finite_corr.size and np.any(np.abs(finite_corr) > 1.0 + 1e-10):
        raise ValueError("correlation values must lie in [-1, 1]")
    finite_n = count_values[np.isfinite(count_values)]
    if finite_n.size and (
        np.any(finite_n < 0) or not np.allclose(finite_n, np.rint(finite_n))
    ):
        raise ValueError("pairwise_n values must be non-negative integers")
    aligned_corr.attrs.update(corr.attrs)
    aligned_n.attrs.update(pairwise_n.attrs)
    return aligned_corr, aligned_n


def assert_spectral_metrics_allowed(
    corr: pd.DataFrame,
    pairwise_n: pd.DataFrame,
) -> int:
    """Require proof of one complete common sample before spectral analysis.

    Equal off-diagonal counts alone are insufficient: diagonal and
    off-diagonal counts must all be the same.  This guarantees, from the
    information available in a pairwise-n matrix, that every feature used the
    same observation set.  A heterogeneous count matrix raises
    :class:`MixedSampleSizeError`.
    """

    aligned_corr, aligned_n = _aligned_corr_and_counts(corr, pairwise_n)
    count_values = aligned_n.to_numpy(dtype=float)
    if not np.isfinite(count_values).all():
        raise MixedSampleSizeError(
            "spectral metrics are forbidden: pairwise_n contains unavailable cells"
        )
    unique_counts = np.unique(np.rint(count_values).astype(np.int64))
    if len(unique_counts) != 1:
        raise MixedSampleSizeError(
            "spectral metrics are forbidden for a mixed-n matrix; "
            f"observed sample sizes={unique_counts.tolist()}"
        )
    common_n = int(unique_counts[0])
    if common_n < 2:
        raise MixedSampleSizeError(
            "spectral metrics require at least two common observations"
        )

    values = aligned_corr.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise MixedSampleSizeError(
            "spectral metrics are forbidden unless the complete correlation "
            "matrix is finite (constant or under-supported features remain)"
        )
    if not np.allclose(np.diag(values), 1.0, atol=1e-10):
        raise ValueError("spectral metrics require a unit correlation diagonal")
    return common_n


def spectral_correlation_metrics(
    corr: pd.DataFrame,
    pairwise_n: pd.DataFrame,
) -> dict[str, float | int | bool]:
    """Calculate spectral diagnostics only after the common-sample guard."""

    common_n = assert_spectral_metrics_allowed(corr, pairwise_n)
    aligned_corr, _ = _aligned_corr_and_counts(corr, pairwise_n)
    values = aligned_corr.to_numpy(dtype=float)
    values = (values + values.T) / 2.0
    eigenvalues = np.linalg.eigvalsh(values)
    tolerance = max(1e-10, len(values) * np.finfo(float).eps * 10.0)
    if float(eigenvalues.min()) < -tolerance:
        raise ValueError(
            "common-n correlation matrix is not positive semidefinite; "
            "spectral metrics are unsafe"
        )
    nonnegative = np.clip(eigenvalues, 0.0, None)
    total = float(nonnegative.sum())
    probabilities = nonnegative[nonnegative > tolerance] / total
    effective_rank = (
        float(np.exp(-np.sum(probabilities * np.log(probabilities))))
        if probabilities.size and total > 0
        else 0.0
    )
    positive = nonnegative[nonnegative > tolerance]
    condition_number = (
        float(positive.max() / positive.min())
        if len(positive) == len(eigenvalues) and positive.size
        else float("inf")
    )
    inverse = np.linalg.pinv(values, rcond=tolerance, hermitian=True)
    vifs = np.diag(inverse)
    return {
        "common_sample_n": common_n,
        "effective_rank": effective_rank,
        "effective_rank_normalized": effective_rank / len(values),
        "correlation_condition_number": condition_number,
        "median_vif": float(np.median(vifs)),
        "max_vif": float(np.max(vifs)),
        "minimum_eigenvalue": float(eigenvalues.min()),
        "positive_semidefinite": True,
    }


def safe_correlation_quality_metrics(
    corr: pd.DataFrame,
    pairwise_n: pd.DataFrame,
    manifest: ManifestLike,
    *,
    include_spectral: bool = False,
) -> dict[str, Any]:
    """Wrap Stage 1 quality metrics with a fail-closed spectral contract."""

    aligned_corr, aligned_n = _aligned_corr_and_counts(corr, pairwise_n)
    meta = _manifest_table(manifest, aligned_corr.index)
    if include_spectral:
        assert_spectral_metrics_allowed(aligned_corr, aligned_n)
    return correlation_quality_metrics(
        aligned_corr,
        meta.reset_index(drop=True),
        spectral_metrics=bool(include_spectral),
    )


def _bundle_mapping(
    features: Sequence[str],
    manifest: ManifestLike | None,
    bundle_by_feature: Mapping[str, Any] | None,
) -> dict[str, str]:
    feature_names = [str(feature) for feature in features]
    if bundle_by_feature is not None:
        normalized = {}
        for key, value in bundle_by_feature.items():
            try:
                missing_value = value is None or bool(pd.isna(value))
            except (TypeError, ValueError):
                missing_value = value is None
            normalized[str(key)] = "" if missing_value else str(value).strip()
    else:
        if manifest is None:
            raise ValueError("manifest or bundle_by_feature is required")
        meta = _manifest_table(manifest, feature_names)
        field = _first_field(meta, _BUNDLE_FIELDS)
        if field is None:
            raise ValueError(
                "manifest needs bundle_id/service_bundle/bundle/semantic_group/axis"
            )
        normalized = {
            feature: "" if pd.isna(meta.loc[feature, field]) else str(meta.loc[feature, field]).strip()
            for feature in feature_names
        }
    missing = [feature for feature in feature_names if not normalized.get(feature, "")]
    if missing:
        raise ValueError(
            "every atlas feature must have a bundle assignment; missing="
            + ", ".join(missing[:10])
        )
    return {feature: normalized[feature] for feature in feature_names}


def cross_bundle_separation_metrics(
    corr: pd.DataFrame,
    bundle_by_feature: Mapping[str, Any],
    *,
    high_threshold: float = 0.80,
    extreme_threshold: float = 0.95,
) -> dict[str, float | int]:
    """Summarize absolute correlations between scores in different bundles."""

    if not 0.0 <= high_threshold <= 1.0:
        raise ValueError("high_threshold must lie in [0, 1]")
    if not 0.0 <= extreme_threshold <= 1.0:
        raise ValueError("extreme_threshold must lie in [0, 1]")
    if extreme_threshold < high_threshold:
        raise ValueError("extreme_threshold must be >= high_threshold")
    # Counts are irrelevant to this pure matrix statistic, but validate the
    # square matrix using a temporary aligned frame to avoid private Stage 1 APIs.
    if not isinstance(corr, pd.DataFrame) or corr.shape[0] != corr.shape[1] or corr.empty:
        raise ValueError("corr must be a non-empty square DataFrame")
    matrix = corr.copy()
    matrix.index = matrix.index.map(str)
    matrix.columns = matrix.columns.map(str)
    if set(matrix.index) != set(matrix.columns):
        raise ValueError("corr rows and columns must contain the same features")
    matrix = matrix.reindex(index=matrix.index, columns=matrix.index).apply(
        pd.to_numeric, errors="coerce"
    )
    values = matrix.to_numpy(dtype=float)
    if not np.allclose(values, values.T, equal_nan=True, atol=1e-10):
        raise ValueError("corr must be symmetric")
    bundles = _bundle_mapping(matrix.index, None, bundle_by_feature)

    candidate = 0
    observed: list[float] = []
    for left in range(len(matrix)):
        for right in range(left + 1, len(matrix)):
            left_name = matrix.index[left]
            right_name = matrix.index[right]
            if bundles[left_name] == bundles[right_name]:
                continue
            candidate += 1
            value = values[left, right]
            if np.isfinite(value):
                observed.append(abs(float(value)))
    if candidate == 0:
        raise ValueError("cross-bundle separation requires at least two bundles")
    if not observed:
        raise ValueError("no finite cross-bundle correlations are available")

    absolute = np.asarray(observed, dtype=float)
    high_count = int(np.sum(absolute >= high_threshold))
    extreme_count = int(np.sum(absolute >= extreme_threshold))
    return {
        "cross_bundle_candidate_pair_count": candidate,
        "cross_bundle_valid_pair_count": int(len(absolute)),
        "cross_bundle_missing_pair_count": int(candidate - len(absolute)),
        "cross_bundle_valid_pair_fraction": float(len(absolute) / candidate),
        "cross_bundle_median_abs_rho": float(np.median(absolute)),
        "cross_bundle_abs_rho_ge_0_80_count": high_count,
        "cross_bundle_abs_rho_ge_0_80_rate": float(high_count / len(absolute)),
        "cross_bundle_abs_rho_ge_0_95_count": extreme_count,
    }


def _score_frame(value: pd.DataFrame | pd.Series, name: str) -> pd.DataFrame:
    if isinstance(value, pd.Series):
        frame = value.rename(str(value.name or name)).to_frame()
    elif isinstance(value, pd.DataFrame):
        frame = value.copy()
    else:
        raise TypeError(f"{name} must be a pandas Series or DataFrame")
    if frame.empty or frame.shape[1] == 0:
        raise ValueError(f"{name} cannot be empty")
    if not frame.index.is_unique:
        raise ValueError(f"{name} index must be unique")
    if not frame.columns.is_unique:
        raise ValueError(f"{name} columns must be unique")
    frame.columns = frame.columns.map(str)
    if not frame.columns.is_unique:
        raise ValueError(f"{name} columns collide after string conversion")
    frame = frame.apply(pd.to_numeric, errors="coerce")
    return frame


def _priority_top_index(series: pd.Series, k: int, higher_is_priority: bool) -> set[Any]:
    sortable = pd.DataFrame(
        {
            "score": series,
            "tie_break": series.index.map(str),
        },
        index=series.index,
    )
    ordered = sortable.sort_values(
        ["score", "tie_break"],
        ascending=[not higher_is_priority, True],
        kind="mergesort",
    )
    return set(ordered.index[:k])


def stage1_overlap_metrics(
    stage2_scores: pd.DataFrame | pd.Series,
    stage1_scores: pd.DataFrame | pd.Series,
    *,
    top_fraction: float = 0.20,
    min_periods: int = 3,
    higher_is_priority: bool = True,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Compare Stage 2 score ranks with one or more frozen Stage 1 scores.

    Returned rows contain pairwise Spearman correlation, deterministic top-set
    overlap/Jaccard, and mean absolute percentile-rank displacement.  Index
    sets must match exactly so a regional join cannot silently drop units.
    """

    if not 0.0 < top_fraction < 1.0:
        raise ValueError("top_fraction must lie strictly between 0 and 1")
    if not isinstance(min_periods, (int, np.integer)) or int(min_periods) < 2:
        raise ValueError("min_periods must be an integer >= 2")
    left = _score_frame(stage2_scores, "stage2_score")
    right = _score_frame(stage1_scores, "stage1_score")
    if set(left.index) != set(right.index):
        only_stage2 = left.index.difference(right.index).tolist()
        only_stage1 = right.index.difference(left.index).tolist()
        raise ValueError(
            "Stage 1 and Stage 2 scores must have identical region indexes; "
            f"only_stage2={only_stage2[:5]}, only_stage1={only_stage1[:5]}"
        )
    right = right.reindex(left.index)

    rows: list[dict[str, Any]] = []
    for stage2_feature in left.columns:
        for stage1_feature in right.columns:
            pair = pd.concat(
                [left[stage2_feature], right[stage1_feature]], axis=1
            ).dropna()
            n = len(pair)
            rho = (
                float(pair.iloc[:, 0].corr(pair.iloc[:, 1], method="spearman"))
                if n >= int(min_periods)
                else float("nan")
            )
            if n:
                k = max(1, int(math.ceil(n * top_fraction)))
                stage2_top = _priority_top_index(
                    pair.iloc[:, 0], k, higher_is_priority
                )
                stage1_top = _priority_top_index(
                    pair.iloc[:, 1], k, higher_is_priority
                )
                intersection = len(stage2_top & stage1_top)
                union = len(stage2_top | stage1_top)
                rank_left = pair.iloc[:, 0].rank(
                    method="average", pct=True, ascending=not higher_is_priority
                )
                rank_right = pair.iloc[:, 1].rank(
                    method="average", pct=True, ascending=not higher_is_priority
                )
                mean_rank_shift = float((rank_left - rank_right).abs().mean())
                overlap_rate = float(intersection / k)
                jaccard = float(intersection / union)
            else:
                k = 0
                intersection = 0
                mean_rank_shift = float("nan")
                overlap_rate = float("nan")
                jaccard = float("nan")
            rows.append(
                {
                    "stage2_feature": stage2_feature,
                    "stage1_feature": stage1_feature,
                    "pairwise_n": n,
                    "spearman_rho": rho,
                    "abs_spearman_rho": abs(rho) if np.isfinite(rho) else np.nan,
                    "top_fraction": float(top_fraction),
                    "top_k": k,
                    "top_overlap_count": intersection,
                    "top_overlap_rate": overlap_rate,
                    "top_jaccard": jaccard,
                    "mean_abs_percentile_rank_shift": mean_rank_shift,
                }
            )
    detail = pd.DataFrame(rows)

    def _finite_stat(column: str, operation: str) -> float:
        values = pd.to_numeric(detail[column], errors="coerce").dropna()
        if values.empty:
            return float("nan")
        return float(values.median() if operation == "median" else values.max())

    summary: dict[str, float | int] = {
        "stage1_overlap_pair_count": int(len(detail)),
        "stage1_overlap_valid_rho_count": int(detail["spearman_rho"].notna().sum()),
        "stage1_overlap_median_abs_rho": _finite_stat("abs_spearman_rho", "median"),
        "stage1_overlap_max_abs_rho": _finite_stat("abs_spearman_rho", "max"),
        "stage1_overlap_median_top_overlap_rate": _finite_stat(
            "top_overlap_rate", "median"
        ),
        "stage1_overlap_max_top_overlap_rate": _finite_stat(
            "top_overlap_rate", "max"
        ),
        "stage1_overlap_median_top_jaccard": _finite_stat("top_jaccard", "median"),
        "stage1_overlap_median_rank_shift": _finite_stat(
            "mean_abs_percentile_rank_shift", "median"
        ),
    }
    return detail, summary


def build_specialty_bundle_atlas(
    score_frame: pd.DataFrame,
    manifest: ManifestLike,
    output_dir: str | Path,
    *,
    bundle_by_feature: Mapping[str, Any] | None = None,
    stage1_scores: pd.DataFrame | pd.Series | None = None,
    min_periods: int = 3,
    top_fraction: float = 0.20,
    include_spectral: bool = False,
    stem: str = "stage2_specialty_bundle_atlas",
    title: str = "MEDIROAD Stage 2 Specialty / Bundle Score Atlas",
    annotate: bool | None = None,
) -> SpecialtyBundleAtlasResult:
    """Build, diagnose, and render a same-granularity Stage 2 score atlas."""

    precluster_features = [str(column) for column in score_frame.columns]
    stable_manifest = stabilize_feature_manifest(manifest, precluster_features)
    corr, pairwise_n = same_granularity_correlation(
        score_frame, stable_manifest, min_periods=min_periods
    )
    meta = _manifest_table(stable_manifest, corr.index)
    granularity = str(corr.attrs["stage2_granularity"])
    bundles = _bundle_mapping(corr.index, meta, bundle_by_feature)
    # In this atlas the service bundle is the semantic separation target.
    # Preserve an upstream axis for audit, while making the Stage 1 metadata
    # ribbon and its block-quality metrics use the actual bundle assignment.
    if "axis" in meta.columns:
        meta["source_axis"] = meta["axis"]
    meta["axis"] = [bundles[feature] for feature in meta.index]
    separation = cross_bundle_separation_metrics(corr, bundles)
    quality = safe_correlation_quality_metrics(
        corr,
        pairwise_n,
        meta,
        include_spectral=include_spectral,
    )

    overlap_detail: pd.DataFrame | None = None
    overlap_summary: dict[str, float | int] | None = None
    if stage1_scores is not None:
        overlap_detail, overlap_summary = stage1_overlap_metrics(
            score_frame,
            stage1_scores,
            top_fraction=top_fraction,
            min_periods=min_periods,
        )

    # Contiguous bundle blocks expose within-versus-cross-bundle structure at
    # a glance.  Features remain hierarchically clustered inside each block;
    # bundle order follows the preregistered manifest/score order.
    bundle_order = list(dict.fromkeys(bundles[feature] for feature in corr.index))
    order: list[str] = []
    for bundle in bundle_order:
        members = [feature for feature in corr.index if bundles[feature] == bundle]
        order.extend(hierarchical_order(corr.loc[members, members]))
    ordered_corr = corr.loc[order, order].copy()
    ordered_n = pairwise_n.loc[order, order].copy()
    ordered_corr.attrs.update(corr.attrs)
    ordered_n.attrs.update(pairwise_n.attrs)
    high_correlation_ledger = high_correlation_equivalence_ledger(
        ordered_corr,
        stable_manifest,
        threshold=0.98,
    )

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    static_paths = plot_correlation_map(
        ordered_corr,
        ordered_n,
        meta.reset_index(drop=True),
        destination / stem,
        title,
        annotate=len(order) <= 25 if annotate is None else bool(annotate),
        mask_n=int(min_periods),
    )
    html_path = save_interactive_heatmap(
        ordered_corr,
        ordered_n,
        meta.reset_index(drop=True),
        destination / f"{stem}.html",
        title,
    )
    corr_path = destination / f"{stem}_correlation.csv"
    n_path = destination / f"{stem}_pairwise_n.csv"
    manifest_path = destination / f"{stem}_manifest.csv"
    ledger_path = destination / f"{stem}_high_correlation_equivalence_ledger.csv"
    ordered_corr.to_csv(corr_path, encoding="utf-8-sig")
    ordered_n.to_csv(n_path, encoding="utf-8-sig")
    stable_manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")
    high_correlation_ledger.to_csv(ledger_path, index=False, encoding="utf-8-sig")
    artifacts = {
        "png": static_paths[0],
        "svg": static_paths[1],
        "html": html_path,
        "correlation_csv": str(corr_path),
        "pairwise_n_csv": str(n_path),
        "manifest_csv": str(manifest_path),
        "high_correlation_ledger_csv": str(ledger_path),
    }
    return SpecialtyBundleAtlasResult(
        correlation=ordered_corr,
        pairwise_n=ordered_n,
        ordered_features=tuple(order),
        granularity=granularity,
        separation_metrics=separation,
        stage1_overlap=overlap_detail,
        stage1_overlap_summary=overlap_summary,
        quality_metrics=quality,
        manifest=stable_manifest,
        high_correlation_ledger=high_correlation_ledger,
        artifacts=artifacts,
    )


def _coerce_month_profile(
    month_profile: pd.DataFrame,
    month_column: str | None,
) -> pd.DataFrame:
    if not isinstance(month_profile, pd.DataFrame) or month_profile.empty:
        raise ValueError("month_profile must be a non-empty DataFrame")
    frame = month_profile.copy()
    resolved_month_column = month_column
    if resolved_month_column is None and "month" in frame.columns:
        resolved_month_column = "month"
    if resolved_month_column is not None:
        if resolved_month_column not in frame.columns:
            raise KeyError(f"month column not found: {resolved_month_column}")
        months = frame.pop(resolved_month_column)
    else:
        months = pd.Series(frame.index, index=frame.index)
    numeric_months = pd.to_numeric(months, errors="coerce")
    if numeric_months.isna().any() or not np.allclose(
        numeric_months, np.rint(numeric_months)
    ):
        raise ValueError("month values must be integers 1 through 12")
    month_numbers = numeric_months.astype(int).to_numpy()
    if len(np.unique(month_numbers)) != len(month_numbers):
        raise ValueError("month values must be unique")
    if set(month_numbers.tolist()) != set(range(1, 13)):
        raise ValueError("a month-profile heatmap requires exactly months 1 through 12")
    if frame.shape[1] == 0 or not frame.columns.is_unique:
        raise ValueError("month_profile requires unique profile columns")
    frame.columns = frame.columns.map(str)
    if not frame.columns.is_unique:
        raise ValueError("month profile names collide after string conversion")
    frame = frame.apply(pd.to_numeric, errors="coerce")
    if frame.notna().sum().eq(0).any():
        missing = frame.columns[frame.notna().sum().eq(0)].tolist()
        raise ValueError("month profiles contain no numeric values: " + ", ".join(missing))
    frame.index = pd.Index(month_numbers, name="month")
    return frame.sort_index()


def render_temporal_month_profile_heatmap(
    month_profile: pd.DataFrame,
    manifest: ManifestLike,
    output_dir: str | Path,
    *,
    month_column: str | None = None,
    center: float = 1.0,
    stem: str = "stage2_temporal_month_profile",
    title: str = "MEDIROAD Stage 2 Common Specialty-Month Profile",
    annotate: bool = True,
) -> TemporalMonthProfileResult:
    """Render a validated 12-month specialty/bundle profile as PNG/SVG/HTML.

    Rows in the figure are specialties or service bundles and columns are
    months.  The title/subtitle explicitly describes the layer as a common or
    coarse profile; this plot must not be interpreted as local monthly demand.
    """

    profile = _coerce_month_profile(month_profile, month_column)
    meta = _manifest_table(manifest, profile.columns)
    granularity = _single_granularity(meta)
    if not np.isfinite(float(center)):
        raise ValueError("center must be finite")
    values = profile.T.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if not finite.size:
        raise ValueError("month_profile has no finite values")
    maximum_deviation = max(float(np.max(np.abs(finite - center))), 1e-6)
    vmin = float(center - maximum_deviation)
    vmax = float(center + maximum_deviation)

    label_field = _first_field(meta, _LABEL_FIELDS)
    labels = [
        str(meta.loc[feature, label_field])
        if label_field is not None and not pd.isna(meta.loc[feature, label_field])
        else feature
        for feature in profile.columns
    ]
    row_labels = [
        f"{feature} | {label}" if label != feature else feature
        for feature, label in zip(profile.columns, labels)
    ]
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    base = destination / stem
    png_path = base.with_suffix(".png")
    svg_path = base.with_suffix(".svg")
    html_path = base.with_suffix(".html")

    try:
        import matplotlib as mpl

        mpl.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - dependency failure path
        raise ImportError("temporal month-profile plotting requires matplotlib") from exc
    korean_font = _matplotlib_korean_font()

    width = max(10.0, 0.72 * 12 + 3.0)
    height = max(4.5, min(22.0, 0.52 * len(profile.columns) + 2.8))
    with mpl.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [korean_font, "DejaVu Sans"],
            "axes.unicode_minus": False,
        }
    ):
        figure, axis = plt.subplots(figsize=(width, height), constrained_layout=True)
        colour_map = mpl.colormaps["RdBu_r"].with_extremes(bad="#BDBDBD")
        image = axis.imshow(
            np.ma.masked_invalid(values),
            cmap=colour_map,
            vmin=vmin,
            vmax=vmax,
            aspect="auto",
            interpolation="nearest",
        )
        axis.set_xticks(np.arange(12), [str(month) for month in range(1, 13)])
        axis.set_yticks(np.arange(len(row_labels)), row_labels)
        axis.set_xlabel("Month")
        axis.set_ylabel("Specialty / recommended service bundle")
        axis.set_title(title, fontsize=13, fontweight="bold", pad=18)
        axis.text(
            0.5,
            1.01,
            "Common/coarse temporal evidence; not a region-specific monthly forecast",
            transform=axis.transAxes,
            ha="center",
            va="bottom",
            fontsize=8,
            color="#555555",
        )
        colour_bar = figure.colorbar(image, ax=axis, fraction=0.035, pad=0.02)
        colour_bar.set_label(f"Profile value (diverging around {center:g})")
        if annotate and len(row_labels) <= 30:
            for row in range(values.shape[0]):
                for column in range(values.shape[1]):
                    value = values[row, column]
                    if not np.isfinite(value):
                        continue
                    contrast = abs(value - center) / maximum_deviation
                    axis.text(
                        column,
                        row,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        fontsize=6.5,
                        color="white" if contrast >= 0.55 else "#222222",
                    )
        figure.savefig(png_path, dpi=300, facecolor="white", bbox_inches="tight")
        figure.savefig(svg_path, dpi=300, facecolor="white", bbox_inches="tight")
        plt.close(figure)

    try:
        import plotly.graph_objects as go
    except ImportError as exc:  # pragma: no cover - dependency failure path
        raise ImportError("temporal month-profile plotting requires plotly") from exc

    interactive = go.Figure(
        data=go.Heatmap(
            z=values,
            x=list(range(1, 13)),
            y=row_labels,
            zmin=vmin,
            zmax=vmax,
            zmid=float(center),
            colorscale=[
                [0.0, "#2166AC"],
                [0.5, "#F7F7F7"],
                [1.0, "#B2182B"],
            ],
            colorbar={"title": "Profile"},
            hoverongaps=False,
            hovertemplate=(
                "Profile: %{y}<br>Month: %{x}<br>Value: %{z:.4f}<extra></extra>"
            ),
        )
    )
    interactive.update_layout(
        title={
            "text": (
                f"{title}<br><sup>Common/coarse temporal evidence; "
                "not a region-specific monthly forecast</sup>"
            ),
            "x": 0.5,
        },
        width=1200,
        height=min(1600, max(600, 38 * len(row_labels) + 220)),
        xaxis={"title": "Month", "dtick": 1},
        yaxis={"title": "Specialty / recommended service bundle", "autorange": "reversed"},
        paper_bgcolor="white",
        plot_bgcolor="#BDBDBD",
        margin={"l": 220, "r": 100, "t": 100, "b": 80},
    )
    interactive.write_html(
        str(html_path), include_plotlyjs=True, full_html=True, auto_open=False
    )
    return TemporalMonthProfileResult(
        month_profile=profile,
        granularity=granularity,
        artifacts={
            "png": str(png_path),
            "svg": str(svg_path),
            "html": str(html_path),
        },
    )


# Concise aliases for callers that use noun-style API names.
specialty_bundle_score_atlas = build_specialty_bundle_atlas
temporal_month_profile_heatmap = render_temporal_month_profile_heatmap


__all__ = [
    "MixedSampleSizeError",
    "SpecialtyBundleAtlasResult",
    "TemporalMonthProfileResult",
    "stabilize_feature_manifest",
    "high_correlation_equivalence_ledger",
    "same_granularity_correlation",
    "assert_spectral_metrics_allowed",
    "spectral_correlation_metrics",
    "safe_correlation_quality_metrics",
    "cross_bundle_separation_metrics",
    "stage1_overlap_metrics",
    "build_specialty_bundle_atlas",
    "specialty_bundle_score_atlas",
    "render_temporal_month_profile_heatmap",
    "temporal_month_profile_heatmap",
]
