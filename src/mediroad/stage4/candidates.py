from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .types import Stage3Paths
from .utils import coalesce_column, normalize_0_1, read_table, robust_percentile


def canonicalize_interface(stage3: Stage3Paths, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    aliases = config["column_aliases"]
    raw = read_table(stage3.interface)
    mapping: dict[str, str | None] = {}
    for key in [
        "venue_id", "venue_name", "admin_code", "admin_name", "sigungu", "bundle_id", "cluster_id",
        "shortlist_rank", "pareto_tier", "travel_minutes", "known_overlap",
    ]:
        mapping[key] = coalesce_column(raw, aliases.get(key, [key]), required=key in {"venue_id", "admin_code", "sigungu", "bundle_id"})

    rename = {source: target for target, source in mapping.items() if source is not None}
    df = raw.rename(columns=rename).copy()
    df["venue_id"] = df["venue_id"].astype(str)
    df["admin_code"] = df["admin_code"].astype(str)
    df["sigungu"] = df["sigungu"].astype(str)
    df["bundle_id"] = df["bundle_id"].astype(str)
    if "venue_name" not in df:
        df["venue_name"] = df["venue_id"]
    if "cluster_id" not in df:
        df["cluster_id"] = np.nan
    if "shortlist_rank" not in df:
        df["shortlist_rank"] = np.nan
    if "pareto_tier" not in df:
        df["pareto_tier"] = np.nan

    # Stage 3 may keep shortlist and coverage-cluster contracts in separate artifacts.
    # Merge those contracts rather than silently degrading every venue to a singleton.
    if stage3.coverage_clusters is not None and stage3.coverage_clusters.exists():
        cluster_src = read_table(stage3.coverage_clusters)
        cluster_venue = coalesce_column(cluster_src, aliases["venue_id"])
        cluster_id_col = coalesce_column(
            cluster_src, aliases["cluster_id"] + ["coverage_cluster_id", "coverage_equivalent_cluster_id"]
        )
        cluster_src = cluster_src[[cluster_venue, cluster_id_col]].rename(
            columns={cluster_venue: "venue_id", cluster_id_col: "cluster_id_external"}
        )
        cluster_src["venue_id"] = cluster_src["venue_id"].astype(str)
        cluster_src = cluster_src.drop_duplicates("venue_id")
        df = df.merge(cluster_src, on="venue_id", how="left", validate="many_to_one")
        df["cluster_id"] = df["cluster_id"].fillna(df["cluster_id_external"])
        df = df.drop(columns=["cluster_id_external"])

    if stage3.shortlist is not None and stage3.shortlist.exists():
        shortlist_src = read_table(stage3.shortlist)
        shortlist_venue = coalesce_column(shortlist_src, aliases["venue_id"])
        keep = [shortlist_venue]
        rename_short = {shortlist_venue: "venue_id"}
        for canonical, candidates in [
            ("shortlist_rank", aliases["shortlist_rank"]),
            ("pareto_tier", aliases["pareto_tier"]),
        ]:
            source = coalesce_column(shortlist_src, candidates, required=False)
            if source is not None:
                keep.append(source)
                rename_short[source] = f"{canonical}_external"
        for optional in ["fallback_venue_1", "fallback_venue_2", "field_validation_required"]:
            if optional in shortlist_src.columns:
                keep.append(optional)
                rename_short[optional] = f"{optional}_external"
        shortlist_src = shortlist_src[keep].rename(columns=rename_short)
        shortlist_src["venue_id"] = shortlist_src["venue_id"].astype(str)
        shortlist_src = shortlist_src.drop_duplicates("venue_id")
        df = df.merge(shortlist_src, on="venue_id", how="left", validate="many_to_one")
        for canonical in ["shortlist_rank", "pareto_tier", "fallback_venue_1", "fallback_venue_2", "field_validation_required"]:
            external = f"{canonical}_external"
            if external in df.columns:
                if canonical not in df.columns:
                    df[canonical] = df[external]
                else:
                    df[canonical] = df[canonical].fillna(df[external])
                df = df.drop(columns=[external])

    df["cluster_id"] = df["cluster_id"].fillna(df["venue_id"]).astype(str)
    if "travel_minutes" not in df:
        df["travel_minutes"] = np.nan
    if "known_overlap" not in df:
        df["known_overlap"] = "unknown"

    venue_cols = [
        "venue_id", "venue_name", "admin_code", "sigungu", "cluster_id", "shortlist_rank", "pareto_tier",
        "travel_minutes", "known_overlap",
    ]
    optional_meta = [
        "admin_name", "venue_type", "latitude", "longitude", "field_validation_required",
        "venue_readiness_prior", "transit_context", "transit_context_score", "road_context",
        "raw_elderly_exposure", "need_weighted_exposure", "high_need_elderly_exposure",
        "own_admin_exposure", "cross_admin_exposure", "cross_sigungu_exposure",
        "overlap_weighted_degree", "fallback_venue_1", "fallback_venue_2",
        "temporal_release_type", "temporal_confidence", "recommended_season", "fallback_season",
    ]
    for col in optional_meta:
        if col in df.columns and col not in venue_cols:
            venue_cols.append(col)
    venue = df[venue_cols].sort_values(["venue_id"]).drop_duplicates("venue_id", keep="first").reset_index(drop=True)

    bundle_value_candidates = config["bundle_assignment"]["value_column_priority"]
    value_col = next((c for c in bundle_value_candidates if c in df.columns), None)
    if value_col is None:
        raise KeyError(
            "Stage 3 interface lacks bundle value. Expected one of " + ", ".join(bundle_value_candidates)
        )
    bundle = df[["venue_id", "bundle_id", value_col]].copy()
    bundle = bundle.rename(columns={value_col: "bundle_value"})
    bundle["bundle_value"] = pd.to_numeric(bundle["bundle_value"], errors="coerce").fillna(0.0)
    if "specialty_gap" in df.columns:
        extra = df[["venue_id", "bundle_id", "specialty_gap"]].copy()
        bundle = bundle.merge(extra, on=["venue_id", "bundle_id"], how="left", validate="one_to_one")
    if bundle.duplicated(["venue_id", "bundle_id"]).any():
        raise ValueError("Stage 3 interface has duplicate venue_id × bundle_id keys")
    return venue, bundle


def add_candidate_scores(venue: pd.DataFrame) -> pd.DataFrame:
    out = venue.copy()
    for col in [
        "raw_elderly_exposure", "need_weighted_exposure", "high_need_elderly_exposure",
        "venue_readiness_prior", "transit_context_score", "overlap_weighted_degree",
    ]:
        if col not in out:
            out[col] = np.nan
        out[f"pct__{col}"] = robust_percentile(out[col])
    out["candidate_priority_tiebreak"] = (
        0.40 * out["pct__need_weighted_exposure"]
        + 0.25 * out["pct__raw_elderly_exposure"]
        + 0.15 * out["pct__high_need_elderly_exposure"]
        + 0.10 * out["pct__venue_readiness_prior"]
        + 0.10 * out["pct__transit_context_score"]
    )
    return out


def reduce_candidates(
    venue: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = config["candidate_reduction"]
    top_n = int(cfg["top_n_per_admin"])
    max_candidates = int(cfg["max_candidates"])
    pareto_max = int(cfg["pareto_tier_max"])

    work = add_candidate_scores(venue)
    rank = pd.to_numeric(work["shortlist_rank"], errors="coerce")
    pareto = pd.to_numeric(work["pareto_tier"], errors="coerce")
    eligible = work[(rank.le(top_n) | rank.isna()) & (pareto.le(pareto_max) | pareto.isna())].copy()

    if eligible.empty or eligible["admin_code"].nunique() < work["admin_code"].nunique():
        fallback = (
            work.sort_values(["admin_code", "candidate_priority_tiebreak", "venue_id"], ascending=[True, False, True])
            .groupby("admin_code", as_index=False)
            .head(top_n)
        )
        eligible = pd.concat([eligible, fallback], ignore_index=True).drop_duplicates("venue_id")

    eligible = eligible.sort_values(
        ["admin_code", "shortlist_rank", "pareto_tier", "candidate_priority_tiebreak", "venue_id"],
        ascending=[True, True, True, False, True],
        na_position="last",
    )
    eligible["candidate_rank_within_admin_stage4"] = eligible.groupby("admin_code").cumcount() + 1
    eligible = eligible[eligible["candidate_rank_within_admin_stage4"].le(top_n)].copy()

    if len(eligible) > max_candidates:
        required = eligible.groupby("admin_code", as_index=False).head(1)
        remaining = eligible[~eligible["venue_id"].isin(required["venue_id"])].sort_values(
            ["candidate_priority_tiebreak", "venue_id"], ascending=[False, True]
        )
        slots = max(0, max_candidates - len(required))
        eligible = pd.concat([required, remaining.head(slots)], ignore_index=True)

    if bool(cfg["require_all_admins"]):
        missing = sorted(set(work["admin_code"]) - set(eligible["admin_code"]))
        if missing:
            raise RuntimeError(f"Candidate reduction dropped {len(missing)} admin dongs: {missing[:10]}")

    eligible = eligible.sort_values(["admin_code", "candidate_rank_within_admin_stage4", "venue_id"]).reset_index(drop=True)
    excluded = work[~work["venue_id"].isin(eligible["venue_id"])].copy()
    excluded["stage4_exclusion_reason"] = "outside_provisional_top_n_or_candidate_cap"
    return eligible, excluded


def encode_known_overlap(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.lower().str.strip()
    known_true = text.isin(["known", "yes", "true", "1", "confirmed", "high"])
    known_false = text.isin(["no", "false", "0", "none", "low", "not_known_overlap"])
    return pd.Series(np.where(known_true, 1, np.where(known_false, 0, -1)), index=series.index, dtype=int)

