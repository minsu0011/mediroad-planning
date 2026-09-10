"""Resolve Stage 4 spatial anchors to verified physical venues.

BUSPROXY records describe useful catchment anchors, not verified buildings.
The routines here preserve that distinction and only emit a final physical
venue when a same-cluster, non-BUSPROXY candidate passes the field contract.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

from .field_validation import (
    FieldValidationContractError,
    validate_field_validation_schema,
)


class BusProxyResolutionError(FieldValidationContractError):
    """Raised when a BUSPROXY/physical-venue boundary would be violated."""


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise BusProxyResolutionError(f"{label} is missing required columns: {missing}")


def _unique_ids(frame: pd.DataFrame, label: str) -> pd.Series:
    _require_columns(frame, ["venue_id"], label)
    if frame["venue_id"].isna().any():
        raise BusProxyResolutionError(f"{label}.venue_id contains null values")
    ids = frame["venue_id"].astype("string").str.strip()
    if ids.eq("").any():
        raise BusProxyResolutionError(f"{label}.venue_id contains blank values")
    duplicates = ids[ids.duplicated(keep=False)].unique().tolist()
    if duplicates:
        raise BusProxyResolutionError(
            f"{label}.venue_id must be unique; duplicate examples: {duplicates[:5]}"
        )
    return ids


def is_busproxy_id(value: Any) -> bool:
    """Return True only for the canonical BUSPROXY identifier namespace."""

    return isinstance(value, str) and value.strip().upper().startswith("BUSPROXY_")


def _normalise_inputs(
    anchors: pd.DataFrame,
    candidates: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    anchor_ids = _unique_ids(anchors, "anchors")
    candidate_ids = _unique_ids(candidates, "candidates")
    _require_columns(anchors, ["cluster_id"], "anchors")
    _require_columns(candidates, ["cluster_id"], "candidates")
    anchor_frame = anchors.copy()
    candidate_frame = candidates.copy()
    anchor_frame["venue_id"] = anchor_ids
    candidate_frame["venue_id"] = candidate_ids
    for label, frame in (("anchors", anchor_frame), ("candidates", candidate_frame)):
        frame["cluster_id"] = frame["cluster_id"].astype("string").str.strip()
        if frame["cluster_id"].isna().any() or frame["cluster_id"].eq("").any():
            raise BusProxyResolutionError(f"{label}.cluster_id must be complete")

    indexed = candidate_frame.set_index("venue_id", drop=False)
    missing = sorted(set(anchor_ids) - set(candidate_ids))
    if missing:
        raise BusProxyResolutionError(f"anchors are absent from candidates: {missing[:5]}")
    for row in anchor_frame.itertuples(index=False):
        canonical_cluster = str(indexed.loc[str(row.venue_id), "cluster_id"])
        if str(row.cluster_id) != canonical_cluster:
            raise BusProxyResolutionError(
                f"anchor {row.venue_id!r} cluster {row.cluster_id!r} disagrees with "
                f"candidate cluster {canonical_cluster!r}"
            )
    candidate_frame["is_busproxy_anchor"] = candidate_frame["venue_id"].map(is_busproxy_id)
    return anchor_frame, candidate_frame


def _ranked_cluster_physical_candidates(
    cluster_id: str,
    candidates: pd.DataFrame,
    *,
    exclude: set[str],
) -> list[str]:
    pool = candidates[
        candidates["cluster_id"].eq(cluster_id)
        & ~candidates["is_busproxy_anchor"]
        & ~candidates["venue_id"].isin(exclude)
    ].copy()
    for column in ("venue_readiness_prior", "raw_elderly_exposure", "need_weighted_exposure"):
        if column not in pool.columns:
            pool[column] = np.nan
        pool[column] = pd.to_numeric(pool[column], errors="coerce").fillna(float("-inf"))
    pool = pool.sort_values(
        ["venue_readiness_prior", "raw_elderly_exposure", "need_weighted_exposure", "venue_id"],
        ascending=[False, False, False, True],
        kind="stable",
    )
    return pool["venue_id"].tolist()


def actual_facility_fallbacks(
    anchors: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    max_fallbacks: int = 2,
) -> pd.DataFrame:
    """Attach one or two same-cluster, non-BUSPROXY fallback candidates.

    Existing Stage 3 fallback references are honoured first when they satisfy
    the physical-facility and cluster constraints.  Invalid BUSPROXY references
    are ignored; references absent from the supplied universe fail closed so a
    truncated candidate table cannot masquerade as a complete fallback audit.
    """

    if int(max_fallbacks) not in {1, 2}:
        raise BusProxyResolutionError("max_fallbacks must be 1 or 2")
    anchor_frame, candidate_frame = _normalise_inputs(anchors, candidates)
    indexed = candidate_frame.set_index("venue_id", drop=False)
    records: list[dict[str, Any]] = []

    for anchor in anchor_frame.to_dict("records"):
        anchor_id = str(anchor["venue_id"])
        cluster_id = str(anchor["cluster_id"])
        canonical = indexed.loc[anchor_id]
        preferred: list[str] = []
        rejected: list[str] = []
        for column in ("fallback_venue_1", "fallback_venue_2"):
            for source in (anchor, canonical):
                if column not in source or pd.isna(source[column]):
                    continue
                fallback_id = str(source[column]).strip()
                if not fallback_id or fallback_id == anchor_id:
                    continue
                if fallback_id not in indexed.index:
                    raise BusProxyResolutionError(
                        f"fallback {fallback_id!r} is absent from the supplied candidate universe"
                    )
                fallback = indexed.loc[fallback_id]
                if str(fallback["cluster_id"]) != cluster_id:
                    rejected.append(f"{fallback_id}:OUTSIDE_COVERAGE_CLUSTER")
                    continue
                if is_busproxy_id(fallback_id):
                    rejected.append(f"{fallback_id}:BUSPROXY_SPATIAL_ANCHOR_ONLY")
                    continue
                if fallback_id not in preferred:
                    preferred.append(fallback_id)

        for fallback_id in _ranked_cluster_physical_candidates(
            cluster_id, candidate_frame, exclude={anchor_id}
        ):
            if fallback_id not in preferred:
                preferred.append(fallback_id)
        chosen = preferred[: int(max_fallbacks)]
        record: dict[str, Any] = {
            "venue_id": anchor_id,
            "cluster_id": cluster_id,
            "spatial_anchor_is_busproxy": is_busproxy_id(anchor_id),
            "actual_facility_fallback_count": len(chosen),
            "fallback_resolution_status": (
                "READY" if len(chosen) == int(max_fallbacks) else "PARTIAL" if chosen else "NONE"
            ),
            "rejected_fallback_reference_count": len(set(rejected)),
            "rejected_fallback_references": "|".join(sorted(set(rejected))),
        }
        for index in range(1, int(max_fallbacks) + 1):
            record[f"actual_facility_fallback_{index}"] = (
                chosen[index - 1] if len(chosen) >= index else pd.NA
            )
        records.append(record)

    result = pd.DataFrame(records)
    for column in [f"actual_facility_fallback_{i}" for i in range(1, int(max_fallbacks) + 1)]:
        values = result[column].dropna().astype(str)
        if values.map(is_busproxy_id).any():
            raise BusProxyResolutionError("BUSPROXY leaked into an actual-facility fallback slot")
    return result


def resolve_final_physical_venues(
    anchors: pd.DataFrame,
    candidates: pd.DataFrame,
    field_evidence: pd.DataFrame,
    *,
    max_fallbacks: int = 2,
) -> pd.DataFrame:
    """Resolve anchors, leaving ``venue_unresolved`` when proof is absent.

    A BUSPROXY can never be returned in ``final_physical_venue_id``.  A direct
    venue or fallback must be in the same coverage cluster and have a derived
    ``field_verified == True`` under the 12-field contract.  Missing field rows
    are treated as unverified rather than guessed.
    """

    anchor_frame, candidate_frame = _normalise_inputs(anchors, candidates)
    fallback_table = actual_facility_fallbacks(
        anchor_frame, candidate_frame, max_fallbacks=max_fallbacks
    )
    evidence = validate_field_validation_schema(field_evidence)
    unknown_evidence = sorted(set(evidence["venue_id"]) - set(candidate_frame["venue_id"]))
    if unknown_evidence:
        raise BusProxyResolutionError(
            f"field evidence references venues outside candidates: {unknown_evidence[:5]}"
        )
    evidence_lookup = evidence.set_index("venue_id", drop=False)
    candidate_lookup = candidate_frame.set_index("venue_id", drop=False)
    fallback_lookup = fallback_table.set_index("venue_id", drop=False)
    records: list[dict[str, Any]] = []

    def verified(candidate_id: str) -> bool:
        return (
            candidate_id in evidence_lookup.index
            and bool(evidence_lookup.loc[candidate_id, "field_verified"])
        )

    for anchor in anchor_frame.to_dict("records"):
        anchor_id = str(anchor["venue_id"])
        cluster_id = str(anchor["cluster_id"])
        anchor_is_busproxy = is_busproxy_id(anchor_id)
        ordered: list[str] = []
        if not anchor_is_busproxy:
            ordered.append(anchor_id)
        fallback_row = fallback_lookup.loc[anchor_id]
        for index in range(1, int(max_fallbacks) + 1):
            value = fallback_row[f"actual_facility_fallback_{index}"]
            if not pd.isna(value) and str(value) not in ordered:
                ordered.append(str(value))
        final_id = next((candidate_id for candidate_id in ordered if verified(candidate_id)), None)
        unresolved = final_id is None
        if final_id is not None:
            final_row = candidate_lookup.loc[final_id]
            if is_busproxy_id(final_id):
                raise BusProxyResolutionError("BUSPROXY cannot be a final physical venue")
            if str(final_row["cluster_id"]) != cluster_id:
                raise BusProxyResolutionError("final physical venue escaped the anchor coverage cluster")
            resolution_status = (
                "RESOLVED_DIRECT_VERIFIED_PHYSICAL_VENUE"
                if final_id == anchor_id
                else "RESOLVED_VERIFIED_CLUSTER_FALLBACK"
            )
            final_name = final_row.get("venue_name", pd.NA)
            final_type = final_row.get("venue_type", pd.NA)
        else:
            resolution_status = (
                "VENUE_UNRESOLVED_BUSPROXY_SPATIAL_ANCHOR_ONLY"
                if anchor_is_busproxy
                else "VENUE_UNRESOLVED_NO_VERIFIED_PHYSICAL_VENUE"
            )
            final_name = pd.NA
            final_type = pd.NA

        record = dict(anchor)
        record.update(
            {
                "spatial_anchor_venue_id": anchor_id,
                "spatial_anchor_is_busproxy": anchor_is_busproxy,
                "location_selected": True,
                "final_physical_venue_id": final_id if final_id is not None else pd.NA,
                "final_physical_venue_name": final_name,
                "final_physical_venue_type": final_type,
                "final_physical_venue_field_verified": final_id is not None,
                "venue_unresolved": unresolved,
                "resolution_status": resolution_status,
            }
        )
        for index in range(1, int(max_fallbacks) + 1):
            record[f"actual_facility_fallback_{index}"] = fallback_row[
                f"actual_facility_fallback_{index}"
            ]
        records.append(record)

    result = pd.DataFrame(records)
    resolved = result[~result["venue_unresolved"]]
    if resolved["final_physical_venue_id"].astype(str).map(is_busproxy_id).any():
        raise BusProxyResolutionError("resolved output contains a BUSPROXY final venue")
    if not resolved["final_physical_venue_field_verified"].all():
        raise BusProxyResolutionError("resolved output contains an unverified physical venue")
    return result


__all__ = [
    "BusProxyResolutionError",
    "actual_facility_fallbacks",
    "is_busproxy_id",
    "resolve_final_physical_venues",
]
