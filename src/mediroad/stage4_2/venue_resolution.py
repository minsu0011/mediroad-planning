from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .errors import ContractError
from .utils import normalize_bool


@dataclass
class ResolutionResult:
    resolution: pd.DataFrame
    unresolved: pd.DataFrame
    summary: dict[str, Any]


def resolve_physical_venues(
    anchor_interface: pd.DataFrame,
    field_results: pd.DataFrame,
    venue_catalog: pd.DataFrame,
) -> ResolutionResult:
    required = [
        "venue_id",
        "cluster_id",
        "spatial_anchor_venue_id",
        "spatial_anchor_is_busproxy",
        "actual_facility_fallback_1",
        "actual_facility_fallback_2",
    ]
    missing = [c for c in required if c not in anchor_interface]
    if missing:
        raise ContractError(f"Anchor interface missing resolution columns: {missing}")
    if "field_verified" not in field_results:
        raise ContractError("Field results missing field_verified")
    verified = dict(
        zip(
            field_results["venue_id"].astype(str),
            field_results["field_verified"].map(normalize_bool).fillna(False).astype(bool),
        )
    )
    field_cluster = dict(zip(field_results["venue_id"].astype(str), field_results["cluster_id"].astype(str)))
    catalog = venue_catalog.drop_duplicates("venue_id").copy()
    catalog["venue_id"] = catalog["venue_id"].astype(str)
    catalog_by_id = catalog.set_index("venue_id", drop=False)

    rows: list[dict[str, Any]] = []
    for _, anchor in anchor_interface.iterrows():
        anchor_id = str(anchor["spatial_anchor_venue_id"])
        cluster = str(anchor["cluster_id"])
        parsed_bus = normalize_bool(anchor["spatial_anchor_is_busproxy"])
        is_bus = bool(parsed_bus) if parsed_bus is not None else anchor_id.startswith("BUSPROXY_")
        fallbacks = [anchor.get("actual_facility_fallback_1"), anchor.get("actual_facility_fallback_2")]
        fallbacks = [str(x) for x in fallbacks if pd.notna(x) and str(x).strip()]
        candidates: list[tuple[str, str]] = []
        if not is_bus:
            candidates.append((anchor_id, "DIRECT_NON_BUS_ANCHOR"))
        candidates.extend((v, f"DECLARED_FALLBACK_{idx+1}") for idx, v in enumerate(fallbacks))
        resolved_id = None
        resolution_source = None
        rejected: list[str] = []
        for candidate_id, source in candidates:
            if candidate_id.startswith("BUSPROXY_"):
                rejected.append(f"{candidate_id}:BUSPROXY_FORBIDDEN_FINAL")
                continue
            if candidate_id not in catalog_by_id.index:
                rejected.append(f"{candidate_id}:MISSING_FROM_VENUE_CATALOG")
                continue
            candidate_cluster = field_cluster.get(candidate_id, str(catalog_by_id.loc[candidate_id].get("cluster_id", "")))
            if candidate_cluster and candidate_cluster != cluster:
                rejected.append(f"{candidate_id}:OUTSIDE_DECLARED_COVERAGE_CLUSTER")
                continue
            if not bool(verified.get(candidate_id, False)):
                rejected.append(f"{candidate_id}:NOT_FIELD_VERIFIED")
                continue
            resolved_id = candidate_id
            resolution_source = source
            break
        payload = anchor.to_dict()
        payload.update(
            {
                "final_physical_venue_id": resolved_id,
                "final_physical_venue_field_verified": bool(resolved_id),
                "venue_unresolved": resolved_id is None,
                "resolution_source": resolution_source,
                "resolution_rejections": "|".join(rejected),
                "resolution_status": "RESOLVED_VERIFIED_PHYSICAL_VENUE" if resolved_id else "UNRESOLVED_NO_VERIFIED_PHYSICAL_VENUE",
            }
        )
        if resolved_id:
            rec = catalog_by_id.loc[resolved_id]
            for target, source_col in [
                ("final_physical_venue_name", "venue_name"),
                ("final_physical_venue_type", "venue_type"),
                ("final_physical_latitude", "latitude"),
                ("final_physical_longitude", "longitude"),
            ]:
                payload[target] = rec.get(source_col, np.nan)
        rows.append(payload)
    resolution = pd.DataFrame(rows)
    unresolved = resolution.loc[resolution["venue_unresolved"]].copy()
    summary = {
        "anchors": int(len(resolution)),
        "resolved": int((~resolution["venue_unresolved"]).sum()),
        "unresolved": int(resolution["venue_unresolved"].sum()),
        "busproxy_final_count": int(resolution["final_physical_venue_id"].fillna("").astype(str).str.startswith("BUSPROXY_").sum()),
        "all_resolved": bool(not resolution["venue_unresolved"].any()),
    }
    return ResolutionResult(resolution=resolution, unresolved=unresolved, summary=summary)
