from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from .errors import ContractError
from .utils import normalize_bool


@dataclass
class CandidateReductionResult:
    candidates: pd.DataFrame
    coverage: sparse.csr_matrix
    ledger: pd.DataFrame


def candidate_cost_coefficients(candidates: pd.DataFrame, config: dict[str, Any]) -> np.ndarray:
    """Return the exact venue-level cost coefficients used by the spatial MILP.

    Candidate reduction is only safe when it uses the same coefficient vector as
    the optimizer.  Keeping the calculation here prevents a subtle drift where a
    broader-coverage replacement could be more expensive in the final lexical
    cost stage.
    """

    frame = candidates.reset_index(drop=True)
    coeff = np.zeros(len(frame), dtype=float)
    travel = pd.to_numeric(frame.get("travel_minutes", pd.Series(np.nan, index=frame.index)), errors="coerce")
    if not isinstance(travel, pd.Series):
        travel = pd.Series(np.nan, index=frame.index)
    if travel.notna().any():
        fill = float(travel.median())
        coeff += travel.fillna(fill).clip(lower=0).to_numpy(float) * float(
            config.get("optimization", {}).get("travel_penalty_scale", 1.0)
        )
    known = frame.get("known_overlap", pd.Series("UNKNOWN", index=frame.index))
    known_flag = known.astype(str).str.strip().str.lower().isin({"known", "yes", "true", "1", "confirmed", "high"})
    coeff += known_flag.to_numpy(float) * float(config.get("optimization", {}).get("known_overlap_penalty", 1000.0))
    tie = pd.to_numeric(frame.get("candidate_priority_tiebreak", 0.0), errors="coerce")
    if not isinstance(tie, pd.Series):
        tie = pd.Series(0.0, index=frame.index)
    tie_rank = tie.fillna(0.0).rank(method="first", ascending=False).to_numpy(float)
    coeff += tie_rank * float(config.get("optimization", {}).get("deterministic_tiebreak_scale", 1e-6))
    return coeff


def build_candidate_set(
    venues: pd.DataFrame,
    membership: pd.DataFrame,
    candidate_set: str,
    *,
    verified_only: bool = False,
) -> pd.DataFrame:
    required = ["candidate_set", "venue_id"]
    missing = [c for c in required if c not in membership.columns]
    if missing:
        raise ContractError(f"Candidate membership missing columns: {missing}")
    member = membership.loc[membership["candidate_set"].astype(str).eq(candidate_set)].copy()
    if member.empty:
        raise ContractError(f"Candidate set {candidate_set!r} is empty or unavailable")
    member["venue_id"] = member["venue_id"].astype(str)
    out = member.merge(venues, on="venue_id", how="left", suffixes=("_membership", ""), validate="one_to_one")
    if out["admin_code"].isna().any():
        bad = out.loc[out["admin_code"].isna(), "venue_id"].head(10).tolist()
        raise ContractError(f"Candidate membership contains venues missing from Stage 3 interface: {bad}")
    for col in ["admin_code", "sigungu", "cluster_id"]:
        member_col = f"{col}_membership"
        if member_col in out:
            base = out[col].astype(str)
            other = out[member_col].astype(str)
            mismatch = base.ne(other) & out[member_col].notna()
            if mismatch.any():
                raise ContractError(f"Candidate membership mismatch in {col}: {int(mismatch.sum())}")
            out = out.drop(columns=[member_col])
    out["candidate_set_id"] = candidate_set
    if verified_only:
        if "field_verified" not in out:
            raise ContractError("verified candidate set requested without field_verified column")
        out = out.loc[out["field_verified"].map(normalize_bool).fillna(False)].copy()
    return out.drop_duplicates("venue_id").reset_index(drop=True)


def _row_signature(matrix: sparse.csr_matrix, row: int) -> str:
    start, end = matrix.indptr[row], matrix.indptr[row + 1]
    indices = np.asarray(matrix.indices[start:end], dtype=np.int64)
    return hashlib.sha256(indices.tobytes()).hexdigest()


def exact_constraint_signature(frame: pd.DataFrame, idx: int) -> tuple[str, ...]:
    row = frame.iloc[idx]
    known = str(row.get("known_overlap", "UNKNOWN")).strip().upper()
    return (
        str(row.get("admin_code", "")),
        str(row.get("sigungu", "")),
        str(row.get("cluster_id", row.get("venue_id", ""))),
        known,
    )


def compress_exact_equivalent_candidates(
    candidates: pd.DataFrame,
    coverage: sparse.csr_matrix,
    config: dict[str, Any],
) -> CandidateReductionResult:
    """Losslessly collapse venues with identical coverage and identical hard-constraint signature.

    The representative is chosen deterministically. Non-BUS, field-verified, higher readiness,
    lower travel and higher Stage-3 tiebreak are preferred. This is lossless for the spatial
    MILP because all collapsed rows have identical coverage and identical admin/sigungu/cluster/
    overlap constraints.
    """
    if len(candidates) != coverage.shape[0]:
        raise ContractError("Candidate frame and coverage row counts differ")
    work = candidates.reset_index(drop=True).copy()
    cost = candidate_cost_coefficients(work, config)
    keys: list[tuple[Any, ...]] = []
    for i in range(len(work)):
        # Equal spatial coverage and constraints are not enough: the final
        # lexicographic cost coefficient must also be exactly equal.
        keys.append((_row_signature(coverage, i), *exact_constraint_signature(work, i), cost[i].hex()))
    work["_equivalence_key"] = ["|".join(map(str, key)) for key in keys]

    is_bus = work["venue_id"].astype(str).str.startswith("BUSPROXY_")
    verified = work.get("field_verified", pd.Series(False, index=work.index)).map(normalize_bool).fillna(False)

    def numeric_column(name: str, default: float) -> pd.Series:
        raw = work[name] if name in work.columns else pd.Series(default, index=work.index, dtype=float)
        return pd.to_numeric(raw, errors="coerce")

    readiness = numeric_column("venue_readiness_prior", 0.0).fillna(0.0)
    travel = numeric_column("travel_minutes", np.nan)
    travel = travel.fillna(float(travel.median()) if travel.notna().any() else 0.0)
    tiebreak = numeric_column("candidate_priority_tiebreak", 0.0).fillna(0.0)
    work["_priority"] = (
        (~is_bus).astype(int) * 1_000_000
        + verified.astype(int) * 100_000
        + readiness * 10_000
        - travel * 10
        + tiebreak
    )
    work = work.sort_values(["_equivalence_key", "_priority", "venue_id"], ascending=[True, False, True])
    keep = work.groupby("_equivalence_key", sort=False).head(1).copy()
    keep_indices = keep.index.to_numpy(dtype=int)
    compressed = coverage[keep_indices].tocsr()
    representatives = keep.sort_index().copy()

    representative_by_key = dict(zip(keep["_equivalence_key"], keep["venue_id"].astype(str)))
    ledger = work[["venue_id", "_equivalence_key"]].copy()
    ledger["representative_venue_id"] = ledger["_equivalence_key"].map(representative_by_key)
    ledger["retained"] = ledger["venue_id"].astype(str).eq(ledger["representative_venue_id"].astype(str))
    ledger["reason"] = np.where(ledger["retained"], "RETAINED", "EXACT_COVERAGE_AND_CONSTRAINT_EQUIVALENT")

    representatives = representatives.drop(columns=["_equivalence_key", "_priority"]).reset_index(drop=True)
    # The sparse rows must follow the same order as representatives.
    original_positions = keep.sort_index().index.to_numpy(dtype=int)
    compressed = coverage[original_positions].tocsr()
    return CandidateReductionResult(candidates=representatives, coverage=compressed, ledger=ledger.reset_index(drop=True))


def _is_subset(row_a: np.ndarray, row_b: np.ndarray) -> bool:
    if len(row_a) > len(row_b):
        return False
    # Both arrays are sorted CSR indices.
    ia = ib = 0
    while ia < len(row_a) and ib < len(row_b):
        if row_a[ia] == row_b[ib]:
            ia += 1
            ib += 1
        elif row_a[ia] > row_b[ib]:
            ib += 1
        else:
            return False
    return ia == len(row_a)


def prune_safe_dominated_candidates(
    candidates: pd.DataFrame,
    coverage: sparse.csr_matrix,
    config: dict[str, Any],
    *,
    max_group_size: int = 80,
) -> CandidateReductionResult:
    """Remove only provably dominated candidates inside identical hard-constraint groups.

    A is dominated by B when A's covered grids are a subset of B's, both have identical
    admin/sigungu/cluster/known-overlap constraints, and B has no greater confirmed travel
    cost. This pruning is optional and fail-safe; large groups are skipped.
    """
    if len(candidates) != coverage.shape[0]:
        raise ContractError("Candidate frame and coverage row counts differ")
    work = candidates.reset_index(drop=True).copy()
    work["_constraint_key"] = ["|".join(exact_constraint_signature(work, i)) for i in range(len(work))]
    cost = candidate_cost_coefficients(work, config)
    retain = np.ones(len(work), dtype=bool)
    dominated_by: dict[int, int] = {}
    for _, group in work.groupby("_constraint_key", sort=False):
        idxs = group.index.to_list()
        if len(idxs) <= 1 or len(idxs) > max_group_size:
            continue
        rows = {}
        for i in idxs:
            rows[i] = coverage.indices[coverage.indptr[i] : coverage.indptr[i + 1]]
        # Prefer broader coverage, then the exact optimizer cost coefficient.
        ordered = sorted(idxs, key=lambda i: (-len(rows[i]), float(cost[i]), str(work.iloc[i]["venue_id"])))
        for pos, b in enumerate(ordered):
            if not retain[b]:
                continue
            for a in ordered[pos + 1 :]:
                if not retain[a] or float(cost[b]) > float(cost[a]) + 1e-12:
                    continue
                if _is_subset(rows[a], rows[b]):
                    retain[a] = False
                    dominated_by[a] = b
    keep_idx = np.flatnonzero(retain)
    ledger_rows: list[dict[str, Any]] = []
    for i in range(len(work)):
        if i in dominated_by:
            b = dominated_by[i]
            ledger_rows.append(
                {
                    "venue_id": str(work.iloc[i]["venue_id"]),
                    "retained": False,
                    "representative_venue_id": str(work.iloc[b]["venue_id"]),
                    "reason": "SAFE_COVERAGE_SUBSET_DOMINATED_WITHIN_IDENTICAL_CONSTRAINT_GROUP",
                }
            )
        else:
            ledger_rows.append(
                {
                    "venue_id": str(work.iloc[i]["venue_id"]),
                    "retained": True,
                    "representative_venue_id": str(work.iloc[i]["venue_id"]),
                    "reason": "RETAINED",
                }
            )
    return CandidateReductionResult(
        candidates=work.iloc[keep_idx].drop(columns=["_constraint_key"]).reset_index(drop=True),
        coverage=coverage[keep_idx].tocsr(),
        ledger=pd.DataFrame(ledger_rows),
    )
