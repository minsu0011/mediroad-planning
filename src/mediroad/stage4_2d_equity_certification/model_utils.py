from __future__ import annotations

import copy
import math
from dataclasses import replace
from typing import Iterable

import numpy as np
from scipy import sparse

from mediroad.stage4_2_certification.types import LinearMipModel

from .types import CutRecord, FixingRecord


def clone_model(model: LinearMipModel, *, name: str | None = None) -> LinearMipModel:
    cloned = replace(
        model,
        name=name or model.name,
        c=np.asarray(model.c, dtype=np.float64).copy(),
        A=model.A.copy().tocsc(),
        row_lower=np.asarray(model.row_lower, dtype=np.float64).copy(),
        row_upper=np.asarray(model.row_upper, dtype=np.float64).copy(),
        col_lower=np.asarray(model.col_lower, dtype=np.float64).copy(),
        col_upper=np.asarray(model.col_upper, dtype=np.float64).copy(),
        integrality=np.asarray(model.integrality, dtype=np.int8).copy(),
        variable_names=list(model.variable_names),
        extra_slices=dict(model.extra_slices),
        metadata=copy.deepcopy(model.metadata),
    )
    cloned.validate()
    return cloned


def apply_fixings(model: LinearMipModel, fixings: Iterable[FixingRecord]) -> LinearMipModel:
    output = clone_model(model)
    applied: list[dict[str, object]] = []
    for fixing in fixings:
        col = output.x_slice.start + int(fixing.candidate_index)
        value = float(fixing.value)
        output.col_lower[col] = value
        output.col_upper[col] = value
        applied.append(
            {
                "candidate_index": int(fixing.candidate_index),
                "value": int(fixing.value),
                "metric": fixing.metric,
                "proof_type": fixing.proof_type,
            }
        )
    output.metadata.setdefault("stage42d", {})["fixings"] = applied
    output.validate()
    return output


def append_cuts(model: LinearMipModel, cuts: Iterable[CutRecord]) -> LinearMipModel:
    cuts = list(cuts)
    if not cuts:
        return clone_model(model)
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    names: list[str] = []
    for r, cut in enumerate(cuts):
        indices = np.asarray(cut.indices, dtype=np.int64)
        values = np.asarray(cut.values, dtype=np.float64)
        if indices.size != values.size:
            raise ValueError(f"Cut {cut.name}: index/value length mismatch")
        if indices.size and (indices.min() < 0 or indices.max() >= model.x_slice.stop - model.x_slice.start):
            raise ValueError(f"Cut {cut.name}: candidate index outside x slice")
        finite = np.isfinite(values) & (np.abs(values) > 0.0)
        indices = indices[finite]
        values = values[finite]
        rows.extend([r] * int(indices.size))
        cols.extend((model.x_slice.start + indices).tolist())
        data.extend(values.tolist())
        lower.append(float(cut.lower))
        upper.append(float(cut.upper))
        names.append(cut.name)
    extra = sparse.coo_matrix(
        (np.asarray(data, dtype=np.float64), (np.asarray(rows), np.asarray(cols))),
        shape=(len(cuts), model.n_col),
    ).tocsc()
    output = clone_model(model)
    output.A = sparse.vstack([output.A, extra], format="csc")
    output.row_lower = np.concatenate([output.row_lower, np.asarray(lower, dtype=np.float64)])
    output.row_upper = np.concatenate([output.row_upper, np.asarray(upper, dtype=np.float64)])
    row_names = list(output.metadata.get("row_names", []))
    row_names.extend(names)
    output.metadata["row_names"] = row_names
    output.metadata.setdefault("stage42d", {})["cut_count"] = len(cuts)
    output.metadata["stage42d"]["cut_kinds"] = sorted({cut.kind for cut in cuts})
    output.validate()
    return output


def local_branching_cut(selected: np.ndarray, visit_count: int, radius: int, *, mode: str) -> CutRecord:
    selected = np.unique(np.asarray(selected, dtype=np.int32))
    if selected.size != visit_count:
        raise ValueError("Local-branching seed size differs from visit_count")
    if not 0 <= radius < visit_count:
        raise ValueError("Local-branching radius outside valid range")
    if mode == "inside":
        # At most radius replacements: |T ∩ S| >= k-radius.
        lower, upper = float(visit_count - radius), math.inf
        kind = "LOCAL_BRANCH_INSIDE"
    elif mode == "exclude":
        # Exclude a ball already proved infeasible.
        lower, upper = -math.inf, float(visit_count - radius - 1)
        kind = "LOCAL_BRANCH_EXCLUSION"
    else:
        raise ValueError(mode)
    return CutRecord(
        name=f"stage42d::{kind.lower()}::r{radius}::{abs(hash(tuple(selected.tolist()))) % 10**10}",
        indices=selected,
        values=np.ones(selected.size, dtype=np.float64),
        lower=lower,
        upper=upper,
        kind=kind,
        anchor_size=int(selected.size),
        metadata={"radius": int(radius)},
    )


def solution_cutoff_tolerance(value: float) -> float:
    return 1e-8 * max(1.0, abs(float(value)))
