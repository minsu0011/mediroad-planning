"""Fail-closed audit helpers for Stage 2B temporal inputs and outputs.

The temporal layer combines sources with different spatial and temporal
resolution.  These helpers keep that provenance explicit and, in particular,
prevent unavailable operational fields from being silently encoded as zero.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd


TEMPORAL_AUDIT_COLUMNS = [
    "dataset",
    "time_range",
    "time_resolution",
    "spatial_resolution",
    "specialty_resolution",
    "source",
    "missingness",
    "usable_for_model",
    "usable_for_validation",
    "confidence",
    "limitations",
]


def temporal_audit_row(
    frame: pd.DataFrame,
    *,
    dataset: str,
    time_resolution: str,
    spatial_resolution: str,
    specialty_resolution: str,
    source: str,
    usable_for_model: bool,
    usable_for_validation: bool,
    confidence: str,
    limitations: str,
    time_values: pd.Series | Sequence[Any] | None = None,
    required_columns: Iterable[str] = (),
) -> dict[str, Any]:
    """Create one machine-readable temporal audit record.

    ``missingness`` is measured only over explicitly required columns.  This
    avoids making an unrelated descriptive column change a quality gate.
    """

    if not dataset or not source:
        raise ValueError("dataset and source must be non-empty")
    required = list(required_columns)
    missing_columns = sorted(set(required) - set(frame.columns))
    if missing_columns:
        raise KeyError(f"Temporal dataset {dataset} lacks required columns: {missing_columns}")

    if required:
        missingness = float(frame[required].isna().to_numpy().mean())
    else:
        missingness = float(frame.isna().to_numpy().mean()) if frame.size else 0.0

    if time_values is None:
        time_range = "not_applicable_or_not_provided"
    else:
        values = pd.Series(time_values).dropna()
        if values.empty:
            time_range = "unknown"
        else:
            numeric = pd.to_numeric(values, errors="coerce")
            if numeric.notna().all() and numeric.between(1900, 2200).all():
                time_range = f"{int(numeric.min())}..{int(numeric.max())}"
            else:
                parsed = pd.to_datetime(values, errors="coerce")
                if parsed.notna().all():
                    time_range = (
                        f"{parsed.min().date().isoformat()}..{parsed.max().date().isoformat()}"
                    )
                else:
                    text = values.astype(str)
                    time_range = f"{text.min()}..{text.max()}"

    return {
        "dataset": str(dataset),
        "time_range": time_range,
        "time_resolution": str(time_resolution),
        "spatial_resolution": str(spatial_resolution),
        "specialty_resolution": str(specialty_resolution),
        "source": str(source),
        "missingness": missingness,
        "usable_for_model": bool(usable_for_model),
        "usable_for_validation": bool(usable_for_validation),
        "confidence": str(confidence),
        "limitations": str(limitations),
        "row_count": int(len(frame)),
        "column_count": int(len(frame.columns)),
    }


def build_temporal_data_audit(records: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    """Validate and combine audit records in the prompt-defined column order."""

    rows = [dict(record) for record in records]
    if not rows:
        raise ValueError("Temporal audit requires at least one record")
    for index, row in enumerate(rows):
        missing = [column for column in TEMPORAL_AUDIT_COLUMNS if column not in row]
        if missing:
            raise KeyError(f"Temporal audit row {index} lacks fields: {missing}")
    extra = sorted(set().union(*(row.keys() for row in rows)) - set(TEMPORAL_AUDIT_COLUMNS))
    return pd.DataFrame(rows)[TEMPORAL_AUDIT_COLUMNS + extra]


def assert_unknown_not_zero(
    frame: pd.DataFrame,
    columns: Iterable[str],
) -> None:
    """Require explicit unknown status tokens; numeric/missing zero is invalid.

    The unavailable operational components in Stage 2B are excluded from the
    score denominator.  They must remain explicit status strings so no caller
    can accidentally interpret ``0`` as known infeasibility or no overlap.
    """

    for column in columns:
        if column not in frame.columns:
            raise KeyError(f"Missing required unknown-status column: {column}")
        values = frame[column]
        if values.isna().any():
            raise ValueError(f"{column} contains missing values instead of explicit unknown status")
        invalid: list[Any] = []
        for value in values.unique().tolist():
            if isinstance(value, (int, float, np.integer, np.floating)):
                invalid.append(value)
                continue
            token = str(value).strip().lower()
            if not (token == "unknown" or token.startswith("unknown_") or token.startswith("unavailable_")):
                invalid.append(value)
        if invalid:
            raise ValueError(
                f"{column} must encode unknown explicitly and never as zero; invalid={invalid[:5]}"
            )


def assert_complete_bounded_scores(
    frame: pd.DataFrame,
    columns: Iterable[str],
    *,
    lower: float = 0.0,
    upper: float = 100.0,
) -> None:
    """Fail if any model score is missing, non-finite, or outside its contract."""

    for column in columns:
        if column not in frame.columns:
            raise KeyError(f"Missing score column: {column}")
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any() or not np.isfinite(values.to_numpy(float)).all():
            raise ValueError(f"{column} contains missing or non-finite values")
        outside = ~values.between(lower, upper)
        if outside.any():
            raise ValueError(
                f"{column} has {int(outside.sum())} values outside [{lower}, {upper}]"
            )
