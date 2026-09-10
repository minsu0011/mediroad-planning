from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .utils import atomic_write_text


def _table(frame: pd.DataFrame, max_rows: int = 30) -> str:
    if frame.empty:
        return "_No rows._"
    try:
        return frame.head(max_rows).to_markdown(index=False)
    except Exception:
        return "```text\n" + frame.head(max_rows).to_string(index=False) + "\n```"


def write_prepare_report(path: Path, summary: dict[str, Any], queue: pd.DataFrame) -> None:
    text = f"""# MEDIROAD Stage 4.2 — Field Validation Preparation

## Status

- Queue rows: `{summary.get('queue_rows')}`
- Official anchor rows: `{summary.get('anchor_rows')}`
- BUSPROXY anchors without fallback: `{summary.get('busproxy_without_fallback')}`
- Certified Efficiency∩Balanced anchors: `{summary.get('certified_intersection')}`
- Stage 5 started: `false`

## Project direction

Stage 4.2 does not alter Stage 1 Need, Stage 2A Specialty Gap, Stage 2B abstention, or Stage 3 coverage. It separates:

1. computational uncertainty — candidate expansion and Equity solver gap;
2. operational uncertainty — actual venue, team base, vehicle, travel and calendar evidence.

The current queue is a field-validation priority list, not a final twenty-site plan.

## Highest-priority rows

{_table(queue[[c for c in ['stage4_2_field_rank','priority_tier','priority_reason','venue_id','venue_name','sigungu','is_busproxy_anchor','actual_facility_fallback_count'] if c in queue]], 25)}
"""
    atomic_write_text(path, text)


def write_computational_report(path: Path, decision: dict[str, Any], comparison: pd.DataFrame, metrics: pd.DataFrame) -> None:
    text = f"""# MEDIROAD Stage 4.2 — Computational Hardening Report

## Decision

```json
{json.dumps(decision, ensure_ascii=False, indent=2)}
```

## Interpretation

- A candidate-expansion result is comparable only when both the Top3 reference and expanded set are certified under the frozen 0.5% gap contract.
- An uncertified Top5/coarse-Pareto incumbent must not be used to claim that Top3 is lossless or inferior.
- Equity is policy evidence only after all lexicographic stages are certified.
- Stage 5 is not started by this run.

## Candidate expansion comparison

{_table(comparison, 20)}

## Scenario metrics

{_table(metrics, 30)}
"""
    atomic_write_text(path, text)


def write_field_audit_report(path: Path, summary: dict[str, Any], resolution_summary: dict[str, Any], blockers: list[str]) -> None:
    text = f"""# MEDIROAD Stage 4.2 — Field and Operational Readiness Audit

## Field validation

```json
{json.dumps(summary, ensure_ascii=False, indent=2)}
```

## Physical venue resolution

```json
{json.dumps(resolution_summary, ensure_ascii=False, indent=2)}
```

## Remaining blockers

{chr(10).join('- ' + b for b in blockers) if blockers else '- None'}

This audit does not create dates, routes, or Stage 5 schedules.
"""
    atomic_write_text(path, text)
