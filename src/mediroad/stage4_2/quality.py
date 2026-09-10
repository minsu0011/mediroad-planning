from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .utils import atomic_write_csv


def build_quality_gate(checks: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(checks)
    required = ["gate_id", "gate_type", "passed", "observed", "expected", "message"]
    for col in required:
        if col not in frame:
            frame[col] = None
    frame = frame[required]
    frame["passed"] = frame["passed"].astype(bool)
    return frame


def save_quality_gate(path: Path, checks: list[dict[str, Any]]) -> pd.DataFrame:
    frame = build_quality_gate(checks)
    atomic_write_csv(path, frame)
    return frame


def hard_pass(frame: pd.DataFrame) -> bool:
    hard = frame.loc[frame["gate_type"].astype(str).str.upper().eq("HARD")]
    return bool(len(hard) and hard["passed"].all())
