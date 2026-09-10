#!/usr/bin/env python3
"""Run the currently authorized MODEL V1 Stage-1 quality gate only."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from mediroad.stage1_pipeline import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
