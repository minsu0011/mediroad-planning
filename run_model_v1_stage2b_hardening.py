"""Repository-root entry point for MEDIROAD MODEL V1 Stage 2B hardening."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mediroad.stage2b_hardening_pipeline import main


if __name__ == "__main__":
    raise SystemExit(main())
