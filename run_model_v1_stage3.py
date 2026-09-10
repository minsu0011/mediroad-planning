from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mediroad.stage3_pipeline import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main(["--package-root", str(ROOT), *sys.argv[1:]]))
