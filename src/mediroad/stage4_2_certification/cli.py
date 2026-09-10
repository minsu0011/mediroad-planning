from __future__ import annotations

import argparse
import json
from pathlib import Path

from .runner import run_certification


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MEDIROAD Stage 4.2C certification hardening")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--stage42-config",
        type=Path,
        default=Path("configs/model_v1/stage4_2.yaml"),
    )
    parser.add_argument(
        "--certification-config",
        type=Path,
        default=Path("configs/model_v1/stage4_2_certification.yaml"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_certification(
        project_root=args.project_root,
        stage42_config_path=args.stage42_config,
        certification_config_path=args.certification_config,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 2
