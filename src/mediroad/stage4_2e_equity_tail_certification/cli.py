from __future__ import annotations

import argparse
import json
from pathlib import Path

from .runner import run_equity_tail_certification


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(type(value).__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MEDIROAD Stage 4.2E Top3 Equity tail certification"
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--stage42-config", type=Path, default=Path("configs/model_v1/stage4_2.yaml")
    )
    parser.add_argument(
        "--stage42c-config",
        type=Path,
        default=Path("configs/model_v1/stage4_2_certification.yaml"),
    )
    parser.add_argument(
        "--stage42d-config",
        type=Path,
        default=Path("configs/model_v1/stage4_2d_equity_certification.yaml"),
    )
    parser.add_argument(
        "--stage42e-config",
        type=Path,
        default=Path("configs/model_v1/stage4_2e_equity_tail_certification.yaml"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_equity_tail_certification(
        project_root=args.project_root,
        stage42_config_path=args.stage42_config,
        stage42c_config_path=args.stage42c_config,
        stage42d_config_path=args.stage42d_config,
        stage42e_config_path=args.stage42e_config,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=_json_default))
    return 0 if result["passed"] else 2

