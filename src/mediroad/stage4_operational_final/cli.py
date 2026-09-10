from __future__ import annotations

import argparse
import json
from pathlib import Path

from .runner import run_readiness


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Stage 4.2H-bound Operational Final pipeline without inventing field facts."
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--config", default="configs/model_v1/stage4_operational_final.yaml")
    parser.add_argument(
        "--field-validation",
        help="Optional returned field-validation CSV. If omitted, a fail-closed manual form is generated.",
    )
    parser.add_argument(
        "--field-evidence",
        help="Optional returned field-evidence CSV paired one-to-one with --field-validation.",
    )
    parser.add_argument(
        "--operational-input-dir",
        help="Optional directory containing confirmed operational input files.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.project_root).resolve()
    config = Path(args.config)
    if not config.is_absolute():
        config = root / config
    result = run_readiness(
        root,
        config,
        field_validation_path=Path(args.field_validation).resolve() if args.field_validation else None,
        field_evidence_path=Path(args.field_evidence).resolve() if args.field_evidence else None,
        operational_input_dir=Path(args.operational_input_dir).resolve() if args.operational_input_dir else None,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
