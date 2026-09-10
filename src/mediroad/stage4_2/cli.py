from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config
from .discovery import find_project_root
from .pipeline import Stage42Pipeline


def _operational_paths(args: argparse.Namespace) -> dict[str, Path] | None:
    values = {
        "team_bases": args.team_bases,
        "team_capability": args.team_capability,
        "vehicles": args.vehicles,
        "calendar": args.calendar,
        "venue_calendar": args.venue_calendar,
        "travel": args.travel,
    }
    if not any(values.values()):
        return None
    missing = [k for k, v in values.items() if not v]
    if missing:
        raise SystemExit(f"Operational input arguments must be supplied together; missing: {missing}")
    return {k: Path(v).resolve() for k, v in values.items()}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="MEDIROAD Stage 4.2 next-work pipeline")
    p.add_argument("--project-root", default=".", help="MEDIROAD project root")
    p.add_argument("--config", default="configs/model_v1/stage4_2.yaml")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare", help="Create prioritized field-validation and operational-input templates")
    sub.add_parser("computational", help="Run candidate expansion and long Equity certification")
    audit = sub.add_parser("audit-field", help="Validate returned field evidence and resolve physical venues")
    audit.add_argument("--field-validation", required=True)
    final = sub.add_parser("operational-final", help="Run verified-venue Stage 4 operational finalization")
    final.add_argument("--field-validation", required=True)
    for target in [audit, final]:
        target.add_argument("--team-bases")
        target.add_argument("--team-capability")
        target.add_argument("--vehicles")
        target.add_argument("--calendar")
        target.add_argument("--venue-calendar")
        target.add_argument("--travel")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = find_project_root(Path(args.project_root))
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path
    config = load_config(config_path)
    pipeline = Stage42Pipeline(root, config)
    if args.command == "prepare":
        result = pipeline.prepare()
    elif args.command == "computational":
        result = pipeline.computational()
    elif args.command == "audit-field":
        result = pipeline.audit_field(Path(args.field_validation).resolve(), _operational_paths(args))
    elif args.command == "operational-final":
        paths = _operational_paths(args)
        if paths is None:
            raise SystemExit("operational-final requires all operational input paths")
        result = pipeline.operational_final(Path(args.field_validation).resolve(), paths)
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
