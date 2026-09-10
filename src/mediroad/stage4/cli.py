from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import run_stage4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MEDIROAD Stage 4 pre-validation and provisional multi-objective optimization."
    )
    parser.add_argument("--package-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--mode", choices=["official", "diagnostic"], default="official")
    parser.add_argument("--solver", choices=["cp-sat", "greedy"], default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-network-validation", action="store_true")
    parser.add_argument("--skip-frontier", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_root = run_stage4(
        args.package_root,
        config_path=args.config,
        mode=args.mode,
        solver_name=args.solver,
        force=args.force,
        skip_network_validation=args.skip_network_validation,
        run_frontier=not args.skip_frontier,
    )
    print(f"Stage 4 completed: {run_root}")
    return 0

