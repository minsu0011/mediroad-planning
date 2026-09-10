from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from .runner import run


def _safe(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, dict): return {str(k): _safe(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)): return [_safe(v) for v in value]
    return value


def main(argv=None) -> int:
    p=argparse.ArgumentParser(description='MEDIROAD Stage4.2H coarse candidate finalization')
    p.add_argument('--project-root', type=Path, required=True)
    p.add_argument('--config', type=Path, default=Path('configs/model_v1/stage4_2h_coarse_equity.yaml'))
    a=p.parse_args(argv)
    result=run(a.project_root, a.config)
    print(json.dumps(_safe(result), ensure_ascii=False, indent=2))
    return 0 if result.get('passed') else 2
