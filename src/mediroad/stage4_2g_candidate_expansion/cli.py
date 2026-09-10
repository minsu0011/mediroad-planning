from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from .runner import run

def _json_default(x):
    if isinstance(x,np.ndarray): return x.tolist()
    if isinstance(x,np.generic): return x.item()
    if isinstance(x,Path): return str(x)
    raise TypeError(type(x).__name__)

def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument('--project-root',type=Path,required=True); p.add_argument('--config',type=Path)
    a=p.parse_args(argv); cfg=a.config or (a.project_root/'configs/model_v1/stage4_2g_candidate_expansion.yaml')
    result=run(a.project_root,cfg); print(json.dumps(result,ensure_ascii=False,indent=2,default=_json_default)); return 0
if __name__=='__main__': raise SystemExit(main())
