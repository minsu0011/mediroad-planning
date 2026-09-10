from __future__ import annotations

import json, subprocess, sys, time
from pathlib import Path
from typing import Any

from mediroad.stage4_2_certification.backends import HighsBackend
from mediroad.stage4_2_certification.types import LinearMipModel


def write_mps(model: LinearMipModel, path: Path) -> None:
    backend=HighsBackend(require_minimum_version=True)
    h=backend.HighsClass(); backend._pass_model(h,model)  # pinned internal transport used only to serialize exact current model
    path.parent.mkdir(parents=True,exist_ok=True); h.writeModel(str(path))
    try: backend.HighsClass.resetGlobalScheduler(True)
    except Exception: pass


def run_portfolio(model: LinearMipModel, output_dir: Path, *, seeds: list[int], workers: int, time_limit_sec: float, memory_limit_mib: float | None = None) -> dict[str, Any]:
    output_dir.mkdir(parents=True,exist_ok=True)
    mps=output_dir/"threshold_oracle.mps"; write_mps(model,mps)
    # Highs writes unnamed columns as c0, c1, ... in MPS.  Keep the candidate
    # position explicitly so a SCIP witness can be mapped back without relying
    # on the in-memory LinearMipModel names (which are not serialized to MPS).
    xmap=[
        {"mps_name":f"c{column_index}","selected_index":selected_index}
        for selected_index,column_index in enumerate(range(model.x_slice.start,model.x_slice.stop))
    ]
    xfile=output_dir/"x_names.json"; xfile.write_text(json.dumps(xmap),encoding='utf-8')
    active=[]; pending=list(seeds); results=[]
    def start(seed:int):
        out=output_dir/f"worker_{seed}.json"
        log=output_dir/f"worker_{seed}.scip.log"
        cmd=[sys.executable,'-m','mediroad.stage4_2g_candidate_expansion.scip_worker','--mps',str(mps),'--out',str(out),'--log',str(log),'--seed',str(seed),'--time-limit',str(time_limit_sec),'--x-names',str(xfile)]
        if memory_limit_mib is not None: cmd.extend(['--memory-limit-mib',str(memory_limit_mib)])
        return {"seed":seed,"out":out,"log":log,"proc":subprocess.Popen(cmd,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)}
    terminal=None
    try:
        while pending and len(active)<workers: active.append(start(pending.pop(0)))
        while active:
            time.sleep(1.0)
            for item in list(active):
                if item["proc"].poll() is None: continue
                active.remove(item)
                if item["out"].exists():
                    data=json.loads(item["out"].read_text(encoding='utf-8')); results.append(data)
                    status=str(data.get("status","" )).upper()
                    if "INFEASIBLE" in status:
                        terminal={"kind":"INFEASIBLE_PROOF","worker":data}
                    elif data.get("selected_indices"):
                        terminal={"kind":"FEASIBLE_WITNESS","worker":data}
                if terminal is None and pending: active.append(start(pending.pop(0)))
            if terminal is not None:
                break
    finally:
        if active:
            for item in active:
                try: item["proc"].terminate()
                except Exception: pass
            for item in active:
                try: item["proc"].wait(timeout=10)
                except Exception:
                    try: item["proc"].kill()
                    except Exception: pass
    summary={"terminal":terminal,"results":results,"mps":str(mps),"worker_count":workers,"seeds":seeds,"time_limit_sec":time_limit_sec,"memory_limit_mib":memory_limit_mib,"worker_logs":[str(output_dir/f"worker_{seed}.scip.log") for seed in seeds]}
    (output_dir/"portfolio_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    return summary
