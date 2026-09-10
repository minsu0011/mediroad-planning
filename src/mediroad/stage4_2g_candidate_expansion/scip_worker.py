from __future__ import annotations
import argparse, json, math, time
from pathlib import Path


def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument('--mps',required=True); p.add_argument('--out',required=True); p.add_argument('--log',required=True); p.add_argument('--seed',type=int,required=True); p.add_argument('--time-limit',type=float,required=True); p.add_argument('--memory-limit-mib',type=float); p.add_argument('--x-names',required=True)
    a=p.parse_args()
    started=time.perf_counter()
    payload={"seed":a.seed,"status":"ERROR","selected_variable_names":[],"selected_indices":[],"wall_time_sec":0.0}
    try:
        from pyscipopt import Model
        m=Model(); m.hideOutput(); m.readProblem(a.mps)
        raw_xmap=json.loads(Path(a.x_names).read_text(encoding='utf-8'))
        xmap=[]
        for position,item in enumerate(raw_xmap):
            if isinstance(item,dict):
                xmap.append((str(item["mps_name"]),int(item["selected_index"])))
            else:  # Backward-compatible with diagnostic artifacts made before Stage 4.2H.
                xmap.append((str(item),position))
        variables_by_name={str(var.name):var for var in m.getVars()}
        missing=[name for name,_ in xmap if name not in variables_by_name]
        if missing:
            raise RuntimeError(f"MPS candidate-column mapping failed for {len(missing)} columns; first={missing[0]!r}")
        m.setLogfile(str(a.log)); m.setIntParam("display/verblevel",3); m.hideOutput(True)
        for name,value in [("limits/time",float(a.time_limit)),("limits/gap",0.0),("randomization/randomseedshift",int(a.seed)),("randomization/permutationseed",int(a.seed)),("parallel/maxnthreads",1),("lp/threads",1)]:
            try: m.setParam(name,value)
            except Exception: pass
        if a.memory_limit_mib is not None:
            m.setRealParam("limits/memory",float(a.memory_limit_mib))
        m.optimize(); status=str(m.getStatus()).upper()
        payload["status"]=status
        payload["n_nodes"]=int(m.getNNodes())
        payload["primal_bound"]=float(m.getPrimalbound()) if math.isfinite(float(m.getPrimalbound())) else None
        payload["dual_bound"]=float(m.getDualbound()) if math.isfinite(float(m.getDualbound())) else None
        payload["memory_limit_mib"]=a.memory_limit_mib
        payload["log_path"]=str(a.log)
        payload["variable_count"]=int(m.getNVars())
        payload["constraint_count"]=int(m.getNConss())
        payload["binary_variable_count"]=sum(str(v.vtype()).upper()=="BINARY" for v in m.getVars())
        if status in {"OPTIMAL","BESTSOLLIMIT","TIMELIMIT","GAPLIMIT","SOLLIMIT"} and m.getNSols()>0:
            sol=m.getBestSol(); selected=[]; selected_indices=[]
            for name,selected_index in xmap:
                var=variables_by_name[name]
                if m.getSolVal(sol,var)>0.5:
                    selected.append(str(name)); selected_indices.append(int(selected_index))
            payload["selected_variable_names"]=selected
            payload["selected_indices"]=selected_indices
    except Exception as exc:
        payload["error"]=repr(exc)
    payload["wall_time_sec"]=float(time.perf_counter()-started)
    Path(a.out).write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    return 0
if __name__=='__main__': raise SystemExit(main())
