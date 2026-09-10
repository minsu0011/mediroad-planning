from __future__ import annotations
import hashlib, json, os, tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import pandas as pd

def sha256_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def _default(v:Any)->Any:
    if isinstance(v,Path): return str(v)
    if hasattr(v,'tolist'): return v.tolist()
    if hasattr(v,'item'): return v.item()
    raise TypeError(type(v).__name__)

def atomic_write_json(path:Path,payload:Any)->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix=path.name,suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(fd,'w',encoding='utf-8',newline='\n') as f:
            json.dump(payload,f,ensure_ascii=False,indent=2,default=_default); f.write('\n')
        os.replace(tmp,path)
    except Exception:
        try: os.unlink(tmp)
        except FileNotFoundError: pass
        raise

def atomic_write_csv(path:Path,frame:pd.DataFrame)->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix=path.name,suffix='.tmp',dir=path.parent); os.close(fd)
    try:
        frame.to_csv(tmp,index=False,encoding='utf-8-sig'); os.replace(tmp,path)
    except Exception:
        try: os.unlink(tmp)
        except FileNotFoundError: pass
        raise

def utc_now()->str: return datetime.now(timezone.utc).isoformat()
def make_run_id(prefix:str,fingerprint:str)->str:
    return f"{prefix}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{fingerprint[:12]}"

def tree_inventory(paths:list[Path],root:Path)->list[dict[str,Any]]:
    rows=[]
    for p in sorted({x.resolve() for x in paths if x.exists()}):
        rows.append({'relative_path':p.relative_to(root.resolve()).as_posix(),'size_bytes':p.stat().st_size,'sha256':sha256_file(p)})
    return rows
