from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .errors import ContractError


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def safe_token(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_text(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ContractError(f"Expected a JSON object: {path}")
    return data


def atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        tmp.write_text(text, encoding=encoding)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=False) + "\n")


def atomic_write_csv(path: Path, frame: pd.DataFrame, *, encoding: str = "utf-8-sig") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        frame.to_csv(tmp, index=False, encoding=encoding)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        try:
            return pd.read_parquet(path)
        except ImportError as exc:
            raise ContractError(
                f"Parquet support is required for {path}. Install pyarrow from requirements_stage4_2.txt."
            ) from exc
    if suffix in {".csv", ".gz"} or path.name.lower().endswith(".csv.gz"):
        return pd.read_csv(path, low_memory=False)
    if suffix in {".json"}:
        data = json.loads(path.read_text(encoding="utf-8"))
        return pd.DataFrame(data)
    raise ContractError(f"Unsupported table format: {path}")


def ensure_columns(frame: pd.DataFrame, columns: Iterable[str], *, label: str) -> None:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ContractError(f"{label} missing required columns: {missing}")


def normalize_bool(value: Any) -> bool | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "예", "네"}:
        return True
    if text in {"false", "0", "no", "n", "아니오", "아니요"}:
        return False
    return None


def normalize_yes_no_unknown(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "UNKNOWN"
    text = str(value).strip().upper()
    aliases = {
        "Y": "YES",
        "TRUE": "YES",
        "1": "YES",
        "예": "YES",
        "네": "YES",
        "N": "NO",
        "FALSE": "NO",
        "0": "NO",
        "아니오": "NO",
        "아니요": "NO",
        "": "UNKNOWN",
        "NA": "UNKNOWN",
        "N/A": "UNKNOWN",
        "미확인": "UNKNOWN",
    }
    return aliases.get(text, text)


def parse_date(value: Any) -> pd.Timestamp | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    if not text or text.upper() in {"UNKNOWN", "NA", "N/A"}:
        return None
    ts = pd.to_datetime(text, errors="coerce")
    if pd.isna(ts):
        return None
    return pd.Timestamp(ts).normalize()


def relative_to(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def build_inventory(root: Path, *, exclude_names: set[str] | None = None) -> pd.DataFrame:
    exclude_names = exclude_names or set()
    rows: list[dict[str, Any]] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.name not in exclude_names):
        rows.append(
            {
                "relative_path": path.relative_to(root).as_posix(),
                "size_bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )
    return pd.DataFrame(rows, columns=["relative_path", "size_bytes", "sha256"])


@contextmanager
def single_writer_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise ContractError(f"Another Stage 4.2 writer appears active: {path}") from exc
    try:
        os.write(fd, f"pid={os.getpid()}\nstarted={utc_now_iso()}\n".encode("utf-8"))
        os.close(fd)
        yield
    finally:
        path.unlink(missing_ok=True)


def deterministic_run_id(prefix: str, contract: dict[str, Any]) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{safe_token(prefix)}_{timestamp}_{sha256_json(contract)[:12]}"
