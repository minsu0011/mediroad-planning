from __future__ import annotations

import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    tmp = Path(raw)
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2, default=_json_default) + "\n")


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    tmp = Path(raw)
    try:
        frame.to_csv(tmp, index=False, encoding="utf-8-sig")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _json_default(value: Any) -> Any:
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
    except Exception:
        pass
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "__dict__"):
        return value.__dict__
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def build_inventory(root: Path, *, exclude: Iterable[str] = ()) -> pd.DataFrame:
    excluded = set(exclude)
    rows: list[dict[str, Any]] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.name not in excluded):
        rows.append(
            {
                "relative_path": path.relative_to(root).as_posix(),
                "size_bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )
    return pd.DataFrame(rows, columns=["relative_path", "size_bytes", "sha256"])


def verify_inventory(root: Path, inventory: pd.DataFrame) -> None:
    failures: list[str] = []
    for row in inventory.to_dict("records"):
        path = root / str(row["relative_path"])
        if not path.exists():
            failures.append(f"missing:{row['relative_path']}")
        elif path.stat().st_size != int(row["size_bytes"]):
            failures.append(f"size:{row['relative_path']}")
        elif sha256_file(path) != str(row["sha256"]):
            failures.append(f"sha:{row['relative_path']}")
    if failures:
        raise RuntimeError(f"Inventory verification failed: {failures[:10]}")


@contextmanager
def single_writer_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(f"Another Stage4.2C writer appears active: {path}") from exc
    try:
        os.write(fd, f"pid={os.getpid()}\nstarted={utc_now_iso()}\n".encode())
        os.close(fd)
        yield
    finally:
        path.unlink(missing_ok=True)


def run_id(prefix: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    entropy = hashlib.sha256(f"{prefix}|{stamp}|{os.getpid()}".encode()).hexdigest()[:12]
    return f"{prefix}_{stamp}_{entropy}"
