from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(block_size), b""):
            h.update(block)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding=encoding, delete=False, dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    ) as tmp:
        tmp.write(text)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2, default=str))


def atomic_write_csv(df: pd.DataFrame, path: Path, *, index: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.parent / f".{path.name}.{os.getpid()}.tmp"
    df.to_csv(tmp_path, index=index, encoding="utf-8-sig")
    os.replace(tmp_path, path)


def atomic_write_parquet(df: pd.DataFrame, path: Path, *, index: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.parent / f".{path.name}.{os.getpid()}.tmp"
    try:
        df.to_parquet(tmp_path, index=index)
    except ImportError as exc:
        raise RuntimeError(
            "Parquet output requires pyarrow. Install requirements_stage4.txt before the official run."
        ) from exc
    os.replace(tmp_path, path)


def read_table(path: Path, **kwargs: Any) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        try:
            return pd.read_parquet(path, **kwargs)
        except ImportError as exc:
            raise RuntimeError(
                f"Reading {path.name} requires pyarrow. Install requirements_stage4.txt."
            ) from exc
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path, low_memory=False, **kwargs)
    if suffix in {".json"}:
        return pd.DataFrame(json.loads(path.read_text(encoding="utf-8")))
    raise ValueError(f"Unsupported table format: {path}")


def coalesce_column(df: pd.DataFrame, aliases: list[str], *, required: bool = True) -> str | None:
    lower_map = {str(c).lower(): str(c) for c in df.columns}
    for alias in aliases:
        if alias in df.columns:
            return alias
        found = lower_map.get(alias.lower())
        if found is not None:
            return found
    if required:
        raise KeyError(f"None of the required columns exist: {aliases}")
    return None


def as_numeric(series: pd.Series, *, fill: float | None = None) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce")
    if fill is not None:
        out = out.fillna(fill)
    return out


def normalize_0_1(series: pd.Series, *, invert: bool = False) -> pd.Series:
    x = as_numeric(series)
    finite = x[np.isfinite(x)]
    if finite.empty or float(finite.max()) == float(finite.min()):
        out = pd.Series(np.zeros(len(x), dtype=float), index=x.index)
    else:
        out = (x - finite.min()) / (finite.max() - finite.min())
        out = out.clip(0.0, 1.0).fillna(0.0)
    return 1.0 - out if invert else out


def robust_percentile(series: pd.Series) -> pd.Series:
    x = as_numeric(series)
    return x.rank(method="average", pct=True).fillna(0.5)


def gini(values: np.ndarray | pd.Series) -> float:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    if np.min(x) < 0:
        x = x - np.min(x)
    total = x.sum()
    if total <= 0:
        return 0.0
    x = np.sort(x)
    n = x.size
    return float((2.0 * np.sum((np.arange(1, n + 1)) * x) / (n * total)) - (n + 1) / n)


def hhi(values: np.ndarray | pd.Series) -> float:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    total = x.sum()
    if total <= 0:
        return 0.0
    shares = x / total
    return float(np.square(shares).sum())


def ensure_relative_to(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def inventory_files(root: Path, *, exclude_names: set[str] | None = None) -> pd.DataFrame:
    exclude_names = exclude_names or set()
    rows: list[dict[str, Any]] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.name not in exclude_names):
        rows.append(
            {
                "relative_path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return pd.DataFrame(rows)


@contextlib.contextmanager
def single_writer_lock(lock_path: Path, *, stale_after_seconds: int = 12 * 3600) -> Iterator[None]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    owner_token = uuid.uuid4().hex
    payload = {
        "pid": os.getpid(),
        "owner_token": owner_token,
        "created_at": utc_now_iso(),
        "cwd": str(Path.cwd()),
    }
    encoded = (json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        age = time.time() - lock_path.stat().st_mtime
        content = lock_path.read_text(encoding="utf-8", errors="replace")
        stale_note = (
            f" Lock age {age:.0f}s exceeds the {stale_after_seconds}s stale threshold, "
            "but automatic deletion is forbidden; verify the recorded process and remove "
            "the stale lock explicitly."
            if age > stale_after_seconds
            else ""
        )
        raise RuntimeError(
            f"Stage 4 writer lock already exists: {lock_path}\n{content}{stale_note}"
        ) from exc
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        lock_path.unlink(missing_ok=True)
        raise
    try:
        yield
    finally:
        try:
            current = json.loads(lock_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            current = None
        if isinstance(current, dict) and current.get("owner_token") == owner_token:
            lock_path.unlink(missing_ok=True)


def copy_atomic(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.parent / f".{dst.name}.{os.getpid()}.tmp"
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)


def floor_fraction(value: int | float, fraction: float) -> int:
    return int(np.floor(float(value) * float(fraction) + 1e-9))


def ceil_fraction(value: int | float, fraction: float) -> int:
    return int(np.ceil(float(value) * float(fraction) - 1e-9))
