from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from sklearn.model_selection import train_test_split

from ..config import Config


def cache_as_parquet(frame: pl.DataFrame | pl.LazyFrame, out: str | Path) -> Path:
    """
    Write parquet.
    - DataFrame -> write_parquet
    - LazyFrame -> sink_parquet (streaming/out-of-core when supported)
    """
    outp = Path(out)
    outp.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(frame, pl.LazyFrame):
        try:
            frame.sink_parquet(outp)
        except Exception:
            # Fallback: collect then write (still type-correct)
            frame.collect(engine="streaming").write_parquet(outp)
    else:
        frame.write_parquet(outp)

    return outp


def scan_table(
    path: str | Path,
    columns: list[str] | None = None,
    limit: int | None = None,
) -> pl.LazyFrame:
    """Scan (lazy) CSV/Parquet."""
    p = Path(path)
    lf = pl.scan_parquet(p) if p.suffix.lower() in {".parquet", ".pq"} else pl.scan_csv(p)

    if columns:
        existing = [c for c in columns if c in lf.columns]
        if existing:
            lf = lf.select([pl.col(c) for c in existing])

    if limit is not None:
        lf = lf.limit(limit)

    return lf


def load_table(
    path: str | Path,
    columns: list[str] | None = None,
    limit: int | None = None,
) -> pl.DataFrame:
    """
    Load into a DataFrame.
    """
    lf = scan_table(path, columns=columns, limit=limit)
    try:
        return lf.collect(engine="streaming")
    except Exception:
        return lf.collect()


def ingest(src_path: str | Path, dst_raw_full: str | Path) -> Path:
    """Write RAW/full.parquet from CSV/Parquet lazily."""
    lf = scan_table(src_path)
    return cache_as_parquet(lf, dst_raw_full)


def file_sha256(path: str | Path, chunk_size: int = 2**20) -> str:
    """Compute SHA-256 of a file in chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def load_Xy(
    path: str | Path,
    target_col: str = "target",
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return (X, y, feature_names)."""
    df = load_table(path)
    if target_col not in df.columns:
        raise ValueError(f"Missing target column '{target_col}' in {path}")

    feature_names = df.select(pl.exclude(target_col)).columns
    X = df.select(pl.exclude(target_col)).to_numpy()
    y = df[target_col].to_numpy()
    return X, y, feature_names


def _write_split(
    *,
    dataset_path: str | Path,
    train_path: str | Path,
    test_path: str | Path,
    test_size: float,
    random_state: int,
    target_col: str = "target",
) -> dict[str, Any]:
    """
    Eager sklearn-style split (exact sizes, uses train_test_split).
    """
    df = load_table(dataset_path)
    if target_col not in df.columns:
        raise ValueError(f"Missing target column '{target_col}' in {dataset_path}")

    idx = np.arange(df.height)
    tr_idx, te_idx = train_test_split(idx, test_size=test_size, random_state=random_state)

    df_train = df[tr_idx]
    df_test = df[te_idx]

    train_path = Path(train_path)
    test_path = Path(test_path)
    train_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.parent.mkdir(parents=True, exist_ok=True)

    df_train.write_parquet(train_path)
    df_test.write_parquet(test_path)

    return {
        "train_path": str(train_path),
        "test_path": str(test_path),
        "n_train": int(df_train.height),
        "n_test": int(df_test.height),
    }


def write_split_for_stage(
    cfg: Config, stage: str = "pre", target_col: str = "target"
) -> dict[str, Any]:
    """Split FULL into TRAIN/TEST for a given stage ('raw' or 'pre')."""
    stage = stage.lower()
    if stage not in {"raw", "pre", "preprocessed"}:
        raise ValueError("stage must be one of: 'raw', 'pre', 'preprocessed'")
    st = "pre" if stage in {"pre", "preprocessed"} else "raw"
    return _write_split(
        dataset_path=cfg.path(st, "full"),
        train_path=cfg.path(st, "train"),
        test_path=cfg.path(st, "test"),
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        target_col=target_col,
    )
