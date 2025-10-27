from __future__ import annotations

from pathlib import Path

from ..config import Config
from .io import cache_as_parquet, load_table


def preprocess_copy(cfg: Config) -> dict:
    src = cfg.path("raw", "full")
    dst = cfg.path("pre", "full")
    if not Path(src).exists():
        raise FileNotFoundError(f"RAW full not found at {src}. Run `register-data` first.")
    df = load_table(src)
    cache_as_parquet(df, dst)
    return {"src": str(src), "dst": str(dst), "rows": int(df.height), "cols": int(len(df.columns))}
