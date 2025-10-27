from __future__ import annotations

import json
import secrets
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path

import polars as pl
import skops.io as sio

from .config import Config


def _new_id(prefix: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{ts}_{prefix}_{secrets.token_hex(4)}"


def _score_value(rec: dict) -> float:
    v = rec.get("cv_score", None)
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("-inf")


def _jsonify(obj):
    """Recursively convert dataclasses, Paths, tuples/sets to JSON-safe types."""
    if is_dataclass(obj):
        obj = asdict(obj)
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def register(
    cfg: Config,
    *,
    model_key: str,
    estimator,
    params: dict,
    cv_score: float | None,
    data_digest: str,
    meta: dict | None = None,
) -> dict:
    cfg.out_models.mkdir(parents=True, exist_ok=True)

    index: list[dict] = []
    if cfg.index_path.exists():
        index = json.loads(cfg.index_path.read_text())

    model_id = _new_id(model_key)
    art_path = cfg.out_models / f"{model_id}.skops"
    sio.dump(estimator, art_path)

    rec = {
        "id": model_id,
        "model_key": model_key,
        "artifact_path": str(art_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "params": params,
        "cv_score_type": cfg.scoring,
        "cv_score": cv_score,
        "data_digest": data_digest,
        "cfg": _jsonify(cfg),
        "meta": meta or {},
    }
    index.append(rec)
    cfg.index_path.write_text(json.dumps(index, indent=2))

    best = max(index, key=lambda r: (_score_value(r), r["created_at"]))
    cfg.current_best_path.write_text(json.dumps({"id": best["id"]}, indent=2))
    return rec


def load_estimator(cfg: Config, model_id: str = "best"):
    """
    When model_id == 'best', choose the best model **for the current scorer**
    in cfg.scoring.
    """
    if not cfg.index_path.exists():
        raise FileNotFoundError("No index.json found. Run `package search` first.")

    index: list[dict] = json.loads(cfg.index_path.read_text())

    if model_id != "best":
        rec = next((r for r in index if r["id"] == model_id), None)
        if rec is None:
            raise FileNotFoundError(f"Model id not found: {model_id}")
        est = sio.load(Path(rec["artifact_path"]))
        return rec, est

    # score method aware
    scorer = cfg.scoring

    candidates = [
        r for r in index if r.get("cv_score_type") == scorer and r.get("cv_score") is not None
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No models in the registry were trained/evaluated with scoring='{scorer}'.\n"
            f"Set [eval].scoring = '{scorer}' in your TOML and run `package search` to register models for that scorer."
        )

    best = max(candidates, key=lambda r: (_score_value(r), r["created_at"]))
    est = sio.load(Path(best["artifact_path"]))
    return best, est


def models_table(cfg: Config, top: int | None = None) -> pl.DataFrame:
    if not cfg.index_path.exists():
        return pl.DataFrame({"id": [], "model": [], "cv_score": [], "created_at": [], "params": []})

    index: list[dict] = json.loads(cfg.index_path.read_text())
    rows = [
        {
            "id": r["id"],
            "model": r.get("model_key"),
            "cv_score_type": r.get("cv_score_type"),
            "cv_score": r.get("cv_score"),
            "created_at": r.get("created_at"),
            "params": r.get("params", {}),
        }
        for r in index
    ]
    df = pl.DataFrame(rows).with_columns(pl.col("cv_score").cast(pl.Float64, strict=False))
    df = df.sort(["cv_score", "created_at"], descending=[True, True])
    if top:
        df = df.head(top)
    return df
