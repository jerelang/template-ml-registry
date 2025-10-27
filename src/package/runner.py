from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from .config import Config
from .data import io as data_io
from .eval.metrics import evaluate
from .grid import GRID_SPACES, run_grid_search
from .models import build_models
from .registry import load_estimator, register


def search(cfg: Config) -> pl.DataFrame:
    data_path = cfg.path("pre", "train")
    X_tr, y_tr, feats = data_io.load_Xy(data_path, target_col=cfg.target_name)
    models = build_models(cfg)
    results, cv_summary = run_grid_search(cfg, X_tr, y_tr, models)

    for key, model in results.items():
        meta = {
            "feature_names": list(feats),
            "grid_space": GRID_SPACES.get(key),
        }
        register(
            cfg,
            model_key=key,
            estimator=model.best_estimator_,
            params=model.best_params_,
            cv_score=float(model.best_score_),
            data_digest=data_io.file_sha256(data_path),
            meta=meta,
        )
    cfg.out_search.mkdir(parents=True, exist_ok=True)
    cv_summary.write_json(cfg.out_search / "cv_summary.json")
    return cv_summary


def train(cfg: Config, spec: dict | None = None) -> pl.DataFrame:
    data_path = cfg.path("pre", "train")
    X_tr, y_tr, feats = data_io.load_Xy(data_path, target_col=cfg.target_name)

    if spec is None:
        rec, _ = load_estimator(cfg, model_id="best")
        model_key, params = rec["model_key"], rec["params"]
    else:
        model_key, params = spec["model_key"], spec["params"]

    model = build_models(cfg)[model_key]
    model.set_params(**params)
    model.fit(X_tr, y_tr)
    meta = {
        "feature_names": list(feats),
        "grid_space": GRID_SPACES.get(model_key),
    }
    reg = register(
        cfg,
        model_key=model_key,
        estimator=model,
        params=params,
        cv_score=None,
        data_digest=data_io.file_sha256(data_path),
        meta=meta,
    )
    return pl.DataFrame([{"id": reg["id"], "model": model_key}])


def predict(
    cfg: Config,
    *,
    model_id: str = "best",
    make_plots: bool = False,
    plots_out: Path | None = None,
) -> tuple[pl.DataFrame, dict, dict | None]:
    data_path = cfg.path("pre", "test")

    rec, est = load_estimator(cfg, model_id=model_id)
    X, y_true, _ = data_io.load_Xy(data_path, target_col=cfg.target_name)
    y_pred = est.predict(X)

    metrics = evaluate(y_true, y_pred)
    pred_df = pl.DataFrame({cfg.target_name: y_true, "pred": y_pred})

    # Save under outputs/metrics/<model_id or best-id>/
    mid_for_dir = rec["id"] if model_id == "best" else model_id
    out_dir = Path("outputs") / "metrics" / str(mid_for_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_df.write_parquet(out_dir / "predictions.parquet")

    metrics_path = out_dir / "metrics.json"
    with metrics_path.open("w") as f:
        json.dump(
            {
                **metrics,
                "model_id": mid_for_dir,
                "data_path": str(data_path),
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
            f,
            indent=2,
        )

    plot_paths: dict[str, Path] | None = None
    if make_plots:
        from .eval.plots import prediction_plots  # placeholder

        plots_dir = Path(plots_out) if plots_out else (out_dir / "plots")
        title_prefix = f"model={mid_for_dir}, data={Path(str(data_path)).name}"
        # The placeholder accepts arbitrary args/kwargs and returns {}
        plot_paths = prediction_plots(
            y_true=y_true, y_pred=y_pred, out_dir=plots_dir, title_prefix=title_prefix
        )

    return pred_df, metrics, plot_paths
