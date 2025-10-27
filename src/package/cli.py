from __future__ import annotations

from pathlib import Path
from typing import Optional

import polars as pl
import typer

from .config import load_config
from .data.io import ingest, write_split_for_stage
from .data.preprocessing import preprocess_copy
from .registry import models_table
from .runner import predict as cmd_predict
from .runner import search as cmd_search
from .runner import train as cmd_train

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command(help="Write RAW full.parquet from a CSV/Parquet file.")
def register_data(
    config: Path = typer.Option(Path("configs/default.toml"), "--config", "-c"),
    src: Path = typer.Option(..., "--in", help="Source CSV/Parquet file"),
):
    """Ingest a CSV/Parquet file into RAW/full.parquet under the dataset root."""
    cfg, _ = load_config(config)
    if not src.exists():
        raise typer.BadParameter(f"Source file does not exist: {src}")
    dst = cfg.path("raw", "full")
    dst.parent.mkdir(parents=True, exist_ok=True)
    out_path = ingest(src, dst)
    typer.echo(f"Wrote: {out_path}")
    typer.echo(f"Dataset root: {cfg.dataset_root}")


@app.command(help="RAW full -> PREPROCESSED full.")
def preprocess(
    config: Path = typer.Option(Path("configs/default.toml"), "--config", "-c"),
):
    """Preprocess and copy data/raw/full.parquet to data/preprocessed/full.parquet."""
    cfg, _ = load_config(config)
    meta = preprocess_copy(cfg)
    typer.echo(meta)


@app.command(help="FULL -> TRAIN/TEST for a stage (raw|pre).")
def split(
    config: Path = typer.Option(Path("configs/default.toml"), "--config", "-c"),
    stage: str = typer.Option("pre", "--stage", "-s"),
):
    """Materialize a random train/test split for the selected stage, random_state and split ratio (configured in config/.toml)."""
    cfg, _ = load_config(config)
    meta = write_split_for_stage(cfg, stage=stage)
    typer.echo(meta)


@app.command(help="GridSearchCV on PRE train; register best.")
def search(
    config: Path = typer.Option(Path("configs/default.toml"), "--config", "-c"),
):
    """Run grid search over configured models and register each best estimator."""
    cfg, _ = load_config(config)
    cv_df = cmd_search(cfg)
    with pl.Config(tbl_rows=-1):
        typer.echo(cv_df)


@app.command(help="Deterministic train on PRE train; register.")
def train(
    config: Path = typer.Option(Path("configs/default.toml"), "--config", "-c"),
):
    """Fit a specific model spec (or the current best), then register the fitted artifact."""
    cfg, raw = load_config(config)
    tr = raw.get("train", {}) or {}
    spec = (
        {"model_key": tr["model_key"], "params": tr.get("params", {})}
        if "model_key" in tr
        else None
    )
    res = cmd_train(cfg, spec=spec)
    with pl.Config(tbl_rows=-1):
        typer.echo(res)


@app.command(help="Predict with best or a specific model id on PRE test.")
def predict(
    config: Path = typer.Option(Path("configs/default.toml"), "--config", "-c"),
    model_id: Optional[str] = typer.Option(None, "--model-id"),
    plots: bool = typer.Option(False, "--plots/--no-plots"),
    plots_out: Optional[Path] = typer.Option(None, "--plots-out"),
):
    """Run predictions on PRE/test, save predictions/metrics, and optionally plots."""
    cfg, raw = load_config(config)
    mid = model_id or (raw.get("predict", {}) or {}).get("model_id", "best")
    preds, metrics, plot_paths = cmd_predict(
        cfg, model_id=mid, make_plots=plots, plots_out=plots_out
    )

    typer.echo("=== Predictions (head) ===")
    with pl.Config(tbl_rows=10):
        typer.echo(preds)
    if metrics:
        typer.echo("=== Metrics ===")
        typer.echo(metrics)
    if plot_paths:
        typer.echo("=== Plots ===")
        for name, path in plot_paths.items():
            typer.echo(f"{name}: {path}")


@app.command(help="List registered models.")
def models(
    config: Path = typer.Option(Path("configs/default.toml"), "--config", "-c"),
    top: Optional[int] = typer.Option(None, "--top"),
):
    """Show the registry table (optionally top-K by score/recency)."""
    cfg, _ = load_config(config)
    df_models = models_table(cfg, top)
    typer.echo(df_models)


if __name__ == "__main__":
    app()
