from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import polars as pl
from typer.testing import CliRunner

from package.cli import app


def _write_smoke_config(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        """
[data]
dataset_name = "smoke_ds"
target_name = "target"

[eval]
random_state = 123
test_size = 0.2
cv_splits = 3
scoring = "neg_root_mean_squared_error"
optimize_type = "max"
n_jobs = 1

[search]
model_keys = ["dummy_regressor"]

[predict]
model_id = "best"
""".lstrip()
    )


def _write_dummy_csv(path: Path, *, n_rows: int = 120) -> None:
    rng = np.random.default_rng(0)
    df = pl.DataFrame(
        {
            "x1": rng.normal(size=n_rows),
            "x2": rng.normal(size=n_rows),
            "target": rng.normal(size=n_rows),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(path)


def test_smoke_pipeline(tmp_path: Path, monkeypatch):
    """
    End-to-end smoke test:
    register-data -> preprocess -> split -> search -> predict -> models
    Runs in a temp CWD so all outputs land under tmp_path, then cleans them.
    """
    runner = CliRunner()
    monkeypatch.chdir(tmp_path)

    cfg_path = tmp_path / "configs" / "smoke.toml"
    src_csv = tmp_path / "dummy.csv"
    _write_smoke_config(cfg_path)
    _write_dummy_csv(src_csv)

    try:
        r = runner.invoke(app, ["register-data", "-c", str(cfg_path), "--in", str(src_csv)])
        assert r.exit_code == 0, r.output

        r = runner.invoke(app, ["preprocess", "-c", str(cfg_path)])
        assert r.exit_code == 0, r.output

        r = runner.invoke(app, ["split", "-c", str(cfg_path), "--stage", "pre"])
        assert r.exit_code == 0, r.output

        r = runner.invoke(app, ["search", "-c", str(cfg_path)])
        assert r.exit_code == 0, r.output

        r = runner.invoke(app, ["predict", "-c", str(cfg_path), "--no-plots"])
        assert r.exit_code == 0, r.output

        r = runner.invoke(app, ["models", "-c", str(cfg_path), "--top", "5"])
        assert r.exit_code == 0, r.output

        # Assertions on artifacts
        dataset_root = tmp_path / "data" / "smoke_ds"
        assert (dataset_root / "raw" / "full.parquet").exists()
        assert (dataset_root / "preprocessed" / "full.parquet").exists()
        assert (dataset_root / "preprocessed" / "train.parquet").exists()
        assert (dataset_root / "preprocessed" / "test.parquet").exists()

        cv_summary = tmp_path / "outputs" / "gridsearch" / "cv_summary.json"
        assert cv_summary.exists()

        index_path = tmp_path / "outputs" / "models" / "index.json"
        assert index_path.exists()
        index = json.loads(index_path.read_text())
        assert len(index) >= 1
        model_dir = tmp_path / "outputs" / "models"
        assert any(p.suffix == ".skops" for p in model_dir.iterdir())
        best_id = index[-1]["id"]  # last registered; sufficient for smoke
        metrics_dir = tmp_path / "outputs" / "metrics" / best_id
        assert (metrics_dir / "predictions.parquet").exists()
        assert (metrics_dir / "metrics.json").exists()

    finally:
        shutil.rmtree(tmp_path / "data", ignore_errors=True)
        shutil.rmtree(tmp_path / "outputs", ignore_errors=True)
