from __future__ import annotations

import tomllib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class Config:
    """Runtime configuration and I/O layout for data, models, and CV settings."""

    random_state: int = 123
    test_size: float = 0.20
    cv_splits: int = 5
    scoring: str = "neg_root_mean_squared_error"  # generic default
    n_jobs: int = -1

    dataset_name: str = "example_data"
    target_name: str = "target"

    out_search: Path = Path("outputs") / "gridsearch"
    out_models: Path = Path("outputs") / "models"

    model_keys: tuple[str, ...] = ("logistic", "svm_rbf", "lgbm")

    @property
    def dataset_root(self) -> Path:
        return Path("data") / self.dataset_name

    def stage_dir(self, stage: Literal["raw", "pre"]) -> Path:
        """Return (and create) the directory for a stage ('raw' or 'preprocessed')."""
        name = "preprocessed" if stage == "pre" else "raw"
        d = self.dataset_root / name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def path(self, stage: Literal["raw", "pre"], split: Literal["full", "train", "test"]) -> Path:
        """Return the default path for a stage/split Parquet file."""
        return self.stage_dir(stage) / f"{split}.parquet"

    @property
    def index_path(self) -> Path:
        """Path to the model registry index (JSON)."""
        return self.out_models / "index.json"

    @property
    def current_best_path(self) -> Path:
        """Path to the file storing the current best model id."""
        return self.out_models / "current_best.json"


DEFAULT_CONFIG = Config()


def load_config(path: Path) -> tuple[Config, dict]:
    """Load TOML config and return a populated Config plus the raw dict."""
    with path.open("rb") as f:
        raw = tomllib.load(f)

    eval_ = raw.get("eval", {}) or {}
    data_ = raw.get("data", {}) or {}
    search_ = raw.get("search", {}) or {}

    cfg = replace(
        DEFAULT_CONFIG,
        random_state=int(eval_.get("random_state", DEFAULT_CONFIG.random_state)),
        test_size=float(eval_.get("test_size", DEFAULT_CONFIG.test_size)),
        cv_splits=int(eval_.get("cv_splits", DEFAULT_CONFIG.cv_splits)),
        scoring=str(eval_.get("scoring", DEFAULT_CONFIG.scoring)),
        n_jobs=int(eval_.get("n_jobs", DEFAULT_CONFIG.n_jobs)),
        dataset_name=str(data_.get("dataset_name", DEFAULT_CONFIG.dataset_name)),
        target_name=str(data_.get("target_name", DEFAULT_CONFIG.target_name)),
        model_keys=tuple(search_.get("model_keys", list(DEFAULT_CONFIG.model_keys))),
    )
    return cfg, raw
