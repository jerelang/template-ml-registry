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
    optimize_type: Literal["max", "min"] = "max"  # max or min scoring
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


DEFAULT_CONFIG = Config()


def _validate_config(cfg: Config) -> None:
    """Validation for configs."""
    issues: list[str] = []

    if not (0.0 < cfg.test_size < 1.0):
        issues.append(f"[eval].test_size must be in (0, 1); got {cfg.test_size!r}.")
    if cfg.cv_splits < 2:
        issues.append(f"[eval].cv_splits must be >= 2; got {cfg.cv_splits!r}.")
    if cfg.optimize_type not in ("max", "min"):
        issues.append(f"[eval].optimize_type must be 'max' or 'min'; got {cfg.optimize_type!r}.")
    if cfg.n_jobs == 0:
        issues.append("[eval].n_jobs must not be 0 (use -1 for all cores or a positive int).")

    if not cfg.dataset_name.strip():
        issues.append("[data].dataset_name must be non-empty.")
    if not cfg.target_name.strip():
        issues.append("[data].target_name must be non-empty.")

    if not cfg.model_keys:
        issues.append("[search].model_keys must contain at least one model key.")
    else:
        dupes = sorted({k for k in cfg.model_keys if cfg.model_keys.count(k) > 1})
        if dupes:
            issues.append(f"[search].model_keys contains duplicates: {dupes}.")

    if issues:
        msg = "Invalid config:\n" + "\n".join(f"- {i}" for i in issues)
        raise ValueError(msg)


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
        scoring=str(eval_.get("scoring", DEFAULT_CONFIG.scoring)).strip(),
        optimize_type=str(eval_.get("optimize_type", DEFAULT_CONFIG.optimize_type)).lower(),
        n_jobs=int(eval_.get("n_jobs", DEFAULT_CONFIG.n_jobs)),
        dataset_name=str(data_.get("dataset_name", DEFAULT_CONFIG.dataset_name)).strip(),
        target_name=str(data_.get("target_name", DEFAULT_CONFIG.target_name)).strip(),
        model_keys=search_.get("model_keys", list(DEFAULT_CONFIG.model_keys)),
    )
    _validate_config(cfg)
    return cfg, raw
