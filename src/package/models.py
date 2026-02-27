from __future__ import annotations

from typing import Dict

from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import Config


def build_models(cfg: Config) -> Dict[str, Pipeline]:
    """Build and return all candidate Pipelines keyed by model name (filtered by config)."""
    models: Dict[str, Pipeline] = {}

    # Smoke tests dummy model
    models["dummy_regressor"] = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", DummyRegressor(strategy="mean")),
        ]
    )
    models["<modelname>"] = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", DummyRegressor(strategy="mean")),
        ]
    )
    if set(cfg.model_keys) - set(models.keys()):
        raise KeyError("Unknown model_keys in config.")
    return {k: v for k, v in models.items() if k in cfg.model_keys}
