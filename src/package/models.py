from __future__ import annotations

from typing import Dict

from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import Config


def build_models(cfg: Config) -> Dict[str, Pipeline]:
    """
    Placeholder. Define all your models here as Pipelines.
    """
    models: Dict[str, Pipeline] = {}

    models["<modelname>"] = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                DummyRegressor(  # Dummy Estimator
                    strategy="mean",
                ),
            ),
        ]
    )
    return {k: v for k, v in models.items() if k in cfg.model_keys}
