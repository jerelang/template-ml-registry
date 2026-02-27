from __future__ import annotations

from typing import Dict, Iterable

import polars as pl
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline

from .config import Config

""" Placeholder. Fill per model key with dictionaries like this {parameter name: [value range]}."""
GRID_SPACES: dict[str, dict] = {
    "dummy_regressor": {},  # smoke test dummy estimator, no parameters
    "<modelname>": {},
}


def validate_grid_spaces(model_keys: Iterable[str]) -> None:
    """Every requested model needs a grid-space entry."""
    keys = list(model_keys)
    missing = sorted(set(keys) - set(GRID_SPACES))
    if missing:
        raise KeyError(f"Missing GRID_SPACES entries for: {missing}.")


def run_grid_search(
    cfg: Config,
    X_train,
    y_train,
    models: Dict[str, Pipeline],
) -> tuple[Dict[str, GridSearchCV], pl.DataFrame]:
    """Run GridSearchCV per model, refit best on full training data, and return (fits, CV summary table)."""
    if not models:
        raise ValueError("No model names specified for grid search. Check [search].model_keys.")
    validate_grid_spaces(models.keys())
    cv = KFold(n_splits=cfg.cv_splits, shuffle=True, random_state=cfg.random_state)

    results: Dict[str, GridSearchCV] = {}
    rows: list[dict] = []

    for name, estimator in models.items():
        gs = GridSearchCV(
            estimator=estimator,
            param_grid=GRID_SPACES[name],
            scoring=cfg.scoring,
            cv=cv,
            n_jobs=cfg.n_jobs,
            refit=True,
            verbose=0,
        )
        gs.fit(X_train, y_train)
        results[name] = gs
        rows.append(
            {
                "model": name,
                "cv_score_type": cfg.scoring,
                "best_cv": float(gs.best_score_),
                "best_params": gs.best_params_,
            }
        )

    descending = cfg.optimize_type == "max"
    cv_summary = (
        pl.DataFrame(rows).sort("best_cv", descending=descending)
        if rows
        else pl.DataFrame({"model": [], "cv_score_type": [], "best_cv": [], "best_params": []})
    )
    return results, cv_summary
