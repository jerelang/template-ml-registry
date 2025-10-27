from __future__ import annotations

from typing import Dict

import polars as pl
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline

from .config import Config

""" Placeholder. Fill per model key."""
GRID_SPACES: dict[str, dict] = {
    "<modelname>": {},
}


def run_grid_search(
    cfg: Config,
    X_train,
    y_train,
    models: Dict[str, Pipeline],
) -> tuple[Dict[str, GridSearchCV], pl.DataFrame]:
    cv = KFold(n_splits=cfg.cv_splits, shuffle=True, random_state=cfg.random_state)

    results: Dict[str, GridSearchCV] = {}
    rows: list[dict] = []

    for name, estimator in models.items():
        if name not in GRID_SPACES:
            continue
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

    cv_summary = (
        pl.DataFrame(rows).sort("best_cv", descending=True)
        if rows
        else pl.DataFrame({"model": [], "cv_score_type": [], "best_cv": [], "best_params": []})
    )
    return results, cv_summary
