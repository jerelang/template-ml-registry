from __future__ import annotations

import numpy as np


def _is_numeric(arr: np.ndarray) -> bool:
    try:
        return np.issubdtype(np.asarray(arr).dtype, np.number)
    except Exception:
        return False


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """
    Placeholder, task-agnostic evaluation:
    - numeric targets -> RMSE + MAE
    - non-numeric targets -> exact-match accuracy
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError(
            f"y_true and y_pred have different lengths: {y_true.shape[0]} vs {y_pred.shape[0]}"
        )

    if _is_numeric(y_true) and _is_numeric(y_pred):
        diff = y_true - y_pred
        mae = float(np.nanmean(np.abs(diff)))
        rmse = float(np.sqrt(np.nanmean(diff**2)))
        return {"mae": mae, "rmse": rmse}

    # Fallback for labels / non-numeric targets
    acc = float(np.mean(y_true == y_pred))
    return {"accuracy": acc}
