"""Baseline LightGBM training + MAE → baseline_score derivation.

`train_and_score` mirrors the participant baseline (n_estimators=200,
default LightGBM hyperparameters) so the calibration matches what
participants will see when they run eval.py.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import polars as pl
from lightgbm import LGBMRegressor

_FEATURE_COLS = [
    "ghi_fcst",
    "t2m_fcst",
    "wind10m_fcst",
    "cloud_cover_fcst",
    "hour",
    "dow",
    "month",
]


def compute_baseline_score(mae: float) -> float:
    """Map observed MAE to a baseline_score such that the LightGBM score = 0.5.

    Concretely: baseline_score = 2 * MAE, rounded to 3 decimals.
    The endpoint's score function is max(0, 1 - mae / baseline_score), so
    a model achieving MAE = baseline_score / 2 lands at score = 0.5.
    """
    return round(2.0 * mae, 3)


def train_and_score(
    *,
    x_train: pl.DataFrame,
    y_train: pl.DataFrame,
    x_test: pl.DataFrame,
    y_test: list[float],
    target_column: str,
) -> float:
    """Train LightGBM on x_train / y_train, predict on x_test, return MAE.

    Params match the frozen participant baseline (`eval.py`) exactly so the
    calibrated baseline_score lines up with what participants observe.
    """
    model = LGBMRegressor(n_estimators=200, verbosity=-1)
    model.fit(
        x_train.select(_FEATURE_COLS).to_pandas(),
        y_train[target_column].to_list(),
    )
    preds = cast(np.ndarray, model.predict(x_test.select(_FEATURE_COLS).to_pandas()))
    return float(np.mean(np.abs(np.asarray(y_test) - np.asarray(preds))))
