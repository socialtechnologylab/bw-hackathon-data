"""Baseline LightGBM training → MAE.

`train_and_score` mirrors the participant baseline (n_estimators=200,
default LightGBM hyperparameters) so the reported MAE matches what
participants will see when they run eval.py on the shipped parquets.

The MAE is informational — used to populate the per-task README's
"what good looks like" line. The endpoint compares raw predictions to
ground truth; there's no normalization or baseline_score derivation.
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
    reported MAE matches what a participant will see on first run.
    """
    model = LGBMRegressor(n_estimators=200, verbosity=-1)
    model.fit(
        x_train.select(_FEATURE_COLS).to_pandas(),
        y_train[target_column].to_list(),
    )
    preds = cast(np.ndarray, model.predict(x_test.select(_FEATURE_COLS).to_pandas()))
    return float(np.mean(np.abs(np.asarray(y_test) - np.asarray(preds))))
