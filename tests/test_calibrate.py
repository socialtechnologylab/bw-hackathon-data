"""Tests for the calibration step.

The pure function `compute_baseline_score(mae)` is trivially `round(2*mae, 3)`
and uses the same formula the endpoint uses (score = max(0, 1 - mae/baseline)).

`train_and_score(X_train, y_train, X_test, y_test, target_col)` trains a
LightGBM with the participant baseline's params and returns MAE.

The fixture parquets are tiny synthetic data — enough to verify the
function runs and returns a sensible number.
"""

import polars as pl

from bw_hackathon_data.calibrate import compute_baseline_score, train_and_score


def test_compute_baseline_score_rounds_to_three_decimals():
    assert compute_baseline_score(0.347) == 0.694
    assert compute_baseline_score(0.0) == 0.0


def test_train_and_score_runs_on_tiny_data():
    """Train + predict on a trivial linear relationship; assert MAE is small."""
    # y = ghi_fcst / 10 + epsilon (linear in one feature).
    rows_train = [
        {
            "timestamp": f"2025-01-{d:02d}T00:00:00+00:00",
            "ghi_fcst": float(d * 10),
            "t2m_fcst": 5.0,
            "wind10m_fcst": 4.0,
            "cloud_cover_fcst": 0.5,
            "hour": 0,
            "dow": 0,
            "month": 1,
        }
        for d in range(1, 21)
    ]
    y_train_rows = [
        {"timestamp": r["timestamp"], "solar_mwh": r["ghi_fcst"] / 10.0} for r in rows_train
    ]
    rows_test = [
        {
            "timestamp": f"2025-02-{d:02d}T00:00:00+00:00",
            "ghi_fcst": float(d * 10),
            "t2m_fcst": 5.0,
            "wind10m_fcst": 4.0,
            "cloud_cover_fcst": 0.5,
            "hour": 0,
            "dow": 0,
            "month": 2,
        }
        for d in range(1, 11)
    ]
    y_test = [r["ghi_fcst"] / 10.0 for r in rows_test]

    mae = train_and_score(
        x_train=pl.DataFrame(rows_train),
        y_train=pl.DataFrame(y_train_rows),
        x_test=pl.DataFrame(rows_test),
        y_test=y_test,
        target_column="solar_mwh",
    )

    # Should be small; LightGBM on a linear single-feature relationship.
    assert mae < 1.0
    assert mae >= 0.0
