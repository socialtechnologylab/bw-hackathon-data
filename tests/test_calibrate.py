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
    """Train + predict on a learnable relationship; assert MAE is bounded.

    300 train rows so LightGBM with default min_child_samples=20 has plenty
    of room to grow trees. Test points share the train feature range so
    leaf averaging works in the model's favour. We must NOT lower
    min_child_samples just to fit fewer rows — the calibration has to
    mirror the frozen participant baseline exactly, which uses LightGBM
    defaults (n_estimators=200, verbosity=-1, everything else default).
    """

    # y = ghi_fcst / 10. ghi spans 10..3000 in train; test points are
    # a 50-row sample drawn from the same range.
    def _row(ghi: float, idx: int) -> dict:
        return {
            "timestamp": (
                f"2025-{(idx - 1) // 28 + 1:02d}-{((idx - 1) % 28) + 1:02d}T00:00:00+00:00"
            ),
            "ghi_fcst": ghi,
            "t2m_fcst": 5.0,
            "wind10m_fcst": 4.0,
            "cloud_cover_fcst": 0.5,
            "hour": 0,
            "dow": 0,
            "month": (idx - 1) // 28 + 1,
        }

    rows_train = [_row(float(d * 10), d) for d in range(1, 301)]  # ghi 10..3000
    y_train_rows = [
        {"timestamp": r["timestamp"], "solar_mwh": r["ghi_fcst"] / 10.0} for r in rows_train
    ]
    # Test fixture: ghi spans the train range (15, 75, 135, ... up to ~2975).
    rows_test = [_row(float(15 + d * 60), d + 1000) for d in range(50)]
    y_test = [r["ghi_fcst"] / 10.0 for r in rows_test]

    mae = train_and_score(
        x_train=pl.DataFrame(rows_train),
        y_train=pl.DataFrame(y_train_rows),
        x_test=pl.DataFrame(rows_test),
        y_test=y_test,
        target_column="solar_mwh",
    )

    # y range in test is 1.5..299.1; MAE under 10 = under 3.5% of the range,
    # which is "the model is learning the linear relationship". Loose enough
    # to survive LightGBM's leaf-averaging on synthetic data, tight enough
    # to catch a function that returns NaN, +inf, or random noise.
    assert 0.0 <= mae < 10.0
