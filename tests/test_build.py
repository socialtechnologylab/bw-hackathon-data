"""Tests for per-task feature + target join.

build_task takes:
 - a target DataFrame (timestamp, value)
 - a dict of GFS cache DataFrames keyed by cycle datetime
 - a lead_hours, task_id, target_column_name, and the train/test window split

It returns a BuildResult with (X_train, y_train, X_test, ground_truth_dict,
drop_count, total_count).
"""

from datetime import UTC, datetime

import polars as pl

from bw_hackathon_data.build import build_task


def d(*args) -> datetime:
    return datetime(*args, tzinfo=UTC)


def _iso(*args) -> str:
    return d(*args).isoformat()


def _gfs_for_cycle(cycle: datetime, fxx_values: list[int]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "fxx": fxx_values,
            "ghi_fcst": [10.0 * f for f in fxx_values],
            "t2m_fcst": [5.0 + 0.1 * f for f in fxx_values],
            "wind10m_fcst": [4.0] * len(fxx_values),
            "cloud_cover_fcst": [0.5] * len(fxx_values),
        }
    )


def test_build_task_solar_join_and_split():
    """Targets at 00:00 + 01:00 + 02:00 → X_train + X_test split + ground_truth."""
    target_df = pl.DataFrame(
        {
            "timestamp": [_iso(2025, 1, 1, 0), _iso(2025, 1, 1, 1), _iso(2025, 1, 1, 2)],
            "value": [0.0, 5.0, 80.0],
        }
    )
    cycle = d(2024, 12, 31, 0)  # 24 h before 2025-01-01T00
    gfs_cache = {cycle: _gfs_for_cycle(cycle, [24, 25, 26])}

    result = build_task(
        target_df=target_df,
        gfs_cache=gfs_cache,
        lead_hours=24,
        task_id="solar-1d-ahead",
        target_column="solar_mwh",
        train_window=(d(2025, 1, 1, 0), d(2025, 1, 1, 2)),
        test_window=(d(2025, 1, 1, 2), d(2025, 1, 1, 3)),
    )

    # Train window [00, 02) → 00, 01 (2 rows). Test [02, 03) → 02 (1 row).
    assert result.x_train.height == 2
    assert result.x_test.height == 1
    assert result.y_train.height == 2

    # Columns
    expected_x_cols = [
        "timestamp",
        "ghi_fcst",
        "t2m_fcst",
        "wind10m_fcst",
        "cloud_cover_fcst",
        "hour",
        "dow",
        "month",
    ]
    assert result.x_train.columns == expected_x_cols
    assert result.x_test.columns == expected_x_cols
    assert result.y_train.columns == ["timestamp", "solar_mwh"]

    # Ground truth (test labels only).
    assert result.ground_truth == {_iso(2025, 1, 1, 2): 80.0}

    # Calendar features sanity.
    assert result.x_train["hour"].to_list() == [0, 1]
    assert result.x_train["dow"].to_list() == [2, 2]  # 2025-01-01 = Wednesday
    assert result.x_train["month"].to_list() == [1, 1]


def test_build_task_drops_row_when_no_features_available():
    """No cycle in cache for a target → drop the row, count it."""
    target_df = pl.DataFrame(
        {
            "timestamp": [_iso(2025, 1, 1, 0), _iso(2025, 1, 1, 1)],
            "value": [0.0, 5.0],
        }
    )
    cycle = d(2024, 12, 31, 0)
    # Cache only has fxx=24 (for the 00:00 target), nothing for 01:00 → second row dropped.
    gfs_cache = {cycle: _gfs_for_cycle(cycle, [24])}

    result = build_task(
        target_df=target_df,
        gfs_cache=gfs_cache,
        lead_hours=24,
        task_id="solar-1d-ahead",
        target_column="solar_mwh",
        train_window=(d(2025, 1, 1, 0), d(2025, 1, 1, 2)),
        test_window=(d(2025, 1, 1, 2), d(2025, 1, 1, 3)),
    )

    assert result.x_train.height == 1
    assert result.drop_count == 1
    assert result.total_count == 2


def test_build_task_drops_row_when_target_missing():
    """If the target value is None, drop the row."""
    target_df = pl.DataFrame(
        {
            "timestamp": [_iso(2025, 1, 1, 0), _iso(2025, 1, 1, 1)],
            "value": [0.0, None],
        }
    )
    cycle = d(2024, 12, 31, 0)
    gfs_cache = {cycle: _gfs_for_cycle(cycle, [24, 25])}

    result = build_task(
        target_df=target_df,
        gfs_cache=gfs_cache,
        lead_hours=24,
        task_id="solar-1d-ahead",
        target_column="solar_mwh",
        train_window=(d(2025, 1, 1, 0), d(2025, 1, 1, 2)),
        test_window=(d(2025, 1, 1, 2), d(2025, 1, 1, 3)),
    )

    assert result.x_train.height == 1
    assert result.drop_count == 1
