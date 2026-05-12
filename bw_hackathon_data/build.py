"""Per-task feature + target join.

`build_task` takes a target time series + a GFS cache and produces the
X_train / y_train / X_test parquets plus a ground_truth dict keyed by
test-window timestamp.

Drops rows where the target is missing or any feature is unavailable
(no aligned GFS cycle, or the cycle is missing the required fxx).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime

import polars as pl

from bw_hackathon_data.align import align_features

_X_SCHEMA = {
    "timestamp": pl.Utf8,
    "ghi_fcst": pl.Float64,
    "t2m_fcst": pl.Float64,
    "wind10m_fcst": pl.Float64,
    "cloud_cover_fcst": pl.Float64,
    "hour": pl.Int64,
    "dow": pl.Int64,
    "month": pl.Int64,
}


@dataclass
class BuildResult:
    """Output of build_task — parquets + telemetry."""

    x_train: pl.DataFrame
    y_train: pl.DataFrame
    x_test: pl.DataFrame
    ground_truth: dict[str, float]
    drop_count: int
    total_count: int
    task_id: str
    target_column: str


def build_task(
    *,
    target_df: pl.DataFrame,
    gfs_cache: Mapping[datetime, pl.DataFrame],
    lead_hours: int,
    task_id: str,
    target_column: str,
    train_window: tuple[datetime, datetime],
    test_window: tuple[datetime, datetime],
) -> BuildResult:
    """Join the target with aligned GFS features, split by window."""
    rows_x: list[dict] = []
    rows_y: list[dict] = []
    rows_test_x: list[dict] = []
    ground_truth: dict[str, float] = {}

    train_start, train_end = train_window
    test_start, test_end = test_window
    total = 0
    drops = 0

    for ts_str, val in zip(
        target_df["timestamp"].to_list(),
        target_df["value"].to_list(),
        strict=True,
    ):
        t = datetime.fromisoformat(ts_str)
        in_train = train_start <= t < train_end
        in_test = test_start <= t < test_end
        if not (in_train or in_test):
            # Row falls outside both windows — irrelevant for this build.
            # Don't count it toward `total` so drop_rate has the right denominator.
            continue

        total += 1
        if val is None:
            drops += 1
            continue
        feats = align_features(t, lead_hours, gfs_cache)
        if feats is None:
            drops += 1
            continue

        x_row = {
            "timestamp": ts_str,
            "ghi_fcst": feats["ghi_fcst"],
            "t2m_fcst": feats["t2m_fcst"],
            "wind10m_fcst": feats["wind10m_fcst"],
            "cloud_cover_fcst": feats["cloud_cover_fcst"],
            "hour": t.hour,
            "dow": t.weekday(),
            "month": t.month,
        }

        if in_train:
            rows_x.append(x_row)
            rows_y.append({"timestamp": ts_str, target_column: float(val)})
        else:
            rows_test_x.append(x_row)
            ground_truth[ts_str] = float(val)

    schema_y = {"timestamp": pl.Utf8, target_column: pl.Float64}

    return BuildResult(
        x_train=pl.DataFrame(rows_x, schema=_X_SCHEMA),
        y_train=pl.DataFrame(rows_y, schema=schema_y),
        x_test=pl.DataFrame(rows_test_x, schema=_X_SCHEMA),
        ground_truth=ground_truth,
        drop_count=drops,
        total_count=total,
        task_id=task_id,
        target_column=target_column,
    )
