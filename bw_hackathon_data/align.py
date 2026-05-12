"""GFS forecast-alignment rule.

For a target timestamp t with lead time L hours, the rule picks the
latest GFS init cycle c such that c <= t - L hours. If the nominally
latest cycle isn't in the cache, fall back to the next earlier
available cycle.

Returned feature row sits at forecast hour fxx = (t - c) hours.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timedelta

import polars as pl


def latest_cycle_for(
    target: datetime,
    lead_hours: int,
    available: Iterable[datetime],
) -> datetime | None:
    """Latest cycle c in `available` with c <= target - lead_hours.

    Returns None if no such cycle exists.
    """
    cutoff = target - timedelta(hours=lead_hours)
    valid = [c for c in available if c <= cutoff]
    if not valid:
        return None
    return max(valid)


def align_features(
    target: datetime,
    lead_hours: int,
    gfs_cache: dict[datetime, pl.DataFrame],
) -> dict[str, float] | None:
    """Look up the feature row for `target` from the GFS cache.

    `gfs_cache` maps each cycle datetime to a DataFrame with columns
    (fxx, ghi_fcst, t2m_fcst, wind10m_fcst, cloud_cover_fcst).

    Returns a dict of the feature scalars for the chosen (cycle, fxx),
    or None if no valid cycle/fxx is available.
    """
    cycle = latest_cycle_for(target, lead_hours, gfs_cache.keys())
    if cycle is None:
        return None
    fxx = int((target - cycle).total_seconds() // 3600)
    df = gfs_cache[cycle]
    row = df.filter(pl.col("fxx") == fxx)
    if row.is_empty():
        return None
    record = row.to_dicts()[0]
    record.pop("fxx", None)
    return record
