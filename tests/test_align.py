"""Tests for the GFS forecast-alignment rule.

Rule: for target timestamp t with lead time L hours, pick the latest GFS
init cycle c such that c <= t - L. If c is missing from the available set,
fall back to the next earlier available cycle.
"""

from datetime import UTC, datetime

from bw_hackathon_data.align import align_features, latest_cycle_for


def d(*args):
    return datetime(*args, tzinfo=UTC)


# ── latest_cycle_for ──────────────────────────────────────────────────────────


def test_latest_cycle_for_typical_day_ahead():
    """t = 2025-05-01T12:00Z, L = 24h → 2025-04-30T12:00Z."""
    available = {
        d(2025, 4, 30, 0),
        d(2025, 4, 30, 6),
        d(2025, 4, 30, 12),
        d(2025, 4, 30, 18),
        d(2025, 5, 1, 0),
    }
    assert latest_cycle_for(d(2025, 5, 1, 12), 24, available) == d(2025, 4, 30, 12)


def test_latest_cycle_for_typical_two_hour_ahead():
    """t = 2025-05-01T05:00Z, L = 2h → latest cycle <= 03:00 = 2025-05-01T00:00Z."""
    available = {
        d(2025, 4, 30, 18),
        d(2025, 5, 1, 0),
        d(2025, 5, 1, 6),
    }
    assert latest_cycle_for(d(2025, 5, 1, 5), 2, available) == d(2025, 5, 1, 0)


def test_latest_cycle_for_boundary_exactly_l_hours_before():
    """Cycle exactly at t - L is valid (>=, not strict >)."""
    available = {d(2025, 4, 30, 12)}
    assert latest_cycle_for(d(2025, 5, 1, 12), 24, available) == d(2025, 4, 30, 12)


def test_latest_cycle_for_falls_back_when_latest_missing():
    """Latest valid cycle missing → next-earlier."""
    available = {d(2025, 4, 30, 0), d(2025, 4, 30, 6)}
    # nominally would pick 2025-04-30T12 (24h before 2025-05-01T12); it's missing.
    assert latest_cycle_for(d(2025, 5, 1, 12), 24, available) == d(2025, 4, 30, 6)


def test_latest_cycle_for_returns_none_when_no_valid_cycle():
    """No cycle in the available set is at or before t - L."""
    available = {d(2025, 5, 1, 0)}
    assert latest_cycle_for(d(2025, 5, 1, 12), 24, available) is None


def test_latest_cycle_for_returns_none_on_empty_set():
    assert latest_cycle_for(d(2025, 5, 1, 12), 24, set()) is None


# ── align_features ────────────────────────────────────────────────────────────


def test_align_features_returns_row_at_correct_fxx():
    """The feature row is at fxx = (t - cycle).hours."""
    import polars as pl

    cycle = d(2025, 4, 30, 12)
    cache_for_cycle = pl.DataFrame(
        {
            "fxx": [0, 12, 24, 30],
            "ghi_fcst": [0.0, 600.0, 50.0, 0.0],
            "t2m_fcst": [8.0, 15.0, 10.0, 9.0],
            "wind10m_fcst": [3.0, 5.0, 4.0, 3.5],
            "cloud_cover_fcst": [0.5, 0.2, 0.6, 0.8],
        }
    )
    gfs_cache = {cycle: cache_for_cycle}

    result = align_features(d(2025, 5, 1, 12), lead_hours=24, gfs_cache=gfs_cache)

    assert result is not None
    assert result["ghi_fcst"] == 50.0
    assert result["t2m_fcst"] == 10.0
    assert result["wind10m_fcst"] == 4.0
    assert result["cloud_cover_fcst"] == 0.6


def test_align_features_returns_none_when_no_cycle():
    """No valid cycle in cache → None."""
    import polars as pl

    gfs_cache = {d(2025, 5, 1, 0): pl.DataFrame({"fxx": [0]})}
    assert align_features(d(2025, 5, 1, 12), 24, gfs_cache) is None


def test_align_features_returns_none_when_fxx_missing_in_cache():
    """Cycle exists but the required fxx row isn't there."""
    import polars as pl

    cycle = d(2025, 4, 30, 12)
    gfs_cache = {
        cycle: pl.DataFrame(
            {
                "fxx": [0, 6, 12],
                "ghi_fcst": [0.0, 200.0, 600.0],
            }
        )
    }
    # Need fxx=24, not in cache.
    assert align_features(d(2025, 5, 1, 12), 24, gfs_cache) is None
