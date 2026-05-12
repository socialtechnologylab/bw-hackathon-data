"""Tests for the NOAA ISD METAR parser.

ISD per-station-per-year files are space-delimited. The columns we care
about live at fixed positions:
  - YEAR  MO  DA  HR  MIN  : station report time (UTC)
  - TEMP  (tenths °C, +9999 if missing)

Real ISD lines look like (this is a synthetic but format-accurate sample):
  2025 01 01 00 00  ...  +0023 ...
meaning 2025-01-01T00:00 UTC, temp = 2.3 °C.
"""

import polars as pl

from bw_hackathon_data.isd import parse_isd_lines

FIXTURE_LINES = [
    # (year, mo, da, hr, min, ... padded ..., temp_tenths)
    "2025 01 01 00 00  064510 99999  +0023",
    "2025 01 01 00 20  064510 99999  +0021",  # later in same hour — kept if 00 closer
    "2025 01 01 01 00  064510 99999  +0018",
    "2025 01 01 02 30  064510 99999  +0015",  # 30 min off — ok closest-to-hour
    "2025 01 01 03 00  064510 99999  +9999",  # missing → dropped
    "2025 01 01 04 00  064510 99999  -0050",  # -5.0 C
]


def test_parse_isd_lines_extracts_hourly_temps():
    df = parse_isd_lines(FIXTURE_LINES)

    assert df.columns == ["timestamp", "value"]
    assert df["timestamp"].dtype == pl.Utf8
    assert df["value"].dtype == pl.Float64

    # Expect 4 rows: hours 00, 01, 02, 04.
    # Hour 00 has two candidates (min 0 and min 20); min 0 wins, so only one row.
    # The +9999 row at 03:00 is dropped (missing sentinel).
    assert df.height == 4
    # Hour 00 picks the obs at minute 00 (delta = 0 < 20).
    assert df.filter(pl.col("timestamp") == "2025-01-01T00:00:00+00:00")["value"][0] == 2.3
    assert df.filter(pl.col("timestamp") == "2025-01-01T01:00:00+00:00")["value"][0] == 1.8
    # Hour 02: only obs at 02:30, delta = 30 min ≤ 30 → kept and bucketed to 02:00.
    assert df.filter(pl.col("timestamp") == "2025-01-01T02:00:00+00:00")["value"][0] == 1.5
    # Hour 04 → negative temp parsing.
    assert df.filter(pl.col("timestamp") == "2025-01-01T04:00:00+00:00")["value"][0] == -5.0


def test_parse_isd_lines_drops_obs_more_than_30_min_off_hour():
    """An obs at HH:31 is closer to the next hour than the current one;
    we drop it (the next-hour candidate may be a better observation that
    we'd pick instead)."""
    lines = ["2025 01 01 05 31  064510 99999  +0100"]
    df = parse_isd_lines(lines)
    assert df.height == 0


def test_parse_isd_lines_handles_empty_input():
    assert parse_isd_lines([]).height == 0
