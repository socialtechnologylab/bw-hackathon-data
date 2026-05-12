"""Tests for the ENTSO-E client wrapper.

Real network calls are not exercised here; we mock the entsoe-py
client's `query_generation` and `query_load` and verify our wrapper
converts pandas → polars correctly, sums B17+B18 for wind, and yields
the expected (timestamp, value) shape.
"""

from datetime import UTC, datetime
from unittest.mock import MagicMock

import pandas as pd
import polars as pl

from bw_hackathon_data.entsoe import (
    fetch_demand,
    fetch_solar,
    fetch_wind,
)


def _hourly_pandas_series(start: datetime, values: list[float]) -> pd.Series:
    idx = pd.date_range(start=start, periods=len(values), freq="h", tz="UTC")
    return pd.Series(values, index=idx)


def test_fetch_solar_returns_timestamp_value_parquet_shape():
    """B16 series → (timestamp str ISO-8601 UTC, solar_mwh float) DataFrame."""
    client = MagicMock()
    series = _hourly_pandas_series(datetime(2025, 1, 1, tzinfo=UTC), [0.0, 5.0, 50.0, 80.0])
    # entsoe-py returns a DataFrame keyed by PSR code → series.
    client.query_generation.return_value = pd.DataFrame({"B16": series})

    df = fetch_solar(client, datetime(2025, 1, 1, tzinfo=UTC), datetime(2025, 1, 1, 4, tzinfo=UTC))

    assert df.columns == ["timestamp", "value"]
    assert df["timestamp"].dtype == pl.Utf8
    assert df["value"].dtype == pl.Float64
    assert df["timestamp"][0] == "2025-01-01T00:00:00+00:00"
    assert df["value"].to_list() == [0.0, 5.0, 50.0, 80.0]


def test_fetch_solar_converts_brussels_local_time_to_utc():
    """ENTSO-E returns Brussels-local timestamps (+01:00 in winter); we must
    convert to UTC before emitting. Regression test for the pilot's Task 12
    discovery that ENTSO-E timestamps came out as `+01:00` not `+00:00`."""
    client = MagicMock()
    # Build a series with CET (winter) tz: 00:00 Brussels = 23:00 UTC previous day.
    idx = pd.date_range(start="2025-01-01 00:00", periods=3, freq="h", tz="Europe/Brussels")
    series = pd.Series([0.0, 10.0, 50.0], index=idx)
    client.query_generation.return_value = pd.DataFrame({"B16": series})

    df = fetch_solar(client, datetime(2025, 1, 1, tzinfo=UTC), datetime(2025, 1, 1, 4, tzinfo=UTC))

    # First Brussels-CET timestamp (00:00 Brussels = 23:00 UTC the previous day).
    assert df["timestamp"][0] == "2024-12-31T23:00:00+00:00"
    assert df["timestamp"][1] == "2025-01-01T00:00:00+00:00"
    assert df["timestamp"][2] == "2025-01-01T01:00:00+00:00"


def test_fetch_wind_sums_b17_and_b18():
    """B17 + B18 → single summed `value` column.

    fetch_wind calls query_generation twice (once per PSR), so the mock
    uses side_effect to return onshore on the first call, offshore on the
    second.
    """
    client = MagicMock()
    onshore = _hourly_pandas_series(datetime(2025, 1, 1, tzinfo=UTC), [100.0, 200.0, 150.0])
    offshore = _hourly_pandas_series(datetime(2025, 1, 1, tzinfo=UTC), [50.0, 60.0, 70.0])
    client.query_generation.side_effect = [
        pd.DataFrame({"B17": onshore}),
        pd.DataFrame({"B18": offshore}),
    ]

    df = fetch_wind(client, datetime(2025, 1, 1, tzinfo=UTC), datetime(2025, 1, 1, 3, tzinfo=UTC))

    assert df["value"].to_list() == [150.0, 260.0, 220.0]


def test_fetch_wind_handles_missing_offshore_column():
    """If B18 isn't in the response (e.g. unavailable), treat as zero."""
    client = MagicMock()
    onshore = _hourly_pandas_series(datetime(2025, 1, 1, tzinfo=UTC), [100.0, 200.0])
    client.query_generation.side_effect = [
        pd.DataFrame({"B17": onshore}),
        pd.DataFrame(),  # empty offshore response
    ]

    df = fetch_wind(client, datetime(2025, 1, 1, tzinfo=UTC), datetime(2025, 1, 1, 2, tzinfo=UTC))

    assert df["value"].to_list() == [100.0, 200.0]


def test_fetch_demand_uses_query_load():
    """Demand goes through query_load, returns single-column shape."""
    client = MagicMock()
    series = _hourly_pandas_series(datetime(2025, 1, 1, tzinfo=UTC), [9000.0, 9500.0])
    client.query_load.return_value = pd.DataFrame({"Actual Load": series})

    df = fetch_demand(client, datetime(2025, 1, 1, tzinfo=UTC), datetime(2025, 1, 1, 2, tzinfo=UTC))

    assert df["value"].to_list() == [9000.0, 9500.0]
    client.query_load.assert_called_once()


def test_fetch_demand_falls_back_to_first_column_if_named_differently():
    """entsoe-py sometimes returns 'Actual Load' or just an unnamed column."""
    client = MagicMock()
    series = _hourly_pandas_series(datetime(2025, 1, 1, tzinfo=UTC), [9000.0, 9500.0])
    client.query_load.return_value = pd.DataFrame({"0": series})

    df = fetch_demand(client, datetime(2025, 1, 1, tzinfo=UTC), datetime(2025, 1, 1, 2, tzinfo=UTC))

    assert df["value"].to_list() == [9000.0, 9500.0]
