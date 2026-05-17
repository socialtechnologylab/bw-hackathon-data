"""Tests for the ENTSO-E client wrapper.

Real network calls are not exercised here; we mock the entsoe-py
client's `query_generation` and `query_load` and verify our wrapper
converts pandas → polars correctly, sums B19+B18 for wind (B19=Wind
Onshore, B18=Wind Offshore — B17 is *Waste*, not wind), and yields
the expected (timestamp, value) shape.

The mocks return the column shapes entsoe-py actually emits:
  - Wind Onshore / Solar: flat DataFrame, column named after the
    human-readable PSR label ('Wind Onshore', 'Solar', ...).
  - Wind Offshore: MultiIndex columns, e.g.
    ('Wind Offshore', 'Actual Aggregated') + ('Wind Offshore',
    'Actual Consumption'). We extract 'Actual Aggregated'.
"""

from datetime import UTC, datetime
from unittest.mock import MagicMock

import pandas as pd
import polars as pl
import pytest

from bw_hackathon_data.entsoe import (
    fetch_demand,
    fetch_solar,
    fetch_wind,
)


def _hourly_pandas_series(start: datetime, values: list[float]) -> pd.Series:
    idx = pd.date_range(start=start, periods=len(values), freq="h", tz="UTC")
    return pd.Series(values, index=idx)


def _flat_df(label: str, series: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({label: series})


def _offshore_df(agg: pd.Series, consumption: pd.Series | None = None) -> pd.DataFrame:
    """Mimic entsoe-py's offshore response: MultiIndex with 'Actual Aggregated'
    and 'Actual Consumption' second-level columns."""
    if consumption is None:
        consumption = pd.Series([float("nan")] * len(agg), index=agg.index)
    cols = pd.MultiIndex.from_tuples([
        ("Wind Offshore", "Actual Aggregated"),
        ("Wind Offshore", "Actual Consumption"),
    ])
    df = pd.DataFrame(index=agg.index, columns=cols, dtype="float64")
    df[("Wind Offshore", "Actual Aggregated")] = agg.values
    df[("Wind Offshore", "Actual Consumption")] = consumption.values
    return df


def test_fetch_solar_returns_timestamp_value_parquet_shape():
    client = MagicMock()
    series = _hourly_pandas_series(datetime(2025, 1, 1, tzinfo=UTC), [0.0, 5.0, 50.0, 80.0])
    client.query_generation.return_value = _flat_df("Solar", series)

    df = fetch_solar(client, datetime(2025, 1, 1, tzinfo=UTC), datetime(2025, 1, 1, 4, tzinfo=UTC))

    assert df.columns == ["timestamp", "value"]
    assert df["timestamp"].dtype == pl.Utf8
    assert df["value"].dtype == pl.Float64
    assert df["timestamp"][0] == "2025-01-01T00:00:00+00:00"
    assert df["value"].to_list() == [0.0, 5.0, 50.0, 80.0]


def test_fetch_solar_converts_brussels_local_time_to_utc():
    """Regression for the pilot: ENTSO-E timestamps come back as Brussels-local."""
    client = MagicMock()
    idx = pd.date_range(start="2025-01-01 00:00", periods=3, freq="h", tz="Europe/Brussels")
    series = pd.Series([0.0, 10.0, 50.0], index=idx)
    client.query_generation.return_value = _flat_df("Solar", series)

    df = fetch_solar(client, datetime(2025, 1, 1, tzinfo=UTC), datetime(2025, 1, 1, 4, tzinfo=UTC))

    assert df["timestamp"][0] == "2024-12-31T23:00:00+00:00"
    assert df["timestamp"][1] == "2025-01-01T00:00:00+00:00"
    assert df["timestamp"][2] == "2025-01-01T01:00:00+00:00"


def test_fetch_solar_raises_when_label_missing():
    """Silent fallback to iloc[:, 0] was how 'B17 returns Waste' went unnoticed."""
    client = MagicMock()
    series = _hourly_pandas_series(datetime(2025, 1, 1, tzinfo=UTC), [1.0, 2.0])
    client.query_generation.return_value = _flat_df("Waste", series)

    with pytest.raises(ValueError, match="expected 'Solar'"):
        fetch_solar(client, datetime(2025, 1, 1, tzinfo=UTC), datetime(2025, 1, 1, 2, tzinfo=UTC))


def test_fetch_wind_sums_onshore_and_offshore():
    client = MagicMock()
    on = _hourly_pandas_series(datetime(2025, 1, 1, tzinfo=UTC), [100.0, 200.0, 150.0])
    off_agg = _hourly_pandas_series(datetime(2025, 1, 1, tzinfo=UTC), [50.0, 60.0, 70.0])
    client.query_generation.side_effect = [
        _flat_df("Wind Onshore", on),
        _offshore_df(off_agg),
    ]

    df = fetch_wind(client, datetime(2025, 1, 1, tzinfo=UTC), datetime(2025, 1, 1, 3, tzinfo=UTC))

    assert df["value"].to_list() == [150.0, 260.0, 220.0]


def test_fetch_wind_raises_when_onshore_label_missing():
    """Regression for the B17/Waste bug: silent fallback must not return."""
    client = MagicMock()
    waste = _hourly_pandas_series(datetime(2025, 1, 1, tzinfo=UTC), [230.0, 240.0])
    off_agg = _hourly_pandas_series(datetime(2025, 1, 1, tzinfo=UTC), [1000.0, 1100.0])
    client.query_generation.side_effect = [
        _flat_df("Waste", waste),
        _offshore_df(off_agg),
    ]

    with pytest.raises(ValueError, match="expected 'Wind Onshore'"):
        fetch_wind(client, datetime(2025, 1, 1, tzinfo=UTC), datetime(2025, 1, 1, 2, tzinfo=UTC))


def test_fetch_wind_raises_when_offshore_label_missing():
    client = MagicMock()
    on = _hourly_pandas_series(datetime(2025, 1, 1, tzinfo=UTC), [100.0, 200.0])
    client.query_generation.side_effect = [
        _flat_df("Wind Onshore", on),
        pd.DataFrame(),
    ]

    with pytest.raises(ValueError, match="expected 'Wind Offshore'"):
        fetch_wind(client, datetime(2025, 1, 1, tzinfo=UTC), datetime(2025, 1, 1, 2, tzinfo=UTC))


def test_fetch_demand_uses_query_load():
    client = MagicMock()
    series = _hourly_pandas_series(datetime(2025, 1, 1, tzinfo=UTC), [9000.0, 9500.0])
    client.query_load.return_value = pd.DataFrame({"Actual Load": series})

    df = fetch_demand(client, datetime(2025, 1, 1, tzinfo=UTC), datetime(2025, 1, 1, 2, tzinfo=UTC))

    assert df["value"].to_list() == [9000.0, 9500.0]
    client.query_load.assert_called_once()


def test_fetch_demand_falls_back_to_first_column_if_named_differently():
    """Demand only has one PSR — no peer column to mix it up with — so we
    keep the lenient fallback there. Solar/wind no longer tolerate."""
    client = MagicMock()
    series = _hourly_pandas_series(datetime(2025, 1, 1, tzinfo=UTC), [9000.0, 9500.0])
    client.query_load.return_value = pd.DataFrame({"0": series})

    df = fetch_demand(client, datetime(2025, 1, 1, tzinfo=UTC), datetime(2025, 1, 1, 2, tzinfo=UTC))

    assert df["value"].to_list() == [9000.0, 9500.0]
