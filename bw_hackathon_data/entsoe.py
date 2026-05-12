"""ENTSO-E client wrapper.

Thin layer over `entsoe-py`'s EntsoePandasClient. Each top-level function
returns a polars DataFrame with columns (timestamp: str, value: float)
where timestamp is ISO-8601 UTC with a +00:00 offset.

Caller is responsible for supplying a constructed EntsoePandasClient
with an `api_key` and for retry/backoff if needed.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd
import polars as pl

from bw_hackathon_data import config


def _series_to_parquet_df(series: pd.Series) -> pl.DataFrame:
    """Convert an hourly pandas Series → (timestamp, value) polars DF.

    Timestamps are emitted as ISO-8601 strings with an explicit `+00:00` UTC
    offset. ENTSO-E returns series indexed by the requested country's local
    time (Brussels is CET/CEST, not UTC), so we convert here rather than rely
    on the caller's tz assumption.
    """
    idx: pd.DatetimeIndex = series.index  # type: ignore[assignment]
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    return pl.DataFrame(
        {
            "timestamp": [ts.isoformat() for ts in idx],
            "value": series.astype(float).tolist(),
        }
    )


def fetch_solar(client: Any, start: datetime, end: datetime) -> pl.DataFrame:
    """B16 (Solar) actual generation, Belgium control area, hourly MWh."""
    df = client.query_generation(
        country_code=config.ENTSOE_AREA_BE,
        start=pd.Timestamp(start),
        end=pd.Timestamp(end),
        psr_type=config.ENTSOE_PSR_SOLAR,
    )
    series = df[config.ENTSOE_PSR_SOLAR] if config.ENTSOE_PSR_SOLAR in df.columns else df.iloc[:, 0]
    return _series_to_parquet_df(series)


def fetch_wind(client: Any, start: datetime, end: datetime) -> pl.DataFrame:
    """B17 (Wind Onshore) + B18 (Wind Offshore) summed, Belgium, hourly MWh.

    Onshore tolerates a first-column fallback (entsoe-py occasionally returns
    the series under an unexpected name). Offshore requires the B18 column
    exactly — if it's missing or empty, treat as zero rather than risk
    silently picking up the wrong PSR.
    """
    onshore = client.query_generation(
        country_code=config.ENTSOE_AREA_BE,
        start=pd.Timestamp(start),
        end=pd.Timestamp(end),
        psr_type=config.ENTSOE_PSR_WIND_ONSHORE,
    )
    offshore = client.query_generation(
        country_code=config.ENTSOE_AREA_BE,
        start=pd.Timestamp(start),
        end=pd.Timestamp(end),
        psr_type=config.ENTSOE_PSR_WIND_OFFSHORE,
    )

    s_on: pd.Series = (
        onshore[config.ENTSOE_PSR_WIND_ONSHORE]
        if config.ENTSOE_PSR_WIND_ONSHORE in onshore.columns
        else onshore.iloc[:, 0]
    )
    s_off: pd.Series | None = (
        offshore[config.ENTSOE_PSR_WIND_OFFSHORE]
        if config.ENTSOE_PSR_WIND_OFFSHORE in offshore.columns
        else None
    )

    summed = s_on if s_off is None else s_on.add(s_off, fill_value=0.0)
    return _series_to_parquet_df(summed)


def fetch_demand(client: Any, start: datetime, end: datetime) -> pl.DataFrame:
    """Actual Total Load, Belgium, hourly MWh.

    ENTSO-E publishes Belgian ATL at 15-minute cadence (1 MTU = 15 min); the
    hackathon contract is hourly UTC. Resample to hourly mean — ATL is
    average power over the MTU, so the hourly mean of 4 quarter-hourly
    values is the correct hourly average power.
    """
    df = client.query_load(
        country_code=config.ENTSOE_AREA_BE,
        start=pd.Timestamp(start),
        end=pd.Timestamp(end),
    )
    series = df["Actual Load"] if "Actual Load" in df.columns else df.iloc[:, 0]
    # Resample to hourly mean. label='left' so the hour-mark stamps the START
    # of the hour (e.g. 00:00 = mean over 00:00–00:59), matching how solar /
    # wind are emitted by ENTSO-E.
    series = series.resample("1h", label="left", closed="left").mean()
    return _series_to_parquet_df(series)
