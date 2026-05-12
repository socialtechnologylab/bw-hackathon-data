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
    """Convert a tz-aware hourly pandas Series → (timestamp, value) polars DF."""
    idx: pd.DatetimeIndex = series.index  # type: ignore[assignment]
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    timestamps = idx.strftime("%Y-%m-%dT%H:%M:%S%z").map(
        lambda s: s[:-2] + ":" + s[-2:] if len(s) >= 5 else s
    )
    return pl.DataFrame(
        {
            "timestamp": timestamps.tolist(),
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
    """B17 (Wind Onshore) + B18 (Wind Offshore) summed, Belgium, hourly MWh."""
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

    def _col(df: pd.DataFrame, code: str) -> pd.Series:
        col: pd.Series = df[code] if code in df.columns else df.iloc[:, 0]  # type: ignore[assignment]
        return col

    s_on = _col(onshore, config.ENTSOE_PSR_WIND_ONSHORE)
    s_off = _col(offshore, config.ENTSOE_PSR_WIND_OFFSHORE) if not offshore.empty else None

    summed = s_on.copy()
    if s_off is not None:
        summed = summed.add(s_off, fill_value=0.0)

    return _series_to_parquet_df(summed)


def fetch_demand(client: Any, start: datetime, end: datetime) -> pl.DataFrame:
    """Actual Total Load, Belgium, hourly MWh."""
    df = client.query_load(
        country_code=config.ENTSOE_AREA_BE,
        start=pd.Timestamp(start),
        end=pd.Timestamp(end),
    )
    series = df["Actual Load"] if "Actual Load" in df.columns else df.iloc[:, 0]
    return _series_to_parquet_df(series)
