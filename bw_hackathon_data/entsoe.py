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
    return _series_to_parquet_df(_extract_aggregated(df, "Solar"))


def _extract_aggregated(df: pd.DataFrame, expected_label: str) -> pd.Series:
    """Pull the 'Actual Aggregated' series from a query_generation response.

    entsoe-py returns two shapes depending on whether the PSR has a
    consumption component:
      - Flat: one column named after the human label (e.g. 'Wind Onshore').
      - MultiIndex: ('<label>', 'Actual Aggregated') + ('<label>', 'Actual
        Consumption') — Wind Offshore, Pumped Storage, etc.
    Fails loud if `expected_label` doesn't appear — silent column fallback
    is how 'B17 returns Waste' went unnoticed.
    """
    if isinstance(df.columns, pd.MultiIndex):
        labels = list(df.columns.get_level_values(0).unique())
        if expected_label not in labels:
            raise ValueError(
                f"expected '{expected_label}' in entsoe response, got {labels!r}"
            )
        return df[(expected_label, "Actual Aggregated")]
    if expected_label in df.columns:
        return df[expected_label]
    raise ValueError(
        f"expected '{expected_label}' column in entsoe response, got {list(df.columns)!r}"
    )


def fetch_wind(client: Any, start: datetime, end: datetime) -> pl.DataFrame:
    """B19 (Wind Onshore) + B18 (Wind Offshore) summed, Belgium, hourly MWh.

    ENTSO-E PSR codes per the EIC reference manual: B17=Waste, B18=Wind
    Offshore, B19=Wind Onshore. An earlier version of this module used B17
    for onshore (it's actually Waste); the bug went unnoticed because the
    fetch fell back to `iloc[:, 0]` when the label-keyed lookup missed,
    silently shipping Waste generation as 'wind'. Fixed: PSR code corrected
    and extraction now fails loud if the expected label isn't present.
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

    s_on = _extract_aggregated(onshore, "Wind Onshore")
    s_off = _extract_aggregated(offshore, "Wind Offshore")
    summed = s_on.add(s_off, fill_value=0.0)
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
