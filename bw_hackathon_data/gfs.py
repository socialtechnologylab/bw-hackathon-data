"""GFS feature fetch + aggregation.

`fetch_cycle` pulls one GFS init cycle from the AWS public S3 mirror via
Herbie, byte-range-subsetting to the 4 variables we care about, then
calls `aggregate_to_belgium_means` to produce the cached parquet row.

The aggregation is split out as a pure function so we can unit-test it
without hitting the network. The network-hitting `fetch_cycle` is
exercised only by the gated integration smoke test.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import cast

import numpy as np
import polars as pl
import xarray as xr

from bw_hackathon_data import config


def combine_uv_to_speed(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """sqrt(u^2 + v^2). Works element-wise on numpy arrays."""
    return np.hypot(u, v)


def aggregate_to_belgium_means(
    ds: xr.Dataset,
    cycle: datetime,
    bbox_lat: tuple[float, float],
    bbox_lon: tuple[float, float],
    var_rename: dict[str, str],
) -> pl.DataFrame:
    """Crop to bbox, area-mean each variable, return (cycle_utc, valid_time, fxx, *vars).

    `ds` must have a `step` dimension (forecast lead time as timedelta64).
    `var_rename` maps source variable names (in `ds`) to output column names.
    """
    lat_lo, lat_hi = bbox_lat
    lon_lo, lon_hi = bbox_lon

    # Some GFS GRIBs index latitude from 90 → -90 (descending). Normalize to
    # ascending up-front so the subsequent slice works unconditionally, and a
    # zero-size result surfaces as a real bbox bug rather than a silent retry.
    if ds.latitude.values[0] > ds.latitude.values[-1]:
        ds = ds.isel(latitude=slice(None, None, -1))

    cropped = ds.sel(latitude=slice(lat_lo, lat_hi), longitude=slice(lon_lo, lon_hi))
    means = cropped.mean(dim=("latitude", "longitude"))

    # `step` is a timedelta; convert to integer forecast hours.
    fxx_hours = (means.step.values / np.timedelta64(1, "h")).astype(int)

    cycle_iso = cycle.isoformat()

    rows: dict[str, list] = {
        "cycle_utc": [cycle_iso] * len(fxx_hours),
        "valid_time": [(cycle + timedelta(hours=int(h))).isoformat() for h in fxx_hours],
        "fxx": fxx_hours.tolist(),
    }

    for src_name, out_name in var_rename.items():
        rows[out_name] = means[src_name].values.astype(float).tolist()

    return pl.DataFrame(rows)


def fetch_cycle(cycle: datetime, cache_dir: Path) -> Path:
    """Fetch one GFS init cycle and write the aggregated parquet to cache_dir.

    Skips network if the target parquet already exists. Returns the parquet path.
    """
    # Late import so unit tests don't need herbie installed in the import path.
    from herbie import Herbie  # type: ignore[import-untyped]

    out_path = cache_dir / f"{cycle.strftime('%Y%m%dT%H%MZ')}.parquet"
    if out_path.exists():
        return out_path

    cache_dir.mkdir(parents=True, exist_ok=True)
    fxx_list = list(config.GFS_FXX_RANGE)

    dsets: list[xr.Dataset] = []
    for fxx in fxx_list:
        h = Herbie(
            cycle.strftime("%Y-%m-%d %H:%M"),
            model="gfs",
            product="pgrb2.0p25",
            fxx=fxx,
            priority=["aws"],
            verbose=False,
        )
        raw = h.xarray(config.GFS_HERBIE_SEARCH, remove_grib=True)
        # If multiple xarray Datasets are returned (one per GRIB record group),
        # merge them on the shared coords. Cast needed because Herbie has no
        # type stubs and pyright infers an overly-broad return type.
        if isinstance(raw, list):
            ds: xr.Dataset = xr.merge(cast(list[xr.Dataset], raw))
        else:
            ds = cast(xr.Dataset, raw)
        ds = ds.expand_dims({"step": [np.timedelta64(fxx, "h")]})
        dsets.append(ds)

    merged = xr.concat(dsets, dim="step")

    # Combine U/V → wind10m_fcst then drop the originals.
    if "u10" in merged.data_vars and "v10" in merged.data_vars:
        merged["wind10m_fcst"] = (
            ("step", "latitude", "longitude"),
            combine_uv_to_speed(merged["u10"].values, merged["v10"].values),
        )
    if "tcc" in merged.data_vars:
        merged["tcc"] = merged["tcc"] / 100.0  # → fraction

    # Drop renames whose source variable wasn't returned for this cycle.
    var_rename = {k: v for k, v in config.GFS_VAR_RENAME.items() if k in merged.data_vars}

    df = aggregate_to_belgium_means(
        merged,
        cycle=cycle,
        bbox_lat=config.BBOX_LAT,
        bbox_lon=config.BBOX_LON,
        var_rename=var_rename,
    )

    tmp_path = out_path.with_suffix(".parquet.tmp")
    df.write_parquet(tmp_path)
    tmp_path.replace(out_path)
    return out_path
