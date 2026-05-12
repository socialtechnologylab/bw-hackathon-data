"""GFS feature fetch + aggregation.

`fetch_window` pulls a slice of GFS forecast cycles from dynamical.org's
public ARCO Zarr store (Apache 2.0), area-means each variable over the
Belgium bbox, and writes one parquet per init cycle to the cache dir.

The aggregation is split out as a pure function so we can unit-test it
without hitting the network. `fetch_window` is exercised only by the
gated integration smoke test and by the full-pipeline build (Task 13+).
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import xarray as xr

from bw_hackathon_data import config

# Chunk size for the windowed fetch. Each chunk = one calendar month
# (~120 cycles, ~120 s of wall-clock per dynamical.org benchmark). Failures
# only cost a chunk's worth of work to retry, and partial progress is durable.
CHUNK_MAX_RETRIES = 3
CHUNK_RETRY_BACKOFF_S = 10


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


def _open_zarr(url: str) -> xr.Dataset:
    """Open the dynamical GFS-forecast zarr. Wrapped so tests can monkeypatch."""
    return xr.open_zarr(url, consolidated=True)


def _month_starts(start: datetime, end: datetime) -> list[datetime]:
    """Yield the first-of-each-month between [start, end), in order."""
    cur = start.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    out: list[datetime] = []
    while cur < end:
        out.append(cur)
        if cur.month == 12:
            cur = cur.replace(year=cur.year + 1, month=1)
        else:
            cur = cur.replace(month=cur.month + 1)
    return out


def _fetch_chunk(
    ds: xr.Dataset,
    chunk_start: datetime,
    chunk_end: datetime,
    cache_dir: Path,
    fxx: range,
) -> int:
    """Pull one chunk's slice, area-mean, write per-cycle parquets. Returns count."""
    keep = list(config.GFS_VAR_RENAME) + [config.GFS_WIND_U_VAR, config.GFS_WIND_V_VAR]
    bbox_lat_hi, bbox_lat_lo = config.BBOX_LAT[1], config.BBOX_LAT[0]
    lead_lo = np.timedelta64(fxx.start, "h")
    lead_hi = np.timedelta64(fxx.stop - 1, "h")

    sliced = ds[keep].sel(
        init_time=slice(
            np.datetime64(chunk_start.replace(tzinfo=None)),
            np.datetime64(chunk_end.replace(tzinfo=None)),
        ),
        latitude=slice(bbox_lat_hi, bbox_lat_lo),
        longitude=slice(*config.BBOX_LON),
        lead_time=slice(lead_lo, lead_hi),
    )

    means = sliced.mean(dim=("latitude", "longitude")).load()

    # Derive wind10m magnitude and rescale cloud cover so the parquet matches
    # the contract (W/m², °C, m/s, 0–1).
    wind = combine_uv_to_speed(
        means[config.GFS_WIND_U_VAR].values,
        means[config.GFS_WIND_V_VAR].values,
    )
    means = means.drop_vars([config.GFS_WIND_U_VAR, config.GFS_WIND_V_VAR])
    means["wind10m_fcst"] = (means.dims, wind)
    if "total_cloud_cover_atmosphere" in means.data_vars:
        means["total_cloud_cover_atmosphere"] = means["total_cloud_cover_atmosphere"] / 100.0

    var_rename = {**config.GFS_VAR_RENAME, "wind10m_fcst": "wind10m_fcst"}
    fxx_hours = (means.lead_time.values / np.timedelta64(1, "h")).astype(int)

    n_written = 0
    for init_value in means.init_time.values:
        init_ns = np.datetime64(init_value, "ns").astype("int64")
        cycle = datetime.fromtimestamp(init_ns / 1e9, tz=config.UTC)
        out_path = cache_dir / f"{cycle.strftime('%Y%m%dT%H%MZ')}.parquet"
        if out_path.exists():
            continue

        per_cycle = means.sel(init_time=init_value)
        cycle_iso = cycle.isoformat()
        rows: dict[str, list] = {
            "cycle_utc": [cycle_iso] * len(fxx_hours),
            "valid_time": [(cycle + timedelta(hours=int(h))).isoformat() for h in fxx_hours],
            "fxx": fxx_hours.tolist(),
        }
        for src_name, out_name in var_rename.items():
            rows[out_name] = per_cycle[src_name].values.astype(float).tolist()

        df = pl.DataFrame(rows)
        tmp = out_path.with_suffix(".parquet.tmp")
        df.write_parquet(tmp)
        tmp.replace(out_path)
        n_written += 1

    return n_written


def _chunk_is_complete(
    chunk_start: datetime, chunk_end: datetime, cache_dir: Path
) -> bool:
    """Cheap idempotency check: does the cache already have all the parquets
    we'd expect for this month? Counts parquets whose filename-stamp falls
    in [chunk_start, chunk_end). The first run computes the expected count
    from cycles per day × days; a perfect match means we can skip the whole
    chunk's network fetch.
    """
    days = (chunk_end - chunk_start).days
    expected = days * len(config.GFS_CYCLE_HOURS)
    have = 0
    for path in cache_dir.glob("*.parquet"):
        try:
            stamp = datetime.strptime(path.stem, "%Y%m%dT%H%MZ").replace(tzinfo=config.UTC)
        except ValueError:
            continue
        if chunk_start <= stamp < chunk_end:
            have += 1
    return have >= expected


def fetch_window(
    init_start: datetime,
    init_end: datetime,
    cache_dir: Path,
    *,
    fxx_range: range | None = None,
    zarr_url: str | None = None,
) -> int:
    """Fetch all GFS init cycles in [init_start, init_end) from dynamical's zarr.

    Processes the window in monthly chunks so transient network failures
    only cost ~120 s of work, and the cache provides natural resume across
    runs (existing per-cycle parquets are skipped).

    Returns the total number of cycles newly written.
    """
    fxx = fxx_range or config.GFS_FXX_RANGE
    url = zarr_url or config.GFS_ZARR_URL

    cache_dir.mkdir(parents=True, exist_ok=True)

    # Open the zarr store once; reused across all chunks.
    ds = _open_zarr(url)

    months = _month_starts(init_start, init_end)
    total_written = 0
    overall_t0 = time.time()

    for i, month_start in enumerate(months):
        # Next-month boundary (or init_end on the final iteration).
        if month_start.month == 12:
            month_end = month_start.replace(year=month_start.year + 1, month=1)
        else:
            month_end = month_start.replace(month=month_start.month + 1)
        month_end = min(month_end, init_end)

        if _chunk_is_complete(month_start, month_end, cache_dir):
            print(
                f"[gfs] {month_start:%Y-%m} ({i + 1}/{len(months)}): already cached, skip"
            )
            continue

        for attempt in range(1, CHUNK_MAX_RETRIES + 1):
            try:
                t0 = time.time()
                n = _fetch_chunk(ds, month_start, month_end, cache_dir, fxx)
                elapsed = time.time() - t0
                total_written += n
                print(
                    f"[gfs] {month_start:%Y-%m} ({i + 1}/{len(months)}): "
                    f"wrote {n} cycles in {elapsed:.1f}s "
                    f"(total {total_written}, overall {time.time() - overall_t0:.0f}s)"
                )
                break
            except Exception as exc:  # noqa: BLE001
                wait = CHUNK_RETRY_BACKOFF_S * attempt
                print(
                    f"[gfs] {month_start:%Y-%m} attempt {attempt}/"
                    f"{CHUNK_MAX_RETRIES} failed: {exc!r}. Sleep {wait}s, then retry."
                )
                time.sleep(wait)
                # Reopen the zarr store in case the failure was a stale
                # connection. open_zarr is cheap (metadata fetch only).
                ds = _open_zarr(url)
        else:
            print(f"[gfs] {month_start:%Y-%m}: gave up after {CHUNK_MAX_RETRIES} attempts")

    return total_written
