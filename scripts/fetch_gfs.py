"""Fetch all GFS init cycles in the train + test window from dynamical.org.

Usage:
    uv run python scripts/fetch_gfs.py [--start YYYY-MM-DD] [--end YYYY-MM-DD]

Default range: config.TRAIN_START → config.TEST_END.

Reads from the public ARCO Zarr at config.GFS_ZARR_URL in a single windowed
slice (much faster than the per-cycle Herbie byte-range approach the pilot
used). Writes one parquet per init cycle into cache/gfs/, matching the
layout build_all.py expects.
"""

from __future__ import annotations

import argparse
import time
from datetime import UTC, datetime
from pathlib import Path

from bw_hackathon_data import config, gfs

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE = REPO_ROOT / "cache" / "gfs"


def _parse_iso_date(s: str) -> datetime:
    return datetime.fromisoformat(s).replace(tzinfo=UTC)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default=config.TRAIN_START.date().isoformat())
    parser.add_argument("--end", default=config.TEST_END.date().isoformat())
    args = parser.parse_args()

    start = _parse_iso_date(args.start)
    end = _parse_iso_date(args.end)

    print(f"[gfs] fetching dynamical zarr {start.date()} → {end.date()}")
    print(f"[gfs]   variables: {sorted(config.GFS_VAR_RENAME) + ['wind10m_fcst (from u/v)']}")
    print(f"[gfs]   fxx range: {config.GFS_FXX_RANGE.start}..{config.GFS_FXX_RANGE.stop - 1}")
    print(f"[gfs]   bbox lat: {config.BBOX_LAT}, lon: {config.BBOX_LON}")
    print(f"[gfs]   source:  {config.GFS_ZARR_URL}")

    t0 = time.time()
    n_written = gfs.fetch_window(start, end, CACHE)
    elapsed = time.time() - t0
    total_in_cache = len(list(CACHE.glob("*.parquet")))
    print(
        f"[gfs] done. wrote={n_written} total_cached={total_in_cache} "
        f"elapsed={elapsed:.1f}s"
    )


if __name__ == "__main__":
    main()
