"""Fetch all GFS init cycles in the train + test window.

Usage:
    uv run python scripts/fetch_gfs.py [--start YYYY-MM-DD] [--end YYYY-MM-DD]

Default range: config.TRAIN_START → config.TEST_END (~4400 cycles, slow).
For Task 12 (solar pilot), pass --start 2025-01-01 --end 2025-02-01.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime, timedelta
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

    CACHE.mkdir(parents=True, exist_ok=True)
    current = start
    n_done = 0
    n_skipped = 0
    while current < end:
        for hour in config.GFS_CYCLE_HOURS:
            cycle = current.replace(hour=hour, minute=0, second=0, microsecond=0)
            if cycle < start or cycle >= end:
                continue
            out = CACHE / f"{cycle.strftime('%Y%m%dT%H%MZ')}.parquet"
            if out.exists():
                n_skipped += 1
                continue
            try:
                gfs.fetch_cycle(cycle, CACHE)
                n_done += 1
            except Exception as exc:  # noqa: BLE001
                print(f"[gfs] {cycle.isoformat()}: FAILED ({exc}); continuing")
        current += timedelta(days=1)

    print(f"[gfs] done. fetched={n_done} skipped(cached)={n_skipped}")


if __name__ == "__main__":
    main()
