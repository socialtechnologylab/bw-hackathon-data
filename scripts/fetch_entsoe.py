"""Fetch ENTSO-E target series for solar / wind / demand.

Usage:
    uv run python scripts/fetch_entsoe.py [--refresh]
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from entsoe.entsoe import EntsoePandasClient

from bw_hackathon_data import config
from bw_hackathon_data import entsoe as entsoe_wrap

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE = REPO_ROOT / "cache" / "entsoe"


def _atomic_write_parquet(df, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    df.write_parquet(tmp)
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="ignore cache")
    args = parser.parse_args()

    load_dotenv()
    token = os.environ.get("ENTSOE_API_KEY")
    if not token:
        raise SystemExit("ENTSOE_API_KEY not set; copy .env.example → .env and fill it in.")

    client = EntsoePandasClient(api_key=token)
    start = config.TRAIN_START
    end = config.TEST_END

    targets = {
        "solar": (entsoe_wrap.fetch_solar, CACHE / "solar.parquet"),
        "wind": (entsoe_wrap.fetch_wind, CACHE / "wind.parquet"),
        "demand": (entsoe_wrap.fetch_demand, CACHE / "demand.parquet"),
    }

    for name, (fn, out_path) in targets.items():
        if out_path.exists() and not args.refresh:
            print(f"[entsoe] {name}: cached at {out_path}, skipping")
            continue
        print(f"[entsoe] {name}: fetching {start.date()} → {end.date()}")
        for attempt in range(3):
            try:
                df = fn(client, start, end)
                _atomic_write_parquet(df, out_path)
                print(f"[entsoe] {name}: wrote {df.height} rows → {out_path}")
                break
            except Exception as exc:  # noqa: BLE001
                wait = 2**attempt
                print(f"[entsoe] {name}: attempt {attempt + 1} failed ({exc}); retry in {wait}s")
                time.sleep(wait)
        else:
            raise SystemExit(f"[entsoe] {name}: 3 attempts failed — aborting")


if __name__ == "__main__":
    main()
