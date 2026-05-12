"""Fetch NOAA ISD per-year files for EBBR and parse into hourly temperature.

Usage:
    uv run python scripts/fetch_isd.py [--refresh]

NOAA ISD per-station-per-year files live at:
    https://www.ncei.noaa.gov/data/global-hourly/access/<year>/<station>.csv

The `.csv` form has a header. This script downloads it, projects each row
into the space-delimited line shape that `bw_hackathon_data.isd.parse_isd_lines`
expects, and writes one combined hourly-temperature parquet.
"""

from __future__ import annotations

import argparse
import csv
import io
from pathlib import Path

import httpx

from bw_hackathon_data import config, isd

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE = REPO_ROOT / "cache" / "isd"
URL_TEMPLATE = "https://www.ncei.noaa.gov/data/global-hourly/access/{year}/{station}.csv"


def _csv_row_to_isd_line(row: dict[str, str]) -> str | None:
    """Project a NOAA ISD CSV row into the space-delimited shape parse_isd_lines reads.

    NOAA CSV columns we need:
      - DATE: '2025-01-01T00:00:00'
      - TMP : '+0023,1' (tenths °C followed by quality flag) or '+9999,?' for missing
    """
    date = row.get("DATE")
    tmp = row.get("TMP", "")
    if not date or not tmp:
        return None
    try:
        dt_part, time_part = date.split("T")
        y, mo, da = dt_part.split("-")
        hr, mi, _ = time_part.split(":")
    except ValueError:
        return None
    temp_tenths = tmp.split(",")[0]
    if not (temp_tenths.startswith("+") or temp_tenths.startswith("-")):
        return None
    # Synthesize a space-delimited line the parser understands.
    return f"{y} {mo} {da} {hr} {mi}  064510 99999  {temp_tenths}"


def fetch_year(year: int) -> list[str]:
    url = URL_TEMPLATE.format(year=year, station=config.ISD_STATION_EBBR)
    print(f"[isd] fetching {url}")
    resp = httpx.get(url, timeout=60.0, follow_redirects=True)
    resp.raise_for_status()
    reader = csv.DictReader(io.StringIO(resp.text))
    return [line for line in (_csv_row_to_isd_line(r) for r in reader) if line is not None]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    out = CACHE / "temp.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and not args.refresh:
        print(f"[isd] cached at {out}, skipping")
        return

    # TEST_END is exclusive (e.g. 2026-01-01 means "up through end of 2025").
    # Only include a year that contains at least one hour in the window.
    last_year = (
        config.TEST_END.year
        if (config.TEST_END.month, config.TEST_END.day) > (1, 1)
        else config.TEST_END.year - 1
    )
    years = list(range(config.TRAIN_START.year, last_year + 1))
    print(f"[isd] fetching years {years[0]}..{years[-1]}")
    all_lines: list[str] = []
    for y in years:
        all_lines.extend(fetch_year(y))

    df = isd.parse_isd_lines(all_lines)
    tmp = out.with_suffix(".parquet.tmp")
    df.write_parquet(tmp)
    tmp.replace(out)
    print(f"[isd] wrote {df.height} hourly rows → {out}")


if __name__ == "__main__":
    main()
