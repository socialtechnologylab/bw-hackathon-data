"""NOAA ISD parser for hourly station temperature.

ISD per-station-per-year files use a fixed format. We extract:
- The report timestamp from columns 0–4 (year, month, day, hour, minute).
- The air temperature from a `+NNNN` or `-NNNN` token (tenths of °C),
  sentinel +9999 meaning missing.

After parsing each row, we keep the observation closest to the hour
boundary (within ±30 minutes) and bucket it to the round hour. If two
observations are equally close, the first wins.

The HTTPS endpoint for the per-station-per-year files lives at:
https://www.ncei.noaa.gov/data/global-hourly/access/<year>/<station>.csv

The `.csv` form is comma-separated with a header — this parser is for the
older space-delimited `.dat`-style line dumps which several Python
wrappers still emit. If you switch to the CSV access endpoint, write a
`parse_isd_csv` companion and route the script to it.
"""

from __future__ import annotations

import polars as pl

_MISSING = 9999  # tenths of °C sentinel


def _parse_one_line(line: str) -> tuple[str, int, float] | None:
    """Return (hour_iso, minute_offset, temp_celsius) or None if unparseable.

    Rows where the report minute is > 30 are dropped — they're closer to
    the next hour, and we'd rather pick the next hour's own (better)
    observation if one exists.
    """
    parts = line.split()
    if len(parts) < 8:
        return None
    try:
        year, month, day, hour, minute = (int(p) for p in parts[:5])
    except ValueError:
        return None
    if minute > 30:
        return None

    # The temperature token is the last 5-char `±NNNN` token (sign + 4 digits
    # of tenths of °C, including the +9999 sentinel). The `== 5` width avoids
    # picking up shorter ISD quality flags like `+1` or `+123`.
    temp_token = None
    for tok in reversed(parts):
        if len(tok) == 5 and (tok.startswith("+") or tok.startswith("-")):
            try:
                int(tok)
                temp_token = tok
                break
            except ValueError:
                continue
    if temp_token is None:
        return None

    temp_tenths = int(temp_token)
    if abs(temp_tenths) == _MISSING:
        return None

    iso = f"{year:04d}-{month:02d}-{day:02d}T{hour:02d}:00:00+00:00"
    return iso, minute, temp_tenths / 10.0


def parse_isd_lines(lines: list[str]) -> pl.DataFrame:
    """Parse a list of ISD raw lines into (timestamp, value) hourly DataFrame.

    Keeps the observation closest to each hour boundary (within ±30 min).
    Drops sentinel `+9999` (missing) values.
    """
    by_hour: dict[str, tuple[int, float]] = {}

    for line in lines:
        parsed = _parse_one_line(line)
        if parsed is None:
            continue
        iso, offset, temp_c = parsed
        existing = by_hour.get(iso)
        if existing is None or offset < existing[0]:
            by_hour[iso] = (offset, temp_c)

    rows = sorted((iso, val) for iso, (_, val) in by_hour.items())
    if not rows:
        return pl.DataFrame(schema={"timestamp": pl.Utf8, "value": pl.Float64})
    return pl.DataFrame(
        {
            "timestamp": [r[0] for r in rows],
            "value": [r[1] for r in rows],
        }
    )
