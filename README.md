# bw-hackathon-data

Real public-data pipeline for the BW hackathon. Produces the parquets +
endpoint state files that the `bw-training` scaffold expects.

Sibling of `bw-training`. The two repos communicate only through the
GitHub release tarball this repo publishes.

## Quickstart

```bash
uv sync
cp .env.example .env
# fill in ENTSOE_API_TOKEN

# fetch raw data (slow — hours for GFS)
uv run python scripts/fetch_entsoe.py
uv run python scripts/fetch_isd.py
uv run python scripts/fetch_gfs.py

# build, calibrate, release
uv run python scripts/build_all.py
uv run python scripts/calibrate.py
uv run python scripts/release.py
```

See `bw-training/docs/superpowers/specs/2026-05-12-bw-hackathon-data-design.md`
for the design.

## Tests

```bash
uv run pytest                            # unit tests, no network
BW_DATA_NETWORK=1 uv run pytest          # also runs integration smoke
```
