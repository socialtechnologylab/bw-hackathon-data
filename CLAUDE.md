# Conventions

- Python ≥ 3.11. Package management: `uv` only. No `pip`.
- Code style: `ruff format`, `ruff check`, `pyright` (basic mode).
- Tests: `pytest`. Network-hitting tests marked `@pytest.mark.integration` and gated by `BW_DATA_NETWORK=1`.
- Data: `polars` for parquet I/O. `pandas` only at the LightGBM interface in `calibrate.py`.
- Library code under `bw_hackathon_data/`; thin orchestrator scripts under `scripts/`.
- Constants live in `bw_hackathon_data/config.py`. Don't sprinkle magic numbers.
- All parquet writes are atomic (tmpfile-then-rename).
- All timestamps are timezone-aware UTC. Write to parquet as ISO-8601 strings with `+00:00`.

## The pipeline

Five stages, each idempotent:
1. `scripts/fetch_entsoe.py` — solar / wind / demand targets
2. `scripts/fetch_isd.py` — temperature target
3. `scripts/fetch_gfs.py` — GFS forecast features
4. `scripts/build_all.py` — per-task feature + target join
5. `scripts/calibrate.py` — LightGBM → MAE → baseline_score → tasks.json + READMEs

Plus `scripts/release.py` (tarball + GitHub release) and `scripts/install_endpoint_state.py` (trainer-side).

Spec: `bw-training/docs/superpowers/specs/2026-05-12-bw-hackathon-data-design.md`.
