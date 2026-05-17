"""Run LightGBM on built parquets, record baseline MAE, write tasks.json + per-task READMEs.

Usage:
    uv run python scripts/calibrate.py
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import polars as pl

from bw_hackathon_data import calibrate, config

REPO_ROOT = Path(__file__).resolve().parent.parent
RELEASE = REPO_ROOT / "release"

README_TEMPLATE = """# {display_name} — data

**Target column:** `{target_column}`

**Cadence:** hourly, UTC. Timestamps are ISO-8601 with explicit `+00:00`.

**Train range:** {train_start} → {train_end}
**Test range:**  {test_start} → {test_end} (labels live on the scoring endpoint)

## Files

- `X_train.parquet` — features for the training window.
- `y_train.parquet` — `(timestamp, {target_column})`.
- `X_test.parquet` — features for the test window. **No `y_test`.**

## Feature columns

| Column | Description | Units |
|---|---|---|
| `timestamp` | ISO-8601 hourly timestamp, UTC | — |
| `ghi_fcst` | Forecast global horizontal irradiance (GFS `sdswrf`) | W/m² |
| `t2m_fcst` | Forecast 2-metre air temperature (GFS `tmp`) | °C |
| `wind10m_fcst` | Forecast 10-metre wind speed (GFS magnitude of u,v) | m/s |
| `cloud_cover_fcst` | Forecast total cloud cover (GFS `tcdc` / 100) | 0–1 |
| `hour` | Hour of day | 0–23 |
| `dow` | Day of week | 0–6 (Mon=0) |
| `month` | Month | 1–12 |

## Provenance

- Targets: {target_provenance}
- Features: GFS 0.25° from the AWS public S3 mirror (`s3://noaa-gfs-bdp-pds/`),
  init cycles 00 / 06 / 12 / 18 UTC, area-mean over Belgium bbox
  (lat {bbox_lat_lo}–{bbox_lat_hi}, lon {bbox_lon_lo}–{bbox_lon_hi}).
- Forecast alignment: latest cycle initialised ≥ {lead_hours} hours before t.

## Quirks

- {drop_count} of {total_count} hours dropped due to missing data ({drop_rate:.2%}).
- Timestamps with no aligned GFS cycle are dropped silently.

## Baseline

Observed baseline MAE: **{observed_mae:.3f} {metric_unit}** from
`LGBMRegressor(n_estimators=200, verbosity=-1)` on the listed feature
columns. That's the number to beat. Measured {calibration_date}.

Source: `bw-hackathon-data/scripts/calibrate.py`.
"""


_TARGET_PROVENANCE = {
    "solar-1d-ahead": (
        "ENTSO-E B16 (Solar) actual generation, Belgium control area (10YBE----------2)"
    ),
    "wind-1d-ahead": ("ENTSO-E B19 (Wind Onshore) + B18 (Wind Offshore) summed, Belgium"),
    "temp-1d-ahead": (
        "NOAA ISD station 064510 (EBBR / Brussels Airport), hourly METAR-derived t2m"
    ),
    "demand-1d-ahead-test": "ENTSO-E Actual Total Load, Belgium",
}

_METRIC_UNIT = {
    "solar-1d-ahead": "MWh",
    "wind-1d-ahead": "MWh",
    "temp-1d-ahead": "°C",
    "demand-1d-ahead-test": "MWh",
}


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    calibration_date = date.today().isoformat()
    tasks_json: dict = {}
    calibration_summary: dict = {}

    report = json.loads((RELEASE / "endpoint" / "build_report.json").read_text())

    for task_id in config.TASK_IDS:
        target_col = config.TASK_TARGET_COLUMN[task_id]
        task_dir = RELEASE / "participant" / task_id

        x_train = pl.read_parquet(task_dir / "X_train.parquet")
        y_train = pl.read_parquet(task_dir / "y_train.parquet")
        x_test = pl.read_parquet(task_dir / "X_test.parquet")
        gt = json.loads((RELEASE / "endpoint" / "ground_truth.json").read_text())[task_id]
        y_test = [gt[ts] for ts in x_test["timestamp"].to_list()]

        mae = calibrate.train_and_score(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            target_column=target_col,
        )
        print(f"[baseline-mae] {task_id}: {mae:.4f} {_METRIC_UNIT[task_id]}")

        envelope = config.CALIBRATION_ENVELOPE.get(task_id)
        if envelope is not None and not (envelope[0] <= mae <= envelope[1]):
            print(f"[baseline-mae] WARNING {task_id}: MAE {mae:.4f} outside envelope {envelope}")

        # tasks.json is metadata only — no baseline_score. Lower MAE = better;
        # leaderboard sorts ascending on metric_value.
        tasks_json[task_id] = {
            "name": config.TASK_DISPLAY_NAME[task_id],
            "metric": config.TASK_METRIC_LABEL[task_id],
            "score_direction": "min",
        }
        calibration_summary[task_id] = {
            "baseline_mae": round(mae, 4),
            "unit": _METRIC_UNIT[task_id],
        }

        drop = report[task_id]
        readme = README_TEMPLATE.format(
            display_name=config.TASK_DISPLAY_NAME[task_id],
            target_column=target_col,
            train_start=config.TRAIN_START.date(),
            train_end=config.TRAIN_END.date(),
            test_start=config.TEST_START.date(),
            test_end=config.TEST_END.date(),
            target_provenance=_TARGET_PROVENANCE[task_id],
            bbox_lat_lo=config.BBOX_LAT[0],
            bbox_lat_hi=config.BBOX_LAT[1],
            bbox_lon_lo=config.BBOX_LON[0],
            bbox_lon_hi=config.BBOX_LON[1],
            lead_hours=config.TASK_LEAD_HOURS[task_id],
            drop_count=drop["drop_count"],
            total_count=drop["total_count"],
            drop_rate=drop["drop_rate"],
            observed_mae=mae,
            metric_unit=_METRIC_UNIT[task_id],
            calibration_date=calibration_date,
        )
        (task_dir / "README.md").write_text(readme)

    out = RELEASE / "endpoint" / "tasks.json"
    out.write_text(json.dumps(tasks_json, indent=2) + "\n")
    print(f"[calibrate] wrote {out}")

    summary_path = RELEASE / "endpoint" / "calibration_summary.json"
    summary_path.write_text(json.dumps(calibration_summary, indent=2) + "\n")
    print(f"[calibrate] summary → {summary_path}")


if __name__ == "__main__":
    main()
