"""Build per-task parquets + ground_truth from the populated cache.

Usage:
    uv run python scripts/build_all.py [--tasks solar-1d-ahead,wind-2h-ahead,...]

Reads cache/entsoe/<series>.parquet, cache/isd/temp.parquet, and
cache/gfs/<cycle>.parquet. Writes release/participant/<task>/* and
release/endpoint/ground_truth.json.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import polars as pl

from bw_hackathon_data import build, config

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE = REPO_ROOT / "cache"
RELEASE = REPO_ROOT / "release"


def _load_gfs_cache() -> dict[datetime, pl.DataFrame]:
    cache: dict[datetime, pl.DataFrame] = {}
    for path in sorted((CACHE / "gfs").glob("*.parquet")):
        stem = path.stem  # e.g. 20250101T0000Z
        cycle = datetime.strptime(stem, "%Y%m%dT%H%MZ").replace(tzinfo=config.UTC)
        cache[cycle] = pl.read_parquet(path)
    return cache


def _load_target(task_id: str) -> pl.DataFrame:
    if task_id == "solar-1d-ahead":
        return pl.read_parquet(CACHE / "entsoe" / "solar.parquet")
    if task_id == "wind-2h-ahead":
        return pl.read_parquet(CACHE / "entsoe" / "wind.parquet")
    if task_id == "demand-1d-ahead-test":
        return pl.read_parquet(CACHE / "entsoe" / "demand.parquet")
    if task_id == "temp-1d-ahead":
        return pl.read_parquet(CACHE / "isd" / "temp.parquet")
    raise ValueError(f"unknown task_id: {task_id}")


def _atomic_write_parquet(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    df.write_parquet(tmp)
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default=",".join(config.TASK_IDS))
    args = parser.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    print(f"[build] loading GFS cache from {CACHE / 'gfs'}")
    gfs_cache = _load_gfs_cache()
    print(f"[build] loaded {len(gfs_cache)} GFS cycles")

    ground_truth_all: dict[str, dict[str, float]] = {}
    drop_report: dict[str, dict[str, int | float]] = {}

    for task_id in tasks:
        target_col = config.TASK_TARGET_COLUMN[task_id]
        lead = config.TASK_LEAD_HOURS[task_id]
        target_df = _load_target(task_id)

        result = build.build_task(
            target_df=target_df,
            gfs_cache=gfs_cache,
            lead_hours=lead,
            task_id=task_id,
            target_column=target_col,
            train_window=(config.TRAIN_START, config.TRAIN_END),
            test_window=(config.TEST_START, config.TEST_END),
        )

        out_dir = RELEASE / "participant" / task_id
        _atomic_write_parquet(result.x_train, out_dir / "X_train.parquet")
        _atomic_write_parquet(result.y_train, out_dir / "y_train.parquet")
        _atomic_write_parquet(result.x_test, out_dir / "X_test.parquet")
        print(
            f"[build] {task_id}: train={result.x_train.height} "
            f"test={result.x_test.height} dropped={result.drop_count}/{result.total_count}"
        )

        drop_rate = result.drop_count / max(result.total_count, 1)
        threshold = config.DROP_THRESHOLD[task_id]
        drop_report[task_id] = {
            "drop_count": result.drop_count,
            "total_count": result.total_count,
            "drop_rate": round(drop_rate, 4),
            "threshold": threshold,
        }
        if drop_rate > threshold:
            print(f"[build] WARNING {task_id}: drop_rate {drop_rate:.4f} > threshold {threshold}")

        ground_truth_all[task_id] = result.ground_truth

    gt_path = RELEASE / "endpoint" / "ground_truth.json"
    gt_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = gt_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(ground_truth_all, indent=2) + "\n")
    tmp.replace(gt_path)
    print(f"[build] wrote ground_truth.json → {gt_path}")

    report_path = RELEASE / "endpoint" / "build_report.json"
    report_path.write_text(json.dumps(drop_report, indent=2) + "\n")
    print(f"[build] drop report → {report_path}")


if __name__ == "__main__":
    main()
