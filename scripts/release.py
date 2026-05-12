"""Tar `release/` and create a GitHub release.

Usage:
    uv run python scripts/release.py --tag v1.0.0 [--repo <owner>/<name>]
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from bw_hackathon_data import release

REPO_ROOT = Path(__file__).resolve().parent.parent
RELEASE_DIR = REPO_ROOT / "release"
BUILD_DIR = REPO_ROOT / "build"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True, help="git tag for the release (e.g. v1.0.0)")
    parser.add_argument("--repo", default=None, help="optional <owner>/<name>")
    args = parser.parse_args()

    build_date = date.today().isoformat()
    top_level = f"bw-hackathon-data-{build_date}"
    tarball = BUILD_DIR / f"{top_level}.tar.gz"
    BUILD_DIR.mkdir(parents=True, exist_ok=True)

    calibration = json.loads((RELEASE_DIR / "endpoint" / "calibration_summary.json").read_text())
    manifest_path = release.write_manifest(
        RELEASE_DIR,
        build_date=build_date,
        release_tag=args.tag,
        train_window=("2023-01-01T00:00:00+00:00", "2025-01-01T00:00:00+00:00"),
        test_window=("2025-01-01T00:00:00+00:00", "2026-01-01T00:00:00+00:00"),
        entsoe_pull_date=build_date,
        isd_pull_date=build_date,
        gfs_cycles_description="00/06/12/18 UTC, 2023-01-01 → 2026-01-01",
        calibration=calibration,
    )
    print(f"[release] manifest → {manifest_path}")

    sha = release.tar_release(RELEASE_DIR, tarball, top_level)
    print(f"[release] tarball → {tarball}  (SHA256: {sha})")

    notes_path = BUILD_DIR / "release_notes.md"
    notes_path.write_text(
        f"# bw-hackathon-data {args.tag}\n\n"
        f"Built {build_date}. Tarball SHA256: `{sha}`.\n\n"
        f"## Calibration\n\n"
        + "\n".join(
            f"- **{task}**: observed MAE = {info['observed_mae']:.4f}, "
            f"baseline_score = {info['baseline_score']}"
            for task, info in calibration.items()
        )
        + "\n"
    )

    url = release.gh_release_create(args.tag, tarball, notes_path, repo=args.repo)
    print(f"[release] {url}")
    print()
    print(f"  RELEASE_URL = '{url}'")
    print(f"  EXPECTED_SHA = '{sha}'")
    print()
    print("Paste these into bw-training/participant_template/scripts/download_data.py (Task 16).")


if __name__ == "__main__":
    main()
