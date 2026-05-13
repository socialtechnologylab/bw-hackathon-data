"""Tar `release/` and create a GitHub release.

Cuts two tarballs:
  1. <name>-participant.tar.gz  — parquets + per-task README only.
     This is the PUBLIC artefact attached to the GitHub release;
     participants download it via participant_template/scripts/download_data.py.
     Contains zero ground-truth labels.
  2. <name>-full.tar.gz         — everything (participant/ + endpoint/).
     Kept locally for the trainer to push to /var/lib/bw-endpoint/state/.
     NOT uploaded anywhere public.

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
    participant_tarball = BUILD_DIR / f"{top_level}-participant.tar.gz"
    full_tarball = BUILD_DIR / f"{top_level}-full.tar.gz"
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

    # Participant-only tarball: parquets + READMEs. NO ground_truth.
    participant_sha = release.tar_release(
        RELEASE_DIR, participant_tarball, top_level, include=("participant",)
    )
    print(f"[release] participant tarball → {participant_tarball}  (SHA256: {participant_sha})")

    # Full tarball: includes endpoint/. Kept locally only.
    full_sha = release.tar_release(RELEASE_DIR, full_tarball, top_level)
    print(f"[release] full tarball       → {full_tarball}  (SHA256: {full_sha})")
    print(f"[release]   (full tarball is internal — push only its endpoint/ to /var/lib/<app>)")

    notes_path = BUILD_DIR / "release_notes.md"
    notes_path.write_text(
        f"# bw-hackathon-data {args.tag}\n\n"
        f"Built {build_date}. Participant tarball SHA256: `{participant_sha}`.\n\n"
        f"This release ships **{participant_tarball.name}** — the per-task\n"
        f"parquets and READMEs only. `ground_truth.json` lives on the\n"
        f"scoring endpoint and is not in this archive.\n\n"
        f"## Baseline MAE per task\n\n"
        + "\n".join(
            f"- **{task}**: {info['baseline_mae']:.4f} {info['unit']}"
            for task, info in calibration.items()
        )
        + "\n"
    )

    url = release.gh_release_create(args.tag, participant_tarball, notes_path, repo=args.repo)
    print(f"[release] {url}")
    print()
    asset_url = url.replace(
        f"/releases/tag/{args.tag}",
        f"/releases/download/{args.tag}/{participant_tarball.name}",
    )
    print(f"  RELEASE_URL  = '{asset_url}'")
    print(f"  EXPECTED_SHA = '{participant_sha}'")
    print()
    print("Paste these into bw-training/participant_template/scripts/download_data.py.")


if __name__ == "__main__":
    main()
