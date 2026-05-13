"""Tarball + manifest + GitHub release upload."""

from __future__ import annotations

import hashlib
import json
import subprocess
import tarfile
from pathlib import Path


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(
    release_dir: Path,
    *,
    build_date: str,
    release_tag: str,
    train_window: tuple[str, str],
    test_window: tuple[str, str],
    entsoe_pull_date: str,
    isd_pull_date: str,
    gfs_cycles_description: str,
    calibration: dict,
) -> Path:
    """Walk `release_dir` and write manifest.json with per-file SHAs + metadata."""
    files = []
    for path in sorted(release_dir.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            rel = path.relative_to(release_dir).as_posix()
            files.append(
                {
                    "path": rel,
                    "sha256": sha256_of(path),
                    "size_bytes": path.stat().st_size,
                }
            )

    manifest = {
        "build_date": build_date,
        "release_tag": release_tag,
        "train_window": list(train_window),
        "test_window": list(test_window),
        "entsoe_pull_date": entsoe_pull_date,
        "isd_pull_date": isd_pull_date,
        "gfs_cycles": gfs_cycles_description,
        "files": files,
        "calibration": calibration,
    }
    out = release_dir / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2) + "\n")
    return out


def tar_release(
    release_dir: Path,
    tarball_path: Path,
    top_level_name: str,
    *,
    include: tuple[str, ...] | None = None,
) -> str:
    """Tar `release_dir` into tarball_path with a single top-level directory.

    `include`: optional tuple of subdirectory prefixes (relative to release_dir)
    to limit the archive to. Each path is included iff its relative-to-release_dir
    starts with one of those prefixes OR equals one of them. If None, everything
    under release_dir is tarred.

    Use case: emitting a participant-only tarball (`include=('participant',)`)
    so the public release artefact does NOT carry the endpoint state — in
    particular `endpoint/ground_truth.json`, which is y_test for the workshop
    tasks and must not be shipped to participants.

    Returns the tarball's SHA256.
    """
    tarball_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tarball_path, "w:gz") as tar:
        for path in sorted(release_dir.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(release_dir).as_posix()
            if include is not None:
                top = rel.split("/", 1)[0]
                if top not in include:
                    continue
            arcname = Path(top_level_name) / path.relative_to(release_dir)
            tar.add(path, arcname=arcname.as_posix())
    return sha256_of(tarball_path)


def gh_release_create(
    tag: str,
    tarball_path: Path,
    notes_path: Path,
    repo: str | None = None,
) -> str:
    """Invoke `gh release create`. Returns the release URL printed by gh."""
    cmd = ["gh", "release", "create", tag, str(tarball_path), "--notes-file", str(notes_path)]
    if repo is not None:
        cmd.extend(["--repo", repo])
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"gh release create failed: {result.stderr.strip()}\n"
            f"Manual upload command: {' '.join(cmd)}"
        )
    # gh prints the release URL to stdout on success.
    return result.stdout.strip()
