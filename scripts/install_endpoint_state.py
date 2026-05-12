"""Install endpoint state (tasks.json + ground_truth.json) into $BW_STATE_DIR.

Usage (on the trainer-side endpoint deploy box):
    BW_STATE_DIR=/var/lib/bw/endpoint_state \\
        uv run python scripts/install_endpoint_state.py \\
        --release-url https://github.com/.../releases/download/v1.0.0/bw-hackathon-data-2026-XX-XX.tar.gz \\
        --expected-sha <sha256>
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import tarfile
import tempfile
from pathlib import Path

import httpx


def _sha256_stream(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-url", required=True)
    parser.add_argument("--expected-sha", required=True)
    args = parser.parse_args()

    state_dir = os.environ.get("BW_STATE_DIR")
    if not state_dir:
        raise SystemExit("BW_STATE_DIR not set")
    state_path = Path(state_dir)
    state_path.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        tarball_path = tmp / "release.tar.gz"
        print(f"[install] fetching {args.release_url}")
        with httpx.stream("GET", args.release_url, timeout=120.0, follow_redirects=True) as resp:
            resp.raise_for_status()
            with tarball_path.open("wb") as out:
                for chunk in resp.iter_bytes(1 << 20):
                    out.write(chunk)

        sha = _sha256_stream(tarball_path)
        if sha != args.expected_sha:
            raise SystemExit(f"[install] SHA mismatch: got {sha} expected {args.expected_sha}")

        with tarfile.open(tarball_path, "r:gz") as tar:
            members = [m for m in tar.getmembers() if "/endpoint/" in m.name and m.isfile()]
            for m in members:
                stem = m.name.split("/endpoint/", 1)[1]
                target = state_path / stem
                target.parent.mkdir(parents=True, exist_ok=True)
                extracted = tar.extractfile(m)
                if extracted is None:
                    continue
                with target.open("wb") as out:
                    shutil.copyfileobj(extracted, out)
                print(f"[install] {m.name} → {target}")


if __name__ == "__main__":
    main()
