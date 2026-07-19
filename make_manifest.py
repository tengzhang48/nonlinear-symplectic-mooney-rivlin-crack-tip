#!/usr/bin/env python3
"""Write the SHA-256 manifest for tracked publication files."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "MANIFEST.sha256"
STRIP = ROOT / "data" / "fem" / "strip"
STRIP_OUT = STRIP / "ARTIFACTS.sha256"


def digest_lines(paths: list[Path], relative_to: Path) -> list[str]:
    lines = []
    for path in paths:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.relative_to(relative_to).as_posix()}")
    return lines


def write_strip_manifest() -> int:
    """Lock the curated strip scalars, rays, summary, and full fields."""
    members = sorted(
        [
            *STRIP.glob("ps_*.json"),
            *STRIP.glob("rays_*.csv"),
            *STRIP.glob("psfield_*.npz"),
            STRIP / "summary.csv",
        ],
        key=lambda path: path.name,
    )
    if not all(path.is_file() for path in members):
        raise FileNotFoundError("curated strip artifact is missing")
    STRIP_OUT.write_text(
        "\n".join(digest_lines(members, STRIP)) + "\n", encoding="utf-8"
    )
    return len(members)


def tracked_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, check=True, text=True,
        capture_output=True,
    )
    return [ROOT / name for name in result.stdout.splitlines()
            if name and name != OUT.name]


def main() -> None:
    strip_count = write_strip_manifest()
    lines = digest_lines(tracked_files(), ROOT)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {STRIP_OUT.relative_to(ROOT)}: {strip_count} files")
    print(f"wrote {OUT.name}: {len(lines)} files")


if __name__ == "__main__":
    main()
