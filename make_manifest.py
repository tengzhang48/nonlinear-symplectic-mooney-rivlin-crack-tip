#!/usr/bin/env python3
"""Write the SHA-256 manifest for tracked publication files."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "MANIFEST.sha256"


def tracked_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, check=True, text=True,
        capture_output=True,
    )
    return [ROOT / name for name in result.stdout.splitlines()
            if name and name != OUT.name]


def main() -> None:
    lines = []
    for path in tracked_files():
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.relative_to(ROOT).as_posix()}")
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT.name}: {len(lines)} files")


if __name__ == "__main__":
    main()
