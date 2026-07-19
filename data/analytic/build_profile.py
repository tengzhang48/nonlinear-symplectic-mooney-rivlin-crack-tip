#!/usr/bin/env python3
"""Regenerate the selected leading angular profile deterministically."""

from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis"))

from leading_field import (  # noqa: E402
    A1, A2, DELTA, G0, f, fp, g_family, solve_g_leading,
)


def write_npz(path: Path, arrays: list[tuple[str, np.ndarray]]) -> None:
    """Write ordered NPY members with fixed ZIP metadata and compression."""
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=9) as archive:
        for name, array in arrays:
            stream = io.BytesIO()
            np.lib.format.write_array(stream, np.asanyarray(array),
                                      allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, stream.getvalue(), compress_type=zipfile.ZIP_DEFLATED,
                             compresslevel=9)


def main() -> None:
    theta = np.linspace(0.0, np.pi, 2001)
    solution = solve_g_leading()
    g, gp = g_family(solution)
    arrays = [
        ("theta", theta),
        ("f", f(theta)),
        ("fp", fp(theta)),
        ("g", g(theta)),
        ("gp", gp(theta)),
        ("a1", np.asarray(A1)),
        ("a2", np.asarray(A2)),
        ("Delta_const", np.asarray(DELTA)),
        ("g0", np.asarray(G0)),
    ]
    output = Path(__file__).with_name("mr_leading_profile.npz")
    write_npz(output, arrays)
    loaded = np.load(output, allow_pickle=False)
    print(f"wrote {output.relative_to(ROOT)}: {theta.size} points, "
          f"g(0)={loaded['g'][0]:.10f}, g(pi)={loaded['g'][-1]:.10f}")


if __name__ == "__main__":
    main()
