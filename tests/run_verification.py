#!/usr/bin/env python3
"""Run the paper-scope analytic verification programs with expected sentinels.

Exploratory higher-order and historical-scaffold scripts under ``analysis/``
are intentionally outside this release gate.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Check:
    path: str
    sentinel: str
    cwd: str


CHECKS = [
    Check("verification/verify_equations.py", "PASSED 35   FAILED 0", "."),
    Check("analysis/leading_field.py", "Leading-field self-checks passed.", "analysis"),
    Check("analysis/check_g_selection.py", "PASSED 4/4", "analysis"),
    Check("analysis/check_row1_flux.py", "[PASS] row-1 flux", "analysis"),
    Check("analysis/energy_release_rate.py", "Energy-release-rate self-checks passed", "analysis"),
    Check("analysis/derive_constraint_row.py", "[PASS] derived linearized constraint", "analysis"),
    Check("analysis/second_variation_reduction.py", "GATES: 14/14 passed", "analysis"),
    Check("analysis/reduction_steps234.py", "GATES: ALL PASS", "analysis"),
]


def main() -> None:
    env = dict(os.environ, PYTHONHASHSEED="0")
    failures: list[str] = []
    for index, check in enumerate(CHECKS, start=1):
        path = ROOT / check.path
        print(f"[{index:02d}/{len(CHECKS):02d}] {check.path}", flush=True)
        result = subprocess.run(
            [sys.executable, str(path)],
            cwd=ROOT / check.cwd,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=600,
        )
        if result.returncode != 0 or check.sentinel not in result.stdout:
            failures.append(check.path)
            print(result.stdout)
            print(f"FAILED: return={result.returncode}, "
                  f"sentinel={check.sentinel!r}")
        else:
            matches = [line.strip() for line in result.stdout.splitlines()
                       if check.sentinel in line]
            print(f"  PASS: {matches[-1] if matches else check.sentinel}")

    if failures:
        raise SystemExit("analytic verification failed: " + ", ".join(failures))
    print(f"All {len(CHECKS)} public verification programs passed.")


if __name__ == "__main__":
    main()
