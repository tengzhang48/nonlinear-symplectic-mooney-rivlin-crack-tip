#!/usr/bin/env python3
"""Deterministic gates for the principal claims backed by stored FEM data."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "data" / "fem" / "strip" / "summary.csv"
LEDGER = ROOT / "data" / "claims" / "principal_claims.json"


def gate(name: str, condition: bool, detail: str) -> None:
    print(f"[{'PASS' if condition else 'FAIL'}] {name}: {detail}")
    if not condition:
        raise AssertionError(name)


def main() -> None:
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    gate("claims ledger schema", ledger.get("schema_version") == 1,
         f"{len(ledger.get('claims', []))} claims")

    with SUMMARY.open(newline="", encoding="utf-8") as stream:
        rows = {row["tag"]: row for row in csv.DictReader(stream)}
    gate("stored strip sweep count", len(rows) == 14, f"{len(rows)} cases")

    all_g_errors = [float(row["GJ_err"]) for row in rows.values()]
    gate("pure-shear energy balance", max(all_g_errors) <= 0.003,
         f"maximum relative error {100 * max(all_g_errors):.3f}%")

    load_tags = ["MR_lam13", "MR_lam15", "MR_lam16", "MR_lam18", "MR_lam22"]
    mr = [rows[tag] for tag in load_tags]
    p_errors = [float(row["P_err"]) for row in mr]
    gate("parameter-free P(lambda)", max(p_errors) <= 0.031,
         f"relative-error range {100 * min(p_errors):.2f}–{100 * max(p_errors):.2f}%")

    j_errors = [abs(float(row["J_exp"]) + 0.25) for row in mr]
    open_errors = [abs(float(row["open_exp"]) - 0.5) for row in mr]
    inplane_errors = [abs(float(row["inplane_exp"]) - 1.25) for row in mr]
    gate("MR radial exponents", max(j_errors) < 0.004
         and max(open_errors) < 0.055 and max(inplane_errors) < 0.025,
         f"max |Δp|: J={max(j_errors):.4f}, opening={max(open_errors):.4f}, "
         f"in-plane={max(inplane_errors):.4f}")

    mr_spreads = [float(row["spread"]) for row in mr]
    control_spreads = [float(rows[tag]["spread"])
                       for tag in ("NH_lam15", "NH_lam18")]
    gate("I2-specific angular plateau contrast",
         max(mr_spreads) < 0.10 and min(control_spreads) > 1.0,
         f"MR {100 * min(mr_spreads):.1f}–{100 * max(mr_spreads):.1f}% "
         f"versus controls {100 * min(control_spreads):.1f}–"
         f"{100 * max(control_spreads):.1f}%")

    mesh_p = [float(rows[tag]["P_meas"])
              for tag in ("MESH_nr48", "MESH_nr96", "MR_lam16")]
    mesh_spread = (max(mesh_p) - min(mesh_p)) / sum(mesh_p) * 3
    gate("strip mesh sensitivity", mesh_spread < 0.002,
         f"relative P range {100 * mesh_spread:.3f}%")

    disk = subprocess.run(
        [sys.executable, str(ROOT / "fem" / "check_new_signatures.py")],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(disk.stdout, end="")
    gate("disk stress/tip-shape cross-check", disk.returncode == 0,
         f"return code {disk.returncode}")

    print("All principal stored-data claims passed.")


if __name__ == "__main__":
    main()
