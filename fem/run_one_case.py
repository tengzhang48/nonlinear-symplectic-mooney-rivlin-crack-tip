"""Solve one (c1, c2, lambda) case on the fine near-tip mesh and save rays.

The JSON stores solve/mesh metadata and the sampled fields. Derived fits are
reported to stdout and intentionally not serialized: current analysis scripts
recompute them from the rays, avoiding stale estimator-dependent summaries.

Usage:
  python run_one_case.py --c1 1 --c2 1 --lam 2.0 --tag MR_lam20 --out outputs/
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from mr_fem_mesh import MeshConfig
from mr_fem_solve import SolveConfig, solve
from mr_fem_extract import run_tests


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c1", type=float, default=1.0)
    ap.add_argument("--c2", type=float, default=1.0)
    ap.add_argument("--lam", type=float, default=2.0)
    ap.add_argument("--tag", type=str, required=True)
    ap.add_argument("--out", type=str, default="outputs")
    ap.add_argument("--r_min", type=float, default=2e-6)
    ap.add_argument("--ratio", type=float, default=1.15)
    ap.add_argument("--n_theta", type=int, default=72)
    ap.add_argument("--n_steps", type=int, default=16)
    args = ap.parse_args()

    mcfg = MeshConfig(r_min=args.r_min, R=1.0, ratio=args.ratio, n_theta=args.n_theta)
    scfg = SolveConfig(c1=args.c1, c2=args.c2, lam_target=args.lam, n_steps=args.n_steps)

    t0 = time.time()
    res = solve(scfg, mcfg)
    solve_t = time.time() - t0

    window = (1e-4, 1e-3)
    t = run_tests(res, window, n_r=44)
    rays_out = [
        {k: (v.tolist() if isinstance(v, np.ndarray) else v)
         for k, v in ray.items()}
        for ray in t.pop("rays")
    ]

    out = {
        "tag": args.tag,
        "c1": args.c1, "c2": args.c2, "lam": args.lam,
        "material": "MR" if args.c2 != 0 else "NeoHookean",
        "mesh": {"r_min": args.r_min, "ratio": args.ratio, "n_theta": args.n_theta,
                 "n_cells": res["info"]["n_cells"], "n_points": res["info"]["n_points"]},
        "analysis_window": list(window),
        "solve_time_s": solve_t, "n_newton": res["n_newton"],
        "rays": rays_out,
    }
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    fpath = outdir / f"fem_case_{args.tag}.json"
    fpath.write_text(json.dumps(out, indent=2))
    print(f"[{args.tag}] solve {solve_t:.1f}s newton {res['n_newton']} -> {fpath}")
    # quick echo
    print(f"[{args.tag}] open_exp={t['p_open']:.3f} J_exp={t['J_exp_mean']:.3f} "
          f"plateau={t['plateau_mean']:.4f} spread={t['plateau_rel_spread']*100:.1f}% "
          f"sqrt(P/2)={t['sqrt_P_over_2']:.4f} T3err={t['T3_rel_err']*100:.1f}%")


if __name__ == "__main__":
    main()
