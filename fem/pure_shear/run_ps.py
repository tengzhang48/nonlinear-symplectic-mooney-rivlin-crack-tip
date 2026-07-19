"""Solve and analyze one pure-shear case; write JSON and per-ray CSV data."""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from ps_mesh import StripConfig
from ps_solve import SolveConfig, solve
from ps_extract import analyze
import ps_extract as pex


def to_jsonable(o):
    if isinstance(o, dict):
        return {k: to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c1", type=float, default=1.0)
    ap.add_argument("--c2", type=float, default=1.0)
    ap.add_argument("--lam", type=float, default=1.6)
    ap.add_argument("--a", type=float, default=3.0)
    ap.add_argument("--b", type=float, default=6.0)
    ap.add_argument("--H", type=float, default=0.5)
    ap.add_argument("--r_min", type=float, default=1e-5)
    ap.add_argument("--n_r", type=int, default=64)
    ap.add_argument("--n_theta", type=int, default=120)
    ap.add_argument("--n_steps", type=int, default=18)
    ap.add_argument("--win_lo", type=float, default=3e-4)
    ap.add_argument("--win_hi", type=float, default=8e-3)
    ap.add_argument("--export-field", action="store_true",
                    help="also write the full-field NPZ from this solve")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()

    mcfg = StripConfig(a=args.a, b=args.b, H=args.H,
                       r_min=args.r_min, n_r=args.n_r, n_theta=args.n_theta)
    scfg = SolveConfig(c1=args.c1, c2=args.c2, lam=args.lam, n_steps=args.n_steps)

    t0 = time.time()
    res = solve(scfg, mcfg)
    solve_t = time.time() - t0
    out = analyze(res, window=(args.win_lo, args.win_hi))
    out["tag"] = args.tag
    out["material"] = "MR" if args.c2 != 0 else "NeoHookean"
    out["solve_time_s"] = solve_t
    out["n_newton"] = res["n_newton"]
    out["n_cells"] = res["info"]["n_cells"]
    out["w_over_h0"] = res["info"]["w_over_h0"]
    out["protocol"] = {
        "n_steps": res["n_steps"],
        "fit_window": [args.win_lo, args.win_hi],
        "mesh": {
            "n_r": args.n_r,
            "n_theta_base": args.n_theta,
            "n_sectors": res["info"]["n_sectors"],
            "r_min": args.r_min,
            "angular_scheme": res["info"]["angular_scheme"],
            "corner_angles": res["info"]["corner_angles"],
        },
    }

    # per-ray sampling for CSV / plots
    rays = pex.mfx.sample_rays(res, (2, 45, 90, 135, 178),
                               args.win_lo * 0.5, args.win_hi * 2, n_r=40)

    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"ps_{args.tag}.json").write_text(json.dumps(to_jsonable(out), indent=2))
    for ray in rays:
        th = int(round(ray["theta_deg"]))
        with open(outdir / f"rays_{args.tag}_theta{th}.csv", "w", newline="") as fh:
            w = csv.writer(fh, lineterminator="\n")
            w.writerow(["r", "theta_deg", "Y1", "Y2", "J", "lam1", "lam2",
                        "n_cells", "n_steps", "n_r", "n_theta_base",
                        "n_sectors", "angular_scheme", "corner_angle_right",
                        "corner_angle_left"])
            for i in range(len(ray["r"])):
                w.writerow([ray["r"][i], th, ray["Y1"][i], ray["Y2"][i],
                            ray["J"][i], ray["lam1"][i], ray["lam2"][i],
                            res["info"]["n_cells"], res["n_steps"], args.n_r,
                            args.n_theta, res["info"]["n_sectors"],
                            res["info"]["angular_scheme"],
                            *res["info"]["corner_angles"]])

    if args.export_field:
        from ps_export import write_snapshot
        write_snapshot(res, args.tag, outdir)

    e = out["energy_release"]; s = out["signatures"]
    print(f"[{args.tag}] lam={res['lam_reached']:.3f} a={args.a} solve {solve_t:.0f}s "
          f"newton {res['n_newton']} cells {res['info']['n_cells']}")
    print(f"[{args.tag}] G_J={e['G_domain_J']:.4f} G_spec={e['G_spec_theory']:.4f} "
          f"(GJ err {100*e['rel_err_GJ_vs_spec']:.2f}%)  P_meas={e['P_measured']:.4f} "
          f"P_pred={e['P_pred_from_G_spec']:.4f} (P err {100*e['rel_err_P_vs_pred']:.1f}%)  "
          f"J_exp={s['J_exp']:.3f} plateau_spread={100*s['Jr14_spread']:.1f}%")


if __name__ == "__main__":
    main()
