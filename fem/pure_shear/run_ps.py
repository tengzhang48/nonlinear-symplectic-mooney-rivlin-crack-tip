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
    ap.add_argument(
        "--export-p2-profile", action="store_true",
        help=("write a dense polar profile evaluated directly from the live "
              "P2 displacement, including theta=0 and 180 degrees"))
    ap.add_argument("--profile-r-lo", type=float, default=None)
    ap.add_argument("--profile-r-hi", type=float, default=None)
    ap.add_argument("--profile-n-r", type=int, default=160)
    ap.add_argument(
        "--profile-n-theta", type=int, default=181,
        help="number of equally spaced angles on [0,180], including both faces")
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

    if args.export_p2_profile:
        profile_r_lo = (args.win_lo * 0.5 if args.profile_r_lo is None
                        else args.profile_r_lo)
        profile_r_hi = (args.win_hi * 2.0 if args.profile_r_hi is None
                        else args.profile_r_hi)
        if not (args.r_min < profile_r_lo < profile_r_hi <= args.H):
            raise ValueError(
                "profile radii must satisfy r_min < r_lo < r_hi <= H")
        if args.profile_n_r < 2 or args.profile_n_theta < 3:
            raise ValueError(
                "profile-n-r must be >=2 and profile-n-theta must be >=3")
        profile_r = np.logspace(
            np.log10(profile_r_lo), np.log10(profile_r_hi), args.profile_n_r)
        profile_theta = np.linspace(0.0, 180.0, args.profile_n_theta)
        profile = pex.sample_p2_polar_grid(
            res, profile_theta, profile_r)
        profile_path = outdir / f"p2profile_{args.tag}.npz"
        np.savez_compressed(
            profile_path,
            **profile,
            c1=res["c1"], c2=res["c2"],
            lam_target=res["lam_target"], lam_reached=res["lam_reached"],
            a=res["a"], b=res["b"], H=res["H"],
            core_treatment="excised-traction-free-semicircle",
            r_min=res["info"]["r_min"],
            element_family="Lagrange", displacement_degree=2,
            n_r_mesh=res["info"]["n_r"],
            n_theta_mesh=res["info"]["n_theta_base"],
            n_sectors=res["info"]["n_sectors"],
            angular_scheme=res["info"]["angular_scheme"],
            corner_angles=np.asarray(res["info"]["corner_angles"]),
            sampling_scheme="controlled-log-radii/equispaced-angles",
        )
        if not np.all(profile["valid"]):
            n_bad = int(np.size(profile["valid"]) -
                        np.count_nonzero(profile["valid"]))
            raise RuntimeError(
                f"{n_bad} requested P2 profile samples lie outside the mesh")
        out["protocol"]["p2_profile"] = {
            "path": profile_path.name,
            "source_field": "live P2 displacement (no P1 projection)",
            "r_range": [profile_r_lo, profile_r_hi],
            "n_r_samples": args.profile_n_r,
            "theta_range_deg": [0.0, 180.0],
            "n_theta_samples": args.profile_n_theta,
            "includes_exact_crack_face": True,
        }

    (outdir / f"ps_{args.tag}.json").write_text(
        json.dumps(to_jsonable(out), indent=2))
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
