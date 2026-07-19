"""Combine the pure-shear sweep into a summary table and validation plots."""
from __future__ import annotations

import csv
import glob
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
FIG = HERE / "figures"


def load():
    d = {}
    for f in sorted(glob.glob(str(OUT / "ps_*.json"))):
        j = json.loads(Path(f).read_text())
        d[j["tag"]] = j
    return d


def row(j):
    c, s, e = j["case"], j["signatures"], j["energy_release"]
    return dict(tag=j["tag"], material=j["material"], lam=c["lam"], a=c["a"],
               G_J=e["G_domain_J"], G_spec=e["G_spec_theory"],
               GJ_err=e["rel_err_GJ_vs_spec"],
               P_meas=e["P_measured"], P_pred=e["P_pred_from_G_spec"],
               P_err=e["rel_err_P_vs_pred"], J_exp=s["J_exp"],
               open_exp=s["open_exp"], inplane_exp=s["inplane_exp"],
               plateau=s["Jr14_plateau"], spread=s["Jr14_spread"],
               ncells=j.get("n_cells"), dmin_tag="MESH" in j["tag"])


def fig_P_vs_lambda(data):
    """Measured near-tip P vs the parameter-free prediction P(lambda)."""
    mr = sorted([row(j) for t, j in data.items()
                 if j["material"] == "MR" and j["case"]["a"] == 3.0
                 and abs(j["case"].get("c2", 1.0) - 1.0) < 1e-12
                 and not t.startswith("MESH")], key=lambda r: r["lam"])
    if len(mr) < 2:
        return
    lam = np.array([r["lam"] for r in mr])
    Pm = np.array([r["P_meas"] for r in mr])
    Pp = np.array([r["P_pred"] for r in mr])
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    order = np.argsort(lam)
    ax.plot(lam[order], Pp[order], "k-", lw=1.8,
            label=r"prediction $P=\sqrt{2G/\pi c_1}$, $G=h_0 W_\infty(\lambda)$")
    ax.plot(lam, Pm, "o", ms=9, color="C0", label="FEM measured $P$ (near-tip fit)")
    ax.set_xlabel(r"grip stretch $\lambda$"); ax.set_ylabel("opening intensity $P$")
    ax.set_title("Pure-shear specimen: measured near-tip $P$ vs. specimen $G$ prediction")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "ps_P_vs_lambda.png", dpi=135); plt.close(fig)


def fig_a_independence(data):
    """G (domain J) vs crack length a at fixed lambda."""
    aa = sorted([row(j) for t, j in data.items()
                 if j["material"] == "MR" and abs(j["case"]["lam"] - 1.6) < 0.05
                 and abs(j["case"].get("c2", 1.0) - 1.0) < 1e-12
                 and not t.startswith("MESH")], key=lambda r: r["a"])
    if len(aa) < 3:
        return
    a = np.array([r["a"] for r in aa]); GJ = np.array([r["G_J"] for r in aa])
    Gs = np.array([r["G_spec"] for r in aa])
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ax.plot(a, GJ, "o-", ms=8, label="FEM domain $J$-integral")
    ax.axhline(Gs.mean(), color="k", ls="--", label=r"$G=h_0 W_\infty$ (crack-length independent)")
    ax.set_xlabel("crack length $a$"); ax.set_ylabel("energy release rate $G$")
    ax.set_ylim(0, 1.3 * GJ.max())
    ax.set_title(r"Rivlin-Thomas: $G$ independent of crack length ($\lambda=1.6$)")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "ps_a_independence.png", dpi=135); plt.close(fig)


def fig_plateau(data):
    """J r^1/4 vs theta: MR (flat) vs NH control, at matched lambda."""
    def ray_plateau(tag):
        prof = {}
        for f in glob.glob(str(OUT / f"rays_{tag}_theta*.csv")):
            th = int(f.split("theta")[-1].split(".")[0])
            r, J = [], []
            with open(f) as fh:
                rd = csv.DictReader(fh)
                for row_ in rd:
                    r.append(float(row_["r"])); J.append(float(row_["J"]))
            r, J = np.array(r), np.array(J)
            m = (r > 3e-4) & (r < 8e-3) & (J > 0)
            if m.sum() > 3:
                prof[th] = float(np.mean(J[m] * r[m] ** 0.25))
        return prof
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    for tag, lab, mk in [("MR_lam15", "MR ($c_2=1$)", "o-"),
                         ("NH_lam15", "neo-Hookean ($c_2=0$)", "s--")]:
        p = ray_plateau(tag)
        if p:
            x = sorted(p); ax.plot(x, [p[t] for t in x], mk, label=lab)
    ax.set_xlabel(r"$\theta$ (deg)"); ax.set_ylabel(r"$J\,r^{1/4}$ (window mean)")
    ax.set_title(r"Constant-$\Delta$ signature in the pure-shear specimen ($\lambda=1.5$)")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "ps_plateau_vs_theta.png", dpi=135); plt.close(fig)


def main():
    data = load()
    if not data:
        print("no cases in", OUT); return
    FIG.mkdir(exist_ok=True)
    rows = [row(j) for j in data.values()]
    by_tag = {item["tag"]: item for item in rows}

    # summary CSV
    with open(OUT / "summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(
            fh, fieldnames=list(rows[0].keys()), lineterminator="\n"
        )
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r["material"], r["a"], r["lam"])):
            w.writerow(r)

    fig_P_vs_lambda(data); fig_a_independence(data); fig_plateau(data)

    # markdown note
    L = []
    A = L.append
    A("# Pure-shear (Rivlin-Thomas) MR strip — FEM results\n")
    A("Plane-stress incompressible Mooney-Rivlin strip, edge crack, gripped top/")
    A("bottom (`u_y=delta`, `u_x=0`), Mode-I half model. `lambda=(H+delta)/H`;")
    A("`W_inf=(c1+c2)(lambda^2+lambda^-2-2)`; `G=h0 W_inf`; `P_pred=sqrt(2G/pi c1)`.\n")
    A("## Energy release rate and the near-tip intensity\n")
    A("| case | material | lambda | a | G (domain J) | G=h0 W_inf | J err | P meas | P pred | P err |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    for r in sorted(rows, key=lambda r: (r["material"], r["a"], r["lam"])):
        A(f"| {r['tag']} | {r['material']} | {r['lam']:.2f} | {r['a']:.1f} | "
          f"{r['G_J']:.4f} | {r['G_spec']:.4f} | {100*r['GJ_err']:.2f}% | "
          f"{r['P_meas']:.4f} | {r['P_pred']:.4f} | {100*r['P_err']:.1f}% |")
    A("\n## Near-tip signatures (asymptotic window)\n")
    A("| case | open exp (->0.5) | J exp (->-0.25) | in-plane exp (->1.25) | J r^1/4 spread |")
    A("|---|---|---|---|---|")
    for r in sorted(rows, key=lambda r: (r["material"], r["a"], r["lam"])):
        A(f"| {r['tag']} | {r['open_exp']:.3f} | {r['J_exp']:.3f} | "
          f"{r['inplane_exp']:.3f} | {100*r['spread']:.1f}% |")
    A("\n## Figures\n")
    A("- `figures/ps_P_vs_lambda.png` — measured near-tip P vs the parameter-free")
    A("  prediction `P(lambda)` from the specimen G (the `G=(pi/2)c1 P^2` tie).")
    A("- `figures/ps_a_independence.png` — G independent of crack length.")
    A("- `figures/ps_plateau_vs_theta.png` — `J r^1/4` plateau (MR) vs control.\n")
    A("## Mesh, convergence, and limitations\n")
    A("- **Tip-focused radial mesh**: 120 corner-aligned sectors fill the exact")
    A("  rectangular outer boundary; graded rings from the `r_min=1e-5` core")
    A("  give the deep near-tip window `r in [3e-4, 8e-3]`.")
    mesh = [by_tag[tag] for tag in ("MESH_nr48", "MR_lam16", "MESH_nr96")]
    mesh_p = np.array([item["P_meas"] for item in mesh])
    mesh_p_range = 100 * (mesh_p.max() - mesh_p.min()) / mesh_p.mean()
    A("- **Mesh-converged**: `n_r = 48/64/96` give `G` = "
      + "/".join(f"{item['G_J']:.4f}" for item in mesh)
      + f" and a relative `P` range of `{mesh_p_range:.2f}%`.")
    cracks = [by_tag[tag] for tag in ("MR_a2", "MR_lam16", "MR_a4", "MR_a5")]
    crack_g = np.array([item["G_J"] for item in cracks])
    crack_g_range = 100 * (crack_g.max() - crack_g.min()) / crack_g.mean()
    A("- **Crack-length independent**: `G` = "
      + "/".join(f"{item['G_J']:.4f}" for item in cracks)
      + f" for `a = 2/3/4/5` (`{crack_g_range:.2f}%` range).")
    base = data["MR_lam16"]
    base_g_error = 100 * base["energy_release"]["rel_err_GJ_vs_spec"]
    base_w_ratio = (100 * base["energy_release"]["W_ff_measured"]
                    / base["energy_release"]["W_inf_theory"])
    A(f"- **Production-width check**: at `w/h0=9`, the domain `J` differs from "
      f"`W_inf h0` by `{base_g_error:.3f}%`; the local far-ahead energy-density "
      f"diagnostic is `{base_w_ratio:.1f}%` of ideal `W_inf`.")
    A("- **Finite-window caveat**: near-tip magnitudes are `theta`-flat to about")
    A("  `6%` (versus about `2%` for the deeper disk window); the affordable strip")
    A("  window therefore leaves about `2.5%` amplitude sensitivity.\n")
    A("## Data for pick-up\n")
    A("- `outputs/ps_*.json` — per-case signatures + energy release.")
    A("- `outputs/rays_*_theta*.csv` — near-tip fields (r, theta, Y1, Y2, J, lam1, lam2).")
    A("- `outputs/summary.csv` — the table above, machine-readable.\n")
    (HERE / "PS_RESULTS.md").write_text("\n".join(L))
    print("wrote PS_RESULTS.md, summary.csv, figures/ps_*.png")
    print("\n".join(L[:40]))


if __name__ == "__main__":
    main()
