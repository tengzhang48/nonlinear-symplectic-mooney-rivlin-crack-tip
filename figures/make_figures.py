"""Reproducibility figures for the Mooney--Rivlin crack-tip project.

Reproducibility set (strip = physical specimen, disk = deep-window check):
  fig_master     Fig 1: specimen / circular cut with pseudo-time and (q,p) /
                 constrained tip state
  fig_hierarchy  historical scaffold diagnostic; not a completed coupled
                 operator (computed live from analysis/symplectic_dae.py)
  fig_ps_portrait, fig_chain, fig_plateau (strip data)
  fig_cratio     c2/c1 family maps (strip, deep window)
  fig_solution_compare, fig_sigma_G (disk deep window)
  fig_profile_correction (C_s-aware disk/strip profile audit)

Run from the repository root:  python figures/make_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FIG = HERE / "rendered"
FIG.mkdir(exist_ok=True)
FEMOUT = ROOT / "data" / "fem" / "disk"
WINDOW = (1e-4, 1e-3)

plt.rcParams.update({
    "font.size": 12, "axes.labelsize": 13.5, "axes.titlesize": 13,
    "legend.fontsize": 10.5, "xtick.labelsize": 11, "ytick.labelsize": 11,
    "lines.linewidth": 1.8, "lines.markersize": 6,
    "mathtext.fontset": "cm", "savefig.bbox": "tight", "savefig.dpi": 300,
})

MR_COLOR, NH_COLOR, TH_COLOR = "#c1272d", "#1f77b4", "0.15"


# ------------------------------------------------------------------ helpers
def load_case(tag):
    d = json.loads((FEMOUT / f"fem_case_{tag}.json").read_text())
    rays = {ray["theta_deg"]: {k: np.asarray(v) for k, v in ray.items()
                               if isinstance(v, list)} for ray in d["rays"]}
    return d, rays


def fit_face_P(rays):
    face = rays[max(rays)]
    r, y2 = face["r"], face["Y2"]
    lo, hi = WINDOW
    m = (r >= lo) & (r <= hi) & np.isfinite(y2)
    A = np.column_stack([np.sqrt(r[m]), r[m]])
    (P, _), *_ = np.linalg.lstsq(A, y2[m], rcond=None)
    return float(P)


def fit_regular_residual(r, q, window, exp=1.25):
    lo, hi = window
    m = (r >= lo) & (r <= hi) & np.isfinite(q)
    design = np.column_stack([np.ones(m.sum()), r[m], r[m] ** exp])
    scale = np.linalg.norm(design, axis=0)
    coefficients, *_ = np.linalg.lstsq(design / scale, q[m], rcond=None)
    c0, regular, residual = coefficients / scale
    return float(c0), float(regular), float(residual)


def window_mean_vs_theta(rays, value_fn, n_probe=5):
    lo, hi = WINDOW
    r_probe = np.logspace(np.log10(lo), np.log10(hi), n_probe)
    thetas, vals = [], []
    for thd in sorted(rays):
        ray = rays[thd]
        v = value_fn(ray)
        m = np.isfinite(v) & (v > 0)
        if m.sum() < 4:
            continue
        vv = np.exp(np.interp(np.log(r_probe), np.log(ray["r"][m]),
                              np.log(v[m])))
        thetas.append(thd)
        vals.append(float(np.mean(vv)))
    return np.array(thetas), np.array(vals)


def panel_label(ax, letter, x=0.02, y=0.985):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=13,
            fontweight="bold", va="top")


def save_pair(fig, stem):
    """Write a deterministic PDF/PNG pair without wall-clock PDF metadata."""
    fig.savefig(FIG / f"{stem}.pdf",
                metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(FIG / f"{stem}.png")


def fig_solution_compare():
    """The analytical solution (lines/curves) with the FEM on top (markers):
    (a) the three radial power laws along rays; (b) the angular profiles
    f(theta) = sin(theta/2) and g(theta), with FEM values at three radii
    collapsing onto the curves."""
    d, rays = load_case("MR_lam20")
    P = fit_face_P(rays)
    near = rays[min(rays)]                       # ~2 deg ray
    face = rays[max(rays)]                       # ~178 deg ray
    mid = rays[sorted(rays, key=lambda t: abs(t - 90))[0]]
    c0, b_near, _ = fit_regular_residual(
        near["r"], near["Y1"], WINDOW, 1.25
    )

    prof = np.load(ROOT / "data" / "analytic" / "mr_leading_profile.npz")
    th_p, g_p = prof["theta"], prof["g"]
    g_at = lambda t: np.interp(t, th_p, g_p)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.0, 4.3))

    # ---- (a) radial power laws --------------------------------------------
    rline = np.logspace(np.log10(4e-5), np.log10(2.5e-3), 100)
    sets = [
        (face["r"], face["Y2"], P * np.sqrt(rline),
         "#1f77b4", "o", r"opening $y_2$ on the face$\;\propto r^{1/2}$"),
        (near["r"], np.abs(near["Y1"] - c0 - b_near * near["r"]),
         P ** -0.5 * g_at(np.deg2rad(min(rays))) * rline ** 1.25,
         "#c1272d", "^", r"detrended in-plane residual$\;\propto r^{5/4}$"),
        (mid["r"], mid["J"], np.sqrt(P / 2.0) * rline ** -0.25,
         "#3d8c40", "s", r"Jacobian $J$ at $\theta=90^\circ\propto r^{-1/4}$"),
    ]
    for rr, vv, theo, color, mark, lab in sets:
        m = (rr >= 5e-5) & (rr <= 2e-3) & np.isfinite(vv) & (vv > 0)
        axA.loglog(rline, theo, color=color, lw=1.8)
        axA.loglog(rr[m][::2], vv[m][::2], ls="none", marker=mark, ms=6,
                   mfc="none", mew=1.5, color=color, label=lab)
    axA.axvspan(*WINDOW, color="0.5", alpha=0.10, zorder=0)
    axA.annotate("fit window", (1.1e-4, 11.5), fontsize=9.5, color="0.4")
    axA.annotate("slope $-1/4$", (4.5e-4, 8.2), color="#3d8c40",
                 fontsize=10.5)
    axA.annotate("slope $1/2$", (8e-4, 2.1e-2), color="#1f77b4",
                 fontsize=10.5)
    axA.annotate("slope $5/4$", (5.5e-4, 2.0e-5), color="#c1272d",
                 fontsize=10.5)
    axA.set_xlabel(r"distance from the tip $r$ (reference)")
    axA.set_ylabel("field value")
    axA.legend(frameon=False, fontsize=9.5, loc="center left",
               bbox_to_anchor=(0.02, 0.74))
    panel_label(axA, "a")

    # ---- (b) angular profiles ---------------------------------------------
    th_deg = np.array(sorted(rays))
    r_pick = [1e-4, 3e-4, 1e-3]
    fills = [0.35, 0.65, 1.0]
    thc = np.linspace(0, np.pi, 200)
    axB.plot(np.rad2deg(thc), np.sin(thc / 2), color="#1f77b4", lw=1.9,
             label=r"theory $f(\theta)=\sin(\theta/2)$")
    axB.plot(np.rad2deg(th_p), g_p, color="#c1272d", lw=1.9,
             label=r"theory residual $g(\theta)$")
    # opening profile: pointwise collapse of Y2/(P sqrt(r)) at three radii
    for rp, al in zip(r_pick, fills):
        fvals = []
        for tdeg in th_deg:
            ray = rays[tdeg]
            i = np.argmin(np.abs(ray["r"] - rp))
            fvals.append(ray["Y2"][i] / (P * np.sqrt(ray["r"][i])))
        axB.plot(th_deg, fvals, "o", color="#1f77b4", alpha=al, ms=6.5,
                 mfc="none", mew=1.6)
    # in-plane profile: the r^{5/4} term is tiny and rides on a smooth
    # background (c + b r from the outer field), so extract its amplitude by
    # a per-ray window fit  Y1 = c + b r + a r^{5/4}  and plot a sqrt(P).
    gvals = []
    lo, hi = WINDOW
    for tdeg in th_deg:
        ray = rays[tdeg]
        m = (ray["r"] >= lo * 0.5) & (ray["r"] <= hi * 2.0)
        rr, y1 = ray["r"][m], ray["Y1"][m]
        A = np.column_stack([np.ones_like(rr), rr, rr ** 1.25])
        coef, *_ = np.linalg.lstsq(A, y1, rcond=None)
        gvals.append(coef[2] * P ** 0.5)
    axB.plot(th_deg, gvals, "^", color="#c1272d", ms=7.5, mfc="none",
             mew=1.7)
    axB.plot([], [], "o", color="#1f77b4", mfc="none",
             label=r"FEM opening at $r=10^{-4},3{\cdot}10^{-4},10^{-3}$")
    axB.plot([], [], "^", color="#c1272d", mfc="none",
             label=r"FEM in-plane $r^{5/4}$ amplitude (per-ray fit)")
    axB.set_xlabel(r"angle $\theta$ (deg)")
    axB.set_ylabel("angular profile")
    axB.set_xticks([0, 45, 90, 135, 180])
    axB.set_ylim(-0.05, 2.95)
    axB.annotate(r"$g(\theta)$: residual", (155, 2.42), color="#c1272d",
                 fontsize=11)
    axB.annotate(r"$f(\theta)$: opening", (120, 0.62), color="#1f77b4",
                 fontsize=11)
    axB.legend(frameon=False, fontsize=9, loc="upper left")
    panel_label(axB, "b")

    fig.tight_layout()
    save_pair(fig, "fig_solution_compare")
    plt.close(fig)
    print(f"wrote fig_solution_compare  "
          f"(P = {P:.4f}, c0 = {c0:.4f}, b_2deg = {b_near:.4f})")


def fig_profile_correction():
    """Regenerate the C_s-aware profile audit and its structured output."""
    subprocess.run(
        [sys.executable, str(ROOT / "analysis" / "profile_mode_audit.py"),
         "--write"],
        cwd=ROOT,
        check=True,
    )


def fig_sigma_G():
    fig, ax = plt.subplots(figsize=(5.8, 4.3))
    style = {"MR_lam15": dict(color=MR_COLOR, marker="o", ls="--",
                              label=r"MR, $\lambda=1.5$"),
             "MR_lam20": dict(color=MR_COLOR, marker="^", ls="-",
                              label=r"MR, $\lambda=2.0$"),
             "NH_lam15": dict(color=NH_COLOR, marker="s", ls="--",
                              label=r"neo-Hookean, $\lambda=1.5$"),
             "NH_lam20": dict(color=NH_COLOR, marker="v", ls="-",
                              label=r"neo-Hookean, $\lambda=2.0$")}
    for tag, st in style.items():
        d, rays = load_case(tag)
        c1, c2 = d["c1"], d["c2"]
        P = fit_face_P(rays)
        pred = c1 * P ** 2 / 2.0

        def s1r(ray, c1=c1, c2=c2):
            lam1, lam2 = ray["lam1"], ray["lam2"]
            lam3 = 1.0 / (lam1 * lam2)
            return (2.0 * (lam1 ** 2 - lam3 ** 2)
                    * (c1 + c2 * lam2 ** 2) * ray["r"])

        th, v = window_mean_vs_theta(rays, s1r)
        ax.plot(th, v / pred, mfc="none", mew=1.5, **st)
    ax.axhspan(0.95, 1.05, color="0.5", alpha=0.12, zorder=0)
    ax.axhline(1.0, color="k", lw=1.8)
    ax.annotate(r"theory: $\sigma_{22}\,r=G/\pi$   ($\pm5\%$ band)",
                (6, 1.058), fontsize=11.5)
    ax.annotate("all four cases collapse onto one line",
                (28, 0.855), fontsize=11, color="0.35", style="italic")
    ax.set_xlabel(r"angle $\theta$ (deg)")
    ax.set_ylabel(r"near-tip stress  $\sigma_1 r\,/\,(G/\pi)$")
    ax.set_xticks([0, 45, 90, 135, 180])
    ax.set_ylim(0.72, 1.18)
    ax.legend(frameon=False, ncol=2, loc="lower left", columnspacing=1.0)
    save_pair(fig, "fig_sigma_G")
    plt.close(fig)
    print("wrote fig_sigma_G")


# =============================== pure-shear era figures (2026-07-03)
PSOUT = ROOT / "data" / "fem" / "strip"


def fig_master():
    """Fig 1: (a) Rivlin-Thomas strip specimen, (b) circular cut with
    pseudo-time and the (q, p) state, (c) constrained tip state (parabolas)."""
    fig = plt.figure(figsize=(12.8, 8.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1.0], hspace=0.06,
                          wspace=0.08)
    axA = fig.add_subplot(gs[0, :])
    axB = fig.add_subplot(gs[1, 0])
    axC = fig.add_subplot(gs[1, 1])

    # ---- (a) the specimen --------------------------------------------------
    a, b, H = 3.0, 6.0, 0.5
    axA.add_patch(Rectangle((-a, -H), a + b, 2 * H, fill=False, lw=1.6,
                            ec="0.2"))
    for sgn in (+1, -1):                                # hatched grips
        axA.add_patch(Rectangle((-a, sgn * H - (0.13 if sgn < 0 else 0)),
                                a + b, 0.13, fill=False, hatch="/////",
                                ec="0.45", lw=0))
        for x in (-2.0, 0.0, 2.0, 4.0):
            axA.annotate("", xy=(x, sgn * (H + 0.82)),
                         xytext=(x, sgn * (H + 0.16)),
                         arrowprops=dict(arrowstyle="-|>", color="0.35",
                                         lw=1.8))
    axA.annotate(r"clamped wide grips:  $u_y=\pm\delta$,  $u_x=0$"
                 r"  ($\lambda=1+2\delta/h$)",
                 (1.5, H + 1.02), fontsize=13, ha="center", color="0.25")
    axA.plot([-a, 0], [0, 0], color="k", lw=3.0, solid_capstyle="butt")
    axA.plot(0, 0, "ko", ms=4)
    axA.annotate("edge crack (traction-free)", (-a + 0.1, 0.15), fontsize=12.5)
    axA.annotate("tip (b)", (0.55, 0.13), fontsize=12.5)
    axA.add_patch(Rectangle((1.8, -H), b - 1.8 - 0.02, 2 * H, fc=NH_COLOR,
                            alpha=0.10, ec="none"))
    axA.annotate("pure shear $(1,\\lambda,1/\\lambda)$\n"
                 r"$W_\infty$ closed form $\Rightarrow\; G=h\,W_\infty$",
                 (3.9, 0.0), fontsize=11.5, ha="center", va="center",
                 color="0.15")
    # height dimension
    axA.annotate("", xy=(-a - 0.35, -H), xytext=(-a - 0.35, H),
                 arrowprops=dict(arrowstyle="<->", color="0.35", lw=1.1))
    axA.annotate("$h$", (-a - 0.82, -0.10), fontsize=13, color="0.35")
    # circle to panel (b)
    axA.add_patch(plt.Circle((0, 0), 0.42, fill=False, ec=MR_COLOR, lw=1.6,
                             ls=(0, (4, 2))))
    axA.set_xlim(-a - 1.0, b + 0.4)
    axA.set_ylim(-1.85, 1.9)
    axA.set_aspect("equal")
    axA.axis("off")
    panel_label(axA, "a", x=0.0, y=1.0)

    # ---- (b) the circular cut: pseudo-time and the state -------------------
    for k, rr in enumerate((1.0, 0.55, 0.3025)):        # equal factors
        axB.add_patch(plt.Circle((0, 0), rr, fill=False, ec="0.6", lw=1.1))
    axB.plot([-1.45, 0], [0, 0], color="k", lw=3.2, solid_capstyle="butt")
    axB.plot([0, 1.45], [0, 0], color="0.55", lw=1.0, ls=(0, (2, 2)))
    axB.plot(0, 0, "ko", ms=4)
    axB.annotate("faces", (-1.30, 0.09), fontsize=11)
    axB.annotate("ligament", (0.55, 0.09), fontsize=11, color="0.45")
    # xi arrow inward
    axB.annotate("", xy=(0.30 * 0.707, -0.30 * 0.707),
                 xytext=(1.0 * 0.707, -1.0 * 0.707),
                 arrowprops=dict(arrowstyle="-|>", color="0.3", lw=1.4))
    axB.annotate("equal radius factors\n= equal steps of $\\xi=\\ln r$;\n"
                 "the tip is $\\xi\\to-\\infty$",
                 (0.02, -1.62), fontsize=11, ha="center", color="0.25")
    # state on the outer circle: q (gray, tangential-ish) and p (red, inward)
    for th0 in np.deg2rad([35, 90, 145]):
        x0, y0 = np.cos(th0), np.sin(th0)
        axB.annotate("", xy=(x0 * 0.72, y0 * 0.72), xytext=(x0, y0),
                     arrowprops=dict(arrowstyle="-|>", color=MR_COLOR,
                                     lw=1.6))
        tx, ty = -np.sin(th0), np.cos(th0)
        axB.annotate("", xy=(x0 + 0.20 * tx + 0.1 * x0,
                             y0 + 0.20 * ty + 0.1 * y0),
                     xytext=(x0, y0),
                     arrowprops=dict(arrowstyle="-|>", color="0.45",
                                     lw=1.6))
    axB.annotate(r"$q(\theta)$: displacement trace",
                 (-1.05, 1.80), fontsize=11.5, color="0.35")
    axB.annotate(r"$p(\theta)=r\,P\,e_r$: force per"
                 "\nunit angle, inward",
                 (-1.05, 1.62), fontsize=11.5, color=MR_COLOR, va="top")
    axB.set_xlim(-1.6, 1.6)
    axB.set_ylim(-1.98, 1.85)
    axB.set_aspect("equal")
    axB.axis("off")
    panel_label(axB, "b", x=0.0, y=1.0)

    # ---- (c) constrained tip state (transplanted parabola art) -------------
    XL, YL = 1.55, 1.30
    for s0 in (0.06, 0.2, 0.45):
        x1 = np.linspace(-s0, XL, 300)
        x2 = 2.0 * np.sqrt(s0 * (x1 + s0))
        for sgn in (+1, -1):
            axC.plot(x1, sgn * x2, color=NH_COLOR, lw=1.7, alpha=0.9)
    for t0 in (0.06, 0.2, 0.45):
        x1 = np.linspace(-XL, t0, 300)
        x2 = 2.0 * np.sqrt(t0 * (t0 - x1))
        for sgn in (+1, -1):
            axC.plot(x1, sgn * x2, color=MR_COLOR, lw=1.7, alpha=0.9,
                     ls=(0, (5, 3)))
    axC.annotate("constant\nopening", (0.86, 0.96), color=NH_COLOR,
                 fontsize=11.5, bbox=dict(fc="white", ec="none", alpha=0.88, pad=1.2))
    axC.annotate("tension\ndirections", (-1.50, 0.94), color=MR_COLOR,
                 fontsize=11.5, bbox=dict(fc="white", ec="none", alpha=0.88, pad=1.2))
    axC.plot([-XL, 0], [0, 0], color="k", lw=4.0, solid_capstyle="butt")
    axC.plot(0, 0, "ko", ms=5)
    x0, y0 = 0.55, 0.75
    r0 = np.hypot(x0, y0)
    gs_ = np.array([(x0 / r0 - 1.0) / 2.0, y0 / (2.0 * r0)])
    e1 = gs_ / np.linalg.norm(gs_)
    e2 = np.array([-e1[1], e1[0]])
    axC.add_patch(plt.Circle((x0, y0), 0.035, color="0.25", zorder=5))
    for sgn in (+1, -1):
        axC.annotate("", xy=tuple((x0, y0) + sgn * 0.21 * e1),
                     xytext=(x0, y0), zorder=6,
                     arrowprops=dict(arrowstyle="-|>", color="k", lw=2.2))
        axC.annotate("", xy=tuple((x0, y0) + sgn * 0.09 * e2),
                     xytext=(x0, y0), zorder=6,
                     arrowprops=dict(arrowstyle="-|>", color="0.5", lw=1.4))
    axC.annotate("$\\lambda_1$", (x0 - 0.35, y0 + 0.20), fontsize=12.5,
                 bbox=dict(fc="white", ec="none", alpha=0.88, pad=1.2))
    axC.annotate("$\\lambda_2=\\lambda_3=\\lambda_1^{-1/2}$",
                 (0.72, 0.30), fontsize=11, color="0.35", bbox=dict(fc="white", ec="none", alpha=0.88, pad=1.2))
    axC.annotate("every point is in uniaxial tension:\na stretched fiber",
                 (-1.50, -1.16), fontsize=11.5, color="0.2", va="top",
                 bbox=dict(fc="white", ec="none", alpha=0.88, pad=1.2))
    axC.set_xlim(-XL, XL)
    axC.set_ylim(-YL - 0.22, YL + 0.22)
    axC.set_aspect("equal")
    axC.axis("off")
    panel_label(axC, "c", x=0.0, y=1.0)

    fig.tight_layout()
    save_pair(fig, "fig_master")
    plt.close(fig)
    print("wrote fig_master")


def fig_chain():
    """Fig: measured P vs closed-form P(lambda); a-collapse; G parity inset."""
    import csv
    rows = list(csv.DictReader(open(PSOUT / "summary.csv")))
    lam_c = np.linspace(1.22, 2.3, 300)

    def P_pred(lam, c1, c2, h0=1.0):
        W = (c1 + c2) * (lam ** 2 + lam ** -2 - 2)
        return np.sqrt(2 * h0 * W / (np.pi * c1))

    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    ax.plot(lam_c, P_pred(lam_c, 1, 1), color=MR_COLOR, lw=2.0,
            label=r"closed form $P(\lambda)$, Mooney-Rivlin")
    ax.plot(lam_c, P_pred(lam_c, 1, 0), color=NH_COLOR, lw=2.0,
            ls=(0, (5, 3)), label=r"closed form, neo-Hookean control")
    # The main curve is the c1=c2 load sweep.  The two c2/c1 variation
    # cases have their own predictions and belong in the ratio study.
    mr = [r for r in rows if r["material"] == "MR"
          and abs(float(r["a"]) - 3.0) < 0.1
          and "MESH" not in r["tag"] and "c2_" not in r["tag"]]
    ax.plot([float(r["lam"]) for r in mr], [float(r["P_meas"]) for r in mr],
            "o", color=MR_COLOR, ms=8, mec="k", mew=0.6, zorder=5,
            label="FEM, $a=3$")
    av = [r for r in rows if r["material"] == "MR"
          and abs(float(r["a"]) - 3.0) > 0.1]
    ax.plot([float(r["lam"]) for r in av], [float(r["P_meas"]) for r in av],
            "s", mfc="none", mec=MR_COLOR, ms=10, mew=1.6, zorder=6,
            label="FEM, $a=2,4,5$ (collapse)")
    nh = [r for r in rows if r["material"] != "MR"]
    ax.plot([float(r["lam"]) for r in nh], [float(r["P_meas"]) for r in nh],
            "^", color=NH_COLOR, ms=8, mec="k", mew=0.6, zorder=5,
            label="FEM, control")
    ax.set_xlabel(r"grip stretch $\lambda$")
    ax.set_ylabel(r"tip opening amplitude $P$")
    ax.legend(loc="upper left", framealpha=0.95)

    axi = ax.inset_axes([0.64, 0.17, 0.33, 0.30])
    axi.set_facecolor("white")
    axi.patch.set_alpha(1.0)
    gs_ = [float(r["G_spec"]) for r in rows]
    gj = [float(r["G_J"]) for r in rows]
    lim = (0.4, 8.0)
    axi.plot(lim, lim, color="0.4", lw=1.0)
    axi.plot(gs_, gj, "o", ms=4.5, color="0.15")
    axi.set_xscale("log"); axi.set_yscale("log")
    axi.set_xlim(lim); axi.set_ylim(lim)
    from matplotlib.ticker import FixedLocator, ScalarFormatter, NullLocator
    for axis in (axi.xaxis, axi.yaxis):
        axis.set_major_locator(FixedLocator([0.5, 1, 2, 4]))
        axis.set_major_formatter(ScalarFormatter())
        axis.set_minor_locator(NullLocator())
    axi.set_xlabel(r"$h\,W_\infty$", fontsize=9, labelpad=2.5)
    axi.set_ylabel("domain integral", fontsize=9, labelpad=1)
    axi.tick_params(labelsize=8)
    axi.text(0.06, 0.94, r"$\leq0.15\%$, all cases", transform=axi.transAxes,
             ha="left", va="top", fontsize=8.5)

    fig.tight_layout()
    save_pair(fig, "fig_chain")
    plt.close(fig)
    print(f"wrote fig_chain ({len(rows)} cases)")


def _load_psfield(tag):
    """Strip field npz -> mirrored triangulation colored by log10(lambda1)."""
    import matplotlib.tri as mtri
    dat = np.load(PSOUT / f"psfield_{tag}.npz")
    coords, disp, tris = dat["coords"], dat["disp"], dat["tris"]
    lam = float(dat["lam_reached"])
    xdef = coords + disp
    X1, X2, X3 = coords[tris[:, 0]], coords[tris[:, 1]], coords[tris[:, 2]]
    x1, x2, x3 = xdef[tris[:, 0]], xdef[tris[:, 1]], xdef[tris[:, 2]]
    Dm = np.stack([X2 - X1, X3 - X1], axis=-1)
    Ds = np.stack([x2 - x1, x3 - x1], axis=-1)
    F = Ds @ np.linalg.inv(Dm)
    C = np.transpose(F, (0, 2, 1)) @ F
    tr = C[:, 0, 0] + C[:, 1, 1]
    det = C[:, 0, 0] * C[:, 1, 1] - C[:, 0, 1] * C[:, 1, 0]
    lam1 = np.sqrt(0.5 * (tr + np.sqrt(np.maximum(tr ** 2 - 4 * det, 0.0))))
    N = coords.shape[0]
    ssum, scnt = np.zeros(N), np.zeros(N)
    for j in range(3):
        np.add.at(ssum, tris[:, j], lam1)
        np.add.at(scnt, tris[:, j], 1.0)
    lam1v = ssum / np.maximum(scnt, 1.0)
    xs = np.concatenate([xdef[:, 0], xdef[:, 0]])
    ys = np.concatenate([xdef[:, 1], -xdef[:, 1]])
    tri = mtri.Triangulation(xs, ys, np.vstack([tris, tris + N]))
    vals = np.log10(np.concatenate([lam1v, lam1v]))
    face = np.where(np.isclose(coords[:, 1], 0.0, atol=1e-12)
                    & (coords[:, 0] < -1e-12))[0]
    face = face[np.argsort(coords[face, 0])]
    # outer boundary polyline: true edge nodes, ordered by reference angle
    aa, bb, HH = float(dat["a"]), float(dat["b"]), float(dat["H"])
    tol = 1e-9
    on_edge = (np.isclose(coords[:, 0], -aa, atol=tol)
               | np.isclose(coords[:, 1], HH, atol=tol)
               | np.isclose(coords[:, 0], bb, atol=tol))
    eidx = np.where(on_edge)[0]
    ang = np.arctan2(coords[eidx, 1], coords[eidx, 0])
    eidx = eidx[np.argsort(ang)]          # theta: 0 (right) -> pi (left)
    return dict(coords=coords, xdef=xdef, tri=tri, vals=vals, lam=lam,
                face=face, edge=eidx,
                tipx=float(xdef[face[-1], 0]), a=aa, b=bb, H=HH)


def fig_ps_portrait(tags=("MR_lam13", "MR_lam16", "MR_lam22"), wp=0.25):
    """Portrait: (a) whole strip lam=1.6; (b) tip windows at 3 loads;
    (c) x14 tip end at lam=1.6."""
    snaps = {t: _load_psfield(t) for t in tags}
    mid = snaps["MR_lam16"]

    fig = plt.figure(figsize=(13.4, 7.6))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.15, 1.0], hspace=0.18,
                          wspace=0.14)
    axA = fig.add_subplot(gs[0, :])
    axBs = [fig.add_subplot(gs[1, k]) for k in range(3)]
    axC = fig.add_subplot(gs[1, 3])

    def draw(ax, snap, kw, lw_face=1.6):
        tp = ax.tripcolor(snap["tri"], snap["vals"], shading="gouraud",
                          rasterized=True, **kw)
        for sgn in (+1, -1):
            ax.plot(snap["xdef"][snap["face"], 0],
                    sgn * snap["xdef"][snap["face"], 1], color="k",
                    lw=lw_face)
            ax.plot(snap["xdef"][snap["edge"], 0],
                    sgn * snap["xdef"][snap["edge"], 1], color="0.25",
                    lw=1.0)
        return tp

    kwA = dict(cmap="viridis", vmin=0.0, vmax=np.log10(6.0))
    tpA = draw(axA, mid, kwA)
    tipx = mid["tipx"]
    axA.add_patch(Rectangle((tipx - 1.4 * wp, -1.15 * wp), 2.55 * wp,
                            2.3 * wp, fill=False, ec="w", lw=1.4,
                            ls=(0, (4, 2))))
    axA.annotate("near-tip window (b)", (tipx + 1.3 * wp, 1.3 * wp),
                 fontsize=10, color="w")
    axA.set_aspect("equal")
    axA.axis("off")
    axA.set_title(r"computed pure-shear strip, $\lambda=1.6$"
                  r"   (color: $\log_{10}\lambda_1$)", fontsize=12)
    panel_label(axA, "a", x=0.005, y=1.02)
    cbA = fig.colorbar(tpA, ax=axA, fraction=0.025, pad=0.01)
    cbA.set_label(r"$\log_{10}\lambda_1$", fontsize=10)

    for k, t in enumerate(tags):
        s = snaps[t]
        tp = draw(axBs[k], s, kwA)
        tx = s["tipx"]
        axBs[k].plot(tx, 0, "k.", ms=4)
        axBs[k].set_xlim(tx - 1.4 * wp, tx + 1.15 * wp)
        axBs[k].set_ylim(-1.15 * wp, 1.15 * wp)
        axBs[k].set_aspect("equal")
        axBs[k].axis("off")
        axBs[k].set_title(rf"$\lambda={s['lam']:g}$", fontsize=11)
        if k == 0:
            panel_label(axBs[k], "b", x=0.02, y=1.04)
            axBs[k].plot([tx - 1.3 * wp, tx - 1.3 * wp + 0.05],
                         [-1.02 * wp] * 2, color="k", lw=2.5)
            axBs[k].annotate(r"$0.05\,h$", (tx - 1.27 * wp, -0.93 * wp),
                             fontsize=9)

    w2 = wp / 14.0
    kwC = dict(cmap="viridis", vmin=0.0, vmax=np.log10(40.0))
    tpC = draw(axC, mid, kwC, lw_face=2.0)
    axC.plot(tipx, 0, "wo", ms=4, mec="k")
    axC.set_xlim(tipx - 1.5 * w2, tipx + 1.15 * w2)
    axC.set_ylim(-1.05 * w2, 1.05 * w2)
    axC.set_aspect("equal")
    axC.axis("off")
    axC.set_title(r"tip end $\times14$ ($\lambda=1.6$)", fontsize=11)
    panel_label(axC, "c", x=0.02, y=1.04)
    axC.annotate("open crack\n(gap wider\nthan frame)",
                 (tipx - 1.45 * w2, 0.45 * w2), fontsize=8.5, color="0.35")
    axC.annotate("$180^\\circ$: end wall,\nblunt", (tipx + 0.1 * w2,
                 -0.75 * w2), fontsize=8.5, color="w")
    cbC = fig.colorbar(tpC, ax=axC, fraction=0.05, pad=0.02)
    cbC.set_label(r"$\log_{10}\lambda_1$", fontsize=9)

    save_pair(fig, "fig_ps_portrait")
    plt.close(fig)
    print("wrote fig_ps_portrait:",
          ", ".join(f"{t}: lam={snaps[t]['lam']:g}" for t in tags))


def _ps_rays(tag):
    import csv
    rays = {}
    physical_columns = ("r", "theta_deg", "Y1", "Y2", "J", "lam1", "lam2")
    for th in (2, 45, 90, 135, 178):
        f = PSOUT / f"rays_{tag}_theta{th}.csv"
        if not f.exists():
            continue
        rows = list(csv.DictReader(open(f)))
        rays[th] = {key: np.array([float(row[key]) for row in rows])
                    for key in physical_columns}
    return rays


def _ps_summary():
    import csv
    return {r["tag"]: r for r in csv.DictReader(open(PSOUT / "summary.csv"))}


def fig_plateau_ps(window=(3e-4, 8e-3)):
    """J r^{1/4}/sqrt(P/2) vs theta for MR and control at two loads (strip)."""
    S = _ps_summary()
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    for tag, color, mk, lab in (
            ("MR_lam15", MR_COLOR, "o", r"Mooney-Rivlin, $\lambda=1.5$"),
            ("MR_lam18", MR_COLOR, "D", r"Mooney-Rivlin, $\lambda=1.8$"),
            ("NH_lam15", NH_COLOR, "^", r"control $c_2=0$, $\lambda=1.5$"),
            ("NH_lam18", NH_COLOR, "v", r"control $c_2=0$, $\lambda=1.8$")):
        rays = _ps_rays(tag)
        P = float(S[tag]["P_meas"])
        ths, vals = [], []
        for t in sorted(rays):
            ray = rays[t]
            mm = (ray["r"] >= window[0]) & (ray["r"] <= window[1])
            vals.append(np.mean(ray["J"][mm] * ray["r"][mm] ** 0.25)
                        / np.sqrt(P / 2))
            ths.append(t)
        ls = "-" if tag.startswith("MR") else (0, (4, 2))
        ax.plot(ths, vals, mk, ls=ls, color=color, ms=8, mec="k", mew=0.5,
                lw=1.3, label=lab)
    ax.axhspan(0.95, 1.05, color="0.9", zorder=0)
    ax.axhline(1.0, color="0.4", lw=1.0)
    ax.annotate(r"prediction $\sqrt{P/2}$ ($\pm5\%$)", (95, 1.07),
                fontsize=10, color="0.35")
    ax.set_xlabel(r"$\theta$ (deg)")
    ax.set_ylabel(r"$J\,r^{1/4}/\sqrt{P/2}$  (each case: own $P$)")
    ax.set_yscale("log")
    ax.set_ylim(0.3, 4.5)
    ax.legend(fontsize=9.5, loc="lower left")
    fig.tight_layout()
    save_pair(fig, "fig_plateau")
    plt.close(fig)
    print("wrote fig_plateau (strip)")


def _psfield_Jratio(tag):
    """Mirrored triangulation colored by log2 of J r^{1/4}/sqrt(P/2)."""
    import matplotlib.tri as mtri
    dat = np.load(PSOUT / f"psfield_{tag}.npz")
    coords, disp, tris = dat["coords"], dat["disp"], dat["tris"]
    xdef = coords + disp
    X1, X2, X3 = coords[tris[:, 0]], coords[tris[:, 1]], coords[tris[:, 2]]
    x1, x2, x3 = xdef[tris[:, 0]], xdef[tris[:, 1]], xdef[tris[:, 2]]
    Dm = np.stack([X2 - X1, X3 - X1], axis=-1)
    Ds = np.stack([x2 - x1, x3 - x1], axis=-1)
    F = Ds @ np.linalg.inv(Dm)
    Jc = F[:, 0, 0] * F[:, 1, 1] - F[:, 0, 1] * F[:, 1, 0]
    N = coords.shape[0]
    jsum, cnt = np.zeros(N), np.zeros(N)
    for j in range(3):
        np.add.at(jsum, tris[:, j], Jc)
        np.add.at(cnt, tris[:, j], 1.0)
    Jv = jsum / np.maximum(cnt, 1.0)
    face = np.where(np.isclose(coords[:, 1], 0.0, atol=1e-12)
                    & (coords[:, 0] < -1e-12))[0]
    face = face[np.argsort(coords[face, 0])]
    rref = -coords[face, 0]
    y2f = xdef[face, 1]
    mm = (rref >= 3e-4) & (rref <= 8e-3)
    A2 = np.column_stack([np.sqrt(rref[mm]), rref[mm]])
    (P, _), *_ = np.linalg.lstsq(A2, y2f[mm], rcond=None)
    rv = np.hypot(coords[:, 0], coords[:, 1])
    ratio = Jv * rv ** 0.25 / np.sqrt(max(P, 1e-9) / 2.0)
    xs = np.concatenate([xdef[:, 0], xdef[:, 0]])
    ys = np.concatenate([xdef[:, 1], -xdef[:, 1]])
    tri = mtri.Triangulation(xs, ys, np.vstack([tris, tris + N]))
    vals = np.log2(np.maximum(np.concatenate([ratio, ratio]), 1e-3))
    return dict(tri=tri, vals=vals, xdef=xdef, face=face,
                tipx=float(xdef[face[-1], 0]), P=float(P),
                lam=float(dat["lam_reached"]))


def fig_cratio_ps(tags=(("psfield_NH_lam16", "0 (neo-Hookean)"),
                        ("psfield_MR_lam16_c2_third", "1/3"),
                        ("psfield_MR_lam16", "1"),
                        ("psfield_MR_lam16_c2_3", "3")), w=0.015):
    """One constrained tip for the family: J r^{1/4}/sqrt(P/2) maps at
    lam=1.6 (strip), deep window, with a scale bar."""
    from matplotlib.colors import TwoSlopeNorm
    snaps = [(_psfield_Jratio(t.replace("psfield_", "")), lab)
             for t, lab in tags]
    fig, axs = plt.subplots(1, len(snaps), figsize=(3.3 * len(snaps), 3.6))
    axs = np.atleast_1d(axs)
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-1.0, vmax=1.8)
    for k, (ax, (s, lab)) in enumerate(zip(axs, snaps)):
        tp = ax.tripcolor(s["tri"], s["vals"], shading="gouraud", norm=norm,
                          cmap="RdBu_r", rasterized=True)
        for sgn in (+1, -1):
            ax.plot(s["xdef"][s["face"], 0], sgn * s["xdef"][s["face"], 1],
                    color="k", lw=1.4)
        tx = s["tipx"]
        ax.set_xlim(tx - 1.3 * w, tx + 1.05 * w)
        ax.set_ylim(-1.15 * w, 1.15 * w)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(f"$c_2/c_1=$ {lab}", fontsize=11)
        panel_label(ax, "abcd"[k], x=0.02, y=1.03)
        if k == 0:
            ax.plot([tx - 1.2 * w, tx - 1.2 * w + 0.005], [-1.02 * w] * 2,
                    color="k", lw=2.5)
            ax.annotate(r"$0.005\,h$", (tx - 1.18 * w, -0.90 * w),
                        fontsize=9)
    cb = fig.colorbar(tp, ax=axs, fraction=0.03, pad=0.015,
                      ticks=[-1, 0, 1, np.log2(3)])
    cb.ax.set_yticklabels(["1/2", "1", "2", "3"])
    cb.set_label(r"$J\,r^{1/4}/\sqrt{P/2}$   (white $=$ on prediction)",
                 fontsize=10)
    save_pair(fig, "fig_cratio")
    plt.close(fig)
    print(f"wrote fig_cratio (strip, {len(snaps)} panels): "
          + ", ".join(f"P={s['P']:.3f}" for s, _ in snaps))


def fig_hierarchy():
    """Combined hierarchy figure: (a) the crank, (b) ladder + computed
    eigenvalues on one axis, (c) the 9/4 obstruction in plain terms."""
    import sys
    sys.path.insert(0, str(ROOT / "analysis"))
    from symplectic_dae import (assemble_pencil, classified_mode_roots,
                                finite_real_eigs, cluster,
                                reaction_basis_spectrum)
    from matplotlib.patches import FancyBboxPatch

    fig = plt.figure(figsize=(13.6, 7.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.08],
                          height_ratios=[0.44, 0.56], hspace=0.42,
                          wspace=0.16)
    axA = fig.add_subplot(gs[:, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 1])

    # ---- (a) the crank ------------------------------------------------------
    def box(xy, w, h, text, fc="0.96", ec="0.25", fs=12.5):
        axA.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.07",
                                     fc=fc, ec=ec, lw=1.3))
        axA.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center",
                 va="center", fontsize=fs)

    def arr(p, q, text="", dx=0.12, fs=11, color="0.25"):
        axA.annotate("", xy=q, xytext=p,
                     arrowprops=dict(arrowstyle="-|>", color=color, lw=1.6))
        if text:
            axA.text((p[0] + q[0]) / 2 + dx, (p[1] + q[1]) / 2, text,
                     fontsize=fs, color=color, ha="left", va="center")

    box((1.3, 8.9), 4.2, 0.95, "levels $<k$ solved")
    arr((3.4, 8.9), (3.4, 8.15), "residue $=$ forcing")
    box((1.3, 7.15), 4.2, 1.0,
        "$(A-\\Lambda_k E)\\,\\varphi_k=$ forcing\n(same operator)", fs=12)
    arr((3.4, 7.15), (3.4, 6.4))
    box((1.85, 5.5), 3.1, 0.9, "$\\Lambda_k$ eigenvalue?", fc="0.99")
    arr((2.3, 5.5), (1.55, 4.7), "no", dx=-0.5)
    arr((4.5, 5.5), (5.25, 4.7), "yes", dx=0.12)
    box((0.1, 3.55), 2.9, 1.15, "solve; endpoints\nfix the constants",
        fs=11.5)
    box((3.95, 3.55), 2.9, 1.15,
        "$\\Omega$-solvable:\n$+$ free amplitude\n($P,\\,B,\\,Q_k$)",
        fs=11.5)
    arr((6.2, 3.55), (6.2, 2.75), "if not:", dx=-1.1, fs=10.5)
    box((4.2, 1.85), 2.45, 0.9, "half powers\n($\\Lambda=9/4$)",
        fc="#fbeaea", ec=MR_COLOR, fs=11.5)
    arr((1.55, 3.55), (1.55, 0.95))
    arr((5.0, 1.85), (4.2, 0.95))
    box((1.3, 0.05), 4.2, 0.9, "next level: $\\Lambda\\to\\Lambda+1/2$")
    axA.plot([5.5, 7.05], [0.5, 0.5], color="0.6", lw=1.5)
    axA.annotate("", xy=(7.05, 9.4), xytext=(7.05, 0.5),
                 arrowprops=dict(arrowstyle="-|>", color="0.6", lw=1.5))
    axA.plot([7.05, 5.5], [9.37, 9.37], color="0.6", lw=1.5)
    axA.text(7.28, 4.9, "repeat", fontsize=11.5, color="0.5", rotation=90,
             va="center")
    axA.set_xlim(-0.15, 7.75)
    axA.set_ylim(-0.3, 10.15)
    axA.axis("off")
    panel_label(axA, "a", x=0.0, y=1.0)

    # ---- (b) ladder + computed eigenvalues on one axis ----------------------
    th, A, E = assemble_pencil(N=40)
    raw = cluster(finite_real_eigs(A, E))
    win = np.array(classified_mode_roots(raw))
    mus = [(1.0, "filled", "shear $s$", 1),
           (1.25, "filled", "slave $g$", -1),
           (1.5, "filled", "shear $s^{3/2}$", 1),
           (2.0, "open", "$Q_2$", 1),
           (2.25, "flag", "obstruction", -1),
           (2.5, "filled", "reaction $f^{2}$", 1),
           (3.0, "open", "$Q_3$", 1)]
    axB.axhline(0.55, color="0.3", lw=1.1)
    for mu, kind, lab, side in mus:
        y0 = 0.55
        if kind == "open":
            axB.plot(mu, y0, "o", ms=13, mfc="white", mec=MR_COLOR,
                     mew=2.2, zorder=5)
        elif kind == "filled":
            axB.plot(mu, y0, "o", ms=10, color="0.35", zorder=4)
        else:
            axB.plot(mu, y0, "s", ms=13, mfc="#fbeaea", mec=MR_COLOR,
                     mew=2.2, zorder=5)
        if side > 0:
            axB.annotate(lab, (mu, 0.78), fontsize=10, ha="center",
                         va="bottom")
        else:
            axB.annotate(lab, (mu, 0.32), fontsize=10, ha="center",
                         va="top")
    # computed values, plotted beneath the same axis
    axB.plot(win, np.full_like(win, -0.25), "x", ms=9, mew=2.2,
             color="k", zorder=5)
    for v in win:
        axB.plot([v, v], [-0.25, 0.72], color="0.85", lw=0.9, zorder=1)
    axB.annotate("computed eigenvalues (collocation)", (0.95, -0.62),
                 fontsize=10.5, color="k")
    axB.annotate("the homogeneous modes, and what they are",
                 (0.95, 1.28), fontsize=11, color="0.25")
    for mu in (1.0, 1.5, 2.0, 2.5, 3.0):
        axB.annotate(f"{mu:g}", (mu, -0.05), fontsize=9.5, ha="center",
                     color="0.45")
    axB.set_xlim(0.88, 3.25)
    axB.set_ylim(-0.85, 1.5)
    axB.axis("off")
    panel_label(axB, "b", x=0.0, y=1.04)

    # ---- (c) the obstruction, plainly ---------------------------------------
    ms_ = (3, 4, 5, 6, 8)
    dsm, dhf = [], []
    for m in ms_:
        sm = np.array(reaction_basis_spectrum(list(range(2, 2 * m + 1, 2))))
        hf = np.array(reaction_basis_spectrum([1.5 + 0.5 * k
                                               for k in range(2 * m)]))
        dsm.append(min(abs(sm - 2.25)))
        dhf.append(max(min(abs(hf - 2.25)), 1e-16))
    axC.semilogy(ms_, dsm, "s-", color=MR_COLOR, ms=8, lw=1.8,
                 label="smooth functions only")
    axC.semilogy(ms_, dhf, "o-", color="0.3", ms=7, lw=1.8,
                 label="half powers allowed")
    axC.annotate("the $9/4$ mode never appears\n(gap stuck at $1/4$)",
                 (3.15, 2.5e-3), fontsize=10.5, color=MR_COLOR, va="top")
    axC.annotate("it appears exactly", (5.1, 2e-11), fontsize=10.5,
                 color="0.3")
    axC.set_xlabel("number of basis functions", fontsize=11)
    axC.set_ylabel("how far the $9/4$ mode is missed", fontsize=11)
    axC.set_xticks(ms_)
    axC.set_ylim(1e-16, 1.5)
    axC.legend(fontsize=10, loc="center right", framealpha=0.95)
    panel_label(axC, "c", x=-0.14, y=1.06)

    fig.tight_layout()
    save_pair(fig, "fig_hierarchy")
    plt.close(fig)
    print("wrote fig_hierarchy")

if __name__ == "__main__":
    required = [
        *(FEMOUT / f"fem_case_{tag}.json" for tag in
          ("MR_lam15", "MR_lam20", "NH_lam15", "NH_lam20")),
        ROOT / "data" / "analytic" / "mr_leading_profile.npz",
        PSOUT / "summary.csv",
        *(PSOUT / f"psfield_{tag}.npz" for tag in
          ("MR_lam13", "MR_lam16", "MR_lam22", "NH_lam16",
           "MR_lam16_c2_third", "MR_lam16_c2_3")),
        *(PSOUT / f"rays_{tag}_theta{theta}.csv" for tag in
          ("MR_lam15", "MR_lam16", "MR_lam18", "MR_lam22",
           "NH_lam15", "NH_lam18")
          for theta in (2, 45, 90, 135, 178)),
    ]
    missing = [path.relative_to(ROOT) for path in required if not path.exists()]
    if missing:
        raise SystemExit("missing figure inputs:\n  "
                         + "\n  ".join(map(str, missing)))

    # Reproducibility set: strip = physical specimen; disk = a deep-window
    # consistency check.  The profile figure is the C_s-aware replacement for
    # the withdrawn target-selected raw-tip-shape plot.
    fig_master()
    fig_hierarchy()
    fig_chain()
    fig_ps_portrait()
    fig_plateau_ps()
    fig_cratio_ps()
    fig_solution_compare()
    fig_profile_correction()
    fig_sigma_G()
    print("done ->", FIG)
