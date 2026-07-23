"""Render the five current paper figures from analytic and strip FEM inputs.

Outputs:
  fig_master       Figure 1: specimen, radial state, constrained tip state
  fig_chain        Figure 2: pure-shear loading-to-tip-amplitude chain
  fig_ps_portrait  Figure 3: strip solution portraits
  fig_plateau      Figure 4: compensated-Jacobian angular comparison
  fig_cratio       Figure 5: material-ratio comparison

The quarantined disk boundary-value problem, withdrawn profile estimator, and
historical hierarchy scaffold are deliberately outside this dispatcher.

Run from the repository root:  python figures/make_figures.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FIG = HERE / "rendered"
FIG.mkdir(exist_ok=True)
PSOUT = ROOT / "data" / "fem" / "strip"

plt.rcParams.update({
    "font.size": 12, "axes.labelsize": 13.5, "axes.titlesize": 13,
    "legend.fontsize": 10.5, "xtick.labelsize": 11, "ytick.labelsize": 11,
    "lines.linewidth": 1.8, "lines.markersize": 6,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "mathtext.fontset": "cm", "savefig.bbox": "tight", "savefig.dpi": 300,
})

MR_COLOR, NH_COLOR = "#c1272d", "#1f77b4"


# ------------------------------------------------------------------ helpers
def panel_label(ax, letter, x=0.02, y=0.985):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=13,
            fontweight="bold", va="top")


def save_pair(fig, stem):
    """Write a deterministic PDF/PNG pair without wall-clock PDF metadata."""
    fig.savefig(FIG / f"{stem}.pdf",
                metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(FIG / f"{stem}.png")


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


if __name__ == "__main__":
    required = [
        PSOUT / "summary.csv",
        *(PSOUT / f"psfield_{tag}.npz" for tag in
          ("MR_lam13", "MR_lam16", "MR_lam22", "NH_lam16",
           "MR_lam16_c2_third", "MR_lam16_c2_3")),
        *(PSOUT / f"rays_{tag}_theta{theta}.csv" for tag in
          ("MR_lam15", "MR_lam18", "NH_lam15", "NH_lam18")
          for theta in (2, 45, 90, 135, 178)),
    ]
    missing = [path.relative_to(ROOT) for path in required if not path.exists()]
    if missing:
        raise SystemExit("missing figure inputs:\n  "
                         + "\n  ".join(map(str, missing)))

    fig_master()
    fig_chain()
    fig_ps_portrait()
    fig_plateau_ps()
    fig_cratio_ps()
    print("done ->", FIG)
