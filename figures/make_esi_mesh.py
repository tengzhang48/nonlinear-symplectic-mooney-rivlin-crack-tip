#!/usr/bin/env python3
"""ESI mesh figure: annotated strip half-model (schematic + mesh) on top;
strip and disk tip zooms below, in units of 10^-3 h.

Builds both meshes fresh with the repo's own generators (no solve) and
writes figures/rendered/fig_esi_mesh.{pdf,png}.
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FEM = str(ROOT / "fem")
FIG = str(Path(__file__).resolve().parent / "rendered")
sys.path.insert(0, FEM + "/pure_shear")
sys.path.insert(0, FEM)

from ps_mesh import StripConfig, build_strip          # noqa: E402
from mr_fem_mesh import MeshConfig, build_mesh        # noqa: E402

plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42, "ps.fonttype": 42,
                     "savefig.dpi": 300, "savefig.bbox": "tight"})


def tri_arrays(msh):
    x = msh.geometry.x[:, :2]
    msh.topology.create_connectivity(2, 0)
    cells = msh.topology.connectivity(2, 0).array.reshape(-1, 3)
    return x, cells


strip, _info = build_strip(StripConfig())
xs, cs = tri_arrays(strip)
disk, _dinfo = build_mesh(MeshConfig())
xd, cd = tri_arrays(disk)

fig = plt.figure(figsize=(12.4, 5.9))
gs = GridSpec(2, 3, height_ratios=[0.8, 2.0], hspace=0.10, wspace=0.30)

# (a) full strip half-model with boundary-condition annotations
ax = fig.add_subplot(gs[0, :])
ax.triplot(xs[:, 0], xs[:, 1], cs, lw=0.08, color="0.45", rasterized=True)
ax.plot([-3, 0], [0, 0], color="crimson", lw=2.0)
ax.plot(0, 0, "o", color="crimson", ms=5, zorder=5)
ax.set_aspect("equal")
ax.set_xlim(-3.35, 6.35)
ax.set_ylim(-0.02, 0.54)
ax.set_xlabel(r"$x/h$", labelpad=1)
ax.set_ylabel(r"$y/h$")
ax.set_title(r"(a) strip half-model: rosette mesh and boundary conditions",
             fontsize=10)
ax.annotate(r"grip: $u_x=0,\ u_y=\delta$", xy=(-1.0, 0.52),
            xytext=(-2.9, 0.74), ha="left", fontsize=9,
            arrowprops=dict(arrowstyle="-|>", lw=0.9))
ax.annotate("crack face: traction-free", xy=(-1.8, 0.0),
            xytext=(-2.9, 0.26), fontsize=9, color="crimson",
            arrowprops=dict(arrowstyle="-|>", lw=0.9, color="crimson"))
ax.annotate(r"ligament: $u_y=0$ (Mode-I symmetry)", xy=(3.2, 0.0),
            xytext=(3.4, 0.26), fontsize=9,
            arrowprops=dict(arrowstyle="-|>", lw=0.9))
ax.annotate("tip", xy=(0.0, 0.0), xytext=(0.35, 0.14), fontsize=9,
            color="crimson",
            arrowprops=dict(arrowstyle="-|>", lw=0.9, color="crimson"))

# (b) strip intermediate zoom: the rosette ring structure
ax = fig.add_subplot(gs[1, 0])
ax.triplot(xs[:, 0], xs[:, 1], cs, lw=0.3, color="0.3")
ax.plot([-0.15, 0], [0, 0], color="crimson", lw=1.8)
ax.set_xlim(-0.15, 0.15)
ax.set_ylim(0, 0.15)
ax.set_aspect("equal")
ax.set_xticks([-0.15, 0, 0.15])
ax.set_yticks([0, 0.15])
ax.set_xlabel(r"$x/h$", labelpad=1)
ax.set_ylabel(r"$y/h$")
ax.set_title("(b) strip, graded rings", fontsize=10)

# (c) strip tip zoom in units of 1e-3 h
S = 1.0e-3
ax = fig.add_subplot(gs[1, 1])
ax.triplot(xs[:, 0] / S, xs[:, 1] / S, cs, lw=0.3, color="0.3")
ax.plot([-4, 0], [0, 0], color="crimson", lw=1.8)
ax.set_xlim(-4, 4)
ax.set_ylim(0, 4)
ax.set_aspect("equal")
ax.set_xticks([-4, -2, 0, 2, 4])
ax.set_yticks([0, 2, 4])
ax.set_xlabel(r"$x/h\ \times10^{-3}$", labelpad=1)
ax.set_ylabel(r"$y/h\ \times10^{-3}$")
ax.set_title("(c) strip, tip zoom", fontsize=10)

# (d) disk tip zoom in units of 1e-3 h
ax = fig.add_subplot(gs[1, 2])
ax.triplot(xd[:, 0] / S, xd[:, 1] / S, cd, lw=0.3, color="0.3")
ax.plot([-0.8, 0], [0, 0], color="crimson", lw=1.8)
ax.set_xlim(-0.8, 0.8)
ax.set_ylim(0, 0.8)
ax.set_aspect("equal")
ax.set_xticks([-0.8, -0.4, 0, 0.4, 0.8])
ax.set_yticks([0, 0.4, 0.8])
ax.set_xlabel(r"$x/h\ \times10^{-3}$", labelpad=1)
ax.set_ylabel(r"$y/h\ \times10^{-3}$")
ax.set_title("(d) disk, tip zoom", fontsize=10)

fig.savefig(f"{FIG}/fig_esi_mesh.pdf",
            metadata={"CreationDate": None, "ModDate": None})
fig.savefig(f"{FIG}/fig_esi_mesh.png")
print("wrote fig_esi_mesh; strip cells:", len(cs), "disk cells:", len(cd))
