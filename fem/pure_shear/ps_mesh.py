"""Rivlin-Thomas pure-shear (planar-tension) strip with an edge crack.

Upper-half model (Mode-I symmetry), crack TIP AT THE ORIGIN so the near-tip ray
extraction (theta from +x) is reused verbatim:

    reference domain  x in [-a, b],  y in [0, H]         (H = h0/2)
    crack face        y = 0,  x < 0        (traction free)
    ligament          y = 0,  x > 0        (symmetry u_y = 0)   -- tip at (0,0)
    grip              y = H                (u_y = delta, u_x = 0)
    left / right edge x = -a, x = b        (traction free)

Mesh: a TIP-FOCUSED RADIAL mesh (crack-tip rosette) that fills the strip.  For
each wedge angle theta_j in [0,pi] a ray is shot from the tip to the specimen
boundary at distance R_max(theta); graded rings r_min..R_max give a deep
near-tip window and coarsen outward.
This distributes elements radially about the tip (unlike a tensor grid, which is
anisotropic off the axes).  A tiny core r_min keeps the singular point out.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from mpi4py import MPI

import basix.ufl
from dolfinx import mesh as dmesh


@dataclass
class StripConfig:
    a: float = 3.0        # crack length   (x in [-a, 0])
    b: float = 6.0        # ligament ahead (x in [0, b])
    H: float = 0.5        # half-height (full height h0 = 2H)
    r_min: float = 1.0e-5  # core radius at the tip
    n_r: int = 64          # graded rings per ray
    n_theta: int = 120     # base angular divisions over [0, pi]
    matching_radius: float | None = None
    matching_n_inner: int | None = None

    @property
    def h0(self) -> float:
        return 2.0 * self.H

    @property
    def w(self) -> float:
        return self.a + self.b


def _R_max(theta, a, b, H):
    """Distance from the tip to the rectangle boundary along direction theta."""
    c, s = np.cos(theta), np.sin(theta)
    cand = []
    if s > 1e-12:
        cand.append(H / s)                 # top edge y = H
    if c > 1e-12:
        cand.append(b / c)                 # right edge x = b
    elif c < -1e-12:
        cand.append(a / (-c))              # left edge x = -a
    return min(cand)


def strip_theta_nodes(cfg: StripConfig) -> np.ndarray:
    """Return the deterministic angular rays used by the strip mesh."""
    theta = np.linspace(0.0, np.pi, cfg.n_theta + 1)
    # Align the nearest two interior rays with the far corners.  Merely using
    # uniform angles cuts a small triangle from each corner; inserting extra
    # rays instead creates two sliver sectors that extend all the way to the
    # tip.  Snapping preserves both the exact rectangle and n_theta sectors.
    theta_corners = np.array([
        np.arctan2(cfg.H, cfg.b),
        np.pi - np.arctan2(cfg.H, cfg.a),
    ])
    if cfg.n_theta < 3:
        raise ValueError("n_theta must be at least 3 for corner alignment")
    available = set(range(1, cfg.n_theta))
    for theta_corner in theta_corners:
        j = min(available,
                key=lambda idx: (abs(theta[idx] - theta_corner), idx))
        theta[j] = theta_corner
        available.remove(j)
    theta.sort()
    return theta


def _matching_ring(cfg: StripConfig, theta: np.ndarray) -> int | None:
    """Choose the radial row occupied by the explicit matching semicircle."""
    if cfg.matching_radius is None:
        if cfg.matching_n_inner is not None:
            raise ValueError(
                "matching_n_inner requires a matching_radius")
        return None
    Rm = float(cfg.matching_radius)
    if not (cfg.r_min < Rm):
        raise ValueError("matching_radius must exceed r_min")
    shortest_ray = min(
        _R_max(th, cfg.a, cfg.b, cfg.H) for th in theta)
    if not (Rm < shortest_ray):
        raise ValueError(
            "matching_radius must lie strictly inside every strip ray "
            f"(matching_radius={Rm:g}, shortest ray={shortest_ray:g})")
    if cfg.n_r < 4:
        raise ValueError(
            "n_r must be at least 4 when a matching interface is requested")
    if cfg.matching_n_inner is not None:
        n_inner = int(cfg.matching_n_inner)
    else:
        # Allocate radial cells approximately uniformly per logarithmic
        # decade on the shortest ray, then retain at least two cells on each
        # side of the matching interface.
        fraction = (
            np.log(Rm / cfg.r_min)
            / np.log(shortest_ray / cfg.r_min)
        )
        n_inner = int(round(cfg.n_r * fraction))
    if not (2 <= n_inner <= cfg.n_r - 2):
        raise ValueError(
            "matching_n_inner must leave at least two radial cells on each "
            f"side of the interface; got {n_inner} for n_r={cfg.n_r}")
    return n_inner


def build_strip(
        cfg: StripConfig | None = None,
        comm: MPI.Comm = MPI.COMM_WORLD):
    if cfg is None:
        cfg = StripConfig()
    theta = strip_theta_nodes(cfg)
    theta_corners = np.array([
        np.arctan2(cfg.H, cfg.b),
        np.pi - np.arctan2(cfg.H, cfg.a),
    ])
    interface_ring = _matching_ring(cfg, theta)
    nth = theta.size
    nr = cfg.n_r + 1

    X = np.empty((nr, nth))
    Y = np.empty((nr, nth))
    for j, th in enumerate(theta):
        Rmax = _R_max(th, cfg.a, cfg.b, cfg.H)
        if interface_ring is None:
            eta = np.arange(cfg.n_r + 1) / cfg.n_r
            r = cfg.r_min * (Rmax / cfg.r_min) ** eta
        else:
            Rm = float(cfg.matching_radius)
            n_inner = interface_ring
            n_outer = cfg.n_r - n_inner
            eta_inner = np.arange(n_inner + 1) / n_inner
            eta_outer = np.arange(1, n_outer + 1) / n_outer
            r = np.concatenate([
                cfg.r_min * (Rm / cfg.r_min) ** eta_inner,
                Rm * (Rmax / Rm) ** eta_outer,
            ])
        X[:, j] = r * np.cos(th)
        Y[:, j] = r * np.sin(th)

    def idx(k, j):
        return k * nth + j

    # DOLFINx expects one global input mesh, not one copy per MPI rank.
    # Supplying the arrays only on rank zero lets create_mesh partition the
    # cells collectively.  The previous all-ranks construction duplicated the
    # complete mesh under mpiexec.
    if comm.rank == 0:
        points = np.column_stack([X.ravel(), Y.ravel()]).astype(np.float64)
        cells = []
        for k in range(cfg.n_r):
            for j in range(nth - 1):
                p00, p10 = idx(k, j), idx(k + 1, j)
                p11, p01 = idx(k + 1, j + 1), idx(k, j + 1)
                cells.append([p00, p10, p11])
                cells.append([p00, p11, p01])
        cells = np.asarray(cells, dtype=np.int64)
    else:
        points = np.empty((0, 2), dtype=np.float64)
        cells = np.empty((0, 3), dtype=np.int64)

    elem = basix.ufl.element("Lagrange", "triangle", 1, shape=(2,))
    msh = dmesh.create_mesh(comm, cells, elem, points)

    info = {
        "a": cfg.a, "b": cfg.b, "H": cfg.H, "h0": cfg.h0, "w": cfg.w,
        "n_r": cfg.n_r, "n_theta_base": cfg.n_theta,
        "n_sectors": int(nth - 1), "r_min": cfg.r_min,
        "angular_scheme": "corner-snapped-v1",
        "corner_angles": theta_corners.tolist(),
        "theta_nodes": theta.copy(),
        "matching_radius": (
            None if cfg.matching_radius is None
            else float(cfg.matching_radius)),
        "matching_ring": interface_ring,
        "matching_n_inner": interface_ring,
        "matching_n_outer": (
            None if interface_ring is None
            else int(cfg.n_r - interface_ring)),
        "n_points": int(nr * nth),
        "n_cells": int(2 * cfg.n_r * (nth - 1)),
        "w_over_h0": cfg.w / cfg.h0,
    }
    return msh, info


if __name__ == "__main__":
    cfg = StripConfig()
    msh, info = build_strip(cfg)
    for k, v in info.items():
        print(f"{k:12s}: {v}")
    # near-tip radial spacing at the shortest (theta=pi/2) and longest (theta=0) rays
    for th in (np.pi / 2, 0.0):
        Rm = _R_max(th, cfg.a, cfg.b, cfg.H)
        r = cfg.r_min * (Rm / cfg.r_min) ** (np.arange(3) / cfg.n_r)
        print(f"  theta={np.degrees(th):5.1f}: R_max={Rm:.3f}  first radii {r}")
