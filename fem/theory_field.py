"""The semi-analytical leading-order Mooney-Rivlin crack-tip field.

This module evaluates the field predicted by the asymptotic theory and provides
an interpolation utility for comparisons or separately designed boundary
conditions.  The disk solver in ``mr_fem_solve.py`` instead prescribes a
homogeneous remote isochoric stretch and does not import this module.

Leading deformed map, with amplitude P:

    y2(r,theta) = P      * r^(1/2) * f(theta),   f = sin(theta/2)   (opening)
    y1(r,theta) = C_s*r*sin(theta/2)^2
                  + P^(-1/2)*r^(5/4)*g(theta)                      (in-plane)

f is exact; g solves the Delta-constraint ODE  a1 f' g - a2 f g' = 2^(-1/2)
with the forced regular value g(0) = 4 sqrt2 / 5, and is read from the committed
leading profile.  The $P^{-1/2}$ scaling applies to the chosen-branch residual;
the
leading constraint leaves $C_s$ undetermined because the first term is a
function of $y_2$ and hence does not change the Jacobian.

Leading principal-stretch magnitudes (angularly uniform at leading order):
    lambda1 = (P/2) r^(-1/2),   lambda2 = lambda3 = sqrt(2/P) r^(1/4),
    J = lambda1 lambda2 = sqrt(P/2) r^(-1/4).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

A1, A2 = 1.25, 0.5
_NPZ = (Path(__file__).resolve().parent.parent / "data" / "analytic"
        / "mr_leading_profile.npz")
_d = np.load(_NPZ)
_THETA = _d["theta"]
_G = CubicSpline(_THETA, _d["g"])
G0 = float(_d["g"][0])            # 4 sqrt2 / 5
DELTA = float(_d["Delta_const"])  # 2^(-1/2)


def f_open(theta):
    return np.sin(0.5 * np.asarray(theta))


def g_inplane(theta):
    th = np.clip(np.abs(np.asarray(theta)), 0.0, np.pi)
    return _G(th)


def deformed(r, theta, P, C_s=0.0):
    """Selected outer coordinates, retaining an optional regular C_s mode."""
    r = np.asarray(r, float)
    y2 = P * r ** A2 * f_open(theta)
    s = r * f_open(theta) ** 2
    y1 = C_s * s + P ** -0.5 * r ** A1 * g_inplane(theta)
    return y1, y2


def u_theory_factory(P, C_s=0.0):
    """Return a callable u(x)->(2,N) for dolfinx Function.interpolate."""
    def u(x):
        X, Y = x[0], x[1]
        r = np.sqrt(X ** 2 + Y ** 2)
        theta = np.arctan2(Y, X)        # upper half (Y>=0) -> [0, pi]
        theta = np.where(theta < 0.0, theta + 2 * np.pi, theta)
        y1, y2 = deformed(r, theta, P, C_s=C_s)
        return np.vstack([y1 - X, y2 - Y])
    return u


def predictions(P):
    """Parameter-free leading magnitudes at amplitude P."""
    return {"L1": P / 2.0, "L2": np.sqrt(2.0 / P), "Jr": np.sqrt(P / 2.0)}


if __name__ == "__main__":
    for P in (1.0, 2.0):
        pr = predictions(P)
        print(f"P={P}:  lambda1 r^1/2 = {pr['L1']:.4f}  lambda2 r^-1/4 = {pr['L2']:.4f}  "
              f"J r^1/4 = {pr['Jr']:.4f}")
    print(f"g(0)={G0:.5f}  g(pi/2)={float(g_inplane(np.pi/2)):.5f}  "
          f"g(pi)={float(g_inplane(np.pi)):.5f}  Delta={DELTA:.5f}")
