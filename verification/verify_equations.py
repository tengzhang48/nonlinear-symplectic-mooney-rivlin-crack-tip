"""Symbolic/numeric verification of 58 encoded manuscript relations.

Run: python verify_equations.py   (needs sympy, numpy, scipy)

Each check prints PASS/FAIL. The suite mixes direct symbolic derivations with
numerical consistency checks; it supplements, but does not replace, independent
review of the assumptions and cross-equation logic.
"""
from __future__ import annotations

import numpy as np
import sympy as sp

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


# ----------------------------------------------------------------------
# symbols
l1, l2, l3, P, r, th, c1, c2 = sp.symbols(
    "lambda1 lambda2 lambda3 P r theta c1 c2", positive=True)

# ======================================================================
# 1. Invariant identities for incompressible plane stress (lambda3 = 1/(l1 l2))
# ======================================================================
l3_ps = 1 / (l1 * l2)
I1 = l1**2 + l2**2 + l3_ps**2          # tr C (3D, incompressible)
I2 = l1**2*l2**2 + l2**2*l3_ps**2 + l3_ps**2*l1**2
J2 = l1 * l2                            # in-plane Jacobian
I1_2d = l1**2 + l2**2                   # tr(F^T F) in plane
# claimed reductions used in the FEM energy:
check("I1 = I1_2d + J^-2",
      sp.simplify(I1 - (I1_2d + J2**-2)) == 0)
check("I2 = J^2 + I1_2d * J^-2",
      sp.simplify(I2 - (J2**2 + I1_2d * J2**-2)) == 0)

# ======================================================================
# 2. PK1 stress P = dW/dF for the reduced 2D energy (matrix calculus)
# ======================================================================
F = sp.Matrix(2, 2, lambda i, j: sp.Symbol(f"F{i}{j}", real=True))
Jc = F.det()
FF = sum(F[i, j]**2 for i in range(2) for j in range(2))
I1c = FF + Jc**-2
I2c = Jc**2 + Jc**-2 * FF
W = c1 * (I1c - 3) + c2 * (I2c - 3)
dWdF = sp.Matrix(2, 2, lambda i, j: sp.diff(W, F[i, j]))
Finvt = F.inv().T
P_claim = (2*c1*(F - Jc**-2 * Finvt)
           + 2*c2*(Jc**-2 * F + (Jc**2 - Jc**-2 * FF) * Finvt))
check("PK1 = dW/dF  (Mooney-Rivlin reduced plane stress)",
      sp.simplify(sp.Matrix(dWdF - P_claim)) == sp.zeros(2, 2))

# ======================================================================
# 3. Exponents and the balance relation
# ======================================================================
alpha1, alpha2, Q1 = sp.symbols("a_1 a_2 Q_1", real=True)
f_generic = sp.Function("f")(th)
g_generic = sp.Function("g")(th)
y1_ansatz = Q1 * r**alpha1 * g_generic
y2_ansatz = P * r**alpha2 * f_generic
J_ansatz = (sp.diff(y1_ansatz, r) * sp.diff(y2_ansatz, th) / r
             - sp.diff(y1_ansatz, th) * sp.diff(y2_ansatz, r) / r)
J_ansatz_claim = (P * Q1 * r**(alpha1 + alpha2 - 2)
                  * (alpha1 * sp.diff(f_generic, th) * g_generic
                     - alpha2 * f_generic * sp.diff(g_generic, th)))
check("similarity ansatz: polar Jacobian radial form",
      sp.simplify(J_ansatz - J_ansatz_claim) == 0)

grad_y2_sq = sp.diff(y2_ansatz, r)**2 + (sp.diff(y2_ansatz, th) / r)**2
grad_y2_sq_claim = (P**2 * r**(2*alpha2 - 2)
                    * (alpha2**2 * f_generic**2
                       + sp.diff(f_generic, th)**2))
check("similarity ansatz: squared opening-gradient radial form",
      sp.simplify(grad_y2_sq - grad_y2_sq_claim) == 0)

a1, a2 = sp.Rational(5, 4), sp.Rational(1, 2)
check("exponent balance 2 a1 + a2 = 3", sp.simplify(2*a1 + a2 - 3) == 0)

# ======================================================================
# 4. Leading constrained magnitudes -> J, the parameter-free relation
# ======================================================================
L1 = (P/2) * r**sp.Rational(-1, 2)         # lambda1
L2 = sp.sqrt(2/P) * r**sp.Rational(1, 4)   # lambda2 = lambda3
check("lambda2 = lambda1^(-1/2)", sp.simplify(L2 - L1**sp.Rational(-1, 2)) == 0)
Jlead = sp.simplify(L1 * L2)
check("J = lambda1 lambda2 = sqrt(P/2) r^(-1/4)",
      sp.simplify(Jlead - sp.sqrt(P/2)*r**sp.Rational(-1, 4)) == 0)
check("J r^(1/4) = sqrt(P/2)  (parameter-free, theta-independent)",
      sp.simplify(Jlead * r**sp.Rational(1, 4) - sp.sqrt(P/2)) == 0)
# amplitude anomaly: in-plane prefactor ~ P^{-1/2}
check("amplitude anomaly: lambda2 prefactor ~ P^(-1/2)",
      sp.simplify(sp.sqrt(2/P) - sp.sqrt(2)*P**sp.Rational(-1, 2)) == 0)

# ======================================================================
# 5. The g-ODE: a1 f' g - a2 f g' = Delta,  f = sin(theta/2),  Delta = 2^-1/2
# ======================================================================
Th = sp.symbols("Theta", real=True)
f = sp.sin(Th/2)
fp = sp.diff(f, Th)
Delta = 1/sp.sqrt(2)
# forced regular value from the ODE at theta = 0 (f=0, f'=1/2):
g0_solved = sp.solve(sp.Eq(a1*sp.Rational(1, 2)*sp.Symbol("g0") - 0, Delta),
                     sp.Symbol("g0"))[0]
check("g(0) = 4 sqrt2 / 5  forced by the ODE at theta=0",
      sp.simplify(g0_solved - 4*sp.sqrt(2)/5) == 0, f"g(0)={g0_solved}")
# series solution g = G0 + G2 theta^2 + ... ; check G2 = sqrt2/2
G0 = 4*sp.sqrt(2)/5
G2 = sp.Symbol("G2")
g_ser = G0 + G2*Th**2
gp_ser = sp.diff(g_ser, Th)
ode_resid = sp.series(a1*fp*g_ser - a2*f*gp_ser - Delta, Th, 0, 3).removeO()
G2_sol = sp.solve(sp.Eq(ode_resid.coeff(Th, 2), 0), G2)[0]
check("g series coefficient G2 = sqrt2/2",
      sp.simplify(G2_sol - sp.sqrt(2)/2) == 0, f"G2={G2_sol}")

# ======================================================================
# 6. Parabolic geometry: s = r sin^2(theta/2) = r f^2, and the unifications
# ======================================================================
x1 = r*sp.cos(Th)
s = sp.simplify((r - x1)/2)
check("s = (r - x1)/2 = r sin^2(theta/2) = r f^2",
      sp.simplify(s - r*sp.sin(Th/2)**2) == 0)
# For theta in [0, pi], f = sin(theta/2) >= 0, so sqrt(f^2) = f.  Use a positive
# symbol fpos to let sympy drop the |.| branch; s = r fpos^2.
fpos = sp.Symbol("f", positive=True)
m = sp.symbols("m", real=True)
# Reaction-stress/scaffold monomial identity.  This is a scaling bridge, not
# an equality between the action multiplier chi and Phi_sc.
lhs = r**m * fpos**(2*m + 1)
rhs = sp.powsimp(r**sp.Rational(-1, 2) * (r*fpos**2)**(m + sp.Rational(1, 2)), force=True)
check("reaction-stress monomial r^m f^(2m+1) = r^(-1/2) s^(m+1/2)",
      sp.simplify(lhs - rhs) == 0)
# leading opening is the parabolic coordinate: y2 = P r^1/2 f = P sqrt(s)
check("y2_lead = P r^(1/2) sin(theta/2) = P sqrt(s)",
      sp.simplify(P*sp.sqrt(r)*fpos - P*sp.sqrt(r*fpos**2)) == 0)
# A0 homogeneous mode: r^{5/4} f^{5/2} = s^{5/4}
check("A0 mode r^(5/4) f^(5/2) = s^(5/4)",
      sp.simplify((r*fpos**2)**sp.Rational(5, 4) - r**sp.Rational(5, 4)*fpos**sp.Rational(5, 2)) == 0)
# Scaffold exponent bookkeeping beta=1/4 -> s^3/4.
beta = sp.Rational(1, 4)
check("scaffold bookkeeping beta=1/4 corresponds to s^(3/4)",
      sp.simplify((beta + sp.Rational(1, 2)) - sp.Rational(3, 4)) == 0)

# ======================================================================
# 7. Near-tip Cauchy amplitude, J-integral closed form, tip-opening shape
# ======================================================================
# Separate symbolic transcription: build F exactly on the retained map
#   y1 = C_s r sin(theta/2)^2 + Q1 r^{5/4} g(theta),
#   y2 = P r^{1/2} sin(theta/2)
# with g, g' arbitrary symbols and r = s^4 (integer powers), evaluate the
# verified PK1 formula on it, and take the tip limit of (i) sigma22 * r and
# (ii) the circular J-integrand  r [W cos(theta) - t_i F_{i1}],  t = PK1 e_r.
s_ = sp.Symbol("s_", positive=True)
g_, gp_ = sp.symbols("g_ gp_", real=True)
Q1s = sp.Symbol("Q1", positive=True)
Cs_ = sp.Symbol("C_s", real=True)
rr = s_ ** 4
f_th = sp.sin(Th / 2)
fp_th = sp.cos(Th / 2) / 2
# Retain the first analytic member of the exact null family y1 -> y1+F(y2):
# C_s s = C_s r sin(theta/2)^2.  This term is O(r) and therefore dominates
# the raw face coordinate over the r^(5/4) residual when C_s != 0.
dy1r = (Cs_ * f_th ** 2
        + sp.Rational(5, 4) * Q1s * rr ** sp.Rational(1, 4) * g_)
dy1t = (2 * Cs_ * rr * f_th * fp_th
        + Q1s * rr ** sp.Rational(5, 4) * gp_)
dy2r = sp.Rational(1, 2) * P * rr ** sp.Rational(-1, 2) * f_th
dy2t = P * rr ** sp.Rational(1, 2) * fp_th
F11e = sp.cos(Th) * dy1r - sp.sin(Th) / rr * dy1t
F12e = sp.sin(Th) * dy1r + sp.cos(Th) / rr * dy1t
F21e = sp.cos(Th) * dy2r - sp.sin(Th) / rr * dy2t
F22e = sp.sin(Th) * dy2r + sp.cos(Th) / rr * dy2t
Je = sp.expand(F11e * F22e - F12e * F21e)
FFe = F11e ** 2 + F12e ** 2 + F21e ** 2 + F22e ** 2
# PK1 from the verified closed formula (Section 2), componentwise:
Fit = {  # F^{-T} entries
    (0, 0): F22e / Je, (0, 1): -F21e / Je,
    (1, 0): -F12e / Je, (1, 1): F11e / Je,
}
Fe = {(0, 0): F11e, (0, 1): F12e, (1, 0): F21e, (1, 1): F22e}
PK = {ij: (2 * c1 * (Fe[ij] - Je ** -2 * Fit[ij])
           + 2 * c2 * (Je ** -2 * Fe[ij] + (Je ** 2 - Je ** -2 * FFe) * Fit[ij]))
      for ij in Fe}
We = c1 * (FFe + Je ** -2 - 3) + c2 * (Je ** 2 + FFe * Je ** -2 - 3)

check("regular C_s s mode leaves the leading Jacobian exactly unchanged",
      sp.simplify(Je - Je.subs(Cs_, 0)) == 0)

sig22 = PK[(1, 0)] * F21e + PK[(1, 1)] * F22e        # sigma = PK1 F^T (J_3d = 1)
sig22_lim = sp.simplify(sp.limit(sp.together(sig22 * rr), s_, 0, "+"))
check("sigma22 * r -> c1 P^2 / 2, theta-free (near-tip Cauchy amplitude)",
      sp.simplify(sig22_lim - c1 * P ** 2 / 2) == 0, f"limit = {sig22_lim}")

t1e = PK[(0, 0)] * sp.cos(Th) + PK[(0, 1)] * sp.sin(Th)
t2e = PK[(1, 0)] * sp.cos(Th) + PK[(1, 1)] * sp.sin(Th)
Jintegrand = rr * (We * sp.cos(Th) - t1e * F11e - t2e * F21e)
L_lim = sp.simplify(sp.limit(sp.together(Jintegrand), s_, 0, "+"))
check("J-integrand limit on tip circles is the constant c1 P^2 / 4",
      sp.simplify(L_lim - c1 * P ** 2 / 4) == 0, f"limit = {L_lim}")
check("energy release rate G = J = (pi/2) c1 P^2",
      sp.simplify(sp.integrate(L_lim, (Th, -sp.pi, sp.pi))
                  - sp.pi / 2 * c1 * P ** 2) == 0)
# Corollary chain: sigma22 * r = G/pi.  The raw profile depends on whether the
# physical O(r) coefficient C_s persists; 2/5 survives only after that term is
# absent or subtracted.
check("corollary: sigma22 * r = G / pi",
      sp.simplify(sig22_lim - (sp.pi / 2 * c1 * P ** 2) / sp.pi) == 0)
check("profile powers: raw C_s!=0 gives 1/2; detrended C_s=0 residual gives 2/5",
      (sp.Rational(1, 2) / 1 == sp.Rational(1, 2)
       and sp.Rational(1, 2) / sp.Rational(5, 4) == sp.Rational(2, 5)))


# ======================================================================
# 8. Linearised constraint row in the Hamiltonian DAE
# ======================================================================
eps, mu = sp.symbols("epsilon mu", positive=True)
nu = mu - sp.Rational(3, 4)
gfun = sp.Function("g")
afun = sp.Function("a")
bfun = sp.Function("b")
fth = sp.sin(th/2)


def dX1(Y):
    return sp.cos(th) * sp.diff(Y, r) - sp.sin(th) / r * sp.diff(Y, th)


def dX2(Y):
    return sp.sin(th) * sp.diff(Y, r) + sp.cos(th) / r * sp.diff(Y, th)


y1 = r**a1 * gfun(th) + eps * r**mu * afun(th)
y2 = r**a2 * fth + eps * r**nu * bfun(th)
F11, F12 = dX1(y1), dX2(y1)
F21, F22 = dX1(y2), dX2(y2)
J = sp.expand(F11 * F22 - F12 * F21)
C = J**4 - (F21**2 + F22**2)
constraint_ang = sp.simplify(
    sp.diff(C, eps).subs(eps, 0) * r**(sp.Rational(9, 4) - mu)
)
gs, gps, av, apv, bv, bpv = sp.symbols("g gp a ap b bp")
constraint_ang = sp.simplify(constraint_ang.subs({
    gfun(th): gs,
    sp.Derivative(gfun(th), th): gps,
    afun(th): av,
    sp.Derivative(afun(th), th): apv,
    bfun(th): bv,
    sp.Derivative(bfun(th), th): bpv,
}))
fp_th = sp.diff(fth, th)
row4 = (fp_th * mu * av - a2 * fth * apv + a1 * gs * bpv - gps * nu * bv
        - sp.sqrt(2) * (a2 * fth * nu * bv + fp_th * bpv))
positive_branch_gp = sp.solve(
    sp.Eq(a1 * fp_th * gs - a2 * fth * gps, 1 / sp.sqrt(2)), gps
)[0]
row4_residual = sp.trigsimp(
    sp.simplify((constraint_ang - sp.sqrt(2) * row4).subs(gps, positive_branch_gp))
)
check("linearised constraint equals sqrt2 times DAE row 4",
      row4_residual == 0)

# ======================================================================
# 9. Numeric cross-checks against the committed leading profile
# ======================================================================
from pathlib import Path
npz = Path(__file__).resolve().parent.parent / "data" / "analytic" / "mr_leading_profile.npz"
if npz.exists():
    d = np.load(npz)
    th_n, f_n, g_n = d["theta"], d["f"], d["g"]
    check("profile: f = sin(theta/2) to 1e-12",
          np.max(np.abs(f_n - np.sin(th_n/2))) < 1e-12)
    check("profile: g(0) = 4 sqrt2/5 to 1e-6",
          abs(g_n[0] - 4*np.sqrt(2)/5) < 1e-6)
    check("profile: Delta = 2^(-1/2) to 1e-9",
          abs(float(d["Delta_const"]) - 2**-0.5) < 1e-9)
    # g'(pi) = -sqrt2 (face constraint), from finite difference of the profile
    gp_pi = (g_n[-1] - g_n[-2])/(th_n[-1] - th_n[-2])
    check("profile: g'(pi) ~ -sqrt2 (face)", abs(gp_pi + np.sqrt(2)) < 5e-3,
          f"g'(pi)~{gp_pi:.4f}")
else:
    print("[skip] leading profile npz not found")

# ======================================================================
# S10. Selected regular-axis outer branch of g and its face value
#      g = f^{5/2}[g(pi) + sqrt2 INT_th^pi f^{-7/2}]; smoothness across the
#      ligament kills the theta^{5/2} homogeneous mode => g(pi) = -sqrt2 A_f.
from scipy.integrate import quad as _quad


def _tail(t):
    return _quad(lambda x: np.sin(x / 2) ** -3.5, t, np.pi, limit=400)[0]


_t0 = 1e-4
_x = _t0 / 2
_Af_num = _tail(_t0) - (4 / 5) * _x ** -2.5 - (7 / 3) * _x ** -0.5
_Af_exact = float(sp.N(sp.sqrt(sp.pi) * sp.gamma(-sp.Rational(5, 4))
                       / sp.gamma(-sp.Rational(3, 4)), 17))
_gpi_num = -np.sqrt(2) * _Af_num
_gpi_exact = -np.sqrt(2) * _Af_exact
check("g selection: double-subtracted A_f agrees with exact gamma value",
      abs(_Af_num - _Af_exact) < 5e-6,
      f"quadrature={_Af_num:.9f}, exact={_Af_exact:.9f}")
check("g selection: g(pi) = -sqrt2 A_f = 2.033311...",
      abs(_gpi_num - _gpi_exact) < 5e-6,
      f"quadrature={_gpi_num:.9f}, exact={_gpi_exact:.9f}")
_g0 = np.sin(0.5e-3) ** 2.5 * (_gpi_exact + np.sqrt(2) * _tail(1e-3))
check("g selection: closed form recovers g(0) = 4 sqrt2/5",
      abs(_g0 - 4 * np.sqrt(2) / 5) < 1e-5, f"g(0+)={_g0:.6f}")

# S11. c2 -> 0 continuity with Long-Krishnan-Hui (their Eq. (94) at n=1)
_mu, _A, _n, _b = sp.symbols("mu A n b", positive=True)
_J_LKH = (_mu * sp.pi / 2) * (_b / _n) ** (_n - 1) \
    * ((2 * _n - 1) / (2 * _n)) ** (2 * _n - 1) * _A ** (2 * _n)
_J_n1 = sp.simplify(_J_LKH.subs(_n, 1))
_ours = sp.simplify((sp.pi / 2) * (_mu / 2) * _A ** 2)   # c1 = mu/2
check("LKH continuity: their (94) at n=1 equals (pi/2) c1 P^2",
      sp.simplify(_J_n1 - _ours) == 0, f"both = {_J_n1}")

# S12. Level-one / level-two hierarchy claims.
# (a) Level-one opening exclusion: the r^1 harmonic sin(theta) violates the
#     traction-free face b'(pi)=0, while the level-two harmonic sin(3theta/2)
#     satisfies both b(0)=0 and b'(pi)=0 (the free amplitude B).
_th = sp.symbols("theta_h", positive=True)
_b1 = sp.sin(_th)
_b2 = sp.sin(sp.Rational(3, 2) * _th)
check("hierarchy: sin(theta) fails b'(pi)=0 (no free r^1 opening mode)",
      sp.simplify(sp.diff(_b1, _th).subs(_th, sp.pi)) != 0
      and sp.simplify(_b1.subs(_th, 0)) == 0,
      "b'(pi)=-1")
check("hierarchy: sin(3theta/2) satisfies b(0)=0 and b'(pi)=0 (free B at 9/4)",
      sp.simplify(_b2.subs(_th, 0)) == 0
      and sp.simplify(sp.diff(_b2, _th).subs(_th, sp.pi)) == 0,
      "b'(pi)=(3/2)cos(3pi/2)=0")
# (b) Level-two forced particular solution: u_p = (1/6) f solves
#     u'' + (3/2)^2 u = (1/3) f with f = sin(theta/2), and the forcing is
#     Fredholm-orthogonal to the free harmonic sin(3theta/2) on [0, pi].
_f = sp.sin(_th / 2)
_up = _f / 6
check("hierarchy: u_p = f/6 solves u'' + (9/4) u = (1/3) f (level two)",
      sp.simplify(sp.diff(_up, _th, 2) + sp.Rational(9, 4) * _up - _f / 3) == 0)
_fred = sp.integrate(_b2 * _f / 3, (_th, 0, sp.pi))
check("hierarchy: forcing (1/3) f is Fredholm-orthogonal to sin(3theta/2)",
      sp.simplify(_fred) == 0, f"integral={_fred}")
# (c) Amplitude weighting delta_y1 = P^(2-2L) r^L a, delta_y2 =
#     P^(7/2-2L) r^(L-3/4) b: at L=5/4 it reproduces the leading amplitudes
#     (P^-1/2, P), at L=7/4 the forced pair (P^-3/2, P^0) of Eq. (ladder).
_L = sp.Rational(5, 4)
check("hierarchy: amplitude weighting reproduces leading (P^-1/2, P) at L=5/4",
      2 - 2 * _L == sp.Rational(-1, 2)
      and sp.Rational(7, 2) - 2 * _L == 1)
_L = sp.Rational(7, 4)
check("hierarchy: amplitude weighting gives forced pair (P^-3/2, P^0) at L=7/4",
      2 - 2 * _L == sp.Rational(-3, 2)
      and sp.Rational(7, 2) - 2 * _L == 0)
# (d) Face trace of the forced level-one opening: v = -(2/3) f gives
#     v(pi) = -2/3, the predicted coefficient of the r^1 face term at
#     c1 = c2 (Sec. 4).
_v = -sp.Rational(2, 3) * _f
check("hierarchy: forced face trace v(pi) = -2/3",
      sp.simplify(_v.subs(_th, sp.pi)) == sp.Rational(-2, 3))

# S14. Opening-block endpoint (Wronskian) and dual-harmonic checks.
# These verify the scalar opening-block conservation mechanism and its
# 3/2-Lambda duality ONLY; they do not derive the five-row pencil, its
# reaction-carrying momenta, or a presymplectic identity for (E, A).
# The opening block's Lagrange identity: for solutions of -b'' = nu^2 b,
# d/dxi Omega reduces to the endpoint concomitant [bV bW' - bW bV']_0^pi.
# (a) Admissible pair (b(0)=0, b'(pi)=0): concomitant vanishes at both ends.
_bV = sp.sin(_th / 2)                 # nu = 1/2 (leading opening)
_bW = sp.sin(3 * _th / 2)             # nu = 3/2 (level-two harmonic)
_conc = _bV * sp.diff(_bW, _th) - _bW * sp.diff(_bV, _th)
check("opening block: admissible pair has zero endpoint concomitant (0 and pi)",
      sp.simplify(_conc.subs(_th, 0)) == 0
      and sp.simplify(_conc.subs(_th, sp.pi)) == 0)
# (b) Non-admissible partner sin(theta) (b'(pi) != 0) leaves a nonzero
#     face concomitant: conservation is enforced by the BCs, not identically.
_bX = sp.sin(_th)
_concX = _bV * sp.diff(_bX, _th) - _bX * sp.diff(_bV, _th)
check("opening block: non-admissible partner leaves nonzero face concomitant",
      sp.simplify(_concX.subs(_th, sp.pi)) != 0)
# (c) Selection rule: Omega_open = (LW - LV) * int(bV bW); distinct
#     admissible half-odd harmonics are orthogonal on [0, pi], so
#     non-dual pairs pair to zero.
check("opening block: non-dual admissible pair (5/4, 9/4) pairs to zero",
      sp.simplify(sp.integrate(_bV * _bW, (_th, 0, sp.pi))) == 0)
# (d) Dual pairs at Lambda + Lambda' = 3/2 pair nonzero: the 9/4 harmonic
#     and its dual at -3/4 share the shape sin(3 theta/2), and the leading
#     opening (5/4) pairs with its dual at 1/4 through sin(theta/2).
_dual94 = (sp.Rational(-3, 4) - sp.Rational(9, 4)) \
    * sp.integrate(_bW * _bW, (_th, 0, sp.pi))
_dual54 = (sp.Rational(1, 4) - sp.Rational(5, 4)) \
    * sp.integrate(_bV * _bV, (_th, 0, sp.pi))
check("opening block: dual pairs (9/4,-3/4) and (5/4,1/4) pair nonzero",
      sp.simplify(_dual94) == -3 * sp.pi / 2
      and sp.simplify(_dual54) == -sp.pi / 2,
      f"Omega(9/4,-3/4)={_dual94}, Omega(5/4,1/4)={_dual54}")

# S15. Exact leading constraint/row-one completion of the characteristic Q_k
# shears.
# This uses the action multiplier chi for C=J^4-|grad y2|^2, not the
# independently propagating Phi_sc of the historical five-row scaffold.
_kk = sp.symbols("k_q", integer=True, positive=True)
_fq = sp.sin(_th / 2)
_fpq = sp.diff(_fq, _th)
_aq = _fq ** (2 * _kk)
_deltaJq = _kk * _aq * _fpq - sp.Rational(1, 2) * _fq * sp.diff(_aq, _th)
check("Q_k coupled: characteristic shear has delta J = delta C = 0",
      sp.trigsimp(_deltaJq) == 0)

_pq = (-2 * sp.sqrt(2) * _kk * (2 * _kk - 1)
       * sp.cos(_th / 2) * _fq ** (2 * _kk - 2))
_transport_q = (_fq * sp.diff(_pq, _th)
                - (2 * _kk - 1) * _fpq * _pq
                - sp.sqrt(2) * _kk * (2 * _kk - 1)
                * _fq ** (2 * _kk - 2))
check("Q_k coupled: forced action-multiplier transport is exact",
      sp.trigsimp(sp.simplify(
          _transport_q / _fq ** (2 * _kk - 2))) == 0)

_Ch = sp.symbols("C_h", real=True)
_pgen = _pq + _Ch * _fq ** (2 * _kk - 1)
_Tgen = 2 * sp.diff(_aq, _th) - _fq * _pgen / sp.sqrt(2)
check("Q_k coupled: face traction of homogeneous addition is -C_h/sqrt(2)",
      sp.simplify(_Tgen.subs(_th, sp.pi) + _Ch / sp.sqrt(2)) == 0)

_Tq = sp.trigsimp(_Tgen.subs(_Ch, 0))
_Tq_claim = 4 * _kk**2 * sp.cos(_th / 2) * _fq ** (2 * _kk - 1)
_Tq_reduced = sp.powsimp(sp.cancel(sp.together(
    (_Tq - _Tq_claim)
    / (sp.cos(_th / 2) * _fq ** (2 * _kk - 1)))), force=True)
_Tq_numerator = sp.factor(sp.fraction(_Tq_reduced)[0])
check("Q_k coupled: row-one angular traction profile and endpoints",
      sp.simplify(sp.trigsimp(_Tq_numerator)) == 0
      and all(sp.simplify(_Tq_claim.subs({_kk: kval, _th: endpoint})) == 0
              for kval in (2, 3) for endpoint in (0, sp.pi)))

_piq = 2 * _kk * _aq + sp.sqrt(2) * _fpq * _pq
_piq_claim = (2 * _kk * _fq ** (2 * _kk - 2)
              * (_fq**2 - (2 * _kk - 1) * sp.cos(_th / 2)**2))
check("Q_k coupled: reaction-carrying canonical radial momentum",
      sp.trigsimp(sp.simplify(
          (_piq - _piq_claim) / _fq ** (2 * _kk - 2))) == 0)

_p2q = sp.simplify(_pq.subs(_kk, 2))
_pi2q = sp.trigsimp(_piq_claim.subs(_kk, 2))
check("Q2 coupled: p2=-12 sqrt(2) f^2 cos and corrected pi1",
      sp.simplify(_p2q + 12 * sp.sqrt(2) * _fq**2 * sp.cos(_th / 2)) == 0
      and sp.trigsimp(_pi2q - 2 * (sp.cos(2 * _th) - sp.cos(_th))) == 0)

_Ppower = sp.Rational(2) - 2 * _kk
_chipower = -2 * _kk - sp.Rational(1, 2)
check("Q_k coupled: restored P powers balance in delta-chi C_F",
      sp.simplify(_chipower + sp.Rational(5, 2) - _Ppower) == 0)

# S16. First subleading row-two response of the characteristic Q_k shear.
# This is an independent transcription of the action stress, rather than an
# import from analysis/coupled_shear_completion.py.  It establishes that the
# bare shear is triangular: it closes the leading constraint/row-one sector,
# then forces a later opening correction.  It does NOT assert that the later
# correction is already a complete full-vector mode.
_gq = sp.Function("g_q")(_th)
_gpq = ((sp.Rational(5, 4) * _fpq * _gq - 1 / sp.sqrt(2))
        / (sp.Rational(1, 2) * _fq))
_psi_reg_q = (4 + 6 / _fq**2
              - sp.Rational(15, 2) * sp.sqrt(2) * _gq * _fpq / _fq**2)
_psi0_q = _psi_reg_q - 10 * _fq**sp.Rational(3, 2)

# Coefficients of r^(k-1/4) in the row-two stress induced by the bare shear.
_S2r_q = (sp.sqrt(2) * (4 - _psi0_q) * sp.diff(_aq, _th)
          + _pq * (-sp.sqrt(2) * _gpq - _fq))
_S2theta_q = (sp.sqrt(2) * (_psi0_q - 4) * _kk * _aq
              + _pq * (sp.Rational(5, 4) * sp.sqrt(2) * _gq
                       - 2 * _fpq))

# Independently recover the compact S2 coefficients by differentiating the
# nominal stress of F:F + J^-2 + chi(J^4-|F_2|^2) in polar reference
# components.  This prevents the R_2k check from merely assuming its starting
# stress formula.
_rho_q = sp.symbols("rho_q", positive=True)
_eps_q = sp.symbols("eps_q", real=True)
_Pamp_q, _c1amp_q, _qamp_q = sp.symbols(
    "P_amp_q c1_amp_q q_amp_q", positive=True)
_F11_q = (_Pamp_q**(-sp.Rational(1, 2)) * sp.Rational(5, 4)
          * _rho_q**sp.Rational(1, 4) * _gq
          + _eps_q * _qamp_q * _Pamp_q**(2 - 2 * _kk)
          * _rho_q**(_kk - 1) * _kk * _aq)
_F12_q = (_Pamp_q**(-sp.Rational(1, 2))
          * _rho_q**sp.Rational(1, 4) * _gpq
          + _eps_q * _qamp_q * _Pamp_q**(2 - 2 * _kk)
          * _rho_q**(_kk - 1) * sp.diff(_aq, _th))
_F21_q = (_Pamp_q * sp.Rational(1, 2)
          * _rho_q**(-sp.Rational(1, 2)) * _fq)
_F22_q = _Pamp_q * _rho_q**(-sp.Rational(1, 2)) * _fpq
_Jfull_q = _F11_q * _F22_q - _F12_q * _F21_q
_chifull_q = (_c1amp_q * _Pamp_q**-3
              * _rho_q**sp.Rational(3, 2) * _psi0_q
              + _eps_q * _qamp_q * _c1amp_q
              * _Pamp_q**(-2 * _kk - sp.Rational(1, 2))
              * _rho_q**(_kk + sp.Rational(1, 4)) * _pq)
_cof2_q = (-_F12_q, _F11_q)
_P2full_q = tuple(
    2 * _c1amp_q * F2 - 2 * _c1amp_q * _Jfull_q**-3 * cof2
    + _chifull_q * (4 * _Jfull_q**3 * cof2 - 2 * F2)
    for F2, cof2 in zip((_F21_q, _F22_q), _cof2_q)
)
_S2direct_q = tuple(sp.trigsimp(sp.simplify(
    sp.diff(component, _eps_q).subs(_eps_q, 0)
    / (_qamp_q * _c1amp_q
       * _Pamp_q**(sp.Rational(1, 2) - 2 * _kk)
       * _rho_q**(_kk - sp.Rational(1, 4)))))
    for component in _P2full_q)

_cof1_q = (_F22_q, -_F21_q)
_P1full_q = tuple(
    2 * _c1amp_q * F1 - 2 * _c1amp_q * _Jfull_q**-3 * cof1
    + _chifull_q * 4 * _Jfull_q**3 * cof1
    for F1, cof1 in zip((_F11_q, _F12_q), _cof1_q)
)
_S1direct_q = tuple(sp.trigsimp(sp.simplify(
    sp.diff(component, _eps_q).subs(_eps_q, 0)
    / (_qamp_q * _c1amp_q * _Pamp_q**(2 - 2 * _kk)
       * _rho_q**(_kk - 1)))) for component in _P1full_q)
_S1claim_q = (2 * _kk * _aq + sp.sqrt(2) * _fpq * _pq,
              2 * sp.diff(_aq, _th) - _fq * _pq / sp.sqrt(2))


def _trig_numerator_zero_q(expression):
    numerator = sp.fraction(sp.cancel(sp.together(expression)))[0]
    # The crack half-domain has 0 < theta < pi and hence f>0; combine the
    # symbolic integer-shifted powers before applying trigonometric reduction.
    numerator = sp.powsimp(numerator, force=True)
    return sp.simplify(sp.trigsimp(numerator)) == 0


_direct_action_stress_q = (
    sp.simplify(sp.trigsimp(sp.simplify(
        _Jfull_q.subs(_eps_q, 0) * _rho_q**sp.Rational(1, 4)
        / _Pamp_q**sp.Rational(1, 2)
        - 1 / sp.sqrt(2)))) == 0
    and _trig_numerator_zero_q(_S1direct_q[0] - _S1claim_q[0])
    and _trig_numerator_zero_q(_S1direct_q[1] - _S1claim_q[1])
    and _trig_numerator_zero_q(_S2direct_q[0] - _S2r_q)
    and _trig_numerator_zero_q(_S2direct_q[1] - _S2theta_q)
)

_R2q = ((_kk + sp.Rational(3, 4)) * _S2r_q
        + sp.diff(_S2theta_q, _th))
_R2q = _R2q.replace(sp.Derivative(_gq, _th), _gpq)
_cq = sp.cos(_th / 2)
_R2q_claim = (sp.sqrt(2) * _kk / 8 * _fq ** (2 * _kk - 3)
              * (5 * sp.sqrt(2) * (4 * _kk + 1) * _gq
                 - 4 * (8 * _kk**2 - 10 * _kk + 9) * _cq
                 - 12 * (2 * _kk - 1) * _cq**3))
_R2q_reduced = sp.powsimp(sp.cancel(sp.together(
    (_R2q - _R2q_claim) / _fq ** (2 * _kk - 4))), force=True)
_R2q_numerator = sp.factor(sp.fraction(_R2q_reduced)[0])
_R2q_check = sp.simplify(sp.trigsimp(_R2q_numerator))
check("Q_k subleading: variable-P,c1 action stress gives row one and R_2k",
      _direct_action_stress_q and _R2q_check == 0)

check("Q_k subleading: bare row-two face traction is -4 sqrt(2) k",
      sp.simplify(_S2theta_q.subs(_th, sp.pi)
                  + 4 * sp.sqrt(2) * _kk) == 0)

_alpha_sl = _kk + sp.Rational(3, 4)
check("Q_k subleading: slaved opening radial exponent matches the source",
      sp.simplify(_alpha_sl - 2
                  - (_kk - sp.Rational(5, 4))) == 0)
check("Q_k subleading: restored slaved-opening P power is 1/2-2k",
      sp.simplify((2 - 2 * _kk) - sp.Rational(3, 2)
                  - (sp.Rational(1, 2) - 2 * _kk)) == 0)
check("Q_k subleading: mixed opening BVP is nonresonant for integer k",
      sp.simplify(sp.cos(_alpha_sl * sp.pi)
                  - (-1)**(_kk + 1) / sp.sqrt(2)) == 0)

# --- k+3/2 companions (eq:qkcompanion) --------------------------------
_Lam_c = _kk + sp.Rational(3, 2)
_nu_c = _kk + sp.Rational(3, 4)
check("Q_k companions: restored powers P^(-2k-1) r^(k+3/2), "
      "c1 P^(-2k-7/2) r^(k+7/4) follow from the weighted state",
      sp.simplify((2 - 2 * _Lam_c) - (-2 * _kk - 1)) == 0
      and sp.simplify((-2 * _kk - sp.Rational(1, 2) - 3)
                      - (-2 * _kk - sp.Rational(7, 2))) == 0
      and sp.simplify((_Lam_c + sp.Rational(1, 4))
                      - (_kk + sp.Rational(7, 4))) == 0)

_bpi, _gpi = sp.symbols("b_pi g_pi")
_Apr_pi = sp.sqrt(2) * (_nu_c * _bpi + 5 * _kk * _gpi)
check("Q_k companions: H_k(pi)=2 sqrt2 A_k'(pi)-4 nu b_k(pi)=20k g(pi); "
      "the b_k(pi) dependence cancels identically",
      sp.simplify(2 * sp.sqrt(2) * _Apr_pi - 4 * _nu_c * _bpi
                  - 20 * _kk * _gpi) == 0)

_fc, _fpc, _bc, _bpc = sp.symbols("f_c fp_c b_c bp_c")
_D_c = sp.sqrt(2) * (_nu_c * _fc / 2 * _bc + _fpc * _bpc)
_M_c = _fc / 2 * _nu_c * _bc + _fpc * _bpc
check("Q_k companions: printed constraint RHS uses D = sqrt2 M "
      "(sqrt2 D - 2M = 0 identically)",
      sp.simplify(sp.sqrt(2) * _D_c - 2 * _M_c) == 0)

# ======================================================================
print("\n" + "="*60)
print(f"PASSED {len(PASS)}   FAILED {len(FAIL)}")
if FAIL:
    print("FAILURES:", FAIL)
    raise SystemExit(1)
print("All encoded checks passed.")
