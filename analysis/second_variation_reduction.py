#!/usr/bin/env python3
"""Interior leading-order reduction of the MR constrained hierarchy (gated).

SCOPE: interior (0.4 < theta < 2.6 samples), leading radial order, P=c1=1
gauge; several gates are sampled evidence or reports, not exact proofs.
Endpoint behavior, the scaled action normalization, and the coupled spectrum
are NOT established here.

Multiplier action  A[y,Phi] = int (W1(F) + Phi C(F)) dA,
W1 = F:F + J^-2 - 3 (c1 = 1),  C = J^4 - (F21^2 + F22^2),  P = 1 gauge.
Leading orbit  y1 = r^(5/4) g,  y2 = r^(1/2) f,  f = sin(theta/2); the
g-ODE (5/4) f' g - (1/2) f g' = Delta = 2^(-1/2) gives J0 = Delta r^(-1/4)
and C(F0) = 0 exactly.

Representation: all r-dependence is monomial ({exponent: theta-coeff}
graded dicts); theta-profiles are plain symbols with a chain-rule
derivative map (G for g with the g-ODE encoded; A0.., B0.., P0.., Q0..
for the increments a, b, p and the base reaction psi0). Zero tests are
numeric sampling (several random draws, tol 1e-9); the printed final
objects are exact expressions.

Gates: 0 formula transcription; A orbit/constraint exactness; B base
reaction transport (closed form) + base opening consistency; C1 row-4
match; C2 opening harmonic; C3 in-plane structure; D momenta content.
Exit nonzero on FAIL.
"""
from __future__ import annotations

import random
import sys

import sympy as sp

th = sp.Symbol("theta", positive=True)
L = sp.Symbol("Lambda", real=True)
a1, a2 = sp.Rational(5, 4), sp.Rational(1, 2)
nu = L - sp.Rational(3, 4)
Delta = 1 / sp.sqrt(2)
f = sp.sin(th / 2)
fp = sp.diff(f, th)

# profile symbols and the derivative map
G = sp.Symbol("G")            # g(theta)
A0, A1s, A2s, A3s = sp.symbols("A0 A1 A2 A3")
B0, B1s, B2s, B3s = sp.symbols("B0 B1 B2 B3")
P0, P1s, P2s = sp.symbols("P0 P1 P2")
Q0, Q1s, Q2s = sp.symbols("Q0 Q1 Q2")
DMAP = {G: (a1 * fp * G - Delta) / (a2 * f),
        A0: A1s, A1s: A2s, A2s: A3s,
        B0: B1s, B1s: B2s, B2s: B3s,
        P0: P1s, P1s: P2s,
        Q0: Q1s, Q1s: Q2s}


def dth(expr):
    out = sp.diff(expr, th)
    for s, ds in DMAP.items():
        out += sp.diff(expr, s) * ds
    return out


SAMPLES = []
random.seed(11)
for _ in range(4):
    SAMPLES.append({th: random.uniform(0.4, 2.6),
                    L: random.uniform(0.9, 3.1),
                    **{s: random.uniform(-2, 2) for s in
                       (G, A0, A1s, A2s, A3s, B0, B1s, B2s, B3s,
                        P0, P1s, P2s, Q0, Q1s, Q2s)}})


def is_zero(expr, tol=1e-9):
    if expr == 0:
        return True
    for smp in SAMPLES:
        v = complex(sp.N(expr.subs(smp), 30))
        if abs(v) > tol:
            return False
    return True


GATES = []


def gate(name, cond, detail=""):
    GATES.append((name, bool(cond)))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" +
          (f"  ({detail})" if detail else ""), flush=True)


# ----------------------------------------------------- graded algebra
class T(dict):
    def __add__(self, o):
        out = T(self)
        for k, v in o.items():
            kk = _key(k, out)
            out[kk] = out.get(kk, 0) + v
        return out

    def __mul__(self, o):
        out = T()
        if isinstance(o, T):
            for k1, v1 in self.items():
                for k2, v2 in o.items():
                    kk = _key(k1 + k2, out)
                    out[kk] = out.get(kk, 0) + sp.expand(v1 * v2)
            return out
        return T({k: sp.expand(v * o) for k, v in self.items()})

    __rmul__ = __mul__

    def __neg__(self):
        return T({k: -v for k, v in self.items()})

    def __sub__(self, o):
        return self + (-o)

    def dth_(self):
        return T({k: dth(v) for k, v in self.items()})

    def dr_(self):
        return T({k - 1: sp.expand(k * v) for k, v in self.items()})

    def shift(self, dk):
        return T({k + dk: v for k, v in self.items()})


def _key(k, d):
    ks = sp.nsimplify(k)
    for kk in d:
        if kk == ks or sp.simplify(kk - ks) == 0:
            return kk
    return ks


def clean(t: T) -> T:
    return T({k: sp.expand(v) for k, v in t.items() if not is_zero(v)})


def lead(t: T):
    t = clean(t)
    if not t:
        return None, None
    ks = sorted(t.keys(), key=lambda z: float(z.subs(L, 1.75)))
    return ks[0], t[ks[0]]


# ---------------------------------------------- fields (graded, symbolized)
F0 = {(0, 0): T({sp.Rational(1, 4): a1 * G}),
      (0, 1): T({sp.Rational(1, 4): DMAP[G] * 0 + sp.Symbol("Gp")}),
      (1, 0): T({-a2: a2 * f}),
      (1, 1): T({-a2: fp})}
# use Gp symbol for g' with its own derivative rule (from the g-ODE)
Gp = sp.Symbol("Gp")
DMAP[Gp] = dth(DMAP[G]).subs(sp.Symbol("Gp"), DMAP[G])  # placeholder fix below
# recompute properly: g' = DMAP[G]; so represent F0[(0,1)] with DMAP[G]
F0[(0, 1)] = T({sp.Rational(1, 4): DMAP[G]})
del DMAP[Gp]
for smp in SAMPLES:
    smp.pop(Gp, None)

dF = {(0, 0): T({L - 1: L * A0}),
      (0, 1): T({L - 1: A1s}),
      (1, 0): T({nu - 1: nu * B0}),
      (1, 1): T({nu - 1: B1s})}
PHI0 = T({sp.Rational(3, 2): Q0})
S = L + sp.Rational(1, 4)
dPHI = T({S: P0})


def Jstar(F):
    return {(0, 0): F[(1, 1)], (0, 1): -F[(1, 0)],
            (1, 0): -F[(0, 1)], (1, 1): F[(0, 0)]}


J0 = T({sp.Rational(-1, 4): Delta})
J0m3 = T({sp.Rational(3, 4): Delta**-3})
J0m4 = T({1: Delta**-4})
J02 = T({sp.Rational(-1, 2): Delta**2})
J03 = T({sp.Rational(-3, 4): Delta**3})

Js0 = Jstar(F0)
dJs = Jstar(dF)
dJ = T()
for k in F0:
    dJ = dJ + Js0[k] * dF[k]

# ------------------------------------------- gate 0: formula verification
Fsym = {k: sp.Symbol(f"F{k[0]}{k[1]}") for k in F0}
dFsym = {k: sp.Symbol(f"dF{k[0]}{k[1]}") for k in F0}
Jsym = Fsym[(0, 0)] * Fsym[(1, 1)] - Fsym[(0, 1)] * Fsym[(1, 0)]
eps = sp.Symbol("epsilon")
pert = {Fsym[k]: Fsym[k] + eps * dFsym[k] for k in Fsym}
def num_zero(expr, syms, n=5, tol=1e-9):
    for _ in range(n):
        smp = {s: random.uniform(0.5, 2.0) for s in syms}
        if abs(complex(sp.N(expr.subs(smp), 25))) > tol:
            return False
    return True


fsyms = list(Fsym.values()) + list(dFsym.values())
ok0 = True
for k in F0:
    dW = 2 * Fsym[k] - 2 * Jsym**-3 * sp.diff(Jsym, Fsym[k])
    dC = 4 * Jsym**3 * sp.diff(Jsym, Fsym[k]) \
        - (2 * Fsym[k] if k[0] == 1 else 0)
    dJsym = sum(sp.diff(Jsym, Fsym[q]) * dFsym[q] for q in Fsym)
    dJs_sym = sp.diff(Jsym, Fsym[k]).subs(
        {Fsym[q]: dFsym[q] for q in Fsym})
    claim_dW1 = (2 * dFsym[k] + 6 * Jsym**-4 * dJsym
                 * sp.diff(Jsym, Fsym[k]) - 2 * Jsym**-3 * dJs_sym)
    direct = sp.diff(dW.subs(pert), eps).subs(eps, 0)
    ok0 &= num_zero(direct - claim_dW1, fsyms)
    claim_dC = (12 * Jsym**2 * dJsym * sp.diff(Jsym, Fsym[k])
                + 4 * Jsym**3 * dJs_sym
                - (2 * dFsym[k] if k[0] == 1 else 0))
    directC = sp.diff(dC.subs(pert), eps).subs(eps, 0)
    ok0 &= num_zero(directC - claim_dC, fsyms)
gate("0: delta-stress formulas match direct differentiation (sampled)", ok0)

print("[stage] gate 0 done", flush=True)
# ---------------------------------------------------------------- gate A
C0_check = J0 * J0 * J0 * J0 - (F0[(1, 0)] * F0[(1, 0)]
                                + F0[(1, 1)] * F0[(1, 1)])
kA, vA = lead(C0_check)
gate("A: C(F0) = 0 exactly", vA is None, f"lead={kA}")
J0_direct = F0[(0, 0)] * F0[(1, 1)] - F0[(0, 1)] * F0[(1, 0)]
kJ, vJ = lead(J0_direct - J0)
gate("A: J0 = Delta r^(-1/4) exactly", vJ is None, f"lead={kJ}")

print("[stage] gate A done", flush=True)
# --------------------------------------------------- base stress, gate B
Pi0 = {}
for k in F0:
    t = 2 * F0[k] - 2 * (J0m3 * Js0[k])
    t = t + PHI0 * (4 * (J03 * Js0[k]) - (2 * F0[k] if k[0] == 1 else T()))
    Pi0[k] = clean(t)


def divergence(Pir: T, Pith: T) -> T:
    return (Pir.shift(1).dr_() + Pith.dth_()).shift(-1)


R0 = [divergence(Pi0[(i, 0)], Pi0[(i, 1)]) for i in range(2)]
k1b, v1b = lead(R0[0])
gate("B: base in-plane residual leading power = -3/4",
     k1b is not None and sp.simplify(k1b + sp.Rational(3, 4)) == 0,
     f"power={k1b}")
# linear first-order ODE in psi0: cA psi0' + cB psi0 + cG = 0
cA = sp.trigsimp(sp.expand(sp.diff(v1b, Q1s)))
cB = sp.trigsimp(sp.expand(sp.diff(v1b, Q0)))
cG = sp.expand(v1b - cA * Q1s - cB * Q0)
gate("B: ODE is linear in (psi0, psi0')",
     is_zero(sp.expand(v1b - (cA * Q1s + cB * Q0 + cG))))
print("\nbase-reaction ODE coefficients:")
print("  cA =", cA)
print("  cB =", cB)
print("  cG =", sp.trigsimp(cG) if len(str(cG)) < 400 else cG)
# homogeneous solution: psi_h = exp(-int cB/cA). The calculation gives
# cB/cA = -(3/2) f'/f, hence psi_h = f^(3/2) (consistent with the +1
# label shift of the reaction row; an earlier prediction of f^(-1/2)
# in this comment was wrong and is corrected here).
ratio_BA = sp.trigsimp(sp.cancel(sp.together(cB / cA)))
print("cB/cA =", ratio_BA)
IF = sp.trigsimp(sp.integrate(ratio_BA, th))
print("integral of cB/cA =", IF)
print("=> homogeneous base reaction psi_h = exp(-int) =",
      sp.trigsimp(sp.exp(-IF)))
psi_h_B = f**sp.Rational(3, 2)
expected_ratio_B = -sp.Rational(3, 2) * fp / f
homogeneous_B = sp.trigsimp(cA * sp.diff(psi_h_B, th) + cB * psi_h_B)
gate("B: base transport ratio and psi_h=f^(3/2) verified",
     is_zero(sp.together(ratio_BA - expected_ratio_B))
     and homogeneous_B == 0)

k2b, v2b = lead(R0[1])
gate("B: base opening residual has NO r^(-3/2) term",
     k2b is None or sp.simplify(k2b + sp.Rational(3, 2)) != 0,
     f"leading power={k2b}")

print("[stage] gate B done", flush=True)
# ------------------------------------------------------ linearized rows
dPi = {}
for k in F0:
    t = 2 * dF[k] + 6 * (J0m4 * dJ * Js0[k]) - 2 * (J0m3 * dJs[k])
    t = t + PHI0 * (12 * (J02 * dJ * Js0[k]) + 4 * (J03 * dJs[k])
                    - (2 * dF[k] if k[0] == 1 else T()))
    t = t + dPHI * (4 * (J03 * Js0[k]) - (2 * F0[k] if k[0] == 1 else T()))
    dPi[k] = clean(t)

dC_lin = 4 * (J03 * dJ) - 2 * (F0[(1, 0)] * dF[(1, 0)]
                               + F0[(1, 1)] * dF[(1, 1)])
kc, vc = lead(dC_lin)
tau1 = L * A0
tau2 = nu * B0
row4_printed = (fp * tau1 - a2 * f * A1s + a1 * G * B1s
                - DMAP[G] * tau2 - sp.sqrt(2) * (a2 * f * tau2 + fp * B1s))
r_samples = [complex(sp.N((vc / row4_printed).subs(smp), 30))
             for smp in SAMPLES]
const_ratio = all(abs(r_samples[0] - x) < 1e-9 for x in r_samples[1:])
gate("C1: linearized constraint = printed row 4 (constant factor)",
     const_ratio and is_zero(sp.expand(vc - r_samples[0].real
                                       * row4_printed)),
     f"factor~{r_samples[0].real:.6f}")

R1lin = [divergence(dPi[(i, 0)], dPi[(i, 1)]) for i in range(2)]
k2l, v2l = lead(R1lin[1])
harm = B2s + nu**2 * B0
c2l = [complex(sp.N((v2l / harm).subs(smp), 30)) for smp in SAMPLES]
c2const = all(abs(c2l[0] - x) < 1e-9 for x in c2l[1:])
gate("C2: linearized opening = harmonic row at leading order",
     k2l is not None and c2const
     and is_zero(sp.expand(v2l - c2l[0].real * harm)),
     f"power={k2l}, factor~{c2l[0].real:.6f}")

k1l, v1l = lead(R1lin[0])
present = {n: any(v1l.has(s) for s in ss) for n, ss in
           [("p", (P0, P1s)), ("psi0", (Q0, Q1s)),
            ("a", (A0, A1s, A2s)), ("b", (B0, B1s, B2s))]}
print("\nlinearized in-plane equation: leading power =", k1l)
print("term content:", present)
gate("C3: in-plane equation carries the reaction increment at leading order",
     present["p"])

# ---- Gate C4: reduce the in-plane equation with row 4 + harmonic + base ODE
A1_sol = sp.solve(sp.Eq(vc, 0), A1s)[0]
A2_sol = sp.expand(dth(A1_sol).subs(A1s, A1_sol))
red = sp.expand(v1l.subs(A2s, A2_sol).subs(A1s, A1_sol))
red = sp.expand(red.subs(B3s, dth(-nu**2 * B0)).subs(B2s, -nu**2 * B0))
Q1_sol = sp.solve(sp.Eq(cA * Q1s + cB * Q0 + cG, 0), Q1s)[0]
red = sp.expand(red.subs(Q2s, dth(Q1_sol).subs(Q1s, Q1_sol))
                .subs(Q1s, Q1_sol))
coefs = {n: sp.trigsimp(sp.cancel(sp.together(sp.expand(
    sp.diff(red, sym))))) for n, sym in
    [("P1", P1s), ("P0", P0), ("A0", A0), ("B0", B0), ("B1", B1s)]}
rest = sp.expand(red - sum(sp.diff(red, sym) * sym for sym in
                           (P1s, P0, A0, B0, B1s)))
print("\nreduced in-plane equation, coefficient structure:")
for nname, c in coefs.items():
    print(f"  coeff[{nname}] =", c if len(str(c)) < 300 else "(long)")
print("  remainder (should be 0):", "0" if is_zero(rest) else rest)
# is the p-part proportional to the printed transport row
# f p' + 3 f' p = 2 Lp f' p at the increment label Lp = Lambda + 1 ?
cP1, cP0 = coefs["P1"], coefs["P0"]
disp_free = all(is_zero(coefs[nm]) for nm in ("A0", "B0", "B1"))
gate("C4a: reduced in-plane equation is linear forced p-transport "
     "(sampled zero test)", is_zero(rest) and not is_zero(cP1),
     "pure-p" if disp_free else "displacement couplings present: canonical "
     "slaving of the reaction increment")
print("\nexact displacement-coupling coefficients (forcing of the "
      "p-transport):")
for nm in ("A0", "B0", "B1"):
    print(f"  F[{nm}] =", sp.trigsimp(sp.cancel(sp.together(coefs[nm]))))
ratioP = sp.trigsimp(sp.cancel(sp.together(cP0 / cP1)))
print("  cP0/cP1 =", ratioP)
target = sp.trigsimp(sp.cancel(((3 - 2 * (L + 1)) * fp) / f))
print("  printed-row-5 ratio at label Lambda+1:", target)
gate("C4b: p-transport ratio matches printed row 5 at label Lambda+1",
     is_zero(sp.expand(sp.together(ratioP - target))))

print("[stage] gate C done", flush=True)
# ------------------------------------------------------------- momenta
dp1 = clean(dPi[(0, 0)].shift(1))
dp2 = clean(dPi[(1, 0)].shift(1))
kp1, vp1 = lead(dp1)
kp2, vp2 = lead(dp2)
print("\ncanonical momentum increments (leading):")
print(f"pi1 @ r^{kp1}:")
sp.pprint(sp.collect(sp.expand(vp1), (A0, A1s, B0, B1s, P0, Q0)))
print(f"pi2 @ r^{kp2}:")
sp.pprint(sp.collect(sp.expand(vp2), (B0, B1s, P0, Q0)))
gate("D: pi2 leading is reaction-free and equals 2 nu b",
     kp2 is not None and sp.simplify(kp2 - nu) == 0
     and not (vp2.has(P0) or vp2.has(Q0))
     and is_zero(sp.expand(vp2 - 2 * nu * B0)),
     "")
gate("D: pi1 leading carries the reaction increment",
     kp1 is not None and sp.simplify(kp1 - L) == 0 and vp1.has(P0),
     f"power={kp1}")

print("\n" + "=" * 60)
bad = [n for n, ok in GATES if not ok]
print(f"GATES: {sum(ok for _, ok in GATES)}/{len(GATES)} passed")
if bad:
    print("FAILED:", bad)
    sys.exit(1)
