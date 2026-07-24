# Global--local verification of the \(r^{5/4}\) crack-tip field

## Result

The bounded MPI campaign supports the \(r^{5/4}\) in-plane residual as a
realized asymptotic class in the pure-shear Mooney--Rivlin specimen. The
evidence combines matching-radius independence, core convergence, angular
convergence, agreement with the parameter-free exact-axis amplitude, and a
full-angle check of the analytical ODE family.

The calculation does not select the analytic-axis member \(C_h=0\), and it
does not yet provide converged specimen-selected values of \(C_s\) or \(C_h\).

## What was solved

The global domain is the upper half of a Rivlin--Thomas pure-shear strip with
\(c_1=c_2=1\) and \(\lambda=1.6\). Its mesh contains an explicit internal
semicircle at \(r=R_m\). This curve is a conforming mesh interface, not a hole.

The baseline local domain is the exact strip-cell submesh inside \(R_m\). The
complete live-P2 displacement field supplies both the local initial state and
the imposed outer trace. The local nonlinear problem is then solved at the
final load. Opposing weak reactions are assembled without Dirichlet
elimination and compared on the common P2 trace. The retained profiles used
by the estimators are sampled from this exact-restriction solution, not
directly from the untouched global-strip object.

Six new global strip configurations enter the Figure 7 evidence. Four at
\(R_m=0.01h\) form the plotted core/angular sequence, and two additional
configurations test \(R_m=0.005h\) and \(0.02h\). Each is followed by its
exact-restriction consistency solve. A seventh campaign entry contains the
independently refined local submodel and does not enter Figure 7.

This is a one-way global--local submodel. It is not a fully coupled
deformation--traction algorithm because the outer strip does not respond to
an independently refined local correction.

## Verification of the discrete construction

The final production case used 16 MPI ranks, 15,360 strip triangles, and
10,320 triangles in the exact local restriction.

| check | result |
|---|---:|
| manufactured P2 transfer error | \(1.69\times10^{-15}\) |
| prescribed interface trace error | 0 |
| valid live-P2 polar samples | 28,960 / 28,960 |
| exact-restriction P2 reaction-coefficient defect | \(4.21\times10^{-9}\) |
| final restricted-field change from transferred seed | \(1.02\times10^{-7}\) |
| local full-load Newton iterations | 4 |
| local minimum sampled \(J\) | 1.871 |
| strip minimum sampled \(J\) | 0.99988 |

The manufactured transfer and affine constant-stress reaction patch tests
give the same result in serial and on 2 and 8 MPI ranks. The affine
endpoint-excluded reaction defect is \(2.51\)–\(3.44\times10^{-14}\).

The reaction value is the relative Euclidean norm of matching,
endpoint-excluded P2 weak-reaction coefficients. It is an exact-space
equilibrium diagnostic, not a mesh-invariant traction norm.

Five of the six exact-restriction configurations change the transferred seed
by at most \(4.46\times10^{-7}\) in relative norm. The 60-sector,
\(r_{\min}=2.5\times10^{-6}h\) case requires continuation from a mixed
interior seed and changes by \(2.02\times10^{-4}\). The reported sequence is
therefore an exact-cell matching-circle calculation, not a claim that the
profile bytes are an untouched export of the global field.

## Matching-radius independence

The exact-axis fit uses

\[
y_1(r,0)=c_0+A_{\rm ax} r^q+D r^{7/4}.
\]

Both \(C_s r\sin^2(\theta/2)\) and
\(C_h r^{5/4}\sin^{5/2}(\theta/2)\) vanish on this axis.

At \(r_{\min}=10^{-5}\) and \(n_\theta=60\):

| \(R_m\) | \(W_{\rm I}\) | \(W_{\rm II}\) | \(W_{\rm III}\) |
|---:|---:|---:|---:|
| 0.005 | 1.274704 | 1.262892 | 1.258613 |
| 0.010 | 1.274587 | 1.262807 | 1.258064 |
| 0.020 | 1.274565 | 1.262861 | 1.258225 |

The maximum span is \(5.50\times10^{-4}\).

Here
\(W_{\rm I}=[1.5\times10^{-4},1.6\times10^{-3}]h\),
\(W_{\rm II}=[3.0\times10^{-4},3.0\times10^{-3}]h\), and
\(W_{\rm III}=[6.0\times10^{-4},3.8\times10^{-3}]h\). These are
overlapping exact-axis fit windows, not disjoint annuli.

## Core and angular convergence

At \(R_m=0.01\):

| \(r_{\min}\) | \(n_\theta\) | \(W_{\rm I}\ q\) | \(W_{\rm II}\ q\) | \(W_{\rm III}\ q\) | \(W_{\rm III}\) amplitude error |
|---:|---:|---:|---:|---:|---:|
| \(10^{-5}\) | 60 | 1.274587 | 1.262807 | 1.258064 | 6.41% |
| \(5\times10^{-6}\) | 60 | 1.262966 | 1.256846 | 1.254860 | 3.81% |
| \(2.5\times10^{-6}\) | 60 | 1.256056 | 1.252764 | 1.251932 | 1.49% |
| \(2.5\times10^{-6}\) | 120 | 1.255612 | 1.252297 | 1.251529 | 1.24% |

The final widest-window exponent is 0.12% above \(5/4\). Doubling the angular
resolution changes the three values by no more than
\(4.67\times10^{-4}\).

### The next power need not be fixed

The table above uses the analytically expected \(r^{7/4}\) nuisance column.
As a separate check, both exponents were freed in

\[
y_1(r,0)=c_0+A_{\rm ax} r^q+D r^p.
\]

Alternating radii were used for fitting and holdout. Across five broader
radial windows, the data recovered

\[
1.24980\le q\le1.25152,\qquad
1.68971\le p\le1.74746.
\]

The holdout RMS error was below \(3.15\times10^{-5}\) of the fitted
\(r^q\) component. Thus the \(5/4\) result is not created by fixing the
nuisance exponent at \(7/4\). The deterministic check is
`fem/audit_r54_two_power.py`, and its retained output is
`data/fem/global_local/r54_axis_two_power_sensitivity.json`.

## Full-angle field and coefficient scope

After projecting the regular
\(C_s r\sin^2(\theta/2)\) background, the final widest-window exponent is

\[
q=1.25349.
\]

The residual angular amplitude agrees to 2.55% in relative \(L^2\) with

\[
g_{\rm a}(\theta)+C_h\sin^{5/2}\frac{\theta}{2}.
\]

The full-angle field is therefore consistent with the \(5/4\) angular ODE
family when the homogeneous member is admitted. Because the background
subtraction and coefficient decomposition remain window-sensitive, this is
not an independent verification of a selected angular profile. It does not
verify \(C_h=0\).

At the smallest core, separate projections give
\(C_s\approx-3.2\) to \(-3.3\) and favor a nonzero \(C_h\) near 1.2. The
estimates remain sensitive to radial decomposition and window. They are
specimen-level diagnostics, not universal constants or converged selected
coefficients.

## Independent local refinement

An independently refined local solve ended with positive Jacobian but retained
a 2.10% P2 reaction-coefficient defect relative to the unchanged outer strip.
Because this is an Euclidean coefficient norm, it should not be read as a
mesh-invariant 2.10% traction error. The complete P2 displacement is still
imposed exactly, and the refined inner field is a valid one-way Dirichlet
submodel. The coefficient difference records the reaction change that is not
fed back to the unchanged outer strip.

## Claim supported by this campaign

> The FEM fields support the \(r^{5/4}\) residual class through
> matching-radius, core, angular, and amplitude convergence. After removal of
> the regular \(C_s\) motion, the full-angle field is consistent with the
> analytical ODE family when its homogeneous
> \(C_h\sin^{5/2}(\theta/2)\) member is retained.

## Reproduce

Create and activate the environment:

```bash
conda env create -f environment-fem.yml
conda activate mr-crack-tip-fem
```

Run the complete seven-case solver matrix, six deterministic estimators, and
summary:

```bash
bash fem/run_global_local_campaign.sh
```

Run the transfer test separately in serial and MPI:

```bash
python fem/test_global_local_transfer.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
  mpiexec -n 2 python fem/test_global_local_transfer.py
```

The retained inputs and estimator records are in
`data/fem/global_local/`. The compact machine-readable result is
`data/fem/global_local/global_local_campaign_summary_2026-07-23.json`.
