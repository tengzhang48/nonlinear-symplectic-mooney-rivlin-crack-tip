# Matching-circle global--local workflow

## Purpose

This document connects the physical construction, the FEniCSx
implementation, the retained data, and the asymptotic estimators used to test
the \(r^{5/4}\) in-plane crack-tip field. It is the code-level companion to
[`GLOBAL_LOCAL_RESULTS.md`](GLOBAL_LOCAL_RESULTS.md), which records the
numerical results and claim boundaries.

The calculation has two distinct ingredients:

1. a pure-shear strip supplies the specimen-scale displacement on an internal
   matching semicircle; and
2. the field inside that semicircle is either extracted exactly or solved
   again on an independently refined local mesh.

The prescribed displacement trace is a complete P2 finite-element trace, not
a leading asymptotic boundary condition. The local calculation is therefore a
standard displacement-driven submodel. It is one-way because the reaction
from an independently refined local mesh is not returned to the outer strip.
That absence of feedback is a coupling limitation, not an approximation in
the imposed interface displacement.

Three related calculations appear in the records and must not be conflated:

| calculation | mesh change | role in the result |
|---|---|---|
| tip-refined matching-circle strip sequence | reduces \(r_{\min}\) and increases angular resolution in the global strip | supplies the complete P2 matching trace for each configuration |
| exact inner restriction | reuses and re-solves the strip cells inside \(R_m\) | supplies the sampled profiles used for the accepted \(q\to5/4\) and amplitude sequence, while also checking transfer and reactions |
| independently refined inner submodel | refines only the local cells at fixed \(r_{\min}=10^{-5}\) | demonstrates the displacement-driven submodel and measures the reaction change without outer feedback; it does not enter Figure 7 |

Figure 7 therefore belongs to a global-strip plus exact-restriction
calculation. It uses six new global configurations, each followed by the
same-cell inner consistency solve. Four form the plotted core/angular
sequence and two test the matching radius. The separate inner-refinement run
tests how the method could later be used more economically; it is not a
source for the reported exponent.

## End-to-end map

```text
Strip geometry with an explicit internal semicircle
    fem/pure_shear/ps_mesh.py
                  |
                  v
Global nonlinear pure-shear solution
    fem/pure_shear/ps_solve.py
                  |
                  v                               v
Exact inner-cell consistency solve
    solve_restricted_local()
                  |
                  +-------------------------------+
                  |                               |
                  v                               v
P2 interface/reaction audits             Optional refined inner mesh
181-angle x 160-radius profile            solve_refined_local()
    fem/global_local_submodel.py
                  |                       method check only
                  v
Exact-axis and full-angle estimators
    fem/pure_shear/fit_r54_campaign_stage0.py
                  |
        +---------+---------+
        |                   |
        v                   v
Free two-power audit                  Campaign summary and figure
fem/audit_r54_two_power.py            fem/summarize_global_local_campaign.py
```

The complete case matrix and command sequence are frozen in
[`run_global_local_campaign.sh`](run_global_local_campaign.sh).

## 1. Outer strip and matching circle

The reference half-strip is

\[
 -3\le X_1\le 6,\qquad 0\le X_2\le 0.5,
\]

with the crack tip at the origin. The top grip imposes the Rivlin--Thomas
pure-shear stretch, the ligament has Mode-I symmetry, and the crack face is
traction-free. The verification campaign uses \(c_1=c_2=1\) and
\(\lambda=1.6\).

`StripConfig.matching_radius` in
[`pure_shear/ps_mesh.py`](pure_shear/ps_mesh.py) inserts an exact radial row
at \(r=R_m\). On the upper-half model this row is a polygonal semicircle. It
is an internal mesh interface, not a hole and not a traction-free boundary.
`matching_n_inner` fixes the number of radial cells between the physical core
and this interface.

All angular rays meet the same interface row. Consequently:

- the strip and the restricted local domain have identical P1 interface
  chords;
- their P2 trace spaces contain the same vertices and edge-midpoint nodes;
  and
- hierarchical local refinement preserves the exterior chords and their P2
  trace nodes.

These are the same nodes required by the matching-circle construction.

## 2. Global solve

[`run_global_local.py`](run_global_local.py) calls the ordinary pure-shear
solver in [`pure_shear/ps_solve.py`](pure_shear/ps_solve.py). It solves the
exact reduced incompressible plane-stress Mooney--Rivlin energy with P2 vector
displacements and quadrature degree 6. The load is ramped to the final grip
stretch with PETSc SNES.

The near-tip field is not imposed. The internal circle only partitions the
mesh so that a trace and an inner subdomain can be recovered without geometric
search ambiguity.

## 3. Two local paths

### 3.1 Exact restriction

`extract_strip_inner_mesh()` takes the strip cells inside \(R_m\).
`solve_restricted_local()` transfers the live P2 strip field and solves the
same inner discrete problem with the strip displacement prescribed on the
outer semicircle.

Because this mesh is an exact restriction of the strip mesh, this path is a
same-cell consistency solve rather than an independently refined local model:

- the P2 transfer must be exact;
- the local Newton solve should remain close to the transferred strip state;
  and
- the opposing weak reactions must cancel to solver tolerance.

The exact-restriction solution is the MPI-safe source for the dense polar
profiles used in Figure 7. In the final case its relative change from the
transferred seed is \(1.02\times10^{-7}\). Five of the six configurations
change by at most \(4.46\times10^{-7}\); the 60-sector smallest-core case
requires continuation from a mixed interior seed and changes by
\(2.02\times10^{-4}\). It should not be described either as the untouched raw
global field or as a separately refined boundary-value solution.

### 3.2 Independently refined local submodel

`refine_local_mesh()` hierarchically refines the inner triangles while
preserving the matching-circle facets. `solve_refined_local()` transfers the
coarse inner field as an initial state, imposes the same complete P2 outer
trace, and solves the nonlinear local problem on the refined mesh.

This is a valid one-way Dirichlet submodel. If the displacement trace at
\(R_m\) has converged before the singular tip field, the local mesh can be
refined without resolving the whole strip at the same density.

The retained refinement run changed the inner cell count from 3,720 to
14,820 while keeping \(r_{\min}=10^{-5}\). It therefore tests discretization
refinement at a fixed excised core. It does not test removal of the core
regularization. The strongest \(q\to5/4\) evidence instead comes from the
separate core sequence

\[
r_{\min}=10^{-5},\quad 5\times10^{-6},\quad 2.5\times10^{-6},
\]

with a final angular refinement from 60 to 120 sectors. Those cases rebuild
the strip and its exact inner restriction at each core size.

This distinction matters:

- local refinement reduces element error at fixed geometry;
- core reduction moves the regularized problem closer to the singular limit;
  and
- both should eventually be combined while holding a demonstrably converged
  matching-circle trace.

## 4. P2 transfer and interface checks

The transfer and audit functions are in
[`global_local_submodel.py`](global_local_submodel.py).

| function | role |
|---|---|
| `transfer_function()` | MPI nonmatching interpolation of a live P2 field |
| `manufactured_transfer_error()` | checks an exactly representable quadratic vector field |
| `interface_audit()` | checks facet count, angular nodes, polygon, and normal convention |
| `_outer_trace_max_error()` | compares the imposed and realized P2 displacement trace |
| `boundary_reaction_trace()` | assembles the unmodified weak residual on the matching trace |
| `compare_interface_reactions()` | compares local and strip-outer coefficient vectors |
| `minimum_jacobian()` | evaluates \(J\) at assembly quadrature points |

The manufactured P2 error is \(1.69\times10^{-15}\) in the final case, and
the imposed live-trace error is zero to stored precision.

The reaction comparison is deliberately separate from the displacement
check. Its reported value is the relative Euclidean norm of matching P2 weak
reaction coefficients, excluding the two endpoint vector blocks. It is not a
trace-mass dual norm and is therefore not a mesh-invariant physical traction
error.

For the exact restriction the reaction-coefficient defect is
\(4.21\times10^{-9}\). For the once-refined independent submodel it is
2.10%. The latter does not mean that the prescribed displacement is wrong by
2.10%. It measures the reaction change produced by refining the inner problem
without feeding that change back into the outer strip.

For a one-way displacement submodel, the relevant accuracy checks are:

1. convergence of the matching-circle displacement with outer discretization;
2. insensitivity of the inferred inner quantity to \(R_m\);
3. convergence with inner mesh and core size; and
4. exact enforcement of the transferred trace.

The current campaign directly establishes items 2 and 4 and supplies a joint
strip/core/angular sequence for item 3. A future efficiency study can hold
one converged outer trace fixed while reducing the inner core independently.

## 5. Sampling and asymptotic estimators

`sample_polar_displacement()` evaluates the live P2 solution at 160
logarithmically spaced radii and 181 angles, including \(\theta=0\) and
\(\theta=\pi\). MPI ownership is checked for every point. The final profile
contains 28,960 valid samples.

### Exact-axis channel

On the intact axis, \(\theta=0\),

\[
C_s r\sin^2(\theta/2)=0,\qquad
C_h r^{5/4}\sin^{5/2}(\theta/2)=0.
\]

The estimator can therefore fit

\[
Y_1(r,0)=c_0+A_{\rm ax} r^q+D r^{7/4}
\]

without either matching coefficient. Alternating radii are used for fitting
and holdout. Three overlapping radial windows test sensitivity:
\(W_{\rm I}=[1.5\times10^{-4},1.6\times10^{-3}]h\),
\(W_{\rm II}=[3.0\times10^{-4},3.0\times10^{-3}]h\), and
\(W_{\rm III}=[6.0\times10^{-4},3.8\times10^{-3}]h\). They are
post-processing intervals on the same exact-axis field, not separate
subdomains. The predicted amplitude is

\[
A_{\rm ax,pred}=\frac{4\sqrt2}{5\sqrt P},
\]

where \(P\) is measured independently from the crack-face opening.

The final widest-window values are

\[
q=1.251529,\qquad
A_{\rm ax}/A_{\rm ax,pred}=1.012420.
\]

The amplitude is therefore within 1.3% of the parameter-free prediction.

### Free two-power audit

[`audit_r54_two_power.py`](audit_r54_two_power.py) replaces the fixed
\(r^{7/4}\) nuisance column by

\[
Y_1(r,0)=c_0+A_{\rm ax} r^q+D r^{p_{\rm next}},
\]

and fits both exponents. For the final case 4, five radial windows return

\[
1.24980\le q\le1.25152,\qquad
1.68971\le p_{\rm next}\le1.74746.
\]

Thus the recovered \(5/4\) exponent is not created by fixing the next power.
The fitted \(p_{\rm next}\) is consistent with the \(r^{7/4}\) in-plane slot
in the \(F=0\) bookkeeping ordering. This does not test that correction's
angular profile or amplitude.

### Full-angle channel

The full-angle estimator first determines a regular
\(b(\theta)r\) background in an outer window. It freezes and subtracts that
background, rediscovers the residual exponent, and compares its angular
amplitude with

\[
g_{\rm a}(\theta)+C_h\sin^{5/2}(\theta/2).
\]

The finest calculation gives \(q=1.25349\) and a 2.55% relative \(L^2\)
profile mismatch after a finite-window \(C_h\) is fitted. This supports the
angular ODE family, but it does not select \(C_h=0\). The extracted \(C_s\)
and \(C_h\) values remain window-sensitive specimen-level diagnostics.

## 6. Code-to-data map

| file | input | output or responsibility |
|---|---|---|
| `pure_shear/ps_mesh.py` | strip and interface geometry | distributed strip mesh with explicit \(R_m\) row |
| `pure_shear/ps_solve.py` | mesh, material, grip stretch | global P2 displacement field |
| `global_local_submodel.py` | global field and local mesh | restriction, refinement, transfer, local solve, reactions, samples |
| `run_global_local.py` | one case configuration | solver metadata JSON and optional polar-profile NPZ |
| `pure_shear/fit_r54_campaign_stage0.py` | metadata JSON and profile NPZ | exact-axis and full-angle estimator JSON |
| `audit_r54_two_power.py` | finest profile and metadata | free-\(q,p_{\rm next}\) audit JSON |
| `summarize_global_local_campaign.py` | accepted case records | compact campaign summary with SHA-256 provenance |
| `run_global_local_campaign.sh` | frozen seven-case matrix | complete regenerated scratch campaign |

Regenerated solver files go to the ignored
`fem/global_local_outputs/` directory. Accepted public records live in
`data/fem/global_local/`. The compact result used by documentation and
figures is
`data/fem/global_local/global_local_campaign_summary_2026-07-23.json`.

The independently refined profile is reproducible but is not part of the
accepted \(q\)-convergence sequence. Its metadata is retained to document the
mesh, exact trace, positive final Jacobian, and reaction diagnostic.

## 7. Reproduction

Create the pinned environment:

```bash
conda env create -f environment-fem.yml
conda activate mr-crack-tip-fem
```

Use the `python` and `mpiexec` from that environment and keep threaded
libraries at one thread per MPI rank.

Run the transfer test in serial and MPI:

```bash
python fem/test_global_local_transfer.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
  mpiexec -n 2 python fem/test_global_local_transfer.py
```

Run the complete solver, estimator, and summary matrix:

```bash
bash fem/run_global_local_campaign.sh
```

The campaign script includes:

- three matching radii \(R_m=0.005,0.01,0.02\);
- three core sizes;
- the 60-to-120-sector angular check;
- one independent inner refinement at fixed core;
- the three-window exact-axis/full-angle estimator; and
- the free two-power audit.

To draw the residual-convergence figure directly from a fresh summary, run

```bash
python figures/make_fig_r54_axis_convergence.py \
  --summary fem/global_local_outputs/global_local_campaign_summary.json \
  --profile fem/global_local_outputs/global_local_profile_lam16_core2p5e6_nt120_rm1e2.npz
```

## 8. What the calculation supports

Supported for the tested \(c_1=c_2=1,\lambda=1.6\) strip:

- realization of the exact-axis \(r^{5/4}\) asymptotic class;
- the parameter-free exact-axis amplitude;
- insensitivity of the fitted exponent to the tested matching radius;
- convergence with core and angular resolution; and
- consistency of the full-angle residual with the analytical ODE family when
  its homogeneous member is admitted.

Not established:

- a universal numerical value of \(C_s\) or \(C_h\);
- selection of the analytic-axis member \(C_h=0\);
- two-way displacement--traction coupling;
- uniqueness or stability of the global nonlinear solution; or
- independent core reduction on a fixed converged outer trace.

The last item is the natural next numerical efficiency test. It is not needed
for the current \(5/4\) asymptotic-class claim because the existing
matching-radius, core, angular, amplitude, holdout, and free-two-power checks
already address that claim directly.
