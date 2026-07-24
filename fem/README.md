# Finite-element models

## Paper-facing model: pure-shear strip

`pure_shear/` is the sole specimen-scale FEM validation used by the paper. Its
far-ahead state gives

```text
G = h W_infinity
```

in closed form, so the measured crack-tip amplitude can be compared with a
parameter-free prediction.

Create the environment from the repository root with
`conda env create -f environment-fem.yml`, then run:

```bash
cd fem/pure_shear
python run_ps.py --c1 1 --c2 1 --lam 1.6 --tag MR_lam16
python ps_export.py
python ps_report.py
```

Fresh output is written beneath ignored local output directories. Curated
publication inputs live under `data/fem/strip/` and are never overwritten
implicitly.

## Matching-circle residual campaign

The matching-circle calculation uses the same pure-shear strip rather than a
second specimen. An explicit internal semicircle makes the complete P2
displacement trace available without imposing a near-tip asymptotic field.
The accepted convergence sequence refines the strip core and angular mesh,
then solves the exact inner cell restriction with the complete P2 strip trace
and reads that field on the intact axis, where the undetermined \(C_s\) and
\(C_h\) contributions vanish. It gives the Figure 7 test of the
\(r^{5/4}\) residual exponent and its parameter-free amplitude.

Run the frozen seven-case MPI campaign from the repository root:

```bash
bash fem/run_global_local_campaign.sh
```

The same code can restrict the solved strip to the cells inside the matching
circle or solve an independently refined inner mesh with the complete strip
trace prescribed. The first path supplies the Figure 7 profiles as well as
the transfer and equilibrium checks. The second is a one-way Dirichlet
submodel: its changed reaction is not fed back to the outer strip and it does
not enter Figure 7. Neither path is presented as a fully coupled algorithm or
as a selection of \(C_s\) and \(C_h\).

Implementation, convergence, and claim boundaries are recorded in
[`GLOBAL_LOCAL_WORKFLOW.md`](GLOBAL_LOCAL_WORKFLOW.md) and
[`GLOBAL_LOCAL_RESULTS.md`](GLOBAL_LOCAL_RESULTS.md). Curated campaign
records live under `data/fem/global_local/`; fresh outputs go to the ignored
`fem/global_local_outputs/` directory.

## Auxiliary cross-geometry model: focused disk

The top-level disk mesh and solver files retain a separate cross-geometry
boundary-value problem. Their full-arc boundary condition,
`F_far=diag(lambda^(-1),lambda)`, imposes crack-parallel compression and pins
the outer mouth. It is not equivalent to the Rivlin–Thomas strip. The stored
high-load upper-face boundary-vertex trace becomes nonmonotone and contains
segment crossings away from the tip, while the solver has no contact,
stability, or global-injectivity constraint.

Accordingly:

- disk outputs are not current quantitative validation or a pure-shear
  surrogate;
- `check_new_signatures.py` is an archival diagnostic, not a standard test;
- no disk data enter the paper's quantitative figures or current claim
  ledger; and
- rerunning `run_one_case.py` reproduces this auxiliary BVP rather than the
  strip calculation.

The constitutive convention in both models is
`W=c1(I1-3)+c2(I2-3)` without an extra factor of one half. The `c2=0` strip
cases are neo-Hookean controls for the `I2`-specific compensated-Jacobian
contrast; they are not expected to fail class-universal relations.
