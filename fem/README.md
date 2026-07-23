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

## Quarantined auxiliary model: focused disk

The top-level disk mesh and solver files are retained only as negative
provenance. Their full-arc boundary condition,
`F_far=diag(lambda^(-1),lambda)`, imposes crack-parallel compression and pins
the outer mouth. It is not equivalent to the Rivlin–Thomas strip. The stored
high-load branch also develops a same-upper-face self-intersection, while the
solver has no contact or global-injectivity constraint.

Accordingly:

- disk outputs are not physical validation or a pure-shear surrogate;
- `check_new_signatures.py` is an archival diagnostic, not a standard test;
- no disk data enter the paper figures or current claim ledger; and
- rerunning `run_one_case.py` reproduces only this quarantined auxiliary BVP.

The constitutive convention in both models is
`W=c1(I1-3)+c2(I2-3)` without an extra factor of one half. The `c2=0` strip
cases are neo-Hookean controls for the `I2`-specific compensated-Jacobian
contrast; they are not expected to fail class-universal relations.
