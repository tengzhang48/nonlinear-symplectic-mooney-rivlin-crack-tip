# Finite-element models

Two reduced incompressible plane-stress models have distinct roles:

- `pure_shear/` is the primary specimen-scale validation. Its remote state
  gives $G=hW_\infty$ in closed form, so the measured crack-tip amplitude can
  be compared with a parameter-free prediction.
- the files in this directory implement a focused disk with homogeneous remote
  isochoric stretch prescribed on its outer boundary. It is a secondary
  deep-window consistency check, not independent specimen validation.

Create the environment from the repository root with
`conda env create -f environment-fem.yml`. Run commands from the directory
containing the selected driver so that its local imports resolve.

Examples:

```bash
# From fem/
python run_one_case.py --c1 1 --c2 1 --lam 2.0 --tag MR_lam20
python check_new_signatures.py  # checks the committed disk leading stress

# From fem/pure_shear/
python run_ps.py --c1 1 --c2 1 --lam 1.6 --tag MR_lam16
python ps_export.py             # six field snapshots used by figures
python ps_report.py             # aggregate locally generated cases
```

The constitutive convention is
$W=c_1(I_1-3)+c_2(I_2-3)$ without an extra factor of $1/2$. The $c_2=0$
cases are neo-Hookean controls. They fail the tested $I_2$-specific Jacobian
contrast, but share class-universal opening and leading-stress results; they
are not expected to fail every relation. The raw face exponent is no longer
used as a material discriminator: a persistent regular $C_s s$ term gives a
raw $1/2$ profile in the Mooney–Rivlin field as well.
