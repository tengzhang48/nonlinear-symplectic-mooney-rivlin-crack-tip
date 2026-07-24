# Nonlinear symplectic Mooney–Rivlin crack tip

This is the reproducibility companion to the manuscript

> Teng Zhang, “Constrained asymptotic crack-tip fields of a Mooney–Rivlin
> sheet in plane stress: a symplectic analysis” (2026).

It contains the analytic verification code, the reduced plane-stress finite-
element implementations, the exact stored data used by the figures, and the
figure generators. The manuscript and supplementary-information (ESI/SI)
source and PDF files, referee correspondence, development history, and
third-party reference files are intentionally not part of this repository.

## Documentation map

- [`docs/README.md`](docs/README.md) — index to the scientific,
  reproducibility, and process documentation.
- [`docs/THEORY_NOTES.md`](docs/THEORY_NOTES.md) — derivation map,
  predictions, and explicit claim boundaries.
- [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) — clean-clone checks,
  figure reproduction, and fresh FEM recomputation.
- [`fem/GLOBAL_LOCAL_WORKFLOW.md`](fem/GLOBAL_LOCAL_WORKFLOW.md) — the
  matching-circle mesh, P2 transfer, local submodel, estimators, and
  code-to-data map.
- [`fem/GLOBAL_LOCAL_RESULTS.md`](fem/GLOBAL_LOCAL_RESULTS.md) — the bounded
  numerical campaign and its claim limits.

## Main result

For the incompressible plane-stress Mooney–Rivlin energy

$$W=c_1(I_1-3)+c_2(I_2-3), \qquad c_1>0,\;c_2>0,$$

the diverging $I_2$ penalty imposes a constrained, locally uniaxial crack-tip
state. Let

$$s=r\sin^2(\theta/2).$$

The leading opening and compensated Jacobian are

$$y_2=P r^{1/2}\sin(\theta/2),\qquad
Jr^{1/4}=\sqrt{P/2}.$$

The constraint also has the exact null family
$y_1\mapsto y_1+F(y_2)$. Its first analytic Mode-I member is $C_s s$, so the
local constrained problem alone does not select the horizontal crack-face
coordinate. On the superposed truncated map, the leading contour flux is
exactly independent of arbitrary $C_s$, giving

$$G=\frac{\pi}{2}c_1P^2.$$

The constraint-active in-plane correction has the form

$$y_1-c_0-C_s s
=P^{-1/2}r^{5/4}
\left[g_a(\theta)+C_h\sin^{5/2}(\theta/2)\right]+\cdots.$$

Local theory does not select either $C_s$ or $C_h$. A dedicated
matching-circle campaign avoids both coefficients on the intact axis, where
their angular factors vanish. Six new global strip configurations are each
followed by a same-cell exact-restriction consistency solve, which supplies
the retained profiles. Four cases form the plotted core/angular sequence and
two test the matching radius. The independently refined local submodel is not
used in Figure 7. Systematic core and angular refinement gives

$$q=1.251529,\qquad
\frac{A_{\rm ax}}{A_{\rm ax,pred}}=1.012420.$$

Thus the final amplitude is within $1.3\%$ of its parameter-free prediction.
A free two-power audit of the final case recovers $q$ near $5/4$ and a next
power near $7/4$ without prescribing either exponent. The latter checks the
radial ordering, not the $7/4$ correction's angular profile or amplitude.
The full-angle residual is consistent with the analytical ODE family when
its $C_h$ member is admitted. These results support the $r^{5/4}$ asymptotic
class for the tested $c_1=c_2=1$, $\lambda=1.6$ strip. They do not determine
specimen-selected values of $C_s$ or $C_h$.

For a Rivlin–Thomas pure-shear strip of reference height $h$ and grip stretch
$\lambda$,

$$G=h(c_1+c_2)(\lambda^2+\lambda^{-2}-2),$$

which fixes $P$ without a specimen-scale fit. The stored strip solutions test
this amplitude relation, the opening power, and the $I_2$-specific compensated-
Jacobian plateau. The matching-circle calculation starts from the same strip,
so the strip remains the sole FEM specimen used by the paper.

The first finite constitutive correction on the chosen $F=0$ representative
closes at pencil label $\Lambda=7/4$. A separate $\Lambda=11/4$
stationary-background calculation is a formal checkpoint. Their audited
restricted interaction reaches the $\Lambda=13/4$ opening kernel, which
corresponds to an $r^{5/2}$ opening coefficient and requires a logarithmic
companion in that restricted coefficient. The complete same-grade source,
coupled response, and net specimen amplitude remain open. Figure 2 shows this
fork-and-merge structure without presenting the open program as a completed
expansion.

A focused half-disk boundary-value problem is retained as a separate
cross-geometry calculation. Its full-arc displacement condition imposes
crack-parallel compression and is not equivalent to Rivlin–Thomas pure shear.
On the high-load stored branch, the upper-face boundary-vertex trace becomes
nonmonotone and contains segment crossings away from the tip. The current
paper therefore does not use it as quantitative validation. The code and data
remain available for a future loading, size, admissibility, and stability
study.

## Reproduce the evidence

Python 3.13 was used for the analytic and figure workflows. Create an isolated
environment and run:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python tests/run_verification.py
python tests/check_claims.py
python figures/make_figures.py
sha256sum -c MANIFEST.sha256
```

The verification runner executes ten paper-scope programs: the equation
suite, seven focused leading-field, selection, flux, energy, constraint, and
reduction programs, the closed first-material rung, and the restricted
$13/4$ log verifier.
These checks are heterogeneous and overlapping: they include direct symbolic
derivations, numerical consistency checks, and transcription reductions. They
must not be interpreted as a collection of independent proofs.

The figure command rebuilds the seven current paper figures from the tracked
analytic and strip inputs.
`figures/make_esi_mesh.py` rebuilds the mesh figure and requires the FEniCSx
environment below. Figure 7 combines the matching-circle field,
overlapping-window map, and convergence test. See
[docs/FIGURES.md](docs/FIGURES.md) for the exact input map.
The complete clean-clone workflow and current manuscript numbering are in
[docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).
The claims check also rejects embedded Type 3 fonts and enforces the exact
eight-figure PDF/PNG inventory. The generators request embedded TrueType fonts
for publisher preflight compatibility.

## Re-run the finite elements

The reduced sheet simulations use FEniCSx. The tested environment is recorded
in `environment-fem.yml`:

```bash
conda env create -f environment-fem.yml
conda activate mr-crack-tip-fem

# Pure-shear strip (one example)
(cd fem/pure_shear && python run_ps.py \
  --c1 1 --c2 1 --lam 1.6 --tag MR_lam16)

# Complete matching-circle residual campaign (8 and 16 MPI ranks)
bash fem/run_global_local_campaign.sh
```

The strip command defaults to an initial continuation subdivision of 18;
failed Newton steps trigger adaptive step halving. The `c2/c1=1/3` production
case uses `--n_steps 32`; adding `--export-field`
writes the full-field NPZ from the same solve. Stored strip artifacts include
their continuation and mesh metadata (120 sectors, with the two nearest
angular rays aligned exactly to the far rectangle corners).

Fresh solver output is written beneath `fem/pure_shear/outputs/`,
`fem/global_local_outputs/`, or the focused-disk path `fem/outputs/`.
These directories are ignored by Git and never substituted silently for the
curated figure inputs in `data/fem/`.

## Scope

The constrained opening map and null family, Jacobian plateau, energy-release
relation, pure-shear amplitude chain, closed opening-sector pairing, and
tested-strip $r^{5/4}$ residual evidence are reproduced here at their declared
evidence levels. The numerical campaign supports the residual class and its
parameter-free exact-axis amplitude to within $1.3\%$. Global matching of
$C_s$ and $C_h$ remains open.

The paper claim ledger and standard verification runner now include the closed
$\Lambda=7/4$ first-material calculation and the narrowly scoped restricted
$\Lambda=13/4$ scalar log verifier. Other higher-order scripts and
`analysis/symplectic_dae.py` remain exploratory provenance outside the release
gate. A full coupled finite-compliance spectrum, normalized extraction
integrals, mixed endpoint operator, inner axis layer, complete same-grade
$13/4$ response, and matched higher amplitudes remain open.

See [docs/CODE_MAP.md](docs/CODE_MAP.md) for the evidence level and role of
every script, and
[data/claims/principal_claims.json](data/claims/principal_claims.json) for the
machine-readable claims ledger.

## License and citation

Repository code, documentation, and data are released under the MIT License.
Citation metadata are provided in `CITATION.cff`. No article DOI is asserted
until one exists.
