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

- [`THEORY_NOTES.md`](THEORY_NOTES.md) — plane-stress reduction, constrained
  tip field, radial Hamiltonian structure, exact opening block, predictions,
  and explicit open boundaries.
- [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) — clean-clone commands, expected
  checks, current paper/ESI figure numbers, stored-data reproduction, and fresh
  FEM recomputation.
- [`FIGURES.md`](FIGURES.md) — exact input file map for every rendered panel.
- [`CODE_MAP.md`](CODE_MAP.md) — evidence level and role of every public
  analysis and solver script.
- [`PUBLICATION_WORKFLOW.md`](PUBLICATION_WORKFLOW.md) — public inclusion and
  exclusion policy, release circuit, and correction protocol.
- [`PROCESS_AND_LESSONS.md`](PROCESS_AND_LESSONS.md) — concise project lessons
  retained in the scientific companion.

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

The formal constraint-active in-plane correction
$P^{-1/2}r^{5/4}g(\theta)$ and its regular-axis outer representative are
retained in the theory code. Their full finite-compliance axis/matching
selection is unresolved, and the FEM data are not used to validate that
residual power or a universal crack-face-shape exponent.

For a Rivlin–Thomas pure-shear strip of reference height $h$ and grip stretch
$\lambda$,

$$G=h(c_1+c_2)(\lambda^2+\lambda^{-2}-2),$$

which fixes $P$ without a specimen-scale fit. The stored strip solutions test
this amplitude relation, the opening power, and the $I_2$-specific compensated-
Jacobian plateau. The strip is the sole FEM evidence used by the paper.

An older focused-disk boundary-value problem is retained only as quarantined
negative provenance. Its full-arc displacement condition imposes strong
crack-parallel compression and is not equivalent to Rivlin–Thomas pure shear;
the high-load stored branch also develops a same-face self-intersection
without a contact or global-injectivity model. It is excluded from claims,
figures, and standard tests.

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

The verification runner executes the paper-scope equation suite and seven
focused leading-field, selection, flux, energy, constraint, and reduction
programs.
These checks are heterogeneous and overlapping: they include direct symbolic
derivations, numerical consistency checks, and transcription reductions. They
must not be interpreted as a collection of independent proofs.

The figure command rebuilds the five current paper figures from the tracked
strip inputs.
`figures/make_esi_mesh.py` rebuilds the mesh figure and requires the FEniCSx
environment below. See [FIGURES.md](FIGURES.md) for the exact input map.
The complete clean-clone workflow and current manuscript numbering are in
[REPRODUCIBILITY.md](REPRODUCIBILITY.md).
The claims check also rejects embedded Type 3 fonts and enforces the exact six-
figure PDF/PNG inventory; the generators request embedded TrueType fonts for publisher
preflight compatibility.

## Re-run the finite elements

The reduced sheet simulations use FEniCSx. The tested environment is recorded
in `environment-fem.yml`:

```bash
conda env create -f environment-fem.yml
conda activate mr-crack-tip-fem

# Pure-shear strip (one example)
(cd fem/pure_shear && python run_ps.py \
  --c1 1 --c2 1 --lam 1.6 --tag MR_lam16)
```

The strip command defaults to an initial continuation subdivision of 18;
failed Newton steps trigger adaptive step halving. The `c2/c1=1/3` production
case uses `--n_steps 32`; adding `--export-field`
writes the full-field NPZ from the same solve. Stored strip artifacts include
their continuation and mesh metadata (120 sectors, with the two nearest
angular rays aligned exactly to the far rectangle corners).

Fresh solver output is written beneath `fem/outputs/` or
`fem/pure_shear/outputs/` and is ignored by Git. It is never substituted
silently for the curated figure inputs in `data/fem/`.

## Scope

The constrained opening map and null family, Jacobian plateau, energy-release
relation, pure-shear amplitude chain, exact opening-block pairing, and stated
strip FEM comparisons are reproduced here at their declared evidence levels.
The formal $r^{5/4}g(\theta)$ residual remains unmatched and is not promoted to
a FEM-validated crack-shape law.

The repository retains higher-order calculation scripts and
`analysis/symplectic_dae.py` as explicitly exploratory research provenance.
They are outside the paper claim ledger, standard verification runner, and
figure dispatcher. A full coupled finite-compliance spectrum, normalized
extraction integrals, mixed endpoint operator, inner axis layer, and matched
higher amplitudes remain open.

See [CODE_MAP.md](CODE_MAP.md) for the evidence level and role of every script,
and [data/claims/principal_claims.json](data/claims/principal_claims.json) for
the machine-readable claims ledger.

## License and citation

Repository code, documentation, and data are released under the MIT License.
Citation metadata are provided in `CITATION.cff`. No article DOI is asserted
until one exists.
