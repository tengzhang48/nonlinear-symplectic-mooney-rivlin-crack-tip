# Nonlinear symplectic Mooney–Rivlin crack tip

This is the reproducibility companion to the manuscript

> Teng Zhang, “Crack-tip fields and the energy release rate of a
> Mooney–Rivlin sheet in plane stress: a symplectic analysis” (2026).

It contains the analytic verification code, the reduced plane-stress finite-
element implementations, the exact stored data used by the figures, and the
figure generators. The manuscript source, referee correspondence, development
history, and third-party reference files are intentionally not part of this
repository.

## Main result

For the incompressible plane-stress Mooney–Rivlin energy

$$W=c_1(I_1-3)+c_2(I_2-3), \qquad c_1>0,\;c_2>0,$$

the diverging $I_2$ penalty imposes a constrained, locally uniaxial crack-tip
state. With reference polar coordinates $(r,\theta)$, the leading deformed map
is

$$y_2=P r^{1/2}\sin(\theta/2), \qquad
y_1=c_0+P^{-1/2}r^{5/4}g(\theta)+\cdots,$$

where $g$ satisfies

$$\frac54 f'g-\frac12fg'=2^{-1/2},\qquad f=\sin(\theta/2),$$

with the smooth-branch selection $g(0)=4\sqrt2/5$ and
$g(\pi)=2.033311\ldots$. Consequences include

$$Jr^{1/4}=\sqrt{P/2},\qquad
y_2\propto |y_1-c_0|^{2/5},\qquad
G=\frac{\pi}{2}c_1P^2.$$

For a Rivlin–Thomas pure-shear strip of reference height $h$ and grip stretch
$\lambda$,

$$G=h(c_1+c_2)(\lambda^2+\lambda^{-2}-2),$$

which fixes $P$ without a specimen-scale fit. The stored strip solutions test
this amplitude relation and the $I_2$-specific Jacobian plateau. The stored
disk solutions are a secondary, deeper-window cross-check of radial powers,
tip shape, and the class-universal stress relation $\sigma_{22}r=G/\pi$.

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

The verification runner executes the 57-check manuscript suite and the five
consolidated analysis suites with headline counts 14, 22, 7, 31, and 28, plus
the reduction and historical-scaffold suites and the focused leading-field,
selection, flux, energy, and constraint checks.
These checks are heterogeneous and overlapping: they include direct symbolic
derivations, numerical consistency checks, and transcription reductions. They
must not be interpreted as 159 independent proofs.

The figure command rebuilds all nine main figures from the tracked analytic and
FEM inputs. `figures/make_esi_mesh.py` rebuilds the mesh figure and requires the
FEniCSx environment below. See [FIGURES.md](FIGURES.md) for the exact input map.

## Re-run the finite elements

The reduced sheet simulations use FEniCSx. The tested environment is recorded
in `environment-fem.yml`:

```bash
conda env create -f environment-fem.yml
conda activate mr-crack-tip-fem

# Disk cross-check (one example)
cd fem
python run_one_case.py --c1 1 --c2 1 --lam 2.0 --tag MR_lam20

# Primary pure-shear strip (one example)
cd pure_shear
python run_ps.py --c1 1 --c2 1 --lam 1.6 --tag MR_lam16
```

Fresh solver output is written beneath `fem/outputs/` or
`fem/pure_shear/outputs/` and is ignored by Git. It is never substituted
silently for the curated figure inputs in `data/fem/`.

## Scope

The leading constrained orbit, its smooth branch, the energy-release relation,
and the reported FEM comparisons are the established results reproduced here.
The repository also contains a formally derived higher-order hierarchy, but
its scope is deliberately narrower than a complete spectrum of the coupled
finite-compliance problem. In particular, the full five-block conserved
pairing, normalized extraction integrals for candidate higher parameters, the
inner axis layer, and the generated $k+3$ rung remain open.

`analysis/symplectic_dae.py` is therefore labeled and used only as a historical
five-row spectral scaffold. The consolidated completion scripts establish the
specific blocks and companion relations stated in their own headers; they do
not turn that scaffold into a completed coupled Hamiltonian pencil.

See [CODE_MAP.md](CODE_MAP.md) for the evidence level and role of every script,
and [data/claims/principal_claims.json](data/claims/principal_claims.json) for
the machine-readable claims ledger.

## License and citation

Repository code, documentation, and data are released under the MIT License.
Citation metadata are provided in `CITATION.cff`. No article DOI is asserted
until one exists.
