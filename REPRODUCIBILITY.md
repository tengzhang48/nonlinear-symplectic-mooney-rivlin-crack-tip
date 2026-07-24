# Reproducing the analysis and paper figures

This guide separates three tasks:

1. verify the paper-scope analytic identities and stored strip claims;
2. regenerate the submitted figures from the curated strip and
   matching-circle records; and
3. independently rerun the finite-element strip boundary-value problems.

The first two tasks do not require FEniCSx. A fresh solve is substantially more
expensive and is not needed merely to redraw the figures.

## 1. Fast path from a clean clone

The analytic and figure workflow was tested with Python 3.13.

```bash
git clone https://github.com/tengzhang48/nonlinear-symplectic-mooney-rivlin-crack-tip.git
cd nonlinear-symplectic-mooney-rivlin-crack-tip
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt

python tests/run_verification.py
python tests/check_claims.py
python figures/make_figures.py
sha256sum -c MANIFEST.sha256
```

`tests/check_claims.py` also recomputes the strip-only ESI finite-window table
from the stored rays and compares it with
`data/derived/profile_mode_audit.json`. That table records fitted regular
coefficients, raw face slopes, and target-free nested residual powers. It does
not establish a universal or asymptotically resolved residual exponent.

The analytic checks are heterogeneous and partly overlapping; their count is
not a count of independent proofs.

## 2. Current paper and ESI figure map

| Placement | Public output | Regeneration route |
|---|---|---|
| Figure 1 | `fig_master.{pdf,png}` | analytic schematic; `figures/make_figures.py` |
| Figure 2 | `fig_asymap.{pdf,png}` | conceptual physics map; `figures/make_fig_asymap.py` |
| Figure 3 | `fig_chain.{pdf,png}` | stored strip summary; `figures/make_figures.py` |
| Figure 4 | `fig_ps_portrait.{pdf,png}` | stored strip full fields; `figures/make_figures.py` |
| Figure 5 | `fig_plateau.{pdf,png}` | stored strip rays; `figures/make_figures.py` |
| Figure 6 | `fig_cratio.{pdf,png}` | stored strip full fields; `figures/make_figures.py` |
| Figure 7 | `fig_r54_axis_convergence.{pdf,png}` | hashed matching-circle campaign summary; `figures/make_fig_r54_axis_convergence.py` |
| ESI Figure S1 | `fig_esi_mesh.{pdf,png}` | fresh strip mesh construction; `figures/make_esi_mesh.py` |

The exact file-level inputs are listed in [`FIGURES.md`](FIGURES.md). The
standard renderer emits exactly the seven paper pairs. The committed ESI pair
is generated separately in the pinned FEniCSx environment.

## 3. Regenerating derived analytic artifacts

The formal leading angular profile is deterministic:

```bash
python data/analytic/build_profile.py
```

The ESI strip table can be regenerated and checked independently:

```bash
python analysis/profile_mode_audit.py --write
python analysis/profile_mode_audit.py --check-stored
```

This command writes only `data/derived/profile_mode_audit.json`; it does not
create a manuscript figure.

Ordinary users should not run `make_manifest.py` after an exploratory change,
because doing so would bless changed bytes locally. Maintainers update the
manifest only after reviewing and accepting a complete regeneration.

## 4. Stored-data reproduction versus fresh simulation

`data/fem/strip/` contains the scalar, ray, and full-field inputs used by the
specimen figures. `data/fem/global_local/` contains the live-P2 profiles,
solver metadata, estimator records, and compact hashed summary used by
Figure 7. The stored-data lane answers:

> Do the archived inputs regenerate the reported summaries and figures under
> the current public analysis code?

A fresh finite-element solve answers a different question:

> Does the pinned strip solver and mesh independently reproduce the archived
> boundary-value result within the declared numerical tolerances?

Neither lane silently substitutes for the other.

`data/fem/disk/` is different: it preserves a quarantined auxiliary
boundary-value problem that is not equivalent to pure shear and is not used by
the paper, claims tests, or figure generators. See its directory README.

## 5. Fresh FEniCSx strip recomputation

```bash
conda env create -f environment-fem.yml
conda activate mr-crack-tip-fem

(cd fem/pure_shear && python run_ps.py \
  --c1 1 --c2 1 --lam 1.6 --tag MR_lam16)

# Seven-case matching-circle residual campaign
bash fem/run_global_local_campaign.sh
```

Fresh single-strip output is written beneath ignored
`fem/pure_shear/outputs/`; the matching-circle campaign writes beneath
ignored `fem/global_local_outputs/`. The figure code never consumes either
location implicitly. Before replacing any curated input, verify the mesh,
continuation, case metadata, numerical tolerances, and cross-file provenance
described in [`data/README.md`](data/README.md) and
[`fem/README.md`](fem/README.md).

The production strip mesh uses 120 angular sectors and aligns the two nearest
interior rays exactly with the far corners. The `c2/c1=1/3` production case
uses 32 initial continuation subdivisions. These settings are stored in the
curated artifacts and checked by `tests/check_claims.py`.

The matching-circle matrix uses 8 and 16 MPI ranks and writes scratch records
under ignored `fem/global_local_outputs/`. It includes three matching radii,
three core sizes, a 60-to-120-sector angular check, one independently refined
one-way local submodel, the exact-axis/full-angle estimators, and a free
two-power audit.

To redraw Figure 7 from a fresh campaign summary:

```bash
python figures/make_fig_r54_axis_convergence.py \
  --summary fem/global_local_outputs/global_local_campaign_summary.json
```

## 6. Rebuilding the ESI mesh figure

With the FEniCSx environment active:

```bash
python figures/make_esi_mesh.py
```

This constructs the strip mesh afresh and does not read a stored mesh image.

## 7. Reproduced scope

The standard public lane reproduces:

- the constrained leading-field identities and closed opening-sector pairing;
- the Jacobian plateau, energy-release relation, and pure-shear amplitude tie;
- the tested-strip $r^{5/4}$ residual class and exact-axis amplitude;
- the closed $\Lambda=7/4$ first-material rung and restricted
  $\Lambda=13/4$ scalar log channel at their stated scope;
- all seven current paper figures and the strip-only ESI mesh figure; and
- hashes and provenance for the public inputs and outputs.

It does not establish:

- global matching that selects `Cs`, `Ch`, or higher amplitudes;
- two-way coupling of an independently refined local mesh to the outer strip;
- the complete same-grade $\Lambda=13/4$ source and coupled response;
- a completed coupled finite-compliance spectrum;
- normalized higher-order extraction integrals; or
- the quarantined disk solution as physical evidence.

Other higher-order and historical-scaffold scripts remain available as
non-paper research provenance and are excluded from the standard runner.

## 8. Maintainer release check

```bash
python tests/run_verification.py
python tests/check_claims.py
python figures/make_figures.py
# In the pinned FEniCSx environment when the mesh figure changes:
python figures/make_esi_mesh.py
python make_manifest.py
(cd data/fem/strip && sha256sum -c ARTIFACTS.sha256)
sha256sum -c MANIFEST.sha256
git diff --check
```

Run the manifest generator last and review every changed rendering before
committing. The release procedure is documented in
[`PUBLICATION_WORKFLOW.md`](PUBLICATION_WORKFLOW.md).
