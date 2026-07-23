# Verification process and reusable lessons

The project used iterative human and AI-assisted derivation, implementation,
checking, adversarial review, and language revision. Teng Zhang is the author
of record and made the scientific, scope, and publication decisions. The
workflow described here is provenance context, not external peer review or a
quality guarantee.

Raw conversations, review reports, development commits, and superseded files
are deliberately excluded. This concise record retains only lessons that can
be checked against the public scientific artifacts.

The longer, research-method case study is maintained separately in
[`ai-mechanics-resources`](https://github.com/tengzhang48/ai-mechanics-resources/blob/main/case_studies/symplectic_mooney_rivlin_crack_tip.md).
The present repository keeps only the process information needed to understand
and reproduce its scientific release; signed agent reports and internal
handoffs are not public release artifacts.

## 1. Cross-equation consistency is a separate test

An equation can be algebraically correct in isolation while its amplitude,
power, or label is inconsistent with a neighboring table or selection rule.
The final workflow therefore checked not only local identities but also the
maps between field expansions, endpoint conditions, spectra, amplitude gauges,
and manuscript statements.

## 2. A unit gauge can hide a dimensional mistake

Calculations performed at $P=1$ cannot expose a missing power of $P$. The
project restored the amplitude symbol before promotion and also evaluated the
full stress at varied numerical scales. This caught a missing $P^{-3/2}$ factor
that dimensionless shape checks alone could not see.

## 3. Component decomposition resolves mechanism disputes

The $O(r^{1/2})$ excess obtained by inserting the truncated leading map into a
finite-radius $J$-integral was initially assigned to the wrong contribution.
Separating energy and traction terms showed that it is carried by the opening-
row $c_2$ traction, with coefficient $\pi c_2P r^{1/2}$. On the exact
constraint manifold the row-one flux is instead $O(c_2r^2)$, and the angularly
constant $c_2$ energy term integrates to zero against $\cos\theta$.

## 4. Degenerate endpoints require branch analysis

The slave-profile ODE admits a homogeneous $f^{5/2}$ contribution invisible to
the forced endpoint values alone. A finite-part calculation and smoothness
selection fix $g(\pi)=2.033311\ldots$. Backward integration from that selected
endpoint is numerically stable; outward integration can drift onto the wrong
branch. The same lesson motivated exact endpoint series for the reaction
profile: the regular particular solution has $\psi_{\mathrm{reg}}(\pi)=10$,
while the traction-free total profile is
$\psi_0=\psi_{\mathrm{reg}}-10f^{3/2}$ and hence $\psi_0(\pi)=0$.

## 5. Constraint nullspaces must survive until matching

The leading relation $J^4=|\nabla y_2|^2$ is invariant under
$y_1\mapsto y_1+F(y_2)$. Because $y_2=P\sqrt{s}$, its first analytic Mode-I
member is the physical displacement $C_s s$. Labeling that direction a gauge
prematurely removed an $O(r)$ term from the face coordinate. Unless matching
forces $C_s=0$, it dominates the $r^{5/4}$ residual and changes the raw face
power from $2/5$ to $1/2$.

Several otherwise correct gates were blind to this error: the null term leaves
the constraint, Jacobian plateau, and leading energy flux unchanged; an
ahead-of-tip ray suppresses it because $s=0$ on the exact symmetry axis; and a
vertical tangent does not distinguish raw exponents $1/2$ and $2/5$. The
corrected workflow therefore enumerates the kernel before fixing amplitudes,
fits one physical $c_0$ across angles, projects the fitted $O(r)$ coefficient
onto $\sin^2(\theta/2)$, and keeps raw and detrended profile claims separate.

## 6. A passing gate must test the claimed content

Some early checks were weak, definitional, or insensitive to a missing chain-
rule term. They were replaced with symbolic residuals, exact series
coefficients, varied-scale evaluations, or independent component routes. The
current counts describe heterogeneous, overlapping checks; the repository does
not add them up and call the total a number of independent proofs.

## 7. Scope is part of mathematical correctness

A generic constrained second variation establishes canonical structure, but it
does not by itself identify every row of a particular printed five-row pencil.
The final scope therefore separates the exact opening block and specifically
completed constrained-action responses from the historical spectral scaffold.
Open matching, inner-layer, and extraction-integral problems remain labeled as
open rather than being inferred from a plausible hierarchy.

## 8. Corrections create fresh error surfaces

Fixes to labels, dimensional factors, or boundary clauses can introduce new
inconsistencies elsewhere. Every material correction was followed by a fresh
suite run, a build or rendering check where relevant, and a synchronization
audit of captions, conclusions, summaries, and code comments.

## 9. Stored evidence and fresh computation have different roles

The tracked FEM arrays make figure reproduction fast and deterministic. They do
not replace a solver rerun. Conversely, a solver rerun does not guarantee that
the exact arrays used in a submitted figure have been preserved. This
distinction became concrete in the final audit: scalar/ray records and
full-field snapshots had been produced on two nearly identical strip meshes,
so reproducing one headline case did not establish a common provenance for all
figures. The publication data were regenerated on one exact-corner mesh, the
continuation and mesh settings were embedded in every JSON and NPZ artifact,
and the claim gate now inventories both artifact classes. This repository
supports both stored-data and fresh-solve routes and records their environments
separately.

The geometry itself required a three-way check. Uniform angular rays clipped
the two far corners, while merely inserting corner rays created narrow sectors
that ran to the tip. The production construction instead retains 120 sectors
and aligns the nearest two rays with the corners; the automated gate checks the
outer area, angular gaps, and stored full-field coordinates.

## 10. Human judgment sets the stopping line

Automation can expose residuals and inconsistencies; it cannot decide how much
of a formal hierarchy belongs in a paper. The author retained responsibility
for constitutive assumptions, established-versus-open boundaries, claims of
novelty, and the decision to stop extending the derivation.
