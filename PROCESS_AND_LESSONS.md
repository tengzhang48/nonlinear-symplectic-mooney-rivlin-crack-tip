# Verification process and reusable lessons

The project used iterative human and AI-assisted derivation, implementation,
checking, adversarial review, and language revision. Teng Zhang is the author
of record and made the scientific, scope, and publication decisions. The
workflow described here is provenance context, not external peer review or a
quality guarantee.

Raw conversations, review reports, development commits, and superseded files
are deliberately excluded. This concise record retains only lessons that can
be checked against the public scientific artifacts.

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

## 5. A passing gate must test the claimed content

Some early checks were weak, definitional, or insensitive to a missing chain-
rule term. They were replaced with symbolic residuals, exact series
coefficients, varied-scale evaluations, or independent component routes. The
current counts describe heterogeneous, overlapping checks; the repository does
not add them up and call the total a number of independent proofs.

## 6. Scope is part of mathematical correctness

A generic constrained second variation establishes canonical structure, but it
does not by itself identify every row of a particular printed five-row pencil.
The final scope therefore separates the exact opening block and specifically
completed constrained-action responses from the historical spectral scaffold.
Open matching, inner-layer, and extraction-integral problems remain labeled as
open rather than being inferred from a plausible hierarchy.

## 7. Corrections create fresh error surfaces

Fixes to labels, dimensional factors, or boundary clauses can introduce new
inconsistencies elsewhere. Every material correction was followed by a fresh
suite run, a build or rendering check where relevant, and a synchronization
audit of captions, conclusions, summaries, and code comments.

## 8. Stored evidence and fresh computation have different roles

The tracked FEM arrays make figure reproduction fast and deterministic. They do
not replace a solver rerun. Conversely, a solver rerun does not guarantee that
the exact arrays used in a submitted figure have been preserved. This
repository supports both routes and records their environments separately.

## 9. Human judgment sets the stopping line

Automation can expose residuals and inconsistencies; it cannot decide how much
of a formal hierarchy belongs in a paper. The author retained responsibility
for constitutive assumptions, established-versus-open boundaries, claims of
novelty, and the decision to stop extending the derivation.
