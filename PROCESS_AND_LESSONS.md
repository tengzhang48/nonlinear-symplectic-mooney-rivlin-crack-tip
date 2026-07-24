# Verification process and reusable lessons

The project used iterative human and AI-assisted derivation, implementation,
checking, adversarial review, and language revision. Teng Zhang is the author
of record and made the scientific, scope, and publication decisions. The
workflow described here is provenance context, not external peer review or a
quality guarantee.

Raw conversations, review reports, development commits, and superseded files
are deliberately excluded. This concise record retains only lessons that can
be checked against the public scientific artifacts.

This account was reconstructed from the repository history, signed review
records in the research workspace, executable checks, stored numerical data,
and manuscript diffs. It does not use conversation memory as evidence. Git
author fields identify the human account that recorded the work, not which
model performed a particular audit. Agent attribution in the internal record
therefore comes only from signed reports, named branch handoffs, and commit
subjects.

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

## 10. Verify the boundary-value problem before interpreting a clean solve

An auxiliary focused disk was initially treated as a second validation
geometry. Its full-arc displacement condition was mathematically well defined
but physically different from the intended Rivlin–Thomas strip: it imposed
strong crack-parallel compression and prescribed the outer mouth. At high
load, the computed upper face intersected itself away from the tip.

This exposed four distinct checks that a small residual norm cannot replace:

- the convenient numerical BVP must be equivalent to the intended physical
  BVP in the loading directions that matter;
- a stationary discrete solution need not be globally injective or stable;
- a same-face self-intersection is not opposing-face crack contact; and
- a locally smooth inner window cannot rescue a globally folded,
  post-onset branch when no contact or injectivity model is present.

The disk solver and data are preserved as quarantined negative provenance, but
they are excluded from paper claims, figures, and standard tests. The strip is
the sole FEM evidence.

## 11. Human judgment sets the stopping line

Automation can expose residuals and inconsistencies; it cannot decide how much
of a formal hierarchy belongs in a paper. The author retained responsibility
for constitutive assumptions, established-versus-open boundaries, claims of
novelty, and the decision to stop extending the derivation.

## 12. A challenge should produce a test, not only a narrower claim

During revision, uncertainty sometimes triggered a defensive response:
remove the disputed figure, narrow the statement, and describe the calculation
as outside the paper. This is useful as temporary publication control, but it
is not a scientific resolution. It can turn an unresolved mechanism into an
apparently absent result.

The focused disk calculation provides the concrete example. Its stored
high-load boundary-vertex trace is nonmonotone and has segment crossings in
post-processing. That observation does not by itself prove self-contact of
the continuous P2 boundary, and the original face fit did not separate the
allowed \(O(r)\) motion from the \(r^{5/4}\) residual. These findings require
the disk claim to be scoped carefully. They also open specific questions about
remote loading, dimensionless disk geometry, stationary-branch selection,
stability, contact, and convergence of the inner field. The appropriate
response is to vary those factors and retain the calculation as a named
hypothesis, not to make the troublesome observation disappear.

The project now uses four dispositions for a challenged result: false,
unsupported, conditional, or open but testable. Only the first justifies
retraction as the scientific conclusion. An unsupported result is demoted,
a conditional result keeps its assumptions, and an open result receives a
discriminating test. Editorial narrowing can accompany any of these states
while the work is incomplete, but it does not replace the mechanism audit.

For a scale-free membrane calculation, “disk size” means dimensionless ratios
such as outer radius to crack length and core radius to outer radius, not a
uniform change of all lengths. Once bending is included, thickness supplies
another length. The released tip-centered half-disk has only
$R_{\mathrm{disk}}/a=1$, because its slit runs from the tip at the center to
the circular rim. It therefore contains no disk-size sweep. A genuine test
must hold crack length fixed and vary the outer radius, for example with an
eccentric edge-cracked disk, while also comparing the existing
$\operatorname{diag}(1/\lambda,\lambda)$ loading with a
$\operatorname{diag}(1,\lambda)$ control. This distinction turns a vague
objection into a reproducible size, loading, and contact study.

## 13. The collaboration was a sequence of claims, audits, and dispositions

The work did not proceed as one uninterrupted derivation. The human author
set the physical problem, decided which claims mattered, and set the stopping
line. Multiple AI agents then worked in bounded roles: derivation,
implementation, numerical testing, manuscript integration, or adversarial
review. A result was promoted only after its equations, boundary conditions,
amplitude scaling, and manuscript wording were checked together.

The recurring cycle was:

1. state a bounded claim and its assumptions;
2. attach an executable identity, convergence test, or clearly stated
   analytical argument;
3. give the claim to a different review pass;
4. classify each challenge as false, unsupported, conditional, or open but
   testable;
5. correct the mathematics or scope, then rerun the affected tests;
6. synchronize the manuscript, ESI, figures, code, and stored data; and
7. stop only after a clean release audit.

This process found both ordinary implementation defects and conceptual scope
errors. Examples include the missing power of \(P\), the initially suppressed
\(C_s s\) null motion, an incomplete \(13/4\) source census, and a disk
boundary-value problem that did not represent the intended strip validation.
The corrections did not all narrow the paper. The \(C_s\) challenge, for
example, led to a sharper numerical observable rather than abandonment of the
\(r^{5/4}\) prediction.

The internal chronology records named-agent contributions and exact commits.
It is kept with the research project because those reports contain abandoned
branches and review discussion. The public case study extracts the reusable
method without presenting AI review as external peer review.

## 14. Remove nuisance directions before estimating a singular power

The local constraint allows specimen-selected \(C_s\) and \(C_h\) terms. A
full-angle fit must account for both, and the available window does not select
\(C_h=0\). On the intact axis, however, both nuisance contributions vanish
identically. That observation determined the final matching-circle test.

The same global pure-shear strip supplies a converged displacement trace on an
interior circle. A tip-refined submodel receives that P2 trace without
interpolation through a separate geometric parametrization. Across matching
radius, core size, and angular-resolution checks, the exact-axis estimator
converges to

\[
q=1.251529,\qquad
A_{\mathrm{ax}}/A_{\mathrm{ax,pred}}=1.012420 .
\]

A free two-power fit recovers a leading power near \(5/4\) and a next power
near \(7/4\) without fixing either exponent. The test therefore supports the
predicted asymptotic class and its parameter-free exact-axis amplitude. It
does not determine the specimen-level values of \(C_s\) or \(C_h\), and the
one-way submodel is not presented as a two-way global--local coupling.

## 15. Final release checks are part of the scientific record

The release gate checks more than whether the PDF compiles. It verifies the
symbolic identities and cross-file claim ledger, checks the stored FEM
provenance and matching-circle campaign summary, rebuilds the retained
figures, and confirms that the manuscript, ESI, code, and data state the same
scope. The version tag identifies the exact public state used by the paper.
The DOI is a later archival identifier and is not needed to make that state
reproducible now.
