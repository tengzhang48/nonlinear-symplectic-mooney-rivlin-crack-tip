# Focused-disk auxiliary cases

These four JSON files preserve an auxiliary cross-geometry boundary-value
problem. They are not used in the paper's quantitative comparisons and are
not read by the standard claims checks or figure generators.

The disk model prescribes

```text
F_far = diag(lambda^(-1), lambda)
```

as displacement data around the full outer arc. This is not the
Rivlin–Thomas pure-shear strip: it imposes substantial crack-parallel
compression and also prescribes the outer crack-mouth displacement. At
`c1=c2=1` and `lambda=2`, for example, the homogeneous disk state has
`P_xx=-15`, whereas the strip far-ahead state has no imposed horizontal
contraction.

On the stored high-load branch, the upper-face boundary-vertex trace becomes
nonmonotone and contains segment crossings away from the tip. This is a
discrete folding diagnostic, not a certified self-contact result for the
continuous P2 boundary. The solver contains no contact, global-injectivity,
bending, or stability constraint, so the inner and global diagnostics must be
assessed separately.

The files are retained so the auxiliary setup and its outputs remain
auditable. They should not be cited as quantitative cross-geometry validation,
a crack-contact solution, or a source for a manuscript figure in the current
release.

These limitations apply to the stored boundary-value problem, not to circular
domains in general. A future disk cross-check could be useful if its remote
displacement or traction data are redesigned for the intended physical
loading and its continuation explicitly checks size ratios, stability,
orientation, global injectivity, and contact. Such a calculation would be new
evidence rather than a reinterpretation of these files.
