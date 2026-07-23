# Quarantined focused-disk cases

These four JSON files preserve negative provenance for an auxiliary
boundary-value problem. They are not paper evidence and are not read by the
standard claims checks or figure generators.

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

The stored high-load disk branch is also not globally admissible. Its upper
face self-intersects away from the tip, and the solver contains no contact,
global-injectivity, bending, or stability constraint. A smooth local inner
window cannot repair that global failure.

The files are retained so the failed auxiliary setup and its outputs remain
auditable. They must not be cited as a cross-check, validation geometry,
crack-contact solution, or source for a manuscript figure.

This quarantine applies to the stored boundary-value problem, not to circular
domains in general. A future disk cross-check could be useful if its remote
displacement or traction data are redesigned for the intended physical
loading and its continuation explicitly checks stability, orientation,
global injectivity, and contact. Such a calculation would be new evidence,
not a reinterpretation of these files.
