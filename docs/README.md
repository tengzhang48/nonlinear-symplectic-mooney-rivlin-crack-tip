# Documentation

This directory collects the long-form notes for the public reproducibility
companion. Code- and data-specific README files remain beside the components
they describe.

## Understand the scientific result

- [`THEORY_NOTES.md`](THEORY_NOTES.md) derives the plane-stress reduction,
  constrained crack-tip field, radial Hamiltonian structure, leading
  predictions, closed opening sector, and bounded higher-order results.
- [`FIGURES.md`](FIGURES.md) maps every paper and ESI panel to its generator
  and exact tracked inputs.

## Reproduce and audit it

- [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) gives the clean-clone workflow,
  expected checks, stored-data route, and fresh finite-element route.
- [`CODE_MAP.md`](CODE_MAP.md) states the role and evidence level of each
  public analysis, verification, solver, and figure script.
- [`../data/README.md`](../data/README.md) describes the curated arrays and
  machine-readable claim records.
- [`../fem/README.md`](../fem/README.md) introduces the strip,
  matching-circle, and focused-disk finite-element calculations.

## Maintain the public companion

- [`PUBLICATION_WORKFLOW.md`](PUBLICATION_WORKFLOW.md) records the inclusion
  policy, dependency propagation, release circuit, and correction protocol.
- [`PROCESS_AND_LESSONS.md`](PROCESS_AND_LESSONS.md) records reusable lessons
  from the human–AI-assisted research and verification process. It is a
  process note, not external peer review.

Unless a document says otherwise, paths shown in code font are relative to
the repository root.
