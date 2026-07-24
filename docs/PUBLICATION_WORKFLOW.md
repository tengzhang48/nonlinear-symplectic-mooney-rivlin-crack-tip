# Public companion curation and release workflow

This repository is a curated reproducibility companion, not a mirror of the
private research worktree. The distinction is deliberate: a reader should be
able to locate the governing theory, execute the checks, identify the exact
figure inputs, and see what remains open without navigating internal handoffs
or superseded calculations.

## Public inclusion policy

| Include | Reason |
|---|---|
| analysis and verification code used by a public claim | makes the theory-to-code edge executable |
| concise theory and scope notes | exposes assumptions, notation, and open boundaries |
| curated FEM and analytic arrays used by figures | permits deterministic reproduction without a full re-solve |
| solver and mesh code plus pinned environments | permits an independent, more expensive recomputation |
| figure generators and rendered PDF/PNG assets | binds panels to public inputs |
| machine-readable claims and hash manifests | binds claims, files, and bytes |
| license and citation metadata | makes reuse and attribution explicit |

## Deliberate exclusions

The public companion does not contain:

- raw AI conversations or signed agent-review reports;
- internal handoff, scratch, and working-note files;
- superseded code presented as an alternative current result;
- unfinished or deliberately unrun numerical batteries;
- machine-specific job launchers;
- publisher-owned article PDFs; or
- manuscript/referee files not needed to reproduce the public evidence.

General lessons from the research process are synthesized separately in the
public
[`ai-mechanics-resources` crack-tip case study](https://github.com/tengzhang48/ai-mechanics-resources/blob/main/case_studies/symplectic_mooney_rivlin_crack_tip.md).
That case study reports transferable failure modes and mitigations rather than
shipping internal review documents.

## Evidence graph

Every paper-facing public item should fit the chain

```text
physical claim
  -> governing equation and scope
  -> verification or solver code
  -> exact input artifact
  -> derived summary
  -> figure panel
  -> hash manifest and release commit.
```

`docs/CODE_MAP.md`, `data/claims/principal_claims.json`, and
`docs/FIGURES.md` expose the three central parts of this graph. A missing edge
is a release blocker even if the final image looks plausible.

## Update procedure

### 1. Classify the change

Record whether it changes:

- theory or claim scope;
- executable analysis;
- stored data or derived summaries;
- a figure generator or rendering;
- dependencies/environment; or
- documentation only.

A correction can affect several categories. Do not classify a mathematical
change as documentation-only merely because the numerical headline survives.

### 2. Propagate dependencies

For a changed theory statement, inspect at least:

```text
docs/THEORY_NOTES.md
docs/CODE_MAP.md
data/claims/principal_claims.json
README.md
docs/FIGURES.md and panel annotations
docs/PROCESS_AND_LESSONS.md
```

For a changed dataset or mesh, inspect the extractor, summary, claims gate,
figure generator, directory artifact lock, and repository manifest.

### 3. Preserve the two reproduction lanes

- **Stored-data lane:** fast, deterministic checks and rendering from the
  exact curated arrays.
- **Fresh-solve lane:** independent FEM recomputation in the pinned solver
  environment.

A fresh solve must not overwrite curated inputs automatically. A stored-data
pass must not be described as a fresh solver reproduction.

### 4. Execute the release circuit

From a clean candidate worktree:

```bash
python tests/run_verification.py
python tests/check_claims.py
python analysis/profile_mode_audit.py --check-stored
python figures/make_figures.py
# In the pinned FEniCSx environment when the ESI mesh changes:
python figures/make_esi_mesh.py
python make_manifest.py
(cd data/fem/strip && sha256sum -c ARTIFACTS.sha256)
sha256sum -c MANIFEST.sha256
git diff --check
```

The profile audit in this circuit is strip-only and reproduces the ESI table;
it creates no figure and does not resolve the residual exponent. Run the
FEniCSx mesh/solver lane when the affected change touches geometry,
discretization, solver behavior, or stored FEM provenance. It is not required
for a prose-only correction that leaves the curated arrays and generators
unchanged.

### 5. Review the candidate as a public reader

Before pushing:

- inspect every changed rendered figure;
- follow every README and figure-map link;
- confirm no ignored publisher PDF or local output entered the index;
- scan for hard-coded local paths and credentials;
- verify that open work is not presented as an established result;
- verify that the title and citation metadata match the current manuscript;
- inspect `git status` and `git diff --check`; and
- review the generated archive or clean clone, not only the development tree.

### 6. Freeze and publish

Commit the exact candidate, push it to the public default branch, and verify
the anonymous browser/clone view. For an article release, tag the exact commit
named by the manuscript and attach or archive the same manifest. Add a DOI only
after an immutable record exists.

## Correction protocol

When a public claim is weakened or withdrawn:

1. identify every dependent text, code, data, and figure artifact;
2. preserve historical provenance outside the current public path when useful;
3. update the current claim ledger and theory note first;
4. regenerate affected summaries and figures;
5. rerun the complete relevant release circuit; and
6. state the new evidence boundary without replacing it with an unsupported
   physical or observability argument.

Passing old tests does not grandfather a corrected release. A fix creates a
new candidate with a new evidence graph.

## Release gate

A public update is ready only when:

- the public claim has a stated mathematical and physical scope;
- its verification path is present and runnable;
- all figure inputs are tracked and mapped;
- stored and fresh-solve provenance are not conflated;
- the rendered assets and manifests match the candidate commit;
- excluded internal or third-party files remain absent; and
- remaining research questions are explicitly nondependencies of the shipped
  claims.
