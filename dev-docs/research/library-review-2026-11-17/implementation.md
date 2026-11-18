# Implementation of the library review, 2026-11-17 to 2026-11-18

This note records what was implemented from the review in this folder, how it was checked, and what
is left. Caller-facing changes are in `CHANGELOG.md`; open work is in the LR section of
[`dev-docs/roadmap.md`](../../roadmap.md).

## Scope

The maintainer accepted all seven recommended decisions (LR-D1 to LR-D7) and asked for every fix to
be implemented. LR-01 to LR-21 were implemented. LR-23 (upstream filings) was drafted in
[`dev-docs/upstream-drafts`](../../upstream-drafts/) and not filed. LR-22 (the luma.gl alpha.16 bump)
was left out on purpose so that numerical changes can be attributed. Nothing was published.

No new operator family and no new public `GPU*` class was added. Wrong or inconsistent public names
were changed without aliases, because the package is unpublished.

## How the work was done

- 54 implementation streams, each a `gpt-6.1-sol` session with a brief naming its finding IDs,
  worked in one staging copy of the repository. Each kept a manifest of the files it touched, a list
  of specs to run and a status file. Each fix was required to land with a discriminating test: a
  fixture on which the old code gives a different caller-visible answer, seen failing first.
- Each stream was reviewed by a separate `gpt-6.2-astra` session that had not written the code, and
  fixed what the review found. The implementors could not run a browser or a GPU; the lead ran every
  headless spec on WebGPU (Chromium, Metal) and sent the failures back.
- When every stream was clean and the first full gates had passed, eleven reviewers read the
  combined diff by area. They reported 147 findings, 42 of them graded as wrong results or broken consumers.
  Almost all were of one kind: one stream had changed the meaning of a status or a name
  (`overflow` against `candidateOverflow`, a completeness flag, `Ids` against `Rows`) and a consumer
  written by another stream, or an example, lens or website entry that no stream owned, still read
  the old meaning.
- The owners fixed those. The same reviewers then re-ran their reproductions: 87 of the 147 were
  confirmed fixed, 48 partly fixed and 12 fixed wrongly, and they reported 102 findings that were
  still open or new in the fix diff (63 graded as wrong results). Three kinds recurred: props, port maps or status maps read again after construction (so a
  getter or a later mutation changes what was validated); a rejected call that leaves the graph
  unusable for a corrected retry; and a showcase, explorer, website or Kepler consumer that turns a
  suppressed or incomplete result into a definite statement. The owners fixed those in a second
  round. That round was not reviewed a third time.

The lesson of the review holds for its own implementation: passing specs did not find these
defects, and each independent reading found more. The second fix round is covered by its own tests
and by the gates below, not by a further review.

## What the gates found that reading did not

- A WGSL helper called with the wrong number of arguments, which made every dasymetric submission
  invalid; a suppression test that read a status buffer nothing wrote, so every result was NaN.
- A constant NaN produced by `bitcast` in shader source, which the shader compiler rejects; the
  shared runtime helper is used instead (trajectory mobility, modal split, SKATER).
- The shader memory estimator, once it counted local arrays honestly, found one production shader
  function 8 bytes over the 8,192-byte limit (raster seasonal break, `leastSquares`); three
  accumulators were merged.
- Status resources named per contributor instead of per output collided when two weights producers
  shared a graph.
- An analysis spec that imported scene code from both example applications pulled their deck.gl UI
  modules into the browser bundle and failed to load; the wiring under test moved to deck-free modules.

## Gates on the final tree

Run in the staging copy on 2026-11-18, one MacBook (36 GB, Chromium headless, WebGPU on Metal), on the
tree that was landed. The steps ran as separate parallel jobs, not as one `yarn verify` invocation:
`yarn verify` refused its last step when free disk fell just under its own 8 GiB floor (7.98 GiB), so
the packed-consumer check was run directly.

| Gate | Result |
| --- | --- |
| `yarn lint` (Biome, 10,300 files) | no errors |
| `yarn check-upstream-first` | no violations |
| `yarn check-contributor-contract --check-stale` | no new findings, no stale entries |
| `yarn typecheck` | pass |
| `yarn typecheck-tests` (every stage) | pass |
| `yarn build`, `yarn verify-package-exports` | pass |
| `yarn check-packed-consumer` | pass: 11 entry points (ESM, CommonJS, types), 28 README links, notices for 1,755 files, the documented CPU first run |
| Node suite | 1,803 of 1,808 files (23,768 tests) in the full run; the four failing files were fixed and rerun with the cross-cutting folders (206 files, 5,213 tests pass) |
| GPU suite, analysis | 1,205 of 1,205 files, 14,105 tests pass, 2 skipped |
| GPU suite, lenses | 27 of 27 files, 195 tests pass; 384 structure checks skip by design (see below) |
| GPU suite, kepler | 163 of 164 files; the failure predates this pass (see below) |
| Lens capture | all 427 scenarios capture; twins, website models (427) and the catalog inventory regenerated |

## Review convergence

The same eleven reviewers were resumed for each round, so each re-ran its own reproductions.

| Round | Findings | Graded wrong result | Earlier findings confirmed fixed |
| --- | --- | --- | --- |
| 1 (combined diff) | 147 | 42 | not applicable |
| 2 (fix diff) | 102 | 63 | 87 of 147 |
| 3 (severity tied to ordinary, unusual or extreme use) | 52 | 20 | 89 of 102 |
| 4 (ordinary use only) | 1 | 1 | 10 of 11 areas clean |

Round 3 was the first to converge. Until then, findings that need a caller to mutate props after
construction, alias two ports to one buffer or retry after a rejected call were graded as wrong
results and grew each round. They are fixed where the fix was local (construction snapshots,
validation before the first `graph.add`) and otherwise refused at construction or documented.

## Second tranche

Twelve further streams worked through "Left open" in a second staging copy while the first tranche's
last review rounds finished, and the two were merged (twelve conflicted files, resolved by their
owners). What landed:

- Multipart point and line feature joins for `intersects`, `within`, `coveredBy` and nearest
  neighbours, on shared multipart ports (GEO-12; other predicates still refuse).
- Minimum spanning forest, SKATER and REDCAP read weights loss on the GPU instead of refusing
  status-bearing weights (LR-07).
- Network flow path decomposition in the GPU form and the CPU twin (NET-10).
- Every recipe prepares its whole declaration before touching the caller's graph, so a rejected
  configuration leaves the graph unchanged (RCP-04). The cost is a second declaration per recipe.
- A declaration, choice and record-signature gate for all 427 lens scenarios (LENS-06), and the
  lens public surface as named exports (LENS-09: 461 names to 8 at `/computation-views`).
- A tile-count ceiling of 4,096 on the three uncapped regression merges; thirteen loose root files
  moved; the shader memory audit's last unresolved layouts cleared; the kepler WKB selection spec
  that failed before the pass fixed (an empty chunk bound with an aligned size); website and
  example typechecks clean and built in CI before typechecking; a typecheck measurement script.

Five reviewers then read the twelve streams together with the merge: 49 findings, 22 graded wrong
results in ordinary use, all sent to owners and fixed. That fix round was not reviewed again.

### Gates on the merged tree

Same machine, parallel jobs. Free disk fell to 2.4 GB during this pass because swap grew under the
parallel load; the first staging copy was removed and later steps ran one at a time. `yarn verify`
needs 8 GiB free and could not run on the merged copy; it ran on the checkout after landing (see
"`yarn verify` on the landed checkout").

| Gate | Result |
| --- | --- |
| Lint, upstream-first, contributor contract (`--check-stale`) | clean |
| `yarn typecheck`, `yarn typecheck-tests` | pass (every test stage) |
| `yarn build`, `yarn verify-package-exports` (55 entry points), `yarn check-packed-consumer` | pass |
| Node suite | 1,844 of 1,854 files (28,384 tests) in the full run; the ten failing files were fixed and rerun with the affected folders (308 files, 9,981 tests pass) |
| GPU suite, analysis | 1,211 of 1,215 files in the full run; the four failing files were fixed and rerun green |
| GPU suite, lenses | 30 of 30 files, 204 tests |
| GPU suite, kepler | 163 of 164 files in the full run (the WKB spec now passes); the kriging binding spec was fixed and rerun green |
| Lens capture | all 427 scenarios capture; twins, website models and inventory regenerated |

### `yarn verify` on the landed checkout

Running the gate on the real checkout, rather than a staging copy, found two things no staging run
could. Four tests and one README link depended on the staging copies (a staging commit hash, a
`git show` of a revision, files under a scratch folder); they now check the same facts without git
and without scratch paths. One lens spec passed alone and failed after other specs in the
one-worker run; it and three neighbours that leaked fixtures were made order-independent and
checked under shuffled orders.

After those fixes every step passed on the landed checkout, across two runs rather than one. A
run resumed at the Node step passed the Node suite (1,853 files pass, 1 skipped; 28,472 tests;
1,544 s with one worker, peak 10.8 GiB) and the GPU suite (1,409 files in 212 directory chunks,
every chunk passing). A fresh run then passed lint, both checkers, source and test typechecks,
build, export and packed-consumer checks, and was stopped when it reached the Node suite again.
Only this write-up changed between the two.

## luma.gl alpha.16, prepared as a patch

The bump (LR-22) was done in a third staging copy on top of the landed tree and is delivered as a
two-part patch kept beside the checkout, not applied, so that a regression can be attributed to
either the review fixes or the bump. Part one (61 files) holds the pins, the lockfile, the source
and test changes and the docs; part two holds the lens records and website models recaptured on
alpha.16.

- Published alpha.16 changed the grid cell coordinate function. Five local readers carried a copy of
  the alpha.14 function and would have looked in a different cell from the one the published
  builder filled (x = −177 in [−180, 180] with 360 cells: cell 2 before, cell 3 now). They now
  include the published `GPU_GRID_COORDINATE_WGSL`. A 30-case GPU regression compares each
  reader's cell with the builder's.
- The kepler metered graph forwards compile options, which alpha.16 passes through.
- Checks on the bumped tree: lint, both checkers, source and test typechecks, build, 55 entry
  points and the packed consumer pass. GPU suite: analysis 1,214 of 1,216 files in the full run,
  lenses 30 of 30, kepler 164 of 164; all 427 lens scenarios capture. Of the two analysis files,
  the new grid regression had a defect in its own probe (a binding named with a reserved WGSL
  word) and passes 30 of 30 after the fix; the other is the open item below. Node suite: 1,855
  files pass, 1 skipped (28,516 tests). `yarn verify` was not run as one command on the bumped
  tree.
- Open: `gpu-cost-distance-continuation.spec.ts` failed its round-count equality in 2 of 10 runs
  on alpha.16 (167 and 163 rounds against 168; final costs equal) and in none of 9 on alpha.14. A
  later 20 runs all passed. No cause was found and no production change was made; the strict
  assertion stays and the spec now records raw state on a mismatch.

## Left open

See the LR table in the roadmap. In short: publishing and the canonical repository URL (LR-04);
applying the luma.gl bump patch and the continuation question above (LR-22); filing the upstream issues (LR-23); the remaining
multipart predicates and nearest modes (LR-08, LR-24); numerical regeneration of the visibility
oracle fixtures (needs GDAL, GRASS and R); recipe failures during the final graph mutation, which
published luma.gl cannot roll back (LR-10); peak-memory measurements for the typecheck (LR-19);
21 loose raster and network root files (LR-17); the regression tile ceiling (LR-26).

Findings that did not reproduce are listed as STALE in the stream status files; none of the 36
verified wrong-result defects was among them.

## Known gaps in the checks

- Only 39 of the 427 lens scenarios have a structure check, so a stale lens record is not detected
  by a test for the others. After the last fix round every scenario was recaptured on the GPU (all
  427 capture, including the 14 that had been pending) and the twins and website models were
  regenerated. Recapturing everything also found two scenarios that no longer ran and six that
  recorded incomplete choices; no other check runs every scenario.
- Every `yarn verify` step passed on the landed checkout, but across two runs, not as one
  uninterrupted command.
- The second tranche's cross-stream fixes, the round-four fix, the staging-dependency fixes, the
  order-independence fixes and the bump were not read again by a reviewer.
