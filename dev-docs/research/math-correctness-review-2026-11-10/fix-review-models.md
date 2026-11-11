# Fix review: scope `models` (2026-11-10)

Independent review of the math-fix pass against the snapshot
`f5dcdc4119616fcd21b9f44ea84bee571db6b09c`, for the directories of
`modules/analysis/src/gpu-spatial-analysis` named in the brief (interpolation, density, validation,
point process, change points, distribution dynamics, clustering, regionalization, location
allocation, accessibility, landscape ecology, urban morphology, recipes), their specs and their
`docs/api` pages. Checked against the Resolution sections of `interpolation-density.md`,
`point-process-clustering.md` and `applied-recipes.md`.

Result: no blocker and no major finding. Nine minor findings (eight confirmed, one suspected).
Every item the lead asked to be looked at hard computes what its Resolution row says.

Paths are relative to `modules/analysis` unless they start with `docs/` or `dev-docs/`.

## 1. Coverage

Read in full (the diff and the surrounding kernel):

- `src/gpu-spatial-analysis/change-points/gpu-location-change-points.ts`
- `spatial-validation/gpu-spatial-cross-validation.ts` (whole file), the fold wiring of
  `gpu-nested-interpolation-validation.ts` and `gpu-spatial-cross-validated-ols.ts`
- `spatial-regionalization/gpu-skater-regions.ts` (diff, and lines 636-965: tree statistics, arc
  statistics, candidates, best gain, apply, relabel, ancestors), `regionalization-inputs.ts`,
  the standardize call of `gpu-spatial-weights-minimum-spanning-tree.ts`,
  `gpu-region-partition-evaluation.ts` (diff)
- `spatial-clustering/gpu-kmeans.ts` (seeding rounds 659-817, inertia lanes),
  `spatial-clustering-kernels.ts` (lattice and neighbour walk)
- `distribution-dynamics/gpu-class-assignment.ts`, `gpu-spatial-markov.ts` (diff), `index.ts`
- `point-density/point-density-kernels.ts` (key kernel), `space-time-kernel-density-math.ts`,
  `line-density/line-density-kernels.ts`, `gpu-line-density.ts` (documentation)
- `spatial-interpolation/gpu-local-kriging.ts` (every hunk, and the kernel body 756-850),
  `gpu-kriging.ts`, `gpu-vecchia-log-likelihood.ts`, `gpu-vecchia-space-time-log-likelihood.ts`,
  `vecchia-likelihood-nodes.ts` (site-term kernel), the two conditional simulation hunks,
  `prediction-covariance.ts` (comment), and
  `src/dataframe-analysis/pair-statistics/variogram-model.ts` (`VARIOGRAM_MODEL_SHAPE_WGSL`)
- `urban-morphology/street-profiles.ts` (facade kernel and predicate), `urban-ray-utils.ts`
- `location-allocation/gpu-location-allocation.ts` (score and choose kernels)
- `recipes/hot-spot-analysis-recipe.ts`, `recipes/spatial-regression-recipe.ts` (documentation)
- `catchment-accessibility/gpu-huff-trade-areas.ts`, `landscape-ecology/landscape-metrics.ts`,
  `segregation-counterfactual.ts`, `study-window/gpu-study-window-preparation.ts`,
  `scoring-rules/gpu-ensemble-scores.ts`
- Documentation-only diffs: `optimal-transport`, `space-time-hawkes`, `redcap-regions`,
  `gpu-rank-markov.ts`
- `src/analysis-core/kernels/philox-random.ts` (diff), `addition-rounding-error-wgsl.ts`
- Specs and oracles: `spatial-cross-validation-seeds.spec.ts` (whole file), the diffs of
  `statistics-upstream.spec.ts`, `spatial-validation-advanced.spec.ts`,
  `nested-validation-oracle.py`, `statistics-blockcv.R`, `stx06-location-change-points.spec.ts`,
  the two Vecchia specs, `gpu-clustering-extras.spec.ts`, `gpu-location-allocation.spec.ts`,
  `recipe-status-stages.node.spec.ts`, and both grid-coordinate pin specs (whole files)

Read in part:

- `gpu-local-kriging.ts` grid-index rework (chunked candidates, per-query rescan, the new `'auto'`
  rule). It belongs to the other session (`dev-docs/research/parallel-pass-2026-11-10/fix-indexes.md`,
  findings 3 and 4), not to the math Resolution. I read the diff, not the search it drives.
- `landscape-ecology/raster-resistance-connectivity.ts`: the file is untracked and not in the
  snapshot, so no diff exists. I read the documentation block (233-248) and checked its numbers.
- New specs, by outline and assertions only: `gpu-skater-regions-gain.spec.ts`,
  `standardize-columns.spec.ts`, `street-profile-facades.spec.ts`,
  `local-cokriging-missing-variable.spec.ts`, `gpu-point-density-cell-mapping.spec.ts`.
- `docs/api`: searched for statements the changes contradict; not read line by line.

Not read:

- Spec bodies of `kriging-small-lag`, `variogram-model-shape`, `conditional-simulation-open-uniform`,
  `landscape-metrics-counts`, `gpu-region-partition-evaluation-nonfinite`, the study-window,
  line-density, ensemble-score and distribution-dynamics additions.
- `variogram-model-fit-nodes.ts` beyond the shape call (weight normalisation and the new status 2
  are in `dataframe-analysis`, outside this scope).
- `gpu-focal-statistics.ts` (excluded by the brief). Lens twin records and website runs.

Spec runs (four of the six allowed), all passed:

- `--project headless`: `change-points/stx06-location-change-points.spec.ts` and
  `spatial-regression/spatial-validation-advanced.spec.ts`, 29 tests. Settles that the Poisson
  variant compiles (see section 3, E1) and that seed 8 gives the oracle's inner partition in both
  outer folds.
- `--project node`: `point-density/point-density-grid-coordinate-pin.node.spec.ts` and
  `spatial-interpolation/grid-coordinate-pin.node.spec.ts`, 2 tests.

Scratch scripts are in
`/private/tmp/claude-501/-Users-robertchristie-Documents-GitHub-luma-contributors/9972295f-91de-4373-ba95-d2f799611a84/scratchpad/review-models/`
(`facade.mjs`, `hash.mjs`).

## 2. Findings

| ID | Severity | Status | Where | What is wrong | Evidence | Fix I would make |
| --- | --- | --- | --- | --- | --- | --- |
| fr-models-01 | minor | CONFIRMED | `src/gpu-spatial-analysis/urban-morphology/street-profiles.ts:58-59` | The "lower row owns a coincident stretch" rule is pairwise and not transitive. Three walls within `2 * slack` of each other, in reverse row order, hide one another in a cycle and the stretch has no owner: the side reports 0 length for a wall in plain view. | Street `(0,0)-(100,0)`, `maximumDistance` 50, default tolerance, so `slack = 0.001001`. Rows 0, 1, 2 span x 20 to 80 at y = 10.0016, 10.0008, 10. Row 2 is hidden by row 1 (within slack, lower row), row 1 by row 0 (same), row 0 by row 2 (nearer by more than slack). The float32 replica `facade.mjs` prints `visible=false` for all three. Two walls can never do this; it needs three. | Decide ownership against the nearest hit: with `nearest` the smallest candidate distance on the sightline (own wall included), the wall is visible when it is within `slack` of `nearest` and no lower row is. That set is never empty and has one lowest row. |
| fr-models-02 | minor | CONFIRMED | `docs/api/gpu-spatial-analysis/coverage-statistics.md:367-368` | Still says "GPU masks are compared exactly against both. The seed hash is our explicit deterministic assignment rule". The hash is gone and the spec now compares the designs as partitions. | The same page was updated at line 201 and `spatial-validation-advanced.md:3` describes the Philox rank. | Reword: the designs are compared with blockCV as partitions; blocks are dealt in a seeded Philox order. |
| fr-models-03 | minor | CONFIRMED | `docs/api/gpu-spatial-analysis/interpolation-expansion.md:40`, `operators-density.md:571-579`; `src/gpu-spatial-analysis/spatial-interpolation/gpu-local-kriging.ts:245-247` | The new public prop `emptyNeighborhood` appears nowhere under `docs/api`, and the co-kriging row still says only that `"ordinary"` "imposes a separate unknown intercept per variable", with no word on the dropped constraint. The class documentation says a neighbourhood without a target sample "reports status 2"; with fewer retained rows than kept constraints it is status 1. | `grep -rn emptyNeighborhood docs/api` is empty. Two variables, one retained secondary sample, no target: `driftColumnCount` is 2, `retained` is 1, and `if (retained == 0u \|\| retained < driftColumnCount) { return; }` leaves status 1 (unchanged from before the fix). | Add both to `interpolation-expansion.md`; say "status 1 or 2" in the class documentation. |
| fr-models-04 | minor | CONFIRMED | `src/gpu-spatial-analysis/distribution-dynamics/gpu-spatial-markov.ts:296` | `lagIntervalClosure` is not validated in the `GPUSpatialMarkov` constructor. A wrong value throws from the inner `GPUClassAssignment` inside `getCommandNodes`, with the inner ID (contract section 10: construction-time checks). No spec passes the prop, so the pass-through is checked by reading only. | The constructor is lines 142-183; the only use is line 296. `grep -rn lagIntervalClosure modules/analysis/test` is empty. | Validate in the constructor with `validateRequirement`; add one spatial Markov case with a lag tied at a lag edge. |
| fr-models-05 | minor | CONFIRMED | `test/gpu-spatial-analysis/spatial-interpolation/local-cokriging-missing-variable.spec.ts:134-137` | The spec uses two variables with `targetVariable: 0`, so the kept-constraint table is always `[0]`, the identity. The indirection `driftColumns[column]` (kernel lines around 611-640) is never exercised with a real remapping, and a version that ignored it would pass. | With variable 0 present and 1 absent, `driftColumns = [0]`. A remapping needs `targetVariable: 1` with variable 0 absent, or three variables with the middle one absent (`[0, 2]`). I read the code as right for both. | Add one case with three variables and the middle one absent, expected values from a float64 two-variable solve. |
| fr-models-06 | minor | CONFIRMED | `test/gpu-spatial-analysis/recipes/recipe-status-stages.node.spec.ts:602-641` | The hot spot spec checks the `statistic` string the permutation contributor receives and the three throws. Nothing checks that, with `selfWeight: 0`, the mapped Gi and the permutation's observed `'localG'` are the same number, which is the defect the wiring fix was for. | The spec is a node spec and never runs a graph. | One headless case: `selfWeight: 0` on a small lattice, the permutation stage's observed statistic against the Gi stage's z-score ordering, or a hand value. |
| fr-models-07 | minor | CONFIRMED (declared) | `src/gpu-spatial-analysis/geographic-distribution/gpu-rose-statistic.ts:323`, `spatial-interpolation/gpu-natural-neighbor.ts:310-311`, `urban-morphology/isovists.ts:170` | Bare `atan2` whose second argument is a coordinate difference. Contract section 11 says such a call uses `guardedAtan2`. The Resolution leaves them as the builtin because the test GPU bins the one `-0.0` input correctly. | The contract rule is about other GPUs, which the passing spec on this GPU says nothing about. `fix-atan2.md` lists the same sites as "next candidates". | Adopt `guardedAtan2` at the three sites (one line each), or record the exemption in the contract. |
| fr-models-08 | minor | SUSPECTED | `src/gpu-spatial-analysis/urban-morphology/street-profiles.ts:46-50, 79` | The comment says both obstacle indices "visit the same blockers, so they agree bit for bit". The loop over all segments treats a segment within `tolerance` of the sightline origin as a blocker at distance 0 even when it lies behind the origin; the BVH form prunes by `getBoxEntry`, which a box behind the origin fails. | Segment `(50,-5)-(50,-5e-7)`, sightline from `(50,0)` towards `+y`: `distanceToSegment` is `5e-7 <= 1e-6`, so the loop hides the wall; the box has `exit = -5e-7 < entry = 0`, so the BVH never tests it. Sightlines start at interval midpoints, which sit half an interval away from such an endpoint, so I could not build a facade input that reaches it. `castRay` has had the same shape since before this pass. | Pad the pruning test by `tolerance` behind the origin, or soften the comment. |
| fr-models-09 | minor | CONFIRMED | `dev-docs/research/math-correctness-review-2026-11-10/point-process-clustering.md:407`; `test/gpu-spatial-analysis/spatial-validation/spatial-cross-validation-seeds.spec.ts:142` | Two wrong statements. The Resolution says no Python package is importable "(no numpy, mapclassify or giddy)": `/usr/bin/python3` has NumPy 2.0.2 and SciPy 1.13.1 (the interpolation stream generated a fixture with it); mapclassify, giddy, libpysal and ruptures are indeed missing. The spec comment says the chance of a repeated design among 8 seeds is below 1e-3; it is about `28 / 5040 = 0.0056`. | Ran the imports. The spec is deterministic, so the comment does not affect the result. | Correct the words. |

## 3. Fixes checked and found right

Interpolation and density:

- interpolation-density-01: `getCoordinate` in `point-density-kernels.ts:286-290` is character for
  character the function in the published `gpu-grid-binning.js:193-197` and
  `gpu-grid-aggregation.js:236-240`. The pin spec reads the published files as text (no import) and
  passed when run; the IDW pin against `gpu-grid-index-internals.js` passed too.
- interpolation-density-02: one Philox word per `(seed, block)`; pass one marks the lowest usable
  row of each block; pass two counts marked rows with a smaller `(key, block id)`, so the rank is
  over distinct blocks and a key tie is broken by block id. Masked and non-finite rows get
  `0xffffffff` and are never marked. Fewer blocks than folds leaves the higher folds empty, as
  documented. Nothing was lost in the scripted rewrite: the diff removes only the hash import and
  the old assign body. The two regression specs are not weakened: `statistics-upstream.spec.ts`
  requires each GPU fold to equal one blockCV fold in test rows and training rows and the matching
  to be a bijection; the oracle's literal `[1, 1, 0, 0, 1, 0]` is what the old `hash(block ^ 2) % 2`
  gives (recomputed in `hash.mjs`); a new GPU test pins seed 8 to that partition for both outer
  folds and passed.
- interpolation-density-03: the kept-constraint table always holds the target; the bordered system
  is `retained + driftColumnCount`; the zero block, the right-hand side and the variance term all
  use the reduced count. No target sample gives a zero constraint row and status 2.
- interpolation-density-05: `sin(a + h) - sin(a) = 2 cos(a + h/2) sin(h/2)`, as written.
- interpolation-density-06: the `'prior'` branch runs after selection and before any solve, writes
  `(mean, sum of support covariances, 1, 0)` and status 0, and is rejected outside mode `'simple'`.
  The site-term kernel then reads mean 0 and variance `C(0)`; the space-time kernel writes
  `spaceTimeCovariance(0, 0, true, true)`. Both specs use hand values and compare with the Fisher
  sibling.
- interpolation-density-07: the default one-slice window passes only when the window is defaulted;
  an explicit window still needs `start < end`.
- interpolation-density-08: each shape checked against its definition. Exponential series
  `a - a^2/2 + a^3/6 - a^4/24 + a^5/120`; Gaussian is the same function of `3 s^2`; Matern 3/2
  coefficients `(k - 1)/k!` give `a^2/2 - a^3/3 + a^4/8 - a^5/30 + a^6/144`; Matern 5/2 coefficients
  `(1 - k + k(k - 1)/3)/k!` give `a^2/6 - a^4/24 + a^5/45 - a^6/144` (no cubic term). The first
  omitted term is below 1e-7 of the result at the switch (0.1). Spherical is unchanged and 1 beyond
  the range. The range factors satisfy `a - ln(1 + a) = 3` and `a - ln(1 + a + a^2/3) = 3`.
- interpolation-density-12: the new SciPy fixture reproduces exactly (difference 0 for all eight
  kernels) when recomputed with `/usr/bin/python3`.
- Open Box-Muller uniform: `(word >> 9) + 0.5` is exact in float32, range `[2^-24, 1 - 2^-24]`.

Point process and clustering:

- point-process-clustering-01 and E1: the forward recurrence visits starts in ascending order, so
  `F(start)` is final when relaxed; an unreachable start is skipped (this subsumes the old
  `start < minimumLength` test); the strict comparison keeps the smallest start, as before.
  Shifting by the segment's first value leaves the mean, mean-variance and slope costs unchanged;
  `'variance'` and `'poisson'` use no shift. `bestCost[0] = -penalty` gives no penalty for the first
  segment. Slope time sums stay below `2^32` (`256 * sum t^2 <= 4.26e9`). By hand, `[0, 0, 10, 10]`
  with cost `'mean'`: standardized `[-1, -1, 1, 1]`, optimum one change at slice 2, `optimalCost`
  equal to the penalty in data units. The Poisson kernel's local `var series` shadows the new
  workgroup array of the same name, which is legal, and the parity cases pass.
- point-process-clustering-02 and E7: `nA nB / (nA + nB) |meanA - meanB|^2` equals
  `SSD(parent) - SSD(A) - SSD(B)` (values `0, 0, 6` cut as `{0, 0} | {6}`: 24 both ways). A
  non-finite gain returns before the key is written, and the key was cleared to 0 first.
- point-process-clustering-03: with the default the kernel text and the variant are unchanged, so
  the pipeline is the old one. `'right'` counts edges strictly below the value, which is
  `numpy.digitize(..., right=True)`.
- point-process-clustering-04: one kernel, called by AZP, Ward, max-p and the tree. A constant
  column has every `value - origin` equal to 0 in any association, so the result is 0.
- point-process-clustering-05: a neighbour across the antimeridian is within one minimum cell
  width of the image at `x - 360` or `x + 360`; the walk scans those columns over the same three
  rows and skips columns already visited, so no point is counted twice. All three callers use it.
- point-process-clustering-07 and E6: the race key is `-ln(u) / w` with one Philox block per
  `(row, round)`, so rounds are independent and round 0 is uniform. With no weight left every key
  is `0x7f800000` and the lowest valid row is taken, as documented.
- E2, E3, E5: read and found right.

Applied and recipes:

- applied-recipes-01: predicate and edge-on rule as described, apart from fr-models-01.
  `distanceToSegment` is the same quantity with the offset formed first.
- applied-recipes-02: the relation to Circuitscape holds (`g = 1 / (mean(R) * edge length)`); the
  2 x 2 case is `1 / (1/2 + 1/2 + 1/sqrt(2)) = 2 - sqrt(2)`.
- applied-recipes-03: the maximum like adjacencies `2 s (s - 1) + 2 m - shortfall` and the integer
  square root are right; sides map to the right cell dimension.
- applied-recipes-04: only mode 2 replaces the score with the row count. The p-median branches and
  the maximum coverage score are untouched, dense and sparse.
- applied-recipes-05, -06 and the hot spot recipe wiring: read and found right.

## 4. Tests that do not pin their fix

- fr-models-05 and fr-models-06 above.
- `lagIntervalClosure` has no spec (fr-models-04).
- Declared by the implementors and confirmed by reading: the ensemble score spec
  (interpolation-density-04) and the k-means inertia spec (E3) pass before and after the change on
  the test GPU.

## 5. Undeclared behaviour changes

- `GPUSpatialCrossValidation`: a block's fold now depends on which other blocks hold a usable row,
  because the rank is taken among those blocks. Masking the only row of one block can move every
  block ranked after it. The Resolution says fold numbers are no longer predictable from the block
  id but not this. `GPUNestedInterpolationValidation` passes each outer training mask, so its inner
  folds can differ between outer folds for one seed.
- `GPULocalKriging`, ordinary co-kriging: a neighbourhood with fewer retained rows than variables
  but at least one target sample now predicts where it reported status 1. The Resolution lists
  only the status 2 case.
- `GPULocalKriging`: the `'auto'` sample index rule, a new construction throw when neither the
  scan nor the grid fits, and the internal prop `sampleIndexChunkQueryCount`. These are the other
  session's and are recorded in `parallel-pass-2026-11-10/fix-indexes.md`, not in the math
  Resolution.

No exported symbol in this scope was renamed or removed. New files carry SPDX headers. The diff in
this scope adds no bare `assert`, no private `@luma.gl` import and no debug output.
