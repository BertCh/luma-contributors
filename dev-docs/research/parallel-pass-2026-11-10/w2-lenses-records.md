# w2-lenses-records report

Scratchpad evidence: `/private/tmp/claude-501/-Users-robertchristie-Documents-GitHub-luma-contributors/310b8947-6a4f-4e58-adb4-aef2ba8200cd/scratchpad/` (`sizes-before.txt`, `sizes-after-1.txt`, `records-before/` copy of the 410 old records, `all-diff.txt` with a per-record structural diff of old against new, `sample-diff.txt`, `captures-changed.txt`, `numeric-captions.txt`, scripts `compare-records.mts`, `validate-records.ts`, `snap.mts`).

## 1. Done

### Item 1: failing lens specs

- `s3-geometry-pack.node.spec.ts` (GPUShapeGenerator).
  - Cause: the kernel no longer computes the destination inline. It calls the shared half-angle `geodesicDestination` in `geometry-measures/geodesic-wgsl.ts`, so `latitude2` and `angularDistance` are gone, and the old formula (`asin(sin φ cos δ + cos φ sin δ cos β)`) no longer describes the arithmetic.
  - Fix in `modules/lenses/src/gpu-spatial-analysis/computation-pack/operators/geometry-formulas.ts`: the formula now states the half-angle form.
    - `Δs = −2 sin φ sin²(δ/2) + cos φ sin δ cos β` and `sin φ' = clamp(sin φ + Δs)`.
    - `Δφ = 2 asin(Δs / (2 cos((φ+φ')/2)))`, with the `asin` fallback when that cosine is at most 1e-3.
    - `Δλ = atan2(…)` and `p' = c + 180/π (Δλ, Δφ)`.
  - The formula is bound to `phi1`, `angle`, `sinDelta`, `sinPhi2`, `deltaPhi` and `deltaLambda` in `geodesic-wgsl.ts`, and to `bearing`, `distance`, `EARTH_RADIUS`, `center` and `positionsOut` in the shape generator.
  - The spec's source list for `GPUShapeGenerator` gained `geodesic-wgsl.ts`.
- `statistics-regions-pack.node.spec.ts` (GPUSkaterRegions).
  - Cause: the gain is now the direct Ward form `count * remainder * squaredDifference / regionCount` (`gpu-skater-regions.ts:798-808`), so `getSsd`, `squares` and `sumSquares` no longer exist.
  - Fix in `statistics-regions.ts`: `g_e = n_A n_B / n_R · Σ_c d_c²`, with `d_c = S_Ac/n_A − (S_Rc − S_Ac)/n_B`.
  - It is bound to `count`, `remainder`, `regionCount`, `squaredDifference`, `difference`, `childSum` and `regionSum`. The ascii also states that a non-finite gain is not a candidate.
- All other lens node specs: the only other failures were record drift: `record-structure` for 4 gpu-network records, and `lens-twins` for 2 stale twins. LN-REC2 resolved both.

### Item 2: the 15 lens call sites and the stale lens texts

Each line gives the kernel evidence first, then what changed.

- Terrain derivatives, curvature and solar irradiance (one-sided border, `gpu-terrain-derivatives.ts:145-154`, `gpu-terrain-curvature.ts:236-248`):
  - Both pack entries (`gpu-terrain/computation-pack/operators/analysis.ts`, `geomorphometry.ts`) gained a `borderMode` choice and an ascii sentence on the one-sided quadratic extension.
  - The `terrain-derivatives` and `terrain-curvature` lenses record the choice `borderMode: 'one-sided'`.
  - Comments were added in the three examples.
- k-means (seeded k-means++), Louvain, K function and conditional simulation (seeded draws):
  - No caption or value stated old numbers.
  - The Louvain reason now says the visit order is a Philox Fisher-Yates shuffle (`gpu-network-louvain.ts:35-82`). The Louvain lens graph has 4 vertices and no seed, so it is unaffected.
  - The conditional simulation ascii now says the Box-Muller radius uniform is open, (0, 1) (`gpu-conditional-gaussian-simulation.ts:625`).
  - Docs captions for the K function (278 pairs, K ≈ 35,210.9) were checked against the new record: unchanged.
- Cross-validation trio (`gpu-spatial-cross-validation.ts:185-224`): the GPUSpatialCrossValidation reason now states the block order: a Philox key per block, fold = rank mod foldCount, so folds differ by at most one block. The nested and OLS lens specs already pass with the new folds.
- Network coarsening (`gpu-network-coarsening.ts:292-308`): the reason now states 64-bit fixed-point sums whose unit follows the largest term, so each sum is the exact sum rounded to float32. Docs caption "216 intersections" was checked: unchanged.
- Drive-time catchment (`drive-time-catchment-recipe.ts:185-190`, `:870-945`):
  - Formula `gpu-network.catchment.band` changed from `t < B[k]` to `t ≤ B[k]`. The ascii now gives `(a, b]`, the demand-time rule per direction (`cost >= 0`), the direct along-road cost to facilities on the same road, and the +Infinity for unreached demand.
  - Caption in `docs/api/gpu-network/operators.md:850`: "about 44.8 seconds" became "about 58.6 seconds" (datum 5: 44.78 before, 58.62 after; band 0 both times).
- Network isochrones (`gpu-network-isochrones.ts:309-312`, `:644`): the ascii adds upper-inclusive bands and the max-mode rule. The comment in `network-isochrones.lens.ts:20` changed from `[a, b)` to `(a, b]`.
- Coverage simplification (`gpu-coverage-simplification.ts:98-108`): the `topologyStatistics` meaning and the pack reason now count swept spans as well as crossing pairs.
- Raster reconstruction (`gpu-raster-reconstruction.ts:63`, `:141`): the ascii now says N(c) is the 8-neighbourhood by default. A comment in `expansion-runner.ts` explains the corner flood.
- Stale texts:
  - `isolines.ts`: the saddle rule is now the bilinear decider (`ISOLINES_SADDLE_WGSL`, `isolines-utils.ts:80-89`).
  - `indicators.ts`: the PCA is a closed form for d=2, otherwise Jacobi plus polish power steps, with the zero-sum tie sign rule (`gpu-composite-score.ts:712-790`). The review's "d+1 starts" description is itself stale.
  - `cells.ts`: fallback accuracy now runs up to resolution 10, with the measured errors (`gpu-cell-measures.ts:106-113`).

### Item 3: kepler

- `analysis-bindings.ts` `addKeplerHotSpots`: decided to **validate**.
  - Dropping the prop would silently change the statistic. Translating is impossible for a fractional `selfWeight`. Host parsers accept any `selfWeight >= 0`.
  - With `permutation` and statistic `localG` or `localGStar` (explicit, or the recipe's default from `selfWeight`), any other self weight throws before graph work, in host terms: `kepler-hot-spots analysis.selfWeight must be 1 when analysis.permutation uses 'localGStar', not 0.5; omit permutation to map another self weight without confirmation`.
  - Other statistics, and calls without `permutation`, pass through unchanged.
  - Today no kepler caller passes `permutation`: `recipe-bindings.ts` lattice does not, and `point-hotspots.ts` uses GPUHotSpotAnalysis directly.
- New spec `modules/kepler/test/hot-spots-self-weight.node.spec.ts` (2 tests): 4 rejected combinations; 6 accepted ones run to completion on the null device.
- `catalog-gpu-network.ts:1169`: the `isochrone-bands-exclusive` diagnostic now says `(minimum_cost, maximum_cost]`.
- k-means call site (`catalog-platform4-geometry-groups.ts:279`): its text is still accurate (seeded k-means++, relabelled) and its spec passes.

### Item 4: EC03

- Records captured: `gpu-spatial-analysis.sparse-resistance-connectivity-recipe` (234,500 B) and `raster-resistance-connectivity-recipe` (255,550 B).
- `urban-ecology.spec.ts` gained both elements.
  - Sparse: status `[1, >0, 1, 0]`; vertex currents 1 at vertices 0 and 8.
  - Raster: `solverStatus[0] = 1`; current at the gap cell 21 ≈ 1; potentials NaN at 3, 9, 15 and 27.
- Both workarounds were deleted in `lens-coverage.node.spec.ts`: the `isDroppedScenario` clause and the marker exemption.
- The raster scenario's `maximumIterations` went from 256 to 64, with a comment: the solve converges in 17 and still converges. At 256 the unrolled solver gave a 661 KB record (1,569 nodes, sources only 59 KB), so turning off source text could not bring it under budget.

### Item 5: LN-REC2

- Before: 410 records, 29,148,493 B in total, largest 389,076 B (buffer-surface). No record had source text. Twins: 1,849 files, 55,344 KB on disk.
- First capture: 412/412 passed in 93 s; 49,154,607 B in total. Three records were over 512 KB:
  - `gpu-terrain.terrain-height-above-drainage`: 526,834 B.
  - `gpu-terrain.terrain-hydrologic-indices`: 533,630 B.
  - `raster-resistance-connectivity-recipe`: 661,068 B.
- **Source text turned off for 2 scenarios**: `gpu-terrain.terrain-height-above-drainage` (now 477,904 B) and `gpu-terrain.terrain-hydrologic-indices` (now 489,960 B).
  - New `LENS_RECORDS_WITHOUT_SOURCE_TEXT` (scenario id to reason) and `getLensRecordSourceText(scenarioId)` in `lens-view-snapshot.ts`; `raster-lens-harness.ts` passes it as the `sourceText` option.
  - The raster resistance record was fixed by the scenario size instead (item 4). The budget was not raised and `OVER_BUDGET` is still empty.
- After: 412 records, about 48.9 MB (budget 64 MB), all within 512 KB. Twins: 1,855 files, 37,232 KB.
- No record lost a scenario. The old set is a subset of the new one; the two EC03 ids are new. `spatial-k-means` is now truly re-recorded.
- All 412 validate with `validateGPUComputationRecord` (0 invalid).
  - 405 carry source text: 412, minus the 2 turned off, minus records without helper kernels.
  - 407 carry bindings; 182 metrics are timestamped.
- Ten sampled records whose contributors nobody changed today (the "not reached" list): calendar-buckets, scalar-scale, time-window-filter, adjacency-matrix-order, network-flow, raster-evi, raster-ndwi, label-point, offset-curve, spherical-polygon-contains.
  - **Captures are identical in all ten.**
  - Differences are only:
    - added node `source`, `bindings`, `constants`, `dispatch` and `workgroupSize`;
    - added `refs.sources`;
    - graph `attrs`;
    - resource `ownership` flips;
    - timestamps (`recordedAt`, `createdAt`);
    - for network and raster records, timed metrics (`timestamped`, `timebase`, per-node entries).
  - The two network records also show changed `refs.formulas` ascii and tex: these are exactly my `gpu-network.isochrones.cost` and `gpu-network.catchment.band` edits, because network records carry the family's shared formula table.
- Overall, 110 of the 410 old records have changed captures (`captures-changed.txt`).
- Website: `node --experimental-strip-types src/lens/generate-view-models.ts` wrote 412 view models (6.95 MB). `summary.json`: recordCount 410 → 412, kernels 32,936 → 35,009, formulas 887 → 889.

## 2. Verified

- `npx vitest run --project node modules/lenses/test`: 95 files, 1,196 passed, 376 skipped, 0 failed. This includes record-budget, record-structure, record-encoding, lens-twins, lens-coverage and the pack specs.
- `LENS_COVERAGE_REQUIRE_RECORDS=1` on `lens-coverage.node.spec.ts`: 18/18.
- Headless, all lens specs except capture and benchmark (17 files): 165 passed, 374 skipped, 0 failed. This includes record-structure, kernel-records, urban-ecology (13/13), statistics-coverage, terrain-processing and nested-interpolation.
- `VITE_CAPTURE_LENS_RECORDS=1 … capture-records.spec.ts`: 412/412, then re-run for the 3 over-budget records (3/3).
- `UPDATE_LENS_TWINS=1 … lens-twins.node.spec.ts`, then the same spec without the variable: 7/7.
- Kepler node: 185 files, 1,217 passed, 1 failed (see Not done). Kepler headless: 52 files, 179/179. New kepler spec: 2/2.
- `typecheck-status 'modules/lenses|modules/kepler'`: 0 errors. `NODE_OPTIONS=--max-old-space-size=8192 node scripts/typecheck-tests.mjs`: exit 0, no output. `check-upstream-first`: no violations. `check-contributor-contract`: no new findings. biome on every edited file: clean.
- Not run: `yarn build`, which is forbidden. **A rebuild is needed**: the website reads lenses through `dist`, and the view models were generated with the stale lenses dist. Regenerate them after the rebuild to confirm. The website inventory (`extract-inventory.ts`) was not run: `website/src/catalog/generated` is not mine.

## 3. Not done

- `modules/kepler/test/resident-select-geometry.node.spec.ts` ("sliced native Arrow XYZ geometry") fails.
  - It tests `examples/deck/kepler-compute/src/resident-select-geometry`, which is not mine and does not involve analysis.
  - The likely cause is the `@luma.gl/graph/arrow` alpha.12 bump.
- The 135 numeric docs captions on records whose captures changed (`numeric-captions.txt`) were not checked one by one. I checked drive-time (fixed), coarsening, K function and isochrones (unchanged).

## 4. Website and examples work list

- After the rebuild, rerun `node --experimental-strip-types src/lens/generate-view-models.ts` from `website/` and the inventory extractor. No name changed for website or examples.

## 5. Roadmap

- LN-REC (2.12):
  - (a) LN-REC2: delete. Done; 2 terrain scenarios have source text off (`LENS_RECORDS_WITHOUT_SOURCE_TEXT`).
  - (b) to (d) remain as written.
- EC03-R: delete "lens records, headless lens assertions, obsolete workarounds" (done).
- DUP-02: delete "re-record spatial-k-means" (done).

## 6. Noticed (outside scope)

- No kernel defect confirmed. Drive-time demand times rose (demand 0: 79.5 → 231.8 s; demand 2: 0.88 → 35.9 s), not only fell. The lens road graph has 1,142 one-way arcs out of 2,810, so this matches network-flows-03: one-way edges are now honoured. The table in behaviour-changes.md lists only the decreases.
- `gpu-composite-score.ts:103` (analysis) still says "power iteration" for `enablePrincipalComponent`; the kernel now uses Jacobi.
- The terrain pack text at `gpu-terrain/computation-pack/operators/analysis.ts:103` (terrain contours, "asymptotic determinant") was not reviewed against its kernel.
