# Wave 3 library review (w3-lib-padding, w3-lib-indexes, w3-lib-raster-imagery, w3-lib-raster-algebra)

Reviewer: independent, read-only (only this file and a scratch spec folder were written; the scratch
folder `modules/analysis/test/review-scratch-w3/` was deleted at the end). Date 2026-11-11.

## 1. Findings, by severity

### Blocker

**B1. Missed consumer: `GPURasterTemporal` still gets `validity:` in a spec, the mask is silently
dropped and the spec now fails.**
- `modules/analysis/test/analysis-core/table-audit/table-audit-raster.spec.ts:240`:
  `validity: harness.integers(validity),` inside `new GPURasterTemporal({...})`. The file was not
  edited today (mtime 07:29). The imagery agent's scratch tsconfig and spec runs did not cover
  `test/analysis-core/table-audit/`.
- Evidence 1: a scratch tsconfig over the 205 files in `modules/analysis/test`, `modules/lenses`,
  `modules/kepler` that name a changed class gives exactly one error:
  `table-audit-raster.spec.ts(240,11): error TS2353: ... 'validity' does not exist in type 'GPURasterTemporalProps'`.
- Evidence 2: `npx vitest run --project headless modules/analysis/test/analysis-core/table-audit/table-audit-raster.spec.ts`
  fails "table audit site 2: temporal composites choose the right slice" with 33 mismatches, all
  from counting masked observations as valid, for example
  `sliceCount 16 median cell 2: got {"value":-8.75,...,"count":16} expected {"value":-7.25,...,"count":12}`.
- Fix: line 240 `validity:` -> `mask:`.

### Major

**M1. The imagery stream gave no website or examples work list, and three of the stale keys it
leaves are spreads that TypeScript will not flag, so masks are lost silently once `dist` is
rebuilt.** The full list is in section 2. The silent ones:
- `examples/deck/spatial-analysis-showcase/src/scenes/raster/grid-alignment.compute.ts:361` and
  `:367`: `referenceValidity` / `movingValidity` inside `...(mode === 'periodic' ? {...} : {...})`
  spread into `new GPURasterRegistration`. Excess-property checks do not apply to a spread, so the
  masked-linear registration runs with no masks and gives a different shift without an error. Not in
  either raster report.
- `examples/deck/spatial-analysis-explorer/modes/patches-mode.ts:195` and
  `examples/deck/spatial-analysis-showcase/src/scenes/raster/burn-severity.compute.ts:493`:
  `validity: labelValidity` in a `patchInput` object that is spread into `GPURasterPatchMetrics` and
  `GPURasterSieve`. The algebra report lists both lines, but does not say they are silent.
- Evidence: my constructor-argument scanner over `website/src` and `examples` (parses the argument
  of every `new <renamed class>(` and flags old keys and spreads), then reading each hit by hand.

**M2. New code breaks contract section 6: the row edges of `cellSizeMode` are shader literals.**
- `modules/analysis/src/raster-analysis/raster-grid/raster-cell-area-scale.ts` (lines 34 to 46,
  `getRasterRowAreaScaleWGSL`) and `raster-patches/gpu-raster-patch-metrics.ts` (`getGroundWeights`,
  `mix(${northEdge}, ${southEdge}, rowFraction)`) bake `northEdge` and `southEdge` from
  `grid.affine` into the WGSL. The contract says raster contributors "read the affine from
  parameters, not from shader literals". The terrain classes this copies (`getRasterGroundCellSizeWGSL`,
  `writeRasterGridParameterValues` in `analysis-core/raster-grid.ts`) read the edges from parameter
  slots. The distribution and categorical classes already take a `parameters` view (determinant in
  slot 6), so with a repacked georeference the cell area follows `parameters` while the ground scale
  follows the compile-time affine. The algebra report lists this under "Remains". It is a departure
  added today, not one inherited.
- Fix: write the north and south edges into parameter slots with `writeRasterGridParameterValues`
  and read them in `getRowAreaScale`. Or, until then, record the departure in the roadmap STD-06 row.

**M3. The island-attachment fallback is a silent single-invocation scan over every row above the
cut-over.**
- `spatial-weights/gpu-island-attachment.ts:324-332`: when `!proven`, that island runs `scanWGSL`, a
  loop over all `rows` (up to 65,535) inside one invocation, on the `'grid'` path and on `'auto'` above
  4,096 rows. Contract section 2 forbids data-sized loops in one invocation. Section 3 forbids brute
  force as the silent default above its cap. The TSDoc states the fallback, but no status port reports
  it.
- When it triggers, from the code: at least 16 rows within 1.00001 times the chosen distance (15 or
  more other rows coincident with, or tied at, the nearest distance); an empty list (non-finite
  derived bounds, when coordinates are around 1.7e38 or larger); a non-finite chosen or farthest
  distance (squared distances overflow). On realistic projected data these are rare. Geocoded
  duplicates with 15 or more points at one address next to an island are the plausible case.
- Correctness is fine. My adversarial spec (fewer than 16 finite rows among 5,000 with NaN rows;
  15, 16 and 17 rows; a ring of 24 equidistant rows around an island; separations of 1e-21 whose
  squared distances underflow; 1e19 coordinates; 40 coincident islands beside 3 nearer rows) gave
  grid equal to scan in all 5 cases.
- Fix: publish a scalar `approximation`-style flag, or an explicit `candidateOverflow` count of
  islands that rescanned, so the cost is visible. Or cap the rescan at a named maximum and report the
  islands left over.

**M4. Two padding specs poison rows that the kernel under test can never read.**
- `test/gpu-spatial-analysis/concave-hull/concave-hull-padding-rows.spec.ts`, the grouped case:
  `GPUGroupConcaveHull` runs Delaunay, retain and boundary on transient `packedTriangles` /
  `packedEdges` (`gpu-concave-hull.ts:315-366`). The caller outputs it poisons are cleared by
  `clear-triangles` and `clear-edges` before the scatter, so a kernel that read packed padding rows
  would still pass. The ungrouped case is meaningful. The Delaunay `materialize` kernel returns early
  for `index >= requiredCount` (`gpu-delaunay-tessellation.ts:1887`), so poisoned rows survive into
  `retain`, which must gate them.
- `test/gpu-spatial-analysis/delaunay-tessellation/delaunay-padding-rows.spec.ts`, the Delaunay
  cases: they poison Delaunay's own outputs, which Delaunay writes and never reads. They prove only
  that. The Voronoi case, with stale triangle rows as input, is meaningful.
- Fix: for the grouped hull, feed `retain`/`boundary` through a test that hands them a poisoned
  triangle view directly. Or say in the spec header that the grouped case checks only clearing.

### Minor

**m1. kNN on the grid path: a query whose squared distance to every target overflows gets an
arbitrary one of the tied targets, not the lowest ID.**
- My scratch spec: 200 integer targets in [0, 50]², queries `(1e30, 1e30)`, `(-1e30, 25)`,
  `(25, 3e38)`, `(-3e38, -3e38)`, `(1e19, 1e19)`, `(3.4e38, 0)`, k = 1. The BVH path matches brute
  force. The grid path returns `[4, 0, 1, 0, 0, 0]` where the f32 oracle gives `[0, 0, 0, 0, 0, 0]`.
- Cause: `neighbor-search-kernels.ts:430`,
  `reach = sqrt(max(bestDistances[K-1] - rowGapSquared, 0.0)) + slack` with both terms `inf` gives
  NaN, so the full-row column range shrinks to the last column and the lower-ID ties in other columns
  are skipped. Before today such queries were empty and counted in `invalidCount`, so this is new
  behaviour for cross joins. It is unrealistic, but it contradicts the "equals a brute-force scan"
  documentation.
- Fix: narrow the full-row range only when `isFiniteFloat(bestDistances[K - 1u])`. Or document that
  ties among overflowed distances are unordered.

**m2. A query far outside a side or a corner makes the kNN ring loop visit the whole lattice.**
- The stop rule (`neighbor-search-kernels.ts:464-492`) uses the smallest clearance over the open
  sides. For a query far beyond one side, the clearance of the perpendicular sides is only
  ring × cell size, so the rings run to `maximumRing`. That is O(cells) of cell tests per query in
  one invocation, although target loads are still pruned.
- Measured (scratch timing spec, headless, 200,000 targets in [0, 1000]², `gridSize` [512, 512],
  2,000 queries, k = 8, three runs after warm-up): inside 8.1 / 7.5 / 7.0 ms; 1e6 beyond the right
  side 25.4 / 28.7 / 26.3 ms; 1e6 beyond a corner 24.7 / 24.1 / 24.0 ms; just past a corner
  4.3 / 3.9 / 4.3 ms. The results are correct. The cost is not documented.
- Fix: bound each open side by `sqrt(sideClearance² + d⊥²)`, where `d⊥` is the query's distance
  outside the lattice along the other axis. Or say in the docs that queries far outside cost a full
  lattice walk.

**m3. Fixture `geometry-neighbor-search-clustered.json`: the change is exactly the intended one, but
two fields changed without being reported.**
- I compared the fixture at `74dddcf5` with the current one row by row. Scenes `clustered` and
  `collinear` are identical for every k and radius. In `cross`, only query row 9, at (-1, 4) and
  the only finite query outside the bounds, changed: knn 1 `[]` -> `[109]`, knn 4 `[]` ->
  `[9, 109, 449, 572]`, knn 32 `[]` -> 32 ids. Radius 2 and radius 5 are unchanged (no target is that
  close).
- I recomputed row 9 by numpy brute force over the valid targets (finite, inside the bounds, mask
  nonzero), with `(d², id)` lexicographic order. Neighbors and `distancesSquared` match for k = 1, 4
  and 32.
- Not reported: `versions.numpy` 2.5.4 -> 2.0.2 (the report gives only the new value) and
  `generator` `modules/graph/...` -> `modules/analysis/...`. The generator docstring still says
  `python3 -I`, which cannot import numpy with `/usr/bin/python3`.

**m4. The island-attachment spec has a misleading comment, and the ranking-disagreement rescan is
never exercised.**
- `island-attachment-index.spec.ts:221-245`: the comment says the islands attach to the lowest-row
  coincident representative at (10, 10). The islands are 40 coincident rows at (11, 10), so each one
  attaches to another island at distance 0.
- The agent says disabling the rescan fails only the ±1.5e38 case. So no spec covers the case the
  1e-5 margin exists for: the grid's f32 `dx²+dy²` ranking disagreeing with the scan's scaled
  comparison at the 16th slot.

**m5. Patch-metrics ground weights are quantized, not exact.**
- `gpu-raster-patch-metrics.ts` `getGroundWeights` rounds each row weight to 2^-20 of the largest
  row weight. The 64-bit sum is exact, but the TSDoc and `raster-patches.md` say only "exact 64-bit
  fixed point".
- The relative weight error grows toward the poles. NumPy, using the WGS84 formula of
  `getRasterGroundScale`:

  | Grid | Largest error | Where |
  |---|---|---|
  | Global, 0.1° | 4.65e-4 | at 89.95° |
  | Global, 1° | 5.1e-5 | |
  | 60-70°N, 1/60° | 6.9e-7 | |

  A float32 weight would be about 6e-8.
- Fix: say "weights quantized to 2^-20 of the largest row weight" in the docs. Or raise the scale:
  a run holds at most 16 pixels or 32 faces, so a 2^26 scale still fits a u32 contribution and makes
  the error 64 times smaller.

**m6. The new `cellSizeMode` specs compare the GPU with the library's own CPU twin.**
- `fractional-cell-size-mode.spec.ts:138` and `patch-metrics-cell-size-mode.spec.ts` build their
  "float64 reference" from `getRasterRowAreaScale` / `getRasterGroundScale`, so an error in the shared
  scale formula would pass both.
- Fix: one independent value, for example the pyproj `Geod.polygon_area_perimeter` of a 1° cell at
  60°N, with a tolerance that allows the row-center approximation (about 1.3e-5 relative for 1°
  cells).

**m7. `GPURasterFillNoData.output.status.invalidCount` gives the port a different meaning.**
- `analysis-core/contracts/status.ts:41` documents `invalidCount` as "Number of input rows rejected by
  validation". The fill class uses it for output cells still missing after the search.
- `status` is now required, and `validateGPUStatusPort` calls `Object.entries(status)`. So a
  JavaScript caller still passing `unresolvedCount` (the website run, until it is fixed) gets
  `TypeError: Cannot convert undefined or null to object`, not a message with the contributor ID.
- Fix: add a sentence to the port documentation ("or cells left invalid by the operation"), and add a
  `validateRequirement(id, props.output.status !== undefined, 'output.status is required')`.

**m8. STD-04: the per-cell codes are left unnamed.**
- Unmixing code 4, seasonal break code 8 and HANTS code 8 mean "iteration limit". Remapping, temporal
  and registration codes are per-row or per-result words.
- This matches contract section 9 ("per-row status words document every bit") and the earlier
  precedent of documented bits in point cloud and the Kalman filter.
- It does not match the region-statistics precedent (named scalar ports beside the header words), or
  roadmap row STD-04, which lists "registration, temporal, unmixing and remapping status" as
  Remaining. The roadmap row must keep them. A scalar `iterationLimitReached` beside the per-cell word
  for unmixing and seasonal break would follow contract section 4.

**m9. A verification claim in the algebra report is wrong.**
- The report says the `*.parity.spec.ts` files "are not picked up by the projects I ran". The
  headless project includes `modules/**/*.spec.{ts,js}`, so they are. They were simply not run.
- I ran them: `npx vitest run --project headless` on `raster-algebra/`, `raster-patches/`,
  `fractional-cell-size-mode.spec.ts` and the two fill expansion specs gave 20 files and 319 tests,
  all passing, parity specs included.

**m10. `conditionResult` is defensible.**
- The output is 1 for true and 0 for false or nodata, and `burn-severity.compute.ts` feeds it to the
  next stage as a condition raster. `validity` would claim the opposite, and `mask` is the contract's
  name for input filters.
- It still mixes "false" and "nodata"; consumers who need to tell them apart read `values`, which is
  NaN for nodata. The catalog's current description "Validity flag per cell" (line 583) is wrong under
  either name.

**m11. Checked and not a defect.**
- The `constantB` sentinel change in `GPURasterArithmetic` and `GPURasterConditional` appears in the
  diff, but it is from an earlier session and recorded in `behaviour-changes.md:259` (raster-imagery-01).
- The `parameters` prop on the distribution and categorical classes is likewise from an earlier
  session (`behaviour-changes.md:252`).
- `GPUNeighborSearch` cross-join consumers in the library: `GPULocalKriging` packs its support
  centers into the targets before deriving bounds, so its queries were always inside the bounds.
  Island attachment derives its bounds from the same rows it queries. No library result changes.
- Raster algebra kernels:
  - The `mask` is gated into `isDefined` in arithmetic, cell statistics and overlay.
  - `variety` uses `select(0u, variety, isDefined)`. `counts` uses `select(0u, count, isKept)`, which
    keeps the earlier rule that counts ignore the minimum count and the propagation policy.
  - The overlay `scoreRange` excludes masked cells, because they score NaN and `isScoreDefined` is
    false.
- Patch metrics fixed-point arithmetic:
  - A run spans at most 16 columns (`PATCH_SEGMENT_LENGTH`), so each u32 contribution stays below 2^25.
  - The carry detection `previous + contribution < previous` is correct.
  - Low and high words are read only in a later dispatch.
- The `cellSizeMode` formula is the same as `getRasterGroundCellSizeWGSL` (ground scale at the row
  center). The docs say "evaluated at the row center".
- `cellSizeMode` together with `cellAreas` is rejected with the contributor ID first.

### Other checks that passed
- `GPUNeighborSearch` outside-bounds fix. My adversarial spec:
  - Scene: 21×21 integer lattice with every point duplicated, bounds [0, 0, 20, 20]; integer queries
    1, 3, 17, 60 and 300 beyond every side and corner, plus 40 random queries in [-90, 110]².
  - Cases: kNN k = 1, 5, 16; kNN k = 9 with radius 20; radius 3 and radius 25.
  - Variants: grid and BVH, `gridSize` [4, 4], [16, 16], [64, 2] and [1, 1].
  - Result: all 8 configurations equal the f32 brute-force oracle, with `invalidCount` 0 and no
    overflow.
- A great-circle cross-join query outside the bounds is still empty and counted (`neighborCounts`
  [1, 0], `invalidCount` 1). This is documented in the props, the class TSDoc and
  `operators-weights.md:333-337`. No spec in the repository covered it; mine did and passed. Fix: add
  that case to `gpu-neighbor-search-outside-queries.spec.ts`.
- The time axis is covered by the agent's new case in `point-neighbor-grid-unification.spec.ts`. The
  bucket range clamps through `getTimeBucket`, so the visited range stays a superset.
- The imagery renames introduce no WGSL binding mismatches. A grep for `validity[`, `validityOffset`
  and `name: 'validity'` in the renamed directories finds only `GPURasterEmbeddingDequantize`'s
  output.
- `npx vitest run --project headless` on `raster-time-series`, `change-detection`, `remote-sensing`
  and `raster-embeddings`: 24 files, 286 tests passed, including `rts04-cold-pyxccd.spec.ts` and
  `rts04-change-monitoring.spec.ts`. Their fixtures are older than the imagery stream:
  `rts04-cold-pyxccd*.json` 08:32, `rts04-change-monitoring-numpy.json` 07:19, both before 09:10.
- `npx vitest run --project node` on `gpu-raster`, `geospatial-expansion`, `kernel-audit`,
  `neighbor-search`, `spatial-weights` and `analysis-vocabulary`: 126 files, 3,386 tests passed. The
  first attempt had about 20 failures, in GPUSpatialPanelRegression, GPUConstrainedDelaunayTriangulation,
  GPUSubdividePolygons, dasymetric, GPUDotDensity and two GPURasterFractionalCoverage audit cases; the
  rerun passed everything, so concurrent edits or load.
- `doc-parameters.node.spec.ts` passes now.
- Renamed names in `modules/lenses` and `modules/kepler`: none left on the renamed classes. The hits
  for `outputValidity`, `unresolvedCount` and `status` there belong to other classes or to the recipe's
  own output.
- No `GPUGroupConcaveHull` caller in `modules/lenses`, `modules/kepler`, `website/src` or `examples`
  passes a `groupIds` longer than its positions. Each one uploads one label per point: site-meshes
  uses `data.groupIds.length` for both views; the website run uploads `points.labels`; the lens
  example passes 6 and 6; the recipe uses `labels` of `pointCount`.
- `node scripts/check-upstream-first.mjs`: no violations.
- `node scripts/check-contributor-contract.mjs`: no new findings. The root stub
  `raster-analysis/raster-cell-area-scale.ts` is gone.
- `node scripts/typecheck-status.mjs 'modules/'`: 0 errors (cycle 9:45:36 AM).
- SPDX headers are present on every new file I checked: raster-cell-area-scale.ts and the 10 new
  specs.
- The kernel-audit cases are updated for every class whose props changed.

## 2. Consumer work list for `website/src` and `examples/` (both raster reports)

Each line is `file:line: old -> new [class]`. A `validity:` key not listed here belongs to a class
that was not renamed, or is an output flag (output `validity` stays).

### website/src/gpu/runs/entries
- `raster-composites-climate/best-available-pixel.ts:53: validity: validity.view -> mask: validity.view [GPURasterBestAvailablePixel]`
- `raster-composites-climate/medoid-composite.ts:62: validity: validity.view -> mask: validity.view [GPURasterMedoidComposite]`
- `raster-composites-climate/geometric-median-composite.ts:76: validity: validity.view -> mask: validity.view [GPURasterGeometricMedianComposite]`
- `raster-bands-indices/temporal-composite.ts:36: validity: validity.view -> mask: validity.view [GPURasterTemporal]`
- `raster-bands-indices/region-adjacency-graph.ts:82: status: status.view -> status: {iterationLimitReached: status.view} [GPURasterRegionGraph output]`
- `raster-bands-indices/fill-nodata.ts:50: unresolvedCount: unresolvedCount.view -> status: {invalidCount: unresolvedCount.view} [GPURasterFillNoData output]` (line 48 `validity` is the fill output and stays)
- `raster-embeddings-flow/raster-embedding-covariance.ts:61: validity: validity.view -> mask: validity.view [GPURasterEmbeddingCovariance]`
- `raster-embeddings-flow/raster-embedding-pyramid-level.ts:43: validity: validity.view -> mask: validity.view [GPURasterEmbeddingPyramidLevel input]`

### website/src/catalog/entries (catalog parameter, output and snippet text)
- `raster-bands-indices.ts:583: makeOutput('outputMask', UINT, 'Validity flag per cell (optional).') -> makeOutput('conditionResult', UINT, 'Condition result per cell: 1 true, 0 false or nodata (optional).') [GPURasterConditional]`
- `raster-bands-indices.ts:689: makeOutput('outputValidity', ...) -> makeOutput('validity', ...) [GPUWeightedOverlay]`
- `raster-bands-indices.ts:1633: makeOutput('unresolvedCount', ...) -> makeOutput('status.invalidCount', UINT, 'Gaps no donor reached (optional).') [GPURasterFillNoData]`
- `raster-bands-indices.ts:1646: unresolvedCount: view('unresolved', 'uint32', 1) -> status: {invalidCount: view('unresolved', 'uint32', 1)} [GPURasterFillNoData snippet]`
- `raster-bands-indices.ts:1674: makeParam('validity', ...) -> makeParam('mask', ...) [GPURasterPatchMetrics]`
- `raster-bands-indices.ts:1725: makeParam('validity', ...) -> makeParam('mask', ...) [GPURasterSieve]`
- `raster-bands-indices.ts:1869: makeOutput('status', UINT, 'Merge status word.') -> makeOutput('status.iterationLimitReached', UINT, '1 when a further round would change the mapping.') [GPURasterRegionGraph]`
- `raster-bands-indices.ts:2197: makeParam('validity', ...) -> makeParam('mask', ...) [GPURasterConservativeRemapping]`
- `raster-bands-indices.ts:2310: makeParam('validity', ...) -> makeParam('mask', ...) [GPURasterSpectralUnmixing]`
- `raster-bands-indices.ts:2425: makeParam('validity', ...) -> makeParam('mask', ...) [GPURasterSeasonalBreak]`
- `raster-bands-indices.ts:2475: makeParam('validity', ...) -> makeParam('mask', ...) [GPURasterTemporal]`
- `raster-bands-indices.ts:2500: validity: view('clear', ...) -> mask: view('clear', ...) [GPURasterTemporal snippet]`
- `raster-composites-climate.ts:117 [GPURasterBestAvailablePixel], :178 [GPURasterMedoidComposite], :231 [GPURasterGeometricMedianComposite], :351 [GPURasterClimateRunAccumulator], :436 [GPURasterStandardizedIndexCalibration], :509 [GPURasterStandardizedIndex], :573 [GPURasterSnapshotCovariance], :629 [GPURasterEOFProjection]: makeParam('validity', ...) -> makeParam('mask', ...)`
- `raster-composites-climate.ts:192 [Medoid snippet], :255 [GeometricMedian snippet]: validity: view('qa', ...) -> mask: view('qa', ...)`
- `raster-embeddings-flow.ts:83: makeParam('validity', ...) -> makeParam('mask', ...)` (shared `PIXEL_PARAMS`, used by Similarity :152, NearestPixels :201, KMeans :291, LinearProbe :397, Covariance :447, Projection :486; all renamed)
- `raster-embeddings-flow.ts:250: 'beforeValidity' -> 'beforeMask'; :251: 'afterValidity' -> 'afterMask' [GPURasterEmbeddingChange]`
- `raster-embeddings-flow.ts:464: validity: view('valid', ...) -> mask: view('valid', ...) [GPURasterEmbeddingCovariance snippet]`
- `raster-embeddings-flow.ts:599: makeParam('validity', ...) -> makeParam('mask', ...) [GPURasterEmbeddingPyramidLevel input]` (`:603` output `validity` stays)
- `raster-time-series.ts:92 [BFASTMonitor], :146 [COLDMonitor], :207 [HANTS], :257 [WhittakerSmoother], :310 [HarmonicRegression], :423 [RollingMean], :478 [TemporalResample], :534 [StreamingAccumulator], :588 [LaggedCorrelation]: makeParam('validity', ...) -> makeParam('mask', ...)`
- New props to add: `mask` on the arithmetic, cell statistics and weighted overlay entries; `cellSizeMode` on the fractional categorical, fractional distribution and patch metrics entries; optionally `observationMask` on `GPUChangeDetection` (`raster-bands-indices.ts:2348`, which lists only `mask` today). Then regenerate `website/src/catalog/generated/inventory.json` after the build (line 528 still has `outputValidity` for `GPUWeightedOverlay`).

### examples/
- `deck/spatial-analysis-explorer/modes/patches-mode.ts:195: validity: labelValidity -> mask: labelValidity [GPURasterPatchMetrics :201 and GPURasterSieve :219, through a spread: silent]`
- `deck/spatial-analysis-showcase/src/scenes/raster/burn-severity.compute.ts:336: outputMask: -> conditionResult: [GPURasterConditional]`
- `.../burn-severity.compute.ts:358: outputMask: -> conditionResult: [GPURasterConditional]`
- `.../burn-severity.compute.ts:370: outputMask: -> conditionResult: [GPURasterConditional]`
- `.../burn-severity.compute.ts:493: validity: labelValidity -> mask: labelValidity [GPURasterPatchMetrics :502 and GPURasterSieve :521, through a spread: silent]`
- `.../burn-severity.scene.ts:623: outputMask: cloudMask -> conditionResult: cloudMask [snippet text]`
- `.../burn-severity.scene.ts:627: outputMask: burnedMask -> conditionResult: burnedMask [snippet text]`
- `.../habitat-connectivity.compute.ts:303: validity: labelValidityView -> mask: labelValidityView [GPURasterPatchMetrics]`
- `.../spectral-texture.compute.ts:563: validity: wordView('texture-validity', ...) -> mask: wordView('texture-validity', ...) [GPURasterTexture]`
- `.../pixel-boundaries.compute.ts:660: validity: prepareView(...) -> mask: prepareView(...) [GPURasterConservativeRemapping]`
- `.../pixel-boundaries.compute.ts:748: validity: view(...) -> mask: view(...) [GPURasterConservativeRemapping]`
- `.../grid-alignment.compute.ts:361: referenceValidity: -> referenceMask: [GPURasterRegistration, through a spread: silent]`
- `.../grid-alignment.compute.ts:367: movingValidity: -> movingMask: [GPURasterRegistration, through a spread: silent]`
- `.../tiled-observation-topology.ts:539: status: regionStatusView -> status: {iterationLimitReached: regionStatusView} [addRasterRegionGraphTiles output]`
- `.../fire-scar-topology.compute.ts:724: status: views.classStatus -> status: {iterationLimitReached: views.classStatus} [GPURasterRegionGraph]`
- `.../fire-scar-topology.compute.ts:1065: status: focusSourceViews.regionStatus -> status: {iterationLimitReached: focusSourceViews.regionStatus} [GPURasterRegionGraph]`
- `.../missing-observations.compute.ts:427: unresolvedCount: view(`${key}-unresolved`, output.unresolved, 'uint32', 1) -> status: {invalidCount: view(`${key}-unresolved`, output.unresolved, 'uint32', 1)} [GPURasterFillNoData]` (`:425` output `validity` stays; `:450` is the recipe's own output and stays)

Classes that appear in `website/src` and `examples` but need no change because they pass none of
the renamed keys:
- `GPURasterTemporal` in `composite-evidence.compute.ts` and `ocean-seasonality.compute.ts`.
- `GPUChangeDetection` in `vegetation-trends.compute.ts` (passes `mask`) and `space-time-mode.ts`.
- `GPUWeightedOverlay` in `site-suitability.*` and `suitability-mode.ts`.
- `GPURasterConditional` in `site-suitability.compute.ts` and `suitability-mode.ts`.
- The website's glcm-texture, conservative-remapping, raster-registration, seasonal-break, unmixing,
  time-series, KMeans and dequantize runs.

## 3. Not checked
- Lens records and twins: `docs/lens-records` still name `unresolvedCount` (`gpu-raster.raster-fill-no-data.json`,
  the surface-preparation twins) and the old region-graph and imagery bindings. Both agents left
  regeneration to LN-REC2.
- Lens records of `huff-trade-areas` and `catchment-accessibility`: they are cross joins with
  caller-given bounds, and I did not check whether any of their queries lie outside the bounds. If any
  do, the records move.
- A mutation run of the island rescan (forbidden to edit source). I could not reproduce the agent's
  claim that disabling the rescan fails only the overflow case.
- Website pictures and examples at run time: they compile against `dist`, which has not been rebuilt.
- `GPUSegmentIntersection` alone with padded positions (the padding report lists it as open).
- `spatial-regression/**`, `polygon-triangulation/**`, `ring-assembly/**` and the `*-operation.ts` files
  (out of scope). The new `gpu-raster-region-graph-operation.ts` already uses
  `status: {iterationLimitReached}`.
