# Library review: dataframe analytics and the shared core (2026-11-09)

Domain: `modules/analysis/src/dataframe-analysis`, `analysis-core` (`contracts/`, `kernels/`,
`graph-support/`, loose root files; not `polygon-triangulation/` or `ring-assembly/`), `testing/`,
the nine entry barrels and `modules/analysis/package.json` exports. Paths are relative to
`modules/analysis/src` unless they start with `node_modules/` or `dev-docs/`. The earlier
architecture review (`dev-docs/research/spatiotemporal-review-2026-11-09/code-architecture.md`,
"the architecture review" below) already covers the contributor contract, the missing temporal
context, unused capacity planners and fixtures; this report cites it and does not repeat it.

## 1. Scope and method

| Area | Files | Lines |
| --- | --- | --- |
| `dataframe-analysis` | 77 | 24,134 (working tree; `gpu-space-time-variogram.ts` and `temporal-reduction-word-division.ts` are new and uncommitted) |
| `analysis-core` in scope | 44 | about 10,900 |
| `geospatial-analysis` | 4 | 499 |
| `testing` | 3 | 374 |

Method. Usage counts came from a session script that, for every exported
symbol of every in-scope `analysis-core` file, counts files outside `analysis-core` that mention the
name and import the defining file (or `contracts/index`). Exported symbol sets per entry point came
from the TypeScript checker (`getExportsOfModule` on each barrel).
WGSL duplication counts are `grep` over `fn name(` and bit patterns. Published `@luma.gl/graph`
exports were read from `node_modules/@luma.gl/graph/package.json` and the `.d.ts` indexes.

Read in full: `analysis-core/index.ts`, `contracts/index.ts`, `kernels/gpu-contributor-types.ts`,
`cell-key-wgsl.ts`, `kernels/wgsl-common.ts`, `geospatial-analysis/projection-spatial-context.ts`,
`dataframe-analysis/pair-statistics/gpu-cross-variogram.ts`, all nine barrels and the ten
`dataframe-analysis/*/index.ts`. Read in part (headers, props, class TSDoc, the kernels named in
findings): `kernels/wgsl-kernel-nodes.ts` (230-570), `kernels/gpu-contributor-utils.ts` (1-230),
`kernels/fixed-point-accumulator.ts`, `kernels/sorted-segment-offsets.ts`, `kernels/philox-random.ts`,
`graph-support/*`, `gpu-group-statistics.ts` (1-200), `group-statistics-moments.ts` (1-120, 160-372),
`group-statistics-order.ts`, `group-statistics-sort.ts` (headers), `gpu-key-join.ts` (1-240, 520-570,
770-828), `gpu-column-profile.ts`, `column-profile-kernels.ts` (20-240, 395-430, 795-830),
`gpu-class-breaks.ts` (1-200), `exact-natural-breaks-nodes.ts` (180-297), `gpu-column-quantiles.ts`,
`gpu-color-scale.ts`, `gpu-bivariate-classification.ts`, `gpu-scalar-scale.ts`,
`color-scale-wgsl.ts`, `scalar-scale-wgsl.ts`, `gpu-composite-score.ts`, `gpu-inequality.ts`,
`gpu-variable-standardization.ts` (1-120), `gpu-temporal-reduction.ts`, `gpu-time-window-filter.ts`,
`gpu-calendar-buckets.ts` (headers), `pair-statistics-grid.ts` (30-200, 420-450), `pair-histogram.ts`
(headers), `gpu-variogram.ts`, `gpu-spatial-correlogram.ts`, `gpu-ripley.ts`, `gpu-ripley-envelope.ts`,
`gpu-ripley-distance-functions.ts`, `gpu-point-pattern-indices.ts`, `gpu-space-time-variogram.ts`,
`gpu-marked-point-statistics.ts`, `gpu-variogram-model-fit.ts` (headers and the kernels named),
`geospatial-analysis/gpu-point-buffer.ts`, `gpu-datum-grid-transform.ts` (props and validation).
Not opened: `natural-breaks-nodes.ts` and `monotone-natural-breaks-nodes.ts` beyond grep, the
`*-parameters.ts` packers beyond grep, `time-zone-transitions.ts`, `time-window-sorted-nodes.ts`,
`time-window-classify-node.ts`, `contracts/spatial-context.ts`, `partitioning.ts`,
`capacity-planning.ts`, `local-frame.ts`, `spatial-frame*.ts`, `network-topology.ts`, `raster-grid.ts`,
`spherical-cell-area.ts`, `load-balanced-for-each.ts`, `polygon-edge-slabs.ts`,
`parallel-merge-tree.ts`, `study-window-wgsl.ts` (grep only).

## 2. Contributor table

Abbreviations: P = parallelism, Ctx = context, `pv` = caller `parameters` view written by a
`getGPU*ParameterValues` packer (per frame, no recompile). "Helpers" = `validateCapacity` /
`validatePackedView` and `throw new Error(message)`.

| Symbol | File (lines) | Computes (reference) | P | Work | Iteration | Hard limits | Params | Inputs | Ctx | Status | Validation | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `GPUGroupStatistics` | group-statistics/gpu-group-statistics.ts (542 + 1,450 helpers) | group-by count, exact sums, mean, min, max, var, sd, skew, kurt, median, percentiles, mode, unique, z (pandas `groupby.agg`) | row + cooperative (one workgroup per group for moments) | O(n log) LSD sort by key; moments O(n) but skew-bound | fixed passes | 4 value columns (`:34`), groups 8,388,608 (binding, inherent) | `percentiles` view; `sumScale` baked | packed | none (correct) | `GPUCardinalityPort & GPUOverflowStatusPort` | helpers | align (moments dispatch, see 3.1) |
| `GPUKeyJoin` | key-join/gpu-key-join.ts (828) | left/inner attribute join, first-match gather, 1:n count/sum/mean/min/max (pandas `merge` + agg) | row | sort right O(m), binary search per left row O(n log m) | none | 3 gathers per kernel (`:35`, batching, not a cap) | none; `sumScale` baked | packed | none | `GPUCompactOutput` for `inner` | helpers | align (private compaction, 3.6) |
| `GPUColumnProfile` | column-profile/gpu-column-profile.ts (311 + 1,014) | count, nulls, min, max, moments, histogram, HyperLogLog distinct, top-k categories (pandas `describe`, `value_counts`) | row + cooperative | O(n) per column; top-k O(k · categories / 256) in one workgroup | fixed per column (GPU cursor) | 2^24 rows (`:29`, `:186`, design: f32 counts); 16 columns; 2^20 bins | `pv` (domains) | packed | none | slot array `statistics` | helpers | align |
| `GPUColumnQuantiles` | column-classification/gpu-column-quantiles.ts (170 + 495) | exact quantiles, percentile filter (numpy `quantile`) | row | four 8-bit radix-select passes, O(n) each | fixed 4 passes | 2^24 rows (`:26`, `:109`, design); 64 quantiles | `pv` | packed | none | `validCount` scalar | helpers | align |
| `GPUClassBreaks` | column-classification/gpu-class-breaks.ts (1,038 + 2,700 helpers) | equal interval, quantile, std-mean, head/tail, box plot, max breaks, Fisher-Jenks (mapclassify) | row; many one-thread finishers | O(n) per pass; exact Jenks O(k n²) | ≤8 refinement rounds, unrolled with GPU gates | 2^24 rows (`:54`, `:264`, design); 64 classes; exact Jenks 8,192 rows, monotone 262,144 (design) | `pv` (method, k, edges) | packed | none | `classCount` scalar | helpers | align |
| `GPUColorScale` | column-classification/gpu-color-scale.ts (488) | d3 scales to RGBA | row | O(n) | none | domain/palette capacity (topology) | `pv` | packed | none | per-row no-class sentinel | helpers | conforming |
| `GPUBivariateClassification` | column-classification/gpu-bivariate-classification.ts (371) | bivariate choropleth, value-by-alpha | row | O(n log k) | none | 16 classes per axis (`:28`) | `pv` | packed | none | sentinel | helpers | conforming |
| `GPUScalarScale` | column-classification/gpu-scalar-scale.ts (253) | d3 linear/sqrt/log/identity/point to a scalar | row | O(n); point mode O(capacity) per row | none | 256 domain entries | `pv` | packed | none | `output.validity` | 4 bare `assert` (`:99-119`) | merge (transforms with `GPUColorScale`) |
| `GPUCompositeScore` | composite-indicators/gpu-composite-score.ts (850) | min-max/z/rank scaling, weighted sum, geometric mean, PCA (CARTO, ArcGIS index) | row + tile tree; PCA one thread | O(n d²); rank: one `GPUSort` per column | fixed 64-step power iteration | 16 indicators | `pv` (weights live) | packed | none | none | helpers | align |
| `GPUInequality` | composite-indicators/gpu-inequality.ts (528) | Gini, Theil T/L, Atkinson, Hoover, Palma, Lorenz, Theil decomposition (PySAL `inequality`) | serial per zone (`:160`) | O(n log n) sort; then one thread walks each zone | none | 1,025 Lorenz knots | `pv` | packed | none | none | helpers | rework (3.2) |
| `GPUVariableStandardization` | variable-standardization/gpu-variable-standardization.ts (357) | GeoDa transforms: demean, z, MAD, range | row via `GPUReduction` | O(n) | none | 2^24 rows (`:86`) | baked method | chunked | none | none | helpers | merge (with composite scalers, group z) |
| `GPUTemporalReduction` | temporal-reduction/gpu-temporal-reduction.ts (697) | per (cell, bucket) count, min, max, first, last (resample downsample) | row; sorted mode indirect over range | O(n) or O(range) | none | 2^20 float buckets (`:54`, inherent f32) | `pv` (origin, width) | packed | none | `GPUCompactOutput` | helpers | conforming |
| `GPUTimeWindowFilter` | time-window-filter/gpu-time-window-filter.ts (557 + 857) | instant or interval window filter, fade, clip, per-track counts, indirect draw | row; sorted mode O(log n) | O(n) or O(log n) | none | none found | `pv` (window) | chunked | none | `GPUCompactOutput` | helpers | conforming |
| `GPUCalendarBuckets` | calendar-buckets/gpu-calendar-buckets.ts (590 + 203) | year…quarter, ISO week, hour×weekday matrix, tz table (pandas `dt`) | row | O(n log transitions) | none | 10^9 days | `pv` (offset, first weekday) | packed | none | per-row sentinels | helpers | conforming |
| `GPUVariogram` | pair-statistics/gpu-variogram.ts (540) | empirical semivariogram, directional, Cressie-Hawkins (gstat) | row over 3x3 cells | candidate pairs, indexed (private grid) | none | rows ≈ 11.2 M / 2^24 − 1 (`pair-statistics-grid.ts:53-60`) | `pv` | packed | spatialContext (planar) | `GPUOverflowStatusPort` | helpers | conforming |
| `GPUSpaceTimeVariogram` | pair-statistics/gpu-space-time-variogram.ts (383, new) | space-time and cross semivariogram (gstat `variogramST`) | row | indexed pairs | none | as above | `pv` | packed | spatialContext | per-bin codes incl. overflow (`:374`) | helpers | conforming |
| `GPUCrossVariogram` | pair-statistics/gpu-cross-variogram.ts (204) | cross and space-time semivariance | row per (row, bin) | O(n² · bins), brute force | none | 16,384 rows, 128 bins (`:75-77`, design) | `pv` | packed | spatialContext | per-bin codes 0/1/2 | inline throw, one message for all shape errors | merge into `GPUSpaceTimeVariogram` (3.3) |
| `GPUSpatialCorrelogram` | pair-statistics/gpu-spatial-correlogram.ts (744) | Moran's I per distance band (esda) | row | indexed pairs | none | 64 bands | `pv` | packed | spatialContext | `GPUOverflowStatusPort` | helpers | conforming |
| `GPURipley` | pair-statistics/gpu-ripley.ts (522) | K, L, L−r, pcf; none/border/isotropic (spatstat `Kest`) | row | indexed pairs | none | 256 radii | `pv` | packed | spatialContext, study window | none | helpers | conforming |
| `GPURipleyEnvelope` | pair-statistics/gpu-ripley-envelope.ts (766) | CSR pointwise and global envelopes (spatstat `envelope`) | row per simulation | S × indexed pairs | unrolled per simulation (`:430`) | 199 simulations (`:60`, node budget) | `pv` (seed) | packed | spatialContext, window | none | helpers | align; overlaps `GPUPointPatternEnvelope` |
| `GPURipleyDistanceFunctions` | pair-statistics/gpu-ripley-distance-functions.ts (755) | G, F, J with rs/km/han (spatstat) | row | indexed NN | none | 256 radii, 2^24 references | `pv` | packed | spatialContext, window | none | helpers | conforming |
| `GPUPointPatternIndices` | pair-statistics/gpu-point-pattern-indices.ts (1,036) | Clark-Evans, quadrat dispersion (pointpats) | row | expanding ring search | none | 65,536 quadrats | `pv` | packed | spatialContext, window | none | helpers | conforming |
| `GPUMarkedPointStatistics` | pair-statistics/gpu-marked-point-statistics.ts (255) | cross K/L, inhomogeneous K, mark correlation | row per (row, radius); one-thread count | O(n² · radii), brute force; count kernel O(n) on 1 thread (`:153-160`) | none | 16,384 rows, 128 radii (`:87-90`, design) | `pv` | packed | spatialContext | none | inline throw | rework (3.3) |
| `GPUVariogramModelFit` | pair-statistics/gpu-variogram-model-fit.ts (119 + 238) | spherical/exp/Gaussian fit, Cressie weights (gstat `fit.variogram`) | cooperative over bins | O(models · ladder · bins) | ordered rounds | 3 models, 1,024 range steps | model list is topology | packed | none | per-model status codes | helpers | conforming |
| `fitVariogramModel`, `evaluateVariogramModel` | pair-statistics/variogram-model.ts (273) | CPU twin of the GPU fit | cpu | small | CPU loop | none | — | — | — | — | throw | move to tests (CPU oracle in the public API) |
| `GPUDatumGridTransform` | geospatial-analysis/gpu-datum-grid-transform.ts (274) | NTv2-style bilinear datum shift | row | O(n) | bounded inverse iteration | none | buffer | packed | none (own grid) | per-row bit flags | 11 bare `assert` (`:94-144`) | align |
| `GPUPointBuffer` | geospatial-analysis/gpu-point-buffer.ts (160) | circle ring vertices around points | row | O(n · v) | none | none | `pv` | packed | own `coordinateSystem` prop (`:49`), not spatialContext | none | 4 bare `assert` | align (use spatialContext; move) |
| core: `GPUParameterBuffer` | kernels/gpu-contributor-utils.ts | host-written parameter view | — | — | — | — | buffer | — | — | — | helpers | align (dataframe uses raw `pv`, 4.3) |
| core: `graph-support/*` | 3 files, 513 | Arrow table preparation, Int64 time upload | cpu | O(n) | — | — | — | — | — | — | throw | move (only kepler uses it) |
| core: `network-topology.ts` | 245 | CSR topology views | — | — | — | — | — | — | — | — | helpers | move to `gpu-network` (20 of 21 users) |
| core: `lon-lat-ring-wgsl.ts` | 160 | lon/lat ring checks | — | — | — | — | — | — | — | — | — | move to `gpu-spatial-analysis/lon-lat-polygons` (6 users, all spatial) |
| core: capacity planners, partitioning | 564 + 344 | planning helpers | cpu | — | — | — | — | — | — | — | throw | drop or adopt (architecture review, "Other core findings") |

Verdict counts for the 26 classes: conforming 12, align 9, merge 3, rework 2, move 0 (plus the
CPU variogram helpers and four core files proposed to move).

Counter-examples to the two-generation hypothesis. Small files are not always weak:
`GPUSpaceTimeVariogram` (383 lines, new) uses the indexed pair histogram. Large baseline files are
not always scalable: `GPUInequality` (528 lines, full TSDoc, deterministic) walks each zone on one
thread, and `GPUGroupStatistics` reduces each group's moments in one workgroup. The weak files in
this domain (`GPUCrossVariogram`, `GPUMarkedPointStatistics`, `GPUScalarScale`) share the
expansion traits: no `*Output` type, inline output object, one generic error message or bare
`assert`, a fixed 16,384-row cap. **CONFIRMED** by reading each.

## 3. GPU quality findings

**3.1 Group moments: one workgroup per group, dispatched over capacity.** CONFIRMED.
`group-statistics-moments.ts:354-372` dispatches `capacity * 256` invocations and each workgroup
strides over one group's sorted range. With the default capacity (8,388,608 groups) the dispatch is
2^31 invocations even when ten groups are occupied, and one group holding most rows runs on one
workgroup. Better: gate the dispatch on the GPU group count (`createWGSLActiveCountDispatch`, already
in core) and reduce moments with a segmented tree over the sorted rows (`sorted-segment-sums` or
`load-balanced-for-each` in core), keeping the fixed-order merge for determinism. Effort M.

**3.2 `GPUInequality` walks each zone serially.** CONFIRMED (`gpu-inequality.ts:160-163`, by its
own TSDoc). One thread per zone computes the Lorenz trapezoids, Theil and Atkinson sums in f32 in
sorted order. With one zone (the global Gini) the pass is O(n) on one thread and f32 sums lose
precision past millions of rows. Better: after the existing two `GPUSort` passes, a segmented
inclusive `GPUScan` of `w` and `w·x` gives the Lorenz coordinates for every row in parallel; the
trapezoid, Theil and Atkinson terms are then row-parallel and reduce with `sorted-segment-sums` or
the 96-bit `fixed-point-accumulator`. Effort M.

**3.3 Brute-force pair statistics beside an indexed path.** CONFIRMED.
`GPUCrossVariogram` runs `rows * bins` invocations that each scan every later row and skip pairs not
in their bin (`gpu-cross-variogram.ts:140`), so every pair is visited `bins` times, then a finishing
thread per bin loops over all rows (`:169`); limits 16,384 rows and 128 bins. The same file
re-exports `GPUSpaceTimeVariogram`, which computes the same signed cross term with times on the
indexed pair histogram. `GPUMarkedPointStatistics` is the same pattern (`:153-214`) and its
cross-type K duplicates what `GPURipley` says is "not implemented" (`gpu-ripley.ts:182`). Better: drop
`GPUCrossVariogram` (roadmap FX-11 asked for exactly this rebuild; the working tree has done it for
the space-time case), and rebuild marked/cross K as a `pair-histogram` channel set with a type pair
index, like `GPURipley`. Effort S and M.

**3.4 2^24-row ceiling across the dataframe family.** CONFIRMED. `GPUColumnProfile`,
`GPUColumnQuantiles`, `GPUClassBreaks`, `GPUVariableStandardization` and the pair statistics cap rows
at 2^24 because counts or ranks are held in f32 (`gpu-class-breaks.ts:53-54` says so). This is a
design limit, not a WebGPU one: kepler.gl and DuckDB-sized tables exceed 16.7 M rows. Better: keep
counts and ranks as `u32` (and 64-bit where summed), convert to f32 only in the final division.
Effort M (touches moment records in `class-breaks-kernels.ts` and `column-profile-kernels.ts`).

**3.5 Top-k categories in one workgroup.** CONFIRMED (`column-profile-kernels.ts:802-830`). One
256-lane workgroup makes `k` passes over all `categoryCount` codes (up to 33 M), O(k · C / 256)
serial steps. Better: sort `(count, code)` keys with `GPUSort` (or a radix select on counts, as
`GPUColumnQuantiles` already does) and take the first `k`. Effort S.

**3.6 Private compaction and histogram where published primitives exist.** CONFIRMED.
`GPUKeyJoin` compacts matched rows with `GPUScan` plus its own scatter kernel
(`gpu-key-join.ts:779-815`), while `GPUTemporalReduction` and `GPUTimeWindowFilter` use the
published `GPUVisibilityWorkflow`. `GPUColumnProfile` and `GPUClassBreaks` each implement their own
equal-width histogram (`getBinIndex` in `class-breaks-kernels.ts` and `column-profile-kernels.ts`,
plus `gpu-spatial-analysis/shared/bin-index-wgsl.ts`); published `GPUHistogram` (mask, vector input,
`'auto'` domain) is used by six analysis files, none in dataframe. Effort S each.

**3.7 Unrolled Monte Carlo.** CONFIRMED. `GPURipleyEnvelope` appends one node set per simulation
(`gpu-ripley-envelope.ts:430`) and caps at 199 to bound the graph (`:60`). The simulation index is
already a GPU counter word, so the pipelines are shared, but the node count grows linearly. Better:
add the simulation index as a dispatch dimension of the pair-histogram kernel (one dispatch for all
simulations, accumulators `simulation × radius`), which removes the cap up to the binding limit.
Effort M.

**3.8 Three fixed-point schemes in core.** CONFIRMED. `cell-key-wgsl.ts:34-68` (64-bit, caller
`sumScale` baked into WGSL, used by group statistics, key join, cell aggregation, cell table
compare), `kernels/fixed-point-accumulator.ts` (96-bit, data-derived power-of-two unit, used by point
density and remote sensing only), and the pair-histogram unit from the robust variance
(`pair-statistics-fixed-point-unit.ts`). The 64-bit add-with-carry pattern is written out in about
ten files (`group-statistics-moments.ts:216-219`, `gpu-key-join.ts`, `gpu-cell-aggregation`,
`gpu-map-matching`, `gpu-network-reachability`, space-time tests, pair histogram). The 96-bit
accumulator is the better general design (no caller scale to pick, no saturation at `2^62`). Converge
on it for sums whose scale is unknown; keep the pair-histogram unit where the term scale is known.
Effort M.

Not found in this domain: float atomics (none; every min/max uses ordered keys), readback inside a
contributor (none), recompiles from per-frame values (dataframe consistently uses parameter views).

## 4. Consistency findings

### 4.1 Package surface (question 4)

| Entry | Style | Symbols | Notes |
| --- | --- | --- | --- |
| `.` (root `index.ts`) | `export *` | 89 | Only `computation-record`. The package main entry exports no analysis at all. CONFIRMED (`index.ts:5`). |
| `computation-record` | named | 89 | consistent |
| `gpu-spatial-analysis` | `export *` of 89 sub-indexes | 1,823 | Re-exports all of `analysis-core/contracts` (`index.ts:8`) and `ring-assembly` (`:9`): 189 core symbols. 9 WGSL builders public (`GEODESIC_WGSL`, `getCholeskyWGSL`, ...). |
| `gpu-network` | `export *` + named core | 220 | Re-exports `GPUParameterBuffer`, `GPUCompactOutput`, `GPUUint32Rows` (`index.ts:31-38`) |
| `gpu-terrain` | `export *` + named core | 344 | Same three re-exports (`index.ts:22-28`) plus raster-grid types |
| `raster-analysis` | `export *`, some straight from files | 266 | `export * from './gpu-raster-skeleton'` etc. leaks `@internal` `getRasterMedialAxisNodes`, `getRasterWatershedForestNodes` |
| `pointcloud-analysis` | `export *` from every file | 37 | Leaks 5 `@internal` builders (`createPointCloudGridIndex`, `getPlaneSolveSource`, ...) |
| `dataframe-analysis` | `export *` of sub-indexes; sub-indexes name symbols | 224 | Only leak: `TIME_WORDS_WGSL` (`time-window-filter/index.ts:20`); also exports CPU `fitVariogramModel` |
| `geospatial-analysis` | `export *` from files | 11 | Leaks `@internal` `getGPUProjectionSpatialContext`, `getGPUProjectionFrameOrigin` |
| `analysis-core` | named | 16 | `createWGSLKernelNode`, `WGSLKernelBinding`, `WGSLKernelProps` (all tagged `@internal` in `wgsl-kernel-nodes.ts:237-326`), `getGraphViewChunks`, graph-support |

Symbols exported from more than one entry (checker result): `GPUParameterBuffer`,
`GPUParameterBufferProps`, `GPUParameterFormat`, `GPUCompactOutput`, `GPUUint32Rows`, each from
`gpu-spatial-analysis`, `gpu-network` and `gpu-terrain`. The `gpu-spatial-analysis/ontology` entry is
a subset of `gpu-spatial-analysis`. CONFIRMED.

Can a third party write a contributor from `analysis-core`? No. CONFIRMED. The entry gives the kernel
builder but not `validatePackedView`, `validateCapacity`, `validateGraphViewsBelongToGraph`,
`validateGraphOutputsDisjointFromInputs`, `createPublishNode`, `createFillNode`,
`getWGSLFloatLiteral`, the WGSL helpers (`IS_FINITE_BITS_WGSL`, `GET_ORDERED_KEY_WGSL`,
`PHILOX_WGSL`), or the contracts. A contributor that takes a `spatialContext` must import
`resolveGPUSpatialContext` from `gpu-spatial-analysis`, and `GPUCompactOutput` from one of three
domain entries. The README row "Shared contracts, kernel helpers and WGSL utilities" for
`analysis-core` does not match its 16 symbols. There is no public test harness either (`testing/` is
excluded from `files`). Recommended: make `analysis-core` the one home of the contract (contracts,
validators, publish/fill nodes, WGSL helpers, `GPUParameterBuffer`), drop the copies from the three
domain entries, and remove `@internal` from what it exports.

A related hazard: core `validatePackedView` (`kernels/gpu-contributor-utils.ts:205-217`) wraps and
shadows the published `validatePackedView` of `@luma.gl/graph/gpu-core` with the same name and
arguments but different behaviour on `GraphVectorView`; 315 imports use the core one and 14 import
the published one directly (including `time-window-filter`, `pointcloud-kernel-nodes.ts`,
`raster-band-utils.ts`). CONFIRMED. Rename the core wrapper (`validateSinglePackedView`).

`geospatial-analysis` is not a coherent package. CONFIRMED: a datum shift (`GPUDatumGridTransform`),
a point-ring generator with its own `coordinateSystem` prop (`GPUPointBuffer`), and two `@internal`
CRS-context helpers used only by kepler and tests. Recommended: move the datum transform and the
projection helpers next to `global-coordinates` (or into `contracts/spatial-frame`), move
`GPUPointBuffer` beside `polygon-buffer` with a `spatialContext`, and retire the entry point.

### 4.2 Contract adoption (question 1, second part)

Counts are source files outside `analysis-core` that use the type or function.

| Contract | Adoption | Comment |
| --- | --- | --- |
| Status ports | `GPUOverflowStatusPort` 20, `GPUCardinalityPort` 17, `validateGPUStatusPort` 14, `GPUBoundedResultStatusPort` 6, `GPUIterationStatusPort` 4, `GPURecipeStageStatusPort` 3, `GPUUncertaintyStatusPort` 2, `GPUInvalidStatusPort` 1 (kriging), `GPUCandidateStatusPort` 0, `GPUApproximationStatusPort` 0, `validateGPURecipeStatusPort` 0 (tests only) | Iteration, invalid, uncertainty, candidate and approximation ports are below five |
| model-result | 5 files, all `spatial-regression` | one family only |
| inference | `GPUInferenceMetadata` 16, `GPUMultipleTestingMethod` 13, `GPUInferenceAlternative` 9, `GPUSolverStatusPort` 6, `GPUPermutationMetadata` 1, `GPUPermutationStatisticAdapter` 1, `GPUInferenceResultPort` 0 (ontology only) | result port unused |
| parameter-schema | `defineGPUSpatialParameterSchema` 6, `packGPUSpatialParameterValues` 3, `validateGPUSpatialParameterSchema` 0 | all in `gpu-spatial-analysis`; dataframe uses its own packers (4.3) |
| partitioning | 1 file (`spatial-join/gpu-spatial-join-prepared.ts`); 12 of 14 exports unused | |
| capacity-planning | types in 3 files; `getGPUCapacityRecovery` 1; all nine planners 0 | as the architecture review found |
| study-window | `resolveGPUStudyWindow` 7 (5 dataframe pair statistics, 2 spatial) | |
| local-frame | `createGPULocalFrame` 35 | healthy |
| vertical-datum | 5, all terrain | |
| `contract-names` | 0 in source (tests and ontology) | |

Below five contributors: iteration, invalid, uncertainty, candidate and approximation status ports;
model-result (one family); `GPUInferenceResultPort`, `GPUPermutationMetadata`,
`GPUPermutationStatisticAdapter`; partitioning; capacity planners. CONFIRMED by grep (type-only uses
inside other contract types are not counted).

### 4.3 Dataframe conventions (question 5)

| Aspect | What the code does | One way? |
| --- | --- | --- |
| Masks | `mask?: GraphDataView<'uint32'>`, nonzero keeps (20 of 24 classes); `leftMask`/`rightMask` in key join; `validity` in `GPUScalarScale`; output masks are `outputMask` (quantiles, window filter) or `output.validity` (scalar scale) | nearly; rename `GPUScalarScale.validity` to `mask` and its output to `outputMask` |
| Nulls | f32 NaN is null; `0xffffffff` is a null key or category; `-2^31` for signed calendar fields; Arrow validity is expanded to `uint32` per row (architecture review) | yes |
| 64-bit keys | `uint32x2` `(low, high)` in group statistics and key join; all-ones key reserved | yes |
| Time | Int64 words (`uint32x2`), f32 relative, f32 high+low; calendar buckets now take a `unit` (working tree, roadmap FX-01) | three encodings; tracked as TC-01 |
| Wide values | values are f32 only (class breaks also converts `uint32`/`sint32`); Int64 and Decimal become f32 on the CPU under a Float64 policy (`graph-support/arrow-analytics-wide-numeric.ts`) | yes, but f32-only |
| Sums | exact 64-bit fixed point with caller `sumScale` (group, join); f32 tree sums (profile, composite, inequality); 96-bit accumulator nowhere in dataframe | no (3.8) |
| Variance convention | group statistics sample by default; composite z population; variable standardization sample; column profile reports both | no; name the convention in one prop (`variance: 'sample' \| 'population'`, as group statistics has) |
| Per-frame parameters | caller `parameters` view plus `getGPU*ParameterValues` packer (22 of 24 classes; not key join or variable standardization) | yes within dataframe, but a third mechanism beside `GPUParameterBuffer` (34 files) and `defineGPUSpatialParameterSchema` (6). Recommend: keep the packer pattern and make `GPUParameterBuffer` the documented host for those views |
| Input chunking | packed only, except `GPUTimeWindowFilter` and `GPUVariableStandardization` | packed is the norm; `GPUVariableStandardization` re-implements `getGraphViewChunks` privately (`getGraphVectorData`, `sliceView`) |
| Outputs | `*Output` type per class, except the six newest (`GPUCrossVariogram`, `GPUSpaceTimeVariogram`, `GPUMarkedPointStatistics`, `GPUVariogramModelFit`, `GPUScalarScale`, `GPUVariableStandardization`) | mostly |
| Statistic names | `minimums`/`maximums` (group, temporal) versus `minimum` in profile slots; `validCount` (quantiles) versus `count` (status) versus `classCount` (breaks) | no |
| Status | named ports only in group statistics, key join, temporal, window, variogram, correlogram; others write slot arrays (`statistics`, `globalSummary`, `principalComponentSummary`) or nothing | partial |
| Validation | helpers with messages everywhere except `GPUScalarScale` (bare `assert`) and the two dense pair contributors (one generic message) | yes |

Placement: the nine pair-statistics classes are spatial point-pattern and geostatistics operators
(they take `spatialContext` and study windows). They sit beside `GPUPointPatternEnvelope`,
`GPUInhomogeneousPointPattern` and kriging in `gpu-spatial-analysis`, and duplicate parts of them
(4.4, 5.4). Recommend moving `pair-statistics/` to `gpu-spatial-analysis/point-pattern` (or
`point-process`). SUSPECTED that no consumer depends on the entry path beyond lenses
(`modules/lenses/src/gpu-dataframe/computation-pack/operators/pair-statistics.ts` names the source
files).

### 4.4 Naming (question 6)

Package-wide checks (checker over all 3,020 exported symbols): all 392 exported classes have the
`GPU` prefix; 391 have a `*Props` type (`GPUSnapshotArena` does not); every lowercase exported
function starts with a verb. 33 classes live in files not named for them (multi-class files such as
`gpu-raster-spectral-indices.ts`, `gpu-additional-spatial-statistics.ts`, and `GPURasterMedialAxis` in
`gpu-raster-skeleton.ts`); all dataframe classes match their file. WGSL builders use three name
forms: 129 `get*WGSL` functions, 115 `*_WGSL` constants and 42 `get*Source` functions; only 45 files
are named `*-wgsl.ts` and 86 files export WGSL under other names. 43 exported types lack the `GPU`
prefix (`VariogramModel*`, `WGSLKernel*`, `ArrowAnalytics*`, `PointCloud*`, `SolarPosition*`).

Sample of 40 exported symbols (every 75th in alphabetical order); departures:

| Symbol | Departure |
| --- | --- |
| `GPULOSHResidualPermutationTest` | acronym, while the sibling is `GPULocalSpatialHeteroscedasticity` |
| `getMultipleTestingAdjustedPValues` | no `GPU` in a function name whose siblings are `getGPU*` |
| `GPUCrossVariogramProps` | inline `output`; no `GPUCrossVariogramOutput` |
| `GPUPartitionMemoryPlan` | exported, unused |
| `GPUSpatialWeightsTransposeOutput` | file `spatial-weights-transpose-pipeline.ts` lacks `gpu-` and the class name |
| `GPUShapeGeneratorSettings` | `*Settings` for packer input; elsewhere `*Options`, `*Parameters` |
| `flattenPointCloudAttribute` | `PointCloud` types and functions without `GPU` while the classes have it |
| `GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_PSEUDO_R_SQUARED_SPATIAL` | slot constants per summary field; acceptable but verbose |

The other 32 conform. In dataframe: Ripley output fields `k`, `l`, `g`, `f`, `j`, `lMinusR` and
`theilT`/`theilL` are single-letter domain names (AGENTS.md "never abbreviate"); acceptable as
statistical names but should be documented as the exception. `TIME_WORDS_WGSL` is the only public
WGSL constant without `GPU_`.

## 5. Duplication and shared-helper candidates

### 5.1 Use of core exports outside `analysis-core` (question 1)

Heavily used (20 or more files): `createWGSLKernelNode` 447, `validateGraphViewsBelongToGraph` 337,
`validatePackedView` 321, `validateGraphOutputsDisjointFromInputs` 255, `WGSLKernelBinding` 251,
`validateCapacity` 194, `GPUSpatialContext` 146, `resolveGPUSpatialContext` 117, `getWGSLFloatLiteral`
108, `createFillNode` 76, `GET_NAN_WGSL` 71, `IS_FINITE_VALUE_WGSL` 69, `IS_FINITE_BITS_WGSL` 57,
`validateMaximumCount` 56, `getStorageBindingElementLimit` 42, `createGPULocalFrame` 35,
`GPUPackedGeometryPort` 32, `createPublishNode` 31, `GPU_PLANAR_OR_GREAT_CIRCLE_SPATIAL_CONTEXT_SUPPORT`
30, `GPUCompactOutput` 30 + 20, `EXACT_ORIENTATION_WGSL` 28, `validateCompactOutput` 28,
`GPU_PLANAR_SPATIAL_CONTEXT_SUPPORT` 28, `createWGSLDispatchGate` 26, `GPURasterCellSizeMode` 25,
`IS_FINITE_FLOAT_WGSL` 25, `PHILOX_PURPOSES` 23, `validateGPUNetworkTopology` 20.

Single-directory users (candidates to move out of core or to promote): `network-topology.ts` (all
`gpu-network`, one terrain use), `lon-lat-ring-wgsl.ts`, `hash-uint32.ts`, `polygon-edge-slabs.ts`,
`exact-lattice-wgsl.ts` (one file), `wgsl-kernel-tiles.ts`, `gpu-prediction-utils.ts`
(`gpu-spatial-analysis` plus one dataframe use), `vertical-datum.ts` (terrain).

Exported with no source user outside core (tests only or none): `createChangedBindingResolver`,
`createCachedBindingResolver`, `getPhiloxUnitFloat`, `GPU_PREDICTION_STATUS`,
`getLoadBalancedTileCount`, `LOAD_BALANCED_*` constants, `FIXED_POINT_MAXIMUM_TERM_COUNT`,
`CELL_MAXIMUM_SCALED_VALUE`, `getRasterGroundScale`, `getRasterSphereRadius`,
`getLonLatRingWGSL`, `GPU_SPATIAL_ANALYSIS_CONTRACT_NAMES`, every capacity planner, 12 of 14
partitioning exports, `validateGPUSpatialParameterSchema`, `GPU_ELLIPSOIDAL_VERTICAL_DATUM`,
`GPU_SINUSOIDAL_KILOMETERS_PER_DEGREE`, `GPU_WGS84_INVERSE_FLATTENING`, and all `graph-support`
functions (used only by `modules/kepler` and `examples/deck/kepler-compute`). The full 378-row table was produced in the
reviewer's session scratchpad and is not kept; rerun the method in section 1 to regenerate it.

### 5.2 Kernel-building abstractions (question 3)

`createWGSLKernelNode` is the only node builder that owns a pipeline: no contributor calls
`new Kernel` or `createGPUComputeCommandNode` directly, except `computation-record/gpu-transfer-mask.ts:142`
and `gpu-contributor-utils.ts` itself. CONFIRMED. Thin layers on top of it:

| Abstraction | Defined in | Users | What it adds |
| --- | --- | --- | --- |
| `createWGSLKernelNode` | `kernels/wgsl-kernel-nodes.ts:326` | 447 files, all domains | the base |
| `createTiledWGSLKernelNodes` | `kernels/wgsl-kernel-tiles.ts:20` | 4 (dorling, neighbor search, interpolation) | tiling |
| `createRecipeKernelNode` | `gpu-spatial-analysis/recipes/expansion-recipe-utils.ts:35` | 26 files (recipes, landscape ecology, urban morphology) | ownership and alias validation |
| `getPointCloudKernelNodes` | `pointcloud-analysis/pointcloud-kernel-nodes.ts:43` | 10 point-cloud files | named `read*`/`write*` accessors, `safeNormal` |
| `rasterBinding`, `createRasterCountNodes` | `raster-analysis/remote-sensing/raster-kernel-utils.ts:43,121` | 12 and 4 (remote sensing, terrain processing) | binding type from format; workgroup counting |
| `terrainBinding` | `gpu-terrain/terrain-processing/terrain-processing-utils.ts` | 8 | same as `rasterBinding` |
| `readBinding`/`writeBinding`/`atomicBinding` | `group-statistics-common.ts`, `parallel-merge-tree.ts`, `hierarchical-density-tree.ts` | 3 copies | binding records |
| local `read`/`write`/`u32` lambdas | 36, 34 and 41 files | all domains | binding records and transient views |
| `getWorkgroupCounterKernelSource` | `gpu-network/network-workgroup-counters.ts:44` | 5 network files | WGSL source generator |
| specialized core node builders | `load-balanced-for-each`, `sorted-segment-sums`, `scan-total`, `fixed-point-accumulator`, `masked-weak-components`, `parallel-merge-tree`, `sorted-segment-offsets` | 1 to 10 each | algorithms, all on `createWGSLKernelNode` |

So there is one base and no parallel pipeline builder, but four domain wrappers that solve the same
two problems: a binding record from a view's format, and named row accessors. Add
`getReadBinding(name, view)`, `getWriteBinding`, `getAtomicBinding` (element type from the view
format) and a `getRowAccessorWGSL(binding, components)` to `kernels/wgsl-kernel-nodes.ts`, then let
`rasterBinding`, `terrainBinding`, the three `readBinding` copies and the 100-odd local lambdas use
them. `getPointCloudKernelNodes` would reduce to the accessor helper. Effort S to add, M to adopt.

### 5.3 Shared WGSL modules the core should have (question 2)

| Module | Exists today | Private copies | Proposal |
| --- | --- | --- | --- |
| finite and NaN tests | `wgsl-common.ts:17-60` has seven variants for two predicates: `getNaN`, `getQuietNaN`, `getQuietNaN(seed)`, `isNanBits`, `isFiniteBits`, `isFiniteValue`, `isFiniteFloat` (the comment says "same as ... under the name the older kernels use"); `gpu-spatial-analysis/shared/spatial-wgsl-point-helpers.ts:32-45` adds `isSpatialFinite`, `getSpatialNaN` | the exponent-mask test is inlined in 43 files (7 in column classification, 5 in network optimization); `0x7fc00000u` appears in 104 files; private `isFiniteBits` in `group-statistics-moments.ts:358`, `PAIR_STATISTICS_FLOAT_WGSL` (`pair-statistics-grid.ts:161-178`) | one `FLOAT_TESTS_WGSL` (`isFiniteValue`, `isNanValue`, `getNaN`, `getNaN(seed)`), aliases removed |
| ordered keys | `GET_ORDERED_KEY_WGSL`, `DECODE_ORDERED_KEY_WGSL`, `ORDERED_KEY_WGSL` in `cell-key-wgsl.ts` | the bit trick in 29 files (`getOrderedFloatKey` in pair statistics, group value key, column classification, terrain, raster zonal) | use the core pair everywhere |
| positions and local origin | `getReadPointWGSL` in `gpu-spatial-analysis/shared` (14 users); `createGPULocalFrame` is host-side only | `getPosition` in 12 files (census), 27 files with `getPosition`/`readPosition`-style functions, 59 files with a `fn x(row: u32) -> vec2f`, 116 files read `[offset + 2u * row]` inline; `localOrigin` handled separately in pointcloud, raster sampling, isolines | `positions-wgsl.ts` in core: `getReadPointWGSL` (2D, 3D, chunked), plus a local-origin subtract helper that takes the `GPULocalFrame` origin as parameter words |
| binary search | `sorted-segment-offsets.ts` (per-segment lower bound), published `GPUGallopingSearch` (node, not WGSL) | `lowerBound` in 6 files; 13 files with lower/upper bound, `lastAtMost`, `findTrack`; `gpu-key-join.ts:560`; calendar transitions | `binary-search-wgsl.ts`: `getLowerBoundWGSL(name, binding, keyType)` for `u32`, `f32` via ordered key, and `vec2u` 64-bit |
| 64-bit integer words | `getFixedPointWGSL` (`cellAddI64`), time words in `time-words.ts`, `temporal-reduction-word-division.ts` (new) | add-with-carry atomics in about 10 files; `subtract64` in `gpu-ripley.ts` and `terrain-summed-area-table.ts` | `uint64-wgsl.ts`: add, subtract, compare, atomic add with carry, divide by u32 |
| small dense solvers | `getCholeskyWGSL` (`spatial-regression/spatial-regression-solve.ts:28`, 19 users), `getEquilibratedSolveWGSL` (`spatial-interpolation/equilibrated-solve-wgsl.ts:19`), `getEquilibratedCholeskyWGSL` | `solveSystem` in 3 files; power iteration in `gpu-composite-score.ts:668-700` | move the Cholesky and equilibrated solver to `kernels/dense-solve-wgsl.ts`; add a Jacobi symmetric eigen solver for PCA |
| point-in-polygon | `study-window-wgsl.ts` (`isInsideStudyWindow`), `polygon-edge-slabs.ts` (slab query), `exact-geometry-wgsl.ts` (`exactProbeCrossing`, `exactSegmentsCross`) | `segmentsIntersect` in 3 files, `ringWinding` in 2, `insideWinding` in overlay | already in core in three forms; document which to use and route `spatial-join` and validity to them |
| sphere and ellipsoid distance | `GEODESIC_WGSL` (`gpu-spatial-analysis/geometry-measures/geodesic-wgsl.ts`, 29 users) and Vincenty in `geodesic-kernels.ts`; host radii in `contracts/spatial-context.ts` | `edgeLength`, `getDistance` in 3 files each | move `GEODESIC_WGSL` (and rhumb) to core; it is already the de facto shared module |
| random numbers | `PHILOX_WGSL` with a purpose registry (`philox-random.ts:37`), 15 users | `fn random` in `gpu-spatial-balanced-sample.ts:212`, `hashRow` in `gpu-kmeans.ts:728`, `hashPair` in `gpu-minimum-bounds.ts:573` | route these three through Philox purposes |
| masks | none | every contributor writes `mask[maskOffset + index] != 0u` and its own chunk handling | `getMaskTestWGSL(binding)` that handles absent masks and `GPUUint32Rows` chunks |
| erfc / normal | `ERFC_VALUE_WGSL` (15 users) and CPU `getComplementaryErrorFunction` in `gpu-spatial-analysis/shared` | none found | fine; move the CPU twin to core |

`gpu-spatial-analysis/shared/` is a second, unofficial core (bin index, erfc, point helpers,
triangle adjacency, and a re-export shim of core `hash-uint32`). CONFIRMED. Fold it into
`analysis-core/kernels`.

TypeScript helpers defined in several directories: `getInputs` (15 files) and `getOutputs` (14) in
network and spatial; `getOutputViews` (5, duplicating core `getGraphOutputViews`);
`getGraphVectorData`/`sliceView` in `gpu-variable-standardization.ts` (duplicating core
`getGraphViewChunks`); `getBindings` (3). CONFIRMED by grep.

### 5.4 Overlaps with published `@luma.gl/graph` and inside the domain

| Contributor | Published counterpart | Recommendation |
| --- | --- | --- |
| `GPUGroupStatistics` | `GPUGroupAggregation` (count, sum, min, max, mean; dense `uint32` keys; float32 values), `GPUDataFrame.groupBy` | build count/sum/min/max on the published op where keys are dense; propose upstream: 64-bit keys, exact sums, higher moments, percentiles, mode, unique |
| `GPUKeyJoin` | `GPUHashIndex`, `GPUHashJoin`, `GPUDataFrame` join (`inner`, `left`, `semi`, `anti`, unique right key) | keep sort-based probe (deterministic, 64-bit keys); propose 1:n aggregate join and 64-bit keys upstream |
| `GPUColumnProfile`, `GPUClassBreaks` histograms | `GPUHistogram` | use it |
| `GPUColumnQuantiles` | none (`GPUDataFrame` has no quantile) | good upstream candidate: exact radix-select quantiles as a GPU Core op |
| `GPUTimeWindowFilter` | `GPUCrossfilter` range dimension (float scalar, inclusive) | keep; propose Int64 and interval ranges upstream |
| `GPUTemporalReduction` | `GPUTimeSeriesEnvelope` (min, max, first, last per bin, one uniform series) | different shape (irregular, per cell); keep |
| `GPUKeyJoin` compaction | `GPUCompaction`, `GPUVisibilityWorkflow` | use them (3.6) |
| `GPUCrossVariogram` | `GPUSpaceTimeVariogram` | merge (3.3) |
| `GPUMarkedPointStatistics` | `GPURipley`, `GPUInhomogeneousPointPattern` | rebuild on pair histogram |
| `GPURipleyEnvelope` | `GPUPointPatternEnvelope` (point process, generic over curves) | one envelope engine |
| `GPUScalarScale` | `GPUColorScale` | one `scale-transform-wgsl` (linear, sqrt, pow, log with explicit sign rule, symlog) |
| `GPUVariableStandardization` | `GPUCompositeScore` scalers, `GPUGroupStatistics` z | one standardization contributor with `groupKeys?` and a variance convention |
| `fitVariogramModel` (CPU) | `GPUVariogramModelFit` | keep CPU as test oracle only |
| pair-statistics grid | `GPUGridIndex`; `neighbor-search-kernels.ts`, `spatial-clustering-kernels.ts` share `getCellRow`, `getCellColumn`, `readLattice`, `isPointValid` with it | the pair grid derives cells per frame from `maximumDistance` (a reason not to use `GPUGridIndex`); share one lattice WGSL with neighbor search and clustering (also roadmap PLAT-03) |

Nothing in dataframe uses `GPUDataFrame`, `GPUSQLContext`, `GPUCrossfilter`, `GPUUnique`,
`GPUHashJoin`, `GPUSegmentedSort` or `GPUTimeSeries`. CONFIRMED by grep.

## 6. Comprehensiveness

Ranked by how often a pandas, Polars, cuDF or DuckDB user reaches for it. "Tracked" items are
excluded except to name them.

| # | Missing | GPU-suitable | Reuses | Effort |
| --- | --- | --- | --- | --- |
| 1 | Window functions over sorted groups: cumulative sum/count/min/max, `shift`/`lag`/`lead`, `diff`, `pct_change`, `rank`/`dense_rank`/`row_number` per group | yes (segmented scan) | group-statistics sort layout, `GPUScan`, `sorted-segment-sums` | M |
| 2 | Rolling and time-based windows (`rolling(n)`, `rolling('15min')`, exponentially weighted mean) | yes (prefix sums plus binary search per row) | time words, binary-search module (5.3), `GPUScan` | M |
| 3 | Per-group correlation, covariance and simple regression (`corr`, `cov`, DuckDB `regr_slope`, `regr_r2`) | yes | group moment path with a second column | S |
| 4 | Multi-column group keys (composite keys) and more than four value columns | yes | key packing to `uint32x2` or hashing; existing sort | S |
| 5 | Weighted group statistics (weighted mean, variance, quantile) | yes | group moments; inequality already has weights | S |
| 6 | Join variants: n:m expanding inner join with pair output, semi/anti/outer, as-of join with tolerance (`merge_asof`), interval join | yes (count, scan, scatter; binary search for as-of) | key-join sort and probe, `GPUCompactOutput` | M; other operations tracked as KJ-01 |
| 7 | Resample fill and upsample: forward/back fill, linear fill on a regular grid, `asfreq` | yes | temporal reduction slots, segmented scan | S to M; sum and mean cubes tracked as TC-04 |
| 8 | Imputation and robust preprocessing: `SimpleImputer` (mean, median, constant), `RobustScaler` (median, IQR), `QuantileTransformer`, Box-Cox and Yeo-Johnson, `KBinsDiscretizer` output codes | yes | column quantiles, class breaks, standardization | S each |
| 9 | Pivot and crosstab (two keys to a dense matrix, with margins) | yes | group statistics dense mode with key `a * nb + b` | S (a helper and docs) |
| 10 | Approximate quantiles for very large or streamed columns (t-digest or KLL), and distinct count per group by HyperLogLog | yes | column profile HLL registers | M |
| 11 | DuckDB aggregates: `arg_min`/`arg_max` per group, `product`, `entropy`, `bool_and`/`bool_or`, `count_if` | yes | ordered-key atomics, group layout | S |
| 12 | mapclassify: Jenks-Caspall, pretty breaks, geometric interval, goodness-of-variance fit and TAI as outputs | yes (small k) | class breaks state | S |
| 13 | PySAL inequality: spatial Gini (Rey and Smith), spatial Theil decomposition with permutation inference, Wolfson polarization | yes | inequality sorts, spatial weights, Philox | M |
| 14 | Exact 64-bit or double-single value columns (sums and means of Float64 measures without the 2^24 cliff) | partly (no f64 in WGSL) | fixed-point accumulator, time-word arithmetic | M |
| 15 | kepler.gl and deck.gl aggregation glue: one contributor that bins (grid or hexagon), aggregates with `GPUGroupStatistics`, classifies and colours, with the percentile filter, as a recipe | yes | cell aggregation, group statistics, class breaks, color scale | S (recipe) |

Present and good: exact quantiles, choropleth classifiers (eight methods), d3-compatible colour and
scalar scales, bivariate maps, calendar decode with time-zone tables, time-window filtering with
indirect draw, temporal min/max/first/last cubes, column profiling with HyperLogLog, composite
indices, inequality indices, and the geostatistics in pair statistics.

Out of place in this domain: `pair-statistics/` (spatial point-pattern statistics; move to
`gpu-spatial-analysis`), the CPU `fitVariogramModel`/`evaluateVariogramModel` in the public API,
`TIME_WORDS_WGSL` as public API, and `graph-support` in `analysis-core` (Arrow ingestion used only by
kepler; it belongs in `dataframe-analysis` or, with `GPUAnalyticsDictionary`, upstream in
`@luma.gl/graph/arrow`). In core: `network-topology.ts` and `lon-lat-ring-wgsl.ts` (one domain
each). `geospatial-analysis` as an entry (4.1).

## 7. Top ten actions

| # | Action | Effort | Files |
| --- | --- | --- | --- |
| 1 | Make `analysis-core` the complete contributor SDK: export contracts, validators, publish/fill nodes, `GPUParameterBuffer`, WGSL helpers; stop re-exporting them from `gpu-spatial-analysis`, `gpu-network`, `gpu-terrain`; remove `@internal` from what is exported; rename the shadowing `validatePackedView` | M | `analysis-core/index.ts`, `gpu-spatial-analysis/index.ts:8,88-96`, `gpu-network/index.ts:31-38`, `gpu-terrain/index.ts:22-28`, `kernels/gpu-contributor-utils.ts:205`, 315 importers (mechanical) |
| 2 | One float-test and ordered-key WGSL module; delete the aliases and the 43 inline copies | S to M | `kernels/wgsl-common.ts`, `cell-key-wgsl.ts`, `pair-statistics-grid.ts:161`, `gpu-spatial-analysis/shared/spatial-wgsl-point-helpers.ts`, 43 + 29 files |
| 3 | Binding and accessor helpers in core; retire `rasterBinding`, `terrainBinding`, the three `readBinding` copies and local lambdas over time | S, then M | `kernels/wgsl-kernel-nodes.ts`, `raster-kernel-utils.ts`, `terrain-processing-utils.ts`, `group-statistics-common.ts`, `pointcloud-kernel-nodes.ts` |
| 4 | Remove the dense pair contributors: drop `GPUCrossVariogram` for `GPUSpaceTimeVariogram`; rebuild marked and cross K on the pair histogram | S and M | `gpu-cross-variogram.ts`, `gpu-marked-point-statistics.ts`, `pair-histogram.ts`, `gpu-ripley.ts:182` |
| 5 | Group statistics moments: GPU-gated dispatch and segmented reduction | M | `group-statistics-moments.ts:340-400` |
| 6 | `GPUInequality` on a segmented scan with fixed-point sums | M | `gpu-inequality.ts`, `inequality-wgsl.ts` |
| 7 | Lift the 2^24-row ceiling (integer counts and ranks) | M | `column-profile-kernels.ts`, `class-breaks-kernels.ts`, `column-quantiles-nodes.ts`, `gpu-variable-standardization.ts:86`, `pair-statistics-grid.ts:53-65` |
| 8 | Positions, binary-search, uint64 and dense-solve WGSL modules in core; move `GEODESIC_WGSL` and `gpu-spatial-analysis/shared` into core | M | `kernels/` (new files), `geometry-measures/geodesic-wgsl.ts`, `spatial-regression/spatial-regression-solve.ts`, `spatial-interpolation/equilibrated-solve-wgsl.ts` |
| 9 | Window functions and per-group correlation (section 6 rows 1 to 3) | M | new `dataframe-analysis/window-functions/`, `group-statistics-moments.ts` |
| 10 | Reorganize entries: move `pair-statistics` to `gpu-spatial-analysis`, retire `geospatial-analysis`, move `graph-support`, `network-topology.ts`, `lon-lat-ring-wgsl.ts` to their one domain; use published `GPUHistogram` and `GPUVisibilityWorkflow` in profile, breaks and key join | M | barrels, `package.json` exports, lenses source references, `gpu-key-join.ts:779-815`, `column-profile-kernels.ts` |
