# Cross-cutting architecture review: spatial and spatiotemporal

Repository: `luma-contributors`, branch `upstream-first`, working tree as of 2026-11-09 (uncommitted
edits included). Read-only review: no builds, tests, typechecks or yarn were run. Ontology counts
come from importing the existing `modules/analysis/dist/gpu-spatial-analysis/ontology/index.js`
with plain `node` (dist built 2026-11-08 10:03, newer than every ontology source file).

Tags: **CONFIRMED** means I traced the code path or file directly. **SUSPECTED** means inferred
from partial evidence; verify before acting.

Paths are relative to the repository root unless they start with `src/` (=`modules/analysis/src/`)
or `test/` (=`modules/analysis/test/`).

---

## A. analysis-core: the contributor contract

### What a contributor is

- A class implementing published `GPUCommandNodeProducer` (`getCommandNodes(graph)`), 376 such
  classes in 368 files, plus 39 `add*Recipe` functions that compose them. Nodes are built with
  `createWGSLKernelNode` (`src/analysis-core/kernels/wgsl-kernel-nodes.ts:326`), which bakes a
  bounded 3D dispatch from published `getBoundedDispatchLayout` and the device limit (`:364-368`).
- Props are split, by TSDoc convention, into **per-frame** (buffer contents, `GPUParameterBuffer`
  values, no recompile) and **topology / compile-time** (row counts, capacities, which optional
  views exist, `k`, slice counts). Every class repeats this split in its props comment.
- Outputs are **caller-owned** views; status is **GPU-resident one-row `uint32` scalars**, never
  read back by the contributor (ontology field `readback: 'none'` for 260 of 261 operators).
- Validation throws at construction (shape, format, capacity), with `validateCapacity` /
  `validateMaximumCount` in `src/analysis-core/kernels/gpu-contributor-utils.ts`.

### Ports and status (`src/analysis-core/contracts/ports.ts`, `status.ts`)

- Geometry ports: points, lines, polygons (GeoArrow offsets), trajectories; all positions are
  `float32x2` (`ports.ts:20-72`). `GPUPackedGeometryPort` narrows chunked vectors to one packed view
  (`:83-95`).
- Status vocabulary is one composable shape: cardinality (`count`, `requiredCount`), `overflow`,
  `candidateOverflow`, `uncertainCount`, `invalidCount`, iteration status (`iterationCount`,
  `converged`, `iterationLimitReached`) and `approximation` (`status.ts:19-72`). Recipes aggregate it
  per named stage (`GPURecipeStatusPort`, `status.ts:75-83`). This is the convergence contract;
  it is uniform and good.
- Capacity: `GPUCompactOutput` (`gpu-contributor-types.ts:42`) clamps `count` to capacity so it is
  safe as an indirect draw count. Overflow is detected on the GPU; recovery (grow and re-run) is a
  host decision that needs a status readback.

### Spatial context (`src/analysis-core/contracts/spatial-context.ts`)

- `GPUSpatialContext` (`:45-75`): `coordinateSpace` (planar, longitude-latitude, discrete-grid,
  topology-only), `metric` (none, native, great-circle, rhumb, ellipsoidal), `units`
  (native, meters, kilometers), sphere radius or ellipsoid, `edgeModel`, declared-only `datum`,
  `gridFamily`/`gridResolution`, and a planar `frame` identity. CRS transformation is deliberately
  absent; the library checks declarations and never converts.
- `resolveGPUSpatialContext` (`:260-322`) is called from 119 files and is the single gate: it
  rejects lon/lat where unsupported, enforces metric per space, distance-versus-area frame kinds
  and edge models. 135 files take a `spatialContext?` prop. This is the strongest cross-cutting
  contract in the codebase.

### Temporal context: none comparable

There is no `GPUTemporalContext`. Time is described per contributor, in three physical encodings
and several unit conventions:

| Encoding | Where | Unit / epoch |
| --- | --- | --- |
| `float32` relative to a caller origin | trajectory ports (`ports.ts:48-53`), `GPUTemporalReduction` (`src/dataframe-analysis/temporal-reduction/gpu-temporal-reduction.ts:90`), Knox/Mantel (`getGPUSpaceTimeRelativeTimes`, `src/gpu-spatial-analysis/space-time-tests/space-time-parameters.ts:138`), Kalman (`f32` times) | "any unit", origin subtracted on the CPU |
| `float32` high + `float32` low (double-single) | `splitTimestamps` (`src/dataframe-analysis/time-window-filter/time-window-parameters.ts:76`), `GPUFlowAggregation` `timestampsLow` (`src/gpu-network/flow-aggregation/gpu-flow-aggregation.ts:138`), trajectory metrics, playhead, zone events | caller unit |
| `uint32x2` Int64 words | calendar buckets (`gpu-calendar-buckets.ts:98`), time-window filter, temporal reduction word mode, trajectories, space-time hot spots recipe | calendar code requires **epoch milliseconds**; others say "for example epoch milliseconds" |

The unit does exist, but only on the CPU side: `ArrowAnalyticsTimeUnit`
(`src/analysis-core/graph-support/arrow-analytics-normalization.ts:28`) and
`PreparedArrowAnalyticsTimeColumn.unit` / `timeZone` (`:44-54`). It is dropped when
`makeGPUTimeVectorFromPreparedArrowColumn` returns a bare `GPUVector<'uint32x2'>`
(`arrow-analytics-time-upload.ts`). Kepler re-declares the same union as `GPUKeplerTimeUnit`
(`modules/kepler/src/selection/gpu-time-selection.ts:18`). There is no regular-versus-irregular axis
descriptor either: raster temporal composites are regular-slot only (roadmap RS06), slices in the
hot-spot recipe are calendar steps, temporal reduction uses fixed-width buckets.

### Uncommitted diff in `modules/analysis/src` (10 files, +15 / -29)

What it does: deletes the local `GPUFloat32Positions` type from
`src/analysis-core/kernels/gpu-contributor-types.ts` and imports the published one from
`@luma.gl/graph/geospatial` in eight contributors (flow aggregation, point density, region mask,
region statistics, region statistics grid, nearest-feature join, nearest nodes, zonal statistics)
and in `ports.ts`. The published type is identical:
`GraphDataView<'float32x2'> | GraphVectorView<'float32x2'>`
(`node_modules/@luma.gl/graph/dist/geospatial/types.d.ts:3`), and `./geospatial` is a public
`exports` key, so the upstream-first guard (`scripts/check-upstream-first.mjs`) will pass.

Completeness: no remaining import of `GPUFloat32Positions` from `gpu-contributor-types` anywhere in
`modules/*/src`, `modules/*/test` or the website (checked by grep). Type-level, it is complete.

Consistency problems:

1. `ports.ts:16` now does `export type {GPUFloat32Positions} from '@luma.gl/graph/geospatial'`, and
   `contracts/index.ts:30` re-exports it, and `src/gpu-spatial-analysis/index.ts:8` does
   `export * from '../analysis-core/contracts/index'`. So the public entry
   `@luma-contributors/analysis/gpu-spatial-analysis` now re-exports a luma.gl type. AGENTS.md
   says "Never fork, re-implement or re-export luma.gl APIs". Before the diff it was a local
   duplicate (re-implementation); after it is a re-export. It is the only `export ... from
   '@luma.gl...'` in `modules/analysis/src`. The guard script checks import paths only, so it will
   not catch this. **CONFIRMED.**
2. `dev-docs/upstream-first-foundation.md:17` still lists `GPUFloat32Positions` as living in
   `kernels/gpu-contributor-types.ts`. **CONFIRMED.**

### Other core findings

- Capacity planners: `capacity-planning.ts` holds nine planners plus `getGPUCapacityRecovery`
  (`:171-404`, 564 lines). No source file calls any planner function; only types are imported
  (spatial-weights) and `contracts/index.ts:158` says "no contributor grow loop uses them". They
  are exercised only by tests. The trajectory-encounter and event planners
  (`:239`, `:360`) are the only spatiotemporal capacity guidance and are unused. **CONFIRMED.**
- Validity: masks are expanded `uint32` per row (nonzero accepts); Arrow validity is uploaded as
  one `uint32` per row, not as a bitmap (`arrow-analytics-time-upload.ts:22-23`). Invalid outputs
  use `0xffffffff` / `-2^31` / NaN sentinels. Uniform in practice, but no shared type names the
  sentinel convention. **CONFIRMED** (sentinels from `gpu-calendar-buckets.ts:31-36` and others).
- `cell-key-wgsl.ts` is fixed-point sum scale plus order-preserving f32 keys, not cell IDs; the
  name is misleading but harmless. **CONFIRMED.**

---

## B. Spatial indexing and joins

### Index structures that exist

| Structure | Implementation | Users | Limits |
| --- | --- | --- | --- |
| Uniform grid (2D **and 3D**) | published `GPUGridIndex` | 8 contributors: neighbor search, KDE, IDW, kriging, clustering, Dorling, GWR, trajectory encounters | `cellCount + 1` offsets in one binding: 2^25 - 1 cells on a default device (`gpu-contributor-utils.ts:297`) |
| BVH | published `GPUBVH` | spatial-join passes, segment intersection; neighbor search `searchStructure: 'bvh'` reuses `getNearestBVHNodes` | BVH neighbor search is planar only (`gpu-neighbor-search.ts:297`) and opt-in; `'auto'` means grid |
| Hilbert keys | local `GPUHilbertKeys` | BVH packing, joins | `uint32` key, order at most 16 (`hilbert-keys/gpu-hilbert-keys.ts:41`) |
| Sorted DGGS cell tables | local Quadbin and H3 WGSL (`cell-indexing/h3-*-wgsl.ts`), S2 coverings and hierarchy (`cell-topology/gpu-s2-*.ts`), radix-sorted 64-bit keys | `GPUCellAggregation`, rollup, pyramid, cell cover, table compare | capacity-bounded table that keeps the smallest keys on overflow; positions only for Quadbin (`gpu-cell-aggregation.ts:148`), H3 points via `GPUPointToCell` |
| Hash index | published `GPUHashIndex` | flow aggregation only | power-of-two `pairCapacity` |
| Key join | local `GPUKeyJoin` (sort plus binary search), `uint32` or `uint32x2` keys | dataframe | `src/dataframe-analysis/key-join/gpu-key-join.ts:105-110` |

Skew: the default for neighbor search is the uniform grid; the BVH "remains an explicit opt-in
until a measured workload justifies selection" (`gpu-neighbor-search.ts:79-84`). Heavily clustered
points therefore fall on the grid path by default, where per-cell lists grow. A clustered-data
oracle exists (`test/oracles/geometry-neighbor-search-clustered.py`) and an A/B spec
(`neighbor-search-structure-ab.spec.ts`), so correctness under skew is covered; performance under
skew I did not verify. **SUSPECTED** (performance only).

### (x, y, t) and (cell, period)

- **(x, y, t) indexing exists.** `GPUTrajectoryEncounters` builds a 3D `GPUGridIndex` with the time
  bucket as the third axis: `gridSize: [columns, rows, bucketCount]`
  (`src/gpu-spatial-analysis/trajectory-encounters/gpu-trajectory-encounters.ts:421`), and the published
  index accepts `float32x3` positions (`gpu-grid-index.d.ts:3-10`). It is used only there; it is a
  bucketed (discrete-time) index, and the doc calls the result "a discrete approximation, a preview".
  **CONFIRMED.**
- **(cell, period) dense cubes exist, in two places, and the roadmap does not mention them:**
  1. `GPUTemporalReduction`: dense `cellCount * bucketCount` slots with count, min, max, first, last
     and a compact list of occupied slots; fixed-width buckets; caller supplies `uint32` cell IDs;
     float32 mode capped at 2^20 buckets (`gpu-temporal-reduction.ts:53`); **no sum or mean**
     (the file has no `sum` at all). **CONFIRMED.**
  2. `addSpaceTimeHotSpotsRecipe` (`src/gpu-spatial-analysis/recipes/space-time-hot-spots-recipe.ts`):
     `GPUCalendarBuckets` decodes Int64 epoch-ms, an inline adapter kernel computes an absolute
     calendar step index (year, quarter, month, week, day, hour) and bins events into a dense
     `cell * sliceCount + slice` `uint32` cube by `atomicAdd`, or, on request, through
     `GPUGroupStatistics` and a scatter. No readback. Counts only (no value aggregation). Slices at
     most 256 (`emerging-hot-spot-parameters.ts:14`), cells below 2^24, cube below 2^31 bins
     (`gpu-emerging-hot-spots.ts:286-296`). Spatial cells must be a planar lattice or caller cell
     IDs; lon/lat lattices are rejected. **CONFIRMED.**
- The recipe states why it does not use the DGGS path: `GPUCellAggregation` keys "must be valid
  Quadbin or H3 cells and cannot carry a slice index", and the `GPUPointToCell` +
  `GPUCellAggregation` route "would need a cell-key to row lookup" (`space-time-hot-spots-recipe.ts:337-340`).
  Invalid keys are skipped by `GPUCellAggregation` (`gpu-cell-aggregation.ts:76`). **CONFIRMED.**
- The calendar-step-to-slice computation is inline WGSL inside the recipe, not a reusable
  contributor; `GPUCalendarBuckets` outputs only cyclic fields (year, month, day, hour, ...), no
  absolute period index (`gpu-calendar-buckets.ts:37-63`). **CONFIRMED.**

### How a DGGS (cell, period) cube composes without readback today

Every piece below exists and stays on the GPU; one small adapter kernel is missing:

1. `GPUPointToCell` (or pre-keyed `uint32x2` cells) gives one 64-bit H3/Quadbin key per event.
2. `GPUCalendarBuckets` gives calendar columns; **missing**: a reusable "absolute period index"
   node (today only the recipe's inline WGSL).
3. Either (a) `GPUGroupStatistics` directly on a composite key: it accepts `uint32x2` keys
   (`src/dataframe-analysis/group-statistics/gpu-group-statistics.ts:156-160`), so a kernel that packs
   `(cell row or truncated cell bits, period)` into 64 bits gives sparse (cell, period) count, sum,
   mean, min, max, variance and percentiles in a capacity-bounded sorted table; or (b)
   `GPUCellAggregation` once to get the distinct cells, `GPUKeyJoin` (`uint32x2` keys) to map every
   event to its cell row, then `row * periodCount + period` as a `uint32` key into
   `GPUGroupStatistics` or the dense atomic path. Option (a) is cheaper; option (b) gives the dense
   cube that `GPUEmergingHotSpots` reads.
4. Two-period comparison is already covered by `GPUCellTableCompare` and
   `addPeriodComparisonRecipe` (one cell table per period, compile-time number of periods).

**SUSPECTED** (composition reasoned from the prop types; not built or run).

### Findings (B)

- B1. (x, y, t) bucket indexing exists via 3D `GPUGridIndex` but only inside trajectory encounters.
  **CONFIRMED** (`gpu-trajectory-encounters.ts:418-427`).
- B2. Two space-time cube paths exist (`GPUTemporalReduction`, the hot-spot recipe); neither
  aggregates values by sum or mean, and neither uses DGGS cells. **CONFIRMED.**
- B3. There is no reusable absolute-period-index node; the only one is inline in the recipe
  (`space-time-hot-spots-recipe.ts:461-477` step expression and slice test, `:492` `daysFromCivil`).
  **CONFIRMED.**
- B4. Published `GPUSegmentedBVH`, `GPUSegmentedSort`, `GPUBatchSort`, `GPURunLengthEncode` are
  public in `@luma.gl/graph/gpu-core` (`index.d.ts`) and unused here; a segmented BVH per time bucket
  is a plausible (x, y) index per period without a new structure. **SUSPECTED** (not evaluated).

---

## C. Iteration and scale constraints inherited from published luma.gl

### What the published graph offers

`GPUCommandGraphNodeCondition` is either a CPU predicate or a GPU indirect dispatch triple
(`node_modules/@luma.gl/graph/dist/gpu-core/gpu-command-graph-types.d.ts:372-391`). There is no
loop or continuation node. So iteration is always unrolled at compile time; a GPU-written gate
(`createWGSLDispatchGate`, `wgsl-kernel-nodes.ts:183`) can only turn later rounds into zero-workgroup
dispatches. 28 source files use the gate. **CONFIRMED.**

### Register entries verified

1. Centrality guard (workarounds section 3): `MAXIMUM_COMMAND_NODE_COUNT = 32768` at
   `src/gpu-network/network-centrality/gpu-network-centrality.ts:70`, enforced at `:418-423`. Matches.
   It is the only command-node budget in the codebase; no other contributor estimates its node count.
   **CONFIRMED.**
2. Segmented sum (workarounds section 2, line 43, and foundation section 2 line 36): the register says
   "`offsetsWithinInput` and `implementation` options are accepted and ignored". Neither option exists:
   `createSegmentSumNode` props are `id, operation, segmentCount, input, segmentOffsets, output,
   reduction, condition, extraResources` (`src/analysis-core/kernels/sorted-segment-sums.ts:235-251`),
   and no file in `modules/` mentions `offsetsWithinInput`. The register text is stale.
   **CONFIRMED.**
3. Raster vertical datum (workarounds section 2, line 48): option at
   `src/raster-analysis/raster-grid/gpu-raster-grid.ts:69`, attached through a cast at `:99-100`.
   Matches. **CONFIRMED.**
4. Bounded dispatch above about 4.2M rows (workarounds section 3, line 63; roadmap section 3, line
   247): the published 10.0.0-alpha.10 primitives do apply bounded layouts with the device limit:
   `GPUScan` (`dist/gpu-core/gpu-scan.js:79`, `:694-695`), `GPUSort` (`gpu-sort.js:105`, `:131`,
   `:204`), `GPUCompaction` (`gpu-compaction.js:65`, `:97`, `:193`), `GPUReduction`
   (`gpu-reduction.js:244`), `GPUGridIndex` (`gpu-grid-index.js:89`), `GPUBVH`
   (`gpu-bvh.js:520-525`). The code that the row says is missing is present. The row itself says
   "Unverified here: no test pins the failure mode". **CONFIRMED** that the bounded layouts are in
   the published code; **SUSPECTED** that the regression row is obsolete (a runtime test above
   4.2M rows would settle it).

### Where the real scale ceilings are for spatial and spatiotemporal work

- Single packed views plus the storage-binding limit (128 MiB default) bound row counts well before
  dispatch does: `GPUCalendarBuckets`, `GPUTemporalReduction`, `GPUCellAggregation`,
  `GPUEmergingHotSpots` and the hot-spot recipe require single packed views (`validatePackedView`
  throws on `GraphVectorView`, `gpu-contributor-utils.ts:205-214`) and none of them calls
  `validateGPUStorageBindingByteLength` or `getStorageBindingElementLimit` (grep count 0). For
  `uint32x2` timestamps that is about 16.7M rows per binding on a default device; above it the
  failure would surface as a WebGPU binding validation error, not a construction error. Other
  contributors (spatial join candidates, cell cover, grid index) do model the binding limit
  explicitly. **SUSPECTED** (failure mode not run).
- Compile-time slice and class counts: emerging hot spots at most 256 slices; Markov transition
  matrices at most 64 classes (`distribution-dynamics/gpu-transition-matrix.ts:25`); Knox/Mantel at
  most 2^16 permutations and 2^24 - 1 rows, with the Besag-Clifford batch loop bounded to 256
  batches (`space-time-tests/space-time-stopping.ts:29`, `:145-146`), so Knox/Mantel is properly
  bounded. **CONFIRMED.**
- Serial time loops per thread: `GPUTrajectoryKalman` runs one thread per track and loops over the
  track's observations in WGSL (`trajectory-estimation/gpu-trajectory-kalman.ts:122`, `:151`), with
  `f32` times. Parallelism is the track count; a few very long tracks give a long single-thread loop.
  **CONFIRMED** (structure); latency impact **SUSPECTED**.
- Float32: WGSL has no f64. The workarounds in use are: CPU origin subtraction to f32 relative time
  (Knox/Mantel, Kalman), double-single high/low pairs (time-window filter, flow aggregation,
  trajectory metrics, playhead, zone events), Int64 `uint32x2` words with integer arithmetic
  (calendar, temporal reduction word mode, time-window word mode), fixed-point 64-bit accumulators
  for sums (`kernels/fixed-point-accumulator.ts`, cell aggregation), and published
  `GPUFloat64Positions` raw binary64 input for projection into a local f32 frame
  (`GPULocalFrameProjection`). Contributors most exposed to f32: trajectory analytics in float mode
  (time and position), space-time variogram (`f32` times only), Knox/Mantel (relative f32 time, with an
  explicit `requiredResolution` guard in `getGPUSpaceTimeRelativeTimes`), and planar neighbor
  search (documented ulp warning, `gpu-neighbor-search.ts:210-214`). **CONFIRMED** (from the files
  and the f32/word/low-part tally per contributor).

---

## D. Ontology and catalog

- Model (`src/gpu-spatial-analysis/ontology/spatial-analysis-ontology.ts`): 14 data kinds (including
  `trajectories` and `cells`, no time-series, event-table or cube kind, `:6-21`), 9 stages
  (`:27-37`), 11 domains including `movement` (`:43-55`), 7 coordinate models (no temporal model,
  `:61-69`). Operators carry `dimensionality: '2d' | '2d+time' | 'topology' | 'mixed' | 'unspecified'`
  (`:281`).
- Counts (from dist): 49 capability families (48 available, 1 opportunity:
  `advanced-spatial-models`); 261 operators; 23 `proven`, 238 `partial` evidence.
  Operators per domain: geometry 54, spatial-statistics 48, spatial-models 34, regional-analysis 32,
  workflow 23, discrete-global-grids 23, topology 13, density-and-interpolation 12, movement 11,
  proximity 10, execution 1.
- Temporal classification is effectively absent:
  - Only **1 of 261** operators is `2d+time` (`addFleetDwellRecipe`,
    `spatial-analysis-operators.ts:701`); 195 are `unspecified` because the default is
    `override.dimensionality ?? 'unspecified'` (`:841`). Knox, Mantel, emerging hot spots, Markov,
    every trajectory contributor, the space-time hot-spot and period-comparison recipes are
    `unspecified`. **CONFIRMED.**
  - Space-time statistics sit in `spatial-statistics` (Knox in
    `spatial-autocorrelation-and-inference`, emerging hot spots and Markov in
    `distribution-dynamics`); recipes in `workflow`. **CONFIRMED.**
  - The catalog covers only two packages: `package` is
    `'@luma-contributors/analysis/gpu-spatial-analysis' | '.../gpu-terrain'` (`ontology.ts:276-278`).
    So `GPUCalendarBuckets`, `GPUTemporalReduction`, `GPUTimeWindowFilter`, `GPUSpaceTimeVariogram`,
    `GPURasterTemporal`, `GPUNetworkTimeRouting` and `GPUFlowAggregation` are not in the catalog
    (checked with `getGPUSpatialAnalysisOperator`). The temporal building blocks are invisible to
    capability discovery. **CONFIRMED.**
- `coverage-expansion-catalog.ts` adds 8 `coverage-*` families (movement, geometry refinement,
  global coordinates, local prediction, statistical contracts, point processes, urban morphology,
  landscape ecology); `kepler-analysis-catalog.ts` adds 3. **CONFIRMED.**

---

## E. Tests

### Spec files per test directory (1,237 `*.spec.ts`, of which 563 `*.node.spec.ts`)

| Test directory | Source directory | Specs |
| --- | --- | --- |
| `gpu-spatial-analysis` | `gpu-spatial-analysis` | 670 |
| `gpu-terrain` | `gpu-terrain` | 133 |
| `gpu-raster` | `raster-analysis` | 120 |
| `gpu-network` | `gpu-network` | 103 |
| `gpu-dataframe` | `dataframe-analysis` | 85 |
| `analysis-core` | `analysis-core` (plus kernel audit) | 42 |
| `utils` | `testing` and helpers | 24 |
| `geospatial-expansion` | mixed | 22 |
| `gpu-pointcloud` | `pointcloud-analysis` | 11 |
| `gpu-graph` | mixed | 8 |
| `computation-record` | `computation-record` | 7 |
| `gpu-project` | `geospatial-analysis` | 6 |
| `gpu-core`, `oracles`, `geospatial`, `gpu-geometry` | | 2, 2, 1, 1 |

Test directory names still follow the luma.gl branch (`gpu-dataframe`, `gpu-raster`, `gpu-project`),
not the source directories. `GPUTrajectoryRestriction` (a `gpu-spatial-analysis` contributor) is
tested under `test/gpu-network/network-optimization/`. Seven empty, untracked directories remain
(`execution`, `experimental`, `gpu-data`, `gpu-physics`, `gpu-simple-graph`, `gpu-vector`,
`operations`). `test/arrow/` holds 53 tracked GeoArrow data files (220 KB) that nothing references
(`git grep` for `geoarrow-data` and `test/arrow` outside the directory returns nothing).
**CONFIRMED.**

### Pinned reference fixtures in spatial and spatiotemporal directories

Specs that load pinned fixtures (`oracle-fixture`, `oracles/fixtures/*.json`, parity fixture
modules) versus all specs:

| Directory | Specs | With pinned fixtures |
| --- | --- | --- |
| trajectory-analysis | 6 | 0 |
| trajectory-interpolation | 3 | 0 |
| track-similarity | 6 | 0 |
| distribution-dynamics | 7 | 0 |
| neighbor-search | 7 | 0 |
| cell-topology | 7 | 0 |
| hilbert-keys, bounds-filter, pair-gather | 3, 2, 1 | 0 |
| calendar-buckets (`gpu-dataframe`) | 3 | 0 JSON; the zoneinfo reference is a TS module (`calendar-buckets/time-zone-reference.ts`) |
| trajectory-encounters | 8 | 2 |
| trajectory-zones | 10 | 5 |
| trajectory-estimation | 1 | 1 |
| space-time-tests | 10 | 3 |
| emerging-hot-spots | 11 | 6 |
| spatial-join | 42 | 7 |
| cell-aggregation / cell-cover / cell-table-compare / cell-indexing | 5 / 6 / 4 / 10 | 3 / 2 / 1 / 1 |
| region-statistics / zonal-statistics | 12 / 8 | 2 / 4 |
| temporal-reduction / time-window-filter | 5 / 8 | 1 / 1 |
| raster remote-sensing | 9 | 6 |

The zero rows rely on in-repo CPU oracles and property tests, not external-library fixtures.
Coverage-matrix rows that claim direct references (for example B-MV01 MovingPandas, MV03 tslearn)
have their external evidence in other directories or in the `geospatial-expansion` and
`gpu-network` specs. **CONFIRMED** (counts by grep; "relies on CPU oracles" from spot reads).

### Fixture generation and reproducibility

- Fixtures live in `test/oracles/fixtures/` (738 files on disk; `GENERATORS.md` counts 746).
  Generators are in the repository: 236 `.py`, `.R`, `.cpp` files plus one `.mjs` in `test/oracles/`, with
  `requirements.txt` and `GENERATORS.md` mapping fixtures to scripts. Nothing was left behind on
  the luma.gl branch for these. **CONFIRMED.**
- Mapping quality: 540 literal, 204 **inferred** (matched by name stem, "lower confidence"), 2
  unresolved (`GENERATORS.md:903`); 33 scripts write outside `fixtures/` or are unresolved
  (`:895-899`). **CONFIRMED.**
- Stale paths: `GENERATORS.md:5` and 64 of 229 Python generators name `modules/graph/test/oracles/...`
  in their usage text and in the `generator` field they write into fixtures (for example
  `statistics-emerging-hot-spots.py:10`, `:181`). 227 of them resolve output paths from `__file__`, so
  running them still writes to the right place, but regenerated fixtures will record a dead path.
  **CONFIRMED.**
- No runner: nothing in `scripts/` or `package.json` regenerates fixtures; regeneration is manual,
  per script, under a pinned Python environment (and R, CGAL, Boost for some). Reproducible in
  principle from this repository, not automated, and two generators assert older exact versions
  (`GENERATORS.md` "Reproducing"). **CONFIRMED.**
- Root `test/` holds only `kepler-compute-example-specs.json` and `utils/`; `scripts/` has no
  fixture tooling. **CONFIRMED.**

---

## F. Downstream (short look)

- Lenses use the temporal contributors as scenario operators: `GPUCalendarBuckets` (6 files),
  `GPUTrajectoryMetrics` (5), `GPUTimeWindowFilter`, `GPUTemporalReduction`,
  `GPUTrajectoryResample`, `GPUEmergingHotSpots`, `GPUFlowAggregation`, `GPUCellTableCompare` (4
  each), and the space-time hot-spot and period-comparison recipes (3 each). **CONFIRMED** (grep counts).
- Kepler uses `GPUTimeWindowFilter` with Int64 words for time selection
  (`modules/kepler/src/selection/gpu-time-selection.ts:13-16`), owns its own unit type and
  conversion to a target unit (`:18`, `:52-68`), plans trip animation on the CPU
  (`modules/kepler/src/analysis/trip-animation.ts:21`, matching kepler's trip-layer origin
  subtraction), and imports `makeGPUTimeVectorFromPreparedArrowColumn` from `analysis-core`.
  Kepler does not use calendar buckets, temporal reduction or any trajectory contributor.
  **CONFIRMED.**
- **Correction to roadmap LN-V5 (`dev-docs/roadmap.md:208`)**: "space-time cube" there means a
  computation-visualization view of wavefronts from `settledAt` (archive
  `computation-visualization-roadmap.md:356`, `:619`, `:670`), not a data cube. A data space-time
  cube already exists (`addSpaceTimeHotSpotsRecipe`, `GPUTemporalReduction`) and is documented as one
  (`docs/api/gpu-spatial-analysis/operators-statistics.md:1664-1668`). The row should say "space-time
  cube view of solver wavefronts" to avoid the conflation. **CONFIRMED.**

---

## G. docs/api

### Coverage of spatiotemporal contributors

Every spatiotemporal contributor I checked (35 symbols) is mentioned in `docs/`. Most have their
own heading: calendar buckets, temporal reduction, time-window filter, space-time variogram
(`docs/api/gpu-dataframe-analysis.md`); trajectory metrics, resample, playhead, encounters, closest
approach, zone events, track similarity, fleet dwell (`gpu-spatial-analysis/operators-lines.md`,
`recipes.md`); Knox, Mantel, spatial/LISA Markov (`operators-dynamics.md`); emerging hot spots and the
space-time recipe (`operators-statistics.md`); raster temporal and seasonal break
(`gpu-raster/remote-sensing.md`).

Without an API heading (only coverage-table rows, lens embeds and prose):
`GPUTrajectoryKalman`, `GPUTrajectoryRestriction`, `GPUTrackSimilaritySearch` (examples in
`gpu-spatial-analysis/movement-coverage.md`, contracts in prose in
`gpu-network/network-movement-coverage.md:85-130`), `GPUNetworkTimeRouting`
(`network-movement-coverage.md:16`, `:266`), `addFleetDwellZoneEventsRecipe` (bold paragraph,
`operators-lines.md:1118`), and the helpers `getGPUTimeZoneTransitionTable`,
`getGPUSpaceTimeRelativeTimes`, `prepareGPUTrajectoryObservations` (prose mentions only).
`makeGPUTimeVectorFromPreparedArrowColumn` is not in `docs/` at all. The movement contracts for
`gpu-spatial-analysis` classes live under `docs/api/gpu-network/`. **CONFIRMED.**

### Spot checks against source

1. `GPUCalendarBuckets` (`docs/api/gpu-dataframe-analysis.md:1358-1424` against
   `src/dataframe-analysis/calendar-buckets/gpu-calendar-buckets.ts`): outputs, sentinels, offset ranges,
   time-zone table semantics and packer signature all match. One overclaim: line 1367 says "The
   columns feed `GPUTemporalReduction`, `GPUCellAggregation` keys and heatmaps". `GPUCellAggregation`
   skips any key that is not a valid Quadbin or H3 cell (`gpu-cell-aggregation.ts:74-78`), and the
   space-time recipe says calendar slices cannot be carried in its keys
   (`space-time-hot-spots-recipe.ts:337-340`); `GPUTemporalReduction` takes timestamps, not calendar
   columns. **CONFIRMED** drift.
2. `addSpaceTimeHotSpotsRecipe` (`docs/api/gpu-spatial-analysis/operators-statistics.md:1784-1835`):
   chain, slice formula, rejected cyclic fields, time-zone exclusivity, lon/lat rejection and status
   stages match the source. It omits the hard limits (256 slices, cells below 2^24, cube below 2^31
   bins, `cellCount * sliceCount < 0xfffffffe` at `space-time-hot-spots-recipe.ts:398`) and the fact
   that only counts are aggregated. Omission, not drift. **CONFIRMED.**
3. `GPUTemporalReduction` (`docs/api/gpu-dataframe-analysis.md:1304-1356`): outputs, bucket rule,
   2^20 float bucket cap, word mode, sorted mode and single-packed-view rule match the source. No
   drift. **CONFIRMED.**

---

## What the architecture makes cheap, and what it blocks, for spatiotemporal work

### Cheap

- **Bucketed (cell, time) aggregation without readback.** Calendar decode, absolute slices,
  dense atomic cubes, sorted group statistics with `uint32x2` keys and per-period cell tables all
  exist and compose inside one command graph. Adding a value-weighted space-time cube or a DGGS
  cube is one adapter kernel plus existing contributors.
- **Exact time arithmetic.** Int64 word timestamps with integer kernels give exact calendar fields,
  windows and buckets on every adapter; the patterns (Hinnant civil-from-days, nibble division,
  binary-searched time-zone tables) are already written and tested against zoneinfo.
- **Per-frame scrubbing.** Time windows, bucket origin and width, and hot-spot settings are
  per-frame parameter buffers; sorted inputs get O(range) indirect dispatch
  (`GPUTemporalReduction` `sorted`). Animation and playhead views need no recompile.
- **Uniform status and capacity semantics.** Every new spatiotemporal contributor can reuse
  `GPUCompactOutput`, the bounded status vocabulary and recipe stage status, so overflow and
  convergence are reported the same way everywhere.
- **(x, y, t-bucket) candidate search.** The 3D `GPUGridIndex` pattern in trajectory encounters
  generalises to any space-time proximity question at bucket resolution.

### Blocked or expensive

- **Iteration with data-dependent length.** No GPU continuation in the published graph; every
  iterative space-time method (permutation inference, regionalization over time, EM-style or
  smoother iterations) is unrolled with a compile-time cap and gated to no-ops. Fine for hundreds of
  rounds, not for thousands, and only centrality has a node budget guard.
- **Time semantics as a contract.** Without a temporal context, nothing checks that two inputs share
  a unit, epoch or calendar; `GPUCalendarBuckets` silently mis-decodes nanosecond or second words;
  float32 relative times silently lose resolution unless the caller used the guarded helper.
- **Continuous time.** Encounters are discrete-bucket; Kalman is one serial thread per track;
  irregular raster time axes are not supported (RS06). Continuous space-time joins (moving object
  against moving object at exact times) exist only as closest-approach on raw samples.
- **Large single-view temporal inputs.** Temporal contributors take one packed view, so the row
  ceiling is the storage-binding size (about 16.7M Int64 timestamps on a default device), with no
  construction-time check in the contributors I examined.
- **Discovery.** The ontology cannot answer "which operators consume time", because time is not a
  data kind or coordinate model, the dimensionality field is unset for almost everything, and the
  dataframe and raster temporal contributors are outside the catalog.

---

## Ranked recommendations

1. **Fix the uncommitted diff before committing**: drop `GPUFloat32Positions` from the
   `ports.ts:16` re-export and `contracts/index.ts:30`, so the public
   `gpu-spatial-analysis` entry no longer re-exports a luma.gl type (AGENTS.md rule); callers import
   it from `@luma.gl/graph/geospatial`. Update `dev-docs/upstream-first-foundation.md:17`. Consider
   teaching `scripts/check-upstream-first.mjs` to flag `export ... from '@luma.gl/...'`.
2. **Add a minimal temporal contract** in `analysis-core/contracts` (unit, epoch, encoding
   `float32-relative | double-single | int64-words`, optional time zone, regular or irregular axis),
   carried from `PreparedArrowAnalyticsTimeColumn` onto the uploaded vector and checked by
   `GPUCalendarBuckets` (must be milliseconds) and by multi-input temporal contributors, the way
   `resolveGPUSpatialContext` works. Retire `GPUKeplerTimeUnit` in favour of it.
3. **Extract the absolute period index** from `addSpaceTimeHotSpotsRecipe` into a contributor
   (calendar step or fixed width, origin, interval, count; `uint32` period per row plus a skipped
   count). It unlocks DGGS cubes and value cubes through `GPUGroupStatistics` with no new index.
4. **Correct the registers**: retire or re-word the 4.2M bounded-dispatch row (workarounds section 3,
   roadmap section 3) after a single runtime test above 4.2M rows per primitive; remove the
   non-existent `offsetsWithinInput` / `implementation` options from the segmented-sum row
   (workarounds line 43, foundation line 36); re-word LN-V5 as a computation-visualization view.
5. **Make time visible in the ontology**: set `dimensionality: '2d+time'` on the roughly two dozen
   temporal operators, add an `events` or `time-series` data kind, and extend `package` to include
   `dataframe-analysis` and `raster-analysis` (or add a separate temporal catalog), so the calendar,
   window, reduction and raster temporal blocks are discoverable.
6. **Guard binding-size ceilings** in single-view temporal and cell contributors
   (`validateGPUStorageBindingByteLength`), so oversized inputs fail at construction with a named
   limit.
7. **Fixture hygiene**: replace `modules/graph/test/oracles` with `modules/analysis/test/oracles`
   in `GENERATORS.md` and the generators' `generator` fields; add a small runner that regenerates
   one family and diffs; resolve the 204 inferred mappings opportunistically. Delete the
   unreferenced `test/arrow/` data and the seven empty test directories.
8. **Docs**: fix `gpu-dataframe-analysis.md:1367`; add the hard limits and "counts only" to the
   space-time recipe section; give `GPUTrajectoryKalman`, `GPUTrajectoryRestriction`,
   `GPUTrackSimilaritySearch` and `GPUNetworkTimeRouting` API headings in the
   `gpu-spatial-analysis` and `gpu-network` operator pages.
9. **Decide on capacity planners**: either wire `getGPUCapacityRecovery` and the event/encounter
   planners into a host-side grow loop helper (the one place readback is legitimate), or move them
   to `test/` to stop shipping 560 lines of unused internal code.
