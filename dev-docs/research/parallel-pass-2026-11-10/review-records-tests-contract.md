# Review: recording, test configuration, contract alignment (diff vs 74dddcf5)

Read-only review. Ran: `node scripts/check-upstream-first.mjs` (no violations),
`npx vitest run --project node modules/analysis/test/computation-record/wgsl-kernel-records.node.spec.ts`
(4/4 pass), one scratch script (`scan-records.mjs` in this folder) that runs the new node-field checks
over all 2328 JSON files in `docs/lens-records` and `modules/lenses`, plus small `node -e` checks.

## Defects, most severe first

### 1. (Confirmed, by reading) The new unindexed-snapping cap tightens GPUNetworkKFunction, and the check runs in `getCommandNodes`, not at construction
- `modules/analysis/src/gpu-network/network-accessibility/gpu-network-snapping.ts:288-296` adds
  `validateMaximumCount(positions x edges <= 2^27)` whenever `candidateCapacity` is unset. This is a
  new limit, not a converted assert. Inputs that used to work are now refused.
- `GPUNetworkKFunction` builds its `GPUNetworkSnapping` inside `getCommandNodes`
  (`modules/analysis/src/gpu-network/network-k-function/gpu-network-k-function.ts:451`). So a K function
  with 2^22 events on 64 edges (2^28 tests), which built before, now throws at `getCommandNodes`. The
  error carries the nested id `<id>-snap` ("network-snapping positions times topology edges ..."), not
  the K function's own prop. Contract section 10 says such checks run in the constructor.
- The kernel-audit case was cut to fit, not left to fail: `cases-network.ts:2044` drops the topology from
  64 to 32 edges ("32 edges keep events times edges at the 2^27 limit"). The K function has no rejection
  case for this limit. `operator-annotations.ts` lists no limit for `GPUNetworkKFunction`, and no doc in
  `docs/api/gpu-network/` mentions the cap.
- Concrete input: `new GPUNetworkKFunction({positions: 4194304 rows, topology with 64 edges, ...})` with
  no `candidateCapacity` constructs, then `getCommandNodes` throws.

### 2. (Confirmed, derived by hand and with `Math.fround`) `prepareGPUTrajectoryObservations` accepts a slightly out-of-range origin
- `modules/analysis/src/gpu-spatial-analysis/trajectory-estimation/prepare-trajectory-observations.ts:56-57`
  replaces the double-precision assert `longitude >= -180 && longitude <= 180` (and the same for
  latitude) with `validateFiniteNumber`. That function compares `Math.fround(value)`
  (`gpu-contributor-utils.ts:538`).
- `Math.fround(180.000001) === 180` and `Math.fround(90.000003) === 90`, so origins the assert rejected
  now pass. The origin is used in float64 CPU math (spheroid anchor, ENU basis), so float32 rounding has
  no reason to apply here. The per-row checks on the same lines still use doubles, so the origin and the
  rows now follow different rules.
- Severity is low (tiny overshoot), but the conversion does not reject exactly what the assert rejected.

### 3. (Confirmed, by reading) `iterationLimitReached` is set when no limit was reached (two places)
- Stream order, Euler-tour path: `gpu-terrain-stream-order.ts:487-489` clears `status[3]` when a stream
  cell has no tour place (a receiver cycle). `createTerrainIterationLimitNode` writes `1 - status[3]`
  (`terrain-flow-tracing.ts:247`). So a cycle reports `iterationLimitReached = 1` even with budget left.
  The type doc (`gpu-terrain-stream-order.ts:59`) admits this ("budget ran out or a receiver cycle was
  found"). Contract section 4 ties `iterationLimitReached` to the limit only. A consumer that raises
  `maximumIterations` when the flag is 1 will loop on a cyclic flow-direction raster.
- Spike repair: `gpu-terrain-spike-repair.ts:951-958` publishes `1 - converged`. The published
  `GPUGraphAlgorithmsConnectedComponents` also writes `converged = 0` on topology overflow
  (`@luma.gl/graph` `gpu-graph-connected-components-internals.js:108`), not only when iterations run out.
  This is suspected to be unreachable for the internal pixel topology, but nothing guarantees it.
- On convergence, at the limit, and when the limit falls on the same round as convergence, the gate is
  correct (checked in `raster-relaxation.ts:172-186` and `:303-330`): `changed == 0` is tested before
  `iterationCount >= max`, so a last round with no change gives converged = 1 and limit = 0. Profile
  peaks gives exact complements (both are computed from undecided candidates).

### 4. (Suspected, low) The kernel-audit fuzzy-contiguity "maximum" case is not the worst case for polygon count
- `cases-coverage-statistics.ts:386-401` raises features from 512 to 65536 to match the new annotation
  limit. `polygonOffsets` and `ringOffsets` stay at 16385, so 65536 features share 16384 polygons. The
  case still tests the 65536 feature and vertex limits it names, but no case covers 65536 features with
  one polygon each.

### 5. (Suspected, low) The deprecated aliases resolve differently from class to class
Each class defines what happens when both names are given, but the rule is not the same everywhere:
- `GPUDatumGridTransform` (`mask` over `validity`) and `GPUPointCloudStatisticalFilter` (`validity` over
  `mask`): the new name silently wins.
- `GPUTrajectoryThinning`, `GPUTrajectoryEncounters` and `GPUTrajectoryClosestApproach`: giving both
  throws.

When the new name is omitted, the old behaviour is kept in every class checked.

### 6. (Suspected, low) Lens runs leak resources when reading timings fails
In `raster-lens-harness.ts` and `spatial-scenario-utils.ts`, `readLensRunMetrics` (which awaits
`readTimings`) now runs before `readLensViewSnapshot`. If reading timings rejects, the snapshot arena is
never destroyed. If anything throws between `submitTimedLensGraph` and `readTimings`, the query set is
never freed. The leak only happens on the error path.

## Areas checked with no defect found
- **Recording side table** (`wgsl-kernel-nodes.ts:512-614`):
  - Pays one lookup per kernel: a graph that never opts in pays one `WeakMap.get` per kernel and keeps
    nothing.
  - No memory kept after GC: values are `Map<id, {props, source, workgroupSize, layout}>`. Values that
    reference the graph through binding handles do not keep the key alive, because WeakMap entries are
    ephemerons. Only `GPUComputationRecorder.kernelGraph` holds the graph strongly, for as long as the
    recorder lives.
  - No id collisions: ids are scoped per graph, and a rebuilt graph object gets a new map.
- **Timings:** absent rather than zero. `readTimings` returns `undefined` unless a query set was created
  and `canReadGPUTimings` is true. `readLensRunMetrics` then writes `timing: {nodes: []}` and
  `timestamped: false`. A query set of `2 x nodeOrder.length` is safe: luma's `CommandEncoder` skips
  passes past the count (`command-encoder.js:66`).
- **Old records still validate:** none of the 2328 JSON records carries `constants`, `workgroupSize` or
  `dispatch`, and the new checks only run when a field is present.
- **Private imports:** `check-upstream-first` reports no violations, and `@luma.gl/graph/gpu-core` is a
  public entry point.
- **Test configuration:**
  - Nothing that used to run is now excluded. `excludedFromRuns` (131) is a subset of the old list
    (271), as is `excludedFromTypecheck` (59). The two lists do not overlap, every listed file exists, and
    140 specs were re-enabled.
  - Both consumers read the new JSON shape: `vitest.config.ts` uses `excludedFromRuns` keys, and
    `scripts/typecheck-tests.mjs` uses both lists.
  - The skip reasons are real. Published alpha.10 does not export `OutOfMemoryError` from
    `@luma.gl/core` or `walkTextGlyphs` from `@luma.gl/engine/text`. It does not export
    `prepareArrowTableForGPUAnalytics` from `@luma.gl/graph/arrow` either; that name exists only in
    analysis-core.
  - No kepler spec diff weakens an assertion: the edits are bracket access, `readonly`, `!`, a
    `Promise.withResolvers` shim, and `lumaComputePlot?.status` (still fails if missing).
  - The casts do not hide a real mismatch: `markdownOutput` and `isValid` are added after the object is
    built, and the `layerIds`/`enabled` fields the filter fixture gained are not read by
    `selectFilteredRows`.
- **Converted asserts:**
  - Each converted check rejects exactly what the assert rejected (apart from defect 2). Checked: peak
    prominence, summed-area table, summits (`< 8192` became `<= 8191`), the raster gate slot, the datum
    grid (widths, offsets, spacing, tolerance, iterations), point buffer, track-similarity search,
    local permutation and study window. The removed `assert(!node.condition)` could never fail.
  - Track-similarity `lcssTolerance` now compares in float32 (only `-0`-rounding inputs differ).
- **Status ports:**
  - `ports.status` on depression hierarchy: the port node runs after the node that writes `status` and
    reads words 1 and 2. `getOutputViews` replaces every `Object.values(output)` use.
  - `GPUOPTICS` strips `ports` before `Object.values`.
- **Point-cloud `rowDirection: 'south'`:**
  - The flip is applied consistently. The cell source and the grid index use the same domain (y in
    (0, h]) and the same row (`ceil - 1`). Every `PointCloudRasterGrid` consumer (Surfaces, Height with
    nearest, bilinear and IDW, Ground, SMRF) goes through `getPointCloudCellSource`, and SMRF and
    Surfaces pass `rowDirection` to the index.
  - Bilinear and IDW work in flipped row units, so the weights mirror. SMRF's gradient uses `length()`,
    so its sign does not matter.
- **Kernel-audit limits:**
  - The resized cases still name and hit their limit. Snapping: 4 x maxRows(4) = 2^27, and 2^24 x 8 + 8
    is over it. Change: 101^3 is within and 102^3 is above 2^20. Neighborhoods: 2^24 x 256. Peak
    prominence: max + 1.

## Test gaps that would hide a defect
- No headless test runs `rowDirection: 'south'` through SMRF, Ground, or Height with bilinear or IDW. The
  row-direction spec covers only Surfaces and nearest Height.
- No test of `iterationLimitReached` on stream order with a receiver cycle, and none checking that
  `converged = 1` and `limit = 0` when the last allowed round is also the first unchanged round (the
  same-round case) for terrain pointer jumping.
- No rejection case for `GPUNetworkKFunction` (or the drive-time catchment recipe) at the new 2^27
  unindexed-snapping limit.
- No test that the origin of `prepareGPUTrajectoryObservations` is rejected just outside +-180/+-90.
- No test that both alias names given to `GPUDatumGridTransform` or `GPUPointCloudStatisticalFilter`
  pick the documented winner.
- No test of the recorder default (`recorder.graph`) when `describeGPUCommandGraph` gets a compiled graph
  from a different graph than the one recorded. Facts for matching ids would be attached silently.
