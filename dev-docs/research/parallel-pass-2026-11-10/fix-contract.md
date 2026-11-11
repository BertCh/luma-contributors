# fix-contract report

## 1. Done
1. Snapping cap (reproduced). Spec `GPUNetworkKFunction schedules events times edges above the unindexed snapping limit`
   (K node spec: 2^21 events, 128 edges = 2^28 tests) failed before with
   `k-snap positions times topology edges without candidateCapacity must be at most 134217728` at getCommandNodes; passes now.
   DEVIATION from the brief: I did NOT derive `candidateCapacity` in the K function. It cannot work: the indexed path needs a
   `maximumSnapDistance` radius (the K function usually has none, as in the audit case) and an `output.status.overflow` flag;
   `candidateCapacity` is a global candidate count (all points), so an all-pairs capacity is the same 2^28 words and breaks the
   128 MiB binding, and a smaller one could overflow silently. Instead:
   - `GPUNetworkSnapping` gets an optional `maximumUnindexedWork` (default = the 2^27 constant; validated with validateCapacity).
   - `GPUNetworkKFunction` passes `Number.MAX_SAFE_INTEGER`, so everything that built at 74dddcf5 still builds; nothing can throw late.
   - Audit case restored to 64 edges (`cases-network.ts`, kFunctionEventsCase call).
   - Where the cap remains (direct snapping caller): the kernel-audit rejection case, operator-catalog limit and docs
     already existed; I fixed the docs sentence in `docs/api/gpu-network/operators.md` (it claimed the K function and recipe are capped)
     and added a node spec in `gpu-network-accessibility.node.spec.ts` (limit 29 refuses at 30 tests with the pinned message, 30 accepts,
     0 rejected).
   - No CPU-reference comparison above the cap: pair counting is O(events^2), infeasible at 2^21 events. The K path above the cap is
     the same code as below it (only the limit value differs); existing parity specs cover the results.
   Kepler paths that hit the cap and fall to the CPU (not edited): `modules/kepler/src/analysis/catalog-gpu-snapping.ts`
   `runSnap(false)` retries at lines 141 (when `maximumDistance` is 0 or non-finite, `canIndex` false so `useIndex` false and the error is
   rethrown), 146 (after an indexed failure) and 149 (after candidate overflow); the final catch at line 192 returns `engine: 'cpu-reference'`.
   Only when points x segments > 2^27. Fix there: pass `maximumUnindexedWork` (new prop) to the scan, or chunk the points.
   Also not mine: `drive-time-catchment-recipe.ts:439,594` forwards an unset `candidateCapacity` to snapping, so it now throws above the cap
   (built before). Same remedy: pass `maximumUnindexedWork` there.
2. Trajectory origin (reproduced). Real file is `gpu-spatial-analysis/trajectory-estimation/prepare-trajectory-observations.ts` (not
   trajectory-analysis). Exact double comparison via `validateRequirement` with the same message form. Spec
   `prepareGPUTrajectoryObservations compares the origin in double precision` (180.000001 and 90.000003 rejected, 180/90 accepted) failed
   before, passes now (in `trajectory-contract-completion.node.spec.ts`).
3. iterationLimitReached (reproduced in stream order). Spec `GPUTerrainStreamOrder reports a receiver cycle as unconverged, not as an
   iteration limit` (`gpu-terrain-drainage.spec.ts`, 4x1 raster with a 2-cell cycle) failed before (limit read 1, converged 0), passes now
   (converged 0, limit 0). Fix: the Euler-tour path publishes the flag right after the ranking rounds, before the unknown-order gather
   clears `converged`. Its round count is a fixed bound that always suffices, so the flag is 0 there; `converged` is unchanged. Type docs updated.
   Spike repair: kept `1 - converged`; the labelling runs a fixed iteration count and its only other way to clear `converged` is topology
   overflow, which the fixed-stride CSR (2 slots per pixel, `gpu-terrain-spike-repair.ts` graph node) cannot produce. Comment added; the
   existing spec (spike-repair.spec.ts:303,311) covers converged and limited runs.
   Other classes, by reading: all publish `1 - converged` after a gated loop that only stops by convergence or the limit:
   `raster-relaxation.ts:478` and `raster-expansion-utils.ts:151` (cost distance, reachability-like rasters), `gpu-cost-distance.ts:1034`
   (`1 - (cost & tie)`), `gpu-raster-watershed-segmentation.ts:331`, `gpu-network-cost-matrix.ts:595` (`1 - allConverged`),
   `gpu-network-reachability.ts` (min-converged node near line 720 region via `limitReached = 1 - min(converged,1)`),
   `terrain-flow-tracing.ts:248` (height above drainage, watersheds, pointer jumping; nothing else clears word 3 there), profile peaks
   (`gpu-profile-peaks.ts:708-733`, exact complement from undecided candidates). The other writers of `status[3] = 0`
   (`terrain-flow-passes.ts:305`, `terrain-flow-spill-fill.ts:819`) do not publish a limit flag. Not verified by running; no defect found by reading.
4. Aliases (spec written; see Verified for caveat). `GPUDatumGridTransform` and `GPUPointCloudStatisticalFilter` now throw
   `<id> mask replaces the deprecated validity; pass only mask` / `<id> validity replaces the deprecated mask; pass only validity`; JSDoc updated.
   Specs in `gpu-datum-grid-transform.node.spec.ts` and `gpu-pointcloud-analysis.node.spec.ts`.
5. Gaps in my files: K above the cap (item 1), aliases (item 4), stream-order cycle (item 3), origin (item 2). Not done: the same-round
   convergence/limit case for terrain pointer jumping.

## 2. Verified
- `npx vitest run --project node` on K spec + accessibility spec: 13/13; datum, pointcloud, trajectory-analysis, geospatial-analysis and all
  gpu-network node specs: 59 files, 313 tests pass; whole `analysis-core` node directory (kernel audit): 71 files, 2924 tests pass.
- `npx vitest run --project headless modules/analysis/test/gpu-terrain/hydrology/gpu-terrain-drainage.spec.ts`: 7/7 (the new cycle spec failed first).
- Spec-first failures observed: K above cap, origin, stream-order cycle. NOT observed failing first: the two alias specs (first run failed
  on my fixture's duplicate resource ids, then I fixed the fixture and the fix together); the pre-fix behavior (both names accepted) is by reading.
- `node scripts/typecheck-status.mjs` over my files: 0 errors. `node scripts/check-upstream-first.mjs`: no violations.
  `check-contributor-contract --all` showed nothing for my classes. biome check --write on my files: clean.
- Not run: headless spike-repair spec, kernel-audit headless project, full K function headless parity.

## 3. Not done
- Derived `candidateCapacity` in the K function (infeasible, see 1). Same-round pointer-jumping spec.

## 4. Wave 2 requests
- Kepler `catalog-gpu-snapping.ts` and `drive-time-catchment-recipe.ts` (see 1): pass `maximumUnindexedWork` or choose the indexed path.
- No renames. `maximumUnindexedWork` is a new optional prop (additive).

## 5. Roadmap
- Snapping cap row, if any: "kepler snapping scan and drive-time recipe still refuse above 2^27 tests".

## 6. Noticed
- The lead's path for the trajectory file was wrong; real path is under trajectory-estimation.
- Item 3 type doc for the pull fallback: at the limit with a final round that changed things, converged reads 0 and limit 1 even if the
  next round would have found nothing; consistent with the rule (rounds ran out), just conservative.

## Follow-up
- Drive-time recipe (`drive-time-catchment-recipe.ts`): both snappings now pass `maximumUnindexedWork: Number.MAX_SAFE_INTEGER` (only matters without `candidateCapacity`). Node spec `drive-time recipe still schedules the exact scan above the unindexed snapping limit` (2^24 demand points, 16 edges = 2^28 tests; point count is capped at 2^24 by the 128 MiB binding) failed before (`drive-time-catchment-snap-demand ... must be at most 134217728`, observed by temporarily removing the line, restored), passes now. Recipe node specs 7/7, K function specs pass.
- Complete grep of `new GPUNetworkSnapping` outside tests: in modules/analysis/src only the recipe (2 sites) and the K function, both fixed. Others, not edited: `modules/kepler/src/analysis/catalog-gpu-snapping.ts:105` (retries the scan above the cap, falls back to CPU, see item 1), `modules/lenses/src/computation-lenses/scenarios/gpu-network/network-cost-matrix.example.ts:34` and `network-snapping.example.ts:25` (small fixed scenarios), `examples/deck/spatial-analysis-explorer/modes/accessibility-mode.ts:231`, `examples/deck/spatial-analysis-showcase/src/scenes/networks/job-accessibility.compute.ts:658,735` and `job-accessibility.scene.ts:422` (user-sized point sets; could exceed 2^27 only on very large inputs; they should pass `candidateCapacity` or `maximumUnindexedWork`).
- Documentation: one sentence added to the `candidateCapacity` JSDoc of the recipe props and of the K function props saying the exact scan has no index, the limit is raised for it, and it becomes slow near 2^27 point-edge tests. (Prop docs live in the source; `docs/api` operators text already describes the raised limit.)
- Alias specs: now observed failing before the fix. I temporarily disabled each new throw in a scratch copy: both alias specs failed (2 failed, 7 passed), then restored the files (the throws are present again; the two files each contain the new message once).
