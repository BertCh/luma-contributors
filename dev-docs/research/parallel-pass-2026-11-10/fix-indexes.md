# fix-indexes report

## 1. Done

### Finding 1: fuzzy contiguity, asymmetric CSR at the tolerance. Reproduced on the GPU.
- The defect reproduced in headless Chromium/WebGPU on the current code. With the reviewer's input
  (tolerance 100, A = [0.0861, 0.1861] x [0, 0.1], B.minX = 100.18610382080078) the result was
  `[[], [0]]`, so A did not list B. The shuffled case (8 groups of A, B one f32 step past the tolerance
  and C two steps further) produced the asymmetric pair `[2, 17]`.
- Fix (`gpu-fuzzy-contiguity.ts`): the query slack is now `(largest |own coordinate| + TOLERANCE) * 1e-5`.
  A comment explains why this is conservative for any tolerance and coordinate magnitude, and why the
  old slack was not.
- Specs in `fuzzy-contiguity-index.spec.ts`. The runner was refactored into `runFuzzyContiguity`.
  - Two new specs failed before the fix and pass after it: the reviewer's example, and the shuffled
    near-tolerance case. The shuffled case checks symmetry, checks that every pair the f64 brute-force
    reference accepts is present, and checks that any extra pair lies within 1e-6 of the tolerance.
  - Three new specs passed before the fix and are kept as pins: a gap equal to the tolerance (dyadic,
    exact), a large tolerance (2^20, plus 1e17 giving all pairs), and coordinates near 2^22 (f32 step
    0.5) with gaps of 0, 0.5 and 1.
  - The existing 579-feature shuffled reference test now also checks symmetry.

### Finding 2: fuzzy contiguity performance. Done with the published GPUSort and a Morton-sorted BVH.
- The BVH is now built with `getSegmentBVHNodes(..., spatialSort: true)` from
  `segment-intersection/segment-bvh.ts`, a Morton sort of the bound centres using the published
  `GPUSort` and `GPUBVH` `sourceIds`. Traversal maps leaf slots to rows through `leafIds`.
- Leaves are no longer in row order, so emitted pairs carry the key `(row << shift) | neighbor`. They
  are radix-sorted (published `GPUSort`, `keyBits = 2 * ceil(log2 rows)`) together with the distance
  bits, then one finish pass writes neighbors, weights, added and distances. The output is in the
  same ascending order as before; the existing exact-equality reference spec passes unchanged.
- Node and row bounds are packed into one `searchBounds` binding, so the emit passes stay at 8
  storage bindings.
- Timing is GPU submit-to-readback on the headless test GPU, the median of 3 runs before and after
  (the "before" runs include the finding 1 fix):

| Input | Before | After |
|---|---|---|
| 4,096 squares, shuffled | 18.9 ms | 4.0 ms |
| 16,384 squares, shuffled | 111 ms | 5.4 ms |
| 16,384 squares, grid order | 4.0 ms | 5.8 ms |
| 65,536 rows (16,384 squares + 49,152 empty), shuffled | 203 ms | 9 ms |

- 65,536 *valid* features cannot be built: the vertex cap is 65,536, so 16,384 squares is the
  practical maximum. The 65,536 cap stays; it is now also the key-packing limit, and the comments and
  catalog reason say so. The harness was a temporary spec, now deleted; a copy is kept at
  `/private/tmp/claude-501/-Users-robertchristie-Documents-GitHub-luma-contributors/310b8947-6a4f-4e58-adb4-aef2ba8200cd/scratchpad/fuzzy-contiguity-timing.spec.ts.txt`.
- Behaviour change on overflow only: the row in which the capacity runs out now keeps an unspecified
  subset of its neighbors. Before, it kept the lowest-numbered ones. This is documented in the class
  documentation. Results are unchanged when there is no overflow.
- Docs updated in `docs/api/gpu-spatial-analysis/coverage-statistics.md` and the catalog reason in
  `ontology/all-entry-points/operator-annotations.ts`.

### Finding 3: local kriging `'auto'` uncapped scan. Confirmed by reading (no GPU run needed); fixed.
- Rule now, also written in the class documentation, the prop documentation and
  `docs/api/gpu-spatial-analysis/interpolation-expansion.md`:
  - `'auto'` scans only when there are at most 4,096 samples **and** at most 2^32 sample-query tests
    (the named `GPU_LOCAL_KRIGING_MAXIMUM_BRUTE_FORCE_WORK`).
  - Otherwise it uses the grid.
  - It throws, with a named `validateMaximumCount` message, only when the grid cannot fit and the scan
    exceeds the cap. That needs more than 2^24 samples plus queries and more than 2^32 tests.
- To avoid rejecting configurations that worked at 74dddcf5 (where `'auto'` had no cap at all), the
  grid was changed so that it runs wherever the scan ran:
  - The grid kernel no longer binds `positions`. Sample positions are copied, as raw bits, after the
    candidate rows in the same binding, so the grid binds exactly as many storage buffers as the scan.
    The binding-count fallback is gone.
  - Queries are split into chunks whose candidate rows plus the copy of the positions fit the device's
    `maxStorageBufferBindingSize`, so the 128 MiB fallback is gone. There is one `GPUNeighborSearch`,
    pad pass and kriging kernel per chunk; chunk 0 keeps the node id `id`.
  - New optional, compile-time `@internal` prop `sampleIndexChunkQueryCount`, used to test chunking on
    small inputs.
- The only newly rejected corner is more than 2^24 samples plus queries together with more than 2^32
  scan tests.
- Results change for configurations whose resolution moved from scan to grid: the crowded drift +
  `parameters` + weights case above 4,096 samples, and 4,096 or fewer samples with more than 2^32
  tests. Because of finding 4 the grid gives the same neighbors, so the results are the same.
- Spec `local-kriging-sample-index.spec.ts`:
  - Updated: the crowded case, which asserted the old behaviour, now expects `'grid'`.
  - New: 4,096 samples x (2^20 + 1) queries resolves to `'grid'`, and an explicit `'brute-force'`
    with the same sizes is rejected with the named message.
  - New: `'auto'` above the cut-over gives the same values as the scan, end to end.
  - New: a chunked run (chunks of 7 queries) matches the scan.

### Finding 4: local kriging near-ties. Reproduced on the GPU.
- The reviewer's configuration (row 0 one f32 d^2 step above eight rows at d^2 = 25, `neighborCount`
  4) gave different values on grid and scan with the old code (`-0.075` against `-0.322`).
- A circle of 40 samples at radius 5 (a near-tie group larger than the margin) also failed with the
  old code.
- Fix: the scan is untouched; it still selects by `distance()` with ties to the lower row. The grid
  kernel uses the same selection code (`retainSample`) and adds an exactness guard:
  - When the candidate list is full and its farthest candidate (by `distance()`) is within 1e-6
    relative of the retained boundary (the `neighborCount`-th distance, or the search radius), that
    query rescans every sample.
  - Candidates are ascending by row, so the tie order is the scan's.
  - The margin grew from 4 to `max(neighborCount, 8)`, so lattice tie rings rarely trigger a rescan.
  - The constants and the reasoning are documented in the source.
- Specs. Grid and scan are compared for bitwise-identical values, status and weights:
  - Failed before the fix, pass after it: the near-tie case (with CPU float32 preconditions asserted)
    and the 40-sample circle case at `neighborCount` 4 and 16.
  - Passed before the fix, kept as pins: random non-lattice cases for ordinary, chunked, co-kriging
    (2 variables, ordinary means, target variable 1) and external drift (2 columns, search radius 4).

### Finding 5: explicit `'grid'` limits. Only partly reproduced (by reading).
- An oversized explicit grid was **already** rejected at `getCommandNodes`. `createWGSLKernelNode`
  throws `GPULocalKriging: kernel <id> binds N storage buffers, more than the device limit`, and it
  also checks binding sizes. So it did not reach device validation, as the review assumed.
- The existing `space-time-kriging.node.spec.ts` pins that message. I kept it, and did not add a
  second binding-count check that would have changed the message.
- Added at the start of grid node building:
  - `validateGPUStorageBindingByteLength` for the search targets (samples plus queries,
    `float32x2`).
  - A named `validateRequirement` when not even one query fits beside the copy of the sample
    positions.
  - The candidate rows can no longer be oversized, because they are chunked.
- Spec (null WebGPU device with the default limits): 9 bindings is rejected with the generic message,
  2^24 + 1 targets is rejected with the named message, and an in-limit explicit grid builds.

### Finding 6: pair statistics `gridSize` single-invocation loops. Confirmed by reading; fixed.
- New helper `getPairStatisticsSegmentTotalNodes` in `pair-statistics-cell-neighbors.ts`: per-segment
  totals in levels where each invocation adds at most 256 entries, in a fixed order (3 levels at
  2^24 rows). It is u32-exact for counts.
  - The published `GPUSegmentedReduction` is not exported from `@luma.gl/graph/gpu-core`.
  - The core `createSegmentSumNode` is f32-only, and counts must stay exact.
- `GPUCrossVariogram` grid path: the pair pass writes bin-major `binSums` (f32) and `binCounts` (u32,
  now exact integers instead of f32 counts). Segment totals feed a new `grid-finish` pass.
- `GPUMarkedPointStatistics` grid path:
  - The pair pass writes bin-major sums (cumulative, shell, products) and counts. Segment totals feed
    a grid finish.
  - The moments come from a per-row terms pass, 4 published `GPUReduction` sums, and a one-thread
    moments pass. The means are sum/count, not the dense running mean.
  - The finish formula moved into `getFinishWGSL(props)`, shared by both paths with identical text.
  - The dense path no longer allocates its `rows * bins * 6` partials on the grid path.
- The dense paths' WGSL is textually unchanged (same loops, same formulas), so dense results are
  unchanged. The grid docs already allowed last-bit differences and now describe the tree order.
- Specs in `pair-statistics-grid-paths.spec.ts`:
  - New, structural: a 100,000-row grid build of both classes on the null device. No kernel contains a
    `row < 100000u; row++` loop. The old finish and moments text matched that pattern, so this spec
    fails by construction on the old code; I did not run the old code.
  - New, kept as pins: lattice pairs on cell edges and bin or shell edges (bins left-closed, final edge
    included), with two grid sizes each, for both classes.
  - New, kept as pins: a radius larger than the extent (one-cell collapse), for both classes.
  - All compare against the dense path: exact counts and status, values within 1e-4.
  - The existing 8 grid-path tests still pass.
- Not changed: when the lattice collapses to one cell, the grid *pair* pass still loops over every row
  per focus (dense cost). This is now documented in `statistical-expansion.md`.

## 2. Verified

All runs below are headless (Chromium/WebGPU) unless marked node.

| Specs | Passed / total |
|---|---|
| `fuzzy-contiguity-index.spec.ts` | 7/7 |
| The same, before the fix | 2 failed, 5 passed |
| `coverage-estimators.spec.ts` + `statistics-upstream.spec.ts` + fuzzy | 57/57 |
| Node: `coverage-statistics.node.spec.ts` + `kernel-audit/coverage-statistics.node.spec.ts` | 31/31 |
| `local-kriging-sample-index.spec.ts` | 13/13 |
| The same, before the fix | near-tie and circle failed, 10 passed |
| `interpolation-advanced`, `covariance-parameters`, `interpolation-coverage`, `kriging-conditioning` | 84/84 |
| `vecchia-prediction`, `prediction-scale-robustness` | 8/8 |
| Node: `space-time-kriging`, `interpolation-coverage`, `covariance-parameters`, `interpolation-spatial-context`, `interpolation-performance` | 21/21 (after keeping the generic binding message) |
| Node: `kernel-audit-coverage-interpolation.node.spec.ts` + `space-time-kriging.node.spec.ts` | 44/44 |
| `pair-statistics-grid-paths.spec.ts` | 13/13 |
| `geospatial-expansion.spec.ts` | passed |
| Node: `geospatial-expansion.node.spec.ts` | 2/2 |
| Node: `kernel-audit-expansion.node.spec.ts` + lenses `dataframe-formulas.node.spec.ts` | 150/150 |
| `gpu-marked-point-statistics.spec.ts` | **2 failed** (see Noticed) |

Other checks:
- `node scripts/check-upstream-first.mjs`: no violations.
- `yarn check-contributor-contract`: 2 new findings, both in `pointcloud-analysis/index.ts`, which is
  not mine. `--all` lists the dense moments loop at `gpu-marked-point-statistics.ts:185`; it is in the
  baseline and unchanged.
- `node scripts/typecheck-status.mjs` with my paths: 0 errors. I could not confirm whether that cycle
  covers test files.
- `npx biome check` on all touched files: clean.

## 3. Not done
- Single-invocation O(n^2) loop in the grid pair pass when one cell holds most rows (the one-cell
  collapse). Documented only. A per-cell occupancy cap or a tiled pair pass is a separate change.
- I did not lower any registered limit: the fuzzy cap stays 65,536 (now fast) and the grid row cap
  stays 2^24 - 1 (its loops are now parallel).

## 4. Wave 2 requests
- No renames.
- Coordination: the agent that added the untracked
  `modules/analysis/test/gpu-dataframe/pair-statistics/gpu-marked-point-statistics.spec.ts` (same-type
  K normalisation, n(n-1)) should know that the finish formula now lives in `getFinishWGSL(props)` and
  is shared by both paths. Its normalisation change belongs there, plus the dense moments pass and the
  new `grid-moments` pass, which carry the focus and neighbor counts.

## 5. Roadmap
- Indexed-search review rows for fuzzy contiguity, local kriging `sampleIndex` and pair-statistics
  `gridSize`: delete the fuzzy and kriging items. What remains for pair statistics is "grid pair pass
  degenerates to O(n^2) when one cell holds most rows (radius at or above the extent); add an
  occupancy cap or a tiled pair pass."

## 6. Noticed
- Review finding 7 (kepler snapping retry above the 2^27 unindexed cap falls back to the CPU) was out
  of scope and is untouched.
- `getSegmentBVHNodes` (`segment-intersection/segment-bvh.ts`) is now shared by fuzzy contiguity. Its
  comment says it is a private copy kept for one contributor; consider moving it to `analysis-core`.
- WGSL `sqrt` is only 2.5 ULP accurate, so any code that assumes `distance()` equals
  `sqrt(index d^2)` bitwise across paths is fragile. The kriging guard tolerates this; other indexed
  kNN consumers may not.
