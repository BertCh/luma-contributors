# weights-regression-pairs (wave 1, rerun after the disk was freed)

## 1. Done

**Fuzzy contiguity (`GPUFuzzyContiguity`)**: indexed by default, results unchanged.
- `modules/analysis/src/gpu-spatial-analysis/spatial-weights/gpu-fuzzy-contiguity.ts`: the dense rows x rows
  distance table and the single-invocation `offsets` loop are gone. The validate pass now also writes per-feature
  bounds (empty bounds for invalid features). A `GPUBVH` over those bounds (unsorted, so leaves stay in row order)
  is probed per row with the box widened by the tolerance plus a 1e-5 relative slack. Only candidates run the
  same exact segment/containment test (it now returns early on containment; the result is the same). The CSR goes
  through `createCSRStages` (counts, `GPUScan`, clamped offsets, overflow, required count), and a per-slot pass
  writes the weights and `added`. Every kernel binds at most 8 storage buffers.
- Feature limit raised from 512 to 65,536 (an additive change: inputs that were rejected before are now
  accepted). The vertex limit stays at 65,536.
- Catalog: `operator-annotations.ts` (features 65536, new reasons). Docs: `docs/api/gpu-spatial-analysis/coverage-statistics.md`.
  Kernel audit: `cases-coverage-statistics.ts` max case now 65,536 features.
- New spec `modules/analysis/test/gpu-spatial-analysis/spatial-regression/fuzzy-contiguity-index.spec.ts`: 585
  shuffled features (a grid with gaps that touch, fall within the tolerance and fall beyond it, a frame with a
  hole, a contained square, empty features) checked against a brute-force CPU reference: offsets, neighbours,
  added, weights, required count, status. It also has a rejection case,
  `fuzzy-contiguity features must be an integer in [1, 65536]`.

**Island attachment (`GPUIslandAttachment`)**: the O(rows^2) counts/write passes are gone, results unchanged.
- `gpu-island-attachment.ts`: islands are grouped by the row they attach to, using `getSlotGroupingNodes` (a
  stable sort, so ascending order is kept). Each output row merges its original CSR row, its own attachment and
  the islands attached to it, through `createCSRStages`. The cost per row is now its degree, not a scan of all
  rows. The `nearest` pass is unchanged (islands x rows); see Not done.
- Docs text updated. New spec `island-attachment-merge.spec.ts`: 3,000 rows with more than 100 islands and
  one-way edges, reciprocal and directed, full and truncated capacity, checked against a brute-force reference
  (attachment, offsets, neighbours, weights, added, overflow, required count).

**Cross-variogram and marked statistics**: optional grid path; omitting it keeps today's dense traversal.
- New optional compile-time prop `gridSize?: [columns, rows]` on `GPUCrossVariogram` and
  `GPUMarkedPointStatistics`. It needs no compile-time distance bound: the shared pair-statistics lattice is
  fitted every encoding with cells at least `maximumDistance` / `maximumRadius` wide, so those stay per-frame.
  Cross-variogram: an inclusion pass writes complete-case flags, packed attributes and atomic order-preserving
  bounds. A parameters pass builds `[bounds, maximumDistance, maximumTime]` (an unusable maximum time gives a NaN
  radius, so no pairs). Then come `getPairStatisticsInputNodes` and one invocation per focus over the 3 x 3
  cells. Marked statistics reuse the caller's window parameters directly. Each pair goes into its first
  containing shell, and the cumulative sums are prefix sums. Both write the dense partial layout, so the finish
  passes are unchanged.
- Row limits with a grid: rows x bins <= 2^24 (cross) and rows x radii x 3 <= 2^24 (marked). Without a grid,
  rows stay at 16,384. The row check now uses `validateCapacity`: `cross-variogram positions length must be an
  integer in [1, 16384]`. Before, it was the combined "invalid bounded pair shape" message, which no spec
  matched.
- New file `pair-statistics/pair-statistics-cell-neighbors.ts` (the 3 x 3 cell loop WGSL, grid-size validation).
- Catalog entries, docs (`statistical-expansion.md`) and two kernel-audit grid cases (`cases-expansion.ts`,
  131,072 rows x 128 bins and 43,690 rows x 128 radii).
- New spec `modules/analysis/test/gpu-dataframe/pair-statistics/pair-statistics-grid-paths.spec.ts`: grid vs
  dense on 2,000 rows (cross with mask, univariate, space-time, invalid distance; marked homogeneous, and
  types/marks/intensities/mask uncorrected). Pair counts and status are equal, and floats agree within 1e-4
  relative. Also 20,000 rows on the grid against a brute-force CPU pair count, and the dense-path rejection.
- Behaviour note: the grid path adds sums in cell order, so its floats can differ from the dense path in the last
  bits. A marked `maximumRadius` whose square overflows float32 gives no pairs on the grid. Both are documented.

**STD-03 bare asserts in this stream's directories**, all replaced with named-limit helpers:
`pair-histogram.ts` (3), `spatial-maximum-likelihood-wgsl.ts` (1), `spatial-maximum-likelihood-nodes.ts` (1).

## 2. Verified
- `npx vitest run --project headless`: fuzzy-contiguity-index (2/2), coverage-estimators + fuzzy-index (18/18),
  island-attachment-merge + coverage-statistics.spec (25/25), statistics-upstream (33/33), geospatial-expansion
  (10/10), pair-statistics-grid-paths (8/8), pair-histogram + gpu-variogram + pair-statistics-limits (13/13),
  spatial-lag/error maximum likelihood (21/21). All passed.
- `npx vitest run --project node`: kernel-audit coverage-statistics + coverage-statistics.node (31/31), lenses
  statistics-coverage.node (14/14), kernel-audit-expansion (105/105), geospatial-expansion.node (2/2),
  pair-statistics-limits.node + gpu-variogram.node + kernel-audit-statistics (436/436),
  spatial-maximum-likelihood-model-result.node (4/4). All passed.
- `node scripts/typecheck-status.mjs '<my paths>'`: 0 matching errors (0 total), cycle finished 6:50:54 PM, after
  my last source edit. I did not confirm whether that typecheck covers test files.
- `yarn check-contributor-contract --all`: "no new findings". The fuzzy/island single-invocation loops no longer
  appear. The remaining findings in my directories are listed in section 3.
- `node scripts/check-upstream-first.mjs`: no violations. `npx biome check` on all 14 touched files: clean.
- Not run: the full suites and `yarn test` (forbidden in this wave).

## 3. Not done
- **MGWR local fits (GPU-05)**: not touched. `gpu-multiscale-geographically-weighted-regression.ts` (827 lines)
  still runs rows x candidates local fits over all rows, plus three single-invocation loops (contract finding at
  :285). The route is `createGridIndexNodes` from `gwr-grid-index.ts`, as `GPUGeographicallyWeightedRegression`
  already does (it gives a Gaussian kernel finite support through `gaussianCutoff`, plus bisquare and adaptive
  k-th neighbour). It is an L-sized change to backfitting and bandwidth search, and I could not finish it safely
  here (the disk also dropped to 1.2 GB free during the pass).
- **Island attachment `nearest`**: still brute force (islands x rows). `GPUNeighborSearch` k=1 would need the
  lowest-ID tie rule and the overflow-safe shared-scale comparison kept bit for bit.
- Remaining contract findings in these directories that are not bare asserts:
  `gpu-marked-point-statistics.ts:180` (moments, one invocation), `variogram-model-fit-nodes.ts:118`,
  `generalized-gwr-selection.ts:92`, `gpu-spatial-panel-lag-maximum-likelihood.ts:200` x3,
  `gpu-spatial-panel-regression.ts:221`, `panel-diagnostics.ts:129`.

## 4. Wave 2 requests
- No renames.
- Lens records for `GPUFuzzyContiguity` (graph twins) will change, because the node list changed. That is for
  LN-REC2 regeneration; I did not touch `docs/lens-records`.

## 5. Roadmap
- GPU-05: remove "fuzzy contiguity", "cross-variogram" and "marked statistics" from the brute-force list (the last
  two have an opt-in `gridSize`; the dense default stays for compatibility). "island attachment" becomes "island
  attachment nearest search (output merge is indexed)". MGWR stays.
- PLAT-03: island attachment's nearest search is still not on the shared neighbour index.
- STD-03: delete pair-statistics and spatial-regression from the remaining list (0 bare asserts left there).

## 6. Noticed
- Free disk swung between 1.2 GB and 14 GB during this pass.
- `docs/api/gpu-dataframe-analysis.md` says the `GPUCrossVariogram` dense times path is superseded. That still
  holds; the grid path does not change the advice to prefer `GPUSpaceTimeVariogram` for space-time work.
