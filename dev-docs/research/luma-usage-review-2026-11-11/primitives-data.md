# Stream `primitives-data`: data-parallel primitives of `@luma.gl/graph/gpu-core` (alpha.13)

Reviewer: read-only pass, 2026-11-11. Scope: `modules/analysis/src` against the gather, scan,
compaction, sort, run/segment/search, hash, aggregation and load-balancing primitives of
`node_modules/@luma.gl/graph/dist/gpu-core` (installed `10.0.0-alpha.13`). Nothing was run on a GPU;
no spec was run.

## 1. Coverage

Read on the published side (`.d.ts` contract and, where marked, the `.js` kernel):
- `gpu-gather` (d.ts + js, whole file), `gpu-uint32-gather` (d.ts + js), `gpu-scan` (d.ts),
  `gpu-flag-offsets` (d.ts + js), `gpu-compaction` (d.ts), `gpu-mask` (d.ts), `gpu-sort` (d.ts +
  js for `auto` resolution and radix digit loop, `gpu-sort.js:86-95`, `:499-531`),
  `gpu-batch-sort`, `gpu-segmented-sort` (d.ts), `gpu-run-length-encode` (d.ts + js, whole file),
  `gpu-segment-offsets`, `gpu-segmented-layout` (d.ts), `gpu-coo-to-csr` (d.ts + js, whole file),
  `gpu-galloping-search` (d.ts), `gpu-hash-join` (d.ts), `gpu-group-aggregation` (d.ts),
  `gpu-reduction` (d.ts), `gpu-histogram` (d.ts, `auto` path of js), `gpu-elementwise` (d.ts),
  `gpu-chunked-indexed-scatter`, `gpu-indexed-range-compaction` (d.ts),
  `graph-data-view-utils.js:76-93` (`createTransientView`), `gpu-command-graph.d.ts:230`
  (`createDataView`), `index.d.ts` export list (to confirm what is public).
- Confirmed NOT public in alpha.13: `GPUSegmentedReduction`, `GPUSegmentedScan`, `GPUScatter`,
  `GPUAdaptiveSpMV` (files exist under `dist/gpu-core`, no `index.d.ts` export).

Read on the local side (whole helper or the kernel and its neighbours):
- `analysis-core/kernels/`: `scan-total.ts`, `sorted-segment-offsets.ts`, `binary-search-wgsl.ts`
  (header), `stable-lexicographic-index-sort.ts`, `sorted-segment-sums.ts:160-215`,
  `load-balanced-for-each.ts` (header), `wgsl-kernel-nodes.ts:837-857` (`createFillNode`).
- Gathers: all 42 `x[o + y[o + index]]` lines found by `git grep` were triaged; 20 read in context
  (listed in PD-01/PD-02/PD-03). All 53 kernels with a `gather|permut|scatter` variant name were
  listed; 18 read.
- Run heads / run tables: all 32 variants named `*head*|*run*|*unique*|*boundar*` listed; read
  `healpix-cell-aggregation.ts:220-356`, `cell-table.ts:270-300`, `key-join.ts:405-470`,
  `group-statistics-sort.ts:60-140, 220-265`, `line-split.ts:505-535`,
  `contiguity-weights.ts:635-665`, `terrain-catchment-summary.ts:200-262`.
- Totals after a scan: all 37 `offsets[last] + counts[last]` sites listed; 6 read.
- Hand-written prefix sums: 68 `out[...] = running|total|...` lines listed; 6 read
  (`network-k-function.ts:595-640`, `dot-density-sampling.ts:465-483`,
  `spatial-scan-statistic.ts:900-1000`, `region-statistics-grid.ts:215-235`,
  `geometry-predicates.ts:770-790`, `multiple-testing-adjustment.ts:500-560`).
- In-kernel sorts: all 14 insertion/selection-sort comments listed; `neighbor-search.ts:340-360`
  read. Hash tables: all 8 `atomicCompareExchangeWeak` files listed by name only (not read).
- Published-primitive call sites sampled (29, across 9 directories): `segment-bvh.ts:123`,
  `spatial-join-passes.ts:398`, `:1281`, `flow-aggregation.ts:625`, `:644`, `:673`, `:740`,
  `trajectory-metrics.ts:540`, `:547`, `:626`, `spatial-clustering.ts:698`, `:714`, `:741`,
  `network-service-areas.ts:474`, `:483`, `network-k-function.ts:522`, `:570`, `:622`,
  `pycnophylactic.ts:427`, `:751`, `sparse-resistance-system.ts:206`,
  `terrain-catchment-summary.ts:225`, `terrain-peak-prominence.ts:176`,
  `utilization-volume-isopleths.ts:207`, `bounded-key-pair-sort.ts:60-82`,
  `point-density.ts:543`, `:687`, `:699`, `network-reachability.ts:739`, `make-valid.ts:496-512`,
  `isolines.ts:899`, `space-time-hot-spots-recipe.ts:579-680`.
- `GPUSort` props: a script read all 92 `new GPUSort(` blocks (67 set `keyBits`, 14 force
  `'radix'`, 25 leave `keyBits` at 32; listed in PD-07). 9 of the 25 read.

Not covered (a later pass should not assume these were checked):
- The 8 files with `atomicCompareExchangeWeak` (clustering, ward, terrain hydrology, peak
  prominence tree): not read; they may hold union-find, not hash tables.
- 22 of the 42 one-line gather lines (mostly gathers fused into larger bodies: `ring-assembly`,
  `make-valid`, `coverage-simplification`, `polygon-overlay`, `terrain-flow-spill-fill`,
  `raster-watershed-forest.ts:221` pointer jumping).
- The 14 local `copy` kernels against `GPUElementwise('copy')` / `createGPUCopyCommandNode`
  (listed only; see PD-10).
- `GPUGallopingSearch` against the 52 inline `lowerBound`/`upperBound` call sites: not compared
  site by site.
- Whether the 20-odd inline scan totals that sum arbitrary counts (not flags) are bounded below
  2^32 (PD-11 is a lead only).
- `modules/lenses`, `modules/kepler`, website, examples: out of this stream's scope.

## 2. Findings

### PD-01 Fifteen hand-written uint32 gathers duplicate `GPUUint32Gather`, which the repository already uses
- Class: `adopt-now`.
- Local (each kernel body is exactly `out[i] = source[indices[i]]` on `u32` views, indices a sort
  permutation or otherwise in range):
  `dataframe-analysis/group-statistics/group-statistics-common.ts:62-84` (`createGatherNode`,
  5 calls: `group-statistics-order.ts:124`, `group-statistics-sort.ts:95`, `:120`, `:236`, `:248`);
  `dataframe-analysis/key-join/gpu-key-join.ts:411-416`, `:432-437`;
  `gpu-network/adjacency-matrix/adjacency-matrix-order.ts:136-157`;
  `gpu-network/network-noding/gpu-network-noding.ts:489-500`;
  `gpu-spatial-analysis/geometry-topology/gpu-snap-rounded-noding.ts:676-687`;
  `gpu-spatial-analysis/group-geometry/gpu-group-convex-hull.ts:471-479`, `:489-497`;
  `gpu-spatial-analysis/line-merge/gpu-line-merge.ts:340-345`;
  `gpu-spatial-analysis/spatial-weights/key-pair-grouping.ts:81-92`;
  `gpu-spatial-analysis/spatial-regionalization/gpu-spatial-weights-minimum-spanning-tree.ts:543-555`.
  15 call sites in 9 files.
- Published: `node_modules/@luma.gl/graph/dist/gpu-core/gpu-uint32-gather.d.ts:5-9`,
  `gpu-gather.js:18-48` (validation), `:186-224` (kernel). Precedent in this repository:
  `gpu-spatial-analysis/areal-interpolation/bounded-key-pair-sort.ts:67`,
  `gpu-spatial-analysis/cell-aggregation/cell-table.ts:232`, `:253`.
- Evidence: read both.
- Change: replace each node with `...new GPUUint32Gather({id, source, indices, output})
  .getCommandNodes(graph)` and delete `createGatherNode`. One dispatch per gather either way.
  The published constructor checks what the local code never checks: packed `uint32`, output
  length >= indices length, output in a separate buffer (transients from `createTransientView`
  are separate buffer handles, `graph-data-view-utils.js:87`).
- Verify: `test/gpu-dataframe/group-statistics/gpu-group-statistics.spec.ts`,
  `test/gpu-dataframe/key-join/gpu-key-join.spec.ts` and `parity-wp6-*`, the noding, line-merge,
  convex-hull and contiguity specs (all existing; headless).
- Risk: none for in-range indices (all sites read a permutation from `GPUSort`). Out-of-range
  index: published writes `invalidValue` (0 by default) where the local kernel reads whatever the
  robust-access clamp returns, so the published result is the more defined one. Loss: the
  computation recorder no longer captures the kernel text of these nodes (register section 2,
  first row: published primitives carry no kernel facts), and node IDs change from
  `${id}-gather-*` to the given ID (no spec greps for them; checked with `git grep` over
  `modules/*/test`).

### PD-02 Four float32 gathers duplicate `GPUGather<'float32'>`, which `GPUPycnophylactic` already uses for the same job
- Class: `adopt-now`.
- Local: `analysis-core/kernels/sorted-segment-sums.ts:189-206` (one gather per reduction, runs
  for every `getSortedSegmentSumNodes` caller); `gpu-network/flow-aggregation/flow-aggregation-kernels.ts:678-705`
  (`createFlowSortedGatherNode`); `gpu-spatial-analysis/healpix-cells/gpu-healpix-cell-aggregation.ts:343-355`;
  `gpu-spatial-analysis/areal-interpolation/gpu-areal-interpolation.ts:653-664`. 4 sites.
- Published: `gpu-gather.d.ts:3-30`. Precedent: `gpu-spatial-analysis/areal-interpolation/gpu-pycnophylactic.ts:427-432`
  feeds `GPUGather` into the same local `createSegmentSumNode` that `sorted-segment-sums.ts`
  feeds with its hand-written gather.
- Evidence: read both.
- Change: `new GPUGather({id, source: contributions, indices: sortedIndices, output})`. Same
  pass count. `sorted-segment-sums.ts` is the shared helper behind the exact segment sums, so the
  change reaches every caller of `getSortedSegmentSumNodes` at once.
- Verify: `test/utils/` and the segmented-sum specs that pin the register's "bit-identical"
  claim; `gpu-flow-aggregation.spec.ts`; healpix and areal-interpolation specs. A gather copies
  32-bit words, so the sums are bit-identical.
- Risk: same as PD-01 (zero row on an out-of-range index; recorder loses the text).

### PD-03 `GPUPairGather` column copies are `GPUUint32Gather` with `invalidValue`
- Class: `adopt-now` for uint32 columns, `adopt-with-cost` for float32 columns.
- Local: `gpu-spatial-analysis/pair-gather/gpu-pair-gather.ts:438-468` (`columnNode`: raw-word
  copy, `0xffffffff` fill for uint32, `0x7fc00000` (NaN) fill for float32 when the row is
  `NO_ROW` or `>= SOURCE_LENGTH`). One helper, used for every left and right column.
- Published: `gpu-uint32-gather.js` (constructor, `invalidValue`), `gpu-gather.js:204-221`
  (`sourceIndex - SOURCE_ROW_OFFSET < SOURCE_LENGTH` else write `invalidValue`).
- Evidence: read both.
- Change: uint32 columns: `new GPUUint32Gather({source, indices: rows, output, invalidValue:
  0xffffffff})`, identical semantics (`NO_ROW` is out of range because `SOURCE_LENGTH < 2^32 - 1`).
  float32 columns: either keep the local node, or gather through `uint32` aliases made with the
  public `graph.createDataView(view.buffer, {format: 'uint32', length, byteOffset})` (33 local uses
  of `createDataView` already) and `invalidValue: 0x7fc00000`. `GPUGather<'float32'>` alone
  would write 0, not NaN, for an unmatched row: that would be a loss.
- Verify: the pair-gather specs under `test/gpu-spatial-analysis/` (left join with unmatched rows).
- Risk: the alias route reinterprets the float buffer, which the local kernel also does
  (`type: 'u32'`); NaN payload stays `0x7fc00000`.

### PD-04 Hand-written two- and three-key LSD sorts repeat what two local helpers already do
- Class: `misuse` (local duplication around `GPUSort`); the luma.gl side is `blocked`, see PD-09.
- Local: sort by the low key, gather the next key by the permutation, sort again, written out by
  hand in `gpu-key-join.ts:405-430`, `group-statistics-sort.ts:67-140` (`getKeyOrderNodes`),
  `gpu-network-noding.ts:480-505`, `gpu-snap-rounded-noding.ts:670-695`,
  `gpu-group-convex-hull.ts:460-500` (three keys), `gpu-line-merge.ts:333-350`,
  `key-pair-grouping.ts:70-100`, `adjacency-matrix-order.ts:120-175`. 8 sites. The shared helpers
  are `analysis-core/kernels/stable-lexicographic-index-sort.ts:46` (packs narrow keys so that,
  for example, two 16-bit keys cost one sort) and
  `gpu-spatial-analysis/areal-interpolation/bounded-key-pair-sort.ts:40-82` (already on
  `GPUUint32Gather`).
- Published: `gpu-sort.d.ts:3-24` (one uint32 key, `keyBits`), no multi-key option.
- Evidence: read both.
- Change: move the 8 sites onto `createStableLexicographicIndexSortNodes` (and make its
  unpacked-key pass and `primary-gather` use `GPUUint32Gather`, `stable-lexicographic-index-sort.ts:93-110`,
  `:128-142`). Where both keys fit in 32 bits together (lattice keys of `LATTICE_KEY_BITS`,
  feature IDs below `featureCount`), the helper's packing removes one full radix sort.
  This is also what contributor contract section 11 asks ("do not copy ... into a contributor
  when `analysis-core` has one").
- Verify: per-site specs listed in PD-01; the order is the same stable lexicographic order.
- Risk: none in order (both are stable LSD). `key-pair-grouping.ts` and `gpu-key-join.ts` return
  the intermediate sorted key views to later kernels; the helper returns only the primary sorted
  key, so a site that reads the sorted low word needs one more gather.

### PD-05 Spatial scan statistic runs a 256-thread hand-written prefix sum over every cell
- Class: `adopt-with-cost`.
- Local: `gpu-spatial-analysis/scan-statistics/gpu-spatial-scan-statistic.ts:941-1000`
  (`weight-sums`: one thread per block sums `CELLS_PER_BLOCK` u32 weights; `weight-scan`: one
  thread scans the block sums; `cdf`: one thread per block writes the running prefix). Blocks are
  capped at `MAXIMUM_BLOCKS = 256` (`:67`, `:547-548`), so 1M cells are scanned by 256 threads in
  serial runs of 4,096.
- Published: `gpu-scan.d.ts:21-55` (hierarchical, 256-thread blocks, wraps modulo 2^32).
- Evidence: read one (local kernels) and the other's documentation.
- Change: one pass writes the u32 cell weights (the `cellWeight(...)` expression) into a
  `cells`-row transient, `GPUScan({mode: 'inclusive'})` writes `cumulative`, and the total row
  is `cumulative[cells - 1]`. 3 nodes become 1 + scan levels.
- Verify: `test/gpu-spatial-analysis/scan-statistics/gpu-spatial-scan-statistic.spec.ts`,
  `parity-wp1-scan.parity.spec.ts`, `stx07-expectation-based-scan.spec.ts`. Integer addition is
  associative and both wrap modulo 2^32, so `cumulative` is bit-identical.
- Risk: cost is one `cells * 4` byte buffer. Only the weight half moves; the float baseline sum
  in `:925-937` is a separate single-thread loop and is not a uint32 scan.

### PD-06 Dot-density triangle CDF is a serial per-feature prefix that `GPUScan` with `segmentFlags` does in parallel
- Class: `adopt-with-cost`; `GPUScan.segmentFlags` is used nowhere in `modules/analysis/src`
  (`git grep segmentFlags` finds only an unrelated ring-assembly field).
- Local: `gpu-spatial-analysis/dot-density/dot-density-sampling.ts:465-483`: one thread per
  feature loops over its triangles and writes the running fixed-point area. A feature with 10^5
  triangles is one thread.
- Published: `gpu-scan.d.ts:29-30` (`segmentFlags`: "nonzero flags that start new segments. Row
  zero is always a segment start"), inclusive mode.
- Evidence: read one and the other's documentation (segmented path of `gpu-scan.js` not read).
- Change: one pass per triangle writes its quantized area (0 for an unusable feature) and a
  segment-start flag (first triangle of a feature), then
  `GPUScan({input, output: cumulativeAreas, mode: 'inclusive', segmentFlags})`.
- Verify: `test/gpu-spatial-analysis/dot-density/gpu-dot-density.spec.ts` (integer CDF, so
  bit-identical expected). Add a case with one large feature.
- Risk: two `triangleCount` u32 buffers. Empty features (no triangles) have no row and need no
  flag. If triangles of different features can interleave (not checked), segment flags do not apply.

### PD-07 Basin sort keeps a `0xffffffff` sentinel and so pays eight radix digits instead of `log16(basinCount)`
- Class: `misuse` (prop not used).
- Local: `gpu-terrain/terrain-processing/gpu-terrain-catchment-summary.ts:223-231` writes
  `0xffffffff` for an excluded cell and sorts without `keyBits`; every reader already treats any
  key `>= basinCount` as excluded (`:247`). Then `:239-249` builds per-basin ranges with a
  hand-written boundary kernel that `createSortedSegmentOffsetsNode` (`analysis-core/kernels/sorted-segment-offsets.ts:26`)
  already does deterministically and without the clear pass.
- Published: `gpu-sort.js:502-531`: one histogram, one `GPUScan` and one scatter per 4-bit digit
  of `keyBits`.
- Evidence: read both.
- Change: write `basinCount` as the exclusion key and pass `keyBits: getSortKeyBits(basinCount)`
  (helper at `analysis-core/kernels/sorted-segment-sums.ts`, already used by
  `gpu-contiguity-weights.ts:640`). For 65,535 basins this is 4 digits instead of 8, about half
  the sort passes. Replace `ranges-initialize` + `ranges` with the shared offsets node (it writes
  `basinCount + 1` offsets instead of begin/end pairs; the statistics kernel reads both ends).
- Verify: catchment-summary spec under `test/gpu-terrain/terrain-processing/`.
- Risk: none in order. Of the other 24 `GPUSort` calls without `keyBits` (list in Coverage), the
  9 read sort float bit keys or ordered keys that need 32 bits (`depression-hierarchy:213`,
  `dem-coregistration:295`, `peak-prominence:176`, `utilization-volume-isopleths:207`,
  `knox-test:419`, `classification-fit:230`, `segment-bvh:123`, `spatial-join-passes:398`, and
  `adjacency-matrix-order:124` with caller tie keys). The rest were not checked: lead.

### PD-08 `spatial-join-passes.ts` collect-match total is `GPUFlagOffsets`
- Class: `adopt-now` (small).
- Local: `gpu-spatial-analysis/spatial-join/spatial-join-passes.ts:1256-1301`: 0/1 flags,
  `GPUScan` exclusive, then a one-thread node writes `offsets[last] + flags[last]` into
  `state[2]`.
- Published: `gpu-flag-offsets.d.ts:3-27`, `gpu-flag-offsets.js:23-62` (the same scan and the
  same one-thread count expression).
- Evidence: read both.
- Change: `new GPUFlagOffsets({flags, offsets, count: graph.createDataView(state.buffer,
  {format: 'uint32', length: 1, byteOffset: state.byteOffset + 8})})`. Same node count. The
  capacity-bounded write kernel at `:1305-1335` stays (`GPUCompaction` needs output capacity >=
  input length, `gpu-compaction.d.ts:47-49`). The other 36 inline totals (Coverage) mostly fold the
  count into a status word or read it inside the consumer kernel, where `GPUFlagOffsets` would add
  a pass; not recommended there.
- Verify: spatial-join specs (`test/gpu-spatial-analysis/spatial-join/`).
- Risk: none; flags are 0/1, so the count cannot wrap.

### PD-09 `GPUSort` has no two-word or multi-key mode, so every 64-bit key pays a gather and a second sort
- Class: `blocked` (not in the register as a `GPUSort` gap; the register names 64-bit keys only
  for `GPURunLengthEncode`, `GPUHashJoin` and `GPUGallopingSearch`).
- Local: the 8 sites of PD-04 plus `stable-lexicographic-index-sort.ts` (14 importing files) and
  `bounded-key-pair-sort.ts`.
- Published: `gpu-sort.d.ts:9-24` (`keys: uint32`, `keyBits <= 32`, `gpu-sort.js:78-80`).
- What luma.gl needs: a `keysHigh` (or `keys: uint32x2`) option on `GPUSort` with `keyBits` up
  to 64, radix over both words. That retires the gather between passes and the second set of
  scratch buffers at every 64-bit call site, and would make PD-04 a one-call change.

### PD-10 Lead: one-line copy kernels against `GPUElementwise('copy')` or a copy node
- Class: `unused-feature` (lead, inferred).
- Local: 14 kernels with a `copy` variant and about 15 one-line `dst[i] = src[i]` bodies, e.g.
  `ward-active-graph.ts:346-352`, `gpu-multiscale-geographically-weighted-regression.ts:376`,
  `:396`, `gpu-study-window-preparation.ts:503-511`, `:669-675`, `gpu-cost-distance.ts:737`,
  `gpu-concave-hull.ts:410`, `gpu-normalize-geometry.ts:146-163`.
- Published: `gpu-elementwise.d.ts:1-28` (`copy`), `createGPUCopyCommandNode`
  (`gpu-command-node.d.ts:22`, a buffer copy with no shader; used once, `gpu-contributor-utils.ts:419`).
- Not checked: whether these copies sit behind a `condition` (a gated node), which a plain copy
  node may not support. A later pass should check before recommending.

### PD-11 Lead: inline scan totals over counts (not flags) are not saturated
- Class: `misuse` (lead, inferred).
- `GPUScan` wraps modulo 2^32 (`gpu-scan.d.ts:36`). `analysis-core/kernels/scan-total.ts`
  saturates the total for that reason (13 files use it). About 20 of the 37 inline
  `offsets[last] + counts[last]` totals sum per-row counts, not flags, without the carry check,
  e.g. `line-split.ts:633`, `:751`, `line-segmentize-kernels.ts:978`, `rectangle-clip-kernels.ts:556`,
  `gpu-isobands.ts:400`, `:524`, `spatial-join-candidates.ts:235`,
  `gpu-segment-intersection.ts:476`, `gpu-nearest-feature-weights.ts:307`. Each needs a proof that
  the counts sum below 2^32 under the class's named limits, or `createSaturatedScanTotalNodes`.
  Not checked site by site.

## 3. Checked and fine

- `GPUCompaction` (22 calls): used where the output holds the whole input (e.g.
  `trajectory-metrics.ts:540-552`, row IDs by run flags). Local flag-scan-scatter sites that do
  not use it either need a bounded capacity (`spatial-join-passes.ts:1330`,
  `spatial-predicate-join.ts:1545`), scatter several columns from one scan
  (`group-convex-hull.ts:686`, `isolines.ts:911`, `cell-compaction.ts:543`) or reuse the scan for
  later layout (`make-valid.ts:496-518`). `GPUCompaction` compacts one uint32 column and runs its
  own scan, so delegating would add one scan per column. Correctly not used.
- `GPUIndexedRangeCompaction`, `GPUPartitionedIndexedRangeCompaction`, `GPUChunkedIndexedScatter`:
  candidate-range and indirect-dispatch scene selection; no local code does this job.
- `GPUMask`: no local mask-composition kernel found in `modules/analysis/src`; masks are fused into
  classify kernels. Used in kepler.
- `GPUBatchSort`, `GPUSegmentedSort`: no `GPUSort` is built in a loop over CPU-known segments
  (`git grep -B6 "new GPUSort("` finds no loop). In-thread insertion sorts (14 sites) sort private
  arrays of at most K entries; `GPUSegmentedSort` sorts storage segments with CPU-known bounds.
  Correctly not used.
- `GPURunLengthEncode`, `GPUUnique`: every local run detection reads two-word keys through a sort
  permutation and excludes a sentinel (`healpix-cell-aggregation.ts:239-262`,
  `key-join.ts:452-470`, `cell-table.ts:280-300`, `contiguity-weights.ts:645-662`,
  `line-split.ts:515-530`). Register row holds. Additional reasons in section 4.
- `GPUSegmentOffsets`, `GPUSegmentedLayout`: need element flags and offsets from separate scans;
  local run tables need no element scan. Not worth it.
- `GPUHashIndex` / `GPUHashIndexQuery` (`flow-aggregation.ts:625-660`): idiomatic, statistics and
  probe bounds passed through; followed by `GPUGroupAggregation` count on representative rows.
  `GPUHashJoin` needs unique right keys (key-to-one-row index, `gpu-hash-join.d.ts:3-5`), so it
  cannot replace `GPUKeyJoin` 1:n; register row holds. `GPUBatchHashIndex`/`GPUBatchHashJoin`: no
  local batched hash code found.
- `GPUGroupAggregation` (24 calls): counts are integer atomics (exact). Float `sum` is used only
  behind an opt-in `sumOrder: 'atomic'` (`spatial-clustering.ts:724-755`, flow aggregation,
  point density) with the exact sorted path as default. One exception: `network-service-areas.ts:483`
  (`facilityCostSums`) sums float costs with the order-dependent atomic and its prop documentation
  (`:143-144`) does not say so. Not a luma.gl issue; contract section 11 wants such sums documented
  or exact.
- `space-time-hot-spots-recipe.ts:579-596`: hand-written atomic count into a dense cube is a fused
  `GPUGroupAggregation('count')`; delegating would materialize the key and mask views that the
  fused kernel computes inline. Fine as is.
- `GPUReduction` (62 calls): `extent` used where both bounds are needed; `max`/`sum` on packed
  scalars. The local 2D bounds reductions (`segment-bvh.ts:76`, `spatial-join-passes.ts:326`,
  `gwr-grid-index.ts:87`, `:112`, `dorling-cartogram.ts:353`) reduce values derived inline from
  interleaved or computed coordinates; `GPUReduction` takes packed scalar views only
  (`gpu-reduction.d.ts:15-29`) and would need two materialized columns plus a validity mask.
- `GPUHistogram` (3 calls): `point-density.ts:687-706` reuses a published extent as a GPU domain
  (correct; `'auto'` would compute the extent a second time); `network-reachability.ts:739`
  literal domain. Local histograms are radix-select digit histograms
  (`column-quantiles-nodes.ts:300`) or pair-distance histograms computed inline (variogram,
  correlogram, Ripley, Knox); the raster FDR mask delegates to `GPURasterHistogram`.
- `GPUScanUint64` (2 calls): correct for 64-bit prefix sums. `scan-total.ts` saturation is one
  pass instead of a 64-bit scan; keep it.
- `GPUCOOToCSR` (`sparse-resistance-system.ts:206`): correct use. It always copies
  `columnIndices` and `values` into its outputs with two `GPUElementwise` passes
  (`gpu-coo-to-csr.js:45-58`), so the entry kernel at `:180-200` writes data that is copied once
  more: cost of the published API, not misuse.
- `GPUGallopingSearch`: needs a query view and segment records; local searches are lower bounds
  inside fused kernels (`binary-search-wgsl.ts`, 3 importing files). Not worth it.
- `load-balanced-for-each.ts`: no published merge-path CSR for-each over command-graph views.
  `GPUGraphSpMV` is a semantic `GPUGraph` operation (`index.d.ts:155`) and `GPUAdaptiveSpMV` is
  not exported. Keep.
- `multiple-testing-adjustment.ts:500-560`: hand-written min/max scan; `GPUScan` is a uint32 sum
  only. Keep.
- `network-k-function.ts:600-620`: 128-edge chunk prefix is part of a deliberate quantization
  scheme bounded below 2^31. Keep.

## 4. Register corrections (`dev-docs/upstream-workarounds.md`)

1. Section 4, row `GPURunLengthEncode`, `GPUUnique` (line 110): "Two trajectory contributors
   delegate" is wrong. No file under `modules/*/src` imports either class (usage: examples only,
   `examples/deck/spatial-analysis-showcase/src/scenes/points/point-patterns-neighbours.ts:208`).
   `gpu-trajectory-metrics.ts:225` and `gpu-trajectory-staypoints.ts:199` only mention it in
   documentation as advice to callers. Add two reasons read in `gpu-run-length-encode.js:124-158`:
   run lengths are counted with one `atomicAdd` per input row into the run's slot (a long run
   serializes), and no run-start offsets are published (the local run tables need them).
   `GPUUnique` is an empty subclass (`:168-169`), so it still requires and computes `lengths`.
2. Section 4, row `GPUGallopingSearch`, `GPUSegmentOffsets` (line 100): add `GPUCOOToCSR` as the
   closest published equivalent. Its offsets kernel (`gpu-coo-to-csr.js:59-80`) is the same
   per-row lower bound over sorted row keys, one dispatch, `rows + 1` outputs, exactly what
   `createSortedSegmentOffsetsNode` computes (read both: identical comparisons, identical results).
   Delegation costs an nnz-length `columnIndices` and a float32 `values` input that the callers do
   not have, and two copy passes; still not worth doing. The "7 callers" counts direct callers only;
   12 files use it (7 direct, 5 through `getSortedSegmentOffsetNodes`). The register's "Delegation
   is possible today at the cost of extra passes and buffers" holds and is not worth doing anywhere
   read in this pass.
3. Section 2, row "Segmented sum (`GPUSegmentedReduction` fast path)" (line 47): still correct at
   alpha.13 (`GPUSegmentedReduction` and `GPUSegmentedScan` are in `dist/gpu-core` but not exported
   from `index.d.ts`). Note in the row that the local gather in `sorted-segment-sums.ts` can be
   `GPUGather` today (PD-02).
4. Section 4, row `GPUCompaction` (line 114): still correct (`gpu-compaction.d.ts:47-49`, output
   must hold the whole input). `GPUFlagOffsets` plus the local bounded scatter is the closest
   published route (PD-08).
5. Not in the register and should be: `GPUSort` has no 64-bit or multi-key mode (PD-09).
