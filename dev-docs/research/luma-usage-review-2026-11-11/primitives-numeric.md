# Stream `primitives-numeric` (PN): numeric and spatial-index primitives vs `modules/analysis/src`

Installed release `@luma.gl/*@10.0.0-alpha.13`. Read-only; no GPU runs. One CPU float32 model was
run with `node` (script: `scratchpad/review/primitives-numeric/grid-query-edge.mjs`).
Published paths below are relative to `node_modules/@luma.gl/graph/dist/`.

## 1. Coverage

Read (published `.d.ts`, and the `.js` kernel where a comparison needed it):
`gpu-core/gpu-matmul`, `gpu-matvec`, `gpu-transpose`, `gpu-fft1d`, `gpu-fft2d`, `gpu-convolution`,
`gpu-finite-difference-2d`, `gpu-grid-index`, `gpu-grid-index-query` (`.js` lines 40-340),
`gpu-bvh-query` (`.js` dispatch), `gpu-segmented-bvh`, `gpu-grid-binning` (bounds type),
`gpu-semantic-spmv`, `gpu-graph-compiler`, `gpu-conjugate-gradient-program` (`.d.ts` + `.js`),
`gpu-ancestor-projection`, `gpu-hierarchy-layout`, `gpu-graph-traversal`;
`geospatial/gpu-pairwise-point-segment-distance` (`.js` 98-112), `gpu-pairwise-point-linestring-nearest`
(`.js` kernel), `gpu-pairwise-point-distance`, `gpu-sinusoidal-projection`, `gpu-haversine-distance`
(`.js` 60-75); `gpu-vector-search/gpu-similarity-search` (`.js` validation and scoring 460-715),
`gpu-ivf-flat-index`, `embedding-matrix`; `gpu-timeseries/*`.

Local code read (whole kernel or the relevant pass): `dataframe-analysis/cube-layout/gpu-cube-layout-transpose.ts`;
`gpu-spatial-analysis/region-statistics/region-statistics-grid.ts` (1-380); `shared/grid-cell-row-order.ts`;
`shared/point-neighbor-grid-index.ts`; `spatial-regression/gwr-grid-index.ts` (100-170);
`dorling-cartogram/gpu-dorling-cartogram.ts` (400-470); `dataframe-analysis/pair-statistics/pair-statistics-grid.ts`
(doc comments and structure); `raster-analysis/raster-embeddings/gpu-raster-embedding-knn-classifier.ts` (all),
`gpu-raster-embedding-nearest-pixels.ts`, `raster-embedding-common.ts:110`; `spatial-interpolation/conditional-simulation-blocked.ts`
(400-520); `spatial-regression/gpu-ordinary-least-squares.ts` (class doc and layout);
`dataframe-analysis/composite-indicators/gpu-composite-score.ts` (doc); `raster-time-series/raster-snapshot-eof-decomposition.ts` (1-80);
`spatial-join/nearest-wgsl.ts` (70-180); `spatial-join/gpu-nearest-feature-join.ts` (945-1020);
`linear-referencing/linear-referencing-kernels.ts` (project node), `gpu-line-locate.ts`; class docs of
`neighbor-search/gpu-neighbor-search.ts`, `vertex-snap`, `minimum-clearance`, `line-density`, `track-similarity`,
`similar-locations`, `optimal-transport/gpu-sinkhorn-divergence.ts`, `space-time-statistics/gpu-space-time-cross-correlation.ts`,
`point-density/gpu-kernel-density.ts` (binned method), `gpu-point-density.ts` (smoothing), `gpu-terrain/terrain-illumination/gpu-texture-shading.ts`,
`raster-analysis/lagrangian-diagnostics/gpu-finite-time-lyapunov-exponent.ts`, `remote-sensing/gpu-raster-registration.ts`,
`raster-masked-registration.ts`, `location-allocation/location-allocation-sparse.ts` (170-260),
`spatial-clustering/hierarchical-density-clustering-kernels.ts` (union-find), `calendar-buckets` doc.

Counted (git grep): 46 files with a CSR row loop `for (var k = offsets[...]` (spatial lag and similar);
53 files that mention "brute", "all pairs" or `O(n^2)` (list in
`scratchpad/review/pn-allpairs-files.txt`); 26 analysis files mention `GPUGridIndex`, 7 build one directly.

Not done (a later pass can pick these up): no per-kernel read of the 89 `spatial-regression` files beyond OLS
(GWR/MGWR local solves were judged from structure: per-location p x p systems, nothing published);
`spatial-weights`, `spatial-autocorrelation` and `point-process` were sampled only through the CSR-loop and
all-pairs counts, not read kernel by kernel; `gpu-network` sparse kernels other than the CG note; `modules/lenses`,
`modules/kepler`, website and examples (out of this stream's scope). The `GPUBVHQuery` node-bound
conservativeness was not modelled (only `GPUGridIndexQuery`'s was).

## 2. Findings

### PN-01 `GPUCubeLayoutTranspose` is a hand-written dense transpose; delegate its kernel to `GPUTranspose`
- Class: `adopt-now`. Evidence: read both.
- Local: `modules/analysis/src/dataframe-analysis/cube-layout/gpu-cube-layout-transpose.ts:109-131`
  (one scatter thread per element, binding type `f32` for float32 cubes). Published:
  `gpu-core/gpu-transpose.d.ts` (`GPUTranspose`, `uint32 | sint32 | float32`, rows/columns, chunked views),
  kernel `gpu-core/gpu-transpose.js:140-142` (binds input and output as `array<u32>`, 16x16 workgroup tile).
- Change: keep the public class, its validation and its storage-binding check; replace the one
  `createWGSLKernelNode` with `new GPUTranspose({input: values, output, rows, columns}).getCommandNodes(graph)`
  where `from: 'cell-major'` is `rows = cellCount, columns = periodCount` and `'period-major'` the reverse.
  One kernel, one file; constructions elsewhere (1 doc, 2 website, 5 specs) are unaffected.
- Gain: the class doc (`:55`) promises a bit-for-bit copy "NaN payloads ... alike", but the local kernel moves
  float32 values through an `f32` binding, where WGSL does not guarantee NaN payloads survive; the published
  kernel moves `u32` words, so the promise becomes true. Coalesced tiled writes instead of strided scatter.
- Loss: the local kernel uses `override` sizes so all cube shapes share one shader module; the published one
  bakes `ROWS`/`COLUMNS`, so each shape compiles its own pipeline (compile cost only).
- Verify: `modules/analysis/test/gpu-dataframe/cube-layout/gpu-cube-layout-transpose.spec.ts` (headless) and
  `.node.spec.ts`; add a float32 cube with a signalling-NaN payload and assert the words.
- Risk: none on values (pure copy); the node IDs change (lens records that key on them).

### PN-02 `GPUGridIndexQuery` is unused for good reasons, and its cell test can miss a point inside the query
- Class: `blocked` (published defect plus missing features; not in the register). Evidence: read both;
  CPU float32 model of the published arithmetic (not run on a GPU; WGSL division is allowed 2.5 ULP, so exact
  numbers will differ but the mechanism does not).
- Why nothing uses it: (a) one query per instance (`gpu-core/gpu-grid-index-query.js:48-54` requires exactly
  `dimension`, `2*dimension` or `dimension+1` floats), so none of the batched per-row neighbor loops can use it;
  (b) its gather dispatches one invocation per indexed object with a binary search over cells
  (`.js:119-219`), O(indexed objects) per query, where the one local single-query gather is
  O(gridHeight + candidateCapacity) (`modules/analysis/src/gpu-spatial-analysis/region-statistics/region-statistics-grid.ts:146-260`,
  contract at `region-statistics-types.ts:216`); (c) output order is atomic-append (`.d.ts:37-40`), while
  region statistics promises IDs bit-identical to the brute-force path (`region-statistics-types.ts:205-209`).
- Defect: objects are placed by `getCoordinate` (division form) but cells are selected with
  `cellMinimum/cellMaximum = minimum*(1-ratio) + maximum*ratio` (`.js:171-180`). These do not agree at cell
  edges. In the float32 model, 1,422 of 53,975 values probed within 8 ULP of a cell edge (8 domains) are
  binned into a cell whose computed bounds exclude them. Example: domain `[-180, 180]`, 360 columns,
  x = -95.0000076 is binned into column 85 whose bounds are `[-95, -94]`; a `bounds` query `[-100, -95.000004]`
  contains x but rejects column 85 and selects column 84, so the point is not returned. The class documents
  its results as conservative (`.d.ts:37`). The local gather avoids this with its one-cell margin
  (`region-statistics-grid.ts:188-190`), which is a reason to keep it.
- What luma.gl needs: cell selection computed with the same `getCoordinate` as the build (or a one-cell
  margin), an O(candidates) range gather, and an ordered (cell, then slot) output option. Report upstream.
- Verify: a headless parity spec that builds `GPUGridIndex` on points one ULP below cell edges and runs
  `GPUGridIndexQuery` with a box ending between the point and the edge.

### PN-03 `GPUGridIndex` literal bounds and unordered cells cost three local workarounds that the register does not list
- Class: `blocked` (register omission). Evidence: read both (`gpu-core/gpu-grid-index.d.ts:24,39`;
  `gpu-core/gpu-grid-binning.d.ts:4` for the contrast).
- `GPUGridIndex` takes literal bounds only and leaves in-cell order unspecified. Local consequences:
  1. Unit-domain normalisation pass so a per-frame extent can use a literal `[0, 1]` domain: 5 files
     (`shared/point-neighbor-grid-index.ts:24-28` `POINT_NEIGHBOR_GRID_UNIT_POSITION_WGSL`,
     `neighbor-search/gpu-neighbor-search.ts`, `spatial-clustering/spatial-clustering-kernels.ts`,
     `spatial-regression/gwr-grid-index.ts:131-160`, `dorling-cartogram/gpu-dorling-cartogram.ts:402-414`).
     It adds a full pass and a second rounding of every position.
  2. Order restoration after the atomic scatter: `shared/grid-cell-row-order.ts:39-120` binary-searches every
     slot back to its cell and radix-sorts all rows by cell (7 contributors: neighbor search, KDE,
     adaptive KDE, space-time KDE, spatial clustering, IDW, GWR). The published scatter is then pure overhead:
     only its cell mapping and offsets survive. Dorling repeats this with its own per-cell shell sort
     (`gpu-dorling-cartogram.ts:421-447`; local duplication, could use `createGridCellRowOrderNodes`).
  3. `dataframe-analysis/pair-statistics/pair-statistics-grid.ts` (used by 13 files: Ripley, variograms,
     correlogram, K functions, space-time tests) builds its own index (cell key, `GPUScan`, stable `GPUSort`)
     and never uses `GPUGridIndex`, because its lattice is derived per frame from bounds and maximum distance
     (`:183-191`).
- What luma.gl needs: GPU-resident bounds on `GPUGridIndex` (as `GPUGridBinningBounds` already allows) and an
  option for ascending source order inside a cell (or a per-row cell-key output). Add a section 2 row naming
  the three workarounds.
- Risk if left: none on results; cost is one extra pass plus one full radix sort per encoding in 7 contributors.

### PN-04 `GPURasterEmbeddingKNearestClassifier` hand-writes the top-k scan of `GPUSimilaritySearch`
- Class: `adopt-with-cost`. Evidence: read both.
- Local: `modules/analysis/src/raster-analysis/raster-embeddings/gpu-raster-embedding-knn-classifier.ts:161-235`
  (one thread per pixel scans every training vector, keeps k in private arrays, then votes). Published:
  `gpu-vector-search/gpu-similarity-search.d.ts` (exact top-k, ties to lower source row, non-finite rows
  excluded, dataset sharded at the binding limit), scoring `gpu-similarity-search.js:643-713`.
- Change: `GPUSimilaritySearch({dataset: training vectors, queries: pixel vectors with the pixel mask as chunk
  validity, k: neighborCount, metric: 'squared-euclidean' | 'cosine'})` into transient ids/scores, then a vote
  kernel that reads `outputIds`, `resultCounts` and the pixel validity. One contributor.
- Gains: the 4,096 training-vector cap (`:33`) goes away (published tiles the dataset); the published cosine
  pre-scales both vectors (`.js:666`) where the local one forms `squaredLength * sampleLength`, which overflows
  for large norms; a local edge case goes away: when fewer than k training rows are finite, the unused slots
  keep `nearestSample = 0` (`:196`) and vote for sample 0's class; the published `resultCounts` makes that visible.
- Loss: a transient of `pixelCount * k * 8` bytes (64 MiB at one million pixels and k = 16) and one extra pass;
  the published search also scores masked pixels (no per-query mask short of the query-major
  `queryFilterMask`), which the vote kernel then discards.
- Verify: `modules/analysis/test/gpu-raster/raster-embeddings/rts03-embedding-rasters.spec.ts`, the kernel-audit
  cases in `modules/analysis/test/analysis-core/kernel-audit/cases-rts-01-03-embedding-fdr.ts` and the table audit
  (`table-audit-raster.spec.ts`, which holds the private `votes` table), plus a parity case
  with exact distance ties and a training set holding NaN rows.
- Risk: same order for Euclidean (both sum `(a-b)^2`); cosine values may differ in the last ULP because of
  the pre-scaling; ties keep the lower row in both.

### PN-05 Every `GPUGraph` semantic operation shares the sparse-solver blocker; the public CG has no preconditioner
- Class: `blocked` / register correction. Evidence: read one (published) and the local patterns by count.
- `GPUGraphCompiler.compile` returns its own `GPUCommandGraph` and binds `GPUVectorInput`s
  (`gpu-core/gpu-graph-compiler.d.ts:28-35,59`); nothing public lowers a `GPUGraph` into a caller graph. This
  blocks `GPUGraphSpMV`, `GPUGraphDotProduct`, `GPUGraphVectorMADD` and the alpha.13 `GPUGraphReduction` /
  `GPUGraphGroupAggregation` for the 46 local files that fuse a CSR row loop (`W y` spatial lags, local
  statistics) into their own kernels, not only for the sparse resistance solver of register row 59.
- `createGPUConjugateGradientGraph` is plain CG: its program (`gpu-core/gpu-conjugate-gradient-program.js`)
  has no preconditioner (0 matches for "precondition", "diagonal", "jacobi"). The local solver is Jacobi-PCG
  (`gpu-spatial-analysis/landscape-ecology/sparse-resistance-solver.ts`), so even with lowering the published
  program would converge more slowly on a grounded Laplacian with a wide conductance range.
- Change: in register row 59 say "the public program is unpreconditioned"; add one sentence (section 2 or 4
  intro) that the whole semantic-operation family is blocked by the same lowering gap.

### PN-06 FFT and convolution limits duplicated instead of consulted
- Class: `misuse` (minor). Evidence: read both.
- `MAXIMUM_PERIODIC_REGISTRATION_DIMENSION = 2048` (`modules/analysis/src/raster-analysis/remote-sensing/gpu-raster-registration.ts:41-42`)
  restates the public `GPU_FFT2D_MAX_DIMENSION` (`gpu-core/gpu-fft2d.d.ts`), and its justification (three
  fields in one 128 MiB binding) assumes the default device. `getGPUFFT2DSupport` and `getGPUConvolutionSupport`
  are used nowhere in `modules/*/src`, so a device with a smaller binding limit fails inside the published
  `getCommandNodes` (`gpu-fft2d.js:53-55`, `gpu-convolution.js:83-95`) with the published message.
- Change: import `GPU_FFT2D_MAX_DIMENSION` for the periodic cap; in the `getCommandNodes` of the two
  registration contributors (6 `GPUFFT2D` constructions) and of `GPUPointDensity` smoothing (3 `GPUConvolution`
  constructions) call the support function first and throw with the contributor ID and prop name.
- Risk: none on results.

### PN-07 Local kernels beside a published class that register section 4 does not list
- Class: `blocked` (register omission; each is correctly local today). Evidence: read both unless noted.
  - `GPUFiniteDifference2D`: Jacobian of `GPUFiniteTimeLyapunovExponent`
    (`raster-analysis/lagrangian-diagnostics/gpu-finite-time-lyapunov-exponent.ts:184-190`): spacing is a
    per-frame parameter and a lost particle (NaN) must propagate; the published class has compile-time
    `spacing` and no no-data rule (same reasons as the terrain rows).
  - `GPUConvolution`: texture shading's Gaussian cascade (`gpu-terrain/terrain-illumination/gpu-texture-shading.ts:201-221`)
    needs a replicate border, normalized convolution and decimated levels; the published boundary is
    `'zero' | 'wrap'`. Binned KDE blur (`point-density/gpu-kernel-density.ts:305-313`) has a per-frame,
    GPU-computed bandwidth.
  - `GPUMatMul`: embedding projection and linear probe (`raster-embeddings/gpu-raster-embedding-projection.ts:95`,
    `gpu-raster-embedding-linear-probe.ts:61`) are fused per-pixel products with centering, NaN masks and a
    softmax; a matmul would add a pixels x outputs transient and lose the masks (read one).
  - `GPUSimilaritySearch`: PN-04. `GPUTranspose`: PN-01. `GPUGridIndex`: PN-03.

### PN-08 Needless cast to `GPUBVH`
- Class: `misuse` (trivial). `modules/analysis/src/gpu-spatial-analysis/spatial-join/gpu-nearest-feature-join.ts:959`
  casts the `NearestBVH` pick (`nearest-nodes.ts:65-74`, which may be prepared storage) to the full `GPUBVH`;
  the probe it feeds takes a `Pick<GPUBVH, ...>` (`spatial-join-passes.ts:873`). Drop the cast. 1 site.

### PN-09 Lead: binary64 point-to-segment distance is published and unused
- Class: `unused-feature` (lead). Evidence: read both.
- `GPUPairwisePointSegmentDistance` and `GPUPairwisePointLinestringNearest` accept raw binary64 positions and
  return double-single distances (`geospatial/gpu-pairwise-point-segment-distance.d.ts`,
  `gpu-pairwise-point-linestring-nearest.d.ts`). Every local point-to-segment function is float32 in a local frame
  (at least 13 WGSL functions: `spatial-join/nearest-wgsl.ts:74-83,162-172`, `spatial-predicate-wgsl.ts:103`,
  `label-point/gpu-label-point.ts:527`, `line-simplification/line-simplification-kernels.ts:121`,
  `polygon-coverage-topology/gpu-coverage-validity.ts:414`, `urban-morphology/urban-ray-utils.ts:145`,
  `spatial-weights/gpu-fuzzy-contiguity.ts:191`, `study-window-wgsl.ts:210`, `gpu-network/network-spatial-context.ts:48,66`).
  The planar float32 formula is identical to the published float32 path (`clamp(dot(p-a, s)/dot(s, s), 0, 1)`,
  `gpu-pairwise-point-segment-distance.js:100-110`). No current caller needs only a distance for aligned rows
  (linear referencing needs fraction, foot, measure and side; nearest joins traverse a BVH), so this is a lead
  for a future "precise distance" output, not a change today.

## 3. Checked and fine

- `GPUMatMul` / `GPUMatVec` vs OLS, regimes, GWR/MGWR, composite score, EOF: the local cross-products are
  centered two-pass moments with masks, fixed-order tile merges and mean correction
  (`gpu-ordinary-least-squares.ts:132-140`, `gpu-composite-score.ts:153-155`); the published kernels are plain
  float32 raw products (`gpu-matmul.js:80-112`, `gpu-matvec.js:70-80`) and would lose accuracy. GWR solves are
  per-location p x p systems. Conditional simulation's tiled `L_qq Z` (`conditional-simulation-blocked.ts:440-482`)
  reads a strided sub-block whose offset (`observedCount`) is decided on the GPU; `GPUMatMul` needs packed,
  CPU-sized matrices.
- Dense solvers and eigen: no public Cholesky, LU, QR, SVD, eigen or least-squares export (searched all entry
  points); the register's statement under section 4 holds.
- `GPUBVHQuery`: one query per instance (point or bounds), unordered; every local BVH traversal is batched
  per row and most are nearest searches (vertex snap, minimum clearance, nearest feature join, spherical
  linear referencing, urban rays), which nothing published does.
- `GPUSegmentedBVH`: segments are CPU-known and at most 128 leaves each; minimum clearance's per-feature
  ranges are data and features can exceed 128 segments.
- `GPUPairwisePointLinestringNearest`: needs contiguous per-row `geometryOffsets`; no local caller pairs a point
  with one known line, and track-similarity's directed Hausdorff uses a cooperative early-exit scan.
- `GPUPairwisePointDistance`: no local planar aligned-pair distance contributor exists.
- `GPUSinusoidalProjection`: cuSpatial's 40,000 km local projection, not an equal-area projection; local code
  uses Lambert equal-area frames. (The dead constant is already deferred, roadmap 12.4.)
- `GPUHaversineDistance`: register row still true at alpha.13 (raw longitude subtraction, `geospatial/gpu-haversine-distance.js:69`).
- `GPUSimilaritySearch`: used correctly by `gpu-raster-embedding-nearest-pixels.ts:80`. `GPUSimilarLocations`
  needs ranks and distances for every row with weighted standardized columns; top-k alone does not serve it.
- `GPUIVFFlatIndex`, `importGPUEmbeddingTable`/`Vector`: no local approximate search; local matrices are built from
  views already in the graph (`raster-embedding-common.ts:110`), which the import helpers do not take.
- `GPUKMeans`: register row stands.
- `GPUTimeSeries`, `GPUTimeSeriesEnvelope`: one uniform series into display bins; nothing in `modules/analysis/src`
  does that (calendar buckets decode timestamps; trajectory and raster temporal resampling are per-track or
  per-pixel and irregular).
- `GPUFFT1D`: no local DFT. Harmonic regression and HANTS fit irregular dates; lagged and space-time
  cross-correlation are pairwise-complete per lag; Sinkhorn is a log-semiring separable pass.
- `GPUFFT2D` (registration) and `GPUConvolution` (point-density smoothing): used, apart from PN-06.
- `GPUFiniteDifference3D`: no 3D stencil in the scope.
- `GPUTranspose` in `gpu-terrain/topographic-position/terrain-summed-area-table.ts:322`: correct. Sparse CSR
  transposes (`location-allocation-sparse.ts:173-258`, global statistics) are sort plus the registered segment
  offsets; `GPUCOOToCSR` would only replace the offsets and gather and lacks the excluded-key rule.
- `GPUAncestorProjection`, `GPUHierarchyLayout`: HDBSCAN and REDCAP use union-find and dendrogram arrays whose
  depth is O(n); no expand/collapse layout in the scope.
- `GPUPointSpatialFilter`: used correctly as the exact predicate on gathered candidates (`region-statistics-grid.ts:346`).
- `GPUGraphTraversal`, `GPUGridBinning`, `GPUGridAggregation`, `GPUPairwisePointInPolygon`, `GPUPointSpatialQuery`:
  register rows still hold.
- Brute-force paths (53 files mention them): about 14 contributors keep an explicit O(n^2) path. Most have
  a grid or BVH alternative already (OPTICS, local kriging, GWR, Dorling, pair statistics, urban rays,
  point-cloud neighborhoods); those without one are pair statistics by definition or capped reference tiers
  (inhomogeneous point pattern, spatial cross-validation buffer exclusion, Mantel, network inhomogeneous K,
  small-hull minimum bounds, make-valid face probe, embedding kNN classifier). None is a single-query
  workload, so no published query class applies; spatial cross-validation could use the local
  `GPUNeighborSearch` (local work, not luma.gl).

## 4. Register corrections (`dev-docs/upstream-workarounds.md`)

- Row 57 ("Primitive signature changes") says call sites use "`GPUGridIndex` `boundsBuffer`". At alpha.13
  `GPUGridIndex` has no such prop: `bounds: GPUGridIndexBounds` is a literal tuple (`gpu-core/gpu-grid-index.d.ts:24`).
  GPU-resident bounds exist only on `GPUGridBinning` / `GPUGridAggregation` (`gpu-grid-binning.d.ts:4`).
- Row 59 (sparse grounded-Laplacian solve): add that the public `createGPUConjugateGradientGraph` is
  unpreconditioned, so lowering alone would not match the local Jacobi-PCG (PN-05).
- Section 2: add a row for the `GPUGridIndex` workarounds of PN-03 (unit-domain pass, cell-order re-sort,
  pair-statistics' own index).
- Section 4: add the rows of PN-07, and PN-01/PN-04 until they are delegated.
- Not in the register at all: the `GPUGridIndexQuery` edge defect (PN-02), for the upstream proposal.
