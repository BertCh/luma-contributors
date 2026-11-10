# Library review: raster analysis and remote sensing (2026-11-09)

Scope: `modules/analysis/src/raster-analysis` (96 files, 25,086 lines), `analysis-core/raster-grid.ts`
(299 lines) and `gpu-spatial-analysis/spatial-interpolation/gpu-focal-statistics.ts` (337 lines).
Paths below are relative to `modules/analysis/src`. Published reference: `@luma.gl/graph/gpu-raster`
at `10.0.0-alpha.10` (`node_modules/@luma.gl/graph/dist/gpu-raster/index.d.ts`).

## 1. Scope and method

Read in full (kernel and TypeScript): `analysis-core/raster-grid.ts`, `raster-grid/gpu-raster-grid.ts`,
`raster-band-utils.ts`, `raster-expansion-utils.ts`, `gpu-raster-fill-nodata.ts`,
`gpu-raster-rank-filter.ts`, `gpu-raster-reconstruction.ts`, `gpu-raster-skeleton.ts`,
`raster-medial-axis.ts`, `gpu-raster-watershed-segmentation.ts`, `raster-watershed-forest.ts`,
`gpu-raster-walking-cost.ts`, `gpu-raster-reprojection.ts`, `gpu-raster-tile-reprojection.ts`,
`gpu-raster-spectral-indices.ts`, `gpu-raster-fractional-coverage.ts`, all of
`remote-sensing/` except `seasonal-break-extended.ts` and the body of `raster-masked-registration.ts`,
`raster-algebra/gpu-raster-arithmetic.ts`, `raster-algebra/raster-algebra-utils.ts`. About 4,600 lines.

Skimmed (class TSDoc, props, limits, every `invocationCount: 1` node, grep for loops and asserts):
`gpu-raster-fractional-zonal-distribution.ts` (lines 60-310), `gpu-raster-fractional-categorical-statistics.ts`,
`remote-sensing/raster-masked-registration.ts`, `cost-distance/*` (props, header, `raster-relaxation.ts`
lines 1-120, `raster-grid-utils.ts`), `distance-field/gpu-distance-field.ts` and parameters,
`isolines/gpu-isolines.ts`, `gpu-isobands.ts`, `gpu-isoband-rings.ts`, `isolines-parameters.ts`,
`flow-texture/*`, `particle-advection/gpu-particle-advection.ts`, `raster-zonal-statistics/*`,
`raster-stretch/gpu-raster-stretch.ts`, `change-detection/gpu-change-detection.ts`,
`polygon-rasterization/*`, `raster-pyramid/*`, `raster-patches/*`, `raster-sampling/*`,
`focal-pairs/*`, the other four `raster-algebra` contributors, `gpu-focal-statistics.ts` (lines 1-100
and the window loop).

Not opened: `remote-sensing/seasonal-break-extended.ts` (beyond grep), `distance-field/distance-field-kernels.ts`
bodies, `isolines/isobands-wgsl.ts`, `isolines/isolines-utils.ts`, `raster-sampling/gpu-raster-profile-scan.ts`,
`vector-field-common/vector-field-sampling.ts`, most `*-parameters.ts` packers, all tests.
Findings on skimmed files are tagged by what was actually read.

## 2. Contributor table

Abbreviations. Par: row / coop / serial / cpu. Params: `buf` = per-frame `parameters` float32 view
with a `get*ParameterValues` packer (the raster family's own per-frame convention, not
`GPUParameterBuffer`); `baked` = WGSL literals. Inputs: every contributor here takes packed views
only. Ctx: `grid` = `GPURasterMetadata`; `grid+csm` = also `cellSizeMode`; `count` = `cellCount` or
`width/height`, no grid. Val: `msg` = helpers and `throw new Error(message)`; `bare` = bare `assert(`.

| Symbol | File (lines) | Computes (reference) | Par | Work | Iteration | Hard limits | Params | Ctx | Status | Val | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `GPURasterRankFilter` | gpu-raster-rank-filter.ts (110) | median / quantile filter (ndimage `rank_filter`) | row | O(N·k⁴) insertion sort per cell | none | radius ≤ 8 :52 design | baked radius, quantile | grid | none | bare | move (upstream image op) |
| `GPURasterFillNoData` | gpu-raster-fill-nodata.ts (179) | nearest / IDW / GDAL 4-quadrant fill (`gdal_fillnodata`) | row | O(missing·d²), d ≤ 256 brute window | none | d ≤ 256 :76 design | baked | grid | ad hoc `unresolvedCount` | bare | rework |
| `GPURasterReconstruction` | gpu-raster-reconstruction.ts (218) | grayscale reconstruction (skimage `reconstruction`) | row + serial check | O(N·iter) | unrolled, no gate | default 64 rounds :89, no max | baked | grid (pixels) | `converged` only | bare | rework, then move |
| `GPURasterFillHoles` | same | binary hole fill (ndimage `binary_fill_holes`) | row | via reconstruction | unrolled, no gate | same | baked | grid | `converged` | bare | rework |
| `GPURasterSkeleton` | gpu-raster-skeleton.ts (165) | Zhang-Suen thinning (skimage `skeletonize`) | row + serial check | O(N·iter) | unrolled 2×(iter+1), no gate | default 64 :71, no max | baked | grid | `converged` | bare | rework, then move |
| `GPURasterMedialAxis` | gpu-raster-skeleton.ts, raster-medial-axis.ts (149) | maximal-disk medial axis (skimage `medial_axis`) | row | O(N) on exact EDT | fixed passes | 32,768 per side :152 inherent | baked | grid (pixels) | none | bare | align (move candidate) |
| `GPURasterWatershedSegmentation` | gpu-raster-watershed-segmentation.ts (249), raster-watershed-forest.ts (282) | marker watershed (skimage `watershed`) | FIFO: serial; MSF: row | FIFO O(N log N) one thread; MSF O(N log² N) | FIFO none; MSF Boruvka with GPU gate | FIFO ≤ 16,384 cells :89 design; MSF < 2³⁰ :85 inherent | baked | grid | `converged` | bare | align (drop or demote FIFO) |
| `GPURasterWalkingCost` | gpu-raster-walking-cost.ts (185) | Tobler / GRASS `r.walk` anisotropic cost | row + serial check | O(N·iter) Jacobi | unrolled, no gate | default 128 :131, no max | baked affine, model | grid (affine) | `converged` | bare | merge into `GPUCostDistance` |
| `GPURasterReprojection` | gpu-raster-reprojection.ts (165) | `gdalwarp` nearest / bilinear | row | O(N_dst) | none | float32 coordinates | baked affine | grid ×2 + `CompiledProjection` | none | bare | align |
| `GPURasterTileReprojection` | gpu-raster-tile-reprojection.ts (143) | mosaic then warp | row | O(N_src + N_dst) | none | mosaic must fit one binding | baked | grid | none | bare | align (build on published tile cache) |
| `GPURasterConservativeRemapping` | remote-sensing/gpu-raster-conservative-remapping.ts (194) | area-weighted regrid (xESMF conservative) | row | O(N_dst·c), c ≤ 256 | none | c ≤ 256 :86 design | baked | own `GPURasterRectilinearGrid` | per-cell code | bare | merge (into reprojection as `average`) |
| `GPURasterFractionalCoverage` | gpu-raster-fractional-coverage.ts (260) | exact cell coverage (exactextract) | row + serial compaction | O(Z·N·T) dense, brute | none | Z·N ≤ 4,194,304 :69 design | baked | grid (affine) | count/requiredCount/overflow | bare | rework |
| `GPURasterFractionalZonalStatistics` | same | coverage-weighted sum / mean | row per zone | O(Z·tuples) brute | none | none named | baked | grid | per-zone validity | bare | rework |
| `GPURasterFractionalCategoricalStatistics` | gpu-raster-fractional-categorical-statistics.ts (318) | coverage by class (exactextract `frac`, `majority`) | row | sorted zone/category sums | none | Z·(C+1) ≤ 65,535; tuples ≤ 1,048,576 :88-91 design | baked | grid | validity | bare | align |
| `GPURasterFractionalZonalDistribution` | gpu-raster-fractional-zonal-distribution.ts (506) | weighted quantiles, variance (exactextract) | row; per-zone serial scans | stable lexicographic sort, then O(tuples) | none | tuples ≤ 65,536, zones ≤ 65,536, probabilities ≤ 32 :115-120 design | buf (probabilities) | grid | validity | bare | align |
| `GPURasterNDWI`, `NBR`, `EVI`, `SAVI` | gpu-raster-spectral-indices.ts (232) | spectral indices on published `GPURasterBandMath` | row | 1-6 passes | none | none | baked (via band scale/offset) | count (width, height) | output validity | msg | align (not a `GPUCommandNodeProducer`) |
| `GPURasterArithmetic` | raster-algebra/gpu-raster-arithmetic.ts (169) | `gdal_calc` binary / unary op | row | O(N) | none | none | buf | count | none | msg | merge (with published `GPURasterBandMath`) |
| `GPURasterConditional`, `GPURasterReclassify`, `GPURasterCellStatistics`, `GPUWeightedOverlay` | raster-algebra/* (182, 264, 324, 359) | `Con`, reclassify, cell statistics, suitability overlay | row (+ coop count/range) | O(N·L), O(N·L²) majority | none | 64 layers (cell stats), 16 layers (overlay) design | buf | count | none / class counts | msg | align |
| `GPUFocalStatistics` | gpu-spatial-analysis/spatial-interpolation/gpu-focal-statistics.ts (337) | `FocalStatistics`, `r.neighbors` | row | O(N·r²) direct gather | none | radius ≤ 64 :29 design | buf | grid (pixels) | none | msg | move to `raster-analysis/focal`; align |
| `GPURasterFocalPairs` | focal-pairs/* (346) | focal covariance, correlation, OLS | row | O(N·r²), three passes | none | radius ≤ 8 design | baked | grid (pixels) | per-cell bits | bare | align |
| `GPURasterTexture` | remote-sensing/gpu-raster-texture.ts (139) | GLCM Haralick (skimage `graycoprops`, OTB) | row | O(N·r² + N·L²) | none | levels ≤ 32, radius ≤ 8 :55-57 design | baked | grid (pixels) | none | bare | align |
| `GPUCostDistance` | cost-distance/gpu-cost-distance.ts (1178) + raster-relaxation.ts (1060) | cost distance, back-links, isochrones (`r.cost`) | coop (16×16 tiles) | O(N·iter), tiled | unrolled with GPU gate and sweeps; continuation | ≤ 1,024 iterations :26 design | buf | grid+csm | `converged`, `iterationCount` | msg | conforming |
| `GPUCostDistancePath` | cost-distance/gpu-cost-distance-path.ts (234) | least-cost path trace | serial | O(path) | none | one target | buf | grid | `converged`-style | msg | conforming (inherent) |
| `GPUDistanceField` | distance-field/* (1544) | exact EDT + allocation (PBA), JFA preview | row (banded) | O(N) | fixed passes | 32,768 per side inherent | buf | grid | none | msg | conforming |
| `GPUIsolines` | isolines/gpu-isolines.ts (1061) | marching squares, stitched polylines | row | O(N·levels) + scan | pointer jumping, gated | capacities | buf | extent in packer | overflow | msg | align (overlaps published `GPURasterContours`) |
| `GPUIsobands`, `GPUIsobandRings` | isolines/* (569, 197) | filled contours, rings | row | O(N·bands) + scan | ring assembly | capacities | buf | extent in packer | overflow | msg | conforming |
| `GPUPolygonRasterization`, `GPURasterJoin` | polygon-rasterization/* (695, 506) | scanline rasterize, raster join | row | O(V log R + crossings + covered) | none | capacities | buf | grid + `spatialContext` | crossing count, overflow | msg | conforming |
| `GPURasterZonalStatistics` | raster-zonal-statistics/* (1214) | zonal count/sum/mean/min/max/variance | row + coop | sorted segmented or atomic | none | zone capacity (device) | buf | grid+csm | overflow | msg | conforming |
| `GPURasterStretch` | raster-stretch/* (899) | linear / percentile / equalize stretch, colormap | row | O(N + bins) | none | extent ≤ 2²⁴ inherent | buf | grid | statistics row | msg | align (overlaps `GPURasterContrast`) |
| `GPUChangeDetection` | change-detection/* (869) | differencing, Welch t, Sen, Mann-Kendall, CVA | row | O(N·T²log T) for Sen | none | T ≤ 256, Sen T ≤ 64 design | buf | count | none | msg | conforming |
| `GPURasterSieve`, `GPURasterPatchMetrics` | raster-patches/* (1136) | `gdal_sieve`, patch area / perimeter | row | O(N) atomics on published components | uses gate state | none | buf | none (labels) | unresolved sentinel | msg | conforming |
| `GPURasterExtremaPyramid` | raster-pyramid/* (615) | min/max mipmap | row | O(N) | fixed levels | first block ≤ 256 | buf | grid | none | msg | conforming |
| `GPURasterSampling`, `GPURasterProfile` | raster-sampling/* (1690) | point extract, profiles | row | O(points), scan | none | 2²⁴ samples per line inherent | buf | grid (packer) | none | msg | conforming |
| `GPUStreamlines`, `GPULineIntegralConvolution`, `GPUParticleAdvection` | flow-texture/*, particle-advection/* (1891) | Jobard-Lefer, LIC, RK2 particles | row | O(seeds·L), O(N·L) | gated rounds | seeds ≤ 65,535, L ≤ 256 | buf | field extent | overflow, iteration limit | msg | conforming; move (visualization) |
| `GPURasterTemporal` | remote-sensing/gpu-raster-temporal.ts (141) | composite, median, Savitzky-Golay | row | O(N·T²) insertion sort | none | T ≤ 256 :52 design | baked | count | per-cell code | bare | merge (with `GPUTemporalReduction`) |
| `GPURasterSpectralUnmixing` | remote-sensing/gpu-raster-spectral-unmixing.ts (180) | linear / NNLS / FCLS unmixing | row | O(N·B·M²) QR per pixel | per-pixel loop ≤ 4,096 | B ≤ 256, M ≤ 8, 8 KiB private :52-65 design | baked | count | per-cell code, iterations | bare | align |
| `GPURasterSeasonalBreak` | remote-sensing/gpu-raster-seasonal-break.ts (212) + extended (188) | one-break harmonic model (BFAST-lite) | row | O(N·T²·k²) | per-pixel loops | T ≤ 256, 8 KiB private :72-115 design | baked | count | per-cell code | bare | align |
| `GPURasterRegistration` | remote-sensing/gpu-raster-registration.ts (335) + masked (406) | phase correlation (skimage `phase_cross_correlation`) | row FFT + serial peak | O(N log N) + serial O(N) argmax | none | periodic ≤ 2048², masked ≤ 512² :129-131 design | baked | grid (pixels) | one code | bare | rework (serial reductions) |
| `GPURasterRegionGraph`, `addRasterRegionGraphTiles` | remote-sensing/gpu-raster-region-graph.ts (322), raster-region-tiles.ts (105) | region adjacency graph, merging (skimage RAG) | row + serial merge | O(R²) per round, dense | unrolled, no gate | regions ≤ 256, rounds ≤ 256 :69-70 design | baked | grid | one "pending" word | bare | rework |
| `makeGPURasterMetadata`, `makeGPURasterBand` | raster-grid/gpu-raster-grid.ts (101) | grid and band builders | - | - | - | - | - | grid | - | msg | conforming |

Verdict count (40 rows, about 55 exported symbols): conforming 12, align 14, rework 8, merge 4,
move 2 (rank filter, focal statistics). Four more rows are also move candidates (reconstruction and
fill holes, skeleton, medial axis, the flow-visualization trio).

## 3. GPU quality findings

**G1. Three contributors run unrolled Jacobi loops with no GPU gate, no node budget and too few
rounds for realistic rasters. CONFIRMED.** `GPURasterWalkingCost` (`gpu-raster-walking-cost.ts:131`,
default 128 rounds), `GPURasterReconstruction` (`gpu-raster-reconstruction.ts:89`, default 64) and
`GPURasterSkeleton` (`gpu-raster-skeleton.ts:71`, 2 × 65 passes) emit one kernel node per round. Every
round always runs. Each round moves the front one cell, so a 1,000 × 1,000 walking-cost grid with one
source needs about 1,000 rounds. With the default, `converged` is 0 and the publish kernel writes NaN
over the whole grid (`gpu-raster-walking-cost.ts:180`). `maximumIterations` has no upper bound
(`:74`, `:55`, `:47`), so the fix-by-prop path unrolls thousands of nodes. `GPURasterFillHoles` inherits
this: any hole farther than 65 cells from the border leaves every pixel `0xfffffffe`.
Better design: the tiled, GPU-gated relaxation that already exists in `cost-distance/raster-relaxation.ts`
(16 × 16 workgroup tiles iterate in shared memory, indirect-dispatch gate, directional sweeps,
continuation). Walking cost is that relaxation with a direction-dependent edge cost. Fill holes is one
pass of published `GPURasterConnectedComponents` on the background plus a border-touch flag.
Effort: M for walking cost (extend `GPUCostDistance` edge-cost hook), S for fill holes, M for
reconstruction and skeleton on the shared gated loop.

**G2. Single-invocation convergence loops over every cell. CONFIRMED.** `gpu-raster-reconstruction.ts:125`,
`gpu-raster-skeleton.ts:101`, `gpu-raster-walking-cost.ts:166`: `for (cell < count)` in one thread. At
16 M cells this is a multi-second dispatch and a watchdog risk. The library already has the parallel
replacement, `createRasterCountNodes` (`remote-sensing/raster-kernel-utils.ts:115-171`, whose comment
says "Parallel replacement for single-thread counting loops"), and `gpu-raster-fill-nodata.ts:133` uses
it. Effort: S.

**G3. Serial reductions in registration. CONFIRMED.** `gpu-raster-registration.ts:234` (max of spectral
products), `:275` and `:277` (two passes for peak and secondary peak) each loop over up to 2048² = 4.2 M
cells in one thread; masked mode does the same at `raster-masked-registration.ts:164` and `:279`.
Better: published `GPUReduction` with an order-preserving packed (value, index) key, as
`GPURasterStretch` already does for min/max. Effort: S.

**G4. Fractional coverage is a dense zone × cell product with a one-thread compaction. CONFIRMED.**
`gpu-raster-fractional-coverage.ts:69` caps `zones × cells` at 4,194,304 (one 2,048 × 2,048 raster and
one zone; 64 zones on 256 × 256). Every pair clips every triangle of its zone (`:140-148`), and one
thread walks all pairs to compact (`:165`). `GPURasterFractionalZonalStatistics` then gives each zone one
thread that scans all tuples (`:240`), O(zones × tuples). Better: one invocation per triangle emits its
bounding-box cells through a scan (the pattern `GPUPolygonRasterization` uses, or
`load-balanced-for-each`), clips, and writes tuples through `GPUCompaction`; aggregation reuses the
stable lexicographic sort and sorted segment sums that `GPURasterFractionalZonalDistribution` already
uses. Effort: M.

**G5. FIFO watershed is one thread. CONFIRMED.** `gpu-raster-watershed-segmentation.ts:137` (dispatch
of 1, binary heap, ≤ 16,384 cells at `:89`, already tracked as R7.2). The same file ships a scalable
rooted Boruvka forest (`raster-watershed-forest.ts`, GPU-gated, log rounds × log pointer jumps). The
FIFO path should become a CPU test oracle or be dropped; the default should be the forest. Effort: S.

**G6. Region graph is a dense R² matrix with a serial merge. CONFIRMED.** `gpu-raster-region-graph.ts:69`
limits regions to 256; adjacency and weights are `R²` (`:78`); the adaptive merge is one thread looping
R² per round (`:289-316`), one merge per round, at most 256 rounds, no gate. Better: emit boundary pairs
as sorted 64-bit keys (published `GPUSort`, then run-length compaction) into a CSR region graph, compute
region means with published `GPURasterRegionMeasurements` (already used by patches), and merge with the
same rooted Boruvka scheme as the watershed forest. Effort: M.

**G7. Per-pixel private-array work that limits occupancy. CONFIRMED (arrays), SUSPECTED (spill).**
`GPURasterTexture` keeps a `levels²` u32 histogram per invocation (up to 4 KiB, `gpu-raster-texture.ts:99`);
`GPURasterRankFilter` keeps `(2r+1)²` floats and insertion-sorts them (`gpu-raster-rank-filter.ts:87-99`);
`GPURasterTemporal`, `GPURasterSeasonalBreak` and `GPURasterSpectralUnmixing` keep up to 8 KiB per
invocation (`gpu-raster-spectral-unmixing.ts:58-65`, `gpu-raster-seasonal-break.ts:71-82`). These scale in
cell count but each thread is heavy; on most GPUs such arrays live in local (spilled) memory. Better:
for rank filters, a per-workgroup tile in shared memory with histogram or partial selection; for GLCM,
a tile-shared quantized patch. Effort: M each; low priority.

**G8. Spectral unmixing refactors the same endmember matrix for every pixel. CONFIRMED.**
`gpu-raster-spectral-unmixing.ts:121-135` builds and QR-factors the `B × M` design per pixel, although the
endmembers are shared. When no band is masked, a single precomputed pseudo-inverse (`M × B`, computed
once) reduces per-pixel work from O(B·M²) to O(B·M). Keep the per-pixel QR only for pixels with masked
bands. The projected-gradient step uses the Frobenius norm as its Lipschitz bound (`:140`), which is
valid but slows convergence; the spectral norm of the 8 × 8 Gram matrix is cheap to compute once.
Effort: S.

**G9. Brute-force fill of no-data. CONFIRMED.** `gpu-raster-fill-nodata.ts:113-123` scans a
`(2d+1)²` window per missing pixel, with d up to 256 (263,169 reads per pixel). `nearest` is exactly
`GPUDistanceField` with `nearestCells` (exact, O(N)). IDW fill is a push-pull pyramid (GDAL-like
results with smoothing) or a bounded window over a JFA candidate set. Effort: S for `nearest`, M for IDW.

**G10. Focal statistics is a direct gather. CONFIRMED.** `gpu-focal-statistics.ts:317-327`: O(r²) reads per
cell, radius up to 64 (16,641 reads). Sum, mean, count and standard deviation on a rectangle are O(1)
per cell with a summed-area table (two published `GPUScan` passes) or two separable box passes
(published `GPURasterBoxBlur`); min and max are published grayscale `GPURasterErosion` / `GPURasterDilation`
(or van Herk/Gil-Werman, O(1) per cell). Keep the gather for circles and custom shapes. Effort: M.

**G11. Recompiles for values that are per-frame in the rest of the domain. CONFIRMED.** Every root-level
file and every remote-sensing file bakes numeric props into WGSL: `quantile`
(`gpu-raster-rank-filter.ts:104`), `maximumDistance` (`gpu-raster-fill-nodata.ts:113`), `frictionWeight`
and GRASS coefficients (`gpu-raster-walking-cost.ts:149-150`), the affine (`gpu-raster-reprojection.ts:119,157`),
`range` (`gpu-raster-texture.ts:98`), `mergeThreshold` (`gpu-raster-region-graph.ts:222`), `period`
(`gpu-raster-seasonal-break.ts:179`). The baseline raster families (21 files) take a per-frame
`parameters` float32 view with a `get*ParameterValues` packer. Effort: S per contributor.

**G12. Float32 exposure in baked affines. CONFIRMED.** `gpu-raster-reprojection.ts:119` writes
`affine[2]`, `affine[5]` (absolute origin) as f32 literals; a UTM northing of 4.6e6 m loses 0.5 m.
`gpu-raster-walking-cost.ts:148` and `gpu-raster-fractional-coverage.ts:133` do the same. The baseline
packers subtract a `localOrigin` in float64 first (`distance-field-parameters.ts`, `raster-sampling-parameters.ts:95-101`).
Effort: S.

## 4. Consistency findings

**C1. Raster representation: five input shapes coexist. CONFIRMED.**

| Shape | Who |
| --- | --- |
| `grid: GPURasterMetadata` + published `GPURasterBand` (buffer or texture, `validity`, `noDataValue`, `scale`, `offset`) canonicalized by `getRasterBufferBandNodes` (`raster-band-utils.ts:175`) | cost distance, zonal statistics, pyramid, focal pairs, all root-level files |
| `grid` + raw `GraphDataView<'float32'>` + optional `validity` | sampling, profile, isolines, isobands, stretch, LIC, focal statistics, texture, region graph, registration, reconstruction, skeleton |
| `cellCount` + raw views, no grid | raster algebra (4), temporal, unmixing, seasonal break, change detection |
| `width`, `height`, no affine | spectral indices |
| own `GPURasterRectilinearGrid` (lower-left origin, `spacing`) | conservative remapping (`gpu-raster-conservative-remapping.ts:23-28`) |

Band layout: published bands are one band per view. Stacks come in two layouts: band-sequential
(`GPURasterCellStatistics`, `GPUWeightedOverlay`: layer `i` at rows `[i·N, (i+1)·N)`) and cell-major
interleaved (`GPUChangeDetection` `(cell·T + slice)·B + band`, `GPURasterTemporal`, `GPURasterSeasonalBreak`
`cell·T + slice`, `GPURasterSpectralUnmixing` `cell·B + band`). Textures are accepted only where the
published band is used (via `GPURasterTextureToBuffer`, `raster-band-utils.ts:201-210`).

**C2. No-data: four input conventions and four output conventions. CONFIRMED.**
Input: (a) published band: validity view, raw sentinel compared before calibration, non-finite rejected;
(b) NaN is always no-data plus an optional finite `noDataValue` sentinel and no validity view
(`raster-algebra-utils.ts:28-36`); (c) NaN plus optional `validity` (remote sensing, sampling, focal
statistics); (d) label sentinels (`0xfffffffe` "unresolved" in skeleton, fill holes, watershed;
`GPU_RASTER_PATCH_UNRESOLVED`; `GPU_COST_DISTANCE_UNRESOLVED`). Output: NaN plus `validity`
(root-level files), NaN plus `outputValidity` (spectral indices, sampling, overlay), NaN plus
`outputMask` (conditional), NaN plus a per-cell status code (temporal, unmixing, seasonal break, remapping).
Three spellings of the quiet-NaN helper: inline `bitcast<f32>(0x7fc00000u | (index & 0u))` (15 files),
`missing()` (10), `getNaN()` (10), plus `missingValue()` and `getQuietNaN()`.
Recommend: the published band contract on input everywhere (it already carries validity, sentinel and
calibration), and `values` + `validity` on output, named `validity`.

**C3. Georeferencing reaches a contributor six ways. CONFIRMED.** No raster contributor takes
`spatialContext` except the two that consume vector geometry (`GPUPolygonRasterization`,
`GPURasterJoin`). The others:
(1) per-frame packer derived from `grid.affine`, with float64 `localOrigin` subtraction (cost distance,
distance field, zonal statistics, sampling, profile); ground meters only through `cellSizeMode`
(`analysis-core/raster-grid.ts:24`), which only `GPUCostDistance` and `GPURasterZonalStatistics` use;
(2) `grid.affine` baked into WGSL (reprojection, walking cost, the four fractional contributors);
(3) a world `extent` passed in the packer settings, independent of `grid.affine`, with only width and
height taken from the grid (`isolines/isolines-parameters.ts:23-44`), so the two can disagree;
(4) pixel units only, affine ignored (rank, fill, focal statistics and pairs, texture, reconstruction,
skeleton, medial axis, watershed, region graph, registration shift);
(5) own rectilinear grid (remapping);
(6) nothing (`cellCount` contributors).
`coordinateReferenceSystem` is read in exactly one place, a JSON string comparison
(`gpu-raster-reprojection.ts:76-77`). Walking cost converts slope from affine units without
`cellSizeMode`, so on a geographic grid it divides metres of rise by degrees of run (SUSPECTED misuse;
nothing rejects it). Recommend: treat `GPURasterMetadata` as the raster's spatial context; add a
`resolveGPURasterContext(id, grid, {cellSizeMode, rowDirection})` in `analysis-core/raster-grid.ts` that
validates axis alignment and returns the packed georeference words; require it wherever a distance,
slope, area or world coordinate is computed.

**C4. File layout. CONFIRMED.** The 11 root-level `gpu-raster-*.ts` files (plus `raster-medial-axis.ts`,
`raster-watershed-forest.ts`) are the coverage-expansion wave: all but spectral indices validate through
`raster-expansion-utils.ts`, use bare `assert`, bake parameters and use `{status: {converged}}`. The 17
subdirectories are the baseline families (each with a `*-parameters.ts` packer, named exports in
`index.ts`, `throw new Error`), except `remote-sensing/` and `focal-pairs/`, which are expansion-style
with a second private helper set (`remote-sensing/raster-kernel-utils.ts`). There is no rule; there are
three validation helper sets (`raster-band-utils`, `raster-expansion-utils`, `raster-kernel-utils`) plus
`raster-algebra-utils`. The root `index.ts` uses `export *` for root files, which publishes internal node
builders `getRasterWatershedForestNodes` and `getRasterMedialAxisNodes` (`index.ts:27,32`; both marked
`@internal`). Recommended layout by map-algebra class: `local/` (algebra, spectral indices),
`focal/` (focal statistics, rank, focal pairs, texture), `zonal/` (zonal statistics, fractional*,
patches), `distance/` (distance field, cost distance, cost path, walking cost), `morphology/`
(reconstruction, skeleton, medial axis, watershed, sieve), `resampling/` (reprojection, tile
reprojection, remapping, sampling, profile, pyramid), `conversion/` (polygon rasterization, join,
isolines), `time-series/` (temporal, change detection, seasonal break), `remote-sensing/`
(unmixing, registration, region graph), `flow-visualization/`, and one `raster-common/` for grid,
band, relaxation and count helpers.

**C5. Tiling and halo are per contributor, not uniform. CONFIRMED.** Published
`GPURasterTile`, `GPURasterTileHaloAssembler`, `GPURasterTileCoreExtract`, `GPURasterTileCache` and
`GPURasterCrossTileComponents` are used nowhere in the package (grep). Three raster files expose
`requiredHalo` (`gpu-raster-texture.ts:46`, `gpu-raster-focal-pairs.ts`, `gpu-isolines.ts`); the terrain
domain exposes it in 16 files. Focal contributors without it: `GPUFocalStatistics`, `GPURasterRankFilter`,
`GPURasterFillNoData`. Global contributors (cost distance, distance field, watershed, reconstruction,
region graph) require a complete domain; `addRasterRegionGraphTiles` and `GPURasterTileReprojection`
assemble tiles into one full-domain buffer first (`raster-region-tiles.ts:72-101`,
`gpu-raster-tile-reprojection.ts:84-129`), so they are not tiled execution. Recommend: every focal
contributor declares `requiredHalo` and runs on the published halo assembler; every global contributor
declares that it needs a complete domain.

**C6. Iterative operators use five schemes. CONFIRMED.**

| Operator | Scheme |
| --- | --- |
| `GPUCostDistance` (and terrain hydrology, isolines linking, streamlines, patch labels) | `raster-relaxation.ts`: tiled shared-memory relaxation, indirect-dispatch gate, sweeps, continuation |
| `GPUDistanceField` | exact banded Felzenszwalb (PBA), fixed passes; JFA preview |
| `GPURasterWatershedSegmentation` (forest) | own Boruvka with `createWGSLDispatchGate` |
| `GPURasterWalkingCost`, `GPURasterReconstruction`, `GPURasterSkeleton`, `GPURasterRegionGraph` | own unrolled ping-pong, no gate |
| `GPURasterFillNoData` | no iteration, bounded window |

The shared relaxation lives in `cost-distance/raster-relaxation.ts` but serves 16 terrain files; it
belongs in `raster-common/` or `analysis-core`.

**C7. Status. CONFIRMED.** None of the 44 files uses a named status port from
`analysis-core/contracts/status.ts`. Baseline families use the vocabulary words loosely
(`converged`, `iterationCount`, `overflow`, `requiredCount`); `GPURasterRegionGraph.output.status` is a
"more merging possible" word (`gpu-raster-region-graph.ts:51`) that should be `iterationLimitReached`;
registration, temporal, unmixing, seasonal break and remapping return integer codes with different
meanings for the same numbers (status 2 is "ambiguous peak" in registration, "rank failure" elsewhere).

**C8. Contract shape. CONFIRMED.** The spectral indices do not implement `GPUCommandNodeProducer`;
they use `addToGraph` (`gpu-raster-spectral-indices.ts:130`). `addRasterRegionGraphTiles` calls
`graph.add` directly instead of returning nodes. `GPURasterReprojection` and `GPURasterTileReprojection`
are single-use (`assert(!this.projection)`, `gpu-raster-reprojection.ts:88`).

**C9. Validation style by family. CONFIRMED.** Bare `assert` (AGENTS.md style): 11 root files,
`focal-pairs`, `remote-sensing` (about 110 calls in scope). Helpers with messages: all baseline families.
Within one contributor the two mix (`gpu-raster-fill-nodata.ts` uses both).

**C10. Two saddle rules for contours. CONFIRMED.** `GPUIsolines` resolves saddles by the cell-centre
average (`gpu-isolines.ts` TSDoc); published `GPURasterContours`, used by `gpu-terrain/terrain-analysis/gpu-terrain-contours.ts`,
uses the bilinear asymptotic decider. The same DEM contoured through the two paths can differ at saddles.

## 5. Duplication and shared-helper candidates

Overlaps with published `@luma.gl/graph/gpu-raster`:

| Local | Published | Recommendation |
| --- | --- | --- |
| `GPURasterArithmetic` | `GPURasterBandMath` (add, subtract, multiply, divide, normalized difference with calibration and validity) | Keep the local one only for its per-frame operation switch and unary operations; take `GPURasterBand` inputs. Propose unary operations and clamp upstream. |
| `GPUFocalStatistics` mean/sum/min/max | `GPURasterNeighborhood`, `GPURasterBoxBlur`, `GPURasterErosion`/`GPURasterDilation` | Build mean, min and max on them; keep standard deviation, range, count. |
| `GPURasterStretch` | `GPURasterContrast`, `GPURasterHistogram`, `GPURasterGlobalPercentile` | Align; the colormap, gamma and sigmoid steps are the local value. |
| `GPUIsolines` | `GPURasterContours`, `GPURasterContourClassifier` | Converge on one saddle rule; the stitched multi-level polylines are the local value. |
| `GPURasterRegionGraph` means | `GPURasterRegionMeasurements` | Use the published measurements. |
| `GPURasterFillHoles` | `GPURasterConnectedComponents` | Rebuild on components. |
| `GPURasterTileReprojection` mosaic, `addRasterRegionGraphTiles` | `GPURasterTileCache`, `GPURasterTileHaloAssembler` | Use the published tile path. |
| `GPURasterExtremaPyramid` | `GPURasterOverview` (mean/mode only) | Justified (comment at `gpu-raster-extrema-pyramid.ts`); propose min/max as an overview mode upstream. |

Belong upstream as generic image processing (no geospatial semantics; AGENTS.md "luma.gl improvements
go upstream"): `GPURasterRankFilter`, `GPURasterReconstruction`, `GPURasterFillHoles`,
`GPURasterSkeleton`, `GPURasterMedialAxis`, the phase-correlation core of `GPURasterRegistration`
(already on published `GPUFFT2D`), and the GLCM core of `GPURasterTexture`. The flow-visualization
trio (`GPUStreamlines`, `GPULineIntegralConvolution`, `GPUParticleAdvection`) is rendering-side and
overlaps the package's own `FlowParticleSimulation`; it is a candidate for luma.gl effects or the kepler
package.

Overlaps inside the package:
- `GPURasterWalkingCost` duplicates `GPUCostDistance` (which lists anisotropic cost as a non-goal).
- `GPURasterTemporal` (mean, min, max, median per cell over T slices) overlaps
  `dataframe-analysis/temporal-reduction/gpu-temporal-reduction.ts`; Mann-Kendall and Sen slope are
  implemented in both `GPUChangeDetection` and `gpu-spatial-analysis/emerging-hot-spots/`.
- `GPURasterFillNoData` `nearest` duplicates `GPUDistanceField` allocation.
- `GPURasterConservativeRemapping` and `GPURasterReprojection` are both grid-to-grid resamplers with
  different grid types.
- `GPURasterFractionalZonalStatistics` (per-zone scan) and `GPURasterFractionalZonalDistribution`
  (sorted) compute overlapping sums two ways.

Helpers defined more than once:
- D8 neighbour lookup: `cost-distance/raster-grid-utils.ts` (`getD8Neighbor`, packed offsets),
  `gpu-cost-distance-path.ts:189-200` (array offsets), `raster-watershed-forest.ts:83-90` (`edgeTarget`),
  and inline loops in walking cost, reconstruction, skeleton, rank filter, fill no-data.
- NaN and finiteness: `isNaNValue` in `raster-sampling-utils.ts`, `raster-algebra-utils.ts:32`,
  `raster-grid-utils.ts`; `finite`/`missing` in `raster-kernel-utils.ts:55-57`; quiet-NaN spellings in C2.
- Packed-view, aliasing and graph-ownership checks: `raster-band-utils.ts:96-136`,
  `raster-expansion-utils.ts:15-36`, `raster-kernel-utils.ts:25-42`, `raster-algebra-utils.ts:84-106`
  (four copies of the same three checks).
- Bilinear sampling: `raster-sampling-utils.ts` (`getRasterSamplingWGSL`, with bicubic),
  `gpu-raster-reprojection.ts:159`, `vector-field-common/vector-field-sampling.ts`.
- Argmax / min-max reduction: done with order-preserving keys in stretch and overlay, serially in
  registration and masked registration.
- Least squares: `getRasterLeastSquaresWGSL` (`raster-kernel-utils.ts:60`) is a reusable per-thread QR;
  it belongs in `analysis-core/kernels` next to the other numerics.

Belong in `analysis-core` or `raster-common`: `raster-relaxation.ts` (used by 16 terrain files),
`createRasterCountNodes`, `getRasterLeastSquaresWGSL`, one D8 helper, one NaN helper, one raster view
validator.

## 6. Comprehensiveness

What a GDAL / rasterio / xarray-spatial / WhiteboxTools / scikit-image / SciPy ndimage / terra / OTB /
Earth Engine user looks for and does not find, ranked by how often it is used. "GPU" is GPU
suitability.

| # | Missing | GPU | Reuses | Effort |
| --- | --- | --- | --- | --- |
| 1 | QA bit-mask decoding (Landsat `QA_PIXEL`, Sentinel-2 SCL, MODIS state) into validity; Earth Engine `bitwiseAnd` masks. The spectral indices say decoding is caller-owned (`gpu-raster-spectral-indices.ts:129`) | trivial, row | published band `validity` | S |
| 2 | Clip / mask a raster by polygons (`gdalwarp -cutline`, `rio mask`, `terra::mask`) | row | `GPUPolygonRasterization` + published band validity | S |
| 3 | Rasterize lines and points with burn values, `all-touched`, and per-cell count/sum/mean/max (`gdal_rasterize`, `terra::rasterize`, xarray-spatial `rasterize`) | row + scan | rasterization boundary pass, `sorted-segment-sums` | M |
| 4 | Polygonize a categorical raster (`gdal_polygonize`, `rasterio.features.shapes`, `terra::as.polygons`) | row + ring assembly | label boundary edges, `GPUSegmentRingAssembly` (as `GPUIsobandRings` does) | M |
| 5 | Zonal median, percentile, majority, minority, variety on raster zones (ArcGIS Zonal Statistics, `exactextract`, Earth Engine `reduceRegions`); listed as a non-goal of `GPURasterZonalStatistics` | sort | stable lexicographic sort, as in fractional distribution | S |
| 6 | Zonal histogram / tabulate area / cross-tabulation of two categorical rasters (`terra::crosstab`, ArcGIS Tabulate Area) | row atomics | composite key histogram | S |
| 7 | Focal majority, minority, variety, percentile at large radius; weighted, annulus and wedge kernels (`r.neighbors`, xarray-spatial `focal`, Earth Engine kernels) | row / tile | rank filter, focal statistics | S to M |
| 8 | Band linear transforms: PCA, tasseled cap, arbitrary matrix (OTB `DimensionalityReduction`, Earth Engine `matrixMultiply`) | row | `GPUCompositeScore` PCA, per-pixel matrix | S |
| 9 | Unsupervised and supervised pixel classification: k-means / ISODATA, minimum distance, maximum likelihood (OTB, Earth Engine `Clusterer`, `Classifier`) | row | `gpu-spatial-analysis/spatial-clustering/gpu-kmeans.ts` on band vectors | M |
| 10 | Euclidean and cost allocation, back-direction (ArcGIS Distance Allocation, `r.cost -n`); cost allocation is listed as an open extension in `GPUCostDistance` | row | relaxation carrying a source ID | S |
| 11 | Expand, shrink, nibble, boundary clean, majority filter (ArcGIS Generalization) | row | distance field allocation, rank filter | S |
| 12 | Block statistics / coarsen with sum, min, max, majority (`terra::aggregate`, xarray `coarsen`) | row | published `GPURasterOverview`, extrema pyramid | S |
| 13 | Histogram matching, pan-sharpening (Brovey, IHS, Gram-Schmidt) (OTB, `rio pansharpen`) | row | stretch histogram and CDF | S each |
| 14 | Raster calculator expressions (`gdal_calc`, `terra::app`, Earth Engine `expression`); `GPURasterArithmetic` is one operation per node | compile to one kernel | arithmetic, conditional | M |
| 15 | Speckle and edge-preserving filters (Lee, Frost, bilateral), Canny, local binary patterns | row | belongs upstream as image processing | M |

Already tracked, not repeated: resampling filters and source-tile routing for reprojection (R-REPROJ);
larger watershed and other segmentation methods (R7.2); FFT filtering (R7.4); CLAHE (R7.1); sparse
zone IDs (RS02); curvilinear remapping (RS03); irregular dates and streaming accumulators past 256
slices (RS06 / TC-07); endmember discovery (RS10); multi-break BFAST (RS11, RTS-04); smoothers and
compositing (RTS-02, RTS-05); seam-safe contours (R6.3); fractional composition envelopes (R6.4).

Out of place in a GPU analysis library: the flow-visualization trio (rendering); `GPURasterWatershedSegmentation`
FIFO mode (a serial oracle shipped as a method); the generic image-processing operators listed in
section 5, which belong in `@luma.gl/graph/gpu-raster`. `GPUFocalStatistics` belongs in
`raster-analysis` (a focal family), not in `gpu-spatial-analysis/spatial-interpolation`; it already
imports `raster-analysis/raster-band-utils` (`gpu-focal-statistics.ts:22`).

Map-algebra matrix (local symbols plus the published ones a user would combine them with):

| Class | Local | Published |
| --- | --- | --- |
| Local | `GPURasterArithmetic`, `GPURasterConditional`, `GPURasterReclassify`, `GPURasterCellStatistics`, `GPUWeightedOverlay`, NDWI / NBR / EVI / SAVI, `GPURasterStretch` (apply), `GPURasterTemporal`, `GPUChangeDetection`, `GPURasterSeasonalBreak`, `GPURasterSpectralUnmixing` | `GPURasterBandMath`, `GPURasterNDVI`, `GPURasterThreshold` |
| Focal | `GPUFocalStatistics` (mean, sum, min, max, range, std, count), `GPURasterRankFilter`, `GPURasterFocalPairs`, `GPURasterTexture` | `GPURasterNeighborhood`, `GPURasterConvolution`, Gaussian and box blur, Sobel / Scharr / Laplacian / gradient, morphology |
| Zonal | `GPURasterZonalStatistics`, four fractional contributors, `GPURasterPatchMetrics`, `GPURasterSieve` | `GPURasterRegionMeasurements`, connected components |
| Global | `GPUDistanceField`, `GPUCostDistance`, `GPUCostDistancePath`, `GPURasterWalkingCost`, `GPURasterWatershedSegmentation`, `GPURasterReconstruction`, `GPURasterFillHoles`, `GPURasterSkeleton`, `GPURasterMedialAxis`, `GPURasterFillNoData`, `GPURasterExtremaPyramid`, `GPURasterRegionGraph`, `GPURasterRegistration` | global statistics, histogram, percentile, Otsu, overview |
| Resampling / conversion | `GPURasterReprojection`, `GPURasterTileReprojection`, `GPURasterConservativeRemapping`, `GPURasterSampling`, `GPURasterProfile`, `GPUPolygonRasterization`, `GPURasterJoin`, `GPUIsolines`, `GPUIsobands`, `GPUIsobandRings` | contours, buffer/texture conversion, tile cache and halo |

Remote-sensing sizes (largest accepted input and why):

| Contributor | Largest input | Bound | Scales? |
| --- | --- | --- | --- |
| `GPURasterTexture` | `7·N < 2³²` (`:54`) and the feature binding (28 bytes per cell; about 4.8 M cells at a 128 MiB binding limit) | levels ≤ 32, radius ≤ 8 | yes, row-parallel; heavy threads |
| `GPURasterTemporal` | `N·T < 2³²`, T ≤ 256 (`:52-53`) | private arrays of T | yes in N |
| `GPURasterSpectralUnmixing` | `N·B < 2³²`; B ≤ 256, M ≤ 8, 2,048-float private workspace (`:52-65`) | per-pixel QR | yes in N; O(B·M²) wasted per pixel (G8) |
| `GPURasterSeasonalBreak` | `3·N·T < 2³²`, T ≤ 256, private workspace ≤ 2,048 floats; iterative mode ≤ 16.7 M per-cell operations (`:105-109`) | per-pixel split search O(T²k²) | yes in N; slow per pixel |
| `GPURasterRegistration` | one image pair: periodic ≤ 2048 × 2048 power of two (`:129-131`), masked ≤ 512 × 512 (`raster-masked-registration.ts:27-28`) | FFT size and padded fields | no: serial argmax (G3) |
| `GPURasterRegionGraph` | pixels < 2³⁰; regions ≤ 256; rounds ≤ 256 (`:68-70`) | dense R² matrices | no (G6) |
| `GPURasterConservativeRemapping` | `N_dst · c < 2³²`, c ≤ 256 candidate source cells (`:86`), so a destination cell may span at most about 15 × 15 source cells | fixed-stride weights | yes in N_dst |

## 7. Top ten actions

| # | Action | Effort | Files |
| --- | --- | --- | --- |
| 1 | Replace the three ungated unrolled Jacobi loops: walking cost becomes a direction-dependent edge cost inside `GPUCostDistance`; reconstruction and skeleton move to the gated relaxation loop with a bounded iteration count | M | `gpu-raster-walking-cost.ts`, `gpu-raster-reconstruction.ts`, `gpu-raster-skeleton.ts`, `cost-distance/gpu-cost-distance.ts`, `cost-distance/raster-relaxation.ts` |
| 2 | Remove every single-thread full-grid loop: convergence counts through `createRasterCountNodes`; registration and masked registration peaks through `GPUReduction` | S | `gpu-raster-reconstruction.ts:125`, `gpu-raster-skeleton.ts:101`, `gpu-raster-walking-cost.ts:166`, `remote-sensing/gpu-raster-registration.ts:234,275`, `raster-masked-registration.ts:164,279` |
| 3 | Rebuild fractional coverage as per-triangle candidate emission plus `GPUCompaction`, and fractional zonal statistics on the sorted segment path | M | `gpu-raster-fractional-coverage.ts`, `gpu-raster-fractional-zonal-distribution.ts` |
| 4 | One raster input contract: published `GPURasterBand` + `grid` for every contributor, `values` + `validity` outputs, one no-data rule; migrate raster algebra, remote sensing, spectral indices (make them `GPUCommandNodeProducer`) and conservative remapping (drop `GPURasterRectilinearGrid`) | M | `raster-algebra/*`, `remote-sensing/*`, `gpu-raster-spectral-indices.ts` |
| 5 | Add `resolveGPURasterContext` (grid validation, `cellSizeMode`, `localOrigin`, per-frame georeference words) and remove baked affines; make isolines take the extent from `grid.affine` | S to M | `analysis-core/raster-grid.ts`, `gpu-raster-reprojection.ts`, `gpu-raster-walking-cost.ts`, fractional files, `isolines/isolines-parameters.ts` |
| 6 | Uniform tiling: `requiredHalo` on every focal contributor, published `GPURasterTileHaloAssembler` / `GPURasterTileCache` for focal and mosaic paths; global contributors declare complete-domain | M | focal files, `gpu-raster-tile-reprojection.ts`, `raster-region-tiles.ts` |
| 7 | Rebuild fill holes on `GPURasterConnectedComponents`, fill no-data `nearest` on `GPUDistanceField`, region graph as a sorted CSR graph with Boruvka merging; make the watershed forest the default and demote FIFO to a test oracle | M | `gpu-raster-reconstruction.ts`, `gpu-raster-fill-nodata.ts`, `remote-sensing/gpu-raster-region-graph.ts`, `gpu-raster-watershed-segmentation.ts` |
| 8 | Reorganize files by map-algebra class; one `raster-common/` for relaxation, D8, NaN, view validation, count and least-squares helpers; named exports only (stop exporting internal node builders); move `GPUFocalStatistics` into the focal family | S to M | `raster-analysis/index.ts`, `raster-expansion-utils.ts`, `remote-sensing/raster-kernel-utils.ts`, `raster-algebra/raster-algebra-utils.ts`, `cost-distance/raster-relaxation.ts`, `gpu-focal-statistics.ts` |
| 9 | Per-frame parameter views for the baked props in the expansion files; validation messages instead of bare `assert` in raster files (or record the exception) | S per file | 11 root files, `remote-sensing/*`, `focal-pairs/*` |
| 10 | Upstream proposals: rank filter, grayscale reconstruction, skeleton and medial axis, phase correlation, unary band math, min/max overview; decide the contour saddle rule with published `GPURasterContours`; then add the top four missing capabilities (QA masks, clip by polygon, line/point rasterization, categorical polygonize) | M to L | section 5 and 6 lists |
