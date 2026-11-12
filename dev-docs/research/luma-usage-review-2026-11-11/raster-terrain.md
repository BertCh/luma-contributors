# Stream `raster-terrain`: luma.gl usage review (2026-11-11, alpha.13)

Scope: `modules/analysis/src/raster-analysis` (171 files) and `modules/analysis/src/gpu-terrain`
(105 files) against `@luma.gl/graph/gpu-raster`, `@luma.gl/graph/lucim` and the raster parts of
`@luma.gl/graph/gpu-core`. Published paths below are under `node_modules/@luma.gl/graph/dist/`.

## 1. Coverage

Read in full (published `.d.ts`): `gpu-raster/index.d.ts`, `raster-device-limits.d.ts`,
`gpu-raster-connected-components.d.ts`, `gpu-raster-dense-components.d.ts`,
`gpu-raster-region-measurements.d.ts`, `gpu-raster-cross-tile-components.d.ts`,
`gpu-raster-tile-halo.d.ts`, `gpu-raster-overview.d.ts`, `gpu-raster-neighborhood.d.ts`,
`gpu-raster-convolution.d.ts`, `gpu-raster-edges.d.ts`, `gpu-raster-threshold.d.ts`,
`gpu-raster-statistics.d.ts`, `gpu-raster-contrast.d.ts`, `gpu-raster-global-statistics.d.ts`,
`gpu-raster-histogram.d.ts`, `gpu-raster-contours.d.ts`, `gpu-raster.d.ts`,
`gpu-raster-buffer-to-texture.d.ts`; `gpu-core/gpu-convolution.d.ts`, `gpu-fft2d.d.ts`,
`gpu-finite-difference-2d.d.ts`, `gpu-texture-history.d.ts`; `gpu-timeseries/*.d.ts`.
Published kernels read: `gpu-raster-neighborhood.js` (validity rule, `:195-305`),
`gpu-raster-connected-components.js` (`:8`, `:50-58`, hook pass `:185-230`),
`gpu-raster-tile-halo.js` (`GPURasterTileHaloFill` validation `:130-175`). Morphology,
overview, region measurements, statistics, threshold, global statistics and contrast kernels: only
their `.d.ts`.

Local code read (relevant parts, not every line): `raster-analysis/gpu-raster-reconstruction.ts`
(full), `raster-expansion-utils.ts:95-140`, `gpu-raster-tile-reprojection.ts` (full),
`raster-band-utils.ts:195-268`, `raster-patches/gpu-raster-patch-metrics.ts:1-200`,
`raster-patches/patch-labels.ts:1-80`, `raster-patches/gpu-raster-sieve.ts` (doc),
`raster-zonal-statistics/gpu-raster-zonal-statistics.ts:300-360`,
`raster-stretch/gpu-raster-stretch.ts:290-470`, `raster-embeddings/gpu-raster-embedding-pyramid-level.ts:30-120`,
`raster-pyramid/gpu-raster-extrema-pyramid.ts` (doc), `remote-sensing/gpu-raster-registration.ts:120-300`,
`remote-sensing/raster-region-tiles.ts:1-60`, `remote-sensing/gpu-raster-region-graph.ts` (doc),
`lagrangian-diagnostics/gpu-finite-time-lyapunov-exponent.ts:80-200`,
`cost-distance/gpu-cost-distance.ts:760-810`, `cost-distance/raster-relaxation.ts:755-1000` (halo
only), `raster-algebra/gpu-raster-conditional.ts` (doc), `gpu-raster-fill-nodata.ts` (doc);
`gpu-terrain/terrain-analysis/gpu-terrain-derivatives.ts:225-470`,
`terrain-illumination/gpu-relief-shading.ts` (gradient and validity lines),
`terrain-illumination/gpu-texture-shading.ts:40-260`, `relief-visualization/gpu-relief-blend.ts:60-210`,
`terrain-decode/gpu-terrain-spike-repair.ts` (component lines), `terrain-features/gpu-terrain-critical-points.ts` (doc),
`terrain-flow-field/gpu-terrain-flow-field.ts` (doc); `analysis-core/kernels/wgsl-kernel-nodes.ts:60-370`,
`analysis-core/kernels/gpu-contributor-utils.ts:260-480`; the showcase example
`examples/deck/spatial-analysis-showcase/src/scenes/raster/tiled-observations.compute.ts` (published
tile pipeline, grep only).

Not done (a later pass should not assume these were checked): the time-series folder
(`raster-time-series`, 21 classes) and `raster-embeddings` beyond the pyramid level were checked only
for the absence of a published per-pixel counterpart (none exists; `gpu-timeseries` is a single-series
envelope); hydrology (`gpu-terrain/hydrology`, `terrain-processing` except catchment summary),
viewshed, horizon, geomorphons, local dominance, TPI/TRI/VRM, Weiss landforms kernels were classified
from their documentation and halo declarations, not read line by line. `lucim` (3D volumes): no
local 3D voxel labelling or morphology exists in the two directories (grep `voxel`); the only voxel
code in the repository is space-time KDE in `gpu-spatial-analysis/point-density/`, outside this
stream (a lead for whoever owns it: `GPUVolumeThreshold` plus `GPUVolumeConnectedComponents` for
space-time hotspot clusters). No GPU run; no spec run.

## 2. Findings

### RT-01 `GPURasterFillHoles` floods one pixel per round; published connected components converge in about log2(N) rounds

- Class: `adopt-now` (no loss once converged; a status-semantics change on non-convergence).
- Local: `modules/analysis/src/raster-analysis/gpu-raster-reconstruction.ts:236` (class),
  `:285-320` (boundary marker plus `GPURasterReconstruction` flood), `:31` (`DEFAULT_MAXIMUM_ITERATIONS = 64`),
  round kernel `:154-190` (one 3x3 dilation per round), unrolled rounds
  `modules/analysis/src/raster-analysis/raster-expansion-utils.ts:104`.
- Published: `gpu-raster/gpu-raster-connected-components.d.ts:34` (class), `:23` (budget 1 to 64,
  default `ceil(log2(pixelCount)) + 2`, `gpu-raster-connected-components.js:50-58`), concurrent
  root-to-root `atomicMin` hooking plus compression (`.js:185-230`).
- Evidence: read both.
- What: the hole fill is a geodesic reconstruction from the border, one pixel of propagation per
  unrolled round. Background farther than 64 pixels (geodesic) from the border is reported as a hole
  with `converged = 0`; the showcase raises the budget to 128 (`missing-observations.compute.ts:736`),
  and every extra round is a command node. The published `GPURasterConnectedComponents` labels the
  background (input: a one-kernel inverted mask as `GPURasterBufferBand<'uint32'>`, connectivity 4
  or 8 passes straight through) with union-find rounds whose count does not grow with the hole size.
  Then one local kernel marks the roots of border pixels (`flag[label - 1] = 1`, N words) and one
  publishes `input != 0 || (label != 0 && flag[label - 1] == 0)`. Three small local kernels plus the
  published class, replacing the 64-round loop. Call sites: 1 class
  (`GPURasterFillHolesOperation`, `raster-morphology/gpu-raster-fill-holes-operation.ts:53`, follows),
  the showcase, the lens scenario `modules/lenses/src/computation-lenses/scenarios/gpu-raster/expansion-runner.ts:129`.
  `GPURasterReconstruction` itself (grey reconstruction) stays: that is the registered row.
- Verify: `modules/analysis/test/geospatial-expansion/raster-terrain-expansion.spec.ts:130`, `:507`,
  `:837`, `:918` and `raster-terrain-expansion.node.spec.ts:218` (results must be identical where the
  old loop converged); add a case with a corridor longer than 64 pixels that the old class could not
  finish, and the kernel-audit case `modules/analysis/test/analysis-core/kernel-audit/cases-expansion.ts:102`.
- Risk: none on converged results (binary, exact). On non-convergence the published class clears
  every label and validity (`.d.ts:30-33`), whereas the local class publishes a partial fill; the
  `status.converged` contract stays, but "partial fill" becomes "unresolved". `maximumIterations`
  changes meaning (round budget of a different algorithm, capped at 64).

### RT-02 Roadmap UPS-03 overstates the halo work: no contributor builds its own tile halo

- Class: `stale-claim` (roadmap), with a small `unused-feature` (the published type).
- Local: `dev-docs/roadmap.md:322` ("instead of per-contributor halos"). 19 contributors already
  declare `readonly requiredHalo` for the published `GPURasterHaloStage` contract (`git grep -l requiredHalo`
  over `modules/analysis/src`: 16 in `gpu-terrain`, 3 in `raster-analysis`, e.g.
  `gpu-terrain/terrain-analysis/gpu-terrain-derivatives.ts:174`, `raster-analysis/isolines/gpu-isolines.ts:164`,
  `raster-analysis/focal-pairs/gpu-raster-focal-pairs.ts:123`). None imports the type or writes
  `implements GPURasterHaloStage`; the contract is named only in comments. The only `halo` buffers in
  kernels are workgroup-shared tiles inside one dispatch (`cost-distance/raster-relaxation.ts:759-998`),
  not tile halos. No local cross-tile labelling exists; `remote-sensing/raster-region-tiles.ts:31-34`
  expects equivalence IDs "e.g. from cross-tile components".
- Published: `gpu-raster/gpu-raster-tile-halo.d.ts:6` (`GPURasterHaloStage`), `:67`
  (`GPURasterTileHaloAssembler.plan` composes stage radii), `gpu-raster-cross-tile-components.d.ts`.
  Used end to end only by the showcase (`tiled-observations.compute.ts:272-583`).
- Evidence: read both (declarations), inferred for "nothing else builds a halo" (grep for `halo`
  in both directories: 24 files, all listed above or comments).
- What: rewrite UPS-03 as (a) `import type {GPURasterHaloStage}` and `implements GPURasterHaloStage`
  on the 19 classes (typechecks the contract; no behaviour change); (b) add `requiredHalo` to the
  bounded-stencil contributors that lack it: `GPURasterRankFilter` (`radius`,
  `raster-analysis/gpu-raster-rank-filter.ts:32`), `GPUFocalStatistics` (`maximumRadius`,
  `gpu-spatial-analysis/spatial-interpolation/gpu-focal-statistics.ts:80`), `GPUIsobands` and
  `GPUIsobandRings` (1, as `GPUIsolines`), `GPUTerrainCriticalPoints` (1, 3x3 ring),
  `GPUTerrainFlowField` (1, central differences), `GPUFiniteTimeLyapunovExponent` (1); (c) document
  that global operators (flow, watersheds, cost distance, fill holes, reconstruction, distance field,
  viewshed) cannot be tiled by a halo. Cross-tile components need no local adoption until a local
  contributor labels across tiles.
- Verify: `yarn typecheck` for (a); a node spec asserting `requiredHalo` per class for (b).
- Risk: none (metadata only).

### RT-03 `GPURasterTileReprojection` assembles its mosaic by hand; `GPURasterTileHaloFill` does the complete-partition case

- Class: `adopt-with-cost`; register row missing (NEW).
- Local: `modules/analysis/src/raster-analysis/gpu-raster-tile-reprojection.ts:104-145` (clear kernel,
  per-tile `getRasterBufferBandNodes` canonicalisation, route kernel per tile).
- Published: `gpu-raster/gpu-raster-tile-halo.d.ts:95` (`GPURasterTileHaloFill`), validation
  `gpu-raster-tile-halo.js:152-173`.
- Evidence: read both.
- What: when the sources are a complete, non-overlapping partition with identical format and
  calibration (what `GPURasterTileHaloAssembler.plan` produces), the clear and route kernels can be
  one `GPURasterTileHaloFill` into native-format transients, followed by one canonicalisation of the
  whole mosaic. The local class must keep its own path for what the published class rejects: missing
  tiles left invalid, overlapping tiles with "later wins" priority, and per-tile scale, offset and
  nodata. One file, one call site in the showcase (`tiled-observations.compute.ts:788`).
- Verify: `modules/analysis/test/geospatial-expansion/raster-tile-reprojection.node.spec.ts` plus a GPU case on a
  complete partition, comparing with the current path bit for bit.
- Risk: none on the shared case (pure copies; calibration moves after assembly, same arithmetic per
  pixel). Recommendation: record a register row; delegate only if the complete-partition case is the
  common one.

### RT-04 Zonal statistics and patch metrics sit beside `GPURasterRegionMeasurements` without a register row; one code comment gives the wrong reason

- Class: register addition (`adopt-with-cost` at best); comment `stale-claim`.
- Local: `raster-analysis/raster-zonal-statistics/gpu-raster-zonal-statistics.ts:303-350` (count, sum,
  mean, min, max per dense zone; `sumOrder: 'atomic'` already uses `GPUGroupAggregation`, like the
  published class); `raster-analysis/raster-patches/gpu-raster-patch-metrics.ts:119` comment "Not
  GPURasterRegionMeasurements: it takes dense component labels with convergence/overflow gating and
  yields no bbox or perimeter faces".
- Published: `gpu-raster/gpu-raster-region-measurements.d.ts:63` (pixel and intensity counts, float
  atomic sums, extrema, means, centroids, areas; dense labels from 1; guarded by `converged`,
  `componentCount`, `overflow`).
- Evidence: read local, read published declaration.
- What: zonal statistics could feed `GPURasterRegionMeasurements` for the atomic case with constant
  guard views and a `zone + 1` label kernel, but would lose the deterministic `'sorted'` default, the
  geographic cell weights, variances, the workgroup-private fast path, and its out-of-range-zone
  flag; no gain since it already delegates the atomic sums to `GPUGroupAggregation`. Add a section 4
  row. Patch metrics correctly complements the published class (perimeters, extents, exact integer
  counts); fix the comment: both classes take dense labels with the same guards; the real reasons
  are perimeter/extent columns and exact, order-independent counts and areas.
- Verify: nothing to run; register text only.
- Risk: none.

### RT-05 `GPURasterRegistration` copies the FFT limits instead of reading the published constants

- Class: `misuse` (minor).
- Local: `raster-analysis/remote-sensing/gpu-raster-registration.ts:42`
  (`MAXIMUM_PERIODIC_REGISTRATION_DIMENSION = 2048`), `:152` (`validatePowerOfTwo(..., 2, 2048)`).
- Published: `gpu-core/gpu-fft2d.d.ts:6`, `:8` (`GPU_FFT2D_MIN_DIMENSION`, `GPU_FFT2D_MAX_DIMENSION`,
  exported from `gpu-core/index.d.ts:71`), `getGPUFFT2DSupport` (`:66`, device check of scratch size).
- Evidence: read both.
- What: define the local constant as `GPU_FFT2D_MAX_DIMENSION` (keep the exported local name for
  callers) and the minimum as `GPU_FFT2D_MIN_DIMENSION`, so an upstream change propagates; in
  `getCommandNodes` call `getGPUFFT2DSupport(graph.device, {width, height, batchCount})` and throw
  its `reason` with the contributor ID, instead of failing inside the published constructor. One
  file, two lines plus one check.
- Verify: `modules/analysis/test` registration spec; a node spec that the exported maximum equals the
  published constant.
- Risk: none.

### RT-06 `GPUReliefBlend` says percent stretches "need a histogram"; the published percentile chain exists

- Class: `stale-claim` / `unused-feature` (`adopt-with-cost`).
- Local: `gpu-terrain/relief-visualization/gpu-relief-blend.ts:205-206`; per-layer minimum and maximum
  are rows of the per-frame `parameters` view (`:82-117`).
- Published: `gpu-raster/gpu-raster-histogram.d.ts:26`; `gpu-raster-global-statistics.d.ts:48-80`
  (`GPURasterGlobalInitialize`, `GPURasterGlobalStatisticsMerge`, `GPURasterGlobalHistogramMerge`,
  `GPURasterGlobalPercentile`), all GPU-resident.
- Evidence: read local, read published declarations.
- What: a caller (or a helper beside the class) can produce the RVT "percent" bounds on the GPU:
  initialise an accumulator, merge statistics, merge the histogram, run two `GPURasterGlobalPercentile`
  and copy the two scalars into the layer's parameter rows. Update the comment to name this route and
  its limit. Zero files to change for the claim; a helper would be new code.
- Verify: a spec comparing with `numpy.percentile` on a test DEM within one bin width.
- Risk: the published percentile is a bin-centre estimate over at most 256 bins (`gpu-raster-global-statistics.d.ts:87`
  "interior estimates choose the center of their first matching bin"), not RVT's exact percentile.

### RT-07 Relief mean filters, texture shading and FTLE are local kernels beside published stencils with no register row

- Class: register addition (reasons hold at alpha.13).
- Local: `gpu-terrain/relief-visualization/relief-visualization-utils.ts:123` (comment: radius capped
  at 8, absolute float32) used by `GPUSimpleLocalRelief` and `GPUMultiScaleRelief`;
  `gpu-terrain/terrain-illumination/gpu-texture-shading.ts:108-118` (Gaussian radius `ceil(3 sigma)`,
  cascade sigmas of 4 and more), `:209-213` (normalized convolution with a validity channel);
  `raster-analysis/lagrangian-diagnostics/gpu-finite-time-lyapunov-exponent.ts:178-198`.
- Published: `gpu-raster/gpu-raster-neighborhood.d.ts` (radius "bounded to eight", centre invalid stays
  invalid), `gpu-raster-convolution.d.ts:12-14` (smoothing radius 0 to 8), `gpu-core/gpu-convolution.d.ts`
  (boundary `'zero' | 'wrap'`, no validity), `gpu-fft2d.d.ts` (power of two, at most 2048),
  `gpu-finite-difference-2d.d.ts` (compile-time spacing, second-order one-sided border, scalar input).
- Evidence: read both (declarations and local kernels).
- What: add three section 4 rows. Relief: needs radius above 8 and a centre-relative sum. Texture
  shading: needs radius above 8, clamp border and normalized convolution (Leland Brown's exact method is
  spectral; `GPUFFT2D`/`GPUConvolution` `'fft'` would need a non-power-of-two size and a non-periodic
  border). FTLE: needs per-frame spacing, float32x2 input, first-order border; two `GPUFiniteDifference2D`
  passes would add a deinterleave and change border values.
- Verify / risk: none (documentation).

### RT-08 Register row `GPURasterStretch`: "GPUHistogram: reading a raw band" is answered by `GPURasterHistogram`

- Class: `stale-claim` (register and one code comment).
- Local: `dev-docs/upstream-workarounds.md:93` (last cell: "`GPUHistogram`: reading a raw band");
  `raster-analysis/raster-stretch/gpu-raster-stretch.ts:410-411` ("needs a materialized calibrated view").
- Published: `gpu-raster/gpu-raster-histogram.d.ts:5-26`: reads a raw `GPURasterBufferBand` with
  validity, exact nodata and a GPU or `'valid-auto'` domain.
- Evidence: read both.
- What: the raw-band need is met at alpha.13 (the stretch's raw `values` plus `validity` plus
  `noDataValue` wrap into a `GPURasterBufferBand` literal with no copy). The reasons that remain are the
  per-frame window and region mask (the published histogram has neither) and the boundary-bin rounding
  rule the comment mentions. Rewrite the cell and the comment to say that.
- Verify / risk: none.

### RT-09 The 12 raw-value raster contributors are still 12; three are easy to move

- Class: `adopt-with-cost` (roadmap 12.4 status, not new).
- Local (props with `values` plus `noDataValue` and no `GPURasterBand`): `raster-algebra/gpu-raster-arithmetic.ts`,
  `gpu-raster-cell-statistics.ts`, `gpu-raster-conditional.ts`, `gpu-raster-reclassify.ts`,
  `gpu-weighted-overlay.ts`; `isolines/gpu-isolines.ts`, `gpu-isobands.ts`, `gpu-isoband-rings.ts`;
  `raster-sampling/gpu-raster-profile.ts`, `gpu-raster-sampling.ts`; `raster-stretch/gpu-raster-stretch.ts`;
  `gpu-spatial-analysis/spatial-interpolation/gpu-focal-statistics.ts` (count: grep `noDataValue?: number`
  minus `GPURasterBand`, 12, unchanged from luma-official.md 5.2).
- Published: `gpu-raster/types.d.ts` (`GPURasterBufferBand`).
- Evidence: read (props), inferred for the effort.
- What: easiest are the single-input per-pixel classes `GPURasterReclassify`, `GPURasterConditional`
  (condition raster) and `GPURasterStretch`: accept a `GPURasterBand`, call the existing
  `getRasterBufferBandNodes(..., 'reuse-canonical')` (`raster-band-utils.ts:206`) and keep the kernels.
  Multi-input ones (arithmetic, cell statistics, weighted overlay) take N bands with N
  canonicalisations; isolines and sampling read raw values inside geometry kernels and gain texture
  input only through `GPURasterTextureToBuffer`.
- Risk: one extra identity pass per band unless the band is float32 with validity and no calibration;
  calibrated integer bands would start to work (a gain), and `noDataValue` comparison moves from the
  local float compare to the published exact raw-format compare.

## 3. Checked and fine

- `GPURasterGradient` in `GPUTerrainDerivatives`/`GPUReliefShading`: well used. The published validity
  skips zero taps (`gpu-raster-neighborhood.js:290`), so x and y validity differ; the shade kernels pass
  only the x validity but also require `isFiniteValue(rawY)` (`gpu-terrain-derivatives.ts:432`,
  `gpu-relief-shading.ts:596`, `:771`), which catches the published NaN. `scale: 1` with local
  per-row division is right (published `scale` is a constant; geographic rows need per-row cell size).
- `GPURasterContours` in `GPUTerrainContours`; `GPURasterBandMath` in the four spectral indices
  (`GPURasterNDVI` correctly not duplicated locally); `GPUFFT2D` in registration (apart from RT-05).
- `getRasterBufferBandNodes` canonicalises a buffer band with an identity `GPURasterNeighborhood`
  (radius 0) and a texture band with `GPURasterTextureToBuffer` (40 files call it): the only public
  route that applies scale, offset, nodata and validity for a buffer band; `'reuse-canonical'` skips the
  pass when nothing is to do.
- `GPURasterBufferToTexture` in 12 files: output display textures; fine.
- `GPUTerrainRGBDecode` reads `rgba8unorm` with `textureLoad`: `GPURasterTextureToBuffer` takes one
  scalar channel and cannot decode RGB-encoded elevation.
- `GPURasterSieve`, `GPURasterPatchMetrics` consume published `GPURasterConnectedComponents` and
  `GPURasterDenseComponents` labels (lens scenarios `raster-sieve.example.ts:42`, `raster-patch-metrics.example.ts:41`).
- `GPUTerrainSpikeRepair`: binary `GPURasterConnectedComponents` correctly not used (components split
  on an edge predicate `|dh| <= jump`, reason at `gpu-terrain-spike-repair.ts:195`); flats
  (`hydrology/terrain-flow-flats.ts:43`) are equal-surface components, also not binary; the watershed
  forest (`raster-watershed-forest.ts:22`) is a minimum spanning forest.
- `GPURasterExtremaPyramid` (min/max acceleration pyramid; `GPURasterOverview` is mean/mode only,
  comment `:244` correct), `GPURasterEmbeddingPyramidLevel` (unit-vector mean; reason correct).
- `GPURasterConservativeRemapping`: general rational ratios; `GPURasterOverview` covers only the
  integer-aligned mean case.
- Thresholding and Otsu: no local thresholding-to-mask or Otsu kernel exists (`GPURasterConditional`
  is `where(c, a, b)`); `GPURasterThreshold`/`GPURasterOtsuThreshold` unused for want of a caller.
- Sobel/Scharr/Laplacian/edge magnitude: no local copy (FTLE takes a vector-field Jacobian, RT-07).
- Opening, closing, morphology modes: no local copy beyond the registered rank filter and reconstruction.
- `GPURasterRankFilter`, `GPUFocalStatistics`, `GPUTerrainCurvature`, `GPUTerrainFlowField`,
  `GPUTerrainCatchmentSummary`, `GPUIsolines`/`GPUIsobands`, `GPURasterArithmetic`, `GPURasterStretch`:
  registered reasons still hold at alpha.13 (radius cap 8, `gpu-raster-neighborhood.d.ts`; no per-row
  spacing in `GPUFiniteDifference2D`; float atomic sums in region measurements).
- `planRasterDispatchStripes` / `getRasterDeviceLimits`: correctly not used. Every local kernel goes
  through `createWGSLKernelNode`, which uses the published `getBoundedDispatchLayout` (3D wrap beyond
  `maxComputeWorkgroupsPerDimension`) and checks each binding against `maxStorageBufferBindingSize`
  (`analysis-core/kernels/wgsl-kernel-nodes.ts:335-359`). Stripes help only per-pixel kernels on rasters
  above one binding, which no contributor supports today.
- `GPURaster`: not needed; its helpers (`getPixelArea`, `isCompatibleWith`) need a graph-owned band set.
  13 hand-written affine determinants (`git grep` over `modules/analysis/src`) are a local-helper
  cleanup, not a luma.gl one.
- `GPUTextureHistory`: no local texture ping-pong in the two directories.
- `GPURasterCategoricalOverview`, `makeRasterOverviewMetadata`, global accumulators, tile reader/cache
  and leases: no local code does these jobs (the showcase uses them directly).
- `gpu-timeseries`: single-series envelopes; no overlap with per-pixel raster time series.

## 4. Register corrections (`dev-docs/upstream-workarounds.md`)

- Row `:93` (`GPURasterStretch`): the `GPUHistogram` need "reading a raw band" is met by
  `GPURasterHistogram`; remaining: per-frame window, region mask, boundary-bin rule (RT-08).
- Row `:95` (`GPURasterReconstruction` note): add that `GPURasterFillHoles` can delegate to
  `GPURasterConnectedComponents` today (RT-01).
- New rows: `GPURasterTileHaloFill` vs the `GPURasterTileReprojection` mosaic (RT-03);
  `GPURasterRegionMeasurements` vs `GPURasterZonalStatistics` and `GPURasterPatchMetrics` (RT-04);
  `GPURasterBoxBlur`/`GPURasterNeighborhood` vs relief mean filters, `GPURasterGaussianBlur` vs texture
  shading, `GPUFiniteDifference2D` vs FTLE (RT-07).
- Roadmap `dev-docs/roadmap.md:322` (UPS-03): rewrite per RT-02.

## 5. Exported `GPU*` classes in the two directories (152 classes, 146 files)

Column 3 lists the published raster/core classes the file constructs (script: `new <published name>`
per file, plus `getRasterBufferBandNodes` callers). Full 152-row table:
`<scratchpad>/review/raster-terrain/table.md`.
Below: every class with a verdict other than "own kernel, no published equivalent" or "adapter of the
local class".

| Class | File | Published raster/core classes constructed | Verdict |
| --- | --- | --- | --- |
| `GPUMultiScaleRelief` | `modules/analysis/src/gpu-terrain/relief-visualization/gpu-multi-scale-relief.ts` | band input via `getRasterBufferBandNodes` (`GPURasterNeighborhood` identity / `GPURasterTextureToBuffer`) | beside `GPURasterBoxBlur`/`Neighborhood` [NEW row; reason in code holds, RT-07] |
| `GPUReliefBlend` | `modules/analysis/src/gpu-terrain/relief-visualization/gpu-relief-blend.ts` | - | own kernel; percent stretch could use published percentile chain (RT-06) |
| `GPUSimpleLocalRelief` | `modules/analysis/src/gpu-terrain/relief-visualization/gpu-simple-local-relief.ts` | band input via `getRasterBufferBandNodes` (`GPURasterNeighborhood` identity / `GPURasterTextureToBuffer`) | beside `GPURasterBoxBlur`/`Neighborhood` [NEW row; reason in code holds, RT-07] |
| `GPUTerrainContours` | `modules/analysis/src/gpu-terrain/terrain-analysis/gpu-terrain-contours.ts` | `GPURasterContours`; band input via `getRasterBufferBandNodes` (`GPURasterNeighborhood` identity / `GPURasterTextureToBuffer`) | delegates (`GPURasterContours`) |
| `GPUTerrainDerivatives` | `modules/analysis/src/gpu-terrain/terrain-analysis/gpu-terrain-derivatives.ts` | `GPURasterBufferToTexture`, `GPURasterGradient`; band input via `getRasterBufferBandNodes` (`GPURasterNeighborhood` identity / `GPURasterTextureToBuffer`) | delegates (`GPURasterGradient`); padded border local [registered] |
| `GPUTerrainCurvature` | `modules/analysis/src/gpu-terrain/terrain-curvature/gpu-terrain-curvature.ts` | band input via `getRasterBufferBandNodes` (`GPURasterNeighborhood` identity / `GPURasterTextureToBuffer`) | beside `GPUFiniteDifference2D`/`GPURasterConvolution` [registered] |
| `GPUTerrainSpikeRepair` | `modules/analysis/src/gpu-terrain/terrain-decode/gpu-terrain-spike-repair.ts` | band input via `getRasterBufferBandNodes` (`GPURasterNeighborhood` identity / `GPURasterTextureToBuffer`) | beside graph CC [registered]; `GPURasterConnectedComponents` correctly not used (edge predicate) |
| `GPUTerrainFlowField` | `modules/analysis/src/gpu-terrain/terrain-flow-field/gpu-terrain-flow-field.ts` | band input via `getRasterBufferBandNodes` (`GPURasterNeighborhood` identity / `GPURasterTextureToBuffer`) | beside `GPUFiniteDifference2D` [registered] |
| `GPUReliefShading` | `modules/analysis/src/gpu-terrain/terrain-illumination/gpu-relief-shading.ts` | `GPURasterBufferToTexture`, `GPURasterGradient`; band input via `getRasterBufferBandNodes` (`GPURasterNeighborhood` identity / `GPURasterTextureToBuffer`) | delegates (`GPURasterGradient`); padded border local [registered] |
| `GPUTextureShading` | `modules/analysis/src/gpu-terrain/terrain-illumination/gpu-texture-shading.ts` | `GPURasterBufferToTexture`; band input via `getRasterBufferBandNodes` (`GPURasterNeighborhood` identity / `GPURasterTextureToBuffer`) | beside `GPURasterGaussianBlur` (and `GPUFFT2D`) [NEW row, RT-07] |
| `GPUTerrainCatchmentSummary` | `modules/analysis/src/gpu-terrain/terrain-processing/gpu-terrain-catchment-summary.ts` | - | beside `GPURasterRegionMeasurements` [registered] |
| `GPUCostDistance` | `modules/analysis/src/raster-analysis/cost-distance/gpu-cost-distance.ts` | `GPUHistogram`; band input via `getRasterBufferBandNodes` (`GPURasterNeighborhood` identity / `GPURasterTextureToBuffer`) | own kernel, no published equivalent; band counts delegate (`GPUHistogram`) |
| `GPURasterRankFilter` | `modules/analysis/src/raster-analysis/gpu-raster-rank-filter.ts` | band input via `getRasterBufferBandNodes` (`GPURasterNeighborhood` identity / `GPURasterTextureToBuffer`) | beside `GPURasterErosion`/`Dilation` at q=0/1 [registered] |
| `GPURasterReconstruction` | `modules/analysis/src/raster-analysis/gpu-raster-reconstruction.ts` | - | beside `GPURasterDilation` [registered] |
| `GPURasterFillHoles` | `modules/analysis/src/raster-analysis/gpu-raster-reconstruction.ts` | - | beside `GPURasterConnectedComponents` [NEW, RT-01] |
| `GPURasterNDWI` | `modules/analysis/src/raster-analysis/gpu-raster-spectral-indices.ts` | `GPURasterBandMath` | delegates (`GPURasterBandMath`) |
| `GPURasterNBR` | `modules/analysis/src/raster-analysis/gpu-raster-spectral-indices.ts` | `GPURasterBandMath` | delegates (`GPURasterBandMath`) |
| `GPURasterEVI` | `modules/analysis/src/raster-analysis/gpu-raster-spectral-indices.ts` | `GPURasterBandMath` | delegates (`GPURasterBandMath`) |
| `GPURasterSAVI` | `modules/analysis/src/raster-analysis/gpu-raster-spectral-indices.ts` | `GPURasterBandMath` | delegates (`GPURasterBandMath`) |
| `GPURasterTileReprojection` | `modules/analysis/src/raster-analysis/gpu-raster-tile-reprojection.ts` | band input via `getRasterBufferBandNodes` (`GPURasterNeighborhood` identity / `GPURasterTextureToBuffer`) | mosaic beside `GPURasterTileHaloFill` [NEW, RT-03]; sampling own |
| `GPUIsobandRings` | `modules/analysis/src/raster-analysis/isolines/gpu-isoband-rings.ts` | - | downstream of `GPUIsobands` [registered via isobands] |
| `GPUIsobands` | `modules/analysis/src/raster-analysis/isolines/gpu-isobands.ts` | - | beside `GPURasterContours` [registered] |
| `GPUIsolines` | `modules/analysis/src/raster-analysis/isolines/gpu-isolines.ts` | - | beside `GPURasterContours` [registered] |
| `GPUFiniteTimeLyapunovExponent` | `modules/analysis/src/raster-analysis/lagrangian-diagnostics/gpu-finite-time-lyapunov-exponent.ts` | - | beside `GPUFiniteDifference2D` [NEW row, low value] |
| `GPURasterArithmetic` | `modules/analysis/src/raster-analysis/raster-algebra/gpu-raster-arithmetic.ts` | - | beside `GPURasterBandMath` [registered] |
| `GPURasterEmbeddingPyramidLevel` | `modules/analysis/src/raster-analysis/raster-embeddings/gpu-raster-embedding-pyramid-level.ts` | - | own kernel; `GPURasterOverview` wrong for unit vectors (reason in doc holds) |
| `GPURasterPatchMetrics` | `modules/analysis/src/raster-analysis/raster-patches/gpu-raster-patch-metrics.ts` | - | complements `GPURasterRegionMeasurements`; counts/areas overlap [NEW note, RT-04] |
| `GPURasterSieve` | `modules/analysis/src/raster-analysis/raster-patches/gpu-raster-sieve.ts` | - | own kernel on published `GPURasterDenseComponents` labels |
| `GPURasterExtremaPyramid` | `modules/analysis/src/raster-analysis/raster-pyramid/gpu-raster-extrema-pyramid.ts` | band input via `getRasterBufferBandNodes` (`GPURasterNeighborhood` identity / `GPURasterTextureToBuffer`) | own kernel; `GPURasterOverview` has no min/max (reason in code holds) |
| `GPURasterStretch` | `modules/analysis/src/raster-analysis/raster-stretch/gpu-raster-stretch.ts` | - | beside `GPURasterContrast`/`Statistics`/`GPUHistogram` [registered; RT-08] |
| `GPURasterZonalStatistics` | `modules/analysis/src/raster-analysis/raster-zonal-statistics/gpu-raster-zonal-statistics.ts` | band input via `getRasterBufferBandNodes` (`GPURasterNeighborhood` identity / `GPURasterTextureToBuffer`) | beside `GPURasterRegionMeasurements` [NEW, RT-04] |
| `GPURasterRegistration` | `modules/analysis/src/raster-analysis/remote-sensing/gpu-raster-registration.ts` | `GPUFFT2D` | delegates the transforms (`GPUFFT2D`); limits copied by hand (RT-05) |

Classes with verdict "own kernel, no published equivalent" (some use `getRasterBufferBandNodes` or `GPURasterBufferToTexture`; see the full table) or "adapter of the local class" (`*Operation`):

- `gpu-terrain/geomorphons`: `GPUGeomorphons`
- `gpu-terrain/hydrology`: `GPUTerrainFlowDistanceOperation`, `GPUTerrainFlowDistance`, `GPUTerrainFlowOperation`, `GPUTerrainFlow`, `GPUTerrainHeightAboveDrainage`, `GPUTerrainHydrologicIndices`, `GPUTerrainStreamNetwork`, `GPUTerrainStreamOrderOperation`, `GPUTerrainStreamOrder`, `GPUTerrainWatershedsOperation`, `GPUTerrainWatersheds`
- `gpu-terrain/point-horizon`: `GPUPointHorizonProfile`, `GPUPointHorizonVisibility`
- `gpu-terrain/relief-visualization`: `GPULocalDominance`
- `gpu-terrain/terrain-analysis`: `GPUTerrainCumulativeViewshed`, `GPUTerrainLineOfSight`, `GPUTerrainViewshed`
- `gpu-terrain/terrain-decode`: `GPUTerrainRGBDecode`
- `gpu-terrain/terrain-features`: `GPUProfilePeaksOperation`, `GPUProfilePeaks`, `GPUTerrainCriticalPoints`, `GPUTerrainPeakProminenceOperation`, `GPUTerrainPeakProminence`, `GPUTerrainPeakSnap`, `GPUTerrainSummits`
- `gpu-terrain/terrain-illumination`: `GPUSolarIrradiance`, `GPUSolarPositionOperation`, `GPUSolarPosition`, `GPUSolarShadowMask`, `GPUTerrainCastShadow`, `GPUTerrainHorizon`
- `gpu-terrain/terrain-processing`: `GPUTerrainBranchingFlowOperation`, `GPUTerrainBranchingFlow`, `GPUTerrainBreachOperation`, `GPUTerrainBreach`, `GPUTerrainCatchmentSummaryOperation`, `GPUTerrainDEMCoregistration`, `GPUTerrainDepressionHierarchyOperation`, `GPUTerrainDepressionHierarchy`, `GPUTerrainFlowFractions`, `GPUTerrainTiledFlow`
- `gpu-terrain/topographic-position`: `GPUTerrainRuggedness`, `GPUTerrainTopographicPosition`, `GPUTerrainVectorRuggedness`, `GPUTerrainWeissLandforms`
- `raster-analysis`: `GPURasterFillNoData`, `GPURasterFractionalCategoricalStatistics`, `GPURasterFractionalCoverage`, `GPURasterFractionalZonalDistribution`, `GPURasterFractionalZonalStatistics`, `GPURasterMedialAxis`, `GPURasterReprojection`, `GPURasterSkeleton`, `GPURasterWalkingCost`, `GPURasterWatershedSegmentation`
- `raster-analysis/change-detection`: `GPUChangeDetection`
- `raster-analysis/cost-distance`: `GPUCostDistanceOperation`, `GPUCostDistancePathOperation`, `GPUCostDistancePath`, `GPURasterWalkingCostOperation`
- `raster-analysis/distance-field`: `GPUDistanceField`
- `raster-analysis/flow-texture`: `GPULineIntegralConvolution`, `GPUStreamlinesOperation`, `GPUStreamlines`
- `raster-analysis/focal-pairs`: `GPURasterFocalPairs`
- `raster-analysis/isolines`: `GPUIsolinesOperation`
- `raster-analysis/lagrangian-diagnostics`: `GPULagrangianFlowMap`
- `raster-analysis/particle-advection`: `GPUParticleAdvection`
- `raster-analysis/polygon-rasterization`: `GPUPolygonRasterization`, `GPURasterJoin`
- `raster-analysis/raster-algebra`: `GPURasterCellStatistics`, `GPURasterConditional`, `GPURasterReclassify`, `GPUWeightedOverlay`
- `raster-analysis/raster-embeddings`: `GPURasterEmbeddingChange`, `GPURasterEmbeddingCovariance`, `GPURasterEmbeddingDequantize`, `GPURasterEmbeddingKMeans`, `GPURasterEmbeddingKNearestClassifier`, `GPURasterEmbeddingLinearProbe`, `GPURasterEmbeddingNearestPixels`, `GPURasterEmbeddingProjection`, `GPURasterEmbeddingSimilarity`, `GPURasterEmbeddingZoneMean`
- `raster-analysis/raster-morphology`: `GPURasterFillHolesOperation`, `GPURasterReconstructionOperation`
- `raster-analysis/raster-sampling`: `GPURasterProfile`, `GPURasterSampling`
- `raster-analysis/raster-time-series`: `GPURasterBFASTMonitor`, `GPURasterBestAvailablePixel`, `GPURasterCOLDMonitor`, `GPURasterClimateAnomaly`, `GPURasterClimateRunAccumulator`, `GPURasterEOFProjection`, `GPURasterGeometricMedianCompositeOperation`, `GPURasterGeometricMedianComposite`, `GPURasterHANTS`, `GPURasterHarmonicRegression`, `GPURasterLaggedCorrelation`, `GPURasterMedoidCompositeOperation`, `GPURasterMedoidComposite`, `GPURasterPhenologyMetrics`, `GPURasterRollingMean`, `GPURasterSnapshotCovariance`, `GPURasterStandardizedIndexCalibration`, `GPURasterStandardizedIndex`, `GPURasterStreamingAccumulator`, `GPURasterTemporalResample`, `GPURasterWhittakerSmoother`
- `raster-analysis/raster-watershed`: `GPURasterWatershedSegmentationOperation`
- `raster-analysis/raster-zonal-statistics`: `GPURasterFractionalZonalDistributionOperation`, `GPURasterFractionalZonalStatisticsOperation`
- `raster-analysis/remote-sensing`: `GPURasterConservativeRemapping`, `GPURasterRegionGraphOperation`, `GPURasterRegionGraph`, `GPURasterSeasonalBreak`, `GPURasterSpectralUnmixing`, `GPURasterTemporal`, `GPURasterTexture`
