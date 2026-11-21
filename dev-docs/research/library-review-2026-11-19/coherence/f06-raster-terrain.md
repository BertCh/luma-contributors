# Raster and terrain as a GIS user

## Summary

The most consequential error is an instruction to interchange two NDVI workflows. On three ordinary reflectance observations, NDVI of the spectral geometric median was **0.26006**. Median NDVI was **0.33333**. Both calculations succeeded.

Two visibility APIs also disagree about the same observer and target. A valid target beside a no-data pixel was visible in the viewshed and no-data in line of sight. A second fixture showed an off-ray no-data pixel making a real ridge disappear from the visibility test.

The zonal statistics guide gives an incorrect conversion from sample to population variance for unequal-area cells. Nearest-neighbor reprojection has a separate categorical hazard: it converts labels to float32, so two distinct, valid zone IDs can become one.

The most valuable first-use change is a checked walkthrough that retains the upstream `GPURaster` through alignment, selects a named band, declares ground units, and follows one result into its next consumer. The existing `assertCompatibleWith`, `getBand`, and `GPUCubeLayoutTranspose` already solve substantial parts of that journey.

**URT-01 through URT-10 are my ten priorities for the alpha.** URT-11 through URT-14 are additional independently actionable findings.

## Census

I used the installed TypeScript API with `createProgram` on the raster and terrain public `index.ts` files, without calling the compiler, emitting files, or running diagnostics. I resolved `getExportsOfModule` aliases, selected exported `GPU*` classes with an instance `getCommandNodes`, and read the first constructor argument’s properties, including inherited properties. This measures the **public call shape**, rather than counting files that mention a grid.

| Constructor shape | Raster producers | Terrain producers |
|---|---:|---:|
| Has `grid`, without a separate `cellCount` | 36 | 44 |
| Has both `grid` and `cellCount` | 4 | 0 |
| Has `sourceGrid` and `destinationGrid` | 2 | 0 |
| Has `width` and `height` | 4 | 0 |
| Has `cellCount`, without `grid` | 23 | 0 |
| Other shape | 17 | 2 |
| **Public producers** | **86** | **46** |
| All exported `GPU*` classes, including operation forms | 99 | 57 |

The 23 cell-count constructors include 21 temporal or change-analysis producers plus spectral unmixing and seasonal break as classified by constructor shape. The names are listed below. A cell-major time series legitimately needs no affine. The consequential inconsistency is that a caller doing *spatial band arithmetic* can supply two equally long bands on different grids and pass every check.

I also checked the public exports for the names a reader encounters in these journeys. `GPURasterNBR` and `GPURasterTemporal` resolve from `raster-analysis`, `GPUFocalStatistics` from `gpu-spatial-analysis`, `GPUCubeLayoutTranspose` from `dataframe-analysis`, and `GPURasterNDVI` from the upstream `@luma.gl/graph/gpu-raster`. The last one has `addToGraph` as its only public prototype method besides the constructor.

**Evidence notation.** `R` means `modules/analysis/src/raster-analysis`, `T` means `modules/analysis/src/gpu-terrain`, and `D` means `docs/api`. `KI` identifies the newer `luma-contributors-ki` copy of a staging-owned file. Other paths refer to the requested `r19` checkout. I executed CPU functions and host graph declarations through the permitted esbuild-to-Node pipe. NumPy was 2.0.2 and SciPy was 1.13.1. Device outcomes below are marked `SOURCE`.

## Proposed convention

A raster operation should let a caller identify the **grid, band representation, physical units, validity rule, and output layout** at the point where the result is connected to the next operation. Use the published `GPURaster` and its compatibility check for spatial bands, name the layout for packed cubes, and name the unit or statistic when two same-format ports carry different quantities.

Mechanically check each documented cross-family handoff with two small fixtures: one that must compose and one that must be rejected or produce a distinguishable result. A flat DEM, one raster cell, or two identical dates cannot detect most of the departures below.

## Findings

| ID | Kind | Reach | Change | Cost | Where | One line |
|---|---|---|---|---|---|---|
| URT-01 | W/D | ORDINARY | ADDITIVE | S | Geometric median docs, source and catalog | The instructed NDVI operation order changes the answer. |
| URT-02 | W | ORDINARY | BREAKING | M | Viewshed, line of sight, cumulative viewshed | Zero-weight no-data neighbors can erase an occluder or invalidate only one consumer’s target. |
| URT-03 | W/D | ORDINARY | ADDITIVE | S | Whole-cell zonal variance | The documented population conversion is wrong for unequal weights. |
| URT-04 | W | ORDINARY | ADDITIVE | M | Nearest warp, tile mosaic, categorical cell statistics | Continuous float32 conversion merges distinct integer labels. |
| URT-05 | W | UNUSUAL | ADDITIVE | S | Eye-height and lighting packers | A missing eye height can produce an ordinary visible result. |
| URT-06 | W | ORDINARY | ADDITIVE | S | Reprojection and rank filter | Unsupported GIS option strings silently select a different algorithm. |
| URT-07 | S | ORDINARY | ADDITIVE | S | Direct and tiled reprojection | Reordered CRS keys fail, and one rejected mosaic prevents a corrected retry. |
| URT-08 | E/C | ORDINARY | ADDITIVE | S | Grid and band construction, map algebra, cost distance | The checked grid can be lost before either alignment or distance units are established. |
| URT-09 | E/D | ORDINARY | ADDITIVE | S | NDVI, temporal reduction, cube transpose | The correct two-date workflow crosses three entries and two stack layouts. |
| URT-10 | C | ORDINARY | BREAKING | S | Derivatives, hydrologic indices, solar inputs | `slope` can mean 45 degrees or a clamped D8 tangent of 1. |
| URT-11 | D/C | ORDINARY | ADDITIVE | S | Hydrology composition example | One example silently mixes a branching accumulation with single-receiver drainage products. |
| URT-12 | E/D | ORDINARY | ADDITIVE | S | Raster operation index and focal statistics | The page for focal pairs is easy to find. The scalar focal mean and its different center rule are not. |
| URT-13 | C | UNUSUAL | ADDITIVE | S | Shared pure CPU band type | A band declared `uint32` can contain negative or fractional float samples. |
| URT-14 | W | EXTREME | ADDITIVE | S | Temporal medoid | Squared distances can all vanish or overflow, yielding the first date with success. |

### URT-01: NDVI of the geometric median and median NDVI are different products

**Where.** `KI D/gpu-raster/raster-time-series.md:232`, `R/raster-time-series/gpu-raster-geometric-median-composite.ts:97`, `website/src/catalog/entries/raster-composites-climate.ts:212`. The same premise appears in `dev-docs/research/spatiotemporal-review-2026-11-09/research-raster-time-series.md:232`.

**Evidence — EXECUTED.** One cell, three dates, calibrated `[nearInfrared, red]` pairs `[[0.5,0.25],[1,0.25],[0.5,0.75]]`. I ran `computeRasterGeometricMedianCompositeOnCPU` with the public default epsilon and 64 iterations on the two bands, then on the three scalar NDVIs.

| Route | Result |
|---|---:|
| Two-band composite | `[0.6056624, 0.3556624]` |
| NDVI of that composite | **0.2600577** |
| NDVI per date | `[0.3333333, 0.6, -0.2]` |
| Scalar composite of those NDVIs | **0.3333333** |

Both statuses were zero. The two-band final step was `6.0e-13`. Independent SciPy minimization gave NDVI `0.2600578`, and NumPy’s median gave `0.3333333`.

**Now.** A reader can replace `median(dates.map(ndvi))` with `ndvi(geometricMedian(dates))` on the explicit assurance that the result is equal.

**Wanted.** Replace the claim in all three user-facing locations with “The geometric median fits bands jointly. To obtain median NDVI, calculate NDVI per valid acquisition and then take its temporal median. NDVI of the composite describes the synthetic composite spectrum.” Put the three-date example beside it.

**In-repository callers that change.** None. **Value and cost.** Remote-sensing users can select the scientific quantity they intended. S.

### URT-02: A pixel outside the sight line can change its visibility

**Where.** `T/terrain-analysis/gpu-terrain-viewshed.ts:481,519,541`, `T/terrain-analysis/terrain-sight-line.ts:251,321`, `T/terrain-analysis/gpu-terrain-line-of-sight.ts:472`, `T/terrain-analysis/gpu-terrain-cumulative-viewshed.ts:498`. The shared-model description is at `KI D/gpu-terrain/operators.md:76`.

**Evidence — EXECUTED reference; SOURCE device.** I ran the repository’s binary64 viewshed and pairwise sight-line functions with the public packers, unit cells, zero eye/target heights, and default curvature.

- DEM `[0,0,NaN]`, observer `(0,0)`, target `(1,0)`: viewshed **1** (visible), line of sight **3** (no data). The target is exactly at a finite sample. The pairwise route also tests its right neighbor, whose interpolation weight is zero.
- DEM `[[0,10,0],[0,0,0]]`, observer `(0,0)`, target `(2,0)`: **0** (hidden). Change only the bottom-right cell to NaN: **1** (visible). At the ridge `(1,0)`, that bottom-right cell has zero weight. The four-corner test discards the entire ridge sample.

**Now.** `viewshed[target]` and `lineOfSight[pair]` answer the same question differently. A no-data hole off the ray can expose a target behind a ridge.

**Wanted.** Use one positive-weight bilinear sampler for endpoints and intervening samples, including the direct viewshed path. At an integer center, test only that center. Avoid evaluating `0 * NaN`. Keep the existing policy for missing **participating** ray samples. The two fixtures above are the smallest device probes; run both march and pyramid, plus one observer in cumulative mode.

**In-repository callers that change.** Zero signature edits. `rg -o 'new (GPUTerrainViewshed|GPUTerrainLineOfSight|GPUTerrainCumulativeViewshed)\b'` found **66** constructor occurrences: 29, 19 and 18 respectively. Of these, 46 are in analysis tests, 11 in examples, six on the website and three in lenses. The changed edge-of-mask results make this **BREAKING**. **Value and cost.** Visibility users get agreement across batched, single-observer and cumulative analyses. M.

### URT-03: The zonal variance conversion only works for equal weights

**Where.** `KI D/gpu-raster/operations-analysis.md:362–368`, `R/raster-zonal-statistics/gpu-raster-zonal-statistics.ts:195,206,385`, `R/raster-zonal-statistics/raster-zonal-reductions.ts:364`. This adds a reproducible error to `KNOWN(DEC-VARIANCE)`. It is independent of the Earth-model decision in `KNOWN(RT-106)`.

**Evidence — EXECUTED.** For values `[1,3]` and reliability weights `[1,3]`, NumPy returned sample variance **2** and population variance **0.75**. The guide’s `sampleVariance * (n - 1) / n` gives **1**. The correct factor is `1 - sum(w²) / sum(w)²`, here `0.375`.

**Now.** A user following the conversion for a latitude-weighted zone gets a plausible, incorrect number. The whole-cell output offers `weightSums`. Its squared-weight sum is internal. The fractional distribution sibling already exposes `squaredWeightSums` and `populationVariances`.

**Wanted.** Qualify the count-based conversion with “equal weights only.” Give the reliability-weight formula and say which outputs suffice. An optional `populationVariances` port would remove the need for a second reduction. For the GPU check, use one zone spanning two unequal-area latitude rows and compare against `numpy.cov` with the actual two row areas and `ddof=0,1`.

**In-repository callers that change.** None. **Value and cost.** GIS users comparing population zonal statistics get the correct divisor. S for the documentation.

### URT-04: Nearest-neighbor warp does not preserve integer zone identity

**Where.** `R/gpu-raster-reprojection.ts:44,51,385,437`, `R/gpu-raster-tile-reprojection.ts:233,251,274`, `R/raster-band-utils.ts:281`. The related categorical map-algebra path is `R/raster-algebra/raster-algebra-stack-input.ts:113` and `R/raster-algebra/gpu-raster-cell-statistics.ts:328`. Compare the exact uint32 input contract at `R/gpu-raster-fractional-categorical-statistics.ts:279`.

**Evidence — EXECUTED conversion; SOURCE device.** NumPy converted uint32 `[16777216,16777217]` to float32 and back as **`[16777216,16777216]`**. In both warp routes the values have already become float32 before nearest-neighbor selection. Similarly, cell statistics performs its advertised exact equality scan after band-stack conversion. Two native categories can therefore produce `variety = 1`.

**Now.** `method: 'nearest'` is the obvious choice for aligning zone IDs. Changing interpolation cannot restore the lost bit. A later `zoneDictionary` cannot distinguish the merged IDs either.

**Wanted.** Say “continuous calibrated float32 output” directly beside the nearest example. Add a native `uint32`/`sint32` nearest-label route with identity calibration, native no-data comparison, and separate validity. Document the representability boundary on categorical cell statistics too. Device probe: identity-warp these two IDs, then count distinct values. The categorical result must retain both IDs and both validity words.

**In-repository callers that change.** None for a separate categorical route. **Value and cost.** Rasterio/GDAL users can align parcel or zone labels without changing feature identity. M; the documentation is S.

### URT-05: A missing observer height can make every target visible

**Where.** `T/terrain-analysis/gpu-terrain-viewshed.ts:121,519,548`, `T/terrain-analysis/gpu-terrain-line-of-sight.ts:144,454`, `T/terrain-analysis/terrain-sight-line.ts:291`, `T/terrain-analysis/gpu-terrain-cumulative-viewshed.ts:393`. The related light-parameter validity issue is at `T/terrain-analysis/gpu-terrain-derivatives.ts:98,445,469`.

**Evidence — EXECUTED host and reference; SOURCE device.** `getGPUTerrainViewshedParameterValues({observer:[0,0], observerHeight:NaN}, grid)` packed NaN. The binary64 direct viewshed on `[0,10,0]` returned **`[1,1,1]`**. The line-of-sight packer also accepted NaN eye height. The derivatives packer accepted `azimuthDegrees:NaN`; the shade pass sets output validity from gradient terms without testing the light angles.

**Now.** `observerHeight: heights[row]` can turn a missing height into a successful visibility classification. Height arrays supplied through graph views bypass the host packers entirely.

**Wanted.** Reject explicitly nonfinite physical settings in the public packers, naming the field. Check dynamic eye and target heights before classifying a ray; return `noData` for a pair and omit an invalid observer from cumulative counts. Check light angles when publishing hillshade. Device probe: the ridge above with a NaN eye, then a flat 3×3 DEM with NaN azimuth and `hillshade` plus `outputValidity`.

**In-repository callers that change.** None with finite settings. **Value and cost.** Missing survey heights and lighting data cannot become credible analytical outputs. S.

### URT-06: A GDAL resampling name can silently select bilinear interpolation

**Where.** `R/gpu-raster-reprojection.ts:47,249,437`, `R/gpu-raster-tile-reprojection.ts:85,278`, `R/gpu-raster-rank-filter.ts:43,63,130`. The neighboring fill contributor validates its method at `R/gpu-raster-fill-nodata.ts:85`, and focal pairs validates both policies at `R/focal-pairs/gpu-raster-focal-pairs.ts:143`.

**Evidence — EXECUTED host; SOURCE device.** From JavaScript, `new GPURasterReprojection({...props, method:'near'})` constructed and returned two graph nodes. Only the exact string `'nearest'` selects nearest. `new GPURasterRankFilter({...props, borderMode:'reflect'})` also constructed and returned two nodes; its fallback is clamp. `reflect` is a valid option on the sibling focal-pairs contributor.

**Now.** `method: gdalOptions.resampling` can blend categories while every binding and output validity flag looks ordinary. TypeScript annotations do not check JSON settings.

**Wanted.** At construction validate `method` and `noDataPolicy` on both warps and `borderMode` and `noDataPolicy` on rank. Example error: `raster-reprojection method must be nearest or bilinear (got near)`. Device discriminator for the current fallback: source `[0,100]`, destination sample at source column `0.25`; nearest gives `0`, bilinear gives `25`.

**In-repository callers that change.** None using a declared option. **Value and cost.** Configuration-driven callers get an actionable error before computing the wrong raster. S.

### URT-07: CRS key order can reject a valid warp, and a failed tiled warp leaves scratch behind

**Where.** `R/gpu-raster-reprojection.ts:294`, `R/gpu-raster-tile-reprojection.ts:90,150,233,278`. The upstream raster compatibility API is in `node_modules/@luma.gl/graph/src/gpu-raster/gpu-raster.ts:85`.

**Evidence — EXECUTED.** Two CRS descriptors with identical `authority` and `wellKnownText` but reversed property insertion order passed upstream `GPURaster.isCompatibleWith`. Direct `GPURasterReprojection` threw `sourceGrid and destinationGrid must share a coordinate reference system`. Reversing keys inside `projectionJson` also failed. For a genuinely mismatched tiled CRS, the constructor succeeded and `getCommandNodes` failed. A corrected new producer on the same graph and ID then threw **`GPUCommandGraph resource id "raster-tile-reprojection-mosaic-values" is already in use`**.

**Now.** Loading metadata from independently serialized sources can reject identical grids. Correcting the error can require recreating an otherwise usable graph.

**Wanted.** Compare declared CRS fields by value. Compare JSON object members independently of key order while preserving array order. Run the CRS/projection compatibility check in the tiled constructor before allocating mosaic scratch. The nested-JSON comparison also differs by key order upstream; propose the common fix there.

**In-repository callers that change.** None. **Value and cost.** Equivalent metadata works, and a genuine metadata error has a clean retry path. S.

### URT-08: Keep the raster object until alignment and ground units are established

**Where.** `R/raster-grid/gpu-raster-grid.ts:18,44,87,118`, `R/raster-algebra/raster-algebra-input.ts:116`, `R/raster-algebra/gpu-raster-arithmetic.ts:51`, `R/spectral-indices/raster-spectral-index.ts:58`, `D/gpu-terrain/concepts.md:12,47`. Cost evidence uses `KI R/cost-distance/cpu-cost-distance-operation.ts`. The upstream facilities already exist at `node_modules/@luma.gl/graph/src/gpu-raster/gpu-raster.ts:60,69,85`.

**Evidence — EXECUTED.**

- I wrapped two 2×2 bands from grids shifted by one column, then passed both to `GPURasterArithmetic` with the first grid. It declared **three nodes**. A default `makeGPURasterBand` result had only `id`, `format` and `storage`. With the enclosing `GPURaster` objects retained, `assertCompatibleWith` rejected the shift. In a 30 m example their first centers were `[115,185]` and `[145,185]`.
- Passing a GDAL-style signed step as `cellSize:[30,-30]` to the default north-up factory returned affine `[30,0,0,0,30,60]` for origin `[0,60]`: rows advance north.
- Two one-degree cells centered on the equator, declared `EPSG:4326`, unit friction, source cell 0: the staged CPU cost twin returned **`[0,1]`** with the default mode and **`[0,111319.4921875]`** with `'geographic'`. A 10 m projected control with friction `[1,3,1]` returned `[0,20,40]`, matching SciPy Dijkstra.

**Now.** The natural `makeGPURasterBand(values, grid)` call looks as if it retains the georeference. It uses the shape to bound storage. The separate CRS declaration also does not select the distance model.

**Wanted.** Start both family guides with a real upstream `GPURaster({metadata,bands})`. Show `first.assertCompatibleWith(second)`, `getBand('red')`, and `getPixelWorldPosition(0,0)` before extracting views. Give the GDAL coefficient mapping `[GT1,GT2,GT0,GT4,GT5,GT3]`, define `cellSize` as magnitudes when using `rowDirection`, and show `cellSizeMode` beside the CRS in the cost example. Give both the one-degree and 10 m expected answers. No new raster container is needed.

**In-repository callers that change.** None. **Value and cost.** GIS users can catch half-cell shifts, reversed rows, and degree-as-meter distances before submission. S.

### URT-09: The NDVI-to-temporal-mean recipe already exists in pieces

**Where.** `D/gpu-raster/spectral-indices.md:7,16,50`, `KI D/gpu-raster/remote-sensing.md:25`, `KI D/gpu-raster/raster-time-series.md:29`, `R/remote-sensing/gpu-raster-temporal.ts:34,216`, `R/raster-algebra/gpu-raster-cell-statistics.ts:224`, `R/raster-algebra/gpu-weighted-overlay.ts:328`. The existing adapter is `modules/analysis/src/dataframe-analysis/cube-layout/gpu-cube-layout-transpose.ts:28`.

**Evidence — COUNT, EXECUTED host and NumPy; SOURCE device.** The four documented local spectral indices use `graph.add` and support buffer or texture bands. Upstream `GPURasterNDVI` takes buffer bands and has `addToGraph`. The raster barrel does not export that class. Two date images `[0,0.2]` and `[0.6,1]` concatenated as `[0,0.2,0.6,1]` give per-cell means **`[0.1,0.8]`** when read as the temporal contributor’s cell-major layout. The expected means are **`[0.3,0.6]`**. I declared the public transpose and temporal mean on a null graph: one node each.

**Now.** The two obvious lines from the spectral page are `graph.add(new GPURasterNDVI(props))` and `values: concatenatedDateBands`. The first uses the wrong scheduling method. The second passes shape validation and changes which observations belong to each pixel.

**Wanted.** Publish the actual handoff, including the imports and the validity layout. On buffer bands, the core of the example is:

```ts
new GPURasterNDVI(ndviProps).addToGraph(graph);

graph.add(new GPUCubeLayoutTranspose({
  values: dateMajorNDVI,
  from: 'period-major',
  cellCount,
  periodCount: sliceCount,
  output: cellMajorNDVI
}));

graph.add(new GPURasterTemporal({
  cellCount, sliceCount, values: cellMajorNDVI,
  operation: 'mean', output: {values, counts, chosenSlice, status}
}));
```

Show how the per-date outputs occupy the period-major buffer. Transpose externally supplied QA the same way. For texture bands, identify the published texture gather or the existing buffer/texture-capable local normalized-difference route. Pin the two-date device readback to `[0.3,0.6]`.

**In-repository callers that change.** None. **Value and cost.** An xarray or Earth Engine user can go from acquisitions to a correct per-pixel reduction without inventing a packing kernel. S.

### URT-10: Reserve `slope` for an explicitly identified quantity

**Where.** `T/terrain-analysis/gpu-terrain-derivatives.ts:107,451`, `T/hydrology/gpu-terrain-hydrologic-indices.ts:94,141,251`, `T/terrain-illumination/gpu-solar-shadow-mask.ts:137,331`, `T/terrain-illumination/gpu-solar-irradiance.ts:266,548`.

**Evidence — EXECUTED reference and SciPy; SOURCE device.** On unit cells with each elevation row `[0,1,2]`, the two repository references returned center `slope` values **45** and **1**. Both solar consumers interpret their `slope` input as degrees. Hydrologic indices publishes the steepest downhill D8 rise/run after `minimumSlope` clamping. Even on a flat cell it can be positive.

For an independent method check, use a 3×3 zero DEM with only the northeast corner set to 8. SciPy Sobel/Horn and the repository derivative reference agreed on center slope **54.735610°**, downslope aspect **225°**, and hillshade **0.985599** for light *from* 225°, altitude 45°. The Zevenbergen–Thorne cardinal differences both give zero. This fixture detects the method, aspect sign, and lighting convention; a flat DEM detects none of them.

**Now.** `solarProps.slope = hydrologyOutput.slope` is type-correct. A tangent of 1 is read as one degree. Running `atan` also cannot turn a clamped steepest-neighbor estimate into a Horn estimate on general terrain.

**Wanted.** Rename the hydrologic output to `descentTangent` before alpha. Retain `slopeUnits` on derivatives. Label both solar input descriptions “Horn or compatible slope **in degrees**” and place the nonflat fixture in the terrain guide. Read both outputs on the two DEMs above for device confirmation.

**In-repository callers that change.** `rg 'new GPUTerrainHydrologicIndices\b'` found **11** occurrences including documentation and display strings. **Five constructor sites** actually request the renamed output: the lens example, website GPU run, kernel audit, hydrology parity spec, and the device spec’s generated output map. Two additional node-test calls supply `slope` through a fixture helper. The wetness-only examples need no edit. **Value and cost.** A hydrology-to-illumination wiring error becomes visible in review and in TypeScript. S.

### URT-11: The drainage example combines two routing models without identifying the split

**Where.** `KI D/gpu-terrain/operators.md:2009–2023`, `KI T/hydrology/gpu-terrain-flow.ts:381`, `KI T/hydrology/cpu-terrain-flow-operation.ts:213,288`. Also `R/raster-grid/gpu-raster-grid.ts:79` describes directions as an example of a band to wrap, while `T/hydrology/gpu-terrain-height-above-drainage.ts:126` takes a raw direction view.

**Evidence — EXECUTED.** On the three-cell ridge `[0,10,0]`, D8 produced directions `[0,1,0]` and accumulation `[1,1,2]`. Freeman MFD produced **the same directions** and accumulation **`[1.5,1,1.5]`**. The center’s returned pointer explains only its eastern contribution. As an encoding control, the staged watershed twin gave labels `[2,2,2]` for eastward ESRI `[1,1,0]`, Whitebox `[2,2,0]`, and TauDEM `[1,1,0]` with the respective encoding selected.

**Now.** The guide sets `flowRouting:'mfd-freeman'`, then passes the returned `flowDirections` to HAND, watersheds and stream order. A reader tracing the published directions cannot reconstruct the published accumulation.

**Wanted.** Use `'d8'` for the first end-to-end drainage example. Give the ridge as a separate branching example and label `flowDirections` “steepest D8 companion under every routing option.” Pass that view directly to HAND; wrap `filledElevation` as a band. Retain the existing encoding table and its distinction between raster offsets and compass bearings.

**In-repository callers that change.** None in implementation. **Value and cost.** WhiteboxTools and GRASS users can tell which drainage model each downstream product represents. S.

### URT-12: A search for a focal mean lands on a different operation and missing-data rule

**Where.** `D/gpu-raster/operations.md:9,26`, `D/gpu-raster/focal-pairs.md:10,45`, `KI D/gpu-raster/expansion.md:54`, `R/gpu-raster-rank-filter.ts:125`, `R/focal-pairs/gpu-raster-focal-pairs.ts:83`, `modules/analysis/src/gpu-spatial-analysis/spatial-interpolation/gpu-focal-statistics.ts:73,100`.

**Evidence — COUNT, EXECUTED reference; SOURCE device.** Three local public neighborhood operations span two entry points:

| Requested task | Public class | Default behavior at missing data or a physical edge |
|---|---|---|
| Moving mean, sum, extrema, population standard deviation | `GPUFocalStatistics` in `gpu-spatial-analysis` | Clips the window, skips invalid neighbors, may fill a missing center |
| Covariance, correlation, regression | `GPURasterFocalPairs` in `raster-analysis` | Clips by default, uses jointly valid pairs, can separately require a valid center |
| Rank or median | `GPURasterRankFilter` in `raster-analysis` | Clamps physical edges by default; a missing center stays missing even with `'ignore'` |

For `[1,NaN,3]` at radius one, SciPy’s clipped valid-sample center mean was **2**. The center rank output remains invalid by construction. With response `[2,NaN,6]`, NumPy gave sample covariance **4** and slope **2** from the surviving pair positions.

**Now.** `import {GPUFocalStatistics} from '@luma-contributors/analysis/raster-analysis'` does not resolve. The task index links paired focal statistics without identifying where to get the ordinary moving mean.

**Wanted.** Add the three-row lookup above to the raster index with actual imports. Include `maximumRadius` versus per-frame `radius`, the center policy, and the instruction to expand by the advertised halo and crop the core at internal tile boundaries. Use `[1,NaN,3]` for the device example with all three operators.

**In-repository callers that change.** None. **Value and cost.** A GIS user can choose a neighborhood statistic and predict what happens at holes and tile edges. S.

### URT-13: The CPU band’s declared format and actual sample array can disagree

**Where.** `KI R/raster-grid/raster-operation-band.ts:35,100,125`. The public pure-function inputs reach this shape through `computeTerrainFlowOnCPU`, `computeTerrainPeakProminenceOnCPU`, `computeCostDistanceOnCPU`, `computeRasterWalkingCostOnCPU`, `computeRasterWatershedSegmentationOnCPU`, `computeRasterFractionalZonalStatisticsOnCPU` and `computeRasterFractionalZonalDistributionOnCPU`.

**Evidence — EXECUTED.** I passed the staged flow twin `{id:'dem', format:'uint32', values:Float32Array.of(3.25,2.25,1.25)}`. It returned filled elevations **`[3.25,2.25,1.25]`** and accumulation `[1,2,3]`, exactly as when declaring `float32`. No uint32 band can contain the published elevations. The current type independently unions the format and array, so this input satisfies it.

**Now.** `format: rasterDataType, values: decodedSamples` can silently describe two different rasters on CPU and GPU.

**Wanted.** Check the typed-array kind once at the public CPU boundary and reject an inconsistent descriptor with, for example, `elevation.values must be Uint32Array for format uint32`. Expose a discriminated type or typed factory so new callers get the same check in the editor. Preserve subarrays and offsets.

**In-repository callers that change.** None for well-formed inputs with a runtime guard. **Value and cost.** CPU qualification can catch an ingestion mistake before it reaches the device. S.

### URT-14: A temporal medoid can select the first date solely because of measurement scale

**Where.** `R/raster-time-series/gpu-raster-medoid-composite.ts:162–173`, `R/raster-time-series/cpu-raster-medoid-composite-operation.ts:46–63`, `KI D/gpu-raster/raster-time-series.md:221`.

**Evidence — EXECUTED twin and float32 model; SOURCE device.** One cell, one band, three observations `[0,s,2s]`. The CPU twin selected slice **1** at `s=1`, `1e-25` and `1e20`. Modeling the shader’s squared distances gave candidate distance sums `[3,2,3]`, `[0,0,0]`, and `[Infinity,Infinity,Infinity]`, respectively. The shader’s first-candidate initialization and strict comparison select slice **0** in the latter two cases and publish status zero. These cases have a unique mathematical medoid and are well separated in the input format.

**Now.** Changing the measurement unit can change which acquisition is reported as provenance. The CPU TSDoc acknowledges possible distance overflow; the underflow case loses the ordering as well.

**Wanted.** Scale the observations or differences by a common finite per-cell scale before comparing candidate sums, or publish an explicit numerical-failure status when ordering is unavailable. Device probe: rerun the three one-cell stacks and require slice 1, or a documented failure code.

**In-repository callers that change.** None for a numerically stable comparison. **Value and cost.** Users retain correct acquisition provenance across rescalings. S.

## Departures by name

The census exceptions to the documented single-grid description are reproducible. They need different documentation according to whether they operate on a physical grid or on an already packed column.

- **Width and height:** `GPURasterNDWI`, `GPURasterNBR`, `GPURasterEVI`, `GPURasterSAVI`. All four operate on spatially aligned bands and benefit from the compatibility example in URT-08.
- **Grid plus cell count:** `GPUWeightedOverlay`, `GPURasterCellStatistics`, `GPURasterConditional`, `GPURasterArithmetic`. The column routes explain the independent count. The spatial-band route should tell the user when the count is derived and when a prefix is intentional.
- **Two named grids:** `GPURasterReprojection`, `GPURasterTileReprojection`. Two grids are necessary. Their CRS and preflight behavior should agree.
- **Cell count without grid:** `GPUChangeDetection`, `GPURasterTemporal`, `GPURasterSpectralUnmixing`, `GPURasterSeasonalBreak`, `GPURasterStreamingAccumulator`, `GPURasterRollingMean`, `GPURasterWhittakerSmoother`, `GPURasterHANTS`, `GPURasterTemporalResample`, `GPURasterStandardizedIndexCalibration`, `GPURasterStandardizedIndex`, `GPURasterClimateRunAccumulator`, `GPURasterClimateAnomaly`, `GPURasterMedoidComposite`, `GPURasterGeometricMedianComposite`, `GPURasterBestAvailablePixel`, `GPURasterPhenologyMetrics`, `GPURasterHarmonicRegression`, `GPURasterCOLDMonitor`, `GPURasterBFASTMonitor`, `GPURasterSnapshotCovariance`, `GPURasterEOFProjection`, `GPURasterLaggedCorrelation`. State the layout at ingestion; these operations do not all need a spatial grid.
- **Other raster shapes:** `GPUStreamlines`, `GPULagrangianFlowMap`, `GPUFiniteTimeLyapunovExponent`, `GPUParticleAdvection`, `GPURasterConservativeRemapping`, and `GPURasterEmbedding{Dequantize,Quantize,Similarity,NearestPixels,Change,KMeans,KNearestClassifier,LinearProbe,Covariance,Projection,ZoneMean,PyramidLevel}`.
- **Other terrain shapes:** `GPUProfilePeaks`, `GPUSolarPosition`. Profiles and solar observations justify exceptions to the claim that every terrain contributor takes a grid.
- **Same logical quantity or handoff with different interpretations:** the complete observed groups are the three sight-line classes in URT-02, the two slope producers and two degree consumers in URT-10, and the three focal operators in URT-12.

## Quick wins

- [ ] Correct the geometric-median source, guide and catalog using the three-date NDVI counterexample.
- [ ] Correct the geographic population-variance multiplier and identify the missing squared-weight output.
- [ ] Put the float32 label boundary beside nearest-neighbor warp and categorical cell statistics.
- [ ] Reject unsupported resampling, border and no-data policy strings before declaring graph work.
- [ ] Validate finite eye and lighting parameters after float32 conversion.
- [ ] Reject incompatible tiled CRS metadata before allocating mosaic scratch. Test a corrected retry on the same graph.
- [ ] Show the upstream raster compatibility check, GDAL affine ordering and a one-degree cost calculation on the entry pages.
- [ ] Add the checked NDVI → period-major transpose → temporal mean example, with distinct dates and cells.
- [ ] Make D8 the first end-to-end hydrology example and label the D8 companion when introducing MFD.
- [ ] Add the focal lookup table and its missing-center fixture to the raster operation index.

## Larger proposals

1. **Unify visibility sampling before alpha.** Changing the default to positive-weight support fixes both the disappearing ridge and the viewshed/LOS disagreement. An opt-in policy would preserve existing pixels but leave the ordinary path inconsistent. I recommend changing the default and pinning both march and pyramid.
2. **Provide an exact categorical warp.** A separately named or typed nearest-label route can preserve integer values, native no-data and mask semantics without changing the continuous reprojection contract. Prioritize identity and half-cell translation tests before adding more interpolation methods.
3. **Rename hydrologic `slope` to `descentTangent`.** Documentation alone leaves the degree-consuming solar inputs type-compatible with the wrong buffer. The measured migration is small enough to make the rename worthwhile before publication.
4. **Use the existing raster and cube types as the front door.** First ship the checked examples. If callers still have to repeat spatial dimensions, add grid-based overloads for the four spectral indices. Preserve the useful count-only paths for independent cells, profiles and time series.

## Not read

I read the grid/band factory, scalar raster-operation band helper, temporal composite/reduction paths used in the probes, focal-pair kernel, nearest/bilinear sampling path, and the relevant derivative and sight-line calculations. I sampled the larger operator guides by the nine requested user journeys and traced through their constructors, packers, kernels, CPU references and named consumers. I checked the synthesis, raster/terrain report, API/docs/catalog and direction reports, and the decision file for overlap. For staging-owned docs and the flow, watershed, cost-distance and CPU band paths cited above, I checked the newer `KI` copy.

The census covers both public entries. I did not inspect every embedding, illumination, relief, segmentation or climate kernel, audit every example, or qualify a release. I executed no shader, browser, build or test suite. The GPU probes above remain necessary to establish device behavior.