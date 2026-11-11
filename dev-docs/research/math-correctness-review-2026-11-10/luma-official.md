# luma.gl official implementations: library-wide review (scope `luma-official`)

Date: 2026-11-10. Baseline: installed `@luma.gl/{core,shaders,graph,engine}@10.0.0-alpha.10`.
Question: where does `@luma-contributors/analysis` do something its own way that published luma.gl
already does, and where does it do the same thing with a different convention.

Paths are relative to `modules/analysis/src` unless they start with `node_modules`, `dev-docs`,
`scripts` or `modules/`. "Official" means the published package. "Local" means this repository.

## 1. Coverage and method

What was done:

- **Official surface.** Every public entry of `@luma.gl/graph` was loaded in Node and its runtime
  exports listed (script `list-exports.mjs`). Type exports were listed with the TypeScript checker
  from each entry's `index.d.ts` (script `collisions.mjs`, 3,462 distinct exported names over
  `graph`, `core`, `shaders`, `engine`). One-line meanings were read from the declaration files.
  The shader modules of `@luma.gl/shaders` were listed with their WGSL functions.
- **Local surface.** All 1,156 `.ts` files under `modules/analysis/src` were parsed. 485 exported
  `GPU*` classes and 41 `add*Recipe` functions were listed with file and line.
- **Imports.** Every `@luma.gl/*` import in the local source was tabulated per symbol, value
  against type use (script `local-imports.mjs`). This is the evidence for every `delegates` row.
- **Name collisions.** Every local exported name was intersected with the official name set.
- **Rules.** `node scripts/check-upstream-first.mjs` was run. A second scan looked for what that
  script does not check: `export {X}` of an imported luma.gl symbol, pure type aliases, `const`
  aliases, subclasses, dynamic imports, and local bodies identical to official `dist` code.
- **Conventions.** Whole-library counts (all 485 classes, all 2,502 `createWGSLKernelNode` calls)
  plus the class documentation, props and parts of the kernels of 47 contributors in all seven
  source directories (list at the end of this section).
- **Register.** Every row of `dev-docs/upstream-workarounds.md` was checked against the installed
  declarations and against the cited file and line.

Scratch scripts are in the session scratchpad under `luma-official/`. No file in the repository was
changed except this report. No GPU run and no spec run was made (0 of 5 allowed runs used).

Files read in full: `analysis-core/contracts/earth.ts`, `raster-analysis/gpu-raster-rank-filter.ts`,
`analysis-core/kernels/masked-weak-components.ts`, `AGENTS.md`, the four `dev-docs` files named in
the task. Official files read in full: `gpu-project/web-mercator.js`,
`gpu-core/graph-data-view-utils.js` (first 150 lines, all validation helpers).

Files read in part (class documentation, props, and the kernel section that matters to the match):
the 47 contributors below, `analysis-core/kernels/{sphere-distance-wgsl, uint64-wgsl,
gpu-contributor-utils, sorted-segment-offsets, scan-total, stable-lexicographic-index-sort,
load-balanced-for-each, philox-random}.ts`, `analysis-core/{network-topology, cell-key-wgsl,
index}.ts`. Official kernels read in part: `geospatial/gpu-haversine-distance.js`,
`gpu-dggs/gpu-dggs-cell-projection.js`, `gpu-graph-algorithms/gpu-graph-connected-components-internals.js`,
`gpu-graph-algorithms/gpu-graph-modularity-internals.js`, `gpu-raster/gpu-raster-contours.js`,
`gpu-raster/gpu-raster-band-math.js`, `gpu-core/gpu-grid-index-internals.js`,
`gpu-core/gpu-grid-binning.js`, `gpu-core/gpu-grid-aggregation.js`, and about 25 declaration files.

Sampled contributors (47): `GPUKMeans`, `GPUGeodesicPairs`, `GPUCellGeometry`, `GPUPointToCell`,
`GPUPointInPolygonJoin`, `GPUSpatialPredicateJoin`, `GPUFocalStatistics`, `GPUBoundsFilter`,
`GPUNeighborSearch`, `GPUCellAggregation`, `GPUSimilarLocations`, `GPURegionStatistics`,
`GPUInverseDistanceWeighting`, `GPUPointDensity`, `GPUHilbertKeys`, `GPUTrajectoryEncounters`
(gpu-spatial-analysis); `GPUNetworkAnalyticsColumns`, `GPUNetworkStatistics`,
`GPUNetworkReachability`, `GPUNetworkNeighborhood`, `GPUNetworkLouvain`, `GPUFlowAggregation`,
`GPUNetworkCentrality` (gpu-network); `GPUTerrainDerivatives`, `GPUTerrainSpikeRepair`,
`GPUTerrainContours`, `GPUTerrainCurvature`, `GPUReliefShading` (gpu-terrain); `GPUKeyJoin`,
`GPUGroupStatistics`, `GPURipley` (dataframe-analysis); `GPURasterRankFilter`, `GPURasterNDWI`,
`GPURasterNBR`, `GPURasterEVI`, `GPURasterSAVI`, `GPURasterArithmetic`, `GPURasterStretch`,
`GPUIsolines`, `GPUIsobands`, `GPURasterExtremaPyramid`, `GPURasterEmbeddingPyramidLevel`,
`GPURasterEmbeddingSimilarity`, `GPURasterEmbeddingKMeans`, `GPURasterReprojection`
(raster-analysis); `GPUPointCloudNeighborhoods`, `GPUPointCloudStatisticalFilter`,
`GPUPointCloudVoxelMerge`, `GPUPointCloudSMRF`, `GPUPointCloudGround` (pointcloud-analysis);
`GPUPointBuffer`, `GPUDatumGridTransform` (geospatial-analysis).

Not covered:

- No pair of official and local kernels was run on a GPU. "Same math" is `yes` only where both
  kernels were read or replicated in Node. Everything else says `unchecked`.
- `modules/lenses` and `modules/kepler` were only scanned by the guard scripts.
- `@luma.gl/engine` was not inventoried (the analysis source does not import it).
- `gpu-dataframe`, `gpu-sql`, `gpu-crossfilter`, `gpu-tables`, `arrow`: class list only, no
  implementation read.
- Local point-to-segment distance, local gather kernels and local run-length code were not compared
  with `GPUPairwisePointSegmentDistance`, `GPUGather` and `GPURunLengthEncode`.
- The examples and the docs under `docs/api` were not reviewed.

## 2. Official surface

`@luma.gl/graph` has 38 `exports` keys. Runtime exports per entry that matters to analysis code
(C classes, F functions; benchmark and stats helpers left out):

| Entry | Classes and functions |
| --- | --- |
| `gpu-core` (97 C, 54 F) | Graph: `GPUCommandGraph`, `CompiledGPUCommandGraph`, `GraphDataView`, `GraphVectorView`, `GraphBufferHandle`, `GraphTextureView`, `createTransientView`, `createTransientVectorView`, `createGPUComputeCommandNode`, `createGPUCopyCommandNode`, `createGPURenderCommandNode`, `getViewBinding`, `getViewBindingRange`, `getViewElementOffset`, `doGraphDataViewsOverlap`, `getBoundedDispatchLayout`, `getBoundedInvocationIndexSource`, `validatePackedView`, `validatePackedUint32View`, `validateMatchingVectorTopology`. Primitives: `GPUReduction`, `GPUScan`, `GPUScanUint64`, `GPUSort`, `GPUBatchSort`, `GPUSegmentedSort`, `GPUCompaction`, `GPUFlagOffsets`, `GPUIndexedRangeCompaction`, `GPUMask`, `GPUVisibilityWorkflow`, `GPUGather`, `GPUUint32Gather`, `GPUByteRangeGather`, `GPUChunkedIndexedScatter`, `GPUCOOToCSR`, `GPUSegmentOffsets`, `GPUSegmentedLayout`, `GPURunLengthEncode`, `GPUUnique`, `GPUGallopingSearch`, `GPUHashIndex`, `GPUHashIndexQuery`, `GPUHashJoin`, `GPUBatchHashIndex`, `GPUBatchHashJoin`, `GPUBVH`, `GPUBVHQuery`, `GPUSegmentedBVH`, `GPUGridIndex`, `GPUGridIndexQuery`, `GPUGridBinning`, `GPUGridAggregation`, `GPUGroupAggregation`, `GPUHistogram`, `GPUPointSpatialFilter`, `GPUElementwise`, `GPUConvolution`, `GPUFFT1D`, `GPUFFT2D`, `GPUFiniteDifference2D`, `GPUFiniteDifference3D`, `GPUTranspose`, `GPUMatMul`, `GPUMatVec`, `GPUGraphTraversal`, `GPUReadbackRing`. Semantic programs: `GPUGraph`, `GPUGraphCSRMatrix`, `GPUGraphSpMV`, `GPUGraphDotProduct`, `GPUGraphVectorMADD`, `createGPUConjugateGradientGraph`. |
| `geospatial` (8 C) | `GPUHaversineDistance`, `GPUPairwisePointDistance`, `GPUPairwisePointSegmentDistance`, `GPUPairwisePointLinestringNearest`, `GPUPairwisePointInPolygon` (+ `GPU_POINT_IN_POLYGON_CLASSIFICATION`), `GPUPointSpatialQuery`, `GPUSinusoidalProjection`, `GPUGridIndex` (same class as `gpu-core`). |
| `gpu-graph-algorithms` (14 C) | `GPUGraphAlgorithmsData`, `...Topology`, `...Degree`, `...BreadthFirstSearch`, `...SingleSourceShortestPath`, `...ConnectedComponents`, `...PageRank`, `...CoreNumber`, `...LabelPropagation`, `...LocalClusteringCoefficient`, `...Modularity`, `...ModularityOptimization`, `...ForceLayout`, `...SpatialForceLayout`. |
| `gpu-vector-search` (3 C, 2 F) | `GPUKMeans`, `GPUSimilaritySearch`, `GPUIVFFlatIndex`, `importGPUEmbeddingTable`, `importGPUEmbeddingVector`. |
| `gpu-dggs`, `gpu-h3`, `gpu-a5` | `GPUDGGSCellProjection`, `GPUH3CellProjection`, `GPUA5CellProjection` (cell to centre, as longitude and latitude or unit vector). |
| `gpu-timeseries` (2 C) | `GPUTimeSeries`, `GPUTimeSeriesEnvelope` (uniform samples and min/max envelopes for charts). |
| `gpu-pointcloud` (3 C) | `GPUPointCloudOctree`, `GPUPointCloudRadiusFilter`, `GPUPointCloudVoxelDownsample`. |
| `gpu-raster` (44 C, 4 F) | Band math: `GPURasterBandMath`, `GPURasterNDVI`, `GPURasterThreshold`, `GPURasterOtsuThreshold`, `GPURasterContrast`. Stencils: `GPURasterNeighborhood`, `GPURasterConvolution`, `GPURasterBoxBlur`, `GPURasterGaussianBlur`, `GPURasterGradient`, `GPURasterSobel`, `GPURasterScharr`, `GPURasterGradientMagnitude`, `GPURasterLaplacian`. Morphology: `GPURasterMorphology`, `GPURasterErosion`, `GPURasterDilation`, `GPURasterOpening`, `GPURasterClosing`. Statistics: `GPURasterStatistics`, `GPURasterHistogram`, `GPURasterGlobalStatisticsMerge`, `GPURasterGlobalHistogramMerge`, `GPURasterGlobalPercentile`, `GPURasterGlobalInitialize`. Regions: `GPURasterConnectedComponents`, `GPURasterDenseComponents`, `GPURasterCrossTileComponents`, `GPURasterRegionMeasurements`. Geometry: `GPURasterContours`, `GPURasterContourClassifier`. Pyramids: `GPURasterOverview`, `GPURasterCategoricalOverview`. Transfer and tiles: `GPURasterBufferToTexture`, `GPURasterTextureToBuffer`, `GPURasterTile*` (8). Helpers: `planRasterDispatchStripes`, `getRasterDeviceLimits`, `makeRasterOverviewMetadata`, `getRasterRegionWorldCentroid`. |
| `gpu-project` (5 C, 14 F), `gpu-project/crs` | `GPUProjection`, `GPUProjectionProgram`, `GPUProjectionTable`, `CompiledProjection`, `compileProjectionPlan`, `compileProjectionProgram`, `planProjection`, `createWebMercatorProjection`, `WEB_MERCATOR_EARTH_RADIUS`, `WEB_MERCATOR_MAX_LATITUDE`; `planCRSProjection`, `prepareCRSProjection`, `planProjectionPipeline`. |
| `gpu-dataframe` (19 C, 6 F) | `GPUDataFrame`, query and compiled classes for aggregation, grouped aggregation, histogram, join, lookup, sort, global sort; `GPUExpression`, `column`, `literal`, `parameter`, `and`, `or`, `not`. |
| `gpu-sql`, `gpu-crossfilter` | `GPUSQLContext`, `GPUSQLQuery`; `GPUCrossfilter`, `GPUCrossfilterSelection`. |
| `gpu-data`, `arrow` | `GPUVector`, `GPUData`, `GPUDataView`, format helpers; 71 Arrow conversion functions. |
| `lucim` (9 C) | Volume threshold, morphology, connected components, region measurements. Not used here. |

`@luma.gl/shaders` WGSL modules that analysis kernels could use: `dggs` (159 functions: 64-bit
words, H3, A5, S2, geohash, quadkey decoders), `fp64arithmetic` (111 functions: binary64 decoding,
64-bit words, `twoSum`, `twoProd`, `sum_fp64`, `mul_fp64`, `div_fp64`, `sqrt_fp64`), `fp64`
(`sin_fp64`, `cos_fp64`, `fp64_sincos`, `exp_fp64`, `log_fp64`, `radians_fp64`), `fp32`, `random`
(one `fract(sin(dot))` hash). There is no `math` and no `project` module in alpha.10.

Not public in alpha.10 (checked by name in every entry): `GPUSegmentedReduction`, `GPUScatter`,
`GPUSegmentedScan`, `GPUJacobiPCG`, `getSpatialOperationNodes`, `getBoundedDispatchOverrides`,
`GPU_GRID_COORDINATE_WGSL`, `makeGPURasterBand`, `makeGPURasterMetadata`, `validateRasterBand`,
`validateRasterMetadata`, `validateRasterExtent`, `getRasterDispatchSize`, `getRasterLocalOrigin`,
`assertRasterStorageBindingFits`, `prepareArrowTableForGPUAnalytics`,
`getGPUPartitionDescriptorFromArrowTable`, `makeGeometryGPUVectors`,
`GPUGraphAlgorithmsStronglyConnectedComponents`, `makeGPUQueryExpressionShaderPlan`,
`addGPUQueryPredicatePass`, `OutOfMemoryError`.

## 3. Match table

Classification counts:

| Classification | Count | Meaning |
| --- | --- | --- |
| `delegates` | 36 official symbols | Local code constructs or calls the official class. Confirmed by a value import. |
| `wraps-and-changes` | 3 | Delegates, with a change of meaning or a bypass. |
| `re-implements` | 27 rows | Own kernel for the same operation. 9 state the reason in code, 2 are in the register, 16 state nothing. |
| `name-collision` | 3 exported names, 2 concepts | `GPUKMeans`, `GPUKMeansProps`, `validatePackedView`. |

### 3.1 Delegates

| Official | Local callers | Note |
| --- | --- | --- |
| `GPUScan`, `GPUScanUint64` | 87 files, 2 files | |
| `GPUSort` | 62 files | |
| `GPUReduction` | 32 files | Float sums are not compensated (register, section 3). |
| `GPUCompaction`, `GPUFlagOffsets` | 19 files, 3 files | |
| `GPUGroupAggregation` | 10 files | Also re-implemented, see 3.3. |
| `GPUGridIndex` | 8 files | Also re-implemented, see 3.3. |
| `GPUVisibilityWorkflow` | 7 files | |
| `GPUHistogram` | 4 files | |
| `GPUBVH` | `gpu-spatial-analysis/segment-intersection/segment-bvh.ts:154`, `spatial-join/spatial-join-passes.ts:135`, `spatial-weights/gpu-fuzzy-contiguity.ts:134` | Traversal is local WGSL over the public caller-owned node views. |
| `GPUGridBinning`, `GPUGridAggregation`, `GPUConvolution` | `gpu-spatial-analysis/point-density/gpu-point-density.ts:590`, `:601`, `:647` | See finding 02. |
| `GPUHashIndex`, `GPUHashIndexQuery` | `gpu-network/flow-aggregation/gpu-flow-aggregation.ts` | |
| `GPUGather`, `GPUUint32Gather` | 1 file, 2 files | Nine local `create*GatherNode` helpers also exist. Not compared. |
| `GPUCOOToCSR` | `gpu-spatial-analysis/landscape-ecology/sparse-resistance-system.ts` | |
| `GPUFFT2D` | `raster-analysis/remote-sensing/gpu-raster-registration.ts`, `raster-masked-registration.ts` | No local FFT exists. |
| `GPUTranspose` | `gpu-terrain/topographic-position/terrain-summed-area-table.ts` | |
| `GPUPointSpatialFilter`, `GPUReadbackRing` | `gpu-spatial-analysis/region-statistics/*` | |
| `GPUPairwisePointInPolygon` | `gpu-spatial-analysis/spatial-join/gpu-spatial-predicate-join.ts:1311` | |
| `GPUKMeans` (official) | `raster-analysis/raster-embeddings/gpu-raster-embedding-kmeans.ts:6` | |
| `GPUSimilaritySearch` | `raster-analysis/raster-embeddings/gpu-raster-embedding-nearest-pixels.ts:6` | Local cosine rule follows the official one (`gpu-raster-embedding-similarity.ts:101`). |
| `GPUProjection`, `compileProjectionPlan` | `gpu-spatial-analysis/geometry-measures/gpu-local-frame-projection.ts` | |
| `GPUProjectionProgram` | `raster-analysis/gpu-raster-reprojection.ts` | |
| `GPURasterGradient` | `gpu-terrain/terrain-analysis/gpu-terrain-derivatives.ts:239`, `terrain-illumination/gpu-relief-shading.ts:460` | Checked: raw Sobel response is divided by `8 * cell` (`gpu-terrain-derivatives.ts:366-367`), and both gradient values are tested for finiteness (`:375-377`), so using only the x validity is safe. |
| `GPURasterContours` | `gpu-terrain/terrain-analysis/gpu-terrain-contours.ts:186` | Maps official `segmentCount`, `overflow`, `requiredSegmentCount` to `status.count`, `status.overflow`, `status.requiredCount`. |
| `GPURasterBandMath` | `raster-analysis/gpu-raster-spectral-indices.ts:79-113` (`GPURasterNDWI`, `NBR`, `EVI`, `SAVI`) | These four classes are written in the official style. See 5.1. |
| `GPURasterHistogram` | `gpu-spatial-analysis/multiple-testing/gpu-raster-false-discovery-rate-mask.ts` | |
| `GPURasterNeighborhood` | `raster-analysis/raster-band-utils.ts:212`, `gpu-spatial-analysis/recipes/raster-summary-recipes.ts:32` | |
| `GPURasterBufferToTexture`, `GPURasterTextureToBuffer` | 12 files, 1 file | |
| `dggs` shader module | 21 files under `gpu-spatial-analysis/cell-*` | Used as `dggs.source` text. See finding 06. |
| `getBoundedDispatchLayout`, `getViewBinding`, `getViewElementOffset`, `createGPUComputeCommandNode` | `analysis-core/kernels/wgsl-kernel-nodes.ts` and 7 other files | |

### 3.2 Wraps and changes

| Official | Local | What changes |
| --- | --- | --- |
| `validatePackedView` | `analysis-core/kernels/gpu-contributor-utils.ts:209` | Same name. Adds one check that the official function already performs. Finding 07. |
| `GPUGraphAlgorithmsConnectedComponents` | `gpu-terrain/terrain-decode/gpu-terrain-spike-repair.ts:496-509` through `analysis-core/network-topology.ts:193-245` | Receives an object cast to `GPUGraphAlgorithmsTopology` that was not built by the official constructor. Finding 03. |
| `dggs_h3_get_center_lnglat`, `dggs_a5_get_center_lnglat` (the kernels of `GPUDGGSCellProjection`) | `gpu-spatial-analysis/cell-indexing/gpu-cell-geometry.ts:131` | Same decoder, different result for invalid cells. Finding 09. |

### 3.3 Re-implements and name collisions

"Reason stated" means the local file says why the official class is not used.

| Official | Local counterpart | Class | Same math | Can delegate today | Missing upstream |
| --- | --- | --- | --- | --- | --- |
| `GPUKMeans` (`gpu-vector-search`) | `gpu-spatial-analysis/spatial-clustering/gpu-kmeans.ts:273` | name-collision, re-implements | no: seeding, stop rule and empty-cluster rule differ (finding 04) | no | k-means++ seeding, a centre-shift tolerance, restarts, per-row squared distance |
| `validatePackedView` (`gpu-core`) | `analysis-core/kernels/gpu-contributor-utils.ts:209` | name-collision | yes for packed views; messages differ for chunked vectors | yes | nothing |
| `GPUHaversineDistance` | `gpu-spatial-analysis/geometry-measures/gpu-geodesic-pairs.ts:95` (`'sphere'` model, `geodesic-wgsl.ts:132-150`) | re-implements, no reason stated | yes for the central angle (float64 replication, 6 pairs); default radius and unit differ (finding 08) | yes for distance, with `radius` passed | bearings and midpoints are local-only and not needed upstream |
| `GPUDGGSCellProjection`, `GPUH3CellProjection`, `GPUA5CellProjection` | `gpu-spatial-analysis/cell-indexing/gpu-cell-geometry.ts:131` | re-implements the contributor over the official shader functions | yes for valid cells; invalid cells differ (finding 09) | yes for H3 and A5 centres | boundaries, vertex counts, S2, geohash, quadkey and Quadbin as command nodes |
| `GPURasterContours` | `raster-analysis/isolines/gpu-isolines.ts:152`, `gpu-isobands.ts:149` | re-implements, no reason stated | no on saddle cells (finding 01, same as `raster-fields-06`) | no | several levels in one pass, world coordinates, stitched polylines, bands |
| `GPURasterBandMath` | `raster-analysis/raster-algebra/gpu-raster-arithmetic.ts:63` | re-implements, no reason stated | yes at the default `epsilon` of 0 for add, subtract, multiply, divide, normalized difference (both reject a zero denominator); unchecked beyond that | partly | operation chosen per frame, unary operations, power, minimum, maximum |
| `GPURasterContrast`, `GPURasterStatistics`, `GPUHistogram` | `raster-analysis/raster-stretch/gpu-raster-stretch.ts:125` | re-implements; reason stated for `GPURasterStatistics` (`:304`) and `GPUHistogram` (`:411`), none for `GPURasterContrast` | gamma rule is the same (`t ** (1 / gamma)` in both); output differs: official keeps the calibrated domain, local writes `[0, 1]` | no | percentile bounds, window and region mask, colour output |
| `GPURasterNeighborhood`, `GPURasterBoxBlur`, `GPURasterConvolution` | `gpu-spatial-analysis/spatial-interpolation/gpu-focal-statistics.ts:99` | re-implements, no reason stated | unchecked. Policies differ: official keeps an invalid centre invalid; local fills it unless `propagateCenterNoData` | partly (mean only, radius up to 8) | minimum, maximum, range, standard deviation, valid count, per-frame radius |
| `GPURasterErosion`, `GPURasterDilation`, `GPURasterOpening` | `pointcloud-analysis/gpu-pointcloud-smrf.ts:268-345`, `gpu-pointcloud-ground.ts:57`; `raster-analysis/gpu-raster-rank-filter.ts:45` at quantile 0 or 1 | re-implements, no reason stated | unchecked | partly (official radius is 0 to 8; a larger diamond is a repeated radius-1 pass) | a radius above 8 |
| `GPUHashJoin`, `GPUDataFrameJoinQuery`, `GPUDataFrameLookupQuery` | `dataframe-analysis/key-join/gpu-key-join.ts:140` | re-implements (sort and binary search), no reason stated | unchecked | no | 64-bit keys, row masks, repeated right keys with 1:n aggregates |
| `GPUGroupAggregation`, `GPUDataFrameGroupedAggregationQuery` | `dataframe-analysis/group-statistics/gpu-group-statistics.ts:249`, `gpu-spatial-analysis/cell-aggregation/gpu-cell-aggregation.ts:116` | re-implements (sort and segments), no reason stated | unchecked | no | sparse 64-bit keys, order statistics, exact fixed-point sums |
| `GPUGridIndex` | `pointcloud-analysis/pointcloud-grid-index.ts:85` | re-implements, reason stated (`:64-83`, six gaps) | not intended to be the same | no | a cell size, a half-open domain, stable order inside a cell, a Z-ignoring mode, named limits, plain read views |
| `GPUGridIndexQuery`, and the private cell mapping of `GPUGridIndex` and `GPUGridBinning` | five inline copies (finding 02) | re-implements, copies official private WGSL | four identical, one copy disagrees with the class it claims to follow | no | a public cell-coordinate WGSL snippet |
| `GPUGallopingSearch`, `GPUSegmentOffsets` | `analysis-core/kernels/sorted-segment-offsets.ts:26` (13 callers), `binary-search-wgsl.ts` | re-implements, reason stated (`sorted-segment-offsets.ts:21-23`) | unchecked | yes, at a cost of extra passes and buffers | a one-dispatch lower-bound form |
| `GPUGraphAlgorithmsConnectedComponents` | `analysis-core/kernels/masked-weak-components.ts:34` | re-implements, reason stated (`:28-31`) | yes once converged: both leave the smallest vertex index of the component. Round counts differ (official hooks and jumps, local propagates with one shortcut) | no | CSR as command-graph views, masked slots |
| `GPUGraphAlgorithmsSingleSourceShortestPath` | `gpu-network/network-reachability/gpu-network-reachability.ts:282` | re-implements, reason stated (`:272-280`) | unchecked | no | several sources with start costs, per-frame views, compact frontier |
| `GPUGraphAlgorithmsBreadthFirstSearch`, `GPUGraphTraversal` | `gpu-network/network-analysis/gpu-network-neighborhood.ts:117` | re-implements, reason stated (`:105-108`) | unchecked | no | hop distances on command-graph views |
| `GPUGraphAlgorithmsDegree` | `gpu-network/network-statistics/gpu-network-statistics.ts:271`, `network-analysis/gpu-network-analytics-columns.ts:117` | re-implements, reason stated (`gpu-network-statistics.ts:242-246`) | no by design: local counts live slots after masks | no | masks |
| `GPUGraphAlgorithmsModularity` | `gpu-network/network-statistics/gpu-network-statistics.ts:252-260` | re-implements, reason stated | yes for unweighted graphs: same formula, same `resolution`, self-loop counted twice in the volume. Official is weighted and order dependent; local is unweighted and exact | no | deterministic accumulation, command-graph views |
| `GPUGraphAlgorithmsModularityOptimization` | `gpu-network/network-optimization/gpu-network-louvain.ts:67` (at most 32 vertices) | re-implements, reason partly stated (`:65`) | unchecked | no | multilevel coarsening |
| `createGPUConjugateGradientGraph`, `GPUGraphSpMV` | `gpu-spatial-analysis/landscape-ecology/sparse-resistance-solver.ts` | re-implements, in the register | unchecked | no (register) | lowering into a caller graph |
| `GPUSegmentedReduction` (not public) | `analysis-core/kernels/sorted-segment-sums.ts` | re-implements, in the register | register says bit-identical; unchecked | no | public class |
| `GPUPairwisePointInPolygon` | inline classifier of `gpu-spatial-analysis/spatial-join/gpu-point-in-polygon-join.ts:636-703` | re-implements, no reason stated; uses the official result codes plus one extra bit | unchecked (finding 12) | no | candidate pairs from an index, a boundary rule |
| `fp64arithmetic`, `fp64` shader modules | `gpu-spatial-analysis/cell-indexing/h3-index-extended-wgsl.ts:113-216` (two-sum, two-product, add, multiply, divide, sine and cosine) | re-implements, no reason stated | unchecked | partly (finding 10) | public selection of the platform defines |
| `dggs_u64_*`, `fp64_u64_*` | `analysis-core/kernels/uint64-wgsl.ts:31`, `analysis-core/cell-key-wgsl.ts:35`, `gpu-terrain/topographic-position/terrain-summed-area-table.ts:56`, `dataframe-analysis/pair-statistics/gpu-ripley.ts:416`, `dataframe-analysis/calendar-buckets/gpu-calendar-buckets.ts` | re-implements | yes for add and subtract; word order differs (finding 05) | partly | a small standalone 64-bit word module with multiply and divide |
| `random` shader module | `analysis-core/kernels/philox-random.ts` | re-implements, reason stated (`:8`) | not intended; the official hash is not a reproducible generator | no | a counter-based generator |
| `GPURunLengthEncode`, `GPUUnique` | run heads plus `GPUScan` inside `gpu-key-join.ts` and `group-statistics-*.ts` | unchecked | unchecked | unchecked | |

Official classes with no local code and no local use: `GPUSinusoidalProjection` (one dead local
constant, finding 08), `GPUPairwisePointDistance`, `GPUPointSpatialQuery`, `GPUFFT1D`,
`GPUFiniteDifference2D/3D`, `GPUMatMul`, `GPUMatVec`, `GPUElementwise`, `GPUTimeSeries*`,
`GPUPointCloudOctree`, `GPUPointCloudRadiusFilter`, `GPUPointCloudVoxelDownsample`, PageRank, core
number, label propagation, local clustering coefficient, and the `gpu-raster` statistics, overview,
component, threshold and blur classes. The coverage matrix lists these as "Published".

## 4. Findings

Severity uses the brief's vocabulary plus `rule` for a departure from the upstream-first rules that
gives correct numbers today.

| ID | Severity | Status | `file:line` | What the local code does | What official does | Concrete input | Suggested fix |
| --- | --- | --- | --- | --- | --- | --- | --- |
| luma-official-01 | inconsistent | CONFIRMED (Node replication of both rules). Same defect as `raster-fields-06`. | `raster-analysis/isolines/gpu-isolines.ts:409`, `:960-962`; `isolines/isobands-wgsl.ts:302-304`; against `gpu-terrain/terrain-analysis/gpu-terrain-contours.ts:186` | `GPUIsolines` and `GPUIsobands` join the high corners of a saddle when the mean of the four corners is at or above the level. | `GPURasterContours` (which `GPUTerrainContours` delegates to) uses the sign of `(a - L)(c - L) - (b - L)(d - L)`, the bilinear saddle value (`node_modules/@luma.gl/graph/dist/gpu-raster/gpu-raster-contours.js:440-449`). | Cell top-left 100, top-right -9, bottom-right 1.1, bottom-left -9, level 1. Determinant is -0.00901, so official separates the two high corners. Mean is 20.775, so local joins them. | Use the bilinear determinant in the local kernels. It is exact for the bilinear surface both draw. |
| luma-official-02 | rule, and wrong-edge-case in one copy | CONFIRMED (script compares function bodies; float32 replication for the mismatch). The mismatch is `interpolation-density-01`. | `gpu-spatial-analysis/region-statistics/region-statistics-grid.ts:172`, `spatial-interpolation/gpu-inverse-distance-weighting.ts:376`, `trajectory-encounters/encounter-kernels.ts:143`, `point-density/point-density-kernels.ts:279`, `gpu-network/flow-aggregation/flow-aggregation-kernels.ts:124` | Five private copies of the official cell-mapping function `getCoordinate`. Four are character-identical (ignoring whitespace) to `gpu-grid-index-internals.js:180-194`. The fifth is identical to `gpu-grid-binning.js:193-197`. | The function is not exported. `GPUGridIndex` uses a variant that divides by `max(abs(min), abs(max))` when the bounds straddle zero. `GPUGridBinning` and `GPUGridAggregation` do not. | `point-density-kernels.ts:231` says "same cell arithmetic as `GPUGridBinning`" but holds the `GPUGridIndex` variant. Bounds `[-180, 180]`, 360 columns, x = -177: `GPUGridBinning` gives column 3, the copy gives column 2. 967 of 12,407 values next to cell edges disagree over seven tested bounds. With weights and the default sum mode, the count goes to one cell and the weight to its neighbour (`gpu-point-density.ts:590`, `:617`). | Point density: use the plain formula, or take counts from the same keys. All five: propose a public cell-coordinate snippet upstream; until then, add a test that compares each copy with the text of the official kernel so a luma.gl bump cannot change one side silently. |
| luma-official-03 | rule | CONFIRMED (read) | `analysis-core/network-topology.ts:193-245`; used at `gpu-terrain/terrain-decode/gpu-terrain-spike-repair.ts:496` | Builds a plain object and casts it `as unknown as GPUGraphAlgorithmsTopology`. `edgeIds` aliases `neighbors`. `count`, `overflow` and `invalidEdgeCount` share one word. | The official constructor checks that these are distinct allocations. The object skips it. | None: results are correct today. The comment at `:186-188` states the dependency: "Every gpu-graph algorithm reads only ..." That is a fact about official internals, not public API. | Use `createMaskedWeakComponentNodes` here too, or propose upstream a topology constructor that accepts a caller-built CSR. Add the row to the register. |
| luma-official-04 | inconsistent, doc-mismatch | CONFIRMED (read both declarations; path checked against `package.json` exports) | `gpu-spatial-analysis/spatial-clustering/gpu-kmeans.ts:175`, `:234`, `:273`; official used at `raster-analysis/raster-embeddings/gpu-raster-embedding-kmeans.ts:6` | The package exports its own `GPUKMeans` and also uses the official `GPUKMeans`. Prop names differ for the same things: `k` against `clusterCount`, `maximumIterations` against `maxIterations`, `output.centers` against `centroids`, `output.sizes` against `counts`, two status scalars against one three-word `status` view. | Evenly spaced seeds, stops when no label changes, an empty cluster keeps its centroid. Local: k-means++ from a seed, stops on centre shift, relocates empty clusters. | The doc at `:234` sends the reader to `@luma-contributors/analysis/gpu-vector-search`. That entry does not exist. The class is at `@luma.gl/graph/gpu-vector-search`. | Rename the local class for what it is (2D points, k-means++), for example `GPUPointKMeans`. Fix the path. Use the official prop names where the concept is the same. |
| luma-official-05 | inconsistent | CONFIRMED (read) | `analysis-core/kernels/uint64-wgsl.ts:31-35`; `analysis-core/cell-key-wgsl.ts:35`; `gpu-terrain/topographic-position/terrain-summed-area-table.ts:56`; `dataframe-analysis/pair-statistics/gpu-ripley.ts:416` | The shared helper `UINT64_WGSL` holds a 64-bit value as `vec2<u32>(low, high)`. So do the private copies in terrain and Ripley. `CELL_KEY_WGSL` uses `vec2u(high, low)`. | Both official modules (`dggs_u64_*`, `fp64_u64_*`) use `vec2u(high, low)` and convert at the storage boundary with `dggs_u64_from_little_endian_words`. | Ten cell kernels concatenate `dggs.source` with `CELL_KEY_WGSL`. Adding `UINT64_WGSL` to one of them would mix two orders in one shader. `UINT64_WGSL` has no caller in `src` yet, so the order can still be changed for free. | Flip `UINT64_WGSL` to `(high, low)` before the private copies move onto it. Propose a small 64-bit word module upstream. |
| luma-official-06 | rule | CONFIRMED that the code parses shader text; SUSPECTED for the failure mode | `gpu-spatial-analysis/cell-indexing/h3-index-extended-wgsl.ts:11-17`, `:28-29` | `parseDggsTable` finds `fn dggs_h3_get_face_center_geo` and `fn dggs_h3_get_face_axis_azimuths` inside `dggs.source` with `indexOf` and a regular expression, and builds float64 constants from the literals. | The module's public contract is its WGSL functions, not the layout of its text. | Replicated in Node with the installed module: today 20 and 20 rows. With a function name that does not exist: 0 rows and no error. The WGSL then declares `array<vec2f, 180>()` with no elements. | Check that both tables have 20 rows at module load and throw. Propose upstream that the face tables be exported as data. |
| luma-official-07 | inconsistent | CONFIRMED (Node script calls the official function with a `GraphVectorView`) | `analysis-core/kernels/gpu-contributor-utils.ts:209` | Local `validatePackedView` rejects a `GraphVectorView` with a short message, then calls the official function. 376 files import the local one. 14 files import the official one of the same name. 50 files import the official `validatePackedUint32View`. | The official function already throws for a `GraphVectorView` (its `byteOffset` is undefined): "must be packed, uint32-aligned float32 GPU data". | A chunked vector passed to the two functions gives two different messages for one mistake, depending on which file is called. | Delete the local wrapper and import the official one, or rename it so the two cannot be confused. |
| luma-official-08 | inconsistent (documented convention, not a defect) | CONFIRMED (float64 replication of both formulas on 6 pairs; central angles agree to 3e-16) | `gpu-spatial-analysis/geometry-measures/gpu-geodesic-pairs.ts:95`; `analysis-core/contracts/earth.ts:9`, `:28` | Great-circle distance in the context's unit on radius 6,371,008.8 m. | `GPUHaversineDistance` defaults to radius 6371 and returns kilometres. | New York (-73.98, 40.75) to Paris (2.35, 48.85): official default 5833.150 km, local 5,833,158.6 m. The 8.1 m is the radius only. `earth.ts:28` keeps `GPU_SINUSOIDAL_KILOMETERS_PER_DEGREE` "for `GPUSinusoidalProjection` parity"; no source file uses it (one spec does). | Say in the `GPUGeodesicPairs` documentation that the official class exists and differs by default radius and unit. Remove the dead constant. |
| luma-official-09 | inconsistent | CONFIRMED (read both kernels) | `gpu-spatial-analysis/cell-indexing/gpu-cell-geometry.ts:60-64`, `:284-292` | An invalid cell gets NaN in both centre components. There is no validity output. | `GPUDGGSCellProjection` writes 0 and a separate `validity` flag (`gpu-dggs-cell-projection.js:158-162`). | Cell key 0 with family `'h3'`: official centre `(0, 0)` and validity 0; local centre `(NaN, NaN)`. | Add `output.validity`, as contract section 7 asks. Keep NaN. |
| luma-official-10 | rule | SUSPECTED (functions listed, arithmetic not compared) | `gpu-spatial-analysis/cell-indexing/h3-index-extended-wgsl.ts:113-216` | A private double-float library (`cellIndexH3ExtTwoSum`, `...QuickTwoSum`, `...Split`, `...TwoProduct`, add, multiply, divide, Taylor sine and cosine). | `fp64arithmetic` publishes `twoSum`, `quickTwoSum`, `split`, `twoProd`, `sum_fp64`, `mul_fp64`, `div_fp64`; `fp64` publishes `fp64_sincos`. The official code picks compiler-safety defines per device with a helper that is not public. | Not tested. | Check on the test GPU that the private two-sum survives compiler contraction as the official module's guards do. Propose upstream a public way to get the module with the right defines. |
| luma-official-11 | rule | CONFIRMED (read) | `scripts/check-upstream-first.mjs` (the `PUBLIC_EXPORTS` filter) | The guard treats every `exports` key without `*` as public. | `@luma.gl/graph` lists `./_internal/arrow` as an explicit key. | `import ... from '@luma.gl/graph/_internal/arrow'` would pass the guard. No file does this today. The guard also cannot see casts (finding 03), copied WGSL (finding 02) or parsing of shader text (finding 06). | Reject any key that contains `_internal`. |
| luma-official-12 | untested | SUSPECTED | `gpu-spatial-analysis/spatial-join/gpu-point-in-polygon-join.ts:636-703`; `gpu-spatial-predicate-join.ts:1311` | Two point-in-polygon classifiers in one package: an inline exact-orientation one with a `boundaryRule`, and the official double-single one. | The official one returns `uncertain` when the arithmetic envelope is exceeded. | Not tested: whether a point on a shared edge, or a row the official classifier calls `uncertain`, is joined the same way by both contributors. | Add one spec that runs both on the same boundary points. |
| luma-official-13 | doc-mismatch | CONFIRMED (name checked in every entry) | `dev-docs/upstream-first-foundation.md` section 2, last row | Says `makeGPUQueryExpressionShaderPlan` is public in `@luma.gl/graph/gpu-dataframe`. | It is not exported from any entry in alpha.10. | No local file imports it. | Correct the sentence. |

Notes on individual findings.

**01 and 02 are the two places where official and local give different numbers on the same
input.** Both were found independently by the domain reviewers. They are listed here because the
cause is the subject of this review: the library has an official implementation and a local one of
the same operation side by side, and nothing ties them together.

**02.** The three `GPUGridIndex` copies that query an index built by the official class are safe
today because the copy is identical. Two of them also widen the query by one cell
(`region-statistics-grid.ts:188`, `gpu-inverse-distance-weighting.ts:536`); `encounter-kernels.ts`
relies on the identical text alone (`:142`). The risk is the next luma.gl bump. The foundation document
(`dev-docs/upstream-first-foundation.md` section 4, `GPU_GRID_COORDINATE_WGSL` row) allows writing
this snippet locally, while `AGENTS.md` forbids copied luma.gl source. The two documents disagree.
The float32 replication assumes correctly rounded division. WGSL allows 2.5 ULP, so the exact
values that disagree on a given GPU may differ from the ones listed.

**03.** `getGraphViewGPUVector` (`analysis-core/network-topology.ts:150`) is fine: it uses the
public `GraphBufferHandle.defaultBuffer`.

**04.** A numeric difference between the two k-means on one input was not produced, because the
official seeding was not replicated. The algorithms differ by construction, so they are not
interchangeable.

## 5. Convention differences

Each item is systematic, and none is in the workarounds register.

### 5.1 How a contributor is attached to a graph

- Official domain entries use `addToGraph(graph): void` (`geospatial` 7 files, `gpu-raster` 20,
  `gpu-graph-algorithms` 13, `gpu-vector-search` 3, `gpu-project` 3, `gpu-dggs`, `gpu-timeseries`,
  `lucim`, `gpu-crossfilter`). Official `gpu-core`, `gpu-pointcloud` and `gpu-geometry` use
  `getCommandNodes(graph)`.
- Local: 475 of 485 classes use `getCommandNodes` and 474 declare
  `implements GPUCommandNodeProducer`. Six use `addToGraph` only: `GPURasterNDWI`, `GPURasterNBR`,
  `GPURasterEVI`, `GPURasterSAVI` (`raster-analysis/gpu-raster-spectral-indices.ts:118-205`),
  `GPURasterEmbeddingKMeans`, `GPURasterEmbeddingNearestPixels`.
- So the local raster, terrain and geospatial contributors follow the `gpu-core` convention, not
  the convention of the official entry for their own domain. About 20 files bridge the two with
  `captureGraphCommandNodes(graph, () => new X(...).addToGraph(graph))`.

### 5.2 Shape of props and outputs

- Official: flat props. Outputs are top-level (`output`, `outputValidity`, `count`, `overflow`,
  `statistics`, `segmentCount`). Rasters are `width`, `height`, `input: GPURasterBufferBand`, and
  optional `metadata`.
- Local (contract sections 6, 8, 9): `output: {...}` with a nested `status: {...}`, and
  `grid: GPURasterMetadata` (180 declarations). The local source also has 301 `outputs`
  declarations against 1,211 `output`.
- 12 local contributor files take a raster as raw `values` plus optional `validity` plus
  `noDataValue` instead of the official band: `raster-algebra` (arithmetic, cell statistics,
  conditional, reclassify, weighted overlay), `isolines` (3), `raster-sampling` (2),
  `raster-stretch`, `gpu-focal-statistics.ts`. About 55 files take `GPURasterBand` or
  `GPURasterBufferBand`. The contract and luma.gl agree on the band; these 12 follow neither.

### 5.3 Validity and no-data names

- Output flag: official rasters use `outputValidity` (37 declarations). `gpu-dggs` and others use
  `validity`. Local has 28 `outputValidity` declarations (terrain, relief, spectral indices) and
  132 `validity` declarations. Contract section 7 says `validity`. The contract and the official
  raster entry disagree, and the local raster code is split between them.
- Policy when a neighbour is missing: official `GPURasterNoDataPolicy` is
  `'propagate' | 'ignore-renormalize'`; official `GPURasterMorphologyNoDataPolicy` is
  `'propagate' | 'ignore'`. Local uses `'propagate' | 'ignore'` (rank filter, local operations,
  weighted overlay, focal pairs) and a third pair `'strict' | 'renormalize'`
  (`raster-analysis/raster-sampling/raster-sampling-parameters.ts:20`,
  `gpu-raster-reprojection.ts:40`) for the same two meanings.
- Border: local mostly uses the official name `borderMode` and its values. Exceptions:
  `edgeMode: 'nodata' | 'extrapolate'` (`gpu-terrain/topographic-position/gpu-terrain-ruggedness.ts:33`)
  and `edgeMode: 'interp' | 'truncate'` (`raster-analysis/remote-sensing/gpu-raster-temporal.ts:43`).
- Value of an invalid output: official rasters and haversine write a quiet NaN and a zero flag;
  official DGGS writes 0 and a zero flag. Local writes NaN (finding 09).

### 5.4 Count and overflow

- Official is not uniform. `GPUHashJoin.count` and `GPUBVHQuery.count` hold the required count
  before truncation. `GPUPointSpatialQuery.count` and `GPURasterContours.segmentCount` hold
  `min(required, capacity)`.
- Local `GPUCompactOutput.count` is always clamped, with `status.requiredCount` unclamped. A future
  local wrapper around `GPUHashJoin`, `GPUBVHQuery` or `GPUGridIndexQuery` must clamp. None exists
  today.
- Official status is often one packed view (`status: [executedIterations, changedLabels,
  converged]`, four-word `statistics`). Local status is one scalar view per named field.

### 5.5 Names of the same prop

- Iteration cap: local `maximumIterations` in 62 files, `maxIterations` in 23 files (all of
  hydrology, isolines, service areas, map colouring, line simplification), `iterations` in 8.
  Official is also mixed: `iterations` (graph algorithms), `maxIterations` (vector search, shortest
  path, conjugate gradient), `maximumIterations` (raster components). `AGENTS.md` says never
  abbreviate, which matches the official raster entry.
- Default node ID prefix: official `'gpu-<class name>'` in 146 of 148 classes. Local drops `gpu-`
  in 492 of 504 defaults (`'kmeans'`, `'raster-rank-filter'`). The 12 local exceptions are the ten
  point-cloud classes, `GPULocalFrameProjection` and `GPUSnapshotArena`; the four spectral indices
  also pass `'gpu-raster-*'` defaults.
- Option values: official uses kebab-case (`'normalized-difference'`). Some local selectors are
  camelCase (`'normalizedDifference'` in `raster-algebra/local-operations-parameters.ts:126`,
  `'standardDeviation'`, `'zScore'`).
- Earth radius and unit: official haversine defaults to 6371 km; local distances use
  6,371,008.8 m through the spatial context. Web Mercator latitude limit: local literal `85.051129`
  (`gpu-spatial-analysis/cell-aggregation/cell-keys.ts:201`, `cell-cover/cell-cover-kernels.ts:309`)
  against official `WEB_MERCATOR_MAX_LATITUDE = 85.05112877980659`. Both round to the same float32
  (85.05113220214844), so there is no numeric difference, only a literal where a public constant
  exists.

### 5.6 Inputs the official classes accept and local ones do not

- Chunked `GraphVectorView` inputs: most official primitives add one node per chunk. Local
  contributors require one packed view (contract section 10); `GraphVectorView` is imported in 27
  local files.
- Float64 positions: official `geospatial` classes accept raw binary64 rows (`GPUFloat64Positions`,
  `uint32x4`). Local position props are `GraphDataView<'float32x2'>` (423 `positions`
  declarations); one local file uses `GPUGeospatialPositions`.

### 5.7 How kernels are written

- Node construction: local uses `createWGSLKernelNode` for 2,502 kernels in 572 files, plus 220
  `createFillNode` or `createPublishNode` calls and 32 `getPointCloudKernelNodes` calls. Only three
  files call `createGPUComputeCommandNode` or `new Kernel` directly (two core helpers and
  `computation-record/gpu-transfer-mask.ts`). Official classes each write the whole shader, with
  explicit `@group(0) @binding(n)` or `@binding(auto)`, and add it with `graph.addComputePass`.
  luma.gl publishes no kernel-node helper, so this is a local layer by necessity. It is uniform.
- Bounded dispatch: applied inside the helper. Eight local files call `getBoundedDispatchLayout`
  themselves.
- View offsets: both bake the element offset into the shader text. Official names it
  `<NAME>_OFFSET` (a `const`); local names it `<name>Offset`.
- Validation: no bare `assert` remains (0 hits). Named-limit helpers are local; luma.gl has none.
- WGSL style: constants are CAPITAL_CASE in both. Functions are camelCase in all local kernels
  (2,370 definitions, 0 snake_case). Official shader modules use module-prefixed snake_case
  (`dggs_u64_add`); official kernels use camelCase. Vector types: local is split (1,632 long form
  `vec2<f32>`, 1,630 short form `vec2f`). Official is split by entry (`gpu-raster` and
  `gpu-graph-algorithms` long form only, `gpu-project` short form only). There is no single
  official style to converge on here.
- 64-bit words: finding 05.
- Shader modules: only `dggs` is imported from `@luma.gl/shaders`. Private WGSL covers what
  `fp64arithmetic` and `fp64` publish (finding 10). The official `random` module is correctly not
  used (`analysis-core/kernels/philox-random.ts:8`).
- Eight comments still call luma.gl's graph package "gpgpu" (for example
  `analysis-core/kernels/gpu-contributor-utils.ts:201`, `spatial-weights/gpu-contiguity-weights.ts:248`).

## 6. Workarounds register check

`node scripts/check-upstream-first.mjs` prints `upstream-first: no violations` (exit 0). The wider
scan found no re-export, alias or deep import in `modules/analysis/src`. Outside it, two classes
extend the official graph: `LensCommandGraph` (`modules/lenses/src/computation-lenses/lens-view-snapshot.ts:39`)
and `MeteredCommandGraph` (`modules/kepler/src/session/metered-command-graph.ts:11`). Both override
only public methods.

Every "what luma.gl needs" symbol in the register is still absent from alpha.10 (list in section 2),
so no workaround row can be retired on this release. The "primitive signature changes" row was
checked against the declarations and is accurate (`GPUSortAlgorithm`, `GPUScan` `mode`,
`GPUCommandGraph` constructor props, no `targetFrame`).

Rows that are stale:

| Row | What is stale |
| --- | --- |
| Section 2, "Computation-record annotations" | The working tree (uncommitted, another session) now has a per-graph record: `recordWGSLKernelNodes`, `isRecordingWGSLKernelNodes`, `getWGSLKernelNodeRecord` (`analysis-core/kernels/wgsl-kernel-nodes.ts:566-591`, exported at `analysis-core/index.ts:10-14`), and `LensCommandGraph` turns it on. The row still says lens records carry "structure and timings only, no source text, bindings". `dev-docs/upstream-first-recording.md` is also modified in the working tree. |
| Section 1, "Command-graph-transient CSR and column views" | Says "Nothing to restore here". Two local stand-ins exist: `analysis-core/kernels/masked-weak-components.ts:34` and the cast in `analysis-core/network-topology.ts:193-245`. |
| Section 2, "Kepler-compute example" | Says 271 of 367 kepler spec files are excluded. `test/kepler-compute-example-specs.json` lists 131 under `excludedFromRuns`, and `modules/kepler/test` holds 368 spec files. |
| Section 3, "Centrality compile cost" | Cites `gpu-network-centrality.ts:70` and `:418-423` for the guard. It is now at `:402-410`. |
| Section 2, `waitForSubmittedWork` | Cites `gpu-table-inspector.ts:539`; the call is at `:540`. |
| `upstream-first-foundation.md` section 2 | Finding 13. |

Rows that are accurate as written (file and line checked): strongly connected components (limits
still 1,024 and 65,536, `gpu-network-structure.ts:30-32`), raster validation (`raster-band-utils.ts:38`, `:252`,
`:266`), vertical datum (`gpu-raster-grid.ts:69`, `:79`, `:99`), arrow normalisation
(`analysis-core/index.ts:6`), out-of-memory detection (`dataset-registry.ts:852-854`), bind-group
reuse, dispatch gate, bounded dispatch overrides, segmented sum, sparse Laplacian solve, geometry
vector helper, point-cloud kernel nodes. The "bounded dispatch above 4.2M rows" row is marked
obsolete; its spec exists and was not re-run.

Rows that are missing:

1. Local copies of the official cell mapping (finding 02). Needs: a public cell-coordinate snippet.
2. `createPointCloudGridIndex` (`pointcloud-analysis/pointcloud-grid-index.ts:85`). The six upstream
   gaps are written in the code comment at `:64-83` and nowhere else.
3. `createSortedSegmentOffsetsNode` in place of `GPUGallopingSearch`
   (`analysis-core/kernels/sorted-segment-offsets.ts:21-23`).
4. `createMaskedWeakComponentNodes` and the cast topology (finding 03). Needs: graph algorithms on
   caller-built CSR views.
5. `parseDggsTable` (finding 06). Needs: the H3 face tables as data.
6. Private double-float WGSL and 64-bit word WGSL (findings 05, 10). Needs: a small 64-bit word
   module, and public selection of the `fp64arithmetic` defines.
7. The local `GPUKMeans`, `GPUKeyJoin`, `GPUGroupStatistics`, `GPUCellAggregation`, `GPUIsolines`,
   `GPURasterArithmetic`, `GPURasterStretch`, `GPUFocalStatistics`, and the min/max windows of the
   point-cloud ground filters. Each stands beside an official class; the register is where the
   "what luma.gl would need for us to delete this" belongs.
8. The network contributors that explain in their class documentation why the official algorithm
   is not used (`GPUNetworkReachability`, `GPUNetworkNeighborhood`, `GPUNetworkStatistics`,
   `GPUNetworkLouvain`). The reasons are good; they are not collected anywhere.
9. `captureGraphCommandNodes` as the bridge for official `addToGraph` classes (about 20 files).

Also: `AGENTS.md` ("never ... copied luma.gl source") and `upstream-first-foundation.md` section 4
(write the cell-coordinate snippet in the consuming kernel) give opposite instructions.

## 7. Recommended order of work

Fix first (official and local disagree on numbers):

1. `GPUPointDensity` weighted sums: one cell mapping for counts and sums
   (`point-density-kernels.ts:279`). Finding 02.
2. Saddle rule of `GPUIsolines` and `GPUIsobands`: take the official bilinear determinant.
   Finding 01.

Then remove the reliance on official internals (cheap, no numeric change):

3. Replace the cast topology in `gpu-terrain-spike-repair.ts:496` with the local component helper.
   Finding 03.
4. Guard `parseDggsTable` with a row-count check. Finding 06.
5. Add a test that pins each `getCoordinate` copy to the official kernel text. Finding 02.
6. Close the `_internal` gap in `scripts/check-upstream-first.mjs`. Finding 11.

Then remove the name collisions and align names (API change, no numeric change):

7. Rename the local `GPUKMeans` and fix its documentation path. Finding 04.
8. Delete or rename the local `validatePackedView`. Finding 07.
9. Flip `UINT64_WGSL` to `(high, low)` while it still has no caller. Finding 05.
10. One no-data policy vocabulary (`'propagate' | 'ignore'`), one iteration-cap name
    (`maximumIterations`), and `output.validity` on `GPUCellGeometry`. Section 5.3, 5.5, finding 09.
11. Decide `outputValidity` against `validity` and `addToGraph` against `getCommandNodes` once, and
    write the decision in the contract. Today the contract and the official raster entry disagree,
    and the code follows both. Sections 5.1, 5.3.

Replace local code with the official class (possible today, needs a parity spec first):

12. `GPUCellGeometry` centres for H3 and A5: `GPUDGGSCellProjection`.
13. `GPUGeodesicPairs` sphere distance: `GPUHaversineDistance` with `radius` from the spatial
    context. Keep the local kernel for bearings, midpoints, Vincenty and rhumb.
14. Min/max windows of `GPUPointCloudSMRF` and `GPUPointCloudGround`: `GPURasterErosion` and
    `GPURasterDilation`, repeated for radii above 8.
15. `GPURasterArithmetic` for the five operations `GPURasterBandMath` has, once the 12 raw-value
    raster contributors take a `GPURasterBufferBand` (section 5.2).

Propose upstream (ordinary luma.gl pull requests; nothing here waits on them):

- A public grid cell-coordinate WGSL snippet, and one formula shared by `GPUGridIndex`,
  `GPUGridBinning` and `GPUGridAggregation`. The `GPUGridIndex` variant places x = -177 in column 2
  of 360 over `[-180, 180]`; the other two place it in column 3.
- Graph algorithms that accept a caller-built CSR as command-graph views, with masks.
- `GPUGridIndex`: a cell-size mode, a half-open domain, stable order inside a cell.
- `GPURasterContours`: several levels per pass and world coordinates.
- `GPUHashJoin` and `GPUGroupAggregation`: 64-bit keys and sparse keys.
- `GPUKMeans`: k-means++ seeding and a tolerance stop.
- A small 64-bit word shader module with multiply and divide, and the H3 face tables as data.
- A single documented meaning of `count` (clamped or required) across `GPUHashJoin`, `GPUBVHQuery`,
  `GPUPointSpatialQuery` and `GPURasterContours`.
- The items already in the register.

## Resolution (2026-11-10)

This stream owned the registers, the guard and the upstream proposal, not the source. Findings whose
file is under `modules/analysis/src` were fixed by the stream that owns the directory; the row says
which, and "per its Resolution" when that stream's Resolution section existed when this was written.
Six streams had not written theirs yet (`interpolation-density`, `point-process-clustering`,
`polygon-topology`, `raster-fields`, `spatial-statistics`, `terrain-hydrology`); for those the row
says what the working tree held, read at the time of writing. Their "Upstream gaps" lists, when
they appear, still have to be merged into section 4 of the register. Paths are relative to the repository root.

Counts over the 13 findings: 8 fixed (11 and 13 by this stream, and the list, pins and register of
02), 2 not reproduced (10, 12), 2 fixed in part with the rest deferred to roadmap 12.4 (04, 05), and
1 deferred whole (07). 08 is counted as fixed for its documentation half; its dead constant is
deferred.

| ID | Outcome | What changed | Spec that pins it | Behaviour change a caller sees |
| --- | --- | --- | --- | --- |
| luma-official-01 | fixed by `raster-fields` (in the working tree; no Resolution section yet) | `modules/analysis/src/raster-analysis/isolines/isolines-utils.ts:80` (`ISOLINES_SADDLE_WGSL`, the bilinear decider), called as `isSaddleJoined` in `gpu-isolines.ts` and `isobands-wgsl.ts`. Register section 4, `GPURasterContours` row. | `modules/analysis/test/gpu-raster/isolines/isolines-saddle-rule.spec.ts` (new, untracked; not run by this stream). | Saddle cells of `GPUIsolines` and `GPUIsobands` now connect as `GPURasterContours` does. |
| luma-official-02 | fixed: the mismatch by `interpolation-density` (in the working tree; no Resolution section yet); the pins by five streams; the list and the register by this stream | `GPUPointDensity` now holds the `GPUGridBinning` form (`modules/analysis/src/gpu-spatial-analysis/point-density/point-density-kernels.ts:286`, character-identical to `gpu-grid-binning.js:193`). No copy was removed: luma.gl exports no cell-coordinate function. New `modules/analysis/test/analysis-core/official-wgsl-copies.node.spec.ts` lists the five `getCoordinate` copies with their pin specs, pins two further copies the review did not list (`safeNormal` in `pointcloud-analysis/pointcloud-kernel-nodes.ts:107`, `encodeOrderedFloat` in `raster-analysis/raster-zonal-statistics/gpu-raster-zonal-statistics.ts`), and scans the source for unlisted copies. Register section 2, two new rows. `dev-docs/upstream-first-foundation.md` section 4 no longer tells the reader to write the snippet locally. Proposal section 1. | The five per-copy pins (`encounter-grid-coordinate-pin`, `region-statistics-grid-coordinate-pin`, `spatial-interpolation/grid-coordinate-pin`, `point-density-grid-coordinate-pin`, `flow-aggregation-grid-coordinate-pin`, all `.node.spec.ts`); `official-wgsl-copies.node.spec.ts` (9 tests; a scratch copy with one entry removed and one pin path broken failed on both, as intended); `gpu-point-density-cell-mapping.spec.ts` for the weighted sums (not run by this stream). | Weighted `GPUPointDensity` sums land in the cell that holds the count. |
| luma-official-03 | fixed by `terrain-surface` (per its Resolution) | The cast and `createNetworkAnalyticsTopology` are gone (`grep` over `modules/analysis/src` finds neither); spike repair uses `createMaskedWeakComponentNodes`. Register: "Retired" entry, and the section 4 row for `GPUGraphAlgorithmsConnectedComponents`. | `modules/analysis/test/gpu-terrain/terrain-decode/spike-repair-component-labels.spec.ts`. | A run that stops at `maximumComponentIterations` may stop at a different iteration. |
| luma-official-04 | documentation fixed by `point-process-clustering` (in the working tree; no Resolution section yet); the rename and the prop names deferred (roadmap 12.4) | The class documentation of the local `GPUKMeans` now names `@luma.gl/graph/gpu-vector-search`. Register section 4, `GPUKMeans` row; proposal section 7. | None. | None. |
| luma-official-05 | `UINT64_WGSL` fixed by `core-numerics` (per its Resolution); the three private `(low, high)` copies deferred (roadmap 12.4, per `dataframe`) | `modules/analysis/src/analysis-core/kernels/uint64-wgsl.ts` is `(high, low)`. Register section 2, "64-bit integer words in WGSL" row; proposal section 11. | `modules/analysis/test/utils/wgsl-words-search.spec.ts`. | Internal only. |
| luma-official-06 | fixed by `cells-joins` (per its Resolution) | `parseDggsFaceTable` throws unless a table has 20 rows of finite numbers. Register section 2, "H3 face tables as data" row; proposal section 11. | `modules/analysis/test/gpu-spatial-analysis/cell-indexing/h3-index-extended.node.spec.ts`. | None with the published shader. |
| luma-official-07 | deferred (roadmap 12.4: rename or removal of an exported symbol; per `core-numerics`) | Register section 4, `validatePackedView` row. | n/a | None. |
| luma-official-08 | documentation fixed by `geometry-measures` (per its Resolution); removing `GPU_SINUSOIDAL_KILOMETERS_PER_DEGREE` deferred (roadmap 12.4). Order-of-work item 13 (delegate the sphere distance) not done: the published float32 kernel subtracts raw longitudes | Register section 4, `GPUHaversineDistance` row; proposal section 3. | None for the documentation. | None. |
| luma-official-09 | fixed by `cells-joins` (per its Resolution) | `output.validity` on `GPUCellGeometry`. Register section 4, `GPUDGGSCellProjection` row. | `modules/analysis/test/gpu-spatial-analysis/cell-indexing/gpu-cell-geometry.spec.ts`. | New optional output. |
| luma-official-10 | not reproduced (per `cells-joins`): the private two-sum survives on the test GPU | No source change. Register section 2, "Double-float WGSL" row; proposal sections 13 and 14 (public selection of the `fp64arithmetic` defines). | `modules/analysis/test/gpu-spatial-analysis/cell-indexing/h3-index-extended.spec.ts`. | None. |
| luma-official-11 | fixed by this stream | `scripts/check-upstream-first.mjs`: new exported `findNonPublicLumaImports` rejects any `@luma.gl/*` specifier with an `_internal` segment before it consults the `exports` map, for every package (not only the four dependencies); the `UPSTREAM_FIRST_SCRATCH_FILE` self-test now applies the import rule as well as the re-export rule. The script prints `upstream-first: no violations` on the tree. Still not seen by the guard: casts (none left), parsing of shader text (guarded, finding 06), copied WGSL (now `official-wgsl-copies.node.spec.ts`). | `modules/analysis/test/analysis-core/upstream-first-guard.node.spec.ts` (2 tests: a fixture outside the tree with public imports passes; one with six bad imports and a re-export gives exactly 7 violations). | `import ... from '@luma.gl/graph/_internal/arrow'` now fails `yarn check-upstream-first`. |
| luma-official-12 | not reproduced (per `cells-joins`): the two classifiers agree | No source change. Register section 4, `GPUPairwisePointInPolygon` row; proposal section 10. | `modules/analysis/test/gpu-spatial-analysis/spatial-join/gpu-point-in-polygon-classifier-agreement.spec.ts`. | None. |
| luma-official-13 | fixed by this stream | `dev-docs/upstream-first-foundation.md` section 2, last row: neither `makeGPUQueryExpressionShaderPlan` nor `addGPUQueryPredicatePass` is exported from any entry of alpha.10 (all 38 `exports` keys loaded in Node and checked). | None (words). | None. |

### Order of work, items 12 to 15 (delegate where possible)

| Item | Outcome | Note |
| --- | --- | --- |
| 12. `GPUCellGeometry` centres through `GPUDGGSCellProjection` | deferred (per `cells-joins`) | The parity spec passes, but keeping NaN for invalid cells and the boundary output would take three passes for identical numbers. Register section 4. |
| 13. `GPUGeodesicPairs` sphere distance through `GPUHaversineDistance` | not done (per `geometry-measures`) | The published kernel loses 1.5e-5 degrees across the antimeridian. Register section 4, proposal section 3. |
| 14. Point-cloud windows through `GPURasterErosion` and `GPURasterDilation` | `GPUPointCloudSMRF` delegated; `GPUPointCloudGround` deferred (per `core-numerics`) | Register "Retired" and section 4; proposal section 5. |
| 15. `GPURasterArithmetic` through `GPURasterBandMath` | deferred (per `raster-imagery`, roadmap 12.4) | The item is conditional on the raw-value raster props (section 5.2), which is roadmap 12.4. Register section 4. |

### Section 5 (convention differences)

Deferred to roadmap 12.4 by the brief: `addToGraph` against `getCommandNodes` (5.1), `outputValidity`
against `validity` and the no-data policy vocabulary (5.3), the iteration-cap name and the node ID
prefix (5.5), the raw-value raster props (5.2). The register now records the bridge
(`captureGraphCommandNodes`, section 2). The meaning of `count` (5.4) is proposal section 12. Nothing
in section 5 was changed in code by this stream.

### Section 6 (register)

`dev-docs/upstream-workarounds.md`, status date 2026-11-10:

- Stale rows corrected, each re-checked against the tree: computation-record annotations (the
  per-graph kernel record exists; lens graphs opt in), command-graph CSR views (the text of the
  `network-paths` stream; status now `workaround`), kepler-compute example (131 of 368 spec files
  excluded from runs, 59 more from the test typecheck), centrality guard (`:290`, `:437-445`),
  `waitForSubmittedWork` (`:540-542`). Also corrected: the strongly-connected-components line
  reference and the two exclusion line references of the geometry-vector row.
- The nine missing rows: seven new rows in section 2 (cell-coordinate copies; other carried WGSL;
  H3 face tables; double-float WGSL; 64-bit words; the `addToGraph` bridge; the projection plan
  origin, which `geometry-measures` reported as unregistered), and the rest in the new section 4.
- New section 4, "Local kernels beside a published class": 29 rows. 22 come from the 27 rows of
  section 3.3 above (of the other five, three are new section 2 rows: cell mapping, `fp64`
  double-float, 64-bit words; two already had a row: the conjugate-gradient solve and the segmented
  sum), and 7 from the "Upstream gaps" lists of the Resolution sections (`GPUHistogram` against
  `GPUColumnProfile`, `GPUPointSpatialQuery` against the lasso, `GPUFiniteDifference2D`, the
  gradient border, voxel centroids, `GPUCompaction`, topology construction).
- Section 3, compensated sums: rewritten with the measurement of this pass.
- "Retired": the SMRF windows, the cast topology, the guard gap.
- The upgrade procedure has a new step for the pin specs.

One "Upstream gap" in another report is not a luma.gl gap and was not registered: `network-paths`
asks for upper-inclusive bands on `GPUIsobands`, which is a local class
(`modules/analysis/src/raster-analysis/isolines/gpu-isobands.ts`). The same list asks for `GPUScan`
on 64-bit integers; `GPUScanUint64` is public in alpha.10, so only the 64-bit reduction was
registered.

Also corrected in files this stream owns: three places in `upstream-first-foundation.md` that told
the reader to write a bare `assert` (against `AGENTS.md`), and the row of
`upstream-first-pending.md` that said the network contributors hold physical vectors.

### Upstream proposal

`dev-docs/upstream-proposal-math-review.md` (draft, not sent): 13 proposed pull requests and a table
of 15 smaller gaps, each with the kind of evidence this repository has. Evidence is a committed spec
for three of them (the point-in-ring predicate, the compensated sums, the local side of the 64-bit
words and H3 tables), a reading or a Node replication for the rest. The `GPUHistogram` domain-view
result is recorded in a spec comment only and is marked as needing confirmation.

### Checks run at the end

- `node scripts/check-upstream-first.mjs`: `upstream-first: no violations`.
- `node scripts/check-contributor-contract.mjs`: `contributor-contract: no new findings`
  (totals: single-invocation-data-loop 46, internal-export 13, multiple-classes 21, root-file 21, all
  in the baseline; bare-assert, missing-producer-declaration, luma-reexport and
  unregistered-contributor 0).
- A TypeScript-AST scan of every file under `modules/analysis/src` (imports, exports, dynamic
  imports, import types, `require`): 12 distinct `@luma.gl` specifiers, all public keys, none with
  `_internal` or `dist`. No `export ... from '@luma.gl/...'`, no `export {X}` of an imported luma.gl
  name, no `const` alias, no subclass, no empty interface extension. Three exported type aliases
  instantiate a luma.gl generic under a domain name and are not re-exports of a luma.gl name:
  `GPUStatusScalar = GraphDataView<'uint32'>` (`analysis-core/kernels/gpu-contributor-types.ts:16`),
  `GPUFlowAggregationActiveGridSize` (`gpu-network/flow-aggregation/gpu-flow-aggregation.ts:84`),
  `NearestPolygonRings` (`gpu-spatial-analysis/spatial-join/nearest-types.ts:59`).

### Not done

- No spec of another stream was run here, and no GPU spec at all: this stream's two specs are Node
  specs that read text.
- The proposal's concrete inputs for sections 4, 5, 7, 8 and 9 follow from the published
  declarations and were not run.
- Line numbers in the register for files that were being edited while it was written
  (`gpu-kmeans.ts`, `gpu-isolines.ts`, `gpu-raster-stretch.ts`, `gpu-focal-statistics.ts`,
  `gpu-raster-zonal-statistics.ts`) were replaced by symbol names. The other line numbers were
  checked by script after the last edit and can still move before the pass ends.

### Review follow-up

Documentation only, after all eighteen streams had written their Resolution sections and most a
"Review follow-up". Where this subsection and the text above disagree, this subsection is current
(in particular: no stream is without a Resolution any more, and their gap lists are merged).

**Upstream gaps merged into the register (section 4, now 31 rows).** All eighteen "Upstream gaps"
lists and every "Review follow-up" were read.

| From | What was added or changed in `dev-docs/upstream-workarounds.md` |
| --- | --- |
| `interpolation-density` | New row `GPUGridBinning`, `GPUGridAggregation` (weighted sums): the grid keys kernel of `GPUPointDensity`; needs a per-row cell key output or a deterministic sum mode. Its other gap (one cell mapping, exported) was already the section 2 row. |
| `point-process-clustering` | `GPUKMeans` row: restarts with an inertia output and empty-cluster relocation added. `GPUGraphAlgorithmsConnectedComponents` row: an edge mask input added, and the hooking kernel of `GPUAZPRegions` named as a second local counterpart. |
| `raster-fields` | `GPURasterContours` row: global edge IDs and a band mode added; and a difference that remains after the convergence, on an exact determinant tie (the published kernel scales the corners first, `gpu-raster-contours.js:441-447`). `GPURasterErosion`/`GPURasterDilation` row: `GPURasterReconstruction` added (mask and fixpoint loop). |
| `terrain-hydrology` | New row `GPURasterRegionMeasurements` against `GPUTerrainCatchmentSummary`. `GPUFiniteDifference2D` row: the gradient of `GPUTerrainFlowField` added. Its third gap (no exact reduction in `gpu-core`) was already the compensated-sums row of section 3. |
| `raster-imagery` | Already merged into the `GPURasterStretch` row before the first final message (equalization and percentile rules now follow the published classes; what each would need). |
| `polygon-topology`, `spatial-statistics`, `regression`, `applied-recipes` | None: no published class computes the same thing. |
| The other streams' follow-ups | No new luma.gl gap. Three carried compiler facts (below). |

**The two wrong gaps in `network-paths`.** A one-line correction was added under that report's
"Upstream gaps": `GPUIsobands` is a local class, and `GPUScanUint64` is already public, so only the
64-bit `GPUReduction` is missing upstream. The register and the proposal never listed either.

**Compiler facts (register section 6, new; proposal section 13, now 13.1 to 13.4).**

- The register has a new section 6, "Shader-compiler behaviour that kernels work around", with four
  rows: removed compensation terms; the regrouping of `(value - shift) - small terms`; the
  function-scope identity table that reads back as zeros at 40 to 96 entries; division that is not
  correctly rounded and a loop sum that is not the IEEE serial sum. Each row has the measurement,
  the local cure and the spec.
- The proposal's section 13 now holds the same four as facts for luma.gl's kernels and its WGSL
  authoring guidance. It says in its first paragraph that the cause inside the compiler was not
  found for any of them, that only one adapter was tried, and that the table fault is most likely a
  browser or driver defect to reduce and report there, not to luma.gl.
- Evidence as recorded: specs for the first three (`addition-rounding-error-wgsl.spec.ts`;
  `math-review-restricted-regimes.spec.ts`, `math-review-error-ml-high-fit.spec.ts` and the 1e7
  case of `gpu-group-statistics.spec.ts`; `function-scope-table-readback.spec.ts` and
  `gpu-network-louvain-visit-order.spec.ts`). The division fact (`6.0 / 3.0 = 1.9999999`) is an
  observation of the `point-process-clustering` stream: no committed spec asserts the inexact
  quotient; one spec comment records it
  (`gpu-region-partition-evaluation-nonfinite.spec.ts:89`).
- A text search of the published kernels for an identity write `table[i] = i` found three, all into
  storage bindings; no published kernel is known to have the failing shape. Nothing was run.

**Convergences recorded (register, end of section 4).** `GPUCompositeScore` on the shared
`symmetric-eigen-wgsl.ts`; the drive-time catchment recipe on published `GPUSort` plus the shared
`getBinarySearchWGSL`; `GPUContiguityWeights` on `getCoverageSnapKeyWGSL`. None is a luma.gl
workaround; they are noted so the next match table does not list them as private copies.

**`GPUHistogram` domain view (proposal section 6).** The section now describes the committed test
that would replace the spec comment: a NumPy generator and fixture (`/usr/bin/python3`, 3.9.6, has
NumPy 2.0.2, SciPy 1.13.1, scikit-image 0.24.0, OpenCV 4.12.0, pyproj 3.6.1, Shapely 2.0.6; all
imported here to check), and a headless spec that runs the published class with the domain as a
literal and as a view against the fixture. `numpy.histogram` was run by hand for the four reported
values and gives bins 8, 8, 4 and 12. The test itself was not written (documentation-only
follow-up, and a full test run was in progress). Closing pass (2026-11-11): the spec is written
and finds a second defect on bin edges; see "Closing pass" at the end.

**Proposal, other changes.** Section 2 (edge mask), section 7 (restarts with inertia), section 8
(the saddle-tie scaling), and five rows added to the table of section 14 (20 rows now).

**Paths and symbols re-checked.** Every `file:line` in the register and the proposal was printed by
script against the tree and read; every range, caller count and spec path was re-grepped. Three had
moved and were corrected: `raster-band-utils.ts` (`validateRasterBand:297`,
`validateRasterParameterExtent:283`), `gpu-network-centrality.ts` (`:295`, `:442-450`),
`gpu-network-louvain.ts` (`:111`). The kepler counts (131 of 368, 59), the caller counts (20, 7, 5,
10) and the seven listed WGSL copies are unchanged.

**Specs.** `official-wgsl-copies.node.spec.ts` and `upstream-first-guard.node.spec.ts`: 2 files, 11
tests, all pass (the guard spec runs `scripts/check-upstream-first.mjs` on the tree: no
violations). No other spec was run.

**Could not verify.**

- The exact-tie difference between `GPURasterContours` and the local saddle decider is the
  `raster-fields` stream's reading plus the published kernel text; the published class was not run
  on a tie.
- "Not compared" rows of section 4 are still not compared: no parity spec was written in this
  follow-up.
- The published float sums called order-dependent (`GPUGridAggregation`,
  `GPURasterRegionMeasurements`, `GPUGraphAlgorithmsModularity`) are readings of the declarations
  and kernels; none was run twice to show a difference.
- The compiler facts are from one adapter. Which of them are permitted behaviour and which are
  defects of the browser or the driver was not established beyond what WGSL's rules say.

### Closing pass (2026-11-11)

Stream "raster" of roadmap 12.6, the `GPUHistogram` item.

| Item | Outcome | Files changed | Spec that pins it | Evidence |
| --- | --- | --- | --- | --- |
| Published `GPUHistogram` with a literal domain and with a domain view, on values exactly on bin edges | defect in published luma.gl confirmed and pinned; nothing to fix here (no local source changed) | new `modules/analysis/test/gpu-core/gpu-histogram-domain-view.spec.ts`; `dev-docs/upstream-proposal-math-review.md` section 6 and its row in the summary table | the new spec, 12 of 12 on the headless GPU, three runs | `numpy.histogram` (NumPy 2.5.4, pinned CPython 3.14 set) on the same float32 values; the expected counts are also evident by hand (one per bin, two in the last) |

- The spec runs `GPUHistogram` from `@luma.gl/graph/gpu-core` (10.0.0-alpha.12) twice per case on
  one device: the domain as a literal pair, then the same two numbers in a two-row float32 view.
  Each case is `n` bins of width `w` and one value on every one of the `n + 1` edges; all are exact
  in float32.
- Neither form is right for every input, and they differ from each other. 49 bins of width 1: the
  literal form is right, the view form counts edges 7, 14, 27, 28, 35, 41 and 47 one bin low. 3, 6
  and 12 bins of width 41 and 7 bins of width 83: both forms count some edges one bin low, and for
  6 and 12 bins not the same ones (the full lists are in the spec and in proposal section 6). Both
  are right for 7, 255 and 1000 bins of width 1, 12 bins of width 0.25 and 30 bins of width 8.
- The wide-domain result of the first pass reproduces: 16 bins over `[-1e38, 1e38]`, the view form
  puts all four values in bin 0, the literal form gives bins 8, 8, 4 and 12. The comment in
  `gpu-column-profile.spec.ts:573-575` is therefore right as written and was left alone.
- The defect cases assert the observed counts with a comment that names proposal section 6, so the
  suite passes and a luma.gl or browser upgrade that fixes a case fails its assertion.
- Not verified: a second adapter; the generated shader (the two evaluation orders named in the
  spec comment are inferred from the counts, and do not fit two other cases tried); domains that
  bracket 2^126; whether the local product test of `GPUColumnProfile` is right on these same
  inputs (its own spec covers 30 bins only).
- After the independent review (frc-fields-06, [fix-review-closing-fields.md](./fix-review-closing-fields.md)):
  proposal section 6 no longer carries the superseded description of the test, which said the
  literal form should pass every case. The subnormal explanation of the wide-domain result now
  says it also needs flush-to-zero and that both parts are inferred. The case that the formula is
  the cause now names the edges an IEEE evaluation gets wrong (27 of 49 bins; 251, 253, 502, 506
  and 511 of 1000 bins, which the test GPU gets right). The "would retire" sentence says the
  `GPUColumnProfile` kernel is documented to use the product test and was not run by this spec.
  No spec changed.
