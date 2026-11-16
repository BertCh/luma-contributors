# Upstream proposals

Nothing has been filed upstream: no issue and no pull request. Measurements cover one adapter: headless Chromium WebGPU on Apple Metal. Repository pins are `10.0.0-alpha.14` (bumped 2026-11-14 for the Kepler port, with deck.gl 10.0.0-alpha.1 and math.gl 5.0.0-beta.8). Between alpha.13 and alpha.14 luma.gl changed only colour conversion, the expression operation and SQL files, so proposals 1 to 14 are unchanged; proposal 15 comes from the Kepler port and was checked against the alpha.14 `dist`.

## 1. Summary for the maintainer

Basis: about 570 kernel files (2,523 compute-node constructions) built only on the public entry points of @luma.gl/graph, core and shaders. Checked against 10.0.0-alpha.14.

For most items below a first implementation with tests already exists: the luma.gl fork these contributors came from (commit `e0ed33791`, 72 commits ahead of upstream at `704183953`) changed about 360 luma.gl-owned files in graph, core and engine and added 304 exported names, none of which is in the published `dist`. Section 5 maps the items to that code.

**Top five**

1. Ship a kernel-node authoring factory. Local createWGSLKernelNode is called 2,523 times and new Kernel 12 times. One binding list produces the WGSL declarations, offsets, resource uses, bounded dispatch and limit checks that name the kernel and binding.
2. Let published graph algorithms take a CSR that already lives in the caller's GPUCommandGraph. Each has addToGraph, but its only input is a GPUGraphAlgorithmsTopology rebuilt from physical GPUVectors, and GPUGraph programs compile to a private graph, so none is imported here and degree, components, modularity, BFS, SSSP and a Jacobi PCG are re-implemented. Accept CSR as GraphDataViews plus node and edge masks.
3. 64-bit and multi-key keys throughout: GPUSort (keyBits at most 32), GPUHashJoin, GPUGroupAggregation (dense uint32 keys) and GPURunLengthEncode cannot take H3, Quadbin or Int64 ids.
4. A deterministic sum mode. GPUGroupAggregation, GPUGridAggregation, GPURasterRegionMeasurements and modularity use float compare-exchange sums that depend on thread order. The local fixed-point accumulator can be contributed.
5. A GPU-gated loop that is not unrolled. GPULoopOperation unrolls maximumIterations copies, so centrality is capped at 32,768 command nodes.

**Correctness (small, each has a reproduction)**

- Grid cell rule: GPUGridIndex and GPUGridBinning/GPUGridAggregation use different getCoordinate formulas; x = -177 over [-180, 180] with 360 columns gives column 2 and column 3. One formula, exported as WGSL.
- GPUGridIndexQuery: a bounds query can miss a point within a few ULP of a cell edge, because build and query use different arithmetic (CPU float32 model, not a GPU run).
- GPUHistogram: values exactly on a bin edge land one bin low for some bin counts (3 bins over [0, 123], values 0, 41, 82, 123 give 2, 1, 1), and a GPU-resident domain near the float32 maximum puts everything in bin 0. Decide the bin by products and comparisons.
- count semantics: unclamped in GPUHashJoin and GPUBVHQuery, clamped in GPUPointSpatialQuery and GPURasterContours. Clamped count plus requiredCount and overflow everywhere.
- getResourceCounts reads 'Texture2Ds Active', which luma.gl 10 never increments, so the texture count is always 0.
- GPUReadbackRing: the constructor does not free earlier slots if a later createBuffer throws.
- GPUTranspose: the invocation count overflows for degenerate shapes such as 268,435,456 x 1.
- GPUHaversineDistance: the raw longitude subtraction loses about 1.7 m across the antimeridian. The default radius of 6371 gives kilometres; the declaration says so, the class name and output prop do not.
- Self-loops: Degree counts a self-loop once and Modularity counts it twice.
- Metal fast math deletes float compensation terms (Kahan, Neumaier, two-sum), so a compensated sum is the plain float32 sum bit for bit; any future compensated reduction needs an integer path, and this deserves a WGSL authoring page. A reduced reproduction of an Apple back-end miscompile of small integer tables exists.

**Export what is already in dist**

- GPUSegmentedReduction, GPUSegmentedScan, GPUScatter, GPUJacobiPCG.
- getGPUComputeDispatchWorkgroups and gateGPUCommandNodes; without them a gated loop cannot gate a published primitive.
- getPlatformFP64Arithmetic and getClassicFP64Defines; the assembler picks the safe integer variant only on Apple.
- getSpatialOperationNodes, validateRasterBand, and the GPUScalar type (public lowerers return it).
- The four H3 tables as data (they are parsed out of the WGSL text here), and a standalone uint64 shader module with multiply and divide (pulling in dggs costs about 70 KB per kernel).

**Ergonomics**

- One attachment convention: core primitives return nodes from getCommandNodes; domain classes only have addToGraph(): void. The local bridge temporarily replaces three graph methods (21 files).
- GPU-resident parameters, consistently: GPUGridBinning takes GPU bounds while GPUGridIndex takes literals only; raster neighbourhood, morphology and blur are capped at radius 8, which keeps five terrain and point-cloud kernels local.
- Shader modules for compute: the assembler returns a module's bindings, but nothing wires them into a command-graph node's resource declarations, so the local factory accepts only binding-free modules and kernels are composed by string concatenation with hand-kept ordering rules.
- Errors: uncaptured WebGPU errors are re-wrapped as plain Error, so out-of-memory is undetectable, and error scopes do nothing unless debug is set.
- Limits: feature level 'max' couples adapter limits to all features; the limits alone are wanted.
- Dataframe validity: new GPUTable({vectors}) sets nullable: false and the expression shader then ignores validity; .nullable is patched after construction in 12 files.
- Dataframe lifecycle: compiled queries have no compileAsync, no statistics or preflight, and no mask-only filter.
- Arrow: the analytics adapter refuses Bool, 8- and 16-bit integers, Float64, plain Utf8 and exact Int64 time.
- GPUCrossfilter needs external-mask dimensions and invalidate; all dimensions feed one GPUMask, which probably exceeds 8 storage bindings at 8 dimensions (inferred, not run).
- Test kit: GPUCommandGraph rejects the null device, so 467 node specs go through a shim that overwrites its type and limits. A recording device with WebGPU type and real limits would replace it.
- Packaging: npm i @luma.gl/graph installs alpha.5 (the latest tag) next to core 9.4.2; gpu-core has 184 value exports and no authoring guide; a luma.gl 9 to 10 entry-point crosswalk would replace the 1,875-line deck.gl 9.4 patch.

**Can be contributed from here (all with GPU specs)**

- The kernel-node factory and named-limit validators.
- Fixed-point accumulator and sorted-segment sums (57 importing files).
- Philox 4x32-10 with a CPU mirror (47 files); the published random is fract(sin(dot)).
- Exact orientation predicate in WGSL (36 files), which would resolve uncertain in GPUPairwisePointInPolygon.
- Stable multi-key sort, sorted-segment offsets and load-balanced CSR traversal.
- Dense Cholesky/LU and Jacobi eigen WGSL.
- k-means++ seeding and a tolerance stop for GPUKMeans.
- Tests: the histogram edge spec and a 33.5M-row correctness spec for scan, sort, compaction and reduction (they pass).

## 2. Details and reproductions

Proposal numbers 1–14 below identify the math proposals; 15 collects the Kepler port items (luma.gl and deck.gl). Local source paths are under `modules/analysis/src`; published graph paths are under `node_modules/@luma.gl/graph/dist`. `spec` means committed GPU or Node evidence; `replication` means CPU arithmetic from published source, not a GPU run; `reading` means source/declarations only; `measured, not committed` means a reported test-GPU run without a committed reproducing spec. [upstream-workarounds.md](./upstream-workarounds.md) owns the retirement register. Evidence remains limited to the stated adapter and inputs.

### 1. One grid cell-coordinate formula, public as WGSL

Problem: `GPUGridBinning` (`gpu-core/gpu-grid-binning.js`, `getCoordinate`) and `GPUGridAggregation` (`gpu-core/gpu-grid-aggregation.js`, `getCoordinate`) use `min(u32((value - minimum) / (maximum - minimum) * f32(size)), size - 1u)`. `GPUGridIndex` (`gpu-core/gpu-grid-index-internals.js`, `getCoordinate`), query, point-spatial-query and spatial force layout rescale zero-straddling bounds by `max(abs(minimum), abs(maximum))` first. For `[-180, 180]`, 360 columns, x = -177, float32 gives `3 / 360 * 360 = 3.0000002` (column 3) versus `(-0.98333335 + 1) / 2 * 360 = 2.9999971` (column 2). Differences occur at 45 of 360 integer degrees and 967 of 12,407 near-edge values over seven bounds. Correctly rounded CPU division is a model; WGSL permits 2.5 ULP.

Change: one rule, exported as `GPU_GRID_COORDINATE_WGSL` from `@luma.gl/graph/gpu-core`, defining `getCoordinate(value, minimum, maximum, size) -> u32`; document that `maximum` maps to the last cell. Retires five registered `getCoordinate` copies: `gpu-spatial-analysis/trajectory-encounters/encounter-kernels.ts`, `gpu-spatial-analysis/region-statistics/region-statistics-grid.ts`, `gpu-spatial-analysis/spatial-interpolation/gpu-inverse-distance-weighting.ts`, `gpu-spatial-analysis/point-density/point-density-kernels.ts`, `gpu-network/flow-aggregation/flow-aggregation-kernels.ts`, and their pins.

Evidence: reading/replication; pins `modules/analysis/test/gpu-spatial-analysis/point-density/point-density-grid-coordinate-pin.node.spec.ts`, `modules/analysis/test/gpu-spatial-analysis/trajectory-encounters/encounter-grid-coordinate-pin.node.spec.ts`, `modules/analysis/test/gpu-spatial-analysis/region-statistics/region-statistics-grid-coordinate-pin.node.spec.ts`, `modules/analysis/test/gpu-spatial-analysis/spatial-interpolation/grid-coordinate-pin.node.spec.ts`, `modules/analysis/test/gpu-network/flow-aggregation/flow-aggregation-grid-coordinate-pin.node.spec.ts`. Consequence spec: `modules/analysis/test/gpu-spatial-analysis/point-density/gpu-point-density-cell-mapping.spec.ts` covers counts and weights landing in different cells. No spec compares the two published classes on the same point.

### 2. Graph algorithms on caller CSR with masks; one self-loop rule

Problem: `GPUGraphAlgorithmsTopology` (`gpu-graph-algorithms/gpu-graph-topology.d.ts`) rebuilds CSR from physical `GPUVector<'uint32'>` edge columns into separate physical outputs. A caller's `GraphDataView` CSR or per-frame closure mask cannot be supplied. For a masked road network, subtracting offsets counts closed slots too. Degree documents one contribution per undirected self-loop (`gpu-graph-degree.d.ts`); Modularity documents two (`gpu-graph-modularity.d.ts`). One vertex with a self-loop and an edge to a second has degree 2 but modularity volume 3.

Change: accept `GPUGraphAlgorithmsTopologyView` beside `GPUGraphAlgorithmsTopology`: CSR `GraphDataView`s (`offsets`, `neighbors`, optional `edgeIds`, `edgeWeights`), `nodeMask` and `slotMask` or a documented `neighbor >= nodeCount` sentinel; outputs as `GraphDataView`s on the caller's graph. State one self-loop rule, or expose `selfLoops: 'once' | 'twice'`. Use deterministic modularity accumulation (integer atomics unweighted; fixed-order tree weighted). This also supports REDCAP/AZP cut edges.

Retires `createMaskedWeakComponentNodes` in `analysis-core/kernels/masked-weak-components.ts` (five callers), degree/modularity in `gpu-network/network-statistics/gpu-network-statistics.ts` (`GPUNetworkStatistics`), degree in `gpu-network/network-analysis/gpu-network-analytics-columns.ts` (`GPUNetworkAnalyticsColumns`), and potentially hop traversal in `gpu-network/network-analysis/gpu-network-neighborhood.ts` (`GPUNetworkNeighborhood`) and CSR construction in `gpu-network/network-noding/gpu-network-noding.ts` (`GPUNetworkNoding`). Enables PageRank, core number and label propagation on network views.

Evidence: reading; local-rule specs `modules/analysis/test/gpu-network/network-analysis/gpu-network-analytics-columns.spec.ts` and `modules/analysis/test/gpu-terrain/terrain-decode/spike-repair-component-labels.spec.ts`. No same-input comparison with a published algorithm.

### 3. `GPUHaversineDistance`: units and seam-safe subtraction

Problem: default radius 6371 means kilometres on a rounded radius (`geospatial/gpu-haversine-distance.js`, `GPUHaversineDistance`); the declaration states the unit, but the output prop does not. New York (-73.98, 40.75) to Paris (2.35, 48.85): 5833.150 by default versus 5,833,158.6 on the IUGG mean radius in metres; 8.1 m is radius error, 1000 is the unit factor. Raw `rightDegrees - leftDegrees` for (179.94722, 0) to (-179.99805, 0), 6.1 km apart, loses 1.53e-5 degrees or 1.7 m at the equator.

Change: retain radius 6371, document kilometres at the output and export `GPU_HAVERSINE_DEFAULT_RADIUS_KILOMETERS = 6371`; compute small seam-side distances before adding, guarded against reassociation. Retires the sphere path of `GPUGeodesicPairs` in `gpu-spatial-analysis/geometry-measures/gpu-geodesic-pairs.ts`; the model is `geodesicLongitudeDelta` in `gpu-spatial-analysis/geometry-measures/geodesic-wgsl.ts`. Bearings, midpoints, Vincenty and rhumb lines remain.

Evidence: float64 replication agrees within 3e-16 on six pairs; source reading. Local seam specs `modules/analysis/test/gpu-spatial-analysis/geometry-measures/gpu-geodesic.spec.ts` (1.70 m error) and `modules/analysis/test/gpu-spatial-analysis/geometry-measures/parity-wp2-geodesic-seam.parity.spec.ts`; published seam case not run.

### 4. `GPUFiniteDifference2D`: row spacing and no-data

Problem: constant `spacing` pair and no validity ports (`gpu-core/gpu-finite-difference-2d.d.ts`). One arc-second cells are 30.9 m wide at the equator and 15.4 m at latitude 60: a factor of two. NaN/sentinel samples yield unflagged invalid derivatives. Near 8,000 m, float32 steps are about 5e-4 m; `f(x+h) - 2 f(x) + f(x-h)` cancels terms near 16,000.

Change: `spacing` as a numeric pair, a pair with per-row `GraphDataView<'float32'>` east-west spacing, or `GPURasterMetadata` for derived row widths; optional `validity`/`outputValidity` with `'propagate'`; centre-relative zero-sum stencils `(f(x+h) - f(x)) - (f(x) - f(x-h))`, also useful in `GPURasterNeighborhood`. Retires stencils in `gpu-terrain/terrain-curvature/gpu-terrain-curvature.ts` (`GPUTerrainCurvature`), whose borders follow published one-sided stencils. Evidence: reading only; no parity spec.

### 5. `GPURasterMorphology`: radius and valid count

Problem: radius is an integer 0–8 (`gpu-raster/gpu-raster-morphology.js`, `GPURasterMorphology`); invalid centres stay invalid under `'ignore'`, so composition fails across holes. In a 17-cell row, cell 0 = 1, cells 1–15 invalid, cell 16 = 9: radius-16 erosion gives 1 at cell 16, but two radius-8 passes give 9 because cell 8 stays invalid. Valid-sample counts are absent.

Change: larger radii (e.g. van Herk/Gil-Werman for squares), or internal composition that propagates across invalid centres; optional `output.validCount` uint32 or completeness flag. Retires square windows in `pointcloud-analysis/gpu-pointcloud-ground.ts` (`GPUPointCloudGround`, radius up to 32, `GPU_POINT_CLOUD_GROUND_STATUS_INCOMPLETE_WINDOW`); enables delegation of quantiles 0/1 in `raster-analysis/gpu-raster-rank-filter.ts` (`GPURasterRankFilter`). Evidence: reading; the 17-cell case is not run. `modules/analysis/test/gpu-pointcloud/pointcloud-diamond-morphology.spec.ts` pins working `GPUPointCloudSMRF` composition through published erosion/dilation, radius at most 8 per pass, bit-identical for radii 1–37.

### 6. `GPUHistogram`: domain views and bin edges

Problem: literal, `'auto'` and two-row GPU domains use `u32((value - minimum) / (maximum - minimum) * f32(BIN_COUNT))` (`gpu-core/gpu-histogram.js`, `binCalculation`). For 16 bins over `[-1e38, 1e38]`, values 0, 1, -5e37, 5e37 give bins 8, 8, 4, 12 with literals and all bin 0 with a view. Reciprocal multiplication plus subnormal flushing is an inference: `1 / 2e38 = 5e-39`, range above 2^126; generated code was not inspected for this case.

Exact float32 edges also land low. Inputs are `n + 1` edges of `n` bins, width `w`, from `minimum`; NumPy 2.5.4 `numpy.histogram` with float32 inputs gives one per bin and two in the last. Observed on Apple M3 Pro/headless Chromium with alpha.12:

| Bins | Width, minimum | Low edge indices: literal | Low edge indices: GPU view |
| --- | --- | --- | --- |
| 49 | 1, 0 | none | 7, 14, 27, 28, 35, 41, 47 |
| 49 | 8, 1024 | none | 7, 14, 27, 28, 35, 41, 47 |
| 3 | 41, 0 | 1, 2 | 1, 2 |
| 6 | 41, 0 | 1, 2, 4 | 1, 2, 3, 4 |
| 12 | 41, 0 | 1, 2, 4, 7, 8 | 1, 2, 3, 4, 6, 8 |
| 7 | 83, 0 | 1, 2, 4 | 1, 2, 4 |

Both forms are correct for 7, 255, 1000 bins of width 1 from 0; 12 of width 0.25 from -16; 30 of width 8 from 1024. Smallest case: 3 bins `[0, 123]`, values 0, 41, 82, 123: expected 1, 1, 2; observed 2, 1, 1. At 49 bins `[0, 49]`, values 0–49, only the view puts 7 in bin 6, 14 in bin 13, etc. Literal/view lists match constant `BIN_COUNT / range` multiplication/runtime reciprocal multiplication respectively in these rows, not every case. Correctly rounded CPU division/multiplication instead misses edge 27 of 49 and edges 251, 253, 502, 506, 511 of 1000 width-1 bins, although the GPU gets the latter case right. All three evaluations have boundary errors.

Change: decide bins using `k * range <= (value - minimum) * BIN_COUNT < (k + 1) * range` (exact when products are exact), or power-of-two scaling above 2^126. Product remedy is not GPU-verified on these cases. Add wide-domain and every-edge tests for both forms. Separately permit domain/output offsets in larger views and NaN for finite extent. Retires `BIN_WGSL` in `dataframe-analysis/column-profile/column-profile-kernels.ts` (`GPUColumnProfile` histogram).

Evidence: `modules/analysis/test/gpu-core/gpu-histogram-domain-view.spec.ts` has 12 tests and runs published literal/view forms twice per case and pins observed counts, including failures of the desired convention; flip assertions when fixed. `modules/analysis/test/gpu-dataframe/column-profile/gpu-column-profile.spec.ts` documents the boundary reason but the reproduction does not run its local kernel. Confirmation needed on a second adapter; domains bracketing 2^126, narrow domains far from zero and bin-centre inputs remain untested.

### 7. `GPUKMeans`: seeding and tolerance

Problem: `'evenly-spaced'` valid-row seeds only, stop on unchanged labels, empty cluster retains centroid (`gpu-vector-search/gpu-k-means.js`, `GPUKMeans`). For 80 blob-A rows then 10 B then 10 C, `clusterCount: 3`, `floor(index * rowCount / clusterCount)` selects 0, 33, 66, all A. Recovery depends on geometry; case not run.

Change: `seed: 'evenly-spaced' | 'k-means++'`, `seedValue: number`, GPU distance pass and counter-based weighted draw per centroid; `tolerance?: number` on largest centroid shift with status. Optional restarts, inertia, per-row squared distance and empty-cluster relocation to the farthest row. Retires matching portions of `GPUSpatialKMeans` in `gpu-spatial-analysis/spatial-clustering/gpu-spatial-kmeans.ts` (2D points). Evidence: reading; no same-input numeric comparison.

### 8. `GPURasterContours`: levels and world coordinates

Problem: one `level` per instance and local pixel-centre output (`gpu-raster/gpu-raster-contours.d.ts`). Ten levels on 4,096 × 4,096 cells means ten scans of 16.8 million cells, ten guessed segment capacities and a world-transform pass per output, with a local origin needed for precision. `gpu-raster/gpu-raster-contours.js` scales all four corners and the level by the largest magnitude before the saddle determinant, so division rounding can lose an integer-raster tie that should connect the high diagonal.

Change: `levels: number[] | GraphDataView<'float32'>`, shared segment output with level indices/offsets and `requiredSegmentCount`; `coordinates: 'pixel' | 'world'` relative to a stated origin. Take unscaled differences and scale only when products risk overflow (local threshold 1e18). Later options: global edge IDs for stitching and two-level bands. Retires segment output in `raster-analysis/isolines/gpu-isolines.ts` (`GPUIsolines`); `gpu-isobands.ts` (`GPUIsobands`) needs bands. Local classes use the bilinear saddle rule; this alone does not establish numeric parity.

Evidence: declaration reading; `modules/analysis/test/gpu-raster/isolines/gpu-isolines.spec.ts` supplies the random-field tie regression, and `modules/analysis/test/gpu-raster/isolines/isolines-saddle-rule.spec.ts` covers saddle topology and integer ties. The latter checks corners 7, 4, 7, 1 at level 5 (differences 2, -1, 2, -4; determinant 0) and the mirrored 4, 7, 1, 7 locally; it does not assert published tie behavior. `GPUTerrainContours` in `gpu-terrain/terrain-analysis/gpu-terrain-contours.ts` already delegates one level.

### 9. `GPUHashJoin` and `GPUGroupAggregation`: wide and sparse keys

Problem: packed uint32 join keys and dense uint32 aggregation keys below `output.length` (`gpu-core/gpu-hash-join.d.ts`, `gpu-group-aggregation.d.ts`). H3 `0x8928308280fffff`, Quadbin and Arrow Int64 do not fit. Keys 7 and 4,000,000,000 require four billion output rows. Float compare-exchange sums depend on thread order; join has no row mask.

Change: `uint32x2` keys, low word first in storage, for hash index/join/grouping; compact sparse `(key, statistic)` output with count/overflow; left mask and 1:n aggregates over repeated right keys. Retires `GPUKeyJoin` in `dataframe-analysis/key-join/gpu-key-join.ts`, count/sum/mean/min/max of `GPUGroupStatistics` in `dataframe-analysis/group-statistics/gpu-group-statistics.ts` (moments/order statistics remain), and table build of `GPUCellAggregation` in `gpu-spatial-analysis/cell-aggregation/gpu-cell-aggregation.ts`. Evidence: declaration reading.

### 10. Exact point-in-ring fallback

Problem: `GPUPairwisePointInPolygon` returns `uncertain` (3) for arithmetic-envelope overflow or degeneracy (`geospatial/gpu-pairwise-point-in-polygon.d.ts`). A square `(10, 0)-(14, 4)` with a second ring `[(11, 1), (12, 2)]` has an area-free ring: published classification is uncertain; exact-orientation even/odd classification skips it.

Change: integer exact orientation when the fast filter fails; `EXACT_ORIENTATION_WGSL` in `analysis-core/kernels/exact-orientation-wgsl.ts` computes six products exact in 48 bits, aligns exponents and compares two wide unsigned accumulators. Keep uncertain only for malformed offsets; expose `boundaryRule: 'closed' | 'open' | 'half-open'`. Retires inline classification in `gpu-spatial-analysis/spatial-join/gpu-point-in-polygon-join.ts` (`GPUPointInPolygonJoin`) and fallback in `gpu-spatial-predicate-join.ts` (`GPUSpatialPredicateJoin`), with indexed candidate pairs as a separate request.

Evidence: `modules/analysis/test/gpu-spatial-analysis/spatial-join/gpu-point-in-polygon-classifier-agreement.spec.ts` compares local and published classification on four scenes: shared edges/vertices, projected metres, one float32 step either side of an oblique edge, and the degenerate ring (the disagreement).

### 11. Standalone uint64 and H3 tables as data

Problem: WGSL has no int64; `dggs_u64_*` and `fp64_u64_*` use `vec2u(high, low)` but omit multiply/divide and require whole shader modules. Epoch-millisecond division, accumulators and key comparison need local words. H3 centres/azimuths exist only in `dggs_h3_get_face_center_geo` and `dggs_h3_get_face_axis_azimuths`; a renamed function yields 0 parsed rows and an empty dependent array without validation.

Change: standalone `uint64` add/subtract/negate/compare, `shiftLeft`, `shiftRight`, `multiply32` (64 by 32), `divide32`, float32 conversion and storage-order converters; dggs/fp64arithmetic depend on it. Export centre/azimuth data from `@luma.gl/shaders` as two frozen 20-row arrays and generate WGSL from them; the summary requests all four H3 tables. Retires `UINT64_WGSL` in `analysis-core/kernels/uint64-wgsl.ts`, word helpers in `analysis-core/cell-key-wgsl.ts` (`getFixedPointWGSL`) and `gpu-spatial-analysis/cell-aggregation/cell-keys.ts` (`CELL_KEY_WGSL`), `TERRAIN_SUMMED_AREA_WGSL_HELPERS` in `gpu-terrain/topographic-position/terrain-summed-area-table.ts`, `add64`/`subtract64` in `dataframe-analysis/pair-statistics/gpu-ripley.ts`, `divideUnsigned64`/`negateInt64` in `dataframe-analysis/calendar-buckets/gpu-calendar-buckets.ts`, plus `parseDggsFaceTable` in `gpu-spatial-analysis/cell-indexing/h3-index-extended-wgsl.ts`.

Evidence: `modules/analysis/test/utils/wgsl-words-search.spec.ts` compares words with BigInt; `modules/analysis/test/gpu-spatial-analysis/cell-indexing/h3-index-extended.node.spec.ts` requires 20 parser rows. Published side: reading.

### 12. One meaning of count

| API | Current count | Unclamped value |
| --- | --- | --- |
| `GPUHashJoin` (`gpu-core/gpu-hash-join.d.ts`) | Required matches before truncation | `count`; `overflow` flags excess |
| `GPUBVHQuery` (`gpu-core/gpu-bvh-query.d.ts`) | Full match count | `count` |
| `GPUPointSpatialQuery` (`geospatial/gpu-point-spatial-query.d.ts`) | Clamped to `output.ids.length`, safe for indirect draw | `totalCount` |
| `GPURasterContours` (`gpu-raster/gpu-raster-contours.d.ts`) | `segmentCount = min(required, capacity)` | `requiredSegmentCount` |

Problem: 1,000 matches into capacity 600 means `i < count` reads 400 rows past join output but safely consumes point-query output. Change: clamped `count`, unclamped `requiredCount`, `overflow = requiredCount > capacity`; add requiredCount to hash join, BVH query and grid-index query before a documented count change. Retires nothing: `GPUCompactOutput` in `analysis-core/kernels/gpu-contributor-types.ts` already follows this rule through `status.requiredCount`. Evidence: four declarations, reading only.

### 13. Four shader-compiler facts

`research/math-correctness-review-2026-11-10/shader-compiler-findings.md` includes generated Metal and reductions. Facts 13.1, 13.2 and 13.4 are Metal fast math: the generated Metal gives the IEEE result with fast math off; Chromium 151 exposes no switch. Fact 13.3 is an Apple GPU back-end miscompile of correct Metal, independent of fast math; browser/Apple reports are drafted and unfiled. Reassociation and inexact division are permitted by WGSL. The table failure is not. All measurements cover one machine, OS and browser version.

**13.1 Compensation disappears.** Kahan `(next - sum) - term`, Neumaier `(sum - next) + term` and float two-sum are optimized to zero. A million terms, mean 1234.567, one quarter negative: true sum 1,234,041,587.7; correction 0, plain/compensated GPU total 1,234,025,472 (16,115 off); IEEE serial CPU 1,234,033,152. Integer-significand rounding errors give a total 0.44 off before final rounding; fixed point gives nearest float32. The local H3 guarded double-float path has 0 h3-js mismatches where float32 misses more than 0.5%.

Cure/change: integer fixed point with known range, per-addition error from integer significands, or guarded fp64arithmetic two-sum; publish the device-appropriate fp64 defines/runtime-zero guard (`LUMA_FP64_CODE_ELIMINATION_WORKAROUND`). A float `x + ONE * 0.0` barrier folds away. Test GPU sums at a million terms, not just CPU twins. Retires applicable `getAdditionRoundingError` in `analysis-core/kernels/addition-rounding-error-wgsl.ts` and fixed-point paths in `analysis-core/kernels/fixed-point-accumulator.ts` when published reductions/scans meet the needed accuracy. No installed published graph kernel contains Kahan/Neumaier/two-sum by text search; future compensated sums need this protection.

Evidence: `modules/analysis/test/utils/addition-rounding-error-wgsl.spec.ts` pins the totals and 4,022 signed pairs bit for bit; `modules/analysis/test/gpu-spatial-analysis/linear-referencing/gpu-linear-referencing.spec.ts` covers 86,400 equal segments, a prefix near 1.2 million off by 16.43; `modules/analysis/test/gpu-spatial-analysis/geometry-measures/gpu-geometry-measures-summation.spec.ts`, `modules/analysis/test/gpu-spatial-analysis/shape-descriptors/shape-descriptors-moments-summation.spec.ts`, `modules/analysis/test/gpu-spatial-analysis/cell-indexing/h3-index-extended.spec.ts`.

**13.2 Differences regroup.** `(y - mean) - intercept - slope * (x - xmean)` becomes `y - (mean + intercept + ...)`, even across statements. For response near 10,000 and residuals about 0.002, residuals become multiples of the 10,000 float32 step; Chow F is 12% off and spatial-error first-stage `s2` 5% off. Values 1e7, 1e7 + 1, +2, +3 have mean 1e7 + 1.5, stored rounded with correction -0.5; first z-score becomes -1.549 rather than -1.162.

Cure/change: XOR the first difference's bits with runtime-zero buffer data (a literal folds away). Publish a rounded-difference/kept-intermediate helper; retires `getRoundedDifference` in `gpu-spatial-analysis/spatial-regression/rounded-difference-wgsl.ts` and the same guard in `dataframe-analysis/group-statistics/group-statistics-moments.ts`. Evidence: `modules/analysis/test/gpu-spatial-analysis/spatial-regression/math-review-restricted-regimes.spec.ts` ("response mean far above its residuals"), `modules/analysis/test/gpu-spatial-analysis/spatial-regression/math-review-error-ml-high-fit.spec.ts` ("response mean of 10000"), `modules/analysis/test/gpu-dataframe/group-statistics/gpu-group-statistics.spec.ts` (1e7). Lag-ML, GM and 2SLS residual chains are untested; tested `GPUOrdinaryLeastSquares` does not regroup. No published kernels were tested for this shape.

**13.3 Integer tables lose stores.** Function-local or `var<private>` `array<u32, N>`/`array<i32, N>`, filled only by constant-trip loops then read at unresolved indices, returns zeros for any stored value (index, constant, sentinel, buffer copy). Reduced WebGPU: lost at 33 to about 250 entries with 3–96 loop iterations; right through 32. Louvain: lost from 12 through every size tried to 48; from 33 when a helper with two float arrays is removed. No size is generally safe. Tint's Metal and Metal front-end intermediate are correct; the Apple back end loses stores after zero initialization when unrolling. Metal `#pragma clang loop unroll(disable)` cures it; fast math on/off does not.

Cures: literal initializer/module `const` for known tables; runtime-bound fill such as `N * min(arrayLength(&binding), 1u)` otherwise; remove identity tables. `var<private>` alone is not a cure. Float/vector tables, later data-dependent stores, simultaneous storage writes and conditional stores also passed, for unknown reasons. Current audited kernels pass their permitted shapes; this proves only those compiled inputs. Retires no luma.gl replacement: keep the authoring rule and send the reduced bug to browser/Apple. Published identity fills in `gpu-physics/sphere-world-shaders.js`, `gpu-core/gpu-text-selection.js`, `gpu-core/gpu-sort-chunks.js` write storage, not local arrays; no published fault is known from that limited text search.

Evidence: findings report above; `modules/analysis/test/analysis-core/shader-compiler-table-fill.spec.ts`, `modules/analysis/test/analysis-core/shader-compiler-table-shapes.spec.ts`, `modules/analysis/test/analysis-core/table-audit/*.spec.ts`, `modules/analysis/test/utils/function-scope-table-readback.spec.ts` (one-binding reduction, dense LU pivot table, natural neighbour, phenology), `modules/analysis/test/gpu-network/network-optimization/gpu-network-louvain-visit-order.spec.ts` (13 sizes, 2–512, five seeds against TypeScript Fisher-Yates).

**13.4 Division is inexact.** A region mean computes `6.0 / 3.0 = 1.9999999`; WGSL permits 2.5 ULP. CPU twins cannot demand exact quotients or serial loop sums. Cure/change: products and comparisons at bin/cell boundaries (`k * range <= (value - minimum) * count < (k + 1) * range`), with product-rounding limits respected; no local code retirement by this observation alone. Evidence is an observation, not an exact-result assertion: `modules/analysis/test/gpu-spatial-analysis/spatial-regionalization/gpu-region-partition-evaluation-nonfinite.spec.ts` uses relative 1e-6; recorded in `research/math-correctness-review-2026-11-10/point-process-clustering.md`.

Request for all four: one WGSL authoring page with cures and CPU-twin limits, public runtime-zero helpers, and GPU tests large enough to expose summation, centring and local-table failures. The table miscompile belongs with browser/Apple, not a luma.gl implementation fix.

### 14. Smaller gaps

All rows are declaration readings. Local paths identify the symbols that could delegate or simplify.

| Published API | Request | Local code it would retire or simplify |
| --- | --- | --- |
| `GPUGridIndex` | Cell-size mode `origin + k * cellSize`, half-open domain, ascending cell order, Z-ignoring raster mode, named limits, offsets/sorted IDs as read views | `createPointCloudGridIndex`, `pointcloud-analysis/pointcloud-grid-index.ts` |
| `GPUGridBinning` | Per-frame active grid size | `GPUFlowAggregation`, `gpu-network/flow-aggregation/gpu-flow-aggregation.ts` grid zones |
| `GPUGridBinning`, `GPUGridAggregation` | Per-row cell key or deterministic fixed-point sum instead of float compare-exchange | `GPUPointDensity`, `gpu-spatial-analysis/point-density/point-density-kernels.ts` (`createPointDensityGridKeysNode`) |
| `GPURasterRegionMeasurements` | Exact deterministic sums and per-row cell area | `GPUTerrainCatchmentSummary`, `gpu-terrain/terrain-processing/gpu-terrain-catchment-summary.ts` area/mean/extrema |
| `GPURasterDilation`, `GPURasterErosion` | Mask and GPU-gated fixpoint reconstruction | `GPURasterReconstruction`, `raster-analysis/gpu-raster-reconstruction.ts` |
| `GPURasterContrast`, `GPURasterGlobalPercentile` | Statistics window/region mask, per-frame mode, `[0, 1]` output, caller histogram and per-frame percentile | `GPURasterStretch`, `raster-analysis/raster-stretch/gpu-raster-stretch.ts` |
| `GPUFiniteDifference2D` | Per-row spacing and no-data (proposal 4) | `GPUTerrainFlowField`, `gpu-terrain/terrain-flow-field/gpu-terrain-flow-field.ts` gradient |
| `GPUGallopingSearch` | One-dispatch lower bound over `0..segmentCount`, 64-bit keys | `createSortedSegmentOffsetsNode`, `analysis-core/kernels/sorted-segment-offsets.ts` |
| `GPUDGGSCellProjection` | Invalid-cell value option; boundary/vertex-count outputs | `GPUCellGeometry`, `gpu-spatial-analysis/cell-indexing/gpu-cell-geometry.ts` |
| `GPURasterGradient`, `GPURasterNeighborhood` | One-sided/extrapolating border | `GPUTerrainDerivatives`, `gpu-terrain/terrain-analysis/gpu-terrain-derivatives.ts`; `GPUReliefShading`, `gpu-terrain/terrain-illumination/gpu-relief-shading.ts` padded bands |
| `GPURasterBandMath` | Per-frame operation; unary, power, minimum, maximum | `GPURasterArithmetic`, `raster-analysis/raster-algebra/gpu-raster-arithmetic.ts` |
| `GPURasterNeighborhood` | Minimum, maximum, range, standard deviation, valid count | `GPUFocalStatistics`, `gpu-spatial-analysis/spatial-interpolation/gpu-focal-statistics.ts` |
| `GPUCompaction` | Bounded capacity and required count | `GPUNetworkSubgraphFilter`, `gpu-network/network-subgraph-filter/gpu-network-subgraph-filter.ts` scan/scatter |
| `GPUReduction` | Two-word 64-bit integers beside `GPUScanUint64` | Network K edge-draw prefix, `gpu-network/network-k-function/gpu-network-k-function.ts` (`GPUNetworkKFunction.getCommandNodes`, `chunk-prefix` variant) |
| `GPUPointCloudVoxelDownsample` | Per-voxel centroids/attribute means | `GPUPointCloudVoxelMerge`, `pointcloud-analysis/gpu-pointcloud-voxel-merge.ts` input |
| `compileProjectionPlan` | `destinationOrigin` so error-budget validation uses the actual relative origin | `GPULocalFrameProjection`, `gpu-spatial-analysis/geometry-measures/gpu-local-frame-projection.ts` override |
| `GPUGraphAlgorithmsSingleSourceShortestPath` | Multiple sources/start costs, per-frame views, compact frontier | `GPUNetworkReachability`, `gpu-network/network-reachability/gpu-network-reachability.ts` |
| `GPUGraphAlgorithmsModularityOptimization` | Multilevel coarsening | `GPUNetworkLouvain`, `gpu-network/network-optimization/gpu-network-louvain.ts` (32 vertices) |
| `random` shader module | Counter-based Philox and open-interval unit float | `PHILOX_WGSL`, `getPhiloxOpenUnitFloat`, `analysis-core/kernels/philox-random.ts` |
| `GPUCommandGraph` | Per-node shader description on `preflight.nodes[]` | Per-graph kernel record table, [upstream-first-recording.md](./upstream-first-recording.md) |
| `GPUScan` (and sort/reduction recursion) | Size-independent recursion levels driven by runtime parameters, so pipeline counts do not grow with input size (36 to 38 variants between 1M and 2M rows, scalability pass SC) | Chunked contributors' constant-pipeline guarantees, for example `GPUGeographicallyTemporallyWeightedRegression` and `GPUKernelDensity` |
| `GPUBVH` | Two-dimensional or bounded refit dispatch; the one-dimensional refit exceeds `maxComputeWorkgroupsPerDimension` above 16,777,216 leaves | `GPUFuzzyContiguity` dispatch pre-check (`gpu-spatial-analysis/spatial-weights/gpu-fuzzy-contiguity.ts`) |
| `GPUReduction` and sort kernels | Runtime chunk, recursion-level and input-size bounds instead of specializing shader source for each size; stable pipelines across unequal chunks | Profile, statistics and tiled aggregation contributors still inherit published reduction/sort shader variants (IMP F-DF/F-STATS) |
| `GPUCommandGraph` | Scratch lifetime planning for independently seeded batches: permit explicit batch dependencies or a scheduling policy that retires scratch before the next batch | CSR visibility planned 35.7 GB while independent centrality batches stayed live simultaneously; explicit dependencies bounded their lifetimes (IMP SC3-VIS) |

**Suggested filing order**

1. Proposals 1, 12 and 13; report 13.3 to the browser first.
2. Proposal 6 after second-adapter confirmation.
3. Proposal 2: largest retirement here, six local kernels.
4. Proposals 3, 5, 10 and 11: contained kernel/copy removals.
5. Remaining features: 4, 7, 8, 9 and 14.
6. Proposal 15: 15.1, 15.2 and 15.4 are small luma.gl changes; 15.6 and 15.8 go to deck.gl first because the port carries a patch for 15.6 and 15.8 affects every native non-billboard path on a globe.

### 15. Kepler port items (luma.gl alpha.14 and deck.gl 10.0.0-alpha.1)

Found while porting `examples/deck/kepler-compute` onto published packages (KE-PORT in [roadmap.md](./roadmap.md)). Nothing is filed. Paths: luma.gl under `node_modules/@luma.gl/<package>/dist`, deck.gl under `node_modules/@deck.gl/core`. Local code that each item would retire is listed in [upstream-workarounds.md](./upstream-workarounds.md).

**15.1 Export three helpers that exist unexported in alpha.14.** Evidence: `reading` plus a failed import.

- `walkTextGlyphs`, with `TextGlyphWalkOptions`, `TextGlyphVisit` and `TextRowMetrics`, is declared in `engine/dist/text/fonts/atlas/text-glyph-walker.d.ts` and documented as the canonical CPU atlas layout walk. `engine/dist/text/index.d.ts`, `experimental.d.ts` and `text-2d/index.d.ts` do not export it.
- `alignGraphVectorViews` and `getGraphVectorData` are declared in `graph/dist/gpu-core/graph-vector-view-utils.d.ts`; `graph/dist/gpu-core/index.d.ts` does not export that module.
- Reproduction (`reading`: the entry-point declarations were read, the import was not compiled separately): `import {walkTextGlyphs} from '@luma.gl/engine/text'` has no exported member to resolve. Likewise `import {alignGraphVectorViews} from '@luma.gl/graph/gpu-core'`. An `_internal` or deep `dist` import is forbidden here ([upstream-first-plan.md](./upstream-first-plan.md)).
- Request: export the three names from `@luma.gl/engine/text` and `@luma.gl/graph/gpu-core` with their types. Retires `walkKeplerLabelGlyphs` and `alignKeplerGraphVectorViews`; the local copies match the declared behaviour and are used by the label and resident-chart paths.

**15.2 A typed out-of-memory error.** Published `@luma.gl/core` exports no `OutOfMemoryError`, and an asynchronous WebGPU out-of-memory is not thrown. Reproduction (`reading`, plus the example's scope helper): allocate buffers until the device reports out-of-memory; the error arrives through the uncaptured-error path as a plain `Error`, and `Device.pushErrorScope` does nothing outside debug mode. Request: the fork's typed error and scope balancing (`modules/core/src/api/utils/out-of-memory-error.ts` at `e0ed33791`; section 5). Kepler works around it with `KeplerDeviceOutOfMemoryError` and a raw-handle scope in `binding5-device-oom-scope.ts`.

**15.3 Resource planning before compile.** `GPUCommandGraph.planResources()` and `compilationStats` are absent, so Kepler admits after compile (`MeteredCommandGraph`). Request and fork reference are in section 5 ("Resource planning without allocation").

**15.4 Text atlas options the port needed.** Evidence: `spec` for mipmaps (`modules/kepler/test/gpu-layers/gpu-point-label-layer.spec.ts`), `reading` for the cache.

- `GPUTextResourcesProps` has `fontAtlas` and `id` only; the atlas texture is a one-level `2d-array` with `{minFilter: 'linear', magFilter: 'linear'}` and no `mipmapFilter` (`engine/dist/text/text-2d/gpu-text-resources.js`). The fork accepted `mipmaps` and set `mipmapFilter: 'linear'` with `mipLevels: 'auto'`.
- Reproduction: build a bitmap atlas at the default 64 px, draw labels at 15 to 24 px with `GPUTextResources`, and compare with the deck.gl 10 `TextLayer` on the same text and font. The label paints one pixel narrower (about 290 differing pixels of about 360 painted, maximum channel error 167). Swapping in a `DynamicTexture` with `mipmaps: true`, `mipLevels: 'auto'` and `mipmapFilter: 'linear'` brings the same comparison under the original tolerance (12 differing pixels, error at most 12), in cartesian and geographic views, for three anchor and alignment settings.
- `BrowserFontAtlasSettings` has no `cache`. `buildBitmapFontAtlas` always caches per font settings (three entries) and extends the shared canvas and mapping in place; the fork's `cache: false` returned a private atlas for explicit ownership (`modules/engine/src/text/fonts/build-bitmap-font-atlas.ts`).
- Request: `mipmaps` on `GPUTextResourcesProps` (and the mip filter on the sampler when set), and `cache?: boolean` on the bitmap atlas settings.

**15.5 Geometry, polygon-contiguity and trip preparation.** The fork's engine arrow module (`modules/engine/src/arrow/geometry/` at `e0ed33791`) prepared native GeoArrow, WKB and JSON geometry for GPU rendering, globe cutting, timed trips and fp64 residuals. The port re-homed it as 22 Kepler-prefixed symbols in `examples/deck/kepler-compute/src/geometry-preparation/` on public APIs. Request: a public geometry-preparation module in engine or graph/arrow. One finding for `@math.gl/arrow` (evidence: `spec`, `modules/kepler/test/resident-select-geometry.node.spec.ts`; the failing route was probed in the geometry stream): a `FixedSizeList(3, Field('value', Float64))` point vector throws `Ambiguous three-component coordinates` on the default, generic and explicit-dimension routes; unnamed three-wide lists should default to `xyz` or accept a dimension. The local helper `nameKeplerCoordinateDimension` renames the child field and shares the buffers.

**15.6 Three hunks against deck.gl 10.0.0-alpha.1.** Carried as a yarn patch of `@deck.gl/core` (`.yarn/patches/@deck.gl-core-npm-10.0.0-alpha.1-*.patch`; the hunks are in `src/lib/deck-renderer.ts`, `src/lib/deck-picker.ts` and `src/lib/deck.ts`). Evidence: `reading` of deck.gl source and confirmation that the patch is installed; none was exercised in a browser during the port. Reproductions below are the sequences the code paths imply.

1. Depth attachment on the effect input. `DeckRenderer` creates its two render buffers with `colorAttachments` only (`deck-renderer.ts`, around `deck-renderbuffer-${i}`). Reproduction: render two overlapping layers with depth testing (for example extruded `ColumnLayer`s at different distances) with any effect in `effects`; the scene pass runs with no depth attachment, so later layers paint over nearer ones. Fix: `depthStencilAttachment: 'depth24plus'` on the first buffer; the second (swap) buffer only receives fullscreen effects.
2. Hover clear on canvas leave. `DeckPicker.pickObject` and `pickObjects` return `{result: [], emptyInfo}` early when there are no viewports, without `processPickInfo`, so `lastPickedInfo` keeps the previous object. Reproduction: hover a pickable layer with `onHover` and a highlight, then move the pointer off the canvas; `onHover` is not called with a cleared info and the highlight stays. Fix: run `processPickInfo` with `pickedColor: null, pickedObjectIndex: -1` in that branch and return the infos that carry a layer.
3. `isHovering` from `info.picked`. `Deck._onPointerMove` sets `cursorState.isHovering = result.length > 0` (`deck.ts`). A result array can hold infos with `picked: false` (a cleared layer), so the cursor keeps its picking style over empty space. Fix: `result.some(info => info.picked)`.

Request: three small pull requests, or one, with these reproductions turned into tests in deck.gl's picking and effects suites.

**15.7 Flow particle spawn region.** `FlowParticleSimulation` in alpha.14 (`graph/dist/experimental/simulation/flow-particle-simulation.d.ts`) has `setField`, `reset` and `step` only. The fork's `setSpawnRegion(region)` (`modules/graph/src/experimental/simulation/flow-particle-simulation.ts`, `flow-particle-shaders.ts` at `e0ed33791`) added a `spawn` rectangle uniform in normalized field coordinates: particles spawn inside it and respawn when they leave it. Reproduction: zoom a Flow Field layer to a tenth of the field; with the published class roughly a hundredth of the particle budget is on screen (area scales with the square of the zoom; not measured), where the fork kept the budget on screen. Evidence: `reading` and the fork shader. Request: a spawn region on the simulation props or as a setter, defaulting to the full field.

**15.8 deck.gl 10 WGSL `PathLayer` on a globe, non-billboard paths.** In `core/src/shaderlib/path/path.wgsl.ts` (lines 149 to 152) the join code writes `deltaA3 = rotationResult.transform * deltaA3` and the same for `deltaB3`. The GLSL path layer (`layers/dist/path-layer/path-layer-vertex.glsl.js`, `deltaA3 = deltaA3 * rotationMatrix`) multiplies from the left, which is the product with the transposed matrix. WGSL `M * v` is the untransposed product, so the tangent-frame rotation of the segment deltas is inverted. Reproduction (measured in the port's headless run by the gpu-layers stream; not committed as a spec): render the same path on a `GlobeView` with `billboard: false` through deck's native WGSL `PathLayer` and through the resident path layer, which applies the transpose; the native layer paints 38 pixels where the resident one paints 254. With `billboard: true` the two agree, because the rotation branch is skipped. Request: use `transpose(rotationResult.transform) * delta` (or multiply from the left) in `path.wgsl.ts`, with a `GlobeView` render test for non-billboard paths.

## 3. GPU-resident iteration (PLAT-04)

Public compute nodes already accept `condition: {source: 'gpu', mode: 'indirect', buffer, byteOffset}` (`GPUCommandGraphGPUIndirectCondition` in published `gpu-core/gpu-command-graph-types.d.ts`). Local `createWGSLActiveCountDispatch` and `createWGSLDispatchGate` in `analysis-core/kernels/wgsl-kernel-nodes.ts` use it. The missing request is a GPU-gated continuation that does not unroll the entire worst-case iteration budget, plus one convergence readback per batch of rounds. Published primitives also need public dispatch metadata or `gateGPUCommandNodes` so a caller can gate their exact dispatch geometry.

`GPULoopOperation` unrolls `maximumIterations` copies. `GPUNetworkCentrality` in `gpu-network/network-centrality/gpu-network-centrality.ts` unrolls reachability/sigma/delta rounds, `maximumIterations` 1–1,024, default 64, and rejects over 32,768 command nodes (NW-CENT). Exact betweenness around 15,000–20,000 nodes needs sampled pivots under this budget. A continuation would replace this expansion and allow reconsidering the guard; local iteration gates remain useful until then.

WGLog (arXiv 2607.17571, 2026-07-20), described in [research-gpu-landscape.md](./research/spatiotemporal-review-2026-11-09/research-gpu-landscape.md), reports approximately 2.5 ms for a 4-byte mapAsync readback (285 times a CUDA 4-byte copy) and 0.18 ms for an empty fence. Its GPU writes indirect arguments, dispatches zero after convergence, batches K = 30 rounds (adaptively halved), double buffers state and reads one fixpoint word per batch. Indirect dispatch alone gives 4.7 times, batching 15.4 times over readback-per-round on recursive joins; it runs in Chrome on M5 Pro. These are the paper's numbers, not measurements of this library.

Proposed `GPUIterationBatch` (name open): record caller-controlled K copies of a subgraph, with a device-side active flag, a subgraph-written convergence word and one readback per batch; the host chooses the next K. Specify zero-workgroup behavior, double buffering to avoid read/write aliasing within a pass, and known-iteration/early-convergence tests.

## 4. Adapter limits, subgroups and buffer_view (PLAT-05)

`createDevice({featureLevel: 'max'})` already requests adapter limits and every supported feature: see `getRequiredWebGPUFeatures` and `WebGPUAdapter.create` in `node_modules/@luma.gl/core/dist/webgpu/adapter/webgpu-adapter.js`. The request is adapter limits without enabling all features, for example a limits-only policy or selected adapter-limit names; explicit numeric `requiredLimits` already works. Spec defaults are 8 storage buffers per stage, 256 MiB buffers and 128 MiB bindings. The recorded headless Apple device reports binding/buffer limits 4,294,967,292, 10 storage buffers and 1,024 invocations. `getAnalysisDeviceLimits` in `modules/lenses/src/computation-views/hosts/device-profile.ts` already requests up to 16 storage buffers and adapter buffer/binding maxima. PLAT-02 storage-binding guards remain a separate repository item.

The cited web3dsurvey reports (collection date unspecified) show 98% with ≥10 storage buffers, 80% with ≥16, 77% with ≥2 GiB `maxBufferSize` and 93% with ≥512 workgroup invocations; the cited browser coverage gives Chrome 146 up to 16 storage buffers and subgroups from Chrome 134, with none found through Safari 27.2/Firefox 162. Proposed `createDevice({requiredLimits: 'adapter-max'})` or selected limit names would request adapter values and report grants in `device.limits`.

Published `GPUReduction` already selects `subgroups` or `portable` via `getGPUReductionStrategy`; request equally tested feature-and-WGSL-extension-gated paths for `GPUScan`/`GPUSort`, retaining hierarchical workgroup-memory fallbacks. The landscape evidence reports about 68% subgroup availability, 11–28 times slower emulation on M4, and radix-sort rates near 1 G keys/s portable on M1 Max versus 3 G/s with real Metal subgroups; these are external reports, not local measurements. `bufferView<T>()`, `bufferArrayView<T>()` and `bufferLength()` pack arrays into fewer bindings and could relieve geographically weighted regression binding pressure, but the landscape places them in Chrome 153–154 only. Wait for a second browser before API work and preserve a portable fallback.

## 5. Implementations that already exist in the fork

The fork is luma.gl commit `e0ed33791` (branch `backup/map-contributors-pre-split-2026-11-07`, local only). Its code is public at the same paths on [`BertCh/luma.gl`, branch `reference/map-contributors-pre-split-code`](https://github.com/BertCh/luma.gl/tree/reference/map-contributors-pre-split-code) (commit `244ddc5ba`, datasets and fixtures left out). [research/fork-comparison-2026-11-13](./research/fork-comparison-2026-11-13/README.md) has the full table with fork paths, tests, the alpha.13 status of each and the measurements the fork recorded. The fork code was read for that comparison, not run. Paths below are under the fork's `modules/graph/src` unless they name another module.

| Item above | Fork code | Note |
| --- | --- | --- |
| Kernel-node factory (top five 1) | `gpu-core/gpu-kernel-node.ts`: `createGPUKernelNode`; `utils/gpu-spatial-operation.ts` | Every fork primitive was written with it |
| Algorithms on a caller CSR (top five 2, 2.2) | `gpu-graph-algorithms/gpu-graph-topology-view.ts` and view-accepting degree, components, PageRank, core number, label propagation | No node or slot masks yet |
| Deterministic sums (top five 4) | 96-bit fixed-point sums in `gpu-vector-search/gpu-k-means.ts`, `gpu-raster/gpu-raster-cross-tile-components.ts`, `gpu-raster-global-statistics.ts` | Chunk-order reproducibility is tested for k-means |
| GPU-gated loops (top five 5, section 3) | `gpu-graph-algorithms/gpu-graph-iteration-gate.ts`, compact frontiers in BFS and SSSP, outputs invalidated on non-convergence | The gates still unroll; a continuation is new work |
| One grid cell formula, GPU bounds (2.1) | `gpu-core/gpu-spatial-utils.ts`, `gpu-grid-index.ts` (`boundsBuffer`) | |
| Haversine units and seam (2.3) | `geospatial/gpu-haversine-distance.ts` (`units`, metres by default, wrapping) | |
| Histogram edges (2.6) | `gpu-core/gpu-histogram.ts` edge correction with a parity spec | Not checked against the cases of 2.6 |
| Exact point-in-ring (2.10) | `geospatial/gpu-pairwise-point-in-polygon.ts` (exact float32 containment) | |
| Smaller gaps (2.14) | `compileProjectionPlan` at an explicit origin (`gpu-project/projection-plan.ts`); NaN for invalid DGGS cells (`gpu-dggs/gpu-dggs-cell-projection.ts`); voxel centroids and attribute means (`gpu-pointcloud/gpu-pointcloud-voxel-downsample.ts`); raster local origin (`gpu-raster/raster-utils.ts`) | |
| Exports and sort or scan speed | Onesweep sort, look-back scan, block compaction, cooperative segmented scan (`gpu-core/`) | Fork numbers: compaction 1.4 ms against 3.6 ms at 8M rows on Apple |
| External crossfilter masks | `gpu-crossfilter/gpu-crossfilter.ts`, `gpu-selection.ts`, `histogram-cube.ts` | |
| Dataframe statistics and description | `gpu-dataframe/gpu-query-compiler.ts` (`compilationStats`), `gpu-query-describe.ts` | |
| Exact Int64 time | `arrow/vectors/arrow-temporal-gpu-vector.ts`: `makeArrowTemporalWordGPUVector` | |
| Typed out-of-memory errors | `modules/core/src/api/utils/out-of-memory-error.ts`, scope balancing in the WebGPU device | |
| Per-node recording | `gpu-core/gpu-command-graph-describe.ts`, `gpu-command-node-record.ts`, the `recording` option | |

The comparison also found fork changes that are not in the summary above and are worth offering:

- **Resource planning without allocation.** `GPUCommandGraph.planResources()` returns a frozen plan equal to the compiled allocation, without allocating or compiling, and picks the smallest fitting allocation for reuse. An application can refuse a query before it allocates. Kepler measures after compiling today.
- **Readback and transfer.** Failed-map replacement and cancellation in `GPUReadbackRing`; a staging-buffer pool and an upload ring in core, so `readAsync` need not create one staging buffer per call.
- **Bind-group cache in `Kernel`**, keyed by buffer, offset and size and excluding destroyed resources, with a spec.
- **Strongly connected components** with O(V) scratch, checked against Tarjan to 5,000 vertices.
- **Core fixes:** std140 matrix alignment in the shader block layout, 16-byte rounding in place of 1 KiB uniform allocations, `destroyTestDevices`, borrowed-buffer and resize races in the dynamic resources, leaks in `Computation`.
- **Query language:** `choose` and `coalesce`, nullable `COUNT(column)`, quoted identifiers, `USING`, join-error readback.
- **Candidate-driven queries:** `GPUBVHQuery` on indirect frontiers and a `GPUGridIndexQuery` that gathers only the selected slots; the published versions scan every node or slot.
- **Arrow converters** on bounded two-dimensional dispatch; the published converters dispatch in one dimension.
