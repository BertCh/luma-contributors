# Proposals for luma.gl from the math correctness review (2026-11-10)

**Status: a draft kept in this repository. Nothing has been sent upstream: no issue, no pull request, no message to the luma.gl maintainers.** Under [AGENTS.md](../AGENTS.md) every item below belongs in luma.gl as an ordinary pull request, so none is implemented here. This repository does not wait on any of them.

Source: the math correctness review of 2026-11-10 ([luma-official.md](./research/math-correctness-review-2026-11-10/luma-official.md) and the "Upstream gaps" lists at the end of the other reports in that folder). Baseline: published `@luma.gl/{core,shaders,graph}@10.0.0-alpha.10`. Published paths below are under `node_modules/@luma.gl/graph/dist`; local paths are under `modules/analysis/src` unless they start with `modules/`.

Each section is one proposed pull request: the problem on a concrete input, the public API change, the local code it would retire (the rows of [upstream-workarounds.md](./upstream-workarounds.md)), and the evidence. Evidence is one of:

- **spec**: a committed spec in this repository shows it on the GPU or in Node;
- **replication**: arithmetic repeated in Node from the published kernel text, not run on a GPU;
- **reading**: the published declaration or kernel was read, nothing was run;
- **measured, not committed**: a stream of the fix pass ran it on the test GPU and reported the numbers, and no committed spec repeats the run.

The test GPU is the headless Chromium WebGPU device on an Apple GPU (Metal). Nothing here was run on another adapter.

| # | Proposal | Evidence |
| --- | --- | --- |
| 1 | One grid cell-coordinate formula, public as WGSL | reading and replication; the consequence has a spec |
| 2 | Graph algorithms on a caller-built CSR with masks; one self-loop rule | reading |
| 3 | `GPUHaversineDistance`: state the kilometre default, seam-safe longitude difference | replication; the local seam fix has a spec |
| 4 | `GPUFiniteDifference2D`: per-row spacing and no-data | reading |
| 5 | `GPURasterMorphology`: radii above 8, valid-sample count | reading; the delegation that works today has a spec |
| 6 | `GPUHistogram`: GPU-resident domain view, and values on bin edges | committed spec that pins the observed counts on one adapter; needs a second adapter |
| 7 | `GPUKMeans`: k-means++ seeding and a tolerance stop | reading |
| 8 | `GPURasterContours`: several levels per pass, world coordinates | reading |
| 9 | `GPUHashJoin` and `GPUGroupAggregation`: 64-bit and sparse keys | reading |
| 10 | Exact point-in-ring predicate where `GPUPairwisePointInPolygon` returns `uncertain` | spec |
| 11 | A 64-bit word shader module; the H3 face tables as data | spec for the local side, reading for the published side |
| 12 | One documented meaning of `count` | reading |
| 13 | Float compensated sums are removed by the Metal shader compiler; regrouped differences; a function-scope table that reads back as zeros; division rounding | spec for the first three, an observation for the fourth; the cause in the compiler was not found |
| 14 | Smaller gaps | reading |

## 1. One grid cell-coordinate formula, public as WGSL

**Problem.** Three published classes map a value to a grid cell with a private WGSL function named `getCoordinate`, and they do not agree. `GPUGridBinning` (`gpu-core/gpu-grid-binning.js:193`) and `GPUGridAggregation` (`gpu-core/gpu-grid-aggregation.js:236`) compute

```wgsl
min(u32((value - minimum) / (maximum - minimum) * f32(size)), size - 1u)
```

`GPUGridIndex` (`gpu-core/gpu-grid-index-internals.js:180`, repeated in `gpu-grid-index-query.js`, `geospatial/gpu-point-spatial-query.js` and the spatial force layout) first divides value and bounds by `max(abs(minimum), abs(maximum))` when the bounds straddle zero, then applies the same quotient.

Input: bounds `[-180, 180]`, 360 columns, x = -177. In float32 the plain form gives `3 / 360 * 360 = 3.0000002`, column 3. The rescaled form gives `(-0.98333335 + 1) / 2 * 360 = 2.9999971`, column 2. Over the 360 integer degrees of that grid, 45 land in different columns. The review found 967 of 12,407 values next to cell edges disagreeing over seven tested bounds.

So a point binned by `GPUGridBinning` and looked up through `GPUGridIndex` over the same bounds and grid size can be in two different cells. Any caller that pairs the two classes, or pairs either with its own kernel, has to know which private form it is paired with.

**Proposed change.**

- One formula for the three classes. Which one is a luma.gl decision; the plain quotient is what two of the three use and what a reader expects.
- Export it: `GPU_GRID_COORDINATE_WGSL` (the name the pre-split branch used) from `@luma.gl/graph/gpu-core`, a WGSL string defining `getCoordinate(value, minimum, maximum, size) -> u32`, documented as the cell rule of every published grid class.
- State in the class documentation that a value equal to `maximum` belongs to the last cell.

**Would retire here.** The five private copies and their five pin specs (register section 2, "Grid cell-coordinate function"): `trajectory-encounters/encounter-kernels.ts:145`, `region-statistics/region-statistics-grid.ts:172`, `spatial-interpolation/gpu-inverse-distance-weighting.ts:376`, `point-density/point-density-kernels.ts:286`, `gpu-network/flow-aggregation/flow-aggregation-kernels.ts:124`.

**Evidence.** Reading: both published forms, compared as text by the pin specs (`point-density-grid-coordinate-pin.node.spec.ts` asserts the binning form has no rescaling; `encounter-grid-coordinate-pin.node.spec.ts` asserts the index form has it). Replication: the x = -177 arithmetic above, with correctly rounded float32 division; WGSL allows 2.5 ULP on division, so the exact values that disagree on a given GPU may differ. Spec for the consequence, not for the two published classes against each other: `modules/analysis/test/gpu-spatial-analysis/point-density/gpu-point-density-cell-mapping.spec.ts` failed on the old `GPUPointDensity`, which took counts from `GPUGridBinning` and keyed weights with the `GPUGridIndex` form (count in one cell, weight in its neighbour). No spec here runs `GPUGridIndex` and `GPUGridBinning` on the same point.

## 2. Graph algorithms on a caller-built CSR with masks; one self-loop rule

**Problem.** Every class of `@luma.gl/graph/gpu-graph-algorithms` takes a `GPUGraphAlgorithmsTopology`. That class rebuilds a CSR from source edge columns (`GPUGraphAlgorithmsData`) into caller-owned physical `GPUVector<'uint32'>` outputs, each in its own allocation (`gpu-graph-algorithms/gpu-graph-topology.d.ts`). A caller that already holds a CSR in its own `GPUCommandGraph` (built by `GPUCOOToCSR`, written by an earlier kernel, or changing per frame) cannot run degree, connected components, breadth-first search, shortest paths or modularity on it. Nor can it mask nodes or slots without rebuilding the CSR.

Input: a road network whose edges are filtered per frame by a closure mask. The live degree of a node is the count of unmasked slots in its row. `GPUGraphAlgorithmsDegree` subtracts CSR offsets, which counts every stored slot.

Second problem, visible in the declarations: the published classes use two self-loop rules. `GPUGraphAlgorithmsDegree`: "an undirected self-loop contributes one" (`gpu-graph-degree.d.ts:21`). `GPUGraphAlgorithmsModularity`: undirected edges contribute to both endpoint volumes, "including twice for a self-loop" (`gpu-graph-modularity.d.ts:27`). Input: one vertex with one self-loop and one edge to a second vertex. Its degree is 2 from the first class; its volume in the second is 3. A caller who feeds published degrees into a modularity formula gets a different number from the published modularity.

**Proposed change.**

- An adjacency input type that is a set of `GraphDataView`s (`offsets`, `neighbors`, optional `edgeIds`, `edgeWeights`) on the caller's graph, accepted by each algorithm beside `GPUGraphAlgorithmsTopology`. The pre-split branch had this as `GPUGraphAlgorithmsTopologyView`.
- Optional `nodeMask` and `slotMask` (edge mask) views, or a documented dead-slot sentinel (`neighbor >= nodeCount`), honoured by degree, components, traversal and modularity. Regionalization needs the edge mask too: REDCAP and AZP label the components of a contiguity graph with some edges cut.
- Outputs as `GraphDataView`s on the caller's graph.
- One self-loop rule stated once for the entry, with degree offering it as an option if both are kept (`selfLoops: 'once' | 'twice'`).
- For modularity: a deterministic accumulation (integer atomics for unweighted graphs, a fixed-order tree for weighted ones). The published sum uses float atomics and depends on thread order.

**Would retire here.** `createMaskedWeakComponentNodes` (`analysis-core/kernels/masked-weak-components.ts:34`, five callers), the degree and modularity kernels of `GPUNetworkStatistics` (`gpu-network/network-statistics/gpu-network-statistics.ts:278`), the degree column of `GPUNetworkAnalyticsColumns` (`gpu-network/network-analysis/gpu-network-analytics-columns.ts:127`), and in time `GPUNetworkNeighborhood` (hop distances) and the CSR build of `GPUNetworkNoding`. It would also let the network columns offer the published PageRank, core number and label propagation, which they cannot today.

**Evidence.** Reading of the published declarations. The local kernels' own rules have specs (`modules/analysis/test/gpu-network/network-analysis/gpu-network-analytics-columns.spec.ts`, `gpu-terrain/terrain-decode/spike-repair-component-labels.spec.ts`); none compares a local kernel with the published algorithm, because the published one cannot be given the same input.

## 3. `GPUHaversineDistance`: state the kilometre default, seam-safe longitude difference

**Problem.** Two things.

The default `radius` is 6371 (`geospatial/gpu-haversine-distance.js:24`), so the default output is kilometres on a rounded radius. The declaration says so; the class name and the output prop do not. Input: New York (-73.98, 40.75) to Paris (2.35, 48.85). Published default: 5833.150. The same pair on the IUGG mean radius in metres: 5,833,158.6. The 8.1 m is the radius alone; the factor of 1000 is the unit.

The float32 path subtracts raw longitudes: `let deltaDegrees = rightDegrees - leftDegrees;` (`gpu-haversine-distance.js:69`). Input: a = (179.94722, 0), b = (-179.99805, 0), 6.1 km apart across the antimeridian. The raw difference is formed near 360 and wrapped afterwards, which loses 1.53e-5 degrees: 1.7 m at the equator.

**Proposed change.**

- Keep 6371 as the default, since changing it would change published results, and say "kilometres" where the output is declared. Export the constant (`GPU_HAVERSINE_DEFAULT_RADIUS_KILOMETERS = 6371`) so callers can name what they get.
- Form the longitude difference on the antimeridian side before subtracting (wrap each longitude's distance to the seam, then add), so the small separation is computed from small numbers. The local helper `geodesicLongitudeDelta` (`gpu-spatial-analysis/geometry-measures/geodesic-wgsl.ts`) does this through an integer path that a reassociating compiler cannot undo.

**Would retire here.** The `'sphere'` distance of `GPUGeodesicPairs` (`gpu-spatial-analysis/geometry-measures/gpu-geodesic-pairs.ts:107`) could delegate with `radius` from the spatial context. Bearings, midpoints, Vincenty and rhumb lines stay local.

**Evidence.** Replication: both central-angle formulas in float64 on six pairs agree to 3e-16, so the formula is the same and only the radius, the unit and the seam differ. Reading: the raw subtraction in the published kernel. Spec for the seam effect on the local kernels that had the same subtraction: `modules/analysis/test/gpu-spatial-analysis/geometry-measures/gpu-geodesic.spec.ts` (equatorial pair across the antimeridian, 1.70 m off before the fix) and `parity-wp2-geodesic-seam.parity.spec.ts`. The published class was not run on the seam pair.

## 4. `GPUFiniteDifference2D`: per-row spacing and no-data

**Problem.** `spacing` is one pair of constants (`gpu-core/gpu-finite-difference-2d.d.ts:15`) and there is no validity input or output. A raster in geographic coordinates has an east-west cell width that changes with the row. Input: a one arc-second grid. A cell is 30.9 m wide at the equator and 15.4 m at latitude 60, so one `spacing[0]` is wrong by a factor of two somewhere on a grid that spans both. A no-data cell (NaN or a sentinel) in a stencil gives a NaN or a wrong number with no flag.

A numeric point for second derivatives: the stencils combine absolute float32 samples. On elevations near 8,000 m a float32 step is about 5e-4 m, and `f(x+h) - 2 f(x) + f(x-h)` forms `2 f(x)` and a sum of magnitude 16,000 before the cancellation. Taking differences from the centre sample first, `(f(x+h) - f(x)) - (f(x) - f(x-h))`, makes both inner subtractions exact for neighbouring elevations.

**Proposed change.**

- `spacing` as either a pair of numbers or a pair where the first is a per-row `GraphDataView<'float32'>` (or a `GPURasterMetadata` from which the class derives per-row widths).
- Optional `validity` input and `outputValidity` output, with the `'propagate'` rule of the raster entry.
- Centre-relative accumulation for stencils whose weights sum to zero. The same option would help `GPURasterNeighborhood`.

**Would retire here.** The stencils of `GPUTerrainCurvature` (`gpu-terrain/terrain-curvature/gpu-terrain-curvature.ts:272`), whose border formulas already follow the published one-sided stencils.

**Evidence.** Reading only. The terrain stream of the fix pass read the published kernel and kept the local stencils for these three reasons; no parity spec was written.

## 5. `GPURasterMorphology`: radii above 8, valid-sample count

**Problem.** `radius` must be an integer from 0 to 8 (`gpu-raster/gpu-raster-morphology.js:60` throws otherwise). A larger window has to be composed from passes. That works for a full raster and fails on a raster with holes under the `'ignore'` policy, because an invalid centre stays invalid ("Missing centers always remain invalid, independently of the neighborhood nodata policy", `gpu-raster-morphology.js:12`) and cannot pass a value along.

Input: one row of 17 cells, erosion (minimum), `'ignore'`. Cell 0 holds 1, cells 1 to 15 are invalid, cell 16 holds 9. A window of radius 16 at cell 16 covers cell 0, so the answer is 1. Two passes of radius 8: after the first, cell 16 still holds 9 (its window is cells 8 to 16, all invalid but itself) and cell 8 is still invalid, so the second pass also gives 9.

The classes also do not say how many samples of a window were valid. A ground filter needs that to flag a window that reached past the covered cells.

**Proposed change.**

- Lift the radius limit, either with a van Herk/Gil-Werman pass for square footprints or by documenting and implementing the composition internally with a rule that carries values across invalid centres.
- An optional `output.validCount` (`uint32` per cell), or a completeness flag, beside the `'ignore'` result.

**Would retire here.** The square windows of `GPUPointCloudGround` (`pointcloud-analysis/gpu-pointcloud-ground.ts:76`, radii up to 32, status bit `INCOMPLETE_WINDOW`). `GPURasterRankFilter` at quantile 0 or 1 already fits the published limit and could delegate with the valid count.

**Evidence.** Reading for the limit and for the invalid-centre rule; the 17-cell example follows from that rule and was not run. Spec for the part that works today: `modules/analysis/test/gpu-pointcloud/pointcloud-diamond-morphology.spec.ts` shows the diamond windows of `GPUPointCloudSMRF`, composed from published `GPURasterErosion` and `GPURasterDilation` passes of radius at most 8, bit-identical to a full clipped-diamond scan for radii 1 to 37. That delegation was made in this pass.

## 6. `GPUHistogram`: GPU-resident domain view

**Needs confirmation upstream before anything is filed.**

**Problem.** `GPUHistogram` accepts its domain as a literal pair, as `'auto'`, or as a two-row GPU view. For float32 the bin is `u32((value - minimum) / (maximum - minimum) * f32(BIN_COUNT))` (`gpu-core/gpu-histogram.js`, `binCalculation`). With a literal domain the divisor is a shader constant. With a view it is a runtime value.

Input, as reported by the `dataframe` stream of the fix pass: 16 bins, domain `[-1e38, 1e38]`, values 0, 1, -5e37 and 5e37. With the literal domain the published class gives bins 8, 8, 4, 12, which is `(v + 1e38) / 2e38 * 16`. With the same two numbers in a GPU-resident domain view it put all four values in bin 0 on the test GPU. Reproduced in the closing pass (2026-11-11) and pinned by the spec named under Evidence. The stream's explanation is that the runtime division is compiled as a multiplication by a reciprocal, and `1 / 2e38` (5e-39) is subnormal in float32 because the range is above 2^126. A subnormal is not zero under IEEE arithmetic, so the explanation also needs the GPU to flush subnormals to zero; with both, every product is 0. Both parts are inferences from the result, not something observed in generated code.

A second point, which the closing pass measured: a value exactly on a bin edge is counted one bin too low for some bin counts and widths, by both domain forms, and not on the same values. The inputs are `n` bins of width `w` from `minimum` and one float32 value on each of the `n + 1` edges; every edge and the range are exact in float32. `numpy.histogram(values, bins=n, range=(lo, hi))` (NumPy 2.5.4, float32 input) puts each edge value in the bin above it and the maximum in the last bin, for every row. On the test GPU (Apple M3 Pro, headless Chromium, luma.gl 10.0.0-alpha.12):

| Bins | Width, minimum | Edges counted one bin low, literal domain | Edges counted one bin low, domain view |
| --- | --- | --- | --- |
| 49 | 1, 0 | none | 7, 14, 27, 28, 35, 41, 47 |
| 49 | 8, 1024 | none | 7, 14, 27, 28, 35, 41, 47 |
| 3 | 41, 0 | 1, 2 | 1, 2 |
| 6 | 41, 0 | 1, 2, 4 | 1, 2, 3, 4 |
| 12 | 41, 0 | 1, 2, 4, 7, 8 | 1, 2, 3, 4, 6, 8 |
| 7 | 83, 0 | 1, 2, 4 | 1, 2, 4 |

Both forms are right for 7, 255 and 1000 bins of width 1 from 0, 12 bins of width 0.25 from -16 and 30 bins of width 8 from 1024. The smallest reproduction: 3 bins, domain `[0, 123]`, values 0, 41, 82, 123. The counts should be 1, 1, 2 and are 2, 1, 1 with either form. For the view form alone: 49 bins, domain `[0, 49]`, values 0 to 49; the literal domain gives one per bin and two in the last, the view puts 7 in bin 6, 14 in bin 13 and so on.

In these rows the literal lists are what one multiplication by the float32 constant `BIN_COUNT / range` gives, and the view lists what a multiplication by the float32 reciprocal of the range followed by one by `BIN_COUNT` gives (replicated in NumPy float32). That is read off the counts, not off generated code, and it does not fit every other case tried. The expression as written, with a correctly rounded division and multiplication (NumPy float32), is wrong on yet another set: edge 27 of 49 bins of width 1, and edges 251, 253, 502, 506 and 511 of 1000 bins of width 1, a case the test GPU gets right in both forms. Each of the three evaluations is wrong on some exact edge, so the cause is the formula and not one adapter's rounding. A product test, `k * range <= (value - minimum) * BIN_COUNT < (k + 1) * range`, uses only multiplications and comparisons and is exact whenever the two products are. It was not run against these cases here.

**Proposed change.**

- Decide the bin with the product test, or scale `value - minimum` and `range` by a power of two before dividing when `range` is above 2^126.
- A test in luma.gl with a domain view near the float32 range, and one with a value on every edge at bin counts that are not powers of two, in both domain forms (the rows above).
- Separately (a feature, not a fix): a domain given as two rows at an offset inside a larger view, with NaN meaning "the finite extent", and an output at an offset of a larger view, so several histograms share one pair of views.

**Would retire here.** The histogram kernel of `GPUColumnProfile` (`BIN_WGSL` in `dataframe-analysis/column-profile/column-profile-kernels.ts`), which is documented to use the product test and the published bin convention. The spec below does not run it.

**Evidence.** Committed spec (closing pass, 2026-11-11): `modules/analysis/test/gpu-core/gpu-histogram-domain-view.spec.ts`, 12 tests, runs the published class through `@luma.gl/graph/gpu-core` twice per case on one device, with the domain as a literal and as a GPU-resident view. It asserts the NumPy counts where the class is right, and the observed counts where it is not (the table above and the all-in-bin-0 result of the wide domain), so the suite passes; an upgrade that fixes a case fails its assertion, which must then be flipped. The comment in `modules/analysis/test/gpu-dataframe/column-profile/gpu-column-profile.spec.ts:573-575` is confirmed by it. The expected counts need no fixture: one per bin and two in the last, which `numpy.histogram` (NumPy 2.5.4) returns for every case. All of this is from one adapter. The same spec, run on a second adapter, is the confirmation this section asks for, and its body is the test to offer luma.gl. Not done: two domains that bracket 2^126, a narrow domain far from zero, and values at bin centres.

## 7. `GPUKMeans`: k-means++ seeding and a tolerance stop

**Problem.** The only seeding is `'evenly-spaced'` (`gpu-vector-search/gpu-k-means.js:42-55`): seeds are the valid rows nearest evenly spaced positions in the source order. The stop rule is "no label changed"; an empty cluster keeps its centroid.

Input: a table sorted by a column that correlates with the clusters, which is the normal state of a table sorted by region or time. 80 rows of blob A, then 10 of blob B and 10 of blob C, `clusterCount: 3`. The target rows are `floor(index * rowCount / clusterCount)` (`gpu-k-means.js`, seed selection pass): rows 0, 33 and 66, all three in blob A. Lloyd iterations then start with A split three ways and B and C unseeded. Whether they recover depends on the geometry. This was not run; the point is that the result depends on row order, and that there is no seeded alternative.

**Proposed change.**

- `seed: 'evenly-spaced' | 'k-means++'` with a `seedValue: number` for the second. k-means++ needs one distance pass and one weighted draw per centroid, all GPU-resident with a counter-based generator.
- `tolerance?: number`: stop when the largest centroid shift is below it, reported in `status`.
- Optional: `restarts` with an inertia output (so the best restart can be chosen on the GPU), a per-row squared distance output, and relocation of an empty cluster to the row farthest from its centroid.

**Would retire here.** The local `GPUKMeans` for 2D points (`gpu-spatial-analysis/spatial-clustering/gpu-kmeans.ts`), which has the same name as the published class. Renaming it is on the local roadmap whatever happens upstream.

**Evidence.** Reading of both classes. No numeric comparison of the two on one input was made in the review or in the fix pass.

## 8. `GPURasterContours`: several levels per pass, world coordinates

**Problem.** One instance extracts one `level` (`gpu-raster/gpu-raster-contours.d.ts:26`), and vertex positions are local pixel-centre coordinates, with the affine applied by the caller (`:71-72`).

Input: ten contour levels on a 4,096 by 4,096 elevation raster. That is ten instances, each with its own passes over 16.8 million cells, and ten segment buffers whose capacities the caller has to guess separately. A caller who wants map coordinates then adds a pass per buffer, and has to subtract a local origin first to keep float32 precision.

**Proposed change.**

- `levels: number[] | GraphDataView<'float32'>` with one segment buffer and a per-segment level index (or per-level offsets), and one `requiredSegmentCount`.
- `coordinates: 'pixel' | 'world'`, the second applying `metadata` relative to a stated local origin.
- Later, as separate work: global edge IDs so segments can be stitched into polylines, and a band (two-level) mode.
- A small fix found while converging on the published saddle rule: the kernel divides the four corners and the level by the largest magnitude before forming the saddle determinant (`gpu-raster/gpu-raster-contours.js:441-447`). On an integer raster an exact tie, which the class documents as "connects the high-valued diagonal", can then be lost to the rounding of the divisions. Take the differences unscaled and scale only when the products could overflow (the local decider scales above a magnitude of 1e18). Found by a local spec on a random integer field while the local kernel used the scaled form; the published class was not run on that cell.

**Would retire here.** `GPUIsolines` (`raster-analysis/isolines/gpu-isolines.ts`) for the segment output. `GPUIsobands` needs bands. Both local classes took the published saddle rule (the bilinear decider) in this pass, so they already agree with `GPURasterContours` cell by cell.

**Evidence.** Reading of the declaration. `GPUTerrainContours` already delegates to the published class for one level.

## 9. `GPUHashJoin` and `GPUGroupAggregation`: 64-bit and sparse keys

**Problem.** `GPUHashJoin` takes packed `uint32` keys (`gpu-core/gpu-hash-join.d.ts:10`). `GPUGroupAggregation` takes dense `uint32` keys below `output.length` (`gpu-core/gpu-group-aggregation.d.ts:16`, `:40`).

Input: an H3 cell index such as `0x8928308280fffff`, a Quadbin key, or an Arrow `Int64` identifier. None fits a `uint32`. Two group keys 7 and 4,000,000,000 would need an output of four billion rows. Geospatial tables are keyed this way as a rule.

Related: the float sums of `GPUGroupAggregation` are compare-exchange sums whose result depends on thread order (see section 13), and there is no row mask on the join.

**Proposed change.**

- Two-word keys (`uint32x2`, low word first in storage) in `GPUHashIndex`, `GPUHashJoin` and `GPUGroupAggregation`.
- A sparse mode for `GPUGroupAggregation`: sort or hash the keys, publish a compact table of `(key, statistic)` with a count and an overflow flag.
- A row mask on the left side of the join; 1:n aggregates over repeated right keys.

**Would retire here.** `GPUKeyJoin` (`dataframe-analysis/key-join/gpu-key-join.ts:160`), `GPUGroupStatistics` for count, sum, mean, minimum and maximum (`dataframe-analysis/group-statistics/gpu-group-statistics.ts:277`; order statistics and moments would stay), and the table build of `GPUCellAggregation` (`gpu-spatial-analysis/cell-aggregation/gpu-cell-aggregation.ts:119`).

**Evidence.** Reading of the declarations. The local classes state the reason in their class documentation.

## 10. Exact point-in-ring predicate where `GPUPairwisePointInPolygon` returns `uncertain`

**Problem.** The classifier returns `uncertain` (code 3) when its double-single arithmetic envelope is exceeded or the geometry is degenerate, "rather than a silent inside/outside classification" (`geospatial/gpu-pairwise-point-in-polygon.d.ts:46`). That is honest, and it leaves the caller with rows it cannot assign.

Input: a polygon whose second ring has two vertices, `[(11, 1), (12, 2)]`, inside the square `(10, 0)-(14, 4)`. The ring bounds no area. The published classifier reports the row as `uncertain`. An even/odd count with exact orientation signs skips the ring and answers.

**Proposed change.**

- When the fast filter cannot decide the sign of an orientation, fall back to an exact one in integer arithmetic. The local `EXACT_ORIENTATION_WGSL` (`analysis-core/kernels/exact-orientation-wgsl.ts`) does it: the determinant is six products of raw float32 coordinates, each exact in 48 bits, aligned to the smallest exponent and added into two wide unsigned accumulators that are compared. It uses integers because a float two-sum does not survive the compiler (section 13).
- Keep `uncertain` only for malformed offsets.
- A `boundaryRule: 'closed' | 'open' | 'half-open'` option. Zonal statistics need the half-open rule so that a point on a shared edge belongs to one polygon.

**Would retire here.** The inline classifier of `GPUPointInPolygonJoin` (`gpu-spatial-analysis/spatial-join/gpu-point-in-polygon-join.ts:656`) and the fallback path of `GPUSpatialPredicateJoin`, together with candidate pairs from an index (a separate request).

**Evidence.** Spec: `modules/analysis/test/gpu-spatial-analysis/spatial-join/gpu-point-in-polygon-classifier-agreement.spec.ts` runs the published classifier (inside `GPUSpatialPredicateJoin`) and the local one on four scenes. They agree on shared edges and vertices, at projected-metre coordinates and one float32 step either side of an oblique edge; the degenerate-ring scene is where the published one reports `uncertain`.

## 11. A 64-bit word shader module; the H3 face tables as data

**Problem.** WGSL has no 64-bit integer. `@luma.gl/shaders` has 64-bit word helpers in two places, `dggs_u64_*` inside the `dggs` module and `fp64_u64_*` inside `fp64arithmetic`, both `vec2u(high, low)`, neither with multiply or divide, and neither importable without the rest of its module. Kernels that need a 64-bit accumulator, a 64-bit key compare or a division of epoch milliseconds write their own.

Separately, the H3 face centres and axis azimuths exist only as literals inside the WGSL functions `dggs_h3_get_face_center_geo` and `dggs_h3_get_face_axis_azimuths`. A caller that needs them in float64 (to build double-float constants) has to parse the shader text. Input: with a function name that no longer exists, a text search returns 0 rows and the dependent WGSL declares an empty array.

**Proposed change.**

- A small standalone shader module, for example `uint64`, with `add`, `subtract`, `negate`, `compare`, `shiftLeft`, `shiftRight`, `multiply32` (64 by 32), `divide32` and conversion to float32, in the `(high, low)` order the two existing sets use, plus the storage-order converters `dggs` already has. `dggs` and `fp64arithmetic` could then depend on it.
- Export the H3 tables as data from `@luma.gl/shaders` (two frozen arrays of 20 rows), and generate the WGSL literals from them.

**Would retire here.** `UINT64_WGSL` (`analysis-core/kernels/uint64-wgsl.ts:36`), the 64-bit helpers of `CELL_KEY_WGSL`, the three private copies in the terrain summed-area table, Ripley's K and the calendar buckets, and `parseDggsFaceTable` (`gpu-spatial-analysis/cell-indexing/h3-index-extended-wgsl.ts:25`).

**Evidence.** Spec for the local side: `modules/analysis/test/utils/wgsl-words-search.spec.ts` (64-bit words against BigInt), `modules/analysis/test/gpu-spatial-analysis/cell-indexing/h3-index-extended.node.spec.ts` (the parser throws when a table does not have 20 rows). Reading for the published side.

## 12. One documented meaning of `count`

**Problem.** Capacity-bounded outputs report a count, and the published classes do not agree on what it counts.

| Class | `count` holds | Unclamped value |
| --- | --- | --- |
| `GPUHashJoin` | required matches before truncation (`gpu-core/gpu-hash-join.d.ts:19`) | `count` itself; `overflow` flags it |
| `GPUBVHQuery` | the full match count (`gpu-core/gpu-bvh-query.d.ts:27`) | `count` itself |
| `GPUPointSpatialQuery` | clamped to `output.ids.length`, "safe to consume as an indirect draw count" (`geospatial/gpu-point-spatial-query.d.ts:76-77`) | `totalCount` |
| `GPURasterContours` | `segmentCount = min(required, capacity)` (`gpu-raster/gpu-raster-contours.d.ts:29`) | `requiredSegmentCount` |

Input: a join with 1,000 matches into an output of capacity 600. A kernel that loops `for (i < count)` over the output reads 400 rows past the published data with `GPUHashJoin` and is correct with `GPUPointSpatialQuery`.

**Proposed change.** One rule for the package: `count` is always clamped to capacity and safe to loop or draw with; `requiredCount` is the unclamped number; `overflow` is `requiredCount > capacity`. Add `requiredCount` to `GPUHashJoin`, `GPUBVHQuery` and `GPUGridIndexQuery` first, then change `count` in a release that says so.

**Would retire here.** Nothing today. The local `GPUCompactOutput` already follows the proposed rule (`count` clamped, `status.requiredCount` unclamped), and no local wrapper around the unclamped classes exists. It removes a trap for the next one.

**Evidence.** Reading of the four declarations.

## 13. Float compensated sums are removed by the Metal shader compiler, and three related facts

This section is four facts about what the shader compiler of the test GPU does to WGSL that is correct as written. They concern luma.gl in two ways: its own kernels are compiled by the same toolchain, and its WGSL authoring guidance is where a kernel author would expect to be warned. **The cause inside the compiler was not found for any of them.** The generated Metal source was not inspected, only one adapter was tried (headless Chromium WebGPU on an Apple GPU), and nothing says whether the browser's shader translation (Tint) or the Metal compiler is responsible. The closing pass (2026-11-11) did inspect it (`dev-docs/research/math-correctness-review-2026-11-10/shader-compiler-findings.md`): 13.1, 13.2 and 13.4 are Metal fast math (the Metal source Tint generates gives the IEEE result when compiled with fast math off, and Chromium 151 has no switch), and 13.3 is a miscompile in the Apple GPU back end of correct Metal source, independent of fast math, reduced to plain WebGPU with a drafted, unfiled report. The second and fourth are within what WGSL permits (it does not fix the association of floating-point expressions or the rounding of division). The third, a table that reads back as zeros, is not permitted by any reading of the language and is most likely a browser or driver defect: it should be reduced further and reported there, not to luma.gl. What luma.gl can do is know the shapes and test for them.

### 13.1 Compensation terms are removed

**Finding.** A float compensation term, Kahan's `(next - sum) - term`, Neumaier's `(sum - next) + term` or the float two-sum, is algebraically zero. The shader compiler of the test GPU (Metal, through Chromium's WebGPU) reassociates floating-point expressions and removes it. A kernel written as a compensated sum then computes the plain float32 sum, bit for bit, and a CPU twin of the same kernel does not show it.

Input: a million terms with mean 1234.567, a quarter of them negative. True sum 1,234,041,587.7. In one kernel on the test GPU: the float form `(sum - next) + term` leaves a correction of exactly 0 and a total 16,115 away from the true sum. The rounding error of each addition read off the integer significands of the three float32 values gives a total 0.44 away before the final rounding. A fixed-point accumulator gives the nearest float32. A second fact from the same run: the plain float32 running sum of the kernel (1,234,025,472) is not the IEEE serial sum of the same terms in the same order (1,234,033,152), so a CPU twin cannot expect a GPU loop sum bit for bit either.

**Why luma.gl should care.** No published `@luma.gl/graph` kernel of alpha.10 contains a Kahan, Neumaier or two-sum term (searched for the words in `dist`), so nothing published is wrong today. The workarounds register has asked for compensated summation in `GPUReduction` since the split. If it is added as a float expression it will pass its CPU oracle and do nothing on Metal. The `fp64arithmetic` shader module already knows the problem: its comment says a float barrier such as `x + ONE * 0.0` "folds away" and it guards intermediates with an integer OR of runtime-zero bits (`LUMA_FP64_CODE_ELIMINATION_WORKAROUND`, `@luma.gl/shaders` `fp64-arithmetic-wgsl.js`). The local double-float code of `GPUPointToCell` uses the same guard and survives (`h3-index-extended.spec.ts`: 0 mismatches against h3-js where the float32 path misses more than 0.5%).

**Proposed change.**

- State the rule in the luma.gl contributor documentation: an error-free transformation in WGSL must not be an expression a reassociating compiler can prove zero.
- For sums in `GPUReduction`, `GPUScan`, `GPUGroupAggregation` and any segmented reduction: fixed point in integer words where the range is known, otherwise the per-addition rounding error from the integer significands, or the guarded two-sum of `fp64arithmetic`.
- Test each on the GPU at a size where the plain float32 sum is visibly wrong (a million equal terms), never only against a CPU twin.
- A public way to get `fp64arithmetic` with the right defines for a device, so callers do not carry their own two-sum.

**Would retire here.** Parts of `analysis-core/kernels/addition-rounding-error-wgsl.ts` and `fixed-point-accumulator.ts` wherever a published reduction or scan would then be exact enough, and the register row "Exact or compensated sums in published reductions and scans".

**Evidence.** Spec: `modules/analysis/test/utils/addition-rounding-error-wgsl.spec.ts` pins the three totals above in one kernel on the GPU, with 4,022 pairs of both signs checked bit for bit against the exact float64 error. `analysis-core/kernels/addition-rounding-error-wgsl.ts` is the integer form. Further specs from the same pass: `gpu-linear-referencing.spec.ts` (a prefix of 86,400 equal segments was 16.43 off at 1.2 million, the plain float32 sum bit for bit), `gpu-geometry-measures-summation.spec.ts`, `shape-descriptors-moments-summation.spec.ts`. One adapter only.

### 13.2 A chain `(value - shift) - small terms` is regrouped

**Finding.** A residual such as `(y - mean) - intercept - slope * (x - xmean)` is accurate in float32 only when `y - mean` is formed first (it is exact when the two are close) and the small terms are subtracted from that difference. The compiler regroups the chain as `y - (mean + intercept + ...)`, so the small terms are absorbed into the large one before the subtraction. Writing the first difference on its own line does not help: the regrouping crosses statements. A correction subtracted in the same expression as the value it corrects is lost the same way.

Inputs, both measured on the test GPU:

- Regression residuals of a response near 10,000 with residuals of about 0.002: every residual came out as a multiple of the float32 step of 10,000, the sum of squares was wrong and a Chow F statistic was 12% off; the first-stage `s2` of a spatial error model was 5% off.
- A group z-score of the four values 1e7, 1e7 + 1, 1e7 + 2, 1e7 + 3. The mean, 1e7 + 1.5, is not a float32, so the kernel stores the rounded mean and a correction of -0.5. The z-score of the first value came out as -1.549 for a true -1.162: the correction had been folded into the mean and rounded away.

**Cure used here.** The bits of the first difference pass through an integer operation with a zero the compiler cannot know (an exclusive-or with a `u32` read from a buffer whose value is always 0; a literal would be folded). The value is unchanged and the subtraction has to be carried out as written: `getRoundedDifference(minuend, subtrahend, runtimeZero)` in `gpu-spatial-analysis/spatial-regression/rounded-difference-wgsl.ts`. It is the same device as the runtime-zero guard of `fp64arithmetic`.

**Why luma.gl should care.** Any published kernel that centres a value and then subtracts small terms in one chain is exposed: z-scores and variances about a mean, residuals, coordinates relative to a local origin followed by a small offset. Published kernels were not searched for the shape and none was run for it. A public WGSL helper for "this difference, rounded once" beside the `fp64arithmetic` guard would give every author the same cure.

**Evidence.** Spec: `modules/analysis/test/gpu-spatial-analysis/spatial-regression/math-review-restricted-regimes.spec.ts` (the case "response mean far above its residuals", which failed before the cure), `math-review-error-ml-high-fit.spec.ts` (the case "response mean of 10000", which failed before), and `modules/analysis/test/gpu-dataframe/group-statistics/gpu-group-statistics.spec.ts` (the 1e7 case, which fails without the bit operation). Not every such chain in this library was tested: the lag-ML, GM and 2SLS residual passes were not, and one residual pass that was tested (`GPUOrdinaryLeastSquares`) is not regrouped, so the compiler's choice depends on the kernel.

### 13.3 A local integer table filled only by constant-count loops reads back as zeros

**Finding.** A local `var table: array<u32, N>` or `array<i32, N>` whose only stores are those of loops with a constant trip count, read at an index the compiler cannot resolve (an index or a loop bound from a buffer, or a loop nest it does not fold), reads back as zeros. Local means function scope and also `var<private>` at module scope: Tint emits a private variable as a zero-initialised local of the entry point. Any stored value is lost: the index, a constant, a sentinel, values copied from a buffer. In plain WebGPU with nothing else in the shader the table is lost at 33 to about 250 entries for a loop of 3 to 96 iterations, and is right up to 32 entries. Inside a larger kernel (Louvain, as it was before this review) the table was lost from 12 entries up, and from 33 again once a helper that declares two float arrays was no longer called, so the sizes depend on what else the kernel holds.

The Metal source Tint generates is correct and so is the intermediate code of the Metal front end. The Apple GPU back end loses the stores when it unrolls the fill loop after the zero initialisation (`#pragma clang loop unroll(disable)` on the loop cures it in Metal). Fast math plays no part.

Right at every size tried, by mechanism (no loop is unrolled into the table): a literal initializer, a module-scope `const`, a fill loop whose bound is a run-time value. Right at every size tried for a reason that is not known: `f32` and vector entries, a table that later receives a store at a data-dependent index, a fill loop that also writes storage or stores under a data condition.

**Cure used here.** A table whose contents are known when the shader is built gets a literal initializer or is a module-scope `const`; otherwise its fill loop is bounded by a value the compiler cannot know (in a kernel without a uniform, `N * min(arrayLength(&binding), 1u)`); a table whose value is its index is removed. `var<private>` with a fill loop is not a cure: it read right inside the Louvain kernel, which is why it was first taken for one, and it is lost in the reduced shader. This is a rule in the local contributor contract (section 11). An audit of the local kernels (2026-11-11) ran every kernel that holds such a table on inputs with a known answer and found none that loses it today (their fill loops also write storage, store under a data condition, or are followed by stores at data-dependent indices). A table that reads right at the sizes tried is evidence for those compiled kernels only.

**Why luma.gl should care.** A kernel that builds a permutation, a visit order, a set of sentinels or a small lookup table in a local integer array by a counted loop has the shape. A text search of the published kernels for an identity write (`table[i] = i`) finds three, in `gpu-physics/sphere-world-shaders.js`, `gpu-core/gpu-text-selection.js` and `gpu-core/gpu-sort-chunks.js`, and all three write a storage binding, not a local array, so no published kernel is known to have the shape. The search was by text, for the identity fill only, and nothing was run. The useful thing for luma.gl is the authoring rule and the reproduction; the report belongs with the browser and with Apple, and both are drafted (not filed) in the findings report.

**Evidence.** `dev-docs/research/math-correctness-review-2026-11-10/shader-compiler-findings.md` (the reduction, the generated Metal source, the size tables, the drafted reports, and the library audit). Specs: `modules/analysis/test/analysis-core/shader-compiler-table-fill.spec.ts` (plain WebGPU), `shader-compiler-table-shapes.spec.ts` (the table shapes local kernels use, and the old Louvain kernel with its cures), `modules/analysis/test/analysis-core/table-audit/*.spec.ts` (the audit), `modules/analysis/test/utils/function-scope-table-readback.spec.ts` (a kernel of one binding, and three local kernels: the dense LU pivot table, natural neighbour and the phenology metrics) and `modules/analysis/test/gpu-network/network-optimization/gpu-network-louvain-visit-order.spec.ts` (13 sizes from 2 to 512 by 5 seeds against a TypeScript Fisher-Yates). One machine, one operating-system version, one browser version.

### 13.4 Float division is not correctly rounded

**Finding.** `6.0 / 3.0` came back as `1.9999999` in a region mean on the test GPU. WGSL allows 2.5 ULP of error on division, so this is permitted; it is still not what a CPU twin computes. Section 6 has a larger case of what may be the same cause (a division by a runtime value near the top of the float32 range giving 0).

**Why luma.gl should care.** Any kernel that decides a cell, a bin or a class with a quotient gives adapter-dependent results for values within a few units in the last place of a boundary: `getCoordinate` of the grid classes (section 1) and the bin of `GPUHistogram` (section 6). A test with products and comparisons, `k * range <= (value - minimum) * count < (k + 1) * range`, is the same on every adapter. And a published test that asserts an exact quotient against a CPU computation will fail on some adapter.

**Evidence.** Observed and recorded, not pinned: no committed spec asserts the inexact quotient. `modules/analysis/test/gpu-spatial-analysis/spatial-regionalization/gpu-region-partition-evaluation-nonfinite.spec.ts:89` compares to a relative 1e-6 for this reason and says so. The observation is in the Resolution of `point-process-clustering.md`.

### What to ask of luma.gl for 13.1 to 13.4

- One page of WGSL authoring guidance with the four shapes, each with its cure, and the statement that a CPU twin proves nothing about them.
- The existing runtime-zero guard of `fp64arithmetic` made available as a small public helper (a rounded difference, a kept intermediate), so authors stop inventing their own.
- GPU tests at sizes where the fault is visible, for any published kernel that sums, centres or builds an index table.
- Nothing in luma.gl is asked to work around the table fault beyond the authoring rule. It should go to the browser's issue tracker with a reproduction that does not use luma.gl.

## 14. Smaller gaps

Each of these came from one contributor that could otherwise delegate. All are readings of the published declarations.

| Published API | Request | Local code it would retire or simplify |
| --- | --- | --- |
| `GPUGridIndex` | A cell-size mode (`origin + k * cellSize`), a half-open domain, ascending point order inside a cell, a Z-ignoring raster mode, named limits, offsets and sorted IDs as plain read views | `createPointCloudGridIndex` (`pointcloud-analysis/pointcloud-grid-index.ts:85`) |
| `GPUGridBinning` | A per-frame active grid size | Grid zones of `GPUFlowAggregation` |
| `GPUGridBinning`, `GPUGridAggregation` | A per-row cell key output, or a deterministic (fixed-point) sum mode in place of the float compare-exchange sum | Grid keys kernel of `GPUPointDensity` |
| `GPURasterRegionMeasurements` | A deterministic exact sum per region and a per-row cell area | Area, mean and extrema of `GPUTerrainCatchmentSummary` |
| `GPURasterDilation`, `GPURasterErosion` | A mask input and a GPU-gated fixpoint loop (morphological reconstruction) | `GPURasterReconstruction` |
| `GPURasterContrast`, `GPURasterGlobalPercentile` | A statistics window and region mask, a per-frame mode, output in `[0, 1]`; a caller histogram and a per-frame percentile | Equalization and percentile bounds of `GPURasterStretch` |
| `GPUFiniteDifference2D` (see section 4) | The same per-row spacing and no-data rule | Gradient of `GPUTerrainFlowField` |
| `GPUGallopingSearch` | A one-dispatch lower bound over `0..segmentCount`, 64-bit keys | `createSortedSegmentOffsetsNode` |
| `GPUDGGSCellProjection` | An option for the value written to an invalid cell; boundary and vertex-count outputs | `GPUCellGeometry` |
| `GPURasterGradient`, `GPURasterNeighborhood` | A one-sided or extrapolating border mode | The padded band of `GPUTerrainDerivatives` and `GPUReliefShading` |
| `GPURasterBandMath` | The operation chosen per frame; unary operations, power, minimum, maximum | `GPURasterArithmetic` |
| `GPURasterNeighborhood` | Minimum, maximum, range, standard deviation and valid count per window | `GPUFocalStatistics` |
| `GPUCompaction` | A bounded output capacity with a required count | Scan and scatter of `GPUNetworkSubgraphFilter` |
| `GPUReduction` | Two-word (64-bit) integers, beside the existing `GPUScanUint64` | Two-level prefix of the network K-function edge draw |
| `GPUPointCloudVoxelDownsample` | Optional per-voxel centroids and attribute means | Input of `GPUPointCloudVoxelMerge` |
| `compileProjectionPlan` | A `destinationOrigin` option, so the error budget is validated for the origin the rows are relative to | The override in `GPULocalFrameProjection` |
| `GPUGraphAlgorithmsSingleSourceShortestPath` | Several sources with start costs, per-frame views, a compact frontier | `GPUNetworkReachability` |
| `GPUGraphAlgorithmsModularityOptimization` | Multilevel coarsening | `GPUNetworkLouvain` (32 vertices) |
| `random` shader module | A counter-based generator (Philox) with an open-interval unit float | `analysis-core/kernels/philox-random.ts` |
| `GPUCommandGraph` | A per-node shader description on `preflight.nodes[]` | The per-graph kernel record table (see [upstream-first-recording.md](./upstream-first-recording.md)) |

## Suggested order

1. Sections 1, 12 and 13 first: they are about what published numbers mean, and each is small. For 13.3, report to the browser first.
2. Section 6 after it is confirmed on a second adapter.
3. Section 2: the largest retirement here (six local kernels) and a known shape (the pre-split branch had it).
4. Sections 3, 5, 10, 11: each removes one local kernel or copy with a contained change.
5. Sections 4, 7, 8, 9 and the table in 14 as features.
