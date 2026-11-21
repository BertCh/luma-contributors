# GPU spatial analysis cross-reference

:::caution Experimental
The contributors on this page live in experimental entry points. APIs may change between releases
without a deprecation period.
:::

This page maps tool names from turf, PostGIS, GeoPandas and Shapely, PySAL (esda, libpysal, spreg,
pointpats, giddy, segregation, tobler, mapclassify, spaghetti and friends), QGIS, ArcGIS and CARTO to
the luma.gl analysis contributor that does the same job. It is organised by task. Search the page for
the name you already know.

## How to read the tables

- Contributors are classes added to a caller-owned `GPUCommandGraph` with `graph.add(contributor)`.
  Recipes are `addXRecipe(graph, props)` builder functions that chain several contributors. Each
  contributor name links to its section in the reference page.
- Entry points: no suffix means `@luma-contributors/analysis/gpu-spatial-analysis`. `(network)` is
  `@luma-contributors/analysis/gpu-network`, `(terrain)` is
  `@luma-contributors/analysis/gpu-terrain` and `(dataframe)` is
  `@luma-contributors/analysis/dataframe-analysis`. `(raster)` is
  `@luma-contributors/analysis/raster-analysis`, except the published luma.gl raster operators
  (band math, filters, morphology, gradients, thresholds, contours, histogram and statistics, such
  as `GPURasterNDVI`, `GPURasterBandMath`, `GPURasterSobel`, `GPURasterConnectedComponents` and
  `GPURasterRegionMeasurements`), which are imported from `@luma.gl/graph/gpu-raster`.
- GPU contributors use caller-sized outputs without implicit readback. Variable-cardinality
  outputs expose counts or capacity diagnostics where specified; fixed-size outputs may instead
  use per-row codes, validity flags or no status port. Consult each output contract. CPU library
  return shapes also vary. Where a result is raster-approximate, capacity-bounded or follows a
  specific tie rule, the Notes column says so.
- Partial matches name the exact supported subset and the external tool to use for the remainder.
- A dash means the tool has no equivalent in that column's family.
- CARTO names are Analytics Toolbox functions.

## Joins and predicates

| turf | PostGIS | GeoPandas / Shapely | PySAL | QGIS / ArcGIS / CARTO | luma.gl | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| `tag` (one zone per point) | Exclusive coverage assignment | One-zone assignment | - | Assign points to coverage zones | [`GPUPointInPolygonJoin`](/docs/api/gpu-spatial-analysis/operators-joins#gpupointinpolygonjoin-and-gpunearestfeaturejoin) | Exact orientation fallback for near-boundary points. Each point gets the containing feature with the smallest row. Capacity-bounded candidates and matches; `output.status.overflow` aggregates BVH leaf, candidate and final-match capacity loss; optional `output.status.candidateOverflow` isolates intermediate loss; `output.matches.status.overflow` diagnoses final compact storage when `matches` is supplied. Optional `output.matches.status.requiredCount` counts retained assignments before final compaction, so it cannot recover dropped candidates. |
| `booleanPointInPolygon(point, polygon, {ignoreBoundary: true})` + enumeration | `ST_Within` + `GROUP BY` | `sjoin(predicate='within')` then `groupby().agg` | - | Count every strict point–polygon match | [`GPUSpatialPredicateJoin`](/docs/api/gpu-spatial-analysis/operators-joins#gpuspatialpredicatejoin), [`GPUGroupStatistics`](/docs/api/gpu-dataframe-analysis#gpugroupstatistics) (dataframe) | Enumerate every strict interior pair with points on the left and polygons on the right; group the readable pair prefix by polygon `rightRows`. See the overlap example below. |
| `booleanPointInPolygon` (default), `pointsWithinPolygon` | `ST_CoveredBy` (point side) | `sjoin(predicate='covered_by')` | - | Boundary-inclusive point–polygon matches | [`GPUSpatialPredicateJoin`](/docs/api/gpu-spatial-analysis/operators-joins#gpuspatialpredicatejoin) | Use `predicate: 'coveredBy'` with points on the left. Includes outer and hole boundaries; returns every matching pair. Turf `booleanPointInPolygon` takes point and polygon arguments; `{ignoreBoundary: true}` above is its options argument. |
| `booleanIntersects`, `booleanContains`, `booleanWithin`, `booleanOverlap`, `booleanCrosses`, `booleanTouches` | `ST_Intersects`, `ST_Contains`, `ST_Within`, `ST_Covers`, `ST_Touches`, `ST_Crosses`, `ST_Overlaps`, `ST_Equals`, `ST_Relate` | `sjoin(predicate=...)`, `shapely.relate` | - | Extract by location, Select by location, Join attributes by location | [`GPUSpatialPredicateJoin`](/docs/api/gpu-spatial-analysis/operators-joins#gpuspatialpredicatejoin) | All nine point, line and polygon combinations. OGC boundary rules and a DE-9IM `relate` pattern. Planar coordinates. The `dwithin` distance and the `relate` pattern can be per-frame views. |
| - | `&&` (bounding-box overlap) | `STRtree.query` | - | Spatial index | [`GPUSpatialJoinCandidates`](/docs/api/gpu-spatial-analysis/operators-joins#gpuspatialjoincandidates) | The bounding-box stage alone, without refinement. |
| - | prepared geometry | `shapely.prepare` | - | - | [`GPUSpatialJoinPrepared`](/docs/api/gpu-spatial-analysis/operators-joins#gpuspatialjoinprepared) | Reuse one prepared right-hand side across many left-hand sets. |
| `tag` + aggregate | Exclusive zone assignment + `GROUP BY` | One-zone assignment then aggregation | - | Coverage-zone summary | [`GPUZonalStatistics`](/docs/api/gpu-spatial-analysis/operators-statistics#gpuzonalstatistics), [`GPUGroupStatistics`](/docs/api/gpu-dataframe-analysis#gpugroupstatistics) (dataframe); recipe `addPointsInPolygonsChoroplethRecipe` | Counts, sums, means, unbiased variances and extents per polygon with a deterministic sum order (densities need spherical `areas` in longitude/latitude). For the recipe, exclusive assignment applies to single-point layout without `featureOffsets`; multipart features instead contribute once to each polygon matched by any part. The recipe adds class breaks and colours. |
| - | `ST_DWithin` | `sjoin(predicate='dwithin')`, `buffer` then `sjoin` | `DistanceBand` | Select within distance | [`GPUBufferSelection`](/docs/api/gpu-spatial-analysis/operators-joins#gpubufferselection), [`GPUSpatialPredicateJoin`](/docs/api/gpu-spatial-analysis/operators-joins#gpuspatialpredicatejoin) (`'dwithin'`) | The distance is an inclusive per-frame planar distance, not a materialised buffer polygon. |
| - | `ST_Within` per pick region | - | - | Select by rectangle, circle or polygon | [`GPURegionMask`](/docs/api/gpu-spatial-analysis/operators-statistics#gpuregionstatistics), [`GPUPickRegionMask`](/docs/api/gpu-spatial-analysis/operators-statistics#gpuregionstatistics), [`GPURegionStatistics`](/docs/api/gpu-spatial-analysis/operators-statistics#gpuregionstatistics) | Interactive rectangle, radius or polygon selection with per-frame bounds. Rectangle and polygon accept longitude/latitude (straight edges in degrees); radius is planar only. |
| `intersect`, `lineIntersect`, `kinks` | `ST_Intersection` (crossing points), `ST_IsSimple` | `shapely.intersection` (points only) | - | Line intersections | [`GPUSegmentIntersection`](/docs/api/gpu-spatial-analysis/operators-validity#gpusegmentintersection) | Crossing points and kinds between segments, not clipped geometry. |

### All matches or one coverage zone?

Take polygon rows 0 and 1 as the rectangles `[0,0]–[2,2]` and `[1,0]–[3,2]`, and one
point at `(1.5,1)`. `GPUSpatialPredicateJoin` with `predicate: 'within'` emits pairs `(0,0)`
and `(0,1)` (point row, polygon row). Grouping by `rightRows` gives counts `[1,1]`.
`GPUPointInPolygonJoin` assigns the smallest containing polygon row, so it and
`GPUZonalStatistics` give counts `[1,0]` for the single-point layout without `featureOffsets`.
The choropleth recipe uses exclusive assignment only for that layout, in both backends.
Multipart features contribute once to each polygon matched by any part, using the configured
boundary rule. Supplying `featureOffsets: [0,1]` for this same one-point feature therefore gives
`[1,1]` in both backends, even though it has just one part.
For a point at `(0,1)` on the outer left boundary, strict `within` gives `[0,0]`, while the
assignment default `boundaryRule: 'half-open'` gives `[1,0]`: left and bottom edges are included,
right and top edges excluded. `boundaryRule: 'open'` excludes every boundary point; `'closed'`
includes touching boundaries before choosing the smallest row.

Read only `output.pairs.status.count` pairs. Before calling a count complete, require final
`overflow` and `candidateOverflow` on the pair status to be zero, or check the aggregate
`output.status.overflow`; also require `output.status.uncertainCount === 0` to certify the strict
predicate classifications. An uncertain decision can occur with every overflow flag clear.

For polygon-indexed counts such as `[0,0]`, use `GPUGroupStatistics` dense mode with `keyCount` equal to the polygon count,
uint32 `rightRows` as keys, and output views sized to that count. Mask every allocated pair row
at or beyond the live `output.pairs.status.count` (or bind only that prefix), so stale tail keys
cannot enter grouping. Dense row `k` then counts polygon `k`, including a zero for an empty polygon.
Default compact grouping returns only occupied groups and cannot promise `[0,0]` for no matches.
Check the group-stage `output.status.overflow` independently (dense mode reports zero); a compact
alternative needs enough group capacity and an explicit scatter into a zero-filled polygon table.
A retained pair prefix alone cannot certify a complete count. These are planar point-in-polygon
rules; see the [join contract](/docs/api/gpu-spatial-analysis/operators-joins#gpuspatialpredicatejoin).

## Nearest neighbours and distances

| turf | PostGIS | GeoPandas / Shapely | PySAL | QGIS / ArcGIS / CARTO | luma.gl | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| `nearestPoint` | `ST_Distance` KNN (`<->`), `ST_ClosestPoint` | `sjoin_nearest`, `STRtree.nearest` | - | Join attributes by nearest, Near, Generate Near Table | [`GPUNearestFeatureJoin`](/docs/api/gpu-spatial-analysis/operators-joins#gpupointinpolygonjoin-and-gpunearestfeaturejoin) | Nearest point or segment feature within a per-frame radius. The tie rule is set by `GPUNearestTieMode`. `exclusive` skips the query's own ID, `neighborQueryPoints` gives `shortest_line` with the foot point, and `onAttribute` restricts to key-equal rows. |
| - | - | `sjoin_nearest` with weights | `KNN`, `DistanceBand` | - | [`GPUNearestFeatureWeights`](/docs/api/gpu-spatial-analysis/operators-joins#gpunearestfeatureweights) | Writes the nearest-feature pairs as spatial weights. |
| `nearestPointOnLine` | `ST_LineLocatePoint`, `ST_ClosestPoint` (lines) | `shapely.project`, `shapely.shortest_line` | - | Snap points to geometries | [`GPULinearReferencing`](/docs/api/gpu-spatial-analysis/operators-lines#gpulinearreferencing-and-gpulinelocate) | Measure along a line and the snapped point. `spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'great-circle'}` gives meters on a sphere (no `candidateCapacity` needed), and `normalizedMeasures` gives `normalized=True`. |
| `distance` | `ST_Distance` | `GeoSeries.distance` | `KNN`, `DistanceBand` | Distance matrix, Generate Near Table | [`GPUNeighborSearch`](/docs/api/gpu-spatial-analysis/operators-weights#gpuneighborsearch) | Fixed-capacity kNN or radius neighbours from a grid index. Ties follow a deterministic row order. |
| `distance`, `bearing`, `midpoint` | `ST_Distance(geography)`, `ST_Azimuth`, `ST_LineInterpolatePoint` | `pyproj.Geod.inv` | - | Distance (ellipsoidal) | [`GPUGeodesicPairs`](/docs/api/gpu-spatial-analysis/operators-lines#gpugeometrymeasures-gpugeodesicpairs-and-gpugeodesicdestination) | Geodesic distance, azimuth and midpoint per pair. `spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'rhumb', units: 'meters'}` gives turf `rhumbDistance` (meters) and `rhumbBearing` (degrees). For ellipsoidal distance use `metric: 'ellipsoidal', units: 'meters'` in that context. |
| `destination` | `ST_Project` | `pyproj.Geod.fwd` | - | - | [`GPUGeodesicDestination`](/docs/api/gpu-spatial-analysis/operators-lines#gpugeometrymeasures-gpugeodesicpairs-and-gpugeodesicdestination) | Point at a bearing and distance. `spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'rhumb', units: 'meters'}` gives `rhumbDestination`: longitude/latitude and bearing in degrees, distance in meters. |
| - | `ST_MaxDistance` (Sedona) | `shapely.hausdorff_distance`, `frechet_distance` (with `densify`) | - | - | [`GPUTrackSimilarity`](/docs/api/gpu-spatial-analysis/operators-lines#gputracksimilarity) | Track-to-track similarity metrics. Hausdorff is vertex to segment, as GEOS, so it changed from earlier vertex-to-vertex values. |
| - | - | - | `pointpats.PointPattern.nnd`, `ripley` indices | Nearest neighbour analysis, Average Nearest Neighbor | [`GPUPointPatternIndices`](/docs/api/gpu-dataframe-analysis#gpuvariogram-gpuspatialcorrelogram-gpuripley-gpuripleydistancefunctions-and-gpupointpatternindices) (dataframe) | Nearest-neighbour index with Donnelly edge correction and p-value, plus quadrat chi-square test. |
| - | `pgr_dijkstraCost` | `spaghetti.Network.allneighbordistances`, `pandana` | `spaghetti` | OD cost matrix, Distance matrix (network) | [`GPUNetworkCostMatrix`](/docs/api/gpu-network/operators#gpunetworksnapping-gpunetworkcostmatrix-and-gpunetworkaccessibility) (network) | Network cost between origins and destinations over a CSR network. Capacity-bounded. |

## Spatial weights

| turf | PostGIS | GeoPandas / Shapely | PySAL | QGIS / ArcGIS / CARTO | luma.gl | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| - | - | `touches`, `sjoin` for neighbours | `libpysal.weights.Queen`, `Rook` (`from_dataframe`) | Generate Spatial Weights Matrix (contiguity) | [`GPUContiguityWeights`](/docs/api/gpu-spatial-analysis/operators-weights#gpucontiguityweights) | Queen and rook from polygon geometry. Fixed neighbour capacity. |
| - | - | - | `libpysal.weights.lat2W`, `block_weights` on grids | Generate Spatial Weights Matrix (grid) | [`GPULatticeWeights`](/docs/api/gpu-spatial-analysis/operators-weights#gpulatticeweights) | Raster or lattice neighbourhoods for gridded data. |
| - | `ST_DWithin` self join | `sjoin_nearest` | `KNN`, `DistanceBand`, `Kernel`, `Kernel(fixed=False)` | Generate Spatial Weights Matrix (fixed distance, k nearest), Generate Network Spatial Weights | [`GPUNeighborSearch`](/docs/api/gpu-spatial-analysis/operators-weights#gpuneighborsearch) | kNN, radius and kernel weights from a grid index. Capacity-bounded rows. |
| - | - | - | `W.transform = 'R'`, `'B'`, `'D'`, `'V'` | Row standardisation | [`GPUSpatialWeightsTransform`](/docs/api/gpu-spatial-analysis/operators-weights#gpuspatialweightstransform) | Row, binary, double, variance, kernel and reciprocal averaging transforms. |
| - | - | - | `W.symmetrize` | - | [`GPUSpatialWeightsTranspose`](/docs/api/gpu-spatial-analysis/operators-weights#gpuspatialweightstranspose) and [`GPUSpatialWeightsAlgebra`](/docs/api/gpu-spatial-analysis/operators-weights#gpuspatialweightsalgebra) | Take the support union with `operation: 'union'`, original weights on the left and `weightRule: 'left'`. A dedicated support-union transform is not provided. |
| - | - | - | `weights.set_operations` (`w_union`, `w_intersection`, `w_difference`, `w_symmetric_difference`), `higher_order`, `w_subset`, `block_weights` | - | [`GPUSpatialWeightsAlgebra`](/docs/api/gpu-spatial-analysis/operators-weights#gpuspatialweightsalgebra) | Set operations, higher order, subset and block weights. |
| - | - | - | `W.s0`, `.s1`, `.s2`, `.cardinalities`, `.islands`, `.asymmetry()` | - | [`GPUSpatialWeightsSummary`](/docs/api/gpu-spatial-analysis/operators-weights#gpuspatialweightssummary) | Moments, cardinalities and islands. |
| - | - | - | `W.sparse.T` | - | [`GPUSpatialWeightsTranspose`](/docs/api/gpu-spatial-analysis/operators-weights#gpuspatialweightstranspose) | Transposed (reverse) weights. |
| - | - | - | `lag_spatial` | - | [`GPUSpatialLag`](/docs/api/gpu-spatial-analysis/operators-weights#gpuspatiallag) | `W y` for any weights. |
| - | - | - | `libpysal.weights.lag_categorical`, `momepy.describe_agg` | - | [`GPUNeighborhoodSummary`](/docs/api/gpu-spatial-analysis/operators-statistics#gpuneighborhoodsummary) | Neighbourhood statistics and categorical lag. |
| - | - | - | `mapclassify.greedy` | Topological coloring (QGIS) | [`GPUMapColoring`](/docs/api/gpu-spatial-analysis/operators-weights#gpumapcoloring) | Greedy parallel colouring (Jones-Plassmann, seeded hashed priorities, lowest ID on ties) of a symmetric weights graph. Rows left unfinished are flagged. |
| - | `pgr_dijkstraCost` | - | `spaghetti.Network` distance weights | Generate Network Spatial Weights | [`GPUNetworkCostMatrix`](/docs/api/gpu-network/operators#gpunetworksnapping-gpunetworkcostmatrix-and-gpunetworkaccessibility) (network) | Network cost between locations, which you threshold into weights. |

## Spatial autocorrelation and hot spots

| turf | PostGIS | GeoPandas / Shapely | PySAL | QGIS / ArcGIS / CARTO | luma.gl | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| - | - | - | `esda.Moran`, `Geary`, `G` (General G), `Moran_BV`, `Join_Counts` | Spatial Autocorrelation (Global Moran's I), High/Low Clustering, CARTO `ST_MORANS_I_H3` | [`GPUGlobalSpatialStatistics`](/docs/api/gpu-spatial-analysis/operators-statistics#gpuglobalspatialstatistics) | Analytic statistics with z-scores. |
| - | - | - | `esda.Moran` `.p_sim`, `.z_sim`, `.seed` | - | [`GPUGlobalPermutationTest`](/docs/api/gpu-spatial-analysis/operators-statistics#gpulocalpermutationtest-and-gpuglobalpermutationtest) | Conditional or global permutation tests with a deterministic stream. |
| - | - | - | `esda.Moran_Local` | Cluster and Outlier Analysis (Anselin Local Moran's I), CARTO `ST_LOCALMORANS_H3` | [`GPULocalMoran`](/docs/api/gpu-spatial-analysis/operators-statistics#gpuhotspotanalysis-and-gpulocalmoran) | Quadrants and analytic p-values. `quadrantGating: 'analytic'` gates on analytic p-values; `'none'` leaves quadrants ungated. Permutation gating requires a separate test and its significance mask. |
| - | - | - | `esda.Moran_Local_BV` | Bivariate local Moran permutation inference | [`GPULocalPermutationTest`](/docs/api/gpu-spatial-analysis/operators-statistics#gpulocalpermutationtest-and-gpuglobalpermutationtest) | Use `statistic: 'localBivariateMoran'`, `values: x`, `secondValues: y`; `output.observed` is the observed statistic and `output.pseudoPValues` the permutation probability. Optional `output.quadrants` is ungated; combine with `output.significant` for a cluster map. |
| - | - | - | `esda.G_Local` (`Gi`, `Gi*`) | Hot Spot Analysis (Getis-Ord Gi*), CARTO `ST_GETISORD_H3` | [`GPUHotSpotAnalysis`](/docs/api/gpu-spatial-analysis/operators-statistics#gpuhotspotanalysis-and-gpulocalmoran) | Gi and Gi* z-scores with optional false discovery rate. |
| - | - | - | `esda.Moran_Local` `.p_sim`, `G_Local` `.p_sim` | - | [`GPULocalPermutationTest`](/docs/api/gpu-spatial-analysis/operators-statistics#gpulocalpermutationtest-and-gpuglobalpermutationtest) | Permutation p-values for local Moran, Gi and Gi*. Capped by `maximumPermutations`. |
| - | - | - | `esda.fdr` | Apply False Discovery Rate correction | `multipleTesting` option of [`GPUHotSpotAnalysis`](/docs/api/gpu-spatial-analysis/operators-statistics#gpuhotspotanalysis-and-gpulocalmoran), [`GPULocalMoran`](/docs/api/gpu-spatial-analysis/operators-statistics#gpuhotspotanalysis-and-gpulocalmoran), [`GPULocalPermutationTest`](/docs/api/gpu-spatial-analysis/operators-statistics#gpulocalpermutationtest-and-gpuglobalpermutationtest) | Benjamini-Hochberg style control. |
| - | - | - | `smoothing.Empirical_Bayes` | Unclamped global smoothing | [`GPUEmpiricalBayesRates`](/docs/api/gpu-spatial-analysis/operators-statistics#gpuempiricalbayesrates) | Set `clampPriorVariance: false` and supply `output.smoothedRates` for PySAL smoothing. |
| - | - | - | - | Empirical Bayesian smoothing (GeoDa) | [`GPUEmpiricalBayesRates`](/docs/api/gpu-spatial-analysis/operators-statistics#gpuempiricalbayesrates), [`GPUSpatialEmpiricalBayesRates`](/docs/api/gpu-spatial-analysis/operators-statistics#gpuspatialempiricalbayesrates) | `GPUEmpiricalBayesRates` defaults to `clampPriorVariance: true`. `GPUSpatialEmpiricalBayesRates` always clamps its neighborhood prior and has no `clampPriorVariance` prop. Pass smoothed rates to global or local Moran for a cluster analysis of those rates. |
| - | - | - | `assuncao_rate`, `esda.Moran_Rate`, `Moran_Local_Rate` (`adjusted=True`) | Assunção–Reis rate standardization then Moran | [`GPUEmpiricalBayesRates`](/docs/api/gpu-spatial-analysis/operators-statistics#gpuempiricalbayesrates) → [`GPUGlobalSpatialStatistics`](/docs/api/gpu-spatial-analysis/operators-statistics#gpuglobalspatialstatistics) or [`GPULocalMoran`](/docs/api/gpu-spatial-analysis/operators-statistics#gpuhotspotanalysis-and-gpulocalmoran) | Feed `output.standardizedRates` as Moran `values`. Standardization does not clamp the prior variance and is distinct from smoothing. |
| - | - | - | `esda.Moran` correlogram (`spatial_correlogram`) | Incremental Spatial Autocorrelation | [`GPUSpatialCorrelogram`](/docs/api/gpu-dataframe-analysis#gpuvariogram-gpuspatialcorrelogram-gpuripley-gpuripleydistancefunctions-and-gpupointpatternindices) (dataframe) | Statistic across distance bands. |
| - | - | - | `pointpats.k_test`, `Ripley K, F, G, J` | Multi-Distance Spatial Cluster Analysis (Ripley's K) | [`GPURipley`](/docs/api/gpu-dataframe-analysis#gpuvariogram-gpuspatialcorrelogram-gpuripley-gpuripleydistancefunctions-and-gpupointpatternindices), [`GPURipleyDistanceFunctions`](/docs/api/gpu-dataframe-analysis#gpuvariogram-gpuspatialcorrelogram-gpuripley-gpuripleydistancefunctions-and-gpupointpatternindices) (dataframe) | Planar edge corrections only, in a rectangular window or a valid polygon window prepared with `GPUStudyWindowPreparation`. `GPURipleyDistanceFunctions` adds Kaplan-Meier and Hanisch corrections (spatstat `km`, `han`). `GPURipleyEnvelope` adds pointwise and global Monte Carlo CSR envelopes with a MAD p-value (spatstat `envelope`). Bounded distance bins. |
| - | - | - | `pointpats.Knox`, `Mantel` | Space-Time Knox | [`GPUKnoxTest`](/docs/api/gpu-spatial-analysis/operators-dynamics#gpuknoxtest-and-gpumanteltest), [`GPUMantelTest`](/docs/api/gpu-spatial-analysis/operators-dynamics#gpuknoxtest-and-gpumanteltest) | Permutation based space-time tests. `GPUMantelTest` takes the pointpats `mantel` transform `(x + constant) ** power` as its compile-time `distanceTransform` (`scon`, `spow`) and `timeTransform` (`tcon`, `tpow`) options. |
| - | - | - | SaTScan (Kulldorff), R `smerc`, `rsatscan` | CARTO `DETECT_SPACETIME_ANOMALIES` | [`GPUSpatialScanStatistic`](/docs/api/gpu-spatial-analysis/operators-dynamics#gpuspatialscanstatistic) | Poisson space-time scan with cylinders, secondary clusters and Monte Carlo p-values. Use SaTScan or `smerc` for other probability models. |
| - | - | - | `esda.Geary_Local`; no package counterpart for the two edge statistics (`esda.Spatial_Pearson` is Lee's L, see `leesL` in `GPUGlobalPermutationTest`; `esda.Gamma` is an uncentered sum) | - | [`GPUSpatialPearson`](/docs/api/gpu-spatial-analysis/operators-statistics#gpuspatialpearson-and-gpugammastatistic), [`GPULocalGeary`](/docs/api/gpu-spatial-analysis/operators-statistics#gpulocalgeary), [`GPUGammaStatistic`](/docs/api/gpu-spatial-analysis/operators-statistics#gpuspatialpearson-and-gpugammastatistic) | `GPUSpatialPearson` is the edge-weighted correlation of `x_i` with `y_j`; `GPUGammaStatistic` is the edge-weighted mean of a centered cross-product, squared or absolute difference. The [permutation adapters](/docs/api/gpu-spatial-analysis/operators-statistics#additional-spatial-statistics-and-bounded-optimization) name statistics of `GPUGlobalPermutationTest` and `GPULocalPermutationTest`: exact for local Geary and squared-difference Gamma; the cross-product Gamma and Pearson adapters run Moran and bivariate Moran tests, the same ordering only on regular weights. |
| - | - | - | PySAL Gi* + FDR + map | Hot Spot Analysis (end to end) | recipe `addHotSpotAnalysisRecipe` | Points or lattice source, Gi* and permutation, class breaks and colours. |
| - | - | - | `pysal` emerging hot spots workflow | Emerging Hot Spot Analysis, CARTO `ST_SPACETIME_HOTSPOTS` | [`GPUEmergingHotSpots`](/docs/api/gpu-spatial-analysis/operators-statistics#gpuemerginghotspots); recipe `addSpaceTimeHotSpotsRecipe` | Space-time cube trend and Gi* categories. The recipe aggregates a dense cube first. |
| - | - | - | - | Compare two periods (change map) | recipe `addPeriodComparisonRecipe` | Two cell tables compared over a union mask, then class breaks. |

### Local inference and rate outputs

All views below are caller-owned graph views, allocated before construction. For bivariate
permutation inference, `observed` and `pseudoPValues` are float32 rows; `exceedances`, `significant`
and `quadrants` are uint32 rows; `permutationOverflow` is one uint32 status row:

```ts
import {
  GPULocalPermutationTest,
  GPUEmpiricalBayesRates
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import type {
  GPULocalPermutationTestProps,
  GPUEmpiricalBayesRatesProps
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import type {GraphDataView} from '@luma.gl/graph/gpu-core';

declare const x: GPULocalPermutationTestProps['values'];
declare const y: NonNullable<GPULocalPermutationTestProps['secondValues']>;
declare const weights: GPULocalPermutationTestProps['weights'];
declare const parameters: GPULocalPermutationTestProps['parameters'];
declare const observed: GraphDataView<'float32'>;
declare const pseudoPValues: GraphDataView<'float32'>;
declare const exceedances: GraphDataView<'uint32'>;
declare const significant: GraphDataView<'uint32'>;
declare const quadrants: GraphDataView<'uint32'>;
declare const permutationOverflow: GraphDataView<'uint32'>;
declare const events: GPUEmpiricalBayesRatesProps['events'];
declare const populations: GPUEmpiricalBayesRatesProps['populations'];
declare const smoothedRates: GraphDataView<'float32'>;

// x, y, weights and parameters (getGPUPermutationParameterValues) are supplied by the caller.
// The packed permutation count must be in [1, 999].
const bivariate = new GPULocalPermutationTest({
  statistic: 'localBivariateMoran', values: x, secondValues: y, weights, parameters,
  maximumPermutations: 999,
  alternative: 'directed', islands: 'keep', // legacy tie-free PySAL convention
  output: {observed, pseudoPValues, exceedances, significant, quadrants,
    status: {overflow: permutationOverflow}}
});
// Supply smoothedRates as a float32 output view, then read that view after execution.
const smoothing = new GPUEmpiricalBayesRates({
  events, populations, clampPriorVariance: false, output: {smoothedRates}
});
```

Add these contributors to the graph before compiling. Check `permutationOverflow` before
interpreting inference: capacity or incomplete weights can suppress tests. For **univariate** permutation-gated
quadrants, run `GPULocalMoran` with `quadrantGating: 'none'` and a separate
`GPULocalPermutationTest` with `statistic: 'localMoran'`; retain a quadrant only where the test's
`output.significant` is nonzero. For the bivariate test above, apply the same mask to its own
`output.quadrants`. There is no `quadrantGating: 'permutation'` option.

The example explicitly selects `alternative: 'directed', islands: 'keep'` for legacy tie-free
PySAL parity and its analysis population. The library defaults are `'two-sided'` and
`islands: 'exclude'`. With 99 tie-free draws and inclusive tail counts 3 and 96, the default
probability is `2 * (3 + 1) / 100 = 0.08`; directed gives `0.04`, changing significance at `0.05`.
Directed inference is anti-conservative and is retained for legacy comparisons. Ties are included
in both library tails, unlike esda's strict-complement legacy tail, so directed is not an exact
parity promise for tied simulations. See the [permutation contract](/docs/api/gpu-spatial-analysis/operators-statistics#gpulocalpermutationtest-and-gpuglobalpermutationtest).

For events `[10,11,9]` and populations `[100,100,100]`, the pooled rate is `0.1` and prior
variance is `-0.0009333333…`. PySAL-style unclamped smoothed rates are `[0.10,-0.04,0.24]`;
clamped smoothing gives `[0.10,0.10,0.10]`. Assunção–Reis standardized scores are
`[0,1.224745,-1.224745]` in either clamp mode. To reproduce adjusted `Moran_Rate`, pass
`standardizedRates` into `GPUGlobalSpatialStatistics` with `statistics: ['moran']`; for adjusted
`Moran_Local_Rate`, pass them into `GPULocalMoran`. `addRateClusterMapRecipe` defaults to `analyze: 'standardized'` (adjusted local Moran rates);
`analyze: 'smoothed'` selects smoothing, with `clampPriorVariance: false` for unclamped PySAL
smoothing. The analyzed rate and inference settings are separate choices.

## Kepler and GeoDa analysis recipes

These are compute operations for host integration. Their contracts describe bounded behavior,
including intentional differences from the pinned GeoDa implementations.

| Analysis | luma.gl | Contract boundary |
| --- | --- | --- |
| Selected-quantile LISA | [GPUQuantileLISA](/docs/api/gpu-spatial-analysis/quantile-lisa) | Single variable, type-5 quantiles and conditional binary join counts. Uniform Philox sampling; legacy directed p-value limitations are explicit. |
| Dorling cartogram | [GPUDorlingCartogram](/docs/api/gpu-spatial-analysis/gpu-dorling-cartogram) | Planar circle relaxation with origin attraction, overlap and convergence diagnostics. At most 1,073,741,822 rows on the neighbor-grid route (the default from 32,768 rows), 65,536 on the all-pairs route, and 256 passes; heuristic layout. |
| Exact natural breaks | [GPUClassBreaks](/docs/api/gpu-dataframe-analysis) | Opt-in exact dynamic programming for a normalized float32 objective, at most 8192 rows/64 classes, with recomputation above 1024 rows to avoid quadratic storage. Explicit monotone mode uses approximate quantized costs for larger inputs; approximate default preserved. |
| Variable standardization | [GPUVariableStandardization](/docs/api/variable-standardization) | Deviation, z-score, mean absolute deviation and two range methods; explicit masks, constants and sample/population conventions. |
| Excess risk and spatial rates | [addExcessRiskRecipe and addSpatialRateRecipe](/docs/api/gpu-spatial-analysis/rate-mapping) | Pooled expected counts and binary-neighborhood ratio of sums; existing EB smoothing remains available. |
| SQL data preparation | [LuSQLContext](https://luma.gl/docs/api-reference/experimental/gpu-sql) | Quoted identifiers and bounded single-key JOIN USING with left projections. Does not supply general DuckDB compatibility. |

## Regression

| turf | PostGIS | GeoPandas / Shapely | PySAL | QGIS / ArcGIS / CARTO | luma.gl | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| - | - | - | `spreg.OLS` | Ordinary Least Squares (Generalized Linear Regression) | [`GPUOrdinaryLeastSquares`](/docs/api/gpu-spatial-analysis/operators-regression#gpuordinaryleastsquares) | Bounded regressor count. |
| - | - | - | `spreg.diagnostics_sp.LMtests`, `MoranRes` | Spatial Autocorrelation of residuals | [`GPUSpatialRegressionDiagnostics`](/docs/api/gpu-spatial-analysis/operators-regression#gpuspatialregressiondiagnostics); recipe `addSpatialRegressionRecipe` | LM tests and residual Moran. The recipe needs a finite residual on every row. |
| - | - | - | `spreg.GM_Lag` | - | [`GPUSpatialTwoStageLeastSquares`](/docs/api/gpu-spatial-analysis/operators-regression#gpuspatialtwostageleastsquares) | Spatial lag by two-stage least squares. `instrumentOrder: 2` is `w_lags=2`. |
| - | - | - | `spreg.GM_Error` | - | [`GPUSpatialErrorGM`](/docs/api/gpu-spatial-analysis/operators-regression#gpuspatialerrorgm) | Spatial error by generalised moments. |
| - | - | - | `mgwr.GWR` | Geographically Weighted Regression | [`GPUGeographicallyWeightedRegression`](/docs/api/gpu-spatial-analysis/operators-regression#gpugeographicallyweightedregression) | Fixed or adaptive bandwidth, with the local condition number (`local_collinearity`). Use `mgwr` when multiscale backfitting is required. |
| - | - | - | `mgwr.GWR.spatial_variability` | GWR4 | [`GPUGeographicallyWeightedRegressionNonstationarityTest`](/docs/api/gpu-spatial-analysis/operators-regression#gpugeographicallyweightedregressionnonstationaritytest) | Monte Carlo test of coefficient non-stationarity at the selected bandwidth. |
| - | - | - | `spreg.ML_Lag`, `ML_Error`, `spat_impacts` | - | [`GPUSpatialLagMaximumLikelihood`](/docs/api/gpu-spatial-analysis/operators-regression#gpuspatiallagmaximumlikelihood), [`GPUSpatialErrorMaximumLikelihood`](/docs/api/gpu-spatial-analysis/operators-regression#gpuspatialerrormaximumlikelihood), [`GPUSpatialLagImpacts`](/docs/api/gpu-spatial-analysis/operators-regression#gpuspatiallagimpacts) | `method='full'` equivalents with a truncated-trace log-determinant; impact standard errors are not provided. |
| - | - | - | `spreg` panel, SUR, regimes, probit | - | [`GPUOrdinaryLeastSquares`](/docs/api/gpu-spatial-analysis/operators-regression#gpuordinaryleastsquares) | Caller-partitioned OLS covers panel and regime fits. Use `spreg` when SUR or probit inference is required. |
| - | - | - | `inequality.gini.Gini`, `theil.Theil`, `Gini_Spatial` | - | [`GPUInequality`](/docs/api/gpu-dataframe-analysis#gpuinequality) (dataframe) | Inequality indices. |
| - | - | - | `segregation` aspatial and spatial indices | - | [`GPUSegregation`](/docs/api/gpu-spatial-analysis/operators-accessibility#gpusegregation) | Dissimilarity, isolation and relatives, with local outputs. Undefined indices are NaN; no permutation inference. |

## Clustering and classification

| turf | PostGIS | GeoPandas / Shapely | PySAL | QGIS / ArcGIS / CARTO | luma.gl | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| `clustersDbscan` | `ST_ClusterDBSCAN` | `sklearn.cluster.DBSCAN` on coordinates | `esda.adbscan` | DBSCAN clustering, Density-based Clustering | [`GPUDBSCAN`](/docs/api/gpu-spatial-analysis/operators-statistics#gpudbscan) | Clusters at or above `maximumClusterCount` are dropped. Planar or longitude/latitude with a great-circle `epsilon` (antimeridian-wrapped neighbor search; rotate longitudes and bounds about a central meridian before clustering when seam-safe arithmetic centroids are needed); recenter large planar coordinates (float32). |
| - | - | `sklearn.cluster.HDBSCAN`, `hdbscan.HDBSCAN` | - | HDBSCAN, Density-based Clustering (HDBSCAN) | [`GPUHDBSCAN`](/docs/api/gpu-spatial-analysis/operators-statistics#gpuhdbscan) | Excess-of-mass selection, planar, up to 4,194,304 rows on default `coreDistanceSearch: 'bvh'`; opt-in `'neighbor-search'` reference allows 1,048,576 rows and rejects times; ties in mutual reachability may condense a few points differently from scikit-learn. |
| `clustersKmeans` | `ST_ClusterKMeans` | `sklearn.cluster.KMeans` | - | K-means clustering, CARTO `ST_CLUSTERKMEANS` | [`GPUSpatialKMeans`](/docs/api/gpu-spatial-analysis/operators-statistics#gpuspatialkmeans) | Iteration cap with an optional early-out `tolerance`. k-means++ by default, `restartCount` best-of-n restarts, total `inertia` output; planar only. |
| `convex` | `ST_ConvexHull` per group | `GeoSeries.convex_hull` per `groupby` | - | Convex hull, Minimum Bounding Geometry | [`GPUGroupConvexHull`](/docs/api/gpu-spatial-analysis/operators-statistics#gpugroupconvexhull), [`GPUGroupGeometry`](/docs/api/gpu-spatial-analysis/operators-statistics#gpugroupgeometry); recipe `addClusterAndOutlineRecipe` | Per-label hulls and bounds. The recipe chains cluster, hull and area. |
| `standardDeviationalEllipse`, `centralMean`, `medianCenter` | - | - | `pointpats.centrography` | Directional Distribution, Mean Center, Median Center | [`GPUGeographicDistribution`](/docs/api/gpu-spatial-analysis/operators-statistics#gpugeographicdistribution) | Planar only (`longitude-latitude` throws; spherical centre deferred). The ellipse convention is selectable; no small-n df correction. |
| - | - | `mapclassify.classify` (`Quantiles`, `EqualInterval`, `FisherJenks`, `HeadTailBreaks`, `BoxPlot`, `StdMean`, `MaximumBreaks`, `UserDefined`) | `mapclassify` | Graduated symbology modes (Quantile, Natural Breaks, Pretty, Std Dev, Equal) | [`GPUClassBreaks`](/docs/api/gpu-dataframe-analysis#gpucolumnquantiles-gpuclassbreaks-gpucolorscale-and-gpubivariateclassification) (dataframe), [`GPUClassAssignment`](/docs/api/gpu-spatial-analysis/operators-dynamics#gpuclassassignment), [`GPUColorScale`](/docs/api/gpu-dataframe-analysis#gpucolumnquantiles-gpuclassbreaks-gpucolorscale-and-gpubivariateclassification) (dataframe) | Breaks are computed on the GPU without readback. |
| - | - | - | `mapclassify` bivariate, `geoplot` | Bivariate choropleth | [`GPUBivariateClassification`](/docs/api/gpu-dataframe-analysis#gpucolumnquantiles-gpuclassbreaks-gpucolorscale-and-gpubivariateclassification) (dataframe) | - |
| - | - | - | `mapclassify` fit metrics (`gadf`, `adcm`) | - | [`GPUClassificationFit`](/docs/api/gpu-spatial-analysis/operators-dynamics#gpuclassificationfit) | Goodness of classification fit. |
| - | - | - | `giddy.Markov`, `Spatial_Markov`, `LISA_Markov`, `directional`, `Rose` | - | [`GPUTransitionMatrix`](/docs/api/gpu-spatial-analysis/operators-dynamics#gputransitionmatrix), [`GPUSpatialMarkov`](/docs/api/gpu-spatial-analysis/operators-dynamics#gpuspatialmarkov-and-gpulisamarkov), [`GPULISAMarkov`](/docs/api/gpu-spatial-analysis/operators-dynamics#gpuspatialmarkov-and-gpulisamarkov), [`GPURoseStatistic`](/docs/api/gpu-spatial-analysis/operators-statistics#additional-spatial-statistics-and-bounded-optimization) | Distribution dynamics from class sequences plus directional or axial circular summaries. |
| - | - | - | - | Calculate Composite Index | [`GPUCompositeScore`](/docs/api/gpu-dataframe-analysis#gpucompositescore) (dataframe) | Standardised weighted index. |
| - | - | - | `spopt.region.Skater` (`SpanningForest`), R `spdep::skater` | Spatially Constrained Multivariate Clustering | [`GPUSpatialWeightsMinimumSpanningTree`](/docs/api/gpu-spatial-analysis/operators-statistics#gpuspatialweightsminimumspanningtree-gpuskaterregions-and-gpuregionpartitionevaluation), [`GPUSkaterRegions`](/docs/api/gpu-spatial-analysis/operators-statistics#gpuspatialweightsminimumspanningtree-gpuskaterregions-and-gpuregionpartitionevaluation) | Boruvka minimum spanning tree over contiguity, then greedy SKATER cuts. Matches spopt partitions. |
| - | - | - | `spopt.region.AZP`, `MaxPHeuristic`, `Ward` | Spatially Constrained Multivariate Clustering | [`GPUAZPRegions`](/docs/api/gpu-spatial-analysis/operators-statistics#gpuazpregions), [`GPUMaxPRegions`](/docs/api/gpu-spatial-analysis/operators-statistics#gpumaxpregions), [`GPUWardRegions`](/docs/api/gpu-spatial-analysis/operators-statistics#gpuwardregions) | Greedy steepest-descent AZP, randomized multi-start greedy Max-P (Philox seed order per start, optional AZP refinement), and adjacent Ward linkage; attributes are standardized by default and directed weights are symmetrized. Use `spopt` when its annealing Max-P/AZP search must be reproduced exactly. |
| - | - | - | `esda.silhouettes`, `sklearn.metrics.silhouette_score`, `calinski_harabasz_score` | - | [`GPURegionPartitionEvaluation`](/docs/api/gpu-spatial-analysis/operators-statistics#gpuspatialweightsminimumspanningtree-gpuskaterregions-and-gpuregionpartitionevaluation) | Within and between sums of squares, sizes and a cross-link compactness measure for any label column. Not a silhouette score. |
| - | - | - | `sklearn.neighbors` on attribute space | Similarity Search | [`GPUSimilarLocations`](/docs/api/gpu-spatial-analysis/operators-statistics#gpusimilarlocations) | Ranks rows by weighted distance in standardised attribute space to a reference. |
| - | - | - | `spopt.locate.PMedian`, `MCLP`, `LSCP` | Location-Allocation | [`GPULocationAllocation`](/docs/api/gpu-spatial-analysis/operators-statistics#gpulocationallocation), [`GPUNetworkCostMatrix`](/docs/api/gpu-network/operators#gpunetworksnapping-gpunetworkcostmatrix-and-gpunetworkaccessibility) (network) | Bounded greedy facility selection plus a Teitz-Bart swap phase over dense costs, with assignments, objectives, uncovered demand, swap and completion status. Use `spopt` when an integer-programming optimum or richer constraints are required. |

## Density and interpolation

| turf | PostGIS | GeoPandas / Shapely | PySAL | QGIS / ArcGIS / CARTO | luma.gl | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| `hexGrid` + count, `collect` | `ST_HexagonGrid` + `ST_Within` | `hexbin` | - | Point Density | [`GPUPointDensity`](/docs/api/gpu-spatial-analysis/operators-density#gpupointdensity) | Binned count (or weight sum) per cell, optionally smoothed with a kernel measured in cells. Not an intensity and not a kernel density; use `GPUKernelDensity`. |
| - | - | `spatstat.geom::owin` | - | Study area | [`GPUStudyWindowPreparation`](/docs/api/gpu-spatial-analysis/operators-density#gpustudywindowpreparation) | GPU polygon study window with holes and islands: oriented edges, area, perimeter, edge index and sampler cells. Planar only. |
| - | - | `scipy.stats.gaussian_kde`, `sklearn.neighbors.KernelDensity` | - | Kernel Density, Heatmap (Kernel Density) | [`GPUKernelDensity`](/docs/api/gpu-spatial-analysis/operators-density#gpukerneldensity) | Intensity per unit area, bandwidth in ground units, gaussian, quartic and epanechnikov kernels, uniform or Diggle edge correction. Planar only. |
| `hexGrid` + count | `h3_lat_lng_to_cell` + `GROUP BY` | `h3pandas`, `quadbin` | - | CARTO `QUADBIN_FROMGEOGPOINT` | [`GPUCellAggregation`](/docs/api/gpu-spatial-analysis/operators-cells#gpucellaggregation-gpucellrollup-gpucellpyramid-and-gpucelllevelselection), [`GPUPointToCell`](/docs/api/gpu-spatial-analysis/operators-cells#gpupointtocell-and-gpucellgeometry) | H3 and Quadbin cells. Cell keys must be valid. Points are longitude/latitude; pre-keyed rows are `discrete-grid`. |
| `idw`, `interpolate` | - | `scipy.interpolate`, `gstat` | `mgwr` kernels | IDW interpolation, CARTO `ST_INTERPOLATE_POINTS` | [`GPUInverseDistanceWeighting`](/docs/api/gpu-spatial-analysis/operators-density#gpuinversedistanceweighting) | Neighbourhood IDW onto a raster, planar or great-circle (lon/lat, meters), optional Franke-Nielson taper. |
| - | - | `skgstat.Variogram` | - | Variogram (SAGA) | [`GPUVariogram`](/docs/api/gpu-dataframe-analysis#gpuvariogram-gpuspatialcorrelogram-gpuripley-gpuripleydistancefunctions-and-gpupointpatternindices) (dataframe) | Empirical variogram to choose a model. Pair it with `GPUKriging`. |
| - | - | `skgstat.OrdinaryKriging`, `pykrige.OrdinaryKriging` (`n_closest_points`), R `gstat::krige` | - | Ordinary Kriging (SAGA), Kriging (ArcGIS) | [`GPUKriging`](/docs/api/gpu-spatial-analysis/operators-density#gpukriging) | Local ordinary kriging on a raster with kriging variance, ill-conditioning and leave-one-out diagnostics, planar or great-circle (lon/lat, meters). Use `gstat` or PyKrige for global and universal kriging. |
| `randomPoint`, `sample` | `ST_GeneratePoints` | `GeoSeries.sample_points` | - | Random points in polygons, Dot density | [`GPURandomPointsInPolygon`](/docs/api/gpu-spatial-analysis/operators-density#gpudotdensity-and-gpurandompointsinpolygon), [`GPUDotDensity`](/docs/api/gpu-spatial-analysis/operators-density#gpudotdensity-and-gpurandompointsinpolygon) | Deterministic seed. Bounded point count. |
| - | `ST_Length(ST_Intersection(line, cell))` summed | - | - | Line density | [`GPULineDensity`](/docs/api/gpu-spatial-analysis/operators-geometry#gpulinedensity) | Length per cell divided by cell area (box kernel), exact per segment. Not the circular search-radius line density of ArcGIS and QGIS. |
| - | - | - | `tobler.area_interpolate` | Areal interpolation | [`GPUArealInterpolation`](/docs/api/gpu-spatial-analysis/operators-accessibility#gpuarealinterpolation); recipe `addChangeOfSupportRecipe` | Areal weighting on a rasterised overlap; intensive weights default to the overlap denominator, and per-source conserved fractions, anchors for sub-cell sources and spherical cell areas for longitude/latitude rasters are available. The recipe adds a spatial lag. |
| - | - | - | `tobler.pycno` | Pycnophylactic interpolation | [`GPUPycnophylacticInterpolation`](/docs/api/gpu-spatial-analysis/operators-accessibility#gpupycnophylacticinterpolation) | Iterative smoothing with volume preservation, an optional convergence tolerance with `converged` status, anchor cells for zones smaller than a cell, and density-per-area smoothing on longitude/latitude rasters. |
| - | - | - | `access.Access.two_step_fca`, `weighted_catchment`, `Huff` | Enrich, Huff Model | [`GPUCatchmentAccessibility`](/docs/api/gpu-spatial-analysis/operators-accessibility#gpucatchmentaccessibility), [`GPUHuffTradeAreas`](/docs/api/gpu-spatial-analysis/operators-accessibility#gpuhufftradeareas) | Floating catchment and Huff trade areas. |
| - | - | - | - | Focal Statistics, Neighborhood Statistics | [`GPUFocalStatistics`](/docs/api/gpu-spatial-analysis/operators-statistics#gpufocalstatistics) | Focal windows on a lattice. |

## Geometry utilities

| turf | PostGIS | GeoPandas / Shapely | PySAL | QGIS / ArcGIS / CARTO | luma.gl | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| `area`, `length`, `centroid`, `bbox`, `center`, `centerOfMass` | `ST_Area`, `ST_Length`, `ST_Perimeter`, `ST_Centroid` | `GeoSeries.area`, `.length`, `.centroid`, `.bounds` | - | Polygon centroids, Add geometry attributes, Calculate Geometry | [`GPUGeometryMeasures`](/docs/api/gpu-spatial-analysis/operators-lines#gpugeometrymeasures-gpugeodesicpairs-and-gpugeodesicdestination) | Per feature and per group. Planar, geographic or `'geodesic'` (Karney-style area), per-feature extreme vertices, and point features. |
| `pointOnFeature` | `ST_PointOnSurface`, `ST_MaximumInscribedCircle` | `representative_point()` | - | Pole of inaccessibility, Point on surface | [`GPULabelPoint`](/docs/api/gpu-spatial-analysis/operators-geometry#gpulabelpoint) | Pole of inaccessibility. |
| - | - | `momepy` compactness | `esda.shape` (`isoperimetric_quotient`, `convex_hull_ratio`) | Shape indices | [`GPUShapeDescriptors`](/docs/api/gpu-spatial-analysis/operators-geometry#gpushapedescriptors) | Elongation, orientation and compactness. |
| `booleanValid` | `ST_IsValid`, `ST_IsValidDetail`, `ST_MakeValid` | `is_valid`, `shapely.explain_validity`, `make_valid` | - | Check validity, Fix geometries | [`GPUGeometryValidity`](/docs/api/gpu-spatial-analysis/operators-validity#gpugeometryvalidity), [`GPUMakeValid`](/docs/api/gpu-spatial-analysis/operators-cells#gpulinenoding-gpupolygonize-and-gpumakevalid) | Reports invalid rows for polygons, lines and points; bounded polygon repair rebuilds valid faces and reports collapsed or unused edges. |
| - | `ST_ShiftLongitude`, `ST_WrapX`, `ST_Segmentize(geography)` | Python `antimeridian` | - | GDAL `-wrapdateline` | [`GPULonLatPolygonPreparation`](/docs/api/gpu-spatial-analysis/operators-geometry#gpulonlatpolygonpreparation) | GPU short-way longitude unwrapping, optional great-circle densification, nearer-pole capping and antimeridian splitting; fixed doubled polygon/ring topology. |
| `bboxClip`, `intersect` | `ST_ClipByBox2D`, `ST_Intersection` | `shapely.clip_by_rect`, `clip`, `overlay` | - | Clip by extent, Intersection | [`GPURectangleClip`](/docs/api/gpu-spatial-analysis/operators-geometry#gpurectangleclip), [`GPULineClipByPolygon`](/docs/api/gpu-spatial-analysis/operators-lines#gpulineclipbypolygon-and-gpusharedpaths), [`GPUPolygonOverlay`](/docs/api/gpu-spatial-analysis/operators-geometry#gpupolygonoverlay-and-gpubuffersurface) | Rectangle, line-by-polygon, and bounded polygon-by-polygon clipping. |
| `squareGrid`, `hexGrid`, `triangleGrid`, `pointGrid` | `ST_SquareGrid`, `ST_HexagonGrid` | - | - | Create grid | [`GPUGridGenerator`](/docs/api/gpu-spatial-analysis/operators-geometry#gpugridgenerator) | Square, hexagon (pointy or flat) and triangle grids with a capacity cap, and an optional extent mask. |
| `circle`, `sector`, `ellipse` | `ST_Buffer` on a point, `ST_MakeEllipse` | `shapely.Point.buffer` | - | Create wedge buffers | [`GPUShapeGenerator`](/docs/api/gpu-spatial-analysis/operators-geometry#gpushapegenerator) | Planar or geodesic circles, sectors and ellipses per feature. Circles and sectors match turf. |
| - | `ST_Hilbert` | `GeoSeries.hilbert_distance`, `spatialpandas` `hilbert_distance` | - | - | [`GPUHilbertKeys`](/docs/api/gpu-spatial-analysis/operators-geometry#gpuhilbertkeys) | Hilbert keys and a curve-order permutation, order up to 16. |
| `buffer` | `ST_Buffer` | `buffer` | - | Buffer | [`GPUOutlineGeometry`](/docs/api/gpu-spatial-analysis/operators-geometry#gpuoutlinegeometry), [`GPUBufferSurface`](/docs/api/gpu-spatial-analysis/operators-geometry#gpupolygonoverlay-and-gpubuffersurface) | `GPUOutlineGeometry` is render-only; `GPUBufferSurface` creates bounded planar polygon surfaces with cap and join styles. |
| `simplify` | `ST_Simplify`, `ST_SimplifyVW` | `GeoSeries.simplify` | - | Simplify | [`GPULineSimplification`](/docs/api/gpu-spatial-analysis/operators-lines#gpulinesimplification) | Douglas-Peucker, or approximate Visvalingam-Whyatt (`method: 'visvalingam'`, about 0.1 to 3 percent of vertices differ from the heap order). |
| `simplify` (topology preserving) | `ST_CoverageSimplify` | `shapely.simplify_coverage` | - | Simplify (topological) | [`GPUCoverageSimplification`](/docs/api/gpu-spatial-analysis/operators-lines#gpucoveragesimplification) | Shared edges stay gap-free, with exact crossing detection and repair. The criterion is distance, not area. `simplifyBoundary: false` keeps unshared edges. |
| `lineSplit` | `ST_Split`, `ST_Node` | `shapely.ops.split`, `shapely.node` | - | Split with lines, Feature To Line | [`GPULineSplit`](/docs/api/gpu-spatial-analysis/operators-lines#gpulinesplit-and-gpulinemerge) | Splits lines at every crossing, touch and shared vertex, as `unary_union` of the lines does. Capacity-bounded pieces. |
| - | `ST_LineMerge` | `shapely.line_merge` | - | Merge lines, Dissolve (unsplit lines) | [`GPULineMerge`](/docs/api/gpu-spatial-analysis/operators-lines#gpulinesplit-and-gpulinemerge) | Joins lines at endpoints shared by exactly two lines. Exact endpoint match. `directed: true` joins only end to start. |
| `polygonSmooth`, `bezierSpline` | `ST_ChaikinSmoothing` | - | - | Smooth | [`GPULineSmooth`](/docs/api/gpu-spatial-analysis/operators-lines#gpulinesegmentize-gpugreatcirclearcs-gpulinesmooth-and-gpulinechunk) | Chaikin corner cutting. |
| `lineChunk`, `lineSegment` | `ST_Segmentize` | `shapely.segmentize` | - | Densify by interval, Split lines by maximum length | [`GPULineSegmentize`](/docs/api/gpu-spatial-analysis/operators-lines#gpulinesegmentize-gpugreatcirclearcs-gpulinesmooth-and-gpulinechunk), [`GPULineChunk`](/docs/api/gpu-spatial-analysis/operators-lines#gpulinesegmentize-gpugreatcirclearcs-gpulinesmooth-and-gpulinechunk) | Fixed-capacity output with overflow flags. |
| `along` | `ST_LineInterpolatePoint` (fraction) | `shapely.line_interpolate_point` | - | Interpolate point on line | [`GPULineLocate`](/docs/api/gpu-spatial-analysis/operators-lines#gpulinearreferencing-and-gpulinelocate) | Returns a position. Use `measureMode: 'distance'` for Turf `along` (cumulative context length), and `measureMode: 'fraction'` for `ST_LineInterpolatePoint` and normalized interpolation (0–1). `ST_LocateAlong` uses stored M coordinates and has no equivalent here. Projection of a point to a line instead uses `GPULinearReferencing`. |
| `lineSliceAlong` | `ST_LineSubstring` (distance measures here) | `shapely.ops.substring` | - | Extract line substring | [`GPULineChunk`](/docs/api/gpu-spatial-analysis/operators-lines#gpulinesegmentize-gpugreatcirclearcs-gpulinesmooth-and-gpulinechunk) | Use `mode: 'substring'` and packed start/end measures; retains intermediate vertices. See the measure example below. |
| `greatCircle` | `ST_Segmentize(geography)` | - | - | Geodesic arcs, CARTO `ST_GREATCIRCLE` | [`GPUGreatCircleArcs`](/docs/api/gpu-spatial-analysis/operators-lines#gpulinesegmentize-gpugreatcirclearcs-gpulinesmooth-and-gpulinechunk) | Densified great-circle arcs. |
| `lineIntersect`, `kinks` | `ST_Intersection` (points) | - | - | Line intersections | [`GPUSegmentIntersection`](/docs/api/gpu-spatial-analysis/operators-validity#gpusegmentintersection) | Crossing points between segment sets. |
| `polygonize` | `ST_Polygonize`, `ST_BuildArea`, `h3_cells_to_multi_polygon_wkb` | `shapely.polygonize`, `polygonize_full` | - | Polygonize | [`GPUPolygonize`](/docs/api/gpu-spatial-analysis/operators-cells#gpulinenoding-gpupolygonize-and-gpumakevalid), [`GPULineNoding`](/docs/api/gpu-spatial-analysis/operators-cells#gpulinenoding-gpupolygonize-and-gpumakevalid), [`GPUSegmentRingAssembly`](/docs/api/gpu-spatial-analysis/operators-cells#gpusegmentringassembly), [`GPUCellSetOutline`](/docs/api/gpu-spatial-analysis/operators-cells#gpucellsetoutline) | Bounded planar noding and polygonization with closed-ring, open-chain, cut-edge, dangle and unused-edge diagnostics. |
| - | `ST_Length(ST_Intersection(line, polygon))` | `overlay` + `length` | - | Sum line lengths | [`GPULineLengthPerPolygon`](/docs/api/gpu-spatial-analysis/operators-geometry#gpulinelengthperpolygon) | Line length per polygon. |
| `union`, `intersect`, `difference`, `dissolve`, `voronoi`, `tin`, `concave` | `ST_Union`, `ST_Intersection`, `ST_Difference`, `ST_Buffer`, `ST_Voronoi*`, `ST_Delaunay*`, `ST_MakeValid`, `ST_Subdivide` | `overlay`, `dissolve`, `unary_union`, `voronoi_polygons`, `delaunay_triangles`, `concave_hull`, `make_valid` | - | Dissolve, Union, Intersection, Difference, Buffer, Delaunay, Voronoi, Fix geometries | [`GPUPolygonOverlay`](/docs/api/gpu-spatial-analysis/operators-geometry#gpupolygonoverlay-and-gpubuffersurface), [`GPUBufferSurface`](/docs/api/gpu-spatial-analysis/operators-geometry#gpupolygonoverlay-and-gpubuffersurface), [`GPUDelaunayTessellation`](/docs/api/gpu-spatial-analysis/operators-geometry#gpudelaunaytessellation-and-gpuvoronoidiagram), [`GPUVoronoiDiagram`](/docs/api/gpu-spatial-analysis/operators-geometry#gpudelaunaytessellation-and-gpuvoronoidiagram), [`GPUMakeValid`](/docs/api/gpu-spatial-analysis/operators-cells#gpulinenoding-gpupolygonize-and-gpumakevalid) | Bounded planar outputs with explicit capacities and completeness status. For shipped concave hull and bounded subdivision alternatives see the rows below; coverage-only union is `GPUCoverageDissolve` and raster Voronoi is [`GPUDistanceField`](/docs/api/gpu-raster/operations-analysis#gpudistancefield). |
| `concave` | `ST_ConcaveHull` (different criteria) | `concave_hull` (ratio criterion) | - | Concave hull | [`GPUConcaveHull`, `GPUGroupConcaveHull`](/docs/api/gpu-raster/expansion), [`GPURatioConcaveHull`](/docs/api/gpu-geometry-refinement) | Planar area hulls: alpha/circumradius filtering, grouped alpha hulls, or ratio-based Delaunay erosion. Collinear/isolated points emit no area. Ratio erosion is parallel and can differ from GEOS largest-first erosion. See the [geometry-refinement comparison](/docs/api/gpu-geometry-refinement) for modes, limits and status. |
| - | `ST_Subdivide` | - | - | Subdivide polygons | [`GPUSubdividePolygons`](/docs/api/gpu-geometry-refinement) | Bounded planar subdivision into convex pieces, `maximumVertices` 3–128; `method: 'rectilinear'` requires at least 4. Holes remain excluded. Seed decomposition and piece ordering differ from PostGIS; compare piece union and vertex bound. See the [comparison](/docs/api/gpu-geometry-refinement) for capacities and incomplete splitting. |
| - | - | - | - | Model Designer, graphical model | recipes, or compose contributors in one `GPUCommandGraph` | - |

### Substrings preserve bends

`GPULineLocate.eventMeasures` uses cumulative lengths with `measureMode: 'distance'` (the default)
or unitless fractions of total line length with `measureMode: 'fraction'`. For
`[(1,3),(1,7),(7,7)]`, measure `0.5` locates `(1,3.5)` in distance mode and `(2,7)` in fraction mode.
`GPULineChunk` start/end measures are always cumulative lengths in planar position units, or the
great-circle spatial context's length unit (meters by default).
For `[(0,0),(2,0),(2,2)]`, substring measures 1 through 3 return
`[(1,0),(2,0),(2,1)]`, including the original bend. Pack
`getGPULineChunkParameterValues({startMeasure: 1, endMeasure: 3})` into the caller's
`parameters` view and construct `new GPULineChunk({positions, lineOffsets, mode: 'substring', parameters, output})`.
After execution, returned line `i` uses vertex rows from `output.lineOffsets[i]` inclusive to
`output.lineOffsets[i + 1]` exclusive in `output.positions`; in this example offsets are `[0,3]`.
Check output overflow and read only the line prefix bounded by `output.status.count`.

## GeoPandas, Shapely and Rust geo parity

polars-st and GeoPolars mirror Shapely's function names, so every Shapely row on this page also covers
the polars-st (`st.*`) and GeoPolars equivalent. The middle column lists names from the Rust `geo` crate,
Apache Sedona and h3 that have no Shapely spelling. A dash means no such name exists.

| GeoPandas / Shapely (and polars-st, GeoPolars) | geo crate, Sedona, h3, PostGIS | luma.gl | Notes |
| --- | --- | --- | --- |
| `shapely.affinity.*` (`affine_transform`, `translate`, `scale`, `rotate`, `skew`) | `ST_Affine`, `ST_Translate`, `ST_Scale`, `ST_Rotate` (Sedona) | [`GPUAffineTransform`](/docs/api/gpu-spatial-analysis/operators-geometry#gpuaffinetransform-gpugeometryorientation-and-gpugeometrycleanup) | One per-frame matrix or per-feature matrices. The origin can be a point, center or centroid. |
| `shapely.reverse`, `orient_polygons` | `ST_Reverse`, `ST_ForcePolygonCW`, `ST_ForcePolygonCCW` (Sedona) | [`GPUGeometryOrientation`](/docs/api/gpu-spatial-analysis/operators-geometry#gpuaffinetransform-gpugeometryorientation-and-gpugeometrycleanup) | Layout unchanged. |
| `shapely.remove_repeated_points`, `set_precision(mode='pointwise')` | `ST_RemoveRepeatedPoints`, `ST_SnapToGrid` (Sedona) | [`GPUGeometryCleanup`](/docs/api/gpu-spatial-analysis/operators-geometry#gpuaffinetransform-gpugeometryorientation-and-gpugeometrycleanup) | Compacted output. GEOS pointwise precision keeps repeats, so set `removeRepeatedPoints: false` for parity. Use Shapely when canonical `normalize` ordering is required. |
| `shapely.minimum_rotated_rectangle`, `oriented_envelope`, `minimum_bounding_circle`, `minimum_bounding_radius`, `envelope` | `ST_MinimumBoundingCircle`, `ST_OrientedEnvelope`, `ST_Envelope` (Sedona) | [`GPUMinimumBounds`](/docs/api/gpu-spatial-analysis/operators-geometry#gpuminimumbounds) | Per feature or per label group. Area and radius parity, not vertex order. |
| `shapely.minimum_clearance`, `minimum_clearance_line` | `MinimumClearance` (geo) | [`GPUMinimumClearance`](/docs/api/gpu-spatial-analysis/operators-validity#gpuminimumclearance) | GEOS rule. `Infinity` when there is no clearance. |
| `shapely.is_simple`, `is_ring`, `is_closed`, `is_ccw`, `get_num_points`, `count_coordinates`, `get_num_interior_rings`, `get_num_geometries`, `extract_unique_points`, `equals_exact`, `equals_identical` | `ST_IsSimple`, `ST_NPoints`, `ST_NumInteriorRings` (Sedona) | [`GPUGeometryPredicates`](/docs/api/gpu-spatial-analysis/operators-validity#gpugeometrypredicates) | Equality compares row `i` with row `i`; normalize first on the CPU when `equals_exact(normalize=True)` semantics are required. |
| `shapely.is_valid`, `is_valid_reason` for lines and points | - | [`GPUGeometryValidity`](/docs/api/gpu-spatial-analysis/operators-validity#gpugeometryvalidity) | `lines` and `points` inputs. |
| `geopandas.clip(lines, polygon)`, `shapely.intersection(line, polygon)`, `difference(line, polygon)` | `Clip` (geo, lines) | [`GPULineClipByPolygon`](/docs/api/gpu-spatial-analysis/operators-lines#gpulineclipbypolygon-and-gpusharedpaths) | Bounded because no new polygon intersections are computed. Lines are not noded at their own crossings. |
| `shapely.shared_paths` | - | [`GPUSharedPaths`](/docs/api/gpu-spatial-analysis/operators-lines#gpulineclipbypolygon-and-gpusharedpaths) | Forward and backward runs per line pair. |
| `GeoDataFrame.dissolve(by, method='coverage')`, `shapely.coverage_union_all`, `dissolve`, `unary_union` | `ST_CoverageUnion` (PostGIS) | [`GPUCoverageDissolve`](/docs/api/gpu-spatial-analysis/operators-lines#gpucoveragedissolve), [`GPUPolygonOverlay`](/docs/api/gpu-spatial-analysis/operators-geometry#gpupolygonoverlay-and-gpubuffersurface) | Use the coverage path for valid coverages and bounded polygon overlay for general dissolve. |
| `shapely.coverage_is_valid`, `coverage_invalid_edges(gap_width)` | `ST_CoverageInvalidEdges` (PostGIS) | [`GPUCoverageValidity`](/docs/api/gpu-spatial-analysis/operators-validity#gpucoveragevalidity) | Per-segment flags. The gap rule is the GEOS one; equal to Shapely 2.2.0 on 222 fixture cases with gaps. |
| `shapely.coverage_simplify(simplify_boundary=False)` | - | [`GPUCoverageSimplification`](/docs/api/gpu-spatial-analysis/operators-lines#gpucoveragesimplification) | `simplifyBoundary: false`. |
| `shapely.offset_curve` | `OffsetCurve` (geo) | [`GPUOffsetCurve`](/docs/api/gpu-spatial-analysis/operators-geometry#gpuoffsetcurve-and-gpuvertexsnap) | Partial: no loop removal, so tight inside corners can self-cross. |
| `shapely.snap` | `Snap` (geo) | [`GPUVertexSnap`](/docs/api/gpu-spatial-analysis/operators-geometry#gpuoffsetcurve-and-gpuvertexsnap) | Partial: vertex to vertex only, no insertion onto segments. |
| `shapely.line_merge(directed=True)` | - | [`GPULineMerge`](/docs/api/gpu-spatial-analysis/operators-lines#gpulinesplit-and-gpulinemerge) | `directed: true`. |
| `GeoSeries.make_grid(flat_topped=True, intersect=True)` | - | [`GPUGridGenerator`](/docs/api/gpu-spatial-analysis/operators-geometry#gpugridgenerator) | `hexOrientation: 'flat'` and `extent` with `output.intersects`; derive corners from the emitted cell geometry when needed. |
| `GeoSeries.sample_points` (lines) | - | [`GPURandomPointsOnLine`](/docs/api/gpu-spatial-analysis/operators-density#gpudotdensity-and-gpurandompointsinpolygon) | Uniform along arc length with a stable prefix. Use GeoPandas for clustered Poisson line sampling. |
| `sjoin(on_attribute=)`, `STRtree.query_nearest(exclusive=True)`, `shapely.shortest_line`, `ops.nearest_points` | - | [`GPUSpatialPredicateJoin`](/docs/api/gpu-spatial-analysis/operators-joins#gpuspatialpredicatejoin), [`GPUNearestFeatureJoin`](/docs/api/gpu-spatial-analysis/operators-joins#gpupointinpolygonjoin-and-gpunearestfeaturejoin) | Options `onAttribute` and `neighborQueryPoints`; `exclusive: true` excludes equal IDs, unlike Shapely's equal-geometry exclusion. A coincident target with a different ID survives. See the counterexamples below. |
| `sjoin` merge (`how='left'`), `sjoin_nearest(distance_col, how)`, `explode`, `get_parts(return_index)`, `get_coordinates(return_index)`, `.cx[...]` | - | [`GPUPairGather`](/docs/api/gpu-spatial-analysis/operators-joins#gpupairgather-gpuoffsetexpansion-and-gpuboundsfilter), [`GPUOffsetExpansion`](/docs/api/gpu-spatial-analysis/operators-joins#gpupairgather-gpuoffsetexpansion-and-gpuboundsfilter), [`GPUBoundsFilter`](/docs/api/gpu-spatial-analysis/operators-joins#gpupairgather-gpuoffsetexpansion-and-gpuboundsfilter) | Left rows come after matched rows. `.cx` is bounds level. |
| `GeoSeries.hausdorff_distance`, `frechet_distance` (`densify`) | `ST_MaxDistance` (Sedona), `HausdorffDistance`, `FrechetDistance` (geo) | [`GPUTrackSimilarity`](/docs/api/gpu-spatial-analysis/operators-lines#gputracksimilarity) | Hausdorff is vertex to segment, as GEOS. |
| - | `OutlierDetection`, `ST_LocalOutlierFactor` (Sedona), `sklearn LocalOutlierFactor` | [`GPULocalOutlierFactor`](/docs/api/gpu-spatial-analysis/operators-statistics#gpulocaloutlierfactor) | Over a kNN table. Matches `-negative_outlier_factor_`. |
| `GeoSeries.geometry_area_perimeter` (pyproj `Geod`) | `GeodesicArea` (geo) | [`GPUGeometryMeasures`](/docs/api/gpu-spatial-analysis/operators-lines#gpugeometrymeasures-gpugeodesicpairs-and-gpugeodesicdestination) | `spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'ellipsoidal', units: 'meters'}`; edges follow geodesics by default (`ellipsoidalEdgeModel: 'geodesic'`). |
| `pyproj.Geod.npts`, `fwd` (spherical) | `HaversineClosestPoint`, `CrossTrackDistance`, `InterpolatePoint` (geo) | [`GPULinearReferencing`](/docs/api/gpu-spatial-analysis/operators-lines#gpulinearreferencing-and-gpulinelocate), [`GPULineLocate`](/docs/api/gpu-spatial-analysis/operators-lines#gpulinearreferencing-and-gpulinelocate) | `spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'great-circle'}`. Linear referencing walks a great-circle arc BVH branch-and-bound. |
| - | `Rhumb::distance`, `bearing`, `destination` (geo) | [`GPUGeodesicPairs`](/docs/api/gpu-spatial-analysis/operators-lines#gpugeometrymeasures-gpugeodesicpairs-and-gpugeodesicdestination), [`GPUGeodesicDestination`](/docs/api/gpu-spatial-analysis/operators-lines#gpugeometrymeasures-gpugeodesicpairs-and-gpugeodesicdestination) | `spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'rhumb', units: 'meters'}`, sphere only; distances in meters, bearings and coordinates in degrees. |
| `simplification.simplify_coords_vw` | `SimplifyVw` (geo) | [`GPULineSimplification`](/docs/api/gpu-spatial-analysis/operators-lines#gpulinesimplification) | Approximate parallel Visvalingam-Whyatt. Use `geo` when the preserve variant's heap order is required. |
| - | `TriangulateEarcut`, `ST_TriangulatePolygon` (Sedona), `earcut`, Delaunay | [`GPUPolygonTriangulation`](/docs/api/gpu-spatial-analysis/operators-geometry#gpupolygontriangulation), [`GPUDelaunayTessellation`](/docs/api/gpu-spatial-analysis/operators-geometry#gpudelaunaytessellation-and-gpuvoronoidiagram) | Earcut preserves polygon constraints and holes; Delaunay triangulates unconstrained point sites. |
| - | h3 `grid_distance`, `grid_path_cells`, `are_neighbor_cells`, `cell_area`, `edge_length`, `is_pentagon` | [`GPUCellGridPath`](/docs/api/gpu-spatial-analysis/operators-cells#gpucellgridpath-and-gpucellmeasures), [`GPUCellMeasures`](/docs/api/gpu-spatial-analysis/operators-cells#gpucellgridpath-and-gpucellmeasures) | Grid paths and measures are GPU-resident; use h3 for directed-edge and vertex index construction. |

### Geometry query distinctions

- **Nearest exclusion:** query `(0,0)` with ID 0; targets `(0,0)`, `(0,0)`, `(2,0)` with IDs
  0, 1, 2. `exclusive: true` excludes target ID 0; row 1 survives at distance 0 because it has a
  different ID. Shapely `STRtree.query_nearest(exclusive=True)` excludes both equal geometries
  and returns row 2 instead.
- **Disjoint:** left point `0`, right points `[0,1]` on the x axis. An anti-`intersects` join
  returns no left rows because left 0 intersects right 0. The pairwise disjoint result is
  `[(0,1)]`. Anti-`intersects` means left features disjoint from **every** right feature;
  enumeration of disjoint pairs is unavailable.
- **Touches:** a point `(1,2)` on the hole boundary of shell `[0,0]–[4,4]`, hole `[1,1]–[3,3]`,
  touches the polygon (DE-9IM `F0FFFF212`). Their geometries intersect and their interiors do not
  intersect; the point itself has an empty boundary.
- **Empty equality:** the predicate join enumerates zero pairs for two empty operands because
  there are no candidates. Aligned topological equality of two empty operands is true
  (`FFFFFFFF2`); zero enumerated pairs does not mean aligned inequality.

## Discrete global grid cells

| turf | PostGIS | GeoPandas / Shapely | PySAL | QGIS / ArcGIS / CARTO | luma.gl | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| - | `h3_lat_lng_to_cell` | `h3pandas.geo_to_h3` | - | CARTO `QUADBIN_FROMGEOGPOINT` | [`GPUPointToCell`](/docs/api/gpu-spatial-analysis/operators-cells#gpupointtocell-and-gpucellgeometry) | H3 and Quadbin. |
| - | `h3_cell_to_boundary` | `h3pandas.h3_to_geo_boundary` | - | - | [`GPUCellGeometry`](/docs/api/gpu-spatial-analysis/operators-cells#gpupointtocell-and-gpucellgeometry) | Centres and boundaries. |
| - | `h3_grid_disk`, `h3_cell_to_parent` | `h3.grid_disk` | - | CARTO `QUADBIN_KRING`, `QUADBIN_TOPARENT` | [`GPUCellTopology`](/docs/api/gpu-spatial-analysis/operators-cells#gpucelltopology-and-gpucellcompaction) | Neighbours and parents. |
| - | `h3_polygon_to_cells` | `h3pandas.polyfill` | - | CARTO `QUADBIN_POLYFILL` | [`GPUCellCover`](/docs/api/gpu-spatial-analysis/operators-cells#gpucellcover) | Polygon to cell cover with a containment mode. |
| - | `GROUP BY h3_cell_to_parent` | `h3pandas.h3_to_parent` + `groupby` | - | - | [`GPUCellRollup`](/docs/api/gpu-spatial-analysis/operators-cells#gpucellaggregation-gpucellrollup-gpucellpyramid-and-gpucelllevelselection), [`GPUCellPyramid`](/docs/api/gpu-spatial-analysis/operators-cells#gpucellaggregation-gpucellrollup-gpucellpyramid-and-gpucelllevelselection), [`GPUCellLevelSelection`](/docs/api/gpu-spatial-analysis/operators-cells#gpucellaggregation-gpucellrollup-gpucellpyramid-and-gpucelllevelselection) | Roll-ups across resolutions. |
| - | `h3_cells_to_multi_polygon_wkb` | `h3.cells_to_h3shape` | - | - | [`GPUCellSetOutline`](/docs/api/gpu-spatial-analysis/operators-cells#gpucellsetoutline) | Outline and optional closed rings for a cell set. |
| - | - | - | - | Compact, uncompact cells | [`GPUCellCompaction`](/docs/api/gpu-spatial-analysis/operators-cells#gpucelltopology-and-gpucellcompaction) | - |
| - | - | - | - | Compare cell tables | [`GPUCellTableCompare`](/docs/api/gpu-spatial-analysis/operators-cells#gpucelltablecompare) | Used by `addPeriodComparisonRecipe`. |

## Trajectories

| turf | PostGIS | GeoPandas / Shapely | PySAL | QGIS / ArcGIS / CARTO | luma.gl | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| - | - | `movingpandas.TrajectoryStopDetector` | - | Diameter-based stop detection | no exact counterpart | A maximum stop diameter differs from slow-step runs and anchor-distance staypoints; see the [family comparison](/docs/api/gpu-spatial-analysis/movement-semantics). |
| - | - | trackintel sliding staypoints | - | Anchor-distance stays | [`GPUTrajectoryStaypoints`](/docs/api/gpu-spatial-analysis/movement-semantics#gputrajectorystaypoints) | First fix at or beyond `distanceThreshold` closes the anchor window; `timeThreshold` and durations use the timestamp unit. `endRows` is exclusive; `finishRows` supplies the finish time. Pack `includeLast: true` to emit a qualifying window still open at track end; omitted means discard it. See the [family comparison](/docs/api/gpu-spatial-analysis/movement-semantics) for gap and endpoint rules. |
| - | `ST_Length` per track | `movingpandas` `.get_length`, `.get_speed`; slow-step stop runs | - | Trajectories to stops, Track Intelligence | [`GPUTrajectoryMetrics`](/docs/api/gpu-spatial-analysis/operators-lines#gputrajectorymetrics) | Consecutive slow-step stops, lengths and speeds, plus per-step speed, heading and acceleration columns. A step is slow below `stopSpeedThreshold` (context length per timestamp unit), or when distance is zero and time is nondecreasing. Qualifying final runs are emitted; stop `endRows` is inclusive and `stopMinimumDuration` is inclusive in the timestamp unit. See the [family comparison](/docs/api/gpu-spatial-analysis/movement-semantics) for thresholds and endpoints. |
| - | `ST_Intersects` + `LAG` | `movingpandas` zone enter/exit | - | Enter and Exit events | [`GPUZoneEvents`](/docs/api/gpu-spatial-analysis/operators-lines#gpuzoneevents) | Zone entry and exit events with crossing positions, diagnostics and a sparse (track, zone) visit table. |
| - | - | Slow-step stop runs + zone assignment + `groupby` | - | Stop dwell time in zones | recipe `addFleetDwellRecipe` | Dense zone rows; `table.counts` counts stops per zone, and `sumValues` totals stop dwell. |
| - | - | Zone visits + `groupby` | - | Track dwell time in zones | recipe `addFleetDwellZoneEventsRecipe` | Dense zone rows; `table.counts` counts visiting tracks per zone, and `sumValues` totals visit dwell. A moving track can contribute dwell without a slow step. |
| - | - | `movingpandas` `interpolate_position_at`, `TrajectoryCollection.resample` | - | Points along track | [`GPUTrajectoryPlayhead`](/docs/api/gpu-spatial-analysis/operators-lines#gputrajectoryplayhead-and-gputrajectoryresample), [`GPUTrajectoryResample`](/docs/api/gpu-spatial-analysis/operators-lines#gputrajectoryplayhead-and-gputrajectoryresample) | Position at a time and uniform resampling. |
| - | - | `movingpandas` close-by encounters | - | Proximity events | [`GPUTrajectoryEncounters`](/docs/api/gpu-spatial-analysis/operators-lines#gputrajectoryencounters); `addClockEncounters` | Pairs of tracks within a distance in the same time bucket (MobilityDB `tdwithin`). Capacity-bounded. |
| - | - | MobilityDB `tdwithin`, `nearestApproachDistance` | - | Exact proximity events | [`GPUTrajectoryClosestApproach`](/docs/api/gpu-spatial-analysis/operators-lines#gputrajectoryclosestapproach) | Closest point of approach between linearly moving tracks: minimum distance, its time, first time and duration within a distance. Capacity-bounded. |
| - | `ST_FrechetDistance`, `ST_HausdorffDistance` | `shapely.frechet_distance`, `hausdorff_distance` | - | - | [`GPUTrackSimilarity`](/docs/api/gpu-spatial-analysis/operators-lines#gputracksimilarity) | Track similarity. |
| `along` | `ST_LineInterpolatePoint` | `shapely.line_interpolate_point` | - | - | [`GPULineLocate`](/docs/api/gpu-spatial-analysis/operators-lines#gpulinearreferencing-and-gpulinelocate) | `measureMode: 'distance'` for Turf; `measureMode: 'fraction'` for PostGIS and normalized Shapely interpolation. |
| - | - | `movingpandas.TrajectoryCollection.add_speed` | - | - | [`GPUTimeWindowFilter`](/docs/api/gpu-dataframe-analysis#gputimewindowfilter), [`GPUTemporalReduction`](/docs/api/gpu-dataframe-analysis#gputemporalreduction) (dataframe) | Time-window filter and reduction. |

## Networks

| turf | PostGIS | GeoPandas / Shapely | PySAL | QGIS / ArcGIS / CARTO | luma.gl | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| - | `pgr_dijkstra`, `pgr_dijkstraCost` | `osmnx.shortest_path`, `networkx` | `spaghetti.Network` | Shortest path (point to point, layer to point), Route | [`GPUNetworkReachability`](/docs/api/gpu-network/operators#gpunetworkreachability), [`GPUNetworkPathExtraction`](/docs/api/gpu-network/operators#gpunetworkpathextraction-gpunetworkserviceareas-gpunetworkneighborhood-and-gpunetworkanalyticscolumns) (network) | Multi-source cost and predecessors. Path extraction is bounded. |
| - | `pgr_drivingDistance` | `osmnx` | `spaghetti` | Service area (from point, from layer), Service Area | [`GPUNetworkServiceAreas`](/docs/api/gpu-network/operators#gpunetworkpathextraction-gpunetworkserviceareas-gpunetworkneighborhood-and-gpunetworkanalyticscolumns), [`GPUNetworkIsochrones`](/docs/api/gpu-network/operators#gpunetworkisochrones) (network); recipe `addDriveTimeCatchmentRecipe` | Edge cut costs are exact. Isochrone raster or cell outline polygons are cell-resolution. |
| - | `pgr_withPointsDD` | `momepy.snap` | `spaghetti.Network.snapobservations` | Snap points to network, Network Analyst Locations | [`GPUNetworkSnapping`](/docs/api/gpu-network/operators#gpunetworksnapping-gpunetworkcostmatrix-and-gpunetworkaccessibility) (network) | Snaps points onto edges. |
| - | `ST_Node`, `pgr_nodeNetwork` | `shapely.node` | - | Feature To Line, GRASS `v.clean` (break) | [`GPUNetworkNoding`](/docs/api/gpu-network/operators#gpunetworknoding) (network) | Splits lines into edges and snaps their end points into nodes with a per-frame tolerance. Builds the CSR for routing. |
| - | - | `networkx.line_graph`, `momepy.gdf_to_nx(approach='dual')`, igraph `linegraph` | - | Turn restrictions (Network Analyst) | [`GPUNetworkLineGraph`](/docs/api/gpu-network/operators#gpunetworklinegraph) (network) | Line graph in CSR with turn angle, U-turn and banned-turn costs. |
| - | - | Valhalla and OSRM map matching, `fmm`, `leuvenmapmatching` | - | - | [`GPUMapMatching`](/docs/api/gpu-network/operators#gpumapmatching) (network) | Hidden Markov model with bounded route search. Planar distances. |
| - | - | - | `spaghetti.NetworkK`, `GlobalAutoK` | SANET network K | [`GPUNetworkKFunction`](/docs/api/gpu-network/operators#gpunetworkkfunction) (network) | Network-constrained K with a simulated envelope. Build cross-K from separate source and target event tables when needed. |
| - | `pgr_connectedComponents`, `pgr_pagerank`, `pgr_degree`, `pgr_betweenness` | `networkx` centralities | `spaghetti` | Network statistics | [`GPUNetworkStatistics`](/docs/api/gpu-network/operators#gpunetworkstatistics), [`GPUNetworkAnalyticsColumns`](/docs/api/gpu-network/operators#gpunetworkpathextraction-gpunetworkserviceareas-gpunetworkneighborhood-and-gpunetworkanalyticscolumns) (network) | - |
| - | - | `networkx.quotient_graph` | - | Dissolve by group (network) | [`GPUNetworkCoarsening`](/docs/api/gpu-network/operators#gpunetworkcoarsening) (network) | Summarises a CSR network by a vertex group label (community, component or cell) into supernodes for zoomed-out views. |
| - | - | `networkx.ego_graph` | - | - | [`GPUNetworkNeighborhood`](/docs/api/gpu-network/operators#gpunetworkpathextraction-gpunetworkserviceareas-gpunetworkneighborhood-and-gpunetworkanalyticscolumns) (network) | k-hop ego networks around per-frame seeds, up to a compile-time `maximumHops`. |
| - | - | `networkx.subgraph` | - | Extract by attribute | [`GPUNetworkSubgraphFilter`](/docs/api/gpu-network/operators#gpunetworksubgraphfilter) (network) | Induced subgraph from attribute and time predicates on vertex and edge rows. |
| - | - | - | `access.Access.weighted_catchment`, `pandana` | Network accessibility | [`GPUNetworkAccessibility`](/docs/api/gpu-network/operators#gpunetworksnapping-gpunetworkcostmatrix-and-gpunetworkaccessibility) (network) | Decay and catchment variants. |
| - | - | `networkx.to_numpy_array` | - | Adjacency matrix | [`GPUAdjacencyMatrix`](/docs/api/gpu-network/operators#gpuadjacencymatrix-and-gpuadjacencymatrixorder), [`GPUAdjacencyMatrixOrder`](/docs/api/gpu-network/operators#gpuadjacencymatrix-and-gpuadjacencymatrixorder) (network) | - |
| - | - | - | - | Edge bundling, Flow map | [`GPUEdgeBundling`](/docs/api/gpu-network/operators#gpuedgebundling), [`GPUFlowAggregation`](/docs/api/gpu-network/operators#gpuflowaggregation) (network) | Origin-destination aggregation by zone and time window. |

## Raster

| turf | PostGIS | GeoPandas / Shapely | PySAL | QGIS / ArcGIS / CARTO | luma.gl | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| `isolines`, `isobands` | `ST_Contour`, `ST_DumpAsPolygons` | `skimage.measure.find_contours`, `matplotlib contour` | - | Contour, Contour polygons | [`GPUIsolines`](/docs/api/gpu-raster/operations-analysis#gpuisolines-and-gpuisobands), [`GPUIsobands`](/docs/api/gpu-raster/operations-analysis#gpuisolines-and-gpuisobands), [`GPURasterContours`](/docs/api/gpu-spatial-analysis#postgis) (raster) | Marching squares with indirect overlays. Overflow flags. |
| - | `ST_MapAlgebra`, `ST_Clip` | `rasterio` + `numpy` | - | Raster calculator, Raster Calculator | [`GPURasterBandMath`](/docs/api/gpu-spatial-analysis#postgis), [`GPURasterArithmetic`](/docs/api/gpu-spatial-analysis#qgis), [`GPURasterConditional`](/docs/api/gpu-spatial-analysis#postgis) (raster) | Per-pixel math with nodata handling. |
| - | `ST_Reclass` | `numpy.digitize` | - | Reclassify by table, Reclassify | [`GPURasterReclassify`](/docs/api/gpu-raster/operations-analysis#gpurasterreclassify-gpuweightedoverlay-and-local-raster-operations) (raster) | - |
| - | `ST_SummaryStats` per polygon | `rasterstats.zonal_stats` | - | Zonal statistics (raster), Zonal Statistics as Table | [`GPURasterZonalStatistics`](/docs/api/gpu-raster/operations-analysis#gpurasterzonalstatistics) (raster); recipe `addStraightLineCatchmentsRecipe` | Deterministic sum order. |
| - | `ST_AsRaster` | `rasterio.features.rasterize` | - | Rasterize | [`GPUPolygonRasterization`](/docs/api/gpu-raster/operations-analysis#gpupolygonrasterization-and-gpurasterjoin), [`GPURasterJoin`](/docs/api/gpu-raster/operations-analysis#gpupolygonrasterization-and-gpurasterjoin) (raster) | - |
| - | `ST_ValueCount`, `ST_Histogram` | `numpy.histogram` | - | Raster layer unique values report, Histogram | [`GPURasterHistogram`](/docs/api/gpu-spatial-analysis#postgis), [`GPURasterStatistics`](/docs/api/gpu-spatial-analysis#postgis) (raster) | Global tiled statistics use a first pass domain and a replay. |
| - | - | `scipy.ndimage.distance_transform_edt` | - | Proximity (raster distance), Euclidean Distance, Voronoi (raster) | [`GPUDistanceField`](/docs/api/gpu-raster/operations-analysis#gpudistancefield) (raster); recipe `addStraightLineCatchmentsRecipe` | Euclidean distance and nearest-seed allocation. Ties break on the smallest seed ID. |
| - | - | `skimage.graph.route_through_array` | - | Cost Distance, Least-cost path, Cost Path | [`GPUCostDistance`](/docs/api/gpu-raster/operations-analysis#gpucostdistance-and-gpucostdistancepath), [`GPUCostDistancePath`](/docs/api/gpu-raster/operations-analysis#gpucostdistance-and-gpucostdistancepath) (raster) | Iterated on the GPU with convergence flags. |
| - | - | `scipy.ndimage.label` | - | Fill / clump, Region Group | [`GPURasterConnectedComponents`](/docs/api/gpu-spatial-analysis#qgis) (`@luma.gl/graph/gpu-raster`) | Four or eight connectivity, bounded dense labels. |
| - | - | `rasterio.features.sieve` | - | Sieve | [`GPURasterSieve`](/docs/api/gpu-spatial-analysis#qgis) (raster) | - |
| - | - | `scipy.ndimage.gaussian_filter`, `uniform_filter`, `sobel`, `convolve` | - | Gaussian filter, Convolve | [`GPURasterGaussianBlur`](https://luma.gl/docs/api-reference/experimental/gpu-raster/operations-filters-morphology#separable-gaussian-smoothing), [`GPURasterBoxBlur`](https://luma.gl/docs/api-reference/experimental/gpu-raster/operations-filters-morphology#separable-box-smoothing), [`GPURasterConvolution`](https://luma.gl/docs/api-reference/experimental/gpu-raster/operations-filters-morphology#direct-convolution), [`GPURasterSobel`](/docs/api/gpu-spatial-analysis#postgis), [`GPURasterScharr`](/docs/api/gpu-spatial-analysis#qgis) (raster) | - |
| - | - | `scipy.ndimage.binary_dilation`, `binary_erosion` | - | Dilation, Erosion | [`GPURasterDilation`](https://luma.gl/docs/api-reference/experimental/gpu-raster/operations-filters-morphology#binary-and-grayscale-morphology), [`GPURasterErosion`](https://luma.gl/docs/api-reference/experimental/gpu-raster/operations-filters-morphology#binary-and-grayscale-morphology), [`GPURasterOpening`](https://luma.gl/docs/api-reference/experimental/gpu-raster/operations-filters-morphology#binary-and-grayscale-morphology), [`GPURasterClosing`](https://luma.gl/docs/api-reference/experimental/gpu-raster/operations-filters-morphology#binary-and-grayscale-morphology) (raster) | - |
| - | - | `skimage.measure.regionprops` | - | Landscape metrics, Region statistics | [`GPURasterRegionMeasurements`](https://luma.gl/docs/api-reference/experimental/gpu-raster/operations-regions-contours#four-connected-versus-eight-connected-regions) (`@luma.gl/graph/gpu-raster`), [`GPURasterPatchMetrics`](./gpu-raster/raster-patches.md) (`@luma-contributors/analysis/raster-analysis`) | - |
| - | - | `skimage.filters.threshold_otsu` | - | Threshold | [`GPURasterOtsuThreshold`](https://luma.gl/docs/api-reference/experimental/gpu-raster/operations-pixel#analytical-threshold-masks), [`GPURasterThreshold`](https://luma.gl/docs/api-reference/experimental/gpu-raster/operations-filters-morphology#binary-classification-and-observation-validity) (raster) | - |
| - | `ST_MapAlgebra` NDVI | `rasterio` NDVI | - | NDVI | [`GPURasterNDVI`](https://luma.gl/docs/api-reference/experimental/gpu-raster/operations-pixel#ndvi) (`@luma.gl/graph/gpu-raster`) | `import {GPURasterNDVI} from '@luma.gl/graph/gpu-raster';` |
| - | `ST_Value`, `ST_ValueCount` | `rasterio.sample` | - | Sample raster values, Profile | [`GPURasterSampling`](/docs/api/gpu-raster/operations-analysis#gpurastersampling-and-gpurasterprofile), [`GPURasterProfile`](/docs/api/gpu-raster/operations-analysis#gpurastersampling-and-gpurasterprofile) (raster) | - |
| - | - | - | - | Weighted Overlay, Suitability | [`GPUWeightedOverlay`](/docs/api/gpu-raster/operations-analysis#gpurasterreclassify-gpuweightedoverlay-and-local-raster-operations) (raster) | - |
| - | - | `numpy.diff` change maps | - | Change detection | [`GPUChangeDetection`](/docs/api/gpu-raster/operations-analysis#gpuchangedetection) (raster) | - |
| - | - | `matplotlib.streamplot` | - | Streamlines, LIC | [`GPUStreamlines`](/docs/api/gpu-raster/operations-analysis#gpustreamlines), [`GPULineIntegralConvolution`](/docs/api/gpu-raster/operations-analysis#gpulineintegralconvolution), [`GPUParticleAdvection`](/docs/api/gpu-raster/operations-analysis#gpuparticleadvection) (raster) | - |
| - | - | - | - | Contrast stretch | [`GPURasterStretch`](/docs/api/gpu-raster/operations-analysis#gpurasterstretch) (raster) | - |

## Terrain

| turf | PostGIS | GeoPandas / Shapely | PySAL | QGIS / ArcGIS / CARTO | luma.gl | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| - | `ST_Slope`, `ST_Aspect`, `ST_HillShade` | `richdem.TerrainAttribute`, `xrspatial.slope` | - | Slope, Aspect, Hillshade | [`GPUTerrainDerivatives`](/docs/api/gpu-terrain/operators#gputerrainderivatives-gputerraincontours-gputerrainviewshed-gputerrainlineofsight-and-gputerraincumulativeviewshed) (terrain) | Horn slope, aspect and hillshade. |
| - | - | `richdem.TerrainAttribute` (curvature) | - | Profile and plan curvature | [`GPUTerrainCurvature`](/docs/api/gpu-terrain/operators#gputerraincurvature) (terrain) | - |
| - | - | `xrspatial.viewshed` | - | Viewshed, Visibility analysis | [`GPUTerrainViewshed`](/docs/api/gpu-terrain/operators#gputerrainderivatives-gputerraincontours-gputerrainviewshed-gputerrainlineofsight-and-gputerraincumulativeviewshed), [`GPUTerrainCumulativeViewshed`](/docs/api/gpu-terrain/operators#gputerrainderivatives-gputerraincontours-gputerrainviewshed-gputerrainlineofsight-and-gputerraincumulativeviewshed), [`GPUTerrainLineOfSight`](/docs/api/gpu-terrain/operators#gputerrainderivatives-gputerraincontours-gputerrainviewshed-gputerrainlineofsight-and-gputerraincumulativeviewshed) (terrain) | Raster result on planar, projected-metre grids with a per-frame observer. |
| - | - | `richdem.FlowAccumulation`, `pysheds` | - | r.watershed, Flow accumulation, Fill sinks | [`GPUTerrainFlow`](/docs/api/gpu-terrain/operators#gputerrainflow) (terrain) | D8 routing, depression filling with convergence flags. |
| - | - | `pysheds` catchments | - | Watershed basins, Strahler order, HAND | [`GPUTerrainWatersheds`](/docs/api/gpu-terrain/operators#gputerrainheightabovedrainage-gputerrainwatersheds-gputerrainstreamorder-and-gputerrainhydrologicindices), [`GPUTerrainStreamOrder`](/docs/api/gpu-terrain/operators#gputerrainheightabovedrainage-gputerrainwatersheds-gputerrainstreamorder-and-gputerrainhydrologicindices), [`GPUTerrainHeightAboveDrainage`](/docs/api/gpu-terrain/operators#gputerrainheightabovedrainage-gputerrainwatersheds-gputerrainstreamorder-and-gputerrainhydrologicindices), [`GPUTerrainHydrologicIndices`](/docs/api/gpu-terrain/operators#gputerrainheightabovedrainage-gputerrainwatersheds-gputerrainstreamorder-and-gputerrainhydrologicindices) (terrain) | - |
| - | - | - | - | Terrain Ruggedness Index, Roughness | [`GPUTerrainRuggedness`](/docs/api/gpu-terrain/operators#gputerrainruggedness-and-gputerrainvectorruggedness), [`GPUTerrainVectorRuggedness`](/docs/api/gpu-terrain/operators#gputerrainruggedness-and-gputerrainvectorruggedness) (terrain) | - |
| - | - | - | - | Topographic Position Index, Geomorphons | [`GPUTerrainTopographicPosition`](/docs/api/gpu-terrain/operators#gputerraintopographicposition-and-gputerrainweisslandforms), [`GPUTerrainWeissLandforms`](/docs/api/gpu-terrain/operators#gputerraintopographicposition-and-gputerrainweisslandforms), [`GPUGeomorphons`](/docs/api/gpu-terrain/operators#gpugeomorphons) (terrain) | - |
| - | - | - | - | Peaks, summits | [`GPUTerrainSummits`](/docs/api/gpu-terrain/operators#gputerrainsummits-gputerrainpeaksnap-gputerraincriticalpoints-and-gpuprofilepeaks), [`GPUTerrainPeakSnap`](/docs/api/gpu-terrain/operators#gputerrainsummits-gputerrainpeaksnap-gputerraincriticalpoints-and-gpuprofilepeaks), [`GPUTerrainCriticalPoints`](/docs/api/gpu-terrain/operators#gputerrainsummits-gputerrainpeaksnap-gputerraincriticalpoints-and-gpuprofilepeaks) (terrain) | - |
| - | - | - | - | Cast shadows, Solar radiation | [`GPUSolarShadowMask`](/docs/api/gpu-terrain/operators#gputerrainhorizon-gpusolarshadowmask-gputerraincastshadow-gpusolarirradiance-gpureliefshading-gputextureshading-and-gpusolarposition), [`GPUTerrainCastShadow`](/docs/api/gpu-terrain/operators#gputerrainhorizon-gpusolarshadowmask-gputerraincastshadow-gpusolarirradiance-gpureliefshading-gputextureshading-and-gpusolarposition), [`GPUSolarIrradiance`](/docs/api/gpu-terrain/operators#gputerrainhorizon-gpusolarshadowmask-gputerraincastshadow-gpusolarirradiance-gpureliefshading-gputextureshading-and-gpusolarposition), [`GPUTerrainHorizon`](/docs/api/gpu-terrain/operators#gputerrainhorizon-gpusolarshadowmask-gputerraincastshadow-gpusolarirradiance-gpureliefshading-gputextureshading-and-gpusolarposition) (terrain) | - |
| - | - | - | - | Terrain RGB decode, Fill sinks (spikes) | [`GPUTerrainRGBDecode`](/docs/api/gpu-terrain/operators#gputerrainrgbdecode-and-gputerrainspikerepair), [`GPUTerrainSpikeRepair`](/docs/api/gpu-terrain/operators#gputerrainrgbdecode-and-gputerrainspikerepair) (terrain) | - |
| - | - | - | - | Contours from DEM | [`GPUTerrainContours`](/docs/api/gpu-terrain/operators#gputerrainderivatives-gputerraincontours-gputerrainviewshed-gputerrainlineofsight-and-gputerraincumulativeviewshed) (terrain) | - |
| - | - | - | - | Relief shading, Multi-scale relief | [`GPUReliefShading`](/docs/api/gpu-terrain/operators#gputerrainhorizon-gpusolarshadowmask-gputerraincastshadow-gpusolarirradiance-gpureliefshading-gputextureshading-and-gpusolarposition), [`GPUMultiScaleRelief`](/docs/api/gpu-terrain/operators#gpusimplelocalrelief-gpumultiscalerelief-gpulocaldominance-and-gpureliefblend), [`GPUSimpleLocalRelief`](/docs/api/gpu-terrain/operators#gpusimplelocalrelief-gpumultiscalerelief-gpulocaldominance-and-gpureliefblend), [`GPULocalDominance`](/docs/api/gpu-terrain/operators#gpusimplelocalrelief-gpumultiscalerelief-gpulocaldominance-and-gpureliefblend), [`GPUReliefBlend`](/docs/api/gpu-terrain/operators#gpusimplelocalrelief-gpumultiscalerelief-gpulocaldominance-and-gpureliefblend) (terrain) | - |

## Recipes

Recipes chain contributors with `addXRecipe(graph, props)`. See [Recipes](/docs/api/gpu-spatial-analysis/recipes) for the chains and notes.

| Recipe | Replaces this workflow | Chain |
| --- | --- | --- |
| `addHotSpotAnalysisRecipe` | PySAL `G_Local` + `fdr` + map, ArcGIS Hot Spot Analysis, CARTO `ST_GETISORD_H3` | [`GPUPointToCell`](/docs/api/gpu-spatial-analysis/operators-cells#gpupointtocell-and-gpucellgeometry) (`@luma-contributors/analysis/gpu-spatial-analysis`), [`GPUCellAggregation`](/docs/api/gpu-spatial-analysis/operators-cells#gpucellaggregation-gpucellrollup-gpucellpyramid-and-gpucelllevelselection) (`@luma-contributors/analysis/gpu-spatial-analysis`), [`GPUNeighborSearch`](/docs/api/gpu-spatial-analysis/operators-weights#gpuneighborsearch) (`@luma-contributors/analysis/gpu-spatial-analysis`), [`GPUHotSpotAnalysis`](/docs/api/gpu-spatial-analysis/operators-statistics#gpuhotspotanalysis-and-gpulocalmoran) (`@luma-contributors/analysis/gpu-spatial-analysis`), [`GPULocalPermutationTest`](/docs/api/gpu-spatial-analysis/operators-statistics#gpulocalpermutationtest-and-gpuglobalpermutationtest) (`@luma-contributors/analysis/gpu-spatial-analysis`), [`GPUClassBreaks`](/docs/api/gpu-dataframe-analysis#gpucolumnquantiles-gpuclassbreaks-gpucolorscale-and-gpubivariateclassification) (`@luma-contributors/analysis/dataframe-analysis`), [`GPUColorScale`](/docs/api/gpu-dataframe-analysis#gpucolumnquantiles-gpuclassbreaks-gpucolorscale-and-gpubivariateclassification) (`@luma-contributors/analysis/dataframe-analysis`) |
| `addRateClusterMapRecipe` | Assunção–Reis standardized rates + local Moran by default; `analyze: 'smoothed'` for smoothing | [`GPUEmpiricalBayesRates`](/docs/api/gpu-spatial-analysis/operators-statistics#gpuempiricalbayesrates), [`GPUContiguityWeights`](/docs/api/gpu-spatial-analysis/operators-weights#gpucontiguityweights), [`GPULocalMoran`](/docs/api/gpu-spatial-analysis/operators-statistics#gpuhotspotanalysis-and-gpulocalmoran), [`GPULocalPermutationTest`](/docs/api/gpu-spatial-analysis/operators-statistics#gpulocalpermutationtest-and-gpuglobalpermutationtest) |
| `addPointsInPolygonsChoroplethRecipe` | Single-point layout: exclusive zone assignment; multipart layout: one contribution per matched polygon | [`GPUZonalStatistics`](/docs/api/gpu-spatial-analysis/operators-statistics#gpuzonalstatistics) (`@luma-contributors/analysis/gpu-spatial-analysis`) or [`GPUPointInPolygonJoin`](/docs/api/gpu-spatial-analysis/operators-joins#gpupointinpolygonjoin-and-gpunearestfeaturejoin) (`@luma-contributors/analysis/gpu-spatial-analysis`) + [`GPUGroupStatistics`](/docs/api/gpu-dataframe-analysis#gpugroupstatistics) (`@luma-contributors/analysis/dataframe-analysis`), then [`GPUClassBreaks`](/docs/api/gpu-dataframe-analysis#gpucolumnquantiles-gpuclassbreaks-gpucolorscale-and-gpubivariateclassification) (`@luma-contributors/analysis/dataframe-analysis`), [`GPUColorScale`](/docs/api/gpu-dataframe-analysis#gpucolumnquantiles-gpuclassbreaks-gpucolorscale-and-gpubivariateclassification) (`@luma-contributors/analysis/dataframe-analysis`) |
| `addSpaceTimeHotSpotsRecipe` | ArcGIS Emerging Hot Spot Analysis, CARTO `ST_SPACETIME_HOTSPOTS` | [`GPUCalendarBuckets`](/docs/api/gpu-dataframe-analysis#gpucalendarbuckets) (`@luma-contributors/analysis/dataframe-analysis`), [`GPUPeriodIndex`](/docs/api/gpu-dataframe-analysis#gpuperiodindex) (`@luma-contributors/analysis/dataframe-analysis`), [`GPUGroupStatistics`](/docs/api/gpu-dataframe-analysis#gpugroupstatistics) (`@luma-contributors/analysis/dataframe-analysis`), [`GPUEmergingHotSpots`](/docs/api/gpu-spatial-analysis/operators-statistics#gpuemerginghotspots) (`@luma-contributors/analysis/gpu-spatial-analysis`), optional [`GPUColorScale`](/docs/api/gpu-dataframe-analysis#gpucolumnquantiles-gpuclassbreaks-gpucolorscale-and-gpubivariateclassification) (`@luma-contributors/analysis/dataframe-analysis`) |
| `addClusterAndOutlineRecipe` | DBSCAN + `convex_hull` + area | [`GPUDBSCAN`](/docs/api/gpu-spatial-analysis/operators-statistics#gpudbscan), [`GPUGroupGeometry`](/docs/api/gpu-spatial-analysis/operators-statistics#gpugroupgeometry), [`GPUGroupConvexHull`](/docs/api/gpu-spatial-analysis/operators-statistics#gpugroupconvexhull), [`GPUGeometryMeasures`](/docs/api/gpu-spatial-analysis/operators-lines#gpugeometrymeasures-gpugeodesicpairs-and-gpugeodesicdestination) |
| `addSpatialRegressionRecipe` | `spreg.OLS` + `LMtests` + residual Moran, ArcGIS OLS and GWR | [`GPUOrdinaryLeastSquares`](/docs/api/gpu-spatial-analysis/operators-regression#gpuordinaryleastsquares), [`GPUSpatialRegressionDiagnostics`](/docs/api/gpu-spatial-analysis/operators-regression#gpuspatialregressiondiagnostics), [`GPULocalMoran`](/docs/api/gpu-spatial-analysis/operators-statistics#gpuhotspotanalysis-and-gpulocalmoran), [`GPUGeographicallyWeightedRegression`](/docs/api/gpu-spatial-analysis/operators-regression#gpugeographicallyweightedregression) |
| `import {addDriveTimeCatchmentRecipe} from '@luma-contributors/analysis/gpu-network';` | pgRouting `pgr_drivingDistance`, QGIS Service area, ArcGIS Service Area | [`GPUNetworkSnapping`](/docs/api/gpu-network/operators#gpunetworksnapping-gpunetworkcostmatrix-and-gpunetworkaccessibility) (`@luma-contributors/analysis/gpu-network`) for facilities and demand; [`GPUNetworkServiceAreas`](/docs/api/gpu-network/operators#gpunetworkpathextraction-gpunetworkserviceareas-gpunetworkneighborhood-and-gpunetworkanalyticscolumns) (`@luma-contributors/analysis/gpu-network`); `GPUSort` (`@luma.gl/graph/gpu-core`) sorts facilities by road for direct on-road costs; band adapters → [`GPUGroupStatistics`](/docs/api/gpu-dataframe-analysis#gpugroupstatistics) (`@luma-contributors/analysis/dataframe-analysis`). The optional polygon route uses [`GPUNetworkIsochrones`](/docs/api/gpu-network/operators#gpunetworkisochrones) (`@luma-contributors/analysis/gpu-network`), with a demand join only when requested via `isochrones.joinDemand`: [`GPUPointInPolygonJoin`](/docs/api/gpu-spatial-analysis/operators-joins#gpupointinpolygonjoin-and-gpunearestfeaturejoin) (`@luma-contributors/analysis/gpu-spatial-analysis`). |
| `addStraightLineCatchmentsRecipe` | QGIS Proximity + Zonal statistics, Thiessen catchments | [`GPUDistanceField`](/docs/api/gpu-raster/operations-analysis#gpudistancefield) (`@luma-contributors/analysis/raster-analysis`), [`GPURasterZonalStatistics`](/docs/api/gpu-raster/operations-analysis#gpurasterzonalstatistics) (`@luma-contributors/analysis/raster-analysis`) |
| `addChangeOfSupportRecipe` | `tobler.area_interpolate`, ArcGIS Apportion Polygon | [`GPUPolygonRasterization`](/docs/api/gpu-raster/operations-analysis#gpupolygonrasterization-and-gpurasterjoin) (`@luma-contributors/analysis/raster-analysis`), [`GPUArealInterpolation`](/docs/api/gpu-spatial-analysis/operators-accessibility#gpuarealinterpolation) (`@luma-contributors/analysis/gpu-spatial-analysis`), [`GPUSpatialLag`](/docs/api/gpu-spatial-analysis/operators-weights#gpuspatiallag) (`@luma-contributors/analysis/gpu-spatial-analysis`) |
| `addFleetDwellRecipe` | Slow-step stop runs + exclusive zone assignment | [`GPUTrajectoryMetrics`](/docs/api/gpu-spatial-analysis/operators-lines#gputrajectorymetrics) (`@luma-contributors/analysis/gpu-spatial-analysis`), [`GPUPointInPolygonJoin`](/docs/api/gpu-spatial-analysis/operators-joins#gpupointinpolygonjoin-and-gpunearestfeaturejoin) (`@luma-contributors/analysis/gpu-spatial-analysis`), [`GPUGroupStatistics`](/docs/api/gpu-dataframe-analysis#gpugroupstatistics) (`@luma-contributors/analysis/dataframe-analysis`) |
| `addFleetDwellZoneEventsRecipe` | `movingpandas` zone events, geofence dwell | [`GPUZoneEvents`](/docs/api/gpu-spatial-analysis/operators-lines#gpuzoneevents) (`@luma-contributors/analysis/gpu-spatial-analysis`), [`GPUGroupStatistics`](/docs/api/gpu-dataframe-analysis#gpugroupstatistics) (`@luma-contributors/analysis/dataframe-analysis`) |
| `addPeriodComparisonRecipe` | Before and after change map (CARTO, ArcGIS Compare) | [`GPUCellAggregation`](/docs/api/gpu-spatial-analysis/operators-cells#gpucellaggregation-gpucellrollup-gpucellpyramid-and-gpucelllevelselection) (`@luma-contributors/analysis/gpu-spatial-analysis`), [`GPUCellTableCompare`](/docs/api/gpu-spatial-analysis/operators-cells#gpucelltablecompare) (`@luma-contributors/analysis/gpu-spatial-analysis`), [`GPUClassBreaks`](/docs/api/gpu-dataframe-analysis#gpucolumnquantiles-gpuclassbreaks-gpucolorscale-and-gpubivariateclassification) (`@luma-contributors/analysis/dataframe-analysis`), [`GPUColorScale`](/docs/api/gpu-dataframe-analysis#gpucolumnquantiles-gpuclassbreaks-gpucolorscale-and-gpubivariateclassification) (`@luma-contributors/analysis/dataframe-analysis`) |

## See also

- [GPU Spatial Analysis](/docs/api/gpu-spatial-analysis)
- [GPU Network](/docs/api/gpu-network)
- [GPU Terrain](/docs/api/gpu-terrain)
- [GPURaster operations](/docs/api/gpu-raster/operations)
- [GPU Dataframe analysis contributors](/docs/api/gpu-dataframe-analysis)

## Great-circle linear-referencing fragment

Import the contributor from its public spatial entry. The caller supplies packed graph views:
`positions` contains longitude/latitude vertices, `lineOffsets` delimits lines, and `queryPositions`
contains longitude/latitude query points. `radius` is one float32 row in meters; `measures` has one
row per query, and `candidateOverflow` is one uint32 row. Allocate the views in the caller's graph
before this step. Great-circle distances use the default mean Earth sphere; the ellipsoidal
geometry-measures declaration in the table above applies specifically to `GPUGeometryMeasures`.

```ts
import {GPULinearReferencing} from '@luma-contributors/analysis/gpu-spatial-analysis';
import type {GraphDataView} from '@luma.gl/graph/gpu-core';

declare const positions: GraphDataView<'float32x2'>;
declare const lineOffsets: GraphDataView<'uint32'>;
declare const queryPositions: GraphDataView<'float32x2'>;
declare const radius: GraphDataView<'float32'>;
declare const measures: GraphDataView<'float32'>;
declare const candidateOverflow: GraphDataView<'uint32'>;

const linearReferencing = new GPULinearReferencing({
  positions,
  lineOffsets,
  queryPositions,
  radius,
  spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'great-circle'},
  output: {measures, status: {candidateOverflow}}
});
```

Add `linearReferencing` to the caller-owned `GPUCommandGraph` before compiling. This fragment
checks the public option and view types; it does not allocate, submit or read back a computation.
