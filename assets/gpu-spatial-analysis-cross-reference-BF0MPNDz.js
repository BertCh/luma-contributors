var e={slug:`api/gpu-spatial-analysis-cross-reference`,title:`GPU spatial analysis cross-reference`,html:`<h1 id="gpu-spatial-analysis-cross-reference">GPU spatial analysis cross-reference</h1>
<p>:::caution Experimental
The contributors on this page live in experimental entry points. APIs may change between releases
without a deprecation period.
:::</p>
<p>This page maps tool names from turf, PostGIS, GeoPandas and Shapely, PySAL (esda, libpysal, spreg,
pointpats, giddy, segregation, tobler, mapclassify, spaghetti and friends), QGIS, ArcGIS and CARTO to
the luma.gl analysis contributor that does the same job. It is organised by task. Search the page for
the name you already know.</p>
<h2 id="how-to-read-the-tables">How to read the tables</h2>
<ul>
<li>Contributors are classes added to a caller-owned <code>GPUCommandGraph</code> with <code>graph.add(contributor)</code>.
Recipes are <code>addXRecipe(graph, props)</code> builder functions that chain several contributors. Each
contributor name links to its section in the reference page.</li>
<li>Entry points: no suffix means <code>@luma-contributors/analysis/gpu-spatial-analysis</code>. <code>(network)</code> is
<code>gpu-network</code>, <code>(raster)</code> is <code>gpu-raster</code>, <code>(terrain)</code> is <code>gpu-terrain</code> and <code>(dataframe)</code> is
<code>gpu-dataframe</code>.</li>
<li>Every contributor runs on the GPU with bounded, caller-sized outputs and an overflow or count word
instead of a readback. The CPU libraries return unbounded geometry and tables. Where a result is
raster-approximate, capacity-bounded or follows a specific tie rule, the Notes column says so.</li>
<li>Partial matches name the exact supported subset and the external tool to use for the remainder.</li>
<li>A dash means the tool has no equivalent in that column's family.</li>
<li>CARTO names are Analytics Toolbox functions.</li>
</ul>
<h2 id="joins-and-predicates">Joins and predicates</h2>
<table>
<thead>
<tr>
<th>turf</th>
<th>PostGIS</th>
<th>GeoPandas / Shapely</th>
<th>PySAL</th>
<th>QGIS / ArcGIS / CARTO</th>
<th>luma.gl</th>
<th>Notes</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>booleanPointInPolygon</code>, <code>pointsWithinPolygon</code>, <code>tag</code></td>
<td><code>ST_Within</code>, <code>ST_Contains</code> (point side)</td>
<td><code>sjoin</code> (points to polygons), <code>GeoSeries.within</code></td>
<td>-</td>
<td>Join attributes by location, Count points in polygon, Spatial Join</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-joins#gpupointinpolygonjoin-and-gpunearestfeaturejoin"><code>GPUPointInPolygonJoin</code></a></td>
<td>Exact orientation fallback for near-boundary points. Each point gets the containing feature with the smallest row. Capacity-bounded candidates and matches with one overflow flag.</td>
</tr>
<tr>
<td><code>booleanIntersects</code>, <code>booleanContains</code>, <code>booleanWithin</code>, <code>booleanOverlap</code>, <code>booleanCrosses</code>, <code>booleanTouches</code></td>
<td><code>ST_Intersects</code>, <code>ST_Contains</code>, <code>ST_Within</code>, <code>ST_Covers</code>, <code>ST_Touches</code>, <code>ST_Crosses</code>, <code>ST_Overlaps</code>, <code>ST_Equals</code>, <code>ST_Relate</code></td>
<td><code>sjoin(predicate=...)</code>, <code>shapely.relate</code></td>
<td>-</td>
<td>Extract by location, Select by location, Join attributes by location</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-joins#gpuspatialpredicatejoin"><code>GPUSpatialPredicateJoin</code></a></td>
<td>All nine point, line and polygon combinations. OGC boundary rules and a DE-9IM <code>relate</code> pattern. Planar coordinates. The <code>dwithin</code> distance and the <code>relate</code> pattern can be per-frame views.</td>
</tr>
<tr>
<td>-</td>
<td><code>&#x26;&#x26;</code> (bounding-box overlap)</td>
<td><code>STRtree.query</code></td>
<td>-</td>
<td>Spatial index</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-joins#gpuspatialjoincandidates"><code>GPUSpatialJoinCandidates</code></a></td>
<td>The bounding-box stage alone, without refinement.</td>
</tr>
<tr>
<td>-</td>
<td>prepared geometry</td>
<td><code>shapely.prepare</code></td>
<td>-</td>
<td>-</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-joins#gpuspatialjoinprepared"><code>GPUSpatialJoinPrepared</code></a></td>
<td>Reuse one prepared right-hand side across many left-hand sets.</td>
</tr>
<tr>
<td><code>pointsWithinPolygon</code> + <code>collect</code></td>
<td><code>ST_Within</code> + <code>GROUP BY</code></td>
<td><code>sjoin</code> then <code>groupby().agg</code></td>
<td>-</td>
<td>Join attributes by location (summary), Zonal statistics (vector), Summarize Within</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-statistics#gpuzonalstatistics"><code>GPUZonalStatistics</code></a>, <a href="https://luma.gl/docs/api-reference/experimental/gpu-dataframe-analysis#gpugroupstatistics"><code>GPUGroupStatistics</code></a> (dataframe); recipe <code>addPointsInPolygonsChoroplethRecipe</code></td>
<td>Counts, sums, means, unbiased variances and extents per polygon with a deterministic sum order (densities need spherical <code>areas</code> in longitude/latitude). The recipe adds class breaks and colours.</td>
</tr>
<tr>
<td>-</td>
<td><code>ST_DWithin</code></td>
<td><code>sjoin(predicate='dwithin')</code>, <code>buffer</code> then <code>sjoin</code></td>
<td><code>DistanceBand</code></td>
<td>Select within distance</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-joins#gpubufferselection"><code>GPUBufferSelection</code></a>, <a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-joins#gpuspatialpredicatejoin"><code>GPUSpatialPredicateJoin</code></a> (<code>'dwithin'</code>)</td>
<td>The distance is an inclusive per-frame planar distance, not a materialised buffer polygon.</td>
</tr>
<tr>
<td>-</td>
<td><code>ST_Within</code> per pick region</td>
<td>-</td>
<td>-</td>
<td>Select by rectangle, circle or polygon</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-statistics#gpuregionstatistics"><code>GPURegionMask</code></a>, <a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-statistics#gpuregionstatistics"><code>GPUPickRegionMask</code></a>, <a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-statistics#gpuregionstatistics"><code>GPURegionStatistics</code></a></td>
<td>Interactive rectangle, radius or polygon selection with per-frame bounds. Rectangle and polygon accept longitude/latitude (straight edges in degrees); radius is planar only.</td>
</tr>
<tr>
<td><code>intersect</code>, <code>lineIntersect</code>, <code>kinks</code></td>
<td><code>ST_Intersection</code> (crossing points), <code>ST_IsSimple</code></td>
<td><code>shapely.intersection</code> (points only)</td>
<td>-</td>
<td>Line intersections</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-validity#gpusegmentintersection"><code>GPUSegmentIntersection</code></a></td>
<td>Crossing points and kinds between segments, not clipped geometry.</td>
</tr>
</tbody>
</table>
<h2 id="nearest-neighbours-and-distances">Nearest neighbours and distances</h2>
<table>
<thead>
<tr>
<th>turf</th>
<th>PostGIS</th>
<th>GeoPandas / Shapely</th>
<th>PySAL</th>
<th>QGIS / ArcGIS / CARTO</th>
<th>luma.gl</th>
<th>Notes</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>nearestPoint</code></td>
<td><code>ST_Distance</code> KNN (<code>&#x3C;-></code>), <code>ST_ClosestPoint</code></td>
<td><code>sjoin_nearest</code>, <code>STRtree.nearest</code></td>
<td>-</td>
<td>Join attributes by nearest, Near, Generate Near Table</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-joins#gpupointinpolygonjoin-and-gpunearestfeaturejoin"><code>GPUNearestFeatureJoin</code></a></td>
<td>Nearest point or segment feature within a per-frame radius. The tie rule is set by <code>GPUNearestTieMode</code>. <code>exclusive</code> skips the query's own ID, <code>neighborQueryPoints</code> gives <code>shortest_line</code> with the foot point, and <code>onAttribute</code> restricts to key-equal rows.</td>
</tr>
<tr>
<td>-</td>
<td>-</td>
<td><code>sjoin_nearest</code> with weights</td>
<td><code>KNN</code>, <code>DistanceBand</code></td>
<td>-</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-joins#gpunearestfeatureweights"><code>GPUNearestFeatureWeights</code></a></td>
<td>Writes the nearest-feature pairs as spatial weights.</td>
</tr>
<tr>
<td><code>nearestPointOnLine</code></td>
<td><code>ST_LineLocatePoint</code>, <code>ST_ClosestPoint</code> (lines)</td>
<td><code>shapely.project</code>, <code>shapely.shortest_line</code></td>
<td>-</td>
<td>Snap points to geometries</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-lines#gpulinearreferencing-and-gpulinelocate"><code>GPULinearReferencing</code></a></td>
<td>Measure along a line and the snapped point. <code>coordinateSystem: 'spherical'</code> gives meters on a sphere, and <code>normalizedMeasures</code> gives <code>normalized=True</code>.</td>
</tr>
<tr>
<td><code>distance</code></td>
<td><code>ST_Distance</code></td>
<td><code>GeoSeries.distance</code></td>
<td><code>KNN</code>, <code>DistanceBand</code></td>
<td>Distance matrix, Generate Near Table</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-weights#gpuneighborsearch"><code>GPUNeighborSearch</code></a></td>
<td>Fixed-capacity kNN or radius neighbours from a grid index. Ties follow a deterministic row order.</td>
</tr>
<tr>
<td><code>distance</code>, <code>bearing</code>, <code>midpoint</code></td>
<td><code>ST_Distance(geography)</code>, <code>ST_Azimuth</code>, <code>ST_LineInterpolatePoint</code></td>
<td><code>pyproj.Geod.inv</code></td>
<td>-</td>
<td>Distance (ellipsoidal)</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-lines#gpugeometrymeasures-gpugeodesicpairs-and-gpugeodesicdestination"><code>GPUGeodesicPairs</code></a></td>
<td>Geodesic distance, azimuth and midpoint per pair. <code>model: 'rhumb'</code> gives turf <code>rhumbDistance</code> and <code>rhumbBearing</code>.</td>
</tr>
<tr>
<td><code>destination</code></td>
<td><code>ST_Project</code></td>
<td><code>pyproj.Geod.fwd</code></td>
<td>-</td>
<td>-</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-lines#gpugeometrymeasures-gpugeodesicpairs-and-gpugeodesicdestination"><code>GPUGeodesicDestination</code></a></td>
<td>Point at a bearing and distance. <code>model: 'rhumb'</code> gives <code>rhumbDestination</code>.</td>
</tr>
<tr>
<td>-</td>
<td><code>ST_MaxDistance</code> (Sedona)</td>
<td><code>shapely.hausdorff_distance</code>, <code>frechet_distance</code> (with <code>densify</code>)</td>
<td>-</td>
<td>-</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-lines#gputracksimilarity"><code>GPUTrackSimilarity</code></a></td>
<td>Track-to-track similarity metrics. Hausdorff is vertex to segment, as GEOS, so it changed from earlier vertex-to-vertex values.</td>
</tr>
<tr>
<td>-</td>
<td>-</td>
<td>-</td>
<td><code>pointpats.PointPattern.nnd</code>, <code>ripley</code> indices</td>
<td>Nearest neighbour analysis, Average Nearest Neighbor</td>
<td><a href="https://luma.gl/docs/api-reference/experimental/gpu-dataframe-analysis#gpuvariogram-gpuspatialcorrelogram-gpuripley-gpuripleydistancefunctions-and-gpupointpatternindices"><code>GPUPointPatternIndices</code></a> (dataframe)</td>
<td>Nearest-neighbour index with Donnelly edge correction and p-value, plus quadrat chi-square test.</td>
</tr>
<tr>
<td>-</td>
<td><code>pgr_dijkstraCost</code></td>
<td><code>spaghetti.Network.allneighbordistances</code>, <code>pandana</code></td>
<td><code>spaghetti</code></td>
<td>OD cost matrix, Distance matrix (network)</td>
<td><a href="/luma-contributors/docs/api/gpu-network/operators#gpunetworksnapping-gpunetworkcostmatrix-and-gpunetworkaccessibility"><code>GPUNetworkCostMatrix</code></a> (network)</td>
<td>Network cost between origins and destinations over a CSR network. Capacity-bounded.</td>
</tr>
</tbody>
</table>
<h2 id="spatial-weights">Spatial weights</h2>
<table>
<thead>
<tr>
<th>turf</th>
<th>PostGIS</th>
<th>GeoPandas / Shapely</th>
<th>PySAL</th>
<th>QGIS / ArcGIS / CARTO</th>
<th>luma.gl</th>
<th>Notes</th>
</tr>
</thead>
<tbody>
<tr>
<td>-</td>
<td>-</td>
<td><code>touches</code>, <code>sjoin</code> for neighbours</td>
<td><code>libpysal.weights.Queen</code>, <code>Rook</code> (<code>from_dataframe</code>)</td>
<td>Generate Spatial Weights Matrix (contiguity)</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-weights#gpucontiguityweights"><code>GPUContiguityWeights</code></a></td>
<td>Queen and rook from polygon geometry. Fixed neighbour capacity.</td>
</tr>
<tr>
<td>-</td>
<td>-</td>
<td>-</td>
<td><code>libpysal.weights.lat2W</code>, <code>block_weights</code> on grids</td>
<td>Generate Spatial Weights Matrix (grid)</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-weights#gpulatticeweights"><code>GPULatticeWeights</code></a></td>
<td>Raster or lattice neighbourhoods for gridded data.</td>
</tr>
<tr>
<td>-</td>
<td><code>ST_DWithin</code> self join</td>
<td><code>sjoin_nearest</code></td>
<td><code>KNN</code>, <code>DistanceBand</code>, <code>Kernel</code>, <code>Kernel(fixed=False)</code></td>
<td>Generate Spatial Weights Matrix (fixed distance, k nearest), Generate Network Spatial Weights</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-weights#gpuneighborsearch"><code>GPUNeighborSearch</code></a></td>
<td>kNN, radius and kernel weights from a grid index. Capacity-bounded rows.</td>
</tr>
<tr>
<td>-</td>
<td>-</td>
<td>-</td>
<td><code>W.transform = 'R'</code>, <code>'B'</code>, <code>'D'</code>, <code>'V'</code>, <code>W.symmetrize</code></td>
<td>Row standardisation</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-weights#gpuspatialweightstransform"><code>GPUSpatialWeightsTransform</code></a></td>
<td>Row, binary, symmetrise and kernel transforms.</td>
</tr>
<tr>
<td>-</td>
<td>-</td>
<td>-</td>
<td><code>weights.set_operations</code> (<code>w_union</code>, <code>w_intersection</code>, <code>w_difference</code>, <code>w_symmetric_difference</code>), <code>higher_order</code>, <code>w_subset</code>, <code>block_weights</code></td>
<td>-</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-weights#gpuspatialweightsalgebra"><code>GPUSpatialWeightsAlgebra</code></a></td>
<td>Set operations, higher order, subset and block weights.</td>
</tr>
<tr>
<td>-</td>
<td>-</td>
<td>-</td>
<td><code>W.s0</code>, <code>.s1</code>, <code>.s2</code>, <code>.cardinalities</code>, <code>.islands</code>, <code>.asymmetry()</code></td>
<td>-</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-weights#gpuspatialweightssummary"><code>GPUSpatialWeightsSummary</code></a></td>
<td>Moments, cardinalities and islands.</td>
</tr>
<tr>
<td>-</td>
<td>-</td>
<td>-</td>
<td><code>W.sparse.T</code></td>
<td>-</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-weights#gpuspatialweightstranspose"><code>GPUSpatialWeightsTranspose</code></a></td>
<td>Transposed (reverse) weights.</td>
</tr>
<tr>
<td>-</td>
<td>-</td>
<td>-</td>
<td><code>lag_spatial</code></td>
<td>-</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-weights#gpuspatiallag"><code>GPUSpatialLag</code></a></td>
<td><code>W y</code> for any weights.</td>
</tr>
<tr>
<td>-</td>
<td>-</td>
<td>-</td>
<td><code>libpysal.weights.lag_categorical</code>, <code>momepy.describe_agg</code></td>
<td>-</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-statistics#gpuneighborhoodsummary"><code>GPUNeighborhoodSummary</code></a></td>
<td>Neighbourhood statistics and categorical lag.</td>
</tr>
<tr>
<td>-</td>
<td>-</td>
<td>-</td>
<td><code>mapclassify.greedy</code></td>
<td>Topological coloring (QGIS)</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-weights#gpumapcoloring"><code>GPUMapColoring</code></a></td>
<td>Greedy parallel colouring (Jones-Plassmann, seeded hashed priorities, lowest ID on ties) of a symmetric weights graph. Rows left unfinished are flagged.</td>
</tr>
<tr>
<td>-</td>
<td><code>pgr_dijkstraCost</code></td>
<td>-</td>
<td><code>spaghetti.Network</code> distance weights</td>
<td>Generate Network Spatial Weights</td>
<td><a href="/luma-contributors/docs/api/gpu-network/operators#gpunetworksnapping-gpunetworkcostmatrix-and-gpunetworkaccessibility"><code>GPUNetworkCostMatrix</code></a> (network)</td>
<td>Network cost between locations, which you threshold into weights.</td>
</tr>
</tbody>
</table>
<h2 id="spatial-autocorrelation-and-hot-spots">Spatial autocorrelation and hot spots</h2>
<table>
<thead>
<tr>
<th>turf</th>
<th>PostGIS</th>
<th>GeoPandas / Shapely</th>
<th>PySAL</th>
<th>QGIS / ArcGIS / CARTO</th>
<th>luma.gl</th>
<th>Notes</th>
</tr>
</thead>
<tbody>
<tr>
<td>-</td>
<td>-</td>
<td>-</td>
<td><code>esda.Moran</code>, <code>Geary</code>, <code>G</code> (General G), <code>Moran_BV</code>, <code>Join_Counts</code></td>
<td>Spatial Autocorrelation (Global Moran's I), High/Low Clustering, CARTO <code>ST_MORANS_I_H3</code></td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-statistics#gpuglobalspatialstatistics"><code>GPUGlobalSpatialStatistics</code></a></td>
<td>Analytic statistics with z-scores.</td>
</tr>
<tr>
<td>-</td>
<td>-</td>
<td>-</td>
<td><code>esda.Moran</code> <code>.p_sim</code>, <code>.z_sim</code>, <code>.seed</code></td>
<td>-</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-statistics#gpulocalpermutationtest-and-gpuglobalpermutationtest"><code>GPUGlobalPermutationTest</code></a></td>
<td>Conditional or global permutation tests with a deterministic stream.</td>
</tr>
<tr>
<td>-</td>
<td>-</td>
<td>-</td>
<td><code>esda.Moran_Local</code>, <code>Moran_Local_BV</code></td>
<td>Cluster and Outlier Analysis (Anselin Local Moran's I), CARTO <code>ST_LOCALMORANS_H3</code></td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-statistics#gpuhotspotanalysis-and-gpulocalmoran"><code>GPULocalMoran</code></a></td>
<td>Quadrants and analytic p-values. <code>quadrantGating</code> selects analytic or permutation gating.</td>
</tr>
<tr>
<td>-</td>
<td>-</td>
<td>-</td>
<td><code>esda.G_Local</code> (<code>Gi</code>, <code>Gi*</code>)</td>
<td>Hot Spot Analysis (Getis-Ord Gi*), CARTO <code>ST_GETISORD_H3</code></td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-statistics#gpuhotspotanalysis-and-gpulocalmoran"><code>GPUHotSpotAnalysis</code></a></td>
<td>Gi and Gi* z-scores with optional false discovery rate.</td>
</tr>
<tr>
<td>-</td>
<td>-</td>
<td>-</td>
<td><code>esda.Moran_Local</code> <code>.p_sim</code>, <code>G_Local</code> <code>.p_sim</code></td>
<td>-</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-statistics#gpulocalpermutationtest-and-gpuglobalpermutationtest"><code>GPULocalPermutationTest</code></a></td>
<td>Permutation p-values for local Moran, Gi and Gi*. Capped by <code>maximumPermutations</code>.</td>
</tr>
<tr>
<td>-</td>
<td>-</td>
<td>-</td>
<td><code>esda.fdr</code></td>
<td>Apply False Discovery Rate correction</td>
<td><code>multipleTesting</code> option of <a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-statistics#gpuhotspotanalysis-and-gpulocalmoran"><code>GPUHotSpotAnalysis</code></a>, <a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-statistics#gpuhotspotanalysis-and-gpulocalmoran"><code>GPULocalMoran</code></a>, <a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-statistics#gpulocalpermutationtest-and-gpuglobalpermutationtest"><code>GPULocalPermutationTest</code></a></td>
<td>Benjamini-Hochberg style control.</td>
</tr>
<tr>
<td>-</td>
<td>-</td>
<td>-</td>
<td><code>esda.Moran_Rate</code>, <code>Moran_Local_Rate</code>, <code>smoothing.Empirical_Bayes</code>, <code>assuncao_rate</code></td>
<td>Empirical Bayesian smoothing (GeoDa)</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-statistics#gpuempiricalbayesrates"><code>GPUEmpiricalBayesRates</code></a>, <a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-statistics#gpuspatialempiricalbayesrates"><code>GPUSpatialEmpiricalBayesRates</code></a>; recipe <code>addRateClusterMapRecipe</code></td>
<td>Global and spatial empirical Bayes rates. The recipe chains rates, weights, local Moran and quadrant colours.</td>
</tr>
<tr>
<td>-</td>
<td>-</td>
<td>-</td>
<td><code>esda.Moran</code> correlogram (<code>spatial_correlogram</code>)</td>
<td>Incremental Spatial Autocorrelation</td>
<td><a href="https://luma.gl/docs/api-reference/experimental/gpu-dataframe-analysis#gpuvariogram-gpuspatialcorrelogram-gpuripley-gpuripleydistancefunctions-and-gpupointpatternindices"><code>GPUSpatialCorrelogram</code></a> (dataframe)</td>
<td>Statistic across distance bands.</td>
</tr>
<tr>
<td>-</td>
<td>-</td>
<td>-</td>
<td><code>pointpats.k_test</code>, <code>Ripley K, F, G, J</code></td>
<td>Multi-Distance Spatial Cluster Analysis (Ripley's K)</td>
<td><a href="https://luma.gl/docs/api-reference/experimental/gpu-dataframe-analysis#gpuvariogram-gpuspatialcorrelogram-gpuripley-gpuripleydistancefunctions-and-gpupointpatternindices"><code>GPURipley</code></a>, <a href="https://luma.gl/docs/api-reference/experimental/gpu-dataframe-analysis#gpuvariogram-gpuspatialcorrelogram-gpuripley-gpuripleydistancefunctions-and-gpupointpatternindices"><code>GPURipleyDistanceFunctions</code></a> (dataframe)</td>
<td>Planar edge corrections only (rectangular window). <code>GPURipleyDistanceFunctions</code> adds Kaplan-Meier and Hanisch corrections (spatstat <code>km</code>, <code>han</code>). <code>GPURipleyEnvelope</code> adds pointwise and global Monte Carlo CSR envelopes with a MAD p-value (spatstat <code>envelope</code>). Bounded distance bins.</td>
</tr>
<tr>
<td>-</td>
<td>-</td>
<td>-</td>
<td><code>pointpats.Knox</code>, <code>Mantel</code></td>
<td>Space-Time Knox</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-dynamics#gpuknoxtest-and-gpumanteltest"><code>GPUKnoxTest</code></a>, <a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-dynamics#gpuknoxtest-and-gpumanteltest"><code>GPUMantelTest</code></a></td>
<td>Permutation based space-time tests. <code>GPUMantelTest</code> takes the pointpats <code>mantel</code> transform <code>(x + constant) ** power</code> as its compile-time <code>distanceTransform</code> (<code>scon</code>, <code>spow</code>) and <code>timeTransform</code> (<code>tcon</code>, <code>tpow</code>) options.</td>
</tr>
<tr>
<td>-</td>
<td>-</td>
<td>-</td>
<td>SaTScan (Kulldorff), R <code>smerc</code>, <code>rsatscan</code></td>
<td>CARTO <code>DETECT_SPACETIME_ANOMALIES</code></td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-dynamics#gpuspatialscanstatistic"><code>GPUSpatialScanStatistic</code></a></td>
<td>Poisson space-time scan with cylinders, secondary clusters and Monte Carlo p-values. Use SaTScan or <code>smerc</code> for other probability models.</td>
</tr>
<tr>
<td>-</td>
<td>-</td>
<td>-</td>
<td><code>esda.Geary_Local</code>; no package counterpart for the two edge statistics (<code>esda.Spatial_Pearson</code> is Lee's L, see <code>leesL</code> in <code>GPUGlobalPermutationTest</code>; <code>esda.Gamma</code> is an uncentered sum)</td>
<td>-</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-statistics#additional-spatial-statistics-and-bounded-optimization"><code>GPUSpatialPearson</code></a>, <a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-statistics#additional-spatial-statistics-and-bounded-optimization"><code>GPULocalGeary</code></a>, <a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-statistics#additional-spatial-statistics-and-bounded-optimization"><code>GPUGammaStatistic</code></a></td>
<td><code>GPUSpatialPearson</code> is the edge-weighted correlation of <code>x_i</code> with <code>y_j</code>; <code>GPUGammaStatistic</code> is the edge-weighted mean of a centered cross-product, squared or absolute difference. The adapters name statistics of <code>GPUGlobalPermutationTest</code> and <code>GPULocalPermutationTest</code>: exact for local Geary and squared-difference Gamma; the cross-product Gamma and Pearson adapters run Moran and bivariate Moran tests, the same ordering only on regular weights.</td>
</tr>
<tr>
<td>-</td>
<td>-</td>
<td>-</td>
<td>PySAL Gi* + FDR + map</td>
<td>Hot Spot Analysis (end to end)</td>
<td>recipe <code>addHotSpotAnalysisRecipe</code></td>
<td>Points or lattice source, Gi* and permutation, class breaks and colours.</td>
</tr>
<tr>
<td>-</td>
<td>-</td>
<td>-</td>
<td><code>pysal</code> emerging hot spots workflow</td>
<td>Emerging Hot Spot Analysis, CARTO <code>ST_SPACETIME_HOTSPOTS</code></td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-statistics#gpuemerginghotspots"><code>GPUEmergingHotSpots</code></a>; recipe <code>addSpaceTimeHotSpotsRecipe</code></td>
<td>Space-time cube trend and Gi* categories. The recipe aggregates a dense cube first.</td>
</tr>
<tr>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>Compare two periods (change map)</td>
<td>recipe <code>addPeriodComparisonRecipe</code></td>
<td>Two cell tables compared over a union mask, then class breaks.</td>
</tr>
</tbody>
</table>
<h2 id="kepler-and-geoda-analysis-recipes">Kepler and GeoDa analysis recipes</h2>
<p>These are compute operations for host integration. Their contracts describe bounded behavior,
including intentional differences from the pinned GeoDa implementations.</p>
<table>
<thead>
<tr>
<th>Analysis</th>
<th>luma.gl</th>
<th>Contract boundary</th>
</tr>
</thead>
<tbody>
<tr>
<td>Selected-quantile LISA</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/quantile-lisa">GPUQuantileLISA</a></td>
<td>Single variable, type-5 quantiles and conditional binary join counts. Uniform Philox sampling; legacy directed p-value limitations are explicit.</td>
</tr>
<tr>
<td>Dorling cartogram</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/gpu-dorling-cartogram">GPUDorlingCartogram</a></td>
<td>Planar circle relaxation with origin attraction, overlap and convergence diagnostics. At most 1024 circles/256 passes; heuristic layout.</td>
</tr>
<tr>
<td>Exact natural breaks</td>
<td><a href="https://luma.gl/docs/api-reference/experimental/gpu-dataframe-analysis">GPUClassBreaks</a></td>
<td>Opt-in exact dynamic programming for a normalized float32 objective, at most 8192 rows/64 classes, with recomputation above 1024 rows to avoid quadratic storage. Explicit monotone mode uses approximate quantized costs for larger inputs; approximate default preserved.</td>
</tr>
<tr>
<td>Variable standardization</td>
<td><a href="/luma-contributors/docs/api/variable-standardization">GPUVariableStandardization</a></td>
<td>Deviation, z-score, mean absolute deviation and two range methods; explicit masks, constants and sample/population conventions.</td>
</tr>
<tr>
<td>Excess risk and spatial rates</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/rate-mapping">addExcessRiskRecipe and addSpatialRateRecipe</a></td>
<td>Pooled expected counts and binary-neighborhood ratio of sums; existing EB smoothing remains available.</td>
</tr>
<tr>
<td>SQL data preparation</td>
<td><a href="https://luma.gl/docs/api-reference/experimental/gpu-sql">LuSQLContext</a></td>
<td>Quoted identifiers and bounded single-key JOIN USING with left projections. Does not supply general DuckDB compatibility.</td>
</tr>
</tbody>
</table>
<h2 id="regression">Regression</h2>
<table>
<thead>
<tr>
<th>turf</th>
<th>PostGIS</th>
<th>GeoPandas / Shapely</th>
<th>PySAL</th>
<th>QGIS / ArcGIS / CARTO</th>
<th>luma.gl</th>
<th>Notes</th>
</tr>
</thead>
<tbody>
<tr>
<td>-</td>
<td>-</td>
<td>-</td>
<td><code>spreg.OLS</code></td>
<td>Ordinary Least Squares (Generalized Linear Regression)</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-regression#gpuordinaryleastsquares"><code>GPUOrdinaryLeastSquares</code></a></td>
<td>Bounded regressor count.</td>
</tr>
<tr>
<td>-</td>
<td>-</td>
<td>-</td>
<td><code>spreg.diagnostics_sp.LMtests</code>, <code>MoranRes</code></td>
<td>Spatial Autocorrelation of residuals</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-regression#gpuspatialregressiondiagnostics"><code>GPUSpatialRegressionDiagnostics</code></a>; recipe <code>addSpatialRegressionRecipe</code></td>
<td>LM tests and residual Moran. The recipe needs a finite residual on every row.</td>
</tr>
<tr>
<td>-</td>
<td>-</td>
<td>-</td>
<td><code>spreg.GM_Lag</code></td>
<td>-</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-regression#gpuspatialtwostageleastsquares"><code>GPUSpatialTwoStageLeastSquares</code></a></td>
<td>Spatial lag by two-stage least squares. <code>instrumentOrder: 2</code> is <code>w_lags=2</code>.</td>
</tr>
<tr>
<td>-</td>
<td>-</td>
<td>-</td>
<td><code>spreg.GM_Error</code></td>
<td>-</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-regression#gpuspatialerrorgm"><code>GPUSpatialErrorGM</code></a></td>
<td>Spatial error by generalised moments.</td>
</tr>
<tr>
<td>-</td>
<td>-</td>
<td>-</td>
<td><code>mgwr.GWR</code></td>
<td>Geographically Weighted Regression</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-regression#gpugeographicallyweightedregression"><code>GPUGeographicallyWeightedRegression</code></a></td>
<td>Fixed or adaptive bandwidth, with the local condition number (<code>local_collinearity</code>). Use <code>mgwr</code> when multiscale backfitting is required.</td>
</tr>
<tr>
<td>-</td>
<td>-</td>
<td>-</td>
<td><code>mgwr.GWR.spatial_variability</code></td>
<td>GWR4</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-regression#gpugeographicallyweightedregressionnonstationaritytest"><code>GPUGeographicallyWeightedRegressionNonstationarityTest</code></a></td>
<td>Monte Carlo test of coefficient non-stationarity at the selected bandwidth.</td>
</tr>
<tr>
<td>-</td>
<td>-</td>
<td>-</td>
<td><code>spreg.ML_Lag</code>, <code>ML_Error</code>, <code>spat_impacts</code></td>
<td>-</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-regression#gpuspatiallagmaximumlikelihood"><code>GPUSpatialLagMaximumLikelihood</code></a>, <a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-regression#gpuspatialerrormaximumlikelihood"><code>GPUSpatialErrorMaximumLikelihood</code></a>, <a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-regression#gpuspatiallagimpacts"><code>GPUSpatialLagImpacts</code></a></td>
<td><code>method='full'</code> equivalents with a truncated-trace log-determinant; impact standard errors are not provided.</td>
</tr>
<tr>
<td>-</td>
<td>-</td>
<td>-</td>
<td><code>spreg</code> panel, SUR, regimes, probit</td>
<td>-</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-regression#gpuordinaryleastsquares"><code>GPUOrdinaryLeastSquares</code></a></td>
<td>Caller-partitioned OLS covers panel and regime fits. Use <code>spreg</code> when SUR or probit inference is required.</td>
</tr>
<tr>
<td>-</td>
<td>-</td>
<td>-</td>
<td><code>inequality.gini.Gini</code>, <code>theil.Theil</code>, <code>Gini_Spatial</code></td>
<td>-</td>
<td><a href="https://luma.gl/docs/api-reference/experimental/gpu-dataframe-analysis#gpuinequality"><code>GPUInequality</code></a> (dataframe)</td>
<td>Inequality indices.</td>
</tr>
<tr>
<td>-</td>
<td>-</td>
<td>-</td>
<td><code>segregation</code> aspatial and spatial indices</td>
<td>-</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-accessibility#gpusegregation"><code>GPUSegregation</code></a></td>
<td>Dissimilarity, isolation and relatives, with local outputs. Undefined indices are NaN; no permutation inference.</td>
</tr>
</tbody>
</table>
<h2 id="clustering-and-classification">Clustering and classification</h2>
<table>
<thead>
<tr>
<th>turf</th>
<th>PostGIS</th>
<th>GeoPandas / Shapely</th>
<th>PySAL</th>
<th>QGIS / ArcGIS / CARTO</th>
<th>luma.gl</th>
<th>Notes</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>clustersDbscan</code></td>
<td><code>ST_ClusterDBSCAN</code></td>
<td><code>sklearn.cluster.DBSCAN</code> on coordinates</td>
<td><code>esda.adbscan</code></td>
<td>DBSCAN clustering, Density-based Clustering</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-statistics#gpuspatialclustering"><code>GPUSpatialClustering</code></a></td>
<td>Clusters at or above <code>maximumClusterCount</code> are dropped. Planar or longitude/latitude with a great-circle <code>epsilon</code> (no antimeridian wrap); recenter large planar coordinates (float32).</td>
</tr>
<tr>
<td>-</td>
<td>-</td>
<td><code>sklearn.cluster.HDBSCAN</code>, <code>hdbscan.HDBSCAN</code></td>
<td>-</td>
<td>HDBSCAN, Density-based Clustering (HDBSCAN)</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-statistics#gpuhierarchicaldensityclustering"><code>GPUHierarchicalDensityClustering</code></a></td>
<td>Excess-of-mass selection, planar, up to 2^20 points; ties in mutual reachability may condense a few points differently from scikit-learn.</td>
</tr>
<tr>
<td><code>clustersKmeans</code></td>
<td><code>ST_ClusterKMeans</code></td>
<td><code>sklearn.cluster.KMeans</code></td>
<td>-</td>
<td>K-means clustering, CARTO <code>ST_CLUSTERKMEANS</code></td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-statistics#gpukmeans"><code>GPUSpatialKMeans</code></a></td>
<td>Iteration cap with an optional early-out <code>tolerance</code>. k-means++ by default, <code>restartCount</code> best-of-n restarts, total <code>inertia</code> output; planar only.</td>
</tr>
<tr>
<td><code>convex</code></td>
<td><code>ST_ConvexHull</code> per group</td>
<td><code>GeoSeries.convex_hull</code> per <code>groupby</code></td>
<td>-</td>
<td>Convex hull, Minimum Bounding Geometry</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-statistics#gpugroupconvexhull"><code>GPUGroupConvexHull</code></a>, <a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-statistics#gpugroupgeometry"><code>GPUGroupGeometry</code></a>; recipe <code>addClusterAndOutlineRecipe</code></td>
<td>Per-label hulls and bounds. The recipe chains cluster, hull and area.</td>
</tr>
<tr>
<td><code>standardDeviationalEllipse</code>, <code>centralMean</code>, <code>medianCenter</code></td>
<td>-</td>
<td>-</td>
<td><code>pointpats.centrography</code></td>
<td>Directional Distribution, Mean Center, Median Center</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-statistics#gpugeographicdistribution"><code>GPUGeographicDistribution</code></a></td>
<td>Planar only (<code>longitude-latitude</code> throws; spherical centre deferred). The ellipse convention is selectable; no small-n df correction.</td>
</tr>
<tr>
<td>-</td>
<td>-</td>
<td><code>mapclassify.classify</code> (<code>Quantiles</code>, <code>EqualInterval</code>, <code>FisherJenks</code>, <code>HeadTailBreaks</code>, <code>BoxPlot</code>, <code>StdMean</code>, <code>MaximumBreaks</code>, <code>UserDefined</code>)</td>
<td><code>mapclassify</code></td>
<td>Graduated symbology modes (Quantile, Natural Breaks, Pretty, Std Dev, Equal)</td>
<td><a href="https://luma.gl/docs/api-reference/experimental/gpu-dataframe-analysis#gpucolumnquantiles-gpuclassbreaks-gpucolorscale-and-gpubivariateclassification"><code>GPUClassBreaks</code></a> (dataframe), <a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-dynamics#gpuclassassignment"><code>GPUClassAssignment</code></a>, <a href="https://luma.gl/docs/api-reference/experimental/gpu-dataframe-analysis#gpucolumnquantiles-gpuclassbreaks-gpucolorscale-and-gpubivariateclassification"><code>GPUColorScale</code></a> (dataframe)</td>
<td>Breaks are computed on the GPU without readback.</td>
</tr>
<tr>
<td>-</td>
<td>-</td>
<td>-</td>
<td><code>mapclassify</code> bivariate, <code>geoplot</code></td>
<td>Bivariate choropleth</td>
<td><a href="https://luma.gl/docs/api-reference/experimental/gpu-dataframe-analysis#gpucolumnquantiles-gpuclassbreaks-gpucolorscale-and-gpubivariateclassification"><code>GPUBivariateClassification</code></a> (dataframe)</td>
<td>-</td>
</tr>
<tr>
<td>-</td>
<td>-</td>
<td>-</td>
<td><code>mapclassify</code> fit metrics (<code>gadf</code>, <code>adcm</code>)</td>
<td>-</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-dynamics#gpuclassificationfit"><code>GPUClassificationFit</code></a></td>
<td>Goodness of classification fit.</td>
</tr>
<tr>
<td>-</td>
<td>-</td>
<td>-</td>
<td><code>giddy.Markov</code>, <code>Spatial_Markov</code>, <code>LISA_Markov</code>, <code>directional</code>, <code>Rose</code></td>
<td>-</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-dynamics#gputransitionmatrix"><code>GPUTransitionMatrix</code></a>, <a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-dynamics#gpuspatialmarkov-and-gpulisamarkov"><code>GPUSpatialMarkov</code></a>, <a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-dynamics#gpuspatialmarkov-and-gpulisamarkov"><code>GPULISAMarkov</code></a>, <a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-statistics#additional-spatial-statistics-and-bounded-optimization"><code>GPURoseStatistic</code></a></td>
<td>Distribution dynamics from class sequences plus directional or axial circular summaries.</td>
</tr>
<tr>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>Calculate Composite Index</td>
<td><a href="https://luma.gl/docs/api-reference/experimental/gpu-dataframe-analysis#gpucompositescore"><code>GPUCompositeScore</code></a> (dataframe)</td>
<td>Standardised weighted index.</td>
</tr>
<tr>
<td>-</td>
<td>-</td>
<td>-</td>
<td><code>spopt.region.Skater</code> (<code>SpanningForest</code>), R <code>spdep::skater</code></td>
<td>Spatially Constrained Multivariate Clustering</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-statistics#gpuspatialweightsminimumspanningtree-gpuskaterregions-and-gpuregionpartitionevaluation"><code>GPUSpatialWeightsMinimumSpanningTree</code></a>, <a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-statistics#gpuspatialweightsminimumspanningtree-gpuskaterregions-and-gpuregionpartitionevaluation"><code>GPUSkaterRegions</code></a></td>
<td>Boruvka minimum spanning tree over contiguity, then greedy SKATER cuts. Matches spopt partitions.</td>
</tr>
<tr>
<td>-</td>
<td>-</td>
<td>-</td>
<td><code>spopt.region.AZP</code>, <code>MaxPHeuristic</code>, <code>Ward</code></td>
<td>Spatially Constrained Multivariate Clustering</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-statistics#additional-spatial-statistics-and-bounded-optimization"><code>GPUAZPRegions</code></a>, <a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-statistics#additional-spatial-statistics-and-bounded-optimization"><code>GPUMaxPRegions</code></a>, <a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-statistics#additional-spatial-statistics-and-bounded-optimization"><code>GPUWardRegions</code></a></td>
<td>Greedy steepest-descent AZP, randomized multi-start greedy Max-P (Philox seed order per start, optional AZP refinement), and adjacent Ward linkage; attributes are standardized by default and directed weights are symmetrized. Use <code>spopt</code> when its annealing Max-P/AZP search must be reproduced exactly.</td>
</tr>
<tr>
<td>-</td>
<td>-</td>
<td>-</td>
<td><code>esda.silhouettes</code>, <code>sklearn.metrics.silhouette_score</code>, <code>calinski_harabasz_score</code></td>
<td>-</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-statistics#gpuspatialweightsminimumspanningtree-gpuskaterregions-and-gpuregionpartitionevaluation"><code>GPURegionPartitionEvaluation</code></a></td>
<td>Within and between sums of squares, sizes and a cross-link compactness measure for any label column. Not a silhouette score.</td>
</tr>
<tr>
<td>-</td>
<td>-</td>
<td>-</td>
<td><code>sklearn.neighbors</code> on attribute space</td>
<td>Similarity Search</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-statistics#gpusimilarlocations"><code>GPUSimilarLocations</code></a></td>
<td>Ranks rows by weighted distance in standardised attribute space to a reference.</td>
</tr>
<tr>
<td>-</td>
<td>-</td>
<td>-</td>
<td><code>spopt.locate.PMedian</code>, <code>MCLP</code>, <code>LSCP</code></td>
<td>Location-Allocation</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-statistics#additional-spatial-statistics-and-bounded-optimization"><code>GPULocationAllocation</code></a>, <a href="/luma-contributors/docs/api/gpu-network/operators#gpunetworksnapping-gpunetworkcostmatrix-and-gpunetworkaccessibility"><code>GPUNetworkCostMatrix</code></a> (network)</td>
<td>Bounded greedy facility selection plus a Teitz-Bart swap phase over dense costs, with assignments, objectives, uncovered demand, swap and completion status. Use <code>spopt</code> when an integer-programming optimum or richer constraints are required.</td>
</tr>
</tbody>
</table>
<h2 id="density-and-interpolation">Density and interpolation</h2>
<table>
<thead>
<tr>
<th>turf</th>
<th>PostGIS</th>
<th>GeoPandas / Shapely</th>
<th>PySAL</th>
<th>QGIS / ArcGIS / CARTO</th>
<th>luma.gl</th>
<th>Notes</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>hexGrid</code> + count, <code>collect</code></td>
<td><code>ST_HexagonGrid</code> + <code>ST_Within</code></td>
<td><code>hexbin</code></td>
<td>-</td>
<td>Point Density</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-density#gpupointdensity"><code>GPUPointDensity</code></a></td>
<td>Binned count (or weight sum) per cell, optionally smoothed with a kernel measured in cells. Not an intensity and not a kernel density; use <code>GPUKernelDensity</code>.</td>
</tr>
<tr>
<td>-</td>
<td>-</td>
<td><code>spatstat.geom::owin</code></td>
<td>-</td>
<td>Study area</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-density#gpustudywindowpreparation"><code>GPUStudyWindowPreparation</code></a></td>
<td>GPU polygon study window with holes and islands: oriented edges, area, perimeter, edge index and sampler cells. Planar only.</td>
</tr>
<tr>
<td>-</td>
<td>-</td>
<td><code>scipy.stats.gaussian_kde</code>, <code>sklearn.neighbors.KernelDensity</code></td>
<td>-</td>
<td>Kernel Density, Heatmap (Kernel Density)</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-density#gpukerneldensity"><code>GPUKernelDensity</code></a></td>
<td>Intensity per unit area, bandwidth in ground units, gaussian, quartic and epanechnikov kernels, uniform or Diggle edge correction. Planar only.</td>
</tr>
<tr>
<td><code>hexGrid</code> + count</td>
<td><code>h3_lat_lng_to_cell</code> + <code>GROUP BY</code></td>
<td><code>h3pandas</code>, <code>quadbin</code></td>
<td>-</td>
<td>CARTO <code>QUADBIN_FROMGEOGPOINT</code></td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-cells#gpucellaggregation-gpucellrollup-gpucellpyramid-and-gpucelllevelselection"><code>GPUCellAggregation</code></a>, <a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-cells#gpupointtocell-and-gpucellgeometry"><code>GPUPointToCell</code></a></td>
<td>H3 and Quadbin cells. Cell keys must be valid. Points are longitude/latitude; pre-keyed rows are <code>discrete-grid</code>.</td>
</tr>
<tr>
<td><code>idw</code>, <code>interpolate</code></td>
<td>-</td>
<td><code>scipy.interpolate</code>, <code>gstat</code></td>
<td><code>mgwr</code> kernels</td>
<td>IDW interpolation, CARTO <code>ST_INTERPOLATE_POINTS</code></td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-density#gpuinversedistanceweighting"><code>GPUInverseDistanceWeighting</code></a></td>
<td>Neighbourhood IDW onto a raster, planar or great-circle (lon/lat, meters), optional Franke-Nielson taper.</td>
</tr>
<tr>
<td>-</td>
<td>-</td>
<td><code>skgstat.Variogram</code></td>
<td>-</td>
<td>Variogram (SAGA)</td>
<td><a href="https://luma.gl/docs/api-reference/experimental/gpu-dataframe-analysis#gpuvariogram-gpuspatialcorrelogram-gpuripley-gpuripleydistancefunctions-and-gpupointpatternindices"><code>GPUVariogram</code></a> (dataframe)</td>
<td>Empirical variogram to choose a model. Pair it with <code>GPUKriging</code>.</td>
</tr>
<tr>
<td>-</td>
<td>-</td>
<td><code>skgstat.OrdinaryKriging</code>, <code>pykrige.OrdinaryKriging</code> (<code>n_closest_points</code>), R <code>gstat::krige</code></td>
<td>-</td>
<td>Ordinary Kriging (SAGA), Kriging (ArcGIS)</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-density#gpukriging"><code>GPUKriging</code></a></td>
<td>Local ordinary kriging on a raster with kriging variance, ill-conditioning and leave-one-out diagnostics, planar or great-circle (lon/lat, meters). Use <code>gstat</code> or PyKrige for global and universal kriging.</td>
</tr>
<tr>
<td><code>randomPoint</code>, <code>sample</code></td>
<td><code>ST_GeneratePoints</code></td>
<td><code>GeoSeries.sample_points</code></td>
<td>-</td>
<td>Random points in polygons, Dot density</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-density#gpudotdensity-and-gpurandompointsinpolygon"><code>GPURandomPointsInPolygon</code></a>, <a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-density#gpudotdensity-and-gpurandompointsinpolygon"><code>GPUDotDensity</code></a></td>
<td>Deterministic seed. Bounded point count.</td>
</tr>
<tr>
<td>-</td>
<td><code>ST_Length(ST_Intersection(line, cell))</code> summed</td>
<td>-</td>
<td>-</td>
<td>Line density</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-geometry#gpulinedensity"><code>GPULineDensity</code></a></td>
<td>Length per cell divided by cell area (box kernel), exact per segment. Not the circular search-radius line density of ArcGIS and QGIS.</td>
</tr>
<tr>
<td>-</td>
<td>-</td>
<td>-</td>
<td><code>tobler.area_interpolate</code></td>
<td>Areal interpolation</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-accessibility#gpuarealinterpolation"><code>GPUArealInterpolation</code></a>; recipe <code>addChangeOfSupportRecipe</code></td>
<td>Areal weighting on a rasterised overlap; intensive weights default to the overlap denominator, and per-source conserved fractions, anchors for sub-cell sources and spherical cell areas for longitude/latitude rasters are available. The recipe adds a spatial lag.</td>
</tr>
<tr>
<td>-</td>
<td>-</td>
<td>-</td>
<td><code>tobler.pycno</code></td>
<td>Pycnophylactic interpolation</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-accessibility#gpupycnophylactic"><code>GPUPycnophylactic</code></a></td>
<td>Iterative smoothing with volume preservation, an optional convergence tolerance with <code>converged</code> status, anchor cells for zones smaller than a cell, and density-per-area smoothing on longitude/latitude rasters.</td>
</tr>
<tr>
<td>-</td>
<td>-</td>
<td>-</td>
<td><code>access.Access.two_step_fca</code>, <code>weighted_catchment</code>, <code>Huff</code></td>
<td>Enrich, Huff Model</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-accessibility#gpucatchmentaccessibility"><code>GPUCatchmentAccessibility</code></a>, <a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-accessibility#gpuhufftradeareas"><code>GPUHuffTradeAreas</code></a></td>
<td>Floating catchment and Huff trade areas.</td>
</tr>
<tr>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>Focal Statistics, Neighborhood Statistics</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-statistics#gpufocalstatistics"><code>GPUFocalStatistics</code></a></td>
<td>Focal windows on a lattice.</td>
</tr>
</tbody>
</table>
<h2 id="geometry-utilities">Geometry utilities</h2>
<table>
<thead>
<tr>
<th>turf</th>
<th>PostGIS</th>
<th>GeoPandas / Shapely</th>
<th>PySAL</th>
<th>QGIS / ArcGIS / CARTO</th>
<th>luma.gl</th>
<th>Notes</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>area</code>, <code>length</code>, <code>centroid</code>, <code>bbox</code>, <code>center</code>, <code>centerOfMass</code></td>
<td><code>ST_Area</code>, <code>ST_Length</code>, <code>ST_Perimeter</code>, <code>ST_Centroid</code></td>
<td><code>GeoSeries.area</code>, <code>.length</code>, <code>.centroid</code>, <code>.bounds</code></td>
<td>-</td>
<td>Polygon centroids, Add geometry attributes, Calculate Geometry</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-lines#gpugeometrymeasures-gpugeodesicpairs-and-gpugeodesicdestination"><code>GPUGeometryMeasures</code></a></td>
<td>Per feature and per group. Planar, geographic or <code>'geodesic'</code> (Karney-style area), per-feature extreme vertices, and point features.</td>
</tr>
<tr>
<td><code>pointOnFeature</code></td>
<td><code>ST_PointOnSurface</code>, <code>ST_MaximumInscribedCircle</code></td>
<td><code>representative_point()</code></td>
<td>-</td>
<td>Pole of inaccessibility, Point on surface</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-geometry#gpulabelpoint"><code>GPULabelPoint</code></a></td>
<td>Pole of inaccessibility.</td>
</tr>
<tr>
<td>-</td>
<td>-</td>
<td><code>momepy</code> compactness</td>
<td><code>esda.shape</code> (<code>isoperimetric_quotient</code>, <code>convex_hull_ratio</code>)</td>
<td>Shape indices</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-geometry#gpushapedescriptors"><code>GPUShapeDescriptors</code></a></td>
<td>Elongation, orientation and compactness.</td>
</tr>
<tr>
<td><code>booleanValid</code></td>
<td><code>ST_IsValid</code>, <code>ST_IsValidDetail</code>, <code>ST_MakeValid</code></td>
<td><code>is_valid</code>, <code>shapely.explain_validity</code>, <code>make_valid</code></td>
<td>-</td>
<td>Check validity, Fix geometries</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-validity#gpugeometryvalidity"><code>GPUGeometryValidity</code></a>, <a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-cells#gpulinenoding-gpupolygonize-and-gpumakevalid"><code>GPUMakeValid</code></a></td>
<td>Reports invalid rows for polygons, lines and points; bounded polygon repair rebuilds valid faces and reports collapsed or unused edges.</td>
</tr>
<tr>
<td>-</td>
<td><code>ST_ShiftLongitude</code>, <code>ST_WrapX</code>, <code>ST_Segmentize(geography)</code></td>
<td>Python <code>antimeridian</code></td>
<td>-</td>
<td>GDAL <code>-wrapdateline</code></td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-geometry#gpulonlatpolygonpreparation"><code>GPULonLatPolygonPreparation</code></a></td>
<td>GPU short-way longitude unwrapping, optional great-circle densification, nearer-pole capping and antimeridian splitting; fixed doubled polygon/ring topology.</td>
</tr>
<tr>
<td><code>bboxClip</code>, <code>intersect</code></td>
<td><code>ST_ClipByBox2D</code>, <code>ST_Intersection</code></td>
<td><code>shapely.clip_by_rect</code>, <code>clip</code>, <code>overlay</code></td>
<td>-</td>
<td>Clip by extent, Intersection</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-geometry#gpurectangleclip"><code>GPURectangleClip</code></a>, <a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-lines#gpulineclipbypolygon-and-gpusharedpaths"><code>GPULineClipByPolygon</code></a>, <a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-geometry#gpupolygonoverlay-and-gpubuffersurface"><code>GPUPolygonOverlay</code></a></td>
<td>Rectangle, line-by-polygon, and bounded polygon-by-polygon clipping.</td>
</tr>
<tr>
<td><code>squareGrid</code>, <code>hexGrid</code>, <code>triangleGrid</code>, <code>pointGrid</code></td>
<td><code>ST_SquareGrid</code>, <code>ST_HexagonGrid</code></td>
<td>-</td>
<td>-</td>
<td>Create grid</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-geometry#gpugridgenerator"><code>GPUGridGenerator</code></a></td>
<td>Square, hexagon (pointy or flat) and triangle grids with a capacity cap, and an optional extent mask.</td>
</tr>
<tr>
<td><code>circle</code>, <code>sector</code>, <code>ellipse</code></td>
<td><code>ST_Buffer</code> on a point, <code>ST_MakeEllipse</code></td>
<td><code>shapely.Point.buffer</code></td>
<td>-</td>
<td>Create wedge buffers</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-geometry#gpushapegenerator"><code>GPUShapeGenerator</code></a></td>
<td>Planar or geodesic circles, sectors and ellipses per feature. Circles and sectors match turf.</td>
</tr>
<tr>
<td>-</td>
<td><code>ST_Hilbert</code></td>
<td><code>GeoSeries.hilbert_distance</code>, <code>spatialpandas</code> <code>hilbert_distance</code></td>
<td>-</td>
<td>-</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-geometry#gpuhilbertkeys"><code>GPUHilbertKeys</code></a></td>
<td>Hilbert keys and a curve-order permutation, order up to 16.</td>
</tr>
<tr>
<td><code>buffer</code></td>
<td><code>ST_Buffer</code></td>
<td><code>buffer</code></td>
<td>-</td>
<td>Buffer</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-geometry#gpuoutlinegeometry"><code>GPUOutlineGeometry</code></a>, <a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-geometry#gpupolygonoverlay-and-gpubuffersurface"><code>GPUBufferSurface</code></a></td>
<td><code>GPUOutlineGeometry</code> is render-only; <code>GPUBufferSurface</code> creates bounded planar polygon surfaces with cap and join styles.</td>
</tr>
<tr>
<td><code>simplify</code></td>
<td><code>ST_Simplify</code>, <code>ST_SimplifyVW</code></td>
<td><code>GeoSeries.simplify</code></td>
<td>-</td>
<td>Simplify</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-lines#gpulinesimplification"><code>GPULineSimplification</code></a></td>
<td>Douglas-Peucker, or approximate Visvalingam-Whyatt (<code>method: 'visvalingam'</code>, about 0.1 to 3 percent of vertices differ from the heap order).</td>
</tr>
<tr>
<td><code>simplify</code> (topology preserving)</td>
<td><code>ST_CoverageSimplify</code></td>
<td><code>shapely.simplify_coverage</code></td>
<td>-</td>
<td>Simplify (topological)</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-lines#gpucoveragesimplification"><code>GPUCoverageSimplification</code></a></td>
<td>Shared edges stay gap-free, with exact crossing detection and repair. The criterion is distance, not area. <code>simplifyBoundary: false</code> keeps unshared edges.</td>
</tr>
<tr>
<td><code>lineSplit</code></td>
<td><code>ST_Split</code>, <code>ST_Node</code></td>
<td><code>shapely.ops.split</code>, <code>shapely.node</code></td>
<td>-</td>
<td>Split with lines, Feature To Line</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-lines#gpulinesplit-and-gpulinemerge"><code>GPULineSplit</code></a></td>
<td>Splits lines at every crossing, touch and shared vertex, as <code>unary_union</code> of the lines does. Capacity-bounded pieces.</td>
</tr>
<tr>
<td>-</td>
<td><code>ST_LineMerge</code></td>
<td><code>shapely.line_merge</code></td>
<td>-</td>
<td>Merge lines, Dissolve (unsplit lines)</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-lines#gpulinesplit-and-gpulinemerge"><code>GPULineMerge</code></a></td>
<td>Joins lines at endpoints shared by exactly two lines. Exact endpoint match. <code>directed: true</code> joins only end to start.</td>
</tr>
<tr>
<td><code>polygonSmooth</code>, <code>bezierSpline</code></td>
<td><code>ST_ChaikinSmoothing</code></td>
<td>-</td>
<td>-</td>
<td>Smooth</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-lines#gpulinesegmentize-gpugreatcirclearcs-gpulinesmooth-and-gpulinechunk"><code>GPULineSmooth</code></a></td>
<td>Chaikin corner cutting.</td>
</tr>
<tr>
<td><code>lineChunk</code>, <code>lineSegment</code></td>
<td><code>ST_Segmentize</code></td>
<td><code>shapely.segmentize</code></td>
<td>-</td>
<td>Densify by interval, Split lines by maximum length</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-lines#gpulinesegmentize-gpugreatcirclearcs-gpulinesmooth-and-gpulinechunk"><code>GPULineSegmentize</code></a>, <a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-lines#gpulinesegmentize-gpugreatcirclearcs-gpulinesmooth-and-gpulinechunk"><code>GPULineChunk</code></a></td>
<td>Fixed-capacity output with overflow flags.</td>
</tr>
<tr>
<td><code>along</code>, <code>lineSliceAlong</code></td>
<td><code>ST_LineInterpolatePoint</code>, <code>ST_LocateAlong</code></td>
<td><code>shapely.line_interpolate_point</code>, <code>.project</code></td>
<td>-</td>
<td>Interpolate point on line</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-lines#gpulinearreferencing-and-gpulinelocate"><code>GPULineLocate</code></a>, <a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-lines#gpulinearreferencing-and-gpulinelocate"><code>GPULinearReferencing</code></a></td>
<td>Locate by distance or by projection.</td>
</tr>
<tr>
<td><code>greatCircle</code></td>
<td><code>ST_Segmentize(geography)</code></td>
<td>-</td>
<td>-</td>
<td>Geodesic arcs, CARTO <code>ST_GREATCIRCLE</code></td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-lines#gpulinesegmentize-gpugreatcirclearcs-gpulinesmooth-and-gpulinechunk"><code>GPUGreatCircleArcs</code></a></td>
<td>Densified great-circle arcs.</td>
</tr>
<tr>
<td><code>lineIntersect</code>, <code>kinks</code></td>
<td><code>ST_Intersection</code> (points)</td>
<td>-</td>
<td>-</td>
<td>Line intersections</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-validity#gpusegmentintersection"><code>GPUSegmentIntersection</code></a></td>
<td>Crossing points between segment sets.</td>
</tr>
<tr>
<td><code>polygonize</code></td>
<td><code>ST_Polygonize</code>, <code>ST_BuildArea</code>, <code>h3_cells_to_multi_polygon_wkb</code></td>
<td><code>shapely.polygonize</code>, <code>polygonize_full</code></td>
<td>-</td>
<td>Polygonize</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-cells#gpulinenoding-gpupolygonize-and-gpumakevalid"><code>GPUPolygonize</code></a>, <a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-cells#gpulinenoding-gpupolygonize-and-gpumakevalid"><code>GPULineNoding</code></a>, <a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-cells#gpusegmentringassembly"><code>GPUSegmentRingAssembly</code></a>, <a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-cells#gpucellsetoutline"><code>GPUCellSetOutline</code></a></td>
<td>Bounded planar noding and polygonization with closed-ring, open-chain, cut-edge, dangle and unused-edge diagnostics.</td>
</tr>
<tr>
<td>-</td>
<td><code>ST_Length(ST_Intersection(line, polygon))</code></td>
<td><code>overlay</code> + <code>length</code></td>
<td>-</td>
<td>Sum line lengths</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-geometry#gpulinelengthperpolygon"><code>GPULineLengthPerPolygon</code></a></td>
<td>Line length per polygon.</td>
</tr>
<tr>
<td><code>union</code>, <code>intersect</code>, <code>difference</code>, <code>dissolve</code>, <code>voronoi</code>, <code>tin</code>, <code>concave</code></td>
<td><code>ST_Union</code>, <code>ST_Intersection</code>, <code>ST_Difference</code>, <code>ST_Buffer</code>, <code>ST_Voronoi*</code>, <code>ST_Delaunay*</code>, <code>ST_MakeValid</code>, <code>ST_Subdivide</code></td>
<td><code>overlay</code>, <code>dissolve</code>, <code>unary_union</code>, <code>voronoi_polygons</code>, <code>delaunay_triangles</code>, <code>concave_hull</code>, <code>make_valid</code></td>
<td>-</td>
<td>Dissolve, Union, Intersection, Difference, Buffer, Delaunay, Voronoi, Fix geometries</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-geometry#gpupolygonoverlay-and-gpubuffersurface"><code>GPUPolygonOverlay</code></a>, <a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-geometry#gpupolygonoverlay-and-gpubuffersurface"><code>GPUBufferSurface</code></a>, <a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-geometry#gpudelaunaytessellation-and-gpuvoronoidiagram"><code>GPUDelaunayTessellation</code></a>, <a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-geometry#gpudelaunaytessellation-and-gpuvoronoidiagram"><code>GPUVoronoiDiagram</code></a>, <a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-cells#gpulinenoding-gpupolygonize-and-gpumakevalid"><code>GPUMakeValid</code></a></td>
<td>Bounded planar outputs with explicit capacities and completeness status. Concave hull and generic subdivision remain external; coverage-only union is <code>GPUCoverageDissolve</code> and raster Voronoi is <a href="/luma-contributors/docs/api/gpu-raster/operations-analysis#gpudistancefield"><code>GPUDistanceField</code></a>.</td>
</tr>
<tr>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>Model Designer, graphical model</td>
<td>recipes, or compose contributors in one <code>GPUCommandGraph</code></td>
<td>-</td>
</tr>
</tbody>
</table>
<h2 id="geopandas-shapely-and-rust-geo-parity">GeoPandas, Shapely and Rust geo parity</h2>
<p>polars-st and GeoPolars mirror Shapely's function names, so every Shapely row on this page also covers
the polars-st (<code>st.*</code>) and GeoPolars equivalent. The middle column lists names from the Rust <code>geo</code> crate,
Apache Sedona and h3 that have no Shapely spelling. A dash means no such name exists.</p>
<table>
<thead>
<tr>
<th>GeoPandas / Shapely (and polars-st, GeoPolars)</th>
<th>geo crate, Sedona, h3, PostGIS</th>
<th>luma.gl</th>
<th>Notes</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>shapely.affinity.*</code> (<code>affine_transform</code>, <code>translate</code>, <code>scale</code>, <code>rotate</code>, <code>skew</code>)</td>
<td><code>ST_Affine</code>, <code>ST_Translate</code>, <code>ST_Scale</code>, <code>ST_Rotate</code> (Sedona)</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-geometry#gpuaffinetransform-gpugeometryorientation-and-gpugeometrycleanup"><code>GPUAffineTransform</code></a></td>
<td>One per-frame matrix or per-feature matrices. The origin can be a point, center or centroid.</td>
</tr>
<tr>
<td><code>shapely.reverse</code>, <code>orient_polygons</code></td>
<td><code>ST_Reverse</code>, <code>ST_ForcePolygonCW</code>, <code>ST_ForcePolygonCCW</code> (Sedona)</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-geometry#gpuaffinetransform-gpugeometryorientation-and-gpugeometrycleanup"><code>GPUGeometryOrientation</code></a></td>
<td>Layout unchanged.</td>
</tr>
<tr>
<td><code>shapely.remove_repeated_points</code>, <code>set_precision(mode='pointwise')</code></td>
<td><code>ST_RemoveRepeatedPoints</code>, <code>ST_SnapToGrid</code> (Sedona)</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-geometry#gpuaffinetransform-gpugeometryorientation-and-gpugeometrycleanup"><code>GPUGeometryCleanup</code></a></td>
<td>Compacted output. GEOS pointwise precision keeps repeats, so set <code>removeRepeatedPoints: false</code> for parity. Use Shapely when canonical <code>normalize</code> ordering is required.</td>
</tr>
<tr>
<td><code>shapely.minimum_rotated_rectangle</code>, <code>oriented_envelope</code>, <code>minimum_bounding_circle</code>, <code>minimum_bounding_radius</code>, <code>envelope</code></td>
<td><code>ST_MinimumBoundingCircle</code>, <code>ST_OrientedEnvelope</code>, <code>ST_Envelope</code> (Sedona)</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-geometry#gpuminimumbounds"><code>GPUMinimumBounds</code></a></td>
<td>Per feature or per label group. Area and radius parity, not vertex order.</td>
</tr>
<tr>
<td><code>shapely.minimum_clearance</code>, <code>minimum_clearance_line</code></td>
<td><code>MinimumClearance</code> (geo)</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-validity#gpuminimumclearance"><code>GPUMinimumClearance</code></a></td>
<td>GEOS rule. <code>Infinity</code> when there is no clearance.</td>
</tr>
<tr>
<td><code>shapely.is_simple</code>, <code>is_ring</code>, <code>is_closed</code>, <code>is_ccw</code>, <code>get_num_points</code>, <code>count_coordinates</code>, <code>get_num_interior_rings</code>, <code>get_num_geometries</code>, <code>extract_unique_points</code>, <code>equals_exact</code>, <code>equals_identical</code></td>
<td><code>ST_IsSimple</code>, <code>ST_NPoints</code>, <code>ST_NumInteriorRings</code> (Sedona)</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-validity#gpugeometrypredicates"><code>GPUGeometryPredicates</code></a></td>
<td>Equality compares row <code>i</code> with row <code>i</code>; normalize first on the CPU when <code>equals_exact(normalize=True)</code> semantics are required.</td>
</tr>
<tr>
<td><code>shapely.is_valid</code>, <code>is_valid_reason</code> for lines and points</td>
<td>-</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-validity#gpugeometryvalidity"><code>GPUGeometryValidity</code></a></td>
<td><code>lines</code> and <code>points</code> inputs.</td>
</tr>
<tr>
<td><code>geopandas.clip(lines, polygon)</code>, <code>shapely.intersection(line, polygon)</code>, <code>difference(line, polygon)</code></td>
<td><code>Clip</code> (geo, lines)</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-lines#gpulineclipbypolygon-and-gpusharedpaths"><code>GPULineClipByPolygon</code></a></td>
<td>Bounded because no new polygon intersections are computed. Lines are not noded at their own crossings.</td>
</tr>
<tr>
<td><code>shapely.shared_paths</code></td>
<td>-</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-lines#gpulineclipbypolygon-and-gpusharedpaths"><code>GPUSharedPaths</code></a></td>
<td>Forward and backward runs per line pair.</td>
</tr>
<tr>
<td><code>GeoDataFrame.dissolve(by, method='coverage')</code>, <code>shapely.coverage_union_all</code>, <code>dissolve</code>, <code>unary_union</code></td>
<td><code>ST_CoverageUnion</code> (PostGIS)</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-lines#gpucoveragedissolve"><code>GPUCoverageDissolve</code></a>, <a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-geometry#gpupolygonoverlay-and-gpubuffersurface"><code>GPUPolygonOverlay</code></a></td>
<td>Use the coverage path for valid coverages and bounded polygon overlay for general dissolve.</td>
</tr>
<tr>
<td><code>shapely.coverage_is_valid</code>, <code>coverage_invalid_edges(gap_width)</code></td>
<td><code>ST_CoverageInvalidEdges</code> (PostGIS)</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-validity#gpucoveragevalidity"><code>GPUCoverageValidity</code></a></td>
<td>Per-segment flags. The gap rule is the GEOS one; equal to Shapely 2.2.0 on 222 fixture cases with gaps.</td>
</tr>
<tr>
<td><code>shapely.coverage_simplify(simplify_boundary=False)</code></td>
<td>-</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-lines#gpucoveragesimplification"><code>GPUCoverageSimplification</code></a></td>
<td><code>simplifyBoundary: false</code>.</td>
</tr>
<tr>
<td><code>shapely.offset_curve</code></td>
<td><code>OffsetCurve</code> (geo)</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-geometry#gpuoffsetcurve-and-gpuvertexsnap"><code>GPUOffsetCurve</code></a></td>
<td>Partial: no loop removal, so tight inside corners can self-cross.</td>
</tr>
<tr>
<td><code>shapely.snap</code></td>
<td><code>Snap</code> (geo)</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-geometry#gpuoffsetcurve-and-gpuvertexsnap"><code>GPUVertexSnap</code></a></td>
<td>Partial: vertex to vertex only, no insertion onto segments.</td>
</tr>
<tr>
<td><code>shapely.line_merge(directed=True)</code></td>
<td>-</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-lines#gpulinesplit-and-gpulinemerge"><code>GPULineMerge</code></a></td>
<td><code>directed: true</code>.</td>
</tr>
<tr>
<td><code>GeoSeries.make_grid(flat_topped=True, intersect=True)</code></td>
<td>-</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-geometry#gpugridgenerator"><code>GPUGridGenerator</code></a></td>
<td><code>hexOrientation: 'flat'</code> and <code>extent</code> with <code>output.intersects</code>; derive corners from the emitted cell geometry when needed.</td>
</tr>
<tr>
<td><code>GeoSeries.sample_points</code> (lines)</td>
<td>-</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-density#gpudotdensity-and-gpurandompointsinpolygon"><code>GPURandomPointsOnLine</code></a></td>
<td>Uniform along arc length with a stable prefix. Use GeoPandas for clustered Poisson line sampling.</td>
</tr>
<tr>
<td><code>sjoin(on_attribute=)</code>, <code>STRtree.query_nearest(exclusive=True)</code>, <code>shapely.shortest_line</code>, <code>ops.nearest_points</code></td>
<td>-</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-joins#gpuspatialpredicatejoin"><code>GPUSpatialPredicateJoin</code></a>, <a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-joins#gpupointinpolygonjoin-and-gpunearestfeaturejoin"><code>GPUNearestFeatureJoin</code></a></td>
<td>Options <code>onAttribute</code>, <code>exclusive</code> and <code>neighborQueryPoints</code>.</td>
</tr>
<tr>
<td><code>sjoin</code> merge (<code>how='left'</code>), <code>sjoin_nearest(distance_col, how)</code>, <code>explode</code>, <code>get_parts(return_index)</code>, <code>get_coordinates(return_index)</code>, <code>.cx[...]</code></td>
<td>-</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-joins#gpupairgather-gpuoffsetexpansion-and-gpuboundsfilter"><code>GPUPairGather</code></a>, <a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-joins#gpupairgather-gpuoffsetexpansion-and-gpuboundsfilter"><code>GPUOffsetExpansion</code></a>, <a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-joins#gpupairgather-gpuoffsetexpansion-and-gpuboundsfilter"><code>GPUBoundsFilter</code></a></td>
<td>Left rows come after matched rows. <code>.cx</code> is bounds level.</td>
</tr>
<tr>
<td><code>GeoSeries.hausdorff_distance</code>, <code>frechet_distance</code> (<code>densify</code>)</td>
<td><code>ST_MaxDistance</code> (Sedona), <code>HausdorffDistance</code>, <code>FrechetDistance</code> (geo)</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-lines#gputracksimilarity"><code>GPUTrackSimilarity</code></a></td>
<td>Hausdorff is vertex to segment, as GEOS.</td>
</tr>
<tr>
<td>-</td>
<td><code>OutlierDetection</code>, <code>ST_LocalOutlierFactor</code> (Sedona), <code>sklearn LocalOutlierFactor</code></td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-statistics#gpulocaloutlierfactor"><code>GPULocalOutlierFactor</code></a></td>
<td>Over a kNN table. Matches <code>-negative_outlier_factor_</code>.</td>
</tr>
<tr>
<td><code>GeoSeries.geometry_area_perimeter</code> (pyproj <code>Geod</code>)</td>
<td><code>GeodesicArea</code> (geo)</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-lines#gpugeometrymeasures-gpugeodesicpairs-and-gpugeodesicdestination"><code>GPUGeometryMeasures</code></a></td>
<td><code>coordinateSystem: 'geodesic'</code>.</td>
</tr>
<tr>
<td><code>pyproj.Geod.npts</code>, <code>fwd</code> (spherical)</td>
<td><code>HaversineClosestPoint</code>, <code>CrossTrackDistance</code>, <code>InterpolatePoint</code> (geo)</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-lines#gpulinearreferencing-and-gpulinelocate"><code>GPULinearReferencing</code></a>, <a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-lines#gpulinearreferencing-and-gpulinelocate"><code>GPULineLocate</code></a></td>
<td><code>coordinateSystem: 'spherical'</code>. Brute force over segments.</td>
</tr>
<tr>
<td>-</td>
<td><code>Rhumb::distance</code>, <code>bearing</code>, <code>destination</code> (geo)</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-lines#gpugeometrymeasures-gpugeodesicpairs-and-gpugeodesicdestination"><code>GPUGeodesicPairs</code></a>, <a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-lines#gpugeometrymeasures-gpugeodesicpairs-and-gpugeodesicdestination"><code>GPUGeodesicDestination</code></a></td>
<td><code>model: 'rhumb'</code>, sphere only.</td>
</tr>
<tr>
<td><code>simplification.simplify_coords_vw</code></td>
<td><code>SimplifyVw</code> (geo)</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-lines#gpulinesimplification"><code>GPULineSimplification</code></a></td>
<td>Approximate parallel Visvalingam-Whyatt. Use <code>geo</code> when the preserve variant's heap order is required.</td>
</tr>
<tr>
<td>-</td>
<td><code>TriangulateEarcut</code>, <code>ST_TriangulatePolygon</code> (Sedona), <code>earcut</code>, Delaunay</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-geometry#gpupolygontriangulation"><code>GPUPolygonTriangulation</code></a>, <a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-geometry#gpudelaunaytessellation-and-gpuvoronoidiagram"><code>GPUDelaunayTessellation</code></a></td>
<td>Earcut preserves polygon constraints and holes; Delaunay triangulates unconstrained point sites.</td>
</tr>
<tr>
<td>-</td>
<td>h3 <code>grid_distance</code>, <code>grid_path_cells</code>, <code>are_neighbor_cells</code>, <code>cell_area</code>, <code>edge_length</code>, <code>is_pentagon</code></td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-cells#gpucellgridpath-and-gpucellmeasures"><code>GPUCellGridPath</code></a>, <a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-cells#gpucellgridpath-and-gpucellmeasures"><code>GPUCellMeasures</code></a></td>
<td>Grid paths and measures are GPU-resident; use h3 for directed-edge and vertex index construction.</td>
</tr>
</tbody>
</table>
<h2 id="discrete-global-grid-cells">Discrete global grid cells</h2>
<table>
<thead>
<tr>
<th>turf</th>
<th>PostGIS</th>
<th>GeoPandas / Shapely</th>
<th>PySAL</th>
<th>QGIS / ArcGIS / CARTO</th>
<th>luma.gl</th>
<th>Notes</th>
</tr>
</thead>
<tbody>
<tr>
<td>-</td>
<td><code>h3_lat_lng_to_cell</code></td>
<td><code>h3pandas.geo_to_h3</code></td>
<td>-</td>
<td>CARTO <code>QUADBIN_FROMGEOGPOINT</code></td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-cells#gpupointtocell-and-gpucellgeometry"><code>GPUPointToCell</code></a></td>
<td>H3 and Quadbin.</td>
</tr>
<tr>
<td>-</td>
<td><code>h3_cell_to_boundary</code></td>
<td><code>h3pandas.h3_to_geo_boundary</code></td>
<td>-</td>
<td>-</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-cells#gpupointtocell-and-gpucellgeometry"><code>GPUCellGeometry</code></a></td>
<td>Centres and boundaries.</td>
</tr>
<tr>
<td>-</td>
<td><code>h3_grid_disk</code>, <code>h3_cell_to_parent</code></td>
<td><code>h3.grid_disk</code></td>
<td>-</td>
<td>CARTO <code>QUADBIN_KRING</code>, <code>QUADBIN_TOPARENT</code></td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-cells#gpucelltopology-and-gpucellcompaction"><code>GPUCellTopology</code></a></td>
<td>Neighbours and parents.</td>
</tr>
<tr>
<td>-</td>
<td><code>h3_polygon_to_cells</code></td>
<td><code>h3pandas.polyfill</code></td>
<td>-</td>
<td>CARTO <code>QUADBIN_POLYFILL</code></td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-cells#gpucellcover"><code>GPUCellCover</code></a></td>
<td>Polygon to cell cover with a containment mode.</td>
</tr>
<tr>
<td>-</td>
<td><code>GROUP BY h3_cell_to_parent</code></td>
<td><code>h3pandas.h3_to_parent</code> + <code>groupby</code></td>
<td>-</td>
<td>-</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-cells#gpucellaggregation-gpucellrollup-gpucellpyramid-and-gpucelllevelselection"><code>GPUCellRollup</code></a>, <a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-cells#gpucellaggregation-gpucellrollup-gpucellpyramid-and-gpucelllevelselection"><code>GPUCellPyramid</code></a>, <a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-cells#gpucellaggregation-gpucellrollup-gpucellpyramid-and-gpucelllevelselection"><code>GPUCellLevelSelection</code></a></td>
<td>Roll-ups across resolutions.</td>
</tr>
<tr>
<td>-</td>
<td><code>h3_cells_to_multi_polygon_wkb</code></td>
<td><code>h3.cells_to_h3shape</code></td>
<td>-</td>
<td>-</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-cells#gpucellsetoutline"><code>GPUCellSetOutline</code></a></td>
<td>Outline and optional closed rings for a cell set.</td>
</tr>
<tr>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>Compact, uncompact cells</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-cells#gpucelltopology-and-gpucellcompaction"><code>GPUCellCompaction</code></a></td>
<td>-</td>
</tr>
<tr>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>Compare cell tables</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-cells#gpucelltablecompare"><code>GPUCellTableCompare</code></a></td>
<td>Used by <code>addPeriodComparisonRecipe</code>.</td>
</tr>
</tbody>
</table>
<h2 id="trajectories">Trajectories</h2>
<table>
<thead>
<tr>
<th>turf</th>
<th>PostGIS</th>
<th>GeoPandas / Shapely</th>
<th>PySAL</th>
<th>QGIS / ArcGIS / CARTO</th>
<th>luma.gl</th>
<th>Notes</th>
</tr>
</thead>
<tbody>
<tr>
<td>-</td>
<td><code>ST_ClusterDBSCAN</code> on points, <code>ST_Length</code> per track</td>
<td><code>movingpandas.TrajectoryStopDetector</code>, <code>.get_length</code>, <code>.get_speed</code></td>
<td>-</td>
<td>Trajectories to stops, Track Intelligence</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-lines#gputrajectorymetrics"><code>GPUTrajectoryMetrics</code></a></td>
<td>Stops, lengths and speeds, plus per-step speed, heading and acceleration columns. Bounded outputs.</td>
</tr>
<tr>
<td>-</td>
<td><code>ST_Intersects</code> + <code>LAG</code></td>
<td><code>movingpandas</code> zone enter/exit</td>
<td>-</td>
<td>Enter and Exit events</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-lines#gpuzoneevents"><code>GPUZoneEvents</code></a></td>
<td>Zone entry and exit events with crossing positions, diagnostics and a sparse (track, zone) visit table.</td>
</tr>
<tr>
<td>-</td>
<td>-</td>
<td><code>movingpandas</code> stop detection + <code>sjoin</code> + <code>groupby</code></td>
<td>-</td>
<td>Dwell time in zones</td>
<td>recipe <code>addFleetDwellRecipe</code>, <code>addFleetDwellZoneEventsRecipe</code></td>
<td>One row per zone.</td>
</tr>
<tr>
<td>-</td>
<td>-</td>
<td><code>movingpandas</code> <code>interpolate_position_at</code>, <code>TrajectoryCollection.resample</code></td>
<td>-</td>
<td>Points along track</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-lines#gputrajectoryplayhead-and-gputrajectoryresample"><code>GPUTrajectoryPlayhead</code></a>, <a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-lines#gputrajectoryplayhead-and-gputrajectoryresample"><code>GPUTrajectoryResample</code></a></td>
<td>Position at a time and uniform resampling.</td>
</tr>
<tr>
<td>-</td>
<td>-</td>
<td><code>movingpandas</code> close-by encounters</td>
<td>-</td>
<td>Proximity events</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-lines#gputrajectoryencounters"><code>GPUTrajectoryEncounters</code></a>; <code>addClockEncounters</code></td>
<td>Pairs of tracks within a distance in the same time bucket (MobilityDB <code>tdwithin</code>). Capacity-bounded.</td>
</tr>
<tr>
<td>-</td>
<td>-</td>
<td>MobilityDB <code>tdwithin</code>, <code>nearestApproachDistance</code></td>
<td>-</td>
<td>Exact proximity events</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-lines#gputrajectoryclosestapproach"><code>GPUTrajectoryClosestApproach</code></a></td>
<td>Closest point of approach between linearly moving tracks: minimum distance, its time, first time and duration within a distance. Capacity-bounded.</td>
</tr>
<tr>
<td>-</td>
<td><code>ST_FrechetDistance</code>, <code>ST_HausdorffDistance</code></td>
<td><code>shapely.frechet_distance</code>, <code>hausdorff_distance</code></td>
<td>-</td>
<td>-</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-lines#gputracksimilarity"><code>GPUTrackSimilarity</code></a></td>
<td>Track similarity.</td>
</tr>
<tr>
<td><code>along</code></td>
<td><code>ST_LineInterpolatePoint</code></td>
<td><code>shapely.line_interpolate_point</code></td>
<td>-</td>
<td>-</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-lines#gpulinearreferencing-and-gpulinelocate"><code>GPULineLocate</code></a></td>
<td>-</td>
</tr>
<tr>
<td>-</td>
<td>-</td>
<td><code>movingpandas.TrajectoryCollection.add_speed</code></td>
<td>-</td>
<td>-</td>
<td><a href="https://luma.gl/docs/api-reference/experimental/gpu-dataframe-analysis#gputimewindowfilter"><code>GPUTimeWindowFilter</code></a>, <a href="https://luma.gl/docs/api-reference/experimental/gpu-dataframe-analysis#gputemporalreduction"><code>GPUTemporalReduction</code></a> (dataframe)</td>
<td>Time-window filter and reduction.</td>
</tr>
</tbody>
</table>
<h2 id="networks">Networks</h2>
<table>
<thead>
<tr>
<th>turf</th>
<th>PostGIS</th>
<th>GeoPandas / Shapely</th>
<th>PySAL</th>
<th>QGIS / ArcGIS / CARTO</th>
<th>luma.gl</th>
<th>Notes</th>
</tr>
</thead>
<tbody>
<tr>
<td>-</td>
<td><code>pgr_dijkstra</code>, <code>pgr_dijkstraCost</code></td>
<td><code>osmnx.shortest_path</code>, <code>networkx</code></td>
<td><code>spaghetti.Network</code></td>
<td>Shortest path (point to point, layer to point), Route</td>
<td><a href="/luma-contributors/docs/api/gpu-network/operators#gpunetworkreachability"><code>GPUNetworkReachability</code></a>, <a href="/luma-contributors/docs/api/gpu-network/operators#gpunetworkpathextraction-gpunetworkserviceareas-gpunetworkneighborhood-and-gpunetworkanalyticscolumns"><code>GPUNetworkPathExtraction</code></a> (network)</td>
<td>Multi-source cost and predecessors. Path extraction is bounded.</td>
</tr>
<tr>
<td>-</td>
<td><code>pgr_drivingDistance</code></td>
<td><code>osmnx</code></td>
<td><code>spaghetti</code></td>
<td>Service area (from point, from layer), Service Area</td>
<td><a href="/luma-contributors/docs/api/gpu-network/operators#gpunetworkpathextraction-gpunetworkserviceareas-gpunetworkneighborhood-and-gpunetworkanalyticscolumns"><code>GPUNetworkServiceAreas</code></a>, <a href="/luma-contributors/docs/api/gpu-network/operators#gpunetworkisochrones"><code>GPUNetworkIsochrones</code></a> (network); recipe <code>addDriveTimeCatchmentRecipe</code></td>
<td>Edge cut costs are exact. Isochrone raster or cell outline polygons are cell-resolution.</td>
</tr>
<tr>
<td>-</td>
<td><code>pgr_withPointsDD</code></td>
<td><code>momepy.snap</code></td>
<td><code>spaghetti.Network.snapobservations</code></td>
<td>Snap points to network, Network Analyst Locations</td>
<td><a href="/luma-contributors/docs/api/gpu-network/operators#gpunetworksnapping-gpunetworkcostmatrix-and-gpunetworkaccessibility"><code>GPUNetworkSnapping</code></a> (network)</td>
<td>Snaps points onto edges.</td>
</tr>
<tr>
<td>-</td>
<td><code>ST_Node</code>, <code>pgr_nodeNetwork</code></td>
<td><code>shapely.node</code></td>
<td>-</td>
<td>Feature To Line, GRASS <code>v.clean</code> (break)</td>
<td><a href="/luma-contributors/docs/api/gpu-network/operators#gpunetworknoding"><code>GPUNetworkNoding</code></a> (network)</td>
<td>Splits lines into edges and snaps their end points into nodes with a per-frame tolerance. Builds the CSR for routing.</td>
</tr>
<tr>
<td>-</td>
<td>-</td>
<td><code>networkx.line_graph</code>, <code>momepy.gdf_to_nx(approach='dual')</code>, igraph <code>linegraph</code></td>
<td>-</td>
<td>Turn restrictions (Network Analyst)</td>
<td><a href="/luma-contributors/docs/api/gpu-network/operators#gpunetworklinegraph"><code>GPUNetworkLineGraph</code></a> (network)</td>
<td>Line graph in CSR with turn angle, U-turn and banned-turn costs.</td>
</tr>
<tr>
<td>-</td>
<td>-</td>
<td>Valhalla and OSRM map matching, <code>fmm</code>, <code>leuvenmapmatching</code></td>
<td>-</td>
<td>-</td>
<td><a href="/luma-contributors/docs/api/gpu-network/operators#gpumapmatching"><code>GPUMapMatching</code></a> (network)</td>
<td>Hidden Markov model with bounded route search. Planar distances.</td>
</tr>
<tr>
<td>-</td>
<td>-</td>
<td>-</td>
<td><code>spaghetti.NetworkK</code>, <code>GlobalAutoK</code></td>
<td>SANET network K</td>
<td><a href="/luma-contributors/docs/api/gpu-network/operators#gpunetworkkfunction"><code>GPUNetworkKFunction</code></a> (network)</td>
<td>Network-constrained K with a simulated envelope. Build cross-K from separate source and target event tables when needed.</td>
</tr>
<tr>
<td>-</td>
<td><code>pgr_connectedComponents</code>, <code>pgr_pagerank</code>, <code>pgr_degree</code>, <code>pgr_betweenness</code></td>
<td><code>networkx</code> centralities</td>
<td><code>spaghetti</code></td>
<td>Network statistics</td>
<td><a href="/luma-contributors/docs/api/gpu-network/operators#gpunetworkstatistics"><code>GPUNetworkStatistics</code></a>, <a href="/luma-contributors/docs/api/gpu-network/operators#gpunetworkpathextraction-gpunetworkserviceareas-gpunetworkneighborhood-and-gpunetworkanalyticscolumns"><code>GPUNetworkAnalyticsColumns</code></a> (network)</td>
<td>-</td>
</tr>
<tr>
<td>-</td>
<td>-</td>
<td><code>networkx.quotient_graph</code></td>
<td>-</td>
<td>Dissolve by group (network)</td>
<td><a href="/luma-contributors/docs/api/gpu-network/operators#gpunetworkcoarsening"><code>GPUNetworkCoarsening</code></a> (network)</td>
<td>Summarises a CSR network by a vertex group label (community, component or cell) into supernodes for zoomed-out views.</td>
</tr>
<tr>
<td>-</td>
<td>-</td>
<td><code>networkx.ego_graph</code></td>
<td>-</td>
<td>-</td>
<td><a href="/luma-contributors/docs/api/gpu-network/operators#gpunetworkpathextraction-gpunetworkserviceareas-gpunetworkneighborhood-and-gpunetworkanalyticscolumns"><code>GPUNetworkNeighborhood</code></a> (network)</td>
<td>k-hop ego networks around per-frame seeds, up to a compile-time <code>maximumHops</code>.</td>
</tr>
<tr>
<td>-</td>
<td>-</td>
<td><code>networkx.subgraph</code></td>
<td>-</td>
<td>Extract by attribute</td>
<td><a href="/luma-contributors/docs/api/gpu-network/operators#gpunetworksubgraphfilter"><code>GPUNetworkSubgraphFilter</code></a> (network)</td>
<td>Induced subgraph from attribute and time predicates on vertex and edge rows.</td>
</tr>
<tr>
<td>-</td>
<td>-</td>
<td>-</td>
<td><code>access.Access.weighted_catchment</code>, <code>pandana</code></td>
<td>Network accessibility</td>
<td><a href="/luma-contributors/docs/api/gpu-network/operators#gpunetworksnapping-gpunetworkcostmatrix-and-gpunetworkaccessibility"><code>GPUNetworkAccessibility</code></a> (network)</td>
<td>Decay and catchment variants.</td>
</tr>
<tr>
<td>-</td>
<td>-</td>
<td><code>networkx.to_numpy_array</code></td>
<td>-</td>
<td>Adjacency matrix</td>
<td><a href="/luma-contributors/docs/api/gpu-network/operators#gpuadjacencymatrix-and-gpuadjacencymatrixorder"><code>GPUAdjacencyMatrix</code></a>, <a href="/luma-contributors/docs/api/gpu-network/operators#gpuadjacencymatrix-and-gpuadjacencymatrixorder"><code>GPUAdjacencyMatrixOrder</code></a> (network)</td>
<td>-</td>
</tr>
<tr>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>Edge bundling, Flow map</td>
<td><a href="/luma-contributors/docs/api/gpu-network/operators#gpuedgebundling"><code>GPUEdgeBundling</code></a>, <a href="/luma-contributors/docs/api/gpu-network/operators#gpuflowaggregation"><code>GPUFlowAggregation</code></a> (network)</td>
<td>Origin-destination aggregation by zone and time window.</td>
</tr>
</tbody>
</table>
<h2 id="raster">Raster</h2>
<table>
<thead>
<tr>
<th>turf</th>
<th>PostGIS</th>
<th>GeoPandas / Shapely</th>
<th>PySAL</th>
<th>QGIS / ArcGIS / CARTO</th>
<th>luma.gl</th>
<th>Notes</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>isolines</code>, <code>isobands</code></td>
<td><code>ST_Contour</code>, <code>ST_DumpAsPolygons</code></td>
<td><code>skimage.measure.find_contours</code>, <code>matplotlib contour</code></td>
<td>-</td>
<td>Contour, Contour polygons</td>
<td><a href="/luma-contributors/docs/api/gpu-raster/operations-analysis#gpuisolines-and-gpuisobands"><code>GPUIsolines</code></a>, <a href="/luma-contributors/docs/api/gpu-raster/operations-analysis#gpuisolines-and-gpuisobands"><code>GPUIsobands</code></a>, <a href="/luma-contributors/docs/api/gpu-spatial-analysis#postgis"><code>GPURasterContours</code></a> (raster)</td>
<td>Marching squares with indirect overlays. Overflow flags.</td>
</tr>
<tr>
<td>-</td>
<td><code>ST_MapAlgebra</code>, <code>ST_Clip</code></td>
<td><code>rasterio</code> + <code>numpy</code></td>
<td>-</td>
<td>Raster calculator, Raster Calculator</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis#postgis"><code>GPURasterBandMath</code></a>, <a href="/luma-contributors/docs/api/gpu-spatial-analysis#qgis"><code>GPURasterArithmetic</code></a>, <a href="/luma-contributors/docs/api/gpu-spatial-analysis#postgis"><code>GPURasterConditional</code></a> (raster)</td>
<td>Per-pixel math with nodata handling.</td>
</tr>
<tr>
<td>-</td>
<td><code>ST_Reclass</code></td>
<td><code>numpy.digitize</code></td>
<td>-</td>
<td>Reclassify by table, Reclassify</td>
<td><a href="/luma-contributors/docs/api/gpu-raster/operations-analysis#gpurasterreclassify-gpuweightedoverlay-and-local-raster-operations"><code>GPURasterReclassify</code></a> (raster)</td>
<td>-</td>
</tr>
<tr>
<td>-</td>
<td><code>ST_SummaryStats</code> per polygon</td>
<td><code>rasterstats.zonal_stats</code></td>
<td>-</td>
<td>Zonal statistics (raster), Zonal Statistics as Table</td>
<td><a href="/luma-contributors/docs/api/gpu-raster/operations-analysis#gpurasterzonalstatistics"><code>GPURasterZonalStatistics</code></a> (raster); recipe <code>addStraightLineCatchmentsRecipe</code></td>
<td>Deterministic sum order.</td>
</tr>
<tr>
<td>-</td>
<td><code>ST_AsRaster</code></td>
<td><code>rasterio.features.rasterize</code></td>
<td>-</td>
<td>Rasterize</td>
<td><a href="/luma-contributors/docs/api/gpu-raster/operations-analysis#gpupolygonrasterization-and-gpurasterjoin"><code>GPUPolygonRasterization</code></a>, <a href="/luma-contributors/docs/api/gpu-raster/operations-analysis#gpupolygonrasterization-and-gpurasterjoin"><code>GPURasterJoin</code></a> (raster)</td>
<td>-</td>
</tr>
<tr>
<td>-</td>
<td><code>ST_ValueCount</code>, <code>ST_Histogram</code></td>
<td><code>numpy.histogram</code></td>
<td>-</td>
<td>Raster layer unique values report, Histogram</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis#postgis"><code>GPURasterHistogram</code></a>, <a href="/luma-contributors/docs/api/gpu-spatial-analysis#postgis"><code>GPURasterStatistics</code></a> (raster)</td>
<td>Global tiled statistics use a first pass domain and a replay.</td>
</tr>
<tr>
<td>-</td>
<td>-</td>
<td><code>scipy.ndimage.distance_transform_edt</code></td>
<td>-</td>
<td>Proximity (raster distance), Euclidean Distance, Voronoi (raster)</td>
<td><a href="/luma-contributors/docs/api/gpu-raster/operations-analysis#gpudistancefield"><code>GPUDistanceField</code></a> (raster); recipe <code>addStraightLineCatchmentsRecipe</code></td>
<td>Euclidean distance and nearest-seed allocation. Ties break on the smallest seed ID.</td>
</tr>
<tr>
<td>-</td>
<td>-</td>
<td><code>skimage.graph.route_through_array</code></td>
<td>-</td>
<td>Cost Distance, Least-cost path, Cost Path</td>
<td><a href="/luma-contributors/docs/api/gpu-raster/operations-analysis#gpucostdistance-and-gpucostdistancepath"><code>GPUCostDistance</code></a>, <a href="/luma-contributors/docs/api/gpu-raster/operations-analysis#gpucostdistance-and-gpucostdistancepath"><code>GPUCostDistancePath</code></a> (raster)</td>
<td>Iterated on the GPU with convergence flags.</td>
</tr>
<tr>
<td>-</td>
<td>-</td>
<td><code>scipy.ndimage.label</code></td>
<td>-</td>
<td>Fill / clump, Region Group</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis#qgis"><code>GPURasterConnectedComponents</code></a> (raster)</td>
<td>Four or eight connectivity, bounded dense labels.</td>
</tr>
<tr>
<td>-</td>
<td>-</td>
<td><code>rasterio.features.sieve</code></td>
<td>-</td>
<td>Sieve</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis#qgis"><code>GPURasterSieve</code></a> (raster)</td>
<td>-</td>
</tr>
<tr>
<td>-</td>
<td>-</td>
<td><code>scipy.ndimage.gaussian_filter</code>, <code>uniform_filter</code>, <code>sobel</code>, <code>convolve</code></td>
<td>-</td>
<td>Gaussian filter, Convolve</td>
<td><a href="https://luma.gl/docs/api-reference/experimental/gpu-raster/operations-filters-morphology#separable-gaussian-smoothing"><code>GPURasterGaussianBlur</code></a>, <a href="https://luma.gl/docs/api-reference/experimental/gpu-raster/operations-filters-morphology#separable-box-smoothing"><code>GPURasterBoxBlur</code></a>, <a href="https://luma.gl/docs/api-reference/experimental/gpu-raster/operations-filters-morphology#direct-convolution"><code>GPURasterConvolution</code></a>, <a href="/luma-contributors/docs/api/gpu-spatial-analysis#postgis"><code>GPURasterSobel</code></a>, <a href="/luma-contributors/docs/api/gpu-spatial-analysis#qgis"><code>GPURasterScharr</code></a> (raster)</td>
<td>-</td>
</tr>
<tr>
<td>-</td>
<td>-</td>
<td><code>scipy.ndimage.binary_dilation</code>, <code>binary_erosion</code></td>
<td>-</td>
<td>Dilation, Erosion</td>
<td><a href="https://luma.gl/docs/api-reference/experimental/gpu-raster/operations-filters-morphology#binary-and-grayscale-morphology"><code>GPURasterDilation</code></a>, <a href="https://luma.gl/docs/api-reference/experimental/gpu-raster/operations-filters-morphology#binary-and-grayscale-morphology"><code>GPURasterErosion</code></a>, <a href="https://luma.gl/docs/api-reference/experimental/gpu-raster/operations-filters-morphology#binary-and-grayscale-morphology"><code>GPURasterOpening</code></a>, <a href="https://luma.gl/docs/api-reference/experimental/gpu-raster/operations-filters-morphology#binary-and-grayscale-morphology"><code>GPURasterClosing</code></a> (raster)</td>
<td>-</td>
</tr>
<tr>
<td>-</td>
<td>-</td>
<td><code>skimage.measure.regionprops</code></td>
<td>-</td>
<td>Landscape metrics, Region statistics</td>
<td><a href="https://luma.gl/docs/api-reference/experimental/gpu-raster/operations-regions-contours#four-connected-versus-eight-connected-regions"><code>GPURasterRegionMeasurements</code></a>, <a href="/luma-contributors/docs/api/gpu-raster/raster-patches"><code>GPURasterPatchMetrics</code></a> (raster)</td>
<td>-</td>
</tr>
<tr>
<td>-</td>
<td>-</td>
<td><code>skimage.filters.threshold_otsu</code></td>
<td>-</td>
<td>Threshold</td>
<td><a href="https://luma.gl/docs/api-reference/experimental/gpu-raster/operations-pixel#analytical-threshold-masks"><code>GPURasterOtsuThreshold</code></a>, <a href="https://luma.gl/docs/api-reference/experimental/gpu-raster/operations-filters-morphology#binary-classification-and-observation-validity"><code>GPURasterThreshold</code></a> (raster)</td>
<td>-</td>
</tr>
<tr>
<td>-</td>
<td><code>ST_MapAlgebra</code> NDVI</td>
<td><code>rasterio</code> NDVI</td>
<td>-</td>
<td>NDVI</td>
<td><a href="https://luma.gl/docs/api-reference/experimental/gpu-raster/operations-pixel#ndvi"><code>GPURasterNDVI</code></a> (raster)</td>
<td>-</td>
</tr>
<tr>
<td>-</td>
<td><code>ST_Value</code>, <code>ST_ValueCount</code></td>
<td><code>rasterio.sample</code></td>
<td>-</td>
<td>Sample raster values, Profile</td>
<td><a href="/luma-contributors/docs/api/gpu-raster/operations-analysis#gpurastersampling-and-gpurasterprofile"><code>GPURasterSampling</code></a>, <a href="/luma-contributors/docs/api/gpu-raster/operations-analysis#gpurastersampling-and-gpurasterprofile"><code>GPURasterProfile</code></a> (raster)</td>
<td>-</td>
</tr>
<tr>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>Weighted Overlay, Suitability</td>
<td><a href="/luma-contributors/docs/api/gpu-raster/operations-analysis#gpurasterreclassify-gpuweightedoverlay-and-local-raster-operations"><code>GPUWeightedOverlay</code></a> (raster)</td>
<td>-</td>
</tr>
<tr>
<td>-</td>
<td>-</td>
<td><code>numpy.diff</code> change maps</td>
<td>-</td>
<td>Change detection</td>
<td><a href="/luma-contributors/docs/api/gpu-raster/operations-analysis#gpuchangedetection"><code>GPUChangeDetection</code></a> (raster)</td>
<td>-</td>
</tr>
<tr>
<td>-</td>
<td>-</td>
<td><code>matplotlib.streamplot</code></td>
<td>-</td>
<td>Streamlines, LIC</td>
<td><a href="/luma-contributors/docs/api/gpu-raster/operations-analysis#gpustreamlines"><code>GPUStreamlines</code></a>, <a href="/luma-contributors/docs/api/gpu-raster/operations-analysis#gpulineintegralconvolution"><code>GPULineIntegralConvolution</code></a>, <a href="/luma-contributors/docs/api/gpu-raster/operations-analysis#gpuparticleadvection"><code>GPUParticleAdvection</code></a> (raster)</td>
<td>-</td>
</tr>
<tr>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>Contrast stretch</td>
<td><a href="/luma-contributors/docs/api/gpu-raster/operations-analysis#gpurasterstretch"><code>GPURasterStretch</code></a> (raster)</td>
<td>-</td>
</tr>
</tbody>
</table>
<h2 id="terrain">Terrain</h2>
<table>
<thead>
<tr>
<th>turf</th>
<th>PostGIS</th>
<th>GeoPandas / Shapely</th>
<th>PySAL</th>
<th>QGIS / ArcGIS / CARTO</th>
<th>luma.gl</th>
<th>Notes</th>
</tr>
</thead>
<tbody>
<tr>
<td>-</td>
<td><code>ST_Slope</code>, <code>ST_Aspect</code>, <code>ST_HillShade</code></td>
<td><code>richdem.TerrainAttribute</code>, <code>xrspatial.slope</code></td>
<td>-</td>
<td>Slope, Aspect, Hillshade</td>
<td><a href="/luma-contributors/docs/api/gpu-terrain/operators#gputerrainderivatives-gputerraincontours-gputerrainviewshed-gputerrainlineofsight-and-gputerraincumulativeviewshed"><code>GPUTerrainDerivatives</code></a> (terrain)</td>
<td>Horn slope, aspect and hillshade.</td>
</tr>
<tr>
<td>-</td>
<td>-</td>
<td><code>richdem.TerrainAttribute</code> (curvature)</td>
<td>-</td>
<td>Profile and plan curvature</td>
<td><a href="/luma-contributors/docs/api/gpu-terrain/operators#gputerraincurvature"><code>GPUTerrainCurvature</code></a> (terrain)</td>
<td>-</td>
</tr>
<tr>
<td>-</td>
<td>-</td>
<td><code>xrspatial.viewshed</code></td>
<td>-</td>
<td>Viewshed, Visibility analysis</td>
<td><a href="/luma-contributors/docs/api/gpu-terrain/operators#gputerrainderivatives-gputerraincontours-gputerrainviewshed-gputerrainlineofsight-and-gputerraincumulativeviewshed"><code>GPUTerrainViewshed</code></a>, <a href="/luma-contributors/docs/api/gpu-terrain/operators#gputerrainderivatives-gputerraincontours-gputerrainviewshed-gputerrainlineofsight-and-gputerraincumulativeviewshed"><code>GPUTerrainCumulativeViewshed</code></a>, <a href="/luma-contributors/docs/api/gpu-terrain/operators#gputerrainderivatives-gputerraincontours-gputerrainviewshed-gputerrainlineofsight-and-gputerraincumulativeviewshed"><code>GPUTerrainLineOfSight</code></a> (terrain)</td>
<td>Raster result on planar, projected-metre grids with a per-frame observer.</td>
</tr>
<tr>
<td>-</td>
<td>-</td>
<td><code>richdem.FlowAccumulation</code>, <code>pysheds</code></td>
<td>-</td>
<td>r.watershed, Flow accumulation, Fill sinks</td>
<td><a href="/luma-contributors/docs/api/gpu-terrain/operators#gputerrainflow"><code>GPUTerrainFlow</code></a> (terrain)</td>
<td>D8 routing, depression filling with convergence flags.</td>
</tr>
<tr>
<td>-</td>
<td>-</td>
<td><code>pysheds</code> catchments</td>
<td>-</td>
<td>Watershed basins, Strahler order, HAND</td>
<td><a href="/luma-contributors/docs/api/gpu-terrain/operators#gputerrainheightabovedrainage-gputerrainwatersheds-gputerrainstreamorder-and-gputerrainhydrologicindices"><code>GPUTerrainWatersheds</code></a>, <a href="/luma-contributors/docs/api/gpu-terrain/operators#gputerrainheightabovedrainage-gputerrainwatersheds-gputerrainstreamorder-and-gputerrainhydrologicindices"><code>GPUTerrainStreamOrder</code></a>, <a href="/luma-contributors/docs/api/gpu-terrain/operators#gputerrainheightabovedrainage-gputerrainwatersheds-gputerrainstreamorder-and-gputerrainhydrologicindices"><code>GPUTerrainHeightAboveDrainage</code></a>, <a href="/luma-contributors/docs/api/gpu-terrain/operators#gputerrainheightabovedrainage-gputerrainwatersheds-gputerrainstreamorder-and-gputerrainhydrologicindices"><code>GPUTerrainHydrologicIndices</code></a> (terrain)</td>
<td>-</td>
</tr>
<tr>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>Terrain Ruggedness Index, Roughness</td>
<td><a href="/luma-contributors/docs/api/gpu-terrain/operators#gputerrainruggedness-and-gputerrainvectorruggedness"><code>GPUTerrainRuggedness</code></a>, <a href="/luma-contributors/docs/api/gpu-terrain/operators#gputerrainruggedness-and-gputerrainvectorruggedness"><code>GPUTerrainVectorRuggedness</code></a> (terrain)</td>
<td>-</td>
</tr>
<tr>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>Topographic Position Index, Geomorphons</td>
<td><a href="/luma-contributors/docs/api/gpu-terrain/operators#gputerraintopographicposition-and-gputerrainweisslandforms"><code>GPUTerrainTopographicPosition</code></a>, <a href="/luma-contributors/docs/api/gpu-terrain/operators#gputerraintopographicposition-and-gputerrainweisslandforms"><code>GPUTerrainWeissLandforms</code></a>, <a href="/luma-contributors/docs/api/gpu-terrain/operators#gpugeomorphons"><code>GPUGeomorphons</code></a> (terrain)</td>
<td>-</td>
</tr>
<tr>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>Peaks, summits</td>
<td><a href="/luma-contributors/docs/api/gpu-terrain/operators#gputerrainsummits-gputerrainpeaksnap-gputerraincriticalpoints-and-gpuprofilepeaks"><code>GPUTerrainSummits</code></a>, <a href="/luma-contributors/docs/api/gpu-terrain/operators#gputerrainsummits-gputerrainpeaksnap-gputerraincriticalpoints-and-gpuprofilepeaks"><code>GPUTerrainPeakSnap</code></a>, <a href="/luma-contributors/docs/api/gpu-terrain/operators#gputerrainsummits-gputerrainpeaksnap-gputerraincriticalpoints-and-gpuprofilepeaks"><code>GPUTerrainCriticalPoints</code></a> (terrain)</td>
<td>-</td>
</tr>
<tr>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>Cast shadows, Solar radiation</td>
<td><a href="/luma-contributors/docs/api/gpu-terrain/operators#gputerrainhorizon-gpusolarshadowmask-gputerraincastshadow-gpusolarirradiance-gpureliefshading-gputextureshading-and-gpusolarposition"><code>GPUSolarShadowMask</code></a>, <a href="/luma-contributors/docs/api/gpu-terrain/operators#gputerrainhorizon-gpusolarshadowmask-gputerraincastshadow-gpusolarirradiance-gpureliefshading-gputextureshading-and-gpusolarposition"><code>GPUTerrainCastShadow</code></a>, <a href="/luma-contributors/docs/api/gpu-terrain/operators#gputerrainhorizon-gpusolarshadowmask-gputerraincastshadow-gpusolarirradiance-gpureliefshading-gputextureshading-and-gpusolarposition"><code>GPUSolarIrradiance</code></a>, <a href="/luma-contributors/docs/api/gpu-terrain/operators#gputerrainhorizon-gpusolarshadowmask-gputerraincastshadow-gpusolarirradiance-gpureliefshading-gputextureshading-and-gpusolarposition"><code>GPUTerrainHorizon</code></a> (terrain)</td>
<td>-</td>
</tr>
<tr>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>Terrain RGB decode, Fill sinks (spikes)</td>
<td><a href="/luma-contributors/docs/api/gpu-terrain/operators#gputerrainrgbdecode-and-gputerrainspikerepair"><code>GPUTerrainRGBDecode</code></a>, <a href="/luma-contributors/docs/api/gpu-terrain/operators#gputerrainrgbdecode-and-gputerrainspikerepair"><code>GPUTerrainSpikeRepair</code></a> (terrain)</td>
<td>-</td>
</tr>
<tr>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>Contours from DEM</td>
<td><a href="/luma-contributors/docs/api/gpu-terrain/operators#gputerrainderivatives-gputerraincontours-gputerrainviewshed-gputerrainlineofsight-and-gputerraincumulativeviewshed"><code>GPUTerrainContours</code></a> (terrain)</td>
<td>-</td>
</tr>
<tr>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>Relief shading, Multi-scale relief</td>
<td><a href="/luma-contributors/docs/api/gpu-terrain/operators#gputerrainhorizon-gpusolarshadowmask-gputerraincastshadow-gpusolarirradiance-gpureliefshading-gputextureshading-and-gpusolarposition"><code>GPUReliefShading</code></a>, <a href="/luma-contributors/docs/api/gpu-terrain/operators#gpusimplelocalrelief-gpumultiscalerelief-gpulocaldominance-and-gpureliefblend"><code>GPUMultiScaleRelief</code></a>, <a href="/luma-contributors/docs/api/gpu-terrain/operators#gpusimplelocalrelief-gpumultiscalerelief-gpulocaldominance-and-gpureliefblend"><code>GPUSimpleLocalRelief</code></a>, <a href="/luma-contributors/docs/api/gpu-terrain/operators#gpusimplelocalrelief-gpumultiscalerelief-gpulocaldominance-and-gpureliefblend"><code>GPULocalDominance</code></a>, <a href="/luma-contributors/docs/api/gpu-terrain/operators#gpusimplelocalrelief-gpumultiscalerelief-gpulocaldominance-and-gpureliefblend"><code>GPUReliefBlend</code></a> (terrain)</td>
<td>-</td>
</tr>
</tbody>
</table>
<h2 id="recipes">Recipes</h2>
<p>Recipes chain contributors with <code>addXRecipe(graph, props)</code>. See <a href="/luma-contributors/docs/api/gpu-spatial-analysis/recipes">Recipes</a> for the chains and notes.</p>
<table>
<thead>
<tr>
<th>Recipe</th>
<th>Replaces this workflow</th>
<th>Chain</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>addHotSpotAnalysisRecipe</code></td>
<td>PySAL <code>G_Local</code> + <code>fdr</code> + map, ArcGIS Hot Spot Analysis, CARTO <code>ST_GETISORD_H3</code></td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-cells#gpupointtocell-and-gpucellgeometry"><code>GPUPointToCell</code></a>, <a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-cells#gpucellaggregation-gpucellrollup-gpucellpyramid-and-gpucelllevelselection"><code>GPUCellAggregation</code></a>, <a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-weights#gpuneighborsearch"><code>GPUNeighborSearch</code></a>, <a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-statistics#gpuhotspotanalysis-and-gpulocalmoran"><code>GPUHotSpotAnalysis</code></a>, <a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-statistics#gpulocalpermutationtest-and-gpuglobalpermutationtest"><code>GPULocalPermutationTest</code></a></td>
</tr>
<tr>
<td><code>addRateClusterMapRecipe</code></td>
<td><code>esda.Moran_Local_Rate</code>, <code>Empirical_Bayes</code>, GeoDa cluster map</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-statistics#gpuempiricalbayesrates"><code>GPUEmpiricalBayesRates</code></a>, <a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-weights#gpucontiguityweights"><code>GPUContiguityWeights</code></a>, <a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-statistics#gpuhotspotanalysis-and-gpulocalmoran"><code>GPULocalMoran</code></a>, <a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-statistics#gpulocalpermutationtest-and-gpuglobalpermutationtest"><code>GPULocalPermutationTest</code></a></td>
</tr>
<tr>
<td><code>addPointsInPolygonsChoroplethRecipe</code></td>
<td><code>sjoin</code> + <code>groupby</code>, QGIS Count points in polygon</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-statistics#gpuzonalstatistics"><code>GPUZonalStatistics</code></a> or <a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-joins#gpupointinpolygonjoin-and-gpunearestfeaturejoin"><code>GPUPointInPolygonJoin</code></a> + <a href="https://luma.gl/docs/api-reference/experimental/gpu-dataframe-analysis#gpugroupstatistics"><code>GPUGroupStatistics</code></a>, then <a href="https://luma.gl/docs/api-reference/experimental/gpu-dataframe-analysis#gpucolumnquantiles-gpuclassbreaks-gpucolorscale-and-gpubivariateclassification"><code>GPUClassBreaks</code></a>, <a href="https://luma.gl/docs/api-reference/experimental/gpu-dataframe-analysis#gpucolumnquantiles-gpuclassbreaks-gpucolorscale-and-gpubivariateclassification"><code>GPUColorScale</code></a></td>
</tr>
<tr>
<td><code>addSpaceTimeHotSpotsRecipe</code></td>
<td>ArcGIS Emerging Hot Spot Analysis, CARTO <code>ST_SPACETIME_HOTSPOTS</code></td>
<td><a href="https://luma.gl/docs/api-reference/experimental/gpu-dataframe-analysis#gpucalendarbuckets"><code>GPUCalendarBuckets</code></a>, <a href="https://luma.gl/docs/api-reference/experimental/gpu-dataframe-analysis#gpugroupstatistics"><code>GPUGroupStatistics</code></a>, <a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-statistics#gpuemerginghotspots"><code>GPUEmergingHotSpots</code></a></td>
</tr>
<tr>
<td><code>addClusterAndOutlineRecipe</code></td>
<td>DBSCAN + <code>convex_hull</code> + area</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-statistics#gpuspatialclustering"><code>GPUSpatialClustering</code></a>, <a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-statistics#gpugroupgeometry"><code>GPUGroupGeometry</code></a>, <a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-statistics#gpugroupconvexhull"><code>GPUGroupConvexHull</code></a>, <a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-lines#gpugeometrymeasures-gpugeodesicpairs-and-gpugeodesicdestination"><code>GPUGeometryMeasures</code></a></td>
</tr>
<tr>
<td><code>addSpatialRegressionRecipe</code></td>
<td><code>spreg.OLS</code> + <code>LMtests</code> + residual Moran, ArcGIS OLS and GWR</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-regression#gpuordinaryleastsquares"><code>GPUOrdinaryLeastSquares</code></a>, <a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-regression#gpuspatialregressiondiagnostics"><code>GPUSpatialRegressionDiagnostics</code></a>, <a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-statistics#gpuhotspotanalysis-and-gpulocalmoran"><code>GPULocalMoran</code></a>, <a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-regression#gpugeographicallyweightedregression"><code>GPUGeographicallyWeightedRegression</code></a></td>
</tr>
<tr>
<td><code>addDriveTimeCatchmentRecipe</code></td>
<td>pgRouting <code>pgr_drivingDistance</code>, QGIS Service area, ArcGIS Service Area</td>
<td><a href="/luma-contributors/docs/api/gpu-network/operators#gpunetworksnapping-gpunetworkcostmatrix-and-gpunetworkaccessibility"><code>GPUNetworkSnapping</code></a>, <a href="/luma-contributors/docs/api/gpu-network/operators#gpunetworkpathextraction-gpunetworkserviceareas-gpunetworkneighborhood-and-gpunetworkanalyticscolumns"><code>GPUNetworkServiceAreas</code></a>, <a href="/luma-contributors/docs/api/gpu-network/operators#gpunetworkisochrones"><code>GPUNetworkIsochrones</code></a>, <a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-joins#gpupointinpolygonjoin-and-gpunearestfeaturejoin"><code>GPUPointInPolygonJoin</code></a>, <a href="https://luma.gl/docs/api-reference/experimental/gpu-dataframe-analysis#gpugroupstatistics"><code>GPUGroupStatistics</code></a></td>
</tr>
<tr>
<td><code>addStraightLineCatchmentsRecipe</code></td>
<td>QGIS Proximity + Zonal statistics, Thiessen catchments</td>
<td><a href="/luma-contributors/docs/api/gpu-raster/operations-analysis#gpudistancefield"><code>GPUDistanceField</code></a>, <a href="/luma-contributors/docs/api/gpu-raster/operations-analysis#gpurasterzonalstatistics"><code>GPURasterZonalStatistics</code></a></td>
</tr>
<tr>
<td><code>addChangeOfSupportRecipe</code></td>
<td><code>tobler.area_interpolate</code>, ArcGIS Apportion Polygon</td>
<td><a href="/luma-contributors/docs/api/gpu-raster/operations-analysis#gpupolygonrasterization-and-gpurasterjoin"><code>GPUPolygonRasterization</code></a>, <a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-accessibility#gpuarealinterpolation"><code>GPUArealInterpolation</code></a>, <a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-weights#gpuspatiallag"><code>GPUSpatialLag</code></a></td>
</tr>
<tr>
<td><code>addFleetDwellRecipe</code></td>
<td><code>movingpandas</code> stop detection + <code>sjoin</code></td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-lines#gputrajectorymetrics"><code>GPUTrajectoryMetrics</code></a>, <a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-joins#gpupointinpolygonjoin-and-gpunearestfeaturejoin"><code>GPUPointInPolygonJoin</code></a>, <a href="https://luma.gl/docs/api-reference/experimental/gpu-dataframe-analysis#gpugroupstatistics"><code>GPUGroupStatistics</code></a></td>
</tr>
<tr>
<td><code>addFleetDwellZoneEventsRecipe</code></td>
<td><code>movingpandas</code> zone events, geofence dwell</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-lines#gpuzoneevents"><code>GPUZoneEvents</code></a>, <a href="https://luma.gl/docs/api-reference/experimental/gpu-dataframe-analysis#gpugroupstatistics"><code>GPUGroupStatistics</code></a></td>
</tr>
<tr>
<td><code>addPeriodComparisonRecipe</code></td>
<td>Before and after change map (CARTO, ArcGIS Compare)</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-cells#gpucellaggregation-gpucellrollup-gpucellpyramid-and-gpucelllevelselection"><code>GPUCellAggregation</code></a>, <a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-cells#gpucelltablecompare"><code>GPUCellTableCompare</code></a>, <a href="https://luma.gl/docs/api-reference/experimental/gpu-dataframe-analysis#gpucolumnquantiles-gpuclassbreaks-gpucolorscale-and-gpubivariateclassification"><code>GPUClassBreaks</code></a>, <a href="https://luma.gl/docs/api-reference/experimental/gpu-dataframe-analysis#gpucolumnquantiles-gpuclassbreaks-gpucolorscale-and-gpubivariateclassification"><code>GPUColorScale</code></a></td>
</tr>
</tbody>
</table>
<h2 id="see-also">See also</h2>
<ul>
<li><a href="/luma-contributors/docs/api/gpu-spatial-analysis">GPU Spatial Analysis</a></li>
<li><a href="/luma-contributors/docs/api/gpu-network">GPU Network</a></li>
<li><a href="/luma-contributors/docs/api/gpu-terrain">GPU Terrain</a></li>
<li><a href="/luma-contributors/docs/api/gpu-raster/operations">GPURaster operations</a></li>
<li><a href="https://luma.gl/docs/api-reference/experimental/gpu-dataframe-analysis">GPU Dataframe analysis contributors</a></li>
</ul>`};export{e as default};