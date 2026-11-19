var e={slug:`api/gpu-spatial-analysis`,title:`GPU Spatial Analysis`,html:`<h1 id="gpu-spatial-analysis">GPU Spatial Analysis</h1>
<p>:::caution Experimental
<code>@luma-contributors/analysis/gpu-spatial-analysis</code> is an experimental entry point. APIs may change between
releases without a deprecation period.
:::</p>
<p>The <code>@luma-contributors/analysis/gpu-spatial-analysis</code> entry point is the analysis layer above luma.gl's
GPU spatial primitives. It does not reimplement them. It composes the
<a href="/luma-contributors/docs/api/geospatial">geospatial kernels</a> (projection, distances,
point-in-polygon, <code>GPUGridIndex</code>, <code>GPUPointSpatialQuery</code>) and the GPU Core primitives (sort, scan,
compaction, hash and grid indexes, <code>GPUBVH</code>, segmented reductions) into higher-level workflows:
spatial weights and their algebra, statistics computed on those weights, predicate and nearest
joins, geometry validity, density, discrete global grid cells, line and trajectory geometry,
regression, change of support, distribution dynamics, accessibility, and prebuilt recipes. New WGSL
is written only where no primitive fits.</p>
<p>Reach for the geospatial kernels when you need one primitive (a distance, a containment test, one
indexed query per frame). Reach for this entry point when the question is about every row at once,
or about statistics over the neighborhood structure.</p>
<p>For <a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-statistics#gpuglobalspatialstatistics">global Moran</a>,
<a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-statistics#gpuhotspotanalysis-and-gpulocalmoran">local Moran and hot-spot analysis</a>,
pass a neighbor search's <code>outputWeights</code> directly to the statistic. It carries completeness status
with the CSR buffers. Final-capacity or candidate-capacity loss suppresses inference; the statistic's
optional <code>output.status.overflow</code> and <code>candidateOverflow</code> forward the producer's flags separately.
Overflow is final-slot loss; candidateOverflow is intermediate loss, including contiguity pair scratch.
Chunk-local loss flags are rejected by these statistics; use aggregate status on the outer port.</p>
<p>See <a href="/luma-contributors/docs/api/gpu-spatial-analysis/expansion-lenses">expansion computation lenses</a> for executable examples of the new contributors and recipes. These cards expose actual command graphs and status outputs on explicit illustrative inputs.</p>
<h2 id="overview">Overview</h2>
<p>Each contributor is an algorithm or workflow that declares
resources and command nodes into a caller-owned <code>GPUCommandGraph</code> through <code>getCommandNodes(graph)</code>
(<code>GPUCommandNodeProducer</code>), and is added with <code>graph.add(contributor)</code>. They are composed from GPU
Core primitives (<code>GPUVisibilityWorkflow</code>, <code>GPUBVH</code>, <code>GPUGridBinning</code>, <code>GPUHistogram</code>, <code>GPUReduction</code>,
<code>GPUAncestorProjection</code>, and others) and the raster operators; new WGSL is written only where no
primitive fits.</p>
<h2 id="quick-start">Quick start</h2>
<p>Run the complete <a href="/luma-contributors/docs/api/operation-forms-and-cpu-twins#point-density-from-device-to-readback">point-density walkthrough</a>.
It creates its own device and five positions, then reads <code>[2, 0, 1, 1]</code> from a 2 × 2 grid and
releases every resource. Use that program as the shared execution scaffold for the other families.</p>
<p>Point density's domain-specific ports are:</p>
<table>
<thead>
<tr>
<th>Port</th>
<th>Format and rows</th>
<th>Meaning</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>positions</code></td>
<td><code>float32x2</code>, one row per point</td>
<td>Planar XY positions; nonfinite and out-of-bounds points are excluded.</td>
</tr>
<tr>
<td><code>bounds</code></td>
<td><code>float32</code>, 4 rows</td>
<td><code>[minimumX, minimumY, maximumX, maximumY]</code>; buffer contents can change each submission.</td>
</tr>
<tr>
<td><code>output.values</code></td>
<td><code>float32</code>, columns × rows</td>
<td>Per-cell statistic, X varying fastest; <code>'count'</code> is the default.</td>
</tr>
<tr>
<td><code>output.status.overflow</code></td>
<td>optional <code>uint32</code>, 1 row</td>
<td>Grid mode writes 0; hexagon mode writes 1 if an in-bounds point missed the lattice.</td>
</tr>
</tbody>
</table>
<p>The walkthrough requests count values only. Other statistics, weights, smoothing, extent,
histogram and texture outputs are described in the <a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-density">density reference</a>.</p>
<p>Read <a href="/luma-contributors/docs/api/gpu-spatial-analysis/concepts">GPU Spatial Analysis concepts</a> for the ontology, shared ports, conventions,
coordinate precision and the compile, encode and submit pattern. The operator pages group the
contributors by task; the <a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators">operators index</a> lists them all.</p>
<table>
<thead>
<tr>
<th>Section</th>
<th>Contributors</th>
</tr>
</thead>
<tbody>
<tr>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-weights">Spatial weights</a></td>
<td><code>GPUNeighborSearch</code>, <code>GPUContiguityWeights</code>, <code>GPULatticeWeights</code>, <code>GPUSpatialWeightsTransform</code>, <code>GPUSpatialWeightsAlgebra</code>, <code>GPUSpatialWeightsSummary</code>, <code>GPUSpatialWeightsTranspose</code>, <code>GPUSpatialWeightsValidation</code>, <code>GPUSpatialLag</code>, <code>GPUMapColoring</code></td>
</tr>
<tr>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-statistics">Spatial statistics</a></td>
<td><code>GPUGlobalSpatialStatistics</code>, <code>GPUMultipleTestingAdjustment</code>, <code>GPUAssuncaoReisStandardization</code>, <code>GPUHotSpotAnalysis</code>, <code>GPULocalMoran</code>, <code>GPULocalGeary</code>, <code>GPUSpatialPearson</code>, <code>GPUGammaStatistic</code>, <code>GPURoseStatistic</code>, <code>GPULocalPermutationTest</code>, <code>GPUGlobalPermutationTest</code>, <code>GPUNeighborhoodSummary</code>, <code>GPUEmpiricalBayesRates</code>, <code>GPUSpatialEmpiricalBayesRates</code>, <code>GPUSpatialClustering</code>, <code>GPUHierarchicalDensityClustering</code>, <code>GPUSpatialKMeans</code>, <code>GPUSpatialWeightsMinimumSpanningTree</code>, <code>GPUSkaterRegions</code>, <code>GPUAZPRegions</code>, <code>GPUMaxPRegions</code>, <code>GPUWardRegions</code>, <code>GPULocationAllocation</code>, <code>GPURegionPartitionEvaluation</code>, <code>GPUGroupGeometry</code>, <code>GPUGroupConvexHull</code>, <code>GPUEmergingHotSpots</code>, <code>GPUGeographicDistribution</code>, <code>GPURegionStatistics</code>, <code>GPUZonalStatistics</code>, <code>GPUFocalStatistics</code>, <code>GPUSimilarLocations</code></td>
</tr>
<tr>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-dynamics">Distribution dynamics and space-time tests</a></td>
<td><code>GPUClassAssignment</code>, <code>GPUClassificationFit</code>, <code>GPUTransitionMatrix</code>, <code>GPUSpatialMarkov</code>, <code>GPULISAMarkov</code>, <code>GPUTransitionDiagnostics</code>, <code>GPUKnoxTest</code>, <code>GPUMantelTest</code>, <code>GPUSpatialScanStatistic</code></td>
</tr>
<tr>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/space-time-statistics">Space-time statistics</a></td>
<td><code>GPUSpatialWeightsAlgebra</code> operation <code>'spaceTime'</code>, <code>GPUSpaceTimeLocalMoran</code>, <code>GPUSpaceTimeCrossCorrelation</code></td>
</tr>
<tr>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/space-time-clustering">Space-time clustering</a></td>
<td>ST-DBSCAN and ST-OPTICS through <code>times</code> on <code>GPUSpatialClustering</code> and <code>GPUOPTICS</code></td>
</tr>
<tr>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/movement-semantics">Movement semantics</a></td>
<td><code>GPUTrajectoryStaypoints</code>, <code>GPUTrajectoryLocations</code>, <code>GPUTrajectorySegments</code>, <code>GPUMobilityIndicators</code>, <code>GPUModalSplit</code>, <code>GPUTrajectoryOutlierFlags</code>, <code>GPUTrajectoryHampel</code>, <code>GPUTrajectorySplit</code>, <code>GPUTrajectoryThinning</code>, <code>GPUTrajectoryTrackRelativeTimes</code></td>
</tr>
<tr>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/track-similarity-measures">Track similarity measures</a></td>
<td>EDR, ERP, SSPD and bounded search on <code>GPUTrackSimilarity</code> and <code>GPUTrackSimilaritySearch</code></td>
</tr>
<tr>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/scoring-rules">Proper scoring rules</a></td>
<td><code>GPUEnsembleScores</code>, <code>GPUGaussianPredictiveScores</code></td>
</tr>
<tr>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/temporal-context">Temporal context</a></td>
<td><code>GPUTemporalContext</code> and the contributors that accept one</td>
</tr>
<tr>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-accessibility">Accessibility and change of support</a></td>
<td><code>GPUCatchmentAccessibility</code>, <code>GPUHuffTradeAreas</code>, <code>GPUArealInterpolation</code>, <code>GPUPycnophylactic</code>, <code>GPUSegregation</code></td>
</tr>
<tr>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-joins">Spatial joins</a></td>
<td><code>GPUSpatialPredicateJoin</code>, <code>GPUSpatialJoinPrepared</code>, <code>GPUSpatialJoinCandidates</code>, <code>GPUPointInPolygonJoin</code>, <code>GPUNearestFeatureJoin</code>, <code>GPUNearestFeatureWeights</code>, <code>GPUBufferSelection</code></td>
</tr>
<tr>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-validity">Geometry validity and topology</a></td>
<td><code>GPUSegmentIntersection</code>, <code>GPUGeometryValidity</code>, <code>GPULineNoding</code>, <code>GPUPolygonize</code>, <code>GPUMakeValid</code>, <code>GPUPolygonOverlay</code>, <code>GPUBufferSurface</code></td>
</tr>
<tr>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-density">Density &#x26; interpolation</a></td>
<td><code>GPUPointDensity</code>, <code>GPUKernelDensity</code>, <code>GPUInverseDistanceWeighting</code>, <code>GPUKriging</code>, <code>GPUDotDensity</code>, <code>GPURandomPointsInPolygon</code></td>
</tr>
<tr>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-cells">Cells (DGGS)</a></td>
<td><code>GPUCellAggregation</code>, <code>GPUCellRollup</code>, <code>GPUCellPyramid</code>, <code>GPUCellLevelSelection</code>, <code>GPUPointToCell</code>, <code>GPUCellGeometry</code>, <code>GPUCellTopology</code>, <code>GPUCellCompaction</code>, <code>GPUCellCover</code>, <code>GPUCellSetOutline</code>, <code>GPUSegmentRingAssembly</code>, <code>GPUCellTableCompare</code></td>
</tr>
<tr>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-lines">Lines &#x26; trajectories</a></td>
<td><code>GPULineSegmentize</code>, <code>GPUGreatCircleArcs</code>, <code>GPULineSmooth</code>, <code>GPULineChunk</code>, <code>GPULineSplit</code>, <code>GPULineMerge</code>, <code>GPUGeometryMeasures</code>, <code>GPUGeodesicPairs</code>, <code>GPUGeodesicDestination</code>, <code>GPULocalFrameProjection</code>, <code>GPULinearReferencing</code>, <code>GPULineLocate</code>, <code>GPULineSimplification</code>, <code>GPUCoverageSimplification</code>, <code>GPUTrajectoryMetrics</code>, <code>GPUTrajectoryPlayhead</code>, <code>GPUTrajectoryResample</code>, <code>GPUZoneEvents</code>, <code>GPUTrajectoryEncounters</code>, <code>GPUTrajectoryClosestApproach</code>, <code>addClockEncounters</code>, <code>GPUTrackSimilarity</code></td>
</tr>
<tr>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-geometry">Geometry utilities and tessellation</a></td>
<td><code>GPUOutlineGeometry</code>, <code>GPULabelPoint</code>, <code>GPUShapeDescriptors</code>, <code>GPULineDensity</code>, <code>GPULineLengthPerPolygon</code>, <code>GPUGridGenerator</code>, <code>GPUShapeGenerator</code>, <code>GPUHilbertKeys</code>, <code>GPURectangleClip</code>, <code>GPUDelaunayTessellation</code>, <code>GPUVoronoiDiagram</code></td>
</tr>
<tr>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-regression">Regression</a></td>
<td><code>GPUOrdinaryLeastSquares</code>, <code>GPUSpatialRegressionDiagnostics</code>, <code>GPUSpatialTwoStageLeastSquares</code>, <code>GPUSpatialErrorGM</code>, <code>GPUSpatialLagMaximumLikelihood</code>, <code>GPUSpatialErrorMaximumLikelihood</code>, <code>GPUSpatialWeightsTraces</code>, <code>GPUSpatialLagImpacts</code>, <code>GPUGeographicallyWeightedRegression</code>, <code>GPUGeographicallyWeightedRegressionNonstationarityTest</code></td>
</tr>
<tr>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/recipes">Recipes</a></td>
<td><code>addHotSpotAnalysisRecipe</code> and eleven more prebuilt chains</td>
</tr>
<tr>
<td><a href="#cross-reference">Cross-reference</a></td>
<td>turf, PostGIS, GeoPandas, PySAL, QGIS and ArcGIS tools mapped to contributors</td>
</tr>
</tbody>
</table>
<p>Contributors for networks, rasters and dataframe columns live in their own entries:
<a href="/luma-contributors/docs/api/gpu-network">GPU Network</a>,
<a href="https://luma.gl/docs/api-reference/experimental/gpu-raster/operations-regions-contours">GPU Raster</a> and
<a href="https://luma.gl/docs/api-reference/experimental/gpu-dataframe-analysis">GPU Dataframe</a>.</p>
<h2 id="catalog">Catalog</h2>
<p>One figure per contributor, each drawn from a recorded GPU run on real data; click through to its card.</p>
<p>::lens-catalog{entry=gpu-spatial-analysis}</p>
<h2 id="cross-reference">Cross-reference</h2>
<p>See the <a href="/luma-contributors/docs/api/gpu-spatial-analysis-cross-reference">task-organized cross-reference</a>.</p>
<p>Contributors live in <code>@luma-contributors/analysis/gpu-spatial-analysis</code> unless the entry says
<code>(network)</code>, <code>(raster)</code> or <code>(dataframe)</code>, meaning <code>gpu-network</code>, <code>gpu-raster</code> and <code>gpu-dataframe</code>.
Recipes are the <code>add...Recipe</code> builder functions. "Not built" rows point at the library to use
instead (see the GPU Core maintainer roadmap). Every contributor runs on the GPU with
bounded outputs and does not read back; the CPU libraries return unbounded geometry.</p>
<h3 id="turf">turf</h3>
<table>
<thead>
<tr>
<th>turf</th>
<th>luma.gl</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>area</code>, <code>length</code>, <code>centroid</code>, <code>bbox</code>, <code>center</code></td>
<td><code>GPUGeometryMeasures</code> (area, length or perimeter, centroid, bounds per feature and per group)</td>
</tr>
<tr>
<td><code>centerOfMass</code></td>
<td><code>GPUGeometryMeasures</code> centroid; <code>GPUGroupGeometry</code> for weighted centers of label groups</td>
</tr>
<tr>
<td><code>pointOnFeature</code></td>
<td><code>GPULabelPoint</code> (pole of inaccessibility)</td>
</tr>
<tr>
<td><code>distance</code>, <code>bearing</code>, <code>midpoint</code>, <code>rhumbDistance</code>, <code>rhumbBearing</code></td>
<td><code>GPUGeodesicPairs</code> (<code>model: 'rhumb'</code> for the rhumb forms)</td>
</tr>
<tr>
<td><code>destination</code>, <code>rhumbDestination</code></td>
<td><code>GPUGeodesicDestination</code></td>
</tr>
<tr>
<td><code>greatCircle</code></td>
<td><code>GPUGreatCircleArcs</code></td>
</tr>
<tr>
<td><code>along</code>, <code>lineSliceAlong</code></td>
<td><code>GPULineLocate</code></td>
</tr>
<tr>
<td><code>lineChunk</code>, <code>lineSegment</code></td>
<td><code>GPULineChunk</code>, <code>GPULineSegmentize</code></td>
</tr>
<tr>
<td><code>nearestPointOnLine</code></td>
<td><code>GPULinearReferencing</code></td>
</tr>
<tr>
<td><code>nearestPoint</code></td>
<td><code>GPUNearestFeatureJoin</code>; <code>GPUNeighborSearch</code> (knn)</td>
</tr>
<tr>
<td><code>pointsWithinPolygon</code>, <code>tag</code></td>
<td><code>GPUPointInPolygonJoin</code>; <code>GPUZonalStatistics</code> for counts and sums</td>
</tr>
<tr>
<td><code>collect</code></td>
<td><code>GPUZonalStatistics</code>, <code>GPUGroupStatistics</code> (dataframe)</td>
</tr>
<tr>
<td><code>buffer</code></td>
<td><code>GPUBufferSurface</code> for bounded repaired polygon topology; <code>GPUOutlineGeometry</code> for render-only geometry; <code>GPUBufferSelection</code> for "within distance" selection</td>
</tr>
<tr>
<td><code>simplify</code></td>
<td><code>GPULineSimplification</code>; <code>GPUCoverageSimplification</code> when neighbors must stay gap-free</td>
</tr>
<tr>
<td><code>polygonSmooth</code>, <code>bezierSpline</code></td>
<td><code>GPULineSmooth</code> (Chaikin)</td>
</tr>
<tr>
<td><code>bboxClip</code></td>
<td><code>GPURectangleClip</code></td>
</tr>
<tr>
<td><code>kinks</code>, <code>booleanValid</code></td>
<td><code>GPUGeometryValidity</code>; <code>GPUSegmentIntersection</code> for the crossing points</td>
</tr>
<tr>
<td><code>lineIntersect</code></td>
<td><code>GPUSegmentIntersection</code></td>
</tr>
<tr>
<td><code>squareGrid</code>, <code>hexGrid</code>, <code>triangleGrid</code>, <code>pointGrid</code></td>
<td><code>GPUGridGenerator</code></td>
</tr>
<tr>
<td><code>interpolate</code>, <code>idw</code></td>
<td><code>GPUInverseDistanceWeighting</code>; <code>GPUVariogram</code> to choose a model</td>
</tr>
<tr>
<td><code>isolines</code>, <code>isobands</code></td>
<td><code>GPUIsolines</code>, <code>GPUIsobands</code> (raster)</td>
</tr>
<tr>
<td><code>convex</code></td>
<td><code>GPUGroupConvexHull</code>; recipe <code>addClusterAndOutlineRecipe</code></td>
</tr>
<tr>
<td><code>clustersDbscan</code></td>
<td><code>GPUSpatialClustering</code></td>
</tr>
<tr>
<td><code>clustersKmeans</code></td>
<td><code>GPUSpatialKMeans</code></td>
</tr>
<tr>
<td><code>randomPoint</code> inside polygons, <code>sample</code></td>
<td><code>GPURandomPointsInPolygon</code>, <code>GPUDotDensity</code></td>
</tr>
<tr>
<td><code>hexbin</code>-style counting, <code>pointsWithinPolygon</code> + count</td>
<td><code>GPUPointDensity</code>; <code>GPUCellAggregation</code> for H3 or Quadbin cells</td>
</tr>
<tr>
<td><code>standardDeviationalEllipse</code>, <code>centralMean</code>, <code>medianCenter</code> (turf-extra)</td>
<td><code>GPUGeographicDistribution</code>, <code>GPUGroupGeometry</code></td>
</tr>
<tr>
<td><code>union</code>, <code>intersect</code>, <code>difference</code>, <code>dissolve</code>, <code>voronoi</code>, <code>tin</code>, <code>concave</code></td>
<td><code>GPUPolygonOverlay</code> for bounded Boolean overlay/dissolve; <code>GPUDelaunayTessellation</code> and <code>GPUVoronoiDiagram</code> for vector tessellation. Concave construction remains out of scope; coverage-only union is <code>GPUCoverageDissolve</code> and raster Voronoi is <code>GPUDistanceField</code></td>
</tr>
<tr>
<td><code>polygonize</code></td>
<td><code>GPUPolygonize</code> for arbitrary finite planar linework; <code>GPULineNoding</code> exposes the atomic noded substrate and provenance; <code>GPUSegmentRingAssembly</code> remains the lower-level directed-boundary primitive</td>
</tr>
</tbody>
</table>
<h3 id="postgis">PostGIS</h3>
<table>
<thead>
<tr>
<th>PostGIS</th>
<th>luma.gl</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>ST_Area</code>, <code>ST_Length</code>, <code>ST_Perimeter</code>, <code>ST_Centroid</code></td>
<td><code>GPUGeometryMeasures</code></td>
</tr>
<tr>
<td><code>ST_PointOnSurface</code>, <code>ST_MaximumInscribedCircle</code></td>
<td><code>GPULabelPoint</code></td>
</tr>
<tr>
<td><code>ST_Distance(geography)</code>, <code>ST_Azimuth</code></td>
<td><code>GPUGeodesicPairs</code></td>
</tr>
<tr>
<td><code>ST_Project</code></td>
<td><code>GPUGeodesicDestination</code></td>
</tr>
<tr>
<td><code>ST_DWithin</code></td>
<td><code>GPUNeighborSearch</code> (radius), <code>GPUBufferSelection</code>, <code>GPUSpatialPredicateJoin</code> with a distance</td>
</tr>
<tr>
<td><code>ST_Within</code>, <code>ST_Contains</code>, <code>ST_Intersects</code>, <code>ST_Covers</code>, <code>ST_Touches</code> joins</td>
<td><code>GPUPointInPolygonJoin</code> (points), <code>GPUSpatialPredicateJoin</code> (features); <code>GPUSpatialJoinCandidates</code> for the bounding-box stage alone</td>
</tr>
<tr>
<td><code>ST_Distance</code> KNN (<code>&#x3C;-></code>)</td>
<td><code>GPUNeighborSearch</code> (knn), <code>GPUNearestFeatureJoin</code></td>
</tr>
<tr>
<td><code>ST_ClusterDBSCAN</code></td>
<td><code>GPUSpatialClustering</code></td>
</tr>
<tr>
<td><code>ST_ClusterKMeans</code></td>
<td><code>GPUSpatialKMeans</code></td>
</tr>
<tr>
<td><code>ST_ConvexHull</code> per group</td>
<td><code>GPUGroupConvexHull</code></td>
</tr>
<tr>
<td><code>ST_Segmentize</code></td>
<td><code>GPULineSegmentize</code></td>
</tr>
<tr>
<td><code>ST_Simplify</code>, <code>ST_SimplifyVW</code></td>
<td><code>GPULineSimplification</code> (<code>method: 'visvalingam'</code> is approximate)</td>
</tr>
<tr>
<td><code>ST_CoverageSimplify</code></td>
<td><code>GPUCoverageSimplification</code></td>
</tr>
<tr>
<td><code>ST_IsValid</code>, <code>ST_IsValidDetail</code></td>
<td><code>GPUGeometryValidity</code> (polygons, lines, points)</td>
</tr>
<tr>
<td><code>ST_IsSimple</code>, <code>ST_IsRing</code>, <code>ST_IsClosed</code>, <code>ST_NPoints</code>, <code>ST_NumInteriorRings</code>, <code>ST_NumGeometries</code></td>
<td><code>GPUGeometryPredicates</code></td>
</tr>
<tr>
<td><code>ST_MinimumClearance</code>, <code>ST_MinimumClearanceLine</code></td>
<td><code>GPUMinimumClearance</code></td>
</tr>
<tr>
<td><code>ST_MinimumBoundingCircle</code>, <code>ST_OrientedEnvelope</code>, <code>ST_LongestLine</code>, <code>ST_MaxDistance</code></td>
<td><code>GPUMinimumBounds</code></td>
</tr>
<tr>
<td><code>ST_Affine</code>, <code>ST_Translate</code>, <code>ST_Scale</code>, <code>ST_Rotate</code>, <code>ST_Reverse</code>, <code>ST_ForcePolygonCW</code>, <code>ST_RemoveRepeatedPoints</code>, <code>ST_SnapToGrid</code></td>
<td><code>GPUAffineTransform</code>, <code>GPUGeometryOrientation</code>, <code>GPUGeometryCleanup</code></td>
</tr>
<tr>
<td><code>ST_OffsetCurve</code>, <code>ST_Snap</code></td>
<td><code>GPUOffsetCurve</code> (no loop removal), <code>GPUVertexSnap</code> (vertex to vertex)</td>
</tr>
<tr>
<td><code>ST_TriangulatePolygon</code></td>
<td><code>GPUPolygonTriangulation</code> (earcut, not constrained Delaunay)</td>
</tr>
<tr>
<td><code>ST_CoverageUnion</code>, <code>ST_CoverageInvalidEdges</code>, <code>ST_CoverageIsValid</code></td>
<td><code>GPUCoverageDissolve</code>, <code>GPUCoverageValidity</code></td>
</tr>
<tr>
<td><code>ST_LocalOutlierFactor</code> (Sedona)</td>
<td><code>GPULocalOutlierFactor</code></td>
</tr>
<tr>
<td><code>ST_ClipByBox2D</code></td>
<td><code>GPURectangleClip</code></td>
</tr>
<tr>
<td><code>ST_LineInterpolatePoint</code>, <code>ST_LocateAlong</code></td>
<td><code>GPULineLocate</code></td>
</tr>
<tr>
<td><code>ST_LineLocatePoint</code>, <code>ST_ClosestPoint</code> on lines</td>
<td><code>GPULinearReferencing</code></td>
</tr>
<tr>
<td><code>ST_SquareGrid</code>, <code>ST_HexagonGrid</code></td>
<td><code>GPUGridGenerator</code></td>
</tr>
<tr>
<td><code>ST_HausdorffDistance</code>, <code>ST_FrechetDistance</code></td>
<td><code>GPUTrackSimilarity</code></td>
</tr>
<tr>
<td><code>ST_Length(ST_Intersection(line, cell))</code> summed per cell</td>
<td><code>GPULineDensity</code></td>
</tr>
<tr>
<td><code>ST_Buffer</code></td>
<td><code>GPUBufferSurface</code> for repaired polygon output; <code>GPUOutlineGeometry</code> for render-only geometry</td>
</tr>
<tr>
<td><code>ST_Intersection</code>, <code>ST_Union</code>, <code>ST_Difference</code>, <code>ST_SymDifference</code>, general <code>ST_Dissolve</code></td>
<td><code>GPUPolygonOverlay</code> with explicit arrangement and output capacities</td>
</tr>
<tr>
<td><code>ST_DelaunayTriangles</code>, <code>ST_VoronoiLines</code></td>
<td><code>GPUDelaunayTessellation</code>, then <code>GPUVoronoiDiagram</code> with dynamic clip bounds</td>
</tr>
<tr>
<td><code>ST_MakeValid</code></td>
<td><code>GPUMakeValid</code> rebuilds planar polygon boundaries from their noded bounded faces and reports collapsed/unused edges</td>
</tr>
<tr>
<td><code>ST_Polygonize</code>, <code>ST_BuildArea</code>, <code>ST_MakePolygon</code> (from boundary segments or closed lines)</td>
<td><code>GPUPolygonize</code>; <code>GPULineNoding</code> and <code>GPUSegmentRingAssembly</code> expose the independently measurable noding and face-extraction stages</td>
</tr>
<tr>
<td><code>ST_Clip</code>, <code>ST_MapAlgebra</code>, <code>ST_Reclass</code> (raster)</td>
<td><code>GPURasterBandMath</code>, <code>GPURasterReclassify</code>, <code>GPURasterConditional</code> (raster)</td>
</tr>
<tr>
<td><code>ST_Slope</code>, <code>ST_Aspect</code>, <code>ST_Hillshade</code></td>
<td><code>GPURasterGradient</code>, <code>GPURasterGradientMagnitude</code>, <code>GPURasterSobel</code> (raster)</td>
</tr>
<tr>
<td><code>ST_Contour</code>, <code>ST_DumpAsPolygons</code> on thresholds</td>
<td><code>GPUIsolines</code>, <code>GPUIsobands</code>, <code>GPURasterContours</code> (raster)</td>
</tr>
<tr>
<td><code>ST_ValueCount</code>, <code>ST_Histogram</code></td>
<td><code>GPURasterHistogram</code>, <code>GPURasterStatistics</code> (raster)</td>
</tr>
<tr>
<td><code>ST_SummaryStats</code> per polygon (zonal)</td>
<td><code>GPURasterZonalStatistics</code> (raster)</td>
</tr>
<tr>
<td><code>ST_AsRaster</code></td>
<td><code>GPUPolygonRasterization</code> (raster)</td>
</tr>
<tr>
<td><code>pgRouting</code> <code>pgr_dijkstraCost</code>, <code>pgr_drivingDistance</code></td>
<td><code>GPUNetworkReachability</code>, <code>GPUNetworkCostMatrix</code>, <code>GPUNetworkIsochrones</code> (network, <code>cellOutline.rings</code> for isochrone polygons); recipe <code>addDriveTimeCatchmentRecipe</code> (<code>isochroneRings</code>, demand join)</td>
</tr>
<tr>
<td><code>pgr_withPointsDD</code> (points on edges)</td>
<td><code>GPUNetworkSnapping</code> (network)</td>
</tr>
<tr>
<td><code>pgr_connectedComponents</code>, <code>pgr_pagerank</code>, <code>pgr_degree</code></td>
<td><code>GPUNetworkAnalyticsColumns</code>, <code>GPUNetworkStatistics</code> (network)</td>
</tr>
<tr>
<td><code>h3_lat_lng_to_cell</code>, <code>h3_cell_to_boundary</code>, <code>h3_grid_disk</code>, <code>h3_polygon_to_cells</code></td>
<td><code>GPUPointToCell</code>, <code>GPUCellGeometry</code>, <code>GPUCellTopology</code>, <code>GPUCellCover</code></td>
</tr>
<tr>
<td><code>h3_cells_to_multi_polygon_wkb</code>, h3-js <code>cellsToMultiPolygon</code></td>
<td><code>GPUCellSetOutline</code> with <code>rings</code> (<code>GPUSegmentRingAssembly</code>): closed shells and holes of an H3 or Quadbin cell set on the GPU, with <code>ringShells</code> and polygon layout</td>
</tr>
<tr>
<td><code>h3</code> roll-up <code>GROUP BY h3_cell_to_parent</code></td>
<td><code>GPUCellAggregation</code>, <code>GPUCellRollup</code>, <code>GPUCellPyramid</code></td>
</tr>
</tbody>
</table>
<h3 id="geopandas-and-shapely">GeoPandas and Shapely</h3>
<table>
<thead>
<tr>
<th>GeoPandas / Shapely</th>
<th>luma.gl</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>GeoSeries.area</code>, <code>.length</code>, <code>.centroid</code>, <code>.bounds</code></td>
<td><code>GPUGeometryMeasures</code></td>
</tr>
<tr>
<td><code>GeoSeries.representative_point()</code></td>
<td><code>GPULabelPoint</code></td>
</tr>
<tr>
<td><code>GeoSeries.distance</code>, <code>pyproj.Geod.inv</code> / <code>.fwd</code></td>
<td><code>GPUGeodesicPairs</code>, <code>GPUGeodesicDestination</code></td>
</tr>
<tr>
<td><code>GeoDataFrame.sjoin</code></td>
<td><code>GPUSpatialPredicateJoin</code>, <code>GPUPointInPolygonJoin</code></td>
</tr>
<tr>
<td><code>GeoDataFrame.sjoin_nearest</code></td>
<td><code>GPUNearestFeatureJoin</code></td>
</tr>
<tr>
<td><code>sjoin</code> then <code>groupby(...).agg</code></td>
<td>recipe <code>addPointsInPolygonsChoroplethRecipe</code>; <code>GPUZonalStatistics</code>; <code>GPUGroupStatistics</code> (dataframe)</td>
</tr>
<tr>
<td><code>GeoSeries.convex_hull</code> per <code>groupby</code></td>
<td><code>GPUGroupConvexHull</code>; recipe <code>addClusterAndOutlineRecipe</code></td>
</tr>
<tr>
<td><code>GeoSeries.simplify</code>, <code>shapely.simplify_coverage</code></td>
<td><code>GPULineSimplification</code>, <code>GPUCoverageSimplification</code></td>
</tr>
<tr>
<td><code>GeoSeries.is_valid</code>, <code>shapely.explain_validity</code></td>
<td><code>GPUGeometryValidity</code></td>
</tr>
<tr>
<td><code>shapely.segmentize</code></td>
<td><code>GPULineSegmentize</code></td>
</tr>
<tr>
<td><code>shapely.clip_by_rect</code>, <code>GeoDataFrame.clip</code> (rectangle)</td>
<td><code>GPURectangleClip</code></td>
</tr>
<tr>
<td><code>shapely.line_interpolate_point</code>, <code>.project</code></td>
<td><code>GPULineLocate</code>, <code>GPULinearReferencing</code></td>
</tr>
<tr>
<td><code>shapely.hausdorff_distance</code>, <code>.frechet_distance</code> (with <code>densify</code>)</td>
<td><code>GPUTrackSimilarity</code></td>
</tr>
<tr>
<td><code>shapely.set_precision</code> (pointwise), <code>remove_repeated_points</code>, <code>reverse</code>, <code>orient_polygons</code>, <code>affinity.*</code></td>
<td><code>GPUGeometryCleanup</code>, <code>GPUGeometryOrientation</code>, <code>GPUAffineTransform</code></td>
</tr>
<tr>
<td><code>GeoSeries.buffer</code> (as geometry)</td>
<td><code>GPUBufferSurface</code> (planar; line and polygon joins/caps plus repair)</td>
</tr>
<tr>
<td><code>shapely.offset_curve</code>, <code>snap</code></td>
<td><code>GPUOffsetCurve</code>, <code>GPUVertexSnap</code> (partial, see their sections)</td>
</tr>
<tr>
<td><code>shapely.minimum_rotated_rectangle</code>, <code>oriented_envelope</code>, <code>minimum_bounding_circle</code>, <code>minimum_bounding_radius</code></td>
<td><code>GPUMinimumBounds</code></td>
</tr>
<tr>
<td><code>shapely.minimum_clearance</code>, <code>minimum_clearance_line</code></td>
<td><code>GPUMinimumClearance</code></td>
</tr>
<tr>
<td><code>shapely.is_simple</code>, <code>is_ring</code>, <code>is_closed</code>, <code>is_ccw</code>, <code>get_num_points</code>, <code>extract_unique_points</code>, <code>equals_exact</code></td>
<td><code>GPUGeometryPredicates</code></td>
</tr>
<tr>
<td><code>GeoSeries.dissolve(method='coverage')</code>, <code>shapely.coverage_union_all</code>, <code>coverage_is_valid</code>, <code>coverage_invalid_edges</code></td>
<td><code>GPUCoverageDissolve</code>, <code>GPUCoverageValidity</code></td>
</tr>
<tr>
<td><code>geopandas.clip(lines, polygon)</code>, <code>shapely.shared_paths</code></td>
<td><code>GPULineClipByPolygon</code>, <code>GPUSharedPaths</code></td>
</tr>
<tr>
<td><code>sjoin</code> merge, <code>sjoin_nearest(distance_col)</code>, <code>explode</code>, <code>get_parts</code>, <code>.cx</code></td>
<td><code>GPUPairGather</code>, <code>GPUOffsetExpansion</code>, <code>GPUBoundsFilter</code></td>
</tr>
<tr>
<td><code>GeoSeries.make_grid</code>, <code>sample_points</code> (lines)</td>
<td><code>GPUGridGenerator</code> (<code>hexOrientation</code>, <code>extent</code>), <code>GPURandomPointsOnLine</code></td>
</tr>
<tr>
<td><code>overlay</code>, <code>dissolve</code>, <code>unary_union</code></td>
<td><code>GPUPolygonOverlay</code>; coverage-only dissolve remains <code>GPUCoverageDissolve</code></td>
</tr>
<tr>
<td><code>voronoi_polygons</code>, <code>delaunay_triangles</code>, <code>make_valid</code></td>
<td><code>GPUVoronoiDiagram</code>, <code>GPUDelaunayTessellation</code>, <code>GPUMakeValid</code></td>
</tr>
<tr>
<td><code>concave_hull</code></td>
<td>Not built; a definition and parameter contract remain evidence-gated</td>
</tr>
<tr>
<td><code>shapely.polygonize</code>, <code>polygonize_full</code></td>
<td><code>GPULineNoding</code> plus <code>GPUPolygonize</code>; diagnostics distinguish rings, open chains, cut edges, dangles, unused edges, and degenerates</td>
</tr>
<tr>
<td><code>shapely.minimum_bounding_circle</code>, <code>oriented_envelope</code></td>
<td><code>GPUShapeDescriptors</code> (elongation, orientation); <code>GPUGroupGeometry</code> (bounds, ellipse)</td>
</tr>
<tr>
<td><code>tobler.area_interpolate</code></td>
<td><code>GPUArealInterpolation</code>; recipe <code>addChangeOfSupportRecipe</code></td>
</tr>
<tr>
<td><code>tobler.pycno</code></td>
<td><code>GPUPycnophylactic</code></td>
</tr>
<tr>
<td><code>rasterstats.zonal_stats</code></td>
<td><code>GPURasterZonalStatistics</code> (raster); recipe <code>addStraightLineCatchmentsRecipe</code></td>
</tr>
<tr>
<td><code>rasterio.features.rasterize</code></td>
<td><code>GPUPolygonRasterization</code> (raster)</td>
</tr>
<tr>
<td><code>momepy</code> compactness (<code>circular_compactness</code>, <code>convexity</code>, <code>elongation</code>, <code>orientation</code>)</td>
<td><code>GPUShapeDescriptors</code></td>
</tr>
<tr>
<td><code>momepy.describe_agg</code>, <code>libpysal.weights.lag_categorical</code></td>
<td><code>GPUNeighborhoodSummary</code></td>
</tr>
<tr>
<td><code>mapclassify.classify</code> (<code>Quantiles</code>, <code>EqualInterval</code>, <code>FisherJenks</code>, <code>HeadTailBreaks</code>, <code>BoxPlot</code>, <code>StdMean</code>, <code>MaximumBreaks</code>, <code>UserDefined</code>)</td>
<td><code>GPUClassBreaks</code> (dataframe); <code>GPUClassAssignment</code>; <code>GPUColorScale</code> (dataframe)</td>
</tr>
<tr>
<td><code>mapclassify</code> bivariate / <code>geoplot</code> choropleth</td>
<td><code>GPUBivariateClassification</code> (dataframe)</td>
</tr>
<tr>
<td><code>sklearn.cluster.DBSCAN</code>, <code>KMeans</code> on coordinates</td>
<td><code>GPUSpatialClustering</code>, <code>GPUSpatialKMeans</code></td>
</tr>
<tr>
<td><code>movingpandas</code> stop detection, trajectory stats</td>
<td><code>GPUTrajectoryMetrics</code>; <code>GPUZoneEvents</code> for zone entry/exit; recipe <code>addFleetDwellRecipe</code></td>
</tr>
<tr>
<td><code>movingpandas</code> interpolation / resample</td>
<td><code>GPUTrajectoryPlayhead</code>, <code>GPUTrajectoryResample</code></td>
</tr>
<tr>
<td><code>gstat</code> / <code>scikit-gstat</code> <code>Variogram</code></td>
<td><code>GPUVariogram</code> (dataframe)</td>
</tr>
</tbody>
</table>
<h3 id="pysal">PySAL</h3>
<table>
<thead>
<tr>
<th>PySAL</th>
<th>luma.gl</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>libpysal.weights.Queen</code>, <code>Rook</code></td>
<td><code>GPUContiguityWeights</code></td>
</tr>
<tr>
<td><code>libpysal.weights.lat2W</code></td>
<td><code>GPULatticeWeights</code></td>
</tr>
<tr>
<td><code>KNN</code>, <code>DistanceBand</code>, <code>Kernel</code>, <code>Kernel(fixed=False)</code></td>
<td><code>GPUNeighborSearch</code> (knn, radius, kernel weights); <code>GPUSpatialWeightsTransform</code> (kernel)</td>
</tr>
<tr>
<td><code>W.transform = 'R'</code> / <code>'B'</code>, symmetrize</td>
<td><code>GPUSpatialWeightsTransform</code></td>
</tr>
<tr>
<td><code>weights.set_operations</code> (<code>w_union</code>, <code>w_intersection</code>), <code>higher_order</code>, <code>w_subset</code>, <code>block_weights</code></td>
<td><code>GPUSpatialWeightsAlgebra</code></td>
</tr>
<tr>
<td><code>W.s0</code>, <code>.s1</code>, <code>.s2</code>, <code>.cardinalities</code>, <code>.islands</code>, <code>.asymmetry()</code></td>
<td><code>GPUSpatialWeightsSummary</code></td>
</tr>
<tr>
<td><code>lag_spatial</code></td>
<td><code>GPUSpatialLag</code></td>
</tr>
<tr>
<td><code>esda.Moran</code>, <code>Geary</code>, <code>G</code> (General G), <code>Moran_BV</code>, <code>Join_Counts</code></td>
<td><code>GPUGlobalSpatialStatistics</code></td>
</tr>
<tr>
<td><code>esda.Moran</code> <code>.p_sim</code>, <code>.z_sim</code></td>
<td><code>GPUGlobalPermutationTest</code></td>
</tr>
<tr>
<td><code>esda.Moran_Local</code></td>
<td><code>GPULocalMoran</code>; <code>GPULocalPermutationTest</code> for <code>p_sim</code></td>
</tr>
<tr>
<td><code>esda.G_Local</code> (Gi, Gi*)</td>
<td><code>GPUHotSpotAnalysis</code>; <code>GPULocalPermutationTest</code> (<code>'localG'</code>, <code>'localGStar'</code>); recipe <code>addHotSpotAnalysisRecipe</code></td>
</tr>
<tr>
<td><code>smoothing.Empirical_Bayes</code></td>
<td><code>GPUEmpiricalBayesRates</code> (<code>clampPriorVariance: false</code> for exact parity); recipe <code>addRateClusterMapRecipe</code></td>
</tr>
<tr>
<td><code>esda.Moran_Rate</code>, <code>Moran_Local_Rate</code>, <code>assuncao_rate</code></td>
<td><code>GPUAssuncaoReisStandardization</code> followed by <code>GPUGlobalSpatialStatistics</code> or <code>GPULocalMoran</code></td>
</tr>
<tr>
<td><code>esda.fdr</code>, <code>statsmodels.multipletests</code></td>
<td><code>GPUMultipleTestingAdjustment</code>; <code>multipleTesting</code> option of <code>GPUHotSpotAnalysis</code>, <code>GPULocalMoran</code>, <code>GPULocalPermutationTest</code></td>
</tr>
<tr>
<td><code>pointpats.PointPattern</code>, <code>ripley.k_test</code>, <code>f_test</code>, <code>g_test</code>, <code>j_test</code></td>
<td><code>GPUPointPatternIndices</code>, <code>GPURipley</code>, <code>GPURipleyDistanceFunctions</code> (dataframe)</td>
</tr>
<tr>
<td><code>pointpats.Knox</code>, <code>Mantel</code></td>
<td><code>GPUKnoxTest</code>, <code>GPUMantelTest</code> (<code>distanceTransform</code> and <code>timeTransform</code> options for the <code>mantel</code> transforms)</td>
</tr>
<tr>
<td><code>esda.Moran</code> correlogram (<code>spatial_correlogram</code>)</td>
<td><code>GPUSpatialCorrelogram</code> (dataframe)</td>
</tr>
<tr>
<td><code>esda.Spatial_Pearson</code>, <code>Spatial_Pearson_Local</code> (Lee's L)</td>
<td><code>GPUGlobalPermutationTest</code> (<code>'leesL'</code>), <code>GPULocalPermutationTest</code> (<code>'localLeeL'</code>)</td>
</tr>
<tr>
<td><code>esda.Local_Geary</code>, <code>Gamma</code></td>
<td><code>GPULocalGeary</code>, <code>GPUGammaStatistic</code>; shared permutation adapters use <code>GPUGlobalPermutationTest</code> or <code>GPULocalPermutationTest</code></td>
</tr>
<tr>
<td><code>spreg.OLS</code></td>
<td><code>GPUOrdinaryLeastSquares</code></td>
</tr>
<tr>
<td><code>spreg.diagnostics_sp.LMtests</code>, <code>MoranRes</code></td>
<td><code>GPUSpatialRegressionDiagnostics</code>; recipe <code>addSpatialRegressionRecipe</code></td>
</tr>
<tr>
<td><code>spreg.GM_Lag</code></td>
<td><code>GPUSpatialTwoStageLeastSquares</code></td>
</tr>
<tr>
<td><code>spreg.ML_Lag</code>, <code>ML_Error</code> (<code>method='full'</code>), <code>spat_impacts</code></td>
<td><code>GPUSpatialLagMaximumLikelihood</code>, <code>GPUSpatialErrorMaximumLikelihood</code>; impacts <code>GPUSpatialLagImpacts</code> over <code>GPUSpatialWeightsTraces</code></td>
</tr>
<tr>
<td><code>spreg</code> panel, SUR, regimes, probit</td>
<td>Use <code>GPUSpatialPanelRegression</code> for pooled/unit-effects spatial-lag panels, <code>GPUSpatialRegimeRegression</code> for fully varying regime OLS, and <code>spreg</code> when SUR or probit inference is required</td>
</tr>
<tr>
<td><code>mgwr.GWR</code>, <code>mgwr.MGWR</code></td>
<td><code>GPUGeographicallyWeightedRegression</code>; <code>GPUMultiscaleGeographicallyWeightedRegression</code> with bounded backfitting, LOO/AICc bandwidth selection and bandwidth-conditioned inference</td>
</tr>
<tr>
<td><code>giddy.Markov</code>, <code>LISA_Markov</code>, <code>Spatial_Markov</code></td>
<td><code>GPUTransitionMatrix</code>, <code>GPULISAMarkov</code>, <code>GPUSpatialMarkov</code></td>
</tr>
<tr>
<td><code>giddy.directional</code>, <code>Rose</code></td>
<td><code>GPURoseStatistic</code> with explicit directional or axial circular semantics</td>
</tr>
<tr>
<td><code>segregation</code> aspatial and spatial indices</td>
<td><code>GPUSegregation</code></td>
</tr>
<tr>
<td><code>inequality.gini.Gini</code>, <code>theil.Theil</code>, <code>Gini_Spatial</code></td>
<td><code>GPUInequality</code> (dataframe)</td>
</tr>
<tr>
<td><code>esda.shape</code> (<code>isoperimetric_quotient</code>, <code>convex_hull_ratio</code>)</td>
<td><code>GPUShapeDescriptors</code></td>
</tr>
<tr>
<td><code>spopt.region.AZP</code>, <code>MaxPHeuristic</code>, <code>Skater</code>, <code>Ward</code></td>
<td><code>GPUAZPRegions</code> (greedy steepest-descent AZP), <code>GPUMaxPRegions</code> (randomized multi-start greedy construction with optional AZP refinement), <code>GPUSkaterRegions</code>, <code>GPUWardRegions</code> (bounded adjacent Ward linkage)</td>
</tr>
<tr>
<td><code>spopt.locate.PMedian</code>, <code>MCLP</code>, <code>LSCP</code></td>
<td><code>GPULocationAllocation</code> provides bounded greedy p-median, maximum-coverage, and set-covering selection (plus a swap improvement phase) over <code>GPUNetworkCostMatrix</code> or another dense cost matrix; use <code>spopt</code> when an integer-programming optimum or richer constraints are required</td>
</tr>
<tr>
<td><code>access.Access.two_step_fca</code>, <code>weighted_catchment</code>, <code>Huff</code></td>
<td><code>GPUCatchmentAccessibility</code>, <code>GPUHuffTradeAreas</code>, <code>GPUNetworkAccessibility</code> (network)</td>
</tr>
<tr>
<td><code>spaghetti</code>, <code>pandana</code> network distance and accessibility</td>
<td><code>GPUNetworkReachability</code>, <code>GPUNetworkCostMatrix</code>, <code>GPUNetworkAccessibility</code> (network)</td>
</tr>
<tr>
<td><code>PySAL</code> hot spot workflow (Gi* + FDR + map)</td>
<td>recipe <code>addHotSpotAnalysisRecipe</code></td>
</tr>
</tbody>
</table>
<h3 id="qgis">QGIS</h3>
<table>
<thead>
<tr>
<th>QGIS tool</th>
<th>luma.gl</th>
</tr>
</thead>
<tbody>
<tr>
<td>Create grid (rectangle, hexagon)</td>
<td><code>GPUGridGenerator</code></td>
</tr>
<tr>
<td>Heatmap (kernel density)</td>
<td><code>GPUPointDensity</code></td>
</tr>
<tr>
<td>Count points in polygon, Zonal statistics (vector)</td>
<td><code>GPUZonalStatistics</code>; recipe <code>addPointsInPolygonsChoroplethRecipe</code></td>
</tr>
<tr>
<td>Join attributes by location, Join attributes by location (summary)</td>
<td><code>GPUSpatialPredicateJoin</code>, <code>GPUPointInPolygonJoin</code>, <code>GPUZonalStatistics</code></td>
</tr>
<tr>
<td>Join attributes by nearest</td>
<td><code>GPUNearestFeatureJoin</code></td>
</tr>
<tr>
<td>Extract by location, Select by location</td>
<td><code>GPUSpatialPredicateJoin</code>, <code>GPUBufferSelection</code>, <code>GPURegionMask</code></td>
</tr>
<tr>
<td>Zonal statistics (raster)</td>
<td><code>GPURasterZonalStatistics</code> (raster)</td>
</tr>
<tr>
<td>Rasterize (vector to raster)</td>
<td><code>GPUPolygonRasterization</code> (raster)</td>
</tr>
<tr>
<td>Raster calculator</td>
<td><code>GPURasterBandMath</code>, <code>GPURasterArithmetic</code> (raster)</td>
</tr>
<tr>
<td>Reclassify by table / layer</td>
<td><code>GPURasterReclassify</code> (raster)</td>
</tr>
<tr>
<td>Slope, Aspect, Hillshade, TRI</td>
<td><code>GPURasterGradient</code>, <code>GPURasterSobel</code>, <code>GPURasterScharr</code> (raster)</td>
</tr>
<tr>
<td>Contour, Contour polygons</td>
<td><code>GPUIsolines</code>, <code>GPUIsobands</code> (raster)</td>
</tr>
<tr>
<td>Sieve, Fill / clump</td>
<td><code>GPURasterSieve</code>, <code>GPURasterConnectedComponents</code> (raster)</td>
</tr>
<tr>
<td>Proximity (raster distance), Voronoi (raster)</td>
<td><code>GPUDistanceField</code> (raster); recipe <code>addStraightLineCatchmentsRecipe</code></td>
</tr>
<tr>
<td>Cost distance / Least-cost path</td>
<td><code>GPUCostDistance</code>, <code>GPUCostDistancePath</code> (raster)</td>
</tr>
<tr>
<td>IDW interpolation</td>
<td><code>GPUInverseDistanceWeighting</code></td>
</tr>
<tr>
<td>Service area (from point), Service area (from layer)</td>
<td><code>GPUNetworkReachability</code>, <code>GPUNetworkIsochrones</code> (network, ring polygons through <code>cellOutline.rings</code>); recipe <code>addDriveTimeCatchmentRecipe</code></td>
</tr>
<tr>
<td>Polygonize / Fix geometries</td>
<td><code>GPULineNoding</code>, <code>GPUPolygonize</code>, and <code>GPUMakeValid</code>; <code>GPUSegmentRingAssembly</code> remains the lower-level already-noded primitive</td>
</tr>
<tr>
<td>Shortest path (point to point / layer)</td>
<td><code>GPUNetworkReachability</code> with <code>GPUNetworkPathExtraction</code> (network)</td>
</tr>
<tr>
<td>Distance matrix</td>
<td><code>GPUNeighborSearch</code>; <code>GPUNetworkCostMatrix</code> for network cost</td>
</tr>
<tr>
<td>Nearest neighbour analysis</td>
<td><code>GPUPointPatternIndices</code> (dataframe)</td>
</tr>
<tr>
<td>Random points in polygons</td>
<td><code>GPURandomPointsInPolygon</code>, <code>GPUDotDensity</code></td>
</tr>
<tr>
<td>Pole of inaccessibility</td>
<td><code>GPULabelPoint</code></td>
</tr>
<tr>
<td>Polygon centroids, Centroids</td>
<td><code>GPUGeometryMeasures</code></td>
</tr>
<tr>
<td>Mean coordinate(s), Standard deviational ellipse (Processing / plugin)</td>
<td><code>GPUGeographicDistribution</code>, <code>GPUGroupGeometry</code></td>
</tr>
<tr>
<td>Convex hull, Minimum bounding geometry</td>
<td><code>GPUGroupConvexHull</code>, <code>GPUGroupGeometry</code></td>
</tr>
<tr>
<td>Simplify, Smooth, Densify by interval</td>
<td><code>GPULineSimplification</code>, <code>GPULineSmooth</code>, <code>GPULineSegmentize</code></td>
</tr>
<tr>
<td>Clip by extent</td>
<td><code>GPURectangleClip</code></td>
</tr>
<tr>
<td>Line intersections</td>
<td><code>GPUSegmentIntersection</code></td>
</tr>
<tr>
<td>Check validity</td>
<td><code>GPUGeometryValidity</code></td>
</tr>
<tr>
<td>Line density, Sum line lengths</td>
<td><code>GPULineDensity</code></td>
</tr>
<tr>
<td>DBSCAN clustering, K-means clustering</td>
<td><code>GPUSpatialClustering</code>, <code>GPUSpatialKMeans</code></td>
</tr>
<tr>
<td>Cluster-and-outline workflow (cluster, hull, area)</td>
<td>recipe <code>addClusterAndOutlineRecipe</code></td>
</tr>
<tr>
<td>Moran's I, Getis-Ord Gi*, LISA (SAGA, GeoDa plugin)</td>
<td><code>GPUGlobalSpatialStatistics</code>, <code>GPUHotSpotAnalysis</code>, <code>GPULocalMoran</code></td>
</tr>
<tr>
<td>Graduated symbology modes (Quantile, Natural Breaks, Pretty, Std Dev, Equal)</td>
<td><code>GPUClassBreaks</code>, <code>GPUColorScale</code> (dataframe)</td>
</tr>
<tr>
<td>Dissolve, Union, Intersection, Difference, Buffer (geometry), Delaunay, Voronoi (vector), Fix geometries</td>
<td><code>GPUPolygonOverlay</code>, <code>GPUBufferSurface</code>, <code>GPUDelaunayTessellation</code>, <code>GPUVoronoiDiagram</code>, and <code>GPUMakeValid</code>; spatial-index and virtual-raster ownership remain in their dedicated packages</td>
</tr>
<tr>
<td>Model Designer / Processing graphical model</td>
<td>Recipes, or compose contributors in one <code>GPUCommandGraph</code></td>
</tr>
<tr>
<td>ArcGIS Hot Spot Analysis (Getis-Ord Gi*), Emerging Hot Spot Analysis</td>
<td><code>GPUHotSpotAnalysis</code>, <code>GPUEmergingHotSpots</code>; recipes <code>addHotSpotAnalysisRecipe</code>, <code>addSpaceTimeHotSpotsRecipe</code></td>
</tr>
<tr>
<td>ArcGIS Spatial Autocorrelation, Incremental Spatial Autocorrelation</td>
<td><code>GPUGlobalSpatialStatistics</code>, <code>GPUSpatialCorrelogram</code> (dataframe)</td>
</tr>
<tr>
<td>ArcGIS Directional Distribution, Mean Center, Median Center</td>
<td><code>GPUGeographicDistribution</code></td>
</tr>
<tr>
<td>ArcGIS Generate Network Spatial Weights, Generate Spatial Weights Matrix</td>
<td><code>GPUContiguityWeights</code>, <code>GPUNeighborSearch</code>, <code>GPUNetworkCostMatrix</code></td>
</tr>
<tr>
<td>ArcGIS Geographically Weighted Regression, Ordinary Least Squares</td>
<td><code>GPUGeographicallyWeightedRegression</code>, <code>GPUOrdinaryLeastSquares</code></td>
</tr>
<tr>
<td>ArcGIS Calculate Composite Index</td>
<td><code>GPUCompositeScore</code> (dataframe)</td>
</tr>
</tbody>
</table>
<h2 id="defaults-that-break-a-quick-parity-check">Defaults that break a quick parity check</h2>
<p>These defaults differ from the reference library or a common expectation. Each is documented where the operator is described.</p>
<table>
<thead>
<tr>
<th>Area</th>
<th>Default here</th>
<th>Reference behaviour</th>
<th>For parity</th>
</tr>
</thead>
<tbody>
<tr>
<td>Islands (rows with no neighbor)</td>
<td><code>islands: 'exclude'</code> removes them from <code>n</code> and the moments (<code>GPU_DEFAULT_ISLAND_POLICY</code>)</td>
<td>esda keeps islands in <code>n</code> and the moments</td>
<td><code>islands: 'keep'</code></td>
</tr>
<tr>
<td>Two-sided permutation p-value</td>
<td><code>alternative: 'two-sided'</code> reports <code>min(2 (min(g, l) + 1), P + 1) / (P + 1)</code></td>
<td>esda's undirected test is percentile based and differs under ties; its legacy default <code>'directed'</code> is anti-conservative</td>
<td><code>'folded'</code> matches esda's undirected test; <code>'greater'</code> and <code>'lesser'</code> match esda's one-sided tests</td>
</tr>
<tr>
<td>k-means coordinates</td>
<td><code>GPUSpatialKMeans</code> sums raw float32 coordinates (<code>centerCoordinates</code> defaults to <code>false</code>)</td>
<td>Raw float32 sums lose precision for coordinates far from the origin</td>
<td><code>centerCoordinates: true</code> for coordinates far from the origin</td>
</tr>
<tr>
<td>KDE edge correction</td>
<td><code>'uniform'</code> divides each cell by the kernel mass inside the window</td>
<td>Uncorrected estimate, or spatstat <code>diggle = TRUE</code></td>
<td><code>'none'</code> or <code>'diggle'</code></td>
</tr>
<tr>
<td>Line density</td>
<td><code>GPULineDensity</code> writes line length per grid cell (<code>lengths</code>); <code>densities</code> (length per area) is optional</td>
<td>A density raster, or per-polygon length (<code>GPULineLengthPerPolygon</code>)</td>
<td>Read <code>lengths</code>, or divide by cell area</td>
</tr>
</tbody>
</table>
<h2 id="see-also">See also</h2>
<ul>
<li>
<p><a href="/luma-contributors/docs/api/geospatial">Geospatial kernels</a> for the projection, distance, point-in-polygon, grid-index and point-query primitives these contributors compose</p>
</li>
<li>
<p><a href="https://luma.gl/docs/api-reference/experimental/gpu-core/gpu-command-graph">GPUCommandGraph</a></p>
</li>
<li>
<p><a href="/luma-contributors/docs/api/gpu-network">GPU Network</a> and <a href="/luma-contributors/docs/api/gpu-terrain">GPU Terrain</a> for the network and elevation analysis contributors</p>
</li>
<li>
<p><a href="https://luma.gl/docs/api-reference/experimental/gpu-dataframe-analysis">GPU Dataframe analysis contributors</a> for the column statistics, joins, and time filters the spatial contributors consume</p>
</li>
<li>
<p><a href="https://luma.gl/docs/api-reference/experimental/gpu-core/gpu-grid-index">GPUGridIndex</a></p>
</li>
<li>
<p><a href="/luma-contributors/docs/api/gpu-spatial-analysis/adaptive-generalized-gwr">Adaptive generalized GWR</a></p>
</li>
<li>
<p><a href="/luma-contributors/docs/api/gpu-spatial-analysis/panel-fixed-effects">Unit fixed effects</a></p>
</li>
<li>
<p><a href="/luma-contributors/docs/api/gpu-spatial-analysis/losh-residual-permutation">LOSH residual permutation</a></p>
</li>
</ul>`};export{e as default};