var e={slug:`api/gpu-spatial-analysis/recipes`,title:`GPU Spatial Analysis: Recipes`,html:`<h1 id="recipes">Recipes</h1>
<p>See <a href="/luma-contributors/docs/api/gpu-spatial-analysis/expansion-recipes">Geospatial workflow recipes</a> for zonal distributions, network and facility scenarios, interpolation diagnostics, exposure, raster change, drainage, track quality, local variability, cell flows and surface preparation.</p>
<p>The local-variability recipe retains every declared bounded-status view on its intermediate
weights port and named weights stage, including approximation, uncertainty and iteration diagnostics.</p>
<p>Recipes are plain functions, not classes: <code>addXRecipe(graph, props)</code> constructs the contributors, calls
<code>graph.add</code> in chain order, wires outputs to inputs and returns a <code>GPURecipeResult</code>: <code>{contributors, outputs, intermediates, status}</code>. <code>outputs</code> holds the named terminal views, <code>intermediates</code> the named ports between stages and
<code>status</code> the GPU-resident completeness and convergence views, grouped by stage. <code>contributors</code>
lists every stage the recipe added, in order: composed contributors, and each raw glue kernel as one named
stage. Published calibration stages and drive-time road/time/band adapters are included in that
same declaration-order list. Status counts over rows or cells (invalid
rows, unresolved cells) run as a row-parallel kernel plus a published <code>GPUReduction</code> sum, never as one
invocation looping over the rows. Views you
pass in (<code>zScores</code>, <code>bins</code>, <code>colors</code>, ...) are caller-owned and readable; every other intermediate is a
graph transient. Inference recipes (hot spot, rate cluster, space-time hot spots, spatial regression) take
one <code>settings</code> object and <code>GPUParameterBuffer</code>s they fill themselves, derive <code>result.inference</code> from the same
object and repack it with <code>result.setSettings(settings)</code>; the other recipes pass per-frame parameter views
(<code>getGPU...ParameterValues</code>) through. Neither rebuilds the graph. Geometry inputs are the shared ports:
<code>GPUPointGeometryPort</code>, <code>GPUPolygonGeometryPort</code> (four-level layout, one row per feature) and
<code>GPUPackedGeometryPort&#x3C;GPUTrajectoryGeometryPort></code>.</p>
<p>All 24 <code>add…Recipe</code> functions exported by the spatial recipes entry point and all four exported by
the network recipes entry point prepare their complete graph
declaration before changing the caller's graph. Contributor constructors and cross-stage checks
finish before the builders expand and schedule contributors. Preparation uses a separate,
uncompiled graph with the same device limits, preserving buffer aliases, view offsets, texture
subresources and vector chunks. Nested recipes share their enclosing preparation. If validation
or contributor expansion fails, the caller's nodes, resources and parameter-buffer contents stay
unchanged: correct the configuration and retry with the same recipe ID. Temporary contributors
that own resources are destroyed even when a later stage fails. Parameter writes are validated
without upload and applied only after successful scheduling; frozen <code>GPUParameterBuffer</code> wrappers
are supported. Later <code>setSettings</code> calls still write immediately. Contributors and interleaved
adapter kernels retain their declaration order when scheduling is deferred.</p>
<p>Auxiliary luma resources and devices carried alongside a recipe's ports keep their identities and
ownership. Preparation treats them as opaque references and does not inspect their native state;
graph views are mirrored separately. Each consumed configuration field, including inherited getters
and JavaScript Proxy reads, is captured once and shared by both declarations. Unrelated fields are
not evaluated. Configuration records returned in metadata are captured facades; caller graph views,
parameter buffers and declared prepared-join handles retain their original identities.
Retained committed metadata is detached from the temporary preparation graph after preflight.
The every-stage rollback matrix covers the 28 functions in the two recipes barrels; the public
family inventory also lists 13 domain recipes outside that matrix, with their own family contracts.</p>
<p>Preparation repeats host declaration work once and never compiles, encodes, submits or reads data.
Use unused recipe/resource IDs on a mutable graph. The published graph has no mutable node-ID
inspection or transaction API, so this configuration guarantee cannot cover collisions with
pre-existing caller node IDs, externally overridden scheduling methods, a graph already frozen by
compilation, or allocation/device failure during final declaration. Successful recipes retain the
same concrete command nodes, outputs and status meanings.</p>
<p>Recipe implementation form and result compatibility are checked separately. Population support
alignment returns an empty <code>intermediates</code> map and three named diagnostic substages under
<code>status.stages</code>; their scalar <code>invalidCount</code> views alias its missing-target, ambiguous-target and
excluded-source words. Use <code>result.packedStatus</code> for the original four-word certificate. The
contributor and both operation forms retain their packed <code>output.status</code> contract.</p>
<p>The hot-spot, rate-cluster, space-time, fleet (both variants), period-comparison, change-of-support,
cluster-and-outline, spatial-regression, points-in-polygons and straight-line-catchment results type
their <code>outputs</code> and <code>intermediates</code> maps. The exported
<code>GPU&#x3C;Name>RecipeOutputs</code> and, where present, <code>GPU&#x3C;Name>RecipeIntermediates</code> types describe those
maps; the top-level convenience fields remain available. For example, <code>spaceTime.outputs.cube</code>
feeds <code>GPUEmergingHotSpotsProps.values</code>, and <code>fleet.outputs.stops.centroids</code> is a typed
<code>float32x2</code> view. Optional stages remain optional in both paths, so check for
<code>hotSpots.outputs.permutation</code> or <code>spaceTime.outputs.colors</code> before using them.</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> hotSpots</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> addHotSpotAnalysisRecipe</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(graph, {</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  source: {kind: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'points'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, positions, family: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'quadbin'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, resolution: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">8</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, cellCapacity: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">4096</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    studyArea: {kind: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'polygons'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, positions: studyPositions, featureOffsets, polygonOffsets, ringOffsets,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">      candidateCapacity: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">16384</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}},</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  settings: {significanceLevel: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">0.05</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, seed: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">7</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, permutationCount: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">199</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">},</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  parameters: autocorrelationParameters, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// GPUParameterBuffers the recipe fills from settings</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  outputs: {zScores, bins},</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  permutation: {parameters: permutationParameters, maximumPermutations: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">199</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">},</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  color: {classBreaksParameters, maximumClassCount: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">5</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, colorScaleParameters, palette, maximumPaletteCount: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">5</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">});</span></span></code></pre>
<table>
<thead>
<tr>
<th>Recipe</th>
<th>Chain</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>addHotSpotAnalysisRecipe</code></td>
<td>GPUCellCover study area -> GPUCellAggregation (sorted unique study cells) + (GPUPointToCell) -> GPUCellAggregation (points) -> join (zeros kept) -> area density (GPUCellMeasures for H3) -> GPUCellTopology disk -> CSR weights (or GPULatticeWeights for raster source) -> GPUHotSpotAnalysis -> GPULocalPermutationTest -> GPUClassBreaks -> GPUColorScale</td>
</tr>
<tr>
<td><code>addRateClusterMapRecipe</code></td>
<td>GPUEmpiricalBayesRates -> GPUContiguityWeights -> GPUSpatialWeightsTransform(row) -> GPULocalMoran (<code>quadrantGating: 'none'</code> with permutation) -> GPULocalPermutationTest -> quadrant colors</td>
</tr>
<tr>
<td><code>addPointsInPolygonsChoroplethRecipe</code></td>
<td>backend 'zonal' (GPUZonalStatistics) or 'group' (GPUPointInPolygonJoin -> dense GPUGroupStatistics) -> GPUClassBreaks -> GPUColorScale</td>
</tr>
<tr>
<td><code>addSpaceTimeHotSpotsRecipe</code></td>
<td>GPUCalendarBuckets -> GPUPeriodIndex (absolute time-step slices) -> atomic count into the dense cube (sorted GPUGroupStatistics path when its scratch or overflow outputs are passed) -> GPUEmergingHotSpots -> ordinal GPUColorScale</td>
</tr>
<tr>
<td><code>addClusterAndOutlineRecipe</code></td>
<td>GPUSpatialClustering -> GPUGroupGeometry + GPUGroupConvexHull -> GPUGeometryMeasures</td>
</tr>
<tr>
<td><code>addSpatialRegressionRecipe</code></td>
<td>GPUOrdinaryLeastSquares -> GPUSpatialRegressionDiagnostics -> GPU specification decision (Anselin's rule: 0 OLS, 1 lag, 2 error, 3 both robust, 4 undetermined) + residual GPULocalMoran (+ optional GPUGeographicallyWeightedRegression); <code>mask</code> and <code>restandardizeRows</code> flow through OLS and diagnostics</td>
</tr>
<tr>
<td><code>addDriveTimeCatchmentRecipe</code> (from <code>gpu-network</code>)</td>
<td>GPUNetworkSnapping -> GPUNetworkServiceAreas -> (GPUNetworkIsochrones raster and/or cell outline with rings -> GPUPointInPolygonJoin demand join) -> snap demand -> band adapters -> GPUGroupStatistics</td>
</tr>
<tr>
<td><code>addStraightLineCatchmentsRecipe</code></td>
<td>GPUDistanceField allocation -> GPURasterZonalStatistics</td>
</tr>
<tr>
<td><code>addChangeOfSupportRecipe</code></td>
<td>2x GPUPolygonRasterization -> GPUArealInterpolation (with source conservation diagnostics) -> GPUSpatialLag</td>
</tr>
<tr>
<td><code>addFleetDwellRecipe</code></td>
<td>GPUTrajectoryMetrics stops -> active-stop mask and join positions -> GPUPointInPolygonJoin -> GPUGroupStatistics (dense, one row per zone)</td>
</tr>
<tr>
<td><code>addFleetDwellZoneEventsRecipe</code></td>
<td>GPUZoneEvents -> GPUGroupStatistics (dense, one row per zone; zone-event variant of the fleet dwell recipe)</td>
</tr>
<tr>
<td><code>addPeriodComparisonRecipe</code></td>
<td>2x GPUCellAggregation -> GPUCellTableCompare (with mask) -> GPUClassBreaks -> GPUColorScale</td>
</tr>
</tbody>
</table>
<p>Notes:</p>
<ul>
<li>
<p>Rate-cluster weights publish <code>weightsCandidateOverflow</code> for <code>pairCapacity</code> loss and
<code>weightsOverflow</code> for final <code>neighborCapacity</code> loss. Inspect both flags. Grow <code>pairCapacity</code> first
when the candidate flag is set; <code>weightsRequiredCount</code> is then only a lower bound for neighbor slots.
The Kepler rate-cluster binding refuses either loss flag and uses its CPU fallback. Explorer and
showcase readouts distinguish pair loss from final neighbor loss and mark dependent results unavailable.</p>
</li>
<li>
<p>Spatial regression diagnostics suppress statistics and p-values with status code 3 on either
weights-loss flag; specification selection returns 4 (undetermined). Clean reuse restores inference.
Caller-provided regression writes cannot alias either weights-loss input; rejection precedes all
resource registration so correcting the output permits a same-ID retry.</p>
</li>
<li>
<p>All caller-provided fleet stop outputs, including durations, row indices and status words, must
have logical byte ranges disjoint from <code>scratch.stopMask</code>. Adjacent views
that share an aligned storage binding remain supported through separate adapter dispatches.</p>
</li>
<li>
<p>Fleet zone-event lenses and website readouts distinguish final event overflow, combined event
candidate loss (scratch or per-track), and scratch-only loss. Either event-loss flag makes crossing
counts incomplete; only scratch loss makes dense dwell totals incomplete. Pending readbacks are
labelled pending. Truncated per-track event lists can retain complete dwell totals.
Crossing counts require the count and zone-column readbacks; positions are needed only for drawing
dots. Missing numeric readbacks are pending or unavailable, including missing dwell totals, never zero.</p>
</li>
<li>
<p>The space-time recipe counts events into a dense <code>cell * sliceCount + slice</code> cube with integer atomics,
because <code>GPUCellAggregation</code> keys must be valid Quadbin or H3 cells. Passing the sorted-path scratch
views or <code>outputs.status.groupOverflow</code> selects the <code>GPUGroupStatistics</code> path instead.
Group status views need at least one packed uint32 word; spare words are untouched.
On the supplied-cell-ID route, the <code>weights</code> status stage preserves both <code>overflow</code> and
<code>candidateOverflow</code>; a nonzero flag means neighborhood inference is incomplete. These input
words cannot alias recipe writes. The slice interval, including seven days per week, must fit
the period index's signed divisor limit of 2^31 - 1 before recipe resources are registered.
The origin's whole calendar-step index must fit [-2^31, 2^31 - 1] too; hourly indices count hours
from the Unix epoch. This check also precedes allocation. Every declared writable port and every
published weights status word (including invalidCount) participates in alias validation. Declared
fields, including inherited/getter-backed status fields, are captured once for validation and scheduling.
<code>slices.count</code> is at most 256; trend autocorrelation correction has the lower limit documented by
<code>GPUEmergingHotSpots</code>. Every contributor, settings object and glue binding is validated before any
recipe command nodes are scheduled. <code>contributors</code> includes <code>GPUPeriodIndex</code> and every glue stage
in execution order on both counting paths.
Event positions must be packed <code>float32x2</code>, and event cell IDs packed <code>uint32</code>, with exactly one
row per timestamp. Strided positions are rejected; pack them before construction. Caller-view
ownership, shape and binding conflicts, parameter length, settings and palette preconditions are
checked before registering recipe resources, so correcting those props permits retry on the same
graph and recipe ID. Settings are packed on the host first and written to the caller's parameter
buffer only after scheduling succeeds; a rejected second recipe cannot change an existing
recipe's settings through a shared parameter buffer.
Initial packing preserves all classification settings, including <code>persistentFraction</code>,
<code>minimumTrendSliceCount</code> and <code>sporadicRequiresFinalSlice</code>; an identical <code>setSettings</code> call writes
identical parameter words. A six-slice series can carry a trend when its requested minimum is three.</p>
</li>
<li>
<p>The drive-time catchment recipe classifies demand in network space (the minimum over the snapped edge
ends) and also returns the isochrone triangles for drawing. It assumes both directions are in the CSR.
Supplied <code>demandTimes</code> and <code>demandBands</code> must each have exactly one logical row per demand point,
regardless of backing-buffer capacity or byte offset. Caller output/scratch buffers must be distinct
from original routing inputs and independently retained results, including separate imports of the
same physical buffer. Rejection leaves the graph unchanged for a corrected same-ID retry.
<code>isochrones</code> takes <code>raster</code> and/or <code>cellOutline</code> (at least one). <code>cellOutline.rings</code> returns closed
isochrone rings in <code>result.isochroneRings</code>. <code>isochrones.joinDemand</code> (<code>candidateCapacity</code> and the
optional <code>pointFeatureIds</code>, <code>overflow</code>, <code>uncertainCount</code>) joins the demand points to the ring polygon (it needs
<code>cellOutline.rings.output.polygons</code>) with <code>GPUPointInPolygonJoin</code>, and <code>result.isochroneDemand.pointFeatureIds</code>
is 0 inside and <code>GPU_SPATIAL_JOIN_NO_FEATURE</code> outside. That answers "inside the isochrone at
<code>cellCostLimit</code>" at cell resolution (the union of cells that hold reached nodes), where the band
classification stays exact at the snapped edge. Isoband triangles cannot feed the join, because their
topology is GPU-written and has no ring structure; the cell rings can. Sparse networks drop cells, so the
polygon join is coarser than the network-space bands. With <code>candidateCapacity</code>, each snapping reports
BVH candidate overflow in <code>result.snapping</code> and the <code>snap-facilities</code> and <code>snap-demand</code> status stages,
so a facility or demand point snapped past a dropped candidate is never silent. The other stages are
<code>service-areas</code>, the caller-owned isochrone outputs (<code>isochrone-bands</code>, <code>isochrone-band-edges</code>,
<code>isochrone-band-rings</code> on the raster path; <code>isochrone-cells</code>, <code>isochrone-outline</code>, <code>isochrone-rings</code> on
the cell path) and <code>band-statistics</code> (count, overflow, required count). The service-area <code>assignments</code>
are passed to <code>GPUNetworkIsochrones</code>, so <code>raster.output.pixelFacilities</code>, <code>triangleFacilities</code> and
<code>cellOutline.byFacility</code> work through the recipe; facility rows are seed rows (<code>2 * facility + endpoint</code>).</p>
</li>
<li>
<p>Every recipe reports each inner contributor's status as a <code>status.stages</code> entry (rule 10): the
cluster recipe ends with <code>hull-measures</code> (<code>invalidCount</code>), the straight-line recipe has
<code>zonal-statistics</code> (<code>overflow</code>), and the table, join and rasterization stages also carry <code>requiredCount</code>
or <code>uncertainCount</code> where the inner contributor computes them. Result-level status words sit in
<code>result.status</code> next to <code>stages</code> (for example <code>status.weightsOverflow</code>, <code>status.invalidCount</code>,
<code>status.rasterOverflow</code>), sub-results own a <code>status</code> (<code>result.permutation.status.overflow</code>,
<code>result.table.status.count</code>), and the <code>outputs</code> overrides mirror those paths (<code>outputs.status.weightsOverflow</code>).</p>
</li>
<li>
<p>The drive-time, straight-line catchment and fleet dwell recipes take an optional <code>spatialContext</code> and
return the resolved one as <code>result.spatialContext</code>. The straight-line recipe and both fleet dwell recipes
are planar only (longitude/latitude throws; the join and zone edges are straight lines in the plane, and the
distance field is Euclidean). The drive-time recipe also accepts longitude/latitude (metric <code>'none'</code> or
<code>'great-circle'</code>; the cell-outline isochrones need degrees). Both of its snappings then run great-circle:
edges are ranked by great-circle distance and <code>maximumSnapDistance</code> is in meters, with no east-west bias at
any latitude. Drive times stay in the units of <code>network.weights</code> (the snapped fraction splits the edge
cost). For the other recipes, project with <code>createGPULocalFrame</code> and pass its <code>spatialContext</code> (planar,
meters), so the distance-field <code>parameters</code> and the straight-line <code>distances</code> output are in meters.
The fleet dwell recipes used to hand the trajectory stage a native planar context whatever the data;
they now resolve one context and pass it to the metrics, join and zone stages.</p>
</li>
<li>
<p>The rate cluster map recipe runs <code>GPULocalMoran</code> with <code>quadrantGating: 'none'</code> when <code>permutation</code> is
given and keeps the quadrant where the permutation <code>significant</code> mask is set; without <code>permutation</code> the
quadrant is gated by the analytic p-value. <code>analyticQuadrants</code> returns the local Moran quadrants (gated
analytically without <code>permutation</code>, ungated with it).</p>
</li>
<li>
<p>The points-in-polygons choropleth recipe's <code>'group'</code> back-end uses dense <code>GPUGroupStatistics</code>
(<code>keyCount: featureCount</code>, so empty polygons keep a row), writing <code>featureValues</code> directly. For
<code>statistic: 'count'</code>, <code>featureValues</code> is the <code>uint32</code> counts view, which <code>GPUClassBreaks</code> and
<code>GPUColorScale</code> read as numbers (<code>integerValues: 'numeric'</code>); the <code>featureValues</code> prop is rejected for
<code>'count'</code>, and the result type is <code>float32</code> or <code>uint32</code>.</p>
</li>
<li>
<p>The spatial regression recipe requires a finite residual on every row; the weights pattern may be asymmetric.
Optional local fits pass their <code>spatialContext</code> to <code>GPUGeographicallyWeightedRegression</code>; omitted means
native planar, and fixed bandwidths use the context's length unit.
Its <code>settings.significanceLevel</code> (default 0.05) is packed once into the autocorrelation <code>parameters</code>
buffer and read per frame by both the specification search and the residual <code>GPULocalMoran</code> (it used to
be a compile-time constant of the search kernel); <code>result.inference</code> and <code>result.setSettings</code> follow it.</p>
</li>
<li>
<p>The fleet dwell recipes take <code>trajectories</code> (<code>GPUPackedGeometryPort&#x3C;GPUTrajectoryGeometryPort></code>) and polygon <code>zones</code>
(<code>GPUPolygonGeometryPort</code>, <code>sourceIds</code> as stable zone IDs); the zone events variant passes the zones to
<code>GPUZoneEvents</code> directly (no edge lists) with <code>trackEventCapacity</code> events kept per track. The
straight-line catchments recipe takes <code>grid</code>, <code>parameters</code> and <code>facilities</code> (<code>GPUPointGeometryPort</code>,
<code>sourceIds</code> as zone IDs); the cluster recipe takes <code>points</code> (<code>GPUPointGeometryPort</code>).</p>
</li>
<li>
<p>For <code>addFleetDwellZoneEventsRecipe</code>, inspect both <code>events.status.overflow</code> (final capacity) and
<code>events.status.candidateOverflow</code> (scratch or per-track loss) before treating the crossing list as
complete. The separate recipe <code>status.candidateOverflow</code> identifies scratch loss affecting zone
dwell totals and visits; per-track event-list truncation alone leaves those totals intact.</p>
</li>
<li>
<p>The cluster recipe's optional <code>spatialContext</code> defaults to native planar and is returned as
<code>result.spatialContext</code>. It is planar only because group summaries and convex hulls are planar:
longitude/latitude throws, so project first. Clustering measures distance and rejects
<code>local-lambert-equal-area</code> frames; <code>epsilon</code> uses the context's length unit.</p>
</li>
<li>
<p>The cluster recipe drops clusters at or above <code>maximumClusterCount</code>, and the caller supplies the
geometry parameters. The change-of-support recipe's <code>status.overflow</code> covers the overlap list only; the raster
flags are returned as <code>status.rasterOverflow</code> and <code>status.rasterRequiredCount</code> (stages <code>source-rasterization</code> and
<code>target-rasterization</code>).</p>
</li>
</ul>
<h3 id="drive-time-contributor-inventory">Drive-time contributor inventory</h3>
<p>The basic drive-time recipe returns 12 contributors. With raster isochrones and no polygon-demand
join (the drive-time GPU fixture), it returns the following 15 in execution order: five analysis
contributors and ten adapters. Preparation instances are never included. A single declaration with
preparation bypassed returns the same list. Previously the result omitted the ten adapters and
reported only five; the larger list completes the documented stage inventory.</p>
<p>Prefix each suffix below with <code>\${id}-</code>; the default <code>id</code> is <code>drive-time-catchment</code>. Native classes
have an own <code>id</code>. Adapter producers have class <code>Object</code> and no own <code>id</code>; their stable emitted-stage
ID is shown instead. The oriented-fraction adapter also contains the facility-road sort nodes.</p>
<table>
<thead>
<tr>
<th>Order</th>
<th>Contributor ID / emitted-stage suffix</th>
<th>Class</th>
</tr>
</thead>
<tbody>
<tr>
<td>1</td>
<td><code>edge-sources-kernel-clear</code></td>
<td><code>Object</code> adapter</td>
</tr>
<tr>
<td>2</td>
<td><code>edge-sources-kernel</code></td>
<td><code>Object</code> adapter</td>
</tr>
<tr>
<td>3</td>
<td><code>snap-facilities</code></td>
<td><code>GPUNetworkSnapping</code></td>
</tr>
<tr>
<td>4</td>
<td><code>facility-roads-kernel</code></td>
<td><code>Object</code> adapter</td>
</tr>
<tr>
<td>5</td>
<td><code>facility-starts-kernel</code></td>
<td><code>Object</code> adapter</td>
</tr>
<tr>
<td>6</td>
<td><code>facility-oriented-fractions-kernel</code></td>
<td><code>Object</code> adapter</td>
</tr>
<tr>
<td>7</td>
<td><code>service-areas</code></td>
<td><code>GPUNetworkServiceAreas</code></td>
</tr>
<tr>
<td>8</td>
<td><code>source-path-geometry</code></td>
<td><code>Object</code> adapter</td>
</tr>
<tr>
<td>9</td>
<td><code>source-path-topology</code></td>
<td><code>Object</code> adapter</td>
</tr>
<tr>
<td>10</td>
<td><code>isochrones</code></td>
<td><code>GPUNetworkIsochrones</code></td>
</tr>
<tr>
<td>11</td>
<td><code>snap-demand</code></td>
<td><code>GPUNetworkSnapping</code></td>
</tr>
<tr>
<td>12</td>
<td><code>demand-roads-kernel</code></td>
<td><code>Object</code> adapter</td>
</tr>
<tr>
<td>13</td>
<td><code>demand-times-kernel</code></td>
<td><code>Object</code> adapter</td>
</tr>
<tr>
<td>14</td>
<td><code>demand-bands-kernel</code></td>
<td><code>Object</code> adapter</td>
</tr>
<tr>
<td>15</td>
<td><code>band-statistics</code></td>
<td><code>GPUGroupStatistics</code></td>
</tr>
</tbody>
</table>
<p>The basic form omits rows 8–10. The result audit checks all public recipe names, including both
visibility entry points: contributor instances/IDs are unique, returned ports belong to the caller
graph, and contributor counts, node IDs and port layouts match a single-declaration baseline.
Intentional aliases between named output/intermediate ports remain valid. Legacy family results
without a <code>contributors</code> property are identified separately rather than counted as empty analyses.</p>
<h3 id="addhotspotanalysisrecipe"><code>addHotSpotAnalysisRecipe</code></h3>
<p>Bins observations onto Quadbin cells inside a study area and returns Getis-Ord Gi* z-scores and confidence bins per cell.</p>
<p>::lens{scenario=gpu-spatial-analysis.hot-spot-analysis-recipe rung=result caption="2,344 observations over 13 community areas; each analysed cell is coloured by its Gi* z-score."}</p>
<p>Points on a study area, or a raster, to Getis-Ord Gi* hot and cold spots.</p>
<p>Points source (breaking change from the occupied-cells design):</p>
<ul>
<li>Recipe <code>spatialContext</code> describes study-area polygons (default longitude/latitude, metric <code>'none'</code>,
linear edges). For antimeridian crossings or great-circle edges, set <code>prepare: true</code> or prepare with
<code>GPULonLatPolygonPreparation</code>. Preparation requires <code>preparedPositionCapacity</code> and a per-frame
<code>preparationParameters</code> view from <code>getGPULonLatPolygonPreparationParameterValues(edgeModel, settings)</code>.
It runs before cell cover and preserves feature rows; <code>status.stages</code> adds <code>study-area-preparation</code>
with <code>overflow</code>, <code>requiredCount</code>, <code>invalidCount</code> and <code>approximation</code>. Preparation requires a points
source and a longitude/latitude context. The cover's <code>invalidCount</code> counts features with unsplit
non-pole antimeridian-crossing edges, not wide bounding boxes.</li>
<li><code>studyArea</code> (a <code>GPUPolygonGeometryPort</code> plus <code>candidateCapacity</code>, optional <code>coverCapacity</code>) and
<code>cellCapacity</code> are required. The recipe covers the polygons with <code>GPUCellCover</code> (<code>'center'</code> containment),
deduplicates and sorts the cells with <code>GPUCellAggregation</code>, aggregates the points the same way and joins them
by binary search of each 64-bit key. Every study cell is a row, with value 0 when it holds no point, so the
null mean is not inflated and clusters of empty cells can be cold spots. Points outside the study area are
excluded; <code>result.status.pointsOutsideStudyArea</code> counts them. <code>result.table.cells</code> are the sorted study cells and
<code>table.counts</code> / <code>table.sumValues</code> the joined values.</li>
<li>Neighbors are topological: the cells within <code>neighborRings</code> grid steps (<code>GPUCellTopology</code> disk) that are also
study cells, self excluded, binary weights, rows ascending. The defaults are 1 ring for Quadbin (8 neighbors)
and 2 for H3 (18 neighbors), the smallest disks that reach <code>minimumNeighbors: 8</code>. This replaces the degree
radius and the <code>neighborSearchParameters</code>, <code>gridSize</code> and <code>tableCapacity</code> props. <code>neighborCapacity</code> is
optional (defaults to the largest disk); truncation sets the <code>weights</code> status <code>overflow</code>.</li>
<li><code>distanceBand</code> (optional, exclusive with <code>neighborRings</code>) uses a metric band instead: <code>GPUNeighborSearch</code> in radius
mode over the cell centers with <code>spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'great-circle'}</code>,
so the radius in <code>neighborSearchParameters</code> is in meters (the context's <code>units</code>). Any other context throws: a
radius in degrees is an ellipse that narrows east-west with latitude. <code>minimumNeighbors</code> still applies.
<code>distanceBand.rowStandardize</code> (default <code>false</code>, classic binary Gi*) row-standardizes the band through the
search's compile-time <code>rowStandardize</code>, overriding the per-frame field of <code>neighborSearchParameters</code>. The
described weights (<code>information.transform</code> <code>'row'</code> or <code>'binary'</code>, and the search status) feed <code>GPUHotSpotAnalysis</code>.
Every source publishes the unclamped neighbor total as the <code>weights</code> stage <code>requiredCount</code> (the grid-disk
adapter included), for <code>getGPUSpatialWeightsCapacityPlan</code>.</li>
<li><code>analyze: 'density'</code> (default) analyses value (or count) per square kilometer. Web Mercator tiles shrink
with the cosine of latitude, so raw Quadbin counts compare cells of different area; <code>'count'</code> is allowed for
H3 only (cells are approximately equal area) and throws for Quadbin. Quadbin area is the exact spherical
rectangle <code>R^2 dLon (sin(lat1) - sin(lat2))</code> and H3 area comes from <code>GPUCellMeasures</code>, both on the authalic
sphere (<code>GPU_EARTH_AUTHALIC_RADIUS_METERS</code>, which is H3's own radius), so the families share one model.
Quadbin areas are within f32 rounding (about 1e-6 relative); H3 pentagons and edge cells above resolution 5 are
unreliable (see <code>GPUCellMeasures</code>).</li>
</ul>
<p>Statistics and legend (both sources):</p>
<ul>
<li><code>minimumNeighbors</code> defaults to 8 (ArcGIS guidance for analytic Gi*). Rows with fewer neighbors other than themselves
get NaN <code>zScores</code> and <code>pValues</code>, bin 0, and are outside the multiple-testing family;
<code>status</code> stage <code>minimum-neighbors</code> reports <code>invalidCount</code>. Quadbin with one ring has exactly 8 neighbors at
most, so the cells on the study-area border are NaN; use <code>neighborRings: 2</code> or a lower minimum for them.</li>
<li><code>multipleTesting</code> defaults to <code>'benjamini-hochberg'</code> (it replaces <code>falseDiscoveryRate</code>). Legend contract,
stable under any setting: bins <code>+-1</code>, <code>+-2</code>, <code>+-3</code> are 90%, 95% and 99% confidence hot (<code>+</code>) or cold (<code>-</code>)
spots; without correction that is <code>|z| >= 1.645, 1.960, 2.576</code>, with Benjamini-Hochberg the step-up test at
q = 0.10, 0.05, 0.01. <code>result.inference</code> is a <code>GPUInferenceMetadata</code> (alternative <code>'two-sided'</code>, analytic so
<code>permutationCount: 0</code>, the method, and <code>testedRowsDefinition</code>, for example
<code>'included rows with at least 8 active neighbors (islands excluded) and a defined z-score'</code>);
<code>result.permutationInference</code> describes the optional permutation confirmation. Both come from
the contributors' <code>getInferenceMetadata</code> of the one <code>settings</code> object (<code>GPUHotSpotAnalysisRecipeSettings</code>:
<code>significanceLevel</code>, optional fixed moments, <code>seed</code>, <code>permutationCount</code>) that the recipe packs into its
<code>parameters</code> and <code>permutation.parameters</code> <code>GPUParameterBuffer</code>s; <code>result.setSettings</code> repacks them per frame.</li>
<li>The lattice source keeps its queen/rook weights and gets the same guard and metadata.
Values 0 and 1 of <code>minimumNeighbors</code> both use an effective minimum of one: masked rows, islands
and undefined scores stay outside the analytic family. The contributor's full eligibility
description is preserved when settings change.</li>
<li>Permutation confirmation does not apply <code>minimumNeighbors</code>. Its family contains included finite-value
rows with 1 to <code>permutation.maximumNeighbors</code> positive-weight neighbors (default maximum 32), as stated
by <code>result.permutationInference.testedRowsDefinition</code>. The analytic family has its own description in
<code>result.inference.testedRowsDefinition</code>; on a 5×5 queen lattice at the default minimum these families
contain 25 and 9 rows respectively. Multiple-testing correction uses each stage's actual family.</li>
<li>The permutation confirms the statistic the Gi stage maps. <code>GPULocalPermutationTest</code> has a fixed focal
weight (1 for <code>'localGStar'</code>, none for <code>'localG'</code>), so <code>permutation.statistic</code> defaults to <code>'localGStar'</code>,
or to <code>'localG'</code> when <code>selfWeight</code> is 0. <code>'localGStar'</code> with a <code>selfWeight</code> other than 1, or <code>'localG'</code> with
a <code>selfWeight</code> other than 0, throws at construction. <code>outputs.permutation.observed</code> (optional, <code>float32</code>
per row) receives the permutation test's observed statistic in esda scaling, so the two stages can be
compared.</li>
</ul>
<h4 id="how-it-computes">How it computes</h4>
<p>{/* lens:addHotSpotAnalysisRecipe */}</p>
<p>::lens{scenario=gpu-spatial-analysis.hot-spot-analysis-recipe rung=kernel caption="Read the kernels of the composed contributors, in dispatch order."}</p>
<p>::lens{scenario=gpu-spatial-analysis.hot-spot-analysis-recipe rung=graph caption="Follow the contributor-grain command graph from inputs to outputs."}</p>
<h3 id="addrateclustermaprecipe"><code>addRateClusterMapRecipe</code></h3>
<p>Turns bird and total observation counts per tract into smoothed rates and significant local Moran clusters.</p>
<p>::lens{scenario=gpu-spatial-analysis.rate-cluster-map-recipe rung=result caption="Bird observations over all observations in 199 tracts; quadrants mark significant clusters."}</p>
<p>Event counts and populations per polygon to significant local Moran clusters of empirical-Bayes rates (esda
<code>Moran_Local_Rate</code>).</p>
<ul>
<li><code>analyze</code> is <code>'standardized'</code> (default, Assuncao-Reis) or <code>'smoothed'</code> (global empirical-Bayes rate).
<code>clampPriorVariance</code> (default <code>true</code>) is passed to <code>GPUEmpiricalBayesRates</code> and only changes the <code>'smoothed'</code>
path: the prior variance is clamped at zero (Marshall), so counts no more dispersed than Poisson shrink fully
to the pooled rate. <code>false</code> reproduces esda <code>Empirical_Bayes</code>. The standardized path is unaffected.</li>
<li><code>multipleTesting</code> defaults to <code>'benjamini-hochberg'</code> (it replaces <code>permutation.falseDiscoveryRate</code>) and applies
to whichever inference runs: the analytic <code>GPULocalMoran</code> p-values without <code>permutation</code>, or the permutation
pseudo p-values with it. The family is every polygon with a valid rate and at least one neighbor, never a viewport.</li>
<li>Analytic inference is a preview (two-sided normal approximation of the conditional-randomization moments).
Permutation inference is the reference; esda uses 999 permutations and so should you. The alternative defaults to
<code>'two-sided'</code>. Analytic p-values are exposed as <code>result.analytic</code>, permutation ones as <code>result.permutation</code>
(<code>pseudoPValues</code>, and <code>adjustedPValues</code> after the correction).</li>
<li>Islands and polygons with a NaN rate (zero population, masked) end at quadrant 0 and have NaN p-values: they are
outside the test family, not counted as tested but not significant.</li>
<li>Legend contract, stable under any setting: palette index 0 is "not significant" and 1, 2, 3, 4 are high-high,
low-high, low-low, high-low. The correction changes which polygons reach significance, never what a color means.</li>
<li><code>palette</code> must contain at least five packed <code>uint32</code> entries. Supplied <code>outputs.colors</code> and
<code>outputs.quadrants</code> must each have exactly one logical row per polygon feature, even when their
backing buffers have spare capacity. Independent outputs must use separate buffers and must not
alias any original input, including the palette and parameter buffers. Active scratch is checked
with the retained outputs too: for example, <code>scratch.spatialLag</code> cannot overwrite <code>smoothedRates</code>.
Invalid ports are rejected
before scheduling, so corrected views can reuse the same recipe ID.</li>
<li><code>events</code> and <code>populations</code> accept <code>float32</code> or <code>uint32</code> (for example the <code>counts</code> of
<code>addPointsInPolygonsChoroplethRecipe</code>); <code>uint32</code> columns are converted by one node (exact below 2^24).</li>
<li>Rows are features: <code>polygons</code> is a <code>GPUPolygonGeometryPort</code> (four-level layout, one packed <code>positions</code>
view), so multipart features get one row and the <code>counts</code> of <code>addPointsInPolygonsChoroplethRecipe</code> on the same
features chain in directly. The <code>weights</code> stage publishes <code>requiredCount</code> for <code>getGPUSpatialWeightsCapacityPlan</code>.</li>
<li>Settings have one source of truth. Pass <code>GPUParameterBuffer</code>s as <code>parameters</code> and <code>permutation.parameters</code>
plus <code>settings: {significanceLevel?, seed?, permutationCount?}</code>: the recipe packs the settings into both buffers
and builds <code>result.inference</code> from the same object, and <code>result.setSettings(settings)</code> repacks them per frame
(an invalid setting throws and writes nothing). Caller-packed views are not accepted.</li>
<li><code>result.inference</code> is the <code>getInferenceMetadata</code> of the <code>GPULocalPermutationTest</code> the recipe built (or of
<code>GPULocalMoran</code> without <code>permutation</code>). It holds the alternative, <code>permutationCount</code> (0 for analytic), <code>seed</code>,
<code>significanceLevel</code> (default 0.05), <code>multipleTesting</code> and <code>testedRowsDefinition</code>. An invalid permutation
count or seed throws when the recipe is built.</li>
</ul>
<h4 id="how-it-computes-1">How it computes</h4>
<p>{/* lens:addRateClusterMapRecipe */}</p>
<p>::lens{scenario=gpu-spatial-analysis.rate-cluster-map-recipe rung=kernel caption="Read the kernels of the composed contributors, in dispatch order."}</p>
<p>::lens{scenario=gpu-spatial-analysis.rate-cluster-map-recipe rung=graph caption="Follow the contributor-grain command graph from inputs to outputs."}</p>
<h3 id="addperiodcomparisonrecipe"><code>addPeriodComparisonRecipe</code></h3>
<p>Counts two periods of observations per Quadbin cell and returns the classified, coloured change.</p>
<p>::lens{scenario=gpu-spatial-analysis.period-comparison-recipe rung=result caption="655 spring and 566 autumn observations compared per cell; classes diverge from no change."}</p>
<p>Two periods of events to a classified, colored change map.</p>
<p>For <code>measure: 'sum'</code>, pass one <code>sumScale</code> (default 65536) for both aggregation stages and the
comparison. For example, <code>sumScale: 2 ** 18</code> preserves a one-cell change from 0 to <code>2 ** -18</code>;
the default scale rounds that value to zero. Choose a scale with <code>getGPUCellSumScale</code> using the
largest magnitude and group row count across both periods.</p>
<p>The comparison stage separates destination <code>overflow</code> from <code>candidateOverflow</code> inherited from
either aggregation table. Its <code>requiredCount</code> is the retained-input union requirement and becomes
a lower bound after source loss. Inspect <code>status.stages</code> to find the capacity that needs growing;
adding union slots cannot recover an earlier aggregation's dropped cells.</p>
<ul>
<li><code>delta</code> (after minus before) is always defined. <code>ratio</code> and <code>percentChange</code> are NaN, not 0, for cells whose earlier
value is 0; NaN rows are skipped by <code>GPUClassBreaks</code> and take the palette no-data color.</li>
<li><code>zScore</code> is a descriptive score of the change (<code>'poisson'</code>: <code>delta / sqrt(before + after)</code>, or <code>'standardized'</code>
deltas). It has no null distribution and no multiple-testing correction, so a threshold on it is not a significance
level; use <code>addHotSpotAnalysisRecipe</code> or <code>addRateClusterMapRecipe</code> for inference. With <code>measure: 'count'</code> every
union cell has <code>before + after > 0</code>, so the Poisson denominator is never zero.</li>
</ul>
<h4 id="how-it-computes-2">How it computes</h4>
<p>{/* lens:addPeriodComparisonRecipe */}</p>
<p>::lens{scenario=gpu-spatial-analysis.period-comparison-recipe rung=kernel caption="Read the kernels of the composed contributors, in dispatch order."}</p>
<p>::lens{scenario=gpu-spatial-analysis.period-comparison-recipe rung=graph caption="Follow the contributor-grain command graph from inputs to outputs."}</p>
<h3 id="addchangeofsupportrecipe"><code>addChangeOfSupportRecipe</code></h3>
<p>Moves tract observation counts onto community areas by raster overlap area and reports what each tract could not transfer.</p>
<p>::lens{scenario=gpu-spatial-analysis.change-of-support-recipe rung=result caption="199 census tracts rasterized with 13 community areas; counts are redistributed by overlap area."}</p>
<p>Transfers values between two polygon zone systems by area: both systems are rasterized to one grid
(<code>GPUPolygonRasterization</code>), <code>GPUArealInterpolation</code> builds the overlap areas and both weight sets, and
<code>GPUSpatialLag</code> moves <code>sourceValues</code> to the targets as <code>extensiveValues</code> and <code>intensiveValues</code>.</p>
<ul>
<li>Props: <code>grid</code> (<code>GPURasterMetadata</code>), <code>parameters</code> (<code>getGPUPolygonRasterizationParameterValues(grid)</code>),
<code>source</code> and <code>target</code> zone systems (<code>GPUPolygonGeometryPort</code> plus <code>crossingCapacity</code>),
<code>pairCapacity</code>, optional <code>cellWeights</code>, <code>sourceValues</code> and <code>columnCount</code>, <code>categories</code>, <code>denominator</code>,
<code>sourceAnchorCells</code>, <code>spatialContext</code> and <code>outputs</code> / <code>scratch</code> overrides.</li>
<li>Denominators (default change): extensive <code>'zone'</code>, intensive <code>'overlap'</code>, so a target only partly
covered by sources is the mean of what covers it. Pass <code>denominator: 'zone'</code> for the old intensive result.</li>
<li>Conservation: the result always carries <code>sourceAreas</code>, <code>sourceTransferredFractions</code> and <code>status.invalidCount</code>;
the last one is also in the <code>areal-interpolation</code> stage <code>status</code>. A source entirely outside the targets,
or smaller than a cell, shows up there instead of silently losing mass; <code>sourceAnchorCells</code> rescues the
latter.</li>
<li>Coordinates: <code>spatialContext</code> omitted is native planar (cell counts are areas). <code>longitude-latitude</code>
requires metric <code>'great-circle'</code> and declares a longitude/latitude raster whose <code>parameters</code> are in degrees.
The areal stage weights each cell by its area on the authalic sphere in the context's length unit squared
(square meters by default; square kilometers with <code>units: 'kilometers'</code>). It measures area, so
<code>local-azimuthal-equidistant</code> frames throw. Rasterization itself only
tests cell centers with straight polygon edges in degrees. For antimeridian crossings or great-circle
edges, set <code>prepare: true</code> or prepare with <code>GPULonLatPolygonPreparation</code>.
The resolved context is returned as <code>result.spatialContext</code>.</li>
<li><code>prepare</code> defaults to false and requires a longitude/latitude context, <code>preparedPositionCapacity</code>
(applied separately to each zone system), and a per-frame <code>preparationParameters</code> view from
<code>getGPULonLatPolygonPreparationParameterValues(edgeModel, settings)</code>. Preparation runs before each
rasterization and preserves zone IDs. Stages <code>source-preparation</code> and <code>target-preparation</code> expose
<code>overflow</code>, <code>requiredCount</code>, <code>invalidCount</code> and <code>approximation</code>.</li>
<li>Accuracy is bounded by the raster resolution (cell centers decide membership), never an exact polygon
intersection.</li>
</ul>
<h4 id="how-it-computes-3">How it computes</h4>
<p>{/* lens:addChangeOfSupportRecipe */}</p>
<p>::lens{scenario=gpu-spatial-analysis.change-of-support-recipe rung=kernel caption="Read the kernels of the composed contributors, in dispatch order."}</p>
<p>::lens{scenario=gpu-spatial-analysis.change-of-support-recipe rung=graph caption="Follow the contributor-grain command graph from inputs to outputs."}</p>
<h3 id="addpointsinpolygonschoroplethrecipe"><code>addPointsInPolygonsChoroplethRecipe</code></h3>
<p>Counts points per polygon and classifies the counts into a coloured choropleth.</p>
<p>::lens{scenario=gpu-spatial-analysis.points-in-polygons-choropleth-recipe rung=result caption="2,344 observations counted in 199 tracts and shaded in five quantile classes."}</p>
<p>Points to a per-polygon statistic to class breaks to colors, with <code>backend: 'zonal'</code> (default,
<code>GPUZonalStatistics</code>) or <code>'group'</code> (<code>GPUPointInPolygonJoin</code> then dense <code>GPUGroupStatistics</code>). <code>statistic</code>
is <code>'count'</code> (default), <code>'sum'</code>, <code>'mean'</code>, <code>'minimum'</code> or <code>'maximum'</code>; features with no points keep NaN (mean,
minimum, maximum) or 0 (count, sum).
Both backends require individual point rows. A point port with <code>featureOffsets</code> groups multipart
features and is rejected before declaration, rather than counting its parts as independent features.
Expand part rows with explicit ownership or aggregate through a feature-level join first.</p>
<ul>
<li>Coordinates: <code>spatialContext</code> omitted is native planar. <code>longitude-latitude</code> is accepted because the
mapped statistics are topological: containment uses straight polygon edges in degrees (not geodesics) and
for antimeridian crossings or great-circle edges, set <code>prepare: true</code> or prepare with
<code>GPULonLatPolygonPreparation</code>. No density is mapped, so no polygon area is needed; for
a density per square meter use <code>GPUZonalStatistics</code> directly with spherical <code>areas</code>. Other spaces throw.
The resolved context is returned as <code>result.spatialContext</code> (linear edges after preparation).</li>
<li><code>prepare</code> defaults to false and requires a longitude/latitude context, <code>preparedPositionCapacity</code>,
and a per-frame <code>preparationParameters</code> view from
<code>getGPULonLatPolygonPreparationParameterValues(edgeModel, settings)</code>. Both backends prepare before
containment; feature rows and output counts retain their original IDs. Stage <code>polygon-preparation</code>
exposes <code>overflow</code>, <code>requiredCount</code>, <code>invalidCount</code> and <code>approximation</code>.</li>
</ul>
<h4 id="how-it-computes-4">How it computes</h4>
<p>{/* lens:addPointsInPolygonsChoroplethRecipe */}</p>
<p>::lens{scenario=gpu-spatial-analysis.points-in-polygons-choropleth-recipe rung=kernel caption="Read the kernels of the composed contributors, in dispatch order."}</p>
<p>::lens{scenario=gpu-spatial-analysis.points-in-polygons-choropleth-recipe rung=graph caption="Follow the contributor-grain command graph from inputs to outputs."}</p>
<h3 id="addclusterandoutlinerecipe"><code>addClusterAndOutlineRecipe</code></h3>
<p>Clusters points with DBSCAN and outlines each cluster with its convex hull, area and perimeter.</p>
<p>::lens{scenario=gpu-spatial-analysis.cluster-and-outline-recipe rung=result caption="2,344 observations clustered at 250 m with six points minimum; each cluster is outlined."}</p>
<p>The chain of this recipe is listed in the chain table above.</p>
<h4 id="how-it-computes-5">How it computes</h4>
<p>{/* lens:addClusterAndOutlineRecipe */}</p>
<p>::lens{scenario=gpu-spatial-analysis.cluster-and-outline-recipe rung=kernel caption="Read the kernels of the composed contributors, in dispatch order."}</p>
<p>::lens{scenario=gpu-spatial-analysis.cluster-and-outline-recipe rung=graph caption="Follow the contributor-grain command graph from inputs to outputs."}</p>
<h3 id="addfleetdwellrecipe"><code>addFleetDwellRecipe</code></h3>
<p>Finds where vessels stop and totals the dwell time per anchorage zone.</p>
<p>The stops variant forwards <code>trajectories.temporalContext</code>, <code>timestamps</code> and optional <code>timestampsLow</code>
to <code>GPUTrajectoryMetrics</code>. Float32-relative, Int64 words and double-single times are supported; the
declared encoding must match the bound views. Durations use the declared tick unit. The zone-events
variant takes <code>GPUFleetDwellZoneEventsTrajectories</code>: float32-relative or Int64 words only. It rejects
<code>timestampsLow</code> and incompatible declarations before scheduling nodes, and forwards supported contexts
to <code>GPUZoneEvents</code>.
Both variants reject incompatible time declarations before registering their recipe outputs, so
the caller can correct the declaration and retry on the same graph with the same recipe ID.</p>
<p>Only the GPU-written <code>stops.status.count</code> prefix enters the zone join. A private adapter view gives
unused slots nonfinite coordinates, which the join excludes before counting candidates; the public
centroids keep the metrics contributor's zero-filled padding. An empty stop list therefore cannot
consume <code>zones.candidateCapacity</code> or set <code>status.joinOverflow</code> because of spare stop slots.
<code>joinOverflow</code> still reports capacity loss among real stop candidates, and <code>joinUncertainCount</code>
reports unresolved containment among them. Check those words and the metrics stage's overflow before
using a per-zone summary as complete.
When caller-owned centroids and a stop mask occupy disjoint subviews whose aligned storage bindings
overlap, the recipe produces the mask and private join positions in separate dispatches. For example,
two centroids at bytes 0–15 and two mask values at bytes 16–23 of one buffer remain supported.</p>
<p>::lens{scenario=gpu-spatial-analysis.fleet-dwell-recipe rung=result caption="12 vessel tracks over 13 anchorage zones; zones are shaded by total stop time."}</p>
<p>The chain of this recipe is listed in the chain table above.</p>
<h4 id="how-it-computes-6">How it computes</h4>
<p>{/* lens:addFleetDwellRecipe */}</p>
<p>::lens{scenario=gpu-spatial-analysis.fleet-dwell-recipe rung=kernel caption="Read the kernels of the composed contributors, in dispatch order."}</p>
<p>::lens{scenario=gpu-spatial-analysis.fleet-dwell-recipe rung=graph caption="Follow the contributor-grain command graph from inputs to outputs."}</p>
<h3 id="addspatialregressionrecipe"><code>addSpatialRegressionRecipe</code></h3>
<p>Fits observations per tract on income and car-free households and tests the residuals for spatial dependence.</p>
<p>::lens{scenario=gpu-spatial-analysis.spatial-regression-recipe rung=result caption="199 tracts; residuals of the fit show where observations exceed or fall short of the model."}</p>
<p>The chain of this recipe is listed in the chain table above.</p>
<h4 id="how-it-computes-7">How it computes</h4>
<p>{/* lens:addSpatialRegressionRecipe */}</p>
<p>::lens{scenario=gpu-spatial-analysis.spatial-regression-recipe rung=kernel caption="Read the kernels of the composed contributors, in dispatch order."}</p>
<p>::lens{scenario=gpu-spatial-analysis.spatial-regression-recipe rung=graph caption="Follow the contributor-grain command graph from inputs to outputs."}</p>
<h3 id="addstraightlinecatchmentsrecipe"><code>addStraightLineCatchmentsRecipe</code></h3>
<p>Allocates every raster cell to its nearest facility and sums a value raster per catchment.</p>
<p>::lens{scenario=gpu-spatial-analysis.straight-line-catchments-recipe rung=result caption="150 facilities split the North Side into nearest-facility catchments on a 250 m grid."}</p>
<p>The chain of this recipe is listed in the chain table above.</p>
<h4 id="how-it-computes-8">How it computes</h4>
<p>{/* lens:addStraightLineCatchmentsRecipe */}</p>
<p>::lens{scenario=gpu-spatial-analysis.straight-line-catchments-recipe rung=kernel caption="Read the kernels of the composed contributors, in dispatch order."}</p>
<p>::lens{scenario=gpu-spatial-analysis.straight-line-catchments-recipe rung=graph caption="Follow the contributor-grain command graph from inputs to outputs."}</p>
<h2 id="ancillary-weighted-redistribution">Ancillary-weighted redistribution</h2>
<p><a href="/luma-contributors/docs/api/gpu-spatial-analysis/dasymetric-redistribution">Dasymetric redistribution</a> combines fractional source-cell coverage,
an ancillary raster and explicit whole-cell target ownership. Source totals remain traceable through
allocated and unallocated outputs; the recipe does not approximate target polygon intersections by
multiplying marginal coverage fractions.</p>`};export{e as default};