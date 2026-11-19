var e={slug:`api/gpu-spatial-analysis/operators-accessibility`,title:`GPU Spatial Analysis: Accessibility & change of support`,html:`<h1 id="accessibility--change-of-support">Accessibility &#x26; change of support</h1>
<h3 id="gpucatchmentaccessibility"><code>GPUCatchmentAccessibility</code></h3>
<p>Scores each demand location by the supply-to-demand ratios of the facilities whose catchments reach it, with two-step floating catchment areas.</p>
<p>::lens{scenario=gpu-spatial-analysis.catchment-accessibility rung=result caption="14 libraries and 199 tract centroids: 2 km straight-line catchments, not travel time or network distance."}</p>
<p>Floating catchment accessibility on any <code>GPUSpatialWeights</code> (the <code>access</code> package <code>two_step_fca</code>).
<code>method: '2sfca'</code> or <code>'3sfca'</code>, with <code>supply</code>, <code>demand</code>, <code>facilityWeights</code> (row <code>j</code> lists the demand
points <code>i</code> with weight <code>f(cost)</code>), optional <code>demandWeights</code> (row <code>i</code> lists facilities; it defaults to
<code>facilityWeights</code>, which is only valid for symmetric self-join weights and is not checked) and the
caller-owned <code>output: {accessibility, ratios?, reachableFacilities?, validity?}</code> (<code>GPUCatchmentAccessibilityOutput</code>).
A demand row whose <code>demand</code> is NaN or infinite is missing: it adds nothing to any facility's ratio, and
the optional <code>validity</code> output (<code>uint32</code> per demand row) is 0 for it and 1 otherwise, as in
<code>GPUHuffTradeAreas</code>. Its own accessibility does not read its demand and stays valid.</p>
<p>Pass a weights producer's <code>outputWeights</code> port to retain its status and metadata. Supplied nonzero
<code>overflow</code> or <code>candidateOverflow</code> on either weights table suppresses all accessibility scores to
NaN. Facility-weights loss suppresses the ratios too; demand-weights loss suppresses ratios only
for 3SFCA, whose selection totals depend on those rows. Demand-weights loss sets every
<code>reachableFacilities</code> entry to <code>0xffffffff</code> (unknown), while facility-weights loss alone leaves
the complete forward count usable. <code>validity</code> continues to report finite demand only. After the
producer rebuilds complete weights and clears the flags, the next submission restores the ordinary
scores/counts without recompilation. Both trade-area demos forward described search ports and
display unavailable summaries instead of totals computed from suppressed outputs.</p>
<ul>
<li>2SFCA: <code>R_j = S_j / sum_i P_i w_ji</code>, <code>A_i = sum_j w_ij R_j</code>.</li>
<li>3SFCA (Wan, Zou and Sternberg 2012; PySAL <code>access</code> <code>three_stage_fca</code>) applies the selection weight
<code>G_ij = w_ij / T_i</code>, <code>T_i = sum_k w_ik</code> over demand row <code>i</code>, together with the decay <code>w</code>. The ratio step
reads the facility weights, <code>R_j = S_j / sum_i P_i w_ji^2 / T_i</code>, and the score step the demand weights,
<code>A_i = sum_j (w_ij^2 / T_i) R_j</code>. Both are <code>G_ij W_ij</code> only when the demand weights are the transpose of
the facility weights. Demand rows with <code>T_i = 0</code> are skipped. Matches <code>access</code> 1.1.10 on network costs
for binary, exponential and Gaussian decay.</li>
<li>Supply conservation holds for both methods, with any decay function (binary, Gaussian, a network-cost
kernel), when the demand-side weights are the transpose of the facility-side weights: <code>w_ij = w_ji</code> for
every pair, with the same set of pairs. Then <code>sum_i P_i A_i = sum_j S_j</code> over the facilities that capture
demand (<code>sum_i P_i w_ji > 0</code>, or <code>sum_i P_i G_ij > 0</code> for 3SFCA). For 2SFCA this is
<code>sum_i P_i A_i = sum_j R_j sum_i P_i w_ij</code>, and the inner sum is the denominator of <code>R_j</code>, so each
facility contributes exactly <code>S_j</code>. It breaks when the two steps use different weights (different
decays or radii, or a <code>demandWeights</code> that is not the transpose), and by the supply of facilities with no
demand in range, which is not allocated to anyone. Sums are f32, so the identity holds to rounding. The
default <code>demandWeights = facilityWeights</code> satisfies the condition for symmetric self-join weights.
3SFCA differs in what <code>A_i</code> means, a selection-weighted average of the <code>R_j</code>, not in conservation.</li>
</ul>
<p>A non-finite demand, or a non-finite ratio from a non-finite supply, is a missing value and adds nothing.
The operator reads weights, not positions, so it has no <code>spatialContext</code>; distance units and the coordinate
space (planar, or great-circle meters) are those of the weights' producer.
When both <code>facilityWeights</code> and <code>demandWeights</code> carry <code>spatialContext</code>, the contexts must be
compatible; otherwise construction throws.</p>
<p>This is the one 2SFCA/3SFCA kernel. Over a network cost matrix, turn the matrix into weights with
<code>GPUNetworkCostWeights</code>, or use the <code>catchment</code> option of
<a href="/luma-contributors/docs/api/gpu-network"><code>GPUNetworkAccessibility</code></a>, which composes the two.</p>
<h4 id="how-it-computes">How it computes</h4>
<p>{/* lens:GPUCatchmentAccessibility */}</p>
<p>::lens{scenario=gpu-spatial-analysis.catchment-accessibility rung=datum datum=22 caption="Inspect tract 17031020802, the most populous tract with 8,715 residents."}</p>
<p>::lens{scenario=gpu-spatial-analysis.catchment-accessibility rung=formula caption="Facility supply over catchment demand, then summed back to each tract."}</p>
<p>::lens{scenario=gpu-spatial-analysis.catchment-accessibility rung=kernel caption="Read the ratio and score kernels, one invocation per facility or tract."}</p>
<p>::lens{scenario=gpu-spatial-analysis.catchment-accessibility rung=graph caption="Follow the two radius searches, then the ratio and score passes."}</p>
<h3 id="gpuhufftradeareas"><code>GPUHuffTradeAreas</code></h3>
<p>Estimates the probability that each demand location patronises each facility from facility attractiveness and distance-decay weights, and each facility's expected demand.</p>
<p>::lens{scenario=gpu-spatial-analysis.huff-trade-areas rung=result caption="Tracts shaded by modal-park probability: 102 parks weighted by area, inverse-square distance within 2.5 km."}</p>
<p>Huff probabilities <code>p_ij = A_j^alpha w_ij / sum_k A_k^alpha w_ik</code> from <code>attractiveness</code> and
<code>demandWeights</code>, with <code>alpha</code> from a one-element <code>parameters</code> view. The trade area of demand <code>i</code> is its
modal facility (lowest ID on ties, <code>GPU_HUFF_NO_TRADE_AREA</code> when none). Props: <code>attractiveness</code> (<code>float32</code>,
one row per facility), <code>demandWeights</code> (row <code>i</code> lists the facilities of demand <code>i</code>), optional <code>parameters</code>
(<code>float32</code>, <code>GPU_HUFF_TRADE_AREAS_PARAMETER_LENGTH</code> = 1 word written by
<code>getGPUHuffTradeAreasParameterValues({exponent})</code>; without the view <code>alpha = 1</code>), and the caller-owned
<code>output</code> (<code>GPUHuffTradeAreasOutput</code>) with the required <code>tradeArea</code> (<code>uint32</code> per demand row). Optional
outputs are <code>probabilities</code> (one per <code>demandWeights</code> slot, aligned with <code>demandWeights.neighbors</code>),
<code>tradeAreaProbability</code> (per demand row) and <code>expectedDemand</code>, <code>expectedDemand_j = sum_i P_i p_ij</code>, which
needs <code>facilityWeights</code> (the transpose of <code>demandWeights</code>) and <code>demand</code>. A demand row whose <code>demand</code> is
NaN or infinite is missing: as in <code>GPUCatchmentAccessibility</code> it adds nothing to <code>expectedDemand</code>, and
the optional <code>validity</code> output (<code>uint32</code> per demand row, needs <code>demand</code>) is 0 for it and 1 otherwise.
Like <code>GPUCatchmentAccessibility</code> it reads weights only and has no
<code>spatialContext</code>. When both <code>demandWeights</code> and <code>facilityWeights</code> carry <code>spatialContext</code>, the contexts
must be compatible; otherwise construction throws.</p>
<p>Each submission writes zero to probability slots past the terminal CSR offset, including when a
reused weights table shrinks or becomes empty. Zero attractiveness or zero weight contributes no
probability. Normalization uses row-centred log scores before exponentiation; changing the common
scale of positive attractiveness leaves probabilities, the winning facility and expected demand
unchanged within float32 rounding, even when the unnormalized powers overflow or underflow.
Relative scores use the original binary mantissas, retaining differences between adjacent float32
attractiveness values; exact weighted ties at the default exponent keep the lowest facility ID.
Expected-demand contributions combine the demand's binary exponent before rounding: a share that
rounds to zero may still make a representable contribution when demand is large. Signed finite
demand keeps its sign; non-finite demand remains missing.</p>
<p>Supplied nonzero <code>demandWeights.status.overflow</code> or <code>candidateOverflow</code> suppresses every trade area
to <code>GPU_HUFF_NO_TRADE_AREA</code> and every probability (including padding and winning probabilities) to
NaN. Either forward or reverse weights loss suppresses every <code>expectedDemand</code> value to NaN;
reverse weights loss leaves forward probabilities usable. The optional demand <code>validity</code> still
reports finite demand only. Clear the input flags after rebuilding complete weights to restore
ordinary results on the next submission of the same compiled graph.</p>
<h4 id="how-it-computes-1">How it computes</h4>
<p>{/* lens:GPUHuffTradeAreas */}</p>
<p>::lens{scenario=gpu-spatial-analysis.huff-trade-areas rung=datum datum=22 caption="Inspect tract 17031020802: its modal park and that park's probability."}</p>
<p>::lens{scenario=gpu-spatial-analysis.huff-trade-areas rung=formula caption="Park attraction times decay weight, normalised over the tract's reachable parks."}</p>
<p>::lens{scenario=gpu-spatial-analysis.huff-trade-areas rung=kernel caption="Read the trade-area kernel that totals each row and picks the modal park."}</p>
<p>::lens{scenario=gpu-spatial-analysis.huff-trade-areas rung=graph caption="Follow the two radius searches, then the trade-area, probability and demand passes."}</p>
<p>{/* parameters: GPUHuffTradeAreas generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h4 id="gpuhufftradeareas-parameters"><code>GPUHuffTradeAreas</code> parameters</h4>
<ul>
<li>Packer: <code>getGPUHuffTradeAreasParameterValues(settings?)</code>, writes <code>GPU_HUFF_TRADE_AREAS_PARAMETER_LENGTH</code> (1) <code>float32</code> values.</li>
</ul>
<p>Settings of <code>getGPUHuffTradeAreasParameterValues</code>:</p>
<table>
<thead>
<tr>
<th>Field</th>
<th>Type</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>exponent</code> (optional)</td>
<td><code>number</code></td>
<td>Attractiveness exponent <code>alpha</code>, positive and finite. Defaults to 1.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h3 id="gpuarealinterpolation"><code>GPUArealInterpolation</code></h3>
<p>targetZoneCount times categories.categoryCount is at most 4,294,967,294, reserving the final scan
sentinel; device bindings impose lower limits.</p>
<p>Transfers source-zone values to target zones by rasterized overlap area, with extensive or intensive area-share weights.</p>
<p>::lens{scenario=gpu-spatial-analysis.areal-interpolation rung=result caption="199 tracts (719,651 residents) shared to 13 community areas by overlap area on a 100 m raster."}</p>
<p>Maximum: at most <code>getGPUArealInterpolationMaximumCellCount(device)</code> (<code>2^25 - 1</code> on a default device) grid cells; with <code>cellWeights</code>, <code>unweightedFastPath: false</code> or a longitude/latitude raster at most <code>GPU_AREAL_INTERPOLATION_MAXIMUM_WEIGHTED_PAIR_COUNT</code> (<code>2^24 - 1</code>) pair slots.</p>
<p>Area-weighted transfer between two zone systems (tobler <code>area_interpolate</code>, CARTO <code>ENRICH_POLYGONS</code>) on
a common fine raster. Rasterize both systems to the same grid with <code>GPUPolygonRasterization</code> (or any zone
raster) and pass the two zone views, so H3 or hexagon rasters work too.</p>
<ul>
<li>Props: <code>grid</code> (<code>GPURasterMetadata</code>; its <code>width * height</code> is the cell count), <code>sourceZones</code>, <code>targetZones</code>
(<code>uint32</code>, one row per grid cell; a value at or above the count means no zone), <code>sourceZoneCount</code>, <code>targetZoneCount</code>, optional <code>cellWeights</code> (a dasymetric raster; area becomes a
sum of weights, and non-finite or non-positive cells carry no area), <code>denominator</code> (per kind, see
below), <code>mode</code> (<code>'extensive'</code> by default or <code>'intensive'</code>, which selects <code>output.weights.weights</code>),
<code>categories</code> (<code>{sourceCategories, categoryCount}</code>) and the caller-owned <code>output</code>
(<code>GPUArealInterpolationOutput</code>): <code>weights</code> (a CSR with <code>targetZoneCount + 1</code> offsets; the neighbor
capacity is the maximum number of pairs), <code>status.overflow</code>, and the optional <code>status.requiredPairCount</code>,
<code>alternateWeights</code> (the other normalization, from the same run), <code>areas</code> (raw overlap area per slot),
the diagnostics below and <code>categoryShares</code> (<code>T x K</code>, required with <code>categories</code>).</li>
<li>The cell weight <code>m</code> (1, or <code>cellWeights</code>) summed per source <code>s</code> and target <code>t</code> gives <code>a_st</code>. Extensive:
<code>w_ts = a_st / A_s</code>. Intensive: <code>w_ts = a_st / B_t</code>. <code>A_s</code> and <code>B_t</code> are the zone or overlap areas.
Rows are targets and neighbors are sources, so the CSR invariants hold; it is a cross weights matrix.</li>
<li>Denominators (default change). <code>denominator</code> is one value for both kinds or
<code>{extensive?, intensive?}</code>, and the default is <code>{extensive: 'zone', intensive: 'overlap'}</code>. <code>'zone'</code>
divides by the whole zone (for extensive variables, tobler <code>area_interpolate</code> with
<code>allocate_total=False</code>): the part of a source that falls outside the target system is not
moved onto the targets, and is now reported (below). <code>'overlap'</code> divides by the part overlapping the
other system (tobler's own default, <code>allocate_total=True</code>; tobler's intensive denominator is always
the overlap). The extensive default therefore differs from tobler's: a source half outside the
target system with value 100 transfers 50 here and 100 in tobler, unless <code>'overlap'</code> is passed. The intensive default is <code>'overlap'</code> because a target only partly covered by sources
should be the mean of what covers it, not diluted toward zero; earlier versions used <code>'zone'</code> for both.
Pass <code>denominator: 'zone'</code> to restore the old intensive result.</li>
<li>Mass-conservation diagnostics. <code>output.sourceAreas</code> (<code>sourceZoneCount</code> rows) is the raster area of each source
(0 when it rasterized to no cell), <code>sourceTransferredFractions</code> is the overlap area over the extensive
denominator (<code>1 - f</code> of the extensive mass is lost; with <code>'zone'</code>, the share of the source that lies
in the target system; NaN without raster area), and <code>status.invalidCount</code> (one <code>uint32</code>) counts sources with no
raster area, whose mass is lost. They are exact for the rasterized zones, not for the polygons.</li>
<li>Sources smaller than a cell. <code>sourceAnchorCells</code> (<code>sourceZoneCount</code> cell indices; out of range means none)
gives a zone without cells a fallback cell, for example the cell containing its label point. The claim
is made in a transient copy of <code>sourceZones</code>, in ascending source order, and succeeds only if the cell
belongs to no source or to one that keeps another cell. Remaining empty sources stay in <code>status.invalidCount</code>.</li>
<li>Coordinates. Planar by default (areas are cell counts or <code>cellWeights</code> sums, which is an equal-share
only on an equal-area raster). With <code>spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'great-circle'}</code> and <code>parameters</code> (the per-frame <code>getGPUPolygonRasterizationParameterValues(grid)</code>
view, the same view as <code>GPUPolygonRasterization.parameters</code>), each cell weighs its spherical area
<code>R^2 dLon (sin(top) - sin(bottom))</code>, times <code>cellWeights</code>. <code>R</code> comes from <code>getGPUAuthalicSphereRadius</code>:
the WGS84 authalic radius or <code>context.sphereRadius</code>, in the context's length unit. Pair <code>areas</code> and
<code>sourceAreas</code> use that unit squared: square meters by default, square kilometers with
<code>units: 'kilometers'</code> (previously still square meters).
<strong>Breaking:</strong> longitude/latitude now requires <code>metric: 'great-circle'</code>; <code>'none'</code> throws.
Declares <code>measure: 'area'</code>: an azimuthal-equidistant local frame throws; project into a Lambert equal-area frame.
North-up grids (a negative packed cell height) are handled: the area of a stored row follows the mirrored row.
Other spaces throw. Checked against a numeric integral of <code>R^2 cos(lat)</code> (<code>spherical_cell_area.py</code>).</li>
<li>Categorical share <code>(t, k)</code> is the overlap area with sources of class <code>k</code> divided by the area of <code>t</code>
covered by any source, so it sums to 1 where covered.</li>
<li>On overflow the offsets clamp to capacity, <code>status.overflow</code> is 1 and <code>status.requiredPairCount</code> gives the size
needed.</li>
<li><code>unweightedFastPath</code> (default on, used when there are no <code>cellWeights</code>) takes pair areas from run lengths
and zone areas from counts instead of per-cell accumulation. Results are bitwise identical; it was 1.4x
faster on the explorer's graph, about 35 percent on 384 by 384 zone denominators and about equal for the
overlap denominator, where the two radix sorts dominate.</li>
<li>Cell centers decide membership, so every area is a cell count and accuracy is about one cell along every
boundary; slivers smaller than a cell can be missed. It is never an exact polygon intersection. f32 sums
are exact up to 2^24 cells per zone. Results are deterministic.</li>
</ul>
<p>Transfer the values with <code>GPUSpatialLag</code> over the area shares (its source count is the <code>values</code> row
count, <code>sourceZoneCount</code>; <code>columnCount</code> transfers several variables at once). The tobler <code>area_interpolate</code> extensive and intensive results match to 1e-5 on
grid-aligned zones when <code>allocate_total</code> is set to match the denominator (<code>False</code> for <code>'zone'</code>, <code>True</code> for <code>'overlap'</code>).</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUArealInterpolation</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  grid, sourceZones, targetZones, sourceZoneCount, targetZoneCount,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  output: {weights: {offsets, neighbors, weights}, alternateWeights, status: {overflow}}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUSpatialLag</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  weights: {offsets, neighbors, weights}, values: population, output: populationOnTargets</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span></code></pre>
<h4 id="how-it-computes-2">How it computes</h4>
<p>{/* lens:GPUArealInterpolation */}</p>
<p>::lens{scenario=gpu-spatial-analysis.areal-interpolation rung=datum datum=5 caption="Inspect community area row 5, Lake View, and its tract weights."}</p>
<p>::lens{scenario=gpu-spatial-analysis.areal-interpolation rung=formula caption="Each tract's value is split by the share of its raster cells in each area."}</p>
<p>::lens{scenario=gpu-spatial-analysis.areal-interpolation rung=kernel caption="Read the weight kernel that divides each overlap area by the source area."}</p>
<p>::lens{scenario=gpu-spatial-analysis.areal-interpolation rung=graph caption="Follow rasterization, pair sorting and run detection, then the transfer."}</p>
<h3 id="gpupycnophylactic"><code>GPUPycnophylactic</code></h3>
<p>Smooths zone totals into a non-negative raster surface whose cells sum back to each zone total.</p>
<p>::lens{scenario=gpu-spatial-analysis.pycnophylactic rung=result caption="199 tract totals (719,651 residents) smoothed onto a 100 m raster; tract outlines for reference."}</p>
<p>Tobler pycnophylactic interpolation on a raster. Props: <code>grid</code> (<code>GPURasterMetadata</code>), <code>zones</code> (<code>uint32</code>; a value at or
above <code>zoneCount</code> is outside), <code>zoneCount</code>, <code>totals</code> (per zone), <code>maximumIterations</code> (0 to 512,
compile-time), <code>kernel</code> (<code>'rook'</code>, the default 4-neighbor mean, or <code>'box'</code>, a 3x3 mean including the cell)
and the caller-owned <code>output</code> (<code>GPUPycnophylacticOutput</code>) whose <code>mass</code> is the mass per cell. Each cell starts at <code>total / area(zone)</code>. Each iteration takes the focal mean over in-area
neighbors, restores the zone total additively, clamps negatives to 0 and rescales multiplicatively to the
total; zero-sum zones fill uniformly. Totals hold to f32 accuracy and values are non-negative after every
iteration. Zone sums are fixed-order segmented sums over cells sorted once by zone. There is no ancillary
initial surface and no barrier. <code>'rook'</code> on a checkerboard has a -1 eigenmode (Tobler's original); <code>'box'</code>
is smoother. Reference: Tobler (1979), JASA 74(367).</p>
<ul>
<li>Deviation from Tobler (1979). The restoring step is additive, then clamped, then multiplicative.
Tobler alternates smoothing with a volume-restoring adjustment; the multiplicative step is kept
because it is the only variant here that keeps exact totals and non-negativity together, but where
the additive step clamped it can leave flat plateaus, and an additive-only variant would go negative
next to much larger zones.</li>
<li>Convergence. <code>maximumIterations</code> is the cap. A one-row <code>tolerance</code> (per frame) compares, after each
iteration, the largest <code>|new - old| / (total / area(zone))</code> over all cells with it. Once at or below
the tolerance the surface is final: the per-cell kernels of every remaining compiled iteration are
skipped through a GPU-written indirect dispatch, and the surface is not copied forward (one settle pass
copies the scratch raster only when the last iteration that ran wrote it). Deterministic.
<code>output.status.converged</code> and <code>output.status.iterationCount</code> (one <code>uint32</code> each; they need <code>tolerance</code>) report
whether it converged and how many iterations ran; with 0 iterations both are 0. A NaN or negative tolerance never converges.</li>
<li>Zones without cells. A zone smaller than a cell has no raster cell and its total used to vanish
silently. <code>zoneAnchorCells</code> (<code>zoneCount</code> cell indices) lets such a zone claim a fallback cell (see
<code>GPUArealInterpolation.sourceAnchorCells</code>; ascending zone order, the cell's owner must keep one cell),
and <code>output.status.invalidCount</code> counts zones that still have no cell or no area.</li>
<li>Coordinates. With a <code>longitude-latitude</code> <code>spatialContext</code> plus <code>parameters</code> (as above; north-up grids are handled) the smoothed
quantity is density per context length unit squared (per square meter by default, per square kilometer
with <code>units: 'kilometers'</code>, previously still per square meter). Cell areas use
<code>getGPUAuthalicSphereRadius</code> in the context's length unit, and totals are enforced on density times the spherical
cell area, so <code>output.mass</code> is mass per cell (density times area) and the optional <code>output.density</code>
view is the surface itself. Without it every cell counts the same, which is only right on an equal-area raster.
<strong>Breaking:</strong> longitude/latitude now requires <code>metric: 'great-circle'</code>; <code>'none'</code> throws.
Declares <code>measure: 'area'</code>: an azimuthal-equidistant local frame throws; project into a Lambert equal-area frame.
The focal mean is still by cell index, so away from the equator it is anisotropic on the ground.</li>
<li>Limits. Counts are f32 sums, exact to 2^24 cells per zone; a larger zone loses count precision (the
mass still holds to f32 accuracy). Oracles: a float64 numpy reimplementation of the documented
algorithm including tolerance and anchors (<code>pycnophylactic.py</code>).</li>
</ul>
<h4 id="how-it-computes-3">How it computes</h4>
<p>{/* lens:GPUPycnophylactic */}</p>
<p>::lens{scenario=gpu-spatial-analysis.pycnophylactic rung=datum datum=3576 caption="Inspect the cell containing the centroid of tract 17031020802."}</p>
<p>::lens{scenario=gpu-spatial-analysis.pycnophylactic rung=formula caption="One iteration: smooth, restore the zone total, clamp at zero, rescale."}</p>
<p>::lens{scenario=gpu-spatial-analysis.pycnophylactic rung=kernel caption="Read the smoothing, restore and rescale kernels of one iteration."}</p>
<p>::lens{scenario=gpu-spatial-analysis.pycnophylactic rung=graph caption="Follow the zone sort, then up to 30 smoothing iterations gated by the tolerance."}</p>
<h3 id="gpusegregation"><code>GPUSegregation</code></h3>
<p>Computes aspatial and spatial segregation indices from a unit-by-group count table, with per-unit contributions.</p>
<p>::lens{scenario=gpu-spatial-analysis.segregation rung=result caption="Five population groups across 199 tracts; tracts shaded by their contribution to the entropy index H."}</p>
<p>Segregation indices from a unit-by-group count table, aspatial or spatial (PySAL <code>segregation</code>).</p>
<ul>
<li>Props: <code>groupCount</code> <code>K</code> (2 to 16), <code>groupCounts</code> (<code>units * K</code> float32, row-major; the unit count is
<code>groupCounts.length / K</code>),
<code>scales</code> (default <code>[null]</code>; <code>null</code> is aspatial, otherwise one square <code>GPUSpatialWeights</code> per bandwidth,
which gives the multiscale profile), <code>selfWeight</code> (1), <code>atkinsonInequalityAversion</code> (<code>b</code>, 0.5, in
<code>(0, 1)</code>), <code>spatialForm</code> and <code>output</code>: <code>indices</code> (<code>scales * stride</code> floats, layout from <code>getGPUSegregationLayout(K)</code>: entropy,
multi-group dissimilarity, diversity, dissimilarity <code>[K]</code>, isolation <code>[K]</code>, Atkinson <code>[K]</code>, interaction
<code>[K * K]</code>) and optional per-unit views (<code>localEnvironment</code>, <code>localEntropy</code>, <code>localDissimilarity</code>,
<code>localTheil</code>, per scale; they sum to the global indices).</li>
<li><code>spatialForm: 'environment'</code> (default) is the Reardon and O'Sullivan (2004) form: <code>pi_im = x~_im / t~_i</code>
with <code>x~_im = selfWeight x_im + sum_j w_ij x_jm</code> (<code>j</code> over the active slots: positive weight, another
unit; a self slot in the weights is not read, so a unit is never counted twice) replaces <code>p_im</code>, and the sums keep the unit populations
<code>t_i</code>. <code>'smoothed-population'</code> replaces the whole table by the smoothed table <code>x~</code> and runs the aspatial
formulas on it, which equals PySAL's spatially implicit indices (PySAL also rounds the smoothed counts).</li>
<li>With <code>P_m = X_m / T</code>, <code>E = sum P ln(1 / P)</code>, <code>E_i = sum pi ln(1 / pi)</code> and <code>I = sum P (1 - P)</code>: Theil
<code>H = sum_i t_i (E - E_i) / (E T)</code>; multi-group <code>D = sum_i sum_m t_i |pi_im - P_m| / (2 T I)</code>;
<code>D_g = sum_i t_i |pi_ig - P_g| / (2 T P_g (1 - P_g))</code>; isolation <code>xPx_g = sum_i (x_ig / X_g) pi_ig</code>;
interaction <code>xPy_gh = sum_i (x_ig / X_g) pi_ih</code>; Atkinson
<code>A_g = 1 - P / (1 - P) |sum_i (1 - pi_ig)^(1 - b) pi_ig^b t_i / (P T)|^(1 / (1 - b))</code>. Undefined indices
are NaN, never 0 (0 means perfectly even): H and the diversity with an empty total, H also with a single
present group (<code>E = 0</code>, while <code>E</code> itself is the defined 0), D with an empty total or <code>I = 0</code>, <code>D_g</code> and
<code>A_g</code> for a group with zero or all of the population, and isolation and interaction of a group with no
population. A unit with an empty environment (no population in it or its neighbors) has NaN local values
and contributes nothing to the sums, so use a NaN-skipping sum to add the local outputs up to the global
indices; unit contributions to <code>D_g</code> and H are also NaN when that global index is undefined. Behavior
change: these were 0.</li>
<li>Small-sample bias: the indices are not bias corrected. <code>D</code> and <code>H</code> are biased upward when units hold
few people, because sampling noise alone makes unit compositions differ from the global one even under
random allocation (Cortese, Falk and Cohen 1976; Winship 1977; Carrington and Troske 1997). Compare
units of similar size, or against a random-allocation null, before reading a small <code>D</code> or <code>H</code> as
segregation. Isolation and interaction are less affected but depend on the group share.</li>
<li>Float32 counts and totals are integer exact only up to 2^24 = 16,777,216 total population; above that
<code>T</code> and <code>X_m</code> round (about 6e-8 relative per addition).</li>
<li>Sums are f32, within 1e-5 to 2e-4 relative of float64 and of PySAL on the test fixtures. Memory is
<code>O(units * K^2)</code> per scale. The boundary-based <code>SpatialDissim</code> and permutation inference are not
provided: PySAL's evenness null redraws every person's unit from the unit sizes (a binomial or
multinomial draw per unit) and reruns the whole index pipeline per simulation, whereas the permutation
operators here shuffle values across rows of an existing weights structure, so it needs its own
count-table resampling kernel. <code>atkinsonInequalityAversion</code> and <code>selfWeight</code> are compile-time.</li>
</ul>
<h4 id="how-it-computes-4">How it computes</h4>
<p>{/* lens:GPUSegregation */}</p>
<p>::lens{scenario=gpu-spatial-analysis.segregation rung=datum datum=22 caption="Inspect tract 17031020802: its diversity and contribution to H."}</p>
<p>::lens{scenario=gpu-spatial-analysis.segregation rung=formula caption="Each tract's share of H: population times the diversity gap to the whole."}</p>
<p>::lens{scenario=gpu-spatial-analysis.segregation rung=kernel caption="Read the per-tract term kernel that also writes local entropy and Theil."}</p>
<p>::lens{scenario=gpu-spatial-analysis.segregation rung=graph caption="Follow the totals, environment, block reductions and index finalization."}</p>
<p>{/* parameters: GPUSegregation generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h4 id="gpusegregation-parameters"><code>GPUSegregation</code> parameters</h4>
<ul>
<li>Packer: <code>getGPUSegregationParameterValues(settings?, id?)</code>, writes <code>GPU_SEGREGATION_PARAMETER_LENGTH</code> (2) <code>float32</code> values.
<ul>
<li><code>id</code> (<code>unknown</code>)</li>
</ul>
</li>
</ul>
<p>Settings of <code>getGPUSegregationParameterValues</code>:</p>
<table>
<thead>
<tr>
<th>Field</th>
<th>Type</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>selfWeight</code> (optional)</td>
<td><code>number</code></td>
<td>-</td>
</tr>
<tr>
<td><code>atkinsonInequalityAversion</code> (optional)</td>
<td><code>number</code></td>
<td>-</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>`};export{e as default};