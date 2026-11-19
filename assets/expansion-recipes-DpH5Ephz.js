var e={slug:`api/gpu-spatial-analysis/expansion-recipes`,title:`Geospatial workflow recipes`,html:`<h1 id="geospatial-workflow-recipes">Geospatial workflow recipes</h1>
<p>All recipes exported from the spatial and network recipes entry points use the
<a href="/luma-contributors/docs/api/gpu-spatial-analysis/recipes">shared preparation and retry contract</a>: validation and expansion failures leave the
caller graph unchanged, including when a raster wrapper composes another recipe.</p>
<p>These recipes add nodes to a caller-owned <code>GPUCommandGraph</code>. They never submit work, read back a
result, fit a model on the CPU, or silently change coordinate systems. Import spatial, raster and
terrain workflows from <code>@luma-contributors/analysis/gpu-spatial-analysis</code>; import network scenarios and the track-quality/map-matching composite from
<code>@luma-contributors/analysis/gpu-network</code>.</p>
<p>Every result exposes <code>outputs</code>, <code>intermediates</code>, <code>contributors</code> and <code>status.stages</code>. Inspect the
stage status as well as the numerical output: a partial routing result or an incomplete geometry
is not a completed analysis. Graph-owned transient outputs can feed another contributor or renderer.
Use a caller-owned output override where exposed, or an explicit graph copy to a caller-owned
readback buffer. Destroy the compiled graph after its last submission has completed.</p>
<p>Catalog kinds describe each recipe's actual ports. Local variability consumes CSR spatial weights,
values and parameters and returns per-row attributes; it has no raster input or network output.
Zonal distribution returns attributes and summaries from values with zones or a point/polygon join;
its raster adapter accepts a surface. Exposure returns zone summaries, regional raster change also
returns a difference surface, and interpolation produces a surface plus sample diagnostics from
point observations. Drainage is a surface-to-network workflow. Cell-flow topology takes H3 cells
and an OD pair table and returns cell/edge attributes, topology and boundary geometry. Surface
preparation returns the repaired surface, provenance attributes and an unresolved count.</p>
<h2 id="zonal-distributions">Zonal distributions</h2>
<p><code>addZonalDistributionRecipe</code> accepts calibrated float32 values and either dense zone IDs or point
and polygon inputs for <code>GPUPointInPolygonJoin</code>. It returns dense zone IDs, assigned row counts,
finite-value counts, medians, modes, unique counts and optional quantiles. Empty zones remain in the
output, with zero counts and NaN order statistics. Modes break ties toward the lowest value;
quantiles use linear interpolation. Probabilities are supplied in a live float32 view.</p>
<p>The point path assigns each observation to the join's single deterministic owning feature. Supply
explicitly expanded zone/value rows when one observation should count in multiple overlapping zones.
Source feature IDs are retained in the input; output zone IDs are feature row indices.
The point branch accepts <code>source.join.prepared</code> when that <code>GPUSpatialJoinPrepared</code> has already been
added to the caller graph. Preparation mirrors its geometry and storage for validation; the
committed join retains the original handle. An undeclared handle is rejected before any recipe
nodes or resources are registered.</p>
<p><code>addRasterZonalDistributionRecipe</code> additionally accepts <code>grid</code>, <code>input</code>, <code>zones</code> and <code>zoneCount</code>.
It resolves native-format nodata and validity before applying scale/offset. These are unweighted
whole-cell quantiles. Fractional coverage does not turn them into weighted quantiles.
Caller outputs must not alias the original raster band, its validity storage, zone IDs or
probabilities, even though the nested distribution reads a calibrated intermediate. Distinct
outputs also require separate buffers; disjoint offset views of one buffer still alias.
Only declared zonal fields are forwarded to the nested recipe; auxiliary metadata such as <code>preview</code>
is ignored. A genuinely foreign raster input still fails graph ownership validation. The ordered
<code>contributors</code> list includes calibration before the zonal statistics stage (and both calibration
stages before exposure statistics in <code>addExposureByZoneRecipe</code>).</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">import</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8"> {addRasterZonalDistributionRecipe} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">from</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF"> '@luma-contributors/analysis/gpu-spatial-analysis'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">;</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> distributions</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> addRasterZonalDistributionRecipe</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(graph, {</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  grid,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  input: elevationBand,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  zones: zoneIds,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  zoneCount: administrativeAreaCount,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  probabilities: quantileProbabilities, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// e.g. [0.1, 0.5, 0.9]</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  outputs: {medians: medianOutput, quantiles: quantileOutput}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">});</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// distributions.status.stages retains grouping overflow and, for point sources, join status.</span></span></code></pre>
<h2 id="network-disruption-comparison">Network disruption comparison</h2>
<p><code>addNetworkDisruptionRecipe</code> accepts a cost-matrix <code>search</code>, explicit <code>rowCount</code>, opportunity
weights, accessibility parameters, and an optional edge mask or replacement edge costs. Zero mask
words close edges. The adapter retains the original CSR slots, including parallel edges, so IDs
remain stable. Targets, when supplied, define the opportunity-weight column order.</p>
<p>Outputs include baseline/scenario matrices, cumulative accessibility, accessibility differences,
finite pair-cost differences and a pairwise newly-unreachable mask. Cost differences are NaN when
both costs cannot be compared. Newly-unreachable flags are set only after both searches converge.
With a finite search cost cutoff, “unreachable” means outside that routing horizon; it does not
establish global disconnection. Always retain the two routing convergence words.</p>
<p>To avoid rerunning the baseline, pass <code>{costs, status: {converged}}</code> as <code>baseline</code>, imported into the scenario
graph from retained caller-owned buffers. The caller must invalidate that baseline when topology,
weights, starts, targets or routing parameters change. Reusing transient storage from another graph
is not supported. Accessibility is cumulative opportunity weight within the supplied threshold.</p>
<h2 id="interpolation-diagnostics">Interpolation diagnostics</h2>
<p><code>addInterpolationDiagnosticsRecipe</code> takes the existing <code>GPUKriging</code> inputs, including an explicitly
supplied variogram and neighborhood. It returns the prediction raster and variance, leave-one-out
sample predictions/variances, residuals, standardized residuals, bias, MAE, RMSE and finite counts.
Masked samples and unusable predictions are excluded. A standardized residual needs a finite,
strictly positive variance and a finite quotient; zero-variance duplicates remain undefined.</p>
<p>Summary counts and error metrics use row contributions and published parallel reductions. Residuals
are divided by their maximum absolute value and finite count before summing or squaring, so means
and RMSE remain finite even for maximum finite float32 residuals. Normalization uses the shared
scaled quotient (bounded mantissas and integer exponents) to avoid an underflowing reciprocal of a
near-maximum residual scale. All-invalid validation sets return zero valid counts and NaN metrics;
all-zero residuals return zero metrics. Reduction rounding can differ from the former streaming
means (absolute tolerance <code>1e-6</code> plus <code>2e-6</code> times the maximum residual
magnitude). Work remains O(sampleCount), with hierarchical reductions instead of a serial sample
scan. Separate invalid/uncertain raster statuses and invalid validation counts remain visible. No
variogram fitting, automatic model selection or residual significance is implied.</p>
<h2 id="facility-scenario-evaluation">Facility scenario evaluation</h2>
<p>Independent output buffers must be physically distinct, including when different graph handles
import one buffer. In particular, <code>nearestCosts</code> cannot alias <code>accessibility</code>; correcting the outputs
allows a retry with the same recipe ID before any caller nodes or resources have been added.</p>
<p><code>addFacilityScenarioRecipe</code> reads a complete, reusable facility-major cost matrix, demand, supply,
a live facility mask and network-accessibility parameters. Outputs are gravity accessibility,
nearest selected facility/cost per demand row, covered/uncovered demand, and invalid-demand count.
Equal nearest costs choose the lowest facility row. Nonfinite or negative demand is omitted and
counted; unreachable valid demand remains uncovered. An empty selected set leaves every nearest
cost infinite and facility ID <code>0xffffffff</code>. Covered/uncovered demand and invalid counts use parallel
row contributions and published reductions. Demand totals use float32 tree sums; acceptance tolerance
is <code>1e-6 + 2e-6 * abs(total)</code> against a float64 sum.</p>
<p>Run the recipe once per scenario with a distinct ID and the same matrix/demand to compare results.
Changing masks or scoring parameters does not rerun routing. The caller must ensure the matrix's
routing search completed and its horizon covers the scoring threshold. Evaluation does not select
facilities or impose capacities; use the separately documented bounded allocation contributor for
those objectives. Nearest-cost outputs can feed the zonal distribution recipe for travel-cost
quantiles, with unreachable demand counted separately.</p>
<h2 id="exposure-by-administrative-area">Exposure by administrative area</h2>
<p><code>addExposureByZoneRecipe</code> accepts aligned hazard/population buffer bands, a common grid, whole-cell
zone IDs and a live one-row threshold. Values at or above the threshold are exposed. For
<code>weightMeaning: 'count'</code>, population is an extensive cell count. For <code>'density'</code>, <code>cellAreas</code> is
required and must use the area units of the density denominator.</p>
<p>Outputs are total known population, population with a known hazard, exposed population, observed
population coverage and exposed share <strong>among observed population</strong>. Missing hazards do not become
safe observations. Zero denominators produce NaN shares. Raw nodata, masks and calibration are
resolved independently for each band. A nonfinite threshold sets <code>thresholdInvalidCount</code> and
returns NaN exposed totals and shares while preserving observed population coverage. Grid
alignment and polygon-to-cell ownership are explicit upstream operations; this recipe does not
infer fractional boundary membership.</p>
<h2 id="regional-raster-change">Regional raster change</h2>
<p><code>addRegionalRasterChangeRecipe</code> accepts an aligned, calibrated, cell-major single-band slice stack,
explicit <code>GPUChangeDetection</code> comparison parameters, zone IDs, per-cell areas and a live absolute
change threshold. Encode missing samples as NaN and use the optional cell mask. The comparison is
<code>after - before</code> on common support.</p>
<p>Outputs include cell differences, per-zone finite comparison counts, unweighted mean differences,
changed area, valid comparison area, total area and area coverage. Areas must be finite and
positive. Means/counts describe finite difference observations; area accounting additionally
requires usable cell areas. Thresholds must be finite and nonnegative; otherwise
<code>thresholdInvalidCount</code> is set and changed area is NaN, while observed coverage is retained. These descriptive regional
summaries do not claim significance of a regional mean.</p>
<h2 id="cluster-footprints">Cluster footprints</h2>
<p><code>addClusterAndOutlineRecipe</code> retains its convex default and exposes the separately documented
concave outline option. Alpha-shape geometry uses a circumradius threshold, with holes and
multiple pieces retained. An alpha radius is not Shapely's concave-hull ratio. Inspect geometry
completion and capacity status before consuming its measures.
Select <code>outline: {kind: 'alpha', maximumCircumradius}</code> and consume <code>concaveHull</code> mesh and boundary
outputs. Legacy single-ring convex-hull fields are empty in alpha mode. Areas, perimeters and
centroids describe the selected geometry.</p>
<h2 id="drainage-network-analysis">Drainage network analysis</h2>
<p><code>addDrainageNetworkRecipe</code> composes D8 flow, stream order, watersheds, downstream flow distance,
connected stream reaches and per-catchment distance summaries on one complete domain. Pass ordinary
<code>GPUTerrainFlow</code> inputs in <code>flow</code>; its live parameter buffer supplies the stream threshold. Optional
pour points are cell IDs. Depression filling and flat resolution remain explicit flow options.</p>
<p>The result includes directions, accumulation, stream mask/order, catchment labels, downstream
distance/terminal cells, and a compact network with source/target cell IDs, vertex offsets,
positions and lengths. Without pour points, catchments are keyed by terminal-cell IDs. With pour
points, they are keyed by pour-point row. Summary counts and mean distances include resolved valid
cells; unresolved cells remain available through raster statuses.</p>
<p>Reach capacity defaults to cell count and vertex capacity to twice cell count; both are explicit
options. Preserve fill, flat, accumulation, order, watershed and network completion evidence.
<code>maximumSteps</code> bounds tracing. This recipe does not stitch independent tiles or interpret branching
D-infinity/MFD flow as D8. Lengths/distances use the planar grid affine's units.</p>
<h2 id="track-quality-and-route-conformity">Track quality and route conformity</h2>
<p><code>addTrackQualityRecipe</code> is exported by <code>@luma-contributors/analysis/gpu-network</code>, since its optional map-matching stage consumes network topology. It preserves the trajectory's raw observation order and timestamp representation,
including double-single and Int64 words. Its live two-row threshold view holds maximum speed and
maximum gap in trajectory units. The recipe forwards <code>trajectory.temporalContext</code> to trajectory
metrics, preserving the declared unit, epoch and encoding; a declaration inconsistent with the
timestamp views is rejected before graph resources or nodes are declared, allowing a corrected
retry on the same graph. Raw flags use bits 1 invalid position, 2 nonpositive/nonfinite time step,
4 excessive/nonfinite step speed, 8 time gap and 16 invalid threshold. First observations
have no step checks but still report an invalid timestamp;
rows outside the track-offset domain carry <code>0xffffffff</code>. Per-track counts count flagged observations.</p>
<p>Optional <code>similarity</code> compares the raw tracks to explicit reference pairs using Hausdorff, Frechet,
DTW and LCSS. Its densification and dynamic-programming limits remain explicit, and <code>pairStatus</code>
retains failure evidence. <code>pairA</code> and <code>pairB</code> must have equal logical lengths. Invalid pair shapes
leave the graph unchanged, and corrected pairs can retry with the same recipe ID. Supplying
<code>frechetEpsilon</code> also returns <code>frechetWithin</code>. Optional <code>mapMatching</code> returns matched edge IDs, snap distances, breaks
and matched count, plus candidate and route-overflow stages. Route-search overflow must not be
interpreted as a reliable off-route result. Resampling is a separate upstream choice; raw flags do
not silently become flags on a resampled track.</p>
<p>By default bit 4 is the step-speed test, so the valid fix after an outlier is flagged too; set <code>speedOutlierMode: 'point-centred'</code> to flag a fix only when the speed test fails on both of its adjacent steps (see <a href="/luma-contributors/docs/api/gpu-network/network-movement-coverage">network movement coverage</a>).</p>
<h2 id="local-variability-comparison">Local variability comparison</h2>
<p><code>addLocalVariabilityRecipe</code> uses one explicit weights graph, values and mask for LOSH and local
Moran. It returns separate statistic and classification columns, per-row statuses and invalid counts.
LOSH is descriptive. Moran's analytic-normal p-values and optional correction apply to the Moran
family only; there is no combined significance score. See the statistical expansion documentation
for residual-power, island, bandwidth and inference contracts.</p>
<h2 id="cell-flow-topology">Cell flow topology</h2>
<p><code>addCellFlowTopologyRecipe</code> reads distinct, same-resolution H3 cells and original OD pair rows.
It returns incoming/outgoing cell totals, canonical directed edges, neighboring-cell edge flows,
shared-boundary geometry and pair validity. It retains the original arbitrary pair table. Self and
nonneighbor pairs contribute to cell totals but do not become H3 edges. Total outgoing flow equals
total incoming flow and the valid-pair sum; edge-flow totals may be smaller. See the
<a href="/luma-contributors/docs/api/gpu-network/expansion">network expansion reference</a> for the bounded direct-scan cost and H3
boundary precision.</p>
<h2 id="surface-preparation-with-provenance">Surface preparation with provenance</h2>
<p><code>addSurfacePreparationRecipe</code> applies bounded continuous nodata filling and an optional rank filter.
Its independent outputs must use separate physical buffers, including through separate graph imports;
read-only input sharing remains legal. The recipe captures declared output fields once, including
inherited getters, and checks the caller-provided boundary before allocating default outputs or
scheduling nodes. Each supplied output is also checked for packed scalar storage, its expected
format and exact row count before any defaults are registered. <code>values</code> is float32; the other
five outputs are uint32. Each output has <code>grid.width * grid.height</code> rows except <code>unresolvedCount</code>,
which has exactly one row. After a boundary or layout rejection, correct the supplied views and
retry on the same graph with the same recipe ID.
It returns a float32 buffer band, original validity, filled and filtered masks, downstream validity,
and unresolved-gap counts before and after filtering. Original valid calibrated observations are
preserved in fill-only mode; interpolated values never become donors during that fill pass.</p>
<p><code>maximumDistance</code> is in pixels. <code>method: 'nearest'</code> chooses the nearest original donor;
<code>'inverse-distance'</code> weights all original donors in the disk. This is distinct from GDAL's
four-quadrant donor selection. Rank filtering uses a square footprint, explicit border/nodata rules
and the SciPy rank convention (upper middle for an even median population). Filling missing
measurements is separate from filling drainage depressions. Supply the halo needed by both stages
when preparing tiles and publish only nonoverlapping cores.</p>`};export{e as default};