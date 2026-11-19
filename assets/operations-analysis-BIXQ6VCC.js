var e={slug:`api/gpu-raster/operations-analysis`,title:`GPURaster analysis contributors`,html:`<h1 id="gpuraster-analysis-contributors">GPURaster analysis contributors</h1>
<h2 id="overview">Overview</h2>
<p>The <code>@luma-contributors/analysis/raster-analysis</code> entry point exports analysis contributors that build on the
operations in the other families: zonal statistics, contrast stretch, isolines and isobands,
Euclidean distance fields, cost distance, polygon rasterization, raster joins, reclassification,
sampling and profiles, change detection, particle advection, streamlines, and line integral
convolution, and exact min/max pyramids. Each is an algorithm or workflow that declares resources and command nodes into a
caller-owned <code>GPUCommandGraph</code> through <code>getCommandNodes(graph)</code> (<code>GPUCommandNodeProducer</code>). It never
compiles, submits, encodes, or reads back.</p>
<h2 id="when-to-use-it">When to use it</h2>
<p>Use these contributors when a map analysis needs a finished, bounded result rather than a
building block, and when its per-frame inputs (levels, seeds, windows, thresholds) change without
recompiling.</p>
<ul>
<li><code>GPUIsolines</code> and <code>GPUIsobands</code> extract several levels at once, order segments
deterministically, and can stitch polylines or filled bands. Use <code>GPURasterContours</code> for a single
level of raw marching-squares segments, and <code>GPUTerrainContours</code> (see
<a href="/luma-contributors/docs/api/gpu-terrain">GPU Terrain</a>) for elevation levels with one overflow flag.</li>
<li><code>GPURasterStretch</code> computes linear, percentile, and equalization stretches together with a colormap
lookup and statistics row, using integer atomics only. Use <code>GPURasterContrast</code> for a
calibrated, nodata-aware contrast or gamma that keeps the scalar domain, and <code>GPURasterHistogram</code>
when you only need the histogram.</li>
<li><code>GPURasterZonalStatistics</code> reduces a raster band over a pre-rasterized grid of zone IDs. Use
<code>GPUZonalStatistics</code> from <a href="/luma-contributors/docs/api/gpu-spatial-analysis">GPU Spatial Analysis</a> when the input
is points and polygon features instead of a raster.</li>
<li><code>GPUDistanceField</code> and <code>GPUCostDistance</code> are the Euclidean and friction-weighted surfaces. Use
<a href="/luma-contributors/docs/api/gpu-network">GPU Network</a> reachability when travel follows a
road network rather than a grid.</li>
</ul>
<h2 id="quick-start">Quick start</h2>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">import</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8"> {GPUCommandGraph} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">from</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF"> '@luma.gl/graph/gpu-core'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">;</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">import</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8"> {GPUParameterBuffer} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">from</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF"> '@luma-contributors/analysis/analysis-core'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">;</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">import</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8"> {</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  GPUDistanceField,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  getGPUDistanceFieldParameterValues</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">from</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF"> '@luma-contributors/analysis/raster-analysis'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">;</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> graph</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUCommandGraph</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(device, {id: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'raster-analysis'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">});</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> parameters</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUParameterBuffer</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(device, {id: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'df-parameters'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, format: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'float32'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, length: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">8</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">});</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// \`grid\` is a \`GPURasterMetadata\` (see \`makeGPURasterMetadata\`). \`seedPositions\`, \`seedCount\`,</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// \`seedMask\`, and the output views are graph views created on \`graph\`.</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUDistanceField</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  grid, parameters: parameters.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">importToGraph</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(graph),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  seedPositions, seedCount, seedMask,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  output: {distances, allocation, withinDistance}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> compiled</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> await</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8"> graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">compileAsync</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(); </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// once</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">parameters.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">write</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">getGPUDistanceFieldParameterValues</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({maximumDistance: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">500</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}, grid));</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">compiled.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">encode</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(device.commandEncoder, {parameters: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">undefined</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">});</span></span></code></pre>
<p><code>encode()</code> only records; submit with <code>device.submit()</code>. <a href="/luma-contributors/docs/api/gpu-spatial-analysis/concepts#running-contributors">Running contributors</a>
covers <code>compileAsync()</code>, rebuilding without a stall, submission and why <code>parameters</code> is <code>undefined</code>.</p>
<h2 id="conventions">Conventions</h2>
<ul>
<li><strong>One grid.</strong> Grid contributors take <code>grid: GPURasterMetadata</code>; <code>makeGPURasterMetadata({width, height, cellSize, origin, rowDirection})</code> builds an axis-aligned area grid. Packers whose layout
holds a cell size, extent or latitude band take the frame's grid as their second argument and
read it from <code>grid.affine</code>, so it is never typed twice. <code>makeGPURasterBand(view, grid)</code> wraps an
output view as the input band of the next contributor. Values and optional <code>validity</code> may have
spare capacity: the helper borrows exactly <code>grid.width * grid.height</code> entries at each view's
existing byte offset, without copying. Both views must be packed, cover the grid and belong to
the same graph.</li>
<li><strong><code>parameters</code> in, <code>output</code> out.</strong> The packed per-frame view is <code>parameters</code> (a second view of
another format is <code>uint32Parameters</code>); outputs live in one <code>output</code> bag, the validity output is
<code>validity</code> (an input row filter is <code>mask</code>), and one-row status words are in <code>output.status</code>.</li>
<li><strong>Graph views in, graph views out.</strong> Inputs and outputs are <code>GraphDataView</code>, <code>GraphVectorView</code>, or
<code>GraphTextureView</code> objects created on the target graph. Outputs are always caller-owned; scratch
storage is graph transients that die with <code>compiled.destroy()</code>. Contributors never compile, encode,
submit, or read back.</li>
<li><strong>Per-frame values never recompile.</strong> Viewports, thresholds, radii, observer positions, budgets,
and region shapes are read from storage views, usually a <code>GPUParameterBuffer</code> that the
application rewrites with <code>write()</code>. Lengths, capacities, grid sizes, and which optional views
exist are compile-time topology; each prop's TSDoc says which category it belongs to.</li>
<li><strong>Bounded results report overflow on the GPU.</strong> Compact ID lists use <code>GPUCompactOutput</code>
(<code>ids</code> and a <code>status</code> with <code>count</code>, <code>overflow</code> and optional <code>requiredCount</code>). <code>status.count</code> is clamped
to <code>ids.length</code> and can be an indirect draw instance count; <code>status.overflow</code> is rewritten every
encoding.</li>
<li><strong>Stable IDs.</strong> Result IDs are the caller's <code>sourceIds[row]</code> (or tile IDs) when given and zero-based
rows otherwise. Node and transient IDs are <code>\${id}-&#x3C;step></code>, so two instances in one graph need
different <code>id</code> props.</li>
<li><strong>Shared contracts.</strong> Four shapes mean the same on every analysis page:
<ul>
<li><strong>Status.</strong> Every GPU-written status word sits in the <code>status</code> member of the bag that owns it:
<code>output.status.overflow</code>, <code>output.status.converged</code>, and <code>output.status.code</code> for model fits.</li>
<li><strong>Grid.</strong> Raster and terrain contributors take <code>grid: GPURasterMetadata</code>, and their packers read
the cell size from it (see <a href="/luma-contributors/docs/api/gpu-terrain/concepts#conventions">GPU Terrain conventions</a>).</li>
<li><strong>Network.</strong> Network contributors take <code>topology: GPUNetworkTopology</code> (<code>offsets</code>, <code>neighbors</code>,
<code>weights</code>, optional <code>nodePositions</code>). Its CSR part can be passed as <code>GPUSpatialWeights</code> (see
<a href="/luma-contributors/docs/api/gpu-network/concepts#conventions">GPU Network conventions</a>).</li>
<li><strong>Polygons.</strong> Polygons use the four-level <code>GPUPolygonGeometryPort</code>: <code>kind: 'polygons'</code>,
<code>positions</code>, <code>featureOffsets</code>, <code>polygonOffsets</code> and <code>ringOffsets</code>. Contributors with a single
geometry take the same fields flat, without <code>kind</code> (<code>GPUPolygonGeometryFields</code>). See
<a href="/luma-contributors/docs/api/gpu-spatial-analysis/concepts#shared-ports-status-and-planning">Shared ports</a>.</li>
</ul>
</li>
<li><strong>Extent-local coordinates.</strong> Positions, extents, seeds, sample points, and output vertices are
float32, whose ulp is 0.5 m at a UTM northing of 4.6e6 m and 2 m at 2e7 m in EPSG:3857. The
parameter packers of <code>GPUDistanceField</code>, <code>GPUIsolines</code>, <code>GPUIsobands</code> (and <code>GPUIsobandRings</code>),
<code>GPURasterSampling</code>, and <code>GPURasterProfile</code> take an optional float64 <code>localOrigin</code>: they subtract
it from the extent or grid origin in float64, seeds, points, and line vertices are uploaded
relative to it, and contour, band, and sample-position outputs come back relative to it. Add it
back on the CPU or as the layer's coordinate origin. With the raster or tile corner as origin,
positions keep about <code>1.2e-7</code> of their distance from it (a 2.5 m UTM raster places contour
vertices within 0.02 mm of a float64 reference, against 0.34 m with absolute coordinates). The
default <code>[0, 0]</code> keeps absolute coordinates. <code>GPUPolygonRasterization</code> follows the same rule
with caller-subtracted coordinates.</li>
<li><strong>Row direction.</strong> <code>GPURasterSampling</code>, <code>GPURasterProfile</code>, <code>GPUDistanceField</code>, <code>GPUCostDistance</code>,
<code>GPUPolygonRasterization</code>, <code>GPURasterReprojection</code> and the fractional coverage classes take the
row direction from <code>grid.affine</code>. <code>GPUIsolines</code>, <code>GPUIsobands</code>, <code>GPUIsobandRings</code> and the flow
family (<code>GPULineIntegralConvolution</code>, <code>GPUStreamlines</code>, <code>GPUParticleAdvection</code>,
<code>GPULagrangianFlowMap</code>, <code>GPUFiniteTimeLyapunovExponent</code>) do not: they place row 0 at the
smallest y of their extent and ignore the affine, with no check at construction. Give them a
raster whose rows run from south to north (reverse the row order of a north-up raster before
upload). In the flow family <code>v</code> is the velocity toward larger y. Making them follow the affine
is deferred (roadmap 12.4).</li>
<li><strong>Non-finite samples are nodata.</strong> NaN and both infinities are nodata in every raster
contributor on this page, <code>GPURasterSampling</code>, <code>GPURasterProfile</code> and <code>GPUFocalStatistics</code>
included (before 2026-11-10 those three kept an infinity as a value).</li>
<li><strong>One import per buffer.</strong> Import a buffer once (for example with <code>graph.importGPUData()</code> or
<code>GPUParameterBuffer.importToGraph()</code>) and pass the returned view to every contributor that reads
it, so the graph tracks hazards on one logical handle.</li>
</ul>
<h2 id="gpucostdistance-and-gpucostdistancepath"><code>GPUCostDistance</code> and <code>GPUCostDistancePath</code></h2>
<p><code>GPUCostDistance</code>: With continuation, settledAt stores cumulative last-decrease rounds across
resumed submissions. Explicit or automatic restart clears it together with costs/state; removing all
sources restarts the result.</p>
<p>Accumulated cost surface over a friction raster from per-frame source cells, a source mask, or
both: 8-connected moves cost ground distance times mean friction, invalid or negative friction is
a barrier, and a per-frame cost limit, isoline-ready bands with counts, and D8 back-links are
optional. Relaxation runs in 16x16 workgroup tiles with workgroup-memory repeats and per-tile
activity, gated on the GPU like <code>GPUNetworkReachability</code>. <code>GPUCostDistancePath</code> walks the
back-links from a per-frame target into a compact cell list. The cost surface feeds
<code>GPUTerrainContours</code> directly for isolines. Both take a <code>grid</code> and write into an <code>output</code> bag
(<code>costs</code>, <code>backLinks</code>, <code>bands</code>, <code>bandCounts</code> and <code>status</code> for <code>GPUCostDistance</code>; <code>path</code> and
<code>status</code> for <code>GPUCostDistancePath</code>). The grid supplies the width, height, cell size and latitude
band, and the packer takes it as its second argument. The move lengths are <code>hypot(a, d)</code> along a
row, <code>hypot(b, e)</code> along a column and the root of the sum of their squares on a diagonal, which
needs perpendicular pixel axes: axis-aligned and rotated grids are accepted, and a sheared affine
(<code>a * b + d * e</code> not zero) throws at construction and in the packer, because its two diagonals
have different lengths.</p>
<p>When <code>maximumIterations</code> stops the relaxation before a fixpoint, passable costs are NaN and bands
and band counts are <code>GPU_COST_DISTANCE_UNRESOLVED</code>, so a truncated surface never reads as a
converged one. Every relaxation round shares one pipeline per pass variant: its gate-group position is a GPU
word, not WGSL text.</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> grid</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> makeGPURasterMetadata</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({width, height, cellSize: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">10</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">});</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUCostDistance</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  grid, friction, parameters: parameters.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">importToGraph</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(graph),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  sources: sources.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">importToGraph</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(graph), maximumIterations: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">512</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  output: {costs, backLinks, status: {converged}}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUCostDistancePath</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  grid, backLinks, target: target.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">importToGraph</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(graph), output: {path: output}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">parameters.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">write</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">getGPUCostDistanceParameterValues</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({costLimit: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">5000</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}, grid));</span></span></code></pre>
<p><code>frictionParameters</code> (a <code>GraphDataView</code> of <code>[scale, offset]</code>, per frame; its presence is compile-time)
applies <code>friction = value * scale + offset</code> on top of the band's own calibration, so a friction slider is a
buffer write. Results that are negative or non-finite are impassable. Relaxation uses one gate node per four
rounds, so the unrolled loop has about a third fewer nodes (271 to 189 in the explorer, converging after 43
of 128 iterations).</p>
<p>Back-links across zero-friction plateaus come from a second, tie-level relaxation. It now defaults to
<code>maximumTieIterations = maximumIterations</code> (it was 8) and uses the cost phase's schedule, with directional tile
sweeps after <code>sweepAfterIteration</code>, so long or winding plateaus resolve. A link that a budget cut short
is <code>GPU_COST_DISTANCE_UNRESOLVED</code> (<code>0xfffffffe</code>), never 0, because 0 would claim a source. That covers
every passable cell when the cost relaxation stops at <code>maximumIterations</code>, and plateau cells whose tie
level did not converge. <code>GPUCostDistancePath</code> returns an empty path when the walk meets an unresolved
link, and sets its optional one-row <code>output.status.iterationLimitReached</code>. It also returns an empty path for
links that do not form a valid forest (an invalid code, a link off the grid or into an unreached
cell, a cycle), so it only ever reports a complete path from the target to a source.</p>
<p><code>maximumIterations</code> defaults to 512. After <code>sweepAfterIteration</code> iterations (default 96) relaxation adds
directional tile sweeps that carry a front along a whole row or column of tiles, so long corridors and
spirals converge in far fewer iterations; open terrain converges before the switch. Pass
<code>sweepAfterIteration >= maximumIterations</code> for plain tiled passes only.</p>
<p><code>continuation: {state, values}</code> (opt-in) spreads one relaxation over several submissions instead of
unrolling every round: <code>maximumIterations</code> then defaults to 64 and is the budget of one execution.
<code>state</code> is a zero-filled imported <code>uint32</code> buffer of <code>getGPURasterRelaxationContinuationStateLength(grid)</code>
words and <code>values</code> an imported <code>float32</code> per-cell buffer (neither can be a graph transient, which the
planner may alias). An execution resumes the previous one when that stopped unconverged, skipping
initialization and seeding; after a converged run, or after the caller writes a nonzero value at
element <code>GPU_RASTER_RELAXATION_RESTART_WORD</code> of <code>state</code> (the GPU clears it), it starts from scratch.
Set the restart word whenever an input changes. <code>output.status</code> is cumulative, the sweep schedule
follows the cumulative count, and <code>output.costs</code> stays NaN for passable cells until the converging
execution. Poll <code>converged</code> and submit again until it is 1. Not supported with <code>output.backLinks</code>.</p>
<p>Back-links are cycle-safe across zero-cost plateaus: a cell links to its strictly cheaper tight
neighbor or, when it was reached only across zero-cost moves, to the equal-cost neighbor one tie
level closer to a strict entry or a source. The tie phase runs only when <code>output.backLinks</code> is
requested and takes up to <code>maximumTieIterations</code> (default <code>maximumIterations</code>) extra gated
iterations; <code>output.status.converged</code> is 0 if either phase was truncated. The isochrone breaks are
the per-frame <code>costBreaks</code> input, required exactly when <code>output.bands</code> or <code>output.bandCounts</code> is given.</p>
<p>Generated transient IDs are prefixed with the contributor <code>id</code>, for example <code>&#x3C;id>-total</code>. If one
collides with a caller resource, the error names the contributor and the generated ID.</p>
<table>
<thead>
<tr>
<th>Output</th>
<th>Type</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>output.settledAt</code> (optional)</td>
<td><code>GraphDataView&#x3C;'uint32'></code></td>
<td>Packed, one value per cell: the 1-based gated relaxation round in which its cost last decreased; 0 for sources and cells whose cost never decreased. For a converged run, visualizes the propagation wavefront. Adds one store per improved cell; omit in production. Must be distinct from inputs and other outputs. Not supported with <code>continuation</code>.</td>
</tr>
</tbody>
</table>
<h4 id="gpucostdistance"><code>GPUCostDistance</code></h4>
<p>With continuation, settledAt stores cumulative last-decrease rounds across resumed submissions.
Explicit or automatic restart clears it together with costs/state; removing all sources restarts the
result.</p>
<p>Relaxes accumulated travel cost outward from a source cell over a friction raster until no cell improves.</p>
<p>::lens{scenario=gpu-raster.cost-distance rung=result caption="Accumulated cost from Greenville; water is impassable and stays unreached."}</p>
<p>{/* parameters: GPUCostDistance generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h5 id="gpucostdistance-parameters"><code>GPUCostDistance</code> parameters</h5>
<ul>
<li>Packer: <code>getGPUCostDistanceParameterValues(settings, grid)</code>, writes <code>GPU_COST_DISTANCE_PARAMETER_LENGTH</code> (8) <code>float32</code> values.
<ul>
<li><code>grid</code> (<code>GPURasterMetadata</code>): the frame grid, which supplies the cell size and row edges</li>
</ul>
</li>
</ul>
<p>Settings of <code>getGPUCostDistanceParameterValues</code>:</p>
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
<td><code>costLimit</code> (optional)</td>
<td><code>number</code></td>
<td>Candidate costs above this value stay unreached. Defaults to <code>Infinity</code>.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h4 id="how-it-computes">How it computes</h4>
<p>{/* lens:GPUCostDistance */}</p>
<p>::lens{scenario=gpu-raster.cost-distance rung=datum datum=2157 caption="One cell and its eight neighbours: ground length times mean friction, then the minimum."}
::lens{scenario=gpu-raster.cost-distance rung=formula caption="Least accumulated cost over eight-connected moves, written as a fixpoint of the relaxation."}
::lens{scenario=gpu-raster.cost-distance rung=iterations caption="The cost surface at key rounds; later gates skip work once nothing improves."}
::lens{scenario=gpu-raster.cost-distance rung=kernel caption="Tiled relaxation: each 16×16 tile repeats the update in workgroup memory."}
::lens{scenario=gpu-raster.cost-distance rung=graph caption="Friction preparation, seeding, gated relaxation rounds and the finalize node."}</p>
<h4 id="gpucostdistancepath"><code>GPUCostDistancePath</code></h4>
<p>Walks D8 back-links from each target cell to the source and lists the least-cost route cell by cell.</p>
<p>::lens{scenario=gpu-raster.cost-distance-path rung=result caption="Three least-cost routes from the Greenville source to tree-cover targets."}</p>
<h4 id="how-it-computes-1">How it computes</h4>
<p>{/* lens:GPUCostDistancePath */}</p>
<p>::lens{scenario=gpu-raster.cost-distance-path rung=datum datum=2930 caption="Follow one target cell by cell through its back-links until the walk reaches the source."}
::lens{scenario=gpu-raster.cost-distance-path rung=iterations caption="Cost relaxation rounds that produce the back-links the walk reads."}
::lens{scenario=gpu-raster.cost-distance-path rung=kernel caption="One thread walks the chain of back-links, bounded by the cell count."}
::lens{scenario=gpu-raster.cost-distance-path rung=graph caption="Cost distance with back-links feeds three walks; a histogram turns each path into a mask."}</p>
<h2 id="gpurasterzonalstatistics"><code>GPURasterZonalStatistics</code></h2>
<p>Optional zoneDictionary maps sorted native uint32 IDs to dense rows. sourceZoneIds and denseZoneRows
make row association explicit; invalidCount reports IDs missing from the dictionary. Omitting the
dictionary retains dense IDs. Strings and automatic dictionary construction are not supported.
For example, sourceZoneIds <code>[100, 900]</code> corresponds to denseZoneRows <code>[0, 1]</code>: the former preserves
caller identities and the latter publishes physical output rows (or <code>0xffffffff</code> for an invalid dictionary).</p>
<p>Counts cells and computes the mean, lowest and highest elevation of each land-cover zone from a zone raster.</p>
<p>::lens{scenario=gpu-raster.raster-zonal-statistics rung=result caption="Elevation statistics per land-cover zone; tree cover spans the widest range."}</p>
<p>Per-zone cell count, valid-value count, sum, mean, minimum, and maximum of a value band over a
dense <code>uint32</code> zone raster (for example rasterized administrative areas). Calibration, nodata,
validity, and non-finite samples are honored; one ignored zone ID is skipped silently, and zone IDs
at or above the compile-time capacity raise a GPU overflow flag. Composes <code>GPUGroupAggregation</code>.</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPURasterZonalStatistics</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  grid, zones, values: {id: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'ndvi'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, format: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'float32'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, storage: {kind: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'buffer'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, values}},</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  zoneCapacity: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">256</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, ignoredZone: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  output: {valueCounts, means, minimums, maximums, status: {overflow}}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span></code></pre>
<p><code>sumOrder</code> defaults to <code>'sorted'</code>: cells are stably sorted by zone ID and each zone is reduced in a
fixed tree (the same sorted segmented sum as <code>GPUZonalStatistics</code>), so <code>sums</code> and <code>means</code> are bitwise
reproducible on one device, at about 6-25 ms at 2048x2048 for any zone layout. <code>'atomic'</code> skips the
sort and is at most about 1.5x faster, only with 1000 or more spatially scattered zones; it can be
10-60x slower when neighboring cells share a zone, because float compare-exchange atomics contend on
one address, and its last bits can vary between runs. Counts, minimums, and maximums are exact either way.</p>
<p>Longitude/latitude rasters set <code>cellSizeMode: 'geographic'</code>, <code>parameterLayoutVersion: 2</code>, and a
per-frame <code>parameters</code> view <code>[cellSizeX, cellSizeY, northEdge, southEdge, signedRowStep, originPoleDistance, poleSign]</code> (version 2),
packed from the grid with <code>getGPURasterZonalStatisticsParameterValues(grid)</code>.
The signed row step preserves thin-cell area when the float32 latitude edges round equal.
<code>originPoleDistance = 90 - abs(firstRowEdgeLatitude)</code> is packed before float32 rounding;
<code>poleSign</code> is -1 south of the equator and +1 otherwise. Evaluating the midpoint relative to that
pole preserves shifted thin polar windows. Geographic weighting requires an axis-aligned grid.
The default <code>parameterLayoutVersion: 1</code> always decodes the legacy four-word prefix and ignores
all padding, including nonzero padding. It derives the row step from the edge difference and can
lose thin-cell area. Version 2 is explicitly selected and requires at least seven words; view
length never selects a layout. A packer target shorter than seven words requests the legacy prefix.
Each cell is then weighted by its area on the WGS84 authalic sphere: <code>sums</code> are <code>sum(a * v)</code> in
value times square meters, <code>means</code> are <code>sum(a * v) / sum(a)</code>, and the optional <code>weightSums</code> column is
the valid area of each zone. Counts, minimums, and maximums are never weighted, and the default
<code>'uniform'</code> mode is unchanged. Optional <code>variances</code> and <code>standardDeviations</code> are unbiased two-pass
estimates with the <code>GPUZonalStatistics</code> convention: <code>sum (v - mean)^2 / (n - 1)</code>, or with cell areas
as reliability weights <code>sum a (v - m)^2 / (V1 - V2 / V1)</code> (<code>numpy.cov(v, aweights=a, ddof=1)</code>). They
add a second pass over the cells, so request them only when needed. The divisor is <code>n - 1</code> (sample
form): two cells with values 1 and 3 give a standard deviation of 1.414. ArcGIS Zonal Statistics,
rasterstats and <code>GPURasterCellStatistics</code> divide by <code>n</code> (1 for the same cells); multiply a variance
by <code>(n - 1) / n</code> to convert.</p>
<h4 id="how-it-computes-2">How it computes</h4>
<p>{/* lens:GPURasterZonalStatistics */}</p>
<p>::lens{scenario=gpu-raster.raster-zonal-statistics rung=datum datum=0 caption="Zone 0, tree cover: valid cells are summed and divided by their count, with the lowest and highest kept."}
::lens{scenario=gpu-raster.raster-zonal-statistics rung=formula caption="The zone mean is the sum of valid values over their count; extremes are taken over the same cells."}
::lens{scenario=gpu-raster.raster-zonal-statistics rung=kernel caption="A mask pass flags in-zone valid cells, then sorted segment sums and a fused pass reduce each zone."}
::lens{scenario=gpu-raster.raster-zonal-statistics rung=graph caption="Canonicalize the band, mask, sort and sum by zone, then reduce counts and extremes."}</p>
<p>{/* parameters: GPURasterZonalStatistics generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h3 id="gpurasterzonalstatistics-parameters"><code>GPURasterZonalStatistics</code> parameters</h3>
<ul>
<li>Packer: <code>getGPURasterZonalStatisticsParameterValues(grid)</code>, writes <code>GPU_RASTER_ZONAL_STATISTICS_PARAMETER_LENGTH</code> (7) <code>float32</code> values.
<ul>
<li><code>grid</code> (<code>GPURasterMetadata</code>): The frame's grid; its width and height must match the contributor's.</li>
</ul>
</li>
</ul>
<p>{/* /parameters */}</p>
<h2 id="gpudistancefield"><code>GPUDistanceField</code></h2>
<p>Turns a seed mask into the exact Euclidean distance in metres from every cell to its nearest seed.</p>
<p>::lens{scenario=gpu-raster.distance-field rung=result caption="Distance to the nearest water body, from the WorldCover water cells of the Dixie window."}</p>
<p>Euclidean distance transform and nearest-seed allocation (Voronoi zones) on a raster grid, from
seed points snapped to cells, a labeled seed raster, or both. <code>distances</code> holds the grid distance
from each cell center to the nearest seed cell center (<code>+Infinity</code> when unreached), <code>allocation</code>
the nearest seed's ID, and <code>nearestCells</code> its cell; ties break on the smallest seed ID. A
per-frame <code>maximumDistance</code> clears cells beyond it and fills an optional <code>withinDistance</code> mask, and an
optional <code>r32float</code> texture receives a copy of <code>distances</code>. Cell sizes may differ in x and y.</p>
<p>Distances use grid units (world units of the affine), so degree grids yield degree distances, not
meters; ground-meter conversion is a non-goal of <code>GPUDistanceField</code>—use <code>GPUCostDistance</code> with
<code>cellSizeMode: 'geographic' | 'web-mercator'</code> and unit friction for ground-meter distances.</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> parameters</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUParameterBuffer</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(device, {id: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'df-parameters'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, format: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'float32'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, length: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">8</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">});</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUDistanceField</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  grid, parameters: parameters.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">importToGraph</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(graph),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  seedPositions, seedCount, seedMask,  </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// capacity is seedPositions.length</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  output: {distances, allocation, withinDistance}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">parameters.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">write</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">getGPUDistanceFieldParameterValues</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({maximumDistance: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">500</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}, grid));</span></span></code></pre>
<p><code>mode: 'exact'</code> (default) is the separable Felzenszwalb-Huttenlocher transform, banded as in the
Parallel Banding Algorithm (Cao et al. 2010): banded column scans find the nearest seed row, then
32-column band envelopes of those candidates merge pairwise in <code>ceil(log2(width / 32))</code> rounds and
a banded running maximum colors each row. Every pass runs about <code>width * height / 32</code> invocations
instead of one per row or column, and the result is bit-identical to the sequential envelope.
Candidates compare by exact integer squared offsets when the cell is square
and by f32 squared ground distance otherwise, so allocation matches a brute-force oracle exactly
on square cells. <code>'exact'</code> is exact on isotropic grids only: with unequal cell width and height the
f32 keys can misorder near-ties, and envelope crossings can be off by a cell. A correctly rounded square root keeps distances within 1 ULP of an f64 reference
(0 ULP when the cell size is a power of two). <code>mode: 'jump-flood'</code> is a cheaper preview for
dragging seeds: <code>ceil(log2(max(width, height)))</code> full-grid passes plus
<code>jumpFloodRefinementIterationCount</code> extra passes (exactly 0, 1, or 2; default 1). It always reports the distance to a real seed, so it can only overestimate; on random
256x256 scenes JFA+1 misallocated at most 0.009% of cells with errors up to 1.24 cells, and JFA+2
was exact in the measured scenes (not guaranteed).</p>
<p>A seed mask value <code>v</code> is the seed ID <code>v - 1</code>: write <code>1</code> for a plain mask or <code>label + 1</code> for
per-zone allocation. Seed point IDs default to the row index or come from <code>seedIds</code>. Rewriting
seeds, the count, the mask, or parameters never recompiles the graph. The grid origin and cell size
come from <code>grid</code> (axis-aligned, north-up or north-down), not from the settings.</p>
<p>For projected grids pass <code>localOrigin</code> (for example the raster corner) to
<code>getGPUDistanceFieldParameterValues</code> and upload <code>seedPositions</code> relative to it, so seeds near a
cell edge snap to the same cell as in float64. The grid origin stays absolute.</p>
<h4 id="how-it-computes-3">How it computes</h4>
<p>{/* lens:GPUDistanceField */}</p>
<p>::lens{scenario=gpu-raster.distance-field rung=datum datum=3228 caption="One cell: the squared offsets to the nearest seed, then the correctly rounded root."}
::lens{scenario=gpu-raster.distance-field rung=formula caption="Distance is the cell size times the root of the smallest squared cell offset to any seed."}
::lens{scenario=gpu-raster.distance-field rung=kernel caption="Banded column scans and row envelopes find the nearest seed, then one finalize pass scales it."}
::lens{scenario=gpu-raster.distance-field rung=graph caption="Seed rasterization, banded exact passes and the finalize node."}</p>
<p>{/* parameters: GPUDistanceField generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h3 id="gpudistancefield-parameters"><code>GPUDistanceField</code> parameters</h3>
<ul>
<li>Packer: <code>getGPUDistanceFieldParameterValues(settings, grid)</code>, writes <code>GPU_DISTANCE_FIELD_PARAMETER_LENGTH</code> (8) <code>float32</code> values.
<ul>
<li><code>grid</code> (<code>GPURasterMetadata</code>): The frame's grid; its width and height must match the contributor's.</li>
</ul>
</li>
</ul>
<p>Settings of <code>getGPUDistanceFieldParameterValues</code>:</p>
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
<td><code>maximumDistance</code> (optional)</td>
<td><code>number</code></td>
<td>Cells farther than this ground distance from every seed get distance <code>+Infinity</code>, allocation <code>GPU_DISTANCE_FIELD_NONE</code>, and a <code>0</code> in the within-distance mask. Defaults to <code>Infinity</code>.</td>
</tr>
<tr>
<td><code>localOrigin</code> (optional)</td>
<td><code>readonly [number, number]</code></td>
<td>Float64 point that seed positions are uploaded relative to. The packer subtracts it from the grid origin in float64, so seeds snap to cells from origin-relative float32 values. Defaults to <code>[0, 0]</code> (absolute seeds). Pass the raster or tile corner for projected grids: absolute float32 seeds and origins are quantized to 0.5 m at a UTM northing of 4.6e6 m, so seeds within that of a cell edge snap to the wrong cell; origin-relative ones keep about <code>1.2e-7</code> of their distance from the origin (1.2 cm at 100 km).</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h2 id="gpupolygonrasterization-and-gpurasterjoin"><code>GPUPolygonRasterization</code> and <code>GPURasterJoin</code></h2>
<p><code>GPUPolygonRasterization</code> scan-converts polygon features (the GeoArrow layout of
<code>GPUPointInPolygonJoin</code>: <code>positions</code>, <code>featureOffsets</code>, <code>polygonOffsets</code>, <code>ringOffsets</code>) into
a dense <code>uint32</code> zone raster on the GPU, plus optional boundary-cell flags. <code>GPURasterJoin</code> then
aggregates points by looking up their cell: O(1) per point, independent of polygon complexity.
The raster placement (origin, cell size, and row direction, north-up or north-down) comes from the
<code>grid</code>, packed by <code>getGPUPolygonRasterizationParameterValues(grid)</code>.</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> parameters</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUParameterBuffer</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(device, {id: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'raster-parameters'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, format: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'float32'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, length: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">4</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">});</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">parameters.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">write</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">getGPUPolygonRasterizationParameterValues</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(grid));</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> parametersView</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8"> parameters.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">importToGraph</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(graph);</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUPolygonRasterization</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  grid, parameters: parametersView,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  positions, featureOffsets, polygonOffsets, ringOffsets,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  crossingCapacity: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> &#x3C;&#x3C;</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> 20</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  output: {zones, boundary, status: {overflow, crossingCount}}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPURasterJoin</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  grid, parameters: parametersView, points, values, zones, boundary, zoneCount: featureCount,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  output: {counts, sums, boundaryCounts, unassignedBoundaryCount, outsideCount, pointBoundaryMask}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">parameters.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">write</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">getGPUPolygonRasterizationParameterValues</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(nextGrid)); </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// no recompile</span></span></code></pre>
<p><strong>Longitude/latitude polygons — rasterization.</strong> Pass <code>spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'none'}</code> for linear edges in degrees; omitted contexts remain planar.
The resolved context is exposed as <code>spatialContext</code>. Great-circle metrics and great-circle edges
are rejected. Prepare polygons crossing ±180 or enclosing a pole with
<code>GPULonLatPolygonPreparation</code>. Rasterization uses longitude copies shifted by -360, 0, and +360
degrees, selected on the GPU from the per-frame grid extent, so a split Fiji polygon covers both
sides of a 170..190-degree grid. Crossings outside the grid remain clamped to its border to preserve
fill parity. Grids inside [-180, 180] retain their existing output. A grid straddling ±180 can count
each edge twice; size <code>crossingCapacity</code> accordingly and inspect <code>status.crossingCount</code>.
Empty rings contribute no crossings.</p>
<p><strong>Longitude/latitude polygons — raster join.</strong> <code>GPURasterJoin</code> accepts and exposes the same optional
<code>spatialContext</code>, with linear edges and metric <code>'none'</code>. Give the rasterization and join the same
context. The join consumes caller-provided <code>zones</code> and <code>boundary</code>; it does not rasterize polygons
itself or shift point coordinates. Points must use the grid's longitude branch. Prepare crossing
or pole rings with <code>GPULonLatPolygonPreparation</code> before producing the zone raster; rasterization's
±180 seam handling supplies the shifted polygon copies.</p>
<h4 id="gpupolygonrasterization"><code>GPUPolygonRasterization</code></h4>
<p>Scan-converts polygon rings into a zone raster whose cells hold the feature covering their centre.</p>
<p>::lens{scenario=gpu-raster.polygon-rasterization rung=result caption="The Dixie perimeter on the 120 m grid; about 89 percent of cell centres fall inside."}</p>
<p>{/* parameters: GPUPolygonRasterization generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h5 id="gpupolygonrasterization-parameters"><code>GPUPolygonRasterization</code> parameters</h5>
<ul>
<li>Packer: <code>getGPUPolygonRasterizationParameterValues(grid)</code>, writes <code>GPU_POLYGON_RASTERIZATION_PARAMETER_LENGTH</code> (4) <code>float32</code> values.
<ul>
<li><code>grid</code> (<code>GPURasterMetadata</code>): The frame's grid: axis-aligned; its width and height must match the contributor's.</li>
</ul>
</li>
</ul>
<p>{/* /parameters */}</p>
<h4 id="how-it-computes-4">How it computes</h4>
<p>{/* lens:GPUPolygonRasterization */}</p>
<p>::lens{scenario=gpu-raster.polygon-rasterization rung=datum datum=3360 caption="A cell on the perimeter edge: its row centre line meets the edge crossings that pair into a span."}
::lens{scenario=gpu-raster.polygon-rasterization rung=kernel caption="One thread per vertex counts the rows its edge crosses, then spans are filled with an atomic minimum."}
::lens{scenario=gpu-raster.polygon-rasterization rung=graph caption="Edge counts, scan, crossing records, radix sort, ranking and span fill, plus boundary flags."}</p>
<h4 id="gpurasterjoin"><code>GPURasterJoin</code></h4>
<p>Assigns each point the zone stored in its raster cell, counting points per zone in O(1) per point.</p>
<p>::lens{scenario=gpu-raster.raster-join rung=result caption="Burned cell centres coloured by zone; the few outside the perimeter take no zone."}</p>
<p>{/* parameters: GPURasterJoin generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h5 id="gpurasterjoin-parameters"><code>GPURasterJoin</code> parameters</h5>
<ul>
<li>Packer: <code>getGPUPolygonRasterizationParameterValues(grid)</code>, writes <code>GPU_POLYGON_RASTERIZATION_PARAMETER_LENGTH</code> (4) <code>float32</code> values.
<ul>
<li>Input port: <code>parameters</code>.</li>
<li><code>grid</code> (<code>GPURasterMetadata</code>): The frame's grid: axis-aligned; its width and height must match the contributor's.</li>
</ul>
</li>
</ul>
<p>{/* /parameters */}</p>
<h4 id="how-it-computes-5">How it computes</h4>
<p>{/* lens:GPURasterJoin */}</p>
<p>::lens{scenario=gpu-raster.raster-join rung=datum datum=112 caption="One burned point near the perimeter: its cell index, the zone stored there and the boundary status."}
::lens{scenario=gpu-raster.raster-join rung=formula caption="The point takes the zone of the cell that contains it, found by flooring the offset from the origin."}
::lens{scenario=gpu-raster.raster-join rung=kernel caption="One thread per point floors its position into a cell and reads the zone and boundary words."}
::lens{scenario=gpu-raster.raster-join rung=graph caption="Rasterization fills the zone raster, then the join looks up points and tallies counts."}</p>
<h2 id="gpurasterreclassify-gpuweightedoverlay-and-local-raster-operations"><code>GPURasterReclassify</code>, <code>GPUWeightedOverlay</code>, and local raster operations</h2>
<p>Per-cell map algebra over float32 rasters or plain columns, with every table, weight, threshold,
and operation code in a per-frame parameter view, so sliders never recompile. Rasters are packed
row-major float32; NaN is always nodata, and an optional finite <code>noDataValue</code> sentinel is
compared exactly. The sentinel marks raster cells only: a per-frame constant operand that equals it is a
value, so <code>GPURasterConditional</code> with <code>noDataValue: 0</code> and <code>constantB: 0</code> writes 0 where the condition
fails, and <code>GPURasterArithmetic</code> <code>maximum(a, 0)</code> with <code>noDataValue: 0</code> is defined wherever <code>a</code> is. Stacks are band-sequential: layer <code>i</code> occupies rows <code>[i * cellCount, (i + 1) * cellCount)</code>.</p>
<p><code>GPURasterReclassify.values</code> and <code>GPURasterConditional.conditionValues</code>, <code>a</code>, and <code>b</code> also accept
published <code>GPURasterBand</code> inputs. Supply <code>grid</code> when any input is a band. Buffer bands may be
float32, uint32 or sint32; texture bands use the published scalar/channel storage contract.
The published raster conversion resolves native no-data and band validity before applying scale
and offset. Non-finite band samples are missing. Legacy float views keep the NaN/sentinel rules
above, including infinity handling. The contributor-level <code>noDataValue</code> applies only to legacy
views; each band owns its native sentinel. In a conditional, only the chosen operand must be valid,
and an invalid condition always produces NaN. An additional reclassify <code>mask</code> still filters cells.
Band storage and validity may have padding after the grid; only the grid prefix is read. Reclassify
uses the grid cell count by default for a band and can classify a smaller <code>rowCount</code> prefix.
Band metadata and grid dimensions are compile-time; input contents and masks can change each frame.</p>
<p><code>GPURasterCellStatistics.stack</code> and <code>GPUWeightedOverlay.stack</code> also accept an ordered array of
published bands: <code>stack: [elevationBand, moistureBand, ...]</code>. Supply exactly <code>layerCount</code> bands and
<code>grid</code> with exactly <code>cellCount</code> cells; every band must already occupy that complete grid. Buffer
and texture storage and native scalar formats can be mixed. Each band's validity, native nodata,
scale and offset are resolved independently. The top-level <code>noDataValue</code> applies only to the legacy
packed stack, so a calibrated band value equal to it stays valid. The existing <code>mask</code>, nodata
policy, minimum valid count, frequency tie rules and remapping operate on calibrated observations.</p>
<p>Band arrays are converted and packed on the GPU into one float32 stack, so the analytical kernels
keep one input binding even at 64 statistics layers or 16 overlay layers. The extra stack occupies
<code>4 * cellCount * layerCount</code> bytes, plus published band-conversion scratch. Its binding must fit
the device's storage limit; reduce the grid or layer count if construction of the command nodes
rejects it. <code>estimatedCommandNodeCount</code> bounds conversion, packing and analysis nodes and is
checked at construction. Compiled graphs own the scratch; these contributors need no <code>destroy()</code>.
Legacy packed stacks retain their existing path and results.</p>
<p>Band metadata is validated at construction. Calibration and non-NaN float sentinels must round to
finite float32; integer sentinels must be integers within the native uint32 or sint32 range.
Textures must select one 2D mip, one array layer and one sample, preserve the declared band scalar
format, and select an existing channel. Validity must belong to the same graph as the band storage.</p>
<ul>
<li><code>GPURasterReclassify</code> maps values through <code>n</code> ascending per-frame breaks: the class is the number
of breaks <code>&#x3C;= value</code> (left-closed <code>[b[k - 1], b[k])</code>, default) or <code>&#x3C; value</code> (right-closed), found
by binary search. Outputs: <code>classes</code> (<code>0xffffffff</code> for nodata), <code>reclassified</code>
(<code>classValues[class]</code>), and deterministic <code>classCounts</code> (integer atomics).</li>
<li><code>GPUWeightedOverlay</code> (suitability) scores up to 16 layers: each value is remapped linearly from
<code>[inputMinimum, inputMaximum]</code> to <code>[0, 1]</code> (clamped, optionally inverted) or through a per-layer break
table, then <code>score = sum(weight * remapped)</code> in fixed layer order. A NaN table value marks a
restricted class (ArcGIS "Restricted"). Options: normalize by the sum of absolute weights,
nodata <code>'propagate'</code> or <code>'ignore'</code>. <code>scoreRange</code> is the exact min/max of the defined scores
(order-preserving integer atomics) and <code>validity</code> flags the defined scores. An optional <code>mask</code>
input (nonzero keeps the cell) marks whole cells nodata.</li>
<li><code>GPURasterCellStatistics</code> (ArcGIS "Cell Statistics") reduces up to 64 layers per cell:
minimum, maximum, range, sum, mean, population standard deviation (divisor <code>n</code>, centred second
pass; <code>GPURasterZonalStatistics</code> divides by <code>n - 1</code>),
majority, minority (ties pick the smallest value), variety, and valid <code>counts</code>. Frequencies are an
<code>O(layerCount^2)</code> exact-equality scan per cell. An optional <code>mask</code> input (nonzero keeps the cell)
marks the whole cell nodata: float outputs NaN, <code>variety</code> and <code>counts</code> 0.</li>
<li><code>GPURasterConditional</code> is <code>where(condition, a, b)</code>: a <code>uint32</code> mask, or a float raster compared
per frame (<code>&#x3C;</code>, <code>&#x3C;=</code>, <code>></code>, <code>>=</code>, <code>==</code>, <code>!=</code>, inclusive <code>between</code>); <code>a</code> and <code>b</code> are rasters or
per-frame constants. The optional condition result is <code>output.conditionResult</code> (<code>1</code> where the condition holds, else
<code>0</code>; it is a condition flag, not a no-data flag).</li>
<li><code>GPURasterArithmetic</code> evaluates <code>op(a * scaleA + offsetA, b * scaleB + offsetB)</code> with a per-frame
operation: add, subtract, multiply, divide, minimum, maximum, power, absolute difference,
normalized difference (NDVI), or a unary absolute/square root/log/exp/floor/ceil/round
(half to even) on <code>a</code>, then an optional clamp (<code>clampMinimum</code>, <code>clampMaximum</code>). Domain errors give NaN.
An optional <code>mask</code> input (nonzero keeps the cell) marks a cell nodata: the result is NaN.</li>
</ul>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> overlayParameters</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUParameterBuffer</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(device, {</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  id: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'overlay'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, format: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'float32'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, length: </span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">getGPUWeightedOverlayParameterLength</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">3</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">});</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUWeightedOverlay</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  stack, layerCount: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">3</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, cellCount: width </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">*</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8"> height,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  parameters: overlayParameters.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">importToGraph</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(graph),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  remapBreaks, remapValues, maximumBreakCount: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">4</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// optional per-layer tables</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  output: {score, scoreRange}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">overlayParameters.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">write</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">getGPUWeightedOverlayParameterValues</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  layers: [</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    {weight: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">0.5</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, inputMinimum: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, inputMaximum: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">30</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, invert: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">true</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// slope: flatter is better</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    {weight: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">0.3</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, mode: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'table'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, breakCount: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">4</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">},            </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// land cover classes</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    {weight: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">0.2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, inputMinimum: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, inputMaximum: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">5000</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}            </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// distance to roads</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  ],</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  normalizeWeights: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">true</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">})); </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// no recompile</span></span></code></pre>
<p>Every contributor is one invocation per cell reading inputs in fixed order, so results are
deterministic per device. Backends may contract <code>a * b + c</code> into an FMA (Metal does), so scores
and scaled operands can differ from an unfused CPU evaluation by a few roundings; reclassify,
conditional, extremes, sums, frequencies, and counts match a CPU oracle bit for bit.</p>
<h4 id="gpurasterarithmetic"><code>GPURasterArithmetic</code></h4>
<p>Operands <code>a</code> and <code>b</code> accept published buffer or texture raster bands alongside legacy float32
views. Supply <code>grid</code> for a band; validity and the native sentinel are applied before band calibration.
The operation's per-frame scale and offset then apply to those calibrated values. The top-level
<code>noDataValue</code> applies only to legacy operands. An absent <code>b</code> still uses <code>constantB</code>, including when
that constant equals the legacy sentinel. All band metadata is compile-time; buffer contents and
validity are borrowed and reread on every encoding.</p>
<p>Scales two rasters and combines them per cell; here near-infrared reflectance after the fire minus before it.</p>
<p>::lens{scenario=gpu-raster.raster-arithmetic rung=result caption="Change in near-infrared reflectance; the burn scar is the strongest decrease."}</p>
<p>{/* parameters: GPURasterArithmetic generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h5 id="gpurasterarithmetic-parameters"><code>GPURasterArithmetic</code> parameters</h5>
<ul>
<li>Packer: <code>getGPURasterArithmeticParameterValues(settings)</code>, writes <code>GPU_RASTER_ARITHMETIC_PARAMETER_LENGTH</code> (8) <code>float32</code> values.</li>
</ul>
<p>Settings of <code>getGPURasterArithmeticParameterValues</code>:</p>
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
<td><code>operation</code></td>
<td><code>GPURasterBinaryOperation | GPURasterUnaryOperation</code></td>
<td>Operation applied to the scaled operands.</td>
</tr>
<tr>
<td><code>scaleA</code> (optional)</td>
<td><code>number</code></td>
<td><code>a' = a * scaleA + offsetA</code>. Defaults to 1.</td>
</tr>
<tr>
<td><code>offsetA</code> (optional)</td>
<td><code>number</code></td>
<td>Defaults to 0.</td>
</tr>
<tr>
<td><code>scaleB</code> (optional)</td>
<td><code>number</code></td>
<td><code>b' = b * scaleB + offsetB</code>. Defaults to 1.</td>
</tr>
<tr>
<td><code>offsetB</code> (optional)</td>
<td><code>number</code></td>
<td>Defaults to 0.</td>
</tr>
<tr>
<td><code>constantB</code> (optional)</td>
<td><code>number</code></td>
<td><code>b</code> when no <code>b</code> view is bound. Defaults to 0.</td>
</tr>
<tr>
<td><code>clampMinimum</code> (optional)</td>
<td><code>number</code></td>
<td>Optional output clamp lower bound. Defaults to <code>-Infinity</code>.</td>
</tr>
<tr>
<td><code>clampMaximum</code> (optional)</td>
<td><code>number</code></td>
<td>Optional output clamp upper bound. Defaults to <code>+Infinity</code>.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h4 id="how-it-computes-6">How it computes</h4>
<p>{/* lens:GPURasterArithmetic */}</p>
<p>::lens{scenario=gpu-raster.raster-arithmetic rung=datum datum=2400 caption="One burn-scar cell: raw digital numbers are checked for no data, scaled, then subtracted."}
::lens{scenario=gpu-raster.raster-arithmetic rung=formula caption="Check the raw values for no data, scale each, subtract, and keep NaN for missing input."}
::lens{scenario=gpu-raster.raster-arithmetic rung=kernel caption="One invocation per cell evaluates the operation chosen in the parameters."}
::lens{scenario=gpu-raster.raster-arithmetic rung=graph caption="A single per-cell node reads both dates and writes the difference."}</p>
<h4 id="gpurastercellstatistics"><code>GPURasterCellStatistics</code></h4>
<p>Reduces a stack of aligned rasters to one statistic per cell; here the mean of near-infrared before and after.</p>
<p>::lens{scenario=gpu-raster.raster-cell-statistics rung=result caption="Mean near-infrared reflectance over both dates; cells missing either date are NaN."}</p>
<p>{/* parameters: GPURasterCellStatistics generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h5 id="gpurastercellstatistics-parameters"><code>GPURasterCellStatistics</code> parameters</h5>
<ul>
<li>Packer: <code>getGPURasterCellStatisticsParameterValues(settings?)</code>, writes <code>GPU_RASTER_CELL_STATISTICS_PARAMETER_LENGTH</code> (4) <code>float32</code> values.</li>
</ul>
<p>Settings of <code>getGPURasterCellStatisticsParameterValues</code>:</p>
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
<td><code>noDataPolicy</code> (optional)</td>
<td><code>'ignore' | 'propagate'</code></td>
<td><code>'ignore'</code> (default, ArcGIS "DATA"): statistics use the valid layers of each cell. <code>'propagate'</code>: any nodata layer makes the cell nodata.</td>
</tr>
<tr>
<td><code>minimumValidCount</code> (optional)</td>
<td><code>number</code></td>
<td>Cells with fewer valid layers are nodata. Defaults to 1.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h4 id="how-it-computes-7">How it computes</h4>
<p>{/* lens:GPURasterCellStatistics */}</p>
<p>::lens{scenario=gpu-raster.raster-cell-statistics rung=datum datum=2400 caption="One cell: both layers are checked for validity, summed in order and divided by the count."}
::lens{scenario=gpu-raster.raster-cell-statistics rung=formula caption="Mean of the valid layers, written only when enough layers are valid."}
::lens{scenario=gpu-raster.raster-cell-statistics rung=kernel caption="Each invocation sweeps its layers in fixed order and counts the valid ones."}
::lens{scenario=gpu-raster.raster-cell-statistics rung=graph caption="One moments node reads the stack and writes the mean."}</p>
<h4 id="gpurasterconditional"><code>GPURasterConditional</code></h4>
<p>Chooses one of two values per cell by comparing a raster with a threshold; here friction from built-up land.</p>
<p>::lens{scenario=gpu-raster.raster-conditional rung=result caption="Friction by land cover: 0.4 where WorldCover says built-up, 1 elsewhere."}</p>
<p>{/* parameters: GPURasterConditional generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h5 id="gpurasterconditional-parameters"><code>GPURasterConditional</code> parameters</h5>
<ul>
<li>Packer: <code>getGPURasterConditionalParameterValues(settings?)</code>, writes <code>GPU_RASTER_CONDITIONAL_PARAMETER_LENGTH</code> (8) <code>float32</code> values.</li>
</ul>
<p>Settings of <code>getGPURasterConditionalParameterValues</code>:</p>
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
<td><code>comparison</code> (optional)</td>
<td><code>GPURasterComparison</code></td>
<td>Comparison of <code>conditionValues</code> against <code>threshold</code>. Ignored with a <code>mask</code>. Defaults to <code>'>'</code>.</td>
</tr>
<tr>
<td><code>threshold</code> (optional)</td>
<td><code>number</code></td>
<td>Threshold, or the inclusive lower bound for <code>'between'</code>. Defaults to 0.</td>
</tr>
<tr>
<td><code>upperThreshold</code> (optional)</td>
<td><code>number</code></td>
<td>Inclusive upper bound for <code>'between'</code>. Defaults to <code>threshold</code>.</td>
</tr>
<tr>
<td><code>constantA</code> (optional)</td>
<td><code>number</code></td>
<td>Value used where the condition holds when no <code>a</code> view is bound. Defaults to 1.</td>
</tr>
<tr>
<td><code>constantB</code> (optional)</td>
<td><code>number</code></td>
<td>Value used where the condition fails when no <code>b</code> view is bound. Defaults to 0.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h4 id="how-it-computes-8">How it computes</h4>
<p>{/* lens:GPURasterConditional */}</p>
<p>::lens{scenario=gpu-raster.raster-conditional rung=datum datum=1893 caption="Greenville cell: the comparison is true, so the first value is chosen."}
::lens{scenario=gpu-raster.raster-conditional rung=formula caption="Compare the land cover with the threshold and take the branch it selects."}
::lens{scenario=gpu-raster.raster-conditional rung=kernel caption="One invocation compares and selects; the unchosen branch is never read for no data."}
::lens{scenario=gpu-raster.raster-conditional rung=graph caption="A single select node reads the land cover and writes the friction."}</p>
<h4 id="gpurasterreclassify"><code>GPURasterReclassify</code></h4>
<p>Maps raster values through ascending breaks to class values; here land cover codes to suitability ranks.</p>
<p>::lens{scenario=gpu-raster.raster-reclassify rung=result caption="Suitability rank per land cover class, from 1 for water to 5 for cropland."}</p>
<p>{/* parameters: GPURasterReclassify generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h5 id="gpurasterreclassify-parameters"><code>GPURasterReclassify</code> parameters</h5>
<ul>
<li>Packer: <code>getGPURasterReclassifyParameterValues(settings)</code>, writes <code>GPU_RASTER_RECLASSIFY_PARAMETER_LENGTH</code> (4) <code>float32</code> values.</li>
</ul>
<p>Settings of <code>getGPURasterReclassifyParameterValues</code>:</p>
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
<td><code>breakCount</code></td>
<td><code>number</code></td>
<td>Number of active ascending breaks, at most the contributor's <code>maximumBreakCount</code>.</td>
</tr>
<tr>
<td><code>closed</code> (optional)</td>
<td><code>'left' | 'right'</code></td>
<td>Which side of each interval is closed. <code>'left'</code> (default) gives intervals <code>[b[k - 1], b[k])</code>, so the class is the number of breaks <code>&#x3C;= value</code>; <code>'right'</code> gives <code>(b[k - 1], b[k]]</code>, so the class is the number of breaks <code>&#x3C; value</code>.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h4 id="how-it-computes-9">How it computes</h4>
<p>{/* lens:GPURasterReclassify */}</p>
<p>::lens{scenario=gpu-raster.raster-reclassify rung=datum datum=1526 caption="One cropland cell: count the breaks at or below its code, then look up the rank."}
::lens{scenario=gpu-raster.raster-reclassify rung=formula caption="The interval index is the number of breaks at or below the value; the rank is its table entry."}
::lens{scenario=gpu-raster.raster-reclassify rung=kernel caption="A binary search over the breaks classifies each cell in one invocation."}
::lens{scenario=gpu-raster.raster-reclassify rung=graph caption="A single classify node reads the codes, breaks and class values."}</p>
<h4 id="gpuweightedoverlay"><code>GPUWeightedOverlay</code></h4>
<p>Scores each cell as a weighted sum of remapped layers; here land cover and elevation suitability.</p>
<p>::lens{scenario=gpu-raster.weighted-overlay rung=result caption="Suitability from land cover (weight 0.7) and elevation (weight 0.3), on a 0–1 scale."}</p>
<p>{/* parameters: GPUWeightedOverlay generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h5 id="gpuweightedoverlay-parameters"><code>GPUWeightedOverlay</code> parameters</h5>
<ul>
<li>Packer: <code>getGPUWeightedOverlayParameterValues(settings, options?)</code>, writes <code>getGPUWeightedOverlayParameterLength(...)</code> <code>float32</code> values.
<ul>
<li><code>options</code> (<code>{hasTables?: boolean}</code>): <code>hasTables: false</code> declares that the overlay was built without remap tables, so a <code>'table'</code> layer throws here instead of scoring NaN (nodata) on every cell. Defaults to true.</li>
</ul>
</li>
</ul>
<p>Settings of <code>getGPUWeightedOverlayParameterValues</code>:</p>
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
<td><code>layers</code></td>
<td><code>readonly GPUWeightedOverlayLayerSettings[]</code></td>
<td>One record per input layer, in stack order.</td>
</tr>
<tr>
<td><code>normalizeWeights</code> (optional)</td>
<td><code>boolean</code></td>
<td>Divide the weighted sum by the sum of absolute weights of the contributing layers.</td>
</tr>
<tr>
<td><code>noDataPolicy</code> (optional)</td>
<td><code>'propagate' | 'ignore'</code></td>
<td><code>'propagate'</code> (default): a nodata input makes the cell nodata. <code>'ignore'</code>: nodata layers are skipped (and excluded from weight normalization); a cell with no valid layer is nodata.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h4 id="how-it-computes-10">How it computes</h4>
<p>{/* lens:GPUWeightedOverlay */}</p>
<p>::lens{scenario=gpu-raster.weighted-overlay rung=datum datum=1893 caption="Greenville cell: each layer is remapped through its table, then weighted and summed."}
::lens{scenario=gpu-raster.weighted-overlay rung=formula caption="Each layer is remapped by its break table; the score is the weighted sum."}
::lens{scenario=gpu-raster.weighted-overlay rung=kernel caption="One invocation remaps every layer and accumulates the weighted sum in layer order."}
::lens{scenario=gpu-raster.weighted-overlay rung=graph caption="A single score node reads the stack, break tables and weights."}</p>
<h2 id="gpurasterstretch"><code>GPURasterStretch</code></h2>
<p>Optional output.validCount carries the exact uint32 valid population. The legacy float32 statistics
row remains rounded above 2^24.</p>
<p>Rescales a raster to 0–1 between its smallest and largest valid values, with optional percentile, equalize, gamma and sigmoid.</p>
<p>::lens{scenario=gpu-raster.raster-stretch rung=result caption="Near-infrared reflectance after the fire, stretched between its smallest and largest valid values."}</p>
<p>Contrast stretches and colormap lookups for a float32 raster, computed on the GPU every frame with
no readback: a linear min/max stretch, a percentile stretch (for example 2%/98%), and histogram
equalization, with optional display gamma (<code>t^(1/gamma)</code>, so <code>gamma > 1</code> brightens, as in rio-color
and GIMP levels) and sigmoidal contrast (normalized so 0 and 1 are fixed). GDAL has no gamma; its
<code>-exponent e</code> (<code>gdal_translate</code>, VRT) is <code>t^e</code>, so it equals <code>gamma: 1 / e</code>. Outputs (any subset): per-cell <code>stretched</code> values in <code>[0, 1]</code>, packed rgba8 <code>colors</code>
(<code>r | g &#x3C;&#x3C; 8 | b &#x3C;&#x3C; 16 | a &#x3C;&#x3C; 24</code>) through a per-frame <code>palette</code> (nearest or linear), a <code>lut</code> of
<code>lutSize</code> normalized values with optional <code>lutColors</code>, the <code>histogram</code>, and an 8-row <code>statistics</code>
row <code>[domainMinimum, domainMaximum, lo, hi, validCount, binWidth, 0, 0]</code> (<code>GPU_RASTER_STRETCH_STATISTICS_INDEX</code>).</p>
<p><code>values</code> accepts a published <code>GPURasterBand</code> as well as a legacy float view. Band validity,
native no-data and calibration are resolved through the published raster primitives before both
statistics and application. Band samples must be finite; legacy infinities retain their documented
clamping behavior. <code>noDataValue</code> applies only to legacy views. <code>mask</code> remains an additional cell
filter, while <code>regionMask</code> filters statistics only. Only the grid prefix of padded band views is
used; band metadata is compile-time and band contents/validity are per-frame. Construction validates
the same band metadata limits as reclassify and conditional above.</p>
<p>Statistics (exact min/max, a <code>binCount</code> histogram, its CDF) come from the valid, finite cells inside a
per-frame window <code>[column0, row0, column1, row1)</code> and an optional per-frame <code>regionMask</code>; the apply
step always covers every cell. A viewport mask (for example from <code>GPURegionMask</code>) gives QGIS-style
"stretch to visible extent" per frame. The domain is <code>'auto'</code> (exact min/max) or an explicit
<code>[min, max]</code>. Percentile bounds use the rank rule of the published <code>GPURasterGlobalPercentile</code>: the bin
that holds order statistic <code>floor((N - 1) * p)</code> of the <code>N</code> counted cells, reported at its centre, so they
are within half a <code>binWidth</code> of that order statistic; with an automatic domain <code>p = 0</code> and <code>p = 1</code> are the
exact minimum and maximum. (Ten values 0, 10, ..., 90 at <code>p = 0.98</code> give 80.02; NumPy's default linear
rule gives 88.2.) Equalization uses the rule of the published <code>GPURasterContrast</code> and OpenCV
<code>equalizeHist</code>, <code>(cdf[bin] - cdfFirst) / (total - cdfFirst)</code> with <code>cdfFirst</code> the cumulative count of the
first occupied bin: it depends only on ranks, so the smallest value maps to 0, the largest to 1 and three
equally frequent values map to 0, 0.5 and 1. NaN, the applicable no-data sentinel, and cells
rejected by band validity or <code>mask</code> are nodata (<code>stretched</code> NaN, <code>colors</code> 0). A constant raster (<code>lo == hi</code>) maps values below to 0,
equal to 0.5, above to 1; with an automatic domain and no included cell, <code>statistics</code> is NaN and
<code>validCount</code> 0.</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> stretch</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPURasterStretch</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  values, grid, noDataValue: </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">-</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">9999</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  binCount: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">1024</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, lutSize: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">256</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// topology</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  parameters: stretchParameters.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">importToGraph</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(graph),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  regionMask: viewportMask, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// optional, contents per frame</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  palette, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// uint32 rgba8, contents per frame</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  output: {colors, lut, histogram, statistics}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">});</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(stretch);</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">stretchParameters.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">write</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">getGPURasterStretchParameterValues</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  mode: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'percentile'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, percentiles: [</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">98</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">], gamma: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">0.8</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, sigmoidContrast: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">4</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, paletteInterpolation: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'linear'</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">})); </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// no recompile</span></span></code></pre>
<p>Pipeline: order-preserving integer min/max keys, integer-atomic histogram, <code>GPUScan</code> CDF, finalize,
lookup table, apply; every step is deterministic. Mode, domain, percentiles, window, gamma,
sigmoid, palette interpolation, mask, and palette contents never recompile.</p>
<h4 id="how-it-computes-11">How it computes</h4>
<p>{/* lens:GPURasterStretch */}</p>
<p>::lens{scenario=gpu-raster.raster-stretch rung=datum datum=1791 caption="The brightest cell: its value sits at the top of the domain and maps to 1."}
::lens{scenario=gpu-raster.raster-stretch rung=formula caption="Subtract the lower bound, divide by the domain width and clamp to 0–1."}
::lens{scenario=gpu-raster.raster-stretch rung=kernel caption="The apply pass maps every cell through the bounds the statistics passes found."}
::lens{scenario=gpu-raster.raster-stretch rung=graph caption="Extent, histogram, scan, finalize and apply nodes run in order."}</p>
<p>{/* parameters: GPURasterStretch generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h3 id="gpurasterstretch-parameters"><code>GPURasterStretch</code> parameters</h3>
<ul>
<li>Packer: <code>getGPURasterStretchParameterValues(settings?)</code>, writes <code>GPU_RASTER_STRETCH_PARAMETER_LENGTH</code> (16) <code>float32</code> values.</li>
</ul>
<p>Settings of <code>getGPURasterStretchParameterValues</code>:</p>
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
<td><code>window</code> (optional)</td>
<td><code>readonly [number, number, number, number]</code></td>
<td>Statistics window <code>[column0, row0, column1, row1)</code> in cells; clamped to the raster. Defaults to the whole raster. Only statistics (min/max, histogram) use it: the apply step runs on all cells.</td>
</tr>
<tr>
<td><code>domain</code> (optional)</td>
<td><code>'auto' | readonly [number, number]</code></td>
<td><code>'auto'</code> (default) takes the exact minimum and maximum of the included valid cells, or an explicit <code>[min, max]</code> with <code>min &#x3C;= max</code>. Cells outside an explicit domain are not counted in the histogram and clamp to 0 or 1 when applied.</td>
</tr>
<tr>
<td><code>mode</code> (optional)</td>
<td><code>GPURasterStretchMode</code></td>
<td><code>'linear'</code> (default) maps the domain to <code>[0, 1]</code>. <code>'percentile'</code> maps the <code>percentiles</code> of the histogram, each taken as the centre of the bin that holds order statistic <code>floor((N - 1) * p)</code> (the rule of the published <code>GPURasterGlobalPercentile</code>). <code>'equalize'</code> maps through the histogram CDF as <code>(cdf[bin] - cdfFirst) / (total - cdfFirst)</code> (the rule of the published <code>GPURasterContrast</code>).</td>
</tr>
<tr>
<td><code>percentiles</code> (optional)</td>
<td><code>readonly [number, number]</code></td>
<td>Percentile mode: <code>[low, high]</code> in percent, <code>0 &#x3C;= low &#x3C;= high &#x3C;= 100</code>. Defaults to <code>[2, 98]</code>.</td>
</tr>
<tr>
<td><code>gamma</code> (optional)</td>
<td><code>number</code></td>
<td>Display gamma: applies <code>t' = t ** (1 / gamma)</code> after the mode, as rio-color and GIMP levels do, so <code>gamma > 1</code> brightens and <code>gamma &#x3C; 1</code> darkens. Finite and positive, defaults to 1. GDAL has no gamma; its <code>-exponent e</code> (<code>gdal_translate</code>, VRT) is <code>t ** e</code>, so it equals <code>gamma: 1 / e</code>.</td>
</tr>
<tr>
<td><code>sigmoidContrast</code> (optional)</td>
<td><code>number</code></td>
<td>Sigmoidal contrast (rio-color / ImageMagick style) applied after gamma, normalised so 0 maps to 0 and 1 to 1. Zero or undefined disables it; otherwise <code>(0, 80]</code>.</td>
</tr>
<tr>
<td><code>sigmoidMidpoint</code> (optional)</td>
<td><code>number</code></td>
<td>Sigmoid midpoint in <code>(0, 1)</code>. Defaults to 0.5.</td>
</tr>
<tr>
<td><code>paletteInterpolation</code> (optional)</td>
<td><code>'nearest' | 'linear'</code></td>
<td>Palette lookup: <code>'nearest'</code> (default) or <code>'linear'</code> channel interpolation.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h2 id="gpuisolines-and-gpuisobands"><code>GPUIsolines</code> and <code>GPUIsobands</code></h2>
<p>For cell ownership, shared sample anchors, halo layout and cross-tile edge IDs, use the
<a href="/luma-contributors/docs/api/gpu-raster/isolines-tiles">isoline tiling guide</a>, covering the contributor and GPU/CPU operation forms.</p>
<p>Isoband triangle and boundary-edge scans saturate their totals at <code>0xffffffff</code>. Offsets stay
monotonic even when the requirement exceeds uint32, so capacity overflow keeps the output prefix
and sets the existing <code>overflow</code> or <code>edgeOverflow</code> status.</p>
<p>Marching-squares contours of any float32 raster (terrain, density, interpolation output) with
per-frame levels. Both contributors share one definition, so filled bands and lines drawn together
coincide bit for bit on a device. Samples sit at cell centres of the per-frame world extent (row 0
at <code>minY</code>). A cell with a nodata corner (NaN or an infinity, optional <code>noDataValue</code>, optional
<code>validity</code>) emits nothing; infinities are nodata because a crossing cannot interpolate them. The
extent must be finite with <code>max > min</code> on both axes (a y-down or zero-size extent throws). Corners with
<code>value >= level</code> are high; saddles are resolved by the bilinear (asymptotic) decider of the published
<code>GPURasterContours</code>: the two high corners are joined when the saddle value of the bilinear surface
through the four corners, <code>(v0 * v2 - v1 * v3) / (v0 + v2 - v1 - v3)</code>, is at or above the level
(a tie joins). Before 2026-11-10 the rule was the corner mean <code>((v0 + v1) + (v2 + v3)) / 4</code>; the two
differ only on saddles whose corners are far from symmetric, such as <code>10, 0, 1.1, 0</code> at level 1
(mean 2.775 joined them, saddle value 0.991 separates them). Crossings are computed once per
canonical edge (left to right, bottom to top), so neighbouring cells produce identical vertices.</p>
<p><code>GPUIsolines.values</code> and <code>GPUIsobands.values</code> also accept a published <code>GPURasterBand</code> on <code>grid</code>.
Buffer bands support float32, uint32 and sint32; texture bands use the published scalar/channel
storage contract. Conversion honors native nodata and nonzero band validity before applying scale
and offset. Non-finite samples become missing. The contributor-level <code>noDataValue</code> applies only to
legacy float views, so a calibrated band value equal to that sentinel remains valid. An additional
<code>mask</code> still filters samples, and any missing corner excludes its marching cell from all geometry.
Classes, segments, stitched polylines, triangles and boundary edges use the same calibrated values.</p>
<p>Band metadata and grid dimensions are compile-time; sample values, band validity and masks can
change between encodings. Buffer storage and validity may have byte offsets and padding after the
grid; only the grid prefix is converted. Canonicalization allocates graph-owned float32 values and
uint32 validity, each <code>4 * grid.width * grid.height</code> bytes, and checks the device binding limit before
allocation. For tiled isolines the band covers the local grid including its halo, while parameters
still describe the shared domain. Contour coordinates continue to use the explicit parameter
extent and the row direction documented below.</p>
<p><strong>Isoline tiles.</strong> Optional compile-time <code>GPUIsolines.tile</code> partitions the domain into owned
marching cells with halo samples. The <a href="/luma-contributors/docs/api/gpu-raster/isolines-tiles">tiling guide</a> is the canonical contract
for ownership, shared placement parameters, per-tile status, fragment joining and composition
with external vector boundaries. The existing Dixie Fire lens uses one resident grid without <code>tile</code>.</p>
<p><strong>Row direction.</strong> <code>GPUIsolines</code>, <code>GPUIsobands</code> and <code>GPUIsobandRings</code> place row 0 of the raster at
<code>minY</code> of the extent and read sample dimensions (<code>grid</code>, or the domain for tiled isolines), not
<code>grid.affine</code>. Construction validates tile bounds but does not infer row direction. A north-up grid
(row 0 at the top, the default of
<code>makeGPURasterMetadata</code>) therefore comes out mirrored north to south unless you reverse the row
order of the raster before upload. Mirroring the output instead (<code>y' = minY + maxY - y</code>) also
swaps the documented side: high values are then on the right of a segment and ring windings are
reversed. Making these classes follow the grid affine is deferred (roadmap 12.4).</p>
<p>A corner exactly on a level gives a zero-length <code>GPUIsolines</code> segment. The raw segment output
keeps it, because the edge-based stitching needs it as the bridge between the neighbouring cells'
chains; a contour through a sample stays one polyline. Stitched polyline vertex lists collapse
bit-identical consecutive vertices (a polyline made only of such segments keeps one vertex), and
the vertex capacity is checked before that collapse.</p>
<p><code>GPUIsolines</code> writes proper two-endpoint segments <code>[x0, y0, x1, y1]</code> (high side on the left),
their level index, and optionally the global edge ids <code>[startEdge, endEdge]</code>, ordered by
(cell, level, slot). Unlike <code>GPUTerrainContours</code>, a record is one segment, not a vertex pair to
reassemble. Passing <code>output.polylines</code> stitches segments of the same level into deterministic polylines
by pointer jumping in a fixed <code>ceil(log2(capacity)) + 1</code> rounds: open chains start at the segment
with no predecessor, closed rings at their smallest segment index and repeat the first vertex;
polylines are ordered by head segment. Publishing the longest prefix that fits the vertex capacity
uses binary search over at most <code>segmentCapacity</code> offsets, taking at most
<code>ceil(log2(segmentCapacity + 1))</code> steps in one invocation.</p>
<p><code>GPUIsobands</code> writes <code>bandClasses</code> (number of breaks <code>&#x3C;=</code> each sample, <code>0xffffffff</code> for nodata) for
fragment-side shading, and band geometry as counter-clockwise triangles with a band index:
each cell's fragment of band <code>[b[k - 1], b[k])</code> is <code>Above(b[k - 1]) ∩ Below(b[k])</code>, built
combinatorially from the cell boundary walk (no float clipping), at most two convex pieces per
band and cell. An optional <code>status.vertexCount</code> (<code>3 * status.count</code>) feeds a non-indexed indirect draw, and
<code>firstBand</code>/<code>lastBand</code> limit the emitted bands per frame.</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> lineParameters</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUParameterBuffer</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(device, {</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  id: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'isolines'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, format: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'float32'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, length: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">GPU_ISOLINES_PARAMETER_LENGTH</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">});</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> grid</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> makeGPURasterMetadata</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({width, height});</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUIsolines</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  grid, values, levels, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// levels: maximumLevelCount rows, contents per frame</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  parameters: lineParameters.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">importToGraph</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(graph),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  output: {</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    segments, segmentLevels,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    polylines: {vertices, polylineOffsets, polylineLevels, polylineClosed, polylineCount, vertexCount},</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    status: {count, requiredCount, overflow, polylineOverflow}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  }</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUIsobands</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  grid, values, breaks: levels,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  parameters: bandParameters.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">importToGraph</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(graph),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  output: {</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    bandClasses, triangles, triangleBands,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    status: {count: triangleCount, vertexCount: drawVertexCount, overflow: triangleOverflow}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  }</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">lineParameters.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">write</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">getGPUIsolinesParameterValues</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({levelCount: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">5</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, extent}, grid));</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">bandParameters.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">write</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">getGPUIsobandsParameterValues</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({breakCount: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">5</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, extent}, grid)); </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// no recompile</span></span></code></pre>
<p>The packers take the frame's grid (its width and height give the cell sizes and must match the
contributor's). On projected rasters add <code>localOrigin: [extent[0], extent[1]]</code> to both packers'
settings. Segments,
polyline vertices, triangles and ring vertices are then relative to it, so draw them with it as the
layer's coordinate origin. It also keeps <code>GPUIsobandRings</code> vertex hashes within their i32 range,
which saturates beyond <code>2^31 * vertexTolerance</code> (214,748 units at the default tolerance).</p>
<p><code>GPUIsobands</code> can also write the band boundary as edges: <code>output.edges</code> (<code>float32x4</code>), <code>edgeBands</code>,
<code>status.edgeCount</code>, <code>status.edgeOverflow</code> and an optional <code>status.edgeRequiredCount</code>. Edges are counter-clockwise with the band on
the left; an edge shared by two cells cancels, and a zero-area cell (one or two samples exactly on the
lower break) writes none, so edges lying on a break leave no slits.</p>
<p><code>GPUIsobandRings</code> chains those edges into closed rings with <code>GPUSegmentRingAssembly</code>, grouped by band, so
every band comes as shells and holes (holes attach within a band) and, optionally, GeoArrow polygons with
<code>sourceIds</code>. Props: <code>grid</code>, <code>values</code>, <code>breaks</code>, <code>parameters</code> (same packing as
<code>GPUIsobands</code>), a compile-time <code>edgeCapacity</code>, <code>vertexTolerance</code>, <code>splitTouchingRings</code> and <code>output</code> (the
<code>GPUSegmentRingAssembly</code> outputs plus optional <code>status.edgeCount</code> and required <code>status.edgeOverflow</code>).
Check both <code>status.edgeOverflow</code> (intermediate boundary-edge truncation) and <code>status.overflow</code>
(final ring/vertex capacity) before using geometry. Each edge status view is packed uint32 with at least one word; the edge
publisher writes only its first logical word and leaves padding untouched, including at nonzero
byte offsets. Intermediate edge status ranges must not overlap final ring output ranges; construction
rejects these aliases before allocating scratch. Props and output/status bindings are captured once.
A zero final overflow word alone does not establish complete rings.
Construction refuses an omitted edge-loss word; both flags are overwritten each encoding.
Maximum: <code>edgeCapacity</code> and the ring capacity (<code>output.ringOffsets.length - 1</code>) are each at most <code>getGPUIsobandRingsMaximumEdgeCapacity(device)</code> and <code>getGPUIsobandRingsMaximumRingCapacity(device)</code> (8388608 on a default device). Unlike the triangle fill it
gives true boundary geometry, but filling it for display needs polygon triangulation or rasterization
(<code>GPUPolygonRasterization</code> gives an exact even-odd fill at raster resolution).</p>
<p>Outputs are ordered by count, <code>GPUScan</code>, and scatter, so they are deterministic; capacities are
<code>segments.length</code> and <code>triangleBands.length</code>, with clamped <code>status.count</code>, <code>status.overflow</code>, and <code>status.requiredCount</code>
rewritten every encoding. If segments overflow, stitching produces no polylines and sets its own
<code>status.polylineOverflow</code>; a short <code>vertices</code> buffer keeps only the complete polylines that fit (the worst case is
<code>2 * segmentCapacity</code> vertices). Coordinates match an f32 CPU oracle within a few ULP (GPU
division and FMA contraction); topology, ordering, and band indices match exactly.</p>
<h4 id="gpuisolines"><code>GPUIsolines</code></h4>
<p>Traces marching-squares contour lines of an elevation raster at several levels and stitches them into polylines.</p>
<p>::lens{scenario=gpu-raster.isolines rung=result caption="Contours every 100 m; closed rings ring the summits, open chains run to the window edge."}</p>
<p>{/* parameters: GPUIsolines generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h5 id="gpuisolines-parameters"><code>GPUIsolines</code> parameters</h5>
<ul>
<li>Packer: <code>getGPUIsolinesParameterValues(settings, grid)</code>, writes <code>GPU_ISOLINES_PARAMETER_LENGTH</code> (8) <code>float32</code> values.
<ul>
<li><code>grid</code> (<code>GPURasterMetadata</code>): Frame raster grid; its width and height give the cell sizes and must match the contributor's unless <code>GPUIsolinesProps.tile</code> is set: then pass the DOMAIN grid and extent, and reuse identical parameters (including localOrigin) for all tiles. Tile offsets are applied in domain sample coordinates by the contributor, before interpolation.</li>
</ul>
</li>
</ul>
<p>Settings of <code>getGPUIsolinesParameterValues</code>:</p>
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
<td><code>levelCount</code></td>
<td><code>number</code></td>
<td>Number of active leading rows of the <code>levels</code> view. Clamped by the contributor to its maximum.</td>
</tr>
<tr>
<td><code>extent</code></td>
<td><code>readonly [number, number, number, number]</code></td>
<td>Outer world extent <code>[minX, minY, maxX, maxY]</code>. Sample <code>(column, row)</code> is at <code>(minX + (column + 0.5) * cellWidth, minY + (row + 0.5) * cellHeight)</code>, with cell sizes <code>(maxX - minX) / grid.width</code> and <code>(maxY - minY) / grid.height</code>. Row zero is half a cell above the lower edge. Contours interpolate between sample centers, so their domain ends at the outermost centers, half a cell inside the extent. This placement ignores grid.affine and pixelInterpretation; neither changes the row order.</td>
</tr>
<tr>
<td><code>localOrigin</code> (optional)</td>
<td><code>readonly [number, number]</code></td>
<td>Float64 point that segment and polyline vertices are written relative to; add it back on the CPU or as the layer's coordinate origin. Defaults to <code>[0, 0]</code> (absolute vertices). Pass the raster or tile corner <code>[minX, minY]</code> for projected rasters: absolute float32 vertices are quantized to 0.5 m at a UTM northing of 4.6e6 m, origin-relative ones to about <code>1.2e-7</code> of their distance from the origin (1.2 cm at 100 km).</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h4 id="how-it-computes-12">How it computes</h4>
<p>{/* lens:GPUIsolines */}</p>
<p>::lens{scenario=gpu-raster.isolines rung=datum datum=12 caption="Polyline 12, the 1300 m chain: its level, whether it is closed and where its vertices start."}
::lens{scenario=gpu-raster.isolines rung=formula caption="Each vertex sits where linear interpolation along a cell edge reaches the contour level."}
::lens{scenario=gpu-raster.isolines rung=kernel caption="Cells count and emit segments by their corner mask; stitching links segments by shared edge ids."}
::lens{scenario=gpu-raster.isolines rung=graph caption="Count, scan, scatter and publish segments, then link, jump, rank and gather polylines."}</p>
<h4 id="gpuisobands"><code>GPUIsobands</code></h4>
<p>Fills the elevation bands between consecutive breaks with counter-clockwise triangles tagged by band.</p>
<p>::lens{scenario=gpu-raster.isobands rung=result caption="Triangles of the 1300 to 1600 m bands, coloured by band index."}</p>
<p>{/* parameters: GPUIsobands generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h5 id="gpuisobands-parameters"><code>GPUIsobands</code> parameters</h5>
<ul>
<li>Packer: <code>getGPUIsobandsParameterValues(settings, grid)</code>, writes <code>GPU_ISOBANDS_PARAMETER_LENGTH</code> (12) <code>float32</code> values.
<ul>
<li><code>grid</code> (<code>GPURasterMetadata</code>): Frame raster grid; its width and height give the cell sizes and must match the contributor's.</li>
</ul>
</li>
</ul>
<p>Settings of <code>getGPUIsobandsParameterValues</code>:</p>
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
<td><code>breakCount</code></td>
<td><code>number</code></td>
<td>Number of active ascending breaks, at most the contributor's <code>maximumBreakCount</code> (larger values clamp on the GPU). <code>breakCount + 1</code> bands exist: band <code>k</code> covers <code>[b[k - 1], b[k])</code>.</td>
</tr>
<tr>
<td><code>extent</code></td>
<td><code>readonly [number, number, number, number]</code></td>
<td>Outer world extent <code>[minX, minY, maxX, maxY]</code>. Sample <code>(column, row)</code> is centered at <code>(minX + (column + 0.5) * cellWidth, minY + (row + 0.5) * cellHeight)</code>, with cell sizes <code>(maxX - minX) / grid.width</code> and <code>(maxY - minY) / grid.height</code>. Geometry interpolates over the rectangle between the outermost sample centers, half a cell inside each extent edge. Row zero is at the minY side and rows advance toward larger y. grid.affine and pixelInterpretation do not change this placement. Ring output uses the same convention.</td>
</tr>
<tr>
<td><code>firstBand</code> (optional)</td>
<td><code>number</code></td>
<td>First band emitted as geometry. Defaults to 0. Does not affect <code>bandClasses</code>.</td>
</tr>
<tr>
<td><code>lastBand</code> (optional)</td>
<td><code>number</code></td>
<td>Last band (inclusive) emitted as geometry. Defaults to <code>breakCount</code> (every band).</td>
</tr>
<tr>
<td><code>localOrigin</code> (optional)</td>
<td><code>readonly [number, number]</code></td>
<td>Float64 point that triangle vertices (and <code>GPUIsobandRings</code> ring vertices) are written relative to; add it back on the CPU or as the layer's coordinate origin. Defaults to <code>[0, 0]</code> (absolute vertices). Pass the raster or tile corner <code>[minX, minY]</code> for projected rasters: absolute float32 vertices are quantized to 0.5 m at a UTM northing of 4.6e6 m, origin-relative ones to about <code>1.2e-7</code> of their distance from the origin (1.2 cm at 100 km).</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h4 id="how-it-computes-13">How it computes</h4>
<p>{/* lens:GPUIsobands */}</p>
<p>::lens{scenario=gpu-raster.isobands rung=datum datum=120 caption="Triangle 120: its band index and the three vertices relative to the window corner."}
::lens{scenario=gpu-raster.isobands rung=kernel caption="Each cell walks its boundary, joins level crossings and fan-triangulates the band fragments."}
::lens{scenario=gpu-raster.isobands rung=graph caption="Per-cell triangle counts, scan, scatter and publish for the emitted band window."}</p>
<h4 id="gpuisobandrings"><code>GPUIsobandRings</code></h4>
<p>Chains the band boundary edges of an elevation raster into closed shell and hole rings grouped by band.</p>
<p>::lens{scenario=gpu-raster.isoband-rings rung=result caption="Closed rings of all eight bands, coloured by band index."}</p>
<h4 id="how-it-computes-14">How it computes</h4>
<p>{/* lens:GPUIsobandRings */}</p>
<p>::lens{scenario=gpu-raster.isoband-rings rung=datum datum=0 caption="Ring 0: its band, vertex range and the closed vertex list."}
::lens{scenario=gpu-raster.isoband-rings rung=kernel caption="Boundary vertices are hashed and matched, then rings are traced and ranked by pointer jumping."}
::lens{scenario=gpu-raster.isoband-rings rung=graph caption="Band boundary edges feed hash, sort, match, ring tracing and placement nodes."}</p>
<h2 id="gpurastersampling-and-gpurasterprofile"><code>GPURasterSampling</code> and <code>GPURasterProfile</code></h2>
<p>Both contributors accept a published <code>GPURasterBand</code> in <code>values</code>, including buffer and texture
storage. Band validity, native nodata and calibration are resolved by published raster primitives
before interpolation. <code>mask</code> filters additional cells; the legacy <code>noDataValue</code> applies only to a
bare value view, so a calibrated band value equal to that sentinel remains valid. Padded buffer
views are bounded to the grid, and borrowed validity is reread on every encoding.</p>
<p>Writable sampling and profile outputs must use buffers separate from inputs, including separate
imports of the same physical buffer. Construction errors name both conflicting ports: for example,
<code>output.values conflicts with values.storage.values</code>, or
<code>output.status.count conflicts with values.validity</code>. Bare raster views are named <code>values</code>; a
band's buffer and validity are named <code>values.storage.values</code> and <code>values.validity</code>. Optional masks,
point counts, polyline offsets, positions and parameters retain their own prop names in errors.
The same physical-buffer rules apply to raster algebra, isolines and isobands: separate graph
imports do not make a shared allocation independent. Writable outputs also require distinct physical
allocations. Algebra diagnostics preserve operand names (for example <code>output.values conflicts with a</code>),
band validity (<code>a.validity</code>), and stack indices (<code>stack[1].validity</code>) instead of private binding indices.</p>
<p><code>GPURasterSampling</code> samples a float32 raster at points ("extract raster values to points", draping,
tooltips). The pixel coordinate is <code>u = (x - minX) * (1 / cellWidth) - 0.5</code>, so integer <code>u</code> is a cell
centre; the extent, cell size, and row direction (row 0 at <code>maxY</code> for a north-up grid) come from the
<code>grid</code>. Points outside the closed extent (or with NaN coordinates) give NaN;
inside it, indices beyond the outermost centres clamp to the edge cell. <code>nearest</code> is the cell
containing the point (the <code>maxX</code>/<code>maxY</code> edge belongs to the last cell), <code>bilinear</code> uses the four
surrounding centres, <code>bicubic</code> is Catmull-Rom (<code>a = -0.5</code>) over the 4x4 support. A neighbour
participates only when both axis weights are nonzero, so a point on a cell centre returns that cell.
<code>noDataPolicy: 'strict'</code> gives NaN when a participating neighbour is nodata; <code>'renormalize'</code> divides
the valid bilinear weights by their sum; bicubic with nodata in its support falls back to the bilinear
rule of the same policy. Nodata is NaN or an infinity, an optional finite <code>noDataValue</code>, or a zero <code>validity</code> flag.
<code>method</code>, <code>noDataPolicy</code>, and the packed grid placement are per-frame; <code>pointCount</code> is an optional per-frame active
count (rows beyond it are NaN and 0).</p>
<p><code>GPURasterProfile</code> builds elevation profiles along polylines (<code>positions</code> plus CSR <code>lineOffsets</code>)
at a per-frame <code>spacing</code> (planar distances in grid world units). Each line emits samples at
<code>0, s, 2s, ...</code> below its length plus the final vertex at exactly the length; a zero-length line emits
one sample, an empty line none. Outputs (any subset): sample positions, distances, values, line IDs,
cumulative gain and loss; per line: sample offsets (clamped to capacity), length, gain, loss, minimum,
and maximum over finite samples (gain and loss sum positive and negative differences between
consecutive finite samples); plus <code>status.count</code>, <code>status.overflow</code>, and <code>status.requiredCount</code>.</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> samplingParameters</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUParameterBuffer</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(device, {</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  id: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'sampling'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, format: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'float32'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, length: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">GPU_RASTER_SAMPLING_PARAMETER_LENGTH</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">});</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPURasterSampling</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  grid, values: elevation, positions, parameters: samplingParameters.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">importToGraph</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(graph),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  output: {values: pointElevations, outputValidity}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPURasterProfile</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  grid, values: elevation, positions: lineVertices, lineOffsets,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  parameters: profileParameters.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">importToGraph</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(graph),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  output: {status: {count, overflow}, sampleValues, sampleDistances, lineGains, lineLosses}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">samplingParameters.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">write</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">getGPURasterSamplingParameterValues</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({method: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'bicubic'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}, grid));</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">profileParameters.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">write</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">getGPURasterProfileParameterValues</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({method: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'bilinear'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, spacing: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">25</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}, grid));</span></span></code></pre>
<p>On projected rasters pass <code>localOrigin</code> (for example the raster corner) to either packer and
upload point <code>positions</code> and line <code>positions</code> relative to it; <code>samplePositions</code> come back relative to it.
Distances are unaffected.</p>
<p>The profile pipeline is a serial per-line walk, <code>GPUScan</code>, a per-sample binary search and sample,
then a serial per-line cumulative pass, all in fixed order, so results are deterministic. Exactly
representable (dyadic) inputs match an f32 CPU oracle bit for bit; elsewhere GPU division and FMA
contraction give a few ULP.</p>
<h4 id="gpurastersampling"><code>GPURasterSampling</code></h4>
<p>Reads a raster at point positions with nearest, bilinear or bicubic interpolation, giving one value per point.</p>
<p>::lens{scenario=gpu-raster.raster-sampling rung=result caption="25 DEM sample points coloured by elevation; point 7 sits off cell centres and averages four cells."}</p>
<p>{/* parameters: GPURasterSampling generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h5 id="gpurastersampling-parameters"><code>GPURasterSampling</code> parameters</h5>
<ul>
<li>Packer: <code>getGPURasterSamplingParameterValues(settings, grid)</code>, writes <code>GPU_RASTER_SAMPLING_PARAMETER_LENGTH</code> (12) <code>float32</code> values.
<ul>
<li><code>grid</code> (<code>GPURasterMetadata</code>): The frame's grid; its width and height must match the contributor's.</li>
</ul>
</li>
</ul>
<p>Settings of <code>getGPURasterSamplingParameterValues</code>:</p>
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
<td><code>method</code> (optional)</td>
<td><code>GPURasterSamplingMethod</code></td>
<td>Interpolation method. Defaults to <code>'bilinear'</code>.</td>
</tr>
<tr>
<td><code>noDataPolicy</code> (optional)</td>
<td><code>GPURasterSamplingNoDataPolicy</code></td>
<td>Nodata handling for bilinear and bicubic. Defaults to <code>'strict'</code>.</td>
</tr>
<tr>
<td><code>localOrigin</code> (optional)</td>
<td><code>readonly [number, number]</code></td>
<td>Float64 point that query positions (sampling points, profile line vertices) are given relative to, and that profile sample positions are written relative to. Defaults to <code>[0, 0]</code> (absolute positions). Pass the raster or tile corner <code>[minX, minY]</code> for projected rasters: absolute float32 positions are quantized to 0.5 m at a UTM northing of 4.6e6 m, origin-relative ones to about <code>1.2e-7</code> of their distance from the origin (1.2 cm at 100 km).</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h4 id="how-it-computes-15">How it computes</h4>
<p>{/* lens:GPURasterSampling */}</p>
<p>::lens{scenario=gpu-raster.raster-sampling rung=datum datum=7 caption="Point 7 sits on a cell corner, so four cells each weigh 0.25."}
::lens{scenario=gpu-raster.raster-sampling rung=formula caption="Each point blends the four surrounding cell centres by distance; nodata gives NaN under strict."}
::lens{scenario=gpu-raster.raster-sampling rung=kernel caption="One invocation per point finds its cell and weights, then reads the raster."}
::lens{scenario=gpu-raster.raster-sampling rung=graph caption="A single sampling node reads the raster, the points and the parameters."}</p>
<h4 id="gpurasterprofile"><code>GPURasterProfile</code></h4>
<p>Samples a raster along a polyline at a fixed spacing, giving elevation and distance per sample and total ascent per line.</p>
<p>::lens{scenario=gpu-raster.raster-profile rung=result caption="33 samples every 250 m along the line, coloured by elevation, fall from the ridge to the valley."}</p>
<p>{/* parameters: GPURasterProfile generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h5 id="gpurasterprofile-parameters"><code>GPURasterProfile</code> parameters</h5>
<ul>
<li>Packer: <code>getGPURasterProfileParameterValues(settings, grid)</code>, writes <code>GPU_RASTER_PROFILE_PARAMETER_LENGTH</code> (12) <code>float32</code> values.
<ul>
<li><code>grid</code> (<code>GPURasterMetadata</code>): The frame's grid; its width and height must match the contributor's.</li>
</ul>
</li>
</ul>
<p>Settings of <code>getGPURasterProfileParameterValues</code>:</p>
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
<td><code>method</code> (optional)</td>
<td><code>GPURasterSamplingMethod</code></td>
<td>Interpolation method. Defaults to <code>'bilinear'</code>.</td>
</tr>
<tr>
<td><code>noDataPolicy</code> (optional)</td>
<td><code>GPURasterSamplingNoDataPolicy</code></td>
<td>Nodata handling for bilinear and bicubic. Defaults to <code>'strict'</code>.</td>
</tr>
<tr>
<td><code>localOrigin</code> (optional)</td>
<td><code>readonly [number, number]</code></td>
<td>Float64 point that query positions (sampling points, profile line vertices) are given relative to, and that profile sample positions are written relative to. Defaults to <code>[0, 0]</code> (absolute positions). Pass the raster or tile corner <code>[minX, minY]</code> for projected rasters: absolute float32 positions are quantized to 0.5 m at a UTM northing of 4.6e6 m, origin-relative ones to about <code>1.2e-7</code> of their distance from the origin (1.2 cm at 100 km).</td>
</tr>
<tr>
<td><code>spacing</code></td>
<td><code>number</code></td>
<td>Distance between samples along a line, in grid world units. Must be finite and positive.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h4 id="how-it-computes-16">How it computes</h4>
<p>{/* lens:GPURasterProfile */}</p>
<p>::lens{scenario=gpu-raster.raster-profile rung=datum datum=14 caption="Sample 14, 3500 m along the line, after the steepest drop of the profile."}
::lens{scenario=gpu-raster.raster-profile rung=formula caption="Samples sit every spacing along the polyline, with a last one at its end."}
::lens{scenario=gpu-raster.raster-profile rung=kernel caption="Edge lengths are scanned, each sample finds its segment, then the sampling rule and a gain scan run."}
::lens{scenario=gpu-raster.raster-profile rung=graph caption="Vertex distances, sample counts, scan, locate, sample and gain nodes run in order."}</p>
<h2 id="gpuparticleadvection"><code>GPUParticleAdvection</code></h2>
<p>Moves particles through a vector field with midpoint steps and records each position in a trail ring buffer.</p>
<p>::lens{scenario=gpu-raster.particle-advection rung=result caption="200 particles drift for 24 hours of wind, coloured by final speed; none left the window."}</p>
<p>Advects particles through a 2D vector field (wind, ocean currents, flow directions) one frame per
encoding, with an optional trail ring buffer that a <code>PathLayer</code> or <code>LineLayer</code> can draw without CPU
readback. Unlike screen-space fading trails, the trails are in data space, so they survive panning
and zooming.</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> parameters</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUParameterBuffer</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(device, {id: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'wind'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, format: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'float32'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, length: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">12</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">});</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> words</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUParameterBuffer</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(device, {id: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'wind-words'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, format: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'uint32'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, length: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">4</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">});</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">  new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUParticleAdvection</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    velocities, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// GraphDataView&#x3C;'float32x2'>, (u, v) per cell, row 0 = smallest y</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    fieldWidth: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">360</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    fieldHeight: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">180</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    parameters: parameters.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">importToGraph</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(graph),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    uint32Parameters: words.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">importToGraph</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(graph),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    state: {positions, ages, generations}, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// particleCount rows each, updated in place</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    output: {</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">      previousPositions, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// optional: segment start for a LineLayer</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">      speeds, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// optional: colour by speed</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">      trails: {positions: trailPositions, length: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">16</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">} </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// optional ring, particleCount * 16 rows</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    }</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  })</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">);</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// Every frame, no recompile:</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">parameters.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">write</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">  getGPUParticleAdvectionParameterValues</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    {fieldExtent: [</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">-</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">180</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">-</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">90</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">], timeStep: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> /</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> 60</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, speedScale: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">0.2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, dropRate: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">0.003</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">},</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    [</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">360</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">180</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">]</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  )</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">);</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">words.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">write</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">getGPUParticleAdvectionUint32ParameterValues</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({seed: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, frame, maximumAge: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">120</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, reset: frame </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">===</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> 0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span></code></pre>
<ul>
<li>Each frame takes one RK2 (midpoint) step with <code>h = timeStep * speedScale</code>, sampling the field by
manual bilinear interpolation between cell centres (no filterable float texture needed).</li>
<li>A particle respawns inside <code>spawnBounds</code> (default: the field bounds) when <code>reset</code> is set, its age
reaches <code>maximumAge</code> (0 disables ageing), any sample leaves the field or touches a NaN cell, it is
slower than <code>minimumSpeed</code>, or a per-frame drop test fires with probability <code>dropRate</code>. On <code>reset</code>
ages are staggered so particles do not all expire together.</li>
<li>Exact replays: random numbers come from a Philox 4x32-10 counter generator, keyed
<code>(seed, particle, generation)</code> for spawn positions and <code>(seed, particle, frame)</code> for drop tests,
with no persistent RNG state. The same parameter sequence replays bit for bit.</li>
<li>Trails: particle <code>i</code> owns rows <code>i * L</code> to <code>i * L + L - 1</code>; frame <code>f</code> writes slot <code>f % L</code> (newest),
the oldest is <code>(f + 1) % L</code>. A respawn fills the whole ring with the spawn position.</li>
<li>State is updated in place: every invocation touches only its own particle, so no ping-pong copy is
needed.</li>
</ul>
<h4 id="how-it-computes-17">How it computes</h4>
<p>{/* lens:GPUParticleAdvection */}</p>
<p>::lens{scenario=gpu-raster.particle-advection rung=datum datum=11 caption="Particle 11, the farthest traveller: its trail of twelve positions and final wind speed."}
::lens{scenario=gpu-raster.particle-advection rung=formula caption="One midpoint step per frame: sample the wind at the start, then at the half step."}
::lens{scenario=gpu-raster.particle-advection rung=kernel caption="Each particle samples the field three times, then moves or respawns; a second pass writes its trail slot."}
::lens{scenario=gpu-raster.particle-advection rung=graph caption="Twelve step nodes and twelve trail nodes run in order on the same particle state."}</p>
<p>{/* parameters: GPUParticleAdvection generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h3 id="gpuparticleadvection-parameters"><code>GPUParticleAdvection</code> parameters</h3>
<ul>
<li>Packer: <code>getGPUParticleAdvectionParameterValues(settings, fieldSize)</code>, writes <code>GPU_PARTICLE_ADVECTION_PARAMETER_LENGTH</code> (12) <code>float32</code> values.
<ul>
<li><code>fieldSize</code> (<code>readonly [number, number]</code>): <code>[width, height]</code> of the field in cells, used for the default spawn bounds.</li>
</ul>
</li>
<li>Packer: <code>getGPUParticleAdvectionUint32ParameterValues(settings)</code>, writes <code>GPU_PARTICLE_ADVECTION_UINT32_PARAMETER_LENGTH</code> (4) <code>uint32</code> values.</li>
</ul>
<p>Settings of <code>getGPUParticleAdvectionParameterValues</code>:</p>
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
<td><code>fieldExtent</code></td>
<td><code>readonly [number, number, number, number]</code></td>
<td>Field raster placement <code>[originX, originY, cellWidth, cellHeight]</code>, not min/max bounds. Positive cellHeight puts row zero on the smallest-y side; signed sizes set index direction. Cell <code>(i, j)</code> is centered at <code>origin + (i + 0.5, j + 0.5) * cellSize</code>. A cell size that is zero or not finite throws.</td>
</tr>
<tr>
<td><code>timeStep</code></td>
<td><code>number</code></td>
<td>Time step of this frame, in the time unit of the field velocities.</td>
</tr>
<tr>
<td><code>speedScale</code> (optional)</td>
<td><code>number</code></td>
<td>Multiplier on the time step, for example a playback speed. Defaults to 1.</td>
</tr>
<tr>
<td><code>dropRate</code> (optional)</td>
<td><code>number</code></td>
<td>Probability in <code>[0, 1]</code> that a particle respawns this frame regardless of age. Defaults to 0.</td>
</tr>
<tr>
<td><code>minimumSpeed</code> (optional)</td>
<td><code>number</code></td>
<td>Particles slower than this (world units per time unit, after <code>speedScale</code>) respawn. Defaults to 0.</td>
</tr>
<tr>
<td><code>spawnBounds</code> (optional)</td>
<td><code>readonly [number, number, number, number]</code></td>
<td>Respawn rectangle <code>[minX, minY, maxX, maxY]</code>. Defaults to the field bounds. Particles spawned outside the field respawn again on the next frame.</td>
</tr>
</tbody>
</table>
<p>Settings of <code>getGPUParticleAdvectionUint32ParameterValues</code>:</p>
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
<td><code>seed</code></td>
<td><code>number</code></td>
<td>Random seed. Every random decision is a pure function of the seed and particle counters.</td>
</tr>
<tr>
<td><code>frame</code></td>
<td><code>number</code></td>
<td>Frame counter. Keys the per-frame drop test and selects the trail ring slot <code>frame % L</code>.</td>
</tr>
<tr>
<td><code>maximumAge</code></td>
<td><code>number</code></td>
<td>Particles respawn when their age reaches this many frames. <code>0</code> disables ageing.</td>
</tr>
<tr>
<td><code>reset</code> (optional)</td>
<td><code>boolean</code></td>
<td>When true, every particle respawns with generation 0 and a staggered age, which initialises the state buffers. Pass it on the first frame and whenever the scene is reset.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h2 id="gpulineintegralconvolution"><code>GPULineIntegralConvolution</code></h2>
<p>Smears seeded white noise along the streamlines of a vector field, giving a 0 to 1 texture that shows flow direction.</p>
<p>::lens{scenario=gpu-raster.line-integral-convolution rung=result caption="Noise streaks follow the wind across the 16 by 12 degree window."}</p>
<p>Line integral convolution: smears white noise along the streamlines of a vector field so the flow
direction is visible everywhere. Output is a float raster in <code>[0, 1]</code> (plus an optional <code>r32float</code>
texture and a speed raster) for a <code>BitmapLayer</code>, typically modulated by a speed colour ramp.</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">  new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPULineIntegralConvolution</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    velocities, fieldWidth: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">360</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, fieldHeight: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">180</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    grid: </span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">makeGPURasterMetadata</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({width: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">1024</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, height: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">512</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}), </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// output pixels</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    stepCount: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">20</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// steps per direction</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    parameters: parameters.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">importToGraph</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(graph), </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// float32, 12</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    uint32Parameters: words.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">importToGraph</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(graph), </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// uint32, 4</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    output: {values, speeds, texture}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  })</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">);</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">parameters.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">write</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">  getGPULineIntegralConvolutionParameterValues</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    fieldExtent: [</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">-</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">180</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">-</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">90</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">],</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    outputExtent: [</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">-</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">180</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">-</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">90</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">360</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> /</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> 1024</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">180</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> /</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> 512</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">],</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    stepLength: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">0.5</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// output pixels</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    period: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">8</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// animated ripple, in steps; 0 = static LIC</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    phase: frame </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">/</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> 30</span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D"> // advance every frame to make the texture flow</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  })</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">);</span></span></code></pre>
<ul>
<li>Noise is Philox white noise per output pixel keyed by <code>(seed, column, row)</code>; the kernel is a Hann
window, optionally times an animated ripple <code>0.5 * (1 + cos(2 * pi * (s / period - phase)))</code>.</li>
<li>Streamlines step along the normalised field (RK2) and stop at the field or output edge, NaN data,
or speeds at or below <code>minimumSpeed</code>. Pixels whose centre has no data are NaN. A step is tangent
to the field in world space and <code>stepLength</code> output pixels long, also on non-square output cells
(before 2026-11-10 the step was scaled per axis, which bent the trace on non-square cells).</li>
<li>The output extent is independent of the field, so a viewport renders at screen resolution from a
coarse field.</li>
</ul>
<h4 id="how-it-computes-18">How it computes</h4>
<p>{/* lens:GPULineIntegralConvolution */}</p>
<p>::lens{scenario=gpu-raster.line-integral-convolution rung=datum datum=6208 caption="The centre pixel: its wind speed and the value from its weighted noise samples."}
::lens{scenario=gpu-raster.line-integral-convolution rung=formula caption="A Hann-weighted mean of the noise along the streamline, 24 steps each way."}
::lens{scenario=gpu-raster.line-integral-convolution rung=kernel caption="Each pixel traces its streamline both ways and averages the noise it visits."}
::lens{scenario=gpu-raster.line-integral-convolution rung=graph caption="A noise node fills the white noise, then the convolution node reads it."}</p>
<p>{/* parameters: GPULineIntegralConvolution generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h3 id="gpulineintegralconvolution-parameters"><code>GPULineIntegralConvolution</code> parameters</h3>
<ul>
<li>Packer: <code>getGPULineIntegralConvolutionParameterValues(settings)</code>, writes <code>GPU_LINE_INTEGRAL_CONVOLUTION_PARAMETER_LENGTH</code> (12) <code>float32</code> values.</li>
<li>Packer: <code>getGPULineIntegralConvolutionUint32ParameterValues(settings)</code>, writes <code>GPU_LINE_INTEGRAL_CONVOLUTION_UINT32_PARAMETER_LENGTH</code> (4) <code>uint32</code> values.</li>
</ul>
<p>Settings of <code>getGPULineIntegralConvolutionParameterValues</code>:</p>
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
<td><code>fieldExtent</code></td>
<td><code>readonly [number, number, number, number]</code></td>
<td>Field placement <code>[originX, originY, cellWidth, cellHeight]</code>, not min/max bounds. The origin is an outer corner; sample <code>(column, row)</code> is at origin + <code>(column + 0.5, row + 0.5)</code> times cell size. Positive cellHeight places row zero at the smallest y; signed cell sizes set the direction of increasing indices. Zero or non-finite sizes throw. The field is sampled bilinearly between centers, with edge clamping inside its half-open outer bounds.</td>
</tr>
<tr>
<td><code>outputExtent</code></td>
<td><code>readonly [number, number, number, number]</code></td>
<td>Output raster placement <code>[originX, originY, cellWidth, cellHeight]</code>. LIC starts at each output center, origin + <code>(column + 0.5, row + 0.5)</code> times cell size, independently of the field lattice. Signed cell sizes set row/column direction; zero or non-finite sizes throw. grid.affine and pixelInterpretation do not change this placement.</td>
</tr>
<tr>
<td><code>stepLength</code> (optional)</td>
<td><code>number</code></td>
<td>Streamline step in output pixels. Defaults to 0.5. The step follows the field in world space; on non-square output cells its world length is the one that spans <code>stepLength</code> pixels in that direction.</td>
</tr>
<tr>
<td><code>minimumSpeed</code> (optional)</td>
<td><code>number</code></td>
<td>A streamline stops where the field is slower than this. Defaults to 0.</td>
</tr>
<tr>
<td><code>phase</code> (optional)</td>
<td><code>number</code></td>
<td>Animation phase in kernel periods; advance it every frame to make the texture flow. Defaults to 0.</td>
</tr>
<tr>
<td><code>period</code> (optional)</td>
<td><code>number</code></td>
<td>Ripple period in steps. <code>0</code> disables the animated ripple (static LIC). Defaults to 0.</td>
</tr>
<tr>
<td><code>seed</code> (optional)</td>
<td><code>number</code></td>
<td>Seed of the white noise. Defaults to 0.</td>
</tr>
</tbody>
</table>
<p>Settings of <code>getGPULineIntegralConvolutionUint32ParameterValues</code>:</p>
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
<td><code>seed</code> (optional)</td>
<td><code>number</code></td>
<td>Seed of the white noise. Defaults to 0.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h2 id="gpustreamlines"><code>GPUStreamlines</code></h2>
<p>Traces evenly spaced streamlines from jittered seeds of a vector field and prunes them to a separation distance.</p>
<p>::lens{scenario=gpu-raster.streamlines rung=result caption="Seeded lines pruned to 0.5 degree separation, coloured by seed index."}</p>
<p>Evenly spaced streamlines (after Jobard and Lefer) whose pruning is resolved by a random priority
per seed instead of processing order, so the result is deterministic. Output is CSR polylines for a
<code>PathLayer</code>.</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">  new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUStreamlines</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    velocities, fieldWidth: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">360</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, fieldHeight: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">180</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    gridWidth: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">180</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, gridHeight: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">90</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// occupancy grid; its cell is the separation distance</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    seedColumns: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">90</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, seedRows: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">45</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, stepsPerDirection: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">60</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, maximumIterations: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">16</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    parameters: parameters.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">importToGraph</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(graph), </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// float32, 12</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    uint32Parameters: words.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">importToGraph</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(graph), </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// uint32, 4</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    output: {</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">      lines: {ids, status: {count, overflow, requiredCount}}, lineOffsets, positions, positionCount,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">      status: {iterationLimitReached}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    }</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  })</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">);</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">parameters.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">write</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">  getGPUStreamlinesParameterValues</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    fieldExtent: [</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">-</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">180</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">-</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">90</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">],</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    gridExtent: [</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">-</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">180</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">-</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">90</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">],</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    stepLength: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">0.5</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  })</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">);</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">words.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">write</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">getGPUStreamlinesUint32ParameterValues</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({seed: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, minimumPoints: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">8</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span></code></pre>
<ul>
<li>Seeds form a Philox-jittered lattice over the grid extent; each is traced <code>L</code> RK2 steps backward and
forward. Priorities are Philox keys with ties toward the lower seed index.</li>
<li>The result equals a greedy pass in descending priority: reject a line whose seed cell is occupied,
cut each direction before its first occupied cell, accept it if it keeps <code>minimumPoints</code> points,
and occupy its cells. The GPU reaches it in at most <code>maximumIterations</code> GPU-gated rounds of claim (<code>atomicMax</code> of
keys over the cells a line can still keep) and decide (accept lines that win all their cells);
<code>status.iterationLimitReached</code> is 1 if rounds ran out (undecided lines are dropped).</li>
<li>Published lines are in ascending seed order: line <code>i</code> is <code>positions[lineOffsets[i] .. lineOffsets[i + 1])</code>.
<code>lines.status.overflow</code> is 1 when lines were dropped because the line or point capacity was full.</li>
<li>Optional <code>lines.status.candidateOverflow</code> is 1 when the pruning budget leaves undecided seeds;
it is independent of final capacity. <code>requiredCount</code> counts decided accepted lines and is then
a lower bound. The CPU twin and both operation forms honor <code>maximumIterations</code> (default 32)
and expose the same flags. <code>GPU_STREAMLINES_CANDIDATE_UNCONVERGED</code> (1) replaces the old
overflow bit-2 constant; a zero final-capacity flag alone does not establish completeness.</li>
<li>Optional <code>candidates</code> outputs expose the traced lines before pruning, for diagnostics.</li>
</ul>
<p>Both operation forms capture iteration budgets and nested output bindings once at construction,
including inherited getters. Mutating the original JavaScript maps later does not change the
operation; changes to bound buffer contents still apply on each execution.</p>
<h4 id="how-it-computes-19">How it computes</h4>
<p>{/* lens:GPUStreamlines */}</p>
<p>::lens{scenario=gpu-raster.streamlines rung=datum datum=11 caption="Line 11: its seed index and where its points start in the shared position list."}
::lens{scenario=gpu-raster.streamlines rung=kernel caption="Seeds are traced both ways, then rounds claim and decide occupancy-grid cells."}
::lens{scenario=gpu-raster.streamlines rung=graph caption="Trace, pruning rounds, scans and publish nodes build the CSR polylines."}</p>
<p>{/* parameters: GPUStreamlines generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h3 id="gpustreamlines-parameters"><code>GPUStreamlines</code> parameters</h3>
<ul>
<li>Packer: <code>getGPUStreamlinesParameterValues(settings)</code>, writes <code>GPU_STREAMLINES_PARAMETER_LENGTH</code> (12) <code>float32</code> values.</li>
<li>Packer: <code>getGPUStreamlinesUint32ParameterValues(settings)</code>, writes <code>GPU_STREAMLINES_UINT32_PARAMETER_LENGTH</code> (4) <code>uint32</code> values.</li>
</ul>
<p>Settings of <code>getGPUStreamlinesParameterValues</code>:</p>
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
<td><code>fieldExtent</code></td>
<td><code>readonly [number, number, number, number]</code></td>
<td>Field placement <code>[originX, originY, cellWidth, cellHeight]</code>, not min/max bounds. The origin is an outer corner; sample <code>(column, row)</code> is at origin + <code>(column + 0.5, row + 0.5)</code> times cell size. Positive cellHeight puts row zero at the smallest y; signed cell sizes determine the direction of increasing indices. Zero or non-finite sizes throw. Sampling is bilinear between centers, clamped inside the outer bounds. No grid affine changes this placement.</td>
</tr>
<tr>
<td><code>gridExtent</code></td>
<td><code>readonly [number, number, number, number]</code></td>
<td>Occupancy grid placement <code>[originX, originY, cellWidth, cellHeight]</code>. The cell size is the separation distance between streamlines; seeds and traced points stay inside the grid. Cell <code>(column, row)</code> is the half-open rectangle from origin + <code>(column, row)</code> times cell size to origin + <code>(column + 1, row + 1)</code> times cell size. Both cell sizes must be positive.</td>
</tr>
<tr>
<td><code>stepLength</code></td>
<td><code>number</code></td>
<td>Integration step in world units.</td>
</tr>
<tr>
<td><code>minimumSpeed</code> (optional)</td>
<td><code>number</code></td>
<td>A streamline stops where the field is slower than this. Defaults to 0.</td>
</tr>
<tr>
<td><code>seed</code> (optional)</td>
<td><code>number</code></td>
<td>Seed of the jittered seed lattice and the line priorities. Defaults to 0.</td>
</tr>
<tr>
<td><code>minimumPoints</code> (optional)</td>
<td><code>number</code></td>
<td>Streamlines with fewer points after trimming are rejected. Defaults to 2.</td>
</tr>
</tbody>
</table>
<p>Settings of <code>getGPUStreamlinesUint32ParameterValues</code>:</p>
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
<td><code>seed</code> (optional)</td>
<td><code>number</code></td>
<td>Seed of the jittered seed lattice and the line priorities. Defaults to 0.</td>
</tr>
<tr>
<td><code>minimumPoints</code> (optional)</td>
<td><code>number</code></td>
<td>Streamlines with fewer points after trimming are rejected. Defaults to 2.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h2 id="gpuchangedetection"><code>GPUChangeDetection</code></h2>
<p>Subtracts the before slice from the after slice for each band; here near-infrared and shortwave-infrared reflectance around the fire.</p>
<p>::lens{scenario=gpu-raster.change-detection rung=result caption="After minus before for near-infrared; strong decreases mark the fire scar."}</p>
<p><code>GPUChangeDetection</code> computes per-cell change statistics over a dense float32 stack of time slices
indexed <code>(cell * sliceCount + slice) * bandCount + band</code> (cell-major, like <code>GPUTemporalReduction</code>).
A sample is missing when it is not finite (NaN or an infinity) or when the optional per-observation
<code>observationMask</code> view (<code>cell * sliceCount + slice</code>, for example a per-date cloud mask) is zero for it, the
rule of the raster time-series contributors. The modes are selected at compile time by which outputs are present; each
present output group adds one compute node (<code>two-slice</code>, <code>multiband</code>, <code>t-test</code>, <code>sen-slope</code>,
<code>mann-kendall</code>), and one thread handles one cell with fixed-order loops, so results are
deterministic and no atomics are used.</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> parameters</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUParameterBuffer</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(device, {</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  id: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'change-parameters'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  format: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'float32'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  length: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">GPU_CHANGE_DETECTION_PARAMETER_LENGTH</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">});</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">  new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUChangeDetection</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    slices: stack, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// cellCount * sliceCount float32 rows</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    parameters: parameters.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">importToGraph</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(graph),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    cellCount,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    sliceCount,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    output: {difference, logRatio, tStatistic, tPValue, senSlope, mannKendallS, significance}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  })</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">);</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">parameters.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">write</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">  getGPUChangeDetectionParameterValues</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({beforeSlice: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, afterSlice: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">11</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, splitSlice: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">6</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, alpha: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">0.05</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">})</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">);</span></span></code></pre>
<ul>
<li><code>compile-time</code>: <code>cellCount</code>, <code>sliceCount</code> (2 to 256, at most 64 with <code>senSlope</code>), <code>bandCount</code>
(default 1), the optional cell <code>mask</code>, <code>significanceSource</code>, and which outputs exist.</li>
<li><code>per frame</code> (parameter buffer, no rebuild): <code>beforeSlice</code>, <code>afterSlice</code>, <code>epsilon</code>, <code>alpha</code>,
<code>splitSlice</code>.</li>
<li>Two-slice outputs (any <code>bandCount</code>, <code>cellCount * bandCount</code> rows): <code>difference</code> (after minus
before), <code>logRatio</code> (<code>ln((after + epsilon) / (before + epsilon))</code>, NaN when either side is
<code>&#x3C;= -epsilon</code>) and <code>percentChange</code> (<code>100 * (after - before) / |before|</code>, NaN when before is 0).</li>
<li>Welch t-test between slices <code>[0, splitSlice)</code> and <code>[splitSlice, sliceCount)</code>: <code>tStatistic</code>,
<code>tDegreesOfFreedom</code> (Welch-Satterthwaite) and two-sided <code>tPValue</code>. Missing slices are dropped; each
group needs 2 valid slices and a positive standard error. Variance is two-pass. The p-value is
the regularized incomplete beta <code>I_x(df / 2, 1 / 2)</code>, <code>x = df / (df + t^2)</code>, by continued
fraction (at most 64 iterations) with a Stirling log-gamma, in float32 (about 1e-4 absolute for
<code>df</code> up to 254).</li>
<li><code>senSlope</code>: exact Theil-Sen median of the pairwise slopes over valid slices, by in-thread bounded
max-heap selection. <code>O(T^2 log T)</code> per cell, so <code>sliceCount &#x3C;= 64</code>.</li>
<li><code>mannKendallS</code> (<code>sint32</code>, integer exact), <code>mannKendallZ</code> and <code>mannKendallP</code>: tie-corrected
variance, <code>Z = (S - sign(S)) / sqrt(var)</code>, <code>p = erfc(|Z| / sqrt(2))</code> (Numerical Recipes <code>erfcc</code>,
relative error below 1.2e-7). Fewer than 2 valid slices give NaN; all values tied give Z 0, p 1.</li>
<li>Multiband (<code>bandCount >= 2</code>, at most 16): <code>changeMagnitude</code> (Euclidean norm of the band
difference vector) and <code>changeDirection</code> (float32 <code>atan2(d1, d0)</code> for 2 bands; uint32 sign code
for 3 to 16 bands with bit <code>b</code> set for an increase and bit <code>16 + b</code> for a decrease; NaN or
<code>0xffffffff</code> when any band is missing).</li>
<li><code>significance</code> (uint32): 1 significant increase, 2 significant decrease, 0 otherwise, from the
t-test p-value (default) or Mann-Kendall p-value (<code>significanceSource: 'mann-kendall'</code>) below
<code>alpha</code>; the sign follows the t statistic or S.</li>
<li>Masked cells write NaN (float), 0 (<code>mannKendallS</code>, <code>significance</code>) or <code>0xffffffff</code> (multiband
direction code).</li>
</ul>
<h4 id="how-it-computes-20">How it computes</h4>
<p>{/* lens:GPUChangeDetection */}</p>
<p>::lens{scenario=gpu-raster.change-detection rung=datum datum=2400 caption="One burn-scar cell: select the before and after slices, then subtract."}
::lens{scenario=gpu-raster.change-detection rung=formula caption="Difference of the two selected slices; masked, out-of-range or missing input gives NaN."}
::lens{scenario=gpu-raster.change-detection rung=kernel caption="One thread per cell reads both slices and writes the difference."}
::lens{scenario=gpu-raster.change-detection rung=graph caption="One two-slice node per band, sharing one parameter buffer."}</p>
<p>{/* parameters: GPUChangeDetection generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h3 id="gpuchangedetection-parameters"><code>GPUChangeDetection</code> parameters</h3>
<ul>
<li>Packer: <code>getGPUChangeDetectionParameterValues(settings)</code>, writes <code>GPU_CHANGE_DETECTION_PARAMETER_LENGTH</code> (5) <code>float32</code> values.</li>
</ul>
<p>Settings of <code>getGPUChangeDetectionParameterValues</code>:</p>
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
<td><code>beforeSlice</code></td>
<td><code>number</code></td>
<td>Slice index of the "before" slice for two-slice comparisons. Out of range makes them NaN.</td>
</tr>
<tr>
<td><code>afterSlice</code></td>
<td><code>number</code></td>
<td>Slice index of the "after" slice for two-slice comparisons. Out of range makes them NaN.</td>
</tr>
<tr>
<td><code>epsilon</code> (optional)</td>
<td><code>number</code></td>
<td>Denominator guard of the log ratio. Defaults to <code>1e-6</code>.</td>
</tr>
<tr>
<td><code>alpha</code> (optional)</td>
<td><code>number</code></td>
<td>Two-sided significance level for the <code>significance</code> class. Defaults to <code>0.05</code>.</td>
</tr>
<tr>
<td><code>splitSlice</code> (optional)</td>
<td><code>number</code></td>
<td>First slice of the "after" group of the Welch t-test: slices <code>[0, splitSlice)</code> are "before" and <code>[splitSlice, sliceCount)</code> are "after". Must be in <code>[1, sliceCount - 1]</code>. Defaults to 1.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h2 id="gpurasterextremapyramid"><code>GPURasterExtremaPyramid</code></h2>
<p>Builds a min and max mip chain of a raster band, each cell holding the extremes of its pixel block.</p>
<p>::lens{scenario=gpu-raster.raster-extrema-pyramid rung=result caption="Highest elevation of each 960 m block, level 1 of the pyramid."}</p>
<p>An exact min/max mip chain (Tevs et al. 2008 maximum mipmaps) for conservative ray skipping. The
<a href="/luma-contributors/docs/api/gpu-terrain">GPU Terrain</a> sight-line and point-horizon contributors use it for
<code>traversal: 'pyramid'</code>.</p>
<ul>
<li><strong>Grid and outputs.</strong> The raster is <code>grid</code> (a <code>GPURasterMetadata</code>; its width and height size the
pyramid, cells live in pixel index space so no cell size applies). The caller-owned views go in
<code>output</code> (<code>GPURasterExtremaPyramidViews</code>): <code>maximum</code>, <code>minimum</code>, or one <code>combined</code> view of
<code>2 * layout.length</code> floats. With <code>combined</code>, <code>pyramid.output</code> (a <code>GPURasterExtremaPyramidOutput</code>,
<code>{layout, combined}</code>) is what the consumers take as their <code>pyramid</code> prop.</li>
<li><strong>Levels.</strong> Level <code>L</code> has blocks of <code>firstBlockSize·2^L</code> pixels (default 4), down to 1×1 or
<code>levelCount</code>. Levels are packed into one float32 view per extremum, with offsets from
<code>getGPURasterExtremaPyramidLayout(grid, options)</code>.</li>
<li><strong>Footprint.</strong> With <code>footprint: 'bilinear'</code> (the default), each cell also covers one extra
pixel column and row. Every bilinear sample whose base pixel lies in the block is then bounded by
that one cell. <code>'cell'</code> gives the plain block extrema.</li>
<li><strong>Validity.</strong> Invalid pixels are excluded. Cells with no valid pixel hold ∓FLT_MAX
(<code>GPU_RASTER_EXTREMA_PYRAMID_EMPTY_*</code>).</li>
<li><strong>Exactness.</strong> The GPU output is bit-identical to a CPU build. Level 0 reduces from pixels and
each higher level reduces the 2×2 cells below in a fixed order.</li>
</ul>
<h4 id="how-it-computes-21">How it computes</h4>
<p>{/* lens:GPURasterExtremaPyramid */}</p>
<p>::lens{scenario=gpu-raster.raster-extrema-pyramid rung=datum datum=48 caption="Block 48, the highest block: its four children at level 0 and their maximum."}
::lens{scenario=gpu-raster.raster-extrema-pyramid rung=formula caption="A parent maximum is the largest of its 2 by 2 children; level 0 reduces valid pixels of a block."}
::lens{scenario=gpu-raster.raster-extrema-pyramid rung=kernel caption="Level 0 scans each block; a fused workgroup pass reduces 16 by 16 tiles through the higher levels."}
::lens{scenario=gpu-raster.raster-extrema-pyramid rung=graph caption="Canonicalize the band, reduce level 0, then one fused pass for the remaining levels."}</p>
<h2 id="related-pages">Related pages</h2>
<ul>
<li><a href="/luma-contributors/docs/api/gpu-raster/operations">GPURaster operations reference</a></li>
<li><a href="/luma-contributors/docs/api/gpu-terrain">GPU Terrain</a></li>
<li><a href="https://luma.gl/docs/api-reference/experimental/gpu-core">GPU Core</a></li>
</ul>`};export{e as default};