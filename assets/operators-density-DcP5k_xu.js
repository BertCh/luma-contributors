var e={slug:`api/gpu-spatial-analysis/operators-density`,title:`GPU Spatial Analysis: Density & interpolation`,html:`<h1 id="density--interpolation">Density &#x26; interpolation</h1>
<h3 id="gpupointdensity"><code>GPUPointDensity</code></h3>
<p>Chunked positions and weights use fixed tile scratch for packed hexagon/masked routes, controlled by
scratchTileSize. Shared grid outputs remain packed; fixed-point scale and counts span all chunks.</p>
<p>Bins points into grid or hexagon cells and writes a per-cell count, sum or mean raster.</p>
<p>::lens{scenario=gpu-spatial-analysis.point-density rung=result caption="2,344 observations (grey) binned into 125 m cells (blue); counts per cell, not per km²."}</p>
<p>Bins points into a square grid (<code>GPUGridBinning</code>, <code>GPUGridAggregation</code>) or pointy-top hexagons
(keys kernel plus <code>GPUGroupAggregation</code>) and writes per-cell counts, sums, means, a heatmap field,
its <code>[min, max]</code> extent, a histogram, and an optional <code>r32float</code> texture. Bounds and hexagon radius
may be GPU views; the field can be smoothed with a per-frame <code>GPUConvolution</code> kernel. Instead of a full
<code>kernel</code>, pass a separable <code>separableKernel: {horizontal, vertical}</code> (one of the two is required), which smooths
in two 1D passes.</p>
<p><strong>This is binned counting, not a kernel density estimate.</strong> <code>output.values</code> is the count (or weight
sum or mean) per cell, so it changes with <code>gridSize</code> and is not an intensity per unit area or a
probability density. Optional <code>smoothing</code> convolves that field with a kernel whose radius and sigma
are in <strong>cells</strong>, not ground units, using a zero boundary: kernel mass leaks out of the domain, so the
field is underestimated within about two sigma of the border, and a grid whose cells are not square on
the ground smooths anisotropically. For intensity per unit area, a bandwidth in ground units, a
choice of kernels, and edge correction use <a href="#gpukerneldensity"><code>GPUKernelDensity</code></a>. <code>spatialContext</code>
is planar/native only (omitted means planar/native); longitude/latitude throws, so project first.
It measures area, so <code>local-azimuthal-equidistant</code> frames throw.</p>
<p>Weighted sums default to <code>sumAccumulation: 'auto'</code>, which always selects the deterministic
<code>'workgroup'</code> path, because reproducible sums are the contract. <code>'atomic'</code> (direct float
compare-exchange atomics, no key pass) is an explicit opt-in for known diffuse input and is not
deterministic: its last float bits depend on the atomic interleaving. Each 256-row
workgroup pre-aggregates in shared memory before writing to the grid, so a hotspot no longer
serializes on one cell (90% of 1M points in one cell drops from seconds to about 4 ms, at about 2 ms
extra on uniform data). <code>'workgroup'</code> sums are fixed point: each weight is quantized to one
power-of-two unit derived from the largest weight and the row count, then added as an exact 96-bit
integer, so the sums are bitwise identical across runs, row orders and chunkings and are the f32
nearest to the exact sum of the quantized weights. Weights smaller than about <code>2^(h - 71)</code> times
the largest (<code>h</code> is the bit length of the row count, so about <code>1e-15</code> at a million rows) round to
the unit. <code>'atomic'</code> sums can differ from a CPU sum, and between runs, in the last bits.</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUPointDensity</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  positions, weights, bounds: viewport.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">importToGraph</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(graph), gridSize: [</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">256</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">256</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">],</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  statistic: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'mean'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  smoothing: {kernel: kernel.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">importToGraph</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(graph), kernelWidth: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">5</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, kernelHeight: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">5</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">},</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  output: {values, extent, histogram, texture}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span></code></pre>
<p><code>mask</code> is an optional per-row <code>uint32</code> view with the same length as <code>positions</code>. Rows whose mask
is <code>0</code> are excluded from count, sum, mean, smoothing input, extent and histogram; any nonzero value
includes the row, like the mask inputs of <code>GPURegionStatistics</code> and <code>GPUFlowAggregation</code>. Contents
are per-frame and rewriting them never recompiles. Hexagon binning reads the mask in its keys
kernel; grid binning adds one pass that writes masked positions as NaN into a transient copy.
Without <code>mask</code> the graph is unchanged. Pass a residency arena <code>liveMask</code>, or
<code>GPUTimeWindowFilter.outputMask</code>, to get time- or tile-gated density without poisoning position
columns.</p>
<p><code>getGPUPointDensityHexagonGridSize</code> sizes a hexagon lattice that covers given bounds, and
<code>GPU_POINT_DENSITY_HEXAGON_WGSL</code> lets render shaders place hexagon bins.</p>
<h4 id="how-it-computes">How it computes</h4>
<p>{/* lens:GPUPointDensity */}</p>
<p>::lens{scenario=gpu-spatial-analysis.point-density rung=datum datum=3882 caption="Inspect cell 3882, containing the first iNaturalist observation."}</p>
<p>::lens{scenario=gpu-spatial-analysis.point-density rung=formula caption="Count accepted observations in each grid cell."}</p>
<p>::lens{scenario=gpu-spatial-analysis.point-density rung=kernel caption="Read the count-to-float finalization kernel."}</p>
<p>::lens{scenario=gpu-spatial-analysis.point-density rung=graph caption="Follow grid binning, aggregation and count publication."}</p>
<h3 id="gpustudywindowpreparation"><code>GPUStudyWindowPreparation</code></h3>
<p>Prepares a polygon study window once: oriented edges, bounds, area, perimeter and sampling cells for window-aware statistics.</p>
<p>::lens{scenario=gpu-spatial-analysis.study-window-preparation rung=result caption="13 community areas prepared as one study window (blue outline) in an equal-area frame."}</p>
<p>Prepares one valid polygon or multipolygon, including holes and disconnected islands, as a reusable
<code>studyWindow</code> handle for point-pattern statistics and kernel-density edge correction. Add the
preparation to the graph before adding its consumers. One handle belongs to one graph; declaring
it a second time throws, including in the same graph. <code>isDeclaredIn(graph)</code> reports its declaration.</p>
<p>The first submitted encoding captures a persistent geometry snapshot. <code>invalidate()</code> or a true
CPU <code>rebuildWhen(parameters)</code> requests a new generation. Its applied generation is committed only
after the complete preparation executes, so discarded encodings and abandoned prefixes retry
automatically. Published preparation stages replay against that snapshot every encoding; geometry
refresh alone is GPU-gated. This repeats preparation work and adds one copy of each geometry view
plus 24 generation-control bytes, with the same view lengths and storage binding limits.
Changed geometry needs invalidation, otherwise consumers deliberately reuse the previous geometry.
Consumers evaluate their
current inputs on every encoding. Changing view lengths requires a new handle and graph.
<code>destroy()</code> frees the handle's buffers and allows re-adding it; destroy compiled graphs using it
first. <code>getViews(graph)</code> throws until this handle has been declared in that graph.</p>
<p>The first encoding and first encoding after explicit invalidation bypass <code>rebuildWhen</code>; other
encodings call it once. <code>encodedBuildCount</code> counts CPU refresh-request encodings, including
discarded requests; automatic GPU retries and replayed builds do not increment it.</p>
<p>The props and handle accept a frame type parameter, for example
<code>GPUStudyWindowPreparation&#x3C;{geometryChanged: boolean}></code>; use that type on <code>GPUCommandGraph</code> too.
The default is <code>unknown</code>; <code>invalidate()</code> alone also works with a void graph. See the
<a href="/luma-contributors/docs/api/gpu-spatial-analysis/concepts">execution guide</a> for a typed <code>rebuildWhen</code> example. No additional invalidation is
needed after discarding an encoding. Use one active resumable execution per handle/graph.</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> studyWindow</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUStudyWindowPreparation</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  positions: windowPositions,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  featureOffsets: windowFeatureOffsets, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// exactly two entries: one feature</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  polygonOffsets: windowPolygonOffsets,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  ringOffsets: windowRingOffsets,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  spatialContext,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  samplingGrid: [</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">256</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">256</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">],</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  output: {summary: windowSummary} </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// optional caller-owned float32[8] readback copy</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">});</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(studyWindow);</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// Pass studyWindow to consumers added after this preparation.</span></span></code></pre>
<p>The geometry uses the GeoArrow polygon layout: the first ring of each polygon is its shell,
regardless of winding. Only planar frames are supported. Project longitude/latitude positions first
with <code>GPULocalFrame.projectArray</code>, using the same frame as the observations. The polygon must be
valid; consider <code>GPUGeometryValidity</code>. Self-intersections or overlapping polygons invalidate the
fan identity used by consumers.</p>
<p><code>GPURectangleClip</code> polygon output can feed the preparation directly, including empty polygon and
ring slots. A ring with fewer than three vertices, or exactly zero orientation for every triangle
<code>(first, vertex, next)</code> in its fan, is degenerate. Such rings contribute no area, perimeter, edge
count, bounding box or uncertainty flag; their oriented edges are NaN. The constructor still needs
at least three vertex rows and one ring slot. An all-degenerate window is legal and produces EMPTY
at encode time. The raster sampler reads raw positions, so degenerate rings can only add boundary
flags (extra exact inclusion tests), never change coverage.</p>
<p>The GPU copies oriented edges with the interior on their left and computes the eight-float
<code>GPU_STUDY_WINDOW_SUMMARY</code> row: <code>[minX, minY, maxX, maxY, area, perimeter, edgeCount, flags]</code>.
Area subtracts holes; perimeter includes them. Flag 1
(<code>GPU_STUDY_WINDOW_FLAG_ORIENTATION_UNCERTAIN</code>) indicates a ring area within a conservative f32
rounding bound of zero, or non-finite orientation; flag 2 (<code>GPU_STUDY_WINDOW_FLAG_EMPTY</code>) indicates
non-finite or non-positive total area. Ring areas use a fixed pairwise reduction of determinants
relative to the first vertex. The orientation bound is
<code>8 * epsilon_f32 * (1 + ceil(log2(vertexCount))) * sum(abs(products))</code> on doubled area.</p>
<p><code>samplingGrid</code> is compile-time, defaults to <code>[256, 256]</code>, and must contain positive integers with a
product no larger than 1048576. The sampler compacts inside or boundary cells in row-major order;
bit 31 marks a boundary cell that needs an inclusion test. <code>crossingCapacity</code> is forwarded to
<code>GPUPolygonRasterization</code> and defaults to <code>4 * samplingGrid[1] * ringCount</code>; size it for polygons
with many crossings per row. On crossing overflow, a slower exact GPU center test preserves
sampler coverage. <code>slabEntryCapacity</code> controls the optional edge index, whose overflow
falls back to walking edges. <code>getViews(graph)</code> exposes the edge, summary, slab and sampler views;
its packed view is internal to consumers.</p>
<p>All preparation runs on the GPU. Each rebuild's fixed ring tree costs O(V log V); sampler cost
also depends on the grid and crossing capacity.</p>
<h4 id="how-it-computes-1">How it computes</h4>
<p>{/* lens:GPUStudyWindowPreparation */}</p>
<p>::lens{scenario=gpu-spatial-analysis.study-window-preparation rung=datum datum=4 caption="Inspect summary field 4: the prepared window area in square metres."}</p>
<p>::lens{scenario=gpu-spatial-analysis.study-window-preparation rung=kernel caption="Read edge orientation and summary kernels."}</p>
<p>::lens{scenario=gpu-spatial-analysis.study-window-preparation rung=graph caption="Follow edge preparation, slab indexing and sampling-cell compaction."}</p>
<h3 id="gpukerneldensity"><code>GPUKernelDensity</code></h3>
<p>Explicit method binned accepts aligned chunked positions and weights. Corrected-weight scalar
scratch splits when a global binding cannot fit; split totals use fixed-point accumulation. Exact
and auto retain packed indexed inputs, and shared grid outputs must fit.</p>
<p>Estimates event intensity on a raster from points with a planar kernel, a ground-unit bandwidth and edge correction.</p>
<p>::lens{scenario=gpu-spatial-analysis.kernel-density rung=result caption="2,344 iNaturalist observations (grey) and their 400 m gaussian density on 125 m cells."}</p>
<p>Maximum: <code>indexGridSize</code> has at most <code>getGPUGridIndexMaximumCellCount(device)</code> (<code>2^25 - 1</code> on a default device: <code>cells + 1</code> uint32 cell offsets in one 128 MiB binding) cells.</p>
<p>A true planar kernel density estimate on a raster, with the bandwidth in ground units. For each output
cell center <code>u</code> it evaluates <code>lambda(u) = sum_i w_i K_h(|u - x_i|)</code> over the points inside the kernel
support, where <code>K_h(d) = K(d / h) / h^2</code> is radially symmetric and integrates to one over the plane
(Silverman 1986, Density Estimation, Table 3.1):</p>
<table>
<thead>
<tr>
<th><code>kernel</code></th>
<th><code>K_h(d)</code> for <code>t = d^2 / h^2</code></th>
<th>Support</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>'gaussian'</code> (default)</td>
<td><code>exp(-t / 2) / (2 pi h^2)</code></td>
<td>truncated at <code>5h</code>; lost mass <code>exp(-12.5)</code>, 3.7e-6</td>
</tr>
<tr>
<td><code>'quartic'</code> (biweight), alias <code>'bisquare'</code></td>
<td><code>3 / pi (1 - t)^2 / h^2</code></td>
<td><code>t &#x3C; 1</code></td>
</tr>
<tr>
<td><code>'epanechnikov'</code></td>
<td><code>2 / pi (1 - t) / h^2</code></td>
<td><code>t &#x3C; 1</code></td>
</tr>
</tbody>
</table>
<p><code>'bisquare'</code> is the name <code>GPUNeighborSearch</code> and <code>GPUSpatialWeightsTransform</code> use for the same shape (their weights are
normalised in one dimension, <code>15/16 (1 - z^2)^2</code>, where the density kernel is normalised over the plane). KDE accepts
either name for <code>kernel</code>; both run the same kernel and <code>kernel</code> reads back as <code>'quartic'</code>.</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> kdeParameters</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUParameterBuffer</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(device, {id: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'kde'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, format: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'float32'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, length: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">9</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">});</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">  new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUKernelDensity</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    positions, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// float32x2, planar frame with equal units on both axes</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    weights, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// optional float32 (populations, counts); NaN or negative excludes the point</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    parameters: kdeParameters.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">importToGraph</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(graph),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    grid, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// axis-aligned GPURasterMetadata; its width and height size the raster</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    indexGridSize: [</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">64</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">64</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">], </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// internal GPUGridIndex over the points</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    indexBounds: [minX, minY, maxX, maxY], </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// must cover the points that should count</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    kernel: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'quartic'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    edgeCorrection: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'uniform'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// default</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    normalization: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'intensity'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// default</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">    // bandwidthRule: 'scott', // optional: bandwidth computed on the GPU, output.bandwidth reads it</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    output: {values, weightSum, status: {invalidCount}} </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// weightSum and status are optional</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  })</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">);</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">kdeParameters.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">write</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">  getGPUKernelDensityParameterValues</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({bandwidth: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">250</span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D"> /* meters */</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}, grid) </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// no recompile</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">);</span></span></code></pre>
<ul>
<li><strong>Grid.</strong> <code>grid</code> is a <code>GPURasterMetadata</code>: its <code>width</code> and <code>height</code> size the kernels (compile-time)
and the packer reads the placement from the same object, so pass the frame's grid to
<code>getGPUKernelDensityParameterValues(settings, grid)</code>. It must be axis-aligned (a rotated or sheared grid
throws). Output rows follow the grid's row order: row 0 is the grid's first row, the top row of a
north-up grid (<code>makeGPURasterMetadata</code> default, <code>affine[4] &#x3C; 0</code>) and <code>minY</code> when rows advance north
(<code>rowDirection: 'north'</code>). Cell <code>(x, y)</code> is evaluated at its center, from the outer corner of
cell (0, 0), half a cell in with <code>pixelInterpretation: 'point'</code>.</li>
<li><strong>Units.</strong> <code>normalization: 'intensity'</code> (default) is the sum of weights per unit area, position
units to the power -2: points per square meter in a meter frame, so the value does not change with
the raster resolution. Multiply by a cell area to get expected weight per cell. <code>'probability'</code>
divides by the sum of included weights (<code>output.weightSum</code>), giving a probability density that
integrates to about one over the window (exactly one only without edge correction and with the
whole kernel inside the window). Cells are evaluated at their centers, not averaged over the cell.</li>
<li><strong>Edge correction</strong> repairs the downward bias within one bandwidth of the study window, where kernel
mass falls outside the data region. <code>'uniform'</code> (default) divides each cell by <code>e(u)</code>, the kernel
mass inside the window (Diggle 1985; spatstat <code>density.ppp(edge = TRUE)</code>). <code>'diggle'</code> instead
weights each point by <code>1 / e(x_i)</code> (Jones 1993; spatstat <code>diggle = TRUE</code>), which keeps the total
closer to the point count but is noisier at the edge. <code>'none'</code> is the plain sum. The window is
<code>window</code> in <code>getGPUKernelDensityParameterValues</code> and defaults to the rectangle of the grid; with correction, points
outside the window are excluded and cells whose centers are outside are NaN. <code>e(u)</code> is closed form,
with no quadrature: normal-CDF differences for the gaussian, and an exact inclusion-exclusion of
four disk-clipped quadrant integrals for quartic and epanechnikov. <code>getGPUKernelDensityWindowMass</code>
is the float64 rectangle reference. For a polygon, pass a <code>GPUStudyWindowPreparation</code> as the
compile-time <code>studyWindow</code> prop (default <code>'rectangle'</code>) and add it to the graph before the KDE.
Set <code>window: 'study-window'</code> in the parameter packer; slots 5–8 become NaN and polygon kernels
read the prepared window instead. A rectangle-mode operator given this sentinel with edge
correction produces NaN cells. Polygon windows work with <code>'exact'</code>, <code>'binned'</code> and <code>'auto'</code>.
Outside points are excluded and cells with centers outside W are NaN. Polygon <code>e(u)</code> uses
oriented fan decomposition, with analytic compact-kernel integrals and Gaussian quadrature
(2e-5 relative accuracy); only edges within the kernel support contribute corrections.
Uniform correction divides by <code>e(u)</code>; Diggle correction uses <code>w_i / e(x_i)</code>. Both retain the
rectangle small-mass guard (<code>mass > 1e-6</code>). The window must be valid (consider
<code>GPUGeometryValidity</code>) and in the same planar frame as the KDE. Window queries use absolute
frame coordinates, without subtracting <code>gridStart</code>. Polygon plus <code>edgeCorrection: 'none'</code>
throws <code>\${id} studyWindow is ignored without edge correction; set edgeCorrection to 'uniform' or 'diggle'</code>.</li>
<li><strong>Bandwidth.</strong> It is per-frame, in position units, and should reflect the question (a service
distance, a neighborhood). Pass <code>bandwidthRule: 'scott'</code> or <code>'silverman'</code> to have the GPU compute a
normal-reference bandwidth from the points every encoding, with no read-back: <code>h = n_eff^(-1/6) sqrt((variance x + variance y) / 2)</code>, the pooled <code>isotropic</code> bandwidth of <code>scipy.stats.gaussian_kde</code>
with the unbiased weighted covariance (<code>numpy.cov(aweights = w)</code>) and <code>n_eff = (sum w)^2 / sum w^2</code>.
In two dimensions the two rules coincide (the IQR-based 1D Silverman rule is not used). Rows with a
non-finite position or a non-finite or negative weight are skipped; <code>indexBounds</code> and the window are
not applied. The moments are a two-stage hierarchical reduction (workgroup partials, then further
levels) in two passes (mean, then centered sums) relative to the grid's start corner, so large projected
coordinates do not cancel. <code>parameters</code> slot 4 is then ignored, and <code>output.bandwidth</code> (optional,
one float32 row) receives the value the kernels use (NaN, and an all-NaN raster, when undefined: fewer
than two rows, <code>n_eff &#x3C;= 1</code>, or zero spread). A normal-reference rule oversmooths clustered data;
likelihood cross-validation and Diggle's selector are not provided.</li>
<li><strong>Exclusions.</strong> A non-finite position or weight, a negative weight, and a position outside
<code>indexBounds</code> exclude the row; <code>output.status.invalidCount</code> is their
number. Zero-weight rows are included.</li>
<li><strong>Determinism and precision.</strong> The index is rebuilt every encoding, IDs are put back in ascending
row order inside each index cell by a stable sort of the rows by cell (linear in the row count, however many rows share a cell), and each cell sums in a fixed order (index row, column, ascending row), with no float
atomics. Distances use offsets from the grid's start corner to limit f32 cancellation; the f32 inputs
themselves are not recentered, so use local coordinates for fine resolution far from the origin.</li>
<li><strong>Large bandwidths (<code>method</code>).</strong> The default <code>'exact'</code> gather costs <code>(5h / cell)^2</code> cells per point and
explodes when the bandwidth spans many cells. <code>method: 'binned'</code> (gaussian only; other kernels throw)
splits each included weight over the four nearest cell centers (linear, cloud-in-cell binning, Wand
1994) with the shared deterministic fixed-point accumulator, then blurs the bins along x and y with
three running-sum box blurs and one two-sided exponential pass per axis, variance-matched to
<code>(h / cell)^2</code> (Wells 1986; Kovesi 2010; the exponential cancels the boxes' fourth cumulant). The
cost is a few passes over the grid whatever the bandwidth; every bandwidth-dependent filter parameter
is computed on the GPU, so changing <code>h</code> never recompiles. Edge correction, normalization and NaN
rules are the same as <code>'exact'</code>. <code>method: 'auto'</code> builds both and the GPU picks one every encoding:
exact while <code>5h</code> is at most 32 cells, binned above. Error versus <code>'exact'</code>, for <code>h >= 3</code> cells, in
units of <code>W / (2 pi h^2)</code> (<code>W</code> the included weight, the height of one point's kernel): at most
<code>r^2 / 4 + 0.016 + 1.4e-5 + 6e-8 (13 / r + 210)</code> with <code>r = cell / h</code> (<code>getGPUKernelDensityBinnedErrorBound</code>):
<code>r^2 / 4</code> from the linear binning, 0.016 from the filter shape, the rest from truncation and float32.
For example 1.9% at <code>h = 10</code> cells and 1.6% at <code>h = 64</code>; smooth fields err far less than one point. The
blur drops the mass that leaves its array, so it needs a margin of <code>2.5h</code> around the cells you read:
<code>binnedPadding</code> (default: the larger grid side, so <code>h</code> up to 40% of it) adds that margin and also
extends the region whose points are counted. <code>output.status.insufficientPadding</code> flags a <code>'binned'</code>
run that lacked it; <code>'auto'</code> falls back to the exact gather instead. The array costs
<code>(width + 2 binnedPadding + 2) * (height + 2 binnedPadding + 2)</code> cells of 20 bytes.</li>
<li>Pinned to <code>sklearn.neighbors.KernelDensity</code> (gaussian, epanechnikov), numpy (quartic), and
<code>scipy.integrate.dblquad</code> (window mass, edge-corrected fields); bandwidth rules to <code>scipy.stats.gaussian_kde</code>.</li>
<li><code>spatialContext</code> is planar/native only (omitted means planar/native). It measures area, so <code>local-azimuthal-equidistant</code> frames throw.</li>
</ul>
<p>Not provided: adaptive or space-time bandwidths, network KDE (see the network contributors), reflection
boundaries.</p>
<h4 id="how-it-computes-2">How it computes</h4>
<p>{/* lens:GPUKernelDensity */}</p>
<p>::lens{scenario=gpu-spatial-analysis.kernel-density rung=datum datum=4887 caption="One cell near Montrose Harbor: its kernel sum divided by the window mass."}</p>
<p>::lens{scenario=gpu-spatial-analysis.kernel-density rung=formula datum=4887 caption="Gaussian kernel truncated at 5h, divided by the in-window kernel mass e_h(u)."}</p>
<p>::lens{scenario=gpu-spatial-analysis.kernel-density rung=kernel caption="The gather kernel: one invocation per output cell visits grid-index rows within 5h."}</p>
<p>::lens{scenario=gpu-spatial-analysis.kernel-density rung=graph caption="Grid index build, then the gather pass; no readback between them."}</p>
<p>{/* parameters: GPUKernelDensity generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h4 id="gpukerneldensity-parameters"><code>GPUKernelDensity</code> parameters</h4>
<ul>
<li>Packer: <code>getGPUKernelDensityParameterValues(settings, grid)</code>, writes <code>GPU_KERNEL_DENSITY_PARAMETER_LENGTH</code> (9) <code>float32</code> values.
<ul>
<li><code>grid</code> (<code>GPURasterMetadata</code>): The output grid: axis-aligned, with a nonzero finite cell size; its width and height must match the contributor's.</li>
</ul>
</li>
</ul>
<p>Settings of <code>getGPUKernelDensityParameterValues</code>:</p>
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
<td><code>bandwidth</code></td>
<td><code>number</code></td>
<td>Kernel bandwidth <code>h</code> in position units (ground units). Positive and finite. Ignored (the GPU computes it) when <code>GPUKernelDensityProps.bandwidthRule</code> is set; pass any positive value.</td>
</tr>
<tr>
<td><code>window</code> (optional)</td>
<td><code>GPUKernelDensityWindow | 'study-window'</code></td>
<td>Study window <code>[minX, minY, maxX, maxY]</code> of the edge correction. Defaults to the ordered rectangle of the output grid. Ignored without edge correction. Use <code>'study-window'</code> with a polygon <code>studyWindow</code>; it writes NaN rectangle slots, so a rectangle-mode operator with edge correction yields NaN cells.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h3 id="gpuinversedistanceweighting"><code>GPUInverseDistanceWeighting</code></h3>
<p>Interpolates sample values onto a raster as inverse-distance-weighted means of the nearest samples.</p>
<p>::lens{scenario=gpu-spatial-analysis.inverse-distance-weighting rung=result caption="Helene rain gauges (grey) to a rainfall surface: power 2, up to eight gauges within 150 km."}</p>
<p>Maximum: <code>indexGridSize</code> has at most <code>getGPUGridIndexMaximumCellCount(device)</code> (<code>2^25 - 1</code> on a default device: <code>cells + 1</code> uint32 cell offsets in one 128 MiB binding) cells.</p>
<p>Interpolates scattered samples onto a raster with inverse distance weighting (<code>w = 1 / d^p</code>),
the GPU counterpart of turf <code>interpolate</code>, GDAL <code>invdist</code>, and GRASS <code>v.surf.idw</code>. The output
placement (the <code>grid</code> passed to the packer), search radius, power, nearest-neighbor limit, and
minimum neighbor count are per-frame; the grid's size is compile-time.</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">  new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUInverseDistanceWeighting</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    positions, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// float32x2 per sample</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    values, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// float32 per sample, NaN skips</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    mask, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// optional uint32, 0 skips</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    parameters: idwParameters.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">importToGraph</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(graph), </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// 9 float32</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    grid, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// axis-aligned GPURasterMetadata; its width and height size the raster</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    indexGridSize: [</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">128</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">128</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">], </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// internal GPUGridIndex over the samples</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    indexBounds: [minX, minY, maxX, maxY], </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// samples outside are ignored</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    maximumNeighborCount: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">16</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// compile-time capacity for k; 0 compiles radius-only</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    output: {values: surface, counts} </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// float32 (NaN = nodata), optional uint32</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  })</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">);</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">idwParameters.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">write</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">  getGPUInverseDistanceWeightingParameterValues</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    {</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">      searchRadius: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">250</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">      power: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">      taper: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'none'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// or 'modified-shepard' (Franke-Nielson, needs a finite radius)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">      neighborCount: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">8</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// 0 = every sample within the radius</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">      minimumNeighborCount: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">3</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    },</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    grid </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// the frame's placement: pan or zoom by passing a different grid of the same size</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  )</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">); </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// no recompile</span></span></code></pre>
<ul>
<li>Gather form: one invocation per output cell samples the cell center
<code>gridStart + (x + 0.5, y + 0.5) * cellSize</code> and visits the grid-index rows that overlap the
search square. No atomics touch values. <code>grid</code> must be axis-aligned (the packer and constructor
throw otherwise); <code>gridStart</code> is the outer corner of cell (0, 0) and <code>cellSize</code> is signed, so
output rows follow the grid's row order: row 0 is the top row of a north-up grid
(<code>affine[4] &#x3C; 0</code>) and <code>minY</code> when rows advance north.</li>
<li>Samples with <code>d^2 &#x3C;= radius^2</code> contribute. With <code>neighborCount = k > 0</code> the <code>k</code> nearest are kept
by <code>(d^2, row)</code>, so equal distances keep the smallest row. <code>k</code> is clamped to
<code>maximumNeighborCount</code> (at most 64).</li>
<li>An exact hit (<code>d == 0</code>) returns that sample's value, the smallest row among several, regardless
of <code>minimumNeighborCount</code>. Otherwise a cell with fewer than <code>max(minimumNeighborCount, 1)</code>
contributors is nodata (NaN). <code>counts</code> holds the contributors after the <code>k</code> limit.</li>
<li>Negative or NaN radius, or a negative or non-finite power, produces an all-NaN raster.
<code>searchRadius: Infinity</code> searches the whole index domain.</li>
<li>Units and coordinates: by default positions, grid and radius share one planar unit and
distances are differences of offsets from the grid's start corner, so large absolute coordinates (UTM
northings) do not quantize them. With <code>spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'great-circle'}</code> positions are absolute lon/lat degrees, the index and output grid stay
in degrees, and distances and <code>searchRadius</code> use the context's length unit (meters by default;
kilometers with <code>units: 'kilometers'</code>), on a sphere of <code>sphereRadius</code> or the mean Earth radius in that unit. Neighbors and radius
tests order by the haversine term. The query box is conservative (longitude span widened by the
cosine of the highest latitude reached, full span near the poles) but does not wrap the
antimeridian, and the nearest-k limit scans the whole box instead of early-stopping.
Pass optional <code>spatialContext</code> to <code>getGPUInverseDistanceWeightingParameterValues(settings, grid)</code> to warn on short <code>searchRadius</code> values; see <a href="/luma-contributors/docs/api/gpu-spatial-analysis/concepts#precision-of-longitudelatitude-positions">Precision of longitude/latitude positions</a>.
Other longitude/latitude metrics
(rhumb, ellipsoidal, none) throw; project to a planar frame (for example <code>createGPULocalFrame</code>)
for them. It measures distance, so <code>local-lambert-equal-area</code> frames throw.</li>
<li><code>taper: 'none'</code> (default) is <code>w = 1 / d^p</code> inside the radius, a hard cutoff: a sample entering
or leaving the radius changes the surface discontinuously. <code>taper: 'modified-shepard'</code> is the
Franke and Nielson (1980) weight <code>w = ((R - d)+ / (R d))^p</code> with <code>R = searchRadius</code>, which falls
to zero at the radius (a sample exactly at <code>R</code> has weight zero). It needs a finite radius
(<code>getGPUInverseDistanceWeightingParameterValues</code> throws otherwise). <code>counts</code> still counts every
sample inside the radius. Default stays <code>'none'</code> for backward compatibility; prefer the taper
when a finite radius is a modelling choice rather than just a speed limit.</li>
<li>Parameter guidance: <code>power</code> 2 is conventional, not optimal; higher powers flatten the surface
around samples. <code>minimumNeighborCount</code> defaults to 1, so a lone distant sample inside the radius
paints a constant plateau of its value; use 3 or more to leave sparse cells as nodata. IDW
has no direction or sector search and no leave-one-out output; use <code>GPUKriging</code>'s
cross-validation to compare models on the same samples.</li>
<li>Determinism: the grid index's in-cell order depends on atomics, so the contributor re-sorts IDs inside
each index cell every encoding; sums then run in a fixed order (index row, column, ascending
row; or <code>(d^2, row)</code> with <code>k</code>). Weights are accumulated in log space against a running maximum,
so tiny distances with large powers do not overflow f32.</li>
<li>The index is rebuilt every encoding, so positions and values may change per frame. Keep a few
samples per index cell: the in-cell sort is quadratic in cell population.</li>
<li><code>output.values</code> is a row-major float32 raster that <code>GPUTerrainContours</code> accepts directly as
<code>elevation: {id, format: 'float32', storage: {kind: 'buffer', values: surface}}</code>; NaN cells
clear their marching-squares cells.</li>
</ul>
<h4 id="how-it-computes-3">How it computes</h4>
<p>{/* lens:GPUInverseDistanceWeighting */}</p>
<p>::lens{scenario=gpu-spatial-analysis.inverse-distance-weighting rung=datum datum=1659 caption="Inspect cell 1659, containing station USW00013872, Asheville, and its contributing-station count."}</p>
<p>::lens{scenario=gpu-spatial-analysis.inverse-distance-weighting rung=formula caption="Normalize inverse-distance weights; the lowest-row exact hit takes precedence."}</p>
<p>::lens{scenario=gpu-spatial-analysis.inverse-distance-weighting rung=kernel caption="Read nearest-neighbor selection and log-space accumulation."}</p>
<p>::lens{scenario=gpu-spatial-analysis.inverse-distance-weighting rung=graph caption="Follow station indexing, sorting and rainfall gathering."}</p>
<p>{/* parameters: GPUInverseDistanceWeighting generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h4 id="gpuinversedistanceweighting-parameters"><code>GPUInverseDistanceWeighting</code> parameters</h4>
<ul>
<li>Packer: <code>getGPUInverseDistanceWeightingParameterValues(settings, grid)</code>, writes <code>GPU_INVERSE_DISTANCE_WEIGHTING_PARAMETER_LENGTH</code> (9) <code>float32</code> values.
<ul>
<li><code>grid</code> (<code>GPURasterMetadata</code>): The output grid: axis-aligned; its width and height must match the contributor's.</li>
</ul>
</li>
</ul>
<p>Settings of <code>getGPUInverseDistanceWeightingParameterValues</code>:</p>
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
<td><code>spatialContext</code> (optional)</td>
<td><code>GPUSpatialContext</code></td>
<td>Spatial context of the positions; when it is longitude/latitude, a distance under 10 float32 quanta (about 17 m) logs a warning (see Precision of longitude/latitude positions). Does not change the packed values.</td>
</tr>
<tr>
<td><code>searchRadius</code></td>
<td><code>number</code></td>
<td>Search radius in position units. <code>Infinity</code> searches the whole index domain, except with the <code>'modified-shepard'</code> taper, which needs a finite radius.</td>
</tr>
<tr>
<td><code>power</code> (optional)</td>
<td><code>number</code></td>
<td>Distance power <code>p</code> in <code>w = 1 / d^p</code>. Zero gives an unweighted mean. Defaults to 2.</td>
</tr>
<tr>
<td><code>taper</code> (optional)</td>
<td><code>GPUInverseDistanceWeightingTaper</code></td>
<td>Weight taper at the search radius. Defaults to <code>'none'</code> (hard cutoff).</td>
</tr>
<tr>
<td><code>neighborCount</code> (optional)</td>
<td><code>number</code></td>
<td>Nearest-neighbor limit <code>k</code>; zero (default) uses every sample within the radius. Values above the contributor's compile-time <code>maximumNeighborCount</code> are clamped to it.</td>
</tr>
<tr>
<td><code>minimumNeighborCount</code> (optional)</td>
<td><code>number</code></td>
<td>Cells with fewer contributing samples (and no exact hit) are nodata. Defaults to 1.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h3 id="gpukriging"><code>GPUKriging</code></h3>
<p>Predicts values and kriging variance on a raster by solving an ordinary kriging system per cell from nearby samples.</p>
<p>::lens{scenario=gpu-spatial-analysis.kriging rung=result caption="Helene rain gauges to rainfall: spherical variogram, 150 km range; nearer gauges get larger weights."}</p>
<p>Maximum: <code>indexGridSize</code> has at most <code>getGPUGridIndexMaximumCellCount(device)</code> (<code>2^25 - 1</code> on a default device: <code>cells + 1</code> uint32 cell offsets in one 128 MiB binding) cells.</p>
<p>Local ordinary kriging on a raster (PyKrige <code>OrdinaryKriging</code> with <code>n_closest_points</code>, GSTAT, ArcGIS Kriging).
For every output cell the <code>k &#x3C;= 16</code> nearest samples within the radius form a <code>(k + 1) x (k + 1)</code> system
under a fitted variogram, solved per invocation by pivoted Gaussian elimination in f32 (the variogram is
normalized by its total sill and the prediction is relative to the nearest value). The samples come from
the same internal <code>GPUGridIndex</code> as <code>GPUInverseDistanceWeighting</code>.</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">  new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUKriging</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    positions, values, parameters: krigingParameters.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">importToGraph</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(graph),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    grid, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// axis-aligned GPURasterMetadata; its width and height size the raster</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    indexGridSize, indexBounds, maximumNeighborCount: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">16</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    output: {</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">      values: surface, variance, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// float32 rasters</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">      crossValidationPrediction, crossValidationVariance, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// optional float32 per SAMPLE row</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">      status: {invalidCount, uncertainCount} </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// optional uint32 scalars</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    }</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  })</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">);</span></span></code></pre>
<ul>
<li>Props: <code>positions</code>, <code>values</code> (NaN skips), optional <code>mask</code>, <code>parameters</code> (<code>getGPUKrigingParameterValues</code>,
at least 12 float32), <code>grid</code>, <code>indexGridSize</code>, <code>indexBounds</code>, <code>maximumNeighborCount</code> (1 to 16,
compile-time) and <code>output: {values, variance?}</code>.</li>
<li>Per frame: radius, <code>k</code>, minimum neighbors, the placement (the <code>grid</code> passed to
<code>getGPUKrigingParameterValues(settings, grid)</code>, axis-aligned, same width and height as the props; output
rows follow its row order, row 0 at the top of a north-up grid and at <code>minY</code> when rows advance north), model (<code>'spherical'</code>, <code>'exponential'</code> or <code>'gaussian'</code>,
the <code>fitVariogramModel</code> convention) and nugget, sill and range. <code>range</code> is the effective range: the
exponential model is <code>1 - exp(-3 h / range)</code> and the Gaussian model <code>1 - exp(-3 (h / range)^2)</code>, both
at 95 percent of the sill at <code>range</code>. Converting to another library:
<ul>
<li>PyKrige: pass <code>sill + nugget</code> as its "sill" (PyKrige's value is the total sill). Its exponential and
spherical models take the same <code>range</code>. Its built-in Gaussian model is
<code>1 - exp(-(h / (range * 4 / 7))^2)</code>, exponent 3.0625 instead of 3, so pass
<code>range * 7 / (4 * sqrt(3))</code> (about <code>1.0104 * range</code>) to get the same surface.</li>
<li>gstat <code>vgm</code>: the range parameter is <code>range / 3</code> for <code>"Exp"</code>, <code>range / sqrt(3)</code> for <code>"Gau"</code> and
<code>range</code> for <code>"Sph"</code>.</li>
</ul>
</li>
<li><code>values</code> is a row-major raster and nodata is NaN: too few distinct neighbors, an ill-conditioned
system, or an unreliable variance. <code>variance</code> is the kriging variance in squared value units, 0 at exact
hits, NaN at nodata.</li>
<li>Units and coordinates: by default positions, grid, radius and variogram range share one planar
unit; distances are differences of offsets from the grid's start corner (cross-validation: from the
index domain minimum), so UTM-sized coordinates keep full f32 precision relative to the
neighborhood. With <code>spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'great-circle'}</code> positions are absolute lon/lat degrees, the index and grid stay in degrees,
and the search radius, variogram range and all sample-sample and sample-cell distances are
haversine meters (<code>sphereRadius</code>, else 6371008.8 m; kilometers with <code>units: 'kilometers'</code>).
Neighbors are ordered by <code>(haversine term, row)</code>; exact hits and duplicate coalescing compare
degree positions exactly; leave-one-out uses the same metric. The query box is conservative and
does not wrap the antimeridian, and the ring scan's early stop is disabled (the whole box is
scanned, so an unbounded radius is <code>O(samples)</code> per cell).
Pass optional <code>spatialContext</code> to <code>getGPUKrigingParameterValues(settings, grid)</code> to warn on short <code>searchRadius</code> and <code>range</code> values; see <a href="/luma-contributors/docs/api/gpu-spatial-analysis/concepts#precision-of-longitudelatitude-positions">Precision of longitude/latitude positions</a>.
Other longitude/latitude metrics throw. It measures distance, so <code>local-lambert-equal-area</code> frames throw.</li>
<li>Status: <code>invalidCount</code> counts cells that are NaN for lack of data or invalid settings (fewer than
<code>minimumNeighborCount</code> distinct positions, no sample in the radius, outside the index domain).
<code>uncertainCount</code> counts cells rejected by the numerical checks below. Both are computed over the
raster of the last encoding, so a nonzero <code>uncertainCount</code> means the surface has holes that are
not "no data" but "do not trust the solve here".</li>
<li>Singularity is a relative test, not an absolute one: a pivot at or below
<code>KRIGING_RELATIVE_PIVOT_TOLERANCE</code> (1e-6) of the system scale is zero, and a system whose smallest
pivot is below <code>KRIGING_PIVOT_RATIO_TOLERANCE</code> (1e-4) of its largest is rejected as ill conditioned
(the same limit <code>GPULocalKriging</code> and <code>GPULocalRBF</code> apply),
since an f32 solve loses about <code>cond * 6e-8</code> of the weights' accuracy and the ratio tracks <code>1 / cond</code>.
The usual cause is a Gaussian model with a near-zero nugget and closely spaced samples: add a nugget
(0.001 of the sill moves the ratio from 1e-6 to 5e-5 for 12 clustered samples) or thin the samples. The
constants were calibrated against emulated f32 solves compared with float64.</li>
<li>Variance is <code>mu + sum(w * gamma)</code> in units of the total sill and is never negative for a valid
model. A negative value above <code>-KRIGING_NEGATIVE_VARIANCE_TOLERANCE</code> (1e-4) is f32 rounding and is
clamped to zero; a larger negative variance makes the cell NaN (value and variance) and counts it in
<code>uncertainCount</code>. It is never silently clamped.</li>
<li>Duplicate sample positions inside a neighborhood are coalesced into one kriging point whose value is
the mean of their values (summed in <code>(d^2, row)</code> order, so the result is deterministic), instead of
making the system singular. Duplicates still occupy neighbor slots, so a neighborhood of <code>k</code> rows can
hold fewer than <code>k</code> points, and <code>minimumNeighborCount</code> counts distinct positions. An exact hit returns the
mean of the samples at that position with variance 0.</li>
<li>The nugget is treated as micro-scale variation: the predictor is exact at the data even with a
positive nugget (zero variance at exact hits). Filtered kriging with a measurement-error nugget is not
provided.</li>
<li>Leave-one-out cross-validation (<code>crossValidationPrediction</code>, optional <code>crossValidationVariance</code>, one
float32 row per sample) krigs each sample from its <code>k</code> nearest OTHER samples with the same variogram,
radius, <code>k</code> and minimum-neighbor settings and the same solver code. Rows that are NaN, masked or
unestimable are NaN. Another sample at exactly the same position is an exact hit (its mean, variance 0).
Compare models with the standardized error <code>(value - prediction) / sqrt(variance)</code>, which should have
mean near 0 and variance near 1, and the root mean squared error of <code>value - prediction</code>.</li>
<li>A sample <code>mask</code> is folded into a transient value copy first (a one-pass kernel), and the per-cell status
goes to a transient before the counters are accumulated, so every kernel stays within 8 storage buffers.</li>
<li>Pinned to PyKrige 1.7.3 with 8 closest points: exponential, spherical and Gaussian within 2e-5 of
the data scale on values (about 1.4e-3 on values near 70) and 1e-4 on variance relative to
<code>max(1, variance)</code>. The Gaussian fixture uses a custom PyKrige variogram with exponent 3, because
PyKrige's built-in Gaussian has the other range convention (see above); the pivot-ratio, duplicate-coalescing, mask, leave-one-out and
translation fixtures are float64 numpy ordinary kriging (Cressie 1993, eq. 3.2.12 to 3.2.16) that
reproduces the PyKrige numbers to 5e-7. Global and universal kriging, anisotropy, octant search and
filtered (measurement-error) kriging are not provided.</li>
</ul>
<h4 id="how-it-computes-4">How it computes</h4>
<p>{/* lens:GPUKriging */}</p>
<p>::lens{scenario=gpu-spatial-analysis.kriging rung=datum datum=1659 caption="Inspect prediction and variance in cell 1659, containing station USW00013872, Asheville."}</p>
<p>::lens{scenario=gpu-spatial-analysis.kriging rung=formula caption="Solve variogram weights with a Lagrange multiplier; range controls decorrelation, not the search cutoff."}</p>
<p>::lens{scenario=gpu-spatial-analysis.kriging rung=kernel caption="Read duplicate coalescing, pivot checks and variance rejection."}</p>
<p>::lens{scenario=gpu-spatial-analysis.kriging rung=graph caption="Follow station indexing and eight-neighbor ordinary kriging."}</p>
<p>{/* parameters: GPUKriging generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h4 id="gpukriging-parameters"><code>GPUKriging</code> parameters</h4>
<ul>
<li>Packer: <code>getGPUKrigingParameterValues(settings, grid)</code>, writes <code>GPU_KRIGING_PARAMETER_LENGTH</code> (12) <code>float32</code> values.
<ul>
<li><code>grid</code> (<code>GPURasterMetadata</code>): The output grid: axis-aligned; its width and height must match the contributor's.</li>
</ul>
</li>
</ul>
<p>Settings of <code>getGPUKrigingParameterValues</code>:</p>
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
<td><code>spatialContext</code> (optional)</td>
<td><code>GPUSpatialContext</code></td>
<td>Spatial context of the positions; when it is longitude/latitude, a distance under 10 float32 quanta (about 17 m) logs a warning (see Precision of longitude/latitude positions). Does not change the packed values.</td>
</tr>
<tr>
<td><code>searchRadius</code></td>
<td><code>number</code></td>
<td>Search radius in position units. <code>Infinity</code> searches the whole index domain.</td>
</tr>
<tr>
<td><code>neighborCount</code> (optional)</td>
<td><code>number</code></td>
<td>Neighborhood size <code>k</code>. Clamped to the contributor's compile-time <code>maximumNeighborCount</code>. Defaults to that capacity.</td>
</tr>
<tr>
<td><code>minimumNeighborCount</code> (optional)</td>
<td><code>number</code></td>
<td>Cells with fewer contributing samples (and no exact hit) are nodata. Defaults to 3.</td>
</tr>
<tr>
<td><code>variogram</code></td>
<td><code>{ /** Model family. */ model: VariogramModelType; /** Nugget, at least 0. */ nugget: number; /** Partial sill, at least 0. */ sill: number; /** Effective range, positive. */ range: number; }</code></td>
<td>Fitted variogram, for example the result of <code>fitVariogramModel</code>: <code>gamma(h) = nugget + sill * shape(h / range)</code> for <code>h > 0</code>, with the same effective-range convention. The nugget plus the partial sill must be positive.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h3 id="gpulocalkriging"><code>GPULocalKriging</code></h3>
<p>Predicts values and latent variance at query points or quadrature supports by solving a local covariance system per query.</p>
<p>Local kriging in simple, ordinary, universal, external-drift, block and co-kriging forms, with spherical, exponential, Gaussian or Matérn (smoothness 1/2, 3/2, 5/2) covariances. Unlike <code>GPUKriging</code>, which fills a raster from an indexed grid, it takes explicit query rows and scans the samples or a supplied neighborhood. Full reference: <a href="/luma-contributors/docs/api/gpu-spatial-analysis/interpolation-expansion">Bounded GPU interpolation and simulation</a>.</p>
<ul>
<li>Covariance parameters are either compile-time (<code>covariance</code>, <code>coregionalization</code>, <code>coregionalizationModels</code>) or per-frame: supply <code>parameters</code> from <code>getGPULocalKrigingParameterValues</code> to change nugget, sill, range and model between frames without a recompile. Set <code>covarianceStructureCount</code> for per-frame co-kriging. The parameter view counts one more storage binding.</li>
<li>Matérn <code>range</code> is the effective range (correlation <code>exp(-3)</code> at <code>range</code>, like the exponential and Gaussian models); a Matérn 1/2 model equals the exponential model. Pinned to GSTools 1.7.0 correlation values and simple and ordinary kriging, predictions and variances, with <code>len_scale = sqrt(nu) * range / factor</code>.</li>
<li>A system whose pivot ratio is below <code>KRIGING_PIVOT_RATIO_TOLERANCE</code> (1e-4) reports status 2 and writes its pivot ratio to <code>values.z</code>. No nugget floor is applied; see <a href="/luma-contributors/docs/api/gpu-spatial-analysis/interpolation-expansion#conditioning-of-smooth-covariances">Conditioning of smooth covariances</a>.</li>
</ul>
<p>{/* parameters: GPULocalKriging generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h4 id="gpulocalkriging-parameters"><code>GPULocalKriging</code> parameters</h4>
<ul>
<li>Packer: <code>getGPULocalKrigingParameterValues(settings)</code>, writes <code>GPU_LOCAL_KRIGING_PARAMETER_LENGTH</code> (8) <code>float32</code> values.</li>
</ul>
<p>Settings of <code>getGPULocalKrigingParameterValues</code>:</p>
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
<td><code>covariance</code> (optional)</td>
<td><code>GPUPredictionCovariance</code></td>
<td>Covariance of the univariate modes and of the two-by-two <code>coregionalization</code> shortcut.</td>
</tr>
<tr>
<td><code>coregionalization</code> (optional)</td>
<td><code>readonly [number, number, number, number]</code></td>
<td>Two-by-two row-major loading matrix <code>L</code>, with <code>B = L L^T</code>, as in the contributor props.</td>
</tr>
<tr>
<td><code>coregionalizationModels</code> (optional)</td>
<td><code>readonly { covariance: GPUPredictionCovariance; loadings: readonly number[]; }[]</code></td>
<td>One to four covariance structures with row-major square loading factors.</td>
</tr>
<tr>
<td><code>variableCount</code> (optional)</td>
<td><code>number</code></td>
<td>Co-kriging variables, 2..4. Defaults to 2 when loadings are supplied.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h3 id="gpuspacetimekriging"><code>GPUSpaceTimeKriging</code></h3>
<p>Predicts values and latent variance at space-time query points by solving a local space-time covariance system per query.</p>
<p>Local simple or ordinary space-time kriging over the separable, product-sum, metric, sum-metric and Gneiting covariance families (gstat 2.1-6 definitions, and Gneiting 2002), with metric, octant or staged neighbor search. Full reference: <a href="/luma-contributors/docs/api/gpu-spatial-analysis/interpolation-expansion#space-time-kriging">Space-time kriging</a>.</p>
<ul>
<li>Every covariance parameter and the search radius, time window and anisotropy change per frame through <code>parameters</code> without a recompile.</li>
<li>Pass <code>temporalContext</code>, and <code>queryTemporalContext</code> for the query times; both must be <code>float32-relative</code>, with the same unit and epoch.</li>
<li>A system below <code>KRIGING_PIVOT_RATIO_TOLERANCE</code> (1e-4) reports status 2 and writes its pivot ratio to <code>values.z</code>.</li>
<li>Reference tier: every query scans every sample in one invocation, so <code>positions</code> is limited to 1,048,576 rows.</li>
<li>Verified against independent binary64 solves for every family, mode and search; GSTools 1.7.0 for the metric family. <strong>gstat fixtures pending</strong> (R not available).</li>
</ul>
<p>{/* parameters: GPUSpaceTimeKriging generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h4 id="gpuspacetimekriging-parameters"><code>GPUSpaceTimeKriging</code> parameters</h4>
<ul>
<li>Packer: <code>getGPUSpaceTimeKrigingParameterValues(settings)</code>, writes <code>GPU_SPACE_TIME_KRIGING_PARAMETER_LENGTH</code> (28) <code>float32</code> values.</li>
</ul>
<p>Settings of <code>getGPUSpaceTimeKrigingParameterValues</code>:</p>
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
<td><code>covariance</code></td>
<td><code>SpaceTimeVariogramModel</code></td>
<td>Space-time covariance, for example the result of <code>fitSpaceTimeVariogramModel</code>.</td>
</tr>
<tr>
<td><code>searchAnisotropy</code> (optional)</td>
<td><code>number</code></td>
<td>Spatial units per unit of time of the neighbor metric <code>sqrt(h^2 + (anisotropy u)^2)</code>. Defaults to the <code>anisotropy</code> of a metric or sum-metric covariance; required otherwise.</td>
</tr>
<tr>
<td><code>searchRadius</code> (optional)</td>
<td><code>number</code></td>
<td>Largest spatial lag of a neighbor. Defaults to infinity.</td>
</tr>
<tr>
<td><code>maximumTimeLag</code> (optional)</td>
<td><code>number</code></td>
<td>Largest absolute time lag of a neighbor. Defaults to infinity.</td>
</tr>
<tr>
<td><code>mean</code> (optional)</td>
<td><code>number</code></td>
<td>Known mean of <code>simple</code> mode. Defaults to 0.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h3 id="gpudotdensity-and-gpurandompointsinpolygon"><code>GPUDotDensity</code> and <code>GPURandomPointsInPolygon</code></h3>
<p><code>GPUDotDensity</code>: Normalized features are bounded by 16,777,215. The triangle CDF uses segmented
saturating counts; packed outputs must satisfy logical and co-bound aligned-range checks.</p>
<p><strong><code>GPUDotDensity</code></strong>: Places seeded random dots inside polygons in proportion to a count column, one dot per stated number of units.</p>
<p>::lens{scenario=gpu-spatial-analysis.dot-density rung=result caption="199 tracts (grey) to one dot per 500 residents (blue); dots are not where people live."}</p>
<p><strong><code>GPURandomPointsInPolygon</code></strong>: Samples a requested number of seeded, uniformly distributed points inside each polygon.</p>
<p>::lens{scenario=gpu-spatial-analysis.random-points-in-polygon rung=result caption="13 community areas (grey) with 20 seeded uniform points each (blue)."}</p>
<p><strong><code>GPURandomPointsOnLine</code></strong>: Samples seeded points along lines, uniformly per unit of arc length.</p>
<p>::lens{scenario=gpu-spatial-analysis.random-points-on-line rung=result caption="721 street edges (grey) with two seeded points each (blue), uniform along length."}</p>
<p>Maximum: <code>GPUDotDensity</code> holds at most <code>getGPUDotDensityMaximumCapacity(device)</code> (8388608 on a default device) dots.</p>
<p>Dasymetric dot-density maps: <code>value * dotsPerUnit</code> random dots inside each polygon, per category,
optionally concentrated by a weight raster such as built-up area. Changing the dot value with zoom
only adds or removes dots; existing dots never move, so there is no flicker.
<code>GPURandomPointsInPolygon</code> places an integer count of uniform random points per feature. Its internal triangulation limits the polygon layout to <code>getGPUPolygonTriangulationMaximumNodeCount(device)</code> scratch nodes (see <code>GPUPolygonTriangulation</code>).</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> parameters</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUParameterBuffer</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(device, {id: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'dots'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, format: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'uint32'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, length: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">8</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">});</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">  new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUDotDensity</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    positions, featureOffsets, polygonOffsets, ringOffsets, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// four-level polygon layout</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    values, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// float32, featureCount * categoryCount rows, feature-major</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    categoryCount: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">3</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    parameters: parameters.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">importToGraph</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(graph),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    mask: {weights, width: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">512</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, height: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">512</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// optional, weights in [0, 1]</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    output: {positions, dots: {ids, status: {count, overflow, requiredCount}}, categories, status: {failedCount}}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  })</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">);</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">parameters.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">write</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">  getGPUDotDensityParameterValues</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({seed: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, dotsPerUnit: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> /</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8"> peoplePerDot, maskExtent: [x0, y0, cw, ch]})</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">);</span></span></code></pre>
<ul>
<li>A slot (feature, category) draws <code>ceil(value * dotsPerUnit - u)</code> dots with a per-slot Philox
uniform <code>u</code>: <code>floor(x)</code> dots plus one more with probability <code>frac(x)</code>, monotone in <code>dotsPerUnit</code>.</li>
<li>Dot <code>j</code> of a slot is placed exactly, keyed by Philox <code>(seed, slot, j, attempt)</code> and independent of the
dot value and of other slots, so coarser dot values give a stable prefix of the finer dots. Every
polygon is triangulated on the GPU (<code>GPUPolygonTriangulation</code>; the first ring of a polygon is the
shell, later rings are holes), a triangle is picked in proportion to its area (binary search of the
feature's cumulative areas, 32-bit fixed point) and the point is uniform in it. Thin or concave
polygons in a large bounding box lose no dots. With a mask a candidate is also kept with probability
equal to its cell weight, and a masked dot is retried up to <code>maximumAttempts</code> times (default 32;
ignored without a mask).</li>
<li>Dots are ordered by feature, category and rank (offsets from a prefix scan). <code>dots.ids</code> holds the
feature row of each dot and <code>dots.status.count</code> is safe as an instance count; rows past it are NaN /
<code>0xffffffff</code>.</li>
<li>Compact <code>status.overflow</code> reports final output capacity only. Optional <code>candidateOverflow</code>
reports a per-slot/per-line count clamped before sampling; <code>requiredCount</code> counts retained
requests and is then a lower bound. The uint32 prefix bound limits each slot to
<code>min(capacity + 1, floor(0xffffffff / slotCount), 2^31)</code> requests. With 65,536 slots and capacity
65,536, a slot requesting 65,536 retains 65,535 and sets candidateOverflow, while final overflow
stays zero. Growing the output alone cannot remove the prefix bound.
When that per-slot limit is at or below final capacity, construction requires
<code>dots.status.candidateOverflow</code> (or <code>points.status.candidateOverflow</code> for line sampling),
because clamping could otherwise leave a short result with no visible loss flag.</li>
<li>Empty polygons and degenerate rings from clipping are skipped on the GPU. A ring is degenerate
when it has fewer than three vertices or every first-vertex fan triangle has exact orientation zero,
matching <code>GPUStudyWindowPreparation</code>; an uncertified orientation is retained. A degenerate shell
makes its entire polygon empty. Removing these entries while retaining feature rows preserves
triangle selection and Philox draws bitwise.</li>
<li>Features with zero total area get zero slots, so they consume no capacity and contribute zero to
<code>slotCounts</code>, <code>dots.status.count</code> and <code>dots.status.requiredCount</code>. Their clamped requested dot counts
are reported in <strong><code>status.failedCount</code></strong>, without an exception.</li>
<li><strong><code>status.failedCount</code></strong> also counts other placement failures, which keep a NaN slot: untriangulable
polygons, exceeded triangulation budgets, non-finite areas and masked dots whose attempts all missed
(probability <code>(1 - w)^maximumAttempts</code> for mean mask weight <code>w</code>). A feature with no requested dots
reports nothing.</li>
<li><strong>Coordinates.</strong> Dots are uniform in coordinate space, so <code>spatialContext</code> is planar/native only (omitted
means planar/native), on <code>GPUDotDensity</code>, <code>GPURandomPointsInPolygon</code> and <code>GPURandomPointsOnLine</code>.
Longitude/latitude throws. <code>GPUDotDensity</code> and <code>GPURandomPointsInPolygon</code> declare <code>measure: 'area'</code>:
an azimuthal-equidistant local frame throws; project into a Lambert equal-area frame.
<code>GPURandomPointsOnLine</code> declares <code>measure: 'distance'</code>: a Lambert equal-area local frame throws;
use an azimuthal-equidistant frame.</li>
<li>Ring roles follow GeoArrow position, not an even-odd test: ring 0 of each polygon is the shell and
later rings are holes (either winding). The triangulation scratch is about 56 bytes per input vertex.</li>
</ul>
<p><strong><code>GPURandomPointsOnLine</code></strong> places <code>counts</code> points per line uniformly in arc length (<code>GeoSeries.sample_points</code>
on lines). The Philox key is <code>(line, rank, 0, 0)</code> with key <code>(seed, 3)</code>, so the prefix of points is stable
as counts grow. Outputs are a bounded <code>positions</code> buffer and a compact <code>points</code> record (<code>ids</code> = line row,
<code>status</code> with <code>count</code>, <code>overflow</code>, optional <code>candidateOverflow</code>, <code>requiredCount</code>), plus optional <code>fractions</code>. Parameters reuse
<code>getGPUDotDensityParameterValues</code> (<code>seed</code>). Points matched <code>LineString.interpolate</code> at the pinned Philox
fractions to 2e-5 relative. Not provided: <code>cluster_poisson</code> sampling.</p>
<h4 id="how-it-computes-5">How it computes</h4>
<p>{/* lens:GPUDotDensity */}</p>
<p>::lens{scenario=gpu-spatial-analysis.dot-density rung=datum datum=0 caption="Inspect tract 17031010100: population input and allocated dot count."}</p>
<p>::lens{scenario=gpu-spatial-analysis.dot-density rung=kernel caption="Read stochastic rounding and area-weighted triangle sampling."}</p>
<p>::lens{scenario=gpu-spatial-analysis.dot-density rung=graph caption="Follow triangulation, allocation, sampling and failure reporting."}</p>
<h4 id="how-it-computes-6">How it computes</h4>
<p>{/* lens:GPURandomPointsInPolygon */}</p>
<p>::lens{scenario=gpu-spatial-analysis.random-points-in-polygon rung=datum datum=0 caption="Inspect Rogers Park: requested points and allocated slots."}</p>
<p>::lens{scenario=gpu-spatial-analysis.random-points-in-polygon rung=kernel caption="Read seeded triangle selection and point placement."}</p>
<p>::lens{scenario=gpu-spatial-analysis.random-points-in-polygon rung=graph caption="Follow triangulation, prefix allocation and point publication."}</p>
<h4 id="how-it-computes-7">How it computes</h4>
<p>{/* lens:GPURandomPointsOnLine */}</p>
<p>::lens{scenario=gpu-spatial-analysis.random-points-on-line rung=datum datum=0 caption="Inspect the first sample slot: road row and arc-length fraction."}</p>
<p>::lens{scenario=gpu-spatial-analysis.random-points-on-line rung=kernel caption="Read Philox draws and segment interpolation."}</p>
<p>::lens{scenario=gpu-spatial-analysis.random-points-on-line rung=graph caption="Follow segment lengths, prefix allocation and point publication."}</p>
<p>{/* parameters: GPUDotDensity generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h4 id="gpudotdensity-parameters"><code>GPUDotDensity</code> parameters</h4>
<ul>
<li>Packer: <code>getGPUDotDensityParameterValues(settings)</code>, writes <code>GPU_DOT_DENSITY_PARAMETER_LENGTH</code> (8) <code>uint32</code> values.</li>
</ul>
<p>Settings of <code>getGPUDotDensityParameterValues</code>:</p>
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
<td>Random seed. Every dot position is a pure function of the seed and its (slot, rank).</td>
</tr>
<tr>
<td><code>dotsPerUnit</code> (optional)</td>
<td><code>number</code></td>
<td>Dots per unit of value (the reciprocal of "people per dot"), usually zoom dependent. Ignored by GPURandomPointsInPolygon . Zero or a non-finite value draws no dots. Defaults to 1.</td>
</tr>
<tr>
<td><code>maskExtent</code> (optional)</td>
<td><code>readonly [number, number, number, number]</code></td>
<td>Placement <code>[originX, originY, cellWidth, cellHeight]</code> of the optional dasymetric mask raster (row 0 has the smallest y). Defaults to <code>[0, 0, 1, 1]</code>; ignored without a mask.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<p>{/* parameters: GPURandomPointsOnLine generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h4 id="gpurandompointsonline-parameters"><code>GPURandomPointsOnLine</code> parameters</h4>
<ul>
<li>Packer: <code>getGPUDotDensityParameterValues(settings)</code>, writes <code>GPU_DOT_DENSITY_PARAMETER_LENGTH</code> (8) <code>uint32</code> values.
<ul>
<li>Input port: <code>parameters</code>.</li>
<li>This consumer reads only <code>seed</code>; other settings in the shared layout are ignored.</li>
</ul>
</li>
</ul>
<p>Settings of <code>getGPUDotDensityParameterValues</code>:</p>
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
<td>Random seed. Every dot position is a pure function of the seed and its (slot, rank).</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>`};export{e as default};