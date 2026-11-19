var e={slug:`api/gpu-terrain/operators`,title:`GPU Terrain operators`,html:`<h1 id="gpu-terrain-operators">GPU Terrain operators</h1>
<h3 id="gputerrainderivatives-gputerraincontours-gputerrainviewshed-gputerrainlineofsight-and-gputerraincumulativeviewshed"><code>GPUTerrainDerivatives</code>, <code>GPUTerrainContours</code>, <code>GPUTerrainViewshed</code>, <code>GPUTerrainLineOfSight</code>, and <code>GPUTerrainCumulativeViewshed</code></h3>
<p>Elevation-tile analysis composed from <code>gpu-raster</code> operators. <code>GPUTerrainDerivatives</code> computes
Horn slope, aspect, and hillshade (uniform, Web Mercator, or geographic cell sizes) with an optional
hillshade texture. Its border pixels use one-sided differences by default
(<code>borderMode: 'one-sided'</code>): the surface is extended one cell by the quadratic through the three
nearest samples, which makes the difference across the edge the second-order one-sided stencil
<code>(-3 f0 + 4 f1 - f2) / (2 h)</code> of luma.gl's <code>GPUFiniteDifference2D</code>, exact on a plane.
<code>borderMode: 'clamp'</code> repeats the edge sample instead and halves the gradient across the edge (a
plane of slope 11.31° reads 5.71° in its border column); <code>'reflect'</code>, <code>'constant'</code> and <code>'nodata'</code>
are forwarded to <code>GPURasterGradient</code>. North comes from the grid: <code>rowDirection</code> defaults to the
sign of the row term <code>grid.affine[4]</code> (negative: rows advance south), and an explicit value that
contradicts the grid throws at construction. The same rule holds for <code>GPUTerrainCurvature</code>,
<code>GPUTerrainVectorRuggedness</code>, <code>GPUReliefShading</code>, <code>GPUTerrainHorizon</code> and <code>GPUTerrainCastShadow</code>.
Columns are taken to advance east, and on a rotated <code>'uniform'</code> grid azimuths are measured from
the grid's up axis. <code>GPUTerrainContours</code> extracts marching-squares segments for several fixed or
per-frame levels with one aggregate overflow flag (<code>output.status.overflow</code>). Every contributor
takes a <code>grid</code> (<code>GPURasterMetadata</code>), reads <code>parameters</code>, and writes into an <code>output</code> bag. The grid
supplies the width, height, cell size and latitude band, so the packers take it as their second
argument. The three visibility contributors share one sight-line model, with <code>cellSizeMode</code> selecting
how grid world units become ground meters:</p>
<ul>
<li><code>GPUTerrainViewshed</code> writes a <code>GPU_TERRAIN_VISIBILITY</code> code (<code>output.visibility</code>, or
<code>output.visibilityTexture</code>) per pixel from one per-frame observer.</li>
<li><code>GPUTerrainLineOfSight</code> tests a batch of <code>[observerColumn, observerRow, targetColumn, targetRow]</code>
pairs. Optional per-pair heights override the defaults. It writes <code>output.visibility</code> and can also
write <code>output.clearance</code>: how many metres the target could sink and stay visible.</li>
<li><code>GPUTerrainCumulativeViewshed</code> makes observers the dispatch dimension, like the <code>gdal_viewshed</code>
cumulative mode. It counts, per pixel, the observers that see it (<code>output.visibleCount</code>, optional
<code>output.marginalCount</code>), with <code>observersPerDispatch</code> observers per node. It uses the packer of
<code>GPUTerrainLineOfSight</code> (<code>getGPUTerrainLineOfSightParameterValues</code>) because the heights, distance,
cell size, curvature and tolerance semantics are shared.</li>
</ul>
<p>All three accept <code>heightReference</code> (default <code>'ground'</code>) and, for absolute observer heights,
<code>observerVerticalDatum</code>, checked against the elevation band's <code>verticalDatum</code>. Targets remain
ground-relative. See <a href="/luma-contributors/docs/api/gpu-terrain/concepts#heights-and-datums">Heights and datums</a>.</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> grid</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> makeGPURasterMetadata</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({width, height, cellSize: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">30</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">});</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUTerrainDerivatives</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  grid, elevation, parameters: parameters.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">importToGraph</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(graph), output: {slope, hillshade}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">parameters.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">write</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">getGPUTerrainDerivativesParameterValues</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({azimuthDegrees: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">315</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}, grid));</span></span></code></pre>
<p><code>GPUTerrainContours</code> takes a required <code>grid</code> (its former optional <code>metadata</code>), which also sets
<code>pixelInterpretation</code> for the vertex offset, and reports the aggregate flag in
<code>output.status.overflow</code>. Each level keeps its own <code>vertices</code>, <code>status</code> (<code>count</code>, optional <code>requiredCount</code> and <code>overflow</code>) and
optional <code>draw</code> members.</p>
<p>Each contour level's optional <code>draw</code> record is rewritten on every encoding from the
capacity-clamped GPU segment count. <code>drawLayout: 'instanced'</code> (default) writes
<code>[verticesPerInstance, segments, 0, 0]</code> for renderers that read segment endpoints from storage
(<code>verticesPerInstance</code> defaults to 2; use 4 or 6 for quad-per-segment lines). <code>drawLayout: 'line-list'</code> writes <code>[2 * segments, 1, 0, 0]</code> for one non-instanced <code>line-list</code> draw over the
<code>vertices</code> column used as a vertex buffer.</p>
<p>The three sight-line contributors accept <code>cellSizeMode?: GPURasterCellSizeMode</code>:
<code>'uniform'</code> (default) means grid world units are ground meters on a projected planar grid;
<code>'geographic'</code> means a degree grid; <code>'web-mercator'</code> means an EPSG:3857 grid. The latter two
require an axis-aligned affine and use WGS84 ground meters for <code>maximumDistance</code>, curvature drop
<code>c·d²</code>, tolerance bands, <code>targetIgnoreDistance</code>, and clearance. Elevations and heights must use
meters too. A ray uses one constant ground cell evaluated at its middle row,
<code>0.5 * (observerRow + targetRow) + 0.5</code>. Along-ray scale change is ignored, so range boundaries
and curvature drops are approximate on long rays, especially at high latitudes; the error depends
on the latitude span and ray bearing.</p>
<p>Each sight line joins pixel positions at one-pixel steps with bilinear elevation and ignores
samples with an invalid corner. There is no azimuth input and no pixel-direction conversion.
Row order comes from the affine, so these contributors have no <code>rowDirection</code> prop. Holding the
ground cell constant along a ray preserves the monotone sample distances used by pyramid skipping.</p>
<p>More precisely, for pixel-centre endpoints <code>o</code> and <code>p</code>, set
<code>n = ceil(max(abs(p.x-o.x), abs(p.y-o.y)))</code>. Test only <code>o + (p-o)·k/n</code>,
<code>k = 1, …, n−1</code>; neither endpoint is an occluder. Ground distance is
<code>D = hypot((p.x-o.x)·cellX, (p.y-o.y)·cellY)</code> and sample distance is <code>D·k/n</code>.
The eye is observer ground plus observer height (unless absolute height is selected).
With zero tolerance, a target is hidden only when a sampled apparent slope strictly exceeds
the target's apparent slope; equality is visible. Curvature lowers both intervening terrain
and the target by <code>c·d²</code>, using each one's distance from the eye. Cumulative viewshed sums the
visible codes from this same test; hidden, out-of-range and no-data observers contribute zero.</p>
<p>This finite sampling convention can miss relief between samples, including the bilinear hump
between diagonally adjacent cells, which have no interior sample. It differs from both
<a href="https://grass.osgeo.org/grass-stable/manuals/r.viewshed.html">GRASS's angular sweep</a> and
<a href="https://gdal.org/en/stable/programs/gdal_viewshed.html">GDAL's Wang reference-plane propagation</a>.
Matching heights, curvature and cell centres alone does not make their native 2D masks equal.
For sampler-matched reference testing, resample the DEM along each ray and put sample <code>i</code> at
height <code>i·(z(d_i)−eye−c·d_i²)/d_i</code> on a unit-spaced radial raster, with the target last and the
observer at height zero. GDAL with observer/target offsets and curvature all zero then tests
the identical slopes along one row. This preserves the discrete visibility test exactly;
it does not establish equivalence to GDAL on the original 2D DEM.</p>
<ul>
<li><strong>Tolerance band.</strong> <code>GPUTerrainViewshed</code> takes an optional per-frame <code>toleranceParameters</code>
(<code>[toleranceMeters, tolerancePerKilometer, targetIgnoreDistance, targetIgnoreFraction]</code>, all 0 by
default, packed by <code>getGPUTerrainViewshedToleranceParameterValues</code>; <code>GPUTerrainLineOfSight</code> and
<code>GPUTerrainCumulativeViewshed</code> carry the same four values inside their 12-float layout) that turns
the hard cut into a band
<code>β = (toleranceMeters + tolerancePerKilometer·D/1000)/D</code> around the target slope. Codes are
<code>hidden</code> above <code>T + β</code>, <code>visible</code> at or below <code>T − β</code>, and <code>marginal</code> (4) in between.</li>
<li><strong>Ignored final stretch.</strong> Samples within <code>max(targetIgnoreDistance, targetIgnoreFraction·D)</code> of
the target are ignored. This removes the self-occlusion speckle of targets just behind their own
lip; mt-image uses <code>2 m + 1 m/km</code> and ignores <code>max(150 m, 2 %·d)</code>.</li>
<li><strong>Traversal.</strong> <code>traversal: 'pyramid'</code> builds a min-max
<a href="/luma-contributors/docs/api/gpu-raster/operations-analysis"><code>GPURasterExtremaPyramid</code></a> and skips blocks that provably
cannot change the code. Results are bit-identical to <code>'march'</code> (the default).</li>
<li><strong>Shared pyramid.</strong> Pass <code>pyramid</code> (a <code>GPURasterExtremaPyramidOutput</code>) to reuse one pyramid across
<code>GPUTerrainViewshed</code>, <code>GPUTerrainLineOfSight</code>, <code>GPUTerrainCumulativeViewshed</code>,
<code>GPUPointHorizonProfile</code> and <code>GPUPointHorizonVisibility</code>. It must cover the same grid with a
bilinear footprint. Building one costs 0.3 to 0.9 ms, but the pyramid-skip kernels are still about
2x slower than <code>'march'</code> for viewsheds, so prefer <code>'march'</code> there.</li>
</ul>
<p>With the default <code>'uniform'</code> mode, all sight-line kernels retain their previous WGSL and variants.</p>
<p>Earth curvature and refraction lower terrain at ground distance <code>d</code> by <code>c·d²</code> with
<code>c = getGPUTerrainCurvatureCoefficient(k) = (1 − k)/(2R)</code>, R = 6371008.8 m. The default k = 0.13
is the geodetic standard used by mt-image. GDAL's <code>gdal_viewshed -cc 0.85714</code> is the same model
with <code>cc = 1 − k</code>, k = 1/7, so use <code>getGPUTerrainCurvatureCoefficient(1 / 7)</code> to match GDAL. The
difference matters little in practice. Changing k by ±0.1 moves a skyline by 0.011°, 0.022° and
0.034° at 50, 100 and 150 km, and k ± 0.3 (inversions) by 0.03–0.10°. GDAL and mt-image differ by
Δk ≈ 0.013, about 0.003° at 50 km. Range truncation (120 instead of 150 km: up to 0.28°) and DEM
bias (30 m is 0.034° at 50 km) dominate (mt-image measurements).</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUTerrainLineOfSight</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  grid, elevation, pairs, parameters: parameters.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">importToGraph</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(graph), traversal: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'pyramid'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  output: {visibility, clearance}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">parameters.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">write</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">getGPUTerrainLineOfSightParameterValues</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  curvatureCoefficient: </span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">getGPUTerrainCurvatureCoefficient</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> /</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> 7</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  toleranceMeters: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, tolerancePerKilometer: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, targetIgnoreDistance: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">150</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, targetIgnoreFraction: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">0.02</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}, grid));</span></span></code></pre>
<p>Compile-time prop: <code>cellSizeMode</code> defaults to <code>'uniform'</code>; use <code>'geographic'</code> for degrees or
<code>'web-mercator'</code> for EPSG:3857, with per-ray WGS84 ground-meter distances as described above.</p>
<p>Parameter layout: <code>[observerHeight, targetHeight, maximumDistance, cellSizeX, cellSizeY, curvatureCoefficient, toleranceMeters, tolerancePerKilometer, targetIgnoreDistance, targetIgnoreFraction, northEdge, southEdge]</code>.
The packer always writes cell sizes and row edges in grid world units, regardless of mode.
<code>northEdge</code> is the top edge of row 0 and <code>southEdge</code> the bottom edge of the last row, in latitude
degrees or EPSG:3857 y meters for the respective modes; point grids include the half-row shift.</p>
<p>Compile-time prop: <code>cellSizeMode</code> defaults to <code>'uniform'</code>; use <code>'geographic'</code> for degrees or
<code>'web-mercator'</code> for EPSG:3857, with per-ray WGS84 ground-meter distances as described above.</p>
<p>Parameter layout: <code>[observerHeight, targetHeight, maximumDistance, cellSizeX, cellSizeY, curvatureCoefficient, toleranceMeters, tolerancePerKilometer, targetIgnoreDistance, targetIgnoreFraction, northEdge, southEdge]</code>.
The packer always writes cell sizes and row edges in grid world units, regardless of mode.
<code>northEdge</code> is the top edge of row 0 and <code>southEdge</code> the bottom edge of the last row, in latitude
degrees or EPSG:3857 y meters for the respective modes; point grids include the half-row shift.</p>
<p>Compile-time prop: <code>cellSizeMode</code> defaults to <code>'uniform'</code>; use <code>'geographic'</code> for degrees or
<code>'web-mercator'</code> for EPSG:3857, with per-ray WGS84 ground-meter distances as described above.</p>
<p>Parameter layout: <code>[observerColumn, observerRow, observerHeight, targetHeight, maximumDistance, cellSizeX, cellSizeY, curvatureCoefficient, northEdge, southEdge]</code>.
The packer always writes cell sizes and row edges in grid world units, regardless of mode.
<code>northEdge</code> is the top edge of row 0 and <code>southEdge</code> the bottom edge of the last row, in latitude
degrees or EPSG:3857 y meters for the respective modes; point grids include the half-row shift.</p>
<h4 id="gputerrainderivatives"><code>GPUTerrainDerivatives</code></h4>
<p>Turns an elevation grid into Horn slope, aspect and hillshade with two Sobel passes and one fused per-cell kernel.</p>
<p>The grid's column and row axes must be orthogonal. Both construction and
<code>getGPUTerrainDerivativesParameterValues</code> reject shear with a <code>grid.affine</code> error; resample a sheared
DEM first. Rotated orthogonal grids remain supported in <code>'uniform'</code> mode, with azimuths relative
to the grid's up axis. Latitude-dependent cell sizes require axis-aligned grids.
Each pixel-axis length must remain positive and finite after float32 conversion. Orthogonality
uses the normalized-axis dot product (absolute tolerance 1e-6), so rescaling the affine cannot
make shear pass by underflow or overflow.</p>
<p>::lens{scenario=gpu-terrain.terrain-derivatives rung=result caption="Slope drawn over the elevation greys; the steepest cell is marked."}</p>
<p>{/* parameters: GPUTerrainDerivatives generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h5 id="gputerrainderivatives-parameters"><code>GPUTerrainDerivatives</code> parameters</h5>
<ul>
<li>Packer: <code>getGPUTerrainDerivativesParameterValues(settings, grid)</code>, writes <code>GPU_TERRAIN_DERIVATIVES_PARAMETER_LENGTH</code> (8) <code>float32</code> values.
<ul>
<li><code>grid</code> (<code>GPURasterMetadata</code>): the frame grid, which supplies the cell size and row edges</li>
</ul>
</li>
</ul>
<p>Settings of <code>getGPUTerrainDerivativesParameterValues</code>:</p>
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
<td><code>zFactor</code> (optional)</td>
<td><code>number</code></td>
<td>Elevation multiplier applied to derivatives. Defaults to 1.</td>
</tr>
<tr>
<td><code>azimuthDegrees</code> (optional)</td>
<td><code>number</code></td>
<td>Direction the light comes from, degrees clockwise from north. Defaults to 315.</td>
</tr>
<tr>
<td><code>altitudeDegrees</code> (optional)</td>
<td><code>number</code></td>
<td>Light elevation above the horizon in degrees. Defaults to 45.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h4 id="how-it-computes">How it computes</h4>
<p>{/* lens:GPUTerrainDerivatives */}</p>
<p>::lens{scenario=gpu-terrain.terrain-derivatives rung=datum datum=2907 caption="One cell's 3×3 window: Sobel weights 1-2-1, divided by 8 cell sizes."}
::lens{scenario=gpu-terrain.terrain-derivatives rung=formula caption="Horn slope with the z factor and the row-direction sign as the kernel applies them."}
::lens{scenario=gpu-terrain.terrain-derivatives rung=kernel caption="The fused shade kernel: one invocation per cell reads both gradient buffers."}
::lens{scenario=gpu-terrain.terrain-derivatives rung=graph caption="Two GPURasterGradient passes feed one shade node; no loop."}</p>
<h4 id="gputerraincontours"><code>GPUTerrainContours</code></h4>
<p>Extracts contour segment endpoints from elevation at seven levels, 1,100–1,700 m, spaced every 100 m.</p>
<p>::lens{scenario=gpu-terrain.terrain-contours rung=result caption="Seven contour levels shown as endpoints; the GPU publishes no line offsets."}</p>
<h4 id="how-it-computes-1">How it computes</h4>
<p>{/* lens:GPUTerrainContours */}</p>
<p>::lens{scenario=gpu-terrain.terrain-contours rung=datum caption="Inspect one marching square and its crossed edges."}
::lens{scenario=gpu-terrain.terrain-contours rung=formula caption="Edge interpolation clamps fractions and rescales overflowing height differences."}
::lens{scenario=gpu-terrain.terrain-contours rung=kernel caption="Classify, scan, scatter and publish segment counts for each level."}
::lens{scenario=gpu-terrain.terrain-contours rung=graph caption="Seven raster contour pipelines feed the aggregate overflow flag."}</p>
<h4 id="gputerrainviewshed"><code>GPUTerrainViewshed</code></h4>
<p>Classifies terrain visibility from ridge cell (45, 49), with an eye 1.7 m above the DEM.</p>
<p>::lens{scenario=gpu-terrain.terrain-viewshed rung=result caption="Visibility codes over elevation; the observer at column 45, row 49 is marked."}</p>
<p>{/* parameters: GPUTerrainViewshed generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h5 id="gputerrainviewshed-parameters"><code>GPUTerrainViewshed</code> parameters</h5>
<ul>
<li>Packer: <code>getGPUTerrainViewshedParameterValues(settings, grid)</code>, writes <code>GPU_TERRAIN_VIEWSHED_PARAMETER_LENGTH</code> (10) <code>float32</code> values.
<ul>
<li><code>grid</code> (<code>GPURasterMetadata</code>): the frame grid, which supplies the cell size and row edges</li>
</ul>
</li>
<li>Packer: <code>getGPUTerrainViewshedToleranceParameterValues(settings)</code>, writes <code>GPU_TERRAIN_VIEWSHED_TOLERANCE_PARAMETER_LENGTH</code> (4) <code>float32</code> values.</li>
</ul>
<p>Settings of <code>getGPUTerrainViewshedParameterValues</code>:</p>
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
<td><code>observer</code></td>
<td><code>readonly [number, number]</code></td>
<td>Observer <code>[column, row]</code> in pixel-center index space; fractional values allowed.</td>
</tr>
<tr>
<td><code>observerHeight</code> (optional)</td>
<td><code>number</code></td>
<td>Eye height above terrain, or absolute eye elevation with <code>heightReference: 'absolute'</code>. In elevation-band units. Defaults to 1.7; set explicitly in absolute mode.</td>
</tr>
<tr>
<td><code>targetHeight</code> (optional)</td>
<td><code>number</code></td>
<td>Height added to every target cell. Defaults to 0.</td>
</tr>
<tr>
<td><code>maximumDistance</code> (optional)</td>
<td><code>number</code></td>
<td>Ground radius in meters; <code>&#x3C;= 0</code> means unlimited. Defaults to 0.</td>
</tr>
<tr>
<td><code>curvatureCoefficient</code> (optional)</td>
<td><code>number</code></td>
<td>Drop coefficient <code>c</code> in 1/meters: terrain at distance <code>d</code> is lowered by <code>c * d^2</code>. Defaults to 0.</td>
</tr>
</tbody>
</table>
<p>Settings of <code>getGPUTerrainViewshedToleranceParameterValues</code>:</p>
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
<td><code>toleranceMeters</code> (optional)</td>
<td><code>number</code></td>
<td>Constant half-width of the tolerance band in metres of sight-line height at the target. mt-image uses 2. Defaults to 0.</td>
</tr>
<tr>
<td><code>tolerancePerKilometer</code> (optional)</td>
<td><code>number</code></td>
<td>Additional band half-width per kilometre of target distance, in metres. mt-image uses 1. Defaults to 0.</td>
</tr>
<tr>
<td><code>targetIgnoreDistance</code> (optional)</td>
<td><code>number</code></td>
<td>Samples closer than this many metres to the target are not tested (<code>viewPeaks</code> ignores the last <code>max(150 m, 2 %)</code> of the ray). Defaults to 0.</td>
</tr>
<tr>
<td><code>targetIgnoreFraction</code> (optional)</td>
<td><code>number</code></td>
<td>Fraction of the target distance at the target end that is not tested. Defaults to 0.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h4 id="how-it-computes-2">How it computes</h4>
<p>{/* lens:GPUTerrainViewshed */}</p>
<p>::lens{scenario=gpu-terrain.terrain-viewshed rung=datum caption="Follow the samples between the ridge observer and one target cell."}
::lens{scenario=gpu-terrain.terrain-viewshed rung=formula caption="Compare target tangent with curvature-adjusted terrain tangents."}
::lens{scenario=gpu-terrain.terrain-viewshed rung=kernel caption="One invocation traces each target cell from the observer."}
::lens{scenario=gpu-terrain.terrain-viewshed rung=graph caption="Canonical elevation feeds the sight-line visibility kernel."}</p>
<h4 id="gputerrainlineofsight"><code>GPUTerrainLineOfSight</code></h4>
<p>Tests twelve observer-to-target pairs against terrain, returning visibility codes and clearance in metres.</p>
<p>::lens{scenario=gpu-terrain.terrain-line-of-sight rung=result caption="Twelve UTM sight lines coloured by visibility and target clearance."}</p>
<p>{/* parameters: GPUTerrainLineOfSight generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h5 id="gputerrainlineofsight-parameters"><code>GPUTerrainLineOfSight</code> parameters</h5>
<ul>
<li>Packer: <code>getGPUTerrainLineOfSightParameterValues(settings, grid)</code>, writes <code>GPU_TERRAIN_LINE_OF_SIGHT_PARAMETER_LENGTH</code> (12) <code>float32</code> values.
<ul>
<li><code>grid</code> (<code>GPURasterMetadata</code>): the frame grid, which supplies the cell size and row edges</li>
</ul>
</li>
</ul>
<p>Settings of <code>getGPUTerrainLineOfSightParameterValues</code>:</p>
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
<td><code>observerHeight</code> (optional)</td>
<td><code>number</code></td>
<td>Eye height above terrain, or absolute eye elevation with <code>heightReference: 'absolute'</code>. In elevation-band units. Defaults to 1.7; set explicitly in absolute mode.</td>
</tr>
<tr>
<td><code>targetHeight</code> (optional)</td>
<td><code>number</code></td>
<td>Height added to the target elevation. Defaults to 0.</td>
</tr>
<tr>
<td><code>maximumDistance</code> (optional)</td>
<td><code>number</code></td>
<td>Ground radius in metres; <code>&#x3C;= 0</code> means unlimited. Defaults to 0.</td>
</tr>
<tr>
<td><code>curvatureCoefficient</code> (optional)</td>
<td><code>number</code></td>
<td>Drop coefficient <code>c</code> in 1/metres. Defaults to 0.</td>
</tr>
<tr>
<td><code>toleranceMeters</code> (optional)</td>
<td><code>number</code></td>
<td>Constant tolerance band half-width in metres. Defaults to 0.</td>
</tr>
<tr>
<td><code>tolerancePerKilometer</code> (optional)</td>
<td><code>number</code></td>
<td>Additional band half-width per kilometre of distance, in metres. Defaults to 0.</td>
</tr>
<tr>
<td><code>targetIgnoreDistance</code> (optional)</td>
<td><code>number</code></td>
<td>Metres at the target end that are not tested. Defaults to 0.</td>
</tr>
<tr>
<td><code>targetIgnoreFraction</code> (optional)</td>
<td><code>number</code></td>
<td>Fraction of the distance at the target end that is not tested. Defaults to 0.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h4 id="how-it-computes-3">How it computes</h4>
<p>{/* lens:GPUTerrainLineOfSight */}</p>
<p>::lens{scenario=gpu-terrain.terrain-line-of-sight rung=datum caption="Inspect the target and intervening samples for one pair."}
::lens{scenario=gpu-terrain.terrain-line-of-sight rung=formula caption="Clearance measures target height above the maximum intervening tangent."}
::lens{scenario=gpu-terrain.terrain-line-of-sight rung=kernel caption="Each pair marches its ray without early exit to retain exact clearance."}
::lens{scenario=gpu-terrain.terrain-line-of-sight rung=graph caption="The same caller-owned segments carry both GPU result columns."}</p>
<h4 id="gputerraincumulativeviewshed"><code>GPUTerrainCumulativeViewshed</code></h4>
<p>Counts how many of six ridge observers can see each terrain cell.</p>
<p>::lens{scenario=gpu-terrain.terrain-cumulative-viewshed rung=result caption="Darker cells are visible from more of the six ridge observers."}</p>
<p>{/* parameters: GPUTerrainCumulativeViewshed generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h5 id="gputerraincumulativeviewshed-parameters"><code>GPUTerrainCumulativeViewshed</code> parameters</h5>
<ul>
<li>Packer: <code>getGPUTerrainLineOfSightParameterValues(settings, grid)</code>, writes <code>GPU_TERRAIN_LINE_OF_SIGHT_PARAMETER_LENGTH</code> (12) <code>float32</code> values.
<ul>
<li>Input port: <code>parameters</code>.</li>
<li><code>grid</code> (<code>GPURasterMetadata</code>): the frame grid, which supplies the cell size and row edges</li>
</ul>
</li>
</ul>
<p>Settings of <code>getGPUTerrainLineOfSightParameterValues</code>:</p>
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
<td><code>observerHeight</code> (optional)</td>
<td><code>number</code></td>
<td>Eye height above terrain, or absolute eye elevation with <code>heightReference: 'absolute'</code>. In elevation-band units. Defaults to 1.7; set explicitly in absolute mode.</td>
</tr>
<tr>
<td><code>targetHeight</code> (optional)</td>
<td><code>number</code></td>
<td>Height added to the target elevation. Defaults to 0.</td>
</tr>
<tr>
<td><code>maximumDistance</code> (optional)</td>
<td><code>number</code></td>
<td>Ground radius in metres; <code>&#x3C;= 0</code> means unlimited. Defaults to 0.</td>
</tr>
<tr>
<td><code>curvatureCoefficient</code> (optional)</td>
<td><code>number</code></td>
<td>Drop coefficient <code>c</code> in 1/metres. Defaults to 0.</td>
</tr>
<tr>
<td><code>toleranceMeters</code> (optional)</td>
<td><code>number</code></td>
<td>Constant tolerance band half-width in metres. Defaults to 0.</td>
</tr>
<tr>
<td><code>tolerancePerKilometer</code> (optional)</td>
<td><code>number</code></td>
<td>Additional band half-width per kilometre of distance, in metres. Defaults to 0.</td>
</tr>
<tr>
<td><code>targetIgnoreDistance</code> (optional)</td>
<td><code>number</code></td>
<td>Metres at the target end that are not tested. Defaults to 0.</td>
</tr>
<tr>
<td><code>targetIgnoreFraction</code> (optional)</td>
<td><code>number</code></td>
<td>Fraction of the distance at the target end that is not tested. Defaults to 0.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h4 id="how-it-computes-4">How it computes</h4>
<p>{/* lens:GPUTerrainCumulativeViewshed */}</p>
<p>::lens{scenario=gpu-terrain.terrain-cumulative-viewshed rung=datum caption="Inspect one target cell across all six observers."}
::lens{scenario=gpu-terrain.terrain-cumulative-viewshed rung=formula caption="Count only visible codes; marginal and invalid pairs contribute zero."}
::lens{scenario=gpu-terrain.terrain-cumulative-viewshed rung=kernel caption="Observer-cell invocations atomically increment the visible count."}
::lens{scenario=gpu-terrain.terrain-cumulative-viewshed rung=graph caption="Zero the counts, then accumulate the observer batches."}</p>
<h3 id="gpupointhorizonprofile-and-gpupointhorizonvisibility"><code>GPUPointHorizonProfile</code> and <code>GPUPointHorizonVisibility</code></h3>
<p>A 360° horizon profile from any number of observers, ported from mt-image's horizon march.</p>
<ul>
<li><strong>Inputs.</strong> The elevation raster is the <code>elevation</code> prop on a <code>grid</code>. Observers are
<code>[column, row, height, 0]</code>. <code>heightReference</code> is <code>'ground'</code> (the default) or <code>'absolute'</code>.
In absolute mode, <code>observerVerticalDatum</code> declares the observer heights' datum and is checked
against the elevation band's <code>verticalDatum</code>; see <a href="/luma-contributors/docs/api/gpu-terrain/concepts#heights-and-datums">Heights and datums</a>.</li>
<li><strong>Outputs.</strong> In <code>output</code>, for each (observer, azimuth bin) the profile writes the maximum
apparent-elevation tangent <code>t = (h − h0)/d − c·d</code>, the <code>skylineAngle</code> in degrees (−90 when nothing was sampled), and the
crest distance. Azimuth bin <code>i</code> is at <code>i·360/azimuthCount</code> degrees clockwise from north. Sectors
are set with <code>firstAzimuth</code> and <code>azimuthSpan</code>.</li>
<li><strong>Projections.</strong>
<ul>
<li><code>'planar'</code>: projected metres, a straight ray. The packer takes the cell size from the grid.</li>
<li><code>'web-mercator'</code>: a window of the EPSG:3857 grid (square pixels, rows increasing south). The
packer derives the world size <code>worldPixelSize = 2πa/cellSize</code> (pixels per 360°, a = 6378137 m)
and the Mercator ordinate of the top edge <code>northEdge/a</code> from the grid, so no <code>worldPixelSize</code>
or <code>originY</code> is passed. The ray is a piecewise great circle traced on a sphere of
R = 6371008.8 m. Breakpoints are baked from <code>maximumLatitude</code> and <code>segmentTolerance</code>, and
positions are linear in pixels between them. Distances and azimuths are WGS84 ground metres
and ground bearings, by the same rule as the <code>'web-mercator'</code> row scale of the raster-grid
contributors (<code>M cos(lat)/a</code> north-south and <code>N cos(lat)/a</code> east-west per EPSG:3857 metre): a
sphere metre along sphere azimuth <code>A</code> is <code>sqrt((N sin A)² + (M cos A)²)/R</code> ground metres at the
observer latitude, and a ground bearing <code>T</code> is traced along <code>normalize(M sin T, N cos T)</code>.
The scale is evaluated once per ray at the observer. Tangents, the curvature drop, the
per-frame <code>maximumDistance</code>, <code>targetIgnoreDistance</code> and the distance outputs are ground
metres; the topology <code>maximumDistance</code> sizes the sample lattice in sphere metres, so a ray
reaches <code>min(1, scale)</code> times that far (0.56% short due north at the equator).</li>
<li><code>getGPUPointHorizonProfileParameterValues(settings, grid)</code> and
<code>getGPUPointHorizonVisibilityParameterValues(settings, grid)</code> tag the words with
<code>settings.projection</code> (default <code>'planar'</code>); a contributor whose <code>projection</code> differs reports NaN
(profile) or no-data (visibility) instead of misreading them. The lattice <code>cellSize</code> prop stays
a topology value: it sizes the distance lattice, and on Web Mercator the true pixel size varies
with latitude.</li>
</ul>
</li>
<li><strong>Sample distances.</strong> Samples sit on an exact power-of-two octave lattice: within
<code>[2^k, 2^(k+1))</code> the step is a power of two following mt-image's step rule
<code>max(stepFactor·d, clamp(nearFactor·d, minimumStep, cellSize·cellSteps))</code>, from 20 m out to
<code>maximumDistance</code>. Every distance is exact in float32.
Specifically, evaluate the step rule at the octave's lower endpoint <code>2^k</code>, round it down
to a power of two, and clamp its exponent to <code>[-8, k]</code>. Keep distances
<code>2^k + j·step</code> inside that octave and the requested inclusive distance range.
<code>minimumDistance</code> is in metres (default 20), not cells. Planar distances use both grid cell
sizes; the scalar <code>cellSize</code> option controls only this sampling density.</li>
<li><strong>Precision.</strong> Precision follows mt-image:
<ul>
<li>the eye is held as an integer pixel plus a float32 fraction;</li>
<li>heights are taken relative to the eye as a hi/lo pair behind an <code>opaque()</code> guard, against
Metal fast math;</li>
<li>breakpoint offsets use the cancellation-free <code>bp()</code> with series <code>atan</code> and <code>atanh</code>;</li>
<li>azimuth sin/cos use an exact integer octant reduction, and degrees use an accurate <code>atan</code>
(the WGSL builtins are only 2^-11 or about 4096 ULP accurate).</li>
</ul>
</li>
<li><strong>Traversal.</strong> <code>traversal: 'pyramid'</code> (the default) skips min-max pyramid blocks whose
conservative bound cannot raise the running maximum. Tangent and distance bits are identical to
<code>'march'</code>. A shared <code>pyramid</code> can be passed as for the viewshed contributors.</li>
</ul>
<p>For point visibility with zero ignore distance/fraction, only lattice samples strictly before
the target contribute to occlusion. The target itself is excluded even when it lies exactly
on the lattice. With zero angular and elevation uncertainty, equality is visible. Observer
height is added to bilinear observer ground in ground-relative mode; target height is always
added to bilinear target ground. Curvature subtracts <code>c·d</code> from both apparent tangents.
These denser samples can detect an obstruction that the three coarse sight-line contributors
miss. Native GRASS/GDAL masks can therefore differ here too; the radial reference adapter
above preserves this sample set as well, including its unequal distance intervals.</p>
<p><code>GPUPointHorizonVisibility</code> classifies targets <code>[column, row, height, observerIndex]</code> like
mt-image's peak classifier:</p>
<ul>
<li><strong>The ray.</strong> It marches the full ray along the observer→target bearing. On Web Mercator the
bearing and distance come from a cancellation-free spherical inverse, and the distance is then
scaled to WGS84 ground metres as above, so a <code>maximumDistance</code> cut agrees with
<code>GPUTerrainLineOfSight</code> on the same tile.</li>
<li><strong>What it records.</strong> It keeps the occluder maximum before <code>q = d − max(targetIgnoreDistance, targetIgnoreFraction·d)</code> and the full skyline maximum.</li>
<li><strong>Codes.</strong> With <code>tol = toleranceDegrees + σz/d</code>, a target is <code>hidden</code> below the occluder by more
than <code>tol</code>, <code>marginal</code> from exactly <code>tol</code> below it up to (not including) <code>tol</code> above it, and
otherwise <code>visible</code>. The classes are ordered with no gap between <code>hidden</code> and <code>marginal</code>.</li>
<li><strong>Skyline.</strong> <code>onSkyline</code> is set when the target is within <code>skylineToleranceDegrees</code> of the
skyline.</li>
<li><strong>Details.</strong> The optional <code>output.details</code> holds <code>[targetElevation°, occluderElevation°, skylineElevation°, onSkyline]</code>.</li>
</ul>
<p>Measured against float64 oracles, the elevation error is at most 3.3e-5° (planar) and 3.0e-5°
(Web Mercator).</p>
<h4 id="gpupointhorizonprofile"><code>GPUPointHorizonProfile</code></h4>
<p>Reduces terrain rays to skyline angles and tangents for three ridge observers and 36 azimuth bins.</p>
<p>::lens{scenario=gpu-terrain.point-horizon-profile rung=result caption="A 36-bin by three-observer strip shows skyline angles; columns advance clockwise by 10 degrees."}</p>
<p>{/* parameters: GPUPointHorizonProfile generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h5 id="gpupointhorizonprofile-parameters"><code>GPUPointHorizonProfile</code> parameters</h5>
<ul>
<li>Packer: <code>getGPUPointHorizonProfileParameterValues(settings, grid)</code>, writes <code>GPU_POINT_HORIZON_PROFILE_PARAMETER_LENGTH</code> (8) <code>float32</code> values.
<ul>
<li><code>grid</code> (<code>GPURasterMetadata</code>): the frame grid, which supplies the cell size and row edges</li>
</ul>
</li>
</ul>
<p>Settings of <code>getGPUPointHorizonProfileParameterValues</code>:</p>
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
<td><code>projection</code> (optional)</td>
<td><code>GPUPointHorizonProjection</code></td>
<td>Projection the packed grid words describe. Must equal the contributor's <code>projection</code>, else the kernel reports NaN or no-data. Defaults to <code>'planar'</code>.</td>
</tr>
<tr>
<td><code>curvatureCoefficient</code> (optional)</td>
<td><code>number</code></td>
<td>Drop coefficient <code>c</code> in 1/meters. Defaults to 0.</td>
</tr>
<tr>
<td><code>maximumDistance</code> (optional)</td>
<td><code>number</code></td>
<td>Ray length cap in meters; <code>&#x3C;= 0</code> or absent means the topology <code>maximumDistance</code>.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h4 id="how-it-computes-5">How it computes</h4>
<p>{/* lens:GPUPointHorizonProfile */}</p>
<p>::lens{scenario=gpu-terrain.point-horizon-profile rung=datum caption="Inspect a single observer and azimuth bin."}
::lens{scenario=gpu-terrain.point-horizon-profile rung=formula caption="Maximise relative terrain height divided by distance, minus the curvature term."}
::lens{scenario=gpu-terrain.point-horizon-profile rung=kernel caption="March the exact distance lattice and retain the largest apparent tangent."}
::lens{scenario=gpu-terrain.point-horizon-profile rung=graph caption="Ray maxima feed the skyline-angle conversion pass."}</p>
<h4 id="gpupointhorizonvisibility"><code>GPUPointHorizonVisibility</code></h4>
<p>Classifies four fixed targets from each of three ridge observers and records target, occluder and skyline angles.</p>
<p>::lens{scenario=gpu-terrain.point-horizon-visibility rung=result caption="Twelve target classifications share the three observers used by the skyline profile."}</p>
<p>{/* parameters: GPUPointHorizonVisibility generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h5 id="gpupointhorizonvisibility-parameters"><code>GPUPointHorizonVisibility</code> parameters</h5>
<ul>
<li>Packer: <code>getGPUPointHorizonVisibilityParameterValues(settings, grid)</code>, writes <code>GPU_POINT_HORIZON_VISIBILITY_PARAMETER_LENGTH</code> (12) <code>float32</code> values.
<ul>
<li><code>grid</code> (<code>GPURasterMetadata</code>): the frame grid, which supplies the cell size and row edges</li>
</ul>
</li>
</ul>
<p>Settings of <code>getGPUPointHorizonVisibilityParameterValues</code>:</p>
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
<td><code>projection</code> (optional)</td>
<td><code>GPUPointHorizonProjection</code></td>
<td>Projection the packed grid words describe. Must equal the contributor's <code>projection</code>, else the kernel reports NaN or no-data. Defaults to <code>'planar'</code>.</td>
</tr>
<tr>
<td><code>curvatureCoefficient</code> (optional)</td>
<td><code>number</code></td>
<td>Drop coefficient <code>c</code> in 1/meters. Defaults to 0.</td>
</tr>
<tr>
<td><code>maximumDistance</code> (optional)</td>
<td><code>number</code></td>
<td>Ray length cap in meters; <code>&#x3C;= 0</code> or absent means the topology <code>maximumDistance</code>.</td>
</tr>
<tr>
<td><code>toleranceDegrees</code> (optional)</td>
<td><code>number</code></td>
<td>Angular tolerance in degrees of the occlusion test. Defaults to 0.02.</td>
</tr>
<tr>
<td><code>sigmaZ</code> (optional)</td>
<td><code>number</code></td>
<td>Vertical standard error in meters, widening the tolerance by <code>sigmaZ / d</code> radians. Defaults to 5.</td>
</tr>
<tr>
<td><code>skylineToleranceDegrees</code> (optional)</td>
<td><code>number</code></td>
<td>Tolerance in degrees of the skyline test. Defaults to 0.05.</td>
</tr>
<tr>
<td><code>targetIgnoreDistance</code> (optional)</td>
<td><code>number</code></td>
<td>Meters before the target where the occlusion test stops: terrain in this last stretch of the ray is never tested, so a target inside it cannot be hidden by it. Defaults to one ground pixel along the ray at the target (library rule 4: derived from the cell size), which keeps the target's own bilinear cell from occluding it. mt-image's fixed 150 m is the named preset <code>GPU_POINT_HORIZON_MT_IMAGE_TARGET_IGNORE_DISTANCE</code>.</td>
</tr>
<tr>
<td><code>targetIgnoreFraction</code> (optional)</td>
<td><code>number</code></td>
<td>Fraction of the target distance added to the ignore stretch. Defaults to 0.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h4 id="how-it-computes-6">How it computes</h4>
<p>{/* lens:GPUPointHorizonVisibility */}</p>
<p>::lens{scenario=gpu-terrain.point-horizon-visibility rung=datum caption="Inspect one target row and its observer index."}
::lens{scenario=gpu-terrain.point-horizon-visibility rung=formula caption="Angular tolerances separate hidden, marginal and visible targets."}
::lens{scenario=gpu-terrain.point-horizon-visibility rung=kernel caption="March the full bearing while retaining the pre-target occluder maximum."}
::lens{scenario=gpu-terrain.point-horizon-visibility rung=graph caption="Canonical elevation feeds per-target visibility and angle details."}</p>
<h3 id="gputerrainrgbdecode-and-gputerrainspikerepair"><code>GPUTerrainRGBDecode</code> and <code>GPUTerrainSpikeRepair</code></h3>
<p>These contributors turn RGB-encoded elevation tiles into float32 heights with uint32 validity.</p>
<p><code>GPUTerrainRGBDecode</code> handles two encodings:</p>
<ul>
<li>Terrarium: <code>R·256 + G + B/256 − 32768</code>.</li>
<li>Mapbox terrain-RGB: <code>−10000 + 0.1·(R·65536 + G·256 + B)</code>.</li>
</ul>
<p>It reads either an <code>rgba8unorm</code> texture (with <code>textureLoad</code>) or a buffer of packed little-endian
RGBA8 words, which is <code>getImageData().data</code> viewed as a <code>Uint32Array</code>. It writes heights and
validity. A pixel is nodata when any of these hold, and nodata pixels hold NaN bits and validity 0:</p>
<ul>
<li>its alpha is 0;</li>
<li>it matches the optional <code>noDataRGB</code>;</li>
<li>the optional input <code>validity</code> marks it invalid;</li>
<li>its height falls outside <code>validRange</code> (default <code>[-11000, 9000]</code> m, which removes blank-canvas pixels).</li>
</ul>
<p><code>clampBathymetry</code> is opt-in. It flattens heights in (−12000, 0) to sea level, which also flattens
the Dead Sea and polders.</p>
<p>Decode before you filter. Never sample the encoded texture bilinearly or through mips, because
filtering byte channels mixes carries. Decode with nearest/<code>textureLoad</code>, then resample the float
heights.</p>
<p>Terrarium decoding is bit-exact in f32 for all 2^24 codes: every partial sum is a multiple of 1/256
below 2^16. Mapbox decoding is one correctly rounded multiply, <code>f32(N − 100000) · f32(0.1)</code>. Its
error is at most <code>½ulp + |M|·1.5e-9</code>, which is below 0.6 mm for |h| ≤ 9000 m.</p>
<p><code>GPUTerrainSpikeRepair</code> is opt-in. <code>grid</code> holds at most <code>getGPUTerrainSpikeRepairMaximumCellCount(device)</code>
cells (4194304, 2048 × 2048 on a default device), because the per-component vote bit counts take 32
bytes per cell in one storage binding. It fixes ±256·k m Terrarium red-byte errors, such as canvas
anti-fingerprinting noise:</p>
<ol>
<li>It labels 4-connected components across edges with |Δh| ≤ <code>jump</code>, using the analysis-domain
weak-component kernel (bounded minimum-label propagation with pointer jumping) on a
fixed-stride grid CSR. A label is the lowest pixel index of its component.</li>
<li>A component that is neither the largest nor bigger than 25 % of the valid pixels is shifted by
−<code>step</code>·k when at least 80 % of its border seams agree on a jump of <code>step</code>·k ± <code>tolerance</code>.</li>
</ol>
<p>Run it on full-resolution decoded heights, before any filtering. Nodata is never filled or
repaired. It can shift a real enclosed butte whose walls happen to measure 216–296 m, so enable it
only when the source is known to be noisy.</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> grid</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> makeGPURasterMetadata</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({width, height});</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUTerrainRGBDecode</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  grid, encoding: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'terrarium'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, input: {texture}, output: {values, outputValidity: validity}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUTerrainSpikeRepair</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  grid,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  elevation: </span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">makeGPURasterBand</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(values, grid, {id: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'dem'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, validity}),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  output: {values: repaired, outputValidity: repairedValidity, statistics}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span></code></pre>
<p><code>output.statistics</code> holds <code>[jumpCount, repairedPixelCount, shiftedComponentCount, remainingJumpCount, converged]</code>. If the labelling does not converge within
<code>maximumComponentIterations</code>, nothing is repaired and the valid pixels are marked instead of
copied: heights are <code>+Infinity</code>, validity and labels hold <code>GPU_TERRAIN_SPIKE_REPAIR_UNRESOLVED</code>,
and <code>converged</code> is 0. The optional one-row <code>output.status.converged</code> and <code>output.status.iterationLimitReached</code>
ports carry the same answer under the shared status names (<code>iterationLimitReached</code> is the
complement).</p>
<h4 id="gputerrainrgbdecode"><code>GPUTerrainRGBDecode</code></h4>
<p>Decodes raw Terrarium RGBA words into elevation metres and validity on the 32×32 grid.</p>
<p>::lens{scenario=gpu-terrain.terrain-rgb-decode rung=result caption="Packed Terrarium input becomes decoded elevation on 20 m cells."}</p>
<h4 id="how-it-computes-7">How it computes</h4>
<p>{/* lens:GPUTerrainRGBDecode */}</p>
<p>::lens{scenario=gpu-terrain.terrain-rgb-decode rung=datum caption="Inspect the RGB bytes and decoded height of one cell."}
::lens{scenario=gpu-terrain.terrain-rgb-decode rung=formula caption="Reassemble the integer RGB code before the exact power-of-two height conversion."}
::lens{scenario=gpu-terrain.terrain-rgb-decode rung=kernel caption="One invocation unpacks, decodes and checks validity for each cell."}
::lens{scenario=gpu-terrain.terrain-rgb-decode rung=graph caption="One decode pass writes elevation and validity buffers."}</p>
<h4 id="gputerrainspikerepair"><code>GPUTerrainSpikeRepair</code></h4>
<p>Repairs three injected +900 m spikes by shifting small components whose border jumps agree on the configured 900 m step.</p>
<p>::lens{scenario=gpu-terrain.terrain-spike-repair rung=result caption="Compare the spiked DEM with repaired elevation; no per-cell repair mask is published."}</p>
<h4 id="how-it-computes-8">How it computes</h4>
<p>{/* lens:GPUTerrainSpikeRepair */}</p>
<p>::lens{scenario=gpu-terrain.terrain-spike-repair rung=datum caption="Inspect component membership and seam votes around an injected spike."}
::lens{scenario=gpu-terrain.terrain-spike-repair rung=formula caption="A majority of agreeing border seams selects a whole-component height shift."}
::lens{scenario=gpu-terrain.terrain-spike-repair rung=iterations caption="Component-label rounds precede the final height repair; earlier height frames remain unavailable."}
::lens{scenario=gpu-terrain.terrain-spike-repair rung=kernel caption="Component sizes and seam votes determine which heights shift."}
::lens{scenario=gpu-terrain.terrain-spike-repair rung=graph caption="Connected components feed voting, shift decisions and repair statistics."}</p>
<h3 id="gputerrainpeakprominence"><code>GPUTerrainPeakProminence</code></h3>
<p>Computes elder-rule prominence of every peak of an elevation grid, entirely on the GPU, from a join tree of sorted saddle edges.</p>
<p>::lens{scenario=gpu-terrain.terrain-peak-prominence rung=result caption="Prominence per cell in metres; the most prominent finite peak is marked, the elder peak is infinite."}</p>
<p>Computes eight-connected upper-excursion-set prominence (zero-dimensional persistence): a peak
dies at the highest saddle connecting its component to an elder peak, and prominence is peak
height minus that saddle. Higher peaks are elder; equal heights prefer the lowest cell index.
Plateau members and nonpeaks have zero prominence except for the plateau's representative.
Nodata is a barrier and receives NaN. Each disconnected component's surviving peak has infinite
prominence and a key saddle of negative infinity; the boundary has no artificial sea level.</p>
<p>The GPU sorts the depression-fill spill graph's neighbor edges by descending saddle, then builds
a join tree with the parallel parent-splicing helper shared by HDBSCAN. One label per cell preserves
peaks inside tiles that the fill-only outlet contraction would erase. Cycle edges create unary nodes;
gated ancestor doubling propagates elder peaks in logarithmic dispatch depth, and each binary merge
records one losing peak. Stable saddle-edge ordering preserves <code>parentPeak</code> ties. There is no CPU
readback or fallback. Contended splices can traverse long chains; performance measurements are pending.
Storage is linear and the maximum grid size is <code>getGPUTerrainPeakProminenceMaximumCellCount(device)</code>:
the child table requires 40 bytes per cell per binding (3,355,443 cells on a 128 MiB binding limit),
compared with the previous serial implementation's 16 bytes per cell.
The NumPy vertex-activation reference and definition are in
<code>modules/analysis/test/oracles/terrain-peak-prominence.py</code>.</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUTerrainPeakProminence</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  grid,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  elevation,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  output: {prominence, keySaddle, parentPeak}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span></code></pre>
<h4 id="parameters">Parameters</h4>
<p>No packed settings or packer: <code>grid</code> fixes dimensions and connectivity is eight neighbors.
<code>elevation</code> is per-frame. <code>output.prominence</code> is required; <code>output.keySaddle</code> and
<code>output.parentPeak</code> are optional, each one value per cell. <code>parentPeak</code> records the winning peak
at the merge, with <code>0xffffffff</code> for surviving peaks and nodata; equal-saddle parent links follow
stable cell/direction edge order. Output buffers must be disjoint from inputs and each other.</p>
<h4 id="how-it-computes-9">How it computes</h4>
<p>{/* lens:GPUTerrainPeakProminence */}</p>
<p>::lens{scenario=gpu-terrain.terrain-peak-prominence rung=datum datum=3181 caption="The highest summit in the window: its neighbourhood before the join tree decides."}
::lens{scenario=gpu-terrain.terrain-peak-prominence rung=kernel caption="Spill edges are keyed by saddle height, sorted, then spliced into a join tree."}
::lens{scenario=gpu-terrain.terrain-peak-prominence rung=graph caption="Edge keys, a radix sort, parent splicing and ancestor doubling, all on the GPU."}</p>
<h3 id="gputerrainsummits-gputerrainpeaksnap-gputerraincriticalpoints-and-gpuprofilepeaks"><code>GPUTerrainSummits</code>, <code>GPUTerrainPeakSnap</code>, <code>GPUTerrainCriticalPoints</code>, and <code>GPUProfilePeaks</code></h3>
<p>These contributors find terrain features. Summit and snap discs are metric: the packers read the
cell size and latitude band from the <code>grid</code> they are given (per frame), in <code>uniform</code>,
<code>web-mercator</code> or <code>geographic</code> mode, evaluated at the centre row. Nodata never contributes a
height.</p>
<p><strong><code>GPUTerrainSummits</code></strong> marks pixel <code>p</code> as a summit when all of these hold:</p>
<ul>
<li><code>p</code> is the strict (height, −index) maximum of its radius-<code>r</code> disc;</li>
<li><code>drop = h_p − max(ring) ≥ minimumDrop</code>, where the ring is the disc pixels with an 8-neighbour
outside the disc. Every path leaving the disc crosses the ring, so <code>drop</code> is a radius-limited
lower bound on prominence.</li>
</ul>
<p>Under <code>incompleteNeighborhood: 'reject'</code> (the default), discs clipped by the grid edge or touching
nodata are rejected. <code>output</code> holds an optional <code>summitMask</code>, an optional per-pixel <code>drop</code>, an
optional compact list <code>summits</code> with an optional <code>summitDrop</code> column, and
<code>status.radiusClamped</code>, set when the radius was clamped to <code>maximumRadiusPixels</code>.</p>
<p>Cost is O(R) per candidate pixel in the disc radius R (pixels), not O(R²): the maximum test
descends a <code>'cell'</code>-footprint max pyramid (first block 2) and reads the maximum of every block that
lies wholly inside the disc, and the ring is visited from per-row half-widths. A 3×3 pre-test
rejects almost every pixel first. The optional <code>pyramid</code> prop reuses a prebuilt
<code>GPURasterExtremaPyramid.output</code> (<code>footprint: 'cell'</code>, <code>incompleteNeighborhood: 'ignore'</code> only);
results are bit-identical to the previous full disc scan.</p>
<p><strong><code>GPUTerrainPeakSnap</code></strong> takes at most <code>getGPUTerrainPeakSnapMaximumCandidateCount(device)</code>
candidates (6710886 on a default device), because it keeps five result words per candidate in one
storage binding. It moves caller
candidate points, given as <code>[column, row]</code>, to the highest
valid pixel within a per-candidate or per-frame radius. It keeps the original point in four
cases:</p>
<ul>
<li>the maximum lies on the disc ring (a flank, not a summit);</li>
<li>the move would exceed <code>maximumMove</code>;</li>
<li>the height would change by more than <code>maximumHeightChange</code>;</li>
<li>there is no valid data.</li>
</ul>
<p>Each candidate gets <code>output.snapStatus</code> from <code>GPU_TERRAIN_PEAK_SNAP_STATUS</code>, beside
<code>output.positions</code>, <code>output.heights</code> and the optional <code>output.snapDistance</code>. mt-image's distance
rule
<code>min(250, 60 + 0.004·d)</code> m can be supplied through <code>candidateRadii</code>.</p>
<p>Optional <code>candidateHeights</code> supplies absolute catalogue elevations. <code>candidateVerticalDatum</code>
requires <code>candidateHeights</code> and is checked against the elevation band's <code>verticalDatum</code>;
see <a href="/luma-contributors/docs/api/gpu-terrain/concepts#heights-and-datums">Heights and datums</a>.</p>
<p><strong><code>GPUTerrainCriticalPoints</code></strong> classifies each pixel as regular, peak, pit, saddle, boundary or
noData by counting sign changes around its ring. Equal heights are ordered by index (simulation of
simplicity). There are two rings:</p>
<ul>
<li>8 is the classic Peucker–Douglas ring.</li>
<li>6 is the Freudenthal triangulation, which keeps the Euler relation peaks − saddles + pits
consistent.</li>
</ul>
<p><code>output.classes</code> is required; <code>output.signChanges</code> (per pixel) and <code>output.counts</code> (per class)
are optional.</p>
<p><strong><code>GPUProfilePeaks</code></strong> finds peaks along 1-D profiles in CSR spans, for example <code>GPURasterProfile</code>
<code>sampleValues</code> + <code>lineSampleOffsets</code>, or a horizon profile with <code>wrap: true</code>. For each candidate it:</p>
<ul>
<li>tests for a local maximum over ±2 samples;</li>
<li>computes windowed topographic prominence;</li>
<li>refines the position with a parabola.</li>
</ul>
<p>Non-maximum suppression within <code>suppressionRadius</code> runs in GPU rounds (at most
<code>maximumSuppressionIterations</code>) and reaches exactly the greedy (prominence-descending) result. If
it does not converge, <code>output.status.converged</code> is 0, undecided candidates carry the
<code>GPU_PROFILE_PEAKS_UNRESOLVED_*</code> sentinels and optional <code>output.peaks.status.candidateOverflow</code> is 1.
<code>output.peaks.status.overflow</code> reports final peak-list capacity only. The optional <code>output.status.iterationLimitReached</code> is
1 in that case. <code>minimumProminence</code> is set per frame and
<code>minimumSide</code> at compile time.</p>
<p>The CPU twin and operation forms honor the same explicit suppression budget (default 16).
Both operation forms capture that budget and their declared output maps once at construction,
including inherited getters. Later JavaScript property mutations do not change the operation.
Display only <code>peakMask === 1</code> as a kept peak; the unresolved mask sentinel means suppression is
incomplete and must qualify a peak-count readout.</p>
<p>The three mask states are <code>1</code> kept (measured prominence), <code>0</code> absent (NaN prominence), and
<code>0xffffffff</code> undecided (+Infinity prominence). Undecided rows are not drawable peaks.
The terrain explorer counts every kept peak in its chart and readout; its 64-marker limit applies
only to map markers. A truncated suppression run is labeled incomplete.
CPU rows are visited in source order within a round; the GPU may decide a different subset at a
small budget because its atomic updates happen in invocation order. Both suppress unresolved
rows from the compact result and mark candidate loss. The old overflow bit-2 constant is replaced
by <code>GPU_PROFILE_PEAKS_CANDIDATE_UNCONVERGED</code> (value 1).</p>
<h4 id="gputerrainsummits"><code>GPUTerrainSummits</code></h4>
<p>Marks pixels that are the highest point of a metric disc and stand out from its boundary ring.</p>
<p>::lens{scenario=gpu-terrain.terrain-summits rung=result caption="Summit cells: the strict maxima of a 600 m disc that stand 5 m above its ring."}</p>
<p>{/* parameters: GPUTerrainSummits generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h5 id="gputerrainsummits-parameters"><code>GPUTerrainSummits</code> parameters</h5>
<ul>
<li>Packer: <code>getGPUTerrainSummitsParameterValues(settings, grid)</code>, writes <code>GPU_TERRAIN_SUMMITS_PARAMETER_LENGTH</code> (8) <code>float32</code> values.
<ul>
<li><code>grid</code> (<code>GPURasterMetadata</code>): the frame grid, which supplies the cell size and row edges</li>
</ul>
</li>
</ul>
<p>Settings of <code>getGPUTerrainSummitsParameterValues</code>:</p>
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
<td><code>radius</code></td>
<td><code>number</code></td>
<td>Disc radius in ground metres. Finite and positive.</td>
</tr>
<tr>
<td><code>minimumDrop</code> (optional)</td>
<td><code>number</code></td>
<td>Smallest accepted drop in elevation units. Finite and at least 0. Defaults to 0.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h4 id="how-it-computes-10">How it computes</h4>
<p>{/* lens:GPUTerrainSummits */}</p>
<p>::lens{scenario=gpu-terrain.terrain-summits rung=datum datum=3181 caption="The highest summit: its disc maximum and the ring drop it is measured against."}
::lens{scenario=gpu-terrain.terrain-summits rung=formula caption="Drop below the highest ring pixel, tested against the minimum drop."}
::lens{scenario=gpu-terrain.terrain-summits rung=kernel caption="A pyramid descent finds the disc maximum; the ring is scanned row by row."}
::lens{scenario=gpu-terrain.terrain-summits rung=graph caption="Maximum pyramid levels feed one summit kernel; no loop."}</p>
<h4 id="gputerrainpeaksnap"><code>GPUTerrainPeakSnap</code></h4>
<p>Moves candidate points to the highest pixel of a metric disc around them, subject to move and height limits.</p>
<p>::lens{scenario=gpu-terrain.terrain-peak-snap rung=result caption="Three candidates and their snapped positions; the flank candidate keeps its place."}</p>
<p>{/* parameters: GPUTerrainPeakSnap generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h5 id="gputerrainpeaksnap-parameters"><code>GPUTerrainPeakSnap</code> parameters</h5>
<ul>
<li>Packer: <code>getGPUTerrainPeakSnapParameterValues(settings, grid)</code>, writes <code>GPU_TERRAIN_PEAK_SNAP_PARAMETER_LENGTH</code> (8) <code>float32</code> values.
<ul>
<li><code>grid</code> (<code>GPURasterMetadata</code>): the frame grid, which supplies the cell size and row edges</li>
</ul>
</li>
</ul>
<p>Settings of <code>getGPUTerrainPeakSnapParameterValues</code>:</p>
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
<td><code>radius</code> (optional)</td>
<td><code>number</code></td>
<td>Default disc radius in ground metres, used for candidates without a radius. Defaults to 150.</td>
</tr>
<tr>
<td><code>maximumMove</code> (optional)</td>
<td><code>number</code></td>
<td>Largest accepted move in ground metres. Defaults to 300.</td>
</tr>
<tr>
<td><code>maximumHeightChange</code> (optional)</td>
<td><code>number</code></td>
<td>Largest accepted <code>|best - reference|</code> in elevation units. Defaults to 80.</td>
</tr>
<tr>
<td><code>interior</code> (optional)</td>
<td><code>boolean</code></td>
<td>Keep the candidate when the best pixel lies on the disc ring (a flank). Defaults to true.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h4 id="how-it-computes-11">How it computes</h4>
<p>{/* lens:GPUTerrainPeakSnap */}</p>
<p>::lens{scenario=gpu-terrain.terrain-peak-snap rung=kernel caption="Each candidate scans its disc, then an ordered chain of accept rules sets the status."}
::lens{scenario=gpu-terrain.terrain-peak-snap rung=graph caption="Canonical elevation feeds one search kernel and one unpack kernel; no loop."}</p>
<h4 id="gputerraincriticalpoints"><code>GPUTerrainCriticalPoints</code></h4>
<p>Classifies every pixel as peak, pit, saddle, regular, boundary or nodata by counting ring sign changes.</p>
<p>::lens{scenario=gpu-terrain.terrain-critical-points rung=result caption="Critical-point classes per cell; the table counts cells in each class."}</p>
<h4 id="how-it-computes-12">How it computes</h4>
<p>{/* lens:GPUTerrainCriticalPoints */}</p>
<p>::lens{scenario=gpu-terrain.terrain-critical-points rung=datum datum=1957 caption="A saddle: four sign changes around its eight-neighbour ring."}
::lens{scenario=gpu-terrain.terrain-critical-points rung=formula caption="Class from the number of sign changes and the count of higher neighbours."}
::lens{scenario=gpu-terrain.terrain-critical-points rung=kernel caption="One kernel walks the ring in cyclic order and counts sign changes."}
::lens{scenario=gpu-terrain.terrain-critical-points rung=graph caption="Canonical elevation feeds one classification kernel; no loop."}</p>
<h4 id="gpuprofilepeaks"><code>GPUProfilePeaks</code></h4>
<p>Finds local maxima along a 1-D profile with windowed prominence, parabolic refinement and non-maximum suppression.</p>
<p>::lens{scenario=gpu-terrain.profile-peaks rung=result caption="Prominence of kept peaks along row 40; the most prominent is marked."}</p>
<p>{/* parameters: GPUProfilePeaks generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h5 id="gpuprofilepeaks-parameters"><code>GPUProfilePeaks</code> parameters</h5>
<ul>
<li>Packer: <code>getGPUProfilePeaksParameterValues(settings)</code>, writes <code>GPU_PROFILE_PEAKS_PARAMETER_LENGTH</code> (1) <code>float32</code> values.</li>
</ul>
<p>Settings of <code>getGPUProfilePeaksParameterValues</code>:</p>
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
<td><code>minimumProminence</code></td>
<td><code>number</code></td>
<td>Minimum prominence, in profile-value units. A candidate with <code>prominence &#x3C; minimumProminence</code> is rejected. Must not be NaN; a value at or below zero accepts every candidate.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h4 id="how-it-computes-13">How it computes</h4>
<p>{/* lens:GPUProfilePeaks */}</p>
<p>::lens{scenario=gpu-terrain.profile-peaks rung=formula caption="Prominence is the peak minus the higher of the two side minima."}
::lens{scenario=gpu-terrain.profile-peaks rung=iterations caption="Suppression rounds that decide which candidates survive, with the convergence word."}
::lens{scenario=gpu-terrain.profile-peaks rung=kernel caption="The candidate kernel walks both sides of each local maximum inside the window."}
::lens{scenario=gpu-terrain.profile-peaks rung=graph caption="Candidate pass, gated suppression rounds, then output publication."}</p>
<h3 id="gputerraincurvature"><code>GPUTerrainCurvature</code></h3>
<p>Estimates profile and plan curvature from local partial derivatives of an elevation raster; convex ground is positive.</p>
<p>::lens{scenario=gpu-terrain.terrain-curvature rung=result caption="Profile curvature over the elevation greys; convex ground is positive, hollows negative."}</p>
<p>Florinsky's curvature system from local partial derivatives <code>p, q, r, s, t</code>, estimated by
Evans–Young (3×3, default), Zevenbergen–Thorne (3×3), or Florinsky's 5×5 polynomial (the
WhiteboxTools estimator). <code>curvatures</code> maps any of 15 kinds to a float32 output: <code>profile</code>,
<code>plan</code>, <code>tangential</code>, <code>mean</code>, <code>gaussian</code>, <code>minimal</code>, <code>maximal</code>, <code>unsphericity</code>, <code>difference</code>,
<code>horizontal-excess</code>, <code>vertical-excess</code>, <code>accumulation</code>, <code>ring</code>, <code>rotor</code>, <code>laplacian</code>. Signs follow
WhiteboxTools: a convex hill is positive, a valley floor negative. Where <code>p² + q² ≤ flatGradient²</code>
the direction-dependent kinds are 0, as in Whitebox. Window samples are differenced against the
centre before any sum, so a 1000 m base height costs no float32 precision.
<code>ringCurvature</code> ports mt-image's multi-radius ring curvature: up to four rings of 8 samples,
<code>Σ gain_k · Σ(z_c − z_i) / (8 r_k · cell)</code>, with an optional Padé-tanh squash. Cell sizes are
uniform, Web Mercator, or geographic per row, as for <code>GPUTerrainDerivatives</code>. Window samples
outside the raster follow <code>borderMode</code>: <code>'one-sided'</code> (default) extends the surface with the
quadratic through the three nearest in-grid samples, so first derivatives across the edge are the
second-order one-sided stencil of luma.gl's <code>GPUFiniteDifference2D</code> and every curvature is 0 on a
plane and exact on a quadratic surface up to the border; <code>'clamp'</code> repeats the edge sample, which
reports curvature on a plane (gradient 0.2 on 10 m cells: profile curvature −0.0197 1/m in the
border column with Zevenbergen–Thorne); <code>'nodata'</code> invalidates windows that leave the raster.
<code>'florinsky'</code> is a third-order (cubic) 5×5 fit, not a quadratic one. Outputs go into
<code>output</code> (<code>curvatures</code>, <code>ringCurvature</code>, and the per-pixel <code>outputValidity</code>). <code>zFactor</code>,
<code>flatGradient</code> and <code>ringGains</code> are per-frame settings; the ring radii are compile time.</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUTerrainCurvature</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  grid, elevation, parameters: parameters.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">importToGraph</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(graph),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  output: {curvatures: {profile, plan: planCurvature, mean}, ringCurvature, outputValidity},</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  ringRadii: [</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">8</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">]</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// Cell size and latitude band come from the grid.</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">parameters.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">write</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">getGPUTerrainCurvatureParameterValues</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({zFactor: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}, grid));</span></span></code></pre>
<h4 id="how-it-computes-14">How it computes</h4>
<p>{/* lens:GPUTerrainCurvature */}</p>
<p>::lens{scenario=gpu-terrain.terrain-curvature rung=datum datum=2576 caption="One cell's 3×3 window feeds the five partial derivatives behind profile and plan curvature."}
::lens{scenario=gpu-terrain.terrain-curvature rung=formula caption="Evans-Young partials p, q, r, s and t feed profile and plan curvature, with the flat-gradient guard."}
::lens{scenario=gpu-terrain.terrain-curvature rung=kernel caption="One invocation per cell reads its window and writes up to five curvature outputs."}
::lens{scenario=gpu-terrain.terrain-curvature rung=graph caption="Canonical elevation feeds one curvature kernel that writes both outputs."}</p>
<p>{/* parameters: GPUTerrainCurvature generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h4 id="gputerraincurvature-parameters"><code>GPUTerrainCurvature</code> parameters</h4>
<ul>
<li>Packer: <code>getGPUTerrainCurvatureParameterValues(settings, grid)</code>, writes <code>GPU_TERRAIN_CURVATURE_PARAMETER_LENGTH</code> (12) <code>float32</code> values.
<ul>
<li><code>grid</code> (<code>GPURasterMetadata</code>): the frame grid, which supplies the cell size and row edges</li>
</ul>
</li>
</ul>
<p>Settings of <code>getGPUTerrainCurvatureParameterValues</code>:</p>
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
<td><code>zFactor</code> (optional)</td>
<td><code>number</code></td>
<td>Elevation multiplier converting elevation units into ground units. Defaults to 1.</td>
</tr>
<tr>
<td><code>flatGradient</code> (optional)</td>
<td><code>number</code></td>
<td>Gradients with <code>p^2 + q^2 &#x3C;= flatGradient^2</code> are flat: the direction-dependent curvatures (profile, plan, tangential, difference, excesses, accumulation, ring, rotor) are 0 there. Defaults to 1e-6.</td>
</tr>
<tr>
<td><code>ringGains</code> (optional)</td>
<td><code>readonly number[]</code></td>
<td>Up to four per-ring gains of the multi-radius ring curvature, one per entry of <code>ringRadii</code>. Defaults to <code>[0.55, 0.45, 0, 0]</code>. These are mt-image's pre-division weights. mt-image additionally divides ring 0 by 0.35 and ring 1 by 0.3; reproduce it with <code>[0.55 / 0.35, 0.45 / 0.3]</code>.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h3 id="gpugeomorphons"><code>GPUGeomorphons</code></h3>
<p>Classifies each cell of an elevation raster into one of ten landforms from eight lines of sight.</p>
<p>::lens{scenario=gpu-terrain.geomorphons rung=result caption="Landform codes over the elevation greys; the five largest classes are coloured, the rest grouped."}</p>
<p>Jasiewicz &#x26; Stepinski (2013) geomorphons, mirroring GRASS <code>r.geomorphon</code>: eight line-of-sight
scans per cell out to <code>searchRadius</code> cells (skipping <code>skipRadius</code>), a flatness threshold in degrees
with optional <code>flatDistance</code>, GRASS's <code>anglev1</code> (default), <code>anglev2</code>, or <code>anglev2-distance</code>
comparison, and GRASS's 9×9 table from (minus, plus) counts to ten forms (<code>GPU_GEOMORPHON_FORMS</code>:
flat 1 … pit 10, 0 invalid). <code>pattern</code> writes the raw base-3 code (GRASS <code>ternary_6561</code>, directions
NE, N, NW, W, SW, S, SE, E) and <code>ternary</code> its rotation- and mirror-invariant minimum (GRASS's 498
classes). Comparisons use cross-multiplied heights and step counts instead of <code>atan</code>, so integer
DEMs with exact ties classify identically to the float64 oracle. Cells within <code>skipRadius + 1</code> of
the border are invalid, as in GRASS. A line of sight with no valid sample after the skip gives
digit 0 (flat) in every mode, where GRASS <code>anglev2</code> gives +1 from its initial ±90° angles. Outputs <code>forms</code>, <code>ternary</code>, <code>pattern</code> and <code>outputValidity</code> go
into <code>output</code>. The flatness threshold, <code>flatDistance</code> and <code>zFactor</code> are per-frame settings
(<code>getGPUGeomorphonsParameterValues(settings, grid)</code> takes the cell size and latitude band from the
grid); <code>searchRadius</code> and <code>skipRadius</code> size the line-of-sight loops and are compile time. North
comes from the grid: <code>rowDirection</code> defaults to the direction <code>grid.affine[4]</code> states (negative:
rows advance south) and an explicit value that contradicts the grid throws. It fixes the digit
order of <code>pattern</code>; forms and <code>ternary</code> are mirror-invariant and do not depend on it.</p>
<p>The instance's <code>searchRadius</code> is advisory for tiling: it measures row-based ground distance,
and east-west reach exceeds it when north-south cells are larger than east-west cells. Cell sizes
can change per frame. Geomorphons exposes no <code>requiredHalo</code> and cannot be supplied directly as a
<code>GPURasterHaloStage</code>; the caller must bound both axes over every row and parameter update.</p>
<h4 id="how-it-computes-15">How it computes</h4>
<p>{/* lens:GPUGeomorphons */}</p>
<p>::lens{scenario=gpu-terrain.geomorphons rung=datum datum=3181 caption="Eight lines of sight from the ridge cell at column 45, row 49: zenith, nadir and digit."}
::lens{scenario=gpu-terrain.geomorphons rung=formula caption="Zenith and nadir per direction give a ternary digit; the digit counts index the landform table."}
::lens{scenario=gpu-terrain.geomorphons rung=kernel caption="One invocation per cell scans eight rays and looks up the landform."}
::lens{scenario=gpu-terrain.geomorphons rung=graph caption="Canonical elevation feeds one fused classification kernel."}</p>
<p>{/* parameters: GPUGeomorphons generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h4 id="gpugeomorphons-parameters"><code>GPUGeomorphons</code> parameters</h4>
<ul>
<li>Packer: <code>getGPUGeomorphonsParameterValues(settings, grid)</code>, writes <code>GPU_GEOMORPHONS_PARAMETER_LENGTH</code> (8) <code>float32</code> values.
<ul>
<li><code>grid</code> (<code>GPURasterMetadata</code>): the frame grid, which supplies the cell size and row edges</li>
</ul>
</li>
</ul>
<p>Settings of <code>getGPUGeomorphonsParameterValues</code>:</p>
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
<td><code>zFactor</code> (optional)</td>
<td><code>number</code></td>
<td>Elevation multiplier converting elevation units into ground units. Defaults to 1.</td>
</tr>
<tr>
<td><code>flatThresholdDegrees</code> (optional)</td>
<td><code>number</code></td>
<td>Flatness threshold in degrees, greater than 0 and below 90. Defaults to 1.</td>
</tr>
<tr>
<td><code>flatDistance</code> (optional)</td>
<td><code>number</code></td>
<td>Distance in ground metres beyond which the flatness threshold is lowered to the height <code>tan(threshold) * flatDistance</code>. 0 disables the rule. Defaults to 0. As in GRASS <code>r.geomorphon</code> (<code>main.c</code>: <code>flat_distance &#x3C;= skip_distance || flat_distance >= search_distance</code> sets it to 0), a distance at or inside the skip distance (<code>skipRadius</code> cells) or at or beyond the search distance (<code>searchRadius</code> cells) disables the rule; both are measured with the row's north-south ground cell size.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h3 id="gputerrainruggedness-and-gputerrainvectorruggedness"><code>GPUTerrainRuggedness</code> and <code>GPUTerrainVectorRuggedness</code></h3>
<p><code>GPUTerrainRuggedness</code> reproduces gdaldem's TPI, TRI (Riley, the default since GDAL 3.3, or Wilson)
and roughness on 3×3 windows, including both edge modes: <code>'nodata'</code> (default) and <code>'extrapolate'</code>
(gdaldem <code>-compute_edges</code>, with GDAL's linear edge extrapolation and centre substitution for nodata
neighbours). It needs no parameters buffer; its outputs (<code>topographicPositionIndex</code>,
<code>terrainRuggednessIndex</code>, <code>roughness</code>, <code>outputValidity</code>) go into <code>output</code>.
<code>GPUTerrainVectorRuggedness</code> is Sappington et al.'s (2007) vector ruggedness measure,
<code>1 − |Σn| / N</code> over Horn unit normals in a <code>(2·radius + 1)²</code> window. The window radius is a
per-frame setting: <code>maximumRadius</code> (compile time, default 1) sizes <code>requiredHalo</code>, bounds the loops
and selects the direct square sum (<code>maximumRadius</code> below 3) or the separable row and column passes
(from 3), and <code>getGPUTerrainVectorRuggednessParameterValues({radius}, grid)</code> picks any integer in
<code>[1, maximumRadius]</code> per frame with the same pipelines. An omitted radius uses <code>maximumRadius</code>, the
packer throws for a value that is not an integer of at least 1, and a larger word written to the
buffer by other means clamps to <code>maximumRadius</code>.</p>
<h4 id="gputerrainruggedness"><code>GPUTerrainRuggedness</code></h4>
<p>Computes the Riley terrain ruggedness index of an elevation raster over 3×3 windows, in elevation units.</p>
<p>::lens{scenario=gpu-terrain.terrain-ruggedness rung=result caption="Ruggedness over the elevation greys; the most rugged cell is marked and the outer ring is empty."}</p>
<h4 id="how-it-computes-16">How it computes</h4>
<p>{/* lens:GPUTerrainRuggedness */}</p>
<p>::lens{scenario=gpu-terrain.terrain-ruggedness rung=datum datum=2907 caption="The steepest cell's nine heights: squared differences from the centre sum to the index."}
::lens{scenario=gpu-terrain.terrain-ruggedness rung=formula caption="Riley ruggedness is the root of the summed squared differences from the centre cell."}
::lens{scenario=gpu-terrain.terrain-ruggedness rung=kernel caption="One invocation per cell reads its 3×3 window and writes the index."}
::lens{scenario=gpu-terrain.terrain-ruggedness rung=graph caption="Canonical elevation feeds a single window kernel."}</p>
<h4 id="gputerrainvectorruggedness"><code>GPUTerrainVectorRuggedness</code></h4>
<p>Measures how much unit surface normals diverge inside a window of an elevation raster, from 0 smooth to 1 rugged.</p>
<p>::lens{scenario=gpu-terrain.terrain-vector-ruggedness rung=result caption="Vector ruggedness in a 5 × 5 window; the roughest cell is marked."}</p>
<p>{/* parameters: GPUTerrainVectorRuggedness generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h5 id="gputerrainvectorruggedness-parameters"><code>GPUTerrainVectorRuggedness</code> parameters</h5>
<ul>
<li>Packer: <code>getGPUTerrainVectorRuggednessParameterValues(settings, grid)</code>, writes <code>GPU_TERRAIN_VECTOR_RUGGEDNESS_PARAMETER_LENGTH</code> (8) <code>float32</code> values.
<ul>
<li><code>grid</code> (<code>GPURasterMetadata</code>): the frame grid, which supplies the cell size and row edges</li>
</ul>
</li>
</ul>
<p>Settings of <code>getGPUTerrainVectorRuggednessParameterValues</code>:</p>
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
<td><code>zFactor</code> (optional)</td>
<td><code>number</code></td>
<td>Elevation multiplier converting elevation units into ground units. Defaults to 1.</td>
</tr>
<tr>
<td><code>radius</code> (optional)</td>
<td><code>number</code></td>
<td>Window half-width in pixels: the window is <code>(2 * radius + 1)</code> square. Integer in <code>[1, maximumRadius]</code>. Defaults to <code>maximumRadius</code>. The packer throws for a value that is not an integer of at least 1; a value above <code>maximumRadius</code> written to the buffer by other means is clamped to <code>maximumRadius</code> by the kernel.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h4 id="how-it-computes-17">How it computes</h4>
<p>{/* lens:GPUTerrainVectorRuggedness */}</p>
<p>::lens{scenario=gpu-terrain.terrain-vector-ruggedness rung=datum datum=2835 caption="One cell's 5 × 5 window of unit normals; the length of their sum sets the value."}
::lens{scenario=gpu-terrain.terrain-vector-ruggedness rung=formula caption="One minus the length of the summed unit normals, divided by the number of valid cells."}
::lens{scenario=gpu-terrain.terrain-vector-ruggedness rung=kernel caption="A normals pass feeds a direct window sum, one invocation per cell."}
::lens{scenario=gpu-terrain.terrain-vector-ruggedness rung=graph caption="Horn normals feed the window sum that returns the ruggedness."}</p>
<h3 id="gputerraintopographicposition-and-gputerrainweisslandforms"><code>GPUTerrainTopographicPosition</code> and <code>GPUTerrainWeissLandforms</code></h3>
<p><code>GPUTerrainTopographicPosition</code> evaluates any number of square scales (up to 64) in O(1) per pixel
from one summed-area table: TPI over the annulus <code>(innerRadius, radius]</code>, the deviation from mean
elevation (DEV) over the full window, and Lindsay et al.'s (2015) DEVmax with the radius at which
it occurs. A float32 summed-area table cannot do this: on a 512×384 tile at 3800 m its window means
are off by 5.4 m. Elevations are instead quantized (<code>quantum</code>, default 1/256) into an exact
64-bit modular table built with <code>GPUScanUint64</code>, <code>GPUScan</code>, and <code>GPUTranspose</code>; variances are formed
relative to the centre cell in integer arithmetic. Measured: TPI within 2.2 mm (the quantization)
and DEV within 2.4e-4 of the float64 definition, and within 4.6e-6 m / 7.3e-7 of the quantized
oracle. Windows are clipped to the raster and to valid cells.</p>
<p>Both contributors accept at most <code>getGPUTerrainSummedAreaMaximumCellCount(device)</code> (11184810, a hard constructor cap, about 3344 × 3344 on a default device) cells: the table keeps 12 bytes per cell in one storage binding, and its scans and
transpose run over the whole grid, so a 4096 × 4096 grid must be split into tiles with a halo of the
largest radius. <code>GPUTerrainTopographicPosition</code> also limits <code>grid</code> cells × <code>scales</code> to
<code>getGPUTerrainTopographicPositionMaximumPlaneCount(device)</code> (33554432 on a default device) when it writes
<code>topographicPositionIndex</code> or <code>deviationFromMean</code>.</p>
<p><code>GPUTerrainWeissLandforms</code> classifies Weiss's (2001) ten landforms from a small and a large TPI
scale and Horn slope. <code>'global'</code> standardization (default) uses the grid mean and population
standard deviation of each scale (two-pass <code>GPUReduction</code>); <code>'local'</code> uses DEV. Thresholds
(<code>standardThreshold</code>, default 1; <code>slopeThresholdDegrees</code>, default 5) are per-frame settings, and
<code>getGPUTerrainWeissLandformsParameterValues(settings, grid)</code> takes the cell size for the slope from
the grid. Both contributors write into <code>output</code> (<code>outputValidity</code> is the per-pixel mask); the scales
are compile time.</p>
<p>Weiss exposes <code>localRadius = max(1, smallScale.radius, largeScale.radius)</code> for the local terms.
Use <code>weiss.getHaloStage()</code> with <code>'local'</code> standardization to obtain a complete published halo
stage. It throws in <code>'global'</code> mode because no local radius covers grid-wide statistics.
Weiss instances have no <code>requiredHalo</code> field and cannot be passed directly to the halo assembler.
Both scale radii (including inner radii), standardization and grid dimensions/affine are captured
at construction by reading declared fields once, including inherited and non-enumerable getters.
Output bindings and elevation-band metadata use the same capture rule. Reusing or editing the
caller's scale/grid objects cannot enlarge a live contributor's footprint beyond its advertised
halo. Construct a new contributor to change those topology fields; elevation contents and the
packed per-frame parameters remain updateable.</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUTerrainTopographicPosition</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  grid, elevation, scales: [{radius: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">3</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}, {radius: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">25</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}, {radius: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">100</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}],</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  output: {deviationFromMean, maximumDeviation, maximumDeviationRadius}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span></code></pre>
<h4 id="gputerraintopographicposition"><code>GPUTerrainTopographicPosition</code></h4>
<p>Subtracts the mean elevation of a square annulus from each cell of an elevation raster, per scale.</p>
<p>::lens{scenario=gpu-terrain.terrain-topographic-position rung=result caption="Cells above (positive) or below (negative) the mean of their 11 × 11 window."}</p>
<h4 id="how-it-computes-18">How it computes</h4>
<p>{/* lens:GPUTerrainTopographicPosition */}</p>
<p>::lens{scenario=gpu-terrain.terrain-topographic-position rung=datum datum=2836 caption="The cell's 11 × 11 window minus its centre: the annulus mean from the summed-area table."}
::lens{scenario=gpu-terrain.terrain-topographic-position rung=formula caption="Centre minus annulus mean, computed exactly in quantized integers from the summed-area table."}
::lens{scenario=gpu-terrain.terrain-topographic-position rung=kernel caption="The evaluate kernel reads box sums from the summed-area table once per scale."}
::lens{scenario=gpu-terrain.terrain-topographic-position rung=graph caption="Elevation feeds the summed-area table scans, then one evaluate kernel."}</p>
<h4 id="gputerrainweisslandforms"><code>GPUTerrainWeissLandforms</code></h4>
<p>Classifies each cell of an elevation raster into ten Weiss landforms from standardized topographic position at two scales and slope.</p>
<p>::lens{scenario=gpu-terrain.terrain-weiss-landforms rung=result caption="Ten landform classes from small- and large-scale position; the five largest are coloured, the rest grouped."}</p>
<p>{/* parameters: GPUTerrainWeissLandforms generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h5 id="gputerrainweisslandforms-parameters"><code>GPUTerrainWeissLandforms</code> parameters</h5>
<ul>
<li>Packer: <code>getGPUTerrainWeissLandformsParameterValues(settings, grid)</code>, writes <code>GPU_TERRAIN_WEISS_LANDFORMS_PARAMETER_LENGTH</code> (8) <code>float32</code> values.
<ul>
<li><code>grid</code> (<code>GPURasterMetadata</code>): the frame grid, which supplies the cell size and row edges</li>
</ul>
</li>
</ul>
<p>Settings of <code>getGPUTerrainWeissLandformsParameterValues</code>:</p>
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
<td><code>zFactor</code> (optional)</td>
<td><code>number</code></td>
<td>Elevation multiplier converting elevation units into ground units. Defaults to 1.</td>
</tr>
<tr>
<td><code>standardThreshold</code> (optional)</td>
<td><code>number</code></td>
<td>Standardized position threshold separating low, mid, and high. Defaults to 1.</td>
</tr>
<tr>
<td><code>slopeThresholdDegrees</code> (optional)</td>
<td><code>number</code></td>
<td>Slope in degrees at or below which a mid/mid cell is a plain. Defaults to 5.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h4 id="how-it-computes-19">How it computes</h4>
<p>{/* lens:GPUTerrainWeissLandforms */}</p>
<p>::lens{scenario=gpu-terrain.terrain-weiss-landforms rung=datum datum=3181 caption="The ridge cell at column 45, row 49: two z-scores, their classes and the slope pick mountain top."}
::lens{scenario=gpu-terrain.terrain-weiss-landforms rung=formula caption="Standardize both scales, class each at plus or minus the threshold, then look up the landform."}
::lens{scenario=gpu-terrain.terrain-weiss-landforms rung=kernel caption="Classification reads both position planes, the slope and the global statistics."}
::lens{scenario=gpu-terrain.terrain-weiss-landforms rung=graph caption="Position, slope and global statistics feed the final classification node."}</p>
<h3 id="gputerrainhorizon-gpusolarshadowmask-gputerraincastshadow-gpusolarirradiance-gpureliefshading-gputextureshading-and-gpusolarposition"><code>GPUTerrainHorizon</code>, <code>GPUSolarShadowMask</code>, <code>GPUTerrainCastShadow</code>, <code>GPUSolarIrradiance</code>, <code>GPUReliefShading</code>, <code>GPUTextureShading</code>, and <code>GPUSolarPosition</code></h3>
<p>Cartographic relief and terrain light in data space, for "sun and shadow" maps on DEM tiles. Every
style and sun value is a per-frame parameters write; none of them recompile the graph. Each class
takes the raster <code>grid</code> (its width and height size the kernels; the cell size and the latitude band
of the packers come from it), packed <code>parameters</code>, and a caller-owned <code>output</code> bag. Outputs are
float32 rasters, each with an optional <code>r32float</code> or <code>rgba32float</code> storage texture (channel 0);
validity outputs are named <code>outputValidity</code>.</p>
<ul>
<li><code>GPUTerrainHorizon</code> marches <code>directionCount</code> (4 to 64) azimuth sectors per pixel, one node per
sector, over a baked step schedule (<code>maximumRadius</code> pixels, optional geometric <code>stepGrowth</code>) with
bilinear samples, z factor, earth curvature, and an optional ground <code>maximumDistance</code>. It writes
a pixel-major horizon map (<code>horizon[pixel * directionCount + sector]</code>, degrees, sector <code>d</code> at
azimuth <code>d * 360 / directionCount</code>), the sky-view factor <code>1 - mean(sin(max(h, 0)))</code>, and positive
openness <code>mean(90 - h)</code>, all in <code>output</code>. Sky-view-only use needs no <code>output.horizon</code> buffer.
There are two sky-view factors and they are not interchangeable. <code>output.skyViewFactor</code> is the
solid-angle factor of Zakšek et al. (2011), <code>1 - mean(sin(max(h, 0)))</code>: the visible fraction of
the hemisphere, made for relief visualization. <code>output.irradianceSkyViewFactor</code> is the
cosine-weighted factor of Dozier and Frew (1990) for a horizontal receiver,
<code>mean(cos²(max(h, 0)))</code>: the fraction of isotropic diffuse irradiance received, which is what
<code>GPUSolarIrradiance</code> needs. A uniform 30 degree horizon gives 0.50 and 0.75; an infinite plane
of slope <code>s</code> gives <code>(1 + cos(s)) / 2</code> in the second. The irradiance factor is reduced from the
horizon map (the quantised one when <code>output.horizon</code> is <code>'unorm16'</code>, error below 3e-5; a
transient float32 map when <code>output.horizon</code> is not bound). Sector azimuths are ground
azimuths on any cell shape: on anisotropic cells (<code>'geographic'</code> off the equator, or non-square
grid cells) each ray's pixel step is <code>normalize(groundDirection / groundCell)</code> at the pixel's own
row, so a 45 degree sector at 60 N steps about two columns per row instead of along the pixel
diagonal (a 26.6 degree ground ray). The cell size is not re-evaluated along a ray; at 60 N and
30 km that leaves a 0.25 degree bearing and 0.14% distance error. Options:
<ul>
<li><code>algorithm: 'sweep'</code> replaces the per-pixel march with an exact upper-hull sweep (Stewart 1998
style) over digital lines through pixel centres: amortised O(1) per pixel and sector, equal
to brute force over the same samples, 4x faster at radius 256 and 36x at full-tile radius on
1024² (16 sectors). It is nearest-pixel sampling of the ray, not an exact horizon of the
surface: each sample takes the elevation of a pixel centre on the digital line, whose lateral
offset from the ray through the pixel's own centre is below one pixel (up to 0.86 px at line
slope 0.3), and its distance is measured along the ideal ray (<code>k · hypot(1, slope)</code> cells),
not to that centre. A sample one cell ahead and 10 m higher on 10 m cells reads 41.8° at line
slope 0.5, where centre-to-centre geometry gives 45.0° (side neighbour) or 35.3° (diagonal).
Sky-view
factors differ from the bilinear march by about 0.01 on rough terrain. The sweep has a large
fixed cost per pixel and sector: it wins at large radii (1024² at radius 1023: 41 ms against
about 1300 ms; 512² at full radius: 36 against 109 ms) and loses at small ones (512² at radius
128: 60–125 ms against about 20 ms). Prefer the march for small radii and the sweep near the
full tile radius. One line family has one pixel slope, so on anisotropic cells the sweep uses
the ground cell of row <code>height / 2</code> for every row; on a <code>'geographic'</code> tile the other rows see
sectors rotated by up to about <code>tan(lat) * halfSpan / 2</code> radians (0.44 degrees at the edges of
a 1 degree tall tile at 60 N).
For mosaics, select <code>parameterLayout: 'sweep'</code> and pack <code>sweepOrigin: [column, row]</code> in settings
to anchor digital-line phase relative to the common <code>sweepSourceGrid</code>. Placement and reference
cells update per frame without changing shader source. A level number does not determine exact
anisotropic downsampling; supply overview pixel coordinates. Standard settings keep the local
sampling model and refuse a sweep halo even when their view has spare capacity.</li>
<li><code>horizonFormat: 'unorm16'</code> halves the horizon map (two 16-bit codes per <code>uint32</code>, 0.00275 degree
steps, code 0 = nodata; <code>unpackGPUTerrainHorizonUnorm16</code> decodes on the CPU). <code>GPUSolarShadowMask</code>
and <code>GPUSolarIrradiance</code> read it.</li>
<li><code>output.negativeOpenness</code> is openness of the inverted DEM (Yokoyama), and
<code>output.anisotropicSkyViewFactor</code> is
RVT's weighted SVF <code>1 - sum(w sin h+) / sum(w)</code>, <code>w = (1 - wMin) cos((a - A)/2)^level + wMin</code>,
computed exactly as RVT's <code>sky_view_factor_compute</code>. <code>anisotropyAzimuthDegrees</code> (<code>A</code>) uses RVT's
direction convention, so the same number means the same direction as RVT's <code>asvf_dir</code>: angles
run counterclockwise from north (90 west, 270 east), and the default 315 is RVT's default, which
is compass north-east. <code>a</code> is each sector's RVT angle, <code>(360 - t) mod 360</code> for compass sector
azimuth <code>t</code>. <code>anisotropyLevel</code> is RVT's integer <code>a_poly_level</code> with RVT's signed power (presets:
low 4 with minimum weight 0.4, high 8 with 0.1). Parameter slots 8-10 hold them per frame
(<code>parameters</code> must then hold 12 floats). This is the only counterclockwise angle in gpu-terrain; sun
azimuths already match RVT's compass <code>shadow_az</code>. In compass terms, independent of RVT: the
sector at compass azimuth <code>(360 - A) mod 360</code> gets the largest weight, so <code>A = 315</code> weights the
north-east sky and <code>A = 90</code> the west. This is checked against unmodified rvt-py 2.2.1
(<code>test/oracles/terrain-sky-view-rvt.py</code>): RVT direction <code>i</code> has angle <code>i * 360 / n</code> and looks at
the pixel offset <code>(-cos, -sin) * r</code> over (rows, columns) of a north-up array, which is
counterclockwise from north; a wall to the north, west, south, east or north-east of a pixel
darkens it most at <code>A = 0, 90, 180, 270, 315</code>, in RVT and here, on north-up and rows-north
grids. RVT does not wrap <code>a - A</code>, so odd levels are not periodic in <code>A</code>; the kernel follows.</li>
<li>Agreement with RVT in numbers: RVT samples whole pixels (1 to the radius on the axes, <code>(k, k)</code>
on the diagonals, a rounded staircase in between, each at its centre-to-centre distance) and
reflect-pads the array by the radius. Where the kernel reads the same samples (the march with
4 sectors, the sweep with 4 or 8, any plane with up to 8) the sky-view factor and the
anisotropic factor agree with RVT to 3e-7 and both openness outputs to 3e-5 degrees, away from
the edge band. Otherwise the two differ by sampling, not by definition: on a plane of gradient
0.5 with 16 sectors the march equals the closed form (openness 90), RVT reads a sky-view factor
0.022 and an openness 3.3 degrees lower, and the sweep 0.019 and 2.6 degrees lower. Inside the
edge band a ray here stops at the edge, where RVT sees mirrored terrain.</li>
<li>Breaking change: <code>anisotropyAzimuthDegrees</code> used to be a compass azimuth, so the default 315
weighted the north-west sky. It now weights the north-east sky, as RVT does. To keep a previous
compass azimuth <code>t</code>, pass <code>(360 - t) mod 360</code>. Non-integer levels now throw.</li>
</ul>
</li>
<li><code>GPUSolarShadowMask</code> reads the horizon map and a per-frame sun: it interpolates the horizon at
the sun azimuth and writes <code>output.sunVisibility</code>, the solar-disk area fraction above the horizon
(soft penumbra from the disk's angular radius, hard when 0). With <code>slope</code> and <code>aspect</code> from
<code>GPUTerrainDerivatives</code> it also writes <code>output.illumination = ambient * svf + sun * sunVisibility * max(cos(incidence), 0)</code>. <code>horizon</code>, <code>slope</code>, <code>aspect</code>, and <code>skyViewFactor</code> are inputs. One frame
costs two reads per pixel.</li>
<li><code>GPUTerrainCastShadow</code> casts one sun's shadow with no horizon map: an exact sweep along the
per-frame sun azimuth gives the horizon in that direction, then the same solar-disk penumbra as
<code>GPUSolarShadowMask</code>; it writes <code>output.sunVisibility</code>, <code>output.horizonAngle</code>, and
<code>output.outputValidity</code>. The CPU packer <code>getGPUTerrainCastShadowParameterValues(settings, grid)</code>
computes the line geometry, so moving the sun never recompiles. The packer also stores the sun's
ground direction and takes no row direction or cell size mode: the contributor mirrors the line
for its <code>rowDirection: 'north'</code> and, on anisotropic cells, turns the direction into a pixel slope
with its own <code>cellSizeMode</code> and the ground cell of row <code>height / 2</code>. Use it when only one sun
matters and a full horizon map is too large.
This middle-row model is local to the window. Seamless mosaics require explicit
<code>parameterLayout: 'sweep'</code>, shared source-grid/reference settings, and per-frame window origins;
<code>getHaloStage()</code> then returns the chosen radius. Standard settings refuse a halo claim even
with a padded parameter view. See the cast-shadow API section below for its 32-value layout.
The sweep follows a Bresenham-style digital line family through cell centres, rather than
bilinearly sampling an arbitrary continuous azimuth. For native <code>r.sunmask -z</code> parity on
finite square metric grids, use cardinal azimuths, positive altitude, zero curvature,
<code>angularRadiusDegrees: 0</code>, and a radius covering the whole domain. The comparison then
reduces, away from exact grazing, to <code>max((z_i-z_0)/d_i) > tan(altitude)</code>.
The GPU hard step requires <code>altitude > horizonAngle</code>, so equality is shadowed;
GRASS requires terrain strictly above the solar ray, so mathematical equality is sunlit.
Exact grazing therefore needs a separately specified tie convention. The <code>-z</code> flag makes
GRASS include zero-elevation terrain. Other azimuths can use different digital lines;
this is a sampling difference, not a floating-point error allowance.</li>
<li><code>GPUSolarIrradiance</code> integrates a sun path over a horizon map: <code>output.sunHours</code> (disk-visible
hours) and <code>output.insolation</code> (Wh/m², direct normal irradiance times visibility times cos incidence, plus an
optional diffuse term <code>diffuseIrradiance * skyViewFactor * duration</code>). Bind the horizon's
<code>output.irradianceSkyViewFactor</code> as <code>skyViewFactor</code>, not its solid-angle <code>output.skyViewFactor</code>,
which understates diffuse energy (50 instead of 75 Wh/m² for 100 W/m² over one hour under a
uniform 30 degree horizon). The diffuse term is for a horizontal receiver: slope and aspect
enter the direct term only, and no <code>(1 + cos(slope)) / 2</code> factor is added because a horizon
measured on the same terrain already contains the receiver's own slope. Terrain-reflected
radiation is not modelled. <code>getGPUSolarIrradianceSunTable</code> samples
<code>getSolarPosition</code> over a day or a date range (Meinel clear-sky DNI with Kasten–Young air mass by
default). <code>samplesPerNode</code> splits long tables across dispatches.</li>
<li><code>GPUReliefShading</code> computes a hillshade from up to 8 lights with fixed weights or the USGS
multidirectional oblique weighting (<code>lights: 'mdow'</code>, per-pixel <code>sin^2(aspect - azimuth)</code> as in
GDAL <code>-multidirectional</code>). It writes <code>output.hillshade</code>, and blends a Swiss/Imhof relief
(<code>output.relief</code> luminance and packed RGBA8 <code>output.color</code>, plus <code>output.outputValidity</code>) from the hillshade, an optional sky-view factor, an optional texture shade, an
elevation tint ramp of up to 8 stops, and a warm-lit/cool-shaded aspect tint. Imhof options:
<code>lightWeighting: 'imhof-swing'</code> (with the <code>imhofSwing: true</code> prop) swings each light toward the
slope's side by up to <code>imhofSwingDegrees</code> (default 65), an optional <code>curvature</code> raster adds
<code>curvatureStrength * curvature</code>, and <code>contrastStrength</code> with
<code>contrastLowElevation</code>/<code>contrastHighElevation</code> raises contrast with elevation. Zero values keep the
previous output bit for bit.</li>
<li><code>GPUTextureShading</code> approximates Leland Brown's texture shading (fractional Laplacian
<code>|f|^alpha</code>) with a cascade of separable Gaussian levels (<code>sigma_k = baseSigma * 2^k</code>) summed as
weighted band differences. Nodata uses normalized convolution. <code>detail</code> (alpha), the band
weights, and gain are per frame; it writes <code>output.textureShade</code> and <code>output.outputValidity</code>.
<code>hasNodata: false</code> (default <code>true</code>) promises an all-valid
elevation and blurs one channel instead of value plus validity, which is exact and about halves
the blur cost. <code>downsampleLevels</code> (default <code>true</code>) computes levels with incremental sigma of at
least 4 pixels on a 2x decimated grid and upsamples bilinearly; output stays within 1% of the exact
output range (measured 0.1-0.27%), and <code>false</code> restores the exact full-resolution cascade.</li>
<li><code>GPUSolarPosition</code> evaluates the NOAA solar position for every row of a longitude/latitude
column at one per-frame instant. It has no grid (it takes positions): <code>parameters</code> holds the time
and <code>output</code> the azimuth, altitude (with optional refraction), and a daylight flag.
<code>getSolarPosition</code> is the float64 CPU version used to drive the shadow mask.
Both use the <strong>NOAA daily spreadsheet's Meeus-based approximation</strong>, without observer parallax,
rather than NREL SPA. <a href="https://gml.noaa.gov/grad/solcalc/calcdetails.html">NOAA's published accuracy</a>
is about one minute for sunrise/set within ±72° latitude and ten minutes outside; NOAA does
not give a universal 0.01° angular guarantee. Float32 orbital-angle error increases with time
from J2000, and azimuth is ill-conditioned near zenith/nadir. Time is transported as separate
whole days and day fraction. <code>daylight</code> compares <strong>geometric sun-center altitude</strong> with a strict
<code>></code> threshold (default −0.833°), independently of the optional refraction applied to <code>altitude</code>.
Near that threshold, NOAA and SPA can legitimately produce different daylight flags.</li>
</ul>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> sun</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> getSolarPosition</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(Date.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">now</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(), longitude, latitude);</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> grid</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> makeGPURasterMetadata</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({width, height, cellSize: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">30</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">});</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUTerrainHorizon</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  grid, elevation, parameters: horizonParameters.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">importToGraph</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(graph),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  directionCount: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">16</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, maximumRadius: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">256</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, stepGrowth: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">1.1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  output: {horizon, skyViewFactor}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUSolarShadowMask</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  grid, directionCount: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">16</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, horizon, parameters: shadowParameters.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">importToGraph</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(graph),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  slope, aspect, skyViewFactor, output: {illumination}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">shadowParameters.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">write</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">getGPUSolarShadowMaskParameterValues</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  azimuthDegrees: sun.azimuthDegrees, altitudeDegrees: sun.altitudeDegrees, ambientIntensity: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">0.3</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span></code></pre>
<p>The horizon map needs <code>width * height * directionCount * 4</code> bytes in one binding (268 MB for a
2048² tile with 16 sectors). The contributor checks <code>maxStorageBufferBindingSize</code>. For seamless
march mosaics, pass <code>horizon.getHaloStage()</code> to the published halo assembler: it returns
<code>{requiredHalo: maximumRadius + 1}</code>. The extra cell is required because bilinear sampling checks
the next corner's validity even at zero weight. Horizon instances expose no <code>requiredHalo</code> field.
For sweep, <code>getHaloStage()</code> returns <code>{requiredHalo: maximumRadius}</code> with explicit
<code>parameterLayout: 'sweep'</code>. Both algorithms require that layout for geographic or Web Mercator
halos. It uses a common source-row georeference and the integer source row, so independently
rounded tile edges cannot change which side of a distance cutoff contains a sample. Standard
geographic settings remain usable on whole windows but refuse a halo claim. The march radius
is at most <code>Number.MAX_SAFE_INTEGER - 1</code>, keeping its extra halo cell representable. The
constructor captures that radius; reusing or changing the original props cannot shrink a live halo.</p>
<p>Pack <code>{sweepReferenceCellSize: [groundCellX, groundCellY], sweepSourceGrid, sweepOrigin}</code> with
<code>getGPUTerrainHorizonParameterValues(settings, windowGrid)</code> and pass a view of at least 24 values
with <code>parameterLayout: 'sweep'</code>. <code>GPU_TERRAIN_HORIZON_SWEEP_PARAMETER_LENGTH</code> is 24: base values
0..7, anisotropy 8..11, reference ground cells 12..13, origin coordinates modulo 65536 at 14..15,
source cell sizes 16..17, source north/south edges 18..19, source height and window source-row
offset 20..21, reserved zeros 22..23. The source height is at most 16777215 and the entire window
must lie inside the source grid. Origin defaults to [0, 0] relative to that shared overview grid.
Choose reference ground cells at a fixed mosaic location; use the same source grid and reference
pair for every window and update them consistently per frame. Both cells must remain finite and
positive after float32 packing. The CPU oracle rounds reference components to those same bytes
before curvature selection, retaining the nearest sample on a tie. Comparisons allow ordinary
float32 numerical tolerance. Existing 8- and 12-value settings remain usable in oversized targets;
neither the packer nor the contributor interprets padding as shared-reference data.</p>
<p>March samples split the ray displacement into integer and fractional parts before adding the
integer pixel origin. Bounds checks, validity corners and interpolation use that split in both
the fused and per-sector paths, so changing the window-local coordinate magnitude cannot change
which no-data corner is checked.</p>
<p>March construction bounds <code>stepDistances</code> by the ceiling of the pixel-centre diagonal (at least
one). This filters the original schedule without adding an off-schedule endpoint. Sweep exposes
an empty <code>stepDistances</code> array and constructs no march schedule or WGSL distance literals.
The standalone <code>getGPUTerrainHorizonStepDistances(radius, growth, maximumSampleDistance?)</code>
retains the full schedule by default; supply the optional bound to limit materialization.</p>
<p>Typical 2048² timings on an Apple-silicon laptop are about 100 ms for the horizon
(16 sectors, radius 256, growth 1.1), 3.4 ms per frame for the shadow mask, 3.7 ms for relief
shading, and 28 ms for six-level texture shading.</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUTerrainHorizon</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  grid, elevation, parameters, directionCount: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">16</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, maximumRadius: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">1023</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  algorithm: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'sweep'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, horizonFormat: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'unorm16'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, output: {horizon, irradianceSkyViewFactor}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">const</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8"> {</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">values</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">sampleCount</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">=</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> getGPUSolarIrradianceSunTable</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  longitude: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">8.2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, latitude: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">46.8</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, start: Date.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">UTC</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">2026</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">5</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">21</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">), end: Date.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">UTC</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">2026</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">5</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">22</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">});</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">sunTableBuffer.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">write</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(values);</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUSolarIrradiance</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  grid, directionCount: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">16</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, horizonFormat: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'unorm16'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, horizon,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  sunTable: sunTableBuffer.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">importToGraph</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(graph), sampleCapacity: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">288</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, parameters: irradianceParameters,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  slope, aspect, skyViewFactor: irradianceSkyViewFactor, output: {sunHours, insolation}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span></code></pre>
<p>At 1024² with 16 sectors the sweep horizon takes 41 ms at full radius (98.6 ms at radius 256), and
a 288-sample day of irradiance takes about 29 ms.</p>
<h4 id="gputerrainhorizon"><code>GPUTerrainHorizon</code></h4>
<p>Marches 16 azimuth sectors from every cell over an elevation grid to find horizon angles, then reduces them to sky-view factor and openness.</p>
<p>::lens{scenario=gpu-terrain.terrain-horizon rung=result caption="Sky-view factor, 1 on open ridges; the most enclosed cell is marked."}</p>
<p>{/* parameters: GPUTerrainHorizon generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h5 id="gputerrainhorizon-parameters"><code>GPUTerrainHorizon</code> parameters</h5>
<ul>
<li>Packer: <code>getGPUTerrainHorizonParameterValues(settings, grid)</code>, writes <code>GPU_TERRAIN_HORIZON_PARAMETER_LENGTH</code> (8) <code>float32</code> values.
<ul>
<li><code>grid</code> (<code>GPURasterMetadata</code>): the frame grid, which supplies the cell size and row edges</li>
</ul>
</li>
</ul>
<p>Settings of <code>getGPUTerrainHorizonParameterValues</code>:</p>
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
<td><code>zFactor</code> (optional)</td>
<td><code>number</code></td>
<td>Elevation multiplier. Defaults to 1.</td>
</tr>
<tr>
<td><code>curvatureCoefficient</code> (optional)</td>
<td><code>number</code></td>
<td>Earth curvature and refraction drop in 1/meters: a sample at ground distance <code>d</code> is lowered by <code>c * d^2</code>. See <code>getGPUTerrainCurvatureCoefficient</code>. Defaults to 0.</td>
</tr>
<tr>
<td><code>maximumDistance</code> (optional)</td>
<td><code>number</code></td>
<td>Ground search radius in meters; <code>&#x3C;= 0</code> searches the full pixel radius. Defaults to 0.</td>
</tr>
<tr>
<td><code>sweepReferenceCellSize</code> (optional)</td>
<td><code>readonly [number, number]</code></td>
<td>Shared ground metres per overview pixel <code>[x, y]</code> for sweep geometry and curvature decisions. Use the same positive finite pair for every window of a mosaic, including the whole-grid evaluation. Required for a seamless sweep halo in latitude-dependent cell-size modes. With <code>sweepSourceGrid</code>, packs the explicit 24-value sweep layout; this pair occupies 12..13.</td>
</tr>
<tr>
<td><code>sweepOrigin</code> (optional)</td>
<td><code>readonly [number, number]</code></td>
<td>Source-window origin in overview pixels, relative to <code>sweepSourceGrid</code>; per frame. Defaults to [0, 0].</td>
</tr>
<tr>
<td><code>sweepSourceGrid</code> (optional)</td>
<td><code>GPURasterMetadata</code></td>
<td>Shared full overview grid for row spacing and distance cutoffs; required with sweep settings. Every window uses the same source grid per frame. Its height is at most 16777215 so global source rows can be combined exactly as integers before conversion to float32.</td>
</tr>
<tr>
<td><code>anisotropyAzimuthDegrees</code> (optional)</td>
<td><code>number</code></td>
<td>Direction of maximum weight for <code>anisotropicSkyViewFactor</code>, in RVT's convention: the same number as RVT's <code>asvf_dir</code> / <code>a_main_direction</code> means the same direction. RVT direction angles run counterclockwise from north (0 north, 90 west, 180 south, 270 east), so <code>A</code> is compass azimuth <code>(360 - A) mod 360</code>. Defaults to <code>GPU_TERRAIN_HORIZON_DEFAULT_ANISOTROPY_AZIMUTH_DEGREES</code> (315, RVT's default, which is compass north-east). This is the only counterclockwise angle in gpu-terrain: sector azimuths, sun azimuths and aspect are compass azimuths, clockwise from north.</td>
</tr>
<tr>
<td><code>anisotropyLevel</code> (optional)</td>
<td><code>number</code></td>
<td>Weight exponent of <code>anisotropicSkyViewFactor</code>: a non-negative integer, RVT's <code>a_poly_level</code>. RVT's presets are low (level 4, minimum weight 0.4) and high (level 8, minimum weight 0.1). The power is signed as in RVT, <code>cos((a - A) / 2)^level</code>, so an odd level gives a weight below the minimum weight on the far side of the main direction, exactly like RVT. Defaults to <code>GPU_TERRAIN_HORIZON_DEFAULT_ANISOTROPY_LEVEL</code>.</td>
</tr>
<tr>
<td><code>anisotropyMinimumWeight</code> (optional)</td>
<td><code>number</code></td>
<td>Weight floor of <code>anisotropicSkyViewFactor</code> in <code>[0, 1]</code> (RVT: low 0.4, high 0.1). Defaults to <code>GPU_TERRAIN_HORIZON_DEFAULT_ANISOTROPY_MINIMUM_WEIGHT</code>.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h4 id="how-it-computes-20">How it computes</h4>
<p>{/* lens:GPUTerrainHorizon */}</p>
<p>::lens{scenario=gpu-terrain.terrain-horizon rung=datum datum=2972 caption="A valley cell: the highest elevation angle in each of 16 sectors, then their mean sine."}
::lens{scenario=gpu-terrain.terrain-horizon rung=formula caption="Sky view is one minus the mean sine of the non-negative horizon angles."}
::lens{scenario=gpu-terrain.terrain-horizon rung=kernel caption="One fused kernel marches every sector of a cell, then a finalize kernel writes the outputs."}
::lens{scenario=gpu-terrain.terrain-horizon rung=graph caption="Elevation canonicalization, a fused horizon march and a sky-view finalize node; no loop."}</p>
<h4 id="gpusolarshadowmask"><code>GPUSolarShadowMask</code></h4>
<p>Turns a horizon map and a sun position into the visible fraction of the solar disk for every cell.</p>
<p>::lens{scenario=gpu-terrain.solar-shadow-mask rung=result caption="Sun visibility at 06:45 PDT; west-facing slopes in ridge shadow are dark."}</p>
<p>{/* parameters: GPUSolarShadowMask generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h5 id="gpusolarshadowmask-parameters"><code>GPUSolarShadowMask</code> parameters</h5>
<ul>
<li>Packer: <code>getGPUSolarShadowMaskParameterValues(settings)</code>, writes <code>GPU_SOLAR_SHADOW_MASK_PARAMETER_LENGTH</code> (8) <code>float32</code> values.</li>
</ul>
<p>Settings of <code>getGPUSolarShadowMaskParameterValues</code>:</p>
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
<td><code>azimuthDegrees</code></td>
<td><code>number</code></td>
<td>Direction of the sun, degrees clockwise from north. See <code>getSolarPosition</code>.</td>
</tr>
<tr>
<td><code>altitudeDegrees</code></td>
<td><code>number</code></td>
<td>Elevation of the sun center above the astronomical horizon in degrees.</td>
</tr>
<tr>
<td><code>angularRadiusDegrees</code> (optional)</td>
<td><code>number</code></td>
<td>Angular radius of the solar disk in degrees: 0 gives hard shadows. Defaults to <code>GPU_SOLAR_DISK_ANGULAR_RADIUS_DEGREES</code>.</td>
</tr>
<tr>
<td><code>sunIntensity</code> (optional)</td>
<td><code>number</code></td>
<td>Direct light multiplier for <code>illumination</code>. Defaults to 1.</td>
</tr>
<tr>
<td><code>ambientIntensity</code> (optional)</td>
<td><code>number</code></td>
<td>Ambient light multiplier for <code>illumination</code>, scaled by the sky-view factor. Defaults to 0.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h4 id="how-it-computes-21">How it computes</h4>
<p>{/* lens:GPUSolarShadowMask */}</p>
<p>::lens{scenario=gpu-terrain.solar-shadow-mask rung=datum datum=2832 caption="A shadowed cell: the horizon blended at the sun azimuth stands above the sun altitude."}
::lens{scenario=gpu-terrain.solar-shadow-mask rung=formula caption="The solar disk is clipped by the horizon; its visible area fraction is the shadow value."}
::lens{scenario=gpu-terrain.solar-shadow-mask rung=kernel caption="One invocation per cell blends two horizon sectors and clips the solar disk."}
::lens{scenario=gpu-terrain.solar-shadow-mask rung=graph caption="The horizon march feeds one shadow node; the sun is a parameters write."}</p>
<h4 id="gputerraincastshadow"><code>GPUTerrainCastShadow</code></h4>
<p>Computes the visible fraction of the solar disk per cell with one hull sweep toward the sun, without a horizon map.</p>
<p>Cast-shadow instances expose no <code>requiredHalo</code> field. Use <code>castShadow.getHaloStage()</code> with
<code>parameterLayout: 'sweep'</code>: it returns <code>{requiredHalo: maximumRadius}</code> only for the explicit
shared-source model. The standard 16-value model has window-relative digital-line phase and a
window-local reference row, so padding alone cannot make its results seamless. Its halo method
throws, including when the parameter view has 32 or more words.</p>
<p>For a mosaic, pack <code>sweepReferenceCellSize: [groundCellX, groundCellY]</code>, <code>sweepSourceGrid</code>, and
each window's <code>sweepOrigin</code> alongside the sun settings. The origin is in overview pixels relative
to that common source grid, default [0, 0]. Use the same positive finite reference pair and source
grid for every window (including whole-grid comparisons), and update them consistently per frame.
The source grid's height is at most 16777215, and every window must fit inside it. This anchors
the digital-line phase, curvature decisions and source-row spacing for geographic/Web Mercator
angles and distance cutoffs. Sun and placement updates reuse the same compiled graph.</p>
<p>Those explicit settings make <code>getGPUTerrainCastShadowParameterValues(settings, windowGrid)</code> return
<code>GPU_TERRAIN_CAST_SHADOW_SWEEP_PARAMETER_LENGTH</code> (32) values: the existing solar fields in 0..15,
reference ground cells 16..17, origin modulo 65536 in 18..19, source native cell sizes 20..21,
source north/south edges 22..23, source height and window source-row offset 24..25, reserved zeros
26..31. Ordinary settings still write only 16 values, even in larger caller-owned targets;
spare capacity never selects a layout. Set the same explicit <code>maximumRadius</code> for all mosaic
windows: its default, the window diagonal, varies with window size. Keep only the core outputs
covered by the returned halo. Float32 numerical comparisons use a tolerance rather than requiring
bit-identical angles; visibility near an exact solar-disk boundary retains the documented tie rule.</p>
<p>::lens{scenario=gpu-terrain.terrain-cast-shadow rung=result caption="Sun visibility from the sweep; the shadow edges match the horizon-map result."}</p>
<p>{/* parameters: GPUTerrainCastShadow generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h5 id="gputerraincastshadow-parameters"><code>GPUTerrainCastShadow</code> parameters</h5>
<ul>
<li>Packer: <code>getGPUTerrainCastShadowParameterValues(settings, grid)</code>, writes <code>GPU_TERRAIN_CAST_SHADOW_PARAMETER_LENGTH</code> (16) <code>float32</code> values.
<ul>
<li><code>grid</code> (<code>GPURasterMetadata</code>): the frame grid, which supplies the cell size and row edges</li>
</ul>
</li>
</ul>
<p>Settings of <code>getGPUTerrainCastShadowParameterValues</code>:</p>
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
<td><code>azimuthDegrees</code></td>
<td><code>number</code></td>
<td>Direction of the sun, degrees clockwise from north. See <code>getSolarPosition</code>.</td>
</tr>
<tr>
<td><code>altitudeDegrees</code></td>
<td><code>number</code></td>
<td>Elevation of the sun center above the astronomical horizon in degrees.</td>
</tr>
<tr>
<td><code>angularRadiusDegrees</code> (optional)</td>
<td><code>number</code></td>
<td>Angular radius of the solar disk in degrees: 0 gives hard shadows. Defaults to <code>GPU_SOLAR_DISK_ANGULAR_RADIUS_DEGREES</code>.</td>
</tr>
<tr>
<td><code>zFactor</code> (optional)</td>
<td><code>number</code></td>
<td>Elevation multiplier. Defaults to 1.</td>
</tr>
<tr>
<td><code>curvatureCoefficient</code> (optional)</td>
<td><code>number</code></td>
<td>Earth curvature and refraction drop in 1/meters, see <code>GPUTerrainHorizonSettings</code>. Defaults to 0.</td>
</tr>
<tr>
<td><code>maximumDistance</code> (optional)</td>
<td><code>number</code></td>
<td>Ground search radius in meters; <code>&#x3C;= 0</code> is unlimited. Defaults to 0.</td>
</tr>
<tr>
<td><code>sweepReferenceCellSize</code> (optional)</td>
<td><code>readonly [number, number]</code></td>
<td>Shared positive ground metres per overview pixel <code>[x, y]</code> for line geometry and curvature. Together with <code>sweepSourceGrid</code>, selects the 32-value sweep layout (this pair is at 16..17). Use the same pair in every window of a mosaic and update it consistently per frame.</td>
</tr>
<tr>
<td><code>sweepOrigin</code> (optional)</td>
<td><code>readonly [number, number]</code></td>
<td>Per-frame window origin in overview pixels relative to <code>sweepSourceGrid</code>. Defaults to [0, 0].</td>
</tr>
<tr>
<td><code>sweepSourceGrid</code> (optional)</td>
<td><code>GPURasterMetadata</code></td>
<td>Common full overview grid for row spacing and distance cutoffs. Required with sweep settings; its height is at most 16777215, and every window must lie inside it. Use the same grid in each window, including the whole-grid evaluation.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h4 id="how-it-computes-22">How it computes</h4>
<p>{/* lens:GPUTerrainCastShadow */}</p>
<p>::lens{scenario=gpu-terrain.terrain-cast-shadow rung=datum datum=2832 caption="A shadowed cell: the horizon angle along the sun line exceeds the sun altitude."}
::lens{scenario=gpu-terrain.terrain-cast-shadow rung=formula caption="The horizon toward the sun is the largest elevation tangent along the sun line."}
::lens{scenario=gpu-terrain.terrain-cast-shadow rung=kernel caption="The sweep kernel walks digital lines along the sun azimuth and clips the solar disk."}
::lens{scenario=gpu-terrain.terrain-cast-shadow rung=graph caption="Elevation canonicalization and one sweep node; the sun azimuth sets the line family."}</p>
<h4 id="gpusolarirradiance"><code>GPUSolarIrradiance</code></h4>
<p>Sums sun hours and direct insolation over a day of sun positions from a horizon map.</p>
<p>::lens{scenario=gpu-terrain.solar-irradiance rung=result caption="Direct insolation on the sloped surface over 5 August 2021; north slopes and ridge shade receive least."}</p>
<p>{/* parameters: GPUSolarIrradiance generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h5 id="gpusolarirradiance-parameters"><code>GPUSolarIrradiance</code> parameters</h5>
<ul>
<li>Packer: <code>getGPUSolarIrradianceParameterValues(settings)</code>, writes <code>GPU_SOLAR_IRRADIANCE_PARAMETER_LENGTH</code> (8) <code>float32</code> values.</li>
</ul>
<p>Settings of <code>getGPUSolarIrradianceParameterValues</code>:</p>
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
<td><code>sampleCount</code></td>
<td><code>number</code></td>
<td>Number of valid sun table rows, an integer in <code>[0, sampleCapacity]</code>.</td>
</tr>
<tr>
<td><code>angularRadiusDegrees</code> (optional)</td>
<td><code>number</code></td>
<td>Angular radius of the solar disk in degrees: 0 gives hard shadows. Defaults to <code>GPU_SOLAR_DISK_ANGULAR_RADIUS_DEGREES</code>.</td>
</tr>
<tr>
<td><code>diffuseIrradiance</code> (optional)</td>
<td><code>number</code></td>
<td>Diffuse horizontal irradiance of an open site in W/m^2, added while the sun is up and scaled by <code>GPUSolarIrradianceProps.skyViewFactor</code> when one is bound (the irradiance sky-view factor, not the solid-angle one). Defaults to 0.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h4 id="how-it-computes-23">How it computes</h4>
<p>{/* lens:GPUSolarIrradiance */}</p>
<p>::lens{scenario=gpu-terrain.solar-irradiance rung=datum datum=2908 caption="A shaded cell: 96 quarter-hour sun rows, each clipped by the horizon at its azimuth."}
::lens{scenario=gpu-terrain.solar-irradiance rung=formula caption="Insolation sums DNI times visibility times incidence times duration over the sun table."}
::lens{scenario=gpu-terrain.solar-irradiance rung=kernel caption="Each invocation loops over the 96 table rows with two horizon reads each."}
::lens{scenario=gpu-terrain.solar-irradiance rung=graph caption="Horizon march and Horn slope and aspect feed one accumulation node that reads the sun table."}</p>
<h4 id="gpureliefshading"><code>GPUReliefShading</code></h4>
<p>Computes a multidirectional hillshade and a relief luminance from Horn normals of an elevation grid.</p>
<p>::lens{scenario=gpu-terrain.relief-shading rung=result caption="Relief luminance from four USGS lights weighted by aspect."}</p>
<p>{/* parameters: GPUReliefShading generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h5 id="gpureliefshading-parameters"><code>GPUReliefShading</code> parameters</h5>
<ul>
<li>Packer: <code>getGPUReliefShadingParameterValues(settings, grid)</code>, writes <code>GPU_RELIEF_SHADING_PARAMETER_LENGTH</code> (80) <code>float32</code> values.
<ul>
<li><code>grid</code> (<code>GPURasterMetadata</code>): the frame grid, which supplies the cell size and row edges</li>
</ul>
</li>
</ul>
<p>Settings of <code>getGPUReliefShadingParameterValues</code>:</p>
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
<td><code>zFactor</code> (optional)</td>
<td><code>number</code></td>
<td>Elevation multiplier applied to derivatives. Defaults to 1.</td>
</tr>
<tr>
<td><code>lights</code> (optional)</td>
<td><code>'mdow' | readonly GPUReliefShadingLight[]</code></td>
<td>Lights, or <code>'mdow'</code> for the USGS set: azimuths 225, 270, 315, 360 at 30 degrees altitude with <code>'aspect'</code> weighting. Defaults to <code>'mdow'</code>.</td>
</tr>
<tr>
<td><code>lightWeighting</code> (optional)</td>
<td><code>GPUReliefShadingWeighting</code></td>
<td>Light weighting. Defaults to <code>'aspect'</code> for <code>'mdow'</code> and <code>'fixed'</code> for custom lights.</td>
</tr>
<tr>
<td><code>imhofSwingDegrees</code> (optional)</td>
<td><code>number</code></td>
<td>Largest azimuth swing of <code>'imhof-swing'</code> weighting in degrees (settings slot 19). Defaults to 65 for <code>'imhof-swing'</code> and 0 otherwise; ignored by the other weightings. 0 disables the swing.</td>
</tr>
<tr>
<td><code>curvatureStrength</code> (optional)</td>
<td><code>number</code></td>
<td>Multiplier of the optional <code>curvature</code> raster added to the shade (settings slot 76): <code>shade += curvatureStrength * curvature</code>. Defaults to 1; ignored when no curvature is bound.</td>
</tr>
<tr>
<td><code>contrastLowElevation</code> (optional)</td>
<td><code>number</code></td>
<td>Elevation (calibrated units, before <code>zFactor</code>) where the elevation-dependent contrast starts fading in (settings slot 77). Defaults to 0.</td>
</tr>
<tr>
<td><code>contrastHighElevation</code> (optional)</td>
<td><code>number</code></td>
<td>Elevation where the contrast reaches full strength (slot 78). When it is not above <code>contrastLowElevation</code> the contrast steps at the low elevation. Defaults to 0.</td>
</tr>
<tr>
<td><code>contrastStrength</code> (optional)</td>
<td><code>number</code></td>
<td>Strength in <code>[0, 1]</code> of the elevation-dependent contrast (slot 79): with <code>k = mix(1 - strength, 1, smoothstep(low, high, z))</code> the shade becomes <code>mix(GPU_RELIEF_SHADING_CONTRAST_PIVOT, shade, k)</code>, so low ground is flattened toward grey and high ground keeps full relief. Defaults to 0, which is off and leaves output unchanged.</td>
</tr>
<tr>
<td><code>hillshadeStrength</code> (optional)</td>
<td><code>number</code></td>
<td>Blend of the hillshade into <code>relief</code>: 0 ignores it, 1 uses it fully. Defaults to 1.</td>
</tr>
<tr>
<td><code>skyViewStrength</code> (optional)</td>
<td><code>number</code></td>
<td>Blend of the sky-view factor into <code>relief</code>. Defaults to 1 (only when the input is bound).</td>
</tr>
<tr>
<td><code>textureShadeStrength</code> (optional)</td>
<td><code>number</code></td>
<td>Multiplier of the texture shade added to <code>relief</code>. Defaults to 0.5 (only when bound).</td>
</tr>
<tr>
<td><code>exposure</code> (optional)</td>
<td><code>number</code></td>
<td>Final <code>relief</code> multiplier before clamping to <code>[0, 1]</code>. Defaults to 1.</td>
</tr>
<tr>
<td><code>tintStrength</code> (optional)</td>
<td><code>number</code></td>
<td>Imhof aspect tint strength in <code>[0, 1]</code>. Defaults to 0.3.</td>
</tr>
<tr>
<td><code>warmColor</code> (optional)</td>
<td><code>readonly [number, number, number]</code></td>
<td>Tint toward which lit slopes shift. Defaults to <code>[1, 0.95, 0.85]</code>.</td>
</tr>
<tr>
<td><code>coolColor</code> (optional)</td>
<td><code>readonly [number, number, number]</code></td>
<td>Tint toward which shaded slopes shift. Defaults to <code>[0.8, 0.85, 1]</code>.</td>
</tr>
<tr>
<td><code>elevationStops</code> (optional)</td>
<td><code>readonly GPUReliefShadingStop[]</code></td>
<td>Ascending elevation tint stops; empty gives a 0.85 grey base. Defaults to <code>[]</code>.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h4 id="how-it-computes-24">How it computes</h4>
<p>{/* lens:GPUReliefShading */}</p>
<p>::lens{scenario=gpu-terrain.relief-shading rung=datum datum=2907 caption="The steepest cell's 3×3 window and the four light weights."}
::lens{scenario=gpu-terrain.relief-shading rung=formula caption="Aspect-weighted mean of four lights, then the hillshade strength and exposure blend."}
::lens{scenario=gpu-terrain.relief-shading rung=kernel caption="The hillshade kernel loops over the lights; the compose kernel blends and clamps."}
::lens{scenario=gpu-terrain.relief-shading rung=graph caption="Two Sobel passes feed the hillshade node, then the compose node."}</p>
<h4 id="gputextureshading"><code>GPUTextureShading</code></h4>
<p>Approximates Brown's fractional Laplacian of an elevation grid with a Gaussian band-pass pyramid.</p>
<p>::lens{scenario=gpu-terrain.texture-shading rung=result caption="Texture shade in metres; ridges are positive and hollows negative."}</p>
<p>{/* parameters: GPUTextureShading generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h5 id="gputextureshading-parameters"><code>GPUTextureShading</code> parameters</h5>
<ul>
<li>Packer: <code>getGPUTextureShadingParameterValues(settings?)</code>, writes <code>GPU_TEXTURE_SHADING_PARAMETER_LENGTH</code> (12) <code>float32</code> values.</li>
</ul>
<p>Settings of <code>getGPUTextureShadingParameterValues</code>:</p>
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
<td><code>detail</code> (optional)</td>
<td><code>number</code></td>
<td>Fractional Laplacian order <code>alpha</code> (Leland Brown's "detail"): band <code>k</code> gets weight <code>2^(-k * detail)</code>, which approximates the frequency response <code>|f|^detail</code>. 0 weighs every band equally; 1 is the half Laplacian <code>|f|</code>; 2 approaches a Laplacian, whose response is <code>|f|^2</code>. Defaults to 0.5.</td>
</tr>
<tr>
<td><code>gain</code> (optional)</td>
<td><code>number</code></td>
<td>Output multiplier. Defaults to 1.</td>
</tr>
<tr>
<td><code>levelWeights</code> (optional)</td>
<td><code>readonly number[]</code></td>
<td>Explicit per-band weights (up to 8), overriding <code>detail</code>.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h4 id="how-it-computes-25">How it computes</h4>
<p>{/* lens:GPUTextureShading */}</p>
<p>::lens{scenario=gpu-terrain.texture-shading rung=datum datum=3181 caption="A ridge cell: the band-pass differences at five scales and their weights."}
::lens{scenario=gpu-terrain.texture-shading rung=formula caption="Weighted band differences of five Gaussian blurs, scaled by the gain."}
::lens{scenario=gpu-terrain.texture-shading rung=kernel caption="Each level accumulates its weighted band into one buffer; finalize applies gain and validity."}
::lens{scenario=gpu-terrain.texture-shading rung=graph caption="An anchor and premultiply node, then blur passes and an accumulate node per level."}</p>
<h4 id="gpusolarposition"><code>GPUSolarPosition</code></h4>
<p>Computes the sun azimuth, altitude and daylight flag for each longitude and latitude row at one instant.</p>
<p>::lens{scenario=gpu-terrain.solar-position rung=result caption="Sun altitude at eight northern California sites at 13:00 PDT; the table below lists all outputs."}</p>
<p>{/* parameters: GPUSolarPosition generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h5 id="gpusolarposition-parameters"><code>GPUSolarPosition</code> parameters</h5>
<ul>
<li>Packer: <code>getGPUSolarPositionParameterValues(settings)</code>, writes <code>GPU_SOLAR_POSITION_PARAMETER_LENGTH</code> (4) <code>float32</code> values.</li>
</ul>
<p>Settings of <code>getGPUSolarPositionParameterValues</code>:</p>
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
<td><code>timestamp</code></td>
<td><code>number | Date</code></td>
<td>Unix epoch milliseconds (UTC) or a <code>Date</code>.</td>
</tr>
<tr>
<td><code>daylightAltitudeDegrees</code> (optional)</td>
<td><code>number</code></td>
<td>Geometric altitude above which <code>daylight</code> is 1. Defaults to <code>GPU_SOLAR_SUNRISE_ALTITUDE_DEGREES</code>; use -6, -12, or -18 for civil, nautical, or astronomical twilight.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h4 id="how-it-computes-26">How it computes</h4>
<p>{/* lens:GPUSolarPosition */}</p>
<p>::lens{scenario=gpu-terrain.solar-position rung=datum datum=0 caption="Greenville at 20:00 UTC: hour angle, declination and the resulting angles."}
::lens{scenario=gpu-terrain.solar-position rung=formula caption="NOAA hour angle, altitude and azimuth with refraction, in float32."}
::lens{scenario=gpu-terrain.solar-position rung=kernel caption="One invocation per site evaluates the full NOAA series."}
::lens{scenario=gpu-terrain.solar-position rung=graph caption="A single kernel node; time is a parameters write."}</p>
<h3 id="gpusimplelocalrelief-gpumultiscalerelief-gpulocaldominance-and-gpureliefblend"><code>GPUSimpleLocalRelief</code>, <code>GPUMultiScaleRelief</code>, <code>GPULocalDominance</code>, and <code>GPUReliefBlend</code></h3>
<p>Archaeological and micro-relief visualizations with Relief Visualization Toolbox (RVT; Kokalj and
Somrak 2019) parity. The float64 test oracles follow RVT-py's formulas, including edge padding, NaN
handling and Python's half-even rounding.</p>
<ul>
<li><code>GPUSimpleLocalRelief</code> (SLRM; Hesse 2010): <code>ve * (z - mean(z))</code> over a <code>(2r + 1)²</code> window with
edge-clamped coordinates and nodata excluded. On grids up to
<code>getGPUTerrainSummedAreaMaximumCellCount(device)</code> cells (about 3344 x 3344 on a default device) the mean is an O(1) box query
over an exact 64-bit summed-area table of elevations quantized to <code>quantum</code> (a power of two,
default <code>2^-16</code>), with RVT's edge replication applied as edge-strip weights and the deviation formed
from exact integers; the error against RVT is at most <code>quantum / 2</code> (zero for <code>|z| >= 128</code>) plus a
few float32 ULP, at any radius up to 23170. The GPU selects the first finite valid elevation,
snaps it to the quantum, and subtracts it before building the table. Only offsets from that
anchor of magnitude <code>2^31 * quantum</code> or more (32768 by default) are invalid; absolute elevation
is unrestricted. Larger grids use a separable O(radius) filter with anchored,
centre-relative sums, so any radius keeps sub-millimetre precision at alpine elevations. The radius is a
per-frame setting: <code>maximumRadius</code> (compile time) sizes <code>requiredHalo</code> and bounds the loops, and
<code>getGPUSimpleLocalReliefParameterValues({radius})</code> picks any integer in <code>[1, maximumRadius]</code> per
frame with the same pipelines (an omitted radius uses <code>maximumRadius</code>).</li>
<li><code>GPUMultiScaleRelief</code> (MSRM; Orengo and Petrie 2018): RVT's radii from <code>resolution</code>,
<code>featureMinimum</code>, <code>featureMaximum</code> and <code>scalingFactor</code> (<code>getGPUMultiScaleReliefRadii</code>). RVT's sum of
consecutive low-pass differences telescopes, so the contributor runs only two mean filters, which
share one summed-area table on grids within the cell cap (same <code>quantum</code> option and bounds as SLRM).
<code>cellSizeMode</code> (default <code>'uniform'</code>, unchanged square windows) accepts <code>'geographic'</code> and
<code>'web-mercator'</code> on axis-aligned grids: the RVT index ladder is unchanged, each nonzero radius <code>k^s</code>
becomes <code>k^s * resolution</code> ground metres, and every row converts it to rectangular pixel half-widths
with its WGS84 ground cell (rounded, at least 1). The summed-area path evaluates these rectangles
exactly; the separable path (large grids) averages rows filtered with their own horizontal radius.
<code>requiredHalo</code> is the largest coarsest half-width over all rows.</li>
<li><code>GPULocalDominance</code> (Hesse 2016): the mean dominance of the observer (<code>observerHeight</code> above the
cell) over an annulus of baked shifts. Distances are in pixels, as in RVT.</li>
<li><code>GPUReliefBlend</code> (VAT): up to five float32 layers, bottom to top, each stretched
<code>clamp((x - min)/(max - min))</code>, optionally inverted, and blended (normal, multiply, screen, overlay,
soft light, luminosity) with an opacity. Presets <code>GPU_RELIEF_BLEND_VAT_ARCHAEOLOGICAL</code> (hillshade;
slope 0-50° inverted, luminosity 50 %; positive openness 68-93°, overlay 50 %; SVF 0.7-1, multiply
25 %) and <code>GPU_RELIEF_BLEND_VAT_FLAT</code>. Feed it from <code>GPUReliefShading</code>, <code>GPUTerrainDerivatives</code> and
<code>GPUTerrainHorizon</code>. Percent-clip stretches need a histogram, so pass min and max (for example from
<code>GPURasterStatistics</code>).</li>
</ul>
<p>At 1024²: SLRM r=20 takes 3.3 ms, MSRM at radius 100 takes 5.4 ms, local dominance with the RVT
defaults (264 taps) takes 84.5 ms, and a four-layer VAT blend takes 1.2 ms. All style values are
per-frame <code>parameters</code>. Each contributor takes <code>grid</code> instead of <code>width</code> and <code>height</code> and writes into
<code>output</code> (<code>relief</code>, <code>dominance</code>, <code>blend</code>, <code>color</code>, and <code>outputValidity</code>); the blend <code>layers</code> stay an
input. The blend layer limit is <code>GPU_RELIEF_BLEND_MAXIMUM_LAYER_COUNT</code> and the local dominance
offset limit <code>GPU_LOCAL_DOMINANCE_MAXIMUM_SHIFT_COUNT</code>.</p>
<h4 id="gpusimplelocalrelief"><code>GPUSimpleLocalRelief</code></h4>
<p>Subtracts the local window mean from an elevation raster, leaving small features such as banks and mounds.</p>
<p>::lens{scenario=gpu-terrain.simple-local-relief rung=result caption="Elevation minus the 17 × 17 window mean; ridges are positive, hollows negative."}</p>
<p>{/* parameters: GPUSimpleLocalRelief generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h5 id="gpusimplelocalrelief-parameters"><code>GPUSimpleLocalRelief</code> parameters</h5>
<ul>
<li>Packer: <code>getGPUSimpleLocalReliefParameterValues(settings?)</code>, writes <code>GPU_SIMPLE_LOCAL_RELIEF_PARAMETER_LENGTH</code> (4) <code>float32</code> values.</li>
</ul>
<p>Settings of <code>getGPUSimpleLocalReliefParameterValues</code>:</p>
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
<td><code>verticalExaggeration</code> (optional)</td>
<td><code>number</code></td>
<td>Vertical exaggeration applied to the relief. Defaults to 1.</td>
</tr>
<tr>
<td><code>radius</code> (optional)</td>
<td><code>number</code></td>
<td>Window half-width in pixels, an integer in <code>[1, maximumRadius]</code>. Defaults to <code>maximumRadius</code>. The packer throws for a value that is not an integer of at least 1; a value above <code>maximumRadius</code> written to the buffer by other means is clamped to <code>maximumRadius</code> by the kernel.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h4 id="how-it-computes-27">How it computes</h4>
<p>{/* lens:GPUSimpleLocalRelief */}</p>
<p>::lens{scenario=gpu-terrain.simple-local-relief rung=datum datum=3181 caption="The ridge cell at column 45, row 49 against its 17 × 17 window mean, edge samples included."}
::lens{scenario=gpu-terrain.simple-local-relief rung=formula caption="Centre elevation minus the edge-replicated window mean, times vertical exaggeration."}
::lens{scenario=gpu-terrain.simple-local-relief rung=kernel caption="The deviation kernel reads box sums from the exact summed-area table."}
::lens{scenario=gpu-terrain.simple-local-relief rung=graph caption="Elevation feeds the summed-area table, then one deviation kernel."}</p>
<h4 id="gpumultiscalerelief"><code>GPUMultiScaleRelief</code></h4>
<p>Averages the differences between consecutive low-pass filters of an elevation raster to isolate features between two sizes.</p>
<p>::lens{scenario=gpu-terrain.multi-scale-relief rung=result caption="Relief between feature sizes of 360 and 1200 metres; ridges are positive, hollows negative."}</p>
<p>{/* parameters: GPUMultiScaleRelief generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h5 id="gpumultiscalerelief-parameters"><code>GPUMultiScaleRelief</code> parameters</h5>
<ul>
<li>Packer: <code>getGPUMultiScaleReliefParameterValues(settings?)</code>, writes <code>GPU_MULTI_SCALE_RELIEF_PARAMETER_LENGTH</code> (4) <code>float32</code> values.</li>
</ul>
<p>Settings of <code>getGPUMultiScaleReliefParameterValues</code>:</p>
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
<td><code>verticalExaggeration</code> (optional)</td>
<td><code>number</code></td>
<td>Vertical exaggeration applied to the relief. Defaults to 1.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h4 id="how-it-computes-28">How it computes</h4>
<p>{/* lens:GPUMultiScaleRelief */}</p>
<p>::lens{scenario=gpu-terrain.multi-scale-relief rung=datum datum=2973 caption="A valley cell: its deviations from the radius 1 and radius 5 means, divided by four."}
::lens{scenario=gpu-terrain.multi-scale-relief rung=formula caption="The telescoped sum: the difference of the coarsest and finest deviations over n minus i."}
::lens{scenario=gpu-terrain.multi-scale-relief rung=kernel caption="Two mean filters share one table; a combine kernel scales their difference."}
::lens{scenario=gpu-terrain.multi-scale-relief rung=graph caption="One summed-area table feeds two deviation filters and the combine node."}</p>
<h4 id="gpulocaldominance"><code>GPULocalDominance</code></h4>
<p>Measures how far an observer standing above each cell of an elevation raster rises over rings of surrounding terrain.</p>
<p>::lens{scenario=gpu-terrain.local-dominance rung=result caption="Local dominance in rings of 5 to 10 cells; the most dominant cell is marked."}</p>
<p>{/* parameters: GPULocalDominance generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h5 id="gpulocaldominance-parameters"><code>GPULocalDominance</code> parameters</h5>
<ul>
<li>Packer: <code>getGPULocalDominanceParameterValues(settings?)</code>, writes <code>GPU_LOCAL_DOMINANCE_PARAMETER_LENGTH</code> (4) <code>float32</code> values.</li>
</ul>
<p>Settings of <code>getGPULocalDominanceParameterValues</code>:</p>
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
<td><code>observerHeight</code> (optional)</td>
<td><code>number</code></td>
<td>Height of the observer above the surface, in elevation units. Positive. Defaults to 1.7.</td>
</tr>
<tr>
<td><code>verticalExaggeration</code> (optional)</td>
<td><code>number</code></td>
<td>Vertical exaggeration applied to the elevation. Defaults to 1.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h4 id="how-it-computes-29">How it computes</h4>
<p>{/* lens:GPULocalDominance */}</p>
<p>::lens{scenario=gpu-terrain.local-dominance rung=datum datum=3181 caption="The ridge cell's 144 ring samples: each positive rise adds its ring-weighted share."}
::lens{scenario=gpu-terrain.local-dominance rung=formula caption="Positive rises above each ring sample, ring-weighted and normalized so a flat surface scores one."}
::lens{scenario=gpu-terrain.local-dominance rung=kernel caption="One invocation per cell sums 144 baked ring samples."}
::lens{scenario=gpu-terrain.local-dominance rung=graph caption="Canonical elevation feeds a single sampling node."}</p>
<h4 id="gpureliefblend"><code>GPUReliefBlend</code></h4>
<p>Combines hillshade and slope layers from GPUTerrainDerivatives into one grey image with stretches, inversion and blend modes.</p>
<p>::lens{scenario=gpu-terrain.relief-blend rung=result caption="Hillshade blended with inverted slope into one grey image over the elevation context."}</p>
<p>{/* parameters: GPUReliefBlend generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h5 id="gpureliefblend-parameters"><code>GPUReliefBlend</code> parameters</h5>
<ul>
<li>Packer: <code>getGPUReliefBlendParameterValues(layers)</code>, writes <code>GPU_RELIEF_BLEND_PARAMETER_LENGTH</code> <code>float32</code> values.
<ul>
<li><code>layers</code> (<code>readonly GPUReliefBlendLayerSettings[]</code>)</li>
</ul>
</li>
</ul>
<p>{/* /parameters */}</p>
<h4 id="how-it-computes-30">How it computes</h4>
<p>{/* lens:GPUReliefBlend */}</p>
<p>::lens{scenario=gpu-terrain.relief-blend rung=datum datum=2907 caption="The steepest cell: both layers stretched, the slope inverted, then mixed at 50 percent."}
::lens{scenario=gpu-terrain.relief-blend rung=formula caption="Each layer is stretched, then composited bottom to top with its blend mode and opacity."}
::lens{scenario=gpu-terrain.relief-blend rung=kernel caption="A single kernel stretches each layer and composites them bottom to top."}
::lens{scenario=gpu-terrain.relief-blend rung=graph caption="Derivatives write the layers; the blend node composites them."}</p>
<h3 id="gputerrainflow"><code>GPUTerrainFlow</code></h3>
<p>Turns an elevation grid into filled elevation, D8 directions, cell classes, upstream-cell accumulation and a stream mask.</p>
<p>::lens{scenario=gpu-terrain.terrain-flow rung=result caption="Upstream cells per cell after the fill; the outlet with the most upstream cells is marked."}</p>
<p>Hydrology on an elevation tile: optional Planchon–Darboux depression filling with a per-frame
epsilon, D8 flow directions (ESRI codes, cell-size-aware diagonal distances, uniform, Web Mercator,
or geographic cells) with flat, pit, and outlet classes, flow accumulation in cells or ground area
optionally weighted by a per-cell runoff view, and a stream mask with a per-frame threshold.
With <code>fillEpsilon</code> 0, filling is a tiled spill graph (Barnes 2016): tile-local fill with outlet
labels, a Borůvka spanning tree of the spill graph, and bottleneck paths to the grid boundary, in
O(log n) gated GPU rounds whatever the basin shape, bit-identical to the Planchon–Darboux fill
(and to pysheds <code>fill_depressions</code>); <code>output.status.fillIterationCount</code> then reports Borůvka rounds.
The 32 × 32 fill tile requires 12,320 bytes of workgroup storage: three 4,096-byte arrays and
two flags, each rounded to 16 bytes under WebGPU's limit calculation. Devices below that limit
use 16 × 16 tiles, requiring 3,104 bytes of workgroup storage.
The fill is the greatest solution of <code>W = max(Z, min W(n) + epsilon)</code>, the minimax elevation to the
boundary. With <code>fillEpsilon</code> 0 that equation has lower solutions too (any level between a pit and
its pour point), so a relaxation restarted from a stale surface is not guaranteed to reach the fill;
both paths here start from above and reach it.
A <code>fillEpsilon</code> above 0 selects, per frame on the GPU, the GPU-gated tiled min-relaxation, which is
also used on grids whose <code>8 · (cells + 1)</code> bytes exceed one storage binding. D8 <code>'cells'</code> accumulation without <code>runoff</code> is tree
contraction on the Euler tour of the receiver forest: every cell enumerates its children in fixed
direction order, and gated pointer jumping ranks the <code>2 n</code> tour edges in <code>ceil(log2(2 n)) + 1</code>
rounds, so a cell's count is an exact integer (written as float32) for any stem length and
<code>maximumAccumulationIterations</code> does not apply. Weighted accumulation (<code>runoff</code>, <code>'area'</code>) and the
other routings are a deterministic pull over dependency order with downstream walking, bounded by
<code>maximumAccumulationIterations</code>. Everything stops early on the GPU and reports convergence flags, so
nothing is read back and accumulation is bit-reproducible. <code>maximumFillIterations</code> and
<code>maximumFlatIterations</code> default to 512 (up to 1024); after 96 plain tiled iterations each phase adds
directional row and column sweeps so serpentine and spiral basins converge. Check
<code>output.status.fillConverged</code> and <code>output.status.flatsConverged</code> for longer ones. The optional
one-row outputs <code>output.status.fillIterationCount</code>, <code>output.status.flatsIterationCount</code> (the sum of
the three flat relaxations) and <code>output.status.accumulationIterationCount</code> report the iterations
executed, including the final unchanged one, so a budget can be tuned (the SF terrain used 14 of 256
fill and 2 of 128 accumulation rounds). Every unrolled loop has one gate node per four rounds; rounds
after convergence inside a group are idempotent no-ops, so results and counts are unchanged and the
graph has about a third fewer nodes.</p>
<p>The grid supplies the width, height, cell size and latitude band, so the settings carry only the
algorithm values (<code>fillEpsilon</code>, <code>streamThreshold</code>, <code>flowExponent</code>) and the packer takes the grid as
its second argument. The grid may be rotated but not sheared: <code>GPUTerrainFlow</code>,
<code>GPUTerrainHydrologicIndices</code> and <code>GPUTerrainFlowFractions</code> measure the diagonal D8 step as
<code>sqrt(x² + y²)</code> of the two cell sizes, so they throw at construction when the two pixel axes of the
affine are not orthogonal, as <code>GPUCostDistance</code> does.</p>
<p>When <code>maximumAccumulationIterations</code> runs out on the pull path, unresolved cells keep NaN <code>accumulation</code>, and their
<code>streams</code> word is <code>GPU_TERRAIN_FLOW_STREAM_UNRESOLVED</code> (<code>0xffffffff</code>), never 0: the unresolved cells
are the downstream trunk, so 0 would silently drop the main river. Every consumer of a stream mask
(height above drainage, stream order, stream network, flow distance, the catchment summary and tiled
flow) reads it the same way: a nonzero word is a stream cell, so a mask of Strahler orders works
too, and the unresolved word is a cell of unknown status, never a stream cell.
<code>GPUTerrainHeightAboveDrainage</code> and <code>GPUTerrainStreamOrder</code> propagate the unresolved word.</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> grid</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> makeGPURasterMetadata</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({width, height, cellSize: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">30</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">});</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUTerrainFlow</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  grid, elevation, parameters: parameters.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">importToGraph</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(graph), fillDepressions: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">true</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  output: {flowDirections, accumulation, streams, status: {fillConverged, accumulationConverged}}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">parameters.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">write</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">getGPUTerrainFlowParameterValues</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({fillEpsilon: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">0.01</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, streamThreshold: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">500</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}, grid));</span></span></code></pre>
<p>Weighted and non-D8 accumulation walks downstream after each publication for at most 1024 steps
(<code>TERRAIN_FLOW_MAXIMUM_WALK_LENGTH</code>). That is a performance cliff, not a correctness limit: a drainage path
longer than 1024 cells needs further gated rounds, so convergence within <code>maximumAccumulationIterations</code>
(default 64, at most 1024) can fail on very long unbranched stems and leaves their cells unresolved (NaN).
Exact D8 <code>'cells'</code> accumulation without <code>runoff</code> uses an Euler tour and is not affected.</p>
<p><code>resolveFlats: true</code> routes flow across flats without changing elevations (Barnes, Lehman, and
Mulla 2014). Hop distances toward the flat's draining edge and away from its higher edge are GPU
relaxations over equal-height cells. Each drainable flat cell drains to the equal-height neighbor
whose <code>2 * towardLower + (flatMaximum - awayFromHigher)</code> is strictly lower. Pair it with
<code>fillDepressions</code> and <code>fillEpsilon: 0</code> to drain filled depressions as true flats;
<code>output.status.flatsConverged</code> reports the three relaxations. <code>flowRouting</code> selects how <code>accumulation</code> and <code>streams</code> split flow:
<code>'d8'</code> (default), <code>'d-infinity'</code> (Tarboton 1997 facets), <code>'mfd-freeman'</code> (Freeman 1991,
<code>tan(beta)^p</code>, <code>p = 1.1</code>) or <code>'mfd-quinn'</code> (Quinn et al. 1991, contour-length weighted, <code>p = 1</code>).
Set <code>p</code> per frame with <code>flowExponent</code>. Non-D8 accumulation uses the same deterministic pull, with
each donor's fraction recomputed from the surface, so it is reproducible bit for bit.
<code>flowDirections</code> and <code>cellClasses</code> stay D8.</p>
<p><code>flatContinuation</code> opts into 64 rounds per pass per execution (override with
<code>maximumFlatIterations</code>). Supply <code>lower</code>, <code>higher</code>, and <code>maximum</code>, each with a zero-initialized
<code>uint32</code> state of <code>getGPURasterRelaxationContinuationStateLength(grid)</code> words and a <code>float32</code>
values view with one element per cell. All six views must be distinct imported buffers, since
transients can alias between executions. Completed stages stay frozen; the maximum pass starts
only after the higher-edge distances converge. <code>status.flatsIterationCount</code> is cumulative across
executions. Outputs retain the unresolved contract until all three stages converge. Keep inputs
unchanged while resuming, or set <code>GPU_RASTER_RELAXATION_RESTART_WORD</code> in any stage state to restart
all three. The execution after complete convergence also restarts all three, as cost distance does.</p>
<p>Outputs may share one buffer over disjoint byte ranges. For example, <code>status.fillConverged</code> and
<code>status.accumulationConverged</code> can be two one-row views of an 8-byte summary buffer. Outputs that one
pass writes together (<code>filledElevation</code>, <code>flowDirections</code> and <code>cellClasses</code>; <code>accumulation</code> and
<code>streams</code>) must start in different 256-byte binding windows. No output may share a buffer with
an input.</p>
<h4 id="how-it-computes-31">How it computes</h4>
<p>{/* lens:GPUTerrainFlow */}</p>
<p>::lens{scenario=gpu-terrain.terrain-flow rung=datum datum=1826 caption="A main-stream cell: drops to eight neighbours, diagonals over 1.414 cell sizes, receiver outlined."}
::lens{scenario=gpu-terrain.terrain-flow rung=formula caption="Steepest D8 receiver on the filled surface, the epsilon fill and accumulation as the kernels apply them."}
::lens{scenario=gpu-terrain.terrain-flow rung=iterations caption="Filled surface in each recorded relaxation round; accumulation then resolves in a few pull rounds."}
::lens{scenario=gpu-terrain.terrain-flow rung=kernel caption="The direction kernel: one invocation per cell scans eight neighbours and keeps the steepest drop."}
::lens{scenario=gpu-terrain.terrain-flow rung=graph caption="Fill relaxation loop, direction pass, gated accumulation rounds and the stream mask."}</p>
<p>{/* parameters: GPUTerrainFlow generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h4 id="gputerrainflow-parameters"><code>GPUTerrainFlow</code> parameters</h4>
<ul>
<li>Packer: <code>getGPUTerrainFlowParameterValues(settings, grid)</code>, writes <code>GPU_TERRAIN_FLOW_PARAMETER_LENGTH</code> (8) <code>float32</code> values.
<ul>
<li><code>grid</code> (<code>GPURasterMetadata</code>): the frame grid, which supplies the cell size and row edges</li>
</ul>
</li>
</ul>
<p>Settings of <code>getGPUTerrainFlowParameterValues</code>:</p>
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
<td><code>fillEpsilon</code> (optional)</td>
<td><code>number</code></td>
<td>Minimum rise per D8 step on filled surfaces, in elevation units. Negative or non-finite values act as 0. Defaults to 0. Must exceed the float32 spacing at the elevation magnitude to create drainable gradients; 0 leaves flats.</td>
</tr>
<tr>
<td><code>streamThreshold</code> (optional)</td>
<td><code>number</code></td>
<td>Accumulation at or above which <code>streams</code> is 1. Defaults to <code>Infinity</code> (no streams).</td>
</tr>
<tr>
<td><code>flowExponent</code> (optional)</td>
<td><code>number</code></td>
<td>Exponent <code>p</code> of the multiple-flow-direction weights. Values that are not finite and positive (including the default 0) select the routing's published exponent: 1.1 for <code>'mfd-freeman'</code>, 1 for <code>'mfd-quinn'</code>. Ignored by <code>'d8'</code> and <code>'d-infinity'</code>.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h3 id="gputerrainheightabovedrainage-gputerrainwatersheds-gputerrainstreamorder-and-gputerrainhydrologicindices"><code>GPUTerrainHeightAboveDrainage</code>, <code>GPUTerrainWatersheds</code>, <code>GPUTerrainStreamOrder</code>, and <code>GPUTerrainHydrologicIndices</code></h3>
<p>Products derived from <code>GPUTerrainFlow</code> outputs. Each takes a <code>grid</code> and D8 <code>flowDirections</code>, plus
<code>streams</code> or <code>accumulation</code> where needed, and writes into an <code>output</code> bag. The direction codes are
ESRI by default (1 east, then powers of two clockwise), which is what <code>GPUTerrainFlow</code>, ArcGIS,
pysheds and WhiteboxTools with <code>--esri_pntr</code> write. A grid from another tool needs the compile-time
<code>directionEncoding</code> prop: <code>'whitebox'</code> for the WhiteboxTools default pointer (1 north-east, then
powers of two clockwise) and <code>'taudem'</code> for a TauDEM D8 grid (1 east, then 2 to 8
counter-clockwise). Without it a Whitebox grid is read 45 degrees off with no error, because all its
codes are valid ESRI codes, and most TauDEM codes are read as terminal cells. In every encoding 0
is a terminal cell, <code>0xffffffff</code> a cell without data and <code>0xfffffffe</code> an unresolved direction, so
map the tool's own no-data value to <code>0xffffffff</code>.</p>
<p><strong>Direction codes are raster offsets, not compass bearings.</strong> In every encoding "south" means the
next row (<code>row + 1</code>), "north" the previous row and "east" the next column. No hydrology class reads
the row sign of the grid affine (<code>grid.affine[4]</code>), and <code>GPUTerrainFlow</code> writes its <code>flowDirections</code>
by the same rule, so producer and consumers agree on any grid. On a north-up raster (rows advance
south), which is how ArcGIS, WhiteboxTools and TauDEM write their grids, the names are true compass
directions. On a grid whose rows advance north, code 4 of <code>GPUTerrainFlow</code> points geographically
north. A tool's direction grid held with its rows flipped to advance north still carries the tool's
compass codes and would be followed mirrored north to south with no error: flip the rows back, or
exchange the north and south codes, before passing it in.</p>
<p><code>GPUTerrainFlowDistance</code>,
<code>GPUTerrainStreamNetwork</code>, <code>GPUTerrainCatchmentSummary</code> and <code>GPUTerrainTiledFlow</code> take the same
prop. The inputs of the four products stay inputs; the convergence words are
<code>output.status.converged</code>, <code>output.status.iterationLimitReached</code> (1 when the cap ran out before
convergence) and <code>output.status.iterationCount</code>, and the iteration caps are <code>maximumIterations</code>.</p>
<ul>
<li>
<p><code>GPUTerrainHeightAboveDrainage</code>: HAND (Rennó et al. 2008), the height of each cell above the
first stream cell on its D8 path, with an optional <code>output.drainageCells</code> index output. A path that
first meets an unresolved stream word gives NaN and <code>GPU_TERRAIN_DRAINAGE_UNRESOLVED</code>.
It also accepts D-infinity fractions; the <a href="/luma-contributors/docs/api/gpu-terrain/height-above-drainage">HAND guide</a> defines the
fractional recurrence, branch-contamination rules, drainage codes and mode-specific budgets.</p>
</li>
<li>
<p><code>GPUTerrainWatersheds</code>: labels every cell (<code>output.labels</code>) with its nearest downstream pour point (nested pour
points give nested watersheds). Without pour points it labels drainage basins by their outlet or
pit cell.</p>
</li>
<li>
<p><code>GPUTerrainStreamOrder</code>: Strahler order on the stream mask (<code>output.streamOrder</code>), with an optional
<code>output.status.iterationCount</code>.
Unresolved stream words and the stream cells below them keep the unresolved order <code>0xffffffff</code>.</p>
</li>
<li>
<p><code>GPUTerrainHydrologicIndices</code>: computes, from contributing area in m²
(<code>accumulationUnits: 'area'</code>, any routing), into <code>output.slope</code>, <code>output.specificCatchmentArea</code>,
<code>output.wetnessIndex</code> and <code>output.streamPowerIndex</code>:</p>
<ul>
<li>specific catchment area <code>a = A / b</code>;</li>
<li>topographic wetness <code>ln(a / tan β)</code>;</li>
<li>stream power <code>a · tan β</code>;</li>
</ul>
<p>with <code>tan β</code> the D8 descent slope clamped to a per-frame <code>minimumSlope</code> (default 0.001).</p>
</li>
</ul>
<p>HAND and watersheds resolve D8 paths by GPU pointer jumping, which needs about log₂(path length)
gated rounds. Stream order is Strahler by levels on the Euler tour of the stream forest: one
list ranking, then per level a scatter, a prefix scan (<code>GPUScan</code>) and a gather, at most
<code>floor(log2(n))</code> levels, each gated on the GPU; <code>maximumIterations</code> bounds only the pull rounds
used when the <code>2 n</code> tour words exceed one storage binding (grids above 4096 x 4096 cells), and
<code>status.iterationCount</code> reports the ranking rounds. A receiver cycle (caller-supplied directions
only) leaves every cell of its component at <code>0xffffffff</code> and clears <code>converged</code>. Every iterative contributor stops early on the
GPU and can publish <code>output.status.converged</code>.</p>
<p>Chain the contributors by wrapping a flow output with <code>makeGPURasterBand(view, grid)</code> where a
downstream contributor takes a band (the <code>elevation</code> of HAND and the hydrologic indices); the
direction, stream and accumulation views feed the other inputs directly.</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> grid</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> makeGPURasterMetadata</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({width, height, cellSize: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">30</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">});</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUTerrainFlow</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  grid, elevation, parameters: flowParameters.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">importToGraph</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(graph),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  fillDepressions: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">true</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, resolveFlats: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">true</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, flowRouting: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'mfd-freeman'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, accumulationUnits: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'area'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  output: {filledElevation, flowDirections, accumulation, streams}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> surface</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> makeGPURasterBand</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(filledElevation, grid, {id: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'filled'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">});</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUTerrainHeightAboveDrainage</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  grid, elevation: surface, flowDirections, streams, output: {heightAboveDrainage}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUTerrainWatersheds</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({grid, flowDirections, pourPoints, output: {labels}}));</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUTerrainStreamOrder</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({grid, flowDirections, streams, output: {streamOrder}}));</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUTerrainHydrologicIndices</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  grid, elevation: surface, accumulation, parameters: indexParameters.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">importToGraph</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(graph),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  output: {wetnessIndex}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span></code></pre>
<h4 id="gputerrainheightabovedrainage"><code>GPUTerrainHeightAboveDrainage</code></h4>
<p>Subtracts the elevation of each cell's first downstream stream cell from its own, found by pointer jumping along D8 paths.</p>
<p>::lens{scenario=gpu-terrain.terrain-height-above-drainage rung=result caption="Height above the nearest stream cell; the highest cell is marked."}</p>
<h4 id="how-it-computes-32">How it computes</h4>
<p>{/* lens:GPUTerrainHeightAboveDrainage */}</p>
<p>::lens{scenario=gpu-terrain.terrain-height-above-drainage rung=datum datum=3443 caption="A ridge cell: its D8 receiver and the filled surface it drains over."}
::lens{scenario=gpu-terrain.terrain-height-above-drainage rung=formula caption="HAND is one subtraction of two elevations once the drainage cell is known."}
::lens{scenario=gpu-terrain.terrain-height-above-drainage rung=iterations caption="Pointer-jumping rounds executed, with the convergence word the kernel publishes."}
::lens{scenario=gpu-terrain.terrain-height-above-drainage rung=kernel caption="The height kernel reads the converged pointer and subtracts the drainage elevation."}
::lens{scenario=gpu-terrain.terrain-height-above-drainage rung=graph caption="Flow routing feeds a pointer-jumping loop, then one height kernel."}</p>
<h4 id="gputerrainwatersheds"><code>GPUTerrainWatersheds</code></h4>
<p>Labels every cell with the outlet or pit cell its D8 path reaches, resolved by pointer jumping.</p>
<p>::lens{scenario=gpu-terrain.terrain-watersheds rung=result caption="Drainage basins, one label per outlet or pit cell."}</p>
<h4 id="how-it-computes-33">How it computes</h4>
<p>{/* lens:GPUTerrainWatersheds */}</p>
<p>::lens{scenario=gpu-terrain.terrain-watersheds rung=datum datum=1826 caption="A main-stream cell: the receiver it follows toward its basin outlet."}
::lens{scenario=gpu-terrain.terrain-watersheds rung=iterations caption="Pointer-jumping rounds executed, with the convergence word the kernel publishes."}
::lens{scenario=gpu-terrain.terrain-watersheds rung=kernel caption="The label kernel copies the terminal cell index reached by each pointer."}
::lens{scenario=gpu-terrain.terrain-watersheds rung=graph caption="Flow routing feeds a pointer-jumping loop, then the label kernel."}</p>
<h4 id="gputerrainstreamorder"><code>GPUTerrainStreamOrder</code></h4>
<p>Assigns Strahler order to stream cells by levels on the Euler tour of the stream forest.</p>
<p>::lens{scenario=gpu-terrain.terrain-stream-order rung=result caption="Strahler order along the 40-cell stream network."}</p>
<h4 id="how-it-computes-34">How it computes</h4>
<p>{/* lens:GPUTerrainStreamOrder */}</p>
<p>::lens{scenario=gpu-terrain.terrain-stream-order rung=datum datum=1761 caption="A confluence cell: its stream donors and the receiver it drains to."}
::lens{scenario=gpu-terrain.terrain-stream-order rung=iterations caption="List-ranking rounds of the Euler tour, with the convergence word the kernel publishes."}
::lens{scenario=gpu-terrain.terrain-stream-order rung=kernel caption="Per-level scatter and gather kernels: the same code runs for every Strahler level."}
::lens{scenario=gpu-terrain.terrain-stream-order rung=graph caption="Tour link, gated ranking rounds, then gated order levels."}</p>
<h4 id="gputerrainhydrologicindices"><code>GPUTerrainHydrologicIndices</code></h4>
<p>Combines upstream area and D8 descent slope into specific catchment area, topographic wetness and stream power.</p>
<p>::lens{scenario=gpu-terrain.terrain-hydrologic-indices rung=result caption="Topographic wetness: high in valley bottoms; the wettest cell is marked."}</p>
<p>{/* parameters: GPUTerrainHydrologicIndices generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h5 id="gputerrainhydrologicindices-parameters"><code>GPUTerrainHydrologicIndices</code> parameters</h5>
<ul>
<li>Packer: <code>getGPUTerrainHydrologicIndicesParameterValues(settings, grid)</code>, writes <code>GPU_TERRAIN_HYDROLOGIC_INDICES_PARAMETER_LENGTH</code> (8) <code>float32</code> values.
<ul>
<li><code>grid</code> (<code>GPURasterMetadata</code>): the frame grid, which supplies the cell size and row edges</li>
</ul>
</li>
</ul>
<p>Settings of <code>getGPUTerrainHydrologicIndicesParameterValues</code>:</p>
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
<td><code>minimumSlope</code> (optional)</td>
<td><code>number</code></td>
<td>Lower bound applied to <code>tan(beta)</code> before the indices are formed, so filled flats do not give an infinite wetness index. Negative or non-finite values act as 0 (no clamp: zero slopes give <code>+Infinity</code> wetness and zero stream power). Defaults to <code>GPU_TERRAIN_HYDROLOGIC_INDICES_DEFAULT_MINIMUM_SLOPE</code> (0.001, about 0.06 degrees).</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h4 id="how-it-computes-35">How it computes</h4>
<p>{/* lens:GPUTerrainHydrologicIndices */}</p>
<p>::lens{scenario=gpu-terrain.terrain-hydrologic-indices rung=datum datum=1826 caption="A valley cell: its steepest D8 descent and the upstream area it receives."}
::lens{scenario=gpu-terrain.terrain-hydrologic-indices rung=formula caption="Wetness and stream power from catchment width, area and the clamped D8 slope."}
::lens{scenario=gpu-terrain.terrain-hydrologic-indices rung=kernel caption="One kernel scans eight neighbours for the steepest descent, then forms the indices."}
::lens{scenario=gpu-terrain.terrain-hydrologic-indices rung=graph caption="Flow routing in square metres feeds one index kernel; no loop."}</p>
<h3 id="gputerrainflowfield"><code>GPUTerrainFlowField</code></h3>
<p>Deflects a uniform horizontal wind over an elevation grid onto the terrain tangent plane, giving a wind vector per cell.</p>
<p>::lens{scenario=gpu-terrain.terrain-flow-field rung=result caption="Wind speed from an 8 m/s south-westerly; slopes facing the wind slow it."}</p>
<p>Terrain-following wind for the flow contributors. A uniform horizontal wind is projected onto the
terrain tangent plane, <code>v = w − (w·g) g / (1 + |g|²)</code>, using central-difference gradients in ground
meters (uniform, Web Mercator, or geographic cells, one-sided at the border). Wind across a slope
is unchanged; wind into a slope slows, so flow bends around ridges. The output is the <code>velocities</code>
format of <code>GPUParticleAdvection</code>, <code>GPUStreamlines</code>, and <code>GPULineIntegralConvolution</code>, written in the
raster frame (u along columns, v along rows) to <code>output.velocities</code>. Cells next to nodata are NaN.
Wind and vertical exaggeration are per-frame settings; the cell size and latitude band come from
the <code>grid</code>, which the packer takes as its second argument.</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> grid</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> makeGPURasterMetadata</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({width, height, cellSize: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">30</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">});</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUTerrainFlowField</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  grid, elevation, parameters: windParameters.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">importToGraph</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(graph), output: {velocities}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">windParameters.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">write</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">getGPUTerrainFlowFieldParameterValues</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({wind: [</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">4</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">-</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">]}, grid));</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUParticleAdvection</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  velocities, fieldWidth: width, fieldHeight: height,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  parameters: advectionParameters.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">importToGraph</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(graph),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  uint32Parameters: advectionWords.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">importToGraph</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(graph),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  state: {positions, ages, generations}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span></code></pre>
<h4 id="how-it-computes-36">How it computes</h4>
<p>{/* lens:GPUTerrainFlowField */}</p>
<p>::lens{scenario=gpu-terrain.terrain-flow-field rung=datum datum=3100 caption="A slope facing the wind: the gradient from four neighbours and the removed component."}
::lens{scenario=gpu-terrain.terrain-flow-field rung=formula caption="The wind loses its component along the gradient, divided by one plus the squared gradient."}
::lens{scenario=gpu-terrain.terrain-flow-field rung=kernel caption="One invocation per cell reads four neighbours, forms the gradient and writes (u, v)."}
::lens{scenario=gpu-terrain.terrain-flow-field rung=graph caption="Elevation canonicalization feeds one deflection kernel; no loop."}</p>
<p>{/* parameters: GPUTerrainFlowField generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h4 id="gputerrainflowfield-parameters"><code>GPUTerrainFlowField</code> parameters</h4>
<ul>
<li>Packer: <code>getGPUTerrainFlowFieldParameterValues(settings, grid)</code>, writes <code>GPU_TERRAIN_FLOW_FIELD_PARAMETER_LENGTH</code> (8) <code>float32</code> values.
<ul>
<li><code>grid</code> (<code>GPURasterMetadata</code>): the frame grid, which supplies the cell size and row edges</li>
</ul>
</li>
</ul>
<p>Settings of <code>getGPUTerrainFlowFieldParameterValues</code>:</p>
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
<td><code>wind</code></td>
<td><code>readonly [number, number]</code></td>
<td>Horizontal wind <code>[x, y]</code> in the raster frame: x along increasing columns, y along increasing rows (south for a north-up raster whose row 0 is the northern edge). Any speed unit; the output velocities share it.</td>
</tr>
<tr>
<td><code>verticalExaggeration</code> (optional)</td>
<td><code>number</code></td>
<td>Factor applied to the elevation gradient before deflection. Defaults to 1.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h3 id="parameter-packers">Parameter packers</h3>
<p>Every contributor reads its per-frame settings from a packed <code>float32</code> view, <code>parameters</code>; the
packer writes it and its <code>*_PARAMETER_LENGTH</code> constant sizes it. Each contributor section ends with a
generated "parameters" subsection listing its packer, length constant and settings fields. Packers
with a <code>grid</code> argument take the frame's <code>GPURasterMetadata</code> and write its cell size and row edges, so
those never appear in the settings object.</p>
<p>Helpers that derive packer inputs or kernel tables:</p>
<ul>
<li>point horizon: <code>getGPUPointHorizonDistanceLattice</code>, <code>getGPUPointHorizonSegments</code>;</li>
<li>horizon: <code>getGPUTerrainHorizonDirection</code>, <code>getGPUTerrainHorizonStepDistances</code>;</li>
<li>solar: <code>getGPUSolarIrradianceSunTable</code>, <code>getGPUSolarDirectNormalIrradiance</code>,
<code>getSolarRefractionDegrees</code>, <code>getSolarTimeParameter</code>;</li>
<li>texture shading: <code>getGPUTextureShadingKernel</code>, <code>getGPUTextureShadingCascadeSigmas</code>;</li>
<li>relief: <code>getGPUMultiScaleReliefRadii</code>, <code>getGPULocalDominanceShifts</code>.</li>
</ul>
<h3 id="capped-loops-and-compile-cost">Capped loops and compile cost</h3>
<ul>
<li>A loop stopped by its iteration limit never writes a value that reads as converged (library
rule 9): unresolved cells carry NaN or a <code>*_UNRESOLVED</code> constant, see each contributor's TSDoc.</li>
<li>A loop index never reaches WGSL text: gated rounds, horizon sectors, cumulative-viewshed
batches, point-horizon chunks and jump-flood steps read their position from a GPU word, so
every iteration of a kernel shares one pipeline (default <code>GPUTerrainFlow</code> with fill and flats
compiles 15 relaxation pipelines, not about 60).</li>
<li><code>GPUTerrainCumulativeViewshed</code> sizes each dispatch to about 2^28 sight-line samples
(<code>GPU_TERRAIN_CUMULATIVE_VIEWSHED_SAMPLES_PER_DISPATCH</code>); a large grid splits one observer
across dispatches.</li>
<li><code>GPUPointHorizonVisibility</code> stops the occlusion test one ground pixel before the target by
default; mt-image's 150 m is <code>GPU_POINT_HORIZON_MT_IMAGE_TARGET_IGNORE_DISTANCE</code>.</li>
<li><code>GPUSimpleLocalRelief</code> and <code>GPUTerrainVectorRuggedness</code> compile for <code>maximumRadius</code> and read
the window <code>radius</code> per frame from <code>parameters</code>, so changing the radius does not recompile.</li>
<li>D-infinity and MFD accumulation compute each cell's routing once (<code>routing-prepare</code>) instead of
re-deriving every donor's facets in every round.</li>
</ul>`};export{e as default};