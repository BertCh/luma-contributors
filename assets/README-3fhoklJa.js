var e={slug:`api/gpu-terrain`,title:`GPU Terrain`,html:`<h1 id="gpu-terrain">GPU Terrain</h1>
<p>:::caution Experimental
<code>@luma-contributors/analysis/gpu-terrain</code> is an initial, experimental entry point. APIs may change between
releases without a deprecation period.
:::</p>
<h2 id="overview">Overview</h2>
<p><code>@luma-contributors/analysis/gpu-terrain</code> provides analysis contributors for elevation tiles: Horn slope,
aspect and hillshade, marching-squares contours, viewshed visibility, hydrology (depression
filling, flat resolution, D8, D-infinity and multiple-flow-direction routing, flow accumulation,
streams, height above drainage, watersheds, stream order, hydrologic indices), batched line of sight
and cumulative viewsheds, point horizon profiles, horizon angles, solar position, shadow masks and
irradiance, relief shading, texture shading, archaeological relief visualizations (RVT), terrain-RGB
decoding, summits and critical points, curvature, geomorphons, ruggedness, topographic position,
Weiss landforms, and terrain-following wind fields. Each contributor is an algorithm that declares its
resources and command nodes into a caller-owned
<a href="https://luma.gl/docs/api-reference/experimental/gpu-core/gpu-command-graph"><code>GPUCommandGraph</code></a> through <code>getCommandNodes(graph)</code>
(<code>GPUCommandNodeProducer</code>). It never compiles, submits, encodes, or reads back.</p>
<p>The contributors compose public primitives from <a href="/luma-contributors/docs/api/gpu-raster">GPU Raster</a> and
<a href="https://luma.gl/docs/api-reference/experimental/gpu-core/concepts">GPU Core</a>; new WGSL is written only where no primitive fits.</p>
<h2 id="when-to-use-it">When to use it</h2>
<p>Use GPU Terrain when the elevation values themselves are the product: slope and aspect for
suitability models, contour or viewshed geometry for analysis, flow accumulation for hydrology, or
analytic shading textures that stay on the GPU. Per-frame observers, light directions, contour
levels, and cell-size settings are read from storage views, so interaction never recompiles.</p>
<p>Prefer the generic GPU Raster operations when the field is not an elevation model or you need a
different kernel:</p>
<ul>
<li><code>GPUTerrainDerivatives</code> is Horn's method (two Sobel <code>GPURasterGradient</code> passes plus a fused shade
kernel) with terrain cell-size modes, a z factor, and latitude handling. Use <code>GPURasterGradient</code>
directly for a single signed derivative of any raster.</li>
<li><code>GPUTerrainContours</code> runs one <code>GPURasterContours</code> pipeline per elevation level and reports one
overflow flag. Use <code>GPUIsolines</code> and <code>GPUIsobands</code> from GPU Raster when you need stitched
polylines or filled bands, and <code>GPURasterContours</code> for a single level.</li>
<li>The solar and shading contributors write analytic values and textures. If you only need lit
terrain for presentation, the lighting of a terrain renderer is simpler.</li>
</ul>
<h2 id="quick-start">Quick start</h2>
<p>Start with the <a href="/luma-contributors/docs/api/operation-forms-and-cpu-twins#point-density-from-device-to-readback">complete GPU walkthrough</a>
for the device, buffer import, submission, typed readback and cleanup sequence. Terrain derivatives
add a raster grid, a raster-band input and a packed parameter view:</p>
<p>The elevation input is a nine-row <code>float32</code> view containing zeros. The following checked function
wraps it as a raster band and adds derivatives to the graph from the shared walkthrough.
Allocate <code>slope</code> and <code>hillshade</code> as separate nine-row <code>float32</code> views.</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">import</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> type</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8"> {Device} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">from</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF"> '@luma.gl/core'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">;</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">import</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8"> {GPUCommandGraph, </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">type</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8"> GraphDataView} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">from</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF"> '@luma.gl/graph'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">;</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">import</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8"> {GPUParameterBuffer} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">from</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF"> '@luma-contributors/analysis/analysis-core'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">;</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">import</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8"> {makeGPURasterBand, makeGPURasterMetadata} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">from</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF"> '@luma-contributors/analysis/raster-analysis'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">;</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">import</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8"> {</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  GPUTerrainDerivatives, getGPUTerrainDerivativesParameterValues,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  GPU_TERRAIN_DERIVATIVES_PARAMETER_LENGTH</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">from</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF"> '@luma-contributors/analysis/gpu-terrain'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">;</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">export</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> addTerrainExample</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span></span>
<span class="line"><span style="--shiki-light:#E36209;--shiki-dark:#FFAB70">  device</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">:</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> Device</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#E36209;--shiki-dark:#FFAB70">graph</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">:</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUCommandGraph</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#E36209;--shiki-dark:#FFAB70">elevationView</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">:</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GraphDataView</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">&#x3C;</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'float32'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">>,</span></span>
<span class="line"><span style="--shiki-light:#E36209;--shiki-dark:#FFAB70">  slope</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">:</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GraphDataView</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">&#x3C;</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'float32'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">>, </span><span style="--shiki-light:#E36209;--shiki-dark:#FFAB70">hillshade</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">:</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GraphDataView</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">&#x3C;</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'float32'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">></span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">) {</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">  const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> grid</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> makeGPURasterMetadata</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({width: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">3</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, height: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">3</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, cellSize: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">30</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, origin: [</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">90</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">]});</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">  const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> elevation</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> makeGPURasterBand</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(elevationView, grid, {id: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'elevation'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">});</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">  const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> parameters</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUParameterBuffer</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(device, {</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    id: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'terrain-example-parameters'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    format: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'float32'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    length: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">GPU_TERRAIN_DERIVATIVES_PARAMETER_LENGTH</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    values: </span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">getGPUTerrainDerivativesParameterValues</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({azimuthDegrees: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">315</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}, grid)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  });</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUTerrainDerivatives</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    grid, elevation, parameters: parameters.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">importToGraph</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(graph), output: {slope, hillshade}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  }));</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">  return</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8"> parameters;</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}</span></span></code></pre>
<p>This north-up planar grid has 30 m cells. With no validity view, all nine finite elevations are
usable. Keep the returned parameter buffer until the compiled graph has been destroyed.</p>
<table>
<thead>
<tr>
<th>Output port</th>
<th>Format and rows</th>
<th>Meaning</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>slope</code></td>
<td><code>float32</code>, width × height (9 here)</td>
<td>Slope in degrees (the default); the center of this flat fixture is 0.</td>
</tr>
<tr>
<td><code>hillshade</code></td>
<td><code>float32</code>, width × height (9 here)</td>
<td>Analytic illumination in [0, 1].</td>
</tr>
</tbody>
</table>
<p>Use separate output buffers, follow the walkthrough through readback, and destroy the compiled
graph before the parameter buffer and imported buffers. The packer takes cell size and latitude
settings from <code>grid</code>; rewrite its values before another submission to change lighting.
See <a href="/luma-contributors/docs/api/gpu-terrain/operators#gputerrainderivatives">derivative boundaries and units</a> before interpreting
edge cells or using geographic elevation grids.</p>
<h2 id="concepts-and-operators">Concepts and operators</h2>
<ul>
<li><a href="/luma-contributors/docs/api/gpu-terrain/concepts">GPU Terrain concepts</a>: conventions, units and the GPU execution model.</li>
<li><a href="/luma-contributors/docs/api/gpu-terrain/operators">GPU Terrain operators</a>: every contributor, its inputs, outputs, parameters and how it
computes.</li>
</ul>
<h2 id="catalog">Catalog</h2>
<p>One figure per contributor, each drawn from a recorded GPU run on real data; click through to its card.</p>
<p>::lens-catalog{entry=gpu-terrain}</p>
<h2 id="limits-and-compatibility">Limits and compatibility</h2>
<ul>
<li>GPU Terrain is experimental and WebGPU-only.</li>
<li>Inputs are planar <code>float32</code> elevation values; projection is done upstream (for example with
<a href="https://luma.gl/docs/api-reference/experimental/gpu-project"><code>GPUProjection</code></a>).</li>
<li>Each contributor's TSDoc lists its non-goals. The GPU Core maintainer roadmap
(<code>dev-docs/roadmap.md</code>) tracks what is still open.</li>
</ul>
<h2 id="related-modules">Related modules</h2>
<ul>
<li><a href="/luma-contributors/docs/api/gpu-raster">GPU Raster</a> supplies the raster operators that terrain contributors compose.</li>
<li><a href="https://luma.gl/docs/api-reference/experimental/gpu-core/concepts">GPU Core</a> defines contributors, composition levels, and graph ownership.</li>
<li><a href="/luma-contributors/docs/api/gpu-network">GPU Network</a> shares the conventions above for road and flow analysis.</li>
</ul>
<p><a href="/luma-contributors/docs/api/gpu-terrain/expansion-lenses">Expansion computation lenses</a> record downstream flow distance and compact
stream reaches on a bounded synthetic drainage grid, including unresolved and overflow evidence.</p>`};export{e as default};