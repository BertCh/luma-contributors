var e={slug:`api/gpu-raster`,title:`Raster analysis`,html:`<h1 id="raster-analysis">Raster analysis</h1>
<h2 id="overview">Overview</h2>
<p><code>@luma-contributors/analysis/raster-analysis</code> provides nodata-aware raster analysis contributors that
compose through <code>GPUCommandGraph</code>: zonal statistics, stretches, isolines, distance and cost
surfaces, polygon rasterization, change detection, flow textures, fill nodata, tile reprojection,
fractional coverage and spectral indices. Storage, overviews, pixel operations, statistics, filters,
morphology, connected components and contours are owned by
<a href="https://luma.gl/docs/api-reference/experimental/gpu-raster">luma.gl GPU Raster</a>.</p>
<h2 id="when-to-use-it">When to use it</h2>
<p>Use these contributors for analytical raster pipelines whose intermediate bands, masks, labels or
statistics should remain on the GPU. Image-only presentation effects that do not need analytical
values usually belong in a renderer or shader module instead.</p>
<h2 id="quick-start">Quick start</h2>
<p>Use the <a href="/luma-contributors/docs/api/operation-forms-and-cpu-twins#point-density-from-device-to-readback">complete GPU walkthrough</a>
for allocation, graph imports, submission, typed readback and cleanup. For an exact distance field:</p>
<p>Import <code>Float32Array.of(1.5, 1.5)</code> as a one-row <code>float32x2</code> seed view and allocate
the output views below. This checked function adds an exact distance field for that seed at the
center of a north-up 3 × 3 grid with 1 m cells.</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">import</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> type</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8"> {Device} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">from</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF"> '@luma.gl/core'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">;</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">import</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8"> {GPUCommandGraph, </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">type</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8"> GraphDataView} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">from</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF"> '@luma.gl/graph'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">;</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">import</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8"> {GPUParameterBuffer} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">from</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF"> '@luma-contributors/analysis/analysis-core'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">;</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">import</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8"> {</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  GPUDistanceField, makeGPURasterMetadata, getGPUDistanceFieldParameterValues,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  GPU_DISTANCE_FIELD_PARAMETER_LENGTH</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">from</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF"> '@luma-contributors/analysis/raster-analysis'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">;</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">export</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> addDistanceFieldExample</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span></span>
<span class="line"><span style="--shiki-light:#E36209;--shiki-dark:#FFAB70">  device</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">:</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> Device</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#E36209;--shiki-dark:#FFAB70">graph</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">:</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUCommandGraph</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#E36209;--shiki-dark:#FFAB70">seedPositions</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">:</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GraphDataView</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">&#x3C;</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'float32x2'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">>,</span></span>
<span class="line"><span style="--shiki-light:#E36209;--shiki-dark:#FFAB70">  distances</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">:</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GraphDataView</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">&#x3C;</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'float32'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">>, </span><span style="--shiki-light:#E36209;--shiki-dark:#FFAB70">allocation</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">:</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GraphDataView</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">&#x3C;</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'uint32'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">>,</span></span>
<span class="line"><span style="--shiki-light:#E36209;--shiki-dark:#FFAB70">  withinDistance</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">:</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GraphDataView</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">&#x3C;</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'uint32'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">></span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">) {</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">  const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> grid</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> makeGPURasterMetadata</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({width: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">3</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, height: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">3</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, cellSize: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, origin: [</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">3</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">]});</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">  const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> parameters</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUParameterBuffer</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(device, {</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    id: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'distance-field-example-parameters'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    format: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'float32'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    length: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">GPU_DISTANCE_FIELD_PARAMETER_LENGTH</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    values: </span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">getGPUDistanceFieldParameterValues</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({maximumDistance: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">5</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}, grid)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  });</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUDistanceField</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    grid, seedPositions, parameters: parameters.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">importToGraph</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(graph), mode: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'exact'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    output: {distances, allocation, withinDistance}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  }));</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">  return</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8"> parameters;</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}</span></span></code></pre>
<p>Omitting <code>seedCount</code>, <code>seedIds</code> and <code>seedMask</code> makes the single position active with ID 0.
Keep the returned parameter buffer until the compiled graph has been destroyed.</p>
<table>
<thead>
<tr>
<th>Output port</th>
<th>Format and rows</th>
<th>Expected readback</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>distances</code></td>
<td><code>float32</code>, width × height (9 here)</td>
<td><code>[sqrt(2), 1, sqrt(2), 1, 0, 1, sqrt(2), 1, sqrt(2)]</code> within float32 tolerance</td>
</tr>
<tr>
<td><code>allocation</code></td>
<td><code>uint32</code>, width × height</td>
<td>Nine zero IDs</td>
</tr>
<tr>
<td><code>withinDistance</code></td>
<td><code>uint32</code>, width × height</td>
<td>Nine ones</td>
</tr>
</tbody>
</table>
<p>All output views have separate buffers. Compile, encode, submit and read them with the
walkthrough's sequence, then destroy the compiled graph, parameter buffer and caller-owned buffers.
No-seed or beyond-limit distances are <code>Infinity</code>; <code>withinDistance</code> then becomes 0.
The default exact mode is exact for this isotropic fixture; jump-flood mode is an approximate
preview. See the <a href="/luma-contributors/docs/api/gpu-raster/operations-analysis#gpudistancefield">distance-field reference</a> for limits.</p>
<p>The <a href="/luma-contributors/docs/api/gpu-raster/operations">operations reference</a> indexes the pages.</p>
<h2 id="catalog">Catalog</h2>
<p>One figure per contributor, each drawn from a recorded GPU run on real data; click through to its card.</p>
<p>::lens-catalog{entry=gpu-raster}</p>
<h2 id="limits-and-compatibility">Limits and compatibility</h2>
<ul>
<li>Raster analysis is experimental and WebGPU-only.</li>
<li>Tile sources, decoded arrays, GPU residency and submission remain application-owned.</li>
<li>Capacity-dependent labels, geometry and statistics report overflow or incomplete work.</li>
<li>Same-CRS reprojection composes the destination-to-source sample transform in float64 before
packing and anchors evaluation near source support to preserve scale/translation cancellation.
Manual buffers must use <a href="/luma-contributors/docs/api/gpu-raster/expansion#gpurasterreprojection">reprojection parameter layout version 3</a>.</li>
<li>Tiled reprojection accepts at most 16,382 source tiles under its 32,768-command-node budget;
<a href="/luma-contributors/docs/api/gpu-raster/tile-reprojection#resources-and-reuse">the tile reference</a> explains the construction estimate.</li>
</ul>
<h2 id="related-modules">Related modules</h2>
<ul>
<li><a href="https://luma.gl/docs/api-reference/experimental/gpu-core">luma.gl GPU Core</a> provides scheduling and generic aggregation and indexing.</li>
<li><a href="https://luma.gl/docs/api-reference/experimental/gpu-raster">luma.gl GPU Raster</a> owns raster storage, overviews, statistics, filters and contours.</li>
<li><a href="https://luma.gl/docs/api-reference/experimental/gpu-project">luma.gl GPU Project</a> handles coordinate projection.</li>
</ul>
<p>See <a href="/luma-contributors/docs/api/gpu-raster/expansion">Raster, terrain and geometry expansion</a> for fractional zonal statistics,
continuous-surface preparation, reprojection, directed walking cost and bounded topology operations.</p>
<p><a href="/luma-contributors/docs/api/gpu-raster/expansion-lenses">Expansion computation lenses</a> provide executable bounded synthetic
examples for the added contributors, including validity, convergence and capacity evidence.</p>
<ul>
<li><a href="/luma-contributors/docs/api/gpu-raster/fill-nodata">Four-quadrant nodata filling</a></li>
<li><a href="/luma-contributors/docs/api/gpu-raster/tile-reprojection">Explicit tile reprojection</a></li>
<li><a href="/luma-contributors/docs/api/gpu-raster/raster-patches">Patch metrics and sieving</a></li>
</ul>
<p>Fractional coverage also feeds <a href="/luma-contributors/docs/api/gpu-raster/fractional-zonal-distribution">weighted distributions</a> and
<a href="/luma-contributors/docs/api/gpu-raster/fractional-categorical-statistics">categorical composition</a>, keeping observed support and
missing area separate. <a href="/luma-contributors/docs/api/gpu-raster/focal-pairs">Focal band pairs</a> provide paired-valid covariance,
correlation and local regression; <a href="/luma-contributors/docs/api/gpu-raster/spectral-indices">calibrated spectral indices</a> provide
explicit NDWI conventions, NBR, EVI and SAVI.</p>`};export{e as default};