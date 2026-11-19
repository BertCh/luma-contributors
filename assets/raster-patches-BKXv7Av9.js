var e={slug:`api/gpu-raster/raster-patches`,title:`Raster patch metrics and sieving`,html:`<h1 id="raster-patch-metrics-and-sieving">Raster patch metrics and sieving</h1>
<p><code>GPURasterPatchMetrics</code> and <code>GPURasterSieve</code> are companions to upstream <code>GPURasterConnectedComponents</code> and
<code>GPURasterDenseComponents</code> (<code>@luma.gl/graph/gpu-raster</code>). Both read dense labels (<code>0</code> is background) and take the same optional
<code>mask</code> (an input row filter: zero turns the pixel into background), <code>converged</code>, <code>componentCount</code>,
and <code>overflow</code> guards, the last three as in <code>GPURasterRegionMeasurements</code>; the sparse labels of <code>GPURasterConnectedComponents</code> must go through
<code>GPURasterDenseComponents</code> first.</p>
<ul>
<li><code>GPURasterPatchMetrics</code> writes per-patch columns, where row <code>r</code> is label <code>r + 1</code>: <code>pixelCounts</code>,
<code>areas</code>, <code>perimeterFaceCounts</code>, <code>perimeters</code> (length in CRS units through <code>grid.affine</code>), and the
inclusive bounding box <code>minColumns</code>, <code>minRows</code>, <code>maxColumns</code>, <code>maxRows</code>. Faces on the raster
edge count as perimeter unless <code>countRasterBorder</code> is <code>false</code>. <code>cellSizeMode: 'geographic'</code> (degrees)
or <code>'web-mercator'</code> (EPSG:3857 meters) turns <code>areas</code> and <code>perimeters</code> into WGS84 ground square
meters and meters, with the scale taken at each row center (axis-aligned grids only; the
default <code>'uniform'</code> keeps CRS units). The optional <code>parameters</code> view (layout of
<code>getGPURasterPatchMetricsParameterValues</code>, shared with fractional coverage) supplies affine
steps and area in every mode, plus row latitude in ground modes; without it the initial affine
stays fixed. It
also repeats the pixel count and area so shape ratios (FRAGSTATS patch metrics, GRASS
<code>r.object.geometry</code>) need one node set. Counts, extents and perimeter faces are integer atomics, so
they are exact. The ground sums are 64-bit fixed-point atomics, deterministic and independent of
pixel order, but <strong>not exact</strong>: each row's weight is rounded to a multiple of 2^-26 of the largest
scale the model can reach (equator for x, pole for y), a relative error of at most 2^-27 over the
row's weight, about 1e-7 at 60 degrees latitude and about 1e-5 at 89.95 degrees. A perimeter face
along a parallel is counted at its pixel's row center, so it differs from the geodesic length by a
second-order term (below 1e-4 for one-degree cells).</li>
<li><code>GPURasterSieve</code> (GDAL <code>gdal_sieve</code>, GRASS <code>r.reclass.area</code>) removes patches smaller than
<code>minimumPixels</code>. <code>mode: 'remove'</code> (default) turns them into background. <code>mode: 'merge'</code> gives each
small patch the label of its largest neighbor that is not small (lowest label on ties; <code>connectivity</code>
4 or 8) and removes it when it has none, where <code>gdal_sieve</code> leaves it unaltered. Background (label 0)
is never a merge target, as <code>gdal_sieve</code> never merges into masked pixels. <code>minimumPixels</code> is per-frame, written with
<code>getGPURasterSieveParameterValues</code> into <code>parameters</code>. Outputs are <code>labels</code>, optional <code>patchTargets</code>
(final label per patch) and <code>sievedCount</code>. Merging is a single pass. A clump labeling never has
touching labels, so <code>'merge'</code> is meant for segmentation label rasters.</li>
</ul>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPURasterPatchMetrics</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  grid: rasterMetadata, labels: denseLabels, mask,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  status: {converged, componentCount, overflow}, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// the labeling passes' status scalars</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  output: {pixelCounts, areas, perimeters, minColumns, minRows, maxColumns, maxRows}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPURasterSieve</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  grid: rasterMetadata, labels: denseLabels, patchCapacity: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">256</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  parameters: sieveParameters.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">importToGraph</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(graph),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  mode: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'merge'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, connectivity: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">8</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, output: {labels: sievedLabels, sievedCount}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">sieveParameters.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">write</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">getGPURasterSieveParameterValues</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({minimumPixels: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">16</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span></code></pre>
<h4 id="gpurasterpatchmetrics"><code>GPURasterPatchMetrics</code></h4>
<p>Measures pixel count, area and perimeter of each labelled patch of the burn mask.</p>
<p>Pass <code>parameters</code> packed by <code>getGPURasterPatchMetricsParameterValues(grid)</code> to update the affine
between encodings. The eight-word layout matches fractional coverage. Uniform mode reads the
absolute determinant for area and both affine axis lengths for perimeter, including rotated and
sheared cells. Ground modes also read each row's latitude; they require axis-aligned grids on every
encoding. Their fixed-point weights and perimeter normalization follow the new resolution.
Dimensions, pixel interpretation and <code>cellSizeMode</code> stay fixed. Without <code>parameters</code>, the initial
affine remains fixed for compatibility.
The packer and parameter-driven path require a nonsingular affine representable in float32.
Fixed-affine pixel counts do not depend on that packing, so an underflowing determinant does
not prevent counting patches.</p>
<p>::lens{scenario=gpu-raster.raster-patch-metrics rung=result caption="Burn-mask patches; one patch holds most of the 2473 burned cells."}</p>
<p>{/* parameters: GPURasterPatchMetrics generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h5 id="gpurasterpatchmetrics-parameters"><code>GPURasterPatchMetrics</code> parameters</h5>
<ul>
<li>Packer: <code>getGPURasterPatchMetricsParameterValues(grid)</code>, writes <code>GPU_RASTER_PATCH_METRICS_PARAMETER_LENGTH</code> (8) <code>float32</code> values.
<ul>
<li><code>grid</code> (<code>GPURasterMetadata</code>): Raster grid whose affine will be measured.</li>
</ul>
</li>
</ul>
<p>{/* /parameters */}</p>
<h4 id="how-it-computes">How it computes</h4>
<p>{/* lens:GPURasterPatchMetrics */}</p>
<p>::lens{scenario=gpu-raster.raster-patch-metrics rung=datum datum=0 caption="Patch 0, the largest: its pixel count, exposed faces, area and perimeter."}
::lens{scenario=gpu-raster.raster-patch-metrics rung=formula caption="Area is pixels times cell area; perimeter sums exposed faces times their face length."}
::lens{scenario=gpu-raster.raster-patch-metrics rung=kernel caption="Run segments accumulate counts and exposed faces with integer atomics, then columns are finalized."}
::lens{scenario=gpu-raster.raster-patch-metrics rung=graph caption="Connected components and dense labels feed the key, count, face and area nodes."}</p>
<h4 id="gpurastersieve"><code>GPURasterSieve</code></h4>
<p>Removes patches smaller than a pixel threshold from a patch label raster.</p>
<p>::lens{scenario=gpu-raster.raster-sieve rung=result caption="Burn patches of 12 cells or more survive; smaller patches become background."}</p>
<p>{/* parameters: GPURasterSieve generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h5 id="gpurastersieve-parameters"><code>GPURasterSieve</code> parameters</h5>
<ul>
<li>Packer: <code>getGPURasterSieveParameterValues(settings)</code>, writes <code>GPU_RASTER_SIEVE_PARAMETER_LENGTH</code> (4) <code>uint32</code> values.</li>
</ul>
<p>Settings of <code>getGPURasterSieveParameterValues</code>:</p>
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
<td><code>minimumPixels</code></td>
<td><code>number</code></td>
<td>Patches with fewer pixels than this are sieved. Integer from 1 through 2^32 - 1.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h4 id="how-it-computes-1">How it computes</h4>
<p>{/* lens:GPURasterSieve */}</p>
<p>::lens{scenario=gpu-raster.raster-sieve rung=datum datum=32 caption="Patch 32, 11 cells: smaller than the threshold of 12, so its target is background."}
::lens{scenario=gpu-raster.raster-sieve rung=formula caption="A patch keeps its label when its size reaches the minimum; otherwise it becomes background."}
::lens{scenario=gpu-raster.raster-sieve rung=kernel caption="Counts per patch decide each target, then one pass relabels every pixel."}
::lens{scenario=gpu-raster.raster-sieve rung=graph caption="Labelling feeds counts, per-patch targets and the relabel node."}</p>
<h2 id="related-pages">Related pages</h2>
<ul>
<li><a href="/luma-contributors/docs/api/gpu-raster/operations-analysis">GPURaster analysis contributors</a></li>
<li><a href="https://luma.gl/docs/api-reference/experimental/gpu-raster">luma.gl GPU Raster</a></li>
</ul>`};export{e as default};