var e={slug:`api/gpu-raster/contrast-equalization`,title:`Raster contrast equalization`,html:`<h1 id="raster-contrast-equalization">Raster contrast equalization</h1>
<h2 id="gpurastercontrastequalization"><code>GPURasterContrastEqualization</code></h2>
<p>Cell and histogram populations are uint32, subject to device bindings/backing allocations and
combined matching/reflected population. Quantized distributions and explicit input/reference domains
remain; CLAHE omits the global output rescale.</p>
<p>{/* lens:GPURasterContrastEqualization */}</p>
<p>::lens{scenario=gpu-raster.raster-contrast-equalization rung=graph}</p>
<p>The synthetic lens records reflected two-cell CLAHE tiles over a gradient with one missing source cell. Its graph shows histogram clipping, CDF scans and interpolation.</p>
<p><code>GPURasterContrastEqualization</code> computes contrast limited adaptive histogram equalization (CLAHE)
or matches a raster's histogram to a caller-supplied reference. Import it from
<code>@luma-contributors/analysis/raster-analysis</code>. It complements <code>GPURasterStretch</code> with local
contrast control and reference distribution matching. It ships no imagery, datasets or weights.</p>
<p>Pass a published <code>GPURasterBand</code>, its <code>grid</code>, a <code>parameters</code> view and caller-owned <code>output</code>
views. Buffer and texture bands use the existing raster canonicalization helpers: raw nodata
and validity are applied before scale/offset; nonfinite calibrated samples are omitted. The
reference in <code>'match-histograms'</code> mode supplies its own band and grid and may have a different
shape. Neither method reads world coordinates or computes ground distances, slopes or areas.</p>
<table>
<thead>
<tr>
<th>Output</th>
<th>Layout</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>values</code></td>
<td>float32, at least <code>grid.width * grid.height</code> rows; CLAHE in [0, 1], matching in the reference domain</td>
</tr>
<tr>
<td><code>validity</code></td>
<td>uint32 per source cell; nonzero only when the source and mapping are valid</td>
</tr>
<tr>
<td><code>status.approximation</code></td>
<td>one uint32, always 1 because distributions are quantized</td>
</tr>
</tbody>
</table>
<p>Invalid source cells remain invalid and have NaN output values. An empty reference distribution
invalidates every matching output. In CLAHE, empty tiles contribute no interpolation weight;
weights of nonempty neighbors are renormalized. A valid center with no weighted nonempty tile
is invalid. This mask behavior extends the unmasked scikit-image algorithm.</p>
<h3 id="clahe">CLAHE</h3>
<p>Use <code>mode: 'clahe'</code> (the default), <code>tileWidth</code> and <code>tileHeight</code> (default 32 each), and <code>binCount</code>
(default 256). Tile dimensions and bins are compile-time integers; clip limit and domains are
per-frame parameters written with <code>getGPURasterContrastEqualizationParameterValues</code> into a
<code>GPUParameterBuffer</code>. After float32 rounding, each domain endpoint must be zero or a finite
normal float32 value, and the increasing domain width must be finite and at least 2^-126.
Subnormal endpoints and widths are rejected: <a href="https://www.w3.org/TR/WGSL/#floating-point-evaluation">WGSL permits denormal flushing</a>,
which could otherwise erase the normalization divisor. Zero endpoints and exactly 2^-126 widths
are supported. Band calibration, view topology and grids are compile-time; band and
validity contents may change between executions.</p>
<p>For a finite sample <code>v</code>, first clamp to <code>inputDomain = [a, b]</code> and normalize to
<code>t = (clamp(v, a, b) - a) / (b - a)</code>. Quantize to a 14-bit integer
<code>g = round(16383 * t)</code>, then bin with <code>floor(g / (1 + floor(16384 / binCount)))</code>.
Half-integer rounding follows WGSL <code>round</code>; exact half-integers can choose either nearest
integer on a GPU. This is scikit-image 0.24's unusual bin width, including its occasionally
unoccupied final bins, rather than a uniform <code>floor(t * binCount)</code> histogram.</p>
<p>Each tile has a histogram of its valid samples. The per-bin clip threshold is
<code>max(1, floor(clipLimit * validCount))</code>; <code>clipLimit: 0</code> disables clipping. The packer raises
positive fractions below 2^-126 to the smallest normal float32, preventing conversion underflow
or GPU denormal flushing from selecting the zero sentinel. This preserves threshold one for
every supported tile (at most 65,536 samples). Clipped excess is
redistributed using scikit-image's quotient, threshold and strided residual procedure, with
integer histogram counts. Each productive residual sweep removes at least one excess count,
so the fixed 65,536-sweep budget covers the maximum tile area; skewed three-bin histograms
can require more sweeps than the bin count. If <code>threshold * binCount &#x3C; validCount</code>, the capped histogram cannot
retain every count; saturated excess is discarded as in scikit-image. The denominator stays
<code>validCount</code>. An inclusive published <code>GPUScan</code> computes CDFs; subtracting the previous tile's
final prefix yields the tile CDF. The tile map is
<code>min(16383, floor(16383 * CDF[bin] / validCount)) / 16383</code>.</p>
<p>Tiles cover the image from its upper-left corner. Incomplete trailing tiles reflect the image
without repeating the edge sample, including its validity; singleton dimensions repeat their
only sample. For interpolation, a cell's tile coordinates are
<code>((column + floor(tileWidth / 2)) / tileWidth - 1, (row + floor(tileHeight / 2)) / tileHeight - 1)</code>.
Clamp the four neighboring tile indices at the tile-grid boundary, then bilinearly interpolate
their mappings at the cell's bin. Odd tile dimensions follow scikit-image's integer half-tile
convention. The four weights sum to one before empty-tile handling.</p>
<p>The formula, clipping sequence and boundary conventions were checked against
<a href="https://github.com/scikit-image/scikit-image/blob/v0.24.0/skimage/exposure/_adapthist.py">scikit-image 0.24 <code>_adapthist.py</code></a>,
which implements Zuiderveld's CLAHE (Graphics Gems IV, 1994). The clipping helper retains its
BSD-3-Clause attribution alongside this project's MIT attribution.</p>
<h3 id="histogram-matching">Histogram matching</h3>
<p>Use <code>mode: 'match-histograms'</code> and <code>reference: {input, grid}</code>. Quantize each normalized domain
with <code>bin = floor(t * (binCount - 1) + 0.5)</code>, so half-bin ties round upward. A bin represents
<code>domainMinimum + bin * (domainMaximum - domainMinimum) / (binCount - 1)</code>.
Build source and reference histograms using only their valid cells. At a source bin's inclusive
quantile <code>q = sourceCDF[bin] / sourceCount</code>, find adjacent <strong>occupied</strong> reference bins with
CDF quantiles <code>q0, q1</code> and representatives <code>v0, v1</code>. The mapping is
<code>v0 + (v1 - v0) * (q - q0) / (q1 - q0)</code>, clamped to the first/last occupied representative
outside the reference quantile interval. Empty reference bins are skipped, so gaps and tied
values follow <code>numpy.interp</code> at the actual CDF knots.</p>
<p>Knot ordering uses exact integer cross-products: compare <code>sourceCDF * referenceCount</code> with
<code>referenceCDF * sourceCount</code>, rather than comparing rounded float32 quantiles. Once the bracketing
reference counts <code>lowerCount, upperCount</code> are known, compute the interpolation fraction as
<code>(sourceCDF * referenceCount - lowerCount * sourceCount) / ((upperCount - lowerCount) * sourceCount)</code>. The shared uint64 WGSL helpers keep products (at most
2^48) and their difference exact; only this final ratio and value interpolation use float32.
This preserves one-count CDF gaps even when both normalized knots would round to the same float.
For source counts <code>[16777215, 1]</code> and reference counts <code>[12582911, 1]</code>, the first source bin maps
to 0.25; public scikit-image gives 0.25000000034924597 using float64 quantiles.</p>
<p>This is the quantile rule in
<a href="https://github.com/scikit-image/scikit-image/blob/v0.24.0/skimage/exposure/histogram_matching.py">scikit-image 0.24 <code>histogram_matching.py</code></a>.
Matching is exact for the quantized distributions, subject to float32 arithmetic. It does not
claim exact matching of all distinct float inputs. Values sharing a bin share a source quantile;
this can change the output rank mapping substantially for densely clustered inputs. Increase
bins or choose tighter domains when that distinction matters.</p>
<h3 id="limits-and-execution">Limits and execution</h3>
<table>
<thead>
<tr>
<th>Limit</th>
<th>Reason</th>
</tr>
</thead>
<tbody>
<tr>
<td>Source/reference cells, each at most 16,777,216</td>
<td>64 MiB packed scalar bindings; exact float32 counts</td>
</tr>
<tr>
<td>Reflected CLAHE tile cells, at most 16,777,216</td>
<td>Bounds virtual padding work and histogram totals</td>
</tr>
<tr>
<td>Total histogram bins across tiles, at most 16,777,216</td>
<td>Each histogram, scan and map binding is at most 64 MiB</td>
</tr>
<tr>
<td><code>binCount</code> from 2 through 256</td>
<td>Bounds per-tile redistribution and inverse-CDF lookup</td>
</tr>
<tr>
<td>Tile width/height from 1 through 256</td>
<td>At most 65,536 virtual samples per tile; safe integer map products</td>
</tr>
</tbody>
</table>
<p>Limits and shapes are validated at construction. Input and reference buffer values and validity
must hold exactly one row per grid cell; declared formats, native nodata, texture channels and
extents are checked through the published, non-owning <code>GPURaster</code> API. Output views may have
extra rows, which are left untouched. Device storage-binding limits include the aligned binding
prefix for nonzero view byte offsets and are checked at
the start of <code>getCommandNodes</code>, before scratch or node declarations. Histogram clearing,
integer atomic binning, clipping, scan, mapping and apply are separate GPU passes. No invocation
walks over a data-sized range of pixels; clipping and inverse lookup walk only bounded histogram
bins. <code>estimatedCommandNodeCount</code> reserves 16 nodes for the published scan and eight per band
canonicalization, plus the contributor's fixed passes. Kernel audits check the actual count at
maximum cells, histogram length and bin count. Counts and CDFs are deterministic integers;
float32 interpolation can vary by a few ulps across devices.</p>
<p>Tile larger images explicitly. CLAHE depends on the contextual tile lattice and neighboring
mappings: independently processing arbitrary image chunks can produce seams. Preserve tile
alignment and provide the neighboring contextual tiles when composing chunked processing.</p>
<h3 id="oracle-mapping-and-differences">Oracle mapping and differences</h3>
<p>The checked-in generator is
<code>modules/analysis/test/gpu-raster/raster-contrast-equalization/generate-reference.py</code>, run with
<code>/usr/bin/python3</code>. It records NumPy, SciPy and scikit-image versions in a small JSON fixture.</p>
<ul>
<li>CLAHE: map <code>(tileHeight, tileWidth)</code> to <code>equalize_adapthist(kernel_size=...)</code>,
<code>clipLimit</code> to <code>clip_limit</code>, and <code>binCount</code> to <code>nbins</code>. For ordinary fixtures, inputs span [0, 1]
and remain away from bin boundaries. scikit-image first converts through uint16 and globally
rescales the input to 14 bits; here the caller's explicit domain replaces that conversion and
rescale. Its <code>_clahe</code> supplies the raw output oracle; its public <code>equalize_adapthist</code> supplies
the independently rescaled comparison. The GPU retains fractional interpolation instead of
truncating it to a 14-bit integer, so the raw tolerance is 1.1/16383. It also omits the public
API's final global output min/max rescale; rescale GPU results explicitly for that display
convention. Constant inputs use the fixed domain rather than a data-derived range.</li>
<li>Masked CLAHE: scikit-image has no mask argument. Fixtures use its <code>clip_histogram</code> and
<code>map_histogram</code> and SciPy's linear interpolation, with the documented empty-tile extension.</li>
<li>Matching: <code>match_histograms</code> runs on independently quantized valid source/reference samples,
with <code>channel_axis=None</code>. Fixtures include ties, missing histogram bins, unequal source/reference
sizes, empty distributions and a constant reference. The large-count fixture stores only the
two histograms and output values from public <code>match_histograms</code> on the full synthetic rasters.
Its headless regression builds 4096×4096 and 4096×3072 raster bands, then samples four outputs
to avoid a full raster readback. Matching tolerance is 3e-6 on [0, 1].</li>
</ul>
<p>Headless specs also cover parameter updates without recompilation, histogram clearing between
runs, calibrated bands, no-data/nonfinite inputs, and unwritten output padding rows. Raster work
has no CPU twin under the contributor contract; fixture generation is an offline oracle only.</p>
<h3 id="example">Example</h3>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">import</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8"> {GPUParameterBuffer} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">from</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF"> '@luma-contributors/analysis/analysis-core'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">;</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">import</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8"> {</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  GPURasterContrastEqualization,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  getGPURasterContrastEqualizationParameterValues,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  GPU_RASTER_CONTRAST_EQUALIZATION_PARAMETER_LENGTH</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">from</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF"> '@luma-contributors/analysis/raster-analysis'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">;</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> contrastParameters</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUParameterBuffer</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(device, {</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  id: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'contrast-parameters'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, format: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'float32'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  length: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">GPU_RASTER_CONTRAST_EQUALIZATION_PARAMETER_LENGTH</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  values: </span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">getGPURasterContrastEqualizationParameterValues</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    inputDomain: [</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">], clipLimit: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">0.02</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  })</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">});</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPURasterContrastEqualization</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  grid, input: rasterBand, tileWidth: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">32</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, tileHeight: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">32</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, binCount: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">256</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  parameters: contrastParameters.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">importToGraph</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(graph),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  output: {values: equalizedValues, validity: equalizedValidity, status: {approximation}}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// Between encodings of the same compiled graph:</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">contrastParameters.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">write</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">getGPURasterContrastEqualizationParameterValues</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({clipLimit: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">0.05</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// Destroy the compiled graph before contrastParameters.destroy().</span></span></code></pre>
<p>{/* parameters: GPURasterContrastEqualization generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h3 id="gpurastercontrastequalization-parameters"><code>GPURasterContrastEqualization</code> parameters</h3>
<ul>
<li>Packer: <code>getGPURasterContrastEqualizationParameterValues(settings?)</code>, writes <code>GPU_RASTER_CONTRAST_EQUALIZATION_PARAMETER_LENGTH</code> (8) <code>float32</code> values.</li>
</ul>
<p>Settings of <code>getGPURasterContrastEqualizationParameterValues</code>:</p>
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
<td><code>inputDomain</code> (optional)</td>
<td><code>readonly [number, number]</code></td>
<td>Fixed input domain, clamped before binning. Defaults to [0, 1]. Endpoints must round to zero or normal finite float32 values, with an increasing finite normal width (at least 2^-126).</td>
</tr>
<tr>
<td><code>referenceDomain</code> (optional)</td>
<td><code>readonly [number, number]</code></td>
<td>Reference domain for histogram matching. Defaults to [0, 1]. Endpoints must round to zero or normal finite float32 values, with an increasing finite normal width (at least 2^-126).</td>
</tr>
<tr>
<td><code>clipLimit</code> (optional)</td>
<td><code>number</code></td>
<td>Fraction of valid tile samples per bin, in [0, 1]. Default 0.01; zero disables clipping. Tiny positive values pack to at least 2^-126, preserving threshold one.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>`};export{e as default};