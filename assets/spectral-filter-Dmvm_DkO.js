var e={slug:`api/gpu-raster/spectral-filter`,title:`Spectral raster filtering`,html:`<h1 id="spectral-raster-filtering">Spectral raster filtering</h1>
<h2 id="gpurasterspectralfilter"><code>GPURasterSpectralFilter</code></h2>
<p>Exported by <code>@luma-contributors/analysis/raster-analysis</code>. Filters a complete resident raster by multiplying its Fourier spectrum by a real response and applying the inverse transform. Both transforms are the published <code>GPUFFT2D</code> from <code>@luma.gl/graph/gpu-core</code>; no FFT implementation is carried here. NumPy <code>fft2</code> / <code>ifft2</code> is the independent numerical reference (<code>test/oracles/raster-spectral-filter.py</code> under <code>modules/analysis</code>).</p>
<p><code>grid</code> is a published <code>GPURasterMetadata</code>, and <code>input</code> is a published <code>GPURasterBand</code> (buffer or texture). Each dimension must be a power of two from 2 through <code>MAXIMUM_RASTER_SPECTRAL_FILTER_DIMENSION</code> (2048). Complex fields must also fit the device storage binding limit; the contributor checks that before creating nodes. <code>estimatedCommandNodeCount</code> bounds the two published transforms, band canonicalization and the five filtering/counting nodes. There is no CPU twin: this is raster work with parallel transforms.</p>
<p>The boundary is periodic. There is no padding, apodization, inferred nodata filling, or independent tile filtering. Opposite image edges touch; discontinuities can ring, especially for ideal pass bands. Supply a complete domain with the boundary treatment your analysis requires. Frequencies are in <strong>cycles per pixel</strong> along the column/row axes; the grid affine, CRS and ground cell sizes do not affect this pixel-frequency operation.</p>
<p><code>filter</code> selects the compile-time response. With signed FFT frequencies <code>f = (columnFrequency, rowFrequency)</code> and <code>r = length(f)</code>:</p>
<table>
<thead>
<tr>
<th>Filter</th>
<th>Response</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>low-pass</code></td>
<td>1 for <code>r &#x3C;= cutoff</code>, otherwise 0</td>
</tr>
<tr>
<td><code>high-pass</code></td>
<td>1 for <code>r > cutoff</code>, otherwise 0 (complement of low-pass)</td>
</tr>
<tr>
<td><code>band-pass</code></td>
<td>1 for <code>lowerCutoff &#x3C;= r &#x3C;= upperCutoff</code>, otherwise 0</td>
</tr>
<tr>
<td><code>gaussian</code></td>
<td><code>exp(-0.5 * r² / sigma²)</code>; sigma is in frequency space</td>
</tr>
<tr>
<td><code>notch</code></td>
<td>0 inside or on either disk centered at <code>notchCenter</code> and its negative, otherwise 1</td>
</tr>
</tbody>
</table>
<p>Notch distances wrap on the frequency torus (period 1 per axis), including Nyquist aliases. Conjugate bins share one discrete frequency representative so they receive identical responses even at rounding boundaries. Nyquist-axis distances use <code>0.5 - abs(center)</code> directly, avoiding cancellation in wrapped subtraction. Removing conjugate disks preserves a real result. DC has frequency zero. A zero low-pass cutoff keeps only the mean; a zero high-pass cutoff removes only the mean. A zero notch radius removes exact matching bins. A maximum radial cutoff includes the Nyquist corner.</p>
<p><code>parameters</code> is a packed float32 view with at least <code>GPU_RASTER_SPECTRAL_FILTER_PARAMETER_LENGTH</code> (8) rows. Use <code>getGPURasterSpectralFilterParameterValues(settings)</code> and a caller-owned <code>GPUParameterBuffer</code>. The layout is <code>[cutoff, lowerCutoff, upperCutoff, sigma, notchColumn, notchRow, notchRadius, 0]</code>. Defaults are <code>[0.25, 0, 0.25, 0.1, 0.25, 0, 0.05, 0]</code>. Radial settings lie in <code>[0, sqrt(0.5)]</code>, lower cutoff must not exceed upper cutoff, notch centers lie in <code>[-0.5, 0.5]</code>, and sigma lies in <code>[MINIMUM_RASTER_SPECTRAL_SIGMA, sqrt(0.5)]</code> (minimum approximately <code>1e-6</code>). The packer validates all settings, even those unused by the selected filter. Invalid settings written directly to the view produce invalid outputs on the GPU.</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">import</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8"> {GPUParameterBuffer} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">from</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF"> '@luma-contributors/analysis/analysis-core'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">;</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">import</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8"> {</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  GPURasterSpectralFilter,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  getGPURasterSpectralFilterParameterValues</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">from</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF"> '@luma-contributors/analysis/raster-analysis'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">;</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> parameters</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUParameterBuffer</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(device, {</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  id: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'filter-settings'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, format: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'float32'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, length: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">8</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  values: </span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">getGPURasterSpectralFilterParameterValues</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({cutoff: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">0.125</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">})</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">});</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPURasterSpectralFilter</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  grid, input, filter: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'low-pass'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, parameters: parameters.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">importToGraph</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(graph),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  output: {values, validity, status: {invalidCount}}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// Between encodings of the same compiled graph:</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">parameters.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">write</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">getGPURasterSpectralFilterParameterValues</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({cutoff: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">0.25</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// Destroy the compiled graph before parameters.destroy().</span></span></code></pre>
<p>Storage, calibration and nodata metadata are validated at construction. Buffer values must be packed scalars matching <code>input.format</code>, and validity must be packed uint32. Texture storage must preserve the scalar format (<code>r32float</code>/<code>rgba32float</code>, <code>r32uint</code>/<code>rgba32uint</code>, or <code>r32sint</code>/<code>rgba32sint</code>) and select one 2D mip, one array layer and one sample. Channel zero is the default and the only accepted channel for scalar textures; RGBA textures accept channels 0–3. A selected mip or layer may have a nonzero base index. Scale and offset must fit finite float32; float nodata must fit finite float32 or be NaN. Integer nodata must be an integer in the raw storage range: <code>[0, 4294967295]</code> for uint32 or <code>[-2147483648, 2147483647]</code> for sint32. Band scale/offset, sentinel nodata, validity, and nonfinite observations are canonicalized through published raster primitives. Because all Fourier coefficients depend on the whole image, <strong>one missing input invalidates every output</strong>. Invalid output values are NaN and <code>output.validity</code> is zero. Otherwise validity is one for finite results. Optional <code>output.status.invalidCount</code> is a standard <code>GPUInvalidStatusPort</code> scalar counting missing input cells, before global propagation; it does not count invalid parameter settings or floating-point overflows. Output values and validity cover at least <code>width * height</code> rows. Input and output padding rows are ignored; outputs cannot overlap inputs or one another.</p>
<p>{/* lens:GPURasterSpectralFilter */}</p>
<p>::lens{scenario=gpu-raster.raster-spectral-filter rung=graph}</p>
<p>The synthetic 4×4 image is mean five plus a Nyquist checkerboard of amplitude two. A low-pass
cutoff of zero keeps only DC, so every filtered cell is five, every validity is one, and the
missing-input count is zero. The graph exposes the published forward and inverse transforms.</p>
<p>Compile-time properties are dimensions, band storage/calibration, filter and view layouts. Observations, validity and frequency parameters are per-frame. Float32 FFT arithmetic has the usual rounding and dynamic-range limits; a nonfinite final result is invalid. This is a frequency mask, not general FFT convolution.</p>
<p>{/* parameters: GPURasterSpectralFilter generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h3 id="gpurasterspectralfilter-parameters"><code>GPURasterSpectralFilter</code> parameters</h3>
<ul>
<li>Packer: <code>getGPURasterSpectralFilterParameterValues(settings?, target?)</code>, writes <code>GPU_RASTER_SPECTRAL_FILTER_PARAMETER_LENGTH</code> (8) <code>float32</code> values.
<ul>
<li><code>target</code> (<code>unknown</code>)</li>
</ul>
</li>
</ul>
<p>Settings of <code>getGPURasterSpectralFilterParameterValues</code>:</p>
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
<td><code>cutoff</code> (optional)</td>
<td><code>number</code></td>
<td>Low-pass includes this radius; high-pass excludes it. Default 0.25 cycles/pixel.</td>
</tr>
<tr>
<td><code>lowerCutoff</code> (optional)</td>
<td><code>number</code></td>
<td>-</td>
</tr>
<tr>
<td><code>upperCutoff</code> (optional)</td>
<td><code>number</code></td>
<td>-</td>
</tr>
<tr>
<td><code>sigma</code> (optional)</td>
<td><code>number</code></td>
<td>Gaussian standard deviation in frequency space, default 0.1 cycles/pixel.</td>
</tr>
<tr>
<td><code>notchCenter</code> (optional)</td>
<td><code>readonly [number, number]</code></td>
<td>Notch center (column,row) frequency in [-0.5,0.5]; its conjugate is also removed.</td>
</tr>
<tr>
<td><code>notchRadius</code> (optional)</td>
<td><code>number</code></td>
<td>-</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>`};export{e as default};