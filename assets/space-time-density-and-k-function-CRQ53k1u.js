var e={slug:`api/gpu-spatial-analysis/space-time-density-and-k-function`,title:`Space-time kernel density, relative risk, K function and Diggle test`,html:`<h1 id="space-time-kernel-density-relative-risk-k-function-and-diggle-test">Space-time kernel density, relative risk, K function and Diggle test</h1>
<p>STX-04 and STX-05. <code>GPUSpaceTimeKernelDensity</code> and <code>GPUSpaceTimeRelativeRisk</code> import from
<code>@luma-contributors/analysis/gpu-spatial-analysis</code>. <code>GPUSpaceTimeKFunction</code> and
<code>GPUSpaceTimeDiggleTest</code> import from <code>@luma-contributors/analysis/dataframe-analysis</code>, beside
<code>GPURipley</code> and <code>GPUSpaceTimeVariogram</code>, because they run on the indexed pair histogram.</p>
<p>Times are float32 relative to a nearby origin, in one unit (see
<a href="/luma-contributors/docs/api/gpu-spatial-analysis/temporal-context">Temporal context</a> and <code>getGPUSpaceTimeGuardedRelativeTimes</code>). A
<code>temporalContext</code> prop is optional, must be <code>float32-relative</code>, and is checked and exposed, never
converted. Every time length below (<code>temporalBandwidth</code>, <code>sliceInterval</code>, <code>maximumTimeLag</code>, the time
window) is in that unit.</p>
<h2 id="references-and-status-of-the-oracles">References and status of the oracles</h2>
<p>The references are R-only and GPL. R is not installed on the development machines, so there are no
sparr, stpp or stopp fixtures yet, and the splancs fixture is a NumPy port of the splancs 2.01-45
source, not a run of R (<code>modules/analysis/test/oracles/space-time-diggle-splancs-port.py</code>; see the
Diggle test below). Every spec compares the GPU result with a binary64 CPU
brute force written from the definitions on this page. The reference code shares nothing with the
source under test: the temporal window mass is a Simpson integral, the circle fraction of the edge
weight comes from circle and side-line crossing angles, and the time permutations use the
TypeScript twin of the shuffle. The definitions follow the cited papers and package manuals; where
this page says a convention is unverified, it was not checked against the package.</p>
<table>
<thead>
<tr>
<th>Operator</th>
<th>Definition</th>
<th>Package pinned for the pending fixture</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>GPUSpaceTimeKernelDensity</code></td>
<td>Fernando and Hazelton (2014), fixed bandwidths</td>
<td>sparr 2.3-16 <code>spattemp.density</code></td>
</tr>
<tr>
<td><code>GPUSpaceTimeRelativeRisk</code></td>
<td>the same, time-varying controls</td>
<td>sparr 2.3-16 <code>spattemp.risk</code></td>
</tr>
<tr>
<td><code>GPUSpaceTimeKFunction</code></td>
<td>Gabriel and Diggle (2009)</td>
<td>stpp 2.0-8 <code>STIKhat</code>, <code>PCFhat</code>, <code>LISTAhat</code>; stopp 1.0.0</td>
</tr>
<tr>
<td><code>GPUSpaceTimeDiggleTest</code></td>
<td>Diggle, Chetwynd, Haggkvist and Morris (1995)</td>
<td>splancs 2.01-45 <code>stkhat</code>, <code>stmctest</code> (a port of the source is committed; the R run is pending), surveillance <code>stKtest</code></td>
</tr>
</tbody>
</table>
<h2 id="space-time-kernel-density-gpuspacetimekerneldensity">Space-time kernel density: <code>GPUSpaceTimeKernelDensity</code></h2>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> grid</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> makeGPURasterMetadata</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({width: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">256</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, height: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">256</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">/* cellSize, origin */</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">});</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> parameterValues</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> getGPUSpaceTimeKernelDensityParameterValues</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  {bandwidth: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">500</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, temporalBandwidth: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">3</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, firstSliceTime: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, sliceInterval: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, timeWindow: [</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">63</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">]},</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  grid,</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">  64</span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D"> // sliceCount</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">);</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">parameterBuffer.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">write</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(parameterValues); </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// a GPUParameterBuffer, imported as \`parameters\`</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">  new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUSpaceTimeKernelDensity</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    positions, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// float32x2</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    times, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// float32, relative</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    weights, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// optional float32</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    parameters, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// float32, GPU_SPACE_TIME_KERNEL_DENSITY_PARAMETER_LENGTH (14) values</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    grid,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    sliceCount: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">64</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    indexGridSize: [</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">512</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">512</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">],</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    indexBounds: [minX, minY, maxX, maxY],</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    spatialKernel: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'gaussian'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// or 'quartic' (alias 'bisquare'), 'epanechnikov'</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    temporalKernel: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'gaussian'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// or 'quartic', 'epanechnikov'</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    spatialEdgeCorrection: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'uniform'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// or 'none'</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    temporalEdgeCorrection: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'uniform'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// or 'none'</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    normalization: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'intensity'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// or 'probability'</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    output: {values, weightSum, status: {invalidCount}}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  })</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">);</span></span></code></pre>
<p>For the voxel at cell center <code>u</code> and slice time <code>t_k = firstSliceTime + k * sliceInterval</code>:</p>
<p><code>f(u, t) = sum_i w_i Ks_h(|u - x_i|) Kt_l(t - t_i) / (es(u) et(t))</code></p>
<ul>
<li><code>Ks_h</code> is the radially symmetric 2D kernel of <code>GPUKernelDensity</code> (gaussian truncated at 5
bandwidths, quartic, epanechnikov). <code>Kt_l(d) = K(d / l) / l</code> is a 1D kernel integrating to one:
gaussian <code>exp(-s^2 / 2) / sqrt(2 pi)</code> truncated at <code>s = 5</code>, epanechnikov <code>3/4 (1 - s^2)</code>, quartic
<code>15/16 (1 - s^2)^2</code>, with <code>s = |d| / l</code>.</li>
<li><code>es(u)</code> and <code>et(t)</code> are the kernel masses inside the spatial window (<code>parameters.window</code>, the
grid rectangle by default) and the time window (<code>parameters.timeWindow</code>, first to last slice time
by default) under <code>'uniform'</code> correction, and 1 under <code>'none'</code>. Points outside a corrected window
are excluded and counted in <code>status.invalidCount</code>; voxels whose center or slice time is outside a
corrected window are NaN. This is the <code>sedge</code> / <code>tedge = "uniform"</code> of sparr.</li>
<li><code>'probability'</code> divides by the sum of included weights.</li>
</ul>
<p>The output is cell-major, row <code>(y * width + x) * sliceCount + slice</code>, the layout
<code>GPUCubeLayoutTranspose</code> calls <code>'cell-major'</code>.</p>
<p>{/* lens:GPUSpaceTimeKernelDensity */}</p>
<h3 id="how-it-runs">How it runs</h3>
<p>One thread per voxel over a grid index with a time-sorted run per cell. Rows are stably sorted by
time (radix sort of the order-preserving float key; ignored rows sort last), copied in that order
into a <code>GPUGridIndex</code>, and the row order inside every index cell is restored with
<code>createGridCellRowOrderNodes</code>, which leaves every cell a run in ascending time. A voxel visits the
index cells under its spatial support in a fixed order, binary-searches each run for the first
point inside the temporal support and walks the run until its time leaves the support. There are no
float atomics, so the result is deterministic for a device.</p>
<p>Cost per voxel is the cells under the spatial support times a logarithm, plus the points inside the
space-time cylinder. The number of points per cell is not capped; a very dense cell costs its
points inside the window for every voxel. The voxel count is limited by the 4-byte output binding
(33554432 on a default device) and the row count by the 16-byte cell-ordered point table (8388608).</p>
<p>Per-frame: <code>parameters</code> (both bandwidths, grid placement, both windows, slice times) and every input
buffer. Compile-time: grid size, <code>sliceCount</code>, point count, <code>indexGridSize</code>, <code>indexBounds</code>, both
kernels, both edge corrections, <code>normalization</code> and the optional outputs.</p>
<h3 id="not-implemented">Not implemented</h3>
<p>Diggle-style per-point edge reweighting (<code>GPUKernelDensity</code> has it, the space-time density does
not), adaptive bandwidths, the bandwidth selectors (<code>OS</code>, <code>NS</code>, <code>LSCV</code>, <code>LIK</code>, <code>BOOT</code>),
tolerance contours, and a polygon study window. Choose both bandwidths and the windows explicitly.
The sparr convention for the temporal edge window of the default (<code>tlim</code>) was not verified; this
operator takes the first and last slice time unless <code>timeWindow</code> is given. With one slice that
default is a single instant, which holds no kernel mass: under <code>temporalEdgeCorrection: 'uniform'</code>
every voxel is then NaN, so pass <code>timeWindow</code> for a single slice (or use <code>'none'</code>).</p>
<p>{/* parameters: GPUSpaceTimeKernelDensity generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h3 id="gpuspacetimekerneldensity-parameters"><code>GPUSpaceTimeKernelDensity</code> parameters</h3>
<ul>
<li>Packer: <code>getGPUSpaceTimeKernelDensityParameterValues(settings, grid, sliceCount)</code>, writes <code>GPU_SPACE_TIME_KERNEL_DENSITY_PARAMETER_LENGTH</code> (14) <code>float32</code> values.
<ul>
<li><code>grid</code> (<code>GPURasterMetadata</code>): The output grid, as for <code>getGPUKernelDensityParameterValues</code>.</li>
<li><code>sliceCount</code> (<code>number</code>): Number of slices (a positive integer), to default the time window.</li>
</ul>
</li>
</ul>
<p>Settings of <code>getGPUSpaceTimeKernelDensityParameterValues</code>:</p>
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
<tr>
<td><code>temporalBandwidth</code></td>
<td><code>number</code></td>
<td>Temporal kernel bandwidth, positive and finite, in the unit of the relative <code>times</code>.</td>
</tr>
<tr>
<td><code>firstSliceTime</code></td>
<td><code>number</code></td>
<td>Time of slice 0 in the unit and origin of <code>times</code>. Slice <code>k</code> is evaluated at <code>firstSliceTime + k * sliceInterval</code>.</td>
</tr>
<tr>
<td><code>sliceInterval</code></td>
<td><code>number</code></td>
<td>Time between slices, positive and finite.</td>
</tr>
<tr>
<td><code>timeWindow</code> (optional)</td>
<td><code>readonly [number, number]</code></td>
<td><code>[start, end]</code> of the temporal edge correction window, <code>start &#x3C; end</code>. Defaults to the first and last slice time. Ignored without <code>temporalEdgeCorrection: 'uniform'</code>. With one slice the default is the instant of that slice, which holds no kernel mass: under the uniform temporal correction every voxel is then NaN, so give one slice an explicit window.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h2 id="space-time-relative-risk-gpuspacetimerelativerisk">Space-time relative risk: <code>GPUSpaceTimeRelativeRisk</code></h2>
<p><code>GPUSpaceTimeRelativeRisk</code> builds the case and control densities as two <code>GPUSpaceTimeKernelDensity</code>
pipelines with <code>normalization: 'probability'</code>, the same bandwidths, kernels, windows, grid and
index, and writes one ratio kernel:</p>
<ul>
<li><code>scale: 'log'</code> (default, sparr <code>log = TRUE</code>): <code>log(f / g)</code>, <code>-Infinity</code> where the case density is
zero.</li>
<li><code>scale: 'ratio'</code>: <code>f / g</code>.</li>
<li>NaN where either density is NaN or the control density is not positive.</li>
</ul>
<p>Under equal distributions the risk is 1 (log 0). Optional outputs <code>caseDensity</code> and <code>controlDensity</code>
expose the two densities, and <code>status.caseInvalidCount</code> / <code>controlInvalidCount</code> the excluded rows.</p>
<p>Not implemented: time-static controls (a spatial-only control density with the conditional case
density), asymptotic tolerance contours and p-values (they need the asymptotic variance of Fernando
and Hazelton, which was not verified), and separate case and control bandwidths.</p>
<p>{/* lens:GPUSpaceTimeRelativeRisk */}</p>
<p>{/* parameters: GPUSpaceTimeRelativeRisk generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h3 id="gpuspacetimerelativerisk-parameters"><code>GPUSpaceTimeRelativeRisk</code> parameters</h3>
<ul>
<li>Packer: <code>getGPUSpaceTimeKernelDensityParameterValues(settings, grid, sliceCount)</code>, writes <code>GPU_SPACE_TIME_KERNEL_DENSITY_PARAMETER_LENGTH</code> (14) <code>float32</code> values.
<ul>
<li>Input port: <code>parameters</code>.</li>
<li><code>grid</code> (<code>GPURasterMetadata</code>): The output grid, as for <code>getGPUKernelDensityParameterValues</code>.</li>
<li><code>sliceCount</code> (<code>number</code>): Number of slices (a positive integer), to default the time window.</li>
</ul>
</li>
</ul>
<p>Settings of <code>getGPUSpaceTimeKernelDensityParameterValues</code>:</p>
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
<tr>
<td><code>temporalBandwidth</code></td>
<td><code>number</code></td>
<td>Temporal kernel bandwidth, positive and finite, in the unit of the relative <code>times</code>.</td>
</tr>
<tr>
<td><code>firstSliceTime</code></td>
<td><code>number</code></td>
<td>Time of slice 0 in the unit and origin of <code>times</code>. Slice <code>k</code> is evaluated at <code>firstSliceTime + k * sliceInterval</code>.</td>
</tr>
<tr>
<td><code>sliceInterval</code></td>
<td><code>number</code></td>
<td>Time between slices, positive and finite.</td>
</tr>
<tr>
<td><code>timeWindow</code> (optional)</td>
<td><code>readonly [number, number]</code></td>
<td><code>[start, end]</code> of the temporal edge correction window, <code>start &#x3C; end</code>. Defaults to the first and last slice time. Ignored without <code>temporalEdgeCorrection: 'uniform'</code>. With one slice the default is the instant of that slice, which holds no kernel mass: under the uniform temporal correction every voxel is then NaN, so give one slice an explicit window.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h2 id="space-time-k-pair-correlation-local-statistics-gpuspacetimekfunction">Space-time K, pair correlation, local statistics: <code>GPUSpaceTimeKFunction</code></h2>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> parameterValues</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> getGPUSpaceTimeKFunctionParameterValues</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  bounds: [</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">10_000</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">8_000</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">],</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  maximumDistance: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">1_500</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  maximumTimeLag: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">14</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  timeWindow: [</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">365</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">],</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  spatialBandwidth: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">150</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// pair correlation only</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  temporalBandwidth: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">2</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">});</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">parameterBuffer.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">write</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(parameterValues); </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// a GPUParameterBuffer, imported as \`parameters\`</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">  new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUSpaceTimeKFunction</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    positions,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    times,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    intensities, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// optional: inhomogeneous estimator</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    mask, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// optional</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    parameters, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// GPU_SPACE_TIME_K_FUNCTION_PARAMETER_LENGTH (11) values</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    gridSize: [</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">64</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">64</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">],</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    distanceClassCount: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">8</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    timeClassCount: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">7</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    edgeCorrection: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'isotropic'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// or 'none'</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    output: {k, spatialK, temporalK, d, pairCorrelation, pairCounts, local, status: statusView}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  })</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">);</span></span></code></pre>
<p>Notation: <code>W</code> the rectangle <code>bounds</code> (area <code>|W|</code>), <code>T = [timeStart, timeEnd]</code> (length <code>|T|</code>), <code>n</code> the
included events, ordered pairs <code>i != j</code>, <code>d_ij</code> the distance and <code>t_ij = |t_i - t_j|</code>.</p>
<ul>
<li>Included events have a nonzero mask, a finite position inside <code>W</code>, a finite time inside <code>T</code>, and
with intensities a finite positive one.</li>
<li>Grid: <code>u_a = maximumDistance (a + 1) / distanceClassCount</code>, <code>v_b = maximumTimeLag (b + 1) / timeClassCount</code>. A pair is in distance class <code>a</code> when <code>d / maximumDistance * U</code> is in <code>(a, a + 1]</code> (distance 0 is class 0), the right-closed convention of <code>GPURipley</code>, and likewise for lags.
All outputs are distance-major, row <code>a * timeClassCount + b</code>.</li>
<li>Exact ties follow stpp 2.0-8 (<code>src/stikfunction.f</code>, read from the CRAN source): a pair at
exactly the largest distance or lag is counted (<code>hij &#x3C;= s</code>, <code>tij &#x3C;= t</code>), and a lag that exactly
reaches a limit of the time window keeps temporal weight 1 (the window is closed). splancs
decides both the other way and <code>GPUSpaceTimeDiggleTest</code> follows splancs, so on times recorded
as whole days the two classes can return different <code>K</code>, <code>K_s</code>, <code>K_t</code> and <code>D</code>.</li>
<li><code>K(u, v) = |W||T| / (n (n - 1)) sum 1(d_ij &#x3C;= u) 1(t_ij &#x3C;= v) w_ij</code> (homogeneous). With
<code>intensities</code>, <code>lambda'_i = lambda_i / (n / (|W||T|))</code> and <code>K = |W||T| / n^2 sum 1(...) w_ij / (lambda'_i lambda'_j)</code>, which is <code>(1 / (|W||T|)) sum 1(...) w_ij / (lambda_i lambda_j)</code>. The
homogeneous <code>n (n - 1)</code> and the inhomogeneous <code>n^2</code> differ by the factor <code>(n - 1) / n</code>; whether
stpp's homogeneous default uses <code>n (n - 1)</code> was not verified.</li>
<li>Spatial marginal <code>K_s(u) = |W| / (n (n - 1)) sum 1(d_ij &#x3C;= u) w^s_ij</code> over all lags, temporal
marginal <code>K_t(v) = |T| / (n (n - 1)) sum 1(t_ij &#x3C;= v) w^t_ij</code> over all distances, and Diggle
<code>D(u, v) = K(u, v) - K_s(u) K_t(v)</code>. Homogeneous only.</li>
<li>Pair correlation <code>g(u, v) = |W||T| / (n (n - 1)) / (4 pi u) sum ks(u - d_ij) kt(v - t_ij) w_ij</code>
at the same grid (with the <code>n^2</code> weights when inhomogeneous), <code>ks</code> and <code>kt</code> normalised
<code>'box'</code> (default, stpp's) or <code>'epanechnikov'</code> kernels of half widths <code>spatialBandwidth</code> and
<code>temporalBandwidth</code>.</li>
<li>Local statistics (LISTA) <code>K_i(u, v) = n s sum_{j != i} 1(...) w_ij / (lambda'_i lambda'_j)</code> with
<code>s</code> the global scale, so the mean of <code>K_i</code> over the included events is <code>K</code>. Output row <code>event * bins + a * timeClassCount + b</code>, NaN for excluded rows, at most 64 classes.</li>
</ul>
<h3 id="edge-corrections-implemented">Edge corrections implemented</h3>
<table>
<thead>
<tr>
<th><code>edgeCorrection</code></th>
<th>Spatial weight <code>w^s_ij</code></th>
<th>Temporal weight <code>w^t_ij</code></th>
</tr>
</thead>
<tbody>
<tr>
<td><code>'none'</code></td>
<td>1</td>
<td>1</td>
</tr>
<tr>
<td><code>'isotropic'</code> (default)</td>
<td>Ripley (1977): the reciprocal of the fraction of the circle about <code>x_i</code> through <code>x_j</code> that lies inside <code>W</code>, exact for the rectangle, floored at 1/100 as in <code>GPURipley</code></td>
<td>1 when both <code>t_i - t_ij</code> and <code>t_i + t_ij</code> lie in <code>T</code>, otherwise 2 (the one-dimensional Ripley weight)</td>
</tr>
</tbody>
</table>
<p>Not implemented: stpp's <code>border</code>, <code>modified.border</code> and <code>translate</code> corrections, polygon study
windows (a rectangle only), and <code>infectious = TRUE</code> (pairs ordered in time). Whether stpp's exact
temporal weight for <code>isotropic</code> equals the one-dimensional Ripley weight above was not verified
against the package; the fixture will decide.</p>
<h3 id="how-it-runs-1">How it runs</h3>
<p>The row filter, the 3 x 3 lattice neighborhood of the shared pair histogram and the temporal
marginal. Pair counts and the temporal marginal are exact integers; weighted sums are fixed point
(2^-16 units) in the deterministic pair histogram; results are bitwise reproducible. LISTA sums add
in float32 in the fixed neighbor order. The temporal marginal uses the sorted times and a binary
search per event and class, with the same class test as the pair pass, so the two agree on every
boundary; its cost is <code>n V log n</code>. The pair search radius is <code>maximumDistance</code>, plus
<code>spatialBandwidth</code> for the pair correlation.</p>
<p>Limits: at most 64 distance and 64 time classes, and <code>(classes + distanceClassCount) * channels &#x3C;= 2048</code> accumulators (3 channels, 5 with the pair correlation); 8388608 rows; local output at most 64
classes and <code>rows * classes</code> at most 33554432. Per-frame: window, maximum distance and lag, time
window, bandwidths and the buffers. Compile-time: row count, <code>gridSize</code>, class counts,
<code>edgeCorrection</code>, kernel and the optional views.</p>
<p>{/* parameters: GPUSpaceTimeKFunction generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h3 id="gpuspacetimekfunction-parameters"><code>GPUSpaceTimeKFunction</code> parameters</h3>
<ul>
<li>Packer: <code>getGPUSpaceTimeKFunctionParameterValues(settings)</code>, writes <code>GPU_SPACE_TIME_K_FUNCTION_PARAMETER_LENGTH</code> (11) <code>float32</code> values.</li>
</ul>
<p>Settings of <code>getGPUSpaceTimeKFunctionParameterValues</code>:</p>
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
<td><code>bounds</code></td>
<td><code>readonly [number, number, number, number]</code></td>
<td>Spatial study window <code>[minX, minY, maxX, maxY]</code>, a rectangle with positive area. Rows outside are excluded.</td>
</tr>
<tr>
<td><code>maximumDistance</code></td>
<td><code>number</code></td>
<td>Largest distance <code>u</code> of the K grid, in position units. Class <code>a</code> has radius <code>u * (a + 1) / distanceClassCount</code>.</td>
</tr>
<tr>
<td><code>maximumTimeLag</code></td>
<td><code>number</code></td>
<td>Largest time lag <code>v</code> of the K grid, in the unit of <code>times</code>. Class <code>b</code> has lag <code>v * (b + 1) / timeClassCount</code>.</td>
</tr>
<tr>
<td><code>timeWindow</code></td>
<td><code>readonly [number, number]</code></td>
<td>Observation interval <code>[start, end]</code> of the times, <code>start &#x3C; end</code>. Rows whose time is outside it are excluded. Its length is <code>|T|</code>.</td>
</tr>
<tr>
<td><code>spatialBandwidth</code> (optional)</td>
<td><code>number</code></td>
<td>Spatial bandwidth <code>hs</code> of the pair correlation kernel, positive. Needed with <code>output.pairCorrelation</code>.</td>
</tr>
<tr>
<td><code>temporalBandwidth</code> (optional)</td>
<td><code>number</code></td>
<td>Temporal bandwidth <code>ht</code> of the pair correlation kernel, positive. Needed with <code>output.pairCorrelation</code>.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h2 id="diggle-test-of-space-time-interaction-gpuspacetimediggletest">Diggle test of space-time interaction: <code>GPUSpaceTimeDiggleTest</code></h2>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">  new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUSpaceTimeDiggleTest</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    positions, times, mask,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    parameters, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// as for GPUSpaceTimeKFunction (bandwidths are ignored)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    randomParameters, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// getGPUSpaceTimeDiggleRandomParameterValues(seed)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    gridSize: [</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">64</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">64</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">],</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    distanceClassCount: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">8</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    timeClassCount: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">7</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    permutationCount: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">99</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    output: {d, statistics, summary, permutationMean, permutationStandardDeviation}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  })</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">);</span></span></code></pre>
<p>Under random relabelling of the times over the locations, <code>K_s</code> and <code>K_t</code> do not change, so only <code>K</code>
is recomputed. Each permutation is a swap-or-not shuffle (the shuffle of <code>GPUKnoxTest</code>, on its own
Philox stream) of the included events, ranked in row order, so the permutations depend on the seed
and the rows only, not on <code>gridSize</code>. The statistic is <code>U = sum D(u_a, v_b)</code>, the plain sum of <code>D</code>
over every class, for the observed pattern and for each permutation: the statistic of splancs
<code>stmctest</code> (<code>sum(kst - outer(ks, kt))</code>) and of surveillance <code>stKtest</code>. It is not standardised, so
classes with a large <code>K</code> weigh more; splancs divides by the <code>stsecal</code> standard error only for the
residual plot of <code>stdiagn</code>. The pseudo p-value for interaction is <code>(1 + #{U_p >= U}) / (P + 1)</code>
(<code>summary[GPU_SPACE_TIME_DIGGLE_SUMMARY.pseudoPGreater]</code>), with the regularity and two-sided
p-values and the z-score beside it. The optional outputs <code>permutationMean</code> and
<code>permutationStandardDeviation</code> are the per-class moments of <code>D</code> over the permutations, for a
standardised residual <code>(d - permutationMean) / permutationStandardDeviation</code>; they are diagnostics
and not part of <code>U</code>. (Until the math review of 2026-11-10 <code>U</code> was the sum of those standardised
residuals, which no reference computes.)</p>
<p>Agreement with splancs: <code>K_s</code>, <code>K_t</code>, <code>K</code>, <code>D</code> and <code>U</code> equal a NumPy port of the splancs 2.01-45
source (<code>stkhat.f</code>, <code>stmctest.S</code>) to about 2e-6 of the largest <code>K</code>, for the observed pattern and
six relabellings of a 72-event scene, and the null distribution of 199 permutations has the
spread of the port's. The fixture is a port of the source, not a run of R; the port's edge weight
is checked against Shapely and its <code>stsecal</code> variance against all 720 relabellings of six events.</p>
<p>Exact ties follow splancs <code>stkhat.f</code>, because times recorded as whole days make them ordinary: a
lag that exactly reaches a limit of the time window has temporal weight 2; a pair at exactly
<code>maximumDistance</code> or <code>maximumTimeLag</code> is dropped (the last class is open above, the others are
closed above); and the class is found with products, so a whole-number lag on a whole-number
break stays in the class the break closes. The fixture has a scene of 60 events on days 0 to 30
with three lag grids (8.5 in 5 classes, 7 in 7, 13 in 13), where these rules act on 128 to 201
ordered pairs at the time limits and up to 166 at the largest lag; the kernel equals the port on
it for the observed pattern, three relabellings and its own permutations. <code>GPUSpaceTimeKFunction</code>
follows stpp at the same ties, so its outputs can differ from the <code>k</code>, <code>spatialK</code>, <code>temporalK</code>
and <code>d</code> of this class on such data (the review measured <code>U</code> 43.95 against 52.26 on 60 whole-day
events).</p>
<p>Differences from splancs that remain: the spatial weight is capped at 100 here; at a distance
exactly equal to the largest, <code>stkhat.f</code> drops the pair from <code>ks</code> but adds a stale term to <code>kst</code>,
which is not reproduced (the pair is dropped from both). splancs also takes a polygon window and
unequal class breaks, and has an analytic standard error (<code>stsecal</code>); this class has none of the
three.</p>
<p>Permutations run in batches of <code>floor(1024 / classes)</code> (and as many as the time table of <code>batch * rows</code> floats fits one binding): a visited pair adds the weight of every permutation of the batch
into its own <code>(permutation, distance, lag)</code> slot, so the sums are exact integers. The command node
count grows with <code>ceil(P / batch)</code> and is exposed as <code>estimatedCommandNodeCount</code>, checked against
the node budget. At most 999 permutations.</p>
<p>Not implemented: the inhomogeneous variant, the other stpp corrections, sequential stopping, and a
permutation test of the local statistics (the local statistics are observed values only).</p>
<p>{/* parameters: GPUSpaceTimeDiggleTest generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h3 id="gpuspacetimediggletest-parameters"><code>GPUSpaceTimeDiggleTest</code> parameters</h3>
<ul>
<li>Packer: <code>getGPUSpaceTimeDiggleRandomParameterValues(seed)</code>, writes <code>GPU_SPACE_TIME_DIGGLE_RANDOM_PARAMETER_LENGTH</code> (2) <code>uint32</code> values.
<ul>
<li>Input port: <code>randomParameters</code>.</li>
<li><code>seed</code> (<code>number</code>): Random seed, an integer in <code>[0, 2^53)</code>. The result is a pure function of the seed and the inputs.</li>
</ul>
</li>
<li>Packer: <code>getGPUSpaceTimeKFunctionParameterValues(settings)</code>, writes <code>GPU_SPACE_TIME_K_FUNCTION_PARAMETER_LENGTH</code> (11) <code>float32</code> values.
<ul>
<li>Input port: <code>parameters</code>.</li>
</ul>
</li>
</ul>
<p>Settings of <code>getGPUSpaceTimeKFunctionParameterValues</code>:</p>
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
<td><code>bounds</code></td>
<td><code>readonly [number, number, number, number]</code></td>
<td>Spatial study window <code>[minX, minY, maxX, maxY]</code>, a rectangle with positive area. Rows outside are excluded.</td>
</tr>
<tr>
<td><code>maximumDistance</code></td>
<td><code>number</code></td>
<td>Largest distance <code>u</code> of the K grid, in position units. Class <code>a</code> has radius <code>u * (a + 1) / distanceClassCount</code>.</td>
</tr>
<tr>
<td><code>maximumTimeLag</code></td>
<td><code>number</code></td>
<td>Largest time lag <code>v</code> of the K grid, in the unit of <code>times</code>. Class <code>b</code> has lag <code>v * (b + 1) / timeClassCount</code>.</td>
</tr>
<tr>
<td><code>timeWindow</code></td>
<td><code>readonly [number, number]</code></td>
<td>Observation interval <code>[start, end]</code> of the times, <code>start &#x3C; end</code>. Rows whose time is outside it are excluded. Its length is <code>|T|</code>.</td>
</tr>
<tr>
<td><code>spatialBandwidth</code> (optional)</td>
<td><code>number</code></td>
<td>Spatial bandwidth <code>hs</code> of the pair correlation kernel, positive. Needed with <code>output.pairCorrelation</code>.</td>
</tr>
<tr>
<td><code>temporalBandwidth</code> (optional)</td>
<td><code>number</code></td>
<td>Temporal bandwidth <code>ht</code> of the pair correlation kernel, positive. Needed with <code>output.pairCorrelation</code>.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h2 id="checks">Checks</h2>
<ul>
<li><code>modules/analysis/test/gpu-spatial-analysis/point-density/space-time-kernel-density.spec.ts</code>
(GPU against the binary64 reference, row-order invariance, bitwise reproducibility, relative risk)
and <code>.node.spec.ts</code> (kernels, window mass, parameters, validation, WGSL audit).</li>
<li><code>modules/analysis/test/gpu-dataframe/pair-statistics/space-time-k-function.spec.ts</code> (every
output, both corrections, intensities, both pair correlation kernels, class-edge ties, row-order
invariance), <code>space-time-diggle-test.spec.ts</code> (statistics, p-values, three batch shapes, seeds),
<code>space-time-diggle-splancs.spec.ts</code> (the kernel against the port of the splancs source: <code>K_s</code>,
<code>K_t</code>, <code>K</code>, <code>D</code>, the <code>stmctest</code> statistic, the null distribution, the tie convention),
<code>space-time-diggle-splancs.node.spec.ts</code> (the binary64 K reference against the same port) and
<code>space-time-k-function.node.spec.ts</code> (validation, WGSL audit, node count against the estimate).</li>
<li>Kernel-audit cases and refusals in
<code>modules/analysis/test/analysis-core/kernel-audit/cases-stx04-stx05-space-time-density-and-k.ts</code>.</li>
</ul>`};export{e as default};