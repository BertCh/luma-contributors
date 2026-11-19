var e={slug:`api/variable-standardization`,title:`GPUVariableStandardization`,html:`<h1 id="gpuvariablestandardization">GPUVariableStandardization</h1>
<p>Chunked GraphVectorView columns support up to 4,294,967,295 logical rows, subject to device
bindings. Counts are uint32; float32 statistical fields remain rounded.</p>
<p><code>GPUVariableStandardization</code> is a command-node producer for five named variable transformations.
It composes shared <code>GPUReduction</code> passes and elementwise WGSL. It never submits, reads back,
or computes statistics on the CPU. Add it to a <code>GPUCommandGraph</code> and compile/encode normally.</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> recipe</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUVariableStandardization</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  values, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// GraphDataView&#x3C;'float32'> or GraphVectorView&#x3C;'float32'></span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  mask, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// optional uint32 view/vector: nonzero includes a row</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  method: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'z-score'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  variance: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'sample'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// default; 'population' also available</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  output </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// caller-owned float32 view/vector</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">});</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(recipe);</span></span></code></pre>
<p>For the finite, unmasked rows of the entire logical column, let <code>n</code> be count, <code>mean</code> the
arithmetic mean, <code>minimum</code>/<code>maximum</code> the extrema, and <code>range = maximum - minimum</code>:</p>
<table>
<thead>
<tr>
<th>Method</th>
<th>Formula</th>
<th>Constant population</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>deviation-from-mean</code></td>
<td><code>x - mean</code></td>
<td>zero</td>
</tr>
<tr>
<td><code>z-score</code> (default sample)</td>
<td><code>(x - mean) / sqrt(sum((x - mean)^2) / (n - 1))</code></td>
<td>zero when <code>n >= 2</code></td>
</tr>
<tr>
<td><code>z-score</code>, population</td>
<td><code>(x - mean) / sqrt(sum((x - mean)^2) / n)</code></td>
<td>zero</td>
</tr>
<tr>
<td><code>mad</code></td>
<td><code>(x - mean) / (sum(abs(x - mean)) / n)</code></td>
<td>original value</td>
</tr>
<tr>
<td><code>range-adjust</code></td>
<td><code>x / range</code></td>
<td>original value</td>
</tr>
<tr>
<td><code>range-standardize</code></td>
<td><code>(x - minimum) / range</code></td>
<td>original value</td>
</tr>
</tbody>
</table>
<p>MAD means <strong>mean absolute deviation about the arithmetic mean</strong>. It is not median absolute
deviation and has no normal consistency factor. Sample z-score with fewer than two included
rows produces NaN. Excluded rows (mask zero, NaN, positive or negative infinity) always output
NaN, including in constant populations. An all-excluded column therefore outputs all NaN.
Null validity must be represented by NaN or the caller's mask; this API does not inspect Arrow
validity bitmaps. Every output row is rewritten on each encoding.</p>
<p><code>values</code>, <code>mask</code> and <code>output</code> accept aligned chunked <code>GraphVectorView</code> columns with the same
logical length. Statistics span the complete column. Source and output buffers must be disjoint;
all views must belong to the graph. The logical row count is 0 through 4,294,967,295, subject to
device binding and backing allocation limits. An empty column emits no output rows. Lengths, method, variance convention, and mask presence are topology;
buffer contents are per-frame data. Buffers remain caller-owned and scratch is graph-owned.</p>
<p>Arithmetic uses float32 shared hierarchical reductions, not GeoDa's float64 loops. Summation order
is a hierarchical reduction, not GeoDa's sequential loop, so compare within a float32 tolerance, not bitwise. A shared extent reduction supplies an origin and radius. The origin is the minimum for same-sign extrema, otherwise zero. Values are normalized around
that origin before mean and dispersion reductions; z-scores and MAD are computed in normalized
space. Extreme divisors are rescaled before division to avoid reciprocal underflow on GPUs
that flush subnormal results. This avoids raw-sum/squared-deviation overflow and cancellation from large common offsets.
Underflow and float32 rounding can still reduce accuracy. Unrepresentable final deviations or
range-adjusted values become NaN. This is not a float64 statistics API.</p>
<h2 id="audited-reference">Audited reference</h2>
<p>The formulas were inspected in GeoDa <code>GenUtils.cpp</code> at commit
<a href="https://github.com/GeoDaCenter/geoda/blob/37510860bd9f7b8ae145c15759f7abc5ab217cf3/GenUtils.cpp#L1343"><code>37510860bd9f7b8ae145c15759f7abc5ab217cf3</code></a>:
<code>DeviationFromMean</code>, <code>StandardizeData</code> (sample denominator <code>n - 1</code>),
<code>MeanAbsoluteDeviation</code> (mean-centered absolute deviations divided by <code>n</code>), <code>RangeAdjust</code>, and
<code>RangeStandardize</code>. These independently implemented recipes have no GeoDa runtime dependency.</p>
<p>The named mathematical formulas agree with GeoDa for finite nonconstant data, with these explicit
policies: singleton sample z-score is NaN; excluded values are NaN; infinities are excluded.
GeoDa's zero-MAD and zero-range early returns preserve original values, retained here. Its constant
z-score centers to zero before returning failure, retained for <code>n >= 2</code>. GeoDa's range routines
initialize maximum with <code>DBL_MIN</code> and use an <code>else if</code> update, which can miscompute extrema for
negative-only or strictly decreasing input. These recipes compute the actual extrema instead of
reproducing those implementation defects. No claim of byte-for-byte GeoDa parity is made.</p>
<p>{/* lens:GPUVariableStandardization */}</p>
<p>::lens{scenario=gpu-dataframe.variable-standardization rung=graph}</p>`};export{e as default};