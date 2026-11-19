var e={slug:`api/gpu-dataframe-analysis`,title:`GPU Dataframe analysis contributors`,html:`<h1 id="gpu-dataframe-analysis-contributors">GPU Dataframe analysis contributors</h1>
<h2 id="overview">Overview</h2>
<p>The <code>@luma-contributors/analysis/dataframe-analysis</code> entry point exports analysis contributors for GPU-resident
columns: column profiles, quantiles and class breaks, color scales, group statistics, key joins,
composite scores, inequality measures, calendar buckets, period indices, cube layout transposes, relative times,
time-window filters, temporal reductions,
and pair-statistics tools (variograms, correlograms, Ripley's K, point-pattern indices). Each is an
algorithm or workflow that declares resources and command nodes into a caller-owned
<code>GPUCommandGraph</code> through <code>getCommandNodes(graph)</code> (<code>GPUCommandNodeProducer</code>). A contributor never compiles,
submits, encodes, or reads back.</p>
<h3 id="execution-and-ownership-of-wide-numeric-operations">Execution and ownership of wide numeric operations</h3>
<p>The entry point also exports one-shot functions. These create a separate graph on every call and
await a summary on the CPU; use the contributor when the result must compose into an existing graph.</p>
<table>
<thead>
<tr>
<th>API</th>
<th>Execution and transfer cost</th>
<th>Ownership</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>GPUWideNumericScan</code></td>
<td>Declares nodes in the caller's graph over resident word vectors; no implicit upload, submission or readback.</td>
<td>Caller owns input vectors, initialized output words, compilation, encoding, submission and optional readback.</td>
</tr>
<tr>
<td><code>runWideNumericKernel</code></td>
<td>Allocates and initializes a result buffer, compiles a new graph, encodes, submits, and awaits readback of <code>max(16, resultWordCount * 4)</code> bytes; returns <code>resultWordCount</code> words.</td>
<td>Borrows resident inputs; destroys its compiled graph and result buffer in <code>finally</code>.</td>
</tr>
<tr>
<td><code>aggregateInteger64OnGpu</code>, <code>aggregateInteger64Range</code></td>
<td>Resident-vector one-shot aggregation through <code>runWideNumericKernel</code>; reads 44 bytes and returns a CPU summary.</td>
<td>Borrows input vectors and validity; releases its graph and result buffer.</td>
</tr>
<tr>
<td><code>scanWideNumericColumnOnGpu</code></td>
<td>Resident-vector one-shot narrowing diagnostics; reads 24 bytes and returns a CPU report.</td>
<td>Borrows input vectors and validity; releases its graph and result buffer.</td>
</tr>
<tr>
<td><code>aggregateDecimal128OnGpu</code></td>
<td>Resident-vector one-shot decimal aggregation; reads 76 bytes and returns a CPU summary.</td>
<td>Borrows word vectors and validity; releases its graph and result buffer.</td>
</tr>
<tr>
<td><code>makeDecimal128WordVectors</code></td>
<td>Uploads an Arrow Decimal128 column's words and, when needed, validity. Does not submit a scan or read back.</td>
<td>Caller must invoke the returned <code>destroy()</code> to release the uploaded vectors.</td>
</tr>
<tr>
<td><code>aggregateDecimal128Column</code></td>
<td>Uploads an Arrow Decimal128 column, performs the one-shot decimal aggregation, and awaits its summary readback.</td>
<td>Releases the uploaded vectors in <code>finally</code> as well as the scan's graph and result buffer.</td>
</tr>
</tbody>
</table>
<p>A graph-resident count of present nonzero values can feed another node without a CPU wait:</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">import</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8"> {Buffer, </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">type</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8"> Device} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">from</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF"> '@luma.gl/core'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">;</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">import</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8"> {GPUCommandGraph} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">from</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF"> '@luma.gl/graph/gpu-core'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">;</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">import</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> type</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8"> {GPUVector} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">from</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF"> '@luma.gl/graph/gpu-data'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">;</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">import</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8"> {GPUWideNumericScan} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">from</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF"> '@luma-contributors/analysis/dataframe-analysis'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">;</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">declare</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> device</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">:</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> Device</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">;</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">declare</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> words</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">:</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUVector</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">&#x3C;</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'uint32x2'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">>;</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> resultBuffer</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8"> device.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">createBuffer</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  data: </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> Uint32Array</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">4</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  usage: Buffer.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">STORAGE</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> |</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8"> Buffer.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">COPY_SRC</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> |</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8"> Buffer.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">COPY_DST</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">});</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> graph</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUCommandGraph</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(device);</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> output</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8"> graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">createDataView</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">importBuffer</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({id: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'present-count'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, byteLength: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">16</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, usage: resultBuffer.usage}, resultBuffer),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  {format: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'uint32'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, length: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">4</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">);</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUWideNumericScan</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  id: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'present-count'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, words, resultWordCount: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, output,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  phaseBodies: [</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'if (present &#x26;&#x26; any(pair != vec2u(0u))) { atomicAdd(&#x26;result[0], 1u); }'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">]</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// Add downstream nodes reading output before compiling.</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> compiled</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8"> graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">compile</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">();</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> encoder</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8"> device.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">createCommandEncoder</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">();</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">compiled.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">encode</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(encoder, {parameters: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">undefined</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">});</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">device.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">submit</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(encoder.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">finish</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">());</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// Before reusing this graph, reset resultBuffer to zero, then encode and submit again.</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// Once all consumers have finished: compiled.destroy(); resultBuffer.destroy();</span></span></code></pre>
<p>For an explicit CPU summary, use the one-shot path and await its readback:</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">import</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> type</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8"> {Device} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">from</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF"> '@luma.gl/core'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">;</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">import</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> type</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8"> {GPUVector} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">from</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF"> '@luma.gl/graph/gpu-data'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">;</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">import</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8"> {aggregateInteger64OnGpu} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">from</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF"> '@luma-contributors/analysis/dataframe-analysis'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">;</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">declare</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> device</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">:</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> Device</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">;</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">declare</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> words</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">:</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUVector</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">&#x3C;</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'uint32x2'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">>;</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> summary</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> await</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> aggregateInteger64OnGpu</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({device, words, signed: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">true</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">});</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// summary.sum is a bigint on the CPU. The caller still owns words.</span></span></code></pre>
<h2 id="when-to-use-it">When to use it</h2>
<p>Use these contributors for the fixed analyses that map and dashboard applications repeat every
frame, where inputs are packed graph views and results must stay on the GPU. Use the
<a href="https://luma.gl/docs/api-reference/experimental/gpu-dataframe-operations"><code>GPUDataFrame</code> query API</a> when the question is a general expression,
filter, or aggregation plan.</p>
<ul>
<li><code>GPUGroupStatistics</code> groups by a 32- or 64-bit key and computes many statistics per group (moments,
median, percentiles, mode, unique count). Use <code>GPUDataFrameGroupedAggregationQuery</code> for dense
categorical keys and standard aggregations.</li>
<li><code>GPUKeyJoin</code> joins a left table to a right table by key and supports 1:n aggregates and 64-bit
keys. Use <code>GPUDataFrameJoinQuery</code> for a unique right-side key lookup between dataframes.</li>
<li><code>GPUColumnProfile</code> fuses count, nulls, moments, histogram, distinct estimate, and top categories
into one statistics panel. Use <code>GPUDataFrameHistogramQuery</code> when you only need an explicit
histogram.</li>
<li><code>GPUTimeWindowFilter</code> publishes compacted source rows, optional caller IDs, masks, and fade weights for a window that moves
every frame, including exact <code>Int64</code> time words. Express a fixed time predicate as a dataframe
filter expression instead.</li>
</ul>
<h2 id="quick-start">Quick start</h2>
<p>Use the <a href="/luma-contributors/docs/api/operation-forms-and-cpu-twins#point-density-from-device-to-readback">complete GPU walkthrough</a>
for device creation, buffer imports, submission, typed readback and cleanup. A time-window filter
uses a compact output object rather than a single output view:</p>
<p>Import <code>Float32Array.of(0, 10, 20)</code> as a three-row <code>float32</code> timestamps view. Times and
window bounds share an epoch and units. Allocate the output views below, then add the filter
with this checked function.</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">import</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> type</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8"> {Device} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">from</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF"> '@luma.gl/core'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">;</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">import</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8"> {GPUCommandGraph, </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">type</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8"> GraphDataView} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">from</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF"> '@luma.gl/graph'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">;</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">import</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8"> {GPUParameterBuffer} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">from</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF"> '@luma-contributors/analysis/analysis-core'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">;</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">import</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8"> {</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  GPUTimeWindowFilter, getGPUTimeWindowFilterParameterValues,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  GPU_TIME_WINDOW_FILTER_PARAMETER_LENGTH</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">from</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF"> '@luma-contributors/analysis/dataframe-analysis'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">;</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">export</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> addTimeWindowExample</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span></span>
<span class="line"><span style="--shiki-light:#E36209;--shiki-dark:#FFAB70">  device</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">:</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> Device</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#E36209;--shiki-dark:#FFAB70">graph</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">:</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUCommandGraph</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#E36209;--shiki-dark:#FFAB70">timestamps</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">:</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GraphDataView</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">&#x3C;</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'float32'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">>,</span></span>
<span class="line"><span style="--shiki-light:#E36209;--shiki-dark:#FFAB70">  rows</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">:</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GraphDataView</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">&#x3C;</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'uint32'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">>, </span><span style="--shiki-light:#E36209;--shiki-dark:#FFAB70">count</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">:</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GraphDataView</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">&#x3C;</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'uint32'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">>,</span></span>
<span class="line"><span style="--shiki-light:#E36209;--shiki-dark:#FFAB70">  overflow</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">:</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GraphDataView</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">&#x3C;</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'uint32'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">>, </span><span style="--shiki-light:#E36209;--shiki-dark:#FFAB70">requiredCount</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">:</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GraphDataView</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">&#x3C;</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'uint32'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">></span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">) {</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">  const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> window</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUParameterBuffer</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(device, {</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    id: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'time-window-example-parameters'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    format: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'float32'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    length: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">GPU_TIME_WINDOW_FILTER_PARAMETER_LENGTH</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    values: </span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">getGPUTimeWindowFilterParameterValues</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({start: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">5</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, end: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">15</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">})</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  });</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUTimeWindowFilter</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    timestamps, window: window.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">importToGraph</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(graph),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    output: {rows, status: {count, overflow, requiredCount}}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  }));</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">  return</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8"> window;</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}</span></span></code></pre>
<p>Keep the returned window buffer until the compiled graph has been destroyed.</p>
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
<td><code>rows</code></td>
<td><code>uint32</code>, capacity 3</td>
<td>First <code>count[0]</code> source-row indices only: <code>[1]</code></td>
</tr>
<tr>
<td><code>status.count</code></td>
<td><code>uint32</code>, 1</td>
<td><code>[1]</code>, clamped to capacity</td>
</tr>
<tr>
<td><code>status.requiredCount</code></td>
<td><code>uint32</code>, 1</td>
<td><code>[1]</code>, accepted count before capacity truncation</td>
</tr>
<tr>
<td><code>status.overflow</code></td>
<td><code>uint32</code>, 1</td>
<td><code>[0]</code>; 1 means accepted rows were dropped</td>
</tr>
</tbody>
</table>
<p>The inclusive window accepts only timestamp 10. Spare row slots have no result meaning; use the
count rather than reading the entire capacity as accepted rows. Submit and read the buffers with
the walkthrough's lifecycle, reading overflow alongside IDs. When overflow is nonzero, resize and
rerun or propagate incompleteness. Rewrite the window buffer with the packer to move the window
without recompiling. Destroy the compiled graph before the window and other caller-owned buffers.</p>
<h2 id="conventions">Conventions</h2>
<ul>
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
<li><strong>Rows and IDs.</strong> <code>GPUTimeWindowFilter.output.rows</code> always contains zero-based source rows;
optional <code>output.ids</code> contains caller identities from <code>sourceIds</code>. Node and transient IDs are <code>\${id}-&#x3C;step></code>, so two instances in one graph need
different <code>id</code> props.</li>
<li><strong>One import per buffer.</strong> Import a buffer once (for example with <code>graph.importGPUData()</code> or
<code>GPUParameterBuffer.importToGraph()</code>) and pass the returned view to every contributor that reads
it, so the graph tracks hazards on one logical handle.</li>
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
</ul>
<h2 id="gputimewindowfilter"><code>GPUTimeWindowFilter</code></h2>
<p>Filter timestamped rows against an inclusive time window, producing a row mask and compact accepted source-row indices in <code>output.rows</code>. Use <code>rows</code> as gather indices. Optional <code>sourceIds</code> and <code>output.ids</code> must be supplied together: the latter maps each retained row to its opaque caller identity. Both destinations have the same capacity and must use separate buffers. For times <code>[0, 10]</code>, source IDs <code>[1, 0]</code>, and window <code>[0, 0]</code>, the result is <code>rows = [0]</code>, <code>ids = [1]</code>; gathering attributes <code>[10, 20]</code> by <code>rows</code> returns <code>[10]</code>.</p>
<p>Optional <code>output.status.candidateOverflow</code> is rewritten to zero: this exact filter has no
candidate stage. It is a writable output and must be disjoint from the compact rows and IDs, all other
outputs and every input, including application <code>sourceIds</code>.</p>
<p>The time filter, key join and temporal reduction capture declared input, output and nested status
bindings once at construction, including inherited getters. Later edits to those JavaScript maps
do not redirect writes. Buffer contents remain per-frame values.</p>
<p>::lens{scenario=gpu-dataframe.time-window-filter rung=result caption="588 of 1,500 observations fall inside 2023-06-01 to 2023-09-01 (rows 513 to 1100, CPU check)."}</p>
<p>The lens tags the summary scalar with <code>count</code> and the accepted source-row array with <code>compact</code>.
The first retained source row is 513; the result cardinality and compact panel length are 588.</p>
<p>Filters timestamped rows (instants) or time intervals (trail segments) against a per-frame window
and publishes stable compacted rows, optional caller IDs, a clamped count, and optional mask, fade weights, segment clip
fractions, per-track visible counts, and an indirect <code>instanceCount</code>. All outputs live in one
<code>output</code> prop (<code>GPUTimeWindowFilterOutput</code>: compact <code>rows</code> and <code>status</code>, optional mapped <code>ids</code>, plus <code>outputMask</code>, <code>fadeWeights</code>,
<code>clipFractions</code>, <code>trackVisibleCounts</code>, <code>drawInstanceCount</code>); <code>trackIds</code> is an input. The per-frame settings object is <code>GPUTimeWindowFilterSettings</code> (<code>GPUTimeWindowFilterWordSettings</code> for Int64 words), packed by <code>getGPUTimeWindowFilterParameterValues</code> or <code>getGPUTimeWindowFilterWordParameterValues</code>. Timestamps come in three
forms, chosen by the view format:</p>
<ul>
<li><code>float32</code> relative to an application epoch, the cheapest path;</li>
<li><code>float32</code> plus <code>timestampsLow</code> (double-single, from <code>splitTimestamps</code>) for inputs that arrive as
float64;</li>
<li>exact <code>Int64</code> words: a <code>uint32x2</code> view holding each value's little-endian <code>(low, high)</code> words,
for example Arrow <code>Int64</code> or <code>Timestamp</code> epoch milliseconds uploaded without a CPU pass
(<code>getInt64TimeWords</code>). The window is
then a <code>uint32</code> parameter buffer written with <code>getGPUTimeWindowFilterWordParameterValues</code>, whose start
and end are a <code>number</code> (with a sub-unit fraction for a smooth playhead) or a <code>bigint</code>. Kernels
order times lexicographically on the words and subtract with a borrow before converting to
float32, so acceptance is exact for every <code>Int64</code> value, including an <code>INT64_MIN</code>/<code>INT64_MAX</code>
"unbounded" window over pre-1970 rows, with no epoch to choose; fade and clip differences have the
right sign for any two values and are exact below 2^24 units.</li>
<li>The window is inclusive. A reversed window (<code>end &#x3C; start</code>) accepts nothing, interval rows that
span it included, and NaN timestamps are never accepted (edge-policy table
<code>TIME_WINDOW_EDGE_POLICY</code> in the tests).</li>
</ul>
<h3 id="sorted-range-mode">Sorted range mode</h3>
<p>When the instants are sorted by time (the usual layout of a recorded track or event log), set
<code>sorted: true</code>. The accepted rows are then one contiguous range, found by two binary searches in a
single one-thread kernel: O(log n) per frame for any row count, with no mask pass, scan or
compaction. The range is written to <code>output.rowRange</code> (<code>[firstRow, rowCount]</code>, <code>rowCount</code> unclamped),
<code>output.drawFirstInstance</code> (the draw record's <code>firstInstance</code>, so the instance index is the row
index) and the usual status words and <code>output.drawInstanceCount</code>; <code>output.rows</code> receives only its
first <code>min(count, rows.length)</code> source-row indices from a kernel that returns early past the count.
An optional gather maps these rows through <code>sourceIds</code> into <code>output.ids</code>. A caller that reads only the range passes a small <code>rows</code> view. Acceptance
uses the same comparison helpers as the mask path (double-single <code>timestampsLow</code> and Int64 words
included), so the rows are exactly those the mask path accepts.</p>
<ul>
<li>Precondition: <code>timestamps</code> (with <code>timestampsLow</code>) is non-decreasing in row order and has no NaN. The
default path does not scan for this. Set <code>validateOrder: true</code> with <code>output.orderViolations</code> (one
<code>uint32</code>) to count adjacent descents on every encoding; it is an O(n) pass, as expensive as the mask
path, so use it while developing or after an upload, not per frame.</li>
<li>Instants only: <code>endTimestamps</code>, <code>additionalPredicates</code>, <code>trackIds</code>, <code>output.trackVisibleCounts</code> and
<code>output.clipFractions</code> are refused, and views must be single packed (unchunked).</li>
<li><code>output.outputMask</code> and <code>output.fadeWeights</code> are still allowed and still cost O(n), so leave them
out when scrubbing very large data.</li>
<li>A reversed window (<code>end &#x3C; start</code>) gives an empty range.</li>
</ul>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUTimeWindowFilter</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  sorted: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">true</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  timestamps: sortedInstants,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  window: window.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">importToGraph</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(graph),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  output: {</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">compactOutput, rowRange, drawInstanceCount}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span></code></pre>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUTimeWindowFilter</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  timestamps: segmentStarts, endTimestamps: segmentEnds,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  window: window.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">importToGraph</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(graph),</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">  // \`output\` is a \`GPUTimeWindowFilterOutput\`: the compact list plus the optional views.</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  output: {</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">    ...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">compactOutput, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// rows, status: {count, overflow, optional requiredCount}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    fadeWeights, clipFractions,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    drawInstanceCount: graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">importGPUData</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'trail-count'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, drawCommands.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">getInstanceCountData</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">))</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  }</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">window.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">write</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">getGPUTimeWindowFilterParameterValues</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({start: now </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">-</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8"> trailLength, end: now, startFadeDuration: trailLength}));</span></span></code></pre>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">import</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> type</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8"> {GPUData} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">from</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF"> '@luma.gl/graph/gpu-data'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">;</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">import</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8"> {</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  GPUTimeWindowFilter,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  GPU_TIME_WINDOW_FILTER_WORD_PARAMETER_LENGTH,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  getGPUTimeWindowFilterWordParameterValues,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  getInt64TimeWords</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">from</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF"> '@luma-contributors/analysis/dataframe-analysis'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">;</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> window</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUParameterBuffer</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(device, {</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  id: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'window'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, format: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'uint32'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, length: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">GPU_TIME_WINDOW_FILTER_WORD_PARAMETER_LENGTH</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">});</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// \`departureValues\` is the BigInt64Array of an Arrow Int64 or Timestamp column. Upload its words</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// as a \`uint32x2\` GPUData (one row per timestamp) with the luma.gl Arrow or GPUData APIs.</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> departureWords</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">:</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> Uint32Array</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> getInt64TimeWords</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(departureValues);</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> departures</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">:</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUData</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">&#x3C;</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'uint32x2'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">> </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8"> uploadedDepartureWords; </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// from departureWords</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUTimeWindowFilter</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  timestamps: graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">importGPUData</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'departures'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, departures),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  window: window.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">importToGraph</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(graph), output</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">window.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">write</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">getGPUTimeWindowFilterWordParameterValues</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({start: playheadMs </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">-</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> 3_600_000</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, end: playheadMs}));</span></span></code></pre>
<h4 id="how-it-computes">How it computes</h4>
<p>{/* lens:GPUTimeWindowFilter */}</p>
<p>::lens{scenario=gpu-dataframe.time-window-filter rung=datum bind=page datum=513 caption="Observation 513, 2023-06-01 10:57, is the first row inside the window."}
::lens{scenario=gpu-dataframe.time-window-filter rung=formula bind=page datum=513}
::lens{scenario=gpu-dataframe.time-window-filter rung=kernel caption="Look at the classify kernel: the window start and end are read from the parameter buffer, not baked into WGSL."}
::lens{scenario=gpu-dataframe.time-window-filter rung=graph caption="One classify node, the visibility workflow that compacts source rows, and a publish node that clamps the count."}</p>
<p>{/* parameters: GPUTimeWindowFilter generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h3 id="gputimewindowfilter-parameters"><code>GPUTimeWindowFilter</code> parameters</h3>
<ul>
<li>Packer: <code>getGPUTimeWindowFilterParameterValues(settings)</code>, writes <code>GPU_TIME_WINDOW_FILTER_PARAMETER_LENGTH</code> (8) <code>float32</code> values.</li>
<li>Packer: <code>getGPUTimeWindowFilterWordParameterValues(settings)</code>, writes <code>GPU_TIME_WINDOW_FILTER_WORD_PARAMETER_LENGTH</code> (8) <code>uint32</code> values.</li>
</ul>
<p>Settings of <code>getGPUTimeWindowFilterParameterValues</code>:</p>
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
<td><code>start</code></td>
<td><code>number</code></td>
<td>Inclusive window start.</td>
</tr>
<tr>
<td><code>end</code></td>
<td><code>number</code></td>
<td>Inclusive window end. A window with <code>end &#x3C; start</code> accepts nothing.</td>
</tr>
<tr>
<td><code>startFadeDuration</code> (optional)</td>
<td><code>number</code></td>
<td>Fade weight ramps 0 to 1 over <code>[start, start + startFadeDuration]</code>. 0 (default) disables the ramp.</td>
</tr>
<tr>
<td><code>endFadeDuration</code> (optional)</td>
<td><code>number</code></td>
<td>Fade weight ramps 1 to 0 over <code>[end - endFadeDuration, end]</code>. 0 (default) disables the ramp.</td>
</tr>
</tbody>
</table>
<p>Settings of <code>getGPUTimeWindowFilterWordParameterValues</code>:</p>
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
<td><code>start</code></td>
<td><code>number | bigint</code></td>
<td>Inclusive window start.</td>
</tr>
<tr>
<td><code>end</code></td>
<td><code>number | bigint</code></td>
<td>Inclusive window end. A window with <code>end &#x3C; start</code> accepts nothing.</td>
</tr>
<tr>
<td><code>startFadeDuration</code> (optional)</td>
<td><code>number</code></td>
<td>Fade weight ramps 0 to 1 over <code>[start, start + startFadeDuration]</code>. 0 (default) disables the ramp.</td>
</tr>
<tr>
<td><code>endFadeDuration</code> (optional)</td>
<td><code>number</code></td>
<td>Fade weight ramps 1 to 0 over <code>[end - endFadeDuration, end]</code>. 0 (default) disables the ramp.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h2 id="gpucolumnquantiles-gpuclassbreaks-gpucolorscale-and-gpubivariateclassification"><code>GPUColumnQuantiles</code>, <code>GPUClassBreaks</code>, <code>GPUColorScale</code>, and <code>GPUBivariateClassification</code></h2>
<p>Choropleth classification on the GPU, every frame, with no readback: exact quantiles and
percentile filters, class breaks, and colour scales that write a packed <code>rgba8</code> column a deck.gl
layer can bind as an attribute. Each output can feed the next contributor as a graph view, so a brush
or filter change recolours the map without a CPU round trip.</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUColumnQuantiles</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  values, mask, quantileCount: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">3</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  parameters: quantileParameters.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">importToGraph</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(graph),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  output: {quantiles, validCount, outputMask, filterBounds} </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// outputMask may be a transient</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUClassBreaks</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  values, mask: outputMask, maximumClassCount: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">9</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  methods: [</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'quantile'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'natural-breaks'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'head-tail'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">], </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// default: every method</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  parameters: breakParameters.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">importToGraph</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(graph),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  output: {breaks, classCount, classCounts}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUColorScale</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  values, mask: outputMask, domain: breaks, domainCount: classCount, palette,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  parameters: scaleParameters.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">importToGraph</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(graph),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  maximumDomainCount: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">10</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, maximumPaletteCount: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">9</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  output: {colors, classIndices, classCounts}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">quantileParameters.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">write</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">getGPUColumnQuantilesParameterValues</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({quantiles: [</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">0.25</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">0.5</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">0.75</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">], filterRange: [</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">0.05</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">0.95</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">]}));</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">breakParameters.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">write</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">getGPUClassBreaksParameterValues</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({method: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'natural-breaks'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, classCount: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">7</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">9</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">));</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">scaleParameters.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">write</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">getGPUColorScaleParameterValues</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({scale: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'threshold'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, domainCount: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">8</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, paletteCount: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">9</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// Any of these writes takes effect on the next encoding, with no recompile.</span></span></code></pre>
<ul>
<li>Class rule shared by all four contributors: breaks are edges <code>e[0..k]</code> (<code>e[0]</code> the smallest and
<code>e[k]</code> the largest finite value). A value is in class <code>i</code> when <code>i</code> inner edges <code>e[1..k-1]</code> are at
or below it (d3 <code>scaleThreshold</code>/<code>bisectRight</code>), compared on order-preserving <code>u32</code> keys, so
<code>-0</code> sorts below <code>+0</code>. Values outside <code>[e[0], e[k]]</code> clamp to the end classes. NaN and masked
rows are skipped (class <code>0xffffffff</code>, the no-data colour). Colours are <code>r | g &#x3C;&#x3C; 8 | b &#x3C;&#x3C; 16 | a &#x3C;&#x3C; 24</code>.</li>
<li><code>GPUColumnQuantiles</code>: exact order statistics by four 8-bit radix-select passes over the ordered
keys (no sort): up to 64 per-frame probabilities, <code>'lower' | 'higher' | 'nearest' | 'linear' | 'midpoint'</code> (numpy names; <code>'linear'</code> is d3 and R-7). The rank is <code>fround(fround(n - 1) * p)</code> and
every interpolation step is rounded separately (no fused multiply-add), so results match a
sorted CPU column bit for bit. <code>filterRange</code> gives deck.gl/kepler
lower/upper percentile filtering: bounds <code>x[floor(n * lower)]</code> and <code>x[ceil(n * upper) - 1]</code>, and a
<code>outputMask</code> of the rows inside them. At most 2^24 rows. About 1.8 ms at 1M rows and 3.5 ms at
4M rows for 9 quantiles plus the filter (Apple M-series, headless Chromium).
<ul>
<li>Which rank is exact. The default rank <code>fround(fround(n - 1) * p)</code> is a float32 product and is
not exact: its fraction is lost from rank 2^23 (16,777,216 rows at <code>p = 0.95</code>: <code>linear</code>
returns <code>x[lo]</code>, numpy interpolates by 0.25), and the float32 <code>p</code> can put <code>floor(h)</code> one rank
from numpy's at any size (101 rows at <code>p = 0.59</code>). The optional <code>rationalProbabilities</code> view
(<code>getGPUColumnQuantilesRationalProbabilityValues([[19, 20], [1, 2]])</code>, uint32 pairs
<code>numerator, denominator</code>) replaces the float32 probabilities: the rank
<code>floor((n - 1) numerator / denominator)</code> and its remainder come from a 64-bit integer product
and division, exact for every row count, and the <code>linear</code> fraction is the correctly rounded
float32 of <code>remainder / denominator</code>. It agrees with <code>numpy.quantile</code> 2.5.4 for <code>linear</code>
everywhere tested, and for the other methods except where numpy's own float64 product
<code>(n - 1) * q</code> lands just below or just above an integer the exact product equals: just below,
<code>lower</code> and <code>midpoint</code> are one rank off (101 rows, <code>q = 0.29</code>: numpy <code>lower</code> returns <code>x[28]</code>,
the exact rank is 29); just above, <code>higher</code> and <code>midpoint</code> are (101 rows, <code>q = 0.55</code>: numpy
<code>higher</code> returns <code>x[56]</code>, the exact rank is 55). With it the column may
exceed 2^24 rows (up to one storage binding, 2^25 rows at the default limit) when no filter
output is requested; the filter ranks stay float32. <code>GPUClassBreaks</code> and <code>GPUGroupStatistics</code>
use the float32 rank only. Fixture and generator:
<code>modules/analysis/test/oracles/dataframe-quantile-rational-rank.py</code>.</li>
</ul>
</li>
<li>One order-statistic policy for <code>GPUColumnQuantiles</code>, the <code>GPUClassBreaks</code> quantile and box-plot
methods, and the <code>GPUGroupStatistics</code> medians and percentiles: the sample is the finite unmasked
values (NaN and ±Infinity are missing), <code>-0</code> equals <code>+0</code>, and a probability that is NaN or outside
<code>[0, 1]</code>, or an empty sample, gives NaN. The result is <code>numpy.quantile(x[numpy.isfinite(x)], p)</code>;
numpy itself keeps infinities and raises on an invalid <code>p</code>. When <code>x[high] - x[low]</code> overflows f32
the convex form <code>x[low] (1 - t) + x[high] t</code> keeps the result finite.</li>
<li><code>GPUClassBreaks</code>: per-frame <code>method</code> and <code>classCount</code> among the compiled <code>methods</code>:
<code>'equal-interval'</code>, <code>'quantile'</code> (exact linear quantiles at <code>i / k</code>), <code>'standard-deviation'</code>
(classes of <code>interval</code> sample standard deviations centred on the mean, as mapclassify <code>StdMean</code>
and R classInt <code>'sd'</code>), <code>'head-tail'</code> (Jiang's
breaks with a per-frame head ratio, default 0.4; 1 gives mapclassify's rule), <code>'box-plot'</code>
(six classes, per-frame hinge), <code>'maximum-breaks'</code> (midpoints of the <code>k - 1</code> largest gaps
between distinct sorted values), <code>'natural-breaks'</code> (Fisher-Jenks, see below), and
<code>'custom'</code>. Head/tail and maximum breaks may return fewer classes; <code>classCount</code> reports <code>k</code>
(0 with no finite values, 1 when they are all equal). Infinities are missing for every edge, so
<code>e[0]</code> and <code>e[k]</code> bound the inner edges. Kernels of unselected methods return at
once, except the two radix sorts of <code>'maximum-breaks'</code>, which run every frame when compiled.
Natural-breaks layers and refinement rounds and head/tail rounds read their index from a GPU
state word and are dispatched indirectly, so the pipeline count does not grow with
<code>maximumClassCount</code> and finished rounds launch no workgroups. Moments use a fixed-order Chan merge, so every output is bitwise
reproducible. About 3 to 4 ms per frame at 1M rows with six methods compiled. Every closed-form
method is checked against mapclassify 2.11 and natural breaks against the exact Fisher-Jenks
partition (an f64 dynamic program, agreeing with jenkspy 0.4.1) by the committed generator
<code>test/oracles/dataframe-class-breaks.py</code>.
<ul>
<li><code>naturalBreaksMode: 'exact'</code> explicitly selects a full Fisher–Jenks dynamic program for
<code>'natural-breaks'</code>; the default <code>'approximate'</code> retains the histogram/refinement path below.
Exact mode accepts at most <strong>8192 source rows</strong> (including masked/missing rows), with the
existing capacity of 64 classes. Construction rejects larger columns; it never falls back
to an approximate method. <code>naturalBreaksExactStorage</code> defaults to <code>'auto'</code>: up to 1024 rows
it materializes interval costs; larger inputs recompute the same backward Welford recurrence
during each DP layer. Explicit <code>'materialized'</code> retains the 1024-row cap; <code>'recompute'</code>
can also be selected for small inputs to minimize storage.
A radix sort places selected finite values first. Equal numeric values (including signed
zeros) cannot be split; the produced count is <code>min(requested, distinct finite values)</code>.
Each inner edge is the first value of the upper class, included in that class. This matches
GeoDa's <code>&#x3C; break</code> assignment contract, pinned to
<a href="https://github.com/GeoDaCenter/geoda/blob/37510860bd9f7b8ae145c15759f7abc5ab217cf3/Explore/CatClassification.cpp#L806">GeoDa 37510860, CatClassification.cpp</a>.
GeoDa's randomized natural-break search itself is not the exact reference; the independent
reference is <a href="https://github.com/mthh/jenkspy/blob/3ff493f8b18af5c0e165d9963ba35ad7dc8f5043/jenkspy/src/_jenks.c">jenkspy 0.4.1</a>.
The objective is total within-class squared deviation. Exact means the global optimum of
the documented GPU arithmetic recurrence: divide values by the largest power of two at or
below <code>max(abs(min), abs(max))</code>
(the magnitude itself for subnormal scales, one for zero), subtract each interval's scaled last value, then
compute each interval's centered sum of squares by backward Welford updates in f32,
then enumerate every eligible split using f32 score addition. Division decomposes both operands with <code>frexp</code>,
divides bounded mantissas, then applies their exponent difference with <code>ldexp</code>; no extreme reciprocal is formed. Arithmetic ties retain the
lowest last split, recursively within the prefix. This is not an f64 or real-arithmetic
guarantee: normalization can underflow tiny values and close objectives can round together;
device multiply-add contraction can also affect close partitions. Work is <code>O(k n²)</code> and
scratch is <code>O(n² + k n)</code> when materialized, or <code>O(k n)</code> when recomputed (two score
rows and the partition backpointers). Recomputation eliminates the quadratic allocation
but repeats the Welford arithmetic; it does not promise lower latency. The 8192-row cap
is a bounded-work contract, not a measured interactive-performance guarantee. Scratch
bindings are checked against the actual device limit. There is no CPU solve, submission or readback. NaNs/masked rows are omitted; infinities are
omitted from the objective and counted in the end classes, as in the approximate mode.</li>
<li><code>naturalBreaksMode: 'monotone'</code> selects a separate <strong>approximate</strong> full-data dynamic program,
for up to <strong>262144 source rows</strong> and 64 classes. It sorts the finite selected values and
uses full distinct source-value boundaries with monotone pruning, avoiding the histogram mode's restricted
initial boundaries and subsequent single-cut local search. It is useful when a full-column
partition search is wanted and exhaustive f32 recurrence optimization is too expensive.
Normalized positions are quantized onto <code>[0, 1048575]</code> (20 bits); exact 64-bit integer
prefix moments avoid raw floating-point prefix cancellation. Interval costs and DP score
additions are still f32, and divide-and-conquer pruning assumes mathematical monotonicity.
<strong>Neither real-arithmetic optimality nor parity with exact mode is guaranteed.</strong> Widely
separated outliers can collapse small distinctions under quantization; this is not a universal
accuracy improvement over the histogram/refinement mode. Edges remain actual source values,
equal numeric values (including signed zero) are indivisible, and arithmetic ties prefer the
lowest split. Each breadth-first search level distributes endpoints and candidate scans across
workgroups; pipelines are reused and inactive levels dispatch no work. Normal search work is
<code>O(k n log n)</code> with <code>O(k n)</code> storage. If rounding leaves a feasible endpoint without a valid
candidate in its pruned interval, that endpoint falls back to exhaustive search to preserve a
valid partition, so worst-case work remains <code>O(k n²)</code>. This fallback does not certify
optimality. Device binding limits are checked before scratch allocation. No GPU-to-CPU readback
or CPU solve is performed.</li>
<li>With the default approximate mode, <code>'natural-breaks'</code> seeks the partition into <code>k</code> classes of
least total within-class squared
deviation. A coarse dynamic program over <code>naturalBreaksBinCount</code> equal-width bins of
<code>[min, max]</code> (2 to 2048, default 2048) sums count, position and squared position of each bin as
exact 64-bit integers, so the class cost is evaluated without the cancellation of <code>S2 - S1^2 / n</code>
in f32 and without a bin-by-bin cost table. A layer of the dynamic program is one workgroup that
exploits the monotone split points of the cost (the quadrangle inequality) over <code>log2(bins)</code>
barrier-separated levels: <code>O(k B log B)</code> work in all, against <code>O(k B^2)</code> before. The bins cannot
see structure finer than <code>range / bins</code>, so the breaks are then refined on the rows, and no
longer snap to bin edges: red-black coordinate descent in which each break, with its two
neighbours fixed, takes the best split of the values in its two adjacent classes (odd breaks and
then even breaks, so a pass never moves two interacting breaks and the cost only falls). The
first window of a break covers the data extent of its two classes, so a heavy outlier cannot hide
the structure, and the window then shrinks around the chosen cell boundary until a cell is
below an f32 ulp. If empty classes show that the bins collapsed the most populated class (one
extreme outlier puts the whole bulk into one bin), the coarse stage runs again over the range of
that class with the classes the empty ones would have held. Rounds end once no cut moved with
every break zoomed twice, at most eight; a column usually settles in two rounds, that is five
passes over the rows (one for the class extents, two per round). Every inner edge <code>e[i]</code> is an
actual data value, the smallest value of class <code>i</code>; mapclassify and jenkspy report the
largest value of each class (<code>b[i-1]</code>), so <code>e[i]</code> is the smallest data value above <code>b[i-1]</code>
and the classes are the same. A class count above the number of distinct values (which
mapclassify and jenkspy refuse) yields one class per distinct value, like <code>'maximum-breaks'</code>;
the produced count is <code>classCount</code>.</li>
<li>Accuracy: on the 34 reference cases (10 columns from lognormal and cluster mixtures to heavy
ties, an extreme outlier of 1e6 times the spread, an outlier pair and a Pareto tail, <code>k</code> from 2
to 10) 32 equal the exact partition, bit for bit in the edges. The other two stop in a
neighbouring local optimum of the coordinate descent (a goodness of variance fit within 2e-6
of the optimum; one of them is the partition mapclassify 2.11 itself returns, which is not
optimal). On random well-scaled columns about 1 in 300 does. On 180 random heavy-tailed
columns (Pareto tails, outliers, clusters beside an outlier) 179 were exact and the other was
within a fit of 4e-8. Structure narrower than a bin that is not the most populated class (for
example several clusters in one bin beside an outlier, when a lighter class also collapses)
keeps the coarse allocation and is refined locally only.
The previous implementation snapped breaks to bin edges (an error up to <code>range / 1024</code>) and
cost <code>O(k B^2)</code> for the cost table and the layers.</li>
</ul>
</li>
<li><code>GPUColorScale</code>: <code>linear</code>, <code>sqrt</code>, <code>pow</code>, <code>log</code> (kepler's <code>logFloor</code> for <code>v &#x3C;= 0</code>), <code>symlog</code>,
<code>quantize</code>, <code>threshold</code>/<code>quantile</code> (binary search over the active edges), and <code>ordinal</code> (a
<code>uint32</code> code column). Scale, domain, palette, <code>clamp</code>, and <code>'step'</code>/<code>'linear'</code> interpolation are
per frame. A multi-stop domain with as many stops as palette entries is piecewise linear, like a
d3 multi-stop scale. A <code>domainCount</code> view (for example <code>GPUClassBreaks.classCount</code>) holds a
class count <code>k</code>, and the domain then has <code>k + 1</code> edges. <code>'step'</code> colours are exact; <code>'linear'</code>
blends round half up (within 1 per channel of a double-precision reference).</li>
<li><code>GPUBivariateClassification</code>: two columns and two edge views give <code>classId = yClass * classCountX + xClass</code> (up to 16 per axis), a colour from an <code>n x n</code> palette, per-class counts,
and optional value-by-alpha (<code>alphaValues</code> mapped through a per-frame <code>[low, high]</code> to
<code>[minimumAlpha, 1]</code> times the palette alpha).</li>
<li>Integer columns: <code>GPUClassBreaks.values</code> accepts <code>uint32</code> and <code>sint32</code> in addition to <code>float32</code>. One
internal pass converts the column to <code>float32</code> (exact to 2^24) and every method behaves as for that
float column. <code>GPUColorScale.values</code> accepts <code>sint32</code> (always numeric) and <code>uint32</code> with
<code>integerValues: 'numeric'</code>, which reads the integers as numbers and applies the <code>scale</code> parameter; the
default <code>'ordinal'</code> keeps <code>uint32</code> as category codes. The value format is topology (fixed at compile
time). This is what lets a count column (for example dense <code>GPUGroupStatistics</code> <code>counts</code>) feed a
choropleth without a float copy.</li>
<li><code>GPUClassBreaksOperation</code> and <code>CPUClassBreaksOperation</code> infer the values format and row count
from the input vector. Optional <code>valuesFormat</code> and <code>rowCount</code> assert that inferred schema;
both forms reject a mismatch and require at least one input row, including when <code>rowCount</code>
is omitted. The same props object selects either execution form. Constructors snapshot the
schema, compiled methods and output port names: mutating that caller object later does not
change an existing operation. Declared fields include inherited and non-enumerable getters,
including <code>mask</code> and output port names, and are captured once at construction.
<code>naturalBreaksExactStorage</code> requires <code>naturalBreaksMode: 'exact'</code> on both forms even when
<code>methods</code> excludes natural breaks; omit the storage option otherwise. This dependency is
independent of device limits. Input vector contents and packed parameter values remain per-frame.</li>
<li>Per-frame settings are the CPU objects <code>GPUColumnQuantilesSettings</code>, <code>GPUClassBreaksSettings</code>,
<code>GPUColorScaleSettings</code>, and <code>GPUBivariateClassificationSettings</code>, packed by the matching
<code>getGPU&#x3C;Class>ParameterValues(settings, target?)</code>.</li>
<li>Inputs must be single packed views; chunked vectors are rejected.</li>
</ul>
<h3 id="gpubivariateclassification"><code>GPUBivariateClassification</code></h3>
<p>Classify two columns into an n by n grid and colour each row from a bivariate palette.</p>
<p>::lens{scenario=gpu-dataframe.bivariate-classification rung=result caption="The 3 by 3 grid puts 218 tracts in the low-poverty, low-uninsured class and 64 in the opposite corner (CPU check)."}</p>
<h4 id="how-it-computes-1">How it computes</h4>
<p>{/* lens:GPUBivariateClassification */}</p>
<p>::lens{scenario=gpu-dataframe.bivariate-classification rung=datum bind=page datum=16 caption="Tract 16 falls in joint class 8: high poverty and high uninsured, shared by 64 tracts (CPU check)."}
::lens{scenario=gpu-dataframe.bivariate-classification rung=formula bind=page datum=16}
::lens{scenario=gpu-dataframe.bivariate-classification rung=kernel caption="Look at classY * classCountX + classX: each axis counts its inner edges at or below the value."}
::lens{scenario=gpu-dataframe.bivariate-classification rung=graph caption="A single bivariate kernel reading two value columns, two edge lists, a palette and parameters."}</p>
<p>{/* parameters: GPUBivariateClassification generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h4 id="gpubivariateclassification-parameters"><code>GPUBivariateClassification</code> parameters</h4>
<ul>
<li>Packer: <code>getGPUBivariateClassificationParameterValues(settings)</code>, writes <code>GPU_BIVARIATE_CLASSIFICATION_PARAMETER_LENGTH</code> (8) <code>float32</code> values.</li>
</ul>
<p>Settings of <code>getGPUBivariateClassificationParameterValues</code>:</p>
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
<td><code>classCountX</code></td>
<td><code>number</code></td>
<td>Active classes along X, <code>1..maximumClassCount</code>; <code>breaksX</code> holds <code>classCountX + 1</code> edges.</td>
</tr>
<tr>
<td><code>classCountY</code></td>
<td><code>number</code></td>
<td>Active classes along Y, <code>1..maximumClassCount</code>; <code>breaksY</code> holds <code>classCountY + 1</code> edges.</td>
</tr>
<tr>
<td><code>noDataColor</code> (optional)</td>
<td><code>number</code></td>
<td>Packed <code>rgba8</code> colour (<code>r | g &#x3C;&#x3C; 8 | b &#x3C;&#x3C; 16 | a &#x3C;&#x3C; 24</code>) for no-data rows. Default <code>0</code>.</td>
</tr>
<tr>
<td><code>valueByAlpha</code> (optional)</td>
<td><code>GPUBivariateValueByAlpha</code></td>
<td>Enables value-by-alpha. Omit to leave the palette alpha unchanged.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h3 id="gpuclassbreaks"><code>GPUClassBreaks</code></h3>
<p>Compute choropleth class edges and per-class counts from a float column, with a switchable method.</p>
<p>::lens{scenario=gpu-dataframe.class-breaks rung=result caption="Equal-interval breaks split 788 valid tracts into classes of 257, 250, 168, 85 and 28 (CPU check)."}</p>
<h4 id="how-it-computes-2">How it computes</h4>
<p>{/* lens:GPUClassBreaks */}</p>
<p>::lens{scenario=gpu-dataframe.class-breaks rung=datum bind=page datum=2 caption="Inner edge e2 is 32.1 percent, the lower bound of the third class (CPU check)."}
::lens{scenario=gpu-dataframe.class-breaks rung=formula bind=page datum=2}
::lens{scenario=gpu-dataframe.class-breaks rung=kernel caption="Look at the equal-interval branch: edges step from the sample minimum by (max - min) / k."}
::lens{scenario=gpu-dataframe.class-breaks rung=graph caption="Moment, quantile and method nodes compiled for the chosen methods, publishing breaks and counts."}</p>
<p>{/* parameters: GPUClassBreaks generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h4 id="gpuclassbreaks-parameters"><code>GPUClassBreaks</code> parameters</h4>
<ul>
<li>Packer: <code>getGPUClassBreaksParameterValues(settings, maximumClassCount)</code>, writes <code>getGPUClassBreaksParameterLength(...)</code> <code>float32</code> values.
<ul>
<li><code>maximumClassCount</code> (<code>number</code>): Compile-time class capacity of the contributor.</li>
</ul>
</li>
</ul>
<p>Settings of <code>getGPUClassBreaksParameterValues</code>:</p>
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
<td><code>method</code></td>
<td><code>GPUClassBreaksMethod</code></td>
<td>Classification method. It must be one of the contributor's compiled <code>methods</code>.</td>
</tr>
<tr>
<td><code>classCount</code> (optional)</td>
<td><code>number</code></td>
<td>Requested class count <code>k</code>. Ignored by <code>'box-plot'</code> (always 6) and <code>'custom'</code> (one less than the number of edges). For <code>'head-tail'</code> and <code>'maximum-breaks'</code> it is an upper bound: fewer classes are produced when the data runs out of heads or distinct gaps.</td>
</tr>
<tr>
<td><code>standardDeviationInterval</code> (optional)</td>
<td><code>number</code></td>
<td><code>'standard-deviation'</code>: width of one class in standard deviations. Default 1.</td>
</tr>
<tr>
<td><code>headTailRatio</code> (optional)</td>
<td><code>number</code></td>
<td><code>'head-tail'</code>: largest head fraction that keeps splitting. Default 0.4 (Jiang's 40% rule). Use 1 for mapclassify's rule, which splits until a head has a single distinct value.</td>
</tr>
<tr>
<td><code>boxPlotHinge</code> (optional)</td>
<td><code>number</code></td>
<td><code>'box-plot'</code>: whisker length in interquartile ranges. Default 1.5.</td>
</tr>
<tr>
<td><code>customEdges</code> (optional)</td>
<td><code>ArrayLike&#x3C;number></code></td>
<td><code>'custom'</code>: ascending class edges <code>e[0..k]</code>, at most <code>maximumClassCount + 1</code> of them.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h3 id="gpuscalarscale"><code>GPUScalarScale</code></h3>
<p><a href="#gpuscalarscale"><code>GPUScalarScale</code></a> maps prepared numeric columns to scalar rendering
attributes using replayable linear, signed square-root, same-sign logarithmic, native fixed,
and categorical point scales. It preserves source chunks and validity without source readback.</p>
<p>The optional input row filter is <code>mask</code> (<code>uint32</code>, one entry per source row; nonzero keeps the row, zero maps the row to <code>nullValue</code> with output validity 0). <code>validity</code> is accepted as a <code>@deprecated</code> alias with the same meaning; <code>mask</code> wins when both are set. Output <code>validity</code> is unchanged. Construction errors start with the contributor ID and name the failed limit, for example <code>scalar-scale mask must hold exactly one entry per source row</code>.</p>
<p>{/* parameters: GPUScalarScale generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h4 id="gpuscalarscale-parameters"><code>GPUScalarScale</code> parameters</h4>
<ul>
<li>Packer: <code>getGPUScalarScaleParameterValues(settings)</code>, writes <code>GPU_SCALAR_SCALE_PARAMETER_LENGTH</code> (4) <code>float32</code> values.</li>
</ul>
<p>Settings of <code>getGPUScalarScaleParameterValues</code>:</p>
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
<td><code>scale</code></td>
<td><code>GPUScalarScaleType</code></td>
<td>-</td>
</tr>
<tr>
<td><code>domainCount</code></td>
<td><code>number</code></td>
<td>Numeric modes require exactly two entries. Point accepts zero through the capacity.</td>
</tr>
<tr>
<td><code>nullValue</code> (optional)</td>
<td><code>number</code></td>
<td>A finite value written for nulls, unknown categories and invalid numerical results.</td>
</tr>
<tr>
<td><code>invert</code> (optional)</td>
<td><code>boolean</code></td>
<td>Reverse the GPU domain as the output range of the native fixed (identity) scale only.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h3 id="gpucolorscale"><code>GPUColorScale</code></h3>
<p>Map a float column to packed rgba8 colours and palette classes with a threshold, step or continuous scale.</p>
<p>Continuous domains may ascend or descend. <code>extrapolate: true</code> extends the first and last
palette segments, matching native RGB interpolation with saturated output bytes; class indices
remain clamped. <code>clamp: true</code> takes precedence. The existing default (<code>clamp: false</code>, no
extrapolation) continues to mark out-of-domain rows as no-data. Signed square-root and power
transforms preserve negative values.</p>
<p><code>logMode: 'domain-sign'</code> selects <code>log(x)</code> for a positive domain or <code>-log(-x)</code> for a negative
domain. Use finite, nonzero, same-sign domain stops. Zero or wrong-sign inputs are no-data,
except that a constant valid domain maps every finite input to its midpoint. Native log and
extrapolation reject nonfinite input or normalized values; masks still represent nulls, with
transparent no-data by default. Omitting <code>logMode</code> preserves the existing positive <code>logFloor</code>
behavior. These options reuse the nine-word parameter buffer, so changing them needs no
recompilation. Float32 RGB interpolation can differ from a double-precision reference by one
byte per channel; subnormal arithmetic is outside that parity contract.</p>
<p>::lens{scenario=gpu-dataframe.color-scale rung=result caption="Five poverty bands hold 117, 181, 171, 148 and 171 tracts; darker blue means higher poverty (CPU check)."}</p>
<h4 id="how-it-computes-3">How it computes</h4>
<p>{/* lens:GPUColorScale */}</p>
<p>::lens{scenario=gpu-dataframe.color-scale rung=datum bind=page datum=1 caption="Tract 1 has 43 percent poverty: above the 40 percent edge, class 4, the darkest blue."}
::lens{scenario=gpu-dataframe.color-scale rung=formula bind=page datum=1}
::lens{scenario=gpu-dataframe.color-scale rung=kernel caption="Look at the binary search over inner domain edges that yields the class."}
::lens{scenario=gpu-dataframe.color-scale rung=graph caption="A single colour-scale kernel reading values, domain, palette and parameters."}</p>
<p>{/* parameters: GPUColorScale generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h4 id="gpucolorscale-parameters"><code>GPUColorScale</code> parameters</h4>
<ul>
<li>Packer: <code>getGPUColorScaleParameterValues(settings)</code>, writes <code>GPU_COLOR_SCALE_PARAMETER_LENGTH</code> (9) <code>float32</code> values.</li>
</ul>
<p>Settings of <code>getGPUColorScaleParameterValues</code>:</p>
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
<td><code>scale</code></td>
<td><code>GPUColorScaleType</code></td>
<td>Scale type.</td>
</tr>
<tr>
<td><code>domainCount</code></td>
<td><code>number</code></td>
<td>Number of active domain entries, <code>1..maximumDomainCount</code>. Ignored when the contributor was given a <code>domainCount</code> view, which then supplies a CLASS count <code>k</code> and the domain has <code>k + 1</code> edges.</td>
</tr>
<tr>
<td><code>paletteCount</code></td>
<td><code>number</code></td>
<td>Number of active palette entries, <code>1..maximumPaletteCount</code>.</td>
</tr>
<tr>
<td><code>interpolation</code> (optional)</td>
<td><code>GPUColorScaleInterpolation</code></td>
<td><code>'step'</code> picks <code>palette[class]</code> exactly; <code>'linear'</code> blends adjacent entries per channel. Default <code>'step'</code>.</td>
</tr>
<tr>
<td><code>clamp</code> (optional)</td>
<td><code>boolean</code></td>
<td>Clamp out-of-domain values to the first or last colour instead of no-data. Default <code>false</code>.</td>
</tr>
<tr>
<td><code>extrapolate</code> (optional)</td>
<td><code>boolean</code></td>
<td>Extrapolate continuous scales along the first/last palette segment, saturating color bytes. Default false; clamp true takes precedence. Ignored by discrete scales.</td>
</tr>
<tr>
<td><code>logMode</code> (optional)</td>
<td><code>'positive-floor' | 'domain-sign'</code></td>
<td>Log transform: legacy positive floor (default), or native same-sign positive/negative domain. Native mode ignores logFloor and requires finite nonzero same-sign domain stops.</td>
</tr>
<tr>
<td><code>noDataColor</code> (optional)</td>
<td><code>number</code></td>
<td>Packed <code>rgba8</code> colour (<code>r | g &#x3C;&#x3C; 8 | b &#x3C;&#x3C; 16 | a &#x3C;&#x3C; 24</code>) for no-data rows. Default <code>0</code>.</td>
</tr>
<tr>
<td><code>logFloor</code> (optional)</td>
<td><code>number</code></td>
<td>Value that replaces <code>v &#x3C;= 0</code> (and non-positive domain entries) on the <code>log</code> scale, as kepler does with a small positive floor. <code>NaN</code> treats such rows as no-data. Default <code>1e-5</code>.</td>
</tr>
<tr>
<td><code>exponent</code> (optional)</td>
<td><code>number</code></td>
<td>Exponent of the <code>pow</code> scale. Default <code>1</code>.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h3 id="gpuscalarscale-1"><code>GPUScalarScale</code></h3>
<p>Map a prepared scalar column to numeric rendering values (radius, width, height) with a native-style <code>linear</code>, <code>sqrt</code>, <code>log</code>, <code>identity</code> or <code>point</code> scale. See <a href="#gpuscalarscale"><code>GPUScalarScale</code></a> for the full contract.</p>
<p>Each mode normalizes the transformed value between the transformed domain endpoints and interpolates the two range endpoints, extrapolating without a clamp. Rows with null or nonfinite input, unknown categories, or unsupported log inputs write the fallback value with output validity zero.</p>
<p>::lens{scenario=gpu-dataframe.scalar-scale rung=result caption="Poverty share mapped to a square-root radius between 1 and 20 pixels; suppressed tracts get the fallback and validity zero."}</p>
<h4 id="how-it-computes-4">How it computes</h4>
<p>{/* lens:GPUScalarScale */}</p>
<p>::lens{scenario=gpu-dataframe.scalar-scale rung=datum bind=page datum=1 caption="Tract 1 maps its poverty share through the signed square root to a radius."}
::lens{scenario=gpu-dataframe.scalar-scale rung=formula bind=page datum=1}
::lens{scenario=gpu-dataframe.scalar-scale rung=kernel caption="Look at the normalization: the transformed value is placed between the transformed domain endpoints, then the range endpoints are interpolated."}
::lens{scenario=gpu-dataframe.scalar-scale rung=graph caption="A single scalar-scale kernel reading values, domain, range and parameters."}</p>
<h3 id="gpucolumnquantiles"><code>GPUColumnQuantiles</code></h3>
<p>Select exact quantiles of a float column and a percentile-range row mask, from column to quantile values.</p>
<p>::lens{scenario=gpu-dataframe.column-quantiles rung=result caption="Poverty quartiles are 14.1, 25.6 and 38.1 percent over 788 valid tracts; 711 tracts sit inside the 5-95 filter (CPU check)."}</p>
<h4 id="how-it-computes-5">How it computes</h4>
<p>{/* lens:GPUColumnQuantiles */}</p>
<p>::lens{scenario=gpu-dataframe.column-quantiles rung=datum bind=page datum=0 caption="First quartile (p = 0.25): 14.1 percent of residents below 150% of poverty."}
::lens{scenario=gpu-dataframe.column-quantiles rung=formula bind=page datum=0}
::lens{scenario=gpu-dataframe.column-quantiles rung=kernel caption="Look at the interpolation: rank h, its fractional part, then the two neighbouring order statistics."}
::lens{scenario=gpu-dataframe.column-quantiles rung=graph caption="Four radix passes, each a histogram plus a select node, then a finish node and the filter mask."}</p>
<p>{/* parameters: GPUColumnQuantiles generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h4 id="gpucolumnquantiles-parameters"><code>GPUColumnQuantiles</code> parameters</h4>
<ul>
<li>Packer: <code>getGPUColumnQuantilesParameterValues(settings)</code>, writes <code>getGPUColumnQuantilesParameterLength(...)</code> <code>float32</code> values.</li>
</ul>
<p>Settings of <code>getGPUColumnQuantilesParameterValues</code>:</p>
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
<td><code>quantiles</code></td>
<td><code>ArrayLike&#x3C;number></code></td>
<td>Probabilities in <code>[0, 1]</code>; others (and NaN) produce a NaN quantile.</td>
</tr>
<tr>
<td><code>interpolation</code> (optional)</td>
<td><code>GPUColumnQuantileInterpolation</code></td>
<td>Defaults to <code>'linear'</code>.</td>
</tr>
<tr>
<td><code>filterRange</code> (optional)</td>
<td><code>[number, number]</code></td>
<td>Percentile filter as fractions <code>[lower, upper]</code> in <code>[0, 1]</code>. Defaults to <code>[0, 1]</code>.</td>
</tr>
<tr>
<td><code>quantileCount</code> (optional)</td>
<td><code>number</code></td>
<td>Number of probability slots. Defaults to <code>quantiles.length</code>. Unused slots become NaN.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h2 id="gpucolumnprofile"><code>GPUColumnProfile</code></h2>
<p>Chunked GraphVectorView columns support up to 4,294,967,295 logical rows, subject to device
bindings. Counts are uint32; float32 statistical fields remain rounded.</p>
<p>Profile several columns at once: counts, nulls, extent, moments, histogram and distinct estimate.</p>
<p>::lens{scenario=gpu-dataframe.column-profile rung=result caption="Median household income has 19 suppressed tracts; the three rate columns have 3 each (CPU check)."}</p>
<p>A dataset statistics panel (kepler.gl / Foursquare Studio field summaries) for several columns of
one table, recomputed on the GPU every frame so it follows filters and brushes without readback
of the rows.</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUColumnProfile</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  columns: [</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    {values: fare}, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// float32, numeric</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    {values: vendorCodes, kind: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'category'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, categoryCount: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">500</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">} </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// uint32 dictionary codes</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  ],</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  mask, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// optional uint32, shared by every column</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  parameters: domainParameters.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">importToGraph</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(graph), </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// optional histogram domains</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  histogramBinCount: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">64</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  hyperLogLogPrecision: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">12</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// 2^12 registers per column</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  topCategoryCount: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">10</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  output: {statistics, counts, histograms, hyperLogLogRegisters, topCategories, topCategoryCounts}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">domainParameters.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">write</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">getGPUColumnProfileParameterValues</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">([[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">100</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">], [</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">NaN</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">NaN</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">]])); </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// NaN = auto</span></span></code></pre>
<ul>
<li><code>statistics</code> holds <code>GPU_COLUMN_PROFILE_STATISTIC_COUNT</code> (11) floats per column, indexed by
<code>GPU_COLUMN_PROFILE_STATISTIC</code>: count, nullCount, minimum, maximum, sum, mean, variance
(population), sampleVariance, standardDeviation, distinctEstimate, overflowCount. <code>counts</code>
holds the exact <code>uint32</code> count and null count per column.</li>
<li>NaN (numeric) or <code>0xffffffff</code> (category) rows are nulls. ±Infinity counts toward count, minimum
and maximum but is excluded from the moments. With no rows, the float statistics are NaN and the
distinct estimate is 0. Category columns report their smallest and largest code and NaN moments.</li>
<li>Moments are a fixed-order two-level Chan/Welford reduction (workgroup trees, then one
fixed-order merge), with no float atomics: bitwise identical across runs and stable for large
offsets (values near 1e6 with a spread of 100).</li>
<li>Histograms: equal-width bins over a per-frame <code>[minimum, maximum]</code> per column (a
<code>GPUColumnProfileDomain</code>: a pair or <code>{minimum, maximum}</code>), or the frame's finite minimum and
maximum when a bound is NaN; the last bin includes the maximum and values outside are not counted.
Bin assignment has the convention of the official <code>GPUHistogram</code>,
<code>min(floor((v - minimum) / (maximum - minimum) * bins), bins - 1)</code>, so a value on a bin edge is
in the upper bin for every bin count (-0.875 is in bin 5 of 30 over <code>[-3.5, 12.25]</code>). The bin
is decided by correctly rounded float32 products
(<code>k * range &#x3C;= (v - minimum) * bins &#x3C; (k + 1) * range</code>), not by one division, so it is identical
on every adapter. Category columns get per-code counts in their first bins.</li>
<li>Distinct count: HyperLogLog over a murmur3 <code>fmix32</code> hash of the value bits (<code>-0</code> folded to
<code>+0</code>) with <code>atomicMax</code> registers and linear-counting small-range correction. Registers are
exact, and the estimate is within about <code>1.04 / sqrt(2^p)</code> relative error.</li>
<li>Top categories: exact per-code counts with <code>uint32</code> atomics, then the top <code>topCategoryCount</code>
ordered by count (descending) and code (ascending). Codes at or above <code>categoryCount</code> are
counted in <code>overflowCount</code>.</li>
<li>At most 16 columns and 4,294,967,295 logical rows (within device limits), and at most <code>getGPUColumnProfileMaximumCategoryTotal(device)</code> (2^25 on a default device)
<code>categoryCount</code> summed over the category columns (one 4-byte slot per category in a 128 MiB binding).
Inputs must be single packed views. The per-column kernels read
the current column from a GPU cursor word, so the pipeline count does not grow with the column
count.</li>
</ul>
<h4 id="how-it-computes-6">How it computes</h4>
<p>{/* lens:GPUColumnProfile */}</p>
<p>::lens{scenario=gpu-dataframe.column-profile rung=datum bind=page datum=3 caption="Column 3, median household income: 772 non-null tracts, 19 nulls."}
::lens{scenario=gpu-dataframe.column-profile rung=formula bind=page datum=3}
::lens{scenario=gpu-dataframe.column-profile rung=kernel caption="Look at the Welford and Chan merges: moments never come from raw squared sums."}
::lens{scenario=gpu-dataframe.column-profile rung=graph caption="Per-column moment tiles, a merge, domain, histogram and HyperLogLog nodes, then a finish node."}</p>
<p>{/* parameters: GPUColumnProfile generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h3 id="gpucolumnprofile-parameters"><code>GPUColumnProfile</code> parameters</h3>
<ul>
<li>Packer: <code>getGPUColumnProfileParameterValues(domains, columnCount?)</code>, writes <code>getGPUColumnProfileParameterLength(...)</code> <code>float32</code> values.
<ul>
<li><code>domains</code> (<code>readonly (GPUColumnProfileDomain | undefined)[]</code>): One entry per column, <code>undefined</code> for fully automatic.</li>
<li><code>columnCount</code> (<code>number</code>): Number of columns. Defaults to <code>domains.length</code>.</li>
</ul>
</li>
</ul>
<p>{/* /parameters */}</p>
<h2 id="gpugroupstatistics"><code>GPUGroupStatistics</code></h2>
<p>Reducible statistics use tiled scratch controlled by scratchTileSize. Compact tiled padded rows are
at most 4,294,967,295; packed output and exact order-statistic scratch remain device-binding
bounded.</p>
<p>Groups rows by a key and computes count, mean, variance and median of a value column per group, ascending by key.</p>
<p>::lens{scenario=gpu-dataframe.group-statistics rung=result caption="Mean tract poverty runs from 7.4 percent in community area 12 to 70.2 percent in area 37 across 77 areas (CPU check)."}</p>
<p>Groups rows by a 32- or 64-bit key and computes kepler-style statistics per group and value column on the GPU: count, sum, mean, minimum, maximum, variance, standard deviation, skewness, kurtosis, median, percentiles, mode, unique count and a per-row group z-score. Use it for "aggregate this column by that key" panels, choropleth values keyed by a region id or an H3 or Quadbin cell, and per-group outlier styling.</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> fractions</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUParameterBuffer</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(device, {id: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'fractions'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, format: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'float32'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, length: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">3</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">});</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">  new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUGroupStatistics</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    keys, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// uint32, or uint32x2 (low, high) words; all-ones is "no key"</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    mask, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// optional uint32</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    columns: [</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">      {</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">        values, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// float32 per row; non-finite rows are excluded for this column only</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">        statistics: [</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'count'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'mean'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'variance'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'median'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'percentiles'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'mode'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'zScore'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">],</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">        output: {counts, means, variances, medians, percentiles, modes, zScores}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">      }</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    ],</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    variance: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'sample'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// or 'population'</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    percentiles: fractions.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">importToGraph</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(graph),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    output: {keys: groupKeys, counts: groupCounts, status: {count, overflow, requiredCount}} </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// GPUGroupStatisticsOutput</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  })</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">);</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">fractions.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">write</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(Float32Array.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">of</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">0.1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">0.5</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">0.9</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">)); </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// no recompile</span></span></code></pre>
<ul>
<li>Output groups are ascending by key, capacity is <code>output.keys.length</code>, at most <code>getGPUGroupStatisticsMaximumGroupCount(device)</code> (2^23 on a default device; the 16-byte moment sums per group fit one 128 MiB binding) and, with <code>percentiles</code>, <code>capacity * P</code> at most <code>getGPUGroupStatisticsMaximumPercentileTotal(device)</code> (2^25 on a default device). When more groups exist, the smallest keys are kept, <code>status.overflow</code> is 1 and <code>status.requiredCount</code> is the unclamped number. Tail rows hold the all-ones key, count 0, sums 0 and NaN for every other float statistic.</li>
<li><code>output.counts</code> counts valid rows (mask set, key valid); each column's <code>counts</code> counts finite values. Zero to four columns; each column lists the statistics it needs (a compile-time set) and supplies exactly the matching output views: <code>counts</code>, <code>sums</code> (signed 64-bit fixed point, <code>uint32x2</code>, the sum of <code>roundHalfEven(value * sumScale)</code>, default scale 65536: each value is rounded to <code>1 / sumScale</code> first) and/or <code>sumValues</code> (the sum of the values themselves as float32, from a 96-bit fixed-point accumulator whose unit follows the largest value of the column; it does not depend on <code>sumScale</code>) for <code>sum</code>, <code>means</code>, <code>minimums</code>, <code>maximums</code>, <code>variances</code>, <code>standardDeviations</code>, <code>skewness</code>, <code>kurtosis</code>, <code>medians</code>, <code>percentiles</code> (<code>capacity * P</code>, group-major), <code>modes</code>, <code>uniqueCounts</code>, and <code>zScores</code> (one row per source row).</li>
<li>Exactness: counts, the fixed-point <code>sums</code> words, minimum, maximum, median, percentiles, mode and unique counts are bit-identical to a CPU reference. <code>sumValues</code> is the exact sum rounded once to float32 and <code>means</code> is that divided by the count (a few float32 units in the last place), at any magnitude: values 1.0e-4, 1.2e-4, 1.4e-4 have mean 1.2e-4. A value keeps all its bits in the sum while its magnitude is at least <code>max * 2^(h - 71)</code>, where <code>max</code> is the largest value of the column in a kept group and <code>h</code> the bit length of the row count. Percentiles use the numpy <code>linear</code> rule (<code>h = (n - 1) p</code>, <code>lo + (hi - lo) * (h - floor h)</code> in f32) and the order-statistic policy shared with <code>GPUColumnQuantiles</code>, with <code>p</code> read per frame; a fraction that is NaN or outside [0, 1] gives NaN (it is no longer clamped). Mode is the longest run of equal values (ties: smallest value). <code>-0</code> is treated as <code>+0</code>; NaN and infinities are excluded.</li>
<li>Moments are corrected two-pass: <code>d = v - mean</code> is summed in a fixed order (a binary tree over sorted row blocks), so a second encode is bitwise identical. The signed first and third powers are scaled by a power of two chosen from each group's extremes before reduction and restored afterwards, preventing same-sign branches from overflowing before opposite signs cancel. Values with binary exponent at most 30 use no scaling. Even powers retain their float32 range limits. Each level merges only the group crossing each block midpoint; the work is O(rows), the depth is O(log rows), and a dominant group spreads across workgroups. The four power sums use separate scalar scratch bindings (16 bytes per input row in total); each binding must fit the device storage limit. The row count is also bounded by the published sort maximum of 2^31. <code>estimatedCommandNodeCount</code> includes the merge levels and conservative allowances for published sorts, scans and gathers, checked against the shared command-node budget. Central moments about the true mean come from the shifted power sums. Oracle specs require relative tolerances of 1e-6 for means, 2e-5 for variance, and 1e-4 for skewness, kurtosis and z-scores, with small absolute tolerances near zero. Skewness is Fisher-Pearson <code>g1</code>, kurtosis is excess <code>g2</code>; both are NaN for constant groups, a group whose variance is under 2^-19 of its shifted sum of squares is treated as constant (the shift mean is rounded to float32). Sample variance of one value is NaN; the z-score is <code>(v - mean) / sd</code> with the same corrected mean the variance uses, and is 0 when the deviation is 0 or a sample group has one value, NaN for masked, invalid-key, non-finite or capacity-dropped rows.</li>
<li>Dense mode: <code>keyCount</code> (compile-time) requires <code>uint32</code> keys in <code>[0, keyCount)</code>. <code>output.keys.length</code> and every per-group output hold exactly <code>keyCount</code> rows, and row <code>k</code> is key <code>k</code> (<code>output.keys[k] = k</code>, <code>output.status.count</code> and <code>status.requiredCount</code> are <code>keyCount</code>, <code>status.overflow</code> is 0). Keys with no rows keep a row: <code>counts</code> 0, <code>sums</code> and <code>uniqueCounts</code> 0, and NaN for mean, minimum, maximum, variance, standard deviation, skewness, kurtosis, median, percentiles and mode. Keys outside the range and the reserved all-ones key are skipped like masked rows and are not reported as overflow. It costs one binary search per key instead of a group scan, and suits region IDs or other small dense key spaces (the choropleth recipe uses it so empty polygons keep a row). The default compact table of occupied keys is unchanged.</li>
<li>Cost: one stable key sort (two 32-bit radix sorts for 64-bit keys); each column that asks for median, percentiles, mode or unique count adds a value sort plus a key re-sort. Chunked reducible inputs use fixed tile scratch; exact order statistics still require device-bound packed scratch. Approximate distinct counts (HyperLogLog) are not included.</li>
</ul>
<h4 id="how-it-computes-7">How it computes</h4>
<p>{/* lens:GPUGroupStatistics */}</p>
<p>::lens{scenario=gpu-dataframe.group-statistics rung=datum bind=page datum=34 caption="Community area 35 has 10 tracts, mean poverty 39.3 percent and the widest sample variance, 357.6 (CPU check)."}
::lens{scenario=gpu-dataframe.group-statistics rung=formula bind=page datum=34}
::lens{scenario=gpu-dataframe.group-statistics rung=kernel caption="Look at the corrected two-pass moments: deviations from the fixed-point mean are summed before the variance divides by n - 1."}
::lens{scenario=gpu-dataframe.group-statistics rung=graph caption="A radix sort of rows by key, group ranges from head flags and a scan, then the moment and order-statistic nodes."}</p>
<h2 id="gpukeyjoin"><code>GPUKeyJoin</code></h2>
<p>Compact <code>rows.status</code> destinations, including optional <code>candidateOverflow</code>, are writable outputs:
they must not overlap other outputs or share buffers with input keys. Candidate status clears to
zero for this exact join; final overflow reports only compact capacity.</p>
<p><code>count-distinct</code> counts distinct finite values among matching right rows; missing and nonfinite
values do not contribute. Right tables are bounded by 2,147,483,648 rows, subject to sort and device
limits. Left-row and output ordering remains unchanged.</p>
<p>Looks up each left row's key among unique right keys and gathers the matching right row's value, entirely on the GPU.</p>
<p>::lens{scenario=gpu-dataframe.key-join rung=result caption="1,499 of 1,500 observations pick up their tract's poverty; the one in tract 789 has no estimate and stays unmatched (CPU check)."}</p>
<p>Attribute join of a left table onto a right table by key, for example "join CSV statistics onto
boundaries". Keys are <code>uint32</code> or 64-bit <code>uint32x2</code> little-endian <code>(low, high)</code> words (the same
format on both sides). The all-ones key (<code>0xffffffff</code>, or both words all-ones) means "no key" and
never matches; masked rows (<code>mask == 0</code>) never match.</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> join</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUKeyJoin</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  leftKeys, rightKeys,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  kind: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'left'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// or 'inner'</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  gather: [{column: rightPopulation, output: leftPopulation}],</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  aggregates: [{operation: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'sum'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, column: rightIncome, output: leftIncomeSum}],</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  output: {matched, rightRows, matchCounts, rightMatched}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">});</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(join);</span></span></code></pre>
<p>Valid right rows are sorted by key with a stable radix sort (one pass per key word), so equal keys
stay in right-row order. Segment heads and a scan produce the unique right keys with row ranges;
each left row binary-searches them on the unsigned <code>(high, low)</code> order.</p>
<ul>
<li><code>gather</code> copies a right column (<code>float32</code> or <code>uint32</code>, bit for bit) from the smallest matching
right row. Unmatched rows get NaN or <code>0xffffffff</code>.</li>
<li><code>aggregates</code> run over all matching right rows: <code>count</code>, <code>count-distinct</code>, <code>sum</code>, <code>mean</code>, <code>minimum</code>, <code>maximum</code>.
<code>sum</code> is the exact sum of the finite values in a 96-bit fixed-point accumulator whose unit
follows the largest value among the valid right rows, rounded once to float32; <code>mean</code> divides
it by the finite count. A column of 1e-4 and a column of 1e6 are both right to float32. The
optional <code>sums</code> words are a separate 64-bit sum of <code>roundHalfEven(value * sumScale)</code> (default
65536): each value is rounded to <code>1 / sumScale</code> first, so a value below half a unit adds
nothing there. Extremes use ordered-key atomics. Non-finite values are excluded per aggregate.
Unmatched rows get 0 for <code>count</code> and <code>count-distinct</code>, and NaN otherwise.</li>
<li><code>output.rightRows</code>, <code>matchCounts</code>, <code>matched</code> (usable as a crossfilter live mask) are left-aligned;
<code>rightMatched</code> is right-aligned (1 when some valid left row has the same key).</li>
<li><code>kind: 'inner'</code> additionally compacts matched left row IDs, ascending, into <code>output.rows</code>
(<code>status.count</code> clamped, <code>status.overflow</code>, optional <code>status.requiredCount</code>; ids past <code>status.count</code> are <code>0xffffffff</code>). Other
outputs stay left-aligned. <code>'left'</code> rejects <code>output.rows</code>.</li>
</ul>
<p>All results are integer or fixed-point, independent of thread order, and bitwise reproducible.
Input contents (keys, masks, values) are per-frame and need no rebuild; view lengths, key format,
<code>kind</code>, <code>sumScale</code> and the set of outputs are topology.</p>
<h4 id="how-it-computes-8">How it computes</h4>
<p>{/* lens:GPUKeyJoin */}</p>
<p>::lens{scenario=gpu-dataframe.key-join rung=datum bind=page datum=0 caption="Observation 0 is in tract 46, which matches right row 46 and gathers poverty 9.7 percent (CPU check)."}
::lens{scenario=gpu-dataframe.key-join rung=formula bind=page datum=0}
::lens{scenario=gpu-dataframe.key-join rung=kernel caption="Look at the binary search of each left key over the unique right keys, in unsigned (high, low) order."}
::lens{scenario=gpu-dataframe.key-join rung=graph caption="A radix sort of the valid right keys, segment heads and a scan, then the per-left-row lookup and gather."}</p>
<h2 id="gpuvariogram-gpuspatialcorrelogram-gpuripley-gpuripleydistancefunctions-and-gpupointpatternindices"><code>GPUVariogram</code>, <code>GPUSpatialCorrelogram</code>, <code>GPURipley</code>, <code>GPURipleyDistanceFunctions</code>, and <code>GPUPointPatternIndices</code></h2>
<p>Every contributor here accepts at most <code>getGPUPairStatisticsMaximumRowCount(device)</code> (11,184,810 on a default device, never above 2^24 - 1) <code>positions</code>
rows: the cell-sorted points table holds 12 bytes per row and must fit one 128 MiB storage binding.</p>
<p>Pair statistics over planar points share one deterministic pair-histogram pass: rows are sorted
into a per-frame lattice whose cells are at least the per-frame <code>maximumDistance</code> wide, one thread
per included row visits the 3x3 cell neighbourhood, and each pair adds integer amounts into
per-workgroup shared-memory 64-bit accumulators that are then added into global 64-bit
accumulators. Float terms are fixed point (a bounded term such as a distance maps its maximum to
2^24; the variogram and correlogram unit comes from the typical term, see below, never from the
maximum), so every result is exact or has a documented fixed-point bound, independent of
scheduling, and bitwise reproducible. A 64-bit sum that would wrap raises an <code>overflow</code> word
(<code>readPairOverflow</code>) instead of wrapping silently, and pair counts saturate at <code>0xffffffff</code>.
When <code>maximumDistance</code> covers the extent the pass is a plain all-pairs loop. A row is included when
its mask is nonzero, its value (if any) finite and its position finite and inside the per-frame
<code>bounds</code>. Parameters (<code>[minX, minY, maxX, maxY, maximumDistance, ...]</code>) are per-frame, packed from a
<code>GPU&#x3C;Class>Settings</code> object (<code>GPUVariogramSettings</code>, <code>GPUSpatialCorrelogramSettings</code>,
<code>GPURipleySettings</code>, <code>GPURipleyDistanceFunctionsSettings</code>, <code>GPUPointPatternIndicesSettings</code>) by
<code>getGPU&#x3C;Class>ParameterValues(settings)</code>; bin counts and the set of outputs are compile-time. Every
caller-owned output view goes in one <code>output</code> prop (<code>GPUVariogramOutput</code>,
<code>GPUSpatialCorrelogramOutput</code>, <code>GPURipleyOutput</code>, <code>GPURipleyEnvelopeOutput</code>,
<code>GPURipleyDistanceFunctionsOutput</code>, <code>GPUPointPatternIndicesOutput</code>), including the <code>overflow</code>
status word.</p>
<p>All five classes (and <code>GPURipleyEnvelope</code>) take an optional <code>spatialContext</code> and are planar only:
distances and window areas are Euclidean in the units of <code>positions</code>. Omitting it means native
planar coordinates; a <code>longitude-latitude</code> context throws at construction ("project positions to a
planar frame first"), because degrees give latitude-dependent distances, wrong areas and biased
indices. The resolved context is exposed as <code>spatialContext</code>. By default the window is the <code>bounds</code>
rectangle. <code>GPURipley</code>, <code>GPUVariogram</code> and <code>GPUSpatialCorrelogram</code> also accept a compile-time
<code>studyWindow</code> prepared by <code>GPUStudyWindowPreparation</code> from
<code>@luma-contributors/analysis/gpu-spatial-analysis</code>. Add the preparation to the same graph before the consumer and use
<code>bounds: 'study-window'</code> (<code>GPUPairStatisticsBounds</code>) in the per-frame settings. The GPU copies the
parameters and replaces the bounds with the prepared polygon's bbox; the original buffer is unchanged.
The window and points must share a planar spatial context. Windows may contain holes and disjoint
components, but must be valid, without self-intersections or overlapping polygons; consider
<code>GPUGeometryValidity</code> before preparation. <code>studyWindow</code> exposes the resolved handle or <code>'rectangle'</code>.
Passing <code>bounds: 'study-window'</code> to a rectangle-mode operator produces NaN statistics; unusable polygon
windows exclude every row. For an irregular study area, using its bbox as the observation window
instead of the polygon overstates the area and can make a pattern look clustered.
<code>GPUVariogram</code> and <code>GPUSpatialCorrelogram</code> measure distance and reject <code>local-lambert-equal-area</code> frames.
Ripley statistics (including distance functions and envelopes) and point-pattern indices combine distance
and window area, so neither local frame is exact: use an azimuthal-equidistant frame only where its area
distortion is negligible, or use a projected CRS.</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">  new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUVariogram</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    positions, values, parameters, gridSize: [</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">64</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">64</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">], lagCount: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">24</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, directionCount: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">4</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    output: {semivariances, pairCounts, meanDistances}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  })</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">);</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">parameters.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">write</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">getGPUVariogramParameterValues</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({bounds, maximumDistance: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">500</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, azimuthOffset: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// After readback: CPU weighted least squares.</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> model</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> fitVariogramModel</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({distances, semivariances, pairCounts}, {model: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'spherical'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">});</span></span></code></pre>
<ul>
<li><code>GPUVariogram</code>: polygon windows filter rows only; no area or edge correction is applied.
Matheron semivariance, pair count, mean lag distance and Cressie-Hawkins robust
semivariance per lag and optional azimuth sector (<code>directionCount</code> equal sectors over <code>[0, pi)</code>
from a per-frame <code>azimuthOffset</code>; a pair of coincident points has no direction and is counted
in sector 0). The robust estimator is
<code>(mean |z_i - z_j|^(1/2))^4 / (2 (0.457 + 0.494 / N + 0.045 / N^2))</code>, the Cressie and Hawkins
(1980) correction that scikit-gstat evaluates; the <code>0.045 / N^2</code> term lowers a one-pair bin by
4.7 % against the two-term form. Matheron precision: a term <code>(z_i - z_j)^2 / 2</code> is scaled by
<code>2^24 / u</code> and rounded to an integer, where <code>u</code> is the robust variance <code>(IQR / 1.349)^2</code> of the
included values (exact quartiles), clamped to at most the population variance, so the absolute
error of a bin mean is at most <code>u * 2^-25</code> (never more than <code>variance * 2^-25</code>, itself below the
former <code>(max - min)^2 * 2^-26</code>) plus f32 rounding of the term. The former unit
<code>2^24 / (0.5 * (max - min)^2)</code> rounded every typical term to 0 once one outlier or sentinel
stretched the range (-83% at an isolated outlier of 2e4 standard deviations on 4000 points,
-100% at 1e5) with no status. Terms above 256 <code>u</code> (a difference beyond 22 standard deviations)
are added exactly in a second 64-bit channel at <code>2^12</code> units, with relative error at most
<code>2^-21</code>; the worst-case relative error of <code>gamma</code> per bin is <code>max(u * 2^-25 / gamma, 5e-7 + 2^-24)</code>. Capacity is <code>2^40 u</code> for the first channel and <code>2^52 u</code> for the second. The optional
<code>overflow</code> output is 1 when a sum, a clamped term (<code>2^51 u</code>) or a pair count (saturated at
<code>0xffffffff</code>; previously the count wrapped modulo 2^32 and became the <code>N_h</code> weight of the fit)
left its range. The Cressie-Hawkins channel scales with <code>sqrt(max - min)</code>. The pass uses 4
channels per bin (5 with <code>robustSemivariances</code>), so <code>lagCount * directionCount</code> is at most 512
(409 with the robust channel). Model fitting is CPU post-processing: <code>fitVariogramModel</code> runs on
the CPU over the read-back bins, fits spherical, exponential or Gaussian models
(practical-range convention) with Cressie, pair-count or uniform weights, and should be skipped
when <code>output.status.overflow</code> is 1. The Matheron and Cressie-Hawkins bins, pair counts and mean distances are
checked against scikit-gstat 1.0.24 (including an embedded 2e4 sentinel) by the committed
generator <code>test/oracles/dataframe-variogram.py</code>, with tolerances derived from the bounds above.</li>
<li><code>GPUSpatialCorrelogram</code>: polygon windows filter rows only; no area or edge correction is applied.
Global Moran's I per distance band (<code>'cumulative'</code> bands as PySAL
<code>DistanceBand</code> / ArcGIS Incremental Spatial Autocorrelation, or <code>'annulus'</code>), with <code>E[I]</code>,
normality or randomization variance (per-frame), z, p, unordered pair counts, and
<code>[firstPeakBand, maximumBand]</code>. Variances lost to f32 cancellation (for example a band holding
every pair) are NaN. Pair products use the same typical-term unit as the variogram, <code>2^24 / u</code>
with <code>u</code> the robust variance (the former <code>2^24 / max|c|^2</code> rounded every product to 0, so <code>I</code> was
0, once one outlier or sentinel dominated), so <code>|I|</code> errs by at most <code>2^-25 * u / variance</code> plus
<code>2^-21</code> of the contribution of products above 256 <code>u</code>, which are summed exactly in a second
channel. The optional <code>overflow</code> output is 1 when a product sum, a clamped product or a band pair
count (saturated at <code>0xffffffff</code>) left its range.</li>
<li><code>GPURipley</code>: K, L, L - r and an annulus pair-correlation estimate over <code>radiusCount</code> radii in a
rectangle or prepared polygon window, with <code>'none'</code>, <code>'border'</code> (reduced sample, spatstat <code>Kest</code>
border estimator
with <code>lambda = n / A</code>, so <code>K = A * numerator / (n * borderCount)</code>) or <code>'isotropic'</code> (Ripley 1977
weights capped at 100 like spatstat <code>edge.Ripley</code> <code>maxweight</code>) edge correction
switchable per frame. Border <code>K</code> changed from <code>lambda = (n - 1) / A</code> to <code>n / A</code> to match spatstat,
which lowers it by the factor <code>(n - 1) / n</code>. The isotropic rectangle formula is exact at every
radius (checked against brute-force circle sampling), so there is no radius guard; the practical
limits are variance and the weight cap, which biases <code>K</code> low once less than 1% of a circle is
inside, so keep <code>maximumDistance</code> well below the shorter window side (spatstat's default <code>rmax</code> is
a quarter of it). Polygon mode uses the polygon area, the distance to its boundary for border
correction, and an oriented-edge fan sum for the isotropic circle fraction. Interior circles use
weight 1 without an edge scan. The same weight cap of 100 applies. The pair correlation is an
unsmoothed annulus difference and is noisy.</li>
<li><code>GPURipleyEnvelope</code>: <code>studyWindow</code> defaults to <code>'rectangle'</code>; pass a valid, non-overlapping
polygon prepared by <code>GPUStudyWindowPreparation</code> in the same planar frame and pack
<code>bounds: 'study-window'</code> to use its bbox, inclusion rule and area. The observed pass and every
simulation must share this window. Monte Carlo CSR envelopes of K, L or <code>L - r</code> (<code>statistic</code>, default
<code>'l-minus-r'</code>), with <code>simulationCount</code> (at most 199) binomial CSR patterns of the observed
included count <code>n</code>, drawn with Philox from the per-frame <code>seed</code> (<code>getGPURipleyParameterValues({.., seed})</code>, a uint32 stored as a word) and measured by a full <code>GPURipley</code> pass each, with the same window,
radii and edge correction, without CPU readback. Outputs (<code>output</code>): <code>lower</code> / <code>upper</code> (pointwise: the
<code>rank</code>-th smallest / largest simulated value, spatstat <code>envelope(nrank)</code>, not simultaneous),
<code>globalLower</code> / <code>globalUpper</code> (Ripley 1981; Baddeley et al. 2014: <code>L(r) = r</code> plus or minus the
<code>rank</code>-th largest <code>max_r |L_j(r) - r|</code>, a simultaneous test of level <code>rank / (simulationCount + 1)</code>),
<code>theoretical</code>, and <code>statistics</code> <code>[n, validSimulationCount, observedMaximumDeviation, criticalMaximumDeviation, pValue, rank]</code> with the maximum absolute deviation (MAD) Monte Carlo
p-value <code>(1 + #{T_j >= T_obs}) / (s + 1)</code> when <code>observedK</code> (an input, the observed <code>GPURipley</code> <code>output.k</code>) is given.
Cost is <code>simulationCount + 1</code> times one K pass and about 25 graph nodes per simulation; the graph
aliases transient buffers across simulations. The simulation index is a GPU counter word, not an
override constant, so every simulation shares the same pipelines (an override still compiles one
pipeline per value). Radii where an estimate is NaN are skipped per
curve, so prefer <code>'isotropic'</code> or <code>'none'</code> for exact Monte Carlo validity. Cross-type K and
inhomogeneous K (Kinhom) are not implemented. Polygon simulations uniformly choose an equal-area
sampler cell, draw a point within it, and reject points outside boundary cells; each rejection
restarts from the cell draw. Philox uses <code>(row, simulationIndex, attempt, 0)</code>. Compile-time
<code>maximumAttempts</code> defaults to 32 and must be an integer in <code>[1, 256]</code>. Failed draws are masked;
optional <code>output.unplacedCount</code> is one uint32 word totaling failures across simulations, cleared
each encoding. An empty sampler or unusable window cannot place points. Undefined curves remain
NaN. Rectangle draws retain their original attempt-0 sequence. Using <code>'study-window'</code> bounds
without a polygon produces NaN statistics.</li>
<li><code>GPURipleyDistanceFunctions</code>: the distance functions next to <code>GPURipley</code> (spatstat <code>Gest</code>, <code>Fest</code>,
<code>Jest</code>). <code>G</code> is the nearest-neighbour distance function of the events, <code>F</code> the empty-space function
over a <code>referenceGrid</code> lattice of reference locations, and <code>J = (1 - G) / (1 - F)</code> (below 1 is
clustered, above 1 regular). <code>parameters</code> (<code>getGPURipleyDistanceFunctionsParameterValues</code>, from a <code>GPURipleyDistanceFunctionsSettings</code>) hold <code>bounds</code>,
<code>maximumDistance</code> and <code>edgeCorrection</code> (<code>'border'</code>, the default, <code>'kaplan-meier'</code>, <code>'hanisch'</code> or
<code>'none'</code>, switchable per frame without recompiling); <code>gridSize</code>, <code>referenceGrid</code> and <code>radiusCount</code> are
compile-time. Outputs (<code>output.g</code>, <code>output.f</code>, <code>output.j</code>, <code>output.radii</code>; at least one of <code>g</code>, <code>f</code>, <code>j</code>) hold NaN where a value is undefined. Integer atomics
keep it bitwise reproducible; the F pass sums each workgroup's reference locations in workgroup
memory and adds one exact 64-bit partial per bin and workgroup. <code>'border'</code> is spatstat <code>rs</code>. <code>'kaplan-meier'</code> is spatstat <code>km</code>, with hazards
binned on the radius grid, exact including the censoring of points that have no neighbor within
<code>maximumDistance</code>. <code>'hanisch'</code> is spatstat <code>han</code> for G and Chiu-Stoyan style weights for F, with 14-bit
fixed-point weights (relative error about 1e-4); a point with no neighbor within <code>maximumDistance</code> but a
border distance of at least <code>maximumDistance</code> enters the denominator at <code>d = maximumDistance</code>. That is
too little weight when the point is uncensored (G too high) and too much when it is censored (G too low:
two such points beside one close pair give 0.44 where the Hanisch estimator is 1). The estimate is exact
when every point has a neighbor in range, so set <code>maximumDistance</code> to at least the largest
nearest-neighbor distance when it matters. The corrections are checked against an f64
oracle written from the spatstat definitions, not against spatstat itself, and K and L (<code>GPURipley</code>) have
no Kaplan-Meier or Hanisch option. The pass uses 8 channels per radius (a <code>radiusCount</code> of 256 still fits
the default 16 KB of workgroup memory exactly). F depends on the reference lattice, so keep it coprime with the
window so no reference location sits exactly on a radius. <code>studyWindow</code> accepts a valid polygon
prepared by <code>GPUStudyWindowPreparation</code> in the same planar frame (default <code>'rectangle'</code>). Pack
<code>bounds: 'study-window'</code>: the effective bbox sizes the search and reference lattices, while event
inclusion and border/Kaplan-Meier censoring use the polygon boundary, including holes. F counts
only reference cell centres inside the polygon. Polygon Hanisch estimates eroded area as
<code>A * insideReferencesWithBorderDistanceAtLeast(r) / insideReferenceCount</code>, using an integer
histogram with <code>4 * radiusCount</code> sub-slots and linear interpolation at each nearest distance.
This pass also runs for G-only output. Refine <code>referenceGrid</code> to improve the area estimate; the
weight cap remains 1024. An empty inside reference set gives NaN F and J. <code>'study-window'</code> bounds
in rectangle mode give NaN statistics; the existing rectangle shaders and draws are unchanged.</li>
<li><code>GPUPointPatternIndices</code>: exact nearest neighbour per row (ring search, ties to the smallest row),
Clark-Evans <code>[n, observed, expected, R, standardError, z, pValue]</code>, and quadrat counts with
<code>[m, mean, variance, varianceToMeanRatio, chiSquare, degreesOfFreedom, pValue]</code>. The per-frame
<code>clarkEvansCorrection</code> defaults to <code>'donnelly'</code> (Donnelly 1978 edge correction for the rectangular
window: <code>expected = 0.5 sqrt(A / n) + (0.0514 + 0.041 / sqrt(n)) P / n</code>, variance
<code>0.0703 A / n^2 + 0.037 P sqrt(A / n^5)</code>, <code>P</code> the perimeter; spatstat uses 0.0412 for the second
constant). The plain infinite-plane index (<code>'none'</code>, <code>0.5 sqrt(A / n)</code>, <code>0.26136 sqrt(A) / n</code>,
pointpats / ArcGIS) overstates clustering in small or elongated windows and is no longer the
default. The per-frame <code>alternative</code> (default <code>'two-sided'</code>; <code>'greater'</code> tests for regularity in
Clark-Evans and for clustering in the quadrat chi-square, <code>'lesser'</code> the opposite) sets the tail
of both p-values. The quadrat p-value is the regularized incomplete gamma function at every
degree of freedom (float32 series or continued fraction; no normal approximation); assume
expected counts of about 5 or more per quadrat. <code>GPU_CLARK_EVANS_LENGTH</code> and
<code>GPU_QUADRAT_STATISTICS_LENGTH</code> are now 7.
Set compile-time <code>studyWindow</code> to a <code>GPUStudyWindowPreparation</code> added before this consumer in the
same graph and planar frame. Polygon windows must be valid (non-overlapping, non-self-intersecting);
holes and disjoint components are supported. Pack <code>bounds: 'study-window'</code> and explicitly choose
<code>clarkEvansCorrection: 'guard'</code> or <code>'none'</code>; <code>'donnelly'</code> and an omitted correction are rejected.
A polygon kernel given code 1 directly writes NaN expected distance, standard error, ratio, z and p.
Packing <code>'study-window'</code> without a polygon window leaves invalid bounds and undefined statistics.
For <code>'guard'</code> (also valid for rectangles), set a positive finite per-frame <code>guardDistance</code> (slot 7).
Only included points at least that far from every boundary contribute to the observed mean;
their neighbors still come from all included points. With <code>n_focus</code> focus points, the expectation
is <code>0.5 sqrt(A / n)</code> and standard error is <code>0.26136 sqrt(A / (n * n_focus))</code>. Row 0 still reports
all <code>n</code> included points; zero focus points give NaN statistics. Recommend
<code>guardDistance >= 2 / sqrt(lambda)</code>, where <code>lambda = n / A</code>.
Polygon quadrats use a compile-time <code>quadratSamples</code> by <code>quadratSamples</code> sub-cell-centre lattice
(default 16 per axis, positive integer, <code>quadratCount * quadratSamples² &#x3C;= 16777216</code>). Sampled
areas are rescaled to sum to the window area <code>A</code>. Zero-area quadrats are dropped; with
<code>e_q = n * a_q / A</code> and <code>df = m_eff - 1</code>, rows are
<code>[m_eff, n/m_eff, sum((c_q-e_q)²)/df, chiSquare/df, chiSquare, df, pValue]</code>, where
<code>chiSquare = sum((c_q-e_q)²/e_q)</code>. Equal areas preserve the existing rectangle formula exactly.
Optional <code>output.quadratAreas</code> has one float32 area per row-major quadrat (plain cell areas for
rectangles) and requires <code>quadratGrid</code>, just like counts and statistics. Lattice accuracy depends
on <code>quadratSamples</code>; increase it for narrow boundary features. Counts and fixed-order sums are
deterministic.</li>
</ul>
<p>Measured on an Apple M3 Pro (headless Chromium), <code>GPUVariogram</code> with 32 lags including the cell
sort: 100k points with 54M pairs in range in 10 ms; 20k points all-pairs (200M pairs) in 12 ms.</p>
<h3 id="gpupointpatternindices"><code>GPUPointPatternIndices</code></h3>
<p>Tests whether points cluster or spread out, from nearest-neighbour distances and a quadrat chi-square over a fixed window.</p>
<p>::lens{scenario=gpu-dataframe.point-pattern-indices rung=result caption="Mean nearest-neighbour distance is 122 m against 497 m expected: ratio 0.25 for 1,500 observations, strongly clustered."}</p>
<h4 id="how-it-computes-9">How it computes</h4>
<p>{/* lens:GPUPointPatternIndices */}</p>
<p>::lens{scenario=gpu-dataframe.point-pattern-indices rung=datum bind=page datum=0 caption="The single Clark-Evans row: 1,500 points, observed 122.4 m, expected 497.3 m (Donnelly), ratio 0.246."}
::lens{scenario=gpu-dataframe.point-pattern-indices rung=formula bind=page datum=0}
::lens{scenario=gpu-dataframe.point-pattern-indices rung=kernel caption="Look at the Donnelly edge term added to 0.5 sqrt(A / n), and the guard and none alternatives beside it."}
::lens{scenario=gpu-dataframe.point-pattern-indices rung=graph caption="A cell sort, nearest-neighbour and guard reductions, then Clark-Evans and quadrat statistics nodes."}</p>
<p>{/* parameters: GPUPointPatternIndices generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h4 id="gpupointpatternindices-parameters"><code>GPUPointPatternIndices</code> parameters</h4>
<ul>
<li>Packer: <code>getGPUPointPatternIndicesParameterValues(settings)</code>, writes <code>GPU_POINT_PATTERN_INDICES_PARAMETER_LENGTH</code> (8) <code>float32</code> values.</li>
</ul>
<p>Settings of <code>getGPUPointPatternIndicesParameterValues</code>:</p>
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
<td><code>GPUPairStatisticsBounds</code></td>
<td>Inclusive <code>[minX, minY, maxX, maxY]</code> window. Rows outside it, or with a non-finite coordinate, are excluded. Its area is the study area of the Clark-Evans index and the quadrats tile it. Use 'study-window' for a prepared polygon; writes NaN bounds until GPU window resolution.</td>
</tr>
<tr>
<td><code>maximumDistance</code></td>
<td><code>number</code></td>
<td>Size hint of the neighbor-search cells (cells are at least this wide). Results never depend on it, only speed: a value near the expected nearest-neighbor distance is fastest.</td>
</tr>
<tr>
<td><code>clarkEvansCorrection</code> (optional)</td>
<td><code>GPUClarkEvansCorrection</code></td>
<td>Clark-Evans edge correction. Defaults to <code>'donnelly'</code> (Donnelly 1978: expected mean distance <code>0.5 sqrt(A / n) + (0.0514 + 0.041 / sqrt(n)) P / n</code>, variance <code>0.0703 A / n^2 + 0.037 P sqrt(A / n^5)</code>, <code>P</code> the window perimeter), because the plain expectation is biased toward apparent clustering for windows with a large perimeter. <code>'none'</code> gives the classic <code>0.5 sqrt(A / n)</code> with standard error <code>0.26136 sqrt(A) / n</code> (pointpats, ArcGIS). Polygon windows require explicit 'guard' or 'none'. 'guard' uses all included neighbors but averages only the interior focus points.</td>
</tr>
<tr>
<td><code>guardDistance</code> (optional)</td>
<td><code>number</code></td>
<td>Positive finite guard width (slot 7); required for 'guard'. Recommend >= 2 / sqrt(n / A).</td>
</tr>
<tr>
<td><code>alternative</code> (optional)</td>
<td><code>GPUInferenceAlternative</code></td>
<td>Tail of the Clark-Evans z-test and the quadrat chi-square test. Defaults to <code>'two-sided'</code>. <code>'greater'</code> tests for a statistic above its complete-spatial-randomness expectation (Clark-Evans: regular; quadrat chi-square: clustered), <code>'lesser'</code> for one below it. The permutation-only tails <code>'folded'</code> and <code>'directed'</code> throw.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h3 id="gpuripley"><code>GPURipley</code></h3>
<p>Estimates Ripley's K and L over radii from planar points with an edge correction, in a rectangular window.</p>
<p>::lens{scenario=gpu-dataframe.ripley rung=result caption="L - r is 2,558 m at 200 m and 3,699 m at 2 km: clustering at every radius."}</p>
<h4 id="how-it-computes-10">How it computes</h4>
<p>{/* lens:GPURipley */}</p>
<p>::lens{scenario=gpu-dataframe.ripley rung=datum bind=page datum=4 caption="Radius 1,000 m: K is 55.7 million m2, so L is 4,210 m and L - r is 3,210 m."}
::lens{scenario=gpu-dataframe.ripley rung=formula bind=page datum=4}
::lens{scenario=gpu-dataframe.ripley rung=kernel caption="Look at the isotropic weight 1 / max(circle fraction, 1/100): pairs near the window edge count more."}
::lens{scenario=gpu-dataframe.ripley rung=graph caption="A cell sort, one pair histogram, then finish and derive nodes for K, L and L - r."}</p>
<p>{/* parameters: GPURipley generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h4 id="gpuripley-parameters"><code>GPURipley</code> parameters</h4>
<ul>
<li>Packer: <code>getGPURipleyParameterValues(settings)</code>, writes <code>GPU_RIPLEY_PARAMETER_LENGTH</code> (8) <code>float32</code> values.</li>
</ul>
<p>Settings of <code>getGPURipleyParameterValues</code>:</p>
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
<td><code>GPUPairStatisticsBounds</code></td>
<td>Inclusive <code>[minX, minY, maxX, maxY]</code> observation window. Rows outside it, or with a non-finite coordinate, are excluded. Its area is the study area of the estimators. Use <code>'study-window'</code> with a compile-time polygon window; it writes NaN bounds slots.</td>
</tr>
<tr>
<td><code>maximumDistance</code></td>
<td><code>number</code></td>
<td>Largest radius: radius <code>b</code> is <code>maximumDistance * (b + 1) / radiusCount</code>. Keep it well below the shorter window side (spatstat's default is a quarter): K is unbiased at any radius, but its variance grows and the isotropic weight cap biases it low once a circle is mostly outside.</td>
</tr>
<tr>
<td><code>edgeCorrection</code> (optional)</td>
<td><code>GPURipleyEdgeCorrection</code></td>
<td>Edge-correction mode. Defaults to <code>'isotropic'</code>.</td>
</tr>
<tr>
<td><code>seed</code> (optional)</td>
<td><code>number</code></td>
<td>Monte Carlo seed of <code>GPURipleyEnvelope</code>, a uint32. <code>GPURipley</code> ignores it. Defaults to 0. The same seed reproduces the same simulated patterns exactly.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h3 id="gpuripleydistancefunctions"><code>GPURipleyDistanceFunctions</code></h3>
<p>Estimates the nearest-neighbour G, empty-space F and J functions over radii from planar points, border-corrected.</p>
<p>::lens{scenario=gpu-dataframe.ripley-distance-functions rung=result caption="G is 0.82 at 200 m while F is 0.05: events are far closer to each other than random locations are to events."}</p>
<h4 id="how-it-computes-11">How it computes</h4>
<p>{/* lens:GPURipleyDistanceFunctions */}</p>
<p>::lens{scenario=gpu-dataframe.ripley-distance-functions rung=datum bind=page datum=0 caption="Radius 200 m: G 0.818, F 0.050, so J = (1 - G) / (1 - F) is 0.19."}
::lens{scenario=gpu-dataframe.ripley-distance-functions rung=formula bind=page datum=0}
::lens{scenario=gpu-dataframe.ripley-distance-functions rung=kernel caption="Look at the border test: only events at least r from the window edge enter the G denominator."}
::lens{scenario=gpu-dataframe.ripley-distance-functions rung=graph caption="A cell sort, separate G and F accumulations (event pairs and reference lattice), then a finish node."}</p>
<p>{/* parameters: GPURipleyDistanceFunctions generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h4 id="gpuripleydistancefunctions-parameters"><code>GPURipleyDistanceFunctions</code> parameters</h4>
<ul>
<li>Packer: <code>getGPURipleyDistanceFunctionsParameterValues(settings)</code>, writes <code>GPU_RIPLEY_DISTANCE_FUNCTIONS_PARAMETER_LENGTH</code> (8) <code>float32</code> values.</li>
</ul>
<p>Settings of <code>getGPURipleyDistanceFunctionsParameterValues</code>:</p>
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
<td><code>GPUPairStatisticsBounds</code></td>
<td>Inclusive <code>[minX, minY, maxX, maxY]</code> observation window. Events outside it, or with a non-finite coordinate, are excluded. Use 'study-window' for a prepared polygon; its bbox supplies the reference lattice. Only reference centres inside the window count. Rectangle mode with these NaN bounds is undefined.</td>
</tr>
<tr>
<td><code>maximumDistance</code></td>
<td><code>number</code></td>
<td>Largest radius: radius <code>b</code> is <code>maximumDistance * (b + 1) / radiusCount</code>.</td>
</tr>
<tr>
<td><code>edgeCorrection</code> (optional)</td>
<td><code>GPURipleyDistanceEdgeCorrection</code></td>
<td><code>'border'</code> (default) divides by the points farther from the window edge than <code>r</code>; <code>'kaplan-meier'</code> and <code>'hanisch'</code> use every point (see <code>GPURipleyDistanceFunctions</code>).</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h3 id="gpuripleyenvelope"><code>GPURipleyEnvelope</code></h3>
<p>Simulates complete spatial randomness with a fixed seed and returns pointwise and global envelopes of L - r with a Monte Carlo p-value.</p>
<p>::lens{scenario=gpu-dataframe.ripley-envelope rung=result caption="Observed L - r reaches 3,699 m at 2 km; 39 seeded simulations never leave plus or minus 41 m."}</p>
<h4 id="how-it-computes-12">How it computes</h4>
<p>{/* lens:GPURipleyEnvelope */}</p>
<p>::lens{scenario=gpu-dataframe.ripley-envelope rung=datum bind=page datum=9 caption="Radius 2,000 m: simulations span -19.7 to 30.2 m, the observed L - r is 3,698.8 m, p is 0.025."}
::lens{scenario=gpu-dataframe.ripley-envelope rung=formula bind=page datum=9}
::lens{scenario=gpu-dataframe.ripley-envelope rung=kernel caption="Look at the lower and upper reductions over finite simulations and the rank-based p-value."}
::lens{scenario=gpu-dataframe.ripley-envelope rung=graph caption="39 seeded simulations, each with its own cell keys and sort, feeding the envelope reduction."}</p>
<p>{/* parameters: GPURipleyEnvelope generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h4 id="gpuripleyenvelope-parameters"><code>GPURipleyEnvelope</code> parameters</h4>
<ul>
<li>Packer: <code>getGPURipleyParameterValues(settings)</code>, writes <code>GPU_RIPLEY_PARAMETER_LENGTH</code> (8) <code>float32</code> values.
<ul>
<li>Input port: <code>parameters</code>.</li>
</ul>
</li>
</ul>
<p>Settings of <code>getGPURipleyParameterValues</code>:</p>
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
<td><code>GPUPairStatisticsBounds</code></td>
<td>Inclusive <code>[minX, minY, maxX, maxY]</code> observation window. Rows outside it, or with a non-finite coordinate, are excluded. Its area is the study area of the estimators. Use <code>'study-window'</code> with a compile-time polygon window; it writes NaN bounds slots.</td>
</tr>
<tr>
<td><code>maximumDistance</code></td>
<td><code>number</code></td>
<td>Largest radius: radius <code>b</code> is <code>maximumDistance * (b + 1) / radiusCount</code>. Keep it well below the shorter window side (spatstat's default is a quarter): K is unbiased at any radius, but its variance grows and the isotropic weight cap biases it low once a circle is mostly outside.</td>
</tr>
<tr>
<td><code>edgeCorrection</code> (optional)</td>
<td><code>GPURipleyEdgeCorrection</code></td>
<td>Edge-correction mode. Defaults to <code>'isotropic'</code>.</td>
</tr>
<tr>
<td><code>seed</code> (optional)</td>
<td><code>number</code></td>
<td>Monte Carlo seed of <code>GPURipleyEnvelope</code>, a uint32. <code>GPURipley</code> ignores it. Defaults to 0. The same seed reproduces the same simulated patterns exactly.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h3 id="gpuspatialcorrelogram"><code>GPUSpatialCorrelogram</code></h3>
<p>Computes Moran's I per cumulative distance band for a station value, with z-scores and p-values under randomization.</p>
<p>::lens{scenario=gpu-dataframe.spatial-correlogram rung=result caption="Moran's I is 0.97 in the first 25 km band and 0.82 by 200 km: rainfall is strongly autocorrelated."}</p>
<h4 id="how-it-computes-13">How it computes</h4>
<p>{/* lens:GPUSpatialCorrelogram */}</p>
<p>::lens{scenario=gpu-dataframe.spatial-correlogram rung=datum bind=page datum=0 caption="First band (up to 25 km): I is 0.971 against E[I] = -0.0008, z is 40.9, from 1,752 pairs."}
::lens{scenario=gpu-dataframe.spatial-correlogram rung=formula bind=page datum=0}
::lens{scenario=gpu-dataframe.spatial-correlogram rung=kernel caption="Look at the band test on pair distance and the product (x_i - mean)(x_j - mean) summed per band."}
::lens{scenario=gpu-dataframe.spatial-correlogram rung=graph caption="Moments, a cell sort, the band pair histogram and a fourth-moment sum, then finish and publish nodes."}</p>
<p>{/* parameters: GPUSpatialCorrelogram generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h4 id="gpuspatialcorrelogram-parameters"><code>GPUSpatialCorrelogram</code> parameters</h4>
<ul>
<li>Packer: <code>getGPUSpatialCorrelogramParameterValues(settings)</code>, writes <code>GPU_SPATIAL_CORRELOGRAM_PARAMETER_LENGTH</code> (8) <code>float32</code> values.</li>
</ul>
<p>Settings of <code>getGPUSpatialCorrelogramParameterValues</code>:</p>
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
<td><code>GPUPairStatisticsBounds</code></td>
<td>Inclusive <code>[minX, minY, maxX, maxY]</code> extent. Rows outside it, or with a non-finite coordinate or value, are excluded. The extent also sizes the neighbor-search lattice. Use <code>'study-window'</code> with a compile-time polygon window; it writes NaN bounds slots.</td>
</tr>
<tr>
<td><code>maximumDistance</code></td>
<td><code>number</code></td>
<td>Upper distance of the last band. Band <code>b</code> ends at <code>maximumDistance * (b + 1) / bandCount</code>.</td>
</tr>
<tr>
<td><code>varianceAssumption</code> (optional)</td>
<td><code>GPUSpatialCorrelogramVarianceAssumption</code></td>
<td>Variance assumption of the z-scores and p-values. Defaults to <code>'randomization'</code>.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h3 id="gpuvariogram"><code>GPUVariogram</code></h3>
<p>Computes the empirical semivariogram of a station value over distance lags, with pair counts and mean pair distances.</p>
<p>::lens{scenario=gpu-dataframe.variogram rung=result caption="Semivariance rises from 546 at the first 25 km lag to 2,567 at 300 km: nearby stations see similar rainfall."}</p>
<h4 id="how-it-computes-14">How it computes</h4>
<p>{/* lens:GPUVariogram */}</p>
<p>::lens{scenario=gpu-dataframe.variogram rung=datum bind=page datum=3 caption="Lag 3 (75 to 100 km): semivariance 962.6 from 6,216 pairs at a mean separation of 87.9 km."}
::lens{scenario=gpu-dataframe.variogram rung=formula bind=page datum=3}
::lens{scenario=gpu-dataframe.variogram rung=kernel caption="Look at the lag index min(floor(d / D * H), H - 1) and the squared value difference."}
::lens{scenario=gpu-dataframe.variogram rung=graph caption="Moments and a unit-variance estimate, a cell sort, the pair-histogram pass and a finish node."}</p>
<p>{/* parameters: GPUVariogram generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h4 id="gpuvariogram-parameters"><code>GPUVariogram</code> parameters</h4>
<ul>
<li>Packer: <code>getGPUVariogramParameterValues(settings)</code>, writes <code>GPU_VARIOGRAM_PARAMETER_LENGTH</code> (8) <code>float32</code> values.</li>
</ul>
<p>Settings of <code>getGPUVariogramParameterValues</code>:</p>
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
<td><code>GPUPairStatisticsBounds</code></td>
<td>Inclusive <code>[minX, minY, maxX, maxY]</code> extent. Rows outside it, or with a non-finite coordinate or value, are excluded. The extent also sizes the neighbor-search lattice. Use <code>'study-window'</code> with a compile-time polygon window; it writes NaN bounds slots.</td>
</tr>
<tr>
<td><code>maximumDistance</code></td>
<td><code>number</code></td>
<td>Largest lag: pairs farther apart are ignored. Lag bins split <code>[0, maximumDistance]</code>.</td>
</tr>
<tr>
<td><code>azimuthOffset</code> (optional)</td>
<td><code>number</code></td>
<td>Start of the first direction sector, in radians counterclockwise from +x. Sector <code>s</code> holds pairs whose undirected direction (taken modulo pi) minus this offset, again modulo pi, lies in <code>[s * pi / directionCount, (s + 1) * pi / directionCount)</code>. Ignored when <code>directionCount</code> is 1. Defaults to 0.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h2 id="gpucompositescore"><code>GPUCompositeScore</code></h2>
<p>Scales indicator columns to [0, 1], flips higher-is-worse directions and combines them into one weighted score per row.</p>
<p>::lens{scenario=gpu-dataframe.composite-score rung=result caption="788 of 791 tracts are scored; scores span 0.03 in tract 657 to 0.79 in tract 603 (CPU check)."}</p>
<p><code>GPUCompositeScore</code> builds a composite indicator per row from up to 16 indicator columns
(row-major <code>row * indicatorCount + column</code>), as CARTO's composite-indicator procedures and ArcGIS
index builders do: scale each column, flip "higher is worse" indicators, and combine them. Weights,
directions, the scaler and the aggregation live in a parameter buffer, so weight sliders recolor
every row without rebuilding the graph.</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> parameters</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUParameterBuffer</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(device, {</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  id: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'composite-parameters'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  format: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'float32'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  length: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">GPU_COMPOSITE_SCORE_PARAMETER_LENGTH</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">});</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">  new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUCompositeScore</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    indicators, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// rowCount * 4 float32</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    indicatorCount: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">4</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    parameters: parameters.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">importToGraph</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(graph),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    enableRank: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">true</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    enablePrincipalComponent: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">true</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    output: {score, columnStatistics, loadings, principalComponentSummary}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  })</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">);</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">parameters.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">write</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">  getGPUCompositeScoreParameterValues</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    scaler: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'rank'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    aggregation: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'weighted-sum'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    weights: [</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">0.5</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">],</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    directions: [</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">-</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">]</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  })</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">);</span></span></code></pre>
<ul>
<li>A row is included when its mask is non-zero and every indicator is finite; excluded rows write
NaN and do not enter the statistics.</li>
<li>Scalers: <code>'min-max'</code> (<code>(x - min) / (max - min)</code>, 0 for a constant column), <code>'z-score'</code>
(population standard deviation, 0 for a constant column), <code>'rank'</code> (percentile rank
<code>r / (n - 1)</code> with tied ranks averaged, exact; needs <code>enableRank</code>). A negative direction flips the
scaled value (<code>1 - s</code>, or <code>-s</code> for z-scores).</li>
<li>Aggregations: <code>'weighted-sum'</code> (<code>sum(w * s) / sum(|w|)</code>), <code>'weighted-geometric-mean'</code>
(<code>exp(sum(w * ln(s + epsilon)) / sum(w))</code>; pair it with min-max or rank; the packer refuses a
negative weight here, use a direction to flip an indicator),
<code>'principal-component'</code> (projection of direction-adjusted z-scores on the first eigenvector of
their correlation matrix; needs <code>enablePrincipalComponent</code>; weights are ignored).</li>
<li>The first component of two indicators is the closed form: <code>(1, 1) / sqrt(2)</code> with eigenvalue
<code>1 + r</code> for a correlation <code>r >= 0</code>, <code>(1, -1) / sqrt(2)</code> with <code>1 - r</code> for <code>r &#x3C; 0</code> (at <code>r = 0</code>
the two tie and <code>(1, 1)</code> is taken). For more indicators it is the last eigenvector of a cyclic
Jacobi decomposition in one thread, which separates close eigenvalues; when the largest
eigenvalue is repeated, any unit vector of its eigenspace is a first component and the one
returned is fixed by the rotation order. No power iteration is used.
The loadings are signed so that they sum to a positive value; when they sum to zero the first
loading that ties the largest magnitude is positive. <code>principalComponentSummary</code> holds
<code>[eigenvalue, explainedVarianceRatio, residual]</code>; the ratio divides by the number of
non-constant columns.</li>
<li>Outputs: <code>score</code> (required), <code>scaled</code>, <code>columnStatistics</code> (<code>[minimum, maximum, mean, standardDeviation]</code> per column),
<code>loadings</code>, <code>principalComponentSummary</code>.</li>
<li>Determinism: min and max use u32 order-key atomics; means and centered (co)variances are
two-pass fixed-order sums over 256-row tiles, merged by one 256-lane workgroup per output
(strided tiles, then a fixed binary tree); ranks use one stable radix
sort per column plus a binary search per tie run. No float atomics.</li>
<li><code>compile-time</code>: row count, <code>indicatorCount</code>, <code>enableRank</code>, <code>enablePrincipalComponent</code>, the mask
and which outputs exist. <code>per frame</code>: scaler, aggregation, weights, directions, epsilon.</li>
</ul>
<h4 id="how-it-computes-15">How it computes</h4>
<p>{/* lens:GPUCompositeScore */}</p>
<p>::lens{scenario=gpu-dataframe.composite-score rung=datum bind=page datum=603 caption="Tract 603 scores highest, 0.79: poverty 67.8, unemployment 37.1, uninsured 24.5 and 62.7 percent without a vehicle (CPU check)."}
::lens{scenario=gpu-dataframe.composite-score rung=formula bind=page datum=603}
::lens{scenario=gpu-dataframe.composite-score rung=kernel caption="Look at the min-max step (x - min) / (max - min), the direction flip and the division by the sum of absolute weights."}
::lens{scenario=gpu-dataframe.composite-score rung=graph caption="Per-column extrema and moment reductions merged in tile order, then a score node reading the parameter buffer."}</p>
<p>{/* parameters: GPUCompositeScore generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h3 id="gpucompositescore-parameters"><code>GPUCompositeScore</code> parameters</h3>
<ul>
<li>Packer: <code>getGPUCompositeScoreParameterValues(settings)</code>, writes <code>GPU_COMPOSITE_SCORE_PARAMETER_LENGTH</code> <code>float32</code> values.</li>
</ul>
<p>Settings of <code>getGPUCompositeScoreParameterValues</code>:</p>
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
<td><code>scaler</code> (optional)</td>
<td><code>GPUCompositeScoreScaler</code></td>
<td>Per-indicator scaler. Defaults to <code>'min-max'</code>.</td>
</tr>
<tr>
<td><code>aggregation</code> (optional)</td>
<td><code>GPUCompositeScoreAggregation</code></td>
<td>How scaled indicators combine into the score. Defaults to <code>'weighted-sum'</code>.</td>
</tr>
<tr>
<td><code>weights</code></td>
<td><code>ArrayLike&#x3C;number></code></td>
<td>One weight per indicator (at most 16). Missing trailing weights are 0. A negative weight penalizes an indicator in <code>'weighted-sum'</code>; <code>'weighted-geometric-mean'</code> refuses negative weights (a weighted geometric mean needs <code>w >= 0</code>): use <code>directions</code> to flip an indicator.</td>
</tr>
<tr>
<td><code>directions</code> (optional)</td>
<td><code>ArrayLike&#x3C;number></code></td>
<td>One direction per indicator: a negative value means "higher is worse" and flips the scaled value (<code>1 - s</code> for min-max and rank, <code>-s</code> for z-scores). Defaults to all <code>+1</code>.</td>
</tr>
<tr>
<td><code>epsilon</code> (optional)</td>
<td><code>number</code></td>
<td>Positive shift added before the logarithm of the geometric mean. Defaults to <code>1e-6</code>.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h2 id="gpuinequality"><code>GPUInequality</code></h2>
<p>Parallel and reference routes support at most 2,147,483,648 rows, subject to sort and device
limits. <code>mode: 'reference'</code> preserves the serial oracle.</p>
<p>Parallel reductions are the default; mode reference retains the serial per-zone oracle. Lorenz knots
and output bindings retain their existing limits.</p>
<p>Computes weighted Gini and other inequality indices per zone from a value column, with Lorenz knots and a pooled summary.</p>
<p>::lens{scenario=gpu-dataframe.inequality rung=result caption="Within community areas the income Gini runs from 0.01 (area 37, two tracts) to 0.31 (area 28); pooled it is 0.33 (CPU check)."}</p>
<p><code>GPUInequality</code> computes per-zone income inequality indices from a column of non-negative values: Gini, Theil T and L, Atkinson, Hoover, Palma, Lorenz curve knots, plus the between/within-zone decomposition of Theil T and pooled statistics.</p>
<p>Inputs: <code>values</code> (float32), <code>zoneIds</code> (uint32; <code>0xffffffff</code> or any id <code>>= zoneCount</code> skips the row), <code>zoneCount</code> (at most <code>getGPUInequalityMaximumZoneCount(device)</code>, 3,050,402 on a default device) and <code>lorenzKnotCount</code> (compile-time; <code>zoneCount * lorenzKnotCount</code> at most <code>getGPUInequalityMaximumLorenzKnotTotal(device)</code>, 2^25 on a default device), optional population <code>weights</code> (float32) and <code>mask</code> (uint32), and a <code>parameters</code> view written with <code>getGPUInequalityParameterValues({epsilon, palmaTopShare, palmaBottomShare})</code>. Changing the parameters never rebuilds or recompiles the graph. A row is included when its mask is non-zero, its zone is valid, its value is finite and non-negative and, with weights, its weight is finite and positive. With <code>w</code> the weight (1 without weights), <code>W = sum w</code>, <code>S = sum w x</code> and <code>mu = S / W</code>:</p>
<table>
<thead>
<tr>
<th>Output</th>
<th>Definition</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>gini</code></td>
<td><code>1 - sum (P_i - P_(i-1)) (L_i + L_(i-1))</code>, trapezoid area under the Lorenz curve of the value-sorted rows; equals <code>2 sum i x_(i) / (n S) - (n + 1) / n</code> for unit weights, <code>(n - 1) / n</code> for one holder of everything</td>
</tr>
<tr>
<td><code>theilT</code></td>
<td><code>(1 / W) sum w r ln r</code>, <code>r = x / mu</code>, <code>0 ln 0 = 0</code></td>
</tr>
<tr>
<td><code>theilL</code></td>
<td><code>-(1 / W) sum w ln r</code>; NaN when the zone contains a zero (zeros are never silently dropped)</td>
</tr>
<tr>
<td><code>atkinson</code></td>
<td><code>1 - ((1 / W) sum w r^(1 - e))^(1 / (1 - e))</code>, <code>1 - exp(mean ln r)</code> at <code>e = 1</code>; zeros are fine for <code>e &#x3C; 1</code>, NaN for <code>e >= 1</code>; <code>e</code> in <code>[0, 32]</code> per frame</td>
</tr>
<tr>
<td><code>hoover</code></td>
<td><code>sum w abs(x - mu) / (2 S)</code></td>
</tr>
<tr>
<td><code>palma</code></td>
<td><code>(1 - L(1 - top)) / L(bottom)</code>, cuts per frame (0.1 and 0.4 by default); NaN when <code>L(bottom) = 0</code></td>
</tr>
<tr>
<td><code>mean</code>, <code>counts</code></td>
<td>weighted mean, included row count per zone (<code>counts</code> is <code>uint32</code>)</td>
</tr>
<tr>
<td><code>lorenzKnots</code></td>
<td><code>L(k / (K - 1))</code>, row-major <code>zone * K + k</code>, <code>L(0) = 0</code>, <code>L(1) = 1</code></td>
</tr>
<tr>
<td><code>globalSummary</code></td>
<td>slots <code>GPU_INEQUALITY_GLOBAL_SUMMARY</code>: <code>TOTAL_THEIL_T</code>, <code>BETWEEN_THEIL_T</code>, <code>WITHIN_THEIL_T</code>, <code>GINI</code>, <code>COUNT</code>, <code>MEAN</code>, <code>TOTAL_WEIGHT</code>, <code>TOTAL_INCOME</code></td>
</tr>
</tbody>
</table>
<p>The Lorenz curve is linear inside each row (a row's income spreads over its population share), so a Palma cut or knot inside a row takes a fractional share of that row. Empty zones, and zones with zero total income, report NaN for every index (zero-income zones keep <code>mean = 0</code>). The Theil decomposition is <code>between = sum_g s_g ln(s_g / p_g)</code> and <code>within = sum_g s_g T_g</code> with income share <code>s_g</code> and population share <code>p_g</code>; the pooled Theil T is computed independently from all included rows, so <code>total = between + within</code> is a real check.</p>
<p>Algorithm and cost: rows are sorted by value, re-keyed by zone and stably sorted, so each zone
is a value-ordered segment. The default parallel route builds bounded 256-row Lorenz prefixes,
computes row-parallel terms and reduces them by segment. <code>mode: 'reference'</code> preserves the two
serial per-zone walks and their rounding order; a dominant zone serializes that route. The
parallel summation order can change the last float32 bits; sums and published statistics remain
float32. Pooled and global reductions use fixed-order tiles.</p>
<h4 id="how-it-computes-16">How it computes</h4>
<p>{/* lens:GPUInequality */}</p>
<p>::lens{scenario=gpu-dataframe.inequality rung=datum bind=page datum=27 caption="Community area 28 has the highest Gini, 0.307, over 20 tracts with a population-weighted mean income of 85,684 dollars (CPU check)."}
::lens{scenario=gpu-dataframe.inequality rung=formula bind=page datum=27}
::lens{scenario=gpu-dataframe.inequality rung=kernel caption="Look at the trapezoid sum 1 - sum (P_i - P_(i-1)) (L_i + L_(i-1)) walked over each zone's value-sorted rows."}
::lens{scenario=gpu-dataframe.inequality rung=graph caption="Two sorts, by value and then by zone, a per-zone walk, and a finish node for the pooled summary."}</p>
<p>{/* parameters: GPUInequality generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h3 id="gpuinequality-parameters"><code>GPUInequality</code> parameters</h3>
<ul>
<li>Packer: <code>getGPUInequalityParameterValues(settings?)</code>, writes <code>GPU_INEQUALITY_PARAMETER_LENGTH</code> (4) <code>float32</code> values.</li>
</ul>
<p>Settings of <code>getGPUInequalityParameterValues</code>:</p>
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
<td><code>epsilon</code> (optional)</td>
<td><code>number</code></td>
<td>Atkinson inequality aversion, <code>0</code> to <code>GPU_INEQUALITY_MAXIMUM_EPSILON</code>. <code>1</code> selects the geometric-mean form. Defaults to 1.</td>
</tr>
<tr>
<td><code>palmaTopShare</code> (optional)</td>
<td><code>number</code></td>
<td>Population share of the Palma numerator, in <code>(0, 1]</code>. Defaults to 0.1.</td>
</tr>
<tr>
<td><code>palmaBottomShare</code> (optional)</td>
<td><code>number</code></td>
<td>Population share of the Palma denominator, in <code>(0, 1]</code>. Defaults to 0.4.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h2 id="gputemporalreduction"><code>GPUTemporalReduction</code></h2>
<p>All <code>occupiedSlots.status</code> destinations, including optional <code>candidateOverflow</code>, are writable
outputs and must not overlap other outputs or share buffers with inputs such as <code>cellIds</code>.</p>
<p>Reduce timestamped rows to per cell and time-bucket counts, extremes, first and last values.</p>
<p>::lens{scenario=gpu-dataframe.temporal-reduction rung=result caption="251 of 530 taxon-week slots are occupied; the busiest holds 36 bird observations in week 18 (CPU check)."}</p>
<p>Reduces timestamped rows into one record per (cell, coarse time bucket) so a scrubbing view draws one feature per occupied slot instead of one per row. Bucket <code>b</code> covers <code>[origin + b * width, origin + (b + 1) * width)</code>; the slot is <code>cell * bucketCount + bucket</code>.</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">  new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUTemporalReduction</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    cellIds, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// uint32 per row, 0xffffffff skips</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    timestamps, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// float32 relative times, or uint32x2 Int64 words</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    values, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// float32 per row</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    mask, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// optional uint32</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    parameters: bucketParameters.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">importToGraph</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(graph), </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// [origin, width]</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    cellCount,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    bucketCount,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    output: {counts, minimums, maximums, first, last, occupiedSlots: {ids, status: {count, overflow}}}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  })</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">);</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">bucketParameters.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">write</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">getGPUTemporalReductionParameterValues</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(origin, width)); </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// no recompile</span></span></code></pre>
<ul>
<li>Outputs are dense columns of <code>cellCount * bucketCount</code> rows. Empty slots have count 0 and NaN minimums, maximums, first and last. <code>occupiedSlots.ids</code> lists occupied slots in ascending order (<code>cell = id / bucketCount</code>, <code>bucket = id % bucketCount</code>), clamped to its length with <code>overflow</code> set when more are occupied.</li>
<li><code>first</code> and <code>last</code> are the values at the earliest and latest timestamp in the slot; ties are broken by the lower row index.</li>
<li>Rows with a bucket outside <code>[0, bucketCount)</code>, a skipped cell, a zero mask, or a NaN value or time are dropped, not clamped.</li>
<li>Exact Int64 times: pass <code>uint32x2</code> words and write <code>getGPUTemporalReductionWordParameterValues(origin, width)</code> (integer origin, integer width in <code>[1, 2^64 - 1]</code>, so hourly buckets of nanosecond data use <code>3_600_000_000_000n</code>). Bucket assignment and the first and last time inside a bucket are exact integer arithmetic; the bucket index must still be below <code>bucketCount</code>.</li>
<li>Determinism: u32 atomics only, so results do not depend on thread order. In float mode bucket edges are defined by correctly rounded products (<code>fround(b * width) &#x3C;= fround(t - origin) &#x3C; fround((b + 1) * width)</code>), so assignment is identical on every adapter; at most 2^20 buckets.</li>
<li>Slot limit: <code>cellCount * bucketCount</code> is at most 33554432 slots with float32 times and 16777216 slots with <code>uint32x2</code> times on a default device (128 MiB bindings). Word mode keeps the in-bucket time of each slot as two 32-bit words (8 bytes per slot instead of 4), so its maximum is half the float one. Larger slot counts are refused at declaration.</li>
<li>Inputs must be single packed views; chunked vectors are not supported.</li>
<li><code>sorted: true</code> (default false, compile-time) declares <code>timestamps</code> non-decreasing and non-NaN. The rows with bucket in <code>[0, bucketCount)</code> are then one contiguous range found by two binary searches (same bucket rules as above: the correctly rounded <code>fround(t - origin) &#x3C; fround(bucketCount * width)</code> test in float mode, the exact quotient in word mode), and the classify, extreme and tie kernels are dispatched indirectly over just that range, with the first row read from a GPU word. A scrub costs O(range), not O(rows), and the result is bit-identical to the unsorted path on sorted input. It adds one <code>uint32</code> scratch column of <code>rows</code> entries. The order is not checked; unsorted input silently reduces the wrong rows.</li>
</ul>
<h4 id="how-it-computes-17">How it computes</h4>
<p>{/* lens:GPUTemporalReduction */}</p>
<p>::lens{scenario=gpu-dataframe.temporal-reduction rung=datum bind=page datum=133 caption="Slot 133 is insects in week 27: 32 observations, research grade mixed (minimum 0, maximum 1; CPU check)."}
::lens{scenario=gpu-dataframe.temporal-reduction rung=formula bind=page datum=133}
::lens{scenario=gpu-dataframe.temporal-reduction rung=kernel caption="Look at the bucket test: fround(b * width) &#x3C;= fround(t - origin) &#x3C; fround((b + 1) * width)."}
::lens{scenario=gpu-dataframe.temporal-reduction rung=graph caption="Init, classify, extreme, row-tie and finish nodes, then compaction and publish of occupied slots."}</p>
<p>{/* parameters: GPUTemporalReduction generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h3 id="gputemporalreduction-parameters"><code>GPUTemporalReduction</code> parameters</h3>
<ul>
<li>Packer: <code>getGPUTemporalReductionParameterValues(origin, width)</code>, writes <code>GPU_TEMPORAL_REDUCTION_PARAMETER_LENGTH</code> (2) <code>float32</code> values.
<ul>
<li><code>origin</code> (<code>number</code>): Start of bucket 0, in the unit of the timestamps.</li>
<li><code>width</code> (<code>number</code>): Bucket width, in the unit of the timestamps.</li>
</ul>
</li>
<li>Packer: <code>getGPUTemporalReductionWordParameterValues(origin, width)</code>, writes <code>GPU_TEMPORAL_REDUCTION_WORD_PARAMETER_LENGTH</code> (4) <code>uint32</code> values.
<ul>
<li><code>origin</code> (<code>number | bigint</code>): Integer start of bucket 0, any signed 64-bit value.</li>
<li><code>width</code> (<code>number | bigint</code>): Integer bucket width in <code>[1, 2^64 - 1]</code>. A <code>number</code> must be a safe integer to be exact; pass a <code>bigint</code> beyond 2^53.</li>
</ul>
</li>
</ul>
<p>{/* /parameters */}</p>
<h2 id="gpucalendarbuckets"><code>GPUCalendarBuckets</code></h2>
<p>Decode epoch timestamps (second, millisecond, microsecond or nanosecond) into calendar columns and a weekday-by-hour count matrix.</p>
<p>::lens{scenario=gpu-dataframe.calendar-buckets rung=result caption="Saturday 11:00 is the busiest weekday-hour bucket with 48 of 1,500 observations (CPU check)."}</p>
<p>Decodes epoch timestamps (milliseconds by default, see <code>unit</code>) into calendar columns (year, month, day of month, hour, minute,
weekday, day of year, ISO week, ISO week-year, quarter) and an hour by weekday count matrix, so
time-of-day and weekday charts and calendar bucket keys need no per-row <code>Date</code> on the CPU. The
columns are plain <code>uint32</code> and <code>sint32</code> outputs for heatmaps and charts. They do not feed
<code>GPUTemporalReduction</code> (which buckets by fixed width) or <code>GPUCellAggregation</code> keys (which accept
only Quadbin or H3 cells).</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> parameters</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUParameterBuffer</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(device, {id: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'calendar'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, format: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'sint32'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, length: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">});</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">  new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUCalendarBuckets</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    timestamps, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// GraphDataView&#x3C;'uint32x2'>, Int64 epoch words, e.g. from getInt64TimeWords</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    unit: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'nanosecond'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// the unit of the column; defaults to 'millisecond'</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    parameters: parameters.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">importToGraph</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(graph),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    output: {hour, weekday, hourWeekdayCounts}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  })</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">);</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">parameters.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">write</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">getGPUCalendarBucketsParameterValues</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">-</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">300</span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D"> /* UTC-5 */</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">0</span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D"> /* Monday first */</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">));</span></span></code></pre>
<ul>
<li><code>timestamps</code>: signed 64-bit two's complement epoch times as <code>uint32x2</code> <code>(low, high)</code> rows;
times before 1970 are fine. Optional <code>mask</code> (<code>uint32</code>, zero makes the row invalid).</li>
<li><code>unit</code> (optional, compile-time): <code>'second'</code>, <code>'millisecond'</code> (default), <code>'microsecond'</code> or
<code>'nanosecond'</code>, the <code>ArrowAnalyticsTimeUnit</code> of the column. Set it to the unit the Arrow upload kept
(a pandas <code>timestamp[ns]</code> column needs <code>'nanosecond'</code>); the wrong unit decodes to wrong dates
without an error. Rows are floored to whole milliseconds first (a negative sub-millisecond time
lands in the earlier millisecond), then decoded as before. Seconds beyond +-2^52 are invalid. The
<code>timeZone</code> transitions stay epoch milliseconds whatever the unit.</li>
<li>Host calendar display: <code>getTimestampCalendarMilliseconds(ticks, arrowUnit)</code> floors exact ticks to
the containing millisecond before conversion to Number, matching this decode and the day rendered by
<code>formatTimestampTicks</code>. Arrow units are <code>0</code> seconds, <code>1</code> milliseconds, <code>2</code> microseconds and <code>3</code>
nanoseconds: <code>-1n</code> in microseconds or nanoseconds returns <code>-1</code> millisecond (1969-12-31), while zero
and positive sub-millisecond ticks return <code>0</code>. <code>getTimestampPresentationMilliseconds</code> retains
truncation toward zero for numeric callers. Neither helper changes source ticks or exact export text.
The calendar helper accepts source instants only within Date's inclusive
<code>[-8640000000000000, 8640000000000000]</code> millisecond range and throws <code>RangeError</code> outside it,
before conversion to Number. Every millisecond in that range is exactly representable. The GPU
decoder supports a wider range; group those ticks into days in <code>bigint</code> first, using floor
division, rather than narrowing milliseconds. For microseconds, for example:
<code>day = ticks >= 0n ? ticks / 86400000000n : -((-ticks + 86399999999n) / 86400000000n)</code>.</li>
<li><code>parameters</code> (<code>sint32</code>, 2 elements, per frame, no rebuild): fixed UTC offset in minutes and the first
day of the week for <code>weekday</code> and the matrix (0 = Monday, the ISO default, to 6 = Sunday). ISO week
numbers always start on Monday.</li>
<li>Time zones: the per-frame offset is a fixed offset. Daylight saving time is supported by either
the optional per-row <code>utcOffsets</code> column (<code>sint32</code> minutes, within +-1440, precomputed per row on
the CPU) or the optional <code>timeZone</code> transition table, which needs no per-row CPU work. They are
mutually exclusive and both override the fixed offset.</li>
<li><code>timeZone: {transitions, offsets}</code>: <code>transitions</code> is a <code>uint32x2</code> view of strictly increasing Int64
epoch-millisecond instants (the format of <code>timestamps</code>), <code>offsets</code> a <code>sint32</code> view of UTC offsets in
seconds with <code>transitions.length + 1</code> elements. <code>offsets[0]</code> applies before the first transition
and <code>offsets[i + 1]</code> from transition <code>i</code> on. Each row binary-searches its instant (signed 64-bit
compare, about <code>log2(n)</code> steps) and uses that offset; seconds keep pre-standardization mean time
such as +00:19:32 exact. A table offset beyond +-86400 s makes the row invalid. The table length
is topology (a table of another length needs a new graph); its contents can change per frame.
<code>getGPUTimeZoneTransitionTable(zone, {start, end})</code> builds the table on the CPU once per zone
and range from <code>Intl.DateTimeFormat</code> (offset sampled daily and bisected to the millisecond; a
zone without a transition in the range gets one sentinel transition at the Int64 minimum;
an unknown zone throws). Instants outside the range use the offset at its start or end. The
helper is as exact as the ICU data in the runtime: Node 22 reproduces Python <code>zoneinfo</code> (tzdata
2026d) transition for transition for New_York, London, Lord_Howe, Kathmandu, Amsterdam and
Dublin over 1900 to 2040, including seconds offsets; it differs only where the databases are
different releases (Africa/Casablanca after 2026-09).</li>
<li>Outputs, each optional but at least one required (compile-time): <code>year</code> and <code>isoWeekYear</code>
(<code>sint32</code>), <code>month</code> (1-12), <code>dayOfMonth</code> (1-31), <code>hour</code>, <code>minute</code>, <code>weekday</code> (0-6),
<code>dayOfYear</code> (1-366), <code>isoWeek</code> (1-53), <code>quarter</code> (1-4) as <code>uint32</code>, and <code>hourWeekdayCounts</code>
(168 <code>uint32</code>, <code>weekday * 24 + hour</code>, cleared every encoding).</li>
<li>Invalid rows (masked, local day beyond +-1e9, Int64 overflow with the offset, per-row offset beyond
+-1440) write <code>0xffffffff</code> to <code>uint32</code> outputs and <code>-2^31</code> to <code>year</code> and <code>isoWeekYear</code>, and are not
counted in the matrix.</li>
<li>Exact integer arithmetic only (floor conversion of the unit to milliseconds, nibble-wise 64-bit division, Hinnant <code>civil_from_days</code>), identical on
every adapter. Outputs are split over several decode kernels to respect the 8 storage binding limit.</li>
</ul>
<h4 id="how-it-computes-18">How it computes</h4>
<p>{/* lens:GPUCalendarBuckets */}</p>
<p>::lens{scenario=gpu-dataframe.calendar-buckets rung=datum bind=page datum=0 caption="Observation 0 is Sunday 2023-01-01 08:05: weekday 6, hour 8, month 1."}
::lens{scenario=gpu-dataframe.calendar-buckets rung=formula bind=page datum=0}
::lens{scenario=gpu-dataframe.calendar-buckets rung=kernel caption="Look at the floor division by 86,400,000 and the Hinnant civil-date arithmetic after it."}
::lens{scenario=gpu-dataframe.calendar-buckets rung=graph caption="Decode kernels split over the storage-binding limit, plus a cleared matrix and its count kernel."}</p>
<p>{/* parameters: GPUCalendarBuckets generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h3 id="gpucalendarbuckets-parameters"><code>GPUCalendarBuckets</code> parameters</h3>
<ul>
<li>Packer: <code>getGPUCalendarBucketsParameterValues(utcOffsetMinutes, firstDayOfWeek?, target?)</code>, writes <code>GPU_CALENDAR_BUCKETS_PARAMETER_LENGTH</code> (2) values.
<ul>
<li><code>utcOffsetMinutes</code> (<code>number</code>): Fixed offset east of UTC in minutes, an integer in <code>[-1440, 1440]</code>.</li>
<li><code>firstDayOfWeek</code> (<code>number</code>): First day of the week for the <code>weekday</code> output and the hour by weekday matrix: 0 = Monday (ISO, default) to 6 = Sunday. ISO week numbers always use Monday.</li>
<li><code>target</code> (<code>Int32Array</code>): Optional destination of at least 2 elements.</li>
</ul>
</li>
</ul>
<p>{/* /parameters */}</p>
<h2 id="gpuperiodindex"><code>GPUPeriodIndex</code></h2>
<p>Turn calendar columns or Int64 word times into a <code>uint32</code> period per row: whole calendar steps from an origin, or a fixed width.</p>
<p>The period index is the adapter between <code>GPUCalendarBuckets</code> (or raw Int64 words) and a (cell, period) cube. It is the code that <code>addSpaceTimeHotSpotsRecipe</code> runs between its calendar decode and its cube, extracted so that other space-time pipelines share it and the recipe's slices are exactly these periods. It reads the temporal context of the times, never converts units, and writes one <code>uint32</code> per row with no readback.</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// Calendar axis: 52 weeks from Monday 2024-01-01, from the calendar columns of the same times.</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">  new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUCalendarBuckets</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({temporalContext, timestamps, parameters, output: {year, month, dayOfMonth}})</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">);</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">  new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUPeriodIndex</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    temporalContext, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// 'int64-words' counted from the Unix epoch</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    columns: {year, month, dayOfMonth},</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    periods: {</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">      kind: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'calendar'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">      step: {calendarUnit: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'week'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">},</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">      origin: {year: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">2024</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, month: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, day: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">},</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">      count: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">52</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    },</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    output: {periods, status: {invalidCount}}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  })</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">);</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// Fixed axis: 24 one-hour bins of nanosecond words from 2024-01-01T00:00:00Z.</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">  new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUPeriodIndex</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    temporalContext: nanosecondContext,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    timestamps,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    periods: {kind: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'fixed'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, width: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">3_600_000_000_000</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">n</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, origin: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">1_704_067_200_000_000_000</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">n</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, count: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">24</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">},</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    output: {periods}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  })</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">);</span></span></code></pre>
<ul>
<li><code>temporalContext</code>: resolved with <code>{encodings: ['int64-words'], unixEpoch: true}</code>, so Arrow uploads pass their own context. An unsupported encoding or an epoch other than the Unix epoch throws at construction. The <code>unit</code> is the unit of a fixed <code>width</code> and <code>origin</code>; a calendar axis only reads the decoded columns, whose unit <code>GPUCalendarBuckets</code> already handled.</li>
<li>Calendar axis: <code>step</code> is <code>{calendarUnit, interval?}</code> with <code>calendarUnit</code> one of <code>'year'</code>, <code>'quarter'</code>, <code>'month'</code>, <code>'week'</code>, <code>'day'</code> or <code>'hour'</code> and a positive integer <code>interval</code> (default 1). Period <code>floor((stepIndex(time) - stepIndex(origin)) / stepsPerPeriod)</code> holds a time, where <code>stepIndex</code> counts whole steps (<code>year</code>, <code>year * 4 + quarter - 1</code>, <code>year * 12 + month - 1</code>, days since 1970 for <code>'day'</code> and <code>'week'</code>, hours since 1970 for <code>'hour'</code>) and a week is 7 days anchored to the origin day. <code>origin</code> is <code>{year, month?, day?, hour?}</code> in the local calendar of <code>columns</code>, with the unset fields at the start of the step. <code>columns.year</code> is always read; <code>columns.month</code> for every step but <code>'year'</code>; <code>columns.dayOfMonth</code> for <code>'week'</code>, <code>'day'</code> and <code>'hour'</code>; <code>columns.hour</code> for <code>'hour'</code>. Days are numbered with Hinnant's <code>days_from_civil</code> in <code>i32</code> arithmetic, so leap days, quarter starts and years before 1970 are exact. The CPU twins <code>getGPUPeriodIndexCalendarPeriod</code>, <code>getGPUPeriodIndexStepIndex</code> and <code>getGPUPeriodIndexDaysFromCivil</code> return the same numbers.</li>
<li>Fixed axis: period <code>floor((time - origin) / width)</code> with <code>width</code> a positive integer below 2^63 (a <code>number</code> must be a safe integer; pass a <code>bigint</code> beyond 2^53) and <code>origin</code> signed Int64 ticks after the Unix epoch. The difference and the division run on the Int64 words, so they are exact for every time, including times before 1970 and beyond 2^53 ticks, and for widths past 32 bits.</li>
<li>Output <code>periods</code> holds one <code>uint32</code> per row: the period, <code>GPU_PERIOD_INDEX_OUT_OF_RANGE</code> (<code>0xfffffffe</code>) for a valid time before the origin or past the last period, or <code>GPU_PERIOD_INDEX_INVALID</code> (<code>0xffffffff</code>) for a masked row or a time the calendar decode rejected. Rows are skipped, never clamped.</li>
<li>Status: <code>output.status.invalidCount</code> (optional) is the number of skipped rows, those with <code>GPU_PERIOD_INDEX_OUT_OF_RANGE</code>. Masked and undecodable rows are not counted. It is cleared on every encoding and counted with an integer <code>atomicAdd</code>.</li>
<li>Limits: <code>periods.count</code> is an integer in <code>[1, GPU_PERIOD_INDEX_MAXIMUM_COUNT]</code> (4294967293, the two sentinels are not periods). Each bound range must fit <code>device.limits.maxStorageBufferBindingSize</code> <strong>per chunk</strong>, including any prefix required for an aligned binding offset. At a 128 MiB binding limit with no prefix, a <code>uint32</code> output or calendar column holds up to 33,554,432 rows per chunk, and Int64 <code>timestamps</code> hold up to 16,777,216 rows per chunk. Larger totals are supported by chunking, up to 4,294,967,295 total rows and subject to the command-node budget.</li>
<li>Row inputs and <code>output.periods</code> accept packed views or chunked <code>GraphVectorView</code>s with matching ordered chunk lengths across every active column, the optional <code>mask</code> and the output. Each chunk is packed; mixing mismatched row partitions is rejected. Status views remain packed. An optional <code>mask</code> (<code>uint32</code>, zero writes the invalid sentinel) must have one row per input row.</li>
</ul>
<p>For example, the fixed-axis chunk fixture uses three rows partitioned as <code>[1, 2]</code>. Given caller-owned
GPU vectors <code>timestampVector</code>, <code>maskVector</code> and <code>periodVector</code> with those same ordered chunk lengths:</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">  new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUPeriodIndex</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    id: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'periods'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    temporalContext, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// millisecond, Unix epoch, int64-words</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    periods: {kind: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'fixed'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, origin: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">0</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">n</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, width: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">1000</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, count: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">12</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">},</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    timestamps: graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">importGPUVector</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'chunked-times'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, timestampVector),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    mask: graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">importGPUVector</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'chunked-mask'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, maskVector),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    output: {</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">      periods: graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">importGPUVector</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'chunked-periods'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, periodVector),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">      status: {invalidCount} </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// one packed uint32 word</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    }</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  })</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">);</span></span></code></pre>
<p>This declares <code>periods-clear-invalid-count</code>, <code>periods-periods-chunk-0</code> and
<code>periods-periods-chunk-1</code>. A combined timestamp length above one binding's row limit is accepted
when each bound chunk range fits the device limit.</p>
<p>Calendar hour multiplication, step differences and division use exact 64-bit words, so indices at
<code>2^31 - 1</code>, <code>2^31</code> and above remain usable up to <code>periods.count - 1</code>. Calendar origins must have a
whole-step index in <code>[-2^31, 2^31 - 1]</code>; steps per period (<code>interval</code>, times seven for weeks) must
be in <code>[1, 2^31 - 1]</code>. These construction-time origin and divisor limits do not cap row indices or
the unsigned period count. Row dates follow the supported <code>GPUCalendarBuckets</code> decoder range.</p>
<h3 id="gpuperiodindex-parameters"><code>GPUPeriodIndex</code> parameters</h3>
<ul>
<li>Per frame (no recompile): the contents of <code>columns</code>, <code>timestamps</code> and <code>mask</code>.</li>
<li>Compile-time (needs a new graph): the row count, <code>periods</code> (<code>kind</code>, <code>step</code>, <code>origin</code>, <code>width</code>, <code>count</code>), which calendar columns are present, and whether <code>mask</code> and <code>status.invalidCount</code> are present. The axis is compile-time because it fixes the size of the cube the periods index.</li>
<li>Packer: none.</li>
</ul>
<h2 id="gpucubelayouttranspose"><code>GPUCubeLayoutTranspose</code></h2>
<p>Move a dense (cell, period) cube between the cell-major and period-major layouts.</p>
<p>The space-time hot spot cube is cell-major (<code>cell * periodCount + period</code>); Markov chains, spatial panels and per-period maps are period-major (<code>period * cellCount + cell</code>). The transpose converts one into the other on the GPU for <code>uint32</code> counts or <code>float32</code> values.</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">  new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUCubeLayoutTranspose</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    values: cube, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// GraphDataView&#x3C;'uint32'> or &#x3C;'float32'>, cellCount * periodCount values</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    from: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'cell-major'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// the layout of \`values\`; the output holds the other one</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    cellCount,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    periodCount,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    output: periodMajorCube </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// same format and length</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  })</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">);</span></span></code></pre>
<ul>
<li>Every value is copied bit for bit (counts, NaN payloads and signed zeros alike), and transposing back returns the original cube.</li>
<li>Limits: <code>cellCount * periodCount</code> is an integer at most 4294967295 and the cube must fit one storage binding, 33,554,432 values on a default device. Larger cubes are refused at declaration with the binding limit in the message.</li>
<li>The transpose is a scatter with one thread per element. <code>cellCount</code> and <code>periodCount</code> are overrides of the shader, so cubes of different shapes share one compiled module.</li>
</ul>
<h3 id="gpucubelayouttranspose-parameters"><code>GPUCubeLayoutTranspose</code> parameters</h3>
<ul>
<li>Per frame (no recompile): the contents of <code>values</code> and <code>output</code>.</li>
<li>Compile-time (needs a new graph): <code>from</code>, <code>cellCount</code>, <code>periodCount</code> and the value format.</li>
<li>Packer: none.</li>
</ul>
<h2 id="gputimewordsrelative"><code>GPUTimeWordsRelative</code></h2>
<p>Convert Int64 word times to <code>float32</code> ticks after an origin on the GPU.</p>
<p>Several contributors read relative <code>float32</code> times (<code>GPUKnoxTest</code>, <code>GPUMantelTest</code>, <code>GPUSpaceTimeVariogram</code>, float-mode <code>GPUTemporalReduction</code>) while uploads keep exact Int64 words. This node connects them without a CPU pass.</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> relative</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUTimeWordsRelative</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  timestamps, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// GraphDataView&#x3C;'uint32x2'>, Int64 words</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  temporalContext, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// 'int64-words', any unit and epoch</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  origin: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">1_700_000_000_000</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">n</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// ticks after the Unix epoch, in the context unit</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  output: relativeTimes </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// GraphDataView&#x3C;'float32'>, one row per timestamp</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">});</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(relative);</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">relative.outputTemporalContext; </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// {...temporalContext, encoding: 'float32-relative', epoch: 1_700_000_000_000n}</span></span></code></pre>
<ul>
<li>The difference to the origin is taken on the words with the exact <code>timeWordsDifferenceToF32</code> kernel and only then rounded, so it is exact below 2^24 ticks and has a relative error below 2^-23 beyond, with the right sign for every pair of Int64 values.</li>
<li><code>origin</code> is an instant (signed Int64 ticks after the Unix epoch). When the input context counts from another epoch, the node subtracts that epoch first, so the result is always ticks after <code>origin</code>.</li>
<li><code>outputTemporalContext</code> is <code>{...temporalContext, encoding: 'float32-relative', epoch: origin}</code>: the unit, time zone and axis are the input's. Nothing is converted between units.</li>
<li>Limit: the <code>uint32x2</code> timestamps must fit one storage binding, 16,777,216 rows on a default device; larger inputs are refused at declaration.</li>
</ul>
<h3 id="gputimewordsrelative-parameters"><code>GPUTimeWordsRelative</code> parameters</h3>
<ul>
<li>Per frame (no recompile): the contents of <code>timestamps</code> and <code>output</code>.</li>
<li>Compile-time (needs a new graph): the row count, <code>origin</code> and <code>temporalContext</code> (the origin words are shader overrides, so graphs that differ only in origin share a module).</li>
<li>Packer: none.</li>
</ul>
<h2 id="gpuvaluecube"><code>GPUValueCube</code></h2>
<p>Dense (cell, period) cube of count, sum and mean of a value column.</p>
<p><code>GPUTemporalReduction</code> and the hot spot recipe bin counts, minima, maxima, first and last values but no sum or mean. <code>GPUValueCube</code> adds them in the cell-major layout <code>cell * periodCount + period</code> that <code>GPUEmergingHotSpots</code> reads, so <code>GPUTimeWindowFilter</code> (the <code>mask</code>), <code>GPUPeriodIndex</code> (the <code>periods</code>) and the cube chain to Gi*, trend, Markov chains and panels with no readback.</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">  new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUValueCube</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    cellCount,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    periodCount,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    cells, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// GraphDataView&#x3C;'uint32'>, cell index per row</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    periods: periodIndexOutput, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// GraphDataView&#x3C;'uint32'> from GPUPeriodIndex</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    values, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// GraphDataView&#x3C;'float32'></span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    mask: filterOutputMask, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// optional, for example outputMask of GPUTimeWindowFilter</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    output: {</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">      counts, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// GraphDataView&#x3C;'uint32'>, cellCount * periodCount rows</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">      sums, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// GraphDataView&#x3C;'float32'></span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">      means, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// GraphDataView&#x3C;'float32'>, NaN where a bin is empty</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">      status: {invalidCount} </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// optional</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    }</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  })</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">);</span></span></code></pre>
<ul>
<li>Sums are deterministic and independent of thread order. Each value is quantized to the shared 96-bit fixed-point accumulator (<code>analysis-core/kernels/fixed-point-accumulator.ts</code>) and added with integer atomics; the exact integer is rounded once to float32. There are no float atomics. A float32 running sum that cancels <code>1e8 + 1 - 1e8</code> loses the <code>1</code>; the cube returns it exactly.</li>
<li>The unit is a power of two set by the largest magnitude among the rows that are binned and by the bit length of the row count. A value keeps all 24 significand bits while it is at least <code>max * 2^(h - 71)</code> (<code>h</code> the bit length of the row count; about 1e15 of dynamic range at a million rows); smaller values round to the nearest unit, so a sum of <code>n</code> terms is off by at most <code>n / 2</code> units before the final rounding. Because the unit follows the binned rows, the rounding of a bin can change when another set of rows is binned, for example another time window.</li>
<li><code>means</code> is the rounded sum divided by the count in float32. WGSL division is not correctly rounded (up to 2.5 ULP), so a mean is within a few units in the last place of the true mean; an empty bin is NaN, which <code>GPUEmergingHotSpots</code> treats as missing.</li>
<li>Not binned: masked rows (silent), rows whose period is <code>0xffffffff</code> (<code>GPU_PERIOD_INDEX_INVALID</code>, excluded upstream, silent), and rows with a cell or period out of range (including <code>GPU_PERIOD_INDEX_OUT_OF_RANGE</code>) or a non-finite value. The last group is counted in <code>status.invalidCount</code>.</li>
<li>Limits: <code>cellCount</code>, <code>periodCount</code>, <code>cellCount * periodCount</code> and the row count at most 4294967294; the fixed-point words need 12 bytes per bin in one storage binding, so <code>cellCount * periodCount</code> is at most <code>getGPUValueCubeMaximumBinCount(device)</code>, 11,184,810 on a default device. Larger cubes are refused at declaration with the binding limit in the message. Inputs and outputs are single packed views.</li>
</ul>
<h3 id="gpuvaluecube-parameters"><code>GPUValueCube</code> parameters</h3>
<ul>
<li>Per frame (no recompile): the contents of <code>cells</code>, <code>periods</code>, <code>values</code> and <code>mask</code>.</li>
<li>Compile-time (needs a new graph): the row count, <code>cellCount</code>, <code>periodCount</code>, and which outputs and <code>mask</code> are present.</li>
<li>Packer: none.</li>
</ul>
<h2 id="gpucellperiodkeys"><code>GPUCellPeriodKeys</code></h2>
<p>Packs a 64-bit H3, S2 or Quadbin cell key and a period into one group key for <code>GPUGroupStatistics</code>.</p>
<p>A dense cube needs a cell index below <code>cellCount</code>; a DGGS has billions of possible cells. <code>GPUGroupStatistics</code> already groups by <code>uint32x2</code> keys and computes count, sum, mean, minimum, maximum, variance and percentiles in a capacity-bounded table sorted by key, so a sparse DGGS by period cube is one packing kernel away.</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">  new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUCellPeriodKeys</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    cells: cellKeys, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// GraphDataView&#x3C;'uint32x2'>, little-endian (low, high) words</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    periods: periodIndexOutput, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// GraphDataView&#x3C;'uint32'> from GPUPeriodIndex</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    periodCount: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">365</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    fill: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'ones'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// 'ones' for H3 and Quadbin, 'zeros' for S2</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    mask, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// optional</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    output: {keys: packedKeys, status: {invalidCount}}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  })</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">);</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">  new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUGroupStatistics</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    keys: packedKeys,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    columns: [{values, statistics: [</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'count'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'sum'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'mean'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">], output: {counts, sumValues, means}}],</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    output: {keys: groupKeys, counts: groupCounts, status: {count, overflow}}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  })</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">);</span></span></code></pre>
<p>Which route builds a DGGS (cell, period) cube without readback? Two compose from the existing pieces: (a) this packing kernel and <code>GPUGroupStatistics</code> on the composite key, or (b) <code>GPUCellAggregation</code> once to get the distinct cells, <code>GPUKeyJoin</code> to map every row to its cell row, then <code>row * periodCount + period</code> as a key into <code>GPUValueCube</code> or <code>GPUGroupStatistics</code> in dense mode. Use (a) for DGGS cubes: it needs one sort and no cell-row lookup, its memory follows the number of occupied (cell, period) groups instead of <code>cellCount * periodCount</code>, and cells never need a table. Use (b), or <code>GPUValueCube</code> with a cell index you already have (a lattice, or a weights table), only when the consumer needs the dense cell-major cube, as <code>GPUEmergingHotSpots</code> and the layout transpose do.</p>
<ul>
<li>Packing: with <code>b = ceil(log2(periodCount))</code> period bits, the key is <code>(cellKey &#x26; ~(2^b - 1)) | period</code>. The period replaces the low <code>b</code> bits of the cell key, so the packed key sorts by cell and then period. A packed key is not a valid cell id: unpack the group table keys with <code>unpackGPUCellPeriodKey(key, periodCount, fill)</code>.</li>
<li>What happens to the key bits: the cell key stays 64 bits, nothing is truncated to 32. The <code>b</code> replaced bits must hold the <code>fill</code> pattern. At the resolution of the cells they do: unused H3 digits are 7 (<code>3 * (15 - resolution)</code> low bits, all ones), the Quadbin footer is all ones (<code>52 - 2 * zoom</code> bits), an S2 id has a lone sentinel one above <code>60 - 2 * level</code> zero bits. Within that many free bits the packing is lossless and injective. A cell with any other bits where the period goes (a finer cell) is not packed: its key becomes the reserved all-ones key, which <code>GPUGroupStatistics</code> skips, and it is counted in <code>status.invalidCount</code>; cells are never merged silently. A finer cell whose extra bits happen to equal the fill pattern is packed and unpacks to itself. <code>getGPUCellKeyFreeBits('h3' | 'quadbin' | 's2', resolution)</code> gives the free bits. 365 daily periods need 9 bits: H3 resolution 12 or coarser, Quadbin zoom 21 or coarser, S2 level 25 or coarser. Hourly periods over a year (8,760) need 14 bits: H3 resolution 10, Quadbin zoom 19, S2 level 23.</li>
<li>Rows are skipped silently when masked, when the cell is the all-ones "no cell" key, or when the period is <code>0xffffffff</code>. A period at or above <code>periodCount</code> is skipped and counted.</li>
<li>Limits: <code>periodCount</code> at most 4294967293 (32 period bits); <code>cells</code> fit one storage binding, 16,777,216 rows on a default device. Downstream, <code>GPUGroupStatistics</code> keeps at most <code>getGPUGroupStatisticsMaximumGroupCount</code> (8,388,608 on a default device) distinct (cell, period) groups and keeps the smallest keys on overflow, and its <code>sums</code> words are fixed point with a resolution of <code>1 / sumScale</code> (65536 by default); its <code>sumValues</code> and <code>means</code> use a unit that follows the data, as <code>GPUValueCube</code> does.</li>
</ul>
<h3 id="gpucellperiodkeys-parameters"><code>GPUCellPeriodKeys</code> parameters</h3>
<ul>
<li>Per frame (no recompile): the contents of <code>cells</code>, <code>periods</code> and <code>mask</code>.</li>
<li>Compile-time (needs a new graph): the row count, <code>periodCount</code>, <code>fill</code>, and whether <code>mask</code> and <code>status.invalidCount</code> are present.</li>
<li>Packer: none.</li>
</ul>
<h2 id="related-pages">Related pages</h2>
<ul>
<li><a href="https://luma.gl/docs/api-reference/experimental/gpu-dataframe">GPU Dataframe</a></li>
<li><a href="https://luma.gl/docs/api-reference/experimental/gpu-dataframe-indexes-joins">Indexes and joins</a></li>
<li><a href="/luma-contributors/docs/api/gpu-spatial-analysis">GPU Spatial Analysis contributors</a></li>
</ul>
<h2 id="gpucrossvariogram"><code>GPUCrossVariogram</code></h2>
<p>{/* lens:GPUCrossVariogram */}</p>
<p>::lens{scenario=gpu-dataframe.cross-variogram rung=graph}</p>
<p>This executable four-point projected example captures pair counts, estimator values and validity status. See <a href="/luma-contributors/docs/api/gpu-spatial-analysis/statistical-expansion">the estimator contracts</a> for bin edges, complete-case masks, normalization and traversal bounds.</p>
<p>The default <code>[256, 256]</code> index accepts the same row counts as an explicit <code>gridSize</code>;
<code>rows * lagCount * temporalLagCount</code> must fit the <code>2^24</code> partial-entry budget and the device's
storage binding limit. The former 16,384-row cap on the default has been removed.
If a bin's pair count exceeds <code>0xffffffff</code>, its count saturates at that value, status is 3,
and semivariance is NaN. An exact count of <code>0xffffffff</code> remains valid.</p>
<h2 id="gpuspacetimevariogram"><code>GPUSpaceTimeVariogram</code></h2>
<p>{/* lens:GPUSpaceTimeVariogram */}</p>
<p>::lens{scenario=gpu-dataframe.space-time-variogram rung=graph}</p>
<p>Empirical space-time semivariogram on the indexed pair histogram, the same traversal as <code>GPUVariogram</code>. Each unordered pair within <code>maximumDistance</code> and <code>maximumTimeLag</code> is counted once, in a spatial class and a temporal class. Class 0 of each axis is the exact zero lag, so the output holds <code>(lagCount + 1) * (temporalLagCount + 1)</code> bins, spatial-major (<code>spatialClass * (temporalLagCount + 1) + temporalClass</code>): the first row is the purely temporal variogram, the first column the purely spatial one, and bin 0 is coincident points at the same time. Positive lags use equal-width classes over <code>(0, maximum]</code>.</p>
<p>Props: <code>positions</code>, <code>values</code>, optional <code>secondValues</code> (signed cross-semivariance, default the Matheron semivariogram), <code>times</code>, optional <code>mask</code>, <code>parameters</code> written with <code>getGPUSpaceTimeVariogramParameterValues({bounds, maximumDistance, maximumTimeLag})</code>, <code>gridSize</code>, <code>lagCount</code>, <code>temporalLagCount</code> and <code>output: {semivariances, pairCounts, status}</code>. Status per bin is 0 defined, 1 empty or invalid parameters, 2 nonfinite arithmetic, 3 accumulation overflow.</p>
<p><code>times</code> must be relative to a CPU float64 origin. Raw Unix epoch seconds in float32 are spaced 128 s apart (milliseconds about 2 minutes), which merges every short lag into one. Subtract the origin with <code>getGPUSpaceTimeRelativeTimes</code> from <code>@luma-contributors/analysis/gpu-spatial-analysis</code> before upload; it also returns the remaining float32 resolution. Times closer than that resolution share the zero-lag class. See <a href="/luma-contributors/docs/api/gpu-spatial-analysis/statistical-expansion">the estimator contracts</a> for bin edges, complete-case masks and the validation status: the results match an in-repo brute-force reference, and a gstat <code>variogramST</code> fixture is still pending.</p>
<p>Breaking change: the parameter buffer is now eight values (bounds, then the two maximum lags), <code>gridSize</code> is required, the output has the extra zero-lag classes, and the limit is 409 bins instead of 16,384 rows by 128 bins.</p>
<p>{/* parameters: GPUSpaceTimeVariogram generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h3 id="gpuspacetimevariogram-parameters"><code>GPUSpaceTimeVariogram</code> parameters</h3>
<ul>
<li>Packer: <code>getGPUSpaceTimeVariogramParameterValues(settings)</code>, writes <code>GPU_SPACE_TIME_VARIOGRAM_PARAMETER_LENGTH</code> (8) <code>float32</code> values.</li>
</ul>
<p>Settings of <code>getGPUSpaceTimeVariogramParameterValues</code>:</p>
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
<td><code>Exclude&#x3C;GPUPairStatisticsBounds, 'study-window'></code></td>
<td>Inclusive <code>[minX, minY, maxX, maxY]</code> extent; rows outside it are excluded.</td>
</tr>
<tr>
<td><code>maximumDistance</code></td>
<td><code>number</code></td>
<td>Largest spatial lag, in the units of the positions.</td>
</tr>
<tr>
<td><code>maximumTimeLag</code></td>
<td><code>number</code></td>
<td>Largest time lag, in the unit of the relative <code>times</code>.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h2 id="gpumarkedpointstatistics"><code>GPUMarkedPointStatistics</code></h2>
<p>{/* lens:GPUMarkedPointStatistics */}</p>
<p>::lens{scenario=gpu-dataframe.marked-point-statistics rung=graph}</p>
<p>This executable four-point projected example captures pair counts, estimator values and validity status. See <a href="/luma-contributors/docs/api/gpu-spatial-analysis/statistical-expansion">the estimator contracts</a> for bin edges, complete-case masks, normalization and traversal bounds.</p>
<p>Candidate search defaults to <code>searchStructure: 'grid'</code> with <code>gridSize: [256, 256]</code> maximum
dimensions. It accepts <code>rows * radiusCount * 3 &#x3C;= 2^24</code> float32 partials, subject to the device's
binding size. The explicit <code>searchStructure: 'brute-force'</code> reference is limited to 16,384 rows
and cannot take <code>gridSize</code>. Both use the same estimators and radius edges; indexed sums may differ
within float32 rounding tolerance. Finite radii whose square overflows use a single-cell fallback
with the finite-distance predicate. Broad neighborhoods can still require quadratic pair work.
Counts exceeding <code>0xffffffff</code> saturate, with status 3 and all four statistics NaN for that radius.
An exact count of <code>0xffffffff</code> remains valid.
Category means use a separate scaled sum for large marks, averaging before restoring the scale;
the unnormalized mark sum need not fit float32. Small marks remain unscaled.</p>
<h2 id="gpuvariogrammodelfit"><code>GPUVariogramModelFit</code></h2>
<p>The direct contributor and both variogram fit operations snapshot model order and options at
construction. The operations also capture input and output graph names. Caller edits to model
arrays or output maps cannot change the declared row counts, result keys or later GPU lowering.
The contributor preserves view/buffer identities and dynamic contents. Construct a new fit to
change its model list, options or bindings.</p>
<p>Both operation forms require three distinct output graph names, each different from every input
graph name. Collisions throw at construction with the operation ID and conflicting output port;
shared read-only input names are allowed. This prevents an output from erasing another result or
overwriting an input in the declared graph contract.</p>
<p>{/* lens:GPUVariogramModelFit */}</p>
<p>::lens{scenario=gpu-dataframe.variogram-model-fit rung=result caption="Resident spherical, exponential and Gaussian fits to 20 explicitly synthetic empirical lag bins."}</p>
<p>::lens{scenario=gpu-dataframe.variogram-model-fit rung=graph caption="The bounded GPU range search produces fitted parameters, residual/AIC diagnostics and convergence status."}</p>
<p>This executable example uploads bins generated from an exponential model with nugget <code>0.2</code>, partial sill <code>1.5</code> and effective range <code>2.2</code>, then fits all three candidates on the GPU with pair weights. Model rows are spherical, exponential and Gaussian. Captured parameter rows contain nugget, partial sill, effective range and weighted residual; diagnostic rows contain usable bins, unweighted residual, residual AIC and final log-range bracket. Status 0 is an interior converged fit, 1 insufficient bins, 2 numerical failure and 3 a boundary/budget result. See <a href="/luma-contributors/docs/api/gpu-spatial-analysis/interpolation-expansion">the fitting contract</a> for identifiability, weighting and performance limitations.</p>
<p>The fitter also supports <code>matern-3/2</code> and <code>matern-5/2</code>. With <code>a = factor * h / range</code>, their normalized semivariogram shapes are <code>1 - (1 + a) exp(-a)</code> and <code>1 - (1 + a + a² / 3) exp(-a)</code>. The factors are <code>4.749031386012701</code> and <code>5.924462618951797</code>, respectively, giving correlation <code>exp(-3)</code> at the effective range. These definitions also apply to coregionalization below. CPU evaluation uses a positive exponential-remainder series through order 18 for <code>a &#x3C; 0.5</code>; GPU evaluation uses a bounded Taylor series for <code>a &#x3C; 0.1</code>. These evaluations preserve the quadratic leading term at tiny lag/range ratios. CPU exponential and Gaussian shapes use <code>expm1</code> for the same cancellation concern. The mathematical model and fitting objective are unchanged by these numerical evaluations.</p>
<h2 id="gpucoregionalizationfit"><code>GPUCoregionalizationFit</code></h2>
<p>{/* lens:GPUCoregionalizationFit */}</p>
<p>::lens{scenario=gpu-dataframe.coregionalization-fit rung=result caption="Fit a PSD two-variable structure matrix with a negative cross coefficient; exclude one zero-pair bin."}</p>
<p>::lens{scenario=gpu-dataframe.coregionalization-fit rung=graph caption="Gated residual fits and PSD projections publish matrices, square loading factors and named convergence status."}</p>
<p>Fits a positive semidefinite linear model of coregionalization from complete direct/cross
semivariogram matrices on shared lag supports. The <a href="/luma-contributors/docs/api/dataframe-analysis/coregionalization-fit">coregionalization guide</a>
is the canonical reference for input layout, per-frame parameters, GPU bounds, output sizing,
budget/validity status, independent verification and supplying the fitted model to local kriging.</p>
<p>It covers <code>GPUCoregionalizationFit</code>, <code>GPUCoregionalizationFitOperation</code>,
<code>CPUCoregionalizationFitOperation</code>, <code>computeCoregionalizationFitOnCPU</code> and
<code>fitLinearModelOfCoregionalization</code>. Prefer the float64 CPU twin for small host-resident fits;
the GPU operation uses <code>GPU_CONTRIBUTOR_OPERATION_LOWERING</code> with the same graph properties.</p>
<p>The lens example uses three hand-derived lag matrices from <code>B = [[2, -1], [-1, 3]]</code> and a unit-sill spherical structure of range 2. A fourth, unrelated matrix has zero pairs and is excluded. When run, the scenario uses an eight-sweep budget and records GPU matrices, factors, residual diagnostics and every named status port.</p>
<p>{/* parameters: GPUCoregionalizationFit generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h3 id="gpucoregionalizationfit-parameters"><code>GPUCoregionalizationFit</code> parameters</h3>
<ul>
<li>Packer: <code>getGPUCoregionalizationFitParameterValues(structures, tolerance?)</code>, writes the <code>float32</code> parameter view.
<ul>
<li><code>structures</code> (<code>readonly CoregionalizationStructure[]</code>)</li>
<li><code>tolerance</code> (<code>number</code>)</li>
</ul>
</li>
</ul>
<p>{/* /parameters */}</p>
<h2 id="gpuforestinference"><code>GPUForestInference</code></h2>
<p>{/* lens:GPUForestInference */}</p>
<p>::lens{scenario=gpu-dataframe.forest-inference rung=formula}</p>
<p>Apply an imported random forest or boosted model to resident feature rows or co-registered raster
bands. The synthetic raster-stack lens shows a tied classification leaf, the missing-feature branch,
and a masked padding row. See <a href="/luma-contributors/docs/api/forest-inference">Forest inference</a> for model import formats,
parameters, output scores and leaf indices, limits and validation.</p>
<h2 id="gpuwidenumericscan"><code>GPUWideNumericScan</code></h2>
<p>Low-level word-phase execution shared by exact Int64/Decimal128 aggregates and Float32 narrowing
diagnostics. All rows run in parallel; validity chunks must match the packed word chunks. Its
caller initializes the result and owns the bounded readback. See
<a href="/luma-contributors/docs/api/reused-analysis-preparation">the public preparation and profiling contracts</a> for precision,
row limits, CPU references and the exact bigint integer-error representation. This execution
helper has no standalone lens; consumers use the domain aggregate and diagnostic functions.</p>`};export{e as default};