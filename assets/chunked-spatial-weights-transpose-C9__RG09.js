var e={slug:`api/gpu-spatial-analysis/chunked-spatial-weights-transpose`,title:`GPUChunkedSpatialWeightsTranspose`,html:`<h1 id="gpuchunkedspatialweightstranspose">GPUChunkedSpatialWeightsTranspose</h1>
<p>See also the <a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-weights#gpuchunkedspatialweightstranspose">spatial weights API</a>.</p>
<p><code>GPUChunkedSpatialWeightsTranspose</code> accepts <code>GPUChunkedSpatialWeights</code> and writes either a packed
CSR or independently partitioned output chunks. Import it and
<code>GPUChunkedSpatialWeightsTransposeOutput</code> from
<code>@luma-contributors/analysis/gpu-spatial-analysis</code>.</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> output</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">:</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUChunkedSpatialWeightsTransposeOutput</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8"> {</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  chunks: [</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    {</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">      rowStart: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">      rowCount: firstRowCount,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">      weights: {</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">        offsets: firstOffsets, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// firstRowCount + 1 uint32 entries</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">        neighbors: firstNeighbors, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// global input row IDs</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">        weights: firstWeights,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">        distances: firstDistances, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// optional; requires distances on every input chunk</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">        status: {</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">          count: firstCount,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">          requiredCount: firstRequiredCount,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">          overflow: firstOverflow</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">        }</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">      }</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    },</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    {</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">      rowStart: firstRowCount,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">      rowCount: columnCount </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">-</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8"> firstRowCount,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">      weights: {</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">        offsets: secondOffsets,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">        neighbors: secondNeighbors,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">        weights: secondWeights,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">        status: {</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">          count: secondCount,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">          requiredCount: secondRequiredCount,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">          overflow: secondOverflow</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">        }</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">      }</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    }</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  ],</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  asymmetricSlots </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// optional, square matrices only</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">};</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> transpose</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUChunkedSpatialWeightsTranspose</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({weights: input, columnCount, output});</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(transpose);</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// Feed transpose.outputWeights directly to a consumer accepting GPUChunkedSpatialWeights.</span></span></code></pre>
<p>Each output chunk is an ordinary CSR with offsets starting at zero. Chunks cover
<code>[0, columnCount)</code> consecutively, with at least one row and one slot of physical capacity each.
Their boundaries can differ from the input chunk boundaries. Neighbor IDs remain global input
row IDs, ascending within each output row. Empty rows and entirely empty chunks are supported;
input padding and neighbor IDs outside <code>[0, columnCount)</code> are discarded.</p>
<p>Each chunk requires separate one-element <code>uint32</code> status views. <code>count</code> is the number of retained
slots, <code>requiredCount</code> is the exact requirement before truncation, and <code>overflow</code> is one when this
chunk dropped slots. An overflowing chunk retains the prefix of its complete CSR, possibly ending
inside a row; all its offsets are clamped to capacity. Extra capacity in a different chunk is not
borrowed. Unused slots are zeroed, including distances. These values reset on every encoding.</p>
<p><code>asymmetricSlots</code> counts mismatched positions in the union of the patterns of the input and its
transpose, using <code>symmetryTolerance</code> (default zero). It is computed from the input and remains
exact even when output chunks overflow. A structurally absent reverse slot counts as a mismatch
even when the present weight is zero. The input must satisfy the usual CSR invariants, including
sorted unique neighbor IDs in each row.</p>
<p>The described <code>outputWeights</code> retains the output layout, spatial context and matrix-level input
status. Output chunks retain their own capacity status. Output views must not share buffers with
matrix-level or chunk-level input status views. Row, variance and double transform metadata becomes
<code>unknown</code>: chunk capacity may truncate the result and break global normalization. Chunked output
symmetry metadata is also <code>unknown</code> because capacity can truncate it.</p>
<p>The packed output API remains supported: pass <code>offsets</code>, <code>neighbors</code>, <code>weights</code>, optional
<code>distances</code> and optional <code>asymmetricSlots</code> directly in <code>output</code>. Its capacity must still be at least
the sum of input capacities.</p>
<h2 id="limits-and-work">Limits and work</h2>
<p>No chunked output path packs the whole CSR or allocates scratch for all output rows. Every input
sort and every output scan fits its own storage binding; device-dependent binding checks include
alignment prefixes and run before scratch allocation or node creation. A rejected binding preflight
leaves the graph available for retry with corrected views. The <code>outputWeights</code> description and its
chunk wrappers stay stable, and their CSR view properties and status ports follow same-shaped
corrections made to the caller's output before node generation, including descriptions saved before
a failed preflight.
Replace views before generating successful nodes or wiring downstream consumers; shapes, formats,
chunk ranges and optional-view presence remain fixed at construction. Packed input chunk capacities
are limited to <code>2^31</code> slots by published <code>GPUSort</code>.
The summed input capacity is at most <code>2^32 - 1</code>, or <code>2^31 - 1</code> with symmetry counting, so counts
cannot wrap. <code>columnCount</code> is in <code>[1, 2^32 - 2]</code>.</p>
<p>The shared transpose pipeline runs once per output range and repeats stable input sorts for that
range. Work grows with <code>inputChunks * outputChunks</code>; symmetry checking additionally visits input
chunk pairs. The construction-time <code>estimatedCommandNodeCount</code> includes conservative sort, scan
and reduction allowances and must fit the shared 32,768-node budget. Choose fewer, larger row
chunks when the command budget is exceeded, while respecting device binding sizes.</p>
<p>View shapes, chunk ranges, <code>columnCount</code> and optional output presence are compile-time, matching
the packed transpose. CSR buffer contents and a supplied <code>symmetryTolerance</code> view may change
between encodings; a numeric tolerance sets the construction-time default. There is no CPU operation twin;
the GPU specs compare against an independent column-first CPU transpose.</p>
<p>{/* parameters: GPUChunkedSpatialWeightsTranspose generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h2 id="gpuchunkedspatialweightstranspose-parameters"><code>GPUChunkedSpatialWeightsTranspose</code> parameters</h2>
<ul>
<li>Packer: <code>getGPUSpatialWeightsTransposeParameterValues(settings?)</code>, writes <code>GPU_SPATIAL_WEIGHTS_TRANSPOSE_PARAMETER_LENGTH</code> (1) <code>float32</code> values.
<ul>
<li>Input port: <code>symmetryTolerance</code>.</li>
</ul>
</li>
</ul>
<p>Settings of <code>getGPUSpatialWeightsTransposeParameterValues</code>:</p>
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
<td><code>symmetryTolerance</code> (optional)</td>
<td><code>number</code></td>
<td>Non-negative absolute reverse-weight tolerance. Missing reverse slots always differ. Defaults to 0.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>`};export{e as default};