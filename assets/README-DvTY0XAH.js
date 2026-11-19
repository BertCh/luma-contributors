var e={slug:`api/gpu-network`,title:`GPU Network`,html:`<h1 id="gpu-network">GPU Network</h1>
<p>:::caution Experimental
<code>@luma-contributors/analysis/gpu-network</code> is an initial, experimental entry point. APIs may change between
releases without a deprecation period.
:::</p>
<h2 id="overview">Overview</h2>
<p><code>@luma-contributors/analysis/gpu-network</code> provides analysis contributors for networks stored as CSR
adjacency, passed as one <code>topology: GPUNetworkTopology</code> (<code>offsets</code>, <code>neighbors</code>, <code>weights</code>, optional
<code>nodePositions</code>): weighted reachability and isochrones (isoband or cell-outline polygons), path
extraction, service areas, neighborhoods, snapping, cost matrices, accessibility scores, subgraph
filters, coarsening, network statistics, node-aligned analytics columns, origin-destination flow
aggregation, edge bundling, adjacency-matrix images, and network construction and analysis from line
data (noding, turn-restricted line graphs, map matching, network K function). Each contributor declares resources and
command nodes into a caller-owned <a href="https://luma.gl/docs/api-reference/experimental/gpu-core/gpu-command-graph"><code>GPUCommandGraph</code></a> through
<code>getCommandNodes(graph)</code> (<code>GPUCommandNodeProducer</code>). It never compiles, submits, encodes, or reads
back.</p>
<p>See <a href="/luma-contributors/docs/api/gpu-network/expansion">network structure, alternative routes, edge centrality and spatial topology extensions</a>
for the bounded expansion APIs, their result contracts and reference checks.</p>
<h2 id="when-to-use-it">When to use it</h2>
<p>Use GPU Network for map and road-network tasks that need bounded, renderable results: isochrone
bands that recolor as a cost limit moves, a path that follows a dragged endpoint, a statistics panel
or analytics columns for deck.gl styling, or aggregated flows and bundled edges for drawing.</p>
<p>The generic graph algorithms live in <a href="https://luma.gl/docs/api-reference/experimental/gpu-graph">GPU Graph</a>, and the network contributors build
on them rather than replace them:</p>
<ul>
<li><code>GPUNetworkAnalyticsColumns</code> publishes degree, in-degree and weak component columns on the caller's
CSR views with analysis-owned kernels, then normalizes the columns for styling. Published
luma.gl graph algorithms (<code>GPUGraphAlgorithmsPageRank</code>, <code>GPUGraphAlgorithmsCoreNumber</code>,
<code>GPUGraphAlgorithmsLabelPropagation</code>, <code>GPUGraphAlgorithmsModularityOptimization</code>) accept only
physical <code>GPUGraphAlgorithmsTopology</code>, so call them yourself when you need ranking, core numbers
or communities, or their raw outputs and convergence status.</li>
<li><code>GPUNetworkStatistics</code> reduces a CSR to one summary row (counts, components, degree histograms,
optional modularity) for a stats panel.</li>
<li><code>GPUNetworkReachability</code> is a multi-source, cost-limited shortest-path search with isochrone
bands and cycle-safe predecessors, all written to caller-owned views. Use
<code>GPUGraphAlgorithmsSingleSourceShortestPath</code> when you need plain bounded shortest paths from one source and
the GPU Graph topology and status contracts.</li>
<li><code>GPUGraphAlgorithmsForceLayout</code> and <code>GPUGraphAlgorithmsSpatialForceLayout</code> position nodes; <code>GPUEdgeBundling</code> then
turns the straight edges into renderable polylines.</li>
</ul>
<h2 id="quick-start">Quick start</h2>
<p>Use the <a href="/luma-contributors/docs/api/operation-forms-and-cpu-twins#point-density-from-device-to-readback">complete GPU walkthrough</a>
for device creation, buffer imports, submission, typed readback and cleanup. For reachability,
replace its point-density inputs with this complete three-node directed CSR fixture.
The only edge is 1 → 2 with cost 2; node 0 is isolated.</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">export</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> reachabilityInputs</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8"> {</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  topology: {</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    offsets: Uint32Array.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">of</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    neighbors: Uint32Array.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">of</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    weights: Float32Array.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">of</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  },</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  startNodes: Uint32Array.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">of</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  costLimit: Float32Array.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">of</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">10</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  costBreaks: Float32Array.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">of</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">5</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">10</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">};</span></span></code></pre>
<p>Import every array once, then create scalar <code>uint32</code> views for offsets, neighbors and start nodes,
and scalar <code>float32</code> views for weights, cost limit and cost breaks, with lengths equal to their
array lengths. Pass the three topology views together as <code>topology</code> to <code>GPUNetworkReachability</code>,
and the remaining views as <code>startNodes</code>, <code>costLimit</code> and <code>costBreaks</code>. Set
<code>maximumIterations: 4</code> for this fixture.</p>
<p>Allocate the following outputs in separate buffers and pass their views as <code>output</code>:</p>
<table>
<thead>
<tr>
<th>Port</th>
<th>Format and rows</th>
<th>Expected readback</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>costs</code></td>
<td><code>float32</code>, 3</td>
<td><code>[Infinity, 0, 2]</code></td>
</tr>
<tr>
<td><code>bands</code></td>
<td><code>uint32</code>, 3</td>
<td><code>[0xffffffff, 0, 1]</code></td>
</tr>
<tr>
<td><code>bandCounts</code></td>
<td><code>uint32</code>, 4</td>
<td><code>[1, 1, 0, 0]</code></td>
</tr>
<tr>
<td><code>status.converged</code></td>
<td><code>uint32</code>, 1</td>
<td><code>[1]</code></td>
</tr>
<tr>
<td><code>status.iterationLimitReached</code></td>
<td><code>uint32</code>, 1</td>
<td><code>[0]</code></td>
</tr>
</tbody>
</table>
<p>There is exactly <strong>one start slot</strong>. Every slot is active when <code>startCount</code> is omitted; if reserving
spare slots, supply a one-row <code>uint32</code> <code>startCount</code> view containing 1. Spare zero-filled slots
would seed node 0 too. Cost breaks must be initialized in ascending order in the edge-weight units;
a node belongs to the first break greater than or equal to its cost, or <code>0xffffffff</code> beyond
the final break. Node 0 stays unreachable and contributes to no band.</p>
<p>Compile, encode, submit and read these buffers with the walkthrough's lifecycle. Check
<code>iterationLimitReached</code> with the costs: when it is 1, propagate incompleteness, suppress dependent
results, or continue/fall back rather than treating unreached nodes as proof of no route.</p>
<h2 id="concepts-and-operators">Concepts and operators</h2>
<ul>
<li><a href="/luma-contributors/docs/api/gpu-network/concepts">GPU Network concepts</a>: conventions, units and the GPU execution model.</li>
<li><a href="/luma-contributors/docs/api/gpu-network/operators">GPU Network operators</a>: every contributor, its inputs, outputs, parameters and how it
computes.</li>
</ul>
<h2 id="catalog">Catalog</h2>
<p>One figure per contributor, each drawn from a recorded GPU run on real data; click through to its card.</p>
<p>::lens-catalog{entry=gpu-network}</p>
<h2 id="limits-and-compatibility">Limits and compatibility</h2>
<ul>
<li>GPU Network is experimental and WebGPU-only.</li>
<li>CSR graphs use packed views only. Edge weights must be non-negative where a contributor takes costs.</li>
<li>Each contributor's TSDoc lists its non-goals. The GPU Core maintainer roadmap
(<code>dev-docs/roadmap.md</code>) tracks what is still open.</li>
</ul>
<h2 id="related-modules">Related modules</h2>
<ul>
<li>
<p><a href="https://luma.gl/docs/api-reference/experimental/gpu-graph">GPU Graph</a> supplies the generic algorithms, topology views, and layouts.</p>
</li>
<li>
<p><a href="/luma-contributors/docs/api/gpu-raster">GPU Raster</a> has the raster counterpart of cost distance and isolines.</p>
</li>
<li>
<p><a href="https://luma.gl/docs/api-reference/experimental/gpu-core/concepts">GPU Core</a> defines contributors, composition levels, and graph ownership.</p>
</li>
<li>
<p><a href="/luma-contributors/docs/api/gpu-network/alternative-routes">Bounded alternative routes</a></p>
</li>
</ul>`};export{e as default};