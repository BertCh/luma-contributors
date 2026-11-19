var e={slug:`api/gpu-network/k-shortest-paths`,title:`K-shortest simple paths`,html:`<h1 id="k-shortest-simple-paths">K-shortest simple paths</h1>
<h2 id="gpukshortestpaths">GPUKShortestPaths</h2>
<p>Import from <code>@luma-contributors/analysis/gpu-network</code>. This scale-tier contributor uses
<a href="https://doi.org/10.1287/mnsc.17.11.712">Yen's algorithm (1971)</a> to enumerate the first <code>routeCount</code>
simple edge paths of a directed CSR network, ordered by cost. Every round runs the possible
spur searches concurrently through <code>GPUNetworkReachability</code>, removing the root vertices and
next edges of already accepted routes. Candidates persist across rounds. Parallel CSR edges
are distinct paths; vertices never repeat, including across zero-cost cycles.</p>
<p>Unlike the reference-tier <code>GPUAlternativeRoutes</code>, the network has no 256-vertex limit.
Vertex count is constrained by device storage bindings and the number of concurrent spur lanes.
No coordinates are read, so the coordinate model is topological. Inputs must be packed views;
CSR offsets must be nondecreasing and in range. Unused trailing edge capacity is allowed.
Invalid neighbors, negative weights, NaNs and infinities are impassable. Invalid query endpoints
produce empty converged results. An identical source and target produces one zero-edge path.</p>
<p>Compile-time properties:</p>
<table>
<thead>
<tr>
<th>Property</th>
<th>Default</th>
<th>Accepted range</th>
<th>Purpose</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>routeCount</code></td>
<td>Required</td>
<td>1–32</td>
<td>Number of requested routes per query</td>
</tr>
<tr>
<td><code>maximumPathLength</code></td>
<td>64</td>
<td>1–1,024</td>
<td>Vertex slots per route, root comparisons and spur lanes</td>
</tr>
<tr>
<td><code>candidateCapacity</code></td>
<td>256</td>
<td>1–4,096</td>
<td>Persistent pending routes per query</td>
</tr>
<tr>
<td><code>maximumIterations</code></td>
<td>64</td>
<td>1–1,024</td>
<td>Rounds for each shortest-path and zero-cost tie phase</td>
</tr>
</tbody>
</table>
<p>The shared command-node budget of 32,768 also applies. <code>estimatedCommandNodeCount</code> is a
conservative upper bound, <code>8 + routeCount * (16 + 4 * maximumIterations)</code>. Each shortest-path
round chains up to 64 local hops. Relaxation and tie rounds use GPU-written indirect dispatch.
Finished Yen batches gate every composed shortest-path pass as well as edge filtering and
reconstruction on the GPU. Batch predicates are combined with the live frontier dispatches, so
skipped initialization cannot reuse stale dispatch arguments from an earlier encoding. No status
is read back inside the contributor. All input view contents are per-frame, including topology weights and
query endpoints; capacities and view lengths are compile-time.</p>
<p>Use <code>getGPUKShortestPathsOutputLayout(options)</code> from the same entry point to obtain the scalar
format and required row count of every output before allocating caller-owned buffers. The
constructor, operation form and CPU twin use this function too. <code>queryCount</code> and <code>routeCount</code>
are required; <code>maximumPathLength</code> defaults to 64. Named dimensions are resolved explicitly,
including inherited getters, so validation and output slicing use the same values.
Dimensions and their products must be positive
integers fitting uint32 row counts. GPU execution limits in the table above and device binding
limits still apply to the contributor; the layout function also describes larger CPU budgets.</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">import</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8"> {getGPUKShortestPathsOutputLayout} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">from</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF"> '@luma-contributors/analysis/gpu-network'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">;</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> options</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8"> {queryCount: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, routeCount: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">3</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, maximumPathLength: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">4</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">};</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> layout</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> getGPUKShortestPathsOutputLayout</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(options);</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// Allocate each output with its descriptor's format and length.</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// layout.vertices = {format: 'uint32', length: 24}</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// layout.costs = {format: 'float32', length: 6}</span></span></code></pre>
<table>
<thead>
<tr>
<th>Output</th>
<th>Descriptor</th>
<th>Order and meaning</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>vertices</code></td>
<td><code>layout.vertices</code></td>
<td>Query, route, vertex slot; <code>uint32</code></td>
</tr>
<tr>
<td><code>edges</code></td>
<td><code>layout.edges</code></td>
<td>Query, route, edge slot, same capacity as vertices; <code>uint32</code></td>
</tr>
<tr>
<td><code>lengths</code></td>
<td><code>layout.lengths</code></td>
<td>Query, route; vertex count, <code>uint32</code></td>
</tr>
<tr>
<td><code>costs</code></td>
<td><code>layout.costs</code></td>
<td>Query, route; <code>float32</code></td>
</tr>
<tr>
<td><code>counts</code></td>
<td><code>layout.counts</code></td>
<td>Accepted route count per query, <code>uint32</code></td>
</tr>
<tr>
<td><code>status.&#x3C;name></code></td>
<td><code>layout.status.&#x3C;name></code></td>
<td>One <code>uint32</code> row per batch for each status below</td>
</tr>
</tbody>
</table>
<p>Vertices and edges have query × route × maximumPathLength slots, padded with <code>0xffffffff</code>;
the last vertex has no edge. Lengths and costs have query × route slots: empty slots have length
zero and cost positive infinity. The fixed requested top-k layout does not promise to count all simple paths.
Every output uses its own buffer, separate from inputs.</p>
<p><code>output.status</code> contains four required one-row uint32 views:</p>
<table>
<thead>
<tr>
<th>Port</th>
<th>Meaning</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>converged</code></td>
<td>Every query accepted k routes or exhausted all remaining paths</td>
</tr>
<tr>
<td><code>iterationLimitReached</code></td>
<td>A shortest-path/tie search did not converge, or a shortest spur exceeded <code>maximumPathLength</code></td>
</tr>
<tr>
<td><code>candidateOverflow</code></td>
<td>A query's persistent candidate pool could not retain a new distinct candidate</td>
</tr>
<tr>
<td><code>iterationCount</code></td>
<td>Maximum Yen rounds attempted by any query, at most <code>routeCount</code></td>
</tr>
</tbody>
</table>
<p>On either budget stop, previously certified routes remain published. A path-length stop is a
budget interruption, <strong>not</strong> enumeration restricted to paths of at most that length: simply dropping
an overlong spur could skip a cheaper path. A shortest-search interruption conservatively stops
all queries still active in that batch. Candidate overflow affects only its own query. Status
flags aggregate across queries, so both interruption flags can be set by different queries.</p>
<p><code>decodeGPUKShortestPaths(readback, options)</code> converts typed-array readbacks into
<code>{queries, status}</code>. Each query has <code>{count, routes}</code>; each retained route has
<code>{vertices, edges, length, cost}</code> with padding removed. Query and route array order matches the
output layout. A zero-edge route has one vertex and an empty edge array; a query without routes
has <code>count: 0</code> and <code>routes: []</code>. The decoder checks array formats and minimum required lengths,
accepted counts and path lengths, nonnegative retained costs, and batch status scalars. Positive
infinity is allowed for a retained CPU route whose finite float64 cost overflowed on publication
as float32; counts and lengths distinguish it from padding. NaN and negative costs are rejected.
Typed arrays from other JavaScript realms are accepted when their scalar format matches.
Typed-array byte offsets are respected and spare capacity beyond the layout is ignored. Returned
route arrays are copies. The existing output has no version header; supply the dimensions used
for the computation.</p>
<p>Pass the five data arrays plus the four status arrays nested under <code>readback.status</code>. For the
direct CPU result, whose status fields are flat, use
<code>decodeGPUKShortestPaths({...result, status: result}, options)</code>. The decoder returns all three
flags as booleans and <code>iterationCount</code> as a number. It retains certified routes from incomplete
batches: inspect <code>status.converged</code>, <code>status.iterationLimitReached</code> and <code>status.candidateOverflow</code>
before treating decoded routes as a complete enumeration.</p>
<p>For Q queries, P spur slots, V vertices, E edge slots, C candidates and path length L, scratch is
<code>O(Q * P * (V + E) + Q * (C + k + P) * L)</code>. P is <code>min(L - 1, V - 1)</code>, at least one, and is one
when k is one. The CSR is copied per spur lane because every lane removes different edges.
Storage sizes are checked before nodes are built, including the two-slot shortest-path frontier.
Reduce query batch size or L when a binding is too large. The number of paths can still grow
exponentially; k and the candidate budget bound the enumeration, without enumerating all prefixes.</p>
<p>Costs follow float32 shortest-path relaxation, so very close costs can change order compared with
float64 and overflowing sums are impassable. Equal-cost membership follows reachability's acyclic
predecessor policy; the persistent pool breaks ties by CSR edge IDs. Global lexicographic tie order
is not guaranteed. Use exact binary fractions when testing exact agreement between CPU and GPU.</p>
<p>{/* lens:GPUKShortestPaths */}</p>
<p>::lens{scenario=gpu-network.k-shortest-paths rung=graph}</p>
<p>The directed diamond has two simple routes from vertex 0 to 3: CSR edges <code>[0, 2]</code> cost 2,
and <code>[1, 3]</code> cost 3. The lens retains padded route slots, per-query <code>counts</code> and all four
batch status ports. This synthetic run requests two routes with four vertex slots per route.</p>
<h2 id="gpukshortestpathsoperation">GPUKShortestPathsOperation</h2>
<p>The vector graph operation takes <code>KShortestPathsOperationProps</code>: graph value names replace input
and output views. <code>queryCount</code> declares the batch size; vertex count is inferred from CSR offsets. Status values remain
nested under <code>output.status</code>. Compile with <code>GPU_CONTRIBUTOR_OPERATION_LOWERING</code>.</p>
<h2 id="cpukshortestpathsoperation-and-computekshortestpathsoncpu">CPUKShortestPathsOperation and computeKShortestPathsOnCPU</h2>
<p>The CPU twin accepts the same operation props and publishes every output port with the same layout.
For direct calls, <code>computeKShortestPathsOnCPU({topology, sources, targets}, options)</code> takes typed arrays
and returns typed arrays; the four status fields are flattened on its result. It uses Yen's algorithm
with the shared float64 routing and acyclic predecessor helpers. Prefer it for small networks,
where GPU submission costs dominate.</p>
<p>The CPU version drops GPU-only node, route, device and command limits. An explicit
<code>maximumIterations</code> binds every shortest-search and zero-cost tie phase to at most 64 hops per
round. A synchronous CPU frontier stops before exceeding that allowance and publishes only
previously certified routes, with <code>converged = 0</code> and <code>iterationLimitReached = 1</code> on interruption.
Queries advance together: if any shortest-search lane interrupts, the entire still-active batch
stops before accepting that round's proposals. Routes from earlier completed rounds remain published.
Every active spur contributes its search certificate before a query-local path-length or
candidate-capacity stop is applied; a locally overlong path cannot hide another spur's interruption.
GPU workgroup scheduling can make progress through fewer hops per round. Without an explicit
CPU round budget, searches run to completion. Path length and candidate budgets retain the GPU's meaning.
It computes costs in float64 and rounds once at output, so near ties, overflow and predecessor
tightness may differ. Packed array lengths and CSR offset values are checked on the CPU.
The existing <code>CPUAlternativeRoutesOperation</code> and its reference GPU contributor remain available.</p>
<p>Independent oracle fixtures come from NetworkX <code>shortest_simple_paths</code>, using edge subdivision
to preserve distinct parallel edges. Regenerate with
<code>/usr/bin/python3 modules/analysis/test/oracles/k-shortest-paths.py</code>. Headless specs cover these
fixtures, 700-vertex networks, a 300-vertex path, graph reuse, input padding, ties and budget stops.</p>`};export{e as default};