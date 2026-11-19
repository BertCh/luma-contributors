var e={slug:`api/gpu-network/louvain`,title:`GPUNetworkLouvain`,html:`<h1 id="gpunetworklouvain">GPUNetworkLouvain</h1>
<p>See also the <a href="/luma-contributors/docs/api/gpu-network/operators">network operator guide</a>.</p>
<p><code>GPUNetworkLouvain</code> (<code>@luma-contributors/analysis/gpu-network</code>) optimizes weighted
Newman modularity with multilevel Louvain. Supply <code>topology</code> for sparse parallel
moving. The existing <code>weights</code> dense reference input remains available to 32 nodes;
its seeded sequential visitation and partition choices remain unchanged. Prefer
<code>CPUNetworkLouvainOperation</code> for small dense workloads.</p>
<h2 id="sparse-input-and-outputs">Sparse input and outputs</h2>
<p><code>topology</code> is the analysis <code>GPUNetworkTopology</code>: packed <code>uint32</code> offsets of length
<code>nodeCount + 1</code>, packed <code>uint32</code> neighbors, and aligned packed <code>float32</code> weights.
Directed arcs occur once. Undirected edges occur in both directions and self-loop
weights are doubled, as in the dense adjacency. Parallel arcs are summed; neighbor
order is unrestricted. Undirected reciprocal weight sums must agree within relative
<code>1e-6</code> (the precision of their independent float32 reductions).</p>
<p>Offsets start at zero, are nondecreasing, and end at or below the slot capacity.
Slots beyond the final offset and targets <code>>= nodeCount</code> are ignored, including their
weights. Accepted weights must be finite and nonnegative. Zero weights do not create
move candidates. Bad CSR, asymmetric undirected weights, invalid parameters or
nonfinite total mass fail closed: identity labels, NaN trace, zero validity. A graph
with zero mass returns identity labels, initial objective zero and convergence.</p>
<p><code>output.communities</code> has one <code>uint32</code> label per original vertex; isolates retain
their IDs. <code>objectiveTrace</code> has exactly
<code>1 + maximumLevels * maximumIterations</code> float32 rows: initial modularity followed by
one recomputed objective per completed parallel round; unused rows are NaN.
<code>summary</code> retains <code>[valid, converged, completedLevels, completedRounds]</code>.</p>
<p>Optional one-row <code>uint32</code> ports in <code>output.status</code> are <code>iterationCount</code>, <code>converged</code>,
<code>iterationLimitReached</code>, <code>invalidCount</code>, and <code>approximation</code>. <code>invalidCount</code> counts
invalid input graphs (zero or one). <code>approximation</code> is one: modularity optimization
is heuristic, never a global optimum certificate. Budget exhaustion publishes the
current partition and trace, with <code>converged = 0</code>, <code>iterationLimitReached = 1</code>.
A final stable moving level still needs another aggregation level to certify that
no supervertex can improve the partition.</p>
<p>All output views, including the summary and named status ports, must use distinct
buffers. Separate graph imports of the same default allocation also alias and are
rejected at construction. This keeps later summary/status passes from overwriting
community labels.</p>
<h2 id="parameters-and-limits">Parameters and limits</h2>
<p>Per-frame: input contents, resolution and minimum positive accepted gain. Use
<code>getGPUNetworkLouvainParameterValues({resolution: 1, minimumGain: 1e-5})</code> for the
two-row float32 <code>parameters</code> view, hosted by <code>GPUParameterBuffer</code>. Zero tolerance
is supported; a positive tolerance avoids moves driven by float32 cancellation.</p>
<p>Compile-time: node/slot capacities, directed mode, optional uint32 seed, output
ports, <code>maximumLevels</code> (default 8, 1–32) and <code>maximumIterations</code> per level (default
16, 1–64). Sparse node count is 1–1,073,741,823 and CSR capacity is
0–1,073,741,823 slots. These uint32 addressing bounds account for four strength
words per node and two two-word weak-neighbor records per CSR slot. Pair keys use
separate source and destination words, sorted stably in lexicographic order;
there is no degree or community-size cap.</p>
<p>Device <code>maxStorageBufferBindingSize</code> further restricts both counts: the largest
scratch binding needs 16 bytes per node or CSR slot. A 128 MiB binding limit
therefore permits at most 8,388,608 nodes and 8,388,608 CSR slots independently.
The binding checks run before any command resources are created. Total GPU memory
usage includes several such buffers, so fitting one binding does not guarantee
that the complete graph fits device memory.</p>
<p>The constructor validates the estimated command-node count against 65,536, so not
every combination of level/round maxima is accepted. The conservative bound is
<code>352 + maximumLevels * (320 * maximumIterations + 208) + 1</code>, or 42,977 for the
default 8 levels and 16 rounds. The bound includes the second stable radix pass
needed when both vertex IDs no longer fit one uint32 sort key. Scratch is linear
in vertices and edges for a fixed level/round budget. Recorded command graphs at the
uint32 bounds contain 35,283 nodes for the default schedule and 53,483 for 16 levels
and 12 rounds (estimated at 65,121). These metadata counts justify the 65,536 budget;
they measure graph size, not GPU execution time.</p>
<p>Round/level controls use GPU counters and the bound lengths of runtime budget views.
The trace initialization dispatch has a fixed 2,049-row upper bound and checks the
bound trace length. Increasing the schedule therefore reuses the same shader and
pipeline variants. Caller binding checks include the prefix from the required
256-byte-aligned binding offset, and run before any resource declaration.</p>
<h2 id="algorithm-and-upstream-evaluation">Algorithm and upstream evaluation</h2>
<p>Each round uses published <code>GPUSort</code>, <code>GPUScan</code>, <code>GPUGather</code>, <code>GPUReduction</code>, and the
analysis fixed-tree segmented reducer. Sorted <code>(vertex, neighborCommunity)</code> records
provide connection weights. Each vertex proposes its best strictly improving gain;
equal gains choose the lowest community label. Proposals reserve both affected
communities. Only a proposal winning both reservations commits, so concurrent moves
have disjoint community totals and their modularity gains do not conflict. The
lowest vertex ID wins conflicts, or the lowest <code>vertex ^ seed</code> when seeded. This
seed is a deterministic priority, not a random permutation or the dense Philox order.</p>
<p>When local moving stabilizes after changes, a sort and segmented reduction creates
the weighted quotient graph, including self-loops. Original labels are composed
through levels. Each round/aggregation uses GPU dispatch gates. Installed alpha.16
sort/reduction nodes omit public workload geometry; a domain adapter observes their
public <code>ComputePass.dispatch</code> calls and GPU-gates the exact dimensions, without
private metadata, submission, readback, or replacement upstream kernels. Its resources
belong to the compiled command and are released on destruction.</p>
<p>Evaluation of the installed <strong>published <code>@luma.gl/graph</code> 10.0.0-alpha.16</strong>:
<code>GPUGraphAlgorithmsModularityOptimization</code>, exported from
<code>@luma.gl/graph/gpu-graph-algorithms</code>, accepts topology and GPUVector outputs,
resolution/minimumGain as construction-time numbers, up to 1,024 individual accepted
moves, and optional valid/converged scalars. Its API documentation explicitly says
it performs <strong>single-level</strong> local moving, accepts <strong>one globally best move</strong> per
round, uses unordered float atomic sums, and performs <strong>no graph coarsening</strong>.
It therefore cannot replace this multilevel, parallel, per-frame-parameter contract.
No upstream source or private API is imported here. The installed package version
is alpha.16; this comparison describes that installed version.</p>
<h2 id="cpu-twin-and-verification">CPU twin and verification</h2>
<p><code>GPUNetworkLouvainOperation</code> and <code>CPUNetworkLouvainOperation</code> accept the same graph
value names, including <code>topology: {offsets, neighbors, weights}</code> and optional status
ports. <code>computeNetworkLouvainOnCPU({topology, parameters}, options)</code> is the pure
float64 twin. It uses the same proposal/reservation/coarsening schedule but drops
GPU pair-key, scratch and command-budget caps. Near ties, overflow, subnormals and
threshold decisions can differ from GPU float32. Partitions need not match dense
sequential Louvain or NetworkX on general graphs.</p>
<p><code>test/oracles/network-louvain-sparse.py</code> generates NetworkX Louvain partitions and
scores, independently cross-checked with NumPy's modularity matrix. Fixtures cover
96 weighted vertices, 48 directed vertices, parallel arcs, self-loops, padding,
isolates, and a graph that needs coarsened moves. Node specs also check exact ties,
round/level exhaustion, the largest supported shape, command estimates and generated
kernel audits. The headless spec compares every CPU/GPU output and the NetworkX
quality, plus zero/invalid/empty-slot cases. Additional headless cases compare every
output bit against the retained packed-key reference below the old limits, using
mixed-magnitude duplicate arcs with more than 256 contributions per pair and gains
near tolerance boundaries. They cover the 65,535/65,536/65,537-node sort transition
with duplicates, self-loops and both graph directions, and a 131,072-node weighted
planted graph whose partition improves after quotient aggregation. GPU execution must be verified by the
landing environment; Chromium is unavailable in the implementation sandbox.</p>
<p>{/* parameters: GPUNetworkLouvain generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h2 id="gpunetworklouvain-parameters"><code>GPUNetworkLouvain</code> parameters</h2>
<ul>
<li>Packer: <code>getGPUNetworkLouvainParameterValues(settings?)</code>, writes the <code>float32</code> parameter view.</li>
</ul>
<p>Settings of <code>getGPUNetworkLouvainParameterValues</code>:</p>
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
<td><code>resolution</code> (optional)</td>
<td><code>number</code></td>
<td>-</td>
</tr>
<tr>
<td><code>minimumGain</code> (optional)</td>
<td><code>number</code></td>
<td>-</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>`};export{e as default};