var e={slug:`api/gpu-network/expansion`,title:`Network structure, routes and spatial topology extensions`,html:`<h1 id="network-structure-routes-and-spatial-topology-extensions">Network structure, routes and spatial topology extensions</h1>
<p>These contributors declare work in a <code>GPUCommandGraph</code>. They do not submit GPU work or read
results back. Outputs are caller-owned packed <code>GraphDataView</code>s, and inputs remain GPU-resident.</p>
<h2 id="edge-betweenness">Edge betweenness</h2>
<p><code>GPUNetworkCentrality.output.edgeBetweenness</code> accepts a <code>float32</code> column with one row per
forward CSR edge slot. It can be requested independently of node betweenness. It uses the same
pivots, lane batching, f32 tight-edge test, iteration bounds and <code>betweennessScale</code> as node
betweenness. The existing convergence output covers both dependency phases. A value is the
sum of shortest-path dependencies through that directed edge. Parallel CSR edges are distinct
paths. For an undirected graph listing both directions, each directed slot equals the conventional unnormalized
undirected edge centrality when all nodes are pivots; if combining paired slots, divide their
sum by two. Zero-cost edges are excluded from tight-path counting, as in the original
node implementation. Preserve a slot-to-source-edge mapping if CSR sorting changes edge order.</p>
<h2 id="directed-components-and-undirected-resilience">Directed components and undirected resilience</h2>
<p>{/* lens:GPUNetworkStructure */}</p>
<p>::lens{scenario=gpu-network.network-structure rung=graph}</p>
<p><code>GPUNetworkStructure</code> accepts <code>topology</code> with <code>offsets</code> and <code>neighbors</code>; weights are ignored.</p>
<table>
<thead>
<tr>
<th>Output</th>
<th>Format and meaning</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>strongComponents</code></td>
<td><code>uint32</code>, one per node, minimum node ID in the strongly connected component</td>
</tr>
<tr>
<td><code>bridges</code></td>
<td><code>uint32</code>, one flag per CSR slot</td>
</tr>
<tr>
<td><code>articulationPoints</code></td>
<td><code>uint32</code>, one flag per node</td>
</tr>
<tr>
<td><code>biconnectedComponents</code></td>
<td><code>uint32</code>, one per slot, minimum slot in the edge block</td>
</tr>
</tbody>
</table>
<p>For resilience outputs, <code>reverseEdgeSlots</code> pairs each CSR slot with its exact reverse slot.
The caller must supply an involution with reversed endpoints. Pair parallel edges separately:
two parallel edges form a block and neither is a bridge. A self-loop pairs with itself and is
a singleton block. Invalid neighbor IDs are ignored; CSR offsets must remain monotone and
within capacity. Isolated nodes are separate SCCs and are not articulation points.</p>
<p><code>strongComponents</code> uses an exact mutual-reachability kernel with O(V²) scratch and O(V(V+E))
work, for at most 1024 nodes and 65536 slots. For larger graphs, use
work, for at most 1024 nodes and 65536 slots. For larger directed graphs, use
<code>GPUNetworkStronglyConnectedComponents</code> below.</p>
<p>The bounds of 1024 nodes and 65536 slots apply to the resilience outputs and to the
SCC path. Resilience uses an iterative Tarjan traversal in one GPU invocation. These
are exact bounded subgraph implementations; large-graph throughput and a lower-level generic graph
adapter for resilience remain future optimization work.</p>
<h2 id="large-graph-strongly-connected-components">Large-graph strongly connected components</h2>
<p><code>GPUNetworkStronglyConnectedComponents</code> takes forward CSR <code>topology</code> and caller-owned
<code>output.strongComponents</code> (<code>uint32</code>, one minimum vertex ID per SCC member). It uses O(V)
scratch: GPU trimming, minimum-colour propagation and backward marking within each colour class.
It does not require reverse adjacency. Weights are ignored, invalid neighbour IDs are skipped,
self-loops leave singleton SCCs intact, and isolated vertices label themselves. Offsets must be
monotone and within slot capacity; unused slots beyond the final offset are ignored.</p>
<p><code>maximumIterations</code> is a compile-time state-machine step budget, default 256, range 1–4096.
<code>output.status</code> requires three distinct caller-owned single-row <code>uint32</code> views: <code>converged</code>,
<code>iterationCount</code> (executed steps), and <code>iterationLimitReached</code>. Inspect convergence before using
labels: an exhausted budget publishes <code>0xffffffff</code> for <strong>every</strong> vertex, including resolved ones,
with <code>converged = 0</code> and <code>iterationLimitReached = 1</code>. Deep cycles, long DAGs and chains of SCCs
may need a larger budget. Finished vertex phases are GPU dispatch-gated; there is no readback.
All buffer shapes and the step budget are compile-time; there are no numeric per-frame props.</p>
<p>Node and CSR slot counts must fit uint32 indexing (maximum 4294967294) and each packed binding
must fit the device's storage binding limit. Two vertex-sized scratch columns replace the exact
reference's quadratic reachability table. <code>computeNetworkStructureOnCPU({offsets, neighbors})</code>
is the independent iterative Tarjan reference for the same minimum-ID label layout; it has no
GPU step budget by default. Its optional second argument <code>{maximumIterations}</code> selects budgeted
SCC phases and returns <code>strongComponentsStatus</code> beside the labels. <code>CPUNetworkStructureOperation</code>
also accepts <code>strongComponentsStatus</code> and returns every requested scalar. Exact SCCs report
<code>converged=1</code>, <code>iterationCount=1</code> (one Tarjan traversal), and <code>iterationLimitReached=0</code>. When the
caller explicitly selects parallel SCCs (or supplies auto/budget controls selecting the large-graph
route), the CPU operation uses the trim/colour/backward/finalize state machine with that step budget.
With no mode or budget supplied, CPU SCCs remain exact Tarjan at every accepted size; a 1025-vertex
chain does not acquire the GPU default budget just by crossing the GPU exact-route limit.
The new class keeps this iterative SCC contract separate from
<code>GPUNetworkStructure</code>'s bounded resilience outputs. The algorithm is adapted from the user's
luma.gl fork at <code>e0ed33791</code>, independently implemented for WebGPU and inspired by cuGraph.</p>
<h2 id="large-graph-directed-components">Large-graph directed components</h2>
<p>{/* lens:GPUNetworkStronglyConnectedComponents */}</p>
<p>::lens{scenario=gpu-network.network-strongly-connected-components rung=result caption="The cycle 0↔1 shares label 0; the outgoing singleton and isolated node keep labels 2 and 3."}</p>
<p>::lens{scenario=gpu-network.network-strongly-connected-components rung=graph caption="Follow GPU-gated trim, colour propagation, backward marking and convergence publication."}</p>
<p>The result selects the four <code>strongComponents</code> labels as its primary output. The full-size
table retains <code>converged</code>, <code>iterationCount</code> and <code>iterationLimitReached</code> in their separate
one-row status domain; the thumbnail summarizes the labels. The graph rung shows the command
graph separately. CSR inputs and all status readbacks remain in the record.</p>
<p><code>GPUNetworkStronglyConnectedComponents</code> takes forward CSR <code>topology</code>; weights are ignored.
It uses O(V) scratch and vertex-parallel outgoing-neighbour visits, with GPU gates selecting
trim, minimum-colour propagation, backward marking and component finalization. Each exact
component label is its minimum vertex ID. Invalid neighbors and unused CSR slots are ignored;
isolated vertices and self-loops are supported. CSR offsets must be monotone and within capacity.</p>
<p>Caller-owned <code>output.strongComponents</code> holds one <code>uint32</code> per vertex. All three scalar <code>uint32</code>
views in <code>output.status</code> are required: <code>converged</code>, <code>iterationCount</code>, <code>iterationLimitReached</code>.
<code>maximumIterations</code> is a compiled state-machine step budget (default 256, range 1–4096), not
an edge-traversal count. Deep cycles, long chains and chains of components can exhaust it.
On exhaustion every label is <code>0xffffffff</code>, including labels resolved earlier; <code>converged</code> is 0
and <code>iterationLimitReached</code> is 1. Inspect status before using labels. <code>iterationCount</code> counts
executed state-machine steps, and finished vertex passes are gated off. The graph contains
<code>2 + 6 * maximumIterations</code> command nodes; storage bindings must fit the device. The old
1024-vertex bound of <code>GPUNetworkStructure</code> does not apply to this contributor.</p>
<p>The synthetic lens includes a two-node cycle, an outgoing singleton with a self-loop, an
isolated vertex and an unused CSR slot. Its hand-derived labels are <code>[0, 0, 2, 3]</code>. This is a
correctness illustration, not a large-graph throughput measurement. Its recorded result is
pending the GPU capture run (<code>VITE_CAPTURE_LENS_RECORDS=1</code>); no result is fabricated here.</p>
<h2 id="alternative-routes">Alternative routes</h2>
<p>{/* lens:GPUAlternativeRoutes */}</p>
<p>::lens{scenario=gpu-network.alternative-routes rung=graph}</p>
<p><code>GPUAlternativeRoutes</code> accepts weighted <code>topology</code>, equally sized <code>sources</code> and <code>targets</code>,
<code>routeCount</code> (1–32), and <code>maximumExpansions</code> (default 100000). It returns sorted simple edge
paths: <code>vertices</code> and <code>edges</code> are query × routeCount × nodeCount; <code>lengths</code> counts vertices;
<code>costs</code> is infinity for missing routes; <code>counts</code> and mandatory <code>complete</code> have one row per query.
Padding is <code>0xffffffff</code>. Distinct parallel edges define distinct routes. Ties follow
lexicographic CSR edge order. Source equal to destination has one zero-edge path.</p>
<p>Queries execute independently. A bounded min-heap frontier enumerates simple paths by
nonnegative accumulated cost. <code>maximumFrontierSize</code> bounds retained frontier entries;
<code>maximumExpansions</code> bounds popped states. Either exhausted budget leaves <code>complete</code> zero.
A zero flag means retained paths are candidates, not certified k-shortest routes. Inspect it
before reporting alternatives as optimal. See <a href="/luma-contributors/docs/api/gpu-network/alternative-routes">Alternative routes</a>
for tie ordering, capacities and scratch-memory limits. Negative, NaN and infinite edge costs
and overflowed f32 accumulated costs are impassable.</p>
<h2 id="h3-identities-and-cell-flows">H3 identities and cell flows</h2>
<p><code>GPUH3CellIdentities</code> (spatial-analysis entry point) takes little-endian <code>uint32x2</code> cells and
<code>operation: 'directed-edges' | 'vertices'</code>. The output has six rows per input cell. Directed
edge slot <code>direction - 1</code> stores the H3 edge ID; the deleted pentagon direction is zero.
Vertex slots match H3 <code>cellToVertexes</code>, with pentagon slot five zero. Invalid cells produce
zeros. Vertex ownership is the numerically smallest incident cell. Edge and vertex IDs are
not cell IDs and cannot be passed to cell-only APIs. All key arithmetic is integer-based.</p>
<p><code>addCellFlowTopologyRecipe</code> accepts distinct same-resolution cells, origin/destination row
indices, and nonnegative finite values. It retains the original pair table, sums all valid
incoming/outgoing flows, and separately sums immediate-neighbor directed-edge flows. Repeated
pairs are accumulated in input order. Nonneighbor and self flows remain in the pair table and
cell totals; no route is inferred. It also returns two- or three-vertex shared boundaries in
longitude/latitude, including H3 face distortion. Longitude wraps at the antimeridian and
coordinates have the existing f32 H3 precision limit, especially at resolutions 14–15.</p>
<p><code>validity</code> and <code>status.invalidCount</code> expose rejected rows. Sum(outgoing) and sum(incoming) equal the
valid pair sum up to f32 accumulation error; the neighboring-edge sum may be smaller. The
current direct-scan recipe costs O(cellCount × pairCount); preaggregate large pair tables.</p>
<h2 id="lcss-trajectory-similarity">LCSS trajectory similarity</h2>
<p><code>GPUTrackSimilarity.output.longestCommonSubsequence</code> is matched length divided by the shorter
densified track length. It lies in [0,1], with NaN for skipped/empty/invalid/too-long pairs and
the existing pair status flags. <code>lcssTolerance</code> is a nonnegative finite compile-time point
distance threshold (default 1), in spatial-context units. <code>lcssBandRadius</code> optionally restricts
eligible matches with the existing length-adjusted Sakoe–Chiba convention; unmatched vertices
remain allowed. Densification and <code>maximumFrechetVertexCount</code> apply as for DTW. Matching uses
an anti-diagonal GPU dynamic program, including carried boundary rows for long tracks.</p>
<h2 id="facility-allocation">Facility allocation</h2>
<p><code>GPUBoundedFacilityAllocation</code> is exported by spatial analysis. <code>operation: 'p-center'</code>
enumerates all facility subsets and exactly minimizes the maximum nearest-facility cost in the
supplied f32 matrix. <code>operation: 'capacitated-p-median'</code> evaluates the same subsets using
indivisible, demand-order nearest-feasible assignment, then prefers fewer unassigned rows and
lower weighted cost. <strong>The capacitated mode is a heuristic, and failure is not an infeasibility
certificate.</strong> Change demand order explicitly to investigate sensitivity.</p>
<p>For example, two unit demands with costs <code>[1, 2]</code> and <code>[1, 100]</code>, and two facilities with
unit capacities, produce greedy cost 101 in that demand order. Exhaustive assignment gives
cost 3. The test suite checks this gap and the zero optimality certificate explicitly.</p>
<p>Inputs are demand-major dense <code>costs</code>, <code>demand</code>, <code>capacities</code>, <code>facilityCount</code>, and <code>selectCount</code>.
Bounds are 32 candidate facilities, 8 selected facilities and 65536 subsets. Capacitated mode
accepts 4096 demand rows. P-center accepts a dense cost matrix up to the device's storage binding
limit: <code>floor(maxStorageBufferBindingSize / (4 * facilityCount))</code> demand rows, also subject to
uint32 indexing. For 32 facilities and a 128 MiB binding this is 1,048,576 rows; this is a storage
limit, not a throughput guarantee.</p>
<p>P-center scores demand rows cooperatively within each subset and publishes assignments in
parallel. Both operations use hierarchical winner reductions with the original subset order as
the final tie-breaker. Capacity assignment remains sequential within each subset to preserve its
fixed demand-order behavior. Work still grows with <code>choose(facilityCount, selectCount)</code>; the
parallel implementation does not remove the cost of exhaustive subset search.
Costs must be nonnegative finite to be reachable. Demand and capacity must be nonnegative
finite. The p-center operation ignores capacities and uses unweighted maximum cost.</p>
<p>Outputs are selected flags, demand assignments (<code>0xffffffff</code> if unassigned), scalar objective,
and three status words: unassigned-row count, valid-input flag, global-optimum certificate.
The certificate is always zero for capacitated mode. Objective is infinity if any row is
unassigned or an input is invalid. Inspect both feasibility and status before consuming results.</p>
<h2 id="verification-and-reference-contracts">Verification and reference contracts</h2>
<p>The focused suite <code>test/geospatial-expansion/network-expansion.spec.ts</code> runs real WebGPU kernels.
It compares H3 keys with h3-js 4.4 across all base cells and their first-level children plus
resolutions 0–15; checks directed SCCs and undirected parallel-edge blocks; verifies route order
and truncated-search flags; exercises edge-only centrality; checks p-center and capacities; and
crosses the 256-row LCSS workgroup strip boundary. Additional cases compare eight seeded
undirected graphs with NetworkX, check feasible and infeasible capacity outcomes, and compare
cell-flow boundaries at pentagons and the antimeridian against H3 while checking flow conservation.
LCSS also uses a NetworkX longest-chain oracle over the DAG of eligible match pairs, independent
of the GPU dynamic-program recurrence. The node companion parses generated WGSL
and verifies the eight-storage-binding limit.</p>
<p><code>test/oracles/network-expansion.py</code> generates independent fixtures using NetworkX 3.7 and
Python <code>itertools</code> exhaustive facility subsets and capacitated assignments. Reference conventions are unnormalized directed
edge betweenness, minimum-ID SCC labels, simple paths by total edge cost, and unweighted
p-center. These bounded correctness fixtures are not throughput benchmarks or proof of broad
numerical parity on arbitrary data. The capacitated heuristic has no approximation guarantee.</p>
<h2 id="network-disruption-recipe"><code>network-disruption-recipe</code></h2>
<p>{/* lens:addNetworkDisruptionRecipe */}</p>
<p>::lens{scenario=gpu-network.network-disruption-recipe rung=graph}</p>
<p>The executable synthetic directed-diamond scenario captures the workflow outputs. See <a href="/luma-contributors/docs/api/gpu-spatial-analysis/expansion-recipes">recipe contracts</a> for matrix orientation, convergence and cutoff semantics.</p>
<h2 id="facility-scenario-recipe"><code>facility-scenario-recipe</code></h2>
<p>{/* lens:addFacilityScenarioRecipe */}</p>
<p>::lens{scenario=gpu-network.facility-scenario-recipe rung=graph}</p>
<p>The executable synthetic directed-diamond scenario captures the workflow outputs. See <a href="/luma-contributors/docs/api/gpu-spatial-analysis/expansion-recipes">recipe contracts</a> for matrix orientation, convergence and cutoff semantics.</p>
<h2 id="track-quality-and-route-conformity">Track quality and route conformity</h2>
<p>{/* lens:addTrackQualityRecipe */}</p>
<p>::lens{scenario=gpu-network.track-quality-recipe rung=result}</p>
<p>::lens{scenario=gpu-network.track-quality-recipe rung=graph}</p>
<p><code>addTrackQualityRecipe</code> retains raw observation order and timestamps. The executable projected
four-observation example includes a repeated timestamp and a long time gap, and captures the
actual GPU quality flags, step speeds and track summaries. The two threshold rows are maximum
speed and maximum elapsed time. Optional reference-route similarity and map matching retain
separate failure evidence. See <a href="/luma-contributors/docs/api/gpu-spatial-analysis/expansion-recipes#track-quality-and-route-conformity">the recipe contract</a>
for timestamp encodings, flag bits and bounded route-search semantics.</p>`};export{e as default};