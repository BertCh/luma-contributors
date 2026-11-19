var e={slug:`api/gpu-network/operators`,title:`GPU Network operators`,html:`<h1 id="gpu-network-operators">GPU Network operators</h1>
<h3 id="gpunetworkreachability"><code>GPUNetworkReachability</code></h3>
<p>Compute travel-time costs, bands and predecessors from one start intersection over a directed road network.</p>
<p>::lens{scenario=gpu-network.network-reachability rung=result caption="Follow travel times across 1,199 intersections; the farthest is about 262.7 seconds from the start."}</p>
<p>Maximum: <code>output.costs</code> has at most <code>getGPUNetworkReachabilityMaximumNodeCount(device)</code> (16777216 on a default device) rows, <code>laneCount * nodeCount</code>, because the frontier queues take 8 bytes per row.</p>
<p>Single- or multi-source shortest-path costs over a directed CSR road network with isochrone bands,
band counts, predecessors, and a convergence flag. Relaxation is chaotic Bellman-Ford over a compact
GPU frontier queue with a round-stamped visited set: each of the <code>maximumIterations</code> rounds is one
indirect dispatch sized by the previous round's pushes, so converged frames dispatch nothing and
nothing is read back. Within a round each workgroup chains up to <code>localIterations</code> (default 32)
hops through workgroup memory, so a path of <code>h</code> hops needs about
<code>ceil(h / (0.7 * localIterations)) + 4</code> rounds (the budget of <code>recommendReachabilityIterations</code>,
which <code>GPUNetworkServiceAreas</code> and <code>GPUNetworkIsochrones</code> share); costs
are identical for every setting. Predecessors are cycle-safe: the smallest strictly cheaper tight
in-neighbor, or across equal-cost (zero-weight) edges the smallest tight in-neighbor one BFS level
closer to a strict entry or a source, found in up to <code>maximumTieIterations</code> extra rounds (default 4).
<code>maximumIterations</code> counts rounds, one graph node each, not hops. With the default <code>localIterations</code>,
a budget of about <code>h / 22</code> plus a few rounds suffices; 48 covers the demo networks. The optional
<code>output.unresolvedCount</code> (one <code>uint32</code> row, requires <code>output.predecessors</code>) counts reached nodes that still have no
predecessor because the tie phase stopped at <code>maximumTieIterations</code>; 0 means every reached node has its
tie-rule predecessor, and <code>status.converged</code> is unchanged.
<code>recommendReachabilityIterations()</code> suggests a first-guess <code>maximumIterations</code> from the network size, and
<code>adaptReachabilityIterations()</code> grows or shrinks it from the last frame's <code>output.status.iterationCount</code> and
<code>output.status.converged</code> readback. <code>startCount</code> is a number (compiled in) or a one-row view (per frame).
Build CSR on the GPU with <code>GPUCOOToCSR</code>; a reverse CSR gives "cost to reach" isochrones.
The contributor and both operation forms capture named constructor budgets and port/status maps
once, including inherited getters. Later props mutation cannot raise the validated round limit;
<code>activeIterations</code> remains a per-frame buffer control clamped to that limit.
Each round reads its index from its own 256-byte dispatch slot (<code>[x, y, z, round]</code>), never from the
WGSL text, so every round of a phase shares one pipeline: one reachability compiles one relax and one
tie-level pipeline at any <code>maximumIterations</code>, and so do the label rounds of <code>GPUNetworkServiceAreas</code>.</p>
<p>The optional <code>output.status.relaxationCount</code> and <code>output.status.reachedEdgeCount</code> (two <code>uint32</code> rows each, a 64-bit
count as low and high words) are an inflation probe. <code>relaxationCount</code> is every edge scanned by every relax visit,
including the chained hops and all lanes; <code>reachedEdgeCount</code> is the out-degree sum of every node with a finite final
cost, the edges a label-setting Dijkstra scans once. <code>getGPUNetworkReachabilityInflation(relaxationCount, reachedEdgeCount)</code>
returns their ratio (1 is work-efficient, NaN when nothing was reached). Delta-stepping or near-far is only worth
building when the measured inflation exceeds 3. The counters add one atomic per workgroup per round, so leave them off in production.</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUNetworkReachability</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  topology, startNodes: starts.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">importToGraph</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(graph), costLimit: costLimit.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">importToGraph</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(graph),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  costBreaks: costBreaks.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">importToGraph</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(graph), maximumIterations: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">48</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  output: {costs, bands, bandCounts, status: {converged}}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span></code></pre>
<h4 id="how-it-computes">How it computes</h4>
<p>{/* lens:GPUNetworkReachability */}</p>
<p>::lens{scenario=gpu-network.network-reachability rung=datum bind=page datum=78 caption="Inspect the start neighbour: its final travel time is about 5.1 seconds."}</p>
<p>::lens{scenario=gpu-network.network-reachability rung=formula}</p>
<p>::lens{scenario=gpu-network.network-reachability rung=iterations caption="Watch key frames through round 37; the hatched replay uses a smaller round limit."}</p>
<p>::lens{scenario=gpu-network.network-reachability rung=kernel}</p>
<p>::lens{scenario=gpu-network.network-reachability rung=graph}</p>
<h3 id="gpunetworkcentrality"><code>GPUNetworkCentrality</code></h3>
<p>Measure outward closeness of sampled pivot intersections from weighted road searches over the network and its transpose.</p>
<p>::lens{scenario=gpu-network.network-centrality rung=result caption="Compare 8 sampled pivots; each reaches 1,199 intersections, and the highlight has the highest closeness."}</p>
<p>Radius-bounded betweenness and closeness centrality of a directed weighted CSR network, exact from every
node as a pivot or estimated from a sample. It needs the network and its transpose (<code>topology</code> and
<code>reverseTopology</code>, the same edges reversed with bitwise-identical weights, for example from <code>GPUCOOToCSR</code>;
an undirected network that lists both directions with equal weights may pass one topology twice) and the
<code>pivotNodes</code> to measure from. Outputs (at least one): <code>betweenness</code> (per node), <code>closeness</code>,
<code>harmonicCloseness</code> and <code>reachedCounts</code> (per pivot), and <code>status.converged</code>.</p>
<p>Optional one-row <code>uint32</code> <code>output.status.iterationLimitReached</code> records any reachability,
path-count or dependency phase that exhausted its round budget. It can be requested without
<code>converged</code> or <code>zeroWeightCycle</code>. A detected zero-weight cycle alone clears <code>converged</code> and sets
<code>zeroWeightCycle</code>, while leaving <code>iterationLimitReached = 0</code>; more rounds cannot fix that cycle.
If different batches encounter both conditions, both diagnostic flags can be 1. A zero limit flag
alone does not certify defined betweenness; request <code>converged</code> or <code>zeroWeightCycle</code> as well. The contributor
publishes incumbent centralities on a round-limit exit and rewrites every requested port on each
encoding. The graph operation has the same ports; the exact CPU twin never sets
<code>iterationLimitReached</code>, including on a zero-cycle result. Status views must use separate physical
buffers from inputs and other outputs, even when imported with different graph IDs.</p>
<p>Pivots run <code>laneCount</code> at a time (default <code>recommendLaneCount()</code>, about 1M expanded nodes per batch; closeness
outputs are bit-identical for every lane count). Each batch runs one lane-expanded <code>GPUNetworkReachability</code>
(<code>costLimit</code> is the radius, <code>maximumIterations</code> and <code>localIterations</code> as there; the defaults are 64 and 16).
Closeness comes straight from those costs: one workgroup per lane reduces the reached count, cost sum and
harmonic sum in a fixed order. Betweenness adds Brandes' (2001) accumulation on the tight edges, those with
<code>cost[u] &#x3C; cost[w]</code> and <code>cost[u] + weight == cost[w]</code> in f32, exactly as the relaxation computes it. The
sigma phase (path counts) is a push-ready, pull-sum frontier schedule: a node is queued when its last tight
predecessor releases it through an <code>atomicSub</code> on a counter that starts at its tight in-degree, then sums
its predecessors' sigma in reverse-CSR order. The delta phase mirrors it from the nodes without tight
out-edges, summing <code>sigma(v) / sigma(w) * (1 + delta(w))</code> in forward-CSR order. Each batch adds the
lane-ordered sum of its deltas (skipping each lane's own pivot) times <code>betweennessScale</code> to the output,
zeroed once, so results are deterministic for a given <code>laneCount</code>. Both phases use <code>maximumIterations</code>
rounds each, one pipeline for every round and batch. Scratch memory is the frontier state, three
<code>laneCount * nodeCount</code> word arrays, and a packed copy of both CSRs made once per encoding.</p>
<p>Compile cost: every round of every batch is an unrolled command node, so the graph holds about
<code>batchCount * phases * min(maximumIterations, nodeCount)</code> nodes, where <code>phases</code> is 1 for the reachability search
alone and 3 with betweenness (search, sigma and delta), plus a few nodes per batch. The exact count is the
read-only property <code>estimatedCommandNodeCount</code>, and the constructor throws above 32,768 nodes (for example 64
batches x 3 phases x 1,000 rounds would be about 192,000). Exact all-pivot betweenness on graphs above roughly
15,000 to 20,000 nodes with the default <code>laneCount</code> now exceeds the cap, so large graphs need sampled pivots, a
lower <code>maximumIterations</code> or a higher <code>laneCount</code>. A graph continuation (re-entering the graph until
convergence, which would remove the need to unroll rounds) would remove the limit, but the published luma.gl
compiler does not provide one yet. Per-dispatch gating exists but does not reduce the node count.</p>
<p>Conventions (oracle <code>test/oracles/network-centrality.py</code>, networkx 3.7):</p>
<ul>
<li>Betweenness equals <code>networkx.betweenness_centrality(G, weight='weight', normalized=False)</code> on a <code>DiGraph</code> when
every node is a pivot, and <code>betweenness_centrality_subset(G, sources=pivots, targets=list(G), ...)</code> for a
sample, times <code>betweennessScale</code>. Endpoints are not counted and no 1/2 factor is applied (halve the result for
an undirected graph that lists both directions). For estimates from a sample use
<code>betweennessScale: nodeCount / pivotCount</code>.</li>
<li>Distances run outward from the pivots. networkx's <code>closeness_centrality</code> and <code>harmonic_centrality</code> on a
<code>DiGraph</code> use incoming distances, so the oracle compares with <code>closeness_centrality(G.reverse(), u=pivot, wf_improved=False)</code> and <code>harmonic_centrality(G.reverse(), [pivot])</code>. Swap <code>topology</code> and <code>reverseTopology</code>
to get networkx's incoming convention. <code>closeness</code> is <code>(r - 1) / S</code> for <code>r</code> reached nodes (the pivot
included) and cost sum <code>S</code>, 0 when <code>S</code> is 0; the Wasserman-Faust factor is not applied.</li>
<li><code>costLimit</code> (a one-row per-frame view) is an inclusive radius: nodes beyond it and paths longer than it
are not counted. networkx has no radius option; the oracle uses an explicit Brandes over a cutoff search
that equals networkx when the radius is infinite.</li>
<li>Edges of weight exactly zero between two nodes of equal cost are tight and
are counted like any other edge on a shortest path: on <code>s -1-> a -1-> m -0-> t -1-> u</code> the pivot <code>s</code> gives
<code>a</code>, <code>m</code>, <code>t</code> the dependencies 3, 2, 1, as networkx does. A positive weight below half an ulp of the cost,
which float32 absorbs (1 at a cost of 2^24, 3 mm at 100 km in metres), is never tight: the edge is left out of
the path counts, a node reached only across it is a root with one path, and the run still converges.
Counting zero-weight edges needs the zero-weight tight edges to form no
cycle. With a cycle (an undirected zero-weight edge listed in both directions is one) the number of shortest
paths is not defined (networkx requires positive weights for the same reason): <code>status.zeroWeightCycle</code> is 1,
<code>status.converged</code> is 0 and the betweenness values are incomplete. Give such edges a positive weight or merge
their end nodes. Self-loops are never on a shortest path. Harmonic closeness skips nodes of cost 0, and
parallel edges count as separate paths.</li>
<li>Out-of-range pivots are ignored and their per-pivot rows are 0. <code>status.converged</code> is 1 when every batch's
reachability and, with <code>betweenness</code>, both phases finished within <code>maximumIterations</code> and ordered every
reached node.</li>
<li>Path counts (sigma) are a float32 mantissa with a separate base-2 exponent: exact below 2^24 paths per pair,
rounded to 24 bits beyond, and never out of range (the corner-to-corner count of a 67 x 67 unit lattice
exceeds the float32 maximum). Delta rounds once per division (within 2.5 ULP in WGSL), addition
and multiplication and once per added term, all terms positive, so a node's relative error is at most
<code>h (7 + K) 2^-24</code> for a longest tight chain of <code>h</code> hops and largest tight out-degree <code>K</code>, plus
<code>(pivotCount + 2) 2^-24</code> for the sum over pivots.</li>
<li>The sigma and delta phases run one hop per round (<code>localIterations</code> applies to the search only), so
a node's sum reads its predecessors' values only after a dispatch boundary; each phase needs about
<code>h + 1</code> of the <code>maximumIterations</code> rounds for a longest tight chain of <code>h</code> hops.</li>
</ul>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> centrality</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUNetworkCentrality</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  topology,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  reverseTopology,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  pivotNodes: pivots.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">importToGraph</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(graph),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  costLimit: radius.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">importToGraph</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(graph),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  betweennessScale: nodeCount </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">/</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8"> pivotCount,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  output: {</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    betweenness: betweenness.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">importToGraph</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(graph),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    closeness: closeness.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">importToGraph</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(graph),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    status: {converged: converged.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">importToGraph</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(graph)}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  }</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">});</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(centrality);</span></span></code></pre>
<h4 id="how-it-computes-1">How it computes</h4>
<p>{/* lens:GPUNetworkCentrality */}</p>
<p>::lens{scenario=gpu-network.network-centrality rung=datum bind=page datum=3 caption="Inspect this pivot: it reaches 1,199 intersections, with closeness about 0.00615 per second."}</p>
<p>::lens{scenario=gpu-network.network-centrality rung=formula}</p>
<p>::lens{scenario=gpu-network.network-centrality rung=kernel}</p>
<p>::lens{scenario=gpu-network.network-centrality rung=graph}</p>
<h3 id="gpunetworkcentralityschedulestep"><code>GPUNetworkCentralityScheduleStep</code></h3>
<p>::lens{scenario=gpu-network.network-centrality-schedule-step rung=result caption="Compare eight sampled pivots in the first fixed-capacity schedule batch."}</p>
<p><code>planNetworkCentralitySchedule({pivotCount, laneCount})</code> lazily partitions pivot rows into fixed-lane
submissions. Compile one step, upload each batch into its pivot buffer, pad the final batch with
<code>0xffffffff</code>, update the one-word <code>reset</code> view, and submit the same graph on the same queue. Set reset
nonzero only on the first submission of a run. Node/edge betweenness and status accumulate on GPU;
per-pivot outputs describe the current batch and must be copied or consumed before the next submission.</p>
<p>For sampled estimates keep <code>betweennessScale = nodeCount / originalPivotCount</code> across all steps.
Each batch must still converge within <code>maximumIterations</code>. <code>createNetworkCentralityContinuation</code>
provides host-controlled continuation for longer paths. Do not interleave runs sharing output buffers;
inspect the final accumulated status before consuming the result.</p>
<p>Continuation accepts each planner descriptor directly: upload that step's pivots, then
<code>await runner.runBatch(step)</code>. The planner emits <code>reset: 1</code> for the first batch and <code>reset: 0</code> for
later batches. The runner also accepts booleans: <code>true</code>/<code>1</code> replaces totals and status (the default
when reset is omitted); <code>false</code>/<code>0</code> adds node/edge totals, takes the minimum <code>converged</code>, and retains
earlier failure and approximation flags. Its per-pivot outputs still describe only the current batch.</p>
<h4 id="how-it-computes-2">How it computes</h4>
<p>{/* lens:GPUNetworkCentralityScheduleStep */}</p>
<p>::lens{scenario=gpu-network.network-centrality-schedule-step rung=datum bind=page datum=3}</p>
<p>::lens{scenario=gpu-network.network-centrality-schedule-step rung=formula}</p>
<p>::lens{scenario=gpu-network.network-centrality-schedule-step rung=kernel}</p>
<p>::lens{scenario=gpu-network.network-centrality-schedule-step rung=graph}</p>
<h3 id="gpunetworkpathextraction-gpunetworkserviceareas-gpunetworkneighborhood-and-gpunetworkanalyticscolumns"><code>GPUNetworkPathExtraction</code>, <code>GPUNetworkServiceAreas</code>, <code>GPUNetworkNeighborhood</code>, and <code>GPUNetworkAnalyticsColumns</code></h3>
<p>Network analysis over the same directed CSR road network as <code>GPUNetworkReachability</code>.</p>
<ul>
<li>
<p><code>GPUNetworkPathExtraction</code> walks a reachability predecessor array from per-frame targets. It publishes ordered (source to target) node lists and resolved CSR edge lists as <code>GPUCompactOutput</code>s, together with per-target offsets, costs, and found flags. Each walk is bounded by a compile-time <code>maximumPathLength</code>, so bounded or cyclic walks terminate and set optional <code>output.status.candidateOverflow</code> on both node and edge ports. Final <code>overflow</code> reports only the respective output capacity. The CPU twin and operation forms use the same explicit budget and separate flags; the default budget is 1024. The per-target <code>lineOffsets</code>, <code>pathCosts</code> and <code>pathFound</code> are in <code>output</code>, and <code>targetCount</code> is a number or a one-row view. Pass the search's <code>output.status.converged</code> flag as <code>predecessorsComplete</code>: when it is 0 no target is reported found, because a chain may end at a node the truncated search never linked rather than at a source.</p>
<p>Routing readouts must check search convergence and <code>candidateOverflow</code> before treating an empty
path as unreachable. After a converged search, each target with <code>pathFound=1</code> keeps its certified
cost even if another target loses its walk. Candidate and final capacity flags describe the
batch, so unfound targets are qualified as “incomplete batch” rather than assigned an individual
failure. The explorer and showcase distinguish walk-budget loss (4096 nodes per explorer path,
2048 per showcase path) from final output capacity loss; geometry statistics require complete
output. The showcase clears its turn-comparison chart whenever either route lacks a usable edge
sequence. Only a completed search without batch loss reports “no route found” for <code>pathFound=0</code>.
The path-extraction lens retains the candidate flag and refuses captures whose walk budget was
exhausted.</p>
</li>
<li>
<p><code>GPUNetworkServiceAreas</code> assigns every node to its nearest per-frame facility. Ties go to the smallest facility row; optional <code>predecessors</code> follow only tight edges between nodes of the same assignment, so a path always ends at the facility that <code>assignments</code> names, also on cost ties. It also reports per-facility node counts and the total cost each facility serves, all in <code>output</code>. It composes <code>GPUNetworkReachability</code> with a frontier-based min-label pass over the tight shortest-path edges. <code>maximumLabelIterations</code> bounds the label rounds (default <code>min(maximumIterations, 32)</code>) and <code>localIterations</code> sets the hops chained per round in both phases (default 16), so large-diameter grids converge at the defaults.</p>
</li>
<li>
<p><code>GPUNetworkNeighborhood</code> publishes k-hop ego networks around per-frame <code>startNodes</code> (<code>startCount</code> a number or a view), with per-frame <code>k</code> up to a compile-time <code>maximumHops</code>. <code>output</code> holds hop distances, masks, and compact node and induced-edge IDs.</p>
</li>
<li>
<p><code>GPUNetworkAnalyticsColumns</code> publishes node-aligned <code>degree</code>, <code>inDegree</code> and weak <code>components</code> columns directly on the caller's <code>topology</code> (and optional <code>reverseTopology</code>) with analysis-owned kernels, so a transient CSR built in the same graph by <code>GPUCOOToCSR</code> works as well as imported buffers. A degree counts the live slots of a row (a slot whose neighbor is <code>>= nodeCount</code> is not an edge) and a self-loop slot once, the rule of <code>GPUGraphAlgorithmsDegree</code>; <code>GPUNetworkStatistics</code> counts an undirected self-loop twice. Each column can be normalized to <code>[0, 1]</code> with a GPU extent. PageRank, core number and label propagation are luma.gl <code>GPUGraphAlgorithms*</code> operations that accept only physical topology; run them from <code>@luma.gl/graph/gpu-graph-algorithms</code>.</p>
</li>
</ul>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUNetworkReachability</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({topology, startNodes: origin.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">importToGraph</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(graph),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  maximumIterations: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">48</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, output: {costs, predecessors}}));</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUNetworkPathExtraction</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({predecessors, costs,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  targets: destination.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">importToGraph</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(graph), output: route,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  edges: {topology, output: routeEdges}}));</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUNetworkServiceAreas</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({topology, facilities: stations.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">importToGraph</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(graph),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  costLimit: limit.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">importToGraph</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(graph), output: {assignments, facilityNodeCounts}}));</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUNetworkNeighborhood</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({topology, startNodes: picked.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">importToGraph</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(graph),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  hops: hops.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">importToGraph</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(graph), maximumHops: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">8</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, output: {hopDistances, nodes, edges}}));</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUNetworkAnalyticsColumns</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({topology,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  degree: {output: degree, normalized: degreeSize}, components: {output: componentLabels}}));</span></span></code></pre>
<h4 id="cpu-network-execution-budgets">CPU network execution budgets</h4>
<p><code>computeNetworkFlowOnCPU</code> and <code>CPUNetworkFlowOperation</code> resolve <code>solver</code> and costed <code>mode</code> before
selecting an explicit budget. Reference and costed successive-shortest-path use
<code>maximumAugmentations</code>; uncosted parallel uses <code>maximumRelabels</code>; costed primal-dual uses
<code>maximumCostPhases</code>. The latter two modes ignore <code>maximumAugmentations</code>, including zero.
<code>auto</code> selects the same route as the contributor (including topology, mode and size triggers).
After selection, reference mode refuses <code>maximumRelaxationRounds</code>, <code>maximumRelabels</code>, <code>mode</code>
and <code>maximumCostPhases</code> in both backends. In particular, <code>maximumRelabels</code> alone does not opt
<code>auto</code> into parallel mode; select <code>solver: 'parallel'</code> to supply that budget. Both operation
forms reject incompatible reference settings at construction, before lowering or execution.
CPU parallel phases serially exhaust shortest residual paths of equal cost and hop count;
GPU phases use parallel preflow discharges, so allocation and progress can differ. An explicit
<code>maximumRelaxationRounds</code> binds each synchronous CPU search, including its final unchanged sweep.
A budget exit retains feasible flow, reports the
remaining finite demand in <code>unserved</code>, clears summary <code>complete</code> and status <code>converged</code>, and
sets <code>iterationLimitReached</code>. A finite request already served or an exhausted residual network
is complete even at the budget boundary. Omitted CPU budgets run to completion.</p>
<h4 id="gpunetworkpathextraction"><code>GPUNetworkPathExtraction</code></h4>
<p>Convert reachability predecessors and target intersections into ordered start-to-target node paths with offsets, costs and found flags.</p>
<p>For CPU reachability, explicit <code>maximumIterations</code> and per-frame <code>activeIterations</code> bind the
search to at most <code>localIterations</code> hops per round (default 32); the active count is clamped to
the compiled maximum (default 64). Explicit <code>maximumTieIterations</code> similarly bounds zero-cost
predecessor tie levels. Budget exits retain current labels, clear <code>converged</code>, and set
<code>iterationLimitReached</code>; path extraction suppresses paths when that certificate is clear.
CPU frontier scheduling is synchronous and can advance further than GPU workgroups within a
round. With no explicit CPU budgets, searches run to completion.
Zero active rounds never certify a search, including an empty start set.</p>
<p><code>output.ids</code> contains caller node identities (<code>nodeIds[row]</code>, or the identity mapping when no
map is supplied); optional <code>output.nodeRows</code> separately contains raw node rows for positional
gathers. With edge extraction, <code>edges.output.ids</code> contains caller edge identities and optional
<code>edges.output.edgeRows</code> contains CSR edge slots. Each row output must match its identity output's
capacity and use a separate buffer. The CPU twin always returns <code>nodeRows</code> and, with edges,
<code>edges.edgeRows</code>; graph operations declare those ports when requested. Use rows to gather
positions or attributes, including when caller IDs happen to equal row indices.
Operation props use <code>NetworkPathExtractionNodeOperationOutput</code> for the node bag and
<code>NetworkPathExtractionEdgeOperationOutput</code> for the edge bag. Swapped row fields are rejected
by TypeScript and at runtime before lowering. Both operation forms and the contributor capture
their declared scalar fields, output/status maps and topology maps at construction; later caller
mutation cannot change validated graph declarations. Input view contents remain per-frame.</p>
<p>::lens{scenario=gpu-network.network-path-extraction rung=result caption="Follow the highlighted 25-intersection path from the start to the West Loop library target."}</p>
<h4 id="how-it-computes-3">How it computes</h4>
<p>{/* lens:GPUNetworkPathExtraction */}</p>
<p>::lens{scenario=gpu-network.network-path-extraction rung=datum bind=page datum=648 caption="Inspect the target: travel time is about 162.7 seconds, and its predecessor is intersection 660."}</p>
<p>::lens{scenario=gpu-network.network-path-extraction rung=kernel}</p>
<p>::lens{scenario=gpu-network.network-path-extraction rung=graph}</p>
<h4 id="gpunetworkserviceareas"><code>GPUNetworkServiceAreas</code></h4>
<p>Assign every road intersection to its nearest facility by travel time, with facility labels, minimum costs and optional service totals.</p>
<p>Optional one-row <code>uint32</code> ports <code>output.status.converged</code> and
<code>output.status.iterationLimitReached</code> cover all three phases: cost relaxation, facility-label
propagation and, when requested, predecessor tie levels. <code>iterationLimitReached</code> is exactly
<code>1 - converged</code> and can be requested alone. A budget exit retains the current costs, assignments
and predecessors. With a nonzero round budget, an empty active facility set converges. Both
ports reset on every encoding.
The graph operation exposes the same scalar ports. Its exact CPU twin runs every phase to
completion, so it returns <code>converged = 1</code> and <code>iterationLimitReached = 0</code> even when the GPU would
exhaust a configured phase budget.</p>
<p>::lens{scenario=gpu-network.network-service-areas rung=result caption="Compare 4 service areas; each colour marks the facility nearest by travel time."}</p>
<h4 id="how-it-computes-4">How it computes</h4>
<p>{/* lens:GPUNetworkServiceAreas */}</p>
<p>::lens{scenario=gpu-network.network-service-areas rung=datum bind=page datum=318 caption="Inspect this facility intersection: its assigned facility has travel cost 0 seconds."}</p>
<p>::lens{scenario=gpu-network.network-service-areas rung=formula}</p>
<p>::lens{scenario=gpu-network.network-service-areas rung=kernel}</p>
<p>::lens{scenario=gpu-network.network-service-areas rung=graph}</p>
<h4 id="gpunetworkneighborhood"><code>GPUNetworkNeighborhood</code></h4>
<p>Find intersections within a hop radius of road-network starts and return hop distances, membership masks and induced edges.</p>
<p>::lens{scenario=gpu-network.network-neighborhood rung=result caption="See the 49 intersections within 6 hops of the start, shaded by hop count."}</p>
<h4 id="how-it-computes-5">How it computes</h4>
<p>{/* lens:GPUNetworkNeighborhood */}</p>
<p>::lens{scenario=gpu-network.network-neighborhood rung=datum bind=page datum=1088 caption="Inspect the start intersection: its hop distance is 0."}</p>
<p>::lens{scenario=gpu-network.network-neighborhood rung=formula}</p>
<p>::lens{scenario=gpu-network.network-neighborhood rung=kernel}</p>
<p>::lens{scenario=gpu-network.network-neighborhood rung=graph}</p>
<h4 id="gpunetworkanalyticscolumns"><code>GPUNetworkAnalyticsColumns</code></h4>
<p>Compute node-aligned degree and min-max normalized columns from road connectivity.</p>
<p>::lens{scenario=gpu-network.network-analytics-columns rung=result caption="Find high-degree junctions; the highlighted intersection has 6 outgoing roads."}</p>
<h4 id="how-it-computes-6">How it computes</h4>
<p>{/* lens:GPUNetworkAnalyticsColumns */}</p>
<p>::lens{scenario=gpu-network.network-analytics-columns rung=datum bind=page datum=1088 caption="Inspect this intersection: degree 2 becomes a normalized score of about 0.2."}</p>
<p>::lens{scenario=gpu-network.network-analytics-columns rung=formula}</p>
<p>::lens{scenario=gpu-network.network-analytics-columns rung=kernel}</p>
<p>::lens{scenario=gpu-network.network-analytics-columns rung=graph}</p>
<h3 id="gpunetworkstatistics"><code>GPUNetworkStatistics</code></h3>
<p>Summarize road connectivity into component counts, degree histograms and maxima in one GPU-written table of words.</p>
<p>::lens{scenario=gpu-network.network-statistics rung=result caption="Read the GPU summary: 1,199 intersections, 2,810 directed edges, and one connected component."}</p>
<p>Statistics for a graph panel over the same CSR as <code>GPUNetworkAnalyticsColumns</code>, reduced on the GPU
into one caller-owned <code>uint32</code> summary. It reports live node, edge and slot counts, weak component
count and largest component size, isolated nodes, maximum degrees, out-, in- and total-degree
histograms (<code>'linear'</code> or <code>'log2'</code> bins), and the modularity of a caller-supplied community
labeling. Optional <code>nodeMask</code> and <code>edgeMask</code> restrict every statistic to live rows. Components run
<code>GPUGraphAlgorithmsConnectedComponents</code> on a masked copy of the adjacency, so masking a bridge node splits
its component. Modularity is accumulated with exact integer atomics and a fixed-order f32 finish,
so it is deterministic. On undirected graphs a self-loop adds 2 to the degree, the histograms, the
maxima and the modularity volumes, as in networkx and <code>GPUGraphAlgorithmsModularity</code>; slot and edge counts count
the stored slot once. (The <code>degree</code> column of <code>GPUNetworkAnalyticsColumns</code> counts a self-loop slot once,
as <code>GPUGraphAlgorithmsDegree</code> does, so the two differ on nodes with self-loops.) The resolution
and linear bin width are per-frame parameters (<code>getGPUNetworkStatisticsParameterValues</code>). Read the
summary back and decode it with <code>decodeGPUNetworkStatistics</code>. Scratch (the masked adjacency, labels
and convergence flag) is graph transients; the contributor owns no buffers. Components run at most
<code>maximumComponentIterations</code> rounds.
The slot loops are load-balanced, so a hub row spreads over many workgroups; <code>topology.offsets</code> must be non-decreasing.</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> summary</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8"> graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">importGPUData</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">&#x3C;</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'uint32'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">>(</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'stats'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, statsData);</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUNetworkStatistics</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({topology, nodeMask, communities,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  parameters: statsParameters.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">importToGraph</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(graph), degreeBinning: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'log2'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, output: summary}));</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// after the application submits the encoding</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> bytes</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> await</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8"> statsBuffer.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">readAsync</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(); </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// Uint8Array: view its bytes as uint32 words</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> stats</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> decodeGPUNetworkStatistics</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">  new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> Uint32Array</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(bytes.buffer, bytes.byteOffset, bytes.byteLength </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">/</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> 4</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">);</span></span></code></pre>
<h4 id="how-it-computes-7">How it computes</h4>
<p>{/* lens:GPUNetworkStatistics */}</p>
<p>::lens{scenario=gpu-network.network-statistics rung=datum bind=page datum=2 caption="Inspect the out-degree histogram: 527 intersections have degree 2."}</p>
<p>::lens{scenario=gpu-network.network-statistics rung=formula}</p>
<p>::lens{scenario=gpu-network.network-statistics rung=kernel}</p>
<p>::lens{scenario=gpu-network.network-statistics rung=graph}</p>
<p>{/* parameters: GPUNetworkStatistics generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h4 id="gpunetworkstatistics-parameters"><code>GPUNetworkStatistics</code> parameters</h4>
<ul>
<li>Packer: <code>getGPUNetworkStatisticsParameterValues(settings?)</code>, writes <code>GPU_NETWORK_STATISTICS_PARAMETER_LENGTH</code> (2) <code>uint32</code> values.</li>
</ul>
<p>Settings of <code>getGPUNetworkStatisticsParameterValues</code>:</p>
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
<td>Modularity resolution <code>gamma</code>. Defaults to 1.</td>
</tr>
<tr>
<td><code>degreeBinWidth</code> (optional)</td>
<td><code>number</code></td>
<td>Linear degree bin width, floored and clamped to at least 1. Defaults to 1.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h3 id="gpunetworksubgraphfilter"><code>GPUNetworkSubgraphFilter</code></h3>
<p>Keep road edges and intersections that pass range predicates, returning consistent node and edge masks with live counts.</p>
<p>::lens{scenario=gpu-network.network-subgraph-filter rung=result caption="Compare 908 retained directed road edges with the faint full graph; 805 intersections remain."}</p>
<p>Turns attribute and time predicates on node and edge rows into a consistent induced subgraph of a forward CSR network.</p>
<ul>
<li>Node <code>v</code> is live when its caller <code>nodeMask</code> is nonzero and every enabled node range accepts its f32 column value. Ranges are half-open (<code>min &#x3C;= value &#x3C; max</code>); NaN never passes.</li>
<li>Slot <code>u -> v</code> is live when its caller mask is nonzero, every enabled edge range and time gate accepts it, and both endpoints are live. The f32 <code>edgeTimes</code> window and the exact Int64 <code>edgeTimeWords</code> window are closed (<code>start &#x3C;= t &#x3C;= end</code>).</li>
<li>Undirected graphs list each edge in both slots. By default an edge is live only if both slots pass, so <code>edgeMask</code> is always symmetric; set <code>pairUndirectedSlots: false</code> to skip the O(degree) pairing when columns are already symmetric.</li>
<li><code>dropIsolated</code> also kills nodes with no live incident slot.</li>
<li>Ranges live in one f32 <code>parameters</code> view (<code>getGPUNetworkSubgraphFilterParameterValues</code>) and the time-word window in <code>uint32Parameters</code> (<code>getGPUTimeWindowFilterWordParameterValues</code>); rewriting them never recompiles.</li>
<li>Outputs: <code>output.nodeMask</code> and <code>output.edgeMask</code> (1 live, 0 dead; feed <code>GPUNetworkStatistics</code> and <code>filterMask</code>), <code>counts</code> (decode with <code>decodeGPUNetworkSubgraphFilterCounts</code>), compact <code>liveNodes</code> and <code>liveEdgeSlots</code> (stable ascending, overflow-reporting), and an <code>inducedTopology</code> (<code>offsets</code>, <code>neighbors</code> over the original node ids, optional <code>sourceSlots</code>, <code>status.overflow</code>) built with a prefix sum over live degree. Rows past the live totals (<code>edgeMask</code> slots past <code>offsets[nodeCount]</code>, induced CSR rows past the live slot count) are written every frame: 0 and <code>0xffffffff</code>.</li>
<li>Results are exact: integer arithmetic and comparisons only.</li>
</ul>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> filter</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUNetworkSubgraphFilter</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  topology,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  nodeColumns: [population], edgeColumns: [speed], parameters,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  dropIsolated: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">true</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  output: {nodeMask, edgeMask, counts}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">});</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(filter);</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">parameterBuffer.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">write</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">  getGPUNetworkSubgraphFilterParameterValues</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(layout, {nodeRanges: [[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">1000</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">Infinity</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">]], edgeRanges: [[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">30</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">90</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">]]})</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">);</span></span></code></pre>
<h4 id="how-it-computes-8">How it computes</h4>
<p>{/* lens:GPUNetworkSubgraphFilter */}</p>
<p>::lens{scenario=gpu-network.network-subgraph-filter rung=datum bind=page datum=100 caption="Inspect the rejected edge: its travel time is about 12.4 seconds."}</p>
<p>::lens{scenario=gpu-network.network-subgraph-filter rung=formula}</p>
<p>::lens{scenario=gpu-network.network-subgraph-filter rung=kernel}</p>
<p>::lens{scenario=gpu-network.network-subgraph-filter rung=graph}</p>
<p>{/* parameters: GPUNetworkSubgraphFilter generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h4 id="gpunetworksubgraphfilter-parameters"><code>GPUNetworkSubgraphFilter</code> parameters</h4>
<ul>
<li>Packer: <code>getGPUNetworkSubgraphFilterParameterValues(layout, state?)</code>, writes <code>getGPUNetworkSubgraphFilterParameterLength(...)</code> <code>float32</code> values.
<ul>
<li><code>state</code> (<code>GPUNetworkSubgraphFilterState</code>)</li>
</ul>
</li>
</ul>
<p>Settings of <code>getGPUNetworkSubgraphFilterParameterValues</code>:</p>
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
<td><code>nodeColumnCount</code></td>
<td><code>number</code></td>
<td>Number of node columns.</td>
</tr>
<tr>
<td><code>edgeColumnCount</code></td>
<td><code>number</code></td>
<td>Number of edge columns.</td>
</tr>
<tr>
<td><code>hasEdgeTimes</code></td>
<td><code>boolean</code></td>
<td>Whether an f32 edge time column exists.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h3 id="gpunetworkcoarsening"><code>GPUNetworkCoarsening</code></h3>
<p>Aggregate road intersections by zone label into supernode counts, centroids and sorted weighted superedges.</p>
<p><code>output.edges.status.overflow</code> reports final superedge capacity loss. Optional
<code>output.edges.status.candidateOverflow</code> reports excluded labels and edges; <code>requiredCount</code> counts
retained superedges and is a lower bound on the full requirement when labels were excluded.
The coarsening lens retains both overflow flags and refuses capture when either is nonzero;
nonempty retained groups alone cannot certify that all live input labels were included.</p>
<p>::lens{scenario=gpu-network.network-coarsening rung=result caption="Compare 4 zones joined by 9 superedges; line colour counts directed road edges between zones."}</p>
<p>Maximum: with <code>output.edgeWeights</code>, <code>output.edges.ids</code> has at most <code>getGPUNetworkCoarseningMaximumWeightedEdgeCapacity(device)</code> (16777216 on a default device) rows, because the 64-bit weight accumulators take 8 bytes per superedge.</p>
<p>Summarizes a network <code>topology</code> (weights and <code>nodePositions</code> optional) by a node group label (a community, component or spatial cell) for compound nodes when zoomed out. Every result is in <code>output</code>. Supernode outputs: <code>groupNodeCount</code> plus optional <code>groupIntraEdgeCount</code>, <code>groupIntraWeight</code>, <code>groupCentroid</code> (<code>float32x2</code>), <code>groupBounds</code> (<code>float32x4</code>) and <code>groupValueSum</code>, each with exactly <code>groupCapacity</code> rows. Superedges are a bounded <code>edges</code> output (<code>ids</code> is the source group) with <code>edgeTargetGroups</code>, <code>edgeCounts</code> and optional <code>edgeWeights</code>, sorted by <code>(source, target)</code>; undirected superedges have <code>source &#x3C; target</code>. An optional 8-word <code>summary</code> reports live and overflowed nodes, counted/dropped/intra/inter edges, group count and the unclamped superedge count.</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> coarsening</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUNetworkCoarsening</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  topology: {</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">topology, nodePositions}, labels, groupCapacity: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">1024</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  output: {</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    groupNodeCount, groupCentroid, groupBounds,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    edges: {ids: sourceGroups, status: {count, overflow, candidateOverflow}}, edgeTargetGroups, edgeCounts, edgeWeights</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  }</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">});</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(coarsening);</span></span></code></pre>
<p>Labels are dense group IDs below the compile-time <code>groupCapacity</code>. Its exported addressing ceiling is
<code>GPU_NETWORK_COARSENING_MAXIMUM_GROUP_CAPACITY = floor(0xffffffff / 10)</code> (429,496,729 groups),
because the per-group accumulator uses ten uint32 words. Each per-group output and scratch binding
must also fit the device storage-binding and buffer limits, which usually impose a lower capacity.
The existing large-group headless regression uses 131,072 groups with separate source/target keys;
the addressing ceiling alone is not an allocation guarantee.</p>
<p>A live node with <code>label >= groupCapacity</code> is excluded and sets <code>edges.status.candidateOverflow</code>; final <code>edges.status.overflow</code> remains zero when the retained superedges fit. Undirected CSRs list each edge in both directions and each edge counts once. Counts are integer exact. Weight, position and value sums use 64-bit fixed-point atomics, so they are bitwise reproducible. By default the unit of each sum follows the data (a power of two derived on the GPU from the largest term that is summed, the rule of the shared fixed-point accumulator), so 1,000 weights of 1e-4 sum to 0.1; pass <code>fixedPointScale</code> (a power of two, 65536 was the former default) only to pin a fixed rounding step of <code>1 / fixedPointScale</code> per addend; an addend larger than about <code>4e18</code> steps (6.1e13 at 65536) then saturates at that value. Superedges use a stable sort and segment, so order and the kept prefix on overflow are deterministic. CSR slots past <code>offsets[nodeCount]</code> take the sentinel key and never become superedges. Inputs must be finite.</p>
<h4 id="how-it-computes-9">How it computes</h4>
<p>{/* lens:GPUNetworkCoarsening */}</p>
<p>::lens{scenario=gpu-network.network-coarsening rung=datum bind=page datum=1 caption="Inspect this zone: it contains 216 intersections, summarized by its centroid."}</p>
<p>::lens{scenario=gpu-network.network-coarsening rung=kernel}</p>
<p>::lens{scenario=gpu-network.network-coarsening rung=graph}</p>
<h3 id="gpuadjacencymatrix-and-gpuadjacencymatrixorder"><code>GPUAdjacencyMatrix</code> and <code>GPUAdjacencyMatrixOrder</code></h3>
<p>Bins a forward <code>topology</code> into an <code>R x R</code> matrix image for a matrix view linked to a node-link view. Row-major <code>uint32</code> counts (<code>row * R + column</code>), optional fixed-point weight sums, one-row maxima for color normalization, and an optional <code>r32float</code> storage texture. The optional <code>order</code> view maps node to matrix position (a permutation computed by the caller, for example with <code>GPUAdjacencyMatrixOrder</code> or <code>computeAdjacencyMatrixOrder</code>; identity by default). Per-frame zoom is a four-word window <code>[rowStart, rowEnd, colStart, colEnd)</code> in matrix positions, the <code>parameters</code> view (<code>getGPUAdjacencyMatrixParameterValues</code>); position <code>p</code> lands in bin <code>floor((p - start) * R / (end - start))</code> and positions outside the window are dropped, so panning and zooming never recompile.</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> matrix</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUAdjacencyMatrix</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  topology, order, parameters: windowParameters.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">importToGraph</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(graph),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  resolution: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">64</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  output: {counts, weightSums, maximumCount, maximumWeightSum}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">});</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(matrix);</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">windowParameters.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">write</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">  getGPUAdjacencyMatrixParameterValues</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({rowStart: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, rowEnd: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">500</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, colStart: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, colEnd: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">500</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, resolution: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">64</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">})</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">);</span></span></code></pre>
<p>Undirected graphs follow the convention that the CSR lists both directions, so each slot is written once and the matrix is symmetric; pass <code>mirrorSlots: true</code> for an edge-list CSR that lists each edge once and each slot is also written transposed (self-loops once). Directed: row = source. Masks follow the <code>GPUNetworkStatistics</code> liveness rule. Counts are exact <code>u32</code> atomics. Weight sums are fixed-point <code>u32</code> atomics of <code>round(weight * weightScale)</code> (default scale 1024), so results are bit-identical regardless of thread order; the cost is half a step of quantization per edge and wraparound at 2^32. <code>GPUAdjacencyMatrixOrder</code> builds <code>order</code> from a <code>uint32</code> group label and an optional tie key with two stable <code>GPUSort</code> passes and matches <code>computeAdjacencyMatrixOrder</code> exactly.</p>
<h4 id="gpuadjacencymatrix"><code>GPUAdjacencyMatrix</code></h4>
<p>Bin directed road connections into a small adjacency matrix of edge counts, in a chosen node order.</p>
<p>::lens{scenario=gpu-network.adjacency-matrix rung=result caption="Look along the diagonal of the 8-by-8 matrix; the busiest cell contains 306 directed edges."}</p>
<p>{/* parameters: GPUAdjacencyMatrix generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h5 id="gpuadjacencymatrix-parameters"><code>GPUAdjacencyMatrix</code> parameters</h5>
<ul>
<li>Packer: <code>getGPUAdjacencyMatrixParameterValues(settings)</code>, writes <code>GPU_ADJACENCY_MATRIX_PARAMETER_LENGTH</code> (4) <code>uint32</code> values.</li>
</ul>
<p>Settings of <code>getGPUAdjacencyMatrixParameterValues</code>:</p>
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
<td><code>rowStart</code></td>
<td><code>number</code></td>
<td>First row position.</td>
</tr>
<tr>
<td><code>rowEnd</code></td>
<td><code>number</code></td>
<td>One past the last row position.</td>
</tr>
<tr>
<td><code>colStart</code></td>
<td><code>number</code></td>
<td>First column position.</td>
</tr>
<tr>
<td><code>colEnd</code></td>
<td><code>number</code></td>
<td>One past the last column position.</td>
</tr>
<tr>
<td><code>resolution</code></td>
<td><code>number</code></td>
<td>Matrix resolution <code>R</code> of the <code>GPUAdjacencyMatrix</code>, used to bound the window extents.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h4 id="how-it-computes-10">How it computes</h4>
<p>{/* lens:GPUAdjacencyMatrix */}</p>
<p>::lens{scenario=gpu-network.adjacency-matrix rung=datum bind=page datum=9 caption="Inspect this diagonal matrix cell: it contains 254 directed edges."}</p>
<p>::lens{scenario=gpu-network.adjacency-matrix rung=formula}</p>
<p>::lens{scenario=gpu-network.adjacency-matrix rung=kernel}</p>
<p>::lens{scenario=gpu-network.adjacency-matrix rung=graph}</p>
<h4 id="gpuadjacencymatrixorder"><code>GPUAdjacencyMatrixOrder</code></h4>
<p>Sort intersections by group label into a stable node-to-matrix-position permutation.</p>
<p>::lens{scenario=gpu-network.adjacency-matrix-order rung=result caption="Compare 12 shown intersections with their matrix positions, sorted by group then node id."}</p>
<h4 id="how-it-computes-11">How it computes</h4>
<p>{/* lens:GPUAdjacencyMatrixOrder */}</p>
<p>::lens{scenario=gpu-network.adjacency-matrix-order rung=datum bind=page datum=648 caption="Inspect this intersection: group 4 places it at matrix position 590."}</p>
<p>::lens{scenario=gpu-network.adjacency-matrix-order rung=kernel}</p>
<p>::lens{scenario=gpu-network.adjacency-matrix-order rung=graph}</p>
<h3 id="gpunetworksnapping-gpunetworkcostmatrix-and-gpunetworkaccessibility"><code>GPUNetworkSnapping</code>, <code>GPUNetworkCostMatrix</code>, and <code>GPUNetworkAccessibility</code></h3>
<p>For <code>GPUNetworkSnapping.maximumSnapDistance</code>, no warning is possible because the distance is a GPU view; see <a href="/luma-contributors/docs/api/gpu-spatial-analysis/concepts#precision-of-longitudelatitude-positions">Precision of longitude/latitude positions</a>.</p>
<p>Accessibility over the same directed CSR road network as <code>GPUNetworkReachability</code>, in three steps.</p>
<ul>
<li><code>GPUNetworkSnapping</code> snaps planar points (origins, facilities, opportunities) to the nearest
network edge of a <code>topology</code> with <code>nodePositions</code> (edge costs are <code>topology.weights</code>, or the
edge length when the topology has no weights). For each point it
outputs the edge row (ties go to the smallest row, so a point on a node takes the smallest
incident edge), the fraction along the edge, the snap distance, the snapped position, and the
two start costs: <code>fraction * edgeCost</code> to the edge source and <code>(1 - fraction) * edgeCost</code> to the
target, all in <code>output</code>. <code>output.startNodes</code> / <code>output.startCosts</code> (two rows per point) feed a
search directly. <code>startDirection</code> picks which endpoints count: <code>'both'</code> for undirected
networks, <code>'forward'</code> for a point that leaves along a one-way edge, <code>'reverse'</code> for a point that
a one-way edge arrives at. Without <code>candidateCapacity</code>, every point scans every edge, which is
exact. With it, candidates come from <code>GPUNearestFeatureJoin</code> over the edge segments within a
per-frame <code>maximumSnapDistance</code>, and the required <code>output.status.overflow</code> reports when capacity runs out. With <code>candidateCapacity</code> the network has at most <code>getGPUNetworkSnappingMaximumCandidateEdgeCount(device)</code> (8388608 on a default device) edges. With
<code>spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'great-circle'}</code> points and nodes
are degrees: the closest point of an edge is found in the point's local equirectangular frame
(longitudes wrapped at +-180), and snap distances, <code>maximumSnapDistance</code> and default edge costs are
in the context's length unit (meters or kilometers). Edges should be road-length. With <code>candidateCapacity</code> the context is passed to
<code>GPUNearestFeatureJoin</code>, whose great-circle search ignores the capacity. Edge rows past
<code>offsets[nodeCount]</code> (a CSR shorter than <code>neighbors</code>) are not part of the network:
every edge-source row is cleared before the expansion, so no stale row becomes a phantom edge.</li>
<li><code>GPUNetworkCostMatrix</code> computes a bounded many-to-all cost matrix: one shortest-path search per
row, from that row's starts (<code>startRows</code>, or <code>startsPerRow: 2</code> for snapped points) to every node.
It batches <code>laneCount</code> rows (default <code>recommendLaneCount()</code>: about 1M expanded nodes per batch,
at least 32 lanes, within 128 MB of scratch, at most <code>rowCount</code>, rounded down to make
<code>laneCount * nodeCount</code> a multiple of 64 when there are several batches) into one <code>GPUNetworkReachability</code> over a lane-expanded
copy of the CSR. Lane <code>l</code>, node <code>u</code> becomes node <code>l * nodeCount + u</code>, so the frontier rounds,
hop chaining, per-frame <code>costLimit</code>, and convergence flag serve all lanes at once. The matrix is
bit-identical for every <code>laneCount</code>. Every batch runs at the full lane count (a short last batch
masks its lanes) on 64-element-aligned sub-views, so all batches share one set of pipelines. A
batch whose block of <code>costs</code> is not 256-byte aligned searches into a scratch block and is copied.
<code>targetNodes</code> (a compile-time length <code>targetCount</code>, per-frame node IDs) makes <code>output.costs</code> the
compact <code>[rowCount x targetCount]</code> matrix: column <code>t</code> is the cost to <code>targetNodes[t]</code>, <code>+Infinity</code>
when unreached or out of range, bit-identical to that column of the dense matrix. Each batch
searches into one <code>laneCount * nodeCount</code> scratch block and a gather kernel writes its rows, so
memory is that block plus <code>rowCount * targetCount</code> instead of <code>rowCount * nodeCount</code>; lower
<code>laneCount</code> to shrink the scratch block. <code>columnCount</code> reports the column count.
Compile cost: every round of every batch is an unrolled command node. The exact count is the
read-only property <code>estimatedCommandNodeCount</code>, and the constructor throws above 32,768 (for
example 600 one-lane batches at the default 64 rounds); raise <code>laneCount</code> or lower <code>maximumIterations</code>.
Rows set the number of searches, so put the smaller side on
rows. To score every node, put opportunities or facilities on rows and search the reverse CSR
(an undirected network can reuse its CSR).</li>
<li><code>GPUNetworkAccessibility</code> scores a retained matrix with per-frame parameters
(<code>getGPUNetworkAccessibilityParameterValues</code>: threshold, decay <code>'none' | 'exponential' | 'power'</code>,
beta, minimum cost for power decay). It outputs cumulative opportunities within the threshold,
gravity accessibility, and the two- or three-step floating catchment area (<code>catchment</code>: <code>method</code>
<code>'2sfca'</code> (default) or <code>'3sfca'</code>, supply on rows, demand per node, per-facility ratios, and
per-node accessibility, with the decay applied inside each catchment). The catchment composes
<code>GPUNetworkCostWeights</code> (the matrix as facility weights and their transpose, <code>rowCount x nodeCount</code>
transient slots each) with <code>GPUCatchmentAccessibility</code>, the one floating catchment kernel; for a
large matrix build sparser weights and run the two directly. <code>orientation: 'origin-rows'</code> scores a
forward matrix whose rows are origins.
Per-node sums gather rows in ascending order. Per-row sums use one workgroup and a fixed tree.
Neither uses float atomics, so results are bitwise reproducible. Non-finite opportunity, supply or
demand weights are missing values and are skipped; the encoder rejects a negative or non-finite
<code>beta</code> and a <code>minimumCost</code> that is not positive and finite.</li>
</ul>
<p>Encode the matrix graph when the network or the opportunity set changes. Encode the scoring graph
every frame: threshold, decay, and beta changes never re-run a search. The threshold must not
exceed the <code>costLimit</code> the matrix was built with.</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">matrixGraph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUNetworkSnapping</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({positions: jobs, topology: {</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">topology, nodePositions},</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  output: {snappedEdges, startNodes, startCosts}}));</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">matrixGraph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUNetworkCostMatrix</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({topology: reverseTopology, startNodes, startCosts,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  startsPerRow: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, costLimit: limit.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">importToGraph</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(matrixGraph), maximumIterations: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">48</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  output: {costs: matrix}}));</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">scoreGraph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUNetworkAccessibility</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({costs: matrixView, opportunityWeights: jobCounts,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  parameters: scoring.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">importToGraph</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(scoreGraph), cumulative, gravity}));</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">scoring.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">write</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">getGPUNetworkAccessibilityParameterValues</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({threshold: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">1800</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, decay: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'exponential'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, beta: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">0.002</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span></code></pre>
<h4 id="gpunetworksnapping"><code>GPUNetworkSnapping</code></h4>
<p>Project point locations onto the nearest road edges and return edge rows, fractions, distances and search start costs.</p>
<p>Without <code>candidateCapacity</code> every point scans every edge, which is exact and needs no radius, so the contributor refuses <code>positions</code> times edges above <code>GPU_NETWORK_SNAPPING_MAXIMUM_UNINDEXED_WORK</code> (2^27). Above it, pass <code>candidateCapacity</code>, <code>maximumSnapDistance</code> and <code>output.status.overflow</code> to find candidate edges through a BVH; both paths return the same edges, with ties going to the smallest edge row. <code>maximumUnindexedWork</code> (default the same constant) raises the limit for a contributor that embeds the snapping and must keep accepting configurations it accepted before the limit existed; the network K function does, so its events times edges is not capped by this limit. The drive-time catchment recipe forwards its own <code>candidateCapacity</code> and so meets the limit when it is not given.</p>
<p>::lens{scenario=gpu-network.network-snapping rung=result caption="Compare 19 facility snaps; the longest moves about 95.8 metres onto its nearest road."}</p>
<h4 id="how-it-computes-12">How it computes</h4>
<p>{/* lens:GPUNetworkSnapping */}</p>
<p>::lens{scenario=gpu-network.network-snapping rung=datum bind=page datum=3 caption="Inspect this facility: it snaps about 8.47 metres to edge 49."}</p>
<p>::lens{scenario=gpu-network.network-snapping rung=formula}</p>
<p>::lens{scenario=gpu-network.network-snapping rung=kernel}</p>
<p>::lens{scenario=gpu-network.network-snapping rung=graph}</p>
<h4 id="gpunetworkcostmatrix"><code>GPUNetworkCostMatrix</code></h4>
<p>Compute row-major travel costs from snapped facility starts to every road intersection or to selected target nodes.</p>
<p>::lens{scenario=gpu-network.network-cost-matrix rung=result caption="Read the first facility row: travel times to 1,199 intersections, reaching about 381.3 seconds."}</p>
<h4 id="how-it-computes-13">How it computes</h4>
<p>{/* lens:GPUNetworkCostMatrix */}</p>
<p>::lens{scenario=gpu-network.network-cost-matrix rung=datum bind=page datum=648 caption="Inspect this intersection: travel time from the first facility is about 277.8 seconds."}</p>
<p>::lens{scenario=gpu-network.network-cost-matrix rung=kernel}</p>
<p>::lens{scenario=gpu-network.network-cost-matrix rung=graph}</p>
<h4 id="gpunetworkaccessibility"><code>GPUNetworkAccessibility</code></h4>
<p>Reduce a facility-by-node cost matrix and opportunity weights into cumulative, gravity or floating-catchment accessibility scores.</p>
<p>::lens{scenario=gpu-network.network-accessibility rung=result caption="Find the highest gravity access score, about 2.373, among intersections near the facilities."}</p>
<p>{/* parameters: GPUNetworkAccessibility generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h5 id="gpunetworkaccessibility-parameters"><code>GPUNetworkAccessibility</code> parameters</h5>
<ul>
<li>Packer: <code>getGPUNetworkAccessibilityParameterValues(settings)</code>, writes <code>GPU_NETWORK_ACCESSIBILITY_PARAMETER_LENGTH</code> (4) <code>float32</code> values.</li>
</ul>
<p>Settings of <code>getGPUNetworkAccessibilityParameterValues</code>:</p>
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
<td><code>threshold</code></td>
<td><code>number</code></td>
<td>Cost cutoff: only matrix entries with a finite cost <code>&#x3C;= threshold</code> contribute. <code>Infinity</code> keeps every reached entry. Must not exceed the <code>costLimit</code> the matrix was built with.</td>
</tr>
<tr>
<td><code>decay</code> (optional)</td>
<td><code>GPUNetworkAccessibilityDecay</code></td>
<td>Decay applied to the <code>gravity</code> and two-step floating catchment outputs. Defaults to <code>'none'</code>.</td>
</tr>
<tr>
<td><code>beta</code> (optional)</td>
<td><code>number</code></td>
<td>Decay rate, finite and non-negative. Defaults to 0.</td>
</tr>
<tr>
<td><code>minimumCost</code> (optional)</td>
<td><code>number</code></td>
<td>Power-decay floor so zero costs stay finite. Must be positive and finite. Defaults to 1.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h4 id="how-it-computes-14">How it computes</h4>
<p>{/* lens:GPUNetworkAccessibility */}</p>
<p>::lens{scenario=gpu-network.network-accessibility rung=datum bind=page datum=318 caption="Inspect this intersection: its gravity score is about 2.110."}</p>
<p>::lens{scenario=gpu-network.network-accessibility rung=formula}</p>
<p>::lens{scenario=gpu-network.network-accessibility rung=kernel}</p>
<p>::lens{scenario=gpu-network.network-accessibility rung=graph}</p>
<h3 id="gpunetworkcostweights"><code>GPUNetworkCostWeights</code></h3>
<p>Convert a facility-by-node travel-cost matrix into sparse catchment weights with a cost cutoff and distance decay.</p>
<p>::lens{scenario=gpu-network.network-cost-weights rung=result caption="See the 823 intersections within the first facility's cost limit; shade shows the decay weight."}</p>
<p>Turns a network cost matrix (the <code>output.costs</code> of <code>GPUNetworkCostMatrix</code>, one row per start row)
or a reachability result (one row per lane) into <code>GPUSpatialWeights</code> with a distance decay, so every
weights consumer runs on network costs. The main use is <code>GPUCatchmentAccessibility</code> (2SFCA and 3SFCA)
on travel times: with facilities on the matrix rows, <code>orientation: 'rows'</code> gives the facility weights
and <code>orientation: 'columns'</code> their exact transpose, the demand weights.</p>
<ul>
<li>An entry becomes a slot when its cost is finite, non-negative and <code>&#x3C;= costLimit</code>. Its weight is the
decay <code>f(c)</code> and its distance is the cost. Slots are laid out row by row with ascending neighbor
IDs, so the weights are deterministic; a zero cost (a facility at its own demand location) is kept.</li>
<li>Decays (per frame, <code>getGPUNetworkCostWeightsParameterValues</code>): <code>'none'</code> (1, the binary catchment),
<code>'exponential'</code> <code>exp(-beta c)</code>, <code>'power'</code> <code>max(c, minimumCost)^-beta</code>, and <code>'gaussian'</code>, the Dai
(2010) kernel <code>(exp(-c^2 / 2L^2) - exp(-1/2)) / (1 - exp(-1/2))</code> with <code>L = costLimit</code>, which is 1 at
0 and 0 at the limit (weights consumers skip zero-weight slots).</li>
<li><code>columns</code> maps locations to matrix columns, for example the node each demand location is snapped
to, so neighbor IDs index locations rather than nodes. Out-of-range columns are unreached.</li>
<li>Capacity is <code>output.neighbors.length</code>; <code>rowCount x locations</code> always suffices. Slots past it are
dropped, offsets are clamped to stay a consistent CSR, and <code>output.status.overflow</code> / <code>output.status.requiredCount</code>
feed the <code>getGPUSpatialWeightsCapacityPlan</code> grow loop.</li>
<li>Passes: a per-row count (one workgroup per row), a <code>GPUScan</code> into <code>offsets</code>, an ordered per-row fill
and a finalize pass. No atomics.</li>
</ul>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// \`createWeightsViews(rows, capacity)\` returns fresh \`offsets\`, \`neighbors\` and \`weights\` views.</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> decayView</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8"> decay.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">importToGraph</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(graph);</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">const</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> weights</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8"> (</span><span style="--shiki-light:#E36209;--shiki-dark:#FFAB70">orientation</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">:</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF"> 'rows'</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> |</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF"> 'columns'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#E36209;--shiki-dark:#FFAB70">rows</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> number</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">=></span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8"> {</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">  const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> contributor</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUNetworkCostWeights</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    id: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">\`\${</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">orientation</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">}-weights\`</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, costs: matrix, rowCount: facilityCount, orientation,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    parameters: decayView, output: </span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">createWeightsViews</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(rows, facilityCount </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">*</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8"> nodeCount)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  });</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(contributor);</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">  return</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8"> contributor.outputWeights;</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">};</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUCatchmentAccessibility</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  method: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'2sfca'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, supply, demand,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  facilityWeights: </span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">weights</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'rows'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, facilityCount), demandWeights: </span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">weights</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'columns'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, nodeCount),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  output: {accessibility}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">decay.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">write</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">getGPUNetworkCostWeightsParameterValues</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({costLimit: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">1800</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, decay: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'gaussian'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span></code></pre>
<h4 id="how-it-computes-15">How it computes</h4>
<p>{/* lens:GPUNetworkCostWeights */}</p>
<p>::lens{scenario=gpu-network.network-cost-weights rung=datum bind=page datum=0 caption="Inspect the first facility row: it retains 823 entries beginning at offset 0."}</p>
<p>::lens{scenario=gpu-network.network-cost-weights rung=formula}</p>
<p>::lens{scenario=gpu-network.network-cost-weights rung=kernel}</p>
<p>::lens{scenario=gpu-network.network-cost-weights rung=graph}</p>
<p>{/* parameters: GPUNetworkCostWeights generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h4 id="gpunetworkcostweights-parameters"><code>GPUNetworkCostWeights</code> parameters</h4>
<ul>
<li>Packer: <code>getGPUNetworkCostWeightsParameterValues(settings)</code>, writes <code>GPU_NETWORK_COST_WEIGHTS_PARAMETER_LENGTH</code> (4) <code>float32</code> values.</li>
</ul>
<p>Settings of <code>getGPUNetworkCostWeightsParameterValues</code>:</p>
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
<td><code>costLimit</code></td>
<td><code>number</code></td>
<td>Catchment cost: an entry becomes a neighbor when its cost is finite, non-negative and <code>&#x3C;= costLimit</code>. <code>Infinity</code> keeps every reached entry (not allowed with <code>'gaussian'</code>).</td>
</tr>
<tr>
<td><code>decay</code> (optional)</td>
<td><code>GPUNetworkCostWeightsDecay</code></td>
<td>Decay that turns a cost into a weight. Defaults to <code>'none'</code>.</td>
</tr>
<tr>
<td><code>beta</code> (optional)</td>
<td><code>number</code></td>
<td>Rate of the <code>'exponential'</code> and <code>'power'</code> decays, finite and non-negative. Defaults to 0.</td>
</tr>
<tr>
<td><code>minimumCost</code> (optional)</td>
<td><code>number</code></td>
<td>Floor of the <code>'power'</code> decay so a zero cost stays finite, positive and finite. Defaults to 1.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h3 id="gpunetworkisochrones"><code>GPUNetworkIsochrones</code></h3>
<p>Turn road-network travel times into cost-band triangles or cell outlines around the reachable streets.</p>
<p>::lens{scenario=gpu-network.network-isochrones rung=result caption="Read the travel-time bands across 4,622 output triangles radiating from the start."}</p>
<p>Isochrone polygons from a network <code>topology</code> with <code>nodePositions</code> (pgRouting <code>pgr_drivingDistance</code>
with a buffer, QGIS service area polygons). Costs come from the caller (<code>costs</code>, for example
<code>GPUNetworkServiceAreas</code> <code>output.costs</code>) or from <code>startNodes</code>, which runs <code>GPUNetworkReachability</code>
and writes <code>costs</code>; <code>costs</code> and <code>assignments</code> therefore stay top-level props, inputs or
outputs depending on <code>startNodes</code>. With <code>startNodes</code>, <code>output.status.converged</code>
(one row) reports whether that search reached a fixpoint within <code>maximumIterations</code>, and
<code>localIterations</code> sets its hops per round. A 0 means the costs, and so every polygon, are truncated
too small. As on <code>GPUNetworkReachability</code> and <code>GPUNetworkServiceAreas</code> the flag is optional, and
omitting it accepts that truncation unchecked. The optional one-row <code>uint32</code>
<code>output.status.iterationLimitReached</code> is the complement of <code>converged</code> and can be requested alone.
Both the plain reachability search and the facility-label search forward it; with assignments,
label-round exhaustion also sets it. Both ports require <code>startNodes</code> and separate buffers from
inputs, geometry outputs and each other. Budget exits still publish incumbent costs and geometry.</p>
<p>Every published output must occupy a distinct logical byte range, including <code>costs</code> and
<code>assignments</code> when <code>startNodes</code> is supplied. Construction rejects aliases and names the two output
ports. Outputs written in separate passes may use disjoint ranges of one allocation; bindings
used together must also satisfy their alignment and read/write separation requirements.
The contributor snapshots declared option and port fields at construction, including inherited
and non-enumerable getters on class-backed output maps; later caller mutations do not change
validated bindings. GPU view identities are retained,
so their contents remain per-frame inputs. The labelled decode, triangle-label and cell-label
passes validate their co-bound ports at construction as well.</p>
<ul>
<li><strong>Raster path.</strong> Every edge is sampled at up to <code>raster.maximumSamplesPerEdge</code> points. The cost is
interpolated linearly from the source node along the edge weight, and the minimum (default) or
maximum cost is written to the pixels within the walking buffer (<code>bufferRadius</code>, with cost
<code>cost + walkCostPerUnit * distance</code>) using integer atomics on order-preserving float bits, so the
result is independent of thread order. In <code>'max'</code> mode the cost of a sample is first reduced to
the cheaper of the two directions of its road (<code>min(cost[u] + t w, cost[v] + (1 - t) wReverse)</code>
when a slot <code>v -> u</code> exists), so the maximum runs over the samples and roads within the buffer of
a pixel and never over a road and its own reverse direction. The raster feeds <code>GPUIsobands</code>: band
<code>k</code> is cost in <code>(costBreaks[k - 1], costBreaks[k]]</code> (a cost exactly on a break is within it, as in
<code>GPUNetworkReachability</code> bands), and the last band is beyond the last break, which includes unreached
space. The default <code>lastBand</code> is <code>breakCount - 1</code>, so unreached space gets no triangles.</li>
<li><strong>Cell path.</strong> Nodes with cost at most <code>cellCostLimit</code> are keyed to H3 or Quadbin cells
(<code>GPUCellAggregation</code>, or <code>GPUPointToCell</code> then <code>GPUCellAggregation</code>) and outlined with
<a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-cells#gpucellsetoutline"><code>GPUCellSetOutline</code></a>.
Positions must be longitude and latitude. <code>cellOutline.rings</code> (same options as
<code>GPUCellSetOutline</code> <code>rings</code>: <code>vertexTolerance</code>, <code>normalizeWinding</code>, <code>output</code>) chains the outline into
closed isochrone rings with shells, holes, shell assignment and an optional polygon layout
(<a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-cells#gpusegmentringassembly"><code>GPUSegmentRingAssembly</code></a>).
One run is one cost limit, so bands need one run each. The <code>polygons</code> output plugs into
<code>GPUPointInPolygonJoin</code> for "is this demand point inside the isochrone", which
<code>addDriveTimeCatchmentRecipe</code> wires as <code>isochrones.joinDemand</code>. The recipe is exported from this
entry (<code>@luma-contributors/analysis/gpu-network</code>) and documented with the other recipes in
<a href="/luma-contributors/docs/api/gpu-spatial-analysis">spatial analysis</a>.
For the recipe's <code>maximumSnapDistance</code>, no warning is possible because the distance is a GPU view; see <a href="/luma-contributors/docs/api/gpu-spatial-analysis/concepts#precision-of-longitudelatitude-positions">Precision of longitude/latitude positions</a>.</li>
<li><strong>Parameters.</strong> Per-frame values are packed with
<code>getGPUNetworkIsochronesParameterValues({breakCount, extent, bufferRadius, walkCostPerUnit, firstBand, lastBand, cellCostLimit})</code> into a 12-element float32 view. <code>costBreaks</code> is a float32 view
whose length is the maximum break count. The raster size, <code>maximumBufferPixels</code> (default 6, at most
16), <code>maximumSamplesPerEdge</code> (default 64, at most 1024), <code>unreachedCost</code> (default 1e30) and the
mode are compile-time.</li>
<li><strong>Accuracy.</strong> Pixel accurate. Edges are straight node-to-node segments, the outer contour is inset
by up to one pixel (interpolation toward <code>unreachedCost</code>), and the effective buffer is at least half
a pixel diagonal, truncated at <code>maximumBufferPixels</code>. The raster is planar in the units of
<code>topology.nodePositions</code>, so use projected meters for an isotropic buffer. An edge longer than
<code>maximumSamplesPerEdge * bufferRadius / 2</code> leaves gaps.</li>
<li><strong>Facilities.</strong> <code>assignments</code> (a <code>uint32</code> view of the facility per node) is an input, or, when <code>startNodes</code>
is given, an output: the nearest-facility allocation of <code>GPUNetworkServiceAreas</code> (ties to the lowest
row; <code>maximumLabelIterations</code> bounds its rounds). Raster and outline capacity status is in
<code>raster.output.status</code> and <code>cellOutline.output.status</code>. <code>raster.output.pixelFacilities</code> (minimum mode, at most 255
sources; the pixel word keeps the top 24 cost bits, relative precision 2^-15) and
<code>raster.output.triangleFacilities</code> label the raster path, and <code>raster.rings</code> writes band rings
(<a href="/luma-contributors/docs/api/gpu-raster/operations-analysis#gpuisolines-and-gpuisobands"><code>GPUIsobandRings</code></a>).
On the cell path <code>cellOutline.byFacility</code> with <code>cellFacilities</code> gives each cell the facility of its
cheapest node and outlines with groups, so rings never mix facilities (Quadbin is keyed through
<code>GPUPointToCell</code>). A supplied <code>cellOutline.table</code> determines the row capacity from
<code>table.cells.length</code>; <code>tableCapacity</code> applies only when the table is omitted (default: node
count). Caller-owned <code>cellFacilities</code> must have exactly that many rows; omitted facility and
minimum-cost buffers use the same capacity. With a supplied table, <code>tableCapacity</code> is ignored,
including zero. Without a supplied table, <code>tableCapacity</code> must be at least the node count: the
internal cell-loss flag is not published, so potentially lossy transient tables are refused.
With a smaller caller-owned table, check <code>table.status.overflow</code> before treating any derived
outline or rings as complete; <code>cellOutline.output.status.overflow</code> reports segment-capacity
loss only. The drive-time catchment recipe exposes the supplied table's status as the
<code>isochrone-cells</code> stage and enforces the same transient-capacity rule through this contributor.</li>
<li><strong>Coordinates.</strong> The public <code>spatialContext</code> always holds a resolved context. Omitted means
native planar for <code>raster</code>, or <code>{coordinateSpace: 'longitude-latitude', metric: 'none'}</code> for
<code>cellOutline</code> alone. With both outputs and no context, the raster's planar default applies and
cell-outline positions remain unchecked for compatibility. An explicit context checks both
paths: longitude/latitude with <code>raster</code> throws (project into an AEQD frame first), and planar
with <code>cellOutline</code> throws. Lambert equal-area frames are rejected because buffers measure distance.</li>
<li><strong>Limits.</strong> The cell path samples reached nodes only, so long edges on sparse networks drop cells. The
rings follow the cell outline, so they have cell resolution. Three mid-latitude H3 pentagons (resolutions
2, 3 and 5; alone, in disks, as holes and open rings) match h3-js <code>cellsToMultiPolygon</code>; the two polar
pentagons enclose a pole and have no planar lng/lat orientation.
Before declaring search nodes or allocating scratch, the contributor checks caller STORAGE
bindings, its direct raster/cell scratch bindings, and raster-ring edge/ring capacity limits.
Child contributors also validate their own intermediate allocations. In particular,
a raster needs <code>4 * width * height</code> bytes for its encoded pixels, and the same amount for values
when no values output is supplied. An 8,192 by 8,192 raster needs 256 MiB per such binding and is
rejected on a device with a 128 MiB storage binding limit. A transient cell table needs
<code>8 * tableCapacity</code> bytes for its cell keys; facility scratch uses four bytes per table row.
Raster rings use <code>getGPUIsobandRingsMaximumEdgeCapacity</code> and
<code>getGPUIsobandRingsMaximumRingCapacity</code> for the outer preflight, so on a 128 MiB binding device
<code>raster.rings.edgeCapacity</code> is at most 8,388,608.</li>
</ul>
<h4 id="how-it-computes-16">How it computes</h4>
<p>{/* lens:GPUNetworkIsochrones */}</p>
<p>::lens{scenario=gpu-network.network-isochrones rung=datum bind=page datum=1088 caption="Inspect the start intersection: its travel cost is 0 seconds before the raster splat."}</p>
<p>::lens{scenario=gpu-network.network-isochrones rung=formula}</p>
<p>::lens{scenario=gpu-network.network-isochrones rung=kernel}</p>
<p>::lens{scenario=gpu-network.network-isochrones rung=graph}</p>
<h4 id="adddrivetimecatchmentrecipe"><code>addDriveTimeCatchmentRecipe</code></h4>
<p>Snap facilities and demand points to roads, assign drive-time catchments and count demand per cost band.</p>
<p>Coincident facility and demand endpoints remain accessible at zero cost even when a road is
one-way. Interior movement still obeys its permitted directions. The recipe's min-mode raster
includes the direct paths from each snapped facility to its road endpoints, so early bands can
appear around facilities inside streets even when neither endpoint lies in that band. The recipe
requires min raster mode; the standalone isochrone contributor still supports max mode.
<code>GPUNetworkIsochronesRaster.sourcePaths</code> supplies these additional directed paths through the
same splatter, with optional assignments in the main network's start-row identity space.</p>
<p>::lens{scenario=gpu-network.drive-time-catchment-recipe rung=result caption="Compare drive times from 100 demand points to their nearest of 4 facilities."}</p>
<p>A snapped point lies at fraction <code>f</code> of the straight road between the two end nodes of its edge. The road is
driven in a direction only when the topology has a passable edge that way, at the weight of the cheapest
one, so a one-way edge is driven one way. An interior facility gets one start row (the other row of
<code>startNodes</code> is <code>0xffffffff</code>); a facility coincident with an endpoint also seeds that node at zero cost. The demand time is the smallest of <code>cost[source] + f * w(source -> target)</code>,
<code>cost[target] + (1 - f) * w(target -> source)</code> and, for each facility snapped to the same road, the way
straight along it (<code>abs(f - f_facility) * w</code> on a two-way road of one cost). Demand beside its facility
therefore gets the short way along the road, not the way out to an end node and back. The facilities are
sorted by road and each demand point binary-searches its own, so it is compared only with the facilities
snapped to that road. Band <code>k</code> holds the times in
<code>(costBreaks[k - 1], costBreaks[k]]</code>: a time exactly on a break is within it, as in the bands of
<code>GPUNetworkReachability</code> and <code>GPUNetworkIsochrones</code>. An unsnapped or unreached demand point has the time
<code>+Infinity</code> and no band.</p>
<h4 id="how-it-computes-17">How it computes</h4>
<p>{/* lens:addDriveTimeCatchmentRecipe */}</p>
<p>::lens{scenario=gpu-network.drive-time-catchment-recipe rung=datum bind=page datum=5 caption="Inspect this demand point: its drive time is about 58.6 seconds, in band 0."}</p>
<p>::lens{scenario=gpu-network.drive-time-catchment-recipe rung=formula}</p>
<p>::lens{scenario=gpu-network.drive-time-catchment-recipe rung=kernel}</p>
<p>::lens{scenario=gpu-network.drive-time-catchment-recipe rung=graph}</p>
<p>{/* parameters: GPUNetworkIsochrones generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h4 id="gpunetworkisochrones-parameters"><code>GPUNetworkIsochrones</code> parameters</h4>
<ul>
<li>Packer: <code>getGPUNetworkIsochronesParameterValues(settings)</code>, writes <code>GPU_NETWORK_ISOCHRONES_PARAMETER_LENGTH</code> (12) <code>float32</code> values.</li>
</ul>
<p>Settings of <code>getGPUNetworkIsochronesParameterValues</code>:</p>
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
<td><code>breakCount</code></td>
<td><code>number</code></td>
<td>Number of active ascending cost breaks, at most <code>costBreaks.length</code>. <code>breakCount + 1</code> bands exist: band <code>k</code> holds cost in <code>(costBreaks[k - 1], costBreaks[k]]</code>, band 0 everything at or below the first break and the last band everything above the last active break (including unreached pixels). A cost exactly on a break is within it, the rule of <code>GPUNetworkReachability</code> bands.</td>
</tr>
<tr>
<td><code>extent</code></td>
<td><code>readonly [number, number, number, number]</code></td>
<td>Raster extent <code>[minX, minY, maxX, maxY]</code> in the context's length unit. Row 0 is at <code>minY</code>.</td>
</tr>
<tr>
<td><code>bufferRadius</code> (optional)</td>
<td><code>number</code></td>
<td>Walking buffer radius in the context's length unit. Pixels within it of an edge sample get that sample's cost plus <code>walkCostPerUnit</code> times the distance. The effective radius is at least half a pixel diagonal so every sample marks its own pixel. Defaults to 0.</td>
</tr>
<tr>
<td><code>walkCostPerUnit</code> (optional)</td>
<td><code>number</code></td>
<td>Cost per context length unit walked off the network inside the buffer. Defaults to 0.</td>
</tr>
<tr>
<td><code>firstBand</code> (optional)</td>
<td><code>number</code></td>
<td>First band emitted as geometry. Defaults to 0.</td>
</tr>
<tr>
<td><code>lastBand</code> (optional)</td>
<td><code>number</code></td>
<td>Last band (inclusive) emitted as geometry. Defaults to <code>breakCount - 1</code>, so unreached space has no triangles.</td>
</tr>
<tr>
<td><code>cellCostLimit</code> (optional)</td>
<td><code>number</code></td>
<td>Cost limit of the cell-outline path: nodes with cost <code>&#x3C;=</code> it contribute their cell. Defaults to <code>Infinity</code>, every reached node.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h3 id="gpunetworknoding"><code>GPUNetworkNoding</code></h3>
<p>Split raw centrelines at their intersections, merge end points within a tolerance, and return nodes, edge pieces and a network.</p>
<p>::lens{scenario=gpu-network.network-noding rung=result caption="Compare 16 raw centrelines with 134 noded pieces and 87 nodes; bridges are not distinguished."}</p>
<p>Maximum: <code>intersectionCapacity</code> is at most <code>getGPUNetworkNodingMaximumIntersectionCapacity(device)</code> (1048576 on a default device), a quarter of <code>getGPULineSplitMaximumPieceBound(device)</code>.</p>
<p>Builds a routable network from raw linestrings. <code>GPULineSplit</code> pieces
(see <a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-lines#gpulinesplit-and-gpulinemerge">GPU Spatial Analysis</a>)
are the edges. Their end points are snapped by a per-frame <code>tolerance</code> (a one-row <code>float32</code> view; integer
grid cells <code>floor((x - origin) / tolerance)</code> counted from the minimum corner of the live end points, so UTM
northings keep exact cells; 0 means exact), deduplicated by a stable two-pass <code>GPUSort</code>, and the node ID
is the group rank. Cells are exact up to 2^24 per axis; a wider extent sets <code>output.status.overflow</code>. Props: <code>lines</code>, <code>intersectionCapacity</code>, <code>tolerance</code>, and <code>output</code> with <code>pieces</code>, <code>nodes</code>
(<code>positions</code> and <code>status.count</code>, optional <code>status.requiredCount</code>), <code>edges</code> (<code>fromNodes</code>, <code>toNodes</code>, <code>lengths</code>) and <code>topology</code>
(<code>offsets</code>, <code>neighbors</code>, <code>weights</code>, optional <code>edgeIds</code>), with optional <code>output.status.overflow</code> and <code>output.status.uncertainCount</code>.
The topology is undirected and feeds <code>GPUNetworkReachability</code>, <code>GPUNetworkServiceAreas</code> and the other
contributors directly. Every edge has two entries, one in the row of each end node; a loop (a closed ring,
or a piece whose two ends merge within the tolerance) has one entry in the row of its node, the rule of
<code>GPUNetworkCoarsening</code>, <code>GPUNetworkSubgraphFilter</code>, <code>GPUNetworkStatistics</code> and luma.gl's
<code>GPUGraphAlgorithmsTopology</code>.</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUNetworkNoding</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  lines: {kind: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'lines'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, positions, lineOffsets}, intersectionCapacity: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> &#x3C;&#x3C;</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> 16</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  tolerance: tolerance.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">importToGraph</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(graph),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  output: {</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    pieces, nodes: {positions: nodePositions, status: {count: nodeCount}},</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    edges: {fromNodes, toNodes, lengths}, topology: {offsets, neighbors, weights}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  }</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span></code></pre>
<p>Planar only: the snapping cells, <code>tolerance</code> and edge lengths are Euclidean, so a longitude/latitude
<code>spatialContext</code> throws. Project the lines to a planar frame first (for example <code>createGPULocalFrame</code>).</p>
<p>Limits: the tolerance merges end points only (end points that straddle a grid cell boundary stay separate),
and a line that stops near another line's interior is not connected to it. Crossings that should not
connect (bridges) cannot be excluded. Capacities are fixed, and <code>output.status.overflow</code> reports a truncated result.</p>
<h4 id="how-it-computes-18">How it computes</h4>
<p>{/* lens:GPUNetworkNoding */}</p>
<p>::lens{scenario=gpu-network.network-noding rung=datum bind=page datum=12 caption="Inspect this noded piece: its length is about 969.4 metres."}</p>
<p>::lens{scenario=gpu-network.network-noding rung=kernel}</p>
<p>::lens{scenario=gpu-network.network-noding rung=graph}</p>
<h3 id="gpunetworklinegraph"><code>GPUNetworkLineGraph</code></h3>
<p>Convert directed road edges into line-graph nodes and legal turns into arcs costed by travel and turn time.</p>
<p>::lens{scenario=gpu-network.network-line-graph rung=result caption="Compare road-edge midpoints linked by 5,635 legal turns; U-turns are excluded."}</p>
<p>Maximum: <code>topology</code> has at most <code>getGPUNetworkLineGraphMaximumEdgeCount(device)</code> (16777216 on a default device) edges, or <code>getGPUNetworkLineGraphMaximumGeodesicEdgeCount(device)</code> (8388608) with a great-circle <code>spatialContext</code>, because the edge direction vectors take 8 or 16 bytes per edge.</p>
<p>Turn-restricted routing as a line graph in CSR form: one node per directed edge of the road network and
one arc per allowed turn. The arc cost is <code>weights[next]</code> plus a turn cost from the turn angle: a base
<code>angleCost</code>, extra left and right costs, a U-turn cost or ban, and optional <code>bannedTurns</code> pairs. Props:
<code>topology</code> (with <code>nodePositions</code>), per-frame <code>parameters</code>
(<code>getGPUNetworkLineGraphParameterValues</code>), and <code>output</code>: a line-graph <code>topology</code> (<code>offsets</code>,
<code>neighbors</code>, <code>weights</code>, sized to the compile-time arc capacity, rows past the arc total hold
<code>0xffffffff</code> and -1), <code>arcCount</code> and the required <code>status.overflow</code>. Edge rows past <code>offsets[nodeCount]</code> get no arcs. <code>output.topology</code> is the <code>topology</code> of
<code>GPUNetworkReachability</code> and <code>GPUNetworkCostMatrix</code> unchanged; to route from an origin, start at the
out-edges of the origin node at their own weight. With a great-circle <code>spatialContext</code> positions are degrees and the turn
angle is measured between the great-circle arrival bearing of the incoming edge and the departure
bearing of the outgoing edge at the turn node, so turns are not skewed by the east-west stretch of
degrees. The active <code>bannedTurns</code> pairs are sorted by <code>fromEdge</code> on the GPU every frame and each turn
binary-searches them, so city-scale restriction sets cost <code>O(log bans)</code> per arc. The turn angle special-cases a
zero dot product because Metal returns the wrong sign for <code>atan2(y, -0.0)</code>. Start helpers and node
costs are not provided.</p>
<p>Optional <code>output.turnPenalties</code> has one float32 row per arc-capacity slot, aligned with
<code>output.topology.neighbors</code>; padding is -1. It publishes the turn cost before adding the following
edge's base cost, preserving small penalties alongside large costs. Pass its read-back column to
<code>prepareGPUNetworkPointRoutingTurns</code> for fractional-edge routing. All requested penalty rows are
rewritten on each encoding, and the extra output requires its own disjoint buffer.</p>
<h4 id="how-it-computes-19">How it computes</h4>
<p>{/* lens:GPUNetworkLineGraph */}</p>
<p>::lens{scenario=gpu-network.network-line-graph rung=datum bind=page datum=5 caption="Inspect this turn: its next edge is 1836, with cost about 44.8 seconds."}</p>
<p>::lens{scenario=gpu-network.network-line-graph rung=formula}</p>
<p>::lens{scenario=gpu-network.network-line-graph rung=kernel}</p>
<p>::lens{scenario=gpu-network.network-line-graph rung=graph}</p>
<p>{/* parameters: GPUNetworkLineGraph generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h4 id="gpunetworklinegraph-parameters"><code>GPUNetworkLineGraph</code> parameters</h4>
<ul>
<li>Packer: <code>getGPUNetworkLineGraphParameterValues(settings?)</code>, writes <code>GPU_NETWORK_LINE_GRAPH_PARAMETER_LENGTH</code> (8) <code>float32</code> values.</li>
</ul>
<p>Settings of <code>getGPUNetworkLineGraphParameterValues</code>:</p>
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
<td><code>angleCost</code> (optional)</td>
<td><code>number</code></td>
<td>Cost per radian of absolute turn angle, so a right angle costs <code>angleCost * pi / 2</code>. Defaults to 0.</td>
</tr>
<tr>
<td><code>leftTurnCost</code> (optional)</td>
<td><code>number</code></td>
<td>Extra cost of a left turn (counter-clockwise, <code>angle > straightAngle</code>). Defaults to 0.</td>
</tr>
<tr>
<td><code>rightTurnCost</code> (optional)</td>
<td><code>number</code></td>
<td>Extra cost of a right turn (clockwise, <code>angle &#x3C; -straightAngle</code>). Defaults to 0.</td>
</tr>
<tr>
<td><code>uTurnCost</code> (optional)</td>
<td><code>number</code></td>
<td>Cost of a U-turn (absolute angle at or above <code>uTurnAngle</code>), replacing the angle cost. A negative value bans U-turns. Defaults to -1 (banned).</td>
</tr>
<tr>
<td><code>straightAngle</code> (optional)</td>
<td><code>number</code></td>
<td>Absolute angle in radians below which a turn counts as straight for left and right extras. Defaults to 0.5.</td>
</tr>
<tr>
<td><code>uTurnAngle</code> (optional)</td>
<td><code>number</code></td>
<td>Absolute angle in radians at or above which a turn is a U-turn. Defaults to 2.9 (about 166 degrees).</td>
</tr>
<tr>
<td><code>bannedTurnCount</code> (optional)</td>
<td><code>number</code></td>
<td>Number of active banned turns, at most half the length of <code>bannedTurns</code>. Defaults to 0.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h3 id="gpunetworkcosttable"><code>GPUNetworkCostTable</code></h3>
<p>Search every node within a cost limit and store each reachable target and its cost in one sparse CSR table.</p>
<p>::lens{scenario=gpu-network.network-cost-table rung=result caption="Follow the start intersection's row of the 30-second table, which stores 23,449 entries network-wide."}</p>
<p>Precomputes a bounded all-pairs cost table of a static directed network, in the spirit of the UBODT
of fast map matching (Yang and Gidofalvi 2018). Row <code>u</code> lists every node <code>v</code> whose shortest-path cost
from <code>u</code> is finite and at most <code>costLimit</code>, <code>u</code> itself included at cost 0, as a CSR (<code>offsets</code>,
<code>targets</code>, <code>costs</code>) with ascending <code>targets</code>, so a consumer finds a cost by binary search of the row.
When <code>status.complete</code> is nonzero, the table is exact up to its limit: a target missing from a row
is farther than <code>costLimit</code> or unreachable.
<code>GPUMapMatching.routeTable</code> is the first consumer.</p>
<p>The contributor and both operation forms capture named budgets and port fields at construction.
Later changes to the caller's props do not change the validated command graph; buffer contents
remain per-frame inputs. Status words must occupy distinct storage even on class-instance bags.</p>
<ul>
<li>
<p>Props: <code>topology</code> (with <code>weights</code> as the travel cost), <code>costLimit</code> (one row, per frame), <code>laneCount</code>,
<code>maximumIterations</code>, <code>localIterations</code> (default 16) and <code>output</code>: <code>offsets</code> (<code>nodeCount + 1</code> rows),
<code>targets</code> (its length is the entry capacity), <code>costs</code> (same length) and optional
<code>status.overflow</code>, <code>status.requiredCount</code>, <code>status.converged</code> and <code>status.complete</code>.
The one-word <code>status.complete</code> is 1 only when every batch converged without capacity overflow;
it can be requested alone and is the word to forward to map matching.
Optional <code>status.iterationLimitReached</code> is <code>1 - converged</code> and can be requested alone. It reports
round-budget exhaustion independently of capacity overflow. On a budget exit, retained CSR rows
remain consistent, and <code>requiredCount</code> counts entries found so far; it is not a certificate of
the full table size. The CPU twin searches to completion when <code>maximumIterations</code> is omitted;
an explicit round budget binds the CPU search too. Local hops default to 16. On a budget exit it
publishes its incumbents and sets <code>iterationLimitReached</code> when its frontier remains nonempty.
CPU scheduling is deterministic per source; GPU lane packing and scheduling can change partial
incumbents near the budget boundary. Both forms report convergence independently of capacity.
Completeness also requires no capacity overflow: on the nine-node unit chain, capacity 3 after
convergence gives <code>converged=1, overflow=1, complete=0</code>, while capacity 81 gives
<code>converged=1, overflow=0, complete=1</code>.
All scalar ports reset
on each encoding and need separate physical buffers from inputs and other outputs.</p>
</li>
<li>
<p>Method: source nodes are searched <code>laneCount</code> at a time with one lane-expanded
<code>GPUNetworkReachability</code> per batch into one scratch block, then a per-lane count, a single-workgroup
scan of the batch's lane counts on top of a GPU running entry cursor (which writes the batch's
<code>offsets</code> rows), and a per-lane ordered fill. The batch index never reaches the shader, so every
batch shares pipelines. Converged costs are the f32 fixpoint of reachability and are identical for
every <code>laneCount</code>. Partial incumbents can vary before convergence. The CPU table rounds every
path addition to float32 before the inclusive cutoff test, preserving membership when a CPU
table feeds GPU matching; for example, float32 <code>0.1 + 0.4</code> is admitted at cutoff <code>0.5</code>.</p>
</li>
<li>
<p>Capacity: entries past <code>output.targets.length</code> are dropped and <code>offsets</code> are clamped, so a reader
always sees a consistent, possibly truncated CSR. <code>status.overflow</code> is 1 and <code>status.requiredCount</code>
is the exact entry count (saturating at <code>0xffffffff</code>) for the grow loop. Do not use a table that
overflowed or did not converge. Forward <code>status.complete</code> as <code>routeTable.status.complete</code>:
<code>GPUMapMatching</code> falls back to A* whenever this word is zero or omitted. It does not turn missing
entries from unfinished rows into track breaks; the fallback reports its own route-budget limits.</p>
</li>
<li>
<p>Scaling: each batch sweeps its <code>laneCount x nodeCount</code> scratch block, so work and memory traffic are
<code>O(laneCount * nodeCount)</code> words per batch and
<code>O(ceil(nodeCount / laneCount) * laneCount * nodeCount)</code> words per build including padding,
plus the searches (quadratic overall because <code>laneCount &#x3C;= nodeCount</code>).
Build once and reuse while weights are static; the table holds only entries within <code>costLimit</code>.
The largest scratch binding is the two frontier queues: <code>8 * laneCount * nodeCount</code> bytes.
Every topology and output view must also fit its storage binding.</p>
</li>
<li>
<p>Compile cost: every round of every batch is an unrolled command node, <code>2 + batchCount * (6 + maximumIterations)</code>
nodes, plus one per batch when <code>status.converged</code>, <code>status.iterationLimitReached</code> or <code>status.complete</code>
is requested. The exact count is the read-only property <code>estimatedCommandNodeCount</code>; construction
rejects more than 32,768 commands. For example, 50,000 nodes with 128 lanes and 64 rounds
use 27,763 commands with completeness tracking and 51,200,000 frontier bytes, below a 128 MiB binding.
Lane count must be supplied explicitly for this example. At 100,000 nodes and 64 rounds, no lane count
fits both limits: at most 167 lanes fit the frontier, but at least 217 are needed for commands with
completeness tracking. Lowering the rounds may produce an incomplete table; check <code>status.complete</code>.</p>
</li>
</ul>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// Build once for a static network. The weights must equal the edge lengths GPUMapMatching measures</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// from the node positions (here 100 m blocks, so a 700 m limit covers 7 blocks).</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> table</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUNetworkCostTable</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  id: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'street-table'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  topology: {offsets, neighbors, weights: edgeLengths},</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  costLimit: tableLimit,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  output: {</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    offsets: tableOffsets, targets: tableTargets, costs: tableCosts,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    status: {complete: tableComplete, overflow: tableOverflow, requiredCount: tableRequired, converged: tableConverged}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  }</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">});</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(table);</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUMapMatching</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  positions, trackOffsets, topology: {offsets, neighbors, nodePositions}, parameters,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  cellSize: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">60</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, bounds, output: {matchedEdges},</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  routeTable: {</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    offsets: tableOffsets, targets: tableTargets, costs: tableCosts,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    costLimit: tableLimit, status: {complete: tableComplete}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  }</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span></code></pre>
<h4 id="how-it-computes-20">How it computes</h4>
<p>{/* lens:GPUNetworkCostTable */}</p>
<p>::lens{scenario=gpu-network.network-cost-table rung=datum bind=page datum=1088 caption="Inspect the start row: offsets 21,357 to 21,373 delimit 16 entries."}</p>
<p>::lens{scenario=gpu-network.network-cost-table rung=formula}</p>
<p>::lens{scenario=gpu-network.network-cost-table rung=kernel}</p>
<p>::lens{scenario=gpu-network.network-cost-table rung=graph}</p>
<h3 id="gpumapmatching"><code>GPUMapMatching</code></h3>
<p>Rows are bounded by min(floor(bindingBytes / 12), 16,777,215). Optional
output.status.candidateOverflow mirrors intermediate-table overflow; routeOverflow counts dropped
route nodes. Fixed matched-row output <code>overflow</code> is always zero. Check intermediate loss before
treating a route as complete.</p>
<p>Match noisy GPS tracks to directed road edges with candidate edges, bounded route searches and a Viterbi decode.</p>
<p>In a great-circle context, both CPU and GPU matching project onto the same wrapped start-to-end
longitude arc and compare the projected candidate with both endpoints. A rounded distance tie
preserves an interior projection; ties between endpoint candidates select the start. The selected
fraction and snap distance therefore describe the same point, including for remote antimeridian
queries. Long edges still use the documented local projection approximation.</p>
<p>::lens{scenario=gpu-network.map-matching rung=result caption="Compare 3 simulated GPS tracks with their road-snapped runs; 137 of 139 fixes matched."}</p>
<p>Pass optional <code>spatialContext</code> in the settings of <code>getGPUMapMatchingParameterValues</code> to warn when finite positive <code>sigma</code> or <code>searchRadius</code> is below <code>10·Q</code> (16.97 m with the default sphere); see <a href="/luma-contributors/docs/api/gpu-spatial-analysis/concepts#precision-of-longitudelatitude-positions">Precision of longitude/latitude positions</a>.</p>
<p>Maximums: <code>topology</code> has at most <code>getGPUMapMatchingMaximumEdgeCount(device)</code> (11184810 on a default device) edges and <code>entryCapacity</code> is at most <code>getGPUMapMatchingMaximumEntryCapacity(device)</code> (33554432); the default <code>entryCapacity</code> is capped there and <code>status.candidateOverflow</code> reports lost entries. The per-point scratch (candidates 12 bytes per candidate, the route table <code>candidateCount * (candidateCount + 1) * 4</code> bytes) must fit one storage binding: <code>positions</code> has at most <code>getGPUMapMatchingMaximumPointCount(device, candidateCount, hasTraversedEdges)</code> rows (466033 at 8 candidates on a default device, with <code>output.traversedEdges</code> the 20-byte hop table per point binds only at one candidate), and <code>getCommandNodes</code> throws a <code>positions rows must be at most ...</code> error past it. Lower <code>candidateCount</code> or split the tracks into several <code>GPUMapMatching</code> instances.</p>
<p>Hidden Markov model map matching of GPS tracks to a directed network (Newson and Krumm 2009). Per point,
up to <code>candidateCount</code> (at most 8, default 8: a directed edge row is one candidate, so the default keeps the
4 nearest two-way roads) distinct directed edges within <code>searchRadius</code> come from an edge grid
rebuilt every encoding (<code>cellSize</code> and <code>bounds</code> are compile-time). The emission is Gaussian on the
perpendicular distance (<code>sigma</code>) and the transition exponential on <code>|straight - route|</code> (<code>beta</code>). The route
is the rest of the current edge, a bounded A* search (it stops at <code>routeFactor * straight + routeSlack</code>,
exits early once every target is settled, and uses a <code>routeNodeBudget</code> node table) and the start of the
next edge; a route longer than <code>routeFactor * straight + routeSlack</code>, partial end edges included, is
infeasible. Viterbi runs one thread per track.</p>
<ul>
<li>Props: <code>positions</code>, <code>trackOffsets</code>, the network <code>topology</code> (<code>offsets</code>, <code>neighbors</code>, <code>nodePositions</code>;
weights unused), <code>parameters</code> (<code>getGPUMapMatchingParameterValues</code>: <code>sigma</code>, <code>beta</code>, <code>searchRadius</code>, <code>routeFactor</code>, <code>routeSlack</code>, per
frame), <code>candidateCount</code>, <code>cellSize</code>, <code>bounds</code>, <code>entryCapacity</code>, <code>routeNodeBudget</code>,
<code>minimumFixSpacingSigmas</code>, <code>traversedEdgeCapacity</code> and <code>spatialContext</code>.
The contributor, both operation forms and the track-quality recipe capture named model fields,
bounds and port/status maps once, including inherited getters. Later props mutation does not
alter compiled budgets; input buffer contents remain per-frame values.</li>
<li>Local geographic snaps retain an interior projection when its haversine term ties the endpoint terms
in float32. Local candidates are compared before converting those terms to distances, so inverse
trigonometric rounding does not move a midpoint snap onto an endpoint. Distant queries retain
central-angle comparisons to preserve resolution near antipodes.</li>
<li>Fix spacing (<code>minimumFixSpacingSigmas</code>, compile-time, off by default): Newson and Krumm drop a fix
within 2 <code>sigma</code> of the previous retained fix, so stationary or backward jitter does not look like a
move. Pass 2 to do the same, measured with the frame's <code>sigma</code> and the context's distance. Without it
a fix that jitters backward along a one-way edge has no feasible transition (the route would have to
leave the edge and return within <code>routeFactor * straight + routeSlack</code>) and breaks the track. The
first fix of a track is always kept and each later fix is measured from the last retained one. A
dropped fix is never a candidate source or target and is unmatched in every output
(<code>matchedEdges</code> is <code>GPU_MAP_MATCHING_NONE</code>, <code>breaks</code> is 0, it is not in <code>matchedCount</code>). Dropped fixes
are not matched to the edge of their neighbors, so resample or interpolate them yourself if you need a
value per fix.</li>
<li>Score renormalization: the forward pass subtracts the best score of each step from every score of that
step. This moves all candidates of a step equally, so the chosen path is unchanged, and it keeps the
<code>float32</code> scores small over a track of any length (unnormalized, a 100000-fix track reaches 1e5 to 1e6,
where one <code>float32</code> step is 0.008 to 0.06 and close candidates tie). <code>trackLogLikelihoods</code> adds the
subtracted amounts back, as a whole part and a fraction, so it stays accurate to one <code>float32</code> rounding of
its own magnitude; there is no option to set.</li>
<li>Geodesic: with <code>spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'great-circle'}</code> positions
and nodes are degrees and every distance (emission, straight line, edge and route lengths, and the
distance parameters) is in the context's length unit (meters or kilometers); the A* heuristic stays consistent. <code>cellSize</code> and <code>bounds</code> stay in
degrees, the candidate window widens in longitude by <code>1 / cos</code> of its highest latitude, and longitudes
are unwrapped around the grid center, so give bounds across the antimeridian (for example 179.9 to
180.1) for a network that crosses it.</li>
<li>Outputs (<code>output</code>): <code>matchedEdges</code> (a CSR row per point; <code>GPU_MAP_MATCHING_NONE</code> when unmatched),
<code>matchedFractions</code>, <code>matchedOffsets</code>, <code>snapDistances</code>, <code>snappedPositions</code>, <code>breaks</code>,
<code>trackLogLikelihoods</code>, <code>matchedCount</code>, <code>breakCount</code>, <code>traversedEdges</code>, <code>status.candidateOverflow</code>, <code>status.overflow</code> and <code>status.routeOverflow</code>. A break restarts
the model where no transition is feasible or a point has no candidate (<code>breaks</code> value 1). A break
where a route search into the point dropped nodes is a budget break (value 2): the route may exist
but did not fit the table. <code>status.candidateOverflow</code> is 1 when the edge grid, route search or traversal recovery lost candidates; fixed matched rows fit, so <code>status.overflow</code> is zero, and
<code>status.routeOverflow</code> counts the dropped route nodes (saturating at <code>0xffffffff</code>).</li>
<li>Traversed route (<code>output.traversedEdges</code>, needs <code>traversedEdgeCapacity</code>): <code>matchedEdges</code> is the edge each
fix snaps to; the route is the edges passed between fixes, including edges that no fix lands on. The
caller owns <code>{edges, counts, status?: {overflow?, candidateOverflow?}}</code>: <code>edges</code> has <code>trackCount * traversedEdgeCapacity</code> rows and track
<code>t</code> writes its list, in travel order, from row <code>t * traversedEdgeCapacity</code>; <code>counts[t]</code> is the number of
entries the track needs, so a value above <code>traversedEdgeCapacity</code> tells how much capacity to allocate and
rows past the capacity are not stored; traversal <code>status.overflow</code> is 1 only when a track's list
exceeds capacity. Traversal <code>status.candidateOverflow</code> inherits grid/scoring loss and is also 1
when recovery drops nodes or cannot reconstruct a hop; an unrecovered hop writes a
<code>GPU_MAP_MATCHING_NONE</code> separator before its end
edge. The list holds the first matched edge, then for each recovered hop the shortest-route edges followed by the
next fix's edge, and nothing for a fix that stays on its edge ahead of the previous fix. One
<code>GPU_MAP_MATCHING_NONE</code> entry separates sub-tracks after a break or an unmatched fix, so counting
traversals skips it and invents no adjacency across the break. Each hop repeats the bounded search of the
route kernel for the chosen candidates only, one thread per track in sequence, so it costs about one more
route pass over the matched hops; it needs 20 bytes per point of scratch.
<a href="/luma-contributors/docs/api/gpu-network/map-matched-edge-volumes-mob05"><code>GPUMapMatchedEdgeVolumes</code></a> counts edge volumes over
these lists. Forward both traversal status words; its required <code>output.status.candidateOverflow</code>
marks retained volumes/movements as incomplete, and a requested nested transition candidate flag
includes that loss. Missing matching completeness is refused; only fixed external sequences may
assert <code>completeness: 'known-complete'</code>. With grid/scoring loss the counts are conditional on the
retained match, not lower bounds on an exact match's per-edge counts. Without <code>traversedEdges</code>
nothing changes.</li>
<li><code>routeTable</code> (optional): a <code>GPUNetworkCostTable</code> of the same network, built with <code>weights</code> equal to the
edge lengths this contributor measures from the node positions (in the context's length unit), passed as <code>{offsets, targets, costs, costLimit, status?: {complete}}</code>. A route search whose
allowance (<code>routeFactor * straight + routeSlack</code> minus the tail of the source edge) is within the table's
<code>costLimit</code> reads its network costs by binary search of the source node's row, with no search and no node
budget, so it is exact and never counted in <code>status.routeOverflow</code>. Searches above the limit, or all of
them while the table's <code>complete</code> flag is zero or omitted, run the bounded A* search as before, so there is no cliff and
no extra flag. Matched edges equal the A*-only results where A* is exact; route lengths differ only by f32
rounding of the summed costs (about <code>2^-24</code> relative per hop). The table is built once and reused, which
pays off for large track batches. Without <code>routeTable</code> nothing changes.</li>
<li>Route budget: a full node table keeps the open nodes with the best A* keys and drops the rest. A
search needs about as many nodes as the street blocks between two samples. On a 100 m street grid the
default 64 covers samples up to about 800 m apart and 128 up to about 1.1 km. At 1 km and 64 it still
misses routes (80 of 227 in a mimic, 171 under the old drop-in-discovery-order policy, which reported
nothing). Each miss is reported, so sample more densely or raise <code>routeNodeBudget</code> when <code>status.candidateOverflow</code> is
set. CSR rows past <code>offsets[nodeCount]</code> are not part of the network.</li>
<li>CPU twin: an explicitly supplied <code>routeNodeBudget</code> binds fallback and traversal-recovery searches,
including dropped-node counts, budget breaks and traversal separators. Certified-table lookups
binary-search only the source CSR row, independent of spare capacity. Without a budget CPU searches
remain uncapped. A complete cost table within the hop allowance bypasses fallback search in
both backends; traversal reconstruction still needs its own bounded search. CPU arithmetic is
float64 outside the precomputed table, so boundary and tie differences remain documented.</li>
<li>The route passes run per <code>(point, previous candidate)</code> work item, so Viterbi is a cheap recursion: on a
New York scene the matching graph went from 923 to 30 ms at a budget of 64 (20 ms at 40), with identical
results.</li>
<li>Limits: distances are planar unless a great-circle <code>spatialContext</code> is given, and edges are straight
segments in the coordinates, so on longitude/latitude they should be road-length. The bounded search is
an approximation that can overestimate or miss routes when samples are far apart for the budget
(always reported, see above), and the route table is <code>pointCount * k * (k + 1)</code> floats (checked against the storage binding limit, see Maximums). A two-way road needs both directions, and each takes a candidate slot.
Exact routes and time-aware transitions are not provided.</li>
</ul>
<h4 id="how-it-computes-21">How it computes</h4>
<p>{/* lens:GPUMapMatching */}</p>
<p>::lens{scenario=gpu-network.map-matching rung=datum bind=page datum=40 caption="Inspect this GPS fix: it snaps about 25.05 metres to edge 418 without a break."}</p>
<p>::lens{scenario=gpu-network.map-matching rung=kernel}</p>
<p>::lens{scenario=gpu-network.map-matching rung=graph}</p>
<p>{/* parameters: GPUMapMatching generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h4 id="gpumapmatching-parameters"><code>GPUMapMatching</code> parameters</h4>
<ul>
<li>Packer: <code>getGPUMapMatchingParameterValues(settings)</code>, writes <code>GPU_MAP_MATCHING_PARAMETER_LENGTH</code> (8) <code>float32</code> values.</li>
</ul>
<p>Settings of <code>getGPUMapMatchingParameterValues</code>:</p>
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
<td><code>spatialContext</code> (optional)</td>
<td><code>GPUSpatialContext</code></td>
<td>Spatial context of the positions; when it is longitude/latitude, a distance under 10 float32 quanta (about 17 m) logs a warning (see Precision of longitude/latitude positions). Does not change the packed values.</td>
</tr>
<tr>
<td><code>sigma</code></td>
<td><code>number</code></td>
<td>GPS noise standard deviation in the context's length unit. Must be positive.</td>
</tr>
<tr>
<td><code>beta</code></td>
<td><code>number</code></td>
<td>Scale in the context's length unit of <code>exp(-|straight - route| / beta)</code>. Must be positive.</td>
</tr>
<tr>
<td><code>searchRadius</code></td>
<td><code>number</code></td>
<td>Candidate search radius in the context's length unit. Edges farther away are not candidates. Values above <code>2 * cellSize</code> are clamped to <code>2 * cellSize</code> (on longitude/latitude, to the context's length unit equivalent of <code>2 * cellSize</code> degrees of latitude).</td>
</tr>
<tr>
<td><code>routeFactor</code> (optional)</td>
<td><code>number</code></td>
<td>A route, including the partial edges at both ends, may be at most <code>routeFactor * straightDistance + routeSlack</code> long; longer routes are infeasible. Defaults to 3.</td>
</tr>
<tr>
<td><code>routeSlack</code> (optional)</td>
<td><code>number</code></td>
<td>Additive route allowance in the context's length unit, see <code>routeFactor</code>. Defaults to 50 (50 m in meters; pass for example 0.05 with a kilometer context).</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h3 id="gpunetworkkfunction"><code>GPUNetworkKFunction</code></h3>
<p>Optional output.status.converged is one only when every shortest-path batch reaches a fixpoint;
iterationLimitReached is one if any batch exhausts its round budget. Snapping overflow remains a
separate status.</p>
<p>Count crash pairs by network distance band and return the network K value at each distance threshold.</p>
<p>::lens{scenario=gpu-network.network-k-function rung=result caption="Compare K for 58 snapped crashes against the seeded simulation minimum, mean and maximum."}</p>
<p>For <code>maximumSnapDistance</code> and <code>maximumDistance</code>, no warning is possible because the distance is a GPU view; see <a href="/luma-contributors/docs/api/gpu-spatial-analysis/concepts#precision-of-longitudelatitude-positions">Precision of longitude/latitude positions</a>.</p>
<p>Maximum: <code>rowsPerBlock * events</code> is at most 4294967295 (one 1-D dispatch), checked under the contributor's own ID. Each row block unrolls <code>maximumIterations + 6</code> command nodes; the read-only <code>estimatedCommandNodeCount</code> (an upper bound within 192 of the true count, which covers the snapping, reduction and scan primitives) must be at most 32,768, so raise <code>rowsPerBlock</code> or lower <code>simulationCount</code> or <code>maximumIterations</code> for large pattern counts.</p>
<p>Network-constrained Ripley K function (Okabe and Yamada; spaghetti <code>GlobalAutoK</code>). Events are snapped with
<code>GPUNetworkSnapping</code>, then multi-source reachability searches limited to <code>maximumDistance</code> run in blocks of
<code>rowsPerBlock</code> rows (which bounds scratch memory). The lane-expanded CSR is built once for all rows and
shared by every block, so <code>rowsPerBlock</code> trades only the node count of the relax and initialize passes
against scratch memory. Unordered pairs below each of <code>bandCount</code>
thresholds are counted with integer atomics and <code>K = 2 * pairs * L / n^2</code> (spaghetti's normalization;
multiply by <code>n / (n - 1)</code> for the Okabe-Yamada ratio form), with <code>L</code> the per-frame <code>networkLength</code>.
Optional envelope: <code>simulationCount</code> random patterns (Philox, an edge drawn proportional to its length
in two integer levels: edge lengths are quantized to 2^24 units of the longest live edge, a GPU
maximum, and summed inside chunks of 128 slots; the chunk totals, scaled against the largest chunk
total, get a second prefix sum; a draw picks a chunk and then an edge of it. Short edges keep their
share on large networks, and a wrong <code>networkLength</code> changes K but never the draw). Only the live CSR rows <code>0..offsets[nodeCount]</code> are snapped
to or drawn; <code>networkLength</code> is half the sum of their weights. Blocks and batches run at a fixed shape on
64-element-aligned sub-views and read their first row from a per-block info word, so every block
shares one set of pipelines.</p>
<ul>
<li>Props: <code>positions</code>, the network <code>topology</code> (with <code>nodePositions</code>), <code>maximumDistance</code>,
<code>networkLength</code>, <code>maximumSnapDistance</code>, <code>parameters</code> (<code>getGPUNetworkKFunctionParameterValues</code>: seed and
active simulations), <code>bandCount</code> (at most <code>GPU_NETWORK_K_FUNCTION_MAXIMUM_BAND_COUNT</code>), <code>simulationCount</code>,
<code>rowsPerBlock</code>, <code>candidateCapacity</code>. For many events pass <code>candidateCapacity</code> (with <code>maximumSnapDistance</code>) so
snapping uses the BVH join instead of scanning every edge per event; <code>output.status.overflow</code> is then required and
is set when the snap candidates overflow (K is then inexact). <code>spatialContext</code> is passed to
<code>GPUNetworkSnapping</code>: a great-circle context snaps longitude/latitude events in the context's length unit (give
<code>weights</code>, <code>maximumDistance</code>, <code>maximumSnapDistance</code> and <code>networkLength</code> in that unit too).
<code>kValues</code> and <code>envelope</code> are also in the context's length unit.</li>
<li>Outputs (<code>output</code>): <code>kValues</code>, optional <code>pairCounts</code>, <code>envelope</code> (min, mean and max per band), <code>snappedEventCount</code>,
<code>status.overflow</code> and <code>status.converged</code>.</li>
<li>Counts and K are pinned to spaghetti 1.7.6. Parallel edges between the same two nodes count as one road,
and the envelope is the plain min, mean and max (spaghetti scales the extremes by the threshold).
192 events with 20 patterns on 97,000 edges took about 61 ms (181 ms before the shared CSR expansion
and BVH snapping), of which about 41 ms is relaxation. Cross-K between two patterns is not provided.
Open: a block of 192 rows produced no result in one explorer capture (not investigated, so keep
<code>rowsPerBlock</code> near the 128 MB lane budget of <code>recommendLaneCount</code>), and snap-candidate overflow is
reported only through <code>status.overflow</code>.</li>
</ul>
<h4 id="how-it-computes-22">How it computes</h4>
<p>{/* lens:GPUNetworkKFunction */}</p>
<p>::lens{scenario=gpu-network.network-k-function rung=datum bind=page datum=7 caption="Inspect the 1,000-metre threshold: 278 unordered crash pairs give K of about 35,210.9 metres."}</p>
<p>::lens{scenario=gpu-network.network-k-function rung=formula}</p>
<p>::lens{scenario=gpu-network.network-k-function rung=kernel}</p>
<p>::lens{scenario=gpu-network.network-k-function rung=graph}</p>
<p>{/* parameters: GPUNetworkKFunction generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h4 id="gpunetworkkfunction-parameters"><code>GPUNetworkKFunction</code> parameters</h4>
<ul>
<li>Packer: <code>getGPUNetworkKFunctionParameterValues(settings)</code>, writes <code>GPU_NETWORK_K_FUNCTION_PARAMETER_LENGTH</code> (4) <code>uint32</code> values.</li>
</ul>
<p>Settings of <code>getGPUNetworkKFunctionParameterValues</code>:</p>
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
<td><code>seed</code></td>
<td><code>number</code></td>
<td>Random seed of the simulated patterns, an integer in <code>[0, 2^53)</code>.</td>
</tr>
<tr>
<td><code>activeSimulations</code></td>
<td><code>number</code></td>
<td>Number of simulated patterns evaluated, at most the compile-time <code>simulationCount</code>. The remaining patterns are skipped (their K values and counts are 0).</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h3 id="rendering-network-contributor-outputs">Rendering network contributor outputs</h3>
<p>Node-aligned and edge-aligned outputs are plain <code>GraphDataView</code>s, so any renderer that binds
storage buffers can draw them without readback. Typical pairings are <code>GPUNetworkAnalyticsColumns</code>
normalized degree or component labels for color and size, <code>GPUNetworkReachability</code> <code>output.bands</code>
for isochrone color, <code>GPUNetworkNeighborhood</code> <code>output.nodeMask</code> for highlighting, and
<code>GPUNetworkPathExtraction</code> compact <code>ids</code> for path ranks. Rendering layers are not part of this
library; see the deck.gl and luma.gl documentation for GPU-buffer-backed layers.</p>
<h3 id="gpuedgebundling"><code>GPUEdgeBundling</code></h3>
<p>Bend origin-destination lines toward shared corridors by iterating a density field, keeping both endpoints fixed.</p>
<p>::lens{scenario=gpu-network.edge-bundling rung=result caption="Compare 150 straight taxi flows with their bundled paths after 8 density iterations; endpoints stay fixed."}</p>
<p>Kernel-density edge bundling (KDEEB) as WebGPU compute, producing renderable polylines. Each edge
starts as a straight subdivision. Every iteration does four things:</p>
<ol>
<li>Splats Epanechnikov weights of all control points into a fixed-point <code>atomic&#x3C;u32></code> density
buffer.</li>
<li>Advects interior points one kernel radius along the bilinearly sampled normalized gradient.</li>
<li>Resamples each edge to uniform arc length and applies one Laplacian smoothing pass.</li>
<li>Anneals the radius by <code>lambda</code>.</li>
</ol>
<p>Endpoints are pinned. Density is a storage buffer, so there is no float-renderable texture
requirement and no texture size cap.</p>
<p>Lengths are relative to a square work box computed on the GPU every encoding from live-edge
endpoints, so results are scale invariant. Dead edges (masked, out-of-range or non-finite) do not
bundle, and their path collapses onto the source position. Outputs (<code>output</code>):</p>
<ul>
<li><code>paths</code>: <code>edgeCount * pointsPerEdge</code> <code>float32x2</code> rows, edge-major, in caller coordinates.</li>
<li><code>lineOffsets</code> (optional): for PathLayer binary attributes.</li>
<li><code>drawRecord</code> (optional): a four-word <code>drawIndirect</code> record <code>[pointsPerEdge, edgeCount, 0, 0]</code>
for instanced line strips.</li>
</ul>
<p><code>pointsPerEdge</code> (2 to 64), <code>maximumIterations</code> (1 to 64) and <code>densityResolution</code> are
compile-time. When <code>parameters</code> is given, one gate node writes the indirect dispatches of every
iteration, so iterations beyond <code>activeIterations</code> dispatch nothing (the graph has <code>5 + 3 * maximumIterations</code>
nodes, plus one gate with <code>parameters</code> and one for <code>indices</code>). Each iteration reads its index from a
256-byte aligned iteration-info slot, so all iterations share one splat and one update pipeline. With a longitude/latitude <code>spatialContext</code>
<code>nodePositions</code> are longitude and latitude in degrees and longitude is
scaled by the cosine of the mid-latitude of the live endpoints (computed on the GPU, clamped at 0.01); the
output stays in degrees and endpoints are exact. Bundling measures nothing in physical units, so the
context's metric is not used, and data spanning the antimeridian is not supported. The per-frame
<code>parameters</code> view holds <code>[activeIterations, kernelRadius, lambda, smoothing, stepScale]</code>; pack it with <code>getGPUEdgeBundlingParameterValues</code>. The edge list is
<code>sourceNodes</code> and <code>targetNodes</code> into <code>nodePositions</code>.</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> parameters</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUParameterBuffer</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(device, {id: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'bundling'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, format: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'uint32'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, length: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">5</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  values: </span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">getGPUEdgeBundlingParameterValues</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({kernelRadius: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">0.03</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}, </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'uint32'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">)});</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUEdgeBundling</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({nodePositions, sourceNodes, targetNodes, edgeMask,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  output: {paths, lineOffsets, drawRecord}, pointsPerEdge: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">16</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, maximumIterations: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">15</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  densityResolution: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">256</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, parameters: parameters.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">importToGraph</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(graph)}));</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">parameters.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">write</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">getGPUEdgeBundlingParameterValues</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({activeIterations: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">6</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, kernelRadius: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">0.05</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}, </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'uint32'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">));</span></span></code></pre>
<p>The normalized-gradient schedule is chaotic in f32. The f32 CPU oracle matches below 1e-5 of the
box after one iteration, and only statistically after that. Gradients below four fixed-point
quanta are treated as zero so a lone edge does not drift.</p>
<h4 id="how-it-computes-23">How it computes</h4>
<p>{/* lens:GPUEdgeBundling */}</p>
<p>::lens{scenario=gpu-network.edge-bundling rung=datum bind=page datum=10 caption="Inspect this flow: origin zone 5 connects to destination zone 2; its path starts at vertex 160."}</p>
<p>::lens{scenario=gpu-network.edge-bundling rung=iterations caption="Step through 5 key frames: the straight start, rounds 1, 2, 4, and the final output."}</p>
<p>::lens{scenario=gpu-network.edge-bundling rung=kernel}</p>
<p>::lens{scenario=gpu-network.edge-bundling rung=graph}</p>
<p>{/* parameters: GPUEdgeBundling generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h4 id="gpuedgebundling-parameters"><code>GPUEdgeBundling</code> parameters</h4>
<ul>
<li>Packer: <code>getGPUEdgeBundlingParameterValues(settings, format)</code>, writes <code>GPU_EDGE_BUNDLING_PARAMETER_LENGTH</code> (5) <code>uint32</code> values.
<ul>
<li><code>format</code> (<code>'uint32'</code>)</li>
</ul>
</li>
</ul>
<p>Settings of <code>getGPUEdgeBundlingParameterValues</code>:</p>
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
<td><code>activeIterations</code> (optional)</td>
<td><code>number</code></td>
<td>Iterations to run, clamped to the compiled <code>maximumIterations</code>. Default: the compiled maximum.</td>
</tr>
<tr>
<td><code>kernelRadius</code> (optional)</td>
<td><code>number</code></td>
<td>Initial kernel radius as a fraction of the work box side. Default 0.03.</td>
</tr>
<tr>
<td><code>lambda</code> (optional)</td>
<td><code>number</code></td>
<td>Radius decay per iteration, clamped to <code>[0.5, 0.9]</code>. Default 0.85.</td>
</tr>
<tr>
<td><code>smoothing</code> (optional)</td>
<td><code>number</code></td>
<td>Laplacian smoothing strength, clamped to <code>[0, 1]</code>. Default 0.5.</td>
</tr>
<tr>
<td><code>stepScale</code> (optional)</td>
<td><code>number</code></td>
<td>Advection step multiplier. Default 1.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h3 id="gpuflowaggregation"><code>GPUFlowAggregation</code></h3>
<p>Group origin-destination records by zone pair and rank the pairs by summed trips, with record counts per pair.</p>
<p>Use <code>getCommandNodes(graph)</code> to build the contributor. Binding and scratch preflight runs before
it declares resources. The former internal <code>validateBindings</code> class method is no longer exposed;
map-matched transition composition uses a private module helper with the same allocation-free
checks. Output layouts and command nodes are unchanged.</p>
<p>::lens{scenario=gpu-network.flow-aggregation rung=result caption="Compare 450 taxi records with the top 32 of 150 zone pairs; the highlight is the busiest pair."}</p>
<p>Maximum: <code>rows * maximumProbeCount</code> is at most 4294967295, so inputs above 2^24 rows need a <code>maximumProbeCount</code> below the default.</p>
<p>Origin–destination aggregation for flow maps. Each row's origin and destination are assigned to
square-grid cells, hexagons, or caller zone IDs. Rows are then grouped by zone pair through a
compile-time-capacity <code>GPUHashIndex</code>, counted and weighted with <code>GPUGroupAggregation</code>, and ranked by
two stable <code>GPUSort</code> passes. The result is a deterministic top-K list (weight descending, ties by
origin then destination zone) with zone columns, counts, and weights for an <code>ArcLayer</code>, plus
per-zone outgoing and incoming totals. An optional caller mask and per-frame time window
(<code>getGPUTimeWindowFilterParameterValues</code>) gate rows without recompiling. All results are in
<code>output</code> (a <code>GPUCompactOutput</code> plus the flow and zone columns). <code>output.status.pairOverflow</code> reports a full
pair table, also published as optional <code>status.candidateOverflow</code>. <code>status.overflow</code> reports only
top-K truncation (<code>requiredCount > ids.length</code>). <code>requiredCount</code> counts pairs retained in the table,
so it is a lower bound on the full requirement after pair-table loss.</p>
<p>Filter source rows with <code>mask</code>; a zero weight still contributes an accepted row and can create a
zero-weight pair. The commute-flow website applies its minimum-workers threshold through this
mask. Its hover count describes retained pairs as “displayed flows” when the 160-pair display
capacity truncates the top list; the outgoing jobs total still covers all accepted rows. Pair-table
loss is reported separately as incomplete aggregation, and no rejected input creates a displayed
arrow. The hover shows “pending aggregation” until its count, both loss flags and displayed columns
have arrived; an unread status is never interpreted as loss or a zero flow count.</p>
<p>For grid and hexagon zones, <code>zones.activeGridSize</code> is a per-frame packed <code>uint32</code> <code>[columns, rows]</code>
clamped to <code>1..gridSize</code>, so <code>gridSize</code> becomes the compile-time capacity: zone IDs are row-major in the
active size, zone outputs keep capacity length with zeros past <code>columns * rows</code>, and pair keys use the
capacity zone count (<code>GPUFlowAggregationActiveGridSize</code>). Pair it with the per-frame hexagon <code>radius</code>.
<code>zoneOutCountExtent</code> and <code>zoneInCountExtent</code> (two <code>float32</code> rows <code>[min, max]</code>) hold the extent of the
nonzero per-zone counts (both 0 when none), usable directly as a layer extent buffer.</p>
<p>The time gate accepts the same three timestamp forms as <code>GPUTimeWindowFilter</code>, including exact
<code>Int64</code> words with a <code>uint32</code> word window. <code>sumOrder</code> defaults to <code>'sorted'</code>: rows are sorted by pair and by zone and each group is summed in a
fixed tree, so <code>output.flowWeights</code>, <code>output.zoneOutWeights</code> and <code>output.zoneInWeights</code> are bitwise reproducible, at the
cost of three sorts and scans per encoding (38 instead of 22 nodes). <code>'atomic'</code> uses float
compare-exchange addition and serializes when rows share a pair (1M rows on one pair: 12-17 s against
22 ms sorted); it is faster only when nearly every row has its own pair. The sorted pair sum runs one
256-lane workgroup per stored pair, dispatched indirectly from the stored-pair count, so with weights and
<code>'sorted'</code> <code>pairCapacity</code> is at most 2^23.</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUFlowAggregation</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  zones: {kind: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'hexagon'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, bounds: viewport.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">importToGraph</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(graph), gridSize: [</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">64</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">48</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">], radius: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">250</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">},</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  origins, destinations, weights: tripCounts,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  timeWindow: {timestamps: departures, window: window.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">importToGraph</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(graph)},</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  pairCapacity: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">8192</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, excludeSelfFlows: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">true</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  output: {</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    ids: flowKeys, status: {count, overflow, requiredCount},</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    flowOriginZoneIds, flowDestinationZoneIds, flowWeights, zoneOutCounts, zoneInCounts,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    drawInstanceCount: graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">importGPUData</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'arcs'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, arcDraws.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">getInstanceCountData</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">))</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  }</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span></code></pre>
<h4 id="how-it-computes-24">How it computes</h4>
<p>{/* lens:GPUFlowAggregation */}</p>
<p>::lens{scenario=gpu-network.flow-aggregation rung=datum bind=page datum=0 caption="Inspect the busiest pair: 3 records sum to 54,331 trips."}</p>
<p>::lens{scenario=gpu-network.flow-aggregation rung=formula}</p>
<p>::lens{scenario=gpu-network.flow-aggregation rung=kernel}</p>
<p>::lens{scenario=gpu-network.flow-aggregation rung=graph}</p>
<p>{/* parameters: GPUFlowAggregation generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h4 id="gpuflowaggregation-parameters"><code>GPUFlowAggregation</code> parameters</h4>
<ul>
<li>Packer: <code>getGPUTimeWindowFilterParameterValues(settings)</code>, writes <code>GPU_TIME_WINDOW_FILTER_PARAMETER_LENGTH</code> (8) <code>float32</code> values.
<ul>
<li>Input port: <code>timeWindow.window</code>.</li>
</ul>
</li>
<li>Packer: <code>getGPUTimeWindowFilterWordParameterValues(settings)</code>, writes <code>GPU_TIME_WINDOW_FILTER_WORD_PARAMETER_LENGTH</code> (8) <code>uint32</code> values.
<ul>
<li>Input port: <code>timeWindow.window</code>.</li>
</ul>
</li>
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
<p>{/* /parameters */}</p>`};export{e as default};