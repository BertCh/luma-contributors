var e={slug:`api/gpu-network/network-time-geography-mob09`,title:`Network time geography`,html:`<h1 id="network-time-geography">Network time geography</h1>
<p>Two contributors on top of the routing already in the package.</p>
<h2 id="gpunetworkpotentialpatharea"><code>GPUNetworkPotentialPathArea</code></h2>
<p>The network potential path area of a space-time prism: the nodes and directed edges a traveller
who leaves <code>a</code> and must be at <code>b</code> within a budget could have used. Inputs are the two trees:
<code>forwardCosts</code> (cost from <code>a</code>, a <code>GPUNetworkReachability</code> run) and <code>reverseCosts</code> (cost to <code>b</code>, the
same contributor on the reversed network), plus the per-frame <code>budget</code> row.</p>
<ul>
<li>An edge <code>(u, v)</code> is in the prism when <code>forward[u] + weight + reverse[v] &#x3C;= budget</code>; <code>edgeSlack</code> is
the difference. A node is in it when <code>forward[v] + reverse[v] &#x3C;= budget</code>.</li>
<li>The edge is whole: a directed edge is entered at <code>u</code> and left at <code>v</code>, so every point on it has
the same slack. Two directed edges make an undirected street. Clipping partial edges applies to
anchors in the middle of an edge, which is not supported.</li>
<li><code>output.summary</code> is <code>[nodes in the prism, edges in the prism]</code>.</li>
</ul>
<p>Verified against NetworkX 3.7 <code>single_source_dijkstra_path_length</code> on the network and its reverse
(three random directed networks, fixture <code>network-time-geography-mob09.json</code>), with the trees
computed both by NetworkX and by two <code>GPUNetworkReachability</code> runs.</p>
<p>{/* lens:GPUNetworkPotentialPathArea */}</p>
<h2 id="gpunetworktraveltimepercentiles"><code>GPUNetworkTravelTimePercentiles</code></h2>
<p>Travel-time percentiles over a departure window, the analogue of r5r <code>time_window</code>. The window is
<code>window = [first departure, step]</code> with <code>departureCount &#x3C;= 60</code>. Each departure runs one
<code>GPUNetworkTimeRouting</code> (FIFO earliest arrival on the restriction automaton) in the same graph. A
vertex's travel time is its earliest arrival over its states minus the departure; per vertex the
values are sorted and the <code>p</code>-th percentile is the value of rank <code>ceil(p / 100 * n)</code> (nearest rank).
A departure that cannot reach the vertex counts as <code>+Infinity</code>. <code>output.reachedCounts</code> and
<code>output.status</code> (<code>[valid, converged]</code> per departure) say how complete the numbers are.</p>
<p>CPU and GPU ranks use the same host calculation, evaluated exactly for the supplied binary64
percentile without an epsilon. For travel times <code>[5, 90]</code>, percentiles immediately below 50%, at
50%, and immediately above 50% select <code>[5, 5, 90]</code>. Integer boundaries stay exact: 56% of 25
departures selects rank 14, and the next representable percentile above 56% selects rank 15.</p>
<p>Percentiles and output shapes are captured when a contributor or operation is constructed;
later changes to the caller's percentile array do not change that computation. Every percentile
slot must contain a finite number in <code>(0, 100]</code>; sparse lists are rejected with a named error.
Declared properties are read by name, so inherited getters (including a zero routing budget)
have the same meaning as own properties. Caller-owned GPU views retain their identities.
Both backends honor <code>maximumRelaxations</code> for each departure (default: the number of states).
At zero budget, only the origin is reached, destination travel times stay <code>+Infinity</code>, and each
departure reports <code>[valid, converged] = [1, 0]</code>. A budget exit publishes incumbent travel times;
inspect the status before treating unreached destinations as conclusive.</p>
<p>The percentile rule is nearest rank. r5r's exact index rule was not checked against its source, so
for small windows its percentiles may differ by one rank.</p>
<p>Reference tier: each routing is a single invocation (at most 256 states and 4096 transitions), so
a window costs <code>departureCount</code> such dispatches, and arbitrary departure lists are not supported.</p>
<p>Verified against a binary64 FIFO Dijkstra with affine arrival <code>t + base + slope * t</code> (NetworkX has
no time-dependent routing), and, for slope 0, against NetworkX distances.</p>
<p>{/* lens:GPUNetworkTravelTimePercentiles */}</p>`};export{e as default};