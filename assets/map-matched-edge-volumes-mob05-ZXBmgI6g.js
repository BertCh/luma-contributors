var e={slug:`api/gpu-network/map-matched-edge-volumes-mob05`,title:`Edge volumes of map-matched tracks`,html:`<h1 id="edge-volumes-of-map-matched-tracks">Edge volumes of map-matched tracks</h1>
<p><code>GPUMapMatchedEdgeVolumes</code> turns the <code>output.traversedEdges</code> of <code>GPUMapMatching</code> into traversal
counts per network edge and, optionally, the busiest edge-to-edge movements.</p>
<p><code>GPUMapMatching</code> lists the edges each vehicle passed, including the intermediate edges between two
fixes, in travel order. Between sub-tracks (after a break or an unmatched fix) it writes one
<code>GPU_MAP_MATCHING_NONE</code>. It also inserts this separator when a hop's intermediate edges could not
be recovered, preventing a false turn between the two retained endpoint edges. This contributor
reads that list and its status:</p>
<ul>
<li><code>output.edgeVolumes</code> (<code>edgeCount</code> rows): how many times each edge was traversed. Each listed edge
entry counts once, with an integer atomic add, so the result does not depend on thread order. A
loop that passes an edge twice counts twice. A fix on the same edge as the previous fix adds
nothing, because the route lists the edge once. This is the per-edge count of the matched path
(Valhalla <code>trace_attributes</code> edges, a pgRouting route table), not a count of GPS fixes.</li>
<li><code>output.transitions</code> (optional): a <code>GPUFlowAggregationOutput</code> whose zones are the edges. An edge
followed by another edge of the same sub-track is one movement. The top-K list is sorted by
count, <code>zoneOutCounts[e]</code> counts traversals of <code>e</code> that continued on another edge and
<code>zoneInCounts[e]</code> those that entered it from another edge. No movement is invented across a
separator. This inherits the flow aggregation limit of 65535 zones, so it needs <code>edgeCount &#x3C;= 65535</code>.
<code>edgeVolumes</code> alone has no such limit.</li>
<li><code>output.status.candidateOverflow</code> (required): 1 when a track needed more than
<code>traversedEdgeCapacity</code> entries or the forwarded traversal <code>overflow</code>/<code>candidateOverflow</code> word
is nonzero. The flag resets each encoding. With grid or scoring candidate loss, counts are
conditional on the retained match: another road can win, increasing some edge counts and
decreasing others. Lower bounds apply only to omissions from an otherwise fixed traversal,
such as list capacity or failed recovery after an exact match. This replaces <code>output.truncated</code>.
A requested <code>transitions.status.candidateOverflow</code> combines traversal loss with pair-table loss,
so it can be checked when the transition output is consumed separately. Allocate this nested
flag and the volume flag on separate physical buffers; their publication pass reads one while
writing the other.</li>
</ul>
<table>
<thead>
<tr>
<th>Prop</th>
<th>Meaning</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>traversedEdges</code></td>
<td>Forward <code>{edges, counts, status}</code> from matching; <code>status.candidateOverflow</code> is required and <code>status.overflow</code> is also accepted. Shapes are <code>trackCount * traversedEdgeCapacity</code> and <code>trackCount</code>. Fixed external sequences instead explicitly assert <code>completeness: 'known-complete'</code>; never use that assertion for unchecked matching output.</td>
</tr>
<tr>
<td><code>trackCount</code>, <code>traversedEdgeCapacity</code>, <code>edgeCount</code></td>
<td>Compile-time shapes</td>
</tr>
<tr>
<td><code>transitionPairCapacity</code>, <code>maximumProbeCount</code></td>
<td>Compile-time, with <code>output.transitions</code></td>
</tr>
</tbody>
</table>
<p>Named props and nested port/status maps are captured once at construction, including inherited
getters. Later caller mutation cannot change the validated graph; view contents remain live.</p>
<p>Verified end to end: tracks are generated along known paths of a 4 x 4 grid, matched on the GPU,
and the edge volumes and movements are compared with plain counting over the ground-truth edge
lists (including intermediate edges, opposite directions, a repeated edge and a break). Valhalla
and pgRouting were not run; the semantics above follow their documented output.</p>
<p>{/* lens:GPUMapMatchedEdgeVolumes */}</p>`};export{e as default};