var e={slug:`api/gpu-network/concepts`,title:`GPU Network concepts`,html:`<h1 id="gpu-network-concepts">GPU Network concepts</h1>
<h2 id="spatial-contexts-and-length-units">Spatial contexts and length units</h2>
<p>Distance operators (<code>GPUNetworkSnapping</code>, <code>GPUMapMatching</code>, <code>GPUNetworkKFunction</code>,
<code>GPUNetworkLineGraph</code>, <code>GPUNetworkNoding</code>, <code>GPUNetworkIsochrones</code> and
<code>addDriveTimeCatchmentRecipe</code>) accept planar azimuthal-equidistant (AEQD) frames from
<code>createGPULocalFrame(...).spatialContext</code>, including <code>units: 'meters'</code>, and reject Lambert
equal-area frames. Project positions into the declared frame before passing them to an operator.
Where supported, longitude/latitude degrees with <code>metric: 'great-circle'</code> are an alternative.
<code>GPUEdgeBundling</code> declares no measure: lengths are fractions of its work box, so either local
frame is accepted.</p>
<table>
<thead>
<tr>
<th>Contributor</th>
<th>Longitude/latitude and kilometers</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>GPUNetworkSnapping</code></td>
<td>Great-circle distances in the context's length unit; <code>units: 'kilometers'</code> selects kilometers.</td>
</tr>
<tr>
<td><code>GPUMapMatching</code></td>
<td>Great-circle distances and distance parameters in the context's length unit, including kilometers.</td>
</tr>
<tr>
<td><code>GPUNetworkKFunction</code></td>
<td>Great-circle snapping; weights, distance thresholds, network length, K values and envelope in the context's length unit, including kilometers.</td>
</tr>
<tr>
<td><code>GPUNetworkLineGraph</code></td>
<td>Great-circle turn bearings accept meters or kilometers; angles and caller-owned travel costs are unchanged.</td>
</tr>
<tr>
<td><code>GPUNetworkNoding</code></td>
<td>Planar only; snap tolerance and output lengths are in the context's length unit. Longitude/latitude is rejected.</td>
</tr>
<tr>
<td><code>GPUNetworkIsochrones</code></td>
<td>Raster is planar; buffer radius and extent are in the context's length unit. Cell outlines accept longitude/latitude, including kilometer contexts, but only index positions in degrees.</td>
</tr>
<tr>
<td><code>addDriveTimeCatchmentRecipe</code></td>
<td>Longitude/latitude with metric <code>'none'</code> or <code>'great-circle'</code> uses great-circle snapping in the context's length unit, including kilometers. Travel costs keep the units of <code>topology.weights</code>.</td>
</tr>
<tr>
<td><code>GPUEdgeBundling</code></td>
<td>Longitude/latitude selects a degree-aware work box; kilometer contexts are accepted without physical distance measurements.</td>
</tr>
</tbody>
</table>
<p>Geographic distance units default to meters. <code>sphereRadius</code>, when supplied, is in the context's
length unit. Kilometer inputs must use kilometer-valued radii, thresholds and distance weights;
positions remain longitude/latitude degrees. Map matching's default <code>routeSlack</code> is 50 in the
context's length unit; supply <code>0.05</code> for a 50-meter allowance in a kilometer context.</p>
<p>Great-circle snapping searches all edges regardless of <code>candidateCapacity</code>, in meters or
kilometers alike.</p>
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
(<code>ids</code> and <code>status</code>: <code>count</code>, <code>overflow</code>, optional <code>requiredCount</code>). <code>status.count</code> is clamped to
<code>ids.length</code> and can be an indirect draw instance count; <code>status.overflow</code> is rewritten every encoding.</li>
<li><strong>Stable IDs.</strong> Result IDs are the caller's <code>sourceIds[row]</code> (or tile IDs) when given and zero-based
rows otherwise. Node and transient IDs are <code>\${id}-&#x3C;step></code>, so two instances in one graph need
different <code>id</code> props.</li>
<li><strong>One import per buffer.</strong> Import a buffer once (for example with <code>graph.importGPUData()</code> or
<code>GPUParameterBuffer.importToGraph()</code>) and pass the returned view to every contributor that reads
it, so the graph tracks hazards on one logical handle.</li>
<li><strong>CSR convention.</strong> Undirected graphs list both directions in the CSR. A reverse CSR, built with
<code>GPUCOOToCSR</code>, gives "cost to reach" semantics.</li>
<li><strong>One network input.</strong> Every contributor takes the network as <code>topology: GPUNetworkTopology</code>
(<code>offsets</code>, <code>neighbors</code>, <code>weights</code>, optional <code>nodePositions</code>); connectivity-only contributors
accept a <code>GPUNetworkAdjacency</code>, the same object with optional <code>weights</code>. Its CSR part has the names
and formats of <code>GPUSpatialWeights</code>, so a topology can be passed wherever spatial weights are read.
Contributors that write a network (line graph, noding, induced subgraph) emit the same shape, so
their output feeds the next contributor's <code>topology</code> directly.</li>
<li><strong>Cost-table completeness.</strong> When composing <code>GPUNetworkCostTable</code> with <code>GPUMapMatching</code>, forward
<code>output.status.complete</code> as <code>routeTable.status.complete</code>. This one word is nonzero only after all
searches converge without capacity overflow. The matcher uses A* when it is zero or omitted;
an absent entry in an unfinished table never establishes that a route is impossible. Size lanes
and rounds together: both the command count and the frontier storage binding must fit (see the
<a href="/luma-contributors/docs/api/gpu-network/operators#gpunetworkcosttable">cost-table limits</a>).</li>
<li><strong>Outputs in one bag.</strong> Every GPU-written view is a field of <code>output</code>, and every status word
(<code>converged</code>, <code>iterationCount</code>, <code>overflow</code>, <code>count</code>, <code>requiredCount</code>, and prefixed forms such as
<code>pairOverflow</code>) lives in the bag's <code>status</code> member (<code>output.status.overflow</code>). Graph start
nodes are <code>startNodes</code>/<code>startCosts</code>/<code>startCount</code>, cost bands are <code>costBreaks</code>, iteration caps are
<code>maximum*Iterations</code>, and an active count is <code>number | GraphDataView&#x3C;'uint32'></code> (a number is
compiled in, a view is read every frame).</li>
<li><strong>Shared contracts.</strong> Four shapes mean the same on every analysis page:
<ul>
<li><strong>Status.</strong> Every GPU-written status word sits in the <code>status</code> member of the bag that owns it:
<code>output.status.overflow</code>, <code>output.status.converged</code>, and <code>output.status.code</code> for model fits.</li>
<li><strong>Grid.</strong> Raster and terrain contributors take <code>grid: GPURasterMetadata</code>, and their packers read
the cell size from it (see <a href="/luma-contributors/docs/api/gpu-terrain/concepts#conventions">GPU Terrain conventions</a>).</li>
<li><strong>Network.</strong> Network contributors take <code>topology: GPUNetworkTopology</code> (<code>offsets</code>, <code>neighbors</code>,
<code>weights</code>, optional <code>nodePositions</code>). Its CSR part can be passed as <code>GPUSpatialWeights</code> (see
<a href="#conventions">GPU Network conventions</a>).</li>
<li><strong>Polygons.</strong> Polygons use the four-level <code>GPUPolygonGeometryPort</code>: <code>kind: 'polygons'</code>,
<code>positions</code>, <code>featureOffsets</code>, <code>polygonOffsets</code> and <code>ringOffsets</code>. Contributors with a single
geometry take the same fields flat, without <code>kind</code> (<code>GPUPolygonGeometryFields</code>). See
<a href="/luma-contributors/docs/api/gpu-spatial-analysis/concepts#shared-ports-status-and-planning">Shared ports</a>.</li>
</ul>
</li>
</ul>
<h2 id="gpu-execution-model">GPU execution model</h2>
<p>How network contributors run on the device: the command graph they declare, the kernels it
compiles to, and the resources each pass reads and writes, drawn from a recorded real run.</p>
<p>{/* lens:execution-model */}</p>
<p>Inspect the recorded single-hop search, then contrast its rounds with workgroup-local chaining.</p>
<p>::lens{scenario=gpu-network.network-reachability rung=graph caption="Inspect the declared relax rounds, dependencies and retained observation buffers."}</p>
<p>::lens{scenario=gpu-network.network-reachability rung=kernel caption="Inspect atomic cost minima and the indirect frontier dispatch."}</p>
<p>::lens{scenario=gpu-network.network-reachability.chained rung=iterations caption="Chained rounds span multiple hops; queue stamps record global rounds, not the hops within them."}</p>`};export{e as default};