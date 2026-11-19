var e={slug:`api/gpu-spatial-analysis/concepts`,title:`GPU Spatial Analysis concepts`,html:`<h1 id="gpu-spatial-analysis-concepts">GPU Spatial Analysis concepts</h1>
<h2 id="ontology-and-capability-catalogue">Ontology and capability catalogue</h2>
<p>The entry point exports a small ontology over the flat contributor API. It gives applications,
documentation tools and future planners stable names for the data flowing between contributors:
points, lines, polygons, trajectories, DGGS cells, spatial weights, feature pairs, regions, sampled
surfaces, summaries and render geometry. Capabilities then group public exports by domain and
pipeline stage and state their logical inputs, outputs, coordinate assumptions and output-bounding
contract.</p>
<p>The catalogue stores export names, not constructor references, so using it for discovery does not
eagerly retain every implementation in a bundle. Every public <code>GPU...</code> contributor and <code>add...</code>
recipe belongs to exactly one available capability. Deliberate gaps are <code>status: 'opportunity'</code> and
have no operators, which makes coverage and missing connective tissue queryable with the same API.</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">import</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8"> {</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  getGPUSpatialAnalysisCapabilityForOperator,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  queryGPUSpatialAnalysisCapabilities</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">from</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF"> '@luma-contributors/analysis/gpu-spatial-analysis'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">;</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> producers</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> queryGPUSpatialAnalysisCapabilities</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  status: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'available'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  output: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'spatial-weights'</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">});</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> gaps</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> queryGPUSpatialAnalysisCapabilities</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({status: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'opportunity'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">});</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> localMoranFamily</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> getGPUSpatialAnalysisCapabilityForOperator</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'GPULocalMoran'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">);</span></span></code></pre>
<p>The exported vocabulary is:</p>
<ul>
<li><code>GPU_SPATIAL_ANALYSIS_DATA_ONTOLOGY</code> and <code>GPU_SPATIAL_ANALYSIS_DATA_KINDS</code>: logical data nouns.</li>
<li><code>GPU_SPATIAL_ANALYSIS_DOMAINS</code> and <code>GPU_SPATIAL_ANALYSIS_STAGES</code>: subject and pipeline roles.</li>
<li><code>GPU_SPATIAL_ANALYSIS_CAPABILITIES</code>: available capability families and named opportunities.</li>
<li><code>queryGPUSpatialAnalysisCapabilities(query)</code>: filters by status, domain, stage, input, output,
coordinate model or search text.</li>
<li><code>getGPUSpatialAnalysisCapability(id)</code> and
<code>getGPUSpatialAnalysisCapabilityForOperator(name)</code>: direct lookup from a concept or public export.</li>
<li><code>getGPUSpatialAnalysisCapabilityConnections(id)</code>: resolves the curated adjacent capabilities
used to assemble workflows.</li>
</ul>
<p>Operator-level discovery is available through <code>GPU_SPATIAL_ANALYSIS_OPERATORS</code>,
<code>getGPUSpatialAnalysisOperator(name)</code> and <code>queryGPUSpatialAnalysisOperators(query)</code>. Descriptors
contain export names rather than constructors and distinguish primary capability, secondary tags,
logical ports, spatial support, topology, cardinality, numerical semantics, determinism, status and
explicit readback. Unknown details are reported as <code>unspecified</code>, never inferred as a stronger
guarantee. <code>GPU_SPATIAL_ANALYSIS_AUDIT</code> and <code>formatGPUSpatialAnalysisAuditMarkdown()</code> expose the
same inventory used by package verification.</p>
<h2 id="contributor-operation-contracts">Contributor operation contracts</h2>
<p><code>GPUContributorOperation</code> from <code>@luma-contributors/analysis/analysis-core</code> adapts a contributor
into a published luma.gl <code>GPUGraph</code> operation. Compile with <code>GPU_CONTRIBUTOR_OPERATION_LOWERING</code>.
Input and output graph value names must be non-empty; output formats must be <code>float32</code>, <code>sint32</code>
or <code>uint32</code> vectors with one to four components. Output lengths are validated at construction
and must be positive unless the port is listed in <code>emptyOutputPorts</code>. Errors name the operation
and port before luma.gl plans the contract. An operation cannot write the same graph value twice
or read a value it also writes. Packed single chunks are checked when views are resolved.</p>
<h2 id="shared-ports-status-and-planning">Shared ports, status and planning</h2>
<p>The entry point exports non-owning structural ports for point, line, polygon and trajectory
geometry, compact pairs, generated geometry, spatial weights, cell tables, sampled surfaces and
classified values. Ports only group existing graph views; they do not allocate, pack, submit or
read back buffers. Polygons always use the four-level layout of <code>GPUPolygonGeometryPort</code>
(<code>positions</code>, <code>featureOffsets</code>, <code>polygonOffsets</code>, <code>ringOffsets</code>; rows are features, the first ring of
each polygon is its shell), lines use <code>positions</code> and <code>lineOffsets</code>, and tracks <code>trackOffsets</code>.
<code>GPUPointGeometryPort</code> and <code>GPULineGeometryPort</code> exclude <code>featureOffsets</code>; named multipart ports
are distinct types and cannot be passed to single-part consumers. Those consumers also reject
grouping at construction, including object-spread inputs, before unpacking positions or offsets.
This includes geometry validity, line topology/clipping/refinement, track similarity and the
points-in-polygons, cluster-outline and straight-line catchment recipes. Both choropleth backends
reject a grouped point feature instead of counting its parts separately.</p>
<p><code>GPUMultipartPointGeometryPort</code> adds feature-to-point-part <code>featureOffsets</code> to the point port;
<code>GPUMultipartLineGeometryPort</code> adds feature-to-linestring-part <code>featureOffsets</code> above <code>lineOffsets</code>.
Both use one <code>sourceIds</code> row per feature. Offsets start at zero, are nondecreasing and terminate
at the child count. Equal adjacent offsets describe an empty feature or part, rather than a missing
owner. <code>validateGPUMultipartGeometryPort</code> checks static formats, terminal entries and identity
lengths; it does not read offset contents from the GPU. Measures and affine transform share this
validation and retain their existing flat props. For multipart input, both require at least one
feature and one position.</p>
<p><code>createGPUMultipartRowOwnersNode(graph, id, geometry, featureRows)</code> writes one feature row per
point part or linestring part using the current offsets on each submission. The caller owns the
packed <code>uint32</code> output, whose length must equal the part count. Empty features own no rows; empty
line parts still have an owner. Uncovered parts receive <code>GPU_MULTIPART_NO_FEATURE_ROW</code>
(<code>0xffffffff</code>), never an application ID. The adapter requires disjoint buffers for its packed
<code>featureOffsets</code> and output views, and each aligned binding must fit the device's
<code>maxStorageBufferBindingSize</code>. It does not validate offset contents or make a join feature-aware;
each consumer must explicitly support multipart input.
Predicate joins support feature-level <code>intersects</code>, <code>within</code> and <code>coveredBy</code>; nearest joins support
multipart queries and targets in neighbors mode, and prepared indices use feature bounds.
See <a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-joins">multipart join rules and operation forms</a> for supported routes,
candidate budgets and CPU twins.</p>
<p>Contributors with one geometry take these fields flat (<code>GPUPolygonGeometryFields</code>,
<code>GPULineGeometryFields</code>, <code>GPUTrajectoryGeometryFields</code>); contributors with two geometries, such as
the joins, take whole tagged ports (<code>GPUPackedGeometryPort&#x3C;...></code>, one packed <code>positions</code> view), so a
generated geometry output feeds them directly. <code>GPUSpatialPredicateJoin</code> <code>output.pairs</code> is a
<code>GPUCompactPairPort</code> and feeds <code>GPUPairGather</code> directly. Every GPU-written status scalar lives in the
<code>status</code> of the bag that owns it: capacity-bounded outputs publish <code>status.count</code>, <code>status.requiredCount</code>
when exact, and the applicable overflow, convergence, invalid or uncertainty fields of the shared status
ports (<code>GPUCardinalityPort</code>, <code>GPUOverflowStatusPort</code>, <code>GPUIterationStatusPort</code>, ...). A nested list owns
its own <code>status</code> (<code>output.pieces.status.requiredVertexCount</code>), stage-prefixed words keep their prefix
(<code>output.status.pairOverflow</code>), and model fits report their solver <code>status.code</code>.</p>
<p><code>GPUSpatialContext</code> separates coordinate space, metric, units and sphere/ellipsoid parameters; omitted context means native planar coordinates.
Planar native contexts may declare <code>units: 'meters'</code> (local frames do); <code>'kilometers'</code> is only for geographic metrics.
Planar <code>'native'</code> units are undeclared and compatible with planar <code>'meters'</code>; meters and kilometers are incompatible.
The <code>frame</code> identity has kind <code>'local-azimuthal-equidistant'</code>, <code>'local-lambert-equal-area'</code> or <code>'crs'</code>;
<code>assertCompatibleGPUSpatialContexts</code> rejects contexts with different frame ids.
<code>datum</code> declares the horizontal datum of longitude/latitude positions or of a planar frame's source (for example <code>'EPSG:4326'</code>, <code>'EPSG:4269'</code>, <code>'GCJ-02'</code>); <code>createGPULocalFrame({datum})</code> copies it to the frame's context.
Two declared datums must match by <code>getGPUDatumKey</code> (trimmed, upper-cased; only WGS84 spellings alias to <code>'EPSG:4326'</code>), and an undeclared datum matches any; the library never converts datums.
The check runs where two contexts meet; <code>GPUProjection</code> CRS ids already imply their datum.
Two-input operators (joins, zonal and region statistics, raster join, pairwise point-in-polygon, line clip, overlay, zone events and the choropleth, change-of-support and hot-spot recipes) take an optional context for their second input, named after that input (<code>polygonsSpatialContext</code>, <code>rightSpatialContext</code>, <code>featuresSpatialContext</code>, <code>regionSpatialContext</code>, <code>zonesSpatialContext</code>, ...).
When given, <code>resolveGPUSecondarySpatialContext</code> validates it, applies the operator's edge-model requirement to it and checks it against <code>spatialContext</code>, so a datum, frame, units or metric mismatch between the two sides throws <code>&#x3C;id> &#x3C;side>SpatialContext incompatible spatial contexts: &#x3C;reason></code>; omitted, <code>spatialContext</code> describes both inputs and nothing is checked between them.
<code>edgeModel</code> (longitude/latitude only) says how polygon and line edges run between vertices: <code>'linear'</code> (default, straight in degrees as in RFC 7946, GEOS and turf) or <code>'great-circle'</code> (minor arcs, as in PostGIS <code>geography</code>, BigQuery <code>GEOGRAPHY</code> and S2); ellipsoidal contexts use <code>ellipsoidalEdgeModel</code> instead.
Polygon consumers declare <code>edgeModels: GPU_LINEAR_EDGE_MODELS</code> and reject great-circle input with a pointer to <code>GPULonLatPolygonPreparation</code>, which splits rings at the antimeridian, caps pole rings and densifies great-circle edges; their optional <code>status: GPULonLatPolygonStatusPort</code> counts unprepared rings (<code>antimeridianUncertainCount</code>, <code>poleUncertainCount</code>).
Point-pattern statistics and kernel density take a <code>studyWindow</code>: <code>'rectangle'</code> (default, per-frame bounds) or a <code>GPUPolygonStudyWindow</code> built by <code>GPUStudyWindowPreparation</code> in the same planar frame, checked by <code>resolveGPUStudyWindow</code>.
Support declarations use <code>measure: 'distance' | 'area'</code>: distance operators reject Lambert equal-area frames,
while area/density operators reject azimuthal-equidistant frames, with an error naming the frame to project into.
This measure check accepts <code>'crs'</code> frames and frameless contexts.
Spherical distances use <code>getGPUSphereRadius</code> (mean radius 6371008.8 m); spherical areas use <code>getGPUAuthalicSphereRadius</code> (WGS84 authalic radius 6371007.1809 m).
Both honour <code>sphereRadius</code> in the context's units; <code>units: 'kilometers'</code> divides default radii by 1000, giving areas in km².
Ellipsoidal metrics use <code>getGPUEllipsoid</code> (<code>context.ellipsoid</code> or <code>GPU_WGS84_ELLIPSOID</code>).
Terrain heights declare a vertical datum the same way (<code>GPUTerrainHeightReference</code>, <code>GPU_ELLIPSOIDAL_VERTICAL_DATUM</code>, <code>assertCompatibleGPUVerticalDatums</code>); see <a href="/luma-contributors/docs/api/gpu-terrain">GPU Terrain</a>.
<code>getGPULengthUnits</code> gives the length unit reported by <code>GPUMeasureDescription</code>.
Compatibility helpers reject invalid workflow connections during graph construction and contain no
CRS parsing or reprojection; that remains the responsibility of GPU Project.</p>
<p><code>GPUSpatialParameterSchema</code> describes packed layouts without changing their WGSL representation.
Schemas are published for geometry predicates, neighbor search, line segmentizing, great-circle
arcs, spatial autocorrelation and line density. The generic packer validates defaults, offsets,
ranges and units; contributor-specific packers retain any documented normalization that is not yet
expressed by the generic schema.</p>
<p>Capacity plans (<code>GPUCapacityPlan</code> per stage, <code>GPUContributorCapacityPlan</code> per contributor) report
stage capacities, work items, caller-owned output bytes and transient bytes. They are computed by
the contributor that allocates, from the same constants as its allocation, and stay pure:
applications own allocations. The weights grow loop is <code>getGPUSpatialWeightsCapacityPlan</code> plus the
producers' <code>getCapacityPlan</code> (see <a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-weights">Spatial weights</a>). Candidate and output stages
grow independently: a candidate overflow (for example <code>output.status.candidateOverflow</code> of
<code>GPUContiguityWeights</code>) means the final <code>requiredCount</code> may describe only the candidates that fit,
so growing only the output is invalid.</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> plan</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8"> contiguity.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">getCapacityPlan</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({requiredCount, candidateOverflow: candidateOverflow </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">===</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> 1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">});</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// Reallocate plan.slots.capacity slots and rebuild with pairCapacity: plan.pairs.capacity.</span></span></code></pre>
<p><code>GPUPartitionDescriptor</code> retains contiguous source-batch cores, read halos and deterministic seam
ownership without repacking <code>GPUVector</code> chunks. <code>GPUTilePartitionDescriptor</code> carries spatial core
and halo bounds for tiled consumers. <code>GPUSpatialJoinPrepared</code> validates its <code>partitioning</code> against the
indexed features. The Arrow
analytics adapter publishes its record batches through this contract, including empty batches.
<code>iterateGPUAnalyticsPartitionsFromArrowTable()</code> lazily allocates one source batch at a time while
retaining its global batch ID and row offset; destroying each yielded result before advancing bounds
GPU residency to one partition. The tile-LOD explorer declares explicit core, halo and cross-tile
ownership rules with the tile contract. Execution remains consumer-owned.</p>
<p>Recipe results use a required common envelope with <code>outputs</code>, <code>intermediates</code> and stage-owned
<code>status</code> in addition to <code>contributors</code>. Result-level status words sit in <code>result.status</code> next to
<code>result.status.stages</code> (for example <code>result.status.weightsOverflow</code>), and sub-results own a <code>status</code>
(<code>result.permutation.status.overflow</code>); the <code>outputs</code> overrides mirror those paths.</p>
<h2 id="conventions">Conventions</h2>
<ul>
<li><strong>Graph views in, graph views out.</strong> Inputs and outputs are <code>GraphDataView</code>, <code>GraphVectorView</code>, or
<code>GraphTextureView</code> objects created on the target graph. Outputs are always caller-owned; scratch
storage is graph transients that die with <code>compiled.destroy()</code>. Contributors never compile, encode,
submit, or read back.</li>
<li><strong>Per-frame values never recompile.</strong> Viewports, thresholds, radii, observer positions, budgets,
and region shapes are read from storage views, usually a <code>GPUParameterBuffer</code> that the
application rewrites with <code>write()</code>. Lengths, capacities, grid sizes, and which optional views
exist are compile-time topology; each prop's TSDoc says which category it belongs to.
<code>GPUParameterValues&#x3C;Format></code> maps <code>float32</code>, <code>uint32</code>, and <code>sint32</code> to <code>Float32Array</code>,
<code>Uint32Array</code>, and <code>Int32Array</code>. Initial values and updates must match the declared format;
matching arrays from other JavaScript realms are accepted. Mismatched arrays and supplied
non-array initializers (including <code>null</code>) are rejected without reinterpreting their bytes.
Only <code>undefined</code> omits initialization. <code>length</code> is an integer in
<code>[0, 2^32 - 1]</code>. Initial values must fit that logical length, including when it is zero.
Updates start at an integer <code>firstElement</code> in <code>[0, length]</code> and must fit the remaining elements.</li>
<li><strong>Bounded results report overflow on the GPU.</strong> Compact ID lists use <code>GPUCompactOutput</code>
(<code>ids</code> and <code>status: {count, overflow, candidateOverflow?, requiredCount?}</code>). <code>status.count</code> is clamped
to <code>ids.length</code> and can be an indirect draw instance count. <code>overflow</code> means final output capacity
loss; <code>candidateOverflow</code> means earlier loss that growing the output cannot recover. Both supplied
flags are rewritten every encoding. <code>requiredCount</code> counts accepted rows reaching the final stage,
before output truncation; it is a lower bound after candidate loss or saturation at <code>0xffffffff</code>.
A consumer must propagate supplied incompleteness, suppress dependent results, or refuse/fall back.
The shared <code>createPublishNode</code> refuses nonempty <code>candidateOverflowSources</code> without a candidate
destination. Producers publish their own optional domain diagnostics separately; omitting such
diagnostics is not evidence of completeness.</li>
<li><strong>Stable IDs.</strong> Result IDs are the caller's <code>sourceIds[row]</code> (or tile IDs) when given and zero-based
rows otherwise. Node and transient IDs are <code>\${id}-&#x3C;step></code>, so two instances in one graph need
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
<a href="#shared-ports-status-and-planning">Shared ports</a>.</li>
</ul>
</li>
</ul>
<p><code>defineGPUSpatialParameterSchema</code> checks defaults and <code>packGPUSpatialParameterValues</code> checks
overrides in both their input and encoded domains. Float32 fields are checked after <code>Math.fround</code>:
a finite-only field rejects <code>1e39</code>, which would encode as infinity. Bounds apply to the stored
value too: a default or override of <code>0.1</code> with maximum <code>0.1</code> is rejected because float32 stores
<code>0.10000000149011612</code>. Use representable bounds when that is the intended endpoint. Integer
fields require whole values within their signed or unsigned 32-bit range; they never wrap.
<code>allowNonFinite</code> explicitly permits NaN and infinity for fields whose kernel defines them.</p>
<p>Contributor authors can import <code>validateSinglePackedView(view, formats, name)</code> from
<code>@luma-contributors/analysis/analysis-core</code> for the common packed-format, alignment and stride
check. It rejects <code>GraphVectorView</code> even when it has one chunk.
<code>validateGraphOutputsDisjointFromInputs(id, outputs, inputs)</code> accepts records keyed by port name
and reports both conflicting names, including aliases of imported physical buffers. Array
arguments remain supported; their errors identify the two array indices.</p>
<p>Public props describe supported analysis choices. Benchmark batch sizes, forced cleanup passes,
kriging test chunk sizes, chunk-recursion row bases and extra cell-count destinations belong to
internal composition factories. Use <code>GPUCellPyramid</code> for per-level counts and pass the complete
chunked weights object to <code>GPUSpatialLag</code>; it handles row bases itself. Generalized GWR publishes
fitted means on the response scale; its bandwidth-selection candidates use link-scale fits internally.
Cell-pyramid and host-scheduled centrality compositions also share their binding preflights through
internal helpers. Their standalone contributors run these checks from <code>getCommandNodes</code> before
declaring scratch; callers do not need a separate validation call. The former <code>validateBindings</code>
and <code>validateGraphBindings</code> composition methods are private implementation details.</p>
<p>For contributor authors, <code>getWGSLKernelBinding</code> infers scalar component arrays only for packed,
32-bit-aligned <code>float32</code>, <code>uint32</code>, or <code>sint32</code> scalar/vector views (two to four components).
Atomic access requires an integer view. Other formats and strided layouts are rejected; an
explicit <code>WGSLKernelBinding</code> record is required for deliberate raw-word reinterpretation.
<code>createFillNode</code> fills every scalar component of these layouts by default. Its <code>componentCount</code>
is an integer prefix length from zero to <code>view.length * componentsPerRow</code>, measured in components.
Its <code>type</code> must match the view's scalar format (<code>f32</code>, <code>u32</code>, or <code>i32</code>). A zero-component fill
binds no target and records no commands, including for an empty view at the backing-buffer end.
<code>sliceGraphChunkGroupView</code> borrows a packed parent range with nonnegative integer row bounds
inside <code>view.length</code>; an empty slice at the end is valid. It rejects strided parents and never
expands into spare backing-buffer capacity. This descriptor helper accepts packed byte formats
at unaligned offsets and permits nested slicing; storage-word alignment is checked when binding.</p>
<h2 id="precision-of-longitudelatitude-positions">Precision of longitude/latitude positions</h2>
<p>Longitude/latitude positions are float32 degrees, never split into high/low parts. At
<code>|longitude| >= 128°</code>, the worst-case ground quantum is <code>Q = 2⁻¹⁶° · π/180 · R ≈ 1.70 m</code>
for the mean Earth radius <code>R = 6371008.8 m</code> (the equatorial worst case). Representative combined
east-west/north-south quanta are:</p>
<table>
<thead>
<tr>
<th>Location</th>
<th>Ground quantum</th>
</tr>
</thead>
<tbody>
<tr>
<td>Chicago</td>
<td>0.76 m</td>
</tr>
<tr>
<td>NYC</td>
<td>0.77 m</td>
</tr>
<tr>
<td>San Francisco</td>
<td>0.79 m</td>
</tr>
<tr>
<td>London</td>
<td>0.42 m</td>
</tr>
<tr>
<td>Sydney</td>
<td>1.47 m</td>
</tr>
<tr>
<td>Equator near ±180°</td>
<td>1.70 m</td>
</tr>
</tbody>
</table>
<p>Great-circle distance kernels use haversine, so short-distance arithmetic is not the limit;
the rounded inputs are. Use <code>createGPULocalFrame</code> for sub-17 m work, projecting before rounding
positions to float32 degrees. When a host-known distance is finite, greater than zero and less
than <code>10·Q</code> (16.97 m; 0.01697 km with the default sphere), operators warn once per operator class:
<code>&#x3C;Operation> distance is too short for float32 lon/lat; use createGPULocalFrame</code>.
This precision check never throws; zero, NaN and non-finite distances do not warn.</p>
<p>The exported helpers are <code>getGPULongitudeLatitudeQuantum(context)</code>,
<code>isGPUShortLongitudeLatitudeDistance(context, distance)</code> and
<code>warnGPUShortLongitudeLatitudeDistance(operation, context, distance)</code>, with
<code>GPU_SHORT_LONGITUDE_LATITUDE_DISTANCE_FACTOR = 10</code>. The quantum uses the context's sphere radius
(or ellipsoid semi-major axis) and length unit; it is zero for planar contexts or metric <code>'none'</code>.</p>
<ul>
<li><strong>Tier 1 — construction.</strong> <code>GPUSpatialPredicateJoin</code> checks numeric <code>dwithin</code> distances;
<code>GPUTrajectoryEncounters</code> checks its derived <code>maximumDistance</code>.</li>
<li><strong>Tier 2 — per-frame writers.</strong> Pass optional <code>spatialContext</code> in the settings passed to the
parameter writer to check neighbor-search <code>radius</code>, DBSCAN <code>epsilon</code>, IDW <code>searchRadius</code>,
kriging <code>searchRadius</code> and <code>range</code>, Douglas-Peucker <code>tolerance</code>, line segmentize / great-circle
arcs <code>maximumSegmentLength</code>, line chunk <code>chunkLength</code>, outline <code>distance</code>, map matching
<code>sigma</code> and <code>searchRadius</code>, and GWR fixed <code>bandwidths</code>. Packed parameter layouts are unchanged;
omitting the settings context disables this warning.</li>
<li><strong>Tier 3 — GPU views.</strong> No warning is possible for <code>GPUBufferSelection.distance</code>,
<code>GPUNearestFeatureJoin.radius</code>, a view-valued <code>GPUSpatialPredicateJoin.distance</code>,
<code>GPUTrajectoryEncounters.distance</code>, <code>GPUTrackSimilarity.frechetEpsilon</code>,
<code>GPULinearReferencing.radius</code>, <code>GPUNetworkSnapping.maximumSnapDistance</code>,
<code>GPUNetworkKFunction</code> distance views, the drive-time catchment recipe's <code>maximumSnapDistance</code>, or
<code>GPUGeodesicDestination.distances</code>: the host does not read these GPU values. Trajectory-metrics
stop detection also cannot warn: the affected distance is speed × sampling interval, so float32
position noise means about ±0.8 m/s at 1 Hz in Chicago; use a local frame.</li>
</ul>
<h2 id="running-contributors">Running contributors</h2>
<p>This applies to every contributor page (spatial analysis, network, terrain, raster, dataframe
analysis and geospatial). Applications compile with <code>await graph.compileAsync()</code>. It creates every
GPU pipeline asynchronously and concurrently, so the page keeps responding while a graph with many
kernels compiles. <code>graph.compile()</code> gives the same result but blocks the main thread until every
pipeline exists; keep it for tests and scripts. Either call freezes the graph, and a graph compiles
once, including when <code>compileAsync()</code> rejects.</p>
<p>Compile once per topology and rewrite parameter buffers every frame. A compile-time option (a grid
size, a method that adds nodes, an optional output) needs a new graph. Build the new graph and await
its compile while the old compiled graph keeps encoding. Then swap the new graph in, and only then
destroy the old one. If a later request arrives before the compile finishes, destroy the stale
result instead of swapping it in.</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">type</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> Options</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8"> {</span><span style="--shiki-light:#E36209;--shiki-dark:#FFAB70">gridSize</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">:</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8"> [</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">number</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">number</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">]};</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">declare</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> buildGraph</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#E36209;--shiki-dark:#FFAB70">options</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">:</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> Options</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">)</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">:</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUCommandGraph</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">;</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">let</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8"> compiled </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">=</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> await</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> buildGraph</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(options).</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">compileAsync</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">();</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">let</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8"> latestRequest </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> 0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">;</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">async</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> rebuild</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#E36209;--shiki-dark:#FFAB70">nextOptions</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">:</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> Options</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">)</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">:</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> Promise</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">&#x3C;</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">void</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">> {</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">  const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> request</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> ++</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">latestRequest;</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">  const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> next</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> await</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> buildGraph</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(nextOptions).</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">compileAsync</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(); </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// the old graph keeps encoding</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">  if</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8"> (request </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">!==</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8"> latestRequest) {</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    next.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">destroy</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(); </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// a later rebuild superseded this one</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">    return</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">;</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  }</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">  const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> previous</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8"> compiled;</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  compiled </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8"> next; </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// swap first</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  previous.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">destroy</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(); </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// then release transients and node state</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// every frame: record into the device's default encoder, then submit it</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">compiled.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">encode</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(device.commandEncoder, {parameters: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">undefined</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">});</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">device.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">submit</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">();</span></span></code></pre>
<p><code>encode()</code> only records commands; the application submits (<code>device.submit()</code> for the device's
default encoder, or <code>device.submit(encoder.finish())</code> for its own encoder). The second argument's
<code>parameters</code> is the graph's typed CPU channel, which node conditions and encode callbacks receive.
Numeric per-frame inputs are packed into GPU views (a <code>GPUParameterBuffer</code> written by a
<code>get*ParameterValues</code> packer, see each class's "Parameters" subsection). For those contributors,
use <code>GPUCommandGraph&#x3C;void></code> and pass <code>{parameters: undefined}</code>.</p>
<p><code>GPUSpatialJoinPrepared</code> and <code>GPUStudyWindowPreparation</code> also accept a CPU <code>rebuildWhen</code> predicate.
Give the handle and graph the same frame type. The first encoding and first encoding after
explicit invalidation bypass the predicate; otherwise it receives that frame once per encoding.
Requested geometry generations remain pending on the GPU until complete preparation executes,
including when an encoding is discarded. The current public builders replay against a retained
geometry snapshot each encoding; geometry refresh alone is GPU-gated. Use one active resumable
execution per handle/graph.</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">type</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> Frame</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8"> {</span><span style="--shiki-light:#E36209;--shiki-dark:#FFAB70">geometryChanged</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> boolean</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">};</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> graph</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUCommandGraph</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">&#x3C;</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">Frame</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">>(device);</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> prepared</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUSpatialJoinPrepared</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">&#x3C;</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">Frame</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">>({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  geometry: polygons,</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">  rebuildWhen</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">: </span><span style="--shiki-light:#E36209;--shiki-dark:#FFAB70">frame</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =></span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8"> frame.geometryChanged</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">});</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(prepared);</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// Add the joins that consume prepared before compiling.</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> compiled</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8"> graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">compile</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">();</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">compiled.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">encode</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(device.commandEncoder, {parameters: {geometryChanged: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">true</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}});</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">device.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">submit</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">();</span></span></code></pre>
<p>Both handles default their frame type to <code>unknown</code>; an annotated callback can infer it. When
using <code>invalidate()</code> alone, they also work with a <code>GPUCommandGraph&#x3C;void></code>.</p>
<p><code>GPUAdamStep</code>, <code>GPUFittingCoordinateEncoder</code>, <code>GPUFittingGradientSums</code>,
<code>GPUFittingMinibatchOrder</code>, <code>GPUPolygonTriangulation</code> and <code>GPUSegmentRingAssembly</code> snapshot
construction options, spatial contexts and port maps. Mutating the caller's props or replacing a
port afterward does not change their validated shapes or bindings. GPU views and buffers retain
their identities: update their contents for per-frame inputs, and construct a new contributor to
change topology.</p>
<p>Snapshots read declared fields through property access, so class getters and non-enumerable
ports are supported. Exposed spatial context descriptors and their nested metadata are frozen.</p>
<p>The gpu-raster base operators (<code>GPURasterStatistics</code>, the filters, morphology and region operators)
predate <code>GPUCommandNodeProducer</code>: they declare their work with <code>operator.addToGraph(graph)</code> instead
of <code>graph.add(operator)</code>. Compiling, encoding and submitting are the same.</p>
<p>Destroy a compiled graph before the caller-owned buffers it imports. Never compile, submit or read
back synchronously inside the frame loop.</p>
<h2 id="gpu-execution-model">GPU execution model</h2>
<p>How spatial analysis contributors run on the device: the command graph they declare, the kernels it
compiles to, and the resources each pass reads and writes, drawn from a recorded real run.</p>
<p>{/* lens:execution-model */}</p>
<p>Each contributor adds nodes to a <code>GPUCommandGraph</code>. <code>compile()</code> orders them by the buffers they read and write,
<code>describeGPUCommandGraph(compiled, options)</code> from <code>@luma-contributors/analysis/computation-record</code> records that structure
without running anything, and the lenses below draw it from recorded runs.</p>
<p>::lens{scenario=gpu-spatial-analysis.kernel-density rung=graph caption="Kernel density: grid index over the points, in-cell sort for determinism, then one gather pass."}</p>
<p>::lens{scenario=gpu-spatial-analysis.kernel-density rung=kernel caption="The gather kernel: one invocation per output cell; the binding table names every buffer it reads."}</p>
<p>::lens{scenario=gpu-spatial-analysis.point-in-polygon-join rung=graph caption="A join: bounding-box candidate pairs first, then one exact containment test per candidate pair."}</p>`};export{e as default};