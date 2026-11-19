var e={slug:`api/computation-record`,title:`Computation record`,html:`<h1 id="computation-record">Computation record</h1>
<h2 id="overview">Overview</h2>
<p>A <code>GPUComputationRecord</code> describes one recorded real execution. It identifies the adapter,
implementation variant, run, and creation time. Graphs and semantic metadata describe structure.
Metrics and captures are separate sidecars joined by <code>runId</code>, <code>executionId</code>, graph id, and node or
resource id.
This lets measurements from two runs share a structural comparison without mixing their timings.</p>
<p>The schema belongs to <code>@luma-contributors/analysis/computation-record</code> and does not promise external
compatibility. The entry builds on the public <code>GPUCommandGraph</code> of <code>@luma.gl/graph/gpu-core</code>, which has
no recording hooks, so recording is done beside the graph by this entry's own recorder. Record
construction and export operate on plain data without a GPU or DOM; the snapshot and mask-transfer
helpers below execute GPU work explicitly.</p>
<h2 id="graph-forms-and-identity">Graph forms and identity</h2>
<p>Graphs appear from most abstract to most concrete: <code>intent</code>, <code>lowered</code>, <code>compiled</code>, then <code>encoded</code>.
A record can omit forms. Each graph has a unique id without a colon. Node ids are unique within
that graph and may contain colons.</p>
<p>A node's <code>derivedFrom</code> array links to nodes in earlier graphs using <code>&#x3C;graphId>:&#x3C;nodeId></code>.
Fusion can reference several origins. Unrolled copies can share an origin. Use
<code>formatGPUComputationNodeRef(graphId, nodeId)</code> and <code>parseGPUComputationNodeRef(ref)</code> to handle these
references. Parsing splits at the first colon and throws if either part is empty.</p>
<p><code>namespace</code> is a slash-separated hierarchy. The empty string means top level. <code>loop</code> identifies
an unrolled copy by loop id and zero-based iteration. <code>dedupName</code> names the operation shared by
those copies. Consumers do not need to infer loops from node names.</p>
<p>Compiled resources carry ownership, byte size, allocation identity, and scheduled lifetimes.
Ownership is <code>imported</code> (caller-owned), <code>transient</code> (graph-owned), or <code>unknown</code> when no handle was
recorded. <code>bytesKind: 'logical'</code> identifies a buffer handle's size; <code>'lower-bound'</code> identifies the
largest observed <code>byteOffset + byteLength</code> without a handle; <code>'estimated'</code> identifies a texture
estimate. Recorded texture handles estimate all declared mips, layers/depth and samples using the
published texture-format decoder; preflight-only texture ranges supply a lower bound.
<code>unavailable</code> marks opaque storage (including external images); its numeric <code>bytes: 0</code> is a
placeholder, not a zero-byte estimate. A recorded external-image handle is imported even though
its storage size is unavailable. A lower bound may omit unused trailing storage. Text exports prefix lower bounds with
<code>>=</code> and estimates with <code>~</code>.
Transient resources may share allocations when their inclusive lifetimes do not overlap.
Retained resources describe observation runs that preserve intermediate storage.
Existing <code>/1</code> resources without <code>bytesKind</code> have unspecified size precision. Older unannotated
records could incorrectly say <code>imported</code>; use newly recorded handles to establish ownership.
Lens and website views conservatively display legacy imported resources without size precision as
unknown ownership. Unknown resources are listed separately from transient allocations; their sizes
cannot establish aliasing savings or a complete transient peak. Website summaries carry precision
through totals, lifetimes and dependency sizes and keep equal sizes with different precision or
ownership in separate rows.
Physical allocation totals use recorded <code>allocationBytes</code> for exact sizes. Without that size,
the largest logical member of a known shared allocation supplies a lower bound, independent of
resource order. Unknown ownership or allocation identity makes the total's precision unspecified.
Website lower-bound labels show exact byte counts (<code>≥1040 B</code>) so rounding never overstates a bound.
Text exports mark legacy imported resources as unknown ownership and label their size precision
unknown; explicitly logical imported resources retain their known ownership and size.
An omitted <code>graph.resources</code> inventory means unavailable counts and sizes. An explicitly empty
array establishes exact zero. Website summaries retain inventory presence and label unavailable
totals rather than presenting placeholder zeros as observations.</p>
<h2 id="axes-key-spaces-and-relations">Axes, key spaces and relations</h2>
<p>Axes describe navigation dimensions, including lane, workgroup, row, and iteration. An axis can
name a coarser <code>parent</code> and a <code>keySpace</code>. Relations map one key space to another, with explicit
cardinality. Function relations describe known arithmetic mappings. Buffer relations identify
stored mappings. A mapping that was not captured is <code>unknown</code>; consumers must never guess it.
A node's <code>datum</code> binds its invocations to a key space and an optional relation.</p>
<p>Line and polygon geometry can declare <code>featureKeySpace</code>: the domain of feature rows in the offsets
capture. This is separate from the
coordinate capture's <code>keySpace</code> (vertex rows); for example, a vessel track belongs to <code>track</code>, while
its positions belong to <code>fix</code>. Feature picking and linked selection are unavailable when this
domain is omitted. When supplied, it must name a declared key space.</p>
<p>For point geometry, <code>countCapture</code> names the captured scalar active count (by capture id or name).
It bounds the active prefix of the coordinate allocation; result figures use it for the headline,
projection and point sampling. The declared point key-space size, or otherwise the source buffer's
<code>logical</code> byte size, bounds the count when known; a capture-window end or resource lower bound does
not. Only the intersection of the active prefix and captured window is drawn. Missing, nonintegral,
sentinel or over-capacity bound counts make the panel unavailable. Point and raster capture windows
retain their absolute entity keys.</p>
<h2 id="refs-decisions-choices-and-workflow">Refs, decisions, choices and workflow</h2>
<p>The <code>refs</code> tables deduplicate formulas and source modules. Nodes reference their ids. Formula
symbols can bind to resources or shader variables. Formulas state whether they are derived or
curated. Source text is optional; its id and hash can remain when text retention is disabled.</p>
<p><code>decisions</code> preserve the chosen implementation and every candidate, including rejected and
unsupported candidates with their reasons. <code>choices</code> describe analyst-selected parameters and
alternatives in analysis-stage order. A <code>workflow</code> lists graph invocations in execution order,
including the state or reason that selected them.</p>
<h2 id="recording-a-command-graph">Recording a command graph</h2>
<p><code>GPUComputationRecorder</code> collects node annotations while the graph is built, and
<code>describeGPUCommandGraph(compiled, {recorder})</code> merges them with the facts the compiled graph
publishes (<code>id</code>, <code>stats</code> and <code>preflight</code>). Pass each node through the recorder, for example with
<code>recorder.addNode(graph, node, annotation)</code>, and use <code>recorder.withScope</code> to attach a namespace,
operation or loop to the nodes contributed inside the callback. Scopes nest and their namespaces join.</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">import</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8"> {GPUCommandGraph} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">from</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF"> '@luma.gl/graph/gpu-core'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">;</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">import</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8"> {</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  GPUComputationRecorder,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  createGPUComputationRecord,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  describeGPUCommandGraph,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  getGPUComputationAdapter,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  validateGPUComputationRecord</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">from</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF"> '@luma-contributors/analysis/computation-record'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">;</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> graph</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUCommandGraph</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(device);</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> recorder</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUComputationRecorder</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({sourceText: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">true</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">});</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">recorder.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">withScope</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({namespace: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'solve'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, op: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'relax'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}, () </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">=></span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8"> {</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">  // Contribute the real computation's nodes here, with recorder.addNode(graph, node).</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">});</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> compiled</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> await</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8"> graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">compileAsync</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">();</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> description</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> describeGPUCommandGraph</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(compiled, {recorder});</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> record</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> createGPUComputationRecord</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  id: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'relaxation'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  adapter: </span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">getGPUComputationAdapter</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(device, </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'portable'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  domains: [</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'gpu-core'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">],</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  graphs: [description],</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  datasetVersion: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'example-v1'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  parameters: {relaxation: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">0.5</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">},</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  implementationRevision: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'relaxation-v2'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  seed: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">42</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">});</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> report</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> validateGPUComputationRecord</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(record);</span></span></code></pre>
<h3 id="kernel-facts-without-a-recorder-prop">Kernel facts without a recorder prop</h3>
<p>Analysis contributors do not take a recorder. Every kernel they build through
<code>createWGSLKernelNode</code> (directly, or through <code>createFillNode</code>, <code>createPublishNode</code>,
<code>createWGSLDispatchGate</code> users and similar helpers) can still be described: call
<code>recordWGSLKernelNodes(graph)</code> from <code>@luma-contributors/analysis/analysis-core</code>, or
<code>recorder.recordKernels(graph)</code>, right after creating the graph and before adding contributors.
<code>describeGPUCommandGraph(compiled, {graph, sourceText})</code> (or <code>{recorder}</code> after <code>recordKernels</code>)
then gives each such node its operation name as <code>op</code>, a <code>wgsl:&#x3C;hash></code> source with module id
<code>&#x3C;operation>/&#x3C;variant></code>, entry point and <code>bodyHash</code>, its <code>workgroupSize</code>, <code>dispatch</code> (static bound,
<code>indirect</code> when a GPU-written condition supplies it), per-pipeline <code>constants</code>, and its storage and
sampled-texture <code>bindings</code> with access, element type, offset and length. Resources bound by those
kernels get their real <code>transient</code> or <code>imported</code> ownership, and the graph gets the published
transient allocation totals as <code>attrs</code> (<code>physicalTransientBufferCount</code>, <code>reusedTransientBytes</code>, ...).
Annotations passed to <code>recorder.recordNode</code> win field by field, so packs add formulas, loops and
<code>derivedFrom</code> on top. <code>getWGSLKernelNodeRecord(graph, nodeId)</code> returns the raw facts.</p>
<p>The custom point-cloud chunk, shared row-chunk and classification collective factories also
register their actual modules, storage bindings and dispatch layouts when recording is enabled.
They retain their runtime parameter bindings and pipeline sharing; recording does not change
the shader or dispatch. Other hand-built kernels need an explicit registration to carry facts.</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">import</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8"> {recordWGSLKernelNodes} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">from</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF"> '@luma-contributors/analysis/analysis-core'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">;</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> graph</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUCommandGraph</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(device, {id: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'documented-run'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">});</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">recordWGSLKernelNodes</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(graph);</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(contributor); </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// any analysis contributor, unchanged</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> compiled</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> await</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8"> graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">compileAsync</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">();</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> description</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> describeGPUCommandGraph</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(compiled, {graph, sourceText: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">true</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">});</span></span></code></pre>
<p>The table is keyed weakly by graph and by node id within it (the compiled graph publishes node ids,
and <code>graph.add</code> copies node objects). A graph that never calls <code>recordWGSLKernelNodes</code> keeps
nothing; <code>createWGSLKernelNode</code> then costs one <code>WeakMap</code> lookup. Nodes of published <code>@luma.gl/graph</code>
primitives (<code>GPUScan</code>, <code>GPUSort</code>, <code>GPUReduction</code>, <code>GPUGroupAggregation</code>, raster neighbourhoods) are
not built by the helper and carry only published facts.</p>
<p>Dependency edges are inferred from declared resource usage at buffer or texture granularity, in
scheduled order, so a few edges can be conservative. <code>describeGPUCommandGraphEncoding(stats, props)</code>
describes the realized passes of one encoding. Per-node pass membership is not published by
luma.gl, so it is inferred; when that disagrees with the pass count, each compute node is listed in
its own pass and the graph is marked <code>passGrouping: 'unknown'</code>.</p>
<p><code>createGPUComputationRecord</code> accepts graphs directly or <code>{graph, sources}</code> descriptions. It fills
empty axes, refs, decisions, and choices. Sources with the same id are merged; different retained
text for one id throws. An entry without text does not erase retained text.</p>
<p><code>getGPUComputationRunId</code> returns deterministic <code>run-&#x3C;8 hex></code> comparison/permalink identity from
ordered graph and node ids, all adapter fields, dataset version, seed, <code>parameters</code>, choice
stage/name/value triples, and <code>implementationRevision</code>. Parameter key insertion order does not
matter. Supply all per-frame scalar settings in <code>parameters</code> and identify changed input data with
<code>datasetVersion</code>; buffers are not read or hashed. The default implementation revision is
<code>analysis-2026-11-18-review-1</code>; callers should pass their application/source revision and change it after
correctness fixes. The library default must advance when library fixes invalidate old comparisons.
These expanded hash inputs change default IDs from earlier records. Pass <code>runId</code> explicitly to
retain an existing permalink when appropriate; the id does not hash every field of the record.</p>
<p>Parameter and choice values accept strings, booleans and finite numbers only. Nested values,
NaN and infinities are rejected before hashing (even with an explicit runId) and when loading a
record. Numeric seeds must also be finite. Construction copies adapter fields, choices (including
alternatives), parameter maps and graph/node identities so editing caller inputs for a later run
cannot change an earlier execution's identity data. S2 lenses retain the full scalar choice map as
parameters, including settings omitted from display templates.
S3 declared guard metadata is supplied as presentation choices before record construction.
Readback observations, including conservation sums and island counts, are stored in graph attributes
and do not mutate identity-bearing choices. Spatial guards read both sources. The public lens
decoder validates identity values at its decoded-envelope boundary, like the core validator.</p>
<p><code>createGPUComputationRecord</code> also assigns a fresh UUID <code>executionId</code> for each call. Repeated
executions with identical comparison inputs have the same <code>runId</code> and different <code>executionId</code>s.
<code>createdAt</code> defaults to the current ISO timestamp. Supply both <code>createdAt</code> and <code>executionId</code>
explicitly for reproducible fixtures; creation time is not the execution join key.</p>
<p><code>validateGPUComputationRecord</code> accepts unknown JSON data and returns <code>{valid, issues}</code> without
throwing. Issues contain a level, field path, and message. It checks references, loop bounds,
resource lifetimes, and required structure. Missing compiled resources, unbound formula symbols,
and mappings attached to unknown relations produce warnings.
An omitted compiled resource inventory produces a warning; when an inventory is supplied, every
node <code>reads</code> and <code>writes</code> reference must name a resource in it. An unlisted resource is an error.</p>
<h2 id="sidecars-and-observer-effects">Sidecars and observer effects</h2>
<p><code>createGPUComputationMetrics({runId, executionId, graph, timing, encoding})</code> converts a timing report to
nanosecond metrics. GPU timestamps are relative to the earliest begin timestamp in that report.
Nodes without timestamps keep their CPU encode time. Conditional outcomes distinguish executed,
skipped, and GPU-gated indirect work. The helper does not read timestamps from the GPU.</p>
<p><code>submitGPUCommandGraphWithTimings(device, compiled, encodeOptions, {timestamps?})</code> encodes and
submits once. When the device has <code>timestamp-query</code> (and <code>timestamps</code> is not false) it profiles
every pass through the command encoder's public <code>timeProfilingQuerySet</code>; <code>readTimings()</code> then
returns the published timing report for <code>createGPUComputationMetrics</code> and frees the query set.
Without timestamps <code>timestamped</code> is false and <code>readTimings()</code> resolves to <code>undefined</code>, so a record
states no timing rather than zero.</p>
<p>A metrics sidecar captured on a device with <code>timestamp-query</code> therefore holds device- and
run-specific timings, and a re-capture is not byte-identical. The varying fields are, per node of
<code>nodes</code>, <code>gpuBegin</code> and <code>gpuEnd</code> (relative to the earliest begin of the report, in the sidecar's
<code>timebase</code>, <code>'ms'</code> when the timing report is in milliseconds and <code>'ns'</code> otherwise) and
<code>cpuEncodeMs</code>, together with <code>encoding.timestamped</code>, which is <code>true</code> only when the timestamps were
written. Compare records by their graph, captures and counters, and treat these fields as
measurements rather than as reproducible content.</p>
<p>The required <code>encoding</code> flags describe observer effects: <code>coalesced</code>, <code>timestamped</code>, <code>taps</code>, and
<code>lensTwin</code>, plus optional retained resource ids. Per-node timestamps disable compute-pass
coalescing. Retention and taps can also change memory use and scheduling. Compare measurements
with these differences visible.</p>
<p><code>createGPUComputationCaptures({runId, executionId, snapshots, stamps, replays})</code> fills omitted arrays
with empty arrays. Captures contain caller-supplied values, component counts, key spaces, and readback or
replay provenance. This helper performs no readback or replay.</p>
<p><code>validateGPUComputationSidecars(record, {metrics, captures})</code> checks schemas, run identity, graph
and resource references, timestamp order, and whole component counts. If a capture omits its graph,
resource and node references are resolved across the record's graphs.
Copy both <code>runId: record.runId</code> and <code>executionId: record.executionId</code> into every sidecar for a new
record. A missing execution id on only one side, or different execution ids, is an error even when
run, graph and node ids match. Legacy <code>/1</code> pairs with both execution ids absent remain accepted
using structural identity only; this cannot establish that they came from the same execution.
The <code>/1</code> schema identifiers remain unchanged for these additive fields and the <code>unknown</code> ownership
value. Old readers must accept the extended ownership union before consuming new descriptions.
Lens scenario harnesses copy both ids into captures and timings. Lens serialization, loading and
pure view contexts, plus website extraction, also reject mismatched execution identity before
displaying results. Scalar settings in spatial scenario choice maps are passed as <code>parameters</code>;
other harnesses include their declared choice values in the deterministic run hash.</p>
<h2 id="snapshots">Snapshots</h2>
<p><code>GPUSnapshotArena</code> and <code>GPUSnapshotTap</code> capture intermediate values without graph hooks.
<code>new GPUSnapshotArena(device, {id, slots, capacity, maximumByteLength})</code> allocates one
<code>MAP_READ | COPY_DST</code> buffer. Each slot declares <code>name</code>, <code>byteLength</code>, <code>elementType</code>, and optional
<code>components</code>. <code>encodeSnapshot(encoder, sources, label?)</code> copies all slots to frame <code>count</code> and
returns that frame index. Sources are <code>{buffer, byteOffset?}</code> by slot name. Sizes and offsets must be
multiples of four. Missing sources, exhausted capacity, and an exceeded byte budget throw errors
naming the arena. The default byte budget is 64 MiB; decimate map-scale histories.</p>
<p>A <code>GPUSnapshotTap</code> is an ordinary copy node of the published <code>GPUCommandGraph</code>. Add it right after
the node it observes, because resource hazards follow authoring order. The observed buffers need
<code>Buffer.COPY_SRC</code> usage. At construction, <code>views</code> keys must equal the arena slot names. Each data
view must be packed (<code>byteStride === rowByteLength</code>) with a four-byte-aligned offset, and the
slot's byte length must not exceed <code>view.length * view.rowByteLength</code>. A smaller slot captures a
packed prefix; it never extends into neighbouring storage. Whole-buffer handles start at zero and
must have at least the slot's byte length. The tap snapshots the props and view map at construction;
The tap retains each validated buffer handle and byte offset independently of the caller's view
object; later in-place edits to that object's shape cannot change a copy range. GPU buffer contents
remain live inputs. Direct arena calls still check
copy ranges against their backing buffers. Taps end compute-pass coalescing, so state them in
<code>GPUComputationEncodingFlags.taps</code> and do not present timings of a tapped run as production timings.</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">import</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8"> {GPUSnapshotArena, GPUSnapshotTap} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">from</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF"> '@luma-contributors/analysis/computation-record'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">;</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> arena</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUSnapshotArena</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(device, {</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  id: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'residual'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  slots: [{name: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'residual'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, byteLength: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">4096</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, elementType: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'f32'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}],</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  capacity: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">8</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">});</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUSnapshotTap</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({id: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'tap-residual'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, arena, views: {residual: residualView}, afterNode: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'relax'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span></code></pre>
<p>Submit all frames before calling <code>read({graph, resources, keySpaces})</code>. The arena maps once and
returns independent typed-array copies with <code>gpu-readback</code> provenance and ids <code>&#x3C;arena>-&#x3C;slot>-&#x3C;frame></code>.
Labels accept <code>iteration</code> (defaults to frame index), <code>afterNode</code>, and <code>label</code> (stored as capture
<code>meaning</code>). Call <code>destroy()</code> after reading; returned values remain valid. <code>createGPUComputationCaptures</code>
turns the read snapshots into the captures sidecar.</p>
<h2 id="gputransfermask"><code>GPUTransferMask</code></h2>
<p><code>new GPUTransferMask({id, source, relation, target, mode?})</code> transfers a GPU selection without
readback. Add it with <code>graph.add(mask)</code>.
The two nodes, <code>&#x3C;id>-clear</code> and <code>&#x3C;id>-scatter</code>, clear the target and scatter selected source keys.
All views must be packed <code>uint32</code>; the target must use storage separate from source and relation.</p>
<ul>
<li><code>{layout: 'index', targets}</code> maps source key <code>i</code> to <code>targets[i]</code>. Target and source lengths match.
<code>0xffffffff</code> means no target; all indices outside the target mask are ignored.</li>
<li><code>{layout: 'csr', offsets, neighbors}</code> maps source key <code>i</code> to
<code>neighbors[offsets[i]..offsets[i + 1])</code>. Offsets have <code>source.length + 1</code> entries. Use one entry
per source-to-target pair when counting selected keys; duplicate entries count separately.</li>
</ul>
<p>Nonzero source flags are selected. <code>mode: 'any'</code> (the default) writes one to every reached target;
<code>mode: 'count'</code> counts selected incoming relation entries using atomic addition. Both modes use
bounded three-dimensional dispatch. Nodes declare resource hazards and a <code>GPUTransferMask</code> workload
with variant <code>&#x3C;layout>-&#x3C;mode></code>. Pass a <code>recorder</code> to add operator attributes and hashed WGSL source
refs; <code>GPUComputationRecorder({sourceText: true})</code> additionally retains shader text.</p>
<h2 id="exports">Exports</h2>
<ul>
<li><code>toChromeTraceEvents(record, metrics, {flows})</code> returns Chrome Trace Event JSON for Perfetto.
It rejects mismatched run/execution ids before joining timings, including missing-vs-present
execution identity; matching legacy pairs remain supported.
Events use microseconds even though <code>displayTimeUnit</code> is <code>ns</code>. Namespace depth selects a thread.
RAW dependencies become flow pairs by default. Counter series use the earliest timestamp of a
matching loop iteration when the series axis is <code>iteration</code> or the loop id. If no such timestamp
exists, the counter event uses timestamp zero. Metadata includes the schema, run id, execution id
when present, adapter, and encoding flags. With several metrics sidecars, adapter and encoding metadata are arrays in
input order, and each sidecar retains its relative time origin.</li>
<li><code>toModelExplorerGraph(record)</code> returns one Model Explorer graph per form, with namespaces,
incoming dependencies, and string attributes.</li>
<li><code>toGraphvizDOT(record, {graph, includeResources})</code> returns monochrome DOT. It selects the last
graph by default. Namespace segments become nested clusters. Optional note nodes show resources.</li>
<li><code>printGPUComputationGraph(graph)</code> and <code>printGPUComputationRecord(record)</code> produce deterministic
text for golden tests and logs. Their text formats are not an API promise.</li>
</ul>
<p>All exporters preserve deterministic ordering and add no measurements or GPU work. Serialize the
Chrome trace object as JSON and load it into Perfetto. Serialize the Model Explorer collection
inside the collection array expected by its JSON loader.</p>
<h2 id="production-cost">Production cost</h2>
<p>Recording is opt-in. Graphs built without a <code>GPUComputationRecorder</code> add no nodes, passes,
allocations, per-node metadata objects, or string building. The pure helpers on this page run only
when the application calls them. Timestamp collection, snapshots and taps remain explicit choices.</p>`};export{e as default};