# Computation record

## Overview

A `GPUComputationRecord` describes one recorded real execution. It identifies the adapter,
implementation variant, run, and creation time. Graphs and semantic metadata describe structure.
Metrics and captures are separate sidecars joined by `runId`, `executionId`, graph id, and node or
resource id.
This lets measurements from two runs share a structural comparison without mixing their timings.

The schema belongs to `@luma-contributors/analysis/computation-record` and does not promise external
compatibility. The entry builds on the public `GPUCommandGraph` of `@luma.gl/graph/gpu-core`, which has
no recording hooks, so recording is done beside the graph by this entry's own recorder. Record
construction and export operate on plain data without a GPU or DOM; the snapshot and mask-transfer
helpers below execute GPU work explicitly.

## Graph forms and identity

Graphs appear from most abstract to most concrete: `intent`, `lowered`, `compiled`, then `encoded`.
A record can omit forms. Each graph has a unique id without a colon. Node ids are unique within
that graph and may contain colons.

A node's `derivedFrom` array links to nodes in earlier graphs using `<graphId>:<nodeId>`.
Fusion can reference several origins. Unrolled copies can share an origin. Use
`formatGPUComputationNodeRef(graphId, nodeId)` and `parseGPUComputationNodeRef(ref)` to handle these
references. Parsing splits at the first colon and throws if either part is empty.

`namespace` is a slash-separated hierarchy. The empty string means top level. `loop` identifies
an unrolled copy by loop id and zero-based iteration. `dedupName` names the operation shared by
those copies. Consumers do not need to infer loops from node names.

Compiled resources carry ownership, byte size, allocation identity, and scheduled lifetimes.
Ownership is `imported` (caller-owned), `transient` (graph-owned), or `unknown` when no handle was
recorded. `bytesKind: 'logical'` identifies a buffer handle's size; `'lower-bound'` identifies the
largest observed `byteOffset + byteLength` without a handle; `'estimated'` identifies a texture
estimate. Recorded texture handles estimate all declared mips, layers/depth and samples using the
published texture-format decoder; preflight-only texture ranges supply a lower bound.
`unavailable` marks opaque storage (including external images); its numeric `bytes: 0` is a
placeholder, not a zero-byte estimate. A recorded external-image handle is imported even though
its storage size is unavailable. A lower bound may omit unused trailing storage. Text exports prefix lower bounds with
`>=` and estimates with `~`.
Transient resources may share allocations when their inclusive lifetimes do not overlap.
Retained resources describe observation runs that preserve intermediate storage.
Existing `/1` resources without `bytesKind` have unspecified size precision. Older unannotated
records could incorrectly say `imported`; use newly recorded handles to establish ownership.
Lens and website views conservatively display legacy imported resources without size precision as
unknown ownership. Unknown resources are listed separately from transient allocations; their sizes
cannot establish aliasing savings or a complete transient peak. Website summaries carry precision
through totals, lifetimes and dependency sizes and keep equal sizes with different precision or
ownership in separate rows.
Physical allocation totals use recorded `allocationBytes` for exact sizes. Without that size,
the largest logical member of a known shared allocation supplies a lower bound, independent of
resource order. Unknown ownership or allocation identity makes the total's precision unspecified.
Website lower-bound labels show exact byte counts (`≥1040 B`) so rounding never overstates a bound.
Text exports mark legacy imported resources as unknown ownership and label their size precision
unknown; explicitly logical imported resources retain their known ownership and size.
An omitted `graph.resources` inventory means unavailable counts and sizes. An explicitly empty
array establishes exact zero. Website summaries retain inventory presence and label unavailable
totals rather than presenting placeholder zeros as observations.

## Axes, key spaces and relations

Axes describe navigation dimensions, including lane, workgroup, row, and iteration. An axis can
name a coarser `parent` and a `keySpace`. Relations map one key space to another, with explicit
cardinality. Function relations describe known arithmetic mappings. Buffer relations identify
stored mappings. A mapping that was not captured is `unknown`; consumers must never guess it.
A node's `datum` binds its invocations to a key space and an optional relation.

Line and polygon geometry can declare `featureKeySpace`: the domain of feature rows in the offsets
capture. This is separate from the
coordinate capture's `keySpace` (vertex rows); for example, a vessel track belongs to `track`, while
its positions belong to `fix`. Feature picking and linked selection are unavailable when this
domain is omitted. When supplied, it must name a declared key space.

For point geometry, `countCapture` names the captured scalar active count (by capture id or name).
It bounds the active prefix of the coordinate allocation; result figures use it for the headline,
projection and point sampling. The declared point key-space size, or otherwise the source buffer's
`logical` byte size, bounds the count when known; a capture-window end or resource lower bound does
not. Only the intersection of the active prefix and captured window is drawn. Missing, nonintegral,
sentinel or over-capacity bound counts make the panel unavailable. Point and raster capture windows
retain their absolute entity keys.

## Refs, decisions, choices and workflow

The `refs` tables deduplicate formulas and source modules. Nodes reference their ids. Formula
symbols can bind to resources or shader variables. Formulas state whether they are derived or
curated. Source text is optional; its id and hash can remain when text retention is disabled.

`decisions` preserve the chosen implementation and every candidate, including rejected and
unsupported candidates with their reasons. `choices` describe analyst-selected parameters and
alternatives in analysis-stage order. A `workflow` lists graph invocations in execution order,
including the state or reason that selected them.

## Recording a command graph

`GPUComputationRecorder` collects node annotations while the graph is built, and
`describeGPUCommandGraph(compiled, {recorder})` merges them with the facts the compiled graph
publishes (`id`, `stats` and `preflight`). Pass each node through the recorder, for example with
`recorder.addNode(graph, node, annotation)`, and use `recorder.withScope` to attach a namespace,
operation or loop to the nodes contributed inside the callback. Scopes nest and their namespaces join.

```ts
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  GPUComputationRecorder,
  createGPUComputationRecord,
  describeGPUCommandGraph,
  getGPUComputationAdapter,
  validateGPUComputationRecord
} from '@luma-contributors/analysis/computation-record';

const graph = new GPUCommandGraph(device);
const recorder = new GPUComputationRecorder({sourceText: true});
recorder.withScope({namespace: 'solve', op: 'relax'}, () => {
  // Contribute the real computation's nodes here, with recorder.addNode(graph, node).
});
const compiled = await graph.compileAsync();
const description = describeGPUCommandGraph(compiled, {recorder});
const record = createGPUComputationRecord({
  id: 'relaxation',
  adapter: getGPUComputationAdapter(device, 'portable'),
  domains: ['gpu-core'],
  graphs: [description],
  datasetVersion: 'example-v1',
  parameters: {relaxation: 0.5},
  implementationRevision: 'relaxation-v2',
  seed: 42
});
const report = validateGPUComputationRecord(record);
```

### Kernel facts without a recorder prop

Analysis contributors do not take a recorder. Every kernel they build through
`createWGSLKernelNode` (directly, or through `createFillNode`, `createPublishNode`,
`createWGSLDispatchGate` users and similar helpers) can still be described: call
`recordWGSLKernelNodes(graph)` from `@luma-contributors/analysis/analysis-core`, or
`recorder.recordKernels(graph)`, right after creating the graph and before adding contributors.
`describeGPUCommandGraph(compiled, {graph, sourceText})` (or `{recorder}` after `recordKernels`)
then gives each such node its operation name as `op`, a `wgsl:<hash>` source with module id
`<operation>/<variant>`, entry point and `bodyHash`, its `workgroupSize`, `dispatch` (static bound,
`indirect` when a GPU-written condition supplies it), per-pipeline `constants`, and its storage and
sampled-texture `bindings` with access, element type, offset and length. Resources bound by those
kernels get their real `transient` or `imported` ownership, and the graph gets the published
transient allocation totals as `attrs` (`physicalTransientBufferCount`, `reusedTransientBytes`, ...).
Annotations passed to `recorder.recordNode` win field by field, so packs add formulas, loops and
`derivedFrom` on top. `getWGSLKernelNodeRecord(graph, nodeId)` returns the raw facts.

The custom point-cloud chunk, shared row-chunk and classification collective factories also
register their actual modules, storage bindings and dispatch layouts when recording is enabled.
They retain their runtime parameter bindings and pipeline sharing; recording does not change
the shader or dispatch. Other hand-built kernels need an explicit registration to carry facts.

```ts
import {recordWGSLKernelNodes} from '@luma-contributors/analysis/analysis-core';

const graph = new GPUCommandGraph(device, {id: 'documented-run'});
recordWGSLKernelNodes(graph);
graph.add(contributor); // any analysis contributor, unchanged
const compiled = await graph.compileAsync();
const description = describeGPUCommandGraph(compiled, {graph, sourceText: true});
```

The table is keyed weakly by graph and by node id within it (the compiled graph publishes node ids,
and `graph.add` copies node objects). A graph that never calls `recordWGSLKernelNodes` keeps
nothing; `createWGSLKernelNode` then costs one `WeakMap` lookup. Nodes of published `@luma.gl/graph`
primitives (`GPUScan`, `GPUSort`, `GPUReduction`, `GPUGroupAggregation`, raster neighbourhoods) are
not built by the helper and carry only published facts.

Dependency edges are inferred from declared resource usage at buffer or texture granularity, in
scheduled order, so a few edges can be conservative. `describeGPUCommandGraphEncoding(stats, props)`
describes the realized passes of one encoding. Per-node pass membership is not published by
luma.gl, so it is inferred; when that disagrees with the pass count, each compute node is listed in
its own pass and the graph is marked `passGrouping: 'unknown'`.

`createGPUComputationRecord` accepts graphs directly or `{graph, sources}` descriptions. It fills
empty axes, refs, decisions, and choices. Sources with the same id are merged; different retained
text for one id throws. An entry without text does not erase retained text.

`getGPUComputationRunId` returns deterministic `run-<8 hex>` comparison/permalink identity from
ordered graph and node ids, all adapter fields, dataset version, seed, `parameters`, choice
stage/name/value triples, and `implementationRevision`. Parameter key insertion order does not
matter. Supply all per-frame scalar settings in `parameters` and identify changed input data with
`datasetVersion`; buffers are not read or hashed. The default implementation revision is
`analysis-2026-11-18-review-1`; callers should pass their application/source revision and change it after
correctness fixes. The library default must advance when library fixes invalidate old comparisons.
These expanded hash inputs change default IDs from earlier records. Pass `runId` explicitly to
retain an existing permalink when appropriate; the id does not hash every field of the record.

Parameter and choice values accept strings, booleans and finite numbers only. Nested values,
NaN and infinities are rejected before hashing (even with an explicit runId) and when loading a
record. Numeric seeds must also be finite. Construction copies adapter fields, choices (including
alternatives), parameter maps and graph/node identities so editing caller inputs for a later run
cannot change an earlier execution's identity data. S2 lenses retain the full scalar choice map as
parameters, including settings omitted from display templates.
S3 declared guard metadata is supplied as presentation choices before record construction.
Readback observations, including conservation sums and island counts, are stored in graph attributes
and do not mutate identity-bearing choices. Spatial guards read both sources. The public lens
decoder validates identity values at its decoded-envelope boundary, like the core validator.

`createGPUComputationRecord` also assigns a fresh UUID `executionId` for each call. Repeated
executions with identical comparison inputs have the same `runId` and different `executionId`s.
`createdAt` defaults to the current ISO timestamp. Supply both `createdAt` and `executionId`
explicitly for reproducible fixtures; creation time is not the execution join key.

`validateGPUComputationRecord` accepts unknown JSON data and returns `{valid, issues}` without
throwing. Issues contain a level, field path, and message. It checks references, loop bounds,
resource lifetimes, and required structure. Missing compiled resources, unbound formula symbols,
and mappings attached to unknown relations produce warnings.
An omitted compiled resource inventory produces a warning; when an inventory is supplied, every
node `reads` and `writes` reference must name a resource in it. An unlisted resource is an error.

## Sidecars and observer effects

`createGPUComputationMetrics({runId, executionId, graph, timing, encoding})` converts a timing report to
nanosecond metrics. GPU timestamps are relative to the earliest begin timestamp in that report.
Nodes without timestamps keep their CPU encode time. Conditional outcomes distinguish executed,
skipped, and GPU-gated indirect work. The helper does not read timestamps from the GPU.

`submitGPUCommandGraphWithTimings(device, compiled, encodeOptions, {timestamps?})` encodes and
submits once. When the device has `timestamp-query` (and `timestamps` is not false) it profiles
every pass through the command encoder's public `timeProfilingQuerySet`; `readTimings()` then
returns the published timing report for `createGPUComputationMetrics` and frees the query set.
Without timestamps `timestamped` is false and `readTimings()` resolves to `undefined`, so a record
states no timing rather than zero.

A metrics sidecar captured on a device with `timestamp-query` therefore holds device- and
run-specific timings, and a re-capture is not byte-identical. The varying fields are, per node of
`nodes`, `gpuBegin` and `gpuEnd` (relative to the earliest begin of the report, in the sidecar's
`timebase`, `'ms'` when the timing report is in milliseconds and `'ns'` otherwise) and
`cpuEncodeMs`, together with `encoding.timestamped`, which is `true` only when the timestamps were
written. Compare records by their graph, captures and counters, and treat these fields as
measurements rather than as reproducible content.

The required `encoding` flags describe observer effects: `coalesced`, `timestamped`, `taps`, and
`lensTwin`, plus optional retained resource ids. Per-node timestamps disable compute-pass
coalescing. Retention and taps can also change memory use and scheduling. Compare measurements
with these differences visible.

`createGPUComputationCaptures({runId, executionId, snapshots, stamps, replays})` fills omitted arrays
with empty arrays. Captures contain caller-supplied values, component counts, key spaces, and readback or
replay provenance. This helper performs no readback or replay.

`validateGPUComputationSidecars(record, {metrics, captures})` checks schemas, run identity, graph
and resource references, timestamp order, and whole component counts. If a capture omits its graph,
resource and node references are resolved across the record's graphs.
Copy both `runId: record.runId` and `executionId: record.executionId` into every sidecar for a new
record. A missing execution id on only one side, or different execution ids, is an error even when
run, graph and node ids match. Legacy `/1` pairs with both execution ids absent remain accepted
using structural identity only; this cannot establish that they came from the same execution.
The `/1` schema identifiers remain unchanged for these additive fields and the `unknown` ownership
value. Old readers must accept the extended ownership union before consuming new descriptions.
Lens scenario harnesses copy both ids into captures and timings. Lens serialization, loading and
pure view contexts, plus website extraction, also reject mismatched execution identity before
displaying results. Scalar settings in spatial scenario choice maps are passed as `parameters`;
other harnesses include their declared choice values in the deterministic run hash.

## Snapshots

`GPUSnapshotArena` and `GPUSnapshotTap` capture intermediate values without graph hooks.
`new GPUSnapshotArena(device, {id, slots, capacity, maximumByteLength})` allocates one
`MAP_READ | COPY_DST` buffer. Each slot declares `name`, `byteLength`, `elementType`, and optional
`components`. `encodeSnapshot(encoder, sources, label?)` copies all slots to frame `count` and
returns that frame index. Sources are `{buffer, byteOffset?}` by slot name. Sizes and offsets must be
multiples of four. Missing sources, exhausted capacity, and an exceeded byte budget throw errors
naming the arena. The default byte budget is 64 MiB; decimate map-scale histories.

A `GPUSnapshotTap` is an ordinary copy node of the published `GPUCommandGraph`. Add it right after
the node it observes, because resource hazards follow authoring order. The observed buffers need
`Buffer.COPY_SRC` usage. At construction, `views` keys must equal the arena slot names. Each data
view must be packed (`byteStride === rowByteLength`) with a four-byte-aligned offset, and the
slot's byte length must not exceed `view.length * view.rowByteLength`. A smaller slot captures a
packed prefix; it never extends into neighbouring storage. Whole-buffer handles start at zero and
must have at least the slot's byte length. The tap snapshots the props and view map at construction;
The tap retains each validated buffer handle and byte offset independently of the caller's view
object; later in-place edits to that object's shape cannot change a copy range. GPU buffer contents
remain live inputs. Direct arena calls still check
copy ranges against their backing buffers. Taps end compute-pass coalescing, so state them in
`GPUComputationEncodingFlags.taps` and do not present timings of a tapped run as production timings.

```ts
import {GPUSnapshotArena, GPUSnapshotTap} from '@luma-contributors/analysis/computation-record';

const arena = new GPUSnapshotArena(device, {
  id: 'residual',
  slots: [{name: 'residual', byteLength: 4096, elementType: 'f32'}],
  capacity: 8
});
graph.add(new GPUSnapshotTap({id: 'tap-residual', arena, views: {residual: residualView}, afterNode: 'relax'}));
```

Submit all frames before calling `read({graph, resources, keySpaces})`. The arena maps once and
returns independent typed-array copies with `gpu-readback` provenance and ids `<arena>-<slot>-<frame>`.
Labels accept `iteration` (defaults to frame index), `afterNode`, and `label` (stored as capture
`meaning`). Call `destroy()` after reading; returned values remain valid. `createGPUComputationCaptures`
turns the read snapshots into the captures sidecar.

## `GPUTransferMask`

`new GPUTransferMask({id, source, relation, target, mode?})` transfers a GPU selection without
readback. Add it with `graph.add(mask)`.
The two nodes, `<id>-clear` and `<id>-scatter`, clear the target and scatter selected source keys.
All views must be packed `uint32`; the target must use storage separate from source and relation.

- `{layout: 'index', targets}` maps source key `i` to `targets[i]`. Target and source lengths match.
  `0xffffffff` means no target; all indices outside the target mask are ignored.
- `{layout: 'csr', offsets, neighbors}` maps source key `i` to
  `neighbors[offsets[i]..offsets[i + 1])`. Offsets have `source.length + 1` entries. Use one entry
  per source-to-target pair when counting selected keys; duplicate entries count separately.

Nonzero source flags are selected. `mode: 'any'` (the default) writes one to every reached target;
`mode: 'count'` counts selected incoming relation entries using atomic addition. Both modes use
bounded three-dimensional dispatch. Nodes declare resource hazards and a `GPUTransferMask` workload
with variant `<layout>-<mode>`. Pass a `recorder` to add operator attributes and hashed WGSL source
refs; `GPUComputationRecorder({sourceText: true})` additionally retains shader text.

## Exports

- `toChromeTraceEvents(record, metrics, {flows})` returns Chrome Trace Event JSON for Perfetto.
  It rejects mismatched run/execution ids before joining timings, including missing-vs-present
  execution identity; matching legacy pairs remain supported.
  Events use microseconds even though `displayTimeUnit` is `ns`. Namespace depth selects a thread.
  RAW dependencies become flow pairs by default. Counter series use the earliest timestamp of a
  matching loop iteration when the series axis is `iteration` or the loop id. If no such timestamp
  exists, the counter event uses timestamp zero. Metadata includes the schema, run id, execution id
  when present, adapter, and encoding flags. With several metrics sidecars, adapter and encoding metadata are arrays in
  input order, and each sidecar retains its relative time origin.
- `toModelExplorerGraph(record)` returns one Model Explorer graph per form, with namespaces,
  incoming dependencies, and string attributes.
- `toGraphvizDOT(record, {graph, includeResources})` returns monochrome DOT. It selects the last
  graph by default. Namespace segments become nested clusters. Optional note nodes show resources.
- `printGPUComputationGraph(graph)` and `printGPUComputationRecord(record)` produce deterministic
  text for golden tests and logs. Their text formats are not an API promise.

All exporters preserve deterministic ordering and add no measurements or GPU work. Serialize the
Chrome trace object as JSON and load it into Perfetto. Serialize the Model Explorer collection
inside the collection array expected by its JSON loader.

## Production cost

Recording is opt-in. Graphs built without a `GPUComputationRecorder` add no nodes, passes,
allocations, per-node metadata objects, or string building. The pure helpers on this page run only
when the application calls them. Timestamp collection, snapshots and taps remain explicit choices.
