# Computation record

## Overview

A `GPUComputationRecord` describes one recorded real execution. It identifies the adapter,
implementation variant, run, and creation time. Graphs and semantic metadata describe structure.
Metrics and captures are separate sidecars joined by `runId`, graph id, and node or resource id.
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
Transient resources may share allocations when their inclusive lifetimes do not overlap.
Retained resources describe observation runs that preserve intermediate storage.

## Axes, key spaces and relations

Axes describe navigation dimensions, including lane, workgroup, row, and iteration. An axis can
name a coarser `parent` and a `keySpace`. Relations map one key space to another, with explicit
cardinality. Function relations describe known arithmetic mappings. Buffer relations identify
stored mappings. A mapping that was not captured is `unknown`; consumers must never guess it.
A node's `datum` binds its invocations to a key space and an optional relation.

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
  seed: 42
});
const report = validateGPUComputationRecord(record);
```

Dependency edges are inferred from declared resource usage at buffer or texture granularity, in
scheduled order, so a few edges can be conservative. `describeGPUCommandGraphEncoding(stats, props)`
describes the realized passes of one encoding. Per-node pass membership is not published by
luma.gl, so it is inferred; when that disagrees with the pass count, each compute node is listed in
its own pass and the graph is marked `passGrouping: 'unknown'`.

`createGPUComputationRecord` accepts graphs directly or `{graph, sources}` descriptions. It fills
empty axes, refs, decisions, and choices. Sources with the same id are merged; different retained
text for one id throws. An entry without text does not erase retained text.

`getGPUComputationRunId` returns `run-<8 hex>` from ordered graph and node ids, adapter backend and
variant, dataset version, and seed. Pass `runId` to override it. `createdAt` defaults to the current
ISO timestamp; pass it explicitly for reproducible fixtures. The id hashes those identity inputs,
not every field of the record.

`validateGPUComputationRecord` accepts unknown JSON data and returns `{valid, issues}` without
throwing. Issues contain a level, field path, and message. It checks references, loop bounds,
resource lifetimes, and required structure. Missing compiled resources, unbound formula symbols,
and mappings attached to unknown relations produce warnings.

## Sidecars and observer effects

`createGPUComputationMetrics({runId, graph, timing, encoding})` converts a timing report to
nanosecond metrics. GPU timestamps are relative to the earliest begin timestamp in that report.
Nodes without timestamps keep their CPU encode time. Conditional outcomes distinguish executed,
skipped, and GPU-gated indirect work. The helper does not read timestamps from the GPU.

The required `encoding` flags describe observer effects: `coalesced`, `timestamped`, `taps`, and
`lensTwin`, plus optional retained resource ids. Per-node timestamps disable compute-pass
coalescing. Retention and taps can also change memory use and scheduling. Compare measurements
with these differences visible.

`createGPUComputationCaptures({runId, snapshots, stamps, replays})` fills omitted arrays with empty
arrays. Captures contain caller-supplied values, component counts, key spaces, and readback or
replay provenance. This helper performs no readback or replay.

`validateGPUComputationSidecars(record, {metrics, captures})` checks schemas, run identity, graph
and resource references, timestamp order, and whole component counts. If a capture omits its graph,
resource and node references are resolved across the record's graphs.

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
`Buffer.COPY_SRC` usage. Taps end compute-pass coalescing, so state them in
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
  Events use microseconds even though `displayTimeUnit` is `ns`. Namespace depth selects a thread.
  RAW dependencies become flow pairs by default. Counter series use the earliest timestamp of a
  matching loop iteration when the series axis is `iteration` or the loop id. If no such timestamp
  exists, the counter event uses timestamp zero. Metadata includes the schema, run id, adapter,
  and encoding flags. With several metrics sidecars, adapter and encoding metadata are arrays in
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
