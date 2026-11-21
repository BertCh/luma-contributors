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
`recorder.addNode(graph, node, annotation)`, and use `recorder.withSynchronousScope` to attach a namespace,
operation or loop to the nodes contributed inside the callback. Scopes nest and their namespaces join.
A native async callback is rejected before invocation. For an ordinary callback, a thenable result is rejected after the callback returns;
its rejection is observed, but already-started work cannot be cancelled. Both synchronous methods
unwind the scope on return or error. Use `withAsyncScope` for asynchronous authoring and record
through the supplied recorder, including after each `await` and in nested scopes. It captures a
separate scope stack while sharing nodes, resources and kernel recording with the parent;
concurrent tasks keep independent namespaces even when they interleave or throw.

```ts
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  GPUComputationRecorder,
  createGPUComputationRecord,
  describeGPUCommandGraph,
  getGPUComputationAdapter,
  validateGPUComputationRecord
} from '@luma-contributors/analysis/computation-record';

// Supplied by this application's build manifest, using its actual resolved graph version/commit.
declare const capturingBuild: {dependencyRevision: string};
const graph = new GPUCommandGraph(device);
const recorder = new GPUComputationRecorder({sourceText: true});
recorder.withSynchronousScope({namespace: 'solve', op: 'relax'}, () => {
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
  dependencyRevision: capturingBuild.dependencyRevision,
  seed: 42
});
const report = validateGPUComputationRecord(record);
compiled.destroy();
```

For asynchronous graph authoring, prepare application nodes inside the callback and add them
through its scoped recorder before compiling the graph. Await all tasks before compilation:

```ts
import type {GPUCommandNode} from '@luma.gl/graph/gpu-core';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUComputationRecorder} from '@luma-contributors/analysis/computation-record';

// Each call prepares a node with its own unique graph ID.
declare function prepareNode(namespace: string): Promise<GPUCommandNode>;
const graph = new GPUCommandGraph(device);
const recorder = new GPUComputationRecorder();
await recorder.withAsyncScope({namespace: 'basin'}, async scopedRecorder => {
  const node = await prepareNode('basin');
  scopedRecorder.addNode(graph, node); // Recorded in basin even after the await.
});

const basinTask = recorder.withAsyncScope({namespace: 'basin'}, async scopedRecorder => {
  const node = await prepareNode('basin');
  scopedRecorder.addNode(graph, node);
});
const coastTask = recorder.withAsyncScope({namespace: 'coast'}, async scopedRecorder => {
  const node = await prepareNode('coast');
  scopedRecorder.addNode(graph, node);
});
await Promise.all([basinTask, coastTask]); // Independent basin and coast namespaces.
const compiled = await graph.compileAsync();
compiled.destroy();
```

The record saves `datasetVersion`, `seed` (including zero), parameters, choices, and implementation
and dependency revisions. To replay it, resolve the saved dataset version and the saved implementation
and dependency revisions, rebuild the graph with the recorded parameters, choices and seed, then
submit it as a new execution. Attach the new observations with the new execution ID and
`gpu-replay` provenance when they recover the earlier run's state; retain the original record as
the provenance source. Legacy records may omit dataset and seed. The saved identity inputs can
be passed to `getGPUComputationRunId(record)` to reproduce a generated run ID; they do not recover
an application-supplied custom run ID. Record construction snapshots plain graph, workflow and
other metadata, so later edits to the caller's descriptors do not rewrite this execution.

The following application-level replay uses application-supplied resolvers.
`resolveImplementation` selects the saved computation ID and exact implementation and dependency
revisions. Its builder applies the recorded parameters, choices and seed, including zero, and
reports the implementation variant it actually used for the new record's adapter metadata.
It prepares any snapshot taps before compilation, returns readbacks from that new graph, and
owns its imported buffers. `destroy()` releases those buffers after the compiled graph is destroyed.
Missing legacy identity fields require application-supplied evidence; this example refuses to guess.
`resolveCaptureBuildEvidence` looks up the original execution's trusted build manifest or equivalent
archive and throws if that evidence is unavailable. A record's nonempty `dependencyRevision` alone
does not establish the dependency used: it may be the library's integration-target fallback.
The evidence must verify the actual resolved dependency and agree with the saved revision before
replay. If it disagrees, retain the original record and resolve its provenance before proceeding.

```ts
import type {Device} from '@luma.gl/core';
import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  createGPUComputationRecord, createGPUComputationCaptures,
  describeGPUCommandGraph, getGPUComputationAdapter,
  type GPUComputationRecord, type GPUComputationCapture
} from '@luma-contributors/analysis/computation-record';

declare const savedRecord: GPUComputationRecord;
declare function resolveCaptureBuildEvidence(record: GPUComputationRecord): Promise<{
  dependencyRevision: string;
}>;
declare function resolveDataset(version: string): Promise<unknown>;
declare function resolveImplementation(computationId: string, revision: string,
  dependencyRevision: string): Promise<{
  build(device: Device, dataset: unknown,
    settings: Pick<GPUComputationRecord, 'parameters' | 'choices' | 'seed'>): Promise<{
      graph: GPUCommandGraph;
      variant: string;
      readSnapshots(): Promise<GPUComputationCapture[]>;
      destroy(): void;
    }>;
}>;

if (savedRecord.datasetVersion === undefined || savedRecord.seed === undefined ||
    savedRecord.implementationRevision === undefined || savedRecord.dependencyRevision === undefined) {
  throw new Error('Replay needs the saved dataset, seed, implementation and dependency revisions');
}
const captureBuild = await resolveCaptureBuildEvidence(savedRecord);
if (captureBuild.dependencyRevision !== savedRecord.dependencyRevision) {
  throw new Error('Saved dependencyRevision disagrees with the capturing build evidence');
}
const dataset = await resolveDataset(savedRecord.datasetVersion);
const implementation = await resolveImplementation(
  savedRecord.id, savedRecord.implementationRevision, captureBuild.dependencyRevision
);
const rebuilt = await implementation.build(device, dataset, {
  parameters: savedRecord.parameters, choices: savedRecord.choices, seed: savedRecord.seed
});
let compiled: Awaited<ReturnType<GPUCommandGraph['compileAsync']>> | undefined;
try {
  compiled = await rebuilt.graph.compileAsync();
  const replayRecord = createGPUComputationRecord({
    id: savedRecord.id, adapter: getGPUComputationAdapter(device, rebuilt.variant),
    domains: savedRecord.domains, graphs: [describeGPUCommandGraph(compiled, {graph: rebuilt.graph})],
    axes: savedRecord.axes, parameters: savedRecord.parameters, choices: savedRecord.choices,
    datasetVersion: savedRecord.datasetVersion, seed: savedRecord.seed,
    implementationRevision: savedRecord.implementationRevision,
    dependencyRevision: savedRecord.dependencyRevision
  }); // Fresh executionId and createdAt; runId is recomputed for the replay's adapter and graph.
  const encoder = device.createCommandEncoder();
  compiled.encode(encoder, {parameters: undefined});
  device.submit(encoder.finish());
  const captures = createGPUComputationCaptures({
    runId: replayRecord.runId, executionId: replayRecord.executionId,
    replays: (await rebuilt.readSnapshots()).map(capture => ({
      ...capture, provenance: 'gpu-replay' as const
    }))
  });
  const replayObservation = {
    replayOf: {runId: savedRecord.runId, executionId: savedRecord.executionId},
    record: replayRecord, captures
  };
  console.log(replayObservation); // Save alongside the original record, never over it.
} finally {
  compiled?.destroy();
  rebuilt.destroy();
}
```

`replayOf` is application envelope metadata linking the saved execution to the new observation;
it is not an extra record-schema field. The captures join the new record's identity. They do not
replace the original values or inherit its execution ID, and replay timings describe the new run.

### Kernel facts without a recorder prop

Analysis contributors do not take a recorder. Every kernel they build through
`createWGSLKernelNode` (directly, or through `createFillNode`, `createPublishNode`,
`createWGSLDispatchGate` users and similar helpers) can still be described: call
`recordWGSLKernelNodes(graph)` from `@luma-contributors/analysis/analysis-core`, or
`recorder.recordKernels(graph)`, right after creating the graph and before adding contributors.
`recordKernels` returns the original graph with its frame-parameter type for authoring.
`recorder.getRecordedGraphDescription(compiled)` returns `undefined` if you have yet to call
`recordKernels`. After attaching the recorder, pass the exact object returned by the graph's
`compile()` or `compileAsync()` method. A compilation from another builder is rejected even when
its graph and node IDs match.
`describeGPUCommandGraph(compiled, {graph, sourceText})` (or `{recorder}` after `recordKernels`)
then gives each such node its operation name as `op`, a `wgsl:<hash>` source with module id
`<operation>/<variant>`, entry point and `bodyHash`, its `workgroupSize`, `dispatch` (static bound,
`indirect` when a GPU-written condition supplies it), per-pipeline `constants`, and its storage and
sampled-texture `bindings` with access, element type, offset and length. Resources bound by those
kernels get their real `transient` or `imported` ownership, and the graph gets the published
transient allocation totals as `attrs` (`physicalTransientBufferCount`, `reusedTransientBytes`, ...).
Defined fields merge in this order: explicit `recorder.recordNode` annotations, local kernel-factory
facts, then public preflight kernel facts. Matching source refs merge defined fields and retain
available text when recording opted in; only two conflicting retained texts are rejected. Effective
constants resolve within each annotation as `annotation.constants ?? annotation.source.constants`
before precedence is applied. Packs can add formulas, loops and `derivedFrom` on top.
Compile with `{kernelMetadata: 'layout'}` to retain published entry points, workgroup sizes and
`declaredBindings` (name, type, group and location), or `'source'` to also make WGSL available.
The default compilation does not collect these upstream kernel facts. Public metadata covers
upstream scans, including scans nested in radix sort; an unannotated primitive can still lack it.
Layout-only descriptions keep entry points on nodes even without a source ref. Declared slots do
not identify graph resources, so they never fabricate resource-linked `bindings`, constants or
ownership; preflight-only resources retain unknown ownership and lower-bound byte extents.
Retaining text in a compiled graph does not opt a record into retaining it: the recorder's
`sourceText: true` (or the description's explicit `sourceText: true`) is independently required.
`getWGSLKernelNodeRecord(graph, nodeId)` returns the raw facts.

Nodes whose WGSL text is identical share one source ref, whose `entryPoint` and `constants` are
those of the first such node. Each node keeps its own pipeline specialisation: `constants` is
always on the node when known, including `{}` when shader defaults are used, and `entryPoint` is
on the node when it differs from the shared ref,
so read `node.entryPoint ?? source.entryPoint` and `node.constants`.

The custom point-cloud chunk, shared row-chunk and classification collective factories also
register their actual modules, storage bindings and dispatch layouts when recording is enabled.
They retain their runtime parameter bindings and pipeline sharing; recording does not change
the shader or dispatch. Other hand-built kernels need an explicit registration to carry facts.

```ts
import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {recordWGSLKernelNodes} from '@luma-contributors/analysis/analysis-core';
import {GPUCubeLayoutTranspose} from '@luma-contributors/analysis/dataframe-analysis';
import {describeGPUCommandGraph} from '@luma-contributors/analysis/computation-record';

const graph = new GPUCommandGraph(device, {id: 'documented-run'});
recordWGSLKernelNodes(graph);
// Declaration example: a two-cell, three-period cube; no execution or readback here.
const values = createTransientView(graph, 'values', 'float32', 6);
const output = createTransientView(graph, 'transposed', 'float32', 6);
const contributor = new GPUCubeLayoutTranspose({
  values, output, from: 'cell-major', cellCount: 2, periodCount: 3
});
graph.add(contributor);
const compiled = await graph.compileAsync();
const description = describeGPUCommandGraph(compiled, {graph, sourceText: true});
compiled.destroy();
```

Automatic descriptions use facts captured by the selected kernel's successful compilation, so
unused same-ID declarations cannot replace its source, dispatch or bindings. Custom nodes without
compiled records contribute only attributable published metadata; unknown facts remain absent.
The table is keyed weakly by graph and node ID. A graph builder supports one compilation; standalone
node inspection before or after it cannot replace records captured from the selected graph callbacks.
A graph that never calls `recordWGSLKernelNodes` keeps
nothing; `createWGSLKernelNode` then costs one `WeakMap` lookup. Nodes of published `@luma.gl/graph`
primitives (`GPUScan`, `GPUSort`, `GPUReduction`, `GPUGroupAggregation`, raster neighbourhoods) are
not built by the helper and carry only published facts.

Dependency edges are inferred from declared resource usage at buffer or texture granularity, in
scheduled order, so a few edges can be conservative. `describeGPUCommandGraphEncoding(stats, props)`
describes the realized passes of one encoding. Per-node pass membership is not published by
luma.gl, so it is inferred; when that disagrees with the pass count, each compute node is listed in
its own pass and the graph is marked `passGrouping: 'unknown'`. Pass ids `pass-<n>` number compute
and render passes in one encoded order (copies are separate `copy-<id>` nodes), so in a mixed graph
`n` is not the compute-pass index that `getGPUCommandGraphPassIndices` returns.

`createGPUComputationRecord` accepts graphs directly or `{graph, sources}` descriptions. It fills
empty axes, refs, decisions, and choices. Sources with the same id are merged; different retained
text for one id throws. Matching refs merge defined fields while preserving the first ref’s pipeline
metadata; adding text never replaces its entry point or constants. An entry without text does not
erase retained text, and bare graphs retain the metadata supplied through top-level source refs.

`getGPUComputationRunId` returns deterministic `run-<8 hex>` comparison/permalink identity from
ordered graph and node ids, all adapter fields, dataset version, seed, `parameters`, choice
stage/name/value triples, `implementationRevision`, and `dependencyRevision`. Parameter key insertion
order does not matter. Supply all per-frame scalar settings in `parameters` and identify changed input data with
`datasetVersion`; buffers are not read or hashed. The default implementation revision is
`analysis-2026-11-18-review-1`; callers should pass their application/source revision and change it after
correctness fixes. The library default must advance when library fixes invalidate old comparisons.
New captures also record `dependencyRevision`, defaulting to the concrete integration target
`@luma.gl/graph@10.0.0-alpha.16` from `dev-docs/release-package.md`. This is a source-controlled
revision, not a runtime dependency lookup or the accepted peer range. It does not detect a supported
peer upgrade in the capturing application: changing only the resolved graph package does not
automatically change this default or its hash. Supply `dependencyRevision` for the actual capturing
build until resolved graph revision metadata is available through a public upstream API or build
integration; the integration-target fallback alone does not establish execution provenance.
Use an exact package version or commit for the supplied revision. Both record construction and the standalone ID helper include this revision in the hash;
an explicit `runId` is preserved. Legacy `/1` records without the field remain readable. A supplied
revision must be a nonempty string.
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
metrics with `timebase: 'ns'` for nanosecond fields or `'ms'` when any node carries
`gpuStartTimeMilliseconds`, the published form. GPU timestamps are relative to the earliest begin timestamp in
that report. A finite nonnegative `gpuTimeMilliseconds` is retained as optional node `gpuTimeMs`,
always in milliseconds regardless of the endpoint timebase. Measured zero is retained; absent or
invalid duration measurements are omitted. A duration alone does not invent begin/end timestamps.
Nodes without timing keep their CPU encode time. Conditional outcomes distinguish executed,
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
`timebase`, `'ms'` when the timing report is in milliseconds and `'ns'` otherwise),
`gpuTimeMs` (an independent millisecond duration when measured) and `cpuEncodeMs`, together with
`encoding.timestamped`, which is `true` only when the timestamps were written. Compare records by their graph, captures and counters, and treat these fields as
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
all resource and node references must resolve together to exactly one graph; ambiguous references
require an explicit graph. `resolveGPUComputationCaptureGraph(record, capture)` returns that graph,
or `undefined` for missing, ambiguous or contradictory references, without modifying the capture.
Capture IDs must be unique across snapshots, stamps and replays. Capture ID, name and provenance
are required, components must be a positive integer when supplied, and legacy null lanes are rejected.
Timing endpoints must be finite and ordered. Metrics construction omits invalid endpoint pairs and
chooses its origin from accepted pairs, preserving independent measurements and measured-zero durations.
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

For lossless capture persistence, use the public codec rather than `JSON.stringify` on typed arrays:

```typescript
import {
  createGPUComputationCaptures,
  serializeGPUComputationCaptures,
  deserializeGPUComputationCaptures,
  type GPUComputationRecord
} from '@luma-contributors/analysis/computation-record';

declare const record: GPUComputationRecord; // The execution whose result was read back.
const captures = createGPUComputationCaptures({
  runId: record.runId,
  executionId: record.executionId,
  snapshots: [{
    id: 'finite-and-missing', name: 'result', graph: record.graphs[0].id,
    elementType: 'f32', values: Float32Array.of(2, NaN), provenance: 'gpu-readback'
  }]
});
const saved = serializeGPUComputationCaptures(captures);
const loaded = deserializeGPUComputationCaptures(saved);
const values = loaded.snapshots[0].values;
if (values.length !== 2) throw new Error('Expected exactly two captured values');
if (!Number.isFinite(values[0]) || values[0] !== 2) throw new Error('Expected finite result 2');
if (!Number.isNaN(values[1])) throw new Error('Expected a missing result (NaN)');
```

The JSON wire format keeps explicit `elementType`, `b64:` little-endian 32-bit bytes, and
`valueCount`. It preserves typed-subarray byte windows, NaN payload bits and signed zero, including
arrays from another JavaScript realm. Loading also accepts older numeric arrays and the established
f32 tokens `"NaN"`, `"Infinity"` and `"-Infinity"`; null lanes and invalid integer lanes are rejected.

Raster geometry can include `affine: [a,b,c,d,e,f]`, mapping column/row coordinates to
`x = a*column + b*row + c`, `y = d*column + e*row + f` in `coordinateModel`/`unit` units.
`pixelInterpretation: 'area'` (default) samples cell centres at `(column+0.5,row+0.5)`;
`'point'` samples integer indices. `frameId` identifies the coordinate frame and `crs` the CRS
(for example `EPSG:3857`). Affine placement overrides scalar placement. Older square, axis-aligned
rasters use `[cellSize,0,originX,0,rowStep,originY]`, with cellSize default 1 and rowStep negative
for default southward rows, positive for northward rows. An omitted origin is
`[0,shape[1]*cellSize]` for southward rows and `[0,0]` for northward rows. Thus shape `[1,2]`,
cellSize 3 and omitted origin have first-cell centre `[1.5,4.5]`. The helper
`getGPUComputationRasterSamplePosition(geometry,column,row)` resolves these conventions. Consumers
must compare effective sample placement, frame/CRS identity and interpretation before subtracting
rasters, and must render the affine placement when present.

Kernel bindings have two coordinate systems: `elementOffset` is the shader-relative element prefix
inside an aligned binding window; optional `bindingByteOffset` and `bindingByteLength` describe that
window in graph-resource bytes. Three-u32 views starting at bytes 8 and 264 therefore have window
offsets 0 and 256, lengths 20 and element prefixes 2. Window offsets must be 256-byte aligned,
window lengths positive, and windows must fit known logical resource byte lengths. Historical records
may omit the byte-range fields.

## Snapshots

`GPUSnapshotArena` and `GPUSnapshotTap` capture intermediate values without graph hooks.
`new GPUSnapshotArena(device, {id, slots, capacity, maximumByteLength})` allocates one
`MAP_READ | COPY_DST` buffer. Each slot declares `name`, `byteLength`, `elementType`, and optional
`components`. `encodeSnapshot(encoder, sources, label?)` copies all slots to frame `count` and
returns that frame index. Sources are `{buffer, byteOffset?}` by slot name. Sizes and offsets must be
multiples of four. Missing sources, exhausted capacity, and an exceeded byte budget throw errors
naming the arena. The default byte budget is 64 MiB; decimate map-scale histories.

Allowed slot element types are `f32`, `i32` or `u32`, decoded as Float32Array, Int32Array or
Uint32Array respectively. Unknown element types reject before allocation. The arena copies and
freezes the descriptors and their array at construction: descriptor order and slot properties
are immutable, so later caller edits cannot relabel captured words.
Slot names are arbitrary strings; `sources`, `resources` and `keySpaces` are own-property
dictionaries keyed by those names. An inherited source is treated as missing and rejected;
inherited optional resource/key-space metadata is omitted, including inherited getters and names
such as `constructor` or `toString`. Explicit own entries with those names remain supported.

### `GPUSnapshotTap`

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
import type {GraphDataView} from '@luma.gl/graph/gpu-core';
import {GPUSnapshotArena, GPUSnapshotTap} from '@luma-contributors/analysis/computation-record';

// Reuse the 1,024-word view already written by the producer node `relax` in this graph.
// Its buffer has COPY_SRC usage. Do not import that physical buffer a second time.
declare const residualView: GraphDataView<'float32'>;
const arena = new GPUSnapshotArena(device, {
  id: 'residual',
  slots: [{name: 'residual', byteLength: 4096, elementType: 'f32'}],
  capacity: 8
});
graph.add(new GPUSnapshotTap({id: 'tap-residual', arena, views: {residual: residualView}, afterNode: 'relax'}));
```

Submit all retained frames before calling `read({graph, resources, keySpaces})`. The
first read permanently seals capture, checkpoints, submission marking and discard, even if mapping
fails. Concurrent reads share one in-flight mapping of the populated byte prefix
`(0, count * frameByteLength)`; successful bytes are cached. A later read can retry after rejection,
including a synchronous mapping error or truncated readback, without reopening reservations.
Empty byte ranges are not mapped (no frames, or zero bytes per frame).
Each read returns independent typed-array copies with `gpu-readback` provenance and ids `<arena>-<slot>-<frame>`;
mutating one returned capture cannot change the cached bytes or another read's values.
Labels accept `iteration` (defaults to frame index), `afterNode`, and `label` (stored as capture
`meaning`). Call `destroy()` after reading; returned values remain valid. `createGPUComputationCaptures`
turns the read snapshots into the captures sidecar.

An encoder can reserve frames without being submitted, including through `GPUSnapshotTap` during
graph encoding. Those reservations are not observations. Call `arena.checkpoint()` before encoding
cancellable work, then `arena.discard(checkpoint)` to remove only the unsubmitted suffix, including
tap reservations. Earlier submitted frame numbers remain valid. The discarded encoder must never
subsequently be submitted, because its pending copies could overwrite reused frame slots.

Call `arena.markSubmitted()` immediately before submission to protect the entire contiguous
reserved prefix, including in-flight frames, from discard. Only mark a prefix whose encoders will
all be submitted. A failed submission leaves that prefix protected; it cannot be reused by discard.
When later encoders already have reservations, use `arena.markSubmitted(endCheckpoint)` to protect
only the earlier contiguous prefix. Take that end checkpoint after encoding the earlier prefix
and before reserving the later frames. Neither submission nor discard can split an encoder's
reservations; submitted or discarded encoders cannot reserve more frames. The arena does not
detect arbitrary device submissions: marking the prefix is the caller's responsibility.
`count` includes unsubmitted reservations until they are discarded.

```ts
import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import type {GPUSnapshotArena} from '@luma-contributors/analysis/computation-record';

declare const arena: GPUSnapshotArena;
declare const compiled: Awaited<ReturnType<GPUCommandGraph['compileAsync']>>;
declare const keepFrame: boolean; // Application decision for this encoding.
const checkpoint = arena.checkpoint();
const encoder = device.createCommandEncoder();
try {
  compiled.encode(encoder, {parameters: undefined}); // Taps may reserve several frames.
} catch (error) {
  arena.discard(checkpoint);
  throw error; // This encoder is permanently abandoned.
}
if (keepFrame) {
  arena.markSubmitted();
  device.submit(encoder.finish());
} else {
  arena.discard(checkpoint); // Never finish/submit this encoder afterward.
}
```

For example, reserve two encoders but submit only the first prefix initially:

```ts
import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import type {GPUSnapshotArena} from '@luma-contributors/analysis/computation-record';

declare const arena: GPUSnapshotArena;
declare const compiled: Awaited<ReturnType<GPUCommandGraph['compileAsync']>>;
const earlierEncoder = device.createCommandEncoder();
compiled.encode(earlierEncoder, {parameters: undefined});
const endCheckpoint = arena.checkpoint();
const laterEncoder = device.createCommandEncoder();
compiled.encode(laterEncoder, {parameters: undefined});
arena.markSubmitted(endCheckpoint);
device.submit(earlierEncoder.finish());
// Abandon laterEncoder permanently; its suffix alone is still discardable.
arena.discard(endCheckpoint);
```

For example, after submitted frame 0 holds 13, discarding an unsubmitted tap reservation at
frame 1 removes it from the eventual read. A subsequent submitted value 15 reuses frame 1;
the readable history is `[13, 15]`, with frame 0 unchanged. Use a checkpoint for each arena touched
by the encoder. This supports suffix cancellation, not removing arbitrary frames from its middle.
Destroying the arena also loses unread submitted frames, including those before the cancelled
suffix. Only independent capture copies already returned by an earlier read remain valid.

## `GPUTransferMask`

`new GPUTransferMask({id, source, relation, target, mode?})` transfers a GPU selection without
readback. Add it with `graph.add(mask)`.
The two nodes, `<id>-clear` and `<id>-scatter`, clear the target and scatter selected source keys.
All bindings take `GraphDataView<'uint32'>`. The views must be packed, and the target must use
storage separate from source and relation.

- `{layout: 'index', targets}` maps source key `i` to `targets[i]` and requires
  `relation.targets.length === source.length`. `target.length` is the size of the destination key space;
  it can differ from the source length. For example, six source flags and six relation target indices
  can write a two-word destination mask. `0xffffffff` means no target; all indices outside the
  target mask are ignored.
- `{layout: 'csr', offsets, neighbors}` maps source key `i` to
  `neighbors[offsets[i]..offsets[i + 1])`. Offsets have `source.length + 1` entries. Use one entry
  per source-to-target pair when counting selected keys; duplicate entries count separately.

Nonzero source flags are selected. `mode: 'any'` (the default) writes one to every reached target;
`mode: 'count'` counts selected incoming relation entries using atomic addition. Both modes use
bounded three-dimensional dispatch. Nodes declare resource hazards and a `GPUTransferMask` workload
with variant `<layout>-<mode>`. Pass a `recorder` to add operator attributes and hashed WGSL source
refs; `GPUComputationRecorder({sourceText: true})` additionally retains shader text.

Construction accepts only `mode: 'any'` or `'count'`; omission (`mode === undefined`) defaults to
`'any'`. `relation.layout` must be `'index'` or `'csr'`; there is no default layout, and the matching
relation views are required. Null, empty strings and other unknown values reject before graph authoring,
without allocating GPU work. Selector errors name the prop, accepted values and the received value.
The mode and relation descriptor fields are captured at construction, preserving the GPU view and
buffer identities; later edits to the caller's relation map cannot switch the authored layout.

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
