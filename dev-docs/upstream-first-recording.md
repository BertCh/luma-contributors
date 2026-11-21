# Computation records on published luma.gl

`modules/analysis/src/computation-record` builds records from the public `GPUCommandGraph` API of published `@luma.gl/graph`. The [alpha package contract](./release-package.md) records the installed integration target and accepted peer range separately. No private graph fields or recording hooks are required.

## Public graph data

| Source | Recorded information |
| --- | --- |
| `CompiledGPUCommandGraph.id`, `.stats` | Graph ID, topological `nodeOrder`, aggregate transient buffer/texture allocation and reuse totals. |
| `.preflight.nodes` | Node ID, type, description, operation, variant, workload bounds, declared resources, conditions and publications. Resource summaries include kind, usage, byte range, format, length and size. The describer consumes opt-in `.kernel.entryPoint`, `.workgroupSize`, `.bindings` (name/type/group/location) and optional `.source`, beneath local factory facts and explicit annotations (see below). |
| `GPUCommandGraphEncoding.stats` | Per-node CPU encode time, conditional outcome and pass counters. |
| `GPUCommandGraphEncoding.readTimings()` | GPU start and end in milliseconds when timestamps are available. An independently measured `gpuTimeMilliseconds` becomes `GPUComputationNodeMetrics.gpuTimeMs`, always in milliseconds, including zero and when both endpoints are absent. An unavailable duration stays absent. |
| Public node and resource handles | `dependsOn` and resource ownership for explicitly recorded nodes; buffer/texture dimensions and transient state. |

`describeGPUCommandGraph` in `computation-record/describe-command-graph.ts` walks `stats.nodeOrder` and merges the matching preflight nodes with optional analysis annotations. It returns `{graph, sources}` for `createGPUComputationRecord` in `computation-record/create-computation-record.ts`.

Read/write edges are inferred from declared usage in scheduled order at whole-buffer or whole-texture granularity. Texture mip and layer overlap is not modelled, so `raw`, `war` and `waw` edges can be conservative. Explicit `dependsOn` edges are available only for nodes passed through the recorder. Resource ownership comes from recorder handles or kernel bindings; otherwise the resource has `unknown` ownership, with the summary's byte count (`lower-bound` when only observed bytes exist).

## Kernel and domain facts (LN-REC)

`recordWGSLKernelNodes(graph)` in `analysis-core/kernels/wgsl-kernel-nodes.ts` opts a graph in before nodes are built. `GPUComputationRecorder.recordKernels(graph)` in `computation-record/computation-recorder.ts` does the same and retains the graph reference. `createWGSLKernelNode` then stores props, generated source, workgroup size and bounded layout in a graph-keyed `WeakMap`, indexed by node ID. Without opt-in it performs one lookup and stores nothing.

`createWGSLKernelNodeAnnotation` in `computation-record/describe-command-graph.ts` produces operation, source hash and body hash, entry point, workgroup size, dispatch, constants and typed bindings from that table. Full source text is included only when `sourceText` is enabled. Explicit `recordNode` annotations take precedence field by field.

Published alpha.16 has a separate compile opt-in: `graph.compile({kernelMetadata: 'layout'})`
or `compileAsync` with the same option collects declared entry point, workgroup size and binding
name/type/group/location in `preflight.nodes[].kernel`. Use `'source'` to include WGSL text too;
ordinary `compile()`/`compileAsync()` collects neither. Metadata coverage is per compute node:
`GPUScan` declares these facts on its scan and offset-add nodes. Sort-owned histogram/scatter
kernels and reduction-owned kernels lack metadata, but nested scan nodes in a radix sort expose
their layout and optional source. A declaration-only alpha.16 probe with default sort options
and no subgroups found zero metadata-bearing nodes in the one-node 16-row bitonic route, and
eight in the 24-node 513-row radix route; all eight were scan nodes. Public metadata supplies
neither specialization constants nor typed resource/view links or physical ownership.

`describeGPUCommandGraph` consumes these public facts from standalone and nested scan nodes,
including without a recorder or an opted-in local factory graph:

| Public `preflight.nodes[].kernel` field | Record field |
| --- | --- |
| `entryPoint` | `node.entryPoint`, or the shared source's entry point when equal. Resolve the source by `node.source` in the returned `sources`; read `node.entryPoint ?? source?.entryPoint`. An absent result means the entry point is unknown. |
| `workgroupSize` | `node.workgroupSize`. |
| `bindings` | `node.declaredBindings`, preserving each binding's `name`, `type`, `group` and `location`. This layout is separate from resource-linked `node.bindings`. |
| `source` | A deduplicated WGSL source reference with content-derived `id`/`hash`. Its `entryPoint` and `moduleId` belong to the retained source reference from the first scheduled occurrence of that content ID; `moduleId` uses that occurrence's operation (or ID) and optional variant. `text` is retained only with `sourceText: true`. |

Use `node.op` for the individual node's operation. For example, two public nodes with identical
WGSL, variant `'portable'`, and operations `'First'` then `'Second'` share one source reference
whose `moduleId` is `'First/portable'`; their own `node.op` values remain `'First'` and `'Second'`.

Precedence is field by field for defined values: explicit `recordNode` annotation, then local
factory fact, then public fact. For example, the recorder-owner conflict fixture supplies public
`workgroupSize: [128, 1, 1]` and `entryPoint: 'publicEntry'`, local factory values `[64, 1, 1]`
and `'main'`, and an explicit annotation `{workgroupSize: [32, 1, 1]}`. The record keeps
`[32, 1, 1]` and inherits `'main'` from the factory. Public
`declaredBindings: [{name: 'publicOutput', type: 'storage', group: 0, location: 0}]` also survives
because neither higher-priority annotation sets that field; the factory's separate typed
`bindings` still names its `output` resource. This synthetic conflict demonstrates precedence;
the describer does not reconcile conflicting shader layouts.

Compile-time collection and record retention are independent. `'layout'` supplies layout without
a source reference; `'source'` supplies source text for hashing, but the describer's `sourceText`
option (defaulting to the recorder's setting, otherwise false) controls whether that text is
retained. `sourceText: true` cannot recover WGSL absent from a layout-only public report.
Ordinary compilation supplies no public kernel facts. Nodes without metadata retain their
ordinary graph facts without invented workgroup or binding layout. Public layout supplies no
specialization constants, typed resource/view links or physical ownership. Without recorder or
local kernel handles, observed resources retain `ownership: 'unknown'` and report
`bytesKind: 'lower-bound'` when a byte estimate is available. The four-row uint32 scan, for
example, records a 16-byte lower bound for each input/output buffer. Dimensions alone do not
give a byte estimate for an opaque external texture without a storage format: a sampled
320×180 external texture reports `bytes: 0, bytesKind: 'unavailable'`. That zero is a placeholder
for unavailable storage information, not a measured zero-byte allocation.

The host preflight checks in
`modules/analysis/test/computation-record/computation-preflight.node.spec.ts` cover the standalone
scan's entry point `'main'`, workgroup size `[256, 1, 1]`, and storage bindings `inputValues` /
`outputValues` at group 0, locations 0 / 1, plus source retention, absent metadata and annotation
precedence. The 513-row radix route retains those facts on its eight nested scan nodes while its
other sixteen nodes have no kernel metadata. These are host collection/recording checks, not GPU
execution or whole-primitive metadata coverage claims.

`GPUComputationRecorder` supplies domain facts the graph cannot infer: namespaces, formulas, `derivedFrom`, loop coordinates and datum bindings. Its API is exported from `@luma-contributors/analysis/computation-record`:

```ts
const graph = new GPUCommandGraph(device, {id: 'analysis-record'});
const recorder = new GPUComputationRecorder({sourceText: true});
recorder.recordKernels(graph); // before contributors build their kernels
// Build contributors on graph; optionally annotate individual nodes:
recorder.withSynchronousScope({namespace: 'round', loop: {id: 'relax', iteration: 0}}, () => {
  recorder.addNode(graph, node, createGPUKernelAnnotation(kernelFacts));
});
const compiled = graph.compile();
const description = describeGPUCommandGraph(compiled, {recorder, label: 'Analysis'});
```

`recordNode(node, annotation)` returns the node; `addNode(graph, node, annotation)` records and adds it. Namespaces join with `/`; the innermost `derivedFrom`, `loop`, `op` and `datum` win, and node annotations override scopes. Recording the same ID again replaces its entry: use `getNodeRecord` and merge explicitly when extending it. Without a recorder, `description` supplies the label and `workload.operation` supplies the operation.

`withSynchronousScope` rejects native async callbacks before invocation. Other callbacks returning a PromiseLike reject on return; their already-started
work cannot be cancelled. For asynchronous authoring, use the callback's separate scoped recorder
after every `await`, including for nested scopes:

```ts
await recorder.withAsyncScope({namespace: 'loaded'}, async scopedRecorder => {
  await Promise.resolve(); // For example, wait for host-side annotation inputs.
  scopedRecorder.recordNode(node, {op: 'Loaded analysis'});
});
// The shared record for node now has namespace 'loaded'; the parent scope is unchanged.
```

The callback recorder shares records, resources and kernel graph with its parent, while its
captured scope stack stays independent across concurrent asynchronous callbacks and exceptions.

The alternative `describeGPUCommandGraph(compiled, {graph, sourceText})` reads an opted-in graph without an explicit recorder. Aggregate transient allocation totals are attached as graph attributes when a kernel graph is supplied. Graph IDs must not contain `:`.

## Snapshots and timings

`GPUSnapshotTap` in `computation-record/gpu-snapshot-tap.ts` is an ordinary published copy-node producer. Add a tap immediately after the node it observes; its declared `copy-source` use orders the copy and keeps transient data alive. `afterNode` also declares an explicit dependency. Source buffers need `Buffer.COPY_SRC`.

Create `GPUSnapshotArena` from `computation-record/gpu-snapshot-arena.ts` before building taps,
with named slots, explicit element types/components and frame capacity. Encoding a tap reserves
the next frame even if its encoder is never submitted. Take `checkpoint()` before cancellable
encoding; if abandoning that encoder, call `discard(checkpoint)` to remove its unsubmitted
suffix. Permanently abandon every encoder whose reservations were discarded: never submit it
or encode into it again. Discard cannot split one encoder's reservations or remove a submitted
prefix, and checkpoints belong to one arena and become stale if their prefix is replaced.

Immediately before submission, protect the complete contiguous prefix with `markSubmitted()`.
When a later encoder already has reservations, pass the checkpoint taken at the end of the
earlier prefix: `markSubmitted(endCheckpoint)`. Omitting that boundary would protect the later
reservations too, preventing their cancellation. A failed submission leaves the marked prefix
protected; the arena cannot detect arbitrary device submissions for the caller.

This direct-copy example uses the same reservation lifecycle as graph taps. The submitted
observations are `[13,15]`; the abandoned observation `14` must leave no unwritten frame
between them:

```ts
import {Buffer, type CommandEncoder, type Device} from '@luma.gl/core';
import {GPUSnapshotArena} from '@luma-contributors/analysis/computation-record';

async function captureHistory(device: Device) {
  let source: Buffer | undefined;
  let arena: GPUSnapshotArena | undefined;
  const encoders: CommandEncoder[] = [];
  try {
    source = device.createBuffer({
      data: Uint32Array.of(13, 14, 15), usage: Buffer.COPY_SRC
    });
    arena = new GPUSnapshotArena(device, {
      slots: [{name: 'state', byteLength: 4, elementType: 'u32'}], capacity: 2
    });
    const firstEncoder = device.createCommandEncoder();
    encoders.push(firstEncoder);
    const abandonedEncoder = device.createCommandEncoder();
    encoders.push(abandonedEncoder);
    const replacementEncoder = device.createCommandEncoder();
    encoders.push(replacementEncoder);
    arena.encodeSnapshot(firstEncoder, {state: {buffer: source, byteOffset: 0}});
    const firstCommands = firstEncoder.finish();
    // End of the complete first prefix, before cancellable encoding on the later encoder.
    const endCheckpoint = arena.checkpoint();
    arena.encodeSnapshot(abandonedEncoder, {state: {buffer: source, byteOffset: 4}});
    arena.markSubmitted(endCheckpoint);
    device.submit(firstCommands);
    arena.discard(endCheckpoint); // Never finish or submit abandonedEncoder.
    arena.encodeSnapshot(replacementEncoder, {state: {buffer: source, byteOffset: 8}});
    const replacementCommands = replacementEncoder.finish();
    arena.markSubmitted();
    device.submit(replacementCommands);
    const captures = await arena.read();
    return captures.map(capture => Array.from(capture.values)); // [[13], [15]]
  } finally {
    for (const encoder of encoders.reverse()) encoder.destroy();
    arena?.destroy();
    source?.destroy();
  }
}
```

Submit all retained encoders before `read()`. Its first call permanently seals the frame set:
no more reservations, checkpoints, submission marks or discards are allowed. It shares an
in-flight mapping among concurrent readers and caches successful bytes; readers receive
independent value arrays. A failed map permits retry by calling `read()` again, without
reopening the sealed frame set. Reading does not discard an unsubmitted reservation for you.

The author chooses taps at publications, after individual nodes or at the end; there is no
automatic progressive capture. Taps end compute-pass coalescing, so record their count in
`GPUComputationEncodingFlags.taps` and set `coalesced: false`.

`submitGPUCommandGraphWithTimings` in `computation-record/submit-timed-command-graph.ts` uses the public command-encoder `timeProfilingQuerySet` when `timestamp-query` is available. Its `readTimings()` frees the query set and returns the published report, or `undefined` without timestamps. Timestamped compute nodes occupy separate passes, which is also an observer effect.

`createGPUComputationMetrics` in `computation-record/computation-metrics.ts` accepts published
millisecond endpoints and independently measured durations. Its optional
`GPUComputationNodeMetrics.gpuTimeMs` preserves finite, nonnegative `gpuTimeMilliseconds` in
milliseconds, including measured zero. Unavailable durations stay absent; unavailable endpoints
stay absent even when a duration exists. `gpuTimeMs` always uses milliseconds independently of
the endpoint `timebase`, which defaults to `'ns'` without millisecond starts. For example:

```ts
const buildMetrics = (nodes: GPUComputationTimingInput['nodes'], timestamped = false) =>
  createGPUComputationMetrics({
    runId: 'run',
    graph: 'graph',
    encoding: {coalesced: false, timestamped, taps: 0, lensTwin: false},
    timing: {nodes}
  });

const durations = buildMetrics([
  {id: 'duration', cpuEncodeTimeMilliseconds: 1, gpuTimeMilliseconds: 2},
  {id: 'zero', cpuEncodeTimeMilliseconds: 1, gpuTimeMilliseconds: 0},
  {id: 'unavailable', cpuEncodeTimeMilliseconds: 1}
]);
// durations.nodes:
// duration:    {cpuEncodeMs: 1, gpuTimeMs: 2}
// zero:        {cpuEncodeMs: 1, gpuTimeMs: 0}
// unavailable: {cpuEncodeMs: 1}
// All three omit gpuBegin/gpuEnd; durations.timebase is 'ns'.

const endpoints = buildMetrics([
  {id: 'endpoints', cpuEncodeTimeMilliseconds: 1, gpuTimeMilliseconds: 2,
    gpuStartTimeMilliseconds: 10, gpuEndTimeMilliseconds: 12}
], true); // Illustrated timestamp-query run; encoding.timestamped is true.
// endpoints has timebase: 'ms'; timestamps are relative to the earliest start:
// endpoints.nodes['endpoints']:
// {cpuEncodeMs: 1, gpuTimeMs: 2, gpuBegin: 0, gpuEnd: 2}
```

`GPUComputationTimingInput` and `createGPUComputationMetrics` are exported from
`@luma-contributors/analysis/computation-record`. For an actual timed submission, pass its
`timestamped` flag rather than inferring it from the timing values. These literal cases are covered by the
`ROAD-111` case in `modules/analysis/test/computation-record/computation-record.node.spec.ts`.
Endpoint-only reports still produce relative endpoints without fabricating a duration.
`getGPUCommandGraphPassIndices` infers membership from encoding statistics and checks it against
`computePassCount`. `describeGPUCommandGraphEncoding` uses one pass per compute node with
`attrs.passGrouping = 'unknown'` if that inference disagrees.

## Missing information and upstream requirements

| Gap | Current limit | What luma.gl would need |
| --- | --- | --- |
| Published primitive shaders | The describer consumes public entry point, workgroup size, binding name/type/group/location as `declaredBindings`, and optional source beneath local facts/explicit annotations. `kernelMetadata: 'layout'` collects layout; `'source'` also collects WGSL, retained only with local `sourceText`. Sort-owned histogram/scatter and reduction-owned kernels lack facts; radix-sort scan nodes supply them. Inspect each route's compute nodes rather than assigning coverage to an entire primitive. | Extend missing node coverage, specialization constants and typed resource/view links. Layout alone supplies no physical ownership and cannot replace the local per-graph kernel table or domain annotations. |
| Physical transient aliasing | Per-resource `allocation`, `allocationBytes`, retention and observer-usage details are unavailable; only aggregate allocation/reuse totals are public. | A public logical-to-physical allocation plan. |
| Loop and formula semantics | Shared dispatch gates and unrolled rounds do not supply loop IDs or iterations. Packs must annotate loops, formulas, datum and `derivedFrom`. | Public semantic metadata where the compiler knows it; analysis-specific formulas still belong to analysis. |
| Intent and lowered forms | The schema permits `intent` and `lowered`, but command-graph inspection cannot reconstruct them. Lenses must author plain `GPUComputationGraph` data where needed. | Public query expressions and lowering/loop metadata. Inspector summaries alone are insufficient. |
| Exact edges and pass membership | Whole-resource hazard edges and inferred pass indices can be conservative or unknown. | Published dependency edges and node-to-pass membership. |

`graph.recording`, `graph.withRecordScope`, `node.annotation`, `compiled.describe()` and `captureGPUCommandGraph` are not part of this implementation. They exist, with specs, in the luma.gl fork at `e0ed33791` (`modules/graph/src/gpu-core/gpu-command-graph-describe.ts`, `gpu-command-node-record.ts`, and `GPUCommandGraph.planResources()`, whose spec checks that a dry-run plan equals the compiled allocation); see [research/fork-comparison-2026-11-13](./research/fork-comparison-2026-11-13/README.md). That code is the reference for the upstream request. The capability register and upstream requests are in [upstream-workarounds.md](./upstream-workarounds.md) and [upstream-proposals.md](./upstream-proposals.md).
