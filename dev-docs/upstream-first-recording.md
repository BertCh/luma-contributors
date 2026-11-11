# Computation records on published luma.gl

Status: implemented in `modules/analysis/src/computation-record`. Read this before touching any
code that used `graph.recording`, `graph.withRecordScope`, `compiled.describe()`,
`node.annotation`, `captureGPUCommandGraph` or `getPhysicalBuffer`.

## Decision

Published `@luma.gl/graph@10.0.0-alpha.10` has no recording hooks. Its public surface for
inspection is:

| Source | What it gives |
| --- | --- |
| `CompiledGPUCommandGraph.id`, `.stats` | `stats.nodeOrder` (topological order), buffer and texture byte totals, reuse statistics |
| `CompiledGPUCommandGraph.preflight` | per node: `id`, `type`, `description`, `operation`, `variant`, workload bounds, `resources[]` (id, kind, usage, byteLength, byteOffset, format, length, size), `condition`, `publication` |
| `GPUCommandGraphEncoding.stats` (from `compiled.encode(...)`) | per node CPU encode time, conditional outcome, pass counters |
| `GPUCommandGraphEncoding.readTimings()` | per node GPU start, end and duration in milliseconds |
| Node props | `id`, `description`, `resources`, `dependsOn`, `workload`, `publication` |
| Handles | `GraphBufferHandle`, `GraphTextureHandle` public fields: `id`, `byteLength`, `transient`, `format`, `width`, `height` |
| `compiled.getExecutionPlan(...)` / `createExecution(...)` | planned steps and publications |

We chose a hybrid of the two options, with no luma.gl hook and no private field access:

1. Structure, workload, resource usage, conditions, publications and hazard edges are built from
   what the compiled graph publishes (`preflight`, `stats`). Zero cooperation is needed from the
   code that built the graph, so every graph can be described.
2. Domain facts luma.gl does not know (source text, workgroup size, bindings with element types,
   key-space `datum`, loop coordinates, formulas, namespaces, `derivedFrom`) come from an explicit
   `GPUComputationRecorder` that analysis code calls while it builds nodes. Recording is now an
   analysis concern, not a graph feature. A graph built without a recorder is the production path
   and carries nothing extra.
3. Snapshot capture no longer drives the graph step by step. A `GPUSnapshotTap` is an ordinary
   published copy node that copies views into a `GPUSnapshotArena`. The graph's own hazard
   inference orders it, and because it declares a `copy-source` use it keeps a transient alive, so
   `getPhysicalBuffer` and `recording.retain` are not needed.

## API (`@luma-contributors/analysis/computation-record`)

```ts
new GPUComputationRecorder({sourceText?: boolean})
recorder.withScope(scope, callback)                  // replaces graph.withRecordScope
recorder.recordNode(node, annotation?) -> node       // replaces node.annotation = ...
recorder.addNode(graph, node, annotation?)           // recordNode + graph.add(node)
recorder.getNodeRecord(nodeId) / getResource(id)

describeGPUCommandGraph(compiled, {recorder?, graphId?, label?, domain?})
  -> {graph, sources}                                // replaces compiled.describe()
describeGPUCommandGraphEncoding(encoding.stats, {compiledGraphId})   // encoded form
getGPUCommandGraphPassIndices(encoding.stats)        // node id -> pass index, or undefined

createGPUKernelAnnotation({op, moduleId, source, workgroupSize, workgroups, bindings, ...})
new GPUSnapshotTap({id, arena, views, afterNode?, label?, iteration?, recorder?})
new GPUSnapshotArena(device, {slots, capacity})       // unchanged
createGPUComputationMetrics({timing, passIndices?, ...}) // accepts published ms timings
```

The recorder is per graph. Create it only when a caller asks for a record; pass it down as an
optional prop, exactly like the old `recording` option.

### Kernel facts through `createWGSLKernelNode` (LN-REC, 2026-11-10)

Decision: no `recorder?` prop on the ~430 contributor classes. Almost every analysis kernel is built
by `createWGSLKernelNode`, so that helper is the one capture point:

- `recordWGSLKernelNodes(graph)` (analysis-core) opts one graph in; `recorder.recordKernels(graph)`
  does the same and remembers the graph. Lens graphs (`LensCommandGraph`, the network lens graphs)
  opt in in their constructors.
- While a graph is opted in, `createWGSLKernelNode` stores `{props, source, workgroupSize, layout}`
  under the node id in a `WeakMap` keyed by the graph: only references the node's closures already
  hold. A graph that is not opted in pays one `WeakMap` lookup per kernel and keeps nothing.
- Keyed by id, not by node object: the compiled graph publishes node ids only, and `graph.add` /
  `addGPUCommandNode` copy node objects.
- `describeGPUCommandGraph(compiled, {graph | recorder, sourceText})` turns each entry into the
  annotation `createGPUKernelAnnotation` would build (op, source with hash and `bodyHash`,
  workgroup size, dispatch, per-node `constants`, bindings) and states the real ownership of every
  bound resource. Explicit `recordNode` annotations win field by field.
- Timings: `submitGPUCommandGraphWithTimings` sets the public `timeProfilingQuerySet` when the
  device has `timestamp-query`; lens harnesses pass the report to `createGPUComputationMetrics`.
  No timestamps means an empty node table, never zeros.
- Still pack-level, because no shared helper knows them: formulas, `derivedFrom`, loop coordinates,
  datum bindings. Dispatch gates and unrolled rounds are built by shared helpers but do not carry the
  loop id or iteration; a cheap generic source would be an optional `loop` prop on
  `createWGSLDispatchGate`, adopted family by family (wave 2 or later).
- Not covered: nodes of published `@luma.gl/graph` primitives (`GPUScan`, `GPUSort`, `GPUReduction`,
  `GPUGroupAggregation`, `GPURasterNeighborhood`, `GPURasterTextureToBuffer`). The luma.gl hook that
  would cover them is a public per-node shader description (source, entry point, workgroup size,
  bindings) on `compiled.preflight.nodes[]`; per-resource transient aliasing (`allocation`) also
  needs luma.gl to publish its allocation plan.

## Before and after

### Marking a graph as recording

Before:

```ts
const graph = new GPUCommandGraph(device, {id: 'scan-lens', recording: {sourceText: true}});
```

After:

```ts
const graph = new GPUCommandGraph(device, {id: 'scan-lens'});
const recorder = props.record ? new GPUComputationRecorder({sourceText: true}) : undefined;
```

Operators receive `recorder?: GPUComputationRecorder` in their props instead of reading
`graph.recording`.

### Scopes

Before:

```ts
graph.withRecordScope({namespace: 'level-0', op: 'GPUScan'}, () => graph.add(node));
```

After (a scope only affects nodes passed through the recorder inside it):

```ts
if (recorder) {
  recorder.withScope({namespace: 'level-0', op: 'GPUScan'}, () => recorder.addNode(graph, node));
} else {
  graph.add(node);
}
```

Scope rules are unchanged: namespaces join with `/`, the innermost `derivedFrom`, `loop`, `op` and
`datum` win, and values on the node's own annotation win over scopes. An iteration scope:

```ts
recorder?.withScope(
  {loop: {id: 'relax', iteration, phase: gated ? 'gated' : 'unconditional', maximumIterations}},
  () => addRound(graph, recorder, iteration)
);
```

### Node annotations

Before:

```ts
const node = createGPUComputeCommandNode({id, resources, workload, compile});
if (graph.recording) {
  node.annotation = {op: 'GPUTransferMask', attrs: {layout}, source: {...}, dispatch: {...}, bindings: [...]};
}
```

After:

```ts
const node = createGPUComputeCommandNode({id, resources, workload, compile});
return recorder
  ? recorder.recordNode(node, createGPUKernelAnnotation({
      op: 'GPUTransferMask',
      moduleId: 'GPUTransferMask/scatter',
      source: shaderSource,                       // hashed; text dropped unless sourceText
      workgroupSize: [256, 1, 1],
      workgroups: [dispatch.x, dispatch.y, dispatch.z],
      invocationCount,
      attrs: {layout},
      bindings: entries.map(([name, view]) => ({name, view, access: 'read'}))
    }))
  : node;
```

Rules of thumb: `recordNode` returns the node, so it chains into `graph.add`. Pass `elementType`
on a binding when it differs from the format (`atomic<u32>`). Loop-role style annotations that
extended an existing annotation become a second call: `recorder.recordNode(node, {...previous, attrs})`
replaces the earlier entry for that id, so merge the previous annotation yourself
(`recorder.getNodeRecord(node.id)`).

Every node already carries `description`, `workload.operation` and `workload.variant`; set them
when you create the node, because the describer uses them when no recorder entry exists
(`description` becomes the label, `operation` becomes `op`).

### Describing a compiled graph

Before:

```ts
const compiled = graph.compile();
const description = compiled.describe({label: 'Scan', domain: 'gpu-core'});
```

After:

```ts
const compiled = graph.compile();
const description = describeGPUCommandGraph(compiled, {recorder, label: 'Scan', domain: 'gpu-core'});
const record = createGPUComputationRecord({..., graphs: [description]});
```

The result shape `{graph, sources}` is unchanged, so `createGPUComputationRecord` accepts it as before.

### Capturing intermediate values

Before:

```ts
const graph = new GPUCommandGraph(device, {recording: {retain: ['state']}});
...
const capture = await captureGPUCommandGraph(compiled, {
  encodeOptions: {parameters: undefined}, views: {state}, at: 'publication', runId
});
```

After: build the arena first, add a tap after each node to observe (or one tap last for the
final value), run the compiled graph normally, then read the arena once.

```ts
const arena = new GPUSnapshotArena(device, {
  id: 'lens-snapshots',
  slots: [{name: 'state', byteLength: 4, elementType: 'u32'}],
  capacity: iterationCount                    // one frame per tap encoding
});
graph.addComputePass(iterationNode);
graph.add(new GPUSnapshotTap({id: `observe-${i}`, arena, views: {state}, afterNode: iterationNode.id, recorder}));
...
const compiled = graph.compile();
const commandEncoder = device.createCommandEncoder();
compiled.encode(commandEncoder, {parameters: undefined});
device.submit(commandEncoder.finish());
const captures = await arena.read({graph: description.graph.id, resources: {state: 'state'}, runId});
```

Mapping of old `at` values: `publication` and `step` mean one tap after each publishing or each
node (the author chooses where), `end` means one tap added last. A tap must come right after the
node it observes in authoring order, because hazards follow authoring order. Its buffers need
`Buffer.COPY_SRC`. Slot `elementType` and `components` were previously inferred from the view
format; use `getGPUElementType(view.format)` and set them explicitly in the arena slots.

`observer effect`: taps are copy nodes, so they end compute-pass coalescing. Set
`GPUComputationEncodingFlags.taps` to the number of taps and `coalesced: false`.

### Timings and metrics

Before: `timing.nodes[].gpuBeginNanoseconds`, `passIndex` (forked report).

After: pass the published `await encoding.readTimings()` directly. It reports milliseconds
(`gpuStartTimeMilliseconds`, `gpuEndTimeMilliseconds`) and produces `timebase: 'ms'`. Pass indices
are not published; use `passIndices: getGPUCommandGraphPassIndices(encoding.stats)`.

```ts
const encoding = compiled.encode(commandEncoder, {parameters: undefined});
device.submit(commandEncoder.finish());
const timing = await encoding.readTimings();
const metrics = createGPUComputationMetrics({
  runId, graph: description.graph.id, timing, encoding: flags,
  passIndices: getGPUCommandGraphPassIndices(encoding.stats)
});
```

## Dropped or changed behavior

- `graph.recording`, `GPUCommandGraphRecordingOptions`, `retain`, `withRecordScope`, `node.annotation`,
  `compiled.describe()`: removed. Replacements above.
- `captureGPUCommandGraph` and progressive stepped execution for capture: replaced by
  `GPUSnapshotTap`. No stepping through `createExecution`, so capture no longer splits queue
  submissions per step. `publication` ids are still on nodes and appear in the description.
- Physical buffer aliasing detail: `allocation`, `allocationBytes`, `retained`, `observerUsage`
  on described resources are gone (the compiled graph does not publish its transient allocation plan).
  Ownership (`transient` or `imported`) is known only for resources whose nodes went through a
  recorder; otherwise `imported` is stated. `stats` aggregates (reuse bytes) can be attached by the
  caller as graph `attrs` if a lens needs them.
- Edges are inferred from declared usage in scheduled order at whole buffer or texture granularity;
  texture mip and layer overlap is not modelled, so some `raw`, `war`, `waw` edges can be extra.
  `explicit` (`dependsOn`) edges exist only for recorded nodes.
- Encoded form: node to pass membership is inferred from the published coalescing rule and
  verified against `computePassCount`; on disagreement each compute node gets its own pass and the
  graph carries `attrs.passGrouping = 'unknown'`.
- Intent and lowered graph descriptions (`describeGPUGraphCompilation`,
  `describeGPUDataFrameQueryIntent` and friends, `gpu-graph-computation-record`) are deleted: they
  read loop, conditional and shader-plan metadata only the forked `GPUGraph` compiler produced.
  `GPUComputationGraph` forms `intent` and `lowered` remain in the schema; lenses that need them
  must author them as plain data (`GPUComputationGraph` literals) or drop them. The published
  `inspectGPUGraphCompilation` and `GPUCommandGraphInspector` give summaries, not record graphs.
- `getGPUComputeKernelAnnotation(graph, describe)` becomes `createGPUKernelAnnotation(props)` with no
  graph argument; the caller decides whether to call it.
- `observationTap` annotation attr remains and is set by `GPUSnapshotTap` when given a recorder.
- Graph ids must not contain `:` (unchanged).
