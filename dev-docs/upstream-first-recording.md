# Computation records on published luma.gl

`modules/analysis/src/computation-record` builds records from the public `GPUCommandGraph` API of published `@luma.gl/graph`. The [alpha package contract](./release-package.md) records the installed integration target and accepted peer range separately. No private graph fields or recording hooks are required.

## Public graph data

| Source | Recorded information |
| --- | --- |
| `CompiledGPUCommandGraph.id`, `.stats` | Graph ID, topological `nodeOrder`, aggregate transient buffer/texture allocation and reuse totals. |
| `.preflight.nodes` | Node ID, type, description, operation, variant, workload bounds, declared resources, conditions and publications. Resource summaries include kind, usage, byte range, format, length and size. |
| `GPUCommandGraphEncoding.stats` | Per-node CPU encode time, conditional outcome and pass counters. |
| `GPUCommandGraphEncoding.readTimings()` | GPU start, end and duration in milliseconds when timestamps are available. |
| Public node and resource handles | `dependsOn` and resource ownership for explicitly recorded nodes; buffer/texture dimensions and transient state. |

`describeGPUCommandGraph` in `computation-record/describe-command-graph.ts` walks `stats.nodeOrder` and merges the matching preflight nodes with optional analysis annotations. It returns `{graph, sources}` for `createGPUComputationRecord` in `computation-record/create-computation-record.ts`.

Read/write edges are inferred from declared usage in scheduled order at whole-buffer or whole-texture granularity. Texture mip and layer overlap is not modelled, so `raw`, `war` and `waw` edges can be conservative. Explicit `dependsOn` edges are available only for nodes passed through the recorder. Resource ownership comes from recorder handles or kernel bindings; otherwise the description conservatively says `imported`.

## Kernel and domain facts (LN-REC)

`recordWGSLKernelNodes(graph)` in `analysis-core/kernels/wgsl-kernel-nodes.ts` opts a graph in before nodes are built. `GPUComputationRecorder.recordKernels(graph)` in `computation-record/computation-recorder.ts` does the same and retains the graph reference. `createWGSLKernelNode` then stores props, generated source, workgroup size and bounded layout in a graph-keyed `WeakMap`, indexed by node ID. Without opt-in it performs one lookup and stores nothing.

`createWGSLKernelNodeAnnotation` in `computation-record/describe-command-graph.ts` produces operation, source hash and body hash, entry point, workgroup size, dispatch, constants and typed bindings from that table. Full source text is included only when `sourceText` is enabled. Explicit `recordNode` annotations take precedence field by field.

`GPUComputationRecorder` supplies domain facts the graph cannot infer: namespaces, formulas, `derivedFrom`, loop coordinates and datum bindings. Its API is exported from `@luma-contributors/analysis/computation-record`:

```ts
const graph = new GPUCommandGraph(device, {id: 'analysis-record'});
const recorder = new GPUComputationRecorder({sourceText: true});
recorder.recordKernels(graph); // before contributors build their kernels
// Build contributors on graph; optionally annotate individual nodes:
recorder.withScope({namespace: 'round', loop: {id: 'relax', iteration: 0}}, () => {
  recorder.addNode(graph, node, createGPUKernelAnnotation(kernelFacts));
});
const compiled = graph.compile();
const description = describeGPUCommandGraph(compiled, {recorder, label: 'Analysis'});
```

`recordNode(node, annotation)` returns the node; `addNode(graph, node, annotation)` records and adds it. Namespaces join with `/`; the innermost `derivedFrom`, `loop`, `op` and `datum` win, and node annotations override scopes. Recording the same ID again replaces its entry: use `getNodeRecord` and merge explicitly when extending it. Without a recorder, `description` supplies the label and `workload.operation` supplies the operation.

The alternative `describeGPUCommandGraph(compiled, {graph, sourceText})` reads an opted-in graph without an explicit recorder. Aggregate transient allocation totals are attached as graph attributes when a kernel graph is supplied. Graph IDs must not contain `:`.

## Snapshots and timings

`GPUSnapshotTap` in `computation-record/gpu-snapshot-tap.ts` is an ordinary published copy-node producer. Add a tap immediately after the node it observes; its declared `copy-source` use orders the copy and keeps transient data alive. `afterNode` also declares an explicit dependency. Source buffers need `Buffer.COPY_SRC`.

Create `GPUSnapshotArena` from `computation-record/gpu-snapshot-arena.ts` before building taps, with named slots, explicit element types/components and frame capacity. Each encoded tap copies its views into the next arena frame. Submit the graph normally, then call `arena.read()`. The author chooses taps at publications, after individual nodes or at the end; there is no automatic progressive capture. Taps end compute-pass coalescing, so record their count in `GPUComputationEncodingFlags.taps` and set `coalesced: false`.

`submitGPUCommandGraphWithTimings` in `computation-record/submit-timed-command-graph.ts` uses the public command-encoder `timeProfilingQuerySet` when `timestamp-query` is available. Its `readTimings()` frees the query set and returns the published report, or `undefined` without timestamps. Timestamped compute nodes occupy separate passes, which is also an observer effect.

`createGPUComputationMetrics` in `computation-record/computation-metrics.ts` accepts the published millisecond timings (`timebase: 'ms'`); unavailable timestamps produce no node timings, never fabricated zeros. `getGPUCommandGraphPassIndices` infers membership from encoding statistics and checks it against `computePassCount`. `describeGPUCommandGraphEncoding` uses one pass per compute node with `attrs.passGrouping = 'unknown'` if that inference disagrees.

## Missing information and upstream requirements

| Gap | Current limit | What luma.gl would need |
| --- | --- | --- |
| Published primitive shaders | Nodes from `GPUScan`, `GPUSort`, `GPUReduction`, `GPUGroupAggregation` and raster primitives do not pass through the local kernel factory, so their shader facts are absent. | Per-node source, entry point, workgroup size, constants and typed bindings on `preflight.nodes[]`, or a public shader-description hook. This would replace the local per-graph kernel table. |
| Physical transient aliasing | Per-resource `allocation`, `allocationBytes`, retention and observer-usage details are unavailable; only aggregate allocation/reuse totals are public. | A public logical-to-physical allocation plan. |
| Loop and formula semantics | Shared dispatch gates and unrolled rounds do not supply loop IDs or iterations. Packs must annotate loops, formulas, datum and `derivedFrom`. | Public semantic metadata where the compiler knows it; analysis-specific formulas still belong to analysis. |
| Intent and lowered forms | The schema permits `intent` and `lowered`, but command-graph inspection cannot reconstruct them. Lenses must author plain `GPUComputationGraph` data where needed. | Public query expressions and lowering/loop metadata. Inspector summaries alone are insufficient. |
| Exact edges and pass membership | Whole-resource hazard edges and inferred pass indices can be conservative or unknown. | Published dependency edges and node-to-pass membership. |

`graph.recording`, `graph.withRecordScope`, `node.annotation`, `compiled.describe()` and `captureGPUCommandGraph` are not part of this implementation. They exist, with specs, in the luma.gl fork at `e0ed33791` (`modules/graph/src/gpu-core/gpu-command-graph-describe.ts`, `gpu-command-node-record.ts`, and `GPUCommandGraph.planResources()`, whose spec checks that a dry-run plan equals the compiled allocation); see [research/fork-comparison-2026-11-13](./research/fork-comparison-2026-11-13/README.md). That code is the reference for the upstream request. The capability register and upstream requests are in [upstream-workarounds.md](./upstream-workarounds.md) and [upstream-proposals.md](./upstream-proposals.md).
