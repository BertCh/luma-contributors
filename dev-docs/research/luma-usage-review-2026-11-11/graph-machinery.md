# Stream `graph-machinery`: build, compile, run, iterate, inspect, record (alpha.13)

Published paths are under `node_modules/@luma.gl/`; local paths under `modules/` unless stated.
Helper scripts and raw lists: `review/graph-machinery/` (`loops.txt`, `classify.mjs`,
`published-in-loops.mjs`).

## 1. Coverage

Read in full (published): `graph/dist/gpu-core/index.d.ts`, `gpu-graph/index.d.ts` (to check what is
public), `gpu-command-graph-types.d.ts:360-700` (node condition, preflight, stats, timing),
`gpu-command-graph.d.ts` (builder, compiled graph, execution), `gpu-command-graph.js:175-230, 640-690,
1130-1270` (GPU-indirect condition proxy, node validation, encode loop), `gpu-control-flow-operation.d.ts`,
`gpu-graph-compiler.js:380-600` (conditional and loop lowering, `decorateComputeNode`, `emitProducer`),
`gpu-graph-lowering.d.ts`, `gpu-scalar-dispatch-gate.js` (whole), `gpu-command-dispatch-metadata.{d.ts,js}`,
`gpu-command-node.{d.ts,js}`, `gpu-operation.d.ts`, `gpu-shader-pass-operation.d.ts`,
`gpu-semantic-aggregation-operation.d.ts`, `gpu-incremental-execution.d.ts`, `gpu-readback-ring.d.ts`,
`gpu-command-graph-inspector.d.ts`, `gpu-graph-inspector.d.ts`, `gpu-command-graph-autotuner.d.ts` and the
autotuner use in `gpu-scan.js:497-516`, `gpu-command-graph-budget-controller.d.ts`,
`gpu-conjugate-gradient-program.d.ts`, `gpu-dispatch-utils.js`, `gpu-resource-preview.d.ts`,
`gpu-command-graph-compiler.js:85-125` (async compile), `graph/dist/execution/compute/kernel.{d.ts,js}`;
in `@luma.gl/core`: `webgpu/adapter/resources/webgpu-compute-pass.js`, `webgpu-compute-pipeline.js`,
`api/factories/bind-group-factory.js`, `api/factories/pipeline-factory.js:130-195, 289-294`.

Read in full (local): `analysis-core/kernels/wgsl-kernel-nodes.ts`, `analysis-core/operations/gpu-contributor-operation.ts`,
`computation-record/describe-command-graph.ts`, `submit-timed-command-graph.ts`, `gpu-snapshot-tap.ts`,
`raster-analysis/cost-distance/raster-relaxation.ts:100-433`; in part: `gpu-contributor-utils.ts:395-512`,
`gpu-network-centrality.ts:425-760`, `network-centrality-passes.ts` (sigma round), `sparse-resistance-solver.ts`
(header, loop), `gpu-pycnophylactic.ts:485-800`, `gpu-inequality-operation.ts`,
`kepler/src/session/metered-command-graph.ts`, `kepler/src/analysis/point-weights-streamed.ts:60-130`,
`kepler/src/analysis/catalog-gpu-session.ts:85-100`, `kepler/src/exploration/gpu-column-statistics.ts:160-185`,
`examples/deck/spatial-analysis-shared/vector-timing.ts:90-152`, `analysis-effect.ts:85-115`.

Not done: no GPU run, no spec run (all findings are by reading). Not read: `computation-recorder.ts`,
`create-kernel-annotation.ts`, `gpu-snapshot-arena.ts`, the exporters; `analysis-core/graph-support/**`
(Arrow normalisation, outside this stream's execution question); `GPUTextureHistory`,
`GPUCommandGraphInspector.js` and `GPUIncrementalExecution.js` implementations (declarations only);
`lenses/src/computation-lenses` beyond grep for submits, compiles and scheduling; examples and website
beyond the timing grep. The loop counts in section 2 are from a brace-matching script, not a reading of
all 49 files.

Counts used below (all `git grep`):
- `createWGSLKernelNode`: 3,146 occurrences in 590 files under `modules/*/src`.
- Unrolled iteration loops: 64 `for` loops bounded by an iteration/round/sweep/pass/step variable in
  55 files of `modules/analysis/src`; 49 files have such a loop whose body builds command nodes; in 18
  of them the loop body itself attaches a GPU gate (others gate inside helpers, e.g. reachability's
  frontier nodes, or are log-step passes needing no gate).
- `createWGSLDispatchGate` callers: 33 files (plus `analysis-core/index.ts`); `createRasterIterationGateNode`
  family: 5 files; `createWGSLActiveCountDispatch`: 8 files; files writing `mode: 'indirect'` by hand: 5.
- Typical unrolled loop size: raster relaxation `1 + N + ceil(N/4)` nodes for N iterations
  (`raster-relaxation.ts:364`); sparse Jacobi-PCG 5 nodes per iteration, up to 4,096 iterations = 20,480
  (`sparse-resistance-solver.ts:29-32,132`); centrality per batch `R + 2 + ... + 2R + 3 + ...` with R
  default 64, so about 200 per batch, capped at 32,768 in total (`gpu-network-centrality.ts:433-450`).
- `GPUContributorOperation` subclasses or instances: 125 files in `modules/*/src`; `new GPUGraph(` in
  package sources: 0 (only `modules/analysis/test/utils/mixed-execution-test-utils.ts:80`).
- `device.submit` in `modules/analysis/src`: 2 (the timed submit helper and the test utilities). No
  analysis contributor re-submits in a JavaScript loop.
- `performance.now()` in `modules/*/src`: 0.

## 2. Findings

### GM-01 Bind-group "identity cache" in `createWGSLKernelNode` caches nothing on the GPU; the comments and register row say it does
- Class: `stale-claim` (with a `misuse` lead, see GM-02).
- Local: `analysis-core/kernels/wgsl-kernel-nodes.ts:276-278` ("the bind group is rebuilt only when a
  resolved buffer or texture view changes"), `:475-506` (identity cache), `:859-869` (comment: "lets the
  pipeline keep its cached bind group"), `createChangedBindingResolver` `:878`, `createCachedBindingResolver`
  `:932`. Register: `dev-docs/upstream-workarounds.md:48`.
- Published: `graph/dist/execution/compute/kernel.js:81-100` (`dispatch` calls `setPipeline` then
  `computePass.setBindings(bindings)`); `core/dist/webgpu/adapter/resources/webgpu-compute-pass.js:57-66`
  (`setBindings` calls `getBindGroups(pipeline, bindings)` with no cache keys);
  `core/dist/api/factories/bind-group-factory.js:28-41` (no key: `_createBindGroupWebGPU` every call).
- Evidence: read both.
- What is true: every `Kernel.dispatch(computePass, {bindings})` creates a new `GPUBindGroup`, whatever
  the identity of the binding objects. The local cache only saves the JavaScript allocation of
  `getViewBinding` records. Identity matters only on the `ComputePipeline.setBindings` path
  (`webgpu-compute-pipeline.js:62-88`), which `Kernel` does not use. Published primitives behave the same
  (no file in `graph/dist` passes cache keys). `createChangedBindingResolver` and
  `createCachedBindingResolver` have no caller in any `src` (only `test/utils/wgsl-kernel-nodes.node.spec.ts`
  and `test/geospatial/gpu-point-spatial-query-bindings.node.spec.ts`) and are not exported.
- Change: correct the two comments; delete the two unused resolvers and their two specs (or keep them
  only if GM-02 is adopted); rewrite register row 48: "no published or local path reuses compute bind
  groups; `Kernel.dispatch` has no cache-key option at alpha.13". 2 functions, 2 comments, 2 specs.
- Verify: no behaviour change; `yarn typecheck`, the two node specs removed with the helpers.
- Risk: none (documentation and dead code).

### GM-02 Bind groups can be reused today through public `kernel.pipeline.setBindings` (lead)
- Class: `unused-feature` (lead: needs a measurement before adoption).
- Local: `wgsl-kernel-nodes.ts:501-506`; also the 12 hand-written `Kernel.dispatch` sites in
  `computation-record/gpu-transfer-mask.ts:160` and `kepler/src/**` (11).
- Published: `Kernel.pipeline` (public field, `kernel.d.ts`), `ComputePipeline.setBindings`
  (`core/dist/api/adapter/resources/compute-pipeline.d.ts:32`), `WebGPUComputePass.setPipeline`
  (`webgpu-compute-pass.js:42-52`, applies the pipeline's own bindings with its keyed cache),
  `webgpu-compute-pipeline.js:62-79` (key reset only when a binding object changes).
- Evidence: read both.
- Change: in `encode`, call `kernel.pipeline.setBindings(resolvedBindings)` when the resolved resources
  change and `kernel.dispatch(computePass, {x, y, z})` without `bindings`; an omitted group then keeps the
  bind group set by `setPipeline`. One site in `createWGSLKernelNode` covers about 3,100 call sites.
- Cost: pipelines are shared through `PipelineFactory` by source, layout and constants
  (`pipeline-factory.js:289-294`), and this repository shares them on purpose (override constants,
  rule 12). Two nodes on one pipeline with different bindings reset each other's key, so unrolled loop
  rounds (the largest graphs) gain nothing; one-off kernels gain one `createBindGroup` per encode. The
  route also relies on `ComputePipeline.setBindings`, which carries a `@todo` in its documentation.
- Verify: a headless spec counting `device._createBindGroupWebGPU` calls (spy) over two encodes of a
  graph; bit-identical outputs are expected.
- Risk: results unchanged; a missed key reset would bind stale buffers, so the spec must swap an
  imported buffer between encodes.

### GM-03 Gated loops cannot gate the published primitives inside them; one runs its full reduction every round after convergence
- Class: `blocked` (refines register row 49; the concrete need is not recorded there).
- Local: `gpu-spatial-analysis/areal-interpolation/gpu-pycnophylactic.ts:751-755` (`GPUReduction` max over
  `cellCount` in every iteration of a loop gated by `createWGSLDispatchGate`, loop at `:621`; the
  surrounding local kernels are gated, the reduction is not). Ungated loops with published primitives:
  `cell-topology/gpu-s2-cap-covering.ts:334`, `gpu-s2-polygon-covering.ts:477`,
  `spatial-clustering/gpu-hierarchical-density-clustering.ts:347` (`GPUScan`).
- Published: `gpu-core/gpu-command-dispatch-metadata.js:9-28` (`dispatchWorkgroups` on the node; the
  getter is `@internal`, neither function is exported by any `index.d.ts`); `gpu-reduction.js:507`,
  `gpu-scan.js:492,687`, `gpu-compaction.js:156,314`, `gpu-group-aggregation.js:565` attach it;
  `gpu-scalar-dispatch-gate.js:91-105` (`gateGPUCommandNodes`, not exported).
- Evidence: read both.
- What blocks it: to gate a node with a GPU-indirect condition, the arming kernel must write the exact
  `[x, y, z]` the node would dispatch (`gpu-command-graph.js:180-210` replaces the one `dispatch` by
  `dispatchIndirect`). Published primitives know it and attach it, but nothing public reads it.
- What luma.gl needs: export `getGPUComputeDispatchWorkgroups` (read side), or export
  `gateGPUCommandNodes` / `GPUScalarDispatchGate` taking a caller `GraphDataView<'uint32'>` as the
  predicate. Then the pycnophylactic reduction (1 site today) and any future primitive inside a
  gated round could share the loop's gate.
- Verify: today's spec of the pycnophylactic converged flag; parity would be bit-identical (a gated-off
  reduction only stops recomputing a stale maximum, which the flag kernel already ignores, `:769`).
- Risk: none on results; GPU time only.

### GM-04 `GPULoopOperation` / `GPUConditionalOperation` do not replace the local unrolled gated loops
- Class: `blocked` (records the alpha.13 answer to question 1; register rows 49 and 77 need the correction
  in section 4).
- Local: `raster-relaxation.ts:364-433` (`createRasterGatedLoopNodes`), `wgsl-kernel-nodes.ts:176-223`
  (`createWGSLDispatchGate`), 33 caller files.
- Published: `gpu-control-flow-operation.d.ts:10-37`; `gpu-graph-compiler.js:388-427` (conditional:
  `gpu-indirect-gate`; loop: `bounded-unroll` without predicate, else `gpu-gated-bounded-sequence`, i.e.
  `maximumIterations` copies of the body with every compute node after `minimumIterations` gated);
  `:546-590` (`decorateComputeNode`: one `GPUScalarDispatchGate` update node plus a 12-byte transient
  dispatch buffer per gated compute node, plus a conjunction node per nesting level; throws "must declare
  exact dispatch geometry" for a node without the metadata of GM-03).
- Evidence: read both.
- What each provides, and on which graph:
  - Node `condition` with `source: 'gpu', mode: 'indirect'` is public on `GPUCommandGraph` compute nodes
    (`gpu-command-graph-types.d.ts:379-391`; exported type `GPUCommandGraphGPUIndirectCondition`). This
    is the predicate-controlled indirect dispatch, and this repository already uses it everywhere.
  - `GPUConditionalOperation` and `GPULoopOperation` exist only for `GPUGraph`, with a predicate that is a
    `GPUGraphScalar<'uint32'>`. They lower to the same per-node indirect gate. There is no loop that
    re-runs a body until a GPU predicate clears without unrolling it (`nativeLoops: false`,
    `gpu-graph-compiler.js:41`), and no continuation across submissions.
  - `GPUIncrementalExecution` caches per-batch partials and re-merges on change
    (`gpu-incremental-execution.d.ts`); it is not iteration and fits none of the loops.
- Why no local loop can use them today: (a) local contributors live in a `GPUCommandGraph` with
  `GraphDataView` status words, not in a `GPUGraph` with scalar values; (b) their nodes carry no
  dispatch metadata, so the compiler throws; (c) even if both were solved, the published lowering emits
  about `2N` nodes for N gated rounds of one kernel where `createRasterGatedLoopNodes` emits
  `N + ceil(N/4) + 1` and shares one indirect buffer, so the compile-cost cliff (centrality's 32,768
  cap, sparse PCG's 20,480 nodes) would get worse, not better.
- Change: none in code. Keep the local gate helpers.
- Risk: not applicable.

### GM-05 `createPublishNode` checks a hard-coded 8 storage bindings instead of the device limit
- Class: `misuse` (limits not consulted).
- Local: `wgsl-kernel-nodes.ts:793-795`; the same factory reads `maxStorageBuffersPerShaderStage` at
  `:328-334`.
- Published: `Device.limits.maxStorageBuffersPerShaderStage` (core).
- Evidence: read one (local; the limit is a plain WebGPU limit).
- Change: delete the hard-coded check (the call to `createWGSLKernelNode` at `:802` already rejects
  above the device limit with a better message), or use the device limit. 1 site.
- Verify: `test/utils/wgsl-kernel-nodes.node.spec.ts` and any spec that builds a publish node with 9
  bindings on a default device (expect the same refusal, different message).
- Risk: on devices created with a higher limit a 9-binding publish node becomes legal; no result change.

### GM-06 Sync `graph.compile()` inside async functions in kepler and lenses
- Class: `unused-feature` (`compileAsync`).
- Local: `kepler/src/analysis/point-weights-streamed.ts:110` (inside `async function runGraph`),
  `kepler/src/exploration/gpu-resident-explorer.ts:809` (`async gatherPage`),
  `kepler/src/exploration/gpu-table-inspector.ts:511` (`async readDerivedPage`),
  `kepler/src/exploration/gpu-column-statistics.ts:181` (awaits readbacks right after),
  `lenses/src/computation-lenses/scenarios/gpu-spatial-analysis/s2-lens-run.ts:141` (`async runS2Lens`),
  `spatial-scenario-utils.ts:157` (`async runSpatialScenario`). Six sites. Twelve more sync compiles in
  kepler/lenses are in synchronous methods and are not counted.
- Published: `gpu-command-graph.d.ts:311-324` (`compileAsync`: starts every pipeline compile before
  waiting); `gpu-command-graph-compiler.js:96-103` (a node without `compileAsync` still gets async
  pipelines, because `new Kernel` sees the open `PipelineFactory` async-compilation scope). The
  showcase already follows this (`examples/deck/spatial-analysis-showcase/src/scenes/BUILDER-GUIDE.md:79`).
- Evidence: read both.
- Change: `await graph.compileAsync()` at the six sites. `createWGSLKernelNode` needs no
  `compileAsync` callback for this to work.
- Verify: existing kepler node specs that drive these paths (they mock or skip the device); a headless
  run of a lens scenario.
- Risk: compile errors surface as a rejected promise one tick later; results unchanged.

### GM-07 Budgeted, resumable execution is unused for the largest unrolled graphs (lead)
- Class: `unused-feature` (lead, inferred benefit).
- Local: whole-graph single submissions, for example `kepler/src/analysis/catalog-gpu-session.ts:92-97`;
  the sparse resistance solve documents "in one submission" up to 20,480 nodes
  (`upstream-workarounds.md:59`).
- Published: `CompiledGPUCommandGraph.createExecution` / `getExecutionPlan`
  (`gpu-command-graph.d.ts:401,410`), `GPUCommandGraphExecution.encodeNext`,
  `GPUCommandGraphExecutionBudgetController` (learns the budget from measured step time).
- Evidence: read one (published declarations; not run).
- Change: where a kepler catalog run or a lens run can be long, encode it through
  `createExecution({maximumInvocationCount, maximumNodeCount})` and submit one step per frame or idle
  slot. Every `createWGSLKernelNode` node already declares the workload these budgets read
  (`wgsl-kernel-nodes.ts:431-440`).
- Cost: static estimates overstate gated rounds (an indirect-gated round counts its full maximum), so
  steps are smaller than needed after convergence; results do not change because transients and
  indirect buffers persist in the compiled graph across steps (not verified by a run).
- Verify: a headless spec that runs one gated contributor whole and by budgeted steps and compares
  outputs bit for bit.
- Risk: none expected on results; latency trade-off only.

### GM-08 `getBoundedInvocationIndexSource` is a public copy of the index prologue the kernel factory writes by hand
- Class: `adopt-with-cost` (recommend not adopting; record it).
- Local: `wgsl-kernel-nodes.ts:386-405`.
- Published: `gpu-core/gpu-dispatch-utils.js:25-31` (public via `gpu-core/index.d.ts`).
- Evidence: read both. Same guard (`floor(0xffffffff / workgroupSize) + 1`) and same linearisation.
- Cost: the published text bakes `layout.x` and `layout.y` into the WGSL; the local text reads them from
  override constants so that kernels of every problem size share one shader module (rule 12). Adopting
  it would multiply shader modules and pipelines. Keep the local text; consider a pin like the other
  carried WGSL in `official-wgsl-copies.node.spec.ts`, since it must stay equal to the published
  arithmetic that `getBoundedDispatchLayout` plans for. 1 site.
- Risk: none today.

## 3. Checked and fine

- Predicate-controlled indirect dispatch: public on `GPUCommandGraph` nodes and used correctly
  (`wgsl-kernel-nodes.ts:134-143, 204-213`; the gated node encodes exactly one `dispatch`, which the
  graph swaps, `gpu-command-graph.js:180-210`; the arming node is ungated, `:170-171`).
- Pipeline and shader caching: `createWGSLKernelNode` builds a published `Kernel`, which uses the
  default `PipelineFactory` and `ShaderFactory` (`kernel.js:56-75`); override constants keep sources
  equal so pipelines are shared. Nothing local duplicates the caches.
- `getBoundedDispatchLayout`: used, public, 32 call sites; the indirect active-count dispatch reproduces
  the same layout on the GPU (`wgsl-kernel-nodes.ts:116-132`), which nothing published offers.
- Binding-limit checks at declaration (`wgsl-kernel-nodes.ts:328-354`): earlier and better-named than
  the published post-compile `preflight.fitsDeviceLimits`; both kept.
- Uniform packing: none in the factory (override constants and storage parameter buffers); `Kernel` has
  no uniform support either.
- `GPUContributorOperation` (question 4): still the smallest public bridge. `GPUCompositeOperation` has no
  `contract` (`gpu-operation.d.ts`), `GPUShaderPassOperation` is a fullscreen fragment pass over textures,
  and a bare `GPUGraphPrimitive` in a `GPUGraph` has no ports to plan. The lowering uses only public
  `GPUOperationLoweringContext` members.
- `GPUGraphReduction` / `GPUGraphGroupAggregation`: no local `GPUContributorOperation` or `GPUCPUOperation`
  does only a dense reduction or grouped aggregation, and package sources build no `GPUGraph` (tests
  only). The 95 `new GPUReduction` / `new GPUGroupAggregation` sites are inside command-graph contributors
  on bound views, where the semantic operations do not apply. Where contributors use local fixed-point
  sums instead, the published float compare-exchange sums would lose determinism (register section 4).
- `describeGPUCommandGraph` (question 3) already reads published `stats.nodeOrder`, `preflight.nodes`
  (resources, workload, condition, publication) and transient totals. `GPUCommandGraphInspector` is a
  rolling p50/p95 timing and counter store with no edges, sources or bindings; `inspectGPUGraphCompilation`
  summarises `GPUGraph` lowering only; `getGPUCommandGraphExecutionPlan` is scheduling. None replaces the
  record; the remaining need (per-node source, entry point, workgroup size, bindings) is register row 44.
- `GPUSnapshotTap` versus `GPUCommandGraphEncoding.readResource`: the published preview is "current
  allocation contents, not a historical node snapshot" (`gpu-resource-preview.d.ts:3`), and transients are
  aliased, so taps are still needed (register row 36 holds).
- Timing (question 6): `submitGPUCommandGraphWithTimings` uses the public `timeProfilingQuerySet` and
  `encoding.readTimings()`; `examples/deck/spatial-analysis-shared/vector-timing.ts:101-125` prefers GPU
  timestamps and falls back to a wall clock only without `timestamp-query`. No `performance.now()` in
  package sources; the 129 uses in the three example folders and the website (42 files) that I sampled are animation
  and settle timers plus encode-time display (not all read).
- Kepler scratch accounting reads published `stats.physicalTransientResourceBytes`
  (`kepler/src/session/metered-command-graph.ts:15-19`).
- `GPUReadbackRing`: used in 43 example files, 1 analysis source directory (`region-statistics`) and 2
  website files; kepler page reads are variable-size
  `readAsync` pages with a pending-request map (`resident-dataset.ts:260-307`), which a fixed-capacity ring
  does not fit.
- `GPUCommandGraphAutotuner`: only `GPUScan` consults it (`gpu-scan.js:499-516`); exploration would pick
  scan variants with different float summation order per run, against this repository's determinism
  rules. Not passing `autotuner` is right.
- `GPUIncrementalExecution`: kepler's Group By passes split the key space by memory
  (`kepler/src/aggregation/kepler-group-by-passes.ts`), not batches with cached partials; no match found.
- `validateCommandNodeBudget` (32,768): construction-time guard with no published equivalent; its comment
  (`gpu-contributor-utils.ts:476-482`, "no GPU-gated continuation") is still true at alpha.13.
- `captureGraphCommandNodes` (`gpu-contributor-utils.ts:408-440`): works because published `graph.add` and
  `addGPUCommandNode` route through `graph.addComputePass/addRenderPass/addCopyPass`
  (`gpu-command-node.js`); already register row 65.

## 4. Register corrections (`dev-docs/upstream-workarounds.md`)

- Row 48 (Bind-group reuse): the premise is wrong at alpha.13. No `_bindGroupCacheKeys` option exists on
  `Kernel.dispatch`, and the listed local helpers are dead code that never reaches a bind-group cache
  (GM-01). Say: "every `Kernel.dispatch` creates a bind group, local and published alike; no local
  workaround exists; see GM-02 for the public pipeline-bindings route."
- Row 49 (Indirect-dispatch gate metadata): "Predicate-controlled indirect dispatch swap is lost" is
  stale. The swap is public through node `condition: {source: 'gpu', mode: 'indirect'}` and used by 33
  files through `createWGSLDispatchGate`. What is not public is `setGPUComputeDispatchWorkgroups` /
  `getGPUComputeDispatchWorkgroups` and `gateGPUCommandNodes`: local loops cannot gate published
  primitives in a round (GM-03), and local nodes cannot sit under `GPUConditionalOperation` /
  `GPULoopOperation` in a `GPUGraph` (GM-04).
- Row 77 (Centrality compile cost): "GPU-gated continuation or indirect-dispatch gate in
  `GPUCommandGraph`": the indirect gate exists and the reachability and sigma/delta rounds already use it
  (`network-frontier.ts:357-385`, `network-centrality-passes.ts:468`). Only the continuation (a loop that
  is not unrolled) is missing; `GPULoopOperation` unrolls too (GM-04).
- Row 59 (sparse Laplacian): add that the public `createGPUConjugateGradientGraph` is itself a
  `GPULoopOperation` (`gpu-conjugate-gradient-program.js:148`) with no preconditioner, so it would unroll
  as well and could not replace the Jacobi-PCG even if it could run in a caller graph.
