# Lifecycle review — construction, replay, resize and disposal

## Summary

The clearest lifecycle bug is in network map matching. I reused a complete cost table, changed its shared limit from 0 to 10, and turned a connected track into a break. Both executions reported zero route overflow. Rebuilding the table restored the route.

Three raster stream packers let a later chunk change the unit of numbers already accumulated. Two samples of 1 can consequently decode as 3. A separate short-chunk hazard makes a second cell read an observation from the preceding submission.

Construction snapshots also differ between close siblings. Mutating one destination grid changed the sampler emitted by `GPURasterReprojection` from width 2 to width 1. The tiled contributor given the same grid retained width 2.

Of 498 public producers, seven expose `destroy()`. Two documentation examples construct one of those seven inline and discard the only reference through which the caller can release its resources.

**The most valuable change is to bind the metadata that gives retained data its meaning to the retained data itself.** Start with the cost limit that certifies a route table and the unit exponents that decode raster stream state. Both have small, discriminating two-submission tests.

## Census

The runtime census bundled the source targets of all 11 `modules/analysis/package.json` entries, deduplicated exported values by identity, and selected producers with the following predicates. It used r19 source, including inherited methods, without loading `dist`.

```js
const values = [...new Set(entries.flatMap(Object.values))];
const producers = values.filter(c =>
  typeof c === 'function' && c.prototype && 'getCommandNodes' in c.prototype
);
const owners = producers.filter(c => 'destroy' in c.prototype);
const updateMethods = producers.flatMap(c =>
  Object.getOwnPropertyNames(c.prototype)
    .filter(m => /^(set|invalidate|reset|update)/.test(m))
    .map(m => `${c.name}.${m}`)
);
```

| Variant | Count | Names or interpretation |
|---|---:|---|
| Public command-node producers | 498 | 494 define the method directly. `GPURasterEVI`, `GPURasterNBR`, `GPURasterNDWI` and `GPURasterSAVI` inherit it. |
| Producer exposes `destroy()` | 7 | `GPUConditionalGaussianSimulation`, `GPULocalFrameProjection`, `GPUSpatialJoinPrepared`, `GPUStudyWindowPreparation`, `GPUTerrainSpikeRepair`, `GPURasterReprojection`, `GPURasterTileReprojection` |
| Producer has no `destroy()` | 491 | This counts caller-visible cleanup methods. A compiled node can still own resources. |
| Producer exposes `dispose()` | 0 | `rg -n '\bdispose\(' modules/analysis/src` also returned zero. |
| Producer has an explicit update or invalidation method | 4 | Gaussian simulation has four methods. Local kriging has `setCovariance`. Prepared join and study window each have `invalidate`. The other 494 expose none matching the predicate. |
| Public subclasses of `GPUContributorOperation` | 128 | They inherit cleanup for the most recently constructed contributor. The bridge lowers again on each run. |
| Public producers using the three explicit per-graph command-node caches found by `rg 'this\.commandNodes\.get\(graph\)'` | 6 | The four spectral indices above, `GPURasterEmbeddingKMeans`, `GPURasterEmbeddingNearestPixels` |
| Network prop types with a directly declared `*Count` accepting either a number or a `GraphDataView` | 5 | Reachability, isochrones and neighborhood take `startCount`. Path extraction takes `targetCount`. Service areas takes `facilityCount`. |
| Raster time-series packers writing `settings.activeSliceCount ?? chunkSliceCount` | 4 | Streaming accumulator, climate-run accumulator, standardized-index calibration and standardized-index application. All four accepted fractional, negative and over-capacity counts. |

For the network count I parsed `gpu-network/**/*.ts` with `ts.createSourceFile` and selected direct members of `GPU*Props` type literals. For the raster count, `rg -n 'settings\.activeSliceCount \?\? chunkSliceCount' modules/analysis/src` returned the four packers. These are bounded searches, not estimates for every count-like field in the package.

Of 116 Markdown pages under `docs/api`, 48 matched `\b(getCommandNodes|graph\.add)\b`, 30 mentioned destruction, and 50 matched a broad mutation/construction-capture search. Even the broadest of those searches does not cover every page. The documentation counts used the KI copy for staging-owned paths. They measure text presence, not whether a page explains its lifetime correctly.

Three different preparation lifetimes already have legitimate uses. `prepareGPUNetworkFlowDemands`, `prepareGPUNetworkRestrictionStates` and `prepareGPUNetworkPointRoutingTurns` produce ordinary host data. The spatial preparation classes bind resources to a graph and have explicit invalidation. Raster conservative remapping lets the caller own prepared weight buffers and explicitly select preparation or reuse. The useful common vocabulary is **what was captured, who owns it, and what makes it invalid**.

Iteration likewise has two different costs. The ordinary contributor declares bounded rounds. `createNetworkCentralityContinuation` submits windows and awaits a four-byte frontier count after each window. **EXECUTED:** with five rounds, two rounds per window, betweenness enabled and a nonempty frontier throughout, the host scheduler called the boundary callback nine times: `[2,2,1]` in each of search, sigma and delta. With a zero frontier on each first window it called it three times. That is an explicit host synchronization choice a caller can make without reading back the full result. `gpu-network/network-centrality/network-centrality-continuation.ts:63,690,697`.

Paths below use `S/` for `modules/analysis/src/` and `D/` for `docs/api/`. **KI** identifies the staging-owned copy checked in `luma-contributors-ki`. Other implementation evidence is from r19. The checkout was changing during the review.

## Proposed convention

1. Capture validated numbers, tuples, options and nested port maps at construction. Preserve the identities of the GPU resources they reference.
2. Give each producer one documented declaration per graph. Show how to retain the returned nodes when a caller needs to inspect and schedule them.
3. Replay by encoding the compiled graph. Submit preceding encodings before overwriting queue-visible inputs on which they depend.
4. Distinguish capacity, physical stride and active count. A short write changes bytes only. A documented count, mask or sentinel changes membership.
5. Specify what each execution publishes for active rows, inactive rows and status. Specify which fields intentionally accumulate.
6. Bind cache provenance and state representation to their outputs. List every change that requires reset, invalidation or reconstruction.
7. State who advances an iterative operation, its budget, its convergence signal and each host readback boundary.
8. Preserve supported dynamic resource identities across internal graphs. Document the rebuild boundary when a physical allocation changes.
9. An unsuccessful construction or declaration releases its acquisitions and leaves public readiness indicators accurate.
10. Name each cleanup owner and teardown order. Mechanically check the contract with mutation, smaller-second-run, discard, resize and injected-allocation-failure probes against the public entry points.

## Findings

| ID | Kind | Reach | Change | Cost | Where | One line |
|---|---|---|---|---|---|---|
| LIF-01 | W | ORDINARY | ADDITIVE | M | Network cost table → map matching | A live cutoff can certify a stale table and create a false track break. |
| LIF-02 | W | ORDINARY | ADDITIVE | M | Three raster accumulators | Changing a per-chunk scale changes the interpretation of earlier samples. |
| LIF-03 | W | UNUSUAL | ADDITIVE | S | Four raster parameter packers | A short packing target silently preserves the previous frame’s active count. |
| LIF-04 | W | UNUSUAL | ADDITIVE | M | Point density, raster reprojection, study window | Mutating validated construction inputs changes published bindings or emitted geometry. |
| LIF-05 | W | UNUSUAL | ADDITIVE | S | Snapshot arena | Reordering the exposed slot list after capture silently swaps the labels on the data. |
| LIF-06 | S | UNUSUAL | ADDITIVE | M | Network centrality continuation | Growing an imported `DynamicBuffer` leaves phase graphs bound to its destroyed backing. |
| LIF-07 | D | ORDINARY | ADDITIVE | S | Raster time-series guide | “Short last chunk” omits the fixed row pitch needed to avoid reading the previous chunk. |
| LIF-08 | D | ORDINARY | ADDITIVE | S | Gaussian simulation | A documented dynamic covariance port does not exist. |
| LIF-09 | E | ORDINARY | BREAKING | S | Gaussian simulation operation form | `reusePreparation` allocates retained state that cannot survive the next operation run. |
| LIF-10 | D | ORDINARY | ADDITIVE | S | Cleanup and rebuild examples | Two examples discard the cleanup owner, and the shared rebuild example omits it. |
| LIF-11 | C | UNUSUAL | ADDITIVE | M | Three preparation owners | A failed declaration can report readiness or retain allocations that prevent retry. |
| LIF-12 | M | UNUSUAL | ADDITIVE | S | `GPUParameterBuffer` | A failed initial upload strands an allocation before the caller receives its owner. |
| LIF-13 | E | UNUSUAL | ADDITIVE | M | Snapshot arena and tap | Discarded encodings consume frames and can be reported as successful zero captures. |
| LIF-14 | C | ORDINARY | ADDITIVE | S | Command-node declaration | Inspecting nodes and then adding the producer works for one sibling and fails for another. |
| LIF-15 | E | UNUSUAL | ADDITIVE | S | Snapshot readback | A sparse capture reads the whole reservation and a rejected first read permanently poisons retries. |

### LIF-01: A complete route table can certify the wrong cost limit

**Where.** `S/gpu-network/network-cost-table/cpu-network-cost-table-operation.ts:105`, `S/gpu-network/map-matching/cpu-map-matching-operation.ts:470,494`, `S/gpu-network/map-matching/gpu-map-matching.ts:869,878,887`, and KI `D/gpu-network/operators.md:1233,1240,1258,1270`. `KNOWN(NET-01)` covers an incomplete build. This case starts with a genuinely complete build.

**Evidence — EXECUTED.** Four nodes on a line, three directed unit edges, two fixes at `x=0.5` and `2.5`, one candidate each. I built at cutoff `0`, capacity `16`, using the public CPU functions. The table had `complete=1`. Reusing the same limit array gave:

| Action | Matched edges | Breaks | Route overflow |
|---|---|---|---:|
| Use the table at its built cutoff of 0 | `[0,2]` | `[0,0]` | 0 |
| Change only `costLimit[0]` to 10 | `[0,2]` | **`[0,1]`** | 0 |
| Rebuild at 10 | `[0,2]` | `[0,0]` | 0 |

**Now.** The page says to build once while weights are static and shows `routeTable: {..., costLimit: tableLimit}`. The single-graph snippet recomputes the table on replay. An application that actually builds once and retains the table can change the shared limit independently. A missing entry then means something different.

**Wanted.** Add optional GPU output `builtCostLimit: GraphDataView<'float32'>` and return the captured limit from `computeNetworkCostTableOnCPU`. Use that output in the retained-table recipe. Show the build and repeated matching in separate graphs. State that changed topology, edge lengths or an increased certified limit requires rebuilding. **SOURCE GPU probe:** use the four-node case above and compare the `breaks` buffer with a fresh build.

**In-repository callers that change.** No required changes. Update the operator example and the map-matching table-completeness fixture to exercise the captured limit. **Value and cost.** Map-matching users retain preprocessing without inventing a route break. M.

### LIF-02: Raster state can mix two fixed-point units

**Where.** `S/raster-analysis/raster-time-series/gpu-raster-streaming-accumulator.ts:73,271,293`, `gpu-raster-climate-run-accumulator.ts:60,230,253`, `gpu-raster-standardized-index-calibration.ts:58,234`. KI `D/gpu-raster/raster-time-series.md:56,79,407,433`.

**Evidence — EXECUTED host, SOURCE shader.** With the default stream-length bound, the public streaming packer gave sum exponents `-69` and `-68` for `maximumValueMagnitude` of 1 and 2. I separately encoded one unit sample at each exponent, added the integer words, and passed them to the public decoder at the second exponent. It returned **3**, expected **2**. The kernel reads the prior integer words and adds the next term using the current parameter exponent. The climate degree-day and standardized-calibration moment paths use the same pattern.

**Now.** `pack({reset: false, maximumValueMagnitude: nextBound}, 1)` is a valid call. The existing state carries no exponent against which the new packet can be checked.

**Wanted.** Document the scale-defining settings as fixed for a stream between resets. Provide a stream settings object that captures the scale once and packs only `{reset, activeSliceCount}` on subsequent chunks. Cover `maximumIncrement` for degree days and the calibration location and exponents for standardized indices. **Smallest GPU probe:** one cell, one sample of 1 per submission, reset only the first, bounds 1 then 2. Read count and decoded sum. A supported scale change must give `[2,2]`; an unsupported one should be refused before state changes.

**In-repository callers that change.** None with an additive settings object. **Value and cost.** Climate and remote-sensing users can accumulate changing batches without silently changing units. M.

### LIF-03: A short parameter target silently replays an old active count

**Where.** Under `S/raster-analysis/raster-time-series/`, `gpu-raster-streaming-accumulator.ts:62`, `gpu-raster-climate-run-accumulator.ts:44`, `gpu-raster-standardized-index-calibration.ts:47`, `gpu-raster-standardized-index.ts:41`. `KNOWN(CORE-101)` concerns short CPU projection targets. The new consequence here is stale GPU control state.

**Evidence — EXECUTED.** I initialized a five-word `GPUParameterBuffer` for two active slices, packed the second frame into `new Float32Array(1)`, and called the real `GPUParameterBuffer.write` against a recording buffer. The packer returned `[0]`. Resident words became `[0,2,-69,-67,1]`, although I requested `activeSliceCount: 1`. The second word retained the old `2`. All four packers accepted a short target and counts `-1`, `1.5` and `3` for capacity `2`.

Three packers also partially overwrite a reusable target before rejecting `maximumStreamSliceCount: -1`. Targets prefilled with 8 came back as `[0,1,8,8,8]`, `[0,1,4,8]` and `[0,1,4,8,8,8,8]`.

**Now.** `parameters.write(pack(settings, capacity, scratch))` looks like a complete frame update. A short result is a valid *partial* write to the parameter buffer.

**Wanted.** Validate target length against the exported layout constant, validate integer `activeSliceCount` in `[0,chunkSliceCount]`, and compute every fallible derived word before the first write. Keep zero active slices valid. Errors should name the packer, field and permitted range.

**In-repository callers that change.** None using valid packets. **Value and cost.** All four stream APIs reject a bad update while the caller still has its previous good packet. S.

### LIF-04: Construction metadata and emitted work can describe different grids

**Where.** KI `S/gpu-spatial-analysis/point-density/gpu-point-density.ts:332,341,590,784`; r19 `S/raster-analysis/gpu-raster-reprojection.ts:249,338,396`, `S/gpu-spatial-analysis/study-window/gpu-study-window-preparation.ts:167,169,273`. The positive control is `S/raster-analysis/gpu-raster-tile-reprojection.ts:90,148`. `KNOWN(CORE-08)` and `KNOWN(GEO-126)` covered other named producers.

**Evidence — EXECUTED.**

| Construction followed by caller mutation | Observed at declaration | Expected from the validated construction |
|---|---|---|
| Point density `[2,1]`; reverse the tuple and replace `output.values` with another same-size view | `surface` still said `2×1` and referenced the original output. Emitted WGSL said `WIDTH=1`, `HEIGHT=2`. Node resources contained the replacement output and omitted the original. | One grid and one output identity throughout |
| Construct ordinary and tiled reprojection at width 2; set the shared destination width to 1 | Emitted sampler widths were respectively `1` and `2` | Both `2` |
| Construct a study window with `samplingGrid=[1,1]`; set element zero to 2 | Both `sampleGrid` and `getViews(graph).sampleGrid` reported `[2,1]` | `[1,1]` |

**Now.** `const surface = density.surface; props.output.values = other; graph.add(density)` leaves a downstream consumer holding a surface the producer no longer writes. The binding discrepancy was observed on the host. **Smallest GPU probe:** prefill the two outputs with different sentinels, submit one point and read both.

**Wanted.** Snapshot the grid tuple, grid metadata, nested output/status maps and sampling tuple before validation. Reuse the tiled reprojection grid snapshot for its ordinary sibling. Keep the original view identities so writing their contents continues to work.

**In-repository callers that change.** None following `dev-docs/contributor-contract.md:93`. **Value and cost.** A valid downstream surface remains connected when application code reuses an options object. M across the three classes.

### LIF-05: Capture labels can change after their bytes were recorded

**Where.** `S/computation-record/gpu-snapshot-arena.ts:40,64,124,162`.

**Evidence — EXECUTED.** I recorded one frame with slots `x=7`, `y=9`, called `arena.slots.reverse()` from JavaScript, then read it. The result was `[['y',7],['x',9]]`. Expected `[['x',7],['y',9]]`. Individual slot descriptors are frozen. The exposed array containing them is mutable at runtime.

**Now.** A JavaScript UI sorting `arena.slots` for display changes how the decoder labels existing frames. TypeScript’s `readonly` prevents this particular call in checked code.

**Wanted.** Freeze the copied array as well as every descriptor. Pin recording, attempted reorder and readback in one host regression.

**In-repository callers that change.** None respecting the declared readonly array. **Value and cost.** Capture viewers retain the correct association between values and names. S.

### LIF-06: Continuation graphs lose a dynamic buffer’s identity on resize

**Where.** `S/gpu-network/network-centrality/network-centrality-continuation.ts:168,266,271`; installed `@luma.gl/core/src/dynamic/dynamic-buffer/dynamic-buffer.ts:261,296`.

**Evidence — EXECUTED.** On the null device I declared a two-node, one-pivot closeness continuation. I imported its one-row pivot view through a four-byte `DynamicBuffer` and inspected the real phase graph declarations used by the runner. After `pivots.resize({byteLength: 8})`, **4 of 4** phase imports still pointed to the old physical buffer. `old.destroyed === true`. The original graph handle still held the dynamic wrapper. The logical pivot count remained one.

**Now.** Increasing backing capacity between completed batches can invalidate an otherwise unchanged runner. **SOURCE device probe:** grow four bytes to eight, write pivot 1, run the next batch and compare against a newly constructed runner for pivot 1. The host probe establishes the destroyed binding; device error behavior was not measured.

**Wanted.** Preserve a `DynamicBuffer` default import when rebinding external views into the phase graphs. Preserve the original logical range and hazard tracking. If aliasing prevents supporting a particular combination, reject it during runner construction with the port name and rebuild instruction.

**In-repository callers that change.** None with ordinary buffers. **Value and cost.** A host can grow its arena without leaving the centrality scheduler bound to destroyed storage. M.

### LIF-07: A short last raster chunk still has the full cell pitch

**Where.** KI `D/gpu-raster/raster-time-series.md:29,60`; under `S/raster-analysis/raster-time-series/`, streaming accumulator `:284`, rolling mean `:150`, climate-run accumulator `:232`, standardized-index calibration `:239`, standardized-index application `:202`.

**Evidence — EXECUTED host, SOURCE shader.** From “cell-major” and “`activeSliceCount` allows a short last chunk,” the natural upload for two cells with one remaining observation each is:

```ts
valuesBuffer.write(Float32Array.of(12, 22));
parameters.write(pack({reset: false, activeSliceCount: 1, maximumValueMagnitude: 32}, 2));
```

The host byte-layout probe started with `[10,11,20,21]`. The short write produced `[12,22,20,21]`. Active row addresses for the two cells are 0 and 2, yielding **`[12,20]`**, expected `[12,22]`. The five shaders address cells with constructor `chunkSliceCount`. The direct props TSDoc gives the correct pitch.

**Wanted.** Put the example on the time-series page: for capacity two, upload `[12,pad,22,pad]`, or write the two values at byte offsets 0 and 8. Apply the same pitch to a per-observation mask. **Smallest GPU probe:** use rolling mean with window one, replay that short chunk, and read active outputs at indices 0 and 2.

**In-repository callers that change.** None. **Value and cost.** A raster developer can resize the logical batch without accidentally incorporating yesterday’s observation. S.

### LIF-08: Gaussian simulation documents a covariance update it cannot bind

**Where.** `D/gpu-spatial-analysis/interpolation-expansion.md:122,129,131`; `S/gpu-spatial-analysis/spatial-interpolation/gpu-conditional-gaussian-simulation.ts:61,73,129,466`. The actual mutable covariance API belongs to `gpu-local-kriging.ts:707`.

**Evidence — EXECUTED.** Following the simulation update table, I passed `parameters` alongside valid covariance and views. `Object.hasOwn(simulation.props, 'parameters')` returned `false`. The props type has no such field. The declaration generates covariance code from `props.covariance`.

**Now.** `new GPUConditionalGaussianSimulation({...props, parameters: covarianceView})` is an excess-property error for a typed object literal. JavaScript accepts it and the producer ignores it.

**Wanted.** Remove the `getGPULocalKrigingParameterValues` row from the simulation table. Say “Changing the covariance model or its parameters requires a new simulation and graph. `invalidatePreparation()` refreshes changed geometry using the covariance captured at construction.” Keep the dynamic update example under local kriging.

**In-repository callers that change.** None. **Value and cost.** Simulation users can tell which update actually changes the model. S.

### LIF-09: The simulation operation advertises reuse across a lifetime it does not have

**Where.** `S/gpu-spatial-analysis/spatial-interpolation/gpu-conditional-gaussian-simulation-operation.ts:25,89`; `S/analysis-core/operations/gpu-contributor-operation.ts:91,179`; simulation producer `:417–426`.

**Evidence — EXECUTED.** I resolved the same named ports twice through one `GPUConditionalGaussianSimulationOperation({..., reusePreparation: true})`. The returned contributors were distinct. Calling `setMean(2)` on the first then threw `preparation controls need reusePreparation and a producer that was not destroyed`. The second accepted the update. The bridge destroys the previous owner at lowering; the producer allocates persistent factor, moment, control and validity storage on the retained route.

**Now.** `await compiled.run(...); await compiled.run(...)` rebuilds the contributor. Selecting reuse on the operation pays for persistent storage without retaining its factor into the next run.

**Wanted.** Remove `reusePreparation` from `ConditionalGaussianSimulationOperationProps` before alpha. Leave it on the direct producer, and show that form for a sequence of draw batches. Genuine reusable operation lowering would require a different lifetime guarantee from the executor.

**In-repository callers that change.** **0 call-site edits found.** `rg 'new (GPU|CPU)ConditionalGaussianSimulationOperation\(' modules examples website` and an AST check found 15 constructor sites at the final scan. None supplied this option. The shared operation props, snapshot field list and forwarding implementation change. Existing retained-preparation examples use the direct producer. **Value and cost.** Graph users avoid an ineffective optimization switch. S, breaking public type change.

### LIF-10: Two documentation examples discard the cleanup owner

**Where.** KI `D/gpu-spatial-analysis/operators-lines.md:462`, KI `D/gpu-terrain/operators.md:550`, and KI `D/gpu-spatial-analysis/concepts.md:179,314,329,336`. Implementations: `S/gpu-spatial-analysis/geometry-measures/gpu-local-frame-projection.ts:99,106` and KI `S/gpu-terrain/terrain-decode/gpu-terrain-spike-repair.ts:352,963`.

**Evidence — SOURCE, COUNT.** The projection and spike-repair examples both use `graph.add(new Owner(...))`. The projection owns a plan buffer. Spike repair owns four work buffers after declaration. The general concepts page describes scratch as graph-transient and its replacement example destroys only `next` or `previous`. The seven-owner inventory itself was already recorded; the two lost references and the stale-compile cleanup branch are the actionable additions. `KNOWN(LUMA-116)` concerns a separate allocation-failure path.

**Now.** Following either example leaves no reference for `owner.destroy()`. Following the rebuild pattern with such an owner can repeat that on every superseded compile.

**Wanted.** Retain `projection` and `repair` in the two examples. Show `compiled?.destroy(); owner.destroy()` in `finally`. In the rebuild example, retain the compiled graph and its contributor cleanup together, and clean up both on compile rejection, superseded success and replacement. List the seven owners in the common lifecycle section.

**In-repository callers that change.** Two inline docs examples and one shared rebuild example. No API change. **Value and cost.** Long-lived GIS applications can release an obsolete graph and everything it caused to be allocated. S.

### LIF-11: Failed preparation can leave contradictory readiness state

**Where.** `S/gpu-spatial-analysis/spatial-join/gpu-spatial-join-prepared.ts:320,340,620`; `S/gpu-spatial-analysis/study-window/gpu-study-window-preparation.ts:219,228,365`; `S/gpu-spatial-analysis/spatial-interpolation/gpu-conditional-gaussian-simulation.ts:403,413`.

**Evidence — EXECUTED.** I injected synchronous allocation failures on the null device.

| Failed allocation | State immediately after `graph.add` threw |
|---|---|
| Prepared join `node-minima` | `isDeclaredIn(graph) === true`, four owned buffers live, `storage` threw “needs graph.add first” |
| Prepared study window `edges` | `isDeclaredIn(graph) === true`, seven owned buffers live, `getViews(graph)` threw “needs graph.add before its consumers” |
| Retained Gaussian simulation, second allocation | One owned buffer live. A retry threw “added to one graph, once, before destroy.” |

Explicit `destroy()` recovered the counted buffers in each experiment. The readiness contradiction occurs before any shader execution.

**Now.** `catch { if (prepared.isDeclaredIn(graph)) use(prepared); }` takes the success branch on an incomplete object.

**Wanted.** Keep acquisitions local until declaration succeeds, release them on failure and publish readiness last. The failed graph may already contain imported IDs. Tell the caller to rebuild that graph instead of promising graph rollback. For retained simulation, distinguish failed construction from a successful prior declaration.

**In-repository callers that change.** None on success. **Value and cost.** A failed allocation has a deterministic cleanup and recovery path. M.

### LIF-12: Initial parameter upload failure leaks the new buffer

**Where.** `S/analysis-core/kernels/gpu-contributor-utils.ts:87,94,118`. `KNOWN(CORE-113)` concerned inconsistent property reads during allocation. This failure occurs after successful allocation.

**Evidence — EXECUTED.** With `length: 1`, `format: 'uint32'` and `values: Uint32Array.of(1)`, a recording device allocated one buffer and threw `upload refused` on its initial `write`. Counts after the constructor threw were `{allocated: 1, destroyed: 0}`. Expected `{allocated: 1, destroyed: 1}`.

**Now.** `const parameters = new GPUParameterBuffer(device, props)` throws before the caller can assign `parameters` or register its cleanup.

**Wanted.** Protect initialization with cleanup of the already allocated buffer before rethrowing. Preserve short-initializer zero-fill semantics.

**In-repository callers that change.** None. **Value and cost.** Every consumer of the shared parameter buffer gets exception-safe construction. S.

### LIF-13: An abandoned encoding can become a labeled zero capture

**Where.** `S/computation-record/gpu-snapshot-arena.ts:105,137,145,178`; `S/computation-record/gpu-snapshot-tap.ts:117,129`; `D/computation-record.md:269,303`. `KNOWN(ERG-02)` fixed discarded encodings for geometry generation. Snapshot reservation has a separate host-side lifetime.

**Evidence — EXECUTED.** With a two-frame byte-array recording device, I encoded source value 7, discarded that encoder’s copies, encoded 7 again and applied only the second encoder’s copies. `arena.count` was `2`. `arena.read()` returned `[['discard',0,'gpu-readback'],['submit',7,'gpu-readback']]`.

**Now.** The page requires all frames to be submitted. Once a renderer abandons an encoder, the arena provides no way to satisfy that requirement while preserving earlier good captures. Zero is a plausible measurement.

**Wanted.** Add a checkpoint/discard API for unsubmitted suffixes, including frames reserved by taps. Document that a discarded encoder must never be submitted later. If arbitrary out-of-order cancellation is needed, use explicit frame reservations. Keep submitted frame numbering stable.

**In-repository callers that change.** None for the additive suffix API. **Value and cost.** A renderer can cancel a frame without losing its capture history or publishing invented measurements. M.

### LIF-14: `getCommandNodes(graph)` has no portable inspection-and-add sequence in the docs

**Where.** `S/raster-analysis/spectral-indices/raster-spectral-index.ts:84,98`, `S/raster-analysis/raster-embeddings/gpu-raster-embedding-kmeans.ts:102`, `gpu-raster-embedding-nearest-pixels.ts:99`, and KI point density `:731`. Upstream `gpu-command-graph.ts:918,923` explains why `graph.add(producer)` declares again.

**Evidence — EXECUTED.** On the same graph, two calls on a one-cell `GPURasterNBR` returned the same one-node array. On one-cell `GPUPointDensity`, the second call threw `GPUCommandGraph resource id "point-density-counts-scratch" is already in use`. Both first calls were valid.

**Now.** `inspect(producer.getCommandNodes(graph)); graph.add(producer)` succeeds for the cached spectral producer and fails for point density. For a preparation owner it can also cross an explicit one-declaration guard.

**Wanted.** Document the portable sequence alongside `graph.add(producer)`:

```ts
const nodes = producer.getCommandNodes(graph);
inspect(nodes);
graph.add(nodes);
```

Describe declaration as potentially allocating graph resources. Graph replay uses `compiled.encode(...)`. Keep repeatable declaration an optional implementation detail unless a public consumer needs a stronger rule.

**In-repository callers that change.** None. **Value and cost.** Profilers and composition authors can inspect and schedule the same work once. S.

### LIF-15: Reading one frame maps the reservation, and a transient failure is cached

**Where.** `S/computation-record/gpu-snapshot-arena.ts:145,154,156`.

**Evidence — EXECUTED.** With capacity 1,024, one four-byte slot and one captured frame, the recording buffer saw a request for **4,096 bytes**. I made its first map reject and its second map succeed. Two `arena.read()` calls both rejected with `first map`. The buffer received one request total.

**Now.** An application reserves enough history for a long run, stops after one iteration, then reads the whole reservation. A recoverable failure prevents it from reading the already captured history again.

**Wanted.** Request `readAsync(0, count * frameByteLength)`. Keep the arena closed to further encodings after the first read, share an in-flight read, cache successful bytes, and allow a new mapping attempt after rejection. Test concurrent readers followed by one retry.

**In-repository callers that change.** None. **Value and cost.** Sparse runs pay for the captured prefix, and a transient read error need not force recomputation. S.

## Departures by name

| Lifecycle rule | Confirmed departures |
|---|---|
| Capture construction topology and published port identity | `GPUPointDensity`, `GPURasterReprojection`, `GPUStudyWindowPreparation`; `GPUSnapshotArena` exposes mutable decoder ordering. |
| Keep a retained result attached to its interpretation | `GPUNetworkCostTable` / `GPUMapMatching` and their CPU functions; `GPURasterStreamingAccumulator`, `GPURasterClimateRunAccumulator`, `GPURasterStandardizedIndexCalibration`. |
| Validate a complete active-count packet before publishing it | `getGPURasterStreamingAccumulatorParameterValues`, `getGPURasterClimateRunAccumulatorParameterValues`, `getGPURasterStandardizedIndexCalibrationParameterValues`, `getGPURasterStandardizedIndexParameterValues`. |
| Specify active extent and physical pitch together | `GPURasterStreamingAccumulator`, `GPURasterRollingMean`, `GPURasterClimateRunAccumulator`, `GPURasterStandardizedIndexCalibration`, `GPURasterStandardizedIndex`. |
| Preserve supported backing identities across internal graphs | `createNetworkCentralityContinuation`. |
| Publish readiness after acquisition succeeds | `GPUSpatialJoinPrepared`, `GPUStudyWindowPreparation`, retained `GPUConditionalGaussianSimulation`; `GPUParameterBuffer` also needs initial-upload rollback. |
| Offer updates that can affect the stated lifetime | `GPUConditionalGaussianSimulation` documentation, `GPUConditionalGaussianSimulationOperation` props. |
| Retain the owner in lifecycle examples | `GPULocalFrameProjection`, `GPUTerrainSpikeRepair`, and the shared rebuild example when used with any of the seven owners. |
| Distinguish recorded, submitted and readable frames | `GPUSnapshotArena`, `GPUSnapshotTap`. |

## Quick wins

- [ ] Remove the nonexistent covariance-buffer update from the simulation page. Add the two-cell, one-active-slice example to the raster page.
- [ ] Keep the projection and spike-repair owners in their examples. Show cleanup of superseded and rejected graph replacements.
- [ ] Add the inspect-once, `graph.add(nodes)` example to the common execution guide.
- [ ] Freeze `GPUSnapshotArena.slots` and test that recorded values keep their original names.
- [ ] Validate all four raster packers before modifying a caller target. Pin negative, fractional, over-capacity, undersized-target and failed-update cases.
- [ ] Release the `GPUParameterBuffer` allocation if its initial upload fails.
- [ ] Read only the populated snapshot prefix and retry a failed mapping while keeping the arena sealed.

## Larger proposals

1. **Publish the cost table’s built cutoff.** A documentation-only fix can tell callers to preserve their scalar. An additive output makes the correct composition available to GPU and CPU consumers and gives the completeness certificate a precise scope. I recommend the output plus the four-node regression.
2. **Capture raster stream settings at reset.** An additive host settings object is the cheapest way to make ordinary uploads safe. If independent graph stages can write the control buffer, state also needs a GPU-visible representation identifier or stored exponents to detect incompatible continuation. Start with the host object and the two-sample oracle.
3. **Remove `reusePreparation` from the simulation operation form before alpha.** This is the one recommended public type break. Keep retention on the direct producer, whose setters, graph and owned buffers share a lifetime.
4. **Qualify the seven owners through failure as well as successful teardown.** Inject failure at each owned allocation, verify readiness and live-resource counts, then construct a fresh graph. Add a backing-replacement case for the centrality runner and a discarded-encoder case for the recorder.

## Not read

I read the shared operation bridge, snapshot-arena implementation, recording tap, preparation-control paths, and the relevant time-series packers and state updates. I sampled the seven producer owners, choosing ordinary/tiled reprojection as a sibling comparison, point density as the first-run journey, and network cost-table matching as a real prepared-result consumer. I checked the synthesis, area reports, API/docs catalog review, direction review and decision register against the candidate findings.

I did not inspect every producer body, every generated operator section, all solver convergence kernels, all recipe finalizers or every downstream ownership path. The 498-producer method census is exhaustive for that public runtime surface. The named behavioral departures are the ones established by this review. I excluded a readback-ticket cleanup issue already fixed in KI and a simulation operation shape issue corrected during the review. No GPU, browser, build, test runner or network was used. No files were written.