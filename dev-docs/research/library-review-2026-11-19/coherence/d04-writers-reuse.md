# Requested outputs, missing writers, and buffer reuse

## Summary

Of 498 public command-node producers, 356 accept at least one optional buffer or texture under `output`. I constructed a graph for every one of those 356 and found failures that the full-output examples conceal.  
`GPUIsolines` has the clearest wrong result. Dropping a segment sets `overflow` to 1, yet the CPU reports `polylineOverflow = 0` and the GPU graph omits its writer when stitching is omitted.  
Two spatial-weights failures propagate into another calculation. An unwritten loss flag can suppress a valid spatial lag, and unwritten distances can become the inputs to a Gaussian kernel.  
`GPUGeometryPredicates` accepts a request for its intersection count and returns **zero command nodes**. Asking for an unrelated result makes the count work.  
The staged terrain-flow fix passed all 252 destination checks. The existing writer test nevertheless accepts both a write to the wrong slice and a kernel that performs no store.  
The highest-value change is to make an accepted output a per-submission guarantee, then test the smallest request that promises it, with poisoned storage and a second, smaller result.

## Census

`R` denotes this checkout. `K` denotes the newer `luma-contributors-ki` copy for staging-owned files. Paths below use `S` for `modules/analysis/src`, `P` for `S/gpu-spatial-analysis`, and `T` for `modules/analysis/test`.

| Public producer family | Optional destination under `output` | `output` with no optional destination | Other prop shape |
|---|---:|---:|---:|
| gpu-spatial-analysis | 201 | 68 | 1 |
| raster-analysis | 56 | 26 | 4 |
| gpu-terrain | 37 | 9 | 0 |
| gpu-network | 31 | 8 | 2 |
| dataframe-analysis | 25 | 9 | 0 |
| pointcloud-analysis | 3 | 8 | 0 |
| analysis-core | 2 | 4 | 0 |
| geospatial-analysis | 1 | 0 | 1 |
| computation-record | 0 | 0 | 2 |
| **Total** | **356** | **132** | **10** |

**COUNT.** I used a read-only `node` script with the TypeScript compiler API. Starting at public entry-point barrels, it resolved exported aliases, selected class instance types with `getProperty('getCommandNodes')`, and walked the first constructor’s props through optional parents, unions, and array elements. It counted buffer and texture destinations once per class. This matters for inherited fields such as `output.weights.status.count`, which a text search of the producer misses.

The ten other shapes were `GPUSnapshotTap`, `GPUTransferMask`, `GPUSpatialJoinPrepared`, `GPUNetworkAccessibility`, `GPUNetworkAnalyticsColumns`, `GPURasterTexture`, `GPURasterStreamingAccumulator`, `GPURasterStandardizedIndexCalibration`, `GPURasterClimateRunAccumulator`, and `GPUPointBuffer`. I included `GPUNetworkAnalyticsColumns` in the findings despite its different spelling.

**EXECUTED.** I piped an in-memory esbuild bundle into `node --input-type=module -`, captured constructor props from the 3,101 configurations indexed at `R:T/analysis-core/kernel-audit/kernel-audit-cases.ts:70`, and called the real constructors and `getCommandNodes` on fresh audit graphs. For fixture shapes with up to seven optional keys I tried the powerset. Above seven I tried individual requests with required ports present, all outputs, and leave-one-out requests. I supplemented that with optional scalar ports absent from fixtures and direct chunk and union cases.

| Probe | Result |
|---|---|
| Main partial-request sweep | 18,928 distinct attempts. 13,638 graphs returned, 3,286 requests rejected, 2,004 screened out for resource cost. A rejection alone was not graded as a defect. |
| Follow-up coverage | Reached all 356 optional-output classes, including `GPUChunkedSpatialWeightsTransform`, `GPUChunkedSpatialWeightsTranspose`, `GPUChunkedSpatialWeightsAlgebra`, and `GPUTerrainTiledFlow`. This is class coverage, not every legal specialization. |
| Generated bindings | Scanned 10,801 distinct within-class compute-module texts across 365 classes for bindings reachable from the compute entry point. Four flags were the optional `fp64arithmetic` module uniform in `GPURasterReprojection`, `GPURasterTileReprojection`, `GPULagrangianFlowMap`, and `GPUFiniteTimeLyapunovExponent`. The wrapper deliberately leaves that unused uniform unbound at `R:S/analysis-core/kernels/wgsl-kernel-nodes.ts:352`. No additional supplied-but-inactive Metal binding defect established. |
| Shared scratch | Built 1,115 accepted pairs across 463 classes on the same graph with distinct instance IDs and destinations. The two apparent collisions traced to fixture setup reusing a helper ID or imported draw resource. No cross-instance scratch collision established by this probe. |
| Classification subsets | `GPUColorScale` and `GPUBivariateClassification` each supplied a writer for all seven nonempty subsets of their three data columns. Numerical values are a separate question under `KNOWN(DPC-122)`. |
| Terrain-flow subsets | For `GPUTerrainFlow`, a 1×1 grid, all 31 nonempty data subsets, and both tested flat settings gave 252 destination checks. Two missed `filledElevation` in R, both when requested alone. **Zero missed in K.** See `R:S/gpu-terrain/hydrology/gpu-terrain-flow.ts:621`, `K:…/gpu-terrain-flow.ts:638`, and `KNOWN(RT-122)`. |
| Earlier spatial findings | Staged `GPUIslandAttachment` writes `count` and `uncertainCount` and rejects five unsupported words. Staged `GPULineSimplification` rejects orphan selection columns. I excluded those earlier findings and the already-recorded nearest-feature status and recipe-distance cases from the new counts. |

## Proposed convention

Supplying a destination means the producer writes its documented logical range on every submission, including zero-result and overflow routes, or rejects that destination at construction with the producer ID and prop path. A compact output names its live count and tail policy, and a published result descriptor exposes only columns the producer supplied.

Check each legal singleton and branch-changing subset with distinct subviews, then submit a nonempty or overflowing input followed by a smaller or empty input using the same storage. Check returned values, status words, and adjacent guards. A writable graph binding is a useful first screen.

## Findings

| ID | Kind | Reach | Change | Cost | Where | One line |
|---|---|---|---|---|---|---|
| WRT-01 | W | ORDINARY | ADDITIVE | S | Isolines, CPU and GPU | Omitting stitching loses the advertised combined overflow result. |
| WRT-02 | W | ORDINARY | ADDITIVE | S | Areal interpolation → spatial lag | A stale candidate-loss word can turn a valid lag into NaN. |
| WRT-03 | W | ORDINARY | ADDITIVE | M | Six spatial-weights producers | 42 accepted class/status combinations have no writer. |
| WRT-04 | W | ORDINARY | ADDITIVE | S | Contiguity → kernel weights | Published unwritten distances pass the next producer’s input check. |
| WRT-05 | W | ORDINARY | ADDITIVE | S | Geometry predicates | Intersection count and overflow alone produce an empty graph. |
| WRT-06 | W | ORDINARY | ADDITIVE | S | Track similarity search | Supplying both metric result pairs leaves half the outputs stale. |
| WRT-07 | C | ORDINARY | ADDITIVE | S | Raster profile | Adding one sample column silently reduces other columns’ capacity. |
| WRT-08 | C | UNUSUAL | ADDITIVE | S | Four scalar producers | Six scalar initializers clear padding in a caller’s longer view. |
| WRT-09 | T | ORDINARY | ADDITIVE | S | Terrain output-writer test | Both a sibling-slice write and a no-op dispatch pass the writer check. |
| WRT-10 | E | ORDINARY | ADDITIVE | S | Network analytics columns | An extent request forces allocation and computation of a normalized column. |
| WRT-11 | E | ORDINARY | ADDITIVE | S | Raster profile | A counts-only request still locates samples and reads elevations. |
| WRT-12 | E | ORDINARY | ADDITIVE | M | Adjacency matrix | A weighted maximum or texture forces publication of the whole weight matrix. |
| WRT-13 | C | UNUSUAL | BREAKING | S | Pick region mask | The output type offers two geographic counters that picking can never write. |

### WRT-01: `polylineOverflow` loses segment overflow when stitching is omitted

**Where.** `R:S/raster-analysis/isolines/gpu-isolines.ts:85,249,519,912`, `cpu-isolines-operation.ts:104,193,315`, and `gpu-isolines-operation.ts:112`. The public word is defined as 1 when *segments or polylines* were dropped. The GPU operation wrapper accepts it independently.

**Evidence — EXECUTED.** `computeIsolinesOnCPU` on a 2×2 raster `[0,1,1,0]`, level `[0.5]`, extent `[0,0,2,2]`, and segment capacity 1 returned the following words in order.

| Request | `count` | `requiredCount` | `overflow` | `polylineOverflow` |
|---|---:|---:|---:|---:|
| No `vertexCapacity` | 1 | 2 | 1 | **0** |
| `vertexCapacity: 4` | 1 | 2 | 1 | **1** |
| Four zero pixels, no `vertexCapacity` | 0 | 0 | 0 | 0 |

On the direct GPU route, `output.status: {count, overflow, polylineOverflow}` without `output.polylines` returned five nodes and **zero references** to `polylineOverflow`.

**Wanted and GPU probe — SOURCE.** Publish the segment overflow into the combined word when stitching is absent. With stitching enabled, retain the combined result. Preload the word with 99 and submit the checkerboard followed by the flat raster on the same graph. Expected combined word `1 → 0`. Apply the same rule to the CPU helper and both operation forms.

**Callers, value, cost.** No caller edits. Segment-only users get a truthful reusable completeness indicator. S.

### WRT-02: An unwritten areal-interpolation flag can suppress a correct spatial lag

**Where.** `K:P/areal-interpolation/gpu-areal-interpolation.ts:115,375,457` and `K:P/spatial-weights/gpu-spatial-lag.ts:81`. The staged constructor checks that nested `overflow` aliases the actual top-level overflow. It still preserves nine other nested status destinations.

**Evidence — EXECUTED.** One source zone, one target zone, one raster cell, two offsets, one slot, the correct shared overflow word, and a separate `weights.status.candidateOverflow` produced **17 nodes and zero writes** to the candidate word. Constructing `GPUSpatialLag` with the same weights descriptor bound that exact word for reading in `spatial-lag-incomplete-1`. The downstream API explicitly promises NaN on either loss flag.

**Now.** The natural composition is `new GPUSpatialLag({weights, values, output})`. The caller has to remove a field accepted by the previous producer to make this composition reliable. The other independently accepted unwritten nested words are `count`, `requiredCount`, `uncertainCount`, `invalidCount`, `iterationCount`, `converged`, `iterationLimitReached`, and `approximation`.

**Wanted and GPU probe — SOURCE.** Set a requested `candidateOverflow` to zero on this route, which has no candidate truncation stage. Route supported count aliases explicitly and reject unsupported diagnostic destinations. Seed candidate overflow with 1, assign the sole cell to source and target zone 0, and use source value 7. Expect candidate overflow 0 and lag 7. The present source predicts NaN. This downstream suppression is additional to the generic output-coverage issue in `KNOWN(KI-01)`.

**Callers, value, cost.** No edits for supported calls. Fixing the candidate word alone makes an ordinary interpolation-to-lag pipeline safe. S.

### WRT-03: Six weights producers accept seven uncomputed status words apiece

**Where and evidence — EXECUTED.** `K:S/analysis-core/contracts/ports.ts:230` gives the shared status object a broad optional shape. The intersection at `K:P/spatial-weights/spatial-weights.ts:251` retains those fields. I requested `count`, `uncertainCount`, `invalidCount`, `iterationCount`, `converged`, `iterationLimitReached`, and `approximation` individually, alongside each producer’s required destinations. **All 42 requests were accepted without a writer for the added word.** Paths below are under `P/spatial-weights`.

| Producer | Copy and location | Nodes in tested graph |
|---|---|---:|
| `GPUContiguityWeights` | `K:gpu-contiguity-weights.ts:265,380` | 99 |
| `GPULatticeWeights` | `K:gpu-lattice-weights.ts:183,237` | 13 |
| `GPUSpatialWeightsAlgebra` | `K:gpu-spatial-weights-algebra.ts:515,664` | 21 |
| `GPUChunkedSpatialWeightsAlgebra` | `K:gpu-chunked-spatial-weights-algebra.ts:216,270,375`, first output chunk | 30 |
| `GPUFuzzyContiguity` | `R:gpu-fuzzy-contiguity.ts:153,405,463` | 123 |
| `GPUAnalysisContiguityStatistics` | `R:gpu-analysis-contiguity-statistics.ts:57,163,245`, nested `weights.status` | 51 |

**Now.** `output: {...csr, status: {overflow, count}}` looks like the bounded-result convention. Five of these producers also expose the supplied handle again through their result or chunk descriptor. This count excludes the previously identified nearest-feature producer.

**Wanted and GPU probe — SOURCE.** Publish the actual written slot count where defined. For chunked output, specify and test the chunk-local count. Reject diagnostic words the producer cannot compute, following staged `GPUIslandAttachment` at `K:P/spatial-weights/gpu-island-attachment.ts:195,763`. Start with a 2×1 rook lattice, no self links, capacity two, and `count` preloaded with 99. Expect two directed slots and `count = 2`. For an unsupported `converged` request, expect a constructor error naming the prop.

**Callers, value, cost.** No edits for callers using computed fields. Shared CSR consumers gain meaningful counts and cannot accidentally publish stale diagnostics. M for all six.

### WRT-04: Unwritten contiguity distances pass validation as Gaussian kernel inputs

**Where.** `K:P/spatial-weights/gpu-contiguity-weights.ts:131,265,337`, `spatial-weights.ts:319`, `gpu-spatial-weights-transform.ts:335,615`, and the composing `R:P/spatial-weights/gpu-analysis-contiguity-statistics.ts:163`. The named contiguity operation already omits this destination at `K:P/spatial-weights/gpu-contiguity-weights-operation.ts:61`.

**Evidence — EXECUTED.** For two edge-adjacent unit squares, two neighbor slots, and a supplied distance buffer `d`, `contiguity.outputWeights.distances === d` was `true`. The producer had zero writable resources for `d`. `new GPUSpatialWeightsTransform({weights: contiguity.outputWeights, operation: 'kernel', kernel: 'gaussian', bandwidth: 1, …})` succeeded and bound `d` in the kernel dispatch. The earlier KI spatial census identified the unwritten recipe column. The additional defect is that the **public result descriptor vouches for it to a numerical consumer**.

**Wanted and GPU probe — SOURCE.** Reject `output.distances` for contiguity and omit it from the published descriptor. A separately computed metric can still be attached deliberately. Run the two squares twice with identical geometry, first seeding `d` with `[0,0]`, then `[10,10]`. The current chain reads those different values as physical distances. The corrected request should fail before either dispatch.

**Callers, value, cost.** No change for callers using contiguity’s documented products. Prevents buffer history from determining statistical weights. S.

### WRT-05: Asking for the intersection count alone schedules no intersection work

**Where.** `K:P/geometry-predicates/gpu-geometry-predicates.ts:141,345,479,754`. The writer is inside the simplicity/ring route. This is separate from how dependent predicates interpret overflow in `KNOWN(D103)`.

**Evidence — EXECUTED.** A four-vertex line with `intersectionCapacity: 1` produced these results.

| Requested outputs | Nodes | Writers for requested status |
|---|---:|---|
| `status.intersectionCount` | 0 | 0 |
| `status.overflow` | 0 | 0 |
| Both words | 0 | 0 |
| `isSimple` and both words | 15 | One per word |

**Wanted and GPU probe — SOURCE.** Treat either intersection status request as a reason to schedule the intersection stage. Supply any private companion word internally and validate the capacity directly. For `[(0,0),(1,1),(0,1),(1,0)]`, preload both words with 99 and expect `[intersectionCount, overflow] = [1,0]`. Resubmit a noncrossing open line and expect `[0,0]`.

**Callers, value, cost.** No caller edits. A GIS caller can count self-intersections without manufacturing a predicate column. S.

### WRT-06: Track similarity leaves the inactive metric pair looking current

**Where.** `K:P/track-similarity/gpu-track-similarity-search.ts:105,166,176,398,493`. All four destinations are independently optional in the public type.

**Evidence — EXECUTED.** With both pairs supplied, writer counts in `[distance, lowerBound, similarity, upperBound]` order were **`[1,1,0,0]`** for Fréchet and **`[0,0,1,1]`** for longest common subsequence. The constructor validates the active pair and accepts the inactive pair. A caller can reuse `output` while switching `metric` and read a result from the preceding calculation.

**Wanted and GPU probe — SOURCE.** Reject a distance pair on the similarity route and a similarity pair on distance routes, naming the inactive prop and selected metric. Use two identical one-vertex tracks and thresholds that admit an exact match. The active values should be Fréchet distance 0 and LCSS similarity 1. Passing both pairs should fail at construction.

**Callers, value, cost.** No changes for requests using the applicable pair. The harbor example already selects it at `examples/deck/spatial-analysis-showcase/src/scenes/trajectories/harbor-track-similarity.compute.ts` under its metric branch. Prevents stale scores on method switches. S.

### WRT-07: Raster-profile capacity depends on which optional column is present

**Where.** `R:S/raster-analysis/raster-sampling/gpu-raster-profile.ts:74,129,222,548,636,737`; user-facing output list at `K:docs/api/gpu-raster/operations-analysis.md:1171`.

**Evidence — EXECUTED.** With inferred capacity, `sampleValues.length = 3` gave three lanes. Adding `samplePositions.length = 1` reduced the value dispatch to **one** lane. Reversing the widths produced `raster-profile output.sampleValues must hold at least 3 rows`. Explicit `sampleCapacity: 1` with a three-row value view also succeeded.

**Now.** The page advertises “Outputs (any subset).” A plausible readback is `Array.from(sampleValues).filter(Number.isFinite)`. If the allocation held three finite values on its last use and the new capacity is one, that code can return two old elevations. The source distinguishes the live prefix, the range up to dispatch capacity, and the rest of the caller’s view. Sentinel behavior also differs by column.

**Wanted and GPU probe — SOURCE.** Infer the minimum supplied sample-column length, validate an explicit capacity against every column, and show `sampleValues.subarray(0, count)` on the page. State that offsets and `count` delimit live samples and that rows beyond capacity retain caller storage. Sample a line of length two at spacing one into three rows, then reuse the buffer at capacity one. Expect `count = 1` and exactly one consumable value.

**Callers, value, cost.** No edits for previously successful shapes. The minimum rule also accepts the reversed widths that currently throw. This is a concrete inference and documentation issue beyond `KNOWN(GEO-127)` on compact tails. S.

### WRT-08: Six scalar initializers erase a second word in a longer status view

**Where and evidence — EXECUTED.** I replaced a valid one-word destination with a two-word `uint32` view and inspected the emitted fill record. These scalar fills had `invocationCount = 2`. The shared fill defaults to the full view at `R:S/analysis-core/kernels/wgsl-kernel-nodes.ts:1096,1119`.

| Producer | Scalar destinations | Location |
|---|---|---|
| `GPUCellCover` | `status.invalidCount`, `status.uncertainCount` | `R:P/cell-cover/gpu-cell-cover.ts:686`, `s2-cell-cover.ts:125` |
| `GPUGeometryMeasures` | `status.invalidCount` | `K:P/geometry-measures/gpu-geometry-measures.ts:559` |
| `GPUCoverageValidity` | `status.invalidCount` | `R:P/polygon-coverage-topology/gpu-coverage-validity.ts:593` |
| `GPUAdjacencyMatrix` | `maximumCount`, `maximumWeightSum` | `R:S/gpu-network/adjacency-matrix/gpu-adjacency-matrix.ts:334,349` |

**Now.** `status: {invalidCount: twoWordView}` passes validation. The extra zero store is inside the supplied view, but outside the advertised one-word result. Four additional R hits disappeared in the newer copies: `GPURipleyEnvelope`, `GPUPeriodIndex`, `GPUCellPeriodKeys`, and `GPUValueCube`. They are excluded above.

**Wanted and GPU probe — SOURCE.** Pass `componentCount: 1` for these scalar fills. Initialize each view to `[99, 0x12345678]`. After an empty or clean input, require the correct scalar and the unchanged second word. Keep full-length fills for actual column outputs.

**Callers, value, cost.** No caller edits. Makes scalar ports predictable in shared readback layouts. S.

### WRT-09: The writer regression test passes two demonstrably ineffective writers

**Where.** `R:T/gpu-terrain/hydrology/gpu-terrain-flow-output-writers.node.spec.ts:84` and `R:S/analysis-core/kernels/wgsl-kernel-nodes.ts:438,478`. This strengthens the test used for `KNOWN(RT-122)` and the broader recommendation in `KNOWN(KI-01)`.

**Evidence — EXECUTED.** The test’s resource matcher returned `true` for both of these controls:

1. The requested `uint32` occupies byte offset 4. A fill writes only the sibling word at offset 0 on the same buffer.
2. The exact requested slice is bound `read_write`. Its kernel body is `return;`.

The wrapper can mention a binding to keep it active without performing a store. The two controls expose separate false passes. The existing terrain fixture uses distinct buffers, so the first control does not invalidate the staged terrain result above.

**Wanted and GPU probe — SOURCE.** Compare byte ranges in the host check where available. Keep the two controls and supplement the host gate with a numerical test. Initialize adjacent words to `[17,99]`, request the second word, submit a case with a known answer of zero, and require `[17,0]`. Submit an overflowing case followed by a clean case on the same buffers. Either ineffective writer must fail the numerical assertion.

**Callers, value, cost.** Test changes only. Prevents a regression test from certifying a destination that retains arbitrary bytes. S.

### WRT-10: Network degree extents require an unnecessary normalized column

**Where.** `R:S/gpu-network/network-analysis/gpu-network-analytics-columns.ts:34,193,256,266,274`; existing rejection test at `R:T/gpu-network/network-analysis/gpu-network-analytics-columns.node.spec.ts:165`.

**Evidence — EXECUTED.** `degree: {output: raw, extent}` threw `degree.extent requires degree.normalized`. `inDegree` behaved identically. Supplying `normalized` produced the degree kernel, extent reduction and finalization, then normalization. The desired extent is already available before that last dispatch.

**Wanted and GPU probe — SOURCE.** Compute the extent when `metric.extent || metric.normalized`. Schedule normalization only for `metric.normalized`. Use a directed two-node graph with edge `0 → 1` and its reverse CSR. Expect out-degrees `[1,0]`, in-degrees `[0,1]`, and both extents `[0,1]`, with no normalized destination or dispatch.

**Callers, value, cost.** No application edits. Update the one test that requires rejection. A styling or legend caller can request the two numbers it needs. S.

### WRT-11: A counts-only raster profile still samples elevations

**Where.** `R:S/raster-analysis/raster-sampling/gpu-raster-profile.ts:325,377,548,630,696`.

**Evidence — EXECUTED.** A 1×1 raster, one two-vertex line, `sampleCapacity: 1`, and `output: {status: {count, overflow}}` produced **nine nodes**, including `-locate` and `-sample`. The latter bound the raster values. The plan reserves private positions and values at eight and four bytes per sample respectively. Neither returned word depends on an elevation.

**Wanted and GPU probe — SOURCE.** Keep the geometric length and count work. Omit position generation and raster sampling when only status is requested, and reflect that in storage preflight. A line of length two, spacing one, capacity one should give `count = 1`, `overflow = 1`, and, if requested, `requiredCount = 3`. Repeat with the raster changed from finite values to NaN. The counts must agree, and the status-only graph should have no raster sample dispatch.

**Callers, value, cost.** No edits. A sizing pass avoids two dispatches and capacity-proportional scratch. S.

### WRT-12: An adjacency-matrix scalar or texture requires publishing the entire weighted matrix

**Where.** `R:S/gpu-network/adjacency-matrix/gpu-adjacency-matrix.ts:151,282,285,511,525,550`. Both `maximumWeightSum` and `textureStatistic: 'weightSum'` depend on weighted bins.

**Evidence — EXECUTED.** With edge weights present, `output: {counts, maximumWeightSum}` threw `adjacency-matrix output.maximumWeightSum requires output.weightSums`. The weighted texture route imposes the same public-matrix dependency. A caller after a scalar must supply a persistent `resolution²`-word destination.

**Wanted and GPU probe — SOURCE.** Use `output.weightSums` when supplied. Otherwise allocate graph-transient bins if a weighted maximum or texture needs them, and share those bins between both consumers. For one directed edge of weight 3, resolution one, and `weightScale: 1`, request `counts` and `maximumWeightSum`. Expect `[1]` and `3`. Request the weighted texture without the public sums and expect texel `3`.

**Callers, value, cost.** No caller edits. Removes an unnecessary ownership requirement while accounting for the scratch the algorithm actually needs. M.

### WRT-13: The picking-only mask type exposes geographic diagnostics that can never apply

**Where.** `K:P/region-statistics/region-statistics-types.ts:135` and `K:P/region-statistics/gpu-pick-region-mask.ts:49,75,99`. Both `antimeridianUncertainCount` and `poleUncertainCount` come from `GPURegionMaskOutput`. Its shared documentation explicitly says nonpolygon selections leave them untouched.

**Evidence — EXECUTED.** A batch-filtered pick with mask, overflow, and both geographic counters produced three nodes and zero writers for either counter. Picking has no geographic-polygon route. The distinction is intentional enough that `R:T/gpu-spatial-analysis/spatial-autocorrelation/statistics-continuation.node.spec.ts:584` passes the same view as a mask and an inactive geographic counter and expects success.

**Wanted and GPU probe — SOURCE.** Give this producer `GPUPickRegionMaskOutput` with `mask` and `status: GPUOverflowStatusPort`. Reject either geographic counter if supplied. Do not implement the change by zero-filling the counters: an existing accepted request aliases one to the actual mask. For packed picks `[1,0,0,0]` and one output row, expect mask `[1]` and overflow 0. Adding an inapplicable counter should fail before dispatch.

**Callers, value, cost.** **BREAKING.** `rg -n 'new GPUPickRegionMask\(' --glob '*.ts'` found 17 textual sites including example strings. One explicit call requires migration, the alias test at line 584. The region-statistics composition, lenses, and deck examples already pass only overflow. Removes two impossible choices from autocomplete. S.

## Departures by name

| Departure | Public producers or functions |
|---|---|
| An accepted status or metric result can retain old data | `GPUIsolines`, `GPUIsolinesOperation`, `CPUIsolinesOperation`, `computeIsolinesOnCPU`, `GPUArealInterpolation`, `GPUGeometryPredicates`, `GPUTrackSimilaritySearch` |
| A shared weights shape advertises unwritten fields | `GPUContiguityWeights`, `GPULatticeWeights`, `GPUSpatialWeightsAlgebra`, `GPUChunkedSpatialWeightsAlgebra`, `GPUFuzzyContiguity`, `GPUAnalysisContiguityStatistics`; `GPUArealInterpolation` additionally accepts these on its nested CSR |
| A consumer gives stale data numerical meaning | `GPUSpatialLag` reads the supplied loss word; `GPUSpatialWeightsTransform` reads the supplied distances. Both have good reason to trust a producer’s published descriptor. |
| Scalar initialization touches padding | `GPUCellCover`, `GPUGeometryMeasures`, `GPUCoverageValidity`, `GPUAdjacencyMatrix` |
| One requested output changes capacity or forces unrelated publication or work | `GPURasterProfile`, `GPUNetworkAnalyticsColumns`, `GPUAdjacencyMatrix` |
| A mode-specific producer inherits impossible output fields | `GPUPickRegionMask` |

## Quick wins

- [ ] Fix the CPU and GPU segment-only isoline publishers. Use the 2×2 checkerboard and flat raster as a shared oracle.
- [ ] Write areal-interpolation `candidateOverflow = 0` and test an immediately downstream lag of 7.
- [ ] Schedule the intersection stage for `intersectionCount` or `overflow` alone.
- [ ] Reject inactive track-similarity destinations and unsupported contiguity distances at construction.
- [ ] Set the six scalar fill widths to one and assert a nonzero neighboring guard survives.
- [ ] Show count-bounded profile readback on the API page, infer the smallest supplied sample capacity, and prune the counts-only sampling dispatches.
- [ ] Allow degree extent without normalization. Add the wrong-slice and no-store controls to writer verification.

## Larger proposals

**Make spatial-weights producer outputs explicit before alpha.** Keep the common CSR descriptor for consumers. For producers, name the status words each algorithm actually computes and construct the downstream descriptor from those words. The staged island-attachment rejection and the narrower named contiguity operation provide local precedents. Start with the six classes in WRT-03 and the nested areal-interpolation weights. Implementing real counts and rejecting previously unwritten destinations is additive under this review’s grading. M.

**Use poisoned second submissions as the numerical output gate.** Extend the existing fixtures first for bounded outputs, optional scalar status, and descriptors passed directly to another producer. For each such route, record the expected live range, an adjacent guard, and a result that shrinks or clears on submission two. The host graph sweep can cover many more variants cheaply. The GPU gate should cover the cases where a writable binding, conditional store, or stale counter can fool it. M for the initial families.

**Narrow the picking-only type before publication.** WRT-13 has one known migration and an alias test that explains precisely why adding stores would be unsafe. The narrower signature is the cheaper change now. S.

## Not read

I read the public output shapes across all 498 producers. I read the picking-only implementation and terrain writer regression test in full, and traced the constructor, graph branches, publication, CPU return paths, and downstream reads cited above. I sampled user documentation at the isoline, raster-profile, network-column, and adjacency-matrix entry points. I checked candidates against the first-round synthesis, area reports, API/catalog and direction reviews, decision list, and relevant staged copies.

I did not hand-read every shader, execute a GPU or Metal pipeline, or establish every combination of chunk count, geometry union, device limit, and iteration budget. The broad sweep used fixture-defined modes and resource screens. The generated-binding scan cannot establish backend validation or actual stores. Likewise, distinct-instance host graphs do not prove scratch lifetime on a device. **EXECUTED** above refers to CPU values, constructor outcomes, and returned graph records. Every proposed numerical GPU check is **SOURCE**.