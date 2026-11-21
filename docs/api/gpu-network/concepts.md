---
title: "GPU Network concepts"
description: "Conventions and execution model of GPU Network contributors."
---


# GPU Network concepts


## Spatial contexts and length units

Distance operators (`GPUNetworkSnapping`, `GPUMapMatching`, `GPUNetworkKFunction`,
`GPUNetworkLineGraph`, `GPUNetworkNoding`, `GPUNetworkIsochrones` and
`addDriveTimeCatchmentRecipe`) accept planar azimuthal-equidistant (AEQD) frames from
`createGPULocalFrame(...).spatialContext`, including `units: 'meters'`, and reject Lambert
equal-area frames. Project positions into the declared frame before passing them to an operator.
Where supported, longitude/latitude degrees with `metric: 'great-circle'` are an alternative.
`GPUEdgeBundling` declares no measure: lengths are fractions of its work box, so either local
frame is accepted.

| Contributor | Longitude/latitude and kilometers |
| --- | --- |
| `GPUNetworkSnapping` | Great-circle distances in the context's length unit; `units: 'kilometers'` selects kilometers. |
| `GPUMapMatching` | Great-circle distances and distance parameters in the context's length unit, including kilometers. |
| `GPUNetworkKFunction` | Great-circle snapping; weights, distance thresholds, network length, K values and envelope in the context's length unit, including kilometers. |
| `GPUNetworkLineGraph` | Great-circle turn bearings accept meters or kilometers; angles and caller-owned travel costs are unchanged. |
| `GPUNetworkNoding` | Planar only; snap tolerance and output lengths are in the context's length unit. Longitude/latitude is rejected. |
| `GPUNetworkIsochrones` | Raster is planar; buffer radius and extent are in the context's length unit. Cell outlines accept longitude/latitude, including kilometer contexts, but only index positions in degrees. |
| `addDriveTimeCatchmentRecipe` | Longitude/latitude with metric `'none'` or `'great-circle'` uses great-circle snapping in the context's length unit, including kilometers. Travel costs keep the units of `topology.weights`. |
| `GPUEdgeBundling` | Longitude/latitude selects a degree-aware work box; kilometer contexts are accepted without physical distance measurements. |

Geographic distance units default to meters. `sphereRadius`, when supplied, is in the context's
length unit. Kilometer inputs must use kilometer-valued radii, thresholds and distance weights;
positions remain longitude/latitude degrees. Map matching's default `routeSlack` is 50 in the
context's length unit; supply `0.05` for a 50-meter allowance in a kilometer context.

Great-circle snapping searches all edges regardless of `candidateCapacity`, in meters or
kilometers alike.

## Conventions

- **Graph views in, graph views out.** Inputs and outputs are `GraphDataView`, `GraphVectorView`, or
  `GraphTextureView` objects created on the target graph. Outputs are always caller-owned; scratch
  storage is graph transients that die with `compiled.destroy()`. Contributors never compile, encode,
  submit, or read back.
- **Per-frame values never recompile.** Viewports, thresholds, radii, observer positions, budgets,
  and region shapes are read from storage views, usually a `GPUParameterBuffer` that the
  application rewrites with `write()`. Lengths, capacities, grid sizes, and which optional views
  exist are compile-time topology; each prop's TSDoc says which category it belongs to.
- **Construction captures configuration.** `GPUFlowAggregation` captures the active zone
  variant's literals, including lattice bounds and grid dimensions for grid/hexagon zones,
  and its time-window, output and status maps. `GPUNetworkCoarsening` captures topology
  descriptors and nested compact-output, output and status maps. Mutating those configuration
  objects later does not replace the captured settings or bindings. Writes to the originally
  bound graph views remain visible on later submissions, including view-backed bounds and
  active grid dimensions.
- **Bounded results report overflow on the GPU.** Compact ID lists use `GPUCompactOutput`
  (`ids` and `status`: `count`, `overflow`, optional `requiredCount`). `status.count` is clamped to
  `ids.length` and can be an indirect draw instance count; `status.overflow` is rewritten every encoding.
- **Stable IDs.** Result IDs are the caller's `sourceIds[row]` (or tile IDs) when given and zero-based
  rows otherwise. Node and transient IDs are `${id}-<step>`, so two instances in one graph need
  different `id` props.
- **One import per buffer.** Import a buffer once (for example with `graph.importGPUData()` or
  `GPUParameterBuffer.importToGraph()`) and pass the returned view to every contributor that reads
  it, so the graph tracks hazards on one logical handle.
- **CSR convention.** Vertex `v` owns the half-open slot interval
  `[offsets[v], offsets[v+1])`. The first offset may be nonzero: slots before `offsets[0]`
  are unowned and must not be walked. Undirected graphs list both directions in the CSR.
  A reverse CSR gives "cost to reach" semantics. `GPUCOOToCSR` requires row-sorted COO; sort
  whole edge records stably by source, including costs and edge IDs, and sort the reversed records
  independently for the transpose. Preserve isolated vertices with an explicit vertex list and map
  application vertex IDs to dense rows before uint32 upload. The
  [host preparation and literal shortest-path fixture](./operators.md#prepare-directed-edges-before-routing)
  returns both layouts and both slot-to-edge-ID maps.
- **Network input depends on the operation.** Routing and connectivity contributors generally take
  `topology: GPUNetworkTopology` (`offsets`, `neighbors`, `weights`, optional `nodePositions`).
  Connectivity-only contributors accept `GPUNetworkAdjacency`, with optional weights. Rational access
  and facility optimization instead take travel-cost matrices and demand/supply or capacity arrays.
  Read each operation's input contract. A network CSR is not automatically valid spatial weights:
  spatial weights impose their own diagonal, neighbor-order and duplicate rules.
  Contributors that write a network (line graph, noding, induced subgraph) emit the same shape, so
  their output can feed the next contributor's `topology`. Before binding an induced subgraph's
  offsets and neighbors to an unchecked row walker, require
  `inducedTopology.status.overflow === 0` for that submission. On overflow, its offsets retain
  the required slot counts and can exceed the stored neighbor capacity. Check the flag again
  on each submission; a previous successful submission does not certify changed inputs.
- **Matrix axes and direction.** `GPUNetworkCostMatrix.result` borrows values, start-group rows,
  target columns and the physical trip direction together with convergence and horizon. Reversing
  traversal and transposing storage are different operations; retain the direction when adapting
  a matrix for catchment scoring. Use the public `addNetworkCatchmentAdapter` with an explicit
  facility axis and physical trips; the [directed composition example](./operators.md#gpunetworkcostmatrix)
  preserves those labels while transposing demand-major storage. For noded capacity, keep
  `noding.outputTopology` and bind it as
  matrix `nodedTopology`, retaining live node count and source build evidence. Search convergence
  alone does not certify that the input network was built completely.
- **Matrix score completeness.** Routing convergence certifies the search within its matrix
  horizon; it does not certify scores at a larger cutoff or repair input snapping/build loss.
  After certifying those inputs, pass the matrix's convergence word
  as `costsComplete` and its horizon as `costLimit` to downstream accessibility scoring, and
  require `status.complete` before treating the scores as complete. The scoring cutoff must
  fit within that horizon. `addNetworkDisruptionRecipe` retains its `baseline-routing` and
  `scenario-routing` status stages, adds `baseline-accessibility` and `scenario-accessibility`
  stages with their own `complete` words, adds the joint `accessibility` stage with their
  conjunction in `status.complete`, and exposes `status.baselineAccessibilityComplete` and
  `status.scenarioAccessibilityComplete`. Its accessibility delta is NaN unless both score
  certificates are nonzero; individual accessibility outputs retain their incumbent values.
  On retained baseline reuse, `baseline.costLimit` belongs to that baseline: a supplied view gives
  its effective build horizon, `null` means unbounded, and omission reuses `search.costLimit` and
  explicitly promises the same search configuration. Changing the scenario's live horizon cannot
  retroactively extend the retained baseline's guarantee.
  All four CPU routing families capture `costPrecision` at construction in named forms.
  Reachability, matrices and service areas default to `'float64'`; tables default to
  `'float32-per-edge'`. The selection applies before cutoffs, label tightness and predecessor
  witnesses. GPU named forms support only `'float32-per-edge'`.
  Noded routing combines search convergence with source build completeness. A retained noded
  baseline must keep its **own** `baseline.status.sourceComplete`; the current scenario's noding
  evidence cannot certify an older build. Routing-stage `complete` includes both words.
  Newly-unreachable comparisons use the common smaller baseline/scenario horizon: a baseline
  trip is comparable only if its finite cost lies within that common horizon. An unchanged trip
  costing 7 with horizons 9 and 4 is not newly unreachable; losing a trip costing 3 within both
  horizons is. Finite cost differences retain their independent meaning.

  Look up the joint `accessibility.status.complete` alongside both individual score words when
  consuming the joint-stage protocol. The joint word certifies their conjunction; a missing stage
  is missing evidence and must not be read as success. This checked lookup also refuses an older
  producer that only exposes the two individual stages:

```ts
import {addNetworkDisruptionRecipe} from '@luma-contributors/analysis/gpu-network';
import type {GPUCommandGraph} from '@luma.gl/graph';

declare const graph: GPUCommandGraph;
declare const props: Parameters<typeof addNetworkDisruptionRecipe>[1];
const comparison = addNetworkDisruptionRecipe(graph, props);
const stages = comparison.status.stages;
const baselineComplete = stages.find(stage => stage.stage === 'baseline-accessibility')?.status['complete'];
const scenarioComplete = stages.find(stage => stage.stage === 'scenario-accessibility')?.status['complete'];
const jointComplete = stages.find(stage => stage.stage === 'accessibility')?.status['complete'];
if (!baselineComplete || !scenarioComplete || !jointComplete) {
  throw new Error('Require individual and joint accessibility certificates from the producer');
}
// After submission, read these words successfully and require all three to be nonzero.
// Merely finding a graph view does not certify its GPU-written value.
```

- **Cost-table completeness.** When composing `GPUNetworkCostTable` with `GPUMapMatching`, forward
  `output.status.complete` as `routeTable.status.complete`. This one word is nonzero only after all
  searches converge without capacity overflow. The matcher uses A* when it is zero or omitted;
  an absent entry in an unfinished table never establishes that a route is impossible. Size lanes
  and rounds together: both the command count and the frontier storage binding must fit (see the
  [cost-table limits](./operators.md#gpunetworkcosttable)).
- **Output and status bags.** Most GPU-written views are fields of `output`, and their status words
  (`converged`, `iterationCount`, `overflow`, `count`, `requiredCount`, and prefixed forms such as
  `pairOverflow`) live in the bag's `status` member (`output.status.overflow`).
  `GPUNetworkAccessibility` is an exception: it uses top-level `status.complete`, beside its
  top-level `cumulative`, `gravity` and `catchment` result props. Graph start
  nodes are `startNodes`/`startCosts`/`startCount`, cost bands are `costBreaks`, iteration caps are
  `maximum*Iterations`, and an active count is `number | GraphDataView<'uint32'>` (a number is
  compiled in, a view is read every frame).
- **Shared contracts.** Four shapes mean the same on every analysis page:
  - **Status.** Every GPU-written status word sits in the `status` member of the bag that owns it:
    `output.status.overflow`, `output.status.converged`, and `output.status.code` for model fits;
    accessibility's owning bag is its top-level `status.complete`.
  - **Grid.** Raster and terrain contributors take `grid: GPURasterMetadata`, and their packers read
    the cell size from it (see [GPU Terrain conventions](../gpu-terrain/concepts.md#conventions)).
  - **Network.** Routing usually takes weighted `GPUNetworkTopology` (`offsets`, `neighbors`,
    `weights`, optional `nodePositions`); connectivity-only operations accept `GPUNetworkAdjacency`
    with optional weights. Matrix-based facility and rational-access operations take cost matrices
    plus demand/supply or capacity arrays instead. Reusing a network CSR as `GPUSpatialWeights`
    requires checking its invariants and choosing statistical influence separately from road
    travel cost (see below).
  - **Polygons.** Polygons use the four-level `GPUPolygonGeometryPort`: `kind: 'polygons'`,
    `positions`, `featureOffsets`, `polygonOffsets` and `ringOffsets`. Contributors with a single
    geometry take the same fields flat, without `kind` (`GPUPolygonGeometryFields`). See
    [Shared ports](../gpu-spatial-analysis/concepts.md#shared-ports-status-and-planning).


## Routing costs versus statistical influence

Parallel roads are valid routing input. Spatial weights require their own strictly ascending,
unique-neighbor rows and diagonal policy. The two arrays can share a type without sharing meaning:

```ts
import {computeNetworkReachabilityOnCPU} from '@luma-contributors/analysis/gpu-network';
import {computeSpatialWeightsValidationOnCPU} from '@luma-contributors/analysis/gpu-spatial-analysis';

const offsets = Uint32Array.of(0, 2, 2);
const neighbors = Uint32Array.of(1, 1); // parallel directed roads
const weights = Float32Array.of(2, 3); // seconds, not statistical influence
const routed = computeNetworkReachabilityOnCPU({topology: {offsets, neighbors, weights},
  startNodes: Uint32Array.of(0)});
console.log([...routed.costs]); // [0, 2]: cheaper parallel road costs 2 seconds
const validation = computeSpatialWeightsValidationOnCPU(offsets, neighbors, weights);
console.log([...validation.flags], [...validation.invalidCount]); // [4, 0], [1]: duplicate neighbor

// Explicit policy: collapse parallel directed roads by MINIMUM travel cost, remove self edges,
// and sort unique neighbors ascending. Separately choose binary statistical influence here.
const statisticalOffsets = Uint32Array.of(0, 1, 1);
const statisticalNeighbors = Uint32Array.of(1);
const minimumTravelCostsSeconds = Float32Array.of(2); // retain as distances/metadata if desired
const influence = Float32Array.of(1); // binary adjacency, NOT a cost weight
console.log(computeSpatialWeightsValidationOnCPU(
  statisticalOffsets, statisticalNeighbors, influence)); // flags [0,0], invalidCount [0]
```

Other justified influence rules include inverse minimum travel cost or row-standardized binary
adjacency; select and document that rule before binding weights to a statistical consumer. Retain
producer loss status when transforming a possibly incomplete CSR. Aggregating roads cannot restore
missing edges or certify an unfinished search.

## GPU execution model

How network contributors run on the device: the command graph they declare, the kernels it
compiles to, and the resources each pass reads and writes, drawn from a recorded real run.

{/* lens:execution-model */}

Inspect the recorded single-hop search, then contrast its rounds with workgroup-local chaining.

::lens{scenario=gpu-network.network-reachability rung=graph caption="Inspect the declared relax rounds, dependencies and retained observation buffers."}

::lens{scenario=gpu-network.network-reachability rung=kernel caption="Inspect atomic cost minima and the indirect frontier dispatch."}

::lens{scenario=gpu-network.network-reachability.chained rung=iterations caption="Chained rounds span multiple hops; queue stamps record global rounds, not the hops within them."}


## Traversal phases, limits and usable results

A search round is a frontier dispatch, not a hop radius or an edge examination. Workgroups may
chain local hops, and their scheduling affects partial progress. The numbers below are enforced
GPU bounds; smaller storage-binding or command-node limits can reject a configuration earlier.
`maximumRounds` in internal flow helpers is not a public network-wide budget.

| Route and phase | Unit and enforced limit | Flag scope and usable output after a stop |
| --- | --- | --- |
| Reachability: cost search | Per encoding, `maximumIterations` 1–1,024 rounds (default 64); `activeIterations` clamps this phase to 0..that bound. `localIterations` 1–64 local hops per workgroup (default 32). | Optional scalar `converged`/`iterationLimitReached` cover every lane and requested phase. Finite costs can be overestimates; infinity need not mean unreachable. Cost bands describe incumbents. |
| Reachability: optional predecessors | Independent `maximumTieIterations` 1–1,024 equal-cost level rounds (default 4), with the same local-hop setting. `activeIterations` does not clamp this phase. | Adds tie-phase completion to those scalar flags; `iterationCount` still counts only cost-search rounds. Unresolved plateau parents are `0xffffffff`; consult completion before treating them as roots. Duplicate accepted seeds publish tie level zero atomically; a strict tight predecessor from a lower-cost node still takes precedence, including for nonzero start costs. |
| Service areas: costs, then labels | Per encoding, `maximumIterations` 1–1,024 cost rounds (default 64), then `maximumLabelIterations` 1–1,024 label rounds (default min(maximumIterations, 32)). `activeIterations` clamps each independently; `localIterations` 1–64 (default 16). | Scalar flags cover all facilities and both phases. Costs, facility rows, counts and sums are incumbents until both phases finish. Labels use float32 tight-edge equality and lowest facility row. |
| Service areas: optional predecessors | After labels, independent `maximumTieIterations` 1–1,024 equal-cost level rounds (default 4); unaffected by `activeIterations`. | Extends the same flags to parent completion. Parents follow the winning facility row; an unresolved parent is not a certificate of a facility root. |
| Isochrones: optional search/allocation | With `startNodes`, uses Reachability's cost rounds (default 64) or ServiceAreas' cost and label rounds when `assignments` is supplied, each 1–1,024. With external costs there is no internal search. | Optional search flags describe only the internal search/allocation, not raster/compact capacity. Incumbent costs can draw too-small geometry. Forward external-cost completion separately. |
| Cost matrix / sparse cost table | Per lane batch, `maximumIterations` 1–1,024 cost rounds (default 64); local hops 1–64 (default 16). No predecessor phase. | Scalar convergence aggregates batches. Table `complete` additionally requires no entry overflow; `requiredCount` counts discovered entries after a search stop. A missing entry in an incomplete table is unresolved. |
| Neighborhood | Intended hop radius, `maximumHops` 1–1,024 (default 8); runtime `hops` is clamped to it. Final masks compare hop costs with the requested radius. | A bounded radius is the query definition, not failure to finish an unbounded search. Compact result capacity has independent count/overflow status. |
| Path extraction | Per target, `maximumPathLength` 1–65,536 vertex slots (default 1,024), plus caller-owned compact output capacity. | An overlong walk publishes zero rows for that target, `pathFound=0` and `pathCosts=Infinity`; optional `candidateOverflow` reports the bounded-walk loss. `predecessorsComplete=0` suppresses all targets with those same empty/not-found outcomes. Final compact capacity can independently clip stored rows of a found path: `overflow` reports that capacity loss and `count` is clamped to stored capacity. Read only `count` rows. |
| Yen K-shortest paths | Per query: `routeCount` 1–32; full-route pending pool `candidateCapacity` 1–4,096 (default 256); `maximumPathLength` 1–1,024 vertices (default 64). Every Yen round has independent shortest-search and equal-cost phases, each `maximumIterations` 1–1,024 rounds (default 64), at up to 64 local hops per workgroup. | Scalar batch flags aggregate queries. A shortest/tie interruption stops all still-active queries; candidate overflow stops its own query. Overlong spurs interrupt certification. Only `counts[query]` previously certified routes are usable, in cost order; global lexicographic tie order is not promised. `iterationCount` counts Yen rounds (at most routeCount). See [K-shortest paths](./k-shortest-paths.md). |
| Alternative routes | Reference route, at most 256 vertices and 32 routes/query. Per query per encoding: `maximumExpansions` 1–10,000,000 examined outgoing CSR slots (default 100,000, including rejected slots); `maximumFrontierSize` 1–65,536 simultaneously pending prefixes (default 1,024). | `complete[query]` certifies k routes or exhaustive search, independently per query. On either stop it is 0; the first `counts[query]` routes remain a certified prefix in cost then lexicographic edge-slot order. Neither budget is a hop count. |
| Point routing: scalable | Per query, `maximumRelaxations` 0–1,024 frontier rounds per distance phase and independently per equal-cost predecessor phase (default 64), with up to 64 local hops per workgroup per round. | Per-query status: 0 complete (including unreachable), 1 invalid, 2 exhausted. Incumbent finite routes have reconstructable parents but need not be optimal after a stop. A zero budget publishes direct same-edge costs only. |
| Point routing: reference | Per query, `maximumRelaxations` 0–65,536 serial turn scans (each scan visits the whole allowed-turn list; default edge count). Reference mode admits at most 128 base edges and 4,096 turns. | Same per-query status vocabulary; reference turn order controls relaxation progress and ties. A scan is not one scalable frontier round or one examined turn. A zero budget publishes direct same-edge costs only. |
| CSR visibility sampling | Network metrics use centrality phases: `maximumIterations` 1–1,024 per phase (default max(64, min(nodeCount, 256))), with actual rounds min(maximumIterations, nodeCount); search localIterations 1–64 (default 16), sigma/delta one hop per round. Auto mode selects all viewpoints only when a whole-lane layout is feasible, every adjacency/metric binding and allocation fits the device, and the composed recipe fits 32,768 command nodes. Otherwise auto searches for the largest feasible whole-batch sample up to `min(nodeCount, max(32, laneCount ?? 1))`, using systematically spaced pivots; an explicit laneCount must divide the selected pivot count. Explicit pivots/counts never silently fall back, and exact mode refuses unless every viewpoint fits all these constraints. Use [the public execution planner](./visibility-graph.md) to check the same plan before construction. | `status.approximation=1` for sampling independently of convergence; `converged`/`iterationLimitReached` aggregate pivot batches. Sampled betweenness scales by nodeCount/pivotCount (plus the undirected half factor); integration estimates reachable counts/depth sums from sampled pivots. Closeness/reachedCounts remain per pivot, observed maximum depth is a lower bound, and sampling has no distribution-free error guarantee. Disconnected graphs and vertex order can bias estimates. Incumbent metrics after exhaustion remain incomplete even with all sources. |
| K-function | Per shortest-path batch, `maximumIterations` 1–1,024 frontier rounds (default 64), `localIterations` 1–64 hops/workgroup/round (default 16). This applies across observed and simulation blocks; command and storage budgets also constrain batch dimensions. | `converged=1` only if every shortest-path batch converged; `iterationLimitReached` is its complement. Separate snapping candidate `overflow` reports snapping loss when candidateCapacity is supplied (the flag is then required). Search convergence alone does not certify the K estimate when snapping overflow is 1. |
| AnalyticsColumns: components | Optional `components.maximumIterations` 1–1,024 weak-component label-propagation iterations per encoding (default 32). Degree/in-degree metrics do not use this iterative phase. | Optional `components.status.converged` reports that the final propagation round made no label change. Labels are incumbent minimum labels until convergence; the iteration bound is not a requested hop radius. |
| Edge bundling | `maximumIterations` 1–64 compiled iterations per encoding (default 15); runtime parameter activeIterations is clamped to that bound (0 disables updates). Each enabled iteration clears, splats and updates the paths. | Executes the requested number of iterations; there is no convergence certificate. GPU gating skips compiled slots beyond activeIterations. Paths describe the requested smoothing work, not an optimizer that proved a fixpoint. |
| Centrality: direct / ScheduleStep | Per pivot batch: search, optional sigma and delta phases each use min(`maximumIterations`, nodeCount) rounds; maximumIterations 1–1,024 (default 64), localIterations 1–64 (default 16) for search; sigma/delta use one hop per round. Betweenness adds sigma/delta; closeness alone omits them. Direct graphs reject more than 32,768 command nodes. ScheduleStep contains one fixed-capacity lane batch. | Scalar flags aggregate pivot batches; incomplete phases publish incumbent contributions. `zeroWeightCycle` differs from round exhaustion. ScheduleStep sums betweenness and merges failure flags until reset; per-pivot closeness/reached fields are overwritten. |
| Centrality: continuation | Per pivot batch and separately per phase, `maximumIterations` 1–4,294,967,294 total rounds (default nodeCount), across windows. `roundsPerStep` 1–1,023 rounds/submission (default 64), independent of that total. | Persistent frontiers continue within a batch. `reset: false` accumulates totals and keeps iterationLimitReached sticky through a later successful batch; reset clears it. Per-pivot outputs are overwritten each batch. |
| Directed SCC | `maximumIterations` 1–4,096 state-machine steps/encoding (default 256), spanning trim, colour and backward-mark phases. | Exhaustion invalidates all labels, including previously resolved singletons. Scalar `iterationCount` counts executed state-machine steps; convergence covers all components. |
| Louvain, reference and sparse routes | `maximumLevels` 1–32 (default 8), with `maximumIterations` 1–64 move rounds per level (default 16). A sparse round may accept several disjoint moves. | Returned partition/modularity is an incumbent until stopping status certifies convergence. Level and round limits are distinct from shortest-path rounds. |
| Matching: candidate and route search | `candidateCount` 1–8 (default 8); A* keeps `routeNodeBudget` 8–128 distinct nodes per source-candidate search (default 64), also subject to the runtime route-distance horizon. | `candidateOverflow` is a boolean aggregate flag for grid, route-search or traversal-recovery loss. `routeOverflow` is the saturating count (maximum `0xffffffff`) of discovered nodes dropped from scoring-search tables; three dropped nodes set the flag to 1 and add 3 to this count. Grid/recovery loss can set the flag with routeOverflow 0. Traversal output capacity is separate: `traversedEdges.status.overflow` reports capacity loss, `traversedEdges.counts[t]` includes required unstored slots, and `traversedEdges.status.candidateOverflow` reports incomplete recovery. A full search table can overestimate or miss a transition. |
| Flow: reference / parallel successive shortest path | `maximumAugmentations` 0–65,536 augmentations per solve (defaults 1,024 reference, 8 parallel). Parallel searches each use `maximumRelaxationRounds` 1–8,192 sweeps including the final unchanged sweep (default min(nodeCount, 1,024)). | A finite request is fulfilled only when `unserved` is zero. `complete=1` also permits residual exhaustion below demand: capacity 1 and request 5 can publish value 1, unserved 4 and complete 1. Use the cut only when `cutCertified=1`; a fulfilled finite request may finish without a cut search. An unfinished search retains previously feasible flow with stopping status. |
| Flow: parallel uncosted / primal-dual | `maximumRelabels` (uncosted) or `maximumCostPhases` (costed primal-dual) 0–4,096 phases per solve (default 3); each search, forward discharge and excess return is bounded by `maximumRelaxationRounds` 1–8,192. The shared command budget further limits products. | Failed phases restore earlier feasible flow before publication. Phase exhaustion differs from residual-search exhaustion; inspect complete/converged and stopping status before claiming a maximum or minimum-cost solve. |
| Flow: optional path decomposition | Reference path writer supports 2–128 expanded vertices, at most 1,024 expanded edge slots, and `pathCapacity` 1–1,024 source-to-sink paths. | Decomposition storage overflow is independent of flow convergence. Only count paths are stored; cyclic circulation is omitted. |
| FIFO time routing / travel-time percentiles | Per departure, `maximumRelaxations` 0–65,536 transition sweeps (default expanded state count), at most 256 states and 4,096 transitions; piecewise arrival uses 2–256 samples/transition. Percentiles support at most 60 departures and 16 percentile levels. | Routing summary/status distinguishes invalid input from unfinished search; iterationCount counts sweeps including the final unchanged sweep. Percentile status is per departure [valid, converged]. Requesting status retains incumbent percentiles and reached counts, which require every departure to be valid and converged. Without status, any failed certificate suppresses all percentile travel times with NaN and clears reached counts. |
| Facility enumeration | At most 16 facilities and 64 demand rows. Total candidate space is at most 1,048,576 (facilityCount ** demandCount for capacitated p-median, otherwise 2 ** facilityCount). `maximumCandidates` 0–1,048,576 bounds the examined prefix per solve, default all candidates. | Prefix winners are incumbents unless all candidates were examined. The explicit prefix budget never increases the accepted total-space bound. |
| Rational access | At most 64 origin rows and 32 facility rows. `maximumCycles` 0–4,096 equalization cycles per solve (default 150); `initialLimitCycles` 0–4,096 controls initial restricted cycles (default 20). | Zero cycles still evaluate the initial assignment. Gap/stopping status determines equilibrium; completing a cycle need not move an integral assignment. |

FIFO routing and travel-time percentile GPU operation forms validate declared dimensions at
construction and require resolved state/transition row counts to match those declarations before
creating a contributor. Routing operation twins capture declarations, model options and nested
output/status maps at construction; later caller mutations do not replace those settings.
CPU twins retain their independent size and sweep contracts. Pure CPU
functions and operation forms reject unknown `costModel` and `waitingPolicy` values. Omitting
`costModel` selects `'affine-duration'`; `'piecewise-arrival'` requires `samplesPerTransition`
and supports the optional `'wait-until-first-sample'` waiting policy. Piecewise-only controls
cannot be supplied with the affine model.

Other bounded optimizers have their own certificates: [flow and decomposition](./operators.md)
may return feasible partial flow without a cut certificate, and path capacity is independent of
solver completion. FIFO/time routing and percentile aggregation distinguish invalid departures
from unfinished searches and may aggregate valid incumbents. Facility enumeration certifies
only the examined candidate prefix unless exhaustive; rational access can stop without an
integral shift and still evaluate its initial assignment. Matching distinguishes candidate loss,
route-search exhaustion and truncated output. Supply only certified sparse-table costs to a
consumer that interprets absence as unreachable; otherwise use its documented fallback.
