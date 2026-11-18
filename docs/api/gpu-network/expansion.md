# Network structure, routes and spatial topology extensions

These contributors declare work in a `GPUCommandGraph`. They do not submit GPU work or read
results back. Outputs are caller-owned packed `GraphDataView`s, and inputs remain GPU-resident.

## Edge betweenness

`GPUNetworkCentrality.output.edgeBetweenness` accepts a `float32` column with one row per
forward CSR edge slot. It can be requested independently of node betweenness. It uses the same
pivots, lane batching, f32 tight-edge test, iteration bounds and `betweennessScale` as node
betweenness. The existing convergence output covers both dependency phases. A value is the
sum of shortest-path dependencies through that directed edge. Parallel CSR edges are distinct
paths. For an undirected graph listing both directions, each directed slot equals the conventional unnormalized
undirected edge centrality when all nodes are pivots; if combining paired slots, divide their
sum by two. Zero-cost edges are excluded from tight-path counting, as in the original
node implementation. Preserve a slot-to-source-edge mapping if CSR sorting changes edge order.

## Directed components and undirected resilience

{/* lens:GPUNetworkStructure */}

::lens{scenario=gpu-network.network-structure rung=graph}

`GPUNetworkStructure` accepts `topology` with `offsets` and `neighbors`; weights are ignored.

| Output | Format and meaning |
| --- | --- |
| `strongComponents` | `uint32`, one per node, minimum node ID in the strongly connected component |
| `bridges` | `uint32`, one flag per CSR slot |
| `articulationPoints` | `uint32`, one flag per node |
| `biconnectedComponents` | `uint32`, one per slot, minimum slot in the edge block |

For resilience outputs, `reverseEdgeSlots` pairs each CSR slot with its exact reverse slot.
The caller must supply an involution with reversed endpoints. Pair parallel edges separately:
two parallel edges form a block and neither is a bridge. A self-loop pairs with itself and is
a singleton block. Invalid neighbor IDs are ignored; CSR offsets must remain monotone and
within capacity. Isolated nodes are separate SCCs and are not articulation points.

`strongComponents` uses an exact mutual-reachability kernel with O(V²) scratch and O(V(V+E))
work, for at most 1024 nodes and 65536 slots. For larger graphs, use
work, for at most 1024 nodes and 65536 slots. For larger directed graphs, use
`GPUNetworkStronglyConnectedComponents` below.

The bounds of 1024 nodes and 65536 slots apply to the resilience outputs and to the
SCC path. Resilience uses an iterative Tarjan traversal in one GPU invocation. These
are exact bounded subgraph implementations; large-graph throughput and a lower-level generic graph
adapter for resilience remain future optimization work.

## Large-graph strongly connected components

`GPUNetworkStronglyConnectedComponents` takes forward CSR `topology` and caller-owned
`output.strongComponents` (`uint32`, one minimum vertex ID per SCC member). It uses O(V)
scratch: GPU trimming, minimum-colour propagation and backward marking within each colour class.
It does not require reverse adjacency. Weights are ignored, invalid neighbour IDs are skipped,
self-loops leave singleton SCCs intact, and isolated vertices label themselves. Offsets must be
monotone and within slot capacity; unused slots beyond the final offset are ignored.

`maximumIterations` is a compile-time state-machine step budget, default 256, range 1–4096.
`output.status` requires three distinct caller-owned single-row `uint32` views: `converged`,
`iterationCount` (executed steps), and `iterationLimitReached`. Inspect convergence before using
labels: an exhausted budget publishes `0xffffffff` for **every** vertex, including resolved ones,
with `converged = 0` and `iterationLimitReached = 1`. Deep cycles, long DAGs and chains of SCCs
may need a larger budget. Finished vertex phases are GPU dispatch-gated; there is no readback.
All buffer shapes and the step budget are compile-time; there are no numeric per-frame props.

Node and CSR slot counts must fit uint32 indexing (maximum 4294967294) and each packed binding
must fit the device's storage binding limit. Two vertex-sized scratch columns replace the exact
reference's quadratic reachability table. `computeNetworkStructureOnCPU({offsets, neighbors})`
is the independent iterative Tarjan reference for the same minimum-ID label layout; it has no
GPU step budget by default. Its optional second argument `{maximumIterations}` selects budgeted
SCC phases and returns `strongComponentsStatus` beside the labels. `CPUNetworkStructureOperation`
also accepts `strongComponentsStatus` and returns every requested scalar. Exact SCCs report
`converged=1`, `iterationCount=1` (one Tarjan traversal), and `iterationLimitReached=0`. When the
caller explicitly selects parallel SCCs (or supplies auto/budget controls selecting the large-graph
route), the CPU operation uses the trim/colour/backward/finalize state machine with that step budget.
With no mode or budget supplied, CPU SCCs remain exact Tarjan at every accepted size; a 1025-vertex
chain does not acquire the GPU default budget just by crossing the GPU exact-route limit.
The new class keeps this iterative SCC contract separate from
`GPUNetworkStructure`'s bounded resilience outputs. The algorithm is adapted from the user's
luma.gl fork at `e0ed33791`, independently implemented for WebGPU and inspired by cuGraph.
## Large-graph directed components

{/* lens:GPUNetworkStronglyConnectedComponents */}

::lens{scenario=gpu-network.network-strongly-connected-components rung=result caption="The cycle 0↔1 shares label 0; the outgoing singleton and isolated node keep labels 2 and 3."}

::lens{scenario=gpu-network.network-strongly-connected-components rung=graph caption="Follow GPU-gated trim, colour propagation, backward marking and convergence publication."}

The result selects the four `strongComponents` labels as its primary output. The full-size
table retains `converged`, `iterationCount` and `iterationLimitReached` in their separate
one-row status domain; the thumbnail summarizes the labels. The graph rung shows the command
graph separately. CSR inputs and all status readbacks remain in the record.

`GPUNetworkStronglyConnectedComponents` takes forward CSR `topology`; weights are ignored.
It uses O(V) scratch and vertex-parallel outgoing-neighbour visits, with GPU gates selecting
trim, minimum-colour propagation, backward marking and component finalization. Each exact
component label is its minimum vertex ID. Invalid neighbors and unused CSR slots are ignored;
isolated vertices and self-loops are supported. CSR offsets must be monotone and within capacity.

Caller-owned `output.strongComponents` holds one `uint32` per vertex. All three scalar `uint32`
views in `output.status` are required: `converged`, `iterationCount`, `iterationLimitReached`.
`maximumIterations` is a compiled state-machine step budget (default 256, range 1–4096), not
an edge-traversal count. Deep cycles, long chains and chains of components can exhaust it.
On exhaustion every label is `0xffffffff`, including labels resolved earlier; `converged` is 0
and `iterationLimitReached` is 1. Inspect status before using labels. `iterationCount` counts
executed state-machine steps, and finished vertex passes are gated off. The graph contains
`2 + 6 * maximumIterations` command nodes; storage bindings must fit the device. The old
1024-vertex bound of `GPUNetworkStructure` does not apply to this contributor.

The synthetic lens includes a two-node cycle, an outgoing singleton with a self-loop, an
isolated vertex and an unused CSR slot. Its hand-derived labels are `[0, 0, 2, 3]`. This is a
correctness illustration, not a large-graph throughput measurement. Its recorded result is
pending the GPU capture run (`VITE_CAPTURE_LENS_RECORDS=1`); no result is fabricated here.

## Alternative routes

{/* lens:GPUAlternativeRoutes */}

::lens{scenario=gpu-network.alternative-routes rung=graph}

`GPUAlternativeRoutes` accepts weighted `topology`, equally sized `sources` and `targets`,
`routeCount` (1–32), and `maximumExpansions` (default 100000). It returns sorted simple edge
paths: `vertices` and `edges` are query × routeCount × nodeCount; `lengths` counts vertices;
`costs` is infinity for missing routes; `counts` and mandatory `complete` have one row per query.
Padding is `0xffffffff`. Distinct parallel edges define distinct routes. Ties follow
lexicographic CSR edge order. Source equal to destination has one zero-edge path.

Queries execute independently. A bounded min-heap frontier enumerates simple paths by
nonnegative accumulated cost. `maximumFrontierSize` bounds retained frontier entries;
`maximumExpansions` bounds popped states. Either exhausted budget leaves `complete` zero.
A zero flag means retained paths are candidates, not certified k-shortest routes. Inspect it
before reporting alternatives as optimal. See [Alternative routes](./alternative-routes.md)
for tie ordering, capacities and scratch-memory limits. Negative, NaN and infinite edge costs
and overflowed f32 accumulated costs are impassable.

## H3 identities and cell flows

`GPUH3CellIdentities` (spatial-analysis entry point) takes little-endian `uint32x2` cells and
`operation: 'directed-edges' | 'vertices'`. The output has six rows per input cell. Directed
edge slot `direction - 1` stores the H3 edge ID; the deleted pentagon direction is zero.
Vertex slots match H3 `cellToVertexes`, with pentagon slot five zero. Invalid cells produce
zeros. Vertex ownership is the numerically smallest incident cell. Edge and vertex IDs are
not cell IDs and cannot be passed to cell-only APIs. All key arithmetic is integer-based.

`addCellFlowTopologyRecipe` accepts distinct same-resolution cells, origin/destination row
indices, and nonnegative finite values. It retains the original pair table, sums all valid
incoming/outgoing flows, and separately sums immediate-neighbor directed-edge flows. Repeated
pairs are accumulated in input order. Nonneighbor and self flows remain in the pair table and
cell totals; no route is inferred. It also returns two- or three-vertex shared boundaries in
longitude/latitude, including H3 face distortion. Longitude wraps at the antimeridian and
coordinates have the existing f32 H3 precision limit, especially at resolutions 14–15.

`validity` and `status.invalidCount` expose rejected rows. Sum(outgoing) and sum(incoming) equal the
valid pair sum up to f32 accumulation error; the neighboring-edge sum may be smaller. The
current direct-scan recipe costs O(cellCount × pairCount); preaggregate large pair tables.

## LCSS trajectory similarity

`GPUTrackSimilarity.output.longestCommonSubsequence` is matched length divided by the shorter
densified track length. It lies in [0,1], with NaN for skipped/empty/invalid/too-long pairs and
the existing pair status flags. `lcssTolerance` is a nonnegative finite compile-time point
distance threshold (default 1), in spatial-context units. `lcssBandRadius` optionally restricts
eligible matches with the existing length-adjusted Sakoe–Chiba convention; unmatched vertices
remain allowed. Densification and `maximumFrechetVertexCount` apply as for DTW. Matching uses
an anti-diagonal GPU dynamic program, including carried boundary rows for long tracks.

## Facility allocation

`GPUBoundedFacilityAllocation` is exported by spatial analysis. `operation: 'p-center'`
enumerates all facility subsets and exactly minimizes the maximum nearest-facility cost in the
supplied f32 matrix. `operation: 'capacitated-p-median'` evaluates the same subsets using
indivisible, demand-order nearest-feasible assignment, then prefers fewer unassigned rows and
lower weighted cost. **The capacitated mode is a heuristic, and failure is not an infeasibility
certificate.** Change demand order explicitly to investigate sensitivity.

For example, two unit demands with costs `[1, 2]` and `[1, 100]`, and two facilities with
unit capacities, produce greedy cost 101 in that demand order. Exhaustive assignment gives
cost 3. The test suite checks this gap and the zero optimality certificate explicitly.

Inputs are demand-major dense `costs`, `demand`, `capacities`, `facilityCount`, and `selectCount`.
Bounds are 32 candidate facilities, 8 selected facilities and 65536 subsets. Capacitated mode
accepts 4096 demand rows. P-center accepts a dense cost matrix up to the device's storage binding
limit: `floor(maxStorageBufferBindingSize / (4 * facilityCount))` demand rows, also subject to
uint32 indexing. For 32 facilities and a 128 MiB binding this is 1,048,576 rows; this is a storage
limit, not a throughput guarantee.

P-center scores demand rows cooperatively within each subset and publishes assignments in
parallel. Both operations use hierarchical winner reductions with the original subset order as
the final tie-breaker. Capacity assignment remains sequential within each subset to preserve its
fixed demand-order behavior. Work still grows with `choose(facilityCount, selectCount)`; the
parallel implementation does not remove the cost of exhaustive subset search.
Costs must be nonnegative finite to be reachable. Demand and capacity must be nonnegative
finite. The p-center operation ignores capacities and uses unweighted maximum cost.

Outputs are selected flags, demand assignments (`0xffffffff` if unassigned), scalar objective,
and three status words: unassigned-row count, valid-input flag, global-optimum certificate.
The certificate is always zero for capacitated mode. Objective is infinity if any row is
unassigned or an input is invalid. Inspect both feasibility and status before consuming results.

## Verification and reference contracts

The focused suite `test/geospatial-expansion/network-expansion.spec.ts` runs real WebGPU kernels.
It compares H3 keys with h3-js 4.4 across all base cells and their first-level children plus
resolutions 0–15; checks directed SCCs and undirected parallel-edge blocks; verifies route order
and truncated-search flags; exercises edge-only centrality; checks p-center and capacities; and
crosses the 256-row LCSS workgroup strip boundary. Additional cases compare eight seeded
undirected graphs with NetworkX, check feasible and infeasible capacity outcomes, and compare
cell-flow boundaries at pentagons and the antimeridian against H3 while checking flow conservation.
LCSS also uses a NetworkX longest-chain oracle over the DAG of eligible match pairs, independent
of the GPU dynamic-program recurrence. The node companion parses generated WGSL
and verifies the eight-storage-binding limit.

`test/oracles/network-expansion.py` generates independent fixtures using NetworkX 3.7 and
Python `itertools` exhaustive facility subsets and capacitated assignments. Reference conventions are unnormalized directed
edge betweenness, minimum-ID SCC labels, simple paths by total edge cost, and unweighted
p-center. These bounded correctness fixtures are not throughput benchmarks or proof of broad
numerical parity on arbitrary data. The capacitated heuristic has no approximation guarantee.

## `network-disruption-recipe`

{/* lens:addNetworkDisruptionRecipe */}

::lens{scenario=gpu-network.network-disruption-recipe rung=graph}

The executable synthetic directed-diamond scenario captures the workflow outputs. See [recipe contracts](../gpu-spatial-analysis/expansion-recipes.md) for matrix orientation, convergence and cutoff semantics.

## `facility-scenario-recipe`

{/* lens:addFacilityScenarioRecipe */}

::lens{scenario=gpu-network.facility-scenario-recipe rung=graph}

The executable synthetic directed-diamond scenario captures the workflow outputs. See [recipe contracts](../gpu-spatial-analysis/expansion-recipes.md) for matrix orientation, convergence and cutoff semantics.

## Track quality and route conformity

{/* lens:addTrackQualityRecipe */}

::lens{scenario=gpu-network.track-quality-recipe rung=result}

::lens{scenario=gpu-network.track-quality-recipe rung=graph}

`addTrackQualityRecipe` retains raw observation order and timestamps. The executable projected
four-observation example includes a repeated timestamp and a long time gap, and captures the
actual GPU quality flags, step speeds and track summaries. The two threshold rows are maximum
speed and maximum elapsed time. Optional reference-route similarity and map matching retain
separate failure evidence. See [the recipe contract](../gpu-spatial-analysis/expansion-recipes.md#track-quality-and-route-conformity)
for timestamp encodings, flag bits and bounded route-search semantics.
