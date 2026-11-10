# Network analysis code review, 2026-11-09

Domain: `modules/analysis/src/gpu-network` (81 files, 27,052 lines) plus
`analysis-core/network-topology.ts` (245 lines). Paths below are relative to `modules/analysis/src`.
Map matching, time routing and flow aggregation correctness were reviewed earlier in
`dev-docs/research/spatiotemporal-review-2026-11-09/code-movement.md` (cited as "movement review");
this report does not repeat that defect hunt.

## 1. Scope and method

**Read in full:** `analysis-core/network-topology.ts`; all of `network-optimization/` (six contributors
plus `prepare-flow-demands.ts`, 1,465 lines); the five root files (`gpu-network-inhomogeneous-k.ts`,
`gpu-network-kernel-density.ts`, `network-view-utils.ts`, `network-workgroup-counters.ts`,
`network-spatial-context.ts`); `network-structure/gpu-network-structure.ts`;
`network-reachability/gpu-network-reachability.ts`; `network-accessibility/gpu-network-cost-matrix.ts`
and `network-accessibility-lanes.ts`; `gpu-network/index.ts`, `network-point-process/index.ts`,
`network-optimization/index.ts`.

**Read in part:** `gpu-network-centrality.ts` (1-423 and the batch loop), `gpu-network-service-areas.ts`
(330-628), `gpu-network-k-function.ts` (block sizing and loops), `gpu-network-cost-table.ts` (lane
sizing, batch loop, finalize), `gpu-edge-bundling.ts` (iteration loop), `gpu-network-neighborhood.ts`
(composition), `gpu-network-statistics.ts` (class doc), `gpu-alternative-routes.ts` (props, caps,
dispatch), `gpu-network-snapping.ts` (candidate path), `recipes/network-scenarios-recipe.ts`,
`analysis-core/kernels/masked-weak-components.ts` (1-80). Class TSDoc of every exported class (33).

**Grepped only (not read):** the `*-passes.ts` files, `network-frontier.ts`,
`network-reachability-tuning.ts`, adjacency matrix, coarsening, line graph, noding, subgraph filter,
cost weights, accessibility, analytics columns, path extraction, isochrones bodies, map matching,
flow aggregation, drive-time and track-quality recipes.

**Published baseline checked:** `node_modules/@luma.gl/graph` 10.0.0-alpha.10 exports
`./gpu-graph-algorithms` (`GPUGraphAlgorithmsTopology`, `Degree`, `BreadthFirstSearch`,
`SingleSourceShortestPath`, `ConnectedComponents`, `CoreNumber`, `LabelPropagation`, `Modularity`,
`ModularityOptimization`, `LocalClusteringCoefficient`, `PageRank`, `ForceLayout`,
`SpatialForceLayout`) and, in `./gpu-core`, `GPUCOOToCSR`, `GPUGraphTraversal`, `GPUGraphSpMV`,
`GPUGraphCSRMatrix`. I read the `.d.ts` of topology, SSSP, BFS, connected components, modularity
optimization and PageRank.

## 2. Contributor table

Abbreviations: R = `maximumIterations`, T = `maximumTieIterations`, L = label iterations, B = batch
count, N = nodes, E = CSR slots. "views" = takes `GPUNetworkTopology`/`GPUNetworkAdjacency`.

| Symbol | File (lines) | Computes (reference) | Parallelism | Work | Iteration | Hard limits | Parameters | Inputs | Context | Status | Validation | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `GPUNetworkReachability` | network-reachability/gpu-network-reachability.ts (738) | multi-source SSSP, bands, predecessors (pgRouting drivingDistance) | row (queue entry), workgroup chains hops | frontier chaotic Bellman-Ford, atomic f32-bit min | unrolled R + T rounds, empty rounds dispatch 0 workgroups (indirect) | R, T ≤ 1,024 :53-59 design; N·lanes ≤ binding/8 inherent | buffer (`costLimit`, `activeIterations`, `startCount`, `costBreaks`) | packed | none (costs only; fine) | shared names (`converged`, `iterationCount`) | helpers + messages | conforming |
| `GPUNetworkCostMatrix` | network-accessibility/gpu-network-cost-matrix.ts (578) | many-to-all / many-to-targets matrix (Pandana, r5 OD) | lane-expanded reachability | B·reachability | B·(R+3) unrolled, no budget | lanes·N < 2^32 :235 | buffer | packed | none | `converged` | helpers | align (node budget) |
| `GPUNetworkCostTable` | network-cost-table/gpu-network-cost-table.ts (476) | bounded all-pairs table (UBODT, FMM) | lane-expanded reachability + per-lane compaction | O(N²/lanes) sweep per batch, O(N²) total | B·(R+~7) unrolled, B = ceil(N/lanes) :195 | entry capacity design | buffer | packed | none | `overflow`, `requiredCount`, `converged` | helpers | align (node budget, O(N²)) |
| `GPUNetworkCentrality` | network-centrality/gpu-network-centrality.ts (752) | Brandes betweenness, closeness, harmonic (NetworkX, cuGraph) | lane-expanded frontier | O(pivots·(N+E)·rounds) | unrolled, exact estimate :400-423 | ≤ 32,768 nodes :70 design (NW-CENT) | buffer (`costLimit`) | packed | none | `converged` | helpers | align (NW-CENT) |
| `GPUNetworkServiceAreas` | network-analysis/gpu-network-service-areas.ts (628) | network Voronoi, nearest facility (pgRouting, spopt eval) | frontier | reachability + tight-edge label phase | R + L + T unrolled | as reachability | buffer | packed | none | `converged` | helpers | conforming |
| `GPUNetworkNeighborhood` | network-analysis/gpu-network-neighborhood.ts (392) | k-hop ego network (NetworkX `ego_graph`) | frontier (unit weights) | as BFS | `maximumHops` rounds, per-frame `hops` | ≤ 1,024 | buffer | packed | none | compact output | helpers | conforming |
| `GPUNetworkPathExtraction` | network-analysis/gpu-network-path-extraction.ts (482) | predecessor walk to node/edge lists | row (target) | O(path length) per target + `GPUScan` | bounded walk | path ≤ 65,536 :37 | buffer | packed | none | compact, overflow | helpers | conforming |
| `GPUNetworkAnalyticsColumns` | network-analysis/gpu-network-analytics-columns.ts (369) | degree, in-degree, weak components | row | O(E) per pass | 2 nodes per CC iteration, default 32 | CC ≤ 1,024 iterations | baked counts | packed | none | `GPUIterationStatusPort` | helpers | merge (published Degree/CC; overlaps statistics) |
| `GPUNetworkStatistics` | network-statistics/gpu-network-statistics.ts (932) | counts, degree histograms, components, modularity (NetworkX) | row, load-balanced slot loops | O(E) | CC as above | u32 counts inherent | buffer (`GPUParameterBuffer`) | packed | none | header words | helpers | align (CC convergence) |
| `GPUNetworkStructure` | network-structure/gpu-network-structure.ts (263) | SCC, bridges, articulation, biconnected (NetworkX) | SCC: row with per-thread BFS; resilience: serial (1 thread) :189-191 | SCC O(V(V+E)), O(V²) scratch :113-118 | none (loops) | V ≤ 1,024, E ≤ 65,536 :66-67 design | n/a | packed | none | none | helpers, unprefixed messages | rework (FB-trim SCC; Tarjan-Vishkin) |
| `GPUAlternativeRoutes` | alternative-routes/gpu-alternative-routes.ts (275) | k shortest simple paths (Yen-like UCS, NetworkX `shortest_simple_paths`) | row (one thread per query, heap in storage) | exponential worst case | bounded expansions | N ≤ 256, k ≤ 32 :68-69 design | baked | packed | none | `complete` per query | helpers | rework (NW-KSP) |
| `GPUNetworkSnapping` | network-accessibility/gpu-network-snapping.ts (775) | point-to-edge snap, start costs (pgRouting withPoints) | row | default O(P·E) brute force :151-153, :535; BVH with `candidateCapacity` | none | binding limits | buffer (`maximumSnapDistance`) | packed | spatialContext | overflow | helpers | align (index by default) |
| `GPUNetworkAccessibility` | network-accessibility/gpu-network-accessibility.ts (352) | cumulative, gravity, 2SFCA/3SFCA on a matrix (Pandana, access) | row | O(rows·cols) | none | none | baked mode, buffer thresholds | packed | none | ad hoc | throw | conforming |
| `GPUNetworkCostWeights` | network-cost-weights/gpu-network-cost-weights.ts (531) | matrix → `GPUSpatialWeights` with decay | row + `GPUScan` | O(rows·cols) | none | capacity | baked decay kind | packed | none | `overflow`, `requiredCount` | helpers | conforming |
| `GPUNetworkIsochrones` | network-isochrones/gpu-network-isochrones.ts (823) | isochrone polygons via splat + `GPUIsobands` / cell outline | row (edge sample) | O(E·samples) | via reachability | samples ≤ 1,024 | buffer | packed | spatialContext | `converged` | helpers | conforming |
| `GPUNetworkKFunction` | network-k-function/gpu-network-k-function.ts (986) | network K + envelope (spaghetti `GlobalAutoK`) | lane-expanded reachability | (1+S)·n searches | blocks·(R+~6) unrolled, no budget | rowsPerBlock default 32 :351-364 | buffer | packed | spatialContext | `overflow` | helpers | align (node budget) |
| `GPUNetworkKernelDensity` | gpu-network-kernel-density.ts (159) | triangular network KDE (spatstat.linnet `density.lpp`) | row | O(Q·n) on dense matrix | none | Q·n ≤ 2^24 :61 | baked `bandwidth` :95 | packed | none | per-row 0/1 | bare assert | align (file location, bandwidth) |
| `GPUNetworkInhomogeneousK` | gpu-network-inhomogeneous-k.ts (280) | `linearKinhom` with Ang correction | normalize serial :135; pairs row; finish per radius | Ang O(n²·(V+E)) :204-223; finish O(radii·n²) with ≤256 threads :239,:256 | none | n ≤ 1,024, radii ≤ 256 :81-86 design | buffer radii | packed | none | per-radius bitmask | bare assert | rework |
| `GPUNetworkLineGraph` | network-line-graph/gpu-network-line-graph.ts (817) | turn-cost edge-based graph | row + `GPUSort`/`GPUScan` | O(arcs) | none | arc capacity | buffer | packed | spatialContext | overflow | helpers | conforming |
| `GPUNetworkNoding` | network-noding/gpu-network-noding.ts (752) | lines → planar routable CSR (OSMnx/momepy) | row + `GPULineSplit`, `GPUSort` | O(n log n) | none | capacities | baked tolerance | packed | spatialContext (planar only) | overflow | helpers | conforming |
| `GPUNetworkCoarsening` | network-coarsening/gpu-network-coarsening.ts (1,083) | group summary graph, superedges | row + sort/segment | O(N+E) | none | groups ≤ 65,535 | baked scale | packed | none (centroids of `nodePositions` ignore lon/lat) | overflow | helpers | align (spatialContext) |
| `GPUNetworkSubgraphFilter` | network-subgraph-filter/gpu-network-subgraph-filter.ts (990) | attribute/time induced subgraph | row + `GPUScan` | O(N+E) | none | none | buffer | packed | none | counts, overflow | helpers | conforming |
| `GPUAdjacencyMatrix` | adjacency-matrix/gpu-adjacency-matrix.ts (594) | binned adjacency-matrix image | row per node (hub serializes) | O(E) | none | R² cells | buffer (`GPUParameterBuffer`) | packed | n/a | none | helpers | move (graph visualization) |
| `GPUAdjacencyMatrixOrder` | adjacency-matrix/adjacency-matrix-order.ts (190) | matrix ordering | `GPUSort` | O(N log N) | none | none | n/a | packed | n/a | none | helpers | move (graph visualization) |
| `GPUEdgeBundling` | edge-bundling/gpu-edge-bundling.ts (513) | KDEEB bundling of flow lines | row + atomics grid | O(iter·E·points) | 3 nodes per iteration, GPU gate, ≤ 64 | grid size | buffer | packed | spatialContext | overflow | helpers | conforming |
| `GPUMapMatching` | map-matching/gpu-map-matching.ts (1,925) | HMM map matching | row per point; Viterbi per track | private A* per candidate | none | see movement review §4 | baked | packed | spatialContext | per-row words | helpers | align (movement review; private edge grid) |
| `GPUFlowAggregation` | flow-aggregation/gpu-flow-aggregation.ts (1,031) | OD flows, zone totals | row + `GPUHashIndex`, `GPUSort` | O(n log n) | none | zones ≤ 65,535 | buffer | packed + chunked weights | none | shared words | helpers | conforming (movement review) |
| `GPUNetworkLouvain` | network-optimization/gpu-network-louvain.ts (208) | multilevel Louvain | serial, 1 invocation :100 | ~O(levels·rounds·n⁴) dense | in-kernel loops | n ≤ 32, dense n² weights :54,:75 design | buffer (resolution, tolerance) | packed | none | 4-word summary | bare assert | move (drop for published, NW04) |
| `GPUNetworkFlow` | network-optimization/gpu-network-flow.ts (253) | max flow / min-cost flow (SSP) | serial :125 | O(aug·V·E) | in-kernel, ≤ 65,536 augmentations :72-74 | V ≤ 128, E ≤ 1,024 :69-70 design | buffer (s, t, demand) | packed | none | 8-word summary | bare assert | rework (NW02) |
| `GPUNetworkFacilityOptimization` | network-optimization/gpu-network-facility-optimization.ts (316) | backup coverage, p-dispersion, capacitated p-median (spopt) | row over candidates, 65,536 per batch :128 | exhaustive: 2^F subsets or F^D assignments :70-74 | ≤ 16 batches | F ≤ 16 :60, D ≤ 64 :69, candidates ≤ 2^20 :74 design | baked `selectCount` | packed | none | 4-word status | bare assert | merge (location-allocation) |
| `GPUNetworkTimeRouting` | network-optimization/gpu-network-time-routing.ts (294) | FIFO earliest arrival on restriction states | serial :120 | O(rounds·transitions) | in-kernel | states ≤ 256, transitions ≤ 4,096 :58-61 design | baked counts | packed | none | 4-word summary | bare assert | rework (movement review) |
| `prepareGPUNetworkRestrictionStates` | same file :223-294 | restriction automaton | cpu | O(states·E·prefixes) | none | 256 states | n/a | n/a | n/a | n/a | bare assert | rework (with time routing) |
| `GPUNetworkPointRouting` | network-optimization/gpu-network-point-routing.ts (142) | edge-position routing with turns (pgRouting withPoints) | row per query, private `array<f32,128>` :109 | O(rounds·turns) per query | in-kernel | edges ≤ 128, turns ≤ 4,096 :50 design | n/a | packed | none | per-query code 0/1/2 | bare assert | merge (snapping + line graph + cost matrix) |
| `GPUNetworkRationalAccess` | network-optimization/gpu-network-rational-access.ts (188) | RAAM rational-agent access | serial :106 | O(cycles·O·F) | in-kernel ≤ 4,096 cycles | origins ≤ 64, facilities ≤ 32 :56-61 design | buffer | packed | none | 4-word status | bare assert | rework (and move to catchment-accessibility) |
| `prepareGPUNetworkFlowDemands` | network-optimization/prepare-flow-demands.ts (65) | super source/sink | cpu | O(E) | none | N ≤ 126 | n/a | n/a | n/a | n/a | bare assert | rework (with flow) |
| `addDriveTimeCatchmentRecipe` | recipes/drive-time-catchment-recipe.ts (807) | snap → service areas → isochrones → PIP | composes | | | | | | spatialContext | `GPURecipeStatusPort` | helpers | conforming |
| `addNetworkDisruptionRecipe` | recipes/network-scenarios-recipe.ts:45 | baseline vs closure matrices | composes 2 cost matrices | | | | | | none | ad hoc | throw | conforming |
| `addFacilityScenarioRecipe` | recipes/network-scenarios-recipe.ts:279 | facility closure coverage | row + serial summary :387 | O(D) serial sum | | | buffer | | none | ad hoc | throw | align (use `GPUReduction`) |
| `addTrackQualityRecipe` | recipes/track-quality-recipe.ts (206) | trajectory metrics + map matching | composes | | | | | | spatialContext | `GPURecipeStageStatusPort` | throw | conforming (movement domain) |
| `createNetworkAnalyticsTopology` | analysis-core/network-topology.ts:193-245 | views → `GPUGraphAlgorithmsTopology` | n/a | n/a | n/a | needs default buffers | n/a | packed | n/a | n/a | throw | move (upstream fix; see 3.1) |

Verdict counts (38 rows): conforming 15, align 10, rework 6 (counting the two CPU helpers with their
contributors), merge 3, move 4.

**On the two-generation hypothesis.** It holds for `network-optimization` and the two root
point-process files: terse WGSL, bare `assert`, single-invocation kernels, small caps. Counter-examples:
`GPUNetworkFacilityOptimization` is expansion-style yet row-parallel (65,536-wide batches with a
hierarchical reduction, :128-260); `GPUNetworkKernelDensity` is expansion-style and scales linearly
in its matrix. On the other side, `GPUNetworkStructure` and `GPUAlternativeRoutes` use the baseline
validation style (`validateCapacity`, messages) but are serial or capped at 1,024 and 256 vertices;
`GPUNetworkCostTable` and `GPUNetworkKFunction` are baseline contributors whose command-node count is
unbounded (section 3.2). Style does not predict scale here.

## 3. GPU quality findings

### 3.1 Answers to the five questions

**Q1. One representation or several.** One canonical input: CSR `GPUNetworkTopology`
(`offsets`, `neighbors`, `weights`, optional `nodePositions`; `analysis-core/network-topology.ts:29-47`),
always directed, undirected = both directions listed. 22 files consume it. Edge cost lives only in
`topology.weights` (f32 per slot; negative or NaN = impassable). **CONFIRMED.** Departures:

| Representation | Contributors |
| --- | --- |
| Edge list `uint32x2` | `GPUNetworkFlow`, `GPUNetworkInhomogeneousK`, `GPUNetworkTimeRouting` (`transitions`), `GPUNetworkPointRouting` (`turns`), `GPUEdgeBundling` (`sourceNodes`/`targetNodes`), map matching `edges` |
| Dense N×N matrix | `GPUNetworkLouvain` |
| Dense cost matrix (not a network) | facility optimization, rational access, KDE, inhomogeneous K, accessibility, cost weights |
| Line graph | built by `GPUNetworkLineGraph` as an ordinary CSR (good); point routing takes its own `turns` list instead |

Directedness is signalled four ways: always-directed CSR (reachability family), `directed?: boolean`
(statistics :190, subgraph filter :109, adjacency matrix :99, Louvain :24), presence of a reverse CSR
(analytics columns :70, centrality `reverseTopology` :136), and `reverseEdgeSlots` pairing (structure
:31). Private adjacency rebuilds: centrality packs both CSRs per encoding (`createCentralityPackNode`,
network-centrality-passes.ts:109), statistics builds a masked neighbor copy, map matching builds an
edge grid and edge table, structure keeps a per-thread BFS queue. Reverse CSRs are left to the caller
(docs point to `GPUCOOToCSR`); `GPUSpatialWeightsTranspose` in gpu-spatial-analysis does the same job.

**Q2. Shortest paths and iteration bounds.** One engine: frontier chaotic Bellman-Ford with atomic
minimum on f32 bits (network-reachability-passes.ts:336), workgroup-local hop chaining
(`localIterations`), and indirect dispatch so empty rounds launch zero workgroups (network-frontier.ts:25,
:380). Costs are the unique f32 fixpoint, so they are deterministic. It is not delta-stepping or
near-far; the class documents a decision rule (adopt bucketing only if measured inflation exceeds 3,
gpu-network-reachability.ts:220-231). Every round is still one command node, unrolled at compile time.
Only `GPUNetworkCentrality` computes its node count (`estimatedCommandNodeCount`, :400-423) and
rejects configurations over 32,768. **CONFIRMED** for the formulas below unless marked.

| Contributor | Compile-time command nodes (approximate) | Budget check |
| --- | --- | --- |
| `GPUNetworkReachability` | R + T + ≤ 10 (T only with predecessors); R, T ≤ 1,024 | caps only |
| `GPUNetworkCentrality` | fixed + B·(R'+2+[closeness] + [betweenness: 2R'+3+k]), R' = min(R, N) | yes, 32,768 |
| `GPUNetworkCostMatrix` | ≤ 3 + B·(R + 3 + [copy or gather]), B = ceil(rows/lanes) :238 | **none** |
| `GPUNetworkCostTable` | 1 + B·(R + ~7), B = ceil(N/lanes) :195 | **none** |
| `GPUNetworkKFunction` | ≈ blocks·(R + 6), blocks = ceil((1+S)·n / rowsPerBlock), rowsPerBlock ≤ 32 (doc :240-241, code :351-373) | **none** |
| `GPUNetworkServiceAreas` | R + L + [T + ~6] + ~8, L = min(R, 32) default | caps only |
| `GPUNetworkIsochrones`, drive-time recipe | reachability or service areas + raster nodes | caps only |
| `GPUNetworkNeighborhood` | maximumHops + ~6 | caps only |
| Statistics, analytics columns (`masked-weak-components.ts`) | 1 + 2·iterations (default 32, ≤ 1,024) | caps only |
| `GPUEdgeBundling` | 3·R + ~5, R ≤ 64, GPU gate | bounded by cap |
| `GPUNetworkFacilityOptimization` | 2 + ceil(candidates/65,536)·(2 + reduction) ≤ 16 batches | bounded |
| `addNetworkDisruptionRecipe` | 2 × cost matrix | none |

Worked example (arithmetic from the code, effect SUSPECTED): a 100,000-node network with 200,000 CSR
slots gets `recommendLaneCount` = min(max(32, 10), floor(128 MiB/4/500,000) = 67) = 32 lanes, so
`GPUNetworkCostTable` emits 3,125 batches × ~71 ≈ 220,000 nodes at the default R = 64, seven times the
centrality cap, although its own doc targets "roughly 1e5 nodes" (:350-354). `GPUNetworkKFunction`
with 500 events and 99 simulations: 50,000 rows / 32 = 1,563 blocks × ~70 ≈ 109,000 nodes.

**Q3. network-optimization.** All six keep the work in one kernel; five run one invocation. None takes
`GPUNetworkTopology`. 32 bare `assert` calls (Louvain 4, flow 6, facility 6, point routing 3, rational
access 3, time routing 10) plus 7 in `prepare-flow-demands.ts`; none has an explanatory comment.

| Contributor | Kernel parallelism today | GPU-scale formulation |
| --- | --- | --- |
| Louvain (:54, :100) | 1 thread; every candidate move re-evaluates the full O(n²) objective (:112-128, :175-189), about n⁴ per sweep × 128 sweeps. At n = 32 that is ~2.7e8 serial operations: watchdog risk SUSPECTED | Parallel Louvain/Leiden (cuGraph, Grappolo): all vertices pick best-gain community in parallel from per-community totals (segmented reduce by `(vertex, neighborCommunity)` after `GPUSort`), move-conflict rule (alternate ID parity or colour classes), aggregation by sort + reduce-by-key into a coarse CSR, repeat. Belongs in luma.gl next to `GPUGraphAlgorithmsModularityOptimization`, which today is single-level and accepts one move per synchronized round |
| Max / min-cost flow (:69-70, :125) | 1 thread successive shortest paths, Bellman-Ford O(V·E) per augmentation, up to 65,536 augmentations (:72-74): ~8.6e9 serial operations at the caps, watchdog risk SUSPECTED | Max flow: push-relabel with global relabel by reverse BFS (Vineet-Narayanan, He-Hong; Gunrock). Min-cost: SSP with Johnson potentials so reduced costs are non-negative, each augmentation one `GPUNetworkReachability` + a path kernel, or cost-scaling push-relabel. Needs a GPU-gated continuation (PLAT-04). Generic graph code: upstream |
| Facility optimization (:60, :69-74) | Row-parallel over candidates (good), but exhaustive. Capacitated p-median enumerates F^D assignments, so the real limit is F^D ≤ 2^20 (2 facilities → 20 demand rows; 4 → 10), not the documented 64 rows | Teitz-Bart interchange already exists in `GPULocationAllocation`; add Lagrangian relaxation (subgradient; each iteration row-parallel over demand) for capacitated p-median, greedy + interchange for backup coverage (BACOP) and p-dispersion. It is a cost-matrix operator and belongs in `gpu-spatial-analysis/location-allocation` |
| Time routing (:58-61, :120) | 1 thread Bellman-Ford over ≤ 256 states; automaton built on CPU | Time-dependent label-correcting on the reachability frontier: relax with `arrival(edge, t)` instead of `t + w` (FIFO keeps the fixpoint unique). Two-edge turn bans already come from `GPUNetworkLineGraph`. See movement review §4 |
| Rational access (:56-61, :106) | 1 thread, cycles × origins × facilities | Its update is already Jacobi-style ("parallel origins move simultaneously", :48): per cycle a column reduction for facility loads (`GPUReduction`/segmented sum), a row-parallel shift kernel, a facility inflow reduction; cycles unrolled with a GPU gate on the gap. Belongs with `catchment-accessibility` (2SFCA, Huff) |
| Point routing (:50, :89, :109) | One thread per query, private 128-entry array, Bellman-Ford over turns | Already composable: `GPUNetworkLineGraph` → `GPUNetworkCostMatrix` on the line graph with `startCosts = (1 - fraction)·cost`, then a gather that adds `targetFraction·cost`. Same-edge forward case is one extra kernel. Merge |

**Q4. Loose root files.** Every other top-level directory has only `index.ts` at its root
(`ls gpu-spatial-analysis/*.ts`, `gpu-terrain/*.ts`). `gpu-network` has five loose files.
`network-point-process/index.ts:5-11` contains no code and re-exports the two root contributors with
`../`; `gpu-network/index.ts:5` exports `network-optimization` above the comment that explains the
layout. Recommended rule: **a top-level directory holds only `index.ts`; each contributor family is a
subdirectory; helpers used by one family live in it; helpers used across families of one domain live
in one `network-shared/` subdirectory; helpers with nothing network-specific go to `analysis-core`.**
Applied:

| File | Used by | Move to |
| --- | --- | --- |
| `gpu-network-kernel-density.ts`, `gpu-network-inhomogeneous-k.ts` | exported only | `network-point-process/` (with `network-k-function/` as a sibling or merged in) |
| `network-workgroup-counters.ts` ("library rule 14", :34) | statistics, map matching, noding, subgraph filter, coarsening | `analysis-core/kernels/` (nothing network-specific) |
| `network-view-utils.ts` (`getSubView`, `alignElementCount`) | cost matrix, K-function, centrality | `analysis-core/kernels/` |
| `network-spatial-context.ts` (network metric WGSL) | snapping, map matching | `network-shared/` |

**Q5. Overlaps with gpu-spatial-analysis.** **CONFIRMED** by imports and class docs.

| gpu-spatial-analysis | gpu-network | Relation |
| --- | --- | --- |
| `location-allocation/GPULocationAllocation` (greedy + Teitz-Bart), `GPUBoundedFacilityAllocation` (exhaustive subsets, p-center, capacitated p-median) | `GPUNetworkFacilityOptimization` imports `facility-score-reduction` from there (:15-18) | Three facility solvers over the same cost matrix, two of them exhaustive with different capacitated p-median rules. Merge into location-allocation; network supplies only the matrix |
| `catchment-accessibility/GPUCatchmentAccessibility`, `GPUHuffTradeAreas` | `GPUNetworkAccessibility` composes `GPUNetworkCostWeights` + `GPUCatchmentAccessibility` | Good composition; one 2SFCA. `GPUNetworkRationalAccess` should join this family |
| `spatial-weights` (`GPUSpatialWeights`, transpose, algebra) | `GPUNetworkTopology` shares the CSR names (network-topology.ts:24-27); `GPUNetworkCostWeights` emits weights | Consistent by design. Two ways to build a reverse CSR (`GPUCOOToCSR` in docs, `GPUSpatialWeightsTranspose` in code) |
| `map-coloring/GPUMapColoring` (Jones-Plassmann on a CSR) | none | Generic graph colouring; works on a network CSR unchanged |
| `trajectory-restriction` | tested in `test/gpu-network/network-optimization/network-movement-advanced.spec.ts` | Test placement only; move the spec to `test/gpu-spatial-analysis/trajectory-restriction/` |
| `spatial-regionalization/gpu-spatial-weights-minimum-spanning-tree.ts` | none | The network MST practitioners expect (section 6) |

### 3.2 Findings

| # | Finding | Evidence | Why it matters | Better design | Effort |
| --- | --- | --- | --- | --- | --- |
| G1 | Cost table, cost matrix and K-function emit an unbounded number of unrolled nodes; only centrality has an estimate and a cap | cost-table :195, :268; cost-matrix :238, :399; k-function :351-373, :697 (CONFIRMED); compile-cost effect SUSPECTED | 1e5-node cost tables and 99-simulation K envelopes compile 1e5+ nodes | Short term: hoist `estimatedCommandNodeCount` and the 32,768 guard into one shared helper used by every reachability composer. Structural: PLAT-04 (GPU-gated continuation) | S, then upstream |
| G2 | network-optimization runs whole solvers on one thread | louvain :100, flow :125, time :120, rational :106; point routing per query | Not GPU work; flow and Louvain can hit the GPU watchdog at their own caps (SUSPECTED) | Table in Q3 | M to L each |
| G3 | Published graph algorithms are bypassed because they need physical `GPUVector`s and a `GPUGraphAlgorithmsTopology`; an analysis-core adapter forges one with `as unknown as` and aliased fields | network-topology.ts:231-244; rationales at statistics :250-255, analytics columns :107-112, neighborhood :220-223, reachability :262-270, masked-weak-components.ts:28-31 | Duplicate CC, degree, BFS; the adapter depends on undocumented internals of published classes ("every gpu-graph algorithm reads only...", :186-189) and breaks silently on a luma.gl change; it is used only by terrain spike repair | Upstream: let `GPUGraphAlgorithms*` accept command-graph views or a `GPUGraphAlgorithmsTopology.fromCSR(views)`; then delete the adapter and the local CC/degree | M (luma.gl PR) |
| G4 | Weak components: min-label propagation with one pointer jump, default 32 iterations, unrolled | masked-weak-components.ts:48-80; statistics :519; analytics columns :346 | Iterations needed grow with component diameter and ID order; road networks can exceed 32 (SUSPECTED); the `converged` flag reports it but most callers will not check | Union-find with atomic hooking and path compression (ECL-CC, Afforest) converges in a fixed few passes; upstream into `GPUGraphAlgorithmsConnectedComponents` | M |
| G5 | SCC is O(V(V+E)) work, O(V²) scratch, per-thread BFS queue in private memory; bridges/articulation in one invocation | structure :113-145, :189-257 | Not usable on road networks (cap 1,024) | SCC: forward-backward with trimming (Barnat, Slota multistep) on the reachability frontier; bridges: Tarjan-Vishkin (spanning tree + Euler tour + range min) or chain decomposition | M each |
| G6 | Inhomogeneous K: Ang shell count scans all nodes and edges per pair; radius finish uses ≤ 256 threads over all pairs | inhomogeneous-k :204-223, :239, :256-259 | O(n²(V+E)) = 1.3e11 at the caps | Per source: compute edge level-set crossings once into a sorted distance list (`GPUSort`), each pair binary-searches; finish by sorting pair distances and a prefix sum of weights, one binary search per radius | M |
| G7 | Snapping defaults to O(points·edges) brute force | snapping :151-153, :535 | K-function and drive-time recipes inherit it unless they pass `candidateCapacity` | Default to the BVH path with a radius; keep brute force as opt-in | S |
| G8 | Map matching builds a private uniform edge grid (edges spanning > 16 cells drop entries) while snapping uses `GPUNearestFeatureJoin` (BVH) for the same point-to-edge query | map-matching :195, :370; snapping :466 | Two candidate searches, two failure modes | One shared "candidate edges near point" stage on `GPUNearestFeatureJoin`/`GPUBVH`; private A* noted in movement review | M |
| G9 | Frontier relaxation walks a node's whole row in one thread | network-frontier relax (grepped, not read) | Hubs serialize; harmless on roads, matters on the line graph of dense junctions and on non-road graphs (SUSPECTED) | Use the load-balanced CSR loop that statistics already uses for slot loops | S |
| G10 | Baked numeric props recompile | KDE `bandwidth` :95; facility `selectCount` :208; Louvain/flow loop bounds | Per-frame bandwidth sliders recompile | Read from a `GPUParameterBuffer` | S |
| G11 | Facility-scenario recipe sums demand in one thread | network-scenarios-recipe.ts:387 | O(D) serial | `GPUReduction` with a mask | S |

Precision: all costs are f32 and every reduction in the reachability family has a fixed order or uses
integer atomics; no float atomics found in the domain (the frontier uses `atomicMin` on f32 bits,
coarsening and adjacency matrix use fixed point). The time routing affine model and path counts in
centrality (exact below 2^24) are documented. No CPU readback found.

## 4. Consistency findings

| Topic | Variants found | Recommend |
| --- | --- | --- |
| Validation | (a) `validateCapacity` + `throw new Error(\`${id} ...\`)` in 23 files; (b) bare `assert` from `@luma.gl/core` in network-optimization (32), root point-process files (11), `prepare-*` helpers; (c) `throw` without the id (`'Network structure output length mismatch'`, structure :77-90) | (a) everywhere; named limits as exported constants (`GPU_NETWORK_REACHABILITY_MAXIMUM_ITERATIONS` is the model) |
| Iteration prop names | `maximumIterations`, `maximumTieIterations`, `maximumLabelIterations`, `maximumHops`, `maximumComponentIterations`, `components.maximumIterations`, `maximumRelaxations`, `maximumCycles`, `maximumAugmentations`, `maximumExpansions`, `maximumLevels`, `maximumCandidates`, `initialLimitCycles` | `maximumIterations` for the compile-time unrolled bound, `activeIterations` for the per-frame view; keep algorithm nouns only when there are two loops |
| Status | shared names (`status.converged`, `iterationCount`, `overflow`, `requiredCount`) in the baseline; positional summary arrays (`[valid, converged, levels, sweeps]` Louvain :37, flow 8 words :20-29, time routing :42, facility :43, rational :39); per-row codes (point routing 0/1/2 :31, KDE 0/1); per-radius bitmask (inhomogeneous K :29-30) | Shared vocabulary; a per-row `invalidCount` plus `converged`/`iterationLimitReached` |
| Topology input | `GPUNetworkTopology`, `GPUNetworkAdjacency`, raw `offsets`/`neighbors` (cost weights :111, subgraph filter :58), edge lists, dense matrices (Q1) | `GPUNetworkTopology` for anything that is a graph; a cost matrix type for matrix operators, and move matrix operators out of `gpu-network` |
| Directed flag | four signals (Q1) | Always directed CSR plus an optional exact `reverseTopology`; pairing (`reverseEdgeSlots`) only where an undirected identity is needed |
| Lane sizing doc | `recommendLaneCount` still speaks of the "expanded CSR" scratch (network-accessibility-lanes.ts:10, :23-24) after lanes started sharing one CSR (cost-matrix :106, :133) | Update doc and budget formula |
| Alignment constant | `NETWORK_ALIGNMENT_ELEMENTS` (network-view-utils :8), private `ALIGNMENT_ELEMENTS` in cost matrix :40 and K-function :94, `CENTRALITY_ALIGNMENT_ELEMENTS` (network-centrality-passes :31), all 64 | One exported constant |
| Context | spatialContext on snapping, isochrones, K-function, line graph, noding, edge bundling, map matching, drive-time recipe; missing on coarsening although it averages `nodePositions` | Add to coarsening; reachability family needs none |
| Naming | "Network" prefix on contributors that never see a network (facility optimization, rational access, KDE, inhomogeneous K) | Name and place by input type |
| Unprefixed exports | `GPUAdjacencyMatrix*`, `GPUAlternativeRoutes`, `GPUEdgeBundling`, `GPUMapMatching`, `GPUFlowAggregation` lack the `GPUNetwork` prefix the rest carry | Accept; document the rule (prefix only for CSR consumers) |

## 5. Duplication and shared-helper candidates

- `fn finite(value: f32)` is defined in five network-optimization files (louvain :111, facility
  :137, time routing :139, rational :119, point routing :106) and in trajectory files; the root files
  use `IS_FINITE_FLOAT_WGSL` from `analysis-core/kernels/wgsl-common`. Use the shared one.
- Weak components: `analysis-core/kernels/masked-weak-components.ts` (statistics, analytics columns,
  landscape) duplicates published `GPUGraphAlgorithmsConnectedComponents` because of G3; label
  propagation also appears in spatial clustering, polygon overlay, regionalization (other reviewers).
- Degree: analytics columns and statistics both compute degree and components; statistics justifies
  its own pass (masks, :242-250). Analytics columns could read statistics' per-node pass or the
  published `GPUGraphAlgorithmsDegree` once G3 is fixed.
- Predecessor/tie logic is reused correctly: service areas calls the reachability tie passes
  (service-areas :35-42 imports `createReachabilityTie*`), centrality reuses `network-frontier`.
- Lane-expanded batching (seed expansion, aligned sub-views, per-batch converged words, final
  converged reduction) is written three times: cost matrix :262-518, cost table, K-function :600-720.
  Factor one `createLaneBatchedReachabilityNodes` helper that also owns the node budget (G1).
- Facility solvers: three implementations of "evaluate candidate set on a cost matrix" (Q5).
- Point-to-edge candidates: snapping (BVH) vs map matching (private grid) (G8).
- Routing on turns: point routing's private Bellman-Ford over `turns` duplicates line graph +
  reachability; time routing's automaton duplicates line-graph bans for two-edge restrictions.
- `getGreatestCommonDivisor` (cost-matrix :559) and `getSubView` are generic; `analysis-core`.

## 6. Comprehensiveness

**Published luma.gl coverage and where each local contributor stands.**

| Published (`@luma.gl/graph`) | Local | Relation |
| --- | --- | --- |
| `SingleSourceShortestPath` (one source, dense Bellman-Ford over every vertex, ≤ 1,024 rounds) | `GPUNetworkReachability` | Local is the better GPU design (frontier, multi-source, lanes, cost limit). Upstream it as the frontier mode of the published SSSP; keep only geospatial wrappers here |
| `BreadthFirstSearch` (hop distances, per-frame `activeDepth`) | `GPUNetworkNeighborhood` | Wraps reachability with unit weights instead; would use BFS after G3 |
| `ConnectedComponents`, `Degree` | `masked-weak-components`, analytics columns, statistics | Duplicates forced by G3 |
| `Modularity` (float atomics) | statistics modularity (integer exact, unweighted) | Justified local variant; offer it upstream |
| `ModularityOptimization` (single level, one move per round) | `GPUNetworkLouvain` (32 vertices, dense) | Drop local; contribute parallel multilevel Louvain/Leiden upstream (NW04) |
| `PageRank`, `CoreNumber`, `LabelPropagation`, `LocalClusteringCoefficient`, force layouts | none | Usable today on physical topology; analytics columns explicitly does not offer them |
| `GPUCOOToCSR`, `GPUGraphTraversal`, `GPUGraphSpMV` (gpu-core) | `GPUCOOToCSR` only recommended in docs; SpMV unused | SpMV is the right base for Katz/eigenvector and for gravity accessibility as a sparse product |

**Belongs upstream (generic graph code, AGENTS.md):** reachability engine (as above), Brandes
centrality core, SCC/bridges, Louvain, max/min-cost flow, connected components, adjacency matrix and
its ordering, subgraph filter and coarsening (graph-view utilities), graph statistics. Geospatial and
analysis code that stays: snapping, noding, line graph with turn angles, isochrones, service areas,
cost matrix/table and accessibility, cost weights, K-function, KDE, inhomogeneous K, map matching,
flow aggregation, edge bundling of geographic flows, recipes. Upstream moves depend on G3 first.

**Missing, ranked by how often practitioners use them.** Items in roadmap 2.4/2.15 are excluded
(NW-CENT, NW-SCC, NW-KSP, NW02-NW06, NW-CH rejected, NW-GA, MOB-05 map-matched edge volumes, MOB-09
potential path areas).

| # | Capability (reference) | GPU-suitable | Reuses | Effort |
| --- | --- | --- | --- | --- |
| 1 | OD-to-edge loading: all-or-nothing assignment of an OD demand onto shortest-path edges, then edge volumes (Pandana, AequilibraE, pgRouting) | Yes: predecessor walk per OD pair, fixed-point atomic adds per edge | cost matrix lanes + reachability predecessors (needs lanes with predecessors) + path extraction | M |
| 2 | Pandana-style aggregation and k nearest POIs per node within a network distance | Yes: k-best label propagation over tight edges; aggregation = reverse multi-source with cost limit | service areas label phase, accessibility | M |
| 3 | Street-network stats (OSMnx `basic_stats`): circuity, intersection and street density, edge length stats, orientation entropy | Yes, reductions | statistics, noding, `GPUHistogram` | S |
| 4 | Graph simplification (OSMnx `simplify_graph`, `consolidate_intersections`) | Yes: degree-2 chain detection by list ranking / pointer jumping; consolidation by grid snap + union | noding, coarsening | M |
| 5 | Space syntax / sDNA / UNA metrics: angular integration and choice, straightness, reach, gravity index | Yes: line graph with angular weights + centrality; straightness needs positions | `GPUNetworkLineGraph`, `GPUNetworkCentrality` | S each |
| 6 | Minimum spanning tree / Steiner approximation on a network (NetworkX, pgr_prim/kruskal) | Yes: Borůvka | `gpu-spatial-weights-minimum-spanning-tree.ts` takes a CSR already | S |
| 7 | spatstat.linnet: network G and F functions, cross-K, linear pcf, random points on a network as a public operator | Yes | service areas (nearest event), K-function machinery, the Philox edge sampler inside K-function | S each |
| 8 | Eccentricity, diameter, radius, network-wide average shortest path | Yes, row reductions of the matrix/table | cost matrix + `GPUReduction` | S |
| 9 | p-center at scale and LSCP exact (spopt) | Yes: binary search on radius + set cover | location-allocation | M |
| 10 | TSP 2-opt/Or-opt on a cost matrix (pgr_TSP) | Partly: parallel 2-opt move evaluation | cost matrix | M |
| 11 | Heat-kernel (equal-split diffusion) network KDE (McSwiggan, spatstat `densityHeat`) | Yes: explicit diffusion steps on the edge graph | line/edge CSR | M |
| 12 | Katz / eigenvector / HITS centrality | Yes: SpMV power iteration | `GPUGraphSpMV`; upstream | S (upstream) |

**Exclusions (mark, do not build):** single-query point-to-point latency routing (bidirectional
Dijkstra, A*, CH/PHAST; r5, Valhalla, OSRM do this on CPU with preprocessing), public-transit RAPTOR,
vehicle routing with time windows, turn-by-turn guidance, Chinese postman. Out of place in an analysis
library: `GPUAdjacencyMatrix` and `GPUAdjacencyMatrixOrder` (graph-view rendering helpers; luma.gl or
a deck.gl graph layer package).

## 7. Top ten actions

| # | Action | Effort | Files |
| --- | --- | --- | --- |
| 1 | Shared command-node estimate and guard for every reachability composer; factor the lane-batched reachability helper (G1, section 5) | S-M | network-accessibility/gpu-network-cost-matrix.ts, network-cost-table/gpu-network-cost-table.ts, network-k-function/gpu-network-k-function.ts, network-centrality/gpu-network-centrality.ts, new `network-shared/lane-batched-reachability.ts` |
| 2 | luma.gl PR: graph algorithms accept command-graph views or a CSR-from-views topology; then delete `createNetworkAnalyticsTopology` and local CC/degree (G3) | M | analysis-core/network-topology.ts, analysis-core/kernels/masked-weak-components.ts, gpu-terrain/terrain-decode/gpu-terrain-spike-repair.ts, network-statistics, network-analysis/gpu-network-analytics-columns.ts |
| 3 | Delete `GPUNetworkPointRouting`; document the line graph + cost matrix composition and add a same-edge kernel | S | network-optimization/gpu-network-point-routing.ts, network-line-graph, docs |
| 4 | Move facility optimization into location-allocation; replace exhaustive capacitated p-median with Lagrangian relaxation + interchange; fix the documented 64-row limit (real limit F^D ≤ 2^20) | M | network-optimization/gpu-network-facility-optimization.ts, gpu-spatial-analysis/location-allocation/* |
| 5 | Rewrite rational access as per-cycle row-parallel kernels and move it to catchment-accessibility | M | network-optimization/gpu-network-rational-access.ts, gpu-spatial-analysis/catchment-accessibility/* |
| 6 | Drop `GPUNetworkLouvain` in favour of published modularity optimization; propose parallel multilevel Louvain upstream (NW04) | S here, M upstream | network-optimization/gpu-network-louvain.ts |
| 7 | File layout: root holds only `index.ts`; move KDE and inhomogeneous K into `network-point-process/`, workgroup counters and view utils to `analysis-core/kernels`, spatial context to `network-shared/` | S | gpu-network root files, gpu-network/index.ts, network-point-process/index.ts |
| 8 | Replace bare asserts in network-optimization and the point-process files with named-limit helpers and id-prefixed messages; adopt the shared status vocabulary | S | network-optimization/*, gpu-network-inhomogeneous-k.ts, gpu-network-kernel-density.ts, network-structure (messages) |
| 9 | Inhomogeneous K: sorted-distance Ang correction and sort + scan radius finish (G6) | M | gpu-network-inhomogeneous-k.ts |
| 10 | OD-to-edge loading (missing #1) and street-network stats (#3) as the first comprehensiveness items; both reuse existing passes | M + S | new `network-assignment/`, network-statistics |

Not checked: test coverage per contributor, the bodies of the `*-passes.ts` kernels (only grepped),
runtime behaviour of the node-count and watchdog suspicions (no runs; brief forbids builds and
tests), and whether `recommendReachabilityIterations` is used by any composer.
