# Math correctness review: network flows (2026-11-10)

Scope id `network-flows`. Paths are relative to `modules/analysis/src` unless they start with
`docs/`, `test/` (meaning `modules/analysis/test`) or `node_modules/`.

Scripts that confirm findings are in the session scratchpad under `network-flows/`
(`ang-shell.mjs`, `ang-grid.mjs`, `misc.mjs`). They are pure Node and use `Math.fround` for f32.
No spec was run (0 of the 5 allowed runs).

## 1. Coverage

Read in full:

- `gpu-network/flow-aggregation/`: `flow-aggregation-kernels.ts`, `gpu-flow-aggregation.ts`,
  `gpu-kernel-flow-smoothing.ts`, `gpu-radiation-flow-model.ts`, `gpu-map-matched-edge-volumes.ts`,
  `flow-aggregation-pairs.ts`, `index.ts`.
- `gpu-network/map-matching/gpu-map-matching.ts` (all 2,145 lines).
- `gpu-network/edge-bundling/`: `edge-bundling-passes.ts`, `gpu-edge-bundling.ts`.
- `gpu-network/network-subgraph-filter/`: `gpu-network-subgraph-filter.ts`,
  `network-subgraph-filter-parameters.ts`.
- `gpu-network/network-coarsening/gpu-network-coarsening.ts`.
- `gpu-network/network-line-graph/`: `gpu-network-line-graph.ts`, `network-line-graph-parameters.ts`.
- `gpu-network/adjacency-matrix/`: `gpu-adjacency-matrix.ts`, `adjacency-matrix-order.ts`.
- `gpu-network/network-noding/gpu-network-noding.ts`.
- `gpu-network/recipes/`: `drive-time-catchment-recipe.ts`, `network-scenarios-recipe.ts`,
  `track-quality-recipe.ts`.
- `gpu-network/` root: `gpu-network-inhomogeneous-k.ts`, `gpu-network-kernel-density.ts`, `index.ts`,
  `network-spatial-context.ts`, `network-view-utils.ts`, `network-workgroup-counters.ts`.

Read in part, only to check how the scope composes them:

- `analysis-core/kernels/fixed-point-accumulator.ts` (lines 1-140), `sorted-segment-offsets.ts`
  (1-60), `sorted-segment-sums.ts` (`getSortKeyBits`), `wgsl-common.ts` (finite tests, ordered keys).
- `analysis-core/contracts/earth.ts`, `spatial-context.ts` (`getGPUGreatCircleRadius`).
- `gpu-spatial-analysis/trajectory-staypoints/movement-common.ts` (`getMovementDistanceWGSL`),
  `gpu-spatial-analysis/point-density/point-density-hexagon.ts`.
- `gpu-network/network-accessibility/gpu-network-snapping.ts` (start direction, cost outputs),
  `gpu-network-accessibility.ts` (parameter layout, orientation).
- luma.gl: `gpu-core/gpu-hash-index.js` (build finalize, query), `gpu-group-aggregation.js`
  (acceptance test), `gpu-coo-to-csr.js` (all), `gpu-scan.js` (default mode), `gpu-grid-binning.js`
  (cell formula), `gpu-compaction.d.ts`, `gpu-graph-algorithms/gpu-graph-topology-internals.js`
  (undirected count and scatter), `gpu-graph-degree.d.ts`, `gpu-graph-force-layout.d.ts`.
- Docs: `docs/api/gpu-network/operators.md` sections for coarsening, adjacency matrix, noding,
  line graph, map matching, edge bundling, flow aggregation; `flow-models-mob10.md`;
  `point-process.md`.
- Tests: fixture generators and oracles were opened only to see what they pin (section 6).

Not opened, so not verified: the five `index.ts` files of `map-matching`, `edge-bundling`,
`network-subgraph-filter`, `network-coarsening`, `network-line-graph`, `adjacency-matrix`,
`network-noding`, `recipes` (re-exports only); `geometry-measures/geodesic-wgsl.ts` (the
haversine, bearing and segment-fraction functions that map matching and the line graph call are
trusted here, not checked); `sorted-segment-sums.ts` tree code; the time-window classify nodes;
`scan-total.ts`; `createPublishNode`; `GPULineSplit`; `GPUNetworkServiceAreas`,
`GPUNetworkIsochrones`, `GPUNetworkCostMatrix`, `GPUGroupStatistics`, `GPUTrajectoryMetrics`,
`GPUTrackSimilarity` (the recipes compose them; only the recipe arithmetic was checked).

`gpu-spatial-analysis/network-topology.ts` does not exist.

## 2. Findings

| ID | Severity | Status | File:line | What the kernel computes | What the reference says | Input that gives a wrong output | Suggested fix |
|---|---|---|---|---|---|---|---|
| network-flows-01 | wrong-result | CONFIRMED (Node replica) | `gpu-network/gpu-network-inhomogeneous-k.ts:247`, `:257-262` | Ang shell count `m(u, t)` with exact f32 tests: a vertex counts when `d == t`, an edge root counts when `0 < t - d_first < length`. | spatstat `linearKinhom` counts network locations at distance `t` with a tolerance (`countends`). The destination vertex counts once. | Chain 0-1-2 style networks with non-dyadic lengths. `d[target] = fl(d[previous] + length)` rounds down about half the time, then `t - d[previous] < length` by one ulp, so the edge into the target is counted as an interior crossing and the target vertex is counted too. Path of 40 nodes, lengths 60 to 200: 44% of pairs get `m` one too high, mean pair weight 0.81 of the exact one. 8x8 irregular grid: 32% of pairs, `K` 1 to 3% low. | Treat a root within a relative tolerance of an edge end as that vertex (for example `fromFirst >= length * (1 - 1e-5)` is not interior), and compare vertices with the same tolerance. Add a fixture with non-integer lengths. |
| network-flows-02 | wrong-edge-case | CONFIRMED (hand, script) | `gpu-network/recipes/drive-time-catchment-recipe.ts:635-654` | Demand time `min(cost[source] + f_d w, cost[target] + (1 - f_d) w)`. | Travel along the shared edge is `abs(f_d - f_f) w` when demand and facility are on the same edge. The doc says "exact at the snapped edge". | Edge cost 100, facility at fraction 0.5, demand at the same point: recipe gives 100, true time 0. Facility 0.4, demand 0.6: recipe 100, true 20. | In the adapter, when the demand edge equals a facility edge (or its reverse twin), also take `abs(f_d - f_f) w`. Needs the facility edge and fraction per assignment. |
| network-flows-03 | wrong-edge-case | CONFIRMED (hand, script) | `gpu-network/recipes/drive-time-catchment-recipe.ts:439-458`, `:594-607`, `:642-646` | Facilities start at both edge ends (`GPUNetworkSnapping` default `startDirection: 'both'`); demand may be reached from the edge target going backward. | On a one-way edge a facility leaves only toward the target, and demand is reached only from the source. `gpu-network-snapping.ts:78-81` says `'both'` is for undirected networks. | One-way edge A to B cost 10, the way back B to A costs 100. Facility at 0.9, demand at 0.1: recipe 10, true 102. | Document that the recipe needs both directions with equal cost, or add a `directed` option: `startDirection: 'forward'` for facilities and `cost[source] + sourceCost` only for demand. |
| network-flows-04 | inconsistent | CONFIRMED (reading both kernels) | `gpu-network/network-noding/gpu-network-noding.ts:103-104`, `:652-664`, `:698-703` | The undirected CSR lists a loop edge twice in its node's row. | Siblings and luma.gl list an undirected self-loop once: `network-coarsening/gpu-network-coarsening.ts:135`, `network-subgraph-filter/gpu-network-subgraph-filter.ts:101`, `network-statistics/gpu-network-statistics.ts:180`, luma `gpu-graph-topology-internals.js:201-204`. | One closed ring line. Noding: 1 edge, row `[0, 0]`. `GPUNetworkSubgraphFilter` `liveEdgeCount = (2 - 2) / 2 + 2 = 2`. `GPUNetworkCoarsening` counts both slots (`index <= targetVertex`, `:683`): `intraEdgeCount` 2 and twice the weight. | Emit one slot for a loop, or state the difference in the user doc and in the consumers. |
| network-flows-05 | inconsistent | CONFIRMED (script) | `gpu-network/network-coarsening/gpu-network-coarsening.ts:205-242` | Private 64-bit fixed point: every addend is `round(abs(v) * 65536) / 65536`. | Contract section 11 names the shared accumulator (`analysis-core/kernels/fixed-point-accumulator.ts`), which is 96-bit with a data-derived unit and keeps 24 bits per term. | 1,000 edges of weight 1e-4: sum 0.1068, exact 0.1 (6.8% high). Weight 1e-5: 52% high. Weight 5e-6: sum 0. | Use the shared accumulator for weight, value and position sums. The rounding is documented, so this is a consistency and precision issue, not a hidden bug. |
| network-flows-06 | untested | SUSPECTED | `gpu-network/flow-aggregation/gpu-kernel-flow-smoothing.ts:359-396` | `S_ij` is a kernel-weighted mean of flows (row-normalised in both passes). | Guo and Zhu 2014. The class doc says the paper was not available. | Zones at x = 0, 1, 10, unit weights, `neighborhoodSize` 2: a single flow of 100 at (1,1) sums to 125.3 after smoothing, a flow of 100 at (2,2) sums to 77.6. Total flow is not conserved. | Check the estimator against the paper. If the published form is a weighted sum, the normalisation differs. Say in the doc that totals are not preserved. |
| network-flows-07 | precision | CONFIRMED (script) | `gpu-network/adjacency-matrix/gpu-adjacency-matrix.ts:407-416` against `:76-82` | GPU: `floor(fl(fl(w * scale) + 0.5))`. Exported CPU helper `getGPUAdjacencyMatrixFixedWeight`: `floor(w * scale + 0.5)` in binary64. | The helper is documented as the weight one edge contributes. | `w = 8192.0009765625`, scale 1024: GPU 8388610, helper 8388609. Half of all weights between 8192 and 16384 differ by one step. With scale 1000, 114 of 100,000 random weights in 0 to 50 differ. | Compute the rounding with integer or `floor(x) + (fract(x) >= 0.5)` on the GPU, or make the helper mirror the f32 steps. Relative size is 1e-7, so only exact comparisons notice. |
| network-flows-08 | doc-mismatch | CONFIRMED (reading) | `gpu-network/flow-aggregation/gpu-flow-aggregation.ts:420-426`; `docs/api/gpu-network/operators.md:1318` | Throws when `pairCapacity * 256 > 2^32 - 1`, so the largest value is 2^23. | The doc page says 2^24. | `pairCapacity: 2 ** 24` with weights and `'sorted'` throws. | Change the doc to 2^23. |
| network-flows-09 | doc-mismatch | CONFIRMED (reading) | `docs/api/gpu-network/operators.md:1068-1069` | `GPUMapMatchedEdgeVolumes` exists and counts the traversed lists. | The map-matching doc says "Edge volumes over these lists are not built yet." | n/a | Remove the sentence and link `map-matched-edge-volumes-mob05.md`. |
| network-flows-10 | doc-mismatch | SUSPECTED | `gpu-network/flow-aggregation/gpu-radiation-flow-model.ts:64-68`; `docs/api/gpu-network/flow-models-mob10.md` | `outflows` is used as `T_i`. The doc names `zoneOutWeights` of `GPUFlowAggregation` as the usual source. | In Simini et al. `T_i` is the trips that leave zone `i`. `zoneOutWeights` include self flows unless `excludeSelfFlows` is set. | A zone with 100 trips of which 60 stay inside: `T_i` should be 40, the suggested wiring gives 100. | Say that the aggregation must run with `excludeSelfFlows: true` for this use. |
| network-flows-11 | precision | SUSPECTED | `gpu-network/flow-aggregation/gpu-kernel-flow-smoothing.ts:313`, `:343` | The radius is a distance computed in the `radii` pipeline. Membership is `distance <= radius` with the distance recomputed in the `weights` pipeline. | The zone that defines `h_i` belongs to the neighbourhood. | If the two pipelines round the same expression differently (fused multiply-add), the defining zone falls outside by one ulp. Not observed. | Store the neighbour that defines the radius, or compare with a one-ulp margin. |
| network-flows-12 | inconsistent | CONFIRMED (reading) | see section 4 | Missing float outputs use four markers: -1, +Infinity, 3.4028234e38 and NaN. | Contract section 7 asks for one no-data rule. | n/a | Pick one marker per output kind and add a `validity` view where the contract asks for it. |

Notes on the findings that need them.

**network-flows-01.** The kernel is exact in real arithmetic. The problem is only the comparison in
f32. The distance table comes from relaxations `d[v] = fl(d[u] + w)`. For the tree edge into the
destination the true root is at the edge end, which the strict `< length` test should exclude. When
the addition rounded down, `fl(t - d[u])` is one ulp below `length` and the test passes. The vertex
test on `:247` also passes, because the destination's own table entry equals `t`. The shell count is
then one too high. It is never too low in the simulations. Every fixture uses integer lengths
(`test/gpu-spatial-analysis/point-process/generate-network-point-process-fixtures.py` says so, and
the spatstat fixtures have lengths 2, 3 and 4), so no test sees it. The effect is largest on sparse,
tree-like networks, where the true count is 1 or 2. `ang-shell.mjs` and `ang-grid.mjs` replicate the
kernel lines one for one. `correction: 'none'` is not affected.

**network-flows-02 and -03.** Both come from the same adapter. The spec oracle
(`test/gpu-network/recipes/drive-time-catchment-recipe.spec.ts:51-63`) is "demand time over both
edge ends", a copy of the recipe formula, so it cannot catch either. For -02 the error is an
overestimate of at most one edge cost. It matters most on long edges and for demand next to the
facility. For -03 the error is an underestimate without bound. The service-area search itself is
directed; only the two snapping adapters assume symmetry.

**network-flows-04.** Noding documents its own convention in the type comment, so the kernel does
what it says. The defect is that the user doc says the topology "feeds ... the other contributors
directly" and three of those state the opposite convention. Loops appear for closed rings and for
short pieces whose two ends fall into one tolerance cell.

**network-flows-05.** Counts and bounds in coarsening are exact. Only sums are affected, and only
when values are small against `1 / fixedPointScale`. Longitude and latitude sums are fine (quantum
1.5e-5 degrees, rounding is unbiased). Edge weights in kilometres, hours or probabilities are not.

Earlier findings, checked again:

- Movement review F5 (backward jitter on a one-way edge breaks the track): still present by default
  at `gpu-network/map-matching/gpu-map-matching.ts:1690` and `:1705`. `minimumFixSpacingSigmas`
  (`:997-1020`) is the fix and is off unless set. Documented.
- Movement review F6 (Viterbi scores accumulate in f32): fixed, `:1886-1903` renormalises per step
  and `:1975-1994` sums the offsets as an integer part and a fraction.
- Movement review F4 (speed flag on the step ending at a row): still the default at
  `gpu-network/recipes/track-quality-recipe.ts:123-124`, now documented, with a `'point-centred'`
  mode (`:129-135`) whose logic I checked by cases (interior, first and last row).

## 3. Verified correct

- `GPUFlowAggregation`: pair key `origin * zoneCount + destination` stays below `0xffffffff` for 65,535
  zones (largest key 4,294,836,224, empty-slot key 4,294,836,225); representative row is the lowest
  source row (luma `GPUHashIndex` `atomicMin`); segment offsets of the sorted pair sum equal the
  exclusive scan of per-representative counts; ranking key is the order-preserving float key,
  complemented, with `-0` folded into `+0`; count keys need `bits(rowCount)`; zone totals count every
  accepted row, so `sum(zoneOut) = sum(zoneIn) = accepted rows`. Compared against the class doc and
  the luma primitives it calls.
- Grid zones: same text as luma `GPUGridBinning` `getCoordinate`. Hexagon zones: cube rounding and
  odd-r conversion checked by hand; negative columns are only reachable on the bounds edge.
- `GPURadiationFlowModel`: flux formula and strict `s_ij` against Simini et al. 2012; the
  finite-size factor `1 / (1 - m_i / M)` by the telescoping sum (rows sum to `T_i` without ties).
- `GPUMapMatchedEdgeVolumes`: one count per stored edge entry, no movement across a separator or past
  the stored count; integer atomics.
- `GPUMapMatching`: emission `-log(sigma) - 0.5 log(2 pi) - 0.5 (d / sigma)^2` and transition
  `-log(beta) - abs(straight - route) / beta` against Newson and Krumm equations 1 and 2; Viterbi
  recurrence, restart at a break, back-pointer walk; the log-likelihood equals the sum of per-step
  offsets; A* heuristic (nearest target, great-circle or planar, times 0.999999) is consistent, so
  settled nodes are exact and every node on a route within the bound has a key within the bound;
  backward-shift hash deletion; route `tail + network + toFraction * length` with the bound on the
  whole route; the route table path equals the A* path; fix filter is the 2 sigma rule; hop
  reconstruction walks settled parents only.
- `GPUEdgeBundling`: this is KDEEB (Hurter, Ersoy, Telea 2012), not Holten and van Wijk force
  bundling, so compatibility measures and spring forces do not apply. Epanechnikov splat reach,
  cell-centre convention shared by splat and gradient, advection along the normalised gradient by
  one radius, arc-length resampling, simultaneous Laplacian pass, `radius0 * lambda^k` with lambda in
  0.5 to 0.9, fixed-point exponent that cannot overflow, order-preserving box keys, `toWork` and
  `fromWork` inverse. Step order differs from the paper (resample before smoothing), which is a
  choice.
- `GPUNetworkSubgraphFilter`: owner-row binary search with empty rows, slot pairing symmetry,
  half-open attribute ranges and closed time window as documented, `liveEdgeCount`, induced CSR
  offsets from the shared prefix sum.
- `GPUNetworkCoarsening`: two-word carry is order independent (each wrap of the low word is seen
  once), two's-complement negation, undirected key `min * G + max`, sentinel `G * G` below 2^32,
  segment numbering, `counted = dropped + intra + inter`, superedge counts sum to `interEdgeCount`.
  Checked against the class doc only; no NetworkX `quotient_graph` run.
- `GPUNetworkLineGraph`: arc `e -> f` when `head(e) = tail(f)` (NetworkX `line_graph`), turn angle
  `atan2(cross, dot)` positive to the left, cost terms as documented, the count pass and the fill
  pass accept the same turns (including the two shortcuts), lower-bound search of sorted bans.
- `GPUAdjacencyMatrix` and `GPUAdjacencyMatrixOrder`: bin `floor((p - start) R / (end - start))`
  without `u32` overflow under the validated limits; mirror mode writes a self-loop once; order is
  the stable sort by (group, tie key, node). There is no normalisation step.
- `GPUNetworkNoding`: cell `floor((x - origin) / tolerance)`, two stable sorts give (x, y) order,
  group rank is the node ID, first endpoint gives the node position, CSR entries grouped by node
  without a second sort, lengths are vertex-to-vertex sums.
- `GPUNetworkKernelDensity`: triangular kernel `max(0, 1 - d / h) / h`; dividing by each event's
  quadrature mass gives `sum(length * intensity) = eventCount`.
- `GPUNetworkInhomogeneousK` with `correction: 'none'`, and the three denominators (`L`,
  `sum(1 / lambda)`, `sum(1 / lambda)^2 / L`) against spatstat `linearKinhom`.
- `addNetworkDisruptionRecipe`, `addFacilityScenarioRecipe`: difference and coverage arithmetic; the
  coverage threshold is the same word and the same `<=` as `GPUNetworkAccessibility`.
- `addTrackQualityRecipe`: flag logic in both speed modes.

## 4. Standards consistency

- Order-independent sums are done four ways in this scope: sorted fixed tree (flow aggregation),
  private 64-bit fixed point with a fixed scale (coarsening), `u32` fixed point that wraps at 2^32
  (adjacency matrix, scale 1024), `u32` fixed point with a point-count exponent (edge bundling
  density). None uses the shared 96-bit accumulator. The first is allowed by contract section 11.
  The second and third differ numerically from the shared one (network-flows-05).
- Undirected self-loops: once (coarsening, subgraph filter, adjacency matrix mirror mode, luma
  topology) against twice (noding output). See network-flows-04.
- Flow totals and self flows: `zoneOutCounts` and `zoneInCounts` include self flows unless they are
  excluded from everything. flowmap.gl keeps a separate "within" total. Documented here as "accepted
  rows", so not a defect, but it is why network-flows-10 exists.
- Missing float outputs: -1 (map matching fractions, offsets, snap distances, route rows; line
  graph weights; snapping costs), +Infinity (drive-time `demandTimes`, K and KDE distance inputs),
  3.4028234e38 (`gpu-network-noding.ts:696`), NaN (disruption differences, KDE and K invalid rows).
  Internal "unreached" constants: 3.0e38 (`drive-time-catchment-recipe.ts:634`), -1.0e30 (map
  matching scores).
- Private copies of things `analysis-core` has:
  - Binary search: `gpu-map-matching.ts:1749-1765`, `gpu-network-line-graph.ts:307-327` and
    `:621-630`, `gpu-network-subgraph-filter.ts:608-617`. Same lower or upper bound as
    `binary-search-wgsl.ts`.
  - Finite tests: `gpu-network-noding.ts:65-67`, `gpu-kernel-flow-smoothing.ts:282`, and
    `edge-bundling-passes.ts:154` (stricter: also below 3.0e38). The two root files use the
    deprecated `isFiniteFloat` name.
  - Ordered float keys: `edge-bundling-passes.ts:183-185`, `:405-408` (same bits as the shared
    `getOrderedKey`); `gpu-network-noding.ts:68-74` (folds `-0` into `+0`, the shared one does not).
  - Fixed point: `gpu-network-coarsening.ts:205-242`.
- Sort key bits are computed three ways: `getSortKeyBits` (coarsening), `toString(2).length`
  (`gpu-flow-aggregation.ts:723`, `:778`), `Math.ceil(Math.log2(edgeCount + 1))`
  (`gpu-network-line-graph.ts:705`). All three give enough bits (checked for every count to 2^22 and
  around each power of two to 2^32).
- Literal where a named constant exists: degrees to radians `0.017453292519943295` in
  `edge-bundling-passes.ts:196` (`GEODESIC_DEGREES_TO_RADIANS`).
- Earth radius: every great-circle path in scope goes through `getGPUGreatCircleRadius` or
  `getGPUSphereRadius`. No second radius found.
- Thread order: integer atomics only, except flow aggregation `sumOrder: 'atomic'` (documented
  opt-in). Plain non-atomic writes of the constant 1 to an overflow word from many threads
  (`gpu-map-matching.ts:888`, `:898`, `:1124`) are benign.

## 5. luma.gl official implementations

- `GPUFlowAggregation` delegates to `GPUHashIndex`, `GPUHashIndexQuery`, `GPUGroupAggregation`,
  `GPUSort`, `GPUScan` and `GPUReduction` from `gpu-core`. The arguments keep their meaning: key
  `0xffffffff` is the reserved empty key and pair keys cannot reach it; `firstValue: 0` makes the
  stored value the lowest source row; the atomic sum path skips non-finite values the same way the
  sorted path zeroes them.
- Grid zones (`flow-aggregation-kernels.ts:124-128`) re-implement the cell formula of public
  `GPUGridBinning` (`gpu-core`). The two are textually identical. The local copy exists for the
  per-frame active grid size and chunk offsets. Propose `activeGridSize` upstream, then delegate.
- `GPUNetworkSubgraphFilter` compaction (`gpu-network-subgraph-filter.ts:803-857`) re-implements
  public `GPUCompaction` (`gpu-core`): exclusive scan, then scatter. The math agrees. The local one
  adds a bounded capacity with a required count, which `GPUCompaction` lacks. Propose upstream.
- `GPUNetworkNoding` CSR offsets use `createSortedSegmentOffsetsNode` (lower bound of each row in
  the sorted starts). That is the offsets pass of public `GPUCOOToCSR` (`gpu-core`). The math
  agrees. Its comment explains why `GPUGallopingSearch` is not used but does not mention
  `GPUCOOToCSR`.
- Noding's edge list to undirected CSR overlaps public `GPUGraphAlgorithmsTopology`
  (`gpu-graph-algorithms`). The conventions differ: luma writes an undirected self-loop once and
  leaves the order inside a row unspecified; noding writes it twice in a fixed order. luma's
  self-loop rule is the one the local siblings follow (network-flows-04).
- Map matching builds a private uniform edge grid although `GPUGridIndex` and `GPUBVH` are public.
  Already recorded as G8 in the library review; not repeated.
- No public luma.gl counterpart was found for line graphs, coarsening (the modularity optimiser
  says it does no coarsening), adjacency-matrix binning, edge bundling, map matching, the radiation
  model or flow smoothing. `GPUGraphAlgorithmsForceLayout` is node layout; nothing in scope copies
  its forces. No private import paths are used in scope.

## 6. Test oracles

Independent oracle:

- `GPUMapMatching`: `test/oracles/parity-wp3-map-matching.py`, a Shapely, NetworkX and SciPy HMM
  written from the paper. It is not Leuven or Valhalla; the generator explains why Leuven's model
  differs. Breaks and stationary jitter are stated as not covered.
- `GPUNetworkLineGraph`: `test/oracles/parity-wp3-line-graph.py` (NetworkX `line_graph`; turn costs
  are a labelled composite).
- `GPUNetworkNoding`: `test/oracles/parity-wp3-noding.py` (Shapely `unary_union`).
- `GPUNetworkInhomogeneousK`: spatstat.linnet 3.5.3 fixtures and a NetworkX enumeration, both with
  integer edge lengths only, which is why network-flows-01 is not seen.
- `GPUNetworkKernelDensity`: spatstat shortest paths with R edge integrals
  (`test/gpu-spatial-analysis/point-process/spatstat-parity.spec.ts:220`).

Reference written from the definition in the repository (a different route, not a library):

- `GPUFlowAggregation` (`flow-aggregation-oracle.ts`), `GPUMapMatchedEdgeVolumes` (plain counting),
  `GPURadiationFlowModel` (hand case, sorted prefix sum, closed-form row sum),
  `GPUNetworkSubgraphFilter`, `GPUNetworkCoarsening`, `GPUAdjacencyMatrix`,
  `GPUAdjacencyMatrixOrder` (their `*-oracle.ts` files).

Copy of the contributor's own formula only:

- `GPUEdgeBundling`: `edge-bundling-oracle.ts` mirrors the schedule, rounding and fixed point. No
  KDEEB reference was run.
- `GPUKernelFlowSmoothing`: binary64 reference of the same three-step definition plus two limits.
  The definition itself is unchecked (network-flows-06).
- `addDriveTimeCatchmentRecipe`: the spec repeats the two-end minimum (network-flows-02, -03).

Not determined: what pins `addNetworkDisruptionRecipe`, `addFacilityScenarioRecipe` and
`addTrackQualityRecipe` beyond the single recipe-review case noted in the movement review. I did
not open those specs.

## Resolution (2026-11-10)

Paths are relative to `modules/analysis/src` unless they start with `docs/` or `test/` (meaning
`modules/analysis/test`). Line numbers are those of the working tree when this section was written.
Headless specs ran on the WebGPU test device (Metal); node specs on the null device.

Counts: 15 fixed (the twelve findings with network-flows-12 split in two, plus three extra items),
1 not reproduced, 1 deferred.

| ID | Outcome | What changed | Spec that pins it | Behaviour change a caller sees |
|---|---|---|---|---|
| network-flows-01 | fixed | `gpu-network/gpu-network-inhomogeneous-k.ts:43` (tolerance constant), `:258-292` (shell count). Nodes within `1e-5 * t` of the pair distance are on the shell; an edge root within that distance of an edge end is that node, not an interior crossing; two roots closer than that are one. The destination node always counts. | `test/gpu-network/network-inhomogeneous-k-shell-count.spec.ts` (5 cases: path by hand table, 40-node irrational path by the mirror-image closed form, four-cycle by hand with an antipodal tie, irrational irregular grid by Manhattan level sets in binary64, tenth-unit irregular grid by enumeration over half-unit cut points). All five failed on the old kernel. The spatstat and enumerated-sphere fixtures still pass. | `correction: 'ang'` on non-dyadic edge lengths: `m(u, t)` no longer one too high for about a third to a half of the pairs, so `K` rises by the 1 to 20 percent it was low. Integer-length inputs are unchanged. Level-set locations closer than `1e-5 * t` are merged. |
| network-flows-02 | fixed | `gpu-network/recipes/drive-time-catchment-recipe.ts:279-349` (`addSnappedRoadNodes`), `:759-832` (demand-time adapter, same-road term at `:805-822`). | `test/gpu-network/recipes/drive-time-catchment-direct-travel.spec.ts` (hand values 0, 20, 30 where the old adapter gave 80, 100, 50). `test/gpu-network/recipes/drive-time-catchment-recipe.spec.ts` now uses `test/gpu-network/recipes/drive-time-oracle.ts` (Dijkstra with every snapped point inserted as a node), which replaces the copy of the two-end minimum. | `demandTimes` and `demandBands` of demand on the same road as a facility are smaller: the way along the road instead of out to a node and back. Each demand point is compared with every facility (`demandCount * facilityCount` integer comparisons). |
| network-flows-03 | fixed | Same file: `:560-600` (facility start rows honour direction), `:786-800` (demand reached from an end only along an edge that goes that way). A road direction costs the cheapest passable edge that way; no edge, no travel. | Same spec: one-way loop (102, 0.5, 0, 21, 71 where the old adapter gave 10, 1.5, 2, 21, 39; node costs 101, 1, 41; `startNodes` `[0xffffffff, 1]`) and an asymmetric two-way road (9 and 3; node costs 15 and 5). | On a one-way edge the facility gets one start row (`startNodes` row `0xffffffff`, `startCosts` -1 for the other) and `nodeCosts`, `assignments`, isochrones and `demandTimes` change accordingly. On a two-way road with different costs each way, each direction uses its own cost. Two-way roads of one cost are unchanged apart from network-flows-02. |
| network-flows-04 | fixed | `gpu-network/network-noding/gpu-network-noding.ts:630-637` (a loop keeps only its start entry), type documentation `:102-110`. Converged on roadmap 12.3 (listed once), the rule of official `GPUGraphAlgorithmsTopology`, coarsening, subgraph filter and statistics. | `test/gpu-network/network-noding/gpu-network-noding.spec.ts` ("lists an undirected self-loop once"): a closed ring and a piece collapsed by the tolerance. Failed on the old kernel (6 entries for 4). The Shapely parity fixture `showcase-large-snap` (six loops) encoded the old rule in its Python CSR adapter: `test/oracles/parity-wp3-noding.py:78-79` now lists a loop once, and the CSR arrays of `test/oracles/fixtures/parity-wp3-noding-showcase-large-snap.json` were re-derived from the fixture's own edge list with that loop (Shapely is not importable here, and the CSR is not Shapely output; every other field is byte-identical). The two parity specs expect `2 * pieces - loops` entries. | `output.topology` has `2 * edges - loops` entries instead of `2 * edges`; a loop appears once in its node's row. Degree-style consumers count a loop once. |
| network-flows-05 | fixed | `gpu-network/network-coarsening/gpu-network-coarsening.ts:231-245` (quantize, unit and decode are the shared `FIXED_POINT_WGSL` functions; storage stays two words), `:433-462` (unit selection), `:556-612` and `:726-790` (scale words: largest exponent over the nodes and slots that are summed), `:1040-1160` (superedge weights in their own kernel), `:1240-1275` (intra-group outputs in their own kernel), `network-workgroup-counters.ts` (`'max'` counters). `fixedPointScale` is now an optional pin; omitted, the unit follows the data. | `test/gpu-network/network-coarsening/gpu-network-coarsening.spec.ts`: "sums small weights, values and coordinates" (1,000 weights of 1e-4 give 0.1, 1,000 of 1e-5 give 0.01, by hand; a dropped edge of weight 3.4e38 and an overflowed node at 1e20 do not coarsen the unit) and "keeps the fixed rounding step when fixedPointScale is given" (0.1068 and 0.0153 bit for bit, which is the old default). The oracle `network-coarsening-oracle.ts` was a mirror of the kernel's quantizer; without a fixed scale it now takes the exact sum of the float32 terms in a `bigint`. | Default sums change: every weight, position and value sum is the exact sum rounded to float32 (to within the documented `2^(h - 63)` of the largest term per addend) instead of a sum of terms rounded to 1/65536. Pass `fixedPointScale: 65536` for the old numbers. Non-finite terms add nothing. |
| network-flows-06 | fixed (documentation; the paper could not be checked) | `gpu-network/flow-aggregation/gpu-kernel-flow-smoothing.ts:131-143`, `docs/api/gpu-network/flow-models-mob10.md:39-64`: the contributor is described as what it computes, a row-normalised kernel mean that does not conserve total flow, with the three-zone numbers; the "after Guo and Zhu" reading is reduced to the idea, with no claim of reproducing their estimator. | `test/gpu-network/flow-aggregation/flow-models-mob10.spec.ts` ("is a normalised kernel mean: three zones by hand"): every entry of `100 w w^T` and the totals 125.26 and 77.58, derived by hand from the documented definition. | None in numbers. |
| network-flows-07 | fixed | `gpu-network/adjacency-matrix/gpu-adjacency-matrix.ts:420-432` (GPU: the float32 product feeds only `floor` and comparisons, rounds half up, and is returned unchanged above 2^23), `:75-91` (the helper takes the same float32 product and the same saturation). | `test/gpu-network/adjacency-matrix/gpu-adjacency-matrix.spec.ts` ("rounds each fixed-point weight half up, also above 2^23"): 8388609, 2, 3, 8388611 by exact products (old GPU: 8388610 and 8388612), and 16004 at scale 1000. `gpu-adjacency-matrix.node.spec.ts` pins the helper. | `weightSums` cells whose weights times the scale are odd integers above 2^23 drop by one step each. `getGPUAdjacencyMatrixFixedWeight` with a scale that is not a power of two now rounds the float32 product (16.0035 at 1000 gives 16004, was 16003) and returns `2^32 - 1` from a product of `2^32 - 256`. |
| network-flows-08 | fixed | `docs/api/gpu-network/operators.md:1350` says 2^23. | None (words). | None. |
| network-flows-09 | fixed | `docs/api/gpu-network/operators.md:1098` links `GPUMapMatchedEdgeVolumes` in place of "not built yet". | None (words). | None. |
| network-flows-10 | fixed | `gpu-network/flow-aggregation/gpu-radiation-flow-model.ts:64-69`, `docs/api/gpu-network/flow-models-mob10.md:8-10` and `:25`: the aggregation must run with `excludeSelfFlows: true` for its zone totals to be `T_i`. | None (words). | None. |
| network-flows-11 | fixed (defensive) | `gpu-network/flow-aggregation/gpu-kernel-flow-smoothing.ts:345`, `:353-356`: membership is `distance <= radius * 1.000001`. | No GPU run here shows the defect (the two pipelines round alike on this device); the existing reference, identity and wide-kernel specs and the new hand case pass. | A zone within one part in a million beyond the radius is now inside the neighbourhood. |
| network-flows-12 (noding weights) | fixed | `gpu-network/network-noding/gpu-network-noding.ts:709`: a CSR slot without an entry holds NaN (contract section 7), not 3.4028234e38. | `gpu-network-noding.spec.ts`, same test: the unused slots hold `0xffffffff`, `0xffffffff` and NaN. | `output.topology.weights` past the last entry is NaN (still impassable to the routing contributors). |
| network-flows-12 (the -1 family, `+Infinity` times) | deferred | Nothing changed in values. `demandTimes` documents `+Infinity` as an unreachable cost, the value `nodeCosts` holds (`recipes/drive-time-catchment-recipe.ts`, output type). | n/a | n/a. Reason: -1 on `matchedFractions`, `matchedOffsets`, `snapDistances` (map matching) and on the line-graph weights (where it is also the impassable cost the routing contributors read) is the same documented sentinel as the exported `GPU_NETWORK_SNAPPING_NO_VALUE` of the `network-paths` stream, and it is read by lenses, kepler, the website and the examples, which this pass may not touch. Changing it is the no-data vocabulary decision of roadmap 12.4. |
| network-flows-13 (extra) | fixed | `gpu-network/recipes/drive-time-catchment-recipe.ts:855`: band `k` is `(costBreaks[k - 1], costBreaks[k]]`. The recipe's network-space bands were `[a, b)` while roadmap 12.3 and, after this pass, `GPUNetworkIsochrones` (which the same recipe draws) use `(a, b]`. | `drive-time-catchment-direct-travel.spec.ts` ("puts a drive time exactly on a break inside that band"): exact times 0, 4, 4, 6 with breaks 4 and 100 give bands 0, 0, 0, 1 (old: 0, 1, 1, 1). | A demand time exactly equal to a break moves to the band below. |
| network-flows-14 (extra, compensated sums) | not reproduced | `gpu-network/map-matching/gpu-map-matching.ts:1966-1992` is the only place in this scope that names Neumaier. It is a comment: the float compensation was already replaced by a whole part in `i32` and a fraction in `f32`, which contains no expression a compiler can fold to zero. | `test/gpu-network/map-matching/gpu-map-matching-robustness.spec.ts` ("the log-likelihood of a long track stays accurate"), tightened: at 2,000 fixes and a total of 1e7 a plain float32 running sum of the same terms is 236.6 off and the matcher 0.6 off (one ULP is 1); the spec now asserts both. | None. |
| luma-official-02 (order of work item 5, this scope's copy) | fixed (test only) | No source change: the arithmetic of `gpu-network/flow-aggregation/flow-aggregation-kernels.ts:124` is untouched. | `test/gpu-network/flow-aggregation/flow-aggregation-grid-coordinate-pin.node.spec.ts` compares the local `getCoordinate` with the text of the published `GPUGridBinning` kernel. | None. |

Notes.

- network-flows-01: the count could not be decided by topology alone. An edge whose far end is the
  destination can hold a real interior crossing (a triangle where the destination is reached by a
  shorter way), so "the far end is the destination" does not make a root an edge end. What decides
  is whether `t - d[near]` equals the length, and that needs a tolerance because the table holds
  rounded sums. The tolerance scales with `t`: every near-degenerate decision involves only
  magnitudes below `2 t`.
- network-flows-05 keeps two storage words per sum, so the exported superedge capacity limit and
  the accumulator layouts do not change. Against the shared three-word accumulator the cost is
  dynamic range: a term keeps all 24 bits down to `2^(h - 39)` of the largest instead of
  `2^(h - 71)`. Moving to three words is the STD-10 adoption of roadmap 12.4.
- No float compensation (Kahan, Neumaier, two-sum) was added anywhere in this scope. The cures are
  integer fixed point (coarsening), comparisons against a tolerance (K), and `floor` plus a
  comparison (adjacency matrix).

Upstream gaps (what luma.gl would need for this scope to delegate):

- `GPUGridBinning`: a per-frame active grid size and chunk offsets, or a public cell-coordinate
  WGSL snippet; then the grid zones of `GPUFlowAggregation` use it instead of a pinned copy.
- `GPUCompaction`: a bounded output capacity with a required count; then
  `GPUNetworkSubgraphFilter` drops its own scan and scatter.
- `GPUGraphAlgorithmsTopology`: construction from an edge list held in command-graph views with a
  stated order inside a row and edge-ID and weight columns; then `GPUNetworkNoding` builds its CSR
  with it. The self-loop rule now matches.

Not verified in this pass, for whoever integrates: the lens records and website runs of
`drive-time-catchment-recipe`, `network-noding`, `network-coarsening` and
`network-inhomogeneous-k` (`docs/lens-records/**`, `website/**`, `modules/lenses/**`) were made
with the old behaviour and were not regenerated; the caption "about 44.8 seconds, in band 0" in
`docs/api/gpu-network/operators.md` may no longer match if that demand point shares a road with a
facility.

### Review follow-up

From the gates and `fix-review-fields.md` (fr-fields-03, -06, -09). Each defect was reproduced on
the GPU before its fix, except the type errors.

| Item | Outcome | What changed | Spec that pins it | Behaviour change a caller sees |
|---|---|---|---|---|
| `typecheck-tests` TS2740 | fixed | `test/gpu-network/recipes/drive-time-catchment-direct-travel.spec.ts:30-36`: the result fields are `number[]`, which is what the harness readers return. No assertion changed. | A `tsc --noEmit` over the fifteen test files of this stream, with the repository `tsconfig.json`, reports no error. | None. |
| fr-fields-06 | fixed | `gpu-network/gpu-network-inhomogeneous-k.ts:185-207` (`getShellMergedNode`), `:294-309` (edge loop), class text `:94-99`. Two nodes on the shell that are joined by an edge tight within the tolerance are one location: the later one (larger distance, then larger node ID) is taken off the count, once per node however many edges merge it. The earliest node of a run is never merged, so the count cannot fall below one, and a run entered from two branches keeps both entries. | `test/gpu-network/network-inhomogeneous-k-shell-count.spec.ts` ("takes nodes on the shell that a very short edge joins as one location"): the reviewer's two nodes (m = 2, was 3), three nodes in a row (2, was 4), and two branches meeting just past the shell (3: both crossings, the meeting node merged once). With the subtraction disabled the first case fails (numerator 1.333 for 1.5). | `m(u, t)` is smaller where an edge shorter than about `2e-5 t` straddles the shell. No change otherwise. |
| fr-fields-09 | fixed | `gpu-network/network-coarsening/gpu-network-coarsening.ts:231-262`: with `fixedPointScale` given, `quantize` saturates a term, with its sign, at the float32 nearest 4e18 units (the old clamp), also for an infinite term; the data-derived path is unchanged, since no term can leave two words there. Class text `:303-309`, `docs/api/gpu-network/operators.md` (coarsening paragraph). The oracle's fixed-scale quantizer has the clamp back (`test/gpu-network/network-coarsening/network-coarsening-oracle.ts:102-111`); I had dropped it there too. | `test/gpu-network/network-coarsening/gpu-network-coarsening.spec.ts` ("saturates a term too large for a fixed fixedPointScale"): weights 1.5e14, 3.4028234e38 and -2e14 at 65536 give `fround(fround(4e18) / 65536)` with their signs. Before the fix 1.5e14 decoded as -1.31e14. | With `fixedPointScale` pinned, a term above `4e18 / fixedPointScale` adds the saturated value again instead of wrapping or vanishing. |
| fr-fields-03 | fixed (sort plus search, no cap) | `gpu-network/recipes/drive-time-catchment-recipe.ts:279-370` (`addSnappedRoadNodes` also writes a road key: the smallest edge row between the two end nodes in either direction, the edge capacity when unsnapped), `:626-677` (facility fractions with the edge orientation in the sign bit, sorted by road key with the published `GPUSort`), `:880-905` (the demand kernel binary-searches its road key with the shared `getBinarySearchWGSL` and walks only that run). Class text `:445-449` and `docs/api/gpu-network/operators.md` say so. The loop inside one invocation now runs over the facilities snapped to the demand's own road. | `test/gpu-network/recipes/drive-time-catchment-direct-travel.spec.ts` ("finds every facility on the road of a demand point"): three facilities listed out of road order, two on one road; exact times 4, 2, 2, 6, 4 and node costs 2, 4, 8. The earlier hand cases and the inserted-node oracle still pass. | None in results. Work per demand point is a binary search plus the facilities on its road, not every facility. This replaces the "compared with every facility" note in the network-flows-02 row above. |

Not done: the sentence the reviewer suggested about spatstat's tolerance being relative to the
shortest segment. It cannot be checked on this machine, so the class text states only this
contributor's rule (relative to the pair distance).
