# Math correctness review: network paths (`network-paths`)

Date: 2026-11-10. Scope: `modules/analysis/src/gpu-network/` directories `network-analysis`,
`network-reachability`, `network-accessibility`, `network-isochrones`, `network-centrality`,
`network-statistics`, `network-structure`, `network-cost-weights`, `network-cost-table`,
`network-time-geography`, `alternative-routes`, `network-optimization`, `network-k-function`,
`network-point-process`. Paths below are relative to `modules/analysis/src`.

Summary: 10 findings. 1 `wrong-result`, 1 `wrong-edge-case`, 2 `precision`, 4 `inconsistent`,
2 `doc-mismatch`. 9 confirmed, 1 suspected. The shortest-path core (`GPUNetworkReachability` and
the frontier) is correct, and so is every contributor that only composes it. No contributor in
scope delegates to `@luma.gl/graph/gpu-graph-algorithms`.

## 1. Coverage

Read in full (kernel bodies, packers and class documentation):

- `gpu-network/network-reachability/network-reachability-passes.ts`, `network-frontier.ts`,
  `gpu-network-reachability.ts`
- `gpu-network/network-analysis/network-path-extraction-passes.ts`,
  `network-service-areas-passes.ts`, `network-neighborhood-passes.ts`,
  `gpu-network-analytics-columns.ts`, `gpu-network-service-areas.ts`
- `gpu-network/network-centrality/network-centrality-passes.ts`, `gpu-network-centrality.ts`
- `gpu-network/network-accessibility/network-accessibility-passes.ts`,
  `network-accessibility-lanes.ts`, `network-edge-sources.ts`
- `gpu-network/network-isochrones/network-isochrones-passes.ts`,
  `network-isochrones-parameters.ts`
- `gpu-network/network-statistics/gpu-network-statistics.ts`
- `gpu-network/network-structure/gpu-network-structure.ts`
- `gpu-network/network-optimization/gpu-network-flow.ts`, `gpu-network-facility-optimization.ts`,
  `prepare-flow-demands.ts`
- `gpu-network/network-k-function/network-k-function-parameters.ts`
- `gpu-network/network-point-process/index.ts` and the two files it re-exports,
  `gpu-network/gpu-network-kernel-density.ts` and `gpu-network/gpu-network-inhomogeneous-k.ts`

Read in full except the import block:

- `gpu-network/network-accessibility/gpu-network-accessibility.ts`, `gpu-network-cost-matrix.ts`
- `gpu-network/network-cost-weights/gpu-network-cost-weights.ts`
- `gpu-network/network-cost-table/gpu-network-cost-table.ts`
- `gpu-network/network-time-geography/gpu-network-potential-path-area.ts`,
  `gpu-network-travel-time-percentiles.ts`
- `gpu-network/alternative-routes/gpu-alternative-routes.ts`
- `gpu-network/network-optimization/gpu-network-time-routing.ts`, `gpu-network-point-routing.ts`,
  `gpu-network-rational-access.ts`, `gpu-network-louvain.ts`
- `gpu-network/network-k-function/gpu-network-k-function.ts`

Read in part (kernels and class documentation read, constructor validation skipped):

- `gpu-network/network-analysis/gpu-network-neighborhood.ts` (lines 140 to 195 skipped)
- `gpu-network/network-analysis/gpu-network-path-extraction.ts` (lines 170 to 300 skipped)
- `gpu-network/network-accessibility/gpu-network-snapping.ts` (lines 257 to 330 skipped)
- `gpu-network/network-isochrones/gpu-network-isochrones.ts` (lines 330 to 520 skipped)
- `gpu-network/network-reachability/network-reachability-tuning.ts` (code read, comments skipped)

Read outside the scope because the scope depends on them:
`analysis-core/kernels/masked-weak-components.ts` (full), `gpu-network/network-spatial-context.ts`
(full), `gpu-network/network-workgroup-counters.ts` (part),
`analysis-core/kernels/permutation-random.ts` (`nextPhiloxBelow` only),
`analysis-core/kernels/wgsl-common.ts` (finite tests only).

Not opened: the `index.ts` of every directory except `network-point-process`. Also not opened,
because other scopes own them: `GEODESIC_WGSL` (used by snapping), `GPUNearestFeatureJoin`,
`GPUCatchmentAccessibility`, `GPUIsobands`, `GPUCellSetOutline`, `GPUPointToCell`,
`facility-score-reduction.ts`, `load-balanced-for-each.ts`. A contributor that delegates to one of
these is verified only up to the call.

Official luma.gl read for comparison: the `.d.ts` of `GPUGraphAlgorithmsSingleSourceShortestPath`,
`...BreadthFirstSearch`, `...ConnectedComponents`, `...Degree`, `...Modularity`,
`...ModularityOptimization`, `...Topology`; the relax and predecessor kernels of the shortest-path
internals; the bin formula of `GPUHistogram`.

No existing spec was run (0 of 5). Five scratch scripts were run; they are in the scratchpad
directory `network-paths/` (`isochrone-max.mjs`, `sigma-overflow.mjs`, `sigma-delta-grid.mjs`,
`k-quantize.mjs`, `louvain-seed.mjs`).

## 2. Findings

| ID | Severity | Status | File:line | What the kernel computes | What the reference says | Input that gives a wrong output | Suggested fix |
| --- | --- | --- | --- | --- | --- | --- | --- |
| network-paths-01 | wrong-result (mode `'max'` only) | CONFIRMED (script) | `gpu-network/network-isochrones/network-isochrones-passes.ts:186`, `:209-210` | Per pixel, the maximum over every sample of every directed edge of `cost[source] + t * weight`. On a two-way road that is `max(c_u + t w, c_v + (1 - t) w)`. | A conservative isochrone surface is the largest, over the streets touching a pixel, of each street point's shortest cost `min(c_u + t w, c_v + (1 - t) w)`. | Two nodes, one two-way road of weight 10, start at `u`. The pixel on the start node gets 20. The true cost is 0. | In `'max'` mode take the minimum over the two directions of a road first (splat `min(c_u + t w, c_v + (1 - t) w)` when the reverse edge exists), then the maximum over roads. Or document `'max'` as one-way only. |
| network-paths-02 | wrong-edge-case (documented limit) | CONFIRMED (derivation) | `gpu-network/network-centrality/network-centrality-passes.ts:102-105`, `:428-433` | Zero-weight edges are never tight. A node reached only across one becomes a root with sigma 1, and no dependency flows back across the edge. | Brandes / `networkx.betweenness_centrality` count paths across zero-weight edges. | Path `s -1-> a -1-> m -0-> t -1-> u`, pivot `s`. Kernel: `a` 1, `m` 0, `t` 1. networkx: `a` 3, `m` 2, `t` 1. `status.converged` stays 1. | Order equal-cost plateaus with the tie-level phase reachability already has (tight when costs are equal and the level rises by one). Until then, report the number of zero-weight tight edges in a status word. |
| network-paths-03 | precision | CONFIRMED (script) | `gpu-network/network-centrality/network-centrality-passes.ts:478-482`, `:589` | Path counts (sigma) are f32 sums. They overflow to `+Infinity`, then `inf / inf` makes delta NaN. | networkx keeps exact integers on unweighted graphs and f64 on weighted ones. | Unit-weight 4-neighbour grid, corner pivot. 66 x 66 is correct. 67 x 67: one sigma is `Infinity` and its dependency is dropped (sum of dependencies 291,655, exact 291,786). 68 x 68 with `maximumIterations >= 135`: 4,623 of 4,624 delta values are NaN, so every betweenness is NaN. `status.converged` stays 1. | Keep sigma as a mantissa and a base-2 exponent (two words; delta needs only ratios of neighbouring sigmas), or at least detect a non-finite sigma and clear `converged`. Add the limit to the class documentation, which today only mentions rounding above 2^24. |
| network-paths-04 | precision | CONFIRMED (script) | `gpu-network/network-k-function/gpu-network-k-function.ts:72-75`, `:543-548` | The random edge of a simulated event is drawn with weight `u32(length * range / longestEdge)` (truncated). `range` is 2^15 at 100,000 edges, 2^12 at 600,000 and 2^11 at 2,000,000. | spaghetti draws an edge with probability proportional to its exact length. | 600,000 edges, longest 10 km, lognormal lengths with median 80 m: edges under 10 m are drawn 16% too rarely, 10 to 30 m 4.7% too rarely, edges over 1 km 0.9% too often. At 2,000,000 edges and a 30 km longest edge, 3% of edges are never drawn and 10 to 30 m edges are 34% too rare. | Scale against the sum, not the maximum: `unit = sum / (2^32 - edgeCount)` gives the mean edge about `2^32 / edgeCount` units. Round to nearest instead of truncating. Or draw in two levels (row by exact prefix sum of row totals, then edge inside the row). |
| network-paths-05 | doc-mismatch | CONFIRMED (reading) | `docs/api/gpu-network/operators.md:1141` against `gpu-network/network-k-function/gpu-network-k-function.ts:524-548` | The quantization scale is the GPU maximum of the live weights. | The user page says "quantized against the GPU total of the live weights". The class documentation (`:193`) says "longest live edge". | Any network: the page and the code disagree. | Fix with network-paths-04; the page describes the better scheme. |
| network-paths-06 | inconsistent | CONFIRMED (derivation) | `gpu-network/network-analysis/gpu-network-analytics-columns.ts:304` | `degree = offsets[i + 1] - offsets[i]`. It counts slots whose neighbor is `>= nodeCount`, and counts a self-loop once. | The same file (`:60-61`) and `analysis-core/network-topology.ts:32` say such neighbors are skipped. `GPUNetworkStatistics` skips them and counts an undirected self-loop twice (`gpu-network/network-statistics/gpu-network-statistics.ts:461`). | Row with slots `[1, 0xffffffff]`: degree 2, documented 1. One node with one self-loop slot: `GPUNetworkAnalyticsColumns` degree 1, `GPUNetworkStatistics` maximum degree 2. | Count live slots in the degree kernel (one bounded row loop), and state the self-loop rule in the class documentation. Pick one self-loop rule for both contributors. |
| network-paths-07 | inconsistent | CONFIRMED (reading) | `gpu-network/network-reachability/network-reachability-passes.ts:593` against `gpu-network/network-isochrones/network-isochrones-parameters.ts:11-13` | Reachability band `i` is `(break[i - 1], break[i]]`. Isochrone raster band `k` is `[break[k - 1], break[k])`. | One interval rule for "within t minutes". Accessibility (`<= threshold`), `costLimit` and `cellCostLimit` are inclusive; only the raster bands and the K function (`<`, spaghetti) are exclusive. | A node with cost exactly 300 and a break at 300: `GPUNetworkReachability.bands` says band 0 (inside), the isochrone raster says band 1 (outside). | Document the difference on both pages, or make the isochrone bands upper-inclusive if `GPUIsobands` allows it. |
| network-paths-08 | inconsistent | CONFIRMED (script) | `gpu-network/network-optimization/gpu-network-louvain.ts:171-174` | The seeded visit order uses a private xorshift32 and `state % remaining`. | Contract section 11: random numbers come from the Philox stream with a registered purpose. `nextPhiloxBelow` is unbiased. | Seed `0x9e3779b9`: the state is 0 forever, every draw is 0 and the "random" order is the rotation `1,2,...,n-1,0`. Other seeds carry a modulo bias of at most 32 / 2^32. | Use `createPhiloxStream` with a registered purpose and `nextPhiloxBelow`. |
| network-paths-09 | inconsistent | CONFIRMED (reading) | `gpu-network/network-reachability/gpu-network-reachability.ts:104-106`, `gpu-network/network-analysis/gpu-network-service-areas.ts:98-99`, `gpu-network/network-isochrones/gpu-network-isochrones.ts:264`, `docs/api/gpu-network/operators.md:22-28` | Three round budgets for the same search: `ceil(h / (0.7 L)) + 4`, `ceil(h / L) + 1` and `ceil(h / L)`. | `recommendReachabilityIterations` uses the first one (`network-reachability-tuning.ts`). | `h = 224`, `L = 16`: 24 rounds by the first rule, 15 and 14 by the others. A caller who sizes by the others gets `converged = 0`. | Point the service-area and isochrone documentation at `recommendReachabilityIterations`. |
| network-paths-10 | doc-mismatch | SUSPECTED | `dev-docs/upstream-workarounds.md:24` (row "Command-graph-transient CSR and column views") | The row says `network-statistics`, `network-structure` and `gpu-network-analytics-columns.ts` "use physical vectors. Nothing to restore here." | All three take `GraphDataView`s and run local kernels: degree, weak components (`createMaskedWeakComponentNodes`) and modularity are local re-implementations of published algorithms. | Not a numeric defect. | Record the three re-implementations as workarounds with "what luma.gl needs: algorithms that accept command-graph views". See section 5. |

### network-paths-01: isochrone `'max'` mode

`createIsochronesSplatNode` runs one thread per directed CSR slot. In `'min'` mode this is right:
the two directions of a road give `c_u + t w` and `c_v + (1 - t) w`, and the pixel minimum is the
shortest cost of the street point. In `'max'` mode the same two samples are combined with a
maximum, so each street point takes the cost of its worse direction. The library's own convention
is that undirected roads list both directions, so this is the ordinary input.

`isochrone-max.mjs` replicates the kernel arithmetic for one two-way road of weight 10 and length
100, 11 pixels of 10 units:

```
pixel centre x : 5 15 25 35 45 55 65 75 85 95 105
min raster     : 0 0.34 1.38 2.41 3.45 4.48 5.52 6.55 7.59 8.62 9.31
max raster     : 20 19.66 18.62 17.59 16.55 15.52 14.48 13.45 12.41 11.38 10.69
true cost at x : 0 1 2 3 4 5 6 7 8 9 10
```

The start node is the most expensive pixel. A band at 15 covers only the far half of the road.
The class documentation (`gpu-network/network-isochrones/gpu-network-isochrones.ts:80-84`) says
`'max'` "keeps the largest, a conservative surface where a pixel is within a cost only when every
street sample touching it is". The kernel does keep the largest, so the sentence is literally
true, but the surface is not a conservative isochrone. The lead reviewer may prefer
`doc-mismatch`; I rate it by what a user gets. With a walking buffer the maximum also picks the
farthest sample inside the buffer, which inflates one-way networks too.

Only a CPU copy of the same loop pins this mode
(`test/gpu-network/network-isochrones/network-isochrones-oracle.ts`). The spec relaxes its
assertions for `'max'` with the comment "Max mode raises every pixel next to a costly edge".

### network-paths-02: zero-weight edges in betweenness

The class documentation states the rule
(`gpu-network/network-centrality/gpu-network-centrality.ts:217-220`), so this is a known limit and
not a surprise in the code. I list it because the consequence is larger than the sentence
suggests: it is not only that the plateau node counts one path. Every node upstream of a
zero-weight edge loses the whole dependency through it, and no status word says so. The same
happens when a positive weight is below half an ulp of the cost (4 mm at 100 km in metres).
The documentation also claims networkx parity; that holds only without zero-weight edges on
shortest paths. Graphs with zero-cost connectors (transfers, snapped virtual nodes) hit this.

### network-paths-03: sigma overflow

`sigma-overflow.mjs` sums f32 path counts the way the sigma round does, and
`sigma-delta-grid.mjs` replicates both phases for a corner pivot:

```
grid 66x66: far-corner sigma (f32) = 9.50676291645977e+37
grid 67x67: far-corner sigma (f32) = Infinity
inf / inf * (1 + 0) = NaN ; finite / inf = 0

k   infinite sigma   NaN delta   sum of delta   exact sum
66        0              0        278785.01      278785
67        1              0        291655.02      291786
68        6           4623        NaN
```

A predecessor with a finite sigma under an infinite successor contributes 0, so at 67 x 67 the
result is finite and wrong (the far corner's 131 units of dependency are missing). One with an
infinite sigma contributes NaN, and NaN then flows to every ancestor: at 68 x 68 every node but
the far corner is NaN for that pivot, and the lane sum makes every betweenness NaN. Real road
networks with float weights have few exact ties, so sigma stays small. Unit weights (hop
centrality) on a lattice are the case that breaks. The configuration is legal: 4,624 nodes,
about 21 batches at the default lane count, `maximumIterations: 135`, about 8,700 command
nodes.

### network-paths-04: edge draw of the K-function envelope

The draw is correct in form (integer prefix sum, unbiased `nextPhiloxBelow`, binary search for the
first prefix above the draw). The loss is in the unit. The longest edge maps to `range` units and
`range = 2^floor(log2((2^32 - 1) / edgeCount))`, so a network with a long tail of edge lengths
gives its typical edge only a few units, and truncation removes half a unit on average. Short
edges lose the most. Simulated events then fall too rarely on dense short-edge areas, the
simulated K is biased low at short distances, and the observed pattern looks more clustered than
it is against the envelope. `k-quantize.mjs` output (synthetic lognormal lengths, median 80 m):

```
E=100000  longest=10000 m range=32768 unit=0.31 m : 0-10 m -1.89% | 10-30 m -0.59% | 30-100 m -0.12%
E=600000  longest=10000 m range=4096  unit=2.44 m : 0-10 m -15.79% | 10-30 m -4.69% | 30-100 m -1.00%
E=2000000 longest=10000 m range=2048  unit=4.88 m : 0-10 m -36.11% | 10-30 m -9.61% | 30-100 m -1.97%
E=2000000 longest=30000 m range=2048  unit=14.65 m: 0-10 m -100% | 10-30 m -34.43% | 30-100 m -5.90%
```

Below about 100,000 edges the effect is under 2% for the shortest class and can be ignored. The
length distribution is my assumption; the mechanism does not depend on it.

## 3. Verified correct

- `GPUNetworkReachability` costs: chaotic Bellman-Ford with `atomicMin` on non-negative float bits
  (`-0` folded). The result is the unique f32 fixpoint, independent of round and hop settings.
  Negative and NaN weights are skipped, `costLimit` is inclusive, unreached is `+Infinity`.
  Compared against the Bellman-Ford invariant by hand and against the official relax kernel.
- `network-frontier.ts`: round stamps admit a node once per round, every successful improvement
  is followed by a push (local queue, or next round), a push past the limit sets `truncated`.
  No lost update found. Local queue counters are zero at the start of every chunk.
- `GPUNetworkReachability` predecessors: smallest strictly cheaper tight in-neighbour, then
  levels over equal-cost tight edges. Acyclic by construction (cost falls, or level falls).
  Tightness uses the same f32 expression as the relaxation.
- `GPUNetworkReachability` bands: `(break[i - 1], break[i]]`, as documented. `bandCounts` goes
  through `GPUHistogram` with domain `[0, B]`; its `uint32` path is an exact integer
  `floor(b * B / B)`.
- `GPUNetworkServiceAreas`: minimum-label propagation over tight edges from facilities whose seed
  cost equals the node cost. Fixpoint is the lowest facility row among those that reach the node
  at minimum cost. Predecessors restricted to the same label end at the named facility.
- `GPUNetworkPathExtraction`: bounded predecessor walk, exact off-by-one at `maximumPathLength`,
  a `NONE` ending counts as a start only when the search converged, edges resolved by the same
  tightness test.
- `GPUNetworkNeighborhood`: unit weights, `activeIterations = hops`. After `k` rounds every node
  within `k` hops holds its exact hop count (checked with the frontier's push rule and local
  chaining), farther nodes hold more than `k` or infinity.
- `createMaskedWeakComponentNodes` (used by two contributors): pulls and pushes the minimum label
  over both endpoints, so a forward CSR gives weak components of a directed graph. Converged
  means the last iteration changed nothing.
- `GPUNetworkCentrality` betweenness: Brandes sigma and delta on the strict tight-edge graph,
  topological order by release counters, pivot excluded, endpoints not counted, no 1/2 factor
  (documented). Round cap `min(maximumIterations, nodeCount)` is sufficient. Edge betweenness:
  `sigma_v / sigma_w * (1 + delta_w)` per directed slot; the documented statement that a slot of
  a symmetric CSR equals the undirected value is right. Departures: network-paths-02 and -03.
- `GPUNetworkCentrality` closeness: `(r - 1) / S` outward, harmonic sum over costs above 0,
  Wasserman-Faust not applied. All three are documented and match networkx on `G.reverse()`.
- `GPUNetworkAccessibility`: cumulative `sum_j O_j [c_ij <= T]`, gravity `sum_j O_j f(c_ij)` with
  `exp(-beta c)` or `max(c, c_min)^-beta`, no normalisation (Hansen form). NaN and infinite
  weights skipped. Fixed-order sums.
- `GPUNetworkCostWeights`: inclusive `costLimit`, decays as above plus the Gaussian of Dai (2010),
  `(exp(-c^2 / (2 L^2)) - e^-1/2) / (1 - e^-1/2)`. The transpose orientation is exact. The
  Hillis-Steele slot scan is correct.
- `GPUNetworkCostMatrix`: per-row lane seeds, in-place and copied batches address the same
  elements, target gather writes `+Infinity` for out-of-range targets.
- `GPUNetworkSnapping`: nearest edge with ties to the lowest row, split costs `f w` and
  `(1 - f) w`, start-direction rules (`'forward'` keeps the target, `'reverse'` the source).
  Planar projection formula checked; the great-circle helpers were not opened.
- `GPUNetworkCostTable`: per-lane count, scan on a running cursor with saturation, ordered fill.
- `GPUNetworkIsochrones` raster, `'min'` mode: linear interpolation along each directed edge,
  pixel centres at `min + (i + 0.5) * cell` in both the splat and `GPUIsobands`, pixel reach
  `ceil(radius / cell)` covers every centre within the radius, source lookup by binary search is
  right with empty rows. Each pixel takes the cheapest sample within the radius, so polygons are
  larger by up to the radius (at least half a pixel diagonal); the documentation says "accurate
  to about a pixel". Labelled encoding keeps 24 cost bits as documented.
- `GPUNetworkIsochrones` cell path: mask `cost <= cellCostLimit`, cell facility by minimum cost
  bits then lowest row.
- `GPUNetworkStatistics`: live slot, edge and self-loop counts; undirected modularity
  `intra / 2m - gamma * sum (d_c / 2m)^2` with `2m = liveSlots + selfLoops`; directed
  `intra / m - gamma * sum out_c in_c / m^2`. Matches networkx and the official
  `GPUGraphAlgorithmsModularity` formula. Log2 bins are `1 + floor(log2 d)`.
- `GPUNetworkStructure`: iterative Tarjan with the twin-slot test (parallel edges are not
  bridges), `low > disc` for bridges, `low >= disc` and root child count for articulation
  points, edge stack for blocks; strong components by mutual reachability with the lowest ID.
- `GPUNetworkPotentialPathArea`: edge in the prism when `forward[u] + w + reverse[v] <= budget`,
  node when `forward[v] + reverse[v] <= budget`.
- `GPUNetworkTravelTimePercentiles`: nearest rank `ceil(p n / 100)` with a 1e-9 guard, unreached
  as `+Infinity` sorted last. The rule is the documented one, not checked against r5r.
- `GPUNetworkTimeRouting`: label-correcting earliest arrival with FIFO affine or piecewise-linear
  arrival functions; `stateCount` rounds are enough to converge and detect it.
  `prepareGPUNetworkRestrictionStates` is a correct longest-prefix automaton.
- `GPUNetworkPointRouting`: edge-state Bellman-Ford, source pays `(1 - f_s) c`, target pays
  `f_t c`, same-edge forward travel competes with cycles.
- `GPUNetworkFlow`: successive shortest augmenting paths on residual arcs, `nodeCount - 1`
  Bellman-Ford rounds, cut from the last search, certificate `cutCapacity == value` only when the
  sink is unreachable. The 64-bit exact cost carries correctly. `prepareGPUNetworkFlowDemands`
  builds a correct super source and sink.
- `GPUNetworkFacilityOptimization`: exhaustive backup coverage, p-dispersion (max-min) and
  capacitated p-median; score ordering by float bits is valid for non-negative scores.
- `GPUNetworkRationalAccess`: cost `travel / tau + load / (rho * supply)` and the pairwise
  equalising shift `(high - low) / (1 / cap_min + 1 / cap_max)` match Saxon and Snow. Verified by
  reading only; the oracle is one 2 x 2 case.
- `GPUNetworkLouvain`: objective `internal / total - gamma * sum (out_c / total)(in_c / total)` is
  Newman modularity for the documented matrix convention (diagonal holds twice the loop weight).
- `GPUAlternativeRoutes`: best-first search over simple-path prefixes; pops are in non-decreasing
  f32 cost, ties in lexicographic CSR edge order. Equivalent to `shortest_simple_paths` output.
- `GPUNetworkKFunction`: `K = 2 pairs L / n^2` with strict `<`, spaghetti's `GlobalAutoK` form;
  pair distance is the cheaper endpoint route or the same-road offset difference; band placement
  is corrected against the same threshold expression; simulated patterns have `n` events.
- `GPUNetworkKernelDensity`: triangular kernel divided by each event's quadrature mass, so
  `sum(length * intensity) = eventCount`.
- `GPUNetworkInhomogeneousK`: `sum 1 / (lambda_i lambda_j m(i, d_ij)) / denominator`; the Ang
  shell count (vertices at the distance plus interior roots of
  `min(d_a + s, d_b + l - s) = r`) and the three denominators match `linearKinhom`.

## 4. Standards consistency

- Two self-loop degree rules and two dead-slot rules in sibling contributors
  (network-paths-06).
- Two band interval rules (network-paths-07).
- Three round-budget formulas for one search (network-paths-09).
- Random numbers: `GPUNetworkKFunction` uses Philox with a registered purpose. `GPUNetworkLouvain`
  uses a private xorshift32 (network-paths-08). No other randomness in scope.
- Float "no value" has five spellings: `+Infinity` (costs, routes, arrivals), `-1` (snapping
  fractions, distances and costs), `1e30` (isochrone raster `unreachedCost`), NaN (facility
  objective, rational-access scores, Louvain trace, kernel density) and `-Infinity` (prism edge
  slack). Each is documented where it appears. Integer "none" is `0xffffffff` everywhere.
- Private finite tests: `fn finite` is declared in `gpu-network-potential-path-area.ts:29`,
  `gpu-network-time-routing.ts:187`, `gpu-network-point-routing.ts:135`,
  `gpu-network-rational-access.ts:172`, `gpu-network-facility-optimization.ts:199` and
  `gpu-network-louvain.ts:145`. All are bit-identical to `IS_FINITE_BITS_WGSL` in
  `analysis-core/kernels/wgsl-common.ts`, which the accessibility, cost-weights, cost-table,
  isochrone and K-function kernels import.
- Private binary searches: `isochronesFindSource` and `isochronesFindCellRow`
  (`network-isochrones-passes.ts:129-141`, `:317-341`) and two in the K-function `simulate` kernel
  (`gpu-network-k-function.ts:596-616`). All four are correct. `analysis-core/kernels` has
  `binary-search-wgsl.ts`.
- Shader literals: `INFINITY_BITS = 0x7f800000u` and `NONE = 0xffffffffu` are redeclared in the
  reachability, service-area, path-extraction and centrality pass files.
- Distance on the sphere and the Earth radius: no private copy. Snapping takes the radius from
  `getGPUGreatCircleRadius(spatialContext)` and the public `GEODESIC_WGSL` helpers.
- Sums: per-row and per-lane float sums use fixed-order strided partials and a binary tree
  (accessibility, closeness, modularity, inhomogeneous K). Betweenness sums lanes in lane order.
  No float atomics in scope.

## 5. luma.gl official implementations

No file in the scope imports `@luma.gl/graph/gpu-graph-algorithms`. Every graph algorithm here is
local. The reason given in the class documentation is the same each time: the official classes
take a physical `GPUGraphAlgorithmsTopology` built from `GPUGraphAlgorithmsData` and write
`GPUVector` outputs, while these contributors read per-frame `GraphDataView`s (transient CSRs,
masked copies, lane-expanded costs). That limit is real, so delegation is possible only for a
caller who already holds a physical graph.

| Official class | Local code | Delegates | Math agrees | Delegation possible |
| --- | --- | --- | --- | --- |
| `GPUGraphAlgorithmsSingleSourceShortestPath` | `GPUNetworkReachability` and everything composed from it | No | Distances: yes, both are `atomicMin` on float bits of `distance + weight`, so the same f32 fixpoint. Differences: invalid weights are impassable locally and fail closed upstream; missing weights are 1 upstream and required locally; the predecessor rule differs (both give a valid shortest-path tree); upstream runs dense rounds, one hop each, at most 1,024, so paths over 1,024 hops cannot converge there. | Only single source, static physical graph, no start costs, no cost limit, no lanes. Not for any current composer. |
| `GPUGraphAlgorithmsBreadthFirstSearch` | `GPUNetworkNeighborhood` (unit-weight reachability) | No | Yes: hop distance, `0xffffffff` unreached, seeds with a count, depth cap 1,024 on both sides. | For a physical graph, yes; the interfaces are close. Not for a transient CSR. |
| `GPUGraphAlgorithmsConnectedComponents` | `createMaskedWeakComponentNodes` in `GPUNetworkAnalyticsColumns` and `GPUNetworkStatistics` | No | Yes: weak components, lowest vertex ID, default 32 iterations, converged means the last iteration was a fixed point. | Not for the masked copy in statistics. Possible for an unmasked physical graph. |
| `GPUGraphAlgorithmsDegree` | degree kernel of `GPUNetworkAnalyticsColumns`; live degree in `GPUNetworkStatistics` | No | `GPUNetworkAnalyticsColumns` agrees (offset difference, undirected self-loop counts one) but upstream removes invalid edges when it builds the topology, and the local kernel does not (network-paths-06). `GPUNetworkStatistics` counts a self-loop twice, on purpose. | Same limit. |
| `GPUGraphAlgorithmsModularity` | modularity in `GPUNetworkStatistics`; objective in `GPUNetworkLouvain` | No | Yes, same formula. The local sum is a fixed-order tree over exact integer counts; upstream documents order-dependent float atomics. Upstream is weighted; statistics is unweighted. | Not with masks or a CSR-only input: upstream reads the edge list. |
| `GPUGraphAlgorithmsModularityOptimization` | `GPUNetworkLouvain` | No | Different algorithms. Upstream is single-level, one accepted move per round. Local is multilevel Louvain on a dense matrix of at most 32 nodes. Same objective. | Not a substitute in either direction. |
| `...PageRank`, `...CoreNumber`, `...LabelPropagation`, `...LocalClusteringCoefficient` | None in scope | n/a | n/a | The user page sends callers to the official classes (`docs/api/gpu-network/operators.md:171`). |

Points for the upstream list:

- Upstream itself has two self-loop rules: `GPUGraphAlgorithmsDegree` counts an undirected
  self-loop once, `GPUGraphAlgorithmsModularity` adds it twice to the volume. The two local
  contributors each follow a different one.
- `dev-docs/upstream-workarounds.md` does not list the three local re-implementations (degree,
  weak components, modularity) as workarounds (network-paths-10). They fit the "small helper
  named for the analysis domain" rule only loosely. The proposal to make upstream is "published
  graph algorithms that accept command-graph views".
- Not in luma.gl at all, so nothing to delegate: betweenness and closeness, strongly connected
  components, bridges and articulation points, k shortest simple paths, flow, service areas,
  network K functions.
- No minimum spanning tree, travelling-salesman or vehicle-routing code exists in the scope.
  `GPUAlternativeRoutes` is k shortest simple paths, not a penalty or plateau method.

## 6. Test oracles

Independent oracle (a reference library, fixtures under `test/oracles/fixtures`):

- `GPUNetworkReachability`, `GPUNetworkPathExtraction`, `GPUNetworkCostMatrix`,
  `GPUNetworkServiceAreas`, `GPUNetworkNeighborhood`, `GPUNetworkAnalyticsColumns`: networkx
  (`network-shortest-paths.py`, `parity-wp3-routing.py`, `parity-wp3-cost-matrix.py`,
  `parity-wp3-service-areas.py`, `parity-wp3-network-analysis.py`).
- `GPUNetworkCentrality`: networkx (`network-centrality.py`). I did not check whether a fixture
  has a zero-weight edge or a large lattice; findings -02 and -03 suggest none does.
- `GPUNetworkAccessibility`, `GPUNetworkCostWeights` with 2SFCA: the `access` package
  (`parity-wp3-network-accessibility.py`, `network-cost-weights-2sfca.py`).
- `GPUNetworkSnapping`: shapely and pyproj (`parity-wp3-snapping.py`).
- `GPUNetworkKFunction`: spaghetti (`parity-wp3-network-k-function.py`). The observed K is
  pinned. The simulated draw cannot be pinned by spaghetti (different generator), so
  network-paths-04 is not covered.
- `GPUNetworkStatistics`: networkx (`network-statistics.py`).
- `GPUNetworkStructure`, `GPUAlternativeRoutes`: networkx (`network-expansion.py`,
  `alternative-routes.py`).
- `GPUNetworkPotentialPathArea`: networkx (`network-time-geography-mob09.py`).
- `GPUNetworkFlow`, `GPUNetworkLouvain` (objective only): networkx
  (`network-movement-coverage.py`).
- `GPUNetworkFacilityOptimization`: spopt with CBC; `GPUNetworkRationalAccess`: `access.raam`
  (`network-movement-advanced.py`). Each is one small case.
- `GPUNetworkInhomogeneousK`: a fixture of independently enumerated metric spheres. I did not
  check whether spatstat produced it.

Pinned only by a copy of the same formula:

- `GPUNetworkIsochrones` raster, both modes: `network-isochrones-oracle.ts` repeats the splat
  loop. This is why network-paths-01 passes its test.
- `GPUNetworkTravelTimePercentiles`: the spec repeats the nearest-rank expression.

In-repo CPU reference (a separate algorithm, not a reference library):

- `GPUNetworkCostTable`: expected rows come from `network-reachability-oracle.ts`.

Hand-computed cases only: `GPUNetworkTimeRouting`, `GPUNetworkPointRouting`.

Not determined: `GPUNetworkKernelDensity` (only an invalid-count spec was seen under
`test/gpu-network`).

## Resolution (2026-11-10)

9 fixed, 0 not reproduced, 1 deferred (to the `luma-official` stream, which owns the file). One
extra defect was found and fixed while pinning network-paths-08. Paths are relative to
`modules/analysis`; specs are under `test/gpu-network`.

| ID | Outcome | What changed (`file:line`) | Spec that pins it | Behaviour change a caller sees |
| --- | --- | --- | --- | --- |
| network-paths-01 | fixed | `src/gpu-network/network-isochrones/network-isochrones-passes.ts:116-131` (reverse slot lookup), `:247-262` (sample cost is `min(cost[u] + t w, cost[v] + (1 - t) wReverse)` in `'max'` mode). Class and prop documentation in `gpu-network-isochrones.ts`. The CPU mirror `test/gpu-network/network-isochrones/network-isochrones-oracle.ts` encoded the old rule and was changed with it. | `network-isochrones/gpu-network-isochrones-max-mode.spec.ts`: single two-way road (hand values), one-way road, two roads with a buffer (hand values and bounds). Failed before the fix (start pixel 20). | `'max'` raster on a two-way road of weight 10: start pixel 20 before, 0.69 now (the costliest sample within the half-pixel-diagonal radius); far pixel 10.69 before, 10 now. `'min'` mode is unchanged. Among parallel slots `v -> u` the costliest is taken as the reverse direction. |
| network-paths-02 | fixed | `src/gpu-network/network-centrality/network-centrality-passes.ts:105-108` (a zero-weight edge between equal costs is tight, a self-loop never), `:578-585` (unordered nodes detected on the GPU), `:715-745` (batch status), `src/gpu-network/network-centrality/gpu-network-centrality.ts:103-113` (new optional `output.status.zeroWeightCycle`). CPU mirror `network-centrality-oracle.ts` updated (Kahn order, same tight rule). | `network-centrality/gpu-network-centrality-path-counts.spec.ts`: the report's path gives 3, 2, 1 (networkx values); an acyclic zero-weight plateau by hand; a zero-weight cycle sets the flag; a round limit does not. `gpu-network-centrality.spec.ts` "zero cycles" updated: it pinned the old strict rule. | Zero-weight edges on shortest paths are now counted when they form no cycle (`a`, `m`, `t`: 1, 0, 1 before; 3, 2, 1 now). With a cycle of zero-weight tight edges (an undirected zero-weight edge lists both directions and is one): `status.converged` was 1 with strict-rule values, is now 0, `status.zeroWeightCycle` is 1 and the betweenness is incomplete. |
| network-paths-03 | fixed | `src/gpu-network/network-centrality/network-centrality-passes.ts:110-117` (`getSigmaScale`), `:488-524` (sum at the largest exponent, normalise to mantissa and exponent), `:640-645` and `:834-838` (ratio of mantissas times an exact power of two), `gpu-network-centrality.ts:592-599` (exponent view, one more word per expanded node in its own binding). No float compensation is involved: scaling by a power of two is exact and the exponent is an integer. | `network-centrality/gpu-network-centrality-path-counts.spec.ts` "keeps path counts beyond the float32 range": 72 x 72 unit lattice, two corner pivots, against the float64 definition of the pair dependency (binomials by log-factorials), the closed-form total and the lattice symmetries. `network-centrality.node.spec.ts` binding count updated from 7 to 8. | Lattices with more than about 3.4e38 shortest paths per pair: NaN or silently short betweenness with `converged` 1 before, correct values now. Results where no overflow occurred are unchanged (same rounding). The sigma and delta rounds now use all 8 storage bindings; scratch grows from 3 to 4 words per expanded node. |
| network-paths-04 | fixed | `src/gpu-network/network-k-function/gpu-network-k-function.ts:64-95` (constants), `:572-655` (edge lengths in 2^24 units of the longest live edge, prefix sum inside chunks of 128 slots, chunk totals shifted against the largest chunk total, second prefix sum), `:693-725` (chunk draw, then edge draw). Integers and exact maxima only. CPU mirror `network-k-function-oracle.ts` updated. | `network-k-function/gpu-network-k-function.spec.ts` "draws short edges in proportion to their length": 70,000 slots, short roads hold 34.8% of the length; 800 draws within 5 standard deviations of the exact share. Failed before the fix (no short edge was ever drawn). | The simulated patterns for a given seed change (two bounded draws per event instead of one), so `kValues` rows 1.. and `envelope` change for every seed. On large networks the envelope is no longer biased toward long edges. |
| network-paths-05 | fixed | `docs/api/gpu-network/operators.md:1167-1171` and the class documentation (`gpu-network-k-function.ts`, `networkLength` and class comment) describe the two-level draw. | Documentation. | None. |
| network-paths-06 | fixed | `src/gpu-network/network-analysis/gpu-network-analytics-columns.ts:299-330` (degree counts live slots). Self-loop rule documented in both classes (`gpu-network-analytics-columns.ts` class and `degree` prop; `src/gpu-network/network-statistics/gpu-network-statistics.ts:242-247`) and in `docs/api/gpu-network/operators.md`. Each class keeps its rule (one in the degree column, two in the statistics), as luma.gl's `GPUGraphAlgorithmsDegree` and `GPUGraphAlgorithmsModularity` do. | `network-analysis/gpu-network-analytics-columns.spec.ts` "degree skips dead slots and counts a self-loop slot once" (hand values). | `degree` and `inDegree` of a row with slots whose neighbor is `>= nodeCount`: the slot count before, the live slot count now (row `[1, 0xffffffff, 7]`: 3 before, 1 now). |
| network-paths-07 | fixed | Converged on roadmap 12.3, `(a, b]`. `src/gpu-network/network-isochrones/network-isochrones-passes.ts:69-108` (`createIsochronesInclusiveBreaksNode`: `GPUIsobands` reads each break moved to the next float32 above it), `gpu-network-isochrones.ts:644-657`, `:701`. Documentation in `network-isochrones-parameters.ts:11-14`, the class and `docs/api/gpu-network/operators.md:728-734`, `:802`. `GPUNetworkReachability` bands were already `(a, b]`. | `network-isochrones/gpu-network-isochrones-max-mode.spec.ts` "raster bands are upper-inclusive". Failed before the fix. `getBandAreaBounds` in the oracle changed to the same rule. | A pixel whose cost is exactly a break: band `k + 1` before, band `k` now, in `bandClasses`, triangles and rings. Contours move by one unit in the last place. |
| network-paths-08 | fixed | `src/gpu-network/network-optimization/gpu-network-louvain.ts:176-190` (Fisher-Yates with `nextPhiloxBelow` on the stream `(seed, networkLouvainVisitOrder)`), `src/analysis-core/kernels/philox-random.ts:88-89` (registered purpose `'LOUV'`). | `network-optimization/gpu-network-louvain-visit-order.spec.ts`: 12 seeds including `0x9e3779b9`, the visit order read off the labels of 8 separate pairs, against the order from the TypeScript `PhiloxStream`, plus a fair-coin bound. Failed before the fix (the stuck seed gave the rotation). | Every seed gives a different visit order than before, so community label IDs (and on tied graphs the partition) can change for a seeded run. |
| network-paths-09 | fixed | One formula, the helper's: `ceil(h / (0.7 * localIterations)) + 4`. `src/gpu-network/network-analysis/gpu-network-service-areas.ts:98-102`, `src/gpu-network/network-isochrones/gpu-network-isochrones.ts:273-278`, `docs/api/gpu-network/operators.md:22-26` now state it and point at `recommendReachabilityIterations`. | `network-reachability/network-reachability-tuning.node.spec.ts` "gives the one round budget" (h = 224, L = 16: 24). | None (documentation). |
| network-paths-10 | deferred | `dev-docs/upstream-workarounds.md:24` belongs to the `luma-official` stream. The corrected sentence was handed to the lead. | None. | None. |

Extra finding, fixed:

- `GPUNetworkLouvain` without a seed and with 12 or more vertices made at most one move. The
  identity visit-order table, filled by a plain loop and never modified, read back as zeros on the
  test GPU, so every visit went to vertex 0; the run still reported `valid` and `converged`. 10
  vertices were right, 12 to 24 wrong (measured with separate pairs). The unseeded kernel now uses
  the loop index and has no table (`gpu-network-louvain.ts:176-180`, `:216`). Pinned by
  `gpu-network-louvain-visit-order.spec.ts` "without a seed visits vertices in ascending order"
  (16 vertices, final objective 0.875). The seeded path keeps its table; it is written by
  data-dependent swaps and matched the expected order for all 12 seeds at 16 vertices.

Checked and not present: the `network-flows` report asked whether `GPUNetworkKFunction` has the
exact float comparison of the inhomogeneous K (`value == pairDistance`). It does not: pairs are
classified with `best < threshold`, the band is corrected against the same threshold expression,
and "same road" is decided by node indices. No kernel in this scope uses a Kahan, Neumaier or
other compensated float sum.

Choice made in network-paths-02, for the lead: the tie-level rule (hop order inside an equal-cost
plateau) was not used. It gives a number for every input, but not the number of shortest paths:
with two zero-weight entries `u`, `v` joined by a zero-weight edge it drops the paths that enter at
one and leave at the other, and on the acyclic plateau `m -> t`, `m -> x -> t` it counts one path
to `t` where there are two. Ordering the full tight graph (the counters the class already has) is
exact whenever the zero-weight tight edges form no cycle, and a cycle is reported instead of
counted. The cost is that an undirected zero-weight edge on a shortest path now makes the result
invalid, where the tie-level rule would return the single-entry answer.

Upstream gaps (what luma.gl would need):

- `GPUIsobands`: an option for upper-inclusive bands `(a, b]`. Today the isochrones feed it breaks
  moved up by one float32 step.
- Graph algorithms (`GPUGraphAlgorithmsDegree`, `...ConnectedComponents`, `...Modularity`) that
  accept a caller-built CSR as command-graph views, with masks, and one documented self-loop rule
  for degree and modularity volume.
- `GPUScan` and `GPUReduction` on 64-bit (two-word) integers, which would replace the two-level
  prefix of the K-function edge draw by one exact prefix sum.

Correction (2026-11-10, `luma-official` stream): two items above are not luma.gl gaps. `GPUIsobands` is a local class (`modules/analysis/src/raster-analysis/isolines/gpu-isobands.ts`), so the upper-inclusive band option is a local request; and `GPUScanUint64` is already public in `@luma.gl/graph/gpu-core` (alpha.10), so only the 64-bit `GPUReduction` is missing upstream.

Not done in this pass: `modules/lenses`, `modules/kepler`, `website` and `docs/lens-records` were
not touched (roadmap 12.4). The recorded lenses of `GPUNetworkKFunction`, `GPUNetworkLouvain`,
`GPUNetworkCentrality`, `GPUNetworkIsochrones` and `GPUNetworkAnalyticsColumns` carry kernel text
and, for the seeded ones, values that no longer match.

### Review follow-up

After the independent review (`fix-review-fields.md`, rows fr-fields-02 and fr-fields-05). Where
this subsection and the table above disagree, this subsection is current.

**fr-fields-02 (confirmed, fixed).** The network-paths-02 fix made every edge with
`cost + weight == cost` tight, which includes a positive weight that float32 absorbs. On an
undirected network such an edge is tight in both directions, so it set `zeroWeightCycle`, cleared
`converged` and lost the betweenness around it, and no positive weight could cure it.

- `src/gpu-network/network-centrality/network-centrality-passes.ts:101-111`: an edge is tight when
  the float32 sum equals the target cost and either the cost rises or the weight is exactly zero
  (`sourceCost < targetCost || weight == 0.0`). Only exactly-zero weights can now form a cycle, so
  the flag's name and advice are right. Root comment at `:447-448`.
- A node reached only across an absorbed positive weight is a root with one path and the edge is
  left out of the path counts, as before the pass. This is the float32 limit the class documents;
  no status flag reports it.
- Documentation: `gpu-network-centrality.ts` (class conventions, the `zeroWeightCycle` flag),
  `docs/api/gpu-network/operators.md` (centrality conventions). The CPU mirror
  `network-centrality-oracle.ts` has the same rule.
- Spec: `network-centrality/gpu-network-centrality-path-counts.spec.ts` "leaves an absorbed
  positive weight out instead of calling it a zero-weight cycle": undirected path
  `b -1- a -1- s -16777216- u -1- v -1- x`, pivot `s`. Expected by hand under the documented rule:
  `a` 1, every other node 0, `converged` 1, `zeroWeightCycle` 0. (Exact arithmetic would give `u`
  2 and `v` 1.) The zero-weight path, the acyclic plateau and the cyclic case still pass. This
  spec was written after the change; that it fails on the earlier rule rests on the reviewer's
  derivation, not on a run.
- Behaviour change against the first fix: a network with a positive weight that float32 absorbs
  on a shortest path converges again (`converged` 0 and `zeroWeightCycle` 1 after the first fix,
  1 and 0 now), with that edge left out of the counts.

**fr-fields-05 (cause located, seeded path changed and pinned).** The table is not a graph
resource: it is a function-scope `var` inside one invocation of one node, so there is no second
node, no missing dependency and no aliased binding to find. A kernel with one output binding and
nothing else reproduces the failure, which places it in the compiled shader (Tint and the Metal
compiler on the test machine), not in the command graph. Measured with scratch kernels, sizes 12
to 512:

| Table | Read back |
| --- | --- |
| Function-scope `var`, written only by a counted identity loop, read by a loop index inside a loop nest | all zeros at 40, 48, 56, 64, 72, 96 entries; right at 12 to 32 and at 128, 200, 512. Inside the Louvain kernel the wrong range was 12 to 24. |
| The same table, read by a plain copy loop | right at the 8 sizes tried, 8 to 256 (64 included) |
| The same table declared `var<private>` at module scope | right at every size |
| Function-scope `var` with a literal initializer | right at every size |
| Function-scope `var`, identity loop followed by the Philox Fisher-Yates swaps (one seed) | right at every size |

So the trigger is a function-scope array whose only writes are the identity loop. The mechanism
inside the compiler was not found (the generated Metal source was not inspected), and other
kernels in the library were not searched for the same shape.

- `src/gpu-network/network-optimization/gpu-network-louvain.ts:34-68`: new internal
  `getGPUNetworkLouvainVisitOrderSource(vertexCount, seed)`. The seeded table is now a
  module-scope `var<private>` filled by the identity loop and the Philox swaps; the kernel uses it
  at `:192`, `:223`, `:250`. The unseeded path has no table.
- Spec `network-optimization/gpu-network-louvain-visit-order.spec.ts`:
  - "the Louvain visit-order table reads back as the full Philox order from 2 to 512 entries":
    the contributor's own WGSL in a one-binding kernel, read in the loop-nest shape that failed,
    13 sizes (2 to 512) by 5 seeds, the whole order against the TypeScript `PhiloxStream`
    Fisher-Yates.
  - "seeded runs follow the Philox order at every size up to the 32-vertex limit": the class
    itself at 10 sizes (4 to 32) by 3 seeds, one bit per pair and the final objective.
  - "without a seed visits vertices in ascending order at every size": the class at the same 10
    sizes.
- The class accepts at most 32 vertices, so 64 and above are tested on the table, not on a
  Louvain run.
- The seed is still a shader literal. Reading it from a view would need a new input and was not
  done.
