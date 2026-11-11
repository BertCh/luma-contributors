# Math correctness review: applied recipes (2026-11-10)

Scope id `applied-recipes`. Directories under `modules/analysis/src/gpu-spatial-analysis/`:
`location-allocation`, `catchment-accessibility`, `map-coloring`, `dorling-cartogram`,
`neighborhood-summary`, `landscape-ecology`, `urban-morphology`, `recipes`. Paths below are relative
to `modules/analysis/src`. Read-only review. No spec was run (0 of 5 allowed runs used). Scratch
scripts are in the session scratchpad under `applied-recipes/` (`facade.js`, `greedy-sum.js`,
`bodies.js`).

## 1. Coverage

**Read in full (every kernel body and packer; import headers skipped):**

- `location-allocation/gpu-location-allocation.ts`, `gpu-bounded-facility-allocation.ts`,
  `location-allocation-sparse.ts`, `facility-score-reduction.ts`.
- `catchment-accessibility/gpu-catchment-accessibility.ts`, `gpu-huff-trade-areas.ts`.
- `map-coloring/gpu-map-coloring.ts`.
- `dorling-cartogram/gpu-dorling-cartogram.ts`.
- `neighborhood-summary/gpu-neighborhood-summary.ts`.
- `landscape-ecology/`: `sparse-resistance-system.ts`, `sparse-resistance-solver.ts`,
  `sparse-resistance-connectivity.ts`, `raster-resistance-connectivity.ts`,
  `resistance-connectivity.ts`, `landscape-metrics.ts`, `patch-proximity.ts`,
  `reconcile-patches.ts`, `segregation-decomposition.ts`, `segregation-counterfactual.ts`.
- `urban-morphology/`: `urban-ray-utils.ts`, `isovists.ts`, `visibility-graph.ts`,
  `street-profiles.ts`, `building-adjacency.ts`, `building-structures.ts`,
  `classify-viewpoints.ts`.
- Published `node_modules/@luma.gl/graph/dist/gpu-core/gpu-conjugate-gradient-program.js` and its
  `.d.ts` (the official CG), for the recurrence comparison.

**Read in part:**

- `urban-morphology/building-tessellation.ts`: lines 20-418 (sampling, Voronoi clipping, pack).
  Lines 418-527 (overlay output wiring and the `simple` enclosure clip) not read.
- `recipes/hot-spot-analysis-recipe.ts`: lines 440-800, 905-1000, 1026-1078 (Quadbin area, cover
  mask, join, analysis values). Lines 1078-1426 (weights, Gi\*, permutation, colours) not read.
- `recipes/interpolation-diagnostics-recipe.ts`: 40-215. `recipes/quantile-lisa-recipe.ts`: 44-112,
  150-326. `recipes/spatial-regression-recipe.ts`: 60-116, 150-196. `recipes/rate-mapping-recipes.ts`:
  60-206.
- Inline WGSL bodies only (extracted with `bodies.js`, the TypeScript wiring around them not read):
  `dasymetric-redistribution-recipe.ts`, `cluster-and-outline-recipe.ts`, `fleet-dwell-recipe.ts`,
  `rate-cluster-map-recipe.ts`, `raster-summary-recipes.ts`, `surface-preparation-recipe.ts`,
  `cell-flow-topology-recipe.ts`, `contiguity-order-correlogram-recipe.ts`.
- Contributor wiring blocks only: `straight-line-catchments-recipe.ts` 158-224,
  `change-of-support-recipe.ts` 338-453.
- `period-comparison-recipe.ts`: grep only (it has no arithmetic of its own; it forwards to
  `GPUCellTableCompare`).

**Not opened:** `recipes/recipe-utils.ts`, `recipes/expansion-recipe-utils.ts`, every `index.ts`,
`recipes/space-time-hot-spots-recipe.ts` (covered by the 2026-11-09 space-time review),
`recipes/points-in-polygons-choropleth-recipe.ts`, `recipes/local-variability-recipe.ts`,
`recipes/zonal-distribution-recipe.ts`, `recipes/drainage-network-recipe.ts`, and the unread line
ranges listed above. The unit wiring of those recipes is **not verified**.

Docs read: `docs/api/gpu-spatial-analysis/landscape-ecology.md` (all),
`operators-statistics.md` 924-1010, greps of `urban-morphology.md` and `operators-accessibility.md`.
Oracles read: `test/oracles/urban-landscape.R`, `urban-ecology.py`, `urban-ecology-advanced.py`
(all), the heads of the matching fixtures, and the headers of `p-median-oracle.py`,
`geometry-map-coloring.py`, `parity-wp1-huff-trade-areas.py`, `network-cost-weights-2sfca.py`.

No file looked half-edited. `raster-resistance-connectivity.ts` is untracked (new) and six files in
scope carry uncommitted edits; I reviewed the working tree as it stands.

## 2. Findings

| ID | Severity | Status | File:line | What the kernel computes | What the reference says | Concrete wrong output | Suggested fix |
| --- | --- | --- | --- | --- | --- | --- | --- |
| applied-recipes-01 | wrong-result | CONFIRMED (script) | `gpu-spatial-analysis/urban-morphology/street-profiles.ts:413` | A facade interval counts as visible only when `abs(hit.x - abs(lateral)) <= tolerance` (default 1e-6, absolute) and the hit row is the segment itself. `hit.x` comes from two cross products and a division, `lateral` from a dot product, both in float32. | The docs say the facade length is exact for the corridor "up to the shared geometric tolerance". An unobstructed facade should count in full. | Street `(10.3, 20.7)-(110.3, 45.9)`, one facade `(30.1, 40.2)-(60.4, 47.9)`, nothing in between: `lateral` 14.101734, `hit.x` 14.101732, difference 1.9e-6, so the interval is rejected and the facade length is 0 instead of 31.26. Over 2,000 random streets the facade was rejected in 59% of cases with coordinates near 1 to 50, 79% near 100, 98% near 1,000 and all but one near 500,000 (projected metres). | Decide visibility from the hit row alone (the nearest hit is this segment), or compare with a relative slack such as `tolerance + 1e-5 * max(abs(lateral), largest coordinate)`. Add a spec with non-integer coordinates. |
| applied-recipes-02 | doc-mismatch | CONFIRMED (derivation) | `gpu-spatial-analysis/landscape-ecology/raster-resistance-connectivity.ts:426` (claim at `:228-233`) | Edge conductance `2 / (R(a) + R(b)) / edgeLength`, where `edgeLength` is the ground distance between cell centres. | The class doc calls this "Circuitscape's raster pairwise mode". Circuitscape uses `1 / mean(R)` for cardinal neighbours and `1 / (sqrt(2) * mean(R))` for diagonals. It never divides by the cell size. | Two cells with R = 1 and `cellSize` 30: Circuitscape gives effective resistance 1; this recipe gives 30. Potentials are also 30 times larger. Currents agree. In `'geographic'` and `'web-mercator'` mode the edge length changes with the row, so the current map also differs from Circuitscape. | Say in the doc that the result equals Circuitscape only for unit cells in `'uniform'` mode, and that resistance is "per unit length" here. Or add a `lengthWeighting: 'cell' \| 'ground'` prop with Circuitscape's convention as one choice. Add a Circuitscape raster golden. |
| applied-recipes-03 | precision | CONFIRMED (derivation and script) | `gpu-spatial-analysis/landscape-ecology/landscape-metrics.ts:194` (also `:187`, `:239`) | `likeSides` and `cells` are float32 counters incremented by 1.0 per side and per cell. They stop growing at 2^24 = 16,777,216. | Aggregation index is `g_ii / max_g_ii` with exact counts (FRAGSTATS AI). | One 2,100 x 2,100 patch: exact like sides 17,631,600, kernel 16,777,216, aggregation 95.15 instead of 100. A patch of more than 2^24 cells also under-reports its area. | Count in `u32` and convert once. The class aggregation can read the diagonal of `classAdjacency`, which is already `u32` and exact. |
| applied-recipes-04 | wrong-edge-case | CONFIRMED (derivation) | `gpu-spatial-analysis/location-allocation/gpu-location-allocation.ts:1264-1265` (sparse `:1246-1247`, stop rule `:1346`) | The set-covering greedy step scores a facility by the demand **weight** it newly covers and stops when the best score is not above 0. | LSCP must cover every demand point, whatever its weight (spopt `LSCP` has no weights). The class doc says the step runs "until every reachable demand row is covered". | Rows A (weight 0) and B (weight 1). Facility 0 covers B, facility 1 covers A, `maximumFacilityCount` 2. Step 1 picks facility 0. Step 2 scores facility 1 as 0 and stops. Result: one facility, `uncoveredDemandCount` 1, `complete` 0, although two facilities cover everything. | In `set-covering` mode score by row count (weight 1), or break score ties by rows covered. Otherwise document that zero-weight rows are ignored. |
| applied-recipes-05 | inconsistent | CONFIRMED (reading) | `gpu-spatial-analysis/catchment-accessibility/gpu-huff-trade-areas.ts:336` against `gpu-catchment-accessibility.ts:269` | Huff `expectedDemand` adds `demand * w / total` with no finiteness test. The sibling catchment kernel skips a non-finite demand ("missing and captures nothing"). | Contract section 7: one no-data rule. | `demand = [NaN, 10]`, one facility reaching both: catchment ratio uses 10; Huff `expectedDemand` is NaN for every facility that reaches row 0. | Use the same `isFiniteBits` guard in the Huff expected-demand loop. |
| applied-recipes-06 | doc-mismatch | CONFIRMED (derivation and oracle order) | `gpu-spatial-analysis/landscape-ecology/segregation-counterfactual.ts:35-38` | `compositionOnly` is built from the baseline units (their ranks and totals) with the final table's composition distribution. `spatialOnly` is the reverse. This is what the Shapley step needs and what PySAL's `counterfac_df1` and `counterfac_df2` are. | The type comments say the opposite: "`compositionOnly`: baseline composition on the final spatial structure" and "`spatialOnly`: final composition on the baseline spatial structure". | A reader who trusts the comment swaps the two tables. The decomposition itself is right (it matches `segregation` 2.5.3 in the fixture). | Swap the two comments. |

### Notes on the findings

**applied-recipes-01.** `facade.js` replicates `getSegmentCandidate` (`urban-ray-utils.ts:132-143`)
and the interval test (`street-profiles.ts:406-416`) with `Math.fround`. The existing fixtures use
integer coordinates (`[-4, 0, 4, 0]`, boxes at integer corners), where every product is exact, so the
specs pass. The test on line 410 (`abs(lateral) <= tolerance`) is harmless. The same absolute
tolerance is fine in the other urban kernels, because there it only separates "on the boundary" from
"off the boundary" (`classify-viewpoints.ts:126`, `isovists.ts:255`, `visibility-graph.ts:198`).
The BVH padding already scales with the coordinates (`urban-ray-utils.ts:82`); the facade test does
not.

**applied-recipes-02.** The formula is stated in the doc comment and in the provenance string, so
the code does what it says. The defect is the Circuitscape claim, and the fact that nothing
independent pins the convention: the spec oracle
(`test/gpu-spatial-analysis/urban-ecology/raster-resistance-connectivity.spec.ts:51-90`) rebuilds the
same conductances on the CPU and solves in float64. The division by length is the cost-distance
convention (resistance per unit length). It is defensible, but it is not Circuitscape, and for a
2D sheet the physical conductance of a square cell link does not depend on the cell size.

**applied-recipes-04.** The run reports the gap honestly (`uncoveredDemandCount`, `complete`), so
nothing is hidden. The stop is still premature for a plausible input (zero-population demand
points).

## 3. Verified correct

Each line names what the kernel was compared against.

- **Sparse grounded-Laplacian PCG** (`sparse-resistance-system.ts`, `sparse-resistance-solver.ts`).
  Recurrence checked by hand against the textbook Jacobi-PCG and against the published
  `createGPUConjugateGradientGraph`: `x0 = 0`, `r0 = e_source`, `z = D^-1 r`, `p0 = z0`,
  `alpha = r.z / p.Ap`, `x += alpha p`, `r -= alpha Ap`, `beta = r'.z' / r.z`, `p = z' + beta p`.
  The sink row and column leave the system (`product[sink] = 0`, and `search[sink]` provably stays
  0). The diagonal keeps the edges to the sink, as a grounded Laplacian must. Breakdown on
  `p.Ap <= 0`. Stopping rule: recursive `||r||_2 <= tolerance / 2` and `|sum r| <= tolerance / 2`
  (the second is the sink's own residual, by the zero row-sum identity), then `converged` is decided
  by an independent true residual `max |Lx - b| <= tolerance` over every vertex. At the iteration
  limit the potentials are published raw, `converged` is 0 and the resistance is NaN.
- **Effective resistance, energy and currents** (`sparse-resistance-connectivity.ts`).
  `R_eff = v[source]` for unit injection with the sink at 0. Energy sums `g * dv^2` once per edge
  (0.5 per half-edge) and equals `R_eff`. Node current `max(inflow, outflow)` matches Circuitscape's
  rule (1 at the source and at the sink). The `reachable == 2` certificate (`max residual * n < 1`)
  is sound: an unreachable sink has residual exactly 1.
- **Dense CG** (`resistance-connectivity.ts`, deprecated). Plain CG, same grounding, stop on
  `||r||^2 <= tolerance^2`, true-residual certificate. Identical recurrence to the published CG.
- **Raster edge builder** (`raster-resistance-connectivity.ts:407-430`). Edge slots, off-grid self
  edges, no-data handling, direction table (E, S, SE, SW) and the midpoint-row rule are right. Only
  the length convention is in question (finding 02).
- **Landscape metrics** (`landscape-metrics.ts`). Patch area and perimeter (side lengths the right
  way round for rectangular cells), class mesh `sum a^2 / A`, cohesion
  `100 (1 - sum p / sum p sqrt(a)) / (1 - 1/sqrt(Z))`, aggregation maximum
  `2n(n-1)`, `+2m-1`, `+2m-2` (He et al. 2000, FRAGSTATS AI) with double-counted like sides halved,
  contagion in both modes (FRAGSTATS `P_i g_ik / sum_k g_ik`; landscapemetrics joint adjacency),
  landscape AI as the area-weighted class AI with undefined classes skipped (`lsm_l_ai`). Recomputed
  by hand for the `interior-background` fixture: AI 80, mesh 25/11, cohesion 77.35 for the two-square
  case. Units: area in squared cell units, no hectare conversion, as documented.
- **Patch proximity** (`patch-proximity.ts`). The two-pass transform is an exact Euclidean distance
  (minimum over rows of `hypot(row-wise minimum, row gap)`). Core area in both models (physical
  depth with the right search window; L1 diamond for erosion layers). Proximity
  `sum area / distance^2` over same-class patches within the radius. Zero distance and infinite
  proximity for diagonal contact under `'cell-boundary'` is documented.
- **Patch reconciliation** (`reconcile-patches.ts`). Union-find with lowest-ID roots; the four
  forward directions cover all eight neighbours.
- **Segregation decomposition and counterfactual.** `D = 0.5 sum |x_i/X - y_i/Y|`; two-factor
  Shapley terms sum to the change; average-tie percentile rank `(less + (equal + 1)/2) / n` and
  linear quantile at `rank * (n - 1)` are pandas `rank(pct=True)` and `quantile`; the three methods
  match PySAL `_generate_counterfactual`; pseudo p-value `(extreme + 1) / (valid + 1)`, two-sided.
- **GPULocationAllocation.** Greedy p-median (lexicographic unreachable weight, then cost), greedy
  maximum coverage (inclusive `cost <= distance`), and the Whitaker / Resende-Werneck swap scoring.
  I re-derived every case of `gain`, `loss` and `extra` for the dense and the sparse kernels,
  including the unreachable-second-nearest cases, and for the coverage mode. The dense and sparse
  greedy scores order candidates identically. Dense greedy compares full float32 totals; a script
  (`greedy-sum.js`) shows only ties at a relative gap of 4e-8, never a reversed order, so this is not
  a finding.
- **GPUBoundedFacilityAllocation.** Lexicographic subset unranking by binomials; p-center maximum
  over rows; lexicographic reduction (unassigned, objective bits, rank) is valid because the
  objective is non-negative.
- **GPUCatchmentAccessibility.** 2SFCA (Luo and Wang 2003) and its weighted (enhanced, Gaussian)
  form: `R_j = S_j / sum_i P_i w_ji`, `A_i = sum_j w_ij R_j`. 3SFCA (Wan, Zou and Sternberg 2012):
  `G_ij = w_ij / sum_k w_ik`, applied in both steps. The decay normalisation belongs to the weights
  producer and is not in this kernel.
- **GPUHuffTradeAreas.** `p_ij = A_j^alpha w_ij / sum_k A_k^alpha w_ik`; expected demand; the
  "lowest ID wins ties" rule holds because weights rows are ascending by neighbour ID.
- **GPUMapColoring.** Jones-Plassmann with a strict total order (degree, hash, row ID). The fused
  one-pass round is race free: a row only ever reads final colours of rows that outrank it. Result
  equals sequential greedy in priority order (Welsh-Powell order when degree-first). Window scan
  bound is sufficient (at most `degree` colours in use around a row).
- **GPUDorlingCartogram.** Radius `sqrt(value * areaScale / pi)`; antisymmetric corrections; the
  iteration counter, convergence flag and budget flag are consistent; the grid reach
  `radius + largest radius` plus one cell of margin cannot miss an overlapping pair. The relaxation
  is the library's own (documented: no Dorling 1996 or GeoDa parity).
- **GPUNeighborhoodSummary.** Weighted mean, two-pass weighted population deviation, unweighted
  median with the even-count rule, entropy `-sum p ln p`, mode with lowest-category ties,
  re-standardisation of row-standardised weights over active members.
- **Urban ray casting** (`urban-ray-utils.ts`). Ray-segment parameters, collinear overlap, tie rule
  (lowest row) in both the brute-force and the BVH path; winding number (`classify-viewpoints.ts`)
  is the standard crossing rule.
- **Isovists.** Shoelace area about the viewpoint, polygon perimeter, counterclockwise order.
- **Visibility graph.** Brandes accumulation and the 0.5 factor for undirected choice;
  `2 C / ((n-1)(n-2))`; mean depth, relative asymmetry, Hillier-Hanson diamond value, P-value and
  Teklenburg formulas match depthmapX `pafmath.h` (recomputed by hand for the 5-node path fixture:
  0.3520, 0.2797, 0.2263). Infinite HH integration for a complete component is documented.
- **Street profiles** (ray part). Left and right distance, width, mean height, height to width.
- **Building adjacency.** Axial angle difference with period pi; cardinal deviation folded to
  0-45 degrees and its mean neighbour difference (momepy `orientation` and `alignment`; isolates
  give 0, as momepy 0.10 does); exterior-only structure perimeter.
- **Building tessellation** (lines read). Boundary sampling (start included, end excluded),
  half-plane Voronoi clipping against the perpendicular bisector, convexity and orientation checks.
- **Recipes (parts read).** Quadbin tile area on the authalic sphere
  (`R^2 dLon sinh(step) / (cosh(north) cosh(south))`, shared `GPU_EARTH_AUTHALIC_RADIUS_METERS`);
  interpolation diagnostics (residual observed minus predicted, standardised by `sqrt(variance)`,
  scaled RMSE); the Anselin LM decision tree (CPU and WGSL twins agree); quantile LISA type-5 rank
  mapped to a type-7 probability, permutation parameter words 2 and 3 read correctly, GeoDa folded
  pseudo p-value `(min(g, R - g) + 1) / (R + 1)`; cell-flow key word order is consistent between the
  neighbour write and the destination compare; exposure and regional change thresholds.

## 4. Standards consistency

- **No-data rule.** Catchment skips non-finite demand; Huff does not (finding 05).
- **Angle units inside one recipe.** `building-adjacency.ts` publishes the pair angle in radians
  (`pairs.y`, `:158`) and the cardinal orientation and alignment in degrees (`:245`, constant
  57.29577951308232). Both are documented, and the degrees match momepy, but one output set carries
  two units.
- **Missing-value sentinel for "no same-class neighbour"** in `patch-proximity.ts:236`: +Infinity
  under `'cell-boundary'`, NaN under `'cell-center'`. Documented.
- **NaN idiom.** `bitcast<f32>(0x7fc00000u | (index & 0u))` is still written inline 35 times in
  scope (landscape 12, urban 12, recipes 11; the Infinity twin `0x7f800000u` is extra) where
  `GET_NAN_WGSL` exists. Numerically identical.
- **Private copies of shared numerics.**
  - Binary search: `location-allocation-sparse.ts:72-89` (`getSparseRowCost`) is a private
    lower-bound search; `analysis-core/kernels/binary-search-wgsl.ts` exists. Same result.
  - Orientation: urban kernels use a plain float32 cross product (`crossVectors`) for ray tests and
    the winding number, not the exact orientation predicate of
    `analysis-core/kernels/exact-orientation-wgsl.ts`. Points within `tolerance` of a segment are
    classified as boundary first, so the inexact sign only matters inside that band.
  - Weak components: `building-adjacency.ts:206-225` and `reconcile-patches.ts:53-56` are serial
    union-finds; `analysis-core/kernels/masked-weak-components.ts` (used by the sparse solver) gives
    the same lowest-ID labels.
  - Dissimilarity index: `segregation-decomposition.ts` recomputes D; `GPUSegregation` has it. Same
    formula.
- **Constants as shader literals.** `6.283185307179586` (`isovists.ts:166`, `:173`, `:188`),
  `3.141592653589793` (`building-adjacency.ts:158`), `PI: f32 = 3.14159265358979`
  (`hot-spot-analysis-recipe.ts:451`, one digit shorter, same float32). I found no shared WGSL `PI`
  constant in `analysis-core/kernels/wgsl-common.ts`, so there is nothing to delegate to yet.
- **Random numbers.** Map colouring uses the shared `hashUint32` for priorities (not a statistical
  draw). The segregation randomisation uses Philox with a registered purpose and a counter of
  `(unit, replicate)`; one bit per draw, no modulo bias.
- **Earth radius.** The only radius in scope is the authalic radius from `contracts/earth.ts`.
- **Tolerances.** All urban kernels share one absolute `tolerance` (default 1e-6). It is used both
  as a boundary band and, in the facade kernel, as an equality slack for computed distances. The
  second use needs a relative term (finding 01).

## 5. luma.gl official implementations

- **Conjugate gradient.** Published `createGPUConjugateGradientGraph` (`@luma.gl/graph/gpu-core`)
  is unpreconditioned CG on a CSR matrix, with `r0 = b - A x0`, breakdown on `p.Ap <= 0` and a loop
  predicate `r.r > toleranceSquared`. The local solver is the same recurrence with a Jacobi
  preconditioner (`M = D`); with `M = I` the two are identical step for step. Differences of
  meaning: the official tolerance is the squared 2-norm of the recursive residual; the local
  `tolerance` is the maximum norm of an independently recomputed residual. The local one is the
  stricter certificate. Delegation is not possible today (the published program compiles to its own
  graph and binds `GPUVector`s); this is recorded in `dev-docs/upstream-workarounds.md:55`. Keep the
  upstream request for `GPUJacobiPCG` as command nodes.
- **CSR assembly** uses published `GPUSort` and `GPUCOOToCSR` with unchanged meaning (stable sort,
  keys in `[0, vertexCount]`, invalid half-edges sorted past the last row).
- **Connected components.** `@luma.gl/graph/gpu-raster` publishes `GPURasterConnectedComponents`
  and `GPURasterCrossTileComponents`; `@luma.gl/graph/gpu-graph-algorithms` publishes
  `GPUGraphAlgorithmsConnectedComponents` (lowest stable vertex ID per component). The local serial
  union-finds in `reconcile-patches.ts` and `building-adjacency.ts` give the same labels (lowest
  row). They agree mathematically; they should delegate (reconcile needs class-aware merging, which
  the cross-tile class may or may not cover: check before delegating).
- **Reductions, scan, compaction, grid index, BVH** are used through the public classes
  (`GPUReduction`, `GPUScan`, `GPUCompaction`, `GPUGridIndex`, `GPUBVH` views) with no change of
  meaning. The Dorling grid normalises centres to `[0, 1]` before `GPUGridIndex` and looks cells up
  with a local `getBinIndex`; one cell of margin covers any rounding difference between the two.
- **Raster Laplacian.** Published `GPURasterLaplacian` is an image stencil (centre -4 or -8), not a
  weighted graph Laplacian. It does not replace the resistance edge builder.
- No private path (`_internal`, deep `dist`) is imported in scope.

## 6. Test oracles

**Pinned by an independent oracle:**

- Landscape metrics: landscapemetrics 2.2.1 through `urban-landscape.R` (area, perimeter, mesh,
  cohesion, AI, contagion in `'adjacency'` mode, landscape AI and cohesion; three small rasters).
  The default `'area'` contagion (FRAGSTATS form) coincides with the joint form on the equal-area
  fixture and has a closed-form check in `urban-ecology.py`; it has no fixture where the two modes
  differ.
- Patch proximity: landscapemetrics ENN and core for `distanceModel: 'cell-center'` and
  `coreModel: 'raster-erosion'`. Proximity is "composed" from upstream area and ENN on two patches.
  The **defaults** (`'cell-boundary'`, `'physical'`) are checked by hand values only.
- Sparse and dense resistance: Circuitscape.jl golden outputs for a 7-vertex network (six pairs),
  SciPy sparse solves, a float64 lattice oracle.
- Segregation decomposition and counterfactual: `segregation` 2.5.3 called directly.
- Visibility-graph normalisations: formulas from depthmapX `pafmath.h` (the binary was not run).
  Choice and depth: NetworkX on a 3-node path.
- Isovists: Shapely on one box. Tessellation: momepy 0.10.0 areas. Street profile width and ratio,
  shared walls, orientation, alignment, perimeter wall: momepy 0.10.0.
- Location allocation: brute force and `scipy.optimize.milp` optimum (this bounds the heuristic; it
  does not pin each swap). A `legacy-pair-swap.ts` helper exists in the test folder; I did not read
  how it is used.
- 2SFCA and 3SFCA: PySAL `access` (`two_stage_fca`, `three_stage_fca`) through the network path.
- Huff: NumPy and SciPy (`softmax`), independent of the kernel.
- Map colouring: NetworkX `greedy_color` colour counts (not colours).
- Neighbourhood summary: NumPy and libpysal parity fixtures.

**Pinned only by a copy of their own formula:**

- Raster resistance conductances (`raster-resistance-connectivity.spec.ts` rebuilds
  `2 / (R(a) + R(b)) / edgeLength` on the CPU). See finding 02.
- `GPUCatchmentAccessibility` in its own spec (a TypeScript `catchment-oracle.ts`, which I did not
  open; presumably the same formula); the PySAL check reaches the kernel through the network
  cost-weights path.
- Facade lengths (`streetSegment`): one integer-coordinate fixture, which cannot expose finding 01.

**Pinned by nothing independent:**

- Dorling relaxation (self-consistency: overlap, displacement, grid against all-pairs).
- "Penn" normalisation beyond the 5-node path.
- FRAGSTATS itself was never run; every landscape reference is landscapemetrics or a closed form.
- The recipes whose wiring I did not read (section 1).

## Resolution (2026-11-10)

Paths are relative to `modules/analysis`. Specs are under `test/gpu-spatial-analysis`.

| ID | Outcome | What changed | Spec that pins it | Behaviour change a caller sees |
| --- | --- | --- | --- | --- |
| applied-recipes-01 | fixed | `src/gpu-spatial-analysis/urban-morphology/street-profiles.ts:43-87` (new `isFacadeIntervalVisible`), `:439` (edge-on rule), `:486` (decision). `urban-ray-utils.ts:136-148` (`'local'` ray frame, `distanceToSegment` on the offset). The facade kernel works relative to the street start. A stretch is hidden when another segment blocks its sightline before it, or within `tolerance + 1e-5 * max(street length, maximumDistance)` of it with a lower row. No two computed distances are compared for equality. | `urban-ecology/street-profile-facades.spec.ts` (8 cases, each with brute force and BVH). 5 of the 8 failed on the old kernel on this GPU. | Facade lengths that read 0 at non-integer or projected-metre coordinates now hold the visible length. A wall within `tolerance + 1e-5 * length` of perpendicular to the street reports 0 (before: 0 only when exactly perpendicular in float32, else noise). Coincident segments: the lower row owns the stretch (before: an exact float tie, else neither). |
| applied-recipes-02 | fixed (documentation, roadmap 12.3) | `src/gpu-spatial-analysis/landscape-ecology/raster-resistance-connectivity.ts:233-248` and `docs/api/gpu-spatial-analysis/landscape-ecology.md` (EC03-R): the parity claim is replaced by the exact relation to Circuitscape. Formula unchanged. | `urban-ecology/raster-resistance-connectivity.spec.ts:571-664`: two cells (30), three cells in series (12; potentials 12, 8, 0) and a 2 x 2 grid with diagonals (2 - sqrt(2)), all solved by hand. | None. |
| applied-recipes-03 | fixed | `src/gpu-spatial-analysis/landscape-ecology/landscape-metrics.ts:190-205` (cells, perimeter sides and like sides counted in `u32`; a transient `uint32x2` carries exact counts to the class sums), `:246-263` (class sums in `u32`, exact integer square root). | `urban-ecology/landscape-metrics-counts.spec.ts` (2,100 x 2,100 patch: 17,631,600 like sides, aggregation 100; three hand-computed classes with non-square counts). Written after the fix, so not observed failing on the GPU; the old value 95.15 follows from `2^24 + 1 = 2^24` in float32. | Aggregation index, cell count and like-side count are right past 16,777,216. Patch perimeter is `sides * cellSize` instead of a running float sum: identical for the existing fixtures, last-bit differences possible for cell sizes that are not exact in binary. |
| applied-recipes-04 | fixed | `src/gpu-spatial-analysis/location-allocation/gpu-location-allocation.ts:1231`, `:1252`, `:1271`, `:1276`: set covering scores a facility by the `u32` count of rows it newly covers, dense and sparse. Maximum coverage still scores by weight. | `location-allocation/gpu-location-allocation.spec.ts:214` and `:237` (dense and sparse; both failed before the fix). | Set covering with `demandWeights`: a row of weight 0 is now covered, and the facility order follows row counts, not weight. Without weights nothing changes. |
| applied-recipes-05 | fixed (converged on `GPUCatchmentAccessibility`: skip) | `src/gpu-spatial-analysis/catchment-accessibility/gpu-huff-trade-areas.ts:358` (non-finite demand adds nothing), `:82` and the `validity` node (new optional `output.validity`, `uint32` per demand row, 0 for a non-finite demand). | `catchment-accessibility/gpu-catchment-accessibility.spec.ts:194` (failed before: `[NaN, NaN]`). | `expectedDemand` is finite when a demand is NaN or infinite (before: NaN for every facility reaching that row). New optional output `validity`. |
| applied-recipes-06 | fixed | `src/gpu-spatial-analysis/landscape-ecology/segregation-counterfactual.ts:35-46`: the two comments are swapped and name the PySAL tables. | None (comment only). | None. |

### Recipe wiring check ("Not covered" in section 1)

Read for units and conventions between stages (degrees and radians, mask and validity, counts of
pairs and rows, status words read as data).

- **Fixed.** `recipes/hot-spot-analysis-recipe.ts:570`: `selfWeight` went to `GPUHotSpotAnalysis`
  only, while the permutation confirmation always ran `'localGStar'`, whose focal weight is fixed
  at 1. With `selfWeight: 0` the map was Gi and the confirmation Gi*. The permutation statistic now
  defaults to `'localG'` when `selfWeight` is 0, and a combination the permutation test cannot
  reproduce throws at construction. Spec: `recipes/recipe-status-stages.node.spec.ts:602`.
- **Documented.** `recipes/spatial-regression-recipe.ts:366`: the status stages publish the solver
  `code` word under `invalidCount`. It is a failure code, not a row count. The words say so now; the
  port name is unchanged.
- **Checked and found right** (not opened by the review): `recipe-utils.ts` (key word order of the
  cell search, count and write passes of the disk weights use one predicate),
  `expansion-recipe-utils.ts`, `space-time-hot-spots-recipe.ts` (period sentinels, masked events
  reach `GPUPeriodIndex` as invalid years, cube layout `cell * sliceCount + slice`),
  `points-in-polygons-choropleth-recipe.ts` (class count against `k + 1` edges, both back-ends give
  the same empty-feature values), `local-variability-recipe.ts`, `zonal-distribution-recipe.ts`
  (fractions in [0, 1], the no-feature key is the reserved group key),
  `drainage-network-recipe.ts` (D8 codes, the unresolved stream and terminal sentinels, the
  no-catchment label).
- **Checked and found right** (WGSL read by the review, wiring read now):
  `dasymetric-redistribution-recipe.ts` (density times cell area, extensive shares, the mass
  ledger closes for both denominators), `cluster-and-outline-recipe.ts` (dense labels, edge slot
  `k` is vertices `k` and `k + 1`), `fleet-dwell-recipe.ts` (dwell table is track-major),
  `rate-cluster-map-recipe.ts` (NaN rates are excluded by both inference stages),
  `raster-summary-recipes.ts` (invalid cells are NaN and skipped by the zonal sums, zone rows are
  dense by zone ID), `surface-preparation-recipe.ts`, `cell-flow-topology-recipe.ts` (edge slot
  `d - 1` is direction `d`, boundaries in degrees), `contiguity-order-correlogram-recipe.ts`
  (summary field order).
- **Unread ranges of files the review read in part, read now and found right:**
  `hot-spot-analysis-recipe.ts` 600-905 and 1000-1426 (areas in square kilometres for both
  families, the distance band needs a great-circle longitude/latitude context),
  `spatial-regression-recipe.ts` 196-607, `urban-morphology/building-tessellation.ts` 400-527
  (`status.count` is the feature count the identity kernel assumes),
  `period-comparison-recipe.ts` (contributor calls). `quantile-lisa-recipe.ts` 112-150 and
  `rate-mapping-recipes.ts` 1-60 are declarations only.

### Notes

- applied-recipes-01: the review's own input (street `(10.3, 20.7)-(110.3, 45.9)`) gave the right
  length on this GPU before the fix; the float32 replication does not match the GPU's rounding
  there. The rotated, translated and joined cases did fail.
- float32 inputs resolve 0.5 near 4,500,000, so a fixture rotated by a generic angle and moved to
  `(500000, 4500000)` is a different geometry after rounding. The spec has two cases: one with a
  scale of 12.5 and the 3-4-5 rotation, where every coordinate is exact and the lengths must equal
  12.5 times the fixture's, and one with a rotation of 0.3 radians, where the expected length is a
  closed form evaluated on the rounded input.
- `GPUCatchmentAccessibility` still has no validity output for a non-finite demand; only Huff
  gained one.

### Upstream gaps

None. No finding in this scope has an official luma.gl class that computes the same thing.

### Review follow-up

From `fix-review-models.md`. Paths relative to `modules/analysis`.

| Review row | Outcome | What changed | Spec |
| --- | --- | --- | --- |
| fr-models-01 | fixed | `src/gpu-spatial-analysis/urban-morphology/street-profiles.ts:66-69` (decision), `:72` and `:82` (both forms). Ownership is decided against the nearest hit on the sightline, the wall's own included: the wall owns the stretch when it is within the slack of the nearest hit and no lower row is. Any number of coincident walls has exactly one owner. | `test/gpu-spatial-analysis/urban-ecology/street-profile-facades.spec.ts:280` (three walls, reverse and forward row order, brute force and BVH). Written after the kernel change, so the cycle was not observed on the GPU; the reviewer's float32 replica shows it. |
| fr-models-08 | not reproduced; comment corrected | `street-profiles.ts:45-59`. The BVH boxes are padded by `2 * tolerance` plus the float32 rounding of the coordinates (`addUrbanObstacleBVH`), so a segment within `tolerance` of the sightline origin has the origin inside its box and is tested, behind the origin or not. The reviewer's example leaves the padding out. The comment now gives this argument instead of the bare claim. | `street-profile-facades.spec.ts:300`: a wall 5e-7 behind the street line hides the facade in both forms; at 5e-5 it does not. |
| fr-models-06 | fixed | `src/gpu-spatial-analysis/recipes/hot-spot-analysis-recipe.ts:307`, `:1325`, `:1353`: new optional `outputs.permutation.observed`, passed to `GPULocalPermutationTest`. | `test/gpu-spatial-analysis/recipes/hot-spot-permutation-statistic.spec.ts`: on a 5 x 5 queen lattice, for `selfWeight: 0` and the default, the observed statistic and the z-score each match the Ord and Getis closed form in float64, and the z-score rebuilt from the observed statistic matches the mapped one. |
| fr-models-07 (`isovists.ts`) | fixed | `src/gpu-spatial-analysis/urban-morphology/isovists.ts:172`: `guardedAtan2`. A viewpoint that coincides with a segment endpoint now gets angle 0 for that endpoint, where the builtin returned NaN on the test GPU. | The existing isovist specs (unchanged results). |
| Catchment validity | fixed | `src/gpu-spatial-analysis/catchment-accessibility/gpu-catchment-accessibility.ts:50` and the `validity` node (`:371`): optional `output.validity`, `uint32` per demand row, 0 for a non-finite demand, the same as `GPUHuffTradeAreas`. This supersedes the last note above. | `test/gpu-spatial-analysis/catchment-accessibility/gpu-catchment-accessibility.spec.ts:231` (failed before: the output did not exist). |

Behaviour change from fr-models-01: with three or more walls inside the slack band the owner can
differ from the first version of the fix, and a wall farther than the slack behind the nearest
hit is hidden whatever its row. Two coincident walls behave as before.
