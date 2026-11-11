# Landscape ecology recipes

All recipes append GPU kernels to a caller-owned command graph without hidden CPU computation or
readback. Inputs remain caller-owned. Outputs are graph-owned transient views unless the caller passes
an `output` prop (see below); export transient results through a graph copy when CPU readback is
needed. The current bounded implementations prioritize explicit
numerical definitions, provenance and failure evidence over large-domain performance claims.

## Caller-owned outputs and limits

Each recipe takes an optional `output` prop. Every entry is a view the caller created in the same
graph, with the format and length of the recipe's transient view of that name; an absent entry is a
graph-owned transient view, as before. The `outputs` of the result are the views actually used.

| Recipe | `output` entries |
| --- | --- |
| `addLandscapeMetricsRecipe` | `patchMetrics`, `classMetrics`, `classAdjacency`, `aggregation`, `landscape`, `landscapeAggregation`, `landscapeCohesion`, and with `reconcilePatches` `reconciledPatches`, `patchParents` |
| `addPatchProximityRecipe` | `core`, `distances`, `metrics`, `horizontal`, `fields`, and with `reconcilePatches` `reconciledPatches`, `patchParents` |
| `addResistanceConnectivityRecipe` | `potentials`, `currents`, `diagnostics`, `status` |
| `addSparseResistanceConnectivityRecipe` | `potentials`, `currents`, `vertexCurrents`, `diagnostics`, `status` |
| `addRasterResistanceConnectivityRecipe` | required `potentials`, `vertexCurrents`, `diagnostics`, `solverStatus`; optional `currents`, `edges`, `conductances` and the `status` scalars |
| `addSegregationDecompositionRecipe` | `indices`, `validity`, `decomposition`, and with `randomization` `distribution`, `inference` |
| `addSegregationCounterfactualRecipe` | the decomposition entries, `compositionOnly`, `spatialOnly` |
| `addPopulationSupportAlignmentRecipe` | `populations`, `validity`, `status` |

Limits are named constants and are rejected with a message that starts with the recipe ID and names
the prop: 256 classes (landscape metrics), 256 vertices, 65,536 edges and 4,096 iterations (dense
resistance), 1,048,576 vertices and 2,097,152 edges (sparse resistance), 1,048,576 cells and
2,097,152 edge slots (raster resistance), 65,536 replicates (decomposition) and 2,048 units
(counterfactual). Patch proximity
checks its patches-times-cells distance fields against the device storage binding size.

The landscape and segregation totals are reductions (`GPUReduction`) over per-patch or per-unit
terms, so the sums are tree sums rather than a serial loop and can differ from the former results in
the last float32 digits. Still serial by design: patch reconciliation (one thread unions the raster),
the counterfactual quantile mapping (bounded at 2,048 units) and the class and patch summary
kernels, which scan every cell or patch once per row.

## `addLandscapeMetricsRecipe` — EC01

Supply the complete raster `width`, `height`, physical `cellSize`, one patch ID per cell, and one
class ID per patch. Patch IDs outside `patchClasses.length` are background. Class IDs outside
`classCount` are excluded. The caller labels connected patches using the declared `connectivity`
(4 or 8); the recipe retains that provenance and counts adjacency through four orthogonal sides,
including when patches were labelled using eight-connectivity. To reconcile tile seams on the
complete resident raster, set `reconcilePatches: true`: GPU union/find merges adjacent same-class
patch IDs under the requested connectivity and relabels each cell with its smallest canonical ID.
Tile IDs must be globally unique before this operation; tile-local ID collisions cannot be inferred.
The returned provenance retains both the input and reconciled labels and the parent mapping.
Merged-away patch rows have zero area. Halos must be removed before assembling the complete grid;
there is no implicit out-of-core fetch or independently normalized per-tile summary.

- `patchMetrics`: `[area, perimeter, cellCount, sameClassDirectedSideCount]`.
- `classMetrics`: `[area, sumPatchPerimeters, effectiveMeshSize, cohesionPercent]`.
- `classAdjacency`: directed same/different-class side counts in a dense class-by-class matrix.
- `aggregation`: 100 × observed same-class joins / maximum joins for that class's cell count.
- `landscape`: `[contagionPercent, effectiveMeshSize, validLandscapeArea, presentClassCount]`.
- `landscapeAggregation`: area-weighted class aggregation percentage. Undefined single-cell class
  indices contribute zero, matching `lsm_l_ai` in landscapemetrics 2.2.1; an empty landscape is NaN.
- `landscapeCohesion`: cohesion percentage computed from all valid patches together, using their
  pooled perimeter and perimeter-weighted square-root cell count. It is not an average of class
  cohesion percentages; empty/single-cell landscapes and zero-perimeter policies yield NaN.

Perimeters include domain boundaries and background contacts by default. `countBoundary: false`
excludes outer-domain contacts, and `countBackground: false` independently excludes contacts with
invalid/excluded classes. `backgroundClass` can designate one otherwise-valid class as excluded;
it does not enter class area or adjacency. These choices change perimeter-derived metrics and must
be recorded with results. Background does not enter valid
landscape area or the contagion adjacency distribution. Mesh size is the sum of squared patch areas
divided by total valid landscape area; landscape mesh sums the class values. Cohesion uses
`100 (1 - sum(p) / sum(p sqrt(n))) / (1 - 1/sqrt(N))`, where `p` is patch perimeter, `n` is patch cell
count and `N` is total valid landscape cell count. Square cells match the standard raster perimeter
convention; rectangular cells use their physical perimeters explicitly. Aggregation uses single-count
adjacencies and the compact-square maximum. Classes containing a single cell have undefined
aggregation (NaN), and empty/single-cell landscapes have undefined cohesion.

By default, contagion uses class area proportions multiplied by row-normalized directed adjacency probabilities,
with normalization by twice the log of the number of present classes. It is NaN with fewer than two
classes or when a present class has no valid adjacency. Set `contagionWeighting: 'adjacency'` for
landscapemetrics 2.2.1's joint-adjacency distribution, each pair count divided by all valid pair counts.
The modes differ when class area shares differ from their adjacency shares, including interior
background. This explicit choice preserves the area-weighted default and upstream reproduction.

## `addPatchProximityRecipe` — EC02

Use complete-domain patch labels and physical rectangular cell sizes. `edgeDepth` is a distance in
the same units. A pixel is core only when its center is strictly farther than this depth from its
patch exterior, including background and the outer raster boundary. The exact cell-rectangle
boundary distance is used, not the distance between cell centers. `core` retains the per-cell mask.

`distances` contains pairwise nearest rectangle-boundary distances; diagonal entries are infinity.
`metrics` contains `[patchArea, coreArea, nearestSameClassPatchDistance, proximity]`. Self hits are
excluded; proximity sums other same-class patch area divided by squared edge-to-edge separation for
patches within `searchRadius`. No eligible neighbor gives infinite nearest distance and zero proximity.
Distinct same-class labels touching at an edge/corner yield zero distance and infinite proximity;
reconcile connected-component labels before treating these as ecological patches. Nodata is a core
barrier, but proximity is Euclidean separation, not a least-cost movement path. Process seams before
summaries. The exact separable rectangle transform computes horizontal distance to each patch, then vertical
Euclidean minimization. Core classification examines only the physical depth neighborhood. Work is
`O(P*C*(width+height) + P²*C)` and scratch is `O(P*C)` for `P` patches and `C` cells, replacing the
former pairwise `O(P²*C²)` cell comparison. No approximation or jump-flood error is introduced.
`intermediates.horizontal` and `intermediates.fields` expose the distance construction. Optional
`reconcilePatches: true` with `connectivity` performs the same seam union before measuring distances;
merged-away patch rows remain empty. Large patch counts still require explicit memory budgeting.

The default `distanceModel: 'cell-boundary'` retains rectangle separation. Choose `'cell-center'`
for raster edge-cell-center ENN distances, matching landscapemetrics; no eligible neighbor then gives
NaN nearest distance and zero proximity. The same physical search radius applies to the selected
metric. `coreModel: 'raster-erosion'` interprets integer `edgeDepth` as four-neighbor erosion layers,
while the default `'physical'` retains physical center-to-boundary depth. `coreBoundary: 'edge'`
(default) treats the outer domain as an edge; `'open'` ignores the domain exterior. Interior nodata
and other patches remain barriers in both modes. Direct landscapemetrics 2.2.1 core fixtures map
`consider_boundary=FALSE` to edge and `TRUE` to open, reflecting that pinned implementation's actual
padding behavior. These are explicit alternatives, not silent changes to existing results.

`urban-landscape.R` directly executes upstream core and ENN on two separated patches touching the
domain boundary. The pinned package has no proximity-index function (`lsm_p_prox` is absent); the
proximity fixture explicitly composes its direct patch area and ENN for exactly two patches as
`otherArea / ENN²`. This composed evidence is distinguished from direct upstream core/ENN calls.

## `addResistanceConnectivityRecipe` — EC03 (deprecated)

Deprecated: use `addSparseResistanceConnectivityRecipe`. This recipe solves the whole system in one
GPU thread. It stays exported unchanged for existing callers.

Supply an undirected `edges` list (each edge once), nonnegative finite `conductances`, `vertexCount`,
and a source/sink pair. Parallel conductances add, self edges carry no net current, and zero
conductance is a barrier. Convert resistance to conductance explicitly (`1 / resistance`); raster
neighbor weighting and diagonal-distance conventions belong in that adapter.

The initial implementation solves at most 256 vertices and 65,536 edges with bounded grounded
conjugate gradients. It grounds the sink, identifies its connected component on the GPU, and applies
unit source injection/sink extraction. Vertices outside the component have NaN potentials. The
solver independently recomputes every vertex's current balance after iteration; convergence requires
the true maximum absolute current residual to meet `tolerance`.

- `potentials`: raw candidate voltages (including when the iteration budget is exhausted).
- `currents`: signed current from the first endpoint to the second; NaN unless the solve converged.
- `diagnostics`: `[effectiveResistance, maximumTrueResidual, dissipatedEnergy, recursiveResidualNorm]`.
- `status`: `[converged, iterations, sourceReachable, invalidEdges]`.

Effective resistance is published only after true-residual convergence. A disconnected pair or invalid
conductance produces no finite resistance certificate. Disconnected positive-conductance edges have
undefined currents because their potentials are ungrounded. Iteration exhaustion is not evidence of
an impossible network. Float32 conditioning remains a limit even with tight tolerances; independent
SciPy series/parallel network fixtures check potentials, currents and resistance. The solver is
compact and serial and is limited to 256 vertices and 65,536 edges. For larger systems use
`addSparseResistanceConnectivityRecipe` below, and `addRasterResistanceConnectivityRecipe` for a
resistance raster.

## `addSparseResistanceConnectivityRecipe` — EC03

Sparse grounded weighted-Laplacian resistance solve with unit current injection at `source` and
the `sink` held at potential zero. It runs on public luma.gl APIs: `GPUSort` and `GPUCOOToCSR`
build the CSR matrix, and a domain-specific Jacobi-preconditioned conjugate gradient (an unrolled,
GPU-gated bounded sequence) solves it. Use it instead of `addResistanceConnectivityRecipe` when a
network exceeds 256 vertices or 65,536 edges; the recipes are not interchangeable automatically.

Props: `edges` (`uint32x2`, each undirected edge once; 1 to 2,097,152 rows), `conductances`
(`float32`, one finite nonnegative value per edge), `vertexCount` (2 to 1,048,576), `source`, `sink`,
`maximumIterations` (1 to 4,096, default 1,024, compile-time), `tolerance` (absolute maximum
current-conservation residual, default 1e-4) and `connectivityIterations` (1 to 1,024, default 128,
compile-time). The unrolled iterations set the command node count, which the result reports as
`estimatedCommandNodeCount` (an upper bound: a fixed allowance of 64 for the sort, CSR assembly and
setup, which the published sort and scan grow with the edge count up to 55 at the largest input,
plus 5 per PCG iteration and 2 per connectivity round, at most about 22,700) and checks against the default budget of 32,768 nodes before any node is added. Parallel
conductances add, zero conductance is a barrier and self edges carry no
current. An endpoint at or beyond `vertexCount`, or a non-finite or negative conductance, makes the
edge invalid: invalid edges are counted in `status.w` and nothing is solved.

- `potentials`: candidate voltages, also when the iteration budget is exhausted. NaN when the solve
  did not run, or when the component labels converged and the vertex is outside the sink component.
- `currents`: signed current from the first endpoint to the second; NaN unless converged.
- `vertexCurrents`: Circuitscape-style node current, the larger of the summed inflow and outflow of
  the incident edge currents (equal to `(sum |branch| + |injection|) / 2` under conservation, so the
  source and sink read 1). NaN unless converged, or outside the sink component.
- `diagnostics`: `[effectiveResistance, maximumTrueResidual, dissipatedEnergy, recursiveResidualNorm]`.
  Effective resistance is `potentials[source]`, NaN unless converged. Residual and energy are NaN
  unless the solve ran. The energy of a unit current equals the effective resistance.
- `status`: `[converged, iterations, reachable, invalidEdges]`. `iterations` is the number of PCG
  iterations actually run. `reachable` is 1 when the source and sink share a component label (labels
  spread only along positive-conductance edges, so a match proves connectivity), 0 when the labels
  converged and differ or any edge is invalid, and 2 when they differ but propagation did not
  converge within `connectivityIterations`. Raise `connectivityIterations` when `reachable` is 2.
  The solve still runs when `reachable` is 2; nothing is masked in that case. `converged` is 1 only
  when no edge is invalid, `reachable` is not 0, the independently recomputed true maximum current
  residual (over every vertex) is at most `tolerance`, and the resistance and energy are finite.
  When `reachable` is 2 (labels undetermined), `converged` additionally requires the maximum
  residual to be below `1 / vertexCount`, which proves the source reaches the sink.

The provenance records the inputs, tolerance, both iteration budgets and
`solver: 'grounded-jacobi-conjugate-gradient'`.

Limits: 1,048,576 vertices and 2,097,152 edges (4,194,304 half-edges; this covers a 1024 x 1024
four-neighbour raster with 2,095,104 edges). The graph holds about `5 * maximumIterations +
2 * connectivityIterations + 30` nodes plus the sort nodes (at most about 22,600), below the
repository compile-cost cap of 32,768. The default configuration has about 5,400 nodes.

Precision: storage and arithmetic are float32. Dot products and the residual and energy reductions
are deterministic pairwise trees without float atomics, so results are reproducible for fixed inputs,
device and size. The dominant error is cancellation in `v_i - v_j` (about `eps * |v| * g`), which
grows with conductance contrast times voltage magnitude (`v_source` equals the effective
resistance). Normalise conductances so that the largest is about 1 before calling; the recipe does
not rescale silently. If `converged` is 0 with a small residual, loosen `tolerance` or rescale.
The default tolerance is 1e-4 (the small recipe uses 1e-5). Validation: the Circuitscape.jl 7-vertex
golden for all six pairs (voltages, branch currents and node currents), agreement with the small
recipe on a 200-vertex graph with a disconnected component, and a 64 x 64 lattice against a float64
oracle.

Iteration count and watchdog risk: Jacobi-PCG on an `N x N` grid needs O(N) iterations (roughly 200
to 600 at N = 64, thousands at N = 1024); the cap is 4,096 and an unconverged run reports
`converged = 0` with its residual. The whole graph is one submission with no continuation. A
1024 x 1024 problem at 4,096 iterations moves hundreds of gigabytes in one command buffer and can
trigger a GPU watchdog device loss (about 2 s on Windows and Chrome). Keep large problems to fewer
iterations or smaller grids. About 20,000 dispatches are encoded per execution even after
convergence (gated dispatches run zero workgroups), so expect several to tens of milliseconds of
encoding. No multigrid or stronger preconditioner is provided.

## `addRasterResistanceConnectivityRecipe` — EC03-R

Circuit connectivity across a resistance raster between one source cell and one sink cell. Each cell
is a vertex. Each pair of neighbouring cells with data is an edge with conductance

`g(a, b) = 2 / (R(a) + R(b)) / edgeLength(a, b)`,

the inverse of the mean resistance of the two cells per unit of ground distance between their
centres. Diagonal neighbours use the diagonal length. Resistance is per unit ground length: an edge
is a resistor of `mean(R) * edgeLength`.

This is not Circuitscape's convention. Circuitscape's raster pairwise mode uses `1 / mean(R)` for
cardinal neighbours and `1 / (sqrt(2) * mean(R))` for diagonal ones and never divides by the cell
size. On a planar grid of square cells of size `s` (`cellSizeMode: 'uniform'`) every conductance here
is Circuitscape's divided by `s`, so the current map is the same and potentials and effective
resistance are `s` times Circuitscape's (two cells with `R = 1` and `s = 30` give 30; Circuitscape
gives 1). With rectangular cells, and in `'geographic'` and `'web-mercator'` mode where the edge
length changes with the row, the conductance ratios differ and the current map differs too. The spec
pins the convention with two-cell, three-cell and 2 x 2 circuits solved by hand; no Circuitscape
raster output is used. The edge list and the conductances are built on
the GPU, one invocation per edge slot, and solved by `addSparseResistanceConnectivityRecipe` with
unit current injected at `source` and the `sink` held at potential zero. The recipe composes its
stages through the recipe builder and returns `contributors`, `outputs`, `intermediates`, `status`
(per stage) and `provenance`.

Props: `grid` (the raster grid), `resistance` (a published raster band: buffer or texture; scale,
offset and the no-data value are resolved first), `parameters` (at least 8 float32 values packed by
`getRasterResistanceConnectivityParameterValues(grid)`: the affine and the row georeference, so a
new georeference does not recompile), `neighborhood` (4 or 8, default 8), `cellSizeMode` (default
`'uniform'`), `source` and `sink` (cell indices `row * width + column`), and the sparse solve's
`maximumIterations` (1 to 4,096, default 1,024), `tolerance` (default 1e-4) and
`connectivityIterations` (1 to 1,024, default 128). Compile-time: the grid size, `neighborhood`,
`cellSizeMode`, `source`, `sink`, `tolerance` and the iteration budgets; per-frame: `parameters` and
the resistance values.

Edge length: in `'uniform'` mode the affine maps the pixel step to a world step, so rotated and
sheared grids are exact (lengths are in affine units; geographic degrees are not meters).
`'web-mercator'` and `'geographic'` measure WGS84 ground meters with the row scale at the midpoint
row of the edge, the rule of the terrain flow measures, and need an axis-aligned grid.

Edge slots have a fixed layout, `cell * slotsPerCell + direction`: east and south for four
neighbours (two slots per cell), plus south-east and south-west for eight (four slots per cell).
A slot whose neighbour is off the grid is the self edge `(cell, cell)`; a slot touching a no-data
cell has zero conductance. `currents`, `edges` and `conductances` follow this layout.

No-data, zero and negative resistance:

- No-data (zero validity, the band's no-data value, or a non-finite resistance including
  `+Infinity`) removes the cell from the circuit, like an absolute barrier: its edges have zero
  conductance and its potential and current are NaN.
- A cell with data and a zero or negative resistance is invalid. It cannot be represented in a
  conductance Laplacian (zero resistance is an infinite conductance), so it is not silently treated
  as a barrier or a short circuit: such cells are counted in stage `resistance` (`invalidCount`),
  every edge they share with a cell with data gets a NaN conductance, and the sparse solve counts
  those edges (`solverStatus.w`, stage `solve` `invalidCount`) and publishes no potentials. Clamp
  resistances to a positive minimum, or mark them no-data, before calling.
- Two neighbouring positive resistances so small that `2 / (R(a) + R(b)) / edgeLength` exceeds the
  float32 maximum (about 3.4e38; for example `R = 1e-35` on both sides of a `1e-4` cell) are
  invalid in the same way: the edge gets a NaN conductance and both cells are counted in stage
  `resistance`, so that count is nonzero whenever the solve reports invalid edges.

Outputs (caller-owned): `potentials` and `vertexCurrents` (one per cell, the current map; NaN outside
the sink's component), `diagnostics` and `solverStatus` as in the sparse recipe (`solverStatus` is
its `status`), and optionally `currents` (one per edge slot), `edges` and `conductances`. The status
stages are `resistance` (`invalidCount`) and `solve` (`converged`, `iterationCount`,
`iterationLimitReached`, `invalidCount`); pass `output.status` views to read them, otherwise they
are graph-owned.

Limits: at most 1,048,576 cells (the vertex limit of the sparse solve) and at most 2,097,152 edge
slots (its edge limit): 1024 × 1024 cells with four neighbours, 524,288 cells (for example
1024 × 512) with eight. Both are rejected at construction with a message that names the limit. The
command node count is the sparse recipe's plus an allowance of 16 for the adapter stages, reported
as `estimatedCommandNodeCount`. The precision and watchdog notes of the sparse recipe apply; with
ground meters as edge lengths the conductances are small, which the solve tolerates but does not
rescale.

Validation: a binary64 dense circuit solve of the same edge list on small grids (uniform resistance,
a no-data wall with a one-cell gap that carries the whole current, no-data holes including NaN and
`+Infinity`, both neighbourhoods, a sheared affine and a geographic grid) checks every conductance,
potential and node current and the effective resistance; a closed wall reports the sink unreachable;
zero and negative resistances, and positive resistances whose conductance overflows float32, are
counted and block the solve. The edge list is checked against an adjacency built by comparing
every pair of cells (each neighbouring pair exactly once, no other edges) on non-square grids, one-row
and one-column grids and non-square cells, with four and eight neighbours.

## `addSegregationDecompositionRecipe` — EC04

Supply four common-support population tables in this order: baseline, composition-only
counterfactual, spatial-only counterfactual, final. Every `float32x2` row holds focal and complement
group counts. `supportId` and `counterfactualMethod` record the caller's explicit alignment and
counterfactual construction. This recipe does not silently align geographic units, construct a
counterfactual or infer that two equal-length tables describe the same boundaries.

Each table's dissimilarity is `0.5 sum(abs(focal/focalTotal - complement/complementTotal))`.
Negative/nonfinite counts, or a zero regional group total, invalidate that scenario. `indices`
contains the four statistics, and `validity` preserves each scenario's validity. `decomposition`
contains `[observedChange, compositionComponent, spatialComponent, conservationResidual]` using the
two-factor Shapley average of both replacement orders. Its components sum to observed change up to
floating-point error. This compares supplied counterfactuals. Optional `randomization` adds a paired within-support label
exchange test: each spatial unit independently swaps baseline/final labels, and each replicate
recomputes both dissimilarities. The null requires exchangeability of those two labels within each
unit; it is not a spatial-independence or random-location assumption. Philox seed/replicate/unit
identity is replayable, or supply a `swaps` mask with one 0/1 value per replicate/unit. The distribution
and inference `[observedChange, twoSidedPlusOneP, validReplicates, extremeReplicates]` are separate
outputs. Any invalid replicate withholds the p-value instead of silently dropping it. Alternative
segregation statistics remain separately named analyses.

## Explicit support alignment and counterfactual generation

`addPopulationSupportAlignmentRecipe` joins `sourceIds` and `targetIds` on GPU, reordering focal and
complement counts onto a caller-chosen support. Duplicate or missing IDs produce NaN rows.
`status` is `[complete, missingTargetRows, ambiguousTargetRows, excludedSourceRows]`; completeness
requires a one-to-one support match with no excluded mass. This is exact ID alignment, not implicit
polygon overlay or areal interpolation across changed boundaries. Use the existing conservative
areal redistribution before this join when geographic supports differ.

`addSegregationCounterfactualRecipe` takes aligned baseline/final tables and explicitly selects
`composition`, `dual-composition` or `share`. It implements the PySAL empirical-CDF convention:
average-tie percentile ranks map to linearly interpolated quantiles of the comparison distribution.
Composition preserves each source unit's total; dual composition maps both groups; share normalizes
mapped group shares to preserve each source regional group total. Zero-population units have zero
composition. Both replacement-order tables are returned and passed to the existing Shapley recipe.
The current implementation supports 2,048 units, with bounded GPU sorting/ranking; inputs are not
silently sampled. The package's baseline-minus-final decomposition sign is reversed to this API's
explicit final-minus-baseline convention.

## References and measured scaling

`modules/analysis/test/oracles/urban-ecology.py` records NumPy/SciPy/Shapely/NetworkX composed references.
`urban-ecology-advanced.py` directly executes segregation 2.5.3 for all three counterfactual methods,
Shapley components and all replay assignments of the four-unit paired-label fixture. It pins momepy
0.10.0, NumPy 2.2.6 and Shapely 2.1.2 and records the pandas version; these are direct upstream calls,
not reimplementations of the GPU formulas. `urban-landscape.R` directly executes landscapemetrics
2.2.1 and terra 1.9.50 for patch area/perimeter, class and landscape mesh/cohesion/aggregation (including a single-cell class), both boundary-edge
policies, and joint-adjacency contagion on square-cell fixtures with interior background. Upstream
hectares are converted to square projected units. Rectangular-cell physical edge lengths are tested
independently; this is not blanket upstream metric parity. Circuitscape Julia was not executed locally. The small resistance solver additionally passes six
source/sink comparisons against the pinned Circuitscape upstream `sgNetworkVerify2` golden files:
all seven-node potentials, absolute branch currents and unit-source effective resistances agree
within `1e-4`, alongside true residual and energy checks. These are authentic upstream regression
outputs, not values generated by our SciPy oracle. The pinned tree
`4e0d29d3a7f2ed542a9f2ade782ea554c54b4ae7` identifies Circuitscape.jl 6.1.0, but its legacy
verification configuration declares 3.5.5 and the historical golden-producing binary is unspecified.
No freshly executed Julia-version parity is claimed. The repository provenance manifest and licenses described in `modules/analysis/test/oracles/upstream-golden/README.md`
retain source URLs, hashes and unmodified inputs/outputs. Independent SciPy nonuniform-network
solves and exact electrical-network identities cover additional cases. The focused GPU golden
suite passed six cases; a node integrity test verifies the retained bytes and expected values.

The measured execution record
separates preparation, compilation, first execution, repeated resident wall time, readback and
scratch memory. Native Apple WebGPU passed exact proximity at 32², 128² and 256² cells; the largest resident median was
about 18.4 ms. Sparse resistance benchmarks are pending re-measurement (EC03).

The 2026-10-24 execution record
reruns those proximity curves and adds complete class/landscape summaries at 32²,
128² and 256² cells. Two rectangular classes provide closed-form aggregation, pooled cohesion,
mesh and area checks; the largest absolute certificate error is below `1e-5`. Landscape summary
resident medians were approximately 1.3, 9.2 and 34.4 ms, including seven graph stages and the
certificate copy. These measurements cover two patches, not high patch-count landscapes. The
focused final numerical/lens/reference suites passed 88 GPU tests; node/kernel-capacity audits
passed 35 tests. Final repository-wide gates are recorded separately.
