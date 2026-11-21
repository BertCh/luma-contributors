# Landscape ecology recipes

Related tasks: [urban morphology and visibility](./urban-morphology.md) · [workflow recipes](./recipes.md).

All recipes append GPU kernels to a caller-owned command graph without hidden CPU computation or
readback. Inputs remain caller-owned. Outputs are graph-owned transient views unless the caller passes
an `output` prop (see below); export transient results through a graph copy when CPU readback is
needed. The current bounded implementations prioritize explicit
numerical definitions, provenance and failure evidence over large-domain performance claims.

## Caller-owned outputs and limits

Most recipes take an optional `output` prop. `addRasterResistanceConnectivityRecipe` requires
`output.potentials`, `output.vertexCurrents`, `output.diagnostics` and `output.solverStatus`. Every
supplied view must belong to the same graph and have the required format and length. Omitted
optional destinations use graph-owned transient views. The returned `outputs` map contains the
views the recipe used.

| Recipe | `output` entries |
| --- | --- |
| `addLandscapeMetricsRecipe` | `patchMetrics`, `classMetrics`, `classAdjacency`, `aggregation`, `landscape`, `landscapeAggregation`, `landscapeCohesion`, and with `reconcilePatches` `reconciledPatches`, `patchParents` |
| `addPatchProximityRecipe` | `core`, `distances`, `metrics`, `horizontal`, `fields`, and with `reconcilePatches` `reconciledPatches`, `patchParents` |
| `addResistanceConnectivityRecipe` | `potentials`, `currents`, `diagnostics`, `status` |
| `addSparseResistanceConnectivityRecipe` | `potentials`, `currents`, `vertexCurrents`, `diagnostics`, `status` |
| `addRasterResistanceConnectivityRecipe` | required `potentials`, `vertexCurrents`, `diagnostics`, `solverStatus`; optional `currents`, `edges`, `conductances` and the `status` scalars |
| `addSegregationDecompositionRecipe` | `indices`, `validity`, `decomposition`, and with `randomization` `distribution`, `inference` |
| `addSegregationCounterfactualRecipe` | the decomposition entries, `compositionOnly`, `spatialOnly` |
| `addPopulationSupportAlignmentRecipe` | `populations`, `alignmentCode`, `matchedMask`, `status` (storage for the named status stages) |

Limits are named constants and are rejected with a message that starts with the recipe ID and names
the prop: 256 classes (landscape metrics), 256 vertices, 65,536 edges and 4,096 iterations (dense
resistance), 1,048,576 vertices and 2,097,152 edges (sparse resistance), 1,048,576 cells and
2,097,152 edge slots (raster resistance), 65,536 replicates (decomposition) and 2,048 units
(counterfactual). Patch proximity
checks its patches-times-cells distance fields against the device storage binding size.

The landscape and segregation totals are reductions (`GPUReduction`) over per-patch or per-unit
terms, so the sums are tree sums rather than a serial loop and can differ from the former results in
the last float32 digits. Landscape patch measurements reuse `GPURasterPatchMetrics`; bounded
16-column raster runs accumulate integer same-class side counts and class adjacency. Class
perimeter, squared-area and weighted-perimeter sums use deterministic sorted segmented reductions.
Patch reconciliation unions seam neighbors in parallel and compresses their roots, always retaining
the lowest original patch ID. The counterfactual quantile mapping remains bounded at 2,048 units.

## `addLandscapeMetricsRecipe` — EC01

Caller-owned output buffers must be distinct from each other and from both input buffers,
including the optional reconciliation outputs. The recipe rejects aliases before appending nodes.

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
Every supplied `output` view is checked for packed format, exact logical length, ownership,
STORAGE binding usage/ranges and aliases before allocating defaults or scheduling reconciliation. Patch metrics/parents have
one row per patch; class metrics/aggregation one per class; adjacency has `classCount` squared rows;
reconciled labels have one row per cell; landscape metrics have one row. Correct a rejected view and
retry with the same recipe ID on the unchanged graph.
Use exactly one patch-label spelling: `patches` is the legacy zero-based patch-row raster,
with out-of-range rows as background; `denseLabels` accepts the native `GPURasterPatchLabels`
port (0 background, labels 1..N). The dense adapter maps label `k` to patch row `k - 1` and retains
`converged`, `overflow` and `componentCount` evidence; never pass native labels as `patches`.
For unit cells, native labels `[0, 1, 1]` with one class give patch area **2**, excluding the first
background cell. Keep `patchClasses[0]` as the class for label 1. Incomplete native labels suppress
the dependent measurements; preserve the input status when retrying.

```ts
import type {GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import {addLandscapeMetricsRecipe} from '@luma-contributors/analysis/gpu-spatial-analysis';

declare const graph: GPUCommandGraph;
// Native 3 x 1 labels [0, 1, 1], with complete producer status, imported into graph.
declare const denseLabels: NonNullable<Parameters<typeof addLandscapeMetricsRecipe>[1]['denseLabels']>;
declare const patchClasses: GraphDataView<'uint32'>; // one row, containing class 0
const landscape = addLandscapeMetricsRecipe(graph, {
  id: 'native-landscape', width: 3, height: 1, cellSize: [1, 1],
  denseLabels, patchClasses, classCount: 1, connectivity: 4
});
// landscape.outputs.patchMetrics row 0 area = 2 square units (two foreground cells).
```

Both inputs, `patches` and `patchClasses`, must be packed `uint32` views whose buffers include
`Buffer.STORAGE`. Format, stride, graph ownership and aligned binding ranges/usage are checked before
any defaults or reconciliation, so correcting a rejected input permits the same retry.

- `patchMetrics`: `[area, perimeter, cellCount, sameClassDirectedSideCount]`.

The following slots are scalar words after reading the packed outputs; `p` and `c` are zero-based
patch and class rows. Areas use square `cellSize` coordinate units and perimeters use its length
unit. Percentages are already 0–100, with no hectare conversion.

| Metric | Actual slot | Unit and undefined cases |
| --- | --- | --- |
| AREA | `patchMetrics[4 * p]`, `classMetrics[4 * c]`, `landscape[2]` | Square coordinate units; empty patch rows have 0 area; excluded patch rows retain their measured area; excluded class rows and valid landscape area omit those cells |
| MESH | `classMetrics[4 * c + 2]`, `landscape[1]` | Square coordinate units; class MESH is NaN with no valid landscape cells (the landscape sum is then NaN) |
| COHESION | `classMetrics[4 * c + 3]`, `landscapeCohesion[0]` | Percent; NaN with at most one valid landscape cell or zero weighted perimeter |
| AI | `aggregation[c]`, `landscapeAggregation[0]` | Percent; a class with at most one cell is NaN; its landscape contribution is 0; an empty landscape is NaN |
| CONTAG | `landscape[0]` | Percent; NaN with fewer than two present classes or a present class with no valid adjacency |

On metre grids, MESH in hectares is `landscape[1] / 10_000`; AREA uses the same conversion.
A `100 m × 100 m` single patch has `10,000 m² = 1 ha`, and its landscape MESH is also
`10_000 m²`, hence `1 ha`. For a single valid raster cell, class `aggregation[0]` is NaN and
`landscapeAggregation[0]` is 0; cohesion is NaN. These cell-count cases hold even when the cell's
physical area is 10,000 m².
Patch measurements retain excluded-class areas: patch rows `[0, 0, 1]`, classes `[0, 1]`,
`backgroundClass: 0` and cell area 6 produce patch areas `[12, 6]`, but valid landscape area 6.
Class/landscape exclusion does not erase the measured geometry of an existing patch.

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

The measurement path uses scratch proportional to cells plus patches plus the class adjacency
table, with no patches-times-cells allocation or raster scan per patch/class pair. The input labels
are adapted from zero-based patch rows to the dense one-based raster labels. Class and landscape
formulas and all public output layouts are unchanged; class tree sums may differ from the former
serial sums within float32 rounding tolerance (relative `1e-5`). Empty patch rows remain zero,
including merged-away rows; empty class aggregation and cohesion remain NaN.

The four-sides-per-cell integer counters require at most 1,073,741,823 cells. Device storage binding
limits also apply to the cell labels (4 bytes per cell) and patch output (16 bytes per patch); at the
portable 128 MiB limit these permit at most 33,554,432 cells and 8,388,608 patch rows. These limits
are checked before graph nodes are appended. The final dense class summary remains bounded at
256 classes. No timing claims are made for this implementation until GPU measurements are run.

## `addPatchProximityRecipe` — EC02

Explicit `mode: 'boundary'` uses O(cells + patches + candidateCapacity) storage. `candidateRadius`
(default `searchRadius`) bounds the neighbor search. Inspect per-patch certification and
`status.candidateOverflow`: a radius cannot certify an omitted neighbor, and capacity overflow
invalidates all rows. In boundary mode `status.overflow` is always 0 and `status.requiredCount` is the patch
count; `status.count` is the number of valid patches. The first two words of the four-word
`uint32` view `intermediates.candidateRequirement` are the unclamped directed boundary-cell
candidate count and the capacity-exhaustion flag (0 or 1); the remaining words are reserved scratch.
Insufficient radius can set `candidateOverflow` while these first two words remain `[0, 0]`;
capacity exhaustion sets word 1 and invalidates every patch row. The reference layout remains the
default. Boundary work is bounded by 4,096 neighborhood visits, 1,073,741,823 rows and 1,431,655,765 candidate records, plus device limits.

Use exactly one patch-label spelling: `patches` is the legacy zero-based patch-row raster,
with out-of-range rows as background; `denseLabels` accepts the native `GPURasterPatchLabels`
port (0 background, labels 1..N). The dense adapter maps label `k` to patch row `k - 1` and retains
`converged`, `overflow` and `componentCount` evidence; never pass native labels as `patches`.
For unit cells, native labels `[0, 1, 1]` with one class give patch area **2**, excluding the first
background cell. Keep `patchClasses[0]` as the class for label 1.

The `proximity.inputStatus?.converged`, `proximity.inputStatus?.overflow` and
`proximity.inputStatus?.componentCount` guards are GPU views. After the submission finishes, read word zero
from each supplied guard's buffer, honoring the view's byte offset. Compare the decoded component count
with `patchClasses.length`; compare the decoded convergence and overflow words with zero.
A supplied convergence word of zero, nonzero producer overflow, or too many
components suppresses dependent measurements to NaN. In boundary mode it also clears every
`outputs.validity` row and `status.count` to zero; requested reconciliation labels and parents
become `0xffffffff`. This upstream labeling loss does not set search `status.candidateOverflow`.
That flag retains its radius/candidate-capacity meaning and the four-word
`intermediates.candidateRequirement` layout above remains unchanged. Check both input guards and
search diagnostics: a zero search flag cannot certify unfinished labels.

For nonconvergence, recover at the labeling producer (for example, increase its iteration budget
and finish labeling), then rerun the dependent recipe with the producer's guards still connected.
For producer/component capacity loss, rebuild the producer and affected consumers with sufficient
storage and patch-class rows. For search loss, increase `candidateRadius` or `candidateCapacity`
as indicated by the search diagnostics and rebuild. A later clean submission with converged,
fitting labels and sufficient search coverage restores validity, count, measurements and
reconciliation; resubmission can reuse existing allocations when the new input fits them.

```ts
import type {GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import {addPatchProximityRecipe} from '@luma-contributors/analysis/gpu-spatial-analysis';

declare const graph: GPUCommandGraph;
// Native 3 x 1 labels [0, 1, 1], with complete producer status, imported into graph.
declare const denseLabels: NonNullable<Parameters<typeof addPatchProximityRecipe>[1]['denseLabels']>;
declare const patchClasses: GraphDataView<'uint32'>; // one row, containing class 0
const proximity = addPatchProximityRecipe(graph, {
  id: 'native-proximity', width: 3, height: 1, cellSize: [1, 1],
  denseLabels, patchClasses, edgeDepth: 0, searchRadius: 5
});
// proximity.outputs.metrics row 0 patchArea = 2 square units, not 1 background cell.
```

Use complete-domain patch labels and physical rectangular cell sizes. Landscape metric cell widths
and heights must remain finite and positive after float32 rounding; values such as `1e-47` round
to zero and are rejected before output reservation. `edgeDepth` is a distance in
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
`getRasterResistanceConnectivityParameterValues(grid, target, cellSizeMode)`: the affine and the row georeference, so a
new georeference does not recompile), `neighborhood` (4 or 8, default 8), `cellSizeMode` (default
`'uniform'`), `source` and `sink` (cell indices `row * width + column`), and the sparse solve's
`maximumIterations` (1 to 4,096, default 1,024), `tolerance` (default 1e-4) and
`connectivityIterations` (1 to 1,024, default 128). Compile-time: the grid size, `neighborhood`,
`cellSizeMode`, `source`, `sink`, `tolerance` and the iteration budgets; per-frame: `parameters` and
the resistance values.

Edge length: in `'uniform'` mode the affine maps the pixel step to a world step, so rotated and
sheared grids are exact (lengths are in affine units; geographic degrees are not meters).
`'web-mercator'` and `'geographic'` measure WGS84 ground meters with the row scale at the midpoint
row of the edge, the rule of the terrain flow measures, and need an axis-aligned grid on every
frame. When updating the georeference, call
`getRasterResistanceConnectivityParameterValues(grid, target, cellSizeMode)` with the recipe's
fixed mode; pass `undefined` for `target` when allocating a fresh parameter array. Geographic and
Web Mercator updates reject rotation/shear. Uniform mode supports both rotation and shear.

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
are graph-owned. Read the third word of `solverStatus` for the sparse solve's reachability code.

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
outputs. Any invalid replicate withholds the p-value instead of silently dropping it. A replicate
is extreme when its absolute change is at least that of the observed change recomputed in the
replicates' own serial summation order, so an exchange that leaves both tables unchanged (no swap,
every swap, or swaps of units whose baseline and final rows are equal) ties it exactly; the
published `observedChange` stays the tree-reduced difference of `indices`, which can differ from
it by a few float32 units in the last place. Every option is validated before the recipe adds a
command node. Caller outputs must have disjoint logical byte ranges from all population tables,
optional swap masks and other outputs, including aliases imported through different handles.
Disjoint ranges may share one graph buffer handle when they are used in separate dispatches;
co-bound read/write views must use separate physical buffers. These checks occur before scenario
nodes or transient outputs are added, so a corrected output can retry the same recipe ID.
All supplied binding ranges and STORAGE usages are also checked here, including large optional
swap tables and caller-owned inference outputs; an invalid device binding reserves no output IDs.
Alternative segregation statistics remain separately named analyses.

## Explicit support alignment and counterfactual generation

`addPopulationSupportAlignmentRecipe` joins `sourceIds` and `targetIds` on GPU, reordering focal and
complement counts onto a caller-chosen support. Duplicate or missing IDs produce NaN rows.
The recipe returns `{contributors, outputs, intermediates, status, provenance}`. Under
`status.stages`, `missing-targets`, `ambiguous-targets` and `excluded-sources` each expose a
one-row `uint32` `status.invalidCount`. The `alignment` stage exposes a one-row `uint32`
`status.complete`, set to 1 when the supports match one-to-one with no excluded source rows.
The four views address the corresponding words of the contributor's certificate and require no
extra GPU commands. `intermediates` is empty because this recipe adds one contributor.
Use conservative areal redistribution before aligning IDs when geographic supports differ.

## `GPUPopulationSupportAlignment`

::lens{scenario=gpu-spatial-analysis.population-support-alignment rung=result caption="Inspect reordered populations, ambiguous rows and completeness counts."}

### How it computes

{/* lens:GPUPopulationSupportAlignment */}

::lens{scenario=gpu-spatial-analysis.population-support-alignment rung=graph caption="Follow indexed joins and parallel status reductions."}

The synthetic fixture includes a unique reordered match, duplicate source and target IDs, a missing
target, an excluded source, and a valid `0xffffffff` ID. Its target alignment codes are
`[0, 2, 1, 2, 2, 0]`, its matched mask is `[1, 0, 0, 0, 0, 1]`, and its packed status is
`[0, 1, 3, 1]`.

The class takes `GPUPopulationSupportAlignmentProps` with four required views.
`output.populations` is `float32x2` with one row per target. `output.alignmentCode` and
`output.matchedMask` are `uint32` with one row per target, and `output.status` is one
`uint32x4` row. `GPUPopulationSupportAlignmentRecipeProps` lets callers supply any of these
views under `output` and allocates transient views for omissions. The recipe returns the row
views under `outputs` and the certificate scalars under `status.stages`.
Two indexed `GPUKeyLookup` operations replace exhaustive source/target scans: one matches targets to
sources and reports excluded sources, and one detects repeated target IDs. Counts use parallel
reductions. Work is sorting plus indexed lookups; no invocation scans an entire support table.
Population values are copied without checking whether they are finite or nonnegative. This stage
aligns identifiers; downstream segregation statistics validate population counts.

The diagnostic `alignmentCode` is `0` for a unique match, `1` for no source, and `2` for an ambiguous
source or target. `POPULATION_SUPPORT_ALIGNMENT_CODES` exports `MATCHED`, `MISSING` and `AMBIGUOUS`;
`decodePopulationSupportAlignmentCode(code)` returns `'matched'`, `'missing'`, `'ambiguous'`, or
`'unknown'` for an unrecognized code. A missing source takes precedence over a repeated target.
`matchedMask` is `1` only for a unique match and `0` otherwise. Pass it as a downstream `mask`
(for example, to column quantiles); diagnostic codes must not select rows. The old `validity`
output has been removed. A matched row certifies its IDs only, not finite population values.
The recipe rejects any supplied `output.validity` override with migration guidance. The contributor
requires all four output views and rejects outputs sharing a graph handle or a known imported
physical buffer, naming both conflicting ports. Distinct imports of one buffer do not isolate writes.
Alignment recipes validate input layouts and all supplied output layouts, graph ownership and
aliases before allocating transient outputs, so a corrected override can retry with the same ID
on the same graph after a validation error.
Only the four declared output keys participate in layout, ownership and alias validation;
metadata on an override variable is ignored. The retired `validity` key is still rejected.
The contributor and operation forms use
`[complete, missingTargetRows, ambiguousTargetRows, excludedSourceRows]` for their packed status.
Excluded rows are counted individually even if they share an ID. Every uint32 identifier, including
`0xffffffff`, is a valid support ID. Internally, a zero high word widens keys to uint32x2 so the
key lookup's missing-key sentinel cannot discard that ID.

Per-frame inputs are the contents of `populations`, `sourceIds` and `targetIds`. View lengths and
output bindings are compile-time. All inputs must be single packed views, and widened keys and
sort scratch must fit device storage bindings. There are no coordinates or numeric parameters.

## `CPUPopulationSupportAlignmentOperation`

Prefer this CPU stage when support data is already on the host. It uses maps in expected
O(sourceCount + targetCount) time, then publishes exactly the same populations, alignment codes,
matched mask and packed status as the GPU contributor. `computePopulationSupportAlignmentOnCPU` exposes the pure
typed-array calculation without a device. The CPU implementation has no GPU scratch or binding
limit. Both paths require nonempty source/target support and exactly two population components per
source ID. Nonfinite or negative population values are copied unchanged. No arithmetic or additional
rounding is applied to populations.

## `GPUPopulationSupportAlignmentOperation`

Both operation classes accept the same `PopulationSupportAlignmentOperationProps`: logical graph
names for `populations`, `sourceIds` and `targetIds`, a positive `targetCount`, and output names for
`populations`, `alignmentCode`, `matchedMask` and `status`. Use these stages beside other CPU/GPU
operations in the published `GPUGraph`, compiling GPU stages with `GPU_CONTRIBUTOR_OPERATION_LOWERING`. The executor
handles transfers at CPU/GPU boundaries. The output populations can feed the next GPU contributor
without application-managed readback/upload glue.

```ts
import {GPUGraph} from '@luma.gl/graph/gpu-core';
import {CPUPopulationSupportAlignmentOperation} from '@luma-contributors/analysis/gpu-spatial-analysis';

// Number of target analysis-unit rows, fixed when constructing this operation.
const analysisUnitCount = 4;
const graph = new GPUGraph({id: 'population-alignment'});
graph.add(new CPUPopulationSupportAlignmentOperation({
  populations: 'sourcePopulation',
  sourceIds: 'sourceIds',
  targetIds: 'analysisIds',
  targetCount: analysisUnitCount,
  output: {
    populations: 'alignedPopulation',
    alignmentCode: 'alignmentCodes',
    matchedMask: 'matchedRows',
    status: 'alignmentStatus'
  }
}));
// Later GPU operations consume the logical value 'alignedPopulation'.
```

Node specs compare maps with an independent exhaustive scan over all 729 pairs of three-row
supports, hand-calculated duplicate/missing cases, and a 100,001-row reorder. Headless specs compare
every CPU/GPU output exactly (including NaNs and all uint32 IDs) and chain CPU alignment into GPU
alignment over 257 rows. Headless execution is part of the lead's verification gates.

## Segregation counterfactual generation

`addSegregationCounterfactualRecipe` takes aligned baseline/final tables and explicitly selects
`composition`, `dual-composition` or `share`. It implements the PySAL empirical-CDF convention:
average-tie percentile ranks map to linearly interpolated quantiles of the comparison distribution.
Composition preserves each source unit's total; dual composition maps both groups; share normalizes
mapped group shares to preserve each source regional group total. Zero-population units have zero
composition. Both replacement-order tables are returned and passed to the existing Shapley recipe.
The current implementation supports 2,048 units, with bounded GPU sorting/ranking; inputs are not
silently sampled. The package's baseline-minus-final decomposition sign is reversed to this API's
explicit final-minus-baseline convention.

Caller-owned outputs must have disjoint logical byte ranges from the baseline/final inputs and
from other results. To share an allocation between `compositionOnly` and `spatialOnly`, create both
disjoint subviews on **one graph buffer handle**. Separate imports of the same physical buffer are
rejected when either handle is written, even with disjoint ranges. Views bound together in one
dispatch additionally need disjoint aligned binding ranges. A read-only and a writable binding in
the same dispatch must use **different physical buffers**: for example, `indices` and
`decomposition` cannot share a buffer even at bytes 0 and 256. All supplied output formats, lengths,
graph ownership and these alias rules are checked before declaring transient views or adding
commands. After correcting a rejected override, the recipe can be retried with the same ID on the
same graph.

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

The 2026-10-24 execution record describes the former serial landscape measurement path; the
replacement above has not yet been benchmarked. That historical record
reruns those proximity curves and adds complete class/landscape summaries at 32²,
128² and 256² cells. Two rectangular classes provide closed-form aggregation, pooled cohesion,
mesh and area checks; the largest absolute certificate error is below `1e-5`. Landscape summary
resident medians were approximately 1.3, 9.2 and 34.4 ms, including seven graph stages and the
certificate copy. These measurements cover two patches, not high patch-count landscapes. The
focused final numerical/lens/reference suites passed 88 GPU tests; node/kernel-capacity audits
passed 35 tests. Final repository-wide gates are recorded separately.
