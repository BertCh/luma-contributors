# Landscape ecology recipes

All recipes append GPU kernels to a caller-owned command graph without hidden CPU computation or
readback. Inputs remain caller-owned. Outputs are graph-owned transient views. Export results through
a graph copy when CPU readback is needed. The current bounded implementations prioritize explicit
numerical definitions, provenance and failure evidence over large-domain performance claims.

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

## `addResistanceConnectivityRecipe` — EC03

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
SciPy series/parallel network fixtures check potentials, currents and resistance. The separately
named sparse recipe below removes this small solver's vertex cap; the original API retains its
compact serial algorithm for tiny graphs.

## `addSparseResistanceConnectivityRecipe` and `addRasterResistanceConnectivityRecipe` — EC03

The sparse recipe accepts the same edge/conductance/source/sink contract with up to 16,777,216
vertices, subject to actual storage limits. It constructs connected components by GPU union/find,
row-sorts the grounded Laplacian into CSR (parallel-edge diagonal entries add), and composes the
existing `GPUJacobiPCG`. Diagonal Jacobi preconditioning and per-iteration GPU activity gates are
explicit. Grounding leaves identity rows outside the sink component so the system remains SPD;
those unrelated voltages are masked to NaN after solving. Invalid edges are rejected before the
matrix can publish a certificate. Zero conductance remains a barrier.

The sparse outputs have the same voltage/current/diagnostic meaning as the small solver. Its
`status` is `[converged, configuredIterationBudget, sourceReachable, invalidEdges]`; the second field
is deliberately the configured budget, not an invented executed-iteration count. The solver
independently recomputes the original current residual, including sink conservation, and withholds
resistance/currents on exhaustion. `intermediates` exposes components, CSR and the true residual.
The configured maximum is 4,096 iterations. Float32 conditioning may prevent convergence even when
the mathematical network is connected; status is the certificate, not a finite candidate voltage.

The raster adapter accepts positive finite cell resistance, a planar `cellSize` and 4/8 connectivity.
Other resistance values are barriers. Each undirected neighbor pair is emitted once with conductance
`2/(resistanceA+resistanceB)/physicalEdgeLength`; diagonal lengths use the Euclidean cell diagonal.
The returned raster edge table records this conversion. `outputs.currentDensity` is the maximum of
incoming/outgoing absolute current divided by cell area, including source/sink injection; unrelated
components and failed solves are NaN. Resistance units and this density normalization must be
interpreted with the supplied cell dimensions. Simultaneous sources, alternative conductance
conventions and ensembles remain explicit repeated caller analyses, not least-cost path shortcuts.

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

`modules/graph/test/oracles/urban-ecology.py` records NumPy/SciPy/Shapely/NetworkX composed references.
`urban-ecology-advanced.py` directly executes segregation 2.5.3 for all three counterfactual methods,
Shapley components and all replay assignments of the four-unit paired-label fixture. It pins momepy
0.10.0, NumPy 2.2.6 and Shapely 2.1.2 and records the pandas version; these are direct upstream calls,
not reimplementations of the GPU formulas. `urban-landscape.R` directly executes landscapemetrics
2.2.1 and terra 1.9.50 for patch area/perimeter, class and landscape mesh/cohesion/aggregation (including a single-cell class), both boundary-edge
policies, and joint-adjacency contagion on square-cell fixtures with interior background. Upstream
hectares are converted to square projected units. Rectangular-cell physical edge lengths are tested
independently; this is not blanket upstream metric parity. Circuitscape Julia was not executed locally. The sparse solver now additionally passes six
source/sink comparisons against the pinned Circuitscape upstream `sgNetworkVerify2` golden files:
all seven-node potentials, absolute branch currents and unit-source effective resistances agree
within `1e-4`, alongside true residual and energy checks. These are authentic upstream regression
outputs, not values generated by our SciPy oracle. The pinned tree
`4e0d29d3a7f2ed542a9f2ade782ea554c54b4ae7` identifies Circuitscape.jl 6.1.0, but its legacy
verification configuration declares 3.5.5 and the historical golden-producing binary is unspecified.
No freshly executed Julia-version parity is claimed. The repository provenance manifest and licenses described in `modules/graph/test/oracles/upstream-golden/README.md`
retain source URLs, hashes and unmodified inputs/outputs. Independent SciPy nonuniform-network
solves and exact electrical-network identities cover additional cases. The focused GPU golden
suite passed six cases; a node integrity test verifies the retained bytes and expected values.

The [measured execution record](../../../../dev-docs/perf/urban-ecology-coverage-2026-10-23.json)
separates preparation, compilation, first execution, repeated resident wall time, readback and
scratch memory. Native Apple WebGPU passed sparse resistance at 1,026, 8,194 and 65,538 vertices
and exact proximity at 32², 128² and 256² cells. The largest measured resident medians were about
149.1 ms and 18.4 ms respectively. The sparse network uses parallel two-edge routes with known
one-ohm resistance; it is not a conditioning benchmark or evidence of CPU speedup. Component union
was serial GPU work in that capture. The current recipe retains that path for graphs with at most
256 vertices and 256 edges, and uses bounded parallel unions with verified completion and exact
fallback for larger inputs. The sparse solve still records hundreds of graph stages; broader
topology/patch scaling must preserve the same failure and memory evidence. The component
integration record at
`dev-docs/research/gpu-geospatial-performance-2026-10-23/resistance-components-integration-2026-10-24.md`
tracks separate current measurements and validation in the repository.

The [2026-10-24 execution record](../../../../dev-docs/perf/urban-ecology-complete-2026-10-24.json)
reruns those resistance/proximity curves and adds complete class/landscape summaries at 32²,
128² and 256² cells. Two rectangular classes provide closed-form aggregation, pooled cohesion,
mesh and area checks; the largest absolute certificate error is below `1e-5`. Landscape summary
resident medians were approximately 1.3, 9.2 and 34.4 ms, including seven graph stages and the
certificate copy. These measurements cover two patches, not high patch-count landscapes. The
focused final numerical/lens/reference suites passed 88 GPU tests; node/kernel-capacity audits
passed 35 tests. Final repository-wide gates are recorded separately.
