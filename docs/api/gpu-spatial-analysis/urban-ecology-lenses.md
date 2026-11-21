# Urban morphology and landscape ecology computation lenses

These executable examples use explicitly synthetic planar geometry, categorical cells and population counts. Each lens runs the actual recipe on a caller-owned GPU command graph and captures its output and status views. They illustrate contracts; they are not observed city data or performance evidence.

## `addStreetProfilesRecipe`

A street sample faces walls two and three units away, with building heights six and nine. The captured canyon width is five and the mean height/width ratio is 1.5; the validity mask records observation on both sides.

{/* lens:addStreetProfilesRecipe */}

::lens{scenario=gpu-spatial-analysis.street-profiles-recipe rung=result caption="Inspect the actual synthetic recipe outputs and status."}

::lens{scenario=gpu-spatial-analysis.street-profiles-recipe rung=graph caption="Follow the executable GPU stages and output capture copies."}

## `addBuildingAdjacencyRecipe`

Two buildings share a wall of length two. Their axial orientations straddle the half-turn boundary and differ by 0.02π radians. The docs-sized fixture contains only this adjacent pair; their dissolved structure has exterior perimeter eight. The pair table, component labels, morphology and completeness evidence are all captured.

{/* lens:addBuildingAdjacencyRecipe */}

::lens{scenario=gpu-spatial-analysis.building-adjacency-recipe rung=result caption="Inspect the actual synthetic recipe outputs and status."}

::lens{scenario=gpu-spatial-analysis.building-adjacency-recipe rung=graph caption="Follow the executable GPU stages and output capture copies."}

## `addSparseResistanceConnectivityRecipe`

A 3 × 3 lattice of unit conductances with one weak diagonal chord carries unit current from one corner to the opposite corner. The sparse grounded-Laplacian Jacobi-PCG solve returns the potentials, the edge currents and the Circuitscape-style node currents (one at the source and the sink); the status and the independently recomputed true residual gate the effective resistance.

{/* lens:addSparseResistanceConnectivityRecipe */}

::lens{scenario=gpu-spatial-analysis.sparse-resistance-connectivity-recipe rung=result caption="Inspect the actual synthetic recipe outputs and status."}

::lens{scenario=gpu-spatial-analysis.sparse-resistance-connectivity-recipe rung=graph caption="Follow the sort, CSR assembly and the unrolled conjugate gradient stages."}

## `addRasterResistanceConnectivityRecipe`

A 6 × 5 resistance raster has a no-data wall in its fourth column with a one-cell gap, and a high-resistance cell beside the gap. Eight-neighbour conductances `2 / (R(a) + R(b)) / edgeLength` are built on the GPU, and unit current flows from a cell on the west edge to a cell on the east edge: the whole current crosses the wall through the gap (its node current is one), and the wall cells have no potential.

Uniform mode uses the full affine for cardinal and diagonal distances, including rotation and
shear. For per-frame geographic or Web Mercator updates, call
`getRasterResistanceConnectivityParameterValues(grid, target, cellSizeMode)` with the recipe's
fixed mode: these two modes require an axis-aligned grid on every frame. Existing one- and
two-argument packer calls keep uniform shear support.

{/* lens:addRasterResistanceConnectivityRecipe */}

::lens{scenario=gpu-spatial-analysis.raster-resistance-connectivity-recipe rung=result caption="Inspect the potentials, the current map and the edge conductances."}

::lens{scenario=gpu-spatial-analysis.raster-resistance-connectivity-recipe rung=graph caption="Follow the edge builder, the invalid-cell count and the sparse solve."}

## `addIsovistsRecipe`

Eight rays from the centre recover the square enclosure area and perimeter of sixteen. A second observer lies on the boundary and produces invalid measures, so the result also demonstrates the boundary mask.

{/* lens:addIsovistsRecipe */}

::lens{scenario=gpu-spatial-analysis.isovists-recipe rung=result caption="Inspect the actual synthetic recipe outputs and status."}

::lens{scenario=gpu-spatial-analysis.isovists-recipe rung=graph caption="Follow the executable GPU stages and output capture copies."}

## `addVisibilityGraphRecipe`

The `gpu-network` recipe of the same name supports budgeted exact CSR all-source metrics or deterministic sampled estimates via
metricsMode auto/exact. Auto samples when the exact command budget cannot fit; sampled maximum depth
is a lower bound. Approximation, overflow and convergence are separate statuses; explicit exact
rejects an insufficient budget.

Three collinear observers form a path graph. The middle observer has betweenness one. An edge capacity of one deliberately truncates the two-edge output; the status reports both the written and required counts while dense adjacency still supports the metrics.

This scenario uses the dense reference mode. In CSR mode this spatial recipe provides
adjacency and degree only; [the gpu-network visibility recipe](../gpu-network/visibility-graph.md)
adds pivot betweenness, closeness and reached counts. As in the dense mode, an invalid viewpoint
(on or within tolerance of an obstacle, or non-finite) gets an empty CSR row and appears in no
other row; the rest of the graph is still published and `status.invalidCount` counts the invalid
viewpoints. An invalid `parameters` view invalidates every viewpoint, so the graph is empty.

{/* lens:addVisibilityGraphRecipe */}

::lens{scenario=gpu-spatial-analysis.visibility-graph-recipe rung=result caption="Inspect the actual synthetic recipe outputs and status."}

::lens{scenario=gpu-spatial-analysis.visibility-graph-recipe rung=graph caption="Follow the executable GPU stages and output capture copies."}

## `addBuildingTessellationRecipe`

The dispatcher selects reference mode explicitly with `mode: 'reference'`, local mode explicitly
with `mode: 'local'`, and otherwise selects local when `candidateRadius` is present. The named
[reference](#addbuildingtessellationreferencerecipe) and [local](#addlocalbuildingtessellationrecipe)
entries below expose the route-specific result types and evidence.

The existing `gpu-spatial-analysis.building-tessellation-recipe` scenario is registered under this
legacy name and calls `addBuildingTessellationReferenceRecipe` directly. It demonstrates the
reference route described below.

## `addBuildingTessellationReferenceRecipe`

Eight boundary samples from two rectangular buildings are clipped to a rectangular enclosure and
dissolved by building. The complementary cells each have area twelve. This is reference-route
evidence: `outputs[i]` is a dissolved polygon with scalar `status.count`, `overflow`,
`candidateOverflow` and `requiredRingCount`. Recipe-level `status` contains the packed `sampling`
row and per-building `complete` view. Read both the per-output capacity/loss words and
`status.complete[i]` before consuming geometry.

For the four rectangular fragments per building, the scenario uses capacities of 64 intersections,
32 noded segments, 16 output vertices and four rings per dissolve. These are fixture capacities;
the route's limits are 64 buildings, 128 total samples and 32 total enclosure vertices per call.
See the [reference output and dissolve contract](urban-morphology.md#addbuildingtessellationreferencerecipe).

{/* lens:addBuildingTessellationRecipe */}

{/* lens:addBuildingTessellationReferenceRecipe */}

::lens{scenario=gpu-spatial-analysis.building-tessellation-recipe rung=result caption="Inspect the reference route's dissolved polygons, sampling and completeness."}

::lens{scenario=gpu-spatial-analysis.building-tessellation-recipe rung=graph caption="Follow reference sampling, clipping and per-building dissolve stages."}

## `addLocalBuildingTessellationRecipe`

The registered `gpu-spatial-analysis.local-building-tessellation-recipe` scenario calls the named
local entry. Four boundary samples from one square building partition a square enclosure of area
sixteen. It uses `candidateRadius: 16`, four own-sample and candidate slots, and sixteen vertices
per piece. These fixture settings exercise the union of sample cells, with no per-building dissolve.

Local output uses fixed `positions`, `cells` and `buildingStatus` slots. In general, a simple
nonconvex or holed enclosure is triangulated and each building owns a union of convex pieces.
Validity is per building; the scalar `count` counts usable buildings, not pieces. `overflow`
reports vertex capacity loss, while `candidateOverflow` reports sampling, search-capacity,
index-visit or radius-certification loss. A failed building's piece vertex counts are zero.
Follow the [candidate-radius certificate, per-building validity and separate loss words](urban-morphology.md#local-certification-and-loss-evidence)
before treating the output as complete. Local limits and the fixed row layout are in the
[local route contract](urban-morphology.md#addlocalbuildingtessellationrecipe).

{/* lens:addLocalBuildingTessellationRecipe */}

::lens{scenario=gpu-spatial-analysis.local-building-tessellation-recipe rung=result caption="Inspect the local convex-piece union and per-building certification status."}

::lens{scenario=gpu-spatial-analysis.local-building-tessellation-recipe rung=graph caption="Follow local sampling, indexed candidates and certified cell clipping."}

## `addLandscapeMetricsRecipe`

Two 2×2 patches fill a 4×2 raster with two classes. Each class has area four, perimeter eight and aggregation one hundred percent. Class adjacency and the complete-domain landscape statistics are captured separately.

{/* lens:addLandscapeMetricsRecipe */}

::lens{scenario=gpu-spatial-analysis.landscape-metrics-recipe rung=result caption="Inspect the actual synthetic recipe outputs and status."}

::lens{scenario=gpu-spatial-analysis.landscape-metrics-recipe rung=graph caption="Follow the executable GPU stages and output capture copies."}

## `addPatchProximityRecipe`

Two two-cell patches are separated by one background cell. With edge depth 0.4 and search radius two, each has core area two, nearest boundary distance one and proximity two. Background remains excluded from the core mask.

{/* lens:addPatchProximityRecipe */}

::lens{scenario=gpu-spatial-analysis.patch-proximity-recipe rung=result caption="Inspect the actual synthetic recipe outputs and status."}

::lens{scenario=gpu-spatial-analysis.patch-proximity-recipe rung=graph caption="Follow the executable GPU stages and output capture copies."}

## `addResistanceConnectivityRecipe`

A three-node network carries unit current between the first and last nodes. Potentials are one, one-half and zero; effective resistance is one. The status and independently recomputed current residual gate interpretation of the solver result.

{/* lens:addResistanceConnectivityRecipe */}

::lens{scenario=gpu-spatial-analysis.resistance-connectivity-recipe rung=result caption="Inspect the actual synthetic recipe outputs and status."}

::lens{scenario=gpu-spatial-analysis.resistance-connectivity-recipe rung=graph caption="Follow the executable GPU stages and output capture copies."}

## `addSegregationDecompositionRecipe`

Four supplied population tables share the same two zones. Their dissimilarities are 0.8, 0.6, 0.4 and zero. The two-factor decomposition attributes −0.3 to composition and −0.5 to allocation, with zero reconstruction residual; no counterfactual is inferred by the recipe.

{/* lens:addSegregationDecompositionRecipe */}

::lens{scenario=gpu-spatial-analysis.segregation-decomposition-recipe rung=result caption="Inspect the actual synthetic recipe outputs and status."}

::lens{scenario=gpu-spatial-analysis.segregation-decomposition-recipe rung=graph caption="Follow the executable GPU stages and output capture copies."}


## `addSegregationCounterfactualRecipe`

Empirical-quantile population counterfactuals and Shapley components. This executes a synthetic fixture and preserves status independently from numerical output.

{/* lens:addSegregationCounterfactualRecipe */}

::lens{scenario=gpu-spatial-analysis.segregation-counterfactual-recipe rung=result caption="Inspect the actual GPU outputs and completeness evidence."}

::lens{scenario=gpu-spatial-analysis.segregation-counterfactual-recipe rung=graph caption="Follow graph-native construction and execution."}

## `addPopulationSupportAlignmentRecipe`

The recipe reorders populations by explicit support ID. In `status.stages`, the `alignment` stage
reports `complete` and the `missing-targets`, `ambiguous-targets`, and `excluded-sources` stages
each report `invalidCount`.

{/* lens:addPopulationSupportAlignmentRecipe */}

::lens{scenario=gpu-spatial-analysis.population-support-alignment-recipe rung=result caption="Inspect the actual GPU outputs and completeness evidence."}

::lens{scenario=gpu-spatial-analysis.population-support-alignment-recipe rung=graph caption="Follow graph-native construction and execution."}
