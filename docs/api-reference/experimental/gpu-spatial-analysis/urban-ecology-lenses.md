# Urban morphology and landscape ecology computation lenses

These executable examples use explicitly synthetic planar geometry, categorical cells and population counts. Each lens runs the actual recipe on a caller-owned GPU command graph and captures its output and status views. They illustrate contracts; they are not observed city data or performance evidence.

## `addStreetProfilesRecipe`

A street sample faces walls two and three units away, with building heights six and nine. The captured canyon width is five and the mean height/width ratio is 1.5; the validity mask records observation on both sides.

{/* lens:addStreetProfilesRecipe */}

::lens{scenario=gpu-spatial-analysis.street-profiles-recipe rung=result caption="Inspect the actual synthetic recipe outputs and status."}

::lens{scenario=gpu-spatial-analysis.street-profiles-recipe rung=graph caption="Follow the executable GPU stages and output capture copies."}

## `addBuildingAdjacencyRecipe`

Two buildings share a wall of length two. Their axial orientations straddle the half-turn boundary and differ by 0.02π radians. A third building has no positive-length shared wall. The completeness flag preserves the supplied shared-path evidence.

{/* lens:addBuildingAdjacencyRecipe */}

::lens{scenario=gpu-spatial-analysis.building-adjacency-recipe rung=result caption="Inspect the actual synthetic recipe outputs and status."}

::lens{scenario=gpu-spatial-analysis.building-adjacency-recipe rung=graph caption="Follow the executable GPU stages and output capture copies."}

## `addIsovistsRecipe`

Eight rays from the centre recover the square enclosure area and perimeter of sixteen. A second observer lies on the boundary and produces invalid measures, so the result also demonstrates the boundary mask.

{/* lens:addIsovistsRecipe */}

::lens{scenario=gpu-spatial-analysis.isovists-recipe rung=result caption="Inspect the actual synthetic recipe outputs and status."}

::lens{scenario=gpu-spatial-analysis.isovists-recipe rung=graph caption="Follow the executable GPU stages and output capture copies."}

## `addVisibilityGraphRecipe`

Three collinear observers form a path graph. The middle observer has betweenness one. An edge capacity of one deliberately truncates the two-edge output; the status reports both the written and required counts while dense adjacency still supports the metrics.

{/* lens:addVisibilityGraphRecipe */}

::lens{scenario=gpu-spatial-analysis.visibility-graph-recipe rung=result caption="Inspect the actual synthetic recipe outputs and status."}

::lens{scenario=gpu-spatial-analysis.visibility-graph-recipe rung=graph caption="Follow the executable GPU stages and output capture copies."}

## `addBuildingTessellationRecipe`

Eight boundary samples from two rectangular buildings are clipped to a rectangular enclosure and dissolved by building. The two complementary cells each have area twelve. Sampling, overlay counts and completeness accompany the variable-length geometry.

{/* lens:addBuildingTessellationRecipe */}

::lens{scenario=gpu-spatial-analysis.building-tessellation-recipe rung=result caption="Inspect the actual synthetic recipe outputs and status."}

::lens{scenario=gpu-spatial-analysis.building-tessellation-recipe rung=graph caption="Follow the executable GPU stages and output capture copies."}

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


## `addSparseResistanceConnectivityRecipe`

Sparse Jacobi-preconditioned resistance with a true-current certificate. This executes a synthetic fixture and preserves status independently from numerical output.

{/* lens:addSparseResistanceConnectivityRecipe */}

::lens{scenario=gpu-spatial-analysis.sparse-resistance-connectivity-recipe rung=result caption="Inspect the actual GPU outputs and completeness evidence."}

::lens{scenario=gpu-spatial-analysis.sparse-resistance-connectivity-recipe rung=graph caption="Follow graph-native construction and execution."}

## `addRasterResistanceConnectivityRecipe`

Raster resistance converted to physical neighbor conductance. This executes a synthetic fixture and preserves status independently from numerical output.

{/* lens:addRasterResistanceConnectivityRecipe */}

::lens{scenario=gpu-spatial-analysis.raster-resistance-connectivity-recipe rung=result caption="Inspect the actual GPU outputs and completeness evidence."}

::lens{scenario=gpu-spatial-analysis.raster-resistance-connectivity-recipe rung=graph caption="Follow graph-native construction and execution."}

## `addSegregationCounterfactualRecipe`

Empirical-quantile population counterfactuals and Shapley components. This executes a synthetic fixture and preserves status independently from numerical output.

{/* lens:addSegregationCounterfactualRecipe */}

::lens{scenario=gpu-spatial-analysis.segregation-counterfactual-recipe rung=result caption="Inspect the actual GPU outputs and completeness evidence."}

::lens{scenario=gpu-spatial-analysis.segregation-counterfactual-recipe rung=graph caption="Follow graph-native construction and execution."}

## `addPopulationSupportAlignmentRecipe`

Explicit population support identifiers reordered with completeness evidence. This executes a synthetic fixture and preserves status independently from numerical output.

{/* lens:addPopulationSupportAlignmentRecipe */}

::lens{scenario=gpu-spatial-analysis.population-support-alignment-recipe rung=result caption="Inspect the actual GPU outputs and completeness evidence."}

::lens{scenario=gpu-spatial-analysis.population-support-alignment-recipe rung=graph caption="Follow graph-native construction and execution."}
