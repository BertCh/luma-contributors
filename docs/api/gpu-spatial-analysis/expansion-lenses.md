# Geospatial expansion computation lenses

These executable examples use small, explicitly synthetic inputs to inspect each contributor’s actual GPU command graph and captured outputs. They illustrate API contracts and status behavior; they are not performance benchmarks or observed geographic datasets. Statistical definitions and bounded algorithm contracts are documented in [Statistical expansion](./statistical-expansion.md), [Expansion recipes](./expansion-recipes.md), and the raster and network expansion references.

## `GPULocalSpatialHeteroscedasticity`

Scalable rows are bounded by 4,294,963,199 and actual device bindings. Additive uint32 count outputs
preserve exact populations; float32 statistics remain rounded.

LOSH divides a weighted neighborhood residual moment by the study-wide residual moment; the graph exposes normalization and invalid-row status.

{/* lens:GPULocalSpatialHeteroscedasticity */}

::lens{scenario=gpu-spatial-analysis.local-spatial-heteroscedasticity rung=result caption="Inspect GPU output rows for the explicit synthetic example."}

::lens{scenario=gpu-spatial-analysis.local-spatial-heteroscedasticity rung=graph caption="Follow the actual command graph, including intermediate stages and status outputs."}

## `GPUMultivariateLocalGeary`

Scalable rows are bounded by 4,294,963,199 and actual device bindings. Additive uint32 count outputs
preserve exact populations; float32 statistics remain rounded. Descriptive reductions default to
tiled; mode reference retains its previous limit. Permutation inference retains its own ceiling.

Complete-case, variance-standardized multivariate squared differences are averaged over each weighted neighborhood.

{/* lens:GPUMultivariateLocalGeary */}

::lens{scenario=gpu-spatial-analysis.multivariate-local-geary rung=result caption="Inspect GPU output rows for the explicit synthetic example."}

::lens{scenario=gpu-spatial-analysis.multivariate-local-geary rung=graph caption="Follow the actual command graph, including intermediate stages and status outputs."}

## `GPUGeneralizedGeographicallyWeightedRegression`

This synthetic Poisson fit uses adaptive bandwidthMode with six neighbors and a six-entry selection capacity. Family-specific local IRLS exposes convergence status; see [Adaptive generalized GWR](./adaptive-generalized-gwr.md).

{/* lens:GPUGeneralizedGeographicallyWeightedRegression */}

::lens{scenario=gpu-spatial-analysis.generalized-geographically-weighted-regression rung=result caption="Inspect GPU output rows for the explicit synthetic example."}

::lens{scenario=gpu-spatial-analysis.generalized-geographically-weighted-regression rung=graph caption="Follow the actual command graph, including intermediate stages and status outputs."}

## `GPUMultiscaleGeographicallyWeightedRegression`

Additive multiscale backfitting alternates coefficient updates with explicitly selected bandwidths and bounded convergence checks.

{/* lens:GPUMultiscaleGeographicallyWeightedRegression */}

::lens{scenario=gpu-spatial-analysis.multiscale-geographically-weighted-regression rung=result caption="Inspect GPU output rows for the explicit synthetic example."}

::lens{scenario=gpu-spatial-analysis.multiscale-geographically-weighted-regression rung=graph caption="Follow the actual command graph, including intermediate stages and status outputs."}

## `GPUSpatialPanelRegression`

This balanced synthetic panel uses effects: unit, exposing within transformation before spatial-lag two-stage least squares. See [Unit fixed effects](./panel-fixed-effects.md) for absorbed intercepts and diagnostics.

{/* lens:GPUSpatialPanelRegression */}

::lens{scenario=gpu-spatial-analysis.spatial-panel-regression rung=result caption="Inspect GPU output rows for the explicit synthetic example."}

::lens{scenario=gpu-spatial-analysis.spatial-panel-regression rung=graph caption="Follow the actual command graph, including intermediate stages and status outputs."}

## `GPUSpatialRegimeRegression`

Separate fully varying ordinary least-squares models are solved per caller-defined regime, retaining empty and singular group status.

{/* lens:GPUSpatialRegimeRegression */}

::lens{scenario=gpu-spatial-analysis.spatial-regime-regression rung=result caption="Inspect GPU output rows for the explicit synthetic example."}

::lens{scenario=gpu-spatial-analysis.spatial-regime-regression rung=graph caption="Follow the actual command graph, including intermediate stages and status outputs."}

## `GPUConcaveHull`

Triangle reservation is bounded by 715,827,882 to keep six boundary words per triangle within uint32
addressing; device bindings impose lower limits.

An alpha complex selects Delaunay triangles by maximum circumradius and derives the exposed boundary edges; graph and validity are the appropriate structural explanation.

{/* lens:GPUConcaveHull */}

::lens{scenario=gpu-spatial-analysis.concave-hull rung=result caption="Inspect GPU output rows for the explicit synthetic example."}

::lens{scenario=gpu-spatial-analysis.concave-hull rung=graph caption="Follow the actual command graph, including intermediate stages and status outputs."}

## `GPUGroupConcaveHull`

Independent alpha complexes preserve original point IDs and empty group slots; boundary and triangle capacities remain explicit.

{/* lens:GPUGroupConcaveHull */}

::lens{scenario=gpu-spatial-analysis.group-concave-hull rung=result caption="Inspect GPU output rows for the explicit synthetic example."}

::lens{scenario=gpu-spatial-analysis.group-concave-hull rung=graph caption="Follow the actual command graph, including intermediate stages and status outputs."}

## `GPUConstrainedDelaunayTriangulation`

flipMethod auto selects parallel candidate discovery and adjacency repair when the polygon and
command budget fit; serial retains reference flip ordering. Serial maximumFlips is 4,294,967,294;
parallel polygon and triangle counts are bounded by 536,870,911 and 1,431,655,765 respectively, with
device bindings imposing lower limits.

Polygon triangulation preserves ring constraints and performs legal Lawson edge flips, retaining validity and convergence diagnostics.

{/* lens:GPUConstrainedDelaunayTriangulation */}

::lens{scenario=gpu-spatial-analysis.constrained-delaunay-triangulation rung=result caption="Inspect GPU output rows for the explicit synthetic example."}

::lens{scenario=gpu-spatial-analysis.constrained-delaunay-triangulation rung=graph caption="Follow the actual command graph, including intermediate stages and status outputs."}

## `GPUH3CellIdentities`

Integer H3 topology derives six canonical directed-edge or vertex keys, including deleted pentagon slots; numerical interpolation is not involved.

{/* lens:GPUH3CellIdentities */}

::lens{scenario=gpu-spatial-analysis.h3-cell-identities rung=result caption="Inspect GPU output rows for the explicit synthetic example."}

::lens{scenario=gpu-spatial-analysis.h3-cell-identities rung=graph caption="Follow the actual command graph, including intermediate stages and status outputs."}

## `GPUBoundedFacilityAllocation`

Bounded facility-subset enumeration separates exact p-center certification from demand-order greedy capacitated allocation.

{/* lens:GPUBoundedFacilityAllocation */}

::lens{scenario=gpu-spatial-analysis.bounded-facility-allocation rung=result caption="Inspect GPU output rows for the explicit synthetic example."}

::lens{scenario=gpu-spatial-analysis.bounded-facility-allocation rung=graph caption="Follow the actual command graph, including intermediate stages and status outputs."}

## `addZonalDistributionRecipe`

Composes finite-value filtering, dense zone grouping, ordered quantiles and deterministic modes; a graph explains the separate reductions.

{/* lens:addZonalDistributionRecipe */}

::lens{scenario=gpu-spatial-analysis.zonal-distribution-recipe rung=result caption="Inspect GPU output rows for the explicit synthetic example."}

::lens{scenario=gpu-spatial-analysis.zonal-distribution-recipe rung=graph caption="Follow the actual command graph, including intermediate stages and status outputs."}

## `addRasterZonalDistributionRecipe`

Native nodata is filtered before scale/offset calibration, then the zonal distribution stages run on calibrated values.

{/* lens:addRasterZonalDistributionRecipe */}

::lens{scenario=gpu-spatial-analysis.raster-zonal-distribution-recipe rung=result caption="Inspect GPU output rows for the explicit synthetic example."}

::lens{scenario=gpu-spatial-analysis.raster-zonal-distribution-recipe rung=graph caption="Follow the actual command graph, including intermediate stages and status outputs."}

## `addExposureByZoneRecipe`

Exposure separates total population, hazard-observed population and threshold-exposed population, with explicit missing-support diagnostics.

{/* lens:addExposureByZoneRecipe */}

::lens{scenario=gpu-spatial-analysis.exposure-by-zone-recipe rung=result caption="Inspect GPU output rows for the explicit synthetic example."}

::lens{scenario=gpu-spatial-analysis.exposure-by-zone-recipe rung=graph caption="Follow the actual command graph, including intermediate stages and status outputs."}

## `addRegionalRasterChangeRecipe`

Aligned slice change detection feeds zone summaries on common support; cell areas, missing support and threshold validity remain separate outputs.

{/* lens:addRegionalRasterChangeRecipe */}

::lens{scenario=gpu-spatial-analysis.regional-raster-change-recipe rung=result caption="Inspect GPU output rows for the explicit synthetic example."}

::lens{scenario=gpu-spatial-analysis.regional-raster-change-recipe rung=graph caption="Follow the actual command graph, including intermediate stages and status outputs."}

## `addSurfacePreparationRecipe`

Bounded filling and optional rank filtering preserve original support provenance and report filled, filtered and unresolved cells separately.

{/* lens:addSurfacePreparationRecipe */}

::lens{scenario=gpu-spatial-analysis.surface-preparation-recipe rung=result caption="Inspect GPU output rows for the explicit synthetic example."}

::lens{scenario=gpu-spatial-analysis.surface-preparation-recipe rung=graph caption="Follow the actual command graph, including intermediate stages and status outputs."}


## `addInterpolationDiagnosticsRecipe`

Leave-one-out kriging produces residuals and variance-aware validation summaries with explicit invalid and standardized-count diagnostics.

{/* lens:addInterpolationDiagnosticsRecipe */}

::lens{scenario=gpu-spatial-analysis.interpolation-diagnostics-recipe rung=result caption="Inspect GPU output rows for the explicit synthetic example."}

::lens{scenario=gpu-spatial-analysis.interpolation-diagnostics-recipe rung=graph caption="Follow the actual command graph, including intermediate stages and status outputs."}

## `addDrainageNetworkRecipe`

D8 routing, stream extraction, downstream distance and reach assembly form a multi-stage drainage graph with convergence and capacity status.

{/* lens:addDrainageNetworkRecipe */}

::lens{scenario=gpu-spatial-analysis.drainage-network-recipe rung=result caption="Inspect GPU output rows for the explicit synthetic example."}

::lens{scenario=gpu-spatial-analysis.drainage-network-recipe rung=graph caption="Follow the actual command graph, including intermediate stages and status outputs."}

## `addLocalVariabilityRecipe`

Shared weights feed descriptive LOSH and analytic local Moran inference, preserving separate inference families and row validity.

{/* lens:addLocalVariabilityRecipe */}

::lens{scenario=gpu-spatial-analysis.local-variability-recipe rung=result caption="Inspect GPU output rows for the explicit synthetic example."}

::lens{scenario=gpu-spatial-analysis.local-variability-recipe rung=graph caption="Follow the actual command graph, including intermediate stages and status outputs."}

## `addCellFlowTopologyRecipe`

Original OD pairs feed conserved cell totals and a separate immediate-neighbor edge aggregation with exact H3 identities and boundaries.

{/* lens:addCellFlowTopologyRecipe */}

::lens{scenario=gpu-spatial-analysis.cell-flow-topology-recipe rung=result caption="Inspect GPU output rows for the explicit synthetic example."}

::lens{scenario=gpu-spatial-analysis.cell-flow-topology-recipe rung=graph caption="Follow the actual command graph, including intermediate stages and status outputs."}


## `GPULOSHResidualPermutationTest`

Inspect conditional randomization of a fixed LOSH normalized residual field with an explicit greater tail. This does not refit residuals or implement esda LOSH inference.

{/* lens:GPULOSHResidualPermutationTest */}

::lens{scenario=gpu-spatial-analysis.losh-residual-permutation-test rung=result caption="Inspect synthetic outputs and validity."}

::lens{scenario=gpu-spatial-analysis.losh-residual-permutation-test rung=graph caption="Inspect the actual GPU command graph."}
