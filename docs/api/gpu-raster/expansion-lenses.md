# Raster operation recordings

<a id="expansion-computation-lenses"></a>

Import from `@luma-contributors/analysis/raster-analysis`.

These executable examples use an explicit synthetic 3 × 3 grid. Each graph runs the named public contributor and records real GPU inputs and outputs. The graph rung exposes procedural stages and completeness evidence.

## Recordings by task

- [Raster morphology and surface preparation](./expansion.md#morphology-and-surface-preparation): reconstruction, holes, skeletons, rank and medial axis.
- [Cost surfaces and travel](./expansion.md#costs-and-travel): directed walking costs.
- [Terrain hydrology](../gpu-terrain/expansion-lenses.md): downstream distance and compact stream reaches.
- [Geometry and coverage](./expansion.md#geometry-and-coverage): fractional coverage and polygon integration.
- [Reprojection](./tile-reprojection.md): resident tile assembly, sampling and validity.
- [Flow diagnostics](#flow-diagnostics-recordings): ridge, particle and streamline recordings, plus separate flow-map/FTLE catalog examples.

Historical capability IDs and class fragments below remain stable.

## GPURasterFillNoData

[Inputs, outputs and limits](./fill-nodata.md).

{/* lens:GPURasterFillNoData */}

Disk search uses original valid donors only; nearest or inverse-distance weighted interpolation preserves observed samples.

::lens{scenario=gpu-raster.raster-fill-no-data rung=graph}

## GPURasterRankFilter

[Inputs, outputs and limits](./expansion.md#morphology-and-surface-preparation).

{/* lens:GPURasterRankFilter */}

Select sorted[min(floor(q*n),n-1)] from the finite square neighborhood; invalid centers remain invalid.

::lens{scenario=gpu-raster.raster-rank-filter rung=graph}

## GPURasterFractionalCoverage

[Inputs, outputs and limits](./expansion.md#geometry-and-coverage).

{/* lens:GPURasterFractionalCoverage */}

Coverage is polygon-cell intersection area divided by affine cell area, assembled from clipped triangles.

::lens{scenario=gpu-raster.raster-fractional-coverage rung=graph}

## GPURasterFractionalZonalStatistics

[Inputs, outputs and limits](./expansion.md#fractional-quantities).

{/* lens:GPURasterFractionalZonalStatistics */}

Extensive totals sum value times fraction; density integrals additionally multiply cell area. Means divide weighted values by observed fractions.

::lens{scenario=gpu-raster.raster-fractional-zonal-statistics rung=graph}

## GPURasterReconstruction

[Inputs, outputs and limits](./expansion.md#morphology-and-surface-preparation).

Reconstruction accepts raster bands through the CPU/GPU operation wrappers; caller-owned geometry
and raster metadata remain explicit.

{/* lens:GPURasterReconstruction */}

Geodesic dilation iterates min(mask, neighborhood maximum) until a verified fixed point; erosion is the dual.

::lens{scenario=gpu-raster.raster-reconstruction rung=graph}

## GPURasterFillHoles

[Inputs, outputs and limits](./expansion.md#morphology-and-surface-preparation).

{/* lens:GPURasterFillHoles */}

Reconstruct boundary-connected background and complement it to fill enclosed background components.

::lens{scenario=gpu-raster.raster-fill-holes rung=graph}

## GPURasterReprojection

[Inputs, outputs and limits](./expansion.md#gpurasterreprojection).

{/* lens:GPURasterReprojection */}

Map destination centers through the inverse projection and source affine, then sample the supplied source support.

::lens{scenario=gpu-raster.raster-reprojection rung=graph}

## GPURasterWalkingCost

[Inputs, outputs and limits](./expansion.md#costs-and-travel).

{/* lens:GPURasterWalkingCost */}

Directed least costs obey D(v)=min(D(v),min_u(D(u)+cost(u,v))); Tobler speed is 6 exp(-3.5 abs(slope+0.05)) km/hour.

::lens{scenario=gpu-raster.raster-walking-cost rung=graph}

## GPURasterWatershedSegmentation

[Inputs, outputs and limits](./expansion.md#watershed-methods).

{/* lens:GPURasterWatershedSegmentation */}

Priority flood selects the smallest path maximum elevation, breaking ties by FIFO discovery age.

::lens{scenario=gpu-raster.raster-watershed-segmentation rung=graph}

## GPURasterSkeleton

[Inputs, outputs and limits](./expansion.md#morphology-and-surface-preparation).

{/* lens:GPURasterSkeleton */}

Zhang-Suen alternates two deletion phases preserving connectivity until no foreground cell changes.

::lens{scenario=gpu-raster.raster-skeleton rung=graph}

## GPURasterMedialAxis

[Inputs, outputs and limits](./expansion.md#exact-medial-disk-containment).

{/* lens:GPURasterMedialAxis */}

Retain Euclidean foreground disks that are not contained in another foreground disk, including exterior background.

::lens{scenario=gpu-raster.raster-medial-axis rung=graph}


## `GPURasterTileReprojection`

[Inputs, outputs and limits](./tile-reprojection.md).

Inspect explicit bounded source mosaic assembly and bilinear sampling. Source tiles share a lattice; the caller supplies all tiles and interpolation support.

{/* lens:GPURasterTileReprojection */}

::lens{scenario=gpu-raster.raster-tile-reprojection rung=result caption="Inspect synthetic outputs and validity."}

::lens{scenario=gpu-raster.raster-tile-reprojection rung=graph caption="Inspect the actual GPU command graph."}

## Flow diagnostics recordings

Recorded GPU examples:

- [Scalar-ridge recordings](./flow-map-and-particle-diagnostics.md#gpuscalarfieldridges): result, formula, datum, kernel and graph captures.
- [Particle-advection recordings](./operations-analysis.md#gpuparticleadvection): persistent particle positions and trails.
- [Streamline recordings](./operations-analysis.md#gpustreamlines): traced lines, pruning and compact publication.

Separate catalog examples: [Lagrangian flow-map run](/fn/lagrangian-flow-map) and
[FTLE run](/fn/lagrangian-finite-time-lyapunov-exponent). These catalog examples are not dedicated
recording directives on this guide. No dedicated connectivity or residence-time recording is
available here; use the [connectivity contract](./flow-map-and-particle-diagnostics.md#connectivity-matrix)
and [residence-time contract](./flow-map-and-particle-diagnostics.md#gpulagrangianflowmap).
