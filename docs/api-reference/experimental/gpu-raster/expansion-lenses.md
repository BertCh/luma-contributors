# Expansion computation lenses

These executable examples use an explicit synthetic 3 × 3 grid. Each graph runs the named public contributor and records real GPU inputs and outputs. The graph rung exposes procedural stages and completeness evidence.

## GPURasterFillNoData

{/* lens:GPURasterFillNoData */}

Disk search uses original valid donors only; nearest or inverse-distance weighted interpolation preserves observed samples.

::lens{scenario=gpu-raster.raster-fill-no-data rung=graph}

## GPURasterRankFilter

{/* lens:GPURasterRankFilter */}

Select sorted[min(floor(q*n),n-1)] from the finite square neighborhood; invalid centers remain invalid.

::lens{scenario=gpu-raster.raster-rank-filter rung=graph}

## GPURasterFractionalCoverage

{/* lens:GPURasterFractionalCoverage */}

Coverage is polygon-cell intersection area divided by affine cell area, assembled from clipped triangles.

::lens{scenario=gpu-raster.raster-fractional-coverage rung=graph}

## GPURasterFractionalZonalStatistics

{/* lens:GPURasterFractionalZonalStatistics */}

Extensive totals sum value times fraction; density integrals additionally multiply cell area. Means divide weighted values by observed fractions.

::lens{scenario=gpu-raster.raster-fractional-zonal-statistics rung=graph}

## GPURasterReconstruction

{/* lens:GPURasterReconstruction */}

Geodesic dilation iterates min(mask, neighborhood maximum) until a verified fixed point; erosion is the dual.

::lens{scenario=gpu-raster.raster-reconstruction rung=graph}

## GPURasterFillHoles

{/* lens:GPURasterFillHoles */}

Reconstruct boundary-connected background and complement it to fill enclosed background components.

::lens{scenario=gpu-raster.raster-fill-holes rung=graph}

## GPURasterReprojection

{/* lens:GPURasterReprojection */}

Map destination centers through the inverse projection and source affine, then sample the supplied source support.

::lens{scenario=gpu-raster.raster-reprojection rung=graph}

## GPURasterWalkingCost

{/* lens:GPURasterWalkingCost */}

Directed least costs obey D(v)=min(D(v),min_u(D(u)+cost(u,v))); Tobler speed is 6 exp(-3.5 abs(slope+0.05)) km/hour.

::lens{scenario=gpu-raster.raster-walking-cost rung=graph}

## GPURasterWatershedSegmentation

{/* lens:GPURasterWatershedSegmentation */}

Priority flood selects the smallest path maximum elevation, breaking ties by FIFO discovery age.

::lens{scenario=gpu-raster.raster-watershed-segmentation rung=graph}

## GPURasterSkeleton

{/* lens:GPURasterSkeleton */}

Zhang-Suen alternates two deletion phases preserving connectivity until no foreground cell changes.

::lens{scenario=gpu-raster.raster-skeleton rung=graph}

## GPURasterMedialAxis

{/* lens:GPURasterMedialAxis */}

Retain Euclidean foreground disks that are not contained in another foreground disk, including exterior background.

::lens{scenario=gpu-raster.raster-medial-axis rung=graph}


## `GPURasterTileReprojection`

Inspect explicit bounded source mosaic assembly and bilinear sampling. Source tiles share a lattice; the caller supplies all tiles and interpolation support.

{/* lens:GPURasterTileReprojection */}

::lens{scenario=gpu-raster.raster-tile-reprojection rung=result caption="Inspect synthetic outputs and validity."}

::lens{scenario=gpu-raster.raster-tile-reprojection rung=graph caption="Inspect the actual GPU command graph."}
