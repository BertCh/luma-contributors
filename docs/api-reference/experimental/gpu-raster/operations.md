---
title: Raster analysis operations
description: Index of the raster analysis contributors in @luma-contributors/analysis/raster-analysis.
---

# Raster analysis reference

`@luma-contributors/analysis/raster-analysis` holds the raster analysis contributors that luma.gl
does not provide. Raster storage and residency, analytical overviews, pixel operations, statistics,
filters and morphology, connected components, region measurements and contours are owned by
luma.gl: see [`@luma.gl/graph/gpu-raster`](https://luma.gl/docs/api-reference/experimental/gpu-raster).
This entry composes with those operations through `GPUCommandGraph`; it does not wrap or replace them.

Each contributor declares work into a caller-owned graph and never compiles, submits or reads back.

## Choose a page

| Page | Use it for |
| --- | --- |
| [Analysis contributors](/docs/api-reference/experimental/gpu-raster/operations-analysis) | Zonal statistics, stretches, isolines, distance and cost surfaces, rasterization, change detection, and flow textures. |
| [Patch metrics and sieving](/docs/api-reference/experimental/gpu-raster/raster-patches) | Per-patch area and perimeter, and small-patch removal over dense labels. |
| [Raster, terrain and geometry expansion](/docs/api-reference/experimental/gpu-raster/expansion) | Continuous-surface preparation, reprojection, directed walking cost and bounded topology. |
| [Fill nodata](/docs/api-reference/experimental/gpu-raster/fill-nodata) | Four-quadrant nodata filling. |
| [Tile reprojection](/docs/api-reference/experimental/gpu-raster/tile-reprojection) | Explicit tile reprojection. |
| [Fractional zonal distribution](/docs/api-reference/experimental/gpu-raster/fractional-zonal-distribution) and [categorical statistics](/docs/api-reference/experimental/gpu-raster/fractional-categorical-statistics) | Weighted distributions and composition with observed support. |
| [Focal pairs](/docs/api-reference/experimental/gpu-raster/focal-pairs) | Paired-valid covariance, correlation and local regression. |
| [Spectral indices](/docs/api-reference/experimental/gpu-raster/spectral-indices) | Calibrated NDWI, NBR, EVI and SAVI. |
| [Remote sensing](/docs/api-reference/experimental/gpu-raster/remote-sensing) | Remote-sensing contributors. |

## Related pages

- [Raster analysis overview](/docs/api-reference/experimental/gpu-raster)
- [GPU Terrain](/docs/api-reference/experimental/gpu-terrain)
- [luma.gl GPU Core](https://luma.gl/docs/api-reference/experimental/gpu-core)
