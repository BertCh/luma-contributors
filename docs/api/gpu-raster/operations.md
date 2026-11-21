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

## Choose a neighborhood statistic

```ts
import {GPUFocalStatistics} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {GPURasterFocalPairs, GPURasterRankFilter} from '@luma-contributors/analysis/raster-analysis';
```

| Task and import above | Missing center | Physical edge and support |
| --- | --- | --- |
| Moving mean, sum, extrema or population standard deviation: `GPUFocalStatistics` | Can fill a missing center from valid neighbors; center propagation is a per-frame option | Clips the window and skips invalid neighbors |
| Covariance, correlation or regression: `GPURasterFocalPairs` | Can fill by default; `propagateCenterNoData: true` requires both center values | Clips by default (`borderMode: 'nodata'`, `noDataPolicy: 'ignore'`); only jointly valid pairs participate |
| Median or rank: `GPURasterRankFilter` | Preserves a missing center even with `noDataPolicy: 'ignore'` | Clamps by default, repeating physical-edge samples |

For predictor `[1,NaN,3]`, radius 1 and omitted invalid neighbors, the center moving mean is 2;
the center rank stays invalid. Paired with response `[2,NaN,6]`, the two surviving pairs give
sample covariance 4 and regression slope 2 (center propagation disabled). A paired statistic
is not a moving mean of a single band.

`GPUFocalStatistics.maximumRadius` is a construction-time pixel radius in 0–64; its per-frame
`radius` is clamped to that maximum. `requiredHalo` is the maximum, even when today's radius is
smaller. Focal pairs and rank filters instead fix `radius` at construction in 0–8 and advertise
that radius as `requiredHalo`. Expand every input band and mask by the advertised halo,
execute on the expanded tile, and crop to the core. Apply border policies only at the physical
domain edge; clamping at internal tile boundaries changes the statistic.
See [spatial focal parameters](../gpu-spatial-analysis/operators-statistics.md#gpufocalstatistics),
[paired statistics](./focal-pairs.md), and [rank filtering](./expansion.md#operations-and-contracts).

## Choose a page

| Page | Use it for |
| --- | --- |
| [Analysis contributors](/docs/api/gpu-raster/operations-analysis) | Zonal statistics, stretches, isolines, distance and cost surfaces, rasterization, change detection, and flow textures. |
| [Patch metrics and sieving](/docs/api/gpu-raster/raster-patches) | Per-patch area and perimeter, and small-patch removal over dense labels. |
| [Raster, terrain and geometry expansion](/docs/api/gpu-raster/expansion) | Continuous-surface preparation, reprojection, directed walking cost and bounded topology. |
| [Fill nodata](/docs/api/gpu-raster/fill-nodata) | Four-quadrant nodata filling. |
| [Tile reprojection](/docs/api/gpu-raster/tile-reprojection) | Explicit tile reprojection. |
| [Fractional zonal distribution](/docs/api/gpu-raster/fractional-zonal-distribution) and [categorical statistics](/docs/api/gpu-raster/fractional-categorical-statistics) | Weighted distributions and composition with observed support. |
| [Focal pairs](/docs/api/gpu-raster/focal-pairs) | Paired-valid covariance, correlation and local regression. |
| [Spectral indices](/docs/api/gpu-raster/spectral-indices) | Calibrated NDWI, NBR, EVI and SAVI. |
| [Remote sensing](/docs/api/gpu-raster/remote-sensing) | Remote-sensing contributors. |

## Choose a task

### Raster analysis: Start and choose

- [Raster analysis operations](./operations.md)

- [GPURaster analysis contributors](./operations-analysis.md)

### Raster analysis: Prepare surfaces and geometry

- [Raster, terrain and geometry expansion](./expansion.md)

- [Raster contrast equalization](./contrast-equalization.md)

- [Nodata filling](./fill-nodata.md)

- [GPURasterTileReprojection](./tile-reprojection.md)

- [Raster operation recordings](./expansion-lenses.md)

### Raster analysis: Summarize zones and local structure

- [GPURasterFractionalZonalDistribution](./fractional-zonal-distribution.md)

- [GPURasterFractionalCategoricalStatistics](./fractional-categorical-statistics.md)

- [Focal cross-band statistics](./focal-pairs.md)

- [Raster patch metrics and sieving](./raster-patches.md)

- [GPUIsolines: seam-safe tiles and vector overlays](./isolines-tiles.md)

- [Spectral raster filtering](./spectral-filter.md)

### Raster analysis: Compare spectra and embeddings

- [Calibrated spectral indices](./spectral-indices.md)

- [Embedding raster codecs](./raster-embedding-codecs.md)

- [Embedding rasters](./raster-embeddings.md)

- [Bounded remote-sensing contributors](./remote-sensing.md)

### Raster analysis: Analyze time and transport

- [Raster time series](./raster-time-series.md)

- [Raster change monitoring](./raster-change-monitoring.md)

- [Raster EOF analysis and lagged correlation](./raster-eof-analysis.md)

- [Flow-map and particle diagnostics](./flow-map-and-particle-diagnostics.md)

## Related pages

- [Raster analysis overview](/docs/api/gpu-raster)
- [GPU Terrain](/docs/api/gpu-terrain)
- [luma.gl GPU Core](https://luma.gl/docs/api-reference/experimental/gpu-core)

## Geographic area models

Whole-cell `GPURasterZonalStatistics` geographic outputs use the authalic-sphere latitude-strip
formula `R² |Δlongitude (sin(latitudeTop) - sin(latitudeBottom))|`, with
`R = 6371007.180918475` meters. `sums` multiply values by these square-meter weights;
`weightSums` report their sum, and means and variances use those weights. Counts remain cell counts.

Grid-derived fractional area uses WGS84 ellipsoidal meters per degree at the row center:
`|det(affine)| × (N cos(latitude) π/180) × (M π/180)`, where `N` and `M` are the prime-vertical
and meridional radii of curvature. A fully covered cell can therefore have a different area from
whole-cell zonal weighting. For a .001° square centered at 0° or 80°, fractional divided by
whole-cell area is approximately 0.995530 or 1.008584, respectively. Both formulas are preserved.
The seven-word geographic layout evaluates thin strips relative to the nearer pole at each row,
including rows near the pole opposite the grid origin. Explicit layout 1 needs at least four words;
explicit layout 2 needs at least seven. Omitting the version requires exactly seven words, and
padding never selects a layout.

Caller-supplied `cellAreas` instead define the full-cell area directly, in caller-declared area units
(use square meters when ground area is wanted); the fractional contributors multiply that supplied area by coverage without deriving it from the grid. Covered
and valid areas retain those units, and density/intensive integrals use input units times area.

## Rasterization composition

To consume a possibly overflowing polygon rasterization on the same graph, pass its
`output.status.overflow` as `producerOverflow` to `GPURasterJoin` or
`GPURasterZonalStatistics`, and provide the new `output.status.incomplete` one-word uint32 view.
The guarded route reports 1 for a failed producer frame and suppresses every requested result
column: float columns become NaN and uint32 columns become `0xffffffff`. This includes counts-only
and per-point routes. Empty, complete input keeps ordinary empty-zone results and incomplete 0.
Both the results and incomplete flag reset on subsequent complete encodings. Existing callers
without `producerOverflow` keep their existing route; supplying the guard requires its new status
output so a suppressed per-point assignment cannot be mistaken for an ordinary unassigned point.

Polygon rasterization and join reject overlapping writable result ranges, including scalar status
words and distinct handles importing one physical buffer. Import each physical buffer once.
Aligned, disjoint outputs may share that handle when their dispatch bindings permit it; a join's
point-zone input to aggregation cannot share a physical buffer with its writable count columns.

Historical route: [Lagrangian diagnostics](./lagrangian-diagnostics-mob06.md).
