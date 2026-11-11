
# Raster analysis

## Overview

`@luma-contributors/analysis/raster-analysis` provides nodata-aware raster analysis contributors that
compose through `GPUCommandGraph`: zonal statistics, stretches, isolines, distance and cost
surfaces, polygon rasterization, change detection, flow textures, fill nodata, tile reprojection,
fractional coverage and spectral indices. Storage, overviews, pixel operations, statistics, filters,
morphology, connected components and contours are owned by
[luma.gl GPU Raster](https://luma.gl/docs/api-reference/experimental/gpu-raster).

## When to use it

Use these contributors for analytical raster pipelines whose intermediate bands, masks, labels or
statistics should remain on the GPU. Image-only presentation effects that do not need analytical
values usually belong in a renderer or shader module instead.

## Quick start

```ts
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUParameterBuffer} from '@luma-contributors/analysis/analysis-core';
import {
  GPUDistanceField,
  getGPUDistanceFieldParameterValues
} from '@luma-contributors/analysis/raster-analysis';

const graph = new GPUCommandGraph(device, {id: 'raster-analysis'});
const parameters = new GPUParameterBuffer(device, {id: 'df-parameters', format: 'float32', length: 8});

// `grid` is a `GPURasterMetadata` (see `makeGPURasterMetadata`). `seedPositions`, `seedCount`,
// `seedMask`, and the output views are graph views created on `graph`.
graph.add(new GPUDistanceField({
  grid, parameters: parameters.importToGraph(graph),
  seedPositions, seedCount, seedMask,
  output: {distances, allocation, withinDistance}
}));
const compiled = await graph.compileAsync();

parameters.write(getGPUDistanceFieldParameterValues({maximumDistance: 500}, grid));
compiled.encode(device.commandEncoder, {parameters: undefined});
```

See [Running contributors](../gpu-spatial-analysis/concepts.md#running-contributors) for the compile, encode
and submit pattern every analysis entry shares.

The [operations reference](/docs/api/gpu-raster/operations) indexes the pages.

## Catalog

One figure per contributor, each drawn from a recorded GPU run on real data; click through to its card.

::lens-catalog{entry=gpu-raster}

## Limits and compatibility

- Raster analysis is experimental and WebGPU-only.
- Tile sources, decoded arrays, GPU residency and submission remain application-owned.
- Capacity-dependent labels, geometry and statistics report overflow or incomplete work.

## Related modules

- [luma.gl GPU Core](https://luma.gl/docs/api-reference/experimental/gpu-core) provides scheduling and generic aggregation and indexing.
- [luma.gl GPU Raster](https://luma.gl/docs/api-reference/experimental/gpu-raster) owns raster storage, overviews, statistics, filters and contours.
- [luma.gl GPU Project](https://luma.gl/docs/api-reference/experimental/gpu-project) handles coordinate projection.

See [Raster, terrain and geometry expansion](./expansion.md) for fractional zonal statistics,
continuous-surface preparation, reprojection, directed walking cost and bounded topology operations.

[Expansion computation lenses](./expansion-lenses.md) provide executable bounded synthetic
examples for the added contributors, including validity, convergence and capacity evidence.

- [Four-quadrant nodata filling](./fill-nodata.md)
- [Explicit tile reprojection](./tile-reprojection.md)
- [Patch metrics and sieving](./raster-patches.md)

Fractional coverage also feeds [weighted distributions](./fractional-zonal-distribution.md) and
[categorical composition](./fractional-categorical-statistics.md), keeping observed support and
missing area separate. [Focal band pairs](./focal-pairs.md) provide paired-valid covariance,
correlation and local regression; [calibrated spectral indices](./spectral-indices.md) provide
explicit NDWI conventions, NBR, EVI and SAVI.
