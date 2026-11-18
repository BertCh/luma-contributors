
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

Use the [complete GPU walkthrough](../operation-forms-and-cpu-twins.md#point-density-from-device-to-readback)
for allocation, graph imports, submission, typed readback and cleanup. For an exact distance field:

Import `Float32Array.of(1.5, 1.5)` as a one-row `float32x2` seed view and allocate
the output views below. This checked function adds an exact distance field for that seed at the
center of a north-up 3 × 3 grid with 1 m cells.

```ts
import type {Device} from '@luma.gl/core';
import {GPUCommandGraph, type GraphDataView} from '@luma.gl/graph';
import {GPUParameterBuffer} from '@luma-contributors/analysis/analysis-core';
import {
  GPUDistanceField, makeGPURasterMetadata, getGPUDistanceFieldParameterValues,
  GPU_DISTANCE_FIELD_PARAMETER_LENGTH
} from '@luma-contributors/analysis/raster-analysis';

export function addDistanceFieldExample(
  device: Device, graph: GPUCommandGraph, seedPositions: GraphDataView<'float32x2'>,
  distances: GraphDataView<'float32'>, allocation: GraphDataView<'uint32'>,
  withinDistance: GraphDataView<'uint32'>
) {
  const grid = makeGPURasterMetadata({width: 3, height: 3, cellSize: 1, origin: [0, 3]});
  const parameters = new GPUParameterBuffer(device, {
    id: 'distance-field-example-parameters',
    format: 'float32',
    length: GPU_DISTANCE_FIELD_PARAMETER_LENGTH,
    values: getGPUDistanceFieldParameterValues({maximumDistance: 5}, grid)
  });
  graph.add(new GPUDistanceField({
    grid, seedPositions, parameters: parameters.importToGraph(graph), mode: 'exact',
    output: {distances, allocation, withinDistance}
  }));
  return parameters;
}
```

Omitting `seedCount`, `seedIds` and `seedMask` makes the single position active with ID 0.
Keep the returned parameter buffer until the compiled graph has been destroyed.

| Output port | Format and rows | Expected readback |
| --- | --- | --- |
| `distances` | `float32`, width × height (9 here) | `[sqrt(2), 1, sqrt(2), 1, 0, 1, sqrt(2), 1, sqrt(2)]` within float32 tolerance |
| `allocation` | `uint32`, width × height | Nine zero IDs |
| `withinDistance` | `uint32`, width × height | Nine ones |

All output views have separate buffers. Compile, encode, submit and read them with the
walkthrough's sequence, then destroy the compiled graph, parameter buffer and caller-owned buffers.
No-seed or beyond-limit distances are `Infinity`; `withinDistance` then becomes 0.
The default exact mode is exact for this isotropic fixture; jump-flood mode is an approximate
preview. See the [distance-field reference](./operations-analysis.md#gpudistancefield) for limits.

The [operations reference](/docs/api/gpu-raster/operations) indexes the pages.

## Catalog

One figure per contributor, each drawn from a recorded GPU run on real data; click through to its card.

::lens-catalog{entry=gpu-raster}

## Limits and compatibility

- Raster analysis is experimental and WebGPU-only.
- Tile sources, decoded arrays, GPU residency and submission remain application-owned.
- Capacity-dependent labels, geometry and statistics report overflow or incomplete work.
- Same-CRS reprojection composes the destination-to-source sample transform in float64 before
  packing and anchors evaluation near source support to preserve scale/translation cancellation.
  Manual buffers must use [reprojection parameter layout version 3](./expansion.md#gpurasterreprojection).
- Tiled reprojection accepts at most 16,382 source tiles under its 32,768-command-node budget;
  [the tile reference](./tile-reprojection.md#resources-and-reuse) explains the construction estimate.

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
