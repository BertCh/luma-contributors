
# GPU Terrain


:::caution Experimental
`@luma-contributors/analysis/gpu-terrain` is an initial, experimental entry point. APIs may change between
releases without a deprecation period.
:::

## Overview

`@luma-contributors/analysis/gpu-terrain` provides analysis contributors for elevation tiles: Horn slope,
aspect and hillshade, marching-squares contours, viewshed visibility, hydrology (depression
filling, flat resolution, D8, D-infinity and multiple-flow-direction routing, flow accumulation,
streams, height above drainage, watersheds, stream order, hydrologic indices), batched line of sight
and cumulative viewsheds, point horizon profiles, horizon angles, solar position, shadow masks and
irradiance, relief shading, texture shading, archaeological relief visualizations (RVT), terrain-RGB
decoding, summits and critical points, curvature, geomorphons, ruggedness, topographic position,
Weiss landforms, and terrain-following wind fields. Each contributor is an algorithm that declares its
resources and command nodes into a caller-owned
[`GPUCommandGraph`](https://luma.gl/docs/api-reference/experimental/gpu-core/gpu-command-graph) through `getCommandNodes(graph)`
(`GPUCommandNodeProducer`). It never compiles, submits, encodes, or reads back.

The contributors compose public primitives from [GPU Raster](../gpu-raster/README.md) and
[GPU Core](https://luma.gl/docs/api-reference/experimental/gpu-core/concepts); new WGSL is written only where no primitive fits.

## When to use it

Use GPU Terrain when the elevation values themselves are the product: slope and aspect for
suitability models, contour or viewshed geometry for analysis, flow accumulation for hydrology, or
analytic shading textures that stay on the GPU. Per-frame observers, light directions, contour
levels, and cell-size settings are read from storage views, so interaction never recompiles.

Prefer the generic GPU Raster operations when the field is not an elevation model or you need a
different kernel:

- `GPUTerrainDerivatives` is Horn's method (two Sobel `GPURasterGradient` passes plus a fused shade
  kernel) with terrain cell-size modes, a z factor, and latitude handling. Use `GPURasterGradient`
  directly for a single signed derivative of any raster.
- `GPUTerrainContours` runs one `GPURasterContours` pipeline per elevation level and reports one
  overflow flag. Use `GPUIsolines` and `GPUIsobands` from GPU Raster when you need stitched
  polylines or filled bands, and `GPURasterContours` for a single level.
- The solar and shading contributors write analytic values and textures. If you only need lit
  terrain for presentation, the lighting of a terrain renderer is simpler.

## Quick start

Start with the [complete GPU walkthrough](../operation-forms-and-cpu-twins.md#point-density-from-device-to-readback)
for the device, buffer import, submission, typed readback and cleanup sequence. Terrain derivatives
add a raster grid, a raster-band input and a packed parameter view:

The elevation input is a nine-row `float32` view containing zeros. The following checked function
wraps it as a raster band and adds derivatives to the graph from the shared walkthrough.
Allocate `slope` and `hillshade` as separate nine-row `float32` views.

```ts
import type {Device} from '@luma.gl/core';
import {GPUCommandGraph, type GraphDataView} from '@luma.gl/graph';
import {GPUParameterBuffer} from '@luma-contributors/analysis/analysis-core';
import {makeGPURasterBand, makeGPURasterMetadata} from '@luma-contributors/analysis/raster-analysis';
import {
  GPUTerrainDerivatives, getGPUTerrainDerivativesParameterValues,
  GPU_TERRAIN_DERIVATIVES_PARAMETER_LENGTH
} from '@luma-contributors/analysis/gpu-terrain';

export function addTerrainExample(
  device: Device, graph: GPUCommandGraph, elevationView: GraphDataView<'float32'>,
  slope: GraphDataView<'float32'>, hillshade: GraphDataView<'float32'>
) {
  const grid = makeGPURasterMetadata({width: 3, height: 3, cellSize: 30, origin: [0, 90]});
  const elevation = makeGPURasterBand(elevationView, grid, {id: 'elevation'});
  const parameters = new GPUParameterBuffer(device, {
    id: 'terrain-example-parameters',
    format: 'float32',
    length: GPU_TERRAIN_DERIVATIVES_PARAMETER_LENGTH,
    values: getGPUTerrainDerivativesParameterValues({azimuthDegrees: 315}, grid)
  });
  graph.add(new GPUTerrainDerivatives({
    grid, elevation, parameters: parameters.importToGraph(graph), output: {slope, hillshade}
  }));
  return parameters;
}
```

This north-up planar grid has 30 m cells. With no validity view, all nine finite elevations are
usable. Keep the returned parameter buffer until the compiled graph has been destroyed.

| Output port | Format and rows | Meaning |
| --- | --- | --- |
| `slope` | `float32`, width × height (9 here) | Slope in degrees (the default); the center of this flat fixture is 0. |
| `hillshade` | `float32`, width × height (9 here) | Analytic illumination in [0, 1]. |

Use separate output buffers, follow the walkthrough through readback, and destroy the compiled
graph before the parameter buffer and imported buffers. The packer takes cell size and latitude
settings from `grid`; rewrite its values before another submission to change lighting.
See [derivative boundaries and units](./operators.md#gputerrainderivatives) before interpreting
edge cells or using geographic elevation grids.

## Concepts and operators

- [GPU Terrain concepts](./concepts.md): conventions, units and the GPU execution model.
- [GPU Terrain operators](./operators.md): every contributor, its inputs, outputs, parameters and how it
  computes.

## Catalog

One figure per contributor, each drawn from a recorded GPU run on real data; click through to its card.

::lens-catalog{entry=gpu-terrain}

## Limits and compatibility

- GPU Terrain is experimental and WebGPU-only.
- Inputs are planar `float32` elevation values; projection is done upstream (for example with
  [`GPUProjection`](https://luma.gl/docs/api-reference/experimental/gpu-project)).
- Each contributor's TSDoc lists its non-goals. The GPU Core maintainer roadmap
  (`dev-docs/roadmap.md`) tracks what is still open.

## Related modules

- [GPU Raster](../gpu-raster/README.md) supplies the raster operators that terrain contributors compose.
- [GPU Core](https://luma.gl/docs/api-reference/experimental/gpu-core/concepts) defines contributors, composition levels, and graph ownership.
- [GPU Network](../gpu-network/README.md) shares the conventions above for road and flow analysis.

[Expansion computation lenses](./expansion-lenses.md) record downstream flow distance and compact
stream reaches on a bounded synthetic drainage grid, including unresolved and overflow evidence.
