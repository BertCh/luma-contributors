
# GPU Terrain

## Check alignment before extracting bands

Retain the upstream `GPURaster({metadata, bands})` until alignment has been checked. A bare band
or extracted view does not carry enough grid metadata to detect a shifted acquisition.

```ts
import {GPURaster} from '@luma.gl/graph/gpu-raster';
import type {GraphDataView} from '@luma.gl/graph/gpu-core';
import {makeGPURasterBand, makeGPURasterMetadata} from '@luma-contributors/analysis/raster-analysis';

export function createAlignedRaster(redView: GraphDataView<'float32'>, second: GPURaster) {
  const metadata = makeGPURasterMetadata({
    width: 2, height: 2, origin: [100, 200], cellSize: [30, 30], rowDirection: 'south'
  });
  const first = new GPURaster({
    metadata, bands: [makeGPURasterBand(redView, metadata, {id: 'red'})]
  });
  first.assertCompatibleWith(second); // Reject before extracting any views.
  const red = first.getBand('red');
  const firstCenter = first.getPixelWorldPosition(0, 0); // [115, 185]
  return {first, red, firstCenter};
}
```

Here `second` comes from the same graph and a separately loaded acquisition. Identical grids
pass. A one-column shift to affine `[30,0,130,0,-30,200]` has first center `[145,185]` and
`assertCompatibleWith` throws. Check masks and every required band on the same aligned grid;
this check never resamples.

GDAL's `[GT0,GT1,GT2,GT3,GT4,GT5]` maps to library affine
`[GT1,GT2,GT0,GT4,GT5,GT3]`. In the axis-aligned factory, supply **cell-size magnitudes**
`[30,30]` and choose `rowDirection: 'south'` (default, negative row step) or `'north'`
(positive row step). Passing `[30,-30]` with the default reverses rows.

### CRS and cost units

A CRS declaration does not select a ground-distance model. For a grid declared
`coordinateReferenceSystem: {authority: 'EPSG:4326'}` with one-degree cells centered on the
equator, set `cellSizeMode: 'geographic'` explicitly on `GPUCostDistance` (or its CPU/operation
form). Unit friction and source cell 0 produce `[0,111319.4921875]` across two east-west cells;
omitting `cellSizeMode` uses `'uniform'` and produces `[0,1]` in coordinate units instead.
With a projected meter CRS, 10 m cells, `cellSizeMode: 'uniform'`, friction `[1,3,1]` and
source 0 produce `[0,20,40]`: each transition costs distance times the mean endpoint friction.
Friction is cost per selected distance unit; costs use that distance unit times friction.
See [cost-distance contracts](../gpu-raster/operations-analysis.md#gpucostdistance).



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

See [output oracles](./output-oracles.md) for independent numerical fixtures, optional texture
publication checks, and the meanings of terrain iteration and capacity diagnostics.

See [status and count evidence](./status-and-count-evidence.md) for complete-word, poisoned-buffer
and replay assertions with independently derived stopping, capacity and validity expectations.

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

## Choose a task

### Terrain: Start and choose

- [GPU Terrain concepts](./concepts.md)

- [GPU Terrain operators](./operators.md)

### Terrain: Route water and compare surfaces

- [Height above drainage over D-infinity flow](./height-above-drainage.md)

- [Terrain processing](./terrain-processing.md)

- [Terrain hydrology recordings](./expansion-lenses.md)

### Terrain: Inspect recorded outputs and status

- [Terrain output oracles](./output-oracles.md)

- [Terrain status and count evidence](./status-and-count-evidence.md)

## Quick start

Start with the [complete GPU walkthrough](../README.md#first-run-terrain-derivatives)
for the device, buffer import, submission, typed readback and cleanup sequence. Terrain derivatives
add a raster grid, a raster-band input and a packed parameter view:

The elevation input is a nine-row `float32` view containing
`Float32Array.of(0,30,60, 0,30,60, 0,30,60)`: a plane rising 30 m per 30 m east. The following checked function
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
    values: getGPUTerrainDerivativesParameterValues({azimuthDegrees: 270, altitudeDegrees: 45}, grid)
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
| `slope` | `float32`, width × height (9 here) | Slope in degrees (the default); the center is 45° (atan(30/30)). |
| `hillshade` | `float32`, width × height (9 here) | Analytic illumination in [0, 1]; the center is approximately 1 with this west-facing slope and light from 270° at altitude 45°. |

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

The catalog below filters operations by terrain input or output data type across families.
Cards provide API details and examples; recorded runs are available where linked by each operation.

[Browse the terrain computation catalog](/explore/?data=terrain).

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

[Terrain hydrology recordings](./expansion-lenses.md) record downstream flow distance and compact
stream reaches on a bounded synthetic drainage grid, including unresolved and overflow evidence.
