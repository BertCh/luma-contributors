
# Raster analysis

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

## Quick start

Use the [complete exact-distance-field walkthrough](../README.md#first-run-exact-distance-field)
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

The catalog below filters operations by raster input or output data type across families.
Cards provide API details and examples; recorded runs are available where linked by each operation.

[Browse the raster analysis computation catalog](/explore/?data=raster).

## Limits and compatibility

- Raster analysis is experimental and WebGPU-only.
- Tile sources, decoded arrays, GPU residency and submission remain application-owned.
- Capacity-dependent labels, geometry and statistics report overflow or incomplete work.
- Same-CRS reprojection composes the destination-to-source sample transform in float64 before
  packing and anchors evaluation near source support to preserve scale/translation cancellation.
  Manual buffers must use [reprojection parameter layout version 3](./expansion.md#gpurasterreprojection).
- Tiled reprojection accepts at most 16,382 source tiles under its 32,768-command-node budget;
  [the tile reference](./tile-reprojection.md#resources-and-reuse) explains the construction estimate.

## Wiring uint32 planes and compact status

Match codes at each producer and consumer; sharing a uint32 format does not make planes
interchangeable.

| Plane | Meaning and conversion |
| --- | --- |
| `GPUProfilePeaks.validity` and ordinary validity inputs | 0 is missing; every nonzero word is usable, subject to finite values. A Boolean bad/outlier producer (`0` good, nonzero bad) has the opposite polarity: convert `flag == 0` to validity 1. |
| `GPUProfilePeaks.output.peakMask` | 0 is absent, 1 is kept, 0xffffffff is undecided. Test `word == 1` for kept membership; a nonzero test incorrectly accepts undecided peaks. |
| `GPUDistanceField.seedMask` | 0 is background; a word `v > 0` makes a seed with ID `v - 1`. Plain membership 1 makes ID 0. The word 0xffffffff makes ID 0xfffffffe, so it cannot carry an undecided state. |
| `GPUDistanceField.output.allocation` | ID of the seed found by the selected algorithm, including valid ID 0; 0xffffffff (`GPU_DISTANCE_FIELD_NONE`) means it found no seed within the distance limit. Approximate jump flooding can miss a nearer seed. This is not a validity plane. |
| `GPUDistanceField.output.nearestCells` / `withinDistance` | Selected seed cell index or 0xffffffff; separately, membership is 1 when that selected seed is within the distance limit and 0 otherwise. In approximate jump-flood mode, 0 does not prove that no seed lies inside the limit. |

For example, peak mask `[0, 1, 0xffffffff]` converts to seed membership `[0, 1, 0]`, while
`candidateOverflow = 1` remains attached to the frame. Suppress the distance result until
suppression completes (`candidateOverflow == 0`, `converged == 1`, or
`iterationLimitReached == 0`), then submit the converted mask to `GPUDistanceField`. Do not
discard the undecided evidence or publish a distance result from the incomplete prefix.
The distance contributor has no candidate-status input, so that submission gate belongs to the
caller. A complete peak mask `[0, 1, 0]` supplies one seed with ID 0 at the middle cell.

For compact `GPUProfilePeaks` and `GPUStreamlines` outputs, `overflow` reports only capacity
loss; zero does not imply all candidates were decided. Existing overflow readers must request
`candidateOverflow` or a convergence/iteration-limit status too, including through the GPU and
CPU operation forms. A complete compact frame requires `overflow == 0` **and** either
`candidateOverflow == 0` or equivalent candidate convergence evidence (`converged == 1` or
`iterationLimitReached == 0`). Convergence does not replace the capacity check:
two separated kept peaks with one output slot can report `[count, overflow, candidateOverflow,
converged] = [1, 1, 0, 1]`. `requiredCount == count` alone proves only
that the decided results fit. For `[0,1,0]`, zero peak suppression rounds give
`[count, overflow, candidateOverflow] = [0,0,1]` even with ample compact capacity.

LIC likewise accumulates noise only at supported RK2 landings. A one-cell eastward field,
two noise pixels `[0,1]`, centre x=0.5 and step length 0.75 leave the first output at centre
noise 0: the forward landing x=1.25 has no velocity support. With two supported field cells,
the Hann-weighted first output is 1/3. Output and field placements keep their signed cell sizes.

Streamlines retain a terminal RK2 landing with field data even when its speed falls below
`minimumSpeed` (including zero); no following step starts there. LIC excludes the noise sample
at such a landing. The CPU streamline twin checks field support again at the published float32
seed/landing coordinates, so rounding cannot publish a point on an excluded field edge.

## Related modules

- [luma.gl GPU Core](https://luma.gl/docs/api-reference/experimental/gpu-core) provides scheduling and generic aggregation and indexing.
- [luma.gl GPU Raster](https://luma.gl/docs/api-reference/experimental/gpu-raster) owns raster storage, overviews, statistics, filters and contours.
- [luma.gl GPU Project](https://luma.gl/docs/api-reference/experimental/gpu-project) handles coordinate projection.

See [Raster, terrain and geometry expansion](./expansion.md) for fractional zonal statistics,
continuous-surface preparation, reprojection, directed walking cost and bounded topology operations.

[Raster operation recordings](./expansion-lenses.md) provide executable bounded synthetic
examples for the added contributors, including validity, convergence and capacity evidence.

- [Four-quadrant nodata filling](./fill-nodata.md)
- [Explicit tile reprojection](./tile-reprojection.md)
- [Patch metrics and sieving](./raster-patches.md)

Fractional coverage also feeds [weighted distributions](./fractional-zonal-distribution.md) and
[categorical composition](./fractional-categorical-statistics.md), keeping observed support and
missing area separate. [Focal band pairs](./focal-pairs.md) provide paired-valid covariance,
correlation and local regression; [calibrated spectral indices](./spectral-indices.md) provide
explicit NDWI conventions, NBR, EVI and SAVI.

Historical route: [Lagrangian diagnostics](./lagrangian-diagnostics-mob06.md).
