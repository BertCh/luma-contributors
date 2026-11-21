---
title: Geospatial analysis
description: Datum-grid transforms, point buffers and spatial-context helpers layered on luma.gl GPU Project.
---

# Geospatial analysis

`@luma-contributors/analysis/geospatial-analysis` adds geospatial contributors that luma.gl does not
provide. Projection, distance, point-in-polygon, nearest-feature, grid-index and spatial-query
kernels are owned by luma.gl: see [`@luma.gl/graph/geospatial`](https://luma.gl/docs/api-reference/experimental/geospatial)
and [`@luma.gl/graph/gpu-project`](https://luma.gl/docs/api-reference/experimental/gpu-project).

| Export | Use it for |
| --- | --- |
| `GPUDatumGridTransform` | Bilinear horizontal and vertical datum shifts from caller-decoded grids. |
| `GPUPointBuffer` | Inscribed circle vertices around points in geographic or planar coordinates. |
| `getGeographicViewportBounds` | Padded geographic viewport bounds from explicitly packed XY or XYZ batches. |

Contributors implement `GPUCommandNodeProducer`. `graph.add(contributor)` declares their work on a
caller-owned `GPUCommandGraph`; it does not compile the graph, submit commands, or read results
back.

Source implementations live in the `datum-grid-transform`, `point-buffer` and `viewport-bounds`
family folders; the package entry re-exports each family index.

## `GPUDatumGridTransform`

Shifts longitude, latitude and height with ordered regular grids. The first grid covering a
position wins; nodata in that grid fails the row rather than falling through. Each row gets a
status word from `GPU_DATUM_GRID_TRANSFORM_STATUS` (`success`, `invalid`, `noCoverage`, `nodata`,
`inverseExhausted`). Grid samples are decoded by the caller. Inverse transforms iterate a bounded
number of times until the residual `forward(result) - input` is within `tolerance` degrees
(default `5e-9`, about 0.56 mm; never less than four f32 spacings of the shift itself). The
inverse always takes at least one update, so a shift smaller than the tolerance is still applied.
Pass `positionsLow` and `output.positionsLow` to keep detail below the f32 spacing of a coordinate.

```ts
import {Buffer, type Device} from '@luma.gl/core';
import type {GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import {readPackedBuffer} from '@luma-contributors/analysis/analysis-core';
import {
  GPUDatumGridTransform,
  decodeGPUDatumGridTransformPositions
} from '@luma-contributors/analysis/geospatial-analysis';

declare const graph: GPUCommandGraph;
declare const device: Device;
declare const positions: GraphDataView<'float32x3'>; // longitude/latitude degrees, height in metres
declare const positionsLow: GraphDataView<'float32x3'>; // residuals aligned with positions
declare const samples: GraphDataView<'float32x3'>; // shifts in degrees/degrees/metres
declare const mask: GraphDataView<'uint32'>; // one nodata flag per grid sample
const outputUsage = Buffer.STORAGE | Buffer.COPY_SRC;
const shiftedPositionsBuffer = device.createBuffer({
  id: 'shifted', byteLength: Math.max(1, positions.length) * 12, usage: outputUsage
});
const shiftedPositionsLowBuffer = device.createBuffer({
  id: 'shifted-low', byteLength: Math.max(1, positions.length) * 12, usage: outputUsage
});
const statusBuffer = device.createBuffer({
  id: 'datum-status', byteLength: Math.max(1, positions.length) * 4, usage: outputUsage
});
const shiftedPositions = graph.createDataView(
  graph.importBuffer({id: 'shifted', byteLength: shiftedPositionsBuffer.byteLength, usage: outputUsage}, shiftedPositionsBuffer),
  {format: 'float32x3', length: positions.length}
);
const shiftedPositionsLow = graph.createDataView(
  graph.importBuffer({id: 'shifted-low', byteLength: shiftedPositionsLowBuffer.byteLength, usage: outputUsage}, shiftedPositionsLowBuffer),
  {format: 'float32x3', length: positions.length}
);
const statusWords = graph.createDataView(
  graph.importBuffer({id: 'datum-status', byteLength: statusBuffer.byteLength, usage: outputUsage}, statusBuffer),
  {format: 'uint32', length: positions.length}
);

graph.add(
  new GPUDatumGridTransform({
    positions, // GraphDataView<'float32x3'>: longitude, latitude, height
    positionsLow,
    samples, // GraphDataView<'float32x3'>: longitude, latitude, height shifts
    mask, // GraphDataView<'uint32'>: zero marks nodata
    grids: [{width: 3, height: 3, sampleOffset: 0, origin: [-1, -1], spacing: [1, 1]}],
    operation: 'combined',
    direction: 'forward',
    // Replace these IDs with the vertical references of the decoded grid.
    sourceVerticalReference: 'source-height-datum',
    targetVerticalReference: 'target-height-datum',
    heightUnitMeters: 1,
    output: {positions: shiftedPositions, positionsLow: shiftedPositionsLow, status: statusWords}
  })
);

// Call only after encoding and submitting this graph's work.
async function readShiftedOutput() {
  const [high, low, status] = await Promise.all([
    readPackedBuffer(shiftedPositionsBuffer, 0, 'float32', positions.length * 3),
    readPackedBuffer(shiftedPositionsLowBuffer, 0, 'float32', positions.length * 3),
    readPackedBuffer(statusBuffer, 0, 'uint32', positions.length)
  ]);
  return decodeGPUDatumGridTransformPositions({positions: high, positionsLow: low, status});
}
// After the reads and compiled-graph destruction, destroy all three caller-owned output buffers.
```

The output buffers above are caller-owned and declare `Buffer.COPY_SRC` for host readback;
`readPackedBuffer` uses the buffer's asynchronous read path. Storage usage alone does not permit
a staging copy. See the [shared first-run lifecycle](/docs/api/README#shared-first-run-lifecycle)
for compilation, submission and cleanup. Combined and vertical operations require explicit source
and target vertical-reference IDs; grid height shifts are metres and `heightUnitMeters` declares
the input/output height unit.

Grid and position longitudes are compared as plain numbers, so a grid spanning the antimeridian is
expressed on one continuous branch (for example 170 to 190) with positions on that branch.

### Read split positions without losing the residual

With both low ports supplied, read back high and low XYZ components and the row statuses together.
`decodeGPUDatumGridTransformPositions({positions, positionsLow, status})` returns
`{positions: Float64Array, status: Uint32Array}` in source row order. Longitude and latitude are
degrees; height retains the grid's height unit. Each chunk has three components per row and one
status per row; low chunks must match the high chunks exactly. Typed arrays, `Uint8Array` byte
windows and `ArrayBuffer` readbacks are accepted. Misaligned lengths or chunk counts reject.
Failed rows retain their values, normally NaNs: consult the unchanged status word before use.

```ts
import {
  decodeGPUDatumGridTransformPositions,
  GPU_DATUM_GRID_TRANSFORM_STATUS
} from '@luma-contributors/analysis/geospatial-analysis';

// Two aligned one-row readbacks: one success and one failed row with NaN coordinates.
const high = Float32Array.of(120, 30, 0);
const low = Float32Array.of(0.000001, 0, 0);
const decoded = decodeGPUDatumGridTransformPositions({
  positions: [high, Float32Array.of(NaN, NaN, NaN)],
  positionsLow: [low, Float32Array.of(0, 0, 0)],
  status: [Uint32Array.of(0), Uint32Array.of(GPU_DATUM_GRID_TRANSFORM_STATUS.noCoverage)]
});
// Complete result: six coordinates [about 120.000001, 30, 0, NaN, NaN, NaN], statuses [0, 2].
const oneChunkEquivalent = Float64Array.from(high, (value, index) => value + low[index]);
// decoded.positions[0] and oneChunkEquivalent[0] are about 120.000001.
const roundedAgain = high.map((value, index) => value + low[index]);
// roundedAgain[0] is 120: Float32Array.map stores its result back into float32.
```

## `GPUPointBuffer`

Writes `verticesPerRing` vertex offsets per point, with bearing `2 pi v / verticesPerRing`
clockwise from north. Geographic offsets are degrees of longitude and latitude, so a consumer holding
float64 centers adds them without losing the float32 precision of a small radius. The radius is a
per-frame parameter. In planar mode both radius and offsets use the input XY unit. In geographic
mode centers are longitude/latitude degrees, radius is metres, and offsets are degrees. The
spherical formula uses `GPU_POINT_BUFFER_EARTH_RADIUS_METERS = 6371008.8`.

```ts
import {createTransientView, type GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import type {Device} from '@luma.gl/core';
import {GPUParameterBuffer, type GPUSpatialContext} from '@luma-contributors/analysis/analysis-core';
import {
  GPUPointBuffer,
  getGPUPointBufferParameterValues
} from '@luma-contributors/analysis/geospatial-analysis';

declare const graph: GPUCommandGraph;
declare const device: Device;
declare const points: GraphDataView<'float32x2'>; // longitude/latitude degrees
const offsets = createTransientView(graph, 'buffer-offsets', 'float32x2', points.length * 32);
const radiusParameters = new GPUParameterBuffer(device, {
  id: 'buffer-radius', format: 'float32', length: 2,
  values: getGPUPointBufferParameterValues(250) // metres
});
const sphericalContext: GPUSpatialContext = {
  coordinateSpace: 'longitude-latitude', metric: 'great-circle', units: 'meters'
};
// For planar XY, use this context with radius and offsets in the native XY unit:
const planarContext: GPUSpatialContext = {
  coordinateSpace: 'planar', metric: 'native', units: 'native'
};
graph.add(
  new GPUPointBuffer({
    points, // GraphDataView<'float32x2'>
    parameters: radiusParameters.importToGraph(graph),
    offsets, // GraphDataView<'float32x2'>, verticesPerRing rows per point
    verticesPerRing: 32,
    spatialContext: sphericalContext
  })
);
// On later frames: radiusParameters.write(getGPUPointBufferParameterValues(newRadiusMeters)).
// Destroy the compiled graph before radiusParameters.destroy().
```

Supply `spatialContext` at construction. A planar context may declare `units: 'meters'` when XY
itself is in metres. A geographic context may omit `sphereRadius` or specify exactly 6371008.8
metres and uses the great-circle metric in metres.

Antimeridian wrapping is left to the consumer, which must wrap `center longitude + offset` into
[-180, 180]. Latitude offsets are clamped so that `center + offset` stays within [-90, 90].

{/* parameters: GPUPointBuffer generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

### `GPUPointBuffer` parameters

Import from `@luma-contributors/analysis/geospatial-analysis`: `GPUPointBuffer`.

- Packer: `getGPUPointBufferParameterValues(radius)`, writes `GPU_POINT_BUFFER_PARAMETER_LENGTH` (2) `float32` values.
  - `radius` (`number`)

{/* /parameters */}

## Geographic viewport bounds

`getGeographicViewportBounds(positionBatches, positionSize: 2 | 3)` returns
`[west, south, east, north] | null`, in degrees. Size 2 reads every XY pair; size 3 reads every
XYZ triple and ignores Z. Each batch length must be a multiple of the declared size. Empty
input returns `null`. XY must be finite and longitudes canonical in [-180, 180].

```ts
import {getGeographicViewportBounds} from '@luma-contributors/analysis/geospatial-analysis';

const bounds = getGeographicViewportBounds([Float32Array.of(10, 20, 11, 21, 12, 22)], 2);
// Geometry envelope: [10, 20, 12, 22]; returned viewport bounds: [9.9, 19.9, 12.1, 22.1].
const emptyBounds = getGeographicViewportBounds([], 2); // null
```

Viewport bounds include a 5% margin on each axis, at least a float32 relative epsilon, clipped
to longitude [-180, 180] and latitude [-90, 90]. They are not an unpadded geometry envelope.
Pass the position size explicitly for all batches. For continuous longitude branches or a compact
antimeridian fit, use `getGeographicViewportFit(batches, size)` and its center/span output.

## Related pages

- [GPU Spatial Analysis](/docs/api/gpu-spatial-analysis)
- [luma.gl GPU Project](https://luma.gl/docs/api-reference/experimental/gpu-project)
- [luma.gl geospatial kernels](https://luma.gl/docs/api-reference/experimental/geospatial)
